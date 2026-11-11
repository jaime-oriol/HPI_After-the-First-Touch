"""run_plan_race — plan, flight or race: how the second balls of tracked corners are won.

The registered analysis (data/derived/prereg_2026-10-31/plan_flight_race.md and its deviation of the same day) on the
thirty tracked Serie A matches. Every corner whose second ball the event chain
resolves (`s08_situation.DropSituation._clear_and_next`) is read at three instants, the kick, the clearance and the next
touch, each moved onto the tracking clock of its match and period (`tracking_clock`), and the drop is where the next
touch met the ball. The side that won the second ball already had the player nearest the drop at the kick (plan), got
nearest by the clearance (flight), or neither (race). The groups are counted after the fact, with the drop known, so
they describe how the ball was won and predict nothing.

Beside them: the share that no relation between being nearest at the kick and winning would give; the onset of the
players within NEAR_M of the drop towards it, on every player and on those detected in every frame the derivative
needs; the sheet's sensitivities (each anchor's clock alone, the four nearest detected, keepers out); and two
diagnostics of the alignment, the ball against the corner flag at the kick, read on the corners struck from the flag (the
cross that delivers a short corner is struck from the wing), and the winning side against the drop at the next touch.

Writes to the output folder: plan_race.parquet, the record report_figures binds (block, key, n, point, lo, hi);
plan_race_corners.parquet, one row per second ball; plan_race_onsets.parquet, one row per player near a drop. One core
and a few minutes, since each tracking file is streamed once.

    python pipeline/run_plan_race.py
    python pipeline/run_plan_race.py --limit 3 --boot 200 --out-dir data/derived/smoke
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO / "src"))

import loader                                                           # noqa: E402
import loader_sb as L                                                   # noqa: E402
import tracking_clock as tc                                             # noqa: E402
from constants import (FPS_TRACKING, MAX_PLAYER_SPEED_MS, MIN_CELL, PITCH_LENGTH_M,   # noqa: E402
                       PITCH_WIDTH_M, PITCH_X_SB, PITCH_Y_SB, REACTION_SPEED_THR, SAVGOL_POLYORDER,
                       SAVGOL_WINDOW_FRAMES)
from kinematics import causal_savgol                                    # noqa: E402
from s08_situation import DropSituation                                 # noqa: E402
from validate import cluster_bootstrap_stats                            # noqa: E402

_DER = _REPO / "data" / "derived"
# The sheet's own magnitudes (plan_flight_race.md): the players counted near the drop, the onset window around the
# clearance, and how long the movement towards the drop has to last to count as an onset.
NEAR_M, PRE_S, POST_S, HOLD_FRAMES = 15.0, 1.0, 2.0, 5
CASES, SIDES, CLOCKS = ("plan", "flight", "race"), ("att", "def"), ("mid", "pass", "receipt")
_FLAGS = np.array([(sx * PITCH_LENGTH_M / 2, sy * PITCH_WIDTH_M / 2) for sx in (-1, 1) for sy in (-1, 1)])


# ── one match ────────────────────────────────────────────────────────────
def second_balls(ev: pd.DataFrame, three_sixty: pd.DataFrame) -> pd.DataFrame:
    """One row per corner whose second ball s08 resolves: the event times of the kick, the clearance and the next touch,
    the next touch's location and team, whether the attack won it, and, for the figure, the drop in the attackers' frame
    (StatsBomb yards, goal at x = 120) and the y of the kick."""
    ds = DropSituation(ev, three_sixty)
    rows = []
    for _, c in ds.g.deliveries().iterrows():
        clear, nxt, att = ds._clear_and_next(c)
        if clear is not None and nxt is not None:
            loc = nxt.get("location")
            own = nxt["team.name"] == att
            drop = ((loc[0], loc[1]) if own else (PITCH_X_SB - loc[0], PITCH_Y_SB - loc[1])) if isinstance(loc, list) \
                else (np.nan, np.nan)
            rows.append({"corner_id": c["id"], "anchor": c["anchor"], "period": int(c["period"]), "att": att,
                         "t_kick": float(c["_t"]),
                         "t_clear": float(clear["_t"]), "t_next": float(nxt["_t"]), "next_team": nxt["team.name"],
                         "next_loc": loc, "won": own, "drop_x": float(drop[0]), "drop_y": float(drop[1]),
                         "kick_y": float(c["location"][1])})
    return pd.DataFrame(rows)


def keepers(ev: pd.DataFrame, home_id: int) -> set:
    """{(is_home, jersey)} of the two starting goalkeepers."""
    return {(int(r["team.id"]) == home_id, str(p["jersey_number"]))
            for _, r in ev[ev["type.name"] == "Starting XI"].iterrows()
            for p in r["tactics.lineup"] if p["position.name"] == "Goalkeeper"}


def frame_at(frames: list, t: float) -> tuple | None:
    """The captured frame nearest t, if one lies within a frame and a half of it."""
    k = int(np.argmin([abs(f[0] - t) for f in frames]))
    return frames[k] if abs(frames[k][0] - t) <= 1.5 / FPS_TRACKING else None


def nearest(frame: tuple, drop: np.ndarray, home_is_att: bool, skip: frozenset | set = frozenset()) -> tuple[str, bool]:
    """(side of the player nearest the drop, whether the four nearest are all detected) in one frame."""
    rows = [r for r in frame[2] if (r[0], r[1]) not in skip]
    order = np.argsort([np.hypot(r[3] - drop[0], r[4] - drop[1]) for r in rows])
    return ("att" if rows[order[0]][0] == home_is_att else "def"), all(rows[i][2] for i in order[:4])


def group(win: str, at_kick: str, at_clear: str) -> str:
    """plan when the winning side was nearest at the kick, flight when it was only by the clearance, race otherwise."""
    return "plan" if at_kick == win else "flight" if at_clear == win else "race"


def onsets(frames: list, t_clear: float, drop: np.ndarray, home_is_att: bool) -> list:
    """Each player within NEAR_M of the drop at the clearance: his side, his onset towards the drop in seconds from the
    clearance (censored at POST_S), whether it is the window's first frame (already moving when it opens) and whether he
    is detected in every frame of the window and of the derivative's own window before it.

    Velocities are the causal Savitzky-Golay of `kinematics`, on detected frames only, so an estimated frame leaves the
    velocity undefined and a step across a frame gap too; the onset is the first frame from which the component towards
    the drop stays above REACTION_SPEED_THR for HOLD_FRAMES frames."""
    ts, fn = np.array([f[0] for f in frames]), np.array([f[1] for f in frames])
    k0 = int(np.argmin(np.abs(ts - t_clear)))
    span = (ts >= t_clear - PRE_S) & (ts <= t_clear + POST_S)
    i0 = int(np.flatnonzero(span)[0])
    need = span.copy()
    need[max(0, i0 - SAVGOL_WINDOW_FRAMES):i0] = True
    step = np.diff(fn, prepend=fn[0] - 1) == 1
    out = []
    for r in frames[k0][2]:
        if np.hypot(r[3] - drop[0], r[4] - drop[1]) > NEAR_M:
            continue
        x, y = np.full(len(frames), np.nan), np.full(len(frames), np.nan)
        for k, f in enumerate(frames):
            for q in f[2]:
                if (q[0], q[1]) == (r[0], r[1]) and q[2]:
                    x[k], y[k] = q[3], q[4]
        vx = np.where(step, np.diff(x, prepend=np.nan), np.nan) * FPS_TRACKING
        vy = np.where(step, np.diff(y, prepend=np.nan), np.nan) * FPS_TRACKING
        fast = np.hypot(vx, vy) > MAX_PLAYER_SPEED_MS
        vx[fast], vy[fast] = np.nan, np.nan
        vx, vy = (causal_savgol(v, SAVGOL_WINDOW_FRAMES, SAVGOL_POLYORDER) for v in (vx, vy))
        ux, uy = drop[0] - x, drop[1] - y
        moving = np.nan_to_num((vx * ux + vy * uy) / np.hypot(ux, uy), nan=-np.inf) > REACTION_SPEED_THR
        held = np.array([k + HOLD_FRAMES <= len(moving) and moving[k:k + HOLD_FRAMES].all() for k in range(len(moving))])
        hit = np.flatnonzero(held & span)
        out.append({"side": "att" if r[0] == home_is_att else "def", "jersey": r[1],
                    "onset_s": float(ts[hit[0]] - t_clear) if len(hit) else POST_S, "censored": not len(hit),
                    "already_moving": bool(len(hit)) and hit[0] == i0,
                    "fully_detected": i0 >= SAVGOL_WINDOW_FRAMES and bool(np.isfinite(x[need]).all())})
    return out


def match_rows(mid: int, path: Path) -> tuple[list, list, list]:
    """(corner rows, onset rows, clock rows) of one tracked match."""
    ev = L.load_events(mid)
    sb = second_balls(ev, L.load_360(mid))
    if sb.empty:
        return [], [], []
    home_id, home_name = int(ev["home_team.home_team_id"].iloc[0]), ev["home_team.home_team_name"].iloc[0]
    gks = keepers(ev, home_id)
    # every instant lies within the offset grid of its event time; the onset window and its derivative reach further back
    lo, hi = float(tc.SHIFTS.min()), float(tc.SHIFTS.max())
    spans = {r.corner_id: (r.period, min(r.t_kick, r.t_clear - PRE_S - 0.4) + lo - 0.1, max(r.t_clear + POST_S, r.t_next) + hi + 0.2)
             for r in sb.itertuples()}
    ball, keeper, frames = tc.stream(path, next(j for h, j in gks if h), spans)
    right = tc.home_attacks_right(keeper)
    off = tc.clock_offsets(ev, ball, right, home_id)
    clock = [{"match_id": mid, "period": p, **d} for p, d in off.items()]
    corners, players = [], []
    for r in sb.itertuples():
        row = {"match_id": mid, "corner_id": r.corner_id, "anchor": r.anchor, "period": r.period, "won": r.won,
               "drop_x": r.drop_x, "drop_y": r.drop_y, "kick_y": r.kick_y}
        fr = sorted(frames.get(r.corner_id, []), key=lambda f: f[0])
        if r.period not in off or not isinstance(r.next_loc, list) or not fr:
            corners.append({**row, "status": "no clock, drop or frames"})
            continue
        home_is_att, win = r.att == home_name, ("att" if r.won else "def")
        drop = tc.to_tracking(r.next_loc, r.next_team == home_name, right[r.period])
        for clk in CLOCKS:
            d = off[r.period][clk]
            fk, fc, fx = (frame_at(fr, t + d) for t in (r.t_kick, r.t_clear, r.t_next))
            if fk is None or fc is None or fx is None:
                row[f"group_{clk}"] = None
                continue
            (k_side, k_seen), (c_side, c_seen) = nearest(fk, drop, home_is_att), nearest(fc, drop, home_is_att)
            row[f"group_{clk}"] = group(win, k_side, c_side)
            if clk != "mid":
                continue
            t, bx, by, _ = ball[r.period]
            i = int(np.argmin(np.abs(t - (r.t_kick + d))))
            row.update(offset_s=d, near_kick=k_side, near_clear=c_side, four_detected=k_seen and c_seen,
                       group_keepers_out=group(win, nearest(fk, drop, home_is_att, gks)[0],
                                               nearest(fc, drop, home_is_att, gks)[0]),
                       ball_to_flag_m=float(np.min(np.hypot(bx[i] - _FLAGS[:, 0], by[i] - _FLAGS[:, 1]))),
                       winner_to_drop_m=min(np.hypot(q[3] - drop[0], q[4] - drop[1]) for q in fx[2]
                                            if (q[0] == home_is_att) == (win == "att")))
            players += [{"match_id": mid, "corner_id": r.corner_id, **o}
                        for o in onsets(fr, r.t_clear + d, drop, home_is_att)]
        corners.append({**row, "status": "ok" if row.get("group_mid") else "no frame at an instant"})
    return corners, players, clock


# ── the record ───────────────────────────────────────────────────────────
def _boot(frame: pd.DataFrame, stats: dict, boot: int) -> pd.DataFrame:
    return cluster_bootstrap_stats(frame, stats, n_boot=boot, seed=0)


def _rows(block: str, r: pd.DataFrame, n: dict | int) -> list:
    return [{"block": block, "key": k, "n": n[k] if isinstance(n, dict) else n, "point": float(r.loc[k, "point"]),
             "lo": float(r.loc[k, "lo"]), "hi": float(r.loc[k, "hi"])} for k in r.index]


def shares(t: pd.DataFrame, col: str, block: str, boot: int) -> list:
    """The three groups as shares of the second balls in `t`, with match-clustered intervals; a frame under the
    publication floor gives its counts and no share."""
    sub = t[t[col].notna()]
    counts = {g: int((sub[col] == g).sum()) for g in CASES}
    if len(sub) < MIN_CELL:
        return [{"block": block, "key": g, "n": counts[g]} for g in CASES]
    return _rows(block, _boot(sub, {g: (lambda s, g=g: float((s[col] == g).mean())) for g in CASES}, boot), counts)


def no_relation(t: pd.DataFrame, boot: int) -> list:
    """The plan share against the one it would take if the side nearest at the kick had no relation to the side that
    wins: the sum over the two sides of the share each wins times the share each is nearest."""
    def parts(s: pd.DataFrame) -> tuple[float, float]:
        w, n = s["won"].mean(), (s["near_kick"] == "att").mean()
        return float((s["group_mid"] == "plan").mean()), float(w * n + (1 - w) * (1 - n))
    r = _boot(t, {"observed": lambda s: parts(s)[0], "expected": lambda s: parts(s)[1],
                  "diff": lambda s: parts(s)[0] - parts(s)[1]}, boot)
    return _rows("no_relation", r, len(t))


def onset_rows(p: pd.DataFrame, boot: int) -> list:
    """The onset difference, defenders minus attackers, per corner on the medians of each side and averaged over the
    corners that hold both: on every player near the drop (the sheet) and on the fully detected ones (its deviation),
    with the shares already moving and censored of the fully detected, by side."""
    rows = []
    for key, sub in (("all", p), ("detected", p[p["fully_detected"]])):
        per = sub.groupby(["match_id", "corner_id", "side"])["onset_s"].median().unstack("side").dropna()
        per = per.assign(diff=lambda d: d["def"] - d["att"]).reset_index()
        rows += _rows("onset_diff", _boot(per, {key: lambda s: float(s["diff"].mean())}, boot), len(per))
    full = p[p["fully_detected"]]
    for flag in ("already_moving", "censored"):
        r = _boot(full, {s: (lambda d, s=s, f=flag: float(d.loc[d["side"] == s, f].mean())) for s in SIDES}, boot)
        rows += _rows(flag, r, {s: int((full["side"] == s).sum()) for s in SIDES})
    rows += [{"block": "players", "key": s, "n": int((p["side"] == s).sum()),
              "point": float(p.loc[p["side"] == s, "fully_detected"].mean())} for s in SIDES]
    return rows


def flag_diagnostic(ok: pd.DataFrame) -> dict:
    """The ball against the corner flag at the kick, in the record's row, on the corners struck from the flag: the cross
    that delivers a short corner is struck from the wing, where the ball is nowhere near a flag, and says nothing about
    whether the aligned clock puts the ball where the corner was taken."""
    at_flag = ok[ok["anchor"] == "kick"]
    return {"block": "diagnostic", "key": "ball_to_flag_m", "n": len(at_flag),
            "point": float(at_flag["ball_to_flag_m"].median())}


def record(t: pd.DataFrame, p: pd.DataFrame, clock: pd.DataFrame, boot: int) -> pd.DataFrame:
    ok = t[t["status"] == "ok"]
    rows = [{"block": "population", "key": "second_balls", "n": len(t)},
            {"block": "population", "key": "analysed", "n": len(ok)},
            {"block": "population", "key": "matches", "n": int(t["match_id"].nunique())},
            {"block": "clock", "key": "halves", "n": len(clock), "point": float((clock["mid"] < 0).mean())},
            {"block": "clock", "key": "matches_behind", "n": int(clock["match_id"].nunique()),
             "point": float(clock.groupby("match_id")["mid"].max().lt(0).mean())},
            {"block": "clock", "key": "median_lag_s", "n": len(clock), "point": float(-clock["mid"].median())},
            flag_diagnostic(ok),
            {"block": "diagnostic", "key": "winner_to_drop_m", "n": len(ok),
             "point": float(ok["winner_to_drop_m"].median())}]
    rows += shares(ok, "group_mid", "share", boot)
    for won, side in ((True, "attack"), (False, "defence")):
        rows += shares(ok[ok["won"] == won], "group_mid", f"share_{side}", boot)
    rows += no_relation(ok, boot)
    for clk in ("pass", "receipt"):
        rows += shares(ok, f"group_{clk}", f"clock_{clk}", boot)
    rows += shares(ok[ok["four_detected"].astype(bool)], "group_mid", "four_detected", boot)
    rows += shares(ok, "group_keepers_out", "keepers_out", boot)
    rows += onset_rows(p[p["corner_id"].isin(ok["corner_id"])], boot)
    return pd.DataFrame(rows)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", type=Path, default=_DER)
    ap.add_argument("--limit", type=int, default=None, help="first N tracked matches only (smoke)")
    ap.add_argument("--boot", type=int, default=2000, help="match-clustered bootstrap draws")
    args = ap.parse_args()
    if args.out_dir.resolve() == _DER.resolve() and (args.limit or args.boot != 2000):
        raise SystemExit("STOP: the published plan_race tables come only from every match at 2000 draws; pass --out-dir")
    os.environ.setdefault("HUDL_NO_CACHE", "1")
    loader._build_pairing()
    corners, players, clock = [], [], []
    for mid, path in sorted(loader._PAIRING.items())[:args.limit]:
        c, p, k = match_rows(mid, path)
        corners, players, clock = corners + c, players + p, clock + k
        print(f"{path.name}: {len(c)} second balls", flush=True)
    t, p, k = pd.DataFrame(corners), pd.DataFrame(players), pd.DataFrame(clock)
    rec = record(t, p, k, args.boot)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    t.to_parquet(args.out_dir / "plan_race_corners.parquet", index=False)
    p.to_parquet(args.out_dir / "plan_race_onsets.parquet", index=False)
    rec.to_parquet(args.out_dir / "plan_race.parquet", index=False)
    pd.set_option("display.width", 200)
    print("\n" + rec.round(4).to_string(index=False))
    print(f"\nwrote plan_race, plan_race_corners and plan_race_onsets to {args.out_dir}")


if __name__ == "__main__":
    main()
