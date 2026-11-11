"""loader — Hudl Broadcast X&Y tracking access (Serie A 2025/26, 30-match sample).

Long format (one row per player-or-ball per frame), the shape src/s04_race.py and
src/pitch_control.py expect. The raw JSONL only gives a jersey number and
a home/away flag per player, so team_id/player_id/is_goalkeeper are resolved by
cross-referencing the paired StatsBomb events file (loader_sb, same 30 matches):
team_id from home_team.home_team_id/away_team.away_team_id, is_goalkeeper from the
Starting XI's tactics.lineup (jersey_number -> position.name), player_id as
f"{team_id}_{jerseyNum}" (unique within one match, no global player id needed).

The two vendor feeds use different match-id spaces (Hudl gameRefId here vs the
StatsBomb match_id everywhere else in the pipeline) and different team-name
conventions for the same club (Hudl "Roma"/"Inter"/"Milan" vs StatsBomb "AS
Roma"/"Inter Milan"/"AC Milan") - matches are paired by (home, away, date), not id.
"""

from __future__ import annotations

import json
from array import array
from pathlib import Path

import numpy as np
import pandas as pd

import loader_sb as L
from constants import PITCH_LENGTH_M, PITCH_OPTA, PITCH_WIDTH_M

DATA_DIR = Path(__file__).resolve().parents[1] / "data" / "serie_a_sample" / "tracking"

_TEAM_ALIASES = {"Roma": "AS Roma", "Inter": "Inter Milan", "Milan": "AC Milan"}


def _canon(team: str) -> str:
    """Tracking-side team name -> the StatsBomb-side name used by loader_sb."""
    return _TEAM_ALIASES.get(team, team)


def meters_to_opta(x_m: "float | np.ndarray | pd.Series", y_m: "float | np.ndarray | pd.Series"
                   ) -> "tuple[float | np.ndarray | pd.Series, float | np.ndarray | pd.Series]":
    """Pitch-centred metres -> the 0-100 render scale, on both axes.

    Public because the renders need it for points that never pass through a tracking frame,
    such as a reach-field grid or the loose-ball point. It is the one place the pitch is spelled
    out, so a render never restates 105.0, 68.0, 52.5 and 34.0 as bare literals."""
    return ((x_m + PITCH_LENGTH_M / 2) / PITCH_LENGTH_M * PITCH_OPTA,
            (y_m + PITCH_WIDTH_M / 2) / PITCH_WIDTH_M * PITCH_OPTA)


def _parse_tracking_filename(path: Path) -> tuple[str, str, str]:
    """`<gameRefId>_<home>_vs_<away>_<date>.jsonl` -> (home, away, date), spaces restored."""
    _, rest = path.stem.split("_", 1)
    home_raw, rest = rest.split("_vs_", 1)
    away_raw, date = rest.rsplit("_", 1)
    return home_raw.replace("_", " "), away_raw.replace("_", " "), date


_PAIRING: dict[int, Path] = {}                              # StatsBomb match_id -> tracking file, built once


def _build_pairing() -> None:
    """Pair every tracking file to its StatsBomb match_id via (home, away, date).

    Fails loudly on any unpaired file: the key includes the date, so a fixture whose
    two feeds disagree on it would otherwise vanish from the sample in silence."""
    if _PAIRING:
        return
    cat = L.list_matches(L.SERIE_A_SAMPLE)
    by_key = {(r.home_team, r.away_team, r.match_date): int(r.match_id) for r in cat.itertuples()}
    unpaired = []
    for p in DATA_DIR.glob("*.jsonl"):
        home, away, date = _parse_tracking_filename(p)
        key = (_canon(home), _canon(away), date)
        if key in by_key:
            _PAIRING[by_key[key]] = p
        else:
            unpaired.append(p.name)
    if unpaired:
        raise RuntimeError(f"tracking files with no events match: {sorted(unpaired)}")


def _team_ids_and_gk(match_id: int) -> tuple[int, int, dict[str, bool]]:
    """(home_team_id, away_team_id, {"<team_id>_<jersey>": is_goalkeeper}) from the paired events file."""
    events = L.load_events(match_id)
    home_id = int(events["home_team.home_team_id"].iloc[0])
    away_id = int(events["away_team.away_team_id"].iloc[0])

    gk = {}
    xi = events[events["type.name"] == "Starting XI"]
    for _, row in xi.iterrows():
        team_id = int(row["team.id"])
        for player in row["tactics.lineup"]:
            key = f"{team_id}_{player['jersey_number']}"
            gk[key] = player["position.name"] == "Goalkeeper"
    return home_id, away_id, gk


def _keeper_x(fr: dict, home_id: int, gk: dict[str, bool]) -> list:
    """The home keeper's x values in ONE frame. Empty when he is not detected in it."""
    return [pl["x"] for pl in fr.get("homePlayers", [])
            if pl and pl.get("x") is not None and gk.get(f"{home_id}_{pl.get('jerseyNum')}", False)]


def _directions_from_keeper_x(xs: dict[int, list]) -> dict[int, str]:
    """Per-period {"Left to Right", "Right to Left"} for the home team, from where the
    home GOALKEEPER stands: he plays beside the goal he defends, so the sign of his
    mean x over the period says which goal that is. Keyed by every period SEEN, so a
    period with no keeper detection raises rather than going missing.

    NOT the kickoff shape. The broadcast clock starts before kick-off, so a period's
    first frames are not the kickoff at all: on this sample no period-1 window holds a home
    outfielder (the first frame serialises `homePlayers: [{}]`), which answers
    "Right to Left" on no evidence, and the period-2 windows that do have players land
    mid-restart. That rule reads a large share of the (match, period) slots wrong
    (evidence.md §15). The keeper is unambiguous by comparison: he sits far from halfway
    in every slot and on the opposite side from the other keeper (test_loader re-measures
    both over the tracked sample).

    Raises instead of guessing when a period has no keeper detection: "unknown" and
    "right to left" are different answers and this returns no third state."""
    directions = {}
    for p in sorted(xs):
        if not xs[p]:
            raise ValueError(f"period {p}: no home goalkeeper detected in any frame, so "
                             f"the direction of play is unknown rather than right-to-left")
        directions[p] = "Left to Right" if np.mean(xs[p]) < 0 else "Right to Left"
    return directions


def attack_runs_rtl(frame: pd.DataFrame, attacking_team: int, home_team: int) -> bool:
    """Does the ATTACKING team run right-to-left in this frame (i.e. must a render flip)?

    `direction_of_play` carries the HOME team's direction on EVERY row, including the
    away and ball rows, so filtering by the attacker's rows and reading it directly
    orients every away-team set piece backwards. The away side runs the other way by
    definition."""
    if frame.empty:
        return False
    home_dir = frame["direction_of_play"].mode().iloc[0]
    return home_dir == ("Right to Left" if attacking_team == home_team else "Left to Right")


def tracked_matches() -> list[int]:
    """StatsBomb match_ids of every match with a paired tracking file, sorted.

    The multi-match entry point: run_race.py --all iterates exactly this list, so
    the raced sample and the paired sample cannot drift apart."""
    _build_pairing()
    return sorted(_PAIRING)


def load_tracking(match_id: int, float32: bool = True) -> pd.DataFrame:
    """One match's Broadcast X&Y tracking, long format (one row per player-or-ball per frame).

    Two clocks come out, and they are not interchangeable. `period_elapsed_s` is the
    vendor's game clock, the closer of the two to the StatsBomb timestamps, and still
    behind them in every half of the sample, by about a second: an event-to-frame join
    first moves the event's instant by the offset `tracking_clock.clock_offsets`
    measures for its half, as run_plan_race does and s04_race._corner_frame does not.
    `timeelapsed_from_period_start_s` counts from the period's first frame on the
    VIDEO clock, which starts before kickoff by a per-broadcast margin: it runs ahead
    of the game clock through period 1 (constant within the period) and levels out in
    period 2.

    Measured on the kick instant over a sample of corners: in period 2 the two clocks
    coincide near the true kick moment; in period 1 the game clock stays close to that
    same reference while the video clock overshoots further ahead, often landing its
    anchored frame before the ball is even struck. So in period 1 the ball is detected at
    the anchored frame far more reliably on the game clock than on the video clock.

    COST, because it bounds any fan-out over matches: one match is 3.29M rows and a
    1.06 GB frame, and peaks at 1.83 GB RSS (1.93 GB on the largest of the 30). The
    returned frame is the floor of that peak and only 0.77 GB of it is transient. Size
    worker counts off the peak, not off core count.

    The file is streamed into C-typed buffers to keep it there. Only the direction needs
    the whole file: the keeper's x accumulates per period on the way past, and the
    direction is mapped onto the finished period column in one vectorised pass at the end.

    Args:
        match_id: StatsBomb match id (the same id loader_sb.load_events uses for this match).
        float32: Downcast float64 columns to float32 to halve memory (tracking is big).
    """
    _build_pairing()
    home_id, away_id, gk = _team_ids_and_gk(match_id)

    fc, per, ib = array("q"), array("q"), array("q")
    t_b, g_b, tid, xb, yb, vxb, vyb, spb = (array("d") for _ in range(8))
    gkb = array("b")
    pids: list = []
    vis: list = []
    conf: list = []

    keeper_xs: dict[int, list] = {}
    period_t0: dict[int, float] = {}
    # interned: ~60 distinct player_ids a match, so an object column costs a pointer, not a str
    pid_cache: dict = {}
    str_cache: dict = {}
    nan = float("nan")

    def _f(v: "float | None") -> float:
        """None -> NaN, the way a float column holds it anyway."""
        return nan if v is None else v

    with open(_PAIRING[match_id]) as f:
        # first line is match metadata, not a frame
        f.readline()
        for line in f:
            fr = json.loads(line)
            period = fr["period"]
            if period not in period_t0:
                period_t0[period] = fr["videoTimeMs"]
                keeper_xs[period] = []
            keeper_xs[period] += _keeper_x(fr, home_id, gk)

            frame_count = fr["frameNum"]
            t_s = (fr["videoTimeMs"] - period_t0[period]) / 1000.0
            game_s = fr["periodElapsedTime"]

            for side, team_id in (("homePlayers", home_id), ("awayPlayers", away_id)):
                for pl in fr.get(side, []):
                    # an undetected slot serializes as an empty {}
                    if not pl:
                        continue
                    key = (team_id, pl.get("jerseyNum"))
                    player_id = pid_cache.get(key)
                    if player_id is None:
                        player_id = pid_cache[key] = f"{team_id}_{key[1]}"
                    fc.append(frame_count)
                    per.append(period)
                    ib.append(0)
                    t_b.append(t_s)
                    g_b.append(_f(game_s))
                    tid.append(team_id)
                    pids.append(player_id)
                    gkb.append(gk.get(player_id, False))
                    xb.append(_f(pl.get("x")))
                    yb.append(_f(pl.get("y")))
                    vxb.append(_f(pl.get("vx")))
                    vyb.append(_f(pl.get("vy")))
                    spb.append(_f(pl.get("speed")))
                    v, c = pl.get("visibility"), pl.get("confidence")
                    vis.append(str_cache.setdefault(v, v))
                    conf.append(str_cache.setdefault(c, c))
            for ball in fr.get("balls", []):
                if not ball:
                    continue
                fc.append(frame_count)
                per.append(period)
                ib.append(1)
                t_b.append(t_s)
                g_b.append(_f(game_s))
                tid.append(nan)
                pids.append(None)
                gkb.append(False)
                xb.append(_f(ball.get("x")))
                yb.append(_f(ball.get("y")))
                vxb.append(_f(ball.get("vx")))
                vyb.append(_f(ball.get("vy")))
                spb.append(_f(ball.get("speed")))
                v, c = ball.get("visibility"), ball.get("confidence")
                vis.append(str_cache.setdefault(v, v))
                conf.append(str_cache.setdefault(c, c))

    periods = np.frombuffer(per, dtype=np.int64)
    direction = np.empty(len(periods), dtype=object)
    for p, d in _directions_from_keeper_x(keeper_xs).items():
        direction[periods == p] = d

    # column order is part of the contract, and a schema test pins it
    df = pd.DataFrame({
        "frame_count": np.frombuffer(fc, dtype=np.int64), "period_id": periods,
        "timeelapsed_from_period_start_s": np.frombuffer(t_b, dtype=np.float64),
        "period_elapsed_s": np.frombuffer(g_b, dtype=np.float64),
        "team_id": np.frombuffer(tid, dtype=np.float64), "player_id": pids,
        "is_ball": np.frombuffer(ib, dtype=np.int64),
        "is_goalkeeper": np.frombuffer(gkb, dtype=np.int8).astype(bool),
        "direction_of_play": direction,
        "x_tracking": np.frombuffer(xb, dtype=np.float64),
        "y_tracking": np.frombuffer(yb, dtype=np.float64),
        "vx": np.frombuffer(vxb, dtype=np.float64), "vy": np.frombuffer(vyb, dtype=np.float64),
        "speed": np.frombuffer(spb, dtype=np.float64),
        "visibility": vis, "confidence": conf,
    })

    # 0-100 Opta-style scale alongside the meters used by every geometry computation,
    # for callers that read x/y directly (renders).
    df["x"], df["y"] = meters_to_opta(df["x_tracking"], df["y_tracking"])

    if float32:
        float_cols = df.select_dtypes("float64").columns
        df[float_cols] = df[float_cols].astype("float32")

    return df
