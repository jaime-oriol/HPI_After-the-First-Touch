"""run_race — Run the second-ball race over the granted Hudl Broadcast X&Y sample.

Per corner: first contact -> race to the loose ball (time-to-arrive + measured
reaction lag) -> predicted recoverer + recovery probability (soft-argmin over the
same arrival times, no pitch control; Spearman/OBSO stays only as cited lineage
in s04_race and as a console cross-check in renders/render_race), with the
predicted-vs-actual agreement as a sanity check (validation, NOT a conclusion).

Events come from loader_sb (StatsBomb) and tracking from loader (Broadcast X&Y);
the two feeds use different match-id spaces, so `--match` is the StatsBomb id and
loader pairs it to a tracking file by (home, away, date). Only the 30 tracked
Serie A matches resolve.

The soft-argmin temperature is calibrated OOF grouped by match and adopted only
when the operativity guard clears: one match cannot validate tau, so a single-match
run deliberately falls back to the default. `--all` is the real calibration run:
every tracked match raced, tau tuned across all of them, and one row per raced
corner persisted for the argument.md Q2 analysis. Tracking is the memory-bound stage
(loader.load_tracking documents the per-match peak), so --jobs stays at 2-3 and
never runs alongside an events fan-out.

    python pipeline/run_race.py --match 4004219
    python pipeline/run_race.py --all --jobs 2
    python pipeline/run_race.py --all --jobs 2 --kinds free_kick_flank   # race_corners_<kind>
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO / "src"))
sys.path.insert(0, str(_REPO / "pipeline"))

import loader                                           # noqa: E402
import loader_sb as L                                   # noqa: E402
from build_table import kind_tag                        # noqa: E402
from constants import RECOVERY_TAU_S                    # noqa: E402
from s04_race import (                                  # noqa: E402
    SecondBallRace, find_corners, calibrate_tau, _softmax_p_att,
)


def match_records(mid: int, kinds: tuple = ("corner",)) -> tuple:
    """(race records, n deliveries) for one tracked match: the --all worker.

    Returns records only (small dicts of arrays); the big tracking frame dies with
    the task, which is what keeps a 2-worker fan-out inside the memory budget."""
    ev = L.load_events(mid)
    race = SecondBallRace(loader.load_tracking(mid), ev)
    corners = find_corners(ev, kinds)
    return race.race_records(corners, mid), len(corners)


def run_all(jobs: int, kinds: tuple = ("corner",), out: Path | None = None) -> None:
    """Race every tracked match, calibrate tau across them, persist the corner grain.

    The parquet gets one row per raced corner with P(att recovers) at the ADOPTED
    tau, recomputed in the parent from each record's arrival times, so no second
    pass over tracking is ever needed. `out` defaults to the published name of the
    population."""
    mids = loader.tracked_matches()
    t0 = time.time()
    if jobs == 1:
        chunks = [match_records(m, kinds) for m in mids]
    else:
        chunks = Parallel(n_jobs=jobs, verbose=5)(delayed(match_records)(m, kinds) for m in mids)
    records = [r for recs, _ in chunks for r in recs]
    n_corners = sum(n for _, n in chunks)
    print(f"\n{len(records)} raced / {n_corners} corners over {len(mids)} tracked matches "
          f"in {time.time() - t0:.0f}s ({jobs} worker(s))")

    cal = calibrate_tau(records)
    if cal["operative"]:
        tau = cal["tau"]
        print(f"tau calibrated = {tau:.2f}s  (OOF logloss {cal['oof_log_loss']:.3f}; "
              f"{cal['n']} labelled corners / {cal['n_groups']} matches; "
              f"fold taus {cal['fold_taus']}; default was {RECOVERY_TAU_S}) -> adopted")
    else:
        tau = RECOVERY_TAU_S
        print(f"tau calibration non-operative ({cal['n']} labelled / {cal['n_groups']} groups) -> "
              f"keep default tau={RECOVERY_TAU_S}s")

    rows = []
    for r in records:
        tti = np.sort(np.asarray(r["tti"], dtype=float))
        rows.append({
            "match_id": r["match_id"], "corner_id": r["corner_id"], "period": r["period"],
            "t_contact_s": r["t_contact_s"], "n_racers": len(r["tti"]),
            "n_att_racers": int(np.asarray(r["is_att"]).sum()),
            "pred_att": bool(r["is_att"][np.argmin(r["tti"])]),
            "actual_att": r["y"],
            "p_att": _softmax_p_att(r["tti"], r["is_att"], tau),
            "tti_best_s": float(tti[0]), "tti_margin_s": float(tti[1] - tti[0]),
            "tau_s": tau,
            "ball_estimated_at_contact": r["ball_estimated_at_contact"],
            "ball_estimated_at_loose": r["ball_estimated_at_loose"],
        })
    tab = pd.DataFrame(rows)

    lab = tab[tab["actual_att"].notna()]
    print(f"\nP(att recovers): mean {tab['p_att'].mean():.3f}  "
          f"p10/p50/p90 {tab['p_att'].quantile(0.1):.2f}/{tab['p_att'].median():.2f}/{tab['p_att'].quantile(0.9):.2f}")
    print(f"arrival margin (2nd-1st): median {tab['tti_margin_s'].median():.2f}s")
    if len(lab):
        agree = (lab["pred_att"] == lab["actual_att"]).mean()
        print(f"labelled corners: {len(lab)}/{len(tab)}  att recovered {100 * lab['actual_att'].mean():.1f}%  "
              f"pred-vs-actual agreement {100 * agree:.1f}% (sanity, correlated by construction, not validation)")

    # The bound every figure above carries, and the same figures over the corners whose ball
    # the vendor actually detected at the contact. A bound at the adopted tau, not a refit.
    est_c = tab["ball_estimated_at_contact"].astype(bool)
    est_l = tab["ball_estimated_at_loose"]
    print(f"\nball ESTIMATED at the contact frame on {100 * est_c.mean():.1f}% of raced corners, "
          f"at the loose frame on {100 * est_l.eq(True).mean():.1f}% ({int(est_l.isna().sum())} absent there)")
    vis = lab[~lab["ball_estimated_at_contact"].astype(bool)]
    if len(vis):
        print(f"VISIBLE ball at contact only: {int((~est_c).sum())} raced, {len(vis)} labelled, "
              f"att recovered {100 * vis['actual_att'].mean():.1f}%, "
              f"agreement {100 * (vis['pred_att'] == vis['actual_att']).mean():.1f}%")

    out = out or _REPO / "data" / "derived" / f"race_corners{kind_tag(kinds)}.parquet"
    out.parent.mkdir(parents=True, exist_ok=True)
    tab.to_parquet(out)
    print(f"\nsaved {out}  ({len(tab)} raced corners)  [gitignored]")


def run_one(mid: int, kinds: tuple = ("corner",)) -> None:
    """Race one tracked match and print the per-corner detail (the smoke view)."""
    ev = L.load_events(mid)
    race = SecondBallRace(loader.load_tracking(mid), ev)
    corners = find_corners(ev, kinds)

    cal = calibrate_tau(race.race_records(corners, mid))
    if cal["operative"]:
        race.tau = cal["tau"]
        print(f"tau calibrated = {cal['tau']:.2f}s  (OOF logloss {cal['oof_log_loss']:.3f}; "
              f"{cal['n']} corners / {cal['n_groups']} matches; default was {RECOVERY_TAU_S}) -> adopted\n")
    else:
        print(f"tau calibration non-operative ({cal['n']} corners / {cal['n_groups']} match group) -> "
              f"keep default tau={RECOVERY_TAU_S}s; one match cannot validate it\n")

    raced = agree = 0
    for i, c in corners.iterrows():
        s = race.summary(c)
        if s is None:
            print(f"  corner {i:2d}: no first contact (short / no aerial)")
            continue
        raced += 1
        agree += int(s["pred_matches_actual"])
        print(f"  corner {i:2d}: P(att recov)={s['recovery_prob_att']:.2f}  tti={s['pred_tti_s']:.2f}s  "
              f"pred_att={s['pred_recoverer_att']}  actual_att={s['actual_recoverer_att']}  "
              f"ok={s['pred_matches_actual']}")

    print(f"\n{raced}/{len(corners)} corners raced; predicted recoverer team matches actual "
          f"in {agree}/{raced} (validation, not a conclusion)")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--match", type=int, default=4004219)
    ap.add_argument("--all", action="store_true", help="race all tracked matches and calibrate tau across them")
    ap.add_argument("--jobs", type=int, default=2,
                    help="workers for --all (tracking is memory-bound: keep at 2-3)")
    ap.add_argument("--kinds", type=str, default="corner", help="the s01 population to race")
    ap.add_argument("--out", type=str, default=None,
                    help="--all output parquet (default: data/derived/race_corners[_kind].parquet)")
    args = ap.parse_args()
    kinds = tuple(k.strip() for k in args.kinds.split(","))
    if args.all:
        run_all(args.jobs, kinds, Path(args.out) if args.out else None)
    else:
        run_one(args.match, kinds)


if __name__ == "__main__":
    main()
