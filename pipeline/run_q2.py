"""run_q2 — Plan or race? Join the two halves of the title question.

argument.md Q2 asks whether a won second ball comes from the PLAN (a body staked at the
drop before the delivery) or from the RACE (someone arriving faster on the day). The
two halves live in different stages and on different substrates:

  s08 (`situation.parquet`)  the plan, on the whole grant: the four states and
                             what each is worth in after-clearance xG;
  s04 (`race_corners.parquet`) the race, on the 30 tracked matches only: an
                             arrival-time model over Broadcast X&Y.

Neither answers Q2 alone. This driver joins them on the corner id and reports the
overlap, which is as far as the tracked sample reaches on this grant.

The headline it produces is an INDEPENDENT validation the race stage cannot give
itself. `run_race.py` compares its predicted recoverer against a label read from
the same tracking positions a second later, so those two are correlated by
construction and that agreement is a harness check. Here the comparison is against
`won_second_ball`, derived from the EVENT chain by s08: a different feed, a
different vendor pipeline, no shared construction.

It is persisted for the same reason: a figure nothing regenerates is one no test can
see drift, and every component this one is built from has a generated twin.
`q2_overlap.parquet` is its own.

The bootstrap default is 5000 rather than the 2000 used elsewhere, because the overlap
is the smallest population in the repo and its clustered percentiles need more draws to
settle. The default is what the battery passes, so the artefact reproduces from it.

    python pipeline/run_q2.py
    python pipeline/run_q2.py --kinds free_kick_flank      # reads and writes the <kind> tables
    python pipeline/run_q2.py --tables data/derived/smoke/tracked --boot 500 \\
        --out data/derived/smoke/q2_overlap.parquet         # the smoke's tracked sample
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO / "src"))
sys.path.insert(0, str(_REPO / "pipeline"))
_DER = _REPO / "data" / "derived"

from build_table import kind_tag                                      # noqa: E402
from constants import MIN_CELL                                        # noqa: E402
from s08_situation import FOUR_STATES                                 # noqa: E402
from validate import BootCI, cluster_bootstrap_mean, cluster_bootstrap_stats   # noqa: E402

# Both derived from s08's own map: the print order is its value order, and a "won" state
# is exactly one whose `won` half of the (staked, won) key is True.
STATES = tuple(FOUR_STATES.values())
WON_STATES = tuple(v for (_staked, won), v in FOUR_STATES.items() if won)


def load_join(tag: str = "", der: Path = _DER) -> pd.DataFrame:
    """Race corners joined to their situation row, one row per corner carrying both.

    The join key is the corner's own event id (`corner_id` on the race side,
    `set_piece_id` on the situation side), so a corner only survives if the same
    delivery was both raceable on tracking and cleared in the event chain. `tag` names the
    population both halves were built for, `der` where they live."""
    race = pd.read_parquet(der / f"race_corners{tag}.parquet")
    sit = pd.read_parquet(der / f"situation{tag}.parquet")
    j = race.merge(sit.drop(columns=["match_id"]), left_on="corner_id",
                   right_on="set_piece_id", how="inner")
    assert j["corner_id"].is_unique, "duplicate corner in the race/situation join"
    return j


def records(j: pd.DataFrame, lab: pd.DataFrame, ci: BootCI, self_agree: float,
            val: pd.DataFrame, both: pd.DataFrame) -> pd.DataFrame:
    """Every reported quantity in ONE long schema, so report_figures can bind any of them.

    Four blocks share (n, point, lo, hi): the overlap and its coverage, the agreement rates
    (the independent one, the majority rule it is read against, and the harness's own), the
    race probability by outcome, and the plan x race grid. Only the bootstrapped
    rows carry an interval; the rest leave it null rather than borrowing a column."""
    rows = [{"block": "overlap", "key": "corners", "n": len(j)},
            {"block": "overlap", "key": "matches", "n": int(j["match_id"].nunique())}]
    for col in ("won_second_ball", "staked", "four_state"):
        rows.append({"block": "overlap", "key": col, "n": int(j[col].notna().sum()),
                     "point": float(j[col].notna().mean())})
    rows.append({"block": "agreement", "key": "independent", "n": len(lab),
                 "point": ci.point, "lo": ci.lo, "hi": ci.hi})
    # The bar a hard prediction has to clear is the MAJORITY RULE, not a coin: attackers win most
    # of these second balls, so "always the attacker" is already a strong classifier.
    rows.append({"block": "agreement", "key": "majority_rule", "n": len(lab),
                 "point": float(max(lab["won"].mean(), 1 - lab["won"].mean()))})
    rows.append({"block": "agreement", "key": "self", "n": int(j["actual_att"].notna().sum()),
                 "point": self_agree})
    for k in ("won", "lost", "won-lost"):
        r = val.loc[k]
        rows.append({"block": "race_prob", "key": k, "n": len(lab),
                     "point": r["point"], "lo": r["lo"], "hi": r["hi"]})
    for st in STATES:
        sub = both[both["four_state"] == st]
        if not len(sub):
            continue
        rows.append({"block": "plan_x_race", "key": st, "n": len(sub),
                     "point": float(sub["p_att"].mean()),
                     "value_xg_after": float(sub["value_xg_after"].mean()),
                     "tti_margin_s": float(sub["tti_margin_s"].median()),
                     "below_floor": len(sub) < MIN_CELL})
    for key, sel in (("won_pooled", both["four_state"].isin(WON_STATES)),
                     ("lost_pooled", ~both["four_state"].isin(WON_STATES))):
        rows.append({"block": "plan_x_race", "key": key, "n": int(sel.sum()),
                     "point": float(both.loc[sel, "p_att"].mean())})
    return pd.DataFrame(rows)


def main() -> None:
    """Join plan and race, then report coverage, the independent agreement check,
    and whether the race probability separates won second balls from lost ones."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--boot", type=int, default=5000, help="cluster-bootstrap draws")
    ap.add_argument("--out", type=str, default=None,
                    help="output parquet (default: data/derived/q2_overlap[_kind].parquet)")
    ap.add_argument("--kinds", type=str, default="corner", help="the s01 population to join")
    ap.add_argument("--tables", type=str, default=str(_DER),
                    help="where race_corners and situation live (default: data/derived)")
    args = ap.parse_args()

    tag = kind_tag(tuple(k.strip() for k in args.kinds.split(",")))
    der = Path(args.tables)
    if args.out is None and der.resolve() != _DER.resolve():
        raise SystemExit(f"STOP: tables outside data/derived are not the published ones, so their "
                         f"overlap may not write the published q2_overlap{tag}.parquet; pass --out")
    j = load_join(tag, der)
    print(f"Q2 OVERLAP: {len(j)} corners over {j['match_id'].nunique()} tracked matches")
    print("  (as far as the tracked sample reaches: a corner needs tracking AND a clearance "
          "in the chain)\n")

    print("coverage inside the overlap:")
    for col, label in (("won_second_ball", "won (event chain)"),
                       ("staked", "staked (360 + next touch)"),
                       ("four_state", "four state (both)")):
        print(f"  {label:24s} {int(j[col].notna().sum()):4d}  ({100 * j[col].notna().mean():.0f}%)")

    # ── The one agreement number that is not self-referential ───────────
    lab = j[j["won_second_ball"].notna()].copy()
    lab["won"] = lab["won_second_ball"].astype(bool)
    lab["hit"] = (lab["pred_att"] == lab["won"]).astype(float)
    ci = cluster_bootstrap_mean(lab["hit"].to_numpy(), lab["match_id"].to_numpy(),
                                n_boot=args.boot)
    own = j[j["actual_att"].notna()]
    self_agree = (own["pred_att"] == own["actual_att"]).mean() if len(own) else float("nan")
    majority = float(max(lab["won"].mean(), 1 - lab["won"].mean()))
    print(f"\nINDEPENDENT AGREEMENT (tracking race vs event-chain outcome), n={len(lab)}:")
    print(f"  {100 * ci.point:.1f}% [{100 * ci.lo:.1f}, {100 * ci.hi:.1f}]  against the "
          f"majority rule at {100 * majority:.1f}%, which is the bar rather than a coin")
    if ci.point < majority:
        print("  the race arm's hard prediction does NOT beat always-predict-the-attacker")
    print(f"  run_race.py's own {100 * self_agree:.1f}% is a harness check (label read from the")
    print("  same tracking a second later); THIS one compares two independent feeds.")

    # ── Does the race probability track the outcome? ────────────────────
    print("\nRACE PROBABILITY by outcome (P(att recovers) from arrival times alone):")
    stats = {
        "won": lambda f: f.loc[f["won"], "p_att"].mean(),
        "lost": lambda f: f.loc[~f["won"], "p_att"].mean(),
        "won-lost": lambda f: f.loc[f["won"], "p_att"].mean() - f.loc[~f["won"], "p_att"].mean(),
    }
    val = cluster_bootstrap_stats(lab, stats, n_boot=args.boot)
    for k in ("won", "lost"):
        r = val.loc[k]
        print(f"  {k:9s} {r['point']:.3f} [{r['lo']:.3f}, {r['hi']:.3f}]")
    d = val.loc["won-lost"]
    verdict = "covers zero" if d["lo"] <= 0 <= d["hi"] else "excludes zero"
    print(f"  delta     {d['point']:+.3f} [{d['lo']:+.3f}, {d['hi']:+.3f}]  {verdict}")

    # ── Plan and race side by side, with the publication floor enforced ─
    both = j[j["four_state"].notna()]
    print(f"\nPLAN x RACE, n={len(both)} corners carrying both:")
    print(f"  {'state':16s} {'n':>4s}  {'P(att) race':>12s}  {'xG':>8s}  {'margin s':>9s}")
    for st in STATES:
        sub = both[both["four_state"] == st]
        if not len(sub):
            continue
        flag = "" if len(sub) >= MIN_CELL else f"  [under the n={MIN_CELL} floor, not publishable]"
        print(f"  {st:16s} {len(sub):4d}  {sub['p_att'].mean():12.3f}  "
              f"{sub['value_xg_after'].mean():8.4f}  {sub['tti_margin_s'].median():9.2f}{flag}")

    won = both[both["four_state"].isin(WON_STATES)]
    print(f"\n  won states pooled ({len(won)}): P(att) {won['p_att'].mean():.3f}")
    lost = both[~both["four_state"].isin(WON_STATES)]
    print(f"  lost states pooled ({len(lost)}): P(att) {lost['p_att'].mean():.3f}")

    thin = sum(1 for st in STATES if 0 < (both["four_state"] == st).sum() < MIN_CELL)
    print(f"\nNOTE: {thin} of the four per-state cells sit under the n={MIN_CELL} floor, so the")
    print("per-state race numbers are descriptive. The pooled won-vs-lost delta above")
    print("is the testable statement, and it is bounded by 30 tracked matches.")

    out = Path(args.out) if args.out else _DER / f"q2_overlap{tag}.parquet"
    out.parent.mkdir(parents=True, exist_ok=True)
    records(j, lab, ci, self_agree, val, both).to_parquet(out, index=False)
    print(f"\nwrote {out}  [gitignored]")


if __name__ == "__main__":
    main()
