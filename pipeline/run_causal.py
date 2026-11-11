"""run_causal — cross-fitted AIPW effect of a frontal protector on the cached corner table.

Estimates what holding a protector in the frontal (FIFA's edge of the box) does to the
outcome, doubly robust and cross-fitted by match, with cluster-robust CR1 standard errors, and
seals the result to `data/derived/causal.parquet`, or `causal_<kind>.parquet` from a single
free-kick kind's grant table. The artefact is the point: a figure that reaches only a console
has no registry key and no bound literal, so it drifts with the next change to the design matrix
or to the propensity and nothing in the repo goes red.

`conceded_xg` is reported beside the headline and is expected to come back DEGENERATE.
That is the honest answer, not a gap: the transition arm is not estimable on this data at
any window (pipeline/sweep_counter.py). It is written to the artefact rather than skipped,
because an absent row and a measured refusal read identically to whoever reads the file
next. `after_first_touch` is the estimand that carries the project's question, the one FIFA's
corner study raises by setting two shares side by side, the corners that reach a second phase with
and without a designated edge player (docs/sota/FIFA_2025_FCWC_Corner_Defending.md), and it is
reported as machinery, not as a conclusion.

WHICH ESTIMAND, decided by the overlap and not by habit. A large share of propensities sit
outside the usual [0.05, 0.95] band, which is positivity failing rather than fraying, and a
doubly-robust ATE is the FRAGILE choice under it (Yang et al. 2026). So the run reports all three
targets side by side, which is Ben-Michael & Keele's (2023) workflow: ATT and ATO agreeing is
evidence the ATT is identifiable, and them diverging means the ATO is the only estimand the data
supports. The ATE row stays in view so the size of what clipping papers over is visible.

WHICH CONFOUNDERS. The primary is the one the protector sheet registers
(data/derived/prereg_2026-10-01/protector.md): every feature group but the treatment's,
`box_depth` and `counter_lane`, the two read off the defenders short of the box. Treatment and
confounders are all read in the delivery freeze-frame, so neither precedes the other and no test
on this data separates a confounder from a mediator. What the data CAN say is how much the answer
moves, so around the primary runs a leave-one-group-out curve from the full design. Whatever a
specification names, s05 keeps the treatment and what rebuilds it out of X.

WITH TEAM AND LEAGUE CONTROLS, as form_answers.md Q4 commits to. Every specification carries
the league and the attacking and defending teams as fixed effects (s05_causal). The primary run
without them is estimated once more beside the rest, under `sensitivity`, so what the controls
move is read off the artefact rather than asserted.

    python pipeline/run_causal.py
    python pipeline/run_causal.py --outcome first_touch
    python pipeline/run_causal.py --jobs 1          # the serial reference the parallel run must match
"""

from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed, parallel_config
from sklearn.exceptions import ConvergenceWarning

_REPO = Path(__file__).resolve().parents[1]
_GRANT_TABLE = _REPO / "data" / "derived" / "corners.parquet"
sys.path.insert(0, str(_REPO / "src"))
sys.path.insert(0, str(_REPO / "pipeline"))

from build_table import assert_population, published_tag   # noqa: E402
from s03_value import FEATURE_GROUPS                   # noqa: E402
from s05_causal import TREATMENT_GROUP, CausalRestDefence, CausalResult  # noqa: E402

PRIMARY_SPEC = "primary"
_PRIMARY_OUT = (TREATMENT_GROUP, "box_depth", "counter_lane")
_FIXED_EFFECTS = ("league", "attacking_team", "defending_team")


def _row(r: CausalResult, block: str, spec: str, level: int) -> dict:
    """One CausalResult flattened, with the keys that say which cell it is."""
    o = r.overlap
    lo, hi = r.ci_aipw
    return {
        "block": block, "spec": spec, "confounders_out": r.confounders,
        "fixed_effects": r.fixed_effects, "outcome": r.outcome,
        "target": r.target, "level": level, "n": r.n, "n_treated": r.n_treated,
        "n_control": r.n_control, "n_clusters": r.n_clusters, "ess": r.ess,
        "effect_naive": r.effect_naive, "effect_ipw": r.effect_ipw, "effect": r.effect_aipw,
        "se": r.se_aipw, "lo": lo, "hi": hi,
        "covers_zero": bool(lo <= 0 <= hi) if np.isfinite(lo) else None,
        "n_clipped": None if o is None else o.n_clipped,
        "rho_min": None if o is None else o.rho_min,
        "rho_max": None if o is None else o.rho_max,
        "frac_extreme": None if o is None else o.frac_extreme,
        "overlap_adequate": None if o is None else o.adequate,
        "c_prop": r.c_prop, "c_out1": r.c_out1, "c_out0": r.c_out0,
        "degenerate": r.degenerate, "note": r.note,
    }


def _task(table_path: str, block: str, spec: str, exclude: tuple, outcome: str,
          level: int, fixed_effects: tuple = _FIXED_EFFECTS) -> list:
    """One specification, one treatment level: three estimands off ONE cross-fit.

    Reads the table itself rather than receiving it, so the parent never ships a frame to
    every worker, and returns DICTS rather than a frame: the parent assembles a single
    DataFrame so dtypes are inferred over all rows at once. A per-task frame plus concat
    infers per task, and a column that is all-null in one specification would land on a
    different dtype than the serial path gives."""
    tab = pd.read_parquet(table_path)
    m = CausalRestDefence(exclude_groups=exclude, fixed_effects=fixed_effects)
    return [_row(r, block, spec, level)
            for r in m.results(tab, treatment_level=level, outcome=outcome)]


def _plan(outcome: str) -> list:
    """Every specification as (block, spec, exclude, outcome, level[, fixed effects]), each one
    cross-fit. The primary is no rung of the curve: it leaves out two groups beside the
    treatment's, and a rung leaves out one group from the full design."""
    tasks = [("headline", PRIMARY_SPEC, _PRIMARY_OUT, outcome, 1),
             ("spec_curve", "full", (TREATMENT_GROUP,), outcome, 1)]
    tasks += [("spec_curve", f"drop_{g}", (TREATMENT_GROUP, g), outcome, 1)
              for g in FEATURE_GROUPS if g != TREATMENT_GROUP]
    tasks += [("contrast", PRIMARY_SPEC, _PRIMARY_OUT, outcome, 2),
              ("degenerate", PRIMARY_SPEC, _PRIMARY_OUT, "conceded_xg", 1),
              ("sensitivity", "no_fixed_effects", _PRIMARY_OUT, outcome, 1, ())]
    return tasks


def _show(sub: pd.DataFrame, title: str) -> None:
    """One block of the artefact, printed so a run can be read without opening the parquet."""
    print(f"\n{title}")
    cols = ["spec", "target", "effect", "se", "lo", "hi", "ess", "n_clipped",
            "frac_extreme", "degenerate"]
    print(sub[cols].to_string(index=False,
                              float_format=lambda v: f"{v:+.4f}" if abs(v) < 100 else f"{v:.0f}"))


def main() -> None:
    """Estimate every specification, print the blocks, and seal the artefact."""
    # Never ConvergenceWarning: hiding a half-fitted model is how one wins a grid search.
    # The full reasoning, joblib caveat included, lives once in s03_value._grid_loss.
    warnings.filterwarnings("ignore")
    warnings.filterwarnings("default", category=ConvergenceWarning)
    ap = argparse.ArgumentParser()
    ap.add_argument("--table", type=str, default=str(_GRANT_TABLE))
    ap.add_argument("--out", type=str, default=None,
                    help="where to seal the result; defaults to the published artefact, which "
                         "is only written when --table is the grant table")
    ap.add_argument("--outcome", type=str, default="after_first_touch",
                    help="a competing class (binary) or a continuous column")
    ap.add_argument("--jobs", type=int, default=None,
                    help="specifications in parallel (default: one worker per specification); "
                         "each is one cross-fit with nested tuning")
    args = ap.parse_args()

    if not Path(args.table).exists():
        raise SystemExit(f"no corner table at {args.table} — run pipeline/build_table.py first")
    tab = pd.read_parquet(args.table)
    print(f"table: {len(tab)} deliveries, {tab['match_id'].nunique()} matches")
    tag = published_tag(args.table)
    if tag is not None:
        assert_population(tab, tag)

    plan = _plan(args.outcome)
    jobs = args.jobs or len(plan)
    print(f"{len(plan)} specifications, {jobs} worker(s)\n")
    if jobs == 1:
        chunks = [_task(args.table, *p) for p in plan]
    else:
        # One BLAS thread per worker, the parallelism being at the specification level; results
        # come back in submission order, so the artefact's row order does not depend on timing.
        with parallel_config(backend="loky", inner_max_num_threads=1):
            chunks = Parallel(n_jobs=jobs)(delayed(_task)(args.table, *p) for p in plan)

    ran = [r for chunk in chunks for r in chunk]
    order = {"headline": 0, "contrast": 1, "degenerate": 2, "sensitivity": 3, "spec_curve": 4}
    df = pd.DataFrame(sorted(ran, key=lambda r: order[r["block"]]))

    _show(df[df["block"] == "headline"],
          f"{args.outcome.upper()} — the primary specification, all three targets:")
    print("  ATT and ATO agreeing means the ATT is identifiable; diverging means the ATO")
    print("  is the only estimand this overlap supports. ATE is shown to size what")
    print("  clipping alone hides, never to be read as the headline.")

    deg = df[df["block"] == "degenerate"]
    print(f"\nCONCEDED xG (the transition cost form_answers.md Q2 promised): "
          f"{'DEGENERATE — ' + deg['note'].iloc[0] if bool(deg['degenerate'].iloc[0]) else 'estimable'}")

    _show(df[df["block"] == "contrast"], "at treatment level 2 (two frontal protectors held):")
    _show(df[df["block"] == "sensitivity"], "the primary specification without team and league:")

    curve = df[df["block"] == "spec_curve"]
    _show(curve, "specification curve — one rung per feature group left out:")
    for tg, sub in curve.groupby("target"):
        signs = set(np.sign(sub["effect"].to_numpy()))
        print(f"  {tg}: {int((~sub['covers_zero'].astype(bool)).sum())} of {len(sub)} rungs "
              f"exclude zero, effect in [{sub['effect'].min():+.4f}, {sub['effect'].max():+.4f}], "
              f"sign {'flips' if len(signs) > 1 else 'holds'}")

    # Same grant-table guard as run_value: a --table pointing at a subset must not overwrite the
    # record the registry is built from.
    if args.out is None and tag is None:
        print("\nnot writing the causal record: --table is not a published grant table")
    else:
        out = args.out or str(_REPO / "data" / "derived" / f"causal{tag}.parquet")
        Path(out).parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(out, index=False)
        print(f"\nwrote {out}  ({len(df)} rows)")
    print("NOTE: cross-fitted AIPW machinery. Sign and size are not conclusions, and the")
    print("causal reading is bounded by the confounders the freeze-frame can actually see.")


if __name__ == "__main__":
    main()
