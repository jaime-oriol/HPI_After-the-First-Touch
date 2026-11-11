"""run_value — the competing-outcomes value model on the cached corner table.

Nested tuning first, then the fit, calibration, xSecondBall and ablations of the configuration it
adopts, so the calibration record describes the published head and not the constructor's defaults.
The uniform floor is printed but is NOT the comparison that matters: almost anything beats
ln(#classes). The rung that carries a claim is the delivery-location baseline, and it lives in
pipeline/run_validation.py together with the shuffled controls and league-out.

The nested tuner sweeps 6 C x 2 class weights x 3 penalty families = 36 configs, and two of the
three families need `saga`, which is slow to converge on the grant's table, so the grid is the cost
of the run and `--jobs` fans it out. The nesting matches `run_validation.py` (3 outer x 5 inner) and
the grid does not: that driver sweeps l2 only, so the two adopt different penalty families on one
table while agreeing on C and the class weighting (evidence.md, the paragraph on
hyperparameters). `--deep` runs 5 x 5, which is ~1.7x the work for a tighter outer estimate.

Warnings are NOT suppressed here. A ConvergenceWarning from the tuner means a config was fitted to
max_iter without converging, and hiding that is how a half-fitted model wins a grid search.

Class-wise calibration is printed beside the top-1 figures because top-1 never scores a class the
model does not rank first, so a badly calibrated minority class is invisible in it
(methods_second_ball.md §4). The hurdle's second stage is printed under both families as a
sensitivity: the gate has already removed the zeros, so gamma is the family and methods §2.2 ranks
Tweedie second. Only the grant's table writes the calibration record, one summary row that
report_figures binds and one file of class-wise rows: the driver smoke tests point --table at a
handful of corners, and that must not overwrite the record prose is bound to.

    python pipeline/run_value.py --jobs 12             # uses the cached table
    python pipeline/run_value.py --build --jobs 20      # rebuild from the Big-5 grant
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO / "src"))
sys.path.insert(0, str(_REPO / "pipeline"))
_GRANT_TABLE = _REPO / "data" / "derived" / "corners.parquet"

import loader_sb as L                                   # noqa: E402
from build_table import assert_population, published_tag   # noqa: E402
from s03_value import build_corner_table, CompetingOutcomesModel   # noqa: E402


def main() -> None:
    """Fit, calibrate, nested-tune and ablate the competing-outcomes model."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--table", type=str, default=str(_GRANT_TABLE))
    ap.add_argument("--build", action="store_true", help="rebuild from the Big-5 grant instead of loading")
    ap.add_argument("--jobs", type=int, default=1,
                    help="workers for the rebuild and for the tuning grid, which is "
                         "embarrassingly parallel and is the whole cost of tuning")
    ap.add_argument("--boot", type=int, default=2000, help="bootstrap replicates for the ablation intervals")
    ap.add_argument("--deep", action="store_true",
                    help="5x5 nested tuning instead of 3x5 (~1.7x the work)")
    ap.add_argument("--out-dir", type=str, default=None,
                    help="where the records land (default: data/derived); the pass under partition F "
                         "writes the same names into its own folder")
    args = ap.parse_args()

    if not args.build and not Path(args.table).exists():
        raise SystemExit(f"STOP: {args.table} absent; pipeline/build_table.py writes it, or pass "
                         f"--build to rebuild the corner grant")
    tab = (build_corner_table(L.list_matches()["match_id"].tolist(), n_jobs=args.jobs)
           if args.build else pd.read_parquet(args.table))
    print(f"table: {len(tab)} deliveries, {tab['match_id'].nunique()} matches\n")
    tag = published_tag(args.table)
    if tag is not None:
        assert_population(tab, tag)

    m = CompetingOutcomesModel()
    nest = 5 if args.deep else 3
    tune = m.tune_cv(tab, n_outer=nest, n_jobs=args.jobs)
    if tune.n_outer < 2:
        print(f"TUNING: non-operative ({tune.n_outer} match group) — needs several matches.\n")
    else:
        m.apply_best_params(tune)
        print(f"TUNING (nested grouped): best={tune.best_params}  "
              f"outer_log_loss={tune.outer_log_loss:.4f} +/- {tune.outer_log_loss_std:.4f}  "
              f"(adopted for everything below)\n")

    cv = m.fit_cv(tab)
    print("COMPETING-OUTCOMES (grouped-by-match OOF):")
    print(f"  classes = {cv.class_counts}")
    print(f"  log_loss={cv.log_loss:.4f}  brier={cv.brier:.4f}  acc={cv.accuracy:.4f}  "
          f"smoothECE={cv.smooth_ece:.4f}  (binned ece={cv.ece:.4f}, withdrawn: it moves with "
          f"the bin count)")
    print(f"  uniform floor ln({len(cv.classes)})={np.log(len(cv.classes)):.4f} — a weak bar; "
          f"see run_validation.py for the ladder that matters")

    cal = m.calibrate_cv(tab)
    print("\nCALIBRATION (temperature, fit on inner grouped folds only):")
    print(f"  log_loss {cal.log_loss_before:.4f} -> {cal.log_loss_after:.4f}  |  "
          f"smoothECE {cal.smooth_ece_before:.4f} -> {cal.smooth_ece_after:.4f}  |  "
          f"brier {cal.brier_before:.4f} -> {cal.brier_after:.4f}")
    print(f"  (binned ece {cal.ece_before:.4f} -> {cal.ece_after:.4f}, reported for comparison only)")
    yi = np.array([cv.classes.index(v) for v in tab["target"].to_numpy()])
    cw = m.classwise_smooth_ece(yi, cal.oof_proba_cal, cv.classes)
    print("\n  class-wise (one-vs-rest, full probability range):")
    print(cw.to_string(index=False, float_format=lambda v: f"{v:.4f}"))

    xsb = m.xsecondball(tab)
    print(f"\nxSecondBall (lambda={m.lambda_counter}): mean={xsb.mean():.4f}  "
          f"range=[{xsb.min():.4f}, {xsb.max():.4f}]")
    if m.lambda_counter == 0.0:
        print("  the transition term is switched off: not estimable, see pipeline/sweep_counter.py")
    tw = m.hurdle_xg(tab, "after_clearance_xg", second_stage="tweedie")
    ga = m.hurdle_xg(tab, "after_clearance_xg", second_stage="gamma")
    print(f"  hurdle second stage: gamma mean={ga.mean():.5f}  tweedie mean={tw.mean():.5f}  "
          f"corr={np.corrcoef(ga, tw)[0, 1]:.4f}")

    print("\nABLATIONS (Δ log-loss when a group is dropped; + = the group helps):")
    print(m.ablate(tab, n_boot=args.boot).to_string(index=False,
                                                    float_format=lambda v: f"{v:+.5f}"))
    print("  a Δ whose [lo, hi] covers 0 is not a finding.")

    if tag is None:
        print("\nnot writing the calibration record: --table is not a published grant table")
        return
    out = Path(args.out_dir or _REPO / "data" / "derived") / f"calibration{tag}.parquet"
    out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([{
        "n": len(tab),
        "log_loss_before": cal.log_loss_before, "log_loss_after": cal.log_loss_after,
        "ece_before": cal.ece_before, "ece_after": cal.ece_after,
        "smooth_ece_before": cal.smooth_ece_before, "smooth_ece_after": cal.smooth_ece_after,
        "brier_before": cal.brier_before, "brier_after": cal.brier_after,
    }]).to_parquet(out, index=False)
    cwo = out.with_name(f"calibration_classwise{tag}.parquet")
    cw.to_parquet(cwo, index=False)
    print(f"\nwrote {out}\nwrote {cwo}")


if __name__ == "__main__":
    main()
