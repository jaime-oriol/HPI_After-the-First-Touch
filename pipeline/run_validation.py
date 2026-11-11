"""run_validation — the honest-validation report: does any of this actually hold up?

One console dump a reviewer reads top to bottom, its blocks in the order a sceptic would
ask for them:

  CONTROLS       labels shuffled two ways. Both must land at the marginal entropy of the
                 class mix. A shuffled model that BEATS chance means the harness itself
                 leaks, and nothing else in the repo would notice, because a leaking
                 harness produces a perfectly plausible real number.
  LADDER         uniform -> class prior -> delivery design -> full 360 features ->
                 `full_balanced`, the same design at balanced class weights, all on
                 the identical grouped CV. The rung that carries the scientific claim is
                 `delivery`, and it knows what is true at the STRIKE: the kick position,
                 the side, the swing, the taker's foot, the height and whether it was
                 short or ground. It does NOT know where the ball ended, because that point
                 IS the first contact. Beating uniform proves nothing.
  HEAD-TO-HEAD   the delivery design against all of the 360, fitted and differenced row
                 by row. A contrast and never a step: the two designs do not nest.
  MDE            what size of 360 step this design could have resolved, so a null travels
                 with its resolution rather than being a statement about the sample.
  REFIT          the same step intervalled with the model rebuilt inside every resample,
                 which is what prices how narrow every other interval here is.
  LEAGUE-OUT     five folds, one competition-season each, with train/test disjointness
                 ASSERTED on both match_id and league. The number that matters is the gap
                 to match-out: close means the model is not memorising league style.
  ABLATION       Δ log-loss per feature group, each with a match-clustered PAIRED interval,
                 and the same Δ intervalled with the model rebuilt inside every resample.
  PERMUTED       the same Δ with the block SHUFFLED at fixed design width, which moves the
                 information without moving the number of columns.
  HETEROGENEITY  whether the pooled step hides help on some set pieces, asked once,
                 globally and before any subgroup is opened, beside a placebo that
                 permutes the proxy's features on the same partitions.
  CELLS          and only where it does, whether that benefit lines up with a category a
                 coach can name.
  POOLING        under --by-type only: the pooled model scored on each population's rows
                 against that population's own, since pooling is allowed only where it
                 hurts neither.

There is no out-of-time fold here and there cannot be: the grant is a single season
(Big-5 2025/26), so the dataset has no time axis to hold out on. League-out is the
external-validity axis that replaces it, asking whether the model transfers to a
competition it never saw rather than to a date it never saw.

The console dump stays the primary product, but the quantitative blocks are also written
to `data/derived/validation.parquet` in one long schema, because prose quotes these figures
more than any others and `report_figures` binds them from there. Only `--by-type` on the grant's
table of the paper's two populations (`_TWO_POP_TABLE`) writes it. A table holding one free-kick
kind writes `validation_<kind>.parquet` instead, because `validation.parquet` already carries the
corner and free_kick_flank populations.

    python pipeline/run_validation.py --table data/derived/corners_corner_free_kick_flank.parquet --by-type
    python pipeline/run_validation.py --boot 500 --quick     # skip the ablation sweep, own file
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

from build_table import kind_tag, published_tag              # noqa: E402
from constants import MIN_CELL                                     # noqa: E402
from eda_build import COMBINED_KINDS                               # noqa: E402
from s03_value import LADDER_TUNE, CompetingOutcomesModel, GroupOutResult     # noqa: E402
from s04b_snapshot import mde, power_curve                         # noqa: E402
from validate import cluster_bootstrap_mean                        # noqa: E402

# Candidate 360 steps the MDE sweep asks about, in nats. The grid tops out below the corner
# delivery step (measured.md §validation), so an effect the size of the baseline's own would
# sit off the chart.
_POWER_DELTAS = (0.0005, 0.001, 0.002, 0.003, 0.005, 0.0075, 0.01, 0.015, 0.02)


_OUT = _REPO / "data" / "derived" / "validation.parquet"
_TWO_POP_TABLE = f"corners{kind_tag(COMBINED_KINDS)}.parquet"


def _fmt(v: float, w: int = 8, p: int = 4) -> str:
    return f"{v:>{w}.{p}f}" if np.isfinite(v) else " " * (w - 3) + "n/a"


def _tidy(population: str, ctl: list, lad: pd.DataFrame, go: GroupOutResult, cv_log_loss: float,
          ab: pd.DataFrame | None, h2h: dict | None = None,
          perm: pd.DataFrame | None = None, pw: pd.DataFrame | None = None,
          het: dict | None = None, cells: pd.DataFrame | None = None,
          refit: dict | None = None, ab_refit: pd.DataFrame | None = None) -> pd.DataFrame:
    """Every quantitative block in ONE long schema, so report_figures can bind any of them.

    Ladder and ablation share a shape: a name, a log-loss, a delta and an interval on that
    delta. The ladder additionally carries an interval on the log-loss itself, which the
    ablation has no counterpart for, hence the two interval pairs. Controls and league-out
    each carry quantities nothing else has, so those get their own columns rather than
    borrowed ones; the cost of a few mostly-NaN columns is worth not overloading a name.

    The `__summary__` key on the league-out block is the fold aggregate, and its `delta` is
    the figure evidence.md §6 actually leads on: what holding out a whole competition costs
    against holding out matches."""
    parts = [pd.DataFrame({
        "population": population, "block": "control", "key": [c["mode"] for c in ctl],
        "log_loss": [c["log_loss_mean"] for c in ctl],
        "entropy": [c["marginal_entropy"] for c in ctl],
        "real_log_loss": [c["log_loss_real"] for c in ctl]}),
        pd.DataFrame({
            "population": population, "block": "ladder", "key": lad["rung"],
            "log_loss": lad["log_loss"], "ll_lo": lad["lo"], "ll_hi": lad["hi"],
            "delta": lad["step"], "delta_lo": lad["step_lo"], "delta_hi": lad["step_hi"]}),
        pd.DataFrame({
            "population": population, "block": "league_out",
            "key": [str(f["held_out"]) for f in go.folds],
            "log_loss": [f["log_loss"] for f in go.folds],
            "n_test": [f["n_test"] for f in go.folds],
            "n_train": [f["n_train"] for f in go.folds],
            "n_train_matches": [f["n_train_matches"] for f in go.folds],
            "brier": [f["brier"] for f in go.folds],
            "accuracy": [f["accuracy"] for f in go.folds],
            "ece": [f["ece"] for f in go.folds],
            "smooth_ece": [f["smooth_ece"] for f in go.folds],
            "classes_missing": [bool(f["classes_missing_from_train"]) for f in go.folds]}),
        pd.DataFrame({
            "population": population, "block": "league_out", "key": ["__summary__"],
            "log_loss": [go.log_loss], "ll_lo": [go.log_loss_mean], "ll_hi": [go.log_loss_std],
            "delta": [go.log_loss - cv_log_loss], "real_log_loss": [cv_log_loss],
            "classes_missing": [any(f["classes_missing_from_train"] for f in go.folds)]})]
    if ab is not None:
        parts.append(pd.DataFrame({
            "population": population, "block": "ablation", "key": ab["group"],
            "log_loss": ab["log_loss"], "ll_lo": np.nan, "ll_hi": np.nan,
            "delta": ab["delta"], "delta_lo": ab["lo"], "delta_hi": ab["hi"],
            "p_wilcoxon": ab["p_wilcoxon"]}))
    if h2h is not None:
        # Its own block and not a ladder rung: the two models it compares do not nest, so a
        # `step` column would invite reading it as one.
        parts.append(pd.DataFrame({
            "population": population, "block": "head_to_head", "key": ["delivery_vs_360"],
            "log_loss": [h2h["delivery_log_loss"]], "real_log_loss": [h2h["only_360_log_loss"]],
            "delta": [h2h["delta"]], "delta_lo": [h2h["lo"]], "delta_hi": [h2h["hi"]],
            "p_wilcoxon": [h2h["p_wilcoxon"]]}))
    if perm is not None:
        parts.append(pd.DataFrame({
            "population": population, "block": "permuted", "key": perm["group"],
            "delta": perm["delta"], "delta_lo": perm["lo"], "delta_hi": perm["hi"],
            "p_wilcoxon": perm["p_wilcoxon"]}))
    if pw is not None:
        # `accuracy` carries the power: one more mostly-NaN column on every other block costs
        # more than the borrowed name does, and report_figures reads this by block anyway.
        parts.append(pd.DataFrame({
            "population": population, "block": "power",
            "key": [f"{d:.4f}" for d in pw["delta"]],
            "delta": pw["delta"], "accuracy": pw["power"]}))
    if het is not None and het.get("n_reps_used"):
        parts.append(pd.DataFrame({
            "population": population, "block": "heterogeneity", "key": ["blp", "gates"],
            "delta": [het["blp_slope"], het["gates_gap"]],
            "p_wilcoxon": [het["blp_p"], het["gates_p"]],
            "n_test": [het["n_reps_used"]] * 2,
            "verdict": [het["verdict"], "actionable" if het["actionable"] else "not actionable"]}))
        # One row per placebo mode: `delta`/`p_wilcoxon` carry its BLP, and `ll_lo`/`ll_hi` borrow
        # the columns for its GATES gap and p, as the power and refit blocks borrow theirs.
        for mode, pl in het.get("placebo", {}).items():
            parts.append(pd.DataFrame({
                "population": population, "block": "heterogeneity", "key": [f"placebo_{mode}"],
                "delta": [pl.get("blp_slope", np.nan)], "p_wilcoxon": [pl.get("blp_p", np.nan)],
                "ll_lo": [pl.get("gates_gap", np.nan)], "ll_hi": [pl.get("gates_p", np.nan)],
                "n_test": [pl.get("n_reps_used", 0)], "verdict": [pl.get("verdict", "")]}))
    if cells is not None and len(cells):
        parts.append(pd.DataFrame({
            "population": population, "block": "cells", "key": cells["dimension"],
            "delta": cells["tau"], "ll_lo": cells["i2"], "ll_hi": cells["q"],
            "n_test": cells["n_cells"], "n_train": cells["min_n"],
            "p_wilcoxon": cells["q_p"], "verdict": cells["verdict"]}))
    if refit is not None:
        # The honest width beside the fixed-fit one: delta_lo/hi carry the pivotal form, ll_lo/hi the
        # percentile, and the normal reading is the point plus or minus half that width. `brier` is
        # the borrowed column for the WIDTH RATIO, as `accuracy` is in the power block above.
        parts.append(pd.DataFrame({
            "population": population, "block": "refit", "key": ["step"],
            "delta": [refit["point"]], "delta_lo": [refit["adjusted"][0]],
            "delta_hi": [refit["adjusted"][1]],
            "ll_lo": [refit["refit"][0]], "ll_hi": [refit["refit"][1]],
            "n_test": [refit["n_boot_refit"]], "brier": [refit["width_ratio"]]}))
    if ab_refit is not None:
        # Same borrowed columns as the step's refit row, one row per block.
        parts.append(pd.DataFrame({
            "population": population, "block": "ablation_refit", "key": ab_refit["group"],
            "delta": ab_refit["point"], "delta_lo": ab_refit["adjusted_lo"],
            "delta_hi": ab_refit["adjusted_hi"], "ll_lo": ab_refit["refit_lo"],
            "ll_hi": ab_refit["refit_hi"], "n_test": ab_refit["n_boot"],
            "brier": ab_refit["width_ratio"]}))
    return pd.concat(parts, ignore_index=True)


def report(table: pd.DataFrame, n_boot: int, quick: bool, tune: bool = True,
           population: str = "corner", jobs: int = 8) -> pd.DataFrame:
    """Print every validation block for one population and return the quantitative ones in
    `_tidy`'s long schema (the console stays the primary product).

    Tuned first: the ladder at untuned defaults handicaps the full model against the simpler
    rungs, so the comparison is only fair once the nested grouped CV has chosen the
    hyperparameters rather than the constructor."""
    m = CompetingOutcomesModel()
    if tune:
        t = m.tune_cv(table, n_jobs=jobs, **LADDER_TUNE)
        m.apply_best_params(t)
        print(f"tuned on nested grouped CV: {t.best_params}   "
              f"outer log-loss {t.outer_log_loss:.4f} +/- {t.outer_log_loss_std:.4f}")
    cv = m.fit_cv(table)
    print(f"n = {len(table)} deliveries, {table['match_id'].nunique()} matches, "
          f"{table['league'].nunique()} leagues")
    print("  classes:", cv.class_counts)
    thin = {c: n for c, n in cv.class_counts.items() if n < MIN_CELL}
    if thin:
        print(f"  [cells under n={MIN_CELL}, not publishable: {thin}]")

    print("\n── CONTROLS (shuffled labels must land at the marginal entropy) ──")
    ctl = []
    for mode in ("global", "within_match"):
        c = m.shuffled_control(table, mode=mode, n_repeats=3, seed=0)
        ctl.append({"mode": mode, **c})
        print(f"  {mode:13s} shuffled {c['log_loss_mean']:.4f} +/- {c['log_loss_std']:.4f}"
              f"   entropy {c['marginal_entropy']:.4f}   real {c['log_loss_real']:.4f}"
              f"   -> {c['verdict']}")

    print("\n── BASELINE LADDER (match-clustered CI; the step is PAIRED) ──")
    lad = m.baseline_ladder(table, n_boot=n_boot)
    print(f"  {'rung':10s} {'log_loss':>9s} {'95% CI':>21s} {'step vs prev':>14s} {'step 95% CI':>21s}")
    for _, r in lad.iterrows():
        ci = f"[{_fmt(r['lo'], 6)},{_fmt(r['hi'], 7)}]" if np.isfinite(r["lo"]) else " " * 21
        st = f"[{_fmt(r['step_lo'], 6)},{_fmt(r['step_hi'], 7)}]" if np.isfinite(r["step_lo"]) else " " * 21
        print(f"  {r['rung']:10s} {_fmt(r['log_loss'], 9)} {ci:>21s} {_fmt(r['step'], 14)} {st:>21s}")
    print("  a step whose interval covers 0 did not buy anything.")

    print("\n── HEAD-TO-HEAD (the delivery design against ALL of the 360; +Δ = delivery won) ──")
    h2h = m.delivery_vs_360(table, n_boot=n_boot)
    print(f"  delivery {h2h['delivery_log_loss']:.4f}   only-360 {h2h['only_360_log_loss']:.4f}"
          f"   Δ {h2h['delta']:+.4f}  [{h2h['lo']:+.4f}, {h2h['hi']:+.4f}]"
          f"   p {h2h['p_wilcoxon']:.4f}")
    print("  these two do NOT nest, so it is a contrast and never a ladder step.")

    print("\n── MDE (what size of 360 step this design could have resolved) ──")
    groups_arr = table["match_id"].to_numpy()
    d_step = m.step_rows(table)
    pw = power_curve(d_step, groups_arr, _POWER_DELTAS, n_boot=400, n_rep=200)
    print("  " + "  ".join(f"{d:+.4f}:{p:.2f}"
                           for d, p in zip(pw["delta"], pw["power"], strict=True)))
    d80 = mde(pw)
    reach = (f"MDE at 80% power {d80:.4f} nats" if np.isfinite(d80) else
             f"never reaches 80% power inside {_POWER_DELTAS[-1]:.4f}")
    print(f"  {reach}, against an observed step of {d_step.mean():+.4f}: a null below the "
          f"MDE is a statement about this design, not about football.")

    print("\n── REFIT INTERVAL (the fixed-fit one above is too narrow; by how much?) ──")
    refit = m.step_refit_interval(table, n_jobs=jobs)
    print(f"  fixed {refit['point']:+.5f} [{refit['fixed'][0]:+.5f}, {refit['fixed'][1]:+.5f}]"
          f"   refit [{refit['refit'][0]:+.5f}, {refit['refit'][1]:+.5f}]"
          f"   width x{refit['width_ratio']:.2f}  ({refit['n_boot_refit']} refits)")
    print(f"  normal [{refit['normal'][0]:+.5f}, {refit['normal'][1]:+.5f}], which is what to "
          f"read; pivotal [{refit['adjusted'][0]:+.5f}, {refit['adjusted'][1]:+.5f}]\n  beside "
          f"it. A resample thins the distinct matches and that costs the wider rung more, so "
          f"the\n  percentiles sit below the point and the pivotal form mirrors that shift above "
          f"it. The MDE\n  above scales with the width, so read it as "
          f"{d80 * refit['width_ratio']:.4f} rather than {d80:.4f}.")

    print("\n── LEAGUE-OUT (leave one competition-season out) ──")
    go = m.fit_group_out(table, group_col="league")
    print(f"  leakage asserted on {go.leakage_checked} for all {go.n_folds} folds")
    print(f"  {'held out':16s} {'n_test':>7s} {'n_train':>8s} {'tr_matches':>11s} "
          f"{'log_loss':>9s} {'brier':>7s} {'acc':>6s} {'ece':>6s}")
    for f in go.folds:
        note = "  [class missing from train]" if f["classes_missing_from_train"] else ""
        print(f"  {str(f['held_out']):16s} {f['n_test']:>7d} {f['n_train']:>8d} "
              f"{f['n_train_matches']:>11d} {_fmt(f['log_loss'], 9)} {_fmt(f['brier'], 7, 3)} "
              f"{_fmt(f['accuracy'], 6, 3)} {_fmt(f['ece'], 6, 3)}{note}")
    print(f"  pooled OOF {go.log_loss:.4f}   fold mean {go.log_loss_mean:.4f} "
          f"+/- {go.log_loss_std:.4f}")
    print(f"  grouped-by-MATCH {cv.log_loss:.4f}   ->  league-out costs "
          f"{go.log_loss - cv.log_loss:+.4f}")
    if go.degenerate_folds:
        print(f"  excluded from the fold mean (structurally degenerate): {go.degenerate_folds}")

    if quick:
        return _tidy(population, ctl, lad, go, cv.log_loss, None, h2h, pw=pw, refit=refit)
    print("\n── ABLATIONS (Δ log-loss when a group is dropped; +Δ = it helped) ──")
    ab = m.ablate(table, n_boot=n_boot)
    print(f"  {'group':14s} {'log_loss':>9s} {'delta':>9s} {'95% CI':>21s}")
    for _, r in ab.iterrows():
        ci = f"[{_fmt(r['lo'], 6)},{_fmt(r['hi'], 7)}]" if r["group"] != "ALL" else " " * 21
        print(f"  {r['group']:14s} {_fmt(r['log_loss'], 9)} {_fmt(r['delta'], 9)} {ci:>21s}")
    print("  a Δ whose interval covers 0 is not a finding.")

    print("\n── ABLATIONS, REFIT (each Δ with the model rebuilt inside every resample) ──")
    abr = m.ablation_refit_interval(table, n_jobs=jobs)
    fx = ab.set_index("group").loc[abr["group"]]
    # The refit rebuilds the same fits `ablate` scored, so its point has to be that Δ, or the
    # interval belongs to a different estimator from the number printed beside it.
    if not np.allclose(abr["point"].to_numpy(), fx["delta"].to_numpy(), rtol=0.0, atol=1e-9):
        raise SystemExit("STOP: the ablation refit point does not reproduce the fixed-fit Δ")
    abr["width_ratio"] = ((abr["refit_hi"] - abr["refit_lo"]).to_numpy()
                          / (fx["hi"] - fx["lo"]).to_numpy())
    print(f"  {'group':14s} {'delta':>9s} {'normal 95% CI':>23s} {'width':>7s}   "
          f"({int(abr['n_boot'].iloc[0])} refits)")
    for _, r in abr.iterrows():
        print(f"  {r['group']:14s} {_fmt(r['point'], 9, 5)} [{_fmt(r['normal_lo'], 8, 5)},"
              f"{_fmt(r['normal_hi'], 9, 5)}] {r['width_ratio']:>6.2f}x")
    print("  read the normal form, for the reason the REFIT INTERVAL block gives; the widening is "
          "not one\n  factor across blocks, so the step's ratio cannot stand in for it.")

    print("\n── PERMUTED (same Δ with the block SHUFFLED, at fixed design width) ──")
    pm = m.permuted_ablation(table, n_boot=n_boot)
    print(f"  {'group':14s} {'delta':>9s} {'95% CI':>21s} {'p':>8s}")
    for _, r in pm.iterrows():
        print(f"  {r['group']:14s} {_fmt(r['delta'], 9)} "
              f"[{_fmt(r['lo'], 6)},{_fmt(r['hi'], 7)}] {_fmt(r['p_wilcoxon'], 8, 4)}")
    print("  read against the dropped Δ above: where they part, the block is collinear.")

    print("\n── HETEROGENEITY (does the 360 step hide help on SOME set pieces?) ──")
    het = m.step_heterogeneity(table)
    print(f"  BLP slope {het['blp_slope']:+.4f} p {het['blp_p']:.4f}   "
          f"GATES top-bottom {het['gates_gap']:+.4f} p {het['gates_p']:.4f}   "
          f"({het['n_reps_used']} cross-fitted partitions)  ->  {het['verdict']}")
    for mode, pl in het.get("placebo", {}).items():
        print(f"  placebo {mode:12s} BLP slope {pl.get('blp_slope', np.nan):+.4f} "
              f"p {pl.get('blp_p', np.nan):.4f}   GATES {pl.get('gates_gap', np.nan):+.4f} "
              f"p {pl.get('gates_p', np.nan):.4f}  (proxy features permuted, same partitions)")
    print("  asked BEFORE any cell is opened, and only an ACTIONABLE verdict opens one: a "
          "slope\n  without a resolved group gap says something varies without saying for whom.")

    cells = None
    if het.get("actionable"):
        print("\n── CELLS (does that benefit line up with a category a coach can NAME?) ──")
        cells = m.step_cells(table)
        print(f"  {'dimension':17s} {'cells':>5s} {'min n':>6s} {'tau':>8s} {'I2':>7s} "
              f"{'Q':>7s} {'df':>3s} {'p':>8s}")
        for _, r in cells.iterrows():
            print(f"  {r['dimension']:17s} {r['n_cells']:>5d} {r['min_n']:>6d} "
                  f"{r['tau']:>8.4f} {100 * r['i2']:>6.1f}% {r['q']:>7.2f} {r['df']:>3d} "
                  f"{r['q_p']:>8.4f}")
        print(f"  {len(cells)} dimensions examined, and that count travels with any claim "
              f"made from them.")
    else:
        print("\n  [cells NOT opened: the heterogeneity verdict is not actionable]")
    return _tidy(population, ctl, lad, go, cv.log_loss, ab, h2h, pm, pw, het, cells, refit, abr)


def _tuned(table: pd.DataFrame, tune: bool, jobs: int = 8) -> CompetingOutcomesModel:
    """s03's model, tuned on `table` as the ladder tunes it (LADDER_TUNE) when `tune`, at its
    defaults otherwise."""
    m = CompetingOutcomesModel()
    if tune:
        m.apply_best_params(m.tune_cv(table, n_jobs=jobs, **LADDER_TUNE))
    return m


def pooling_contrast(table: pd.DataFrame, n_boot: int, tune: bool, jobs: int = 8) -> pd.DataFrame:
    """Pooling is allowed only if it hurts NEITHER population.

    Comparing the pooled model's overall log-loss to each population's own is not the
    test: the class mixes differ, so the numbers are not on the same scale. The test
    is per population: score the POOLED model on one population's rows and difference
    it, row by row, against the model fit on that population alone."""
    t = table.reset_index(drop=True)
    pooled = _tuned(t, tune, jobs)
    rows_pooled = pooled._row_logloss(pooled.fit_cv(t), t["target"].to_numpy())

    print("\n" + "=" * 78)
    print("POOLING CONTRAST (positive Δ = pooling COSTS that population)")
    print("=" * 78)
    rows = []
    for kind, sub in t.groupby("set_piece_type"):
        s = sub.reset_index(drop=True)
        own = _tuned(s, tune, jobs)
        own_cv = own.fit_cv(s)
        rows_own = own._row_logloss(own_cv, s["target"].to_numpy())
        d = rows_pooled[sub.index.to_numpy()] - rows_own
        ci = cluster_bootstrap_mean(d, s["match_id"].to_numpy(), n_boot=n_boot, seed=0)
        verdict = ("pooling is safe here" if ci.hi <= 0 else
                   "no evidence either way" if ci.lo <= 0 <= ci.hi else "POOLING HURTS")
        print(f"  {kind:12s} n={len(s):6d}  own {own_cv.log_loss:.4f}  "
              f"Δ(pooled-own) {ci.point:+.4f}  [{ci.lo:+.4f}, {ci.hi:+.4f}]  -> {verdict}")
        rows.append({"population": kind, "block": "pooling", "key": "pooled_minus_own",
                     "log_loss": own_cv.log_loss, "delta": ci.point,
                     "delta_lo": ci.lo, "delta_hi": ci.hi, "verdict": verdict})
    return pd.DataFrame(rows)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--table", type=str, default=str(_REPO / "data" / "derived" / "corners.parquet"))
    ap.add_argument("--boot", type=int, default=2000, help="match-clustered bootstrap replicates")
    ap.add_argument("--quick", action="store_true", help="skip the ablation sweep")
    ap.add_argument("--no-tune", action="store_true",
                    help="report at the constructor defaults instead of the tuned setting")
    ap.add_argument("--jobs", type=int, default=8,
                    help="workers for the tuning grid and the two refit intervals; the serial "
                         "blocks do not use them, so running stages side by side gains more")
    ap.add_argument("--by-type", action="store_true",
                    help="report each set_piece_type separately AND pooled, with the contrast")
    ap.add_argument("--out-dir", type=str, default=str(_OUT.parent),
                    help="where the record lands (default: data/derived); the pass under partition F "
                         "writes the same names into its own folder")
    args = ap.parse_args()

    path = Path(args.table)
    if not path.exists():
        raise SystemExit(f"STOP: {path} absent; pipeline/build_table.py writes it")
    tab = pd.read_parquet(path)

    print("=" * 78)
    print(f"VALIDATION REPORT — {path.name}")
    print("=" * 78)
    print("intervals resample MATCHES and hold the fitted OOF probabilities fixed, so")
    print("they cover evaluation-set sampling error only and are too NARROW: they do")
    print("not propagate the variance of refitting on a resampled training set.")
    print("The REFIT INTERVAL block below measures how narrow: it rebuilds the model inside")
    print("every resample and prints the width ratio, and the MDE beside it scales with that")
    print("ratio rather than being read off the fixed-fit curve.\n")

    blocks, tag = [], ""
    if args.by_type and tab["set_piece_type"].nunique() > 1:
        if not args.quick and path.resolve() != (_OUT.parent / _TWO_POP_TABLE).resolve():
            raise SystemExit(f"STOP: --by-type on {path} may not write the published "
                             f"validation.parquet, which only the grant's {_TWO_POP_TABLE} "
                             f"writes; pass --quick")
        # The two populations are validated SEPARATELY and only pooled if the
        # pooled model does not hurt either. Their rates differ, so a pooled rate is a
        # mix weight rather than a rate.
        for kind, sub in tab.groupby("set_piece_type"):
            print(f"\n{'-' * 78}\nPOPULATION: {kind}\n{'-' * 78}")
            blocks.append(report(sub, args.boot, args.quick, not args.no_tune, kind, args.jobs))
        print(f"\n{'-' * 78}\nPOPULATION: pooled (contrast only, never the headline)\n{'-' * 78}")
        blocks.append(report(tab, args.boot, args.quick, not args.no_tune, "pooled", args.jobs))
        blocks.append(pooling_contrast(tab, args.boot, not args.no_tune, args.jobs))
    else:
        kinds = tab["set_piece_type"].unique()
        population = kinds[0] if len(kinds) == 1 else "pooled"
        tag = "" if population in ("corner", "pooled") else f"_{population}"
        if not args.quick and (not tag or published_tag(path) != tag):
            raise SystemExit(f"STOP: {path.name} without --by-type may not write the published "
                             f"validation{tag}.parquet; pass --by-type on the two-population table, "
                             f"or --quick")
        blocks.append(report(tab, args.boot, args.quick, not args.no_tune, population, args.jobs))

    # A --quick run has no ablation block and an untuned one is another estimator, so each goes to
    # its own file rather than overwriting the grant's record (same guard --limit gets elsewhere).
    out = Path(args.out_dir) / (f"validation{'_quick' if args.quick else ''}"
                                f"{'_untuned' if args.no_tune else ''}{tag}.parquet")
    out.parent.mkdir(parents=True, exist_ok=True)
    pd.concat(blocks, ignore_index=True).to_parquet(out, index=False)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
