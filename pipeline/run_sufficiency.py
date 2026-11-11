"""run_sufficiency — is the delivery freeze-frame a sufficient statistic for who recovers?

The question the submitted contract promised (hudl/form_answers.md Q3) and argument.md §11
committed to reporting. It splits into two measurements with very different power, and
conflating them is how a bounded comparison gets read as an answer.

GRANT — where the question is actually answered. Every corner carrying a delivery
freeze-frame and a second ball the event chain resolved: the whole Big-5 grant, five
leagues, no tracking anywhere in it. The freeze-frame is put against the baseline that
matters, the delivery-location rung (architecture.md §Stages), never ln(#classes) and
never the class prior alone. Three arms read the same frame: the handcrafted s02 geometry
that s03 already models, a nested Optuna-tuned LightGBM on that same matrix, and a STATIC
RACE that fuses raw freeze-frame positions through the soft-argmin s04_race publishes. The
verdict is read on the three steps against the delivery rung with every arm rebuilt inside
each match resample, in the normal form: the point plus or minus half the refit width
(`decisive_steps`; s03_value.step_refit_interval says why that form and not the pivotal one).
The record keeps the pivotal bounds in `lo`/`hi` and the percentiles in `step_lo`/`step_hi`,
and `covers_zero` reads the normal form.

OVERLAP — what tracking buys on top, and the half the grant cannot answer. The corners that
also carry Broadcast X&Y, which is 30 matches and does not grow. Reported with the size of
difference the design could have resolved, because a null without that is a statement about
the sample rather than about football.

THE LADDER IS THE REPO'S OWN. s03's design matrix, fold preparation, baseline ladder,
ablations, leave-one-league-out and shuffle controls are reused rather than restated, so the
rungs here are directly comparable to the ones the value model publishes and inherit its
leakage guards. What is added is the static-race rung, which is not a feature matrix at all.

A smoke run reads two subsets: --limit N cuts the grant half to the N-match tables it built, and
--overlap-tables names the folder whose race and grant the overlap is built from, the tracked
sample's, since the first N matches of the grant hold no tracked one.

    python pipeline/run_sufficiency.py --jobs 20
    python pipeline/run_sufficiency.py --limit 200 --overlap-tables data/derived/smoke/tracked \\
        --boot 200 --no-power --refit-boot 3 --out data/derived/smoke/sufficiency.parquet
    python pipeline/run_sufficiency.py --jobs 20 --kinds free_kick_flank   # the <kind> tables
"""

from __future__ import annotations

import argparse
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.exceptions import ConvergenceWarning

_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO / "src"))
sys.path.insert(0, str(_REPO / "pipeline"))

import loader                                                          # noqa: E402
import loader_sb as L                                                  # noqa: E402
from build_table import kind_tag, published_dir                        # noqa: E402
from constants import DROP_ANCHOR_SB, MIN_CELL                         # noqa: E402
from s01_phases import SetPieceGrammar                                 # noqa: E402
from s03_value import CompetingOutcomesModel, CVResult, _KICK_INTRINSIC   # noqa: E402
from s04_race import (SecondBallRace, TAU_GRID, calibrate_tau,         # noqa: E402
                      find_corners, _softmax_p_att)
from s04b_snapshot import (ARMS, fold_table, mde, paired_delta,        # noqa: E402
                           platt_oof, power_curve, prior_oof, row_logloss, score_arm,
                           shuffled_control, snapshot_record, tuned_gbt_oof)
from validate import (cluster_bootstrap_mean, refit_cluster_bootstrap,   # noqa: E402
                      seed_stability, wilcoxon_paired)

_DER = _REPO / "data" / "derived"
# The ladder, read as "a minus b": a positive delta says the first arm loses. Each step
# isolates one ingredient. snapshot->kick is what the 360 does not see, kick->anchored adds
# velocity, the measured lag and the kick-to-contact delay, anchored->race is knowing the
# drop. The endpoints are kept so the whole gap is reported beside its parts.
OVERLAP_STEPS = (("snapshot", "kick"), ("kick", "anchored"), ("anchored", "race"),
                 ("snapshot", "race"), ("prior", "race"), ("prior", "snapshot"))
POWER_GRID = (0.0, 0.02, 0.05, 0.08, 0.12, 0.16, 0.20, 0.30)
# The three steps `freeze_frame_beats_the_delivery_baseline` is decided on, in the order the refit
# block returns them.
DECISIVE_STEPS = ("static_race-delivery", "geometry360-delivery", "gbt360-delivery")
# The recipe's precision, and the only one the published record is written at.
_BOOT, _TRIALS, _SEEDS, _REFIT_BOOT = 2000, 60, 5, 300


# ── The freeze-frame arm, on whatever population it is handed ────────────
def snapshot_match(mid: int, want: set, kinds: tuple = ("corner",)) -> dict:
    """One match's static-race records, keyed by corner id.

    Returns arrival times rather than probabilities: the fusion is a closed form over them,
    so every temperature the calibration sweeps is free once the 360 has been read."""
    ev = L.load_events(mid)
    fz = L.load_360(mid)
    corners = SetPieceGrammar(ev, kinds=kinds).deliveries()
    out = {}
    for _, c in corners[corners["id"].isin(want)].iterrows():
        rec = snapshot_record(c, fz)
        if rec is not None:
            out[c["id"]] = {"tti": rec["tti"], "is_att": rec["is_att"],
                            "n_bodies": rec["n_bodies"]}
    return out


def build_snapshot(pairs: list, want_by_match: dict, jobs: int,
                   kinds: tuple = ("corner",)) -> dict:
    """The static-race records over many matches. Events are the memory-bound stage here,
    not tracking, so this fans out wide where the overlap half cannot."""
    t0 = time.time()
    if jobs == 1:
        chunks = [snapshot_match(m, want_by_match[m], kinds) for m in pairs]
    else:
        chunks = Parallel(n_jobs=jobs, verbose=5)(
            delayed(snapshot_match)(m, want_by_match[m], kinds) for m in pairs)
    out = {cid: rec for ch in chunks for cid, rec in ch.items()}
    print(f"\nfreeze-frame arm built for {len(out)} corners over {len(pairs)} matches "
          f"in {time.time() - t0:.0f}s ({jobs} worker(s))")
    return out


def fuse(records: dict, ids: pd.Series, tau: float) -> np.ndarray:
    """P(att recovers) per corner at one temperature, aligned to `ids`."""
    return np.array([_softmax_p_att(records[i]["tti"], records[i]["is_att"], tau)
                     if i in records else np.nan for i in ids])


def calibrated_tau(records: dict, table: pd.DataFrame) -> dict:
    """Pick the fusion temperature out of fold, grouped by match, against THIS target.

    s04_race's own calibrator, handed the freeze-frame records instead of the tracked ones.
    Selecting tau on a fold's train corners and scoring it on the held-out ones is what keeps
    the rung honest; borrowing the temperature the tracked race adopted would tune the arm on
    a different label and a different substrate."""
    recs = [{"tti": records[r["set_piece_id"]]["tti"],
             "is_att": records[r["set_piece_id"]]["is_att"],
             "y": bool(r["won"]), "match_id": r["match_id"]}
            for _, r in table.iterrows() if r["set_piece_id"] in records]
    return calibrate_tau(recs, taus=TAU_GRID)


# ── The grant half ───────────────────────────────────────────────────────
def grant_table(limit: int | None, tag: str = "", der: Path = _DER) -> pd.DataFrame:
    """One row per corner carrying a delivery freeze-frame AND a resolved second ball.

    Built from the two published artefacts rather than re-derived, so the population is
    exactly the modelled corner table intersected with s08's outcome. `target` is the binary
    second ball in the two-class shape s03's head consumes. `der` is where the two tables live;
    with `limit` they are the N-match ones a smoke run built, never the grant's cut down."""
    tail = f"_limit{limit}" if limit else ""
    cor = pd.read_parquet(der / f"corners{tag}{tail}.parquet")
    sit = pd.read_parquet(der / f"situation{tag}{tail}.parquet")
    sit = sit.loc[sit["won_second_ball"].notna(), ["set_piece_id", "won_second_ball"]]
    t = cor.merge(sit, on="set_piece_id", how="inner")
    t["won"] = t["won_second_ball"].astype(bool)
    t["target"] = np.where(t["won"], "won", "lost")
    return t.reset_index(drop=True)


def _won_proba(cv: CVResult) -> np.ndarray:
    return cv.oof_proba[:, list(cv.classes).index("won")]


def logistic_rows(frame: pd.DataFrame, m: CompetingOutcomesModel) -> tuple[np.ndarray, np.ndarray]:
    """Per-row log-loss of the delivery and geometry360 arms, both refit on `frame` at the C the
    whole-table grid chose. The geometry step is their difference, and run_refit_draws reads it
    from here so its draws are the ones `decisive_steps` scores."""
    y = frame["won"].to_numpy().astype(int)
    return (row_logloss(_won_proba(m.fit_cv(frame, numeric=_KICK_INTRINSIC,
                                            with_delivery_cats=True)), y),
            row_logloss(_won_proba(m.fit_cv(frame, numeric=None, with_delivery_cats=True)), y))


def decisive_steps(frame: pd.DataFrame, m: CompetingOutcomesModel, records: dict,
                   trials: int, seeds: int) -> np.ndarray:
    """The three decisive deltas against the delivery rung, every arm rebuilt on `frame`.

    What the refit bootstrap hands a resampled table to. The two logistic heads keep the C the
    whole-table grid chose, as `s03_value.step_refit_interval` does for the headline step. The
    static race and the strong learner re-select what their published construction selects
    inside its own out-of-fold loop, the fusion temperature and the Optuna search, because that
    selection is part of the arm and holding it would interval a different estimator.

    The strong learner runs single-process here: the draws are what fan out, and nesting a pool
    of outer folds inside a pool of draws would put more processes on the machine than cores."""
    # A loky worker does not inherit main()'s filters, and without them a 300-draw run writes a
    # feature-name warning per LightGBM predict into the log.
    warnings.filterwarnings("ignore")
    warnings.filterwarnings("default", category=ConvergenceWarning)
    y, g = frame["won"].to_numpy().astype(int), frame["match_id"].to_numpy()
    r_delivery, r_geometry = logistic_rows(frame, m)
    tau = calibrated_tau(records, frame)["tau"]
    r_race = row_logloss(platt_oof(fuse(records, frame["set_piece_id"], tau), y, g), y)
    X, _ = m._design(frame)
    p_gbt = np.mean([tuned_gbt_oof(X, y, g, n_trials=trials, jobs=1, seed=sd)[0]
                     for sd in range(seeds)], axis=0)
    r_gbt = row_logloss(p_gbt, y)
    return np.array([(r_race - r_delivery).mean(), (r_geometry - r_delivery).mean(),
                     (r_gbt - r_delivery).mean()])


def grant_rows(table: pd.DataFrame, records: dict, boot: int, jobs: int,
               trials: int, seeds: int, refit_boot: int = 300) -> list:
    """The grant measurement, where the contract's question is answered with power.

    The arms share one population, and the comparison that decides it is against `delivery`.
    architecture.md §Stages fixes that: a claim that the freeze-frame adds information is a
    delta over the rung that already knows where the ball was kicked from and with what
    swing, never over the uniform floor and never over the class prior alone.

    Three of the arms fit their own intercept and are scored raw. `static_race` does not fit
    one at all, so it alone is recalibrated; scoring it raw beside the others would compare a
    misplaced level against three calibrated ones (s04b_snapshot.score_arm).

    The strong learner is what makes a negative result mean something. Without it the finding
    would be that a linear head sees nothing, which is a statement about the head. Its reported
    arm averages the out-of-fold probabilities over seeds, which can only sharpen a proper score
    in expectation, so a learner that still fails the delivery rung fails at its best; the
    per-seed rows publish the spread that averaging removed."""
    yb = table["won"].to_numpy().astype(int)
    rows = [{"block": "population", "key": "corners", "n": len(table), "population": "grant"},
            {"block": "population", "key": "matches", "n": int(table["match_id"].nunique()),
             "population": "grant"},
            {"block": "population", "key": "leagues", "n": int(table["league"].nunique()),
             "population": "grant"},
            {"block": "population", "key": "positive", "n": int(yb.sum()), "population": "grant"},
            {"block": "population", "key": "negative", "n": int(len(yb) - yb.sum()),
             "population": "grant"},
            {"block": "population", "key": "base_rate", "n": len(yb), "population": "grant",
             "point": float(yb.mean())}]

    m = CompetingOutcomesModel(n_splits=5)
    print(f"\nNESTED TUNING, grouped by match ({jobs} workers)...")
    t0 = time.time()
    tune = m.tune_cv(table, n_jobs=jobs)
    m.apply_best_params(tune)
    print(f"  {len(tune.grid)} configs; adopted {tune.best_params}  "
          f"outer log-loss {tune.outer_log_loss:.4f} "
          f"+-{tune.outer_log_loss_std:.4f}  [{time.time() - t0:.0f}s]")
    rows.append({"block": "tuning", "key": "adopted", "population": "grant", "n": tune.n,
                 "point": tune.outer_log_loss, "params": str(tune.best_params)})

    # ── The ladder: the claim is a delta against DELIVERY, never against the floor ──
    print("\nBASELINE LADDER (the rung that carries the claim is `delivery`, never uniform):")
    lad = m.baseline_ladder(table, n_boot=boot)
    for _, r in lad.iterrows():
        step = "" if pd.isna(r["step"]) else \
            f"  step {r['step']:+.4f} [{r['step_lo']:+.4f}, {r['step_hi']:+.4f}]"
        print(f"  {r['rung']:14s} {r['log_loss']:.4f}{step}")
        rows.append({"block": "ladder", "key": str(r["rung"]), "population": "grant",
                     "n": len(yb), "point": float(r["log_loss"]),
                     "lo": r["lo"], "hi": r["hi"], "step": r["step"],
                     "step_lo": r["step_lo"], "step_hi": r["step_hi"]})

    # ── The static race, the rung that is not a feature matrix ───────────
    cal = calibrated_tau(records, table)
    tau = cal["tau"]
    print(f"\nSTATIC RACE from the freeze-frame: tau calibrated OOF by match = {tau:.2f}s "
          f"(OOF log-loss {cal['oof_log_loss']:.4f}; {cal['n']} corners / {cal['n_groups']} "
          f"matches; fold taus {cal['fold_taus']}) -> "
          f"{'adopted' if cal['operative'] else 'NON-OPERATIVE, default kept'}")
    p_snap = fuse(records, table["set_piece_id"], tau)
    rows.append({"block": "tuning", "key": "tau_snapshot", "population": "grant",
                 "n": cal["n"], "point": tau, "lo": cal["oof_log_loss"],
                 "operative": cal["operative"]})

    ok = np.isfinite(p_snap)
    if ok.sum() != len(p_snap):
        print(f"  {int((~ok).sum())} corners carry no freeze-frame record and are dropped "
              f"from the race rung only")
    sub = table[ok].reset_index(drop=True)
    ys, gs = sub["won"].to_numpy().astype(int), sub["match_id"].to_numpy()
    arms = {"prior": score_arm(prior_oof(ys, gs), ys, gs, recalibrate=False),
            "static_race": score_arm(p_snap[ok], ys, gs)}
    for name, kw in (("delivery", dict(numeric=_KICK_INTRINSIC, with_delivery_cats=True)),
                     ("geometry360", dict(numeric=None, with_delivery_cats=True))):
        cv = m.fit_cv(sub, **kw)
        p = cv.oof_proba[:, list(cv.classes).index("won")]
        arms[name] = score_arm(p, ys, gs, recalibrate=False)

    X_full, _ = m._design(sub)
    print(f"\nSTRONG LEARNER: nested Optuna over LightGBM, {trials} trials per outer fold, "
          f"{seeds} seeds ({jobs} workers)...")
    t0, per_seed = time.time(), []
    for sd in range(seeds):
        p_sd, folds_sd = tuned_gbt_oof(X_full, ys, gs, n_trials=trials, jobs=jobs, seed=sd)
        a_sd = score_arm(p_sd, ys, gs, recalibrate=False)
        d_sd = cluster_bootstrap_mean(a_sd["row_raw"] - arms["delivery"]["row_raw"], gs,
                                      n_boot=boot)
        per_seed.append({"seed": sd, "p": p_sd, "ll": a_sd["log_loss_raw"], "auc": a_sd["auc"],
                         "delta": d_sd.point, "lo": d_sd.lo, "hi": d_sd.hi,
                         "covers_zero": bool(d_sd.lo <= 0 <= d_sd.hi),
                         "rounds": [f["rounds"] for f in folds_sd]})
        print(f"  seed {sd}: log-loss {a_sd['log_loss_raw']:.4f}  AUC {a_sd['auc']:.3f}  "
              f"vs delivery {d_sd.point:+.4f} [{d_sd.lo:+.4f}, {d_sd.hi:+.4f}]"
              f"{'' if per_seed[-1]['covers_zero'] else '  EXCLUDES zero'}")
        rows.append({"block": "gbt_seed", "key": str(sd), "population": "grant", "n": len(ys),
                     "point": a_sd["log_loss_raw"], "auc": a_sd["auc"], "lo": d_sd.lo,
                     "hi": d_sd.hi, "covers_zero": per_seed[-1]["covers_zero"],
                     "params": str(folds_sd[0])})
    lls = np.array([r["ll"] for r in per_seed])
    stable = all(r["covers_zero"] for r in per_seed)
    print(f"  {time.time() - t0:.0f}s; log-loss across seeds {lls.mean():.4f} "
          f"[{lls.min():.4f}, {lls.max():.4f}], spread {np.ptp(lls):.4f}; "
          f"verdict {'STABLE' if stable else 'FLIPS'} across seeds")
    rows.append({"block": "stability", "key": "gbt_across_seeds", "population": "grant",
                 "n": seeds, "point": float(lls.mean()), "lo": float(lls.min()),
                 "hi": float(lls.max()), "covers_zero": stable})
    arms["gbt360"] = score_arm(np.mean([r["p"] for r in per_seed], axis=0), ys, gs,
                               recalibrate=False)

    print(f"\nARMS on the grant (n={len(ys)}, {int(ys.sum())} won / {int(len(ys) - ys.sum())} lost):")
    print(f"  {'arm':13s} {'log-loss':>9s} {'AUC':>7s} {'acc':>7s} {'mean p':>8s}")
    for name in ("prior", "delivery", "geometry360", "gbt360", "static_race"):
        a = arms[name]
        key = "log_loss_recal" if name == "static_race" else "log_loss_raw"
        print(f"  {name:13s} {a[key]:9.4f} {a['auc']:7.3f} {a['accuracy']:7.3f} {a['mean_p']:8.3f}")
        rows.append({"block": "arm", "key": name, "population": "grant", "n": len(ys),
                     "point": a[key], "log_loss_raw": a["log_loss_raw"],
                     "calibration_cost": a["calibration_cost"], "auc": a["auc"],
                     "accuracy": a["accuracy"], "mean_p": a["mean_p"]})

    fixed_ci = {}
    print("\nPAIRED DELTAS on the grant (a minus b; positive = a loses):")
    for a, b in (("static_race", "prior"), ("static_race", "delivery"),
                 ("static_race", "geometry360"), ("geometry360", "delivery"),
                 ("gbt360", "delivery"), ("gbt360", "geometry360"), ("gbt360", "prior"),
                 ("delivery", "prior")):
        ka = "row_recal" if a == "static_race" else "row_raw"
        kb = "row_recal" if b == "static_race" else "row_raw"
        d = arms[a][ka] - arms[b][kb]
        ci = cluster_bootstrap_mean(d, gs, n_boot=boot)
        w = wilcoxon_paired(d, gs)
        verdict = "covers zero" if ci.lo <= 0 <= ci.hi else "EXCLUDES zero"
        print(f"  {a:12s} - {b:12s} {ci.point:+.4f} [{ci.lo:+.4f}, {ci.hi:+.4f}]  "
              f"{verdict}   wilcoxon p={w['p_value']:.4f}")
        rows.append({"block": "delta", "key": f"{a}-{b}", "population": "grant", "n": len(ys),
                     "point": ci.point, "lo": ci.lo, "hi": ci.hi,
                     "covers_zero": bool(ci.lo <= 0 <= ci.hi), "p_wilcoxon": w["p_value"]})
        fixed_ci[f"{a}-{b}"] = (ci.point, ci.lo, ci.hi)

    # A resampling bound computed at one seed is a sample of size one from its own noise, so the
    # digits that survive the seed are the digits that may be quoted. Both bounds are probed:
    # whether the step clears zero is decided by the upper one, the one nearest zero here.
    d_head = arms["geometry360"]["row_raw"] - arms["delivery"]["row_raw"]
    for bound in ("lo", "hi"):
        st = seed_stability(lambda sd, b=bound: getattr(
            cluster_bootstrap_mean(d_head, gs, n_boot=boot, seed=sd), b))
        print(f"SEED STABILITY of the headline interval's {bound} bound: spread "
              f"{st['spread']:.2e}, {st['stable_digits']} decimals survive")
        rows.append({"block": "stability", "key": f"headline_ci_{bound}", "population": "grant",
                     "n": len(st["seeds"]), "point": st["mean"], "lo": st["min"], "hi": st["max"],
                     "stable_digits": st["stable_digits"]})

    # ── The decisive steps, with the model rebuilt inside every resample ──
    if refit_boot:
        print(f"\nREFIT BOOTSTRAP of the three decisive steps: {refit_boot} match resamples, "
              f"{jobs} workers, {trials} Optuna trials x {seeds} seeds per strong-learner draw...")
        t0 = time.time()
        rf = refit_cluster_bootstrap(sub, lambda f: decisive_steps(f, m, records, trials, seeds),
                                     n_boot=refit_boot, seed=0, n_jobs=jobs)
        point = np.atleast_1d(rf.point)
        fixed = np.array([fixed_ci[k][0] for k in DECISIVE_STEPS])
        # The rebuilt arms must land on the published deltas, or the interval belongs to a
        # different estimator from the point it is printed beside.
        if not np.allclose(point, fixed, rtol=0.0, atol=1e-9):
            raise SystemExit(f"STOP: the refit point {point} does not reproduce the published "
                             f"deltas {fixed}")
        lo, hi = np.atleast_1d(rf.lo), np.atleast_1d(rf.hi)
        print(f"  {time.time() - t0:.0f}s; {rf.n_boot} finite draws; point reproduces the deltas")
        for i, key in enumerate(DECISIVE_STEPS):
            adj_lo, adj_hi = 2 * point[i] - hi[i], 2 * point[i] - lo[i]
            half = (hi[i] - lo[i]) / 2
            width = fixed_ci[key][2] - fixed_ci[key][1]
            ratio = (hi[i] - lo[i]) / width if width else float("nan")
            print(f"  {key:22s} {point[i]:+.5f} normal [{point[i] - half:+.5f}, "
                  f"{point[i] + half:+.5f}] pivotal [{adj_lo:+.5f}, {adj_hi:+.5f}] "
                  f"percentile [{lo[i]:+.5f}, {hi[i]:+.5f}] width x{ratio:.2f}")
            rows.append({"block": "refit", "key": key, "population": "grant", "n": rf.n_boot,
                         "point": point[i], "lo": adj_lo, "hi": adj_hi,
                         "step_lo": lo[i], "step_hi": hi[i],
                         "covers_zero": bool(point[i] - half <= 0 <= point[i] + half),
                         "width_ratio": ratio})

    # ── Ablations, external validity, controls ───────────────────────────
    abl = m.ablate(table, n_boot=boot)
    print(f"\nABLATIONS on the {int((abl['group'] != 'ALL').sum())} droppable blocks "
          f"(delta log-loss when dropped):")
    for _, r in abl.iterrows():
        if r["group"] == "ALL":
            continue
        covers = bool(r["lo"] <= 0 <= r["hi"])
        print(f"  {r['group']:14s} {r['delta']:+.5f} [{r['lo']:+.5f}, {r['hi']:+.5f}] "
              f"{'covers zero' if covers else 'EXCLUDES zero'}  p={r['p_wilcoxon']:.3f}")
        rows.append({"block": "ablation", "key": str(r["group"]), "population": "grant",
                     "n": len(yb), "point": float(r["delta"]), "lo": r["lo"], "hi": r["hi"],
                     "covers_zero": covers, "p_wilcoxon": r["p_wilcoxon"]})

    lo_res = m.fit_group_out(table, group_col="league")
    print(f"\nLEAVE-ONE-LEAGUE-OUT: pooled {lo_res.log_loss:.4f}, fold mean "
          f"{lo_res.log_loss_mean:.4f} +-{lo_res.log_loss_std:.4f} over {lo_res.n_folds} leagues; "
          f"{len(lo_res.degenerate_folds)} degenerate")
    rows.append({"block": "league_out", "key": "__summary__", "population": "grant",
                 "n": lo_res.n, "point": lo_res.log_loss, "lo": lo_res.log_loss_mean,
                 "hi": lo_res.log_loss_std})
    for f in lo_res.folds:
        rows.append({"block": "league_out", "key": str(f["held_out"]), "population": "grant",
                     "n": f["n_test"], "point": f["log_loss"], "accuracy": f["accuracy"]})

    print("\nCONTROLS (a shuffled model that beats its own entropy means the harness leaks):")
    for mode in ("within_match", "global"):
        c = m.shuffled_control(table, mode=mode)
        print(f"  {mode:13s} shuffled {c['log_loss_mean']:.4f} vs entropy "
              f"{c['marginal_entropy']:.4f}  {c['verdict']}")
        rows.append({"block": "control", "key": f"geometry360/{mode}", "population": "grant",
                     "n": c["n_repeats"], "point": c["log_loss_mean"],
                     "entropy": c["marginal_entropy"], "clean": c["verdict"] == "clean"})
    for mode in ("within_match", "global"):
        c = shuffled_control(p_snap[ok], ys, gs, mode=mode)
        print(f"  static_race/{mode:13s} shuffled {c['log_loss_mean']:.4f} vs entropy "
              f"{c['marginal_entropy']:.4f}  {'clean' if c['clean'] else 'LEAKAGE SUSPECTED'}")
        rows.append({"block": "control", "key": f"static_race/{mode}", "population": "grant",
                     "n": c["n_repeats"], "point": c["log_loss_mean"],
                     "entropy": c["marginal_entropy"], "clean": c["clean"]})
    return rows


# ── The overlap half ─────────────────────────────────────────────────────
def overlap_population(tag: str = "", der: Path = _DER) -> tuple[pd.DataFrame, pd.DataFrame]:
    """(primary, sensitivity) corner tables for the tracked comparison, all read from `der`.

    Primary carries the event-chain outcome; sensitivity carries the tracking label and is a
    strict superset, so the tracked arms are computed once over the wider one. The race joins the
    whole grant of its own folder, not the grant arm's table: a smoke's N-match subset holds no
    tracked match."""
    race = pd.read_parquet(der / f"race_corners{tag}.parquet")
    with_360 = set(pd.read_parquet(der / f"corners{tag}.parquet")["set_piece_id"])
    wide = race[race["actual_att"].notna() & race["corner_id"].isin(with_360)].copy()
    grant = grant_table(None, tag, der)
    prim = race.merge(grant[["set_piece_id", "won", "league"]], left_on="corner_id",
                      right_on="set_piece_id", how="inner")
    assert prim["corner_id"].is_unique, "duplicate corner in the race/grant join"
    return prim.reset_index(drop=True), wide.reset_index(drop=True)


def tracked_match(mid: int, want: set, kinds: tuple = ("corner",)) -> dict:
    """One match's tracked rungs: arrival times to the observed drop and to the fixed anchor."""
    ev = L.load_events(mid)
    sel = find_corners(ev, kinds)
    sel = sel[sel["id"].isin(want)]
    if not len(sel):
        return {}
    race = SecondBallRace(loader.load_tracking(mid), ev)
    out: dict = {}
    # Three tracked rungs off one tracking load. `kick` reads the delivery frame with no
    # kinematics, so the 360-to-kick step is what the freeze-frame fails to SEE; `anchored` adds
    # velocity, the measured lag and the kick-to-contact delay; `race` adds the known drop.
    for tag, anchor, kick in (("kick", DROP_ANCHOR_SB, True),
                              ("anchored", DROP_ANCHOR_SB, False), ("race", None, False)):
        for r in race.race_records(sel, mid, anchor_sb=anchor, at_delivery=kick):
            out.setdefault(r["corner_id"], {})[f"tti_{tag}"] = r["tti"]
            out[r["corner_id"]][f"att_{tag}"] = r["is_att"]
    return out


def overlap_rows(pop: pd.DataFrame, target: str, label: str, boot: int,
                 power: bool) -> tuple[dict, list]:
    """Score every rung on one overlap population and return (arms, record rows)."""
    y = pop[target].astype(int).to_numpy()
    g = pop["match_id"].to_numpy()
    probs = {"prior": prior_oof(y, g), "snapshot": pop["p_snapshot"].to_numpy(float),
             "kick": pop["p_kick"].to_numpy(float),
             "anchored": pop["p_anchored"].to_numpy(float), "race": pop["p_race"].to_numpy(float)}
    # the prior stays raw: Platt on a fold-constant arm reads the held-out match's base rate
    arms = {k: score_arm(v, y, g, recalibrate=(k != "prior")) for k, v in probs.items()}

    rows = [{"block": "population", "key": k, "n": v, "population": label} for k, v in
            (("corners", len(pop)), ("matches", int(pop["match_id"].nunique())),
             ("positive", int(y.sum())), ("negative", int(len(y) - y.sum())))]
    rows.append({"block": "population", "key": "base_rate", "n": len(y),
                 "point": float(y.mean()), "population": label})
    print(f"\nARMS on {label} (n={len(y)}, {int(y.sum())} positive / {int(len(y) - y.sum())} negative):")
    print(f"  {'arm':10s} {'ll raw':>8s} {'ll recal':>9s} {'cal cost':>9s} {'AUC':>6s} "
          f"{'acc':>6s} {'mean p':>7s}")
    for k in ARMS:
        a = arms[k]
        print(f"  {k:10s} {a['log_loss_raw']:8.4f} {a['log_loss_recal']:9.4f} "
              f"{a['calibration_cost']:9.4f} {a['auc']:6.3f} {a['accuracy']:6.3f} {a['mean_p']:7.3f}")
        rows.append({"block": "arm", "key": k, "population": label, "n": len(y),
                     "point": a["log_loss_recal"], "log_loss_raw": a["log_loss_raw"],
                     "calibration_cost": a["calibration_cost"], "auc": a["auc"],
                     "accuracy": a["accuracy"], "mean_p": a["mean_p"]})

    print(f"\nPAIRED DELTAS on {label} (a minus b; positive = a loses; recalibrated):")
    for a, b in OVERLAP_STEPS:
        for key_name, series in (("delta", "row_recal"), ("delta_raw", "row_raw")):
            d = paired_delta(arms[a], arms[b], g, key=series, n_boot=boot)
            w = wilcoxon_paired(d["diff"], g)
            rows.append({"block": key_name, "key": f"{a}-{b}", "n": len(y), "population": label,
                         "point": d["delta"], "lo": d["lo"], "hi": d["hi"],
                         "covers_zero": d["covers_zero"], "p_wilcoxon": w["p_value"]})
            if key_name == "delta":
                verdict = "covers zero" if d["covers_zero"] else "EXCLUDES zero"
                print(f"  {a:9s} - {b:9s} {d['delta']:+.4f} [{d['lo']:+.4f}, {d['hi']:+.4f}]"
                      f"  {verdict}   wilcoxon p={w['p_value']:.3f}")

    if power:
        base = arms["snapshot"]["row_recal"] - arms["race"]["row_recal"]
        curve = power_curve(base, g, POWER_GRID)
        m80 = mde(curve)
        reached = (f"reaches 80% power at a true difference of {m80:.3f} nats"
                   if np.isfinite(m80) else "never reaches 80% power on the grid")
        print(f"\nPOWER on {label}: this design {reached}")
        for _, r in curve.iterrows():
            print(f"    delta {r['delta']:.2f} -> power {r['power']:.2f}")
            rows.append({"block": "power_curve", "key": f"{r['delta']:.2f}", "population": label,
                         "n": int(r["n_rep"]), "point": float(r["power"])})
        rows.append({"block": "power", "key": "mde_80", "population": label, "n": len(y),
                     "point": m80})
    return arms, rows


def main() -> None:
    """Answer the contract's question on the grant, then bound it on the tracked overlap."""
    # Never ConvergenceWarning: hiding a half-fitted model is how one publishes a false null.
    warnings.filterwarnings("ignore")
    warnings.filterwarnings("default", category=ConvergenceWarning)
    ap = argparse.ArgumentParser()
    ap.add_argument("--jobs", type=int, default=20,
                    help="workers for the events fan-out, the tuning grid, the strong learner's "
                         "search and the refit draws")
    ap.add_argument("--track-jobs", type=int, default=2,
                    help="workers for the tracking pass (memory-bound: keep at 2-3)")
    ap.add_argument("--boot", type=int, default=_BOOT, help="cluster-bootstrap draws")
    ap.add_argument("--trials", type=int, default=_TRIALS,
                    help="Optuna trials per outer fold for the strong learner")
    ap.add_argument("--seeds", type=int, default=_SEEDS,
                    help="seeds for the strong learner; one is a sample of size one")
    ap.add_argument("--no-power", action="store_true", help="skip the power curve")
    ap.add_argument("--refit-boot", type=int, default=_REFIT_BOOT,
                    help="match resamples for the refit interval on the decisive steps; 0 skips it")
    ap.add_argument("--limit", type=int, default=None,
                    help="first N matches only (smoke; the numbers are not publishable)")
    ap.add_argument("--out", type=str, default=None,
                    help="output parquet (default: data/derived/sufficiency[_kind][_limitN].parquet)")
    ap.add_argument("--kinds", type=str, default="corner",
                    help="the s01 population; its corners, situation and race tables are read")
    ap.add_argument("--overlap-tables", type=str, default=str(_DER),
                    help="where the race, the grant and the corner table of the tracked comparison "
                         "live (default: data/derived)")
    args = ap.parse_args()

    kinds = tuple(k.strip() for k in args.kinds.split(","))
    tag = kind_tag(kinds)
    out = (Path(args.out) if args.out else
           _DER / f"sufficiency{tag}{f'_limit{args.limit}' if args.limit else ''}.parquet")
    overlap_der = Path(args.overlap_tables)
    published = published_dir(out.parent) and out.name == f"sufficiency{tag}.parquet"
    precision = (args.boot, args.trials, args.seeds, args.refit_boot, not args.no_power)
    if published and precision != (_BOOT, _TRIALS, _SEEDS, _REFIT_BOOT, True):
        raise SystemExit(f"STOP: the verdict is decided on the refit block at {_REFIT_BOOT} draws, "
                         f"with {_TRIALS} trials x {_SEEDS} seeds and the power curve, so a run at "
                         f"(boot, trials, seeds, refit-boot, power) = {precision} may not write the "
                         f"published {out.name}; pass --out")
    if published and overlap_der.resolve() != _DER.resolve():
        raise SystemExit(f"STOP: overlap tables outside data/derived are not the published ones, so "
                         f"their comparison may not write the published {out.name}; pass --out")
    grant = grant_table(args.limit, tag)
    print(f"SUFFICIENCY — grant {len(grant)} corners / {grant['match_id'].nunique()} matches "
          f"/ {grant['league'].nunique()} leagues, base rate {grant['won'].mean():.4f}")
    if args.limit:
        print(f"  LIMIT {args.limit}: the path runs end to end; the numbers are not publishable")

    # The freeze-frame arm is built over the UNION of both halves. The sensitivity population
    # is tracked corners whose second ball the event chain never resolved, so it is not a
    # subset of the grant; building only the grant's records leaves those corners without an
    # arm and silently shrinks the sensitivity back onto the primary.
    prim, wide = overlap_population(tag, overlap_der)
    overlap_ids = pd.concat([prim, wide])[["corner_id", "match_id"]]
    need = pd.concat([grant[["set_piece_id", "match_id"]],
                      overlap_ids.rename(columns={"corner_id": "set_piece_id"})])
    need = need.drop_duplicates("set_piece_id")
    mids = sorted(need["match_id"].unique())
    want = {int(m): set(need.loc[need["match_id"] == m, "set_piece_id"]) for m in mids}
    records = build_snapshot(mids, want, args.jobs, kinds)

    rows = grant_rows(grant, records, args.boot, args.jobs, args.trials, args.seeds,
                      args.refit_boot)

    # ── The tracked overlap, and what it can and cannot resolve ──────────
    print(f"\nOVERLAP — {len(prim)} corners / {prim['match_id'].nunique()} matches carry "
          f"tracking too; sensitivity {len(wide)} / {wide['match_id'].nunique()}")
    if len(prim) < MIN_CELL:
        print(f"  under the n={MIN_CELL} publication floor; the tracked comparison is not reported")
    else:
        tw = {int(m): set(wide.loc[wide["match_id"] == m, "corner_id"])
              for m in sorted(wide["match_id"].unique())}
        t0 = time.time()
        chunks = Parallel(n_jobs=args.track_jobs, verbose=5)(
            delayed(tracked_match)(m, tw[m], kinds) for m in sorted(tw))
        tracked = {cid: r for ch in chunks for cid, r in ch.items()}
        print(f"\ntracked rungs built for {len(tracked)} corners in {time.time() - t0:.0f}s "
              f"({args.track_jobs} worker(s))")

        tau = float(prim["tau_s"].iloc[0])
        for frame, idcol in ((prim, "corner_id"), (wide, "corner_id")):
            for rung in ("kick", "anchored", "race"):
                frame[f"p_{rung}"] = [
                    _softmax_p_att(tracked[i][f"tti_{rung}"], tracked[i][f"att_{rung}"], tau)
                    if i in tracked and f"tti_{rung}" in tracked[i] else np.nan
                    for i in frame[idcol]]
            frame["p_snapshot"] = fuse(records, frame[idcol], tau)
        # The rebuilt race rung has to land on the published p_att to the last bit, or either
        # this driver fuses differently or s04_race moved underneath race_corners.parquet.
        gap = float((wide["p_race"] - wide["p_att"]).abs().max())
        print(f"  rebuilt race rung against the published p_att: max |diff| = {gap:.3e}")
        if gap > 1e-9:
            raise SystemExit("STOP: the rebuilt race rung does not reproduce race_corners.parquet")

        keep = ["p_snapshot", "p_kick", "p_anchored", "p_race"]
        n0 = len(prim), len(wide)
        prim, wide = prim.dropna(subset=keep), wide.dropna(subset=keep)
        if (len(prim), len(wide)) != n0:
            print(f"  dropped for a missing arm: primary {n0[0] - len(prim)}, "
                  f"sensitivity {n0[1] - len(wide)}")

        agree = float((prim["actual_att"].astype(bool) == prim["won"].astype(bool)).mean())
        print(f"\nLABEL AGREEMENT: the tracking and event-chain labels for the same corner "
              f"agree on {100 * agree:.1f}% of {len(prim)}")
        rows.append({"block": "population", "key": "label_agreement", "population": "overlap",
                     "n": len(prim), "point": agree})

        arms, r = overlap_rows(prim, "won", "overlap", args.boot, not args.no_power)
        rows += r
        y, g = prim["won"].astype(int).to_numpy(), prim["match_id"].to_numpy()
        # The prior is scored raw for a reason, and this is the evidence rather than the claim:
        # recalibrating it beats shuffled labels, which only a transform reading the held-out
        # match's base rate can do. Kept as a standing probe so the exclusion cannot quietly be
        # undone.
        probe = shuffled_control(arms["prior"]["p_raw"], y, g, mode="within_match")
        print(f"\n  prior probe (why the prior is scored raw): shuffled "
              f"{probe['log_loss_mean']:.4f} vs entropy {probe['marginal_entropy']:.4f}  "
              f"{'clean' if probe['clean'] else 'LEAKS'}")
        rows.append({"block": "prior_probe", "key": "within_match", "population": "overlap",
                     "n": probe["n_repeats"], "point": probe["log_loss_mean"],
                     "entropy": probe["marginal_entropy"], "clean": probe["clean"]})

        stab = seed_stability(lambda s: paired_delta(arms["snapshot"], arms["race"], g,
                                                     n_boot=args.boot, seed=s)["delta"])
        print(f"  seed stability of the headline delta: spread {stab['spread']:.2e}, "
              f"{stab['stable_digits']} decimals survive")
        rows.append({"block": "stability", "key": "headline_delta", "population": "overlap",
                     "n": len(stab["seeds"]), "point": stab["mean"], "lo": stab["min"],
                     "hi": stab["max"], "stable_digits": stab["stable_digits"]})

        folds = fold_table(y, g)
        print(f"  {len(folds)} leave-one-match-out folds; "
              f"{int((folds['test_neg'] == 0).sum())} carry no negative in test")
        for _, f in folds.iterrows():
            rows.append({"block": "fold", "key": str(f["held_out"]), "population": "overlap",
                         "n": int(f["n_test"]), "test_pos": int(f["test_pos"]),
                         "test_neg": int(f["test_neg"]), "n_train": int(f["n_train"])})

        _, r = overlap_rows(wide, "actual_att", "sensitivity", args.boot, False)
        rows += r
        print("\n  SENSITIVITY CAVEAT: there the race arm is scored against a label read from")
        print("  the same tracking a second later, so it is correlated with its own outcome by")
        print("  construction. It is not the result and never replaces the primary.")

    out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_parquet(out, index=False)
    print(f"\nwrote {out}  [gitignored]")


if __name__ == "__main__":
    main()
