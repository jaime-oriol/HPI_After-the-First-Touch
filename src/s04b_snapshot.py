"""s04b_snapshot — the freeze-frame arm, and the machinery that scores it against the race.

The submitted contract (hudl/form_answers.md Q3) promised a test of whether the freeze-frame is
a sufficient statistic for who recovers the second ball; this module reads it at the delivery
(argument.md §11 gives that reading). s04_race answers "who gets there" from continuous
tracking; this module answers the same question from the 360 snapshot at the kick, and both
are scored against the outcome s08 derives from the EVENT chain, so neither is graded by a
label read off its own substrate.

THE STATIC RACE IS THE ARM. Freeze-frame positions go through the soft-argmin over arrival
times s04_race already publishes, at the same temperature and the same travel cap, so the two
differ in what feeds tti and in nothing else. There are no velocities in a snapshot, so there
is no reaction lag and no momentum drift, and that absence is the measurement.

THE ANCHOR IS WHAT KEEPS IT LEGAL. Everyone races to DROP_ANCHOR_SB, the global median
landing point constants.py measured, never to this corner's own drop and never to
pass.end_location. Both of those are read after the delivery and definitions.md §16 bars
them from X_t; one global point carries nothing about how any single corner resolved.

CALIBRATION IS MEASURED, NOT ASSUMED AWAY. An arm built without a fitted intercept carries
whatever level its construction gave it, so scoring it raw reads a misplaced intercept as
missing information. Those arms are scored twice, raw and after a two-parameter Platt fitted
on the train side of each fold only. Arms that fit their own intercept are scored raw, and an
arm that is CONSTANT inside a fold must never be recalibrated at all (score_arm).

THE STRONG LEARNER IS WHY A NULL HERE MEANS ANYTHING. A negative result from a linear head
alone is a statement about linear models. `tuned_gbt_oof` puts the same design matrix through
a nested, match-grouped, Optuna-tuned LightGBM, so a frame that beats nothing beats nothing at
the strongest family tabular data of this size admits.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold

from constants import DROP_ANCHOR_SB, PITCH_LENGTH_M, PITCH_WIDTH_M, PITCH_X_SB, PITCH_Y_SB
from pitch_control import default_model_params
from s02_signature import build_frame
from validate import (assert_no_group_leak, cluster_bootstrap_mean,
                      leave_one_group_out, shuffle_labels)

if TYPE_CHECKING:
    import optuna
    from lightgbm import LGBMClassifier

ARMS = ("prior", "snapshot", "kick", "anchored", "race")


# StatsBomb logs on a normalised 120x80 grid, so one unit is not one metre and the two axes
# do not even share a factor. Derived from the pitch constants so it cannot drift off them.
_SB_TO_M = np.array([PITCH_LENGTH_M / PITCH_X_SB, PITCH_WIDTH_M / PITCH_Y_SB])


def snapshot_record(corner: pd.Series, three_sixty: pd.DataFrame,
                    anchor_sb: tuple = DROP_ANCHOR_SB,
                    vmax: float | None = None) -> dict | None:
    """One corner's static race, from the delivery freeze-frame alone.

    Same record shape s04_race.race_records emits, so both arms feed the identical
    soft-argmin: arrival times in seconds and the attacker mask. None when the frame holds
    fewer than two bodies.

    `vmax` defaults to the travel cap the tracking race uses, read from the same
    pitch_control parameters rather than restated. The two rungs are compared at one shared
    temperature, so their arrival times have to be in the same units for that temperature to
    mean the same thing in both; dividing one of them by a different speed rescales its whole
    tti vector and silently makes its soft-argmin softer. constants.MAX_PLAYER_SPEED_MS is
    NOT the number for this: it is the detection-gap cap kinematics clips at, not a speed
    anybody runs.

    Every detected body races, keepers and the taker included, because s04_race._arrivals
    races every player in the tracking frame too. Matching the population is what leaves
    kinematics and anchor as the only differences between the rungs; excluding a side here
    would put a third difference into a two-difference comparison.

    No velocities exist in a freeze-frame, so there is no reaction lag and no momentum
    drift. That absence is the measurement, not a shortcoming of the implementation."""
    frame, _ = build_frame(corner, three_sixty)
    bodies = frame[frame["is_ball"] == 0]
    if len(bodies) < 2:
        return None
    v = default_model_params()["max_player_speed"] if vmax is None else vmax
    xy = bodies[["x", "y"]].to_numpy(float)
    d = np.linalg.norm((xy - np.asarray(anchor_sb, dtype=float)) * _SB_TO_M, axis=1)
    return {"tti": d / v, "is_att": (bodies["team_id"].to_numpy() == 1),
            "corner_id": corner["id"], "n_bodies": int(len(bodies))}


def _clip(p: np.ndarray) -> np.ndarray:
    return np.clip(np.asarray(p, dtype=float), 1e-6, 1 - 1e-6)


def row_logloss(p: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Per-row log-loss, so two arms can be differenced row by row rather than in the mean."""
    q = _clip(p)
    return -(y * np.log(q) + (1 - y) * np.log(1 - q))


def _logit(p: np.ndarray) -> np.ndarray:
    q = _clip(p)
    return np.log(q / (1 - q)).reshape(-1, 1)


def prior_oof(y: np.ndarray, groups: np.ndarray) -> np.ndarray:
    """The class prior refit per fold: the floor every arm has to clear to be information.

    Leave-one-match-out for the prior, Platt and the fold table; the GBT arm is 5-fold outside
    and 3-fold inside (tuned_gbt_oof). On the tracked overlap leave-one-match-out is forced: a
    five-fold test block there is thin enough to lose a class outright. `leave_one_group_out`
    iterates in sorted order, so the fold table printed beside the result reproduces."""
    out = np.zeros(len(y), dtype=float)
    for _, tr, te in leave_one_group_out(groups):
        assert_no_group_leak(groups, tr, te, "match_id")
        out[te] = float(y[tr].mean())
    return out


def platt_oof(p: np.ndarray, y: np.ndarray, groups: np.ndarray) -> np.ndarray:
    """Leave-one-match-out Platt recalibration of one arm's probabilities.

    Two parameters on the logit, fitted on the TRAIN matches of each fold and applied to the
    held-out match only. This deliberately spends the training base rate: that is the point,
    since it removes a nuisance intercept none of the arms was ever given a chance to learn,
    and leaves the comparison on discrimination and slope.

    A train fold with one class left gets that fold's base rate rather than a fitted curve,
    which is the same refusal-to-invent calibrate_tau makes when its guard fails.

    Apply it only to an arm that VARIES inside a fold. On one that does not, the transform can
    fit nothing but the between-fold variation, and under this splitter that variation is the
    held-out match's own base rate (score_arm)."""
    out = np.zeros(len(y), dtype=float)
    for _, tr, te in leave_one_group_out(groups):
        assert_no_group_leak(groups, tr, te, "match_id")
        if len(np.unique(y[tr])) < 2:
            out[te] = float(y[tr].mean())
            continue
        lr = LogisticRegression(C=1e6, max_iter=5000).fit(_logit(p[tr]), y[tr])
        out[te] = lr.predict_proba(_logit(p[te]))[:, 1]
    return out


def fold_table(y: np.ndarray, groups: np.ndarray) -> pd.DataFrame:
    """Per-fold class counts, published rather than summarised.

    A fold with no negative in its test block makes AUC undefined and log-loss unstable, so
    the reader gets the counts instead of an assurance that they were checked."""
    rows = []
    for held, tr, te in leave_one_group_out(groups):
        rows.append({"held_out": held, "n_train": len(tr), "n_test": len(te),
                     "train_pos": int(y[tr].sum()), "train_neg": int(len(tr) - y[tr].sum()),
                     "test_pos": int(y[te].sum()), "test_neg": int(len(te) - y[te].sum())})
    return pd.DataFrame(rows)


def score_arm(p: np.ndarray, y: np.ndarray, groups: np.ndarray,
              recalibrate: bool = True) -> dict:
    """One arm, scored raw and recalibrated on the same folds.

    `calibration_cost` is the gap between the two log-losses. It is what a Murphy-style
    split would call the miscalibration term, obtained here by refitting rather than by
    binning, and it is the number that says whether a raw loss above the prior means the arm
    is uninformative or merely mis-centred.

    `recalibrate=False` is REQUIRED for any arm that is constant inside a fold, and the prior
    is. Platt has no ordering to preserve there, so it can only fit the variation BETWEEN
    folds, and under leave-one-match-out that variation IS the held-out match's own base rate.
    Recalibrating the prior therefore inverts an anti-correlation the fold construction
    planted and beats every real arm with it. The `prior_probe` block of the artefact carries
    the measurement, and its shuffle control is what catches a regression here.

    Pass it False for any arm that already fits its own intercept, too: a fitted head is
    calibrated to its training base rate by construction and has nothing left for Platt."""
    rec = platt_oof(p, y, groups) if recalibrate else np.asarray(p, dtype=float)
    ll_raw, ll_cal = row_logloss(p, y), row_logloss(rec, y)
    return {"p_raw": p, "p_recal": rec, "row_raw": ll_raw, "row_recal": ll_cal,
            "log_loss_raw": float(ll_raw.mean()), "log_loss_recal": float(ll_cal.mean()),
            "calibration_cost": float(ll_raw.mean() - ll_cal.mean()),
            "auc": float(roc_auc_score(y, p)) if len(np.unique(y)) > 1 else float("nan"),
            "accuracy": float(((p >= 0.5).astype(int) == y).mean()),
            "mean_p": float(np.mean(p))}


def paired_delta(a: dict, b: dict, groups: np.ndarray, key: str = "row_recal",
                 n_boot: int = 5000, seed: int = 0) -> dict:
    """Interval on the PAIRED per-row log-loss difference between two arms (a minus b).

    Both arms are scored on the same rows through the same folds, so the difference is a
    per-row quantity and its interval is a bootstrap of that quantity. Differencing two
    marginal intervals is barred across this repo and would be wrong here for the usual
    reason: the two arms are strongly correlated, so their marginal spreads say nothing
    about the spread of the gap."""
    d = a[key] - b[key]
    ci = cluster_bootstrap_mean(d, groups, n_boot=n_boot, seed=seed)
    return {"delta": ci.point, "lo": ci.lo, "hi": ci.hi, "se": ci.se,
            "n_groups": ci.n_groups, "covers_zero": bool(ci.lo <= 0 <= ci.hi), "diff": d}


def replicate_index(picks: np.ndarray, idx_of: dict) -> tuple[np.ndarray, np.ndarray]:
    """(row indices, cluster ids) for one cluster resample drawn with replacement.

    A match drawn twice contributes TWO clusters to that replicate and gets two fresh ids.
    Carrying its original label instead merges the copies into a single cluster of double the
    rows, so the replicate ends up with fewer and fatter clusters than the design has: an
    inner interval too wide, a rejection rate under alpha, and a reported MDE too large.
    Kept separate from power_curve because it is the part a test can pin exactly."""
    rows = np.concatenate([idx_of[g] for g in picks])
    ids = np.concatenate([np.full(len(idx_of[g]), k) for k, g in enumerate(picks)])
    return rows, ids


def power_curve(diff: np.ndarray, groups: np.ndarray, deltas: tuple,
                n_boot: int = 400, n_rep: int = 300, seed: int = 0) -> pd.DataFrame:
    """How large a true difference this design could actually have seen.

    The observed per-row difference is recentred on each candidate delta and resampled by
    match, so the noise driving the answer is the one this comparison really has rather than
    one assumed for it. For every candidate the routine asks how often a clustered interval
    would exclude zero, which is the definition of power at that effect size.

    It travels with the result because argument.md §5 publishes a null only with its reading
    attached and §11 asks for the size the design could have resolved: a null whose design
    could not have resolved the effect either way is a statement about the sample rather than
    about football. Which sample is the caller's: the tracked overlap on the
    sufficiency ladder, the whole grant on the corner step `run_validation` powers."""
    centred = np.asarray(diff, dtype=float) - float(np.mean(diff))
    rng = np.random.default_rng(seed)
    uniq = np.unique(groups)
    idx_of = {g: np.flatnonzero(groups == g) for g in uniq}
    rows = []
    for delta in deltas:
        hits = 0
        for _ in range(n_rep):
            picks = rng.choice(uniq, len(uniq), replace=True)
            draw, relabelled = replicate_index(picks, idx_of)
            ci = cluster_bootstrap_mean(centred[draw] + delta, relabelled, n_boot=n_boot,
                                        seed=int(rng.integers(1 << 30)))
            hits += int(not (ci.lo <= 0 <= ci.hi))
        rows.append({"delta": float(delta), "power": hits / n_rep, "n_rep": n_rep})
    return pd.DataFrame(rows)


def mde(curve: pd.DataFrame, target: float = 0.80) -> float:
    """Smallest candidate delta on the curve reaching `target` power (NaN if none does)."""
    ok = curve[curve["power"] >= target]
    return float(ok["delta"].min()) if len(ok) else float("nan")


def shuffled_control(p: np.ndarray, y: np.ndarray, groups: np.ndarray, mode: str = "within_match",
                     n_repeats: int = 20, seed: int = 0) -> dict:
    """Recalibrate an arm against permuted labels; it must land at the marginal entropy.

    `within_match` is the mode that matters here. A global permutation also destroys the
    within-match correlation of the target, so it cannot see a leak travelling through match
    identity, and match identity is the channel a grouped design is most exposed to. The
    recalibration step is what is on trial: it is the only place in this module where
    anything is fitted against the target at all."""
    q = pd.Series(y).value_counts(normalize=True).to_numpy()
    entropy = float(-(q * np.log(q)).sum())
    losses = []
    for r in range(n_repeats):
        ys = shuffle_labels(y, groups, mode=mode, seed=seed + r)
        losses.append(float(row_logloss(platt_oof(p, ys, groups), ys).mean()))
    mean = float(np.mean(losses))
    return {"mode": mode, "n_repeats": n_repeats, "log_loss_mean": mean,
            "log_loss_std": float(np.std(losses)), "marginal_entropy": entropy,
            "clean": bool(mean >= entropy - 5e-3)}


# ── The strong learner, so a null cannot be an artefact of the model class ──
def _gbt_space(trial: "optuna.Trial", n_pos: int) -> dict:
    """LightGBM search space, sized for the sufficiency grant.

    Leaves and minimum leaf occupancy are the two that decide over-fitting at this n, so both
    are searched wide and `min_child_samples` is floored well above the default: a leaf holding
    a handful of the minority class is memorising a match, not learning a pattern. Its ceiling
    is derived from `n_pos` rather than from a count written here, so it follows whatever
    population the arm is handed; the class split itself is generated, in `measured.md`
    §sufficiency. Depth is capped rather than left free because unlimited depth with few leaves
    produces long thin trees that early stopping cannot see through."""
    return {
        "learning_rate": trial.suggest_float("learning_rate", 0.005, 0.2, log=True),
        "num_leaves": trial.suggest_int("num_leaves", 4, 96, log=True),
        "max_depth": trial.suggest_int("max_depth", 2, 8),
        "min_child_samples": trial.suggest_int("min_child_samples", 20, max(40, n_pos // 20)),
        "feature_fraction": trial.suggest_float("feature_fraction", 0.4, 1.0),
        "bagging_fraction": trial.suggest_float("bagging_fraction", 0.5, 1.0),
        "bagging_freq": trial.suggest_int("bagging_freq", 0, 5),
        "lambda_l1": trial.suggest_float("lambda_l1", 1e-4, 10.0, log=True),
        "lambda_l2": trial.suggest_float("lambda_l2", 1e-4, 10.0, log=True),
    }


def _gbt_outer_fold(X: np.ndarray, y: np.ndarray, groups: np.ndarray, tr: np.ndarray,
                    te: np.ndarray, n_trials: int, seed: int, k: int) -> tuple:
    """Search, refit and score ONE outer fold. Module-level so joblib can ship it."""
    import optuna
    from lightgbm import LGBMClassifier, early_stopping, log_evaluation
    from sklearn.metrics import log_loss
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    inner = GroupKFold(n_splits=min(3, len(np.unique(groups[tr]))))
    splits = list(inner.split(X[tr], y[tr], groups[tr]))
    for a, b in splits:
        assert_no_group_leak(groups[tr], a, b, "match_id")

    def fit(params: dict, a: np.ndarray, b: np.ndarray, n: int = 2000) -> "LGBMClassifier":
        # n_jobs=1 with both determinism flags, and all three are load-bearing: the flags alone
        # still drifted under thread contention, so the single thread is what actually pins it
        clf = LGBMClassifier(n_estimators=n, verbose=-1, n_jobs=1, random_state=seed,
                             deterministic=True, force_row_wise=True, **params)
        clf.fit(X[tr][a], y[tr][a], eval_set=[(X[tr][b], y[tr][b])],
                eval_metric="binary_logloss",
                callbacks=[early_stopping(100, verbose=False), log_evaluation(0)])
        return clf

    def objective(trial: "optuna.Trial") -> float:
        params = _gbt_space(trial, int(y[tr].sum()))
        return float(np.mean([log_loss(y[tr][b], fit(params, a, b).predict_proba(X[tr][b])[:, 1],
                                       labels=[0, 1]) for a, b in splits]))

    study = optuna.create_study(
        direction="minimize",
        sampler=optuna.samplers.TPESampler(seed=seed + k, multivariate=True),
        pruner=optuna.pruners.MedianPruner(n_warmup_steps=1))
    study.optimize(objective, n_trials=n_trials, n_jobs=1, show_progress_bar=False)

    # Rounds come from an inner split, never from the held-out fold: taking them there is the
    # leak this nesting exists to prevent.
    best = dict(study.best_params)
    rounds = int(fit(best, *splits[0]).best_iteration_ or 200)
    final = LGBMClassifier(n_estimators=rounds, verbose=-1, n_jobs=1, random_state=seed,
                           deterministic=True, force_row_wise=True, **best)
    final.fit(X[tr], y[tr])                          # no eval_set: the round count is already set
    return te, final.predict_proba(X[te])[:, 1], {
        "fold": k, "n_test": int(len(te)), "rounds": rounds,
        "inner_log_loss": float(study.best_value), **best}


def tuned_gbt_oof(X: np.ndarray, y: np.ndarray, groups: np.ndarray, n_trials: int = 60,
                  n_splits: int = 5, jobs: int = 1, seed: int = 0) -> tuple[np.ndarray, list]:
    """Out-of-fold probabilities from a NESTED, match-grouped, Optuna-tuned LightGBM.

    Why this arm exists: a null reported from a logistic head alone is a statement about
    linear-in-the-features models, not about the freeze-frame. Gradient-boosted trees remain
    the strongest family on tabular data of this size (Grinsztajn, Oyallon & Varoquaux,
    NeurIPS 2022, arXiv:2207.08815; Shwartz-Ziv & Armon, Information Fusion 2022), so if this
    arm cannot beat the delivery baseline either, the frame is what is empty.

    NESTED, because tuning and scoring on the same folds is optimistically biased by a margin
    that can exceed the effect being measured (Varma & Simon 2006; Cawley & Talbot, JMLR
    2010). Each outer fold runs its own search on its own training matches and is scored once
    on matches that search never saw. TPE rather than a grid, which searches a continuous
    space at a fraction of the evaluations (Bergstra et al. 2011; Bergstra & Bengio, JMLR
    2012; Akiba et al., KDD 2019).

    THE PARALLELISM IS OVER OUTER FOLDS, and everything below one is single-threaded. Parallel
    Optuna trials and multi-threaded LightGBM both break reproducibility, and a parallel run
    that does not return what the serial one returns is not admitted here. Parallel trials
    give TPE a history ordered by completion rather than by trial index, so the study stops
    being a function of its seed. Multi-threaded LightGBM diverges under contention
    even with `deterministic=True`: measured, two identical calls agreed while the machine was
    idle and disagreed inside a loaded test session. Outer folds are independent by
    construction, so splitting there costs nothing and changes nothing.

    Returns (oof probability of the positive class, per-outer-fold best params)."""
    outer = GroupKFold(n_splits=min(n_splits, len(np.unique(groups))))
    folds = list(outer.split(X, y, groups))
    for tr, te in folds:
        assert_no_group_leak(groups, tr, te, "match_id")
    done = Parallel(n_jobs=min(jobs, len(folds)))(
        delayed(_gbt_outer_fold)(X, y, groups, tr, te, n_trials, seed, k)
        for k, (tr, te) in enumerate(folds))

    oof = np.zeros(len(y), dtype=float)
    per_fold = []
    for te, p, meta in done:
        oof[te] = p
        per_fold.append(meta)
    return oof, sorted(per_fold, key=lambda r: r["fold"])
