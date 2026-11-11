"""s03_value — Stage 3 (VALUE): competing-outcomes model for the second ball.

Prices each corner from its delivery-instant configuration (s02 features) against
the outcome it produced (s01 label), following methods_second_ball.md §2:

    xSecondBall(config) = E[xG after the clearance | config] − λ · E[xG conceded | config]

each term an out-of-fold hurdle, P(xg > 0) · E[xg | xg > 0], never re-multiplied by a class
probability (`xsecondball`).

The machinery is the deliverable, not the numbers. What it enforces:
  - features strictly from T=delivery (s02), labels from T>delivery (s01) -> no leakage;
  - CV grouped BY MATCH (GroupKFold) so no corner leaks across train/test, with
    disjointness ASSERTED on every split path here rather than trusted to the splitter;
  - a calibrated multinomial competing-outcomes head (out-of-fold probabilities);
  - NESTED grouped tuning over C × class_weight × penalty (l2/l1/elasticnet),
    with the honest outer estimate, and temperature-scaling calibration fit on
    inner folds only (no resubstitution leak);
  - a hurdle (logistic × gamma, log link) for the sparse after-clearance xG, with
    Tweedie kept as a reported sensitivity;
  - feature-group ablations (Δ log-loss) to show what actually drives the value.

Note on scale: nested tuning and calibration are NON-OPERATIVE on a substrate too
small to split by match, and the code reports that rather than inventing a gain.
The Big-5 grant clears the bar (measured.md §corpus and §coverage_360).

No event/360 parsing here (that is s01/s02); this stage only models their join.

THE TARGET IS THE FRONTIER. Measured over the grant with match-clustered paired
intervals (evidence.md §2): against a direct shot the flick-on and the
prolongation convert measurably higher, while after-clearance is indistinguishable.
So the primary head groups the three into one class rather than pretending the
after-clearance is its own tier. The per-segment breakdown is a DESCRIPTIVE product,
not a second modelling head: pipeline/describe_frontier.py reports it at shot grain
with its own intervals.

`counter` is NOT a class. Q2 promised three terminal outcomes and this is the third,
but it is not estimable and the window was never the reason: swept over every cleared
delivery (pipeline/sweep_counter.py), the possession chain from the clearance breaks
down at every window width tried, and widening buys volume made of other teams'
next attacks rather than more contiguity (evidence.md §8 has the sweep). It
folds into clear_safe and survives as the `counter_shot` boolean, which is where the
rate is reported from. lambda_counter is 0.0 for the same reason.

TWO DEFAULTS carry measurements (evidence.md §5). class_weight is None, not
"balanced": balancing optimises balanced accuracy and deliberately mis-calibrates
against the observed class mix, costing enough OOF log-loss to put the full model
BELOW the class prior on the primary metric. Nested tuning picked None unanimously
across every outer fold, and "balanced" stays in the tuning grid. It holds for the
hurdle's stage-1 gate too, where the same mis-calibration is what decides whether
xSecondBall is an expectation at all.
"""

from __future__ import annotations

import copy
import warnings
from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import (GammaRegressor, LogisticRegression, Ridge,
                                  TweedieRegressor)
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler

import loader_sb as _sb
from constants import MIN_CELL
from s01_phases import SetPieceGrammar
from s02_signature import CommitmentSignature
from validate import (INSPECT, assert_no_group_leak, assert_split_disjoint,
                      cluster_bootstrap_mean, heterogeneity_blp, inspect_design,
                      leave_one_group_out, partial_pooling_cells, refit_cluster_bootstrap,
                      shuffle_features, shuffle_labels, wilcoxon_paired)
from zones import require_built_under, table_columns, zone_cols

# ── Outcome (s01) -> competing class (the multinomial target) ───────────
TARGET_MAP = {
    "shot_direct": "first_touch",
    "shot_flick_on": "after_first_touch",
    "shot_prolongation": "after_first_touch",
    "shot_after_clearance": "after_first_touch",
    "shot_late": "late_shot",
    "clear_counter": "clear_safe",
    "clear_safe": "clear_safe",
    "no_second_ball": "no_second_ball",
}

# ── Feature groups (drive the ablations and s05's specification curve) ──
FEATURE_GROUPS = {
    # The plan at the kick: each side's bodies per zone of the partition the switch names
    # (src/zones.py), and the defenders in the frontal apart, because they are s05's treatment.
    # The three jobs and the box counts are sums of these zones and the central edge band is a
    # diagnostic, so all three are table columns and not design blocks.
    "zones_att": zone_cols("att"),
    "zones_def": zone_cols("def"),
    "frontal_def": ["n_def_frontal"],
    # n_def_behind_half is NOT here on purpose: essentially no bodies sit in the
    # x<60 band across the grant, so it is a constant and its ablation delta is 0.0.
    "box_depth": ["box_block_depth"],
    "rebound": ["control_rebound"],
    "counter_lane": ["open_lane_central"],
    "marking": ["marking_cost"],
    "dispersion": ["def_dispersion", "att_dispersion", "def_hull_area", "att_hull_area"],
    # The legitimate substitute for the density pair barred below: the same inverse-distance
    # kernel, anchored on a FIXED point instead of on pass.end_location, so it cannot read the
    # play's future. The per-body forms carry the coverage. The body count itself stays in the
    # table and out of the design: it is every body the zones count, so beside the two zone
    # blocks it rebuilds the frontal count exactly, s05's treatment.
    "congestion": ["congestion_spot", "congestion_drop", "congestion_pairwise",
                   "congestion_spot_per_body", "congestion_drop_per_body",
                   "congestion_pairwise_per_pair"],
}
# att_density_5/att_density_10 are NOT here. They are measured around pass.end_location, which
# IS the first-contact point (evidence.md §15), so a feature anchored there reads the future of
# the play, which definitions.md §5 bars from X_t. They stay in the table as label-side
# passthrough. The `congestion` group above is the replacement, and evidence.md §11 measures that
# it is not the same quantity.
_DELIVERY_CAT = ["swing", "corner_side", "taker_foot", "delivery_height"]
_THREE_SIXTY = [c for cols in FEATURE_GROUPS.values() for c in cols]

# Read after the delivery, so forbidden in X_t (definitions.md §16). Asserted in the tests.
_LABEL_COLS = frozenset({"outcome", "target", "after_clearance_xg", "conceded_xg",
                         "cleared", "second_ball_time_s", "at_risk_horizon_s",
                         "resolution_time_s",
                         "clear_is_header", "clear_aerial_won", "counter_shot",
                         "first_contact_is_header", "first_contact_hops_rating",
                         "att_density_5", "att_density_10"})

# Penalty families swept by the tuner (each picks its valid solver in _clf).
_PENALTY_GRID = [{"penalty": "l2"}, {"penalty": "l1"}, {"penalty": "elasticnet", "l1_ratio": 0.5}]
# The tuner's C values, public because s05 tunes its propensity head over the same space.
TUNE_C_GRID = (0.05, 0.1, 0.25, 0.5, 1.0, 2.0)
# The nested tuning of the ladder's head. Every driver that tunes a published rung reads it here, so
# they are one estimator.
LADDER_TUNE: dict[str, Any] = dict(C_grid=(0.05, 0.25, 1.0, 4.0),
                                    class_weight_grid=(None, "balanced"),
                                    penalty_grid=({"penalty": "l2"},), n_outer=3)


# What is true at the STRIKE and nothing else. pass.length and pass.angle are NOT here: they
# reconstruct location -> pass.end_location at rounding (evidence.md §15), so they encode the
# first-contact point. On corners that leaves the kick position, which is the flag, so the
# honest delivery baseline is almost entirely categorical.
_KICK_INTRINSIC = ["delivery_x", "delivery_y"]

# The full delivery-instant design, and the kick spot is in it so the ladder's `full` rung
# NESTS `delivery`. It stays OUT of FEATURE_GROUPS, which is the 360 commitment block and is what
# s05's specification curve iterates: a "drop the kick spot" specification of the edge-protector
# effect means nothing.
_NUMERIC = _THREE_SIXTY + _KICK_INTRINSIC

# Every block that can be dropped or permuted, by name, as COLUMNS. `delivery` is the
# categorical half, which _design removes by switching the block off rather than by filtering.
_BLOCKS = {**FEATURE_GROUPS, "kick_spot": _KICK_INTRINSIC,
           "delivery": _DELIVERY_CAT + ["short_corner"]}

# Where every column the design can admit comes from, named as the export field it is built on.
# A column missing from here cannot enter: the test refuses it. Every partition's zone counts are
# declared, the ones the switch leaves out included, because the table carries them all.
_COLUMN_SOURCE = {**{c: "freeze_frame@T" for c in _THREE_SIXTY + table_columns()},
                  "delivery_x": "location", "delivery_y": "location",
                  "corner_side": "location", "swing": "pass.technique.name",
                  "taker_foot": "pass.body_part.name", "delivery_height": "pass.height.name",
                  "short_corner": "pass.height.name|pass.length"}

# Fields that read the play's future (definitions.md §5). A source naming one is refused unless the
# column is excepted here, with the exception measured rather than asserted. The two provider scores
# are barred before anything uses them: xclaim is documented as built from the end location, and
# the end point explains most of pass_success_probability among corners struck identically.
_BARRED_SOURCES = ("pass.end_location", "pass.length", "pass.angle",
                   "pass.xclaim", "pass.pass_success_probability")
_SOURCE_EXCEPTIONS = {"short_corner": "definitions.md §5: the length branch flags 37 corner kicks of 16720, all left with "
                                       "no delivery, and no modelled row"}

# Carried to the table and never in s03's X_t: team controls (s05 reads them as fixed effects),
# the delivery endpoint and the columns read at or after the clearance. _design's whitelist keeps
# them out by admitting a column only by name; membership of _LABEL_COLS is a separate, narrower
# guard.
_LABEL_PASSTHROUGH = ("attacking_team", "defending_team", "delivery_end_x", "delivery_end_y",
                      "at_risk_horizon_s", "resolution_time_s", "counter_shot",
                      "clear_is_header", "clear_aerial_won",
                      "first_contact_is_header", "first_contact_hops_rating")

_LEAGUE_OF_MATCH: dict = {}


def _league_index() -> dict:
    """match_id -> Big-5 league slug, built once from the catalog.

    list_matches(None) spans the five leagues ONLY, and that is load-bearing. Asking
    for every source would tag the 30 tracked matches 'serie_a_sample', and since
    those 30 are a strict subset of the Serie A season they would form a sixth
    leave-one-league-out fold that leaks into the fifth while looking legitimate.
    test_s03_value asserts that all 30 sample matches resolve to serie_a this way."""
    if not _LEAGUE_OF_MATCH:
        cat = _sb.list_matches(None)
        _LEAGUE_OF_MATCH.update(zip(cat["match_id"].astype(int), cat["league"], strict=True))
    return _LEAGUE_OF_MATCH


def _grid_loss(model: "CompetingOutcomesModel", X: np.ndarray, y: np.ndarray,
               groups: np.ndarray, idx: dict, params: dict) -> float | None:
    """Inner grouped-OOF log-loss of one hyperparameter config, or None if unsplittable.

    Module-level so joblib can ship it to a worker; the model travels with it because a
    CompetingOutcomesModel is a handful of scalars.

    A config whose fit did NOT CONVERGE returns None and drops out of the selection.
    That is not fastidiousness: `saga`, which the l1 and elasticnet families require,
    hits max_iter on this design, so leaving them in means the tuner compares a converged
    l2 fit against a half-fitted one and calls the winner. The warning has to be caught
    here rather than filtered by the caller: joblib workers do not inherit the parent
    process's warning filters, so a filter set in a driver never reaches this code."""
    with warnings.catch_warnings():
        warnings.simplefilter("error", ConvergenceWarning)
        try:
            oof = model._oof_proba(X, y, groups, idx, params)
        except ConvergenceWarning:
            return None
    return None if oof is None else model._logloss(oof, y, idx)


def _noise_cost(model: "CompetingOutcomesModel", table: pd.DataFrame, width: int, draw: int, n_repeats: int,
                seed: int) -> float:
    """One decoy block of `CompetingOutcomesModel.noise_floor`: `width` independent standard-normal columns
    appended to a copy of the table and named in `numeric=`, its base fit, and the cost that `n_repeats` global
    permutations of the block charge it. Module-level so joblib can ship it to a worker. The noise and the
    permutations are drawn from the (seed, width, draw) triple, so the draws are independent of each other and
    the same on every run."""
    y, groups = table["target"].to_numpy(), table["match_id"].to_numpy()
    rng = np.random.default_rng([seed, width, draw])
    cols = [f"_noise_{width}_{draw}_{j}" for j in range(width)]
    t = table.assign(**{c: rng.standard_normal(len(table)) for c in cols})
    numeric = list(_NUMERIC) + cols
    base = model._row_logloss(model.fit_cv(t, numeric=numeric), y)
    cost = np.zeros(len(t))
    for perm_seed in rng.integers(0, 2**31 - 1, size=n_repeats):
        s = shuffle_features(t, cols, y=y, groups=groups, mode="global", seed=int(perm_seed))
        cost += model._row_logloss(model.fit_cv(s, numeric=numeric), y) - base
    return float((cost / n_repeats).mean())


def _f(v: Any) -> float:
    """Optional label float -> float, NaN when the provider left it out."""
    return float(v) if pd.notna(v) else float("nan")


def corner_rows(match_id: int, league: str, kinds: tuple = ("corner",)) -> list:
    """One match -> its corner row dicts (s02 features + s01 label + grouping keys).

    Pure and self-contained so it can run in a worker process. It returns DICTS and
    not a DataFrame on purpose: the parent flattens every match at once into a single
    frame, so dtypes are inferred over all rows together. Per-match frames plus
    pd.concat infer per match, and a column that is all-NaN in one match and integer
    in another would then land on a different dtype than the serial path gives, so the
    table would quietly stop reproducing the published numbers with no error."""
    ev = _sb.load_events(match_id)
    fz = _sb.load_360(match_id)
    ids = set(fz["event_uuid"])
    g = SetPieceGrammar(ev, kinds=kinds)
    labels = g.label_all()
    if labels.empty:
        return []
    labels = labels.set_index("set_piece_id")
    return [delivery_row(c, labels.loc[c["id"]], fz, match_id, league)
            for _, c in g.deliveries().iterrows()
            if c["id"] in ids and c["id"] in labels.index]


def delivery_row(d: pd.Series, lab: Any, fz: pd.DataFrame, match_id: int, league: str) -> dict:
    """One delivery's modelling row: the s02 features of its 360 plus its s01 label.

    The one place the row schema lives. `corner_rows` calls it per delivery, and so does any driver
    that models a delivery the grammar does not enumerate (a short corner at its own kick), so the
    two tables cannot disagree on a column. `lab` is anything indexable by label field."""
    return {
        **CommitmentSignature.from_corner(d, fz).row(d["id"]),
        "match_id": match_id, "league": league, "set_piece_type": d["set_piece_type"],
        **{k: lab[k] for k in _LABEL_PASSTHROUGH},
        **{k: lab[k] for k in _KICK_INTRINSIC},
        "delivery_height": lab["delivery_height"],
        # `swing` is a CORNER qualifier and constant on free kicks, where the foot is the only
        # kick technique left.
        "outcome": lab["outcome"], "swing": lab["swing"], "taker_foot": lab["taker_foot"],
        "short_corner": bool(lab["short_corner"]), "corner_side": lab["corner_side"],
        "after_clearance_xg": float(lab["after_clearance_xg"]),
        "conceded_xg": float(lab["conceded_xg"]),
        "cleared": bool(lab["cleared"]),
        "second_ball_time_s": _f(lab["second_ball_time_s"]),
    }


def build_corner_table(match_ids: list, n_jobs: int = 1, kinds: tuple = ("corner",)) -> pd.DataFrame:
    """Modelling table: one row per corner WITH 360 = s02 features/signature + s01
    label, plus the two CV grouping keys (match_id and league), the team controls and
    the competing-outcomes target.

    n_jobs=1 is the reference implementation and n_jobs>1 must reproduce it exactly:
    row order drives GroupKFold fold membership and saga's update order, so a
    reordered table shifts every headline log-loss in the third decimal. Sizing is by peak
    RSS per worker, not by core count, and a fan-out that opens TRACKING is capped at 2-3 by
    the per-match peak load_tracking's own docstring records."""
    leagues = _league_index()
    pairs = [(int(m), leagues.get(int(m))) for m in match_ids]
    # Fail before any work: a corner with no league would silently become its own
    # leave-one-league-out fold and look like a legitimate sixth competition.
    unknown = [m for m, lg in pairs if lg is None]
    if unknown:
        raise ValueError(f"league unresolved for {len(unknown)} match(es): {unknown[:5]}")

    if n_jobs == 1:
        chunks = [corner_rows(m, lg, kinds) for m, lg in pairs]
    else:
        from joblib import Parallel, delayed
        chunks = Parallel(n_jobs=n_jobs, verbose=5)(
            delayed(corner_rows)(m, lg, kinds) for m, lg in pairs)

    df = pd.DataFrame([r for chunk in chunks for r in chunk])
    if not df.empty:
        df["target"] = df["outcome"].map(TARGET_MAP)
        assert_declared_columns(df)
    return df


def assert_declared_columns(table: pd.DataFrame) -> None:
    """Raise when a column the design declares never reached the table.

    _design filters its whitelist by `c in table.columns`, which is what lets a declared column
    go missing without a word, and the baseline would then run short of what the prose describes.
    Every builder of a modelling table calls this, pipeline/build_table included, because that
    driver assembles the published table itself and a guard off that path protects nothing."""
    missing = [c for c in _DELIVERY_CAT + _KICK_INTRINSIC if c not in table.columns]
    if missing:
        raise ValueError(f"the design declares columns this table never wrote: {missing}")


@dataclass
class CVResult:
    classes: list
    oof_proba: np.ndarray
    log_loss: float
    brier: float
    accuracy: float
    ece: float
    n: int
    class_counts: dict = field(default_factory=dict)
    smooth_ece: float = float("nan")   # the headline; `ece` is the withdrawn binned one


@dataclass
class GroupOutResult:
    group_col: str
    classes: list
    oof_proba: np.ndarray
    log_loss: float
    log_loss_mean: float
    log_loss_std: float
    n: int
    n_folds: int
    folds: list
    leakage_checked: tuple
    degenerate_folds: list


@dataclass
class TuneResult:
    best_params: dict
    outer_log_loss: float
    outer_log_loss_std: float
    per_fold: list
    grid: list
    n: int
    n_outer: int
    n_inner: int


@dataclass
class CalibrationResult:
    method: str
    classes: list
    oof_proba_raw: np.ndarray
    oof_proba_cal: np.ndarray
    log_loss_before: float
    log_loss_after: float
    ece_before: float
    ece_after: float
    brier_before: float
    brier_after: float
    reliability_before: pd.DataFrame
    reliability_after: pd.DataFrame
    n: int
    smooth_ece_before: float = float("nan")
    smooth_ece_after: float = float("nan")


class CompetingOutcomesModel:
    """Grouped, tuned, calibrated competing-outcomes value model for one set-piece
    population; `build_corner_table` builds corners, free kicks or both."""

    def __init__(self, lambda_counter: float = 0.0, n_splits: int = 5, C: float = 0.5,
                 class_weight: str | dict | None = None, penalty: str = "l2",
                 l1_ratio: float | None = None) -> None:
        self.lambda_counter = lambda_counter
        self.n_splits = n_splits
        self.C = C
        self.class_weight = class_weight
        self.penalty = penalty
        self.l1_ratio = l1_ratio
        self.best_params_ = None

    # ── Estimator (one place builds every multinomial head) ─────────────
    def _params(self) -> dict:
        """Current hyperparameters as a dict (the tuner overrides these per fold)."""
        return {"C": self.C, "class_weight": self.class_weight,
                "penalty": self.penalty, "l1_ratio": self.l1_ratio}

    def _clf(self, params: dict | None = None) -> LogisticRegression:
        """Build the multinomial head; `params` overrides the instance defaults."""
        p = {**self._params(), **(params or {})}
        pen = p.get("penalty", "l2")
        # saga is the only solver that supports l1 / elasticnet; lbfgs is faster for l2. saga
        # is also stochastic, and without a seed the same fit lands somewhere else on a rerun:
        # measured on the intercept-only ladder rung, which hands it an all-zero design, two
        # calls returned 1.8971 and 1.7251. lbfgs ignores random_state, so pinning it cannot
        # move anything the l2 path has already published.
        kw = dict(max_iter=5000, C=p["C"], class_weight=p["class_weight"], penalty=pen,
                  random_state=0,
                  solver="saga" if pen in ("l1", "elasticnet") else "lbfgs")
        if pen == "elasticnet":
            kw["l1_ratio"] = p.get("l1_ratio") if p.get("l1_ratio") is not None else 0.5
        return LogisticRegression(**kw)

    def _design(self, table: pd.DataFrame, drop_group: str | None = None,
                numeric: list | None = None, with_delivery_cats: bool = True) -> tuple:
        """Design matrix, strictly delivery-instant.

        EVERY baseline-ladder rung is fit unbalanced, which is what makes the ladder
        mean anything: a balanced rung cannot be compared to an unbalanced prior.

        `numeric=None` keeps the whole _NUMERIC block. A list restricts it to exactly
        those columns and `[]` gives an intercept-only prior, which is what the baseline
        ladder needs. The columns are a WHITELIST rather than a drop-list, so a
        post-delivery label cannot enter by accident, only by being named, and a named column
        the table lacks stops the fit: dropped in silence, a table built before a block existed
        would fit without it and still read as the full design. A table built under a partition whose
        jobs and frontal are not the switch's stops it too (`zones.require_built_under`)."""
        require_built_under(table)
        num = list(_NUMERIC if numeric is None else numeric)
        missing = [c for c in num if c not in table.columns]
        if missing:
            raise ValueError(f"the design names columns this table does not carry: {missing}")
        if drop_group:
            num = [c for c in num if c not in _BLOCKS.get(drop_group, [])]
        Xn = table[num].apply(pd.to_numeric, errors="coerce")
        # Missing-indicator per numeric column ever absent (absence is informative: no marking_cost
        # <-> empty side). The 0/1 flag is per row so it cannot leak; the NaN itself is imputed PER
        # FOLD in _fit_prep (train-median only), never with a table-wide median here.
        miss = Xn.isna()
        miss = miss.loc[:, miss.any()].astype(float).add_suffix("_isna")
        parts = [Xn, miss]
        if drop_group != "delivery" and with_delivery_cats:
            cat = pd.get_dummies(table[[c for c in _DELIVERY_CAT if c in table.columns]],
                                 dtype=float)
            parts += [cat, table["short_corner"].astype(float)]
        X = pd.concat(parts, axis=1)
        if X.shape[1] == 0:                              # intercept-only rung needs a carrier column
            X = pd.DataFrame({"_const": np.zeros(len(table))})
        Xa, cols = X.to_numpy(float), list(X.columns)
        # Off by default: the scan runs on the full table and _design is called once per rung
        # per ablation group. HUDL_INSPECT=1 turns it on. It never raises, because an all-NaN
        # column and an intercept-only constant are legitimate here.
        if INSPECT:
            inspect_design(Xa, cols, label=f"s03 design drop={drop_group or 'none'}")
        return Xa, cols

    @staticmethod
    def _fit_prep(Xtr: np.ndarray) -> Callable[[np.ndarray], np.ndarray]:
        """Fit median-impute + standardize on TRAIN ONLY; return a transform.
        The median is the training fold's (no table-wide leak); a column all-NaN
        in train imputes to 0."""
        med = np.nanmedian(Xtr, axis=0)
        med = np.where(np.isnan(med), 0.0, med)
        sc = StandardScaler().fit(np.where(np.isnan(Xtr), med, Xtr))
        return lambda X: sc.transform(np.where(np.isnan(X), med, X))

    def _fit_score(self, X: np.ndarray, y: np.ndarray, tr: np.ndarray, te: np.ndarray,
                   idx: dict) -> np.ndarray:
        """Fit on `tr`, return aligned and renormalised probabilities for `te`.

        The out-of-sample sibling of _oof_proba, and the same three steps fit_group_out runs
        per fold: scaler on train only, one fit, columns placed into the global class order."""
        tf = self._fit_prep(X[tr])
        clf = self._clf().fit(tf(X[tr]), y[tr])
        return self._renorm(self._align(clf.predict_proba(tf(X[te])), clf.classes_,
                                        idx, len(idx)))

    def _n_splits(self, groups: np.ndarray) -> int:
        """n_splits capped at the number of match groups (can't have more folds)."""
        return int(min(self.n_splits, pd.Series(groups).nunique()))

    @staticmethod
    def _align(proba: np.ndarray, clf_classes: np.ndarray, idx: dict, K: int) -> np.ndarray:
        """Place a fold model's predict_proba columns into the global class order."""
        out = np.zeros((proba.shape[0], K))              # zero for classes absent from this fold
        for j, cl in enumerate(clf_classes):
            out[:, idx[cl]] = proba[:, j]
        return out

    @staticmethod
    def _renorm(p: np.ndarray) -> np.ndarray:
        """Row-normalise to sum 1 (aligned/averaged folds may drift off the simplex)."""
        return p / p.sum(axis=1, keepdims=True).clip(1e-9)

    # ── Out-of-fold multinomial probabilities for given hyperparameters ──
    def _oof_proba(self, X: np.ndarray, y: np.ndarray, groups: np.ndarray, idx: dict,
                   params: dict) -> np.ndarray | None:
        ns = self._n_splits(groups)
        if ns < 2:
            return None
        K = len(idx)
        oof = np.zeros((len(y), K))
        for tr, te in GroupKFold(n_splits=ns).split(X, y, groups):
            assert_no_group_leak(groups, tr, te, "match_id")
            tf = self._fit_prep(X[tr])                   # impute+scale fitted on THIS fold's train only
            clf = self._clf(params).fit(tf(X[tr]), y[tr])
            oof[te] = self._align(clf.predict_proba(tf(X[te])), clf.classes_, idx, K)
        return self._renorm(oof)

    @staticmethod
    def _logloss(oof: np.ndarray, y: np.ndarray, idx: dict) -> float:
        """Multiclass log-loss of OOF probabilities against the true class (idx maps
        class name -> column); clipped so a 0-probability true class can't blow up."""
        yi = np.array([idx[v] for v in y])
        return float(-np.log(oof[np.arange(len(y)), yi].clip(1e-12)).mean())

    # ── Multinomial competing-outcomes head (out-of-fold) ───────────────
    def fit_cv(self, table: pd.DataFrame, drop_group: str | None = None,
               numeric: list | None = None, with_delivery_cats: bool = True) -> CVResult:
        """Out-of-fold probabilities of the multinomial head, folds grouped by match, with their
        log loss, Brier, accuracy and calibration. Under two matches nothing can be held out, so
        every row gets the uniform prior rather than a fitted number."""
        X, _ = self._design(table, drop_group, numeric, with_delivery_cats)
        y = table["target"].to_numpy()
        groups = table["match_id"].to_numpy()
        classes = sorted(pd.unique(y))
        idx = {c: i for i, c in enumerate(classes)}
        oof = self._oof_proba(X, y, groups, idx, self._params())
        if oof is None:                                   # <2 match groups -> uniform prior, not a fitted number
            oof = np.full((len(y), len(classes)), 1.0 / len(classes))

        yi = np.array([idx[v] for v in y])
        Y = np.eye(len(classes))[yi]
        ll = float(-(Y * np.log(oof.clip(1e-12))).sum(axis=1).mean())
        brier = float(((oof - Y) ** 2).sum(axis=1).mean())
        acc = float((oof.argmax(1) == yi).mean())
        return CVResult(classes, oof, ll, brier, acc, self.ece(yi, oof), len(y),
                        {c: int((y == c).sum()) for c in classes},
                        self.smooth_ece(yi, oof))

    # ── Leave-one-group-out: does it travel to an unseen LEAGUE? ────────
    def fit_group_out(self, table: pd.DataFrame, group_col: str = "league",
                      drop_group: str | None = None) -> GroupOutResult:
        """Hold out one whole group at a time: the external-validity sibling of fit_cv.

        fit_cv answers "does this generalise to an unseen MATCH". This answers "to an
        unseen LEAGUE", which is what Q2 promises. The number that carries the science is
        the gap between the two: if league-out is close to match-out the model is not
        memorising league style.

        Disjointness is asserted on EVERY grouping key the table carries, not just the
        one being split on. A league-out fold is match-disjoint only if `league`
        resolved for every row, and that is exactly the thing that can go wrong.

        A fold whose test block contains a class absent from its train block gets
        probability clipped to 1e-12 on those rows (per-row loss 27.6), which makes a
        structural gap look like a modelling catastrophe. Those folds are listed and
        excluded from log_loss_mean, never silently averaged in."""
        X, _ = self._design(table, drop_group)
        y = table["target"].to_numpy()
        g = table[group_col].to_numpy()
        classes = sorted(pd.unique(y))
        idx = {c: i for i, c in enumerate(classes)}
        # match_id is mandatory (the fold report reads it unconditionally below); only
        # league is optional, so only league is probed. Filtering match_id here too
        # would let a table that lost it pass the leakage guard silently.
        keys = ("match_id",) + (("league",) if "league" in table.columns else ())
        oof = np.zeros((len(y), len(classes)))
        folds, degenerate = [], []

        for held, tr, te in leave_one_group_out(g):
            assert_split_disjoint(table, tr, te, cols=keys)
            tf = self._fit_prep(X[tr])
            clf = self._clf().fit(tf(X[tr]), y[tr])
            oof[te] = self._align(clf.predict_proba(tf(X[te])), clf.classes_, idx, len(classes))
            missing = sorted(set(y[te]) - set(y[tr]))
            if missing:
                degenerate.append(held)
            folds.append({"held_out": held, "n_train": int(len(tr)), "n_test": int(len(te)),
                          "n_train_matches": int(pd.Series(table["match_id"].to_numpy()[tr]).nunique()),
                          "classes_missing_from_train": missing,
                          "class_counts_test": {c: int((y[te] == c).sum()) for c in classes}})
        oof = self._renorm(oof)

        yi = np.array([idx[v] for v in y])
        Y = np.eye(len(classes))[yi]
        for f, (_, _, te) in zip(folds, leave_one_group_out(g), strict=True):
            f["log_loss"] = self._logloss(oof[te], y[te], idx)
            f["brier"] = float(((oof[te] - Y[te]) ** 2).sum(axis=1).mean())
            f["accuracy"] = float((oof[te].argmax(1) == yi[te]).mean())
            f["ece"] = self.ece(yi[te], oof[te])
            f["smooth_ece"] = self.smooth_ece(yi[te], oof[te])
        clean = [f["log_loss"] for f in folds if not f["classes_missing_from_train"]]
        return GroupOutResult(
            group_col, classes, oof, self._logloss(oof, y, idx),
            float(np.mean(clean)) if clean else float("nan"),
            float(np.std(clean)) if clean else float("nan"),
            len(y), len(folds), folds, keys, degenerate)

    # ── Nested grouped hyperparameter tuning (honest outer estimate) ────
    def tune_cv(self, table: pd.DataFrame, drop_group: str | None = None,
                C_grid: tuple = TUNE_C_GRID,
                class_weight_grid: tuple = ("balanced", None),
                penalty_grid: tuple = tuple(_PENALTY_GRID),
                n_outer: int | None = None, n_jobs: int = 1) -> TuneResult:
        """Nested tuning grouped by match: the inner CV picks a configuration on each outer
        training block and the outer test scores it, so the reported loss never saw its own
        selection. `best_params` is the configuration the outer folds pick most often."""
        X, _ = self._design(table, drop_group)
        y = table["target"].to_numpy()
        groups = table["match_id"].to_numpy()
        idx = {c: i for i, c in enumerate(sorted(pd.unique(y)))}
        grid = [{"C": c, "class_weight": w, **pen}
                for c in C_grid for w in class_weight_grid for pen in penalty_grid]

        n_out = min(n_outer or self.n_splits, pd.Series(groups).nunique())
        if n_out < 2:
            return TuneResult({**self._params()}, float("nan"), float("nan"), [], grid, len(y), int(n_out), 0)

        per_fold = []
        for otr, ote in GroupKFold(n_splits=n_out).split(X, y, groups):
            assert_no_group_leak(groups, otr, ote, "match_id")
            best = self._inner_select(X[otr], y[otr], groups[otr], idx, grid, n_jobs)
            tf = self._fit_prep(X[otr])
            clf = self._clf(best).fit(tf(X[otr]), y[otr])
            p = self._renorm(self._align(clf.predict_proba(tf(X[ote])), clf.classes_, idx, len(idx)))
            per_fold.append({"best_params": best, "log_loss": self._logloss(p, y[ote], idx),
                             "n_test": int(len(ote))})

        lls = [f["log_loss"] for f in per_fold]
        modal = Counter(tuple(sorted(f["best_params"].items())) for f in per_fold).most_common(1)[0][0]
        return TuneResult(dict(modal), float(np.mean(lls)), float(np.std(lls)),
                          per_fold, grid, len(y), int(n_out), int(self.n_splits))

    def _inner_select(self, X: np.ndarray, y: np.ndarray, groups: np.ndarray, idx: dict,
                      grid: list, n_jobs: int = 1) -> dict:
        """Pick the config by mean inner grouped-OOF log-loss on a train block.

        The inner CV is the model's own `n_splits`. The grid is embarrassingly parallel
        (each config is an independent grouped CV), and serially it is the whole cost of
        tuning: the default 6 C x 2 weights x 3 penalties is 36 configs, each an
        n_splits-fold fit on the training block, with `saga` for the l1 and elasticnet
        families. Parallelising cannot change the answer: the losses are
        computed independently and the winner is a deterministic scan over the SAME
        ordered list, with the same tie-break."""
        if n_jobs != 1:
            from joblib import Parallel, delayed, parallel_config
            # inner_max_num_threads=1 is not a nicety. Each worker's LogisticRegression
            # calls into BLAS, which spawns its own thread pool: measured, 14 workers put
            # 27 threads on 24 cores, so the machine thrashes and the parallel run is
            # slower than it should be. One BLAS thread per worker is the right shape
            # when the parallelism is already at the config level.
            with parallel_config(backend="loky", inner_max_num_threads=1):
                losses = Parallel(n_jobs=n_jobs)(
                    delayed(_grid_loss)(self, X, y, groups, idx, p) for p in grid)
        else:
            losses = [_grid_loss(self, X, y, groups, idx, p) for p in grid]

        best, best_ll = None, np.inf
        for p, ll in zip(grid, losses, strict=True):
            if ll is None:
                continue
            # keep the lower loss; on a tie prefer the smaller C (stronger regularisation)
            if ll < best_ll - 1e-9 or (best is not None and abs(ll - best_ll) <= 1e-9 and p["C"] < best["C"]):
                best, best_ll = p, ll
        return best or {**self._params()}

    def apply_best_params(self, tune: TuneResult) -> "CompetingOutcomesModel":
        """Adopt tuned hyperparameters so fit_cv/hurdle/xsecondball reuse them."""
        bp = tune.best_params
        self.C = bp.get("C", self.C)
        self.class_weight = bp.get("class_weight", self.class_weight)
        self.penalty = bp.get("penalty", self.penalty)
        self.l1_ratio = bp.get("l1_ratio", self.l1_ratio)
        self.best_params_ = dict(bp)
        return self

    # ── Temperature-scaling calibration (Guo et al. 2017; no resubstitution leak) ──
    @staticmethod
    def _apply_temperature(p: np.ndarray, T: float) -> np.ndarray:
        """Divide the logits by temperature T (>1 softens, <1 sharpens) and re-softmax."""
        z = np.log(np.clip(p, 1e-12, 1.0)) / T
        z = z - z.max(axis=1, keepdims=True)
        e = np.exp(z)
        return e / e.sum(axis=1, keepdims=True)

    def _optimize_temperature(self, p: np.ndarray, y_idx: np.ndarray) -> float:
        """Find the T that minimises NLL of `p` against the true classes."""
        def nll(T: float) -> float:
            cp = self._apply_temperature(p, T)
            return float(-np.log(cp[np.arange(len(y_idx)), y_idx].clip(1e-12)).mean())
        return float(minimize_scalar(nll, bounds=(0.2, 10.0), method="bounded").x)

    def _fit_temperature(self, X: np.ndarray, y: np.ndarray, groups: np.ndarray,
                         idx: dict) -> float:
        """Fit T on INNER grouped-OOF probabilities of the train block (T=1 if it
        cannot be split), never on the rows the calibrator will be applied to."""
        oof = self._oof_proba(X, y, groups, idx, self._params())
        if oof is None:
            return 1.0
        yi = np.array([idx[v] for v in y])
        return self._optimize_temperature(oof, yi)

    def calibrate_cv(self, table: pd.DataFrame, drop_group: str | None = None,
                     bins: int = 10) -> CalibrationResult:
        """Temperature scaling scored out of fold: each outer fold fits its temperature on the
        inner OOF of its own training block and applies it to its test rows, so the scalar never
        sees the rows it is scored on. Raw and scaled probabilities travel together."""
        X, _ = self._design(table, drop_group)
        y = table["target"].to_numpy()
        groups = table["match_id"].to_numpy()
        classes = sorted(pd.unique(y))
        idx = {c: i for i, c in enumerate(classes)}
        K, n = len(classes), len(y)
        raw, cal = np.zeros((n, K)), np.zeros((n, K))

        for tr, te in GroupKFold(n_splits=self._n_splits(groups)).split(X, y, groups):
            assert_no_group_leak(groups, tr, te, "match_id")
            tf = self._fit_prep(X[tr])
            clf = self._clf().fit(tf(X[tr]), y[tr])
            ptst = self._align(clf.predict_proba(tf(X[te])), clf.classes_, idx, K)
            raw[te] = ptst
            T = self._fit_temperature(X[tr], y[tr], groups[tr], idx)     # T from inner OOF only
            cal[te] = self._apply_temperature(self._renorm(ptst), T)
        raw, cal = self._renorm(raw), self._renorm(cal)

        yi = np.array([idx[v] for v in y])
        Y = np.eye(K)[yi]
        ll = lambda P: float(-(Y * np.log(P.clip(1e-12))).sum(1).mean())
        br = lambda P: float(((P - Y) ** 2).sum(1).mean())
        return CalibrationResult("temperature", classes, raw, cal, ll(raw), ll(cal),
                                 self.ece(yi, raw), self.ece(yi, cal), br(raw), br(cal),
                                 self.reliability_bins(yi, raw, bins),
                                 self.reliability_bins(yi, cal, bins), n,
                                 self.smooth_ece(yi, raw), self.smooth_ece(yi, cal))

    # ── Hurdle (logistic × gamma) for after-clearance xG, out-of-fold ──────
    def hurdle_xg(self, table: pd.DataFrame, col: str = "after_clearance_xg",
                  second_stage: str = "gamma",
                  return_parts: bool = False) -> np.ndarray | tuple:
        """P(xg > 0) × E[xg | xg > 0], both fitted out of fold and grouped by match.

        THE GATE IS FIT UNWEIGHTED, like every other head here, and that is what makes the
        product an expectation instead of a number several times too large. Balancing
        optimises balanced accuracy and deliberately mis-calibrates against the observed
        rate, so a balanced gate stops being P(xg > 0) at all: its mean lands far above the
        observed positive rate and drives xSecondBall to a multiple of the xG it is meant to
        price. Unweighted, the same gate beats a constant base-rate gate out of fold by a
        margin whose match-clustered interval clears zero, so the hurdle adds information
        rather than dressing up the prior. No calibrator sits on top: fitted on inner folds,
        temperature moves the fifth decimal. None of those measurements reaches a derived
        artefact, so none of them is written here.

        `penalty` is pinned rather than inherited. After apply_best_params the instance can
        carry l1 or elasticnet, and the gate would then switch to `saga` behind the caller's
        back; nested tuning does pick l1 on the grant, so this is live and not theoretical.

        `return_parts` hands back (gate, magnitude) rather than their product, which is what
        lets the gate be scored as a probability on its own.

        The second stage is a GAMMA GLM with a log link, per methods_second_ball.md §2.2.
        The hurdle has already removed every zero, so the stage-2 sample is strictly
        positive and right-skewed, which is what the gamma is for. Tweedie at power 1.5 is
        a compound Poisson-gamma: it carries its own mass at zero, so fitting it to a
        zero-free subsample spends a parameter modelling something the gate already
        handled. It stays reachable as `second_stage="tweedie"` and is reported as a
        sensitivity, never as the headline."""
        if second_stage not in ("gamma", "tweedie"):
            raise ValueError(f"second_stage must be 'gamma' or 'tweedie', got {second_stage!r}")
        X, _ = self._design(table)
        groups = table["match_id"].to_numpy()
        xg = table[col].to_numpy(float)
        pos = (xg > 0).astype(int)
        p_oof = np.zeros(len(xg))
        xg_oof = np.zeros(len(xg))
        if pos.sum() < 2 or self._n_splits(groups) < 2:
            return (p_oof, xg_oof) if return_parts else p_oof
        for tr, te in GroupKFold(n_splits=self._n_splits(groups)).split(X, pos, groups):
            assert_no_group_leak(groups, tr, te, "match_id")
            tf = self._fit_prep(X[tr])
            if len(np.unique(pos[tr])) >= 2:
                clf = self._clf({"class_weight": None, "penalty": "l2"}).fit(tf(X[tr]), pos[tr])
                p_oof[te] = clf.predict_proba(tf(X[te]))[:, list(clf.classes_).index(1)]
            ptr = tr[pos[tr] == 1]
            if len(ptr) >= 5:
                reg = (GammaRegressor(alpha=0.1, max_iter=5000) if second_stage == "gamma"
                       else TweedieRegressor(power=1.5, alpha=0.1, max_iter=5000))
                reg.fit(tf(X[ptr]), xg[ptr])
                xg_oof[te] = reg.predict(tf(X[te])).clip(0.0)
        return (p_oof, xg_oof) if return_parts else p_oof * xg_oof

    # ── xSecondBall = attacking upside − λ·transition cost ──────────────
    def xsecondball(self, table: pd.DataFrame) -> np.ndarray:
        """Expected second-ball value. With the default lambda_counter = 0.0 this is
        the attacking term alone, because the transition arm is not estimable on this
        data: pipeline/sweep_counter.py is the evidence, not an assumption. Set
        lambda_counter explicitly to include it, knowing what it rests on.

        Both hurdle terms are already an EXPECTED danger P(event>0)*E[xg|event>0]: the
        stage-1 logistic gate carries the probability. Neither may be re-multiplied by
        the multinomial P(counter), which would count the same probability twice and
        shrink the cost."""
        e_attack = self.hurdle_xg(table, "after_clearance_xg")
        e_counter = self.hurdle_xg(table, "conceded_xg")
        return e_attack - self.lambda_counter * e_counter

    @staticmethod
    def _row_logloss(cv: CVResult, y: np.ndarray) -> np.ndarray:
        """Per-row OOF log-loss, so two models can be differenced row by row."""
        idx = {c: i for i, c in enumerate(cv.classes)}
        yi = np.array([idx[v] for v in y])
        return -np.log(cv.oof_proba[np.arange(len(y)), yi].clip(1e-12))

    # ── Ablations: Δ log-loss when each feature group is removed ─────────
    def ablate(self, table: pd.DataFrame, n_boot: int = 0, seed: int = 0) -> pd.DataFrame:
        """Δ log-loss per feature group, with a match-clustered interval on each Δ.

        The interval is not decoration. Without it this is a ranked table of point
        estimates whose fold-to-fold spread can exceed the spread between rows, and
        a group whose Δ is indistinguishable from zero reads as a finding. Set
        n_boot=0 to skip it when only the ordering is wanted.

        The Δ is PAIRED: both models run the same GroupKFold on the same rows, so it
        is the mean of a per-row difference, not the difference of two means. Two
        marginal intervals that overlap would say nothing about the Δ."""
        y = table["target"].to_numpy()
        groups = table["match_id"].to_numpy()
        base_cv = self.fit_cv(table)
        base_rows = self._row_logloss(base_cv, y)
        rows = [{"group": "ALL", "log_loss": base_cv.log_loss, "delta": 0.0,
                 "lo": 0.0, "hi": 0.0, "n_boot": 0, "p_wilcoxon": float("nan")}]
        for grp in _BLOCKS:
            cv = self.fit_cv(table, drop_group=grp)
            d = self._row_logloss(cv, y) - base_rows        # +delta = dropping it hurt = the group helped
            ci = (cluster_bootstrap_mean(d, groups, n_boot=n_boot, seed=seed) if n_boot
                  else None)
            # The signed-rank test on the same paired difference, clustered by match. It runs on
            # the FIXED fit, so it inherits that interval's narrowness and is not a second
            # opinion: the verdict is the refit one in normal form (ablation_refit_interval).
            rows.append({"group": grp, "log_loss": cv.log_loss, "delta": float(d.mean()),
                         "lo": ci.lo if ci else float("nan"),
                         "hi": ci.hi if ci else float("nan"),
                         "n_boot": ci.n_boot if ci else 0,
                         "p_wilcoxon": wilcoxon_paired(d, groups)["p_value"]})
        return pd.DataFrame(rows).sort_values("delta", ascending=False).reset_index(drop=True)

    # ── The baseline ladder and the shuffled controls ───────────────────
    def baseline_ladder(self, table: pd.DataFrame, n_boot: int = 2000,
                        seed: int = 0) -> pd.DataFrame:
        """Every rung fit with the IDENTICAL grouped CV and head, each with a
        match-clustered interval and a PAIRED interval on the step it gained.

        run_value.py prints ln(#classes), the uniform floor, which almost anything beats.
        The rung that carries the scientific claim is `delivery`: it knows the DESIGN of the
        kick, which is where it was struck from, from which side, with which foot, swing and
        height, and whether it was short.
        It does NOT know where the ball was aimed: pass.end_location IS the first-contact
        point and definitions.md §5 bars it from X_t. Any statement that the 360 commitment
        features add information has to be a delta against THAT, not against uniform or the
        prior.

        Every rung NESTS the one above it, which is what lets a step be read as what the new
        block bought, and `full` carries the kick spot for that reason. Dropping the two
        coordinates from `delivery` would buy the nesting more cheaply and is the wrong repair:
        they are near-constant on corners but real on free kicks, so a baseline without them
        flatters the 360 block on the population where the block wins. No column of the
        `delivery` rung is ever null, so there is no missingness confound either."""
        y = table["target"].to_numpy()
        groups = table["match_id"].to_numpy()
        rungs = [("prior", dict(numeric=[], with_delivery_cats=False), None),
                 ("delivery", dict(numeric=_KICK_INTRINSIC,
                                   with_delivery_cats=True), None),
                 ("full", dict(numeric=None, with_delivery_cats=True), None),
                 ("full_balanced", dict(numeric=None, with_delivery_cats=True), "balanced")]
        rows = [{"rung": "uniform", "log_loss": float(np.log(len(pd.unique(y)))),
                 "lo": np.nan, "hi": np.nan, "step": np.nan, "step_lo": np.nan, "step_hi": np.nan}]
        prev = None
        for name, kw, weight in rungs:
            m = copy.copy(self)
            m.class_weight = weight
            cv = m.fit_cv(table, **kw)
            rowll = self._row_logloss(cv, y)
            ci = cluster_bootstrap_mean(rowll, groups, n_boot=n_boot, seed=seed)
            step = (cluster_bootstrap_mean(prev - rowll, groups, n_boot=n_boot, seed=seed)
                    if prev is not None else None)
            rows.append({"rung": name, "log_loss": cv.log_loss, "lo": ci.lo, "hi": ci.hi,
                         "step": step.point if step else np.nan,
                         "step_lo": step.lo if step else np.nan,
                         "step_hi": step.hi if step else np.nan})
            prev = rowll
        return pd.DataFrame(rows)

    def permuted_ablation(self, table: pd.DataFrame, mode: str = "global", n_repeats: int = 3,
                          n_boot: int = 0, seed: int = 0) -> pd.DataFrame:
        """Δ log-loss per feature group when the block is PERMUTED instead of dropped.

        Dropping a block moves its information, the width of the design and the regularisation
        path together. Permuting moves only the information, so the two tables read side by
        side. Neither is safe alone, and THIS one inflates: permuting one of two collinear
        columns hands the head a self-contradicting row and charges the cost to the block.
        On the corner grant `kick_spot`, whose two columns cannot carry anything there
        (delivery_x is 120 on all but two corners and delivery_y moves with the corner_side
        still in the design), draws a positive permuted Δ against a null dropped one: that is
        the method's own noise floor (evidence.md §17). `delivery` is the block where the two
        agree. A permuted Δ is evidence only where the dropped Δ agrees with it.

        `within_class` is test 2 of Ojala & Garriga (JMLR 11 (2010) 1833-1863) and its null is
        narrow: permuting inside a class preserves that class's marginal, so a block that
        predicts through marginals alone survives untouched. `global` answers the ablation
        question; `within_match` asks whether the block stands in for match identity.

        Δ is averaged over `n_repeats` permutations BEFORE the interval, so the bootstrap
        prices the resampling of matches and not the draw of one permutation."""
        y = table["target"].to_numpy()
        groups = table["match_id"].to_numpy()
        base = self._row_logloss(self.fit_cv(table), y)
        rows = []
        for grp, cols in _BLOCKS.items():
            cols = [c for c in cols if c in table.columns]
            if not cols:
                continue
            acc = np.zeros(len(table))
            for r in range(n_repeats):
                t = shuffle_features(table, cols, y=y, groups=groups, mode=mode, seed=seed + r)
                acc += self._row_logloss(self.fit_cv(t), y) - base
            d = acc / n_repeats                          # +delta = permuting it hurt = it helped
            ci = (cluster_bootstrap_mean(d, groups, n_boot=n_boot, seed=seed) if n_boot
                  else None)
            rows.append({"group": grp, "n_cols": len(cols), "delta": float(d.mean()),
                         "lo": ci.lo if ci else float("nan"),
                         "hi": ci.hi if ci else float("nan"),
                         "n_boot": ci.n_boot if ci else 0, "mode": mode,
                         "n_repeats": n_repeats,
                         "p_wilcoxon": wilcoxon_paired(d, groups)["p_value"]})
        return pd.DataFrame(rows).sort_values("delta", ascending=False).reset_index(drop=True)

    def noise_floor(self, table: pd.DataFrame, widths: list | None = None, n_draws: int = 30,
                    n_repeats: int = 3, seed: int = 0, n_jobs: int = 1) -> pd.DataFrame:
        """What shuffling a block that carries nothing costs, measured on blocks of pure noise.

        `permuted_ablation` charges a block the cost of shuffling it across rows, and that cost is not zero for
        a block with no information: the head has put some weight on the columns it was fitted with, and a
        shuffle moves it. A floor read off a block that cannot carry anything because of what the data happen
        to be (`kick_spot` on corners struck from the flag) stops being a floor when the data change, so this
        builds it by construction. For each width, `n_draws` independent blocks of standard-normal columns are
        appended to a copy of the table and named in `numeric=`, so the design, its regularisation and its
        folds are the published ones plus the decoy. A draw's base is that fit, and its cost is what
        `permuted_ablation` charges a block: the mean over `n_repeats` global permutations of the block of the
        paired row log loss of the permuted fit less the base's, averaged over rows.

        `widths` defaults to the width of every block of `_BLOCKS` present in the table, so each block is read
        against the floor of its own width: the more columns, the more noise there is to shuffle. One row per
        width comes back, with the mean, the sd, the 95th percentile (what a block of that width has to exceed
        to be told from noise) and the maximum of the draws' costs."""
        from joblib import Parallel, delayed, parallel_config
        if widths is None:
            widths = sorted({sum(c in table.columns for c in cols) for cols in _BLOCKS.values()} - {0})
        tasks = [(w, d) for w in widths for d in range(n_draws)]
        with parallel_config(backend="loky", inner_max_num_threads=1):
            costs = Parallel(n_jobs=n_jobs)(delayed(_noise_cost)(self, table, w, d, n_repeats, seed) for w, d in tasks)
        g = pd.DataFrame(tasks, columns=["width", "draw"]).assign(cost=costs).groupby("width")["cost"]
        return pd.DataFrame({"mean": g.mean(), "sd": g.std(ddof=1), "p95": g.quantile(0.95), "max": g.max(),
                             "n_draws": g.size()}).reset_index()

    def step_heterogeneity(self, table: pd.DataFrame, n_reps: int = 20, k_folds: int = 5,
                           n_bins: int = 5, seed: int = 0, placebo: bool = True) -> dict:
        """Does the 360 block help on SOME set pieces, where the pooled step cannot show it?

        A mean of zero is equally consistent with the block being useless everywhere and with
        it helping half the corners and hurting the other half. This asks which, globally, and
        BEFORE any cell is opened, so a verdict that finds nothing actionable is the REASON no
        cell gets opened rather than a gap where the reason should be.

        Everything is refit inside each fold: the per-row difference comes from two models fit
        on the other k-1 folds, and the proxy that ranks it from an INNER cross-fit of those
        same folds, so nothing that touches a fold ever trained on it. k=5 keeps training at
        the 80% the published ladder uses, which a 50/50 split would not, and the
        hyperparameters must match the ladder's too: an untuned head moves the mean step
        (decisiones_cerradas.md).

        Ridge on the 360 block rather than the multinomial head, because the proxy's job is to
        rank set pieces by how much the block helped, not to predict the outcome.

        The PLACEBO runs the same test with the proxy's features permuted against the rows, once
        globally and once within match, on the same partitions (Ojala & Garriga, JMLR 11, 2010,
        test 1). Whatever slope it finds is what the split-and-test produces with the link to the
        360 block broken, so the real slope is read against it. The within-match shuffle keeps
        each match's features in that match, so a slope that survives it lives between matches
        rather than between set pieces. The costly part, d and the training d the proxy fits,
        does not depend on the permutation and is computed once per partition for all three
        proxies; the permutations draw from their own generator, so the real result is the one
        `placebo=False` returns, bit for bit."""
        y = table["target"].to_numpy()
        idx = {c: i for i, c in enumerate(sorted(pd.unique(y)))}
        yi = np.array([idx[v] for v in y])
        g = table["match_id"].to_numpy()
        Xd, _ = self._design(table, numeric=_KICK_INTRINSIC, with_delivery_cats=True)
        Xf, _ = self._design(table)
        Xp, _ = self._design(table, numeric=_THREE_SIXTY, with_delivery_cats=False)
        modes = ("global", "within_match") if placebo else ()
        perm_rng = np.random.default_rng([seed, 7])

        def cross_fit(rng: np.random.Generator) -> tuple:
            matches = np.unique(g)
            fold_of = dict(zip(rng.permutation(matches), np.arange(len(matches)) % k_folds,
                               strict=True))
            fold = np.array([fold_of[v] for v in g])
            proxies = {"real": Xp, **{m: Xp[shuffle_labels(np.arange(len(y)), g, mode=m,
                                                         seed=int(perm_rng.integers(2**31)))]
                                      for m in modes}}
            d = np.zeros(len(y))
            dhat = {name: np.zeros(len(y)) for name in proxies}
            for k in range(k_folds):
                te, tr = np.flatnonzero(fold == k), np.flatnonzero(fold != k)
                assert_no_group_leak(g, tr, te, "match_id")

                def ll(X: np.ndarray, tr: np.ndarray = tr, te: np.ndarray = te) -> np.ndarray:
                    p = self._fit_score(X, y, tr, te, idx)
                    return -np.log(p[np.arange(len(te)), yi[te]].clip(1e-12))

                d[te] = ll(Xd) - ll(Xf)
                sub = table.iloc[tr]
                y_tr = sub["target"].to_numpy()
                d_tr = (self._row_logloss(self.fit_cv(sub, numeric=_KICK_INTRINSIC), y_tr)
                        - self._row_logloss(self.fit_cv(sub), y_tr))
                for name, P in proxies.items():
                    prep = self._fit_prep(P[tr])
                    dhat[name][te] = Ridge(alpha=1.0).fit(prep(P[tr]), d_tr).predict(prep(P[te]))
            return d, dhat

        # heterogeneity_blp draws one partition per rep from default_rng(seed); drawing them here from
        # the same generator in the same order keeps the real result bit-identical to the one-proxy call.
        part_rng = np.random.default_rng(seed)
        parts = [cross_fit(part_rng) for _ in range(n_reps)]

        def replay(name: str) -> Callable:
            stored = iter(parts)

            def next_partition(_rng: np.random.Generator) -> tuple:
                d, dhat = next(stored)
                return d, dhat[name]
            return next_partition

        out = heterogeneity_blp(g, replay("real"), n_reps=n_reps, n_bins=n_bins, seed=seed)
        d = (self._row_logloss(self.fit_cv(table, numeric=_KICK_INTRINSIC), y)
             - self._row_logloss(self.fit_cv(table), y))
        res = {**out, "observed_step": float(d.mean())}
        if placebo:
            keys = ("n_reps_used", "blp_slope", "blp_p", "gates_gap", "gates_p", "verdict")
            res["placebo"] = {
                m: {k: v for k, v in heterogeneity_blp(g, replay(m), n_reps=n_reps, n_bins=n_bins,
                                                        seed=seed).items() if k in keys}
                for m in modes}
        return res

    def step_cells(self, table: pd.DataFrame, min_cell: int = MIN_CELL) -> pd.DataFrame:
        """Does the 360 benefit line up with any cell a coach can NAME?

        Runs only after step_heterogeneity says the benefit varies at all, and answers the
        question that one cannot: it ranks set pieces by a ridge score, which nobody can act
        on. Each named cell gets a shrunken mean and the set gets a Q test, so a dimension
        that merely looks uneven is separated from one that is.

        The cells are the delivery-instant categoricals and NOTHING else. Binning by segment,
        origin or shot body part would condition on the outcome, and d is built from the
        outcome, so those are circular and barred rather than merely unhelpful. On free kicks
        swing is single-valued and drops out on its own.

        What it finds goes in `validation.<population>.cells`, which only a population whose
        benefit varies for someone gets (evidence.md §17 has the heterogeneity verdicts), and the
        number of dimensions examined travels with any claim made from them (Sun, Briel, Walter &
        Guyatt, BMJ 340:c117, 2010)."""
        y = table["target"].to_numpy()
        d = (self._row_logloss(self.fit_cv(table, numeric=_KICK_INTRINSIC), y)
             - self._row_logloss(self.fit_cv(table), y))
        g = table["match_id"].to_numpy()
        rows = []
        for col in _DELIVERY_CAT + ["league"]:
            if col not in table.columns or table[col].nunique() < 2:
                continue
            r = partial_pooling_cells(d, g, table[col].to_numpy(), min_cell=min_cell)
            rows.append({"dimension": col, "n_cells": int(r["n_cells_used"]),
                         "min_n": r.get("min_n", 0), "tau": r["tau"],
                         "i2": r.get("i2", float("nan")), "q": r.get("q", float("nan")),
                         "df": r.get("df", 0), "q_p": r["q_p"], "verdict": r["verdict"]})
        return pd.DataFrame(rows).sort_values("q_p").reset_index(drop=True)

    def step_rows(self, table: pd.DataFrame) -> np.ndarray:
        """Per-row log-loss of the delivery rung minus the full model's: the 360 step, positive
        where the 360 helped."""
        y = table["target"].to_numpy()
        return (self._row_logloss(self.fit_cv(table, numeric=_KICK_INTRINSIC), y)
                - self._row_logloss(self.fit_cv(table), y))

    def _head_to_head(self, table: pd.DataFrame) -> tuple[CVResult, CVResult, np.ndarray]:
        """(delivery fit, 360-only fit, per-row loss of the 360-only design minus the delivery's)."""
        y = table["target"].to_numpy()
        dcv = self.fit_cv(table, numeric=_KICK_INTRINSIC, with_delivery_cats=True)
        fcv = self.fit_cv(table, numeric=_THREE_SIXTY, with_delivery_cats=False)
        return dcv, fcv, self._row_logloss(fcv, y) - self._row_logloss(dcv, y)

    def head_to_head_rows(self, table: pd.DataFrame) -> np.ndarray:
        """Per-row difference behind `delivery_vs_360`, positive where the delivery design won."""
        return self._head_to_head(table)[2]

    def step_refit_interval(self, table: pd.DataFrame, n_boot: int = 300, seed: int = 0,
                            n_jobs: int = 1) -> dict:
        """The 360 step with the whole model REBUILT inside every resample.

        Every other interval here holds the fitted out-of-fold probabilities and resamples
        matches, so it covers evaluation-set sampling error and not the variance of having fit
        on a different sample of matches. An understated spread is an overstated power, which
        is the direction that flatters a published null, so `width_ratio` prices it rather than
        mentioning it, and the MDE beside it is optimistic by about that factor.

        Three readings travel together and `normal` decides. A resample keeps about 63% of the
        distinct matches, and that loss costs the WIDER rung more, so every draw sits below the
        observed step by the learning curve, not by an error of the estimate. The percentile
        form carries that shift and the pivotal one (`adjusted`) mirrors it to the other side,
        so both misplace the centre; `normal`, the observed step plus or minus half the refit
        width, keeps the centre and takes only the spread from the refit. It is the
        normal-theory interval when the draws are symmetric, which pipeline/run_refit_draws.py
        checks on the decisive contrasts. On the empty `kick_spot` block the pivotal form sits
        almost wholly above zero and the normal one is centred on it (evidence.md §17).

        Still not the whole variance, and the gap is named rather than left implied: each draw
        rebuilds the imputer, the scaler and the estimator but keeps the hyperparameters, so the
        search's own variance stays outside. The spread is expected to err wide instead, since
        every draw also fits on fewer distinct matches than the grant holds; that direction is
        argued, not measured (evidence.md §17)."""
        d = self.step_rows(table)
        fixed = cluster_bootstrap_mean(d, table["match_id"].to_numpy(), n_boot=2000, seed=seed)
        refit = refit_cluster_bootstrap(table, lambda sub: float(self.step_rows(sub).mean()),
                                        n_boot=n_boot, seed=seed, n_jobs=n_jobs)
        wf, wr = fixed.hi - fixed.lo, refit.hi - refit.lo
        point = float(d.mean())
        return {"point": point, "fixed": [fixed.lo, fixed.hi], "fixed_width": wf,
                "refit": [refit.lo, refit.hi], "refit_width": wr, "n_boot_refit": refit.n_boot,
                "width_ratio": wr / wf if wf else float("nan"),
                "adjusted": [2 * point - refit.hi, 2 * point - refit.lo],
                "normal": [point - wr / 2, point + wr / 2]}

    def ablation_refit_interval(self, table: pd.DataFrame, n_boot: int = 300, seed: int = 0,
                                n_jobs: int = 1) -> pd.DataFrame:
        """Every ablation Δ with the whole model REBUILT inside each resample, on shared draws.

        `ablate` holds the out-of-fold probabilities fixed, so its intervals are too narrow, and
        the step's widening from `step_refit_interval` cannot be borrowed for them: the widening
        differs block by block (evidence.md §11). One statistic returns the whole vector, so every
        block is scored on the same matches and each draw fits the base model once.

        Hyperparameters are held, and the three readings are the step's: `normal` decides,
        `adjusted` (pivotal) and the percentile travel beside it, for the reason
        `step_refit_interval` gives."""
        def deltas(sub: pd.DataFrame) -> np.ndarray:
            ys = sub["target"].to_numpy()
            base = self._row_logloss(self.fit_cv(sub), ys)
            return np.array([(self._row_logloss(self.fit_cv(sub, drop_group=g), ys) - base).mean()
                             for g in _BLOCKS])

        rf = refit_cluster_bootstrap(table, deltas, n_boot=n_boot, seed=seed, n_jobs=n_jobs)
        point, lo, hi = (np.atleast_1d(np.asarray(x, dtype=float)) for x in (rf.point, rf.lo, rf.hi))
        return pd.DataFrame({"group": list(_BLOCKS), "point": point, "refit_lo": lo,
                             "refit_hi": hi, "adjusted_lo": 2 * point - hi,
                             "adjusted_hi": 2 * point - lo, "normal_lo": point - (hi - lo) / 2,
                             "normal_hi": point + (hi - lo) / 2, "n_boot": rf.n_boot})

    def delivery_vs_360(self, table: pd.DataFrame, n_boot: int = 2000, seed: int = 0) -> dict:
        """The head-to-head the headline claim is about: the delivery design against all
        of the 360, PAIRED and intervalled. +delta = the delivery design won.

        A quotient of the two ladder steps would be the obvious way to say it and is barred: the
        360 step's corner interval covers zero, so the ratio is unstable by construction and
        reads a different number on every population. Fitting both designs and differencing them
        row by row says the same thing and carries an interval.

        The two models do NOT nest, so this is reported as a head-to-head, never as a step. The
        interval here is the fixed-fit one; the refit readings of the same difference come from
        pipeline/run_refit_draws.py (`measured.md` §validation, `delivery_vs_360.refit`)."""
        groups = table["match_id"].to_numpy()
        dcv, fcv, d = self._head_to_head(table)
        ci = cluster_bootstrap_mean(d, groups, n_boot=n_boot, seed=seed)
        return {"delivery_log_loss": dcv.log_loss, "only_360_log_loss": fcv.log_loss,
                "delta": float(d.mean()), "lo": ci.lo, "hi": ci.hi, "n_boot": ci.n_boot,
                "covers_zero": bool(ci.lo <= 0 <= ci.hi),
                "p_wilcoxon": wilcoxon_paired(d, groups)["p_value"]}

    def shuffled_control(self, table: pd.DataFrame, mode: str = "global",
                         n_repeats: int = 5, seed: int = 0) -> dict:
        """Refit the identical grouped CV on permuted labels; it must land at chance.

        Both modes are needed, not one. A global shuffle destroys the within-match
        correlation of y as well as the X->y link, so it CANNOT see a leak that travels
        through match identity, and a grouped CV is exactly what is exposed to that
        channel. `within_match` preserves each match's class composition and isolates
        it. A shuffled model that BEATS the marginal entropy means the harness itself
        leaks, and this is the only thing that would notice, because a leaking harness
        produces a perfectly plausible real number."""
        y = table["target"].to_numpy()
        groups = table["match_id"].to_numpy()
        p = pd.Series(y).value_counts(normalize=True).to_numpy()
        entropy = float(-(p * np.log(p)).sum())
        losses = []
        for r in range(n_repeats):
            t = table.copy()
            t["target"] = shuffle_labels(y, groups, mode=mode, seed=seed + r)
            losses.append(self.fit_cv(t).log_loss)
        mean = float(np.mean(losses))
        return {"mode": mode, "n_repeats": n_repeats, "seed": seed,
                "log_loss_mean": mean, "log_loss_std": float(np.std(losses)),
                "log_loss_real": self.fit_cv(table).log_loss,
                "marginal_entropy": entropy,
                "verdict": "clean" if mean >= entropy - 0.02 else "LEAKAGE SUSPECTED"}

    # ── Calibration diagnostics (shared binning for ece + reliability) ──
    @staticmethod
    def reliability_bins(y_idx: np.ndarray, proba: np.ndarray, bins: int = 10) -> pd.DataFrame:
        """Bin predictions by top-1 confidence; per bin report mean confidence,
        accuracy and count (the reliability-diagram / ECE table)."""
        conf = proba.max(axis=1)
        hit = (proba.argmax(axis=1) == y_idx).astype(float)
        edges = np.linspace(0, 1, bins + 1)
        rows = []
        for b, (lo, hi) in enumerate(zip(edges[:-1], edges[1:], strict=True)):
            m = (conf > lo) & (conf <= hi)
            rows.append({"bin": b,
                         "conf_mean": float(conf[m].mean()) if m.any() else float("nan"),
                         "acc": float(hit[m].mean()) if m.any() else float("nan"),
                         "count": int(m.sum())})
        return pd.DataFrame(rows)

    @classmethod
    def ece(cls, y_idx: np.ndarray, proba: np.ndarray, bins: int = 10) -> float:
        """Top-1 expected calibration error (weight = count_bin / n_total).

        WITHDRAWN as the headline by methods_second_ball.md §4 and kept for comparison
        only: it is discontinuous in the predictions and depends on where the bin edges
        happen to fall, so two runs of the same model can report different numbers.
        smooth_ece is the one to quote."""
        rb = cls.reliability_bins(y_idx, proba, bins)
        n = int(rb["count"].sum())
        if n == 0:
            return 0.0
        m = rb["count"] > 0
        return float((rb.loc[m, "count"] / n * (rb.loc[m, "acc"] - rb.loc[m, "conf_mean"]).abs()).sum())

    @staticmethod
    def _smooth_ece_at(conf: np.ndarray, hit: np.ndarray, sigma: float, m: int = 2048) -> float:
        """smECE_σ: total absolute mass of the kernel-smoothed residual (Błasiok & Nakkiran,
        Algorithm 1; sota/Blasiok_2024_SmoothECE).

        The kernel is REFLECTED, not plain. A plain Gaussian on [0, 1] thins the smoothed
        density near the two ends by about half, and a five-class head puts most of its top-1
        confidence right there, so the plain version would understate the error exactly where
        this model lives. Mirroring the histogram to period 2m and convolving periodically IS
        the reflected kernel, at the cost of one FFT."""
        res = hit - conf
        idx = np.clip(np.round(conf * m).astype(int), 0, m)
        h = np.bincount(idx, weights=res, minlength=m + 1)
        ext = np.concatenate([h, h[-2:0:-1]])            # length 2m, mirrored at both ends
        d = np.minimum(np.arange(2 * m), 2 * m - np.arange(2 * m)) / m   # circular distance, in [0,1] units
        k = np.exp(-0.5 * (d / sigma) ** 2)
        k /= k.sum()
        r = np.fft.irfft(np.fft.rfft(ext) * np.fft.rfft(k), n=2 * m)
        return float(np.abs(r[:m + 1]).sum() / len(conf))

    @classmethod
    def smooth_ece(cls, y_idx: np.ndarray, proba: np.ndarray, tol: float = 1e-4) -> float:
        """Kernel-smoothed calibration error at the self-consistent bandwidth (Algorithm 2).

        σ is not a hyperparameter: smECE_σ decreases in σ, so σ* = smECE_σ* is a unique fixed
        point found by bisection, and the reported error IS that bandwidth. A consistent
        calibration measure in the sense of Błasiok et al. 2023, unlike the binned ECE above.

        Applied to top-1 confidence against top-1 correctness, the same reduction `ece` uses,
        so the two numbers are directly comparable."""
        if len(y_idx) == 0:
            return 0.0
        return cls._smooth_ece_fixed_point(proba.max(axis=1),
                                           (proba.argmax(axis=1) == y_idx).astype(float), tol)

    @classmethod
    def _smooth_ece_fixed_point(cls, f: np.ndarray, y: np.ndarray, tol: float) -> float:
        """Bisect smECE_σ − σ, which is decreasing in σ and so has one root (Algorithm 2)."""
        lo, hi = 0.0, 1.0
        while hi - lo > tol:
            s = 0.5 * (lo + hi)
            if cls._smooth_ece_at(f, y, s) < s:
                hi = s
            else:
                lo = s
        return hi

    @classmethod
    def classwise_smooth_ece(cls, y_idx: np.ndarray, proba: np.ndarray,
                             classes: list, tol: float = 1e-4) -> pd.DataFrame:
        """One-vs-rest smoothECE per class, over the FULL probability range, plus its support.

        methods_second_ball.md §4: with several competing outcomes and heavy imbalance a good
        top-label number can hide a badly calibrated minority class. Top-1 only ever scores the
        class the model already picked, so it is structurally blind to exactly that: a class the
        model never ranks first contributes nothing to it at all.

        Scored on p_k against 1{y = k} directly. Routing this through the top-1 path would
        score max(p_k, 1-p_k) instead, which on a minority head is almost always the 1-p_k
        side and measures the wrong thing."""
        rows = []
        for k, c in enumerate(classes):
            p = proba[:, k]
            rows.append({"class": c, "n": int((y_idx == k).sum()),
                         "smooth_ece": cls._smooth_ece_fixed_point(p, (y_idx == k).astype(float), tol),
                         "mean_proba": float(p.mean()),
                         "base_rate": float((y_idx == k).mean())})
        return pd.DataFrame(rows)
