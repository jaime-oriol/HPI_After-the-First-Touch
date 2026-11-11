"""s05_causal — Stage 5 (CAUSAL): doubly-robust effect of the rest-defence decision.

Estimates the causal effect of the DEFENDING team's rest-defence choice (whether a
protector stands in the frontal, FIFA's edge of the box, src/zones.py) on what the corner
concedes, with cross-fitting (DML, Chernozhukov et al. 2018). Treatment
T = 1{n_def_frontal >= level}; outcome Y = conceded transition xG (continuous) or a binary
second-ball outcome. Both nuisances (propensity ρ(T|X), outcome μ(T,X)) are fit ONLY on
cross-fit training folds grouped by match, so no corner's estimate uses a model trained on
its own match.

X never holds the treatment nor what rebuilds it, whatever a specification names. The
zones partition the pitch, so s02's body total minus both zone blocks IS the frontal count,
and each per-body congestion form divides a kernel by that total. s03 keeps the total out of
its design; `_REBUILDS_TREATMENT` keeps the three forms out of every X here.

Three choices carry the inference:
  - the SE is the EIF cluster-robust one, clustered by match, with the CR1 small-sample
    correction and t_{G-1} critical values, and the cluster count G travels with it. A
    fixed-ψ bootstrap would ignore nuisance uncertainty;
  - μ is fit SEPARATELY per arm (μ1 on treated, μ0 on control) so the outcome model
    carries the full T×X interaction, which is what makes it doubly robust rather than
    additive in T;
  - degeneracy is flagged, not repaired: <2 match clusters, an empty arm, a single-class
    propensity fold, or a constant outcome all set degenerate=True with NaN effects
    rather than fabricating a number (conceded_xg is non-zero on almost no delivery of the
    grant, evidence.md §8).

THE ESTIMAND IS A CHOICE, AND ON THIS DATA THE ATE IS THE WRONG ONE. Measured on the
grant (data/derived/causal.parquet carries the figures), a large share of propensities
fall outside the usual [0.05, 0.95] band. That is positivity FAILING, not a tail to tidy,
and under it a doubly-robust ATE is the fragile estimator rather than the safe one: Yang,
Thomas & Li (2026, arXiv:2602.01648) show DR "generally amplifies the adverse consequences
of extreme weights ... regardless of model specifications, and is often inferior to both
the IPW and outcome modeling estimators", their remedy being to shift the target
population rather than to patch the weights.

So the estimand is explicit, via one tilting function h(e) that generalises all three
(Li, Morgan & Zaslavsky 2018; the weights are written out in Ben-Michael & Keele 2023,
Epidemiology 34(5):637-644, `sota/BenMichael_2023_BalancingWeights_ATT`):

    ate  h = 1            w1 = 1/e,   w0 = 1/(1-e)     unbounded: needs clipping
    att  h = e            w1 = 1,     w0 = e/(1-e)     the treated region
    ato  h = e(1-e)       w1 = 1-e,   w0 = e           BOUNDED BY CONSTRUCTION, no clipping at all

Clipping is not the fix. It caps an extreme weight while keeping the unit and the
estimand, so it hides the violation; moving to the ATO declares it. The clip applies
only to the two unbounded families, and `estimand="ato"` bypasses it entirely because
1-e and e already lie in [0, 1].

Ben-Michael & Keele's workflow is what run_causal reports: with poor overlap, the ATT and
the ATO estimated side by side off one cross-fit (`results`), where agreement is evidence
the ATT is identifiable and divergence means the ATO is the only estimand the data
supports. Two routes stay UNIMPLEMENTED, cited so the omission is a choice and not an oversight:
incremental propensity-score interventions (Kennedy 2019, JASA 114(526):645-656), which
need no positivity, and non-overlap ATE bounds (Susmann, McClean & Díaz 2026,
arXiv:2509.20206).

DEFINING THE ESTIMAND THROUGH THE PROPENSITY has three consequences, and the ATE, whose
h is constant, is exempt from all three:

  - the influence function carries a propensity term. With h depending on e, the moment
    E[h(e)(ψ - τ)] is NOT Neyman-orthogonal in e, so the EIF gains
    h'(e)(T - e)(μ1 - μ0 - τ) (Crump et al. 2009; Li et al. 2025, arXiv:2508.08167 eq. 5;
    Wang, Liu & Yang 2025, arXiv:2509.14502 Thm 3.1). Cross-fitting does not remove it:
    it is first order in (ê - e), not an overfitting remainder. The two terms are
    uncorrelated, so an SE without the second one is anticonservative under CATE
    heterogeneity;
  - double robustness is an ATE property. For the ATT and the ATO a misspecified
    propensity moves the TARGET POPULATION, so a correct outcome model cannot rescue the
    estimate (Tao & Fu 2019 §2.3);
  - the point estimate is the h-weighted mean and is NOT re-orthogonalised here. Solving
    the orthogonalised score would move the ATT and ATO points (for the ATT it reduces
    exactly to DML's ATTE score (5.4)).

The propensity is fit UNWEIGHTED, like every head in s03. Balancing optimises balanced
accuracy and deliberately mis-calibrates against the observed treatment mix, and these
weights are 1/e, so the mis-calibration lands straight in the estimate rather than in a
decision threshold nobody reads.

TEAM ENTERS HERE AND NOWHERE ELSE. Teams that commit players to the drop differ systematically,
and form_answers.md Q4 commits to reporting this effect with team and league controls.
`fixed_effects` adds them to X as dummies: the defending team, which is the side that
decides whether to hold a frontal protector, the attacking team it faces, and the league. Teams are
nested in leagues, and every nuisance is regularised, so the three enter together rather than one
standing in for the others. Each declares its source in `_FE_SOURCE`, as every design column in s03
does. Identity stays out of s03's predictive model; a fixed effect belongs to the causal question.
League enters here as a control as well; elsewhere it is a grouping key for league-out
validation, s02b's robustness covariate and a cell dimension of s03's heterogeneity gate, never
a feature of the predictive model.
s01 leaves `defending_team` empty when the defence never touches the ball in the sequence, so it is
read here as the other side of the same match, and a row still unresolved is its own `unknown` level.

WHAT LEAVES THIS STAGE IS A CURVE, NOT A SIGNED EFFECT. It publishes three estimands side by
side and one rung per feature group, because the sign of the ATE and of the ATT moves with the
confounder set and nothing in this data settles which rung is the right one. A claim about this
lever names the specification it came from and reads it off the artefact.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import LogisticRegression, TweedieRegressor
from sklearn.model_selection import GroupKFold

from s03_value import CompetingOutcomesModel, FEATURE_GROUPS, TARGET_MAP, TUNE_C_GRID, _NUMERIC
from validate import assert_no_group_leak

# Derived from TARGET_MAP, never written out: a stale class name here makes the outcome an
# all-zero vector, which the degeneracy guard reports as "no variation" and nothing else sees.
_BINARY_OUTCOMES = frozenset(TARGET_MAP.values())

TARGETS = ("ate", "att", "ato")
TREATMENT, TREATMENT_GROUP = "n_def_frontal", "frontal_def"
_REBUILDS_TREATMENT = ("congestion_spot_per_body", "congestion_drop_per_body",
                       "congestion_pairwise_per_pair")
_EXTREME_TOL = 0.10           # >10% of propensities in the tail = positivity is failing, not fraying
_FE_SOURCE = {"league": "match catalogue", "attacking_team": "team.name",
              "defending_team": "team.name (the other side of the match)"}


@dataclass
class OverlapReport:
    rho_min: float
    rho_max: float
    rho_treated_mean: float
    rho_control_mean: float
    n_clipped: int
    frac_extreme: float
    adequate: bool            # frac_extreme <= _EXTREME_TOL; False means the ATE is not the estimand to report


@dataclass
class Nuisances:
    """Cross-fitted ρ, μ1, μ0 and the regularisation each fold selected for itself."""
    rho: np.ndarray
    mu1: np.ndarray
    mu0: np.ndarray
    degenerate: bool
    c_prop: float
    c_out1: float
    c_out0: float


@dataclass
class CausalResult:
    estimand: str
    target: str               # "ate" | "att" | "ato": which population the weights tilt to
    treatment: str
    outcome: str
    confounders: str          # the feature groups held OUT of X, so a spec can be told from its twin
    n: int
    n_treated: int
    n_control: int
    n_clusters: int
    ess: float                # effective sample size of the tilted pseudo-population, (Sum h)^2 / Sum h^2
    effect_naive: float
    effect_ipw: float
    effect_aipw: float        # named for the estimator, never for the estimand: `target` says which
    se_aipw: float
    ci_aipw: tuple
    overlap: "OverlapReport | None"
    c_prop: float
    c_out1: float
    c_out0: float
    degenerate: bool
    note: str
    fixed_effects: str = "none"


class CausalRestDefence:
    """Cross-fitted AIPW estimator of the frontal protector's effect for one set-piece population."""

    def __init__(self, n_splits: int = 5, rho_floor: float = 0.05, C: float = 0.5,
                 min_arm: int = 3, target: str = "ato",
                 exclude_groups: tuple = (TREATMENT_GROUP,), C_grid: tuple | None = TUNE_C_GRID,
                 fixed_effects: tuple = ()) -> None:
        undeclared = [c for c in fixed_effects if c not in _FE_SOURCE]
        if undeclared:
            raise ValueError(f"fixed effects without a declared source: {undeclared}")
        self.fixed_effects = tuple(fixed_effects)
        self.n_splits = n_splits
        self.rho_floor = rho_floor
        self.C = C
        self.min_arm = min_arm
        self.target = target
        self.exclude_groups = tuple(exclude_groups)
        self.C_grid = C_grid

    # ── The estimand: one tilting function generalises all three ─────────
    @staticmethod
    def _tilt(rho: np.ndarray, target: str) -> np.ndarray:
        """h(e) selecting the target population (Li, Morgan & Zaslavsky 2018).

        Arm weights are ALWAYS h/e and h/(1-e), so this one function is the whole
        difference between an estimand whose weights explode as e -> 0 or 1 (ate,
        h = 1) and one whose weights cannot leave [0, 1] (ato, h = e(1-e))."""
        if target == "ate":
            return np.ones_like(rho)
        if target == "att":
            return rho
        if target == "ato":
            return rho * (1.0 - rho)
        raise ValueError(f"target must be one of {TARGETS}, got {target!r}")

    @staticmethod
    def _tilt_prime(rho: np.ndarray, target: str) -> np.ndarray:
        """dh/de, which is what makes the influence function below carry a propensity term.

        Zero only for the ATE. Everywhere else the estimand itself is defined through the
        propensity, so estimating e perturbs the target population and the variance has to
        say so."""
        if target == "ate":
            return np.zeros_like(rho)
        if target == "att":
            return np.ones_like(rho)
        if target == "ato":
            return 1.0 - 2.0 * rho
        raise ValueError(f"target must be one of {TARGETS}, got {target!r}")

    # ── Treatment / confounders / outcome ───────────────────────────────
    def _treatment(self, table: pd.DataFrame, level: int = 1) -> np.ndarray:
        """Binary treatment T = 1{at least `level` defenders in the frontal}."""
        return (table[TREATMENT].to_numpy(float) >= level).astype(int)

    def _confounders(self, table: pd.DataFrame) -> tuple:
        """s03's design matrix minus every feature group in `exclude_groups`.

        The treatment and the columns that rebuild it leave X whatever the tuple says:
        conditioning on T is the one thing that would make the estimate meaningless while
        still returning a finite plausible number. The tuple is the SPECIFICATION choice,
        which is why it is a parameter rather than a constant. The columns are named
        explicitly instead of routed through `drop_group`, which drops one group per call."""
        drop = {c for g in self.exclude_groups for c in FEATURE_GROUPS.get(g, ())}
        drop |= {TREATMENT, *_REBUILDS_TREATMENT}
        X, cols = CompetingOutcomesModel()._design(table, numeric=[c for c in _NUMERIC if c not in drop])
        if not self.fixed_effects:
            return X, cols
        fe = pd.get_dummies(_match_sides(table)[list(self.fixed_effects)], dtype=float)
        return np.c_[X, fe.to_numpy()], cols + list(fe.columns)

    def _outcome(self, table: pd.DataFrame, outcome: str) -> tuple:
        """Outcome vector + is_binary flag: a 1{target==name} indicator for a
        competing class, else the raw continuous column (e.g. conceded_xg).

        A name that is neither raises here rather than deeper: an s01 SEGMENT name
        (`after_clearance`) looks plausible but is not a TARGET_MAP class, and the
        silent failure would be a KeyError far from the caller that chose it."""
        if outcome in _BINARY_OUTCOMES:
            return (table["target"] == outcome).astype(float).to_numpy(), True
        if outcome not in table.columns:
            raise KeyError(
                f"outcome '{outcome}' is neither a competing class {sorted(_BINARY_OUTCOMES)} "
                f"nor a column of the table")
        return table[outcome].to_numpy(float), False

    # ── Nuisance regularisation, chosen INSIDE the fold that will not see it ──
    def _select_C(self, X: np.ndarray, y: np.ndarray, groups: np.ndarray) -> float:
        """Pick C by grouped-OOF log-loss on a TRAIN BLOCK, reusing s03's selector.

        Called from inside each cross-fitting fold on that fold's training rows only, so
        the regularisation is chosen without the evaluation fold in the average: the
        nested scheme DoubleML calls `tune_on_folds` and the one Definition 3.1 of
        Chernozhukov et al. (2018) describes, where the nuisance estimator may depend only
        on the complement of the fold it scores.

        s03's `_inner_select` rather than a second implementation: it already carries the
        convergence guard that drops a half-fitted config, the deterministic tie-break
        toward stronger regularisation, and `assert_no_group_leak` on every inner split.
        Its inner CV is the model's own `n_splits`, so the fold count travels through the
        constructor."""
        if self.C_grid is None:
            return self.C
        # C=self.C so a train block too small to split falls back to THIS estimator's default
        # rather than to s03's, which is the value _inner_select returns when no config scores.
        m = CompetingOutcomesModel(n_splits=self.n_splits, C=self.C)
        grid = [{"C": c, "class_weight": None, "penalty": "l2", "l1_ratio": None} for c in self.C_grid]
        return float(m._inner_select(X, y.astype(int), groups, {0: 0, 1: 1}, grid)["C"])

    # ── Nuisances by cross-fitting (DML core) ───────────────────────────
    def _fit_outcome(self, tf: Callable[[np.ndarray], np.ndarray], X: np.ndarray, y: np.ndarray,
                     arm_idx: np.ndarray, te_idx: np.ndarray, is_binary: bool,
                     C: float) -> np.ndarray:
        """Arm-specific outcome model μ_arm, predicted on the held-out fold."""
        if len(arm_idx) < self.min_arm or np.nanstd(y[arm_idx]) < 1e-9:
            return np.full(len(te_idx), float(np.mean(y[arm_idx])) if len(arm_idx) else 0.0)
        Xa, Xte = tf(X[arm_idx]), tf(X[te_idx])
        if is_binary:
            if len(np.unique(y[arm_idx])) < 2:
                return np.full(len(te_idx), float(y[arm_idx].mean()))
            m = LogisticRegression(max_iter=5000, C=C).fit(Xa, y[arm_idx].astype(int))
            return m.predict_proba(Xte)[:, list(m.classes_).index(1)]
        # Tweedie and NOT the gamma hurdle s03 uses. The two outcomes are different shapes:
        # s03 fits its second stage to the strictly positive subsample the gate left, while
        # this arm carries the zeros, which is exactly the mass a compound Poisson-gamma is
        # for. Swapping in a gamma here would fail on them.
        m = TweedieRegressor(power=1.5, alpha=0.1, max_iter=5000).fit(Xa, y[arm_idx])
        return m.predict(Xte).clip(0.0)

    def _arm_C(self, X: np.ndarray, y: np.ndarray, arm_idx: np.ndarray, groups: np.ndarray,
               is_binary: bool) -> float:
        """Regularisation for one arm's outcome model, selected on that arm's own rows.

        Per arm and not once on a pooled [X, T] design, because the pooled model is not the
        model that gets fitted: μ1 and μ0 are separate heads, so a C tuned on a third model
        is a C tuned on nothing that runs. Continuous outcomes take the Tweedie path, which
        has no C, and report NaN rather than a number that decided nothing."""
        if not is_binary:
            return float("nan")
        if len(arm_idx) < self.min_arm or len(np.unique(y[arm_idx])) < 2:
            return self.C
        return self._select_C(X[arm_idx], y[arm_idx], groups[arm_idx])

    def _crossfit(self, X: np.ndarray, T: np.ndarray, y: np.ndarray, groups: np.ndarray,
                  is_binary: bool) -> Nuisances:
        """Cross-fit the nuisances by match: on each fold select the regularisation and fit
        propensity ρ(T|X) and arm-specific outcomes μ1/μ0 on the OTHER matches, then predict
        here. No corner's estimate uses a model trained on its own match, and no fold's
        hyperparameter was chosen with that fold in view.

        Degeneracy is flagged rather than repaired on single-class or tiny-arm folds. The
        reported C is the modal choice across folds, ties to the smaller C (tune_cv breaks
        ties by first appearance); the fits themselves each use their own fold's."""
        n = len(y)
        rho = np.full(n, np.nan)
        mu1, mu0 = np.zeros(n), np.zeros(n)
        degen = False
        picks: dict = {"prop": [], "out1": [], "out0": []}
        for tr, te in GroupKFold(n_splits=min(self.n_splits, pd.Series(groups).nunique())).split(X, T, groups):
            assert_no_group_leak(groups, tr, te, "match_id")
            tf = CompetingOutcomesModel._fit_prep(X[tr])   # train-median impute + scale (no table-wide leak)
            if len(np.unique(T[tr])) < 2:
                rho[te] = float(T[tr].mean())
                degen = True
            else:
                c_prop = self._select_C(X[tr], T[tr], groups[tr])
                picks["prop"].append(c_prop)
                pm = LogisticRegression(max_iter=5000, C=c_prop).fit(tf(X[tr]), T[tr])
                rho[te] = pm.predict_proba(tf(X[te]))[:, list(pm.classes_).index(1)]
            tr1, tr0 = tr[T[tr] == 1], tr[T[tr] == 0]
            if len(tr1) < self.min_arm or len(tr0) < self.min_arm:
                degen = True
            c1, c0 = self._arm_C(X, y, tr1, groups, is_binary), self._arm_C(X, y, tr0, groups, is_binary)
            picks["out1"].append(c1)
            picks["out0"].append(c0)
            mu1[te] = self._fit_outcome(tf, X, y, tr1, te, is_binary, c1)
            mu0[te] = self._fit_outcome(tf, X, y, tr0, te, is_binary, c0)
        return Nuisances(rho, mu1, mu0, degen,
                         _modal(picks["prop"], self.C), _modal(picks["out1"], self.C),
                         _modal(picks["out0"], self.C))

    def _positivity(self, rho: np.ndarray, T: np.ndarray, target: str) -> tuple:
        """Overlap diagnostics, plus the clip only the UNBOUNDED families need.

        The ATO's weights are 1-e and e, already inside [0, 1], so it needs no clip
        and `n_clipped` is 0 there by construction rather than by luck. The report is
        produced for every target regardless, because checking overlap FIRST is what
        decides which estimand may be published at all (Yang et al. 2026)."""
        extreme = float(np.mean((rho < self.rho_floor) | (rho > 1 - self.rho_floor)))
        clipped = rho if target == "ato" else np.clip(rho, self.rho_floor, 1 - self.rho_floor)
        rep = OverlapReport(float(np.nanmin(rho)), float(np.nanmax(rho)),
                            float(rho[T == 1].mean()), float(rho[T == 0].mean()),
                            int((rho != clipped).sum()), extreme,
                            bool(extreme <= _EXTREME_TOL))
        return clipped, rep

    @staticmethod
    def _arm_weights(rho: np.ndarray, target: str) -> tuple:
        """(h/e, h/(1-e)) in CLOSED FORM, never as a ratio.

        Algebraically these are the same two numbers dividing `_tilt` by the propensity
        would give, and writing them out is what keeps the ATO bounded in the code and not
        only in the maths. The ATO is the one family that does not clip, so it can receive
        a propensity of exactly 0 or 1: there 1/(1-e) is inf and h = e(1-e) is 0, and
        0 x inf is NaN rather than the 0 the algebra gives. One such unit turns the whole
        effect into a NaN carried beside a 0.0000 SE and degenerate=False, which reads
        exactly like a measured null. The grant contains such units.

        The two unbounded families receive an already-clipped propensity, so their ratios
        are finite by the time they arrive."""
        if target == "ate":
            return 1.0 / rho, 1.0 / (1.0 - rho)
        if target == "att":
            return np.ones_like(rho), rho / (1.0 - rho)
        if target == "ato":
            return 1.0 - rho, rho
        raise ValueError(f"target must be one of {TARGETS}, got {target!r}")

    @classmethod
    def _ipw_effect(cls, T: np.ndarray, y: np.ndarray, rho: np.ndarray, target: str) -> float:
        """Self-normalised (Hájek) IPW over the tilted population. With h = 1 and a
        constant rho this equals the naive difference in means."""
        aw1, aw0 = cls._arm_weights(rho, target)
        w1, w0 = T * aw1, (1 - T) * aw0
        return float((w1 * y).sum() / w1.sum() - (w0 * y).sum() / w0.sum())

    @classmethod
    def _weighted_contrast(cls, T: np.ndarray, y: np.ndarray, rho: np.ndarray, mu1: np.ndarray,
                           mu0: np.ndarray, target: str) -> np.ndarray:
        """h(e) times the per-unit AIPW contrast: the tilted outcome difference plus the
        weighted residual correction, with both arm weights in closed form.

        Doubly robust for the ATE: right if EITHER rho or mu is. NOT for the ATT or the
        ATO: there h depends on e, so a misspecified propensity changes which population
        is being averaged over and a correct outcome model cannot recover the intended
        estimand (Tao & Fu 2019 §2.3)."""
        h = cls._tilt(rho, target)
        w1, w0 = cls._arm_weights(rho, target)
        return h * (mu1 - mu0) + w1 * T * (y - mu1) - w0 * (1 - T) * (y - mu0)

    @classmethod
    def _aipw_effect(cls, T: np.ndarray, y: np.ndarray, rho: np.ndarray, mu1: np.ndarray,
                     mu0: np.ndarray, target: str) -> float:
        """h-weighted mean of the AIPW contrast. The ATE recovers the plain AIPW."""
        return float(cls._weighted_contrast(T, y, rho, mu1, mu0, target).sum()
                     / cls._tilt(rho, target).sum())

    @classmethod
    def _influence_aipw(cls, T: np.ndarray, y: np.ndarray, rho: np.ndarray, mu1: np.ndarray,
                        mu0: np.ndarray, effect: float, target: str = "ate") -> np.ndarray:
        """Influence function of the h-tilted effect, per unit.

        Two terms. The estimator is a RATIO (an h-weighted mean over Sum h), so the delta
        method leaves the `- effect` term and the 1/mean(h) scale. The second term is the
        price of estimating the propensity the estimand is defined through:
        h'(e)(T - e)(μ1 - μ0 - τ). It vanishes for the ATE (h' = 0), and for the ATT and
        the ATO omitting it understates the spread, because the two terms are uncorrelated
        and the variances add (Crump et al. 2009; Li et al. 2025 eq. 5).

        At h = 1 and effect = 0 this is the bare AIPW residual, so the clustered CR1 spread
        below is the ATE's own EIF standard error."""
        h = cls._tilt(rho, target)
        hp = cls._tilt_prime(rho, target)
        contrast = cls._weighted_contrast(T, y, rho, mu1, mu0, target)
        return (contrast - h * effect + hp * (T - rho) * (mu1 - mu0 - effect)) / h.mean()

    # ── Public estimator ────────────────────────────────────────────────
    def estimate(self, table: pd.DataFrame, treatment_level: int = 1,
                 outcome: str = "conceded_xg", target: str | None = None) -> CausalResult:
        """Cross-fitted AIPW effect of holding a frontal protector on `outcome`, over
        the population `target` tilts to, with naive / IPW / AIPW estimates, a
        match-clustered CR1 SE and t_{G-1} CI, and overlap diagnostics. Guards return
        a degenerate result (NaN effects) rather than a fabricated number when the
        design can't support one."""
        return self.results(table, treatment_level, outcome,
                            (self.target if target is None else target,))[0]

    def _prepare(self, table: pd.DataFrame, treatment_level: int, outcome: str) -> tuple:
        """(T, y, groups, is_binary, refusal): the design's own admissibility check.

        A variance guard alone is not enough for a sparse continuous outcome: a handful
        of non-zero conceded_xg values gives a tiny-but-positive std, so the estimator
        would sail past and return +0.0000 with a zero SE, a number that reads like a
        finding and is really an empty cell. Hence the non-zero-count guard as well."""
        T = self._treatment(table, treatment_level)
        y, is_binary = self._outcome(table, outcome)
        groups = table["match_id"].to_numpy()
        G, n = int(pd.Series(groups).nunique()), len(y)
        nt, nc = int(T.sum()), int((T == 0).sum())
        refusal = None
        if G < 2:
            refusal = "fewer than 2 match clusters — cannot cross-fit"
        elif nt < self.min_arm or nc < self.min_arm:
            refusal = f"an arm is too small (treated={nt}, control={nc})"
        elif np.nanstd(y) < 1e-9:
            refusal = f"outcome '{outcome}' has no variation on this substrate (e.g. no counters)"
        else:
            n_nz = int(np.count_nonzero(np.nan_to_num(y)))
            if n_nz < max(self.min_arm, 10):
                refusal = (f"outcome '{outcome}' is non-zero on only {n_nz} rows of {n} "
                           f"— too sparse to estimate, reported rather than fabricated")
        return T, y, groups, is_binary, refusal

    def _label(self, treatment_level: int, outcome: str, target: str) -> tuple:
        """(estimand, treatment, confounders): how a row names the specification it ran."""
        return (f"{target.upper()}: frontal_protector>={treatment_level} -> {outcome}",
                f"frontal>={treatment_level}", "-".join(self.exclude_groups) or "none")

    def _fe_label(self) -> str:
        return "-".join(self.fixed_effects) or "none"

    def _degenerate(self, T: np.ndarray, y: np.ndarray, groups: np.ndarray, treatment_level: int,
                    outcome: str, target: str, note: str) -> CausalResult:
        estimand, treatment, conf = self._label(treatment_level, outcome, target)
        nan = float("nan")
        return CausalResult(estimand, target, treatment, outcome, conf,
                            len(y), int(T.sum()), int((T == 0).sum()),
                            int(pd.Series(groups).nunique()), nan,
                            nan, nan, nan, nan, (nan, nan), None, nan, nan, nan, True, note,
                            self._fe_label())

    def _effect(self, T: np.ndarray, y: np.ndarray, groups: np.ndarray, nu: Nuisances,
                target: str, treatment_level: int, outcome: str) -> CausalResult:
        """Turn cross-fitted nuisances into one estimand's effect, SE and interval.

        Separate from the cross-fit because ρ, μ1 and μ0 do not depend on the target:
        `results` gets three estimands out of ONE cross-fit rather than re-fitting the same
        nuisances three times."""
        estimand, treatment, conf = self._label(treatment_level, outcome, target)
        G, n = int(pd.Series(groups).nunique()), len(y)
        rho_c, overlap = self._positivity(nu.rho, T, target)
        h = self._tilt(rho_c, target)
        # Ben-Michael & Keele ask for the weighted population to be described alongside the
        # effect, because under the ATO the treated group is not fixed a priori.
        ess = float(h.sum() ** 2 / np.square(h).sum())

        eff_naive = float(y[T == 1].mean() - y[T == 0].mean())
        eff_ipw = self._ipw_effect(T, y, rho_c, target)
        eff_aipw = self._aipw_effect(T, y, rho_c, nu.mu1, nu.mu0, target)
        psi = self._influence_aipw(T, y, rho_c, nu.mu1, nu.mu0, eff_aipw, target)

        cl = pd.Series(psi).groupby(groups).sum().to_numpy()
        var = float((cl ** 2).sum()) * (G / (G - 1)) / n ** 2
        se = float(np.sqrt(var))
        ci_half = float(stats.t.ppf(0.975, G - 1)) * se
        note = ("SE is match-clustered CR1 with t_{G-1}, and the influence function carries "
                "the propensity term the tilted estimand needs. "
                + ("no clipping: the ATO's weights are bounded by construction. "
                   if target == "ato" else
                   f"propensities clipped at {self.rho_floor} (in-sample), which caps the "
                   f"weight without fixing the overlap. ")
                + ("degenerate fold(s) flagged — interpret with care."
                   if nu.degenerate else "machinery only, not a conclusion."))
        return CausalResult(estimand, target, treatment, outcome, conf,
                            n, int(T.sum()), int((T == 0).sum()), G, ess,
                            eff_naive, eff_ipw, eff_aipw, se,
                            (eff_aipw - ci_half, eff_aipw + ci_half), overlap,
                            nu.c_prop, nu.c_out1, nu.c_out0, nu.degenerate, note,
                            self._fe_label())

    def results(self, table: pd.DataFrame, treatment_level: int = 1,
                outcome: str = "conceded_xg", targets: tuple = TARGETS) -> list:
        """Every requested estimand off ONE cross-fit, as a list of CausalResult."""
        bad_target = [t for t in targets if t not in TARGETS]
        if bad_target:
            raise ValueError(f"target must be one of {TARGETS}, got {bad_target[0]!r}")
        T, y, groups, is_binary, bad = self._prepare(table, treatment_level, outcome)
        if bad is not None:
            return [self._degenerate(T, y, groups, treatment_level, outcome, t, bad) for t in targets]
        X, _ = self._confounders(table)
        nu = self._crossfit(X, T, y, groups, is_binary)
        return [self._effect(T, y, groups, nu, t, treatment_level, outcome) for t in targets]


def _match_sides(table: pd.DataFrame) -> pd.DataFrame:
    """league, attacking_team and defending_team as strings, the defender read off the match.

    s01 names the defending side only when it touches the ball in the sequence. In a match the
    defender is always the other side, so an empty row takes the one team of that match, seen in
    the table as attacker or defender, that is not attacking it. A match where that is not exactly
    one team stays `unknown`, an explicit level rather than a row silently sharing the baseline."""
    seen = pd.concat([table[["match_id", c]].set_axis(["match_id", "team"], axis=1)
                      for c in ("attacking_team", "defending_team")]).dropna()
    teams = seen.groupby("match_id")["team"].agg(set)
    other = [next(iter(teams.get(m, set()) - {a})) if len(teams.get(m, set()) - {a}) == 1
             else "unknown"
             for m, a in zip(table["match_id"], table["attacking_team"], strict=True)]
    return pd.DataFrame({"league": table["league"].astype(str),
                         "attacking_team": table["attacking_team"].astype(str),
                         "defending_team": table["defending_team"].where(
                             table["defending_team"].notna(), other).astype(str)},
                        index=table.index)


def _modal(values: list, fallback: float) -> float:
    """Most-frequent fold choice, ties broken toward stronger regularisation."""
    finite = [v for v in values if np.isfinite(v)]
    if not finite:
        return float("nan") if values else fallback
    counts = Counter(finite)
    top = max(counts.values())
    return float(min(v for v, k in counts.items() if k == top))
