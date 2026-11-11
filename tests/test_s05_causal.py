"""Tests for s05_causal — cross-fitted AIPW effect of a protector in the frontal.

All synthetic (no data needed): they pin the estimator's invariants and the
honest degeneracy guards rather than any football number.
"""
import numpy as np
import pandas as pd
import pytest

import s03_value
import s05_causal
from conftest import SameRowsSplit, counted_as_s02_counts
from s03_value import FEATURE_GROUPS, TARGET_MAP, _NUMERIC       # numeric confounder columns of the s03 design
from s05_causal import TARGETS, CausalRestDefence, CausalResult, OverlapReport

# Taken from TARGET_MAP, never written out: a stale class name makes the outcome an
# all-zero vector, which the degeneracy guard reports as "no variation", indistinguishable
# from a real constant.
_OUT = "after_first_touch"
_OTHER = sorted(set(TARGET_MAP.values()) - {_OUT})


def _synth(n: int = 90, seed: int = 0, p_frontal: float | None = None) -> pd.DataFrame:
    """Plausible modelling table: 9 matches, variable frontal defenders, an after-clearance
    signal, and an all-zero conceded_xg (the degenerate path).

    `p_frontal` overrides the frontal-count draw with P(frontal >= 1), which is what makes the
    treatment prevalence a knob the propensity tests can turn."""
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(n):
        feat = {c: float(rng.normal()) for c in _NUMERIC}
        frontal = (int(rng.integers(0, 3)) if p_frontal is None
                   else int(rng.random() < p_frontal) * int(rng.integers(1, 3)))
        feat["n_def_frontal"] = float(frontal)
        sp = 1 if rng.random() < (0.4 - 0.1 * frontal) else 0    # weakly reduces after_clearance
        tgt = _OUT if sp else str(rng.choice(_OTHER))
        rows.append({**feat, "match_id": i % 9, "target": tgt,
                     "swing": ["Inswinging", "Outswinging", "short/straight"][i % 3],
                     "corner_side": ["left", "right"][i % 2], "short_corner": bool(i % 2),
                     "conceded_xg": 0.0, "after_clearance_xg": 0.0})
    return pd.DataFrame(rows)


# ── Estimator invariants (pure) ─────────────────────────────────────────
def test_ipw_equals_naive_under_constant_rho():
    """With constant propensity and h=1, Hajek IPW collapses to the naive difference."""
    rng = np.random.default_rng(1)
    T = rng.integers(0, 2, size=100).astype(int)
    y = rng.normal(size=100)
    naive = y[T == 1].mean() - y[T == 0].mean()
    assert CausalRestDefence._ipw_effect(T, y, np.full(100, 0.5), "ate") == pytest.approx(naive)


def test_influence_zero_when_outcome_perfect():
    """A perfect outcome model (mu==y) zeroes the AIPW contrast and the influence function."""
    T, y, rho = np.array([1, 0, 1, 0]), np.array([1.0, 2, 3, 4]), np.full(4, 0.5)
    contrast = CausalRestDefence._weighted_contrast(T, y, rho, y, y, "ate")
    assert np.allclose(contrast, 0.0)                # mu1=mu0=y -> residuals vanish
    psi = CausalRestDefence._influence_aipw(T, y, rho, y, y, 0.0)
    assert np.allclose(psi, 0.0)                     # so the influence function is exactly zero too


# ── The estimand: the tilting families and what makes ATO safe ───────────
@pytest.mark.parametrize("target,w1,w0", [
    ("ate", lambda e: 1 / e, lambda e: 1 / (1 - e)),
    ("att", lambda e: np.ones_like(e), lambda e: e / (1 - e)),
    ("ato", lambda e: 1 - e, lambda e: e),
])
def test_tilt_reproduces_the_published_arm_weights(target, w1, w0):
    """h/e and h/(1-e) must be exactly the weights Ben-Michael & Keele (2023) write out.

    This is the whole mechanism: the estimand is chosen by one h(e), and everything
    downstream (IPW, AIPW, the influence function) is shared code. The estimator does not
    form those weights as a ratio, so the closed form it does use is pinned to the ratio
    here, on the interior where both agree."""
    e = np.linspace(0.02, 0.98, 40)
    h = CausalRestDefence._tilt(e, target)
    assert np.allclose(h / e, w1(e)) and np.allclose(h / (1 - e), w0(e))
    aw1, aw0 = CausalRestDefence._arm_weights(e, target)
    assert np.allclose(aw1, h / e) and np.allclose(aw0, h / (1 - e))


def test_the_ato_survives_a_propensity_of_exactly_one():
    """The estimand that never clips must not produce a NaN on the units that need it.

    The ATO's arm weights are 1-e and e, so a unit at e = 1 contributes 0 through the
    control arm and nothing breaks. Forming those weights as h/(1-e) instead computes
    inf and then 0 x inf = NaN, which propagates to the whole effect: one unit turns the
    estimate into a NaN carried alongside an SE of 0.0000 and degenerate=False, which
    reads exactly like a measured null. The grant contains such units, so this is the
    regime the estimator runs in and not a constructed edge case."""
    T = np.array([1, 0, 1, 0, 1])
    y = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    rho = np.array([0.5, 0.4, 1.0, 0.6, 0.0])        # two units exactly on the boundary
    mu1, mu0 = np.full(5, 0.5), np.full(5, 0.2)
    eff = CausalRestDefence._aipw_effect(T, y, rho, mu1, mu0, "ato")
    assert np.isfinite(eff)
    psi = CausalRestDefence._influence_aipw(T, y, rho, mu1, mu0, eff, "ato")
    assert np.isfinite(psi).all()
    assert np.isfinite(CausalRestDefence._ipw_effect(T, y, rho, "ato"))


def test_ato_weights_are_bounded_but_ate_weights_are_not():
    """Why the ATO needs no clipping and the ATE does.

    The ATO's arm weights are 1-e and e, so they cannot leave [0, 1] no matter how
    extreme the propensity gets. Clipping the ATE caps the weight while keeping the
    unit and the estimand, which hides the positivity violation instead of declaring
    it (methods_second_ball.md §2.5)."""
    e = np.array([1e-4, 0.5, 1 - 1e-4])              # two units effectively off the overlap region
    h_ato = CausalRestDefence._tilt(e, "ato")
    assert (h_ato / e <= 1.0).all() and (h_ato / (1 - e) <= 1.0).all()
    h_ate = CausalRestDefence._tilt(e, "ate")
    assert (h_ate / e).max() > 1e3                   # the ATE weight explodes on the same input


def test_h_one_reproduces_the_plain_aipw_ate():
    """The generalisation must not move the number the ATE already reported.

    At h=1 the tilted estimator has to equal mean((mu1-mu0) + T/e(Y-mu1) - (1-T)/(1-e)(Y-mu0)),
    which is what this module computes when the estimand is the ATE."""
    rng = np.random.default_rng(5)
    n = 300
    rho = rng.uniform(0.05, 0.95, n)
    T = (rng.random(n) < rho).astype(int)
    y, mu1, mu0 = rng.normal(size=n), rng.normal(size=n), rng.normal(size=n)
    plain = ((mu1 - mu0) + T / rho * (y - mu1) - (1 - T) / (1 - rho) * (y - mu0)).mean()
    assert CausalRestDefence._aipw_effect(T, y, rho, mu1, mu0, "ate") == pytest.approx(plain)


def test_tilt_prime_is_the_derivative_of_the_tilt():
    """h' has to be dh/de and not a hand-copied expression, or the correction below is noise."""
    e = np.linspace(0.05, 0.95, 25)
    step = 1e-6
    for target in TARGETS:
        num = (CausalRestDefence._tilt(e + step, target)
               - CausalRestDefence._tilt(e - step, target)) / (2 * step)
        assert np.allclose(CausalRestDefence._tilt_prime(e, target), num, atol=1e-5)


def test_the_tilted_influence_carries_the_propensity_term():
    """The ATT and the ATO are defined THROUGH the propensity, so the EIF gains a term.

    h'(e)(T-e)(mu1-mu0-tau) is first order in (e_hat - e), not an overfitting remainder
    that cross-fitting removes (Crump et al. 2006; Li et al. 2025 arXiv:2508.08167 eq. 5).
    h' is 0 for the ATE, which is the only reason the two ATE tests above are untouched
    by it, and that is asserted rather than assumed.

    What is pinned here is the decomposition, exactly. That the omission understates the
    interval is an asymptotic statement about the two terms being uncorrelated in
    EXPECTATION, so it does not hold sample by sample and is not asserted from one draw; a
    coverage simulation would measure it, and none is run here."""
    rng = np.random.default_rng(11)
    n = 400
    rho = rng.uniform(0.1, 0.9, n)
    T = (rng.random(n) < rho).astype(int)
    y, mu1, mu0 = rng.normal(size=n), rng.normal(size=n), rng.normal(size=n)

    for target in TARGETS:
        h = CausalRestDefence._tilt(rho, target)
        hp = CausalRestDefence._tilt_prime(rho, target)
        eff = CausalRestDefence._aipw_effect(T, y, rho, mu1, mu0, target)
        psi = CausalRestDefence._influence_aipw(T, y, rho, mu1, mu0, eff, target)
        fixed_h = (CausalRestDefence._weighted_contrast(T, y, rho, mu1, mu0, target)
                   - h * eff) / h.mean()
        extra = hp * (T - rho) * (mu1 - mu0 - eff) / h.mean()
        assert np.allclose(psi, fixed_h + extra)
        if target == "ate":
            assert np.allclose(psi, fixed_h)          # h' = 0, so the ATE carries no propensity term
        else:
            assert not np.allclose(psi, fixed_h)

    # A homogeneous treatment effect kills the new term for EVERY target: mu1 = y + c(1-T)
    # and mu0 = y - cT leave both arms' residuals at zero and a per-unit contrast of exactly
    # c, so the estimate is c and both halves of the influence function vanish. That is what
    # says the term prices CATE heterogeneity and nothing else.
    c = 0.4
    mu1_flat, mu0_flat = y + c * (1 - T), y - c * T
    for target in TARGETS:
        eff = CausalRestDefence._aipw_effect(T, y, rho, mu1_flat, mu0_flat, target)
        assert eff == pytest.approx(c)
        psi = CausalRestDefence._influence_aipw(T, y, rho, mu1_flat, mu0_flat, eff, target)
        assert np.allclose(psi, 0.0)


def test_unknown_target_raises():
    """A typo must fail loudly, not silently fall back to the ATE."""
    with pytest.raises(ValueError, match="target must be one of"):
        CausalRestDefence._tilt(np.full(3, 0.5), "atx")
    with pytest.raises(ValueError, match="target must be one of"):
        CausalRestDefence._tilt_prime(np.full(3, 0.5), "atx")
    with pytest.raises(ValueError, match="target must be one of"):
        CausalRestDefence().estimate(_synth(), outcome=_OUT, target="atx")


def test_treatment_binarises():
    """The treatment threshold binarises the frontal count at the requested level."""
    t = pd.DataFrame({"n_def_frontal": [0, 1, 2, 3]})
    assert list(CausalRestDefence()._treatment(t, 1)) == [0, 1, 1, 1]   # T = 1{frontal >= 1}
    assert list(CausalRestDefence()._treatment(t, 2)) == [0, 0, 1, 1]   # T = 1{frontal >= 2}


def test_confounders_can_exclude_any_feature_group():
    """The confounder set is a SPECIFICATION, and the module reports more than one.

    The congestion block reached X because s03's design is built from FEATURE_GROUPS, not
    because anyone judged its columns to be confounders. Treatment and congestion are both
    read in the delivery freeze-frame, so neither precedes the other and the data cannot
    settle whether they are confounders or mediators. What the module can do is run both
    specifications, which is why exclusion is parameterised."""
    _, kept = CausalRestDefence()._confounders(_synth())
    _, dropped = CausalRestDefence(exclude_groups=("frontal_def", "congestion"))._confounders(_synth())
    cong = set(FEATURE_GROUPS["congestion"]) - set(s05_causal._REBUILDS_TREATMENT)
    assert cong <= set(kept)
    for col in cong:
        assert col not in dropped and f"{col}_isna" not in dropped
    assert set(kept) - set(dropped) == cong          # exactly that group leaves, nothing else


# What rebuilds the treatment, named here and not read from s05: the frontal count, the body total
# (every attacker and defender s02 counts, so it minus both zone blocks is the frontal count) and
# the three forms that divide a kernel by that total.
_REBUILDS_T = {"n_def_frontal", "congestion_bodies", "congestion_spot_per_body",
               "congestion_drop_per_body", "congestion_pairwise_per_pair"}


@pytest.mark.parametrize("exclude", [(), ("frontal_def",)])
def test_no_confounder_set_can_rebuild_the_treatment(exclude):
    """The treatment leaves X whatever a specification names, and so does all that rebuilds it.

    Only the fullest design is checked, because every specification drops more groups than it and
    is a subset of it. By name, because positivity fails for ANY function of X and not only the
    linear ones; and by a linear fit, which is what a logistic propensity would exploit."""
    t = counted_as_s02_counts(_synth())
    X, cols = CausalRestDefence(exclude_groups=exclude)._confounders(t)
    assert not _REBUILDS_T & {c.removesuffix("_isna") for c in cols}
    assert set(FEATURE_GROUPS["zones_def"]) <= set(cols)    # the rest of the design is still there
    A = np.c_[np.ones(len(X)), X]
    y = t["n_def_frontal"].to_numpy(float)
    resid = y - A @ np.linalg.lstsq(A, y, rcond=None)[0]
    assert np.abs(resid).max() > 0.1, "the confounders rebuild the frontal count"


_FE = ("league", "attacking_team", "defending_team")


def _synth_teams() -> pd.DataFrame:
    """The main fixture with a league and the two sides of every match, one defender left empty
    the way s01 leaves it when the defence never touches the ball."""
    t = _synth()
    home, away = t["match_id"].map(lambda m: f"H{m}"), t["match_id"].map(lambda m: f"A{m}")
    t["league"] = np.where(t["match_id"] < 5, "l1", "l2")
    t["attacking_team"] = np.where(t.index % 2 == 0, home, away)
    t["defending_team"] = np.where(t.index % 2 == 0, away, home)
    t.loc[t.index[0], "defending_team"] = None
    return t


def test_fixed_effects_must_declare_their_source():
    """A control with no declared field is refused before any work, as every design column is."""
    with pytest.raises(ValueError, match="without a declared source"):
        CausalRestDefence(fixed_effects=("player",))
    assert set(s05_causal._FE_SOURCE) == set(_FE)
    assert not any(b in s for s in s05_causal._FE_SOURCE.values() for b in s03_value._BARRED_SOURCES)


def test_fixed_effects_widen_the_design_by_their_levels():
    """Team and league enter as one dummy per level and nothing else moves, and an empty defender
    is read as the other side of its own match rather than dropped or pooled with the baseline."""
    t = _synth_teams()
    X0, n0 = CausalRestDefence()._confounders(t)
    X1, n1 = CausalRestDefence(fixed_effects=_FE)._confounders(t)
    assert n1[:len(n0)] == n0 and np.allclose(X1[:, :len(n0)], X0, equal_nan=True)
    assert X1.shape[1] - X0.shape[1] == sum(t[c].nunique() for c in _FE)
    sides = s05_causal._match_sides(t)
    assert sides.loc[t.index[0], "defending_team"] == "A0"
    assert (sides["defending_team"] != sides["attacking_team"]).all()
    r = CausalRestDefence(fixed_effects=_FE).estimate(t, outcome=_OUT)
    assert r.fixed_effects == "-".join(_FE) and np.isfinite(r.effect_aipw)


# ── Nuisances: unweighted, and tuned without seeing the evaluation fold ──
def test_propensity_is_not_balanced():
    """rho has to be P(T=1|X), because these weights are 1/e.

    class_weight="balanced" optimises balanced accuracy and deliberately mis-calibrates
    against the observed treatment mix, pulling the fitted propensity toward 0.5 whatever
    the prevalence. In a classifier that shifts a threshold nobody reads; here it lands
    straight in the weights and therefore in the estimate. On this deliberately skewed
    fixture a balanced propensity averages near 0.5 while the observed rate is ~0.8."""
    t = _synth(n=270, seed=3, p_frontal=0.8)
    m = CausalRestDefence()
    T = m._treatment(t)
    base = float(T.mean())
    assert base > 0.7, "the fixture stopped being skewed, so this test proves nothing"
    X, _ = m._confounders(t)
    y, is_binary = m._outcome(t, _OUT)
    rho = m._crossfit(X, T, y, t["match_id"].to_numpy(), is_binary).rho
    assert abs(rho.mean() - base) < abs(rho.mean() - 0.5)


def test_nested_tuning_never_sees_the_evaluation_fold(monkeypatch):
    """The regularisation is chosen inside the fold, on the training block only.

    Tuning once over the whole table and then cross-fitting with the SAME GroupKFold picks
    a C whose score already includes the rows it will be evaluated on, and the published
    number carries that. The check is on the groups each selection call actually sees:
    every call sees a strict subset of the matches, and every match is invisible to at
    least one call."""
    seen = []
    original = CausalRestDefence._select_C

    def spy(self, X, y, groups):
        seen.append(set(np.unique(groups)))
        return original(self, X, y, groups)

    monkeypatch.setattr(CausalRestDefence, "_select_C", spy)
    t = _synth()
    all_matches = set(t["match_id"].unique())
    r = CausalRestDefence().estimate(t, outcome=_OUT)

    assert seen, "no regularisation was selected at all"
    assert all(s < all_matches for s in seen), "a selection call saw every match"
    for m in all_matches:
        assert any(m not in s for s in seen), f"match {m} was in every selection call"
    assert r.c_prop in CausalRestDefence().C_grid


@pytest.mark.parametrize("module", ["s05_causal", "s03_value"])
def test_both_grouped_splits_assert_disjointness(monkeypatch, module):
    """The guard has to EXECUTE on both of this stage's splits, not merely be imported.

    s05 owns two: the outer cross-fit here, and the inner selection split it borrows from
    s03. Handing either a splitter whose train and test are the same rows puts every match
    on both sides, so a path that skipped the guard would fit happily and return a number
    with nothing in the suite able to tell."""
    monkeypatch.setattr({"s05_causal": s05_causal, "s03_value": s03_value}[module],
                        "GroupKFold", SameRowsSplit)
    with pytest.raises(AssertionError, match="match_id leaks"):
        CausalRestDefence().estimate(_synth(), outcome=_OUT)


# ── Degeneracy guards (fail honest) ─────────────────────────────────────
def test_degenerate_constant_outcome():
    """A constant (all-zero) outcome reports degenerate, not a fabricated effect."""
    r = CausalRestDefence().estimate(_synth(), outcome="conceded_xg")
    assert r.degenerate and np.isnan(r.effect_aipw)


def test_degenerate_single_match():
    """A single match cluster can't cross-fit -> degenerate."""
    t = _synth()
    t["match_id"] = 0
    r = CausalRestDefence().estimate(t, outcome=_OUT)
    assert r.degenerate and np.isnan(r.effect_aipw)


# ── Full estimate ───────────────────────────────────────────────────────
@pytest.mark.parametrize("target", TARGETS)
def test_estimate_runs(target):
    """Every target yields finite effects, a valid CI and an overlap report."""
    r = CausalRestDefence().estimate(_synth(), outcome=_OUT, target=target)
    assert isinstance(r, CausalResult) and r.target == target
    assert r.n_clusters >= 2 and np.isfinite(r.effect_aipw) and np.isfinite(r.se_aipw)
    assert len(r.ci_aipw) == 2 and r.ci_aipw[0] <= r.effect_aipw <= r.ci_aipw[1]
    assert isinstance(r.overlap, OverlapReport)
    assert 0.0 <= r.overlap.rho_min <= r.overlap.rho_max <= 1.0
    assert 0 < r.ess <= r.n                          # the tilted pseudo-population cannot exceed the sample
    # the ATO never clips, so a clipped unit there would mean the bypass broke
    assert r.overlap.n_clipped == 0 if target == "ato" else r.overlap.n_clipped >= 0


def test_default_target_is_the_ato():
    """The default is the estimand this data identifies, not the ATE.

    A large share of propensities fall outside [0.05, 0.95] on the grant, so the ATE is not
    identified here; the figures live in data/derived/causal.parquet rather than in this
    docstring, where they would rot. Pinned so the default cannot drift back to an unidentified
    estimand."""
    assert CausalRestDefence().target == "ato"
    assert CausalRestDefence().estimate(_synth(), outcome=_OUT).target == "ato"


def _synth_continuous(n: int = 90, seed: int = 1) -> pd.DataFrame:
    """Same table, but with a genuinely varying conceded_xg."""
    t = _synth(n, seed)
    rng = np.random.default_rng(seed + 100)
    t["conceded_xg"] = np.where(rng.random(len(t)) < 0.25, rng.gamma(2.0, 0.05, len(t)), 0.0)
    return t


def test_continuous_outcome_path_runs():
    """The Tweedie arm of _fit_outcome actually executes.

    The main fixture's conceded_xg is identically zero, so on it the module's DEFAULT
    outcome stops at the degeneracy guard and the continuous branch of `_fit_outcome` never
    runs. This fixture is what reaches it. That branch has no regularisation to select, so
    its reported C is NaN rather than a number that decided nothing."""
    r = CausalRestDefence().estimate(_synth_continuous(), outcome="conceded_xg")
    assert not r.degenerate
    assert np.isfinite(r.effect_aipw) and np.isfinite(r.se_aipw)
    assert isinstance(r.overlap, OverlapReport)
    assert r.ci_aipw[0] <= r.effect_aipw <= r.ci_aipw[1]
    assert np.isnan(r.c_out1) and np.isnan(r.c_out0)


def test_outcome_rejects_a_name_that_is_neither_class_nor_column():
    """An s01 SEGMENT name is not a TARGET_MAP class, and must fail loudly.

    `after_clearance` reads like a valid outcome but is a segment, so it is neither
    in TARGET_MAP.values() nor a column of the corner table. Raising here rather than
    deeper is the whole point: a KeyError inside the cross-fit names a column, not the
    caller's choice of outcome."""
    m = CausalRestDefence()
    with pytest.raises(KeyError, match="neither a competing class"):
        m._outcome(_synth(), "after_clearance")
