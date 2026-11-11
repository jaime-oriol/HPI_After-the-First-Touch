"""Tests for s03_value — competing-outcomes value model (VALUE stage).

A synthetic multi-match table exercises the full harness (CV, hurdle, ablations,
xSecondBall) offline; a real-sample test confirms the table builder joins s01+s02.
"""
import numpy as np
import pandas as pd
import pytest

import loader_sb as L
import s03_value
from s01_phases import OUTCOMES, SetPieceLabel
from s03_value import (
    CVResult, CalibrationResult, CompetingOutcomesModel, FEATURE_GROUPS, GroupOutResult,
    TARGET_MAP, TuneResult, build_corner_table, _LABEL_COLS, _NUMERIC,
)
from conftest import SameRowsSplit, counted_as_s02_counts, requires_sb

_CLASSES = sorted(set(TARGET_MAP.values()))          # the competing classes, straight from the map


def _synth(n: int = 72, seed: int = 0) -> pd.DataFrame:
    """A plausible modelling table: 7 matches, every competing class present.

    The match count is coprime with the class count on purpose. With both at 6 the
    class became a function of the match, so every GroupKFold fold held out a class
    entirely, _align zero-filled it and the log-loss pinned at -log(1e-12) = 27.63 for
    every model, which made every ablation delta exactly 0.0 and looked like the
    features doing nothing rather than a broken fixture."""
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(n):
        feat = {c: float(rng.normal()) for c in _NUMERIC}
        mid = i % 7
        rows.append({
            **feat, "match_id": mid, "target": _CLASSES[i % len(_CLASSES)],
            "swing": ["Inswinging", "Outswinging", "short/straight"][i % 3],
            "corner_side": ["left", "right"][i % 2], "short_corner": bool(i % 2),
            "after_clearance_xg": float(max(0.0, rng.normal(0.05, 0.05))) if i % 3 == 0 else 0.0,
            "conceded_xg": float(max(0.0, rng.normal(0.04, 0.05))) if i % 7 == 0 else 0.0,
            # post-delivery label columns: present so the anti-leakage guard has
            # something real to exclude, never modelled
            "outcome": "shot_after_clearance" if i % 5 == 0 else "clear_safe",
            "cleared": bool(i % 4), "second_ball_time_s": float(i % 8),
            # derived from match_id, never from i: a match belongs to exactly ONE league
            "league": ["premier_league", "la_liga", "serie_a"][mid % 3],
        })
    return pd.DataFrame(rows)


# ── Static checks (no data) ─────────────────────────────────────────────
def test_target_map_is_total_over_the_outcomes():
    """TARGET_MAP is a function on OUTCOMES: exact set equality, both directions.

    Set equality and not membership: an outcome added to s01 but not to the map would
    resolve to NaN and become a silent null class."""
    from dataclasses import fields
    assert "outcome" in {f.name for f in fields(SetPieceLabel)}
    assert set(TARGET_MAP) == set(OUTCOMES)


def test_the_class_without_a_second_ball_is_named_for_it():
    """The class holds deliveries the attack reached first and kept, untouched ones, balls cleared
    out of play and balls the keeper took: mostly not the defence regaining anything, which a name
    like `regain` would claim. It is named for what they share, no second ball (definitions.md
    §19)."""
    assert TARGET_MAP["no_second_ball"] == "no_second_ball"
    assert "regain" not in set(TARGET_MAP.values()) and "regain_restart" not in TARGET_MAP


def test_the_plan_block_is_the_switch_s_zones_with_the_defending_frontal_apart():
    """The plan block is the zones of the partition the switch names, attackers and defenders, with
    the defenders in the frontal as a group of their own, because they are s05's treatment and s05
    has to leave them out alone. The three jobs, the box counts and the central edge band stay out
    of the design: with width, each is a sum of the zones."""
    import zones
    assert FEATURE_GROUPS["zones_att"] == zones.zone_cols("att")
    assert FEATURE_GROUPS["zones_def"] == zones.zone_cols("def")
    assert FEATURE_GROUPS["frontal_def"] == ["n_def_frontal"]
    assert not {"signature", "box", "edge"} & set(FEATURE_GROUPS)


def test_a_named_column_the_table_lacks_stops_the_design():
    """A whitelist that drops a missing name in silence fits a smaller design than the one the
    prose describes, and a table without the zone columns would lose the whole plan block that way."""
    m = CompetingOutcomesModel()
    t = _synth().drop(columns=["marking_cost"])
    with pytest.raises(ValueError, match="does not carry"):
        m._design(t)


def test_counter_is_not_a_class_and_lambda_is_zero():
    """The counter arm's decision, pinned so it cannot drift back by accident.

    The transition arm is not estimable: swept over every cleared delivery, the possession
    chain from the clearance clears the 95% contiguity bar at no window width at all
    (measured.md §counter carries the sweep, and the figures live there rather than here so
    they cannot rot in a docstring nothing binds). So clear_counter folds into clear_safe and
    lambda_counter is 0.0: a non-zero lambda would subtract a term built mostly from other
    teams' next attacks."""
    assert "counter" not in set(TARGET_MAP.values())
    assert TARGET_MAP["clear_counter"] == "clear_safe"
    assert CompetingOutcomesModel().lambda_counter == 0.0
    m = CompetingOutcomesModel(n_splits=4)
    assert np.allclose(m.xsecondball(_synth()), m.hurdle_xg(_synth(), "after_clearance_xg"))


def test_density_is_barred_from_the_design():
    """att_density_5/10 are LABEL side, because their anchor is the first-contact point.

    They are measured around pass.end_location, and pass.end_location IS where somebody
    touched the ball (evidence.md §15). definitions.md §16 bars anything that reads the future
    of the play from X_t, without an exemption for a feature that happens to be predictive: it
    was the largest ablation delta in the corner table precisely because it was pricing the
    outcome (evidence.md §11)."""
    for col in ("att_density_5", "att_density_10"):
        assert col not in _NUMERIC
        assert col in _LABEL_COLS
        assert not any(col in cols for cols in FEATURE_GROUPS.values())
    assert "density" not in FEATURE_GROUPS and "delivery_density" not in FEATURE_GROUPS
    # `delivery` is a key of _BLOCKS, which ablate iterates, and never of FEATURE_GROUPS
    assert "delivery" not in FEATURE_GROUPS

    # and the design must actually refuse them, not merely omit them from the list
    m = CompetingOutcomesModel()
    _, names = m._design(_synth())
    assert not [n for n in names if n.startswith("att_density")]


def test_congestion_is_the_legitimate_substitute_and_enters_the_design():
    """The replacement for the density pair the test above bars, and the reason it is allowed.

    att_density_5/10 read pass.end_location, which IS the first-contact point. Congestion is the
    same inverse-distance kernel on FIXED anchors, so it cannot read its own corner's outcome. It
    has to be in the design matrix and in the ablation, and none of its columns may be label side.

    It is also not the barred feature under another name: evidence.md §11 measures its correlation
    with att_density_5 on each anchor."""
    cols = FEATURE_GROUPS["congestion"]
    assert len(cols) == 6
    assert not (set(cols) & _LABEL_COLS), "a congestion column may never be label side"
    assert set(cols) <= set(_NUMERIC)
    # the anchors are constants, so nothing per-corner and outcome-derived can reach them
    from constants import DROP_ANCHOR_SB, GOALMOUTH_X, PITCH_Y_MID_SB
    assert DROP_ANCHOR_SB == (102.0, 40.5) and (GOALMOUTH_X, PITCH_Y_MID_SB) == (108.0, 40.0)

    m = CompetingOutcomesModel()
    _, names = m._design(_synth())
    assert set(cols) <= set(names)
    # and dropping the group removes exactly its columns, so its ablation delta means what it says
    _, without = m._design(_synth(), drop_group="congestion")
    assert set(names) - set(without) == set(cols)


def test_the_design_cannot_rebuild_the_frontal_count():
    """s02's body total is every attacker and defender it counts, so beside both zone blocks it
    rebuilds the defenders in the frontal exactly, and the frontal block's ablation would read
    zero by construction. The total stays in the table as the coverage it measures; the design
    without the frontal block must leave the frontal count unexplained by any linear fit."""
    t = counted_as_s02_counts(_synth())
    X, _ = CompetingOutcomesModel()._design(t, drop_group="frontal_def")
    A = np.c_[np.ones(len(X)), X]
    y = t["n_def_frontal"].to_numpy(float)
    resid = y - A @ np.linalg.lstsq(A, y, rcond=None)[0]
    assert np.abs(resid).max() > 0.1, "the design rebuilds the frontal count from the body total"
    assert "congestion_bodies" not in FEATURE_GROUPS["congestion"]


def test_the_delivery_rung_is_kick_intrinsic():
    """The baseline every claim is measured against may not read the first-contact point.

    architecture.md §Stages designates the delivery-location rung as the one every claim has
    to clear, so it may read only what is true at the strike. delivery_end_x/delivery_end_y
    are that same post-outcome point, and a rung reading them would put the contamination
    on both sides of the headline comparison at once."""
    from s03_value import _KICK_INTRINSIC
    assert _KICK_INTRINSIC == ["delivery_x", "delivery_y"]
    # the endpoint itself, and the two fields that RECONSTRUCT it, are all barred: pass.length
    # and pass.angle reproduce location -> end at rounding (evidence.md §15).
    for banned in ("delivery_end_x", "delivery_end_y", "delivery_length", "delivery_angle"):
        assert banned not in _KICK_INTRINSIC


def test_design_no_nan():
    """The design matrix is finite and dropping a group removes exactly its columns."""
    m = CompetingOutcomesModel()
    X, names = m._design(_synth())
    assert X.shape[0] == 72 and X.shape[1] == len(names)
    assert np.isfinite(X).all()
    # exactly its columns and no others, so an ablation delta means what it says
    X2, names2 = m._design(_synth(), drop_group="zones_att")
    assert set(names) - set(names2) == set(FEATURE_GROUPS["zones_att"])
    assert set(names2) - set(names) == set()


def _base_column(name: str, cats: list) -> str:
    """A design column back to the table column it came from: strip the _isna flag, then the
    dummy suffix a categorical picks up from get_dummies."""
    name = name[:-5] if name.endswith("_isna") else name
    return next((c for c in cats if name.startswith(f"{c}_")), name)


def test_every_design_column_declares_where_it_came_from():
    """The guard that makes the declaration executable instead of reviewable.

    One mechanism covers a whole class of defects: something declared that the code does not do.
    `_design` admits a column by naming it and drops a named column that is missing without
    a word, so both directions fail quietly. This closes the entry side: a column reaching the
    design without an entry in _COLUMN_SOURCE fails here, and the entry has to name an export
    field, which is what makes a barred source visible."""
    from s03_value import _COLUMN_SOURCE, _DELIVERY_CAT
    m = CompetingOutcomesModel()
    t = _synth()
    t["taker_foot"] = ["Left Foot", "Right Foot"] * (len(t) // 2)
    t["delivery_height"] = ["High Pass", "Ground Pass"] * (len(t) // 2)
    for kw in (dict(), dict(numeric=[]), dict(with_delivery_cats=False)):
        for col in m._design(t, **kw)[1]:
            if col == "_const":
                continue
            base = _base_column(col, _DELIVERY_CAT)
            assert base in _COLUMN_SOURCE, (
                f"{col!r} reaches the design with no source declared; add it to _COLUMN_SOURCE")


def test_no_design_column_reads_a_barred_field_without_a_measured_exception():
    """definitions.md §5 bars pass.end_location, pass.length and pass.angle because each reads the
    first-contact point, and the two provider pass scores built on it. `short_corner` takes its
    second branch off pass.length, which is the one exception, and it is here rather than
    discovered by a reviewer."""
    from s03_value import _BARRED_SOURCES, _COLUMN_SOURCE, _SOURCE_EXCEPTIONS
    offenders = {c: s for c, s in _COLUMN_SOURCE.items()
                 if any(b in s for b in _BARRED_SOURCES) and c not in _SOURCE_EXCEPTIONS}
    assert not offenders, f"barred source reaching the design undeclared: {offenders}"
    for col in _SOURCE_EXCEPTIONS:
        assert col in _COLUMN_SOURCE
        assert any(b in _COLUMN_SOURCE[col] for b in _BARRED_SOURCES), (
            f"{col!r} is excepted from a bar it does not touch; drop the exception")


def test_the_provider_pass_scores_are_barred_and_a_column_built_on_them_would_be_refused():
    """Neither provider score reaches the design today, and that absence is not the protection: a
    feature added tomorrow on `pass.xclaim` or `pass.pass_success_probability` has to fail here.
    xclaim is documented as built from the end location, and the end point explains most of the
    other among corners struck from the same spot at the same height (definitions.md §5)."""
    from s03_value import _BARRED_SOURCES, _SOURCE_EXCEPTIONS
    for field in ("pass.xclaim", "pass.pass_success_probability"):
        assert field in _BARRED_SOURCES
    candidate = {"xclaim_feature": "pass.xclaim", "psp_feature": "pass.pass_success_probability"}
    offenders = {c for c, s in candidate.items()
                 if any(b in s for b in _BARRED_SOURCES) and c not in _SOURCE_EXCEPTIONS}
    assert offenders == set(candidate)


def test_a_declared_design_column_the_table_never_wrote_raises():
    """The guard every table builder calls, exercised as itself rather than as a copy.

    A declared column can go missing because _design filters its whitelist by
    `c in table.columns`. A test that re-types the guard's predicate stays green whatever the
    builders do, so this one calls the function they call."""
    from s03_value import _DELIVERY_CAT, _KICK_INTRINSIC, assert_declared_columns
    full = pd.DataFrame([{c: "x" for c in _DELIVERY_CAT}
                         | {c: 1.0 for c in _KICK_INTRINSIC} | {"outcome": "clear_safe"}])
    assert_declared_columns(full)
    for dropped in ("taker_foot", "delivery_x"):
        with pytest.raises(ValueError, match=dropped):
            assert_declared_columns(full.drop(columns=[dropped]))


def test_the_full_rung_nests_the_delivery_rung():
    """A ladder step reads as "what the new block bought" only if the rungs nest.

    Without this, `_NUMERIC` can quietly stop containing the kick spot and the published step
    becomes "trade the two coordinates for the 360 block" rather than "add the 360 block".
    Both are plausible numbers and nothing else in this suite tells them apart."""
    from s03_value import _KICK_INTRINSIC
    m, t = CompetingOutcomesModel(), _synth()
    deliv = m._design(t, numeric=_KICK_INTRINSIC, with_delivery_cats=True)[1]
    full = m._design(t, with_delivery_cats=True)[1]
    assert set(deliv) <= set(full)
    assert all(c in full for c in _KICK_INTRINSIC)


def test_the_kick_spot_is_droppable_but_is_not_a_360_group():
    """It has to be ablatable without joining the block s05's specification curve iterates.

    FEATURE_GROUPS is the 360 commitment design and run_causal builds one specification per
    entry, so a "drop the kick spot" arm of the edge-protector effect would be meaningless.
    _BLOCKS is what carries droppable-and-permutable, and it is the wider set."""
    from s03_value import _BLOCKS, _KICK_INTRINSIC, _THREE_SIXTY
    assert "kick_spot" not in FEATURE_GROUPS and "kick_spot" in _BLOCKS
    assert not any(c in _THREE_SIXTY for c in _KICK_INTRINSIC)
    t = _synth()
    kept = CompetingOutcomesModel()._design(t, drop_group="kick_spot")[1]
    assert not any(c in kept for c in _KICK_INTRINSIC)


def test_design_missing_indicators():
    """A NaN feature raises its _isna flag, and the NaN SURVIVES into X on purpose.

    Imputation is per fold in _fit_prep, from the train median only, so _design must
    not fill it here. The main fixture carries no NaN, so the NaN is injected: this is
    the path every real table takes."""
    t = _synth()
    t.loc[t.index[:10], "marking_cost"] = np.nan
    X, names = CompetingOutcomesModel()._design(t)
    assert "marking_cost_isna" in names
    flag = X[:, names.index("marking_cost_isna")]
    assert flag[:10].sum() == 10 and flag[10:].sum() == 0
    assert np.isnan(X[:10, names.index("marking_cost")]).all()


def test_design_never_carries_a_label_column():
    """X_t is delivery-instant only, and nothing read after the delivery may enter it.

    This cannot fail while _design stays a whitelist, and that is the point: the table
    carries many post-delivery columns and the whitelist is the only thing keeping them
    out. A reviewer will look for exactly this assertion."""
    _, names = CompetingOutcomesModel()._design(_synth())
    base = {n[:-5] if n.endswith("_isna") else n for n in names}
    assert base & _LABEL_COLS == set()
    assert not ({"league", "attacking_team", "defending_team"} & base)   # grouping keys are not features


def test_fit_prep_imputes_with_the_train_median():
    """The fill value is the TRAIN block's median, and the same transform is reused."""
    tf = CompetingOutcomesModel._fit_prep(np.array([[1.0], [3.0], [5.0]]))
    assert tf(np.array([[np.nan]]))[0, 0] == pytest.approx(tf(np.array([[3.0]]))[0, 0])
    assert tf(np.array([[3.0]]))[0, 0] == pytest.approx(0.0)         # the median standardises to 0


def test_fit_cv_harness():
    """fit_cv returns proper OOF probabilities and bounded metrics."""
    m = CompetingOutcomesModel(n_splits=4)
    cv = m.fit_cv(_synth())
    assert isinstance(cv, CVResult)
    assert cv.classes == sorted(_CLASSES)
    assert np.allclose(cv.oof_proba.sum(axis=1), 1.0)
    assert cv.n == 72 and 0 <= cv.accuracy <= 1
    assert 0 <= cv.ece <= 1 and cv.log_loss > 0 and cv.brier >= 0


def test_hurdle_nonneg():
    """Expected after-clearance danger from the hurdle is non-negative."""
    m = CompetingOutcomesModel(n_splits=4)
    e = m.hurdle_xg(_synth(), "after_clearance_xg")
    assert e.shape == (72,)
    assert (e >= 0).all()


def test_the_hurdle_second_stage_is_gamma_with_tweedie_reachable():
    """methods_second_ball.md §2.2 prefers a gamma with a log link and demotes Tweedie.

    The gate has already removed every zero, so stage 2 sees a strictly positive sample.
    Tweedie at power 1.5 carries its own mass at zero and spends a parameter on something
    that is no longer there. Both stay runnable so the swap can be reported as a
    sensitivity rather than asserted."""
    m = CompetingOutcomesModel(n_splits=4)
    t = _synth()
    g = m.hurdle_xg(t, "after_clearance_xg")
    tw = m.hurdle_xg(t, "after_clearance_xg", second_stage="tweedie")
    assert g.shape == tw.shape == (72,) and (g >= 0).all() and (tw >= 0).all()
    with pytest.raises(ValueError, match="second_stage"):
        m.hurdle_xg(t, "after_clearance_xg", second_stage="poisson")


def _sparse_gate_table(n: int = 360, n_matches: int = 20, rate: float = 0.12,
                       seed: int = 7) -> pd.DataFrame:
    """The main fixture, re-dealt sparse: the regime the hurdle gate actually runs in.

    The 72-row fixture cannot exercise this. Its after-clearance xG fires on one corner in
    four and leaves 18 positives over 7 matches, too few for the reweighting to move the
    gate off the base rate at all, so a balanced gate scores 0.28 against a base of 0.25 and
    looks healthy. The defect only shows at the grant's sparsity, which the default rate here
    re-creates."""
    t = _synth(n, seed=1)
    rng = np.random.default_rng(seed)
    t["match_id"] = np.arange(len(t)) % n_matches
    t["after_clearance_xg"] = np.where(rng.random(len(t)) < rate,
                                       np.abs(rng.normal(0.05, 0.03, len(t))), 0.0)
    return t


def test_the_hurdle_gate_is_a_probability():
    """Stage 1 has to return P(xg > 0), which is what makes the product an expectation.

    A gate fit with class_weight="balanced" optimises balanced accuracy and drifts toward
    0.5 whatever the base rate, which inflates xSecondBall. The product alone cannot see
    that, so the assertion is on the gate itself. On this fixture a balanced gate returns
    0.4154 against a base of 0.1139 and fails the comparison below."""
    m = CompetingOutcomesModel(n_splits=5)
    t = _sparse_gate_table()
    p, magnitude = m.hurdle_xg(t, "after_clearance_xg", return_parts=True)
    base = float((t["after_clearance_xg"] > 0).mean())
    assert p.shape == (len(t),) and ((p >= 0.0) & (p <= 1.0)).all()
    assert abs(p.mean() - base) < abs(p.mean() - 0.5)
    assert np.allclose(m.hurdle_xg(t, "after_clearance_xg"), p * magnitude)


def _oof_args(m: CompetingOutcomesModel, t: pd.DataFrame) -> tuple:
    """The positional arguments _oof_proba takes, built from a table."""
    X, _ = m._design(t)
    y = t["target"].to_numpy()
    idx = {c: i for i, c in enumerate(sorted(pd.unique(y)))}
    return X, y, t["match_id"].to_numpy(), idx, m._params()


@pytest.mark.parametrize("path", ["_oof_proba", "tune_cv", "calibrate_cv", "hurdle_xg"])
def test_every_split_path_asserts_disjointness(monkeypatch, path):
    """The guard has to EXECUTE on all four GroupKFold paths, not merely be imported.

    Handing each one a splitter whose train and test are the same rows puts every match on
    both sides, so a path that skipped the guard would fit happily and return a number. A
    NaN match_id cannot provoke it instead: sklearn rejects NaN groups inside GroupKFold
    before the guard is ever reached."""
    monkeypatch.setattr(s03_value, "GroupKFold", SameRowsSplit)
    m = CompetingOutcomesModel(n_splits=4)
    t = _synth()
    run = {"_oof_proba": lambda: m._oof_proba(*_oof_args(m, t)),
           "tune_cv": lambda: m.tune_cv(t, C_grid=(0.1,), class_weight_grid=(None,),
                                        penalty_grid=({"penalty": "l2"},)),
           "calibrate_cv": lambda: m.calibrate_cv(t),
           "hurdle_xg": lambda: m.hurdle_xg(t, "after_clearance_xg")}
    with pytest.raises(AssertionError, match="match_id leaks"):
        run[path]()


# ── smoothECE ───────────────────────────────────────────────────────────
def _two_class(conf: np.ndarray, hit: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(y_idx, proba) for a binary head asserting `conf` on class 1 and being right on `hit`."""
    return np.where(hit, 1, 0), np.column_stack([1.0 - conf, conf])


def test_smooth_ece_recovers_a_known_miscalibration():
    """A head that says 0.9 and is right 30% of the time is off by 0.60. The measure has to
    say so, or it is not measuring calibration."""
    rng = np.random.default_rng(0)
    n = 20000
    y, p = _two_class(np.full(n, 0.9), rng.random(n) < 0.3)
    assert CompetingOutcomesModel.smooth_ece(y, p) == pytest.approx(0.60, abs=0.01)


def test_smooth_ece_is_near_zero_when_calibrated():
    rng = np.random.default_rng(1)
    conf = rng.uniform(0.5, 1.0, 20000)
    y, p = _two_class(conf, rng.random(20000) < conf)
    assert CompetingOutcomesModel.smooth_ece(y, p) < 0.02


def test_smooth_ece_does_not_move_with_a_bin_count_because_it_has_none():
    """The reason methods_second_ball.md §4 withdrew the binned ECE: on the same predictions it
    swings by several times its own value purely from where the edges fall, and there is no
    principled choice of bin count to settle it."""
    rng = np.random.default_rng(2)
    conf = np.clip(rng.normal(0.7, 0.02, 20000), 0, 1)
    y, p = _two_class(conf, rng.random(20000) < conf)
    binned = [CompetingOutcomesModel.ece(y, p, b) for b in (8, 10, 12, 15, 20)]
    assert max(binned) > 3 * min(binned), "the fixture no longer exhibits the flaw"
    assert CompetingOutcomesModel.smooth_ece(y, p) < 0.05


def test_smooth_ece_reports_its_own_bandwidth():
    """σ* is a fixed point, not a hyperparameter: the value returned IS the bandwidth, so
    smECE evaluated there has to reproduce it."""
    rng = np.random.default_rng(3)
    y, p = _two_class(np.full(4000, 0.8), rng.random(4000) < 0.5)
    s = CompetingOutcomesModel.smooth_ece(y, p)
    at = CompetingOutcomesModel._smooth_ece_at(p.max(axis=1), (p.argmax(axis=1) == y).astype(float), s)
    assert at == pytest.approx(s, abs=2e-3)


def test_cv_and_calibration_carry_the_smooth_measure():
    m = CompetingOutcomesModel(n_splits=4)
    cv = m.fit_cv(_synth())
    assert 0.0 <= cv.smooth_ece <= 1.0 and np.isfinite(cv.smooth_ece)
    cal = m.calibrate_cv(_synth())
    assert np.isfinite(cal.smooth_ece_before) and np.isfinite(cal.smooth_ece_after)


def test_xsecondball_shape():
    """xSecondBall returns one finite value per corner."""
    m = CompetingOutcomesModel(n_splits=4)
    xsb = m.xsecondball(_synth())
    assert xsb.shape == (72,) and np.isfinite(xsb).all()


def test_ablate():
    """Ablation table covers ALL + every feature group, with ALL's delta at 0."""
    m = CompetingOutcomesModel(n_splits=4)
    ab = m.ablate(_synth())
    assert {"group", "log_loss", "delta"} <= set(ab.columns)
    assert "ALL" in set(ab["group"])
    assert set(FEATURE_GROUPS) | {"delivery"} <= set(ab["group"])
    assert float(ab.loc[ab["group"] == "ALL", "delta"].iloc[0]) == 0.0


def test_permuted_ablation_prices_every_block_at_fixed_width():
    """The control the repo lacked: the label shuffle asks whether the HARNESS leaks, this
    asks what a BLOCK buys without also changing the number of columns."""
    from s03_value import _BLOCKS
    m = CompetingOutcomesModel(n_splits=4)
    pm = m.permuted_ablation(_synth(), n_repeats=2)
    assert set(pm["group"]) == set(_BLOCKS)
    assert np.isfinite(pm["delta"]).all()
    assert set(pm["mode"]) == {"global"}


def test_permuted_ablation_takes_the_three_modes():
    m = CompetingOutcomesModel(n_splits=4)
    for mode in ("global", "within_match", "within_class"):
        assert len(m.permuted_ablation(_synth(), mode=mode, n_repeats=1))
    with pytest.raises(ValueError):
        m.permuted_ablation(_synth(), mode="sideways", n_repeats=1)


def test_the_noise_floor_is_measured_on_decoys_and_leaves_the_design_alone():
    """The floor of the permuted control is built by construction, on blocks of pure noise: one row per width, every
    draw counted, finite, the 95th percentile no higher than the largest draw, the same on every run, and neither the
    caller's table nor the published design touched."""
    from s03_value import _BLOCKS
    m = CompetingOutcomesModel(n_splits=4)
    t = _synth()
    nf = m.noise_floor(t, widths=[1, 3], n_draws=3, n_repeats=1)
    assert list(nf["width"]) == [1, 3] and (nf["n_draws"] == 3).all()
    assert np.isfinite(nf[["mean", "sd", "p95", "max"]]).all().all() and (nf["p95"] <= nf["max"]).all()
    assert nf.equals(m.noise_floor(t, widths=[1, 3], n_draws=3, n_repeats=1))
    assert not any(c.startswith("_noise") for c in t.columns)
    assert not any(c.startswith("_noise") for c in m._design(t)[1])
    default = m.noise_floor(t, n_draws=1, n_repeats=1)
    assert set(default["width"]) == {sum(c in t.columns for c in cols) for cols in _BLOCKS.values()} - {0}


def test_fit_score_never_trains_on_the_rows_it_scores():
    """The whole honesty of the heterogeneity test rests on this one helper: a fold's rows are
    scored by a model that did not see them. A leak here would not raise, it would just make
    every downstream p-value optimistic."""
    m = CompetingOutcomesModel(n_splits=4)
    t = _synth(n=120)
    y = t["target"].to_numpy()
    idx = {c: i for i, c in enumerate(sorted(pd.unique(y)))}
    X, _ = m._design(t)
    tr, te = np.arange(80), np.arange(80, 120)
    p = m._fit_score(X, y, tr, te, idx)
    assert p.shape == (40, len(idx)) and np.allclose(p.sum(axis=1), 1.0)
    # a model fit on tr alone cannot beat one fit on everything, on the rows it never saw, and
    # a helper that quietly trained on every row would return exactly `everything`
    everything = m._fit_score(X, y, np.arange(120), te, idx)
    assert not np.allclose(p, everything)
    row = np.array([idx[v] for v in y[te]])
    assert (-np.log(p[np.arange(40), row].clip(1e-12))).mean() > \
           (-np.log(everything[np.arange(40), row].clip(1e-12))).mean()


def test_step_heterogeneity_cross_fits_and_reports_both_verdicts():
    """Two verdicts on purpose: whether structure EXISTS and whether to open cells. Collapsed
    into one, a single global cross-fit can manufacture heterogeneity, a reading that does not
    survive refitting inside the split (decisiones_cerradas.md)."""
    m = CompetingOutcomesModel(n_splits=4)
    r = m.step_heterogeneity(_synth(n=200), n_reps=2, k_folds=3)
    assert {"blp_p", "gates_p", "verdict", "actionable", "observed_step"} <= set(r)
    assert isinstance(r["actionable"], bool)
    assert r["actionable"] is False or r["gates_p"] < 0.025


def test_step_heterogeneity_placebo_runs_on_the_same_partitions_without_moving_the_real_result():
    """The placebo permutes the proxy's features against the rows, globally and within match, on
    the partitions the real result uses. Its permutations draw from their own generator, so the
    real verdict is the one the call without placebo returns, to the bit."""
    m = CompetingOutcomesModel(n_splits=4)
    t = _synth(n=200)
    with_placebo = m.step_heterogeneity(t, n_reps=2, k_folds=3)
    without = m.step_heterogeneity(t, n_reps=2, k_folds=3, placebo=False)
    assert set(with_placebo["placebo"]) == {"global", "within_match"}
    for mode in ("global", "within_match"):
        assert {"blp_slope", "blp_p", "gates_gap", "gates_p"} <= set(with_placebo["placebo"][mode])
    for k in ("blp_slope", "blp_p", "gates_gap", "gates_p", "observed_step"):
        assert with_placebo[k] == without[k]
    assert "placebo" not in without


def test_step_cells_refuses_every_outcome_derived_column():
    """d is built from the outcome, so a cell drawn from the outcome conditions on the target
    and the contrast means nothing. The cells are delivery-instant categoricals and league."""
    from s03_value import _DELIVERY_CAT
    m = CompetingOutcomesModel(n_splits=4)
    cells = m.step_cells(_synth(n=200))
    assert set(cells["dimension"]) <= set(_DELIVERY_CAT) | {"league"}
    for banned in ("segment", "origin", "shot_body_part", "outcome", "target"):
        assert banned not in set(cells["dimension"])
    assert {"tau", "q_p", "verdict", "min_n"} <= set(cells.columns)


def test_delivery_vs_360_is_a_contrast_and_not_a_ladder_step():
    """A quotient of the two steps would divide by one whose interval covers zero, so this
    carries an interval on a PAIRED difference. The two designs do not nest, hence contrast
    and not step."""
    m = CompetingOutcomesModel(n_splits=4)
    h = m.delivery_vs_360(_synth(), n_boot=50)
    assert h["lo"] <= h["delta"] <= h["hi"]
    assert np.isfinite([h["delivery_log_loss"], h["only_360_log_loss"], h["p_wilcoxon"]]).all()


def test_ablate_delta_carries_an_interval():
    """With n_boot every Δ gets a match-clustered paired interval.

    Without one the table is a ranking of point estimates whose spread between rows
    can be smaller than their own uncertainty, and a Δ indistinguishable from zero
    reads as a finding."""
    ab = CompetingOutcomesModel(n_splits=4).ablate(_synth(), n_boot=200, seed=0)
    body = ab[ab["group"] != "ALL"]
    assert (body["n_boot"] > 0).all()
    assert np.isfinite(body[["lo", "hi", "delta"]].to_numpy()).all()
    assert (body["lo"] <= body["hi"]).all() and (body["hi"] - body["lo"] > 0).all()


def test_fit_group_out_holds_out_whole_leagues():
    """One fold per league, every row held out exactly once, disjointness asserted."""
    m = CompetingOutcomesModel(n_splits=4)
    r = m.fit_group_out(_synth(), group_col="league")
    assert isinstance(r, GroupOutResult) and r.group_col == "league"
    assert r.n_folds == _synth()["league"].nunique() and r.n == 72
    assert np.allclose(r.oof_proba.sum(axis=1), 1.0)
    assert sum(f["n_test"] for f in r.folds) == 72
    assert "match_id" in r.leakage_checked and "league" in r.leakage_checked
    assert np.isfinite(r.log_loss)


def test_fit_group_out_raises_when_a_key_leaks():
    """If league is wrong such that a match spans two leagues, the guard fires.

    The whole point of checking BOTH keys: the league split is match-disjoint only
    when league resolved consistently, and a silently mislabelled row would otherwise
    put the same match on both sides of a fold."""
    t = _synth()
    t.loc[t.index[0], "league"] = "la_liga"                        # row 0's match now straddles two leagues
    t.loc[t.index[0], "match_id"] = t.loc[t.index[1], "match_id"]
    t.loc[t.index[1], "league"] = "premier_league"
    with pytest.raises(AssertionError, match="match_id leaks"):
        CompetingOutcomesModel(n_splits=4).fit_group_out(t, group_col="league")


def test_ece_bounds():
    """Top-1 ECE stays in [0, 1] on random probabilities."""
    rng = np.random.default_rng(1)
    proba = rng.dirichlet(np.ones(4), size=200)
    y = rng.integers(0, 4, size=200)
    assert 0.0 <= CompetingOutcomesModel.ece(y, proba) <= 1.0


def test_apply_temperature():
    """T=1 is the identity; T>1 softens (lower mean top probability), rows still sum to 1."""
    rng = np.random.default_rng(3)
    p = rng.dirichlet(np.ones(4), size=50)
    same = CompetingOutcomesModel._apply_temperature(p, 1.0)
    assert np.allclose(same, p, atol=1e-9) and np.allclose(same.sum(1), 1.0)
    soft = CompetingOutcomesModel._apply_temperature(p, 5.0)
    assert np.allclose(soft.sum(1), 1.0)
    assert soft.max(1).mean() <= p.max(1).mean() + 1e-9


def test_reliability_and_ece_consistent():
    """Reliability bins have the right schema and total count; ECE is bounded."""
    rng = np.random.default_rng(4)
    proba = rng.dirichlet(np.ones(4), size=200)
    y = rng.integers(0, 4, size=200)
    rb = CompetingOutcomesModel.reliability_bins(y, proba, bins=10)
    assert set(rb.columns) == {"bin", "conf_mean", "acc", "count"}
    assert int(rb["count"].sum()) == 200
    assert 0.0 <= CompetingOutcomesModel.ece(y, proba) <= 1.0


def test_calibrate_cv():
    """Temperature calibration keeps probabilities coherent and reports finite metrics."""
    cal = CompetingOutcomesModel(n_splits=4).calibrate_cv(_synth())
    assert isinstance(cal, CalibrationResult) and cal.method == "temperature"
    assert np.allclose(cal.oof_proba_raw.sum(1), 1.0)
    assert np.allclose(cal.oof_proba_cal.sum(1), 1.0)
    assert np.isfinite([cal.log_loss_before, cal.log_loss_after,
                        cal.ece_before, cal.ece_after]).all()
    assert len(cal.reliability_after) == 10


def test_tune_cv_and_apply():
    """Nested tuning is operative on the fixture's matches and apply_best_params adopts the
    winner."""
    m = CompetingOutcomesModel(n_splits=4)
    tune = m.tune_cv(_synth(), C_grid=(0.1, 1.0), class_weight_grid=("balanced", None))
    assert isinstance(tune, TuneResult)
    assert tune.n_outer >= 2 and len(tune.per_fold) == tune.n_outer
    assert tune.best_params["C"] in (0.1, 1.0) and np.isfinite(tune.outer_log_loss)
    m.apply_best_params(tune)
    assert m.C == tune.best_params["C"] and m.best_params_ == tune.best_params


def test_tune_noop_single_group():
    """With one match group the tuner is non-operative and reports NaN honestly."""
    t = _synth()
    t["match_id"] = 0
    tune = CompetingOutcomesModel().tune_cv(t)
    assert tune.n_outer < 2 and tune.per_fold == [] and np.isnan(tune.outer_log_loss)


# ── Real sample: the table builder joins s01 labels + s02 features ──────
@requires_sb
def test_build_corner_table(sb_mid):
    """The real table carries the join columns and a valid competing-outcomes target."""
    tab = build_corner_table([sb_mid])
    assert not tab.empty
    for col in ("set_piece_id", "match_id", "league", "attacking_team", "defending_team",
                "delivery_end_x", "delivery_end_y", "target", "control_rebound",
                "n_attack_cross", "box_block_depth", "after_clearance_xg", "cleared",
                "second_ball_time_s", "conceded_xg", "swing"):
        assert col in tab.columns
    assert tab["target"].isin(set(TARGET_MAP.values())).all()
    assert (tab["match_id"] == sb_mid).all()
    # league is the SECOND grouping key Q2's league-out split needs; it must resolve
    # to a real Big-5 slug, never to the sample's own phantom sixth league
    assert tab["league"].notna().all()
    assert set(tab["league"]) <= set(L.BIG5_LEAGUES)
    assert tab["attacking_team"].notna().all()


@requires_sb
def test_every_tracked_match_folds_into_serie_a():
    """The tracked sample is a strict subset of the Serie A season, so all of it resolves to
    serie_a and none of it forms a sixth league that leave-one-league-out would split off."""
    leagues = s03_value._league_index()
    sample = L.list_matches(L.SERIE_A_SAMPLE)["match_id"].astype(int)
    assert len(sample) == 30
    assert {leagues.get(m) for m in sample} == {"serie_a"}


def test_classwise_calibration_sees_a_head_top_one_cannot():
    """methods_second_ball.md §4's reason for class-wise reporting, made concrete.

    A minority class the model never ranks first contributes NOTHING to the top-1 measure,
    however wrong its probabilities are. Here class 2 is asserted at 0.30 and never occurs,
    and the model always ranks class 0 first, so top-1 calls the head well calibrated."""
    n = 20000
    rng = np.random.default_rng(4)
    p = np.column_stack([np.full(n, 0.6), np.full(n, 0.1), np.full(n, 0.3)])
    y = np.where(rng.random(n) < 0.6, 0, 1)            # class 2 never happens
    cw = CompetingOutcomesModel.classwise_smooth_ece(y, p, ["a", "b", "c"])
    top1 = CompetingOutcomesModel.smooth_ece(y, p)
    assert top1 < 0.02, "top-1 should look clean on this fixture"
    assert cw.set_index("class").loc["c", "smooth_ece"] > 0.25   # asserted 0.30, never happens
    assert cw.set_index("class").loc["a", "smooth_ece"] < 0.02


def test_ablation_refit_point_is_the_fixed_ablation_delta():
    """The refit rebuilds the fits `ablate` scores, so its point is that Δ for every block, and
    the pivotal bounds are the percentile ones reflected through it."""
    t = _synth()
    m = CompetingOutcomesModel(n_splits=4)
    ab = m.ablate(t).set_index("group")
    r = m.ablation_refit_interval(t, n_boot=4, seed=0).set_index("group")
    assert set(r.index) == set(ab.index) - {"ALL"}
    np.testing.assert_allclose(r["point"], ab.loc[r.index, "delta"], rtol=0, atol=1e-12)
    np.testing.assert_allclose(r["adjusted_lo"], 2 * r["point"] - r["refit_hi"])
    np.testing.assert_allclose(r["adjusted_hi"], 2 * r["point"] - r["refit_lo"])
