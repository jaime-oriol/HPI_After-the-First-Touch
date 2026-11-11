"""Tests for s04b_snapshot — the event-only arm and the sufficiency ladder.

The fold machinery is pure and runs everywhere; the freeze-frame arm needs the granted
360 sample and skips cleanly without it.
"""
import ast
from pathlib import Path

import numpy as np
import pytest

import loader_sb
import s04b_snapshot
from constants import (DROP_ANCHOR_SB, PITCH_LENGTH_M, PITCH_WIDTH_M, PITCH_X_SB,
                       PITCH_Y_SB)
from s01_phases import CornerPhaseGrammar
from s04_race import _softmax_p_att
from s04b_snapshot import (_SB_TO_M, fold_table, mde, paired_delta, platt_oof,
                           power_curve, prior_oof, replicate_index, row_logloss, score_arm,
                           snapshot_record, tuned_gbt_oof)
from conftest import requires_sb


def test_scale_is_derived_from_the_pitch_not_typed():
    """The StatsBomb grid is normalised, so its two axes carry different metre factors.

    A single 0.9144 yard would be wrong on both. This asserts the module reads the pitch
    constants, which is what stops the conversion drifting off them."""
    assert _SB_TO_M[0] == pytest.approx(PITCH_LENGTH_M / PITCH_X_SB)
    assert _SB_TO_M[1] == pytest.approx(PITCH_WIDTH_M / PITCH_Y_SB)
    assert _SB_TO_M[0] != _SB_TO_M[1]


def _referenced_names(path: Path) -> set:
    """Every name the module's CODE reaches for: identifiers, attributes, imported modules
    and string literals that are not docstrings. Parsed rather than grepped, so the prose
    explaining an invariant does not read as a violation of it."""
    tree = ast.parse(path.read_text())
    docs = {id(n.body[0].value) for n in ast.walk(tree)
            if isinstance(n, (ast.Module, ast.ClassDef, ast.FunctionDef))
            and ast.get_docstring(n) is not None}
    out = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Name):
            out.add(node.id)
        elif isinstance(node, ast.Attribute):
            out.add(node.attr)
        elif isinstance(node, (ast.Import, ast.ImportFrom)):
            out.update(filter(None, [getattr(node, "module", None)]))
            out.update(a.name for a in node.names)
        elif isinstance(node, ast.Constant) and isinstance(node.value, str) \
                and id(node) not in docs:
            out.add(node.value)
    return out


def test_the_event_only_arm_cannot_reach_tracking_or_the_drop():
    """The freeze-frame arm reads the delivery freeze-frame and nothing else.

    Structural rather than behavioural, because the way this breaks is a later edit reaching
    for a convenience: the tracking loader, the endpoint of the delivery pass (which IS the
    first-contact point), or this corner's own observed drop. None of those would be visible
    in any number the module then produced."""
    names = _referenced_names(Path(__file__).resolve().parents[1] / "src" / "s04b_snapshot.py")
    for forbidden in ("loader", "load_tracking", "pass.end_location", "loose_xy",
                      "drop_x", "drop_y", "won_second_ball", "actual_att"):
        assert forbidden not in names, f"s04b_snapshot reaches for {forbidden!r}"


def test_prior_never_sees_its_own_fold():
    """The per-fold prior is the TRAIN base rate, so a match cannot contribute to its own.

    Built so it fails loudly if it did: one match is all positive while the rest are all
    negative, so a leaking prior lands at 1.0 on that match and an honest one at 0.0."""
    y = np.array([1, 1, 1, 0, 0, 0, 0, 0])
    g = np.array(["a", "a", "a", "b", "b", "c", "c", "d"])
    p = prior_oof(y, g)
    assert np.allclose(p[:3], 0.0)
    assert np.all(p[3:] > 0.0)


def test_platt_never_lets_a_match_reach_its_own_fit():
    """The held-out match's labels must not touch the curve applied to it.

    Built so a leak is visible rather than asserted. Flip the labels of ONE match and refit:
    that match's own predictions must not move, because its fold never saw them, while every
    other match's must, because it sits in their training block. Asserting only the first
    half would pass on a function that ignored the labels entirely, so both halves are here.

    Rank preservation is deliberately not asserted. Platt reorders held-out rows whenever the
    fitted slope is negative, and negative slopes are not hypothetical: they are exactly what
    makes a recalibrated prior leak (score_arm)."""
    rng = np.random.default_rng(0)
    g = np.repeat(np.arange(8), 6)
    p = rng.uniform(0.05, 0.95, len(g))
    y = (p > 0.5).astype(int)
    out = platt_oof(p, y, g)
    y_tampered = y.copy()
    y_tampered[g == 3] = 1 - y_tampered[g == 3]
    tampered = platt_oof(p, y_tampered, g)
    assert np.allclose(out[g == 3], tampered[g == 3]), \
        "a held-out match's own labels reached the curve applied to it"
    assert not np.allclose(out[g == 7], tampered[g == 7]), \
        "tampering with a training match changed nothing, so the fit ignores its labels"


class _SameRowsSplit:
    """Splitter handing back the same rows as train and test, so every match straddles it."""

    def __init__(self, n_splits: int = 5) -> None:
        self.n_splits = n_splits

    def split(self, X, y=None, groups=None):
        idx = np.arange(len(X))
        yield idx, idx


def test_the_prior_asserts_disjointness_on_its_split(monkeypatch):
    """prior_oof runs the guard on every fold: a splitter that puts each match on both sides
    has to raise rather than return a prior that saw its own match."""
    def same_rows(groups):
        idx = np.arange(len(groups))
        yield "all", idx, idx
    monkeypatch.setattr(s04b_snapshot, "leave_one_group_out", same_rows)
    with pytest.raises(AssertionError, match="match_id leaks"):
        prior_oof(np.array([1, 0, 1, 0]), np.array(["a", "a", "b", "b"]))


def test_the_strong_learners_inner_split_asserts_disjointness(monkeypatch):
    """The inner search split runs the guard as well as the outer one, before any trial is
    fit, so a splitter that leaks a match into its own validation block raises."""
    monkeypatch.setattr(s04b_snapshot, "GroupKFold", _SameRowsSplit)
    g = np.repeat(np.arange(6), 4)
    X = np.random.default_rng(0).normal(size=(len(g), 3))
    y = np.tile([0, 1], len(g) // 2)
    with pytest.raises(AssertionError, match="match_id leaks"):
        s04b_snapshot._gbt_outer_fold(X, y, g, np.arange(len(g)), np.arange(0),
                                      n_trials=1, seed=0, k=0)


def test_an_arm_against_itself_is_exactly_zero():
    """The paired delta of an arm with itself must be a hard zero with no width.

    The check that the difference really is per-row and paired: two marginal means of the
    same numbers would also be zero, but their bootstrap would not have zero width."""
    rng = np.random.default_rng(1)
    g = np.repeat(np.arange(6), 5)
    y = rng.integers(0, 2, len(g))
    p = rng.uniform(0.1, 0.9, len(g))
    a = score_arm(p, y, g)
    d = paired_delta(a, a, g, n_boot=200)
    assert d["delta"] == pytest.approx(0.0) and d["lo"] == pytest.approx(0.0)
    assert d["hi"] == pytest.approx(0.0) and d["covers_zero"]


def test_calibration_cost_is_what_recalibration_buys():
    """score_arm's calibration_cost is the raw-minus-recalibrated gap, and a deliberately
    mis-centred arm must show a positive one.

    This is the number the module exists to separate: an arm whose ranking is fine but whose
    level is wrong scores badly raw, and reading that as missing information is the mistake."""
    rng = np.random.default_rng(2)
    g = np.repeat(np.arange(10), 8)
    y = rng.integers(0, 2, len(g))
    honest = np.clip(0.5 + 0.3 * (y - 0.5) + rng.normal(0, 0.1, len(g)), 0.02, 0.98)
    shifted = np.clip(honest - 0.35, 0.02, 0.98)
    assert score_arm(shifted, y, g)["calibration_cost"] > \
        score_arm(honest, y, g)["calibration_cost"]
    assert score_arm(shifted, y, g)["auc"] == pytest.approx(score_arm(honest, y, g)["auc"])


def test_a_match_drawn_twice_counts_as_two_clusters():
    """The resample has to keep as many clusters as it drew.

    A match picked twice contributes two independent clusters to that replicate. Handing the
    inner bootstrap its original label instead merges the copies into one cluster of double
    the rows, leaving fewer and fatter clusters than the design has: the interval widens, the
    rejection rate falls under alpha, and the published MDE comes out too large. Pinned here
    exactly, because the statistical test above only bounds the null rate from one side and
    a rate that is too LOW looks like good behaviour."""
    idx_of = {"a": np.array([0, 1]), "b": np.array([2, 3, 4])}
    rows, ids = replicate_index(np.array(["a", "b", "a"]), idx_of)
    assert list(rows) == [0, 1, 2, 3, 4, 0, 1]
    assert len(np.unique(ids)) == 3, "a duplicated match collapsed into one cluster"
    assert [int((ids == k).sum()) for k in range(3)] == [2, 3, 2]


def test_power_rises_with_the_effect_and_sits_at_alpha_under_the_null():
    """The power curve has to behave like one, or a declared MDE means nothing.

    At a true difference of zero a 95% interval should exclude zero about 5% of the time,
    and power must not fall as the effect grows. Bounds are loose because the estimate is
    itself simulated; what they catch is a curve that is flat, inverted or stuck at 1."""
    rng = np.random.default_rng(3)
    g = np.repeat(np.arange(20), 5)
    diff = rng.normal(0, 0.4, len(g))
    curve = power_curve(diff, g, (0.0, 0.5), n_boot=200, n_rep=60, seed=0)
    null, strong = curve.iloc[0]["power"], curve.iloc[1]["power"]
    assert null <= 0.25, f"the null case rejects {null:.2f} of the time"
    assert strong > null
    assert np.isnan(mde(curve.assign(power=[0.0, 0.1])))


def test_fold_table_partitions_the_rows():
    """Every row is tested exactly once and the per-fold counts add up to the population."""
    y = np.array([1, 0, 1, 1, 0, 0, 1, 0])
    g = np.array([1, 1, 2, 2, 3, 3, 4, 4])
    t = fold_table(y, g)
    assert len(t) == 4
    assert t["n_test"].sum() == len(y)
    assert t["test_pos"].sum() == int(y.sum())
    assert (t["n_train"] + t["n_test"] == len(y)).all()


def test_row_logloss_is_per_row_and_bounded():
    y = np.array([1, 0])
    assert row_logloss(np.array([0.9, 0.1]), y).mean() < row_logloss(np.array([0.5, 0.5]), y).mean()
    assert row_logloss(np.array([1.0, 0.0]), y).max() < 15.0


@pytest.fixture(scope="module")
def sample_corners(sb_mid):
    """A Serie A sample match's corners plus its 360 (events only, no tracking needed)."""
    ev = loader_sb.load_events(sb_mid)
    return CornerPhaseGrammar(ev).corners(), loader_sb.load_360(sb_mid)


@requires_sb
def test_snapshot_record_races_both_sides_from_the_delivery(sample_corners):
    """The freeze-frame arm produces the record shape the shared fusion consumes.

    Both teams must appear: a record with one side is a broken team mapping, and it would
    still return a perfectly valid-looking probability of exactly 0 or 1."""
    corners, fz = sample_corners
    recs = [r for r in (snapshot_record(c, fz) for _, c in corners.iterrows()) if r is not None]
    assert recs, "no corner in the sample match carries a delivery freeze-frame"
    for r in recs:
        assert set(r) == {"tti", "is_att", "corner_id", "n_bodies"}
        assert len(r["tti"]) == len(r["is_att"]) == r["n_bodies"] >= 2
        assert np.isfinite(r["tti"]).all() and (r["tti"] >= 0).all()
        assert r["is_att"].any() and not r["is_att"].all()


@requires_sb
def test_snapshot_probability_is_bounded_and_moves_with_the_anchor(sample_corners):
    """P(att) stays a probability, and moving the anchor moves it.

    A record whose probability is unchanged by racing to a different point would mean the
    positions never entered the arrival times, which is the failure that leaves an arm
    predicting the same number for every corner."""
    corners, fz = sample_corners
    c = next(c for _, c in corners.iterrows() if snapshot_record(c, fz) is not None)
    here = snapshot_record(c, fz)
    far = snapshot_record(c, fz, anchor_sb=(DROP_ANCHOR_SB[0] - 30.0, DROP_ANCHOR_SB[1]))
    p_here = _softmax_p_att(here["tti"], here["is_att"], 0.3)
    assert 0.0 <= p_here <= 1.0
    assert p_here != _softmax_p_att(far["tti"], far["is_att"], 0.3)


def test_the_strong_learner_is_a_function_of_its_seed():
    """Same seed, same probabilities; different seed, different ones.

    Optuna runs parallel trials in threads, so a study given more than one worker sees a
    history ordered by completion and stops being a function of its seed. Measured on the
    grant that moved the arm's log-loss in the fourth decimal between identical calls, which
    is invisible in any single run and fatal to a generated record that has to reproduce.
    Kept small on purpose: what is on trial is determinism, not fit quality."""
    rng = np.random.default_rng(0)
    g = np.repeat(np.arange(24), 8)
    X = rng.normal(size=(len(g), 5))
    y = (rng.uniform(size=len(g)) < 1 / (1 + np.exp(-X[:, 0]))).astype(int)
    a, folds = tuned_gbt_oof(X, y, g, n_trials=4, n_splits=3, jobs=4, seed=0)
    b, _ = tuned_gbt_oof(X, y, g, n_trials=4, n_splits=3, jobs=4, seed=0)
    c, _ = tuned_gbt_oof(X, y, g, n_trials=4, n_splits=3, jobs=4, seed=1)
    assert np.array_equal(a, b), "the tuned arm is not reproducible at a fixed seed"
    assert not np.array_equal(a, c), "the seed does not reach the search at all"
    assert len(folds) == 3 and {"rounds", "inner_log_loss"} <= set(folds[0])
