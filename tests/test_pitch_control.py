"""Tests for pitch_control — the Spearman PPCF at a point.

The causal velocity filter and the ball locator live in `kinematics` and are tested in
`test_kinematics.py`. This module is the point evaluation one render prints as a cross-check
against s04's own reach field. All synthetic, so it runs on a bare clone in under a second.

With a single implementation there is no second one to serve as an oracle, so the tests pin
invariants that stand on their own: the two teams summing to within tolerance, and the control
rates coming back out of the integrator.
"""
import numpy as np
import pandas as pd
import pytest

from constants import PITCH_LENGTH_M, PITCH_WIDTH_M
from pitch_control import _extract_teams, default_model_params, ppcf_at_targets


def _frame(att=((-10.0, 0.0),), deff=((10.0, 0.0),), ball=(0.0, 0.0),
           att_gk=None, def_gk=None) -> pd.DataFrame:
    """One tracking frame: attackers (team 1), defenders (team 2) and the ball.

    `ball=None` omits the ball row, which is what broadcast tracking does when it loses it."""
    att_gk = att_gk if att_gk is not None else (False,) * len(att)
    def_gk = def_gk if def_gk is not None else (False,) * len(deff)
    rows = []
    for i, ((x, y), gk) in enumerate(zip(att, att_gk, strict=True)):
        rows.append({"player_id": f"1_{i}", "team_id": 1, "is_ball": 0, "is_goalkeeper": gk,
                     "x_tracking": x, "y_tracking": y, "vx": 0.0, "vy": 0.0})
    for i, ((x, y), gk) in enumerate(zip(deff, def_gk, strict=True)):
        rows.append({"player_id": f"2_{i}", "team_id": 2, "is_ball": 0, "is_goalkeeper": gk,
                     "x_tracking": x, "y_tracking": y, "vx": 0.0, "vy": 0.0})
    if ball is not None:
        rows.append({"player_id": None, "team_id": None, "is_ball": 1, "is_goalkeeper": False,
                     "x_tracking": ball[0], "y_tracking": ball[1], "vx": 0.0, "vy": 0.0})
    return pd.DataFrame(rows)


def _pitch_targets(n: int) -> np.ndarray:
    """n reproducible points spread over the pitch, shape (n, 2) in metres."""
    rng = np.random.default_rng(0)
    return np.column_stack([rng.uniform(-PITCH_LENGTH_M / 2, PITCH_LENGTH_M / 2, n),
                            rng.uniform(-PITCH_WIDTH_M / 2, PITCH_WIDTH_M / 2, n)])


# ── PPCF core ───────────────────────────────────────────────────────────
def test_default_params_are_internally_consistent():
    p = default_model_params()
    assert p["lambda_def"] == pytest.approx(p["lambda_att"] * p["kappa_def"])
    assert p["lambda_gk"] == pytest.approx(p["lambda_def"] * 3.0)
    assert p["time_to_control_att"] > 0 and p["time_to_control_def"] > 0


def test_ppcf_at_targets_returns_a_probability_per_target():
    """Shape and range over a realistic frame.

    The upper bound holds here but is NOT universal: a degenerate frame overshoots it in one
    Euler step, which the module docstring records. This pins the case the render actually
    meets, not a guarantee the model does not give."""
    targets = _pitch_targets(300)
    p = ppcf_at_targets(_frame(att=((-10.0, 0.0), (-20.0, 8.0)), deff=((10.0, 0.0), (22.0, -6.0))),
                        targets, att_team_id=1)
    assert p.shape == (len(targets),)
    assert np.isfinite(p).all()
    assert (p >= -1e-9).all() and (p <= 1 + 1e-9).all()


def test_the_nearer_team_controls_its_own_side():
    """An attacker at x=-10 and a defender at x=+10 split the pitch the obvious way.

    A sign error in the Euler integration or in the team extraction would flip this,
    and nothing else in the repo would notice."""
    targets = np.array([[-30.0, 0.0], [30.0, 0.0]])
    left, right = ppcf_at_targets(_frame(), targets, att_team_id=1)
    assert left > 0.8 and right < 0.2


def test_the_two_teams_sum_to_within_tolerance_of_one():
    """Swapping which team attacks must give the complementary probability.

    This is the repo's only guard on convergence. The sum reaches 1 only asymptotically:
    the integration stops at the first step past the tolerance, so it lands just under.
    Asserting equality to 1 would be wrong, not strict.

    NO KEEPER on purpose. lambda_gk applies to the defending side alone, so with a keeper
    on the pitch the two runs solve different equations and the sum drops to ~0.74."""
    tol = default_model_params()["model_converge_tol"]
    frame = _frame(att=((-10.0, 0.0), (-20.0, 8.0), (-5.0, -12.0)),
                   deff=((10.0, 0.0), (22.0, -6.0), (4.0, 14.0)))
    targets = _pitch_targets(300)
    total = (ppcf_at_targets(frame, targets, att_team_id=1)
             + ppcf_at_targets(frame, targets, att_team_id=2))
    assert (total > 1 - tol).all() and (total <= 1 + 1e-9).all()


@pytest.mark.parametrize("kappa,keeper", [(1.0, False), (1.72, False), (1.0, True)])
def test_the_control_rates_reach_the_integrator(kappa, keeper):
    """Mirrored players share a time-to-intercept, so their probabilities hold the ratio
    of their control rates exactly, whatever step the integration stops on:
    P_att * (1 + lambda_def/lambda_att) lands within tolerance of 1.

    It is the only test that reads kappa_def and lambda_gk through the public API. The
    keeper case is what pins the factor of 3, and 1.72 is the paper's kappa, which the
    published parameters deliberately do not use."""
    params = default_model_params()
    params["kappa_def"] = kappa
    params["lambda_def"] = params["lambda_att"] * kappa
    params["lambda_gk"] = params["lambda_def"] * 3.0
    ratio = 3.0 * kappa if keeper else kappa

    p_att = ppcf_at_targets(_frame(def_gk=(keeper,)), np.array([[0.0, 0.0]]),
                            att_team_id=1, params=params)[0]
    assert 1 - params["model_converge_tol"] < p_att * (1 + ratio) <= 1 + 1e-9


def test_an_unlocated_player_does_not_poison_every_target():
    """Broadcast tracking loses players, and a NaN position would return NaN for ALL
    targets, not just that player: the min over players propagates it into both veto
    branches. The answer must be the one the frame gives without him."""
    targets = np.array([[0.0, 0.0], [20.0, 10.0], [-30.0, -5.0]])
    with_gap = ppcf_at_targets(_frame(att=((-10.0, 0.0), (np.nan, np.nan), (-5.0, 3.0))),
                               targets, att_team_id=1)
    without = ppcf_at_targets(_frame(att=((-10.0, 0.0), (-5.0, 3.0))), targets, att_team_id=1)
    assert np.isfinite(with_gap).all()
    assert with_gap == pytest.approx(without, abs=0.0)


def test_an_empty_team_hands_the_pitch_to_the_other():
    """The veto arithmetic runs on inf when a side has nobody, which is the only place the
    module leans on IEEE semantics."""
    targets = np.array([[0.0, 0.0], [20.0, 10.0]])
    assert (ppcf_at_targets(_frame(att=()), targets, att_team_id=1) == 0.0).all()
    assert (ppcf_at_targets(_frame(deff=()), targets, att_team_id=1) == 1.0).all()


def test_a_keeper_is_extracted_as_a_keeper():
    """The GK mask reaches the integrator; defending keepers control at lambda_gk, not
    lambda_def."""
    f = _frame(deff=((10.0, 0.0), (30.0, 0.0)), def_gk=(False, True))
    (_, _, att_gk), (_, _, def_gk) = _extract_teams(f, att_team_id=1)
    assert att_gk.tolist() == [False] and def_gk.tolist() == [False, True]
