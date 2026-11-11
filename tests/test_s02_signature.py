"""Tests for s02_signature — commitment signature + delivery-instant geometry (360).

Synthetic-frame unit tests pin the geometry exactly; the real-sample tests run
the full feature set over the cached Serie A sample and skip without the data.
"""
import numpy as np
import pandas as pd
import pytest

import loader_sb as L
import zones
# The magnitudes come from their owner, never through s02: a re-export is an import and
# the guard in test_constants only sees assignments, so this is the one drift it cannot catch.
from constants import BOX_X, DROP_ANCHOR_SB, FK_MIN_X, HOLD_X, REBOUND_X, REBOUND_Y, ZONE_PARTITION
from s01_phases import CornerPhaseGrammar
from s02_signature import CommitmentSignature, ROLES, build_frame, _REBOUND_GRID, _orient
from conftest import requires_sb


def _frame(players: list[tuple], ball: tuple[float, float] = (120.0, 0.5)) -> pd.DataFrame:
    """Build an oriented frame from (x, y, team_id, gk, actor) tuples + a ball."""
    rows = [{"x": x, "y": y, "team_id": t, "is_goalkeeper": g, "is_actor": a, "is_ball": 0}
            for (x, y, t, g, a) in players]
    # Append the ball row (team_id=1 by convention, never a keeper/actor, is_ball=1).
    rows.append({"x": ball[0], "y": ball[1], "team_id": 1, "is_goalkeeper": 0,
                 "is_actor": 0, "is_ball": 1})
    return pd.DataFrame(rows)


# ── Role logic (synthetic, exact) ───────────────────────────────────────
def test_roles_partition():
    """One attacker per job -> one count each and a commit index of 1/3."""
    frame = _frame([
        (113.0, 40.0, 1, 0, 0),   # in the box -> attack_cross
        (99.0, 30.0, 1, 0, 0),    # in front of the box inside its width -> stake_drop
        (85.0, 40.0, 1, 0, 0),    # x<HOLD_X -> held back
        (120.0, 0.5, 1, 0, 1),    # the taker (is_actor=1) -> excluded from the signature
        (110.0, 40.0, 0, 0, 0),   # a defender -> tagged 'defender', not counted
        (118.0, 40.0, 0, 1, 0),   # the keeper (gk) -> excluded
    ])
    # the fixture only means what the comments say while it straddles the thresholds;
    # pin that, or a moved constant silently turns this into a test of nothing
    assert 113.0 >= BOX_X and 85.0 < HOLD_X and HOLD_X <= 99.0 < BOX_X
    sig = CommitmentSignature(frame, delivery_target=None).signature()   # no target -> pure position logic
    assert sig["n_attack_cross"] == 1              # only the x=113 attacker
    assert sig["n_stake_drop"] == 1                # only the x=99 attacker
    assert sig["n_hold_back"] == 1                 # only the x=85 attacker
    assert sig["commit_index"] == 1 / 3            # the share of the three that is in the box


def test_roles_ignore_the_delivery_target():
    """Roles are assigned BY POSITION, never by proximity to where the ball ended.

    form_answers.md Q2 classifies each attacker by position. `delivery_target` comes from
    `pass.end_location`, which on an intercepted delivery is where the defence got to it,
    so letting it force a role would score the plan partly from the outcome. Same frame,
    two very different targets, must give the same signature."""
    frame = _frame([(99.0, 50.0, 1, 0, 0)])        # in front of the box -> stake_drop on position
    on_target = CommitmentSignature(frame, delivery_target=(104.0, 50.0)).signature()
    far_away = CommitmentSignature(frame, delivery_target=(60.0, 5.0)).signature()
    none_at_all = CommitmentSignature(frame, delivery_target=None).signature()
    assert on_target["n_stake_drop"] == 1 and on_target["n_attack_cross"] == 0
    assert on_target == far_away == none_at_all


def test_features_synthetic():
    """Zone, frontal and behind-half counts, control and density match a hand-placed frame."""
    frame = _frame([
        (113.0, 40.0, 1, 0, 0), (104.0, 35.0, 1, 0, 0), (104.0, 45.0, 1, 0, 0),   # 3 attackers, all past BOX_X
        (110.0, 40.0, 0, 0, 0), (104.0, 40.0, 0, 0, 0), (55.0, 40.0, 0, 0, 0),     # 3 outfield defenders
        (118.0, 40.0, 0, 1, 0),                                                    # a keeper -> excluded from n_def
    ])
    feat = CommitmentSignature(frame, delivery_target=(110.0, 40.0)).features()   # target set so density is defined
    assert feat["n_att"] == 3 and feat["n_def"] == 3            # gk excluded from outfield
    # all three attackers are in the box: by band under F, by depth and band under C
    assert (feat["F_att_central"], feat["F_att_near_post"], feat["F_att_far_post"]) == (1, 1, 1)
    assert (feat["C_att_spot_central"], feat["C_att_front_near"], feat["C_att_front_far"]) == (1, 1, 1)
    # the x=104 defender stands in G2's `edge`, the central block of the box, and in no frontal of C's
    assert (feat["C_def_edge"], feat["G2_def_edge"]) == (0, 1)
    assert feat["n_def_frontal"] == feat[f"{ZONE_PARTITION}_def_{zones.FRONTAL[ZONE_PARTITION]}"]
    assert feat["C_def_deep"] == 1                              # the x=55 defender
    assert feat["n_def_behind_half"] == 1                       # the x=55 defender
    assert 0.0 <= feat["control_rebound"] <= 1.0               # Voronoi share of the rebound band -> a fraction
    assert feat["att_density_5"] >= 1                           # someone on the target
    # exact, not a bound: defenders sit at x=110/104/55, so the two deepest average 79.5
    assert feat["box_block_depth"] == 79.5
    assert "restdef_x" not in feat                              # the name must not claim a rest-defence line


def test_congestion_counts_both_teams_and_drops_the_keeper_the_taker_and_the_ball():
    """The population is a declared departure from the provider's rule, so it is pinned exactly.

    Pooling the teams is the departure definitions.md §2 authorises: a defender obstructs a run as
    well as a teammate does. Dropping the taker is the second one, declared: he stands on the
    corner flag, so he is goal-side of either anchor on virtually every corner while being the
    body striking the ball rather than one blocking a run."""
    frame = _frame([(110.0, 40.0, 1, 0, 0),      # attacker, 2 yd from the goalmouth anchor
                    (110.0, 40.0, 0, 0, 0),      # defender on the same spot -> pooling counts him
                    (118.0, 40.0, 0, 1, 0),      # defending keeper -> out, as in the kernel
                    (120.0, 0.5, 1, 0, 1)])      # the taker on the flag -> out
    feat = CommitmentSignature(frame).features()
    assert feat["congestion_bodies"] == 2                    # the two on (110, 40) and nobody else
    # both sit exactly 2 yd from the penalty spot, (GOALMOUTH_X, PITCH_Y_MID_SB) = (108, 40)
    assert feat["congestion_spot"] == pytest.approx(2 * (1 / 2))
    assert feat["congestion_spot_per_body"] == pytest.approx(0.5)
    # anchored on the measured drop instead, the same two bodies sit further away
    d_drop = np.hypot(110.0 - DROP_ANCHOR_SB[0], 40.0 - DROP_ANCHOR_SB[1])
    assert feat["congestion_drop"] == pytest.approx(2 / d_drop)
    # they stand ON each other, so the pair enters at the third-of-a-yard substitution
    assert feat["congestion_pairwise"] == pytest.approx(3.0)
    assert feat["congestion_pairwise_per_pair"] == pytest.approx(3.0)   # exactly one pair


def test_congestion_is_not_occupation():
    """definitions.md §2's whole point, as a test rather than as prose. Eight bodies spread
    across the width occupy the box heavily and congest nothing; four stacked at the near post
    congest severely. An occupation count cannot tell the two apart and this must."""
    spread = _frame([(110.0, y, 1, 0, 0) for y in (20.0, 30.0, 40.0, 50.0, 60.0, 25.0, 35.0, 55.0)])
    stacked = _frame([(110.0, 40.0 + dy, 1, 0, 0) for dy in (0.0, 0.4, 0.8, 1.2)])
    f_spread = CommitmentSignature(spread).features()
    f_stacked = CommitmentSignature(stacked).features()
    assert f_spread["congestion_bodies"] == 8 and f_stacked["congestion_bodies"] == 4
    # twice the bodies, and yet the four stacked ones leave each other far less room
    assert f_stacked["congestion_pairwise"] > f_spread["congestion_pairwise"]


def test_congestion_is_never_anchored_on_the_delivery_target():
    """The anchors are FIXED, which is the only reason this feature is legal at all.

    att_density_5/10 are measured around pass.end_location, which IS the first-contact point, and
    that is why s03 bars them from the design matrix. A congestion anchored on the target would
    inherit the same defect, so the same frame under two wildly different targets must give
    identical congestion, while the density pair beside it moves."""
    frame = _frame([(110.0, 40.0, 1, 0, 0), (104.0, 44.0, 1, 0, 0)])
    near = CommitmentSignature(frame, delivery_target=(110.0, 40.0)).features()
    far = CommitmentSignature(frame, delivery_target=(61.0, 5.0)).features()
    keys = [k for k in near if k.startswith("congestion")]
    assert len(keys) == 7
    assert all(near[k] == pytest.approx(far[k]) for k in keys)
    assert near["att_density_5"] != far["att_density_5"], "the fixture no longer separates the two"


def test_rebound_grid_shape():
    """The sampling grid is 19x49 = 931 points, both endpoints included.

    The band is 18x48 YARDS, which is not the point count. Pinned because
    control_rebound is a mean over exactly these points."""
    assert _REBOUND_GRID.shape == (931, 2)
    xs, ys = np.unique(_REBOUND_GRID[:, 0]), np.unique(_REBOUND_GRID[:, 1])
    assert (len(xs), len(ys)) == (19, 49)
    assert (xs[0], xs[-1]) == REBOUND_X and (ys[0], ys[-1]) == REBOUND_Y   # endpoints sampled, not excluded


def test_rebound_control_takes_no_subsets():
    """control_rebound reads the whole frame, keepers and taker included.

    Taking the (A, D) subsets would read as a Voronoi share over outfielders only. A
    share of reach has to count every body that can get there."""
    frame = _frame([(113.0, 40.0, 1, 0, 1),        # the taker
                    (104.0, 40.0, 0, 0, 0),        # an outfield defender
                    (118.0, 40.0, 0, 1, 0)])       # the defending keeper
    cs = CommitmentSignature(frame)
    # The taker owns the columns past the 108.5 bisector with the defender, 6 of the band's 19,
    # and the keeper's bisector with him (115.5) falls outside the band. A version that took the
    # outfield subsets would leave the defender alone and return 1.0.
    assert cs._rebound_control() == pytest.approx(13 / 19)


def test_orient_does_not_flip_at_the_halfway_line():
    """A delivery at exactly x=60 is NOT flipped.

    The annex of the cut reads the flank free kicks from x >= FK_MIN_X = 60.0, so x == 60.0 is
    the single value that both passes its floor and would trip a non-strict flip, rotating the
    frame 180 degrees."""
    assert FK_MIN_X == 60.0                                     # the two boundaries are the same number
    at = _orient(pd.Series({"location": [60.0, 40.0]}))
    assert at(100.0, 30.0) == (100.0, 30.0)                     # admitted and upright
    below = _orient(pd.Series({"location": [59.9, 40.0]}))
    assert below(100.0, 30.0) == (20.0, 50.0)                   # genuinely low-x deliveries still flip


def test_every_body_gets_exactly_one_tag_in_precedence_order():
    """The four fixed tags are matched before any job, and s08 reads them by value.

    `DropSituation._stake` (s08) filters `role == "stake_drop"` on this very column, so a tag
    that changed spelling or a precedence that flipped would move situation.parquet silently. The ball
    row and the taker both carry team_id=1, so without precedence they would read as
    attackers and take a job; the defending keeper must reach `defender`, not `gk_att`."""
    frame = _frame([(113.0, 40.0, 1, 0, 0),    # attacker in the box -> attack_cross
                    (85.0, 40.0, 1, 0, 0),     # attacker short of HOLD_X -> hold_back
                    (99.0, 40.0, 1, 0, 0),     # in front of the box -> stake_drop
                    (120.0, 0.5, 1, 0, 1),     # the taker: an attacker by team, tagged first
                    (110.0, 40.0, 0, 0, 0),    # outfield defender
                    (118.0, 40.0, 0, 1, 0),    # DEFENDING keeper -> 'defender', not 'gk_att'
                    (112.0, 40.0, 1, 1, 0)])   # attacking keeper in frame -> 'gk_att'
    role = CommitmentSignature(frame).frame["role"]
    assert role.tolist() == ["attack_cross", "hold_back", "stake_drop", "taker",
                             "defender", "defender", "gk_att", "ball"]
    # object, not a numpy '<U12': s08 compares it against a python str and the table stores it
    assert role.dtype == object


def test_a_body_wide_of_the_box_does_not_attack_the_cross():
    """A job reads the width of the box. Past the penalty spot but out by the touchline an attacker
    is not attacking the cross, inside the box short of the spot and past G2's `edge` he is, and the
    staker is the one in front of the box inside its width; depth alone would get the first two wrong."""
    frame = _frame([(110.0, 10.0, 1, 0, 0),    # past the spot's line, wide of the box
                    (106.0, 40.0, 1, 0, 0),    # inside the box, short of the spot's line, past G2's edge
                    (99.0, 40.0, 1, 0, 0)])    # in front of the box, inside its width
    assert CommitmentSignature(frame).frame["role"].tolist()[:3] == \
        ["hold_back", "attack_cross", "stake_drop"]


def test_the_row_carries_both_partitions_counts_mirrored_by_the_kick():
    """The table carries every partition's counts, so turning the switch needs no rebuild, and a
    corner from the attackers' right counts as its mirror image from the left."""
    players = [(116.0, 33.0, 1, 0, 0), (110.0, 47.0, 1, 0, 0), (99.0, 40.0, 0, 0, 0),
               (104.0, 10.0, 0, 0, 0)]
    left = CommitmentSignature(_frame(players, ball=(120.0, 0.5))).features()
    right = CommitmentSignature(_frame([(x, 80.0 - y, t, g, a) for x, y, t, g, a in players],
                                       ball=(120.0, 79.5))).features()
    cols = zones.table_columns()
    assert {c: left[c] for c in cols} == {c: right[c] for c in cols}
    assert left["n_def_frontal"] == 1


def test_the_central_edge_band_is_a_diagnostic_and_no_column_repeats_another():
    """The central edge band (EDGE_X by EDGE_Y_HALF) is counted as a diagnostic beside the
    frontal, and no feature only repeats another column or sums zones."""
    feat = CommitmentSignature(_frame([(104.0, 40.0, 0, 0, 0), (99.0, 40.0, 0, 0, 0)])).features()
    assert feat["n_def_edge_band"] == 1 and (feat["C_def_edge"], feat["G2_def_edge"]) == (1, 2)
    assert feat["n_def_frontal"] == feat[f"{ZONE_PARTITION}_def_{zones.FRONTAL[ZONE_PARTITION]}"]
    for repeated in ("n_att_in_box", "n_def_in_box", "commitment_ratio", "n_edge_att", "n_edge_def",
                     "edge_balance", "congestion_goalmouth", "congestion_goalmouth_per_body"):
        assert repeated not in feat
    assert "n_committed_att" not in CommitmentSignature(_frame([(104.0, 40.0, 1, 0, 0)])).signature()


# ── Real sample: full feature set over granted corners ────────────────
@requires_sb
def test_build_frame_real(sb_events, sb_mid):
    """A real corner's frame is oriented to the attacking half with exactly one ball."""
    corners = CornerPhaseGrammar(sb_events).corners()
    fz = L.load_360(sb_mid)
    ids = set(fz["event_uuid"])
    corner = next(c for _, c in corners.iterrows() if c["id"] in ids)
    frame, target = build_frame(corner, fz)
    assert int(frame["is_ball"].sum()) == 1
    assert frame[frame["is_ball"] == 1]["x"].iloc[0] >= 60.0    # oriented to attacking half
    assert frame["x"].between(0, 120).all() and frame["y"].between(0, 80).all()
    assert target is None or (0 <= target[0] <= 120 and 0 <= target[1] <= 80)


@requires_sb
def test_signature_and_features_real(sb_events, sb_mid):
    """Over every real corner with 360: the signature partitions and features stay in range."""
    corners = CornerPhaseGrammar(sb_events).corners()
    fz = L.load_360(sb_mid)
    ids = set(fz["event_uuid"])
    n_checked = 0                                      # guards a pass that exercised nothing
    for _, c in corners.iterrows():
        if c["id"] not in ids:
            continue
        cs = CommitmentSignature.from_corner(c, fz)
        sig, feat = cs.signature(), cs.features()
        assert sig["n_attack_cross"] + sig["n_stake_drop"] + sig["n_hold_back"] == feat["n_att"]
        assert set(cs.frame["role"]) <= set(ROLES) | {"ball", "taker", "defender", "gk_att"}
        assert feat["n_att"] >= 0 and feat["n_def"] >= 0
        assert 0.0 <= feat["control_rebound"] <= 1.0
        for p, names in zones.ZONES.items():          # every body in one zone of each partition
            assert sum(feat[f"{p}_att_{z}"] for z in names) == feat["n_att"]
            assert sum(feat[f"{p}_def_{z}"] for z in names) == feat["n_def"]
        assert feat["n_def_frontal"] == feat[f"{ZONE_PARTITION}_def_{zones.FRONTAL[ZONE_PARTITION]}"]
        assert 0 <= feat["n_def_frontal"] <= feat["n_def"]
        n_checked += 1
    assert n_checked > 0
