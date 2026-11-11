"""The shot-geometry kernel, against the invariants the plausible reading gets wrong.

Every rule this module encodes is fixed by inverting `shot.freeze_frame` against published
values, or declared by the provider's own published source, and each one has a competing reading
that looks equally sensible and is measurably worse. The tests below are named after the losing
reading rather than the winning one, and each docstring carries the agreement the rule buys, so
a future edit that "simplifies" one of them fails against the number it costs.

TWO LAYERS, AND THE SECOND ONE IS THE POINT. The synthetic frames pin the rules on cases small
enough to check by eye. They cannot see a degradation: swap `>=` for `>` and every one of them
can still be made to pass by editing the frame beside it. So the second layer replays the whole
grant, 43,940 scored shots, and binds the published agreement rates themselves. It is the only
thing here that would notice a recovery degrading. The figures it binds are the ones in
`hudl/data_provider.tex` §sec:density.
"""
from pathlib import Path

import numpy as np
import pytest

import constants as C
import loader_sb as L
import shot_geometry as G

_SHOT = np.array([100.0, 40.0])
# StatsBomb position ids: 1 is the keeper, 2-8 the back line, 9 and up midfield and forward.
_BACK_LINE, _MIDFIELD = 3, 10


def _body(x, y, teammate=False, keeper=False, pid=1.0, position_id=_BACK_LINE) -> dict:
    """One freeze-frame entry. The keeper is flagged by position NAME, as the export writes it."""
    return {"location": [float(x), float(y)], "teammate": bool(teammate), "player.id": pid,
            "position.name": "Goalkeeper" if keeper else "Centre Back",
            "position.id": 1 if keeper else position_id}


def _fa(*bodies) -> G.FrameArrays:
    return G.frame_arrays(list(bodies))


# ── Layer 1: the rules, on frames small enough to check by eye ──

def test_a_defender_level_with_the_shooter_is_goal_side():
    """The test is `>=`, not `>`. Strictly goal-side scores 93.97% against this rule's 100.00%
    on DefendersBehindBall, because defenders level with the shooter are 7.5% of shots."""
    fa = _fa(_body(100.0, 44.0))
    ds = G.defender_distances(_SHOT, fa)
    assert len(ds.goal_side) == 1 and len(ds.behind) == 0
    assert G.behind_ball_counts(_SHOT, fa) == (0, 1)


def test_the_keeper_is_no_part_of_the_defender_set():
    """Counting him takes DefendersBehindBall to 0.59%, and keeping him in the behind-the-shot
    fallback alone costs `distance.ToD2` seven points, 92.72% against 99.64%."""
    fa = _fa(_body(118.0, 40.0, keeper=True), _body(105.0, 40.0))
    ds = G.defender_distances(_SHOT, fa)
    assert len(ds.every) == 1 and ds.goal_side[0] == pytest.approx(5.0)
    assert G.behind_ball_counts(_SHOT, fa)[1] == 1


def test_a_body_on_the_shot_enters_at_a_third_of_a_yard():
    """Dropping it instead loses it from a sum of 1/d altogether. This substitution is what
    carries `density` from 99.993% to 100.00% of 43,940: it is the whole of the residue, three
    shots whose frame holds an opponent on the shot's own coordinates."""
    fa = _fa(_body(*_SHOT))
    assert G.body_distances(_SHOT, fa)[0] == pytest.approx(C.ZERO_DISTANCE_SUBSTITUTE_SB)
    assert G.inverse_distance_sum(_SHOT, fa) == pytest.approx(3.0)
    # The body is counted and priced by one rule, not counted by one and dropped by the other.
    assert G.behind_ball_counts(_SHOT, fa)[1] == 1


def test_the_fallback_takes_the_second_defender_behind_not_the_second_overall():
    """With exactly one defender goal-side the two readings diverge. Taking the second-nearest
    overall holds on 58.28% of the 7,903 where they differ."""
    fa = _fa(_body(105.0, 40.0), _body(96.0, 40.0), _body(90.0, 40.0))
    ds = G.defender_distances(_SHOT, fa)
    assert G.signed_nth_defender(ds, 1) == pytest.approx(5.0)
    # the second BEHIND is 10 yards away; the second of the union would have been 4
    assert G.signed_nth_defender(ds, 2) == pytest.approx(-10.0)


def test_a_negative_distance_is_a_flag_and_not_a_measurement():
    """The sign says no defender was goal-side at all, so the magnitude then describes a
    different player from the one the positive branch describes. Used raw as a feature it
    inverts the ordering it appears to encode."""
    fa = _fa(_body(94.0, 40.0))
    assert G.signed_nth_defender(G.defender_distances(_SHOT, fa), 1) == pytest.approx(-6.0)


def test_an_empty_population_returns_the_thirty_yard_sentinel():
    """`distance.ToD1` is 30 on precisely the shots whose frame holds no outfield opponent, at
    100.00%. Read as a distance it puts a 30-yard defender into the distribution."""
    fa = _fa(_body(118.0, 40.0, keeper=True), _body(105.0, 40.0, teammate=True))
    ds = G.defender_distances(_SHOT, fa)
    assert G.signed_nth_defender(ds, 1) == C.EMPTY_DEFENDER_DISTANCE_SB
    assert G.inverse_distance_sum(_SHOT, fa) == 0.0


def test_the_shooter_counts_in_attackers_behind_ball():
    """He trivially satisfies `x >= x_shot`, which is why the field reaches 11 while the
    defending count stops at 10. Excluding him scores 51.50% against 100.00%."""
    fa = _fa(_body(*_SHOT, teammate=True), _body(110.0, 40.0, teammate=True))
    assert G.behind_ball_counts(_SHOT, fa)[0] == 2


def _body_360(x, y, teammate=False, keeper=False, actor=False) -> dict:
    """One entry of the OTHER freeze-frame: `keeper` and `actor` flags, and no position at all."""
    return {"location": [float(x), float(y)], "teammate": bool(teammate),
            "keeper": bool(keeper), "actor": bool(actor)}


def test_one_keeper_rule_cannot_serve_both_frames():
    """The keeper is read by `position.name`, which the shot frame carries and the 360 frame does
    not, so the rule reads a 360 frame as holding NO keeper at all. It fails silently: nothing
    raises, the keeper simply joins the outfield population and every distance below him shifts."""
    shot_ff = [_body(118.0, 40.0, keeper=True), _body(105.0, 40.0)]
    assert G.frame_arrays(shot_ff).keeper.tolist() == [True, False]
    ff360 = [_body_360(118.0, 40.0, keeper=True), _body_360(105.0, 40.0)]
    assert not G.frame_arrays(ff360).keeper.any()

    # and the cost is a real one: with the keeper misread as outfield he enters the defender set
    assert len(G.defender_distances(_SHOT, G.frame_arrays(ff360)).every) == 2
    assert len(G.defender_distances(_SHOT, G.frame_arrays(shot_ff)).every) == 1


def test_anchored_crowding_is_the_density_kernel_moved_off_the_shot():
    """Same arithmetic as `inverse_distance_sum`, over a population the caller selected.

    Pinned against the kernel itself rather than against a hand-typed float, so the two cannot
    drift: on the population `density` uses, the transplanted form must return `density`."""
    fa = _fa(_body(105.0, 40.0), _body(105.0, 70.0), _body(96.0, 40.0))
    m = ~fa.teammate & ~fa.keeper
    assert G.anchored_crowding(_SHOT, fa.xy[m]) == pytest.approx(G.inverse_distance_sum(_SHOT, fa))
    # 5 yd dead ahead and hypot(5, 30) wide; the body BEHIND the anchor is dropped by the >= test
    assert G.anchored_crowding(_SHOT, fa.xy[m]) == pytest.approx(1 / 5 + 1 / np.hypot(5.0, 30.0))


def test_both_crowding_forms_carry_the_zero_distance_substitution():
    """A body ON the anchor is the whole of `density`'s residue, so neither form may drop it."""
    on_anchor = np.array([[100.0, 40.0], [105.0, 40.0]])
    assert G.anchored_crowding(_SHOT, on_anchor) == pytest.approx(3.0 + 1 / 5)
    # two bodies 5 yd apart -> 1/5; two bodies on each other -> the third of a yard, not a blow-up
    assert G.pairwise_crowding(on_anchor) == pytest.approx(1 / 5)
    assert G.pairwise_crowding(np.array([[100.0, 40.0], [100.0, 40.0]])) == pytest.approx(3.0)


def test_the_empty_and_degenerate_populations_are_zero_and_not_nan():
    """Nobody goal-side is no crowding. A NaN here would be imputed per fold downstream and
    silently become the train median, which is a different claim from 'the box was empty'."""
    assert G.anchored_crowding(_SHOT, np.empty((0, 2))) == 0.0
    assert G.anchored_crowding(_SHOT, np.array([[94.0, 40.0]])) == 0.0   # all behind the anchor
    assert G.pairwise_crowding(np.array([[105.0, 40.0]])) == 0.0         # one body has no pair


# ── Layer 1, continued: the geometry the published source declares ──

def test_the_cone_is_the_real_posts_widened_by_exactly_one_yard():
    """Both yards are DECLARED by the provider's source, not swept against the data, and that
    is what lets the field be marked verified. A fitted cone plateaued at 95.0%."""
    v = G.shooting_cone(_SHOT)
    assert v[0].tolist() == [C.PITCH_X_SB, 35.0] and v[1].tolist() == [C.PITCH_X_SB, 45.0]
    assert C.GOAL_POSTS_SB[0][1] == 36.0 and C.GOAL_POSTS_SB[1][1] == 44.0
    assert C.CONE_POST_SLACK_SB == 1.0 and C.CONE_APEX_BACKOFF_SB == 1.0


def test_the_apex_sits_behind_the_shot_and_not_on_it():
    """With the apex ON the shot the cone is a single point there, so a body level with the
    shooter can only qualify by standing on him. Backing it off a yard opens the wedge to a
    half-width of 5/21 at the shot's own x, and that reading scores 100.00% against 73.73%."""
    apex = G.shooting_cone(_SHOT)[2]
    assert apex.tolist() == [99.0, 40.0]
    half = 5.0 / 21.0                      # the 5-yard half-base carried back over 21 yards of run
    level = np.array([[100.0, 40.0 + half - 0.01], [100.0, 40.0 + half + 0.01]])
    assert G.in_shooting_cone(level, _SHOT).tolist() == [True, False]


def test_the_cone_restricts_density_rather_than_replacing_it():
    """`density.incone` is the same sum over a subset, so it can never exceed `density`; the
    export agrees on 100.00% of the 43,940 shots the layer below scores. That is not the
    44,224 rows the export populates: the 284 between them carry no freeze-frame or no
    location and are never scored. A body wide of the cone is dropped from it."""
    fa = _fa(_body(105.0, 40.0), _body(105.0, 70.0))
    assert G.inverse_distance_sum(_SHOT, fa, cone_only=True) == pytest.approx(0.2)
    assert G.inverse_distance_sum(_SHOT, fa, cone_only=True) < G.inverse_distance_sum(_SHOT, fa)


def test_the_defensive_line_area_spans_position_ids_two_to_eight():
    """The back line alone. Taking every outfield opponent scores 7.38% against 100.00%, so a
    midfielder sitting wide must not widen the box."""
    fa = _fa(_body(100.0, 30.0), _body(110.0, 50.0), _body(60.0, 5.0, position_id=_MIDFIELD))
    assert G.defensive_line_area(fa) == pytest.approx(200.0)
    assert C.DEFENSIVE_LINE_POSITION_IDS == (2, 8)


def test_the_defensive_line_area_falls_back_to_its_own_sentinel():
    """1,000 square yards, and its empty population is NOT `distance.ToD1`'s: the line can be
    missing from a frame that still shows outfield opponents."""
    fa = _fa(_body(105.0, 40.0, position_id=_MIDFIELD), _body(118.0, 40.0, keeper=True))
    assert G.defensive_line_area(fa) == C.EMPTY_DEFENDER_AREA_SB
    assert len(G.defender_distances(_SHOT, fa).every) == 1


# ── Layer 2: the published agreement rates, replayed over the grant ──

_CACHE = Path(__file__).resolve().parents[1] / "data" / "derived" / "events"
_GRANT_MATCHES = 1751
requires_grant = pytest.mark.skipif(
    len(list(_CACHE.glob("*.parquet"))) < _GRANT_MATCHES,
    reason=f"the exact recovery rates are bound over the whole grant ({_GRANT_MATCHES} matches) "
           "and the local parquet cache is incomplete; build it with pipeline/build_table.py")

# (field, published agreement, denominator) exactly as data_provider.tex §sec:density prints them.
_PUBLISHED = [
    ("density", 100.00, 43940),
    ("density.incone", 100.00, 43940),
    ("DefArea", 100.00, 43940),
    ("distance.ToD1", 99.91, 43940),
    ("distance.ToD2", 99.64, 43940),
]
_COLS = ["density", "density.incone", "distance.ToD1", "distance.ToD2", "DefArea",
         "location", "shot.freeze_frame", "player.id"]


def _score_match(match_id: int) -> dict:
    """(scored, agreed) per field for one match, plus the two freeze-frame regimes."""
    ev = L.load_events(match_id, columns=_COLS)
    m = ev["density"].notna() \
        & ev["shot.freeze_frame"].map(lambda v: isinstance(v, list) and len(v) > 0) \
        & ev["location"].map(lambda v: isinstance(v, list) and len(v) >= 2)
    out: dict = {}

    def hit(k: str, ok: bool) -> None:
        c = out.setdefault(k, [0, 0])
        c[0] += 1
        c[1] += bool(ok)

    for _, r in ev.loc[m].iterrows():
        shot = G.shot_xy(r["location"])
        fa = G.frame_arrays(r["shot.freeze_frame"])
        ds = G.defender_distances(shot, fa)
        hit("density", abs(G.inverse_distance_sum(shot, fa) - float(r["density"])) < 5e-4)
        hit("density.incone", abs(G.inverse_distance_sum(shot, fa, cone_only=True)
                                 - float(r["density.incone"])) < 5e-4)
        hit("DefArea", abs(G.defensive_line_area(fa) - float(r["DefArea"])) < 5e-4)
        for k in (1, 2):
            hit(f"distance.ToD{k}",
                abs(G.signed_nth_defender(ds, k) - float(r[f"distance.ToD{k}"])) < 5e-4)
        pid = r.get("player.id")
        seen = bool(np.isclose(fa.player_id, float(pid)).any()) if pid == pid else False
        hit("regime: with the shooter" if seen else "regime: without the shooter", True)
    return out


@pytest.fixture(scope="module")
def grant_agreement():
    """One pass over the grant, folded to (scored, agreed) per field. Serial on purpose: the
    suite already runs under xdist, and a nested pool would put more threads than cores on."""
    agg: dict = {}
    for mid in L.list_matches(None)["match_id"]:
        for k, (n, ok) in _score_match(int(mid)).items():
            c = agg.setdefault(k, [0, 0])
            c[0] += n
            c[1] += ok
    return agg


@requires_grant
@pytest.mark.parametrize("field,pct,denom", _PUBLISHED, ids=[f[0] for f in _PUBLISHED])
def test_the_published_recovery_rate_still_reproduces(grant_agreement, field, pct, denom):
    """The figure `data_provider.tex` prints, recomputed rather than trusted. This is the only
    test here that a silent degradation cannot walk past."""
    n, ok = grant_agreement[field]
    assert n == denom, f"{field} scored over {n} shots, the document claims {denom}"
    assert round(100 * ok / n, 2) == pytest.approx(pct, abs=0.005), (
        f"{field} recovers {100 * ok / n:.4f}% of {n}, the document claims {pct}%")


@requires_grant
def test_the_two_regimes_keep_their_published_sizes(grant_agreement):
    """`shot.freeze_frame` changed composition in January 2026 and the export carries both
    regimes: 22,580 shots without the shooter and 21,360 with him, summing to the 43,940
    scored. A global figure that mixed them would hide which of the two had failed."""
    without = grant_agreement["regime: without the shooter"][0]
    with_him = grant_agreement["regime: with the shooter"][0]
    assert (without, with_him) == (22580, 21360)
    assert without + with_him == 43940
