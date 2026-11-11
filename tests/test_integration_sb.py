"""Integration: build the REAL corner table and run the downstream stages on it.

The s05 unit tests only exercise synthetic tables, so this is the only guard
on the build_corner_table -> s03/s05 column contract as it actually comes off
the granted sample. One match means s05 cannot cross-fit, and the assertion is
that it reports DEGENERATE rather than crashing or inventing a number.
"""
import loader_sb as L
from s01_phases import SEGMENTS, CornerPhaseGrammar
from s03_value import TARGET_MAP, build_corner_table, CompetingOutcomesModel
from s05_causal import CausalRestDefence, CausalResult
from conftest import requires_sb


@requires_sb
def test_real_table_feeds_downstream(sb_mid):
    """The real S01+S02 table carries the contract columns and every downstream stage
    runs on it (single match -> s05 degenerate, s03 finite)."""
    tab = build_corner_table([sb_mid])
    assert not tab.empty
    # Every column a downstream stage indexes by name. A rename that reaches here
    # unnoticed is how a stage starts modelling an all-zero vector in silence.
    for col in ("target", "match_id", "league", "set_piece_id", "set_piece_type",
                "n_def_frontal", "cleared", "second_ball_time_s", "at_risk_horizon_s",
                "clear_is_header", "delivery_end_x"):
        assert col in tab.columns
    assert tab["target"].isin(set(TARGET_MAP.values())).all()

    cv = CompetingOutcomesModel().fit_cv(tab)
    assert cv.n == len(tab)

    r5 = CausalRestDefence().estimate(tab, outcome="after_first_touch")
    assert isinstance(r5, CausalResult) and r5.degenerate


@requires_sb
def test_shot_grain_and_delivery_grain_agree(sb_mid):
    """The frontier partition holds on the real join, at both grains.

    build_table checks this over the whole grant, but nothing else would catch a
    single-match regression before a full run."""
    g = CornerPhaseGrammar(L.load_events(sb_mid))
    labels, shots = g.label_all(), g.shots_all()
    seg = [f"n_{s}" for s in SEGMENTS]
    assert int(labels[seg].to_numpy().sum()) == int(labels["n_att_shots"].sum()) == len(shots)
    assert set(shots["segment"]) <= set(SEGMENTS)
