"""Tests for pipeline/falsify_frontier — the blocking gate on the frontier.

The gate decides whether the five-segment split is football or an artefact of
`shot.key_pass_id`. It re-derives the segment from the event chain deliberately, and that
re-derivation IS the instrument. What it must not re-derive is the taxonomy underneath: the
touch rule, the clearance predicate, the body-part columns and the window anchor each have
one owner in s01, and a private copy drifts silently as the grammar is corrected. The
assertions below pin each to that single source.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "pipeline"))

import falsify_frontier as ff
import loader_sb as L
import s01_phases as s01
from s01_phases import CornerPhaseGrammar
from conftest import requires_sb
from test_s01_phases import ATT, DEF, WIN, _corner, _free_kick, _row


def _shot(**kw) -> pd.Series:
    """One falsify shot row, defaulting to a clean `direct`."""
    row = dict(t_since_delivery=1.0, t_since_first_contact=0.0,
               chain_class="direct", chain_body_part=None)
    row.update(kw)
    return pd.Series(row)


# ── The taxonomy has one owner ──────────────────────────────────────────
def test_the_taxonomies_are_imported_from_s01_not_re_typed():
    """Identity, not equality: a re-typed copy that happens to match today would pass an
    == check and then drift on the next correction."""
    assert ff.genuine_touch is s01.genuine_touch
    assert ff.own_goal_source is s01.own_goal_source
    assert ff.SEGMENTS is s01.SEGMENTS
    assert ff.AFTER_FIRST_TOUCH is s01.AFTER_FIRST_TOUCH
    assert ff._body_part is s01._body_part
    assert ff._col is s01._col
    assert ff._SHOT_WINDOW_S == CornerPhaseGrammar.SECOND_BALL_WINDOW_S


def test_the_shared_helpers_are_the_shared_ones():
    """The bootstrap and the Poisson interval come from validate, not from a local copy: an
    interval written twice is two chances to disagree about what a published bound means."""
    import validate
    assert ff.cluster_bootstrap_stats is validate.cluster_bootstrap_stats
    assert ff.poisson_ratio_ci is validate.poisson_ratio_ci
    assert not hasattr(ff, "boot_many")


def test_the_publication_floor_has_one_owner():
    """The publication floor is constants.MIN_CELL, never a local literal."""
    import constants
    assert ff.MIN_CELL is constants.MIN_CELL


def test_the_legacy_clearance_predicate_stays_local():
    """_DEF_CLEAR is the one deliberate copy: the gate reports a legacy-vs-corrected
    delta, so the legacy predicate has to survive somewhere to be the `before`."""
    assert "Goal Keeper" in ff._DEF_CLEAR
    assert ff._DEF_CLEAR != s01._CLEAR_TYPES


# ── The window anchor ───────────────────────────────────────────────────
def test_late_is_anchored_on_the_first_contact_not_the_delivery():
    """A shot 12 s after the corner but 2 s after the first contact is second-ball play.
    Anchored on the delivery this would return `late` and drop the goal from the Q1 numerator."""
    assert ff.segment(_shot(t_since_delivery=12.0, t_since_first_contact=2.0)) == "direct"
    assert ff.segment(_shot(t_since_delivery=12.0, t_since_first_contact=9.0)) == "late"


def test_a_delivery_that_produced_no_contact_is_late():
    assert ff.segment(_shot(t_since_first_contact=np.nan)) == "late"


# ── The gate must score the grammar the pipeline ships ──────────────────
@requires_sb
def test_the_gate_reproduces_the_shipped_segmentation_exactly():
    """Both taxonomies are derived from the event chain and neither reads key_pass_id,
    so they must agree shot for shot. Any gap is drift, not method.

    Run over the WHOLE Serie A sample, not one match: a single match carries 2 shots and
    the five segments do not all appear, so it would pass while the anchor was wrong."""
    cat = L.list_matches("serie_a_sample")
    shipped_rows, gate_rows = [], []
    for m in cat["match_id"].astype(int):
        rows = ff.match_rows(m, "serie_a")[0]
        if not rows:
            continue
        # dicts accumulated and ONE frame built at the end, per the repo's rule: a concat of
        # per-match frames infers dtypes per match.
        shipped_rows += CornerPhaseGrammar(L.load_events(m)).shots_all().to_dict("records")
        gate_rows += rows
    shipped, gate = pd.DataFrame(shipped_rows), pd.DataFrame(gate_rows)
    gate["segment"] = gate.apply(ff.segment, axis=1)

    # (delivery, shot): a shot belongs to the last delivery struck before it, and keying on both
    # makes a shot either side places under another delivery show up as a difference.
    gate = gate.rename(columns={"corner_id": "set_piece_id"})
    key = ["set_piece_id", "shot_id"]

    assert len(shipped) > 100, "sample too thin to be a regression test"
    assert set(map(tuple, gate[key].values)) == set(map(tuple, shipped[key].values))
    merged = shipped.merge(gate[key + ["segment"]], on=key, suffixes=("_s01", "_gate"))
    assert len(merged) == len(shipped)
    disagree = merged[merged["segment_s01"] != merged["segment_gate"]]
    assert disagree.empty, disagree[key + ["segment_s01", "segment_gate"]].to_string()


def test_a_row_filed_on_no_touch_leaves_the_shot_direct_as_s01_does(monkeypatch):
    """The gate re-derives the chain, and what counts as a touch in it is s01's `genuine_touch`.

    A failed receipt is filed on a player the ball never reached, so it cannot turn a clean
    first-touch finish into a prolongation, nor count as a touch before the shot. On the grant
    this one rule is what splits the gate from the shipped frontier on corner shots, and the
    sample the reproduction test above walks holds none of them."""
    ev = pd.DataFrame([_corner(),
                       _row(1.0, "Ball Receipt*", DEF, player=5.0,
                            **{"ball_receipt.outcome.name": "Incomplete"}),
                       _row(2.0, "Shot", ATT, **{"shot.statsbomb_xg": 0.2,
                                                 "shot.outcome.name": "Saved"})])
    monkeypatch.setattr(ff.L, "load_events", lambda _mid: ev)
    gate = pd.DataFrame(ff.match_rows(1, "serie_a")[0]).iloc[0]
    shipped = CornerPhaseGrammar(ev).shots_all().iloc[0]
    assert shipped["segment"] == "direct"
    assert ff.segment(gate) == shipped["segment"]
    assert gate["n_touches_before"] == shipped["n_touches_before"]


def test_the_gate_places_an_own_goal_where_s01_does(monkeypatch):
    """A direct shot the defence deflects in is a direct goal in the shipped grammar, so the gate's
    partition puts the own goal there too, through the same `own_goal_source`."""
    ev = pd.DataFrame([_corner(),
                       _row(1.0, "Ball Receipt*", ATT, player=9.0),
                       _row(1.0, "Shot", ATT, player=9.0, id="S",
                            **{"shot.statsbomb_xg": 0.1, "shot.outcome.name": "Wayward"}),
                       _row(1.5, "Block", DEF, player=4.0),
                       _row(2.0, "Own Goal For", ATT, id="OG", player=None)])
    monkeypatch.setattr(ff.L, "load_events", lambda _mid: ev)
    gate = pd.DataFrame(ff.match_rows(1, "serie_a")[0])
    gate["segment"] = gate.apply(ff.segment, axis=1)
    gate["segment"] = ff.place_own_goals(gate)
    shipped = CornerPhaseGrammar(ev).shots_all().set_index("shot_id")["segment"].to_dict()
    assert gate.set_index("shot_id")["segment"].to_dict() == shipped == {"S": "direct", "OG": "direct"}


def test_a_delivery_with_no_contact_has_no_clearance_at_the_gate_either(monkeypatch):
    """With no touch inside the window no episode opens, and a clearance found later is later
    play. s01 decides that once, in `SetPieceGrammar._clearances`, and the gate's legacy-vs-current
    comparison has to score the `cleared` the pipeline ships."""
    ev = pd.DataFrame([_corner(), _row(WIN + 1.0, "Clearance", DEF, player=5.0)])
    monkeypatch.setattr(ff.L, "load_events", lambda _mid: ev)
    gate = pd.DataFrame(ff.match_rows(1, "serie_a")[1]).iloc[0]
    assert not CornerPhaseGrammar(ev).label_all().iloc[0]["cleared"]
    assert not gate["cleared_live"]


@requires_sb
def test_a_source_run_never_lands_on_the_published_name(tmp_path, monkeypatch):
    """--source narrows the gate exactly as --limit does, so without --out it writes a tagged name.

    `falsification.parquet` and its two companions are what report_figures reads as the grant's
    gate. The driver runs in-process on six sample matches, with its repo root moved."""
    real = ff.L.list_matches
    monkeypatch.setattr(ff.L, "list_matches", lambda source=None: real(source).head(6))
    monkeypatch.setattr(ff, "_REPO", tmp_path)
    monkeypatch.setattr(sys, "argv", ["falsify_frontier.py", "--source", "serie_a_sample",
                                      "--jobs", "1", "--boot", "20"])
    ff.main()
    der = tmp_path / "data" / "derived"
    assert not (der / "falsification.parquet").exists()
    assert (der / "falsification_serie_a_sample_stats.parquet").exists()


def test_the_flank_twin_walks_the_flank_free_kicks(monkeypatch):
    """--kinds free_kick_flank runs the gate on the paper's second population, through s01's grammar
    for that kind; the corner grammar finds nothing in the same stream."""
    ev = pd.DataFrame([_free_kick(x=95.0, y=10.0),
                       _row(2.0, "Shot", ATT, **{"shot.statsbomb_xg": 0.2,
                                                 "shot.outcome.name": "Saved"})])
    monkeypatch.setattr(ff.L, "load_events", lambda _mid: ev)
    shots, deliveries = ff.match_rows(1, "serie_a", ("free_kick_flank",))
    assert [r["corner_id"] for r in shots] == ["FK"] == [r["corner_id"] for r in deliveries]
    assert ff.match_rows(1, "serie_a") == ([], [])


def test_a_forced_own_goal_reaches_the_gate_flagged(monkeypatch):
    """An own goal is a terminal outcome and a goal, but carries no xG, no key pass and no
    set_piece_phase. It has to arrive so the Q1 headline counts it, and it has to be
    flagged so the four annotation tests drop it instead of scoring a goal against a zero
    xG denominator.

    Synthetic on purpose: the 30-match Serie A sample contains no corner own goal at all
    (0 of 109 shot rows), so a data-driven version of this test would assert nothing."""
    ev = pd.DataFrame([_corner(),
                       _row(1.0, "Clearance", DEF),
                       _row(2.0, "Own Goal For", ATT)])
    monkeypatch.setattr(ff.L, "load_events", lambda _mid: ev)

    got = pd.DataFrame(ff.match_rows(1, "serie_a")[0])
    assert len(got) == 1
    row = got.iloc[0]
    assert row["is_own_goal"] and row["is_goal"]
    assert row["shot_outcome"] == "Own Goal"
    assert not row["has_key_pass"]
    assert pd.isna(row["xg"]), "an own goal must not enter a conversion table with xg=0"
    assert ff.segment(row) == "after_clearance"
