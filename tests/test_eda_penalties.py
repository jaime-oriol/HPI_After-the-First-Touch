"""Tests for pipeline/eda_penalties — which penalties a delivery wins inside its episode."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "pipeline"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import eda_penalties as P
from conftest import requires_sb


def test_a_limited_run_reads_its_own_grain_and_writes_a_tagged_name(monkeypatch, tmp_path):
    monkeypatch.setattr(P, "_DER", tmp_path)
    assert P.paths(150) == (tmp_path / "eda_deliveries_limit150.parquet",
                            tmp_path / "eda_penalties_limit150.parquet")
    assert P.paths(None) == (tmp_path / "eda_deliveries.parquet", tmp_path / "eda_penalties.parquet")


def test_the_episode_closes_one_window_after_the_first_contact_or_after_the_kick_without_one():
    assert P.episode(100.0, 2.5, 8.0) == (102.5, 110.5)
    assert P.episode(100.0, None, 8.0) == (None, 108.0)
    assert P.episode(100.0, float("nan"), 8.0) == (None, 108.0)


def test_the_phase_places_the_whistle_against_the_first_contact():
    assert P.phase(101.0, 102.5) == "before"
    assert P.phase(102.5, 102.5) == "at"
    assert P.phase(104.0, 102.5) == "after"
    assert P.phase(104.0, None) == "before"


def test_a_penalty_counts_on_either_sides_record_and_only_when_it_is_the_attacks():
    ev = pd.DataFrame({
        "type.name": ["Foul Committed", "Foul Won", "Foul Committed", "Foul Committed", "Foul Won",
                      "Shot"],
        "team.name": ["Def", "Att", "Att", "Def", "Def", "Att"],
        "foul_committed.penalty": [True, None, True, None, None, None],
        "foul_won.penalty": [None, True, None, None, True, None]})
    assert P.penalty_fouls(ev, "Att").tolist() == [True, True, False, False, False, False]


def test_a_match_whose_schema_carries_no_penalty_field_gives_none():
    ev = pd.DataFrame({"type.name": ["Foul Committed", "Foul Won"], "team.name": ["Def", "Att"]})
    assert not P.penalty_fouls(ev, "Att").any()


def test_the_kick_is_the_attacks_next_penalty_in_the_same_period():
    ev = pd.DataFrame({
        "id": ["early", "rival", "other_half", "kick", "later"],
        "period": [1, 1, 2, 1, 1],
        "_t": [10.0, 60.0, 70.0, 80.0, 900.0],
        "type.name": ["Shot"] * 5,
        "shot.type.name": ["Penalty"] * 5,
        "team.name": ["Att", "Def", "Att", "Att", "Att"]})
    assert P.next_kick(ev, "Att", 1, 50.0)["id"] == "kick"
    assert P.next_kick(ev, "Att", 1, 950.0) is None


def _tab(times: list, kicks: list | None = None) -> pd.DataFrame:
    return pd.DataFrame({"set_piece_id": ["a", "b", "c"][:len(times)],
                         "set_piece_type": ["corner"] * len(times),
                         "time_to_first_contact_s": times,
                         "kick_id": kicks or [None] * len(times)})


def test_the_run_passes_when_its_deliveries_and_first_contacts_are_the_grains():
    P.check(_tab([1.5, None, 2.0]), _tab([1.5, np.nan, 2.0]))


@pytest.mark.parametrize("grain", [
    _tab([1.5, None]),                                                  # a delivery missing
    _tab([1.5, None, 2.0]).assign(set_piece_id=["a", "b", "z"]),         # another delivery
    _tab([1.5, None, 2.0]).assign(set_piece_type="free_kick_flank"),    # another population
    _tab([1.5, None, 2.5]),                                             # another first contact
    _tab([1.5, 0.5, 2.0])])                                             # a contact where none was
def test_the_run_stops_when_a_delivery_or_its_first_contact_is_not_the_grains(grain):
    with pytest.raises(SystemExit, match="STOP"):
        P.check(_tab([1.5, None, 2.0]), grain)


def test_a_short_corner_with_no_delivery_is_not_a_row_the_run_must_find():
    """The grain counts the corner, but it has no episode, so the run's deliveries are the grain's
    rows that have one."""
    no_delivery = pd.DataFrame({"set_piece_id": ["n"], "set_piece_type": ["corner"],
                                "time_to_first_contact_s": [np.nan], "kick_id": [None], "anchor": ["none"]})
    grain = pd.concat([_tab([1.5, None, 2.0]).assign(anchor="kick"), no_delivery], ignore_index=True)
    P.check(_tab([1.5, None, 2.0]), grain)
    with pytest.raises(SystemExit, match="STOP"):
        P.check(_tab([1.5, None, 2.0]), grain.assign(anchor="kick"))


def test_one_kick_credited_to_two_deliveries_stops_the_run():
    tab = _tab([1.5, None, 2.0], kicks=["k1", None, "k1"])
    with pytest.raises(SystemExit, match="two deliveries"):
        P.check(tab, _tab([1.5, None, 2.0]))


@requires_sb
def test_every_delivery_of_a_match_comes_back_once_with_its_phase(sb_mid):
    import loader_sb as L
    from s01_phases import SetPieceGrammar

    rows = pd.DataFrame(P.match_rows(sb_mid))
    want = SetPieceGrammar(L.load_events(sb_mid), kinds=P.COMBINED_KINDS).deliveries()
    assert rows["set_piece_id"].tolist() == want["id"].tolist()
    assert set(rows["set_piece_type"]) <= set(P.COMBINED_KINDS)
    won = rows[rows["penalty"]]
    assert won["phase"].isin(P.PHASES).all() and rows.loc[~rows["penalty"], "phase"].isna().all()
    assert np.isfinite(won.loc[won["kick_id"].notna(), "kick_xg"]).all()
