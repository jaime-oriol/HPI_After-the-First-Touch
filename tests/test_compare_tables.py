"""Tests for compare_tables, the stage that stops the recipe when its tables are not the plan's: what
counts as a difference, the stop, the smoke's cut to its own matches, the grain read population by
population, and what a corner table built under a partition owes instead of the plan's jobs.
"""
import ast
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

_REPO = Path(__file__).resolve().parents[1]
for p in (_REPO / "src", _REPO / "pipeline"):
    sys.path.insert(0, str(p))

import compare_tables as C                                                 # noqa: E402


def _table(n: int = 6) -> pd.DataFrame:
    return pd.DataFrame({"match_id": np.repeat([1, 2, 3], n // 3), "x": np.arange(n, dtype=float),
                         "kind": (["a", "b"] * n)[:n], "y": ([np.nan, 1.0] * n)[:n]})


def test_the_same_table_has_no_difference_and_a_missing_value_matches_only_itself():
    t = _table()
    assert C.differences(t, t.copy()) == []
    u = t.copy()
    u.loc[1, "y"] = np.nan
    assert C.differences(u, t) == ["y: 1 of 6 rows differ"]


def test_a_changed_cell_a_dtype_or_a_column_is_a_difference():
    t = _table()
    u = t.copy()
    u.loc[0, "x"] = 99.0
    assert C.differences(u, t) == ["x: 1 of 6 rows differ"]
    assert C.differences(t.astype({"x": "float32"}), t) == ["x: dtype float32 against float64"]
    assert "['z'] only here" in C.differences(t.assign(z=0), t)[0]
    assert C.differences(t[["x", "match_id", "kind", "y"]], t) == [
        "columns: the same ones in another order"]
    assert C.differences(t.iloc[:4], t) == ["rows: 4 against 6"]


def test_a_subset_may_change_a_numeric_dtype_and_nothing_else():
    """A column that is float on the grant only because some rows carry NaN comes out int on a
    subset without them, so a --limit comparison reads numeric columns by value; a full one does
    not, and a changed value still differs under either."""
    t = _table().assign(n=[1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    sub = t.astype({"n": "int64"})
    assert C.differences(sub, t, strict_dtype=False) == []
    assert C.differences(sub, t) == ["n: dtype int64 against float64"]
    assert C.differences(sub.assign(n=7), t, strict_dtype=False) == ["n: 6 of 6 rows differ"]


def test_a_column_empty_on_both_sides_is_the_same_whatever_its_dtype():
    """A column no row sets comes out object where nothing shares it and float where it shares a
    table with rows that carry it: a subset of 40 matches, or one population's slice of the joint
    grain. Both comparisons read the two as the same, and a value on one side only still differs."""
    t = _table().assign(c=np.nan)
    sub = t.assign(c=pd.Series([None] * 6, dtype=object))
    assert C.differences(sub, t, strict_dtype=False) == []
    assert C.differences(sub, t) == []
    one = t.assign(c=[np.nan] * 5 + [1.0])
    assert C.differences(sub, one, strict_dtype=False) == ["c: dtype object against float64"]
    assert C.differences(sub, one) == ["c: dtype object against float64"]


def test_a_difference_stops_the_run(tmp_path, monkeypatch):
    der, ref = tmp_path / "der", tmp_path / "ref"
    der.mkdir()
    ref.mkdir()
    _table().to_parquet(der / "shots.parquet")
    _table().to_parquet(ref / "shots.parquet")
    monkeypatch.setattr(C, "_DER", der)
    monkeypatch.setattr(sys, "argv", ["compare_tables.py", "--names", "shots.parquet",
                                      "--reference", str(ref)])
    C.main()
    _table().assign(x=0.0).to_parquet(der / "shots.parquet")
    with pytest.raises(SystemExit, match="shots.parquet: x: 5 of 6 rows differ"):
        C.main()


# Two deliveries' bodies, attackers then defenders: (103.5, 40) stands in G2's edge and in C's front row.
_BODIES = [([(116.0, 33.0), (103.5, 40.0), (99.0, 40.0)], [(117.0, 40.0), (103.5, 41.0), (104.0, 10.0)]),
           ([(110.0, 47.0), (90.0, 60.0)], [(100.0, 41.0)])]
def _corner_table() -> pd.DataFrame:
    """A corner table as s02 writes it under the switch, on _BODIES."""
    import zones
    rows = []
    for i, (att, dfd) in enumerate(_BODIES):
        (xa, ya), (xd, yd) = (np.array(v, dtype=float).T for v in (att, dfd))
        jobs = zones.job_of(xa, ya)
        rows.append({"match_id": i, "zone_partition": zones.ZONE_PARTITION, "n_att": len(xa),
                     "n_def": len(xd), **{f"n_{j}": int((jobs == j).sum()) for j in zones.JOBS},
                     "commit_index": float((jobs == "attack_cross").mean()),
                     **zones.table_counts(xa, ya, xd, yd, kick_y=0.1)})
    return pd.DataFrame(rows)


def _as_the_plan(t: pd.DataFrame) -> pd.DataFrame:
    """The columns the plan's tables carry: C's and F's counts without the defenders' frontal."""
    return t.drop(columns=[c for c in t if c.startswith("G2_")
                           or c in ("C_def_edge", "F_def_frontal", "zone_partition")])


def _under(partition: str, code: str) -> str:
    """What `code` prints in a fresh interpreter with the switch set to `partition`, beside this module."""
    code = (f"import sys\nsys.path[:0] = [{str(_REPO / 'tests')!r}]\n"
            "import pandas as pd\nfrom test_compare_tables import C, _corner_table, _as_the_plan\n" + code)
    env = {k: v for k, v in os.environ.items() if k != "HUDL_ZONES"} | {"HUDL_ZONES": partition}
    out = subprocess.run([sys.executable, "-c", code], env=env, capture_output=True, text=True)
    assert out.returncode == 0, out.stderr
    return out.stdout


def test_a_corner_table_may_carry_partition_columns_and_owes_what_its_partition_defines():
    """Under C, the plan's partition: the table may carry every partition's columns, one no partition
    defines differs, and a moved job differs from the plan's and stops adding up to n_att."""
    same, extra, found = ast.literal_eval(_under("C", (
        "t = _corner_table()\n"
        "print([C.against_reference(t, _as_the_plan(t), True),\n"
        "       C.against_reference(t.assign(extra=1), _as_the_plan(t), True),\n"
        "       C.against_reference(t.assign(n_stake_drop=t['n_stake_drop'] + 1), _as_the_plan(t), True)])")))
    assert same == [] and extra == ["columns: ['extra'] only here and no partition's"]
    assert "n_stake_drop: 2 of 2 rows differ" in found
    assert any(f.startswith("the jobs add up to n_att: 2 of 2") for f in found)


def test_under_g2_the_jobs_and_the_frontal_are_held_to_g2_and_not_to_the_plans(tmp_path):
    """Built under G2 against the plan's, built under C: the attacker in the central block stakes the drop
    under G2 and attacks the cross under C, which is no difference, and what G2 defines holds, so a moved
    job is caught by G2's checks instead of by the plan's."""
    ref = tmp_path / "ref.parquet"
    _under("C", f"_as_the_plan(_corner_table()).to_parquet({str(ref)!r})")
    out = _under("G2", f"t, ref = _corner_table(), pd.read_parquet({str(ref)!r})\n"
                       "print(C.against_reference(t, ref, True), int(t['n_stake_drop'].sum()), "
                       "int(ref['n_stake_drop'].sum()))\n"
                       "print(C.against_reference(t.assign(n_stake_drop=t['n_stake_drop'] + 1), ref, True))")
    first, moved = out.splitlines()
    assert first.split() == ["[]", "2", "1"]
    assert ast.literal_eval(moved) == ["the jobs add up to n_att: 2 of 2 rows fail",
                                       "n_stake_drop is G2_att_edge: 2 of 2 rows fail"]


def test_a_limited_table_is_read_against_the_reference_rows_of_its_matches(tmp_path):
    full = _table(9)
    full[full["match_id"] < 3].to_parquet(tmp_path / "corners_limit2.parquet")
    (got_label, got, want), = C.pairs(["corners.parquet"], False, 2, tmp_path,
                                      _write(tmp_path / "ref", {"corners.parquet": full}))
    assert got_label == "corners.parquet" and C.differences(got, want) == []
    assert sorted(set(want["match_id"])) == [1, 2]


def test_the_grain_holding_both_populations_is_read_population_by_population(tmp_path):
    kinds = list(C.COMBINED_KINDS)
    deliv = pd.DataFrame({"match_id": [1, 1, 2, 2], "set_piece_type": [kinds[0], kinds[1]] * 2,
                          "v": [1.0, 2.0, 3.0, 4.0]})
    matches = pd.DataFrame({"match_id": [1, 2], "home": ["a", "b"]})
    deliv.to_parquet(tmp_path / "eda_deliveries.parquet")
    matches.to_parquet(tmp_path / "eda_matches.parquet")
    ref = _write(tmp_path / "ref", {f"eda_deliveries_{k}.parquet": deliv[deliv["set_piece_type"] == k]
                                    for k in kinds} | {f"eda_matches_{kinds[0]}.parquet": matches})
    found = {label: C.differences(got, want) for label, got, want in
             C.pairs([], True, None, tmp_path, ref)}
    assert found == {f"eda_deliveries.parquet [{k}]": [] for k in kinds} | {
        f"eda_deliveries.parquet [{k}] corners per match": [] for k in kinds} | {
        "eda_matches.parquet": []}


def test_a_population_that_never_sets_a_column_matches_its_own_reference(tmp_path):
    """The joint grain stores `counter_time_s` as float because a corner carries it; the flank free
    kicks never do, and their reference, built alone, stores it as object. The full comparison
    reads that slice as its reference."""
    kinds = list(C.COMBINED_KINDS)
    deliv = pd.DataFrame({"match_id": [1, 1, 2, 2], "set_piece_type": [kinds[0], kinds[1]] * 2,
                          "counter_time_s": [3.5, np.nan, np.nan, np.nan]})
    matches = pd.DataFrame({"match_id": [1, 2], "home": ["a", "b"]})
    deliv.to_parquet(tmp_path / "eda_deliveries.parquet")
    matches.to_parquet(tmp_path / "eda_matches.parquet")
    alone = deliv[deliv["set_piece_type"] == kinds[1]].assign(counter_time_s=None)
    ref = _write(tmp_path / "ref", {f"eda_deliveries_{kinds[0]}.parquet":
                                    deliv[deliv["set_piece_type"] == kinds[0]],
                                    f"eda_deliveries_{kinds[1]}.parquet": alone,
                                    f"eda_matches_{kinds[0]}.parquet": matches})
    assert pd.read_parquet(ref / f"eda_deliveries_{kinds[1]}.parquet")["counter_time_s"].dtype == object
    found = {label: C.differences(got, want) for label, got, want in
             C.pairs([], True, None, tmp_path, ref)}
    assert found == {f"eda_deliveries.parquet [{k}]": [] for k in kinds} | {
        f"eda_deliveries.parquet [{k}] corners per match": [] for k in kinds} | {
        "eda_matches.parquet": []}


def test_the_short_corners_are_the_one_thing_the_comparison_leaves_out():
    """A short or ground corner is delivered by its cross, so its row differs by construction; every
    other row is still the reference's, and a table without the flag is compared whole."""
    t = _table().assign(short_corner=[False, True, False, False, True, False])
    assert C.untouched(t)["x"].tolist() == [0.0, 2.0, 3.0, 5.0]
    assert C.untouched(_table()).equals(_table())
    moved = t.assign(x=np.where(t["short_corner"], 99.0, t["x"]))
    assert C.differences(C.untouched(moved), C.untouched(t)) == []
    assert C.differences(C.untouched(moved.assign(x=moved["x"] + 1)), C.untouched(t)) == [
        "x: 4 of 4 rows differ"]


def test_a_shot_table_holds_out_the_shots_of_the_short_corners_of_each_grain(tmp_path):
    """The reference's shots of a short corner sit under its kick's id and the run's under its cross's:
    each is held out by the short corners of its own grain, and the shots of every other delivery stay."""
    kinds = list(C.COMBINED_KINDS)
    ref_grain = pd.DataFrame({"match_id": [1, 1], "set_piece_type": kinds[0], "set_piece_id": ["a", "k"],
                              "short_corner": [False, True]})
    flank = pd.DataFrame({"match_id": [1], "set_piece_type": kinds[1], "set_piece_id": ["f"],
                          "short_corner": [False]})
    ref = _write(tmp_path / "ref", {f"eda_deliveries_{kinds[0]}.parquet": ref_grain,
                                    f"eda_deliveries_{kinds[1]}.parquet": flank,
                                    "shots.parquet": pd.DataFrame({"match_id": [1, 1, 1], "set_piece_id": ["a", "k", "f"],
                                                                   "xg": [0.1, 0.2, 0.3]})})
    pd.concat([ref_grain.assign(set_piece_id=["a", "x"]), flank]).to_parquet(tmp_path / "eda_deliveries.parquet")
    pd.DataFrame({"match_id": [1, 1, 1], "set_piece_id": ["a", "x", "f"], "xg": [0.1, 0.9, 0.3]}
                 ).to_parquet(tmp_path / "shots.parquet")
    (label, got, want), = C.pairs(["shots.parquet"], False, None, tmp_path, ref)
    assert got["set_piece_id"].tolist() == want["set_piece_id"].tolist() == ["a", "f"]
    assert C.differences(got, want) == []


def test_the_corners_taken_in_a_match_are_the_references(tmp_path):
    """The grain keeps a row for a short corner no cross followed, so what the change must not move,
    the corners a match saw, is compared apart from the rows."""
    kinds = list(C.COMBINED_KINDS)
    ref_grain = pd.DataFrame({"match_id": [1, 1, 2], "set_piece_type": kinds[0], "set_piece_id": ["a", "k", "b"],
                              "short_corner": [False, True, False]})
    flank = pd.DataFrame({"match_id": [1], "set_piece_type": kinds[1], "set_piece_id": ["f"], "short_corner": [False]})
    matches = pd.DataFrame({"match_id": [1, 2], "home": ["a", "b"]})
    ref = _write(tmp_path / "ref", {f"eda_deliveries_{kinds[0]}.parquet": ref_grain,
                                    f"eda_deliveries_{kinds[1]}.parquet": flank,
                                    f"eda_matches_{kinds[0]}.parquet": matches})
    matches.to_parquet(tmp_path / "eda_matches.parquet")
    run = ref_grain.assign(set_piece_id=["a", "x", "b"], anchor=["kick", "cross", "kick"],
                           kick_id=["a", "k", "b"], kick_to_delivery_s=[0.0, 3.0, 0.0])
    for grain, ok in ((run, True), (run.iloc[[0, 2]], False)):        # a corner taken that the run lost
        pd.concat([grain, flank]).to_parquet(tmp_path / "eda_deliveries.parquet")
        found = {label: C.differences(got, want) for label, got, want in C.pairs([], True, None, tmp_path, ref)}
        assert (found[f"eda_deliveries.parquet [{kinds[0]}] corners per match"] == []) is ok
        assert found[f"eda_deliveries.parquet [{kinds[0]}]"] == []


def _write(folder: Path, tables: dict) -> Path:
    folder.mkdir(parents=True, exist_ok=True)
    for name, t in tables.items():
        t.to_parquet(folder / name)
    return folder
