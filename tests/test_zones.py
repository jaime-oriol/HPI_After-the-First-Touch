"""Tests for zones: the partitions, the switch between them, the flip and the jobs.

The switch is read once, at import, from HUDL_ZONES, so the tests that turn it run a fresh
interpreter; the rest pin the geometry on points placed either side of each line.
"""
import ast
import os
import re
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

import zones
from constants import (BOX_X, BOX_Y, GOAL_POSTS_SB, HOLD_X, SIX_YARD_X, SIX_YARD_Y,
                       ZONE_PARTITION)
from s01_phases import corner_side

_SRC = Path(__file__).resolve().parents[1] / "src"
_EPS = 0.01


def _under(partition: str | None, code: str) -> subprocess.CompletedProcess:
    """`code` run in a fresh interpreter with the switch set to `partition` (None: unset)."""
    env = {k: v for k, v in os.environ.items() if k != "HUDL_ZONES"}
    if partition is not None:
        env["HUDL_ZONES"] = partition
    return subprocess.run([sys.executable, "-c", code], cwd=_SRC, env=env, capture_output=True,
                          text=True)


@pytest.mark.parametrize("partition", ["C", "F", "G2"])
def test_the_switch_names_the_partition_every_reader_gets(partition):
    out = _under(partition, "import zones; print(','.join(zones.zone_names()), zones.zone_cols('att')[0])")
    assert out.returncode == 0, out.stderr
    assert out.stdout.split() == [",".join(zones.ZONES[partition]),
                                  f"{partition}_att_{zones.ZONES[partition][0]}"]


def test_the_default_is_g2_and_an_unknown_partition_stops_the_import():
    """G2 is the default, the paper's partition, for the cut a club reads
    (data/derived/prereg_2026-10-13/deviation_2026-10-25.md)."""
    assert _under(None, "import constants; print(constants.ZONE_PARTITION)").stdout.strip() == "G2"
    bad = _under("G", "import zones")
    assert bad.returncode != 0 and "HUDL_ZONES must be one of" in bad.stderr


def test_the_side_is_read_off_the_kick_as_s01_reads_it():
    ys = np.linspace(0.0, 80.0, 801)
    assert list(zones.kicked_from_right(ys)) == [corner_side(v) == "right" for v in ys]
    assert zones.canonical_y(np.array([30.0]), True)[0] == 50.0
    assert zones.canonical_y(np.array([30.0]), False)[0] == 30.0


def test_c_s_cells_fall_where_the_laws_put_the_lines():
    x = np.array([116.0, 110.0, 104.0, 104.0, 99.0, 90.0, 116.0])
    y = np.array([35.0, 40.0, 50.0, 10.0, 40.0, 40.0, 44.0])
    assert list(zones.cells(x, y)) == ["six_near", "spot_central", "front_far", "wide", "edge",
                                       "deep", "six_far"]


def test_c_s_edge_cell_is_the_front_of_the_box_inside_its_width():
    """A body in front of the box but out by the touchline is wide of the box, not at its edge;
    counted as `edge` it would read as a rebound-zone body."""
    x, y = np.array([99.0, 99.0, 99.0, 95.0]), np.array([40.0, 10.0, 70.0, 10.0])
    assert list(zones.cells(x, y)) == ["edge", "wide", "wide", "deep"]


def test_f_s_zones_fall_either_side_of_each_line():
    """The six-yard box runs its own depth past each post; the rest of the box splits at the posts;
    `short` is wide of the box on the kick's side, and the far side's wide ground is `rest`."""
    near, far = GOAL_POSTS_SB[0][1], GOAL_POSTS_SB[1][1]
    x = np.array([SIX_YARD_X, SIX_YARD_X, SIX_YARD_X - _EPS, BOX_X, BOX_X, HOLD_X, BOX_X, BOX_X,
                  HOLD_X - _EPS])
    y = np.array([SIX_YARD_Y[0], SIX_YARD_Y[0] - _EPS, near, near - _EPS, far + _EPS, 40.0,
                  BOX_Y[0] - _EPS, BOX_Y[1] + _EPS, 40.0])
    assert list(zones.zones_f(x, y)) == ["six_yard", "near_post", "central", "near_post",
                                         "far_post", "frontal", "short", "rest", "rest"]


def test_g2_s_cells_fall_where_the_sheet_cuts_them():
    """Three equal columns of the six-yard width over the six and the spot rows, the spot row to 15 yd
    off the goal line and the central block short of it in the `edge` with the frontal strip, a cell on
    each side over the box's depth, and C's cells outside the box."""
    x = np.array([116.5, 116.5, 116.5, 110.0, 105.0, 104.99, 103.0, 99.0, 116.5, 104.0, 110.0, 99.0,
                  110.0, 80.0])
    y = np.array([33.0, 40.0, 47.0, 33.0, 47.0, 40.0, 49.0, 40.0, 24.0, 25.0, 56.0, 10.0, 70.0, 40.0])
    assert list(zones.zones_g2(x, y)) == ["six_near", "six_central", "six_far", "spot_near", "spot_far",
                                          "edge", "edge", "edge", "near_side", "near_side", "far_side",
                                          "wide", "wide", "deep"]


def test_a_job_reads_the_width_of_the_box():
    """In the box attacks the cross, in front of it inside its width stakes the drop, and anything
    else holds back. Depth alone calls a body by the flag at x = 110 a crosser."""
    x = np.array([110.0, 110.0, 99.0, 99.0, 90.0, BOX_X, BOX_X - _EPS, HOLD_X, HOLD_X - _EPS])
    y = np.array([40.0, 10.0, 40.0, 70.0, 40.0, BOX_Y[0], 40.0, BOX_Y[1], 40.0])
    assert list(zones.job_of(x, y)) == ["attack_cross", "hold_back", "stake_drop", "hold_back",
                                        "hold_back", "attack_cross", "stake_drop", "stake_drop",
                                        "hold_back"]


@pytest.mark.parametrize("partition", ["C", "F"])
def test_the_frontal_is_the_same_ground_under_both_partitions_and_the_jobs(partition):
    """Under C and F the frontal is the strip in front of the box inside its width: C's `edge`, F's
    `frontal` and the stake-drop job."""
    code = ("import numpy as np, zones\n"
            "rng = np.random.default_rng(0)\n"
            "x, y = rng.uniform(85, 120, 20000), rng.uniform(0, 80, 20000)\n"
            "f = zones.frontal(x, y)\n"
            "assert (f == (zones.cells(x, y) == 'edge')).all()\n"
            "assert (f == (zones.zones_f(x, y) == 'frontal')).all()\n"
            "assert (f == (zones.job_of(x, y) == 'stake_drop')).all()\n"
            "print('ok')")
    out = _under(partition, code)
    assert out.returncode == 0 and out.stdout.strip() == "ok", out.stderr


def test_under_g2_the_frontal_the_edge_and_staking_the_drop_are_one_ground():
    """G2 for both sides and for the jobs: the frontal, G2's `edge` and the stake-drop job are
    the same ground, the central block of the box included, and the defenders in it are `n_def_frontal`."""
    code = ("import numpy as np, zones\n"
            "rng = np.random.default_rng(0)\n"
            "x, y = rng.uniform(85, 120, 20000), rng.uniform(0, 80, 20000)\n"
            "f = zones.frontal(x, y)\n"
            "assert (f == (zones.zones_g2(x, y) == 'edge')).all()\n"
            "assert (f == (zones.job_of(x, y) == 'stake_drop')).all()\n"
            "assert zones.job_of(np.array([103.0]), np.array([40.0]))[0] == 'stake_drop'\n"
            "c = zones.table_counts(np.array([110.0]), np.array([40.0]), np.array([103.0, 99.0]),\n"
            "                       np.array([40.0, 30.0]), kick_y=0.1)\n"
            "assert c['n_def_frontal'] == c['G2_def_edge'] == 2 and c['C_def_edge'] == 1\n"
            "print('ok')")
    out = _under("G2", code)
    assert out.returncode == 0 and out.stdout.strip() == "ok", out.stderr


def test_the_table_counts_every_partition_and_mirrors_by_the_kick():
    """Every body lands once per partition on each side, the defenders' frontal included; the defenders
    in the switch's frontal are `n_def_frontal`; and a delivery from the right flag counts as its mirror
    image from the left. The defender at (103.5, 40) stands in G2's `edge` and in C's front row."""
    xa, ya = np.array([116.0, 110.0, 99.0, 90.0]), np.array([33.0, 47.0, 40.0, 60.0])
    xd, yd = np.array([117.0, 100.0, 104.0, 103.5]), np.array([40.0, 41.0, 10.0, 40.0])
    left = zones.table_counts(xa, ya, xd, yd, kick_y=0.1)
    assert left == zones.table_counts(xa, 80.0 - ya, xd, 80.0 - yd, kick_y=79.9)
    assert list(left) == zones.table_columns()
    for p in zones.ZONES:
        assert sum(left[f"{p}_att_{z}"] for z in zones.ZONES[p]) == len(xa)
        assert sum(left[f"{p}_def_{z}"] for z in zones.ZONES[p]) == len(xd)
    p = ZONE_PARTITION
    assert left["n_def_frontal"] == left[f"{p}_def_{zones.FRONTAL[p]}"]
    assert left["C_def_edge"] == 1 and left["F_def_short"] == 1 and left["C_def_wide"] == 1
    assert left["G2_def_edge"] == 2 and left["C_def_front_central"] == 1


def _reads(partition: str, built: list | None) -> subprocess.CompletedProcess:
    """A table built under `built` (None: without the column) read under `partition`."""
    table = {"n_def_frontal": [1]} if built is None else {"zone_partition": built}
    return _under(partition, f"import pandas as pd, zones\nzones.require_built_under(pd.DataFrame({table!r}))")


def test_a_table_is_read_under_the_partition_it_was_built_under():
    """C and F share the frontal, so a table built under one is read under the other; G2's is wider, so
    a table built under G2 is not read under C nor one built under C under G2, and one without the
    column passes here."""
    assert _reads("C", ["F", "C"]).returncode == 0 and _reads("C", None).returncode == 0
    assert _reads("G2", ["G2"]).returncode == 0
    for partition, built in (("C", ["G2"]), ("G2", ["C"])):
        out = _reads(partition, built)
        assert out.returncode != 0 and f"rebuild it with HUDL_ZONES={partition}" in out.stderr


# Where a partition may be named: its owner, the switch, the comparison that reads all of them, the
# recipe, which prints the switch it runs under, the recipe's gate on the tables, which holds every
# partition's counts to every body, S's fit, which draws S on C's cells as its sheet registers it
# (data/derived/prereg_2026-10-07/zones_s.md), S's comparison against C, and G's, which draws its two
# rivals on C's cells outside the box as its sheet registers it (data/derived/prereg_2026-10-13/zones_g.md).
_MAY_NAME_A_PARTITION = {"src/zones.py", "src/constants.py", "pipeline/run_zone_comparison.py",
                         "pipeline/run_e2e.py", "pipeline/compare_tables.py", "pipeline/fit_zones_s.py",
                         "pipeline/run_zone_s.py", "pipeline/run_zone_g.py"}
_PARTITION_NAMES = {"cells", "zones_f", "zones_g2", "ZONES", "FRONTAL"}


def _hand_named(path: Path) -> list:
    """Lines of `path` that fix a partition: one of its labellers or tables taken from zones, a
    column literal of one partition, or the switch's variable written by name."""
    hits = []
    for node in ast.walk(ast.parse(path.read_text())):
        if (isinstance(node, ast.ImportFrom) and node.module == "zones"
                and {a.name for a in node.names} & _PARTITION_NAMES) \
                or (isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name)
                    and node.value.id == "zones" and node.attr in _PARTITION_NAMES) \
                or (isinstance(node, ast.Constant) and isinstance(node.value, str)
                    and (re.match(r"(C|F|G2)_(att|def)_", node.value) or node.value == "HUDL_ZONES")):
            hits.append(node.lineno)
    return hits


def test_no_consumer_fixes_the_partition_by_hand():
    """Every consumer reaches the plan's zones through the switch (`zone_of`, `zone_names`,
    `zone_cols`), so a pass under C and one under F differ in HUDL_ZONES alone. The scan has to
    see the comparison's own uses, or it is blind."""
    repo = _SRC.parent
    files = sorted([*repo.glob("src/**/*.py"), *repo.glob("pipeline/*.py"), *repo.glob("renders/*.py")])
    hits = {p.relative_to(repo).as_posix(): _hand_named(p) for p in files}
    assert hits["pipeline/run_zone_comparison.py"], "the scan misses the uses it exists to find"
    hand = {f: lines for f, lines in hits.items() if lines and f not in _MAY_NAME_A_PARTITION}
    assert not hand, f"the partition is fixed by hand at {hand}"


def test_the_switch_s_columns_leave_the_defenders_frontal_to_its_own_group():
    p = ZONE_PARTITION
    att, dfd = zones.zone_cols("att"), zones.zone_cols("def")
    assert f"{p}_def_{zones.FRONTAL[p]}" not in dfd and f"{p}_att_{zones.FRONTAL[p]}" in att
    assert len(att) == len(zones.zone_names()) == len(dfd) + 1
    assert set(att) | set(dfd) <= set(zones.table_columns())
