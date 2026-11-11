"""Smoke tests for pipeline/ and renders/, the layer the module tests never import.

Every other test targets src/, so a driver can rot with the suite green. Two failure classes
do it: a dead import, and a call to a method that no longer exists. The import sweep below
catches the first for EVERY driver; the execution tests catch the second and prove the
artefact is written.

Cost note: each subprocess that touches tracking pays a full `load_tracking`, whose docstring
carries the peak memory, so they run one at a time and on the smallest window that still
exercises the path.
"""
from __future__ import annotations

import importlib.util
import subprocess
import sys
from pathlib import Path
from types import ModuleType

import pytest

from conftest import requires_data, requires_sb

# DERIVED FROM THE TREE, not listed by hand: a written list stops covering the day a driver is added.
_REPO = Path(__file__).resolve().parents[1]
PIPELINE = sorted(p.stem for p in (_REPO / "pipeline").glob("*.py"))
RENDERS = sorted(p.stem for p in (_REPO / "renders").glob("*.py"))

_TRACKED_MATCH = 4004219        # the conftest `mid`: has events, 360 and tracking

# The record is generated from the private research base and is not in the repository, so a clone
# skips the tests that read it once, with the reason, instead of failing on a missing file.
requires_record = pytest.mark.skipif(
    not (_REPO / "docs" / "measured.json").exists(),
    reason="docs/measured.json is the generated record of the private research base and is not in "
           "the repository; regenerate it with pipeline/report_figures.py")

# The optional groups of pyproject.toml, by the module that imports one at module level. Without its group the
# sweep skips the module with the reason, as the module's own tests do; any other missing import still fails it.
_OPTIONAL = {"run_frame_gpu": ("torch",), "run_frame_gnn": ("torch",), "render_paper": ("docx", "lxml")}


def _import_module(path: Path, name: str) -> ModuleType:
    """Import a driver by file path, running its top-level (sys.path setup + imports).

    The module is registered in sys.modules BEFORE it executes, exactly as a real
    import does. Without that, a driver defining a dataclass fails here and nowhere
    else: dataclasses resolves string annotations through sys.modules[cls.__module__],
    which a bare exec_module never populates."""
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    try:
        spec.loader.exec_module(module)
    finally:
        sys.modules.pop(name, None)
    return module


def _run(repo_root: Path, *args: str, timeout: int = 1800) -> subprocess.CompletedProcess:
    """Run a driver as a subprocess and return it, failing loudly with its stderr."""
    proc = subprocess.run([sys.executable, *args], cwd=repo_root, timeout=timeout,
                          capture_output=True, text=True)
    assert proc.returncode == 0, f"{args[0]} failed:\n{proc.stdout}\n{proc.stderr}"
    return proc


# ── Import sweep: every driver must at least load ────────────────────────
@pytest.mark.parametrize("name", PIPELINE)
def test_pipeline_module_imports(repo_root, name):
    """A driver whose imports rotted is dead code the rest of the suite cannot see."""
    for package in _OPTIONAL.get(name, ()):
        pytest.importorskip(package)
    _import_module(repo_root / "pipeline" / f"{name}.py", f"_smoke_pipeline_{name}")


@pytest.mark.parametrize("name", RENDERS)
def test_render_module_imports(repo_root, name):
    """Same sweep over renders/."""
    for package in _OPTIONAL.get(name, ()):
        pytest.importorskip(package)
    _import_module(repo_root / "renders" / f"{name}.py", f"_smoke_render_{name}")


# ── Execution on the granted data ────────────────────────────────────────
@requires_sb
def test_build_table_runs(repo_root, tmp_path):
    """build_table produces a cached corner table over a small match subset."""
    out = tmp_path / "corners.parquet"
    _run(repo_root, "pipeline/build_table.py", "--source", "serie_a_sample",
         "--limit", "5", "--out", str(out))
    assert out.exists()


@requires_sb
def test_build_table_parallel_matches_serial(repo_root, tmp_path):
    """The parallel table must reproduce the serial one exactly, dtypes and order.

    Not a nicety: row order drives GroupKFold fold membership and saga's update
    order, so a reordered or dtype-promoted table shifts every headline log-loss in
    the third decimal with no error anywhere. It holds only because the workers
    return dicts and the parent builds ONE frame; if someone later 'simplifies'
    corner_rows to return a DataFrame, this is the test that catches it."""
    import pandas as pd
    a, b = tmp_path / "serial.parquet", tmp_path / "par.parquet"
    for path, jobs in ((a, "1"), (b, "4")):
        _run(repo_root, "pipeline/build_table.py", "--source", "serie_a_sample",
             "--limit", "5", "--jobs", jobs, "--out", str(path))
    pd.testing.assert_frame_equal(pd.read_parquet(a), pd.read_parquet(b))


def test_the_naming_rule_keeps_every_population_on_its_own_artefact(repo_root, tmp_path):
    """Corners keep the published names; the flank free kicks are suffixed; only a grant table of one
    kind is ever treated as publishable. The model drivers write their records through this."""
    bt = _import_module(repo_root / "pipeline" / "build_table.py", "_smoke_naming")
    fk = "free_kick_flank"
    assert bt.kind_tag(("corner",)) == "" and bt.kind_tag((fk,)) == f"_{fk}"
    assert bt.published_tag(bt._DER / "corners.parquet") == ""
    assert bt.published_tag(bt._DER / f"corners_{fk}.parquet") == f"_{fk}"
    for other in (bt._DER / "corners_corner_free_kick_flank.parquet",
                  bt._DER / "corners_limit40.parquet", tmp_path / "corners.parquet"):
        assert bt.published_tag(other) is None


def test_the_f_pass_folder_is_published_and_no_other_beside_it(repo_root, tmp_path):
    """The recipe's pass under partition F writes the published names into zones_F, so a guard
    reading published_dir keeps a reduced run off them there too; the smoke folders are not
    published."""
    bt = _import_module(repo_root / "pipeline" / "build_table.py", "_smoke_published_dir")
    assert bt.published_dir(bt._DER) and bt.published_dir(bt.ZONES_F)
    for other in (bt._DER / "smoke", bt._DER / "smoke" / "zones_F", tmp_path):
        assert not bt.published_dir(other)


@requires_sb
def test_build_table_free_kick_parallel_matches_serial(repo_root, tmp_path):
    """The flank free-kick table has to reproduce across --jobs for the reason the corner one does."""
    import pandas as pd
    a, b = tmp_path / "serial.parquet", tmp_path / "par.parquet"
    for path, jobs in ((a, "1"), (b, "4")):
        _run(repo_root, "pipeline/build_table.py", "--source", "serie_a_sample", "--limit", "10",
             "--jobs", jobs, "--kinds", "free_kick_flank", "--out", str(path))
    ta = pd.read_parquet(a)
    assert len(ta) and set(ta["set_piece_type"]) == {"free_kick_flank"}
    pd.testing.assert_frame_equal(ta, pd.read_parquet(b))


@requires_sb
def test_eda_build_free_kick_parallel_matches_serial(repo_root, tmp_path):
    """The delivery grain is the EDA's denominator, so it cannot depend on --jobs either."""
    import pandas as pd
    for sub, jobs in (("serial", "1"), ("par", "4")):
        _run(repo_root, "pipeline/eda_build.py", "--source", "serie_a_sample", "--limit", "10",
             "--jobs", jobs, "--kinds", "free_kick_flank", "--out-dir", str(tmp_path / sub))
    for name in ("eda_deliveries", "eda_matches"):
        f = f"{name}_free_kick_flank_serie_a_sample_limit10.parquet"
        pd.testing.assert_frame_equal(pd.read_parquet(tmp_path / "serial" / f),
                                      pd.read_parquet(tmp_path / "par" / f))


@requires_sb
def test_build_table_refuses_a_table_missing_a_declared_column(repo_root, tmp_path, monkeypatch):
    """The published table is assembled by this driver, so s03's column guard has to run here.

    build_table builds the DataFrame itself rather than through build_corner_table, and a guard
    only the library path calls protects nothing the pipeline publishes. The driver runs
    in-process with corner_rows stripped of taker_foot, one of the columns _DELIVERY_CAT names,
    and has to raise before anything is written."""
    bt = _import_module(repo_root / "pipeline" / "build_table.py", "_smoke_build_table_guard")
    real = bt.corner_rows

    def without_taker_foot(*args):
        return [{k: v for k, v in row.items() if k != "taker_foot"} for row in real(*args)]

    out = tmp_path / "corners.parquet"
    monkeypatch.setattr(bt, "corner_rows", without_taker_foot)
    monkeypatch.setattr(sys, "argv", ["build_table.py", "--source", "serie_a_sample",
                                      "--limit", "3", "--out", str(out)])
    with pytest.raises(ValueError, match="taker_foot"):
        bt.main()
    assert not out.exists()


@requires_sb
def test_a_source_run_never_lands_on_the_published_name(repo_root, tmp_path, monkeypatch):
    """--source narrows the grant exactly as --limit does, so without --out it writes a tagged name.

    `corners.parquet` is the name `published_tag` reads as the grant, and every model driver
    publishes from it, so a one-league table written there would be published as the five. The
    driver runs in-process on two matches of the sample, with its output directory moved."""
    bt = _import_module(repo_root / "pipeline" / "build_table.py", "_smoke_build_table_source")
    bt._league_index()      # a process-wide cache off the catalogue: filled before the cut below
    real = bt.L.list_matches
    monkeypatch.setattr(bt.L, "list_matches", lambda source=None: real(source).head(2))
    monkeypatch.setattr(bt, "_DER", tmp_path)
    monkeypatch.setattr(sys, "argv", ["build_table.py", "--source", "serie_a_sample"])
    bt.main()
    assert not (tmp_path / "corners.parquet").exists()
    assert (tmp_path / "corners_serie_a_sample.parquet").exists()


@requires_sb
def test_audit_360_runs(repo_root, tmp_path):
    """The 360 coverage audit runs and writes its parquet."""
    out = tmp_path / "audit.parquet"
    _run(repo_root, "pipeline/audit_360.py", "--source", "serie_a_sample",
         "--limit", "3", "--out", str(out))
    assert out.exists()


@requires_sb
def test_an_audit_of_one_source_never_lands_on_the_grant_name(repo_root, tmp_path, monkeypatch):
    """`audit360.parquet` is the coverage record report_figures publishes for the five leagues, so
    --source tags the default name as --limit does. In-process on three sample matches."""
    a3 = _import_module(repo_root / "pipeline" / "audit_360.py", "_smoke_audit360_source")
    real = a3.L.list_matches
    monkeypatch.setattr(a3.L, "list_matches", lambda source=None: real(source).head(3))
    monkeypatch.setattr(a3, "_REPO", tmp_path)
    monkeypatch.setattr(sys, "argv", ["audit_360.py", "--source", "serie_a_sample"])
    a3.main()
    der = tmp_path / "data" / "derived"
    assert not (der / "audit360.parquet").exists()
    assert (der / "audit360_serie_a_sample.parquet").exists()


@requires_sb
def test_an_audit_of_another_population_counts_it_under_its_own_name(repo_root, tmp_path,
                                                                     monkeypatch):
    """--kinds audits the population it names, its deliveries counted, under a tagged default name:
    the flank free kicks' coverage is not the corners' record. In-process on three matches."""
    import pandas as pd
    a3 = _import_module(repo_root / "pipeline" / "audit_360.py", "_smoke_audit360_kinds")
    real = a3.L.list_matches
    monkeypatch.setattr(a3.L, "list_matches", lambda source=None: real(source).head(3))
    monkeypatch.setattr(a3, "_REPO", tmp_path)
    monkeypatch.setattr(sys, "argv", ["audit_360.py", "--kinds", "free_kick_flank"])
    a3.main()
    der = tmp_path / "data" / "derived"
    assert not (der / "audit360.parquet").exists()
    got = pd.read_parquet(der / "audit360_free_kick_flank.parquet")
    from s01_phases import SetPieceGrammar
    want = sum(len(SetPieceGrammar(a3.L.load_events(int(m)), kinds=("free_kick_flank",)).deliveries())
               for m in got["match_id"])
    assert int(got["corners"].sum()) == want


@requires_sb
def test_audit_360_parallel_matches_serial(repo_root, tmp_path):
    """The coverage audit gates every kinetic claim, so its numbers cannot depend on --jobs.

    Same reason as the table above: the workers return dicts and the parent builds ONE frame.
    A concat of per-match frames would infer dtypes per match and move the aggregate."""
    import pandas as pd
    a, b = tmp_path / "serial.parquet", tmp_path / "par.parquet"
    for path, jobs in ((a, "1"), (b, "4")):
        _run(repo_root, "pipeline/audit_360.py", "--source", "serie_a_sample",
             "--limit", "6", "--jobs", jobs, "--out", str(path))
    pd.testing.assert_frame_equal(pd.read_parquet(a), pd.read_parquet(b))


@pytest.fixture(scope="module")
def small_table(repo_root: Path, tmp_path_factory: pytest.TempPathFactory) -> Path:
    """A 5-match corner table, built ONCE for every model-driver test in this module."""
    out = tmp_path_factory.mktemp("tables") / "corners.parquet"
    _run(repo_root, "pipeline/build_table.py", "--source", "serie_a_sample",
         "--limit", "5", "--out", str(out))
    return out


@requires_sb
@pytest.mark.parametrize("driver", ["run_value.py", "run_causal.py"])
def test_model_driver_runs(repo_root, small_table, driver):
    """The model drivers run off a cached table. A tiny subset makes them report
    non-operative rather than fit, which is the honest guard and still exercises the path."""
    _run(repo_root, f"pipeline/{driver}", "--table", str(small_table))


@pytest.fixture(scope="module")
def small_flank_table(repo_root: Path, tmp_path_factory: pytest.TempPathFactory) -> Path:
    """The whole tracked sample's flank free kicks, the smallest table that still has some."""
    out = tmp_path_factory.mktemp("tables") / "corners_flank.parquet"
    _run(repo_root, "pipeline/build_table.py", "--source", "serie_a_sample", "--limit", "30",
         "--kinds", "free_kick_flank", "--out", str(out))
    return out


@requires_sb
@pytest.mark.parametrize("driver", ["run_value.py", "run_causal.py"])
def test_model_driver_runs_on_the_flank_table(repo_root, small_flank_table, driver):
    """Same path on the flank free kicks; a table outside data/derived writes no record."""
    _run(repo_root, f"pipeline/{driver}", "--table", str(small_flank_table))


@requires_sb
def test_run_situation_runs(repo_root, tmp_path):
    """The four-state situation table builds and caches, with the radii record beside it."""
    out = tmp_path / "situation.parquet"
    _run(repo_root, "pipeline/run_situation.py", "--build", "--source", "serie_a_sample",
         "--limit", "5", "--out", str(out))
    assert out.exists() and (tmp_path / "situation_radii.parquet").exists()


@requires_sb
def test_frontier_drivers_run(repo_root, tmp_path):
    """The frontier drivers run end to end on a small subset.

    They are the ones that produce the artefacts the paper quotes, so an import-only
    check would not be enough: falsify_frontier fits a bootstrap, sweep_counter walks
    every cleared delivery, describe_frontier reads the shot grain.

    Every call passes --out. Without it these drivers fall back to a tagged default inside
    `data/derived/`, which leaves the suite writing into the directory the measurement
    record is built from."""
    _run(repo_root, "pipeline/falsify_frontier.py", "--source", "serie_a_sample",
         "--limit", "6", "--jobs", "2", "--boot", "20", "--out", str(tmp_path / "f.parquet"))
    _run(repo_root, "pipeline/sweep_counter.py", "--limit", "6", "--jobs", "2",
         "--out", str(tmp_path / "sweep.parquet"))
    _run(repo_root, "pipeline/build_table.py", "--source", "serie_a_sample", "--limit", "6",
         "--jobs", "2", "--out", str(tmp_path / "c.parquet"))
    # The shots of a subset need the delivery grain of the same matches: against the grant's, a
    # delivery the grammar enumerates differently would not be found and the origin guard would stop.
    _run(repo_root, "pipeline/eda_build.py", "--source", "serie_a_sample", "--limit", "6",
         "--jobs", "2", "--out-dir", str(tmp_path))
    _run(repo_root, "pipeline/describe_frontier.py", "--boot", "20",
         "--shots", str(tmp_path / "c_shots.parquet"),
         "--deliveries", str(tmp_path / "eda_deliveries_serie_a_sample_limit6.parquet"),
         "--out", str(tmp_path / "frontier.parquet"))


@requires_sb
def test_the_frontier_is_canonical_only_off_the_grant_table(repo_root, tmp_path, monkeypatch):
    """A shot table named like the grant's but sitting anywhere else is a subset.

    frontier.parquet is what report_figures summarises, so the canonical input is decided by its
    path, the way `published_tag` decides it for the corner tables, and not by its file name. The
    subset here is the corners of six sample matches, with the output directory moved."""
    sub = tmp_path / "subset"
    _run(repo_root, "pipeline/build_table.py", "--source", "serie_a_sample", "--limit", "6",
         "--out", str(sub / "corners_corner_free_kick_flank.parquet"))
    _run(repo_root, "pipeline/eda_build.py", "--source", "serie_a_sample", "--limit", "6",
         "--out-dir", str(sub))
    df = _import_module(repo_root / "pipeline" / "describe_frontier.py", "_smoke_frontier_name")
    monkeypatch.setattr(df, "_DER", tmp_path / "derived")
    monkeypatch.setattr(sys, "argv", [
        "describe_frontier.py", "--boot", "20",
        "--shots", str(sub / "shots_corner_free_kick_flank.parquet"),
        "--deliveries", str(sub / "eda_deliveries_serie_a_sample_limit6.parquet"),
        "--situation", str(sub / "absent.parquet")])
    df.main()
    assert not (tmp_path / "derived" / "frontier.parquet").exists()
    assert (tmp_path / "derived" / "frontier_shots_corner_free_kick_flank.parquet").exists()


@requires_sb
def test_probe_throw_ins_sweeps_and_stops_below_the_floor(repo_root, tmp_path):
    """The throw-in probe sweeps its predicate, then refuses to interpret a thin sample.

    Asserting a clean exit here would be wrong: on a handful of matches the selected long throws
    produce fewer shots than MIN_CELL, and the driver is SUPPOSED to stop rather than print an
    obs/exp over a dozen shots. So the smoke test pins both halves: the sweep runs and the floor
    bites, the behaviour a subset must produce."""
    proc = subprocess.run(
        [sys.executable, "pipeline/probe_throw_ins.py", "--limit", "12", "--jobs", "2",
         "--boot", "20", "--out", str(tmp_path / "probe.parquet")],
        cwd=repo_root, timeout=1800, capture_output=True, text=True)
    assert "Predicate sweep" in proc.stdout, f"the sweep did not run:\n{proc.stdout}\n{proc.stderr}"
    assert "in_box_pct" in proc.stdout, "the sweep printed no precision column"
    assert proc.returncode == 0 or "under the publication floor" in proc.stderr, (
        f"probe_throw_ins failed for a reason other than the floor:\n{proc.stdout}\n{proc.stderr}")


@requires_sb
def test_a_probe_on_another_shot_table_never_lands_on_the_record(repo_root, tmp_path, monkeypatch):
    """`throw_in_probe.parquet` is where eda_throw_ins and report_figures read the adopted cell and
    the comparison, and the comparison divides the grant's two-population shots. Another --shots
    table, here the two populations of the grant's first six matches, tags the default name, as
    --limit does. In-process on the smoke's 150 matches, with the derived directory moved."""
    _run(repo_root, "pipeline/build_table.py", "--limit", "6", "--kinds", "corner,free_kick_flank",
         "--out", str(tmp_path / "corners_corner_free_kick_flank.parquet"))
    shots = tmp_path / "shots_corner_free_kick_flank.parquet"
    pt = _import_module(repo_root / "pipeline" / "probe_throw_ins.py", "_smoke_probe_shots")
    real = pt.L.list_matches
    monkeypatch.setattr(pt.L, "list_matches", lambda source=None: real(source).head(150))
    monkeypatch.setattr(pt, "_DER", tmp_path / "derived")
    monkeypatch.setattr(sys, "argv", ["probe_throw_ins.py", "--jobs", "8", "--boot", "20",
                                      "--shots", str(shots)])
    pt.main()
    assert not (tmp_path / "derived" / "throw_in_probe.parquet").exists()
    assert (tmp_path / "derived" / "throw_in_probe_shots_corner_free_kick_flank.parquet").exists()


@requires_data
def test_run_race_runs(repo_root):
    """run_race races every corner of one tracked match."""
    proc = _run(repo_root, "pipeline/run_race.py", "--match", str(_TRACKED_MATCH))
    assert "corners raced" in proc.stdout


@requires_data
def test_render_race_writes_png(repo_root, tmp_path):
    """render_race writes the reach-field snapshot."""
    out = tmp_path / "race.png"
    _run(repo_root, "renders/render_race.py", "--match", str(_TRACKED_MATCH),
         "--index", "0", "--out", str(out))
    assert out.stat().st_size > 0


@requires_data
def test_render_window_draws_the_curve_of_the_log_and_writes_png(repo_root, tmp_path):
    """The annex figure reads the late-boundary log, one row per second from the first, and writes its PNG with the SVG
    beside it."""
    if not (repo_root / "data" / "derived" / "auditoria_2026-10-12" / "evidence" / "late_boundary_curve.log").exists():
        pytest.skip("the late-boundary log is not there")
    out = tmp_path / "window.png"
    _run(repo_root, "renders/render_window.py", "--out", str(out))
    assert out.stat().st_size > 0 and out.with_suffix(".svg").exists()


@requires_data
def test_render_corner_writes_png(repo_root, tmp_path):
    """render_corner writes the delivery snapshot."""
    out = tmp_path / "corner.png"
    _run(repo_root, "renders/render_corner.py", "--match", str(_TRACKED_MATCH),
         "--index", "0", "--out", str(out))
    assert out.stat().st_size > 0


@requires_sb
def test_signature_render_writes_png(repo_root, tmp_path):
    """The 360-only render needs no tracking, so it stays cheap."""
    out = tmp_path / "signature.png"
    _run(repo_root, "renders/render_signature_sb.py", "--match", "4004239", "--index", "0",
         "--out", str(out))
    assert out.stat().st_size > 0


@requires_data
@requires_record
def test_render_scope_writes_png_from_the_record(repo_root, tmp_path):
    """The scope figure prints the record's figures: every population it draws has a scope entry
    there, and the figure is written."""
    import json
    rs = _import_module(repo_root / "renders" / "render_scope.py", "_smoke_scope_record")
    record = json.loads((repo_root / "docs" / "measured.json").read_text())["figures"]
    pops = rs.populations(record)
    assert [p["scope"] for p in pops] == [record["scope"][k]
                                          for k in ("corner", "free_kick_flank", "long_throw")]
    out = tmp_path / "scope.png"
    _run(repo_root, "renders/render_scope.py", "--out", str(out))
    assert out.stat().st_size > 0


@requires_record
def test_render_scope_counts_the_goals_after_the_first_touch_with_their_own_goals(repo_root):
    """Panel B's bar is every goal after the first touch, an own goal included: on the corners the
    count the team model fits, and on each population the shot goals plus the own goals the record
    keeps apart."""
    import json
    record = json.loads((repo_root / "docs" / "measured.json").read_text())["figures"]
    rs = _import_module(repo_root / "renders" / "render_scope.py", "_smoke_scope_goals")
    corner = record["corpus"]["by_segment"]
    if record.get("team_shrinkage"):
        assert rs.after_first_touch_goals(corner) == record["team_shrinkage"]["goals"]
    toy = {"direct": {"goals": 9, "own_goals": 2}, "prolongation": {"goals": 4, "own_goals": 1},
           "after_clearance": {"goals": 5, "own_goals": 0}, "late": {"goals": 3, "own_goals": 1}}
    assert rs.after_first_touch_goals(toy) == 10


@requires_record
def test_render_top_teams_draws_the_record_s_rates_and_writes_png(repo_root, tmp_path):
    """The bars are the record's pooled rates, and the two differences the panels quote are those
    rates' differences: corners per match the secondary contrast, after the first touch the
    registered T2 point."""
    import json
    record = json.loads((repo_root / "docs" / "measured.json").read_text())["figures"]
    if not record.get("top_teams"):
        pytest.skip("eda_top_teams.py has not been run")
    rt = _import_module(repo_root / "renders" / "render_top_teams.py", "_smoke_top_teams")
    n, r = rt.numbers(record["top_teams"]), record["top_teams"]["rates"]
    assert abs(r["T2"]["corners_per_match"] - r["rest"]["corners_per_match"] - n["volume"][0]) <= 5e-4
    assert abs(r["T2"]["aft_xg_per_corner"] - r["rest"]["aft_xg_per_corner"] - n["aft"][0]) <= 5e-5
    for lo, point, hi in ((n["volume"][1], n["volume"][0], n["volume"][2]),
                          (n["aft"][1], n["aft"][0], n["aft"][2])):
        assert lo <= point <= hi
    out = tmp_path / "top_teams.png"
    _run(repo_root, "renders/render_top_teams.py", "--out", str(out))
    assert out.stat().st_size > 0


@requires_record
def test_render_flank_cut_reads_every_band_and_cut_from_the_record(repo_root, tmp_path):
    """Marks and numbers both come from the flank_cut block: every band but the reference, in order
    of x, and each cut printed as stored, so the text read back is the record's number."""
    import json
    record = json.loads((repo_root / "docs" / "measured.json").read_text())["figures"]
    if not record.get("flank_cut"):
        pytest.skip("run_flank_cut.py has not been run")
    rf = _import_module(repo_root / "renders" / "render_flank_cut.py", "_smoke_flank_cut")
    block = record["flank_cut"]
    edges = [lo for lo, _ in rf.bands(block)]
    assert edges == sorted(edges) and len(edges) == len(block["bands"]) - 1
    for cut in block["cuts"].values():
        for c in (v for v in cut.values() if isinstance(v, dict)):
            assert [float(rf.num(x)) for x in (c["below_minus_above"], *c["ci"])] == [
                c["below_minus_above"], *c["ci"]]
    out = tmp_path / "flank_cut.png"
    _run(repo_root, "renders/render_flank_cut.py", "--out", str(out))
    assert out.stat().st_size > 0


@requires_record
def test_render_plan_premium_reads_the_record_s_states_and_readings_and_writes_png(repo_root, tmp_path):
    """The grid prints the situation block's four states, whose two compared means the situation
    table reproduces on the rows the drop_position block counts; the raw and regression readings are
    the same in every cell size, and the raw one is the situation contrast it repeats."""
    import json
    import pandas as pd
    record = json.loads((repo_root / "docs" / "measured.json").read_text())["figures"]
    if not record.get("drop_position"):
        pytest.skip("run_drop_position.py has not been run")
    rp = _import_module(repo_root / "renders" / "render_plan_premium.py", "_smoke_plan_premium")
    drop, sit = record["drop_position"], record["situation"]
    assert {key for key, *_ in rp.STATES} == set(sit["by_state"])
    rs = rp.readings(drop)
    assert len(rs) == len(drop["by_cell"]) + 2
    for cell in drop["by_cell"].values():
        assert (cell["raw"], cell["adjusted"]) == (rs[0][1], rs[-1][1])
    assert abs(rs[0][1]["gap"] - sit["contrasts"]["won_as_planned-rescued"]["delta"]) <= 1e-4
    table = repo_root / "data" / "derived" / "situation.parquet"
    if table.exists():
        t = pd.read_parquet(table)
        rows = t[t["four_state"].isin(("won_as_planned", "rescued")) & t["drop_x"].notna()
                 & t["drop_y"].notna()]
        counts = rows["four_state"].value_counts()
        assert (len(rows), counts["won_as_planned"], counts["rescued"]) == (
            drop["rows"], drop["planned"], drop["rescued"])
        for state in ("won_as_planned", "rescued"):
            mean = rows.loc[rows["four_state"] == state, "value_xg_after"].mean()
            assert abs(mean - sit["by_state"][state]["xg"]) <= 5e-5, state
    out = tmp_path / "plan_premium.png"
    _run(repo_root, "renders/render_plan_premium.py", "--out", str(out))
    assert out.stat().st_size > 0


@requires_record
def test_render_goals_xg_draws_the_record_s_ratios_by_phase_and_body_part_and_writes_png(repo_root, tmp_path):
    """Each row is a goals-over-expected-goals ratio of the frontier block as stored, with its interval around it
    and its shots: the headline pair, the two body parts and the four cells of phase by body part. The group after
    the first touch holds the shots of its three segments, and the two body parts together hold every shot."""
    import json
    corner = json.loads((repo_root / "docs" / "measured.json").read_text())["figures"]["frontier"]["corner"]
    rg = _import_module(repo_root / "renders" / "render_goals_xg.py", "_smoke_goals_xg")
    rs = {r["key"]: r for r in rg.rows(corner)}
    seg = corner["segment"]
    assert len(rs) == 8
    assert (rs["direct"]["point"], rs["direct"]["shots"]) == (corner["headline"]["obs_exp_direct"]["point"], seg["direct"]["shots"])
    assert rs["after_first_touch"]["shots"] == sum(seg[s]["shots"] for s in ("flick_on", "prolongation", "after_clearance"))
    for key, cell in (("head", corner["body_part_pooled"]["head"]), ("non_head", corner["body_part_pooled"]["non_head"]),
                      ("direct|head", corner["body_part"]["direct|head"]),
                      ("after_first_touch|non_head", corner["body_part"]["after_first_touch|non_head"])):
        assert (rs[key]["point"], rs[key]["lo"], rs[key]["hi"], rs[key]["shots"]) == (cell["obs_exp"], *cell["ci"], cell["shots"]), key
    assert rs["head"]["shots"] + rs["non_head"]["shots"] == sum(v["shots"] for v in seg.values())
    assert all(r["lo"] <= r["point"] <= r["hi"] for r in rs.values())
    out = tmp_path / "goals_xg.png"
    _run(repo_root, "renders/render_goals_xg.py", "--out", str(out))
    assert out.stat().st_size > 0 and out.with_suffix(".svg").stat().st_size > 0


@requires_record
def test_render_conversion_draws_the_record_s_rates_and_writes_png(repo_root, tmp_path):
    """Each row is the falsification block's conversion as stored, its point inside its interval, and
    the after-first-touch rate, the three segments pooled, lies between theirs. Each row carries the goals and
    shots behind its rate, and the roll-up's are the three segments' sums, which give back its rate."""
    import json
    record = json.loads((repo_root / "docs" / "measured.json").read_text())["figures"]
    if not record.get("falsification"):
        pytest.skip("falsify_frontier.py has not been run")
    rc = _import_module(repo_root / "renders" / "render_conversion.py", "_smoke_conversion")
    conv = record["falsification"]["conversion"]
    rs = {r["key"]: r for r in rc.rows(conv)}
    for key, r in rs.items():
        c = conv[key]
        assert (r["point"], r["lo"], r["hi"]) == pytest.approx(
            (100 * c["point"], 100 * c["ci"][0], 100 * c["ci"][1])), key
        assert r["lo"] <= r["point"] <= r["hi"], key
    segs = [rs[k]["point"] for k in rc.PARTS]
    assert min(segs) <= rs["after_first_touch"]["point"] <= max(segs)
    for key in ("direct", *rc.PARTS):
        assert (rs[key]["n"], rs[key]["goals"]) == (conv[key]["n"], conv[key]["goals"]), key
    pooled = rs["after_first_touch"]
    assert (pooled["n"], pooled["goals"]) == tuple(sum(conv[k][f] for k in rc.PARTS) for f in ("n", "goals"))
    assert 100 * pooled["goals"] / pooled["n"] == pytest.approx(pooled["point"], abs=0.005)
    out = tmp_path / "conversion.png"
    _run(repo_root, "renders/render_conversion.py", "--out", str(out))
    assert out.stat().st_size > 0


@requires_record
def test_render_clock_draws_the_record_s_shares_of_goals_and_writes_both_versions(repo_root, tmp_path):
    """The bar's three parts are the shares of the goals of corners, own goals included, that the frontier block holds by
    segment: the shot that is the first touch itself, the three kinds after it and the late ones, which add up to one.
    The part after the first touch is the share the paper's text prints, and both versions are written as PNG and SVG."""
    import json
    record = json.loads((repo_root / "docs" / "measured.json").read_text())["figures"]
    rc = _import_module(repo_root / "renders" / "render_clock.py", "_smoke_clock")
    s, seg = rc.shares(record), record["frontier"]["corner"]["segment"]
    n = sum(v["goals"] + v["own_goals"] for v in seg.values())
    assert s["n"] == n and s["first"] + s["after"] + s["late"] == pytest.approx(1)
    assert s["first"] == pytest.approx((seg["direct"]["goals"] + seg["direct"]["own_goals"]) / n)
    assert s["after"] == pytest.approx(record["frontier"]["corner"]["headline"]["share_goals_after_first_touch"]["point"], abs=5e-4)
    for name, extra in (("paper", ()), ("talk", ("--talk",))):
        out = tmp_path / f"{name}.png"
        _run(repo_root, "renders/render_clock.py", "--out", str(out), *extra)
        assert out.stat().st_size > 0 and out.with_suffix(".svg").stat().st_size > 0, name


@requires_record
def test_render_information_draws_the_record_s_steps_and_writes_png(repo_root, tmp_path):
    """Each row is a step of the record as log loss saved, with the interval that decides its
    reading: the ladder's delivery step, the freeze-frame step's refit in normal form, the positive
    control's and the first contact's registered pivotal form, the last two stored as model minus
    baseline and printed with the sign turned. Every point lies inside its interval, and the
    freeze-frame step drawn is the ladder's."""
    import json
    record = json.loads((repo_root / "docs" / "measured.json").read_text())["figures"]
    if not record.get("first_contact_frame") or not record.get("positive_control"):
        pytest.skip("run_first_contact.py or run_positive_control.py has not been run")
    ri = _import_module(repo_root / "renders" / "render_information.py", "_smoke_information")
    ps, v = ri.panels(record), record["validation"]["corner"]
    pc, t1 = record["positive_control"]["decisions"], record["first_contact_frame"]["decisions"]
    assert ps[0]["rows"][0][1:4] == (v["ladder"]["delivery"]["step"], *v["ladder"]["delivery"]["step_ci"])
    assert ps[0]["rows"][1][1:4] == (v["refit"]["step"], *v["refit"]["normal_ci"])
    for row, d, form in ((ps[1]["rows"][0], pc["delivery-prior"], "ci"),
                         (ps[1]["rows"][1], pc["full_360-delivery"], "ci"),
                         (ps[2]["rows"][0], t1["geometry_1-delivery"], "pivotal"),
                         (ps[2]["rows"][1], t1["static_race_1-delivery"], "pivotal")):
        assert row[1:4] == (-d["delta"], -d[form][1], -d[form][0]), row[0]
    for p in ps:
        for label, saved, lo, hi, _ in p["rows"]:
            assert lo <= saved <= hi, label
    assert v["refit"]["step"] == pytest.approx(v["ladder"]["full"]["step"], abs=1e-4)
    out = tmp_path / "information.png"
    _run(repo_root, "renders/render_information.py", "--out", str(out))
    assert out.stat().st_size > 0


@requires_record
def test_render_segments_prints_the_record_s_counts_and_writes_png(repo_root, tmp_path):
    """The diagram's definitions carry the frontier's segment counts as stored, a segment's goals counting
    the own goals the record keeps apart, so the five segments' counts reproduce the two headline shares
    the bracket prints, and the TikZ source compiles."""
    import json
    import shutil
    rs = _import_module(repo_root / "renders" / "render_segments.py", "_smoke_segments")
    fr = json.loads((repo_root / "docs" / "measured.json").read_text())["figures"]["frontier"]["corner"]
    seg, head, d = fr["segment"], fr["headline"], rs.defs(fr)
    counts = {"shots": {k: v["shots"] for k, v in seg.items()}, "goals": {k: v["goals"] + v["own_goals"] for k, v in seg.items()}}
    for key, name, _ in rs.SEGMENT_CARDS:
        k = key.replace("_", "")
        for field, value in (("name", name), ("shots", f"{counts['shots'][key]:,}"),
                             ("goals", f"{counts['goals'][key]:,}")):
            assert f"\\def\\{k}{field}{{{value}}}" in d, (key, field)
    aft = ("flick_on", "prolongation", "after_clearance")
    for what, count in counts.items():
        share = sum(count[k] for k in aft) / sum(count[k] for k, _, _ in rs.SEGMENT_CARDS)
        assert round(share, 4) == head[f"share_{what}_after_first_touch"]["point"], what
    if shutil.which("xelatex") is None or shutil.which("pdftoppm") is None:
        pytest.skip("xelatex or pdftoppm is not installed")
    out = tmp_path / "segments.png"
    _run(repo_root, "renders/render_segments.py", "--out", str(out), "--build", str(tmp_path / "build"))
    assert out.stat().st_size > 0 and out.with_suffix(".svg").stat().st_size > 0


@requires_record
def test_render_pipeline_prints_the_record_s_match_count_and_writes_png(repo_root, tmp_path):
    """The overview's one number is the record's count of matches, as stored, and its TikZ source compiles."""
    import json
    import shutil
    rp = _import_module(repo_root / "renders" / "render_pipeline.py", "_smoke_pipeline_diagram")
    scope = json.loads((repo_root / "docs" / "measured.json").read_text())["figures"]["scope"]
    assert f"\\def\\matches{{{scope['matches']:,}}}" in rp.defs(scope)
    if shutil.which("xelatex") is None or shutil.which("pdftoppm") is None:
        pytest.skip("xelatex or pdftoppm is not installed")
    out = tmp_path / "pipeline.png"
    _run(repo_root, "renders/render_pipeline.py", "--out", str(out), "--build", str(tmp_path / "build"))
    assert out.stat().st_size > 0 and out.with_suffix(".svg").stat().st_size > 0


@requires_sb
def test_render_example_draws_the_corner_the_paper_tells_and_writes_both_figures(repo_root, tmp_path):
    """The example is the author's pick, Timber's second goal from a corner against Leeds United, with a freeze-frame at
    the kick, and the paper's caption names its scorer, opponent and date. The figure asserts the chain it tells and
    fails if the corner changes; both figures are written, and each has checked that no label covers another or a disc."""
    if not (repo_root / "data" / "derived" / "corners.parquet").exists():
        pytest.skip("the corner table is not built")
    ex = _import_module(repo_root / "renders" / "render_example.py", "_smoke_example")
    match_id, _ = ex.pick()
    m = ex.L.list_matches(None).set_index("match_id").loc[match_id]
    assert str(m["match_date"])[:10] == "2025-08-23" and {m["home_team"], m["away_team"]} == {"Arsenal", "Leeds United"}
    paper = repo_root / "docs" / "paper.md"
    if paper.exists():
        caption = next(line for line in paper.read_text().splitlines()
                       if line.startswith("![Figure") and "figures/corner_cut.png" in line)
        assert "Timber" in caption and "Leeds United" in caption and "23 August 2025" in caption
    _run(repo_root, "renders/render_example.py", "--out-dir", str(tmp_path))
    for name in ("corner_cut.png", "corner_plan.png", "corner_cut.svg", "corner_plan.svg"):
        assert (tmp_path / name).stat().st_size > 0, name
