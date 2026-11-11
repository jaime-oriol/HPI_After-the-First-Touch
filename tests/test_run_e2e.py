"""Tests for run_e2e, the recipe: both populations everywhere, one partition per run, and the order the
stages read each other's records in.
"""
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]
for p in (_REPO / "src", _REPO / "pipeline"):
    sys.path.insert(0, str(p))

import run_e2e as E                                                        # noqa: E402

NAMES = [s.name for s in E.STAGES]
# Stages that hold both populations inside one run, or that are not about a population at all.
_BOTH_INSIDE = {"build_table_fk", "eda_build", "compare_tables", "compare_grains", "run_frame_gpu",
                "audit_tracking", "probe_throw_ins", "eda_throw_ins", "eda_penalties",
                "run_flank_cut", "run_validation", "run_zone_comparison", "run_zone_robustness",
                "eda_standings", "describe_frontier", "run_refit_draws", "report_figures"}
_OUTPUTS = ("--out", "--out-dir", "--records")
# Phase two of V1 runs only where phase one leaves V1 undecided, which is the flank: it has no corner
# twin by design.
_FLANK_ONLY = {"run_first_contact_gbt_flank"}


def test_every_stage_names_a_driver_once():
    assert len(NAMES) == len(set(NAMES))
    for s in E.STAGES:
        assert (_REPO / "pipeline" / s.args[0]).exists(), s.name
        assert s.smoke is None or s.smoke[0] == s.args[0], s.name


def test_every_corner_stage_has_its_flank_twin():
    """The paper's two populations run through every stage; a free kick is never short, plan_race was registered on the
    tracked corners alone, and the paper cites no noise floor for the flank free kicks."""
    corner = {n for n in NAMES if not n.endswith("_flank")} - _BOTH_INSIDE
    twins = {n.removesuffix("_flank") for n in NAMES if n.endswith("_flank") and n not in _FLANK_ONLY}
    assert corner - twins == {"run_short_corner_sensitivity", "run_plan_race", "run_noise_floor"}
    assert twins <= corner


def test_one_run_is_one_partition_and_no_stage_names_another():
    """The partition is the switch the run starts with: no stage carries its own, and none writes into
    the folder a pass under another partition would use."""
    for s in E.STAGES:
        for a in [*s.args, *(s.smoke or [])]:
            assert "HUDL_ZONES" not in a and "zones_F" not in a, (s.name, a)


def test_every_record_report_figures_reads_has_a_stage_that_writes_it():
    """A section whose driver has no stage would publish whatever an old run left on disk."""
    import report_figures as RF
    drivers = {s.args[0] for s in E.STAGES}
    assert set(RF.SOURCES.values()) <= drivers, set(RF.SOURCES.values()) - drivers


def _before(a: str, b: str) -> bool:
    return NAMES.index(a) < NAMES.index(b)


def test_a_smoke_run_publishes_nothing_and_reads_only_what_it_built():
    """Every stage has a smoke variant, since without one --smoke would run the full arguments onto
    the published names, and no smoke argument names a published table: it is under smoke/, a
    `_limitN` table the smoke wrote, or the plan's folder (the GPU1 tasks, the gate's tables)."""
    plan = E._DER / "auditoria_2026-10-12"
    for s in E.STAGES:
        args = s.args if s.smoke_only else s.smoke
        assert args is not None, s.name
        for a in args:
            p = Path(a)
            if E._DER in p.parents:
                assert (E._SMOKE_DIR in (p, *p.parents) or "_limit" in p.name
                        or plan in p.parents), (s.name, a)


def test_the_smoke_only_stages_stay_out_of_the_full_run():
    full, smoke = E.stages(smoke=False), E.stages(smoke=True)
    assert [s.name for s in full] == [s.name for s in E.STAGES if not s.smoke_only]
    assert [s.name for s in smoke] == NAMES and any(s.smoke_only for s in smoke)


def test_the_tracked_sample_is_built_before_the_smoke_reads_it():
    """The smoke's grant subset holds no tracked match, so the tracking stages read tables built over
    the tracked sample, which every stage writing into that folder has to finish first."""
    def touches(s: E.Stage, flags: tuple) -> bool:
        args = s.args if s.smoke_only else s.smoke
        return any(a in flags and E._SMOKE_TRACKED in (Path(b), *Path(b).parents)
                   for a, b in zip(args[:-1], args[1:], strict=True))

    names = [s.name for s in E.stages(smoke=True)]
    writers = [names.index(s.name) for s in E.STAGES if touches(s, _OUTPUTS)]
    readers = [names.index(s.name) for s in E.STAGES if touches(s, ("--tables", "--overlap-tables"))]
    assert writers and readers and max(writers) < min(readers)


def test_each_stage_runs_after_the_records_it_reads():
    tables = ("build_table", "build_table_flank", "build_table_fk", "eda_build")
    assert NAMES[:len(tables)] == list(tables)
    assert NAMES[len(tables):len(tables) + 3] == ["compare_tables", "compare_grains", "run_frame_gpu"]
    for pop in ("", "_flank"):
        assert _before(f"eda_repeatability{pop}", f"eda_second_ball{pop}")
        assert _before(f"eda_second_ball{pop}", f"run_team_shrinkage{pop}")
        assert _before(f"run_situation{pop}", f"run_duel_prices{pop}")
        assert _before(f"run_race{pop}", f"run_sufficiency{pop}")
        assert _before(f"run_sufficiency{pop}", f"run_first_contact{pop}")
        assert _before(f"run_sufficiency{pop}", f"run_tracking_curve{pop}")
        assert _before(f"run_sufficiency{pop}", "run_refit_draws")
        assert _before(f"run_situation{pop}", f"run_drop_position{pop}")
        assert _before(f"run_race{pop}", f"run_tracking_curve{pop}")
        assert _before(f"run_situation{pop}", f"run_tracking_curve{pop}")
    assert _before("run_sufficiency_flank", "run_first_contact_gbt_flank")
    assert _before("run_first_contact_flank", "run_first_contact_gbt_flank")
    assert _before("eda_standings", "eda_top_teams")
    assert NAMES[-1] == "report_figures"
