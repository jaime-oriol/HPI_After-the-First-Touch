"""run_e2e — run every driver end to end and say plainly whether anything broke.

A green pipeline is a claim, and the claim is only worth something if a failure is
impossible to miss. This driver runs the stages in dependency order, streams each
one's output live rather than buffering it, times it, and closes with a table that
names every stage and its verdict. It exits non-zero if any stage fails, so it can
gate a commit.

Order follows dependencies, not a grouping by kind: the tables come first because
everything else reads them, each stage runs after the stages whose artefacts it reads,
and `report_figures` runs last because it summarises whatever the run just produced.
Every verdict the plan registered (data/derived/prereg_2026-10-01) was read on the plan's
tables, as s01 builds them with its two attribution rules (deviation_2026-10-12.md), so right
after them the run stops unless its own are the same, and GPU1 loads the tasks fitted on those
tables instead of training them again.

TWO POPULATIONS. The paper models corners and the flank free kicks from x >= 85, and every
corner stage has its twin on the second population, `<stage>_flank`, except the short-corner
sensitivity, since a free kick is never short, and plan_race, registered on the tracked corners alone.

ONE PARTITION. Every stage reads the zones under the switch the run is started with (src/zones.py),
the tables included, so a run under another partition is the whole recipe run again with HUDL_ZONES
set, and the gate holds its tables to the plan's in everything the partition does not define.

THE FULL RUN IS THE RECIPE. Every stage passes the arguments that reproduce the
published artefact, so a full run regenerates what is in `data/derived/` and not a
near miss of it. The comment above a stage gives the reason for an argument where one
is needed, and every artefact `report_figures` seals has a stage here that writes it.

A --smoke run PUBLISHES NOTHING and reads only what it built. Stages that walk matches
take the grant's first _SMOKE_N and write tagged names of their own; the rest are
pointed at those tables or at `data/derived/smoke/`; the tracking stages read the
tracked sample's tables, which stages run only in a smoke build there; s05 reads the
plan's grant tables; and `report_figures` verifies instead of rewriting. So the smoke
run exercises the chain end to end and leaves every published artefact byte for byte
where it was.

    python pipeline/run_e2e.py                  # the full battery
    python pipeline/run_e2e.py --smoke          # about an hour, and writes nothing published
    python pipeline/run_e2e.py --from run_value # resume at a stage
    python pipeline/run_e2e.py --list           # show the stages and stop
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]
_DER = _REPO / "data" / "derived"
# The paper's two populations: corners and the flank free kicks from x >= 85.
_FLANK = "free_kick_flank"
_FK_TABLE = str(_DER / "corners_corner_free_kick_flank.parquet")
_FK_SHOTS = str(_DER / "shots_corner_free_kick_flank.parquet")
# GPU1's sixty tasks, fitted on the reference tables and keyed by what each read (run_frame_gpu).
_GPU_TASKS = str(_DER / "auditoria_2026-10-12" / "gpu" / "tasks")

_SMOKE_DIR = _DER / "smoke"
# The smoke's subset, the grant's first matches: at 40 a flank twin had too few free kicks to fit
# (run_aim_zone's inner fold kept two rows, data/derived/e2e_paso8/smoke_1.log).
_SMOKE_N = "200"
_SMOKE_TAIL = f"_limit{_SMOKE_N}"
_SMOKE_TABLE = str(_DER / f"corners{_SMOKE_TAIL}.parquet")
_SMOKE_FK_TABLE = str(_DER / f"corners_corner_free_kick_flank{_SMOKE_TAIL}.parquet")
_SMOKE_FK_SHOTS = str(_DER / f"shots_corner_free_kick_flank{_SMOKE_TAIL}.parquet")
_SMOKE_GRAIN = str(_DER / f"eda_deliveries{_SMOKE_TAIL}.parquet")
_SMOKE_MATCHES = str(_DER / f"eda_matches{_SMOKE_TAIL}.parquet")
_SMOKE_INPUTS = ["--deliveries", _SMOKE_GRAIN, "--shots", _SMOKE_FK_SHOTS]
_SMOKE_TOP = [*_SMOKE_INPUTS, "--matches", _SMOKE_MATCHES,
              "--standings", str(_SMOKE_DIR / "eda_standings.parquet")]
# The subset holds no tracked match, so the tracking stages' smoke reads the tracked sample's tables,
# which the smoke-only stages build here.
_SMOKE_TRACKED = _SMOKE_DIR / "tracked"
_SMOKE_OVERLAP = ["--overlap-tables", str(_SMOKE_TRACKED)]
# The plan's grant tables, which compare_tables holds the recipe's to. s05's smoke reads them: a
# subset's cross-fit fold can hold one treatment class, and its C selection raises there.
_PLAN_TABLES = _DER / "auditoria_2026-10-12" / "tables"


def _smoke_out(name: str) -> str:
    return str(_SMOKE_DIR / name)


def _table(kind: str, smoke: bool = False) -> str:
    return str(_DER / f"corners_{kind}{_SMOKE_TAIL if smoke else ''}.parquet")


def _situation(kind: str) -> str:
    """The smoke's situation table of one population."""
    return str(_DER / f"situation{'' if kind == 'corner' else f'_{kind}'}{_SMOKE_TAIL}.parquet")


@dataclass
class Stage:
    """One driver invocation and the smoke variant of its arguments. A smoke-only stage runs its `args`
    in a smoke run and nowhere else."""
    name: str
    args: list
    why: str
    smoke: list | None = None
    smoke_only: bool = False


STAGES = [
    Stage("build_table", ["build_table.py", "--jobs", "20"],
          "S01+S02 over the grant: the corner table everything downstream reads",
          ["build_table.py", "--jobs", "8", "--limit", _SMOKE_N]),
    Stage("build_table_flank", ["build_table.py", "--jobs", "20", "--kinds", _FLANK],
          "the same over the flank free kicks, which s01 builds on their own grammar",
          ["build_table.py", "--jobs", "8", "--limit", _SMOKE_N, "--kinds", _FLANK]),
    Stage("build_table_fk", ["build_table.py", "--jobs", "20", "--kinds", f"corner,{_FLANK}"],
          "the same over both populations, for the pooled contrasts",
          ["build_table.py", "--jobs", "8", "--limit", _SMOKE_N, "--kinds", f"corner,{_FLANK}"]),
    Stage("eda_build", ["eda_build.py", "--jobs", "20"],
          "the delivery grain of both populations and the match results",
          ["eda_build.py", "--jobs", "8", "--limit", _SMOKE_N]),
    # The gate on everything the plan registered: no verdict of the plan reproduces on other tables,
    # and GPU1's tasks would not load on them either.
    Stage("compare_tables", ["compare_tables.py"], "are the six tables the plan's",
          ["compare_tables.py", "--limit", _SMOKE_N]),
    Stage("compare_grains", ["compare_tables.py", "--grains"],
          "are the delivery grain and the match results the plan's",
          ["compare_tables.py", "--grains", "--limit", _SMOKE_N]),
    # Loads the sixty tasks and fits none, so it stops on any table the tasks were not keyed by. The
    # smoke takes the same path on the plan's own tables, into smoke/.
    Stage("run_frame_gpu", ["run_frame_gpu.py", "--tables", str(_DER), "--out-dir", str(_DER),
                            "--load-only", "--tasks", _GPU_TASKS, "--jobs", "8", "--build-jobs", "6"],
          "GPU1: the frame network against the delivery rung, loaded",
          ["run_frame_gpu.py", "--out-dir", _smoke_out("gpu"), "--load-only", "--tasks", _GPU_TASKS,
           "--jobs", "8", "--build-jobs", "6"]),
    Stage("falsify_frontier", ["falsify_frontier.py", "--jobs", "20", "--boot", "2000"],
          "the blocking gate: is the frontier football or an annotation artefact",
          ["falsify_frontier.py", "--jobs", "8", "--limit", _SMOKE_N, "--boot", "200"]),
    Stage("falsify_frontier_flank", ["falsify_frontier.py", "--jobs", "20", "--boot", "2000",
                                     "--kinds", _FLANK],
          "the same gate on the flank free kicks",
          ["falsify_frontier.py", "--jobs", "8", "--limit", _SMOKE_N, "--boot", "200",
           "--kinds", _FLANK]),
    Stage("audit_360", ["audit_360.py", "--jobs", "20", "--per-league"],
          "coverage gate: 360 coverage and freeze-frame occupancy",
          ["audit_360.py", "--jobs", "8", "--limit", _SMOKE_N, "--per-league"]),
    Stage("audit_360_flank", ["audit_360.py", "--jobs", "20", "--per-league", "--kinds", _FLANK],
          "the same on the flank free kicks",
          ["audit_360.py", "--jobs", "8", "--limit", _SMOKE_N, "--per-league", "--kinds", _FLANK]),
    # One tracking pass for both populations: the tracking load is the cost, not the population.
    Stage("audit_tracking", ["audit_tracking.py", "--kinds", f"corner,{_FLANK}"],
          "tracked-sample audit: cross-link, coordinates, transition-window coverage",
          ["audit_tracking.py", "--limit", "3", "--kinds", f"corner,{_FLANK}"]),
    Stage("probe_throw_ins", ["probe_throw_ins.py", "--jobs", "20", "--boot", "2000"],
          "are long throws a third population: the measurement behind the exclusion",
          # 150 and not more: past 5 x MIN_CELL shots describe_frontier.diagnose raises on a
          # missing segment, and `flick_on` is rare on long throws (measured.md §throw_ins).
          ["probe_throw_ins.py", "--jobs", "8", "--limit", "150", "--boot", "100",
           "--shots", _SMOKE_FK_SHOTS]),
    # The same first 150 matches as its probe, whose record it reads.
    Stage("eda_throw_ins", ["eda_throw_ins.py", "--jobs", "20"],
          "the long-throw delivery grain the scope block counts, on the probe's adopted cell",
          ["eda_throw_ins.py", "--jobs", "8", "--limit", "150"]),
    # After eda_build, whose delivery grain and first contacts it is checked against.
    Stage("eda_penalties", ["eda_penalties.py", "--jobs", "20"],
          "the penalties a delivery wins inside its episode, which no segment values",
          ["eda_penalties.py", "--jobs", "8", "--limit", _SMOKE_N]),
    Stage("run_flank_cut", ["run_flank_cut.py", "--jobs", "20"],
          "where a flank free kick stops being a corner: the annex behind the x >= 85 cut",
          ["run_flank_cut.py", "--jobs", "8", "--limit", _SMOKE_N, "--boot", "200",
           "--out-dir", str(_SMOKE_DIR)]),
    # on the BOTH-populations table, because --by-type cannot split a corner-only one:
    # evidence.md §5 quotes a free-kick ladder, and it is bound to this stage's record.
    # 2000 is the driver's default and the only draw count validation.parquet reproduces at.
    Stage("run_validation", ["run_validation.py", "--table", _FK_TABLE,
                             "--by-type", "--boot", "2000", "--jobs", "20"],
          "controls, baseline ladder, league-out, ablations, both populations",
          ["run_validation.py", "--table", _SMOKE_FK_TABLE, "--quick", "--boot", "100",
           "--out-dir", str(_SMOKE_DIR)]),
    # The floor its permuted control reads each block against, on the corner rows of the same table and tuned as it
    # tunes them (deviation_noise_floor_2026-11-04.md). Corners only: the paper cites no floor for the flank free kicks.
    Stage("run_noise_floor", ["run_noise_floor.py", "--table", _FK_TABLE, "--jobs", "20"],
          "the floor of the permuted control, measured on blocks of pure noise",
          ["run_noise_floor.py", "--table", _SMOKE_FK_TABLE, "--draws", "2", "--repeats", "1", "--jobs", "8",
           "--out-dir", str(_SMOKE_DIR)]),
    Stage("run_value", ["run_value.py", "--boot", "2000", "--jobs", "20"],
          "the tuned competing-outcomes head and the hurdle terms",
          ["run_value.py", "--table", _SMOKE_TABLE, "--boot", "50", "--jobs", "8"]),
    Stage("run_value_flank", ["run_value.py", "--table", _table(_FLANK), "--boot", "2000",
                              "--jobs", "20"],
          "the same on the flank free kicks",
          ["run_value.py", "--table", _table(_FLANK, smoke=True), "--boot", "50", "--jobs", "8"]),
    Stage("sweep_counter", ["sweep_counter.py", "--jobs", "20"],
          "the counter-attack window sweep that shows the arm is not estimable",
          ["sweep_counter.py", "--jobs", "8", "--limit", _SMOKE_N]),
    Stage("sweep_counter_flank", ["sweep_counter.py", "--jobs", "20", "--kinds", _FLANK],
          "the same on the flank free kicks",
          ["sweep_counter.py", "--jobs", "8", "--limit", _SMOKE_N, "--kinds", _FLANK]),
    # No --jobs: the driver runs one worker per specification, the unit it parallelises over. The
    # smoke reads the plan's grant table (_PLAN_TABLES); its --out keeps it unpublished.
    Stage("run_causal", ["run_causal.py"],
          "the frontal protector: three estimands, the registered primary and the curve",
          ["run_causal.py", "--table", str(_PLAN_TABLES / "corners.parquet"), "--jobs", "8",
           "--out", _smoke_out("causal.parquet")]),
    Stage("run_causal_flank", ["run_causal.py", "--table", _table(_FLANK)],
          "the same on the flank free kicks",
          ["run_causal.py", "--table", str(_PLAN_TABLES / f"corners_{_FLANK}.parquet"), "--jobs", "8",
           "--out", _smoke_out(f"causal_{_FLANK}.parquet")]),
    Stage("run_situation", ["run_situation.py", "--build", "--jobs", "20", "--boot", "2000"],
          "S08: the four states, priced, with match-clustered contrasts",
          ["run_situation.py", "--build", "--jobs", "8", "--limit", _SMOKE_N, "--boot", "200"]),
    Stage("run_situation_flank", ["run_situation.py", "--build", "--kinds", _FLANK, "--jobs", "20",
                                  "--boot", "2000"],
          "the same on the flank free kicks",
          ["run_situation.py", "--build", "--kinds", _FLANK, "--jobs", "8", "--limit", _SMOKE_N,
           "--boot", "200"]),
    # No --tune: that arm re-selects C inside each population, while the question here is what
    # dropping rows does, which needs the regularisation held.
    # --jobs is the events fan-out that finds the cross after each short corner.
    Stage("run_short_corner_sensitivity", ["run_short_corner_sensitivity.py", "--jobs", "20",
                                           "--boot", "2000"],
          "does the headline contrast rest on short corners, and does it hold on their crosses",
          ["run_short_corner_sensitivity.py", "--table", _SMOKE_TABLE, "--limit", _SMOKE_N,
           "--jobs", "8", "--boot", "200",
           "--out", _smoke_out("short_corner_sensitivity.parquet")]),
    # After run_situation, whose four-state table it reads.
    Stage("run_drop_position", ["run_drop_position.py", "--boot", "2000",
                                "--cell", "4", "6", "8", "12"],
          "does the staked-win premium survive holding the drop position equal",
          ["run_drop_position.py", "--table", _situation("corner"), "--boot", "200", "--cell", "6",
           "--out", _smoke_out("drop_position.parquet")]),
    Stage("run_drop_position_flank", ["run_drop_position.py", "--kinds", _FLANK, "--boot", "2000",
                                      "--cell", "4", "6", "8", "12"],
          "the same on the flank free kicks",
          ["run_drop_position.py", "--kinds", _FLANK, "--table", _situation(_FLANK), "--boot", "200",
           "--cell", "6", "--out", _smoke_out(f"drop_position_{_FLANK}.parquet")]),
    # After eda_build, which carries its target, `won_first_contact`, on the delivery grain. It is a
    # control and not a result.
    Stage("run_positive_control", ["run_positive_control.py", "--boot", "2000"],
          "can the harness see signal at all: the delivery frame against who wins the first contact",
          ["run_positive_control.py", "--limit", _SMOKE_N, "--boot", "200",
           "--out", _smoke_out("positive_control.parquet")]),
    Stage("run_positive_control_flank", ["run_positive_control.py", "--kind", _FLANK,
                                         "--boot", "2000"],
          "the same on the flank free kicks",
          ["run_positive_control.py", "--kind", _FLANK, "--limit", _SMOKE_N, "--boot", "200",
           "--out", _smoke_out(f"positive_control_{_FLANK}.parquet")]),
    # After eda_build for the same reason: who reached the ball first is read off the delivery grain.
    # The verdict's 300 draws are the first of stability seed 0, so the five seeds cost the run.
    Stage("run_aim_zone", ["run_aim_zone.py", "--jobs", "20"],
          "the aim read at the kick, and the end-zone map it corrects",
          ["run_aim_zone.py", "--limit", _SMOKE_N, "--boot", "200", "--refit-boot", "3",
           "--stability-seeds", "1", "--stability-boot", "3", "--jobs", "8",
           "--out-dir", str(_SMOKE_DIR)]),
    Stage("run_aim_zone_flank", ["run_aim_zone.py", "--kind", _FLANK, "--jobs", "20"],
          "the same on the flank free kicks",
          ["run_aim_zone.py", "--kind", _FLANK, "--limit", _SMOKE_N, "--boot", "200",
           "--refit-boot", "3", "--stability-seeds", "1", "--stability-boot", "3", "--jobs", "8",
           "--out-dir", str(_SMOKE_DIR)]),
    # M1 names its six partitions itself, so it runs once, both populations inside.
    Stage("run_zone_comparison", ["run_zone_comparison.py", "--jobs", "23"],
          "M1: which partition of the box at the kick carries the plan",
          ["run_zone_comparison.py", "--limit", _SMOKE_N, "--boot", "50", "--refit-boot", "3",
           "--seeds", "1", "--draws", "4", "--jobs", "8", "--out-dir", str(_SMOKE_DIR)]),
    Stage("run_zone_robustness", ["run_zone_comparison.py", "--jobs", "23", "--robustness"],
          "M1's two readings added after the registered run",
          ["run_zone_comparison.py", "--limit", _SMOKE_N, "--boot", "50", "--refit-boot", "3",
           "--seeds", "1", "--draws", "4", "--jobs", "8", "--robustness",
           "--out-dir", str(_SMOKE_DIR)]),
    # After run_situation, whose drop and next touch price the second duel.
    Stage("run_duel_prices", ["run_duel_prices.py", "--jobs", "20"],
          "C1: what each duel of a corner is worth, zone by zone",
          ["run_duel_prices.py", "--limit", _SMOKE_N, "--boot", "50", "--seeds", "1", "--jobs", "8",
           "--out-dir", str(_SMOKE_DIR)]),
    Stage("run_duel_prices_flank", ["run_duel_prices.py", "--kind", _FLANK, "--jobs", "20"],
          "the same on the flank free kicks",
          ["run_duel_prices.py", "--kind", _FLANK, "--limit", _SMOKE_N, "--boot", "50", "--seeds", "1",
           "--jobs", "8", "--out-dir", str(_SMOKE_DIR)]),
    Stage("eda_standings", ["eda_standings.py"], "league table, reconciled against the published one",
          ["eda_standings.py", "--matches", _SMOKE_MATCHES,
           "--out", _smoke_out("eda_standings.parquet")]),
    # Before eda_second_ball, whose team table shrinks its ranking by the reliability written here;
    # it reads only the grain and the shots.
    Stage("eda_repeatability", ["eda_repeatability.py", "--splits", "500", "--kind", "corner"],
          "argument.md Q3: dispersion and split-half reliability",
          ["eda_repeatability.py", "--splits", "50", "--boot", "50", "--kind", "corner",
           *_SMOKE_INPUTS, "--out-dir", str(_SMOKE_DIR)]),
    Stage("eda_repeatability_flank", ["eda_repeatability.py", "--splits", "500", "--kind", _FLANK],
          "the same on the flank free kicks",
          ["eda_repeatability.py", "--splits", "50", "--boot", "50", "--kind", _FLANK,
           *_SMOKE_INPUTS, "--out-dir", str(_SMOKE_DIR)]),
    # The smoke reads the standings and the reliability its own earlier stages wrote.
    Stage("eda_second_ball", ["eda_second_ball.py", "--kind", "corner"],
          "team/league/position descriptive layer with the reliability guard",
          ["eda_second_ball.py", "--kind", "corner", "--boot", "100", *_SMOKE_INPUTS,
           "--records", str(_SMOKE_DIR), "--out-dir", str(_SMOKE_DIR)]),
    Stage("eda_second_ball_flank", ["eda_second_ball.py", "--kind", _FLANK],
          "the same on the flank free kicks",
          ["eda_second_ball.py", "--kind", _FLANK, "--boot", "100", *_SMOKE_INPUTS,
           "--records", str(_SMOKE_DIR), "--out-dir", str(_SMOKE_DIR)]),
    # After eda_second_ball, which writes the per-team table it fits. Twenty-three NUTS fits over
    # the per-team table, and the only stage in the battery that samples rather than optimises.
    Stage("run_team_shrinkage", ["run_team_shrinkage.py"],
          "is any team's after-first-touch rate distinguishable from its league, noise discounted",
          ["run_team_shrinkage.py", "--smoke", "--teams", _smoke_out("eda_teams_corner.parquet"),
           "--out", _smoke_out("team_shrinkage.parquet")]),
    Stage("run_team_shrinkage_flank", ["run_team_shrinkage.py", "--kind", _FLANK],
          "the same on the flank free kicks",
          ["run_team_shrinkage.py", "--kind", _FLANK, "--smoke",
           "--teams", _smoke_out(f"eda_teams_{_FLANK}.parquet"),
           "--out", _smoke_out(f"team_shrinkage_{_FLANK}.parquet")]),
    Stage("eda_club_power", ["eda_club_power.py", "--jobs", "23"],
          "C12: what a club can measure about its own corners, and how long it takes",
          ["eda_club_power.py", *_SMOKE_INPUTS, "--sims", "400", "--boot", "50", "--splits", "20",
           "--seeds", "2", "--jobs", "8", "--out-dir", str(_SMOKE_DIR)]),
    Stage("eda_club_power_flank", ["eda_club_power.py", "--kind", _FLANK, "--jobs", "23"],
          "the same on the flank free kicks",
          ["eda_club_power.py", "--kind", _FLANK, *_SMOKE_INPUTS, "--sims", "400", "--boot", "50",
           "--splits", "20", "--seeds", "2", "--jobs", "8", "--out-dir", str(_SMOKE_DIR)]),
    # After eda_standings, whose final positions form group T.
    Stage("eda_top_teams", ["eda_top_teams.py", "--jobs", "23"],
          "do the best sides, and the best corner sides, live after the first touch",
          ["eda_top_teams.py", *_SMOKE_TOP, "--boot", "100", "--perms", "200", "--splits", "5",
           "--split-perms", "50", "--seeds", "1", "--jobs", "8", "--out-dir", str(_SMOKE_DIR)]),
    Stage("eda_top_teams_flank", ["eda_top_teams.py", "--kind", _FLANK, "--jobs", "23"],
          "the same on the flank free kicks",
          ["eda_top_teams.py", "--kind", _FLANK, *_SMOKE_TOP, "--boot", "100", "--perms", "200",
           "--splits", "5", "--split-perms", "50", "--seeds", "1", "--jobs", "8",
           "--out-dir", str(_SMOKE_DIR)]),
    # Smoke only: the subset holds no tracked match, so the tracking stages' smoke reads the tracked
    # sample's tables, built here; run_race's smoke writes the race beside them.
    Stage("build_table_tracked", ["build_table.py", "--source", "serie_a_sample", "--jobs", "8",
                                  "--out", str(_SMOKE_TRACKED / "corners.parquet")],
          "the tracked sample's corner table, for the tracking stages' smoke", smoke_only=True),
    Stage("build_table_tracked_flank", ["build_table.py", "--source", "serie_a_sample", "--jobs", "8",
                                        "--kinds", _FLANK,
                                        "--out", str(_SMOKE_TRACKED / f"corners_{_FLANK}.parquet")],
          "the same on the flank free kicks", smoke_only=True),
    Stage("run_situation_tracked", ["run_situation.py", "--build", "--source", "serie_a_sample",
                                    "--jobs", "8", "--boot", "200",
                                    "--out", str(_SMOKE_TRACKED / "situation.parquet")],
          "the tracked sample's four states, for the tracking stages' smoke", smoke_only=True),
    Stage("run_situation_tracked_flank", ["run_situation.py", "--build", "--source", "serie_a_sample",
                                          "--kinds", _FLANK, "--jobs", "8", "--boot", "200",
                                          "--out", str(_SMOKE_TRACKED / f"situation_{_FLANK}.parquet")],
          "the same on the flank free kicks", smoke_only=True),
    Stage("run_race", ["run_race.py", "--all", "--jobs", "2"],
          "S04 over every tracked match, with tau calibrated across them",
          ["run_race.py", "--all", "--jobs", "2", "--out", str(_SMOKE_TRACKED / "race_corners.parquet")]),
    Stage("run_race_flank", ["run_race.py", "--all", "--kinds", _FLANK, "--jobs", "2"],
          "the same on the flank free kicks",
          ["run_race.py", "--all", "--kinds", _FLANK, "--jobs", "2",
           "--out", str(_SMOKE_TRACKED / f"race_corners_{_FLANK}.parquet")]),
    Stage("run_q2", ["run_q2.py"],
          "argument.md Q2: the plan and the race joined on the corners that carry both",
          ["run_q2.py", "--tables", str(_SMOKE_TRACKED), "--boot", "500",
           "--out", _smoke_out("q2_overlap.parquet")]),
    Stage("run_q2_flank", ["run_q2.py", "--kinds", _FLANK],
          "the same on the flank free kicks",
          ["run_q2.py", "--kinds", _FLANK, "--tables", str(_SMOKE_TRACKED), "--boot", "500",
           "--out", _smoke_out(f"q2_overlap_{_FLANK}.parquet")]),
    # reads the events and streams the tracking itself, on one core; its smoke takes the first tracked matches
    Stage("run_plan_race", ["run_plan_race.py"],
          "Section 5.1: how the second balls of the tracked corners were won, on the clocks aligned per half",
          ["run_plan_race.py", "--limit", "3", "--boot", "200", "--out-dir", str(_SMOKE_DIR)]),
    # after run_race and run_situation, whose parquets bound its population. Its tracking pass
    # is `--track-jobs`, default 2; `--jobs` is the events, the tuning grid, the seeds and the
    # refit draws. At 20 it writes the same frame as at 2, refit block included: exact equality on
    # 40 matches with 20 refit draws (review_2026-09-23/compare_suff_invariance.log), after the
    # sha256 identity on the whole grant without that block. The smoke
    # variant drops the power curve, a nested resample and the only slow part after tracking, cuts
    # the refit interval to three draws, which checks that the point reproduces, not the bound, and
    # the strong learner's search to the floor. Its grant half is the subset, where a heavily 'won'
    # target still carries both classes; its overlap is the tracked sample's, so the tracked rungs
    # run in a smoke too.
    Stage("run_sufficiency", ["run_sufficiency.py", "--jobs", "20"],
          "form_answers.md Q3: freeze-frame against tracked race on the same event-derived outcome",
          ["run_sufficiency.py", "--jobs", "20", "--limit", _SMOKE_N, *_SMOKE_OVERLAP, "--boot", "500",
           "--trials", "2", "--seeds", "1", "--no-power", "--refit-boot", "3",
           "--out", _smoke_out("sufficiency.parquet")]),
    Stage("run_sufficiency_flank", ["run_sufficiency.py", "--kinds", _FLANK, "--jobs", "20"],
          "the same on the flank free kicks",
          ["run_sufficiency.py", "--kinds", _FLANK, "--jobs", "20", "--limit", _SMOKE_N,
           *_SMOKE_OVERLAP, "--boot", "500", "--trials", "2", "--seeds", "1", "--no-power",
           "--refit-boot", "3", "--out", _smoke_out(f"sufficiency_{_FLANK}.parquet")]),
    # After run_sufficiency, whose grant table it reads through `grant_table`. Nearly all of its
    # cost is the five stability seeds: each one refits every arm inside every resample, which is
    # what makes the interval honest. The smoke variant cuts the draws,
    # the seeds and the Optuna budget to the floor, because what it checks is that the path runs
    # end to end; three draws cannot bound anything.
    Stage("run_first_contact", ["run_first_contact.py", "--jobs", "20"],
          "the conditional estimand at T1: what a camera at the first contact would add",
          ["run_first_contact.py", "--limit", _SMOKE_N, "--boot", "200", "--refit-boot", "3",
           "--stability-seeds", "1", "--stability-boot", "3", "--trials", "2", "--seeds", "1",
           "--jobs", "8", "--out-dir", str(_SMOKE_DIR)]),
    Stage("run_first_contact_flank", ["run_first_contact.py", "--kind", _FLANK, "--jobs", "20"],
          "the same on the flank free kicks",
          ["run_first_contact.py", "--kind", _FLANK, "--limit", _SMOKE_N, "--boot", "200",
           "--refit-boot", "3", "--stability-seeds", "1", "--stability-boot", "3", "--trials", "2",
           "--seeds", "1", "--jobs", "8", "--out-dir", str(_SMOKE_DIR)]),
    # Phase two of V1, where phase one leaves it undecided: the flank. The pre-registration closes an
    # undecided V1 only with the strong learner refitted inside every draw; on corners V1 holds in
    # phase one and this never runs. Ten thousand draws of a tuned LightGBM make it the longest
    # stage of the recipe by far, and it writes `_gbt` records beside phase one's that
    # report_figures reads in their place.
    Stage("run_first_contact_gbt_flank",
          ["run_first_contact.py", "--kind", _FLANK, "--jobs", "24", "--gbt-refit"],
          "phase two of V1 on the flank free kicks: the strong learner refitted inside every draw",
          ["run_first_contact.py", "--kind", _FLANK, "--limit", _SMOKE_N, "--boot", "200",
           "--refit-boot", "3", "--stability-seeds", "1", "--stability-boot", "3", "--trials", "2",
           "--seeds", "1", "--jobs", "8", "--gbt-refit", "--out-dir", str(_SMOKE_DIR)]),
    # After run_race, run_situation and run_sufficiency's grant, which bound its populations.
    Stage("run_tracking_curve", ["run_tracking_curve.py", "--jobs", "23"],
          "TRK1: what more tracked matches would buy, as the N needed",
          ["run_tracking_curve.py", "--tables", str(_SMOKE_TRACKED), "--draws", "20", "--boot", "200",
           "--seeds", "1", "--out-dir", str(_SMOKE_DIR)]),
    Stage("run_tracking_curve_flank", ["run_tracking_curve.py", "--kind", _FLANK, "--jobs", "23"],
          "the same on the flank free kicks, recorded as not estimable under the floor",
          ["run_tracking_curve.py", "--kind", _FLANK, "--tables", str(_SMOKE_TRACKED), "--draws", "20",
           "--boot", "200", "--seeds", "1", "--out-dir", str(_SMOKE_DIR)]),
    # on the BOTH-populations shot table, which is also the driver's own default; the flag stays
    # explicit because this file is the recipe. On the corner-only table the free-kick half of
    # measured.md §frontier would never be written, and evidence.md §2's free-kick obs/exp would
    # lose its generated twin. The driver tags its output instead of overwriting the canonical
    # artefact.
    Stage("describe_frontier", ["describe_frontier.py", "--shots", _FK_SHOTS],
          "the descriptive product at shot grain, both populations",
          ["describe_frontier.py", "--shots", _SMOKE_FK_SHOTS, "--deliveries", _SMOKE_GRAIN,
           "--situation", _situation("corner"), "--boot", "200",
           "--out", _smoke_out("frontier.parquet")]),
    # After every validation population and every sufficiency grant, because it re-reads them and
    # stops unless its rebuilt points and percentiles reproduce what those stages wrote. Its cost is
    # the 300 refit draws per contrast. The smoke variant cuts the matches and the draws, which
    # checks the path and not the bounds.
    Stage("run_refit_draws", ["run_refit_draws.py", "--jobs", "20"],
          "the refit draws behind the decisive contrasts, and the head-to-head's refit interval",
          ["run_refit_draws.py", "--limit", _SMOKE_N, "--boot", "5", "--jobs", "8",
           "--out", _smoke_out("refit_draws.parquet")]),
    Stage("report_figures", ["report_figures.py"], "regenerate measured.md/.json from what this run wrote",
          ["report_figures.py", "--check"]),
]


def stages(smoke: bool) -> list:
    """The stages a run executes, in order: the smoke-only ones only in a smoke run."""
    return [s for s in STAGES if smoke or not s.smoke_only]


def _hms(seconds: float) -> str:
    m, s = divmod(int(seconds), 60)
    return f"{m:d}m{s:02d}s" if m else f"{s:d}s"


def run_stage(stage: Stage, smoke: bool, index: int, total: int) -> tuple[bool, float]:
    """Run one stage with its output streamed live, and report OK/FAILED."""
    args = (stage.smoke or stage.args) if smoke else stage.args
    zones = os.environ.get("HUDL_ZONES")
    print(f"\n{'=' * 78}\n[{index}/{total}] {stage.name}   —   {stage.why}\n"
          f"  $ {f'HUDL_ZONES={zones} ' if zones else ''}python pipeline/{' '.join(args)}\n{'=' * 78}",
          flush=True)
    t0 = time.time()
    # streamed, not captured: a battery that only speaks at the end hides where it hung
    code = subprocess.run([sys.executable, str(_REPO / "pipeline" / args[0]), *args[1:]],
                          cwd=_REPO).returncode
    dt = time.time() - t0
    print(f"\n  --> {'OK' if code == 0 else f'FAILED (exit {code})'}   [{_hms(dt)}]", flush=True)
    return code == 0, dt


def main() -> None:
    """Run the battery in order and exit non-zero if any stage failed."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true",
                    help="the smoke variants: the grant's first matches, an hour and not a day")
    ap.add_argument("--from", dest="start", type=str, default=None, help="resume at this stage name")
    ap.add_argument("--only", type=str, default=None, help="run a single stage by name")
    ap.add_argument("--list", action="store_true", help="print the stage list and exit")
    ap.add_argument("--keep-going", action="store_true", help="run every stage even after a failure")
    args = ap.parse_args()

    if args.list:
        for i, s in enumerate(STAGES, 1):
            print(f"  {i:2d}. {s.name:32s} {'(smoke only) ' if s.smoke_only else ''}{s.why}")
        return
    todo = stages(args.smoke)
    if args.only:
        todo = [s for s in todo if s.name == args.only]
        if not todo:
            raise SystemExit(f"no stage named {args.only!r} in this run; --list shows them")
    elif args.start:
        names = [s.name for s in todo]
        if args.start not in names:
            raise SystemExit(f"no stage named {args.start!r} in this run; --list shows them")
        todo = todo[names.index(args.start):]

    if args.smoke:
        _SMOKE_TRACKED.mkdir(parents=True, exist_ok=True)
    mode = "SMOKE" if args.smoke else "FULL GRANT"
    print(f"E2E — {len(todo)} stages, {mode}. Streaming each stage live; the summary at the\n"
          f"end names every stage and its verdict, and the exit code is non-zero if any failed.")

    results, t0 = [], time.time()
    for i, s in enumerate(todo, 1):
        ok, dt = run_stage(s, args.smoke, i, len(todo))
        results.append((s.name, ok, dt))
        if not ok and not args.keep_going:
            print(f"\nSTOPPED at {s.name}. Fix it and resume with "
                  f"`--from {s.name}`, or pass --keep-going to run the rest anyway.")
            break

    print(f"\n{'=' * 78}\nSUMMARY   ({_hms(time.time() - t0)} total)\n{'=' * 78}")
    for name, ok, dt in results:
        print(f"  {'OK    ' if ok else 'FAILED'}  {name:32s} {_hms(dt):>8s}")
    failed = [n for n, ok, _ in results if not ok]
    skipped = len(todo) - len(results)
    if skipped:
        print(f"  ...{skipped} stage(s) not reached")
    print(f"\n{len(results) - len(failed)}/{len(todo)} stages passed"
          + (f"; FAILED: {', '.join(failed)}" if failed else "; nothing failed"))
    sys.exit(1 if failed or skipped else 0)


if __name__ == "__main__":
    main()
