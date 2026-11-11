"""Tests for report_figures — the generated measurement record.

These exist to kill a specific failure mode: a number typed into prose is right on
the day it is pasted and silently wrong after the next pipeline run. The generator
recomputes from the artefacts, and the cross-stage checks below fail loudly when
two stages stop agreeing about the same quantity, which is what staleness looks
like from the inside.
"""
import json
import sys
from pathlib import Path
from types import ModuleType

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "pipeline"))

from report_figures import build, to_markdown, _OUT_JSON

# No requires_* gate: these read the DERIVED artefacts, not the raw feeds, and each check skips
# itself when the artefact it needs is absent. A gate on a raw feed would skip the cross-stage
# checks green on a tree that carries the grant without that feed.


@pytest.fixture(scope="module")
def payload() -> dict:
    """One build shared by every check (it reads several parquets)."""
    return build()


def test_build_reports_provenance_for_every_source(payload):
    """Every artefact is named with the driver that writes it, so a missing figure
    names its own fix instead of just being absent."""
    assert payload["sources"], "no artefacts declared"
    for name, s in payload["sources"].items():
        assert s["driver"], f"{name} has no driver recorded"


def test_after_clearance_agrees_across_s01_and_s08(payload):
    """s08's value join must price exactly the terminal events s01 segmented.

    Both take the after-clearance segment over the same grant, so a divergence here
    means one stage changed and the other did not, the drift a hand-copied figure
    would hide. s08 counts every terminal event and the corpus splits them into shots and
    own goals, so the two meet on their sum."""
    corpus, sit = payload["figures"]["corpus"], payload["figures"]["situation"]
    if not corpus or not sit:
        pytest.skip("build_table.py or run_situation.py has not been run")
    seg = corpus["by_segment"]["after_clearance"]
    assert sit["after_clearance_shots"] == seg["shots"] + seg.get("own_goals", 0)
    assert sit["after_clearance_goals"] == seg["goals"] + seg.get("own_goals", 0)


def test_the_gate_and_the_corpus_count_the_same_shots_and_goals_per_segment(payload):
    """The gate's conversion table and the corpus split the same in-window shots, own goals out of
    both, so a segment's shots and goals agree to the unit."""
    figs = payload["figures"]
    corpus, gate = figs["corpus"], figs["falsification"]
    if not corpus or not gate:
        pytest.skip("build_table.py or falsify_frontier.py has not been run")
    for seg in ("direct", "flick_on", "prolongation", "after_clearance"):
        assert gate["conversion"][seg]["n"] == corpus["by_segment"][seg]["shots"], seg
        assert gate["conversion"][seg]["goals"] == corpus["by_segment"][seg]["goals"], seg


def test_the_throw_in_comparison_counts_the_shots_the_corpus_counts(payload):
    """The side-by-side of three populations divides the same shots the corpus and the frontier
    count: an own goal is a goal and no shot in either place."""
    figs = payload["figures"]
    corpus, throws, frontier = figs["corpus"], figs["throw_ins"], figs["frontier"]
    if not corpus or not throws or not frontier:
        pytest.skip("build_table.py, probe_throw_ins.py or describe_frontier.py has not been run")
    corner = throws["by_population"]["corner"]
    assert corner["shots"] == corpus["shots_on_the_frontier"]
    assert corner["direct_shots"] == corpus["by_segment"]["direct"]["shots"]
    assert corner["after_first_touch_shots_pct"] == corpus["after_first_touch_shots_pct"]
    for pop in ("corner", "free_kick_flank"):
        assert throws["by_population"][pop]["direct_ci"] == frontier[pop]["segment"]["direct"]["ci"]


def test_cleared_share_agrees_across_audit_and_situation(payload):
    """The coverage audit and the situation table must count the same cleared corners."""
    cov, sit = payload["figures"]["coverage_360"], payload["figures"]["situation"]
    if not cov or not sit:
        pytest.skip("audit_360.py or run_situation.py has not been run")
    audited = cov["corners"] * cov["cleared_pct"] / 100.0
    assert abs(audited - sit["cleared_corners"]) <= 1, "cleared counts diverged"


def test_four_state_cells_sum_to_the_known_subset(payload):
    """The four states partition the rows where both plan and outcome are known."""
    sit = payload["figures"]["situation"]
    if not sit:
        pytest.skip("run_situation.py has not been run")
    assert sum(c["n"] for c in sit["by_state"].values()) == sit["four_state_known"]


def test_shuffled_controls_never_beat_their_own_entropy(payload):
    """The anti-leakage assertion, as a test rather than a line of console output.

    A shuffled model that BEATS the marginal entropy of the class mix means the harness
    itself leaks, and nothing else in the repo would notice, because a leaking harness
    produces a perfectly plausible real number."""
    val = payload["figures"]["validation"]
    if not val:
        pytest.skip("run_validation.py has not been run")
    for pop, entry in val.items():
        for mode, c in entry["controls"].items():
            assert c["clean"], (f"{pop}/{mode} shuffled to {c['shuffled']} against entropy "
                                f"{c['entropy']}: the harness beats chance on noise")


def test_aim_zone_keeps_its_samples_apart_and_its_statistics_consistent(payload):
    """The development league never leaks into the sample that decides, every corner sits in one
    end zone, and D2 is the difference of the two maps it claims to be, off the same draws."""
    az = payload["figures"]["aim_zone"]
    if not az:
        pytest.skip("run_aim_zone.py has not been run")
    conf, dev = az["confirmation"], az["development"]
    assert dev["leagues"] == "['premier_league']" and "premier_league" not in conf["leagues"]
    for s in (conf, dev):
        assert sum(s["end_zones"].values()) == s["long_corners"]
    st = conf["statistics"]
    assert abs(st["D2"]["point"] - (st["naive"]["point"] - st["honest"]["point"])) <= 2e-5
    assert set(az["verdicts"]) == {"A1", "A2"}


def test_league_out_and_the_ladder_agree_on_the_full_model(payload):
    """Two blocks report the same quantity and must not drift apart.

    The ladder's `full` rung and league-out's grouped-by-match reference are both the
    tuned model's match-grouped OOF log-loss, computed in different code paths of the
    same run."""
    val = payload["figures"]["validation"]
    if not val:
        pytest.skip("run_validation.py has not been run")
    for pop, entry in val.items():
        summary, ladder = entry["league_out_summary"], entry["ladder"]
        assert abs(summary["grouped_by_match"] - ladder["full"]["log_loss"]) <= 1e-4, (
            f"{pop}: league-out references {summary['grouped_by_match']} as the match-grouped "
            f"model while the ladder reports {ladder['full']['log_loss']}")


def test_q2_overlap_is_bounded_by_both_its_parents(payload):
    """The overlap cannot exceed either half it is an intersection of.

    A corner reaches `q2` only by being raced on tracking AND cleared in the event chain,
    so an overlap above either parent means the join matched rows it should not have:
    the failure a `how="inner"` on the wrong key produces, which still returns a
    plausible agreement rate."""
    q2, race, sit = (payload["figures"]["q2"], payload["figures"]["race"],
                     payload["figures"]["situation"])
    if not q2 or not race or not sit:
        pytest.skip("run_q2.py, run_race.py or run_situation.py has not been run")
    assert q2["overlap_corners"] <= race["raced_corners"]
    assert q2["overlap_corners"] <= sit["cleared_corners"]
    assert q2["overlap_matches"] <= race["matches"]
    assert q2["n_labelled"] <= q2["overlap_corners"]
    lo, hi = q2["independent_agreement_ci"]
    assert lo <= q2["independent_agreement_pct"] <= hi, "the agreement point is outside its own CI"


def test_sufficiency_separates_the_powered_half_from_the_bounded_one(payload):
    """The two halves must not be readable as one measurement.

    `grant` answers the contract's question on five leagues with no tracking in it; `overlap`
    is what tracking adds on 30 matches. Three failures this catches: an overlap population
    above either parent, which is the join matching rows it should not have; a headline
    interval that does not bracket its own point, which means the bootstrap and the point ran
    on different rows; and an overlap null published without the effect size the design could
    have resolved, which is the one thing argument.md §11 does not license."""
    suf, race, corpus = (payload["figures"]["sufficiency"], payload["figures"]["race"],
                         payload["figures"]["corpus"])
    if not suf or not race or not corpus:
        pytest.skip("run_sufficiency.py, run_race.py or build_table.py has not been run")

    g = suf["grant"]
    assert g["corners"] <= corpus["corners_with_360"], "more corners than carry a delivery 360"
    assert g["positive"] + g["negative"] == g["corners"]
    assert g["leagues"] == 5, "the grant half must span every competition-season"
    assert g["controls_all_clean"], "a shuffled control beat its own marginal entropy"
    for name, a in g["by_arm"].items():
        assert 0.0 < a["log_loss"] < 5.0, f"{name} scored an impossible log-loss"
    # The rung that carries any claim about the freeze-frame is `delivery`; a section that
    # lost it would still publish a plausible headline. The deterministic floor lives in
    # `by_arm`, not in the ladder, because s03's intercept-only rung does not reproduce.
    assert "delivery" in g["ladder"]
    assert {"prior", "delivery", "geometry360", "gbt360", "static_race"} <= set(g["by_arm"])
    for key in ("headline_delta", "geometry_over_delivery"):
        lo, hi = g[key]["ci"]
        assert lo <= g[key]["delta"] <= hi, f"{key} sits outside its own CI"

    o = suf.get("overlap")
    if not o:
        pytest.skip("the tracked overlap sat under the publication floor")
    assert o["corners"] <= race["raced_corners"]
    assert o["corners"] <= g["corners"], "more tracked corners than the grant they came from"
    assert o["matches"] <= race["matches"]
    assert o["mde_80"] > 0, "a null with no detectable-effect size is a sample size"
    lo, hi = o["headline_delta"]["ci"]
    assert lo <= o["headline_delta"]["delta"] <= hi, "the delta sits outside its own CI"
    assert set(o["by_arm"]) == {"prior", "snapshot", "kick", "anchored", "race"}
    assert o["arms_separate"] is (not o["headline_delta"]["covers_zero"])
    # A standing regression on the leak that made a fold-constant arm beat every real one.
    assert o["prior_recalibration_leaks"] is True, (
        "recalibrating the fold-constant prior stopped tripping its own shuffle control, so "
        "either the control lost its power or the prior stopped being fold-constant")


def test_causal_reports_the_registered_primary_and_every_target(payload):
    """S05's effect may not be published as one number without the choices behind it.

    Three failures this catches. A record whose headline is not the primary run_causal
    registers, which would publish another specification under its name. An ATO row carrying
    clipped units, which would mean the bypass broke and the one estimand this overlap supports
    had started papering over the violation it exists to declare. And an interval that does not
    bracket its own point, which means the SE and the estimate ran on different rows."""
    from run_causal import PRIMARY_SPEC
    c = payload["figures"]["causal"]
    if not c:
        pytest.skip("run_causal.py has not been run")
    pop = c["population"]
    assert pop["treated"] + pop["control"] == pop["corners"]
    assert pop["corners"] <= payload["figures"]["corpus"]["corners_with_360"]
    assert c["primary_spec"] == PRIMARY_SPEC and set(c["by_spec"]) == {PRIMARY_SPEC}
    # the transition arm is a measured refusal, and the record has to carry it as one
    assert c["conceded_xg_degenerate"] is True

    for spec, entry in c["by_spec"].items():
        assert set(entry["by_target"]) == {"ate", "att", "ato"}, f"{spec} lost a target"
        for tg, t in entry["by_target"].items():
            lo, hi = t["ci"]
            assert lo <= t["effect"] <= hi, f"{spec}/{tg} sits outside its own CI"
            assert t["covers_zero"] is (lo <= 0 <= hi)
            assert t["se"] > 0, f"{spec}/{tg} reports a zero SE"
        assert entry["by_target"]["ato"]["n_clipped"] == 0, f"{spec}: the ATO clipped"
        assert 0 < entry["by_target"]["ato"]["ess"] <= pop["corners"]
    # the primary specification without team and league, published beside the headline; a record
    # without it lacks the fixed-effects specification and has to be regenerated
    assert c.get("without_fixed_effects"), ("causal.parquet carries no sensitivity block: it "
                                            "lacks the fixed-effects specification; re-run "
                                            "pipeline/run_causal.py")
    assert set(c["without_fixed_effects"]) == {"ate", "att", "ato"}
    for tg, t in c["without_fixed_effects"].items():
        lo, hi = t["ci"]
        assert lo <= t["effect"] <= hi, f"without_fixed_effects/{tg} sits outside its own CI"


def test_the_specification_curve_covers_every_feature_group(payload):
    """A null has to say across how many specifications it holds.

    The confounder set is a choice this data cannot make: treatment and congestion are both
    read in the delivery freeze-frame, so neither precedes the other. The curve is what
    turns that into a measurement, and it is only a measurement if it spans every group,
    with the treatment's own group excluded from all of them."""
    from s03_value import FEATURE_GROUPS
    from s05_causal import TREATMENT_GROUP
    c = payload["figures"]["causal"]
    if not c or "spec_curve" not in c:
        pytest.skip("run_causal.py has not been run")
    curve = c["spec_curve"]
    assert curve["n_specs"] == len(FEATURE_GROUPS), (
        "the curve must carry one rung per feature group other than the treatment's, "
        "plus the full design")
    assert "full" in curve["rungs"]
    assert f"drop_{TREATMENT_GROUP}" not in curve["rungs"], "the treatment's group is never a rung"
    for g in FEATURE_GROUPS:
        if g != TREATMENT_GROUP:
            assert f"drop_{g}" in curve["rungs"], f"{g} has no rung"
    for tg, b in curve["by_target"].items():
        lo, hi = b["effect_range"]
        assert lo <= hi
        assert 0 <= b["n_excluding_zero"] <= curve["n_specs"]
        # a rung whose propensity model collapses would show up here as a NaN range
        assert lo == lo and hi == hi, f"{tg} has a rung with no effect"


def test_the_aerial_block_counts_coverage_over_the_cleared_deliveries_only(monkeypatch, tmp_path):
    """A null over a column that is mostly missing is a statement about coverage, so the block
    publishes the coverage itself. The denominator is the deliveries a defender cleared, where the
    clearing contact carries the aerial context, and a delivery nobody cleared counts nowhere."""
    rows = [{"cleared": True, "clear_is_header": True, "clear_aerial_won": None,
             "first_contact_is_header": True, "first_contact_hops_rating": None},
            {"cleared": True, "clear_is_header": False, "clear_aerial_won": True,
             "first_contact_is_header": None, "first_contact_hops_rating": 0.7},
            {"cleared": False, "clear_is_header": True, "clear_aerial_won": True,
             "first_contact_is_header": True, "first_contact_hops_rating": 0.9}]
    out = _toy(monkeypatch, tmp_path, {"corners.parquet": rows}).aerial()
    assert out == {"cleared": 2, "coverage_pct": {
        "clear_is_header": 100.0, "clear_aerial_won": 50.0, "first_contact_is_header": 50.0,
        "first_contact_hops_rating": 50.0}}
    assert _toy(monkeypatch, tmp_path / "absent", {}).aerial() == {}


def test_race_carries_the_ball_bound_and_scores_the_detected_subset(payload):
    """The race may not publish a figure without the bound it rests on beside it.

    Three failures this catches. A record with no share of interpolated balls, which would
    let every race read as a detection. A detected-ball subset larger than the races it is
    cut from, or with more labels than members, which means the mask was applied to the wrong
    frame. And a subset count that disagrees with the share it was cut by, which means the two
    were computed on different rows."""
    r = payload["figures"]["race"]
    if not r:
        pytest.skip("run_race.py has not been run")
    assert 0.0 <= r["ball_estimated_at_contact_pct"] <= 100.0
    assert 0.0 <= r["ball_estimated_at_loose_pct"] <= 100.0
    assert 0 <= r["ball_absent_at_loose"] <= r["raced_corners"]
    v = r["visible_ball_at_contact"]
    assert v["labelled"] <= v["raced_corners"] <= r["raced_corners"]
    expected = r["raced_corners"] * (1 - r["ball_estimated_at_contact_pct"] / 100)
    assert abs(v["raced_corners"] - expected) <= 1, "the subset and the share disagree"
    assert isinstance(v["below_floor"], bool)
    if v["labelled"]:
        assert 0.0 <= v["att_recovered_pct"] <= 100.0 and 0.0 <= v["self_agreement_pct"] <= 100.0


def _brackets(ci: list, point: float) -> bool:
    return ci[0] <= point <= ci[1]


@pytest.mark.parametrize("pop", ["free_kick_flank"])
def test_free_kick_populations_hold_the_cross_stage_invariants(payload, pop):
    """The corner checks above, carried to the flank free-kick population's own artefacts.

    Structural only: counts that partition, intervals that bracket their point, controls that do
    not beat their entropy and joins bounded by their parents. Nothing here asserts a free-kick
    finding, because a finding is a measurement and not a premise."""
    fk = payload["figures"].get(pop) or {}
    corpus, sit = fk.get("corpus"), fk.get("situation")
    if not corpus:
        pytest.skip(f"build_table.py --kinds {pop} has not been run")
    if sit:
        seg = corpus["by_segment"].get("after_clearance", {"shots": 0, "goals": 0})
        assert sit["after_clearance_shots"] == seg["shots"] + seg.get("own_goals", 0)
        assert sit["after_clearance_goals"] == seg["goals"] + seg.get("own_goals", 0)
        assert sum(c["n"] for c in sit["by_state"].values()) == sit["four_state_known"]
    for p, entry in (fk.get("validation") or {}).items():
        assert entry["controls_all_clean"], f"{pop}/{p}: a shuffled control beat chance"
        summary, ladder = entry["league_out_summary"], entry["ladder"]
        assert abs(summary["grouped_by_match"] - ladder["full"]["log_loss"]) <= 1e-4
    q2, race = fk.get("q2"), fk.get("race")
    if q2 and race and sit:
        assert q2["overlap_corners"] <= min(race["raced_corners"], sit["cleared_corners"])
        assert q2["n_labelled"] <= q2["overlap_corners"]
        assert _brackets(q2["independent_agreement_ci"], q2["independent_agreement_pct"])
    g = (fk.get("sufficiency") or {}).get("grant")
    if g:
        assert g["corners"] <= corpus["corners_with_360"]
        assert g["positive"] + g["negative"] == g["corners"]
        assert g["controls_all_clean"]
        for key in ("headline_delta", "geometry_over_delivery"):
            assert _brackets(g[key]["ci"], g[key]["delta"])
    c = fk.get("causal")
    if c:
        assert c["population"]["treated"] + c["population"]["control"] == c["population"]["corners"]
        for spec, entry in c["by_spec"].items():
            assert set(entry["by_target"]) == {"ate", "att", "ato"}
            for t in entry["by_target"].values():
                if t["effect"] == t["effect"]:                      # a degenerate cell is NaN
                    assert _brackets(t["ci"], t["effect"])


def test_the_flank_cut_block_reads_both_records(monkeypatch, tmp_path):
    """The annex block on a toy record: counts as integers, the interval beside each point, and a
    cut without bounds left undecided rather than read as excluding zero."""
    import pandas as pd
    import report_figures as RF
    pd.DataFrame([{"band": "long_corners", "n": 10, "n_360": 9, "xg_win": 0.04, "xg_win_lo": 0.03,
                   "xg_win_hi": 0.05, "xg_aft": 0.02, "xg_aft_lo": 0.01, "xg_aft_hi": 0.03,
                   "fc_x_median": 114.5, "won_fc": 0.33}]).to_parquet(tmp_path / "flank_cut_bands.parquet")
    pd.DataFrame([{"cut": 85.0, "measure": "p_aft", "below_minus_above": -0.03, "lo": -0.05,
                   "hi": -0.01, "n_below": 5, "n_above": 6},
                  {"cut": 85.0, "measure": "won_fc", "below_minus_above": 0.1, "lo": float("nan"),
                   "hi": float("nan"), "n_below": 5, "n_above": 6}]).to_parquet(
        tmp_path / "flank_cut_contrasts.parquet")
    monkeypatch.setattr(RF, "_DER", tmp_path)
    out = RF.flank_cut()
    band = out["bands"]["long_corners"]
    assert band["deliveries"] == 10 and band["xg_aft"]["ci"] == [0.01, 0.03] and band["won_fc"] == 0.33
    cut = out["cuts"]["85"]
    assert cut["n_above"] == 6 and cut["p_aft"]["covers_zero"] is False
    assert cut["won_fc"]["ci"] is None and cut["won_fc"]["covers_zero"] is None


def test_the_flank_cut_counts_the_paper_s_population_above_its_floor(payload):
    """The annex's kicks from 85 are the paper's flank population as the scope block counts it, and
    its bands partition every flank kick it read."""
    fc, scope = payload["figures"]["flank_cut"], payload["figures"]["scope"]
    if not fc or "free_kick_flank" not in scope:
        pytest.skip("run_flank_cut.py or eda_build.py has not been run")
    cut = fc["cuts"]["85"]
    assert cut["n_above"] == scope["free_kick_flank"]["deliveries"]
    flank = [v for k, v in fc["bands"].items() if k != "long_corners"]
    assert sum(v["deliveries"] for v in flank) == cut["n_below"] + cut["n_above"]


def test_the_scope_block_counts_each_population_inside_the_tracked_sample_and_with_a_360(
        monkeypatch, tmp_path):
    """The scope figure prints these counts, so they come from the record: per population the
    deliveries in the tracked matches and the share whose delivery carries a 360 (a modelled
    population's table, a long throw's own flag), and beside them the grant's and the tracked
    sample's match counts."""
    deliv = ([{"set_piece_type": "corner", "league": "l", "attacking_team": "A", "match_id": 1}] * 3
             + [{"set_piece_type": "corner", "league": "l", "attacking_team": "B", "match_id": 2}]
             + [{"set_piece_type": "free_kick_flank", "league": "l", "attacking_team": "A",
                 "match_id": 1}] * 2)
    throws = [{"league": "l", "attacking_team": t, "match_id": m, "delivery_360": d}
              for t, m, d in (("A", 1, True), ("B", 2, False))]
    RF = _toy(monkeypatch, tmp_path, {"eda_deliveries.parquet": deliv,
                                      "eda_matches.parquet": [{"match_id": 1}, {"match_id": 2}],
                                      "eda_throw_ins.parquet": throws,
                                      "corners.parquet": [{"x": 0}] * 3,
                                      "corners_free_kick_flank.parquet": [{"x": 0}] * 2})
    monkeypatch.setattr(RF._loader, "tracked_matches", lambda: [1])
    out = RF.scope()
    assert (out["matches"], out["tracked_matches"]) == (2, 1)
    got = {k: (v["deliveries"], v["tracked_deliveries"], v["with_360"], v["with_360_pct"])
           for k, v in out.items() if isinstance(v, dict)}
    assert got == {"corner": (4, 3, 3, 75.0), "free_kick_flank": (2, 2, 2, 100.0),
                   "long_throw": (2, 1, 1, 50.0)}


def test_the_throw_in_block_keeps_each_segment_s_own_goals_apart(monkeypatch, tmp_path):
    """A segment's shots and goals leave its own goals out, and the block carries them beside so a
    reader adds them to the goal count: an own goal is a goal and no shot."""
    seg = [{"block": "segment", "segment": s, "shots": n, "goals": g, "own_goals": o, "thin": ""}
           for s, n, g, o in (("direct", 30, 1, 0), ("prolongation", 25, 5, 1))]
    comp = [{"block": "comparison", "population": "throw_in", "set_pieces": 100, "shots": 55,
             "shots_per_100": 55.0, "goals_per_100": 7.0, "aft_shot_pct": 45.0, "direct_shots": 30,
             "direct_obs_exp": 0.2, "lo": 0.0, "hi": 0.5, "excludes_1": True}]
    out = _toy(monkeypatch, tmp_path, {"throw_in_probe.parquet": seg + comp}).throw_ins()
    assert out["by_segment"]["prolongation"] == {"shots": 25, "goals": 5, "own_goals": 1,
                                                 "below_floor": False}
    assert out["by_population"]["throw_in"]["shots"] == 55


def _toy(monkeypatch: pytest.MonkeyPatch, tmp_path: Path, frames: dict) -> ModuleType:
    """report_figures reading `frames` (name -> rows) out of a toy data/derived."""
    import pandas as pd
    import report_figures as RF
    for name, rows in frames.items():
        p = tmp_path / name
        p.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(rows).to_parquet(p)
    monkeypatch.setattr(RF, "_DER", tmp_path)
    return RF


def test_the_unknown_states_split_by_what_is_missing_and_need_the_staker_count():
    """A cleared corner the four-state cross cannot place lacks its delivery 360, its next touch or both, and the
    three counts add up to the states it could not place; a table without the staker count is refused rather
    than split as if every delivery had a 360."""
    import pandas as pd
    import report_figures as RF
    s = pd.DataFrame({"n_stakers": [1.0, None, 2.0, None], "won_second_ball": [1.0, 0.0, None, None]})
    assert RF._unknown_split(s, 1) == {"total": 3, "no_delivery_360": 1, "no_next_touch": 1, "neither": 1}
    with pytest.raises(ValueError, match="do not partition"):
        RF._unknown_split(s, 2)
    with pytest.raises(KeyError):
        RF._unknown_split(s.drop(columns="n_stakers"), 1)


def test_the_causal_block_refuses_a_record_without_the_primary_specification(monkeypatch, tmp_path):
    rows = [{"block": "headline", "spec": "not_the_primary", "target": "att_xg"}]
    with pytest.raises(ValueError, match="carries no"):
        _toy(monkeypatch, tmp_path, {"causal.parquet": rows}).causal()


def test_the_gpu_block_reads_both_populations_and_parses_the_verdict(monkeypatch, tmp_path):
    row = {"point": -0.001, "lo": -0.003, "hi": 0.001, "se": 0.001, "mde80": 0.005,
           "below_margin": False}
    rows = [{"population": p, "block": "seed", "key": str(s), **row} for p in ("corner", "flank")
            for s in range(2)]
    rows += [{"population": p, "block": b, "key": k, **row}
             for p in ("corner", "flank") for b, k in (("control", "shuffled_frame"),
                                                       ("head_to_head", "encoder-full"))]
    rows += [{"population": p, "block": "verdict", "key": "GPU1",
              "params": "{'enters': False, 'seeds_below_margin': [], 'control_clean': True}"}
             for p in ("corner", "flank")]
    rows += [{"population": "corner", "block": "task", "key": "0/0", "seconds": 5.0}]
    out = _toy(monkeypatch, tmp_path, {"frame_gpu.parquet": rows}).frame_gpu()
    assert set(out) == {"corner", "flank"}
    c = out["corner"]
    assert c["verdict"] == {"enters": False, "seeds_below_margin": [], "control_clean": True}
    assert c["seeds"]["1"]["ci"] == [-0.003, 0.001] and c["encoder_minus_full"]["covers_zero"] is True
    assert c["tasks"] == 1 and out["flank"]["tasks"] == 0


def test_the_zone_comparison_block_reads_m1_and_its_robustness(monkeypatch, tmp_path):
    refit = {"population": "corner", "block": "refit", "n": 4, "point": -0.002, "lo": -0.003,
             "hi": -0.001, "bonf_lo": -0.004, "bonf_hi": 0.0005, "bonf_pivotal_lo": -0.005,
             "bonf_pivotal_hi": 0.001, "mde80": 0.003}
    rows = [{**refit, "key": f"{r}-F"} for r in "ABCDE"] + [{**refit, "key": "C-design"}]
    rows += [{"population": "corner", "block": "stability", "key": "C-F", "seed": s, "hi": h}
             for s, h in ((1, 0.0005), (0, -0.0001))]
    rows += [{"population": "corner", "block": "control", "key": "Cshuffled-design", "clean": True},
             {"population": "corner", "block": "verdict", "key": "M1",
              "params": "{'keeps': 'C', 'qualify': ['C'], 'below_zero': ['C'], 'unstable': [], "
                        "'dirty_control': [], 'seeds_run': True}"},
             {"population": "free_kick_flank", "block": "skipped", "key": "too_few_matches"}]
    rob = [{"population": p, "block": "robustness", "key": "C confirmed",
            "params": "{'confirmed': True, 'league_in_design': True}"}
           for p in ("corner", "free_kick_flank")]
    out = _toy(monkeypatch, tmp_path, {"zone_comparison.parquet": rows,
                                       "zone_robustness.parquet": rob}).zone_comparison()
    c = out["corner"]
    assert c["verdict"]["keeps"] == "C" and set(c["rivals"]) == {f"{r}-F" for r in "ABCDE"}
    assert c["rivals"]["C-F"]["normal_bonferroni"] == [-0.004, 0.0005]
    assert c["stability"]["C-F"] == [-0.0001, 0.0005] and c["controls"]["Cshuffled-design"] is True
    assert c["robustness"] == {"confirmed": True, "league_in_design": True}
    assert out["free_kick_flank"] == {"skipped": True, "robustness": {"confirmed": True,
                                                                     "league_in_design": True}}


def test_the_duel_price_block_reads_the_switch_s_partition_records(monkeypatch, tmp_path):
    price = {"sample": "all", "zone": "six_near", "n_won": 30, "n_lost": 40, "thin": False,
             "price": 0.05, "price_lo": 0.01, "price_hi": 0.09, "whatif_xg_per_team_season": 0.2}
    frames = {"duel_prices_F.parquet": [{**price, "duel": d} for d in ("first", "second")]
              + [{**price, "sample": "development", "duel": "first"}],
              "duel_prices_F_decision.parquet": [{"population": "corner", "C1a": False,
                                                  "C1a_any_seed": True, "seeds": 5, "zones": "F",
                                                  "second_spearman": 0.4}],
              "duel_prices_F_c1a.parquet": [{"seed": 0, "holds": True, "spearman": 0.8,
                                             "spearman_lo": 0.1, "spearman_hi": 0.95}]}
    rf = _toy(monkeypatch, tmp_path, frames)
    monkeypatch.setattr(rf, "ZONE_PARTITION", "F")
    out = rf.duel_prices()
    assert out["decision"]["C1a"] is False and out["decision"]["second_spearman"] == 0.4
    assert out["c1a_by_seed"]["0"] == {"holds": True, "spearman": 0.8, "ci": [0.1, 0.95]}
    assert out["prices"]["first"]["six_near"]["ci"] == [0.01, 0.09] and set(out["prices"]) == {
        "first", "second"}


def test_the_club_power_block_reads_its_six_records(monkeypatch, tmp_path):
    t = "_free_kick_flank"
    frames = {f"club_power_decision{t}.parquet": [{"C12a": True, "C12a_any_seed": True, "seeds": 5,
                                                   "luck_share": 0.9, "season_median": 24.0}],
              f"club_power_measures{t}.parquet": [{"measure": "aft_goal", "mu": 0.02, "sd": 0.14,
                                                   "icc_team": 0.01, "icc_team_lo": 0.0,
                                                   "icc_team_hi": 0.02, "icc_match_in_team": 0.03,
                                                   "icc_match_in_team_lo": 0.01,
                                                   "icc_match_in_team_hi": 0.05, "rho": 0.004,
                                                   "rho_lo": 0.0, "rho_hi": 0.01}],
              f"club_power_power{t}.parquet": [{"measure": "aft_goal", "lift": 0.5,
                                                "corners_per_arm": 3000.0,
                                                "seasons_before_after": 250.0}],
              f"club_power_half_season{t}.parquet": [{"measure": "aft_goal",
                                                      "half_season_mde80_lift": 2.5,
                                                      "formula_lift_at_half_season": 2.4}],
              f"club_power_reliability{t}.parquet": [{"metric": "aft_goal", "full_r_sb": 0.1,
                                                      "full_lo": -0.1, "full_hi": 0.3}],
              f"club_power_c12a{t}.parquet": [{"seed": 0, "version": "recalibrated",
                                               "scope": "all", "inside": True}]}
    out = _toy(monkeypatch, tmp_path, frames).club_power(t)
    assert out["decision"]["C12a"] is True and out["decision"]["season_median"] == 24.0
    m = out["measures"]["aft_goal"]
    assert m["icc_team"] == {"point": 0.01, "ci": [0.0, 0.02]} and m["rho"]["ci"] == [0.0, 0.01]
    assert out["power"]["aft_goal"]["0.5"] == {"corners_per_arm": 3000.0, "seasons_before_after": 250.0}
    assert out["reliability"]["aft_goal"] == {"full_r_sb": 0.1, "full_ci": [-0.1, 0.3]}
    assert out["half_season"]["aft_goal"]["mde80_lift"] == 2.5


def test_the_top_teams_block_reads_the_decisions_the_primaries_and_what_the_narrative_reads(
        monkeypatch, tmp_path):
    """The decisions and the two primaries, and beside them the secondary contrasts at seed 0 (G's
    season ones left out), each group's pooled rates and each measure's reliability."""
    c = {"seed": 0, "role": "primary", "metric": "aft_xg_per_corner", "point": 0.01, "lo95": -0.01,
         "hi95": 0.03, "lo95_normal": -0.012, "hi95_normal": 0.032, "mde80": 0.02, "p_perm": 0.3}
    sec = {**c, "role": "secondary", "metric": "corners_per_match", "point": 1.5, "q_bh": 0.001}
    team = {"corners": 20, "matches": 4, "xg_direct": 1.0, "xg_aft": 0.5, "xg_late": 0.5,
            "xg_total": 2.0, "in_T2": True}
    frames = {"top_teams_decisions.parquet": [{"T1": False, "T2": False, "G1": True,
                                               "T_lo95_by_seed": [-0.01, -0.02]}],
              "top_teams_contrasts.parquet": [{**c, "group": "T2", "scope": "season"},
                                              {**c, "group": "G", "scope": "split_half"},
                                              {**c, "seed": 1, "group": "T2", "scope": "season"},
                                              {**sec, "group": "T2", "scope": "season"},
                                              {**sec, "group": "G", "scope": "season"}],
              "top_teams_by_league.parquet": [{"league": "serie_a", "corners": 900.0, "gap": 0.004}],
              "top_teams_table.parquet": [team, {**team, "corners": 10, "matches": 5, "xg_direct": 0.2,
                                                 "xg_aft": 0.2, "xg_late": 0.1, "xg_total": 0.5,
                                                 "in_T2": False}],
              "top_teams_reliability.parquet": [{"metric": "aft_xg", "full_r_sb": 0.0349,
                                                 "full_lo": -0.3252, "full_hi": 0.2844}]}
    out = _toy(monkeypatch, tmp_path, frames).top_teams()
    assert out["decisions"]["G1"] is True and out["decisions"]["T_lo95_by_seed"] == [-0.01, -0.02]
    assert out["primary"]["T2"]["ci95"] == [-0.01, 0.03] and out["primary"]["G"]["p_perm"] == 0.3
    assert out["by_league"]["serie_a"] == 0.004
    assert out["secondary"] == {"T2": {"corners_per_match": {"point": 1.5, "ci95": [-0.01, 0.03],
                                                             "p_perm": 0.3, "q_bh": 0.001}},
                                "G": {}}
    assert out["rates"] == {"T2": {"corners_per_match": 5.0, "direct_xg_per_corner": 0.05,
                                   "aft_xg_per_corner": 0.025, "late_xg_per_corner": 0.025,
                                   "aft_xg_share": 0.25},
                            "rest": {"corners_per_match": 2.0, "direct_xg_per_corner": 0.02,
                                     "aft_xg_per_corner": 0.02, "late_xg_per_corner": 0.01,
                                     "aft_xg_share": 0.4}}
    assert out["reliability"] == {"aft_xg": {"full_r_sb": 0.035, "full_ci": [-0.325, 0.284]}}


def test_the_tracking_curve_block_reads_an_estimable_and_a_thin_statistic(monkeypatch, tmp_path):
    fit = {"stat": "S1", "estimable": True, "n_rows": 150, "intercept": 0.1, "slope": -0.5,
           "slope_lo": -0.7, "slope_hi": -0.3, "TRK1": True, "effect": 0.02, "full_n": 29,
           "full_width": 0.04, "n_star_fit": 60.0, "n_star_fit_lo": 40.0, "n_star_fit_hi": 90.0,
           "n_star_sqrt": 58.0}
    rows = [{**fit, "seed": 0}, {**fit, "seed": 1, "TRK1": False},
            {"stat": "S2", "estimable": False, "n_rows": 15, "full_n": 12}]
    out = _toy(monkeypatch, tmp_path, {"tracking_curve_fit.parquet": rows}).tracking_curve()
    assert out["S1"]["TRK1_by_seed"] == [True, False] and out["S1"]["TRK1"] is False
    assert out["S1"]["slope_ci"] == [-0.7, -0.3] and out["S1"]["n_star_fit_ci"] == [40.0, 90.0]
    assert out["S2"] == {"estimable": False, "n_rows": 15, "matches": 12}


def test_a_team_fit_that_did_not_converge_is_published_as_not_reported(monkeypatch, tmp_path):
    rows = [{"block": "population", "key": "teams", "n": 96, "point": 0.01},
            {"block": "population", "key": "goals", "n": 30},
            {"block": "population", "key": "corners", "n": 2300},
            {"block": "diagnostics", "key": "M1/converged", "point": 0.0}]
    out = _toy(monkeypatch, tmp_path,
               {"team_shrinkage_free_kick_flank.parquet": rows}).team_shrinkage("_free_kick_flank")
    assert out["reported"] is False and out["teams"] == 96 and "cases" not in out


def test_the_duel_prices_are_read_under_the_switch_s_partition():
    """run_duel_prices names its records by the partition it ran under, so the record reads the
    switch's and never another partition's left on disk."""
    import report_figures as RF
    names = [n for n in RF.SOURCES if n.startswith("duel_prices_")]
    assert names and all(n.startswith(f"duel_prices_{RF.ZONE_PARTITION}_") or
                         n.startswith(f"duel_prices_{RF.ZONE_PARTITION}.") for n in names), names


def test_the_flank_section_carries_every_twin(monkeypatch, tmp_path):
    out = _toy(monkeypatch, tmp_path, {}).SECTIONS["free_kick_flank"]()
    assert {"falsification", "coverage_360", "tracking_window", "counter", "first_contact",
            "first_contact_frame", "positive_control", "aim_zone", "team_shrinkage", "aerial",
            "duel_prices", "club_power", "top_teams", "tracking_curve"} <= set(out)


def _first_contact_rows(below: dict) -> list:
    """A toy first-contact record: the step-0 counts, the population, one control, one MDE, and a
    refit row with five stability seeds for each decision in `below` (key -> wholly below zero)."""
    rows = [{"block": "step0", "key": "all/grant", "n": 10},
            {"block": "step0", "key": "all/t1_has_360", "n": 9},
            {"block": "population", "key": "corners", "n": 9, "point": 0.6},
            {"block": "population", "key": "near_spot_share", "n": 9, "point": 0.3},
            {"block": "control", "key": "geometry_1/global", "clean": True},
            {"block": "power", "key": "mde_80/geometry_1-delivery", "n": 9, "point": 0.005,
             "hi": 0.01}]
    for key, b in below.items():
        hi = -0.001 if b else 0.004
        rows.append({"block": "refit", "key": key, "n": 300, "point": -0.002, "lo": -0.008, "hi": hi,
                     "pct_lo": -0.006, "pct_hi": 0.002, "bonf_lo": -0.01, "bonf_hi": 0.006,
                     "below_zero": b, "width_ratio": 1.2})
        rows += [{"block": "stability", "key": key, "seed": s, "n": 2000, "hi": hi, "below_zero": b}
                 for s in range(5)]
    return rows


def test_the_first_contact_block_closes_v1_on_the_phase_two_record(monkeypatch, tmp_path):
    """Phase one leaves V1 undecided. The `_gbt` record repeats its decisions with the strong
    learner's beside them and, where it exists, is the one read, so V1 closes on it."""
    weak = {k: False for k in ("static_race_1-delivery", "geometry_1-delivery",
                               "static_race_1-static_race_D", "geometry_1-geometry360_D",
                               "facts_geometry_1-facts_1")}
    t = "_free_kick_flank"
    one = _toy(monkeypatch, tmp_path / "one", {f"first_contact{t}.parquet": _first_contact_rows(weak)})
    out = one.first_contact_frame(t)
    assert out["phase_two"] is False and out["verdicts"]["V1"].startswith("undecided")
    two = _toy(monkeypatch, tmp_path / "two", {
        f"first_contact{t}.parquet": _first_contact_rows(weak),
        f"first_contact{t}_gbt.parquet": _first_contact_rows({**weak, "gbt_1-delivery": False})})
    out = two.first_contact_frame(t)
    assert out["phase_two"] is True and out["verdicts"]["V1"] == "false"
    assert "gbt_1-delivery" in out["decisions"]


def test_the_penalty_block_credits_each_kick_to_its_phase(monkeypatch, tmp_path):
    """Q1 beside itself with the penalties won in the episode counted as shots: a kick whose foul
    came after the first contact joins the after-first-touch side, one from before it only the
    total, and a penalty with no kick taken is counted but never credited."""
    nan = float("nan")
    pen = [{"set_piece_id": s, "set_piece_type": k, "penalty": p, "phase": ph, "kick_id": kid,
            "kick_goal": g, "kick_xg": x}
           for s, k, p, ph, kid, g, x in (("a", "corner", True, "after", "k1", True, 0.7835),
                                          ("b", "corner", True, "before", "k2", False, 0.7835),
                                          ("c", "corner", False, None, None, False, nan),
                                          ("d", "corner", True, "after", None, False, nan),
                                          ("e", "free_kick_flank", False, None, None, False, nan))]
    grain = [{"set_piece_id": s, "outcome": o} for s, o in (
        ("a", "clear_safe"), ("b", "no_second_ball"), ("c", "shot_direct"), ("d", "clear_safe"),
        ("e", "clear_safe"))]
    shots = [{"segment": s, "is_goal": g, "xg": x} for s, g, x in (
        ("direct", True, 0.3), ("flick_on", False, 0.1), ("after_clearance", True, 0.2),
        ("late", False, 0.1), ("direct", False, 0.1))]
    out = _toy(monkeypatch, tmp_path, {
        "eda_penalties.parquet": pen, "eda_deliveries.parquet": grain, "shots.parquet": shots,
        "shots_free_kick_flank.parquet": shots[:1]}).penalties()
    c = out["corner"]
    assert (c["deliveries"], c["won_a_penalty"], c["kicks"], c["kick_goals"]) == (4, 3, 2, 1)
    assert c["by_phase"] == {"before": 1, "at": 0, "after": 2} and c["kick_xg"] == 1.57
    assert c["labelled_as"] == {"clear_safe": 2, "no_second_ball": 1}
    assert c["after_first_touch_shots_pct"] == {"without": 40.0, "with": 42.86}
    assert c["after_first_touch_goals_pct"] == {"without": 50.0, "with": 66.67}
    assert c["after_first_touch_xg_pct"] == {"without": 37.5, "with": 45.78}
    assert out["free_kick_flank"]["won_a_penalty"] == 0


def test_an_own_goal_counts_in_the_goals_and_in_no_shot_figure(monkeypatch, tmp_path):
    """The record applies one rule: an own goal is a goal the delivery produced and no shot, so the
    shot counts, the conversions and the after-first-touch share of shots leave it out, and the
    goals, the share of goals and the own-goal counts keep it."""
    rows = [{"segment": s, "is_goal": g, "xg": x, "shot_outcome": o, "league": "a", "set_piece_id": i}
            for i, (s, g, x, o) in enumerate((
                ("direct", True, 0.3, "Goal"), ("direct", False, 0.1, "Saved"),
                ("after_clearance", True, 0.2, "Goal"), ("after_clearance", False, 0.1, "Saved"),
                ("after_clearance", True, 0.0, "Own Goal")))]
    out = _toy(monkeypatch, tmp_path, {
        "corners.parquet": [{"match_id": 1, "league": "a", "defending_team": "x"}],
        "shots.parquet": rows}).corpus()
    assert (out["shots_on_the_frontier"], out["goals_on_the_frontier"],
            out["own_goals_on_the_frontier"]) == (4, 3, 1)
    assert out["after_first_touch_shots_pct"] == 50.0 and out["after_first_touch_goals_pct"] == 66.67
    aft = out["by_segment"]["after_clearance"]
    assert (aft["shots"], aft["goals"], aft["own_goals"], aft["conv_pct"]) == (2, 1, 1, 50.0)
    league = _toy(monkeypatch, tmp_path, {"corners.parquet": [{"league": "a"}],
                                          "shots.parquet": rows}).per_league()["a"]
    assert (league["shots"], league["goals"], league["own_goals"]) == (4, 3, 1)


def test_a_figure_the_record_cannot_compute_is_published_as_null():
    """NaN never equals itself, so a NaN figure would make an unchanged record read as stale."""
    import numpy as np
    import report_figures as RF
    v = {"a": [float("nan"), 1.0], "b": {"c": np.float64("nan"), "d": "x"}, "e": None}
    assert RF._json_safe(v) == {"a": [None, 1.0], "b": {"c": None, "d": "x"}, "e": None}


def test_generated_file_is_current(payload):
    """The committed measured.json must match a fresh build.

    Failing here means an artefact moved and the record was not regenerated: run
    `python pipeline/report_figures.py`."""
    if not _OUT_JSON.exists():
        pytest.skip("measured.json has not been generated yet")
    on_disk = json.loads(_OUT_JSON.read_text())
    assert on_disk["figures"] == payload["figures"], (
        "measured.json is stale — re-run `python pipeline/report_figures.py`")


def test_markdown_renders_every_section(payload):
    """The human twin must show each section, including the empty ones."""
    md = to_markdown(payload)
    for section in payload["figures"]:
        assert f"## {section}" in md
