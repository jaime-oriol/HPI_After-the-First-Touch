"""Tests for run_team_shrinkage: the diagnostics gate, the rank summary and the verdict rules.

Sampling is not exercised here beyond one tiny fit; what these pin is the reasoning around it, which
is where a hierarchical model actually goes wrong: reporting a fit that did not converge, ranking
on posterior means, or calling a verdict that the pre-registration leaves undecided.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO / "pipeline"))

pytest.importorskip("pymc")
import run_team_shrinkage as ts                                      # noqa: E402


def _teams(n_teams: int = 8, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    return pd.DataFrame({"league": ["A"] * (n_teams // 2) + ["B"] * (n_teams // 2),
                         "team": [f"t{i}" for i in range(n_teams)],
                         "corners_for": rng.integers(150, 260, n_teams),
                         "aft_goals_for": rng.integers(4, 20, n_teams)})


def test_v1_is_undecided_rather_than_false_when_nothing_converged():
    """The pre-registration says a null built on fits that did not converge is not a null. Reporting
    False there would turn 'we could not measure it' into 'there is no signal'."""
    case = {"a_team_hdi": [-0.1, 0.9], "p_top5": 0.61}
    assert ts.verdicts(case, p_perm=None)["V1_between_team_signal"] is None
    assert ts.verdicts(case, p_perm=0.04)["V1_between_team_signal"] is True
    assert ts.verdicts(case, p_perm=0.077)["V1_between_team_signal"] is False


def test_the_permutation_p_value_has_a_floor_and_the_filter_can_only_move_it_one_way():
    """The +1 on both sides means n permutations cannot give less than 1/(n+1), so a p-value equal
    to its own floor is 'as extreme as this null can say', not 'nearly significant'. And dropping
    the permutations that did not converge is not neutral: they tend to be the ones with the widest
    sigma, so removing them lowers the null and flatters the real value."""
    real = 0.26
    every = np.array([0.30, 0.28, 0.27, 0.20, 0.19, 0.18, 0.17, 0.16, 0.15])
    converged = every[3:]                       # the three widest are the ones that fail to mix
    assert ts.permutation_p(converged, real) == pytest.approx(1 / 7)
    assert ts.permutation_p(converged, real) == pytest.approx(1 / (len(converged) + 1))
    assert ts.permutation_p(every, real) == pytest.approx(4 / 10)
    assert ts.permutation_p(every, real) > ts.permutation_p(converged, real)
    assert ts.permutation_p(np.array([]), real) is None


def test_v2_fires_on_either_side_of_zero():
    """A team measurably WORSE than its league is as much a finding as one measurably better, so
    the rule is that the interval excludes zero, not that it sits above it."""
    assert ts.verdicts({"a_team_hdi": [0.2, 0.9], "p_top5": 0.0}, 0.01)["V2_case_distinguishable"]
    assert ts.verdicts({"a_team_hdi": [-0.9, -0.2], "p_top5": 0.0}, 0.01)["V2_case_distinguishable"]
    assert not ts.verdicts({"a_team_hdi": [-0.13, 0.91], "p_top5": 0.0},
                           0.01)["V2_case_distinguishable"]


def test_v3_needs_more_than_half_the_draws():
    """P(rank <= 5) > 0.5 and not >= : a coin flip is not confidence."""
    assert not ts.verdicts({"a_team_hdi": [-1.0, 1.0], "p_top5": 0.5}, 0.01)["V3_case_top5_confident"]
    assert ts.verdicts({"a_team_hdi": [-1.0, 1.0], "p_top5": 0.5001}, 0.01)["V3_case_top5_confident"]


def test_the_diagnostics_gate_reads_all_three_criteria():
    """One divergence is enough to refuse a fit, however good r_hat and ess look."""
    t = _teams()
    k, n = t["aft_goals_for"].to_numpy(int), t["corners_for"].to_numpy(int)
    lg = np.array([0] * 4 + [1] * 4)
    idata, diag = ts.fit(ts.build(k, n, lg, 2, -2.5, 0.5, 0.5), draws=60, tune=60, chains=2, seed=7)
    assert set(diag) == {"max_r_hat", "min_ess_bulk", "divergences", "converged"}
    assert diag["converged"] == (diag["max_r_hat"] < ts.R_HAT_MAX
                                 and diag["min_ess_bulk"] > ts.ESS_MIN
                                 and diag["divergences"] == 0)


def test_rank_is_summarised_from_the_draws_and_not_from_the_means():
    """Ranking the posterior means would hand back a ranking as confident-looking as the raw one.
    The rank distribution has to be built inside each draw, so a team with a wide posterior gets a
    wide rank interval even when its mean is high."""
    t = _teams()
    k, n = t["aft_goals_for"].to_numpy(int), t["corners_for"].to_numpy(int)
    lg = np.array([0] * 4 + [1] * 4)
    idata, _ = ts.fit(ts.build(k, n, lg, 2, -2.5, 0.5, 0.5), draws=200, tune=200, chains=2, seed=7)
    tab = ts.team_table(idata, t)
    assert (tab["rank_lo"] <= tab["rank_median"]).all() and (tab["rank_median"] <= tab["rank_hi"]).all()
    assert (tab["rank_hi"] - tab["rank_lo"]).max() > 1, "a rank with no spread is a ranking again"
    assert ((tab["p_lo"] <= tab["p_mean"]) & (tab["p_mean"] <= tab["p_hi"])).all()


@pytest.mark.parametrize("kind", [[], ["--kind", "free_kick_flank"]])
def test_another_team_table_may_not_write_the_record(monkeypatch, tmp_path, kind):
    """team_shrinkage[_kind].parquet is each population's record evidence.md §12 reads, so a per-team
    table of any other population or subset needs --out, and refuses before a model is built."""
    other = tmp_path / "eda_teams_subset.parquet"
    _teams().to_parquet(other)

    def work(*a, **k):
        raise AssertionError("the run built a model before refusing")

    monkeypatch.setattr(ts, "build", work)
    monkeypatch.setattr(sys, "argv", ["run_team_shrinkage.py", "--teams", str(other), *kind])
    with pytest.raises(SystemExit, match="may not write the published"):
        ts.main()


def test_the_flank_twin_reads_its_own_team_table(monkeypatch, tmp_path):
    """--kind free_kick_flank reads the per-team table eda_second_ball writes for that population."""
    _teams().to_parquet(tmp_path / "eda_teams_free_kick_flank.parquet")

    def built(*a, **k):
        raise SystemExit("model built")

    monkeypatch.setattr(ts, "_DER", tmp_path)
    monkeypatch.setattr(ts, "build", built)
    monkeypatch.setattr(sys, "argv", ["run_team_shrinkage.py", "--kind", "free_kick_flank"])
    with pytest.raises(SystemExit, match="model built"):
        ts.main()


def test_a_fit_that_did_not_converge_is_recorded_and_ends_the_run_cleanly(monkeypatch, tmp_path):
    """The pre-registration reports M1's non-convergence and no result: an outcome, not a failure.
    The run writes the record with its diagnostics, leaves no per-team table of an earlier fit
    behind for a reader to take as this one's, and exits cleanly, because a recipe stops on a failed
    stage and the flank's thinner per-team counts may well not converge."""
    src = tmp_path / "eda_teams_corner.parquet"
    _teams().to_parquet(src)
    out, stale = tmp_path / "team_shrinkage.parquet", tmp_path / "team_shrinkage_teams.parquet"
    pd.DataFrame({"team": ["from an earlier fit"]}).to_parquet(stale)
    monkeypatch.setattr(ts, "fit", lambda *a, **k: (None, {"max_r_hat": 1.2, "min_ess_bulk": 50.0,
                                                          "divergences": 3, "converged": False}))
    monkeypatch.setattr(sys, "argv", ["run_team_shrinkage.py", "--teams", str(src), "--out", str(out)])
    ts.main()
    rec = pd.read_parquet(out)
    assert rec.loc[rec["key"] == "M1/converged", "point"].tolist() == [0.0]
    assert not {"case", "verdict"} & set(rec["block"])
    assert pd.read_parquet(stale).empty


def test_a_population_with_no_goal_is_recorded_and_ends_the_run_cleanly(monkeypatch, tmp_path):
    """With no after-first-touch goal the pooled logit is minus infinity and the sampler cannot even
    start, which a thin subset of the flank free kicks reaches. There is nothing to shrink, so the
    run records the population's counts, builds no model, leaves no table of an earlier fit behind
    and exits cleanly."""
    src = tmp_path / "eda_teams_corner.parquet"
    _teams().assign(aft_goals_for=0).to_parquet(src)
    out, stale = tmp_path / "team_shrinkage.parquet", tmp_path / "team_shrinkage_perm.parquet"
    pd.DataFrame({"sigma_team_mean": [0.2]}).to_parquet(stale)

    def work(*a, **k):
        raise AssertionError("a model was built on a population with no goal")

    monkeypatch.setattr(ts, "build", work)
    monkeypatch.setattr(sys, "argv", ["run_team_shrinkage.py", "--teams", str(src), "--out", str(out)])
    ts.main()
    rec = pd.read_parquet(out)
    assert set(rec["block"]) == {"population"}
    assert rec.loc[rec["key"] == "goals", "n"].tolist() == [0]
    assert pd.read_parquet(stale).empty


def test_the_empirical_prior_scale_is_a_between_group_sd():
    """S2 anchors the prior in the data. It has to be the spread BETWEEN groups: pooling every
    group into one would give zero and silently turn the sensitivity into a point mass at the
    pooled rate."""
    k = np.array([10, 20, 30, 40])
    n = np.array([200, 200, 200, 200])
    spread = ts.empirical_logit_sd(k, n, np.array(["a", "b", "c", "d"]))
    collapsed = ts.empirical_logit_sd(k, n, np.array(["a", "a", "a", "a"]))
    assert spread > 0
    assert np.isnan(collapsed) or collapsed == 0
