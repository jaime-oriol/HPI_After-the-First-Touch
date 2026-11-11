"""Tests for run_frame_gpu, on the CPU and on frames small enough to fit in a second: the network
reads a set (body order and padding change nothing), the frame is canonical, a fit is reproducible,
and GPU1 is decided as registered.
"""
import sys
from collections.abc import Iterator
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

torch = pytest.importorskip("torch")       # the optional `gpu` group: without it the module is skipped, not an error

_REPO = Path(__file__).resolve().parents[1]
for p in (_REPO / "src", _REPO / "pipeline"):
    sys.path.insert(0, str(p))

import run_frame_gpu as G                                                  # noqa: E402


@pytest.fixture(autouse=True)
def _restore_determinism_flag() -> Iterator[None]:
    """gpu_task switches deterministic algorithms on for its whole process; put it back."""
    before = torch.are_deterministic_algorithms_enabled()
    yield
    torch.use_deterministic_algorithms(before)


def _net() -> G.FrameNet:
    torch.manual_seed(0)
    return G.FrameNet(d_design=3, k=5, phi_width=8, phi_depth=2, rho_width=8, dropout=0.0).eval()


def _frames(n: int = 4, p: int = 6, seed: int = 0) -> tuple:
    g = torch.Generator().manual_seed(seed)
    B = torch.rand((n, p, G._BODY), generator=g)
    B[..., 2], B[..., 3] = (torch.rand((n, p), generator=g) > 0.5).float(), 0.0
    return B, torch.ones((n, p), dtype=torch.bool), torch.rand((n, 3), generator=g)


def test_the_network_does_not_care_in_what_order_the_bodies_come():
    B, M, D = _frames()
    net = _net()
    with torch.no_grad():
        assert torch.allclose(net(B, M, D), net(B[:, torch.randperm(B.shape[1])], M, D), atol=1e-6)


def test_a_padded_body_changes_nothing_whatever_it_holds():
    B, M, D = _frames()
    pad = torch.cat([B, torch.full((B.shape[0], 1, G._BODY), 7.0)], dim=1)
    mask = torch.cat([M, torch.zeros((M.shape[0], 1), dtype=torch.bool)], dim=1)
    net = _net()
    with torch.no_grad():
        assert torch.allclose(net(B, M, D), net(pad, mask, D), atol=1e-6)


def test_a_side_with_no_body_pools_to_zero():
    B, M, D = _frames()
    B[..., 2] = 1.0                                     # every body an attacker: no defender at all
    net = _net()
    with torch.no_grad():
        out = net(B, M, D)
    assert torch.isfinite(out).all()


def _delivery(loc: tuple, players: list) -> tuple:
    d = pd.Series({"id": "k1", "location": list(loc)})
    fz = pd.DataFrame({"event_uuid": ["k1"], "freeze_frame": [[
        {"location": list(xy), "teammate": tm, "keeper": gk, "actor": ac}
        for xy, tm, gk, ac in players]]})
    return d, fz


def test_every_body_but_the_ball_is_read_in_the_canonical_frame():
    players = [((95.0, 72.0), True, False, True), ((110.0, 50.0), True, False, False),
               ((108.0, 45.0), False, False, False), ((119.0, 40.0), False, True, False)]
    right, b = G.frame_rows(*_delivery((95.0, 72.0), players))
    mirror = [((x, 80.0 - y), tm, gk, ac) for (x, y), tm, gk, ac in players]
    left, b_left = G.frame_rows(*_delivery((95.0, 8.0), mirror))
    assert right and not left and np.allclose(b, b_left)
    assert len(b) == 4                                  # taker and both keepers in, the ball out
    assert np.allclose(b[1], [110.0 / 120, 30.0 / 80, 1.0, 0.0])
    assert np.allclose(b[3], [119.0 / 120, 40.0 / 80, 0.0, 1.0])


def _task_data(n_matches: int = 12, per: int = 6, seed: int = 0) -> tuple:
    rng = np.random.default_rng(seed)
    n = n_matches * per
    B = rng.random((n, 5, G._BODY)).astype(np.float32)
    B[..., 2] = rng.random((n, 5)) > 0.5
    M = rng.random((n, 5)) > 0.2
    D = rng.random((n, 3))
    y = rng.integers(0, 3, n)
    groups = np.repeat(np.arange(n_matches), per)
    return B, M, D, y, groups


_P = {"phi_width": 8, "phi_depth": 1, "rho_width": 8, "dropout": 0.1, "weight_decay": 1e-3,
      "lr": 3e-3, "batch_size": 16}


def test_a_fit_reruns_bit_identical():
    B, M, D, y, _ = _task_data()
    data = (torch.as_tensor(B), torch.as_tensor(M), D)
    onehot = torch.nn.functional.one_hot(torch.as_tensor(y), 3).float()
    tr, te = np.arange(0, 48), np.arange(48, 72)
    a = G.fit_predict(_P, data, onehot, tr, te, seed=3)
    b = G.fit_predict(_P, data, onehot, tr, te, seed=3)
    assert np.array_equal(a[0], b[0]) and a[1] == b[1]
    assert np.allclose(a[0].sum(axis=1), 1.0) and a[1] <= G._MAX_EPOCHS


def test_a_task_searches_on_its_training_matches_and_scores_its_test_ones():
    B, M, D, y, g = _task_data()
    tr, te = np.flatnonzero(g < 9), np.flatnonzero(g >= 9)
    r = G.gpu_task(B, M, D, y, g, tr, te, 3, 2, 0, 0)
    assert r["prob"].shape == (len(te), 3) and np.allclose(r["prob"].sum(axis=1), 1.0)
    assert len(r["trials"]) == 2 and r["fixed"]["epochs"] >= 1
    again = G.gpu_task(B, M, D, y, g, tr, te, 3, 0, 0, 0, fixed=r["fixed"])
    assert np.array_equal(again["prob"], r["prob"])


def _stub_task(B: np.ndarray, M: np.ndarray, D: np.ndarray, y: np.ndarray, groups: np.ndarray,
               tr: np.ndarray, te: np.ndarray, k: int, n_trials: int, seed: int, fold: int,
               fixed: dict | None = None) -> dict:
    """A deterministic stand-in for gpu_task: what is under test is the resume, not the fit."""
    prob = np.full((len(te), k), 1.0 / k) + seed * 1e-3 + fold * 1e-4
    return {"fold": fold, "seed": seed, "te": te, "prob": prob, "fixed": fixed or {"epochs": 1},
            "trials": [], "seconds": 0.0, "peak_reserved_mb": 0.0}


def test_a_rerun_loads_every_finished_task_and_fits_only_the_missing_one(monkeypatch, tmp_path):
    """GPU1 runs for hours, so each task is written as it ends and a rerun after a crash loads what
    it finds: the same results in the same order, with only the lost task fitted again."""
    B, M, D, y, g = _task_data()
    folds = [(np.flatnonzero(g % 3 != f), np.flatnonzero(g % 3 == f)) for f in range(3)]
    tasks = [(s, f, tr, te, None) for s in range(2) for f, (tr, te) in enumerate(folds)]
    calls = []

    def counted(*a, **k):
        calls.append(a[9:11])
        return _stub_task(*a, **k)

    monkeypatch.setattr(G, "gpu_task", counted)
    first = G.run_tasks(B, M, D, y, g, tasks, 3, 2, 1, tmp_path)
    assert len(calls) == len(tasks) and len(list(tmp_path.glob("*.pkl"))) == len(tasks)
    sorted(tmp_path.glob("*.pkl"))[0].unlink()
    calls.clear()
    again = G.run_tasks(B, M, D, y, g, tasks, 3, 2, 1, tmp_path)
    assert len(calls) == 1
    for a, b in zip(first, again, strict=True):
        assert (a["seed"], a["fold"]) == (b["seed"], b["fold"])
        assert np.array_equal(a["prob"], b["prob"]) and np.array_equal(a["te"], b["te"])


def test_a_changed_input_never_reuses_an_old_task(monkeypatch, tmp_path):
    """The file is named by a digest of everything the task reads, so another trial budget or
    other frames miss it instead of loading a result that belongs to a different run."""
    B, M, D, y, g = _task_data()
    task = [(0, 0, np.flatnonzero(g < 9), np.flatnonzero(g >= 9), None)]
    monkeypatch.setattr(G, "gpu_task", _stub_task)
    G.run_tasks(B, M, D, y, g, task, 3, 2, 1, tmp_path)
    calls = []
    monkeypatch.setattr(G, "gpu_task", lambda *a, **k: calls.append(1) or _stub_task(*a, **k))
    G.run_tasks(B, M, D, y, g, task, 3, 2, 1, tmp_path)
    G.run_tasks(B, M, D, y, g, task, 3, 3, 1, tmp_path)
    G.run_tasks(B[::-1].copy(), M, D, y, g, task, 3, 2, 1, tmp_path)
    assert len(calls) == 2


def test_a_load_only_run_stops_on_a_missing_task_and_fits_nothing(monkeypatch, tmp_path):
    """The recipe loads the plan's sixty tasks and never trains: a task it cannot find stops the
    run naming how many are missing, and a complete checkpoint loads exactly what a fit wrote."""
    B, M, D, y, g = _task_data()
    folds = [(np.flatnonzero(g % 3 != f), np.flatnonzero(g % 3 == f)) for f in range(3)]
    tasks = [(s, f, tr, te, None) for s in range(2) for f, (tr, te) in enumerate(folds)]
    monkeypatch.setattr(G, "gpu_task", _stub_task)
    fitted = G.run_tasks(B, M, D, y, g, tasks, 3, 2, 1, tmp_path)
    calls = []
    monkeypatch.setattr(G, "gpu_task", lambda *a, **k: calls.append(1) or _stub_task(*a, **k))
    loaded = G.run_tasks(B, M, D, y, g, tasks, 3, 2, 1, tmp_path, load_only=True)
    assert not calls and all(np.array_equal(a["prob"], b["prob"]) for a, b in zip(fitted, loaded,
                                                                                    strict=True))
    sorted(tmp_path.glob("*.pkl"))[0].unlink()
    with pytest.raises(SystemExit, match="1 of 6 tasks"):
        G.run_tasks(B, M, D, y, g, tasks, 3, 2, 1, tmp_path, load_only=True)
    assert not calls


@pytest.mark.parametrize("out_dir", ["data/derived", "data/derived/zones_F",
                                     "data/derived/auditoria_2026-10-12/gpu"])
def test_every_published_name_needs_the_registered_budget(monkeypatch, out_dir):
    """The recipe publishes GPU1 under data/derived and its F pass under zones_F, beside the plan's
    own record: a run off the registered budget may write none of the three."""
    def work(*a, **k):
        raise AssertionError("the run did the work before refusing")

    monkeypatch.setattr(G, "population", work)
    monkeypatch.setattr(sys, "argv", ["run_frame_gpu.py", "--trials", "5",
                                      "--out-dir", str(_REPO / out_dir)])
    with pytest.raises(SystemExit, match="may not write"):
        G.main()


def test_gpu1_needs_every_seed_below_the_margin_and_a_frame_shuffle_that_does_not_win():
    below = {"below_margin": True}
    seeds = {s: below for s in range(5)}
    assert G.decide(seeds, {"hi": 0.001})["enters"]
    assert not G.decide({**seeds, 3: {"below_margin": False}}, {"hi": 0.001})["enters"]
    assert not G.decide(seeds, {"hi": -0.0005})["enters"]
    assert not G.decide({}, {"hi": 0.001})["enters"]


def test_the_margin_is_the_sheets():
    assert G.MARGIN == 0.00331


def _table(n_matches: int = 15, per: int = 10, seed: int = 0) -> pd.DataFrame:
    """A modelling table carrying every column the delivery rung and the full model read."""
    from s03_value import TARGET_MAP, _NUMERIC
    rng = np.random.default_rng(seed)
    classes = sorted(set(TARGET_MAP.values()))
    n = n_matches * per
    t = pd.DataFrame({c: rng.normal(size=n) for c in _NUMERIC})
    t["match_id"] = np.repeat(np.arange(n_matches), per)
    t["target"] = [classes[i % len(classes)] for i in range(n)]
    t["swing"] = rng.choice(["Inswinging", "Outswinging"], n)
    t["corner_side"] = rng.choice(["left", "right"], n)
    t["short_corner"] = False
    return t


def test_the_scored_path_runs_end_to_end(monkeypatch, tmp_path):
    """The path the smoke never takes: every seed's tasks, the tuned delivery rung and the full
    model on the same folds, the shuffled frame refitted at seed 0's configurations, and the
    verdict, every interval finite and around its point, written to parquet as `main` writes it."""
    import argparse
    monkeypatch.setattr(G, "card_state", lambda: "test")
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    t = _table()
    rng = np.random.default_rng(1)
    B = rng.random((len(t), 5, G._BODY)).astype(np.float32)
    B[..., 2] = rng.random((len(t), 5)) > 0.5
    M = rng.random((len(t), 5)) > 0.2
    D = G.CompetingOutcomesModel()._design(t, numeric=G._KICK_INTRINSIC)[0]
    classes = sorted(t["target"].unique())
    y = t["target"].map({c: i for i, c in enumerate(classes)}).to_numpy()
    args = argparse.Namespace(seeds=1, trials=1, jobs=1, boot=50)
    rows = pd.DataFrame(G.evaluate("corner", (t, B, M, D, y, len(classes)), args))
    assert set(rows["block"]) == {"seed", "control", "head_to_head", "verdict", "task"}
    assert (rows["block"] == "task").sum() == G._N_FOLDS
    for block in ("seed", "control", "head_to_head"):
        r = rows[rows["block"] == block].iloc[0]
        assert np.isfinite([r["point"], r["lo"], r["hi"]]).all() and r["lo"] <= r["point"] <= r["hi"]
    assert "enters" in rows.loc[rows["block"] == "verdict", "params"].iloc[0]
    rows.to_parquet(tmp_path / "frame_gpu.parquet", index=False)
    assert set(pd.read_parquet(tmp_path / "frame_gpu.parquet")["key"]) >= {"0", "GPU1"}


def test_other_tables_may_not_write_the_record(monkeypatch, tmp_path):
    """Tables outside data/derived are not the published ones, so a run on them would put its
    readings under the names the record is read from."""
    def work(*a, **k):
        raise AssertionError("the run did the work before refusing")

    monkeypatch.setattr(G, "population", work)
    monkeypatch.setattr(sys, "argv", ["run_frame_gpu.py", "--tables", str(tmp_path),
                                      "--out-dir", str(_REPO / "data" / "derived")])
    with pytest.raises(SystemExit, match="not the published ones"):
        G.main()


def test_another_budget_may_not_write_the_record(monkeypatch):
    def work(*a, **k):
        raise AssertionError("the run did the work before refusing")

    monkeypatch.setattr(G, "population", work)
    monkeypatch.setattr(sys, "argv", ["run_frame_gpu.py", "--trials", "5"])
    with pytest.raises(SystemExit, match="may not write"):
        G.main()
