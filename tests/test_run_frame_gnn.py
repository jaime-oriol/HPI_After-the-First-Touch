"""Tests for run_frame_gnn, on the CPU and on frames small enough to fit in a second: the network
reads a set and a graph (body order and padding change nothing), its layer is torch_geometric's
GATv2Conv written dense, the frame is canonical and flags the taker, nothing after the kick enters,
a fit is reproducible, the tasks resume and load as GPU1's do, the search never sees a test match,
and GNN1, P1 and the better of the two are decided as registered.
"""
import argparse
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

import run_frame_gnn as N                                                  # noqa: E402


@pytest.fixture(autouse=True)
def _restore_torch_state() -> Iterator[None]:
    """gnn_task switches deterministic algorithms on and torch to one thread for its whole
    process; put both back."""
    det, threads = torch.are_deterministic_algorithms_enabled(), torch.get_num_threads()
    yield
    torch.use_deterministic_algorithms(det)
    torch.set_num_threads(threads)


def _net(d_design: int = 3) -> N.GraphNet:
    torch.manual_seed(0)
    return N.GraphNet(d_design=d_design, k=5, layers=2, rho_width=8, dropout=0.0).eval()


def _frames(n: int = 4, p: int = 6, seed: int = 0) -> tuple:
    g = torch.Generator().manual_seed(seed)
    B = torch.rand((n, p, N._NODE), generator=g)
    B[..., 2] = (torch.rand((n, p), generator=g) > 0.5).float()
    B[..., 3:] = 0.0
    B[:, 0, 2], B[:, 0, 4] = 1.0, 1.0                     # the taker, an attacker
    return B, torch.ones((n, p), dtype=torch.bool), torch.rand((n, 3), generator=g)


# ── The network ──────────────────────────────────────────────────────────
def test_the_network_does_not_care_in_what_order_the_bodies_come():
    B, M, D = _frames()
    net = _net()
    with torch.no_grad():
        assert torch.allclose(net(B, M, D), net(B[:, torch.randperm(B.shape[1])], M, D), atol=1e-5)


def test_a_padded_body_changes_nothing_whatever_it_holds():
    B, M, D = _frames()
    pad = torch.cat([B, torch.full((B.shape[0], 1, N._NODE), 7.0)], dim=1)
    mask = torch.cat([M, torch.zeros((M.shape[0], 1), dtype=torch.bool)], dim=1)
    net = _net()
    with torch.no_grad():
        assert torch.allclose(net(B, M, D), net(pad, mask, D), atol=1e-5)


def test_a_side_with_no_body_pools_to_zero():
    B, M, D = _frames()
    B[..., 2] = 1.0                                     # every body an attacker: no defender at all
    with torch.no_grad():
        assert torch.isfinite(_net()(B, M, D)).all()


def test_the_frame_alone_reads_without_a_design():
    B, M, _ = _frames()
    with torch.no_grad():
        out = _net(d_design=0)(B, M, torch.zeros((B.shape[0], 0)))
    assert out.shape == (B.shape[0], 5) and torch.isfinite(out).all()


def test_the_dense_layer_is_torch_geometrics_gatv2_on_a_complete_graph():
    """The same weights in both, every ordered pair an edge with the self-pairs, each carrying the
    same-side feature: GATv2Conv and the dense layer give every body the same output."""
    GATv2Conv = pytest.importorskip("torch_geometric.nn").GATv2Conv      # the reference, needed by this test only
    torch.manual_seed(0)
    n, d = 7, N._WIDTH
    x = torch.randn(n, d, dtype=torch.float64)
    side = torch.tensor([1.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0], dtype=torch.float64)
    ref = GATv2Conv(d, N._HEAD_DIM, heads=N._HEADS, edge_dim=1, add_self_loops=False).double()
    dense = N.DenseGATv2(d, N._HEADS, N._HEAD_DIM).double()
    with torch.no_grad():
        for mine, theirs in ((dense.lin_l, ref.lin_l), (dense.lin_r, ref.lin_r)):
            mine.weight.copy_(theirs.weight)
            mine.bias.copy_(theirs.bias)
        dense.lin_e.weight.copy_(ref.lin_edge.weight)
        dense.att.copy_(ref.att[0])
        dense.bias.copy_(ref.bias)
    i, j = (a.reshape(-1) for a in torch.meshgrid(torch.arange(n), torch.arange(n), indexing="ij"))
    same = (side[:, None] == side[None, :]).double()
    with torch.no_grad():
        want = ref(x, torch.stack([j, i]), same[i, j].unsqueeze(-1))     # source j, target i
        got = dense(x[None], torch.ones((1, n), dtype=torch.bool), same[None])[0]
    assert torch.allclose(got, want, atol=1e-10)


# ── The frames ───────────────────────────────────────────────────────────
def _delivery(loc: tuple, players: list, end: list | None = None) -> tuple:
    d = pd.Series({"id": "k1", "location": list(loc), "pass.end_location": end})
    fz = pd.DataFrame({"event_uuid": ["k1"], "freeze_frame": [[
        {"location": list(xy), "teammate": tm, "keeper": gk, "actor": ac}
        for xy, tm, gk, ac in players]]})
    return d, fz


_PLAYERS = [((119.0, 79.0), True, False, True), ((110.0, 50.0), True, False, False),
            ((108.0, 45.0), False, False, False), ((119.0, 40.0), False, True, False)]


def test_every_body_but_the_ball_is_read_in_the_canonical_frame_with_the_taker_flagged():
    right, b = N.frame_rows(*_delivery((119.0, 79.0), _PLAYERS))
    mirror = [((x, 80.0 - y), tm, gk, ac) for (x, y), tm, gk, ac in _PLAYERS]
    left, b_left = N.frame_rows(*_delivery((119.0, 1.0), mirror))
    assert right and not left and np.allclose(b, b_left)
    assert b.shape == (4, N._NODE)                      # taker and both keepers in, the ball out
    assert np.allclose(b[0], [119.0 / 120, 1.0 / 80, 1.0, 0.0, 1.0])
    assert np.allclose(b[3], [119.0 / 120, 40.0 / 80, 0.0, 1.0, 0.0])


def test_a_frame_flags_one_taker_and_none_when_the_actor_is_off_camera():
    _, a = N.frame_rows(*_delivery((119.0, 79.0), _PLAYERS[:2]))
    _, b = N.frame_rows(*_delivery((119.0, 79.0), _PLAYERS[1:3]))
    B = np.stack([a, b]).astype(np.float32)
    assert N.taker_flags(B, np.ones((2, 2), bool)) == {"one": 0.5, "none": 0.5, "more": 0.0}


def test_the_end_location_changes_no_input():
    """pass.end_location is where the defence got to the ball on most deliveries (evidence.md §15):
    moving it moves nothing the network reads."""
    _, a = N.frame_rows(*_delivery((119.0, 79.0), _PLAYERS, end=[110.0, 40.0]))
    _, b = N.frame_rows(*_delivery((119.0, 79.0), _PLAYERS, end=[96.0, 70.0]))
    assert np.array_equal(a, b)


def test_the_design_reads_no_field_that_reads_the_future():
    from s03_value import _BARRED_SOURCES, _COLUMN_SOURCE, _DELIVERY_CAT, _SOURCE_EXCEPTIONS
    cols = list(N._KICK_INTRINSIC) + _DELIVERY_CAT + ["short_corner"]
    assert not [c for c in cols if c not in _SOURCE_EXCEPTIONS
                and any(b in _COLUMN_SOURCE[c] for b in _BARRED_SOURCES)]


# ── The fit and the tasks ────────────────────────────────────────────────
def _task_data(n_matches: int = 12, per: int = 6, seed: int = 0) -> tuple:
    rng = np.random.default_rng(seed)
    n = n_matches * per
    B = rng.random((n, 5, N._NODE)).astype(np.float32)
    B[..., 2] = rng.random((n, 5)) > 0.5
    B[..., 3:] = 0.0
    M = rng.random((n, 5)) > 0.2
    D = rng.random((n, 3))
    y = rng.integers(0, 3, n)
    return B, M, D, y, np.repeat(np.arange(n_matches), per)


_P = {"layers": 2, "rho_width": 8, "dropout": 0.1, "weight_decay": 1e-3, "lr": 3e-3,
      "batch_size": 16}


def _fit_inputs() -> tuple:
    B, M, D, y, _ = _task_data()
    return (torch.as_tensor(B), torch.as_tensor(M), D), torch.nn.functional.one_hot(
        torch.as_tensor(y), 3).float()


def test_a_fit_reruns_bit_identical():
    data, onehot = _fit_inputs()
    tr, te = np.arange(0, 48), np.arange(48, 72)
    a = N.fit_predict(_P, data, onehot, tr, te, seed=3, max_steps=90)
    b = N.fit_predict(_P, data, onehot, tr, te, seed=3, max_steps=90)
    assert np.array_equal(a[0], b[0]) and a[1] == b[1]
    assert np.allclose(a[0].sum(axis=1), 1.0) and a[1] <= 30


def test_the_cap_is_tacticais_steps_over_the_batches_of_an_epoch_and_a_refit_never_hits_it():
    """48 rows in batches of 16 are 3 steps an epoch: 7 steps allow 2 epochs."""
    assert N._MAX_STEPS == 50_000
    data, onehot = _fit_inputs()
    tr, te = np.arange(0, 48), np.arange(48, 72)
    searched = N.fit_predict(_P, data, onehot, tr, te, seed=3, max_steps=7)
    refit = N.fit_predict(_P, data, onehot, tr, te, seed=3, max_steps=7, epochs=4)
    assert searched[3] and searched[1] <= 2
    assert not refit[3] and refit[1] == 4


def test_a_task_searches_on_its_training_matches_and_scores_its_test_ones():
    B, M, D, y, g = _task_data()
    tr, te = np.flatnonzero(g < 9), np.flatnonzero(g >= 9)
    r = N.gnn_task(B, M, D, y, g, tr, te, 3, 2, 0, 0, max_steps=15)
    assert r["prob"].shape == (len(te), 3) and np.allclose(r["prob"].sum(axis=1), 1.0)
    assert len(r["trials"]) == 2 and r["fixed"]["epochs"] >= 1 and r["peak_rss_mb"] > 0
    again = N.gnn_task(B, M, D, y, g, tr, te, 3, 0, 0, 0, fixed=r["fixed"], max_steps=15)
    assert np.array_equal(again["prob"], r["prob"])


def test_the_search_never_sees_a_test_match(monkeypatch):
    B, M, D, y, g = _task_data()
    tr, te = np.flatnonzero(g < 9), np.flatnonzero(g >= 9)
    real, seen = N.fit_predict, []

    def spy(p, data, onehot, a, b, seed, max_steps, epochs=None):
        seen.append((a, b, epochs))
        return real(p, data, onehot, a, b, seed, max_steps, epochs)

    monkeypatch.setattr(N, "fit_predict", spy)
    N.gnn_task(B, M, D, y, g, tr, te, 3, 2, 0, 0, max_steps=9)
    search = [(a, b) for a, b, e in seen if e is None]
    assert len(search) == 2 * N.G._N_INNER
    for a, b in search:
        assert set(g[a]).isdisjoint(g[te]) and set(g[b]).isdisjoint(g[te])
        assert set(g[a]).isdisjoint(g[b])


def _stub_task(B: np.ndarray, M: np.ndarray, D: np.ndarray, y: np.ndarray, groups: np.ndarray,
               tr: np.ndarray, te: np.ndarray, k: int, n_trials: int, seed: int, fold: int,
               fixed: dict | None = None, max_steps: int = N._MAX_STEPS) -> dict:
    """A deterministic stand-in for gnn_task: what is under test is the resume, not the fit."""
    prob = np.full((len(te), k), 1.0 / k) + seed * 1e-3 + fold * 1e-4
    return {"fold": fold, "seed": seed, "te": te, "prob": prob, "fixed": fixed or {"epochs": 1},
            "trials": [], "seconds": 0.0, "peak_reserved_mb": 0.0, "peak_rss_mb": 0.0}


def _folds_tasks(g: np.ndarray) -> list:
    folds = [(np.flatnonzero(g % 3 != f), np.flatnonzero(g % 3 == f)) for f in range(3)]
    return [(s, f, tr, te, None) for s in range(2) for f, (tr, te) in enumerate(folds)]


def test_a_rerun_loads_every_finished_task_and_fits_only_the_missing_one(monkeypatch, tmp_path):
    B, M, D, y, g = _task_data()
    tasks, calls = _folds_tasks(g), []

    def counted(*a, **k):
        calls.append(a[9:11])
        return _stub_task(*a, **k)

    monkeypatch.setattr(N, "gnn_task", counted)
    first = N.run_tasks(B, M, D, y, g, tasks, 3, 2, 1, tmp_path)
    assert len(calls) == len(tasks) and len(list(tmp_path.glob("*.pkl"))) == len(tasks)
    sorted(tmp_path.glob("*.pkl"))[0].unlink()
    calls.clear()
    again = N.run_tasks(B, M, D, y, g, tasks, 3, 2, 1, tmp_path)
    assert len(calls) == 1
    for a, b in zip(first, again, strict=True):
        assert (a["seed"], a["fold"]) == (b["seed"], b["fold"])
        assert np.array_equal(a["prob"], b["prob"]) and np.array_equal(a["te"], b["te"])


def test_a_changed_input_never_reuses_an_old_task(monkeypatch, tmp_path):
    """The file is named by a digest of everything the task reads, the epoch cap included, so
    another budget, cap or frame misses it instead of loading a result from a different run."""
    B, M, D, y, g = _task_data()
    task = [(0, 0, np.flatnonzero(g < 9), np.flatnonzero(g >= 9), None)]
    monkeypatch.setattr(N, "gnn_task", _stub_task)
    N.run_tasks(B, M, D, y, g, task, 3, 2, 1, tmp_path)
    calls = []
    monkeypatch.setattr(N, "gnn_task", lambda *a, **k: calls.append(1) or _stub_task(*a, **k))
    N.run_tasks(B, M, D, y, g, task, 3, 2, 1, tmp_path)
    N.run_tasks(B, M, D, y, g, task, 3, 3, 1, tmp_path)
    N.run_tasks(B, M, D, y, g, task, 3, 2, 1, tmp_path, max_steps=N._MAX_STEPS + 1)
    N.run_tasks(B[::-1].copy(), M, D, y, g, task, 3, 2, 1, tmp_path)
    assert len(calls) == 3


def test_a_load_only_run_stops_on_a_missing_task_and_fits_nothing(monkeypatch, tmp_path):
    B, M, D, y, g = _task_data()
    tasks = _folds_tasks(g)
    monkeypatch.setattr(N, "gnn_task", _stub_task)
    fitted = N.run_tasks(B, M, D, y, g, tasks, 3, 2, 1, tmp_path)
    calls = []
    monkeypatch.setattr(N, "gnn_task", lambda *a, **k: calls.append(1) or _stub_task(*a, **k))
    loaded = N.run_tasks(B, M, D, y, g, tasks, 3, 2, 1, tmp_path, load_only=True)
    assert not calls and all(np.array_equal(a["prob"], b["prob"])
                             for a, b in zip(fitted, loaded, strict=True))
    sorted(tmp_path.glob("*.pkl"))[0].unlink()
    with pytest.raises(SystemExit, match="1 of 6 tasks"):
        N.run_tasks(B, M, D, y, g, tasks, 3, 2, 1, tmp_path, load_only=True)
    assert not calls


# ── The guards ───────────────────────────────────────────────────────────
def _refuses(monkeypatch, argv: list, match: str) -> None:
    def work(*a, **k):
        raise AssertionError("the run did the work before refusing")

    monkeypatch.setattr(N, "population", work)
    monkeypatch.setattr(sys, "argv", ["run_frame_gnn.py", *argv])
    with pytest.raises(SystemExit, match=match):
        N.main()


@pytest.mark.parametrize("out_dir", ["data/derived", "data/derived/zones_F",
                                     "data/derived/auditoria_2026-09-30/gnn"])
@pytest.mark.parametrize("off", [["--trials", "5"], ["--control-trials", "5"], ["--seeds", "2"],
                                 ["--max-steps", "5000"]])
def test_every_published_name_needs_the_registered_budget(monkeypatch, out_dir, off):
    _refuses(monkeypatch, [*off, "--out-dir", str(_REPO / out_dir)], "may not write")


def test_the_published_record_needs_gpu1s_tasks(monkeypatch):
    _refuses(monkeypatch, ["--gpu-tasks", "", "--out-dir",
                           str(_REPO / "data/derived/auditoria_2026-09-30/gnn")], "--gpu-tasks")


def test_other_tables_may_not_write_the_record(monkeypatch, tmp_path):
    _refuses(monkeypatch, ["--tables", str(tmp_path), "--out-dir", str(_REPO / "data" / "derived")],
             "not the published ones")


# ── The decisions ────────────────────────────────────────────────────────
def test_gnn1_is_decided_by_gpu1s_rule_and_margin():
    assert N.MARGIN == 0.00331 and N.G.decide is not None
    below = {"below_margin": True}
    assert N.G.decide({s: below for s in range(5)}, {"hi": 0.001})["enters"]
    assert not N.G.decide({s: below for s in range(5)}, {"hi": -0.0005})["enters"]


def test_p1_needs_the_frame_below_zero_and_a_clean_shuffled_target():
    assert N.decide_p1({"hi": -0.001}, {"hi": 0.002})["P1"]
    assert not N.decide_p1({"hi": 0.001}, {"hi": 0.002})["P1"]
    assert not N.decide_p1({"hi": -0.001}, {"hi": -0.0001})["P1"]


def test_the_better_of_the_two_is_the_gate_first_then_the_mean_and_a_tie_keeps_gpu1():
    b = N.better_of_two
    assert b({"enters": True, "mean": -0.001}, {"enters": False, "mean": -0.01})["chosen"] == "GNN1"
    assert b({"enters": False, "mean": -0.01}, {"enters": True, "mean": -0.001})["chosen"] == "GPU1"
    assert b({"enters": False, "mean": -0.004}, {"enters": False, "mean": -0.002})["chosen"] == "GNN1"
    assert b({"enters": True, "mean": -0.002}, {"enters": True, "mean": -0.004})["chosen"] == "GPU1"
    assert b({"enters": False, "mean": -0.0020001},
             {"enters": False, "mean": -0.0020004})["chosen"] == "GPU1"


def test_the_control_reads_long_corners_with_a_first_contact_and_every_flank_free_kick():
    t = pd.DataFrame({"set_piece_id": list("abcde"), "match_id": [2, 1, 1, 3, 2],
                      "short_corner": [False, True, False, False, False]})
    grain = pd.DataFrame({"set_piece_id": list("abcde"),
                          "won_first_contact": [True, True, None, False, True]})
    assert list(N.control_rows(t, grain, "corner")["set_piece_id"]) == ["a", "e", "d"]
    assert list(N.control_rows(t, grain, "free_kick_flank")["set_piece_id"]) == ["b", "a", "e", "d"]


def test_the_grain_is_the_plans_file_for_the_population_or_the_joint_one_cut_to_it(tmp_path):
    joint = pd.DataFrame({"set_piece_id": ["a", "b"], "set_piece_type": ["corner", "free_kick_flank"],
                          "won_first_contact": [True, False]})
    joint.to_parquet(tmp_path / "eda_deliveries.parquet")
    assert list(N.grain(tmp_path, "free_kick_flank")["set_piece_id"]) == ["b"]
    joint.iloc[:1].assign(set_piece_id="z").to_parquet(tmp_path / "eda_deliveries_corner.parquet")
    assert list(N.grain(tmp_path, "corner")["set_piece_id"]) == ["z"]


def _table(n_matches: int = 15, per: int = 10, seed: int = 0) -> pd.DataFrame:
    """A modelling table carrying every column the delivery rung, the full model and the control
    read."""
    from s03_value import TARGET_MAP, _NUMERIC
    rng = np.random.default_rng(seed)
    classes = sorted(set(TARGET_MAP.values()))
    n = n_matches * per
    t = pd.DataFrame({c: rng.normal(size=n) for c in _NUMERIC})
    t["match_id"] = np.repeat(np.arange(n_matches), per)
    t["set_piece_id"] = [f"k{i}" for i in range(n)]
    t["target"] = [classes[i % len(classes)] for i in range(n)]
    t["swing"] = rng.choice(["Inswinging", "Outswinging"], n)
    t["corner_side"] = rng.choice(["left", "right"], n)
    t["short_corner"] = rng.random(n) < 0.1
    return t


def test_the_scored_path_runs_end_to_end(monkeypatch, tmp_path):
    """The path the smoke never takes: every seed's tasks, the tuned delivery rung and the full
    model on the same folds, the shuffled-frame twin at seed 0's configurations, the positive
    control on the long corners and the verdicts, every interval finite and around its point,
    written to parquet as `main` writes it."""
    monkeypatch.setattr(N.G, "card_state", lambda: "test")
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    t, rng = _table(), np.random.default_rng(1)
    B = rng.random((len(t), 5, N._NODE)).astype(np.float32)
    B[..., 2] = rng.random((len(t), 5)) > 0.5
    B[..., 3:] = 0.0
    M = rng.random((len(t), 5)) > 0.2
    D = N.CompetingOutcomesModel()._design(t, numeric=N._KICK_INTRINSIC)[0]
    classes = sorted(t["target"].unique())
    y = t["target"].map({c: i for i, c in enumerate(classes)}).to_numpy()
    pd.DataFrame({"set_piece_id": t["set_piece_id"], "won_first_contact": rng.random(len(t)) < 0.4}
                 ).to_parquet(tmp_path / "eda_deliveries_corner.parquet")
    args = argparse.Namespace(seeds=1, trials=1, control_trials=1, jobs=1, boot=50, max_steps=3,
                              tables=str(tmp_path))
    data = (t, B, M, D, y, len(classes))
    rows, ev = N.evaluate("corner", data, args, None, False)
    rows = pd.DataFrame(rows + N.positive_control("corner", data, args, None, False))
    assert {"seed", "control", "twin", "head_to_head", "verdict", "task",
            "positive_control"} <= set(rows["block"])
    assert (rows["block"] == "task").sum() == N.G._N_FOLDS
    for block in ("seed", "control", "twin", "head_to_head", "positive_control"):
        r = rows[rows["block"] == block].iloc[0]
        assert np.isfinite([r["point"], r["lo"], r["hi"]]).all() and r["lo"] <= r["point"] <= r["hi"]
    assert set(rows.loc[rows["block"] == "verdict", "key"]) == {"GNN1", "P1"}
    assert ev["verdict"]["enters"] in (True, False)
    rows.to_parquet(tmp_path / "frame_gnn.parquet", index=False)
