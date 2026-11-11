"""run_frame_gnn — do bodies that see each other at the kick add to the delivery design?

GPU1 (run_frame_gpu) reads the delivery frame body by body and pools each side, the Deep Sets
baseline of TacticAI's own ablation (Wang et al. 2024), and it enters neither population. TacticAI's
encoder lets every body attend to every other by graph attention (GATv2, Brody et al. 2022). This
asks whether that adds to the delivery where bodies read one at a time did not, and picks the
paper's one neural test between the two networks. Pre-registered
(data/derived/prereg_2026-10-07/gnn.md) before any graph network was trained.

THE ARM. Every detected body of the delivery 360 in the canonical frame as (x / 120, y / 80,
attacker, keeper, taker), the taker being the frame's actor, on the complete graph of the frame's
bodies with one edge feature, same side or not. Each body is embedded to width 32 and passed through
L GATv2 layers of eight heads of four, each with a residual connection; GPU1's readout and head then
pool each side by masked mean and max, join s03's delivery-rung design and read the five classes of
s03's target, so the arm and GPU1 differ in the message passing and the taker's flag alone. Optuna
searches it as GPU1 is searched, nested by match with sixty trials per task and five seeds, over a
space that holds TacticAI's grid; a fit may take TacticAI's 50,000 training steps and stops twenty
epochs after its best. The attention is dense on padded frames: on a complete graph that is the
layer itself, it needs no scatter, and a rerun on the card is bit-identical.

GNN1, THE DECISION, is GPU1's: the paired per-row OOF log loss, arm minus delivery rung, has its
match-clustered 95% interval wholly below −MARGIN at every one of five seeds, and the frame shuffled
between deliveries does not beat the delivery.

ITS POSITIVE CONTROL. The same network on the frame alone, without the design, predicts who wins the
first contact on the long corners and on every flank free kick. P1, the frame beats the prior with
the globally shuffled target clean, decides whether GNN1's null may be published: the design alone
beats the prior on that target, so an arm carrying it would pass with a network that read nothing.

THE BETTER OF THE TWO, decided on corners: a network that passes its gate over one that does not,
then the lower mean of the statistic over the five seeds, and a tie keeps GPU1. GPU1 is read from
its tasks and never refitted. Reported, never deciding: the frame's own share (the arm against its
shuffled-frame twin), the arm against GPU1 and against the full handcrafted model, the MDE at 80%,
and the time, VRAM and host RSS of every task; flank free kicks replicate.

    python pipeline/run_frame_gnn.py --smoke --trials 4 --jobs 6     # cost, VRAM, RSS, determinism; no score
    python pipeline/run_frame_gnn.py --jobs 6
    python pipeline/run_frame_gnn.py --tables data/derived --out-dir data/derived --load-only \\
        --tasks data/derived/auditoria_2026-09-30/gnn/tasks --jobs 8        # the recipe's stage
"""
from __future__ import annotations

import argparse
import os
import pickle
import resource
import sys
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd
import torch
from joblib import Parallel, delayed, parallel_config
from sklearn.model_selection import GroupKFold
from torch import nn

if TYPE_CHECKING:
    import optuna

_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO / "src"))
sys.path.insert(0, str(_REPO / "pipeline"))
_DER = _REPO / "data" / "derived"
_PLAN = _DER / "auditoria_2026-09-30"

import loader_sb as L                                                          # noqa: E402
import run_frame_gpu as G                                                      # noqa: E402
from build_table import published_dir                                         # noqa: E402
from constants import PITCH_X_SB, PITCH_Y_SB                                  # noqa: E402
from run_positive_control import oof_proba                                    # noqa: E402
from run_validation import _tuned                                             # noqa: E402
from s01_phases import SetPieceGrammar                                        # noqa: E402
from s02_signature import build_frame                                         # noqa: E402
from s03_value import _KICK_INTRINSIC, CompetingOutcomesModel                 # noqa: E402
from validate import assert_no_group_leak, cluster_bootstrap_mean, shuffle_labels  # noqa: E402
from zones import canonical_y, kicked_from_right                              # noqa: E402

POPULATIONS = G.POPULATIONS
MARGIN = G.MARGIN
_NODE = 5                          # x / 120, y / 80, attacker, keeper, taker
# TacticAI's eight heads, four features each as Groom et al. reimplement them, concatenated so the
# residual keeps one width through every layer; its message-passing steps are the searched depth.
_HEADS, _HEAD_DIM = 8, 4
_WIDTH = _HEADS * _HEAD_DIM
_LAYERS = (1, 2, 4)
# GPU1's budget and patience, so the two networks are searched alike (gnn.md). Each fit may take
# TacticAI's 50,000 training steps (Wang et al. 2024, Methods), whatever its batch, and stops 20 epochs
# after its best: fixed at the smoke's stop, where GPU1's cap of 300 epochs stopped a tenth of the fits.
_TRIALS, _SEEDS, _BOOT, _PATIENCE = G._GPU_TRIALS, G._SEEDS, G._BOOT, G._PATIENCE
_MAX_STEPS = 50_000
_CONTROL_TRIALS = _TRIALS
_CHUNK = 1024


# ── The frames ───────────────────────────────────────────────────────────
def frame_rows(d: pd.Series, fz: pd.DataFrame) -> tuple[bool, np.ndarray]:
    """One delivery's detected bodies, ball out, mirrored onto the y = 0 side when the kick came
    from the attackers' right: (x / 120, y / 80, attacker, keeper, taker) per body. It reads the
    freeze-frame and the kick spot; the delivery target build_frame also returns is dropped."""
    f, _ = build_frame(d, fz)
    right = bool(kicked_from_right(f.loc[f["is_ball"] == 1, "y"].iloc[0]))
    b = f[f["is_ball"] == 0]
    y = canonical_y(b["y"].to_numpy(float), right)
    return right, np.c_[b["x"].to_numpy(float) / PITCH_X_SB, y / PITCH_Y_SB,
                        b[["team_id", "is_goalkeeper", "is_actor"]].to_numpy(float)]


def match_frames(mid: int, kind: str, want: set) -> list:
    """(set_piece_id, right, bodies) for one match's wanted deliveries with a 360."""
    ev, fz = L.load_events(mid), L.load_360(mid)
    have = set(fz["event_uuid"])
    return [(c["id"], *frame_rows(c, fz))
            for _, c in SetPieceGrammar(ev, kinds=(kind,)).deliveries().iterrows()
            if c["id"] in want and c["id"] in have]


def build(table: pd.DataFrame, kind: str, jobs: int) -> tuple[np.ndarray, np.ndarray]:
    """(bodies, mask) row-aligned to `table`: (N, P, 5) zero-padded and (N, P) true on a body.
    Matches go in sorted order and the parent assembles one array, so the parallel build is the
    serial one. The run stops if a row has no frame or its kick side disagrees with s01's."""
    mids = sorted(table["match_id"].unique())
    want = table.groupby("match_id")["set_piece_id"].agg(set).to_dict()
    chunks = ([match_frames(int(m), kind, want[m]) for m in mids] if jobs == 1 else
              Parallel(n_jobs=jobs)(delayed(match_frames)(int(m), kind, want[m]) for m in mids))
    got = {sid: (right, b) for ch in chunks for sid, right, b in ch}
    ids = table["set_piece_id"].to_numpy()
    if missing := [s for s in ids if s not in got]:
        raise SystemExit(f"STOP: {len(missing)} rows have no frame, first {missing[:3]}")
    if not np.array_equal([got[s][0] for s in ids], (table["corner_side"] == "right").to_numpy()):
        raise SystemExit("STOP: the frame's kick side disagrees with s01's corner_side")
    P = max(len(got[s][1]) for s in ids)
    B, M = np.zeros((len(ids), P, _NODE), np.float32), np.zeros((len(ids), P), bool)
    for i, s in enumerate(ids):
        b = got[s][1]
        B[i, :len(b)], M[i, :len(b)] = b, True
    return B, M


# ── The network ──────────────────────────────────────────────────────────
class DenseGATv2(nn.Module):
    """One GATv2 layer (Brody et al. 2022, Eq. 7) over every pair of a padded frame's bodies, the
    pair's edge feature inside the score as in TacticAI's Eq. 4: torch_geometric's GATv2Conv on a
    complete graph with self-pairs, written dense, glorot weights as it initialises them. A padded
    body is a key no body attends to; its own row is computed and left to the pools' mask."""

    def __init__(self, d_in: int, heads: int, head_dim: int) -> None:
        super().__init__()
        self.h, self.c = heads, head_dim
        self.lin_l = nn.Linear(d_in, heads * head_dim)                 # the source body, x_j
        self.lin_r = nn.Linear(d_in, heads * head_dim)                 # the target body, x_i
        self.lin_e = nn.Linear(1, heads * head_dim, bias=False)
        self.att = nn.Parameter(torch.empty(heads, head_dim))
        self.bias = nn.Parameter(torch.zeros(heads * head_dim))
        for w in (self.lin_l.weight, self.lin_r.weight, self.lin_e.weight, self.att):
            nn.init.xavier_uniform_(w)
        nn.init.zeros_(self.lin_l.bias)
        nn.init.zeros_(self.lin_r.bias)

    def forward(self, h: torch.Tensor, mask: torch.Tensor, same: torch.Tensor) -> torch.Tensor:
        n, p = h.shape[:2]
        xl = self.lin_l(h).view(n, p, self.h, self.c)
        xr = self.lin_r(h).view(n, p, self.h, self.c)
        # lin_e on one binary feature is a product: written as one it skips a GEMM of inner size one
        # (data/derived/prereg_2026-10-07/evidence/gnn_compile_probe.json).
        e = same[..., None, None] * self.lin_e.weight.view(self.h, self.c)   # (B, i, j, H, C)
        s = nn.functional.leaky_relu(xr[:, :, None] + xl[:, None] + e, 0.2)
        score = (s * self.att).sum(-1).masked_fill(~mask[:, None, :, None], -1e9)
        alpha = torch.softmax(score, dim=2)
        return torch.einsum("bijh,bjhc->bihc", alpha, xl).reshape(n, p, -1) + self.bias


class GraphNet(nn.Module):
    """Embedding, `layers` residual GATv2 layers, then GPU1's readout and head: masked mean and max
    of the body states over each side, joined with the design and read by ρ into the classes. A
    design of width 0 reads the frame alone, the positive control's arm."""

    def __init__(self, d_design: int, k: int, layers: int, rho_width: int, dropout: float) -> None:
        super().__init__()
        self.embed = nn.Linear(_NODE, _WIDTH)
        self.gat = nn.ModuleList(DenseGATv2(_WIDTH, _HEADS, _HEAD_DIM) for _ in range(layers))
        self.rho = nn.Sequential(nn.Linear(4 * _WIDTH + d_design, rho_width), nn.ReLU(),
                                 nn.Dropout(dropout), nn.Linear(rho_width, k))

    def forward(self, bodies: torch.Tensor, mask: torch.Tensor,
                design: torch.Tensor) -> torch.Tensor:
        side = bodies[..., 2]
        same = (side[:, :, None] == side[:, None, :]).float()
        h = self.embed(bodies)
        for layer in self.gat:
            h = torch.relu(h + layer(h, mask, same))
        pools = []
        for s in (1.0, 0.0):
            m = (mask & (side == s)).unsqueeze(-1)
            n = m.sum(1)
            pools += [(h * m).sum(1) / n.clamp(min=1), h.masked_fill(~m, -1e9).amax(1) * (n > 0)]
        return self.rho(torch.cat(pools + [design], dim=1))


def space(trial: optuna.Trial, n_train: int, n_minority: int) -> dict:
    """GPU1's space with the message passing in place of φ. It holds TacticAI's grid (depth,
    learning rate, batch and L2, Wang et al. 2024, Methods) and searches GPU1's head the same way,
    up to GPU1's ceiling on ρ's width."""
    h_max = int(max(16, min(128, n_train // 110, n_minority)))
    return {"layers": trial.suggest_categorical("layers", list(_LAYERS)),
            "rho_width": trial.suggest_int("rho_width", 8, h_max, log=True),
            "dropout": trial.suggest_float("dropout", 0.0, 0.5),
            "weight_decay": trial.suggest_float("weight_decay", 1e-6, 1e-1, log=True),
            "lr": trial.suggest_float("lr", 5e-5, 3e-3, log=True),
            "batch_size": trial.suggest_categorical("batch_size", [128, 256, 512])}


def _logits(net: GraphNet, B: torch.Tensor, M: torch.Tensor, Dt: torch.Tensor,
            idx: torch.Tensor) -> torch.Tensor:
    """The network on `idx` in chunks: a fold at once would hold (rows, P, P, 32) per layer."""
    return torch.cat([net(B[i], M[i], Dt[i]) for i in idx.split(_CHUNK)])


def fit_predict(p: dict, data: tuple, onehot: torch.Tensor, tr: np.ndarray, te: np.ndarray,
                seed: int, max_steps: int,
                epochs: int | None = None) -> tuple[np.ndarray, int, float, bool]:
    """GPU1's fit with GraphNet: (probabilities on `te`, epochs, validation loss, capped). Without
    `epochs`, `te` is the validation fold: the fit keeps its best epoch and stops _PATIENCE epochs
    after it, or at the last epoch inside `max_steps` updates, which `capped` reports. The design's
    impute and scale are fitted on `tr` alone."""
    B, M, D = data
    Dt = torch.tensor(CompetingOutcomesModel._fit_prep(D[tr])(D) if D.shape[1] else D,
                      dtype=torch.float32, device=B.device)
    torch.manual_seed(seed)
    net = GraphNet(Dt.shape[1], onehot.shape[1], p["layers"], p["rho_width"],
                   p["dropout"]).to(B.device)
    opt = torch.optim.AdamW(net.parameters(), lr=p["lr"], weight_decay=p["weight_decay"])
    gen = torch.Generator().manual_seed(seed)
    tr_t, te_t = torch.as_tensor(tr, device=B.device), torch.as_tensor(te, device=B.device)
    best, best_state, best_epoch, capped = np.inf, None, 0, False
    cap = max(1, max_steps // -(-len(tr) // p["batch_size"]))
    for epoch in range(1, (epochs or cap) + 1):
        net.train()
        for b in tr_t[torch.randperm(len(tr), generator=gen).to(B.device)].split(p["batch_size"]):
            opt.zero_grad()
            G._loss(net(B[b], M[b], Dt[b]), onehot[b]).backward()
            opt.step()
        if epochs is None:
            net.eval()
            with torch.no_grad():
                v = float(G._loss(_logits(net, B, M, Dt, te_t), onehot[te_t]))
            if v < best:
                best, best_epoch = v, epoch
                best_state = {n: w.detach().clone() for n, w in net.state_dict().items()}
            elif epoch - best_epoch >= _PATIENCE:
                break
    else:
        capped = epochs is None
    if best_state is not None:
        net.load_state_dict(best_state)
    net.eval()
    with torch.no_grad():
        prob = torch.softmax(_logits(net, B, M, Dt, te_t), dim=1).cpu().numpy()
    return prob.astype(float), (best_epoch if epochs is None else epochs), best, capped


def gnn_task(B: np.ndarray, M: np.ndarray, D: np.ndarray, y: np.ndarray, groups: np.ndarray,
             tr: np.ndarray, te: np.ndarray, k: int, n_trials: int, seed: int, fold: int,
             fixed: dict | None = None, max_steps: int = _MAX_STEPS) -> dict:
    """One (outer fold, seed): the nested search on the fold's training matches and the refit
    scored on its test matches, or, with `fixed`, the refit alone. Module-level so joblib ships it;
    one TPE trial at a time and one CPU thread, so the study is a function of its seed."""
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    # cuBLAS is deterministic only with a fixed workspace, read when this process first opens it.
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.use_deterministic_algorithms(True)
    torch.set_num_threads(1)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if dev.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    data = tuple(torch.as_tensor(a, device=dev) for a in (B, M)) + (D,)
    onehot = torch.nn.functional.one_hot(torch.as_tensor(y, device=dev), k).float()
    t0, trials = time.time(), []
    if fixed is None:
        inner = list(GroupKFold(G._N_INNER).split(tr, groups=groups[tr]))
        for a, b in inner:
            assert_no_group_leak(groups[tr], a, b, "match_id")
        n_min = int(np.bincount(y[tr], minlength=k).min())

        def objective(trial: "optuna.Trial") -> float:
            p = space(trial, len(tr), n_min)
            out = [fit_predict(p, data, onehot, tr[a], tr[b], seed, max_steps) for a, b in inner]
            trial.set_user_attr("epochs", float(np.mean([o[1] for o in out])))
            trial.set_user_attr("capped", float(np.mean([o[3] for o in out])))
            return float(np.mean([o[2] for o in out]))

        study = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(
            seed=seed + fold, multivariate=True))
        study.optimize(objective, n_trials=n_trials, n_jobs=1, show_progress_bar=False)
        trials = [(t.duration.total_seconds(), t.user_attrs["epochs"], t.user_attrs["capped"])
                  for t in study.trials]
        fixed = {"params": study.best_params, "epochs": max(1, round(
            study.best_trial.user_attrs["epochs"]))}
    prob = fit_predict(dict(fixed["params"]), data, onehot, tr, te, seed, max_steps,
                       epochs=fixed["epochs"])[0]
    peak = torch.cuda.max_memory_reserved() / 2**20 if dev.type == "cuda" else 0.0
    return {"fold": fold, "seed": seed, "te": te, "prob": prob, "fixed": fixed, "trials": trials,
            "seconds": time.time() - t0, "peak_reserved_mb": peak,
            "peak_rss_mb": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024}


def _task_saved(save: Path | None, *args: Any) -> dict:
    """gnn_task, its result written to `save` the moment it ends, through a temporary file and an
    atomic rename, so a crash leaves either the whole task or nothing."""
    out = gnn_task(*args)
    if save is not None:
        tmp = save.with_suffix(".tmp")
        tmp.write_bytes(pickle.dumps(out))
        os.replace(tmp, save)
    return out


def run_tasks(B: np.ndarray, M: np.ndarray, D: np.ndarray, y: np.ndarray, groups: np.ndarray,
              tasks: list, k: int, n_trials: int, jobs: int, ckpt: Path | None = None,
              load_only: bool = False, max_steps: int = _MAX_STEPS) -> list:
    """GPU1's runner for GNN tasks: (seed, fold, tr, te, fixed) tasks, one worker process each on
    the card, in task order. With `ckpt` every task is kept under a digest of all it reads, so a
    rerun resumes by content and a changed input misses the old file; `load_only` fits nothing,
    and a task missing from `ckpt` stops the run."""
    saves: list = [None] * len(tasks)
    if ckpt is not None:
        ckpt.mkdir(parents=True, exist_ok=True)
        base = G._digest(B, M, D, y, groups, k, n_trials, "gnn", max_steps)
        saves = [ckpt / f"{G._digest(base, s, f, tr, te, fx)}.pkl" for s, f, tr, te, fx in tasks]
    todo = [i for i, p in enumerate(saves) if p is None or not p.exists()]
    if load_only and todo:
        raise SystemExit(f"STOP: {len(todo)} of {len(tasks)} tasks are not saved in {ckpt}, and a "
                         f"load-only run fits none: the frames, the design or the target changed")
    with parallel_config(backend="loky", inner_max_num_threads=1):
        fresh = Parallel(n_jobs=max(1, min(jobs, len(todo))))(
            delayed(_task_saved)(saves[i], B, M, D, y, groups, tasks[i][2], tasks[i][3], k,
                                 n_trials, tasks[i][0], tasks[i][1], tasks[i][4], max_steps)
            for i in todo)
    got = dict(zip(todo, fresh, strict=True))
    return [got[i] if i in got else pickle.loads(saves[i].read_bytes()) for i in range(len(tasks))]


# ── The readings ─────────────────────────────────────────────────────────
def interval(d: np.ndarray, g: np.ndarray, boot: int) -> dict:
    """A paired per-row difference and its match-clustered 95% interval."""
    ci = cluster_bootstrap_mean(d, g, n_boot=boot)
    return {"point": ci.point, "lo": ci.lo, "hi": ci.hi, "se": ci.se}


def better_of_two(gnn: dict, gpu: dict) -> dict:
    """gnn.md: a network that passes its gate is better than one that does not; between two that
    both pass or both fail, the lower mean of the paired statistic over the seeds wins, and a tie
    at the fifth decimal keeps GPU1, the simpler network. Each side is {"enters", "mean"}."""
    if gnn["enters"] != gpu["enters"]:
        chosen = "GNN1" if gnn["enters"] else "GPU1"
    else:
        chosen = "GNN1" if round(gnn["mean"], 5) < round(gpu["mean"], 5) else "GPU1"
    return {"chosen": chosen, "GNN1": gnn, "GPU1": gpu}


def decide_p1(frame: dict, shuffled: dict) -> dict:
    """P1: the frame's interval against the prior wholly below zero, and the shuffled target's not."""
    below, clean = bool(frame["hi"] < 0), bool(shuffled["hi"] >= 0)
    return {"P1": below and clean, "frame_below_zero": below, "shuffle_clean": clean}


def control_rows(t: pd.DataFrame, grain: pd.DataFrame, pop: str) -> pd.DataFrame:
    """The registered control's rows (run_positive_control), the deliveries with a first contact in
    match order, without the short corners on corners: a frame shows a short corner, a team-mate
    standing by the flag, and that pass would be the trivial one."""
    c = t.merge(grain[["set_piece_id", "won_first_contact"]], on="set_piece_id", how="inner",
                validate="1:1")
    c = c[c["won_first_contact"].notna()]
    if pop == "corner":
        c = c[~c["short_corner"].astype(bool)]
    return c.sort_values(["match_id", "set_piece_id"]).reset_index(drop=True)


def prior_loss(c: pd.DataFrame, y: np.ndarray, g: np.ndarray) -> np.ndarray:
    """The registered control's prior, an intercept chosen inside every fold, scored per row."""
    X = CompetingOutcomesModel(n_splits=5)._design(c, numeric=[], with_delivery_cats=False)[0]
    p, _ = oof_proba(np.asarray(X, dtype=float), y, g)
    return G.row_logloss(np.c_[1 - p, p], y)


def grain(tables: Path, pop: str) -> pd.DataFrame:
    """The delivery grain carrying `won_first_contact`: the plan's file for the population, or the
    joint one the recipe writes, cut to it."""
    own = tables / f"eda_deliveries_{pop}.parquet"
    if own.exists():
        return pd.read_parquet(own)
    g = pd.read_parquet(tables / "eda_deliveries.parquet")
    return g[g["set_piece_type"] == pop]


# ── The run ──────────────────────────────────────────────────────────────
def population(pop: str, tables: Path, limit: int | None, jobs: int) -> tuple:
    """The table, its frames and the delivery-rung design, row-aligned, as GPU1 reads them."""
    t = pd.read_parquet(tables / POPULATIONS[pop])
    if limit:
        t = t[t["match_id"].isin(sorted(t["match_id"].unique())[:limit])]
    t = t.reset_index(drop=True)
    B, M = build(t, pop, jobs)
    D = CompetingOutcomesModel()._design(t, numeric=_KICK_INTRINSIC)[0]
    classes = sorted(t["target"].unique())
    y = t["target"].map({c: i for i, c in enumerate(classes)}).to_numpy()
    print(f"{pop}: {len(t)} deliveries, {B.shape[1]} bodies at most, {D.shape[1]} design columns, "
          f"classes {classes}", flush=True)
    return t, B, M, D, y, len(classes)


def taker_flags(B: np.ndarray, M: np.ndarray) -> dict:
    """The share of frames carrying exactly one taker flag, none, and more than one."""
    n = (B[..., 4] * M).sum(1)
    return {"one": float(np.mean(n == 1)), "none": float(np.mean(n == 0)),
            "more": float(np.mean(n > 1))}


def smoke(pop: str, data: tuple, args: argparse.Namespace) -> list:
    """The cost of the arm, blind: the first --jobs (seed, outer fold) tasks with --trials trials
    side by side on the card, the first again in another process to check a bit-identical rerun,
    then one positive-control task alone; with the taker flags per frame and the share of fits the
    epoch cap stopped. Every probability stays in memory and no loss is read."""
    t, B, M, D, y, k = data
    g = t["match_id"].to_numpy()
    flags = taker_flags(B, M)
    print(f"  taker flags per frame: {flags}", flush=True)
    folds = list(GroupKFold(G._N_FOLDS).split(B, groups=g))
    tasks = [(s, f, tr, te, None) for s in range(args.seeds)
             for f, (tr, te) in enumerate(folds)][:args.jobs]
    print(f"card before: {G.card_state()}", flush=True)
    t0 = time.time()
    done = run_tasks(B, M, D, y, g, tasks + tasks[:1], k, args.trials, args.jobs + 1,
                     max_steps=args.max_steps)
    wall = time.time() - t0
    same = np.array_equal(done[0]["prob"], done[-1]["prob"])
    rows = []
    for r in done[:-1]:
        per = [s for s, _, _ in r["trials"]]
        rows.append({"population": pop, "block": "task", "seed": r["seed"], "fold": r["fold"],
                     "trials": len(per), "task_s": r["seconds"], "trial_s_mean": float(np.mean(per)),
                     "trial_s_max": float(np.max(per)),
                     "epochs_mean": float(np.mean([e for _, e, _ in r["trials"]])),
                     "capped_share": float(np.mean([c for _, _, c in r["trials"]])),
                     "refit_epochs": r["fixed"]["epochs"], "peak_reserved_mb": r["peak_reserved_mb"],
                     "peak_rss_mb": r["peak_rss_mb"], "concurrent": len(tasks) + 1, "wall_s": wall,
                     "rerun_identical": same, **{f"taker_{n}": v for n, v in flags.items()}})
        print(f"  seed {r['seed']} fold {r['fold']}: {r['seconds']:.0f}s for {len(per)} trials (mean "
              f"{np.mean(per):.1f}s), capped {rows[-1]['capped_share']:.2f}, peak reserved "
              f"{r['peak_reserved_mb']:.0f} MB, RSS {r['peak_rss_mb']:.0f} MB", flush=True)
    print(f"  {len(tasks) + 1} concurrent tasks in {wall:.0f}s; the rerun of the first task is "
          f"{'bit-identical' if same else 'NOT IDENTICAL'}", flush=True)
    c = control_rows(t, grain(Path(args.tables), pop), pop)
    pos = pd.Series(np.arange(len(t)), index=t["set_piece_id"]).loc[c["set_piece_id"]].to_numpy()
    gc = c["match_id"].to_numpy()
    tr, te = next(GroupKFold(G._N_FOLDS).split(pos, groups=gc))
    yc = c["won_first_contact"].astype(bool).to_numpy().astype(int)
    r = run_tasks(B[pos], M[pos], np.zeros((len(c), 0)), yc, gc, [(0, 0, tr, te, None)], 2,
                  args.trials, 1, max_steps=args.max_steps)[0]
    rows.append({"population": pop, "block": "control_task", "trials": len(r["trials"]),
                 "task_s": r["seconds"], "peak_reserved_mb": r["peak_reserved_mb"],
                 "peak_rss_mb": r["peak_rss_mb"], "concurrent": 1})
    print(f"  one control task alone: {r['seconds']:.0f}s for {len(r['trials'])} trials on "
          f"{len(c)} rows", flush=True)
    return rows


def evaluate(pop: str, data: tuple, args: argparse.Namespace, ckpt: Path | None,
             load_only: bool) -> tuple[list, dict]:
    """GNN1 at every seed against the tuned delivery rung on the same folds, the shuffled-frame
    control at seed 0's configurations, the frame's own share and the head-to-head with the full
    model, and the verdict."""
    t, B, M, D, y, k = data
    g = t["match_id"].to_numpy()
    folds = list(GroupKFold(G._N_FOLDS).split(B, groups=g))
    for tr, te in folds:
        assert_no_group_leak(g, tr, te, "match_id")
    if not load_only:
        print(f"card before: {G.card_state()}", flush=True)
    done = run_tasks(B, M, D, y, g, [(s, f, tr, te, None) for s in range(args.seeds)
                                     for f, (tr, te) in enumerate(folds)], k, args.trials,
                     args.jobs, ckpt, load_only, args.max_steps)
    m = _tuned(t, True, args.jobs)
    assert m.n_splits == G._N_FOLDS, "the delivery rung and the arm must score the same outer folds"
    l_del = m._row_logloss(m.fit_cv(t, numeric=_KICK_INTRINSIC), t["target"].to_numpy())
    l_full = m._row_logloss(m.fit_cv(t), t["target"].to_numpy())
    arm = {s: G.row_logloss(p, y) for s, p in G.oof(done, len(y), k).items()}
    seeds = {s: G.paired(l - l_del, g, args.boot) for s, l in arm.items()}
    perm = np.random.default_rng(0).permutation(len(y))
    fixed = {r["fold"]: r["fixed"] for r in done if r["seed"] == 0}
    shuffled = run_tasks(B[perm], M[perm], D, y, g, [(0, f, tr, te, fixed[f]) for f, (tr, te)
                                                     in enumerate(folds)], k, 0, args.jobs, ckpt,
                         load_only, args.max_steps)
    l_shuf = G.row_logloss(G.oof(shuffled, len(y), k)[0], y)
    control = G.paired(l_shuf - l_del, g, args.boot)
    verdict = G.decide(seeds, control)
    rows = [{"population": pop, "block": "seed", "key": str(s), **r} for s, r in seeds.items()]
    rows += [{"population": pop, "block": "control", "key": "shuffled_frame", **control},
             {"population": pop, "block": "twin", "key": "gnn-shuffled_frame",
              **interval(arm[0] - l_shuf, g, args.boot)},
             {"population": pop, "block": "head_to_head", "key": "gnn-full",
              **interval(arm[0] - l_full, g, args.boot)},
             {"population": pop, "block": "verdict", "key": "GNN1", "params": str(verdict)}]
    rows += [{"population": pop, "block": "task", "key": f"{r['seed']}/{r['fold']}",
              "seconds": r["seconds"], "peak_reserved_mb": r["peak_reserved_mb"],
              "peak_rss_mb": r["peak_rss_mb"], "params": str(r["fixed"])} for r in done]
    for s, r in seeds.items():
        print(f"  seed {s}: {r['point']:+.5f} [{r['lo']:+.5f}, {r['hi']:+.5f}]  MDE80 "
              f"{r['mde80']:.5f}", flush=True)
    print(f"  shuffled frame: {control['point']:+.5f} [{control['lo']:+.5f}, {control['hi']:+.5f}]"
          f"\n  GNN1 ({pop}): {verdict}", flush=True)
    return rows, {"verdict": verdict, "seeds": seeds, "arm": arm, "l_del": l_del}


def positive_control(pop: str, data: tuple, args: argparse.Namespace, ckpt: Path | None,
                     load_only: bool) -> list:
    """P1: the network on the frame alone beats the prior on who wins the first contact, and the
    same network refitted at P1's configurations on the globally shuffled target does not beat that
    target's prior. The registered control's folds; GNN1's search at seed 0, two classes."""
    t, B, M = data[:3]
    c = control_rows(t, grain(Path(args.tables), pop), pop)
    pos = pd.Series(np.arange(len(t)), index=t["set_piece_id"]).loc[c["set_piece_id"]].to_numpy()
    Bc, Mc, g = B[pos], M[pos], c["match_id"].to_numpy()
    y, D0 = c["won_first_contact"].astype(bool).to_numpy().astype(int), np.zeros((len(c), 0))
    folds = list(GroupKFold(G._N_FOLDS).split(Bc, groups=g))
    for tr, te in folds:
        assert_no_group_leak(g, tr, te, "match_id")
    done = run_tasks(Bc, Mc, D0, y, g, [(0, f, tr, te, None) for f, (tr, te) in enumerate(folds)],
                     2, args.control_trials, args.jobs, ckpt, load_only, args.max_steps)
    frame, prior = G.row_logloss(G.oof(done, len(y), 2)[0], y), prior_loss(c, y, g)
    ys = shuffle_labels(y, g, mode="global", seed=0)
    fixed = {r["fold"]: r["fixed"] for r in done}
    shuf = run_tasks(Bc, Mc, D0, ys, g, [(0, f, tr, te, fixed[f]) for f, (tr, te)
                                         in enumerate(folds)], 2, 0, args.jobs, ckpt, load_only,
                     args.max_steps)
    p1 = interval(frame - prior, g, args.boot)
    clean = interval(G.row_logloss(G.oof(shuf, len(ys), 2)[0], ys) - prior_loss(c, ys, g), g,
                     args.boot)
    verdict = decide_p1(p1, clean)
    print(f"  positive control ({pop}, {len(c)} rows, attack wins {100 * y.mean():.1f}%): frame "
          f"minus prior {p1['point']:+.5f} [{p1['lo']:+.5f}, {p1['hi']:+.5f}]; shuffled target "
          f"{clean['point']:+.5f} [{clean['lo']:+.5f}, {clean['hi']:+.5f}]\n  {verdict}", flush=True)
    return [{"population": pop, "block": "positive_control", "key": "frame-prior", "n": len(c), **p1},
            {"population": pop, "block": "positive_control", "key": "frame-prior/shuffled_global",
             "n": len(c), **clean},
            {"population": pop, "block": "verdict", "key": "P1", "params": str(verdict)}]


def gpu1_losses(pop: str, data: tuple, args: argparse.Namespace) -> dict:
    """GPU1 as registered, read from its own tasks and never refitted: its frames rebuilt by its
    own builder, every task loaded (a missing one stops the run), per seed and for its shuffled
    frame at seed 0's configurations."""
    t, _, _, D, y, k = data
    g = t["match_id"].to_numpy()
    B4, M4 = G.build(t, pop, args.build_jobs)
    folds = list(GroupKFold(G._N_FOLDS).split(B4, groups=g))
    ckpt = Path(args.gpu_tasks)
    done = G.run_tasks(B4, M4, D, y, g, [(s, f, tr, te, None) for s in range(G._SEEDS)
                                         for f, (tr, te) in enumerate(folds)], k, G._GPU_TRIALS,
                       args.jobs, ckpt, load_only=True)
    perm = np.random.default_rng(0).permutation(len(y))
    fixed = {r["fold"]: r["fixed"] for r in done if r["seed"] == 0}
    shuffled = G.run_tasks(B4[perm], M4[perm], D, y, g, [(0, f, tr, te, fixed[f]) for f, (tr, te)
                                                         in enumerate(folds)], k, 0, args.jobs,
                           ckpt, load_only=True)
    return {"arm": {s: G.row_logloss(p, y) for s, p in G.oof(done, len(y), k).items()},
            "shuffled": G.row_logloss(G.oof(shuffled, len(y), k)[0], y)}


def versus_gpu1(pop: str, data: tuple, ev: dict, args: argparse.Namespace) -> list:
    """GNN1 against GPU1 per row and seed, GPU1's verdict recomputed on its tasks, and the better of
    the two by the sheet's rule: deciding on corners, replicating on flank free kicks."""
    g = data[0]["match_id"].to_numpy()
    gpu = gpu1_losses(pop, data, args)
    seeds = {s: G.paired(l - ev["l_del"], g, args.boot) for s, l in gpu["arm"].items()}
    verdict = G.decide(seeds, G.paired(gpu["shuffled"] - ev["l_del"], g, args.boot))
    rows = [{"population": pop, "block": "versus_gpu1", "key": str(s),
             **interval(ev["arm"][s] - gpu["arm"][s], g, args.boot)} for s in ev["arm"]]
    best = better_of_two(
        {"enters": ev["verdict"]["enters"],
         "mean": float(np.mean([r["point"] for r in ev["seeds"].values()]))},
        {"enters": verdict["enters"], "mean": float(np.mean([r["point"] for r in seeds.values()]))})
    print(f"  GPU1 on its tasks ({pop}): {verdict}\n  better of the two: {best}", flush=True)
    return rows + [{"population": pop, "block": "verdict", "key": "GPU1", "params": str(verdict)},
                   {"population": pop, "block": "better_of_two",
                    "key": "decides" if pop == "corner" else "replicates", "params": str(best)}]


def main() -> None:
    """Frames and designs per population, then the smoke, or GNN1, its positive control and the
    better of the two."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--jobs", type=int, default=6, help="worker processes sharing the card")
    ap.add_argument("--trials", type=int, default=_TRIALS, help="Optuna trials per task")
    ap.add_argument("--control-trials", type=int, default=_CONTROL_TRIALS,
                    help="Optuna trials per positive-control task")
    ap.add_argument("--seeds", type=int, default=_SEEDS)
    ap.add_argument("--boot", type=int, default=_BOOT)
    ap.add_argument("--max-steps", type=int, default=_MAX_STEPS, help="training steps a fit may take")
    ap.add_argument("--limit", type=int, default=None, help="first N matches: the path, not the numbers")
    ap.add_argument("--tables", type=str, default=str(_PLAN / "tables"))
    ap.add_argument("--out-dir", type=str, default=str(_PLAN / "gnn"))
    ap.add_argument("--build-jobs", type=int, default=23, help="workers reading the frames")
    ap.add_argument("--smoke", action="store_true", help="cost, VRAM and RSS per task, no score read")
    ap.add_argument("--tasks", type=str, default=None,
                    help="where every task is kept (default: <out-dir>/tasks)")
    ap.add_argument("--gpu-tasks", type=str, default=str(_PLAN / "gpu" / "tasks"),
                    help="GPU1's tasks, loaded for the better of the two; empty skips it")
    ap.add_argument("--load-only", action="store_true",
                    help="read every task from --tasks and fit none: the recipe's stage")
    args = ap.parse_args()
    clock = time.time()
    out_dir = Path(args.out_dir)
    tail = f"_limit{args.limit}" if args.limit else ""
    published = not tail and (published_dir(out_dir) or
                              out_dir.resolve() == (_PLAN / "gnn").resolve())
    budget = (args.trials, args.control_trials, args.seeds, args.boot, args.max_steps)
    registered = (_TRIALS, _CONTROL_TRIALS, _SEEDS, _BOOT, _MAX_STEPS)
    versus = bool(args.gpu_tasks) and not tail
    if published and not args.smoke and budget != registered:
        raise SystemExit(f"STOP: GNN1 is registered at {registered} (trials, control trials, seeds, "
                         f"draws, step cap), so a run at {budget} may not write frame_gnn.parquet")
    if published and not args.smoke and not versus:
        raise SystemExit("STOP: the published record carries the better of the two, so it reads "
                         "GPU1's tasks; pass --gpu-tasks")
    if not tail and published_dir(out_dir) and Path(args.tables).resolve() != _DER.resolve():
        raise SystemExit("STOP: tables outside data/derived are not the published ones, so their "
                         "readings may not take the published names; pass --out-dir")
    if args.smoke and args.load_only:
        raise SystemExit("STOP: --smoke fits tasks to time them and --load-only fits none")
    tasks = Path(args.tasks) if args.tasks else out_dir / f"tasks{tail}"
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for pop in POPULATIONS:
        data = population(pop, Path(args.tables), args.limit, args.build_jobs)
        if args.smoke:
            rows += smoke(pop, data, args)
            continue
        ev_rows, ev = evaluate(pop, data, args, tasks, args.load_only)
        rows += ev_rows + positive_control(pop, data, args, tasks, args.load_only)
        if versus:
            rows += versus_gpu1(pop, data, ev, args)
    path = out_dir / f"frame_gnn{'_smoke' if args.smoke else ''}{tail}.parquet"
    pd.DataFrame(rows).to_parquet(path, index=False)
    print(f"\nwrote {path}  [{time.time() - clock:.0f}s]", flush=True)


if __name__ == "__main__":
    main()
