"""run_frame_gpu — does a network on the raw delivery frame add to the delivery design?

The 360 step over the delivery rung covers zero on corners (evidence.md §5, §17), and the handcrafted
geometry does not beat the delivery baseline either (§14). Neither says whether a flexible model of
the raw frame, given the delivery design too, finds what the handcrafted block misses.
Pre-registered (data/derived/prereg_2026-10-01/gpu.md) before any such network was trained.

THE ARM. Every detected body of the delivery 360 in the canonical frame (attack towards x = 120, the
kick struck from the y = 0 side) as (x, y, attacker, keeper). A per-body network φ, pooled by masked
mean and max over each side, is joined with s03's delivery-rung design and read by ρ into the five
classes of s03's target: permutation-invariant by construction (DeepSets, Zaheer et al. 2017).
Optuna searches it inside each outer fold on that fold's training matches, three inner folds by
match; every fit stops on its inner validation fold, and the refit on the outer training block
runs the mean of the epochs those fits stopped at. One process per (outer fold,
seed) task on the GPU, deterministic algorithms on, so a rerun reproduces the run. Each task is kept
in `<out-dir>/tasks/` the moment it ends, so a run of hours that stops resumes where it was.

THE BASELINE is s03's delivery rung tuned as run_validation tunes it, scored through the same
GroupKFold on the same rows, so both arms see the same outer folds.

GPU1, THE DECISION: the paired per-row OOF log loss, encoder plus delivery minus delivery, has its
match-clustered 95% interval wholly below −MARGIN at every one of five seeds, and the frame shuffled
between corners does not beat the delivery. Otherwise the arm does not enter and its null is
published with its bound. Reported, never deciding: the MDE at 80% of each seed, the encoder against
the full handcrafted model, the time and the peak VRAM of every task and the card's state before the
run; flank free kicks replicate.

The recipe trains nothing: its stage reads the plan's sixty tasks with --load-only, and a task it
cannot find, because the frames, the design or the target moved, stops the run instead of fitting.

    python pipeline/run_frame_gpu.py --smoke --trials 4 --jobs 5     # cost, VRAM, determinism; no score
    python pipeline/run_frame_gpu.py --trials 60 --jobs 5
    python pipeline/run_frame_gpu.py --tables data/derived --out-dir data/derived --load-only \\
        --tasks data/derived/auditoria_2026-10-12/gpu/tasks --jobs 8        # the recipe's stage
"""
from __future__ import annotations

import argparse
import hashlib
import os
import pickle
import subprocess
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
_PLAN = _DER / "auditoria_2026-10-12"

import loader_sb as L                                                      # noqa: E402
from build_table import published_dir                                     # noqa: E402
from constants import PITCH_X_SB, PITCH_Y_SB                              # noqa: E402
from run_validation import _tuned                                         # noqa: E402
from s01_phases import SetPieceGrammar                                    # noqa: E402
from s02_signature import CommitmentSignature                             # noqa: E402
from s03_value import _KICK_INTRINSIC, CompetingOutcomesModel             # noqa: E402
from validate import Z80, assert_no_group_leak, cluster_bootstrap_mean    # noqa: E402
from zones import canonical_y, kicked_from_right                          # noqa: E402

POPULATIONS = {"corner": "corners.parquet", "free_kick_flank": "corners_free_kick_flank.parquet"}
# The sheet's margin (gpu.md): M2 of the non-inferiority test, a tenth of the delivery step's own
# lower bound on the published record.
MARGIN = 0.00331
_BODY = 4                          # x / 120, y / 80, attacker, keeper
_N_FOLDS, _N_INNER = 5, 3
# The budget the sheet fixed from the smoke's cost: sixty trials per task, five seeds, and a fit
# that stops after twenty epochs without a better validation loss.
_GPU_TRIALS, _SEEDS, _BOOT = 60, 5, 2000
_MAX_EPOCHS, _PATIENCE = 300, 20


# ── The frames ───────────────────────────────────────────────────────────
def frame_rows(d: pd.Series, fz: pd.DataFrame) -> tuple[bool, np.ndarray]:
    """One delivery's detected bodies, ball out, mirrored onto the y = 0 side when the kick came
    from the attackers' right: (x / 120, y / 80, attacker, keeper) per body."""
    f = CommitmentSignature.from_corner(d, fz).frame
    right = bool(kicked_from_right(f.loc[f["is_ball"] == 1, "y"].iloc[0]))
    b = f[f["is_ball"] == 0]
    y = canonical_y(b["y"].to_numpy(float), right)
    return right, np.c_[b["x"].to_numpy(float) / PITCH_X_SB, y / PITCH_Y_SB,
                        b["team_id"].to_numpy(float), b["is_goalkeeper"].to_numpy(float)]


def match_frames(mid: int, kind: str, want: set) -> list:
    """(set_piece_id, right, bodies) for one match's wanted deliveries with a 360."""
    ev, fz = L.load_events(mid), L.load_360(mid)
    have = set(fz["event_uuid"])
    return [(c["id"], *frame_rows(c, fz))
            for _, c in SetPieceGrammar(ev, kinds=(kind,)).deliveries().iterrows()
            if c["id"] in want and c["id"] in have]


def build(table: pd.DataFrame, kind: str, jobs: int) -> tuple[np.ndarray, np.ndarray]:
    """(bodies, mask) row-aligned to `table`: (N, P, 4) zero-padded and (N, P) true on a body. The
    run stops if a row has no frame or its frame's kick side disagrees with s01's corner_side."""
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
    B, M = np.zeros((len(ids), P, _BODY), np.float32), np.zeros((len(ids), P), bool)
    for i, s in enumerate(ids):
        b = got[s][1]
        B[i, :len(b)], M[i, :len(b)] = b, True
    return B, M


# ── The network ──────────────────────────────────────────────────────────
class FrameNet(nn.Module):
    """φ per body, masked mean and max over each side, ρ on [pools | design] into the classes."""

    def __init__(self, d_design: int, k: int, phi_width: int, phi_depth: int, rho_width: int,
                 dropout: float) -> None:
        super().__init__()
        layers, d = [], _BODY
        for _ in range(phi_depth):
            layers += [nn.Linear(d, phi_width), nn.ReLU()]
            d = phi_width
        self.phi = nn.Sequential(*layers)
        self.rho = nn.Sequential(nn.Linear(4 * phi_width + d_design, rho_width), nn.ReLU(),
                                 nn.Dropout(dropout), nn.Linear(rho_width, k))

    def forward(self, bodies: torch.Tensor, mask: torch.Tensor,
                design: torch.Tensor) -> torch.Tensor:
        h, pools = self.phi(bodies), []                           # h: (B, P, W)
        for side in (1.0, 0.0):
            m = (mask & (bodies[..., 2] == side)).unsqueeze(-1)   # (B, P, 1)
            n = m.sum(1)
            pools += [(h * m).sum(1) / n.clamp(min=1), h.masked_fill(~m, -1e9).amax(1) * (n > 0)]
        return self.rho(torch.cat(pools + [design], dim=1))


def space(trial: optuna.Trial, n_train: int, n_minority: int) -> dict:
    """The search space. ρ's width ceiling is derived from the training block: about half its
    rows in head weights at most, and never wider than the smallest class."""
    h_max = int(max(16, min(128, n_train // 110, n_minority)))
    return {"phi_width": trial.suggest_int("phi_width", 8, 64, log=True),
            "phi_depth": trial.suggest_int("phi_depth", 1, 2),
            "rho_width": trial.suggest_int("rho_width", 8, h_max, log=True),
            "dropout": trial.suggest_float("dropout", 0.0, 0.5),
            "weight_decay": trial.suggest_float("weight_decay", 1e-6, 1e-1, log=True),
            "lr": trial.suggest_float("lr", 1e-4, 3e-3, log=True),
            "batch_size": trial.suggest_categorical("batch_size", [128, 256, 512])}


def _loss(logits: torch.Tensor, onehot: torch.Tensor) -> torch.Tensor:
    """Mean cross-entropy as a product with the one-hot target: elementwise work and reductions
    only, where CUDA's own cross-entropy may sum with atomics and break a bit-identical rerun."""
    return -(torch.log_softmax(logits, dim=1) * onehot).sum(1).mean()


def fit_predict(p: dict, data: tuple, onehot: torch.Tensor, tr: np.ndarray, te: np.ndarray,
                seed: int, epochs: int | None = None) -> tuple[np.ndarray, int, float]:
    """Fit on `tr` and return (probabilities on `te`, epochs, validation loss). Without `epochs`
    `te` is the validation fold: the fit stops when its loss has not improved for _PATIENCE epochs
    and keeps the best epoch. The design's impute and scale are fitted on `tr` alone."""
    B, M, D = data
    prep = CompetingOutcomesModel._fit_prep(D[tr])
    Dt = torch.tensor(prep(D), dtype=torch.float32, device=B.device)
    torch.manual_seed(seed)
    net = FrameNet(Dt.shape[1], onehot.shape[1], p["phi_width"], p["phi_depth"], p["rho_width"],
                   p["dropout"]).to(B.device)
    opt = torch.optim.AdamW(net.parameters(), lr=p["lr"], weight_decay=p["weight_decay"])
    gen = torch.Generator().manual_seed(seed)
    tr_t, te_t = torch.as_tensor(tr, device=B.device), torch.as_tensor(te, device=B.device)
    best, best_state, best_epoch = np.inf, None, 0
    for epoch in range(1, (epochs or _MAX_EPOCHS) + 1):
        net.train()
        for b in tr_t[torch.randperm(len(tr), generator=gen).to(B.device)].split(p["batch_size"]):
            opt.zero_grad()
            _loss(net(B[b], M[b], Dt[b]), onehot[b]).backward()
            opt.step()
        if epochs is None:
            net.eval()
            with torch.no_grad():
                v = float(_loss(net(B[te_t], M[te_t], Dt[te_t]), onehot[te_t]))
            if v < best:
                best, best_epoch = v, epoch
                best_state = {n: w.detach().clone() for n, w in net.state_dict().items()}
            elif epoch - best_epoch >= _PATIENCE:
                break
    if best_state is not None:
        net.load_state_dict(best_state)
    net.eval()
    with torch.no_grad():
        prob = torch.softmax(net(B[te_t], M[te_t], Dt[te_t]), dim=1).cpu().numpy()
    return prob.astype(float), (best_epoch if epochs is None else epochs), best


def gpu_task(B: np.ndarray, M: np.ndarray, D: np.ndarray, y: np.ndarray, groups: np.ndarray,
             tr: np.ndarray, te: np.ndarray, k: int, n_trials: int, seed: int, fold: int,
             fixed: dict | None = None) -> dict:
    """One (outer fold, seed): the nested search on the fold's training matches and the refit
    scored on its test matches, or, with `fixed` (params and epochs), the refit alone. Module-level
    so joblib ships it; one TPE trial at a time, so the study is a function of its seed."""
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    # cuBLAS is deterministic only with a fixed workspace, read when this process first opens it.
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.use_deterministic_algorithms(True)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if dev.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    data = tuple(torch.as_tensor(a, device=dev) for a in (B, M)) + (D,)
    onehot = torch.nn.functional.one_hot(torch.as_tensor(y, device=dev), k).float()
    t0, trials = time.time(), []
    if fixed is None:
        inner = list(GroupKFold(_N_INNER).split(tr, groups=groups[tr]))
        for a, b in inner:
            assert_no_group_leak(groups[tr], a, b, "match_id")
        n_min = int(np.bincount(y[tr], minlength=k).min())

        def objective(trial: "optuna.Trial") -> float:
            p = space(trial, len(tr), n_min)
            out = [fit_predict(p, data, onehot, tr[a], tr[b], seed) for a, b in inner]
            trial.set_user_attr("epochs", float(np.mean([o[1] for o in out])))
            return float(np.mean([o[2] for o in out]))

        study = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(
            seed=seed + fold, multivariate=True))
        study.optimize(objective, n_trials=n_trials, n_jobs=1, show_progress_bar=False)
        trials = [(t.duration.total_seconds(), t.user_attrs["epochs"]) for t in study.trials]
        fixed = {"params": study.best_params, "epochs": max(1, round(
            study.best_trial.user_attrs["epochs"]))}
    prob = fit_predict(dict(fixed["params"]), data, onehot, tr, te, seed,
                       epochs=fixed["epochs"])[0]
    peak = torch.cuda.max_memory_reserved() / 2**20 if dev.type == "cuda" else 0.0
    return {"fold": fold, "seed": seed, "te": te, "prob": prob, "fixed": fixed,
            "trials": trials, "seconds": time.time() - t0, "peak_reserved_mb": peak}


def _digest(*parts: object) -> str:
    """sha256 over arrays by dtype, shape and bytes, and over anything else by its repr."""
    h = hashlib.sha256()
    for p in parts:
        if isinstance(p, np.ndarray):
            h.update(f"{p.dtype}{p.shape}".encode())
            h.update(np.ascontiguousarray(p).tobytes())
        else:
            h.update(repr(p).encode())
    return h.hexdigest()[:32]


def _task_saved(save: Path | None, *args: Any) -> dict:
    """gpu_task, its result written to `save` the moment it ends, through a temporary file and an
    atomic rename, so a crash leaves either the whole task or nothing."""
    out = gpu_task(*args)
    if save is not None:
        tmp = save.with_suffix(".tmp")
        tmp.write_bytes(pickle.dumps(out))
        os.replace(tmp, save)
    return out


def run_tasks(B: np.ndarray, M: np.ndarray, D: np.ndarray, y: np.ndarray, groups: np.ndarray,
              tasks: list, k: int, n_trials: int, jobs: int, ckpt: Path | None = None,
              load_only: bool = False) -> list:
    """(seed, fold, tr, te, fixed) tasks, one worker process each on the card, in task order, with
    one CPU thread each because the work is on the card. With `ckpt` every task is written there
    under a digest of all it reads, and a rerun loads what it finds instead of fitting it again: an
    interrupted run resumes by content, and a changed input misses the old file. `load_only` fits
    nothing, and a task missing from `ckpt` stops the run instead."""
    saves: list = [None] * len(tasks)
    if ckpt is not None:
        ckpt.mkdir(parents=True, exist_ok=True)
        base = _digest(B, M, D, y, groups, k, n_trials)
        saves = [ckpt / f"{_digest(base, s, f, tr, te, fx)}.pkl" for s, f, tr, te, fx in tasks]
    todo = [i for i, p in enumerate(saves) if p is None or not p.exists()]
    if load_only and todo:
        raise SystemExit(f"STOP: {len(todo)} of {len(tasks)} tasks are not saved in {ckpt}, and a "
                         f"load-only run fits none: the frames, the design or the target changed")
    with parallel_config(backend="loky", inner_max_num_threads=1):
        fresh = Parallel(n_jobs=max(1, min(jobs, len(todo))))(
            delayed(_task_saved)(saves[i], B, M, D, y, groups, tasks[i][2], tasks[i][3], k,
                                 n_trials, tasks[i][0], tasks[i][1], tasks[i][4]) for i in todo)
    got = dict(zip(todo, fresh, strict=True))
    return [got[i] if i in got else pickle.loads(saves[i].read_bytes()) for i in range(len(tasks))]


# ── The statistics ───────────────────────────────────────────────────────
def row_logloss(prob: np.ndarray, y: np.ndarray) -> np.ndarray:
    return -np.log(np.clip(prob[np.arange(len(y)), y], 1e-12, 1.0))


def oof(done: list, n: int, k: int) -> dict:
    """Out-of-fold probabilities per seed, assembled from the tasks."""
    out: dict = {}
    for r in done:
        out.setdefault(r["seed"], np.zeros((n, k)))[r["te"]] = r["prob"]
    return out


def paired(d: np.ndarray, g: np.ndarray, boot: int) -> dict:
    """A paired per-row difference: its match-clustered 95% interval and the MDE at 80% of the
    rule below −MARGIN (the margin plus 2.80 bootstrap SEs)."""
    ci = cluster_bootstrap_mean(d, g, n_boot=boot)
    return {"point": ci.point, "lo": ci.lo, "hi": ci.hi, "se": ci.se,
            "mde80": MARGIN + Z80 * ci.se,
            "below_margin": bool(ci.hi < -MARGIN)}


def decide(seeds: dict, control: dict) -> dict:
    """GPU1: every seed's interval wholly below −MARGIN and a shuffled frame that does not beat
    the delivery (its interval not wholly below zero)."""
    held = [s for s, r in seeds.items() if r["below_margin"]]
    clean = bool(control["hi"] >= 0)
    return {"enters": bool(seeds) and len(held) == len(seeds) and clean,
            "seeds_below_margin": held, "control_clean": clean}


def card_state() -> str:
    """Utilisation, memory in use and the software power cap, read before a measured run."""
    q = subprocess.run(["nvidia-smi", "--query-gpu=utilization.gpu,memory.used,memory.total",
                        "--format=csv,noheader"], capture_output=True, text=True).stdout.strip()
    perf = subprocess.run(["nvidia-smi", "-q", "-d", "PERFORMANCE"], capture_output=True,
                          text=True).stdout
    cap = next((ln.split(":")[1].strip() for ln in perf.splitlines() if "SW Power Cap" in ln), "?")
    return f"{q}; SW power cap {cap}"


# ── The run ──────────────────────────────────────────────────────────────
def population(pop: str, tables: Path, limit: int | None, jobs: int) -> tuple:
    """The table, its frames and the delivery-rung design, row-aligned."""
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


def smoke(pop: str, data: tuple, args: argparse.Namespace) -> list:
    """The cost of the arm, blind: the first --jobs (seed, outer fold) tasks with --trials trials
    each, side by side on the card, and the first task again in another process to check that a
    rerun is bit-identical. Every probability stays in memory and no loss is computed."""
    t, B, M, D, y, k = data
    g = t["match_id"].to_numpy()
    folds = list(GroupKFold(_N_FOLDS).split(B, groups=g))
    tasks = [(s, f, tr, te, None) for s in range(args.seeds)
             for f, (tr, te) in enumerate(folds)][:args.jobs]
    print(f"card before: {card_state()}", flush=True)
    t0 = time.time()
    done = run_tasks(B, M, D, y, g, tasks + tasks[:1], k, args.trials, args.jobs + 1)
    wall = time.time() - t0
    same = np.array_equal(done[0]["prob"], done[-1]["prob"])
    rows = []
    for r in done[:-1]:
        per = [s for s, _ in r["trials"]]
        rows.append({"population": pop, "seed": r["seed"], "fold": r["fold"], "trials": len(per),
                     "task_s": r["seconds"], "trial_s_mean": float(np.mean(per)),
                     "trial_s_max": float(np.max(per)),
                     "epochs_mean": float(np.mean([e for _, e in r["trials"]])),
                     "refit_epochs": r["fixed"]["epochs"], "peak_reserved_mb": r["peak_reserved_mb"],
                     "concurrent": len(tasks) + 1, "wall_s": wall, "rerun_identical": same})
        print(f"  seed {r['seed']} fold {r['fold']}: {r['seconds']:.0f}s for {len(per)} trials (mean "
              f"{np.mean(per):.1f}s, max {np.max(per):.1f}s), refit {r['fixed']['epochs']} epochs, "
              f"peak reserved {r['peak_reserved_mb']:.0f} MB", flush=True)
    print(f"  {len(tasks) + 1} concurrent tasks in {wall:.0f}s; the rerun of the first task is "
          f"{'bit-identical' if same else 'NOT IDENTICAL'}", flush=True)
    return rows


def evaluate(pop: str, data: tuple, args: argparse.Namespace, ckpt: Path | None = None,
             load_only: bool = False) -> list:
    """The arm at every seed, the tuned delivery rung and full model on the same folds, the
    shuffled-frame control at seed 0's configuration, and GPU1. `ckpt` keeps every task, so a
    rerun resumes, and `load_only` reads every task from it and fits none (run_tasks)."""
    t, B, M, D, y, k = data
    g = t["match_id"].to_numpy()
    folds = list(GroupKFold(_N_FOLDS).split(B, groups=g))
    for tr, te in folds:
        assert_no_group_leak(g, tr, te, "match_id")
    if not load_only:
        print(f"card before: {card_state()}", flush=True)
    done = run_tasks(B, M, D, y, g, [(s, f, tr, te, None) for s in range(args.seeds)
                                     for f, (tr, te) in enumerate(folds)], k, args.trials, args.jobs,
                     ckpt, load_only)
    m = _tuned(t, True, args.jobs)
    assert m.n_splits == _N_FOLDS, "the delivery rung and the arm must score the same outer folds"
    l_del = m._row_logloss(m.fit_cv(t, numeric=_KICK_INTRINSIC), t["target"].to_numpy())
    l_full = m._row_logloss(m.fit_cv(t), t["target"].to_numpy())
    enc = {s: row_logloss(p, y) for s, p in oof(done, len(y), k).items()}
    seeds = {s: paired(loss - l_del, g, args.boot) for s, loss in enc.items()}
    perm = np.random.default_rng(0).permutation(len(y))
    fixed = {r["fold"]: r["fixed"] for r in done if r["seed"] == 0}
    shuffled = run_tasks(B[perm], M[perm], D, y, g, [(0, f, tr, te, fixed[f]) for f, (tr, te)
                                                     in enumerate(folds)], k, 0, args.jobs, ckpt,
                         load_only)
    control = paired(row_logloss(oof(shuffled, len(y), k)[0], y) - l_del, g, args.boot)
    verdict = decide(seeds, control)
    rows = [{"population": pop, "block": "seed", "key": str(s), **r} for s, r in seeds.items()]
    rows += [{"population": pop, "block": "control", "key": "shuffled_frame", **control},
             {"population": pop, "block": "head_to_head", "key": "encoder-full",
              **paired(enc[0] - l_full, g, args.boot)},
             {"population": pop, "block": "verdict", "key": "GPU1", "params": str(verdict)}]
    rows += [{"population": pop, "block": "task", "key": f"{r['seed']}/{r['fold']}",
              "seconds": r["seconds"], "peak_reserved_mb": r["peak_reserved_mb"],
              "params": str(r["fixed"])} for r in done]
    for s, r in seeds.items():
        print(f"  seed {s}: {r['point']:+.5f} [{r['lo']:+.5f}, {r['hi']:+.5f}]  MDE80 "
              f"{r['mde80']:.5f}", flush=True)
    print(f"  shuffled frame: {control['point']:+.5f} [{control['lo']:+.5f}, {control['hi']:+.5f}]"
          f"\n  GPU1 ({pop}): {verdict}", flush=True)
    return rows


def main() -> None:
    """Frames and designs per population, then the smoke or the evaluation."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--jobs", type=int, default=5, help="worker processes sharing the card")
    ap.add_argument("--trials", type=int, default=_GPU_TRIALS, help="Optuna trials per task")
    ap.add_argument("--seeds", type=int, default=_SEEDS)
    ap.add_argument("--boot", type=int, default=_BOOT)
    ap.add_argument("--limit", type=int, default=None, help="first N matches: the path, not the numbers")
    ap.add_argument("--tables", type=str, default=str(_PLAN / "tables"))
    ap.add_argument("--out-dir", type=str, default=str(_PLAN / "gpu"))
    ap.add_argument("--build-jobs", type=int, default=23, help="workers reading the frames")
    ap.add_argument("--smoke", action="store_true", help="cost and VRAM per task, no score read")
    ap.add_argument("--tasks", type=str, default=None,
                    help="where every task is kept (default: <out-dir>/tasks)")
    ap.add_argument("--load-only", action="store_true",
                    help="read every task from --tasks and fit none: the recipe's stage")
    args = ap.parse_args()
    clock = time.time()
    out_dir = Path(args.out_dir)
    tail = f"_limit{args.limit}" if args.limit else ""
    published = not tail and (published_dir(out_dir) or
                              out_dir.resolve() == (_PLAN / "gpu").resolve())
    if published and not args.smoke and (args.trials, args.seeds, args.boot) != \
            (_GPU_TRIALS, _SEEDS, _BOOT):
        raise SystemExit(f"STOP: GPU1 is registered at {_GPU_TRIALS} trials and {_SEEDS} seeds, so a "
                         f"run at {(args.trials, args.seeds, args.boot)} may not write frame_gpu.parquet")
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
        rows += smoke(pop, data, args) if args.smoke else evaluate(pop, data, args, tasks,
                                                                   args.load_only)
    path = out_dir / f"frame_gpu{'_smoke' if args.smoke else ''}{tail}.parquet"
    pd.DataFrame(rows).to_parquet(path, index=False)
    print(f"\nwrote {path}  [{time.time() - clock:.0f}s]", flush=True)


if __name__ == "__main__":
    main()
