"""run_refit_draws — the refit draws behind the decisive contrasts, kept whole.

The published refit intervals keep two numbers per contrast, the 2.5th and 97.5th percentiles of
300 match resamples with the model rebuilt inside each. The rule reads them in normal form, the
observed point plus or minus half that width (s03_value.step_refit_interval says why), and that
is the normal-theory interval only if the draws are symmetric. This driver re-runs the contrasts
that carry a paper claim with the recipe's seed, draw count and hyperparameters and keeps every
draw, so report_figures can publish the SD, the skew and the centred-percentile interval beside
the normal one. The contract was fixed before the first run: the draws count as symmetric enough
when the centred-percentile interval gives the same verdict as the normal one on every contrast.

It also gives the head-to-head its refit interval: `delivery_vs_360`, the contrast the headline
stands on, has only its fixed-fit one in the validation record.

    head_to_head   360-only log-loss minus the delivery design's (positive: delivery won), on
                   corner, free_kick_flank and pooled (the two-population table, split as
                   validation --by-type splits it)
    step_360       delivery rung minus the full model (positive: the 360 helped), same populations
    geometry_step  geometry360 minus delivery on each sufficiency grant (negative: the frame won)

Each contrast must reproduce its published point, and its published percentiles where the record
keeps them, or the draws belong to a different estimator and the driver stops.

    python pipeline/run_refit_draws.py --jobs 20
    python pipeline/run_refit_draws.py --limit 40 --boot 5 --jobs 8 --out data/derived/smoke/refit_draws.parquet
"""
from __future__ import annotations

import argparse
import sys
import time
from collections.abc import Callable
from pathlib import Path

import numpy as np
import pandas as pd

_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO / "src"))
sys.path.insert(0, str(_REPO / "pipeline"))
_DER = _REPO / "data" / "derived"

from build_table import kind_tag                         # noqa: E402
from eda_build import COMBINED_KINDS                     # noqa: E402
from run_sufficiency import grant_table, logistic_rows   # noqa: E402
from run_validation import _TWO_POP_TABLE, _tuned        # noqa: E402
from s03_value import CompetingOutcomesModel             # noqa: E402
from validate import refit_cluster_draws                 # noqa: E402

# The recipe's refit: 300 resamples at seed 0 (s03_value.step_refit_interval, run_sufficiency).
N_BOOT, SEED = 300, 0
_SUFF_TAGS = {kind: kind_tag((kind,)) for kind in COMBINED_KINDS}


def validation_frames(limit: int | None) -> dict[str, pd.DataFrame]:
    """The three validation populations, each exactly the frame run_validation reported on; with
    `limit`, off the N-match two-population table a smoke run built."""
    name = _TWO_POP_TABLE.replace(".parquet", f"_limit{limit}.parquet") if limit else _TWO_POP_TABLE
    two = pd.read_parquet(_DER / name)
    frames = {kind: sub for kind, sub in two.groupby("set_piece_type")}
    frames["pooled"] = two
    return frames


def _published_validation(population: str, records: Path = _DER) -> pd.DataFrame:
    v = pd.read_parquet(records / "validation.parquet")
    return v[v["population"] == population]


def _check(name: str, point: float, draws: np.ndarray, want_point: float,
           want_pct: tuple | None) -> None:
    """Stop unless the rebuilt contrast is the published one."""
    if not np.isclose(point, want_point, rtol=0.0, atol=1e-9):
        raise SystemExit(f"STOP: {name} point {point} does not reproduce the published {want_point}")
    if want_pct is not None:
        pct = np.percentile(draws, [2.5, 97.5])
        if not np.allclose(pct, want_pct, rtol=0.0, atol=1e-9):
            raise SystemExit(f"STOP: {name} percentiles {pct} do not reproduce the published "
                             f"{want_pct}")


def _draw(frame: pd.DataFrame, stat: Callable, boot: int, jobs: int) -> tuple[float, np.ndarray]:
    point, draws, _ = refit_cluster_draws(frame, stat, n_boot=boot, seed=SEED, n_jobs=jobs)
    return float(point[0]), draws[:, 0]


def run(limit: int | None, boot: int, jobs: int, check: bool, records: Path = _DER) -> pd.DataFrame:
    """Every contrast's point and draws, in the long shape report_figures reads, checked against the
    validation and sufficiency records in `records`."""
    rows = []

    def keep(population: str, contrast: str, point: float, draws: np.ndarray) -> None:
        rows.extend({"population": population, "contrast": contrast, "point": point,
                     "draw": i, "value": float(v)} for i, v in enumerate(draws))
        pct = np.percentile(draws, [2.5, 97.5])
        print(f"  {population:22s} {contrast:14s} {point:+.5f}  percentile "
              f"[{pct[0]:+.5f}, {pct[1]:+.5f}]  sd {draws.std(ddof=1):.5f}  ({len(draws)} draws)")

    for population, frame in validation_frames(limit).items():
        t0 = time.time()
        m = _tuned(frame, True, jobs)
        pub = _published_validation(population, records) if check else None
        point, draws = _draw(frame, lambda f, m=m: float(m.head_to_head_rows(f).mean()), boot, jobs)
        if check:
            h = pub[pub["block"] == "head_to_head"].iloc[0]
            _check(f"{population} head_to_head", point, draws, float(h["delta"]), None)
        keep(population, "head_to_head", point, draws)
        point, draws = _draw(frame, lambda f, m=m: float(m.step_rows(f).mean()), boot, jobs)
        if check:
            r = pub[pub["block"] == "refit"].iloc[0]
            _check(f"{population} step_360", point, draws, float(r["delta"]),
                   (float(r["ll_lo"]), float(r["ll_hi"])))
        keep(population, "step_360", point, draws)
        print(f"  [{population}: {time.time() - t0:.0f}s]")

    for kind, tag in _SUFF_TAGS.items():
        t0 = time.time()
        frame = grant_table(limit, tag)
        m = CompetingOutcomesModel(n_splits=5)
        m.apply_best_params(m.tune_cv(frame, n_jobs=jobs))

        def geometry(f: pd.DataFrame, m: CompetingOutcomesModel = m) -> float:
            r_delivery, r_geometry = logistic_rows(f, m)
            return float((r_geometry - r_delivery).mean())

        point, draws = _draw(frame, geometry, boot, jobs)
        if check:
            s = pd.read_parquet(records / f"sufficiency{tag}.parquet")
            r = s[(s["population"] == "grant") & (s["block"] == "refit")
                  & (s["key"] == "geometry360-delivery")].iloc[0]
            _check(f"{kind} geometry_step", point, draws, float(r["point"]),
                   (float(r["step_lo"]), float(r["step_hi"])))
        keep(kind, "geometry_step", point, draws)
        print(f"  [{kind} grant: {time.time() - t0:.0f}s]")
    return pd.DataFrame(rows)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--jobs", type=int, default=20,
                    help="workers for the tuning grid and the resamples; output is identical "
                         "at any count (validate.refit_cluster_draws)")
    ap.add_argument("--limit", type=int, default=None, help="first N matches only (smoke)")
    ap.add_argument("--boot", type=int, default=N_BOOT)
    ap.add_argument("--out", type=str, default=None)
    ap.add_argument("--records", type=str, default=str(_DER),
                    help="the validation and sufficiency records the draws must reproduce (default: "
                         "data/derived); the pass under partition F reads its own")
    args = ap.parse_args()
    canonical = args.limit is None and args.boot == N_BOOT
    if not canonical and args.out is None:
        raise SystemExit("STOP: a --limit or --boot run needs --out; refit_draws.parquet is the "
                         "grant's record")
    t0 = time.time()
    print(f"REFIT DRAWS: {args.boot} match resamples per contrast, seed {SEED}, {args.jobs} workers")
    out = run(args.limit, args.boot, args.jobs, check=canonical, records=Path(args.records))
    path = Path(args.out) if args.out else _DER / "refit_draws.parquet"
    path.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(path, index=False)
    print(f"wrote {path} ({len(out)} rows) in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
