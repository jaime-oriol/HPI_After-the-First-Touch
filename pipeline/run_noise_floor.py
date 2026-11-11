"""run_noise_floor — what shuffling a block that carries nothing costs, measured on pure noise.

The permuted control of run_validation charges every block of the design the cost of shuffling it across corners, and
that cost has a floor even for a block with no information. No block of the design is empty on every row (`kick_spot`
varies on the crosses that deliver a short corner, struck from the wing), so the floor is built by construction
(`s03.noise_floor`, fixed before it ran in data/derived/prereg_2026-11-04/deviation_noise_floor_2026-11-04.md): blocks
of independent standard-normal columns, 30 draws for each width a block of the design has, on the corner rows of the
two-population table, tuned as run_validation tunes them (`s03.LADDER_TUNE`).

A block is read against the floor of its own width: its permuted cost is told from noise only when it exceeds the 95th
percentile of the draws, and, as `permuted_ablation` says, it is evidence only where the dropped cost agrees with it.

    python pipeline/run_noise_floor.py --jobs 20
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO / "src"))
sys.path.insert(0, str(_REPO / "pipeline"))

from build_table import kind_tag                                   # noqa: E402
from eda_build import COMBINED_KINDS                               # noqa: E402
from s03_value import _BLOCKS, LADDER_TUNE, CompetingOutcomesModel     # noqa: E402

_DER = _REPO / "data" / "derived"
_TWO_POP_TABLE = f"corners{kind_tag(COMBINED_KINDS)}.parquet"
POPULATION = "corner"


def floor_rows(table: pd.DataFrame, floors: pd.DataFrame, draws: int, repeats: int) -> pd.DataFrame:
    """The record's long schema: one `floor` row per width, and one `group` row per block of the design with the width
    it has in this table and the floor of that width, so a prose figure names the block and not the width."""
    base = {"population": POPULATION, "n_draws": draws, "n_repeats": repeats}
    rows = [{**base, "block": "floor", "key": f"w{int(r.width)}", "n_cols": int(r.width), "mean": r.mean, "sd": r.sd,
             "p95": r.p95, "max": r.max} for r in floors.itertuples()]
    by_width = floors.set_index("width")
    for group, cols in _BLOCKS.items():
        w = sum(c in table.columns for c in cols)
        if w:
            f = by_width.loc[w]
            rows.append({**base, "block": "group", "key": group, "n_cols": w, "mean": f["mean"], "sd": f["sd"],
                         "p95": f["p95"], "max": f["max"]})
    return pd.DataFrame(rows)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--table", type=str, default=str(_DER / _TWO_POP_TABLE))
    ap.add_argument("--draws", type=int, default=30, help="blocks drawn for each width")
    ap.add_argument("--repeats", type=int, default=3, help="permutations of each block")
    ap.add_argument("--jobs", type=int, default=8, help="the tuning grid and the draws")
    ap.add_argument("--out-dir", type=str, default=str(_DER))
    args = ap.parse_args()

    path = Path(args.table)
    grant = (_DER / _TWO_POP_TABLE).resolve()
    if Path(args.out_dir).resolve() == _DER.resolve() and (
            path.resolve() != grant or (args.draws, args.repeats) != (30, 3)):
        raise SystemExit("STOP: a run on another table, or at other draws, may not write the published "
                         "noise_floor.parquet; pass --out-dir")
    if not path.exists():
        raise SystemExit(f"STOP: {path} absent; pipeline/build_table.py writes it")
    tab = pd.read_parquet(path)
    sub = tab[tab["set_piece_type"] == POPULATION]
    print(f"{len(sub)} {POPULATION} deliveries, {sub['match_id'].nunique()} matches", flush=True)

    m = CompetingOutcomesModel()
    t = m.tune_cv(sub, n_jobs=args.jobs, **LADDER_TUNE)
    m.apply_best_params(t)
    print(f"tuned on nested grouped CV: {t.best_params}", flush=True)
    floors = m.noise_floor(sub, n_draws=args.draws, n_repeats=args.repeats, n_jobs=args.jobs)
    print(floors.round(5).to_string(index=False), flush=True)

    out = Path(args.out_dir) / "noise_floor.parquet"
    out.parent.mkdir(parents=True, exist_ok=True)
    floor_rows(sub, floors, args.draws, args.repeats).to_parquet(out, index=False)
    print(f"\nwrote {out}", flush=True)


if __name__ == "__main__":
    main()
