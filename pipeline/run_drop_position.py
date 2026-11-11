"""run_drop_position — is the staked-win premium about the plan, or about where the ball came down?

s08's four-state grid prices winning the second ball from a zone the plan staked against winning it
on the run. The hypothesis it answers promises value against rescues FROM THE SAME POSITION, and the
raw contrast compares the two states over whatever positions each happens to occupy. If staked wins
tend to drop in better places, part of that gap belongs to the place.

Three readings of the same contrast, on one set of match-clustered draws:

    raw          what the record already publishes, positions unequal
    stratified   the difference taken INSIDE cells of the pitch, averaged with each cell's weight
    adjusted     one least-squares fit with the drop position, its squares and its cross term

A cell holding one state only contributes nothing: it has no contrast to offer, and keeping it
would smuggle the raw comparison back in. The count of rows a stratified estimate can actually use
travels with the result, because a shrinking gap on a shrinking sample means something different
from a shrinking gap on all of it, and the adjusted arm keeps every row precisely so the two can be
told apart.

This does NOT decide whether staking causes anything. Staking is a choice, the sides that stake
differ from the sides that do not, and no adjustment here changes that. What it decides is narrower
and answerable: whether the published premium survives holding the drop position equal.

    python pipeline/run_drop_position.py                          # corners, the four cells published
    python pipeline/run_drop_position.py --cell 5 --out data/derived/smoke/drop_position.parquet
    python pipeline/run_drop_position.py --kinds free_kick_flank
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO / "src"))
sys.path.insert(0, str(_REPO / "pipeline"))
_DER = _REPO / "data" / "derived"

from build_table import kind_tag                     # noqa: E402
from validate import cluster_bootstrap_stats         # noqa: E402

# The pair H3 turns on (argument.md §9): both won the ball, only one had a body there by design.
PAIR = ("won_as_planned", "rescued")
VALUE = "value_xg_after"
# The recipe's sweep and the cells the published record holds: report_figures closes H3 over them.
CELLS = (4.0, 6.0, 8.0, 12.0)


def raw_gap(f: pd.DataFrame) -> float:
    """Mean value of the planned state minus the rescued one, positions as they fall."""
    a = f.loc[f["four_state"] == PAIR[0], VALUE].mean()
    b = f.loc[f["four_state"] == PAIR[1], VALUE].mean()
    return float(a - b)


def stratified_gap(f: pd.DataFrame) -> float:
    """The same difference inside each drop cell, weighted by that cell's share of rows.

    Direct standardisation: the number answers what the gap would be if the two states dropped the
    ball in the same places. Cells holding one state are dropped and the weights renormalise."""
    ok = f[f["four_state"].isin(PAIR)]
    if not len(ok):
        return np.nan
    diffs, weights = [], []
    for _, cell in ok.groupby(["cell_x", "cell_y"]):
        a = cell.loc[cell["four_state"] == PAIR[0], VALUE]
        b = cell.loc[cell["four_state"] == PAIR[1], VALUE]
        if len(a) and len(b):
            diffs.append(a.mean() - b.mean())
            weights.append(len(cell))
    return float(np.average(diffs, weights=weights)) if diffs else np.nan


def adjusted_gap(f: pd.DataFrame) -> float:
    """The coefficient on `is_planned` with the drop position adjusted for, by least squares.

    Position enters as x, y, their squares and their product: enough to bend, few enough terms to
    stay estimable inside a bootstrap draw. A singular design returns NaN, which the cluster
    bootstrap drops rather than counting as a zero."""
    ok = f[f["four_state"].isin(PAIR)]
    if ok["four_state"].nunique() < 2:
        return np.nan
    x, y = ok["drop_x"].to_numpy(float), ok["drop_y"].to_numpy(float)
    planned = (ok["four_state"] == PAIR[0]).to_numpy(float)
    X = np.c_[np.ones(len(ok)), planned, x, y, x ** 2, y ** 2, x * y]
    if not np.isfinite(X).all():
        return np.nan
    try:
        beta, *_ = np.linalg.lstsq(X, ok[VALUE].to_numpy(float), rcond=None)
    except np.linalg.LinAlgError:
        return np.nan
    return float(beta[1])


def covers_zero(lo: float, hi: float) -> bool | None:
    """Whether [lo, hi] holds zero, or None without a finite interval.

    `nan <= 0 <= nan` is False, which would publish EXCLUDES ZERO where there is no interval."""
    return bool(lo <= 0 <= hi) if np.isfinite([lo, hi]).all() else None


def usable_rows(d: pd.DataFrame) -> int:
    """Rows sitting in a cell that holds BOTH states, which is all a stratified estimate can use."""
    cells = d.groupby(["cell_x", "cell_y"])["four_state"].nunique()
    both = set(cells[cells == 2].index)
    return int(d.set_index(["cell_x", "cell_y"]).index.isin(both).sum())


def main() -> None:
    """Score the contrast raw, stratified and adjusted, at each requested cell size."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--boot", type=int, default=2000, help="match-clustered draws")
    ap.add_argument("--cell", type=float, nargs="+", default=list(CELLS),
                    help="drop-cell sides to sweep, StatsBomb yards")
    ap.add_argument("--kinds", type=str, default="corner", help="the s01 population to read")
    ap.add_argument("--table", type=str, default=None)
    ap.add_argument("--out", type=str, default=None)
    args = ap.parse_args()
    t0 = time.time()

    tag = kind_tag(tuple(k.strip() for k in args.kinds.split(",")))
    grant = _DER / f"situation{tag}.parquet"
    src = Path(args.table) if args.table else grant
    if args.out is None and src.resolve() != grant.resolve():
        raise SystemExit(f"STOP: {src.name} is not {grant.name}, so it may not write the published "
                         f"drop_position{tag}.parquet; pass --out")
    if args.out is None and tuple(sorted(args.cell)) != CELLS:
        raise SystemExit(f"STOP: cells {args.cell} are not the recipe's {list(CELLS)}, so they may not "
                         f"write the published drop_position{tag}.parquet; pass --out")
    if not src.exists():
        raise SystemExit(f"STOP: {src.name} absent; run_situation.py --build writes it")
    base = pd.read_parquet(src)
    base = base[base["four_state"].isin(PAIR) & base["drop_x"].notna()
                & base["drop_y"].notna()].copy()
    counts = base["four_state"].value_counts().to_dict()
    print(f"{len(base)} rows with a drop position: {counts}", flush=True)

    rows = []
    for side in args.cell:
        d = base.copy()
        d["cell_x"] = (d["drop_x"] // side).astype(int)
        d["cell_y"] = (d["drop_y"] // side).astype(int)
        usable = usable_rows(d)
        n_cells = d.groupby(["cell_x", "cell_y"]).ngroups
        print(f"\ncells of {side:.0f} yd: {n_cells} cells, {usable} of {len(d)} rows usable by the "
              f"stratified arm", flush=True)
        out = cluster_bootstrap_stats(
            d, {"raw": raw_gap, "stratified": stratified_gap, "adjusted": adjusted_gap},
            n_boot=args.boot)
        for name, r in out.iterrows():
            covers = covers_zero(r["lo"], r["hi"])
            verdict = {True: "covers zero", False: "excludes zero", None: "no interval"}[covers]
            print(f"  {name:12s} {r['point']:+.5f} xG [{r['lo']:+.5f}, {r['hi']:+.5f}]  {verdict}",
                  flush=True)
            rows.append({"block": name, "key": f"cell_{side:.0f}", "point": r["point"],
                         "lo": r["lo"], "hi": r["hi"], "covers_zero": covers, "n": len(d),
                         "n_usable_stratified": usable, "cell_yd": side,
                         "n_planned": counts.get(PAIR[0], 0), "n_rescued": counts.get(PAIR[1], 0)})

    out_path = Path(args.out) if args.out else _DER / f"drop_position{tag}.parquet"
    pd.DataFrame(rows).to_parquet(out_path, index=False)
    print(f"\nwrote {out_path}  [{time.time() - t0:.0f}s]", flush=True)


if __name__ == "__main__":
    main()
