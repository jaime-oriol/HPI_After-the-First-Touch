"""probe_throw_ins — are long throws a third set-piece population, or not this paper's object?

Set pieces include throw-ins and the grammar builds only corners and attacking free kicks, so the
exclusion has to be a measurement rather than a design note. It measures whether the direct shot
from a long throw converts below its own xG as on corners and free kicks, beside the volume and the
segment mix; definitions.md §11 records the decision those figures support.

TWO METHOD CONSTRAINTS, both load-bearing.

The predicate never reads `pass.end_location`, because that point IS the first contact
(definitions.md §5). It reads where the throw was taken and how high, and `pass.length`, which
follows where the ball was met (evidence.md §15), so its length floor reads the defence's first
touch in part, as the free-kick bar does. Selecting on the destination would go the whole way and
define the population by what the defence did to the ball. The destination is used ONLY to
characterise a predicate after the fact, which is exactly what evidence.md §2 did for the free-kick
predicate.

The threshold is SWEPT, not chosen. The free-kick predicate does not transfer: a throw-in is never
a `Ground Pass`, so the height filter selects nothing, and the median throw is longer than the 15 yd
floor, so the length filter selects almost everything. The grid is published whole so the adopted
cell is auditable rather than asserted, and `adopt` is the one rule every reader of it applies.

    python pipeline/probe_throw_ins.py --jobs 16 --boot 2000
    python pipeline/probe_throw_ins.py --limit 20 --jobs 4 --boot 200     # smoke
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed, parallel_config

_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
_DER = _REPO / "data" / "derived"

import loader_sb as L                                        # noqa: E402
from constants import BOX_X, BOX_Y, MIN_CELL                 # noqa: E402
from eda_build import COMBINED_KINDS, kind_inputs            # noqa: E402
from s01_phases import SetPieceGrammar, _is_xy, is_own_goal  # noqa: E402
from s03_value import _league_index                          # noqa: E402
# Reused rather than reimplemented: one obs/exp with one interval convention across the paper.
from describe_frontier import clustered_oe, diagnose, headline, _fmt, _ORDER   # noqa: E402

X_GRID = (60.0, 80.0, 90.0, 102.0)
LEN_GRID = (15.0, 20.0, 25.0, 30.0)


def throw_in_mask(ev: pd.DataFrame, x_min: float, len_min: float, high_only: bool) -> pd.Series:
    """Throw-ins that are a DELIVERY rather than a restart, by position, height and length.

    Mirrors the shape of `s01_phases._delivery_mask`'s free-kick branch (taken far enough up,
    long enough to be a delivery) with the height test inverted. On a free kick the rule excludes
    `Ground Pass`; on a throw-in that value never occurs, so the equivalent discriminator is
    keeping `High Pass` only."""
    m = ev["pass.type.name"] == "Throw-in"
    x = ev["location"].map(lambda v: float(v[0]) if _is_xy(v) else np.nan)
    m &= x >= x_min
    m &= pd.to_numeric(ev.get("pass.length"), errors="coerce") >= len_min
    if high_only:
        m &= ev.get("pass.height.name") == "High Pass"
    return m.fillna(False)


def _ends_in_box(ev: pd.DataFrame) -> pd.Series:
    """Did the throw end inside the box. CHARACTERISATION ONLY: this reads the first contact,
    so it judges a predicate after the fact and never selects."""
    end = ev.get("pass.end_location")
    if end is None:
        return pd.Series(False, index=ev.index)
    x = end.map(lambda v: float(v[0]) if _is_xy(v) else np.nan)
    y = end.map(lambda v: float(v[1]) if _is_xy(v) else np.nan)
    return ((x >= BOX_X) & y.between(*BOX_Y)).fillna(False)


# ── One match ───────────────────────────────────────────────────────────
def grid_match(match_id: int) -> list:
    """(x_min, len_min, high_only) -> n selected and n ending in box, for one match."""
    ev = L.load_events(match_id)
    inbox = _ends_in_box(ev)
    out = []
    for x_min in X_GRID:
        for len_min in LEN_GRID:
            for high_only in (False, True):
                m = throw_in_mask(ev, x_min, len_min, high_only)
                out.append({"x_min": x_min, "len_min": len_min, "high_only": high_only,
                            "n": int(m.sum()), "n_in_box": int((m & inbox).sum())})
    return out


def delivery_counts(g: SetPieceGrammar) -> dict:
    """The paper's two populations in one match, corners and flank free kicks, by the rule that
    builds their shots.

    These are the denominators of the side-by-side comparison, counted by the grammar so they
    cannot disagree with the populations whose shots they divide."""
    built = g.deliveries()["set_piece_type"].value_counts()
    return {k: int(built.get(k, 0)) for k in COMBINED_KINDS}


def frontier_match(match_id: int, league: str, x_min: float, len_min: float,
                   high_only: bool) -> tuple:
    """(n throw-in deliveries, shot rows, corner and free-kick counts) for one match.

    Runs the SHIPPED grammar. `shots()` takes a delivery row and never consults
    `_delivery_mask`, so the five segments, the window and the clearing-contact rule are the ones
    s01 already defines; this driver contributes a population, not a second taxonomy."""
    ev = L.load_events(match_id)
    g = SetPieceGrammar(ev, kinds=COMBINED_KINDS)
    sel = g.ev[throw_in_mask(g.ev, x_min, len_min, high_only)]
    rows = []
    for _, d in sel.iterrows():
        d = d.copy()
        d["set_piece_type"] = "throw_in"
        rows += [{**s.__dict__, "match_id": match_id, "league": league} for s in g.shots(d)]
    return len(sel), rows, delivery_counts(g)


# ── Report ──────────────────────────────────────────────────────────────
def sensitivity(rows: list) -> pd.DataFrame:
    """The grid, with the in-box share as the precision column.

    `in_box_pct` is what the predicate is judged on and is NOT part of it. The free-kick predicate's
    own in-box share (evidence.md §2) is the bar a throw-in predicate has to approach before the two
    are comparable objects."""
    g = pd.DataFrame(rows).groupby(["x_min", "len_min", "high_only"]).sum().reset_index()
    g["in_box_pct"] = 100 * g["n_in_box"] / g["n"].replace(0, np.nan)
    return g.sort_values("in_box_pct", ascending=False)


def adopt(grid: pd.DataFrame) -> pd.Series:
    """The adopted cell: the highest in-box share among cells holding at least 5 x MIN_CELL throws.

    The one copy of the rule. eda_throw_ins and report_figures read the adopted cell back off the
    persisted grid through this, so the three can never select different throws."""
    g = grid.sort_values("in_box_pct", ascending=False)
    return g[g["n"] >= 5 * MIN_CELL].iloc[0]


def comparison_row(pop: str, sub: pd.DataFrame, n_sp: int, boot: int) -> dict:
    """One population's line of the side-by-side comparison: its volume and its direct shot.

    Shots are shots: an own goal is a goal and no shot (`s01_phases.is_own_goal`), so it adds to
    `goals_per_100` and to no shot count, share or ratio over xG."""
    shots = sub[~is_own_goal(sub)]
    d = shots[shots["segment"] == "direct"]
    # The whole frame, own goals in: `clustered_oe` sets them aside itself, and resampling the same
    # matches as describe_frontier is what makes the two intervals the same draw.
    oe = clustered_oe(sub, "segment", ["direct"], boot).loc["direct"]
    return {"population": pop, "set_pieces": n_sp, "shots": len(shots),
            "shots_per_100": 100 * len(shots) / n_sp,
            "goals_per_100": 100 * int(sub["is_goal"].sum()) / n_sp,
            "aft_shot_pct": 100 * shots["after_first_touch"].mean(),
            "direct_shots": len(d), "direct_obs_exp": oe["obs_exp"],
            "lo": oe["lo"], "hi": oe["hi"], "excludes_1": oe["excludes_1"]}


def record_tag(limit: int | None, override: bool, shots: Path) -> str:
    """The record's name tag. A subset, an override or a shot table other than the grant's (the
    comparison divides it) each keep a run off the published name; a subset's tag already does, so
    its record is the `_limitN` one eda_throw_ins --limit N reads."""
    grant_shots = shots.resolve() == kind_inputs("corner")[1].resolve()
    return ((f"_limit{limit}" if limit else "") + ("_override" if override else "")
            + ("" if grant_shots or limit else f"_{shots.stem}"))


def main() -> None:
    """Sweep the predicate, run the frontier on the adopted cell, compare the three populations."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None, help="first N matches only (smoke test)")
    ap.add_argument("--jobs", type=int, default=16)
    ap.add_argument("--boot", type=int, default=2000)
    ap.add_argument("--x-min", type=float, default=None,
                    help="override the adopted cell's taker-x floor; forces high_only, and the "
                         "run is then not canonical")
    ap.add_argument("--len-min", type=float, default=None,
                    help="override the adopted cell's length floor; the run is then not canonical")
    ap.add_argument("--shots", type=str, default=str(kind_inputs("corner")[1]),
                    help="the built populations, for the side-by-side comparison")
    ap.add_argument("--out", type=str, default=None)
    args = ap.parse_args()

    mids = L.list_matches()["match_id"].tolist()
    if args.limit:
        mids = mids[:args.limit]
    leagues = _league_index()
    override = args.x_min is not None or args.len_min is not None
    tag = record_tag(args.limit, override, Path(args.shots))
    out = args.out or str(_DER / f"throw_in_probe{tag}.parquet")

    print("=" * 78)
    print(f"THROW-INS AS A THIRD POPULATION?  ({len(mids)} matches)")
    print("=" * 78)

    # ── 1. The predicate, swept ─────────────────────────────────────────
    with parallel_config(backend="loky", inner_max_num_threads=1):
        grid = Parallel(n_jobs=args.jobs)(delayed(grid_match)(m) for m in mids)
    sens = sensitivity([r for part in grid for r in part])
    print("\nPredicate sweep. `in_box_pct` JUDGES the predicate and is not part of it.\n")
    print(_fmt(sens.head(12).set_index(["x_min", "len_min", "high_only"])))

    best = adopt(sens)
    x_min = args.x_min if args.x_min is not None else float(best["x_min"])
    len_min = args.len_min if args.len_min is not None else float(best["len_min"])
    high_only = bool(best["high_only"]) if args.x_min is None else True
    cell = sens[(sens["x_min"] == x_min) & (sens["len_min"] == len_min)
                & (sens["high_only"] == high_only)]
    stats = (f"n={int(cell['n'].iloc[0])}, in_box={cell['in_box_pct'].iloc[0]:.1f}%"
             if len(cell) else "a cell outside the swept grid")
    print(f"\nadopted: x >= {x_min:.0f}, length >= {len_min:.0f}, high_only={high_only}  -> {stats}")

    # ── 2. The frontier on that population ──────────────────────────────
    with parallel_config(backend="loky", inner_max_num_threads=1):
        res = Parallel(n_jobs=args.jobs)(
            delayed(frontier_match)(m, leagues.get(int(m)), x_min, len_min, high_only)
            for m in mids)
    n_deliv = sum(n for n, _, _ in res)
    shots = pd.DataFrame([r for _, rows, _ in res for r in rows])
    n_built = {k: sum(c[k] for _, _, c in res) for k in COMBINED_KINDS}
    n_shots = 0 if shots.empty else int((~is_own_goal(shots)).sum())
    print(f"\n{n_deliv} long throws produced {n_shots} attacking shots")
    if n_shots < MIN_CELL:
        raise SystemExit(f"STOP: {n_shots} shots is under the publication floor — "
                         f"nothing here is interpretable")

    print("  diagnostics:", diagnose(shots))
    seg = clustered_oe(shots, "segment", _ORDER, args.boot)
    print("\nsegments (obs/exp interval is match-clustered)")
    print(_fmt(seg))
    print("\n  Q1 headline")
    print(_fmt(headline(shots, args.boot)))

    # ── 3. The comparison that decides ──────────────────────────────────
    # Cut to the same matches as the counts, or a --limit run divides a season's shots by forty
    # matches' deliveries.
    built = pd.read_parquet(args.shots)
    built = built[built["match_id"].isin(mids)]
    rows = [comparison_row(pop, sub, n_sp, args.boot)
            for pop, sub, n_sp in [*[(k, built[built.set_piece_type == k], n_built[k])
                                     for k in COMBINED_KINDS], ("throw_in", shots, n_deliv)]]
    comp = pd.DataFrame(rows).set_index("population")
    print("\n" + "=" * 78)
    print("THE COMPARISON. Does the direct shot under-convert on all three?")
    print("=" * 78)
    print(_fmt(comp))

    # An obs/exp is only as strong as its NUMERATOR. MIN_CELL gates the shot count, which is the
    # wrong denominator to gate on here: a cell can clear MIN_CELL shots and still rest on one
    # goal, and a ratio built on one goal is not a finding whatever its interval does. So the
    # verdict names the goal count and refuses to be categorical below the same floor.
    direct_goals = int(shots[(shots["segment"] == "direct") & shots["is_goal"]
                             & ~is_own_goal(shots)].shape[0])
    excl = bool(comp.loc["throw_in", "excludes_1"] and comp.loc["throw_in", "hi"] < 1.0)
    print(f"\nThe long-throw direct shot's obs/exp interval "
          f"{'excludes' if excl else 'covers'} 1.0, on {direct_goals} goal(s) "
          f"over {int(comp.loc['throw_in', 'direct_shots'])} shots.")
    if excl and direct_goals < MIN_CELL:
        print(f"  FRAGILE: under {MIN_CELL} goals in the numerator. Report the interval, do NOT "
              f"lead on it, and do not place it beside the corner figure as if they were equally "
              f"well determined.")
    print(f"\nWhat the volume says, which does not depend on that cell: long throws produce "
          f"{comp.loc['throw_in', 'shots_per_100']:.1f} shots per 100 deliveries against "
          f"{comp.loc['free_kick_flank', 'shots_per_100']:.1f} for flank free kicks, and "
          f"{comp.loc['throw_in', 'aft_shot_pct']:.1f}% of them arrive after the first contact "
          f"against {comp.loc['corner', 'aft_shot_pct']:.1f}% on corners.")

    Path(out).parent.mkdir(parents=True, exist_ok=True)
    pd.concat([sens.assign(block="grid"),
               seg.reset_index().assign(block="segment"),
               comp.reset_index().assign(block="comparison")],
              ignore_index=True).to_parquet(out)
    print(f"\nsaved {out}  [gitignored]")


if __name__ == "__main__":
    main()
