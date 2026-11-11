"""run_situation — Build the SITUATION table and print the four-states grid (S08).

Per cleared corner: the second-ball drop's own freeze-frame (s08) -> local
situation features + the four coach states crossing STAKED (the plan put a body at
the drop) with WON (the attacking team took the next touch) + the value trio
(after-clearance xG/goals, pure Φ). The value block is what makes the grid a
finding rather than a census: it prices each state, with every interval
match-clustered (validate.cluster_bootstrap_stats).

The coverage block prints how many rows carry each anti-leakage layer: `won` comes from
the event stream, `staked` ONLY from the delivery freeze-frame (T = delivery, no
look-ahead) scored against the drop a next touch contested, and the situation features
need a 360 at the drop itself. `four_state` exists only where BOTH the plan and the
outcome are known.

Beside the table it writes what the three chosen situation constants cut on it (the radii
record): where the drops land, the share of the scored bodies inside each NEAR_RADII ring,
and the nearest staker's distance from a contested drop.

    python pipeline/run_situation.py --build --jobs 20            # the full grant
    python pipeline/run_situation.py --build --limit 20 --jobs 8  # smoke
    python pipeline/run_situation.py --build --jobs 20 --kinds free_kick_flank   # situation_<kind>
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import pandas as pd

_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO / "src"))
sys.path.insert(0, str(_REPO / "pipeline"))

import loader_sb as L                                   # noqa: E402
from build_table import kind_tag                        # noqa: E402
from constants import MIN_CELL, NEAR_RADII, STAKE_RADIUS   # noqa: E402
from s08_situation import FOUR_STATES, build_situation_table   # noqa: E402
from validate import cluster_bootstrap_stats            # noqa: E402


STATES = tuple(FOUR_STATES.values())   # print order, from the map that defines them

# The two contrasts argument.md Q2 turns on: plan-and-win against react-and-win (both
# won the ball, only one had a body there by design), and what the stake costs when lost.
CONTRASTS = (("won_as_planned", "rescued"), ("staked_but_lost", "conceded_open"))
# s08's ring columns, inner and outer, in NEAR_RADII's order.
_RINGS = (("n_att_near3", "n_def_near3"), ("n_att_near5", "n_def_near5"))


def _state_mean(frame: pd.DataFrame, state: str, col: str) -> float:
    """Mean of `col` over the rows in one four-state cell."""
    return frame.loc[frame["four_state"] == state, col].mean()


def _state_value_table(tab: pd.DataFrame, n_boot: int) -> pd.DataFrame:
    """Per-state after-clearance value plus the argument.md Q2 contrasts, match-clustered.

    A state's mean is a rate within a subset and a contrast is a difference between
    two such subsets, so every statistic rides ONE set of cluster draws
    (validate.cluster_bootstrap_stats). Subtracting two marginal intervals is not a
    test of their difference, and this repo bars it everywhere else."""
    stats = {}
    for st in STATES:
        stats[f"{st}:xg"] = lambda f, s=st: _state_mean(f, s, "value_xg_after")
        stats[f"{st}:goal%"] = lambda f, s=st: 100 * _state_mean(f, s, "value_goal_after")
    for a, b in CONTRASTS:
        stats[f"{a}-{b}:xg"] = (lambda f, a=a, b=b:
                                _state_mean(f, a, "value_xg_after") - _state_mean(f, b, "value_xg_after"))
    return cluster_bootstrap_stats(tab[tab["four_state"].notna()], stats, n_boot=n_boot)


def value_record(val: pd.DataFrame, counts: pd.Series) -> pd.DataFrame:
    """The per-state prices and the two contrasts, in a long table with their cell counts.

    Persisted rather than printed, because `evidence.md` §14's raw contrasts are read off it
    (`run_drop_position` prices the same contrast at equal drop position from the situation
    table): a figure with no artefact is one no test can bind and nothing can notice moving. The
    counts travel alongside because MIN_CELL is a publication floor on the cell, not on the
    interval.

    `covers_zero` is None on a degenerate cell rather than False. An empty cell bootstraps to
    a NaN bound, and `nan <= 0 <= nan` evaluates False, which reads as EXCLUDES ZERO: the
    strongest claim this record can make, asserted where there is no interval at all. The
    grant never reaches that state; a --limit run does."""
    rows = []
    for k, r in val.iterrows():
        key, metric = k.split(":")
        finite = pd.notna(r["lo"]) and pd.notna(r["hi"])
        rows.append(dict(block="contrast" if "-" in key else "state", key=key, metric=metric,
                         point=r["point"], lo=r["lo"], hi=r["hi"],
                         n=int(counts.get(key, 0)) if "-" not in key else None,
                         covers_zero=bool(r["lo"] <= 0 <= r["hi"]) if finite else None))
    return pd.DataFrame(rows)


def radii_record(tab: pd.DataFrame) -> pd.DataFrame:
    """What the three chosen situation constants cut on this table, in a long (block, key, value,
    n) schema report_figures binds. `drop` is where the drops land, DROP_ANCHOR_SB being its
    median; `near` the share of the scored bodies, actors out, inside each NEAR_RADII ring;
    `stake` the nearest staker's distance from a contested drop, which STAKE_RADIUS cuts."""
    d = tab[tab["drop_x"].notna()]
    rows = [dict(block="drop", key=k, value=float(v), n=len(d)) for k, v in (
        ("median_x", d["drop_x"].median()), ("median_y", d["drop_y"].median()),
        ("mean_x", d["drop_x"].mean()), ("mean_y", d["drop_y"].mean()),
        ("q25_x", d["drop_x"].quantile(0.25)), ("q75_x", d["drop_x"].quantile(0.75)))]
    f = tab[tab["n_scored"].notna()]
    bodies = int(f["n_scored"].sum())
    rows += [dict(block="near", key=f"share_within_{r:g}",
                  value=float(f[list(cols)].to_numpy(float).sum() / bodies) if bodies else
                  float("nan"), n=bodies) for r, cols in zip(NEAR_RADII, _RINGS, strict=True)]
    s = tab["staker_dist"].dropna().astype(float)
    rows += [dict(block="stake", key=k, value=float(v), n=len(s)) for k, v in (
        ("p25", s.quantile(0.25)), ("median", s.median()),
        (f"share_within_{STAKE_RADIUS:g}", (s <= STAKE_RADIUS).mean()))]
    return pd.DataFrame(rows)


def main() -> None:
    """Build (or load) the S08 SITUATION table and print coverage + the four-states grid.

    Both narrowing flags tag the DEFAULT output name, which is the rule build_table states.
    `--source` is in it because it cuts the population as hard as `--limit`: without the tag a
    `--source serie_a_sample` run republishes the grant artefact over 30 matches and nothing
    downstream can tell. An explicit --out is honoured as given. The states artefact is suffixed
    on the stem rather than substring-replaced, because under an --out not ending in the literal
    ".parquet" a replace is a no-op and both writes land on one path."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=str, default=None,
                    help="output parquet (default: situation[_kind][_source][_limitN].parquet)")
    ap.add_argument("--build", action="store_true", help="rebuild instead of loading the cache")
    ap.add_argument("--limit", type=int, default=None, help="first N matches only")
    ap.add_argument("--jobs", type=int, default=1, help="workers for the build (events-only: 20 is safe)")
    ap.add_argument("--boot", type=int, default=2000, help="cluster-bootstrap draws for the value block")
    ap.add_argument("--source", type=str, default=None,
                    help="loader_sb source: a Big-5 league, SERIE_A_SAMPLE, or omit for all Big-5")
    ap.add_argument("--kinds", type=str, default="corner", help="the s01 population to situate")
    args = ap.parse_args()

    kinds = tuple(k.strip() for k in args.kinds.split(","))
    tag = kind_tag(kinds) + (f"_{args.source.lower()}" if args.source else "") + \
          (f"_limit{args.limit}" if args.limit else "")
    out = args.out or str(_REPO / "data" / "derived" / f"situation{tag}.parquet")

    if args.build or not Path(out).exists():
        mids = L.list_matches(args.source)["match_id"].tolist()
        t0 = time.time()
        tab = build_situation_table(mids[:args.limit] if args.limit else mids, n_jobs=args.jobs,
                                    kinds=kinds)
        print(f"built in {time.time() - t0:.0f}s ({args.jobs} worker(s))")
        Path(out).parent.mkdir(parents=True, exist_ok=True)
        tab.to_parquet(out)
    else:
        tab = pd.read_parquet(out)

    n = len(tab)
    print(f"SITUATION: {n} cleared corners, {tab['match_id'].nunique()} matches\n")

    print("coverage (anti-leakage layers):")
    print(f"  won-second-ball known (event stream): {int(tab['won_second_ball'].notna().sum())}  "
          f"({100 * tab['won_second_ball'].notna().mean():.0f}%)")
    print(f"  delivery 360 (stakers counted):       {int(tab['n_stakers'].notna().sum())}  "
          f"({100 * tab['n_stakers'].notna().mean():.0f}%)")
    print(f"  staked known (360 AND next touch):    {int(tab['staked'].notna().sum())}  "
          f"({100 * tab['staked'].notna().mean():.0f}%)")
    print(f"  situation frame (drop 360):           {int(tab['n_seen'].notna().sum())}  "
          f"({100 * tab['n_seen'].notna().mean():.0f}%)")
    print(f"  four-states known (staked AND won):   {int(tab['four_state'].notna().sum())}  "
          f"({100 * tab['four_state'].notna().mean():.0f}%)")

    print("\nWON second ball (attacking team took the next touch):")
    print(f"  {(tab['won_second_ball'].value_counts(normalize=True) * 100).round(1).to_dict()}")

    fs = tab["four_state"].dropna()
    if len(fs):
        print("\nFOUR STATES (staked x won):")
        vc = fs.value_counts()
        # fixed order: plan-worked -> speed-problem -> lucky -> design-failure
        for st in STATES:
            print(f"  {st:16s} {int(vc.get(st, 0)):4d}  ({100 * vc.get(st, 0) / len(fs):.1f}%)")

        print(f"\nVALUE per state (after-clearance xG / goal rate, match-clustered {args.boot}-draw CIs):")
        val = _state_value_table(tab, args.boot)
        for st in STATES:
            xg, gl = val.loc[f"{st}:xg"], val.loc[f"{st}:goal%"]
            print(f"  {st:16s} xG {xg['point']:.4f} [{xg['lo']:.4f}, {xg['hi']:.4f}]   "
                  f"goals {gl['point']:.2f}% [{gl['lo']:.2f}, {gl['hi']:.2f}]  (n={int(vc.get(st, 0))})")
        print(f"  a cell under n={MIN_CELL} is not publishable (the MIN_CELL publication floor); "
              "read counts first")

        print("\nCONTRASTS on the SAME draws (a delta whose interval covers zero is not a finding):")
        for a, b in CONTRASTS:
            d = val.loc[f"{a}-{b}:xg"]
            verdict = ("no interval" if pd.isna(d["lo"]) or pd.isna(d["hi"])
                       else "covers zero" if d["lo"] <= 0 <= d["hi"] else "excludes zero")
            print(f"  {a} - {b}:  {d['point']:+.4f} xG [{d['lo']:+.4f}, {d['hi']:+.4f}]  {verdict}")

        # Written on every run and not only under --build, because the value block is recomputed
        # from the cache too and the table it prices is the artefact's, not this invocation's.
        p = Path(out)
        states = p.with_name(f"{p.stem}_states{p.suffix or '.parquet'}")
        value_record(val, vc).to_parquet(states)
        print(f"\nsaved {states}  [gitignored]")

    seen = tab[tab["n_seen"].notna()]
    if len(seen):
        print("\nlocal situation at the drop (where a drop 360 exists):")
        print(f"  first-to-drop is attacker: {100 * seen['first_to_drop_att'].mean(skipna=True):.1f}%")
        print(f"  mean local control (att share): {seen['local_control'].mean():.3f}")
        print(f"  mean bodies within 5yd: att {seen['n_att_near5'].mean():.2f} / def {seen['n_def_near5'].mean():.2f}")

    # Written on every run for the reason the states record is: it reads the table, cached or not.
    rad = radii_record(tab)
    print("\nwhat the chosen situation constants cut on this table:")
    for _, r in rad.iterrows():
        print(f"  {r['block']:6s} {r['key']:18s} {r['value']:9.4f}  (n={int(r['n'])})")
    p = Path(out)
    radii = p.with_name(f"{p.stem}_radii{p.suffix or '.parquet'}")
    rad.to_parquet(radii, index=False)
    print(f"saved {radii}  [gitignored]")

    print(f"\nNOTE: the four states are known on {int(tab['four_state'].notna().sum())} of "
          f"{n} cleared corners; the rest lack a delivery 360 or a next touch inside the")
    print("episode, and every rate above is over the KNOWN subset. This grid is descriptive:")
    print("staking is a choice a team makes, not a treatment assigned at random.")


if __name__ == "__main__":
    main()
