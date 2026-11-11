"""compare_tables — are the tables this run built the ones the plan's experiments were read on?

The recipe does not retrain the network of the GPU experiment: its stage loads the sixty tasks fitted
on the reference tables (data/derived/prereg_2026-10-01/gpu.md), each keyed by a digest of the
frames, the design and the target it read. Every verdict the plan registered on its tables, that one
included, comes back from the recipe only if the tables do, so this stage compares them before any
model runs, column by column and row by row, and stops on any difference.

The reference is data/derived/auditoria_2026-10-12/tables: the plan's tables as s01 builds them with
its two attribution rules, a shot owned by the last set piece before it and an own goal placed with
the action that produced it (prereg_2026-10-01/deviation_2026-10-12.md). The tables as first
registered stay in data/derived/auditoria_2026-09-30/tables.

    tables   the six `build_table` writes: both populations, alone and together, and their shots
    grains   the delivery grain and the match results, whose one table holds both populations while
             the plan keeps one per population, compared population by population

With --limit N it reads the tables a smoke run built over the first N matches and compares them with
the reference's rows of those matches, which are the same rows in the same order.

THE SHORT CORNERS ARE LEFT OUT OF THE ROW COMPARISON. A short or ground corner is delivered by its first
cross and not by its own pass (s01_phases), while the reference holds it at its own kick, so its row, its
shots and its label differ from the reference's by construction. Every other row is held to the reference:
the rows that are not a short corner (every long corner and every flank free kick) in the corner tables,
the shots of the deliveries that are not a short corner's, and the same rows of each population's grain.
The corners taken in each match are checked on top, since the grain keeps them whole by holding a row for
a short corner no cross followed.

A corner table carries every partition's counts (src/zones.py), so a column the reference lacks is not a
difference when it is a count of a partition or the partition the table was built under. The jobs, the
commit index and the defenders in the frontal follow the frontal of the partition, which G2 widens, so a
table built under G2 differs from the reference there by construction and those columns are held instead to
what the partition defines: on every delivery, every body once per partition on each side, the jobs adding
up to the attackers, staking the drop being the switch's frontal and its defenders `n_def_frontal`.

    python pipeline/compare_tables.py                     # the six tables
    python pipeline/compare_tables.py --grains            # the delivery grain and the matches
    python pipeline/compare_tables.py --names situation.parquet race_corners.parquet
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO / "src"))
sys.path.insert(0, str(_REPO / "pipeline"))
_DER = _REPO / "data" / "derived"
_PLAN_TABLES = _DER / "auditoria_2026-10-12" / "tables"

from constants import ZONE_PARTITION                 # noqa: E402
from eda_build import COMBINED_KINDS                  # noqa: E402
from zones import FRONTAL, SAME_FRONTAL, ZONES, table_columns  # noqa: E402

TABLES: tuple[str, ...] = ("corners.parquet", "shots.parquet", "corners_free_kick_flank.parquet",
                           "shots_free_kick_flank.parquet", "corners_corner_free_kick_flank.parquet",
                           "shots_corner_free_kick_flank.parquet")
# The columns that follow the switch's frontal, and the ones a corner table may carry beyond the reference.
PARTITION_DEFINED = ("n_attack_cross", "n_stake_drop", "n_hold_back", "commit_index", "n_def_frontal")
PARTITION_COLUMNS = frozenset(table_columns()) | {"zone_partition"}
# What the delivery grain carries beyond the reference's: which event is the delivery, and its corner kick.
GRAIN_COLUMNS = ("anchor", "kick_id", "kick_to_delivery_s")


def untouched(table: pd.DataFrame) -> pd.DataFrame:
    """The rows a short corner's delivery does not touch: all but the short or ground corners."""
    if "short_corner" not in table.columns:
        return table
    return table[~table["short_corner"].astype(bool)].reset_index(drop=True)


def short_ids(grain: pd.DataFrame) -> set:
    """The ids of the short or ground corners in a delivery grain."""
    if "short_corner" not in grain.columns:
        return set()
    return set(grain.loc[grain["short_corner"].astype(bool), "set_piece_id"])


def differences(got: pd.DataFrame, want: pd.DataFrame, strict_dtype: bool = True) -> list[str]:
    """Every way `got` differs from `want`, empty when they are the same table: the columns and
    their order, the row count, and per column its dtype and how many rows disagree, a missing
    value agreeing only with a missing value.

    Without `strict_dtype` two numeric columns are read by value: a column that is float on the
    grant because some rows carry NaN comes out int on a subset without them. A column missing on
    every row of both is the same whatever its dtype, under either: a table that never sets it
    reads it as object where nothing shares it, and as float where it shares a table with rows
    that carry it, as one population's slice of the joint grain does."""
    if list(got.columns) != list(want.columns):
        only, missing = sorted(set(got) - set(want)), sorted(set(want) - set(got))
        return [f"columns: {only} only here, {missing} only in the reference" if only or missing
                else "columns: the same ones in another order"]
    if len(got) != len(want):
        return [f"rows: {len(got)} against {len(want)}"]
    out = []
    got, want = got.reset_index(drop=True), want.reset_index(drop=True)
    for c in got.columns:
        a, b = got[c], want[c]
        if a.isna().all() and b.isna().all():
            continue
        numeric = pd.api.types.is_numeric_dtype(a) and pd.api.types.is_numeric_dtype(b)
        if a.dtype != b.dtype and (strict_dtype or not numeric):
            out.append(f"{c}: dtype {a.dtype} against {b.dtype}")
            continue
        if a.dtype != b.dtype:
            a, b = a.astype(float), b.astype(float)
        if not a.equals(b):
            off = ~((a.astype(str) == b.astype(str)) | (a.isna() & b.isna()))
            out.append(f"{c}: {int(off.sum())} of {len(a)} rows differ")
    return out


def partition_checks(t: pd.DataFrame) -> list[str]:
    """What a corner table built under the switch owes on every delivery, whatever the reference holds:
    each body once per partition on each side, the jobs adding up to the attackers, staking the drop
    being the switch's frontal, its defenders `n_def_frontal`, and the switch named as the partition."""
    p, edge = ZONE_PARTITION, FRONTAL[ZONE_PARTITION]
    checks = {f"{q}'s {side} counts add up to {n}": t[[f"{q}_{side}_{z}" for z in ZONES[q]]].sum(axis=1) == t[n]
              for q in ZONES for side, n in (("att", "n_att"), ("def", "n_def"))}
    checks |= {"the jobs add up to n_att": t[["n_attack_cross", "n_stake_drop", "n_hold_back"]].sum(axis=1)
               == t["n_att"],
               f"n_stake_drop is {p}_att_{edge}": t["n_stake_drop"] == t[f"{p}_att_{edge}"],
               f"n_def_frontal is {p}_def_{edge}": t["n_def_frontal"] == t[f"{p}_def_{edge}"],
               f"zone_partition is {p}": t["zone_partition"] == p}
    return [f"{what}: {int((~ok).sum())} of {len(t)} rows fail" for what, ok in checks.items() if not ok.all()]


def against_reference(got: pd.DataFrame, want: pd.DataFrame, strict_dtype: bool) -> list[str]:
    """A corner table against the reference, built under C: the reference's columns as `differences`
    reads them, less those the switch's frontal defines when it is not C's, and only partition columns
    besides them. The rows compared are the ones a short corner's delivery does not touch; what the
    partition defines is owed by every row of this table."""
    extra = [c for c in got.columns if c not in want.columns]
    out = [f"columns: {sorted(set(extra) - PARTITION_COLUMNS)} only here and no partition's"
           ] if set(extra) - PARTITION_COLUMNS else []
    own = [] if "C" in SAME_FRONTAL[ZONE_PARTITION] else list(PARTITION_DEFINED)
    out += differences(untouched(got).drop(columns=extra + own), untouched(want).drop(columns=own),
                       strict_dtype)
    return out + partition_checks(got)


def pairs(names: list[str], grains: bool, limit: int | None, der: Path,
          ref: Path) -> list[tuple[str, pd.DataFrame, pd.DataFrame]]:
    """(label, this run's table, the reference's) for every comparison asked for, the reference
    cut to the matches a --limit table holds."""
    tail = f"_limit{limit}" if limit else ""

    def mine(name: str) -> pd.DataFrame:
        p = der / name.replace(".parquet", f"{tail}.parquet")
        if not p.exists():
            raise SystemExit(f"STOP: {p.name} absent; the stage that writes it has not run")
        return pd.read_parquet(p)

    def cut(want: pd.DataFrame, got: pd.DataFrame) -> pd.DataFrame:
        return want[want["match_id"].isin(set(got["match_id"]))] if limit else want

    def grain_ids() -> tuple[set, set]:
        """The short corners' ids in this run's grain and in the reference's, which decide what each
        shot table holds out of the comparison; none when either grain is absent."""
        own = der / f"eda_deliveries{tail}.parquet"
        theirs = [ref / f"eda_deliveries_{k}.parquet" for k in COMBINED_KINDS]
        if not own.exists() or not all(p.exists() for p in theirs):
            return set(), set()
        return short_ids(pd.read_parquet(own)), set().union(*(short_ids(pd.read_parquet(p)) for p in theirs))

    def per_match(grain: pd.DataFrame) -> pd.DataFrame:
        return grain.groupby("match_id").size().rename("corners").reset_index()

    out, ids = [], None
    for name in names:
        got = mine(name)
        want = cut(pd.read_parquet(ref / name), got)
        if name.startswith("shots") and "set_piece_id" in got.columns:
            ids = ids or grain_ids()
            got = got[~got["set_piece_id"].isin(ids[0])].reset_index(drop=True)
            want = want[~want["set_piece_id"].isin(ids[1])].reset_index(drop=True)
        out.append((name, got, want))
    if grains:
        deliv, matches = mine("eda_deliveries.parquet"), mine("eda_matches.parquet")
        for kind in COMBINED_KINDS:
            got = deliv[deliv["set_piece_type"] == kind]
            want = cut(pd.read_parquet(ref / f"eda_deliveries_{kind}.parquet"), got)
            out.append((f"eda_deliveries.parquet [{kind}]",
                        untouched(got).drop(columns=list(GRAIN_COLUMNS), errors="ignore"), untouched(want)))
            out.append((f"eda_deliveries.parquet [{kind}] corners per match", per_match(got), per_match(want)))
        want = pd.read_parquet(ref / f"eda_matches_{COMBINED_KINDS[0]}.parquet")
        out.append(("eda_matches.parquet", matches, cut(want, matches)))
    return out


def main() -> None:
    """Compare, print one line per table, and exit non-zero on any difference."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--names", type=str, nargs="*", default=None,
                    help="tables to compare by name (default: the six build_table writes)")
    ap.add_argument("--grains", action="store_true",
                    help="the delivery grain and the match results instead, population by population")
    ap.add_argument("--reference", type=str, default=str(_PLAN_TABLES))
    ap.add_argument("--limit", type=int, default=None,
                    help="compare the first-N-match tables a smoke run built")
    args = ap.parse_args()
    names = args.names if args.names is not None else ([] if args.grains else list(TABLES))
    found = []
    for label, got, want in pairs(names, args.grains, args.limit, _DER, Path(args.reference)):
        compare = against_reference if label.startswith("corners") else differences
        diff = compare(got, want, strict_dtype=args.limit is None)
        print(f"  {label:48s} {len(got):6d} rows  {'SAME' if not diff else 'DIFFERS'}", flush=True)
        found += [f"{label}: {d}" for d in diff]
    if found:
        raise SystemExit("STOP: the tables differ from the reference, so no verdict of the plan can be "
                         "reproduced on them:\n  " + "\n  ".join(found))
    print(f"every table matches {args.reference} outside the short corners")


if __name__ == "__main__":
    main()
