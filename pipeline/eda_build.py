"""eda_build — persist the delivery grain and the match results the EDA runs on.

build_table.py computes the S01 label grain over every delivery and then throws it
away, caching only the 360-carrying subset. The team-level EDA needs the full grain:
a team's corner count is a denominator, and denominators taken over only the
deliveries that happen to carry a freeze-frame are not corner counts.

Match results come from the running WinningTeam state, read at the last event of the
match. Counting shot outcomes would miss own goals: the export files those as an
`Own Goal For` / `Own Goal Against` pair and never as a Shot, so a predicate written
against Shot scores none of them (`definitions.md` §4).

    python pipeline/eda_build.py --limit 20 --jobs 8     # smoke
    python pipeline/eda_build.py --jobs 20               # the grant
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Any

import pandas as pd
from joblib import Parallel, delayed

_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO / "src"))
sys.path.insert(0, str(_REPO / "pipeline"))
_DER = _REPO / "data" / "derived"

import loader_sb as L                                   # noqa: E402
from build_table import kind_tag                        # noqa: E402
from s01_phases import KINDS, SetPieceGrammar           # noqa: E402
from s03_value import _league_index                     # noqa: E402

# The paper's two populations: corners and the flank free kicks from x >= 85.
COMBINED_KINDS = ("corner", "free_kick_flank")


def kind_inputs(kind: str) -> tuple[Path, Path]:
    """(delivery grain, shot grain) the descriptive layer reads for one population.

    corner and free_kick_flank share the two-population tables this driver and build_table
    publish, and consumers filter on `set_piece_type`. Any other kind has no tables and raises."""
    if kind not in COMBINED_KINDS:
        raise ValueError(f"no tables for kind {kind!r}; valid: {COMBINED_KINDS}")
    return _DER / "eda_deliveries.parquet", _DER / f"shots{kind_tag(COMBINED_KINDS)}.parquet"


def _first(s: pd.Series) -> Any:
    v = s.dropna()
    return v.iloc[0] if len(v) else None


def match_rows(match_id: int, league: str, kinds: tuple) -> tuple:
    """(delivery-grain label rows, one match-result row) off a single parse."""
    ev = L.load_events(match_id)
    g = SetPieceGrammar(ev, kinds=kinds)
    labels = g.label_all()
    tag = {"match_id": match_id, "league": league}
    rows = [{**r, **tag} for r in labels.to_dict("records")] if len(labels) else []

    home, away = _first(ev["home_team.home_team_name"]), _first(ev["away_team.away_team_name"])
    winner = _first(ev["WinningTeam"].iloc[::-1])
    result = {**tag, "home_team": home, "away_team": away, "winner": winner,
              "n_events": len(ev)}
    return rows, result


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--source", type=str, default=None)
    ap.add_argument("--jobs", type=int, default=1)
    ap.add_argument("--kinds", type=str, default=",".join(COMBINED_KINDS))
    ap.add_argument("--out-dir", type=str, default=None,
                    help="where to write (default: data/derived); the smoke tests redirect it")
    args = ap.parse_args()

    mids = L.list_matches(args.source)["match_id"].tolist()
    if args.limit:
        mids = mids[:args.limit]
    kinds = tuple(k.strip() for k in args.kinds.split(","))
    assert set(kinds) <= set(KINDS), f"unknown kind in {kinds}; valid: {KINDS}"
    leagues = _league_index()
    pairs = [(int(m), leagues.get(int(m))) for m in mids]

    t = time.time()
    if args.jobs == 1:
        chunks = [match_rows(m, lg, kinds) for m, lg in pairs]
    else:
        chunks = Parallel(n_jobs=args.jobs, verbose=5)(
            delayed(match_rows)(m, lg, kinds) for m, lg in pairs)

    deliveries = pd.DataFrame([r for rows, _ in chunks for r in rows])
    matches = pd.DataFrame([res for _, res in chunks])
    print(f"\n{len(deliveries)} deliveries / {len(matches)} matches in {time.time() - t:.1f}s "
          f"({args.jobs} worker(s))")

    assert matches["home_team"].notna().all(), "match with no home team"
    assert matches["winner"].notna().all(), "match with no WinningTeam state"
    print("  kinds:", deliveries["set_piece_type"].value_counts().to_dict())
    print("  per match:", round(len(deliveries) / len(matches), 2))

    # Both narrowing flags tag the name: the descriptive drivers read the grant's tables by name
    # through kind_inputs and their output reaches measured.md, so a subset must never land there.
    tag = (kind_tag(kinds, COMBINED_KINDS) + (f"_{args.source.lower()}" if args.source else "")
           + (f"_limit{args.limit}" if args.limit else ""))
    out = Path(args.out_dir) if args.out_dir else _DER
    out.mkdir(parents=True, exist_ok=True)
    deliveries.to_parquet(out / f"eda_deliveries{tag}.parquet")
    matches.to_parquet(out / f"eda_matches{tag}.parquet")
    print(f"saved {out}/eda_deliveries{tag}.parquet and eda_matches{tag}.parquet  [gitignored]")


if __name__ == "__main__":
    main()
