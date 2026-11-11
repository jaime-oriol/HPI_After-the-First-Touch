"""eda_standings — derive the final table from the events and reconcile it with the published one.

The two sides of this are both needed. The published table (refdata/standings_2025_26.csv,
Wikipedia, versioned because the export does not carry it) has the canonical finishing position;
the export has a different set of club names for the same clubs (Athletic Club / Athletic Bilbao,
AS Roma / Roma). Matching those by eye is unverifiable, so the mapping is proposed by name
similarity and then PROVED by points: a pair whose derived points equal its published points is
matched correctly, whatever the strings look like.

Proved means asserted. The reconciliation raises, because a driver that prints a disagreement and
exits 0 leaves a silently wrong mapping in `eda_standings.parquet` and every figure downstream of
it. The one tolerance is fixtures the grant never shipped: a team missing k matches may sit up to
3k points below its published total and no more, and a team missing none has to match exactly.
Ligue 1 arrives one fixture short (refdata/README.md names the fixture Ligue 1 is missing).

Points come from the running WinningTeam state read at the last event of each match, which
picks up own goals that a count of Goal outcomes would miss.

    python pipeline/eda_standings.py
    python pipeline/eda_standings.py --matches data/derived/eda_matches_limit200.parquet \\
        --out data/derived/smoke/eda_standings.parquet     # a smoke's N-match results
"""

from __future__ import annotations

import argparse
import unicodedata
from difflib import get_close_matches
from pathlib import Path

import pandas as pd

_REPO = Path(__file__).resolve().parents[1]
_DER = _REPO / "data" / "derived"
_REF = _REPO / "refdata" / "standings_2025_26.csv"


def _key(s: str) -> str:
    """Accent-free lowercase, minus the corporate noise that differs between feeds."""
    s = unicodedata.normalize("NFKD", str(s)).encode("ascii", "ignore").decode()
    for junk in ("fc ", " fc", "afc ", "ac ", "as ", "sc ", "vfl ", "vfb ", "tsg ", "fsv ",
                 "ogc ", "1. ", "ud ", "deportivo ", "olympique de ", "stade ", " calcio 1913",
                 " cf", " hove albion", "hellas "):
        s = s.lower().replace(junk, " ")
    return " ".join(s.lower().split())


def derive_table(matches: pd.DataFrame) -> pd.DataFrame:
    """Points, wins, draws and losses per team from the per-match winner."""
    rows = []
    for m in matches.itertuples():
        drew = m.winner == "Drawing"
        for team, opp in ((m.home_team, m.away_team), (m.away_team, m.home_team)):
            pts = 1 if drew else (3 if m.winner == team else 0)
            rows.append({"league": m.league, "team": team, "pts": pts,
                         "w": int(not drew and m.winner == team),
                         "d": int(drew), "l": int(not drew and m.winner == opp)})
    t = pd.DataFrame(rows).groupby(["league", "team"], as_index=False).agg(
        played=("pts", "size"), pts=("pts", "sum"), w=("w", "sum"), d=("d", "sum"), l=("l", "sum"))
    return t.sort_values(["league", "pts"], ascending=[True, False])


def reconcile(tab: pd.DataFrame) -> pd.DataFrame:
    """Rows whose points gap no missing fixture can explain. Empty means the mapping is proved.

    A full round-robin is 2*(n-1) games per team, so a team short of that is short of fixtures the
    grant never shipped, and each is worth at most 3 points. Deriving MORE points than published is
    never explainable and is always a mismatch."""
    tab = tab.copy()
    played_full = tab.groupby("league")["team"].transform("size").sub(1).mul(2)
    tab["missing"] = played_full - tab["played"]
    tab["gap"] = tab["pts_ref"] - tab["pts_derived"]
    return tab[(tab["gap"] < 0) | (tab["gap"] > 3 * tab["missing"])]


def main() -> None:
    """Derive the table, map each club to its published row, and assert the mapping.

    Three assertions in a row, and the order matters: every club must match, no two clubs
    may claim the same published row, and no points gap may survive that a missing fixture
    cannot explain. Each is a different way for the mapping to be wrong, and the second is
    the one a name-similarity match fails silently."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=str, default=None,
                    help="output parquet (default: data/derived/eda_standings.parquet)")
    ap.add_argument("--matches", type=str, default=None,
                    help="the match results (default: data/derived/eda_matches.parquet)")
    args = ap.parse_args()

    grant = _DER / "eda_matches.parquet"
    src = Path(args.matches) if args.matches else grant
    if args.out is None and src.resolve() != grant.resolve():
        raise SystemExit(f"STOP: {src.name} is not the grant's results, so it may not write the "
                         f"published eda_standings.parquet; pass --out")
    matches = pd.read_parquet(src)
    if not _REF.exists():
        raise FileNotFoundError(
            f"{_REF} is missing. It is versioned in the repo (see refdata/README.md); "
            "restore it from git rather than re-collecting it by hand.")
    ref = pd.read_csv(_REF)
    derived = derive_table(matches)

    print(f"derived {len(derived)} teams over {len(matches)} matches")
    bad = derived[derived["played"] != derived[["w", "d", "l"]].sum(axis=1)]
    assert bad.empty, f"W+D+L != played for {len(bad)} teams"

    out, unmatched = [], []
    for lg, g in derived.groupby("league"):
        r = ref[ref["league"] == lg]
        cand = {_key(t): t for t in r["team_ref"]}
        for row in g.itertuples():
            # A loose cutoff is safe here and a tight one is not: a wrong pairing dies on the
            # points assertion below, while a missed pairing raises as unmatched with no hint.
            hit = get_close_matches(_key(row.team), list(cand), n=1, cutoff=0.55)
            if not hit:
                unmatched.append((lg, row.team))
                continue
            ref_team = cand[hit[0]]
            rr = r[r["team_ref"] == ref_team].iloc[0]
            out.append({"league": lg, "team": row.team, "team_ref": ref_team,
                        "pos": int(rr["pos"]), "pts_ref": int(rr["pts"]),
                        "pts_derived": int(row.pts), "played": int(row.played)})

    tab = pd.DataFrame(out)
    dup = tab[tab.duplicated(["league", "team_ref"], keep=False)]
    mismatch = tab[tab["pts_ref"] != tab["pts_derived"]]

    print(f"matched {len(tab)}/{len(derived)}; unmatched {unmatched}")
    print(f"duplicate ref assignments: {len(dup)}")
    if len(dup):
        print(dup.sort_values(["league", "team_ref"]).to_string(index=False))
    assert not unmatched, f"unmatched clubs: {unmatched}"
    assert dup.empty, f"{len(dup)} clubs matched to the same published row"
    unexplained = reconcile(tab)
    assert unexplained.empty, (
        "points disagreement no missing fixture explains:\n"
        + unexplained.sort_values(["league", "pos"]).to_string(index=False))

    print(f"points disagreement: {len(mismatch)}/{len(tab)}, all explained by missing fixtures")
    if len(mismatch):
        print(mismatch.sort_values(["league", "pos"]).to_string(index=False))

    tab = tab.sort_values(["league", "pos"])
    out = Path(args.out) if args.out else _DER / "eda_standings.parquet"
    out.parent.mkdir(parents=True, exist_ok=True)
    tab.to_parquet(out)
    print(f"\nsaved {out}  ({len(tab)} teams)")


if __name__ == "__main__":
    main()
