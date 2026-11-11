# refdata — small versioned reference tables

External reference data the export does not carry. It lives in the repository rather than under the
gitignored `data/`, because `pipeline/eda_standings.py` refuses to run without it and a fresh clone
has to be able to reproduce the pipeline end to end.

## `standings_2025_26.csv`

The final league tables of the five granted competition-seasons: `league`, `pos`, `team_ref`, `pts`.
96 rows, one per club, positions contiguous within each league.

**Source**: the English Wikipedia season articles for 2025-26 (Premier League, La Liga, Serie A,
Bundesliga, Ligue 1), read 2026-09-10.

**Why it is not derived from the events.** The points are, and the driver derives them. The
finishing POSITION is not, because ordering clubs level on points needs five different tie-break
rules, head-to-head in Serie A and La Liga and goal difference in the other three. A table derived
from the same events it exists to validate would also check nothing.

**Why it can be trusted, and it is not the source that says so.** `eda_standings.py` derives every
club's points from the event stream and matches the two tables by name similarity, then asserts the
pairing by points. On this file the assertion passes for 96 of 96 clubs with no duplicate
assignment, and the only two clubs whose points disagree are Toulouse and Nantes, each by one. That
is a coverage fact rather than an error. The grant ships 305 of the 306 Ligue 1 fixtures and the one
it never shipped is Nantes-Toulouse; both clubs sit a point short with 33 games played, which puts
that fixture at a draw.

Regenerate the derived table with:

```bash
python pipeline/eda_standings.py
```

## `zones_s.json`

S, the partition of the box drawn from where the attackers holding a job stand at the kick: a
Gaussian mixture over the long corners of M1's population, positions only and no outcome. The file
holds its size, the components' names, weights, means and covariances, and a `provenance` block that
names the driver, the sheet, the tables and a digest of the points the mixture was fitted on.

**Why it lives here.** Every stage that labels a body under S reads these parameters, and a fresh
clone has to label S without refitting it: a fit on other tables would draw another partition under
the same name.

**How it was drawn** is pre-registered in `data/derived/prereg_2026-10-07/zones_s.md`, before any
score of S was read. The file is frozen: `pipeline/fit_zones_s.py` refuses to write over it unless
it is passed `--overwrite`, and the fit is deterministic, so a rerun on the same tables gives the
same parameters.
