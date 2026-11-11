# Data schema — Hudl HPI 2026

Reference for the datasets the competition grants, described from the files on disk.
Source: <https://www.hudl.com/blog/hpi-2026-research-competition>.

## Track 1 — Broadcast X&Y Tracking + Hudl Statsbomb events

A 30-match sample of Hudl's new **Broadcast X&Y tracking** from the **2025/26
Italian Serie A**, with the matching **Hudl Statsbomb event** data for the same
matches so both can be combined.

- **Broadcast X&Y tracking**: 2D player (and ball) positions derived from the
  broadcast feed, not stadium optical tracking. Expect partial detection
  (players off-camera) and the usual broadcast caveats. Normalise the coordinate
  frame before any geometry. The vendor's own `visibility` flag travels with the
  row rather than filtering it: on the ball it marks an interpolated position,
  and dropping those rows would empty the race stage instead of cleaning it.
- **Events**: the standard Hudl Statsbomb event stream, synchronised to the same
  30 matches, so an event can be anchored to the tracking frame around it.

## Track 2 — Hudl Statsbomb Event & 360 data

Five full **competition-seasons** of event + **360** data: the Big-5 leagues,
2025/26 (Bundesliga, La Liga, Ligue 1, Premier League, Serie A). Larger sample,
no continuous tracking.

### Event data (grain: one row per on-ball event)

Key fields (Statsbomb taxonomy): `match_id`, `id` (event uuid), `period`,
`timestamp`, `minute`, `second`, `type.name` (Pass, Shot, Carry, ...),
`play_pattern.name` (From Corner / From Free Kick / From Throw In / From Kick
Off / Regular Play...), `team.name`, `player.name`, `position.name`,
`location [x, y]`, plus type-specific blocks (`shot.statsbomb_xg`, `pass.*`,
...). Pitch is 120x80 yards, origin top-left. Fields are already flat, dotted
keys as shown, not nested objects.

### 360 data (grain: one freeze-frame per event)

For each event, the on-pitch positions of visible teammates and opponents at the
moment of the event (`freeze_frame`: list of `{location [x, y], teammate,
actor, keeper}`) plus the `visible_area` polygon. This is the key asset for
set-pieces: it gives the **defensive and attacking shape at the moment of the
delivery** (corner / free kick) without continuous tracking.

## Why this matters for the set-pieces theme

- **360 freeze-frames** → marking scheme (zonal vs man), blockers, near/far-post
  load, attacker/defender positions at delivery.
- **Events** → routine type, delivery (inswing/outswing), first contact, outcome
  and `statsbomb_xg` of the resulting shot.
- **Broadcast X&Y** (Serie A 30) → the continuous run-up movement that the
  freeze-frame alone can't show, on a smaller sample.

## Layout of the granted export, on disk

Both tracks live under the gitignored `data/`, split to one file per match:

```
data/
├── serie_a_sample/                     # Track 1: 30-match Broadcast X&Y + matching events (loader.py + loader_sb.py)
│   ├── events_360/<match_id>_<home>_vs_<away>_<date>.json   # events + 360, already combined
│   └── tracking/<gameRefId>_<home>_vs_<away>_<date>.jsonl   # 1st line = match metadata, then 1 line/frame
└── big5_2025_26/<league>/events/<match_id>.json   # Track 2: 5 full competition-seasons (loader_sb.py)
    # league in {bundesliga, la_liga, ligue_1, premier_league, serie_a}
```

See [`../src/loader_sb.py`](../src/loader_sb.py) (events + 360, both tracks;
`list_matches()` defaults to all five Big-5 leagues, pass `SERIE_A_SAMPLE` or a
league name to narrow it) and [`../src/loader.py`](../src/loader.py) (Track 1
tracking only).

**Confirmed schema notes** (from the real files, not the public docs):
- Events arrive **already flattened** (dotted keys like `type.name`, `pass.type.name`,
  so no `json_normalize` needed) with the 360 `freeze_frame`/`visible_area` embedded
  per event, not a separate endpoint. Absent-360 rows carry StatsBomb's `{}`
  sentinel, not `null`/`[]`. Filter on type, not truthiness.
- **The parquet cache normalises that sentinel and restores the type on read.** Arrow
  refuses a column mixing dict and list (`ArrowInvalid: cannot mix list and non-list,
  non-null values` on `related_events`), so the cache writes `{}`/`[]` as null across
  the 14 container columns. Verified lossless: across 20 matches spanning all five
  leagues there is not one non-empty dict. On re-read pyarrow hands back
  `numpy.ndarray` where the JSON had `list`, and every container filter in this repo
  tests `isinstance(v, list)`, which is False for an ndarray: measured, 3095
  freeze-frames became **0 with no exception raised**. `loader_sb` restores the type
  and asserts it at the boundary with `type(v) is list`.
- **A shot carries a SECOND freeze-frame that is not the 360 one and does not share its
  keys.** `shot.freeze_frame` holds `{location, teammate, player.id, player.name,
  position.id, position.name}` and has no `actor` and no `keeper` flag, so the keeper is
  identified by `position.name` there and by the `keeper` flag in the 360 frame. One
  detection rule cannot serve both. Its composition also changed in January 2026: the
  shooter is absent from his own frame before it and present after, so the same field
  means a different population on either side of the break.
- Match metadata (`home_team.home_team_name`, `match_date`, ...) is duplicated on
  **every** event row, not just the first. Cheap to read from any single row.
- Serie A is nested: the 30 Track-1 sample matches are a strict subset of the
  380-match Track-2 `big5_2025_26/serie_a/` season (confirmed by id, not assumed).
- Track 1's two feeds use **different id spaces** for the same match (StatsBomb
  `match_id` in `events_360/`, Hudl's own `gameRefId` in `tracking/`'s filenames)
  and **different team-name conventions** for the same club ("Roma"/"Inter"/"Milan"
  in tracking vs "AS Roma"/"Inter Milan"/"AC Milan" in events). `loader.py` pairs
  them by (home, away, date), not by id.
- Tracking gives a jersey number and a home/away flag per player, not a team id or
  player id. `loader.py` resolves both against the paired events file's
  `home_team.home_team_id`/`away_team.away_team_id` and Starting XI `tactics.lineup`
  (also the source for `is_goalkeeper`; `direction_of_play` is read per period from
  the home KEEPER's mean x, since he plays beside the goal he defends). The kickoff
  shape cannot answer it on this feed: the broadcast clock starts before kick-off, so
  a period's first frames hold no detected player at all, and that rule comes out
  wrong on many (match, period) slots (`tests/test_loader.py` counts them).
- Tracking carries **two per-frame clocks that are not interchangeable**:
  `period_elapsed_s` (the vendor's game clock, the closer of the two to the
  StatsBomb event timestamps and still behind them) and
  `timeelapsed_from_period_start_s` (the broadcast video clock, offset from
  kickoff by a per-broadcast margin). An event-to-frame join uses the game clock
  moved by the offset [`../src/tracking_clock.py`](../src/tracking_clock.py)
  measures per match and half; see `load_tracking` in
  [`../src/loader.py`](../src/loader.py).
- The Big-5 leagues arrived as 5 merged multi-match files per league (Google
  Drive's size-based split, matches interleaved across parts). Split to one file
  per `match_id` before use; nothing in the pipeline reads the merged form.
