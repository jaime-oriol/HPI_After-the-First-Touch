"""tracking_clock — the event clock and the Broadcast X&Y game clock, aligned per match and period.

The two feeds keep separate clocks. Joined as if they agreed, an event lands on a tracking frame about a second late in
every one of the thirty tracked matches, so an analysis that reads the tracking at an event's instant first moves the
instant onto the tracking clock. The offset is measured, never assumed: for one
anchor, the shift that minimises the median distance between each event's location, rotated into tracking metres with
its team's direction, and the tracked ball where the feed detects it. A pass leaves the ball at the passer's feet before
its event and a receipt after its own, so the two anchors lean opposite ways, and the offset is their midpoint.

The file is streamed once, line by line, rather than loaded through `loader.load_tracking`: the clock needs only the ball
and the home keeper, and an analysis only the frames around its own instants, while a full load costs the 1.8 GB the
loader documents.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

from s01_phases import _ts_to_s
from s04_race import anchor_in_tracking

SHIFTS = np.round(np.arange(-3.0, 3.0001, 0.04), 2)      # seconds, tracking minus event: the search grid
ANCHORS = {"pass": "Pass", "receipt": "Ball Receipt*"}
_CLOCK = re.compile(r'"period":(\d+),"periodElapsedTime":([-0-9.eE]+)')


def stream(path: Path, home_gk: str, spans: dict) -> tuple[dict, dict, dict]:
    """One pass over a tracking file: the ball per period as an array (t, x, y, detected), the home keeper's x values per
    period, and every player row (is_home, jersey, detected, x, y) of the frames inside each span, keyed as `spans` is.

    `spans` maps a key to (period, start, end) on the tracking clock. A frame's clock is read off the head of its line,
    so only the frames the clock, the keeper or a span needs are parsed whole."""
    ball, keeper, frames = {}, {}, {}
    with open(path) as f:
        f.readline()                                           # the match's metadata, not a frame
        for line in f:
            m = _CLOCK.search(line, 0, 160)
            if not m:
                continue
            p, t = int(m.group(1)), float(m.group(2))
            fr = json.loads(line)
            b = next((b for b in fr.get("balls", []) if b), None)
            if b:
                ball.setdefault(p, []).append((t, b["x"], b["y"], b.get("visibility") == "VISIBLE"))
            keeper.setdefault(p, []).extend(pl["x"] for pl in fr.get("homePlayers", [])
                                            if pl and pl.get("jerseyNum") == home_gk)
            for key, (sp, a, z) in spans.items():
                if sp == p and a <= t <= z:
                    frames.setdefault(key, []).append((t, fr["frameNum"], [
                        (side == "homePlayers", str(pl["jerseyNum"]), pl.get("visibility") == "VISIBLE",
                         pl["x"], pl["y"]) for side in ("homePlayers", "awayPlayers") for pl in fr.get(side, []) if pl]))
    return {p: np.array(v, dtype=float).T for p, v in ball.items()}, keeper, frames


def home_attacks_right(keeper: dict) -> dict:
    """{period: whether the home side attacks towards +x}, from where its keeper stands (the loader's rule)."""
    return {p: float(np.mean(xs)) < 0 for p, xs in keeper.items() if xs}


def to_tracking(location: list, team_is_home: bool, home_right: bool) -> np.ndarray:
    """An event location, logged in its own team's attacking frame, in tracking metres."""
    return anchor_in_tracking(location, 1.0 if team_is_home == home_right else -1.0)


def offset_curve(events: pd.DataFrame, kind: str, ball: np.ndarray, home_right: bool, home_id: int) -> np.ndarray:
    """Median distance in metres between the located events of `kind` and the detected ball, at every shift of SHIFTS.
    `events` are one period's; `ball` is that period's (t, x, y, detected)."""
    ev = events[events["type.name"].eq(kind) & events["location"].map(lambda v: isinstance(v, list) and len(v) == 2)]
    if ev.empty:
        return np.full(len(SHIFTS), np.nan)
    te = ev["timestamp"].map(_ts_to_s).to_numpy(float)
    loc = np.array([to_tracking(v, int(t) == home_id, home_right)
                    for v, t in zip(ev["location"], ev["team.id"], strict=True)])
    t, bx, by, seen = ball
    out = []
    for d in SHIFTS:
        i = np.clip(np.searchsorted(t, te + d), 1, len(t) - 1)
        ok = (seen[i] > 0.5) & (te + d >= t[0]) & (te + d <= t[-1])
        out.append(np.median(np.hypot(bx[i] - loc[:, 0], by[i] - loc[:, 1])[ok]) if ok.any() else np.nan)
    return np.array(out)


def clock_offsets(events: pd.DataFrame, ball: dict, home_right: dict, home_id: int) -> dict:
    """{period: {"pass": d, "receipt": d, "mid": d}}, tracking time = event time + d, for every period that has both a
    ball and a keeper to orient it."""
    out = {}
    for p in sorted(set(ball) & set(home_right)):
        sub = events[events["period"] == p]
        d = {a: float(SHIFTS[int(np.nanargmin(offset_curve(sub, k, ball[p], home_right[p], home_id)))])
             for a, k in ANCHORS.items()}
        out[p] = {**d, "mid": (d["pass"] + d["receipt"]) / 2}
    return out
