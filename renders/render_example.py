"""render_example — one real corner, drawn twice: event by event and read at the kick.

The corner is the author's pick, Timber's second goal against Leeds United on 23 August 2025: a direct shot that a
defender blocks, and a goal that comes after the clearance, so one corner shows two segments. It is an example of the
segmentation and of the plan, never a result: one corner proves nothing. The two figures read the event chain (s01) and
the freeze-frame of the kick (s02) of that corner, and every time, name and class printed is read from them; the first
fails if the corner stops being the one its caption describes. Both are drawn with the attack towards the top, the way
a coach looks at the box from the halfway line, and with the kick from the left flag, as the model reads every delivery.
The first draws the corner as a sequence map in the manner of Opta Analyst, and beside it the same touches as the rows of
the event stream. The map shows the whole corner, from the flag to the far side of the box, and under it a zoom on the
zone where the touches fall: the touches after the kick lie within a few yards of one another, so only the zoom is wide
enough for a node with its number inside to leave its neighbours their own. In the zoom every touch is a numbered node at
its true place, the ball played on (a pass, a shot, a clearance) is a straight line with an arrow at its end and a touch
or a carry is a dashed line; on the map the touches are dots at the same places. The legend of both is laid out as Opta
Analyst lays its own: one row under the pitch, each key an icon over its label. The figures are laid out by hand, so each
one checks, once drawn, that no label covers another or a node it does not belong to and that no two nodes overlap.

    python renders/render_example.py
"""
from __future__ import annotations

import argparse
import itertools
import sys
from collections.abc import Callable
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Arc, Circle, FancyArrowPatch, FancyBboxPatch, Polygon, Rectangle

_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO / "src"))

import loader_sb as L                                                       # noqa: E402
from constants import (BOX_X, BOX_Y, G2_EDGE_CUT, GOAL_POSTS_SB, GOALMOUTH_X, HOLD_X,   # noqa: E402
                       PITCH_X_SB, SIX_YARD_X, SIX_YARD_Y)
from s01_phases import SetPieceGrammar, first_contact, genuine_touch          # noqa: E402
from s02_signature import CommitmentSignature, build_frame                    # noqa: E402
from viz import paper as P                                                  # noqa: E402
from zones import canonical_y, frontal, kicked_from_right                   # noqa: E402

_DER = _REPO / "data" / "derived"
HOME, AWAY, DATE = "Arsenal", "Leeds United", "2025-08-23"
SCORER, NTH_GOAL = "Timber", 2           # the surname, and which of his goals from a corner in that match
# What the figure tells, as (surname, reading) per touch: the corner is the one the caption describes or the figure fails.
CHAIN = (("Rice", "the kick"), ("Timber", "first contact"), ("Timber", "direct shot"), ("Struijk", "clearing contact"),
         ("Struijk", "clearing contact"), ("Saliba", "attacking touch"), ("Calafiori", "attacking touch"),
         ("Calafiori", "after a clearance"), ("Timber", "attacking touch"), ("Timber", "attacking touch"),
         ("Timber", "after a clearance"))
_SEGMENT = {"direct": "direct shot", "flick_on": "flick-on", "prolongation": "prolongation",
            "after_clearance": "after a clearance", "late": "late"}
_POSTS_Y = (GOAL_POSTS_SB[0][1], GOAL_POSTS_SB[1][1])
_GOAL_DEPTH = 2.0      # yards, the net drawn behind the goal line
_LEGEND_H = 0.62       # inches, the row of the legend under the pitch
ORANGE_TINT, SLATE_TINT, BLUE_TINT = "#FFEEE2", "#EAEEF2", "#E3F1FD"
# A legend key: draws itself at (x, y) inches on the legend's axes.
Icon = Callable[[plt.Axes, float, float], object]


# ── the corner ───────────────────────────────────────────────────────────
def pick(der: Path = _DER) -> tuple[int, str]:
    """(match id, delivery id) of the corner the figures follow: the NTH_GOAL-th goal that SCORER put in from a corner
    in that match, in match order, among the corners whose kick carries a freeze-frame."""
    m = L.list_matches(None)
    m = m[(m["home_team"] == HOME) & (m["away_team"] == AWAY) & (m["match_date"].astype(str).str[:10] == DATE)]
    match_id = int(m["match_id"].iloc[0])
    g = SetPieceGrammar(L.load_events(match_id), kinds=("corner",))
    ev = g.ev.set_index("id")
    mine = sorted((int(ev.at[r.shot_id, "period"]), float(ev.at[r.shot_id, "_t"]), r.set_piece_id)
                  for r in g.shots_all().query("is_goal").itertuples() if str(ev.at[r.shot_id, "player.name"]).endswith(SCORER))
    sp_id = mine[NTH_GOAL - 1][2]
    assert sp_id in set(pd.read_parquet(der / "corners.parquet", columns=["set_piece_id"])["set_piece_id"]), \
        "the corner has no freeze-frame at the kick"
    return match_id, sp_id


def _to_attack_frame(team: str, xy: tuple, attacking: str) -> tuple[float, float]:
    """An event's location in the attackers' frame: the provider stores every event in the acting team's own
    attacking frame, so the defence's are turned half a turn."""
    x, y = float(xy[0]), float(xy[1])
    return (x, y) if team == attacking else (PITCH_X_SB - x, 80.0 - y)


def _what(row: pd.Series) -> str:
    """The event in a few words, from its type and its qualifiers."""
    t = row["type.name"]
    if t == "Pass":
        return "Corner kick" if row.get("pass.type.name") == "Corner" else "Pass"
    if t == "Shot":
        kind = "Header" if row.get("shot.body_part.name") == "Head" else ("Volley" if row.get("shot.technique.name") == "Volley" else "Shot")
        out = {"Goal": "goal", "Blocked": "blocked", "Wayward": "off target", "Off T": "off target", "Saved": "saved"}
        return f"{kind}, {out.get(row['shot.outcome.name'], str(row['shot.outcome.name']).lower())}"
    return {"Ball Receipt*": "Receipt", "Ball Recovery": "Recovery"}.get(t, str(t))


def story(match_id: int, sp_id: str) -> dict:
    """The corner as the figure tells it: the kick, then every genuine touch up to the goal, each with its time from the
    kick, its place in the attackers' frame, its event type and the reading the segmentation gives it (the kick, the first
    contact, a clearing contact, an attacking touch, or for a shot its segment). Fails if the chain is not CHAIN."""
    ev = L.load_events(match_id)
    g = SetPieceGrammar(ev, kinds=("corner",))
    delivery = g.deliveries().query("id == @sp_id").iloc[0]
    kick = g.ev[g.ev["id"] == sp_id].iloc[0]
    t0, att = float(kick["_t"]), kick["team.name"]
    post = g._sequence(delivery)
    post = post[post["id"] != sp_id]
    fc = first_contact(post, t0 + g.SECOND_BALL_WINDOW_S)
    assert fc is not None, "the corner is no longer the one the figure describes"
    clears = set(g._clearances(post, g._defending_team(post, att), float(fc["_t"]))["id"])
    seg = {x.shot_id: x.segment for x in g.shots(delivery)}
    goal = post[(post["type.name"] == "Shot") & (post["team.name"] == att) & (post["shot.outcome.name"] == "Goal")].iloc[0]
    upto = post.iloc[:post.index.get_loc(goal.name) + 1]
    touches = upto[genuine_touch(upto)]

    def reading(r: pd.Series) -> str:
        if r["id"] == fc["id"]:
            return "first contact"
        if r["type.name"] == "Shot":
            return _SEGMENT[seg[r["id"]]]
        return ("clearing contact" if r["id"] in clears else "defending touch") if r["team.name"] != att else "attacking touch"

    def event(r: pd.Series, read: str) -> dict:
        return dict(t=float(r["_t"]) - t0, who=r["player.name"].split()[-1], team=r["team.name"], kind=str(r["type.name"]),
                    what=_what(r), read=read, at=_to_attack_frame(r["team.name"], r["location"], att))

    events = [event(kick, "the kick")] + [event(r, reading(r)) for _, r in touches.iterrows()]
    assert tuple((e["who"], e["read"]) for e in events) == CHAIN, "the corner is no longer the one the figure describes"
    events[0]["end"] = _to_attack_frame(att, kick["pass.end_location"], att)
    events[-1].update(xg=float(goal["shot.statsbomb_xg"]), end=_to_attack_frame(att, goal["shot.end_location"][:2], att))
    return {"events": events, "team": att, "clock": f"{int(kick['minute']):02d}:{int(kick['second']):02d}",
            "match": f"{ev['home_team.home_team_name'].iloc[0]} v {ev['away_team.away_team_name'].iloc[0]}"}


def kick_frame(match_id: int, sp_id: str) -> pd.DataFrame:
    """The freeze-frame of the kick as the model reads it: one row per body with its role (the three jobs for the
    attackers), in the canonical frame of the kick, plotted as (u, v) = (width, depth towards the goal)."""
    g = SetPieceGrammar(L.load_events(match_id), kinds=("corner",))
    kick = g.ev[g.ev["id"] == sp_id].iloc[0]
    frame, target = build_frame(kick, L.load_360(match_id))
    f = CommitmentSignature(frame, target).frame.copy()
    f["u"] = canonical_y(f["y"].to_numpy(), kicked_from_right(float(f.loc[f["is_ball"] == 1, "y"].iloc[0])))
    return f.rename(columns={"x": "v"})


# ── the pitch, StatsBomb yards, goal at the top ──────────────────────────
def draw_pitch(ax: plt.Axes, arc: bool = True) -> None:
    """The lines of the goal end that the figures show, in the card's context grey: goal line and touchlines, the
    penalty box and, unless `arc` is off, its arc, the six-yard box, the spot and the goal with its net."""
    line = dict(color=P.CONTEXT, linewidth=0.9)
    ax.plot([0, 80], [PITCH_X_SB, PITCH_X_SB], solid_capstyle="round", zorder=1, **line)
    ax.plot([0, 0], [60, PITCH_X_SB], solid_capstyle="round", zorder=1, **line)
    ax.plot([80, 80], [60, PITCH_X_SB], solid_capstyle="round", zorder=1, **line)
    for (u0, u1), (v0, v1) in ((BOX_Y, (BOX_X, PITCH_X_SB)), (SIX_YARD_Y, (SIX_YARD_X, PITCH_X_SB))):
        ax.add_patch(Rectangle((u0, v0), u1 - u0, v1 - v0, fill=False, zorder=1, **line))
    ax.add_patch(Rectangle((_POSTS_Y[0], PITCH_X_SB), _POSTS_Y[1] - _POSTS_Y[0], _GOAL_DEPTH, facecolor="white",
                           edgecolor=P.INK2, linewidth=0.9, zorder=1))
    ax.plot(40, GOALMOUTH_X, marker="o", markersize=1.6, color=P.CONTEXT, zorder=1)
    if arc:
        half = np.degrees(np.arccos((GOALMOUTH_X - BOX_X) / 10.0))     # the arc outside the box, radius ten yards
        ax.add_patch(Arc((40, GOALMOUTH_X), 20, 20, theta1=270 - half, theta2=270 + half, zorder=1, **line))


# ── the legend, as Opta Analyst lays it: one row, each key an icon over its label ───
# Icons are drawn in inches on an axes that is exactly the size of the row, so a circle stays a circle.
def _dot(face: str, edge: str = "none", number: str | None = None, ring: str | None = None) -> Icon:
    def draw(ax: plt.Axes, x: float, y: float) -> None:
        if ring:
            ax.add_patch(Circle((x, y), 0.105, facecolor="none", edgecolor=ring, linewidth=1.5))
        ax.add_patch(Circle((x, y), 0.075, facecolor=face, edgecolor=edge, linewidth=0.9))
        if number:
            ax.text(x, y - 0.004, number, ha="center", va="center", fontsize=5.5, fontweight="bold",
                    color=P.INK if face == "white" else "white", gid="inside")
    return draw


def _square(face: str, edge: str) -> Icon:
    return lambda ax, x, y: ax.add_patch(Rectangle((x - 0.065, y - 0.065), 0.13, 0.13, facecolor=face, edgecolor=edge, linewidth=0.9))


def _diamond(face: str) -> Icon:
    return lambda ax, x, y: ax.add_patch(Polygon([(x, y + 0.085), (x + 0.085, y), (x, y - 0.085), (x - 0.085, y)],
                                                 facecolor=face, edgecolor="none"))


def _swatch(face: str, alpha: float, edge: str = "none") -> Icon:
    return lambda ax, x, y: ax.add_patch(Rectangle((x - 0.17, y - 0.07), 0.34, 0.14, facecolor=face, alpha=alpha,
                                                   edgecolor=edge, linewidth=0.9))


def _arrow(color: str) -> Icon:
    return lambda ax, x, y: ax.add_patch(FancyArrowPatch((x - 0.22, y), (x + 0.22, y), arrowstyle="-|>", mutation_scale=7,
                                                         color=color, linewidth=1.2, shrinkA=0, shrinkB=0))


def _dashes() -> Icon:
    """The dashed line of a touch or a carry."""
    return lambda ax, x, y: ax.plot([x - 0.22, x + 0.22], [y, y], color=P.INK2, linewidth=1.0, linestyle=_DASH)


def _triangles(color: str) -> Icon:
    def draw(ax: plt.Axes, x: float, y: float) -> None:
        for k in range(4):
            cx = x - 0.21 + 0.14 * k
            ax.add_patch(Polygon([(cx - 0.05, y - 0.055), (cx + 0.05, y - 0.055), (cx, y + 0.065)], facecolor=color, edgecolor="none"))
    return draw


def legend_row(fig: plt.Figure, items: list) -> None:
    """The legend under the pitch: `items` is a list of (label, icon) spread evenly over the figure's width."""
    ax = fig.add_axes([0, 0, 1, _LEGEND_H / fig.get_figheight()])
    ax.set_xlim(0, P.WIDTH)
    ax.set_ylim(0, _LEGEND_H)
    ax.axis("off")
    for i, (label, icon) in enumerate(items):
        x = P.WIDTH * (i + 0.5) / len(items)
        icon(ax, x, _LEGEND_H * 0.66)
        ax.text(x, _LEGEND_H * 0.27, label, ha="center", va="center", fontsize=6.8, color=P.INK2, linespacing=1.1)


def no_overlap(fig: plt.Figure) -> None:
    """Fails if two labels overlap, or if a label covers a disc it does not sit in. The figures are laid out by hand, and a
    collision is a defect to see here rather than in the document."""
    fig.canvas.draw()
    rend = fig.canvas.get_renderer()
    texts = [t for ax in fig.axes for t in ax.texts if t.get_text().strip() and t.get_gid() != "inside"]
    discs = [p for ax in fig.axes for p in ax.patches if isinstance(p, Circle)]
    boxes = [(t, t.get_window_extent(rend)) for t in texts]
    for i, (a, ba) in enumerate(boxes):
        for b, bb in boxes[i + 1:]:
            if ba.overlaps(bb):
                raise AssertionError(f"labels overlap: {a.get_text()!r} and {b.get_text()!r}")
        for d in discs:
            if ba.overlaps(d.get_window_extent(rend)):
                raise AssertionError(f"label {a.get_text()!r} covers a disc")


# ── Figure: the corner, on the pitch and in the event stream ─────────────
_R = 0.06              # inches, the node of a touch, in the zoom and in the table
_NUM = 5.2             # points, the number inside a node
_WINDOW = (-1.5, 63.5, 101.5, 122.8)     # yards: u0, u1 across, v0, v1 up the pitch; from the corner flag to the far side of the box, from the goal line to its front edge
_ZOOM = (31.0, 49.5, 113.4, 122.8)       # the zone where the touches fall: the goal and the ground in front of it, wide enough for no two nodes to touch
_BALL = ("Pass", "Shot", "Clearance")    # the events that play the ball on; any other touch is a carry
_DASH = (0, (2.5, 2))
_COLS = (("Time", 0.54, BLUE_TINT), ("Player", 0.40, SLATE_TINT), ("Event", 0.70, SLATE_TINT), ("Classified as", 0.76, ORANGE_TINT))


def _node(ax: plt.Axes, xy: tuple, label: str, face: str, edge: str, ring: str | None = None, size: float = _NUM) -> None:
    """A touch: a disc with the number of its row in the event stream inside; `ring` marks the first contact."""
    if ring:
        ax.add_patch(Circle(xy, _R * 1.5, facecolor="none", edgecolor=ring, linewidth=1.5, zorder=7))
    ax.add_patch(Circle(xy, _R, facecolor=face, edgecolor=edge, linewidth=0.9, zorder=8))
    ax.text(xy[0], xy[1] - _R * 0.05, label, ha="center", va="center", fontsize=size, fontweight="bold",
            color=P.INK if face == "white" else "white", zorder=9, gid="inside")


def _look(s: dict, e: dict) -> tuple[str, str]:
    """The node of a touch: orange for a shot after a clearance, white for the attackers' other touches, slate for the defenders'."""
    if e["read"] == "after a clearance":
        return P.AFTER, P.AFTER
    return ("white", P.INK) if e["team"] == s["team"] else (P.INK2, P.INK2)


def _panel(bg: plt.Axes, x: float, y: float, w: float, h: float, title: str) -> None:
    """A dashed frame with its title on the top border, in figure inches."""
    bg.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=0.08", facecolor="none",
                                edgecolor=P.INK2, linewidth=0.7, linestyle=(0, (4, 3))))
    bg.text(x + w / 2, y + h, title, ha="center", va="center", fontsize=7.8, fontweight="bold", color=P.INK,
            bbox=dict(facecolor=P.PAGE, edgecolor="none", pad=2.5))


def _link(ov: plt.Axes, a: np.ndarray, ra: float, b: np.ndarray, rb: float, ball: bool, color: str) -> None:
    """The line from the node at a to the node at b, between their edges: a straight line with an arrow at its end when
    the ball is played on, a dashed line for a touch or a carry. Nodes too close to leave a line between them get none."""
    u = (b - a) / np.hypot(*(b - a))
    a, b = a + u * (ra + 0.012), b - u * (rb + 0.012)
    if np.dot(b - a, u) < (0.09 if ball else 0.07):
        return
    if ball:
        ov.add_patch(FancyArrowPatch(tuple(a), tuple(b), arrowstyle="-|>", mutation_scale=6.5, color=color, linewidth=1.0,
                                     zorder=3, shrinkA=0, shrinkB=0))
    else:
        ov.plot(*np.c_[a, b], color=P.INK2, linewidth=1.0, linestyle=_DASH, zorder=3)


def _zone(fig: plt.Figure, window: tuple, x: float, y: float, w: float) -> Callable[[tuple], np.ndarray]:
    """The `window` of the pitch (yards: u0, u1 across, v0, v1 up the pitch) drawn with the pitch's own lines, `w` inches wide
    with its lower left corner at (x, y) of the figure. A window stops at the front of the box, where the arc starts.
    Returns the converter from a place (v, u) in yards to figure inches."""
    u0, u1, v0, v1 = window
    k = w / (u1 - u0)                                        # inches per yard
    W, H = fig.get_figwidth(), fig.get_figheight()
    ax = fig.add_axes([x / W, y / H, w / W, (v1 - v0) * k / H], zorder=2)
    ax.set_xlim(u0, u1)
    ax.set_ylim(v0, v1)
    ax.axis("off")
    draw_pitch(ax, arc=False)
    if u0 < 0 < u1:
        ax.add_patch(Arc((0, PITCH_X_SB), 2, 2, theta1=270, theta2=360, zorder=1, color=P.CONTEXT, linewidth=0.9))   # the corner flag's
    return lambda p: np.array([x + (p[1] - u0) * k, y + (p[0] - v0) * k])


def _places(evs: list, inch: Callable[[tuple], np.ndarray]) -> list:
    """The touches after the kick as nodes, a receipt and the shot that follows at the same place being one: each with the
    first and last event it holds, its place in figure inches, its numbers and its radius, a ring counting whole."""
    places = []
    for i, e in enumerate(evs[1:], 1):
        if places and evs[places[-1]["first"]]["who"] == e["who"] and evs[places[-1]["first"]]["at"] == e["at"]:
            places[-1]["last"] = i
            places[-1]["nums"].append(i + 1)
        else:
            places.append(dict(first=i, last=i, at=inch(e["at"]), nums=[i + 1]))
    for p in places:
        p["ring"] = evs[p["first"]]["read"] == "first contact"
        p["r"] = _R * 1.5 if p["ring"] else _R
    return places


def _strip(fig: plt.Figure, ov: plt.Axes, s: dict, x: float, y: float, w: float) -> tuple[float, float, float]:
    """The corner from the flag to the far side of the box, `w` inches wide with its lower left corner at (x, y): the kick
    from the flag as an arrow into the first contact, every touch after it a dot at its place (the zoom numbers them) and a
    dashed frame round the zone that the zoom shows. Returns the frame's left and right edges and its lower edge."""
    inch = _zone(fig, _WINDOW, x, y, w)
    evs = s["events"]
    places = _places(evs, inch)
    start = inch(evs[0]["at"])
    _link(ov, start, _R, places[0]["at"], 0.024, True, P.INK2)
    for p in places:
        face, edge = _look(s, evs[p["first"]])
        ov.add_patch(Circle(tuple(p["at"]), 0.024, facecolor=face, edgecolor=edge, linewidth=0.6, zorder=8))
    _node(ov, tuple(start), "1", *_look(s, evs[0]))
    ov.text(start[0] + 0.04, start[1] - _R - 0.08, "kick from the corner flag", ha="left", va="top", fontsize=6.3, color=P.INK2, zorder=9)
    u0, u1, v0, v1 = _ZOOM
    a, b = inch((v0, u0)), inch((v1, u1))
    ov.add_patch(Rectangle(tuple(a), b[0] - a[0], b[1] - a[1], fill=False, edgecolor=P.INK, linewidth=1.3, zorder=6))
    return float(a[0]), float(b[0]), float(a[1])


def _zoom(fig: plt.Figure, ov: plt.Axes, s: dict, x: float, y: float, w: float) -> None:
    """The zone where the touches fall, `w` inches wide with its lower left corner at (x, y): every touch a numbered node at
    its place, the ball played on a straight line with an arrow, a touch or a carry a dashed line, the kick coming in at the
    left edge and the shot that scores ending in the goal. Fails if a touch is outside the zone or two nodes overlap."""
    inch = _zone(fig, _ZOOM, x, y, w)
    u0, u1, v0, v1 = _ZOOM
    evs = s["events"]
    assert all(u0 <= e["at"][1] <= u1 and v0 <= e["at"][0] <= v1 for e in evs[1:]), "a touch falls outside the zoom"
    places = _places(evs, inch)
    assert all(np.hypot(*(p["at"] - q["at"])) >= p["r"] + q["r"] for p, q in itertools.combinations(places, 2)), "two nodes overlap"
    flag, land = inch(evs[0]["at"]), places[0]["at"]
    entry = flag + (land - flag) * ((x - flag[0]) / (land[0] - flag[0]))           # the kick comes in at the left edge
    to_goal = inch((PITCH_X_SB + 0.5, evs[-1]["end"][1]))
    legs = [(entry, 0.0, land, places[0]["r"], True, P.INK2)]  # (from, its radius, to, its radius, ball played on, colour)
    for j, p in enumerate(places):
        e = evs[p["last"]]
        nxt = places[j + 1] if j + 1 < len(places) else None
        legs.append((p["at"], p["r"], nxt["at"] if nxt else to_goal, nxt["r"] if nxt else 0.0, e["kind"] in _BALL,
                     P.AFTER if e["read"] == "after a clearance" else P.INK2))
    for leg in legs:
        _link(ov, *leg)
    for p in places:
        _node(ov, tuple(p["at"]), "·".join(map(str, p["nums"])), *_look(s, evs[p["first"]]),
              ring=P.AFTER if p["ring"] else None, size=_NUM if len("".join(map(str, p["nums"]))) == 1 else 4.6)
    ov.text(entry[0] + 0.05, entry[1] - 0.09, "from the corner flag", ha="left", va="top", fontsize=6.3, color=P.INK2, zorder=9)
    h = (v1 - v0) * w / (u1 - u0)
    ov.text(x + w - 0.1, y + h - 0.1, s["clock"], ha="right", va="top", fontsize=6.8, fontweight="bold", color="white",
            zorder=9, bbox=dict(boxstyle="round,pad=0.3,rounding_size=0.4", facecolor=P.INK, edgecolor="none"))
    ov.add_patch(Rectangle((x, y), w, h, fill=False, edgecolor=P.INK, linewidth=1.3, zorder=6))
    ov.text(x + w / 2, y + h, "Zoom", ha="center", va="center", fontsize=7.8, fontweight="bold", color=P.INK, zorder=9,
            bbox=dict(facecolor=P.PAGE, edgecolor="none", pad=2.5))


def draw_cut(s: dict, out: Path) -> None:
    """The corner event by event. On the left, the corner from the flag to the far side of the box and under it a zoom on the
    zone where the touches fall, each touch numbered by its row in the event stream and the ball in orange after the first
    contact; on the right, the same touches as the rows of the event stream, with the class that the segmentation gives
    each one. The table takes the width its columns need, the map and the zoom the rest, and its rows the height they leave."""
    P.setup()
    evs = s["events"]
    u0, u1, v0, v1 = _WINDOW
    zu0, zu1, zv0, zv1 = _ZOOM
    wt = sum(c[1] for c in _COLS)
    W, pad = P.WIDTH, 0.05
    rw = wt + 0.06
    rx = W - pad - rw
    lw = rx - 0.1 - pad
    w = lw - 0.2                                             # the map and the zoom are as wide as their panel
    strip_h, zoom_h = (v1 - v0) * w / (u1 - u0), (zv1 - zv0) * w / (zu1 - zu0)
    ph = 0.2 + strip_h + 0.14 + zoom_h + 0.12                # the height of both panels
    rh = (ph - 0.77) / (len(evs) - 0.5)                      # one row of the table, in inches
    H = ph + 0.16 + _LEGEND_H + 0.1
    fig = plt.figure(figsize=(W, H))
    bg = fig.add_axes([0, 0, 1, 1], zorder=0)
    bg.set_xlim(0, W)
    bg.set_ylim(0, H)
    bg.axis("off")
    ov = fig.add_axes([0, 0, 1, 1], zorder=4)                # the events, in figure inches, so a node stays round
    ov.set_xlim(0, W)
    ov.set_ylim(0, H)
    ov.axis("off")
    top, bot = H - 0.16, _LEGEND_H + 0.1
    _panel(bg, pad, bot, lw, ph, "On the pitch")
    _panel(bg, rx, bot, rw, ph, "In the event stream")
    zy = bot + 0.12
    left, right, low = _strip(fig, ov, s, pad + 0.1, top - 0.2 - strip_h, w)
    _zoom(fig, ov, s, pad + 0.1, zy, w)
    for x_from, x_to in ((left, pad + 0.1), (right, pad + 0.1 + w)):                # the frame on the map opens onto the zoom
        ov.plot([x_from, x_to], [low, zy + zoom_h], color=P.INK, linewidth=0.8, zorder=3)

    # ── the event stream: the same touches as rows ──
    cols = _COLS
    x0 = rx + (rw - wt) / 2
    yh = top - 0.40                                          # the header chips, then one row each
    x = x0
    for name, wc, tint in cols:
        bg.add_patch(FancyBboxPatch((x + 0.02, yh - 0.11), wc - 0.04, 0.22, boxstyle="round,pad=0,rounding_size=0.05",
                                    facecolor=tint, edgecolor=P.INK2, linewidth=0.6))
        bg.text(x + wc / 2, yh, name, ha="center", va="center", fontsize=6.2, fontweight="bold", color=P.INK)
        x += wc
    for i, e in enumerate(evs):
        y = yh - 0.27 - rh * i
        mark = e["read"] == "first contact" or i == len(evs) - 1
        shot = e["what"].split(",")[0] in ("Shot", "Volley", "Header")
        if mark:
            bg.add_patch(FancyBboxPatch((x0 + 0.02, y - rh / 2 + 0.01), wt - 0.04, rh - 0.02,
                                        boxstyle="round,pad=0,rounding_size=0.05", facecolor=ORANGE_TINT, edgecolor="none"))
        _node(bg, (x0 + 0.13, y), str(i + 1), *_look(s, e), size=_NUM if i < 9 else 4.6)
        bg.text(x0 + 0.26, y, f"{e['t']:.2f} s", ha="left", va="center", fontsize=6.0, fontweight="bold", color=P.INK)
        bg.text(x0 + cols[0][1] + 0.04, y, e["who"], ha="left", va="center", fontsize=6.0, color=P.INK)
        bg.text(x0 + cols[0][1] + cols[1][1] + 0.04, y, e["what"], ha="left", va="center", fontsize=6.0, color=P.INK)
        bg.text(x0 + cols[0][1] + cols[1][1] + cols[2][1] + 0.05, y, e["read"], ha="left", va="center", fontsize=6.0,
                fontweight="bold" if mark or shot else "normal", color=P.AFTER if e["read"] in ("first contact", "after a clearance") else P.INK2)

    other = [m for m in s["match"].split(" v ") if m != s["team"]][0]
    legend_row(fig, [
        (s["team"], _dot("white", P.INK, "1")), (other, _dot(P.INK2, "none", "1")),
        ("First contact", _dot("white", P.INK, "1", ring=P.AFTER)), ("Shot after\na clearance", _dot(P.AFTER, "none", "1")),
        ("Pass, shot\nor clearance", _arrow(P.INK2)), ("Touch\nor carry", _dashes()), ("Attacking\ndirection", _triangles(P.INK2))])
    no_overlap(fig)
    P.save(fig, out)


# ── Figure: the plan ─────────────────────────────────────────────────────
def draw_plan(frame: pd.DataFrame, out: Path) -> None:
    """The eleven cells and the three jobs on the freeze-frame of the kick: the box shaded for the attackers who attack
    the cross, the frontal shaded in orange for those who stake the drop (the strip in front of the box and the central
    block of the box further than G2_EDGE_CUT from goal, the edge), and the rest left to hold back. Fails if the frontal
    drawn is not the one the jobs are read on."""
    P.setup()
    height = 3.05
    fig = plt.figure(figsize=(P.WIDTH, height))
    ax = fig.add_axes([0, _LEGEND_H / height, 1, 1 - _LEGEND_H / height])
    ax.set_xlim(-3, 97)
    ax.set_ylim(88.0, 124.5)
    ax.set_aspect("equal")
    ax.axis("off")
    (u0, u1), (s0, s1), cut = BOX_Y, SIX_YARD_Y, PITCH_X_SB - G2_EDGE_CUT
    v, w = np.meshgrid(np.linspace(HOLD_X - 6, PITCH_X_SB, 145), np.linspace(0, 80, 161))
    drawn = ((v >= HOLD_X) & (v < BOX_X) & (w >= u0) & (w <= u1)) | ((v >= BOX_X) & (v < cut) & (w >= s0) & (w <= s1))
    assert (drawn == frontal(v, w)).all(), "the frontal drawn is not the one the jobs are read on"
    for x, y, wd, ht, face, alpha in ((u0, BOX_X, s0 - u0, PITCH_X_SB - BOX_X, P.INK2, 0.10),
                                      (s1, BOX_X, u1 - s1, PITCH_X_SB - BOX_X, P.INK2, 0.10),
                                      (s0, cut, s1 - s0, PITCH_X_SB - cut, P.INK2, 0.10),
                                      (u0, HOLD_X, u1 - u0, BOX_X - HOLD_X, P.AFTER, 0.18),
                                      (s0, BOX_X, s1 - s0, cut - BOX_X, P.AFTER, 0.18)):
        ax.add_patch(Rectangle((x, y), wd, ht, facecolor=face, alpha=alpha, edgecolor="none", zorder=0))
    draw_pitch(ax)
    grid = dict(color=P.INK2, linewidth=0.6, linestyle=(0, (1.5, 2.2)), zorder=1, alpha=0.8)
    third = (s1 - s0) / 3
    ax.plot([s0, s1], [cut, cut], **grid)
    for u in (s0 + third, s1 - third):
        ax.plot([u, u], [cut, PITCH_X_SB], **grid)
    for u in (s0, s1):
        ax.plot([u, u], [BOX_X, PITCH_X_SB], **grid)
    ax.plot([0, 80], [HOLD_X, HOLD_X], **grid)
    for u in (u0, u1):
        ax.plot([u, u], [HOLD_X, PITCH_X_SB], **grid)
    # headers: the columns over the goal, the depths beside the pitch, the cells that are no column or depth
    lab = dict(fontsize=7, color=P.INK2, va="center")
    for u, text in ((s0 + third / 2, "near"), (40, "central"), (s1 - third / 2, "far")):
        ax.text(u, 123.0, text, ha="center", **lab)
    ax.text(88.5, 123.0, "yards from\nthe goal line", ha="center", fontsize=6.5, color=P.INK2, va="center", linespacing=1.1)
    for v_, text in ((117, "0–6"), (109.5, "6–15"), (103.5, "15–18"), (99, "18–24"), (91.5, "over 24")):
        ax.text(88.5, v_, text, ha="center", **lab)
    ax.text((u0 + s0) / 2, 111, "near\nside", ha="center", linespacing=1.1, **lab)
    ax.text((s1 + u1) / 2, 111, "far\nside", ha="center", linespacing=1.1, **lab)
    ax.text(9, 108, "wide", ha="center", **lab)
    ax.text(71, 108, "wide", ha="center", **lab)
    ax.text(40, 91.5, "deep", ha="center", **lab)
    ax.text(21.5, 99, "edge", ha="left", **lab)
    # the cells numbered, so that they are counted at a glance: the six-yard box is three of them, and the wide areas of both
    # flanks are one (numbered on each side)
    num = dict(fontsize=6.5, fontweight="bold", color=P.INK, ha="left", va="bottom")
    for x_, label in ((s0 + third - 2.0, "1"), (s1 - third - 2.0, "2"), (s1 - 2.0, "3")):      # right corner: clear of the players
        ax.text(x_, 114.6, label, **{**num, "ha": "left"})
    for x_, y_, label in ((s0, cut, "4"), (s0 + third, cut, "5"),
                          (s1 - third, cut, "6"), (u0, BOX_X, "7"), (s1, BOX_X, "8"), (0, HOLD_X, "9"), (u1, HOLD_X, "9"),
                          (u1 - 4, HOLD_X, "10"), (0, 88.0, "11")):
        ax.text(x_ + 0.8, y_ + 0.6, label, **num)

    f = frame
    att = f[f["role"].isin(["attack_cross", "stake_drop", "hold_back"])]
    dfn = f[(f["role"] == "defender") & (f["is_goalkeeper"] == 0)]
    gk = f[(f["role"] == "defender") & (f["is_goalkeeper"] == 1)]
    taker = f[f["role"] == "taker"]
    ax.scatter(att["u"], att["v"], s=24, color=P.INK, zorder=4, linewidths=0)
    ax.scatter(dfn["u"], dfn["v"], s=24, facecolor="white", edgecolor=P.INK2, linewidth=0.9, zorder=4)
    ax.scatter(gk["u"], gk["v"], s=24, marker="s", facecolor="white", edgecolor=P.INK2, linewidth=0.9, zorder=4)
    ax.scatter(taker["u"], taker["v"], s=30, marker="D", color=P.INK, zorder=4, linewidths=0)

    legend_row(fig, [
        ("Attack\nthe cross", _swatch(P.INK2, 0.18)), ("Stake\nthe drop", _swatch(P.AFTER, 0.30)),
        ("Hold\nback", _swatch("none", 1.0, P.CONTEXT)), ("Attacker", _dot(P.INK)),
        ("Defender", _dot("white", P.INK2)), ("Goalkeeper", _square("white", P.INK2)), ("Taker", _diamond(P.INK)),
        ("Attacking\ndirection", _triangles(P.INK2))])
    no_overlap(fig)
    P.save(fig, out)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", type=Path, default=_REPO / "figures")
    args = ap.parse_args()
    match_id, sp_id = pick()
    s = story(match_id, sp_id)
    draw_cut(s, args.out_dir / "corner_cut.png")
    draw_plan(kick_frame(match_id, sp_id), args.out_dir / "corner_plan.png")
    print(f"wrote corner_cut.png and corner_plan.png in {args.out_dir}: {s['team']}, {s['match']}, delivery {sp_id}")


if __name__ == "__main__":
    main()
