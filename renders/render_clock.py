"""render_clock — when the goals of a corner arrive: up to the first touch, after it, and later.

One bar of the share of the goals of corners, own goals included, cut in the order a corner runs: the shot that is the
first touch itself, the three kinds of shot after the first touch within eight seconds of it (the paper's subject, in
orange) and the shots later than that. Under the bar, the clock of a corner: the kick, the first touch at its median
and eight seconds after it. The widths are shares of goals and not time, and the figure says so.

Every number is read from the record (`frontier.corner.segment`, `first_contact`, docs/measured.json) and the
eight seconds from src/constants.py; the headline share of goals after the first touch is checked against the sum of the
segments, so the bar and the paper's text cannot differ.

    python renders/render_clock.py            # the paper's, at the template's 6.5 in
    python renders/render_clock.py --talk     # the slide's, at the 9 in between the template's margins
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt                          # noqa: E402
from matplotlib.patches import Rectangle                 # noqa: E402

_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO / "src"))

import constants as C                                    # noqa: E402
import s01_phases as S                                   # noqa: E402
from viz import paper as P                               # noqa: E402

_RECORD = _REPO / "docs" / "measured.json"
# the slide's geometry in inches: 9 in, the width between the template's 0.5 in margins
W, H = 9.0, 3.1
X0, X1 = 0.45, 8.55
BAR_Y, BAR_H, CLOCK_Y = 1.62, 0.62, 0.60
_CELLS = (("first", P.INK2, "Up to the first touch", "the shot is the first touch\n(a direct shot)"),
          ("after", P.AFTER, "After the first touch", "a flick-on, a prolongation or a shot\nafter a clearance, within eight seconds"),
          ("late", P.CONTEXT, "Later", "more than eight\nseconds after it,\nor no contact"))


def shares(record: dict) -> dict[str, float]:
    """Each part's share of the goals of corners, own goals included, in the order a corner runs, and the goals counted."""
    seg = record["frontier"]["corner"]["segment"]
    goals = {k: seg[k]["goals"] + seg[k]["own_goals"] for k in S.SEGMENTS}
    total = sum(goals.values())
    after = sum(goals[k] for k in S.AFTER_FIRST_TOUCH)
    headline = record["frontier"]["corner"]["headline"]["share_goals_after_first_touch"]["point"]
    assert abs(after / total - headline) < 5e-4, (after / total, headline)
    return {"first": goals["direct"] / total, "after": after / total, "late": goals["late"] / total, "n": total}


def _no_overlap(fig: plt.Figure, texts: list) -> None:
    """No label covers another."""
    fig.canvas.draw()
    boxes = [t.get_window_extent(fig.canvas.get_renderer()) for t in texts]
    for i, a in enumerate(boxes):
        for j in range(i + 1, len(boxes)):
            assert not a.overlaps(boxes[j]), (texts[i].get_text(), "overlaps", texts[j].get_text())


def _clock(ax: plt.Axes, edges: list[float], y: float, first_touch_s: float, foot: float, size: float, lw: float, texts: list) -> None:
    """The clock under the bar: the kick, and the bar's two inner edges as the first touch and eight seconds after it."""
    ax.plot([edges[0], edges[3]], [y, y], color=P.INK2, linewidth=lw, zorder=2)
    for x in edges:
        ax.plot([x, x], [y - foot, y + foot], color=P.INK2, linewidth=lw, zorder=2)
    for x, ha, text in ((edges[0], "left", "Kick"),
                        (edges[1], "center", f"First touch, a median of {first_touch_s:.1f} s after the kick"),
                        (edges[2], "center", f"{C.SECOND_BALL_WINDOW_S:.0f} s after the first touch")):
        texts.append(ax.text(x, y - 2.6 * foot, text, ha=ha, va="center", color=P.INK, fontsize=size, fontweight="bold"))


def draw_talk(s: dict[str, float], first_touch_s: float, out: Path) -> None:
    """The slide's version: 9 in wide with the shares as large numerals, for a room."""
    P.setup()
    fig = plt.figure(figsize=(W, H), facecolor=P.PAGE)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, W)
    ax.set_ylim(0, H)
    ax.axis("off")
    span = X1 - X0
    edges = [X0, X0 + s["first"] * span, X0 + (s["first"] + s["after"]) * span, X1]
    texts: list = []
    for k, (key, color, head, sub) in enumerate(_CELLS):
        a, b = edges[k], edges[k + 1]
        ax.add_patch(Rectangle((a + 0.02, BAR_Y), b - a - 0.04, BAR_H, facecolor=color, edgecolor="none", zorder=2))
        c, is_after = (a + b) / 2, key == "after"
        texts.append(ax.text(c, 2.62 if is_after else 2.5, f"{100 * s[key]:.1f}%", ha="center", va="center",
                             color=P.INK, fontsize=46 if is_after else 28, fontweight="bold"))
        texts.append(ax.text(c, 1.40, head, ha="center", va="center", color=P.INK, fontsize=13.5, fontweight="bold"))
        texts.append(ax.text(c, 0.96, sub, ha="center", va="center", color=P.INK2, fontsize=10.5, linespacing=1.15))
    for x in edges[1:3]:
        ax.plot([x, x], [BAR_Y - 0.02, CLOCK_Y], color=P.INK2, linewidth=0.8, linestyle=(0, (1.5, 2.5)), zorder=1)
    _clock(ax, edges, CLOCK_Y, first_touch_s, 0.07, 10.5, 1.1, texts)
    note = f"Corners in the five big leagues, 2025/26; own goals included; {s['n']} goals. Widths are shares of goals, not time."
    texts.append(ax.text(X1, 0.08, note, ha="right", va="center", color=P.INK2, fontsize=8.5))
    _no_overlap(fig, texts)
    P.save(fig, out)


def draw_paper(s: dict[str, float], first_touch_s: float, out: Path) -> None:
    """The paper's version: the template's 6.5 in and the type of the other figures, 8 pt in ink for what is named and
    7 pt in grey for what defines it, a thin bar with its shares printed on it, and the clock under it."""
    P.setup()
    height = 1.72
    fig = plt.figure(figsize=(P.WIDTH, height), facecolor=P.PAGE)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, P.WIDTH)
    ax.set_ylim(0, height)
    ax.axis("off")
    x0, x1, y0, h = 0.2, P.WIDTH - 0.2, 0.98, 0.34
    span = x1 - x0
    edges = [x0, x0 + s["first"] * span, x0 + (s["first"] + s["after"]) * span, x1]
    texts: list = []
    for k, (key, color, head, sub) in enumerate(_CELLS):
        a, b = edges[k], edges[k + 1]
        ax.add_patch(Rectangle((a + 0.012, y0), b - a - 0.024, h, facecolor=color, edgecolor="none", zorder=2))
        c = (a + b) / 2
        texts.append(ax.text(c, y0 + h + 0.2, head, ha="center", va="center", color=P.INK, fontsize=8, fontweight="bold"))
        texts.append(ax.text(c, y0 + h / 2, f"{100 * s[key]:.1f}%", ha="center", va="center", fontsize=9, fontweight="bold",
                             color="white" if key == "first" else P.INK, zorder=3))
        texts.append(ax.text(c, y0 - 0.21, sub, ha="center", va="center", color=P.INK2, fontsize=7, linespacing=1.15))
    for x in edges[1:3]:
        ax.plot([x, x], [y0 - 0.02, 0.44], color=P.INK2, linewidth=0.6, linestyle=(0, (1.5, 2.5)), zorder=1)
    _clock(ax, edges, 0.44, first_touch_s, 0.04, 7, 0.7, texts)
    note = f"{s['n']} goals of corners in the five leagues, own goals included. Widths are shares of goals, not time."
    texts.append(ax.text(x1, 0.08, note, ha="right", va="center", color=P.INK2, fontsize=6.5))
    _no_overlap(fig, texts)
    P.save(fig, out)


def draw(s: dict[str, float], first_touch_s: float, out: Path, talk: bool = False) -> None:
    (draw_talk if talk else draw_paper)(s, first_touch_s, out)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--talk", action="store_true", help="the slide's version: 9 in wide, larger type")
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()
    out = args.out or _REPO / "figures" / ("talk" if args.talk else "") / "clock.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    record = json.loads(_RECORD.read_text())["figures"]
    draw(shares(record), record["first_contact"]["median_time_to_first_contact_s"], out, args.talk)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
