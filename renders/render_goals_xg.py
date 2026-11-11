"""render_goals_xg — goals against the provider's expected goals, by phase and by how the shot was struck.

Corners, every attacking shot, own goals out: for each group of shots the goals that scored over the expected
goals the provider's model gave them, with its interval grouped by match, so a point under 1 is a group that the
model overstates. The rows come in three groups: the direct shot and the shots after the first touch, the headers
and the other shots of both phases together, and the four cells of phase by body part. The direct shot is nearly all
headers and what follows the first touch nearly all not, so the first group alone cannot tell the phase from the
body part, and the other two are what can. "Other" pools the feet and the provider's `Other`.

Every number printed is read from the record (docs/measured.json) as stored. The record keeps the shots of
the three segments after the first touch apart, so their count is summed; the ratio of that group is its own
headline entry.

    python renders/render_goals_xg.py
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
from matplotlib.lines import Line2D                      # noqa: E402

_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO / "src"))

from s01_phases import AFTER_FIRST_TOUCH                 # noqa: E402
from viz import paper as P                               # noqa: E402

_RECORD = _REPO / "docs" / "measured.json"
# (key in the record, label, colour, y): the colours say the phase, as everywhere in the paper, and the dark ink
# says both phases together; y leaves a gap between the three groups.
_LAYOUT = (("direct", "Direct shot", "direct", 0.0), ("after_first_touch", "After the first touch", "after", 1.0),
           ("head", "Headers", "both", 2.6), ("non_head", "Other shots", "both", 3.6),
           ("direct|head", "Direct shot, header", "direct", 5.2), ("direct|non_head", "Direct shot, other", "direct", 6.2),
           ("after_first_touch|head", "After the first touch, header", "after", 7.2),
           ("after_first_touch|non_head", "After the first touch, other", "after", 8.2))
_COLOUR = {"direct": P.INK2, "after": P.AFTER, "both": P.INK}


def rows(corner: dict) -> list[dict]:
    """Each row's goals over expected goals, its interval and its shots, in the figure's order, from the
    record's `frontier.corner` block."""
    head, seg, cells, pooled = corner["headline"], corner["segment"], corner["body_part"], corner["body_part_pooled"]
    out = []
    for key, label, colour, y in _LAYOUT:
        if key == "direct":
            point, (lo, hi), n = head["obs_exp_direct"]["point"], head["obs_exp_direct"]["ci"], seg["direct"]["shots"]
        elif key == "after_first_touch":
            point, (lo, hi) = head["obs_exp_after_first_touch"]["point"], head["obs_exp_after_first_touch"]["ci"]
            n = sum(seg[s]["shots"] for s in AFTER_FIRST_TOUCH)
        else:
            cell = pooled[key] if key in pooled else cells[key]
            point, (lo, hi), n = cell["obs_exp"], cell["ci"], cell["shots"]
        out.append({"key": key, "label": label, "colour": colour, "y": y, "point": point, "lo": lo, "hi": hi, "shots": n})
    return out


def draw(rs: list[dict], out: Path) -> None:
    P.setup()
    fig = P.figure(3.1)
    ax = fig.add_subplot()
    for r in rs:
        c = _COLOUR[r["colour"]]
        ax.plot([r["lo"], r["hi"]], [r["y"], r["y"]], color=c, linewidth=1.4, solid_capstyle="butt", zorder=3)
        ax.scatter(r["point"], r["y"], s=36, color=c, edgecolor=P.PAGE, linewidth=1.2, zorder=4)
        ax.text(1.26, r["y"], f"{r['point']:.2f}", va="center", fontsize=7, fontweight="bold", color=P.INK)
        ax.text(1.36, r["y"], f"{r['shots']:,} shots", va="center", fontsize=6.5, color=P.INK2)
    ax.axvline(1.0, color=P.INK2, linewidth=0.8, linestyle=(0, (3, 2)), zorder=2)
    ax.text(1.0, -0.95, "As many goals as expected", ha="center", va="bottom", fontsize=6.5, color=P.INK2)
    for y in (1.8, 4.4):
        ax.axhline(y, color=P.CONTEXT, linewidth=0.6)
    ax.set_yticks([r["y"] for r in rs], [r["label"] for r in rs])
    ax.set_ylim(8.75, -1.0)
    P.bold_categories(ax, "y")
    ax.set_xlim(0.4, 1.62)
    ax.set_xticks([0.4, 0.6, 0.8, 1.0, 1.2])
    ax.set_xlabel("Goals over expected goals, corners in the five leagues, own goals out")
    P.hairline_grid(ax, "x")
    dot = dict(marker="o", linestyle="", markersize=6, markeredgecolor=P.PAGE)
    fig.legend([Line2D([], [], color=_COLOUR[k], **dot) for k in ("direct", "after", "both")],
               ["Direct shot", "After the first touch", "Both phases"], loc="outside upper left", ncols=3)
    P.save(fig, out)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=_REPO / "figures" / "goals_xg.png")
    args = ap.parse_args()
    draw(rows(json.loads(_RECORD.read_text())["figures"]["frontier"]["corner"]), args.out)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
