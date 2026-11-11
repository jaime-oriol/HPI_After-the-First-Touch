"""render_conversion — how often a shot scores, by what happened before it.

Corners, every attacking shot inside the window, own goals out: the goals per shot of the direct shot
and of the three ways a shot comes after the first touch, then the three together, each with its
interval grouped by match and, under its name, the goals and shots it comes from, which is what the
width of the interval follows. The dashed line is the direct shot's rate, the reference the paper's
contrasts are read against, and says so on the figure. Whether a segment separates from it is the
paired difference of the record's `falsification.delta`, which the paper's table prints: two intervals
that overlap here can still separate there.

Every number printed is the record's `falsification.conversion` block (docs/measured.json), as stored.

    python renders/render_conversion.py
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

from viz import paper as P                               # noqa: E402

_RECORD = _REPO / "docs" / "measured.json"
ROWS = (("direct", "Direct shot"), ("flick_on", "Flick-on"), ("prolongation", "Prolongation"),
        ("after_clearance", "After a clearance"), ("after_first_touch", "After the first touch"))


PARTS = ("flick_on", "prolongation", "after_clearance")


def rows(conv: dict) -> list[dict]:
    """Each row's goals per shot and its interval, in percent, in the figure's order, with the shots and goals
    behind the rate. The roll-up's counts are its three segments' sums, and they must give back the record's rate."""
    counts = {k: (conv[k]["n"], conv[k]["goals"]) for k in ("direct", *PARTS)}
    n, goals = (sum(counts[k][i] for k in PARTS) for i in (0, 1))
    counts["after_first_touch"] = (n, goals)
    assert round(goals / n, 4) == conv["after_first_touch"]["point"], "the three segments no longer make the roll-up"
    return [{"key": k, "label": label, "point": 100 * conv[k]["point"], "lo": 100 * conv[k]["ci"][0],
             "hi": 100 * conv[k]["ci"][1], "n": counts[k][0], "goals": counts[k][1]} for k, label in ROWS]


def draw(rs: list[dict], out: Path) -> None:
    P.setup()
    fig = P.figure(2.4)
    ax = fig.add_subplot()
    for i, r in enumerate(rs):
        c = P.INK2 if r["key"] == "direct" else P.AFTER
        ax.plot([r["lo"], r["hi"]], [i, i], color=c, linewidth=1.4, solid_capstyle="butt", zorder=3)
        ax.scatter(r["point"], i, s=36, color=c, edgecolor=P.PAGE, linewidth=1.2, zorder=4)
        ax.text(r["hi"] + 0.6, i, f"{r['point']:.2f}%", va="center", fontsize=7, fontweight="bold",
                color=P.INK)
    ax.axvline(rs[0]["point"], color=P.INK2, linewidth=0.8, linestyle=(0, (3, 2)), zorder=2)
    ax.text(rs[0]["hi"] + 4.4, 0, "Direct shot rate (dashed line)", ha="left", va="center", fontsize=6.5,
            color=P.INK2)
    # the three segments above the rule, the after-first-touch roll-up below it
    ax.axhline(len(rs) - 1.5, color=P.CONTEXT, linewidth=0.6)
    ax.set_yticks(range(len(rs)), [])
    ax.set_ylim(len(rs) - 0.4, -0.6)
    for i, r in enumerate(rs):      # the category in bold and, under it, the goals and shots behind its rate
        for text, dy, va, style in ((r["label"], 1.5, "bottom", dict(fontsize=7.5, fontweight="bold", color=P.INK)),
                                    (f"{r['goals']:,} goals, {r['n']:,} shots", -1.5, "top",
                                     dict(fontsize=6.5, color=P.INK2))):
            ax.annotate(text, (0, i), xycoords=("axes fraction", "data"), xytext=(-5, dy), textcoords="offset points",
                        ha="right", va=va, **style)
    ax.set_xlim(0, 32)
    ax.set_xticks(range(0, 31, 5))
    ax.xaxis.set_major_formatter(lambda v, _: f"{v:.0f}%")
    ax.set_xlabel("Goals per shot, corners in the five leagues, own goals out")
    P.hairline_grid(ax, "x")
    P.zero_line(ax, "x")
    dot = dict(marker="o", linestyle="", markersize=6, markeredgecolor=P.PAGE)
    fig.legend([Line2D([], [], color=P.INK2, **dot), Line2D([], [], color=P.AFTER, **dot)],
               ["Direct shot", "After the first touch"], loc="outside upper left", ncols=2)
    P.save(fig, out)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=_REPO / "figures" / "conversion.png")
    args = ap.parse_args()
    conv = json.loads(_RECORD.read_text())["figures"]["falsification"]["conversion"]
    draw(rows(conv), args.out)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
