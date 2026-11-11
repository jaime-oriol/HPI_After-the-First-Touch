"""render_plan_premium — the plan's premium on a cleared corner, raw and at the same drop position (H3).

Corners whose delivery a defender cleared. Left, the four coach states (definitions.md §7), a two by
two: whether an attacker holding the stake-drop job at the kick stood within STAKE_RADIUS of where the
next touch met the ball, crossed with which side made that touch, each with the xG that followed per
second ball and its share of them. Right, the plan's premium, won as planned minus rescued: raw, the
drops as they fall, then at the same drop position, inside cells of four sizes and by a regression on
the drop position, each with its 95% interval resampling matches. Flank free kicks stay in the text.

Every number printed is read from the record's `situation` and `drop_position` blocks
(docs/measured.json), as stored and with a fixed number of decimals, so that a trailing zero does not drop out.

    python renders/render_plan_premium.py
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
from matplotlib.axes import Axes                         # noqa: E402
from matplotlib.patches import FancyBboxPatch            # noqa: E402

_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO / "src"))

from viz import paper as P                               # noqa: E402

_RECORD = _REPO / "docs" / "measured.json"
# (key, name, row, column): row 0 has a staker near the drop, column 0 is the attack's next touch
STATES = (("won_as_planned", "Won as planned", 0, 0), ("staked_but_lost", "Staked but lost", 0, 1),
          ("rescued", "Rescued", 1, 0), ("conceded_open", "Conceded open", 1, 1))


def readings(drop: dict) -> list[tuple[str, dict]]:
    """The premium read raw, inside each drop cell size from the smallest, and by the regression; the
    raw and regression readings do not depend on the cell, so they are read once."""
    by = drop["by_cell"]
    cells = sorted(by, key=lambda k: float(k.removesuffix("yd")))
    first = by[cells[0]]
    return ([("Raw", first["raw"])]
            + [(f"Within {k.removesuffix('yd')}-yd cells", by[k]["stratified"]) for k in cells]
            + [("Regression on the drop point", first["adjusted"])])


def _grid(ax: Axes, by_state: dict) -> None:
    ax.set(xlim=(-0.66, 2), ylim=(0, 2.45))
    ax.axis("off")
    for x, head in ((0.5, "Attack's next touch"), (1.5, "Defence's next touch")):
        ax.text(x, 2.3, head, ha="center", va="center", fontsize=7, color=P.INK2)
    for y, head in ((1.55, "Player staked\nnear the drop"), (0.55, "Nobody staked\nnear the drop")):
        ax.text(-0.08, y, head, ha="right", va="center", fontsize=7, color=P.INK2, linespacing=1.2)
    for key, name, row, col in STATES:
        s, x0, y0 = by_state[key], col + 0.04, 1.08 - row + 0.02
        ax.add_patch(FancyBboxPatch((x0, y0), 0.92, 0.94, boxstyle="round,pad=0,rounding_size=0.03",
                                    facecolor="white", edgecolor=P.CONTEXT, linewidth=0.6))
        ax.text(x0 + 0.46, y0 + 0.7, name, ha="center", va="center", fontsize=7.5,
                fontweight="bold", color=P.INK)
        ax.text(x0 + 0.46, y0 + 0.45, f"{s['xg']:.4f} xG", ha="center", va="center", fontsize=7.5,
                fontweight="bold", color=P.INK)
        ax.text(x0 + 0.46, y0 + 0.2, f"{s['share_pct']:.1f}%\nof second balls", ha="center",
                va="center", fontsize=7, color=P.INK2, linespacing=1.15)


def _premium(ax: Axes, rs: list[tuple[str, dict]]) -> None:
    for i, (_, r) in enumerate(rs):
        lo, hi = r["ci"]
        raw_or_reg = i in (0, len(rs) - 1)
        ax.plot([lo, hi], [i, i], color=P.INK2, linewidth=1.4, solid_capstyle="butt", zorder=3)
        ax.scatter(r["gap"], i, s=34, zorder=4, linewidth=1.2,
                   color=P.INK2 if raw_or_reg else P.PAGE, edgecolor=P.INK2)
        ax.text(hi + 0.0006, i, f"{r['gap']:+.5f}".replace("-", "−"), va="center", fontsize=7,
                fontweight="bold", color=P.INK)
    ax.set_yticks(range(len(rs)), [label for label, _ in rs])
    ax.set_ylim(len(rs) - 0.4, -0.6)
    P.bold_categories(ax, "y")
    ax.set_xlabel("Won as planned minus rescued, xG per second ball")
    P.hairline_grid(ax, "x")
    P.zero_line(ax, "x")


def draw(by_state: dict, rs: list[tuple[str, dict]], out: Path) -> None:
    P.setup()
    fig = P.figure(2.7)
    grid, premium = fig.subplots(1, 2, width_ratios=(1.2, 1))
    _grid(grid, by_state)
    fig.text(0.01, 0.985, "xG per second ball, zero when no shot follows", fontsize=7, color=P.INK2, ha="left", va="top")
    _premium(premium, rs)
    P.save(fig, out)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=_REPO / "figures" / "plan_premium.png")
    args = ap.parse_args()
    record = json.loads(_RECORD.read_text())["figures"]
    draw(record["situation"]["by_state"], readings(record["drop_position"]), args.out)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
