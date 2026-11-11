"""render_window — why the window is eight seconds: the possession chain of a delivery holds through the ninth second, and the cross of a corner played short through the twelfth.

Read from the first contact, second by second, the share of shots that still belong to the possession chain of their delivery stays
at or above the 95% that a finding needs through the ninth second and falls below it in the tenth, when the window begins to count the
next attack of whoever won the ball (docs/definitions.md §5). The cutoff of eight seconds sits a second before the last second that
holds, a margin for the plays whose chain breaks earlier than it does on average. The upper panel draws that curve, with the shots of
each second under it, the seconds inside the window in the paper's orange and the ones after it in slate, and marks the cutoff and the
95%.

The two lower panels ask the same of the wait for the cross that delivers a corner played short, counted from the kick, one measure
each. On the left, the share of the side's crosses still in the corner's possession chain, by second, with the cutoff that the rule of
the shots would give (a second before the last second that holds). On the right, the share of the corner's first crosses that a
defender had touched the ball before, in the bins of the log, with the bins inside the window in orange: it is where eight seconds,
and not the chain's eleven, stops being the routine.

Both curves are the measured quantities of the paper that have no generated twin in the record, because each needs the possession chain
of every shot or cross and no block of `measured.json` carries it. Their scripts and logs are `late_boundary_curve.py` and `.log` in
data/derived/auditoria_2026-10-12/evidence/ and `cross_window_curve.py` and `.log` in data/derived/prereg_2026-11-04/evidence/, and this
figure reads the logs, so it fails if one is not there or is not the curve it expects.

    python renders/render_window.py
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import pandas as pd

_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO / "src"))

from constants import SECOND_BALL_WINDOW_S, SHORT_CROSS_WINDOW_S      # noqa: E402
from viz import paper as P                                           # noqa: E402

LOG = _REPO / "data" / "derived" / "auditoria_2026-10-12" / "evidence" / "late_boundary_curve.log"
CROSS_LOG = _REPO / "data" / "derived" / "prereg_2026-11-04" / "evidence" / "cross_window_curve.log"
SHOWN_S = 20                  # seconds drawn: the fall is clear by then and each second still holds some fifty shots
FINDING_PCT = 95.0            # the contiguity that a finding needs (docs/definitions.md §5 and §6)
HEIGHT = 4.4                  # in: two rows of panels, the second split in two, so that the annex stays on one page
_PANEL = 1.25                 # in, the height of every panel's axes
_WORDS = {8: "eight", 11: "eleven"}     # the seconds the figure names; a curve that moves them fails here
_ROW = re.compile(r"^\s*(\d+)\s+(\d+)\s+(\d+(?:\.\d+)?)\s*$")
_CROSS_ROW = re.compile(r"^\s*(\d+)\s+(\d+)\s+(\d+(?:\.\d+)?)\s+(\d+)\s+(\d+(?:\.\d+)?)\s*$")
_BIN = re.compile(r"^\s*(\d+)-(\d+)\s+s\s+(\d+)\s+(\d+(?:\.\d+)?)% after a defending touch")


def read_curve(path: Path = LOG) -> pd.DataFrame:
    """The log's table, one row per second after the first contact: the shots taken in it and the share of them still in the
    possession chain of their delivery, in percent."""
    rows = [tuple(map(float, m.groups())) for line in path.read_text().splitlines() if (m := _ROW.match(line))]
    curve = pd.DataFrame(rows, columns=["second", "shots", "share"]).astype({"second": int, "shots": int})
    assert list(curve["second"]) == list(range(1, len(curve) + 1)) and len(curve) >= SHOWN_S, f"{path} is not the curve by second"
    return curve.set_index("second")


def read_cross(path: Path = CROSS_LOG) -> tuple:
    """The cross log's two tables: one row per second from the kick of a corner played short (the side's crosses, the share of them
    still in the corner's possession chain, the corner's first crosses, the share of those that a defender touched the ball before),
    and the first crosses in the log's bins as (from, to, first crosses, percent after a defending touch)."""
    lines = path.read_text().splitlines()
    rows = [tuple(map(float, m.groups())) for line in lines if (m := _CROSS_ROW.match(line))]
    cross = pd.DataFrame(rows, columns=["second", "crosses", "chain_pct", "first_crosses", "defender_pct"]).astype(
        {"second": int, "crosses": int, "first_crosses": int}).set_index("second")
    assert list(cross.index) == list(range(1, len(cross) + 1)) and len(cross) >= SHOWN_S, f"{path} is not the curve by second"
    bins = [(int(m[1]), int(m[2]), int(m[3]), float(m[4])) for line in lines if (m := _BIN.match(line))]
    assert bins and bins[0][0] == 0, f"{path} holds no bins"
    return cross, bins


def _axes(fig, left: float, width: float, top: float):
    """An axes `_PANEL` inches high with its top edge `top` inches below the top of the figure; left and width are fractions of the
    figure's width."""
    return fig.add_axes([left, 1 - (top + _PANEL) / HEIGHT, width, _PANEL / HEIGHT])


def _chain_line(ax, share: pd.Series, window: int) -> None:
    """A share by second as a line, its seconds inside the window in the paper's orange and the ones after it in slate."""
    inside, outside = share.loc[:window], share.loc[window:]
    ax.plot(share.index, share.values, color=P.CONTEXT, linewidth=1.2, zorder=2)
    ax.plot(inside.index, inside.values, color=P.AFTER, linewidth=1.6, zorder=3)
    ax.scatter(outside.index[1:], outside.values[1:], s=14, color=P.INK2, zorder=4, linewidths=0)
    ax.scatter(inside.index, inside.values, s=14, color=P.AFTER, zorder=4, linewidths=0)


def _by_second(ax, ylim: tuple, yticks, title: str, ylabel: str, xlabel: str) -> None:
    """The frame of a panel whose x axis is the seconds: title over its left edge, hairline grid, percent ticks."""
    ax.set_xlim(0.4, SHOWN_S + 1.6)
    ax.set_ylim(*ylim)
    ax.set_xticks(range(1, SHOWN_S + 1))
    ax.set_yticks(list(yticks))
    ax.set_yticklabels([f"{v}%" for v in yticks])
    ax.set_title(title, loc="left", fontsize=7.5, fontweight="bold", color=P.INK, pad=5)
    ax.set_ylabel(ylabel)
    ax.set_xlabel(xlabel, labelpad=18)
    P.hairline_grid(ax, "y")
    ax.tick_params(axis="x", length=0)


def _finding_line(ax) -> None:
    """The 95% that a finding needs, dashed, with its label at the right end."""
    ax.axhline(FINDING_PCT, color=P.INK2, linewidth=0.7, linestyle=(0, (3, 2)), zorder=1)
    ax.text(SHOWN_S + 1.5, FINDING_PCT + 0.8, f"{FINDING_PCT:g}%", fontsize=6.5, color=P.INK2, ha="right", va="bottom")


def _counts_row(ax, label: str, xs, counts, label_x: float) -> None:
    """The count behind each mark, as a row under the axis (in points, so that it clears the tick labels at any panel height)."""
    where = dict(xycoords=("data", "axes fraction"), xytext=(0, -19), textcoords="offset points", va="center", color=P.INK2)
    ax.annotate(label, (label_x, 0), fontsize=6.5, ha="right", **where)
    for x, n in zip(xs, counts, strict=True):
        ax.annotate(f"{n:,}", (x, 0), fontsize=5.4, ha="center", **where)


def draw_shots(ax, curve: pd.DataFrame) -> None:
    """The upper panel: shots from the first contact, by second, with the 95% and the cutoff."""
    seen = curve.loc[:SHOWN_S]
    _chain_line(ax, seen["share"], int(SECOND_BALL_WINDOW_S))
    _finding_line(ax)
    cutoff = SECOND_BALL_WINDOW_S + 0.5
    ax.axvline(cutoff, color=P.INK, linewidth=0.9, zorder=1)
    ax.text(cutoff - 0.25, 56, f"Cutoff,\n{_WORDS[int(SECOND_BALL_WINDOW_S)]} seconds", fontsize=7, fontweight="bold",
            color=P.INK, ha="right", va="center", linespacing=1.1)
    # the last second that holds, above its point, and the first that does not, below its own
    ax.text(9, seen.loc[9, "share"] + 2.2, f"{seen.loc[9, 'share']:.1f}%", fontsize=6.5, fontweight="bold", color=P.INK, ha="center", va="bottom")
    ax.text(10, seen.loc[10, "share"] - 2.2, f"{seen.loc[10, 'share']:.1f}%", fontsize=6.5, fontweight="bold", color=P.INK, ha="center", va="top")
    _by_second(ax, (45, 105), range(50, 101, 10), "Shots still in the corner's possession", "Share of shots",
               "Seconds from the first contact")
    _counts_row(ax, "Shots", seen.index, seen["shots"], 0.2)


def draw_chain(ax, cross: pd.DataFrame) -> None:
    """The lower left panel: the side's crosses after a corner played short, by second from the kick, still in the corner's possession
    chain, with the cutoff of the episode and the one that the same rule would give (a second before the last second that holds)."""
    seen = cross.loc[:SHOWN_S]
    _chain_line(ax, seen["chain_pct"], int(SHORT_CROSS_WINDOW_S))
    _finding_line(ax)
    cutoff = SHORT_CROSS_WINDOW_S + 0.5
    ax.axvline(cutoff, color=P.INK, linewidth=0.9, zorder=1)
    ax.text(cutoff - 0.25, 106, f"Cutoff,\n{_WORDS[int(SHORT_CROSS_WINDOW_S)]} seconds", fontsize=7, fontweight="bold",
            color=P.INK, ha="right", va="center", linespacing=1.1)
    held = last_second_that_holds(seen["chain_pct"])
    ax.axvline(held - 0.5, color=P.INK2, linewidth=0.9, linestyle=(0, (3, 2)), zorder=1)
    ax.text(held - 0.25, 106, f"Same rule as for shots:\n{_WORDS[held - 1]} seconds", fontsize=7, color=P.INK2, ha="left", va="center",
            linespacing=1.1)
    ax.text(held + 1, seen.loc[held + 1, "chain_pct"] - 2.2, f"{seen.loc[held + 1, 'chain_pct']:.1f}%", fontsize=6.5, fontweight="bold",
            color=P.INK, ha="center", va="top")
    _by_second(ax, (70, 114), range(70, 101, 10), "Crosses still in the corner's possession", "Share of crosses",
               "Seconds from the kick of a corner played short")
    _counts_row(ax, "Crosses", seen.index, seen["crosses"], 0.2)


def draw_touch(ax, bins: list) -> None:
    """The lower right panel: the share of the corner's first crosses that a defender had touched the ball before, by bin of seconds
    from the kick, the bins inside the window in orange and the ones after it in slate, with the first crosses of each under it."""
    window = SHORT_CROSS_WINDOW_S
    for i, (_, hi, _, pct) in enumerate(bins):
        ax.bar(i, pct, width=0.62, color=P.AFTER if hi <= window else P.INK2, zorder=3)
        ax.text(i, pct + 1.5, f"{pct:.1f}%", fontsize=6.5, fontweight="bold", color=P.INK, ha="center", va="bottom")
    edge = sum(hi <= window for _, hi, _, _ in bins) - 0.5          # between the last bin inside the window and the first outside it
    ax.axvline(edge, color=P.INK, linewidth=0.9, zorder=1)
    ax.text(edge - 0.1, 45, f"Cutoff,\n{_WORDS[int(window)]} seconds", fontsize=7, fontweight="bold", color=P.INK, ha="right",
            va="center", linespacing=1.1)
    ax.set_xlim(-0.6, len(bins) - 0.4)
    ax.set_ylim(0, 50)
    ax.set_xticks(range(len(bins)))
    ax.set_xticklabels([f"{lo}–{hi}" for lo, hi, _, _ in bins])
    ax.set_yticks([0, 20, 40])
    ax.set_yticklabels([f"{v}%" for v in (0, 20, 40)])
    ax.set_title("First crosses after a defender's touch", loc="left", fontsize=7.5, fontweight="bold", color=P.INK, pad=5)
    ax.set_ylabel("Share of first crosses")
    ax.set_xlabel("Seconds from the kick to the cross", labelpad=18)
    P.hairline_grid(ax, "y")
    ax.tick_params(axis="x", length=0)
    _counts_row(ax, "First crosses", range(len(bins)), [n for _, _, n, _ in bins], -0.45)


def draw(curve: pd.DataFrame, cross: pd.DataFrame, bins: list, out: Path) -> None:
    P.setup()
    fig = P.figure(HEIGHT, layout=None)
    draw_shots(_axes(fig, 0.10, 0.88, 0.30), curve)
    draw_chain(_axes(fig, 0.10, 0.47, 2.38), cross)
    draw_touch(_axes(fig, 0.685, 0.295, 2.38), bins)
    P.save(fig, out)


def last_second_that_holds(share: pd.Series) -> int:
    """The last second of the unbroken run, from the first, whose share is at or above the finding line."""
    return int((share >= FINDING_PCT).cummin().sum())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=_REPO / "figures" / "window.png")
    args = ap.parse_args()
    if not LOG.exists():
        raise SystemExit(f"{LOG} is not there: run data/derived/auditoria_2026-10-12/evidence/late_boundary_curve.py corner")
    if not CROSS_LOG.exists():
        raise SystemExit(f"{CROSS_LOG} is not there: run data/derived/prereg_2026-11-04/evidence/cross_window_curve.py")
    curve = read_curve()
    cross, bins = read_cross()
    draw(curve, cross, bins, args.out)
    held, held_cross = last_second_that_holds(curve["share"]), last_second_that_holds(cross["chain_pct"])
    print(f"wrote {args.out}: shots, the share stays at or above {FINDING_PCT:g}% through second {held} and is "
          f"{curve.loc[held + 1, 'share']:.1f}% in second {held + 1}; crosses, through second {held_cross} and "
          f"{cross.loc[held_cross + 1, 'chain_pct']:.1f}% in second {held_cross + 1}")


if __name__ == "__main__":
    main()
