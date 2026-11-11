"""render_plan_race — how the second balls of the tracked corners were won, and when the players near the drop moved.

The figure of Section 5.1, from run_plan_race's tables in data/derived, in the layout of the paper's other charts. Left,
the three ways the winning side got to the drop, as shares of every second ball: it already had the player nearest the
drop when the corner was taken, it got one nearest while the ball was in the air before the clearance, or it won the run
after the clearance; the dashed line on the first bar is the share that no relation between being nearest at the kick
and winning would give. Right, the players near the drop whom the tracking detects throughout: the share of each side
already moving towards it a second before the clearance, at it, and one and two seconds after it.

The cases are counted after the fact, with the drop known, and the intervals live in the paper's Table 9.

    python renders/render_plan_race.py
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import numpy as np                                        # noqa: E402
import pandas as pd                                       # noqa: E402
from matplotlib.axes import Axes                          # noqa: E402

_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO / "src"))
sys.path.insert(0, str(_REPO / "renders"))

import render_example as ex                               # noqa: E402
from viz import paper as P                                # noqa: E402

_DER = _REPO / "data" / "derived"
# (record key, label, colour): how the winning side got to the drop; orange is what happens after the clearance
CASE_STYLE = (("plan", "Already nearest at the kick", P.CONTEXT), ("flight", "Got nearest in the air", P.INK2),
              ("race", "Won the run after the clearance", P.AFTER))


def read(der: Path = _DER) -> tuple[pd.DataFrame, pd.DataFrame]:
    """(the record of run_plan_race, its onsets of the fully detected players)."""
    rec = pd.read_parquet(der / "plan_race.parquet").set_index(["block", "key"])
    onsets = pd.read_parquet(der / "plan_race_onsets.parquet")
    return rec, onsets[onsets["fully_detected"]]


def _pct(share: float) -> str:
    """A share as a whole percentage, rounded the way a reader rounds: 12.5% prints 13%."""
    return f"{int(np.floor(100 * share + 0.5))}%"


def _cases(ax: Axes, rec: pd.DataFrame) -> None:
    for i, (key, label, colour) in enumerate(CASE_STYLE):
        share, n = float(rec.loc[("share", key), "point"]), int(rec.loc[("share", key), "n"])
        ax.barh(i, 100 * share, height=0.55, color=colour, zorder=3)
        ax.text(100 * share + 1.5, i, _pct(share), va="center", fontsize=7, fontweight="bold", color=P.INK)
        for text, dy, va, style in ((label, 1.5, "bottom", dict(fontsize=7.5, fontweight="bold", color=P.INK)),
                                    (f"{n} second balls", -1.5, "top", dict(fontsize=6.5, color=P.INK2))):
            ax.annotate(text, (0, i), xycoords=("axes fraction", "data"), xytext=(-5, dy), textcoords="offset points",
                        ha="right", va=va, **style)
    none = float(rec.loc[("no_relation", "expected"), "point"])
    ax.plot([100 * none] * 2, [-0.36, 0.36], color=P.INK2, linewidth=0.8, linestyle=(0, (3, 2)), zorder=4)
    ax.text(1, -0.62, f"Dashed line: {_pct(none)} if being nearest at the kick did not matter", ha="left", va="center",
            fontsize=6.5, color=P.INK2)
    ax.set_yticks(range(len(CASE_STYLE)), [])
    ax.set_ylim(len(CASE_STYLE) - 0.5, -0.85)
    ax.set_xlim(0, 100)
    ax.set_xticks(range(0, 101, 25))
    ax.xaxis.set_major_formatter(lambda v, _: f"{v:.0f}%")
    ax.set_xlabel(f"Share of the {int(rec.loc[('population', 'analysed'), 'n'])} second balls")
    P.hairline_grid(ax, "x")
    P.zero_line(ax, "x")


def _moving(onsets: pd.DataFrame, side: str) -> list[float]:
    """The share of one side's players already moving towards the drop 1 s before the clearance, at it, and 1 and 2 s
    after it: the window's first frame, then the onsets up to each moment, the players never started left out."""
    sub = onsets[onsets["side"] == side]
    started = sub.loc[~sub["censored"], "onset_s"]
    return [float(sub["already_moving"].mean()), *[float((started <= t).sum() / len(sub)) for t in (0.0, 1.0)],
            float((~sub["censored"]).mean())]


def _starts(ax: Axes, onsets: pd.DataFrame) -> list:
    """The two sides' lines; returns their handles for the legend."""
    moments = np.array([-1.0, 0.0, 1.0, 2.0])
    handles = []
    # attackers filled and defenders hollow, as on the paper's pitches, the attackers' dots on top where the two meet;
    # both lines rise, so the attackers' values sit below and right of their points and the defenders' above and left
    for side, name, colour, face, sign in (("att", "Attackers", P.INK, P.INK, -1), ("def", "Defenders", P.INK2, P.PAGE, 1)):
        shares = _moving(onsets, side)
        ys = 100 * np.array(shares)
        handles += ax.plot(moments, ys, color=colour, linewidth=1.4, marker="o", markersize=4.5, markerfacecolor=face,
                           markeredgecolor=colour, markeredgewidth=1.0, zorder=5 if sign < 0 else 4,
                           label=f"{name} ({int((onsets['side'] == side).sum())} players)")
        for x, y, share in zip(moments, ys, shares, strict=True):
            ax.annotate(_pct(share), (x, y), xytext=(-4 * sign, 5 * sign), textcoords="offset points",
                        ha="right" if sign > 0 else "left", va="bottom" if sign > 0 else "top", fontsize=6.5,
                        fontweight="bold", color=P.INK)
    ax.axvline(0, color=P.INK, linewidth=0.9, zorder=2)
    ax.text(0.06, 6, "Clearance", ha="left", va="center", fontsize=7, fontweight="bold", color=P.INK)
    ax.set_xlim(-1.55, 2.55)
    ax.set_xticks(moments, ["−1", "0", "+1", "+2"])
    ax.set_xlabel("Seconds from the clearance")
    ax.set_ylabel("Already moving\ntowards the drop")
    ax.set_ylim(0, 100)
    ax.set_yticks(range(0, 101, 25))
    ax.yaxis.set_major_formatter(lambda v, _: f"{v:.0f}%")
    P.hairline_grid(ax, "y")
    P.zero_line(ax, "y")
    return handles


def draw(rec: pd.DataFrame, onsets: pd.DataFrame, out: Path) -> None:
    P.setup()
    fig = P.figure(2.5)
    cases, starts = fig.subplots(1, 2, width_ratios=(1.1, 1))
    _cases(cases, rec)
    fig.legend(handles=_starts(starts, onsets), loc="outside upper right", ncols=2)
    ex.no_overlap(fig)
    P.save(fig, out)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=_REPO / "figures" / "plan_race.png")
    ap.add_argument("--tables", type=Path, default=_DER, help="where run_plan_race wrote its tables")
    args = ap.parse_args()
    draw(*read(args.tables), args.out)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
