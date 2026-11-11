"""render_flank_cut — where a flank free kick stops being a corner: the annex behind the x >= 85 cut.

The flank free kicks under the paper's predicate from x = 60, in 5-yard bands of the taker's x
(run_flank_cut), against long corners as the dashed reference:

  left, above   production per delivery: in-window xG and after-first-touch xG, 95% by match
  left, below   the attackers' three jobs at the kick, over the kicks carrying a 360
  right         below the cut minus above it, at 85 (the paper's) and at 90, paired over the same
                match draws: xG per delivery above, shares of deliveries below

Marks and measured numbers alike are read from the record (docs/measured.json, block flank_cut),
which carries every band and cut drawn; the cuts at 85 and 90 are run_flank_cut's. It shows why the
paper cuts at 85: below it the kick changes how it is played, not only how far it travels.

    python renders/render_flank_cut.py
    python renders/render_flank_cut.py --out /tmp/flank_cut.png
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO / "src"))
_RECORD = _REPO / "docs" / "measured.json"

from viz.common import ATT, GRID, LEGEND, ROLE_COLORS, TEXT, draw_header   # noqa: E402

# hold_back's slate is 2.56:1 on white, under the 3:1 a mark needs: render_scope's darker step.
_OTHER, _BAND, _BAND_W = "#64748b", "#eef2f7", 5.0
_PRODUCTION = (("xg_win", "xG in the window", TEXT), ("xg_aft", "xG after the first touch", ATT))
_JOBS = (("n_attack_cross", "attack the cross", ROLE_COLORS["attack_cross"]),
         ("n_stake_drop", "stake the drop", ROLE_COLORS["stake_drop"]),
         ("n_hold_back", "hold back", _OTHER))
_CUT_ROWS = (("xG per delivery", (("xg_aft", "xG after the first touch"), ("xg_direct", "direct xG"))),
             ("Share of deliveries", (("p_aft", "with a shot after the first touch"),
                                      ("p_prol", "with a flick-on or prolongation shot"),
                                      ("won_fc", "the attack wins the first contact"))))


def num(v: float) -> str:
    """A record value as it is stored, signed and positional: no second rounding, no 7e-05."""
    return np.format_float_positional(v, trim="-", sign=True)


def bands(block: dict) -> list[tuple[float, dict]]:
    """The flank bands as (left edge in yards, entry), in order of x."""
    return sorted((float(k.split("-")[0]), v) for k, v in block["bands"].items()
                  if k != "long_corners")


def _style(ax: plt.Axes, title: str, ylabel: str = "") -> None:
    ax.set_title(title, loc="left", fontsize=15, fontweight=700, color=TEXT, pad=12)
    ax.set_ylabel(ylabel, fontsize=12, color=LEGEND)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=LEGEND, labelsize=10)
    ax.grid(axis="y", color=GRID, lw=0.8)
    ax.set_axisbelow(True)


def _cut_marks(ax: plt.Axes, top: float, label: bool) -> None:
    """The paper's population shaded from 85, the cut there and the robustness cut at 90."""
    ax.axvspan(85, 120, color=_BAND, zorder=0)
    ax.axvline(85, color=TEXT, lw=1.4, zorder=1)
    ax.axvline(90, color=TEXT, lw=1.2, ls=(0, (4, 3)), zorder=1)
    if label:
        ax.text(84.4, top, "the paper's cut, x = 85", ha="right", va="top", fontsize=10.5,
                color=TEXT)
        ax.text(90.6, top, "90", ha="left", va="top", fontsize=10.5, color=TEXT)


def _x_axis(ax: plt.Axes, b: list) -> None:
    """Band centres as ticks, each labelled with its range and its deliveries."""
    ax.set_xticks([lo + _BAND_W / 2 for lo, _ in b],
                  [f"{lo:.0f}-{lo + _BAND_W:.0f}\n{v['deliveries']}" for lo, v in b], fontsize=9.5)
    ax.set_xlim(b[0][0], b[-1][0] + _BAND_W + 9)
    ax.set_xlabel("Taker's x in yards, the goal line at 120 (deliveries per band below each range)",
                  fontsize=11, color=LEGEND)


def _series(ax: plt.Axes, b: list, ref: dict, key: str, label: str, color: str, ci: bool) -> None:
    """One measure across the bands, its long-corner value dashed and named at the right end."""
    x = [lo + _BAND_W / 2 for lo, _ in b]
    if ci:
        pts = [v[key]["point"] for _, v in b]
        err = [[v[key]["point"] - v[key]["ci"][0] for _, v in b],
               [v[key]["ci"][1] - v[key]["point"] for _, v in b]]
        ref_val = ref[key]["point"]
    else:
        pts, err, ref_val = [v[key] for _, v in b], None, ref[key]
    ax.errorbar(x, pts, yerr=err, fmt="o-", ms=7, lw=1.8, color=color, elinewidth=1.6, capsize=0,
                mec="white", mew=1.2, zorder=3, label=label)
    ax.axhline(ref_val, color=color, lw=1.4, ls=(0, (6, 3)), zorder=2)
    ax.annotate("long corners", (b[-1][0] + _BAND_W + 0.6, ref_val), xytext=(0, 2),
                textcoords="offset points", ha="left", va="bottom", fontsize=10, color=color)


def panel_production(ax: plt.Axes, block: dict) -> None:
    b, ref = bands(block), block["bands"]["long_corners"]
    for key, label, color in _PRODUCTION:
        _series(ax, b, ref, key, label, color, ci=True)
    top = 1.18 * max(max(v["xg_win"]["ci"][1] for _, v in b), ref["xg_win"]["point"])
    ax.set_ylim(0, top)
    _cut_marks(ax, top * 0.98, label=True)
    ax.legend(frameon=False, fontsize=11, loc="upper left")
    _x_axis(ax, b)
    _style(ax, "Production per delivery, 95% interval by match", "xG per delivery")


def panel_jobs(ax: plt.Axes, block: dict) -> None:
    b, ref = bands(block), block["bands"]["long_corners"]
    for key, label, color in _JOBS:
        _series(ax, b, ref, key, label, color, ci=False)
    top = 1.15 * max(max(v[k] for _, v in b for k, _, _ in _JOBS), max(ref[k] for k, _, _ in _JOBS))
    ax.set_ylim(0, top)
    _cut_marks(ax, top * 0.98, label=False)
    ax.legend(frameon=False, fontsize=11, loc="upper left", ncol=3)
    _x_axis(ax, b)
    _style(ax, "The attackers' three jobs at the kick, over the kicks with a 360",
           "attackers per kick")


def panel_cuts(ax: plt.Axes, block: dict, title: str, rows: tuple) -> None:
    """Below minus above for each measure, the paper's cut filled and 90 open; the 85 reading is
    printed under its label, as the record holds it."""
    cuts, step = block["cuts"], 1.4
    lim = 1.15 * max(abs(x) for c in ("85", "90") for k, _ in rows for x in cuts[c][k]["ci"])
    for i, (key, label) in enumerate(rows):
        y = (len(rows) - 1 - i) * step
        for cut, dy, face in (("85", 0.13, TEXT), ("90", -0.13, "white")):
            c = cuts[cut][key]
            p, (lo, hi) = c["below_minus_above"], c["ci"]
            ax.errorbar(p, y + dy, xerr=[[p - lo], [hi - p]], fmt="o", ms=8, color=TEXT, mfc=face,
                        mew=1.6, elinewidth=1.8, capsize=0, zorder=3)
        c = cuts["85"][key]
        ax.text(-lim, y + 0.45, label, ha="left", va="center", fontsize=11.5, fontweight=700,
                color=TEXT)
        ax.text(-lim, y - 0.45, f"at 85: {num(c['below_minus_above'])}  [{num(c['ci'][0])}, "
                f"{num(c['ci'][1])}]", ha="left", va="center", fontsize=10, color=LEGEND)
        ax.plot([0, 0], [y - 0.28, y + 0.28], color=TEXT, lw=1.2, zorder=2)
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-0.75, (len(rows) - 1) * step + 0.75)
    ax.set_yticks([])
    ax.spines["left"].set_visible(False)
    _style(ax, f"{title}, below the cut minus above it")
    ax.grid(axis="x", color=GRID, lw=0.8)
    ax.grid(axis="y", visible=False)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=str, default=str(_REPO / "figures" / "flank_cut.png"))
    args = ap.parse_args()

    block = json.loads(_RECORD.read_text())["figures"]["flank_cut"]
    fig = plt.figure(figsize=(18, 12))
    draw_header(fig, title="Where a flank free kick stops being a corner",
                subtitle=["Big-5 2025/26, flank free kicks from x = 60 in 5-yard bands of the taker's "
                          "x, long corners dashed as the reference."],
                hdr_band=(0.91, 1.0), text_x=0.05, jo_x=0.93)
    grid = fig.add_gridspec(2, 2, left=0.06, right=0.98, bottom=0.12, top=0.86, hspace=0.42,
                            wspace=0.14, width_ratios=[1.55, 1])
    panel_production(fig.add_subplot(grid[0, 0]), block)
    panel_jobs(fig.add_subplot(grid[1, 0]), block)
    cut_axes = [fig.add_subplot(grid[i, 1]) for i in range(len(_CUT_ROWS))]
    for ax, (title, rows) in zip(cut_axes, _CUT_ROWS, strict=True):
        panel_cuts(ax, block, title, rows)
    handles = [Line2D([], [], ls="", marker="o", ms=8, color=TEXT, mfc=f, mew=1.6, label=lab)
               for f, lab in ((TEXT, "cut at 85"), ("white", "cut at 90"))]
    cut_axes[0].legend(handles=handles, frameon=False, fontsize=11, loc="lower right")
    c85, c90 = block["cuts"]["85"], block["cuts"]["90"]
    fig.text(0.06, 0.03,
             "Flank free kicks under the paper's predicate (outside the box width, not along the ground)"
             "; long corners leave out the short ones. Jobs are the attackers per kick, over the kicks "
             f"carrying a 360.\nThe cuts compare the kicks below and above on the same match draws, "
             f"95%: {c85['n_below']} below and {c85['n_above']} above 85, {c90['n_below']} below and "
             f"{c90['n_above']} above 90.",
             ha="left", va="bottom", fontsize=11, color=LEGEND, linespacing=1.5)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, facecolor="white")
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()
