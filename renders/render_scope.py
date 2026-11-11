"""render_scope — the scoping figure, from pre-model EDA only: where the volume to model lies.

Four panels, each read from an artefact the descriptive layer already wrote, so no panel depends on
a model result:

  A  deliveries per team-season, against the publication floor a team cell needs
  B  after-first-touch goals over one season, with the share of each population's shots they are
  C  goals over the provider's xG for the direct shot and the after-clearance shot, 95% CI by match
  D  deliveries inside the tracked sample, how many were raced, and, grant-wide, the share whose
     delivery carries a 360

Every measured number printed, the header's match counts included, is read from the record
(docs/measured.json), so a figure that moves there moves here; the floor is constants.MIN_CELL, and
the points, bars and intervals are drawn from the derived tables the record summarises. The
counter-evidence is drawn rather than dropped: long throws put the largest share of their shots
after the first touch, and free kicks carry a delivery 360 more often than corners. What the figure
argues is that the second ball is common to set pieces while the volume to model it thins from the
corner to the flank free kick and again to the long throw, the one population the paper leaves
unmodelled.

    python renders/render_scope.py
    python renders/render_scope.py --out /tmp/scope.png
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
import pandas as pd
from matplotlib.lines import Line2D

_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO / "src"))
_DER = _REPO / "data" / "derived"
_RECORD = _REPO / "docs" / "measured.json"

from constants import MIN_CELL                           # noqa: E402
from s01_phases import AFTER_FIRST_TOUCH                 # noqa: E402
from viz.common import ATT, GRID, LEGEND, TEXT, draw_header   # noqa: E402

# Recessive grey for the population the figure argues against. ROLE_COLORS' slate is 2.56:1 on
# white, under the 3:1 a graphical mark needs, so the next step darker in the same family is used.
_OTHER = "#64748b"
_FLANK = "free_kick_flank"
_GRAIN = ["league", "attacking_team", "match_id"]


def _segments(name: str, key: str, **where: object) -> pd.DataFrame:
    """One population's segment table (shots, goals, obs/exp and its interval), indexed by segment."""
    t = pd.read_parquet(_DER / name)
    for col, val in where.items():
        t = t[t[col] == val]
    return t.set_index(key)[["shots", "goals", "obs_exp", "lo", "hi"]]


def populations(record: dict) -> list[dict]:
    """Corner, flank free kick and long throw: the grain and segment table each panel draws, and
    the record's entries each panel prints.

    The two modelled populations read the same files, the flank's tagged by its kind; the long
    throw is the one the paper leaves unmodelled, so it has no frontier block and no race."""
    eda = pd.read_parquet(_DER / "eda_deliveries.parquet", columns=_GRAIN + ["set_piece_type"])
    throws = pd.read_parquet(_DER / "eda_throw_ins.parquet")
    flank = record[_FLANK]
    modelled = [
        {"label": label, "color": ATT, "deliv": eda[eda["set_piece_type"] == kind],
         "seg": _segments("frontier.parquet", "key", block="segment", set_piece_type=kind),
         "scope": record["scope"][kind], "by_segment": corpus["by_segment"], "raced": raced}
        for kind, label, corpus, raced in (
            ("corner", "Corner", record["corpus"], record["race"]["raced_corners"]),
            (_FLANK, "Flank free kick, x ≥ 85", flank["corpus"], flank["race"]["raced_corners"]))]
    return modelled + [
        {"label": "Long throw", "color": _OTHER, "deliv": throws,
         "seg": _segments("throw_in_probe.parquet", "segment", block="segment"),
         "scope": record["scope"]["long_throw"], "by_segment": record["throw_ins"]["by_segment"],
         "raced": None}]


def _style(ax: plt.Axes, title: str, ylabel: str, labels: list) -> None:
    ax.set_title(title, loc="left", fontsize=16, fontweight=700, color=TEXT, pad=14)
    ax.set_ylabel(ylabel, fontsize=12, color=LEGEND)
    ax.set_xticks(range(len(labels)), labels, fontsize=12)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=LEGEND, labelsize=11)
    ax.grid(axis="y", color=GRID, lw=0.8)
    ax.set_axisbelow(True)


def panel_volume(ax: plt.Axes, pops: list) -> None:
    """A: each team-season as a jittered point per population, its median and minimum beside it,
    against the floor a published cell needs: it shows how far the volume thins out."""
    rng = np.random.default_rng(0)
    for i, p in enumerate(pops):
        per_team = p["deliv"].groupby(["league", "attacking_team"]).size().to_numpy()
        ax.scatter(i + rng.uniform(-0.2, 0.2, len(per_team)), per_team, s=26, color=p["color"],
                   alpha=0.8, edgecolor="white", linewidth=0.7, zorder=3)
        med, low = p["scope"]["per_team_median"], p["scope"]["per_team_min"]
        ax.plot([i - 0.3, i + 0.3], [med, med], color=TEXT, lw=2.2, zorder=4)
        # a median sitting on the floor line takes its label above its own points, or the label
        # crosses the floor line and runs into the next population's points
        if med < 1.5 * MIN_CELL:
            ax.text(i, per_team.max() + 6, f"median {med:g}\nmin {low}", ha="center", va="bottom",
                    fontsize=11, color=TEXT)
        else:
            ax.text(i + 0.33, med, f"median {med:g}\nmin {low}", va="center", fontsize=11, color=TEXT)
    ax.axhline(MIN_CELL, color=TEXT, lw=1.2, zorder=2)
    ax.text(-0.55, MIN_CELL + 4, f"floor for a published cell, n = {MIN_CELL}",
            ha="left", va="bottom", fontsize=11, color=TEXT)
    ax.set_xlim(-0.6, len(pops) - 0.1)
    _style(ax, "A  Deliveries per team-season", "deliveries", [p["label"] for p in pops])


def after_first_touch_goals(by_segment: dict) -> int:
    """Every goal after the first touch: a segment's shot goals and its own goals, which the record
    keeps apart because an own goal is a goal and no shot."""
    return sum(v["goals"] + v["own_goals"] for k, v in by_segment.items() if k in AFTER_FIRST_TOUCH)


def panel_second_ball(ax: plt.Axes, pops: list) -> None:
    """B: the goals after the first touch in each population and the share of its shots they are:
    the second ball is common to set pieces, the long throw included, which is why it is drawn."""
    top = 0
    for i, p in enumerate(pops):
        seg = p["by_segment"]
        goals = after_first_touch_goals(seg)
        share = (100 * sum(v["shots"] for k, v in seg.items() if k in AFTER_FIRST_TOUCH)
                 / sum(v["shots"] for v in seg.values()))
        ax.bar(i, goals, width=0.5, color=p["color"], zorder=3)
        ax.text(i, goals, f"{goals} goals\n{share:.0f}% of its shots\nafter the first touch",
                ha="center", va="bottom", fontsize=11, color=TEXT)
        top = max(top, goals)
    ax.set_ylim(0, top * 1.45)
    _style(ax, "B  Goals after the first touch, one season", "goals",
           [p["label"] for p in pops])


def panel_xg(ax: plt.Axes, pops: list) -> None:
    """C: goals over the provider's xG for the direct shot and the after-clearance shot, each with
    its 95% interval by match. The provider's xG is the only model in the figure."""
    shape ={"direct": ("o", -0.13, "direct shot"), "after_clearance": ("s", 0.13, "after clearance")}
    for i, p in enumerate(pops):
        for seg, (marker, dx, _) in shape.items():
            r = p["seg"].loc[seg]
            ax.errorbar(i + dx, r["obs_exp"], yerr=[[r["obs_exp"] - r["lo"]], [r["hi"] - r["obs_exp"]]],
                        fmt=marker, ms=10, color=p["color"], ecolor=p["color"], elinewidth=2.2,
                        capsize=0, mec="white", mew=1.5, zorder=3)
            ax.text(i + dx, r["lo"] - 0.04, f"{p['by_segment'][seg]['goals']} g", ha="center",
                    va="top", fontsize=10, color=LEGEND)
    ax.axhline(1.0, color=TEXT, lw=1.2, zorder=2)
    ax.text(-0.55, 1.01, "goals = xG", ha="left", va="bottom", fontsize=11, color=TEXT)
    handles = [Line2D([], [], ls="", marker=m, ms=10, color=LEGEND, label=lab)
               for m, _, lab in shape.values()]
    ax.legend(handles=handles, frameon=False, fontsize=11, loc="upper left")
    ax.set_xlim(-0.6, len(pops) - 0.4)
    ax.set_ylim(-0.15, None)
    _style(ax, "C  Goals over the provider's xG, 95% CI by match", "goals / xG",
           [p["label"] for p in pops])


def panel_substrate(ax: plt.Axes, pops: list, tracked_matches: int) -> None:
    """D: the deliveries inside the tracked sample and how many were raced, which is what the race
    can be built on; and, for the whole grant, the share whose delivery carries a 360, which is
    what the plan at the kick can be built on. The second is not a property of the tracked sample,
    and the label says so."""
    top = 0
    for i, p in enumerate(pops):
        n, cov = p["scope"]["tracked_deliveries"], p["scope"]["with_360_pct"]
        raced = "not raced" if p["raced"] is None else f"{p['raced']} raced"
        ax.bar(i, n, width=0.5, color=p["color"], zorder=3)
        ax.text(i, n + 5, f"{n} tracked deliveries\n{raced}\ngrant-wide: 360 at the\ndelivery {cov:.1f}%",
                ha="center", va="bottom", fontsize=11, color=TEXT)
        top = max(top, n)
    ax.set_ylim(0, top * 1.6)
    _style(ax, f"D  The tracked sample, {tracked_matches} matches", "deliveries",
           [p["label"] for p in pops])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=str, default=str(_REPO / "figures" / "scope_set_pieces.png"))
    args = ap.parse_args()

    record = json.loads(_RECORD.read_text())["figures"]
    pops, counts = populations(record), record["scope"]
    fig = plt.figure(figsize=(16, 12))
    draw_header(fig, title="The second ball is common to set pieces; the volume to model it thins out",
                subtitle=[f"Big-5 2025/26, {counts['matches']:,} matches. Descriptive counts and the "
                          f"provider's xG only: no model enters any panel."],
                hdr_band=(0.9, 1.0), text_x=0.05, jo_x=0.92)
    grid = fig.add_gridspec(2, 2, left=0.07, right=0.97, bottom=0.07, top=0.85,
                            hspace=0.38, wspace=0.22)
    a, b, c, d = (fig.add_subplot(grid[pos]) for pos in ((0, 0), (0, 1), (1, 0), (1, 1)))
    panel_volume(a, pops)
    panel_second_ball(b, pops)
    panel_xg(c, pops)
    panel_substrate(d, pops, counts["tracked_matches"])
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, facecolor="white")
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()
