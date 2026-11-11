"""plot_corner — Set-piece snapshot, landscape with a right-side legend column.

Pure rendering. Left = vertical half pitch (mplsoccer, attack UP, opta 0-100);
right = legend column (logo on top via the header, stats on the top half, team
nodes + ball on the bottom half). The pitch FILLS its box (its real aspect, so
the legend sits right beside it with no white gap) and is never stretched.

Layout is in inches: FIG_H_IN sets the scale, the header and the paddings stack
top to bottom, and the figure width follows from the pitch's own aspect, so
changing one margin never distorts the pitch. BALL_S is the ball on the pitch;
the legend ball is sized separately in common.draw_side_legend.

Frame schema (one row per player + one ball row):
    x, y            pitch coords (opta 0-100 or statsbomb 120x80), attack -> max x
    team_id, is_ball, is_goalkeeper   (+ optional jersey, role for the signature)
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .common import (
    ATT, BG, DEF, GK, ROLE_COLORS, TEXT, PE_S,
    draw_header, draw_side_legend, new_half_pitch,
)


# ── Canvas + columns (inches) ────────────────────────────────────────────
FIG_H_IN    = 8.00
HEADER_H_IN = 1.00
PAD_TOP_IN  = 0.06
PAD_HDR_IN  = 0.05
PAD_BOT_IN  = 0.12
LEFT_M_IN   = 0.30
GAP_IN      = 0.25
RIGHT_M_IN  = 0.30
LEGEND_W_IN = 2.60
PITCH_AR_WH = 1.2807       # the real half-pitch aspect (width/height); anything else distorts it

_BODY_H_IN  = FIG_H_IN - PAD_TOP_IN - HEADER_H_IN - PAD_HDR_IN - PAD_BOT_IN
_PITCH_W_IN = _BODY_H_IN * PITCH_AR_WH
FIG_W_IN    = LEFT_M_IN + _PITCH_W_IN + GAP_IN + LEGEND_W_IN + RIGHT_M_IN
FIGSIZE     = (FIG_W_IN, FIG_H_IN)

NODE_S    = 290
NODE_EW   = 1.3
BALL_S    = 150
JERSEY_SZ = 7.5

DEFAULT_STATS = [("corner", "—", "corners"), ("shot", "—", "shots"), ("goal", "—", "goals")]


def _layout() -> dict:
    """Header band (top) + pitch (left col) + legend (right col), in fig-fraction."""
    fw, fh = FIG_W_IN, FIG_H_IN
    h_hdr = HEADER_H_IN / fh
    y_hdr = 1.0 - PAD_TOP_IN / fh - h_hdr
    y_bot = PAD_BOT_IN / fh
    body_h = _BODY_H_IN / fh
    return {
        "header": (0.0, y_hdr, 1.0, h_hdr),
        "pitch":  (LEFT_M_IN / fw, y_bot, _PITCH_W_IN / fw, body_h),
        "legend": ((LEFT_M_IN + _PITCH_W_IN + GAP_IN) / fw, y_bot, LEGEND_W_IN / fw, body_h),
    }


def _node_color(row: pd.Series, attacking_team: int) -> str:
    """Team colour for a node (GK its own colour, else att-blue / def-red)."""
    if int(row["is_goalkeeper"]) == 1:
        return GK
    return ATT if int(row["team_id"]) == attacking_team else DEF


def _role_color(row: pd.Series, attacking_team: int) -> str:
    """Attacker coloured by commitment role (att-blue without one); defenders/GK keep team colours."""
    if int(row["is_goalkeeper"]) == 1:
        return GK
    if int(row["team_id"]) != attacking_team:
        return DEF
    return ROLE_COLORS.get(row.get("role"), ATT)


def plot_corner(
    frame: pd.DataFrame,
    *,
    attacking_team: int,
    title: str,
    subtitle: str | list[str] | None = None,
    att_team_name: str = "Attacking",
    def_team_name: str = "Defending",
    stats: list[tuple] | None = None,
    pitch_type: str = "opta",
    color_by: str = "team",
    node_legend: list[tuple] | None = None,
    highlight_xy: list | None = None,
    zones: list[dict] | None = None,
    delivery_target: tuple | None = None,
    crest_path: Optional[str] = None,
    save_path: Optional[str] = None,
    figsize: tuple = FIGSIZE,
) -> plt.Figure:
    """Render one set-piece frame. Returns the figure (saved if save_path).

    `pitch_type` is "opta" for tracking frames and "statsbomb" for 360 freeze-frames.
    `color_by="role"` colours attackers by commitment role (the signature render);
    `node_legend` replaces the bottom legend rows with (color, label) pairs.
    `highlight_xy` rings nodes in black (the edge protectors), `zones` shades bands given
    as {x0, x1, y0, y1, color, alpha}, and `delivery_target` draws an arrow from the ball."""
    L = _layout()
    fig = plt.figure(figsize=figsize, facecolor=BG)

    hb = L["header"]
    crest = crest_path if (crest_path and Path(crest_path).exists()) else None
    jo_cx = L["legend"][0] + L["legend"][2] / 2.0
    draw_header(fig, title=title, subtitle=subtitle, crest_path=crest,
                hdr_band=(hb[1], hb[1] + hb[3]),
                crest_x=0.024, text_x=0.10, jo_x=jo_cx, crest_scale=0.78)

    ax = fig.add_axes(L["pitch"])
    pitch = new_half_pitch(pitch_type=pitch_type)
    pitch.draw(ax=ax)

    if zones:
        for z in zones:
            verts = np.array([[z["x0"], z["y0"]], [z["x1"], z["y0"]],
                              [z["x1"], z["y1"]], [z["x0"], z["y1"]]])
            pitch.polygon([verts], ax=ax, fc=z["color"], ec="none",
                          alpha=z.get("alpha", 0.10), zorder=1)
    if delivery_target is not None:
        ball = frame[frame["is_ball"] == 1]
        if len(ball):
            pitch.arrows(float(ball["x"].iloc[0]), float(ball["y"].iloc[0]),
                         delivery_target[0], delivery_target[1], ax=ax,
                         color=TEXT, width=1.6, headwidth=4.5, headlength=4.5,
                         alpha=0.55, zorder=4)

    players = frame[frame["is_ball"] == 0]
    paint = _role_color if (color_by == "role" and "role" in players.columns) else _node_color
    colors = [paint(p, attacking_team) for _, p in players.iterrows()]
    pitch.scatter(players["x"], players["y"], ax=ax,
                  s=NODE_S, c=colors, edgecolors=TEXT, linewidths=NODE_EW,
                  alpha=0.95, zorder=5)
    if highlight_xy:
        pitch.scatter([float(h[0]) for h in highlight_xy], [float(h[1]) for h in highlight_xy],
                      ax=ax, s=NODE_S * 2.4, facecolors="none", edgecolors=TEXT,
                      linewidths=2.6, zorder=7)
    if "jersey" in players.columns:
        for _, p in players.iterrows():
            if pd.notna(p["jersey"]):
                pitch.annotate(str(int(p["jersey"])), (float(p["x"]), float(p["y"])),
                               ax=ax, ha="center", va="center", color="white",
                               fontsize=JERSEY_SZ, fontweight="bold", zorder=6,
                               path_effects=PE_S)

    ball = frame[frame["is_ball"] == 1]
    if len(ball):
        pitch.scatter([float(ball["x"].iloc[0])], [float(ball["y"].iloc[0])], ax=ax,
                      marker="football", s=BALL_S, edgecolors="black",
                      linewidths=1.0, zorder=10)

    draw_side_legend(fig, L["legend"],
                     att_team_name=att_team_name, def_team_name=def_team_name,
                     att_color=ATT, def_color=DEF, node_legend=node_legend,
                     stats=stats if stats is not None else DEFAULT_STATS)

    if save_path:
        fig.savefig(save_path, dpi=200, facecolor=fig.get_facecolor(), bbox_inches=None)
    return fig
