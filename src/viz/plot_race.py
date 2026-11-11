"""plot_race — Second-ball RACE snapshot: own reach-field surface + the contest.

The freeze-frame stages show who is committed; this shows who GETS THERE. At the
first-contact frame it paints our own REACH FIELD, P(attacking team arrives
first) at each point from the same time-to-arrive race that picks the recoverer
(attacking blue / contested white / defending red), overlaid with the P=0.5
decision frontier (who-gets-there-first line), the players with their velocity
arrows (momentum into the race), the loose ball, and a ring on the predicted
recoverer. One model end to end, no pitch control. Same house layout as
plot_corner (vertical half pitch + right-side legend), opta 0-100 coords.

On a VerticalPitch the axes' horizontal is opta-y and the vertical is opta-x, so
every [y, x] grid is transposed before it is drawn. The surface's extent runs to
the cell EDGES rather than the centres, or it stops short of the pitch lines.

Frame schema: x, y (opta, attack -> x=100), team_id, is_ball, is_goalkeeper,
vx_o, vy_o (velocity already in opta units, scaled to the arrow look-ahead).
`field` = {"x": opta-x centres, "y": opta-y centres, "grid": p_att[y, x] in [0,1],
           "alpha": per-cell opacity [y, x] (scalar or array)}.
"""

from __future__ import annotations

from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .common import (
    ATT, BG, DEF, TEXT, REACH_CMAP,
    draw_header, draw_side_legend, new_half_pitch,
)
from .plot_corner import FIGSIZE, _layout, _node_color, NODE_S, NODE_EW, BALL_S

ARROW_W = 2.2


def plot_race(
    frame: pd.DataFrame,
    field: dict,
    *,
    attacking_team: int,
    loose_xy: tuple[float, float],
    pred_xy: tuple[float, float] | None = None,
    title: str,
    subtitle: str | list[str] | None = None,
    att_team_name: str = "Attacking",
    def_team_name: str = "Defending",
    control_att: float = 0.5,
    save_path: Optional[str] = None,
    figsize: tuple = FIGSIZE,
) -> plt.Figure:
    """Render one second-ball race frame. Returns the figure (saved if save_path)."""
    L = _layout()
    fig = plt.figure(figsize=figsize, facecolor=BG)

    hb = L["header"]
    jo_cx = L["legend"][0] + L["legend"][2] / 2.0
    draw_header(fig, title=title, subtitle=subtitle, crest_path=None,
                hdr_band=(hb[1], hb[1] + hb[3]), crest_x=0.024, text_x=0.10, jo_x=jo_cx)

    ax = fig.add_axes(L["pitch"])
    pitch = new_half_pitch(pitch_type="opta")
    pitch.draw(ax=ax)

    ox, oy, grid = field["x"], field["y"], field["grid"]
    rgba = REACH_CMAP(np.clip(grid, 0.0, 1.0))
    rgba[..., 3] = np.clip(field["alpha"], 0.0, 1.0)
    dy = (oy[-1] - oy[0]) / (len(oy) - 1) if len(oy) > 1 else 2.0
    dx = (ox[-1] - ox[0]) / (len(ox) - 1) if len(ox) > 1 else 2.0
    img = ax.imshow(np.transpose(rgba, (1, 0, 2)),        # (nx, ny, 4): opta-x vertical
                    extent=[oy[0] - dy / 2, oy[-1] + dy / 2, ox[0] - dx / 2, ox[-1] + dx / 2],
                    origin="lower", interpolation="spline36", aspect="auto", zorder=1)
    img.set_clip_path(ax.patch)

    # The P=0.5 frontier, drawn only when it is in view.
    if np.nanmin(grid) < 0.5 < np.nanmax(grid):
        XX, YY = np.meshgrid(oy, ox)
        cs = ax.contour(XX, YY, np.transpose(grid), levels=[0.5], colors=[TEXT],
                        linewidths=1.4, alpha=0.5, linestyles="dashed", zorder=3)
        cs.set_clip_path(ax.patch)

    players = frame[frame["is_ball"] == 0]
    for _, p in players.iterrows():
        vx, vy = float(p["vx_o"]), float(p["vy_o"])
        if abs(vx) + abs(vy) > 0.5:
            pitch.arrows(p["x"], p["y"], p["x"] + vx, p["y"] + vy, ax=ax,
                         color=_node_color(p, attacking_team), width=ARROW_W,
                         headwidth=3.5, headlength=4, alpha=0.8, zorder=4)
    colors = [_node_color(p, attacking_team) for _, p in players.iterrows()]
    pitch.scatter(players["x"], players["y"], ax=ax, s=NODE_S, c=colors,
                  edgecolors=TEXT, linewidths=NODE_EW, alpha=0.95, zorder=5)

    if pred_xy is not None:
        pitch.scatter([pred_xy[0]], [pred_xy[1]], ax=ax, s=NODE_S * 2.4,
                      facecolors="none", edgecolors=TEXT, linewidths=2.6, zorder=7)
    pitch.scatter([loose_xy[0]], [loose_xy[1]], ax=ax, marker="football",
                  s=BALL_S, edgecolors="black", linewidths=1.0, zorder=10)

    draw_side_legend(fig, L["legend"], att_team_name=att_team_name, def_team_name=def_team_name,
                     att_color=ATT, def_color=DEF,
                     gradient={"cmap": REACH_CMAP, "value": control_att,
                               "top_label": "Att", "bottom_label": "Def"})

    if save_path:
        fig.savefig(save_path, dpi=200, facecolor=fig.get_facecolor(), bbox_inches=None)
    return fig
