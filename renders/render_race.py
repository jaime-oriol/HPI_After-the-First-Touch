"""render_race — Second-ball RACE snapshot from Hudl Broadcast X&Y (own reach field).

Picks a corner, finds its first contact (s04_race), and renders the OWN reach
field (P(attacker arrives first), from s04_race.reach_field) + players + velocity
arrows + loose ball + predicted recoverer with viz.plot_race. The Spearman PPCF
(pitch_control.py) is computed only as a console cross-check, not drawn.

Events come from loader_sb and tracking from loader; `--match` is the StatsBomb id
and only the 30 tracked Serie A matches resolve. Coordinates are oriented so the
attack points UP (x=100), matching the corner snapshots. The window is scanned ONCE
and the contact dict threaded into race()/summary().

Usage:
    python renders/render_race.py --match 4004219 --index 0
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO / "src"))

import loader                                          # noqa: E402
import loader_sb as L                                   # noqa: E402
import pitch_control as pc                              # noqa: E402
from constants import (FPS_TRACKING, LOOSE_HORIZON_S, PITCH_LENGTH_M,       # noqa: E402
                       PITCH_OPTA, PITCH_WIDTH_M)
from s04_race import SecondBallRace, find_corners                          # noqa: E402
from viz.plot_race import plot_race                     # noqa: E402

VEL_LOOKAHEAD_S = 1.0      # an arrow is where the player heads in this many seconds
_REACH_GRID_X = 140
FIELD_ALPHA = 0.62


def _opta(x: "float | pd.Series", flip: bool) -> "float | pd.Series":
    """Flip an opta x/y coordinate about the pitch centre when play runs Right-to-Left."""
    return (PITCH_OPTA - x) if flip else x


def _m_to_opta(x_m: float, y_m: float, flip: bool) -> tuple[float, float]:
    """Pitch-centred metres -> opta through loader's conversion, mirroring both axes on flip."""
    ox, oy = loader.meters_to_opta(x_m, y_m)
    return (PITCH_OPTA - ox, PITCH_OPTA - oy) if flip else (ox, oy)


def main() -> None:
    """Render one corner's second-ball reach-field snapshot to PNG (and print a PPCF cross-check)."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--match", type=int, default=4004219)
    ap.add_argument("--index", type=int, default=0, help="which corner (time order)")
    ap.add_argument("--out", type=str, default=None)
    args = ap.parse_args()

    ev = L.load_events(args.match)
    race = SecondBallRace(loader.load_tracking(args.match), ev)
    trk = race.trk
    corners = find_corners(ev)
    if not 0 <= args.index < len(corners):
        raise SystemExit(f"index out of range (0..{len(corners) - 1})")
    corner = corners.iloc[args.index]
    fc = race.first_contact(corner)
    if fc is None:
        raise SystemExit(f"corner {args.index}: no first contact (short / no aerial)")
    rt = race.race(corner, fc)
    summ = race.summary(corner, fc)

    att, att_name = int(corner["team.id"]), corner["team.name"]
    home_id = int(ev["home_team.home_team_id"].iloc[0])
    def_name = (ev["away_team.away_team_name"] if att == home_id else ev["home_team.home_team_name"]).iloc[0]
    period, f_n = fc["period"], fc["frame"]

    fr = trk[(trk["period_id"] == period) & (trk["frame_count"] == f_n)].copy()
    flip = loader.attack_runs_rtl(fr, att, home_id)

    sgn = -1.0 if flip else 1.0
    nodes = pd.DataFrame({
        "x": _opta(fr["x"].astype(float), flip), "y": _opta(fr["y"].astype(float), flip),
        "team_id": fr["team_id"], "is_ball": fr["is_ball"].astype(int),
        "is_goalkeeper": fr["is_goalkeeper"].astype(int),
        "vx_o": sgn * fr["vx"].fillna(0.0) * (PITCH_OPTA / PITCH_LENGTH_M) * VEL_LOOKAHEAD_S,
        "vy_o": sgn * fr["vy"].fillna(0.0) * (PITCH_OPTA / PITCH_WIDTH_M) * VEL_LOOKAHEAD_S,
    })

    p_grid, xg, yg = race.reach_field(corner, fc=fc, n_grid_x=_REACH_GRID_X)
    ox, oy = loader.meters_to_opta(xg, yg)
    if flip:
        # mirror the AXES and the grid together, or the surface detaches from the players
        ox, oy, p_grid = (PITCH_OPTA - ox)[::-1], (PITCH_OPTA - oy)[::-1], p_grid[::-1, ::-1]
    zone = ox >= 50.0
    field = {"x": ox[zone], "y": oy, "grid": p_grid[:, zone], "alpha": FIELD_ALPHA}

    # The ball row LOOSE_HORIZON_S after the contact; without one, s04's own loose point.
    lf = trk[(trk["period_id"] == period) & (trk["frame_count"] == f_n + int(LOOSE_HORIZON_S * FPS_TRACKING))]
    lb = lf[lf["is_ball"] == 1]
    loose = ((_opta(float(lb["x"].iloc[0]), flip), _opta(float(lb["y"].iloc[0]), flip))
             if len(lb) else _m_to_opta(fc["loose_xy"][0], fc["loose_xy"][1], flip))
    pr = fr[fr["player_id"] == rt.iloc[0]["player_id"]]
    pred = (_opta(float(pr["x"].iloc[0]), flip), _opta(float(pr["y"].iloc[0]), flip)) if len(pr) else None

    p_race = summ["recovery_prob_att"]
    out = args.out or str(_REPO / "figures" / f"race_{args.match}_{args.index}.png")
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    plot_race(nodes, field, attacking_team=att, loose_xy=loose, pred_xy=pred,
              title="Second-Ball Race", subtitle="Hudl Broadcast X&Y · own reach field at first contact",
              att_team_name=att_name, def_team_name=def_name,
              control_att=p_race, save_path=out)
    p_viz = float(pc.ppcf_at_targets(fr, np.array([fc["loose_xy"]]), att)[0])
    print(f"saved {out}  (corner {args.index}: recovery P(att)={p_race:.2f} [reach field], "
          f"win tti={summ['pred_tti_s']:.2f}s, pred_att={summ['pred_recoverer_att']}; "
          f"pitch-control cross-check={p_viz:.2f})")


if __name__ == "__main__":
    main()
