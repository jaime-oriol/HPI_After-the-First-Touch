"""render_corner — Build a corner snapshot from the granted Hudl export and render it.

Picks one corner from a match, finds the tracking frame at the kick, orients the
attack UP (toward the top goal), and renders it with viz.plot_corner.

Events come from loader_sb and tracking from loader; `--match` is the StatsBomb id
and only the 30 tracked Serie A matches resolve. The frame is anchored on
`period_elapsed_s`, the vendor GAME clock: the video clock runs ahead of it in
period 1 and would anchor before the ball is even struck (see loader.load_tracking).

Usage:
    python renders/render_corner.py --match 4004219 --index 0
    python renders/render_corner.py --match 4004219 --index 4 --out figures/c4.png
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO / "src"))

import loader                                   # noqa: E402
import loader_sb as L                           # noqa: E402
from constants import PITCH_OPTA                # noqa: E402
from s04_race import find_corners               # noqa: E402
from viz.common import corner_stats             # noqa: E402
from viz.plot_corner import plot_corner         # noqa: E402


def frame_at(tracking: pd.DataFrame, corner: pd.Series) -> pd.DataFrame:
    """Tracking rows of the frame closest in game-clock time to the corner kick."""
    period = int(corner["period"])
    target = float(corner["_t"])
    sub = tracking[tracking["period_id"] == period]
    fcount = sub.loc[(sub["period_elapsed_s"] - target).abs().idxmin(), "frame_count"]
    return tracking[(tracking["period_id"] == period) & (tracking["frame_count"] == fcount)].copy()


def build_frame(frame: pd.DataFrame, attacking_team: int, home_team: int) -> pd.DataFrame:
    """Tidy render frame in opta 0-100, attack toward x=100, with real shirt numbers."""
    flip = loader.attack_runs_rtl(frame, attacking_team, home_team)

    x, y = frame["x"].astype(float), frame["y"].astype(float)
    out = pd.DataFrame({
        "x": (PITCH_OPTA - x) if flip else x,          # mirror BOTH axes so the attack ends up toward x=100
        "y": (PITCH_OPTA - y) if flip else y,
        "team_id": frame["team_id"],
        "is_ball": frame["is_ball"].astype(int),
        "is_goalkeeper": frame["is_goalkeeper"].astype(int),
        "player_id": frame["player_id"],
    })
    # loader encodes the real jersey in player_id as "<team_id>_<jersey>"; ball rows are None
    out["jersey"] = frame["player_id"].str.split("_").str[1]
    return out


def main() -> None:
    """CLI entry: pick a corner, build its render frame, and save the snapshot PNG."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--match", type=int, default=4004219)
    ap.add_argument("--index", type=int, default=0, help="which corner (time order)")
    ap.add_argument("--out", type=str, default=None)
    ap.add_argument("--crest", type=str, default=None, help="crest left of the title")
    ap.add_argument("--title", type=str, default=None)
    ap.add_argument("--subtitle", type=str, default=None)
    args = ap.parse_args()

    events = L.load_events(args.match)
    corners = find_corners(events)
    if corners.empty:
        raise SystemExit(f"no corners found in match {args.match}")
    if not 0 <= args.index < len(corners):
        raise SystemExit(f"index {args.index} out of range (0..{len(corners) - 1})")
    corner = corners.iloc[args.index]
    att_team, att_name = int(corner["team.id"]), corner["team.name"]

    home_id = int(events["home_team.home_team_id"].iloc[0])
    home_name = events["home_team.home_team_name"].iloc[0]
    away_name = events["away_team.away_team_name"].iloc[0]
    def_name = away_name if att_team == home_id else home_name

    tracking = loader.load_tracking(args.match)
    frame = build_frame(frame_at(tracking, corner), att_team, home_id)

    out = args.out or str(_REPO / "figures" / f"corner_{args.match}_{args.index}.png")
    Path(out).parent.mkdir(parents=True, exist_ok=True)

    plot_corner(
        frame, attacking_team=att_team,
        title=args.title or f"{att_name} Corner",
        subtitle=args.subtitle or f"{home_name} vs {away_name} · corner {args.index}",
        att_team_name=att_name, def_team_name=def_name,
        stats=corner_stats(events, att_name), crest_path=args.crest, save_path=out,
    )
    print(f"saved {out}  ({len(corners)} corners in match {args.match})")


if __name__ == "__main__":
    main()
