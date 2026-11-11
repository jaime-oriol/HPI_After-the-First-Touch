"""render_signature_sb — Commitment-signature snapshot (StatsBomb 360).

The house-style 360 corner snapshot, read as the PLAN stage: attackers are coloured by their
COMMITMENT ROLE (attack the cross / stake the drop / hold back, from s02_signature), the role
counts are the headline stat panel, the box and the strip in front of it are shaded, the delivery is
drawn as an arrow, and the defender(s) in the switch's frontal, s05's protector, are ringed in
black. Under G2 that frontal also takes the central block of the box, which stays in the box's tint.

Usage:
    python renders/render_signature_sb.py --match 4004219 --index 0
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO / "src"))

import pandas as pd                                    # noqa: E402
import loader_sb as L                                  # noqa: E402
from s01_phases import CornerPhaseGrammar               # noqa: E402
from constants import BOX_X, BOX_Y, HOLD_X, PITCH_X_SB  # noqa: E402
from s02_signature import CommitmentSignature          # noqa: E402
from viz.common import ROLE_COLORS, DEF, GK             # noqa: E402
from viz.plot_corner import plot_corner                 # noqa: E402
from zones import frontal                               # noqa: E402

# Attackers are keyed by role in the stat panel, so only the non-role nodes need one.
_NODE_LEGEND = [(DEF, "Defender"), (GK, "Goalkeeper")]

# StatsBomb coords (attack -> x=120): the strip in front of the box and the box, both to the box
# width, tinted by the job each carries under C and F (src/zones.py).
_ZONES = [
    {"x0": HOLD_X, "x1": BOX_X, "y0": BOX_Y[0], "y1": BOX_Y[1],
     "color": ROLE_COLORS["stake_drop"], "alpha": 0.10},
    {"x0": BOX_X, "x1": PITCH_X_SB, "y0": BOX_Y[0], "y1": BOX_Y[1],
     "color": ROLE_COLORS["attack_cross"], "alpha": 0.10},
]


def _edge_protectors(frame: pd.DataFrame) -> list[tuple[float, float]]:
    """Outfield defenders in the frontal, FIFA's edge of the box: the treatment s05 estimates.

    team_id is in ATTACKING-CORNER terms here, so 0 is the defending side."""
    d = frame[(frame["team_id"] == 0) & (frame["is_goalkeeper"] == 0) & (frame["is_ball"] == 0)]
    d = d[frontal(d["x"].to_numpy(float), d["y"].to_numpy(float))]
    return list(zip(d["x"].astype(float), d["y"].astype(float), strict=True))


def main() -> None:
    """CLI: render one corner's commitment signature (360 freeze-frame) to a PNG."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--match", type=int, default=4004219)
    ap.add_argument("--index", type=int, default=0, help="which corner with 360 (match order)")
    ap.add_argument("--out", type=str, default=None)
    ap.add_argument("--crest", type=str, default=None)
    ap.add_argument("--title", type=str, default=None)
    ap.add_argument("--subtitle", type=str, default="Serie A 2025/26 · commitment signature")
    args = ap.parse_args()

    matches = L.list_matches(L.SERIE_A_SAMPLE)
    row = matches[matches["match_id"] == args.match]
    if row.empty:
        raise SystemExit(f"match {args.match} not in the Serie A sample: "
                         f"{matches['match_id'].tolist()}")
    home, away = row.iloc[0]["home_team"], row.iloc[0]["away_team"]

    events = L.load_events(args.match)
    corners = CornerPhaseGrammar(events).corners()
    fz = L.load_360(args.match)
    have = set(fz["event_uuid"])
    have_360 = [i for i, c in corners.iterrows() if c["id"] in have]
    if not have_360:
        raise SystemExit(f"no corners with 360 in match {args.match}")
    if not 0 <= args.index < len(have_360):
        raise SystemExit(f"index {args.index} out of range (0..{len(have_360) - 1} have 360)")
    corner = corners.loc[have_360[args.index]]
    att = corner["team.name"]
    def_name = away if att == home else home

    cs = CommitmentSignature.from_corner(corner, fz)
    frame = cs.frame
    sig = cs.signature()
    highlight = _edge_protectors(frame)

    stats = [(ROLE_COLORS["attack_cross"], sig["n_attack_cross"], "cross"),
             (ROLE_COLORS["stake_drop"], sig["n_stake_drop"], "drop"),
             (ROLE_COLORS["hold_back"], sig["n_hold_back"], "hold"),
             ("edge", len(highlight), "edge")]   # "edge" is a keyword the panel styles differently

    title = args.title or f"{att} Corner Commitment vs {def_name}"
    out = args.out or str(_REPO / "figures" / f"signature_sb_{args.match}_{args.index}.png")
    Path(out).parent.mkdir(parents=True, exist_ok=True)

    plot_corner(
        frame, attacking_team=1, pitch_type="statsbomb", color_by="role",
        node_legend=_NODE_LEGEND, highlight_xy=highlight,
        zones=_ZONES, delivery_target=cs.target,
        title=title, subtitle=args.subtitle,
        stats=stats, crest_path=args.crest, save_path=out,
    )
    print(f"saved {out}  ({att}: {sig['n_attack_cross']} cross / {sig['n_stake_drop']} drop / "
          f"{sig['n_hold_back']} hold; {len(highlight)} defender(s) in the frontal)")


if __name__ == "__main__":
    main()
