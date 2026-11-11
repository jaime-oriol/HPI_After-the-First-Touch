"""render_top_teams — the sides at the top take more corners, not better ones after the first touch.

Corners, the top two of each league over the season (group T of the pre-registered sheet,
data/derived/prereg_2026-10-01/top_teams.md) against the rest. Left, corners per match; right, the
expected goals a corner produces, split into after the first touch and the rest of it,
direct or late, each group pooled over its deliveries. Inside each panel, the group's difference with its
95% interval: when neither of the sheet's rules holds, the after-first-touch gap reads "not resolved",
never "the same". Flank free kicks are thinner and stay in the text.

Every number printed is read from the record's `top_teams` block (docs/measured.json): the rates as
stored, the corners to two decimals.

    python renders/render_top_teams.py
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO / "src"))

from viz import paper as P                               # noqa: E402

_RECORD = _REPO / "docs" / "measured.json"
GROUPS = (("T2", "Top two of each league"), ("rest", "The rest"))


def numbers(block: dict) -> dict:
    """What the panels print: each group's rates, the volume and after-first-touch differences with
    their intervals, and whether the sheet's rules resolved the second."""
    vol, aft, d = block["secondary"]["T2"]["corners_per_match"], block["primary"]["T2"], block["decisions"]
    return {"rates": {g: block["rates"][g] for g, _ in GROUPS},
            "volume": (vol["point"], *vol["ci95"]), "aft": (aft["point"], *aft["ci95"]),
            "resolved": d["T1"] or d["T2"]}


def _signed(v: float, fmt: str) -> str:
    return f"{v:+{fmt}}".replace("-", "−")


def draw(n: dict, out: Path) -> None:
    P.setup()
    fig = P.figure(2.3)
    left, right = fig.subplots(1, 2, width_ratios=(1, 1.45))
    ys, bar = (1, 0), dict(height=0.55, edgecolor=P.PAGE, linewidth=0.8)
    for y, (g, _) in zip(ys, GROUPS, strict=True):
        r = n["rates"][g]
        left.barh(y, r["corners_per_match"], color=P.CONTEXT, **bar)
        left.text(r["corners_per_match"] + 0.12, y, f"{r['corners_per_match']:.2f}", va="center",
                  fontsize=7.5, fontweight="bold", color=P.INK)
        rest = r["direct_xg_per_corner"] + r["late_xg_per_corner"]
        right.barh(y, r["aft_xg_per_corner"], color=P.AFTER, label="After the first touch" if y else None,
                   **bar)
        right.barh(y, rest, left=r["aft_xg_per_corner"], color=P.CONTEXT,
                   label="Direct or late" if y else None, **bar)
        right.text(r["aft_xg_per_corner"] / 2, y, f"{r['aft_xg_per_corner']:g}", ha="center",
                   va="center", fontsize=7, fontweight="bold", color=P.INK)
    v, a = n["volume"], n["aft"]
    notes = ((left, f"Difference {_signed(v[0], '.2f')} [{v[1]:.2f}, {v[2]:.2f}]"),
             (right, f"After the first touch {_signed(a[0], 'g')} [{_signed(a[1], 'g')}, "
                     f"{_signed(a[2], 'g')}]" + ("" if n["resolved"] else ", not resolved")))
    for ax, note in notes:
        ax.text(0.015, 1.62, note, va="center", fontsize=7, color=P.INK2,
                transform=ax.get_yaxis_transform())
        ax.set_ylim(-0.6, 1.95)
        P.hairline_grid(ax, "x")
        P.zero_line(ax, "x")
    left.set_yticks(ys, [label for _, label in GROUPS])
    P.bold_categories(left, "y")
    right.set_yticks([])
    left.set_xlim(0, 7.5)
    left.set_xlabel("Corners per match")
    right.set_xlabel("Expected goals per corner")
    P.legend(fig, right)
    P.save(fig, out)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=_REPO / "figures" / "top_teams.png")
    args = ap.parse_args()
    block = json.loads(_RECORD.read_text())["figures"]["top_teams"]
    draw(numbers(block), args.out)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
