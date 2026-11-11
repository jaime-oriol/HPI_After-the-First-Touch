"""render_information — what each piece of information adds, and where the paper would have seen it.

Corners. Three questions, one panel each, because a log loss is read against its own target and never
across them. Top, the corner's outcome, the five competing outcomes of s03: the delivery over the
mix of outcomes, then the freeze-frame at the kick over the delivery. Middle, the positive control, who wins the
first contact: the same two steps on a target where the harness should see signal. Bottom, who wins the
second ball: the frame and the static race at the first contact, each over the delivery. Every bar is
the log loss saved over its row's baseline, positive is better, with the interval that decides that
reading: the refit in normal form for the freeze-frame step, the registered pivotal form for the first
contact, the record's interval elsewhere. The dashed line is the adjusted smallest effect detectable at
80% power where the record has one.

Every number printed is read from the record (docs/measured.json: `validation.corner`,
`positive_control`, `first_contact_frame`); a delta stored as model minus baseline prints with its sign
turned, so that saved log loss reads positive.

    python renders/render_information.py
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
from matplotlib.lines import Line2D                      # noqa: E402

_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO / "src"))

from viz import paper as P                               # noqa: E402

_RECORD = _REPO / "docs" / "measured.json"


def _saved(d: dict, form: str) -> tuple[float, float, float]:
    """A model-minus-baseline delta and its interval, turned into log loss saved."""
    lo, hi = d[form]
    return -d["delta"], -hi, -lo


def panels(rec: dict) -> list[dict]:
    """The three panels: title, rows (label, saved, low, high, read at the first contact) and MDE."""
    v, pc, t1 = rec["validation"]["corner"], rec["positive_control"]["decisions"], rec["first_contact_frame"]
    deliv, ref = v["ladder"]["delivery"], v["refit"]
    t1d = t1["decisions"]
    return [
        {"title": "How the corner ends, five outcomes",
         "rows": [("Delivery, over the mix of outcomes", deliv["step"], *deliv["step_ci"], False),
                  ("Frame at the kick, over the delivery", ref["step"], *ref["normal_ci"], False)],
         "mde": v["power"]["mde_80_adjusted"]},
        {"title": "Who wins the first contact, the positive control",
         "rows": [("Delivery, over the base rate", *_saved(pc["delivery-prior"], "ci"), False),
                  ("Frame at the kick, over the delivery",
                   *_saved(pc["full_360-delivery"], "ci"), False)],
         "mde": None},
        {"title": "Who wins the second ball",
         "rows": [("Frame at the first contact, over the delivery",
                   *_saved(t1d["geometry_1-delivery"], "pivotal"), True),
                  ("Static race at the first contact, over the delivery",
                   *_saved(t1d["static_race_1-delivery"], "pivotal"), True)],
         "mde": t1["mde_80_adjusted"]["geometry_1-delivery"]},
    ]


def draw(ps: list[dict], out: Path) -> None:
    P.setup()
    fig = P.figure(3.9)
    for ax, p in zip(fig.subplots(len(ps), 1), ps, strict=True):
        span = max(max(abs(r[2]), abs(r[3])) for r in p["rows"])
        for i, (_, saved, lo, hi, late) in enumerate(p["rows"]):
            c = P.AFTER if late else P.INK2
            ax.plot([lo, hi], [i, i], color=c, linewidth=1.4, solid_capstyle="butt", zorder=3)
            ax.scatter(saved, i, s=34, color=c, edgecolor=P.PAGE, linewidth=1.2, zorder=4)
            # the values in one column at the panel's right edge, clear of the marks and the MDE line
            ax.text(1.0, i, f"{saved:+.4f}".replace("-", "−"), va="center", ha="right", fontsize=7,
                    fontweight="bold", color=P.INK, transform=ax.get_yaxis_transform())
        if p["mde"]:
            ax.axvline(p["mde"], color=P.INK2, linewidth=0.8, linestyle=(0, (3, 2)), zorder=2)
        ax.set_yticks(range(len(p["rows"])), [r[0] for r in p["rows"]])
        ax.set_ylim(len(p["rows"]) - 0.45, -0.75)
        ax.set_xlim(min(0, *(r[2] for r in p["rows"])) - 0.12 * span, 1.45 * span)
        P.bold_categories(ax, "y")
        ax.set_title(p["title"], loc="left", fontsize=7.5, fontweight="bold", color=P.INK)
        P.hairline_grid(ax, "x")
        P.zero_line(ax, "x")
    fig.supxlabel("Log loss saved over the baseline of its row, in nats\nDashed line: the smallest effect the design "
                  "detects in eight draws out of ten", fontsize=7.5, fontweight="bold", color=P.INK)
    dot = dict(marker="o", linestyle="", markersize=6, markeredgecolor=P.PAGE)
    fig.legend([Line2D([], [], color=P.INK2, **dot), Line2D([], [], color=P.AFTER, **dot)],
               ["At the kick", "At the first contact"], loc="outside upper left", ncols=2)
    P.save(fig, out)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=_REPO / "figures" / "information.png")
    args = ap.parse_args()
    draw(panels(json.loads(_RECORD.read_text())["figures"]), args.out)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
