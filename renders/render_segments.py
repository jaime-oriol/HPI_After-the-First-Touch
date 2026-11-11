"""render_segments — what "after the first touch" is: the rule that gives a corner shot its segment.

A diagram drawn in TikZ (renders/tex/segments.tex, in the house style of renders/tex/hudl_tikz.tex): the
four questions that the rule applies to every attacking shot, in order (definitions.md §4), each answer
sending the shot to one of the five segments. The three after the first touch carry their share of the
corner shots and goals. This driver writes each segment's name, gloss, shots and goals into
segments-defs.tex, compiles the diagram with xelatex and writes it as a PNG at 300 dpi and as an SVG.

Every number printed is the record's `frontier.corner` block (docs/measured.json), as stored: the
segment counts and the two headline shares. A segment's goals count the own goals it produced, which the
record keeps apart, because the headline share of goals counts them: the five segments add up to it.

    python renders/render_segments.py
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]
_RECORD = _REPO / "docs" / "measured.json"
_TEX = _REPO / "renders" / "tex"
SEGMENT_CARDS = (("late", "Late", "Over 8 s, or no contact"),
                 ("direct", "Direct shot", "The shot is the first touch"),
                 ("after_clearance", "After a clearance", "A defender touched it first"),
                 ("flick_on", "Flick-on", "An attacking header"),
                 ("prolongation", "Prolongation", "Another attacking touch"))


def defs(frontier: dict) -> str:
    """The TeX definitions the diagram reads: per segment its name, gloss, shots and goals (own goals
    included), and the two shares after the first touch, in percent."""
    seg, head = frontier["segment"], frontier["headline"]
    lines = []
    for key, name, says in SEGMENT_CARDS:
        k = key.replace("_", "")
        goals = seg[key]["goals"] + seg[key]["own_goals"]
        lines += [rf"\def\{k}name{{{name}}}", rf"\def\{k}says{{{says}}}",
                  rf"\def\{k}shots{{{seg[key]['shots']:,}}}", rf"\def\{k}goals{{{goals:,}}}"]
    lines += [rf"\def\aftshots{{{100 * head['share_shots_after_first_touch']['point']:.2f}}}",
              rf"\def\aftgoals{{{100 * head['share_goals_after_first_touch']['point']:.2f}}}"]
    return "\n".join(lines) + "\n"


def build(tex_defs: str, out: Path, build_dir: Path) -> None:
    """Compile the diagram with its definitions on TEXINPUTS, then write it to `out` as a PNG at 300 dpi and,
    beside it, as the SVG the document embeds."""
    build_dir.mkdir(parents=True, exist_ok=True)
    (build_dir / "segments-defs.tex").write_text(tex_defs)
    env = {**os.environ, "TEXINPUTS": f"{build_dir}{os.pathsep}"}
    r = subprocess.run(["xelatex", "-interaction=nonstopmode", "-halt-on-error",
                        f"-output-directory={build_dir}", "segments.tex"],
                       cwd=_TEX, env=env, capture_output=True, text=True)
    if r.returncode:
        raise RuntimeError(f"xelatex failed:\n{r.stdout[-3000:]}")
    subprocess.run(["pdftoppm", "-png", "-r", "300", "-singlefile", str(build_dir / "segments.pdf"),
                    str(out.with_suffix(""))], check=True)
    subprocess.run(["pdftocairo", "-svg", str(build_dir / "segments.pdf"), str(out.with_suffix(".svg"))], check=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=_REPO / "figures" / "segments.png")
    ap.add_argument("--build", type=Path, default=_REPO / "data" / "derived" / "tex_build")
    args = ap.parse_args()
    frontier = json.loads(_RECORD.read_text())["figures"]["frontier"]["corner"]
    build(defs(frontier), args.out, args.build)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
