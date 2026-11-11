"""render_pipeline — the analysis at a glance: from the events and the freeze-frame of a corner to the results.

A diagram drawn in TikZ (renders/tex/pipeline.tex, in the house style of renders/tex/hudl_tikz.tex): the events split at
the first touch and counted, the freeze-frame turned into the plan and set against the delivery in a model, and the
intervals and nulls that both share. The only number it prints is the count of matches, read from the record's `scope` block
(docs/measured.json). This driver writes it into pipeline-defs.tex, compiles the diagram with xelatex and writes it
as a PNG at 300 dpi and as an SVG.

    python renders/render_pipeline.py
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


def defs(scope: dict) -> str:
    """The TeX definitions the diagram reads: the number of matches, with its thousands separator."""
    return rf"\def\matches{{{scope['matches']:,}}}" + "\n"


def build(tex_defs: str, out: Path, build_dir: Path) -> None:
    """Compile the diagram with its definitions on TEXINPUTS, then write it to `out` as a PNG at 300 dpi and,
    beside it, as the SVG the document embeds."""
    build_dir.mkdir(parents=True, exist_ok=True)
    (build_dir / "pipeline-defs.tex").write_text(tex_defs)
    env = {**os.environ, "TEXINPUTS": f"{build_dir}{os.pathsep}"}
    r = subprocess.run(["xelatex", "-interaction=nonstopmode", "-halt-on-error",
                        f"-output-directory={build_dir}", "pipeline.tex"],
                       cwd=_TEX, env=env, capture_output=True, text=True)
    if r.returncode:
        raise RuntimeError(f"xelatex failed:\n{r.stdout[-3000:]}")
    subprocess.run(["pdftoppm", "-png", "-r", "300", "-singlefile", str(build_dir / "pipeline.pdf"),
                    str(out.with_suffix(""))], check=True)
    subprocess.run(["pdftocairo", "-svg", str(build_dir / "pipeline.pdf"), str(out.with_suffix(".svg"))], check=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=_REPO / "figures" / "pipeline.png")
    ap.add_argument("--build", type=Path, default=_REPO / "data" / "derived" / "tex_build")
    args = ap.parse_args()
    scope = json.loads(_RECORD.read_text())["figures"]["scope"]
    build(defs(scope), args.out, args.build)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
