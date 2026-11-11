"""paper — the house style of the paper's figures: Hudl's colours, Opta Analyst's layout.

A figure in the paper sits at the template's text width, 6.5 in, under a caption the document
carries, so nothing here draws a title, a header or a logo. The colours are the ones Hudl's own
templates in docs/hudl/ use: the orange and the blue of the presentation template for the two
accents, and the slate inks and fills of the Word paper template for text, context, grid and the
card the chart sits on. The layout is the one Opta Analyst's charts share: a light card, no axes but
the zero line, a hairline grid, categories in bold ink, the axis title in bold, rankings sorted by
value, values printed on the marks with their unit, and the legend as one line above the chart.

The dataviz skill's validator passes the two accents on the card; the orange sits at 2.86:1 against
it, under the 3:1 a mark needs, so an orange mark always carries a direct label in ink, and no text
is set in orange. The slates read as grey by design and never carry identity. One meaning per colour
across the paper: orange is what happens after the first touch, the paper's subject, and the slates
are everything else. Barlow, the template's own face, is registered from figures/fonts, copied out of
the template (SIL OFL). Text is set at 7 to 8 pt for the page. A figure is saved twice, as a PNG, the
fallback every viewer draws, and as an SVG beside it, the vector the paper's document embeds so that a
zoom stays sharp.
"""
from __future__ import annotations

import re
from pathlib import Path

import matplotlib.font_manager as fm
import matplotlib.pyplot as plt

_FONT_DIR = Path(__file__).resolve().parents[2] / "figures" / "fonts"

WIDTH = 6.5            # in, the Word template's text width (21.59 cm page, 2.54 cm margins)

AFTER = "#FF6300"      # Hudl orange: what happens after the first touch
ACCENT = "#0A93F5"     # Hudl blue, the second accent
INK = "#232A31"        # primary text, the template's title colour
INK2 = "#4E5D6C"       # secondary text and dark neutral marks
CONTEXT = "#B1BCC4"    # marks the reader is not asked to read one by one
GRID = "#EDF0F2"
PAGE = "#F9FAFB"       # the card: the Word template's lightest fill, where Opta uses F7F7F7

_RC = {
    "font.family": "sans-serif", "font.sans-serif": ["Barlow", "DejaVu Sans"],
    "font.size": 8, "axes.labelsize": 8, "xtick.labelsize": 7, "ytick.labelsize": 7,
    "legend.fontsize": 7.5, "legend.frameon": False, "legend.handlelength": 1.0,
    "legend.handleheight": 0.8, "legend.handletextpad": 0.5, "legend.columnspacing": 1.6,
    "text.color": INK, "axes.labelcolor": INK, "axes.labelweight": "bold",
    "xtick.color": INK2, "ytick.color": INK2,
    "axes.spines.left": False, "axes.spines.bottom": False,
    "axes.spines.top": False, "axes.spines.right": False,
    "xtick.major.size": 0, "ytick.major.size": 0, "xtick.major.pad": 3, "ytick.major.pad": 4,
    "figure.facecolor": PAGE, "axes.facecolor": PAGE, "savefig.facecolor": PAGE,
    "figure.dpi": 100, "savefig.dpi": 300,
    "svg.fonttype": "path", "svg.hashsalt": "hudl",      # text as outlines, so no font is needed; stable ids
}


def setup() -> None:
    """Register Barlow and set the page style; every render calls it before drawing."""
    for f in sorted(_FONT_DIR.glob("Barlow-*.ttf")):
        fm.fontManager.addfont(str(f))
    plt.rcParams.update(_RC)


def figure(height: float, **kw: object) -> plt.Figure:
    """A figure at the text width, laid out so axis labels never fall off the page; a diagram drawn in
    its own coordinates passes layout=None."""
    kw.setdefault("layout", "constrained")
    return plt.figure(figsize=(WIDTH, height), **kw)


def hairline_grid(ax: plt.Axes, axis: str) -> None:
    """The one gridline set a chart keeps, solid and behind the marks."""
    ax.grid(axis=axis, color=GRID, linewidth=0.6, linestyle="-")
    ax.set_axisbelow(True)


def zero_line(ax: plt.Axes, axis: str) -> None:
    """The only axis line drawn: the zero the bars grow from, along `axis`."""
    (ax.axvline if axis == "x" else ax.axhline)(0, color=INK, linewidth=0.8, zorder=3)


def bold_categories(ax: plt.Axes, axis: str) -> None:
    """Category tick labels (teams, phases) in bold ink, the numbers on the other axis staying grey."""
    for t in ax.get_yticklabels() if axis == "y" else ax.get_xticklabels():
        t.set(color=INK, fontweight="bold", fontsize=7.5)


def legend(fig: plt.Figure, ax: plt.Axes) -> None:
    """The legend as one line above the chart, swatch and ink text, never a box."""
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside upper left", ncols=len(labels))


def _clean_svg(path: Path) -> None:
    """Matplotlib's SVG carries a DOCTYPE, a metadata block with the date and a style sheet, none of which Word's
    renderer reads: they go, and the sheet's one rule, round joins and flat caps on every stroke, becomes the
    attributes of the root, which every renderer inherits."""
    text = path.read_text()
    text = re.sub(r"<!DOCTYPE.*?>\s*", "", text, flags=re.S)
    text = re.sub(r"<metadata>.*?</metadata>\s*", "", text, flags=re.S)
    text = re.sub(r'<defs>\s*<style type="text/css">\*\{stroke-linejoin: round; stroke-linecap: butt\}</style>\s*</defs>\s*',
                  "", text, flags=re.S)
    path.write_text(text.replace("<svg ", '<svg stroke-linejoin="round" stroke-linecap="butt" ', 1))


def save(fig: plt.Figure, out: Path | str) -> None:
    """The figure as a PNG at 300 dpi and, beside it, as the SVG the document embeds."""
    out = Path(out)
    fig.savefig(out)
    if out.suffix == ".png":
        fig.savefig(out.with_suffix(".svg"))
        _clean_svg(out.with_suffix(".svg"))
    plt.close(fig)
