"""render_paper — the paper on Hudl's Word template, from docs/paper.md and the record.

Reads docs/paper.md, fills every mark {{path|format}} from docs/measured.json, numbers the citations
[@key] in order of first use and prints the references cited as [n], and lays the text on the template
(docs/hudl/): the whole title as Title and the author's name as Subtitle, sections as Heading 1, subsections as Heading 2,
each figure at the text width with its caption in Heading 3 under it, each table under its caption with
the template table's look, each display equation ($$ fenced, LaTeX) as a numbered picture, and footnotes in
the template's own footnotes part. A wrapped line that opens with a roman-numeral marker, "(i) ", starts a new
line of its paragraph, so an inline list stands one item to a line with no paragraph break. A picture with an SVG
beside its PNG carries both, the way Word attaches an SVG, so a zoom stays sharp and a viewer without SVG draws the
PNG; `--raster` leaves the SVGs out. It works on a copy: the
template is copied to the output path first and only the copy is edited, so the template file is never
written. The sample content goes; the styles, embedded fonts, headers and footers stay. A mark that does
not resolve stops the build, so the paper never prints a number the record does not hold.

    python renders/render_paper.py
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import shutil
import subprocess
from collections.abc import Callable
from pathlib import Path
from typing import Any

import docx
from docx.enum.table import WD_ALIGN_VERTICAL
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.opc.constants import RELATIONSHIP_TYPE as RT
from docx.opc.packuri import PackURI
from docx.opc.part import Part
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shape import InlineShape
from docx.shared import Inches, Pt, RGBColor
from docx.table import _Cell
from docx.text.paragraph import Paragraph
from lxml import etree
from PIL import Image, ImageFont

_REPO = Path(__file__).resolve().parents[1]
_PAPER = _REPO / "docs" / "paper.md"
_RECORD = _REPO / "docs" / "measured.json"
_TEMPLATE = _REPO / "docs" / "hudl" / "_Jamie Research Paper template - HPI 26 .docx"
_EQ_DIR = _REPO / "data" / "derived" / "tex_build" / "equations"
_BARLOW = {False: _REPO / "figures" / "fonts" / "Barlow-Regular.ttf", True: _REPO / "figures" / "fonts" / "Barlow-Bold.ttf"}
_CELL_PAD = 0.18                                             # inches: a table cell's left and right margins and a little air
_CAPTION_GAP, _CAPTION_AIR = 6, 3                            # points above and below a title: the room its double spacing left
_REF_PT = 10                                                 # points: the size of the reference list
_ITEM_AFTER = 6                                              # points of air under each item of a numbered list
_HEADER_DISTANCE = 0.3                                       # inches from the top edge to the logo, so the text keeps clear of it
_W ="http://schemas.openxmlformats.org/wordprocessingml/2006/main"
_A = "http://schemas.openxmlformats.org/drawingml/2006/main"
_SVG = "http://schemas.microsoft.com/office/drawing/2016/SVG/main"      # Office's SVG extension of a picture
_SVG_URI = "{96DAC541-7B7A-43D3-8B79-37D633B846F1}"
MARK =re.compile(r"\{\{([^{}|]+)(?:\|(\w+))?\}\}")
CITE = re.compile(r"\[(@[^\]]+)\]")
NOTE = re.compile(r"\[\^(\w+)\]")
INLINE = re.compile(r"(\*\*[^*]+\*\*|\*[^*]+\*)")
LISTED = re.compile(r"\((?:i{1,3}|iv|v)\) ")
APPENDIX = re.compile(r"Appendix ([A-Z])\b")
URL = re.compile(r"(https?://\S+?)(?=[.,;:)]*(?:\s|$))")      # a link ends before the punctuation that follows it
_LINK_COLOR = "0563C1"                                       # the blue Word draws a link in
# the template table: header on the slate fill, body rows alternating the two light fills
_HEAD_FILL, _ROW_FILLS, _INK = "B1BCC4", ("F9FAFB", "EDF0F2"), RGBColor(0x23, 0x2A, 0x31)


# ── marks ────────────────────────────────────────────────────────────────
def _num(v: float) -> str:
    return (f"{v:,}" if isinstance(v, int) else f"{v:g}").replace("-", "−")


def _ord(n: int) -> str:
    return f"{n}{'th' if 11 <= n % 100 <= 13 else {1: 'st', 2: 'nd', 3: 'rd'}.get(n % 10, 'th')}"


FORMATS = {
    None: _num,
    "int": lambda v: f"{int(round(v)):,}",
    "pct": lambda v: f"{100 * v:.2f}%",
    "pctv": lambda v: f"{v:g}%",
    "pp": lambda v: f"{100 * v:+.2f} pp".replace("-", "−"),
    "signed": lambda v: f"{v:+g}".replace("-", "−"),
    "neg": lambda v: _num(-v),
    "ord": lambda v: _ord(int(v)),
    "ci": lambda v: f"[{_num(v[0])}, {_num(v[1])}]",
    "cipct": lambda v: f"[{100 * v[0]:.2f}%, {100 * v[1]:.2f}%]",
    "cipp": lambda v: f"[{100 * v[0]:+.2f}, {100 * v[1]:+.2f}]".replace("-", "−"),
    "negci": lambda v: f"[{_num(-v[1])}, {_num(-v[0])}]",
}
_FIXED = re.compile(r"(n?)(s?)(p?)(f|ci)(\d)")


def _fixed(kind: str) -> Callable[[Any], str]:
    """'[n][s][p](f|ci)<d>': d decimals; n negates, so an interval is reflected and put back in order,
    s prints the sign, p reads a share as a percentage; f prints one number and ci an interval."""
    neg, sign, pct, form, d = _FIXED.fullmatch(kind).groups()

    def one(x: float) -> str:
        x = (-x if neg else x) * (100 if pct else 1)
        if round(x, int(d)) == 0:
            return f"{0:.{d}f}"  # a value that rounds to zero prints unsigned, never "−0.0000"
        return f"{x:{'+' if sign else ''}.{d}f}".replace("-", "−")
    return one if form == "f" else lambda v: "[{}, {}]".format(*map(one, v[::-1] if neg else v))


def gain_pct(step: float) -> float:
    """A step of log loss, in nats per case, as the extra probability the model gives on average to what happened:
    the exponential of the step less one, in percent. The talk prints every step this way."""
    return 100 * math.expm1(step)


_GAIN = re.compile(r"(n?)g(\d)")


def _gain(kind: str) -> Callable[[float], str]:
    """'[n]g<d>': a step in nats as the extra probability on what happened, in percent with d decimals and no sign, the
    way the prose reads it ("about 3.9% more probability"); n reads the record's loss as a saving."""
    neg, d = _GAIN.fullmatch(kind).groups()
    return lambda v: f"{gain_pct(-v if neg else v):.{d}f}%".replace("-", "−")


def resolve(record: dict, path: str, kind: str | None) -> str:
    """The record's value at a dotted path, a list index where a segment is a number, formatted."""
    node = record
    for seg in path.strip().split("."):
        node = node[int(seg)] if isinstance(node, list) else node[seg]
    fmt = FORMATS.get(kind) or (_fixed(kind) if kind and _FIXED.fullmatch(kind) else
                                _gain(kind) if kind and _GAIN.fullmatch(kind) else None)
    if fmt is None:
        raise KeyError(f"unknown format {kind!r} in {{{{{path}|{kind}}}}}")
    return fmt(node)


def fill(text: str, record: dict) -> str:
    return MARK.sub(lambda m: resolve(record, m.group(1), m.group(2)), text)


# ── equations ────────────────────────────────────────────────────────────
# A display equation is a picture: LaTeX typeset by xelatex in Cambria Math, the face of Word's own equations, as
# a transparent PNG at 600 dpi. At 12 pt it matches the x-height of the body's Barlow at 11 pt, and it takes the
# body colour; Word's native equations (OMML) reached the reader's viewer as flattened text, so they are not used.
_EQ_PT, _EQ_COLOR, _EQ_DPI = 12, "4E5D6C", 600
_EQ_COLS = (0.7, 5.1, 0.7)    # inches over the 6.5 in text width: margin | equation | number
_EQ_TEX = (r"\documentclass[%dpt,border=1pt]{standalone}" "\n"
           r"\usepackage{amsmath}" "\n" r"\usepackage{unicode-math}" "\n" r"\usepackage{xcolor}" "\n"
           r"\setmainfont{Cambria}" "\n" r"\setmathfont{Cambria Math}" "\n" r"\begin{document}" "\n"
           r"\color[HTML]{%s}$\displaystyle %s$" "\n" r"\end{document}" "\n")


def equation_png(latex: str) -> Path:
    """The equation as a PNG, and beside it as an SVG, named by the hash of its whole source, so a build typesets
    only what changed."""
    source = _EQ_TEX % (_EQ_PT, _EQ_COLOR, latex)
    name = "eq-" + hashlib.sha1(source.encode()).hexdigest()[:12]
    png = _EQ_DIR / f"{name}.png"
    if not png.exists():
        _EQ_DIR.mkdir(parents=True, exist_ok=True)
        (_EQ_DIR / f"{name}.tex").write_text(source)
        r = subprocess.run(["xelatex", "-interaction=nonstopmode", "-halt-on-error", f"-output-directory={_EQ_DIR}",
                            f"{name}.tex"], cwd=_EQ_DIR, capture_output=True, text=True)
        if r.returncode:
            raise RuntimeError(f"xelatex failed on {latex[:60]!r}:\n{r.stdout[-2500:]}")
        subprocess.run(["pdftocairo", "-png", "-r", str(_EQ_DPI), "-transp", "-singlefile",
                        str(_EQ_DIR / f"{name}.pdf"), str(_EQ_DIR / name)], check=True)
    if not png.with_suffix(".svg").exists():
        subprocess.run(["pdftocairo", "-svg", str(_EQ_DIR / f"{name}.pdf"), str(png.with_suffix(".svg"))], check=True)
    return png


# ── the source, read into blocks ─────────────────────────────────────────
def _join(lines: list[str]) -> str:
    """The wrapped lines of one block as a text: a line that opens with a marker, "(i) ", starts a new line of the
    paragraph, with no paragraph break, and any other line carries on the sentence."""
    text = lines[0].strip()
    for x in lines[1:]:
        text += ("\n" if LISTED.match(x.strip()) else " ") + x.strip()
    return text


def blocks(md: str) -> tuple[list[tuple[str, object]], dict[str, str], dict[str, str]]:
    """The body as (kind, content) blocks, the footnotes by name and the references by key."""
    lines = re.sub(r"<!--.*?-->", "", md, flags=re.S).splitlines()
    body, notes, refs, i, in_refs = [], {}, {}, 0, False
    while i < len(lines):
        line = lines[i]
        if not line.strip():
            i += 1
            continue
        j = i + 1
        if line.startswith("|"):
            while j < len(lines) and lines[j].startswith("|"):
                j += 1
            # a pipe inside a mark, {{path|format}}, is the format's and not a cell border
            rows = [[c.strip() for c in re.split(r"\|(?![^{}]*\}\})", r.strip().strip("|"))]
                    for r in lines[i:j]]
            body.append(("table", [r for r in rows if not set("".join(r)) <= set("-: ")]))
        elif line.startswith("#"):
            level = len(line) - len(line.lstrip("#"))
            text = line.lstrip("#").strip()
            in_refs = text == "References"
            if not in_refs:
                body.append((f"h{level}", text))
            else:
                body.append(("references", None))
        elif line.startswith("!["):
            m = re.match(r"!\[(.*)\]\((.*)\)", line)
            body.append(("figure", (m.group(1), m.group(2))))
        elif line.strip() == "$$":
            while j < len(lines) and lines[j].strip() != "$$":
                j += 1
            if j == len(lines):
                raise ValueError(f"the $$ block opened at line {i + 1} is never closed")
            body.append(("equation", " ".join(x.strip() for x in lines[i + 1:j])))
            j += 1
        else:
            while j < len(lines) and lines[j].strip() and not re.match(r"(#|\||!\[|- \[@|\d+\. |H\d\. |\[\^|\$\$)", lines[j]):
                j += 1
            text = _join(lines[i:j])
            note = re.match(r"\[\^(\w+)\]:\s*(.*)", text)
            ref = re.match(r"- \[@(\w+)\]\s*(.*)", text)
            if note:
                notes[note.group(1)] = note.group(2)
            elif ref and in_refs:
                refs[ref.group(1)] = ref.group(2)
            elif re.match(r"(\d+|H\d)\. ", text):
                body.append(("item", text))
            else:
                body.append(("para", text))
        i = j
    return body, notes, refs


# ── the document ─────────────────────────────────────────────────────────
def _flush(shape: InlineShape) -> None:
    """Pin the wrap distances of an inline picture at zero. A picture that names none gets its viewer's own: LibreOffice
    adds about 9 pt on each side, so a figure as wide as the text starts that far right of its margin and its caption."""
    for side in ("distT", "distB", "distL", "distR"):
        shape._inline.set(side, "0")


class Paper:
    def __init__(self, template: Path, record: dict, notes: dict, refs: dict, vector: bool = True) -> None:
        self.doc, self.record, self.notes, self.refs = docx.Document(str(template)), record, notes, refs
        self.vector = vector
        self.cited: list[str] = []
        self.note_ids: dict[str, int] = {}
        self.n_equations = self.n_vectors = self.n_bookmarks = 0
        body = self.doc.element.body
        for child in list(body):
            if child.tag != qn("w:sectPr"):
                body.remove(child)
        self.fn_part = next(p for p in self.doc.part.package.iter_parts()
                            if str(p.partname) == "/word/footnotes.xml")
        self.fn_root = etree.fromstring(self.fn_part.blob)
        for fn in self.fn_root.findall(f"{{{_W}}}footnote"):
            self.fn_root.remove(fn)

    def _cite(self, m: re.Match) -> str:
        nums = []
        for key in (k.strip().lstrip("@") for k in m.group(1).split(";")):
            if key not in self.refs:
                raise KeyError(f"[@{key}] has no reference")
            if key not in self.cited:
                self.cited.append(key)
            nums.append(str(self.cited.index(key) + 1))
        return f"[{', '.join(nums)}]"

    def text(self, raw: str) -> str:
        return CITE.sub(self._cite, fill(raw, self.record))

    def _link(self, parent: etree._Element, url: str, part: Part, size: str | None = None) -> None:
        """`url` as a live link in `parent`: a hyperlink run related from `part`, in the blue and underline Word
        draws one in, so a click opens the address, a timestamp in it included."""
        link = etree.SubElement(parent, qn("w:hyperlink"))
        link.set(qn("r:id"), part.relate_to(url, RT.HYPERLINK, is_external=True))
        link.set(qn("w:history"), "1")
        rpr = etree.SubElement(etree.SubElement(link, qn("w:r")), qn("w:rPr"))
        etree.SubElement(rpr, qn("w:color")).set(qn("w:val"), _LINK_COLOR)
        for tag in ("w:sz", "w:szCs") if size else ():
            etree.SubElement(rpr, qn(tag)).set(qn("w:val"), size)
        etree.SubElement(rpr, qn("w:u")).set(qn("w:val"), "single")
        etree.SubElement(link[0], qn("w:t")).text = url

    def runs(self, paragraph: Paragraph, raw: str, size: int | None = None) -> None:
        """Text with *italic*, **bold**, footnote references and live links, into runs of `paragraph`, at `size` points
        when one is given."""
        for k, piece in enumerate(NOTE.split(self.text(raw))):
            if k % 2:
                self._footnote(paragraph, piece)
                continue
            for part in INLINE.split(piece):
                if not part:
                    continue
                bold, italic = part.startswith("**"), part.startswith("*") and not part.startswith("**")
                if bold or italic:
                    run = paragraph.add_run(part.strip("*"))
                    run.bold, run.italic = bold or None, italic or None
                    if size:
                        run.font.size = Pt(size)
                    continue
                for j, bit in enumerate(URL.split(part)):
                    if j % 2:
                        self._link(paragraph._p, bit, self.doc.part, str(2 * size) if size else None)
                    elif bit:
                        self._plain(paragraph, bit, size)

    def _plain(self, paragraph: Paragraph, text: str, size: int | None = None) -> None:
        """`text` into runs of `paragraph`, each "Appendix X" a link to the heading of that appendix."""
        for k, piece in enumerate(APPENDIX.split(text)):
            if k % 2:
                self._anchor(paragraph._p, f"Appendix {piece}", f"appendix_{piece}")
            elif piece:
                run = paragraph.add_run(piece)
                if size:
                    run.font.size = Pt(size)

    def _anchor(self, parent: etree._Element, text: str, name: str) -> None:
        """`text` as a link to the bookmark `name` of this document, in the blue of the other links and not underlined."""
        link = etree.SubElement(parent, qn("w:hyperlink"))
        link.set(qn("w:anchor"), name)
        link.set(qn("w:history"), "1")
        run = etree.SubElement(link, qn("w:r"))
        etree.SubElement(etree.SubElement(run, qn("w:rPr")), qn("w:color")).set(qn("w:val"), _LINK_COLOR)
        etree.SubElement(run, qn("w:t")).text = text

    def bookmark(self, paragraph: Paragraph, name: str) -> None:
        """The paragraph's content between a bookmark's start and end, the target of the links to it."""
        self.n_bookmarks += 1
        start, end = OxmlElement("w:bookmarkStart"), OxmlElement("w:bookmarkEnd")
        start.set(qn("w:id"), str(self.n_bookmarks))
        start.set(qn("w:name"), name)
        end.set(qn("w:id"), str(self.n_bookmarks))
        paragraph._p.insert(1 if paragraph._p.pPr is not None else 0, start)
        paragraph._p.append(end)

    def _footnote(self, paragraph: Paragraph, name: str) -> None:
        nid = self.note_ids.setdefault(name, len(self.note_ids) + 1)
        run = paragraph.add_run()
        rpr = run._r.get_or_add_rPr()
        for tag, val in (("w:rStyle", "FootnoteReference"), ("w:vertAlign", "superscript")):
            el = OxmlElement(tag)
            el.set(qn("w:val"), val)
            rpr.append(el)
        ref = OxmlElement("w:footnoteReference")
        ref.set(qn("w:id"), str(nid))
        run._r.append(ref)
        fn = etree.SubElement(self.fn_root, f"{{{_W}}}footnote")
        fn.set(f"{{{_W}}}id", str(nid))
        p = etree.SubElement(fn, f"{{{_W}}}p")
        mark = etree.SubElement(p, f"{{{_W}}}r")
        mrpr = etree.SubElement(mark, f"{{{_W}}}rPr")
        etree.SubElement(mrpr, f"{{{_W}}}rStyle").set(f"{{{_W}}}val", "FootnoteReference")
        etree.SubElement(mrpr, f"{{{_W}}}vertAlign").set(f"{{{_W}}}val", "superscript")
        etree.SubElement(mark, f"{{{_W}}}footnoteRef")
        for j, bit in enumerate(URL.split(" " + self.text(self.notes[name]))):
            if j % 2:
                self._link(p, bit, self.fn_part, "20")
            elif bit:
                r = etree.SubElement(p, f"{{{_W}}}r")
                rpr2 = etree.SubElement(r, f"{{{_W}}}rPr")
                for tag in ("sz", "szCs"):
                    etree.SubElement(rpr2, f"{{{_W}}}{tag}").set(f"{{{_W}}}val", "20")
                t = etree.SubElement(r, f"{{{_W}}}t")
                t.set("{http://www.w3.org/XML/1998/namespace}space", "preserve")
                t.text = bit

    def para(self, raw: str, style: str = "normal") -> Paragraph:
        p = self.doc.add_paragraph(style=style)
        self.runs(p, raw)
        return p

    def _vector(self, shape: InlineShape, png: Path) -> None:
        """Attach the picture's vector twin, the SVG with the PNG's name, the way Word does: as an extension of the
        picture's blip, whose own image, the PNG, stays as the fallback a viewer without SVG draws. A picture with
        no SVG beside it stays a PNG."""
        svg = png.with_suffix(".svg")
        if not (self.vector and svg.exists()):
            return
        self.n_vectors += 1
        part = Part(PackURI(f"/word/media/vector{self.n_vectors}.svg"), "image/svg+xml", svg.read_bytes(),
                    self.doc.part.package)
        blip = shape._inline.graphic.graphicData.pic.blipFill.blip
        ext = etree.SubElement(etree.SubElement(blip, f"{{{_A}}}extLst"), f"{{{_A}}}ext", uri=_SVG_URI)
        etree.SubElement(ext, f"{{{_SVG}}}svgBlip", nsmap={"asvg": _SVG}).set(
            qn("r:embed"), self.doc.part.relate_to(part, RT.IMAGE))

    def caption(self, raw: str) -> Paragraph:
        """A figure's or a table's title in the template's caption style, Heading 3, on single line spacing: the style is
        double-spaced, which opens a gap between the lines of a title that wraps."""
        p = self.para(raw, "Heading 3")
        p.paragraph_format.line_spacing = 1.0
        p.paragraph_format.space_before = Pt(_CAPTION_GAP)
        p.paragraph_format.space_after = Pt(_CAPTION_AIR)
        return p

    def keep_last_with_next(self) -> None:
        """The last paragraph of the body keeps with the block that follows it: the sentence that introduces a display
        equation stays on the page of the equation."""
        self.doc.paragraphs[-1].paragraph_format.keep_with_next = True

    def figure(self, caption: str, path: str) -> None:
        """The picture and, right under it, its title, which never part: the picture keeps with its caption."""
        p = self.doc.add_paragraph(style="normal")
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        p.paragraph_format.keep_with_next = True
        shape = p.add_run().add_picture(str(_REPO / path), width=Inches(6.5))
        _flush(shape)
        self._vector(shape, _REPO / path)
        self.caption(caption)

    def equation(self, latex: str) -> None:
        """A display equation: its picture centred in a borderless 1 x 3 table (margin | equation | number),
        numbered in order, with the LaTeX as the picture's alternative text. The grid and the cell widths agree,
        or a viewer shrinks the picture to the equal columns it assumes."""
        png = equation_png(latex)
        width = Image.open(png).width / _EQ_DPI
        if width > _EQ_COLS[1]:
            raise ValueError(f"equation {self.n_equations + 1} is {width:.2f} in wide, over its {_EQ_COLS[1]} in column")
        self.n_equations += 1
        t = self.doc.add_table(rows=1, cols=3)
        t.autofit = False
        tblw = t._tbl.tblPr.find(qn("w:tblW"))
        tblw.set(qn("w:type"), "dxa")
        tblw.set(qn("w:w"), str(round(sum(_EQ_COLS) * 1440)))
        for col, cell, w in zip(t.columns, t.rows[0].cells, _EQ_COLS, strict=True):
            col.width = cell.width = Inches(w)
        left, mid, right = (c.paragraphs[0] for c in t.rows[0].cells)
        mid.alignment, right.alignment = WD_ALIGN_PARAGRAPH.CENTER, WD_ALIGN_PARAGRAPH.RIGHT
        for p in (left, mid, right):
            p.paragraph_format.space_before = p.paragraph_format.space_after = Pt(4)
        shape = mid.add_run().add_picture(str(png), width=Inches(width))
        shape._inline.docPr.set("descr", latex)
        _flush(shape)
        self._vector(shape, png)
        right.add_run(f"({self.n_equations})")
        t.rows[0]._tr.get_or_add_trPr().append(OxmlElement("w:cantSplit"))
        for p in (left, mid, right):
            p.paragraph_format.keep_with_next = True       # the equation stays with the "where" that explains it
        for cell in t.rows[0].cells:
            tcpr = cell._tc.get_or_add_tcPr()
            tcpr.append(tcpr.makeelement(qn("w:vAlign"), {qn("w:val"): "center"}))

    def table(self, rows: list[list[str]]) -> None:
        """A table of fixed layout whose grid, whole width and cells agree, or a viewer that reads the grid, which python-docx
        sets to equal columns, sets the numbers in columns narrower than their text and breaks them."""
        rows = [[_capital(c) for c in r] for r in rows]
        t = self.doc.add_table(rows=len(rows), cols=len(rows[0]))
        widths = _widths([[self.text(c) for c in r] for r in rows])
        t.autofit = False
        tblw = t._tbl.tblPr.find(qn("w:tblW"))
        tblw.set(qn("w:type"), "dxa")
        tblw.set(qn("w:w"), str(round(sum(widths) * 1440)))
        for col, w in zip(t.columns, widths, strict=True):
            col.width = Inches(w)
        for i, row in enumerate(rows):
            for j, cell_raw in enumerate(row):
                cell = t.cell(i, j)
                cell.width = Inches(widths[j])
                _cell_look(cell, _HEAD_FILL if i == 0 else _ROW_FILLS[(i - 1) % 2])
                cell.vertical_alignment = WD_ALIGN_VERTICAL.CENTER
                p = cell.paragraphs[0]
                p.alignment = WD_ALIGN_PARAGRAPH.CENTER
                self.runs(p, cell_raw)
                for run in p.runs:
                    run.font.size = Pt(10)
                    if i == 0:
                        run.bold, run.font.color.rgb = True, _INK
        # a table stays whole and under its caption: no row breaks across a page, every row but the last keeps with the
        # next, and the header would repeat if a table ever ran over a page
        for i, tr in enumerate(t.rows):
            trpr = tr._tr.get_or_add_trPr()
            trpr.append(OxmlElement("w:cantSplit"))
            if i == 0:
                trpr.append(OxmlElement("w:tblHeader"))
            if i < len(rows) - 1:
                for cell in tr.cells:
                    for p in cell.paragraphs:
                        p.paragraph_format.keep_with_next = True

    def references(self) -> None:
        """The heading and a placeholder: the list is written once the whole paper, appendix included,
        has been read, so a work first cited after the references still gets its entry."""
        self.doc.add_paragraph("References", style="Heading 1").paragraph_format.page_break_before = True
        self.ref_anchor = self.doc.add_paragraph(style="normal")

    def write_references(self) -> None:
        """The list at `_REF_PT` points on single spacing, the way a reference list is set: it takes a page less than the
        body's 11 pt on the template's 1.15."""
        for n, key in enumerate(self.cited, 1):
            p = self.ref_anchor.insert_paragraph_before(style="normal")
            p.paragraph_format.line_spacing = 1.0
            self.runs(p, f"[{n}] {self.refs[key]}", _REF_PT)
        self.ref_anchor._p.getparent().remove(self.ref_anchor._p)

    def save(self, out: Path) -> None:
        self.fn_part._blob = etree.tostring(self.fn_root, xml_declaration=True, encoding="UTF-8",
                                            standalone=True)
        self.doc.save(str(out))


def _capital(text: str) -> str:
    """A cell of a table with its first letter in capitals, as the template's tables set them; a mark or a symbol stays."""
    return text[:1].upper() + text[1:] if text[:1].islower() else text


def _set_in(text: str, bold: bool) -> float:
    """The width in inches of `text` set on one line in Barlow at the table's 10 pt."""
    font = ImageFont.truetype(str(_BARLOW[bold]), 100)
    return font.getlength(text) / 100 * 10 / 72


def _need(rows: list[list[str]], j: int) -> float:
    """What column j must keep, in inches, to set its cells without breaking a number: its widest body cell on one line and,
    from its header, the longest word, since a header wraps at its spaces."""
    head = max((_set_in(w, True) for w in rows[0][j].split()), default=0.0)
    return max([head] + [_set_in(r[j], False) for r in rows[1:]])


def _widths(rows: list[list[str]]) -> list[float]:
    """Column widths over the 6.5 in text width. A column of numbers or short labels takes what it needs on one line plus
    the cell's margins, so that no number is broken across lines whatever the viewer; the columns of sentences share what is
    left in proportion to their text, and the whole fills the text width."""
    n = len(rows[0])
    need = [1.04 * _need(rows, j) + _CELL_PAD for j in range(n)]
    sentences = [j for j in range(n) if max(len(r[j]) for r in rows) > 30]
    if not sentences:
        return [6.5 * w / sum(need) for w in need]
    room = max(6.5 - sum(need[j] for j in range(n) if j not in sentences), 0.9 * len(sentences))
    total = sum(need[j] for j in sentences)
    widths = [room * need[j] / total if j in sentences else need[j] for j in range(n)]
    # a column of text never takes more than its widest cell on one line: the rest goes to the other columns, so that their
    # headers wrap in two lines at most
    other = [j for j in range(n) if j not in sentences]
    for j in sentences:
        one_line = 1.04 * max(_set_in(r[j], False) for r in rows[1:]) + _CELL_PAD
        if other and widths[j] > one_line:
            extra, widths[j] = widths[j] - one_line, one_line
            for k in other:
                widths[k] += extra / len(other)
    if sum(widths) > 6.5 + 1e-6:
        raise ValueError(f"a table of {n} columns needs {sum(widths):.2f} in, over the 6.5 in text width")
    return [6.5 * w / sum(widths) for w in widths]


def _cell_look(cell: _Cell, fill: str) -> None:
    tcpr = cell._tc.get_or_add_tcPr()
    shd = OxmlElement("w:shd")
    shd.set(qn("w:val"), "clear")
    shd.set(qn("w:fill"), fill)
    tcpr.append(shd)
    borders = OxmlElement("w:tcBorders")
    for side in ("top", "left", "bottom", "right"):
        b = OxmlElement(f"w:{side}")
        for k, v in (("val", "single"), ("sz", "8"), ("space", "0"), ("color", "FFFFFF")):
            b.set(qn(f"w:{k}"), v)
        borders.append(b)
    tcpr.append(borders)


def _blank_before(kind: str, prev: str, text: str = "") -> int:
    """Empty paragraphs before a block, the way the template separates them: its styles carry no spacing,
    so it puts one empty paragraph after the author's name and before a subsection or a figure, two before
    a section, and one between two paragraphs of text; a heading is followed straight by its text, and
    list items and references follow each other with none. A display equation follows its sentence with none,
    and a paragraph that starts in lowercase carries on the sentence an equation interrupted."""
    if prev in ("h2", "h3", "start"):
        return 0
    if kind in ("h2", "references"):
        return 2
    if kind == "equation" or (kind == "item" and prev == "item") or (prev == "equation" and text[:1].islower()):
        return 0
    return 1


def build(paper: Path, record: dict, template: Path, out: Path, vector: bool = True) -> Paper:
    if out.resolve() == template.resolve():
        raise ValueError("the output would overwrite the template")
    shutil.copyfile(template, out)
    body, notes, refs = blocks(paper.read_text())
    doc = Paper(out, record, notes, refs, vector)
    doc.doc.sections[0].header_distance = Inches(_HEADER_DISTANCE)
    head = [b for b in body if b[0] in ("h1", "para")][:2]    # the whole title, then the author
    styles = iter(("Title", "Subtitle"))
    skip = {id(b) for b in head}
    # the template opens on an empty Title paragraph above the title, and leaves one empty line after the name
    doc.para("", "Title")
    for b in head:
        style = next(styles)
        # the title breaks after its colon, so the subtitle stands whole on its own line
        doc.para(b[1].replace(": ", ":\n", 1) if style == "Title" else b[1], style)
    doc.para("")
    prev, last = "start", None
    for b in body:
        kind, content = b
        if id(b) in skip:
            continue
        # the references and the appendix open a page of their own, as in the template; the conclusion follows the text
        new_page = kind == "references" or (kind == "h2" and bool(re.fullmatch(r"Appendix", content)))
        gap = [doc.para("") for _ in range(0 if new_page else _blank_before(kind, prev, content if kind == "para" else ""))]
        if kind == "item" and prev == "para":
            # the sentence that introduces a list stays on the page of its first item
            for p in (last, *gap):
                p.paragraph_format.keep_with_next = True
        if kind == "h2":
            p = doc.para(content, "Heading 1")
            p.paragraph_format.page_break_before = new_page or None
        elif kind == "h3":
            p = doc.para(content, "Heading 2")
            if appendix := re.match(r"([A-Z])\. ", content):
                doc.bookmark(p, f"appendix_{appendix.group(1)}")
        elif kind in ("para", "item"):
            caption = re.match(r"Table [A-Z]?\d+:", content)
            if caption:
                doc.caption(content)
            else:
                last = doc.para(content, "normal")
                if kind == "item":
                    last.paragraph_format.space_after = Pt(_ITEM_AFTER)
            kind = "caption" if caption else kind
        elif kind == "figure":
            doc.figure(*content)
        elif kind == "equation":
            doc.keep_last_with_next()
            doc.equation(content)
        elif kind == "table":
            doc.table(content)
        elif kind == "references":
            doc.references()
        # a table sits right under its caption, as the figure sits right over its own
        prev = "h3" if kind == "caption" else kind
    doc.write_references()
    doc.save(out)
    return doc


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=_REPO / "docs" / "paper.docx")
    ap.add_argument("--raster", action="store_true", help="embed the PNGs only, leaving out the SVG twins")
    args = ap.parse_args()
    record = json.loads(_RECORD.read_text())["figures"]
    doc = build(_PAPER, record, _TEMPLATE, args.out, vector=not args.raster)
    print(f"wrote {args.out}: {len(doc.cited)} references, {len(doc.note_ids)} footnotes, {doc.n_vectors} vector pictures")


if __name__ == "__main__":
    main()
