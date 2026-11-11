"""The paper's build: every mark in docs/paper.md resolves against the record, every citation has its
reference, and the document is laid on a copy of Hudl's template, which is never written."""
from __future__ import annotations

import json
import math
import re
import sys
from pathlib import Path

import pytest

_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO / "renders"))
pytest.importorskip("docx")                 # the optional `paper` group: without it the module is skipped
pytest.importorskip("lxml")
import render_paper as rp  # noqa: E402

_PAPER = _REPO / "docs" / "paper.md"
_RECORD = _REPO / "docs" / "measured.json"
# Most corner goals in a Premier League season before 2025/26, the record the introduction quotes (The Analyst, Opta,
# held jointly by three sides: docs/sota/theanalyst_arsenal_tottenham_corners_record.md).
_PREVIOUS_RECORD = 16

pytestmark = pytest.mark.skipif(not _PAPER.exists() or not _RECORD.exists(),
                                reason="docs/paper.md or the record is missing")


def _source() -> str:
    """The paper as the build reads it, its comments removed."""
    return re.sub(r"<!--.*?-->", "", _PAPER.read_text(), flags=re.S)


def test_every_mark_resolves_against_the_record():
    record = json.loads(_RECORD.read_text())["figures"]
    marks = rp.MARK.findall(_source())
    assert marks
    for path, kind in marks:
        assert rp.resolve(record, path, kind or None), path


def test_the_formats_print_the_record_s_values():
    assert rp.FORMATS["pct"](0.4873) == "48.73%"
    assert rp.FORMATS["signed"](-0.00381) == "−0.00381"
    assert rp.FORMATS["negci"]([-0.11939, -0.10514]) == "[0.10514, 0.11939]"
    assert rp.FORMATS["pp"](0.1369) == "+13.69 pp"
    assert (rp.FORMATS["ord"](17), rp.FORMATS["ord"](1), rp.FORMATS["ord"](12)) == ("17th", "1st", "12th")
    assert rp.FORMATS["int"](16720) == "16,720"
    record = {"share": 0.4873, "ci": [0.0164, 0.0449], "neg": [-0.11939, -0.10514], "step": -0.00381}
    assert rp.resolve(record, "share", "pf1") == "48.7"
    assert rp.resolve(record, "ci", "spci1") == "[+1.6, +4.5]"
    assert rp.resolve(record, "neg", "nci4") == "[0.1051, 0.1194]"
    assert rp.resolve(record, "step", "sf4") == "−0.0038"
    assert rp.resolve({"tiny": -0.00001}, "tiny", "sf4") == "0.0000"
    assert rp.resolve({"step": 0.0381}, "step", "g1") == "3.9%" and rp.resolve({"loss": -0.00632}, "loss", "ng1") == "0.6%"
    with pytest.raises(KeyError):
        rp.resolve(record, "step", "f")


def test_a_hundredth_of_a_nat_is_about_one_percent_more_probability():
    """Section 2.4 reads a step of 0.01 nats as about 1% more probability, on average, for what happened: the exponential
    of the step less one, which is what the talk prints for every step."""
    assert round(100 * math.expm1(0.01)) == 1
    assert "A step of 0.01 nats means that the added information raises, on average, the probability that the model " \
           "gives to what happened by about 1%." in " ".join(_source().split())


def test_every_citation_note_and_figure_has_its_source():
    body, notes, refs = rp.blocks(_PAPER.read_text())
    cited = {k.strip().lstrip("@") for m in rp.CITE.findall(_source()) for k in m.split(";")}
    assert cited <= set(refs), cited - set(refs)
    assert set(rp.NOTE.findall(_source())) == set(notes)
    for kind, content in body:
        if kind == "figure":
            assert (_REPO / content[1]).exists(), content[1]


def test_the_two_named_sides_lead_after_the_first_touch():
    """The introduction says Tottenham holds the highest count of goals after the first touch among the sides of the five
    leagues and Arsenal the joint second-highest: Tottenham strictly above every other side, Arsenal's count the second
    value of the table, shared with at least one more side, and no side above it."""
    import pandas as pd
    table = _REPO / "data" / "derived" / "eda_teams_corner.parquet"
    if not table.exists():
        pytest.skip("the team table is not built")
    t = pd.read_parquet(table)
    top = t["aft_goals_for"].sort_values(ascending=False).tolist()
    spurs, arsenal = (int(t.loc[t["team"] == n, "aft_goals_for"].iloc[0]) for n in ("Tottenham Hotspur", "Arsenal"))
    assert spurs == top[0] > top[1] == arsenal and top.count(arsenal) > 1, top[:4]
    assert "which are the joint second-highest and the highest counts among the" in " ".join(_source().split())


def test_the_two_named_seasons_are_as_the_introduction_tells_them():
    """Arsenal won the Premier League by seven points, Tottenham finished 17th two points above the drop, and
    the two scored more goals from corners than any other side of the league."""
    import pandas as pd
    der = _REPO / "data" / "derived"
    if not (der / "eda_standings.parquet").exists() or not (der / "eda_teams_corner.parquet").exists():
        pytest.skip("the standings or the team table are not built")
    s = pd.read_parquet(der / "eda_standings.parquet").query("league == 'premier_league'").set_index("pos")
    assert (s.loc[1, "team"], s.loc[17, "team"]) == ("Arsenal", "Tottenham Hotspur")
    assert (s.loc[1, "pts_ref"] - s.loc[2, "pts_ref"], s.loc[17, "pts_ref"] - s.loc[18, "pts_ref"]) == (7, 2)
    t = pd.read_parquet(der / "eda_teams_corner.parquet").query("league == 'premier_league'")
    t = t.sort_values("goals_for_all", ascending=False)
    assert list(t["team"][:2]) == ["Arsenal", "Tottenham Hotspur"] and t["goals_for_all"].iloc[1] > t["goals_for_all"].iloc[2]


def test_build_lays_the_paper_on_a_copy_of_the_template(tmp_path):
    import docx
    if not rp._TEMPLATE.exists():
        pytest.skip("the Word template is missing")
    before = rp._TEMPLATE.read_bytes()
    record = json.loads(_RECORD.read_text())["figures"]
    built = rp.build(_PAPER, record, rp._TEMPLATE, tmp_path / "paper.docx")
    assert rp._TEMPLATE.read_bytes() == before
    d = docx.Document(str(tmp_path / "paper.docx"))
    paras = d.paragraphs
    assert (paras[0].style.name, paras[0].text) == ("Title", "")
    assert [p.style.name for p in paras if p.text][:2] == ["Title", "Subtitle"]
    title = next(p for p in paras if p.text and p.style.name == "Title")
    assert title.text.split("\n") == ["After the First Touch:", "What Decides the Second Ball at Set Pieces"], title.text
    breaks = {p.text for p in paras if p.paragraph_format.page_break_before}
    assert {"References", "Appendix"} <= breaks and not any(re.fullmatch(r"\d+\. Conclusion", t) for t in breaks)
    from docx.oxml.ns import qn
    from docx.text.paragraph import Paragraph
    # the body in document order: a table, an equation's included, separates the paragraphs around it
    order = [Paragraph(c, d) if c.tag == qn("w:p") else None for c in d.element.body.iterchildren()
             if c.tag in (qn("w:p"), qn("w:tbl"))]
    refs_at = next(k for k, p in enumerate(order) if p is not None and p.text == "References")
    for a, b in zip(order[:refs_at - 1], order[1:refs_at], strict=True):
        if a is not None and b is not None and a.style.name == b.style.name == "normal" and a.text and b.text:
            assert re.match(r"(\d+|H\d)\. ", b.text), f"two paragraphs with no blank between: {b.text[:40]}"
    section_2 = next(k for k, p in enumerate(paras) if p.style.name == "Heading 1" and p.text.startswith("2. "))
    intro = paras[:section_2]
    first_image = next(k for k, p in enumerate(intro) if p._p.xpath(".//w:drawing"))
    assert any("(Figure 1)" in p.text for p in intro[:first_image]), \
        "the introduction's figure must follow the paragraph that cites it"
    assert sum(len(p.text.split()) for p in intro[:first_image]) >= 300, \
        "the introduction's figure must come after enough text to leave the first page"
    text = "\n".join(p.text for p in paras)
    assert "{{" not in text and "[@" not in text and "[^" not in text
    blocks = rp.blocks(_PAPER.read_text())[0]
    figures = [c for k, c in blocks if k == "figure"]
    assert len(d.inline_shapes) == len(figures) + sum(k == "equation" for k, _ in blocks)
    captions = [p.text.split(":")[0] for p in paras
                if p.style.name == "Heading 3" and p.text.startswith("Figure")]
    assert captions == [f"Figure {n}" for n in range(1, len(figures) + 1)]
    start = next(k for k, p in enumerate(paras) if p.text == "References")
    numbers = [int(m.group(1)) for p in paras[start + 1:] if (m := re.match(r"\[(\d+)\] ", p.text))]
    assert numbers == list(range(1, len(built.cited) + 1))
    parts = {str(p.partname): p for p in d.part.package.iter_parts()}
    assert {"/word/fonts/Barlow-regular.ttf", "/word/header1.xml", "/word/footer1.xml"} <= set(parts)
    assert parts["/word/footnotes.xml"].blob.decode().count("<w:footnote ") == len(built.note_ids)


def test_the_equations_are_numbered_in_order_and_carry_their_latex(tmp_path):
    """Each display equation is a picture in its own borderless table, numbered at the right in the order of the
    source, with its LaTeX as the picture's alternative text; the sentence an equation interrupts follows it with
    no blank line."""
    import docx
    if not rp._TEMPLATE.exists():
        pytest.skip("the Word template is missing")
    record = json.loads(_RECORD.read_text())["figures"]
    rp.build(_PAPER, record, rp._TEMPLATE, tmp_path / "paper.docx")
    d = docx.Document(str(tmp_path / "paper.docx"))
    latex = [c for k, c in rp.blocks(_PAPER.read_text())[0] if k == "equation"]
    tables = [t for t in d.tables if len(t.columns) == 3 and re.fullmatch(r"\(\d+\)", t.rows[0].cells[2].text)]
    assert [t.rows[0].cells[2].text for t in tables] == [f"({n})" for n in range(1, len(latex) + 1)]
    assert [t.rows[0].cells[1]._tc.xpath(".//wp:docPr/@descr")[0] for t in tables] == latex
    assert rp._blank_before("equation", "para") == 0
    assert rp._blank_before("para", "equation", "where the weights are fitted") == 0
    assert rp._blank_before("para", "equation", "The next paragraph") == 1


def test_the_figure_captions_are_one_short_line():
    """A caption names what is drawn and its scope, in one line and no conclusion: the paper's twelve figures hold to it."""
    captions = [c[0] for k, c in rp.blocks(_PAPER.read_text())[0] if k == "figure"]
    assert len(captions) == 12
    for c in captions:
        assert len(c.split()) <= 24 and ". " not in c, c


def test_the_constants_the_method_quotes_are_the_code_s():
    """Section 2 says the window, the free-kick cut, the frame rate, the six yards of the frontal, the fifteen yards from
    goal where G2's second row ends and its edge begins, and how many segments, jobs, cells and outcomes there are, down
    to the equal columns, the rows and the side cells that G2's cells make in the box, in words and symbols, and the wait
    for the cross that delivers a corner played short and the length under which a corner counts as short: each is the
    code's own value, so a change in the code breaks the sentence that quotes it."""
    sys.path.insert(0, str(_REPO / "src"))
    import constants as C
    import zones as Z
    from s01_phases import SEGMENTS
    from s03_value import TARGET_MAP
    from zones import FRONTAL, JOBS, ZONES
    words = {2: "two", 3: "three", 5: "five", 6: "six", 8: "eight", 11: "eleven", 15: "fifteen"}
    text = " ".join(_source().split())
    assert f"{words[int(C.SECOND_BALL_WINDOW_S)]} seconds" in text
    assert f"within {words[int(C.SHORT_CROSS_WINDOW_S)]} seconds of the corner kick" in text
    assert f"travels less than {words[int(C.SHORT_CORNER_MAX_LENGTH)]} yards" in text
    assert f"x ≥ {int(C.FK_FLANK_MIN_X)}" in text
    assert f"{C.FPS_TRACKING} frames per second" in text
    assert C.BOX_X - C.HOLD_X == C.PITCH_X_SB - C.SIX_YARD_X == 6 and C.ZONE_PARTITION == "G2"
    assert f"up to {words[6]} yards short of its line" in text and FRONTAL["G2"] == "edge"
    assert f"between its line and {words[int(C.G2_EDGE_CUT)]} yards from goal" in text
    assert f"from its line to {words[int(C.G2_EDGE_CUT)]} yards from goal" in text
    assert f"same {words[len(ZONES['G2'])]} cells" in text and f"{words[len(JOBS)]} jobs" in text
    inner = [z.split("_") for z in ZONES["G2"] if z.startswith(("six_", "spot_"))]
    rows, columns = {r for r, _ in inner}, {c for _, c in inner}
    sides = [z for z in ZONES["G2"] if z.endswith("_side")]
    widths = (Z._T1 - Z._U0, Z._T2 - Z._T1, Z._U1 - Z._T2)
    assert len(inner) == len(rows) * len(columns) and len(sides) == 2 and max(widths) - min(widths) < 1e-9
    assert f"{words[len(columns)]} equal columns cross {words[len(rows)]} rows" in text
    assert f"in {words[len(columns)]} equal columns, near, central and far, over {words[len(rows)]} rows" in text
    assert "makes one more cell each" in text and "one cell on each side of that width" in text
    assert f"The other {words[len(ZONES['G2']) - len(inner) - len(sides)]} are the edge of the box" in text
    assert f"{words[len(SEGMENTS)]} segments" in text
    assert f"{words[len(set(TARGET_MAP.values()))]} ways" in text and f"{words[len(set(TARGET_MAP.values()))]} outcomes" in text


def test_the_annex_on_the_window_says_what_the_curve_of_the_log_shows():
    """Appendix D says that the possession chain stays at or above the 95% that a finding needs through the ninth second and falls
    below it in the tenth, and that the window closes
    a second before the last second that holds, at the code's eight. The curve has no twin in the record (docs/definitions.md §5),
    so each word is read from its log, whose shots inside the window are the shots that the segments hold."""
    sys.path.insert(0, str(_REPO / "src"))
    sys.path.insert(0, str(_REPO / "renders"))
    import constants as C
    import render_window as rw
    text = " ".join(_source().split())
    assert "through the ninth second and falls below it in the tenth" in text
    assert f"The cutoff sits at {({8: 'eight'}[int(C.SECOND_BALL_WINDOW_S)])} seconds, a second before the last second that holds" in text
    assert "(Appendix D)" in text and "### D. Why eight seconds" in _source()
    if not rw.LOG.exists():
        pytest.skip("the late-boundary log is not there")
    curve = rw.read_curve()
    assert (curve.loc[:9, "share"] >= rw.FINDING_PCT).all() and curve.loc[10, "share"] < rw.FINDING_PCT
    assert 9 - int(C.SECOND_BALL_WINDOW_S) == 1, "the cutoff is one second before the last second that holds"
    conv = json.loads(_RECORD.read_text())["figures"]["falsification"]["conversion"]
    in_window = sum(conv[s]["n"] for s in ("direct", "flick_on", "prolongation", "after_clearance"))
    assert int(curve.loc[:int(C.SECOND_BALL_WINDOW_S), "shots"].sum()) == in_window, "the curve's shots in the window are the segments'"


def test_the_annex_on_the_window_says_what_the_curve_of_the_cross_shows():
    """Appendix D says that, for a corner played short counted from the corner kick, the share of the side's crosses still in the corner's
    possession chain stays at or above the 95% that a finding needs through the twelfth second and falls below it in the
    thirteenth, which puts the rule that gave the episode its eight seconds at eleven, and that of the first crosses fewer
    than one in twenty follow a touch by a defender within eight seconds and more than one in ten between the eighth and the
    twelfth. The curve has no twin in the record, so each word is read from its log, and the wait is the episode's own eight."""
    sys.path.insert(0, str(_REPO / "src"))
    sys.path.insert(0, str(_REPO / "renders"))
    import constants as C
    import render_window as rw
    text = " ".join(_source().split())
    for phrase in ("stays at or above 95% through the twelfth second and falls below it in the thirteenth",
                   "would therefore put the wait for the cross at eleven",
                   "the longer a cross takes, the more time a defender has to touch the ball first",
                   "fewer than one in twenty of the first crosses follow a touch by a defender",
                   "between the eighth and the twelfth second more than one in ten do",
                   "Hence, the wait is the same eight seconds as the episode, and not eleven"):
        assert phrase in text, phrase
    assert rw.FINDING_PCT == 95 and C.SHORT_CROSS_WINDOW_S == C.SECOND_BALL_WINDOW_S == 8
    log = _REPO / "data" / "derived" / "prereg_2026-11-04" / "evidence" / "cross_window_curve.log"
    if not log.exists():
        pytest.skip("the cross curve's log is not there")
    lines = log.read_text().splitlines()
    chain = {int(p[0]): (int(p[1]), float(p[2])) for p in (ln.split() for ln in lines) if len(p) == 5 and p[0].isdigit()}
    held = max(s for s in chain if all(chain[k][1] >= rw.FINDING_PCT for k in range(1, s + 1)))
    assert held == 12 and chain[held + 1][0] >= 20 and chain[held + 1][1] < rw.FINDING_PCT and held - 1 == 11
    touched = {m.group(1): float(m.group(2)) for ln in lines
               if (m := re.search(r"first crosses (within 8 s|from 8 to 12 s) of the kick: \d+, ([\d.]+)%", ln))}
    assert touched["within 8 s"] < 100 / 20 and touched["from 8 to 12 s"] > 100 / 10
    shares = [pct for *_, pct in rw.read_cross()[1]]
    assert len(shares) == 4 and shares == sorted(shares), "the share of first crosses after a defender's touch rises with the wait"


def _sections_done() -> str:
    """The paper as the build reads it, without its list of references: every section and the appendix are finished."""
    return re.sub(r"\n## References\n.*?\n## Appendix", "\n## Appendix", _source(), flags=re.S)


def test_the_sections_done_keep_one_vocabulary():
    """The finished sections keep to the terms the paper fixed: a segmentation (no "cut" or "grammar"), nested
    models and a step (no ladder or rungs), intervals named by their form (no "reading"), models that do not ask,
    read or price, and no "we", because the paper has one author (a quotation may say it). The marks, which are
    paths into the record, are not prose."""
    text = re.sub(r'"[^"]*"|\{\{.*?\}\}', "", " ".join(_sections_done().split()))
    banned = [r"\bcuts?\b", r"\bcutting\b", r"\bgrammar\b", r"\bladder\b", r"\brungs?\b", r"\breading\b",
              r"\b(?:model|segmentation|frame|freeze-frame) (?:asks|reads|sees|prices)\b", r"\bprices?\b",
              r"\binterestingly\b", r"\bdown to chance\b", r"\bdo(?:es)? not hold\b", r"\bdoes not separate\b",
              r"\bthings\b", r"\b(?:we|our|us)\b"]
    found = [m.group(0) for p in banned for m in re.finditer(p, text, flags=re.I)]
    assert not found, found


def test_the_words_that_stand_for_a_figure_are_the_record_s():
    """The method says the frames hold "a single player" in the far half, the introduction says "thirty matches"
    carry tracking, quotes the previous record from The Analyst, and counts its questions and its hypotheses: each
    word is the record's or the text's own."""
    text = " ".join(_sections_done().split())
    record = json.loads(_RECORD.read_text())["figures"]
    assert f"record of {_PREVIOUS_RECORD} [@" in text
    assert "hold a single player in the half of the pitch furthest" in text
    assert record["coverage_360"]["bodies_deep_in_whole_grant"] == 1
    assert "thirty matches" in text and record["scope"]["tracked_matches"] == 30
    assert "two questions" in text and "three hypotheses" in text
    assert len(re.findall(r"(?<![\w(])H[123]\. ", text)) == 3


def test_the_lists_are_numbered_items_or_new_lines_of_one_paragraph(tmp_path):
    """The two questions of the introduction are numbered items, like the hypotheses. An inline list, one that opens
    "(i) ", "(ii) ", stands one item to a line inside its paragraph: a line break and no paragraph break, so
    no item is a paragraph of its own."""
    import docx
    if not rp._TEMPLATE.exists():
        pytest.skip("the Word template is missing")
    record = json.loads(_RECORD.read_text())["figures"]
    rp.build(_PAPER, record, rp._TEMPLATE, tmp_path / "paper.docx")
    texts = [p.text for p in docx.Document(str(tmp_path / "paper.docx")).paragraphs]
    assert [t[:2] for t in texts if t.startswith(("1. How much of the value", "2. When a team wins the second ball"))] == ["1.", "2."]
    for lead, markers in (("Information is added in three nested models", ["(i)", "(ii)", "(iii)"]),
                          ("A difference counts as a finding", ["(i)", "(ii)"])):
        lines = next(t for t in texts if t.startswith(lead)).split("\n")
        assert [x.split(" ")[0] for x in lines[1:]] == markers, lead
    assert not any(t.startswith(("(i)", "(ii)")) for t in texts)


def test_every_picture_carries_its_vector_twin(tmp_path):
    """A figure or an equation holds its PNG as the image every viewer draws and, the way Word attaches an SVG, an
    extension of the picture that points at the SVG beside it, so a zoom stays sharp; `vector=False` leaves them out."""
    import zipfile

    from lxml import etree
    if not rp._TEMPLATE.exists():
        pytest.skip("the Word template is missing")
    record = json.loads(_RECORD.read_text())["figures"]
    ns = {"a": "http://schemas.openxmlformats.org/drawingml/2006/main",
          "asvg": "http://schemas.microsoft.com/office/drawing/2016/SVG/main"}
    embed = "{http://schemas.openxmlformats.org/officeDocument/2006/relationships}embed"

    def pictures(path):
        z = zipfile.ZipFile(path)
        rels = {r.get("Id"): r.get("Target") for r in etree.fromstring(z.read("word/_rels/document.xml.rels"))}
        doc = etree.fromstring(z.read("word/document.xml"))
        return z, rels, doc.findall(".//a:blip", ns), doc.findall(".//asvg:svgBlip", ns)

    rp.build(_PAPER, record, rp._TEMPLATE, tmp_path / "vector.docx")
    rp.build(_PAPER, record, rp._TEMPLATE, tmp_path / "raster.docx", vector=False)
    z, rels, blips, svgs = pictures(tmp_path / "vector.docx")
    assert blips and len(svgs) == len(blips)
    assert all(rels[b.get(embed)].endswith(".png") for b in blips)
    for s in svgs:
        target = rels[s.get(embed)]
        assert target.endswith(".svg") and z.read("word/" + target).lstrip().startswith(b"<?xml"), target
    assert not pictures(tmp_path / "raster.docx")[3]


def test_a_saved_figure_comes_with_a_clean_svg(tmp_path):
    """`save` writes the PNG and, beside it, an SVG that Word can draw: text as outlines, and no DOCTYPE, metadata or
    style sheet, whose one rule, round joins and flat caps, sits on the root instead."""
    from lxml import etree
    sys.path.insert(0, str(_REPO / "src"))
    from viz import paper as P
    P.setup()
    fig = P.figure(1.0)
    ax = fig.add_subplot()
    ax.plot([0, 1], [0, 1])
    ax.set_title("Goals")
    P.save(fig, tmp_path / "f.png")
    svg = (tmp_path / "f.svg").read_text()
    assert (tmp_path / "f.png").stat().st_size > 0
    assert not any(tag in svg for tag in ("<!DOCTYPE", "<metadata", "<style", "<text"))
    assert etree.fromstring(svg.encode()).get("stroke-linejoin") == "round"


def test_a_link_is_live_and_the_video_opens_where_the_text_says(tmp_path):
    """A URL in a footnote or in a reference is a hyperlink, so a click opens it. The introduction's footnote names a
    corner at a time of Arsenal's video, and the link opens the video at that time, in seconds."""
    import zipfile

    from lxml import etree
    if not rp._TEMPLATE.exists():
        pytest.skip("the Word template is missing")
    rp.build(_PAPER, json.loads(_RECORD.read_text())["figures"], rp._TEMPLATE, tmp_path / "paper.docx")
    z = zipfile.ZipFile(tmp_path / "paper.docx")
    w = {"w": "http://schemas.openxmlformats.org/wordprocessingml/2006/main"}
    rid = "{http://schemas.openxmlformats.org/officeDocument/2006/relationships}id"

    def links(part: str, rels: str) -> list[str]:
        target = {r.get("Id"): r.get("Target") for r in etree.fromstring(z.read(rels)) if r.get("TargetMode") == "External"}
        return [target[h.get(rid)] for h in etree.fromstring(z.read(part)).iterfind(".//w:hyperlink", w) if h.get(rid)]

    video = [u for u in links("word/footnotes.xml", "word/_rels/footnotes.xml.rels") if "youtube.com" in u]
    minutes, seconds = re.search(r"corner at (\d+):(\d\d) ", _source()).groups()
    assert len(video) == 1 and "v=Mwtve2XkOus" in video[0]
    assert video[0].endswith(f"&t={60 * int(minutes) + int(seconds)}s")
    assert any("skysports.com" in u for u in links("word/document.xml", "word/_rels/document.xml.rels"))


def test_a_table_gives_every_number_the_width_it_needs(tmp_path):
    """A data table has a fixed layout whose grid, whole width and cells agree, and every column that is not a column
    of sentences is as wide as its widest cell set on one line plus the cell's margins: a viewer that reads the grid, or
    that does not widen a column to fit a word, would otherwise break a number across lines."""
    import docx
    from docx.oxml.ns import qn
    if not rp._TEMPLATE.exists():
        pytest.skip("the Word template is missing")
    rp.build(_PAPER, json.loads(_RECORD.read_text())["figures"], rp._TEMPLATE, tmp_path / "paper.docx")
    d = docx.Document(str(tmp_path / "paper.docx"))
    data = [t for t in d.tables if not re.fullmatch(r"\(\d+\)", t.rows[0].cells[-1].text)]      # an equation sits in a table too
    assert len(data) == sum(k == "table" for k, _ in rp.blocks(_PAPER.read_text())[0])
    for t in data:
        grid = [int(g.get(qn("w:w"))) / 1440 for g in t._tbl.tblGrid.findall(qn("w:gridCol"))]
        assert t._tbl.tblPr.find(qn("w:tblLayout")).get(qn("w:type")) == "fixed"
        assert sum(grid) == pytest.approx(6.5, abs=0.01)
        rows = [[c.text for c in r.cells] for r in t.rows]
        for j, w in enumerate(grid):
            assert all(r.cells[j].width.inches == pytest.approx(w, abs=0.01) for r in t.rows)
            if max(len(r[j]) for r in rows) <= 30:
                need = rp._need(rows, j) + 0.15
                assert w >= need, (rows[0][0], j, round(w, 2), round(need, 2))


def test_the_readings_of_section_3_are_the_record_s():
    """Section 3 says in words what its tables and figures hold, and each word is the record's: a flick-on scores
    about two and a half times as often as a direct shot and a prolongation about twice, a shot after a clearance does
    not differ, the share of goals is the least certain and its interval lies below an even split, the flank is about a
    sixth of the corners, most direct shots are headers and most after the first touch are not, the phase makes no
    difference within headers, and the headers after the first touch do not fall below one under either kind of
    interval."""
    record = json.loads(_RECORD.read_text())["figures"]
    text = " ".join(_source().split())
    conv, delta = record["falsification"]["conversion"], record["falsification"]["delta"]
    corner, flank = record["frontier"]["corner"], record["frontier"]["free_kick_flank"]
    assert "about two and a half times as often as a direct shot" in text and "about twice as often" in text
    assert "about a sixth the size" in text and "even its interval lies below an even split" in text
    assert 2.4 < conv["flick_on"]["point"] / conv["direct"]["point"] < 2.6
    assert 1.9 < conv["prolongation"]["point"] / conv["direct"]["point"] < 2.4
    assert delta["after_clearance-direct"]["covers_zero"] and not delta["flick_on-direct"]["covers_zero"]
    assert not delta["prolongation-direct"]["covers_zero"]
    head = corner["headline"]
    shares = {k: head[f"share_{k}_after_first_touch"] for k in ("shots", "xg", "goals")}
    width = {k: v["ci"][1] - v["ci"][0] for k, v in shares.items()}
    assert width["goals"] == max(width.values()) and shares["goals"]["ci"][1] < 0.5
    assert shares["shots"]["point"] < shares["xg"]["point"]
    assert round(record["scope"]["corner"]["deliveries"] / record["scope"]["free_kick_flank"]["deliveries"]) == 6
    assert flank["headline"]["delta_aft_minus_direct"]["covers_zero"]
    after = ("flick_on", "prolongation", "after_clearance")
    cells, seg = corner["body_part"], corner["segment"]
    assert cells["direct|head"]["shots"] > seg["direct"]["shots"] / 2
    assert cells["after_first_touch|non_head"]["shots"] > sum(seg[s]["shots"] for s in after) / 2
    assert all(seg[s]["ci"][0] < 1 < seg[s]["ci"][1] for s in after) and seg["direct"]["ci"][1] < 1
    assert corner["body_part_contrast"]["segment_within_head"]["covers_zero"]
    assert corner["body_part_pooled"]["head"]["excludes_1"] and not corner["body_part_pooled"]["non_head"]["excludes_1"]
    assert 0.95 < flank["headline"]["obs_exp_direct"]["ci"][1] < 1
    assert flank["body_part_contrast"]["body_part_all_shots"]["covers_zero"]
    table = _REPO / "data" / "derived" / "frontier.parquet"
    if table.exists():
        import pandas as pd
        f = pd.read_parquet(table).query("set_piece_type == 'corner'").set_index(["block", "key"])
        thin = f.loc[("body_part", "after_first_touch|head")]
        assert thin["hi"] > 1 and thin["poisson_hi"] > 1
        assert f.loc[("body_part_pooled", "head"), "poisson_hi"] < 1


def _covers_zero(ci: list) -> bool:
    return ci[0] < 0 < ci[1]


def test_the_claims_of_section_4_are_the_record_s():
    """Section 4 says in words what its tables and figure hold, and each word is the record's. On corners the delivery
    leads the whole 360 block in the normal and the percentile form and misses zero by a hair in the pivotal form, while
    the frame's step covers zero; every group of features but the delivery covers zero (all eleven on the flank);
    permuting the delivery costs about ten times the 95th percentile of noise blocks as wide as it, and of the 360
    groups only marking and congestion cost more than every noise block of their width, while dropping them covers zero;
    the lead holds without short corners and at four windows of the cross, the first of eight seconds; the frame's bound is far below the
    smallest detectable effect and inside the margin in normal form and outside it in pivotal form; the control sees
    signal on corners (long ones too, a shuffled outcome gaining nothing, the frame adding nothing) and none on the
    flank; the gain is predictable but has no resolved group; the aim is read in normal and pivotal form and not as
    percentiles, no league resolves it alone in its direction and Ligue 1 clears zero in the other, the map's discount
    is no discount on either side of the first contact, and the difference between the maps clears zero in pivotal
    form only; the attack wins seven second balls in ten and the frame at the kick covers zero on them in the normal and
    the percentile form and clears it in the pivotal, and at the first contact the first and the third result hold in
    every form and after correction while the second clears zero in all three forms, covers it after correction and
    is not stable across the seeds; on the flank none holds, the first closed by its strong-learner refit."""
    record = json.loads(_RECORD.read_text())["figures"]
    said = " ".join(_source().split())
    c, f = record["validation"]["corner"], record["validation"]["free_kick_flank"]
    assert _covers_zero(c["refit"]["normal_ci"]) and c["ladder"]["delivery"]["step_ci"][0] > 0
    vs = c["delivery_vs_360"]["refit"]
    assert vs["normal_ci"][0] > 0 and vs["refit_ci"][0] > 0 and -1e-4 < vs["adjusted_ci"][0] < 0
    assert "clears zero in the normal and the percentile form of the interval, while the pivotal form reaches down to" in said
    abl = c["ablation"]
    assert [k for k, d in abl.items() if not _covers_zero(d["normal_ci"])] == ["delivery"]
    assert c["n_groups_covering_zero"] == c["n_ablation_groups"] - 1 and f["n_groups_covering_zero"] == f["n_ablation_groups"]
    assert _covers_zero(abl["congestion"]["normal_ci"]) and _covers_zero(f["ladder"]["delivery"]["step_ci"])
    assert "The exception is the delivery, which carries the model." in said
    nf, perm = record["noise_floor"]["by_group"], c["permuted"]
    assert 9 < perm["delivery"]["delta"] / nf["delivery"]["p95"] < 11 and "about ten times that" in said
    over = sorted(g for g, d in nf.items() if g != "delivery" and round(perm[g]["delta"], 4) > round(d["max"], 4))
    assert over == ["congestion", "marking"] and all(_covers_zero(abl[g]["normal_ci"]) for g in over)
    assert "only marking and congestion cost more than every block of their width" in said
    arms = record["short_corner_sensitivity"]["by_arm"]["fixed_C"]
    assert arms["long_only"]["excludes_zero"] and arms["long_only"]["delta"] < arms["all"]["delta"]
    crosses = {a for a in arms if a.startswith("short_cross")}
    assert crosses == {"short_cross_5s", "short_cross_8s", "short_cross_10s", "short_cross_15s"}
    assert all(arms[a]["delivery_wins"] for a in crosses) and record["short_corner_sensitivity"]["short_cross"]["window_s"] == 8.0
    assert arms["kick_anchored"]["delta"] > 3 * arms["all"]["delta"]
    assert "five, ten and fifteen seconds" in said and "so the crosses do not carry it" in said
    ni = c["non_inferiority"]
    assert c["refit"]["normal_ci"][1] < ni["m2"] < c["refit"]["adjusted_ci"][1]
    assert c["refit"]["normal_ci"][1] < c["power"]["mde_80_adjusted"] and ni["m2"] == pytest.approx(ni["m1"] / 10, rel=0.01)
    het = c["heterogeneity"]
    assert het["blp_p"] < 0.025 and het["placebo_clean"] and het["gates_p"] > 0.025 and not het["actionable"]
    assert f["heterogeneity"]["verdict"] == "no detectable structure"
    pc = record["positive_control"]
    dec, long = pc["decisions"], pc["long_corners_only"]
    assert dec["delivery-prior"]["below_zero"] and long["delivery-prior"]["below_zero"]
    assert abs(long["delivery-prior"]["delta"]) < abs(dec["delivery-prior"]["delta"])
    assert pc["shuffled_controls"]["shuffled_global"]["delta"] > 0
    assert _covers_zero(dec["full_360-delivery"]["ci"]) and _covers_zero(long["full_360-delivery"]["ci"])
    fpc = record["free_kick_flank"]["positive_control"]["decisions"]
    assert not any(fpc[k]["below_zero"] for k in ("delivery-prior", "full_360-prior"))
    aim = record["aim_zone"]["confirmation"]
    d1, d2, naive, honest = (aim["statistics"][k] for k in ("D1", "D2", "naive", "honest"))
    assert d1["normal"][1] < 0 and d1["pivotal"][1] < 0 and d1["percentile"][1] > 0
    assert record["aim_zone"]["league_in_design"]["D1"]["fixed"][1] < 0
    league = record["aim_zone"]["by_league"]
    assert not any(league[k]["D1"]["fixed"][1] < 0 for k in ("bundesliga", "la_liga", "ligue_1", "serie_a"))
    assert league["ligue_1"]["D1"]["fixed"][0] > 0
    assert d2["pivotal"][1] < 0 < d2["normal"][1] and naive["point"] < 0
    assert _covers_zero(honest["normal"]) and _covers_zero(honest["pivotal"])
    assert all(_covers_zero(aim["by_first_contact"][s]["naive"]["fixed"]) for s in ("attack_won", "defence_first"))
    assert record["free_kick_flank"]["aim_zone"]["verdicts"] == {"A1": "false", "A2": "false"}
    suf = record["sufficiency"]["grant"]
    assert round(10 * suf["base_rate"]) == 7 and "always naming the attack would be right seven times in ten" in said
    ref = suf["refit"]["geometry_over_delivery"]
    assert _covers_zero(ref["normal_ci"]) and _covers_zero(ref["refit_ci"]) and not _covers_zero(ref["adjusted_ci"])
    assert "covers zero in the normal and the percentile form and clears it in the pivotal form" in said
    assert suf["delivery_over_prior"]["covers_zero"] and suf["strong_learner_over_delivery"]["covers_zero"]
    fc = record["first_contact_frame"]
    assert fc["coverage"]["grant"] == suf["corners"] and fc["coverage"]["t1_is_clear"] < fc["coverage"]["grant"]
    d = fc["decisions"]
    wins = ("geometry_1-delivery", "geometry_1-geometry360_D", "facts_geometry_1-facts_1")
    assert all(d[k][form][1] < 0 for k in wins for form in ("normal", "pivotal", "percentile"))
    assert d[wins[0]]["stable_across_seeds"] and d[wins[2]]["stable_across_seeds"] and not d[wins[1]]["stable_across_seeds"]
    assert d[wins[0]]["bonferroni"][1] < 0 and d[wins[2]]["bonferroni"][1] < 0 < d[wins[1]]["bonferroni"][1]
    rule = fc["verdict_rule"]
    assert "pivotal" in rule and "decisive bound keeps its side" in rule and "Bonferroni never decides" in rule
    assert sum(b > 0 for b in fc["stability_upper_bounds"][wins[1]]) == 1
    assert "requires that the pivotal interval leave zero out and that its decisive bound keep its side in each of five repetitions" in said
    assert "in one of the five repetitions its pivotal bound crosses zero, so the rule leaves that comparison undecided" in said
    assert "its decisive bound keeps its side in every repetition" in said
    assert all(len(v) == 5 for v in fc["stability_upper_bounds"].values())
    assert all(fc["no_taker_reading"][k]["below_zero"] for k in wins)
    assert abs(d[wins[0]]["delta"]) < fc["mde_80_adjusted"][wins[0]]
    race = d["static_race_1-delivery"]
    assert not race["below_zero"] and _covers_zero(race["normal"]) and _covers_zero(race["pivotal"])
    flank_fc = record["free_kick_flank"]["first_contact_frame"]
    assert flank_fc["phase_two"] and flank_fc["verdicts"] == {"V1": "false", "V2": "false", "V3": "false"}
    assert _covers_zero(flank_fc["decisions"]["gbt_1-delivery"]["pivotal"])
    assert round(record["first_contact"]["median_time_to_first_contact_s"], 1) == 1.6


def test_the_claims_of_sections_5_to_8_and_the_appendix_are_the_record_s():
    """Sections 5 to 8 and the appendix say in words what their tables hold, and each word is the record's. A player
    is staked within ten yards of the drop, most second balls the attack wins are rescues, a planned win is worth more
    than a rescue raw on both populations and no more in cells of the pitch of four sizes or by regression, and a
    staked drop that is lost leaves less xG, clear of zero on corners and only in its direction on the flank; the
    defender at the edge of the box and the counter are not estimable; Arsenal shrinks, its own effect includes zero
    and it is not told from Tottenham, whom the model leans to; one season is below the poor band and the control is
    moderate; the top two win more corners and their gap after the first touch covers zero and reaches past the
    margin, up to about a third of the rest's rate; the 360 exists for the fewest corners in the Premier League
    and the most in Serie A; four seasons lift the reliability to 0.62 and six to 0.71; the eleven cells are G2's,
    Opta Analyst's grid with its central row split at fifteen yards into the edge, which is the frontal; the refit's
    symmetry holds on every contrast; the key-pass annotation does not favour the shots that scored and the segments
    separate on the time since the delivery; the flank behaves differently below x = 85. The tracked
    second balls of Section 5.1 have tests of their own."""
    sys.path.insert(0, str(_REPO / "src"))
    import constants as C
    text = " ".join(_source().split())
    record = json.loads(_RECORD.read_text())["figures"]
    sit, drop, flank = record["situation"], record["drop_position"], record["free_kick_flank"]
    states = sit["by_state"]
    assert C.STAKE_RADIUS == 10 and "within ten yards of where that touch met the ball" in text
    assert sum(s["n"] for s in states.values()) == sit["four_state_known"] == record["sufficiency"]["grant"]["corners"]
    assert states["rescued"]["n"] > record["sufficiency"]["grant"]["positive"] / 2
    assert sit["contrasts"]["won_as_planned-rescued"]["ci"][0] > 0 and sit["contrasts"]["staked_but_lost-conceded_open"]["ci"][1] < 0
    cells = drop["by_cell"]
    assert len(cells) == 4 and "6yd" in cells and all(c["stratified"]["covers_zero"] for c in cells.values())
    assert _covers_zero(cells["6yd"]["adjusted"]["ci"]) and drop["no_adjusted_reading_keeps_the_premium"]
    assert drop["every_adjusted_reading_covers_zero"] and drop["adjusted_readings_inverting_it"] == 0
    assert states["staked_but_lost"]["xg"] < states["conceded_open"]["xg"]
    fs = flank["situation"]
    lost = fs["contrasts"]["staked_but_lost-conceded_open"]
    assert lost["delta"] < 0 and _covers_zero(lost["ci"]) and fs["contrasts"]["won_as_planned-rescued"]["ci"][0] > 0
    assert fs["by_state"]["staked_but_lost"]["xg"] < fs["by_state"]["conceded_open"]["xg"]
    assert flank["drop_position"]["every_adjusted_reading_covers_zero"]
    causal, counter = record["causal"], record["counter"]
    spec = causal["spec_curve"]
    assert spec["n_specs"] == 9 and all(t["n_excluding_zero"] == 0 for t in spec["by_target"].values())
    assert "none of the nine specifications tried leaves zero out" in text
    assert causal["by_spec"]["primary"]["frac_extreme_pct"] > 50
    assert "57% of the corners defended with no player designated" in text and "40% of those defended with one" in text
    assert not counter["any_window_clears_the_bar"] and counter["by_window"]["8s"]["events"] < C.MIN_CELL
    later = [counter["by_window"][w]["possession_contiguity_pct"] for w in ("12s", "15s", "20s", "30s", "45s", "60s", "90s")]
    assert later == sorted(later, reverse=True)
    shrink, rep, top = record["team_shrinkage"], record["repeatability"], record["top_teams"]
    arsenal, spurs = shrink["cases"]["Arsenal"], shrink["cases"]["Tottenham Hotspur"]
    assert arsenal["a_team_hdi"][0] < 0 < arsenal["a_team_hdi"][1] and shrink["p_first_above_second"] < 0.5
    assert arsenal["rank_median"] < record["teams"]["n_teams"] / 2 < arsenal["rank_95"][1]
    assert shrink["verdicts"]["V2_case_distinguishable"] == "False" and spurs["raw_rate"] > arsenal["raw_rate"] > arsenal["posterior_rate"]
    spread = rep["dispersion_aft_goals_for/corners_for"]
    assert not spread["excess_by_league"] and 0.025 < spread["p_value_by_league"] < 0.05
    assert rep["aft_goals_per100"]["full_ci"][1] < 0.5 < rep["corners_per_match"]["full_r_spearman_brown"] < 0.75
    rates, gap, dec = top["rates"], top["primary"]["T2"], top["decisions"]
    assert rates["T2"]["corners_per_match"] > rates["rest"]["corners_per_match"] and not dec["T1"] and not dec["T2"]
    assert gap["ci95"][0] < 0 < gap["ci95"][1] and dec["margin"] < gap["ci95"][1]
    assert dec["margin"] == pytest.approx(dec["pooled_rate"] / 4, rel=0.02)
    assert 0.30 < gap["ci95"][1] / rates["rest"]["aft_xg_per_corner"] < 0.34
    assert "a gap of up to about a third of the rest's rate is not excluded" in text
    for fact in ("p-value was 0.86", "3,620 LaLiga corners", "only 23.75%", "more than 22,000 matches"):
        assert fact in text, fact
    cover = {k: v["delivery_360_pct"] for k, v in record["coverage_by_league"].items() if isinstance(v, dict)}
    assert min(cover, key=cover.get) == "premier_league" and max(cover, key=cover.get) == "serie_a"
    named = record["teams"]["named"]
    assert named["Arsenal"]["goals_for_all"] > _PREVIOUS_RECORD < named["Tottenham Hotspur"]["goals_for_all"]
    assert all(named[t]["aft_goals_for_80pct_power"] > named[t]["goals_for_all"] for t in ("Arsenal", "Tottenham Hotspur"))
    assert ("The two scored {{teams.named.Arsenal.goals_for_all|int}} and {{teams.named.Tottenham Hotspur.goals_for_all|int}}"
            " in the season that broke the record") in text
    r = rep["aft_goals_per100"]["full_r_spearman_brown"]
    assert [round(k * r / (1 + (k - 1) * r), 2) for k in (4, 6)] == [0.62, 0.71]
    assert "four seasons of this volume would lift" in text and "to about 0.62, and six to about 0.71" in text
    from zones import FRONTAL, ZONES
    assert len(ZONES["G2"]) == 11 and FRONTAL["G2"] == "edge" and C.G2_EDGE_CUT == 15 and C.ZONE_PARTITION == "G2"
    assert "The eleven cells of Section 2.3" in text and "Here that cell is split at fifteen yards from goal" in text
    agree = {(pop, name): c["verdict_agrees"] for pop, cs in record["refit_readings"].items()
             if isinstance(cs, dict) for name, c in cs.items()}
    assert agree and all(agree.values())
    sep = record["falsification"]["separability"]
    dens = record["falsification"]["density"]["rate_diff"]
    assert not dens["covers_zero"] and dens["ci"][1] < 0
    assert "its interval lies below zero, so the annotation does not favour the shots that scored" in text
    split = (sep["flick_on"]["point"], sep["prolongation"]["point"])
    assert 0.9 < min(split) and max(split) < sep["after_clearance"]["point"]
    kp = record["falsification"]["kp_vs_segment"]
    assert kp["after_clearance|after_clearance"]["n"] > 0
    assert all(v["n"] == 0 for k, v in kp.items() if "after_clearance" in k and k != "after_clearance|after_clearance")
    het = record["validation"]["corner"]["heterogeneity"]
    assert het["blp_p"] < 0.025 and het["placebo_clean"] and het["placebo"]["global"]["blp_p"] > 0.025
    cut = record["flank_cut"]["cuts"]["85"]
    assert cut["won_fc"]["ci"][0] > 0 and cut["p_prol"]["ci"][0] > 0 and cut["p_aft"]["ci"][1] < 0
    bands = record["flank_cut"]["bands"]
    from_the_cutoff = [b for b in bands if re.fullmatch(r"\d+-\d+", b) and int(b.split("-")[0]) >= C.FK_FLANK_MIN_X]
    assert from_the_cutoff and all(bands[b]["fc_x_median"] > C.BOX_X for b in from_the_cutoff)
    assert bands["60-65"]["fc_x_median"] < min(bands[b]["fc_x_median"] for b in from_the_cutoff)
    assert bands["60-65"]["aft_share"] > bands["long_corners"]["aft_share"]


def test_the_tracked_second_balls_are_the_record_s():
    """Section 5.1 says in words what Table 9 holds, and each word is the record's. The tracking runs behind the events
    in every tracked match; the side already nearest the drop at the kick won the most second balls, its interval clear
    of the other two cases' and of the share no relation would give, so the run after the clearance decides few; and
    the defenders started towards the drop before the attackers by a difference that covers zero."""
    text = " ".join(_source().split())
    record = json.loads(_RECORD.read_text())["figures"]
    clock, corner = record["plan_race"]["clock"], record["plan_race"]["corner"]
    assert clock["all_halves_behind"] and clock["n_matches"] == record["scope"]["tracked_matches"]
    assert "in every one of the" in text and "matches the tracking runs behind" in text
    shares, onset = corner["shares"], corner["onset"]
    assert corner["largest"] == "at_kick" and shares["race"]["point"] < shares["at_kick"]["point"]
    assert corner["no_relation"]["diff_excludes_zero"] and corner["no_relation"]["diff"]["point"] > 0
    assert "the run after the clearance decides few of these second balls" in text
    assert onset["diff"]["point"] < 0 and onset["covers_zero"]
    assert "a difference that covers zero, so neither side reacted sooner" in text


def test_the_tracked_matches_are_every_game_among_the_six_sides_that_finished_top():
    """Section 2.1 says the tracking covers the home and away games among the six sides that finished top of Serie A,
    and Sections 5.1 and 7 call them thirty matches between six top sides: the tracked matches are every home and away
    pairing of six clubs, and those six hold the first six places of the standings that eda_standings reconciles with
    the points the events give."""
    from itertools import permutations

    import pandas as pd
    table = _REPO / "data" / "derived" / "eda_standings.parquet"
    if not table.exists() or not (_REPO / "data" / "serie_a_sample" / "tracking").exists():
        pytest.skip("the tracking or the reconciled standings are missing")
    sys.path.insert(0, str(_REPO / "src"))
    import loader
    import loader_sb as L
    tracked = set(loader.tracked_matches())
    pairs = {(m.home_team, m.away_team) for m in L.list_matches(L.SERIE_A_SAMPLE).itertuples() if int(m.match_id) in tracked}
    clubs = {c for pair in pairs for c in pair}
    assert len(tracked) == 30 and len(clubs) == 6 and pairs == set(permutations(clubs, 2))
    standings = pd.read_parquet(table)
    assert clubs == set(standings.loc[(standings["league"] == "serie_a") & (standings["pos"] <= 6), "team"])
    text = " ".join(_source().split())
    assert "the home and away games among the six sides that finished top of the league" in text
    assert "thirty matches between six top sides" in text


def test_the_pages_break_around_tables_figures_and_captions(tmp_path):
    """A table is never split across pages and stays under its title, a figure keeps with the title under it, a display
    equation keeps with the sentence that introduces it, and a title that wraps does not open a gap between its lines,
    which the template's double-spaced caption style would."""
    import docx
    from docx.oxml.ns import qn
    from docx.text.paragraph import Paragraph
    if not rp._TEMPLATE.exists():
        pytest.skip("the Word template is missing")
    record = json.loads(_RECORD.read_text())["figures"]
    rp.build(_PAPER, record, rp._TEMPLATE, tmp_path / "paper.docx")
    d = docx.Document(str(tmp_path / "paper.docx"))
    body = [c for c in d.element.body.iterchildren() if c.tag in (qn("w:p"), qn("w:tbl"))]
    keep = f"{qn('w:pPr')}/{qn('w:keepNext')}"
    n_tables = n_figures = n_equations = 0
    for k, el in enumerate(body):
        if el.tag == qn("w:tbl"):
            rows = el.findall(qn("w:tr"))
            assert all(r.find(f"{qn('w:trPr')}/{qn('w:cantSplit')}") is not None for r in rows), "a row may break"
            for r in rows[:-1]:
                assert all(p.find(keep) is not None for p in r.iter(qn("w:p"))), "a row does not keep with the next"
            if el.find(f".//{qn('w:drawing')}") is not None:
                n_equations += 1
                assert body[k - 1].find(keep) is not None, "an equation parts from the sentence that introduces it"
            else:
                n_tables += 1
                assert rows[0].find(f"{qn('w:trPr')}/{qn('w:tblHeader')}") is not None
                title = Paragraph(body[k - 1], d)
                assert title.style.name == "Heading 3" and re.match(r"Table [A-Z]?\d+:", title.text), title.text
        elif el.find(f".//{qn('w:drawing')}") is not None:
            n_figures += 1
            assert el.find(keep) is not None, "a figure does not keep with its title"
            assert Paragraph(body[k + 1], d).style.name == "Heading 3" and Paragraph(body[k + 1], d).text.startswith("Figure ")
    titles = [Paragraph(el, d) for el in body if el.tag == qn("w:p") and Paragraph(el, d).style.name == "Heading 3"]
    assert titles and all(t.paragraph_format.line_spacing == 1.0 for t in titles)
    assert n_tables and n_figures and n_equations, "the test saw no table, figure or equation"


def test_every_picture_sits_flush_with_the_text(tmp_path):
    """A picture that names no wrap distance gets its viewer's own, and LibreOffice adds about 9 pt on each side: a figure as
    wide as the text then starts that far right of its margin and its caption. Every inline picture names them, at zero."""
    import zipfile

    from lxml import etree
    if not rp._TEMPLATE.exists():
        pytest.skip("the Word template is missing")
    rp.build(_PAPER, json.loads(_RECORD.read_text())["figures"], rp._TEMPLATE, tmp_path / "paper.docx")
    wp = "{http://schemas.openxmlformats.org/drawingml/2006/wordprocessingDrawing}"
    root = etree.fromstring(zipfile.ZipFile(tmp_path / "paper.docx").read("word/document.xml"))
    inlines = list(root.iter(wp + "inline"))
    assert inlines, "the test saw no picture"
    for el in inlines:
        assert [el.get(side) for side in ("distT", "distB", "distL", "distR")] == ["0"] * 4


def test_an_appendix_named_in_the_text_is_a_link_to_its_heading(tmp_path):
    """Each "Appendix X" of the body is a link inside the document to the heading of that appendix, in the order the text
    names them, and none is left as plain text."""
    import zipfile

    from lxml import etree
    if not rp._TEMPLATE.exists():
        pytest.skip("the Word template is missing")
    rp.build(_PAPER, json.loads(_RECORD.read_text())["figures"], rp._TEMPLATE, tmp_path / "paper.docx")
    w = "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}"
    root = etree.fromstring(zipfile.ZipFile(tmp_path / "paper.docx").read("word/document.xml"))
    body = rp.blocks(_PAPER.read_text())[0]
    texts = [c for k, c in body if k in ("para", "item")] + [c[0] for k, c in body if k == "figure"] \
        + [cell for k, c in body if k == "table" for row in c for cell in row]
    named = [m for t in texts for m in rp.APPENDIX.findall(t)]
    assert named, "the paper names no appendix"
    anchors = [h.get(w + "anchor") for h in root.iter(w + "hyperlink") if h.get(w + "anchor")]
    assert sorted(anchors) == sorted(f"appendix_{m}" for m in named)
    bookmarks = {b.get(w + "name") for b in root.iter(w + "bookmarkStart")}
    assert set(anchors) <= bookmarks and {f"appendix_{x}" for x in "ABCD"} <= bookmarks
    plain = [t.text for t in root.iter(w + "t") if rp.APPENDIX.search(t.text or "")
             and not any(a.tag == w + "hyperlink" for a in t.iterancestors())]
    assert not plain, plain


def test_the_reference_list_is_set_small_and_the_logo_keeps_clear_of_the_text(tmp_path):
    """The references are set at 10 pt on single spacing, links included, which is what keeps the paper at thirty pages, and
    the template's logo sits 0.3 in from the top edge so that the first line of a page is not under its foot."""
    import docx
    from docx.oxml.ns import qn
    if not rp._TEMPLATE.exists():
        pytest.skip("the Word template is missing")
    rp.build(_PAPER, json.loads(_RECORD.read_text())["figures"], rp._TEMPLATE, tmp_path / "paper.docx")
    d = docx.Document(str(tmp_path / "paper.docx"))
    ps = d.paragraphs
    start = next(k for k, p in enumerate(ps) if p.style.name == "Heading 1" and p.text == "References")
    end = next(k for k, p in enumerate(ps) if p.style.name == "Heading 1" and p.text == "Appendix")
    refs = [p for p in ps[start + 1:end] if re.match(r"\[\d+\] ", p.text)]
    assert len(refs) > 30
    for p in refs:
        assert p.paragraph_format.line_spacing == 1.0, p.text[:30]
        sizes = {r.find(f"{qn('w:rPr')}/{qn('w:sz')}").get(qn("w:val")) for r in p._p.iter(qn("w:r"))}
        assert sizes == {str(2 * rp._REF_PT)}, (p.text[:30], sizes)
    assert d.sections[0].header_distance.inches == pytest.approx(rp._HEADER_DISTANCE, abs=0.001)


def test_a_numbered_list_has_air_between_its_items_and_keeps_its_introduction(tmp_path):
    """Each item of a numbered list is a paragraph with air under it, and the sentence that introduces a list, with the blank
    line before the first item, keeps with it, so that the sentence is never alone at the foot of a page."""
    import docx
    if not rp._TEMPLATE.exists():
        pytest.skip("the Word template is missing")
    rp.build(_PAPER, json.loads(_RECORD.read_text())["figures"], rp._TEMPLATE, tmp_path / "paper.docx")
    ps = docx.Document(str(tmp_path / "paper.docx")).paragraphs
    items = [k for k, p in enumerate(ps) if re.match(r"(\d+|H\d)\. ", p.text) and p.style.name == "normal"]
    assert len(items) == sum(kind == "item" for kind, _ in rp.blocks(_PAPER.read_text())[0])
    for k in items:
        assert ps[k].paragraph_format.space_after.pt == rp._ITEM_AFTER, ps[k].text[:30]
    opened = [k for k in items if re.match(r"(1|H1)\. ", ps[k].text)]
    assert len(opened) >= 5
    for k in opened:
        j = k - 1
        while not ps[j].text:                                   # the blank line before the first item
            assert ps[j].paragraph_format.keep_with_next, "a blank line parts the list from its sentence"
            j -= 1
        assert ps[j].paragraph_format.keep_with_next, ps[j].text[:40]


def test_a_table_sets_its_text_in_capitals_and_centres_every_cell(tmp_path):
    """Every cell of a data table that opens with a letter opens with a capital, headers and labels alike, and every cell is
    centred, across and down, the first column's too."""
    import docx
    from docx.enum.table import WD_ALIGN_VERTICAL
    from docx.enum.text import WD_ALIGN_PARAGRAPH
    if not rp._TEMPLATE.exists():
        pytest.skip("the Word template is missing")
    rp.build(_PAPER, json.loads(_RECORD.read_text())["figures"], rp._TEMPLATE, tmp_path / "paper.docx")
    d = docx.Document(str(tmp_path / "paper.docx"))
    data = [t for t in d.tables if not re.fullmatch(r"\(\d+\)", t.rows[0].cells[-1].text)]      # an equation sits in a table too
    assert data
    for t in data:
        for row in t.rows:
            for cell in row.cells:
                assert not cell.text[:1].islower(), cell.text
                assert all(p.alignment == WD_ALIGN_PARAGRAPH.CENTER for p in cell.paragraphs), (row.cells[0].text, cell.text)
                assert cell.vertical_alignment == WD_ALIGN_VERTICAL.CENTER, (row.cells[0].text, cell.text)
