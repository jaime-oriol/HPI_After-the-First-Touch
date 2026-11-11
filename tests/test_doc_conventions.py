"""Notation conventions, enforced instead of remembered.

A figure typed into prose goes stale, so `test_docs_figures` binds every quoted literal to the
generated record. This file covers the class underneath it. A figure can be perfectly current
and still be misread because it is written in the wrong convention, and that failure is
invisible to a value check.

The failure is concrete. "6,945 s" beside "9,627" for nine thousand six hundred makes the same
comma a decimal point in one line and a thousands separator in the next, and "1.751 matches"
beside "0.853" for a ratio does the same with the dot. Each reads as a plausible number and is
off by three orders of magnitude, and a value check passes it.

Scope note: the files in `_EXCLUDED` are skipped everywhere here, each for its own reason.
`measured.md` is generated, so checking its notation would be checking the generator's;
`to-do.md` is the author's own working roadmap, written in his working language and read only by
him; and `decisiones_cerradas.md` is the register of decisions already taken, kept apart from
the working rules so a session need not pay for it in context, and in the working language for
the same reason `to-do.md` is. The last entry is personal reusable material, never project
context. None of them is delivered documentation, which is what these conventions protect.
"""
import re
from pathlib import Path

import pytest

_REPO = Path(__file__).resolve().parents[1]
_EXCLUDED = {"prompts.md", "measured.md", "to-do.md", "decisiones_cerradas.md"}
_DOCS = sorted(
    p for p in (_REPO / "docs").glob("*.md") if p.name not in _EXCLUDED
) + [_REPO / "README.md"]
_PY = sorted((_REPO / "src").rglob("*.py")) + sorted((_REPO / "pipeline").glob("*.py")) + \
    sorted((_REPO / "tests").glob("*.py")) + sorted((_REPO / "renders").glob("*.py"))

_FENCE = re.compile(r"^\s*```")
_TREE = re.compile(r"[│├└─]")                  # the repo-layout diagrams in architecture.md
_HTML_COMMENT = re.compile(r"<!--.*?-->")


def _prose_lines(path: Path):
    """(lineno, text) for prose only: fenced code, ASCII tree diagrams and HTML comments out.

    A tree diagram and a shell command are monospace by nature, so ASCII arrows belong there.
    The conventions below are about what a reader reads, not what a terminal prints. HTML
    comments go because their own terminator ENDS in `->`, which the arrow rule would report
    as a violation on every annotated table row."""
    inside = False
    for n, line in enumerate(path.read_text().splitlines(), 1):
        if _FENCE.match(line):
            inside = not inside
            continue
        if inside or _TREE.search(line):
            continue
        yield n, _HTML_COMMENT.sub("", line)


def _ids(paths):
    return [p.name for p in paths]


# One predicate per convention, kept as pure functions on a single line so the guards can be
# fired at synthetic text. A rule that only ever runs over files that already satisfy it is
# untested by construction, and this file exists because of a failure nothing noticed.
_NOUNS = r"(matches|shots|corners|deliveries|events|goals|rows|frames|clips|players)"

RULES = {
    # Two clauses, and the second is the one that matters. A comma before one or two digits
    # cannot be a thousands separator, which needs exactly three. But "6,945 s" has three, so
    # the first clause misses the very case this file was written for; what separates it from
    # "9,627 corners" is that a UNIT follows rather than a noun.
    "comma-decimal": lambda s: (re.search(r"\d,\d(?!\d\d)", s) is not None
                                or re.search(r"\d,\d{3}\*{0,2}\s*(s|%|yd|m|pp|nats)\b", s)
                                is not None),
    # 1.751 and 0.853 are indistinguishable as strings, so this keys on the noun after it:
    # a count of matches is never a three-decimal fraction.
    "dot-thousands": lambda s: re.search(r"\b\d{1,3}\.\d{3}\s+\*{0,2}" + _NOUNS, s) is not None,
    # 38 uses of → against a handful of -> is a drift, not a choice.
    "ascii-arrow": lambda s: "->" in s,
}


@pytest.mark.parametrize("path", _DOCS, ids=_ids(_DOCS))
@pytest.mark.parametrize("rule", sorted(RULES))
def test_documents_follow_the_notation_conventions(rule, path):
    bad = [(n, line.strip()) for n, line in _prose_lines(path) if RULES[rule](line)]
    assert not bad, f"{path.name}: {rule} at {[n for n, _ in bad]}\n" + \
                    "\n".join(f"  {n}: {t}" for n, t in bad[:5])


@pytest.mark.parametrize("rule,offender,innocent", [
    ("comma-decimal", "RMST fell to 6,945 s", "the grant holds 9,627 cleared corners"),
    ("dot-thousands", "the full grant of 1.751 matches", "obs/exp was 0.853 on corners"),
    ("ascii-arrow", "log-loss 1.4415 -> 1.4413", "log-loss 1.4415 → 1.4413"),
])
def test_each_rule_actually_fires(rule, offender, innocent):
    """The guards bite: each offender is the misreading its rule exists for, and each innocent is
    correct prose the rule must let through."""
    assert RULES[rule](offender), f"{rule} missed a known offender"
    assert not RULES[rule](innocent), f"{rule} would reject correct prose"


@pytest.mark.parametrize("path", _PY, ids=_ids(_PY))
def test_comment_separators_are_the_box_drawing_rule(path):
    """The house rule: section separators are U+2500, never runs of ASCII dashes.

    The tree holds it, and this keeps it that way: a convention is cheapest to lock while
    nothing violates it."""
    bad = [n for n, line in enumerate(path.read_text().splitlines(), 1)
           if re.search(r"#.*(-{4,}|={4,}|~{4,})", line)]
    assert not bad, f"{path.name}: ASCII separators at {bad}"


# ── Language ────────────────────────────────────────────────────────────


def test_the_project_documentation_is_in_english():
    """The working notes are written in Spanish and are gitignored. Everything a reader outside
    this laptop can see is English, so a Spanish sentence in a delivered doc means a paragraph
    was drafted in the working language and never translated.

    No document directly under docs/ is exempt. The files `_EXCLUDED` skips above are a
    different rule, listed there with their own reasons."""
    spanish = re.compile(r"\b(el|la|los|las|una|pero|porque|desde|hasta|entre|sobre|"
                         r"cuando|donde|siempre|nunca|también|además)\b", re.I)
    offenders = {}
    for path in _DOCS:
        hits = [n for n, line in _prose_lines(path)
                if len(spanish.findall(line)) >= 3]       # 3+, so "la liga" and "el clasico" pass
        if hits:
            offenders[path.name] = hits
    assert not offenders, f"Spanish prose in delivered documentation: {offenders}"
