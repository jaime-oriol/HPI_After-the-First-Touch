"""The domain constants have exactly one owner, and the tree is checked for it.

Every drift this module guards against is silent. A frame rate, a pitch or a publication floor
declared in several places drifts between the copies, and one identifier carrying two values
(a `WINDOW_S` meaning 6.0 s to one module and 8.0 s to another) changes a window with nothing
failing out loud.

These tests are structural: they parse the tree rather than importing it, so a redeclaration
is caught wherever it is written, including in a module no other test happens to import.
"""
import ast
import pathlib
from collections.abc import Iterator

import pytest

import constants

_REPO = pathlib.Path(__file__).resolve().parents[1]
# viz/ is deliberately excluded: its layout parameters are magic numbers whose effect cannot
# be read off the line without the drawing code, and the repo's own rule keeps them local. None
# of viz/'s module-level UPPER names is one this module owns. renders/ IS swept: it consumes
# constants like any other caller.
_SOURCES = (sorted((_REPO / "src").glob("*.py")) + sorted((_REPO / "pipeline").glob("*.py"))
            + sorted((_REPO / "renders").glob("*.py")))
# The owner is excluded from the redeclaration sweep by construction rather than skipped
# inside it: declaring these literals is its job, so a skip there would be a permanent
# green-but-not-run row. What the owner owes instead is checked positively, below.
_CONSUMERS = [p for p in _SOURCES if p.name != "constants.py"]
_OWNED = {n for n in vars(constants) if n.isupper()}


def _module_level_assignments(path: pathlib.Path) -> Iterator[tuple[str, ast.AST]]:
    """(name, value_node) for every module-level UPPER_CASE assignment in one file.

    A tuple target unpacks element by element against a tuple value of its length, so
    `A, B = 1, 2` reads as A = 1 and B = 2: read whole, every name in it would carry (1, 2) and
    two modules sharing a name and its value would clash on their neighbours."""
    for node in ast.parse(path.read_text()).body:
        if not isinstance(node, ast.Assign):
            continue
        for target in node.targets:
            names = [target] if isinstance(target, ast.Name) else getattr(target, "elts", [])
            values = (node.value.elts if isinstance(node.value, ast.Tuple)
                      and len(node.value.elts) == len(names) and len(names) > 1
                      else [node.value] * len(names))
            for name, value in zip(names, values, strict=True):
                if isinstance(name, ast.Name) and name.id.isupper():
                    yield name.id, value


def _assignment_lines(path: pathlib.Path) -> Iterator[tuple[str, int]]:
    """(name, lineno) for every module-level UPPER_CASE assignment in one file."""
    for node in ast.parse(path.read_text()).body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id.isupper():
                    yield target.id, target.lineno


def _literal(node: ast.AST) -> object:
    """The node's literal value, or _NOT_A_LITERAL when it is an expression."""
    try:
        return ast.literal_eval(node)
    except Exception:
        return _NOT_A_LITERAL


_NOT_A_LITERAL = object()


def test_the_owner_actually_declares_what_it_claims_to_own():
    """The other half of the rule. The sweep below proves nobody ELSE declares these; this
    proves the owner does, at module level and as literals or derivations of them.

    Without it the guard is vacuous in the one way that matters: move a constant out of
    constants.py into some consumer and every redeclaration test still passes, because
    _OWNED is read from the module at import time and would simply shrink with it.

    The count is exact rather than a floor, because a floor lets constants leave without a
    sound, which is the same hollowing-out one paragraph up, only slower."""
    declared = {name for name, _ in _module_level_assignments(_REPO / "src" / "constants.py")}
    missing = _OWNED - declared
    assert not missing, f"constants.py exports {sorted(missing)} without declaring them"
    assert len(_OWNED) == 52, (
        f"{len(_OWNED)} owned constants, expected 52. Losing one fails here. Adding one means "
        "adding it here too, deliberately.")


@pytest.mark.parametrize("path", _CONSUMERS, ids=lambda p: p.name)
def test_no_module_redeclares_a_constant_it_should_import(path):
    """A name constants.py owns may be imported or derived, never re-assigned to a literal.

    The failure this prevents is silent by construction: a local copy shadows the import,
    the module keeps working, and the two values only diverge on the next correction."""
    for name, value in _module_level_assignments(path):
        if name in _OWNED and _literal(value) is not _NOT_A_LITERAL:
            pytest.fail(f"{path.name} re-declares {name} as a literal; import it from constants")


def test_no_identifier_means_two_different_things():
    """The WINDOW_S trap: one name, two modules, two values, no error anywhere.

    Constants that merely COINCIDE in value are fine and several do (1.5 m against 1.5 s,
    6.0 s against 6.0 yd). What is barred is the reverse: one NAME carrying two values."""
    seen: dict = {}
    clashes = []
    for path in _SOURCES:
        for name, value in _module_level_assignments(path):
            v = _literal(value)
            if v is _NOT_A_LITERAL:
                continue
            if name in seen and seen[name][1] != v:
                clashes.append(f"{name}: {seen[name][0]}={seen[name][1]!r} vs {path.name}={v!r}")
            seen.setdefault(name, (path.name, v))
    assert not clashes, "same name, different value:\n  " + "\n  ".join(clashes)


def test_the_two_coordinate_spaces_stay_apart():
    """Metres and StatsBomb yards describe the same pitch and must never be interchanged."""
    assert (constants.PITCH_LENGTH_M, constants.PITCH_WIDTH_M) == (105.0, 68.0)
    assert (constants.PITCH_X_SB, constants.PITCH_Y_SB) == (120.0, 80.0)
    # derived, so a change to the pitch cannot leave the midline or the goal behind
    assert constants.HALF_X == constants.PITCH_X_SB / 2
    assert constants.PITCH_Y_MID_SB == constants.PITCH_Y_SB / 2
    assert constants.GOAL_SB == (constants.PITCH_X_SB, constants.PITCH_Y_SB / 2)


def test_the_two_windows_are_two_decisions():
    """They hold the same value and are not the same quantity: one bounds the episode
    from the FIRST CONTACT, the other bounds a counter from the CLEARANCE. Pinned so a future
    edit collapses them on purpose rather than by tidying."""
    assert constants.SECOND_BALL_WINDOW_S == 8.0
    assert constants.TRANSITION_WINDOW_S == 8.0
    # the race horizon is a THIRD thing again: it scans tracking from the delivery frame
    assert constants.RACE_SCAN_HORIZON_S == 6.0


def test_the_grammar_binds_rather_than_redeclaring():
    """s01 keeps the windows as class attributes because s08 and the drivers reach them that
    way, but the values come from constants."""
    from s01_phases import SetPieceGrammar
    assert SetPieceGrammar.SECOND_BALL_WINDOW_S is constants.SECOND_BALL_WINDOW_S
    assert SetPieceGrammar.TRANSITION_WINDOW_S is constants.TRANSITION_WINDOW_S
    assert SetPieceGrammar.SHORT_CROSS_WINDOW_S is constants.SHORT_CROSS_WINDOW_S


def test_the_cross_of_a_short_corner_has_the_bound_a_first_contact_has():
    """A short or ground corner is delivered by its first cross if that comes within the same
    seconds a first contact has after a long corner: the author's rule, one bound for the kick, so
    the two are bound to one value and cannot be moved apart by tidying."""
    assert constants.SHORT_CROSS_WINDOW_S is constants.SECOND_BALL_WINDOW_S


# The nine the module marks as CHOSEN, the two provider sentinels, the four the provider's own
# published source declares and the two set on a distribution measured over the grant
# (DROP_ANCHOR_SB and FK_FLANK_MIN_X), each against the figures its note has
# to keep. For most of them nothing report_figures reads persists the figures, so the comment IS
# the provenance; the situation table's radii record carries the three situation constants'
# (DROP_ANCHOR_SB, NEAR_RADII, STAKE_RADIUS), and the pin here still guards the note. Five of the
# nine record a distribution measured over the tracked sample and two one over the grant's
# situation frames; the two savgol parameters have no recorded origin at all, so their note
# records what each one does instead. The four the source declares are the opposite case and their
# note has to say so, because a cone geometry that was FITTED and one that was DECLARED support
# different words in the write-up: the fitted sweep plateaued at 95.0% and could only ever be
# reported, the declared one closes.
_NEEDS_PROVENANCE = {
    "NEAR_RADII": ("158,584", "4.75%", "10.13%"),
    "STAKE_RADIUS": ("6,244 corners", "p25 = 9.04", "median 15.42", "28.73%"),
    "CONTACT_RADIUS_M": ("22,518", "p25 = 1.28", "2.79"),
    "MIN_DELIVERY_M": ("174 races", "p1 = 7.1", "14.3"),
    "LOOSE_HORIZON_S": ("5.6 m", "p75 = 10.9"),
    "REACTION_SPEED_THR": ("0.50 m/s", "3,820"),
    "REACTION_MAX_S": ("27.3%",),
    "SAVGOL_WINDOW_FRAMES": ("280 ms", "3 frames"),
    "SAVGOL_POLYORDER": ("polyorder 1", "linear velocity"),
    "EMPTY_DEFENDER_DISTANCE_SB": ("freezeframeinfo.R", "195-198"),
    "EMPTY_DEFENDER_AREA_SB": ("freezeframeinfo.R", "line 171"),
    "CONE_POST_SLACK_SB": ("36-1 and 44+1", "defensiveinfo.R", "95.0%"),
    "CONE_APEX_BACKOFF_SB": ("DistToGoal + 1", ":69-72"),
    "DEFENSIVE_LINE_POSITION_IDS": ("position.id in 2:8", ":142-146"),
    "ZERO_DISTANCE_SUBSTITUTE_SB": ("a third of a yard", "43,940"),
    "DROP_ANCHOR_SB": ("9,627", "6.0 yd", "92.5-110.1"),
    # set on the flank free kicks by depth, outcomes read; and why it carries no length bar
    "FK_FLANK_MIN_X": ("2122", "+0.118 [+0.088, +0.148]", "2465", "34 of them"),
}


def _governing_note(lines: list[str], lineno: int) -> list[str]:
    """The comment block that governs the assignment on `lineno`, as a list of lines.

    Scans upward over blank lines and over the sibling assignments that share one block note,
    which is how the two sentinels are written. Anything else stops the scan, so deleting a
    note cannot be papered over by the previous block's."""
    out, i = [], lineno - 2
    while i >= 0:
        stripped = lines[i].strip()
        if stripped.startswith("#"):
            out.append(stripped)
        elif stripped and stripped.split("=")[0].strip() not in _NEEDS_PROVENANCE:
            break
        elif out:
            break
        i -= 1
    return out


@pytest.mark.parametrize("name", sorted(_NEEDS_PROVENANCE), ids=str)
def test_a_chosen_magnitude_keeps_the_figure_that_justifies_it(name):
    """Every other test here guards a constant's PROPERTY; this one guards its justification.
    Not one of these can be recovered from the code that uses it, so the comment is the
    only thing standing between the value and a guess, and a comment is the one part of a module
    nothing executes.

    It pins the figures verbatim rather than merely checking that a note is there, because
    checking for a note is vacuous: delete the one above CONTACT_RADIUS_M and the scan walks up
    to the section separator, which carries digits of its own, and the row goes green on the
    provenance of a different constant.

    Re-measuring a distribution fails here until this table is updated. That is deliberate.
    These figures live in exactly one place and nothing else in the tree guards them."""
    path = _REPO / "src" / "constants.py"
    lines = path.read_text().splitlines()
    lineno = next(ln for n, ln in _assignment_lines(path) if n == name)
    note = " ".join(reversed(_governing_note(lines, lineno)))
    missing = [f for f in _NEEDS_PROVENANCE[name] if f not in note]
    assert not missing, f"{name}'s note has lost {missing}; it reads {note!r}"
