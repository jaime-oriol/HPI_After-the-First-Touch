"""zones — the plan's partition of the box, the one switch between the partitions, and the flip.

Three partitions exist. C, s02b's twelve count cells, reads the aim better than F on the registered
rule M1, and F, the six zones a coach names, is the easier to act on (data/derived/prereg_2026-10-01/zones.md).
G2 is Opta Analyst's cut of the box as the author set it, eleven cells, the same for both sides
(data/derived/prereg_2026-10-13/zones_g.md), and the paper's: the default for the cut a club reads, not for a
better reading, since its screen found no gain over C (data/derived/prereg_2026-10-13/deviation_2026-10-25.md).
`constants.ZONE_PARTITION` is the switch and HUDL_ZONES sets it for one run. A consumer reads the zones
through `zone_of`, `zone_names` and `zone_cols`, which follow it, and never names a partition:
tests/test_zones.py refuses one that does. The corner table carries every partition's counts
(`table_counts`), so reading another partition's zones costs no rebuild.

ONE FRAME. A point is labelled in the canonical frame, attack towards x = 120 and the kick struck
from the attackers' left flag, y = 0. A delivery from y >= 40, s01's `corner_side == "right"`, is
mirrored first, so the near post is the same post on every corner and on every flank free kick;
`kicked_from_right` is the one place that reads the side and `canonical_y` the one that mirrors.

THE THREE JOBS of the commitment signature (form_answers.md Q2) are read off the frontal and the box: in
the frontal a body stakes the drop zone, elsewhere in the box it attacks the cross, and anywhere else it
holds back. Under C and F the frontal is the ground in front of the box inside its width, F's zone and C's
`edge` cell; under G2 it is G2's `edge`, which adds the central block of the box short of G2_EDGE_CUT. The
defenders in the frontal are s05's treatment, so they move with it too, and a table carries the jobs and the
treatment of the partition it was built under (`zone_partition`), which `require_built_under` holds to the
switch's.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from constants import (BOX_X, BOX_Y, G2_EDGE_CUT, GOAL_POSTS_SB, GOALMOUTH_X, HOLD_X, PITCH_X_SB,
                       PITCH_Y_MID_SB, PITCH_Y_SB, SIX_YARD_X, SIX_YARD_Y, ZONE_PARTITION)

if TYPE_CHECKING:
    import pandas as pd

_NEAR_Y, _FAR_Y = GOAL_POSTS_SB[0][1], GOAL_POSTS_SB[1][1]
_U0, _U1 = SIX_YARD_Y
_T1, _T2 = _U0 + (_U1 - _U0) / 3, _U0 + 2 * (_U1 - _U0) / 3    # Opta's thirds of the six-yard width, equal
_G2_SPOT_FRONT = PITCH_X_SB - G2_EDGE_CUT

ZONES = {"C": tuple(f"{d}_{b}" for d in ("six", "spot", "front") for b in ("near", "central", "far"))
         + ("wide", "edge", "deep"),
         "F": ("six_yard", "central", "near_post", "far_post", "frontal", "short", "rest"),
         "G2": ("six_near", "six_central", "six_far", "spot_near", "spot_central", "spot_far",
                "near_side", "far_side", "wide", "edge", "deep")}
FRONTAL = {"C": "edge", "F": "frontal", "G2": "edge"}
JOBS = ("attack_cross", "stake_drop", "hold_back")
# The partitions whose frontal is the same ground, so a table built under one is read under the other.
SAME_FRONTAL = {"C": {"C", "F"}, "F": {"C", "F"}, "G2": {"G2"}}
if ZONE_PARTITION not in ZONES:
    raise ValueError(f"HUDL_ZONES must be one of {sorted(ZONES)}, got {ZONE_PARTITION!r}")


# ── The flip ──────────────────────────────────────────────────────────────
def kicked_from_right(kick_y: np.ndarray | float) -> np.ndarray:
    """True where the delivery was struck from the attackers' right flag, y >= 40: s01's
    `corner_side` rule, which the swing confirms (test_s01_phases)."""
    return np.asarray(kick_y, dtype=float) >= PITCH_Y_MID_SB


def canonical_y(y: np.ndarray, right: np.ndarray | bool) -> np.ndarray:
    """y mirrored for a delivery from the attackers' right, so every one reads as struck from y = 0."""
    y = np.asarray(y, dtype=float)
    return np.where(right, PITCH_Y_SB - y, y)


# ── The lines of the box ──────────────────────────────────────────────────
def in_width(y: np.ndarray) -> np.ndarray:
    """Inside the width of the penalty box, the lines the jobs and every partition share."""
    return (y >= BOX_Y[0]) & (y <= BOX_Y[1])


def in_box(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Inside the penalty box: past its front line and inside its width."""
    return (x >= BOX_X) & in_width(y)


def frontal(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """The switch's frontal: in front of the box inside its width (F's frontal, C's `edge` and FIFA's
    edge player), and under G2 also the central block of the box short of G2_EDGE_CUT, its `edge`.
    Symmetric in y, so it needs no flip."""
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    strip = (x >= HOLD_X) & (x < BOX_X) & in_width(y)
    if ZONE_PARTITION != "G2":
        return strip
    return strip | (in_box(x, y) & (y >= _U0) & (y <= _U1) & (x < _G2_SPOT_FRONT))


def band(y: np.ndarray) -> np.ndarray:
    """Near, central or far of the two post lines."""
    return np.where(y < _NEAR_Y, "near", np.where(y < _FAR_Y, "central", "far"))


def depth(x: np.ndarray, inner: str) -> np.ndarray:
    """Past the six-yard line, between it and the penalty spot's line, or `inner` short of that."""
    return np.where(x >= SIX_YARD_X, "six", np.where(x >= GOALMOUTH_X, "spot", inner))


def job_of(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """The job of each attacker: in the frontal, elsewhere in the box, or neither. The frontal comes
    first because G2's reaches into the box. Symmetric in y, so it needs no flip."""
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    return np.where(frontal(x, y), JOBS[1],
                    np.where(in_box(x, y), JOBS[0], JOBS[2])).astype(object)


# ── The partitions, on canonical points ───────────────────────────────────
def cells(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """C: a box cell by depth and band; `edge` in front of the box inside its width; `wide` outside
    the width from HOLD_X forward; `deep` short of HOLD_X."""
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    box = np.char.add(np.char.add(depth(x, "front"), "_"), band(y))
    short = np.where(x < HOLD_X, "deep", np.where(in_width(y), "edge", "wide"))
    return np.where(in_box(x, y), box, short).astype(object)


def zones_f(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """F, first match wins: the six-yard box; central, between the post lines short of it; the rest
    of the box near and far of the posts; the frontal; short, outside the width on the kick side
    from the box edge on; anything else is `rest`."""
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    box = in_box(x, y)
    return np.select([(x >= SIX_YARD_X) & (y >= SIX_YARD_Y[0]) & (y <= SIX_YARD_Y[1]),
                      box & (x < SIX_YARD_X) & (y >= _NEAR_Y) & (y <= _FAR_Y),
                      box & (y < _NEAR_Y), box & (y > _FAR_Y), (x >= HOLD_X) & (x < BOX_X) & in_width(y),
                      (x >= BOX_X) & (y < BOX_Y[0])], ZONES["F"][:-1], "rest").astype(object)


def zones_g2(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """G2: inside the six-yard box's width, three equal columns over two rows, the six and the spot,
    the spot row running to G2_EDGE_CUT off the goal line and the central block short of it joining
    the `edge`; a cell on each side of that width over the box's depth; and C's `edge`, `wide` and
    `deep` outside the box."""
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    third = np.where(y < _T1, "near", np.where(y < _T2, "central", "far"))
    row = np.where(x >= SIX_YARD_X, "six", "spot")
    cell = np.where(x < _G2_SPOT_FRONT, "edge", np.char.add(np.char.add(row, "_"), third))
    inner = np.where((y >= _U0) & (y <= _U1), cell, np.where(y < _U0, "near_side", "far_side"))
    short = np.where(x < HOLD_X, "deep", np.where(in_width(y), "edge", "wide"))
    return np.where(in_box(x, y), inner, short).astype(object)


_LABEL = {"C": cells, "F": zones_f, "G2": zones_g2}


# ── What a consumer reads: the switch's partition ─────────────────────────
def zone_names() -> tuple[str, ...]:
    """The zones of the partition the switch names."""
    return ZONES[ZONE_PARTITION]


def zone_of(x: np.ndarray, y: np.ndarray, kick_y: np.ndarray | float) -> np.ndarray:
    """Each point's zone under the switch's partition, read in the canonical frame of its kick."""
    return _LABEL[ZONE_PARTITION](np.asarray(x, dtype=float),
                                  canonical_y(y, kicked_from_right(kick_y)))


def zone_cols(side: str) -> list[str]:
    """The corner table's count columns of one side under the switch. Defenders leave out the
    frontal, which `n_def_frontal` carries as s05's treatment and as a feature group of its own."""
    p = ZONE_PARTITION
    return [f"{p}_{side}_{z}" for z in ZONES[p] if not (side == "def" and z == FRONTAL[p])]


def table_columns() -> list[str]:
    """Every column `table_counts` writes: both sides per zone under every partition, the defenders'
    frontal included, and the defenders in the switch's frontal."""
    return [f"{p}_{s}_{z}" for p in ZONES for s in ("att", "def") for z in ZONES[p]] + ["n_def_frontal"]


def table_counts(xa: np.ndarray, ya: np.ndarray, xd: np.ndarray, yd: np.ndarray,
                 kick_y: float) -> dict[str, int]:
    """One delivery's attackers and defenders per zone under EVERY partition, and the defenders in the
    switch's frontal, in `table_columns` order. Coordinates as s02's frame holds them; the flip is here."""
    right = kicked_from_right(kick_y)
    out = {}
    for p, label in _LABEL.items():
        for side, x, y in (("att", xa, ya), ("def", xd, yd)):
            z = label(np.asarray(x, dtype=float), canonical_y(y, right))
            out.update({f"{p}_{side}_{k}": int((z == k).sum()) for k in ZONES[p]})
    out["n_def_frontal"] = int(frontal(np.asarray(xd, dtype=float), np.asarray(yd, dtype=float)).sum())
    return out


def require_built_under(table: pd.DataFrame) -> None:
    """Stop a read of a corner table whose jobs and defenders in the frontal are another frontal's than
    the switch's, by the partition it names in `zone_partition`. A table without the column passes here
    and carries no G2 column, which `s03._design` refuses under G2 as a column the table lacks."""
    if "zone_partition" not in table:
        return
    built = set(table["zone_partition"].dropna().unique())
    if not built <= SAME_FRONTAL[ZONE_PARTITION]:
        raise ValueError(f"a table built under {sorted(built)} is read under {ZONE_PARTITION}, whose "
                         f"frontal differs: rebuild it with HUDL_ZONES={ZONE_PARTITION}")
