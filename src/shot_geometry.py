"""shot_geometry — the geometry of `shot.freeze_frame`, defined once.

WHY THIS MODULE EXISTS. The goal-side predicate that recovers `density` has more than one
consumer, and a predicate that cannot be imported gets retyped, which is exactly how a magnitude
drifts between copies. One definition of goal-side, one of the keeper, one of the cone.

THE FRAME IS NOT THE 360 FRAME. `shot.freeze_frame` carries `location`, `teammate`,
`player.id/name` and `position.id/name` and has NO `keeper` or `actor` key; the 360
`freeze_frame` has `teammate`, `actor` and `keeper` and no position at all, which
`tests/test_schema_dumps.py` pins. So the keeper here is read by position
(`is_keeper`), and applying that rule to a 360 frame would silently yield no keeper
at all rather than failing.

Coordinates are StatsBomb yards throughout, 120 x 80, attacking towards x = 120, so goal-side of
the shot is `x >= x_shot`. The test is `>=` and not `>`: defenders level with the shooter count,
and the strict reading misses them.

TWO DEFINITIONS THAT LOOK INTERCHANGEABLE AND ARE NOT. `DefenderSet` keeps the non-goal-side
population as its own array rather than only the union, because `distance.ToD2`'s fallback is the
second-nearest defender BEHIND the shot, not the second-nearest overall. The two differ whenever
exactly one defender is goal-side, and the union reading loses agreement there
(hudl/data_provider.tex §sec:density).

NOTHING HERE IS A FITTED PARAMETER. The provider publishes the source behind this block, so the
cone is a declared triangle and the zero-distance substitution is a declared magnitude, both owned
by `constants` with their provenance. That is what lets the fields they recover be called verified
rather than reported. Following that source is not automatic either: it keeps the keeper in the
behind-the-shot population and measuring that costs agreement on `distance.ToD2`, so he stays
out. The agreement rates are in `hudl/data_provider.tex` §sec:density.

Both sentinels are declared by that same published source, so like the cone they are declared
rather than fitted: the provider writes them when the population it would have measured over is
empty. The two empty populations are not the same one. See `EMPTY_DEFENDER_DISTANCE_SB` in
`constants`, which carries the source lines.
"""

from __future__ import annotations

from typing import NamedTuple, Sequence

import numpy as np
from scipy.spatial.distance import pdist

from constants import (CONE_APEX_BACKOFF_SB, CONE_POST_SLACK_SB,
                       DEFENSIVE_LINE_POSITION_IDS, EMPTY_DEFENDER_AREA_SB,
                       EMPTY_DEFENDER_DISTANCE_SB, GOAL_POSTS_SB, GOAL_SB,
                       PITCH_X_SB, PITCH_Y_MID_SB, ZERO_DISTANCE_SUBSTITUTE_SB)


def is_keeper(p: dict) -> bool:
    """Whether a `shot.freeze_frame` entry is a goalkeeper, read by position."""
    return p.get("position.name") == "Goalkeeper"


class FrameArrays(NamedTuple):
    """One `shot.freeze_frame` as parallel arrays, n entries.

    `player_id` and `position_id` are float rather than int so an entry missing either can carry
    NaN instead of a sentinel that collides with a real id."""
    xy: np.ndarray                                     # (n, 2) StatsBomb yards
    teammate: np.ndarray
    keeper: np.ndarray
    player_id: np.ndarray
    position_id: np.ndarray


class DefenderSet(NamedTuple):
    """Distances from the shot to the outfield opponents, each array sorted ascending."""
    goal_side: np.ndarray
    behind: np.ndarray
    every: np.ndarray


def shot_xy(location: Sequence[float]) -> np.ndarray:
    """The strike location as (2,), dropping the z a shot event sometimes carries."""
    return np.asarray(location[:2], dtype=float)


def frame_arrays(frame: list) -> FrameArrays:
    """Unpack a shot freeze frame. `location` may be a list or an ndarray; the cache restores
    only the top level to list and consumers read the entries duck-typed."""
    xy = np.array([p["location"][:2] for p in frame], dtype=float)
    teammate = np.array([bool(p.get("teammate")) for p in frame])
    keeper = np.array([is_keeper(p) for p in frame])
    pid = np.array([float(p.get("player.id", np.nan) or np.nan) for p in frame])
    pos = np.array([float(p.get("position.id", np.nan) or np.nan) for p in frame])
    return FrameArrays(xy, teammate, keeper, pid, pos)


def _substitute_zero(d: np.ndarray) -> np.ndarray:
    """A body at zero distance enters at a third of a yard instead of being dropped.

    One definition for every inverse-distance sum in the tree: it is the whole of `density`'s
    residue, so a second copy that forgot it would still close almost exactly and look fine."""
    return np.where(d == 0.0, ZERO_DISTANCE_SUBSTITUTE_SB, d)


def body_distances(shot: np.ndarray, fa: FrameArrays) -> np.ndarray:
    """Distance from the shot to every body in the frame, (n,) in StatsBomb yards."""
    return _substitute_zero(np.hypot(fa.xy[:, 0] - shot[0], fa.xy[:, 1] - shot[1]))


def defender_distances(shot: np.ndarray, fa: FrameArrays) -> DefenderSet:
    """Split the outfield opponents by side of the shot and return their sorted distances.

    The population behind `density`, `distance.ToD1` and `distance.ToD2`, recovered by inversion
    because the export documents none of the three. The keeper never counts, on either side: he
    is excluded from the behind-the-shot fallback too, and the reading that keeps him there was
    measured and loses agreement on `ToD2`."""
    d = body_distances(shot, fa)
    outfield = ~fa.teammate & ~fa.keeper
    goal_side = outfield & (fa.xy[:, 0] >= shot[0])
    return DefenderSet(np.sort(d[goal_side]), np.sort(d[outfield & ~goal_side]),
                       np.sort(d[outfield]))


def signed_nth_defender(ds: DefenderSet, k: int) -> float:
    """The provider's `distance.ToD{k}`: the k-th goal-side defender, else the k-th behind the
    shot as a NEGATIVE number, else the empty-population sentinel. The sign is a flag saying no
    defender is goal-side, not a distance pointing backwards."""
    if len(ds.goal_side) >= k:
        return float(ds.goal_side[k - 1])
    if len(ds.behind) >= k:
        return -float(ds.behind[k - 1])
    return EMPTY_DEFENDER_DISTANCE_SB


def behind_ball_counts(shot: np.ndarray, fa: FrameArrays) -> tuple[int, int]:
    """(AttackersBehindBall, DefendersBehindBall): bodies goal-side of the strike.

    The attacking count includes the shooter, who trivially satisfies the predicate, so it
    reaches 11 while the defending count stops at 10 with the keeper excluded."""
    gs = fa.xy[:, 0] >= shot[0]
    return int((gs & fa.teammate).sum()), int((gs & ~fa.teammate & ~fa.keeper).sum())


def keeper_xy(fa: FrameArrays) -> np.ndarray | None:
    """The opposing keeper's position, or None when the frame does not show him."""
    m = fa.keeper & ~fa.teammate
    return fa.xy[m][0] if m.any() else None


def bearing_to_goal(xy: np.ndarray) -> float:
    """`AngleToGoal` in degrees: a BEARING onto the goal centre, atan2(dx, dy).

    Not the angle the goalmouth subtends at the point, which is what the field name suggests and
    what the measurement rules out. `AngleToKeeper` is the same bearing taken at the keeper rather
    than at the shot."""
    return float(np.degrees(np.arctan2(PITCH_X_SB - xy[0], PITCH_Y_MID_SB - xy[1])))


def shooting_cone(shot: np.ndarray) -> np.ndarray:
    """The provider's shooting cone as its three vertices, (3, 2) in StatsBomb yards.

    A triangle rather than an unbounded wedge, and neither of the two yards it is built from is
    fitted. Its base is the goal line between the posts widened by `CONE_POST_SLACK_SB` a side,
    and its apex sits `CONE_APEX_BACKOFF_SB` BEHIND the shot rather than on it, which is what
    admits the bodies standing level with the shooter."""
    goal = np.asarray(GOAL_SB, dtype=float)
    reach = float(np.hypot(*(shot - goal)))
    # Degenerate only for a shot on the goal centre itself, where "away from goal" is undefined.
    away = (shot - goal) / reach if reach > 0 else np.array([-1.0, 0.0])
    (gx, y_near), (_, y_far) = GOAL_POSTS_SB
    return np.array([[gx, y_near - CONE_POST_SLACK_SB],
                     [gx, y_far + CONE_POST_SLACK_SB],
                     shot + away * CONE_APEX_BACKOFF_SB])


def in_shooting_cone(xy: np.ndarray, shot: np.ndarray) -> np.ndarray:
    """Boolean mask of the rows inside the cone, the restriction `density.incone` applies.

    Point-in-triangle by the sign of the three edge cross-products: a point is inside when it
    does not fall on both sides at once, which keeps a body exactly on an edge inside.

    The mask is over every row handed in. `density.incone` is this RESTRICTED to density's own
    population, not computed instead of it, so the caller still owes the goal-side and keeper cuts."""
    v = shooting_cone(shot)
    s = [(q[0] - p[0]) * (xy[:, 1] - p[1]) - (q[1] - p[1]) * (xy[:, 0] - p[0])
         for p, q in zip(v, np.roll(v, -1, axis=0), strict=True)]
    return ~(((s[0] < 0) | (s[1] < 0) | (s[2] < 0)) & ((s[0] > 0) | (s[1] > 0) | (s[2] > 0)))


def inverse_distance_sum(shot: np.ndarray, fa: FrameArrays, cone_only: bool = False) -> float:
    """`density`, or `density.incone` when `cone_only`, over the outfield opponents goal-side.

    A crowding measure rather than a headcount: one body at two yards outweighs four at eight.
    Being anchored on a point is what makes it transplantable to another instant."""
    m = ~fa.teammate & ~fa.keeper & (fa.xy[:, 0] >= shot[0])
    if cone_only:
        m = m & in_shooting_cone(fa.xy, shot)
    return float((1.0 / body_distances(shot, fa)[m]).sum()) if m.any() else 0.0


def anchored_crowding(anchor: np.ndarray, xy: np.ndarray) -> float:
    """`inverse_distance_sum`'s kernel over an ALREADY-SELECTED population, goal-side of `anchor`.

    Same formula, transplanted off the shot. The caller owns which bodies count, so a caller that
    departs from the provider's population declares that departure itself.

    `xy` is (n, 2) in StatsBomb yards. Returns 0.0 on an empty goal-side set, the same value the
    provider writes, and never a NaN: nobody goal-side is no crowding, not an unknown."""
    if len(xy) == 0:
        return 0.0
    m = xy[:, 0] >= anchor[0]
    if not m.any():
        return 0.0
    d = _substitute_zero(np.hypot(xy[m, 0] - anchor[0], xy[m, 1] - anchor[1]))
    return float((1.0 / d).sum())


def pairwise_crowding(xy: np.ndarray) -> float:
    """Sum of inverse distances between every PAIR of bodies, with no anchor at all.

    The literal reading of "how little free space the bodies leave each other": what the bodies
    did to each other rather than to one chosen point. It grows with n(n-1)/2, so the body count
    travels beside it rather than folded into it."""
    if len(xy) < 2:
        return 0.0
    return float((1.0 / _substitute_zero(pdist(xy))).sum())


def defensive_line_area(fa: FrameArrays) -> float:
    """`DefArea` in square yards: the bounding box of the opposing DEFENSIVE LINE.

    The line alone, `position.id` 2 to 8, and not every outfield opponent, a wider reading the
    export does not bear out (hudl/data_provider.tex §sec:density). Falls back to the
    empty-population sentinel, which is a different empty population from `distance.ToD1`'s: a
    frame can show outfield opponents and still show no back four."""
    lo, hi = DEFENSIVE_LINE_POSITION_IDS
    m = ~fa.teammate & (fa.position_id >= lo) & (fa.position_id <= hi)
    if not m.any():
        return EMPTY_DEFENDER_AREA_SB
    x, y = fa.xy[m, 0], fa.xy[m, 1]
    return float((x.max() - x.min()) * (y.max() - y.min()))
