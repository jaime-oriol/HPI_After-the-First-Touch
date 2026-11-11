"""s02_signature — Stage 2: commitment signature + delivery-instant geometry (360).

Delivery-instant geometry from the StatsBomb 360 freeze-frame, per
methods_second_ball.md §3. Everything is computed STRICTLY from the frame at
T=delivery (anti-leakage) and over the players the camera actually saw (360 is
detection-masked by construction). Coordinates are oriented so the attack points
to x=120 before anything is measured.

Two products:
  - the COMMITMENT SIGNATURE: each non-taking attacker in one of three coach jobs,
    read with the width of the box (src/zones.py): in the box he attacks the cross,
    in front of it inside its width he stakes the drop zone, anywhere else he holds
    back;
  - the delivery-instant GEOMETRY set: each side's bodies per zone under every
    partition in play (src/zones.py, whose switch picks the one the model reads), the
    defenders in the switch's frontal (s05's treatment), the central edge band
    as a diagnostic, box-block depth, rebound-zone control (Voronoi share), the open
    central lane, marking tightness, dispersion/hulls, CONGESTION and attacker
    density at the target.

Congestion measures distances rather than bodies, which is what separates it from
occupation: attackers spread across the width occupy the box heavily and congest
nothing.

Rest-defence features (rest-line height, counter-exposure, defenders behind the
ball centre) are not computable here: the delivery freeze-frame covers the box and
little more, so there is nobody deep to measure (docs/measured.md §coverage_360,
from audit_360.py). Three names carry that limit instead of hiding it.
`box_block_depth` is the depth of the block, not a rest-defence line height: it
reads the two detected defenders furthest from goal wherever they stand, and on
most corners one of them stands in the central edge band.
`open_lane_central` is the gap left in the central channel by the defenders the
camera caught, not a counter lane, whatever its FEATURE_GROUPS label suggests.
`n_def_behind_half` is the constant that proves the point, kept as a coverage
diagnostic and excluded from the feature groups.

No event-stream logic here (that is s01_phases); no model (that is s03).

References: Bauer, Anzer & Smith 2022 (man-marking and role classification) and
Groom et al. 2026 (role and ghosting taxonomy), both in docs/sota/. The handcrafted
geometry here is the freeze-frame-native, snap-only counterpart to those tracking
models.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from scipy.spatial import ConvexHull
from scipy.spatial.distance import cdist

from constants import (BOX_X, DROP_ANCHOR_SB, EDGE_X, EDGE_Y_HALF, GOALMOUTH_X, HALF_X,
                       PITCH_X_SB, PITCH_Y_MID_SB, PITCH_Y_SB, REBOUND_X, REBOUND_Y,
                       TARGET_DENSITY_RADII, ZONE_PARTITION)
from s01_phases import _is_xy
from shot_geometry import anchored_crowding, pairwise_crowding
from zones import JOBS as ROLES, job_of, table_counts

_GRID_STEP = 1.0              # rebound-grid sampling resolution, not a football quantity

# Rebound-control sampling grid is fixed -> build it ONCE, not per corner
# (matters at Big-5 scale: ~17k corners x 931 grid points). It is 19x49 points and
# not 18x48: that is the band in YARDS, and both endpoints are sampled.
_REBOUND_GRID = np.array([(x, y)
                          for x in np.arange(REBOUND_X[0], REBOUND_X[1] + _GRID_STEP, _GRID_STEP)
                          for y in np.arange(REBOUND_Y[0], REBOUND_Y[1] + _GRID_STEP, _GRID_STEP)])


def _orient(corner: pd.Series) -> Callable[[float, float], tuple]:
    """The function that orients a point so the attack points to x=120 for this delivery.

    A delivery taken from the low-x side is flipped (both axes) so every one is
    measured in one canonical frame. On this export no admitted delivery flips: a
    corner is taken at the high-x end (min x = 118.3 on the grant) and a flank free
    kick is admitted only from FK_FLANK_MIN_X, so this is an inactive guard.

    The comparison is STRICT, so a delivery taken exactly on the halfway line would
    not rotate the whole frame 180 degrees."""
    flip = float(corner["location"][0]) < HALF_X
    def fn(x: float, y: float) -> tuple:
        return (PITCH_X_SB - x, PITCH_Y_SB - y) if flip else (x, y)
    return fn


def build_frame(corner: pd.Series, three_sixty: pd.DataFrame) -> tuple[pd.DataFrame, tuple | None]:
    """One corner's 360 -> oriented tidy frame + oriented delivery target.

    Frame columns: x, y, team_id (1=att, 0=def), is_goalkeeper, is_actor, is_ball.
    The ball row is the corner spot (the freeze-frame carries no ball)."""
    fz = three_sixty[three_sixty["event_uuid"] == corner["id"]]
    if fz.empty:
        raise ValueError(f"no 360 freeze-frame for corner {corner['id']}")
    orient = _orient(corner)

    rows = []
    for p in fz.iloc[0]["freeze_frame"]:
        x, y = orient(float(p["location"][0]), float(p["location"][1]))
        rows.append({"x": x, "y": y,
                     "team_id": 1 if p.get("teammate") else 0,   # 'teammate' is relative to the taker (= attacker)
                     "is_goalkeeper": int(bool(p.get("keeper"))),
                     "is_actor": int(bool(p.get("actor"))), "is_ball": 0})
    bx, by = orient(float(corner["location"][0]), float(corner["location"][1]))
    rows.append({"x": bx, "y": by, "team_id": 1, "is_goalkeeper": 0, "is_actor": 0, "is_ball": 1})

    end = corner.get("pass.end_location")
    target = orient(float(end[0]), float(end[1])) if _is_xy(end) else None
    return pd.DataFrame(rows), target


class CommitmentSignature:
    """Commitment signature and delivery-instant geometry for one corner."""

    def __init__(self, frame: pd.DataFrame, delivery_target: tuple | None = None) -> None:
        self.frame = frame.copy()
        self.target = delivery_target
        self._assign_roles()

    @classmethod
    def from_corner(cls, corner: pd.Series, three_sixty: pd.DataFrame) -> "CommitmentSignature":
        """Convenience constructor straight from a corner row + the match 360."""
        frame, target = build_frame(corner, three_sixty)
        return cls(frame, target)

    # ── Player subsets (outfield, detected) ─────────────────────────────
    def _att(self) -> pd.DataFrame:
        """Non-taking outfield attackers (exclude the taker, the ball, the GK)."""
        f = self.frame
        return f[(f["team_id"] == 1) & (f["is_ball"] == 0) & (f["is_actor"] == 0) & (f["is_goalkeeper"] == 0)]

    def _def(self) -> pd.DataFrame:
        """Outfield defenders, keeper excluded: the edge and depth counts want bodies
        that can contest a delivery, and a keeper defends a line instead."""
        f = self.frame
        return f[(f["team_id"] == 0) & (f["is_ball"] == 0) & (f["is_goalkeeper"] == 0)]

    # ── Commitment signature: the job of each non-taking attacker ───────
    def _assign_roles(self) -> None:
        """Write a 'role' column: the three jobs for attackers, fixed tags for the rest.

        A job is read BY POSITION, which is what form_answers.md Q2 describes ("each attacker
        is classified by position into one of three jobs"), with the width of the box
        (`zones.job_of`). Proximity to the delivery target is not read: that target is
        pass.end_location, where the defence got to the ball on most deliveries (evidence.md
        §15), so it would score the plan from the outcome it is separated from.

        The four tags are matched in precedence order, so a body that satisfies two
        conditions takes the first: the ball row carries team_id=1 and would otherwise
        read as an attacker, and the taker is an attacker too. Only what falls through
        all four gets a job, so `gk_att` needs no team test: a defending keeper has
        already matched `defender`."""
        f = self.frame
        conditions = [f["is_ball"].to_numpy(bool), f["is_actor"].to_numpy(bool),
                      (f["team_id"] == 0).to_numpy(), f["is_goalkeeper"].to_numpy(bool)]
        tags = ["ball", "taker", "defender", "gk_att"]
        jobs = job_of(f["x"].to_numpy(float), f["y"].to_numpy(float))
        f["role"] = np.select(conditions, tags, default=jobs).astype(object)

    def signature(self) -> dict:
        """Job counts and the commit index, the share of the job-holding attackers in the box."""
        roles = self.frame.loc[self.frame["role"].isin(ROLES), "role"]
        counts = {f"n_{r}": int((roles == r).sum()) for r in ROLES}
        counts["commit_index"] = (counts["n_attack_cross"] / len(roles)) if len(roles) else np.nan
        return counts

    # ── Delivery-instant geometry (all at T=delivery) ───────────────────
    def features(self) -> dict:
        """The handcrafted delivery-instant geometry (methods §3).

        Three of these names are deliberately not the ones methods §3 uses; the module
        docstring says which and why the frame cannot support the originals.

        The zone counts come under both partitions in play, read in the canonical frame of
        this delivery's kick, and the switch picks the ones the model reads (src/zones.py).
        `n_def_edge_band` counts the central edge band beside the frontal; it is in no feature
        group.

        `box_block_depth` varies across corners, and removing its group is a rung of
        s05's specification curve (measured.md §causal). `n_def_behind_half` is the
        opposite case and s03 keeps it out of FEATURE_GROUPS, because a constant column
        contributes nothing and its zero ablation delta would read as a finding."""
        A, D = self._att(), self._def()
        ax, ay = A["x"].to_numpy(float), A["y"].to_numpy(float)
        dx, dy = D["x"].to_numpy(float), D["y"].to_numpy(float)
        kick_y = float(self.frame.loc[self.frame["is_ball"] == 1, "y"].iloc[0])
        edge_band = ((dx >= EDGE_X[0]) & (dx <= EDGE_X[1])
                     & (np.abs(dy - PITCH_Y_MID_SB) <= EDGE_Y_HALF))

        feat = {
            "n_att": len(A), "n_def": len(D),
            **table_counts(ax, ay, dx, dy, kick_y),
            "n_def_edge_band": int(edge_band.sum()),
            "box_block_depth": float(np.sort(dx)[:2].mean()) if len(D) else np.nan,
            "n_def_behind_half": int((dx < HALF_X).sum()),
            "control_rebound": self._rebound_control(),
            "open_lane_central": self._open_lane(dx, dy),
            "marking_cost": self._marking_cost(ax, ay, dx, dy),
            "def_dispersion": self._dispersion(dx, dy),
            "att_dispersion": self._dispersion(ax, ay),
            "def_hull_area": self._hull_area(dx, dy),
            "att_hull_area": self._hull_area(ax, ay),
        }
        feat.update(self._congestion())
        feat.update(self._target_density(ax, ay))
        return feat

    # ── Feature helpers ─────────────────────────────────────────────────
    def _rebound_control(self) -> float:
        """Fraction of the rebound band whose nearest player is a defender
        (hard-Voronoi share over a grid).

        Takes no player subsets on purpose: a Voronoi share needs every body that
        can reach the ball, so this reads self.frame directly and counts keepers
        and the taker too. _att()/_def() exclude both and would be the wrong set."""
        f = self.frame
        out = f[(f["is_ball"] == 0)]
        pts = out[["x", "y"]].to_numpy(float)
        is_def = (out["team_id"] == 0).to_numpy()
        if len(pts) == 0:
            return np.nan
        nearest = cdist(_REBOUND_GRID, pts).argmin(axis=1)
        return float(is_def[nearest].mean())

    @staticmethod
    def _open_lane(dx: np.ndarray, dy: np.ndarray) -> float:
        """Largest y-gap between adjacent defenders standing in the central channel and
        SHORT of the box edge. Channel walls count as edges, so a lone defender still
        leaves a measurable gap on each side of him."""
        lo, hi = PITCH_Y_MID_SB - EDGE_Y_HALF, PITCH_Y_MID_SB + EDGE_Y_HALF
        ys = np.sort(dy[(dx < BOX_X) & (np.abs(dy - PITCH_Y_MID_SB) <= EDGE_Y_HALF)])
        bounds = np.concatenate(([lo], ys, [hi]))
        return float(np.diff(bounds).max())

    @staticmethod
    def _marking_cost(ax: np.ndarray, ay: np.ndarray,
                      dx: np.ndarray, dy: np.ndarray) -> float:
        """Mean matched attacker-defender distance (Hungarian assignment; cf. the
        supervised man-marking of Bauer-Anzer-Smith 2022). Lower = tighter
        man-marking; higher = looser / scrambled rest shape."""
        if len(ax) == 0 or len(dx) == 0:
            return np.nan
        cost = cdist(np.c_[dx, dy], np.c_[ax, ay])
        r, c = linear_sum_assignment(cost)
        return float(cost[r, c].mean())

    @staticmethod
    def _dispersion(x: np.ndarray, y: np.ndarray) -> float:
        """RMS spread around the centroid (how stretched the set is)."""
        if len(x) < 2:
            return np.nan
        return float(np.sqrt(x.var() + y.var()))

    @staticmethod
    def _hull_area(x: np.ndarray, y: np.ndarray) -> float:
        """Convex-hull area of the set (0 if fewer than 3 points / degenerate)."""
        if len(x) < 3:
            return 0.0
        try:
            return float(ConvexHull(np.c_[x, y]).volume)     # 'volume' is area in 2D
        except Exception:
            return 0.0

    def _congestion_population(self) -> np.ndarray:
        """The congestion population as (n, 2): every body that can obstruct a run.

        NO box cut here, and the name does not claim one: the delivery frame covers
        the box and little else in the first place, and the cut that matters is the goal-side
        one each anchor applies for itself in `anchored_crowding`.

        Both teams, which is the ONE departure definitions.md §2 authorises. The provider counts
        defenders because it is asking who can block this shot, and the question here is whose run
        gets obstructed. Keepers on both sides are out, as in the provider's rule. The taker is out
        too, and that is a SECOND departure, declared: he is the body striking the ball rather
        than one blocking a run, and the corner flag puts him goal-side of either anchor on
        virtually every corner, so leaving him in would add a constant to the measure."""
        f = self.frame
        b = f[(f["is_ball"] == 0) & (f["is_goalkeeper"] == 0) & (f["is_actor"] == 0)]
        return b[["x", "y"]].to_numpy(float)

    def _congestion(self) -> dict:
        """How little free space the bodies leave each other at the instant of the strike.

        The provider's `density` kernel (shot_geometry) evaluated on the DELIVERY freeze-frame.
        The anchor is what makes it legal: `pass.end_location` IS the first-contact point, so a
        measure anchored there would inherit exactly the defect that barred att_density_5/10 from
        the design matrix. Both anchors here are fixed and global, so neither reads its own
        corner's future.

        `spot` anchors on the penalty spot, (GOALMOUTH_X, PITCH_Y_MID_SB), `drop` on the
        measured median landing point, and the pairwise form on nothing at all.

        The count travels with every one of them. The frame sees the box and little else, so the
        denominator is the bodies the camera caught: fourteen is not comparable with nine, and the
        per-body forms are how that comparison is made rather than assumed."""
        xy = self._congestion_population()
        n = len(xy)
        n_pairs = n * (n - 1) / 2
        spot = anchored_crowding(np.array([GOALMOUTH_X, PITCH_Y_MID_SB]), xy)
        drop = anchored_crowding(np.asarray(DROP_ANCHOR_SB, dtype=float), xy)
        pair = pairwise_crowding(xy)
        return {
            "congestion_bodies": n,
            "congestion_spot": spot,
            "congestion_drop": drop,
            "congestion_pairwise": pair,
            "congestion_spot_per_body": (spot / n) if n else np.nan,
            "congestion_drop_per_body": (drop / n) if n else np.nan,
            "congestion_pairwise_per_pair": (pair / n_pairs) if n_pairs else np.nan,
        }

    def _target_density(self, ax: np.ndarray, ay: np.ndarray) -> dict:
        """Attackers crowding the delivery target (within 5 / 10 yd)."""
        if self.target is None or len(ax) == 0:
            return {"att_density_5": np.nan, "att_density_10": np.nan}
        d = np.hypot(ax - self.target[0], ay - self.target[1])
        near, far = TARGET_DENSITY_RADII
        return {"att_density_5": int((d <= near).sum()), "att_density_10": int((d <= far).sum())}

    def row(self, set_piece_id: str) -> dict:
        """Flat record (signature + features) keyed for the dataset builder, with the partition the
        jobs and the frontal were read under (`zones.require_built_under`)."""
        return {"set_piece_id": set_piece_id, "zone_partition": ZONE_PARTITION, **self.signature(),
                **self.features()}
