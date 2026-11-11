"""s02b_aim — the zone a corner was aimed at, read from what is known at the kick.

`pass.end_location` is the first-contact point (evidence.md §15), so on a delivery the defence
reached first it records where the defence got to the ball, not where the taker aimed it, as
un-xPass notes for failed passes (Robberechts, Van Roy & Davis 2022). A map of chain value by
END zone mixes aim and interception. This module builds what separates them:

    zone grid    canonical, corner struck from the y = 0 side. The base grid splits the band past
                 the six-yard line (x >= SIX_YARD_X, across the width of the box and so wider than
                 the six-yard box) and the spot band (GOALMOUTH_X to SIX_YARD_X) at the posts into
                 near, central and far, and calls the front of the box, outside it and wide of it
                 `other`. `coarse` and `fine` are the robustness grids.
    kick block   s02's delivery frame, its job-holding attackers and outfield defenders counted
                 per zone of the partition the switch names (src/zones.py; C's twelve cells are
                 the block A1 and A2 were registered on), plus where each side stands on average
                 inside the box. T = delivery.
    aim model    multinomial logistic with C chosen by grouped inner CV inside each outer fold,
                 taught only on the corners the attack reached first, where the end zone IS
                 the aim. On a flank free kick both arms carry the canonical kick spot.

The end zone enters twice and never as a feature: as the aim model's target, an observable
outcome, and as the label of the naive map. `_COLUMN_SOURCE` names the field every input is built
on, and `assert_sources` refuses one of s03's barred fields.

Five statistics come off one sample, every model rebuilt inside it (definitions.md §18):

    D1      paired OOF zone log loss, (design + kick block) minus (design), on attack-won corners.
            Negative means the kick block reads the aim.
    D2      naive minus honest. Negative means the near-post discount of the end-zone map belongs
            to where the defence got the ball.
    naive   near minus far chain xG per corner, corners grouped by END zone
    honest  the same, grouped by the aim the kick block infers (argmax)
    soft    the same from v_i = sum_k p_ik mu_k, which undoes misclassification when p is
            calibrated
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold

from constants import GOAL_POSTS_SB, SIX_YARD_X
from s02_signature import CommitmentSignature
from s03_value import (_BARRED_SOURCES, _COLUMN_SOURCE as _S03_SOURCE, _DELIVERY_CAT,
                       CompetingOutcomesModel)
from validate import assert_no_group_leak
from zones import band, canonical_y, depth, in_box, kicked_from_right, zone_names, zone_of

_NEAR_Y, _FAR_Y = GOAL_POSTS_SB[0][1], GOAL_POSTS_SB[1][1]
# The fine grid's outer bands, six yards outside each post. CHOSEN before any reading, as the grid
# a sceptic would try next; nothing was measured to place them.
_FINE_Y = (_NEAR_Y - 6.0, _FAR_Y + 6.0)
GRIDS = {
    "coarse": ("central", "far", "near", "other"),
    "base": ("other", "six_central", "six_far", "six_near", "spot_central", "spot_far",
             "spot_near"),
    "fine": ("box_central", "box_far", "box_far_out", "box_near", "box_near_out", "other",
             "six_central", "six_far", "six_far_out", "six_near", "six_near_out"),
}
CONTRAST = {"coarse": ("near", "far"), "base": ("six_near", "six_far"),
            "fine": ("six_near", "six_far")}

KICK_COLS = [f"{s}_{z}" for s in ("att", "def") for z in zone_names()] + \
            [f"{s}_box_{m}" for s in ("att", "def") for m in ("n", "mean_y", "sd_y", "mean_x")]
DESIGN_COLS = list(_DELIVERY_CAT)
# The canonical kick spot, which the design carries where the ball is not struck from the flag: on a
# flank free kick the block would otherwise be credited with where the foul was.
KICK_SPOT = ("kick_x", "kick_y")
_COLUMN_SOURCE = {**{c: "freeze_frame@T" for c in KICK_COLS},
                  **{c: _S03_SOURCE[c] for c in DESIGN_COLS}, "league": "match catalogue",
                  **dict.fromkeys(KICK_SPOT, "location")}

AIM_C_GRID = (0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1.0)
N_FOLDS, N_INNER = 5, 3
STATS = ("D1", "D2", "naive", "honest", "soft")


def assert_sources() -> None:
    """Raise when an aim-model input is built on a field that reads the play's future."""
    bad = {c: s for c, s in _COLUMN_SOURCE.items() if any(b in s for b in _BARRED_SOURCES)}
    if bad:
        raise ValueError(f"aim-model inputs built on barred fields: {bad}")


# ── The aim grids, on canonical points (the lines and the flip are src/zones.py's) ──
def end_zones(grid: str, x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Zone of each canonical end point under one grid; a point outside the box is `other`."""
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    if grid == "coarse":
        z = band(y)
    elif grid == "base":
        d = depth(x, "")
        z = np.where(d == "", "other", np.char.add(np.char.add(d, "_"), band(y)))
    elif grid == "fine":
        five = np.where(y < _FINE_Y[0], "near_out", np.where(y > _FINE_Y[1], "far_out", band(y)))
        z = np.char.add(np.where(x >= SIX_YARD_X, "six_", "box_"), five)
    else:
        raise ValueError(f"unknown grid {grid!r}")
    return np.where(in_box(x, y), z, "other").astype(object)


def kick_block(corner: pd.Series, three_sixty: pd.DataFrame) -> dict:
    """One corner's kick-time block off s02's frame: the attackers holding a job (taker and keepers
    out) and the outfield defenders, counted per zone of the switch's partition in the canonical
    frame of the kick, and the box stats of each side there."""
    cs = CommitmentSignature.from_corner(corner, three_sixty)
    f = cs.frame
    kick_y = float(f.loc[f["is_ball"] == 1, "y"].iloc[0])
    right = bool(kicked_from_right(kick_y))
    x, y_raw = f["x"].to_numpy(float), f["y"].to_numpy(float)
    y = canonical_y(y_raw, right)
    z, box = zone_of(x, y_raw, kick_y), in_box(x, y)
    row = {"kick_side_right": right}
    for side, m in (("att", f.index.isin(cs._att().index)), ("def", f.index.isin(cs._def().index))):
        row.update({f"{side}_{k}": int((m & (z == k)).sum()) for k in zone_names()})
        b = m & box
        row.update({f"{side}_box_n": int(b.sum()),
                    f"{side}_box_mean_y": float(y[b].mean()) if b.any() else np.nan,
                    f"{side}_box_sd_y": float(y[b].std()) if b.sum() > 1 else np.nan,
                    f"{side}_box_mean_x": float(x[b].mean()) if b.any() else np.nan})
    return row


# ── The aim model ────────────────────────────────────────────────────────
def design(table: pd.DataFrame, extra: tuple = (),
           numeric: tuple = ()) -> tuple[np.ndarray, np.ndarray]:
    """(A, K): the delivery design as dummies, and the design with the kick block beside it. NaN
    stays in; `_proba` imputes it with the training block's median. `extra` adds categoricals and
    `numeric` numbers to BOTH arms, so what the kick block buys is read net of them."""
    undeclared = [c for c in (*extra, *numeric) if c not in _COLUMN_SOURCE]
    if undeclared:
        raise ValueError(f"columns without a declared source: {undeclared}")
    A = pd.get_dummies(table[DESIGN_COLS + list(extra)].astype(str), dtype=float).to_numpy()
    A = np.c_[A, table[list(numeric)].to_numpy(float)]
    return A, np.c_[A, table[KICK_COLS].to_numpy(float)]


def _proba(Xtr: np.ndarray, ytr: np.ndarray, Xte: np.ndarray, C: float, n: int) -> np.ndarray:
    """Zone probabilities for Xte over all n zones, s03's train-only impute and scale; a zone the
    block never saw gets a floor before the rows renormalise."""
    tf = CompetingOutcomesModel._fit_prep(Xtr)
    m = LogisticRegression(C=C, max_iter=5000).fit(tf(Xtr), ytr)
    out = np.full((len(Xte), n), 1e-6)
    out[:, m.classes_] = m.predict_proba(tf(Xte))
    return out / out.sum(1, keepdims=True)


def zone_loss(p: np.ndarray, z: np.ndarray) -> np.ndarray:
    """Per-row log loss of the true zone."""
    return -np.log(np.clip(p[np.arange(len(z)), z], 1e-6, 1.0))


def pick_c(X: np.ndarray, z: np.ndarray, g: np.ndarray, n: int) -> float:
    """The C with the best grouped inner-fold zone log loss on one training block."""
    splits = list(GroupKFold(N_INNER).split(X, groups=g))
    for a, b in splits:
        assert_no_group_leak(g, a, b, "match_id")
    loss = [np.concatenate([zone_loss(_proba(X[a], z[a], X[b], C, n), z[b])
                            for a, b in splits]).mean() for C in AIM_C_GRID]
    return AIM_C_GRID[int(np.argmin(loss))]


def aim_oof(X: np.ndarray, z: np.ndarray, teach: np.ndarray, groups: np.ndarray,
            n: int, shuffle_seed: int | None = None) -> np.ndarray:
    """Out-of-fold aim probabilities (rows x n zones), folds grouped by match. Every row is scored,
    only `teach` rows fit, and C is re-picked inside each outer fold: choosing it on the whole table
    would score each row with a value its own match helped choose. `shuffle_seed` deals the matches
    to the outer folds at random rather than by size, another partition of the same sample."""
    p = np.zeros((len(z), n))
    outer = GroupKFold(N_FOLDS, shuffle=shuffle_seed is not None, random_state=shuffle_seed)
    for tr, te in outer.split(X, groups=groups):
        assert_no_group_leak(groups, tr, te, "match_id")
        fit = tr[teach[tr]]
        p[te] = _proba(X[fit], z[fit], X[te], pick_c(X[fit], z[fit], groups[fit], n), n)
    return p


# ── Maps and statistics ──────────────────────────────────────────────────
def contrast(v: np.ndarray, zones: np.ndarray, a: str, b: str) -> float:
    """Mean v in zone a minus mean v in zone b; NaN when either is empty."""
    ma, mb = zones == a, zones == b
    return float(v[ma].mean() - v[mb].mean()) if ma.any() and mb.any() else np.nan


def soft_contrast(v: np.ndarray, p: np.ndarray, ja: int, jb: int) -> float:
    """mu_a - mu_b from the least-squares fit v_i = sum_k p_ik mu_k."""
    mu = np.linalg.lstsq(p, v, rcond=None)[0]
    return float(mu[ja] - mu[jb])


def statistics(table: pd.DataFrame, grid: str = "base", value: str = "xg",
               extra: tuple = (), numeric: tuple = ()) -> np.ndarray:
    """STATS on one sample, both aim models rebuilt from scratch. `table` carries `zone` (canonical
    end zone under `grid`), `won` (the attack made the first contact), `match_id` and `value`."""
    zones = GRIDS[grid]
    a, b = CONTRAST[grid]
    z = table["zone"].map({k: i for i, k in enumerate(zones)}).to_numpy()
    won, g = table["won"].to_numpy(bool), table["match_id"].to_numpy()
    v = table[value].to_numpy(float)
    A, K = design(table, extra, numeric)
    pa, pk = aim_oof(A, z, won, g, len(zones)), aim_oof(K, z, won, g, len(zones))
    d1 = float((zone_loss(pk, z) - zone_loss(pa, z))[won].mean())
    naive = contrast(v, table["zone"].to_numpy(), a, b)
    honest = contrast(v, np.asarray(zones, dtype=object)[pk.argmax(1)], a, b)
    soft = soft_contrast(v, pk, zones.index(a), zones.index(b))
    return np.array([d1, naive - honest, naive, honest, soft])
