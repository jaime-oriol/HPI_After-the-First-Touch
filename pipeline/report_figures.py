"""report_figures — regenerate every measured figure from the derived parquets.

Hand-typed numbers in prose are the repo's main source of rot: a figure written
into a .md is correct on the day it is pasted and silently wrong after the next
pipeline run, and the only way anyone finds out is a reader spotting two documents
that disagree. This driver recomputes the figures from the artefacts the pipeline
actually wrote and emits them to one generated file, stamped with the mtime of every
parquet it read, so a stale number is visible instead of invisible.

    docs/measured.md    human-readable, the file prose should point at
    docs/measured.json  machine-readable, what tests/test_figures.py checks

No new measurement: every number summarises an artefact some other driver produced,
so if a figure looks wrong the fix belongs in that driver.

    python pipeline/report_figures.py
    python pipeline/report_figures.py --check    # fail if the outputs are stale
"""

from __future__ import annotations

import argparse
import ast
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd
from scipy.stats import skew

_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO / "src"))
sys.path.insert(0, str(_REPO / "pipeline"))
_DER = _REPO / "data" / "derived"
_OUT_MD = _REPO / "docs" / "measured.md"
_OUT_JSON = _REPO / "docs" / "measured.json"

import loader as _loader                        # noqa: E402
import loader_sb as _sb                         # noqa: E402
from audit_tracking import _ADOPTED_VISIBLE_BAR  # noqa: E402
from constants import MIN_CELL, ZONE_PARTITION  # noqa: E402
from eda_penalties import PHASES as _PENALTY_PHASES   # noqa: E402
from eda_second_ball import NAMED_CASES         # noqa: E402
from probe_throw_ins import adopt as _adopt_throw_in   # noqa: E402
from s01_phases import AFTER_FIRST_TOUCH, is_own_goal   # noqa: E402
from sweep_counter import MIN_CONTIGUITY        # noqa: E402
from run_first_contact import decide as _fc_decide   # noqa: E402  the verdict rule, not a copy
from run_positive_control import decide as _pc_decide   # noqa: E402  the same, for the control
from run_aim_zone import decide as _az_decide           # noqa: E402  and for the aim zone
from run_causal import PRIMARY_SPEC as _CAUSAL_PRIMARY  # noqa: E402
from validate import Z95                                 # noqa: E402


def _load(name: str) -> pd.DataFrame | None:
    """A derived parquet, or None when that stage has not been run yet."""
    p = _DER / name
    return pd.read_parquet(p) if p.exists() else None


def _pct(part: float, whole: float) -> float | None:
    return round(100.0 * part / whole, 2) if whole else None


def _ci3(lo: float, hi: float) -> list | None:
    """A bound pair at three decimals, or None when the run did not interval it."""
    return [round(float(lo), 3), round(float(hi), 3)] if np.isfinite([lo, hi]).all() else None


def _ci5(lo: float, hi: float) -> list | None:
    """The same pair at five decimals: a log-loss delta of this size is zero at three."""
    return [round(float(lo), 5), round(float(hi), 5)] if np.isfinite([lo, hi]).all() else None


def _normal(point: float, lo: float, hi: float) -> list:
    """The normal reading of a refit interval: the point plus or minus half its width.

    The pivotal and percentile forms share that width and place it on either side of the point;
    this one keeps the point as the centre, which is the reading that decides
    (s03_value.step_refit_interval says why)."""
    half = (float(hi) - float(lo)) / 2
    return [round(float(point) - half, 5), round(float(point) + half, 5)]


def _covers(ci: list) -> bool:
    return bool(ci[0] <= 0 <= ci[1])


def _scalar(v: Any) -> Any:
    """One cell of a record as the JSON carries it: None for a missing value, a float at four
    decimals, an integer or a boolean as itself, anything else unchanged."""
    if v is None or (np.isscalar(v) and not isinstance(v, str) and pd.isna(v)):
        return None
    if isinstance(v, (bool, np.bool_)):
        return bool(v)
    if isinstance(v, (int, np.integer)):
        return int(v)
    if isinstance(v, (float, np.floating)):
        return round(float(v), 4)
    return v


def _curve_summary(curve: pd.DataFrame) -> dict:
    """A specification curve read as a whole: how many specifications, how many exclude zero per
    target, the range of the effect and whether its sign flips."""
    by_target = {}
    for tg, sub in curve.groupby("target"):
        by_target[str(tg)] = {
            "n_excluding_zero": int((~sub["covers_zero"].astype(bool)).sum()),
            "effect_range": [round(float(sub["effect"].min()), 5),
                             round(float(sub["effect"].max()), 5)],
            "sign_flips": bool(np.sign(sub["effect"]).nunique() > 1)}
    return {"n_specs": int(curve["spec"].nunique()),
            "rungs": sorted(str(s) for s in curve["spec"].unique()),
            "frac_extreme_pct_range": [round(100 * float(curve["frac_extreme"].min()), 2),
                                       round(100 * float(curve["frac_extreme"].max()), 2)],
            "by_target": by_target}


def _draws_by_contrast() -> dict:
    """{(population, contrast): (point, draws)} from run_refit_draws, or {} before it has run."""
    d = _load("refit_draws.parquet")
    if d is None or not len(d):
        return {}
    return {k: (float(g["point"].iloc[0]), g.sort_values("draw")["value"].to_numpy())
            for k, g in d.groupby(["population", "contrast"])}


def _refit_reading(point: float, draws: np.ndarray) -> dict:
    """One contrast's refit draws read every way: the normal form that decides, the pivotal and
    percentile forms beside it, and the two readings that test its symmetry assumption, the
    normal interval on the draws' own SD and the percentiles re-centred on the point."""
    q_lo, q_hi = np.percentile(draws, [2.5, 97.5])
    med, sd = float(np.median(draws)), float(draws.std(ddof=1))
    return {"normal_ci": _normal(point, q_lo, q_hi),
            "adjusted_ci": _ci5(2 * point - q_hi, 2 * point - q_lo),
            "refit_ci": _ci5(q_lo, q_hi),
            "normal_sd_ci": _ci5(point - Z95 * sd, point + Z95 * sd),
            "centred_ci": _ci5(point + q_lo - med, point + q_hi - med),
            "sd": round(sd, 5), "skew": round(float(skew(draws, bias=False)), 3),
            "n_boot": int(len(draws))}


def corpus(tag: str = "") -> dict:
    """Grant shape and the S01/S02 grain, from the cached corner and shot tables. Beside them, the
    deliveries s01 leaves without a defending team, the ones the defence never touched in the
    sequence, which s05 recovers as the match's other side for its fixed effects. Shots are shots
    and goals are every goal: an own goal counts in the goals and apart, and in no shot count or
    conversion (`s01_phases.is_own_goal`)."""
    corners, shots = _load(f"corners{tag}.parquet"), _load(f"shots{tag}.parquet")
    if corners is None or shots is None:
        return {}
    own, aft = is_own_goal(shots), shots["segment"].isin(AFTER_FIRST_TOUCH)
    seg = shots[~own].groupby("segment").agg(shots=("is_goal", "size"), goals=("is_goal", "sum"))
    seg["own_goals"] = shots[own].groupby("segment").size().reindex(seg.index).fillna(0).astype(int)
    seg["conv_pct"] = (100 * seg["goals"] / seg["shots"]).round(2)
    return {
        "corners_with_360": int(len(corners)),
        "defending_team_missing": int(corners["defending_team"].isna().sum()),
        "matches_with_360": int(corners["match_id"].nunique()),
        "shots_on_the_frontier": int((~own).sum()),
        "own_goals_on_the_frontier": int(own.sum()),
        "goals_on_the_frontier": int(shots["is_goal"].sum()),
        "after_first_touch_shots_pct": _pct((aft & ~own).sum(), (~own).sum()),
        "after_first_touch_goals_pct": _pct(shots.loc[aft, "is_goal"].sum(), shots["is_goal"].sum()),
        "by_segment": seg.sort_values("shots", ascending=False).to_dict("index"),
    }


def per_league(tag: str = "") -> dict:
    """The segment mix and conversion league by league, over the full grant.

    The publication floor is applied here rather than left to the reader: a cell
    under MIN_CELL shots is reported with its count and flagged, never as a rate,
    because a per-league flick-on rate on a handful of shots is not a finding. Shots and conversion
    are shots, as in `corpus`; a league's goals count its own goals, which sit apart as well."""
    shots, corners = _load(f"shots{tag}.parquet"), _load(f"corners{tag}.parquet")
    if shots is None or corners is None:
        return {}
    out = {}
    for lg, sub in shots.groupby("league"):
        own = is_own_goal(sub)
        seg = sub[~own].groupby("segment").agg(shots=("is_goal", "size"), goals=("is_goal", "sum"))
        mix = {}
        for s, row in seg.iterrows():
            n = int(row["shots"])
            mix[s] = {"shots": n, "share_pct": _pct(n, int((~own).sum())),
                      "goals": int(row["goals"]),
                      "conv_pct": _pct(row["goals"], n) if n >= MIN_CELL else None,
                      "below_floor": n < MIN_CELL}
        out[str(lg)] = {"shots": int((~own).sum()), "goals": int(sub["is_goal"].sum()),
                        "own_goals": int(own.sum()),
                        "corners_with_360": int((corners["league"] == lg).sum()),
                        "by_segment": mix}
    return out


def penalties() -> dict:
    """The penalties a delivery wins inside its episode (eda_penalties), which no segment of the
    frontier values, and Q1 read with them beside Q1 without them. Per population: how many
    deliveries win one and in which phase, the kicks taken and scored with their xG, how s01
    labels those deliveries, and the after-first-touch share of shots, goals and xG on the
    frontier as `corpus` reads it, then with every kick counted as a shot of its phase: a whistle
    after the first contact goes with what follows the first touch, one before it or at it with
    the delivery."""
    pen, grain = _load("eda_penalties.parquet"), _load("eda_deliveries.parquet")
    if pen is None or grain is None:
        return {}
    labels = grain.set_index("set_piece_id")["outcome"]
    out = {}
    for kind, tag in (("corner", ""), ("free_kick_flank", "_free_kick_flank")):
        shots, sub = _load(f"shots{tag}.parquet"), pen[pen["set_piece_type"] == kind]
        if shots is None or sub.empty:
            continue
        won = sub[sub["penalty"].astype(bool)]
        kicks = won[won["kick_id"].notna()]
        late, aft = kicks["phase"] == "after", shots["segment"].isin(AFTER_FIRST_TOUCH)
        shot = ~is_own_goal(shots)
        parts = {"shots": ((aft & shot).sum(), shot.sum(), late.sum(), len(kicks)),
                 "goals": (shots.loc[aft, "is_goal"].sum(), shots["is_goal"].sum(),
                           kicks.loc[late, "kick_goal"].sum(), kicks["kick_goal"].sum()),
                 "xg": (shots.loc[aft, "xg"].sum(), shots["xg"].sum(),
                        kicks.loc[late, "kick_xg"].sum(), kicks["kick_xg"].sum())}
        out[kind] = {
            "deliveries": int(len(sub)), "won_a_penalty": int(len(won)),
            "won_a_penalty_pct": _pct(len(won), len(sub)),
            "by_phase": {p: int((won["phase"] == p).sum()) for p in _PENALTY_PHASES},
            "kicks": int(len(kicks)), "kick_goals": int(kicks["kick_goal"].sum()),
            "kick_xg": round(float(kicks["kick_xg"].sum()), 2),
            "labelled_as": {str(k): int(v) for k, v in
                            labels.reindex(won["set_piece_id"]).value_counts().items()},
            **{f"after_first_touch_{k}_pct": {"without": _pct(a, n), "with": _pct(a + ka, n + kn)}
               for k, (a, n, ka, kn) in parts.items()}}
    return out


def coverage(tag: str = "") -> dict:
    """360 coverage and freeze-frame occupancy, from the coverage audit's artefact."""
    a = _load(f"audit360{tag}.parquet")
    if a is None:
        return {}
    tot = a.drop(columns="match_id").sum()
    # two denominators on purpose: visible_area stats only accumulate when the
    # polygon parses, the occupancy counts accumulate for every frame present
    n_va, n_fr = tot["n_360_corners"], tot["delivery_360"]
    return {
        "corners": int(tot["corners"]),
        "delivery_360_pct": _pct(tot["delivery_360"], tot["corners"]),
        "clearance_360_pct": _pct(tot["clearance_360"], tot["cleared"]),
        "next_touch_360_pct": _pct(tot["next_360"], tot["next_touch"]),
        "situation_frame_pct": _pct(tot["situation_any"], tot["cleared"]),
        "cleared_pct": _pct(tot["cleared"], tot["corners"]),
        "mean_visible_area_pct": round(float(tot["va_pct_sum"] / n_va), 2) if n_va else None,
        "deliveries_at_80pct_coverage": int(tot["va_ge80_count"]),
        "mean_bodies_in_box": round(float(tot["box_seen_sum"] / n_fr), 2) if n_fr else None,
        "bodies_deep_in_whole_grant": int(tot["deep_seen_sum"]),
    }


def coverage_by_league(tag: str = "") -> dict:
    """Delivery-360 coverage league by league, over every granted corner.

    The modelled population is the 360-carrying subset, so a coverage that varies by
    league means that subset is not a uniform sample of the grant. Leave-one-league-out
    reads differently when the held-out league is the worst-covered one, so the spread
    belongs in the record rather than in a driver's console output."""
    a = _load(f"audit360{tag}.parquet")
    if a is None:
        return {}
    j = a.merge(_sb.list_matches()[["match_id", "league"]], on="match_id", how="left")
    g = j.groupby("league").agg(matches=("match_id", "size"), corners=("corners", "sum"),
                                with_360=("delivery_360", "sum"))
    out = {}
    for lg, r in g.iterrows():
        out[str(lg)] = {"matches": int(r["matches"]), "corners": int(r["corners"]),
                        "delivery_360_pct": _pct(r["with_360"], r["corners"])}
    shares = [v["delivery_360_pct"] for v in out.values() if v["delivery_360_pct"] is not None]
    if shares:
        out["spread_pp"] = round(max(shares) - min(shares), 2)
    return out


def tracking(tag: str = "") -> dict:
    """Transition-window visibility on the tracked sample; a population without a cleared delivery
    in it has an empty record and no block."""
    t = _load(f"audit_tracking{tag}.parquet")
    if t is None or not len(t):
        return {}
    vf = t["visible_frac"]
    return {
        "cleared_corners_audited": int(len(t)),
        "mean_visible_frac": round(float(vf.mean()), 3),
        "median_visible_frac": round(float(vf.median()), 3),
        "share_over_80pct": _pct((vf >= _ADOPTED_VISIBLE_BAR).sum(), len(vf)),
        "share_over_50pct": _pct((vf >= 0.50).sum(), len(vf)),
        # the ball channel, separately: a player row is ESTIMATED when the camera lost the player,
        # a ball row when the vendor interpolated a ball it could not track, mostly airborne
        "mean_ball_frame_frac": round(float(t["ball_frame_frac"].mean()), 3),
        "mean_ball_visible_frac": round(float(t["ball_visible_frac"].mean()), 3),
    }


def race(tag: str = "") -> dict:
    """The second-ball race over the tracked matches.

    `ball_estimated_at_contact_pct` is the bound every figure here carries: the race locates
    the contact and the loose point on the ball row as the vendor emits it, and the vendor
    interpolates the airborne ball, which on a corner is the whole flight. The
    `visible_ball_at_contact` block scores the same figures again over the corners whose ball
    the vendor detected at the contact. It is a bound at the adopted tau and not a refit: tau
    stays the one calibrated over every race, so a reader sees how much survives the mask and
    whether the numbers move, without a second model being fitted."""
    r = _load(f"race_corners{tag}.parquet")
    if r is None:
        return {}
    lab = r[r["actual_att"].notna()]
    est_c = r["ball_estimated_at_contact"].astype(bool)
    est_l = r["ball_estimated_at_loose"]
    vis = lab[~lab["ball_estimated_at_contact"].astype(bool)]
    return {
        "raced_corners": int(len(r)),
        "matches": int(r["match_id"].nunique()),
        "tau_adopted_s": float(r["tau_s"].iloc[0]),
        "mean_p_att": round(float(r["p_att"].mean()), 3),
        "median_arrival_margin_s": round(float(r["tti_margin_s"].median()), 3),
        "att_recovered_pct": _pct(lab["actual_att"].sum(), len(lab)) if len(lab) else None,
        "self_agreement_pct": (_pct((lab["pred_att"] == lab["actual_att"]).sum(), len(lab))
                               if len(lab) else None),
        "ball_estimated_at_contact_pct": _pct(int(est_c.sum()), len(r)),
        "ball_estimated_at_loose_pct": _pct(int(est_l.eq(True).sum()), len(r)),
        "ball_absent_at_loose": int(est_l.isna().sum()),
        "visible_ball_at_contact": {
            "raced_corners": int((~est_c).sum()),
            "labelled": int(len(vis)),
            "below_floor": bool(len(vis) < MIN_CELL),
            "att_recovered_pct": _pct(vis["actual_att"].sum(), len(vis)) if len(vis) else None,
            "self_agreement_pct": (_pct((vis["pred_att"] == vis["actual_att"]).sum(), len(vis))
                                   if len(vis) else None),
            "mean_p_att": (round(float(r.loc[~est_c, "p_att"].mean()), 3)
                           if (~est_c).any() else None),
        },
    }


def situation(tag: str = "") -> dict:
    """The four states, what each is worth, and the two coverages that bound them.

    `four_state_known` against `cleared_corners` is the limit the grid carries: the cross
    needs a delivery freeze-frame for `staked` and a next touch inside the episode for `won`,
    and a corner missing either has no state. Generated rather than divided in prose because
    every rate in `by_state` is over the KNOWN subset and quoting it against the cleared count
    is the easy slip.

    `situation_frame_usable` is the other one. s08 refuses to score bodies read at one instant
    against a drop point reached at another, so it needs the frame of the drop's OWN event;
    `audit_360` measures the rawer thing, whether either window carries a frame at all
    (§coverage_360, `situation_frame_pct`). The two are not the same number and the gap is the
    rule's cost.

    `radii` is what the three chosen situation constants cut on this table (run_situation's
    radii record), the generated twin their provenance in constants.py points at."""
    s = _load(f"situation{tag}.parquet")
    if s is None:
        return {}
    fs = s[s["four_state"].notna()]
    usable = s[s["n_seen"].notna()]
    by = fs.groupby("four_state").agg(
        n=("value_xg_after", "size"),
        xg=("value_xg_after", "mean"),
        goal_pct=("value_goal_after", "mean"))
    by["xg"] = by["xg"].round(4)
    by["goal_pct"] = (100 * by["goal_pct"]).round(2)
    by["share_pct"] = (100 * by["n"] / len(fs)).round(1)
    out = {
        "cleared_corners": int(len(s)),
        "matches": int(s["match_id"].nunique()),
        "four_state_known": int(len(fs)),
        "four_state_known_pct": _pct(len(fs), len(s)),
        "after_clearance_shots": int(s["n_after_shots"].sum()),
        "after_clearance_goals": int(s["value_goal_after"].sum()),
        # the xG of a second ball is zero when no shot follows it: these two say how often one does and what that shot is worth
        "second_balls_with_shot_pct": _pct(int((s["n_after_shots"] > 0).sum()), len(s)),
        "xg_per_shot_after": round(float(s.loc[s["n_after_shots"] > 0, "value_xg_after"].sum()
                                         / s.loc[s["n_after_shots"] > 0, "n_after_shots"].sum()), 4),
        "four_state_unknown": _unknown_split(s, len(fs)),
        "situation_frame_usable": int(len(usable)),
        "situation_frame_usable_pct": _pct(len(usable), len(s)),
        # The coherence rule in s08.label, counted: which event the frame actually came from.
        "situation_frame_source": {str(k): int(v) for k, v in
                                   usable["sit_frame_source"].value_counts().items()},
        "by_state": by.to_dict("index"),
    }
    # Their OWN keys, carrying the INTERVAL alone: the points live in `by_state`, and rounding them
    # a second time here would publish one rate twice at two precisions. Nested rather than dotted
    # because the bindings address a figure by dotted path.
    v = _load(f"situation{tag}_states.parquet")
    if v is not None and len(v):
        ci: dict = {}
        for _, r in v[v["block"] == "state"].iterrows():
            ci.setdefault(str(r["key"]), {})[str(r["metric"])] = [round(float(r["lo"]), 4),
                                                                  round(float(r["hi"]), 4)]
        out["by_state_ci"] = ci
        out["contrasts"] = {
            str(r["key"]): {"delta": round(float(r["point"]), 4),
                            "ci": [round(float(r["lo"]), 4), round(float(r["hi"]), 4)],
                            "covers_zero": bool(r["covers_zero"])}
            for _, r in v[v["block"] == "contrast"].iterrows()}
    rad = _load(f"situation{tag}_radii.parquet")
    if rad is not None and len(rad):
        out["radii"] = {str(b): {str(r["key"]): {"value": round(float(r["value"]), 4),
                                                  "n": int(r["n"])} for _, r in g.iterrows()}
                        for b, g in rad.groupby("block")}
    return out


def _unknown_split(s: pd.DataFrame, n_known: int) -> dict:
    """Which half of the four-state cross a cleared corner lacks, the two being different limits:
    a delivery 360, present wherever the staker count is known, and a next touch that contests
    the loose ball. `staked` needs both, so its own nulls cannot tell them apart."""
    no_360, no_touch = s["n_stakers"].isna(), s["won_second_ball"].isna()
    out = {"total": int(len(s) - n_known), "no_delivery_360": int((no_360 & ~no_touch).sum()),
           "no_next_touch": int((no_touch & ~no_360).sum()),
           "neither": int((no_360 & no_touch).sum())}
    if out["total"] != out["no_delivery_360"] + out["no_next_touch"] + out["neither"]:
        raise ValueError(f"the four-state unknowns do not partition the missing states: {out}")
    return out


def frontier(name: str = "frontier.parquet") -> dict:
    """The elementary analysis: obs/exp per segment, origin and league, clustered by match.

    This is the block argument.md §3 is built on, and the interval is the reason it is
    generated rather than quoted. describe_frontier persists a match-clustered bound beside the
    exact Poisson one; the two condition differently (its module docstring), and `ci` and
    `excludes_1` here carry the clustered one, which is what treats the match as the unit. A cell's
    `shots` and `goals` are its shots, and `own_goals` its own goals, which enter no ratio."""
    f = _load(name)
    if f is None:
        return {}
    out: dict = {}
    for (pop, block), sub in f.groupby(["set_piece_type", "block"]):
        rows = {}
        # Two blocks are differences and shares rather than cells, so they carry no shot count
        # by construction and the skip below would drop them silently.
        if block in ("body_part_contrast", "headline"):
            out.setdefault(str(pop), {})[str(block)] = {
                str(r["key"]): {"point": round(float(r["point"]), 4),
                                "ci": [round(float(r["lo"]), 4), round(float(r["hi"]), 4)],
                                "covers_zero": bool(r["lo"] <= 0 <= r["hi"])}
                for _, r in sub.iterrows()}
            continue
        # A row with no shot count is not a cell: it is an artefact of reindexing over a key
        # that the sample never contained. Skip it rather than publish a NaN as a figure.
        for _, r in sub[sub["shots"].notna()].iterrows():
            rows[str(r["key"])] = {
                "shots": int(r["shots"]), "goals": int(r["goals"]),
                **({"own_goals": int(r["own_goals"])} if pd.notna(r.get("own_goals")) else {}),
                "sum_xg": round(float(r["sum_xg"]), 2),
                "obs_exp": round(float(r["obs_exp"]), 4),
                "ci": [round(float(r["lo"]), 4), round(float(r["hi"]), 4)],
                "excludes_1": bool(r["excludes_1"]),
                "below_floor": bool(str(r.get("thin", "")) != ""),
            }
        out.setdefault(str(pop), {})[str(block)] = rows
    return out


def falsification(tag: str = "") -> dict:
    """The gate that licensed the frontier, from falsify_frontier's record.

    `delta` is the block evidence.md §2 leads on, and its denominator is NOT §corpus's. It
    excludes own goals, which carry no key pass and no xG, and it excludes `late`, so subtracting
    two §corpus percentages to rebuild it gives a different number. `frontier.headline` carries
    the same contrast over every shot including the own goals, and the two are both right
    about different populations."""
    f = _load(f"falsification{tag}_stats.parquet")
    if f is None or not len(f):
        return {}
    out: dict = {}
    for block, sub in f.groupby("block"):
        rows = {}
        for _, r in sub.iterrows():
            e: dict = {}
            if pd.notna(r["point"]):
                e["point"] = round(float(r["point"]), 4)
            if pd.notna(r["lo"]):
                e["ci"] = [round(float(r["lo"]), 4), round(float(r["hi"]), 4)]
                e["covers_zero"] = bool(r["lo"] <= 0 <= r["hi"])
            for col in ("n", "goals", "ours_only", "theirs_only"):
                if col in sub.columns and pd.notna(r.get(col)):
                    e[col] = int(r[col])
            if "sum_xg" in sub.columns and pd.notna(r.get("sum_xg")):
                e["sum_xg"] = round(float(r["sum_xg"]), 2)
            # MIN_CELL floors a RATE, so it is carried on the conversion block alone. A
            # cross-tab cell is a count and a zero there is an empty cell, not a thin one.
            if block == "conversion" and "n" in e:
                e["below_floor"] = bool(e["n"] < MIN_CELL)
            rows[str(r["key"])] = e["point"] if list(e) == ["point"] else e
        out[str(block)] = rows
    # A cell the two partitions disagree on is one the outcome-blind chain and the provider's
    # key pass placed differently, which is the whole quantity test 3 exists to report.
    kp = out.get("kp_vs_segment", {})
    agree = sum(v["n"] for k, v in kp.items() if k.split("|")[0] == k.split("|")[1])
    total = sum(v["n"] for v in kp.values())
    out["partition_agreement"] = {"shots": total, "same_class": agree,
                                  "pct": _pct(agree, total)}
    # Test 1's verdict, derived from the two estimates rather than carried as a flag: PASS is
    # defined as both intervals covering zero, and a stored flag is free to disagree with them.
    dens = out.get("density", {})
    out["annotation_density_passes"] = all(dens[k]["covers_zero"] for k in
                                           ("rate_diff", "logit_coef") if k in dens)
    ph = out.get("vendor_phase", {})
    if "direct|1" in ph:
        out["direct_is_vendor_phase_1_pct"] = _pct(ph["direct|1"]["n"],
                                                   ph["direct|1"]["n"] + ph["direct|2"]["n"])
    return out


def throw_ins() -> dict:
    """Long throws measured against the two built populations, from probe_throw_ins.

    The paper is titled around set pieces, so the exclusion of throw-ins has to be a measurement
    rather than a design note. This is that measurement: a long-throw predicate on the throw's
    position, height and recorded length, swept rather than chosen, and the frontier run over the
    population it selects.

    Read `direct_obs_exp` on the throw-in row with its goal count in view. It rests on
    `direct_goals`, so its interval excluding 1.0 is not the same kind of statement as the corner
    one, and `direct_obs_exp_is_fragile` is carried so no consumer can quote it without the
    caveat."""
    p = _load("throw_in_probe.parquet")
    if p is None or not len(p):
        return {}
    grid = p[p["block"] == "grid"]
    comp = p[p["block"] == "comparison"]
    seg = p[p["block"] == "segment"]
    if not len(comp):
        return {}
    out: dict = {}
    if len(grid):
        best = _adopt_throw_in(grid)
        out["adopted_predicate"] = {
            "x_min": float(best["x_min"]), "length_min": float(best["len_min"]),
            "high_pass_only": bool(best["high_only"]), "n_selected": int(best["n"]),
            # Characterisation, never selection: this reads where the throw ended.
            "ends_in_box_pct": round(float(best["in_box_pct"]), 2)}
    out["by_population"] = {
        str(r["population"]): {
            "set_pieces": int(r["set_pieces"]), "shots": int(r["shots"]),
            "shots_per_100": round(float(r["shots_per_100"]), 2),
            "goals_per_100": round(float(r["goals_per_100"]), 2),
            "after_first_touch_shots_pct": round(float(r["aft_shot_pct"]), 2),
            "direct_shots": int(r["direct_shots"]),
            "direct_obs_exp": round(float(r["direct_obs_exp"]), 4),
            "direct_ci": [round(float(r["lo"]), 4), round(float(r["hi"]), 4)]}
        for _, r in comp.iterrows()}
    if len(seg):
        d = seg[seg["segment"] == "direct"]
        goals = int(d["goals"].iloc[0]) if len(d) else 0
        out["direct_goals"] = goals
        out["direct_obs_exp_is_fragile"] = bool(goals < MIN_CELL)
        out["by_segment"] = {str(r["segment"]): {"shots": int(r["shots"]),
                                                 "goals": int(r["goals"]),
                                                 "own_goals": int(r["own_goals"]),
                                                 "below_floor": bool(str(r["thin"]) != "")}
                             for _, r in seg.iterrows()}
    return out


def flank_cut() -> dict:
    """Where a flank free kick stops being a corner (run_flank_cut): the annex behind x >= 85.

    Per 5-yard band of the taker's x from 60, under the paper's predicate, with long corners as the
    reference: what a delivery produces with its match-clustered interval, how, and the attackers'
    three jobs at the kick. The two cuts carry below minus above over shared match draws. The rate
    per delivery is primary and the after-first-touch share sits beside it, because a far kick
    almost never shoots direct and its share climbs whatever the touch produces (definitions.md §8)."""
    b, c = _load("flank_cut_bands.parquet"), _load("flank_cut_contrasts.parquet")
    if b is None or c is None or not len(b):
        return {}
    rates = ("xg_direct", "p_shot", "p_direct", "p_aft", "p_prol", "p_clear", "won_fc", "aft_share",
             "cleared", "n_attack_cross", "n_stake_drop", "n_hold_back", "n_def_frontal")
    bands = {r["band"]: {"deliveries": int(r["n"]), "with_360": int(r["n_360"]),
                         **{k: {"point": round(float(r[k]), 5), "ci": _ci5(r[f"{k}_lo"], r[f"{k}_hi"])}
                            for k in ("xg_win", "xg_aft")},
                         **{k: round(float(r[k]), 4) for k in rates if k in r.index},
                         "fc_x_median": round(float(r["fc_x_median"]), 1)}
             for _, r in b.iterrows()}
    cuts = {}
    for cut, sub in c.groupby("cut"):
        measures = {}
        for _, r in sub.iterrows():
            ci = _ci5(r["lo"], r["hi"])
            measures[r["measure"]] = {"below_minus_above": round(float(r["below_minus_above"]), 5),
                                      "ci": ci, "covers_zero": None if ci is None else _covers(ci)}
        cuts[f"{cut:.0f}"] = {"n_below": int(sub["n_below"].iloc[0]),
                              "n_above": int(sub["n_above"].iloc[0]), **measures}
    return {"bands": bands, "cuts": cuts}


def _paired(r: pd.Series) -> dict:
    """A paired contrast with its 95% interval at five decimals and the side of zero it reads."""
    ci = _ci5(r["lo"], r["hi"])
    return {"delta": round(float(r["point"]), 5), "ci": ci,
            "covers_zero": None if ci is None else _covers(ci)}


def frame_gpu() -> dict:
    """GPU1, the frame network against the delivery rung (run_frame_gpu), loaded from the sixty
    tasks the plan fitted and never retrained. Per population: each seed's paired contrast with
    its MDE, the shuffled-frame control, the head-to-head against the full model, which reads the
    zones and is reported only, and the registered verdict as the driver's rule wrote it."""
    g = _load("frame_gpu.parquet")
    if g is None or not len(g):
        return {}
    out = {}
    for pop, sub in g.groupby("population"):
        by = dict(tuple(sub.groupby("block")))
        out[str(pop)] = {
            "seeds": {str(r["key"]): {**_paired(r), "mde80": round(float(r["mde80"]), 5),
                                      "below_margin": bool(r["below_margin"])}
                      for _, r in by["seed"].iterrows()},
            "shuffled_frame": _paired(by["control"].iloc[0]),
            "encoder_minus_full": _paired(by["head_to_head"].iloc[0]),
            "verdict": ast.literal_eval(by["verdict"].iloc[0]["params"]),
            "tasks": int(len(by.get("task", [])))}
    return out


def zone_comparison() -> dict:
    """M1, which partition of the box at the kick carries the plan (run_zone_comparison), and the
    two readings added after the registered run. Per population: the verdict as the driver's rule
    wrote it, each rival against F in normal form at 95% and at the Bonferroni level that decides
    with the pivotal one beside, the stability bounds by seed and the shuffled controls."""
    z, rob = _load("zone_comparison.parquet"), _load("zone_robustness.parquet")
    if z is None or not len(z):
        return {}
    out = {}
    for pop, sub in z.groupby("population"):
        if (sub["block"] == "skipped").any():
            e: dict = {"skipped": True}
        else:
            rf = sub[(sub["block"] == "refit") & sub["key"].astype(str).str.endswith("-F")]
            e = {"verdict": ast.literal_eval(sub[sub["block"] == "verdict"].iloc[0]["params"]),
                 "rivals": {str(r["key"]): {"delta": round(float(r["point"]), 5),
                                            "normal": _ci5(r["lo"], r["hi"]),
                                            "normal_bonferroni": _ci5(r["bonf_lo"], r["bonf_hi"]),
                                            "pivotal_bonferroni": _ci5(r["bonf_pivotal_lo"],
                                                                       r["bonf_pivotal_hi"]),
                                            "mde80": round(float(r["mde80"]), 5)}
                            for _, r in rf.iterrows()},
                 "stability": {str(k): [round(float(h), 5) for h in s.sort_values("seed")["hi"]]
                               for k, s in sub[sub["block"] == "stability"].groupby("key")},
                 "controls": {str(r["key"]): bool(r["clean"])
                              for _, r in sub[sub["block"] == "control"].iterrows()}}
        if rob is not None:
            r = rob[(rob["population"] == pop) & (rob["block"] == "robustness")]
            if len(r):
                e["robustness"] = ast.literal_eval(r.iloc[0]["params"])
        out[str(pop)] = e
    return out


def duel_prices(tag: str = "") -> dict:
    """C1, what each duel of a corner is worth zone by zone (run_duel_prices), on the switch's
    partition, which names the records: the decision and C1a seed by seed, and every zone's price on
    the whole grant for each duel, with its match-resampled interval, the what-if a club reads and the
    flag a zone under MIN_CELL duels carries."""
    stem = f"duel_prices_{ZONE_PARTITION}{tag}"
    p, d, s = (_load(f"{stem}{x}.parquet") for x in ("", "_decision", "_c1a"))
    if p is None or d is None or not len(d):
        return {}
    allp = p[p["sample"] == "all"]
    out: dict = {"decision": {str(k): _scalar(v) for k, v in d.iloc[0].items()},
                 "prices": {str(duel): {str(r["zone"]): {
                     "price": round(float(r["price"]), 5), "ci": _ci5(r["price_lo"], r["price_hi"]),
                     "whatif_xg_per_team_season": round(float(r["whatif_xg_per_team_season"]), 5),
                     "n_won": int(r["n_won"]), "n_lost": int(r["n_lost"]), "thin": bool(r["thin"])}
                     for _, r in sub.iterrows()} for duel, sub in allp.groupby("duel")}}
    if s is not None:
        out["c1a_by_seed"] = {str(int(r["seed"])): {
            "holds": None if pd.isna(r["holds"]) else bool(r["holds"]),
            "spearman": _scalar(r["spearman"]), "ci": _ci3(r["spearman_lo"], r["spearman_hi"])}
            for _, r in s.iterrows()}
    return out


def club_power(tag: str = "") -> dict:
    """C12, what a club can measure about its own corners (eda_club_power): the decision on C12a;
    each measure's spread and two intraclass correlations with their cluster-resampled intervals
    and, on a binary one, the share of the variance between teams; the split-half reliability; the
    corners per arm and the seasons a before-and-after needs at each lift; and the half-season
    route beside the formula."""
    rec = {n: _load(f"club_power_{n}{tag}.parquet")
           for n in ("decision", "measures", "power", "half_season", "reliability")}
    if any(v is None or not len(v) for v in rec.values()):
        return {}

    def interval(r: pd.Series, k: str) -> dict:
        return {"point": round(float(r[k]), 4),
                "ci": [round(float(r[f"{k}_lo"]), 4), round(float(r[f"{k}_hi"]), 4)]
                if np.isfinite([r[f"{k}_lo"], r[f"{k}_hi"]]).all() else None}

    measures = {}
    for _, r in rec["measures"].iterrows():
        m = {"mu": round(float(r["mu"]), 5), "sd": round(float(r["sd"]), 5),
             "icc_team": interval(r, "icc_team"),
             "icc_match_in_team": interval(r, "icc_match_in_team")}
        if "rho" in r.index and pd.notna(r["rho"]):
            m["rho"] = interval(r, "rho")
        measures[str(r["measure"])] = m
    power: dict = {}
    for _, r in rec["power"].iterrows():
        power.setdefault(str(r["measure"]), {})[str(float(r["lift"]))] = {
            "corners_per_arm": round(float(r["corners_per_arm"]), 1),
            "seasons_before_after": round(float(r["seasons_before_after"]), 2)}
    return {"decision": {str(k): _scalar(v) for k, v in rec["decision"].iloc[0].items()},
            "measures": measures, "power": power,
            "half_season": {str(r["measure"]): {
                "mde80_lift": round(float(r["half_season_mde80_lift"]), 3),
                "formula_lift": _scalar(r["formula_lift_at_half_season"])}
                for _, r in rec["half_season"].iterrows()},
            "reliability": {str(r["metric"]): {"full_r_sb": round(float(r["full_r_sb"]), 3),
                                               "full_ci": _ci3(r["full_lo"], r["full_hi"])}
                            for _, r in rec["reliability"].iterrows()}}


def _group_rates(g: pd.DataFrame) -> dict:
    """A group's corners per match and xG per corner by route, each pooled over the group's corners
    and matches, the way eda_top_teams pools them for its contrasts."""
    n = g["corners"].sum()
    return {"corners_per_match": round(float(n / g["matches"].sum()), 4),
            **{f"{r}_xg_per_corner": round(float(g[f"xg_{r}"].sum() / n), 5)
               for r in ("direct", "aft", "late")},
            "aft_xg_share": round(float(g["xg_aft"].sum() / g["xg_total"].sum()), 4)}


def top_teams(tag: str = "") -> dict:
    """Do the best sides, and the best corner sides, live after the first touch (eda_top_teams):
    the registered decisions, each holding only at every seed, the two contrasts they read at
    seed 0, group T over the season and group G on its split-half version, and each league's gap.

    Beside them, what the narrative reads and nothing decides: each group's secondary contrasts at
    seed 0 with their Benjamini-Hochberg q (G's on its split-half version only, its season ones being
    inflated by its choice on goals), group T's and the rest's pooled rates, and the split-half
    reliability of each per-team measure."""
    d, c = _load(f"top_teams_decisions{tag}.parquet"), _load(f"top_teams_contrasts{tag}.parquet")
    if d is None or c is None or not len(d):
        return {}
    dec = {str(k): ([round(float(x), 6) for x in v] if isinstance(v, (list, np.ndarray))
                    else _scalar(v)) for k, v in d.iloc[0].items()}
    seed0 = c[c["seed"] == 0]
    out: dict = {"decisions": dec, "primary": {}, "secondary": {}}
    for group, scope in (("T2", "season"), ("G", "split_half")):
        rows = seed0[(seed0["group"] == group) & (seed0["scope"] == scope)]
        r = rows[rows["role"] == "primary"]
        if len(r):
            r = r.iloc[0]
            out["primary"][group] = {"point": round(float(r["point"]), 5),
                                     "ci95": _ci5(r["lo95"], r["hi95"]),
                                     "ci95_normal": _ci5(r["lo95_normal"], r["hi95_normal"]),
                                     "mde80": round(float(r["mde80"]), 5),
                                     "p_perm": _scalar(r["p_perm"])}
        # p and q at six decimals: at four, a permutation p at its 1e-5 floor would publish as zero
        out["secondary"][group] = {str(s["metric"]): {"point": round(float(s["point"]), 5),
                                                      "ci95": _ci5(s["lo95"], s["hi95"]),
                                                      "p_perm": round(float(s["p_perm"]), 6),
                                                      "q_bh": round(float(s["q_bh"]), 6)}
                                   for _, s in rows[rows["role"] == "secondary"].iterrows()}
    b = _load(f"top_teams_by_league{tag}.parquet")
    if b is not None:
        out["by_league"] = {str(r["league"]): round(float(r["gap"]), 5) for _, r in b.iterrows()}
    t = _load(f"top_teams_table{tag}.parquet")
    if t is not None:
        out["rates"] = {"T2": _group_rates(t[t["in_T2"]]), "rest": _group_rates(t[~t["in_T2"]])}
    rel = _load(f"top_teams_reliability{tag}.parquet")
    if rel is not None:
        out["reliability"] = {str(r["metric"]): {"full_r_sb": round(float(r["full_r_sb"]), 3),
                                                 "full_ci": _ci3(r["full_lo"], r["full_hi"])}
                              for _, r in rel.iterrows()}
    return out


def tracking_curve(tag: str = "") -> dict:
    """TRK1, what more tracked matches would buy (run_tracking_curve), as the number of them needed:
    per statistic, TRK1 at every seed, seed 0's slope of log width on log N with its interval, and
    the N at which the half-width reaches the effect, on the fitted curve and on 1/sqrt(N). A
    statistic under the publication floor is recorded with its count as not estimable."""
    f = _load(f"tracking_curve_fit{tag}.parquet")
    if f is None or not len(f):
        return {}
    out = {}
    for stat, sub in f.groupby("stat"):
        if "estimable" in sub and not bool(sub["estimable"].iloc[0]):
            out[str(stat)] = {"estimable": False, "n_rows": int(sub["n_rows"].iloc[0]),
                              "matches": int(sub["full_n"].iloc[0])}
            continue
        s = sub.sort_values("seed")
        r = s.iloc[0]
        out[str(stat)] = {"estimable": True, "TRK1_by_seed": [bool(x) for x in s["TRK1"]],
                          "TRK1": bool(s["TRK1"].all()), "slope": round(float(r["slope"]), 4),
                          "slope_ci": [round(float(r["slope_lo"]), 4), round(float(r["slope_hi"]), 4)],
                          "effect": round(float(r["effect"]), 5), "full_n": int(r["full_n"]),
                          "n_star_fit": round(float(r["n_star_fit"]), 1),
                          "n_star_fit_ci": [round(float(r["n_star_fit_lo"]), 1),
                                            round(float(r["n_star_fit_hi"]), 1)],
                          "n_star_sqrt": round(float(r["n_star_sqrt"]), 1)}
    return out


def first_contact(kind: str = "corner") -> dict:
    """Who wins the first touch after a corner, and what kind of event that touch is.

    The type breakdown is the point: `won_first_contact` counts a player receiving the ball
    the same way it counts a duel a defender had to win, and almost every attacking win is
    filed as a `Ball Receipt*`.

    That receipt share is NOT a contested/uncontested split, and must not be read as one.
    Since `first_contact` drops the loser's `Aerial Lost` row, an attacker who WINS a header
    is credited with the following on-ball event, which is usually a receipt too. Whether
    the touch was contested is not recoverable from the event type alone."""
    d = _load("eda_deliveries.parquet")
    if d is None:
        return {}
    c = d[d["set_piece_type"] == kind]
    known = c[c["won_first_contact"].notna()]
    if not len(known):
        return {}
    won = known[known["won_first_contact"].astype(bool)]
    receipt = won["first_contact_type"].isin(["Ball Receipt*"])
    return {
        "corners": int(len(c)),
        "with_determinate_first_contact": int(len(known)),
        "won_by_attack_pct": _pct(len(won), len(known)),
        "receipt_share_of_attack_wins_pct": _pct(int(receipt.sum()), len(won)),
        "median_time_to_first_contact_s": round(float(known["time_to_first_contact_s"].median()), 3),
        "by_first_contact_type": known["first_contact_type"].value_counts().head(6).to_dict(),
    }


def first_contact_frame(tag: str = "") -> dict:
    """What the 360 AT THE FIRST CONTACT adds about who wins the second ball (run_first_contact).

    The estimand is conditional on a frame read AFTER the delivery, so nothing here is a feature of
    the published model: it prices what a camera at T1 would be worth, not what the plan at the kick
    knows. Every decision carries its four readings. The pivotal form is the one the PRE-REGISTERED
    rule decides on, and a rule fixed before the data does not move after it; the normal form,
    which decides everywhere else in the record, travels beside it, with the percentile form and
    the Bonferroni level over the six decisions. The verdicts are not parsed back out of the
    artefact's text column: they are re-derived here with the driver's own `decide`, so the record
    and the run cannot drift apart.

    Where phase one leaves V1 undecided, the pre-registration's phase two refits the strong learner
    too, into a `_gbt` record that repeats every phase-one decision on the same draws and adds the
    strong learner's. When that record exists it is the one read, and V1 closes on it."""
    phase_two = _load(f"first_contact{tag}_gbt.parquet")
    d = phase_two if phase_two is not None else _load(f"first_contact{tag}.parquet")
    if d is None or not len(d):
        return {}
    step0 = {r["key"].split("/", 1)[1]: int(r["n"]) for _, r in d[d["block"] == "step0"].iterrows()
             if r["key"].startswith("all/")}
    pop = d[d["block"] == "population"].set_index("key")
    ver = d[d["block"] == "refit"].to_dict("records")
    st = d[d["block"] == "stability"]
    stable = {k: bool(sub["below_zero"].nunique() == 1) for k, sub in st.groupby("key")}
    decisions = {}
    for r in ver:
        decisions[r["key"]] = {
            "delta": round(float(r["point"]), 5),
            "pivotal": _ci5(r["lo"], r["hi"]), "percentile": _ci5(r["pct_lo"], r["pct_hi"]),
            "normal": _normal(r["point"], r["pct_lo"], r["pct_hi"]),
            "bonferroni": _ci5(r["bonf_lo"], r["bonf_hi"]),
            "below_zero": bool(r["below_zero"]), "stable_across_seeds": stable.get(r["key"], None),
            "refit_widening": round(float(r["width_ratio"]), 3)}
    no_taker = {r["key"]: {"delta": round(float(r["point"]), 5),
                           "pivotal": _ci5(r["lo"], r["hi"]), "below_zero": bool(r["below_zero"])}
                for _, r in d[d["block"] == "refit_no_taker"].iterrows()}
    # The five per-seed bounds, not just the boolean they collapse to. Prose that says a verdict is
    # stable is claiming something about these five numbers, and a boolean cannot be bound to it.
    bounds = {k: [round(float(v), 5) for v in sub.sort_values("seed")["hi"]]
              for k, sub in st.groupby("key")}
    return {
        "corners": int(pop.loc["corners", "n"]),
        "base_rate": round(float(pop.loc["corners", "point"]), 4),
        "near_spot_share_pct": round(100 * float(pop.loc["near_spot_share", "point"]), 1),
        "coverage": step0,
        "t1_360_coverage_pct": _pct(step0.get("t1_has_360", 0), step0.get("grant", 0)),
        "arms_log_loss": {r["key"]: round(float(r["point"]), 5)
                          for _, r in d[d["block"] == "arm"].iterrows()},
        "decisions": decisions,
        "stability_upper_bounds": bounds,
        "no_taker_reading": no_taker,
        "controls_all_clean": bool(d[d["block"] == "control"]["clean"].astype(bool).all()),
        "mde_80_adjusted": {r["key"].split("/", 1)[1]: round(float(r["hi"]), 5)
                            for _, r in d[d["block"] == "power"].iterrows()
                            if np.isfinite(r["hi"])},
        "phase_two": phase_two is not None,
        "verdicts": {k: v["state"] for k, v in
                     _fc_decide(ver, stable, gbt_refit=phase_two is not None).items()},
        "verdict_rule": "pivotal 95% wholly below zero AND the decisive bound keeps its side "
                        "across the five stability seeds; Bonferroni never decides",
    }


def positive_control(tag: str = "") -> dict:
    """Can this harness see signal at all (run_positive_control)?

    A record made mostly of nulls owes the reader a demonstration that its method could have found
    something. The target is who wins the FIRST contact, which the delivery frame ought to inform,
    and the arms are the same nested ladder the value stage publishes. The long-corner-only reading
    rides along because the crosses after a short corner are another delivery, struck from the wing
    after a few seconds, and a control that passed on them alone would be passing on that
    difference of population; a free kick is never short, so the flank's record carries none."""
    d = _load(f"positive_control{tag}.parquet")
    if d is None or not len(d):
        return {}
    pop = d[d["block"] == "population"].set_index("key")

    def deltas(block: str) -> dict:
        return {r["key"]: {"delta": round(float(r["point"]), 5),
                           "ci": _ci5(r["lo"], r["hi"]), "below_zero": bool(r["below_zero"])}
                for _, r in d[d["block"] == block].iterrows()}

    ctrl = {r["key"].split("/", 1)[1]: {"delta": round(float(r["point"]), 5),
                                        "ci": _ci5(r["lo"], r["hi"]),
                                        "clean": bool(r["clean"])}
            for _, r in d[d["block"] == "control"].iterrows()}
    main, long_only = deltas("delta"), deltas("delta_long_only")
    return {
        "corners": int(pop.loc["corners", "n"]),
        "attack_wins_first_contact_pct": round(100 * float(pop.loc["corners", "point"]), 1),
        "arms_log_loss": {r["key"]: round(float(r["point"]), 5)
                          for _, r in d[d["block"] == "arm"].iterrows()},
        "decisions": main,
        **({"long_corners_only": long_only} if long_only else {}),
        "shuffled_controls": ctrl,
        "verdicts": _pc_decide(main, ctrl["shuffled_global"]["clean"]),
        "verdict_rule": "P1 and P2 each need their interval below zero AND a clean global shuffle; "
                        "the within-match shuffle is reported and never decides",
    }


def aim_zone(tag: str = "") -> dict:
    """Does the delivery 360 carry the plan, and is the end-zone map biased (run_aim_zone)?

    The confirmation sample decides and the development league is reported beside it, never
    pooled. The pre-registered rule reads the pivotal form, as in `first_contact_frame`, and the
    normal form travels beside it. The verdicts are re-derived with the driver's own `decide` off
    the refit rows, the per-seed bounds and the control, as `first_contact_frame` does. The
    robustness block is counted rather than listed: the claim it supports is a pattern across
    grids and value measures, and a count of cells on each side of zero is what a sentence about
    it can cite."""
    d = _load(f"aim_zone{tag}.parquet")
    if d is None or not len(d):
        return {}

    def sample(name: str) -> dict:
        s = d[d["sample"] == name]
        pop = s[s["block"] == "population"].iloc[0]
        out = {"long_corners": int(pop["n"]), "attack_won_pct": round(100 * float(pop["point"]), 1),
               "leagues": pop["params"],
               "end_zones": {r["key"]: int(r["n"]) for _, r in s[s["block"] == "zone_count"].iterrows()}}
        if (s["block"] == "skipped").any():
            return out
        out["aim_log_loss"] = {r["key"]: round(float(r["point"]), 4)
                               for _, r in s[s["block"] == "aim"].iterrows()}
        out["aim_top1_pct"] = {r["key"]: round(100 * float(r["top1"]), 1)
                               for _, r in s[s["block"] == "aim"].iterrows()}
        out["statistics"] = {r["key"]: {
            "point": round(float(r["point"]), 5), "pivotal": _ci5(r["lo"], r["hi"]),
            "percentile": _ci5(r["pct_lo"], r["pct_hi"]), "normal": _normal(r["point"], r["pct_lo"], r["pct_hi"]),
            "bonferroni": _ci5(r["bonf_lo"], r["bonf_hi"]),
            "fixed": _ci5(r["fixed_lo"], r["fixed_hi"]), "refit_widening": round(float(r["width_ratio"]), 3),
            "draws": int(r["n"])} for _, r in s[s["block"] == "refit"].iterrows()}
        out["honest_map_reads"] = s[s["block"] == "equivalence"].iloc[0]["params"]
        bfc = s[s["block"] == "by_first_contact"]
        out["by_first_contact"] = {
            side: {r["key"]: {"point": round(float(r["point"]), 5), "fixed": _ci5(r["lo"], r["hi"])}
                   for _, r in sub.iterrows()}
            for side, sub in bfc[bfc["grid"] != "no_contact"].groupby("grid")}
        # The deliveries no one touched carry a count and no map, so the count goes out on its own.
        nc = bfc[bfc["grid"] == "no_contact"]
        if len(nc):
            out["by_first_contact"]["no_contact"] = int(nc.iloc[0]["n"])
        cal = s[s["block"] == "calibration"]
        out["calibration_max_gap_pct"] = {
            zone: round(100 * float((sub["pred"] - sub["obs"]).abs().max()), 1)
            for zone, sub in cal.groupby("grid")}
        return out

    conf = d[d["sample"] == "confirmation"]
    st = conf[conf["block"] == "stability"]
    stable = {k: bool(sub["below_zero"].nunique() == 1) for k, sub in st.groupby("key")}
    ctrl = conf[conf["block"] == "control"]
    rob = conf[conf["block"] == "robustness"]
    out = {"confirmation": sample("confirmation"), "development": sample("development")}
    if len(ctrl):
        c = ctrl.iloc[0]
        out["shuffled_kick_block_d1"] = {"point": round(float(c["point"]), 5),
                                         "ci": _ci5(c["lo"], c["hi"]), "clean": bool(c["clean"])}
        out["stability_upper_bounds"] = {k: [round(float(v), 5) for v in sub.sort_values("seed")["hi"]]
                                         for k, sub in st.groupby("key")}
        out["league_in_design"] = {r["key"]: {"point": round(float(r["point"]), 5),
                                              "fixed": _ci5(r["lo"], r["hi"])}
                                   for _, r in conf[conf["block"] == "league_in_design"].iterrows()}
        out["robustness_cells"] = int(len(rob) // 3)
        out["robustness_excluding_zero"] = {
            k: int(((sub["hi"] < 0) | (sub["lo"] > 0)).sum()) for k, sub in rob.groupby("key")}
        out["verdicts"] = {k: v["state"] for k, v in _az_decide(
            conf[conf["block"] == "refit"].to_dict("records"), stable, bool(ctrl.iloc[0]["clean"])).items()}
    # A league whose inferred aim never lands far post has no honest map, so its D2 is NaN; it is
    # written as None because NaN never compares equal and would mark the record stale forever.
    out["by_league"] = {lg: {r["key"]: {"point": round(float(r["point"]), 5)
                                        if np.isfinite(r["point"]) else None,
                                        "fixed": _ci5(r["lo"], r["hi"])} for _, r in sub.iterrows()}
                        for lg, sub in d[d["block"] == "league"].groupby("sample")}
    out["verdict_rule"] = ("A1: D1 pivotal 95% wholly below zero, steady across the stability seeds, "
                           "and a clean shuffled kick block; A2: the same for D2 without the control; "
                           "Bonferroni and the development league never decide")
    return out


def short_corner_sensitivity() -> dict:
    """Does the headline delivery-against-360 contrast rest on the corners played short?

    A short or ground corner is delivered by its first cross (s01), so the modelling table holds the
    cross. The contrast is scored on the table, on its long corners alone, on the crosses alone around
    the window s01 fixes, and on every corner at its own kick, the rows the table would hold if the
    reception of a short pass were its first touch.

    Published with its own arms rather than beside §validation's figure, because that one chose its
    regularisation on the table holding corners and free kicks together: the two are not the same
    estimator and reading them as a pair would invite a comparison that is not there. What this
    block supports is the internal one."""
    s = _load("short_corner_sensitivity.parquet")
    if s is None or not len(s):
        return {}
    arms = s[s["block"].isin(("fixed_C", "tuned"))]
    out: dict = {"short_share_pct": round(float(arms["short_share_pct"].iloc[0]), 1), "by_arm": {}}
    for arm, sub in arms.groupby("block"):
        out["by_arm"][arm] = {r["key"]: {"delta": round(float(r["point"]), 5),
                                         "ci": _ci5(r["lo"], r["hi"]),
                                         "corners": int(r["n"]),
                                         "excludes_zero": bool(r["excludes_zero"]),
                                         "delivery_wins": bool(r["delivery_wins"])}
                              for _, r in sub.iterrows()}
    # The short corners and their crosses, with what the ones no cross reached in time produced.
    delay = s[s["block"] == "cross_delay"].set_index("key")
    primary = arms[(arms["block"] == "fixed_C") & arms["primary"].astype(bool)]
    out["short_cross"] = {"short_corners": int(delay.loc["short_corners", "n"]),
                          "followed_by_a_cross": int(delay.loc["crossed", "n"]),
                          "in_window": int(delay.loc["in_window", "n"]),
                          "window_s": float(delay.loc["window_s", "point"]),
                          "delay_median_s": round(float(delay.loc["delay_median_s", "point"]), 2),
                          "delay_p90_s": round(float(delay.loc["delay_p90_s", "point"]), 2),
                          "uncrossed": {k: int(delay.loc[f"uncrossed_{k}", "n"])
                                        for k in ("corners", "shots", "goals", "goal_corners")},
                          "delivery_wins_at_primary": bool(primary["delivery_wins"].iloc[0])}
    fixed = out["by_arm"].get("fixed_C", {})
    # The ratios are the figures the sensitivity exists to report: the verdict survives, and the size
    # moves with the rows. Guarded because a zero denominator would publish an infinity as a finding.
    if {"all", "long_only"} <= set(fixed):
        long_delta = fixed["long_only"]["delta"]
        out["all_over_long_only"] = (round(fixed["all"]["delta"] / long_delta, 2)
                                     if long_delta else None)
        out["verdict_survives_dropping_short"] = bool(fixed["long_only"]["excludes_zero"])
    if {"all", "kick_anchored"} <= set(fixed):
        all_delta = fixed["all"]["delta"]
        out["kick_anchored_over_all"] = (round(fixed["kick_anchored"]["delta"] / all_delta, 2)
                                         if all_delta else None)
    return out


def noise_floor() -> dict:
    """What shuffling a block that carries nothing costs, read off blocks of pure noise (run_noise_floor).

    §validation's permuted control charges each block the cost of shuffling it, and that cost has a floor
    even for a block with no information. The floor is built by construction, on independent
    standard-normal columns, for each width a block of the design has, and `by_group` names each block
    with the floor of its own width: its permuted cost is told from noise only above the 95th percentile."""
    f = _load("noise_floor.parquet")
    if f is None or not len(f):
        return {}

    def row(r: pd.Series) -> dict:
        return {"n_cols": int(r["n_cols"]), "mean": round(float(r["mean"]), 5), "sd": round(float(r["sd"]), 5),
                "p95": round(float(r["p95"]), 5), "max": round(float(r["max"]), 5)}
    return {"draws": int(f["n_draws"].iloc[0]), "repeats": int(f["n_repeats"].iloc[0]),
            "by_width": {k: row(r) for k, r in f[f["block"] == "floor"].set_index("key").iterrows()},
            "by_group": {k: row(r) for k, r in f[f["block"] == "group"].set_index("key").iterrows()}}


def drop_position(tag: str = "") -> dict:
    """Does the staked-win premium survive holding the drop position equal (run_drop_position)?

    §situation prices winning the second ball from a staked zone against winning it on the run, and
    the hypothesis that contrast answers promises value against rescues from the SAME position. The
    raw reading does not hold position equal; the other two do, each in a way with a different
    weakness, so both are published. The stratified arm is transparent but can only use
    rows sitting in a cell that holds both states, and that count travels with it; the adjusted arm
    keeps every row but imposes a shape on how position matters."""
    d = _load(f"drop_position{tag}.parquet")
    if d is None or not len(d):
        return {}
    out: dict = {"planned": int(d["n_planned"].iloc[0]), "rescued": int(d["n_rescued"].iloc[0]),
                 "rows": int(d["n"].iloc[0]), "by_cell": {}}
    for side, sub in d.groupby("cell_yd"):
        key = f"{float(side):.0f}yd"
        out["by_cell"][key] = {
            "usable_rows_stratified": int(sub["n_usable_stratified"].iloc[0]),
            **{r["block"]: {"gap": round(float(r["point"]), 5),
                            "ci": _ci5(r["lo"], r["hi"]),
                            "covers_zero": None if pd.isna(r["covers_zero"])
                            else bool(r["covers_zero"])}
               for _, r in sub.iterrows()}}
    # The reading the argument turns on, lifted out so prose does not have to reach into by_cell:
    # every adjusted reading covering zero is the claim, and one without an interval decides nothing.
    adj = [v for c in out["by_cell"].values() for k, v in c.items()
           if k in ("stratified", "adjusted") and isinstance(v, dict)]
    out["every_adjusted_reading_covers_zero"] = bool(adj) and all(v["covers_zero"] is True
                                                                  for v in adj)
    # H3's claim is one-sided: the premium survives only where a reading clears zero upwards. A
    # reading that excludes zero BELOW it inverts the premium and is counted apart.
    out["no_adjusted_reading_keeps_the_premium"] = bool(adj) and all(
        v["ci"] is not None and v["ci"][0] <= 0 for v in adj)
    out["adjusted_readings_inverting_it"] = sum(v["ci"] is not None and v["ci"][1] < 0 for v in adj)
    out["adjusted_readings"] = len(adj)
    out["raw_covers_zero"] = next(iter(out["by_cell"].values()))["raw"]["covers_zero"]
    return out


def team_shrinkage(tag: str = "") -> dict:
    """What survives of the per-team rate once sampling noise is discounted (run_team_shrinkage).

    §repeatability already bars ranking teams on this rate, and this is the other half of that
    sentence: not "the measure is noisy" but "here is what the named case is worth once the noise
    is taken out". The rank interval is the figure to quote, because it is the one the raw table
    hides. The verdicts are the driver's, as its own rule wrote them into the artefact: re-deriving
    them here would import PyMC, which the record does not otherwise need. An M1 that did not
    converge is published as not reported, with its diagnostics, as the pre-registration says."""
    d = _load(f"team_shrinkage{tag}.parquet")
    if d is None or not len(d):
        return {}

    def block(name: str) -> dict:
        return {r["key"]: (None if pd.isna(r.get("point")) else round(float(r["point"]), 4))
                for _, r in d[d["block"] == name].iterrows()}

    pop = d[d["block"] == "population"].set_index("key")
    base = {"teams": int(pop.loc["teams", "n"]), "goals": int(pop.loc["goals", "n"]),
            "corners": int(pop.loc["corners", "n"]),
            "pooled_rate": round(float(pop.loc["teams", "point"]), 4),
            "diagnostics": block("diagnostics")}
    if not base["diagnostics"].get("M1/converged"):
        return {**base, "reported": False}
    case = block("case")
    named = {}
    for who in NAMED_CASES:
        vals = {k.split("/", 1)[1]: v for k, v in case.items() if k.startswith(f"{who}/")}
        if vals:
            named[who] = {"raw_rate": vals.get("raw_rate"), "posterior_rate": vals.get("p_mean"),
                          "posterior_hdi": [vals.get("p_lo"), vals.get("p_hi")],
                          "a_team_hdi": [vals.get("a_team_lo"), vals.get("a_team_hi")],
                          "rank_median": vals.get("rank_median"),
                          "rank_95": [vals.get("rank_lo"), vals.get("rank_hi")],
                          "p_top5": vals.get("p_top5")}
    shr = d[d["block"] == "shrinkage"].set_index("key")
    nul = d[d["block"] == "null"].set_index("key")
    # The two counts are what make the p-value readable: a null of k converged fits cannot go below
    # 1/(k+1), and that floor travels with it. `p_perm_all` keeps the fits the gate dropped, never
    # decides, and shows which way dropping them moved the p-value.
    null = {"n_perm": int(nul.loc["n_perm", "n"]),
            "n_converged": int(nul.loc["n_converged", "n"]),
            "sigma_team_real": round(float(nul.loc["sigma_team_real", "point"]), 4),
            "p_perm": round(float(nul.loc["p_perm", "point"]), 4),
            "p_floor": round(float(nul.loc["p_floor", "point"]), 4),
            "p_perm_all_including_unconverged": round(float(nul.loc["p_perm_all", "point"]), 4)}
    return {
        **base, "reported": True,
        "cases": named,
        "p_first_above_second": case.get("p_first_above_second"),
        "shrinkage": {
            "spearman_raw_vs_posterior": round(float(shr.loc["spearman_raw_vs_posterior",
                                                             "point"]), 4),
            "teams_moving_over_10_places": int(shr.loc["teams_moving_over_10_places", "n"])},
        "sensitivity": block("sensitivity"),
        "permutation_null": null,
        "verdicts": {r["key"]: r["params"]
                     for _, r in d[d["block"] == "verdict"].iterrows()},
    }


def teams() -> dict:
    """Per-team output on corners, and the spread that motivates the question.

    Reported as ILLUSTRATION, never as a ranking: §repeatability shows the reliability of this
    rate sits below the Koo & Li floor, so the two named cases exist to show the channel is not
    a property of strong teams, not to order clubs."""
    t = _load("eda_teams_corner.parquet")
    if t is None:
        return {}
    col = "aft_goals_per100_for"
    top = t.nlargest(12, col)
    named = {}
    for name in NAMED_CASES:
        r = t[t["team"] == name]
        if len(r):
            r = r.iloc[0]
            named[name] = {"pos": int(r["pos"]), "corners_for": int(r["corners_for"]),
                           "goals_for": int(r["goals_for"]),
                           # `goals_for` is the goals of the deliveries, the denominator of every share
                           # here; the whole of the team's corner goals adds those after a short corner
                           # no cross reached, which have no delivery (run_short_corner_sensitivity).
                           "goals_for_all": int(r["goals_for_all"]),
                           "aft_goals_for": int(r["aft_goals_for"]),
                           "aft_goals_per100_for": round(float(r[col]), 3),
                           "aft_goal_share_for": round(float(r["aft_goal_share_for"]), 4),
                           # What the case could and could not resolve, beside what it shows.
                           "aft_binom_p": round(float(r["aft_binom_p"]), 4),
                           "aft_power_at_n": round(float(r["aft_power_at_n"]), 3),
                           "aft_goals_for_80pct_power": int(r["aft_goals_for_80pct_power"])}
    # The spread is the argument: if this were a luxury of good teams it would track the table.
    bottom_half = (top["pos_pct"] > 0.5).sum()
    return {
        "n_teams": int(len(t)),
        "median_aft_goals_for": float(t["aft_goals_for"].median()),
        "max_aft_goals_for": int(t["aft_goals_for"].max()),
        "min_aft_goals_for": int(t["aft_goals_for"].min()),
        "top12_finishing_in_bottom_half_pct": _pct(int(bottom_half), len(top)),
        # The rate each named case is tested against, pooled over every team in the table. It
        # reproduces `frontier.headline.share_goals_after_first_touch` from the shot grain, which
        # is two independent routes to one number rather than one number written twice.
        "aft_share_pooled": round(float(t["aft_share_pooled"].iloc[0]), 4),
        "named": named,
    }


def counter(tag: str = "") -> dict:
    """The transition arm, and why no window rescues it.

    Contiguity is the column that matters: at every window most of the qualifying defending
    shots are already in a different possession, and past the shortest windows widening makes
    it worse. `outranked` is what no window fixes: the events whose delivery produced its own
    attacking shot, which the outcome ladder ranks above any counter."""
    c = _load(f"counter_sweep{tag}.parquet")
    if c is None or not len(c):
        return {}
    by_window = {}
    for _, r in c.iterrows():
        by_window[f"{float(r['window_s']):.0f}s"] = {
            "events": int(r["events"]), "rate_pct": round(100 * float(r["rate"]), 3),
            "possession_contiguity_pct": round(100 * float(r["contiguity"]), 2),
            "lookahead_pct": round(100 * float(r["lookahead"]), 2),
            "outranked": int(r["outranked"])}
    # The verdict is the driver's own pre-registered rule, read off the column it persists:
    # n >= MIN_CELL AND contiguity AND lookahead, all three. Recomputing it here from one of them
    # would publish a laxer rule than the one `sweep_counter` declares.
    return {"by_window": by_window,
            "contiguity_bar_pct": 100 * MIN_CONTIGUITY,
            "any_window_clears_the_bar": bool(c["passes"].astype(bool).any())}


def repeatability(kind: str = "corner") -> dict:
    """argument.md Q3: dispersion and split-half reliability.

    Both intervals are over team draws inside each league, and `full_ci` is the one beside the
    Spearman-Brown figure prose leads on (`eda_repeatability.split_half`). The dispersion verdict
    that decides is the null at each league's own probability, carried where the record holds it;
    the single probability over every league rides beside."""
    rel, disp = _load(f"eda_reliability_{kind}.parquet"), _load(f"eda_dispersion_{kind}.parquet")
    out: dict = {}
    if rel is not None:
        r = rel.set_index("metric")
        for m in ("aft_goals_per100", "corners_per_match"):
            if m in r.index:
                out[m] = {"half_r": round(float(r.loc[m, "half_r"]), 3),
                          "half_ci": _ci3(r.loc[m, "half_lo"], r.loc[m, "half_hi"]),
                          "full_r_spearman_brown": round(float(r.loc[m, "full_r_sb"]), 3),
                          "full_ci": _ci3(r.loc[m, "full_lo"], r.loc[m, "full_hi"])}
    if disp is not None:
        d = disp.set_index("metric")
        key = "aft_goals_for/corners_for"
        if key in d.index:
            out["dispersion_" + key] = {"p_value": round(float(d.loc[key, "p_value"]), 4),
                                        "excess": bool(d.loc[key, "excess"])}
            if "p_value_by_league" in d.columns:
                out["dispersion_" + key] |= {
                    "p_value_by_league": round(float(d.loc[key, "p_value_by_league"]), 4),
                    "excess_by_league": bool(d.loc[key, "excess_by_league"])}
    return out


def validation(name: str = "validation.parquet") -> dict:
    """Baseline ladder and ablation deltas per population, from run_validation's record.

    The figures prose quotes most, so counts such as how many groups cover zero are derived here
    rather than left to be read off a table. No delivery-over-360 ratio is emitted: its denominator
    is the 360 step, whose corner interval covers zero, so the quotient is unstable by construction.
    `delivery_vs_360` carries the same claim as a paired contrast, with its fixed-fit interval and,
    under `refit`, the readings of run_refit_draws.

    Every refit interval is read in normal form, the point plus or minus half the refit width,
    and that is what decides; the pivotal (`adjusted_ci`) and percentile (`refit_ci`) forms travel
    beside it (s03_value.step_refit_interval)."""
    v = _load(name)
    if v is None:
        return {}
    draws = _draws_by_contrast()
    out: dict = {}
    for pop, sub in v.groupby("population"):
        ctl = sub[sub["block"] == "control"]
        lad = {r["key"]: r for _, r in sub[sub["block"] == "ladder"].iterrows()}
        entry: dict = {"n_ladder_rungs": len(lad), "ladder": {
            k: {"log_loss": round(float(r["log_loss"]), 4),
                "step": None if pd.isna(r["delta"]) else round(float(r["delta"]), 4),
                "step_ci": None if pd.isna(r["delta_lo"]) else
                [round(float(r["delta_lo"]), 4), round(float(r["delta_hi"]), 4)]}
            for k, r in lad.items()}}
        # Fixed-margin non-inferiority, FDA 2016 guidance. The margin is ANCHORED on the
        # reference effect this same run measured, never imported: M1 is the lower bound of
        # the delivery step and M2 a tenth of it, and the test is one-sided on the 360 step's
        # upper bound. Two-sided equivalence rides along because on corners it does NOT hold,
        # and a non-inferiority claim that quietly implied it would be the dishonest version.
        if ("delivery" in lad and "full" in lad and not pd.isna(lad["delivery"]["delta_lo"])
                and not pd.isna(lad["full"]["delta_lo"])):
            m1 = float(lad["delivery"]["delta_lo"])
            m2, lo, hi = m1 / 10, float(lad["full"]["delta_lo"]), float(lad["full"]["delta_hi"])
            # Tested on the refit NORMAL bound when this run measured one. The fixed-fit
            # interval understates the spread (by how much is `refit.width_ratio`), and a
            # non-inferiority claim that leans on the narrow bound is the claim helping itself.
            rf = sub[sub["block"] == "refit"]
            if len(rf):
                r0 = rf.iloc[0]
                lo, hi = _normal(r0["delta"], r0["delta_lo"], r0["delta_hi"])
            # With M1 not above zero there is no reference effect to preserve a fraction of, and
            # M2 < 0 makes both booleans true or false by construction rather than by measurement:
            # a population whose delivery step does not clear zero lands here. Undefined is
            # published as undefined, never as a verdict.
            defined = m1 > 0
            entry["non_inferiority"] = {
                "m1": round(m1, 4), "m2": round(m2, 5),
                "step_ci": [round(lo, 5), round(hi, 5)],
                "ci_is_refit_normal": bool(len(rf)),
                "margin_defined": defined,
                "adds_less_than_m2": bool(hi < m2) if defined else None,
                # M2 was pre-specified at a tenth of M1 and is NOT moved now that the bound is
                # known. The fraction travels beside the verdict so a reader can apply the
                # margin they would have chosen (the FDA's worked examples usually preserve
                # half) without this file quietly choosing it for them after the fact.
                "upper_as_fraction_of_m1": round(hi / m1, 3) if defined else None,
                "two_sided_equivalent": bool(-m2 <= lo and hi <= m2) if defined else None}
            # The normal bound assumes symmetric draws; the re-centred percentile bound drops
            # that assumption, and the verdict is only as firm as their agreement.
            rd = draws.get((pop, "step_360"))
            if rd is not None and defined:
                upper = _refit_reading(*rd)["centred_ci"][1]
                entry["non_inferiority"]["adds_less_than_m2_centred"] = bool(upper < m2)
        h2h = sub[sub["block"] == "head_to_head"]
        if len(h2h):
            r = h2h.iloc[0]
            entry["delivery_vs_360"] = {
                "delivery_log_loss": round(float(r["log_loss"]), 4),
                "only_360_log_loss": round(float(r["real_log_loss"]), 4),
                "delta": round(float(r["delta"]), 4),
                "ci": [round(float(r["delta_lo"]), 4), round(float(r["delta_hi"]), 4)],
                "covers_zero": bool(r["delta_lo"] <= 0 <= r["delta_hi"]),
                "p_wilcoxon": round(float(r["p_wilcoxon"]), 4)}
            rd = draws.get((pop, "head_to_head"))
            if rd is not None:
                ref = _refit_reading(*rd)
                width = float(r["delta_hi"] - r["delta_lo"])
                entry["delivery_vs_360"]["refit"] = {
                    k: ref[k] for k in ("normal_ci", "adjusted_ci", "refit_ci", "n_boot")}
                entry["delivery_vs_360"]["refit"] |= {
                    "width_ratio": round((ref["refit_ci"][1] - ref["refit_ci"][0]) / width, 2),
                    "covers_zero": _covers(ref["normal_ci"])}
        ab = sub[(sub["block"] == "ablation") & (sub["key"] != "ALL")]
        if len(ab):
            entry["ablation"] = {
                r["key"]: {"delta": round(float(r["delta"]), 4),
                           "ci": [round(float(r["delta_lo"]), 4), round(float(r["delta_hi"]), 4)],
                           "covers_zero": bool(r["delta_lo"] <= 0 <= r["delta_hi"])}
                for _, r in ab.iterrows()}
            # Whether a block is a finding is read on ITS OWN refit interval, in normal form, when
            # the run measured one for every block. The widening is not one factor across blocks,
            # so the step's ratio cannot stand in for it, and the fixed `ci` stays beside it.
            abr = {r["key"]: r for _, r in sub[sub["block"] == "ablation_refit"].iterrows()}
            if abr and set(abr) >= set(entry["ablation"]):
                for k, g in entry["ablation"].items():
                    a = abr[k]
                    normal = _normal(a["delta"], a["delta_lo"], a["delta_hi"])
                    g.update({"normal_ci": normal,
                              "adjusted_ci": _ci5(a["delta_lo"], a["delta_hi"]),
                              "refit_ci": _ci5(a["ll_lo"], a["ll_hi"]),
                              "width_ratio": round(float(a["brier"]), 2),
                              "covers_zero": _covers(normal)})
                entry["ablation_interval"] = "refit_normal"
            else:
                entry["ablation_interval"] = "fixed_oof"
            entry["n_ablation_groups"] = int(len(ab))
            entry["n_groups_covering_zero"] = int(sum(g["covers_zero"]
                                                      for g in entry["ablation"].values()))
        pwr = sub[sub["block"] == "power"].sort_values("delta")
        if len(pwr):
            reach = pwr[pwr["accuracy"] >= 0.80]
            mde80 = None if reach.empty else round(float(reach["delta"].iloc[0]), 4)
            entry["power"] = {
                "curve": {f"{float(r['delta']):.4f}": round(float(r["accuracy"]), 2)
                          for _, r in pwr.iterrows()},
                "mde_80": mde80}
            # The curve is built on fixed-fit draws, so it overstates power and the MDE it
            # gives is too small. The adjusted one is what prose quotes; the raw one stays
            # beside it so the correction is visible rather than applied out of sight.
            rf = sub[sub["block"] == "refit"]
            if len(rf) and mde80 is not None:
                entry["power"]["mde_80_adjusted"] = round(mde80 * float(rf.iloc[0]["brier"]), 4)
        het = {r["key"]: r for _, r in sub[sub["block"] == "heterogeneity"].iterrows()}
        if het:
            entry["heterogeneity"] = {
                "blp_slope": round(float(het["blp"]["delta"]), 4),
                "blp_p": round(float(het["blp"]["p_wilcoxon"]), 4),
                "gates_gap": round(float(het["gates"]["delta"]), 4),
                "gates_p": round(float(het["gates"]["p_wilcoxon"]), 4),
                "n_reps": int(het["blp"]["n_test"]),
                "verdict": str(het["blp"]["verdict"]),
                "actionable": str(het["gates"]["verdict"]) == "actionable"}
            # The placebo permutes the proxy's features on the same partitions. Clean means neither
            # permutation reaches the Bonferroni 0.025 the real verdict uses, on BLP or on GATES.
            pl = {k[len("placebo_"):]: r for k, r in het.items() if k.startswith("placebo_")}
            if pl:
                entry["heterogeneity"]["placebo"] = {
                    mode: {"blp_slope": round(float(r["delta"]), 4),
                           "blp_p": round(float(r["p_wilcoxon"]), 4),
                           "gates_gap": round(float(r["ll_lo"]), 4),
                           "gates_p": round(float(r["ll_hi"]), 4)}
                    for mode, r in pl.items()}
                entry["heterogeneity"]["placebo_clean"] = bool(all(
                    float(r["p_wilcoxon"]) >= 0.025 and float(r["ll_hi"]) >= 0.025
                    for r in pl.values()))
        cl = sub[sub["block"] == "cells"]
        if len(cl):
            entry["cells"] = {
                r["key"]: {"tau": round(float(r["delta"]), 4),
                           "i2": round(float(r["ll_lo"]), 3), "q": round(float(r["ll_hi"]), 2),
                           "n_cells": int(r["n_test"]), "min_n": int(r["n_train"]),
                           "q_p": round(float(r["p_wilcoxon"]), 4),
                           "differ": bool(r["p_wilcoxon"] < 0.05)}
                for _, r in cl.iterrows()}
            entry["n_cell_dimensions"] = int(len(cl))
        rf = sub[sub["block"] == "refit"]
        if len(rf):
            r = rf.iloc[0]
            entry["refit"] = {
                "step": round(float(r["delta"]), 5),
                "normal_ci": _normal(r["delta"], r["delta_lo"], r["delta_hi"]),
                "adjusted_ci": [round(float(r["delta_lo"]), 5), round(float(r["delta_hi"]), 5)],
                "refit_ci": [round(float(r["ll_lo"]), 5), round(float(r["ll_hi"]), 5)],
                "width_ratio": round(float(r["brier"]), 2), "n_boot": int(r["n_test"])}
        pm = sub[sub["block"] == "permuted"]
        if len(pm):
            entry["permuted"] = {
                r["key"]: {"delta": round(float(r["delta"]), 4),
                           "ci": [round(float(r["delta_lo"]), 4), round(float(r["delta_hi"]), 4)],
                           "covers_zero": bool(r["delta_lo"] <= 0 <= r["delta_hi"])}
                for _, r in pm.iterrows()}
        lo = sub[sub["block"] == "league_out"]
        if len(lo):
            folds = lo[lo["key"] != "__summary__"]
            entry["league_out"] = {
                r["key"]: {"log_loss": round(float(r["log_loss"]), 4),
                           "n_test": int(r["n_test"]), "n_train": int(r["n_train"]),
                           "n_train_matches": int(r["n_train_matches"]),
                           "brier": round(float(r["brier"]), 3),
                           "accuracy": round(float(r["accuracy"]), 3),
                           "ece": round(float(r["ece"]), 3)}
                for _, r in folds.iterrows()}
            s = lo[lo["key"] == "__summary__"]
            if len(s):
                r = s.iloc[0]
                entry["league_out_summary"] = {
                    "pooled_oof": round(float(r["log_loss"]), 4),
                    "fold_mean": round(float(r["ll_lo"]), 4),
                    "fold_std": round(float(r["ll_hi"]), 4),
                    "grouped_by_match": round(float(r["real_log_loss"]), 4),
                    "cost": round(float(r["delta"]), 4),
                    "any_class_missing": bool(r["classes_missing"])}
        pl = sub[sub["block"] == "pooling"]
        if len(pl):
            r = pl.iloc[0]
            entry["pooling"] = {
                "own_log_loss": round(float(r["log_loss"]), 4),
                "delta_pooled_minus_own": round(float(r["delta"]), 4),
                "ci": [round(float(r["delta_lo"]), 4), round(float(r["delta_hi"]), 4)],
                "verdict": str(r["verdict"])}
        if len(ctl):
            # a shuffled model that BEATS its own marginal entropy means the harness leaks,
            # so `clean` is the assertion this block exists to make, not the log-losses
            entry["controls"] = {
                r["key"]: {"shuffled": round(float(r["log_loss"]), 4),
                           "entropy": round(float(r["entropy"]), 4),
                           "real": round(float(r["real_log_loss"]), 4),
                           "clean": bool(r["log_loss"] >= r["entropy"] - 5e-3)}
                for _, r in ctl.iterrows()}
            entry["controls_all_clean"] = all(c["clean"] for c in entry["controls"].values())
        out[pop] = entry
    return out


def calibration(tag: str = "") -> dict:
    """What temperature scaling does to the head, from run_value's record.

    Reported because it is a NULL that is worth publishing: once class_weight is None and the head
    is tuned, it is already calibrated against the observed mix, so a single scalar temperature
    has almost nothing left to move.

    Both calibration measures are carried, each with its direction as a flag, so prose never has
    to assert which way either moved. smoothECE is the consistent measure and the headline; the
    binned ECE moves with its bin edges (methods_second_ball.md §4) and stays for comparison.

    `by_class` is one-vs-rest over the full probability range, which is the check a top-label
    number cannot make: a class the model never ranks first contributes nothing to it.
    `max_base_rate_gap` travels beside it because evidence.md §5 reads the block as a bound over
    every class, and a bound has to be computed rather than eyeballed off a list."""
    c = _load(f"calibration{tag}.parquet")
    if c is None or not len(c):
        return {}
    r = c.iloc[0]
    out = {
        "n": int(r["n"]),
        "log_loss_before": round(float(r["log_loss_before"]), 4),
        "log_loss_after": round(float(r["log_loss_after"]), 4),
        "ece_before": round(float(r["ece_before"]), 4),
        "ece_after": round(float(r["ece_after"]), 4),
        "smooth_ece_before": round(float(r["smooth_ece_before"]), 4),
        "smooth_ece_after": round(float(r["smooth_ece_after"]), 4),
        "brier_before": round(float(r["brier_before"]), 4),
        "brier_after": round(float(r["brier_after"]), 4),
        "ece_got_worse": bool(r["ece_after"] > r["ece_before"]),
        "smooth_ece_got_worse": bool(r["smooth_ece_after"] > r["smooth_ece_before"]),
    }
    cw = _load(f"calibration_classwise{tag}.parquet")
    if cw is None or not len(cw):
        return out
    out["by_class"] = {str(x["class"]): {"n": int(x["n"]),
                                         "smooth_ece": round(float(x["smooth_ece"]), 4),
                                         "mean_proba": round(float(x["mean_proba"]), 4),
                                         "base_rate": round(float(x["base_rate"]), 4)}
                       for _, x in cw.iterrows()}
    out["max_base_rate_gap"] = round(float((cw["mean_proba"] - cw["base_rate"]).abs().max()), 4)
    return out


def causal(tag: str = "") -> dict:
    """S05's effect of a protector in the frontal, from run_causal's record.

    Every figure here is a specification's own, never a pooled one.

    ONE PRIMARY, REGISTERED, AND A CURVE AROUND IT. `primary_spec` is run_causal's, read off
    the driver, and a record that does not carry it is refused rather than given a borrowed
    name. Treatment and confounders
    are read in the same freeze-frame, so no test on this data separates a confounder from a
    mediator; `spec_curve` leaves one feature group at a time out of the full design, and
    `sign_agrees_across_specs` reads the primary and every rung together, the sign condition
    of the protector's decision.

    Every specification carries team and league fixed effects, as form_answers.md Q4
    commits to; `without_fixed_effects` is the primary one without them, so what the controls
    move is published beside the headline rather than asserted."""
    c = _load(f"causal{tag}.parquet")
    if c is None or not len(c):
        return {}

    def _target_block(sub: pd.DataFrame, full: bool = True) -> dict:
        out = {}
        for _, r in sub.iterrows():
            entry = {"effect": round(float(r["effect"]), 5),
                     "ci": [round(float(r["lo"]), 5), round(float(r["hi"]), 5)],
                     "covers_zero": bool(r["covers_zero"])}
            if full:
                entry |= {"se": round(float(r["se"]), 5), "ess": round(float(r["ess"]), 1),
                          "n_clipped": int(r["n_clipped"]),
                          "overlap_adequate": bool(r["overlap_adequate"])}
            out[str(r["target"])] = entry
        return out

    head, contrast = c[c["block"] == "headline"], c[c["block"] == "contrast"]
    deg, curve = c[c["block"] == "degenerate"], c[c["block"] == "spec_curve"]
    if _CAUSAL_PRIMARY not in set(head["spec"]):
        raise ValueError(f"causal{tag}.parquet carries no {_CAUSAL_PRIMARY!r} specification")
    ref = head.iloc[0]
    out: dict = {
        "population": {"corners": int(ref["n"]), "matches": int(ref["n_clusters"]),
                       "treated": int(ref["n_treated"]), "control": int(ref["n_control"]),
                       "treated_pct": _pct(int(ref["n_treated"]), int(ref["n"]))},
        "primary_spec": _CAUSAL_PRIMARY,
        # A refusal, written down. conceded_xg is non-zero on almost no delivery of the grant
        # (evidence.md §8), so the transition arm is not estimable; an absent block and a
        # measured null read the same to whoever opens this file next.
        "conceded_xg_degenerate": bool(deg["degenerate"].all()) if len(deg) else None,
        "by_spec": {}, "sign_agrees_across_specs": {},
    }
    for spec, sub in head.groupby("spec"):
        r = sub.iloc[0]
        lv2 = contrast[contrast["spec"] == spec]
        entry = {"c_prop": float(r["c_prop"]), "c_out_treated": float(r["c_out1"]),
                 "c_out_control": float(r["c_out0"]),
                 "frac_extreme_pct": round(100 * float(r["frac_extreme"]), 2),
                 "rho_range": [round(float(r["rho_min"]), 4), round(float(r["rho_max"]), 4)],
                 "by_target": _target_block(sub),
                 "every_interval_covers_zero": bool(sub["covers_zero"].all())}
        if len(lv2):
            entry["level2"] = _target_block(lv2, full=False)
        out["by_spec"][str(spec)] = entry
    for tg, sub in pd.concat([head, curve]).groupby("target"):
        out["sign_agrees_across_specs"][str(tg)] = bool(np.sign(sub["effect"]).nunique() == 1)
    sens = c[c["block"] == "sensitivity"]
    if len(sens):
        out["without_fixed_effects"] = _target_block(sens)

    if len(curve):
        out["spec_curve"] = _curve_summary(curve)
    return out


_AERIAL = ("clear_is_header", "clear_aerial_won", "first_contact_is_header",
           "first_contact_hops_rating")


def aerial(tag: str = "") -> dict:
    """How much of the aerial context the cleared deliveries carry, from the published table.

    The aerial covariates are read at or after the clearance, so they never enter the design
    (evidence.md §9), and a statement about a column that is mostly missing is a statement about
    coverage. The denominator is the deliveries a defender cleared, the only ones whose clearing
    contact can carry the flags."""
    t = _load(f"corners{tag}.parquet")
    if t is None or not len(t):
        return {}
    cleared = t[t["cleared"].astype(bool)]
    return {"cleared": int(len(cleared)),
            "coverage_pct": {c: round(100 * float(cleared[c].notna().mean()), 1) for c in _AERIAL}}


def q2(tag: str = "") -> dict:
    """The plan and the race joined, from run_q2's record.

    `independent_agreement_pct` is the only comparison in the repo that is not
    self-referential: the tracking race scored against `won_second_ball`, which s08
    derives from the event chain. `self_agreement_pct` is carried beside it, and the
    pair must never be read as one measurement: the second scores the race against a
    label taken from the same tracking a second later.

    The overlap is the smallest population the repo publishes, so `below_floor` travels
    with every per-state cell rather than being left for a reader to work out."""
    q = _load(f"q2_overlap{tag}.parquet")
    if q is None or not len(q):
        return {}
    idx = q.set_index(["block", "key"])
    agree, delta = idx.loc[("agreement", "independent")], idx.loc[("race_prob", "won-lost")]
    states = q[q["block"] == "plan_x_race"]
    return {
        "overlap_corners": int(idx.loc[("overlap", "corners"), "n"]),
        "overlap_matches": int(idx.loc[("overlap", "matches"), "n"]),
        "four_state_known": int(idx.loc[("overlap", "four_state"), "n"]),
        "n_labelled": int(agree["n"]),
        "independent_agreement_pct": round(100 * float(agree["point"]), 2),
        "independent_agreement_ci": [round(100 * float(agree["lo"]), 2),
                                     round(100 * float(agree["hi"]), 2)],
        # Against the MAJORITY RULE, not a coin: attackers win most of these second balls, so always
        # naming them is already a strong classifier and 50% is not a bar anything has to clear.
        "majority_rule_pct": round(100 * float(idx.loc[("agreement", "majority_rule"), "point"]), 2),
        "independent_agreement_beats_the_majority_rule": bool(
            agree["lo"] > float(idx.loc[("agreement", "majority_rule"), "point"])),
        "self_agreement_pct": round(100 * float(idx.loc[("agreement", "self"), "point"]), 2),
        "race_p_att_won": round(float(idx.loc[("race_prob", "won"), "point"]), 3),
        "race_p_att_lost": round(float(idx.loc[("race_prob", "lost"), "point"]), 3),
        "race_p_att_delta": round(float(delta["point"]), 3),
        "race_p_att_delta_ci": [round(float(delta["lo"]), 3), round(float(delta["hi"]), 3)],
        "race_delta_covers_zero": bool(delta["lo"] <= 0 <= delta["hi"]),
        # .eq(True) rather than a cast: the pooled rows carry None here, and None is not a cell.
        "cells_below_floor": int(states["below_floor"].eq(True).sum()),
        "by_state": {str(r["key"]): {"n": int(r["n"]), "p_att": round(float(r["point"]), 3),
                                     "xg": round(float(r["value_xg_after"]), 4),
                                     "below_floor": bool(r["below_floor"])}
                     for _, r in states[states["below_floor"].notna()].iterrows()},
    }


_PLAN_RACE_CASES = {"plan": "at_kick", "flight": "by_clearance", "race": "race"}


def plan_race() -> dict:
    """How the second balls of the tracked corners were won, from run_plan_race's record (Section 5.1).

    The three cases are counted after the fact, with the drop known: `at_kick` (the side that won already had the player
    nearest the drop when the corner was taken), `by_clearance` and `race`, as shares with match-clustered intervals,
    over every second ball and by the side that won it. `no_relation` sets the first against the share it would take if
    being nearest at the kick had nothing to do with winning. `onset.diff` is the defenders' median onset towards the
    drop minus the attackers', on the players detected throughout (the registered deviation), with the sheet's reading on
    every player in `onset.diff_all`. `clock` carries the offset the analysis corrected, and `largest` applies the
    sheet's rule: a case is the largest only if its interval clears the other two. Shares keep six decimals, since
    four and then the percent's one round 31 of 192 to 16.2% instead of 16.1%."""
    r = _load("plan_race.parquet")
    if r is None or not len(r):
        return {}
    idx = r.set_index(["block", "key"])

    def ci(block: str, key: str) -> dict:
        row = idx.loc[(block, key)]
        return {"point": round(float(row["point"]), 6), "ci": [round(float(row["lo"]), 6), round(float(row["hi"]), 6)],
                "n": int(row["n"])}

    def shares(block: str) -> dict:
        return {name: ci(block, k) for k, name in _PLAN_RACE_CASES.items()}

    def largest(s: dict) -> str:
        return next((k for k, v in s.items() if all(v["ci"][0] > o["ci"][1] for o2, o in s.items() if o2 != k)), "none")

    def point(block: str, key: str, digits: int = 6) -> float:
        return round(float(idx.loc[(block, key), "point"]), digits)

    every = shares("share")
    by_side = {side: shares(f"share_{side}") for side in ("attack", "defence")}
    pas, rec = shares("clock_pass"), shares("clock_receipt")
    players = r[r["block"] == "players"]
    onset, onset_all = ci("onset_diff", "detected"), ci("onset_diff", "all")
    no_rel = {k: ci("no_relation", k) for k in ("observed", "expected", "diff")}
    return {
        "clock": {"n_matches": int(idx.loc[("clock", "matches_behind"), "n"]),
                  "halves": int(idx.loc[("clock", "halves"), "n"]),
                  "all_halves_behind": point("clock", "halves") == 1.0,
                  "median_lag_s": point("clock", "median_lag_s", 2)},
        "diagnostics": {"ball_to_flag_m": point("diagnostic", "ball_to_flag_m", 2),
                        "winner_to_drop_m": point("diagnostic", "winner_to_drop_m", 2)},
        "corner": {
            "n": int(idx.loc[("population", "analysed"), "n"]),
            "second_balls": int(idx.loc[("population", "second_balls"), "n"]),
            "matches": int(idx.loc[("population", "matches"), "n"]),
            "shares": every, "largest": largest(every),
            "by_side": {side: {name: v["point"] for name, v in s.items()} for side, s in by_side.items()},
            "by_side_ci": by_side,
            "largest_by_side": {side: largest(s) for side, s in by_side.items()},
            "no_relation": {**no_rel, "diff_excludes_zero": not no_rel["diff"]["ci"][0] <= 0 <= no_rel["diff"]["ci"][1]},
            "onset": {"diff": onset, "diff_all": onset_all,
                      "covers_zero": onset["ci"][0] <= 0 <= onset["ci"][1],
                      "fully_detected_pct": round(float((players["n"] * players["point"]).sum() / players["n"].sum()), 6),
                      "already_moving": {s: ci("already_moving", s) for s in ("att", "def")},
                      "censored": {s: ci("censored", s) for s in ("att", "def")}},
            "sensitivity": {"clock_pass": pas, "clock_receipt": rec,
                            "clock_move_pts": round(100 * max(abs(pas[k]["point"] - rec[k]["point"]) for k in pas), 1),
                            "four_detected": shares("four_detected"), "keepers_out": shares("keepers_out")}}}


_STABLE_RUNGS = ("delivery", "full", "uniform")


def sufficiency(tag: str = "") -> dict:
    """Is the freeze-frame, read at the delivery, a sufficient statistic for who recovers, from
    run_sufficiency's record. The measurement form_answers.md Q3 promises.

    TWO BLOCKS, TWO POWERS, and reading one for the other is the mistake this splits to
    prevent. `grant` is the answer: every corner carrying a delivery freeze-frame and a
    second ball the event chain resolved, five leagues, no tracking in it. `overlap` is what
    tracking adds on the tracked sample, and it travels with `mde_80` because a null whose design
    could not resolve the effect either way is a statement about the sample.

    The claim-bearing rung is `delivery`, never the uniform floor and never the class prior
    alone (architecture.md §Stages). `headline_delta` is therefore the static race against
    the delivery baseline, and `geometry_over_delivery` the same thing for the handcrafted
    block.

    `label_agreement_pct` bounds the overlap half on its own: the tracking and event-chain
    answers to "who won it" are two different measurements of the same corner, and the arms
    can only rank one of them at a time.

    `freeze_frame_beats_the_delivery_baseline` is decided on the REFIT interval in its normal
    form when the run measured it for all three decisive steps, and `verdict_interval` names the
    interval that decided it. The fixed-fit bounds hold the out-of-fold probabilities and are too
    narrow, and on the grant the geometry step's upper bound sits within a sliver of zero, so a
    verdict read there moves with the form of the interval: what the pivotal and percentile
    forms would have decided travels beside it."""
    s = _load(f"sufficiency{tag}.parquet")
    if s is None or not len(s):
        return {}

    def _pop(frame: pd.DataFrame, block: str, key: str, col: str = "n") -> Any:
        r = frame[(frame["block"] == block) & (frame["key"] == key)]
        return None if not len(r) else r.iloc[0][col]

    def _ci(r: pd.Series) -> list:
        return [round(float(r["lo"]), 4), round(float(r["hi"]), 4)]

    out: dict = {}
    g = s[s["population"] == "grant"]
    if len(g):
        arms, ladder = g[g["block"] == "arm"], g[g["block"] == "ladder"]
        delta, abl = g[g["block"] == "delta"], g[g["block"] == "ablation"]
        lo = g[(g["block"] == "league_out") & (g["key"] == "__summary__")]
        ctl = g[g["block"] == "control"]

        def _step(key: str) -> dict | None:
            r = delta[delta["key"] == key]
            if not len(r):
                return None
            r = r.iloc[0]
            return {"delta": round(float(r["point"]), 5), "ci": _ci(r),
                    "covers_zero": bool(r["covers_zero"]),
                    "p_wilcoxon": round(float(r["p_wilcoxon"]), 4)}

        out["grant"] = {
            "corners": int(_pop(g, "population", "corners")),
            "matches": int(_pop(g, "population", "matches")),
            "leagues": int(_pop(g, "population", "leagues")),
            "positive": int(_pop(g, "population", "positive")),
            "negative": int(_pop(g, "population", "negative")),
            "base_rate": round(float(_pop(g, "population", "base_rate", "point")), 4),
            "tau_snapshot": round(float(_pop(g, "tuning", "tau_snapshot", "point")), 3),
            "by_arm": {str(r["key"]): {"log_loss": round(float(r["point"]), 4),
                                       "auc": round(float(r["auc"]), 3),
                                       "accuracy": round(float(r["accuracy"]), 3),
                                       "mean_p": round(float(r["mean_p"]), 3)}
                       for _, r in arms.iterrows()},
            # Only the rungs shown to reproduce across full runs, and the intercept-only one is
            # not among them; the deterministic floor is the `prior` arm in `by_arm`.
            "ladder": {str(r["key"]): round(float(r["point"]), 4)
                       for _, r in ladder.iterrows() if str(r["key"]) in _STABLE_RUNGS},
            "headline_delta": _step("static_race-delivery"),
            "geometry_over_delivery": _step("geometry360-delivery"),
            "strong_learner_over_delivery": _step("gbt360-delivery"),
            "static_race_over_geometry": _step("static_race-geometry360"),
            "delivery_over_prior": _step("delivery-prior"),
            "ablation": {str(r["key"]): {"delta": round(float(r["point"]), 5), "ci": _ci(r),
                                         "covers_zero": bool(r["covers_zero"])}
                         for _, r in abl.iterrows()},
            # These hold the out-of-fold probabilities fixed: §validation refits its ablations
            # inside every draw; these do not. How much too narrow that makes them is
            # `validation.<pop>.refit`. The field says which kind they are rather than leaving
            # a reader to assume they match the ladder's.
            "ablation_interval": "fixed_oof",
            "n_ablation_groups": int(len(abl)),
            "n_groups_covering_zero": int(abl["covers_zero"].sum()) if len(abl) else 0,
            "controls_all_clean": bool(ctl["clean"].all()) if len(ctl) else None,
        }
        if len(lo):
            out["grant"]["league_out"] = {"pooled": round(float(lo.iloc[0]["point"]), 4),
                                          "fold_mean": round(float(lo.iloc[0]["lo"]), 4),
                                          "fold_std": round(float(lo.iloc[0]["hi"]), 4)}
        decisive = {"headline_delta": "static_race-delivery",
                    "geometry_over_delivery": "geometry360-delivery",
                    "strong_learner_over_delivery": "gbt360-delivery"}
        rft = g[g["block"] == "refit"]
        refit = {}
        for name, key in decisive.items():
            r = rft[rft["key"] == key]
            if len(r):
                r = r.iloc[0]
                normal = _normal(r["point"], r["step_lo"], r["step_hi"])
                refit[name] = {"delta": round(float(r["point"]), 5), "normal_ci": normal,
                               "adjusted_ci": [round(float(r["lo"]), 5), round(float(r["hi"]), 5)],
                               "refit_ci": [round(float(r["step_lo"]), 5),
                                            round(float(r["step_hi"]), 5)],
                               "width_ratio": round(float(r["width_ratio"]), 2),
                               "covers_zero": _covers(normal), "n_boot": int(r["n"])}
        if refit:
            out["grant"]["refit"] = refit
        # The freeze-frame adds information only if a delta against the DELIVERY rung clears zero
        # on the losing side for the delivery, in any of the three representations. A partial
        # refit block would decide on a mix of two interval forms, so it falls back whole.
        if len(refit) == len(decisive):
            out["grant"]["verdict_interval"] = "refit_normal"
            beats = any(v["normal_ci"][1] < 0 for v in refit.values())
            out["grant"]["beats_under_pivotal"] = any(v["adjusted_ci"][1] < 0 for v in refit.values())
            out["grant"]["beats_under_percentile"] = any(v["refit_ci"][1] < 0 for v in refit.values())
        else:
            out["grant"]["verdict_interval"] = "fixed_oof"
            beats = any(st is not None and not st["covers_zero"] and st["delta"] < 0
                        for st in (out["grant"][k] for k in decisive))
        out["grant"]["freeze_frame_beats_the_delivery_baseline"] = bool(beats)

    o = s[s["population"] == "overlap"]
    if len(o):
        arms, delta = o[o["block"] == "arm"], o[o["block"] == "delta"]
        folds = o[o["block"] == "fold"]

        def _ostep(key: str) -> dict | None:
            r = delta[delta["key"] == key]
            if not len(r):
                return None
            r = r.iloc[0]
            return {"delta": round(float(r["point"]), 4), "ci": _ci(r),
                    "covers_zero": bool(r["covers_zero"]),
                    "p_wilcoxon": round(float(r["p_wilcoxon"]), 4)}

        mde = _pop(o, "power", "mde_80", "point")
        agree = _pop(o, "population", "label_agreement", "point")
        out["overlap"] = {
            "corners": int(_pop(o, "population", "corners")),
            "matches": int(_pop(o, "population", "matches")),
            "positive": int(_pop(o, "population", "positive")),
            "negative": int(_pop(o, "population", "negative")),
            "label_agreement_pct": None if agree is None else round(100 * float(agree), 2),
            "by_arm": {str(r["key"]): {"log_loss_recal": round(float(r["point"]), 4),
                                       "log_loss_raw": round(float(r["log_loss_raw"]), 4),
                                       "calibration_cost": round(float(r["calibration_cost"]), 4),
                                       "auc": round(float(r["auc"]), 3)}
                       for _, r in arms.iterrows()},
            "headline_delta": _ostep("snapshot-race"),
            # The whole gap, decomposed. What the 360 does not SEE, then what velocity, the
            # measured lag and the seconds between the kick and the first contact are worth,
            # then what knowing the drop is worth.
            "step_coverage": _ostep("snapshot-kick"),
            "step_kinematics": _ostep("kick-anchored"),
            "step_known_drop": _ostep("anchored-race"),
            # None when the curve never reaches 80% power on the grid: s04b.mde returns NaN there,
            # and a NaN never equals itself, so --check would call the record stale on every run.
            "mde_80": None if mde is None or pd.isna(mde) else round(float(mde), 4),
            "folds": int(len(folds)),
            "folds_without_a_negative_in_test": int((folds["test_neg"] == 0).sum()) if len(folds) else 0,
        }
        probe = o[o["block"] == "prior_probe"]
        if len(probe):
            # Published as a standing regression: recalibrating a fold-constant arm reads the
            # held-out match's base rate, so the prior is scored raw.
            out["overlap"]["prior_recalibration_leaks"] = bool(not probe.iloc[0]["clean"])
        hd = out["overlap"]["headline_delta"]
        if hd is not None and out["overlap"]["mde_80"] is not None:
            out["overlap"]["arms_separate"] = bool(not hd["covers_zero"])
            out["overlap"]["delta_is_inside_the_detectable_range"] = bool(
                abs(hd["delta"]) >= out["overlap"]["mde_80"])

    sens = s[s["population"] == "sensitivity"]
    if len(sens):
        a = sens[(sens["block"] == "arm") & (sens["key"] == "race")]
        out["sensitivity"] = {
            "corners": int(_pop(sens, "population", "corners")),
            # The race arm against a label read from the same tracking a second later. Carried
            # so the gap to its score on the event outcome is visible, never as the result.
            "race_auc_against_its_own_label": round(float(a.iloc[0]["auc"]), 3) if len(a) else None,
        }
    return out


def refit_readings() -> dict:
    """Whether the refit draws of the decisive contrasts license their normal reading
    (run_refit_draws).

    The normal form is the point plus or minus half the refit width, which is the normal-theory
    interval only when the draws are symmetric. The contract was fixed before the draws were
    kept: symmetry is good enough when the percentiles re-centred on the point (`centred_ci`) give
    the same verdict as the normal form on every contrast. The normal interval on the draws' own
    SD and their skew travel beside that verdict, so the size of any asymmetry is visible. The
    head-to-head's own three readings live under `validation.<population>.delivery_vs_360.refit`;
    the step and the geometry step already publish theirs under `validation` and `sufficiency`."""
    draws = _draws_by_contrast()
    if not draws:
        return {}
    out: dict = {}
    for (pop, contrast), (point, d) in sorted(draws.items()):
        ref = _refit_reading(point, d)
        out.setdefault(pop, {})[contrast] = {
            "sd": ref["sd"], "skew": ref["skew"], "normal_sd_ci": ref["normal_sd_ci"],
            "centred_ci": ref["centred_ci"], "n_boot": ref["n_boot"],
            "verdict_agrees": _covers(ref["normal_ci"]) == _covers(ref["centred_ci"])}
    agrees = all(c["verdict_agrees"] for cs in out.values() for c in cs.values())
    out["every_verdict_agrees"] = agrees
    return out


def scope() -> dict:
    """Volume per population, from the delivery grains alone: the counts the scope decision reads.

    Every figure here is a count, read before any model, so the scope argument does not rest on a
    result of the pipeline it scopes. `teams_below_floor` counts the teams with at least one delivery
    whose season total sits under MIN_CELL; a team with none of a kind is absent from its grain, and
    a population with no delivery in the grains is left out. `tracked_deliveries` counts the ones
    inside the tracked sample, and `with_360` the ones whose delivery carries a 360 anywhere in the
    grant, which for a modelled population is its modelling table. The grain also holds the short
    corners no cross followed, which have no delivery: they are the population `short_no_cross`, counted
    in the same fields and out of `corner`, whose `deliveries` are the corners that have one. Beside the
    populations, `matches` is the grant's and `tracked_matches` the tracked sample's."""
    eda, matches, throws = (_load("eda_deliveries.parquet"), _load("eda_matches.parquet"),
                            _load("eda_throw_ins.parquet"))
    if any(t is None for t in (eda, matches, throws)):
        return {}
    tracked = set(_loader.tracked_matches())
    tables = {"corner": _load("corners.parquet"),
              "free_kick_flank": _load("corners_free_kick_flank.parquet")}
    out = {"matches": int(len(matches)), "tracked_matches": len(tracked)}
    corners = eda[eda["set_piece_type"] == "corner"]
    none = corners["anchor"] == "none" if "anchor" in corners.columns else pd.Series(False, index=corners.index)
    for name, d in (("corner", corners[~none]),
                    ("free_kick_flank", eda[eda["set_piece_type"] == "free_kick_flank"]),
                    ("long_throw", throws),
                    ("short_no_cross", corners[none])):
        if d.empty:
            continue
        per_team = d.groupby(["league", "attacking_team"]).size()
        with_360 = (int(d["delivery_360"].sum()) if name == "long_throw"
                    else None if tables.get(name) is None else len(tables[name]))
        out[name] = {"deliveries": int(len(d)), "per_match": round(len(d) / len(matches), 2),
                     "per_team_median": float(per_team.median()), "per_team_min": int(per_team.min()),
                     "teams": int(len(per_team)),
                     "teams_below_floor": int((per_team < MIN_CELL).sum()),
                     "tracked_deliveries": int(d["match_id"].isin(tracked).sum()),
                     "with_360": with_360,
                     "with_360_pct": None if with_360 is None else _pct(with_360, len(d))}
    return out


def _flank_section() -> Callable[[], dict]:
    """The flank free kicks' blocks, the twin of every corner block, each read from its
    `_free_kick_flank` artefact.

    The frontier and validation figures live under their own sections, which the corner and flank
    records publish together, and a second copy would be a second live value for one quantity."""
    kind = "free_kick_flank"
    tag = f"_{kind}"
    blocks = {"corpus": lambda: corpus(tag), "situation": lambda: situation(tag),
              "race": lambda: race(tag), "q2": lambda: q2(tag),
              "sufficiency": lambda: sufficiency(tag), "repeatability": lambda: repeatability(kind),
              "calibration": lambda: calibration(tag), "causal": lambda: causal(tag),
              "aerial": lambda: aerial(tag), "drop_position": lambda: drop_position(tag),
              "per_league": lambda: per_league(tag),
              "falsification": lambda: falsification(tag),
              "coverage_360": lambda: coverage(tag),
              "coverage_by_league": lambda: coverage_by_league(tag),
              "tracking_window": lambda: tracking(tag), "counter": lambda: counter(tag),
              "first_contact": lambda: first_contact(kind),
              "first_contact_frame": lambda: first_contact_frame(tag),
              "positive_control": lambda: positive_control(tag),
              "aim_zone": lambda: aim_zone(tag),
              "team_shrinkage": lambda: team_shrinkage(tag),
              "duel_prices": lambda: duel_prices(tag),
              "club_power": lambda: club_power(tag), "top_teams": lambda: top_teams(tag),
              "tracking_curve": lambda: tracking_curve(tag)}
    return lambda: {name: fn() for name, fn in blocks.items()}


_CLUB_POWER = ("decision", "measures", "power", "half_season", "c12a", "reliability")
_TOP_TEAMS = ("table", "contrasts", "influence", "by_league", "decisions", "reliability")


def _twin_sources(t: str) -> dict:
    """artefact -> driver for the records every population of the paper carries under its tag."""
    return {f"falsification{t}_stats.parquet": "falsify_frontier.py",
            f"audit360{t}.parquet": "audit_360.py", f"audit_tracking{t}.parquet": "audit_tracking.py",
            f"counter_sweep{t}.parquet": "sweep_counter.py",
            f"first_contact{t}.parquet": "run_first_contact.py",
            f"positive_control{t}.parquet": "run_positive_control.py",
            f"aim_zone{t}.parquet": "run_aim_zone.py",
            f"team_shrinkage{t}.parquet": "run_team_shrinkage.py",
            f"tracking_curve_fit{t}.parquet": "run_tracking_curve.py",
            f"tracking_curve_draws{t}.parquet": "run_tracking_curve.py",
            **{f"duel_prices_{ZONE_PARTITION}{t}{x}.parquet": "run_duel_prices.py"
               for x in ("", "_decision", "_c1a")},
            **{f"club_power_{n}{t}.parquet": "eda_club_power.py" for n in _CLUB_POWER},
            **{f"top_teams_{n}{t}.parquet": "eda_top_teams.py" for n in _TOP_TEAMS}}


def _flank_sources() -> dict:
    """artefact -> driver for the flank free kicks, mirroring `_flank_section`."""
    t = "_free_kick_flank"
    return {f"corners{t}.parquet": "build_table.py",
            f"shots{t}.parquet": "build_table.py",
            f"calibration{t}.parquet": "run_value.py",
            f"calibration_classwise{t}.parquet": "run_value.py",
            f"causal{t}.parquet": "run_causal.py",
            f"situation{t}.parquet": "run_situation.py",
            f"situation{t}_states.parquet": "run_situation.py",
            f"situation{t}_radii.parquet": "run_situation.py",
            f"race_corners{t}.parquet": "run_race.py",
            f"q2_overlap{t}.parquet": "run_q2.py",
            f"sufficiency{t}.parquet": "run_sufficiency.py",
            f"drop_position{t}.parquet": "run_drop_position.py",
            f"first_contact{t}_gbt.parquet": "run_first_contact.py"}


SECTIONS: dict[str, Callable[[], dict]] = {
    "corpus": corpus,
    "per_league": per_league,
    "frontier": frontier,
    "penalties": penalties,
    "falsification": falsification,
    "throw_ins": throw_ins,
    "flank_cut": flank_cut,
    "frame_gpu": frame_gpu,
    "zone_comparison": zone_comparison,
    "duel_prices": duel_prices,
    "club_power": club_power,
    "top_teams": top_teams,
    "tracking_curve": tracking_curve,
    "first_contact": first_contact,
    "first_contact_frame": first_contact_frame,
    "positive_control": positive_control,
    "aim_zone": aim_zone,
    "drop_position": drop_position,
    "short_corner_sensitivity": short_corner_sensitivity,
    "noise_floor": noise_floor,
    "team_shrinkage": team_shrinkage,
    "teams": teams,
    "counter": counter,
    "coverage_360": coverage,
    "coverage_by_league": coverage_by_league,
    "tracking_window": tracking,
    "race": race,
    "situation": situation,
    "q2": q2,
    "plan_race": plan_race,
    "sufficiency": sufficiency,
    "repeatability": repeatability,
    "validation": validation,
    "refit_readings": refit_readings,
    "calibration": calibration,
    "causal": causal,
    "aerial": aerial,
    "scope": scope,
    # The flank free kicks reuse the frontier and validation blocks the corner+flank records
    # publish above.
    "free_kick_flank": _flank_section(),
}

# artefact -> the driver that writes it, so a missing section names its own fix. The driver name
# alone: the invocation that publishes each artefact lives in run_e2e, the one recipe.
SOURCES = {
    "corners.parquet": "build_table.py", "shots.parquet": "build_table.py",
    "frontier.parquet": "describe_frontier.py",
    "falsification_stats.parquet": "falsify_frontier.py",
    "situation_states.parquet": "run_situation.py",
    "situation_radii.parquet": "run_situation.py",
    "throw_in_probe.parquet": "probe_throw_ins.py",
    "flank_cut_bands.parquet": "run_flank_cut.py",
    "flank_cut_contrasts.parquet": "run_flank_cut.py",
    "eda_deliveries.parquet": "eda_build.py",
    "eda_teams_corner.parquet": "eda_second_ball.py",
    "counter_sweep.parquet": "sweep_counter.py",
    "first_contact.parquet": "run_first_contact.py",
    "positive_control.parquet": "run_positive_control.py",
    "aim_zone.parquet": "run_aim_zone.py",
    "team_shrinkage.parquet": "run_team_shrinkage.py",
    "drop_position.parquet": "run_drop_position.py",
    "short_corner_sensitivity.parquet": "run_short_corner_sensitivity.py",
    "audit360.parquet": "audit_360.py", "audit_tracking.parquet": "audit_tracking.py",
    "race_corners.parquet": "run_race.py", "situation.parquet": "run_situation.py",
    "q2_overlap.parquet": "run_q2.py",
    "plan_race.parquet": "run_plan_race.py", "plan_race_corners.parquet": "run_plan_race.py",
    "plan_race_onsets.parquet": "run_plan_race.py",
    "sufficiency.parquet": "run_sufficiency.py",
    "validation.parquet": "run_validation.py",
    "noise_floor.parquet": "run_noise_floor.py",
    "refit_draws.parquet": "run_refit_draws.py",
    "calibration.parquet": "run_value.py",
    "calibration_classwise.parquet": "run_value.py",
    "causal.parquet": "run_causal.py",
    "eda_reliability_corner.parquet": "eda_repeatability.py",
    "eda_dispersion_corner.parquet": "eda_repeatability.py",
    "eda_matches.parquet": "eda_build.py",
    "eda_throw_ins.parquet": "eda_throw_ins.py",
    "eda_penalties.parquet": "eda_penalties.py",
    # Teams and position feed no block and are stamped anyway, because a provenance row is what
    # makes an artefact ageing out of step with its corner twin visible.
    "eda_teams_free_kick_flank.parquet": "eda_second_ball.py",
    "eda_position_free_kick_flank.parquet": "eda_second_ball.py",
    "eda_reliability_free_kick_flank.parquet": "eda_repeatability.py",
    "eda_dispersion_free_kick_flank.parquet": "eda_repeatability.py",
    "frame_gpu.parquet": "run_frame_gpu.py",
    "zone_comparison.parquet": "run_zone_comparison.py",
    "zone_robustness.parquet": "run_zone_comparison.py",
    **_twin_sources(""),
    **_twin_sources("_free_kick_flank"),
    **_flank_sources(),
}


def _stamps() -> dict:
    """mtime of every artefact read, so a figure can be dated to its parquet."""
    out = {}
    for name, driver in SOURCES.items():
        p = _DER / name
        out[name] = {"driver": driver,
                     "mtime": (datetime.fromtimestamp(p.stat().st_mtime, timezone.utc)
                               .strftime("%Y-%m-%d %H:%M UTC") if p.exists() else None)}
    return out


def _render_value(v: Any) -> str:
    if isinstance(v, dict):
        return "<br>".join(f"`{k}`: {_render_value(x)}" for k, x in v.items())
    return "—" if v is None else str(v)


def _json_safe(v: Any) -> Any:
    """`v` with every NaN published as None, all the way down. JSON has no NaN, and NaN never equals
    itself, so a figure the record cannot compute would make an unchanged record read as moved."""
    if isinstance(v, dict):
        return {k: _json_safe(x) for k, x in v.items()}
    if isinstance(v, list):
        return [_json_safe(x) for x in v]
    return None if isinstance(v, float) and v != v else v


def build() -> dict:
    """Every section plus the provenance stamps."""
    data = {k: _json_safe(fn()) for k, fn in SECTIONS.items()}
    return {"generated_by": "pipeline/report_figures.py", "sources": _stamps(), "figures": data}


def to_markdown(payload: dict) -> str:
    """The human-readable twin of measured.json."""
    lines = ["# Measured — every figure, regenerated from the derived artefacts", "",
             "GENERATED FILE. Do not edit by hand: `python pipeline/report_figures.py` rewrites it.",
             "Prose elsewhere should point here rather than copy a number, so a figure that moves",
             "moves in one place.", "", "## Provenance", "",
             "| artefact | written by | last written |", "|---|---|---|"]
    for name, s in payload["sources"].items():
        lines.append(f"| `{name}` | `{s['driver']}` | {s['mtime'] or '**missing**'} |")
    for section, figs in payload["figures"].items():
        lines += ["", f"## {section}", ""]
        if not figs:
            lines.append("Not available: the driver that writes its artefact has not been run.")
            continue
        lines += ["| figure | value |", "|---|---|"]
        for k, v in figs.items():
            lines.append(f"| `{k}` | {_render_value(v)} |")
    return "\n".join(lines) + "\n"


def main() -> None:
    """Regenerate measured.md/measured.json, or check they are current."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true",
                    help="exit non-zero if the committed outputs differ from a fresh build")
    args = ap.parse_args()

    payload = build()

    if args.check:
        # Only the `figures` block: `sources` carries mtimes, so a byte comparison reports
        # STALE on any rewritten artefact even when no figure moved. Same comparison as
        # tests/test_figures.py, which is the gate that has to agree with this one.
        if not _OUT_JSON.exists():
            print(f"STALE: {_OUT_JSON.name} does not exist — run `python pipeline/report_figures.py`")
            sys.exit(1)
        on_disk = json.loads(_OUT_JSON.read_text()).get("figures")
        if on_disk != payload["figures"]:
            moved = sorted(k for k in set(on_disk or {}) | set(payload["figures"])
                           if (on_disk or {}).get(k) != payload["figures"].get(k))
            print(f"STALE: {moved} moved — re-run `python pipeline/report_figures.py`")
            sys.exit(1)
        print("measured.json figures are current")
        return

    _OUT_MD.parent.mkdir(parents=True, exist_ok=True)
    _OUT_MD.write_text(to_markdown(payload))
    _OUT_JSON.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str))
    missing = [n for n, s in payload["sources"].items() if s["mtime"] is None]
    for section, figs in payload["figures"].items():
        print(f"  {section:20s} {'—' if not figs else str(len(figs)) + ' figures'}")
    if missing:
        print(f"\nmissing artefacts (their sections are empty): {missing}")
    print(f"\nwrote {_OUT_MD} and {_OUT_JSON}")


if __name__ == "__main__":
    main()
