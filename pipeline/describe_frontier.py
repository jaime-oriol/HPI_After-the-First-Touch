"""describe_frontier — the elementary analysis (Q1's headline, argument.md §3).

Q1 asks "how much of dead-ball value is decided after the first contact, and how much of that
was set before the delivery". This driver answers the first half off the shot-grain table, with
a match-clustered interval on every headline figure and the n of every cell in view. It
concludes nothing on its own: it is the descriptive layer the value model is then asked to
explain, and it is the layer that carries the argument if the model nulls.

Every rate is reported per POPULATION. A pooled corner+free-kick rate is not a rate, it is a
weighted average whose weight is the corner:free-kick ratio of whatever sample was built, and
that ratio moves with league, team and referee.

THE INTERVAL ON obs/exp RESAMPLES MATCHES. Corners inside a match share teams, coach, referee
and scoreline, so the unit of independence is the match here as everywhere else in this repo. The
exact Poisson (Garwood) interval is persisted beside it, and the two are not one bound taken with
and without an independence assumption. Garwood conditions on the expected goals as a known
constant; the cluster bootstrap resamples goals and expected goals together, and their co-movement
inside a resample narrows the ratio. The Poisson bound is the wider of the two, and the two are
persisted side by side so a reader can compare them.

`expected` is the PROVIDER's xG, not ours, so obs/exp is scored against an external model and
that has to be said wherever the ratio is published. An own goal is a goal the delivery produced
but not a shot: it counts in every goal count and goal share, apart as `own_goals`, and in no ratio
over shots or over xG, where a goal no model priced would lift the cell it lands in.

    python pipeline/describe_frontier.py
    python pipeline/describe_frontier.py --shots data/derived/shots_corner_free_kick_flank.parquet
    python pipeline/describe_frontier.py --team "Arsenal" --boot 2000
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pandas as pd

_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO / "src"))
sys.path.insert(0, str(_REPO / "pipeline"))
_DER = _REPO / "data" / "derived"

from constants import MIN_CELL                                       # noqa: E402
from eda_build import kind_inputs                                    # noqa: E402
from s01_phases import AFTER_FIRST_TOUCH, ORIGINS, SEGMENTS, is_own_goal   # noqa: E402
from s08_situation import FOUR_STATES                                # noqa: E402
from validate import cluster_bootstrap_stats, poisson_ratio_ci       # noqa: E402

_ORDER = list(SEGMENTS)
# The paper's two populations, read off the tables the descriptive layer shares.
_DELIVERIES, _SHOTS = kind_inputs("corner")

# The body-part cut crosses the frontier with how the shot was struck. `direct` is overwhelmingly
# headed and what follows the first contact is overwhelmingly not, so a test by segment and a test
# by body part are nearly the same test on this data, and the pooled obs/exp gap between the two
# segments is substantially a mix effect. The cells are exhaustive over every attacking shot, so
# they have to sum back to the segment totals; `late` is carried as its own pair rather than
# dropped, because the two pooled ratios below are scored over every shot.
_BODY_GROUPS = ("direct", "after_first_touch", "late")
BODY_CELLS = tuple(f"{g}|{b}" for g in _BODY_GROUPS for b in ("head", "non_head"))
BODY_PARTS = ("head", "non_head")
# "non_head" pools feet and the provider's `Other`, so it is NOT a foot-shot class. An own goal
# carries no body part at all and sits in no cell.
BODY_PART_CONTRASTS = (("segment_within_head", "direct|head", "after_first_touch|head"),
                       ("segment_within_non_head", "direct|non_head", "after_first_touch|non_head"),
                       ("body_part_all_shots", "head", "non_head"))


# ── Diagnostics, with an explicit stopping band ─────────────────────────
def diagnose(shots: pd.DataFrame) -> dict:
    """Check the shot table before any ratio is computed, and RAISE rather than warn.

    Every band here is a property the frontier must have for a ratio built on it to mean
    anything. A silent violation would still produce a plausible obs/exp, which is exactly
    the failure mode a descriptive layer cannot afford: `segment` and `match_id` decide
    cell membership and the cluster unit, so a null in either is a row that cannot be
    placed, and a zero xG sum is a division that would return inf rather than fail.

    The cardinality bands are conditional on sample size, and deliberately so. An unknown
    segment is always fatal, because it means the taxonomy moved. A MISSING segment is only
    fatal once the table is large enough to expect all five: below 5 x MIN_CELL shots a
    missing `flick_on` is a small sample, not a broken table, and killing the driver there
    would make it unusable for the smoke tests that run it over a handful of matches. The
    threshold is derived from the publication floor rather than picked."""
    d = {"rows": int(len(shots)), "matches": int(shots["match_id"].nunique())}
    for col in ("segment", "match_id", "attacking_team", "is_goal"):
        n_null = int(shots[col].isna().sum())
        d[f"null_{col}"] = n_null
        if col in ("segment", "match_id") and n_null:
            raise SystemExit(f"STOP: {n_null} null {col} — cell membership is undecidable")

    # s01 writes xg 0.0 on an own goal, so a null here means a broken table; the count is
    # reported before anything is summed.
    d["null_xg"] = int(shots["xg"].isna().sum())
    xg = shots["xg"].dropna()
    d["xg_min"], d["xg_max"] = float(xg.min()), float(xg.max())
    if not (0.0 <= d["xg_min"] and d["xg_max"] <= 1.0):
        raise SystemExit(f"STOP: xg outside [0, 1] — [{d['xg_min']}, {d['xg_max']}]")

    seen = set(shots["segment"].unique())
    unknown = seen - set(SEGMENTS)
    if unknown:
        raise SystemExit(f"STOP: unknown segment(s) {sorted(unknown)} — the taxonomy moved")
    d["n_segments"] = len(seen)
    full_sample = len(shots) >= len(SEGMENTS) * MIN_CELL
    d["cardinality_enforced"] = full_sample
    if full_sample and len(seen) != len(SEGMENTS):
        raise SystemExit(f"STOP: {sorted(set(SEGMENTS) - seen)} absent from {len(shots)} shots")
    if not full_sample:
        print(f"  [sample under {len(SEGMENTS) * MIN_CELL} shots: cardinality reported, "
              f"not enforced]")
    d["n_leagues"] = int(shots["league"].nunique())

    # A publishable cell with no expected goals at all would make obs/exp infinite. Below the
    # floor it is just a thin cell (an own goal carries xG 0.0), so it is reported, not fatal.
    zero = {s: len(g) for s, g in shots.groupby("segment") if float(g["xg"].sum()) <= 0.0}
    d["zero_xg_segments"] = zero
    fatal = {s: n for s, n in zero.items() if n >= MIN_CELL}
    if fatal:
        raise SystemExit(f"STOP: zero xG sum in publishable segment(s) {fatal} — obs/exp is "
                         f"undefined there")
    d["by_segment"] = shots["segment"].value_counts().to_dict()
    return d


# ── obs/exp, clustered by match ─────────────────────────────────────────
def _oe(sub: pd.DataFrame) -> float:
    """Goals over summed xG on a subset's shots, or NaN when it has none in a resample."""
    s = sub[~is_own_goal(sub)]
    e = float(s["xg"].sum())
    return float(s["is_goal"].sum()) / e if e > 0 else np.nan


def _oe_stats(mask_col: str, keys: Sequence) -> dict:
    """One obs/exp callable per key, all evaluated on the SAME cluster draws.

    Sharing the draws is what makes two cells comparable: independently drawn intervals
    cannot be differenced, and the paired reading is the one a reader will make."""
    return {str(k): (lambda s, k=k: _oe(s[s[mask_col] == k])) for k in keys}


def clustered_oe(frame: pd.DataFrame, mask_col: str, keys: Sequence, boot: int) -> pd.DataFrame:
    """obs/exp per key with a match-clustered percentile interval, plus the exact Poisson one.

    `excludes_1` reads the clustered interval, and only when it has width. A cell with no goals
    resamples to zero every time, so its interval is [0, 0]: a bootstrap that saw no variation,
    which carries no evidence against parity. `shots` and `goals` are the cell's shots; its own
    goals are counted apart and enter no ratio."""
    own = is_own_goal(frame)
    present = [k for k in keys if ((frame[mask_col] == k) & ~own).any()]
    ci = cluster_bootstrap_stats(frame, _oe_stats(mask_col, present),
                                 group_col="match_id", n_boot=boot)
    rows = []
    for k in present:
        cell = frame[mask_col] == k
        sub = frame[cell & ~own]
        goals, exp = int(sub["is_goal"].sum()), float(sub["xg"].sum())
        p_lo, p_hi = poisson_ratio_ci(goals, exp)
        rows.append({mask_col: k, "shots": len(sub), "goals": goals,
                     "own_goals": int((cell & own).sum()), "sum_xg": round(exp, 2),
                     "obs_exp": ci.loc[str(k), "point"],
                     "lo": ci.loc[str(k), "lo"], "hi": ci.loc[str(k), "hi"],
                     "poisson_lo": p_lo, "poisson_hi": p_hi,
                     "excludes_1": bool(ci.loc[str(k), "hi"] > ci.loc[str(k), "lo"]
                                        and (ci.loc[str(k), "hi"] < 1.0 or ci.loc[str(k), "lo"] > 1.0)),
                     "thin": f"<{MIN_CELL}" if len(sub) < MIN_CELL else ""})
    return pd.DataFrame(rows).set_index(mask_col)


def segment_table(shots: pd.DataFrame, boot: int) -> pd.DataFrame:
    """Volume, conversion, xG and the clustered obs/exp per segment.

    obs/exp is the diagnostic that matters: a segment whose interval excludes 1.0 is one the
    provider's xG systematically mis-prices, which is a claim about the shot type rather than
    about how many of them a team takes. Volume, conversion and xG are the segment's shots; its own
    goals sit apart and count only in `goal_share`, the share of every goal."""
    own = is_own_goal(shots)
    g = shots[~own].groupby("segment")
    out = pd.DataFrame({"shots": g.size(), "goals": g["is_goal"].sum(),
                        "own_goals": shots[own].groupby("segment").size(),
                        "mean_xg": g["xg"].mean(),
                        "median_t_s": g["t_since_delivery_s"].median(),
                        "mean_touches": g["n_touches_before"].mean()})
    counts = ["shots", "goals", "own_goals"]
    out[counts] = out[counts].fillna(0).astype(int)
    out["conv"] = out["goals"] / out["shots"]
    out["shot_share"] = out["shots"] / out["shots"].sum()
    scored = out["goals"] + out["own_goals"]
    out["goal_share"] = scored / scored.sum()
    oe = clustered_oe(shots, "segment", _ORDER, boot)
    out = out.join(oe[["sum_xg", "obs_exp", "lo", "hi", "poisson_lo", "poisson_hi",
                       "excludes_1", "thin"]])
    return out.reindex([s for s in _ORDER if s in out.index])


def headline(shots: pd.DataFrame, boot: int) -> pd.DataFrame:
    """Q1's number: the share of shots, goals and xG landing after the first contact.

    The shot share and the conversions read the shots; the goal share reads every goal, an own goal
    in the segment s01 gives it."""
    def conv(s: pd.DataFrame, after: bool) -> float:
        t = s[~is_own_goal(s)]
        return t.loc[t["after_first_touch"] if after else t["segment"] == "direct", "is_goal"].mean()

    stats = {
        "share_shots_after_first_touch": lambda s: s.loc[~is_own_goal(s), "after_first_touch"].mean(),
        "share_goals_after_first_touch":
            lambda s: s.loc[s["after_first_touch"], "is_goal"].sum() / max(s["is_goal"].sum(), 1),
        "share_xg_after_first_touch":
            lambda s: s.loc[s["after_first_touch"], "xg"].sum() / max(s["xg"].sum(), 1e-9),
        "conv_direct": lambda s: conv(s, False),
        "conv_after_first_touch": lambda s: conv(s, True),
        "delta_aft_minus_direct": lambda s: conv(s, True) - conv(s, False),
        "obs_exp_direct": lambda s: _oe(s[s["segment"] == "direct"]),
        "obs_exp_after_first_touch": lambda s: _oe(s[s["after_first_touch"]]),
    }
    return cluster_bootstrap_stats(shots, stats, n_boot=boot, seed=0)


# ── The three origins inside after_clearance ────────────────────────────
def add_origin(shots: pd.DataFrame, deliv: pd.DataFrame) -> pd.DataFrame:
    """Tag every after_clearance shot with which kind of touch the defence dealt with.

    The three origins are defined in definitions.md §4, and `origin` is a field of s01
    (`ShotSegment.origin`) computed inside the grammar where the shot order is already known.
    What remains here is the same rule kept as an INVARIANT GUARD over the shipped column.

    The rule: the clearing contact IS the first touch (same instant, and the attack did not
    win it) makes it clearance_on_delivery; an earlier attacking shot inside the same delivery
    makes it clearance_on_shot; anything else is clearance_on_touch. It cannot read the contact
    type directly, because `first_contact` does not skip the shooter's own Ball Receipt*, so
    `first_contact_type` is almost never literally "Shot" even when a shot was the first thing
    that happened. The prior-shot test is what recovers that case."""
    cols = ["set_piece_id", "won_first_contact", "time_to_first_contact_s", "time_to_clearance_s"]
    s = shots.sort_values(["set_piece_id", "t_since_delivery_s"]).copy()
    shipped = s.pop("origin")
    s["_prior_shot"] = s.groupby("set_piece_id").cumcount() > 0
    # many_to_one: a duplicated set_piece_id on the delivery grain would silently fan the shot
    # grain out and misalign the guard below against the column it is checking.
    m = s.merge(deliv[cols], on="set_piece_id", how="left", validate="many_to_one")

    same = np.isclose(m["time_to_first_contact_s"], m["time_to_clearance_s"])
    on_delivery = same & (m["won_first_contact"] == False)           # noqa: E712  NaN must not pass
    on_shot = (~on_delivery) & m["_prior_shot"].to_numpy()
    origin = np.where(on_delivery, ORIGINS[0], np.where(on_shot, ORIGINS[1], ORIGINS[2]))
    m["origin"] = np.where(m["segment"] == "after_clearance", origin, None)
    disagree = int((shipped.to_numpy() != m["origin"].to_numpy()).sum())
    if disagree:
        raise AssertionError(f"s01 origin disagrees with the reconstruction on "
                             f"{disagree} of {len(m)} shots")
    return m.drop(columns=["_prior_shot"])


def origin_table(shots: pd.DataFrame, boot: int) -> pd.DataFrame:
    """The three origins, with volume, conversion and a clustered obs/exp each, over their shots;
    their own goals are counted apart."""
    ac = shots[shots["segment"] == "after_clearance"]
    own = is_own_goal(ac)
    g = ac[~own].groupby("origin")
    out = pd.DataFrame({"shots": g.size(), "goals": g["is_goal"].sum(),
                        "own_goals": ac[own].groupby("origin").size()})
    out[["shots", "goals", "own_goals"]] = out[["shots", "goals", "own_goals"]].fillna(0).astype(int)
    out["conv"] = out["goals"] / out["shots"]
    out["share"] = out["shots"] / out["shots"].sum()
    # reindex only over the origins actually present: on a small sample one of the three can be
    # empty, and reindexing over all of them would emit an all-NaN row that reads downstream as
    # a real cell with no shots.
    out = out.join(clustered_oe(ac, "origin", ORIGINS, boot)[
        ["sum_xg", "obs_exp", "lo", "hi", "poisson_lo", "poisson_hi", "excludes_1", "thin"]])
    return out.reindex([o for o in ORIGINS if o in out.index])


# ── The body-part cut, on the same estimator and the same draws ─────────
def add_body_cut(shots: pd.DataFrame) -> pd.DataFrame:
    """Tag every shot with the six-cell segment x body-part key, and with body part alone.

    Assigned without reordering the frame: the cluster draws are row positions at a fixed seed,
    so a sort here would move every block drawn from the same frame. An own goal gets no cell."""
    grp = pd.Series(np.where(shots["segment"] == "direct", "direct",
                             np.where(shots["segment"] == "late", "late", "after_first_touch")),
                    index=shots.index)
    part = pd.Series(np.where(shots["shot_body_part"].eq("Head"), "head", "non_head"),
                     index=shots.index).where(~is_own_goal(shots))
    return shots.assign(body_cell=grp + "|" + part, body_part=part)


def body_contrasts(shots: pd.DataFrame, boot: int) -> pd.DataFrame:
    """The three paired obs/exp differences, off ONE set of cluster draws.

    Two cells' marginal intervals cannot be differenced, and the difference is the reading a
    reader makes on sight: whether the segment gap survives holding the body part fixed, and
    whether the body-part gap survives at all. The first two are `direct` against the three
    after-first-touch segments; the third is over every attacking shot, `late` included."""
    stats = {name: (lambda f, a=a, b=b, c=("body_cell" if "|" in a else "body_part"):
                    _oe(f[f[c] == a]) - _oe(f[f[c] == b]))
             for name, a, b in BODY_PART_CONTRASTS}
    return cluster_bootstrap_stats(shots, stats, n_boot=boot, seed=0)


def _long(frame: pd.DataFrame, index_name: str, kind: str, block: str) -> pd.DataFrame:
    """One clustered_oe or bootstrap table, in frontier.parquet's long shape."""
    return (frame.reset_index().rename(columns={index_name: "key"})
            .assign(set_piece_type=kind, block=block))


# ── Rate, share and conversion, each on its own denominator ─────────────
def rate_share_conversion(shots: pd.DataFrame, deliv: pd.DataFrame, key: str) -> pd.DataFrame:
    """The three team-level quantities, side by side, with the denominator named.

    They get conflated constantly and they disagree on the same teams (definitions.md §8):
    the rate is output per delivery taken, the share is mix over the team's own shots, and
    the conversion is finishing inside the subset. Only the rate is the thesis's quantity, and it
    counts own goals, which are output; the share and the conversion count shots."""
    corners = deliv.groupby(key).size().rename("corners_taken")
    own, aft_rows = is_own_goal(shots), shots["segment"].isin(AFTER_FIRST_TOUCH)
    g, aft = shots[~own].groupby(key), shots[aft_rows & ~own].groupby(key)
    t = pd.DataFrame({"shots": g.size(), "goals": g["is_goal"].sum(),
                      "aft_shots": aft.size(), "aft_goals": aft["is_goal"].sum(),
                      "aft_scored": shots[aft_rows].groupby(key)["is_goal"].sum()})
    t = t.join(corners, how="outer").fillna(0)
    t["rate_aft_goals_per100_corners"] = 100 * t["aft_scored"] / t["corners_taken"]
    t["share_aft_of_own_shots"] = t["aft_shots"] / t["shots"].replace(0, np.nan)
    t["conversion_goals_per_aft_shot"] = t["aft_goals"] / t["aft_shots"].replace(0, np.nan)
    t["thin"] = np.where(t["aft_shots"] < MIN_CELL, f"<{MIN_CELL}", "")
    return t


def _fmt(df: pd.DataFrame) -> str:
    return df.to_string(float_format=lambda v: f"{v:.4f}")


def main() -> None:
    """Print the elementary analysis per population, then the four states, the rate, share and
    conversion grid, and one team's profile.

    Within a population the order is the reading order: diagnostics first, because a ratio
    computed on a broken table is worse than no ratio; then the segment table, because a rate
    is meaningless before its n; then the Q1 headline; then the body-part cut, the origins and
    the per-league cut, where the publication floor bites hardest and every thin cell is flagged
    rather than dropped."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--shots", type=str, default=str(_SHOTS),
                    help="the canonical two-population table; a corner-only table still runs "
                         "but writes frontier_shots.parquet instead of the canonical artefact")
    ap.add_argument("--deliveries", type=str, default=str(_DELIVERIES))
    ap.add_argument("--situation", type=str, default=str(_DER / "situation.parquet"))
    ap.add_argument("--boot", type=int, default=2000)
    ap.add_argument("--top", type=int, default=10)
    ap.add_argument("--team", type=str, default="Arsenal",
                    help="profile one team's own deliveries (illustration, never a ranking)")
    ap.add_argument("--out", type=str, default=None,
                    help="output parquet (default: frontier.parquet, but ONLY when --shots is "
                         "the canonical table; any other input tags the name)")
    args = ap.parse_args()

    # Only the grant's two-population table, decided by its path and not its name, writes the
    # canonical artefact report_figures summarises. A subset by matches or by population (the
    # corner-only table drops the free-kick half of the frontier block) writes a tagged name.
    stem = Path(args.shots).stem
    canonical = Path(args.shots).resolve() == _SHOTS.resolve()
    if args.out is None:
        args.out = str(_DER / ("frontier.parquet" if canonical else f"frontier_{stem}.parquet"))

    shots = pd.read_parquet(args.shots)
    deliv = pd.read_parquet(args.deliveries)
    print("=" * 78)
    print(f"ELEMENTARY ANALYSIS — {Path(args.shots).name}")
    print("=" * 78)
    print(f"{len(shots)} attacking shots over {shots['match_id'].nunique()} matches; "
          f"intervals resample MATCHES.")
    print("obs/exp is scored against the PROVIDER's xG, not ours: 'expected' is an external "
          "model.\n")

    blocks = []
    for kind, sub in shots.groupby("set_piece_type"):
        print(f"── {kind}: {len(sub)} shots ──")
        print("  diagnostics:", diagnose(sub))
        d = add_body_cut(sub.copy())

        seg = segment_table(d, args.boot)
        print("\n  segments (obs/exp interval is match-clustered; poisson_* is the exact Garwood bound)")
        print(_fmt(seg))
        hl = headline(d, args.boot)
        print("\n  Q1 headline")
        print(_fmt(hl))

        cells = clustered_oe(d, "body_cell", BODY_CELLS, args.boot)
        pooled = clustered_oe(d, "body_part", BODY_PARTS, args.boot)
        contr = body_contrasts(d, args.boot)
        print("\n  body part x segment (the cells sum back to the segment totals above)")
        print(_fmt(cells))
        print(_fmt(pooled))
        print("\n  paired obs/exp contrasts on the same draws (a delta covering 0 is not a finding)")
        print(_fmt(contr))

        dk = deliv[deliv["set_piece_type"] == kind]
        origins = origin_table(add_origin(d, dk), args.boot)
        print("\n  three origins inside after_clearance (definitions.md §4)")
        print(_fmt(origins))

        blocks.append(_long(seg, "segment", kind, "segment"))
        blocks.append(_long(origins, "origin", kind, "origin"))
        blocks.append(_long(cells, "body_cell", kind, "body_part"))
        blocks.append(_long(pooled, "body_part", kind, "body_part_pooled"))
        blocks.append(_long(contr, "stat", kind, "body_part_contrast"))
        blocks.append(_long(hl, "stat", kind, "headline"))
        lg = clustered_oe(d, "league", sorted(d["league"].unique()), args.boot)
        print("\n  per league, obs/exp on its own matches")
        print(_fmt(lg))
        blocks.append(_long(lg, "league", kind, "league"))
        print()

    # ── The four states, read at the drop and never in X_t ──────────────
    sit = Path(args.situation)
    if sit.exists():
        s8 = pd.read_parquet(sit)
        fs = s8[s8["four_state"].notna()]
        st = pd.DataFrame({"n": fs.groupby("four_state").size()})
        st["share"] = st["n"] / st["n"].sum()
        st["mean_xg_after"] = fs.groupby("four_state")["value_xg_after"].mean()
        st["goal_pct"] = 100 * fs.groupby("four_state")["value_goal_after"].mean()
        st["thin"] = np.where(st["n"] < MIN_CELL, f"<{MIN_CELL}", "")
        print(f"── the four states ({len(fs)} cleared corners with both halves known) ──")
        # FOUR_STATES is keyed by the (staked, won) pair, so the NAMES are its values.
        print(_fmt(st.reindex([s for s in FOUR_STATES.values() if s in st.index])))
        print()

    # ── Rate, share and conversion, on corners only ─────────────────────
    # Corner-only by construction, so a free-kick table skips it rather than printing an empty grid
    # that reads as a finding.
    corner_shots = shots[shots["set_piece_type"] == "corner"]
    corner_deliv = deliv[deliv["set_piece_type"] == "corner"]
    if corner_shots.empty:
        print("── rate, share and conversion: corners only, and this table carries none ──")
    else:
        rsc = rate_share_conversion(corner_shots, corner_deliv, "attacking_team")
        print("── rate, share and conversion (corners; three questions, three denominators) ──")
        print("  rate       = after-first-touch goals per 100 CORNERS TAKEN")
        print("  share      = after-first-touch shots over the team's OWN SET-PIECE SHOTS")
        print("  conversion = after-first-touch goals per AFTER-FIRST-TOUCH SHOT")
        print(_fmt(rsc.nlargest(args.top, "rate_aft_goals_per100_corners")))
        print(f"  [{int((rsc['thin'] != '').sum())} of {len(rsc)} teams sit under n={MIN_CELL} "
              f"after-first-touch shots and carry no publishable conversion]")

    if args.team and args.team in set(corner_shots["attacking_team"]):
        t = corner_shots[corner_shots["attacking_team"] == args.team]
        print(f"\n── {args.team}: {len(t)} attacking corner shots ──")
        print(_fmt(segment_table(t, args.boot)))
        print("  ILLUSTRATION, not an estimate. One team's goals split five ways cannot support "
              "a team-level claim, and the per-team rate's reliability is below the floor "
              "(measured.md §repeatability).")

    out = pd.concat(blocks, ignore_index=True)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(args.out)
    print(f"\nsaved {args.out}  ({len(out)} rows)  [gitignored]")
    print("  The segment obs/exp and its match-clustered interval are this artefact's. "
          "eda_second_ball.py computes an obs_exp column too, without an interval, and is "
          "descriptive-only.")


if __name__ == "__main__":
    main()
