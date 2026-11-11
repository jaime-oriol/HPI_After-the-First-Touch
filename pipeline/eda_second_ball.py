"""eda_second_ball — the descriptive layer: segment, league, team, and finishing position.

Answers three questions the modelling layer does not:

  (a) volume against rate. The direct shot supplies the most goals of any segment while
      converting worst, so a leaderboard ordered by goals and one ordered by conversion
      disagree, and only one of them is about the second ball.
  (b) whether the segment mix is a league artefact.
  (c) whether teams that finish high are the ones living after the first touch.

On (c): the honest primary is a rank correlation over every team, not a top-N split.
A split discards the ordering inside both groups and invites shopping for the cut that
works; the correlation uses every team and has one degree of freedom. The splits are
reported too, at three cuts, so a reader can see the answer does not hinge on the cut.

Segment deltas resample matches, the position correlations resample teams within league, and
conversion carries a Wilson interval. A cell under MIN_CELL is printed with a flag and never
carries a claim.

OVERLAP WITH describe_frontier, declared rather than left for a reader to notice. Block A
computes an `obs_exp` column too, and block C an `obs_exp_{seg}_for` per team that
`eda_teams_{kind}.parquet` persists, and neither is the published one: both carry no interval,
and the figure the paper cites comes from `describe_frontier.py`, which bounds it with a
match-clustered bootstrap and persists it to `frontier.parquet`. What is unique here is the
paired delta against `direct`, the per-team table and the finishing-position correlation.
Read those obs/exp columns as a sanity check on the same arithmetic, never as the source.

    python pipeline/eda_second_ball.py --boot 2000
    python pipeline/eda_second_ball.py --boot 100 --out-dir data/derived/smoke \\
        --deliveries data/derived/eda_deliveries_limit200.parquet \\
        --shots data/derived/shots_corner_free_kick_flank_limit200.parquet \\
        --records data/derived/smoke                     # a smoke's subset and its own records
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO / "src"))
sys.path.insert(0, str(_REPO / "pipeline"))
_DER = _REPO / "data" / "derived"

from constants import MIN_CELL, RELIABILITY_FLOOR      # noqa: E402
from eda_build import kind_inputs                      # noqa: E402
from s01_phases import AFTER_FIRST_TOUCH, KINDS, SEGMENTS, is_own_goal   # noqa: E402
from validate import Z95, cluster_bootstrap_stats      # noqa: E402

SEG_ORDER = list(SEGMENTS)      # print order, from the one taxonomy that owns it
# The recipe's draws, and the only ones the published records are written at.
_BOOT = 2000
# The two cases argument.md §2 opens on: the champion and the side that survived on the last weekend.
NAMED_CASES = ("Arsenal", "Tottenham Hotspur")


def _load_reliability(kind: str, records: Path | None = None) -> "pd.DataFrame | None":
    """eda_repeatability.py's split-half table for `kind`, read from `records` (default:
    data/derived) and re-indexed onto team_table's own column names, or None if that driver
    has never written there.

    eda_repeatability.py reports unsuffixed metric names (`goals_per100`); every one
    of them is the attacking (FOR) side, which here is suffixed (`goals_per100_for`),
    except `corners_per_match`, which team_table carries unsuffixed already."""
    p = (records or _DER) / f"eda_reliability_{kind}.parquet"
    if not p.exists():
        return None
    rel = pd.read_parquet(p).set_index("metric")
    rel.index = [m if m == "corners_per_match" else f"{m}_for" for m in rel.index]
    return rel


def _uncrossed_goals(t: pd.DataFrame, kind: str, records: Path | None = None) -> pd.Series:
    """Per team, the goals that followed a short or ground corner no cross reached in time, which
    run_short_corner_sensitivity.py counts. Such a corner has no delivery, so those goals are in no
    segment and not in `goals_for`; a team's corner goals told whole are the two together.

    Only corners have short corners, so any other kind holds none. A records folder without the
    artefact leaves them unknown (NaN), never zero: a missing count must not read as no goals."""
    if kind != "corner":
        return pd.Series(0, index=t.index)
    p = (records or _DER) / "short_corner_sensitivity.parquet"
    if not p.exists():
        return pd.Series(np.nan, index=t.index)
    u = pd.read_parquet(p)
    u = u[u["block"] == "uncrossed_team"]
    by = dict(zip(u["key"], u["n"].astype(int), strict=True))
    return pd.Series([by.get(f"{lg}|{tm}", 0) for lg, tm in zip(t["league"], t["team"], strict=True)], index=t.index)


def _shrink(rate: pd.Series, reliability: float, league_rate: float) -> pd.Series:
    """Efron & Morris (1975) empirical-Bayes shrinkage toward `league_rate`, the mean over every
    team the table holds (team_table pools the five leagues), at the single reliability measured
    across all teams (methods_second_ball.md §4).

    A reliability in (0, 1] shared by every team is a positive affine map of `rate`, so it
    can never change which team ranks above which, only how far apart they look. That is
    the point of printing it alongside the raw column, not instead of it: the raw
    spread overstates how different these teams actually are. Outside that range the map
    reverses or inflates the ranking it claims to preserve, so it raises instead."""
    if not 0.0 < reliability <= 1.0:
        raise ValueError(f"_shrink: reliability {reliability} outside (0, 1]")
    return reliability * rate + (1 - reliability) * league_rate


def _reliability_note(rel: "pd.DataFrame | None", col: str) -> str:
    """One line: the measured split-half reliability for `col`, or an explicit
    admission that it has not been measured, never silence, which would read as
    'fine to trust'."""
    if rel is None:
        return f"  [reliability UNMEASURED for {col} — run eda_repeatability.py first]"
    if col not in rel.index:
        return f"  [reliability UNMEASURED for {col} — eda_repeatability.py does not cover it]"
    r = float(rel.loc[col, "full_r_sb"])
    band = ("clears the Koo & Li good-band floor, publishable as a ranking" if r >= RELIABILITY_FLOOR
            else f"BELOW the Koo & Li good-band floor ({RELIABILITY_FLOOR}) — mostly noise")
    return f"  [{col}: split-half reliability {r:.3f}, {band}]"


def _wilson(k: int, n: int, z: float = Z95) -> tuple:
    """Wilson score interval. Beats normal-approximation on the sparse cells here."""
    if n == 0:
        return (np.nan, np.nan)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (100 * (c - h), 100 * (c + h))


def _exact_power(n: int, p0: float, p1: float, alpha: float = 0.05) -> float:
    """Power of the two-sided exact binomial test of p1 against p0 over n trials.

    The upper tail is the one a share above the pooled rate is read from, so the critical
    value is the smallest count the test rejects at alpha/2 there. This is the equal-tailed
    test; `aft_binom_p` beside it is scipy's two-sided one, which orders outcomes by likelihood.
    The two can reject on different counts, so `aft_goals_for_80pct_power` is this test's figure,
    not binomtest's."""
    if n <= 0 or not np.isfinite(p1):
        return float("nan")
    k = float(stats.binom.isf(alpha / 2, n, p0)) + 1
    return float(stats.binom.sf(k - 1, n, p1))


def _n_for_power(p0: float, p1: float, target: float = 0.80, n_max: int = 4000) -> float:
    """Trials that same test needs to reach `target` power, NaN if it never does by n_max.

    Undefined below the pooled rate: a share that low is not the effect being powered for."""
    if not np.isfinite(p1) or p1 <= p0:
        return float("nan")
    for n in range(2, n_max + 1):
        if _exact_power(n, p0, p1) >= target:
            return float(n)
    return float("nan")


def _rate(sub: pd.DataFrame, seg: str) -> float:
    """Conversion of a segment's shots; an own goal is a goal and no shot, so it is not in the rate."""
    m = (sub["segment"] == seg) & ~is_own_goal(sub)
    return float(sub.loc[m, "is_goal"].mean()) if m.any() else np.nan


# ── A. the frontier: volume against rate ──────────────────────────────────────

def segment_table(shots: pd.DataFrame, boot: int, label: str) -> pd.DataFrame:
    """Per-segment counts, conversion with Wilson and clustered CIs, and paired deltas vs direct.

    Shots are shots and goals are every goal: an own goal (`s01_phases.is_own_goal`) is counted
    apart and in `goal_share`, and in no conversion, share of shots or ratio over xG."""
    own = is_own_goal(shots)
    print(f"\n{'=' * 78}\nA. SEGMENTS — {label}   ({int((~own).sum())} shots, "
          f"{int(shots.is_goal.sum())} goals, {int(own.sum())} of them own goals)\n{'=' * 78}")

    g = shots[~own].groupby("segment").agg(shots=("is_goal", "size"), goals=("is_goal", "sum"),
                                           xg=("xg", "sum"))
    g["own_goals"] = shots[own].groupby("segment").size().reindex(g.index).fillna(0).astype(int)
    g = g.reindex([s for s in SEG_ORDER if s in g.index])
    g["conv_pct"] = 100 * g.goals / g.shots
    g["shot_share"] = 100 * g.shots / g.shots.sum()
    scored = g.goals + g.own_goals
    g["goal_share"] = 100 * scored / scored.sum()
    g["obs_exp"] = g.goals / g.xg
    w = [_wilson(int(r.goals), int(r.shots)) for r in g.itertuples()]
    g["conv_lo"], g["conv_hi"] = [x[0] for x in w], [x[1] for x in w]
    g["flag"] = np.where(g.shots < MIN_CELL, f"<{MIN_CELL}", "")
    print(g.round(2).to_string())

    stat = {f"d_{s}": (lambda d, s=s: _rate(d, s) - _rate(d, "direct")) for s in SEG_ORDER
            if s != "direct" and s in g.index}
    if stat:
        ci = cluster_bootstrap_stats(shots, stat, group_col="match_id", n_boot=boot)
        print(f"\n  paired delta vs direct (pp), match-clustered, {boot} draws:")
        for r in ci.itertuples():
            sig = "*" if (r.lo > 0 or r.hi < 0) else " "
            print(f"   {sig} {r.Index:>14s}  "
                  f"{100*r.point:+6.2f}  [{100*r.lo:+6.2f}, {100*r.hi:+6.2f}]")
    return g


# ── B. league ─────────────────────────────────────────────────────────────────

def league_table(shots: pd.DataFrame) -> pd.DataFrame:
    """Per-league shots, goals, segment mix and conversion, with the cells under MIN_CELL listed
    rather than hidden."""
    print(f"\n{'=' * 78}\nB. BY LEAGUE — segment mix and conversion\n{'=' * 78}")
    rows = []
    for lg, s in shots.groupby("league"):
        real = s[~is_own_goal(s)]      # shots; the goals below count every goal, own goals included
        r = {"league": lg, "shots": len(real), "goals": int(s.is_goal.sum())}
        for seg in SEG_ORDER:
            m = real["segment"] == seg
            r[f"n_{seg}"] = int(m.sum())
            r[f"conv_{seg}"] = round(100 * real.loc[m, "is_goal"].mean(), 2) if m.sum() else np.nan
        aft = s["segment"].isin(AFTER_FIRST_TOUCH)
        r["aft_shot_pct"] = round(100 * real["segment"].isin(AFTER_FIRST_TOUCH).mean(), 1)
        r["aft_goal_pct"] = round(100 * s.loc[aft, "is_goal"].sum() / max(s.is_goal.sum(), 1), 1)
        rows.append(r)
    t = pd.DataFrame(rows)
    print(t[["league", "shots", "goals"] + [f"n_{s}" for s in SEG_ORDER] +
            ["aft_shot_pct", "aft_goal_pct"]].to_string(index=False))
    print(f"\n  conversion % by segment (cells under n={MIN_CELL} are unreliable):")
    print(t[["league"] + [f"conv_{s}" for s in SEG_ORDER]].to_string(index=False))
    thin = [(r.league, s, r._asdict()[f"n_{s}"]) for r in t.itertuples()
            for s in SEG_ORDER if r._asdict()[f"n_{s}"] < MIN_CELL]
    print(f"\n  cells under n={MIN_CELL}: {thin if thin else 'none'}")
    return t


# ── C. team ───────────────────────────────────────────────────────────────────

def team_table(deliv: pd.DataFrame, shots: pd.DataFrame, stand: pd.DataFrame, kind: str,
               records: Path | None = None) -> pd.DataFrame:
    """One row per team: attacking and defending, every rate on its own denominator, the ranking
    column shrunk by the reliability eda_repeatability wrote into `records`."""
    print(f"\n{'=' * 78}\nC. BY TEAM — normalised on corners taken / faced\n{'=' * 78}")
    att = deliv.groupby(["league", "attacking_team"]).size().rename("corners_for")
    dfn = deliv.groupby(["league", "defending_team"]).size().rename("corners_against")
    att.index.names = dfn.index.names = ["league", "team"]

    def side(df: pd.DataFrame, who: str, suf: str) -> pd.DataFrame:
        """One side's totals, its own-goal count and its full segment breakdown.

        The segment columns are the primitives (shots, goals, xG) and nothing derived, so a
        consumer can build any of the three team quantities §8 of `definitions.md` separates
        without re-deriving the denominators. A forced own goal is a goal the delivery produced
        and no shot (`s01_phases.is_own_goal`): it adds to `goals`, `aft_goals`, the segment's `g_`
        and its own `og_` counts, and to no shot count nor xG, so a conversion or a ratio over xG
        takes the goals without the own ones."""
        own, keys = is_own_goal(df), ["league", who]
        after = df["segment"].isin(AFTER_FIRST_TOUCH)
        out = df[~own].groupby(keys).agg(**{f"shots{suf}": ("is_goal", "size"),
                                            f"xg{suf}": ("xg", "sum")})
        out[f"goals{suf}"] = df.groupby(keys)["is_goal"].sum()
        out[f"aft_shots{suf}"] = df[after & ~own].groupby(keys).size()
        out[f"aft_goals{suf}"] = df[after].groupby(keys)["is_goal"].sum()
        out[f"og_goals{suf}"] = df[own].groupby(keys).size()
        for seg in SEG_ORDER:
            in_seg = df["segment"] == seg
            out[f"n_{seg}{suf}"] = df[in_seg & ~own].groupby(keys).size()
            out[f"g_{seg}{suf}"] = df[in_seg].groupby(keys)["is_goal"].sum()
            out[f"og_{seg}{suf}"] = df[in_seg & own].groupby(keys).size()
            out[f"xg_{seg}{suf}"] = df[in_seg & ~own].groupby(keys)["xg"].sum()
        out.index.names = ["league", "team"]
        return out.fillna(0)

    t = pd.concat([att, dfn, side(shots, "attacking_team", "_for"),
                   side(shots, "defending_team", "_ag")], axis=1).fillna(0).reset_index()
    t = t.merge(stand[["league", "team", "team_ref", "pos", "pts_ref", "played"]],
                on=["league", "team"], how="left")
    assert t["pos"].notna().all(), "team without a finishing position"
    t["goals_uncrossed_for"] = _uncrossed_goals(t, kind, records)
    t["goals_for_all"] = t["goals_for"] + t["goals_uncrossed_for"]

    for suf, den in (("_for", "corners_for"), ("_ag", "corners_against")):
        t[f"shots_per_corner{suf}"] = t[f"shots{suf}"] / t[den]
        t[f"goals_per100{suf}"] = 100 * t[f"goals{suf}"] / t[den]
        t[f"aft_goals_per100{suf}"] = 100 * t[f"aft_goals{suf}"] / t[den]
        t[f"aft_shot_share{suf}"] = t[f"aft_shots{suf}"] / t[f"shots{suf}"].replace(0, np.nan)
        t[f"aft_goal_share{suf}"] = t[f"aft_goals{suf}"] / t[f"goals{suf}"].replace(0, np.nan)
    # Share, conversion and obs/exp per segment, on the attacking side only. Each is on its own
    # denominator, which is the distinction §8 of definitions.md exists to keep: a share is mix
    # over the team's own shots, a conversion is finishing inside the segment, and obs/exp prices
    # the segment against the provider's model. Under MIN_CELL the conversion is left NaN; the
    # shares and obs/exp keep the thin cells, with `n_{seg}_for` beside them.
    for seg in SEG_ORDER:
        n, goals, xg = t[f"n_{seg}_for"], t[f"g_{seg}_for"], t[f"xg_{seg}_for"]
        shot_goals = goals - t[f"og_{seg}_for"]
        thin = n.where(n >= MIN_CELL)
        t[f"shot_share_{seg}_for"] = n / t["shots_for"].replace(0, np.nan)
        t[f"goal_share_{seg}_for"] = goals / t["goals_for"].replace(0, np.nan)
        t[f"conv_{seg}_for"] = 100 * shot_goals / thin
        t[f"obs_exp_{seg}_for"] = shot_goals / xg.where(xg > 0)
    t["og_goal_share_for"] = t["og_goals_for"] / t["goals_for"].replace(0, np.nan)
    # A per-team share is an illustration, never a finding, so it travels with the power the
    # test behind it actually had. The two cases §2 of argument.md names sit on a handful of
    # corner goals each (measured.md §teams), and at that n an exact binomial cannot resolve the
    # gap it is being read for, so quoting the share without this column invites a reader to
    # treat a question the design could not answer as an answer.
    p0 = float(t["aft_goals_for"].sum() / t["goals_for"].sum())
    kn = list(zip(t["aft_goals_for"].astype(int), t["goals_for"].astype(int), strict=True))
    t["aft_share_pooled"] = p0
    t["aft_binom_p"] = [float(stats.binomtest(k, n, p0).pvalue) if n else np.nan for k, n in kn]
    t["aft_power_at_n"] = [_exact_power(n, p0, k / n) if n else np.nan for k, n in kn]
    t["aft_goals_for_80pct_power"] = [_n_for_power(p0, k / n) if n else np.nan for k, n in kn]
    t["corners_per_match"] = t["corners_for"] / t["played"]
    # Rank 0 = champion, 1 = bottom, so an 18-team league is comparable with a 20-team one.
    t["pos_pct"] = t.groupby("league")["pos"].transform(lambda s: (s - 1) / (s.max() - 1))

    rank_col = "aft_goals_per100_for"
    rel = _load_reliability(kind, records)
    league_rate = float(t[rank_col].mean())
    r_val = float(rel.loc[rank_col, "full_r_sb"]) if (rel is not None and rank_col in rel.index) else None
    # At or below zero the rate does not reproduce at all, and a NaN (halves that never vary) measures
    # nothing: either way there is no signal to shrink toward.
    if r_val is not None and not r_val > 0:
        r_val = None
    t[f"{rank_col}_shrunk"] = (_shrink(t[rank_col], r_val, league_rate)
                              if r_val is not None else np.nan)

    show = ["team", "league", "pos", "corners_for", "shots_for", "goals_for", "aft_goals_for",
            "corners_per_match", "goals_per100_for", rank_col, f"{rank_col}_shrunk", "aft_goal_share_for"]
    print("\n  top 15 by goals from corners after the first touch, per 100 corners "
          f"(n={len(t)} teams):")
    print(_reliability_note(rel, rank_col))
    if r_val is not None and r_val < RELIABILITY_FLOOR:
        print(f"  [the ranking below is sorted by the RAW column, not the shrunk one — a single "
              f"shared reliability is order-preserving, so '{rank_col}_shrunk' changes the honest "
              f"MAGNITUDE only, never the row order]")
    print(t.nlargest(15, rank_col)[show].round(3).to_string(index=False))
    print("\n  the champions' rows:")
    champs = t[t["pos"] == 1]
    print(champs[show].round(3).to_string(index=False))
    return t


# ── D. finishing position ─────────────────────────────────────────────────────

METRICS = [("corners_per_match", "corners taken per match"),
           ("shots_per_corner_for", "shots per corner"),
           ("goals_per100_for", "goals per 100 corners"),
           ("aft_goals_per100_for", "after-1st-touch goals per 100 corners"),
           ("aft_shot_share_for", "share of shots after the 1st touch"),
           ("aft_goal_share_for", "share of goals after the 1st touch"),
           ("goals_per100_ag", "goals CONCEDED per 100 corners faced"),
           ("aft_goals_per100_ag", "after-1st-touch goals conceded per 100")]


def position_analysis(t: pd.DataFrame, boot: int, kind: str, seed: int = 0,
                      records: Path | None = None) -> pd.DataFrame:
    """Spearman of each metric against finishing position, plus three top-N splits.

    Negative rho means the metric is higher for teams that finish HIGH (pos_pct rises
    towards the bottom of the table). The bootstrap resamples teams within league, which
    is the unit the correlation is computed over.

    A correlation is only as trustworthy as the metric feeding it: ranking teams by an
    input whose split-half reliability is mostly noise makes the correlation mostly
    noise too, so every row is tagged with the reliability eda_repeatability measured for
    that column, not re-derived here."""
    print(f"\n{'=' * 78}\nD. FINISHING POSITION — rank correlation over all {len(t)} teams\n{'=' * 78}")
    print("  rho < 0 means the metric is HIGHER for teams that finish higher.\n")
    rel = _load_reliability(kind, records)
    rng = np.random.default_rng(seed)
    idx_by_lg = [g.index.to_numpy() for _, g in t.groupby("league")]

    rows = []
    for col, name in METRICS:
        sub = t[["pos_pct", col]].dropna()
        rho, p = stats.spearmanr(sub["pos_pct"], sub[col])
        draws = []
        for _ in range(boot):
            pick = np.concatenate([rng.choice(ix, size=len(ix), replace=True) for ix in idx_by_lg])
            d = t.loc[pick, ["pos_pct", col]].dropna()
            draws.append(stats.spearmanr(d["pos_pct"], d[col])[0] if len(d) > 5 else np.nan)
        lo, hi = np.nanpercentile(draws, [2.5, 97.5])
        sig = "*" if (lo > 0 or hi < 0) else " "
        r_val = float(rel.loc[col, "full_r_sb"]) if (rel is not None and col in rel.index) else None
        rows.append({"metric": col, "rho": rho, "lo": lo, "hi": hi, "p": p, "n": len(sub),
                     "input_reliability": r_val})
        print(f" {sig} {name:<44s} rho {rho:+.3f}  [{lo:+.3f}, {hi:+.3f}]  p={p:.4f}")
        print(_reliability_note(rel, col))

    print("\n  top-N against the rest (three cuts, so the answer does not hinge on one):")
    for cut in (3, 5, 6):
        top = t[t["pos"] <= cut]
        rest = t[t["pos"] > cut]
        print(f"\n   --- top {cut} (n={len(top)}) vs rest (n={len(rest)}) ---")
        for col, name in METRICS:
            a, b = top[col].dropna(), rest[col].dropna()
            if len(a) < 5 or len(b) < 5:
                continue
            u, p = stats.mannwhitneyu(a, b)
            print(f"      {name:<44s} {a.mean():7.3f} vs {b.mean():7.3f}   p={p:.4f}"
                  f"{'  *' if p < 0.05 else ''}")
    return pd.DataFrame(rows)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--boot", type=int, default=_BOOT)
    ap.add_argument("--kind", type=str, default="corner", choices=list(KINDS))
    ap.add_argument("--out-dir", type=str, default=None,
                    help="where to write (default: data/derived)")
    ap.add_argument("--deliveries", type=str, default=None, help="default: the grant's grain")
    ap.add_argument("--shots", type=str, default=None, help="default: the grant's shots")
    ap.add_argument("--records", type=str, default=str(_DER),
                    help="where eda_standings and eda_repeatability wrote (default: data/derived)")
    args = ap.parse_args()
    out = Path(args.out_dir) if args.out_dir else _DER
    if out.resolve() == _DER.resolve() and args.boot != _BOOT:
        raise SystemExit(f"STOP: --boot {args.boot} is not the recipe's {_BOOT}, so it may not write "
                         f"the published records; pass --out-dir")
    d_def, s_def = kind_inputs(args.kind)
    d_path, s_path = Path(args.deliveries or d_def), Path(args.shots or s_def)
    records = Path(args.records)
    grant = ((d_path.resolve(), s_path.resolve(), records.resolve())
             == (d_def.resolve(), s_def.resolve(), _DER.resolve()))
    if out.resolve() == _DER.resolve() and not grant:
        raise SystemExit("STOP: --deliveries, --shots or --records is not the grant's, so it may not "
                         "write the published records; pass --out-dir")

    deliv, shots = pd.read_parquet(d_path), pd.read_parquet(s_path)
    stand = pd.read_parquet(records / "eda_standings.parquet")
    deliv = deliv[deliv["set_piece_type"] == args.kind]
    shots = shots[shots["set_piece_type"] == args.kind]
    # shots carries the attacking side only; the defending side comes off the delivery.
    shots = shots.merge(deliv[["set_piece_id", "defending_team"]],
                        on="set_piece_id", how="left", suffixes=("", "_d"))
    assert shots["defending_team"].notna().all(), "shot with no defending team"
    print(f"{args.kind}: {len(deliv)} deliveries, {len(shots)} attacking shots, "
          f"{int(shots.is_goal.sum())} goals, {deliv.match_id.nunique()} matches")

    segment_table(shots, args.boot, args.kind)
    league_table(shots)
    t = team_table(deliv, shots, stand, args.kind, records)
    pos = position_analysis(t, args.boot, args.kind, records=records)

    out.mkdir(parents=True, exist_ok=True)
    t.to_parquet(out / f"eda_teams_{args.kind}.parquet")
    pos.to_parquet(out / f"eda_position_{args.kind}.parquet")
    print(f"\nsaved {out}/eda_teams_{args.kind}.parquet and eda_position_{args.kind}.parquet")


if __name__ == "__main__":
    main()
