"""eda_repeatability — is a team's second-ball profile a trait, or is it binomial noise?

The team leaderboard invites a claim it cannot support on its own. A team's season of corners
yields few goals (`measured.md` §teams carries the counts), and a share computed on that many
events carries a standard error wide enough that a table sorted by it will put someone at the
top whether or not teams differ at all. Two tests decide it.

  (1) Dispersion. Simulate every team's count at its own league's pooled probability and
      compare the observed between-team spread with that null: inside its band, the ordering
      is noise once the leagues are allowed to differ, and a league that converts better is
      not read as teams that do. One probability over every league rides beside it.
  (2) Split-half reliability. Split each team's matches in two, compute the metric on
      each half, correlate across teams, then Spearman-Brown up to full length. A trait
      the team carries reproduces on its own other half; a fluctuation does not.

Corner volume is the positive control. It is a real, stable team property, so a method
that cannot detect it is not measuring reliability, and the contrast between it and the
second-ball shares is what licenses the conclusion.

Split-half stands in for the year-over-year test this data cannot run. The grant is one
season, so a team's rate can only be checked against its own other matches, never against
its own next season. Two halves of one season share a squad and a manager, which a real
year-over-year test would not, so this is the optimistic bound on repeatability: whatever
it reports, the across-season figure would be no higher.

Both tests read the grain and the shots only, so this driver runs before eda_second_ball, whose team
table shrinks its ranking column by the reliability written here.

    python pipeline/eda_repeatability.py --splits 500
    python pipeline/eda_repeatability.py --splits 50 --boot 50 --out-dir data/derived/smoke \\
        --deliveries data/derived/eda_deliveries_limit200.parquet \\
        --shots data/derived/shots_corner_free_kick_flank_limit200.parquet    # a smoke's subset
"""

from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO / "src"))
sys.path.insert(0, str(_REPO / "pipeline"))
_DER = _REPO / "data" / "derived"

from eda_build import kind_inputs          # noqa: E402
from s01_phases import AFTER_FIRST_TOUCH, KINDS, is_own_goal   # noqa: E402  the one taxonomy, never re-typed

# The recipe's precision, and the only one the records in data/derived are written at.
_SPLITS, _SIMS, _BOOT = 500, 5000, 2000
# Test (1), (numerator, denominator) per team; the first is the rate the team table ranks on.
DISPERSION = (("aft_goals_for", "corners_for"), ("aft_goals_for", "goals_for"),
              ("aft_shots_for", "shots_for"), ("goals_for", "shots_for"),
              ("shots_for", "corners_for"))


def team_totals(deliv: pd.DataFrame, shots: pd.DataFrame) -> pd.DataFrame:
    """Per attacking team, the deliveries it took and the shots, goals, after-first-touch shots and
    goals they produced: the totals eda_second_ball's team table carries under these names, in its
    row order. Shots are shots and goals are every goal: an own goal adds to the goals and to no
    shot count (`s01_phases.is_own_goal`)."""
    key = ["league", "attacking_team"]
    own, after = is_own_goal(shots), shots["segment"].isin(AFTER_FIRST_TOUCH)
    g, aft = shots.groupby(key), shots[after].groupby(key)
    gs, aft_s = shots[~own].groupby(key), shots[after & ~own].groupby(key)
    t = pd.concat([deliv.groupby(key).size().rename("corners_for"), gs.size().rename("shots_for"),
                   g["is_goal"].sum().rename("goals_for"), aft_s.size().rename("aft_shots_for"),
                   aft["is_goal"].sum().rename("aft_goals_for")], axis=1).fillna(0)
    return t.rename_axis(["league", "team"]).reset_index()


def _null_sd(n: np.ndarray, p: float | np.ndarray, obs: float, n_sim: int,
             rng: np.random.Generator) -> tuple[float, float, float]:
    """(2.5%, 97.5%, Monte Carlo p) of the between-team SD when every team scores at `p`. The p
    is (1 + draws at least as wide) / (1 + draws), so it never reads zero."""
    sim = np.std(rng.binomial(n.astype(int), p, size=(n_sim, len(n))) / n, axis=1, ddof=1)
    lo, hi = np.percentile(sim, [2.5, 97.5])
    return float(lo), float(hi), float((1 + (sim >= obs).sum()) / (1 + n_sim))


def dispersion_test(t: pd.DataFrame, num: str, den: str, n_sim: int, seed: int = 0) -> dict:
    """Observed between-team SD of num/den against two binomial nulls, each on its own generator.

    `*_by_league` decides: every team at its own league's pooled probability, so what clears it is
    spread between teams of one league. The unsuffixed null pools the five leagues into one
    probability and counts a league difference as team excess; it is reported beside."""
    sub = t[t[den] > 0]
    k, n = sub[num].to_numpy(float), sub[den].to_numpy(float)
    p = k.sum() / n.sum()
    by_league = sub.groupby("league")
    p_league = (by_league[num].transform("sum") / by_league[den].transform("sum")).to_numpy(float)
    obs = np.std(k / n, ddof=1)
    lo, hi, p_value = _null_sd(n, p, obs, n_sim, np.random.default_rng(seed))
    lo_l, hi_l, p_l = _null_sd(n, p_league, obs, n_sim, np.random.default_rng([seed, 1]))
    return {"metric": f"{num}/{den}", "n_teams": len(sub), "pooled_p": p,
            "obs_sd": obs, "null_lo": lo, "null_hi": hi, "p_value": p_value, "excess": obs > hi,
            "null_lo_by_league": lo_l, "null_hi_by_league": hi_l, "p_value_by_league": p_l,
            "excess_by_league": obs > hi_l}


def _spearman_rows(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Spearman along the last axis, on the positions both arrays carry, NaN under six pairs.

    `scipy.stats.spearmanr` on the jointly finite pairs, row by row, is what `split_half` publishes;
    this is the same number vectorised over every leading axis, which is what makes a team bootstrap
    over hundreds of partitions affordable."""
    m = np.isfinite(a) & np.isfinite(b)
    ra = stats.rankdata(np.where(m, a, np.nan), axis=-1, nan_policy="omit")
    rb = stats.rankdata(np.where(m, b, np.nan), axis=-1, nan_policy="omit")
    with np.errstate(invalid="ignore", divide="ignore"):
        ca = ra - np.nanmean(ra, axis=-1, keepdims=True)
        cb = rb - np.nanmean(rb, axis=-1, keepdims=True)
        r = np.nansum(ca * cb, axis=-1) / np.sqrt(np.nansum(ca ** 2, axis=-1)
                                                  * np.nansum(cb ** 2, axis=-1))
    return np.where(m.sum(axis=-1) > 5, r, np.nan)


def _team_draws(leagues: np.ndarray, n_boot: int, rng: np.random.Generator) -> np.ndarray:
    """(n_boot, n_teams) team positions drawn with replacement INSIDE each league.

    Stratified because the leagues are fixed by the grant rather than sampled: a draw that handed
    one league six extra teams would mix league differences into the between-team spread."""
    cols = []
    for lg in pd.unique(leagues):
        pos = np.flatnonzero(leagues == lg)
        cols.append(pos[rng.integers(0, len(pos), size=(n_boot, len(pos)))])
    return np.hstack(cols)


def spearman_brown(r: float) -> float:
    """Half-length correlation corrected to full length, 2r / (1 + r), applied to the AVERAGED r.

    Averaging the per-split corrections instead of correcting the averaged correlation gives a
    different number."""
    return 2 * r / (1 + r) if np.isfinite(r) and r > -1 else np.nan


def split_half(deliv: pd.DataFrame, shots: pd.DataFrame, n_splits: int, seed: int = 0,
               scores: pd.DataFrame | None = None, n_boot: int = 0,
               boot_cols: list | None = None) -> pd.DataFrame:
    """Spearman-Brown corrected split-half reliability of five team metrics, plus any
    per-delivery score handed in.

    Matches, not deliveries, are the split unit: two corners in the same match share a
    scoreline, an opponent and a referee, so splitting deliveries would leak the match
    into both halves and inflate every correlation.

    `scores` is one row per delivery with `attacking_team`, `match_id` and one column per
    score, averaged over the team's deliveries in each half. It rides inside this loop
    rather than in a second one so every metric is correlated over the SAME draws: a score
    intervalled on its own draws could not be compared with the positive control, and that
    comparison is the whole point of reporting a control. Omitting it leaves the five
    metrics and the rng consumption untouched.

    The interval is over TEAMS, because the teams are the sample: the spread of r over partitions
    of the same teams holds them fixed and is not an interval of anything. `half_lo`/`half_hi` come
    from `n_boot` draws of teams with replacement inside each league, each draw scoring the mean r
    over the same partitions, which is the estimator `half_r` publishes, and they are NaN when
    `n_boot` is 0. The draws use their own generator, so `half_r` and `full_r_sb` do not move with
    `n_boot`. `full_lo`/`full_hi` carry the same bounds through Spearman-Brown, which is monotone.
    `boot_cols` narrows the draws to the metrics that publish an interval; the rest keep NaN bounds
    and cost nothing."""
    rng = np.random.default_rng(seed)
    teams = deliv.groupby(["league", "attacking_team"])["match_id"].unique()
    dl = deliv.set_index("attacking_team")
    sh = shots.assign(_own=is_own_goal(shots)).set_index("attacking_team")
    sc = scores.set_index("attacking_team") if scores is not None else None
    score_cols = ([c for c in scores.columns if c not in ("attacking_team", "match_id")]
                  if scores is not None else [])

    def metrics(team: str, mids: np.ndarray) -> dict:
        d = dl.loc[[team]] if team in dl.index else dl.iloc[:0]
        s = sh.loc[[team]] if team in sh.index else sh.iloc[:0]
        d, s = d[d["match_id"].isin(mids)], s[s["match_id"].isin(mids)]
        aft = s["segment"].isin(AFTER_FIRST_TOUCH)
        shot = s[~s["_own"]]           # an own goal is a goal and no shot
        out = {"corners_per_match": len(d) / max(len(mids), 1),
               "shots_per_corner": len(shot) / len(d) if len(d) else np.nan,
               "aft_shot_share": shot["segment"].isin(AFTER_FIRST_TOUCH).mean() if len(shot) else np.nan,
               "goals_per100": 100 * s["is_goal"].sum() / len(d) if len(d) else np.nan,
               # the metric eda_second_ball.py actually RANKS teams on (its "top 15"
               # table); goals_per100 above is ALL goals and is a different quantity
               "aft_goals_per100": 100 * s.loc[aft, "is_goal"].sum() / len(d) if len(d) else np.nan}
        if sc is not None:
            z = sc.loc[[team]] if team in sc.index else sc.iloc[:0]
            out.update(z[z["match_id"].isin(mids)][score_cols].mean().to_dict())
        return out

    cols = ["corners_per_match", "shots_per_corner", "aft_shot_share", "goals_per100",
            "aft_goals_per100"] + score_cols
    acc = {c: [] for c in cols}
    halves_a, halves_b = [], []
    for _ in range(n_splits):
        a_rows, b_rows = [], []
        for (_, team), mids in teams.items():
            mids = np.asarray(mids)
            pick = rng.permutation(len(mids))
            half = len(mids) // 2
            a_rows.append(metrics(team, mids[pick[:half]]))
            b_rows.append(metrics(team, mids[pick[half:]]))
        A, B = pd.DataFrame(a_rows)[cols], pd.DataFrame(b_rows)[cols]
        for c in cols:
            m = A[c].notna() & B[c].notna()
            acc[c].append(stats.spearmanr(A.loc[m, c], B.loc[m, c])[0] if m.sum() > 5 else np.nan)
        halves_a.append(A.to_numpy(float).T)
        halves_b.append(B.to_numpy(float).T)

    boot = np.full((max(n_boot, 0), len(cols)), np.nan)
    if n_boot:
        keep = [j for j, c in enumerate(cols) if boot_cols is None or c in boot_cols]
        a_all = np.stack(halves_a)[:, keep]                             # (splits, metrics, teams)
        b_all = np.stack(halves_b)[:, keep]
        draws = _team_draws(teams.index.get_level_values("league").to_numpy(), n_boot,
                            np.random.default_rng([seed, 1]))
        for i, d in enumerate(draws):
            with np.errstate(all="ignore"), warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                boot[i, keep] = np.nanmean(_spearman_rows(a_all[:, :, d], b_all[:, :, d]), axis=0)

    rows = []
    for j, c in enumerate(cols):
        r = np.nanmean(acc[c])
        drawn = boot[:, j]
        lo, hi = (np.nanpercentile(drawn, [2.5, 97.5]) if np.isfinite(drawn).any()
                  else (np.nan, np.nan))
        rows.append({"metric": c, "half_r": r, "half_lo": lo, "half_hi": hi,
                     "full_r_sb": spearman_brown(r), "full_lo": spearman_brown(lo),
                     "full_hi": spearman_brown(hi),
                     "n_splits": n_splits, "n_boot": n_boot})
    return pd.DataFrame(rows)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--splits", type=int, default=_SPLITS)
    ap.add_argument("--sims", type=int, default=_SIMS)
    ap.add_argument("--boot", type=int, default=_BOOT,
                    help="team draws, inside each league, behind the split-half interval")
    ap.add_argument("--kind", type=str, default="corner", choices=list(KINDS))
    ap.add_argument("--out-dir", type=str, default=None,
                    help="where to write (default: data/derived, and only at the recipe's "
                         "precision); a reduced run exercises the path here")
    ap.add_argument("--deliveries", type=str, default=None, help="default: the grant's grain")
    ap.add_argument("--shots", type=str, default=None, help="default: the grant's shots")
    args = ap.parse_args()
    out = Path(args.out_dir) if args.out_dir else _DER
    if (out.resolve() == _DER.resolve()
            and (args.splits, args.sims, args.boot) != (_SPLITS, _SIMS, _BOOT)):
        raise SystemExit(f"STOP: --splits {args.splits} --sims {args.sims} --boot {args.boot} is "
                         f"not the recipe's {_SPLITS}/{_SIMS}/{_BOOT}, so it may not write the "
                         f"published records; pass --out-dir")
    d_def, s_def = kind_inputs(args.kind)
    d_path, s_path = Path(args.deliveries or d_def), Path(args.shots or s_def)
    if (out.resolve() == _DER.resolve()
            and (d_path.resolve(), s_path.resolve()) != (d_def.resolve(), s_def.resolve())):
        raise SystemExit("STOP: --deliveries or --shots is not the grant's, so it may not write the "
                         "published records; pass --out-dir")

    deliv, shots = pd.read_parquet(d_path), pd.read_parquet(s_path)
    deliv = deliv[deliv["set_piece_type"] == args.kind]
    shots = shots[shots["set_piece_type"] == args.kind]
    t = team_totals(deliv, shots)

    print(f"{'=' * 78}\nE. IS IT A TRAIT? — {args.kind}, {len(t)} teams\n{'=' * 78}")
    print("\n  (1) dispersion against each league's own probability, which decides, and against one "
          "probability over every league:")
    disp = pd.DataFrame([dispersion_test(t, num, den, args.sims) for num, den in DISPERSION])
    print(disp.round(4).to_string(index=False))
    print("\n      excess_by_league=True means teams of one league differ beyond binomial noise;")
    print("      excess=True alone can be the leagues differing.")

    print(f"\n  (2) split-half reliability over {args.splits} random splits, intervalled over "
          f"{args.boot} team draws:")
    rel = split_half(deliv, shots, args.splits, n_boot=args.boot)
    print(rel.round(3).to_string(index=False))
    print("\n      corners_per_match is the positive control: it is a real team property,")
    print("      so a near-zero r there would mean the test itself is broken.")

    out.mkdir(parents=True, exist_ok=True)
    disp.to_parquet(out / f"eda_dispersion_{args.kind}.parquet")
    rel.to_parquet(out / f"eda_reliability_{args.kind}.parquet")
    print(f"\nsaved {out}/eda_dispersion_{args.kind}.parquet and eda_reliability_{args.kind}.parquet")


if __name__ == "__main__":
    main()
