"""falsify_frontier — is shot.key_pass_id football, or an annotation artefact?

The "after the first touch" frontier (form_answers.md Q1) is built from the event
chain (s01). This driver tests whether the provider's `shot.key_pass_id` agrees with
it: that field is assigned by the provider, and if it is assigned by a rule correlated
with the outcome the direct-vs-prolongation delta would come for free.

Four tests, and the frontier is only readable as football if all four pass. They run on
every full pass of the recipe:

  1 annotation density  does a shot carry a key pass more often BECAUSE it was a
                        goal? xG-stratified rate difference (primary) plus a ridge
                        logit, both with match-clustered CIs.
                        NULL: no is_goal effect on has_key_pass.
  2 vendor cross-tab    our taxonomy against the provider's `set_piece_phase`,
                        restricted to corners that produced a shot: pooling
                        shotless corners inflates the agreement, because
                        agreeing there means nothing.
  3 outcome-blind       rebuild the split from the event-type chain between
                        delivery and shot, never touching key_pass_id, and report
                        the conversion under both definitions, with paired CIs on
                        the chain's.
  4 separability        time and touch count from delivery to shot, per class,
                        with a rank AUC. If direct and prolongation overlap, the
                        label is not tracking a physical phase.

The same pass measures the legacy-vs-current contact rules evidence.md §3 prices, since it
already walks every corner: Ball Receipt* missing from the first-contact set, and Goal Keeper
counted wholesale as a clearing contact. It also produces the Q1 headline (what
share of dead-ball value lands after the first contact) and the per-league mix
that prefigures the league-out split. --kinds free_kick_flank runs the same pass over
the flank free kicks from x >= 85, the paper's second population.

    python pipeline/falsify_frontier.py --limit 5                 # smoke test
    python pipeline/falsify_frontier.py --jobs 20 --boot 2000      # the blocking run
    python pipeline/falsify_frontier.py --jobs 20 --kinds free_kick_flank   # falsification_<kind>
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy.stats import rankdata
from sklearn.linear_model import LogisticRegression

_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO / "src"))
sys.path.insert(0, str(_REPO / "pipeline"))

import loader_sb as L                                        # noqa: E402
from build_table import kind_tag                             # noqa: E402
from constants import MIN_CELL                               # noqa: E402
from s01_phases import (                                    # noqa: E402
    AFTER_FIRST_TOUCH, SEGMENTS, SetPieceGrammar, _OWN_GOAL_FOR,
    _body_part, _col, clearing_contacts, first_contact, genuine_touch, own_goal_source,
)
from validate import cluster_bootstrap_stats, poisson_ratio_ci   # noqa: E402

# The LEGACY clearance predicate, the one deliberate local copy: every taxonomy the corrected side
# of a comparison needs is imported above, and this is kept only so the driver can report the
# legacy-vs-corrected delta (evidence.md §3). It takes every "Goal Keeper" row, including Shot
# Faced (bookkeeping, no contact) and Collected (the keeper ends the play, the ball never drops).
_DEF_CLEAR = frozenset({"Clearance", "Block", "Goal Keeper"})

# Body part lives on a per-type column. There is NO duel.body_part, so a headed
# aerial duel is invisible here: the blind flick-on split only sees touches that
# carry a body part (no Duel), and that bound is reported, not papered over.

# Goals are rare, so an unpenalised logit quasi-separates and its coefficient runs
# off to +inf with a CI that reads as "PASS" while meaning nothing.
# A mild ridge keeps it finite; the stratified rate difference is the primary anyway.
_C_RIDGE = 1.0
_XG_STRATA = 3               # xG terciles: like-for-like shot quality without over-thinning the cells
_SHOT_WINDOW_S = SetPieceGrammar.SECOND_BALL_WINDOW_S   # bound, not re-typed: a local 8.0 drifts silently


def _chain_class(between: pd.DataFrame, attacking: str, defending: str | None,
                 shooter: Any) -> tuple[str, str | None]:
    """(class, body part of the last attacking touch) from the event chain alone.

    The outcome-blind arm of test 3. It can still be wrong, but it cannot be wrong
    in a way the provider's SHOT annotation caused, which is the whole point:
    key_pass_id is attached to the shot, so any rule used to decide it has already
    seen how the shot went. The chain never reads key_pass_id.

    A touch is s01's `genuine_touch`, so a row filed on a player the ball never reached
    does not make a shot indirect. The shooter's own Ball Receipt* is not an intervening
    touch either: it is the same action as the shot. Without that exclusion nothing is
    ever `direct`, because a receipt is logged before almost every shot."""
    touches = between[genuine_touch(between)]
    own_receipt = ((_col(touches, "type.name") == "Ball Receipt*")
                   & (_col(touches, "player.id") == shooter))
    touches = touches[~own_receipt]
    if touches.empty:
        return "direct", None
    att_touch = touches[_col(touches, "team.name") == attacking]
    bp = _body_part(att_touch.iloc[-1]) if len(att_touch) else None
    # What counts as a clearance is s01's call, not this file's. The CHAIN is re-derived
    # here on purpose (that is the falsification instrument), but the predicate it is
    # re-derived WITH must be the shipped one, or the gate scores a model nobody runs.
    if len(clearing_contacts(touches, defending)):
        return "after_clearance", bp
    return "prolongation", bp


def match_rows(mid: int, league: str,
               kinds: tuple = ("corner",)) -> tuple[list[dict], list[dict]]:
    """(shot rows, delivery rows) for one match off a single parse. The fan-out worker."""
    g = SetPieceGrammar(L.load_events(mid), kinds=kinds)
    return _shot_rows(g, mid, league), _corner_rows(g, mid, league)


def _shot_rows(g: SetPieceGrammar, mid: int, league: str) -> list[dict]:
    """One row per attacking shot inside a delivery's sequence, with every field the
    four tests need. Shot-grain on purpose: the frontier is a property of a shot,
    and aggregating to the corner before testing would hide the disagreements."""
    corners = g.deliveries()
    if corners.empty:
        return []
    by_id = g.ev.set_index("id", drop=False)
    out: list[dict] = []
    for _, c in corners.iterrows():
        att, t0, cid = c["team.name"], c["_t"], c["id"]
        post = g._sequence(c)
        post = post[post["id"] != cid]
        deff = g._defending_team(post, att)
        # Bounded to the episode by s01's own `_clearances`, which also decides that a delivery
        # with no contact opened no episode. Unbounded, the clearance can sit anywhere in the
        # sequence (the longer window plus HORIZON_PAD_S), and a clearance outside the episode is
        # not this episode's clearance.
        fc = first_contact(post, t0 + g.SECOND_BALL_WINDOW_S)
        t_fc = float(fc["_t"]) if fc is not None else None
        live = g._clearances(post, deff, t_fc)
        legacy = post[_col(post, "type.name").isin(_DEF_CLEAR) & (_col(post, "team.name") == deff)]
        t_live = float(live["_t"].min()) if len(live) else None
        t_legacy = float(legacy["_t"].min()) if len(legacy) else None
        # A shot belongs to the last delivery struck before it, the cut s01 ships.
        mine = (post["_t"] < g._next_delivery_t(c)) & (_col(post, "team.name") == att)
        shots = post[mine & (_col(post, "type.name") == "Shot")]
        # A forced own goal is a terminal outcome of the episode and is filed as `Own Goal
        # For`, which no predicate written against "Shot" ever matches. It carries no xG, no
        # key pass and no body part, so it cannot enter the four tests; it is emitted flagged,
        # with the action s01 credits it to, so the Q1 headline counts the goal in the segment
        # the pipeline ships while every test filters it back out.
        owns = post[mine & (_col(post, "type.name") == _OWN_GOAL_FOR)]
        for _, s in pd.concat([shots, owns]).sort_values("_t").iterrows():
            ts = float(s["_t"])
            is_own = s.get("type.name") == _OWN_GOAL_FOR
            between = post[(post["_t"] > t0) & (post["_t"] < ts) & (post["id"] != s["id"])]
            kp = s.get("shot.key_pass_id")
            kp = kp if isinstance(kp, str) else None
            kp_row = by_id.loc[kp] if kp is not None and kp in by_id.index else None
            chain, chain_bp = _chain_class(between, att, deff, s.get("player.id"))
            out.append(dict(
                match_id=mid, league=league, corner_id=cid, shot_id=s["id"],
                period=int(s["period"]), minute=int(s["minute"]),
                t_since_delivery=ts - t0,
                t_since_first_contact=(ts - t_fc) if t_fc is not None else np.nan,
                n_touches_before=int(genuine_touch(between).sum()),
                has_key_pass=kp is not None,
                kp_is_delivery=(kp == cid) if kp is not None else None,
                kp_body_part=_body_part(kp_row) if kp_row is not None else None,
                chain_class=chain, chain_body_part=chain_bp,
                clear_before_live=t_live is not None and t_live < ts,
                clear_before_legacy=t_legacy is not None and t_legacy < ts,
                xg=pd.to_numeric(s.get("shot.statsbomb_xg"), errors="coerce"),
                is_goal=bool(is_own or s.get("shot.outcome.name") == "Goal"),
                is_own_goal=bool(is_own),
                og_source=own_goal_source(between, att) if is_own else None,
                shot_outcome="Own Goal" if is_own else s.get("shot.outcome.name"),
                shot_body_part=s.get("shot.body_part.name"),
                set_piece_phase=pd.to_numeric(s.get("set_piece_phase"), errors="coerce"),
            ))
    return out


def _corner_rows(g: SetPieceGrammar, mid: int, league: str) -> list[dict]:
    """One row per delivery: the legacy-vs-current contact rules evidence.md §3 prices, the
    shotless count printed beside test 2, and the denominators the Q1 headline divides by."""
    corners = g.deliveries()
    if corners.empty:
        return []
    legacy_fc_types = {"Shot", "Clearance", "Block", "Goal Keeper", "Duel", "Miscontrol"}
    out: list[dict] = []
    for _, c in corners.iterrows():
        att = c["team.name"]
        post = g._sequence(c)
        post = post[post["id"] != c["id"]]
        deff = g._defending_team(post, att)
        # The CORRECTED side of every comparison below is s01's, called and not re-typed, so
        # the gate scores the `cleared` the pipeline actually ships.
        fc = first_contact(post, c["_t"] + g.SECOND_BALL_WINDOW_S)
        live = g._clearances(post, deff, None if fc is None else float(fc["_t"]))
        legacy_fc = post[_col(post, "type.name").isin(legacy_fc_types)]
        legacy = post[_col(post, "type.name").isin(_DEF_CLEAR) & (_col(post, "team.name") == deff)]
        mine = (post["_t"] < g._next_delivery_t(c)) & (_col(post, "team.name") == att)
        shots = post[mine & (_col(post, "type.name") == "Shot")]
        owns = post[mine & (_col(post, "type.name") == _OWN_GOAL_FOR)]
        xg = pd.to_numeric(_col(shots, "shot.statsbomb_xg"), errors="coerce").fillna(0.0)
        out.append(dict(
            match_id=mid, league=league, corner_id=c["id"], attacking_team=att,
            n_att_shots=int(len(shots)), sum_xg=float(xg.sum()), max_xg=float(xg.max()) if len(xg) else 0.0,
            any_goal=bool((_col(shots, "shot.outcome.name") == "Goal").any() or len(owns)),
            won_fc_onball=(fc["team.name"] == att) if fc is not None else None,
            won_fc_legacy=(legacy_fc.iloc[0]["team.name"] == att) if len(legacy_fc) else None,
            t_fc_onball=(float(fc["_t"]) - c["_t"]) if fc is not None else np.nan,
            t_fc_legacy=(float(legacy_fc.iloc[0]["_t"]) - c["_t"]) if len(legacy_fc) else np.nan,
            cleared_live=bool(len(live)), cleared_legacy=bool(len(legacy)),
        ))
    return out


def kp_class(row: pd.Series) -> str:
    """The key_pass_id taxonomy, with the clearance dominating the key pass."""
    if row["clear_before_live"]:
        return "after_clearance"
    if not row["has_key_pass"]:
        return "no_key_pass"
    return "direct" if row["kp_is_delivery"] else "prolongation"


def segment(row: pd.Series) -> str:
    """The five-segment taxonomy of definitions.md §4, built on the OUTCOME-BLIND chain.

    Priority-ordered and total, so the segments are mutually exclusive and exhaustive over
    the attacking shots of a corner (I1: the five sum to every attacking shot; I2: the first
    four sum to the in-window ones).

    The window is anchored on the FIRST CONTACT, matching s01_phases._segment_of; anchored on
    the delivery, the gate would score a taxonomy the pipeline does not ship."""
    if not np.isfinite(row["t_since_first_contact"]) or row["t_since_first_contact"] > _SHOT_WINDOW_S:
        return "late"
    if row["chain_class"] == "after_clearance":
        return "after_clearance"
    if row["chain_class"] == "direct":
        return "direct"
    return "flick_on" if row["chain_body_part"] == "Head" else "prolongation"


def place_own_goals(shots: pd.DataFrame) -> pd.Series:
    """`segment` with each own goal in the segment of the action that produced it.

    The rule is s01's `own_goal_source`, carried on the row as `og_source`: an own goal the
    delivery produced is `direct`, one a shot produced takes that shot's segment, and a `late`
    one stays late, so the gate keeps scoring the partition the pipeline ships."""
    seg = shots["segment"].copy()
    by_shot = shots.set_index(["corner_id", "shot_id"])["segment"]
    for i, r in shots[shots["is_own_goal"] & (seg != "late")].iterrows():
        if r["og_source"] == "delivery":
            seg.at[i] = "direct"
        elif isinstance(r["og_source"], str):
            seg.at[i] = by_shot.get((r["corner_id"], r["og_source"]), seg.at[i])
    return seg


# ── Test 1: annotation density ──────────────────────────────────────────
def _density_design(d: pd.DataFrame) -> tuple[np.ndarray, list[str]]:
    """[is_goal | xg | league one-hots]. Columns fixed over the FULL frame so a
    bootstrap replicate that misses a league keeps the same design width."""
    parts = [d[["is_goal"]].astype(float), d[["xg"]].astype(float),
             pd.get_dummies(d["league"], prefix="lg", dtype=float)]
    X = pd.concat(parts, axis=1)
    return X.to_numpy(float), list(X.columns)


def _fit_is_goal_coef(X: np.ndarray, y: np.ndarray) -> float:
    """Coefficient on is_goal (column 0) of a ridge logit."""
    if len(np.unique(y)) < 2:
        return np.nan
    return float(LogisticRegression(C=_C_RIDGE, max_iter=5000).fit(X, y).coef_[0][0])


def _adj_rate_diff(sub: pd.DataFrame) -> float:
    """has_key_pass rate among goals minus among non-goals, pooled across xG strata.

    The primary statistic, because it survives rare goals where the logit does not:
    a difference of two proportions, bounded in [-1, 1], checkable by hand. Strata
    with only one outcome class contribute nothing."""
    num = den = 0.0
    for _, st in sub.groupby("xg_stratum", observed=True):
        g, ng = st[st["is_goal"]], st[~st["is_goal"]]
        if g.empty or ng.empty:
            continue
        w = float(len(st))
        num += w * (g["has_key_pass"].mean() - ng["has_key_pass"].mean())
        den += w
    return num / den if den else np.nan


def annotation_density(df: pd.DataFrame, n_boot: int = 2000, seed: int = 0) -> dict:
    """Test 1. Does a shot carry a key pass more often BECAUSE it was a goal?

    At fixed xG the annotator has no legitimate reason to log the preceding touch
    as a completed pass more often for goals than for misses. A non-zero effect
    means the direct/prolongation partition is manufactured by the labelling rule,
    and the delta it produces is not football."""
    d = df.dropna(subset=["xg"]).copy()
    d["xg_stratum"] = pd.qcut(d["xg"], _XG_STRATA, duplicates="drop")
    X, names = _density_design(d)
    y = d["has_key_pass"].to_numpy(int)
    d = d.reset_index(drop=True)

    def _coef(sub: pd.DataFrame) -> float:
        return _fit_is_goal_coef(X[sub.index.to_numpy()], y[sub.index.to_numpy()])

    ci = cluster_bootstrap_stats(d, {"rate_diff": _adj_rate_diff, "logit_coef": _coef},
                                 n_boot=n_boot, seed=seed)
    covers = ci.apply(lambda r: bool(np.isfinite(r["lo"]) and r["lo"] <= 0.0 <= r["hi"]), axis=1)
    return dict(ci=ci, n=len(d), n_goals=int(d["is_goal"].sum()),
                n_groups=int(d["match_id"].nunique()), names=names, passes=bool(covers.all()))


# ── Tests 2-4 ───────────────────────────────────────────────────────────
def vendor_crosstab(shots: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Test 2. Our taxonomy against the provider's set_piece_phase, over shots only.

    A shotless corner is left out on purpose: there both sides say 'nothing happened', so
    agreeing is free and counting it would inflate the rate. The number is reported, never
    used to retune our definition."""
    s = shots.dropna(subset=["set_piece_phase"]).copy()
    tab = pd.crosstab(s["segment"], s["set_piece_phase"])
    ours = s["segment"].isin(AFTER_FIRST_TOUCH)
    theirs = s["set_piece_phase"] == 2.0
    rows = [dict(league="ALL", n=len(s), agree=float((ours == theirs).mean()),
                 ours_only=int((ours & ~theirs).sum()), theirs_only=int((~ours & theirs).sum()))]
    for lg, sub in s.groupby("league"):
        o, t = sub["segment"].isin(AFTER_FIRST_TOUCH), sub["set_piece_phase"] == 2.0
        rows.append(dict(league=lg, n=len(sub), agree=float((o == t).mean()),
                         ours_only=int((o & ~t).sum()), theirs_only=int((~o & t).sum())))
    return tab, pd.DataFrame(rows).set_index("league")


def conversion_table(shots: pd.DataFrame, col: str) -> pd.DataFrame:
    """Shots, goals, conversion, mean xG and the observed/expected goal ratio with an
    exact Poisson interval, the diagnostic for a class that beats its own xG."""
    g = shots.groupby(col)
    out = pd.DataFrame({"shots": g.size(), "goals": g["is_goal"].sum(),
                        "mean_xg": g["xg"].mean(), "sum_xg": g["xg"].sum()})
    out["conv"] = out["goals"] / out["shots"]
    ci = [poisson_ratio_ci(int(k), float(e))
          for k, e in zip(out["goals"], out["sum_xg"], strict=True)]
    out["obs_exp"] = out["goals"] / out["sum_xg"].replace(0.0, np.nan)
    out["oe_lo"], out["oe_hi"] = [c[0] for c in ci], [c[1] for c in ci]
    out["thin"] = np.where(out["shots"] < MIN_CELL, f"<{MIN_CELL}", "")
    return out.sort_values("shots", ascending=False)


def _conv_of(col: str, klass: str) -> Callable:
    """Conversion rate of one class, as a statistic the bootstrap can resample."""
    def stat(sub: pd.DataFrame) -> float:
        m = sub[col] == klass
        return sub.loc[m, "is_goal"].mean() if m.any() else np.nan
    return stat


def _conv_delta(col: str, a: str, b: str) -> Callable:
    """Paired conversion delta INSIDE one replicate. Two marginal CIs that overlap
    do not imply the difference is insignificant, so the delta gets its own."""
    fa, fb = _conv_of(col, a), _conv_of(col, b)
    return lambda sub: fa(sub) - fb(sub)


def separability(shots: pd.DataFrame) -> pd.DataFrame:
    """Test 4. Rank AUC of time-since-delivery separating each segment from `direct`,
    plus the touch-count distribution. AUC 0.5 means the label tracks no phase."""
    base = shots[shots["segment"] == "direct"]["t_since_delivery"].to_numpy()
    recs = []
    for seg in SEGMENTS:
        v = shots[shots["segment"] == seg]
        t = v["t_since_delivery"].to_numpy()
        if seg == "direct" or len(t) == 0 or len(base) == 0:
            auc = np.nan
        else:
            r = rankdata(np.concatenate([t, base]))
            auc = (r[:len(t)].sum() - len(t) * (len(t) + 1) / 2) / (len(t) * len(base))
        recs.append(dict(segment=seg, n=len(v), t_mean=t.mean() if len(t) else np.nan,
                         t_p25=np.percentile(t, 25) if len(t) else np.nan,
                         t_med=np.median(t) if len(t) else np.nan,
                         t_p75=np.percentile(t, 75) if len(t) else np.nan,
                         touches_mean=v["n_touches_before"].mean() if len(v) else np.nan,
                         auc_vs_direct=auc))
    return pd.DataFrame(recs).set_index("segment")


def _pct(num: float, den: float) -> str:
    return f"{100 * num / den:5.1f}%" if den else "   n/a"


def stats_record(pairs: int, corners: int, shots: int, dens: dict, agree: pd.DataFrame,
                 phase_tab: pd.DataFrame, kp_tab: pd.DataFrame, conv: pd.DataFrame,
                 boot: pd.DataFrame, sep: pd.DataFrame, head: pd.DataFrame) -> pd.DataFrame:
    """Every figure the four tests and the Q1 headline print, in one long table.

    Persisted rather than printed, because evidence.md §1 and §2 quote these figures and a
    printed figure is one no test can bind: prose quoting a print drifts from the shipped
    grammar with nothing able to go red. Nothing is recomputed here: every frame is main()'s."""
    rows: list[dict] = []
    for key, v in (("matches", pairs), ("corners", corners), ("shots", shots)):
        rows.append(dict(block="population", key=key, n=int(v)))
    for k, r in dens["ci"].iterrows():
        rows.append(dict(block="density", key=k, point=r["point"], lo=r["lo"], hi=r["hi"]))
    # The test's OWN population, which is not the corpus: it drops the own goals, since an own
    # goal carries no key pass, no xG and no provider phase and cannot enter the four tests.
    for key, v in (("n_shots", dens["n"]), ("n_goals", dens["n_goals"]),
                   ("n_matches", dens["n_groups"])):
        rows.append(dict(block="density", key=key, n=int(v)))
    for lg, r in agree.iterrows():
        rows.append(dict(block="agreement", key=str(lg), point=r["agree"], n=int(r["n"]),
                         ours_only=int(r["ours_only"]), theirs_only=int(r["theirs_only"])))
    # Our segment against the provider's phase, cell by cell: the `direct` row is the one that
    # bounds the originality claim, since a class the vendor already draws is not ours to claim.
    for seg, row in phase_tab.iterrows():
        for ph, n in row.items():
            rows.append(dict(block="vendor_phase", key=f"{seg}|{float(ph):.0f}", n=int(n)))
    for kp, row in kp_tab.iterrows():
        for seg, n in row.items():
            rows.append(dict(block="kp_vs_segment", key=f"{kp}|{seg}", n=int(n)))
    # conv is indexed by segment, so the aggregated `after_first_touch` class carries no
    # row there and is published with its interval and no cell count, which is what it is.
    for k, r in boot.iterrows():
        kind, key = k.split("[")[0], k.split("[")[1].rstrip("]")
        row = dict(block="conversion" if kind == "conv" else kind, key=key,
                   point=r["point"], lo=r["lo"], hi=r["hi"])
        if key in conv.index:
            row |= dict(n=int(conv.loc[key, "shots"]), goals=int(conv.loc[key, "goals"]),
                        sum_xg=float(conv.loc[key, "sum_xg"]))
        rows.append(row)
    # `separability` reindexes over all five segments and `late` is absent from the in-window
    # frame by construction, so its row is an empty cell rather than a measured zero.
    for seg, r in sep[sep["n"] > 0].iterrows():
        rows.append(dict(block="separability", key=seg, point=r["auc_vs_direct"], n=int(r["n"])))
    for k, r in head.iterrows():
        rows.append(dict(block="headline", key=k, point=r["point"], lo=r["lo"], hi=r["hi"]))
    return pd.DataFrame(rows)


def main() -> None:
    """CLI: run the four falsification tests over a slice of the grant.

    A narrowed run (--source, --limit) writes a tagged name, so it never lands on the grant's gate
    that report_figures reads; an explicit --out is honoured as given."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", type=str, default=None, help="loader_sb source (default: all Big-5)")
    ap.add_argument("--limit", type=int, default=None, help="first N matches (smoke test)")
    ap.add_argument("--jobs", type=int, default=20,
                    help="worker processes; each holds one match's events")
    ap.add_argument("--boot", type=int, default=2000, help="match-clustered bootstrap replicates")
    ap.add_argument("--out", type=str, default=None)
    ap.add_argument("--kinds", type=str, default="corner", help="the s01 population")
    args = ap.parse_args()
    kinds = tuple(k.strip() for k in args.kinds.split(","))

    catalog = L.list_matches(args.source)
    if args.limit:
        catalog = catalog.iloc[:args.limit]
    pairs = list(zip(catalog["match_id"].astype(int), catalog["league"], strict=True))

    chunks = Parallel(n_jobs=args.jobs, verbose=5)(delayed(match_rows)(m, lg, kinds)
                                                    for m, lg in pairs)
    shots = pd.DataFrame([r for sh, _ in chunks for r in sh])
    corners = pd.DataFrame([r for _, co in chunks for r in co])

    if shots.empty:
        raise SystemExit("no attacking shots found — nothing to falsify")

    shots["kp_class"] = shots.apply(kp_class, axis=1)
    shots["segment"] = shots.apply(segment, axis=1)
    shots["segment"] = place_own_goals(shots)
    shots["after_first_touch"] = shots["segment"].isin(AFTER_FIRST_TOUCH)
    # The four tests interrogate the PROVIDER'S annotation, so they run on the rows that
    # carry one. An own goal has no xG, no key pass and no set_piece_phase, and would enter
    # every conversion table as a goal against a zero denominator. It stays in `shots`,
    # which is what the Q1 headline counts, because the goal is real.
    real = shots[~shots["is_own_goal"]]
    in_win = real[real["segment"] != "late"].reset_index(drop=True)

    print(f"\nFALSIFY FRONTIER — {len(pairs)} matches, {len(corners)} corners, "
          f"{len(real)} attacking shots (+{int(shots['is_own_goal'].sum())} forced own goals), "
          f"{len(real) / max(len(corners), 1):.3f} shots/corner\n")

    print("=" * 78)
    print("TEST 1 — annotation density (NULL = no is_goal effect on has_key_pass)")
    print("=" * 78)
    dens = annotation_density(real, n_boot=args.boot)
    print(f"  n={dens['n']} shots, {dens['n_goals']} goals, {dens['n_groups']} matches, "
          f"{args.boot} match-clustered replicates")
    print(dens["ci"].to_string(float_format=lambda v: f"{v:+.4f}"))
    print(f"  -> {'PASS (both CIs cover 0)' if dens['passes'] else 'FAIL (a CI excludes 0)'}")
    print("  note: rate_diff is the primary; the logit is a continuous-xG + league check.\n")

    print("=" * 78)
    print("TEST 2 — vendor cross-tab vs set_piece_phase")
    print("=" * 78)
    tab, agree = vendor_crosstab(real)
    print(tab.to_string())
    print(f"\n  shotless corners, excluded from the rate: {int((corners['n_att_shots'] == 0).sum())}")
    print(agree.to_string(float_format=lambda v: f"{v:.3f}"))
    print("  our definition is NOT a re-derivation of theirs where ours_only is large.\n")

    print("=" * 78)
    print("TEST 3 — outcome-blind re-derivation")
    print("=" * 78)
    print("  key_pass_id taxonomy (in-window shots):")
    print(conversion_table(in_win, "kp_class").to_string(float_format=lambda v: f"{v:.4f}"))
    conv = conversion_table(in_win, "segment")
    print("\n  event-chain taxonomy, blind to key_pass_id:")
    print(conv.to_string(float_format=lambda v: f"{v:.4f}"))
    kp_tab = pd.crosstab(in_win["kp_class"], in_win["segment"])
    print("\n  do the two partitions agree?")
    print(kp_tab.to_string())

    seg_present = [s for s in SEGMENTS if s != "late" and (in_win["segment"] == s).any()]
    stats = {f"conv[{s}]": _conv_of("segment", s) for s in seg_present}
    for s in seg_present:
        if s != "direct":
            stats[f"delta[{s}-direct]"] = _conv_delta("segment", s, "direct")
    stats["conv[after_first_touch]"] = lambda sub: sub.loc[sub["after_first_touch"], "is_goal"].mean()
    stats["delta[aft-direct]"] = lambda sub: (sub.loc[sub["after_first_touch"], "is_goal"].mean()
                                              - _conv_of("segment", "direct")(sub))
    boot = cluster_bootstrap_stats(in_win, stats, n_boot=args.boot)
    print("\n  conversion with match-clustered CIs (deltas are PAIRED inside each replicate):")
    print(boot.to_string(float_format=lambda v: f"{v:+.4f}"))

    print("\n" + "=" * 78)
    print("TEST 4 — temporal separability")
    print("=" * 78)
    sep = separability(in_win)
    print(sep.to_string(float_format=lambda v: f"{v:.3f}"))
    print("  auc_vs_direct is P(this segment's shot came later than a direct shot).\n")

    print("=" * 78)
    print("Q1 HEADLINE — how much lands after the first contact")
    print("=" * 78)
    ct = corners.copy()
    aft = shots[shots["after_first_touch"]].groupby("corner_id").agg(
        aft_goal=("is_goal", "max"), aft_xg=("xg", "sum"))
    ct = ct.join(aft, on="corner_id")
    ct["aft_goal"] = ct["aft_goal"].astype("boolean").fillna(False).astype(bool)
    ct["aft_xg"] = pd.to_numeric(ct["aft_xg"], errors="coerce").fillna(0.0)
    hstats = {
        "share_goals_after_first_touch": lambda s: s["aft_goal"].sum() / max(s["any_goal"].sum(), 1),
        "share_xg_after_first_touch": lambda s: s["aft_xg"].sum() / max(s["sum_xg"].sum(), 1e-9),
        "corners_reaching_a_shot": lambda s: (s["n_att_shots"] > 0).mean(),
    }
    head = cluster_bootstrap_stats(ct, hstats, n_boot=args.boot)
    print(head.to_string(float_format=lambda v: f"{v:.4f}"))
    print(f"  goals from corners: {int(ct['any_goal'].sum())}   "
          f"of which after the first contact: {int(ct['aft_goal'].sum())}\n")

    print("=" * 78)
    print("PER-LEAGUE segment mix (prefigures the league-out split)")
    print("=" * 78)
    mix = pd.crosstab(real["league"], real["segment"], normalize="index")
    cnt = pd.crosstab(real["league"], real["segment"])
    print(mix.to_string(float_format=lambda v: f"{v:.3f}"))
    print(f"\n  counts (cells under {MIN_CELL} are not publishable):")
    print(cnt.to_string())

    print("\n" + "=" * 78)
    print("LEGACY vs CURRENT contact rules, measured on the same pass (evidence.md §3)")
    print("=" * 78)
    for lg, sub in [("ALL", corners)] + list(corners.groupby("league")):
        n = len(sub)
        # won_fc can be None (no on-ball touch at all inside the window), so the
        # rate is over corners WITH a contact, mean(skipna=True), matching
        # build_table.py's won_first_contact denominator, not len(sub).
        print(f"  {lg:16s} n={n:6d}  won_fc: legacy {100 * sub['won_fc_legacy'].mean(skipna=True):5.1f}%"
              f" -> onball {100 * sub['won_fc_onball'].mean(skipna=True):5.1f}%"
              f"   t_fc: {sub['t_fc_legacy'].mean():.2f}s -> {sub['t_fc_onball'].mean():.2f}s"
              f"   cleared: {_pct(sub['cleared_legacy'].sum(), n)} -> {_pct(sub['cleared_live'].sum(), n)}")

    print("\n  shot outcome by segment:")
    print(pd.crosstab(in_win["segment"], in_win["shot_outcome"]).to_string())

    tag = (kind_tag(kinds) + (f"_{args.source.lower()}" if args.source else "")
           + (f"_limit{args.limit}" if args.limit else ""))
    out = Path(args.out or _REPO / "data" / "derived" / f"falsification{tag}.parquet")
    out.parent.mkdir(parents=True, exist_ok=True)
    shots.to_parquet(out)
    # Both companions are suffixed on the stem of `out`, so a --limit run tags all three at once
    # and an --out without ".parquet" still writes three files rather than one.
    corners.to_parquet(out.with_name(f"{out.stem}_corners.parquet"))
    rec = stats_record(len(pairs), len(corners), len(real), dens, agree, tab, kp_tab, conv,
                       boot, sep, head)
    rec.to_parquet(out.with_name(f"{out.stem}_stats.parquet"))
    print(f"\nsaved {out}  ({len(shots)} shots), the corner-grain companion and "
          f"{len(rec)} stat rows  [gitignored]")


if __name__ == "__main__":
    main()
