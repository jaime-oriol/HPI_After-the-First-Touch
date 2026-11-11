"""audit_inferred — test every arithmetic and logical claim the data-provider doc makes
about the undocumented fields, against the data rather than against one example row.

    python pipeline/audit_inferred.py --jobs 20 --diagnose   # the full grant
    python pipeline/audit_inferred.py --sample 60 --jobs 8    # smoke test

WHY THIS EXISTS. Most event columns have no public specification (data_provider.tex counts
them). The doc's definitions for them are marked `inferred`: its own reading of a field name
plus one example value. Those definitions are claims, and several fail against the columns:
AngleDeviation's sign, the DistToKeeper+DistSGK identity, InCone.GK as a boolean.
A guess that survives because nobody checked it is worse than a blank, because it reads
like documentation. Every check here is a claim that can be falsified from the files.

Each check reports the share of rows where it holds. Read the output as: 100% means the
relation is the definition; a high-but-not-100% share means the relation is approximate
and the definition needs a qualifier; a low share means the definition is wrong.

FALSIFYING IS ONLY HALF OF IT. `_shot_geometry` inverts the shot-geometry block: rather
than guess a formula from a field name, it asks which players in `shot.freeze_frame`
reproduce the published value. Where that closes, the losing readings are scored under
their own names too, so the doc can cite the number that killed one instead of asserting
the survivor. Several of those losers are readings data_provider.tex records as refuted.

TWO OF THE FIELDS WERE NOT CLOSED BY INVERSION AT ALL, and the difference decides how they
may be written up. `density.incone` and `DefArea`'s magnitude close here against the source
the provider publishes, so their geometry is DECLARED rather than fitted, and only that lets
them be called verified. The floor is scored beside the winner for the same reason: a
restriction is worth only what it takes the unrestricted formula above.

--diagnose writes the population each recovery was computed over and halts the run when a
field scores under 95% of its populated rows or its prediction takes a single value. A
percentage nobody sized is how a recovery gets believed on a tenth of its rows.

--limit is a HEAD SLICE of a date-ordered catalogue and is not a sample: shot.freeze_frame
changed composition between 2026-01-17 and 2026-01-23. Use --sample.
"""

from __future__ import annotations

import argparse
import sys
import time
from collections.abc import Callable
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO / "src"))

import constants as C                                   # noqa: E402
import loader_sb as L                                   # noqa: E402
import shot_geometry as G                               # noqa: E402

_TOL = 1e-6

_COLS = [
    "obv_for_before", "obv_for_after", "obv_for_net",
    "obv_against_before", "obv_against_after", "obv_against_net", "obv_total_net",
    "Score", "OpposingScore", "TotalScore", "GameState", "OpposingGameState",
    "ElapsedTime", "StartOfPossession", "TimeInPoss", "TimeToPossEnd", "Goal",
    "team.name", "possession_team.name", "OpposingTeam", "home_team.home_team_name",
    "away_team.away_team_name", "type.name", "shot.outcome.name",
    "density", "density.incone", "distance.ToD1", "distance.ToD2",
    "distance.ToD1.360", "distance.ToD2.360", "DefendersInCone", "InCone.GK",
    "AttackersBehindBall", "DefendersBehindBall", "AngleToGoal", "AngleToKeeper",
    "AngleDeviation", "DistToGoal", "DistToKeeper", "DistSGK", "DefArea",
    "shot_impact_height", "avevelocity", "location.x.GK", "location.y.GK",
    "shot.statsbomb_xg", "shot.shot_execution_xg", "shot.shot_execution_xg_uplift",
    "shot.gk_save_difficulty_xg",
    "num_defenders_on_goal_side_of_actor", "distance_to_nearest_defender",
    "ball_receipt_exceeds_distance", "defensive_responsibility.probability_sum",
    "defensive_responsibility.players", "set_piece_phase", "play_pattern.name",
    "duel.hops_rating", "duel.hops_opponent_rating", "duel.duel_win_probability",
    "pass.pass_success_probability", "freeze_frame", "visible_player_counts",
    "location", "shot.freeze_frame", "player.id", "player.id.GK",
]


def _close(a: pd.Series, b: pd.Series, tol: float) -> "tuple[int, int]":
    """(rows where both sides are present, rows where they agree within tol)."""
    m = a.notna() & b.notna()
    if not m.any():
        return 0, 0
    return int(m.sum()), int(np.isclose(a[m].astype(float), b[m].astype(float), atol=tol).sum())


# Each provider field recovered by inversion, and the check whose prediction is scored against it.
# The stop bands in main() read this: a field here that scores under 95% of its populated rows, or
# whose prediction takes a single value, makes the run write its tables and then exit 2.
_INVERTED = {
    "density": "density == sum 1/d over goal-side outfield defenders",
    "density.incone": "density.incone == density restricted to the provider's declared cone",
    "DefArea": "DefArea == the bounding box of the opposing defensive line (position.id 2-8)",
    "AttackersBehindBall": "AttackersBehindBall == teammates in the frame goal-side of the shot",
    "DefendersBehindBall": "DefendersBehindBall == outfield opponents goal-side of the shot",
    "distance.ToD1": "distance.ToD1 == the nearest of that population, signed",
    "distance.ToD2": "distance.ToD2 == the second-nearest of that population, signed",
    "DistToGoal": "DistToGoal == |shot - goal centre|",
    "DistSGK": "DistSGK == |shot - keeper|",
    "DistToKeeper": "DistToKeeper == |keeper - goal centre|",
    "AngleToGoal": "AngleToGoal == the bearing atan2(120 - x, 40 - y) in degrees",
    "AngleToKeeper": "AngleToKeeper == the same bearing taken AT THE KEEPER",
    "location.x.GK": "location.x.GK, location.y.GK == the Goalkeeper entry of shot.freeze_frame",
}


def _subtended_angle(shot: np.ndarray) -> float:
    """The angle the goalmouth subtends at the shot, degrees. The doc's reading of AngleToGoal,
    kept only so the number that falsifies it can be published."""
    a, b = (np.asarray(p, dtype=float) - shot for p in C.GOAL_POSTS_SB)
    cos = float(a @ b / (np.hypot(*a) * np.hypot(*b)))
    return float(np.degrees(np.arccos(np.clip(cos, -1.0, 1.0))))


def _unslacked_cone(xy: np.ndarray, shot: np.ndarray) -> np.ndarray:
    """Inside the cone drawn to the REAL posts with its apex ON the shot: the reading the
    provider's declared geometry beats, kept only so the number that falsifies it can be
    published."""
    v = np.array([C.GOAL_POSTS_SB[0], C.GOAL_POSTS_SB[1], tuple(shot)], dtype=float)
    s = [(q[0] - p[0]) * (xy[:, 1] - p[1]) - (q[1] - p[1]) * (xy[:, 0] - p[0])
         for p, q in zip(v, np.roll(v, -1, axis=0), strict=True)]
    return ~(((s[0] < 0) | (s[1] < 0) | (s[2] < 0)) & ((s[0] > 0) | (s[1] > 0) | (s[2] > 0)))


def _shot_geometry(ev: pd.DataFrame, add: Callable[..., None], diag: Callable[..., None]) -> None:
    """Recover the shot-geometry block from `shot.freeze_frame`, one pass over the shot rows.

    Every field here is `inferred` in the provider doc, meaning its definition is a reading of the
    field name. An identity either reproduces the published value at its own rounding or it does
    not, so each claim is scored as exact agreement within a written tolerance and never as a
    correlation: a correlation can be high on the wrong formula.

    The alternatives that lose are scored too, under their own names, so the doc can cite the
    number that killed a reading rather than assert the survivor. Several of them are readings
    data_provider.tex records as refuted."""
    need = ("density", "shot.freeze_frame", "location")
    if any(c not in ev.columns for c in need):
        return
    m = ev["density"].notna() \
        & ev["shot.freeze_frame"].map(lambda v: isinstance(v, list) and len(v) > 0) \
        & ev["location"].map(lambda v: isinstance(v, list) and len(v) >= 2)
    if not m.any():
        return

    hit: dict = {}
    seen: dict = {}

    def score(name: str, ok: bool, pred: float | None = None) -> None:
        c = hit.setdefault(name, [0, 0])
        c[0] += 1
        c[1] += bool(ok)
        if pred is not None:
            s = seen.setdefault(name, [np.inf, -np.inf, set()])
            s[0], s[1] = min(s[0], float(pred)), max(s[1], float(pred))
            if len(s[2]) < 4096:
                s[2].add(round(float(pred), 6))

    def near(a: float, b: float, tol: float = 5e-4) -> bool:
        return pd.notna(b) and abs(float(a) - float(b)) < tol

    for _, r in ev.loc[m].iterrows():
        shot = G.shot_xy(r["location"])
        fa = G.frame_arrays(r["shot.freeze_frame"])
        ds = G.defender_distances(shot, fa)
        gs_mask = fa.xy[:, 0] >= shot[0]

        pred = G.inverse_distance_sum(shot, fa)
        score("density == sum 1/d over goal-side outfield defenders",
              near(pred, r["density"]), pred)
        score("density == 0 iff no defender is goal-side",
              (float(r["density"]) == 0.0) == (len(ds.goal_side) == 0))

        # ── density.incone: the declared cone, against the floor and the unslacked cone ──
        if pd.notna(r.get("density.incone")):
            inc = G.inverse_distance_sum(shot, fa, cone_only=True)
            score("density.incone == density restricted to the provider's declared cone",
                  near(inc, r["density.incone"]), inc)
            score("density.incone == density with no cone restriction at all (the trivial floor)",
                  near(pred, r["density.incone"]))
            keep = ~fa.teammate & ~fa.keeper & gs_mask & _unslacked_cone(fa.xy, shot)
            old = float((1.0 / G.body_distances(shot, fa)[keep]).sum()) if keep.any() else 0.0
            score("density.incone == the cone drawn to the REAL posts, apex on the shot",
                  near(old, r["density.incone"]))

        # ── the two behind-ball counts, and the readings they beat ──
        att, dfn = G.behind_ball_counts(shot, fa)
        score("AttackersBehindBall == teammates in the frame goal-side of the shot",
              att == r.get("AttackersBehindBall"), att)
        score("DefendersBehindBall == outfield opponents goal-side of the shot",
              dfn == r.get("DefendersBehindBall"), dfn)
        strict = fa.xy[:, 0] > shot[0]
        score("AttackersBehindBall == teammates STRICTLY goal-side (x > x_shot)",
              int((strict & fa.teammate).sum()) == r.get("AttackersBehindBall"))
        score("DefendersBehindBall == outfield opponents STRICTLY goal-side (x > x_shot)",
              int((strict & ~fa.teammate & ~fa.keeper).sum()) == r.get("DefendersBehindBall"))
        score("DefendersBehindBall == opponents goal-side INCLUDING the keeper",
              int((gs_mask & ~fa.teammate).sum()) == r.get("DefendersBehindBall"))

        # The shooter enters their own freeze frame from late January 2026, which shifts this count
        # by one without the definition changing. Both regimes are scored so neither hides the other.
        pid = r.get("player.id")
        shooter = np.isclose(fa.player_id, float(pid)) if pd.notna(pid) else np.zeros(len(fa.xy), bool)
        score("shot.freeze_frame carries the shooter", bool(shooter.any()))
        score("AttackersBehindBall == teammates goal-side EXCLUDING the shooter",
              int((gs_mask & fa.teammate & ~shooter).sum()) == r.get("AttackersBehindBall"))
        tag = "with" if shooter.any() else "without"
        score(f"AttackersBehindBall == teammates goal-side [frames {tag} the shooter]",
              att == r.get("AttackersBehindBall"))

        # ── the signed nearest-defender family, sentinel included ──
        for k, ord_ in ((1, "nearest"), (2, "second-nearest")):
            p = G.signed_nth_defender(ds, k)
            score(f"distance.ToD{k} == the {ord_} of that population, signed",
                  near(p, r.get(f"distance.ToD{k}")), p)

        # The two divergences against the published source, scored apart so the conjecture that
        # BOTH explained the residue is settled per half rather than as one claim.
        raw = np.hypot(fa.xy[:, 0] - shot[0], fa.xy[:, 1] - shot[1])
        alive = ~fa.teammate & ~fa.keeper & (raw > 0)
        dropped = G.DefenderSet(np.sort(raw[alive & gs_mask]), np.sort(raw[alive & ~gs_mask]),
                                np.sort(raw[alive]))
        with_gk = G.DefenderSet(ds.goal_side, np.sort(G.body_distances(shot, fa)[
            ~fa.teammate & ~(~fa.teammate & ~fa.keeper & gs_mask)]), ds.every)
        for k in (1, 2):
            score(f"distance.ToD{k} == the same with the keeper KEPT in the behind-shot fallback",
                  near(G.signed_nth_defender(with_gk, k), r.get(f"distance.ToD{k}")))
            score(f"distance.ToD{k} == the same with a body ON the shot dropped, not substituted",
                  near(G.signed_nth_defender(dropped, k), r.get(f"distance.ToD{k}")))
        # Scored only where the two readings can disagree, so the gap is not diluted to noise.
        if len(ds.goal_side) < 2:
            alt = -ds.every[1] if len(ds.every) > 1 else C.EMPTY_DEFENDER_DISTANCE_SB
            score("distance.ToD2 == the second-nearest defender OVERALL when fewer than two are "
                  "goal-side", near(alt, r.get("distance.ToD2")))
        score("distance.ToD1 == 30 iff the frame carries no outfield opponent",
              near(r.get("distance.ToD1"), C.EMPTY_DEFENDER_DISTANCE_SB) == (len(ds.every) == 0))

        # ── distances and bearings; the doc's own wording is scored as the falsifier ──
        dgoal = float(np.hypot(*(np.array(C.GOAL_SB) - shot)))
        score("DistToGoal == |shot - goal centre|", near(dgoal, r.get("DistToGoal")), dgoal)
        bearing = G.bearing_to_goal(shot)
        score("AngleToGoal == the bearing atan2(120 - x, 40 - y) in degrees",
              near(bearing, r.get("AngleToGoal"), 0.01), bearing)
        score("AngleToGoal == the angle the goalmouth SUBTENDS at the shot (doc's wording)",
              near(_subtended_angle(shot), r.get("AngleToGoal"), 0.01))
        kxy = G.keeper_xy(fa)
        if kxy is not None:
            score("location.x.GK, location.y.GK == the Goalkeeper entry of shot.freeze_frame",
                  near(kxy[0], r.get("location.x.GK")) and near(kxy[1], r.get("location.y.GK")),
                  kxy[0])
            dsgk = float(np.hypot(*(kxy - shot)))
            dkg = float(np.hypot(*(np.array(C.GOAL_SB) - kxy)))
            score("DistSGK == |shot - keeper|", near(dsgk, r.get("DistSGK")), dsgk)
            score("DistToKeeper == |keeper - goal centre|",
                  near(dkg, r.get("DistToKeeper")), dkg)
            score("DistToKeeper == |shot - keeper| (the name's reading)",
                  near(np.hypot(*(kxy - shot)), r.get("DistToKeeper")))
            bk = G.bearing_to_goal(kxy)
            score("AngleToKeeper == the same bearing taken AT THE KEEPER",
                  near(bk, r.get("AngleToKeeper"), 0.01), bk)
            score("AngleToKeeper == the bearing from the shot TO the keeper (doc's wording)",
                  near(np.degrees(np.arctan2(kxy[0] - shot[0], kxy[1] - shot[1])),
                       r.get("AngleToKeeper"), 0.01))
            kid = fa.player_id[fa.keeper & ~fa.teammate]
            score("player.id.GK == that entry's player.id",
                  len(kid) > 0 and near(kid[0], r.get("player.id.GK"), 0.5))

        # ── DefArea: the sentinel came from inversion, the magnitude needed the published source ──
        if pd.notna(r.get("DefArea")) and pd.notna(r.get("distance.ToD1")):
            score("DefArea == 1000 iff distance.ToD1 == 30 (one empty-population sentinel)",
                  (float(r["DefArea"]) == C.EMPTY_DEFENDER_AREA_SB)
                  == near(r["distance.ToD1"], C.EMPTY_DEFENDER_DISTANCE_SB))
        if pd.notna(r.get("DefArea")):
            area = G.defensive_line_area(fa)
            score("DefArea == the bounding box of the opposing defensive line (position.id 2-8)",
                  near(area, r["DefArea"]), area)
            wide = fa.xy[~fa.teammate & ~fa.keeper]
            alt = float((wide[:, 0].max() - wide[:, 0].min())
                        * (wide[:, 1].max() - wide[:, 1].min())) if len(wide) \
                else C.EMPTY_DEFENDER_AREA_SB
            score("DefArea == the bounding box of EVERY outfield opponent", near(alt, r["DefArea"]))
            lo, hi = C.DEFENSIVE_LINE_POSITION_IDS
            absent = not (~fa.teammate & (fa.position_id >= lo)
                          & (fa.position_id <= hi)).any()
            score("DefArea == 1000 iff the defensive line is absent from the frame",
                  (float(r["DefArea"]) == C.EMPTY_DEFENDER_AREA_SB) == absent)

    for name, (n, ok) in hit.items():
        add(name, n, ok)

    blind = ev["shot.freeze_frame"].map(lambda v: not (isinstance(v, list) and len(v) > 0))
    for field, name in _INVERTED.items():
        if field not in ev.columns:
            continue
        lo, hi, vals = seen.get(name, (np.nan, np.nan, set()))
        diag(field, int(ev[field].notna().sum()), int(hit.get(name, [0, 0])[0]),
             int((ev[field].notna() & blind).sum()), lo, hi, vals)


def _checks(ev: pd.DataFrame) -> "tuple[list, list]":
    """Every falsifiable claim, as (check name, rows tested, rows holding).

    The second return is the diagnostic record of the inversion checks: rows scored, and the
    range and cardinality of the value each formula predicted. A percentage computed over a
    population nobody looked at is how a recovery gets believed on a tenth of its rows."""
    out: list = []
    diags: list = []

    def add(name: str, n: int, ok: int) -> None:
        out.append((name, int(n), int(ok)))

    def diag(field: str, pop: int, scored: int, blind: int, lo: float, hi: float,
             vals: set) -> None:
        diags.append((field, int(pop), int(scored), int(blind), float(lo), float(hi),
                      frozenset(vals)))

    def col(c: str) -> pd.Series:
        return ev[c] if c in ev.columns else pd.Series(dtype=float)

    # ── OBV: the doc reads the before/after/net decomposition off the field names ──
    for side in ("for", "against"):
        a, b, net = col(f"obv_{side}_after"), col(f"obv_{side}_before"), col(f"obv_{side}_net")
        if len(a) and len(b) and len(net):
            add(f"obv_{side}_net == after - before", *_close(a - b, net, 1e-4))
    fn, an, tn = col("obv_for_net"), col("obv_against_net"), col("obv_total_net")
    if len(fn) and len(an) and len(tn):
        add("obv_total_net == for_net - against_net", *_close(fn - an, tn, 1e-4))
        add("obv_total_net == for_net + against_net", *_close(fn + an, tn, 1e-4))

    # ── game state ──
    s, os_, ts = col("Score"), col("OpposingScore"), col("TotalScore")
    if len(s) and len(os_) and len(ts):
        add("TotalScore == Score + OpposingScore", *_close(s + os_, ts, _TOL))
    gs = col("GameState")
    if len(gs) and len(s) and len(os_):
        m = gs.notna() & s.notna() & os_.notna()
        if m.any():
            expect = np.where(s[m] > os_[m], "Winning",
                              np.where(s[m] < os_[m], "Losing", "Drawing"))
            add("GameState agrees with Score vs OpposingScore", m.sum(), (gs[m] == expect).sum())
    ogs = col("OpposingGameState")
    if len(ogs) and len(gs):
        m = gs.notna() & ogs.notna()
        if m.any():
            mirror = {"Winning": "Losing", "Losing": "Winning", "Drawing": "Drawing"}
            add("OpposingGameState mirrors GameState", m.sum(),
                (ogs[m] == gs[m].map(mirror)).sum())

    # ── clocks ──
    et, sop, tip, ttpe = col("ElapsedTime"), col("StartOfPossession"), col("TimeInPoss"), col("TimeToPossEnd")
    if len(et) and len(sop) and len(tip):
        add("ElapsedTime == StartOfPossession + TimeInPoss", *_close(sop + tip, et, 0.01))
    if len(tip) and len(ttpe):
        m = tip.notna() & ttpe.notna()
        if m.any():
            add("TimeInPoss >= 0 and TimeToPossEnd >= 0", m.sum(),
                ((tip[m] >= 0) & (ttpe[m] >= 0)).sum())

    # ── Goal flag against the observable outcome ──
    g, so, tp = col("Goal"), col("shot.outcome.name"), col("type.name")
    if len(g) and len(so):
        m = g.notna()
        if m.any():
            add("Goal == 1 iff shot.outcome.name == 'Goal'", m.sum(),
                ((g[m] == 1) == (so[m] == "Goal")).sum())
    if len(g) and len(so):
        m = so == "Goal"
        if m.any():
            add("a scoring Shot carries Goal == 1", m.sum(), (g[m] == 1).sum())
    if len(g) and len(tp):
        m = g == 1
        if m.any():
            add("Goal == 1 lands on a Shot event", m.sum(), (tp[m] == "Shot").sum())

    # ── OpposingTeam: the possession reading against the actor's opponent ──
    tn_, ptn, opp = col("team.name"), col("possession_team.name"), col("OpposingTeam")
    ht, at = col("home_team.home_team_name"), col("away_team.away_team_name")
    if len(opp) and len(ptn) and len(ht) and len(at):
        m = opp.notna() & ptn.notna()
        if m.any():
            other = np.where(ptn[m] == ht[m], at[m], ht[m])
            add("OpposingTeam == the side not in POSSESSION (doc's wording)", m.sum(),
                (opp[m] == other).sum())
    if len(opp) and len(tn_) and len(ht) and len(at):
        m = opp.notna() & tn_.notna()
        if m.any():
            # It mirrors the ACTOR, which differs from the possession team on every defensive
            # event (Pressure, Clearance, Duel, Block).
            other = np.where(tn_[m] == ht[m], at[m], ht[m])
            add("OpposingTeam == the ACTOR's opponent (measured)", m.sum(), (opp[m] == other).sum())
            add("OpposingTeam != team.name", m.sum(), (opp[m] != tn_[m]).sum())

    # ── shot geometry ──
    d, di = col("density"), col("density.incone")
    if len(d) and len(di):
        m = d.notna() & di.notna()
        if m.any():
            add("density.incone <= density", m.sum(), (di[m] <= d[m] + _TOL).sum())
    d1, d2 = col("distance.ToD1"), col("distance.ToD2")
    if len(d1) and len(d2):
        m = d1.notna() & d2.notna()
        if m.any():
            add("|ToD2| >= |ToD1| (nearest vs second-nearest)", m.sum(),
                (d2[m].abs() >= d1[m].abs() - _TOL).sum())
    _shot_geometry(ev, add, diag)
    for k in ("1", "2"):
        a, b = col(f"distance.ToD{k}"), col(f"distance.ToD{k}.360")
        if len(a) and len(b):
            add(f"ToD{k}.360 == ToD{k}", *_close(a, b, 1e-4))
            add(f"ToD{k}.360 == |ToD{k}|", *_close(a.abs(), b, 1e-4))
    ic, dic = col("InCone.GK"), col("DefendersInCone")
    if len(ic) and len(dic):
        m = ic.notna() & dic.notna()
        if m.any():
            add("InCone.GK is boolean (values in {0,1})", m.sum(), ic[m].isin([0, 1]).sum())
            add("InCone.GK - DefendersInCone in {0,1}", m.sum(),
                (ic[m] - dic[m]).isin([0, 1]).sum())
    ag, ak, ad = col("AngleToGoal"), col("AngleToKeeper"), col("AngleDeviation")
    if len(ag) and len(ak) and len(ad):
        add("AngleDeviation == |AngleToGoal - AngleToKeeper|", *_close((ag - ak).abs(), ad, 0.01))
        add("AngleDeviation == AngleToGoal - AngleToKeeper", *_close(ag - ak, ad, 0.01))
    if len(ag):
        m = ag.notna()
        if m.any():
            add("AngleToGoal within [0, 180]", m.sum(), ((ag[m] >= 0) & (ag[m] <= 180)).sum())
    if len(ak):
        m = ak.notna()
        if m.any():
            add("AngleToKeeper within [0, 180]", m.sum(), ((ak[m] >= 0) & (ak[m] <= 180)).sum())
    dg, dk, dsgk = col("DistToGoal"), col("DistToKeeper"), col("DistSGK")
    if len(dg) and len(dk) and len(dsgk):
        add("DistToKeeper + DistSGK == DistToGoal", *_close(dk + dsgk, dg, 0.01))
        m = dg.notna() & dk.notna() & dsgk.notna()
        if m.any():
            add("DistToKeeper + DistSGK >= DistToGoal (triangle)", m.sum(),
                ((dk[m] + dsgk[m]) >= dg[m] - 1e-4).sum())
    ab, db = col("AttackersBehindBall"), col("DefendersBehindBall")
    for name, c, cap in (("AttackersBehindBall <= 11", ab, 11), ("DefendersBehindBall <= 11", db, 11)):
        if len(c):
            m = c.notna()
            if m.any():
                add(name, m.sum(), (c[m] <= cap).sum())
    gx, gy = col("location.x.GK"), col("location.y.GK")
    if len(gx) and len(gy):
        m = gx.notna() & gy.notna()
        if m.any():
            add("keeper within the pitch [0,120]x[0,80]", m.sum(),
                ((gx[m].between(0, 120)) & (gy[m].between(0, 80))).sum())
            add("keeper in the defended half (x > 60)", m.sum(), (gx[m] > 60).sum())
    sih = col("shot_impact_height")
    if len(sih):
        m = sih.notna()
        if m.any():
            add("shot_impact_height > 0 (a real height, not a zero default)", m.sum(),
                (sih[m] > 0).sum())
    av = col("avevelocity")
    if len(av):
        m = av.notna()
        if m.any():
            add("avevelocity <= 120 (a plausible ball speed)", m.sum(), (av[m] <= 120).sum())

    # ── xG family ──
    ex, up, sb = col("shot.shot_execution_xg"), col("shot.shot_execution_xg_uplift"), col("shot.statsbomb_xg")
    if len(ex) and len(up) and len(sb):
        add("execution_xg_uplift == execution_xg - statsbomb_xg", *_close(ex - sb, up, 1e-3))
    for c in ("shot.statsbomb_xg", "shot.shot_execution_xg", "shot.gk_save_difficulty_xg",
              "pass.pass_success_probability", "duel.duel_win_probability",
              "duel.hops_rating", "duel.hops_opponent_rating"):
        s2 = col(c)
        if len(s2):
            m = s2.notna()
            if m.any():
                add(f"{c} within [0, 1]", m.sum(), ((s2[m] >= 0) & (s2[m] <= 1)).sum())

    # ── defensive responsibility ──
    psum, players = col("defensive_responsibility.probability_sum"), col("defensive_responsibility.players")
    if len(psum) and len(players):
        m = psum.notna() & players.map(lambda v: isinstance(v, list) and len(v) > 0)
        if m.any():
            got = players[m].map(lambda v: sum(float(e.get("probability", 0)) for e in v))
            add("probability_sum == sum of players[].probability", *_close(got, psum[m], 1e-3))
            add("probability_sum <= 1", int(m.sum()), int((psum[m] <= 1 + 1e-6).sum()))
            add("defensive_responsibility.players has 22 entries", int(m.sum()),
                int((players[m].map(len) == 22).sum()))

    # ── other counts ──
    nd = col("num_defenders_on_goal_side_of_actor")
    if len(nd):
        m = nd.notna()
        if m.any():
            add("num_defenders_on_goal_side_of_actor <= 11", m.sum(), (nd[m] <= 11).sum())
    dnd = col("distance_to_nearest_defender")
    if len(dnd):
        m = dnd.notna()
        if m.any():
            add("distance_to_nearest_defender <= 60", m.sum(), (dnd[m] <= 60).sum())
    bred = col("ball_receipt_exceeds_distance")
    if len(bred):
        m = bred.notna()
        if m.any():
            add("ball_receipt_exceeds_distance in {2,5,10}", m.sum(), bred[m].isin([2, 5, 10]).sum())

    # ── set_piece_phase, and the 360 block ──
    sp, pp = col("set_piece_phase"), col("play_pattern.name")
    if len(sp) and len(pp):
        m = sp.notna()
        if m.any():
            add("set_piece_phase only on set-piece play patterns", m.sum(),
                pp[m].isin(["From Corner", "From Free Kick", "From Throw In"]).sum())
    vpc, ff = col("visible_player_counts"), col("freeze_frame")
    if len(vpc) and len(ff):
        m = vpc.map(lambda v: isinstance(v, list) and len(v) > 0) & \
            ff.map(lambda v: isinstance(v, list) and len(v) > 0)
        if m.any():
            tot = vpc[m].map(lambda v: sum(int(e.get("count", 0)) for e in v))
            add("sum(visible_player_counts) == len(freeze_frame)", m.sum(),
                (tot == ff[m].map(len)).sum())
    return out, diags


def _match_checks(match_id: int) -> "tuple[list, list]":
    ev = L.load_events(match_id, columns=_COLS)
    return _checks(ev)


def _diagnostics(results: list) -> pd.DataFrame:
    """Fold the per-match diagnostic records into one row per recovered field.

    Sums, min and max are order-insensitive; the set union is capped at 4096 values, and joblib
    returns results in submission order, which is what keeps --jobs identical."""
    agg: dict = {}
    for _, diags in results:
        for field, pop, scored, blind, lo, hi, vals in diags:
            a = agg.setdefault(field, [0, 0, 0, np.inf, -np.inf, set()])
            a[0] += pop
            a[1] += scored
            a[2] += blind
            a[3], a[4] = min(a[3], lo), max(a[4], hi)
            if len(a[5]) < 4096:
                a[5] |= vals
    rows = [{"field": f, "populated": v[0], "scored": v[1], "no_frame": v[2],
             "scored_pct": round(100 * v[1] / v[0], 2) if v[0] else float("nan"),
             "pred_min": v[3], "pred_max": v[4], "pred_cardinality": len(v[5])}
            for f, v in agg.items()]
    return pd.DataFrame(rows).sort_values("scored_pct")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None,
                    help="head slice; BIASED, the catalog is date-ordered. Prefer --sample.")
    ap.add_argument("--sample", type=int, default=None, help="random matches, seeded; the smoke test")
    ap.add_argument("--jobs", type=int, default=1)
    ap.add_argument("--diagnose", action="store_true",
                    help="write the per-field population table and enforce the stop bands")
    args = ap.parse_args()

    catalog = L.list_matches(None)
    if args.sample:
        catalog = catalog.sample(min(args.sample, len(catalog)), random_state=0)
    elif args.limit:
        catalog = catalog.iloc[:args.limit]
        # shot.freeze_frame changes composition in late January 2026 (the shooter enters their own
        # frame), and the catalog is date-ordered, so a short head slice sees one regime and reports
        # its mean as the grant's.
        if args.limit < 200:
            print(f"WARNING: --limit {args.limit} is a date-ordered head slice, not a sample. "
                  f"Use --sample {args.limit} for a figure that generalises.")
    mids = catalog["match_id"].tolist()

    print(f"INFERRED-CLAIM AUDIT -- {len(mids)} matches, {args.jobs} worker(s)")
    t0 = time.time()
    if args.jobs == 1:
        results = [_match_checks(m) for m in mids]
    else:
        results = Parallel(n_jobs=args.jobs, verbose=5)(delayed(_match_checks)(m) for m in mids)

    agg: dict = {}
    for res, _ in results:
        for name, n, ok in res:
            if name not in agg:
                agg[name] = [0, 0]
            agg[name][0] += n
            agg[name][1] += ok

    rows = [{"check": k, "rows": v[0], "holds": v[1],
             "pct": round(100 * v[1] / v[0], 2) if v[0] else float("nan")}
            for k, v in agg.items()]
    df = pd.DataFrame(rows).sort_values("pct")
    print(f"  {len(df)} checks over {time.time() - t0:.1f}s\n")

    with pd.option_context("display.width", 200, "display.max_colwidth", 60):
        print(df.to_string(index=False))

    # A sampled or sliced run is tagged, because the untagged table is the grant's and
    # data_provider.tex cites it.
    tag = f"_sample{args.sample}" if args.sample else (f"_limit{args.limit}" if args.limit else "")
    out = _REPO / "docs" / "hudl" / f"inferred_audit{tag}.tsv"
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, sep="\t", index=False)
    print(f"\nsaved {out}")

    broken = df[df["pct"] < 99.5]
    print(f"\n{len(broken)} of {len(df)} claims hold on under 99.5% of rows.")

    if args.diagnose:
        dg = _diagnostics(results)
        dout = out.with_name(f"inferred_audit_diagnostics{tag}.tsv")
        dg.to_csv(dout, sep="\t", index=False)
        with pd.option_context("display.width", 200):
            print(f"\nPER-FIELD POPULATION\n{dg.to_string(index=False)}")
        print(f"\nsaved {dout}")
        halt = dg[(dg["scored_pct"] < 95.0) | (dg["pred_cardinality"] <= 1)]
        for _, r in halt.iterrows():
            print(f"STOP {r['field']}: scored {r['scored']} of {r['populated']} populated "
                  f"({r['scored_pct']}%), prediction takes {r['pred_cardinality']} value(s)")
        if len(halt):
            sys.exit(2)


if __name__ == "__main__":
    main()
