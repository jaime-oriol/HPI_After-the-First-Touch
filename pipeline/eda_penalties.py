"""eda_penalties — the penalties a delivery wins inside its episode, which no segment values.

s01 values a delivery by the attacking shots inside its window, so a penalty the defence concedes
during the episode is valued nowhere: the foul is a whistle and not a touch, and the kick comes
minutes later as a set piece of its own. This driver finds those penalties on the paper's two
populations, so the after-first-touch share can be read with them credited beside the share
without them (report_figures, block `penalties`).

WON IN THE EPISODE: a foul the referee marks as a penalty, on either side's record (the defence's
`foul_committed.penalty` or the attack's `foul_won.penalty`), from the kick to one
SECOND_BALL_WINDOW_S after the first contact, or to one window after the kick when nothing touched
the ball, which is as far as s01 looks for that contact. Its PHASE places the whistle before the
first contact, at it, or after it; only `after` belongs with what follows the first touch. The KICK
is the attacking team's next penalty shot in the same period, credited at the provider's xG like
every shot of the frontier.

The run stops if its deliveries, or their first contacts, are not the delivery grain's (eda_build),
or if one kick is matched to two deliveries.

    python pipeline/eda_penalties.py --jobs 20
    python pipeline/eda_penalties.py --limit 200 --jobs 8      # smoke, on the smoke's own grain
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO / "src"))
sys.path.insert(0, str(_REPO / "pipeline"))
_DER = _REPO / "data" / "derived"

import loader_sb as L                                    # noqa: E402
from eda_build import COMBINED_KINDS                     # noqa: E402
from s01_phases import SetPieceGrammar, _col             # noqa: E402

PHASES = ("before", "at", "after")


def paths(limit: int | None) -> tuple[Path, Path]:
    """(the delivery grain it is checked against, the record it writes): a --limit run pairs with
    the smoke's grain over the same first matches and never writes the grant's name."""
    tail = f"_limit{limit}" if limit else ""
    return _DER / f"eda_deliveries{tail}.parquet", _DER / f"eda_penalties{tail}.parquet"


def episode(t0: float, to_first_contact: float | None, window: float) -> tuple[float | None, float]:
    """(the first contact's clock, the episode's end) for a kick at `t0`."""
    t_fc = None if to_first_contact is None or pd.isna(to_first_contact) else t0 + to_first_contact
    return t_fc, (t0 if t_fc is None else t_fc) + window


def phase(t_foul: float, t_fc: float | None) -> str:
    """Where the whistle falls against the first contact; with no contact it came first."""
    if t_fc is None or t_foul < t_fc:
        return "before"
    return "at" if t_foul == t_fc else "after"


def penalty_fouls(ev: pd.DataFrame, attack: str) -> pd.Series:
    """True on the events that give `attack` a penalty, on either side's record."""
    typ, team = _col(ev, "type.name"), _col(ev, "team.name")
    # The export leaves the flag out instead of writing False, so a missing flag is no penalty.
    committed = ((typ == "Foul Committed") & _col(ev, "foul_committed.penalty").eq(True)
                 & (team != attack))
    won = (typ == "Foul Won") & _col(ev, "foul_won.penalty").eq(True) & (team == attack)
    return committed | won


def next_kick(ev: pd.DataFrame, attack: str, period: int, after: float) -> pd.Series | None:
    """The attacking team's first penalty shot in `period` after the clock `after`, or None."""
    k = ev[(ev["period"] == period) & (ev["_t"] > after) & (_col(ev, "type.name") == "Shot")
           & (_col(ev, "shot.type.name") == "Penalty") & (_col(ev, "team.name") == attack)]
    return None if k.empty else k.sort_values("_t", kind="stable").iloc[0]


def match_rows(match_id: int) -> list[dict]:
    """One match's deliveries of both populations, each with the penalty it won in its episode."""
    g = SetPieceGrammar(L.load_events(match_id), kinds=COMBINED_KINDS)
    e, rows = g.ev, []
    for _, d in g.deliveries().iterrows():
        t0, per, att = float(d["_t"]), d["period"], d["team.name"]
        to_fc = g.label(d).time_to_first_contact_s
        t_fc, end = episode(t0, to_fc, g.SECOND_BALL_WINDOW_S)
        fouls = e[(e["period"] == per) & e["_t"].between(t0, end) & penalty_fouls(e, att)]
        row = {"match_id": match_id, "set_piece_id": d["id"], "set_piece_type": d["set_piece_type"],
               "attacking_team": att, "time_to_first_contact_s": to_fc, "penalty": not fouls.empty,
               "phase": None, "foul_t": np.nan, "kick_id": None, "kick_t": np.nan,
               "kick_outcome": None, "kick_xg": np.nan, "kick_goal": False}
        if not fouls.empty:
            t_foul = float(fouls["_t"].min())
            row.update(foul_t=t_foul, phase=phase(t_foul, t_fc))
            # The kick is searched over the rest of the period: a VAR review can put minutes
            # between the whistle and the kick, far past any window.
            k = next_kick(e, att, per, t_foul)
            if k is not None:
                result = k["shot.outcome.name"]
                row.update(kick_id=k["id"], kick_t=float(k["_t"]), kick_outcome=result,
                           kick_xg=float(k["shot.statsbomb_xg"]), kick_goal=result == "Goal")
        rows.append(row)
    return rows


def check(tab: pd.DataFrame, grain: pd.DataFrame) -> None:
    """Stop unless the deliveries and their first contacts are the grain's, and every kick is
    credited to one delivery only. The grain also holds the short corners no cross followed, which
    have no delivery and so no episode to win a penalty in: they are not rows the run must find."""
    if "anchor" in grain.columns:
        grain = grain[grain["anchor"] != "none"]
    m = tab.merge(grain[["set_piece_id", "set_piece_type", "time_to_first_contact_s"]],
                  on="set_piece_id", how="outer", suffixes=("", "_grain"), indicator=True)
    if (m["_merge"] != "both").any() or m["set_piece_id"].duplicated().any():
        found = m["_merge"].value_counts().to_dict()
        raise SystemExit(f"STOP: the deliveries are not the grain's ({found})")
    if not (m["set_piece_type"] == m["set_piece_type_grain"]).all():
        raise SystemExit("STOP: a delivery's population differs from the grain's")
    # Exact equality on purpose: both sides come out of the same s01 label on the same clock.
    a = m["time_to_first_contact_s"].astype(float).to_numpy()
    b = m["time_to_first_contact_s_grain"].astype(float).to_numpy()
    na = np.isnan(a)
    if not (np.array_equal(na, np.isnan(b)) and np.array_equal(a[~na], b[~na])):
        raise SystemExit("STOP: a first contact differs from the grain's")
    if tab["kick_id"].dropna().duplicated().any():
        raise SystemExit("STOP: one penalty kick is credited to two deliveries")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--jobs", type=int, default=20)
    ap.add_argument("--limit", type=int, default=None, help="first N matches only (smoke)")
    args = ap.parse_args()
    grain_path, out = paths(args.limit)
    if not grain_path.exists():
        raise SystemExit(f"STOP: {grain_path.name} absent; eda_build.py writes it")

    mids = L.list_matches()["match_id"].tolist()
    if args.limit:
        mids = mids[:args.limit]
    t0 = time.time()
    chunks = Parallel(n_jobs=args.jobs)(delayed(match_rows)(int(m)) for m in mids)
    tab = pd.DataFrame([r for chunk in chunks for r in chunk])
    check(tab, pd.read_parquet(grain_path))
    print(f"{len(tab)} deliveries / {len(mids)} matches in {time.time() - t0:.0f}s "
          f"({args.jobs} worker(s))")
    for kind, sub in tab.groupby("set_piece_type"):
        won = sub[sub["penalty"]]
        kicks = won[won["kick_id"].notna()]
        print(f"  {kind:16s} {len(sub):6d} deliveries, {len(won)} penalties "
              f"({won['phase'].value_counts().reindex(PHASES, fill_value=0).to_dict()}), "
              f"{len(kicks)} kicked, {int(kicks['kick_goal'].sum())} scored, "
              f"xG {kicks['kick_xg'].sum():.2f}")
    tab.to_parquet(out, index=False)
    print(f"saved {out}  [gitignored]")


if __name__ == "__main__":
    main()
