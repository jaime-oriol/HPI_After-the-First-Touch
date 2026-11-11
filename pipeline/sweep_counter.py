"""sweep_counter — is the conceded-counter arm estimable at ANY window?

`clear_counter` fires on almost none of the deliveries at the 8 s transition window
(measured.md §counter has the figure), so one of the three terminal outcomes
form_answers.md Q2 promises is close to an empty class. The window is swept, and the
arm is decided on the numbers the sweep shows.

The sweep is a SENSITIVITY DISPLAY, never a selection. All windows are reported or
none, and the window is not chosen by the effect size, by a p-value, or by whether
the arm becomes estimable. Pre-registered stopping rule, all three must hold:

    n >= MIN_CELL events               the MIN_CELL publication floor
    possession contiguity >= 95%       the "counter" is still the chain that started
                                       at the clearance, not a later possession
    full lookahead >= 98%              the window fits inside the period, so it is
                                       not silently truncated

If no window satisfies all three, the arm is declared NOT ESTIMABLE and this sweep is
the evidence. The honest declaration also needs what the window cannot fix: s01's
outcome ladder gives the delivery's own attacking shot priority over `clear_counter`, so
widening only ever adds counters among deliveries with no shot of their own, a non-random
subsample. Both numbers are persisted per window, `events` and `outranked`.

    python pipeline/sweep_counter.py --jobs 20
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO / "src"))
sys.path.insert(0, str(_REPO / "pipeline"))

import loader_sb as L                                            # noqa: E402
from build_table import kind_tag                                 # noqa: E402
from constants import MIN_CELL                                   # noqa: E402
from s01_phases import SetPieceGrammar, clearing_contacts, _col  # noqa: E402
from validate import poisson_ratio_ci                            # noqa: E402

WINDOWS = (8.0, 12.0, 15.0, 20.0, 30.0, 45.0, 60.0, 90.0)
MIN_CONTIGUITY = 0.95
MIN_LOOKAHEAD = 0.98


def sweep_match(match_id: int, kinds: tuple) -> list:
    """One match of the sweep, read from the events cache."""
    return sweep_events(L.load_events(match_id), match_id, kinds)


def sweep_events(ev: pd.DataFrame, match_id: int, kinds: tuple) -> list:
    """Per delivery: the clearance, the defending shots after it, and for each the
    elapsed time, whether the possession chain is unbroken, and the lookahead left.

    The clearance is the one s01 labels, taken from `label` rather than searched for here, so
    the swept set IS the cleared set and closes on the same episode end. The defending shots
    after it are then searched over the rest of the period, because the widest window runs
    past the episode on purpose. Whether the delivery produced its own attacking shot comes
    from the same label, own goals included: that shot, and no later one, is what the outcome
    ladder ranks above a counter."""
    g = SetPieceGrammar(ev, kinds=kinds)
    d = g.deliveries()
    if d.empty:
        return []
    # The period is the hard edge of the observable horizon: _sequence never crosses it.
    period_end = g.ev.groupby("period")["_t"].max().to_dict()
    out = []
    for _, c in d.iterrows():
        att, t0 = c["team.name"], c["_t"]
        lab = g.label(c)
        if lab.time_to_clearance_s is None:
            continue
        t_clear = float(t0 + lab.time_to_clearance_s)
        post = g.ev[(g.ev["period"] == c["period"]) & (g.ev["_t"] > t0)].sort_values("_t")
        deff = g._defending_team(post, att)
        clear = clearing_contacts(post, deff, t_clear).iloc[0]
        poss_at_clear = clear.get("possession")
        after = post[post["_t"] > t_clear]
        shots = after[(_col(after, "type.name") == "Shot") & (_col(after, "team.name") == deff)]
        rec = {"match_id": match_id, "set_piece_id": c["id"], "set_piece_type": c["set_piece_type"],
               "has_att_shot": lab.n_att_shots > 0,
               "lookahead_s": float(period_end[c["period"]]) - t_clear}
        for w in WINDOWS:
            s = shots[shots["_t"] <= t_clear + w]
            rec[f"n{w:g}"] = int(len(s))
            # Contiguous = the shot is still in the possession chain that the clearance
            # was in. Without this, a 90 s window counts the opponent's NEXT attack.
            rec[f"c{w:g}"] = int(bool(len(s)) and s.iloc[0].get("possession") == poss_at_clear)
        out.append(rec)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--jobs", type=int, default=20)
    # Corners only by default: a rate pooled over two populations is a mix weight, not a rate
    # (definitions.md §8), and the arm this sweep declares not estimable is the corner one.
    ap.add_argument("--kinds", type=str, default="corner")
    ap.add_argument("--out", type=str, default=None,
                    help="output parquet (default: counter_sweep.parquet, tagged by --kinds and "
                         "--limit)")
    args = ap.parse_args()

    kinds = tuple(k.strip() for k in args.kinds.split(","))
    mids = L.list_matches()["match_id"].astype(int).tolist()
    if args.limit:
        mids = mids[:args.limit]
    chunks = Parallel(n_jobs=args.jobs, verbose=5)(delayed(sweep_match)(m, kinds) for m in mids)
    df = pd.DataFrame([r for c in chunks for r in c])

    print(f"\nCOUNTER-WINDOW SWEEP — {len(mids)} matches, {len(df)} cleared deliveries "
          f"({', '.join(kinds)})\n")
    print(f"  {'window':>7s} {'events':>7s} {'rate':>8s} {'95% Garwood':>18s} "
          f"{'contiguous':>11s} {'lookahead':>10s}  verdict")
    rows = []
    for w in WINDOWS:
        n = int((df[f"n{w:g}"] > 0).sum())
        cont = df[f"c{w:g}"].sum() / n if n else float("nan")
        look = float((df["lookahead_s"] >= w).mean())
        lo, hi = poisson_ratio_ci(n, len(df))
        hi = min(hi, 1.0)
        ok = (n >= MIN_CELL) and (cont >= MIN_CONTIGUITY) and (look >= MIN_LOOKAHEAD)
        why = []
        if n < MIN_CELL:
            why.append(f"n<{MIN_CELL}")
        if not np.isnan(cont) and cont < MIN_CONTIGUITY:
            why.append("possession broken")
        if look < MIN_LOOKAHEAD:
            why.append("truncated")
        print(f"  {w:6.0f}s {n:>7d} {100 * n / len(df):>7.2f}% "
              f"[{100 * lo:>6.2f}%,{100 * hi:>6.2f}%] {100 * cont:>10.1f}% {100 * look:>9.1f}%  "
              f"{'PASSES' if ok else 'fails: ' + ', '.join(why)}")
        rows.append({"window_s": w, "events": n, "rate": n / len(df),
                     "contiguity": cont, "lookahead": look, "passes": ok,
                     "outranked": int(((df[f"n{w:g}"] > 0) & df["has_att_shot"]).sum())})

    sweep = pd.DataFrame(rows)
    last = sweep.iloc[-1]
    print(f"\n  structural suppression: {int(last['outranked'])} of the {int(last['events'])} "
          f"deliveries with a defending shot at {last['window_s']:g}s produced their own "
          f"attacking shot.")
    print("  Those never reach the `clear_counter` class, whatever the window, because the")
    print("  outcome ladder gives the attacking shot priority. counter_shot stays a separate")
    print("  boolean on the label, and the rate is reported off THAT, not off the class.")

    if not sweep["passes"].any():
        print("\n  VERDICT: the transition arm is NOT ESTIMABLE on this data.")
        print(f"  No window satisfies n>={MIN_CELL} AND contiguity>={100 * MIN_CONTIGUITY:.0f}% "
              f"AND lookahead>={100 * MIN_LOOKAHEAD:.0f}% at once, so")
        print("  the window was never the problem. lambda in xSecondBall multiplies a term")
        print("  estimated on too few events, and s05 correctly reports conceded_xg degenerate.")
    else:
        best = sweep[sweep["passes"]].iloc[0]
        print(f"\n  VERDICT: the smallest window satisfying all three is {best['window_s']:g}s "
              f"({int(best['events'])} events).")

    # Neither a --limit run nor another population may overwrite the grant's corner sweep: the
    # "not estimable" verdict rests on it.
    out = Path(args.out) if args.out else (
        _REPO / "data" / "derived"
        / f"counter_sweep{kind_tag(kinds)}{f'_limit{args.limit}' if args.limit else ''}.parquet")
    out.parent.mkdir(parents=True, exist_ok=True)
    sweep.to_parquet(out)
    print(f"\nsaved {out}  [gitignored]")


if __name__ == "__main__":
    main()
