"""run_positive_control — can this harness see anything at all?

Most of what this record publishes about the 360 is a null or a boundary call, and a reader is
entitled to ask whether the pipeline could have found signal had there been any. That question is
not answered by another null. It is answered on a target where signal must exist by construction:
the first contact happens a second or two after the kick, so the bodies standing around the ball
AT the kick are the ones who go on to contest it. A design that cannot see that is broken; one
that sees it has earned the nulls it reports elsewhere.

This is a CONTROL and not a claim about football. Who wins the first contact is not the outcome the
paper models, and nothing here enters any published model: the target is `won_first_contact` off the
delivery grain, on the deliveries that have one, and the features are the delivery-instant design s03
already publishes, unchanged.

THE POPULATION IS EVERY CORNER WITH A FIRST CONTACT, and that is the point of the driver rather than
a detail. Running it on the second-ball grant instead looks natural and is wrong: that grant keeps
corners a defender CLEARED, so the first contact is the clearance itself on most of its rows and the
target collapses towards the defence. Conditioning a control on an outcome that arrives after the
contact it predicts would make a pass here meaningless.

Three arms on one set of grouped folds, each scored out of fold on rows it never trained on:

    prior       intercept only, the rate a coach already knows
    delivery    the delivery-location block alone (_KICK_INTRINSIC), the repo's real baseline
    full_360    the whole delivery freeze-frame block

    P1  the freeze-frame beats the prior            pivotal-free: cluster bootstrap below zero
    P2  it beats the delivery-location baseline     the same reading

Both rules were written before the run. A shuffled-target arm rides along, scored against the prior
fitted on that same shuffled target: if the features still buy anything there, the harness leaks and
no reading here means anything. --kind free_kick_flank runs the same control on the flank free
kicks from x >= 85, the paper's second population, without the long-corner reading: a free kick is
never short.

    python pipeline/run_positive_control.py --boot 2000
    python pipeline/run_positive_control.py --boot 2000 --kind free_kick_flank   # _<kind>
    python pipeline/run_positive_control.py --limit 40 --boot 200   # the path, not the numbers
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold

_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO / "src"))
sys.path.insert(0, str(_REPO / "pipeline"))
_DER = _REPO / "data" / "derived"

from build_table import kind_tag, published_dir                     # noqa: E402
from eda_build import COMBINED_KINDS                                 # noqa: E402
from s03_value import CompetingOutcomesModel, _KICK_INTRINSIC        # noqa: E402
from s04b_snapshot import row_logloss                                # noqa: E402
from validate import assert_no_group_leak, cluster_bootstrap_mean, shuffle_labels   # noqa: E402

# The registered precision, and the only one the published record is written at.
_BOOT = 2000

# Declared here rather than inherited: s03's grid tunes a multiclass head, and this is one binary
# control. Same family (l2), same nested-by-fold selection, a grid wide enough to bracket the C
# that head adopts.
CONTROL_C_GRID = (0.01, 0.05, 0.25, 1.0, 4.0)
# (numeric block, delivery categoricals). The flag is NOT decoration: `_design(numeric=[])` empties
# the numeric block and still attaches the delivery dummies and `short_corner`, so a prior asked for
# that way arrives carrying the feature that tells the crosses after a short corner from a long
# corner's delivery, and every arm ties with it.
ARMS = {"prior": ([], False), "delivery": (list(_KICK_INTRINSIC), True), "full_360": (None, True)}
CONTROL_DECISIONS = (("full_360", "prior"), ("full_360", "delivery"), ("delivery", "prior"))


def _fit(X: np.ndarray, y: np.ndarray, C: float) -> LogisticRegression:
    return LogisticRegression(max_iter=5000, C=C, penalty="l2").fit(X, y)


def _scored(Xtr: np.ndarray, ytr: np.ndarray, Xte: np.ndarray, C: float) -> np.ndarray:
    """P(win) on a held-out block, prepared by s03's own train-only median-impute and standardize.

    `_design` leaves the NaN in on purpose, because the published model imputes inside the fold
    and a table-wide median would leak. Reusing `_fit_prep` rather than an imputer of this
    driver's own keeps the control on the same preparation as the model it is controlling."""
    tf = CompetingOutcomesModel._fit_prep(Xtr)
    return _fit(tf(Xtr), ytr, C).predict_proba(tf(Xte))[:, 1]


def _pick_c(X: np.ndarray, y: np.ndarray, g: np.ndarray) -> float:
    """The C with the best grouped inner-fold log loss, chosen inside one outer fold only.

    An inner fold whose training block holds one class is skipped rather than scored: it cannot
    fit, and counting it as a loss would penalise whichever C happened to meet it."""
    best, best_ll = CONTROL_C_GRID[0], np.inf
    inner = GroupKFold(n_splits=min(3, len(np.unique(g))))
    for C in CONTROL_C_GRID:
        lls = []
        for tr, te in inner.split(X, y, g):
            assert_no_group_leak(g, tr, te, "match_id")
            if len(np.unique(y[tr])) < 2:
                continue
            lls.append(row_logloss(_scored(X[tr], y[tr], X[te], C), y[te]).mean())
        if lls and np.mean(lls) < best_ll:
            best, best_ll = C, float(np.mean(lls))
    return best


def oof_proba(X: np.ndarray, y: np.ndarray, g: np.ndarray) -> tuple[np.ndarray, list]:
    """Out-of-fold P(the attack wins the first contact), with C re-selected inside every fold.

    Selecting C once on the whole table would score every row with a value the full data chose,
    which is the resubstitution this repo bars in the calibration step for the same reason."""
    p, picked = np.zeros(len(y), dtype=float), []
    for tr, te in GroupKFold(n_splits=5).split(X, y, g):
        assert_no_group_leak(g, tr, te, "match_id")
        C = _pick_c(X[tr], y[tr], g[tr])
        picked.append(C)
        p[te] = _scored(X[tr], y[tr], X[te], C)
    return p, picked


def arm_losses(table: pd.DataFrame, y: np.ndarray, g: np.ndarray) -> tuple[dict, dict]:
    """Per-row log loss of each arm, and the C each fold selected for it."""
    m = CompetingOutcomesModel(n_splits=5)
    out, chosen = {}, {}
    for name, (numeric, cats) in ARMS.items():
        X, _ = m._design(table, numeric=numeric, with_delivery_cats=cats)
        p, picked = oof_proba(np.asarray(X, dtype=float), y, g)
        out[name] = row_logloss(p, y)
        chosen[name] = picked
    return out, chosen


def shuffled_arm(table: pd.DataFrame, y: np.ndarray, g: np.ndarray, boot: int, mode: str,
                 seed: int = 0) -> dict:
    """The full design on a shuffled target, against the PRIOR fitted on that same target.

    THE GLOBAL SHUFFLE IS THE ONE THAT DECIDES, and the reason is the baseline. A within-match
    shuffle keeps each match's own rate, and the prior here is a single global number, so a model
    that only tracks how rates differ BETWEEN matches beats it while knowing nothing about any row
    (evidence.md §19 carries the size). A global shuffle destroys the between-match structure too,
    so anything the features buy there is the sort of leak this is looking for. The within-match
    reading is still reported, as a description of how much of the fit is match-level rather than
    as a verdict."""
    ys = shuffle_labels(y, g, mode=mode, seed=seed)
    m = CompetingOutcomesModel(n_splits=5)
    out = {}
    for name in ("full_360", "prior"):
        numeric, cats = ARMS[name]
        X, _ = m._design(table, numeric=numeric, with_delivery_cats=cats)
        p, _ = oof_proba(np.asarray(X, dtype=float), ys, g)
        out[name] = row_logloss(p, ys)
    ci = cluster_bootstrap_mean(out["full_360"] - out["prior"], g, n_boot=boot)
    return {"mode": mode, "log_loss": float(out["full_360"].mean()),
            "prior_log_loss": float(out["prior"].mean()),
            "delta": ci.point, "lo": ci.lo, "hi": ci.hi, "clean": bool(ci.hi >= 0)}


def decide(deltas: dict, clean: bool) -> dict:
    """P1 and P2, each a conjunction of its interval and a clean harness.

    The conjunction is the point: an interval below zero on a harness that leaks decides nothing,
    and reporting the two separately would let a reader take the first without the second."""
    return {name: ("true" if deltas[key]["below_zero"] and clean else "false")
            for name, key in (("P1", "full_360-prior"), ("P2", "full_360-delivery"))}


def main() -> None:
    """Build the control table, score the three arms, and decide P1 and P2 against their rule.

    Only the grant's table of the population and the delivery grain, at the registered draws, write
    the published record; any other pair writes a name tagged after its inputs, and --limit after
    the subset."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None, help="first N matches: the path, not the numbers")
    ap.add_argument("--boot", type=int, default=_BOOT, help="match-clustered bootstrap draws")
    ap.add_argument("--table", type=str, default=None,
                    help="the population's table (default: corners[_kind].parquet)")
    ap.add_argument("--deliveries", type=str, default=None,
                    help="the delivery grain carrying won_first_contact (default: data/derived)")
    ap.add_argument("--out", type=str, default=None)
    ap.add_argument("--kind", choices=COMBINED_KINDS, default="corner",
                    help="the population: corners, or the flank free kicks from x >= 85")
    args = ap.parse_args()
    t0 = time.time()

    kt = kind_tag((args.kind,))
    tail = f"_limit{args.limit}" if args.limit else ""
    tab_path = Path(args.table) if args.table else _DER / f"corners{kt}{tail}.parquet"
    dlv_path = Path(args.deliveries) if args.deliveries else _DER / f"eda_deliveries{tail}.parquet"
    grant = (tab_path.resolve() == (_DER / f"corners{kt}.parquet").resolve()
             and dlv_path.resolve() == (_DER / "eda_deliveries.parquet").resolve())
    tag = kt + ("" if grant else (tail or f"_{tab_path.stem}_{dlv_path.stem}"))
    out = Path(args.out) if args.out else _DER / f"positive_control{tag}.parquet"
    name = f"positive_control{kt}.parquet"
    if published_dir(out.parent) and out.name == name and not (grant and args.boot == _BOOT):
        raise SystemExit(f"STOP: a run off the grant's tables or at other than {_BOOT} draws may not "
                         f"write the published {name}; pass --out")
    for p in (tab_path, dlv_path):
        if not p.exists():
            raise SystemExit(f"STOP: {p.name} absent; build_table.py and eda_build.py write it")
    dlv = pd.read_parquet(dlv_path)
    dlv = dlv.loc[dlv["set_piece_type"] == args.kind, ["set_piece_id", "won_first_contact"]]
    table = pd.read_parquet(tab_path).merge(dlv, on="set_piece_id", how="inner")
    table = table[table["won_first_contact"].notna()].sort_values(["match_id", "set_piece_id"])
    table = table.reset_index(drop=True)
    y = table["won_first_contact"].to_numpy().astype(int)
    g = table["match_id"].to_numpy()
    print(f"positive control: {len(table)} {args.kind} deliveries / {pd.Series(g).nunique()} "
          f"matches, the attack wins the first contact on {100 * y.mean():.1f}%", flush=True)

    losses, chosen = arm_losses(table, y, g)
    rows = [{"block": "population", "key": "corners", "n": len(table), "point": float(y.mean())}]
    for name, ll in losses.items():
        print(f"  {name:10s} {ll.mean():.5f}   C per fold {chosen[name]}", flush=True)
        rows.append({"block": "arm", "key": name, "n": len(y), "point": float(ll.mean()),
                     "params": str(chosen[name])})

    print("PAIRED DELTAS (a minus b; negative = a wins):", flush=True)
    for a, b in CONTROL_DECISIONS:
        ci = cluster_bootstrap_mean(losses[a] - losses[b], g, n_boot=args.boot)
        verdict = "BEATS" if ci.hi < 0 else ("covers zero" if ci.lo <= 0 <= ci.hi else "loses")
        print(f"  {a + '-' + b:22s} {ci.point:+.5f} [{ci.lo:+.5f}, {ci.hi:+.5f}]  {verdict}",
              flush=True)
        rows.append({"block": "delta", "key": f"{a}-{b}", "n": len(y), "point": ci.point,
                     "lo": ci.lo, "hi": ci.hi, "below_zero": bool(ci.hi < 0)})

    # The crosses after a short corner are another delivery, struck from the wing a few seconds after
    # the kick, and the attack wins their first contact far more often than a long corner's. If P1
    # rested on them alone the control would pass on that difference of population, so the same
    # three arms run again with those rows out and the reading travels beside the headline. A free
    # kick is never short.
    if args.kind == "corner":
        long = table[~table["short_corner"].astype(bool)].reset_index(drop=True)
        yl, gl = long["won_first_contact"].to_numpy().astype(int), long["match_id"].to_numpy()
        ll_long, _ = arm_losses(long, yl, gl)
        print(f"LONG CORNERS ONLY ({len(long)} rows, attack wins {100 * yl.mean():.1f}%):",
              flush=True)
        for a, b in CONTROL_DECISIONS:
            ci = cluster_bootstrap_mean(ll_long[a] - ll_long[b], gl, n_boot=args.boot)
            print(f"  {a + '-' + b:22s} {ci.point:+.5f} [{ci.lo:+.5f}, {ci.hi:+.5f}]"
                  f"{'  BEATS' if ci.hi < 0 else ''}", flush=True)
            rows.append({"block": "delta_long_only", "key": f"{a}-{b}", "n": len(yl),
                         "point": ci.point, "lo": ci.lo, "hi": ci.hi,
                         "below_zero": bool(ci.hi < 0)})

    ctrls = {}
    for mode in ("global", "within_match"):
        c = shuffled_arm(table, y, g, boot=args.boot, mode=mode)
        ctrls[mode] = c
        note = "clean" if c["clean"] else ("LEAKAGE SUSPECTED" if mode == "global"
                                           else "match-level fit, does not decide")
        print(f"  shuffled {mode:12s} full_360 minus its own prior: {c['delta']:+.5f} "
              f"[{c['lo']:+.5f}, {c['hi']:+.5f}]  {note}", flush=True)
        rows.append({"block": "control", "key": f"full_360-prior/shuffled_{mode}",
                     "clean": c["clean"], "point": c["delta"], "lo": c["lo"], "hi": c["hi"]})
    ctrl = ctrls["global"]

    by = {r["key"]: r for r in rows if r["block"] == "delta"}
    for name, state in decide(by, ctrl["clean"]).items():
        print(f"  {name}: {state}", flush=True)
        rows.append({"block": "verdict", "key": name, "params": state})

    out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_parquet(out, index=False)
    print(f"\nwrote {out}  [{time.time() - t0:.0f}s]", flush=True)


if __name__ == "__main__":
    main()
