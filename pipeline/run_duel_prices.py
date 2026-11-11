"""run_duel_prices — what is each duel of a corner worth, zone by zone?

A corner is two duels: the first contact, and the second ball if the first one is cleared. This
driver prices winning each one, zone by zone, in chain xG, and what ten more points of either would be
worth to a team over a season: the club-facing price list the paper offers. Pre-registered
(data/derived/prereg_2026-10-01/c1_duel_prices.md) before the confirmation leagues were
read for it; the Premier League is where the list was developed, so it is the development sample
and decides nothing.

ZONES are the partition the switch names (src/zones.py; HUDL_ZONES=C or F for one run), read on
canonical points: the first contact is the delivery's end location (evidence.md §15), the second
ball is s08's drop, both mirrored onto the y = 0 side by the kick's own y, s01's corner_side rule.

    first duel    in-window chain xG per corner, summed over its shots (every segment but `late`),
                  by the first-contact zone and by who won it; the price of a zone is attack won
                  minus defence won
    second duel   after-clearance xG per cleared corner by the drop zone and by who took the next
                  touch, the same way; `second_aft` prices the corner's after-first-touch xG, the
                  value the sheet names, on the same corners (both are s01's per-corner maxima, so
                  a what-if read across the two duels is on two aggregations)
    what-if       ten more points won in one zone: its share of the duels times 0.10 times its
                  price, times the duels a team-season holds, in xG

The list is priced on every league, the project's scope, with corners and flank free kicks as two
populations (--kind). C1a decides whether the corner list enters as a club-facing product: on the
confirmation sample (the four leagues other than the development's, pooled) the zone with the
highest first-duel price is the development's, and the first-duel prices of the zones
correlate with the development's at Spearman >= 0.7 with a match-resampled 95% interval clear of
zero, at every one of five seeds. Otherwise the list is published as a description and not as a
product. Zones under MIN_CELL duels on either side of a sample are out of the correlation and named.
Every interval resamples matches; the two samples are drawn independently inside one draw.

Reported, never deciding: each league alone; each confirmation league left out; how often the
development's top zone is the top one across the draws; how far the second duel moves between its
two values; OBV's price of the second ball beside ours, read off the provider's value of the touch
that contests it, as consistency only because OBV's window already holds the second ball; the
zone-to-zone transitions from the first duel to the second; C1a on the flank free kicks.

    python pipeline/run_duel_prices.py --jobs 20 --kind free_kick_flank
    HUDL_ZONES=F python pipeline/run_duel_prices.py --jobs 20 --out-dir data/derived/zones_F
    python pipeline/run_duel_prices.py --limit 200 --boot 50 --seeds 1 \\
        --out-dir data/derived/smoke              # the smoke's N-match tables
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy import stats

_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO / "src"))
sys.path.insert(0, str(_REPO / "pipeline"))
_DER = _REPO / "data" / "derived"

import loader_sb as L                                                   # noqa: E402
from build_table import published_dir                                  # noqa: E402
from constants import MIN_CELL, ZONE_PARTITION                          # noqa: E402
from run_aim_zone import DEVELOPMENT                                    # noqa: E402
from s01_phases import SEGMENTS                                         # noqa: E402
from s08_situation import DropSituation                                 # noqa: E402
from zones import zone_names, zone_of                                   # noqa: E402

IN_WINDOW = tuple(s for s in SEGMENTS if s != "late")
LIFT, SPEARMAN_FLOOR = 0.10, 0.7
# The registered precision, and the only one the published records are written at.
_PRICE_DRAWS, _SEEDS = 2000, 5
# Each population's shots and situation table, read from the folder that holds the delivery grain
# eda_build writes for the paper's two populations.
_DELIVERY_GRAIN = "eda_deliveries.parquet"
_INPUTS = {"corner": ("shots.parquet", "situation.parquet"),
           "free_kick_flank": ("shots_free_kick_flank.parquet", "situation_free_kick_flank.parquet")}


def inputs(der: Path, kind: str, tail: str = "") -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """(deliveries, shots, situation) of one population: its rows of the combined grain, and its
    own shots and situation table; `tail` names the N-match ones a smoke run built."""
    def read(name: str) -> pd.DataFrame:
        return pd.read_parquet(der / name.replace(".parquet", f"{tail}.parquet"))

    deliv = read(_DELIVERY_GRAIN)
    f_shots, f_sit = _INPUTS[kind]
    return deliv[deliv["set_piece_type"] == kind], read(f_shots), read(f_sit)


# ── The two duels, one row each ───────────────────────────────────────────────────
def first_duels(deliv: pd.DataFrame, shots: pd.DataFrame) -> pd.DataFrame:
    """One row per corner with an end location and a resolved first contact: its zone, who won it,
    and its in-window chain xG."""
    d = deliv[deliv["delivery_end_x"].notna() & deliv["won_first_contact"].notna()].copy()
    xg = shots[shots["segment"].isin(IN_WINDOW)].groupby("set_piece_id")["xg"].sum()
    d["value"] = d["set_piece_id"].map(xg).fillna(0.0)
    d["zone"] = zone_of(d["delivery_end_x"], d["delivery_end_y"], d["delivery_y"])
    d["won"] = d["won_first_contact"].astype(bool)
    return d[["set_piece_id", "league", "match_id", "attacking_team", "zone", "won", "value"]]


def second_duels(sit: pd.DataFrame, deliv: pd.DataFrame) -> pd.DataFrame:
    """One row per cleared corner with a drop and a labelled next touch: the drop zone, who took
    the next touch, the after-clearance xG s08 carries and, as `value_aft`, the corner's
    after-first-touch xG the sheet names: both, to see how far the reading moves."""
    d = sit[sit["drop_x"].notna() & sit["won_second_ball"].notna()].merge(
        deliv[["set_piece_id", "league", "attacking_team", "delivery_y", "xg_after_first_touch"]],
        on="set_piece_id", how="inner", validate="1:1")
    d["zone"] = zone_of(d["drop_x"], d["drop_y"], d["delivery_y"])
    d["won"] = d["won_second_ball"].astype(bool)
    d["value"] = d["value_xg_after"].astype(float)
    d["value_aft"] = d["xg_after_first_touch"].astype(float)
    return d[["set_piece_id", "league", "match_id", "attacking_team", "zone", "won", "value",
              "value_aft"]]


def _next_touch_obv(mid: int, kind: str, want: set) -> list:
    """(set_piece_id, OBV of the touch that contests the loose ball, signed for the attack) for one
    match's wanted deliveries: s08's own next touch, so OBV reads the contest the prices read. A
    touch the defence makes is worth to the attack minus what it is worth to the defence, and a
    touch without OBV stays missing."""
    ds = DropSituation(L.load_events(mid), L.load_360(mid), kinds=(kind,))
    out = []
    for _, c in ds.g.deliveries().iterrows():
        if c["id"] not in want:
            continue
        _, nxt, att = ds._clear_and_next(c)
        if nxt is None:
            continue
        obv = nxt.get("obv_total_net")
        v = float(obv) if obv is not None and pd.notna(obv) else np.nan
        out.append((c["id"], v if nxt["team.name"] == att else -v))
    return out


# ── Prices on match-resampled draws ───────────────────────────────────────────────
def match_sums(d: pd.DataFrame, zones: tuple) -> tuple[np.ndarray, np.ndarray]:
    """(matches, zones, 2 states, 2 sums): duels and their summed value, state 0 lost, 1 won."""
    mids = np.sort(d["match_id"].unique())
    g = d.groupby(["match_id", "zone", "won"])["value"].agg(["size", "sum"])
    idx = pd.MultiIndex.from_product([mids, zones, [False, True]], names=["match_id", "zone", "won"])
    return g.reindex(idx, fill_value=0).to_numpy(float).reshape(len(mids), len(zones), 2, 2), mids


def prices(tot: np.ndarray) -> dict:
    """Per zone from summed duels: share of the duels, value won and lost, the price, the what-if
    per 100 duels; `tot` is (..., zones, 2, 2)."""
    n_lost, n_won = tot[..., 0, 0], tot[..., 1, 0]
    with np.errstate(invalid="ignore", divide="ignore"):
        v_won, v_lost = tot[..., 1, 1] / n_won, tot[..., 0, 1] / n_lost
        share = (n_won + n_lost) / (n_won + n_lost).sum(axis=-1, keepdims=True)
    price = v_won - v_lost
    return {"share": share, "v_won": v_won, "v_lost": v_lost, "price": price,
            "whatif_per100": 100 * share * LIFT * price, "n_won": n_won, "n_lost": n_lost}


def _pct(x: np.ndarray) -> tuple[float, float]:
    """The 95% percentile interval of the finite draws, NaN where none is."""
    x = x[np.isfinite(x)]
    return tuple(np.percentile(x, [2.5, 97.5])) if len(x) else (np.nan, np.nan)


def weights(n: int, n_boot: int, seed: list) -> np.ndarray:
    """(n_boot, n) multinomial match weights: each draw resamples the matches with replacement."""
    return np.random.default_rng(seed).multinomial(n, np.full(n, 1.0 / n), size=n_boot).astype(float)


def price_table(d: pd.DataFrame, zones: tuple, n_boot: int, seed: list, sample: str,
                duel: str) -> tuple[pd.DataFrame, np.ndarray]:
    """Every zone's figures with match-resampled 95% intervals, and the (n_boot, zones) price draws."""
    arr, mids = match_sums(d, zones)
    point = prices(arr.sum(axis=0))
    boot = prices(np.einsum("bm,mzsk->bzsk", weights(len(mids), n_boot, seed), arr))
    per_team = len(d) / d["attacking_team"].nunique()
    rows = []
    for k, z in enumerate(zones):
        row = {"sample": sample, "duel": duel, "zone": z, "n_won": int(point["n_won"][k]),
               "n_lost": int(point["n_lost"][k]),
               "thin": bool(min(point["n_won"][k], point["n_lost"][k]) < MIN_CELL)}
        for name in ("share", "v_won", "v_lost", "price", "whatif_per100"):
            lo, hi = _pct(boot[name][:, k])
            row.update({name: float(point[name][k]), f"{name}_lo": float(lo), f"{name}_hi": float(hi)})
        row["whatif_xg_per_team_season"] = row["whatif_per100"] * per_team / 100
        rows.append(row)
    return pd.DataFrame(rows), boot["price"]


# ── C1a ───────────────────────────────────────────────────────────────────────────
def spearman_rows(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Spearman of two (draws, zones) arrays, draw by draw, over the zones both carry finite."""
    out = np.full(len(a), np.nan)
    for i in range(len(a)):
        ok = np.isfinite(a[i]) & np.isfinite(b[i])
        if ok.sum() > 2:
            out[i] = stats.spearmanr(a[i, ok], b[i, ok])[0]
    return out


def c1a(dev: pd.DataFrame, conf: pd.DataFrame, zones: tuple, n_boot: int, seed: int) -> dict:
    """C1a at one seed: the top zone of each sample, the Spearman of the eligible zones' prices with
    its interval over independent match draws of the two samples, and the top zone's support.
    Under three eligible zones a Spearman says nothing, so the verdict is None and not False."""
    dev_t, dev_b = price_table(dev, zones, n_boot, [seed, 1], "development", "first")
    conf_t, conf_b = price_table(conf, zones, n_boot, [seed, 2], "confirmation", "first")
    keep = ~(dev_t["thin"].to_numpy() | conf_t["thin"].to_numpy())
    zs = np.asarray(zones, dtype=object)[keep]
    p_dev, p_conf = dev_t["price"].to_numpy()[keep], conf_t["price"].to_numpy()[keep]
    rho = spearman_rows(dev_b[:, keep], conf_b[:, keep])
    lo, hi = _pct(rho)
    top_dev = zs[np.nanargmax(p_dev)] if len(zs) else None
    top_conf = zs[np.nanargmax(p_conf)] if len(zs) else None
    support = (float(np.mean([zs[np.nanargmax(r)] == top_dev for r in conf_b[:, keep]]))
               if len(zs) else np.nan)
    defined = len(zs) >= 3
    rho_point = float(stats.spearmanr(p_dev, p_conf)[0]) if defined else np.nan
    return {"seed": seed, "zones_used": ", ".join(zs), "zones_thin": ", ".join(
                np.asarray(zones, dtype=object)[~keep]), "top_development": top_dev,
            "top_confirmation": top_conf, "top_support": support, "spearman": rho_point,
            "spearman_lo": float(lo), "spearman_hi": float(hi),
            "holds": (bool(top_dev == top_conf and rho_point >= SPEARMAN_FLOOR and lo > 0)
                      if defined else None)}


def c1a_decision(seeds: pd.DataFrame) -> dict:
    """C1a over the seeds: True only if every seed holds, None if any seed is undecided."""
    h = seeds["holds"]
    return {"C1a": None if h.isna().any() else bool(h.all()),
            "C1a_any_seed": bool(h.eq(True).any())}


def samples(leagues: list) -> dict:
    """Every league first, the project's scope; then C1a's development and confirmation samples;
    then each league alone."""
    return {"all": lambda d: d,
            "development": lambda d: d[d["league"].isin(DEVELOPMENT)],
            "confirmation": lambda d: d[~d["league"].isin(DEVELOPMENT)],
            **{lg: (lambda d, lg=lg: d[d["league"] == lg]) for lg in leagues}}


# ── Reported beside the list ──────────────────────────────────────────────────────
def readings_agree(table: pd.DataFrame) -> dict:
    """How far the second duel moves between its two values on every league: the Spearman of the
    zones' prices and the top zone of each, over the zones neither leaves thin. As in C1a, no zone
    leaves no top zone and under three a Spearman says nothing."""
    a, b = (table[(table["sample"] == "all") & (table["duel"] == d)].set_index("zone")
            for d in ("second", "second_aft"))
    ok = ~(a["thin"] | b["thin"])
    n = int(ok.sum())
    return {"second_spearman": (float(stats.spearmanr(a.loc[ok, "price"], b.loc[ok, "price"])[0])
                                if n >= 3 else np.nan),
            "second_top_after_clearance": a.loc[ok, "price"].idxmax() if n else None,
            "second_top_after_first_touch": b.loc[ok, "price"].idxmax() if n else None}


def obv_consistency(second: pd.DataFrame, zones: tuple, n_boot: int) -> tuple[dict, pd.DataFrame]:
    """OBV's price of the second ball per zone beside ours, on the corners whose contesting touch
    carries an OBV, and the Spearman of the two over draws both share (same rows and seed, so the
    same match weights). Consistency only, because OBV's window already holds the second ball. The
    coverage is split by who won, since a touch type without OBV can belong to one side."""
    has = np.isfinite(second["obv"].to_numpy(float))
    d = second[has]
    split = {f"coverage_{w}": float(has[(second["won"] == (w == "won")).to_numpy()].mean())
             for w in ("won", "lost")}
    ours, ours_b = price_table(d, zones, n_boot, [0, 4], "all", "second")
    theirs, theirs_b = price_table(d.assign(value=d["obv"]), zones, n_boot, [0, 4], "all",
                                   "second_obv")
    keep = ~ours["thin"].to_numpy()
    lo, hi = _pct(spearman_rows(ours_b[:, keep], theirs_b[:, keep]))
    rho = float(stats.spearmanr(ours["price"].to_numpy()[keep], theirs["price"].to_numpy()[keep])[0])
    return {"coverage": len(d) / len(second), **split, "spearman": rho, "spearman_lo": float(lo),
            "spearman_hi": float(hi)}, theirs


# ── The run ───────────────────────────────────────────────────────────────────────
def main() -> None:
    """For one population: both duels on every league and on C1a's two samples, C1a at every seed,
    each league, the second duel's two values, OBV beside them, the transitions."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--tables", type=str, default=str(_DER),
                    help="where the population's delivery grain, shots and situation live")
    ap.add_argument("--kind", choices=tuple(_INPUTS), default="corner")
    ap.add_argument("--jobs", type=int, default=8, help="workers reading the events for OBV")
    ap.add_argument("--boot", type=int, default=_PRICE_DRAWS)
    ap.add_argument("--seeds", type=int, default=_SEEDS)
    ap.add_argument("--out-dir", type=str, default=None,
                    help="where to write (default: data/derived, only at the registered precision)")
    ap.add_argument("--limit", type=int, default=None,
                    help="read the N-match tables a smoke run built: the path, not the numbers")
    args = ap.parse_args()
    out = Path(args.out_dir) if args.out_dir else _DER
    tail = f"_limit{args.limit}" if args.limit else ""
    if published_dir(out) and (args.boot, args.seeds) != (_PRICE_DRAWS, _SEEDS):
        raise SystemExit(f"STOP: the published duel_prices records come only at (boot, seeds) = "
                         f"{(_PRICE_DRAWS, _SEEDS)}; pass --out-dir")
    if published_dir(out) and Path(args.tables).resolve() != _DER.resolve():
        raise SystemExit("STOP: tables outside data/derived are not the published ones, so their "
                         "prices may not take the published names; pass --out-dir")
    if published_dir(out) and tail:
        raise SystemExit("STOP: a --limit run reads the N-match tables a smoke built, so its prices "
                         "may not take the published names; pass --out-dir")

    deliv, shots, sit = inputs(Path(args.tables), args.kind, tail)
    zones = zone_names()
    first, second = first_duels(deliv, shots), second_duels(sit, deliv)
    print(f"{args.kind}, zones {ZONE_PARTITION}: first duels {len(first)} over "
          f"{first['match_id'].nunique()} matches, second duels {len(second)} over "
          f"{second['match_id'].nunique()}", flush=True)

    tables = []
    duels = (("first", first), ("second", second),
             ("second_aft", second.assign(value=second["value_aft"])))
    for name, pick in samples(sorted(first["league"].unique())).items():
        for duel, d in duels:
            tables.append(price_table(pick(d), zones, args.boot, [0, 3], name, duel)[0])
    obv = dict(row for rows in Parallel(n_jobs=args.jobs)(
        delayed(_next_touch_obv)(int(m), args.kind, set(ids))
        for m, ids in second.groupby("match_id")["set_piece_id"]) for row in rows)
    second["obv"] = second["set_piece_id"].map(obv).astype(float)
    obv_rec, obv_table = obv_consistency(second, zones, args.boot)
    table = pd.concat([*tables, obv_table], ignore_index=True)

    dev, conf = first[first["league"].isin(DEVELOPMENT)], first[~first["league"].isin(DEVELOPMENT)]
    seeds = pd.DataFrame([c1a(dev, conf, zones, args.boot, s) for s in range(args.seeds)])
    league_out = pd.DataFrame([{"left_out": lg, **c1a(dev, conf[conf["league"] != lg], zones,
                                                      args.boot, 0)}
                               for lg in sorted(conf["league"].unique())])
    decision = {"population": args.kind, **c1a_decision(seeds), "seeds": len(seeds),
                "zones": ZONE_PARTITION, **readings_agree(table),
                **{f"obv_{k}": v for k, v in obv_rec.items()}}
    both = first.merge(second[["set_piece_id", "zone"]], on="set_piece_id", suffixes=("", "_drop"))
    trans = pd.crosstab(both["zone"], both["zone_drop"])

    pd.set_option("display.width", 260)
    pd.set_option("display.max_columns", 40)
    view = ["sample", "duel", "zone", "n_won", "n_lost", "thin", "v_won", "v_lost", "price",
            "price_lo", "price_hi", "whatif_xg_per_team_season"]
    for s in ("all", "development", "confirmation"):
        print(f"\nPRICES, {s}\n" + table.loc[table["sample"] == s, view].round(4).to_string(index=False))
    print("\nC1a by seed\n" + seeds.round(4).to_string(index=False))
    print("\nC1a, each confirmation league left out (seed 0)\n" + league_out.round(4).to_string(index=False))
    print(f"\nTRANSITIONS, first-duel zone to drop zone ({len(both)} corners carrying both)\n"
          + trans.to_string())
    print("\nDECISION AND THE READINGS BESIDE IT\n"
          + "\n".join(f"  {k}: {v}" for k, v in decision.items()), flush=True)

    out.mkdir(parents=True, exist_ok=True)
    tag = f"duel_prices_{ZONE_PARTITION}{'' if args.kind == 'corner' else f'_{args.kind}'}{tail}"
    table.to_parquet(out / f"{tag}.parquet", index=False)
    seeds.to_parquet(out / f"{tag}_c1a.parquet", index=False)
    league_out.to_parquet(out / f"{tag}_league_out.parquet", index=False)
    trans.to_parquet(out / f"{tag}_transitions.parquet")
    pd.DataFrame([decision]).to_parquet(out / f"{tag}_decision.parquet", index=False)
    print(f"\nsaved the duel_prices records in {out}")


if __name__ == "__main__":
    main()
