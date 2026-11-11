"""Tests for run_positive_control: the folds, the shuffle, and the arms it compares.

A control only means something if it could have failed. What these pin is that the three arms are
nested as declared, that the out-of-fold scores are out of fold, and that the shuffle destroys what
it claims to destroy and nothing else.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

_REPO = Path(__file__).resolve().parents[1]
for p in (_REPO / "src", _REPO / "pipeline"):
    sys.path.insert(0, str(p))

from s03_value import CompetingOutcomesModel, _KICK_INTRINSIC, _NUMERIC     # noqa: E402
import run_positive_control as pc                                    # noqa: E402


def _toy(n: int = 120, n_matches: int = 12, seed: int = 0) -> tuple:
    rng = np.random.default_rng(seed)
    g = np.repeat(np.arange(n_matches), n // n_matches)
    x = rng.normal(size=n)
    y = (x + rng.normal(scale=0.5, size=n) > 0).astype(int)
    return np.c_[x, rng.normal(size=n)], y, g


def test_the_three_arms_are_nested_as_declared():
    """prior is intercept-only, delivery is the kick block, full_360 is the whole numeric block.

    The ladder only reads as `what the next block bought` if each arm contains the one before it,
    which is the rule the repo states for every ladder it publishes. The prior also has to refuse
    the delivery categoricals: `_design` attaches them whenever the flag is on, whatever the
    numeric list says."""
    assert pc.ARMS["prior"] == ([], False)
    assert set(pc.ARMS["delivery"][0]) == set(_KICK_INTRINSIC) and pc.ARMS["delivery"][1]
    assert pc.ARMS["full_360"] == (None, True)
    assert set(_KICK_INTRINSIC) <= set(_NUMERIC)


def test_the_prior_arm_carries_no_feature_at_all():
    """The one test that catches the ladder collapsing. `_design(numeric=[])` still attaches the
    delivery dummies and `short_corner`, so a prior built that way arrives carrying the strongest
    feature in this target and every arm ties with it. A real prior scores the marginal entropy,
    and a design with one live column beats that."""
    tab = pd.DataFrame({"short_corner": [True] * 30 + [False] * 30,
                        "corner_side": ["left"] * 30 + ["right"] * 30,
                        "match_id": np.repeat(np.arange(12), 5)})
    m = CompetingOutcomesModel(n_splits=5)
    numeric, cats = pc.ARMS["prior"]
    X, cols = m._design(tab, numeric=numeric, with_delivery_cats=cats)
    assert X.shape[1] == 1 and np.ptp(X) == 0, f"the prior carries {cols}"
    wide, _ = m._design(tab, numeric=[], with_delivery_cats=True)
    assert wide.shape[1] > 1, "guard is pointless if the flag changes nothing"


def test_pick_c_skips_an_inner_fold_that_holds_one_class():
    """A training block with one class cannot fit, and scoring it as a loss would punish
    whichever C happened to meet it rather than the C that is worse."""
    X, y, g = _toy()
    y[:] = 0
    y[g == 11] = 1
    assert pc._pick_c(X, y, g) in pc.CONTROL_C_GRID


def test_oof_proba_scores_every_row_out_of_fold(monkeypatch):
    """Each row is scored by a model that never saw its match, and every row is scored once."""
    X, y, g = _toy()
    seen = []
    original = pc._fit

    def spy(Xtr, ytr, C):
        seen.append(len(ytr))
        return original(Xtr, ytr, C)

    monkeypatch.setattr(pc, "_fit", spy)
    p, picked = pc.oof_proba(X, y, g)
    assert len(picked) == 5 and np.isfinite(p).all() and ((p > 0) & (p < 1)).all()
    assert max(seen) < len(y)


def test_missing_values_are_imputed_inside_the_fold_rather_than_crashing():
    """`_design` hands over the NaN on purpose, because the published model imputes per fold and a
    table-wide median would leak. A driver that forwards that matrix straight to scikit-learn dies
    on `Input X contains NaN`, and it dies only where a column is actually absent, so a smoke over
    a few matches can miss it entirely. Every row must still be scored."""
    X, y, g = _toy()
    X[::7, 1] = np.nan
    p, picked = pc.oof_proba(X, y, g)
    assert np.isfinite(p).all() and ((p > 0) & (p < 1)).all() and len(picked) == 5


def test_an_all_nan_column_in_train_does_not_poison_the_fold():
    """s03's preparation sends a column that is all-NaN in the training block to zero. Without
    that, one absent column would make every score in the fold NaN."""
    X, y, g = _toy()
    X[:, 1] = np.nan
    p, _ = pc.oof_proba(X, y, g)
    assert np.isfinite(p).all()


def test_the_shuffle_keeps_each_match_rate_and_breaks_the_row_level_link():
    """Every match keeps its own rate, which is why the control's baseline has to be the prior and
    not the marginal entropy: rates that differ between matches beat a global entropy on their own.
    The row-level link must still be gone, or the shuffle controls nothing."""
    rng = np.random.default_rng(2)
    n, k = 240, 12
    g = np.repeat(np.arange(k), n // k)
    y = (rng.random(n) < np.repeat(np.linspace(0.1, 0.9, k), n // k)).astype(int)
    ys = pc.shuffle_labels(y, g, mode="within_match", seed=0)
    assert (pd.Series(y).groupby(g).mean() == pd.Series(ys).groupby(g).mean()).all()
    assert not np.array_equal(y, ys)
    glob = pc.shuffle_labels(y, g, mode="global", seed=0)
    assert glob.sum() == y.sum()
    assert not (pd.Series(y).groupby(g).mean() == pd.Series(glob).groupby(g).mean()).all()


def test_another_input_never_lands_on_the_published_record(monkeypatch, tmp_path):
    """positive_control.parquet is the record evidence.md §19 is bound to, written off the grant's
    corner table and delivery grain. Any other pair tags the name, as --limit does. The arms are
    stubbed: what is under test is where main() writes."""
    n = 40
    tab = pd.DataFrame({"set_piece_id": [f"c{i}" for i in range(n)],
                        "match_id": np.repeat(np.arange(8), 5), "short_corner": [False] * n})
    dlv = pd.DataFrame({"set_piece_id": tab["set_piece_id"], "set_piece_type": "corner",
                        "won_first_contact": [True, False] * (n // 2)})
    tab.to_parquet(tmp_path / "corners_subset.parquet")
    dlv.to_parquet(tmp_path / "eda_deliveries_subset.parquet")
    der = tmp_path / "derived"
    der.mkdir()

    def losses(table, y, g):
        return {k: np.zeros(len(y)) for k in pc.ARMS}, {k: [] for k in pc.ARMS}

    monkeypatch.setattr(pc, "_DER", der)
    monkeypatch.setattr(pc, "arm_losses", losses)
    monkeypatch.setattr(pc, "shuffled_arm",
                        lambda *a, **k: {"delta": 0.0, "lo": -0.1, "hi": 0.1, "clean": True})
    monkeypatch.setattr(sys, "argv", ["run_positive_control.py", "--boot", "20",
                                      "--table", str(tmp_path / "corners_subset.parquet"),
                                      "--deliveries", str(tmp_path / "eda_deliveries_subset.parquet")])
    pc.main()
    assert not (der / "positive_control.parquet").exists()
    assert len(list(der.glob("positive_control_*.parquet"))) == 1


@pytest.mark.parametrize("extra", [
    ["--boot", "200"], ["--boot", "200", "--kind", "free_kick_flank"],
    ["--out", str(_REPO / "data" / "derived" / "zones_F" / "positive_control.parquet"),
     "--table", "elsewhere/corners.parquet"]])
def test_a_published_name_needs_the_grant_at_the_registered_precision(monkeypatch, extra):
    """Either population's record, in data/derived or in zones_F where the pass under partition F
    publishes it, comes only off the grant's tables at the registered draws, and a run that is not
    one refuses before reading anything."""
    def work(*a, **k):
        raise AssertionError("the run read the data before refusing")

    monkeypatch.setattr(pc.pd, "read_parquet", work)
    monkeypatch.setattr(sys, "argv", ["run_positive_control.py", *extra])
    with pytest.raises(SystemExit, match="may not write the published"):
        pc.main()


def test_the_flank_twin_reads_its_rows_of_the_combined_grain(monkeypatch, tmp_path):
    """--kind free_kick_flank merges the flank table with that population's rows of the grain both
    populations share, tags the name with it, and skips the long-corner reading: a free kick is never
    short (definitions.md §5)."""
    n = 40
    tab = pd.DataFrame({"set_piece_id": [f"f{i}" for i in range(n)],
                        "set_piece_type": "free_kick_flank", "match_id": np.repeat(np.arange(8), 5),
                        "short_corner": [False] * n})
    grain = pd.concat([pd.DataFrame({"set_piece_id": tab["set_piece_id"],
                                     "set_piece_type": "free_kick_flank",
                                     "won_first_contact": [True, False] * (n // 2)}),
                       pd.DataFrame({"set_piece_id": ["c0"], "set_piece_type": "corner",
                                     "won_first_contact": [True]})])
    tab.to_parquet(tmp_path / "flank.parquet")
    grain.to_parquet(tmp_path / "grain.parquet")
    der = tmp_path / "derived"
    der.mkdir()
    seen = []

    def losses(table, y, g):
        seen.append(len(table))
        return {k: np.zeros(len(y)) for k in pc.ARMS}, {k: [] for k in pc.ARMS}

    monkeypatch.setattr(pc, "_DER", der)
    monkeypatch.setattr(pc, "arm_losses", losses)
    monkeypatch.setattr(pc, "shuffled_arm",
                        lambda *a, **k: {"delta": 0.0, "lo": -0.1, "hi": 0.1, "clean": True})
    monkeypatch.setattr(sys, "argv", ["run_positive_control.py", "--boot", "20", "--kind",
                                      "free_kick_flank", "--table", str(tmp_path / "flank.parquet"),
                                      "--deliveries", str(tmp_path / "grain.parquet")])
    pc.main()
    assert seen == [n]
    out, = der.glob("positive_control_free_kick_flank_*.parquet")
    assert "delta_long_only" not in set(pd.read_parquet(out)["block"])


def test_verdict_needs_both_the_interval_and_a_clean_control():
    """P1 and P2 are conjunctions: an interval below zero on a harness that leaks decides nothing."""
    deltas = {"full_360-prior": {"below_zero": True},
              "full_360-delivery": {"below_zero": False}}
    assert pc.decide(deltas, clean=True) == {"P1": "true", "P2": "false"}
    assert pc.decide(deltas, clean=False) == {"P1": "false", "P2": "false"}
    deltas["full_360-delivery"]["below_zero"] = True
    assert pc.decide(deltas, clean=True) == {"P1": "true", "P2": "true"}
