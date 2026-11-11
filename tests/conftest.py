"""Shared fixtures. Tests run against the sample under data/ (gitignored);
they skip cleanly when it isn't present so the suite never errors on a bare clone."""
import pathlib
import sys
from collections.abc import Iterator

import numpy as np
import pandas as pd
import pytest

# Put src/ on the import path so tests can `import loader` without installing the package.
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

import loader          # Hudl Broadcast X&Y tracking loader (Serie A 30-match sample)
import loader_sb       # Hudl StatsBomb loader (events + 360, sample + Big-5)

# ── Tracking sample gate ──
# True only if at least one tracking file exists in the local Serie A sample.
_HAS_DATA = any(loader.DATA_DIR.glob("*.jsonl"))
# Marker to hang on tests that need the tracking sample; skips (not errors) when it's absent.
requires_data = pytest.mark.skipif(not _HAS_DATA, reason="tracking sample not present")

# ── StatsBomb sample gate ──
# Granted events+360 substrate: gated on the Serie A sample being on disk.
_HAS_SB = any((loader_sb.DATA_DIR / "serie_a_sample" / "events_360").glob("*.json"))
# Marker for tests that need the granted data; skips when it isn't present.
requires_sb = pytest.mark.skipif(not _HAS_SB, reason="granted Hudl sample not present")


def counted_as_s02_counts(t: pd.DataFrame, seed: int = 1) -> pd.DataFrame:
    """A synthetic design table given the counts s02 writes: integer zone and frontal counts, the
    body total as their sum and each per-body form a kernel divided by it. The total minus both
    zone blocks is then the frontal count, the identity s03 and s05 must not carry into a design."""
    from s03_value import FEATURE_GROUPS
    rng = np.random.default_rng(seed)
    t = t.copy()
    counts = FEATURE_GROUPS["zones_att"] + FEATURE_GROUPS["zones_def"] + ["n_def_frontal"]
    for c in counts:
        t[c] = rng.integers(0, 3, len(t)).astype(float)
    n = t[counts].sum(1)
    assert n.min() >= 2, "the per-pair form needs two bodies on every row"
    t["congestion_bodies"] = n
    for raw in ("congestion_spot", "congestion_drop", "congestion_pairwise"):
        t[raw] = rng.gamma(2.0, 1.0, len(t))
    t["congestion_spot_per_body"] = t["congestion_spot"] / n
    t["congestion_drop_per_body"] = t["congestion_drop"] / n
    t["congestion_pairwise_per_pair"] = t["congestion_pairwise"] / (n * (n - 1) / 2)
    return t


class SameRowsSplit:
    """Splitter handing back the same rows as train and test, so every match straddles it: a split
    path that skipped `assert_no_group_leak` would fit happily and return a number."""

    def __init__(self, n_splits: int = 5) -> None:
        self.n_splits = n_splits

    def split(self, X: np.ndarray | pd.DataFrame, y: np.ndarray | None = None,
              groups: np.ndarray | None = None) -> Iterator[tuple[np.ndarray, np.ndarray]]:
        idx = np.arange(len(X))
        yield idx, idx


@pytest.fixture(scope="session")
def mid():
    """A sample match id with both events+360 and tracking (Juventus vs Inter Milan)."""
    return 4004219                  # StatsBomb match id used across the s04 race / render tests


@pytest.fixture(scope="session")
def sb_mid():
    """A Serie A sample match id (events+360 only, no tracking needed)."""
    return 4004239                  # AC Milan vs Napoli, backing the requires_sb-only tests


@pytest.fixture(scope="session")
def flank_mids():
    """Serie A sample matches carrying flank free kicks. sb_mid carries none, so a parallel build
    over it alone has nothing to reorder."""
    return [4004288, 4004341, 4004289, 4004458, 4004495]


@pytest.fixture(scope="session")
def sb_events(sb_mid):
    """Event stream for the sample match."""
    return loader_sb.load_events(sb_mid)   # served from the per-match parquet cache, built on first use


@pytest.fixture(scope="session")
def repo_root():
    """Repo root, for smoke tests that shell out to the drivers under pipeline/ and renders/."""
    return pathlib.Path(__file__).resolve().parents[1]
