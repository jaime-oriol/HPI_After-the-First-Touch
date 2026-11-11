"""Tests for viz.common — the shared house style, and the legend counts beside it.

`corner_stats` is a legend count, not a modelling predicate: it reads `play_pattern.name`,
which spans the whole possession the corner started rather than the delivery s01 segments.
"""
from viz.common import corner_stats
from conftest import requires_sb


@requires_sb
def test_corner_stats(sb_events):
    """corner_stats returns ordered non-negative counts with goals a subset of shots."""
    stats = corner_stats(sb_events, "AC Milan")
    kinds = {s[0]: s[1] for s in stats}
    assert [s[0] for s in stats] == ["corner", "shot", "goal"]
    assert all(isinstance(v, int) and v >= 0 for v in kinds.values())
    assert kinds["goal"] <= kinds["shot"]
