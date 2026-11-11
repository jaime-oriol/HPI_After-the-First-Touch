"""Tests for audit_360: the occupancy it reports reads the box as s02's zones read it."""
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]
for p in (_REPO / "src", _REPO / "pipeline"):
    sys.path.insert(0, str(p))

import audit_360 as A                                                 # noqa: E402


def test_the_box_count_reads_the_width_of_the_box():
    """A body by the flag at x = 110 is past the box line but not in the box: counted by depth
    alone, the audit would report box occupants the zones never see."""
    ff = [{"location": [110.0, 40.0]}, {"location": [110.0, 10.0]}, {"location": [50.0, 40.0]},
          {"location": [99.0, 40.0]}]
    assert A._zone_counts(ff) == (1, 1)
