"""Tests for render_signature_sb: the ringed defenders are the ones s05 calls treated."""
import os
import subprocess
import sys
from pathlib import Path

import pytest

_REPO = Path(__file__).resolve().parents[1]
for p in (_REPO / "src", _REPO / "renders"):
    sys.path.insert(0, str(p))

import render_signature_sb as R                                       # noqa: E402
from conftest import requires_sb                                      # noqa: E402


@pytest.mark.parametrize("partition, ringed", [("C", "[(99.0, 40.0)]"),
                                               ("G2", "[(99.0, 40.0), (104.0, 40.0)]")])
def test_the_ringed_defenders_are_the_frontal_ones(partition, ringed):
    """The figure rings the treatment s05 estimates, the outfield defenders in the switch's frontal:
    under C not one inside the box, which the central edge band would also count, and under G2 also
    the one in its `edge`, the central block of the box. The switch is read at import."""
    code = (f"import sys\nsys.path[:0] = [{str(_REPO / 'src')!r}, {str(_REPO / 'renders')!r}]\n"
            "import pandas as pd, render_signature_sb as R\n"
            "frame = pd.DataFrame({'x': [99.0, 104.0, 99.0, 99.0, 120.0], 'y': [40.0, 40.0, 10.0, 40.0, 0.1],\n"
            "                      'team_id': [0, 0, 0, 0, 1], 'is_goalkeeper': [0, 0, 0, 1, 0],\n"
            "                      'is_ball': [0, 0, 0, 0, 1]})\n"
            "print(R._edge_protectors(frame))")
    env = {k: v for k, v in os.environ.items() if k != "HUDL_ZONES"} | {"HUDL_ZONES": partition}
    out = subprocess.run([sys.executable, "-c", code], env=env, capture_output=True, text=True)
    assert out.returncode == 0, out.stderr
    assert out.stdout.strip() == ringed


@requires_sb
def test_the_ring_count_is_the_table_s_treatment_on_real_corners(sb_mid):
    import loader_sb as L
    from s01_phases import CornerPhaseGrammar
    from s02_signature import CommitmentSignature
    fz = L.load_360(sb_mid)
    corners = CornerPhaseGrammar(L.load_events(sb_mid)).corners()
    corners = corners[corners["id"].isin(set(fz["event_uuid"]))]
    assert len(corners)
    for _, c in corners.iterrows():
        cs = CommitmentSignature.from_corner(c, fz)
        assert len(R._edge_protectors(cs.frame)) == cs.features()["n_def_frontal"]
