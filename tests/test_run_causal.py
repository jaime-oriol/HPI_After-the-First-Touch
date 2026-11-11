"""Tests for run_causal: the plan runs the registered primary and the curve around it."""
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]
for p in (_REPO / "src", _REPO / "pipeline"):
    sys.path.insert(0, str(p))

import run_causal as R                                                # noqa: E402
from s03_value import FEATURE_GROUPS                                  # noqa: E402
from s05_causal import TREATMENT_GROUP                                # noqa: E402


def test_the_plan_runs_the_registered_primary_and_the_curve_around_it():
    """The primary is the protector sheet's (data/derived/prereg_2026-10-01/protector.md): every
    group but the treatment's, `box_depth` and `counter_lane`, with team and league controls and
    once more without them. The curve leaves one group out of the full design, and no
    specification keeps the treatment's group."""
    plan = R._plan("after_first_touch")
    primary = {TREATMENT_GROUP, "box_depth", "counter_lane"}
    assert [(p[1], set(p[2])) for p in plan if p[0] == "headline"] == [(R.PRIMARY_SPEC, primary)]
    assert {p[0] for p in plan if p[1] == R.PRIMARY_SPEC} == {"headline", "contrast", "degenerate"}
    sens = [p for p in plan if p[0] == "sensitivity"]
    assert len(sens) == 1 and set(sens[0][2]) == primary and sens[0][5] == ()
    curve = {p[1]: set(p[2]) for p in plan if p[0] == "spec_curve"}
    assert curve == {"full": {TREATMENT_GROUP},
                     **{f"drop_{g}": {TREATMENT_GROUP, g} for g in FEATURE_GROUPS
                        if g != TREATMENT_GROUP}}
    assert all(TREATMENT_GROUP in p[2] for p in plan)
