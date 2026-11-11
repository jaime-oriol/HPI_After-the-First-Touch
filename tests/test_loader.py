"""Tests for loader — access to the Hudl Broadcast X&Y tracking sample under data/."""
import hashlib
import json

import numpy as np
import pandas as pd
import pytest

import loader
from conftest import requires_data

_HOME, _GK = 7, {"7_1": True}                        # home team id, and jersey 1 is its keeper


def _keeper_frames(per_period):
    """Frames carrying only the home keeper, at the given x values per period."""
    return [{"period": p, "homePlayers": [{"jerseyNum": 1, "x": x, "y": 0.0}]}
            for p, xs in per_period.items() for x in xs]


def _directions(frames, home_id=_HOME, gk=_GK):
    """Compose the two halves exactly as load_tracking composes them while streaming:
    accumulate the keeper's x per period, then apply the rule."""
    xs = {}
    for fr in frames:
        xs.setdefault(fr["period"], []).extend(loader._keeper_x(fr, home_id, gk))
    return loader._directions_from_keeper_x(xs)


# ── Direction of play: read the keeper, never the kickoff shape ─────────
def test_direction_of_play_reads_the_keeper():
    """The keeper's side of the pitch decides the direction, and it flips at half time.

    The leading empty frames are the real shape of the data: the broadcast clock starts
    before kick-off, so period 1 opens on frames with no detected player at all."""
    frames = [{"period": 1, "homePlayers": [{}]} for _ in range(5)]
    frames += _keeper_frames({1: [-40.0, -41.0], 2: [39.0, 42.0]})
    d = _directions(frames)
    assert d[1] == "Left to Right"                   # keeper on the left -> home defends the left goal
    assert d[2] == "Right to Left"                   # ends swapped at half time


def test_direction_of_play_ignores_the_outfield_shape():
    """Outfield positions do not decide it. Reading them is the kickoff rule, which the
    last test in this file measures as a negative."""
    frames = [{"period": 1, "homePlayers": [
        {"jerseyNum": 1, "x": -40.0, "y": 0.0},      # the keeper, on the left
        {"jerseyNum": 9, "x": 45.0, "y": 0.0},       # outfielders massed on the right
        {"jerseyNum": 10, "x": 48.0, "y": 0.0},
    ]}]
    assert _directions(frames)[1] == "Left to Right"


def test_direction_of_play_raises_when_the_keeper_is_never_seen():
    """No keeper detection means UNKNOWN, which is not the same answer as right-to-left,
    so it raises rather than answering on no evidence as the kickoff rule does."""
    frames = [{"period": 1, "homePlayers": [{"jerseyNum": 9, "x": -30.0, "y": 0.0}]}]
    with pytest.raises(ValueError, match="unknown"):
        _directions(frames)


# ── The scale conversion the renders share ──
def test_meters_to_opta_pins_the_pitch_corners():
    """Centre and both corners, so a redefined pitch cannot pass silently."""
    assert loader.meters_to_opta(0.0, 0.0) == (50.0, 50.0)
    assert loader.meters_to_opta(-loader.PITCH_LENGTH_M / 2, -loader.PITCH_WIDTH_M / 2) == (0.0, 0.0)
    assert loader.meters_to_opta(loader.PITCH_LENGTH_M / 2, loader.PITCH_WIDTH_M / 2) == (100.0, 100.0)


def test_meters_to_opta_takes_arrays():
    """render_race.py feeds it the whole reach-field grid, not a scalar."""
    ox, oy = loader.meters_to_opta(np.array([-52.5, 0.0, 52.5]), np.zeros(3))
    assert np.allclose(ox, [0.0, 50.0, 100.0])
    assert np.allclose(oy, 50.0)


# ── Orientation: the away team runs the other way ──
def _dir_frame(direction: str) -> pd.DataFrame:
    """A tracking frame carrying only what attack_runs_rtl reads."""
    return pd.DataFrame({"direction_of_play": [direction] * 3})


def test_attack_runs_rtl_inverts_for_the_away_team():
    """direction_of_play carries the HOME direction on every row, ball and away rows included,
    so reading it off the attacker's rows orients every away-team set piece backwards. The same
    two frames give opposite answers depending on who is attacking."""
    home, away = 10, 20
    rtl, ltr = _dir_frame("Right to Left"), _dir_frame("Left to Right")
    assert loader.attack_runs_rtl(rtl, home, home) is True
    assert loader.attack_runs_rtl(ltr, home, home) is False
    assert loader.attack_runs_rtl(rtl, away, home) is False
    assert loader.attack_runs_rtl(ltr, away, home) is True


def test_attack_runs_rtl_on_an_empty_frame():
    """A corner with no tracking rows must not flip the render on an empty mode()."""
    assert loader.attack_runs_rtl(_dir_frame("Right to Left").iloc[:0], 20, 10) is False


# ── The paired sample ──
@requires_data
def test_tracked_matches_is_the_paired_sample(mid):
    """run_race --all iterates exactly this, so it decides which matches the race covers."""
    mids = loader.tracked_matches()
    assert len(mids) == 30                           # the granted Serie A tracking sample
    assert mids == sorted(set(mids))
    assert all(isinstance(m, int) for m in mids)     # StatsBomb ids, not paths or numpy ints
    assert mid in mids


@requires_data
def test_load_tracking(mid):
    """Tracking loads with the columns the race/velocity code relies on."""
    df = loader.load_tracking(mid)
    assert len(df) > 0
    for col in ("frame_count", "period_id", "period_elapsed_s", "team_id", "player_id", "x", "y",
                "x_tracking", "y_tracking", "vx", "vy", "is_ball", "is_goalkeeper"):
        assert col in df.columns


@requires_data
def test_load_tracking_identity(mid):
    """team_id/is_goalkeeper are resolved (not left as raw home/away or absent)."""
    df = loader.load_tracking(mid)
    team_ids = df["team_id"].dropna().unique()
    assert len(team_ids) == 2
    gks = df.loc[df["is_goalkeeper"], "player_id"].unique()
    assert 1 <= len(gks) <= 2


# ── The frame's shape is a contract: s04, pitch_control and three renders read it ──
_SCHEMA = (
    ("frame_count", "int64"), ("period_id", "int64"),
    ("timeelapsed_from_period_start_s", "float32"), ("period_elapsed_s", "float32"),
    ("team_id", "float32"), ("player_id", "object"), ("is_ball", "int64"),
    ("is_goalkeeper", "bool"), ("direction_of_play", "object"),
    ("x_tracking", "float32"), ("y_tracking", "float32"), ("vx", "float32"),
    ("vy", "float32"), ("speed", "float32"), ("visibility", "object"),
    ("confidence", "object"), ("x", "float32"), ("y", "float32"),
)


@requires_data
def test_load_tracking_schema_and_column_order(mid):
    """Column ORDER and dtypes both matter downstream: render_corner splits player_id with
    the .str accessor, so an object column turning categorical would break it silently."""
    df = loader.load_tracking(mid)
    assert tuple(df.columns) == tuple(c for c, _ in _SCHEMA)
    assert tuple(str(t) for t in df.dtypes) == tuple(t for _, t in _SCHEMA)


# sha256 per column of match 4004219, recorded from the frame this loader returns. Values, not
# just shape: the columnar build reorders nothing, and x/y stay float64 until the final downcast,
# so recomputing them in float32 (which moves the last bit) would break these and not the schema.
_GOLDEN_MID = 4004219
_GOLDEN_ROWS = 3292453
_GOLDEN = {
    "frame_count": "5acea517d6c4531d55921b1b056b71f0a8223b385acaed82080dfd71b2ef0368",
    "period_id": "56072e8144f4f022a6e720085f39abbd1c186363d1a236b37a9185314b8fc2b2",
    "timeelapsed_from_period_start_s": "1e14dd20af62a4d1132963fabf246f9b5812489bb8d6af6ac07e65cbca3317c9",
    "period_elapsed_s": "47ce641df0c93f5a60da23c6b61c8e810adddd9c216c3d166848f1d47fce85a2",
    "team_id": "ccfb6c5aeb95a373aa83ff3ad9b041b41c1f07ace7d1be7d37dd789e12bb0197",
    "player_id": "1df86de3d7e904eeea4473565389524e19a45d1cd1ac4f97ae58abb79cd3dff4",
    "is_ball": "33dde77c787367b2afe3096975c9a1ef8115cfcd3863387b5fb3d1562b8fd8b6",
    "is_goalkeeper": "56c6998572560fb8ed7873bf19e012f04759c3e8dd387f2f8bc4cec36b35b57a",
    "direction_of_play": "b6e62cb5da86261851bfff07a20f775f3bb26858e99d9823b2c0e81e28aa94f1",
    "x_tracking": "fcbcc8e900ebeda74966137539d95503c17818049ddfcde1c3ca1280c17f3273",
    "y_tracking": "a139b0e4c306305bd92d1a5931a1bf455db9447c4c26a664f4e1c055b9094bb6",
    "vx": "da44385e54f6bd7130334685524dc918e974c0274215079207c07cd983521d33",
    "vy": "96a6ec4f5757bd022b3886e8da1e4445266a6e52216f4327fe2c48f5d4cadbf6",
    "speed": "1a33fbe95bb736170976fd80bb851541141e1e657919db309f236b61093b4f5c",
    "visibility": "627b79405d6394f670cb410da56a9a3f28475e74ffc966569cbe800547577de6",
    "confidence": "fde4859c87b39d6bae428a14fa52b7ba50e388494be97cc3c9b28f79a248ec3a",
    "x": "4ef9f432a34377d0763455039d20e6940ee03b9ecbf01c36c0a3b36b71af9a5e",
    "y": "11c66e614895fb1023052962856019cfa210a70b25fb2bd86c5dc305a4549f10",
}


def _col_sha(s: pd.Series) -> str:
    """Bytes of a column, nulls included: NaN has a stable bit pattern and None stringifies."""
    a = s.to_numpy()
    raw = a.tobytes() if a.dtype.kind in "fiub" else "\x00".join(map(str, a)).encode()
    return hashlib.sha256(raw).hexdigest()


@requires_data
def test_load_tracking_matches_the_recorded_frame():
    """Values, column by column, against a frame recorded off this sample."""
    df = loader.load_tracking(_GOLDEN_MID)
    assert len(df) == _GOLDEN_ROWS
    assert {c: _col_sha(df[c]) for c in df.columns} == _GOLDEN


@requires_data
def test_load_tracking_float32_flag_is_the_only_thing_it_changes(mid):
    """float32=False keeps the wider floats; nothing else about the frame moves."""
    wide = loader.load_tracking(mid, float32=False)
    assert tuple(wide.columns) == tuple(c for c, _ in _SCHEMA)
    for col, dt in _SCHEMA:
        assert str(wide[col].dtype) == ("float64" if dt == "float32" else dt)


# ── The 60-slot invariant, over the whole tracked sample ───────────────
def _old_kickoff_rule(mid: int, home_id: int, gk: dict) -> tuple:
    """The REPLACED kickoff rule, kept verbatim as a measured negative: first 5 frames of
    each period, home outfielders only, falling through to Right-to-Left on an empty window.

    Reference-only, and not part of src/. Only the window is parsed: the rest of the frames
    are skipped on a substring probe, and anything the probe does not recognise is parsed
    rather than dropped, so a changed record layout costs time and never correctness."""
    first5: dict = {}
    done: set = set()
    with open(loader._PAIRING[mid]) as f:
        f.readline()                                 # match metadata, not a frame
        for line in f:
            if any(f'"period":{p}' in line[:80] for p in done):
                continue
            fr = json.loads(line)
            p = fr["period"]
            first5.setdefault(p, []).append(fr)
            if len(first5[p]) == 5:
                done.add(p)

    directions, blind = {}, {}
    for p in sorted(first5):
        xs = [pl["x"] for fr in first5[p] for pl in fr.get("homePlayers", [])
              if pl and not gk.get(f"{home_id}_{pl.get('jerseyNum')}", False)]
        directions[p] = "Left to Right" if xs and np.mean(xs) < 0 else "Right to Left"
        blind[p] = not xs                            # answered with no evidence at all
    return directions, blind, first5[min(first5)][0].get("homePlayers")


@pytest.fixture(scope="module")
def sweep30():
    """One pass over all 30 tracked matches, shared by every claim below.

    Loads each match in turn and keeps only the per-slot summary, so the 1.06 GB frame
    never accumulates. Module-scoped because paying this five times would be five times
    the cost for the same numbers."""
    out = []
    for m in loader.tracked_matches():
        home_id, _, gk = loader._team_ids_and_gk(m)
        df = loader.load_tracking(m)
        keepers = df[df["is_goalkeeper"]]
        slots = {}
        for p, g in keepers.groupby("period_id"):
            slots[int(p)] = {
                "home_x": float(g.loc[g["team_id"] == home_id, "x_tracking"].mean()),
                "away_x": float(g.loc[g["team_id"] != home_id, "x_tracking"].mean()),
                "direction": g["direction_of_play"].iloc[0],
            }
        del df, keepers
        old_dir, old_blind, first_home = _old_kickoff_rule(m, home_id, gk)
        out.append({"match_id": m, "slots": slots, "old_dir": old_dir,
                    "old_blind": old_blind, "first_home": first_home})
    return out


@requires_data
def test_every_slot_puts_the_two_keepers_on_opposite_sides(sweep30):
    """The claim the direction rests on: if both keepers ever sat the same side, the sign of
    one keeper's mean x would not identify the goal he defends."""
    slots = [(r["match_id"], p, s) for r in sweep30 for p, s in r["slots"].items()]
    assert len(slots) == 60                          # 30 matches x 2 periods, no slot missing
    same = [(m, p) for m, p, s in slots if np.sign(s["home_x"]) == np.sign(s["away_x"])]
    assert same == []


@requires_data
def test_the_keeper_sits_far_enough_out_to_be_unambiguous(sweep30):
    """|mean x| measured over the 60 slots spans 29.3 to 45.6 m, never near the halfway
    line where the sign would be noise. Banded rather than pinned, so a re-measure does
    not fail on a last digit."""
    absx = [abs(s["home_x"]) for r in sweep30 for s in r["slots"].values()]
    assert min(absx) >= 29.0
    assert max(absx) <= 46.0


@requires_data
def test_the_direction_flips_between_halves_in_every_match(sweep30):
    """Teams swap ends at half time, so a rule that never flips is reading something else."""
    flipped = [r["match_id"] for r in sweep30
               if r["slots"][1]["direction"] != r["slots"][2]["direction"]]
    assert len(flipped) == 30


@requires_data
def test_the_old_kickoff_rule_is_wrong_on_29_of_the_60_slots(sweep30):
    """The measured negative that justifies reading the keeper instead. Two separate counts,
    and they are NOT the same number: 29 slots came out wrong, while 32 were answered with no
    home outfielder in the window at all (all 30 period-1 windows, plus 2 in period 2). That
    rule never raised. It returned Right-to-Left on an empty window."""
    slots = [(r, p) for r in sweep30 for p in r["slots"]]
    wrong = [(r["match_id"], p) for r, p in slots if r["old_dir"][p] != r["slots"][p]["direction"]]
    blind = [(r["match_id"], p) for r, p in slots if r["old_blind"][p]]
    assert len(wrong) == 29
    assert len(blind) == 32
    assert len([1 for m, p in blind if p == 1]) == 30
    # every match opens on a placeholder frame, so the window has nothing to read
    assert [r["match_id"] for r in sweep30 if r["first_home"] != [{}]] == []
