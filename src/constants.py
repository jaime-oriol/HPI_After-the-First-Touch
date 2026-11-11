"""constants — every physical and business magnitude the pipeline reasons over, defined once.

WHY THIS MODULE EXISTS. A quantity typed out in several places drifts between the copies, and
the failure is silent. The case that decides the rule: a single name such as `WINDOW_S` would
mean 6.0 s to the race and 8.0 s to the episode, and an import between those two modules would
then change a window with nothing failing out loud.

THE RULE IS UNIFY BY QUANTITY, NEVER BY VALUE. Numbers that merely coincide stay apart, and
several here do: CONTACT_RADIUS_M is 1.5 metres while REACTION_MAX_S is 1.5 seconds,
RACE_SCAN_HORIZON_S is 6.0 seconds while s08's sampling disc is 6.0 yards, and FK_MIN_X and
HALF_X are both 60.0 while meaning the annex's floor for a flank free kick and the halfway
line. Collapsing any of those would couple two decisions that have no reason to move together.

THREE COORDINATE CONVENTIONS COEXIST AND ARE NEVER MIXED:

    tracking   metres, origin at the pitch centre, 105 x 68        (loader, s04, pitch_control)
    events     StatsBomb yards, 120 x 80, origin at a corner flag  (s01, s02, s08)
    render     Opta 0-100 on both axes, rescaled by loader from the metres above  (viz)

The suffix marks the space when it is there: `_M` metres, `_SB` StatsBomb yards, `_S` seconds,
`_MS` metres per second. It is NOT a complete scheme. Several constants here are in StatsBomb yards
carrying no suffix at all, among them the whole zone block, the free-kick and short-corner
thresholds and the two contest radii, and EMPTY_DEFENDER_AREA_SB ends in `_SB` while measuring an
AREA in square yards.
Renaming them would change the API across the whole tree, so they stay. Read the section header
for the unit, not the name.

ON THE COMMENT DENSITY HERE. This file runs far above the repo's 3-5 comments per 100 non-blank
lines, the highest in src/ (measure_comments.py --per-file gives the figure), because it is a
data dictionary, not narrated code. Every line is a magnitude that has to carry its unit and where its
value came from, and the harder rule is that a figure without verifiable provenance is not written
down at all. Stripping these notes to hit a density target would trade the rule that matters for
the one that does not.

WHAT IS DELIBERATELY NOT HERE. The layout parameters of src/viz/ and the shape annotations of
src/pitch_control.py stay beside the code that draws and broadcasts them, because their effect
cannot be read off the line without that code. The PPCF parameters (LaurieOnTracking's, not the
paper's) stay in pitch_control.default_model_params, where the citation lives. Tuning grids
(TAU_GRID, the C grids) stay with the estimator that sweeps them: they are search spaces, not
magnitudes.
"""

from __future__ import annotations

import os

# ── Tracking: Hudl Broadcast X&Y, metres, origin at the pitch centre ─────────────
# The frame rate is measured, not declared by the vendor: videoTimeMs advances by exactly
# 40 ms on all 146,124 frame-to-frame steps of a sample match, with no drops. The pitch is
# the vendor's own declared half-extent of +-52.5 m by +-34.0 m.

FPS_TRACKING = 25
PITCH_LENGTH_M = 105.0
PITCH_WIDTH_M = 68.0

# The causal Savitzky-Golay derivative, in frames. CHOSEN, not derived: neither value has any
# provenance in the code or in docs, so what is recorded here is what each one does. Seven
# frames is a 280 ms trailing window at the rate above, and polyorder 1 makes the fit a local
# linear velocity. Reading that window at its NEWEST sample is what makes it one-sided, and the
# one-sidedness is the anti-leakage guarantee rather than a smoothing preference: scipy's centred
# savgol_filter moves a sample (window - 1) / 2 = 3 frames BEFORE a step change, measured,
# which is the future entering a feature read at T.

SAVGOL_WINDOW_FRAMES = 7
SAVGOL_POLYORDER = 1

# ── Events: StatsBomb yards, 120 x 80, attack towards x = 120 ────────────────────
# The convention holds exactly on this export: over 166,081 located events not one row fell
# outside [0, 120] x [0, 80]. audit_tracking treats it as a HARD bound, and the tracking bounds
# above as soft, because broadcast tracking legitimately overruns the touchline.

PITCH_X_SB = 120.0
PITCH_Y_SB = 80.0
PITCH_OPTA = 100.0                     # both axes; the render scale loader rescales metres onto
PITCH_Y_MID_SB = PITCH_Y_SB / 2.0      # the goal-mouth axis; zone bands are measured off it
GOAL_SB = (PITCH_X_SB, PITCH_Y_MID_SB)
GOAL_WIDTH_SB = 8.0                    # the laws' eight-yard mouth; DERIVED posts, so they cannot
GOAL_POSTS_SB = ((PITCH_X_SB, PITCH_Y_MID_SB - GOAL_WIDTH_SB / 2.0),   # drift off the goal centre
                 (PITCH_X_SB, PITCH_Y_MID_SB + GOAL_WIDTH_SB / 2.0))

# ── The provider's empty-population sentinels, StatsBomb yards ───────────────────
# Where the population a shot-geometry field would have been computed over is empty, the export
# writes a fixed number rather than a null. Reading either as a real magnitude puts a 30-yard
# defender and a 1,000-yard² region into the distribution.
#
# Recovered by inversion over the grant first (data_provider.tex §sec:density: ToD1 is 30 on
# precisely the shots whose frame holds no outfield opponent, 100.00% of 43,940; DefArea is 1,000
# on those and nowhere else, 99.97%), and CONFIRMED against the provider's published source:
# StatsBombR R/freezeframeinfo.R writes the 1,000 at line 171 and the 30 at lines 195-198, the
# latter only when a shot has neither a goal-side nor a behind-the-shot opponent.
# The two empty populations are not the same one. DefArea's is the defensive line alone
# (position.id in 2:8, line 144); ToD1's is every outfield opponent (line 106).

EMPTY_DEFENDER_DISTANCE_SB = 30.0
EMPTY_DEFENDER_AREA_SB = 1000.0

# ── The provider's shot cone and defensive line ──────────────────────────────────
# DECLARED BY THE SOURCE, NEVER FITTED, and that distinction is the only reason the fields
# below them can be marked verified rather than reported. Sweeping the cone against the data
# instead plateaued at 95.0%, under either a perpendicular slack or an angular body shadow.
#
# The cone is the triangle whose base spans the goal line at y = 35 and 45, the real posts of
# GOAL_POSTS_SB widened by one yard a side, which defensiveinfo.R writes as 36-1 and 44+1. Its
# apex is not the shot but the shot pushed one yard further out, to DistToGoal + 1
# (freezeframeinfo.R :69-72). DefArea's box spans the opponents with position.id in 2:8, the
# defensive line alone rather than every outfield opponent (:142-146). And a body standing
# exactly on the shot enters the inverse-distance sums at a third of a yard instead of being
# dropped, which is the one substitution that carries a body at zero distance into a sum of 1/d.
#
# Measured over the grant: density, density.incone and DefArea's magnitude each reproduce the
# published value on 100.00% of the 43,940 scored shots. Three of those turn on the 1/3 alone,
# the shots whose frame holds an opponent on the shot's own coordinates.

CONE_POST_SLACK_SB = 1.0
CONE_APEX_BACKOFF_SB = 1.0
DEFENSIVE_LINE_POSITION_IDS = (2, 8)   # inclusive; position.id 1 is the keeper
ZERO_DISTANCE_SUBSTITUTE_SB = 1.0 / 3.0

# ── The episode, in seconds ──────────────────────────────────────────────────────
# SECOND_BALL_WINDOW_S is measured rather than inherited: possession contiguity from the first
# contact holds through the ninth second and breaks in the tenth (evidence.md §15 has the curve),
# so the cutoff sits a second before the last second that holds, a margin for earlier breaks.
#
# TRANSITION_WINDOW_S holds the same value and is NOT the same decision: it bounds how long a
# defending shot can still be blamed on the clearance that preceded it. The two are kept apart
# so neither moves because the other did.

SECOND_BALL_WINDOW_S = 8.0
TRANSITION_WINDOW_S = 8.0
HORIZON_PAD_S = 20.0                   # tail on the event scan so neither window is truncated

# ── Zone thresholds, StatsBomb yards ─────────────────────────────────────────────
# Design choices, not measurements: they name where a coach would say the box, the goalmouth
# and the edge begin. They are stated here so a change is one edit rather than a search.

BOX_X = 102.0                          # edge of the penalty area
BOX_Y = (18.0, 62.0)                   # its width, the laws' 44 yd about PITCH_Y_MID_SB
SIX_YARD_X = 114.0                     # the six-yard line, the laws' 6 yd off the goal line
# The six-yard box runs 6 yd out from each post as far as it runs into the field: DERIVED from
# the posts and SIX_YARD_X, so it cannot drift from either.
SIX_YARD_Y = (GOAL_POSTS_SB[0][1] - (PITCH_X_SB - SIX_YARD_X),
              GOAL_POSTS_SB[1][1] + (PITCH_X_SB - SIX_YARD_X))
GOALMOUTH_X = 108.0                    # the penalty spot's line, 12 yd off the goal line
HOLD_X = 96.0                          # the frontal's back line; short of it an attacker holds back
# G2's middle row runs to this many yards off the goal line, and the central block short of it joins
# its `edge`: the middle of Opta Analyst's twelve-to-eighteen-yard row, as the author set the cut
# (data/derived/prereg_2026-10-13/zones_g.md).
G2_EDGE_CUT = 15.0
HALF_X = PITCH_X_SB / 2.0              # the halfway line; DERIVED, so it cannot drift off the pitch
EDGE_X = (100.0, 108.0)                # the central edge band, reported beside the frontal
EDGE_Y_HALF = 20.0                     # central-channel half-width, measured off PITCH_Y_MID_SB
REBOUND_X = (96.0, 114.0)              # rebound band, the control_rebound denominator
REBOUND_Y = (16.0, 64.0)
TARGET_DENSITY_RADII = (5.0, 10.0)     # att_density_5 / att_density_10 rings around the target

# The drop anchor for congestion, and the only MEASURED one of the two it uses: the median of
# drop_x / drop_y over the 9,627 cleared corners of the situation table (run_situation's radii
# record), against a mean of 100.0 / 40.7 and an IQR of 92.5-110.1 in x. It sits 6.0 yd short of
# GOALMOUTH_X.
#
# Being GLOBAL is what keeps it legal. One number for every corner carries nothing about how any
# one of them resolved, whereas anchoring per corner on its own drop is the defect that barred
# att_density_5 from the design matrix. Its substrate is the cleared corners alone, which the
# outcome selects, and that caveat travels with it.
DROP_ANCHOR_SB = (102.0, 40.5)

# The floor the annex of the cut lowers the flank kind to, to read the free kicks below the paper's
# FK_FLANK_MIN_X with the same predicate (run_flank_cut).
FK_MIN_X = 60.0

# The paper's free-kick population: a kick taken from here outside the box width (BOX_Y), a corner
# with offside. Chosen after reading the grant's flank free kicks by depth, outcomes included, which
# the paper declares; run_flank_cut keeps the reading with the paper's predicate from x = 60
# (measured.md §flank_cut): against the 2465 kicks from 85, the 2122
# from 60 to 85 see the attack win the first contact +0.118 [+0.088, +0.148] more often and turn
# fewer deliveries into an after-first-touch shot, -0.029 [-0.047, -0.011]. No length bar: of
# the 2465 it drops 41, 34 of them on the defence's first touch a median 0.36 s after the kick, 30
# of those a block, and 2 short free kicks (fk_scope/d40_fk_length_bar.log).
FK_FLANK_MIN_X = 85.0

# Below this a corner is SHORT. It is the one declared exception to the pass.length bar:
# `short_corner` is in the design, its length branch fires on a handful of corners and the rest
# are ground passes `delivery_height` already carries (definitions.md §5 has the count).
SHORT_CORNER_MAX_LENGTH = 15.0

# A short or ground corner is played to a team-mate, so its first touch is a reception and says
# nothing about a contest for the ball. Its delivery is the first cross the attacking team plays
# in the same possession, when one comes within this many seconds of the kick: the bound a first
# contact has after a long corner, so the two are bound to one value and move together. A short
# corner with no such cross has no delivery into the box: it is counted and never segmented or
# modelled.
SHORT_CROSS_WINDOW_S = SECOND_BALL_WINDOW_S

# ── The plan's partition of the box ──────────────────────────────────────────────
# Which partition the plan is counted on: "C", s02b's twelve cells, "F", the six zones a coach names
# (data/derived/prereg_2026-10-01/zones.md), or "G2", Opta Analyst's cut of the box
# (data/derived/prereg_2026-10-13/zones_g.md). G2 is the default for a football reason, the cut a club
# reads, and not a statistical one: its screen found no gain over C on M1
# (data/derived/prereg_2026-10-13/deviation_2026-10-25.md). HUDL_ZONES sets it for one run, and every
# consumer reaches it through src/zones.py. C and F share the frontal; G2 widens it to its own `edge`,
# so under G2 the jobs and the defenders in the frontal move with the partition.
ZONE_PARTITION = os.environ.get("HUDL_ZONES", "G2")

# ── The second-ball contest, StatsBomb yards ─────────────────────────────────────
# These two were CHOSEN, not derived. The distribution each one cuts is measured below over the
# grant's cleared corners, by the situation table's radii record (run_situation), which
# report_figures carries, so the figures move when the pipeline moves.

# Bodies this close to the drop are counted as being at it. Selective, as intended: of the
# 158,584 body-to-drop distances the situation frames score, 4.75% fall inside 3 yd and 10.13%
# inside 5 yd.
NEAR_RADII = (3.0, 5.0)

# A stake-drop attacker this close to the drop counts as staked. Over the 6,244 corners carrying
# a staker at a contested drop the distance runs p25 = 9.04 yd, median 15.42 yd, so this admits
# a little over a quarter of them (28.73%) rather than a token few or almost all.
STAKE_RADIUS = 10.0

# ── The race, metres and seconds ─────────────────────────────────────────────────
# Chosen, and measured over the 30-match tracked Serie A sample, the only source of continuous
# positions on the grant, so the figures describe 30 matches and not the whole grant. They are
# one pass with no generated twin: nothing in data/derived/ persists them, so they do not
# move when the pipeline moves. The ball rows that pass read are the feed's, about a third of
# them interpolated rather than detected, and the figures below that read the ball inherit that
# (measured.md §race has the share).

# Ball-to-player gap that counts as a touch. Over 22,518 frame-wise gaps once the ball is in
# flight, p25 = 1.28 m and the median is 2.79 m, so this cuts near the 30th percentile: tight
# enough to mean contact, loose enough to survive broadcast-tracking error.
CONTACT_RADIUS_M = 1.5

# The ball must fly this far before a contact is admitted, or the nearest body is the taker on
# every corner. Measured on the 174 races: when the contact does fire, the ball had already
# flown p1 = 7.1 m and a median 14.3 m, so the gate sits BELOW the first percentile of genuine
# contacts. It removes the taker artefact and costs essentially no real contact.
MIN_DELIVERY_M = 8.0

# Contact -> the point the loose ball is raced to. NOT a settle point, whatever downstream
# prose calls it: the ball still travels a median 5.6 m (p75 = 10.9 m) over this second. It is
# a fixed look-ahead, and the race is to where the ball WILL be, not to where it stops.
LOOSE_HORIZON_S = 1.0

# Velocity component toward the loose ball, in m/s, that counts as reacting. The measured
# toward-component at the contact frame has median 0.50 m/s over 3,820 racers, so this sits
# almost exactly on the median and splits the field in two. Its substrate is a velocity, so it
# depends on kinematics.causal_savgol reading its window at the newest sample (pinned by
# test_kinematics); no artefact re-derives the median.
REACTION_SPEED_THR = 0.5

# Cap on the measured reaction lag, and a note on HOW TO READ A FEATURE rather than on where a cut
# came from. 27.3% of racers never clear REACTION_SPEED_THR inside the window and are assigned this
# cap, so for better than one racer in four the lag is the bound rather than an observation.
# Moving the number does not repair that, it only changes what fraction of the column is censored.
REACTION_MAX_S = 1.5

# Tracking scanned from the DELIVERY frame; not an episode length, which is SECOND_BALL_WINDOW_S
# from FIRST CONTACT.
RACE_SCAN_HORIZON_S = 6.0

# The soft-argmin temperature is a DEFAULT, not a finding: calibrate_tau selects it out of fold
# grouped by match and run_race adopts the calibrated value whenever the operativity guard clears.
RECOVERY_TAU_S = 0.4

MAX_PLAYER_SPEED_MS = 12.0             # velocity cap; a jump above it is a detection gap, not a sprint

# ── Publication floor ────────────────────────────────────────────────────────────
# A cell holding fewer rows than this is printed with its count and a flag, and never as a
# rate. It is quoted in prose all over the drivers, so it is interpolated
# from here rather than typed into the message.

MIN_CELL = 20

# Koo & Li (2016) bands: <0.50 poor, 0.50-0.75 moderate, 0.75-0.90 good, >0.90 excellent. The
# pre-registered rule is that no per-team ranking ships below good.
RELIABILITY_FLOOR = 0.75
