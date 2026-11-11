# Architecture

The stages appear in the order they depend on each other, and each one reads the one before it.

## Data sources

Two local sources, both from the granted Hudl export and neither committed here:

| Source | Contents |
|---|---|
| Serie A sample with tracking | 30 matches: events, 360 freeze-frames and Broadcast X&Y |
| Big-5 2025/26 | five competition-seasons: events and 360 |

Serie A is nested in both. The tracked sample is a subset of its own league season, and the 360
arrives embedded per event as `freeze_frame` rather than served separately. So the 360 accessor
is a filtered view over the events accessor, not a second parse.

Three coordinate spaces coexist and are never mixed. Tracking is in metres, with the origin at
the pitch centre. Events arrive in StatsBomb yards measured from a corner flag, and the render
space is Opta units on both axes. Where a name carries a suffix it names the unit, `_M` for
metres, `_SB` for StatsBomb yards and `_S` for seconds, but the scheme is not complete: the zone
thresholds and the contest radii are StatsBomb yards with no suffix, and `constants` says which.

## Layout

| Path | Holds |
|---|---|
| `src/` | the stages and the shared modules below |
| `pipeline/` | one driver per stage, plus the auditors and the descriptive probes |
| `renders/` | the figure scripts, one per output |
| `docs/` | this file, the data schema and the definitions |
| `figures/` | fonts, icons and the author's logo; generated images are not committed |
| `refdata/` | small versioned reference tables |

## Shared modules

These import nothing from the stages, and are what the stages are built on.

| Module | Responsibility |
|---|---|
| `constants` | every physical and business magnitude, defined once, with its unit and provenance |
| `validate` | group splitters, the leakage guard, match-clustered intervals, label-shuffle controls |
| `loader_sb` | events and 360 for both sources, over a per-match parquet cache |
| `loader` | Broadcast X&Y tracking, paired to events by (home, away, date) |
| `kinematics` | causal Savitzky-Golay velocities and the ball locator, over tracking frames |
| `pitch_control` | Spearman PPCF, cited lineage for one console cross-check |
| `shot_geometry` | the geometry of the shot freeze-frame, and the provider fields recovered from it |
| `zones` | the plan's partition of the box, the one switch between the partitions, and the flip |

`validate` is model-agnostic on purpose, because a guard only one stage runs is no guard at all.
`s02b`, `s03`, `s04`, `s04b` and `s05` assert on every split path they own. On a
grouped split disjointness is the half that comes free, since the split is already disjoint on the
key it is handed; a key that resolved for every row is the half that does not. A key that failed to
resolve compares unequal to itself and slips through a set intersection on both sides at once, so
`s04` checks resolution before it splits at all, where scikit-learn would otherwise raise first
with a message about imputers.

Every headline figure carries an interval that resamples matches, because corners inside a match
share teams, coach, referee and scoreline. Those intervals hold the fitted model fixed, so they
are too narrow; `refit_cluster_bootstrap` is the one that refits inside every draw, and the
headline step, the delivery-against-360 contrast, every ablation and the freeze-frame steps are
read on it. They are read in normal form, the point plus or minus half the refit width: a resample
thins the distinct matches and shifts every draw by the learning curve, which the percentile form
carries and the pivotal form moves to the other side of the point. `refit_cluster_draws` hands back
the draws themselves, and `run_refit_draws` keeps them for the contrasts that carry a claim, so
whether they are symmetric enough for that reading is checked rather than assumed. It takes a
statistic that may be a single number or a whole curve, and a curve is intervalled off the same
draws in one refit each rather than one per point, so the points keep their joint variation.
Both of its branches fit under one BLAS thread: capping only the workers leaves the serial path
reducing the same dense algebra in a different order, and a bound that moves with the worker
count is not a bound of the procedure it names.

`loader_sb` parses each match once and keeps a normalised parquet under `data/derived/`. Two
things force that normalisation. StatsBomb writes `{}` where a container does not apply and arrow
refuses a column mixing dict with list, so the cache stores empties as null and the loader puts the
JSON contract back on read. Then on the way back pyarrow hands over an ndarray where the JSON held
a list, which every `isinstance(v, list)` filter in the tree reads as absent. That one fails
silently, so the loader restores the type and asserts it at its own boundary, and a test pins the
raw side non-empty before it compares anything. The accessor also takes an optional column subset
that reaches the parquet read. What decides the cost of that subset is whether it carries a nested
container, not how many columns it names.

`loader` keeps no cache. Tracking is parsed from the raw JSONL on every call, streamed into typed
buffers rather than held as parsed frames, so a match costs more resident memory than the frame it
hands back (`loader.load_tracking` records the peak). That peak bounds the fan-out of every stage
that opens tracking, and worker counts are sized off it rather than off the core count. Pairing is by (home, away, date): the two
feeds number their matches in different spaces and spell the same clubs differently, so no id joins
them, and a file that fails to pair raises instead of quietly leaving the sample.

Which goal the home side defends is read from where its keeper stands, not from the shape at
kickoff. The broadcast clock starts before the kick, so a period opens on frames with no outfielder
detected at all, and on this sample a kickoff-shape rule leaves many (match, period) slots
unmeasured and gets many of the rest wrong. The keeper is unambiguous by comparison: his mean x
never comes close to halfway, the two keepers sit on opposite sides in every slot, and the
direction flips at half time in every match. `tests/test_loader.py` re-measures each of these
counts on the sample. A period with no keeper detection raises: unknown and
right-to-left are different answers.

`kinematics` and `pitch_control` sit on either side of one boundary. The causal filter in the
first of them runs on every corner the race stage touches, and it is the anti-leakage
guarantee behind every arrival time published here: the centred filter fits a window straddling
the present frame, which reads the future into a quantity measured at the delivery. The PPCF
serves one line of one render, which prints it at the loose-ball point to cross-check the reach
field the race stage builds for itself. It reaches a console and never a figure. From
`pitch_control` the race takes a single value, the travel-speed cap. No surface.

`shot_geometry` holds the geometry of the shot freeze-frame, which is not the 360 freeze-frame and
does not share its keeper rule. Several of the provider's shot fields ship with no specification,
so they were recovered by asking which players in the frame reproduce the published value. Most of
that closed by inversion. Two fields did not, and they close instead against the source the
provider publishes. A geometry swept against the data is fitted and can only ever be reported; one
the source declares can be verified. The cone here therefore carries no free parameter. Each rule
has a competing reading that looks equally sensible and scores worse, and the auditor scores those
losers under their own names so a number can be cited rather than a survivor asserted.

## Stages

The pipeline runs delivery to outcome. Each stage reads the one before it rather than
re-deriving what it needs, so two stages cannot disagree about what a corner is, or about what
counts as a clearance.

| Stage | Reads | Produces |
|---|---|---|
| `s01_phases` | events | the phase grammar: one label per delivery, one row per attacking shot |
| `s02_signature` | events, 360, s01 | the commitment signature and the delivery-instant geometry |
| `s02b_aim` | 360, s01, s02 | the zone grid, the kick-time block and the aim model read from it |
| `s03_value` | s01, s02 | the competing-outcomes model, its ablations and its baseline ladder |
| `s04_race` | tracking, s01 | the race to the loose ball: arrival times, reaction lag, recovery probability |
| `s04b_snapshot` | 360, s01, s02 | the same race run from the delivery snapshot, and the ladder that scores both |
| `s05_causal` | s03 | the cross-fitted effect of a defender held in the frontal, the edge of the box |
| `s08_situation` | events, 360, s01, s02 | the freeze-frame at the drop, and the four states |

The frontier itself is `s01`. It segments every corner and attacking free kick into delivery,
first contact, second-ball window and terminal outcome, then places every attacking shot in
exactly one of five segments and under exactly one delivery, the last set piece struck before it, so
a delivery taken inside another's episode does not share the shots that follow it. A forced own goal
is a goal and no shot, filed in the segment of the action that produced it (`definitions.md` §4).
That segment comes from the event chain. It never comes from the field the provider attaches to the
shot, because any rule behind that field has already seen how the shot turned out. A grammar is
built for the kinds it is handed, `corner` and `free_kick_flank`
(the paper's free kicks, from x ≥ 85 outside the width of the box); `s04` and `s08` take the same
argument, corners by default.

A short or ground corner is a kick and not a delivery: it is played to a team-mate, so the first touch
after it is a reception that says nothing about a contest. Its delivery is the first pass the provider
tags as a cross, by the same side in the same possession within `SHORT_CROSS_WINDOW_S` of the kick, the
same bound a first contact has after a long corner, and the first contact, the window, the segments and
the freeze-frame are all read at that cross. A short corner no cross follows in time keeps a row of its
own with no delivery, so a side's corners taken are still counted, and it has no shot, segment or
model row (`definitions.md` §3).

From there the delivery freeze-frame is read once, by `s02`, and nothing after it. Every delivery is
read in one canonical frame, mirrored by the side of its kick. Each attacker gets one of three jobs
by where they stand, the width of the box included, each side's bodies are counted per zone of the
partition one switch names (`src/zones.py`), and the geometry set is measured over the players the
camera actually saw. The delivery frame covers the box and little else, so there is nobody deep to
measure and the rest-defence quantities are not computable on this data. Two features are named
for what the frame supports: `box_block_depth` is the depth of the defensive block, not a
rest-defence line, and `open_lane_central` is a gap in the central channel, not a counter lane.

One of those features measures distances rather than bodies. Congestion asks how little room the
bodies leave each other, which an occupation count cannot answer: players spread across the width
of the area occupy it heavily and congest nothing. It reuses the provider's own inverse-distance
kernel from `shot_geometry` rather than inventing a measure. That reuse is what makes the keeper
rule there a parameter. The two freeze-frame schemas identify the keeper by different keys, and
one rule applied to the wrong frame finds no keeper at all instead of failing. The anchor is a
fixed point and never the delivery target, because that target is where somebody touched the
ball, so anchoring on it would read the future of the play.

`s03` prices the delivery against the outcome it produced. Features come from `s02` at the
delivery, labels from `s01` after it, cross-validation is grouped by match, and the calibrator is
fitted on inner folds only. Any claim that the 360 features add something has to clear the
delivery baseline, which already knows where the ball was kicked from and with what swing.
Not the uniform floor.

`s02b` reads the same frame for a different target. Whether the 360 prices the outcome is `s03`'s
question; whether it carries the plan is this one. It asks whether the frame predicts the zone the
ball reaches on the corners the attack reached first, where that zone is the aim, and
`run_aim_zone` uses the inferred aim to re-read the map of chain value by zone. The end zone is the
aim model's target and the naive map's label and never an input, and the model's inputs name their
source fields against the same barred list as `s03`. The idea was developed on one league, so that
league is reported apart and the other four decide, under rules written before they were read.

Continuous motion is needed in one place only, `s04`. It finds the first contact, then races
every player to the loose ball with a measured reaction lag and capped-speed travel. A
soft-argmin over those arrival times gives the recovery probability. One model end to end. The
ball it reads is the vendor's, and on most of the corners it publishes the vendor marks that
position as interpolated at the contact and at the loose point: the ball is airborne there, which
is where the provider says its tracking is weakest. The stage reads the row as emitted and states
the bound rather than masking, because a mask would empty the race: each race carries the
vendor's flag for the contact and for the loose point in its record, and the published figures are
scored again over the detected-ball subset as a bound, never as a second model.

The race reads the tracking at each event's instant on the game clock, and that clock runs behind
the events' in every half, by about a second (`evidence.md` §14), so its contact frame comes late and
its figures are not cited. `src/tracking_clock.py` measures the offset per match and half, from passes
and receipts against the detected ball, and `pipeline/run_plan_race.py` reads every instant through
it. On the tracked corners whose second ball the event chain resolves, it counts after the fact
whether the side that won it already had the player nearest the drop at the kick, got one nearest by
the clearance, or won the run after it, and when the players near the drop started towards it.

`s04b` runs that same fusion on the delivery snapshot instead, which is what turns "does
tracking help" into a measurement. Freeze-frame positions race to a fixed pitch point rather
than to the drop this corner produced, because the drop is read after the delivery. The two
arms then differ in what feeds the arrival times and in nothing else, and they are scored
against the same event-derived outcome so neither is graded by a label from its own feed.

That comparison splits into halves with very different power, and reading one for the other is
the trap. The freeze-frame arm needs no tracking, so it runs on every corner in the grant that
carries a delivery frame and a resolved second ball, across all five leagues. The head-to-head
against the tracked race runs only where both feeds exist, which is thirty matches and does not
grow, and since its tracked arm is the race read on the unaligned clock, it is not cited.

On the matches carrying both, the gap between the two arms is reported in pieces rather than as
one number, because three things separate them at once. A tracked rung read at the kick with no
kinematics attached isolates what the freeze-frame fails to SEE; the rung at the first contact
adds velocity, the measured lag and the seconds between the two instants; the published race
adds knowing where the ball came down. Every rung scores the same corners against the same
label, so each step is a difference of predictors and nothing else.

`s05` then asks a different question about the same episode: the effect of holding a defender in
the frontal, the edge of the box. Its estimand is stated rather than assumed: overlap on this substrate
cannot carry an average treatment effect, so the target population is shifted rather than the
weights patched, and all three estimands are reported side by side so the size of what clipping
hides stays visible. Defining the estimand through the propensity has two consequences that an
average effect does not carry: the variance gains a term for having estimated that propensity,
and double robustness lapses, because a wrong propensity moves the population rather than only
the estimate. Both nuisances pick their own regularisation inside each fold, on that fold's
training block, so nothing fitted on data has seen the rows it is scored on.

Which covariates to adjust for is a choice this design cannot settle, and the stage reports it
instead of making it. The congestion block reached the confounders because the value stage's
design matrix is assembled from the feature groups, and treatment and congestion are both read in
the delivery freeze-frame, so neither precedes the other and no test here separates a confounder
from a mediator. It runs one primary specification, registered before it ran, and around it a
leave-one-group-out curve over every feature group. Every one of them holds the league and both
teams fixed, and the primary one is run once more without them, so what those controls move is
published rather than assumed. The curve says whether one block is special or whether the estimate
is not resolved by this design at all. No specification may hold the treatment or what rebuilds
it, and the stage enforces that rather than trusting the caller.

Last, `s08` reads the drop and never the delivery. Once the clearing contact sends the ball
loose, the touch that contests it carries its own freeze-frame. Crossing whether the plan staked
that drop zone with whether the attacking side won it gives the four states.

Two things bound that grid, and both are reported rather than smoothed over. The cross needs
both halves, so a corner whose delivery carries no freeze-frame has no plan side and one where
nothing contests the loose ball inside the episode has no outcome side; neither absence is a
labelling failure, and the share of the cleared set that survives both is generated rather than
divided in prose. And the drop point and the frame come from the same event or the features stay
blank, because a frame borrowed from the other window scores bodies read before the ball
travelled against a point it only reached later. That refusal costs rows against the raw
availability the 360 audit measures, and the gap between the two is published as the price of
the rule.

What locates the clearance is shared rather than re-derived: the same helper answers it for this
stage, for the freeze-frame audit and for the tracking audit, so what counts as a clearance is
one definition instead of three. It is spelled private and it is API in fact.

One stage stands outside that chain on purpose. `run_first_contact` prices what a camera at the
first contact would add, which is not the question the model answers: its estimand is conditional
on a frame read after the delivery, so nothing it computes may enter X_t or the published column
sources. The T1 geometry replaces the delivery geometry under the same column names, in memory and
inside that driver alone, and no table carrying it reaches disk. Its verdicts are settled in code
against rules fixed before the run, on the pivotal refit interval those rules were written for,
with the normal and percentile forms and a Bonferroni level over the six decisions printed beside
each one. A decision whose bound changes side across the five stability seeds does not decide,
and says so.

Two more stages exist to be argued with rather than to add a result. `run_positive_control` asks
whether this harness can find signal at all, on a target where signal ought to exist and which the
paper does not model: who wins the first contact. It reuses the value stage's own design matrix and
folds, so a pass is about the method and not about a second model. Its shuffled arm is scored
against the prior fitted on that same shuffled target, and the permutation that decides is the
global one, because permuting inside the match leaves each match's rate intact and a model that
tracks only how rates differ between matches would beat a global prior without leaking anything.

`run_team_shrinkage` answers the other half of the reliability question. Split-half says the
per-team rate is noisy; it does not say a named team is no better than its league. A binomial-logit
model with team nested in league, sampled rather than optimised, says what survives once sampling
noise is discounted, and reports the rank as a distribution because ordering posterior means would
hand back a ranking as confident-looking as the raw one. Any fit that misses r_hat, effective sample
size or zero divergences reports its non-convergence instead of a result, permutations included.

Two sensitivities re-read published contrasts without changing them. `run_drop_position` asks
whether the value of winning the second ball from a staked zone survives holding the drop position
equal, since the raw contrast compares the two states over whatever positions each happens to
occupy. It reads the contrast three ways on one set of match-clustered draws: raw, stratified inside
cells of the pitch, and adjusted for position in one regression, and it carries the number of rows
the stratified reading can actually use. `run_short_corner_sensitivity` asks whether the delivery
design's lead over the 360 block rests on the crosses that deliver the short or ground corners (the
`short_corner` flag), at one regularisation so that what moves is the rows and nothing else. It scores
the modelling table, its long corners alone, the crosses alone at the window `s01` uses and at three
others around it, and every corner at its own kick, the rows the table would hold if the reception of a
short pass were its first touch. It also counts what the short corners no cross reached in time
produced, per side, which is how a side's corner goals are told whole.

Six drivers read the tables the stages above wrote. `run_flank_cut` measures why the flank free
kicks start at x ≥ 85: it reads them by band of distance against the long corner, on placement,
production and mechanism, and cuts at 85 and at 90. `run_zone_comparison` compares partitions of
the box at the kick by how well each reads the aim, under a rule registered before any block was
scored: the reference stays unless another reads the aim clearly better, corners decide, and the
flank free kicks replicate, with a contradiction written down rather than resolved by choosing. The
paper's partition, G2, is the cut a club reads, chosen on that ground rather than by the rule, which
does not favour it (`src/zones.py`).
`run_duel_prices` prices winning each of a corner's two duels, the first contact and the second ball,
zone by zone in chain xG, and what ten more points won in one zone would be worth to a team over a
season; a registered rule decides whether that list is published as a product or only as a
description. `eda_club_power` turns the per-corner noise of five measures into the corners and the
seasons a club needs to see a given lift, and says which measure to read, after checking the formula
against the data. `eda_top_teams` asks whether the best sides, and the best corner sides, live after
the first touch, against groups fixed before looking and with every decision held to five stability
seeds. `run_tracking_curve` projects what more tracked matches would buy, as the number of matches an
effect would need and never as a fact.

## Anti-leakage

These rules apply across every stage above, and they are what make the rest credible.

The target is always the observable outcome, never another model's score. Anything read at the
delivery is read in that freeze-frame and nowhere later, and anything computed after the first
contact stays out of the delivery design matrix. The second phase is bounded above as well as
below: without an upper bound, a shot arriving much later gets scored into an episode it never
belonged to. Kinematic derivatives use the causal filter. Validation is grouped by match, with
leave-one-league-out as its external-validity sibling. And an ablation delta whose interval
covers zero is not a finding.

Four things read beside that delta rather than after it. A block is PERMUTED as well as
dropped, because dropping moves the width of the design along with the information, and where
the two disagree the block is collinear rather than useful. A shuffle costs something even when a
block carries nothing, so `run_noise_floor` measures that floor on blocks of pure noise as wide as
each block of the design, and a permuted cost is told from noise only above the 95th percentile of
the floor of its width. The fixed-fit intervals hold the
fitted probabilities, so the step that carries the headline and every ablation delta are also
intervalled with the model REBUILT inside each resample; a delta is read on its own refit
interval in normal form, and the published width ratio is what the minimum detectable effect is
scaled by. That step is then tested against a margin anchored on the reference effect the same
run measured, pre-specified and not moved once the bound was known. And whether it hides help on
some set pieces is asked once, globally and cross-fitted, BEFORE any subgroup is opened and
beside a placebo that permutes the proxy's features, so a verdict with no resolved group contrast
is the reason no cell is opened rather than a gap where the reason should be.

## Visualisation

`src/viz/` carries one house identity: palette, typography, pitch, and the header and legend
blocks. On top of it `plot_corner` draws a set-piece snapshot and `plot_race` overlays the reach
field with its decision frontier. Layout parameters live beside the drawing code rather than in
`constants`, because their effect cannot be read off the line without that code.

`renders/` holds one script per figure: the delivery snapshot from tracking, the commitment
signature from the 360, the second-ball race, the goals from corners by side (the introduction tells
it in words) and the scope figure, which reads only descriptive artefacts so that no model result
enters it. The paper's figures are the others: when the goals of a corner arrive, the analysis overview and the segmentation rule (both in TikZ, under
`renders/tex/`), one real corner event by event and the plan's cells on its freeze-frame, the
conversion of each segment, goals against expected goals, what each piece of information adds, the
plan's premium on the second ball, how the second balls of the tracked corners were won, the top sides against the rest,
the flank free kick's cut at x = 85 and the curves that set the window at eight seconds, for the shots
after the first contact and for the cross after a short corner. A figure that prints a measured number
reads it from the generated record, the example corners read their own events and tracking, and the
curves of the window read the logs of their audits, because no block of the record carries them.
`renders/render_paper.py` builds the paper on Hudl's template from the same record.

## Reproducing the figures

Modules and drivers cite documents that are not published here. Those pointers are the project's
private research base, and the figures behind them are reproduced by running the drivers rather
than by reading the notes. Every published number regenerates from the artefacts under
`data/derived/`, so a figure that moves moves in one place.

`pipeline/run_e2e.py` is the recipe: it runs every driver in dependency order with the arguments
that reproduce each artefact, and `--smoke` runs the same chain on a subset while writing nothing
published. It needs the granted export, which is not distributed here.
