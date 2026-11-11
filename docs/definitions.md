# Definitions — what each thing is

The project's own vocabulary, the frontier and its segments, the windows, and what the Hudl export
actually gives us. The companion document is `argument.md`, which says what the paper claims.

**No figure produced by this pipeline is written in this file.** Each one points at the section of
`measured.md` that generates it. Three deliberate exceptions: the first two are definitions rather
than measurements, and the third is the measurement that licenses a design exception:

- **Declared constants** (the window length, the publication floor, the pitch conventions, the frame
  rate) are written out, because the value *is* the definition and it lives in `src/constants.py`.
- **The provider-field recoveries** point at `hudl/data_provider.tex`, which is where they are
  computed. They are not summaries of anything in `data/derived/`, so `report_figures.py` cannot
  carry them.
- **The `short_corner` exception** (§5) writes the count of corner kicks a barred field places,
  because that count is what licenses the exception, and `s03_value._SOURCE_EXCEPTIONS` cites it
  from here.

Section numbers are fixed references that the code and the other documents cite: 17 is unused.

---

## 1. commitment signature

Every attacker on the pitch when the corner or free kick is delivered, except the taker, gets one of
three jobs, read from the 360 freeze-frame at the instant of delivery. The taker, the ball and an
attacking goalkeeper caught in frame are tagged separately and are not part of the signature.

The jobs are read off where the player stands, the width of the box included (`src/zones.py`):

**attack_cross** — inside the penalty box and outside the frontal. Position decides it, and proximity to
the delivery target does not enter: that target is `pass.end_location`, which *is* the first-contact
point (§5), so a job that read it would be scored partly from the outcome the plan exists to be judged
against.

**stake_drop** — in the frontal: in front of the box, from six yards short of its line up to it, and
inside its width. It is the ground FIFA's corner study calls the edge of the box. Under G2 the frontal
is G2's edge, which adds the central block of the box between fifteen and eighteen yards from the goal
line, between the lines of the six-yard box.

**hold_back** — anywhere else, further back or out wide of the box. Wherever the ball is going, this
player is not part of the delivery; the job is cover, not the cross.

`commit_index` is the share of the job-holding attackers who stand in the box.

**The plan the model reads is finer than the three jobs.** Each side's bodies are counted per zone of
a partition of the box and its approaches, in one canonical frame: a delivery from the attackers'
right is mirrored onto their left, the side s01's `corner_side` names. C has twelve cells, depth by
band (the six-yard line, the penalty spot and the front of the box, against near post, between the
posts and far post), plus wide, edge and deep. F has the six zones a coach names, six-yard box,
central, near post, far post, frontal and short, plus the rest; the rule that picks between C and F
was registered before either ran, and C reads the aim better than F on it
(`data/derived/prereg_2026-10-01/zones.md`). G2 is Opta Analyst's cut of the box, eleven cells: the
six-yard box's width in three equal columns over two rows, the six-yard one and one that runs to fifteen
yards off the goal line, a cell on each side of that width over the box's depth, and C's wide, edge and
deep, its edge taking the central block from fifteen to eighteen yards. It is the paper's partition,
chosen for the way a club reads the box and not for a score: it reads the aim no better than C
(`data/derived/prereg_2026-10-13/zones_g.md` and its deviation). Both sides are counted on the same cells.
One switch, `HUDL_ZONES`, picks the partition every stage reads, and the table records the one it was
built under. The three jobs are sums of zones, and the frontal is one ground in the switch's partition and
in the jobs: the strip in front of the box under C and F, G2's edge under G2, where the defenders standing
on it are s05's treatment as well.

Whether these jobs and zones are the right way to carve up the pitch is our own judgement, built off
the shape Spielverlagerung's set-piece theory describes in prose; six candidate partitions were
compared against each other before these two ran on (`zones.md`). The export carries no native field
for it either:
checked against the full column inventory, nothing beyond the freeze-frame's own teammate, actor,
keeper and location tags exists.

## 2. congestion

How little free space the bodies in the box leave each other at the instant the ball is struck.
Counting both sides, because a defender obstructs a run exactly as well as a teammate does, and
counting distances rather than heads.

**The distinction from occupation is the whole point.** Occupation asks how many players a team sent
in, which the signature already answers through the jobs and the zone counts. Congestion asks
what that did to the space. Eight attackers spread across the width of the area occupy it heavily and
congest nothing; five stacked around the near post congest it severely. The mechanism
Spielverlagerung describes is distance, not headcount: reduced distances between attackers make tight
man-marking almost impossible, and a crossing route creates traffic that separates a defender from
his mark by a metre or two. A metre or two decides a header.

**The measure does not have to be invented.** Hudl already ships one, on the wrong events, under a
name it never defines. `density` is a sum of inverse distances over the outfield defenders goal-side
of the shot, keeper excluded, recovered by inversion rather than by fitting
(`hudl/data_provider.tex` §sec:density). That formula is already a distance measure rather than a
headcount, so one body at two yards outweighs four at eight, and it counts only the bodies between
the anchor and the goal. Being anchored on a point is what makes it transplantable to any instant we
choose.

**Congestion at T = delivery** is that same kernel evaluated on the delivery freeze-frame instead of
the shot frame, with one substitution: the anchor moves to a fixed point in the box, because
`pass.end_location` cannot be used. That point *is* the first-contact point (§5), and
`att_density_5` / `att_density_10` were barred from the design matrix for reading it. A congestion
measure anchored on the delivery target would inherit the same defect. Pooling the teams is the one
deliberate departure from the provider's rule: the provider counts defenders because it is answering
who can block this shot, and the question here is whose run gets obstructed.

**What the measure is conditional on.** The delivery freeze-frame covers the box and little else, so
the denominator is the bodies the camera saw and not the bodies that were there. A corner where the
camera caught fourteen bodies is not comparable with one where it caught nine without saying so, and
the coverage has to travel with the number: the per-body forms carry it into the model. The body
count itself stays in the table and out of the model, because it is every body the zones count, so
beside the zone counts it rebuilds the defenders in the frontal exactly, the treatment of the
protector's estimate.

## 3. first_contact

**What the delivery is.** A corner kick or a flank free kick is its own delivery, except a short or
ground corner (§5). That one is played to a team-mate, so the first touch after it is a reception that
says nothing about a contest. Its delivery is the first pass the provider tags as a cross, by the same
side in the same possession within `SHORT_CROSS_WINDOW_S` of the kick, and the first contact, the
window, the segments and the freeze-frame are all read at that cross. A short corner no cross follows
in time puts no ball into the box: it stays a corner taken, in the denominators per corner, with no
delivery, no segment and no model row (`measured.md` §scope, `short_no_cross`).

The first contact is the first genuine touch of any kind after the delivery, by either side. Genuine excludes the rows the
provider files on a player who never met the ball: a failed `Ball Receipt*`, the loser's
`Aerial Lost` duel, the losing side of a `50/50` and the keeper's `Shot Faced` and `Goal Conceded`
bookkeeping. A foul is a whistle, not a touch, and stays out too. Everything else counts: a
clearance, a block, a keeper's save or catch, or an attacker simply controlling the ball. The same
rule decides whether anything touched the ball between the delivery and a shot (§4).

Two different questions get asked of this touch and they are easy to conflate.

**won_first_contact** asks which side the touch belonged to. It counts a player calmly receiving an
uncontested ball the same way it counts a duel a defender had to win: both read as the attack getting
there first. An uncontested ball reaching an attacker still counts as the attack winning. Nobody
contesting it is the defence's failure, not a missing case.

Almost every attacking win is filed as a `Ball Receipt*` (`measured.md` §first_contact). **That is
not a contested-versus-uncontested split and must not be read as one.** Because the grammar drops the
loser's `Aerial Lost` row, an attacker who *wins* a header is credited with his own following
on-ball event, which is usually a receipt as well. Whether the touch was contested is not
recoverable from the event type alone: a foul committed while challenging is not tagged as a duel
either.

**Whether a shot counts as direct** asks a different question: did anything happen between the
delivery and the shot at all. For that question the shooter's own touch receiving the ball right
before the shot does not count as something happening; it is part of the shot. So the most common
direct shot, an attacker meeting the corner clean, reads as the attack winning first contact under
the first definition and as nothing happening under the second. Both are correct.

**Two taxonomy corrections the grammar depends on.** `Ball Receipt*` is in the on-ball set: it is the
first real event after a large share of corners, and leaving it out badly under-counts
`won_first_contact` and overstates the time to first contact. And `Goal Keeper` is subsetted to
live-ball actions, because taking every keeper row counts bookkeeping events with no contact and
balls the keeper caught, neither of which is a second ball.

## 4. after_first_touch, and the five segments

Everything that happens once the first touch on a delivery fails to end the passage of play as a
shot. A defender heads the cross clear, a teammate flicks it on, a shot is blocked, the keeper
parries: the ball is still live and the attacking side can still score. This is the paper's subject.

Two things sit outside it. A direct shot is the delivery's first touch already being the shot, so
there is nothing after it to name. And a shot arriving outside the window, or a delivery with no
contact at all inside it, is too late or too little to belong to the episode: a shot twelve seconds
after the corner is open play building on a set piece, not the set piece itself.

Every attacking shot lands in exactly one of five segments.

**direct** — the shot is the first touch. No contact from either side sits between the delivery and
the shot.

**flick_on** — an attacking header redirects the ball before the shot. The header is not the shot; a
teammate finishes what it started.

**prolongation** — an attacking touch keeps the ball with the attacking side before the shot arrives
and is not recorded as a header: a chest, a foot, a knockdown, a short pass, or a touch the data
carries no body part for at all. Absence of a recorded header, not confirmation of a non-headed
touch, is what lands a shot here. The data often cannot tell the two apart.

**after_clearance** — a defending contact comes first: a clearance, a block, or a save that leaves
the ball live. The ball drops loose, the attacking side wins it back, and a shot follows. What decides
the segment is which side's contact came last before the shot, not what led up to it.

**late** — the shot arrives outside the window from first contact, or the delivery produces no
contact at all inside it.

Three of the five are what `after_first_touch` names: flick_on, prolongation and after_clearance.
`direct` and `late` are outcomes a shot can land in, but neither is part of the roll-up.

### How a shot gets its segment

One rule assigns exactly one segment, in this order.

1. A shot outside the window from first contact, or with no first contact at all, is `late`, whatever
   happened before it.
2. Otherwise, a defending clearing contact before the shot outranks everything else and makes it
   `after_clearance`, regardless of what happened before that contact or who touched the ball after.
3. Failing that, no touch of any kind before the shot makes it `direct`.
4. Failing that, the shot is `flick_on` or `prolongation`, decided by the last attacking touch.

Membership is decided by the shot's own timing and team side only. Nothing in the rule checks that
the shot shares a possession chain with the delivery.

The five are mutually exclusive and exhaustive by construction: the rule returns exactly one for
every attacking shot, never zero and never two. `prolongation` cannot contain `flick_on` or be
contained by it, and neither can sit inside `after_clearance`, because a shot with a clearing contact
in front of it is `after_clearance` regardless of the last touch, header included.

**A clearing contact is one that leaves the ball LIVE.** `Clearance`, `Block`, and only the
live-ball keeper actions. A contact flagged `out` is excluded, because a ball off the pitch leaves
nothing to contest. The keeper branch cannot use that flag, since it is populated on a negligible
share of keeper rows against about a quarter of clearance and block rows, so a keeper action is
admitted on its outcome instead, through a whitelist that fails closed on an unseen spelling.

**A forced own goal is a goal the delivery produced, and no shot.** It is filed as `Own Goal For`
on the benefiting team and never as a `Shot`, so a predicate written against `Shot` scores every
one of them as nothing. s01 files it in a segment all the same, so the goal is not lost. When only
the defence deflecting the ball, a clearance, a block or a keeper action, sits between the last
attacking action and the own goal, that action produced it, and the own goal takes the segment of
the shot it followed, or `direct` when no attacker met the delivery. An attacking pass between them,
or a defender who had the ball, leaves the chain to decide, and an own goal outside the window is
`late`. The readers of the frontier count it in the goal counts and goal shares, apart in
`own_goals`, and in no shot count, conversion, share of shots or ratio over xG, because it carries
no xG (`measured.md` §corpus, `by_segment`). The two that price a team's finishing against its xG,
`eda_top_teams` and `eda_club_power`, leave it out of every count, goals included: a forced own goal
is nobody's finishing. The any-shot value of `run_aim_zone`'s robustness grid, registered to count an
own goal inside the window as a shot, counts none under G2. It decides nothing, and under C taking the own
goals out moved no cell of the confirmation grid on corners across zero
(`data/derived/auditoria_2026-10-12/evidence/aim_any_shot_cells.log`).

### Three origins inside after_clearance

The touch the defending side deals with is not always the same kind of touch, and the distinction
matters: a keeper punching a cross clear is a different event for a coach to act on than a keeper
saving a point-blank shot, even though both start an after_clearance episode.

**clearance_on_delivery** — the defending contact is the first touch of the whole passage. Nobody on
the attacking side reaches the ball before it is cleared, blocked or claimed away.

**clearance_on_shot** — an attacking shot is the first touch. The defending contact that follows is a
block or a save of that shot, not of the delivery, and the loose ball it leaves is what starts the
episode.

**clearance_on_touch** — neither of the above. An attacking or defending touch that is not itself a
shot and not itself a clearing contact happens first (a knockdown, a duel, a loose-ball recovery),
and only then does a defending contact clear the ball.

These three are exhaustive: the touch before the clearing contact is either absent, a shot, or
something else. There is no fourth kind of touch to have.

**How they are derived, and the limitation to state with them.** `origin` is a field of `s01`
(`ShotSegment.origin`), assigned inside the grammar where the shot order is already known:
`clearance_on_delivery` where the first contact and the clearing contact are the same event and the
attack did not win it, `clearance_on_shot` where an earlier attacking shot came in the same
delivery, before the shot being labelled though not necessarily before the clearing contact, which
also catches the rebound of a blocked shot, and `clearance_on_touch` for the rest. The descriptive
layer keeps the same rule as an invariant guard over the shipped column rather than as its source.
The partition closes and every cell clears the publication floor (`measured.md` §frontier, block
`origin`). What it cannot see is the body part of a winning touch that is itself a `Duel` row, for
the same reason the flick-on split cannot: the export has no `duel.body_part`.

### A penalty won inside the episode

A foul the referee punishes with a penalty is a whistle and not a touch, so it opens no segment, and
the kick that follows, minutes later and as a set piece of its own, is no shot of the delivery's. A
corner whose second phase wins a penalty is therefore valued as if nothing had happened.
`eda_penalties` finds those penalties so the share after the first touch can be read with them beside
the share without them (`measured.md` §penalties).

A penalty is **won inside the episode** when a foul flagged as a penalty, on either side's record,
falls between the kick and one window after the first contact, or one window after the kick when
nothing touched the ball, which is as far as the grammar looks for that contact. Its **phase** places
the whistle before the first contact, at it or after it. Only a whistle after it joins what follows
the first touch; one before it or at it belongs with the delivery. The **kick** is the attacking
team's next penalty shot in the same period, valued at the provider's xG like every other shot.

## 5. The window, and why it is eight seconds

`SECOND_BALL_WINDOW_S = 8.0`, measured from the **first contact** and never from the clearance. One
clock covers all five segments, which is what lets a direct shot be `late` too.

The value is checked rather than inherited. Read from the first contact, the share of attacking
shots still sharing their delivery's possession chain stays at or above 95% through the ninth second
and drops below it in the tenth. The cutoff sits a second before the last second that holds.

`SHORT_CROSS_WINDOW_S` is bound to the same value: how long after the kick of a short or ground corner
its cross may come and still be its delivery (§3). Read from the kick, the side's crosses stay in the
corner's possession chain through the twelfth second, which the same rule would turn into eleven, but
the routine is interrupted well before: a first cross after the eighth second follows a defending
touch far more often than one inside it, so the wait is the episode's own eight seconds.

Those two curves are the measured quantities in this document with **no generated twin**: they need
the possession chain per shot or per cross, and no registry block carries it, so `report_figures.py`
cannot and there is no `measured.md` section to point at. Their scripts and logs are
`data/derived/auditoria_2026-10-12/evidence/late_boundary_curve.py` and `.log`, and
`data/derived/prereg_2026-11-04/evidence/cross_window_curve.py` and `.log`.

`TRANSITION_WINDOW_S = 8.0` holds the same value and is **not the same decision**. One bounds how
long a shot can still belong to this delivery's own attack; the other bounds how long a defending
shot can still be blamed on this delivery's clearance. They are kept apart so neither moves because
the other did.

`at_risk_horizon_s` carries what is left of the episode after the clearance.

**`pass.end_location` IS the first-contact point**, not where the ball was aimed. The two sit a
fraction of a yard apart in the median and exactly zero apart when an attacker makes the contact
(`evidence.md` §15). Anything anchored on it reads the future of the play, which is why
`att_density_5` / `att_density_10`, `delivery_end_x` / `delivery_end_y`, `pass.length` and
`pass.angle` are all barred from the design matrix.

The two scores the provider attaches to a pass are barred for the same reason, before anything
uses them. `pass.xclaim` is documented as computed from the start location, the end location and
the height. `pass.pass_success_probability` has no specification, so it was measured: a corner is
struck from a fixed spot, and among corners taken from the same side at the same height a grid on
the end point explains most of the score's variation, which says it reads where the ball arrived.
Both are refused by the same executable bar as the end location itself.

The bar is executable rather than reviewable. Every column the design can admit names the export
field it is built on, a column reaching the design without one fails a test, and a source naming a
barred field fails unless the column carries an exception here. That closes both halves of the
failure it guards against: a named column silently absent, and an unnamed source silently present.

**One bounded exception, declared rather than discovered later.** A corner kick is short when it is a
ground pass OR travels less than `SHORT_CORNER_MAX_LENGTH`, so the second branch reads the kick's
`pass.length`. A short or ground corner is delivered by its cross (§3), so in the modelling table the
`short_corner` flag marks the rows that are a cross, and it reads no length there: every one of those
kicks is a ground pass. The branch acts on the population instead. It flags 37 corner kicks of 16720,
struck off the ground and met within fifteen yards, and no cross followed any of them, so all 37 are
corners taken with no delivery, out of the segments and the model. That is the endpoint deciding where
37 rows go, one corner in four hundred and fifty; what the branch buys is the routine itself, which a
watcher knows before the ball is struck, and leaving it undeclared was not an option.

**The flag reads "short or ground", not "short".** It holds every ground corner whatever its
length, so beside the corners played short to a team-mate it holds low deliveries driven into the
box or cut back to its edge, which only the barred end location could tell apart. Prose citing it
says "short or ground", and a "long" corner is one struck off the ground and at least
`SHORT_CORNER_MAX_LENGTH` long. A free kick never carries the flag: met that close it was blocked by
the wall far more often than played short (`constants.FK_FLANK_MIN_X`), and the length branch would
carry that first touch into the design.

## 6. counter

A shot by the defending team, after they clear a delivery and win the ball back, inside the
transition window measured from that clearance, and only if the shot is still part of the same
possession chain the clearance was in. A defending shot outside that chain is the defending team's
next attack, not a counter off this set piece.

Two things carry the idea at different grains. `counter_shot` is the raw fact per delivery: did a
qualifying defending shot happen. `clear_counter` is the outcome label a delivery gets when it
produced no attacking shot of its own and a `counter_shot` did happen. An attacking shot always
outranks it.

**This does not hold up as its own class, and the window was never the reason.** At eight seconds
most of the shots that qualify are already in a different possession, and there are fewer of them
than the publication floor. No window reaches the 95% a finding needs here, and past the shortest
windows widening makes contiguity worse, because it starts pulling in the other team's next attack
(`measured.md` §counter). Requiring contiguity leaves almost nothing: `conceded_xg` is non-zero on
one delivery in the whole modelled corner set. There is also a structural suppression that no
window fixes, since the outcome ladder gives an attacking shot priority, so widening only ever
samples from deliveries with no attacking shot of their own (`measured.md` §counter, `outranked`).

`clear_counter` therefore folds into `clear_safe` wherever the label becomes a modelling target,
`lambda_counter` defaults to zero, and the value expression collapses to its attacking half. This is
**closed, not pending**.

## 7. The four states

After the ball is cleared, the next touch either belongs to the attacking side or it does not, and
the plan either had a `stake_drop` attacker (§1, the frontal at the kick) within `STAKE_RADIUS`, ten
yards, of where that touch met the ball, or it did not. Crossing those two questions gives four
states.

**won_as_planned** — a staker was near the drop, and the attacking side made the next touch.

**staked_but_lost** — a staker was near the drop, but the defending side made the next touch anyway.

**rescued** — nobody from the plan was near the drop, but the attacking side made the next touch
regardless.

**conceded_open** — nobody from the plan was near the drop, and the defending side made the next
touch.

Both halves of the cross are narrow by design. `staked` only asks whether a `stake_drop` attacker
stood close to where the next touch contested the ball, not whether the plan worked in any broader
sense, and with no next touch there is no contest to score the plan against, so it stays empty with
`won`.
`won` only asks who made the very next touch inside the episode, not what happened afterwards: a
touch the attacking side wins and immediately loses back reads the same as one that ends in a goal.
Whether that touch mattered is a separate question, answered by whether a shot followed, which the
phase grammar already tracks. The two are meant to be read together, not folded into one name. A
restart as the next event, a throw-in, free kick, goal kick, corner or kick-off, leaves `won` empty:
the ball went dead, so nothing loose was contested.

This grid is a coach-facing reading taken from the submitted proposal's framing, not a settled
taxonomy.

## 8. Rate, share and conversion

Three quantities at team level all get called "the second ball", and they answer different questions.
Measured on the same teams they disagree with each other.

**rate** — after-first-touch output per delivery taken. The denominator is corners, which the team
controls. This is what a team produces from its own corners.

**share** — the fraction of a team's set-piece shots that arrive after the first touch. This is mix,
not output. A team can post a high share by creating few direct shots rather than many late ones, so
the number moves for reasons that have nothing to do with the second ball.

**conversion** — goals per after-first-touch shot. Finishing inside the segment. The denominator is
already a subset, so this is the thinnest of the three and the one that needs the most care about
cell sizes.

**The thesis is about output, so the rate is primary.** A share can rise while output falls, and any
claim that a team is good at second balls has to name which of the three it means.

`MIN_CELL = 20` is the publication floor. A cell holding fewer rows is printed with its count and a
flag, and never as a rate. `RELIABILITY_FLOOR = 0.75` is the Koo & Li "good" band, and no per-team
ranking ships below it.

## 9. The two levels of positional knowledge

Two levels of positional knowledge exist on this grant, and the gap between them is stated rather
than bridged.

**Event-based.** The whole grant, using the event chain and the delivery freeze-frame. No continuous
positions. This is the core, it covers every match, and it is what makes the work feasible.

**Tracking.** The 30 matches of Broadcast X&Y. The ceiling of what positional data can answer on this
grant, and the only place a velocity-aware race model can be fitted at all.

What tracking buys over event data is itself a measured question rather than an assumption
(`argument.md` §11). The coverage measurement is what quantifies the gap: `measured.md`
§tracking_window for what the transition window shows, §coverage_360 for what the delivery
freeze-frame does not.

## 10. Features at T = delivery

Not repeated here. `methods_second_ball.md` §3 defines every one of them by name, with its lineage.

The rule that governs all of them: **strictly the frame at T = delivery**, over the players the camera
actually saw. Nothing from after the contact enters it, neither a later position nor the location
the shot turned out to have. Kinematic derivatives use a causal one-sided Savitzky-Golay filter,
because the centred version reads the future.

Two names say exactly what the frame can support (§15). `box_block_depth` is the depth of the block,
measured from the two detected defenders furthest from goal wherever they stand, and it moves with a
defender in the frontal, `s05`'s protector; the protector's primary specification leaves it out
(`data/derived/prereg_2026-10-01/protector.md`). `n_def_behind_half` is a constant on this
data, so it is excluded from the feature groups and kept only as the diagnostic that proves why
nothing deeper is measurable.

---

# What Hudl actually gives us

Field-by-field detail is `hudl/data_provider.tex`, which is the private reference for the export.
This section carries what a reader needs before touching the data.

## 11. The two sources

Two local sources, both StatsBomb-schema JSON, no live connection. Volumes are in
`measured.md` §corpus and §coverage_by_league.

| Source | Path | Content |
|---|---|---|
| Big-5 2025/26 | `data/big5_2025_26/<league>/events/` | Five competition-seasons, events + 360 |
| Serie A sample | `data/serie_a_sample/{events_360,tracking}/` | 30 matches, events + 360 + Broadcast X&Y |

Serie A is nested in both: the tracked sample is a strict subset of its own league season.

**The 360 is embedded per event** as `freeze_frame`, not a separate endpoint. `load_360` is a
filtered view over `load_events`.

### Throw-ins: measured, and in the paper as description rather than as a modelled population

The title says set pieces, so a throw-in belongs to the object by name and the exclusion has to be
a measurement instead of a design note, which it is (`measured.md` §throw_ins).

**A long throw is definable, about as cleanly as a flank free kick.** A predicate that asks where the
throw was taken, how long it was and whether it was aerial selects a population nearly all of which
ends inside the area (`measured.md` §throw_ins, `ends_in_box_pct`). Its length is `pass.length`,
which follows where the ball was met (§5), so the length floor reads the defence's first touch in
part. Note that the free-kick rule does not transfer: a throw-in is never a `Ground Pass`, so the
height filter that excludes a ground free kick selects nothing here, and the equivalent discriminator
is keeping the aerial throws only. The threshold was swept over position, length and height rather
than chosen, and the whole grid is persisted so it can be audited.

**And a long throw produces this paper's phenomenon more purely than either modelled population.** Per
delivery it yields fewer shots than a flank free kick, and **the share of its shots arriving after
the first contact is far higher than on corners or on the flank** (`measured.md` §throw_ins,
`by_population`). Its segment mix says why: the
direct shot stops being the largest attacking segment and almost never scores, because a long
throw almost never produces a clean first-touch finish. Something is headed on, or cleared, first.

**What is NOT claimed.** Whether the direct shot from a long throw also under-converts its xG. The
interval excludes parity, and it rests on too few goals to hold, so `measured.md` carries
`direct_obs_exp_is_fragile` beside it. It is reported and never led on, and it is not placed next to
the corner figure as though the two were equally well determined.

### The decision, and why the way it was reached is part of the argument

Long throws enter the **elementary analysis**, where the argument lives and where the shot count
clears the publication floor several times over. They do **not** enter the model. Three reasons,
each measured rather than asserted:

1. **Power.** The whole population carries few goals, and split across five segments the smallest
   cells hold almost none (`measured.md` §throw_ins). Enough to describe, not enough to fit features
   on and validate league by league.
2. **The contrast this paper is built on barely exists there.** Every claim here compares the direct
   shot against what follows it. On long throws the direct shot is not the largest attacking
   segment and it barely scores, so its obs/exp rests on almost nothing. The left-hand side of the
   comparison is close to empty, and the population cannot carry the paper's central test.
3. **It is a different mechanism.** Thrown by hand, no wall, none of the same restrictions, and a
   defensive response that is not the one a struck dead ball provokes. Pooling it into the same
   model would mix two mechanisms; separating it properly would need its own validation end to end.

**And the exclusion is data-driven, which is the point.** `sota.md` §xg-excludes-set-pieces records
three teams DELETING set-piece shots before fitting an xG model and a fourth keeping them only as a
covariate, one of them stating in as many words that goal probabilities could vary for set pieces and
that they were not interested in modelling the effect. None of the four measured the variation. This
project's whole case rests on that being the wrong move. Excluding throw-ins without measuring them
would repeat it exactly. So they were measured first and bounded second, and the paper is entitled
to say so.

**The predicate is published rather than kept.** The sweep, the adopted cell and its precision are
persisted, so the population is reconstructible by anyone who wants to take the question further.
The honest framing is not "left out" but "measured, bounded and handed over".

**The corner-rate sanity band.** `audit_360.py` flags a league whose corners per match fall outside
±30% of a convergent published estimate of **9.5 to 9.7 corners per match**. That band is the
constant's provenance and the reason the gate exists. Two caveats travel with it. The band is
external literature, not our measurement, and its three sources are **unnamed**, so it is carried
here as [UNVERIFIED: the three sources behind the 9.5-9.7 band] until one of us re-sources it.
What the grant itself measures sits inside the band (`measured.md` §coverage_by_league).

**An attacking free kick is a delivery, not a restart**, and two counts of it exist in this repo.
`s01_phases._delivery_mask` is the grammar, and its free-kick kind, `free_kick_flank`, is the paper's
free-kick population: a `Free Kick` pass taken from `FK_FLANK_MIN_X = 85` yards outside the width of
the box and not along the ground, a corner with offside. It carries no length bar, because there the
bar drops the kicks the wall blocks far more often than short free kicks (`constants.FK_FLANK_MIN_X`
has the measurement).
`audit_360.free_kicks` is a separate local heuristic with no height or length test, used only for
the data-provider volume count. The driver prints it flagged `[unverified]`, so nothing is wrong in
the code; the trap is for a reader who opens the parquet and takes the column at face value. Corner
counts agree exactly across both paths.

## 12. Conventions, and the gotchas that are not derivable from the code

- **Three coordinate spaces coexist and are never mixed.** Tracking is metres with the origin at the
  pitch centre, 105 × 68. Events are StatsBomb yards, 120 × 80, origin at a corner flag, with every
  event stored in the acting team's attacking frame. The render space is Opta 0–100 on both axes.
  Where a name carries a suffix it names the unit, `_M` metres, `_SB` StatsBomb yards and `_S`
  seconds; the zone thresholds and the contest radii are StatsBomb yards with no suffix.
- **Tracking is 25 fps**, 40 ms per frame, in long format, one row per player-or-ball per frame. The
  rate is measured on the feed, not taken from the vendor.
- **The tracking JSONL carries only a shirt number and a home/away flag.** `team_id`, `player_id` and
  `is_goalkeeper` are resolved by crossing the paired events file.
- **The two feeds use different match-id spaces** (Hudl `gameRefId` against StatsBomb `match_id`) and
  different club names for the same team. **Matches pair on (home, away, date), never on id.**
- **Tracking is derived from the TV signal, not from stadium optical.** Partial detection and players
  out of frame are normal, not exceptional. On players, `confidence` has cardinality one and carries
  no information anywhere in the feed; `visibility` is the field that varies, and rather less than
  half of the positions are actual detections rather than interpolations. On the ball the two fields
  are one signal under two names, `LOW` on exactly the `ESTIMATED` rows, and about a third of the
  ball entries are interpolations. `ESTIMATED` is the provider's flag and the export does not say
  whether it marks an interpolation between two detections or an inference by the model; this
  project reads it as interpolation. The provider confirms the ball is tracked poorly while it is off
  the ground and recommends no filter.
- **Fifty-nine event columns are presence flags that only ever hold `True`**, `under_pressure` and
  `counterpress` among them. Filtering any of them with `== False` silently returns nothing. This is
  invisible in a non-null rate and is the single most likely way to misread this export.
- **Two distinct 360 freeze-frame schemas exist** and are nowhere documented as two different things.
  The top-level `freeze_frame` identifies the keeper with a `keeper` flag; `shot.freeze_frame`
  instead requires `position.name == 'Goalkeeper'`. One keeper-detection rule cannot serve both.
- **The tracking metadata line embeds the final score.** It is discarded, and anyone parsing that
  line for match context would be reading the outcome into a feature.

## 13. Two provider fields that will corrupt an analysis

Of the definitions this project inferred from field names, most survived a test over the full grant
and nine did not (`hudl/data_provider.tex` §sec:audit). Two of the nine matter here.

**`Goal` does not sit on the shot that scored.** It reads like a per-shot goal flag and matches no
shot at all: it lands on the *following* event, most often the keeper's. Anything counting goals must
use `shot.outcome.name == "Goal"`, which is what this project does.

**`OpposingTeam` names the opponent of the acting player, not of the team in possession.** The two
readings agree on most rows and disagree on `Pressure`, `Clearance`, `Duel` and `Block`, which are
exactly the defensive events a second-ball question is made of. On a corner the actor of the clearing
contact is the defending side, so the two readings name opposite teams on the events this project is
built from.

## 14. `set_piece_phase` is not our second phase

The provider ships its own set-piece phase annotation, and the name suggests it is equivalent to
`after_clearance_shot`. It is not. It fires on roughly four times as many corners as ours, and only a
small minority of the corners it marks are ours (`hudl/data_provider.tex` §sec:spp). What it marks is
a possession that continued past its first ball, with no requirement of a clearing contact or of a
shot, and that is the provider's own definition of its second phase: the initial delivery has been
dealt with and the set piece is still shaping play, a headed clearance recycled into a second cross
being its example.

How the label is produced is known. It is an event-only classifier trained on possessions that human
annotators phased from video, with features that accumulate over the possession up to the event
being labelled, a rule that never moves a possession back to an earlier phase, and an admission
filter that phases only possessions long enough to warrant it. The per-event decision reads nothing
after the event; the training labels were made with the whole possession in view, and the admission
filter reads its length. Two consequences. It **cannot serve as a validation target**, because it
labels a different population and is itself a model's output, fitted with balanced class weights.
And it enters neither the design matrix nor the target at any instant, under the rule that a target
is an observable outcome and never another model's score. It is carried as a cross-check column and
nothing else.

## 15. What the granted data cannot support

The limits that bound what any claim is allowed to say. All measured on the grant; the figures are
in `measured.md`.

- **The modelled population is not a uniform sample of the grant.** Every corner is segmented, but
  only the ones carrying a delivery freeze-frame are modelled, and that share varies by league
  (`measured.md` §coverage_by_league). The modelled corners over-represent the best-covered leagues,
  so a pooled rate is weighted towards them. And leave-one-league-out does not hold coverage fixed
  across folds, so a small league-out penalty is not by itself evidence that league style does not
  matter. On the flank free kicks the share is high and varies little between leagues
  (`measured.md` §free_kick_flank, `coverage_by_league`), so this limit is a corner one.
- **The delivery freeze-frame covers the box and little else.** Mean visible area is a small fraction
  of the pitch, no delivery reaches an 80% coverage share, and the deep band is essentially empty
  across the whole grant, on the corners and on the flank free kicks alike (`measured.md`
  §coverage_360, and the flank's own under §free_kick_flank). **Rest-line height, counter-exposure and
  `n_def_behind_ball_centre` are not computable on this data**, because there is nobody deep to
  measure. This is the limit behind the names in §10.
- **The transition window is visible most of the time, but rarely completely**
  (`measured.md` §tracking_window). Any kinetic claim has to be bounded by this, and detection
  density stays a **validation** signal, never a training input.
- **The ball the race reads is interpolated on most published contacts.** A corner delivery is
  airborne from the kick to the first contact, and that is where the provider says its ball
  tracking is weakest, with no filter to recommend. The race locates the contact, the loose point
  and its own recovery label on the ball row as emitted, and on about two thirds of the races it
  publishes that row is an `ESTIMATED` position rather than a detection (`measured.md` §race).
  Every arrival time, recovery probability and race label carries that bound, and every race
  carries its two flags in the record. A mask would empty the race, so the sample is not cut:
  the same figures scored over the corners whose ball was detected sit beside the published ones
  as a bound at the adopted temperature, never as a second model.
- **The aerial rating only exists where the duel was contested.** It sits on the loser's
  `Aerial Lost` row, as the loser's rating and the winner's opponent-rating, and nowhere else. So it
  is present for contested headers and **absent for the free header**, which is exactly the case a
  coach most wants explained. That bounds what the aerial block can contribute.
- **The four states are not known on every cleared corner** (`measured.md` §situation,
  `four_state_known_pct` and `four_state_unknown`). The cross needs both halves, so a corner
  whose delivery carries no freeze-frame has no `staked`, and one where nothing contests the
  loose ball inside the episode has no `won`. Neither absence is a labelling failure; both are
  coverage. Every share in that grid is therefore over the known subset and must not be read
  against the cleared count, and the larger of the two missing halves is the delivery frame,
  whose coverage varies by league (`measured.md` §coverage_by_league), so the grid inherits the
  same skew the first limit above already records.
- **There is no per-corner uplift.** What ships is an average effect, and the paper must not
  imply otherwise.
- **The body part of the last attacking touch is sometimes missing, and a missing one files the shot
  as a prolongation rather than a flick-on.** The export has no body part on a duel, which is why
  s01 expects some flick-ons to land there. Measured, that route barely exists: almost every such
  touch is a recovery, a pass, a dribble or a block, and the only one following a duel follows a
  tackle rather than an aerial contest. Only the passes could hide a header, and moving every one of
  them across leaves the flick-on's lead over the direct shot in place.

## 16. Anti-leakage, the rules that make the rest credible

- The target is always the **observable outcome** (second phase, shot, xG in the window), never
  another model's score.
- **T = delivery.** The commitment signature and the 360 features are read only in the freeze-frame
  at the delivery, never after.
- Φ (the xG of the second ball, the xG conceded on the counter) is computed in frames **after** the
  first contact and never enters X_t.
- The second phase is an attacking shot after a defensive clearing contact, **inside the window**,
  measured from the first contact, with an upper bound. Without that bound, look-ahead gets in.
- Validation is grouped by match (GroupKFold), and leave-one-league-out is its external-validity
  sibling. Temperature scaling is fitted on **inner folds only**.
- Every interval resamples **matches**, because corners inside a match share teams, coach, referee
  and scoreline, except the team-reliability one, which resamples teams. The bootstrap holds the
  out-of-fold probabilities fixed, so it covers evaluation-set sampling error only and is **too
  narrow**. How much is measured rather than asserted: the step that carries the headline, the
  delivery-against-360 head-to-head, every ablation and the freeze-frame steps are intervalled again
  with the model rebuilt inside every resample, the width ratio is published beside the step
  (`measured.md` §validation, block `refit`), and a block is a finding only on its own refit
  interval. The form to read there is the NORMAL one, the point plus or minus half the refit width.
  A resample thins the distinct matches, which moves every draw by the learning curve rather than
  by the estimator's error: the percentile form carries that shift and the pivotal form moves it to
  the other side of the point. Both travel beside the normal form, and the draws are checked for
  the symmetry it assumes (`measured.md` §refit_readings).
- **A null needs its resolution attached.** An interval covering zero says what was not found; the
  minimum detectable effect says what could have been. Both travel with any null this project
  publishes, and the MDE is scaled by the width ratio above rather than read off the fixed-fit
  curve (`measured.md` §validation, block `power`).
- **A null is also tested against a margin, and the margin is anchored.** The reference effect
  measured in the same run sets it, it is pre-specified at a tenth of that, and it is not moved
  once the bound is known. What travels beside the verdict is the fraction of the reference the
  block is bounded above at, so a reader can apply the margin they would have chosen
  (`measured.md` §validation, block `non_inferiority`).
- **Subgroups are gated before they are opened.** Whether a pooled step hides structure is asked
  once, globally and cross-fitted, beside a placebo that runs the same test on the proxy's features
  permuted on the same partitions, and only a verdict with a resolved GROUP contrast opens a cell
  (`measured.md` §validation, block `heterogeneity`). A predictable difference with an unresolved
  contrast means something varies without saying for whom, and no cell is opened on it.
- Ablation deltas are **paired** per row inside the same resample, never two marginal intervals
  subtracted.
- `class_weight="balanced"` optimises balanced accuracy, not log loss, and loses to the class prior
  on this data. The default is `None`.
- s08 is read only at or after the drop, never in X_t.

---

## 18. The aim zone, and why the end zone is not it

`pass.end_location` is the first-contact point (§5, measured in `evidence.md` §15). When the
defence reaches the delivery first, it records where the defence got to the ball. A map of chain
value by **end zone** therefore prices two things at once: where the taker aimed and where the ball
was intercepted. *un-xPass* (Robberechts et al. 2022) states the same principle for open-play passes.

**Zone grid.** Canonical, with the corner struck from the attackers' left, the y = 0 side; a corner
from their right (s01's `corner_side == "right"`, y ≥ 40) is mirrored. Which flag is which is read
off the swing, not a convention: a left-footer's inswinger comes only from the attackers' right.
The base grid splits the band past the six-yard line, across the width of the box and so wider than
the six-yard box, and the band from the penalty spot to the six-yard line at the posts into near,
central and far, and calls the front of the box, outside it and wide of it `other`. A coarse grid
(near, central, far) and a fine one (two more bands outside the posts) are the robustness grids.
The rows of short or ground corners (§5) are out: their delivery is a cross struck from the wing a few
seconds after the kick, not a corner aimed from the flag, and the statistics were registered on the
corners struck from the flag.

**Kick block.** s02's delivery frame, counted per zone of the partition the switch names (§1) for
the attackers holding one of the three jobs (taker and keepers out) and for the outfield defenders,
plus the size, mean position and spread of each side inside the box. T = delivery, as §10. D1 and D2
were registered on C's twelve cells; under F or G2, the paper's partition, the same rules read that
partition's zones.

**Aim.** The zone the kick block and the delivery design predict, from a multinomial logistic model
taught only on corners whose first contact the attack made, where the end zone IS the aim. It is
scored out of fold by match, with its regularisation chosen inside each fold.

**Where the end zone may enter.** In two places, and neither is X_t: as the aim model's target, which
is an observable outcome, and as the label of the naive map. The aim model's inputs declare their
source fields, and a barred one (§5) is refused by the same executable bar.

**The two statistics** (`s02b_aim`, pre-registered before the confirmation leagues were read):

- **D1**, the kick block carries the plan: the paired out-of-fold zone log loss of design plus kick
  block, minus design alone, on attack-won corners. Negative means the kick block reads the aim.
- **D2**, the end-zone map is biased: the near-post minus far-post difference in chain xG per corner
  with corners grouped by end zone, minus the same with corners grouped by inferred aim. Negative
  means the near-post discount of the end-zone map belongs to where the defence got the ball.

The map by inferred aim is reported and never decides. It is called flat only inside a margin fixed
in the pre-registration, and otherwise "not distinguishable from zero". A soft version, which solves
each corner's value as its aim probabilities times a value per zone, travels beside it because a
hard argmax flattens any map read off a noisy aim.

## 19. The modelled target: five competing outcomes

The value model (`s03`) predicts one of five mutually exclusive outcomes per delivery. The class is
set by what happened first: a delivery whose first attacking shot is direct stays in the direct class
even if a rebound is shot later. The eight outcomes the grammar labels on a delivery fold into these
five, the three after-first-touch segments into one class and `clear_counter` into `clear_safe` (§6).
A short corner no cross followed carries a ninth, `no_delivery` (§3), and never reaches the model.

**first_touch** — the first attacking shot is the first touch, inside the window (§4, `direct`).

**after_first_touch** — the first attacking shot comes after the first touch, inside the window: a
flick-on, a prolongation or a shot after a clearance (§4).

**late_shot** — the first attacking shot arrives outside the window, or the delivery had no contact
inside it (§4, `late`).

**clear_safe** — no attacking shot, and a defensive contact left the ball live inside the episode:
there was a second ball and nothing came of it.

**no_second_ball** — no attacking shot and no live clearance, so the episode ended without a second
ball. The class is named for what its deliveries share and not for who ended them, because it is not
mostly the defence winning the ball: it holds deliveries the attack reached first and then kept or
lost without a shot, deliveries with no contact inside the
window, clearances that left the pitch, and balls the keeper took. s01 labels the same deliveries
`no_second_ball`.
