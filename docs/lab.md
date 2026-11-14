# The lab

`tatbot lab` runs studies of what the rig can tattoo. A study asks one question, fixes the print its cells sit on and
the constants it holds, and bounds every setting its trials may vary. A trial is one technique drawn at one settings
vector in one cell. A plan turns proposed trials into the runs `tatbot ros draw` draws, and `tatbot lab read`
measures each drawn cell from the rig's own camera frames: no person's photo, no wipe.

The library is `python/tatbot_lab`; the records are `tatbot_contracts.lab`. It runs in the shared offline drawing
interpreter (numpy, OpenCV, PyYAML) and moves no arm.

```bash
tatbot lab new --id card-3rl --question "What does the 3RL lay at each depth?" --stencil /tmp/stencil-bef888 \
  --cartridges lutin-3rl-bugpin --inks true_black,bright_red \
  --palette true_black=inkcap_large_2,bright_red=inkcap_medium_1,water=inkcap_large_1 \
  --bound trim_mm=-2.5:-1.0 --bound speed_mm_s=2.5:6
tatbot lab propose card-3rl --card lutin-3rl-bugpin
tatbot lab plan card-3rl --stencil /tmp/stencil-bef888 --sheet bef888-a
tatbot lab capture card-3rl --plan p0001 --run run-01 --side before
tatbot ros draw ~/tatbot-lab/card-3rl/plans/p0001/run-01/program/program.json --trim-mm=-1
tatbot lab capture card-3rl --plan p0001 --run run-01 --side after
tatbot lab read card-3rl --plan p0001 --run run-01 --stencil /tmp/stencil-bef888
tatbot lab status card-3rl
```

`plan` prints each run's `tatbot ros draw` command, its depth trim included.

## Records

A study is a directory under `TATBOT_LAB_ROOT`, else `~/tatbot-lab`, never in the repository:

| File | Holds |
| --- | --- |
| `study.json` | the study, `tatbot.lab-study/1`, written once |
| `trials.jsonl` | trial records, `tatbot.lab-trial/1`, appended; the last line for an id is the trial now |
| `plans/<plan>/plan.json` | the plan: its runs, their draw commands and modelled minutes |
| `plans/<plan>/run-NN/` | `artwork.json` (one pen a trial), `inks.yaml` (one resource a trial), `program/`, `cells.json` (`tatbot.lab-cells/1`) and, once read, `reads.json` (the wrist's; the overhead's is `reads-overhead.json`) |

A trial moves through `proposed`, `planned`, `drawn` and `read` (or `failed`, `dropped`), and records the program and
run that drew it. Every record is validated
when it is written, and a batch is written whole or not at all. A study holds the machine constants it was run at:
by default the battery at 6.0 V, which runs the pen at 51 Hz, and its 4.0 mm stroke.

## What a trial can vary

| Setting | Level | Applied by |
| --- | --- | --- |
| `trim_mm` | run | `tatbot ros draw --trim-mm` |
| `speed_mm_s` | program | the compile's draw speed |
| `dose_per_mm` | program | a study's bound only: needle pokes per millimetre of line in a stroke's middle, stored as the `speed_mm_s` bounds that give it at the study's `machine_hz` (51 Hz and 3.5 mm/s is 14.6) |
| `mm_per_dip`, `dip_dwell_s`, `dip_above_ink_mm`, `dip_drain_s`, `dip_drain_above_rim_mm` | resource | the trial's own resource's `dip:` in the inks file, over the cartridge datasheet's |
| `lines`, `length_mm`, `pitch_mm`, `passes`, `angle_deg`, `deposition` | geometry | the technique's strokes |

Every trial states its depth trim and its speed; neither is ever defaulted. A setting the study gives no bounds is
held at its technique's default, or for the dip settings the cartridge datasheet's, and a trial that sets it is
refused. Trials that differ in trim or speed cannot share a run, so a plan draws one run per cartridge, trim and
speed.

Every trial is its own resource. Changing to another resource of the same tool and ink costs nothing at the arm, and
the compiler dips afresh at each resource, so every cell starts on a fresh dip whatever the cell before it drew, at
the cost of a dip a cell. A change of ink rinses.

| Technique | Strokes | Read for |
| --- | --- | --- |
| `line` | parallel strokes, 2 mm apart | darkening and ink width per stroke |
| `carry` | a serpentine of 14 strokes 1 mm apart, 168 mm of line | how far one dip carries |
| `solid` | a square of strokes at the pitch | a fill's darkness and its bands |
| `whip` | strokes fading from full deposition to `deposition` | the fade, bands and speckle |
| `corners` | an L, a zigzag and three short strokes | what stopping, turning and lifting leave |

The compiler dips again every `mm_per_dip`, so a carry cell reads one dip whole only at a `mm_per_dip` above its
168 mm of line; at the datasheet's cadence it is read dip by dip and a dip that outlasts its share reads "at least".
A cell 20 mm wide (`--cell-mm 20`, `lines` 18, `length_mm` 18) holds 324 mm, enough for black at depth. Not settings
yet: the rinse, the pen's angle to the stroke, and dots, which the program format has no way to draw. A whip compiles only for a tool whose datasheet records
`fade_lift_mm` ([tools](tools.md)).

## Cells and sheets

The lab draws on the tag prints the rig tracks, so every cell sits over markers. A cell is centred on one grid
marker and is no wider than the grid's pitch less a millimetre, so inking it costs that marker alone. The corner
anchors, which the overhead reads the print by, carry no cell. The silicone print holds 19 cells, named by their
marker (`m12`), so a cell is the same place on every sheet of one print.

Cells are spent from the edge inward, each next one as far as it can be from those already drawn, so the markers
still readable stay spread over the sheet. A sheet is a physical skin: `--sheet` names it, and a fresh name starts
with every cell free. It is the same skin to every study of its print: a study made for new bounds takes only the cells no other study's live trial holds, and a trial marked `dropped` frees its cell. Trials that find no cell stay proposed for the next sheet.

Within a plan, runs go cartridge by cartridge in the study's order, lightest ink to darkest, so black comes last: a
30 s rinse does not clear a heavy black run. Within a run each ink is a layer, lightest first, rinsed between. A
plan that fails to compile leaves no directory and moves no trial.

## Mounts

Any number of skin mounts can stand on the table, each carrying a sheet of the print, and they can be moved, added
or taken away while a run draws on another. They carry no identity of their own: a sheet is told from the others by
where it stands. `tatbot ros page` says where each sheet the overhead reads stands, in mm in the arm's base, and
`--sheet-at X,Y` on `tatbot ros draw`, `ros touch` and `lab capture` makes the one nearest that point the page.
Once chosen, the page follows its own sheet as it slides, and while the arm hides it no other sheet stands in for
it. Moved far at once, it stands lost until chosen again. Choose a sheet before a run draws on it; a mount being
re-skinned stays out of every run until it is chosen. Name the sheet the same way in the plan (`--sheet`) as on the
table.

## The characterization card

`tatbot lab propose STUDY --card CARTRIDGE` proposes the standard card for a cartridge in the study's inks, the
first being its main ink. In each ink: a depth ladder of four lines across the trim bounds, carry cells at the
deepest and a middle depth, and an A/A repeat of the middle line, which gives each metric its noise floor. In the
main ink: a speed ladder, a whip, corners and a solid patch. In two inks that is 19 cells, one silicone sheet, in six
runs. When the study bounds `mm_per_dip`, the carry cells draw at its top, on one dip.

## The dip card

`tatbot lab propose STUDY --dip-card CARTRIDGE` proposes the card that measures how a cartridge takes ink, in the
study's first ink (or `--inks`' first). It is for a cartridge whose datasheet dip is another's: `dip.card.status:
borrowed`, which `tatbot ros compile` refuses for a design ([tools](tools.md)). The lab may still draw with it. All
its cells are drawn at the middle of the study's depth and speed bounds, so it is one run:

- two `line` cells at the datasheet's dip, an A/A pair: the mark (`ink_width_mm`) and the flood's noise floor;
- a `carry` cell at the top of the `mm_per_dip` bound: how far one dip carries;
- three `line` cells over each of `dip_dwell_s`, `dip_above_ink_mm` and `dip_drain_s` that the study bounds (the
  drain's rung at 0 is the first two cells' dip).

Every cell is its own resource and starts on a fresh dip, so each cell's `flood` is its own dip's. Choose the
shallowest dwell and depth whose flood is within the A/A pair's spread and whose line still holds its darkness, the
shortest drain that brings the rest there, and the carry's `strong_mm`; write them into the datasheet's `dip:` and
set `card: {status: read, study: <id>}`. The study states the bounds:

```bash
tatbot lab new --id dip-7rl --question "How does the 1007RL take red?" --stencil /tmp/stencil-bef888 \
  --cartridges lutin-7rl --inks bright_red --palette bright_red=inkcap_medium_1,water=inkcap_large_1 \
  --bound trim_mm=-2.5:-2.5 --bound speed_mm_s=3.5:3.5 --bound mm_per_dip=40:300 \
  --bound dip_dwell_s=0.25:1.5 --bound dip_above_ink_mm=0.5:3 --bound dip_drain_s=0:3
tatbot lab propose dip-7rl --dip-card lutin-7rl
```

## Capturing a run

`tatbot lab capture STUDY --plan P --run R --side before|after` photographs a run's cells on the rig through
`tatbot ros views`. For each cell the pen is held over its centre as `ros inspect`'s close-up holds it, 12 mm over
the skin, and every wrist D405 the arm carries grabs nine frames there, their per-pixel median kept as
`v<NN>-<role>.png`. Every hold is planned, and its wrist checked against the arm model's range, before the arm moves.
The views aim by the newest page set-up (a draw's or a `tatbot ros touch`'s); the after side names the page the
before side used, so both stand at the same poses. Without a set-up, `--sheet-at X,Y` aims by the overhead tracker
at the sheet of the run's print nearest that point (see [Mounts](#mounts)); those holds hover at the stack's close-up
height, 12 mm (the tracker reads the skin to about 2 mm since its re-registration through the wrist D405), and the
after and dry sides aim at the sheet the before side named. The wrist D405s ride about 150 mm behind the tip, so a
lower hold sees a cell only somewhat finer.
Before leaves the arm holding for the draw; after lands it and marks the run's trials drawn. The frames come back to `<run>/views/<side>/` with `views.json` (the page, each view's
joints and pose, each camera's intrinsics). `--rest-view` takes the one view where the arm stands instead, with no
motion: 2026-11-05's method on the WXAI follower.

The overhead D555's view comes first, unless `--no-overhead`: the median of nine exposures from its owner's capture
over the bus, each newer than the last, as `overhead.png`, taken with the arm at rest so it hides the same part of
the table before and after. `--overhead-only` takes it alone and never touches the arm, so one mount can be read
while the arm works on another. `--side dry` takes the views again once the ink has dried, aimed by the before side's
page, and changes no trial.

## Running a planned run

`tatbot lab run STUDY --plan P --run R --stencil DIR --sheet-at X,Y` takes one planned run through the rig in four
steps:

1. `before`: the capture before it draws, which leaves the arm holding.
2. `draw`: the run's `tatbot ros draw`, with its depth trim, on the chosen sheet, with no inspection, the arm landed
   after, so the overhead sees the sheet from the same rest as before. It states the run's cartridge with `--ee-tool`, so a cartridge the mount does not hold is refused, not
   drawn over: a cartridge swap is asked for between runs, and the run is started again once it is fitted.
3. `after`: the capture after it, from rest as the before capture was, which lands the arm.
4. `read`: `lab read` from the wrist views, then from the overhead's, with the ros node's draw run id recorded.

Each step's exit and time go to `<run>/lab-run.json`. A failed step stops the run, and `--from STEP` resumes at it
with what the earlier steps recorded.

## Reading a cell

`tatbot lab read STUDY --plan P --run R --stencil DIR` reads every cell of a drawn run from its captured views (the
after side, or `--side dry`), or from `--before B --after A`. `--instrument overhead` reads `overhead.png` instead
of the wrist views. Both hold frames by the same names, one pair per view, each pair taken from one pose:
the wrist camera's rest view, or one hold, which the arm repeats.

1. **Registration.** The before frame's markers within 52 mm of the cell give a homography to the page, each marker
   weighted by `exp(-(d / 25 mm)²)` of its distance from the cell. The skin curves, so no single homography fits
   every marker, and the weighting keeps the fit truest where the cell is. A marker the fit misses by more than
   25 px is a bad read and is dropped, worst first. The fit is deterministic: RANSAC drew different inliers on
   different calls and moved a cell by a millimetre.
2. **Alignment.** The after frame is aligned to the before in the image by ECC over the cell and 40 mm of skin
   around it, with every cell of the run left out. It is then sampled through the before frame's homography, so the
   curve bends both alike. Ink over a marker cannot move the registration.
3. **Fusion.** A scan's views are combined per pixel by their median, which drops a wet-ink glint that only some
   views see and samples finer than any single view.
4. **Difference.** A gain and offset per channel, fitted on the skin outside the run's cells, matches the light.
   `dL` is the L\* the cell lost, `da` and `db` the a\* and b\* it gained, and `c` the share of light the ink takes
   (1 − after/before in linear luminance), which blur does not change. Subtracting the before frame cancels the
   print. Hue is read only where the before frame is light: on the violet print, red reads as darkening.

| Metric | What it is |
| --- | --- |
| `noise_dL` | the standard deviation of `dL` on the skin outside the cells: the read's own floor |
| `dL_mean`, `da_light`, `db_light` | the cell's mean change; a\* and b\* on unprinted skin only |
| `per_line` | each stroke's darkening within 0.45 mm of it, its ends left out |
| `ink_width_mm` | per stroke, the integral of `c` across it: the width of an opaque line holding the same ink |
| `carry` | per dip cycle, from the dip, where the level falls below half its peak (`strong_mm`) and below a quarter of it and twice the floor (`visible_mm`), interpolated within the stroke; `outlasted` when the cycle ends first |
| `along` | the darkening in tenths of the strokes' length, and first over last |
| `bands` | the strongest periods across the strokes in each quarter of their length |
| `speckle` | the share of the ring 1.5–8 mm around the strokes darkened by more than 8 |
| `past_end` | the darkening in the millimetre beyond the strokes' far end |
| `flood` | what a fresh dip lays before its load settles: the ink width over the first 3 mm of each stroke that follows a dip, over the strokes' steady width (`over_steady`) and over the starts of the strokes that follow no dip (`over_other_starts`), and the share of the strip 0.5–4 mm ahead of where the pen came down darkened by more than 8 (`ahead_speckle`) |
| `stroke_ends` | every technique: the darkening over each stroke's first millimetre (`starts`, where the pen came down), its last (`stops`, where it lifted) and the millimetre about each vertex turning 30° or more (`turns`), against the rest of its line (`middles`), and each over the middles. A stroke under 3 mm has no middle and counts in none. The dose model (`stats.dose` of the run's program) says how many pokes each took |

Every stroke starts on a needle that dwells, so a start is wider than a middle with or without a dip:
`over_other_starts` is the dip's own share, and a cell needs strokes that follow no dip for it. A dip that floods
nothing reads 1.0 on both.

`--instrument overhead` reads the overhead D555's frames, its markers found as the stencil observer finds them
(small ones upsampled). At about 0.86 px/mm it keeps a cell's mean change only and cannot see a line. Two sheets of
one print in its view give every marker twice, so a capture made with `--sheet-at` is read inside the chosen sheet's
region alone: the tracker's pose of that sheet, which the capture records, projected through the overhead's
intrinsics, 15 mm past the print on every side. `--roi x0,y0,x1,y1` gives a region by hand instead, and `reads.json`
says which was used. On 2026-11-08 the merged rig's first capture read forearm18's 18 mm markers at about 14 px a
side, on both mounts at once.

## Comparing settings

A trial's numbers mean something only against the noise between repeats. Every card carries A/A cells, two trials
at one settings vector; a setting changes a metric only where the difference exceeds the largest A/A difference of
that metric, and a change that makes missing ink or speckle worse by more than their A/A spread is not an
improvement whatever else it gains. That is the rule `tatbot research` promoted DrawingBot baselines by, folded into
the lab on 2026-11-08 with the rest of its paired studies.

Each trial records what drew it: the digest of its run's compiled program when it is planned, and the `ros-draw`
run id when it is read (`--run-id`). The run's `meta.json` holds the software the stack drew with
(`ros_runtime.software_digest`), so a difference between two trials can be checked against a change of software
as well as of settings.

## Searching

`tatbot lab propose STUDY --search N --technique T` proposes N trials aimed at what the study does not yet know about
one technique's marks with one cartridge and ink (`--cartridge` and `--ink`, by default the study's first). It moves
only the settings the study bounds, so it never leaves them.

- **The archive.** A read trial's marks are placed on two descriptors: darkness (ΔL\*, in bins from under 3, which is
  no mark, to over 40) and the technique's own one. That is the median width for `line`, how far the first dip carried
  for `carry`, the speckle share for `solid`, the fade's first-over-last for `whip`, and ends over middles for
  `corners`. A bin's elite is its cleanest trial, the one with the least speckle about its strokes; of two as clean,
  the faster one wins. `tatbot lab map STUDY` prints each archive as a grid of elites, darkest at the top.
- **The surrogate.** Once four trials are read, each descriptor gets a model of the settings: a linear trend
  (deeper is darker) and a Gaussian process over what the trend leaves.
- **A batch** mixes three kinds of trial, and each records why it was proposed in its `hypothesis`:
  - Surprises, once eight trials are read: an A/A repeat of a read trial that the surrogate, refitted without it,
    predicted more than three standard deviations off. A bad read and a real cliff look alike until repeated.
  - Quality: a near neighbour of an elite whose speckle is worse than the median elite's.
  - Novelty, the rest: settings the surrogate puts in a bin no trial holds, or else where it is least sure and
    farthest from every trial. Before there is a surrogate, simply the settings farthest from every trial.

Each distinct depth trim and speed is a run of its own, and so a page set-up. A batch therefore holds at most
`--max-runs` of them (default 3), with trims snapped to 0.25 mm and speeds to 0.5 mm/s. Past that, its trials take the
nearest run already in the batch. When depth and speed are all the study lets vary and every run is taken, the rest
of the batch repeats its own picks, which measures their noise floor. `--seed` makes a batch reproducible.

## How closely it reproduces the 2026-11-05 skin tests

These are the wrist camera's rest frames before and after each test on the silicone print, read by the lab and by
the scratch readers that produced that day's numbers (trim −2.5 throughout):

| Test | Scratch reader | `tatbot lab read` |
| --- | --- | --- |
| Black carry block: strong, visible | 257, 296 mm | 277, 308 mm |
| Red after the black: strong, visible | 178, 197 mm | 187, 197 mm |
| Red, fresh cartridge: strong, visible | 217, 217 mm | 213, 225 mm |
| Whip, 60 mm a dip: speckle, first over last, band period | 0.16, 1.1, 1.6 mm | 0.17, 1.08, 1.57 mm |
| Whip, 200 mm a dip: speckle, first over last, band period | 0.10, 1.3, 5.2 mm | 0.10, 1.33, 5.23 mm |

The per-stroke darkening agrees stroke for stroke. The carry distances differ by less than one 19.7 mm stroke,
because the scratch readers counted whole strokes and the lab interpolates within the stroke where the level crosses
its threshold.

## Limits

- The wrist's rest view resolves about 3.7 px/mm, so a 0.5 mm line is two pixels. `ink_width_mm` measures it through
  the blur, but on synthetic frames it read 0.47–0.54 mm for a true 0.47 mm, and less over the violet print, whose
  light the ink takes a smaller share of. A scan's fusion is meant to sample finer, but how much finer is not yet
  measured.
- Unwiped silicone holds ink on its surface. A read straight after drawing sees that ink, and wet ink may hide a
  fade. Read again when it is dry.
- A held view sees its cell from about 170 mm, near 3.9 px/mm, the pen's tube hiding one side of the tip; a second
  D405 turned 180 degrees about the wrist sees the other.
- The overhead hides nothing behind a tube but sees a 10 mm cell as nine pixels across, and the arm at rest hides
  whatever it stands over: a cell under the arm reads in neither of its overhead frames.
