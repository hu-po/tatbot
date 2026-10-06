---
summary: Hardware-independent Tatbot simulation workflow
tags: [simulation, testing]
updated: 2026-09-27
audience: [dev, contributor]
---

# Simulation

Use `python/tatbot_sim/` for offline development, dataset-shape checks, and
control experiments that do not connect to an arm or camera.

## Ownership

| Responsibility | Shared owner |
| --- | --- |
| Robot, tool, surface and sensor selection | Existing registries resolved into `ResolvedConfig` |
| Observation names, units and availability | `tatbot_contracts` and `ObservationBuilder` |
| Supported production reference motion | Cartesian compiler and C++ planner |
| Synthetic episode lifecycle | `Episode`, used by generation, evaluation and presentation |
| Physics, contacts and rendered sensors | `ManiSkillWorld` |

Physical drawing runs on the ROS 2 stack in `ros/` (`tatbot ros draw`); its
own simulation is its `hardware:=mock` launch, described in `ros/README.md`.
This package does not drive that stack. Dataset batching is an episode
scheduler: it produces independent episodes that share world construction,
observations and supported production references.

A future MuJoCo implementation should provide world construction, joint stepping,
contact feedback and camera samples behind these boundaries. Keep its engine
code and dependencies separate while reusing the robot/sensor registry, planner,
observations and dataset writer. The adapter interface can still evolve when a
second engine exercises it.

## Quick start

```bash
cd python/tatbot_sim
uv sync --extra maniskill
uv run --extra maniskill python -m tatbot_sim.factory --list
```

Keep generated episodes and renders outside the repository. Include the source
revision and simulator configuration in any artifact manifest.

Measured fixed-camera configurations require matching camera-bundle and
robot-world calibration IDs. The simulator's root is the follower arm base;
the solver's root is the full rig URDF. `tatbot_sim.calibration` composes that
registration with the canonical arm mount before converting camera poses.
The OpenCV optical axes (right, down, forward) are then converted to SAPIEN
camera axes (forward, left, up). The fixed PoE camera body meshes are not part
of the single-arm simulator model. The shared translation-only draw planner
derives its mount offset from the URDF and refuses a rotated mount it cannot
represent.

The full rig's `rig_center` coincides with its root at the arm-pair midpoint;
+Y is the robot's own left side when looking forward through the fixed cameras.
The single-arm simulator and measured palette poses remain in follower-base
coordinates. A configured synthetic root-frame point is still an explicit
development fixture, not an estimate of the physical rig midpoint. The
simulator's palette is the installed one, loaded from `urdf/palette.urdf` and
`config/palette.yaml` (six caps), at the synthetic scene pose in
`config/palette_geometry.json` ([palette](palette.md)).

Posed-body work (`sim compile`, `sim resolve`, and their tests) resolves every
anchor through the browser's TypeScript InkLang resolver, so a sim host also
needs Node.js 22 and the inkmap dependencies (`npm ci` in `web/inkmap`;
`scripts/check sim` installs them when it can and otherwise reports a named
skip). A user-local Node under `~/.local` is enough.

The private engineering checkout reads its qualified workspace and arm profile.
The public checkout instead falls back to the explicitly simulation-only files
under `config/examples/`. They make imports and geometry-only development
reproducible; they are not calibration, controller limits, or permission to run
hardware. Offline generation continues with nominal geometry when a qualified
calibration is unavailable, and records a development-only warning.

`scripts/check sim` and offline generation do not require a fresh tool
calibration. When only nominal or synthetic geometry is available they run,
stamp `qualification: development`, and retain the reason in
`geometry_warnings`. Pass `--require-qualified-geometry` only when the artifact
is explicitly meant to demonstrate calibrated contact geometry.

## Explicit world construction

`import tatbot_sim` does not select a tool, register a Gym environment, build
assets, or import the physics engine. Configuration and geometry helpers can
be imported with only the standard library. Construct a world explicitly:

```python
from tatbot_sim.resolved import resolve
from tatbot_sim.env import TatbotDrawEnv

config = resolve(tool_id="lutin-ballpoint-dot", seed=7)
world = TatbotDrawEnv(config=config, num_envs=1, obs_mode="rgbd",
                      control_mode="pd_joint_pos", sim_backend="cpu")
try:
    observation, info = world.reset(seed=7)
finally:
    world.close()
```

Resolution snapshots the existing tool, substrate, camera, calibration, timing,
randomization and ink registries. Mutable values are copied at the boundary.
Pass the same resolved configuration to the world, planner and IK solver;
changing an environment variable later cannot change that world. Derived robot
files are keyed by input content, and construction refuses changed source files
after resolution. Factory and cinematic commands select their distribution's
tool explicitly and no longer restart Python.

Layout, camera mounting, lighting, action noise and RGB-D corruption use separate
seeded streams. Generated tool metadata and policy evaluation results retain the
resolved configuration and source digests. Engine reset seeds still identify
individual episodes within a run.

Preview and cinematic cameras, lighting and surface appearance are instance
options. Cinematic mounted views use separate `cine_*` cameras; they do not
resize policy images. A historical lower-wrist shot requires the explicit
historical camera profile. Ordinary construction uses `TatbotDrawEnv` directly;
`gym.make("TatbotDraw-v0")` is no longer registered as an import side effect.

## Shared episode runtime

Generation, policy evaluation, preview and cinematic rendering use
`Episode` with `ManiSkillWorld` and `ObservationBuilder`. The runtime owns reset,
articulation rebinding, material initialization, staged pose placement, reference
refinement, control ticks and completion. Each command supplies its plan or
policy actions and consumes the resulting observations. Every consumer shares
the resolved world and production reference primitives; batching remains an
episode-scheduling concern.

The observation contract lives in `tatbot_contracts.observations`. Its named
seven joint positions and seven external-effort channels keep the existing
LeRobot ordering. `contact` exposes the simulated contact estimate, with the
resolved calibration when present; `unavailable` masks those channels to zero
and records that they are unavailable. Policy evaluation explicitly uses the
latter profile and clean RGB-D. Synthetic generation uses the contact profile
and the recipe's sensor corruption. These are recorded configuration choices.
Simulator contact truth stays separate from policy features.

A tick creates one observation. Contact-feature reduction, saved depth and
policy consumers reuse that sample, including its noise. Integer depth remains
millimetres with zero invalid. The writer retains its existing log-depth codec.
Pixel noise is keyed by camera, episode and control tick, so keeping every third
video frame cannot change samples at the retained ticks. Presentation cameras
remain outside the policy profile.

Runtime time starts at reset and advances by `1 / control_hz` for each completed
step. Dataset timestamps keep LeRobot's zero-based first sample; metadata records
the actual capture ticks and the sampling rule. Completion records whether
the plan horizon or the backend ended the episode. A video truncated by its
frame limit records an unfinished episode.

Episode metadata includes scene/reset seeds, actual camera mount poses and
sampled lights, sensor response parameters and seeds, engine versions, and
measurements of the final solved reference. Scene construction and episode
placement use separate random streams. A scene retained across resets keeps its
recorded construction seed; an explicit rebuilding reset redraws it from the
requested seed. This preserves reproducibility without forcing reconstruction
for every batched run.

### Production reference primitives

The factory can consume the production Cartesian compiler and C++ joint
planner. Select it with `--production-draw-config PATH`, pointing
to an existing `tatbot.draw-config/1` containing the tool and drawing speed:

```bash
tatbot sim generate paper-draw -- --out-dir /tmp/production-reference \
  --design design.json --production-draw-config draw.json \
  --no-tool-calibration-jitter --dr.latency.obs-delay-steps 0 0 \
  --num-episodes 1 --num-envs 1
```

This path currently supports nominal ballpoint geometry on an undisplaced paper
plane, ordinary drawing intent and zero observation delay. The supplied draw
configuration owns approach, speed, easing and carriage settings. The production
compiler owns chunk limits, pen-up travel and reference motion. Native planner
refusals remain refusals; the synthetic expert does not repair or perturb the
accepted reference. Its IK is used only to propose the initial simulated pose
and to measure the final reference. Simulation quality checks still apply.

Every native command executes at 400 Hz with three physics substeps (1200 Hz).
The existing ManiSkill position controller follows the native position
reference. Native velocity references remain in the retained JSON; this adapter
does not apply their feed-forward term or claim vendor-controller equivalence.
The configured 30 Hz camera captures on ticks 14, 27, 40, and so on: the first
controller tick at or after its deadline, less than 2.5 ms late. No command or
pen marker is resampled. LeRobot video/data retain their nominal 30 Hz grid;
`run_meta.json` records each episode's actual `capture_control_ticks`, and
contact-feature derivatives use those actual sample times. The horizon flags
retain their existing duration unit of 30 Hz frames.
Episode `steps_planned`/`steps_executed` count controller ticks;
`frames_recorded` and sidecar lengths count camera samples.

`meta/references/` retains the surface, draw configuration, Cartesian CSVs and
native joint-plan JSONs. Episode motion metadata identifies the planner, source
digests, chunk boundaries, numeric conversion to float32, and absence of expert
action noise. The plane and initial pose are explicitly simulated geometry.

This is reference-primitive reuse in the data factory. The factory path does
not cover palette visits, and it does not claim that ink deposited at 400 Hz is
numerically identical to the synthetic 30 Hz material model. Engine contact and
material behavior require their own measurements.
In the initial CPU replay, controller lag kept measured contact briefly during
a commanded lift. The privileged timeline preserves that discrepancy and its
existing audit reports it. A valid export and matching native references do
not establish drawing-quality acceptance.

## Camera profiles

`deployment` is the default sensor profile. It resolves physical arm assignment,
RGB-D dimensions and cadence from the vision registry, with the checked-in
example as the offline fallback. The follower environment renders the right
arm's one wrist view (`wrist_upper` in the current profile). It does not attach
the left camera to the follower. Camera mounts come from the canonical URDF;
intrinsics derived from nominal field of view are labeled nominal in metadata.

Use `--sensor-profile legacy-two-view` explicitly when reproducing historical
datasets or checkpoints with `wrist_upper` and `wrist_lower` on the follower.
This adds the historical lower camera geometry and does not describe the
current installed robot. Generation, preview, and policy evaluation use the
same selection. Presentation views never enter the dataset's camera features.

```bash
tatbot sim generate paper-draw -- --out-dir /tmp/current-paper
tatbot sim generate paper-draw -- --out-dir /tmp/historical-paper \
  --sensor-profile legacy-two-view
```

Policy evaluation and no-arm wire probes compare the checkpoint's exact image
keys with the selected profile before querying actions. For a server-side
checkpoint, pass `--checkpoint-config /path/to/config.json` with its local
configuration. Simulation evaluation records that config's digest. A missing
view is an error; no image duplication or relabeling fills it.

## Substrates

`config/substrates.yaml` is the one record of what the tools work on. The sim
sizes its geometry and its texture from it, Inkmap's Paper and Cylinder
workspaces start from it, and the real workspace records its plane against
the same numbers. Three substrates exist:

| Substrate | Presentation | Size | Printed |
| --- | --- | --- | --- |
| `paper_pad` | flat pad, 10 mm thick | 190.5 × 279.4 mm (7.5 × 11 in) | white, faint blue ¼ in (6.35 mm) square grid |
| `paper_cylinder` | rigid cylinder, ⌀85 mm | 190.5 mm (7.5 in) long | the same grid all the way round |
| `silicon_skin` | flat sheet or wrapped | 140 × 185 mm | nothing |

A tool datasheet names its default substrate and the others it admits: the
ballpoint draws on either paper fixture, the laser and the 3RL only on the
skin. `TATBOT_SUBSTRATE=paper_cylinder` selects an admitted alternative for a
run; naming one the tool does not admit is refused rather than substituted.
The paper cylinder's canvas is the whole outer surface except the bottom
quarter it rests on — three quarters of the circumference, 135° either side
of the crest — along the full length of the cylinder; the bottom and the end
caps are textured but never drawn on.

## Material and surface profiles

For a sheet, material and shape are independent scenario axes. The
`paper-draw` recipe uses flat paper by default, while both silicone recipes
balance `flat` and `cylinder` members inside each vectorized batch. Sampled
cylinders run along the long canvas direction and wrap the short direction at
a 75-110 mm radius. A cylinder-shaped substrate is not sampled: the paper
cylinder pins the profile to `cylinder` at its own 42.5 mm radius, and a
`balanced` request on it is refused. Every episode records `surface_profile`
and `surface_radius_m`.

Any sheet supports an explicit override:

```bash
tatbot sim generate paper-draw -- --out-dir /tmp/paper-cylinder \
  --dr.surface.profile cylinder
tatbot sim generate skin-tattoo -- --out-dir /tmp/skin-flat \
  --dr.surface.profile flat
TATBOT_SUBSTRATE=paper_cylinder tatbot sim generate paper-draw -- \
  --out-dir /tmp/paper-cylinder-fixture
```

`balanced` means both profiles occur in each batch with at least two
environments. Curved visual geometry and the mathematical contact surface are
generated from the same chart. Curved profiles remain kinematic-contact
development data until their collision mesh is separately qualified.

## A dataset to read without generating one

[`tatbot/sim-paper-draw-demo`](https://huggingface.co/datasets/tatbot/sim-paper-draw-demo)
is a published shard of the `paper-draw` distribution — 8 episodes, 2836
frames, wrist RGB and depth, generated with seed 0 and no flags beyond the
recipe. Use it to inspect the dataset shape, feature names and metadata that
this repository produces before running the factory yourself:

```bash
uv run --project python/lerobot_robot_tatbot python -c "
from lerobot.datasets.lerobot_dataset import LeRobotDataset
ds = LeRobotDataset('tatbot/sim-paper-draw-demo')
print(ds.meta.info['total_episodes'], 'episodes;', ds.meta.info['total_frames'], 'frames')
print(sorted(ds.meta.features))"
```

With a locally qualified workspace and arm profile, regenerate an equivalent
shard with:

```bash
tatbot sim generate paper-draw -- --out-dir <dir> --num-episodes 8 --num-envs 4 --seed 0
```

The public placeholder profile cannot reproduce that shard's qualified contact
geometry or calibration jitter. It can still generate development data using
nominal tool dimensions, with the basis and warning recorded; do not relabel
that output as calibrated evidence.

Each `paper-draw` invocation represents one fitted session: its seed chooses a
single persistent tip offset within the recorded calibration uncertainty, and
the run metadata records both the central calibration and the applied offset.
Across shard seeds this varies plausible seating/calibration geometry while
preserving exact agreement between the visible tip, TCP, collision, and marks.

## Drawing a portable design

`--design FILE` draws one `tatbot.inkmap-design/1` — a plane or cylinder chart
design, as Inkmap's Paper and Cylinder workspaces and
[`tatbot design place`](design.md) export it — in place of the shared
collection, for the artwork and erase tasks. The design's strokes, rotation,
mirror and scale are its own; the trajectory builder, the charge model and
its dips, the reach mask, the cameras and the writer are the factory's.

```bash
tatbot sim generate paper-draw -- --design design.json --out-dir <dir> \
  --num-episodes 1 --num-envs 1 --horizon 20000 --dr.ink.dips
TATBOT_SUBSTRATE=paper_cylinder tatbot sim generate paper-draw -- \
  --design cylinder-design.json --out-dir <dir> --horizon 20000
```

The design and the substrate have to be the same kind of surface: a plane
design goes on a flat substrate as it is, a cylinder design on the paper
cylinder of the same radius, turned a quarter so that Inkmap's `u` (along
the axis) becomes the canvas `y` the simulator runs along the axis and `v`
(arc from the crest) becomes `-x`; both charts are right-handed about the
outward normal, so the artwork keeps its handedness. A mismatch is refused,
never bent to fit.

`--design-placement authored` (the default) keeps the anchor Inkmap saved,
about the canvas centre, and refuses a placement the tool cannot be held
normal to; `sampled` recentres the artwork and draws an offset inside the
reach envelope per env, the way the collection is placed. A design is never
shrunk to fit an episode: the run refuses up front, naming the `--horizon`
its strokes need (a 70 × 75 mm filled design at a 0.3 mm hatch is roughly
3 m of line and 460 s at the ballpoint's pacing, so the default 300 s artwork
horizon will not hold it). The ink footprint is the design's own stroke
width. Every episode records the design's digest and placement under
`artwork` and `program.portable_design`, and the run metadata under `design`.

`scripts/sim_preview.py --design FILE --task artwork` renders the same plan
from the wrist, third-person and top-down cameras without writing a dataset,
and `tatbot sim cinematic -- --portable-design FILE --task artwork` shoots it
path-traced, at social sizes, from the staged cameras aimed at the drawing.

The GPU physics backend also let a fixed-base articulation's root creep
(3.1 mm in 30 s on an idle arm, quadratic in time; none on the CPU backend).
The environment now re-pins the robot root every control step, and
`python/tatbot_sim/tests/test_root_pin.py` holds it there on any node with a CUDA device.

## What simulation proves

Simulation can validate pure transforms, schema handling, deterministic replay,
and software integration. It does not prove camera calibration, contact force,
e-stop behavior on physical hardware, or safe human use.

## Posed-body scenarios

Inkmap placement files describe design intent on a canonical rest-body
surface. `config/inkmap/tattoo-scenario.schema.json` describes one resolved
offline simulation realization: body and rig identity, pose, body/world and
robot/world transforms, support fixture, declared tool, immutable SVG, and the
derived face/barycentric stroke trace.

The editor also exports `tatbot.inkmap-sim-bundle/1`. `sim compile` recognizes
this immutable request and emits the separate typed v3 scenario schema, binding
the source bundle, InkProgram, and tool-profile digest. Multiple placements
require `--placement-id ID` for drawing compilation. Pose/tool/seed/world
overrides refuse rather than mutate the bundle. The v3 color/label renderer
and compiled-preview import preserve the typed program and its surface binding.
See
[design format](design-format.md) for the source, validation, and asset rules.

[InkLang](inklang.md) is the only semantic placement resolver. It turns the
description plus the fixed model/identity binding into a surface-bound
resolution before scenario compilation. Simulation consumes that JSON through
a thin Node.js 22 client; it does not parse InkLang or choose another site
anchor in Python.

The split is intentional. A placement survives pose changes; a scenario is
fully replayable. Inkmap region `uv` is semantic and normalized, so it must not
be used as a metric drawing chart. The simulator maps the frozen metric program through a
surface trace and then skins those face/barycentric points into the selected
pose.

Named body poses are kinematic and static within an episode. This is a geometry
and planning model, not a soft-tissue, breathing, or dynamic-person model. The
checked-in SOMA GLB preserves the indexed mid-face address as an expanded
render/picking view; its pose cache stores generated face vertices for every
named session pose. Browser and Python loaders verify model, identity,
topology, rest-surface, asset, and pose digests before using those bytes.

The optional mechanics ladder is separate. Rigid contact remains the admitted
reference until synchronized, calibrated non-human phantom data qualify a
compliant candidate. Synthetic mechanics fixtures exercise schemas, fitting,
uncertainty, out-of-distribution refusal, solver energy, and gradients, but do
not establish a physical contact model or authorize force/depth control.

Materialize one placement without launching SAPIEN:

```bash
tatbot sim compile config/inkmap/examples/forearm-placement-v6.json -- \
  --pose reclined-left-arm-supported --seed 42 --output /tmp/forearm-scenario.json
```

Compilation is CPU-only. It verifies body, rest-surface, rig, design, and
placement identities before writing, and it fails explicitly on unsupported
artwork constructs, non-manifold/open surface exits, or topology discontinuities.
By default the patch normal is aligned to robot +Z and its +u axis uses the
validated `pi` robot-world yaw; `--target-world-m` and `--patch-yaw-rad` expose
that constrained body-to-robot placement. Dataset generation recomputes every
trajectory's FK and refuses the scenario if any target exceeds the 1 mm IK
residual gate.

Materialize a deterministic coverage suite before launching the simulator:

```bash
tatbot sim sample --count 64 -- \
  --output-dir ~/tatbot-sim/scenarios/seed-42 --seed 42
```

A suite that is at least as large as its site list promises to cover every
site, and that needs two slots per pose: with the default five poses, ask for
at least 10 scenarios (or fewer than the six sites, which makes no coverage
promise). Smaller requests fail up front with the minimum named instead of
dying mid-run.

`--sample N` is the same module as `compile` (`tatbot_sim.inkmap.cli sample`);
because it always builds the `body-tattoo` distribution, it declares the
nominal `lutin-3rl-bugpin` tool independently of the workstation's fitted or
calibrated tool. An explicit contradictory `--ee-tool` is rejected. The
nominal geometry is recorded as a development warning and never blocks this
offline workflow on a calibration gate.

The sampler balances five tattoo-session poses (supine, prone,
reclined with the legs on a chair rest, and reclined with either arm
supported), and six initial atlas sites. By default it selects immutable artwork from the shared collection, using
the training-family split. `--artwork-split validation|test|all` selects another
explicit split. It reuses artwork across independently sampled scenes. The six sites are a simulation distribution subset, not
the 59-site InkLang vocabulary. It constrains every compiled trace to the
requested face-labeled region, pairs supported-arm poses only with a tattoo on
that same arm, and selects an upward-exposed atlas face so the named pose stays
meaningful relative to gravity. If one body/pose/site pairing fails clearance,
the bounded retry advances to a different exposed site for that same balanced
body/pose slot before repeating the pairing. A clearance rejection is retained
and that exact physical pairing is deprioritized for later slots, while missing
site coverage is tried first. It then searches the finite
envelope in `config/inkmap/placement-search.json`. The CPU audit varies body
X/Y, body yaw,
and fixture offset; probes 32 exact trajectory targets; and rejects any
candidate that misses the 1 mm IK, 10 mm non-tool, or 5 mm tool-shaft gates.
The probe is a fast rejection gate only. The candidate that wins it is then
solved in full, the way the generator solves it: the same planner over the
whole 3600-step trajectory, the expert's sequential joint solve, and FK of
that joint reference against every target. A candidate whose full solve
misses 1 mm anywhere is rejected as `ik_full` and the next-ranked candidate is
tried, so an accepted scenario is one the generator will accept. (Before this
gate, 2026-09-03, a placement passed 32 probes at 1e-7 m and then failed
`generate` on 185 of 5400 targets that fell between them.)
URDF collision meshes are sampled to a conservative 2 mm lower bound, while
the body and fixture use named capsule/box proxies. Every selected scenario
embeds the configuration digest, transforms, objective values, clearances, and
complete candidate ledger. `attempts.jsonl` records suite rejects with explicit
reasons; exhausting the bounded retry budget fails the suite. The later dataset
run still recomputes FK for every target and refuses any residual over 1 mm.
`--no-reach-audit` exists for geometry-only debugging;
its output is not reach-qualified. To use acquired artwork, pass
`--design-source directory --generated-design-dir DIR`, where `DIR` holds
complete [DrawingBot V3](drawingbot.md) acquisitions.
Use `--design-source spiral` only for the fixed calibration spiral.
The episode loop never makes a network request, and generated suites must be
outside the checkout.

`tatbot sim materialize` (Inkgen) stores source imagery only: the exact PNG
and traced SVG with their hashes and service metadata. The directory loader
refuses it; acquire those sources through DBV3 first.

Resolve one semantic request through the canonical InkLang parser and bounded
placement search:

```bash
tatbot sim resolve "a dbv3-orbit on the left forearm" -- \
  --output-dir ~/tatbot-sim/resolved/orbit-42 \
  --design-id dbv3-orbit --size-mm 30 30 --support armrest --seed 42
```

The typed `tatbot.scenario-request/1` record captures the prompt, canonical
placement intent, compatibility program, parser/config digest, design subject
or exact ID, size, fixed model binding, site, side, pose, support, and seed. Free text cannot
bypass InkLang validation or write scenario JSON. Normal requests select matching frozen collection artwork or complete immutable
materializations; `--design-id spiral-v1` is the sole built-in regression
exception. Simulation invokes the same batch-capable TypeScript InkLang core
as Inkmap and embeds its complete intent and resolution JSON in PlacementFile
v6 provenance. Node.js 22 or newer is therefore an explicit simulation
dependency; there is no second Python grammar or face resolver. A named
`simulation-exposed-grid-v1` policy may select only among canonically resolved
region-UV candidates for the requested pose. Resolution produces request,
placement, scenario, manifest, and attempt-ledger files. Unknown anatomy,
missing laterality, incompatible pose/support, relative sites unsupported by
this scenario consumer, and unsupported SVGs fail with named errors. The default provenance epoch keeps same-revision
request/seed outputs byte-identical; pass `--created-at` when a wall-clock
timestamp is part of the request provenance.

The compiled scenario enters the normal Tatbot expert, IK, floor-clamp, ink,
render, and LeRobot writer through a separate distribution from the `skin-tattoo` silicone-pad scenes:

```bash
tatbot sim generate body-tattoo -- \
  --scenario /path/to/one/accepted.scenario.json \
  --out-dir ~/tatbot-sim/body-forearm --num-episodes 8 --num-envs 8 --seed 0
```

The scene includes the complete posed body as a kinematic visual, a textured
drawable mesh patch, and conservative body-capsule clearance proxies.
Chair, bed, armrest, and tabletop geometry is no longer instantiated, including
furniture collisions and clearance obstacles. Pad scenes also omit tabletop
clutter. Legacy support IDs remain readable as scenario pose metadata;
legacy table and clutter randomization settings no longer create scene objects.
Generated OBJ caches live under `~/.cache/tatbot/body-scenarios/`; datasets
remain outside the repository.
Body-tattoo approach and inter-stroke hover are capped at 20 mm above the local
surface; this preserves an approach while staying inside the audited 3RL
orientation envelope. The drawable patch mesh reaches past the pigment
field's raster, so its texture is sampled with edge clamping; a repeating
sampler used to tile the drawn design across the whole limb in every wrist
camera.

The current body patch is a curved, kinematically projected contact surface;
the coarse body capsules are avoidance proxies, not a qualified skin contact
mesh. Body datasets are therefore stamped `kinematic-contact-v1` and target the
resolved TCP at zero working offset. The 3RL currently uses nominal datasheet
geometry: generation, preview, compile, audit, and evaluation continue with a
machine-readable development warning instead of waiting on a touch-off. Use
`--require-qualified-geometry` only when the purpose of a run is to prove
calibration eligibility. An axis-inferred body is not itself a warning when its
axisymmetric contact tool has a qualified fixed-point calibration.

## Exact-design simulation evaluation

Ask generation to score each deposition episode while the exact intended path,
surface, and pigment field are still in memory:

```bash
tatbot sim generate paper-draw -- \
  --out-dir ~/tatbot-sim/eval/expert-seed-4200 \
  --num-episodes 8 --num-envs 8 --seed 4200 --judge
tatbot sim eval dataset ~/tatbot-sim/eval/expert-seed-4200 -- \
  --output-dir ~/tatbot-sim/eval/reports/expert-4200 \
  --training-seed-range 0 4095
```

Every batch, for every distribution, is gated on the joint reference the
expert actually solved: FK of that reference must land within 1 mm of every
target, and on every pen-down step it must sit within 0.25 mm of its intended
height above the surface. The second gate exists because the damped IK trades
position against the requested tool lean and its converged reference sat
0.5-2 mm above the sheet on later strokes: inside the residual gate, outside
the 0.5 mm contact band, so the pen hovered and never marked. Generation now
closes the reference on contact, moving each offending target along the
surface normal by its measured error and re-solving (up to three rounds). A
batch that still misses is re-solved once with a longer budget, and an
episode that still misses is dropped, as is an episode whose sheet nothing
touched. Each episode records its final `reference` errors in `run_meta`. Dropped episodes are never written; they are listed in
`run_meta.dropped_episodes` with their reason (`ik_reference`, `dip_reference`
or `idle`) and
the run keeps generating until the requested count is met. Pass `--keep-idle`
to retain blank episodes for a study of them. Before this gate the sequential
solve could drift centimetres off a stroke unnoticed: a paper stroke sat
38 mm above the sheet with the run reporting no fault (2026-09-03).

A dip is gated on the cap rather than on its commanded point. The point is
inside the cap by construction, so the 1 mm residual cannot tell a dip from a
reference that never entered one — and the charge is credited by step index,
so nothing downstream notices either. Generation now checks, at each credited
step, that the tip lies within that cap's own radius of the cap axis and that
the tool is within 15 degrees of the entry axis; an episode that misses is
dropped as `dip_reference`. Measure a placement before generating dip episodes
with `tatbot sim reach`, which reports both per cap. Before this gate, every
cap of the palette installed then, at its synthetic placement, sat 5-25 mm
and 17-26 degrees outside, and full charges were credited for dips that never reached a cap
(2026-09-09).

Typed Inkmap v3 scenarios additionally materialize `target-labels.npz`,
`target-coverage.png`, `target-reference.png`, and a convention/hash manifest
beside the posed scene geometry. These are canonical chart-space intent at
4 px/mm, not runtime state: soft coverage is area sampled, colors are
unassociated sRGB, and integer layer/placement IDs are nearest sampled with
zero as background. Surface-chart rotation and texture mirror follow the same
split as the browser, avoiding a second baked rotation. The body patch starts at the bundle's requested skin tone;
only contact-driven `InkField` deposition changes runtime pigment. This avoids
leaking a completed target tattoo into policy observations.

The reproducible renderer-parity command is:

```bash
uv run --project python/tatbot_sim --extra maniskill \
  python -m tatbot_sim.inkmap.parity_evidence --output /absolute/new/evidence-dir
```

It records accepted/rejected denominators, exact surface-address and posed
position discrepancies, connected-domain/exclusion/semantic-region leakage,
complete-wrap refusal, aligned mask IoU, boundary distance, original/browser/
simulator views, and difference images. Its 12 px/mm comparison raster is an
adequate-resolution qualification view, not a change to the 4 px/mm minimum
target emitted with ordinary scenarios.

Render a strict CPU perception reference from a typed v3 scenario:

```bash
tatbot sim perception /tmp/scenario-v3.json -- \
  --output-dir /tmp/inkmap-perception --views 3 --seed 42
```

Each frame stores RGB, clean and declared-corrupted metric depth with separate
validity, camera-frame geometric normals, visible-body mask, global SOMA face
index and perspective-correct barycentrics, and visible tattoo soft coverage,
placement ID, and layer ID. Integer IDs are nearest sampled and zero means no
tattoo; non-body pixels use face `-1` and zero barycentrics/normals. Occluders
retain valid scene depth while clearing body and tattoo semantics. Calibration,
independent per-axis seeds, sampled appearance/camera/depth/occlusion settings,
source and asset provenance, identity/design/placement/scenario hashes, and a
leakage-safe split are in each manifest. These NPZ sidecars are privileged
labels and are not policy observation features.

Re-run the fail-closed audit with
`python -m tatbot_sim.inkmap.perception_audit --path DIR`. The CPU path is a
semantic oracle, not an assumed production renderer: its report includes
frames/second, peak memory, bytes/frame, and linear 540-frame projections.
Do not scale until an assigned renderer meets the recorded throughput/storage
budget and human review accepts the images. The reference identity is admitted;
three bounded MHR/SOMA candidates remain refused for dataset use until their
human visual-review cells change from pending. Their topology alone is not
transfer authorization.

Plan the complete pilot before assigning compute:

```bash
tatbot sim pilot-plan -- --output-dir /tmp/inkmap-pilot --seed 42
tatbot sim pilot-audit /tmp/inkmap-pilot/pilot-plan.json
```

This starts no renderer and materializes no scenes. It writes a hash-bound,
audited ledger for 60 reference-identity scenario templates, the selected
three-identity/540-view expansion, 12 reach-gated drawing episodes, all named
start/failure variants, and the separate stress/refusal suite. Candidate
identity, compute-host, reach/clearance, GPU, and human-review gates remain
machine-readable and pending rather than being inferred from the plan. Stencil
and failure injection are implemented contracts; only generated evidence for
them remains gated by compute, reach, rendering, and review.

For a generated drawing episode, `--save-privileged-labels` writes an NPZ
timeline under `meta/privileged` rather than adding policy observation fields.
It requires `--texture-refresh-steps 1`; otherwise generation refuses before
launch, because a new deposited-ink label paired with stale RGB is not the same
simulation step. Each retained episode records post-step tool pose, synthetic
contact distance/incidence and pen state, intended target/surface frame,
compiled primitive and TattooProgram layer, elapsed progress, deposited
coverage, remaining target, and a synchronization bit. The dataset auditor
checks hashes, one timeline per episode, exact frame count, target/contact
consistency, monotonic progress, bounded fractions, and synchronization.
Contact and deposition remain outputs of the declared synthetic model—not
measured force, penetration, tissue response, or human-contact evidence.

Pass `--episode-variant` with one of `blank-start`, `stencil-start`,
`missed-stroke`, `interrupted`, `partial-coverage`, `dry-tool`, or `occluded`
when generating a compiled body scenario. Stencil pixels remain a separate
appearance and privileged label below deposited pigment: the per-step
`stencil_visible_fraction` is the mean of stencil coverage times one minus
deposited pigment, so it falls as the drawing covers its guide and the auditor
refuses a timeline where it grows; the constant guide area is
`stencil_area_fraction` in `run_meta`. An `occluded` episode hides exactly
`--occlusion-fraction` of every camera image (the pilot ledger carries the
value from its appearance) and records the achieved area. Failure variants keep
the unmodified intended trajectory as their answer key, change only the named
execution or observation dimension by a declared constant (6 mm tangent slide,
15 mm lift over the middle 40 % of steps, stop at 55 % of intended steps),
retain otherwise-idle episodes, and write an explicit expected outcome rather
than a successful-demonstration label. The plan's reach masks and tool ceiling
are validated on the unperturbed targets, so the lift is checked against the
same ceiling and the joint-reference gate is re-run on the perturbed targets;
a variant that leaves the IK envelope is refused with a message naming the
variant, never blamed on the compiled tattoo.

The judge rasterizes the answer key with the same `InkField` kernel that lays
down simulated pigment. Its headline is tolerance-band F1: precision measures
whether drawn pigment belongs to the design, while recall measures how much of
the design was completed. IoU, directional Chamfer distances, coverage ratio,
blank/engaged fractions, contact duration, interaction frames, and floor-clamp
rates explain that number. Corruption tests pin the expected direction for
offset, scale error, truncation, jitter, and missing strokes.

Every judged episode stores exact intended, drawn, and overlay PNGs with hashes.
`tatbot sim eval dataset` writes a JSON report, `scores.csv`, and a human-readable
Markdown report, including a deterministic 95% bootstrap interval when at least
three episodes exist. A training-seed overlap marks the split contaminated;
dirty dataset source, dirty evaluator source, mixed tools/distributions, or
fewer than three episodes makes the report non-comparable. Producer and
checkpoint identity come from the dataset contract and cannot be relabeled by
the report command.

A clean held-out result is still reported as `screen-only`. Simulation ranking
has not yet established correlation with physical rollouts, and no score from
this command is motion or human-contact authorization.

Run a checkpoint closed-loop through the same async LeRobot server used by a
rollout:

```bash
scripts/eval/serve.sh --policy /models/candidate --env-root ~/il-serve
tatbot sim eval policy -- \
  --server 127.0.0.1:8080 --policy /models/candidate \
  --wire-scenario act_rgbd14_masked --distribution paper-draw \
  --repetitions 3 --seed 20260903 \
  --output-dir ~/tatbot-sim/eval/candidate-20260903
```

A blank-sheet control runs the same worker with no server and no checkpoint:

```bash
tatbot sim eval policy -- \
  --client-mode hold-control --distribution paper-draw \
  --repetitions 3 --seed 20260910 --output-dir ~/tatbot-sim/eval/hold-20260910
```

It sends the worker's own joint state back every step, scores F1 0, and runs
from the simulator's interpreter, so a sim node without the serving
environment can produce it. Its report carries the checkpoint id
`hold-control` and a digest of that contract name, never a model digest.

The LeRobot client and ManiSkill worker remain separate processes and virtual
environments. Their local socket carries bounded JSON plus typed arrays, while
the client deliberately uses the deployed gRPC inference protocol. Feature
keys and shapes come from the actual Tatbot follower declaration. Chunk
overlap, stale-timestep rejection, 30 Hz target filtering, joint slew, worker
protocol, checkpoint digest, zero-effort basis, surface profile, tool geometry
basis/warnings, intended/drawn/overlay hashes, and optional videos are retained
in the episode bundle. `--resume` reuses only completed deterministic episodes.

The fixed spiral is reserved for regression controls. Normal policy batteries
use the seed-generated design stream (or explicitly acquired DBV3 artwork),
so a successful client cannot overfit a small checked-in design catalog.

## Contribution checklist

1. Add a deterministic fixture for the new behavior.
2. Assert units and coordinate frames at the boundary.
3. Run `scripts/check --light`. Run `scripts/check sim` where a locally
   qualified simulation profile is present; otherwise preserve its explicit
   profile-missing skip and exercise the config-independent tests you changed.
4. Label simulated results as simulated in the run manifest.

## Shared artwork and calibration

Normal body, paper and skin drawing use `dbv3-acquired-v1`, the three native
DBV3 acquisitions in Inkmap's manifest. The train, validation and test splits
hold out their source families. Each acquisition has one physical size and pen
configuration; resizing or changing pen width requires regeneration. Acquired
paths retain order and direction through the simulator schedule. The spiral
remains an explicit simulator calibration control, never a finished artwork
fallback. `--generated-design-dir` now reads native acquisition directories with
`result.json`, `artwork.json` and a matching frozen recipe. Older traced-image
materializations require new DBV3 acquisition.

```bash
tatbot sim sample --count 12 -- --output-dir ~/tatbot-sim/artwork-train \
  --artwork-split train
tatbot sim sample --count 4 -- --output-dir ~/tatbot-sim/artwork-test \
  --artwork-split test --poses reclined-left-arm-supported --sites forearm
tatbot sim generate paper-draw -- --out-dir ~/tatbot-sim/paper-art \
  --num-episodes 4 --artwork-split train
tatbot sim qualify-artwork -- --output-dir ~/tatbot-sim/artwork-review
tatbot sim qualify-artwork -- --output-dir ~/tatbot-sim/artwork-review-hatch --fill-style hatch
```

`qualify-artwork` plans the collection through the same scheduled material
strokes the planner schedules for a drawing; `--fill-style` selects the paint planner
([fill styles](design-format.md#fill-styles)) and the report records it, so
the two planners can be compared artwork by artwork on stroke count, ideal
footprint IoU and simulated deposition.

### Recipes

`tatbot sim recipes` expands a frozen artwork library into reproducible
scenario recipes without compiling, rendering or executing anything:

```bash
tatbot sim recipes -- --output-dir ~/tatbot-sim/recipes-v1 --count 1000 --seed 7
tatbot sim recipes-status -- --output-dir ~/tatbot-sim/recipes-v1
# native DBV3 acquisitions instead of the bundled collection:
tatbot sim recipes -- --output-dir ~/tatbot-sim/recipes-gen --count 1000 \
  --artwork-dir ~/tatbot-artwork/dbv3-acquisitions
```

A recipe is a pure function of the frozen plan and its index. Every axis —
artwork, identity, pose, site, scale, rotation, mirror, target pose, and the
named camera/appearance/sensor/compile streams a later stage will use — is
drawn from its own seed derived from (plan seed, recipe key, axis name). So
sharding the work, resuming it, or reordering it produces byte-identical
recipes, and the plan's digest is the run's identity. Rerunning the command
resumes: recipes already on disk are verified against the plan and kept,
anything that no longer matches is rewritten from the plan rather than trusted.

The ledger counts `requested`, `admitted`, `rejected`, `compiled`, `rendered`
and `executed` separately, and the last three stay zero here. A recipe is not a
render, and a render is not a completed drawing. Rejected recipes keep their
reason in `rejected.jsonl`.

Splits are assigned from the artwork's **family** and the identity, before any
augmentation — never from a variant's seed, crop or SVG digest — so a rotated
or rescaled copy of a training artwork cannot become held-out artwork. The
command audits family and identity leakage across the held-out partitions and
exits non-zero if it finds any.

Artwork with no reviewed family — a freshly generated library, for instance —
is one group, so the whole of it moves across splits together. That is the
conservative reading of unknown provenance rather than a meaningful family
split, and the ledger names the artwork in `unknown_family_artworks` instead of
leaving it to be inferred from a single-split tally. Families are reviewed
metadata; nothing here invents one from a subject line.

The collection binds exact acquired JSON hashes, source/license metadata,
recipe identity, generation width, frozen physical size, and artwork families.
The three bundled families have one training, one validation and one test
example. Placement rotation and scene variation retain the family's split;
duplicates cannot cross it. A different physical size requires regeneration
through native DBV3.
Episode metadata records source hashes and family/split. Evaluation reports
block artwork from training or unknown splits and report calibration controls
separately from artwork comparisons. Historical datasets are not relabeled.

Normal sampling and semantic resolution compile typed v3 bundles. The placement
optimizer authors new immutable bundle requests for each candidate, binding
body position, yaw, and bounded support offset. Its candidate/rejection ledger
lives in the suite manifest; it does not mutate an already compiled v3 world
transform. All existing IK and robot/tool-shaft clearance gates remain.

Paper/skin artwork defaults to a 9,000-step (five-minute) cap at 30 Hz —
`config.ARTWORK_HORIZON_STEPS`, raised from 3,600 on 2026-09-10. Complete
artworks must fit; randomizing an artwork never drops a stroke to meet the cap.
The cap is not free: a candidate that used to be refused cheaply at two minutes
now runs the full IK reach audit over three times the trajectory, so a suite
that leaves the audit on takes correspondingly longer. `--no-reach-audit` skips
the CPU yaw selection; final generation still enforces exact FK.

`tatbot sim sample -- --max-seconds N` bounds a suite's wall clock (default
1800, `0` for none). The budget is enforced in two places because one is not
enough: between candidates, and as a share of the remaining budget around each
placement search. Wrapping only the last stage of that search left a measured
run twenty minutes past a ten-minute budget without the check ever being
reached. A search that outruns its share is refused with reason `time_budget`
and recorded like any other rejection; the suite then finishes with the honest
partial report it already produces for an incomplete run.
The default footprint is a **nominal simulated 0.3 mm**, not measured tool
calibration. V3 episodes use their bound uniform footprint width; mixed-width
episodes refuse until per-stroke width execution is available. Continuous
contact is sampled between control frames so a narrow line does not become
disconnected dots; pen lifts and resets break that interpolation.

### What the planner will and will not draw

Two geometric gates decide whether artwork becomes a drawing, and both moved on
2026-09-10 at the fleet owner's direction. Neither is a safety limit: they gate
fidelity, not motion, contact force, retract or ink accounting.

**Paint coverage** (`fill_geometry.PAINT_COVERAGE_FLOOR`, 0.90, was 0.98) is how
much of a painted region the planned stroke path must actually cover. The old
threshold refused 57 of 95 candidates in a measured suite over generated flash.
Those refusals are not one population: their coverage runs 0.0, 0.31, 0.77 …
0.97, and a floor of 0.70 admits exactly the same set as a floor of 0.50,
because a handful cover essentially nothing. 0.90 leaves a tenth of the ink at
most missing, and a drawing that covers nothing is still refused. It costs
little against a looser floor: the same library gives 9 usable artworks at
0.98, 11 at 0.90 and 12 at 0.80. A consumer that wants the old strictness passes
`coverage_floor=0.98`.

**Paint thinner than the tool** used to refuse the whole drawing. A tool that
cannot draw a line thinner than its own tip does not refuse to draw the line:
it traces the middle and the line comes out at tool width, which is what a
person does with a fine liner. Those regions are now traced down the middle,
and ink is allowed outside the artwork only in the halo the tool needs for
them. How much heavier the result may be is bounded by
`NARROW_OVERDRAW_LIMIT` (4x the painted area): a 0.3 mm tool on a 0.2 mm line
lays 1.76x and is drawn; on a 0.05 mm hair it would lay far more and is
refused, as is any tool larger than the region itself.

Measured on a 24-piece generated library at 30-45 mm: 11 of 24 compile to
strokes, against 9 before these changes. Of the twelve that do not, four carry
paint finer than the tool can meaningfully trace and five cover essentially
nothing at any size — properties of what the model drew. Generate roughly twice
the artwork a suite needs and expect to discard about half.

Qualification compares source SVG, shared typed preview, canonical target,
ideal finite-width coverage, complete expert trajectory duration, and runtime
InkField deposition. The reference uses 48 px/mm with 4x target supersampling;
paint comparisons require IoU >=0.98 and runtime deposition requires >=0.90.
The latter is a discrete nominal pigment model, separate from source-paint
parity. Each size, rejection, duration, stroke count, and pen lift is recorded.
Output includes a visual contact sheet and source/compiled/target/deposition
images. This CPU planar qualification does not establish GPU scene, robot IK,
measured deposition, or physical execution acceptance. The body sampler and
bounded generated episodes provide separate evidence for their own stages.

Artwork generation resolves pigment and pad textures at 16 px/mm (the flat
recipe can override `--artwork-pixels-per-m`). Typed body patches use the same
16 px/mm pigment minimum; reference targets retain their independent minimum.
The higher resolution costs GPU memory, so batch size must fit the selected
node. Nominal width is 0.3 mm, independent of the original 2–4 mm legacy ink DR.
Raster imports use the shared polygon tracer, retaining small connected marks
and holes; spline fitting is excluded because it can distort paint without
reporting an error.

Regenerate the five-pose artwork gallery with `tatbot sim showcase-artwork --
--output-dir /path/outside/checkout`. Review and install its six JSON files in
`web/inkmap/public/showcase/`; its manifest preserves offline-only qualification.
The line and geometry files under `config/inkmap/examples/` remain contract
fixtures, not normal design sources or gallery content.


## Dependency and check boundaries

The default `tatbot_sim` install provides CPU preparation, geometry and dataset
utilities. Its numerical dependencies include PyTorch; it does not install
ManiSkill or SAPIEN. Use `uv sync --project python/tatbot_sim --extra maniskill`
on a simulation host. Generation, preview and cinematic launchers select this
extra explicitly. Policy evaluation with a separately
provisioned worker interpreter requires the same extra in that interpreter.

`tatbot check sim` runs the partitions below. `tatbot check sim-fast` selects
the same partitions and excludes marked slow integration cases. Neither runs
in the push hook. Each missing capability reports its own SKIP; it cannot skip
contracts or count as full coverage.

| Check | Scope | Additional requirements |
| --- | --- | --- |
| `sim-contracts` | Resolved configuration, observation/transport contracts, CPU imports | Base Python environment |
| `sim-geometry` | Numerical geometry, planning and design compilation | Node with TypeScript support, C++ toolchain, cached robot assets |
| `sim-engine` | ManiSkill adapter and material/control contracts | Engine extra, cached robot assets and compiler toolchain |
| `sim-render` | Rendered sensors, shared episodes, dataset round trips | Engine extra, render device and compiler toolchain |

Individual groups can run with `tatbot check sim-contracts`, for example.
Slow rendering cases need the C++ planner, a render device and the checkout's
retained tool calibration; missing toolchains and the public nominal profile
report explicit skips. `sim-fast` excludes them. This is developer regression
coverage and adds no hardware-execution gate.
The public profile builds its compiler fixtures only for groups that use them.
Engine assets retain ManiSkill's cache layout; reading the cache path or
constructing texture files does not import the physics engine.
