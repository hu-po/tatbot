# tatbot_travel — the travel demo's synthetic data

The travel demo is one arm in one case: the blue arm with the laser pen
(never emitting) traces the ink on a silicone practice arm -- Sharpie lines,
tatbot tattoos, anything that is not skin -- and keeps tracing while
visitors move the practice arm around. It sees through its wrist D405
and, since v7, a third-person scene camera (as the SO-101 package pairs a
scene view with a wrist view), and runs a FLUX.3 Action policy on one
Jetson Thor. This package
generates the policy's training data in simulation, and runs on the
training node (aarch64) as well as on x86 workstations.

It is deliberately outside the session stack: the demo needs no scan, no
stencil and no contact, so the generator uses kinematics, rendering and a
servo model. It consumes verified SOMA pose bytes; provider execution and
its locked model runtime belong to the separate body export step.

## What an episode is

A 60 s, 30 Hz recording, LeRobot v3 layout, matching the FLUX.3 Action
LeRobot recipe:

| feature | shape | content |
|---|---|---|
| `observation.images.wrist` | 480×640×3 | the blue arm's wrist D405 colour stream |
| `observation.images.scene` | 480×640×3 | the scene camera's sub stream, resized to the wrist stream's size (v7 on; `WorldConfig.scene_view`) |
| `observation.state` | 6 | measured `joint_0..5` positions, rad |
| `action` | 6 | commanded `joint_0..5` positions, rad (absolute; the policy trains on deltas) |
| task | — | `trace the ink on the arm with the laser` |

The carriage is not a channel: it holds the pen cradle at rest. Everything
else the simulator knows (expert mode, the stroke traced and the position on
it, phantom pose, the episode's appearance draw) is written to `labels/`
beside the dataset.

## How it is made

- **Robot** (`urdf_chain.py`, `scene.py`, `tool.py`): the `left/` subtree of
  `urdf/tatbot.urdf` lifted into MJCF, with the pen as a lathe of its
  datasheet profile, the lens face and a *gap point* 20 mm beyond it along
  the pen axis.
- **Camera** (`camera.py`, `postfx.py`): the left wrist D405's factory
  calibration (fx≈392 px, inverse Brown–Conrady) — a pinhole render is
  remapped onto the real pixel grid, then blurred along the camera's own
  motion, auto-exposed, noised and passed through YUYV 4:2:2.
- **Scene camera** (`camera.py`, `postfx.py`, `episode.py`): the rig's PoE
  camera 5 at its calibrated pose in the blue arm's base frame (vision
  calibration through the robot-world calibration), its main-stream lens
  (OpenCV Brown–Conrady, k1≈−0.34) on its sub stream resized to 640×480
  (flux3 lays cameras side by side only at one size; the pixels are not
  square), knocked ±1 cm/1.5° per episode. It captures at 14–17 fps
  and its frames reach the policy 0.7–1.1 s late (the rig's stream: 0.9 s),
  soft, sharpened and compressed. It draws the pen cradle and tag cube,
  which the wrist view takes from real pixels.
- **Self view** (`selfview.py`): the pen cradle and its clamp cap fill the
  lower left of every real frame and do not match their CAD meshes, so they
  are composited from real pixels (two rooms' captures; pixels that stayed
  put while the background changed). Rebuild the layers whenever the cradle,
  the camera mount or the carriage rest changes. The tag cube on the same
  mount is out of the wrist camera's view.
- **Phantom** (`phantom.py`, `shell.py`): the left arm of the rig's
  MHR/SOMA body, cut through the upper arm and capped; in half the worlds
  also cut at the wrist into a domed stump, as the rig's practice arm ends.
  The closed hand uses the shared `left-fist-reference` SOMA pose with
  articulated fingers and pose correctives. Cutting and scaling retain
  canonical face IDs. The forearm texture chart resolves through the
  shared `tatbot_sim.inkmap.TriangleChart` and `PosedBody`: clipped render
  triangles, ink samples and intermediate expert targets use those same
  face IDs and barycentric coordinates on the posed skin.
  Only the ink rendering layer receives a 0.3 mm offset to avoid z-fighting;
  expert targets and saved addresses resolve the underlying skin.
- **Ink** (`ink.py`, `lines.py`, `designs.py`): red or black Sharpie lines,
  and tattoos -- most drawn fresh each time (`designs.py`: polygons and
  stars, blobs, hearts, rose curves, spirals, waves, vines, rings, mandalas
  and Hershey lettering, outlined, filled or hatched, sometimes faded or
  softened), the rest from `web/inkmap/public/designs` (train split, preview
  pieces and the unlisted flash; the validation/test designs are held out
  for evaluation), painted into the shell texture. The
  traced strokes are that same ink thinned to centrelines and split into a
  graph at ends and junctions, so what the camera sees and what the expert
  follows are one object.
- **Handling** (`motion.py`, `people.py`): the phantom rests (mostly),
  gets nudged, relocated, carried slowly (the case the demo exists for), or
  taken away and brought back; hands from the same body model hold it while
  it moves. It lies 0.25-0.45 m out, up to 45 degrees either side of the
  arm's heading.
- **Expert** (`expert.py`, `kinematics.py`): park → approach → trace →
  back off. It acts on privileged state but only on what the cameras
  could show (wrist-view ink visibility gates every transition; parked with
  too little ink in view, it turns its base to face the ink, which the scene
  camera shows), enters at the visible ink nearest the pen once a stretch of
  it is reachable, and walks the
  stroke graph: on through junctions to strokes it has not traced yet, back
  at dead ends, or over bare skin to other ink. It hovers higher while
  hands are on the phantom, backs off when the skin closes on the pen
  faster than it can give way, and every command goes through a
  rate-limited target and pen-pointing IK (roll about the pen axis is free).
- **Servo** (`servo.py`): measured state is a delayed first-order follower
  of the command, per-episode τ and delay, to be fitted from blue-arm logs.
- **World** (`world.py`): tables, walls, bystanders, the rig's clutter
  (racks, shelves, gear, paper, cables, tag boards, kept off the workspace
  and out of the scene camera's sight line to it), lights, skin tones, the
  ink (lines, tattoos or both), chrome reflections, both cameras' mount and
  calibration spread — all drawn from the episode seed. Episodes start on
  the ink (25 %), partway into an approach (30 %) or parked.

## Running

```bash
scripts/tatbot travel preview -- --seed 3 --seconds 30 --out /tmp/travel/ep3
scripts/tatbot travel preview -- --seed 3 --seconds 1 --out /tmp/travel/inspect \
  --profile /path/to/private-bank/profile.json --views --geometry-only
```

Generation writes LeRobot shards and needs LeRobot in the same environment
(the training environment already has it):

```bash
travel generate --root ~/travel-data/v2 --episodes 400 --workers 12
travel merge --root ~/travel-data/v2
travel stats --root ~/travel-data/v2
```

A trained checkpoint is judged by driving held-out seeds (the task string
comes from its own dataset); `--ink` pins what is drawn and
`--held-out-designs` tattoos with flash it never saw:

```bash
travel evaluate --checkpoint RUN/checkpoints/010000/pretrained_model \
  --dataset ~/travel-data/v2 --episodes 10 --held-out-designs --out RUN/eval.json
```

`--steps`/`--guidance` override the checkpoint's sampler (guidance 1 is one
pass per step), and `--latency-ticks N` runs the demo's asynchronous
chunking (`chunking.py`) instead of waiting for each chunk: plans are asked
for early, arrive N ticks after the observation they were computed from,
and are blended in while the previous plan plays; `held_fraction` reports
the ticks the arm had no plan and held. On a Thor, set
`F3_NATTEN_BACKEND=flex-fna` (the public NATTEN wheels have no sm_110
kernels).

## On the real arm

`travel run` runs on the arm node: the policy in its own process, the blue
arm's wrist D405 (re-plugged into that node), and, only with `--move` or
`--hold`, the arm itself through the plugin's lease, e-stop and golden-config
modules. Without either it is a shadow run: the state is pinned at the start
pose and each plan is drawn over the live image. `--hold` takes the arm to the
start pose through every gate and holds it there: the policy sees the real view
and joints from that pose, and its plans are drawn and logged, never sent.
Every moving command passes joint limits,
a speed cap, a forward-kinematics workspace (`--ceiling-m` is required: the
lab rig's overhead cameras hang 0.60 m over the blue arm's base), a tracking
check (a blocked joint ends the run) and a depth-clearance stop along the pen
axis. `train/fetch_checkpoint.sh` brings a checkpoint from the training node.

On exit the arm returns to its staged pose and idles over the run's own driver
session. If that fails, the plugin's recovery landing (`recovery.land_arm`)
takes over with the carriage kept where the run found it. It runs in a
process of its own (`python -m tatbot_travel.hardware`), never in the runner:
once connected (`configure()`'s own handshake included) a vendor driver can
block forever in a TCP read when the controller's TCP server wedges, and only
another process can end that. The
runner first releases its e-stop monitor and then the driver lease, and hands
over only a lease nobody holds; the recovery takes both for itself. It runs in
its own process group under GNU `timeout` with the 45-second budget of
`tatbot arm recover`, and the runner ends that group itself if the budget is
ever overrun. The summary's `landing` is `landed`, `recovered` (the recovery
measured the sleep pose) or `unknown`, and the run exits 1 unless the arm
landed.

The run's own driver session is bounded the same way. The vendor driver
(1.8.5) reads every TCP reply -- configure()'s handshake, each configuration
get and set, cleanup() -- with a blocking recv() that has no timeout, and its
binding holds the GIL, so a wedged controller can stop the runner inside any of
them; a position read or command can wait as long behind the driver's UDP
thread while that thread fetches the controller's error log over TCP. So
`connect()` first starts a watchdog process (`python -m
tatbot_travel.watchdog`) and opens no driver unless it comes up. The runner
announces each driver call to it with a budget -- 10 s, plus a blocking move's
goal time or configure()'s connect timeout -- and a call still in flight at its
deadline ends the runner (SIGTERM, then SIGKILL; the run exits on that
signal). The lease and the e-stop go with it, and the watchdog hands the arm
to the same recovery landing and writes the run's `summary.json` (`ended:
wedged: ...` and the `landing`) -- or, once `connect()` has refused a joint
measured past its limits (the recovery's clamp is the move refused), starts
nothing and records the arm state as `unknown`.
Ending the runner drops its controller connection, and the controller idles
an arm whose connection drops: an unsupported arm can fall. Waits between
calls (on a pressed e-stop, on the recovery landing) have no budget, and the
watchdog never acts when the runner ends any other way.

`TATBOT_REPO` points at a checkout when the package is installed away from
one; rendering defaults to headless EGL on GPU 0.

## Recorded scene profiles

Use a separate bank for each rig. `travel scene-bank --captures CAPTURES
--layout LAYOUT.json --out BANK` builds room meshes, cradle layers, texture
crops, segmented ink and a portable `BANK/profile.json` from existing
RGB-D captures. It never opens a driver and refuses output inside a Git
checkout. The bank and every source image stay private and outside the repo.
`manifest.json` retains the input masks, capture intrinsics, measured joints,
depth coverage and assumptions so the bank can be rebuilt.

The layout supplies `wrist_metadata` (a capture JSON with colour/depth
metadata), `selfview_polygon`, `phantom_polygon`, `crops` (each kind maps to
`file`/`box` entries), and `episode` overrides. `joints.json` supplies the
capture's six measured joints. Aligned native-depth Z is converted into
colour-camera Z before meshing. Depth holes close to measured surfaces use
nearest measured depth; the room beyond the D405's range uses an assumed
distant backdrop. The phantom and cradle are masked from the room surface,
and erased from the backdrop, so baked ink cannot survive a moved phantom.
The table's photographed pixels sit on its measured plane. Its base offset,
phantom size/placement, exposure and initial poses belong in the profile.

`travel preview`, `generate` and `evaluate` accept `--profile BANK/profile.json`.
Asset paths are relative to that file, so the whole bank can move between
compute roles. Dataset generation records its profile hash and each
episode's configured initial pose; evaluation records the same hash.
Set `world.scene_view=false` for a wrist-only dataset. An explicit
`start_pose_rad` samples initial observations at the physical stops; IK
still uses its existing limit margin and command speed limits. The profile
can provide private full-frame RGBA self-view layers, shared by the rendered
occlusion and expert visibility mask, and hide replaced CAD visuals from
the wrist view while retaining the arm's kinematics.

For an inner-forearm-up practice arm with a fist, set
`world.phantom_hand_pose="closed"`, `world.handless_prob=0`,
`world.ink_theta_origin_rad=3.141592653589793`, and placement
`back_up_prob=0`, `palm_up_prob=1`. The hand comes from the same verified
MHR-through-SOMA pose catalog used by Inkmap. Its 15 authored finger joint
rotations are in `config/body-models/mhr-soma-v1/poses.json`; the maintained
exporter applies SOMA pose correctives without a custom vertex warp:

```bash
scripts/tatbot body export --python /path/to/locked-soma/bin/python -- \
  --cache-dir /path/to/verified-soma-cache --extend --pose-id left-fist-reference --check
```

Catalog extension preserves the reviewed rest asset and prior poses. It
compares every runtime neutral vertex with the verified reference within
two quantization units (20 micrometres), and records the measured difference
and runtime versions. The full exporter retains its exact rest digest check.
The new pose and combined asset remain bound to exact byte digests.

The chart offset moves both rendered ink and expert strokes onto the inner
surface. Ink eligibility comes from the canonical forearm patch, so curled
fingers that overlap the wrist cannot enter its chart. Closing the hand
alone does not change which forearm surface faces upward. Episode JSON
records model, identity, pose, topology and catalog digests plus the phantom
transform and scale. `labels/episode_XXXXXX.surface.npz` retains canonical
face IDs, barycentric coordinates, chart coordinates and resolved targets;
the merge command carries these alongside per-frame labels.

`--views` renders wrist, two fixed workspace cameras, overview, top, side and three hand close-ups from
the same world builder as generation, and writes provenance and surface
labels beside them. `--geometry-only` hides captured room meshes in the
external views to make the articulated geometry inspectable. The wrist
view retains the episode's actual appearance.

Floor and lighting randomization are profile controls under `episode.world`:

```json
{
  "real_table_prob": 0.0,
  "surface": {
    "styles": {"wood": 0.3, "cloth": 0.25, "speckle": 0.25, "plain": 0.2},
    "saturation": [0.1, 0.6],
    "value": [0.15, 0.9],
    "neutral_prob": 0.25,
    "randomize_captured_floor": true
  },
  "tabletop": {
    "mat_count": [0, 2],
    "paper_count": [0, 6],
    "x_m": [0.12, 0.62],
    "y_m": [-0.32, 0.32],
    "paper_placement": {
      "x_m": [0.08, 0.72],
      "y_m": [-0.42, 0.42],
      "width_m": [0.09, 0.26],
      "height_m": [0.12, 0.35],
      "layouts": {"scatter": 0.55, "cluster": 0.30, "stack": 0.15},
      "scatter_separation_m": 0.18,
      "cluster_spread_m": 0.11,
      "stack_spread_m": 0.015,
      "stack_yaw_std_deg": 8
    }
  },
  "lighting": {
    "energy": [0.5, 1.8],
    "warmth": [-0.25, 0.25],
    "headlight_ambient": [0.12, 0.35]
  }
}
```

Plain surfaces use solid materials. Other supported styles are cloth,
speckle, wood, grid and rubber; their non-negative weights need not sum to one.
Color, specularity and shininess vary per episode. `real_table_prob` controls
photographed crops independently. When `randomize_captured_floor` is enabled,
horizontal captured triangles within the configured tolerance of the table
plane are replaced visually by the existing flat support plane. Fixtures and
background photo meshes retain their pixels.

Tabletop mats and paper are generated even with `scene_view: false`. Mats
sample cutting grids, rubber and cloth through `tabletop.mat_surface`;
paper samples blank, ruled, graph and printed textures through
`tabletop.paper_styles`. Each thin solid has explicit UVs fitting one texture
to its top. Counts, positions, sizes, rotations, thicknesses and styles are
logged. Omitting `tabletop.paper_count` uses the existing `max_sheets` limit.
`tabletop.paper_placement` controls paper independently of mats. Its XY
ranges inherit the tabletop bounds when omitted. Each episode samples one
layout: scatter prefers separated centers, cluster jitters sheets around
the first sheet, and stack keeps positions and headings close to it. Scatter
chooses the most separated candidate when a crowded area cannot meet the
requested separation. Width, height, yaw and layout spreads are configurable;
layout names are saved with each sheet. Overlapping footprints determine
stack heights, including sheets crossing mat edges.
The motion sampler finds support against the actual SOMA skin triangles,
including triangle/edge intersections with mat and paper footprints. The
resulting pose drives rendering, clearance, ink targets and frame labels.

Lights vary in count, intensity, warmth, position, spotlight cone, shadows and
headlight fill. The full sampled material and light parameters are recorded in
episode and preview metadata. Geometry and ink use a separate random stream:

```bash
scripts/tatbot travel preview --python /path/to/travel/bin/python -- \
  --seed 10000000 --seconds 2 --profile /path/to/private-bank/profile.json \
  --out /tmp/travel/appearance --views --geometry-only \
  --appearance-seeds 11 12 13 14 15 16
```

This saves each variant, its surface labels and an `-appearances.png` comparison
sheet with overview, hand and wrist views. `--appearance-seed` renders one
variant. Generation normally derives its appearance seed from the episode
seed; comparisons override only appearance, preserving skin geometry, ink, kinematics
and placement.

Skin appearance is configured under `episode.world.skin`. Its default tone
center is the current cream practice arm, RGB `[236, 225, 198]`: 80% of draws
use a normal brightness gain centered at 1 with standard deviation 0.10;
20% use a broader gain of 0.40-0.85. Independent warmth and redness vary the
tint. Specularity and shininess stay centered at 0.30 and 0.40, with wider
bounded spreads. Mottling, fine surface variation and pore density also vary
around the existing surface. Each scalar property accepts `center`, `std`
and `bounds`; `tail_prob`, `tail_gain` and texture scales are configurable.
The legacy `world.skin_rgb` remains a center override.

One procedural skin atlas covers the articulated SOMA arm and hand. The
forearm ink chart samples that atlas at the same cylindrical coordinates,
then composites the ink. Both meshes share the sampled material properties.
The skin random stream derives from the appearance seed and is logged along
with every sampled property. It does not advance the geometry/ink stream.

```bash
scripts/tatbot travel preview --python /path/to/travel/bin/python -- \
  --seed 10000000 --seconds 2 --profile /path/to/private-bank/profile.json \
  --out /tmp/travel/skin --views --geometry-only --skin-seeds 0 2 4 6 11 3
```

The `-skins.png` sheet starts with the configured reference tone and material
centers, followed by the requested skin seeds. Scene, lighting, articulated
geometry, placement and ink remain fixed. `--skin-seed` renders one variant;
`--skin-reference` renders the reference alone. Setting `skin.randomize` to
false uses those centers during generation too.

Use full episode seeds to review placement as well as appearance:

```bash
scripts/tatbot travel preview --python /path/to/travel/bin/python -- \
  --seconds 2 --profile /path/to/private-bank/profile.json \
  --out /tmp/travel/scenes --views --geometry-only \
  --episode-seeds 10000000 10000001 10000002 10000003 10000004 10000005
```

The `-episodes.png` sheet uses fixed workspace and overhead inspection
cameras, plus the wrist view; its captions include paper layout, position
and heading. The fixed cameras cover the wider table area.
The close-up cameras follow the forearm and therefore conceal changes in
its heading relative to the robot. Placement controls live under
`episode.motion.placement`; for example, radius `[0.22, 0.46]` m,
azimuth `[-50, 50]` degrees, yaw `[-180, 180]` degrees, `back_up_prob: 0`,
`palm_up_prob: 1`, roll jitter 15 degrees and tilt jitter 5 degrees retain
a palm-up forearm while varying its placement. The SOMA fist stays closed.
Rest placement rejects geometry inside the robot-base clearance; exhausted
attempts raise an error instead of returning a rejected pose.

Rest captures establish the opening view. They do not validate the room's
appearance along an approach; use recorded joints/frames to check other
known views and hand-guided captures to measure missing approach views
before claiming that the sim matches those poses.
