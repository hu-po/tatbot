# Tatbot on ROS 2

This folder is the rebuild of Tatbot's autonomous drawing stack on ROS 2 Jazzy
with `rmw_zenoh`, and it is Tatbot's only drawing path. It takes an
acquired DBV3 artwork, prepares its paths as a robot program, registers the page and tools
to the arm that will draw, and executes the program with that arm. A program
can use several resources (tools and inks): at a tool change the arm lands for
the operator to swap the cartridge, and a dipping tool dips at the palette (§6).

Status: the pink arm draws. On 2026-09-26 `tatbot ros draw` located the
stencil page (overhead cameras fused with the wrist D405), touched it three
times, drew the coffee cup with 71% of the plan inked, inspected it with the
wrist camera and returned to rest. On 2026-09-27 the palette e-stop button,
relayed from the palette Pi, held the pink arm on a press and on relay
silence. Not yet run: two arms, dipping. This page is the design
that code in this folder must follow. Update it in the same commit as any
code that changes it.

## 1. Rules for this folder

1. **One implementation per concern.** One arm-owner process, one planner,
   one e-stop reader, one kinematics source (the URDF, through pinocchio and
   tf2), one executor, one run format.
2. **Only named unsafe states block motion.** These are:
   - the e-stop;
   - controller and joint limits;
   - the carriage contact trip;
   - a tracking stall;
   - measured over-velocity;
   - a first command that steps away from the measured pose.

   Everything else is recorded and shown, never gated: calibration
   residuals, camera freshness, registration quality, coverage.
3. **Every tolerance names the bench measurement it came from.** A tolerance
   tighter than the robot's own noise is a bug. For example, the arm sags
   1–2 mrad at rest and lags about 70 ms while tracking.
4. **Page pose and height are fused from every sensor, fresh each run.**
   - The printed stencil's pose is tracked by the fixed cameras in the camera
     world and crossed into each arm's base through that arm's registration;
     the wrist camera's view of the printed border corrects it in the plane.
   - The height fuses the overhead page, the wrist depth plane and the
     arm's touches, each weighted by its measured error. A touch's guard
     trips on a force read from the joint torques, which the arm's own
     cables also move: on 2026-10-03, 11 of 16 trips stopped 9-25 mm over the
     paper with nothing under the pen, and real contacts' trip heights moved
     2-4 mm with the cables and the tool's turn. So a touch counts only when
     its trips agree (three, or two once the paper is known; each further
     descent turned 20 deg about the tool), the paper the last accepted
     touches found starts the next run's, and the wrist D405 saves a frame at
     every trip. The cameras' own page heights read the paper 8-14 mm high.
   - The pen rides link 6 with the wrist camera, so its tip is a fixed point
     in that camera's frame. The wrist gauge (`tatbot_session.gauge`) gives
     the tip's height over the paper from one depth frame, with neither the
     arm's kinematics nor the camera's mounting (16 contacts within 0.25 mm,
     1 sigma, on 2026-10-03). It is fitted from touch runs' trip frames
     (`ros2 run tatbot_session gauge fit RUN... --write`, into
     `~/tatbot-ros/calib/wrist-gauge-<arm>.json`). `page.height.gauge`:
     `check` holds the pen where each touch starts and records the gauge
     beside it; `use` takes the page from the gauge and makes no descent.
     The fit expires when the tool moves in the gripper or its cartridge is
     swapped, and nothing checks for that yet: the wrist depth that `planes:`
     compares with the gauge rides link 6 too.
   - The same frames say where the tip stands on the print (`page.locate.holds`,
     `inspect.hold_tip`): the gauge's surface plane, the print's artwork matched
     on it, the gauge's tip point carried into the print's frame, again with
     neither the kinematics nor the camera's mounting. The median of where the
     tip stood less where it was sent (`page.json` `holds`) moves the page onto
     the print, so the locate's far views, ~5 mm off in y on a half-skin print,
     no longer place the drawing. The holds scatter ±1.3 mm in y, and the
     reading moves ~1 mm between a 4 mm hover and the surface (2026-10-06).
     `page.trim` adds to it.
   - A height holds only on the IK branch it was measured on: the arm's own
     tip-height error moves ~5 mm per radian of joint 6 (2026-10-03). So the
     ready pose (the pen's heading and the joints) belongs to the page: a new
     page searches it from the rest pose, the first heading within 0.05 rad of
     the best joint-limit margin, and records it; a resume and the next run on
     the same page (carried with the paper) take it again, and a ready pose off
     the recorded one is named.
5. **The bench is the test.** Unit tests cover pure functions (compile,
   motion, fits). No mock executor stands in for the driver.
6. **Calibration is data, never code.** Tool tips, page, station and camera
   poses live in `config/workspace.yaml` and become static transforms.
7. **A code change reaches the arm in under a minute.**
8. **Budgets:**
   - about 20k lines of code;
   - at most 8 custom interface types;
   - at most 3 launch files;
   - no versioned schema zoo.

   Files carry one `format`/`version` pair and no legacy readers.

## 2. Architecture

```
operator node                         arm node (aarch64, real time)
─────────────                         ─────────────────────────────────────────────────
tatbot CLI ── role routing ────────▶  tatbot_session (rclpy orchestrator, one executor per arm)
tatbot_ink compile                      │  program → per-op Cartesian plan → CLIK → FollowJointTrajectory
  (inkmap-design.json → program.json)   │  stroke ledger (JSONL), palette lock, continue/land decisions
tatbot_calib solves                     ▼
                                      controller_manager @ 400 Hz (SCHED_FIFO, pinned cores)
                                        ├─ joint_trajectory_controller  ×2  (position + velocity FF)
                                        ├─ joint_state_broadcaster
                                        ├─ guide controller (gravity compensation, hand-guiding; M7)
                                        ├─ <arm>_safety_controller (GPIO: e-stop, probe, latch per arm)
                                        └─ tatbot_hardware::TatbotArm ×2 (left, right)
                                             SDK owner · e-stop + probe reader · hold latch · guards
                                             carriage trip-retract · stall and velocity guards
                                      robot_state_publisher (URDF + calibration static transforms)
                                      rosbag2 → MCAP (every run)
palette station: e-stop NC contact (+ probe, M2) ── GPIO ──▶ palette RPi: EST1 relay ── UDP, rig LAN ──▶ tatbot_hardware
                 palette RPi camera (camera_ros) ── rmw_zenoh ──▶ tatbot_calib
camera node:     visiond (5 PoE + D555) → stencild ── existing zenoh bus ──▶ tatbot_bridge (arm node)
                 page pose `world_from_target` per printed stencil          → /page (ROS)
viewer node:     Rerun viewer ◀── tatbot_rerun bridge (the only process that calls rr.init)
```

Perception stays where it works for now. The camera services (`visiond`,
`stencild`) keep running unchanged on today's zenoh bus.
- `tatbot_bridge` subscribes to their page poses with zenoh-python and
  republishes them in ROS. In the other direction it puts the arm's
  `/joint_states` on the bus as `tatbot.arm-joints/1`, the measured joints
  `stencild` poses its wrist views by. `rmw_zenoh` cannot talk to plain zenoh
  applications, so the bridge is its own zenoh session to the existing
  router.
- Porting the cameras themselves is a later step:
  - The D555 is an Ethernet/DDS camera, so any Jazzy host can own it with
    `realsense2_camera`.
  - The PoE decoders and the stencil tracker would become composable nodes
    in one process, so full-resolution frames never cross the network.
  - The camera node moves to JetPack 7.2 first.

`rmw_zenohd` runs on every host with explicit peer endpoints and multicast
off, the same network model as today's zenoh bus. The router never runs on the
viewer box. Launch is per host: one systemd unit per host starts that host's
launch file.

### Packages

| Package | Language | Responsibility | Lines (v1 budget) |
|---|---|---|---|
| `tatbot_hardware` | C++ | ros2_control `SystemInterface` per arm on the Trossen SDK. Owns the e-stop/probe serial reader, hold latch, guards, carriage trip-retract and flight ring. A framework-free core plus a thin adapter; this is the only code that talks to the SDK. | 1,800 |
| `tatbot_description` | xacro/YAML | Two-arm URDF (carriage as a prismatic joint, tool mount frames), `ros2_control` tags, controller config; palette-station frames at M2. The palette geometry itself is `urdf/palette.urdf`, the v11 CAD build's scene asset | 600 |
| `tatbot_interfaces` | IDL | The custom interfaces in §9 | 150 |
| `tatbot_ink` | Python, no ROS | Acquired paths → program: rigid placement, ink bindings, timed chunks, preview | 1,800 |
| `tatbot_motion` | Python, no ROS | Shared station observation geometry; program op → Cartesian samples (time laws, approach/descent/lift, dips) → joints (CLIK on pinocchio) | 1,500 |
| `tatbot_session` | Python (rclpy) | Orchestrator: per-arm executors, ledger, registration flow, palette lock, e-stop continue/land | 1,500 |
| `tatbot_calib` | Python | Station fix, arm registration, tool tips across the axis (probe touches, or the palette camera's joint-6 sweep); candidate → apply | 3,200 |
| `tatbot_bridge` | Python | Shared overhead capture/station producer; zenoh → ROS stencil page poses from `stencild` → `/tatbot/page` and tf `world → page/<pattern_id>` | 300 |
| `tatbot_bringup` | launch/YAML | Launch files, `rmw_zenoh` config, systemd templates | 300 |
| `tatbot_rerun` | Python | ROS → Rerun bridge for the fleet viewer | 300 |

`tatbot_ink` and `tatbot_motion` import without ROS, so the operator node, the
tests and the arm node all run the same compile and planning code.

## 3. Frames

| Frame | Meaning | Source |
|---|---|---|
| `world` | The fixed-camera world that `stencild` publishes page poses in: on the demo stack the overhead D555's colour optical frame | The camera bundle `tatbot ros register` writes with each arm's registration (docs/vision.md, Calibration) |
| `<arm>/base_link` | Root of the `left` or `right` arm subtree, placed in `world` by that arm's registration | Adopted per-arm registration → fixed joint `world → <arm>/base_link` in the generated description (`tatbot_description`), published by `robot_state_publisher` |
| `<arm>/tool_mount` | Printed mount on the finger carriage; rides the carriage joint | URDF |
| `<arm>/tcp` | Working point of the fitted tool (ballpoint ball contact; laser working point 10 mm past the nose) | `workspace.yaml`, from the probe station |
| `page/<pattern_id>` | The printed stencil page: origin at the page centre, x along its 100 mm edge (stencil u, rightward), y along its 150 mm edge toward the top of the print (−v), z out of the paper. This is also Inkmap's chart frame for the stencil fixture. | `stencild` pose via `tatbot_bridge`, then height and tilt from touches (§4.3). A test pins the u/v conversion, and prints transferred mirrored are declared in `tracking.json`. |
| `<arm>/station` | Palette station = `palette_root` of `urdf/palette.urdf`, the v11 CAD frame: z = 0 at the rail tops, origin on the probe axis, +x toward the camera and the e-stop enclosure, +y transverse to the rail | Registered per arm (§7), M2 |
| `station/inkcap_<size>_<n>` | Cap support floors, the `inkcap_*` links of `urdf/palette.urdf`: `inkcap_large_1`, `inkcap_medium_1`, `inkcap_small_1`, `inkcap_small_2`, `inkcap_medium_2`, `inkcap_large_2`, the L-M-S-S-M-L crescent from +y to −y. A rim is the floor plus the cap's outside height (`config/palette.yaml`), 32 mm above the rail tops. Programs abbreviate the slots `L1 M1 S1 S2 M2 L2` in the same order (`M1` = `inkcap_medium_1`). | CAD offsets in `urdf/palette.urdf` |

Arms are named `left` and `right` only. Leader/follower is a teleop role, and
tape colours are labels, not identifiers.

## 4. From acquired DBV3 artwork to paper

### 4.1 Generate and place

`tatbot drawingbot generate` acquires ordered metric paths at the intended
physical dimensions and generation pen width. It produces shared
`tatbot.inkmap-artwork/2` with a frozen recipe identity. Inkmap imports that
record for review and rigid placement and exports `tatbot.inkmap-design/1`.
The public SVG tracer is separate from production robot preparation.

Preparation places artwork on the nominal stencil page, 100 × 150 mm with a
62 × 112 mm clear centre. Both the rotated canvas and the stroke footprint
must fit. A print of another size (an 83 × 127 mm half-skin print,
docs/stencil-frames.md) is drawn on as it was generated: the session takes the
page's size, clear centre and inner border edges from the installed print
(§4.3, Page) and refuses, before the arm moves, a drawing whose lines leave that
print's clear centre. Translation, rotation and
mirror reuse acquired paths. A size change requires DBV3 regeneration; an
editor resize preview cannot silently become a scaled robot drawing.
Generation pen width and measured deposited tool width remain separate.

### 4.2 Prepare: `tatbot_ink`

```sh
tatbot ros compile artwork.json --at=0,0 --width 20 --speed 3.5 -o prepared
# Or prepare an Inkmap design containing those same acquired artworks.
tatbot ros draw prepared/program.json
```

`--at` is the canvas centre in page millimetres (x right, y toward the top of
the print). `--width` asserts the acquired canvas width in millimetres; it
does not rescale. `--stencil DIR` names the generated print the design is
drawn on (its artwork directory, with `tracking.json` and `settings.json`):
the program's page is that print's, its clear centre the largest one about
the page centre inside the border's innermost ink, so a design that does not
fit is refused here rather than at the arm. `ros draw` takes the same flags
when it prepares a design. `--speed` sets `draw_speed_m_s` after converting mm/s to m/s.
Preparation is offline Python with NumPy, PyYAML and the shared stdlib
contract package. The local CLI resolves the [locked drawing environment](../docs/drawingbot.md#prepare-and-review)
through `uv`, then executes its interpreter directly; no separate package
installation or activated test environment is required. It imports no ROS
runtime, simulator, Java or Node. Deployed ROS processes keep their own runtime.

1. Verify the frozen artwork/program, design and placement digests before
   applying command-line placement overrides. Admit DBV3 unfilled metric
   paths only. Reject unresolved paint, masks, variable deposition and curved
   targets. All placements share one plane page.
2. Place each path in source order with
   `p = anchor + Rot(rotation) · Mirror · (q − canvas/2)`.
   Preserve direction, pen/layer/path IDs, closedness and repeated passes.
   No secondary fill, simplification, deduplication or scheduling occurs.
3. A single acquired pen is drawn by the arm's fitted tool, its ink not named
   (the `fitted` resource). Several pens need a `tatbot-inks/3` file binding
   each pen to a resource: a tool, an ink, a pen mode and, for a dipping tool,
   its palette cap (its dip settings are the datasheet's). A `tool_change` op starts each
   resource's run of strokes; a dipping resource also gets a `dip` before its
   first stroke and whenever the next contact would pass its `mm_per_dip`.
   Ballpoints carry their own ink and get no dips.
4. Split computational chunks using the planner's Cartesian time law,
   including easing, corners, speed caps and control ticks. The default
   budget is 60 seconds. Each chunk retains the source path ID and arc
   interval. Incoming `continues` means stay pen-down from the previous
   chunk. The executor retains its tested outgoing lift and ledger-aware
   resume decisions. Computational chunks do not represent physical lifts
   or ink replenishment. IK can add execution time beyond this model.
5. Write `program.json` (`tatbot-program` v2) and `preview.svg`. Each stroke
   retains `points_m`, `closed`, `continues`, unique `id`, physical ink and
   source identity. The source addresses are
   `src.{placement,artwork_sha256,program_sha256,layer,path,pen,closed,arc_m}`.
   A chunk-relative arc `a` maps to source arc `src.arc_m[0] + a`.
   Points remain double-precision page metres; splitting preserves shared
   endpoints exactly. An unsplit closed path retains `closed: true`; split
   pieces explicitly include the closing segment.

The program retains input-file identity, actual placement, source/recipe/
artwork identities, tool datasheet hash, generation widths, ink bindings,
preparation source/runtime identity and motion configuration hash. Tool width
is measured, assumed or unknown; unknown stays null. An unknown-width preview
uses the generation width and labels that assumption. It never invents a
physical measurement or changes the geometry to match a tool.

`stats.time_estimate` reports the Cartesian drawing component and approximate
descent, settle, lift and travel costs. Unmeasured pen-change/pause costs stay
explicit, with `total_s: null`. `est_s` is the known subtotal. Setup, IK
retiming and interruptions are excluded. `duration_estimate.pen_change_s`
and `pause_s` in `motion.yaml` may supply measured additional workflow costs.

An explicit resource binding file (illustrative catalog/tool IDs):

```yaml
format: tatbot-inks
version: 3
substrate: paper_pad
resources:
  - id: black
    tool_id: black-ballpoint
    ink_id: black
    pen_mode: press
    activation: {method: manual, action: exchange}
  - id: red
    tool_id: red-ballpoint
    ink_id: red
    pen_mode: press
    activation: {method: manual, action: exchange}
bindings:
  - {artwork_sha256: "<exact artwork digest>", pen_id: pen-1, resource_id: black}
  - {artwork_sha256: "<exact artwork digest>", pen_id: pen-2, resource_id: red}
```

Compile with `tatbot ros compile artwork.json --inks resources.yaml -o prepared`,
then `tatbot ros draw prepared/program.json --arm right`. The first resource is
the one the run starts with: load its cartridge first. At a tool change to
another resource the arm lands and the run stops, naming the cartridge to fit;
fit it and resume with `--resume <run-id>`, which keeps the run's measured page
(§4.4). A change to a resource of the same tool and ink goes on without
landing; a resource may ride at its own `ride_fraction` of the stroke instead
of `motion.yaml`'s. `--from-op` starts at an op, but not past a tool change. A
version-1 program is refused: prepare it again.

Directly authored geometric diagnostics use the same version-2 program with
preparation adapter `native-diagnostic-to-ros/2`: one tool and ink, and only one
analytic straight line, one square, two centred perpendicular cross lines, or a
ladder of three or more equal parallel lines (each rung may be its own resource
at its own `ride_fraction`: a ride-height ladder from one page measurement).
Other geometry is refused; finished artwork goes through DBV3.

Preparation tests cover native traversal, repeated passes, closed-path
chunking, explicit colour sequence, physical bounds and width provenance. The session tests run compiled continuations through the real
Cartesian leg builder with substituted IK/driver, preserving resume/lift
behavior. Physical qualification remains a separate DBV3 A/A drawing trial.

Draw runs record monotonic attempt durations in `run.jsonl`, with a
`draw_timing` summary in `meta.json`. Resumes aggregate all completed attempts;
missing attempt evidence leaves the total unknown. These measure action time,
not physical contact time. The research cost fields are defined in
[`docs/research.md`](../docs/research.md).

### 4.3 Register

- **Arm:** `tatbot ros register --arm right --adopt`, with the stack up on
  the arm, places the arm's base in the overhead D555's frame from its wrist
  tags (docs/vision.md, Calibration). A Touch MODE_MOVE without a guard takes
  the arm through the holds. Each hold is a Cartesian travel, or a joint-space
  one when the planner refuses it (from rest on the joint limits). Either way
  the move is kept clear of the page. A hold whose measured tool is not at the
  hold is not captured, and three in a row stop the run. The arm lands at the
  end, and a landed arm idles until it is woken, the session refusing every
  hold: wake it (`ros2 run tatbot_session client wake --arm <arm>`) before
  registering again. Adoption writes
  `~/tatbot-ros/calib/arm-registration-<arm>-current.json`, which the launch
  turns into the `world → <arm>/base_link` joint on its next start. The
  run's `chain` reports what the rigid fit leaves: the wrist tags' seat and
  joints 1–4's offsets. `tatbot ros chain <run-id>...` pools registrations
  of one arm (docs/vision.md, Calibration).
- **Tool:** the fitted tool's TCP comes from `config/workspace.yaml`, written
  by the probe-station calibration (§7). The stated `--ee-tool` must equal the
  fitted tool, or the run refuses before connecting.
- **Page** (per run, per arm):
  - The operator puts a printed stencil page in front of the arm and
    installs its `tracking.json` (with a coded print's `coded.json` and the
    generator's `settings.json`) on the camera node, then points
    `page.pattern_id` at it, or names it at `tatbot ros up --pattern <id>` (the
    hex digits its sheet prints are enough). `ros up` reads the page's size, clear centre and
    inner border edges from that installed print (`tatbot_session.config.print_page`),
    so a print of any size needs no other configuration; with no print installed
    under that ID, `page.size_m` and `page.clear_m` stand. A coded page must lie
    within 250 mm of the arm's `config/workspace.yaml` pad pivot, where
    stencild searches for it (the drawing spot is about 270 mm from the
    pivot); a legacy floral page anywhere the fixed cameras see it. It must
    be where the arm reaches, and it may move a little during a run.
  - `stencild` tracks it using the PoE cameras and the D555: a coded page is
    decoded in the background (a minute or two per camera on the camera
    node after it is put down or uncovered), then followed by flow with its
    bits re-read every frame. On the demo stack the D555 alone cannot read
    the print. It places the page by its artwork on the table plane instead,
    identity unverified, with the print's top away from the arm (docs/vision.md).
    The wrist camera and the touches then measure it. `tatbot_bridge` publishes the page pose, its
    covariance, and the print ID, verified in every measured sample of a
    coded page. The arm resting over the page after a touch hides it from the
    anchoring camera; a draw needs it measured within `page.max_lost_s`. At
    run start it waits up to `page.wait_s` (240 s) for a lost print to be
    measured again, which covers the second side of a held pair. An arm not
    at rest (a cancelled draw holds where it stopped, over the print) first
    lifts clear and returns to rest, holding, so a resumed run keeps its
    controller; a rest path that would pass the pen within half the standoff
    of the page is not taken.
  - tf2 crosses the pose into the arm's base through the arm's registration.
  - Before the touches (`page.locate.wrist`), the arm points its wrist D405
    at the page from a few rest-near poses on its own side, maps the frames
    onto the page plane through the overhead pose, and fits the printed
    border's inner edges (per segment, so the pen cone and shadows drop out;
    per-axis scale absorbs the oblique view's error, taken about the page
    centre the views aim at; `page.locate.free_scale` holds the y scale at 1,
    because the far top border yields too few segments to fit one). Each side is scanned about and fitted to the
    print's own inner edge, `page.inner_edges_m` (from the installed print's
    `settings.json` `border_inner_mm`), else ±`clear_m`/2: a coded print's
    knots sit on the lattice junctions by their centres and reach past the
    clear center by a different amount on each side, which against ±`clear_m`/2
    put every coded fit ~0.9 mm off in x and 2 % off in x scale. Outward the
    scan stops where a detection would end 2 mm short of the page's own edge
    (`page.size_m`), so a dark mat beyond a border too faint to find cannot
    pass for it. Inward it starts no deeper than the program's drawing
    reaches, so ink in the clear centre (a resume's, a redraw's) cannot pass
    for the border: started 20 mm in, a resume took a drawn wave for one
    side and put the page 10-13 mm off. A view whose edges fit no three sides
    (a transfer with two bands faint, a lattice whose inner edge steps row to
    row) is matched instead to the installed print's artwork, frame only, at
    2 mm per pixel within 15 mm and 6° of the overhead pose. The poses combine by
    median, with the spread as sigma, and the result fuses with the overhead
    pose by inverse variance on x, y and yaw (`page.locate` sigmas). The run's
    `page.json` keeps both and the fused pose; `locate/` keeps the views.
  - At the start of every run the arm touches three points in the clear
    center, a triangle 12 mm about the page centre, slowly (0.2 mm/s), under
    the touch guard (§8): it trips on the tip force from the joint torques
    (`safety.contact`, over its baseline 1.5 s into the slow leg) or, as the
    backstop, on tip lag.
  - The arm is meant to be rigid, but under load its joints, links and EE
    mount give ~1.5 N/mm at the tip while the encoders keep tracking the
    command (bench 2026-09-26: ~20 mm to ~30 N), so FK is wrong under load
    and a trip reads deep. Each touch's first contact is estimated from its
    recordings: the force ramp extrapolated to zero, and the overhead EE
    fiducial (`/tatbot/ee/<arm>`, from `tatbot_bridge`) against FK of the same
    frame, whose gap opens when the EE stops and the encoders do not; the two
    fuse by inverse variance. (The "camera sat 12–25 mm above the touched
    page" of rule 4 was this give.)
  - The page height is fused along the page normal from every sensor: the
    overhead page, the wrist D405 depth plane and each touch's first contact
    (`page.height` sigmas). A source far from the rest is named in a warning,
    never dropped. An unavailable D405 plane is explicitly reported and contributes
    no wrist height. Each locate view retains paired RGB, raw z16 and metric depth,
    native intrinsics/distortion and units, depth-to-color extrinsics, hardware/host
    timestamps, per-frame measured joints and the FK/page/TCP transforms under `locate/depth*.npz` with matching JSON metadata. Rejection
    evidence distinguishes missing page depth, points outside the prior height band
    and an unsupported plane; the raw capture permits replay before changing fusion.
    `tatbot ros inspect --depth-only` records the same evidence at the current held
    pose without motion, with a fresh D555-owner capture and tracked-page reference
    when available. Native `ros touch --move` remains the way to visit unloaded
    writing poses. Its report separates local paper height/tilt, nominal FK clearance
    and camera-relative tip-to-plane clearance, including observed support and
    the uncalibrated 8 mm camera-to-tool floor. Missing/stale/rejected depth reports
    a fallback reason. This observation is valid only for its capture and changes no
    drawing command; qualifying camera-to-tool geometry and cartridge state is
    required before an explicit calibration preview/apply can adopt a correction.
    This height fusion estimates one page-centre offset. Wrist depth crossed through
    FK shares that model's pose error; it is not camera-relative tip-to-paper feedback
    and does not correct position-dependent arm error. The camera keeps the tilt (`touch.fit_tilt: false`): three
    touches 24 mm apart swung 7–13°. A touch trips with the tip pushed up to
    the top of its stroke, so the fused page is where the tip meets the paper
    there, whether the tattoo machine runs or not. `motion.yaml` `pen.mode`
    `press` draws `pen.press.lift_m` (−2 mm) under it with the machine off;
    `ride` rides `pen.ride.fraction` of the fitted tool's stroke over it with
    the machine running (§4.4, docs/tools.md).
  - Lateral accuracy is bounded by the fixed-camera chain: registration
    median about 4 mm, and the two arms disagreed by 8.7 mm / 2.6°. A design
    placed at least 5 mm inside the clear center stays off the border.
- **Improving lateral accuracy later** (M5): the wrist D405 observes the same
  stencil at run start, which gives an arm-relative page pose. The fixed
  cameras then track only motion relative to that registration, so their
  constant bias cancels.
- **Station:** registered only for dipping tools (§6, §7).

### 4.4 Execute: `tatbot_session`

```
tatbot ros draw program.json --arm right
```

Each arm runs its own executor through its program. The next op is always
the first op that is not `done`.

**The pen** (`tatbot_motion.pen_down`). Before any motion the executor
resolves how the pen-down path meets the touched page, `motion.yaml` `pen.mode`,
and refuses a program prepared for another tool than the fitted one. `press`
keeps the tattoo machine off and presses `pen.press.lift_m`, settling 1 s at
touchdown. `ride` runs the machine and rides `pen.ride.fraction` of the fitted
tool's `stroke_mm` over the page, settling 0 s, since a running tip that dwells
dots the page; it needs the tool's `stroke_mm` and a positive
`tip_out_at_top_mm`, or a tube-end contact reference for a needle cartridge
(docs/tools.md), and a switch that answers, or the run is refused before any
motion. The run's first event says which (`pen down: …`),
and `page.json` records it.

**The pen trim** (`tatbot_motion.trim`, `motion.yaml` `pen.trim`). The
operator's numpad on the ros node moves the pen-down path along its depth
axis while the arm draws: 8 lifts it one `step_m` (0.1 mm, lighter), 2
presses it one step, 5 sets it back to 0, within ±`limit_m` (2 mm). A held key
steps once. The trim belongs to the cartridge on the arm (the program
resource) for the whole run: each change is a `pen_trim` ledger row and a
`pen_trim` event, and a resume starts each cartridge at its last one; a new
run starts at 0.
- Every stroke plan carries, per knot, its depth axis (the page normal: a
  surface would give each knot its own) and `dq_dh`, the joints per metre
  along it with the tool's rotation and the carriage held
  (`clik.depth_sensitivity`). A trim h is added as `q + h·dq_dh` when the
  goal is sent: no new IK, and never off the plan's IK branch (§1, rule 4).
  On the pink arm's drawing poses that lands within 4 µm of the target at
  1 mm and 18 µm at 2 mm (2026-10-04); a full re-plan of a 40 s stroke takes
  4-9 s.
- Strokes run 10-60 s as one goal, so a change does not wait for the next
  one. A stroke goal starts on an explicit stamp, and a change replaces the
  running goal with the same knots from `lead_s` (0.1 s) ahead, easing to the
  new trim at no more than `speed_m_s` (2 mm/s), on the same time base: the
  controller runs the replacement from the knot that holds now, which is the
  running goal's until the change starts. One change eases in at a time; one
  that would not settle before the goal ends waits for the next goal.
- The pre-planned next stroke starts at the trim the last goal ended at. A
  plan from the measured joints (a resume, a drifted pre-plan) eases its trim
  in from 0; a resume trimmed more than `resume_reapproach_m` re-approaches.
- Enter answers a waiting pause or latch with `continue`. After a cartridge
  swap lands the arm, `client draw` waits for Enter, wakes the arm and resumes
  the run (Ctrl-C there leaves it to resume by hand).
- The keypad is `tatbot_session.keypad`, from `stack.yaml` `keypad`: raw
  evdev on `/dev/tatbot-keypad` (`config/udev/99-tatbot-keypad.rules`,
  installed by `tatbot ros deploy`), grabbed so its keys reach no console,
  publishing `/tatbot/keys/<arm>`. An absent pad is waited for.

A riding stencil diagnostic can carry `writing_height_reference`: a completed
run id, the digest of a `page` that run measured and a positive lift of at most
3 mm. The run still touches its page. The executor holds a minimum writing
height relative to the tracked overhead pose, then applies the lift to the
writing buffer, so a lower fresh touch estimate cannot cancel it. The reference
requires the same fitted tool and calibration and physical print, a target
within 45 mm, and tracked pose agreement within 3 mm and 0.01 rad. It only raises the ordinary path by at most 3 mm and stays below
10 mm hover. An ordinary path already above the requested minimum stays there.
Refusal leaves the normal geometry and guards intact. This is a
local writing experiment, never an adopted TCP or page calibration.

**The tattoo machine** runs only inside a riding draw's op loop. The switch is
`stack.yaml` `machine`: `pi`, the palette Pi's machine relay (§7), or `sim` on
mock and fake hardware, for the arm it names.
- **On** before each stroke goal is sent, which the switch must report within
  `machine.confirm_s` (1 s); the pen-up travel and descent that follow give it
  time to spin up. A machine not reported running pauses the run with the
  switch's reason (the e-stop, no answer, …): `continue` asks again, `land`
  lands.
- **Off** whenever the loop waits for a decision (a latch, an uncertain op),
  changes a tool, dips or ends, however it ends, and at once when the arm latches. So
  locate, the touches and any Touch goal run with it off: a running tip on a
  0.2 mm/s touch would strike one spot for ~30 s.
- The session sends the Pi `MCH1 <seq> <on>` every 20 ms, off as well as on,
  and counts the machine running only on an answer to a command sent since it
  asked. The Pi powers it only while those keep coming, and cuts it on the
  e-stop by itself (§7, §8).
- The events say `machine on` and `machine off`; `meta.json` records each arm's
  `machine_switch`.

**Run start.** The executor takes the page pose (the stencil's newest
`/tatbot/page` crossed into `<arm>/base_link` through tf, or `page.fixed.<arm>`
with `page.source: fixed`). It then moves in joint space to the tool over the
page centre at the standoff. The arm rests at its staged pose on the
`joint_1`/`joint_2` lower limits, which the Cartesian planner keeps clear of.
With `touch.enabled` it
makes the three page touches (§4.3) at (−s, −s), (s, −s), (0, s) about the
page centre, s = `touch.spread_m` (12 mm): touches spread over the clear
centre landed on the printed border. Each touch starts `touch.start_above_m` (10 mm)
above the last touched page, or above the camera's page after a restart, and
lifts `touch.lift_m` (20 mm) after its trip. `plan_touch`
descends fast to `slow_above_m` before that height and then slowly, and the
session arms the tip-lag guard (`guard_mode = 1`) at the first slow knot, so
the fast leg's own lag cannot trip it. A camera page reading more than 6 mm
too low would meet the paper unguarded in the fast leg; the carriage contact
trip is the backstop, and the camera has so far read high, never low. The
touches give the height; the tilt stays the camera's unless `touch.fit_tilt`
is on (it is off), when the plane through the three contacts gives both. The
camera keeps x, y and yaw. The correction `inv(camera) · touched` holds the
run's page at the touched height; a page that moves later stops the run
(below). With `touch.enabled: false` (mock hardware) the camera or fixed pose
is used as it is.

**Stencil tracking** needs a print its wrist fit can read. A print the wrist
camera has read nowhere in a goal (the seed-102 W thermal transfer on silicone,
2026-10-05) is drawn on the measured page without hover stops, said once.
With `page.track.mode: log`
(`tatbot_session.track`), the wrist D405 stays open through a draw's strokes.
After each stroke's final lift a worker process fits a fresh frame to the
print's knot lattice (`scripts/vision/stencil_tip.py`):
- The fit works on any seed. The frame's own depth gives the paper's plane.
  Every lattice translation is scored by its support, and a margin under 40
  is refused. It reaches about two lattice steps from the run's page, so a fit
  that cannot find the print there is tried again from the overhead's page,
  when the overhead has seen the sheet move. A sheet the overhead sees moved
  past `reach_m`/`reach_rad` from the run's page is set up again: on a
  resume, and mid-draw when a hover finds no print. No stroke goes down
  blind from a page the sheet has left.
- Each measurement is a page event and a row of `<run>/track.jsonl`, with its
  frame under `<run>/track/`. It records:
  - the pen tip on the print (the gauge's tip point plus `tip_offset_m`, kept
    in the wrist camera's axes: it turns with the pen, not the sheet) against
    the stroke's planned end;
  - the print's pose in the base against the run's first measurement.
- With `log` it moves nothing.
- With `correct`, every accepted fit is also a sample of where the pen inks less
  where the stack believes the TCP is: the goal's five gauge holds before the
  first stroke, and every lift after.
  - Each stroke's points move by the field of those samples at them, capped
    at `field.cap_m`. `track.Field` is an affine fit of the samples (a slid,
    turned sheet), plus what that fit leaves, averaged by distance (the arm's
    position-dependent error), both weighted by age. So the ink lands on the
    plan after the sheet slides or turns, and wherever the arm's error is.
  - With `hover_stop`, the arm stops over each stroke's start (`plan_hover`)
    and measures there before it descends, so the stroke takes the field at
    its own start. A sheet moved between strokes is caught before the next
    one inks. The travel is pre-planned with the stroke, which is planned from
    its end.
  - A pen-down chain keeps one field. A pre-plan whose correction moved more
    than `replan_m` by its dispatch is planned again.
  - A sample far from the field is held. A second confirms a real move when
    the two jumps are one rigid move of the sheet: they keep the samples'
    distance, which a lattice misfit does not. The older samples then go.
    A jump at a hover is measured again from a second view, `second_view_m`
    toward the page centre.
  - Each stroke's correction is in its `sent` ledger row and an event.

**Page motion.** Before each op the executor compares the newest page pose
with the one the run located and touched. When it moved more than
`page.replan_translation_m` (50 mm) or `page.replan_rotation_rad` (0.17 rad)
the run lifts a pen that is down and stops with `page moved …: not adopted`,
before the op is sent, so the ledger resumes it; the arm holds for a decision.
The new pose is never adopted: while the arm hides the print the tracker can
publish a false match (2026-09-30: 112 mm and a quarter turn away, adopted,
put the pen 4 mm under the paper and dragged the sheet and its mat), and a
page that really moved needs new touches. A new run locates and touches it
again. Smaller jumps are ignored: the tracking moved ~13 mm twice while the
arm was over the page (2026-09-26).
- The arm hides the page for up to 105 s. While it is hidden, the last
  measured pose stands.
- Motion during a stroke is logged, never a reason to stop. Contact force is
  capped mechanically, so page motion is a drawing-quality issue, not a
  safety one. Every op is planned from the arm's measured
joints while the previous op is still running, then checked against the
measured pose at dispatch:

| Drift at dispatch | Pen down | Pen up |
|---|---|---|
| Joints | 10 mrad | 20 mrad |
| Tip | 0.5 mm | 2 mm |

A plan outside these is re-planned. The carriage seed is the last command,
because its encoder rests 10–20 µm off. Each op is one
`FollowJointTrajectory` goal with knots at 50–100 Hz, which JTC interpolates
at 400 Hz.

| Op | Motion (initial values; the tuning source is `tatbot_motion/config/motion.yaml`) |
|---|---|
| `stroke` | 1. Pen-up travel to 10 mm above the start: tip ≤ 120 mm/s, joints ≤ 1.0 rad/s, one accelerate/cruise/decelerate profile.<br>2. Descend at 10 mm/s; the last 5 mm at 3 mm/s.<br>3. Settle at touchdown: 1 s pressing, 0 s riding.<br>4. Draw at the program speed (3.5 mm/s default; tip ≤ 20 mm/s, joints ≤ 0.375 rad/s) with quintic end eases and corner slow-downs (velocity-change budget 10 mm/s², S-curve ramps 0.2 s).<br>5. Lift 10 mm at 10 mm/s.<br>An op with `continues` starts pen-down where the previous op ended: the previous op skips step 5 and this op skips steps 1–3. |
| `tool_change` | Nothing, or land for the operator to fit the next cartridge (below). |
| `dip` | Lift from the page, rise over the station, travel to the cap's hover, descend along its axis, dwell, withdraw along the axis, return to the page standoff (§6). |

- **Standoff tool** (the laser prop): the same stroke recipe, with the TCP
  held 10 mm above the page instead of on it. This follows the datasheet's
  standoff; there is no contact phase and no touch plane beyond the three
  page touches made with the metal nose.
- **Ledger:** `runs/<run>/ledger.jsonl` holds `sent` before each goal,
  `done` after the goal succeeds, and `aborted` with the arc position the
  stroke reached. A resumed run skips `done` ops.
  - An `aborted` stroke resumes from its arc position after `continue`.
    The arc is the planned sample nearest the measured tip at the latch.
    `plan_op` joins the path pen-down when the tip is within
    `resume_reapproach_m` of it; otherwise it lifts and re-approaches.
  - A stroke stopped in its final lift, or within `resume_reapproach_m` of
    its end, is recorded `done`; after `continue` the arm only lifts to the
    standoff, so the stroke end is not dotted twice.
  - An op left `sent` by a crash is never retried silently; the operator
    chooses `redraw` or `skip`. `tatbot ros draw --resume RUN` re-opens the
    run (its `program.json` and ledger) and waits for that decision.
  - A torn last line (a crash mid-write) is ended before the next append,
    so the rows written after a resume are kept.
  - A stroke's `done`/`aborted` row carries the measured pen-down samples
    (`line`), so `drawn.svg` shows every goal of a resumed run.
- **Tool changes.** `tool_change` is done at once when its resource is already
  on the arm, and for the initial one (the run starts with it). Otherwise the
  executor stops pre-planning, turns the machine off, records the op `sent`,
  says which cartridge to fit and lands the arm (the driver's sleep pose, controller
  idle); the goal ends `landed`. The operator fits it and resumes the run:
  a `sent` tool change at the head of a resume is taken as made. Another tool
  (not just another ink) must first be fitted in `config/workspace.yaml`,
  deployed and calibrated; the executor refuses a resource whose tool is not
  the fitted one.
- **Retained page.** Every page setup writes a `page` row (the page record and
  the arm's `workspace.yaml` section). A resume with the same fitted tool and
  calibration uses that page again, after a fresh camera pose shows the sheet
  where it was touched: no new touches. A moved sheet, a stale pose or a changed
  calibration sets the page up afresh.
- **Dips** (§6) run through `tatbot_session.cap`. A dipping resource has no ink
  in a new goal: the executor dips before its first stroke, then at the
  program's `dip` ops.
- **Decisions.** A `decide` answers only the pause or latch that is waiting
  when it arrives. One the executor cannot take (a second `continue` sent
  while the first unlatches, or one whose caller timed out) is refused, never
  kept for a later latch. `land` on a busy arm is carried out when its goal
  stops, even if the goal finished without seeing it.
- **Landing** from a pause or latch first lifts a pen that is within the
  standoff of the page along its normal (unlatching with the hold goal when
  the e-stop is released), so the driver's staged move does not drag it
  across the paper. With the e-stop still pressed it lands from the hold.
- **Pipelining.** While a stroke's goal runs, the next stroke is planned in
  the background from that plan's last knot. At dispatch,
  `tatbot_motion.dispatch_drift` compares its first knot with the measured
  joints, and a drifted plan is planned again from the measured joints.

### 4.5 Review

`tatbot ros inspect [--run ID]` looks at a finished drawing with the arm's
wrist D405 (on the ros node, opened through pyrealsense2). It lifts the pen,
aims the camera at the design from three directions, and writes
`<run>/inspect/<time>/`: the raw frames, `page<n>.png` per pose (its frames
mapped onto the page plane at 10 px/mm through FK of
`right/realsense_color_optical_frame`, median-stacked) and `overlay<n>.png`
(the plan and the clear-centre outline on it). The pose is the reachable one
nearest the sleep pose, with free roll about the view axis, keeping the camera
and wrist links on the arm's own side of the design (base −y), clear of the
palette mid-table. The camera sits ~48° off the pen, ~160 mm short of the tip, so the
nearest pose with the pen 15 mm clear is ~180 mm out: ~3.7 px/mm, enough for
gaps, blobs and placement. `analysis.json` holds the border fit (where the print
is against the page the run used: §4.3's fit, against `page.inner_edges_m`, its
scale about the design centre the views aim at, where the ink's shift is
measured), the ink's shift against the plan (the camera
mount's error plus the arm's), the coverage (the planned path with ink within
1 mm) and the ink's placement error on the print. The arm ends at rest,
holding; `tatbot ros draw` then lands it (sleep pose, controller idle) unless
`--hold`. The mount pose is CAD, not fitted, so the plan
overlay can be several mm off; ink against the printed border is limited by
image sampling and print registration.

For an installed coded reference, inspection also decodes the same raw wrist
frames into `print<n>.png` independently of the CAD camera pose. Its
`analysis.json` `print_coordinates` separates placement candidates, aligned
forward gap and uncovered planned path at 1 mm tolerance. Review the raw images
for narrow tool occlusion and existing marks before treating a candidate as ink. It records verified print bits,
held-out lattice error, native pixel size and whether the print scale was
physically measured. The existing ink scorer accepts faint lines at 85% of
local paper brightness. Deep shadow or wide occlusion at the candidate drawing,
conflicting bits and inconsistent placements between views make the result
unavailable. Coverage depends on ink width and blur; it does not establish tip
accuracy or penalize every stray mark. Raw views and refused candidates remain
available for review.


A multi-resource drawing is scored per resource against one shared alignment;
where expected footprints overlap, neither resource is scored there. Inspection
identifies no pigment: colour accuracy needs a measured colour-response model.


Each run directory contains:
- `meta.json`: sha, dirty state, argv, tool, calibration hash;
- `program.json` and `ledger.jsonl`;
- `bag/`: MCAP of joint states, transforms, goals and results, events,
  safety state and optional wrist images;
- `drawn.svg`: the executed tip path, from forward kinematics of the
  measured joints, over the planned strokes.

`tatbot ros logs last` finds the newest run on the `ros` node. The Rerun
bridge shows the same data live.

## 5. Two arms

- **Separate pages.** Each arm has its own page and its own program and
  executor. They share only the palette lock.
- **One page, shared by two arms, later.** It needs:
  - both arms to register the same page;
  - a stroke partition by page region, with a keep-apart band covered by a
    zone lock;
  - a zone lock over the arms' collision model (§8) before both arms work
    within reach of each other at once.
- **SDK connects** are serialized across arms. One arm's failure is recorded,
  not propagated, unless the failure is inside a shared zone.
- **Clearance.** Every goal keeps each wrist 100 mm from anything of the
  other arm, and the rigid links 20 mm apart, by the model in §8. It checks one arm's way against the other
  standing still, so running both at once still needs the zone lock above.

## 6. Ink and dipping

Every tool datasheet declares how it gets ink:

| Supply | Tools | In a program |
|---|---|---|
| `none` | laser prop | No ink ops. |
| `cartridge` | ballpoint | No dips. Another ink is a `tool_change`: the arm lands and the operator swaps the cartridge. |
| `dip` (`ink.mode: real`) | 3RL needle cartridge | A `dip` before its first stroke and whenever the next contact would pass its `mm_per_dip`. A one-pen program for the fitted tool names no cap and never dips (a dry pass). |

**Cartridge swaps.** A program with several pens binds each to a resource
(§4.2). At a `tool_change` to another resource the executor lifts, lands the
arm and ends the goal, naming the cartridge to fit. Fit it, then
`tatbot ros draw program.json --arm right --resume <run>`: the resume is the
acknowledgement, and the run's measured page is used again (§4.4), so a colour
swap of the same ballpoint takes no calibration and no page touches. The
Inlumino saturated colours share `lutin-ballpoint-dot`; their seating differs
by less than the tip calibration's uncertainty. A resource on another tool
must first be the arm's fitted tool (`config/workspace.yaml`, `tatbot ros
deploy`, `tatbot ros up`, `tatbot ros calib run`); the page is then touched
again. There is no automatic changer and no wipe.

**Rinses.** A dipping cartridge can change ink without a landing: a resource
whose `activation` is `{method: rinse, slot: <cap>, ink_id: water}` in the
`tatbot-inks/3` file takes the cartridge from the last ink by a rinse in that
cap. At its `tool_change` the executor dips there once (§6's dip, every check
included) with the tool datasheet's `rinse_dwell_s` (the 3RL: 10 s) and the
tube's end `rinse_above_ink_m` over the declared water (0: at it), the machine
running so the needles flush the tube, then goes on to the next ink's dip. The
water cap is declared like any other (`tatbot ros palette load
inkcap_large_1=water --level-mm ...`). An interrupted rinse withdraws and
rinses again; a resumed run whose rinse was left `sent` rinses again.

**Palette contents.** `tatbot ros palette load M1=nighthawk_black --level-mm
M1=6` records on the ROS owner that cap M1 holds that ink, 6 mm deep over its
inner floor (`none`, `absent` and `unknown` declare an empty cap, no cap,
unknown contents); `tatbot ros palette status` reads it back. Deploy leaves the
owner's `config/palette_load.yaml` alone. Nothing records a volume.

**Aiming: the station touch.** The overhead fix places the palette through the
arm's registration, which has put the probe's ball 3-8 mm from where the arm's
touches find it, against cap bores 7-14 mm across. So a dip aims by the touches:
every `tatbot ros calib run` writes its S1 ball, measured through the tcp it
commands, to `~/tatbot-ros/calib/station-touch-<arm>.json`
(`tatbot_motion.station.StationTouch`); `--station-only` runs S1 alone for
that. The caps follow from the URDF offsets about the touched ball, and the
arm's kinematics and the tcp's length cancel over the 40 mm to a cap. Each
dip's fresh overhead fix supplies the palette's yaw and refuses the touch when
the palette moved since (1 mm, 0.5°), or the tool, joint offsets or
registration changed; a tcp moved under 5 mm (a probe calibration adopted
since) carries the ball with it.

**The dip** (`tatbot_session.cap`, planned by `tatbot_motion.dip`). The palette
lease keeps the other arm's goals out of the palette.
1. The cap from the touched station, its bore and floor from
   `config/palette.yaml`, its ink level from the owner's palette load. The tool
   is its datasheet's body of revolution ending at the loaded TCP.
2. Approach: lift off the page to the standoff, rise until the tool's lowest
   point clears the station's top, turn to the dip's heading, cross to over the
   cap and come down its axis to `hover_m` over the rim. The heading is the
   touches' first, then outward from it in 45° turns until one plans (on the
   demo rig the right arm reaches the caps at -45° and -90°, not most others).
3. Descend the axis until the tool's end stands `above_ink_m` over the
   declared ink, dwell `dwell_s`, retract to the hover, and return to the page
   standoff it left.
4. The machine runs only for the dwell, and only when the datasheet records
   `needle_reach_mm`: the tube's end holds over the ink and the needles'
   strokes pull it up into the tube (operator, 2026-10-04). A recorded reach
   must reach the ink and stay off the floor. With none the dwell holds with
   the machine off.
5. Every tick, planned and measured: the arm's bodies clear of the station's
   parts, the tool on the cap's axis (within what the bore leaves beyond
   `wall_margin_m`) and never under its dwell height inside the cap. A fault
   stops the goal. A travel the CLIK cannot track is planned again at half and
   a quarter speed; a phase that cannot be planned at all leaves an
   `unplanned` `dip_phase` row and ends the goal with the arm where the last
   phase left it.
6. The wrist D405 keeps a frame from the hover and one from the dwell,
   `station/<op>-hover.png` and `-dwell.png`: the cap's opening and the
   nozzle in one view, the evidence of where each dip went.

The first dips (2026-10-05, empty caps, the machine off) ran every phase on
the pink arm at about 70 s a dip from the page and back.

The settings are the tool datasheet's `dip:` block (`tatbot_contracts.dip`),
which the program carries per resource: `above_ink_m`, `dwell_s`, `hover_m`,
`speed_m_s`, `wall_margin_m` and `mm_per_dip`, the latter from
`mm_per_dip_by_ink` once measured for the ink. Measure `mm_per_dip` on the
bench: dip, draw a long line at the working speed until it fades, repeat, and
take a conservative length. A resource binds a dipping tool to its cap with
`slot:`; the 3RL's cone fits the large and medium caps filled to ~70%, not the
small ones until its silhouette is measured.

**Interruptions.** An interrupted phase waits for `continue` or `land`, then
withdraws: along the cap axis when the tip may be in the cap, then up and back
to the page. Then it dips again, or lands. A cancelled dip holds where it is.
While the tip may be in a cap a marker (`~/.local/state/tatbot/in-cap-<arm>.json`)
refuses landing, since the driver's staged landing would sweep the tool
through the cap's wall; the run's next draw withdraws first and clears it.
A new goal (a resume) dips before its first stroke, since the ink may have
dried meanwhile.

## 7. Calibration: the palette station

The probe measures the fitted tool's tip across its axis: opposed side pairs
on the stylus at upright yaws, the tip's offset turning with the tool while
the ball stands still. Its length, its axis's tilt and the joint offsets are
not measured here: from one ball and upright yaws they trade off with each
other and the tip (2026-10-01: fitted tips 64-81 mm long, joint offsets of
either sign), and page touches cancel the tip's length for drawing.
`config/workspace.yaml` `joint_offsets_rad`, when present, is applied to the
joint origins by the generated description and the wrist observer; nothing
here writes it.

The v11 palette combines several things on the one longitudinal 2020 rail:
- six ink caps (L-M-S-S-M-L crescent);
- a touch-trigger probe (2 mm ball, upright);
- a Raspberry Pi camera (an OV64A40 64 MP autofocus module on the palette
  Pi's CSI connector) aimed along −x at the ball, 122 mm away; its closest focus is about 12 cm;
- the E-stop, its NC contact wired to the palette Pi's header;
- an AprilTag 36h11 ID 0 (46 mm black square) on the diamond-shaped base seat; its installed
  orientation is recorded by the CAD runtime export (`palette_tag` in
  `urdf/palette.urdf`).

On the demo rig the overhead D555 stands on a 2020 post on the palette's
assembly, so moving one moves both, and the post rises beside the station
where the arm's wrist can meet it (below).

**Probe signal path**
- The probe is normally closed: it reads LOW at rest and HIGH when touched,
  like an open wire. Its output goes, through a 10 kΩ pull-up to 3V3 and a
  Schottky diode, to the palette Pi's GPIO27 (header pin 13).
- The probe has its own relay instance on the palette Pi. `tatbot ros
  relay-install --probe` installs `tatbot-probe-relay`, which sends
  `PRB1 <seq> <state> <edge_ns> <now_ns>` to UDP 7641 (`stack.yaml`
  `probe.*`) on every node whose stack can drive an arm on the station: the
  ros node, and the arm node for the blue arm. The e-stop relay and its
  `EST1` stream are untouched.
- Each GPIO edge is requested without kernel debounce and stamped by the
  kernel on CLOCK_MONOTONIC. It is sent as it is read, and the level every
  20 ms.
- `tatbot_hardware` is the only parser, and it accepts only the relay's
  address.
  - It maps relay time to host time by the least-delayed frame of the last
    100.
  - It keeps 1024 ticks of measured joints and interpolates them at the
    rising edge's stamp.
- `tatbot ros up --probe` enables it. Without it the driver has no probe, and
  `GUARD_PROBE` holds at once.

**Machine switch** (§4.4)
- The tattoo machine's power switch has its own relay instance on the palette
  Pi, `tatbot-machine-relay` (`tatbot ros relay-install --machine`), driving
  the header line `stack.yaml` `machine.gpio` HIGH to power the machine. It
  refuses to install until that line is recorded.
- It takes `MCH1 <seq> <on>` on UDP 7642 from the ros node's `lan` only, and
  `EST1` on the Pi's loopback, 7643, where the e-stop relay always sends it
  (`tatbot ros relay-install`), installed or not.
- The line is HIGH only while the newest command asks for it and is at most
  `machine.timeout_s` (0.2 s) old, and the newest `EST1` reads released and is
  at most 0.15 s old. Once the e-stop reads pressed or goes silent, the line
  stays LOW until the machine is asked off: releasing the e-stop never
  restarts it, the session asks again.
- It answers each command with `MCS1 <seq> <on> <powered> <estop>` (e-stop 1
  released, 0 pressed, 2 silent). The session counts the machine running only
  on `powered 1`, which is the line as driven, not the motor's current.
- The relay drives the line LOW before it exits. A Pi that halts with the line
  HIGH is covered only in hardware: an e-stop contact block in series with the
  machine's supply, or a switch that needs a live signal to stay on.

**Guarded moves**
- A calibration or touch move arms a guard in the driver: `PROBE` (trigger
  edge) or `TIP_LAG` (commanded-minus-measured tip, the component along the
  commanded tool axis toward the paper, over 0.8 mm for 0.15 s (`stack.yaml`
  `safety.tip_lag`: the in-air lag swings ±0.35–0.45 mm, so a lower threshold trips
  in the air; with a rigid pen 0.8 mm presses 1.5–2 mm into a soft mat); lateral error
  and error away from the paper never count).
- When the guard trips, the driver holds the arm. This is a stop, not a
  fault.
- The guard's trip pose is reported through the safety state. Under `PROBE`
  it is the joints at the edge, not at the later tick.
- A probe that is stale, or already triggered when armed, holds without a
  trip (reason 7, `guard_tripped` 0).
- A probe touch (`Touch` `MODE_SINGLE`, `guard` `GUARD_PROBE`) needs
  `prior_distance_m`, the expected contact along the direction.
  - The session arms the guard at rest before the plan: the probe has no
    dynamic false trips.
  - The approach goes up to 30 mm over the start, across, and down, never a
    straight line past the ball. From rest on the joint limits, a joint move
    comes first.
  - The travel past the expected contact is capped by `motion.yaml` `probe`:
    1.0 mm along the stylus and 2.5 mm across it. A stylus pushed past its
    overtravel (2 mm axial, 4 mm sideways) is damaged. Reaching the cap
    untriggered is a missing trigger, not a fault.
  - The slow leg runs at 1 mm/s. After a trip the tool backs out 3 mm the
    way it came, and the probe must then read released.
- `MODE_MOVE` travels to `start_pose` the same way and holds there (a view of
  the station). Under `PROBE`, a trip on the way is an error.
- Probe calibration starts with a fresh fix and a released arm. Guarded
  opposed side searches from the fix centre the stylus without contributing
  to the fit; edge passes bracket the ball's top; opposed pairs on the tool's
  wall measure. Further upright yaws repeat the pairs, the ball placed by the
  fit so far once two yaws are in. Each touch is an existing ROS Touch goal
  with its own run evidence.
- The probe guard protects the stylus and nothing else. A tool that meets the
  station anywhere else is not stopped by it. The calibration keeps clear in
  five ways:
  - it measures the station fresh from its fiducials every run, never from a
    stored pose (the plan's principle 8);
  - it uses the clearance-plane approach;
  - it refuses any route whose tool would meet a part to avoid: the palette
    camera, the e-stop, the tag, the probe's body and collar, the inkcaps and
    the overhead camera's post. The tool is its datasheet profile, never
    inside the ring the side touches measure, up 120 mm from its end. The
    palette's parts are placed about the ball's current estimate, so they
    follow each touch;
  - the floor (operator, 2026-09-29): no goal whose way takes the tool's
    lowest point more than 10 mm under the ball's top is sent. Until S1 has
    touched the top, the top is the fix's and the floor stands 2 mm higher
    for its height. After, the floor stands under the highest top S1's edge
    passes allow: over a laser's ring they stop just short of the ball and
    read it up to a millimetre under its top (2026-09-29). The stylus is all that stands there: the probe's body
    and collar are 23-26 mm under the top. A search pass 11 mm under a stale
    fix's ball met the probe's base the day the floor came in;
  - it refuses any goal that brings the arm's wrist within 15 mm of the
    overhead camera's post: the wrist's bodies (each face of the tag cube,
    the wrist D405, links 4 to 6) as spheres, the post as a column 60 mm in
    radius about the camera's footprint (the arm's registration places the
    camera; the D555 cannot see its own post). Both checks run on the way
    the executor plans to the goal (`tatbot_session.ready.station_way`, then
    the planner's travel) and on each touch's leg to its cap; a goal with no
    plan is not sent either. The same check refuses what the executor would:
    a joint move from rest that dips more than 10 mm under its lower end
    (`ready.station_dip`), and a way within the collision guard's margin of
    the other arm (§8). The heading and the yaws are chosen only among
    attitudes the guard lets pass. On 2026-09-29 a view turned toward the post
    put the tag cube against it.

**The tool.** A calibration touches the station with the arm's fitted tool,
`config/workspace.yaml` `<arm>.tool_id`, whose tip the stack's tcp is built
from. A stated tool (`tatbot --ee-tool`) must be that one. The datasheet
declares the tool's contact model; `tatbot ros calib run` refuses any tool
whose datasheet declares none (exit 3), before anything moves
(`tatbot_calib.tool`).

The halo model (`calibration.face_kind: halo`) treats the tool's tip as a
tube's end, a ring around a hole, with the ring's outside where side touches
meet the ball, the hole, the ring's middle for rim touches and the side
touches' height up the wall.
- **Laser:** a chrome ring 9.0 mm in radius outside around a 7.7 mm hole, the
  lens recessed in it. Emission off.
- **Ballpoint:** a metal tip 2.4 mm long (0.45 mm across at the ball, 1.6 mm
  at the body) under a body 3.8-4.1 mm across (`face_kind: ballpoint_tip`,
  the palette camera, 2026-10-02). The side pairs touch the tip, the point
  that draws, 0.8 mm up it. A side touch trips only after the arm has given
  about 1.2 mm past first contact (`SIDE_GIVE_M`): the trips lie on a circle
  that much inside the tip and the ball, and the touches are planned there.
- **3RL needle cartridge:** the needles retract into the cartridge with the
  machine off, so everything touches the end of its translucent nozzle, ~1 mm
  across (the palette camera, 2026-10-05). Side pairs touch the nozzle 3 mm
  up, where it is ~1.5 mm across; the search lines are spaced by that side
  radius, so a thin nozzle cannot pass between them and the ball.
- Nothing is planned into a hole the ball could enter.

**Station fix** (`tatbot ros station --arm <arm>`, `tatbot_calib station
fix` on the ros node): eight D555 captures of the palette tag through the arm's
adopted registration.
- The tag is about 23 px there. OpenCV 4.6's sub-pixel refinement leaves its
  corners on whole pixels, and its tilt is then ambiguous by 20°, so the
  corners take the AprilTag edge refinement and the palette is held level on
  the arm's table (`station.level_pose`).
- The tag's range from its corners follows its apparent size, and that follows
  the light: in a dimmed room the tag imaged 1.5% larger and the corners put
  the palette 10.8 mm higher. So each shot moves the level pose to where the
  D555's aligned depth puts the tag's centre (`station.tag_centre_by_depth`);
  the corners keep only its yaw. The depth held the tag's height within
  0.1 mm from daylight to the dim room, and matched the corners' in daylight
  within 0.4 mm.
- A tag unseen in the frame is looked for again in the frame stretched to its
  own gray range: dim frames gave the tag in 2 of 10 shots as they came, 7 of
  10 stretched. The shots' mean is the fix, from the depth-placed shots alone
  when there are any.
- Fixes repeat within 0.1–0.2 mm and 0.04°; `--against` calls a move at 1 mm
  or 0.5° (exit 1). A station the D555 cannot measure (the tag hidden or
  unseen, frames from another camera world than the registration's) exits 3,
  a D555 that does not answer 5.
- The fix only has to land inside S1's search; the touches measure the ball.

**Tool tip** (`tatbot ros calib run --arm <arm>`, `tatbot_calib.program`):
1. The opening station fix, and the arm woken at rest (an arm still awake
   where its last goal left it lands first). A tool without a contact model
   is refused (exit 3) before anything moves.
2. The base heading: the one whose yaws the arm reaches over the ball from
   rest with the planner's joint margin, its wrist clear of the camera's post
   (`reach.choose_heading`). Yaws it cannot reach are left out and named.
3. The way in: the tool upright 30 mm over the fix's ball, or by way of a
   staging pose when the executor plans no way there from where the arm
   stands (2026-09-30: from rest pink's wrist swept within the guard's margin
   of the landed blue arm's).
4. S1: the search, opposed side pairs 4 mm under the fix's ball, or 1 mm
   over a lower side height (1.8 mm up the ballpoint's tip), where the stylus
   stands for a fix up to 2 mm off in height, from starts the tool's
   envelope clears for any ball within 12 mm. A tube's reach across a pass
   line is a few millimetres, so when no pass meets the stylus, pairs run
   along lines offset across, nearer first. These contacts go to no fit.
   Edge passes over the found centre bracket the ball's top, and the floor
   rises to it; opposed pairs at the tool's side height measure.
5. S4: further upright yaws (default -60, +60, -30, +30, ... about the
   heading) until three have measured. The first searches as S1 did; later
   ones start from where the fit of the yaws so far places the ball through
   their own tip, searching only when that meets nothing. A yaw the executor
   will not plan from where the last left the arm, or whose pairs do not
   both trip, is skipped and its contacts dropped.
6. One fit: the tip across its axis, the ball and a side radius per yaw and
   pair axis (each direction's give differs, 1.6 against 2.7 mm half-spans
   on pink); the tip's length, its axis and the joint offsets held. Each yaw
   left out says how well the others predict it, and how far the tip moves
   without it.

Each run writes `touches.jsonl`, `events.jsonl`, `candidate.json` and
`report.md` under `~/tatbot-logs/ros-calib/<run>/`. Nothing is adopted by
the run: `tatbot ros calib apply --arm <arm> --run ID` moves the arm's
`pen_tip_offset_x/y` and `mechanical_contact_x/y` by the fitted change and
names the run in the touch-off record, and refuses a candidate from another
tool than the fitted one. Commit, land and deploy it like any config change.
The held-out numbers are for the operator to read, not a gate: on pink the
pairs' give differs with posture by 0.3-0.5 mm, which no fit removes.

**Tool tip, contact-free** (`tatbot ros calib sweep --arm <arm>`,
`tatbot_calib.sweep`): the same opening fix and wake, then holds the palette
camera watches, each a probe-guarded Touch `MODE_MOVE` through the run's way
checks (the floor, the other arm, the camera's post, the arm against the
station, the tool's envelope against the station's parts and 3 mm off the
fix's ball).
1. The park: the tool upright 6 mm over the fix's ball top, at the yaw whose
   joint-6 turns the camera sees best among those the arm reaches from rest
   (+15 to +45 deg on the v11 palette; the probe run's -60 sees a sixth as
   much).
2. Four 3 mm translations: across the camera's line of sight both ways, up,
   and away along it. They give the camera its scale, orientation and depth.
3. Joint 6 alone through -30..+30 deg in 6 deg steps: FK of the park's planned
   joints with joint 6 turned, so joints 1-5 stay put (to the planner's few
   mrad) and their offsets do not enter. The executor's way to each goes up
   30 mm, across and down, under the probe guard; the jog client has neither
   that guard nor the two-arm one, and re-commands joints 1-5 at their
   measured, sagged values. The turns are sent only when the park's still
   shows the tip 2 mm over the ball's top under the lowest turn.

At each hold the CLI's node takes a full-resolution still on the palette
camera's node and copies it into the run (the ros node holds no key there).
The tip is where the metal's contrast over its row's background ends below
the pen's body; a patch about the ball's top marks the camera's drift. One
pinhole fit gives the tip across its axis in the tool mount. A turn cannot see
the tip along joint 6's axis, so the length is held at the installed one, and
`report.md` gives the factor by which a length error (or the detector's bias
along the pen) reads across: about (0, 1) on pink. It also gives the tip's
height over the ball's top at the park, the probe run's reference. The run
writes `holds.jsonl`, `stills/`, `candidate.json` and `report.md` under
`~/tatbot-logs/ros-calib/<run>/`; `calib apply` adopts it as above.

**Station, per arm:**
- The ball centre from the solve fixes translation.
- Yaw comes from the rails, or from the palette tag.
- Caps come from the CAD offsets in `urdf/palette.urdf`.
- Each cap floor is measured once.

**Arm registration** (the `world → <arm>/base_link` transform):
- `tatbot ros register --arm <arm>` measures it against the D555
  (`docs/vision.md`). A moved camera invalidates it: moving the palette's
  post on 2026-09-29 moved the D555 with it, and through the old registration
  the station fix moved 46 mm and the page 150 mm. Register again after any
  move of the post.
- The station offers a better check: the palette tag is seen by the fixed
  camera, and the probe ball is touched by the arm. With the CAD offset
  between them, this measures each arm's world error directly.

## 8. Safety behaviour

These live in `tatbot_hardware` and act on every connected arm within one
tick.

| Condition | Response |
|---|---|
| E-stop pressed, or the last valid heartbeat is too old (checked by frame age, never a latched flag). The source is configuration:<br>- `serial`: a local Pico, 100 ms limit.<br>- `udp`: frames relayed from the palette Pi, 150 ms limit. Accepted only from the relay's address and only with an advancing sequence number.<br>- `none`: no e-stop input, for mock hardware or whenever the operator chooses. The arm's rocker switch cuts power. | Hold latch: command the pose held at the latch instant, zero velocity, motors powered. Incoming commands are ignored until unlatched. |
| Carriage contact over its cap: external effort 20 N above its 800-sample rest baseline, or 2 mm deflection, for 40 consecutive ticks; effort is judged only below 0.3 rad/s | Retract the carriage 32 mm over 0.6 s (blocking, read back), then hold latch |
| Tracking stall (error over 0.35 rad for 2 s with under 0.05 rad of progress) or measured over-velocity | Hold latch |
| First command more than 0.05 rad from measured, or before 100 ms of non-zero feedback after configure; activation would clamp the measured pose past the same arm/carriage step limits | Refuse before sending the position command |
| Deactivate or error | Hold, never idle. Idle is allowed only after a verified landing: staged pose 4 s, sleep pose 3 s, joints within 0.2 rad and carriage within 0.5 mm. |

**The tattoo machine** is not the driver's. The e-stop freezes the arm and
never cuts its power, and a machine left running over a frozen arm strikes one
spot: on paper a hole, on skin an injury. So the palette Pi's machine relay
cuts it on the e-stop by itself and keeps it off until it is asked off and on
again, the session turns it off at once on any latch, and 0.2 s without the
session's commands turns it off at the Pi (§4.4, §7).

Exclusive ownership: the driver takes the existing arm-driver lock
(`/tmp/tatbot-arm-driver.lock`) before connecting, so it can never run beside
`wxai_teleop`, `arm_recover` or LeRobot.

**The other arm** (`tatbot_motion.collision`, in the session, not the
driver). Every goal `ArmIO.execute` sends is checked first against the other
arm; a goal refused here is a failed goal, not a latch.
- **The model.** Both arms are placed in the D555's world by their
  registrations. Each link is the convex hull of its URDF collision mesh, and
  each fitted tool is a chain of capsules along its datasheet profile.
  Distances come from pinocchio on coal, about 0.3 ms a pose on the ros node
  and 50-60 ms a goal.
- **The rule.** Each arm's wrist and end effector (links 5 and 6, the
  carriage, the wrist D405, the tool) are bubbled out by the reach of their
  cable loops, `motion.yaml` `collision.wrist_m` (80 mm). The leads are tied
  along the forearm and loop to the tool and the D405 so the wrist can turn,
  so they cannot be modelled.
  - A pair of bodies keeps the larger of its two bubbles plus
    `collision.arms_m` (20 mm). A loop must not reach a rigid part of the
    other arm, but two loops may brush, so two wrists keep 100 mm apart, not
    180.
  - A wrist keeps 100 mm from anything of the other arm, and rigid links keep
    20 mm from each other.
  - The whole way is checked, not only its end: a pose wherever a joint has
    moved 10 mrad (7 mm at full reach), and the last one.
  - A refused way is not sent. The one exception is a way that never gets
    nearer than it starts, so an arm already that close can back away.
    Backing away is checked too: a straight joint move from the snag below
    back to rest swings the wrist nearer, and is refused.
  - Nothing re-plans around a refusal. A draw's move waits for `continue`
    (re-planned the same way from the measured joints) or `land`. A Touch
    goal fails with the reason. The calibration checks the same guard before
    choosing its heading and views.
- **Why 100 mm.** On 2026-09-29 a pink view planned 83-87 mm from the landed
  blue arm (the pen to the laser prop, by this model), and the pink wrist's
  own loops caught on the blue arm: the pen wand's RCA power lead and the
  wrist D405's USB. Views at 133-147 mm did not.
- **Where the other arm stands.** Where this stack measures it, when it
  drives both. Otherwise at its landed pose (`follower.staged_positions`).
  A peer that drives it elsewhere, such as the LeRobot policy on the arm
  node, publishes no joints, and the refusal says the pose was assumed.
- **The arm against itself** (2026-09-30: an inspection view folded the pink
  wrist tag cube into its own upper arm). The same model pairs one arm's own
  bodies whose joints stand more than two apart; links within two joints
  overlap as hulls about the compact wrist in every pose.
  - The wrist tag cube, which the URDF gives no collision, is a box solved
    from its three tags' calibrated frames (69-70 mm). It is paired with
    every link from link_5 up, not the parts it rides with, and only with
    its own arm: across the arms the wrist's bubble takes its place.
  - A way is refused when two of those bodies come within `collision.self_m`
    (10 mm) and nearer than where it starts, or as near as they stand at
    rest (the landed arm folds onto itself), less 1 mm.
  - `Guard.self_gap(arm, q)` gives a single pose's clearance from itself, for
    choosing views.
- **Both arms moving at once** (2026-09-30: pink drawing while blue
  calibrates, in one stack `--arms right,left`). Where the other arm stands
  says nothing of where its goal takes it next, so:
  - Each goal's way is reserved while it runs (`Guard.reserve`, under the
    guard's one lock). A new way is checked against the other arm where it
    stands and against every row of the other arm's way in flight, each row
    of one against each row of the other. When each arm will be where is
    left out, which only ever refuses more. Long ways are sampled to 60 rows,
    the coarser sampling's slack taken off the margin.
  - A way blocked only by the other arm's way in flight waits up to 30 s for
    it to end (`ArmIO._way_clear`). A standing refusal fails at once. A way
    let through is released when its goal ends, whatever the outcome.
  - A landing, which the driver sweeps in joint space from where the arm
    stands to the staged pose and then the sleep pose, goes unchecked but
    holds its way, so the other arm's next goals keep clear of it.
  - **The palette** (`tatbot_session.lease`): a calibration run holds it by
    an exclusive flock on `/tmp/tatbot-palette.lease`, writing the zone it
    keeps. The zone is a cylinder along the table's normal about every part
    of the palette, 20 mm past the outermost and over the tallest. The other
    arm's ways into the zone wait for the lease to end; a second run is
    refused while one holds it; a crashed holder's lock goes with its
    process.
  - A landed arm is woken alone (`client wake`), so neither agent's run
    restarts the stack under the other; `tatbot ros up` restarts both.
- **Not checked:**
  - the separate-process `ArmIO` of `client.py` (inspect, rest, lift, jog);
  - the driver's own retract;
  - a stack where either arm has no registration, which it logs at start.

Real time:
- `SCHED_FIFO` and core pinning are set before the SDK drivers are
  constructed, so the SDK's threads inherit them.
- Error queries run off the control thread, once a second
  (`controller_query_period_s`). The SDK holds one data mutex for every
  call's whole round trip, so each TCP query stalls the 400 Hz loop by
  1.5–2.5 ms; at 10 Hz that was about 120 late ticks a minute on the pink
  arm. A controller error also arrives as an SDK exception on the next read
  or command, which latches at once.
- The follower carriage limits (−6 to +40 mm, `config/trossen/follower.yaml`)
  are set with `set_joint_limits` at configure: a power-cycled controller
  boots with a −4 mm floor, and the loaded carriage rests near −4.7 mm.

**E-stop release.** The arm stays held, and `tatbot_session` asks the
operator to choose `continue` or `land` (§9). `continue` is refused, leaving
`land` as the only choice, when:
- the controller reported an error;
- the measured pose moved more than 0.05 rad or 5 mm from the latched pose;
- the fitted tool changed;
- the release has waited longer than `resume_timeout_s` (default 300 s).
  The arm then announces and lands.

The pose check compares the joints measured at the latch with the joints
measured now (0.05 rad on any revolute joint, or 5 mm of tip). The tool check
compares the program's `tool.id` with `config/workspace.yaml` read at the
decision. The same rules apply to a JTC goal that failed without a latch,
because the arm then holds under JTC. With nothing waiting, `decide continue`
on a latched arm runs the unlatch protocol, and `decide land` lands. A `land`
during a running goal cancels it, records `aborted` and lands.

Results of each choice:
- **`continue`** unlatches and re-plans the interrupted op from the measured
  pose. A stroke resumes at its arc position, re-approaching if the tip is
  more than 0.5 mm off the path. A dip restarts.
- **`land`** unlatches and lands to idle. The arm then takes no goal until the
  stack restarts (`tatbot ros up`).

## 9. Interfaces

Custom interfaces (`tatbot_interfaces`); everything else is standard:

| Name | Kind | Contents |
|---|---|---|
| `Draw` | action | goal `{program, arms[], from_op}`; feedback `{arm, op_id, index, total, phase}`; result `{done, skipped, uncertain}` |
| `Touch` | action | goal `{mode: SINGLE\|PAGE\|MOVE, arm, start_pose, direction, max_travel_m, speed_m_s, guard, prior_distance_m}`; result `{tripped, contact_poses, joints, plane, run_dir, message}` |
| `Land` | action | goal `{arms[]}` |
| `Decide` | srv | `{arm, decision: CONTINUE\|LAND\|SKIP\|REDRAW}` → `{accepted, reason}` |
| `Event` | msg | `{stamp, arm, kind, op_id, text}` |
| `SafetyState` | msg | Per arm: `{estop_ok, estop_age_s, probe_triggered, latched, latch_reason, guard_tripped}` |
| `Page` | msg | `{stamp, pattern_id, print_id, identity_verified, pose (world), covariance, support, source}` from `tatbot_bridge` |

Standard interfaces:
- `sensor_msgs/JointState`
- tf2
- `control_msgs/FollowJointTrajectory`
- `geometry_msgs/PoseStamped`
- `sensor_msgs/Image`
- `sensor_msgs/CameraInfo`

## 10. What this was ported from

The motion and registration logic retains its existing owners. Artwork paths
come from DBV3; the former generic fill and stroke scheduler are removed:

| Stage | Ported from |
|---|---|
| Acquired metric paths | `scripts/lib/drawingbot/` and shared `tatbot_contracts` |
| Corner-cut addressing | `scripts/lib/stroke_material.py` |
| Time laws and approach/descent/lift | `scripts/lib/pen_path.py` |
| Damped-least-squares CLIK and its gains | `cpp/teleop/square_probe.cpp` |
| E-stop reader, real-time setup, flight recorder, landing sequence | `cpp/teleop/` |

The drawing session it replaced (`sessiond`, `tatbot-fsm`, `reconstructd`, the
Python session layer, `tatbot session`, `tatbot draw` and `tatbot dip`) was
deleted on 2026-09-26. Never revive it from git history. Not carried over
either: the µL ink model, the fixed-camera world, the per-chunk scan/touch
overhead, and six other forward-kinematics copies.

Kept running and bridged in:
- `visiond`, `trackd` and `stencild` on the D555's owner (roles `overhead-depth`, `track`), which track the page
  and the EE fiducials;
- the adopted camera and arm registrations;
- Inkmap and Inkgen, which author designs.

## 11. Depth

| Sensor | Use today | Use here |
|---|---|---|
| D555 (overhead) | Inside `stencild`: lifts matched stencil points into 3D, giving the page pose and plane (plane RMS about 1 mm). Also a pen-blob "height", which reads a 6 mm floor whatever the true height. | Stays inside stencil tracking, for page pose and presence. Never used for height. |
| D405 (wrist) | The pink arm's D405 is this stack's own (no `visiond` unit opens it). Measured before the stack: depth repeats to 0.1 mm but sat 12–13 mm above the paper on the pink arm, so touch overrides its height; a tip-over-page check read 8.6 ± 2.4 mm against 10 mm compiled. | Not needed for flat paper at M1. Later uses: (a) a page-plane prior in the arm frame, so touches start slow descent 4 mm above the page instead of searching 25 mm, about 8 s per touch instead of up to 50 s; (b) the height for the standoff laser, which should not touch the paper; (c) arm-relative stencil registration (M5); (d) curved-surface scans (M6). |

## 12. Build and run

The `ros` role in `config/nodes.json` names the stack's host (the arm node
that drives the right arm). `tatbot ros <verb>` runs anywhere: its backend
(`scripts/lib/tatbot_cli/tatbot_ros.py`) acts on the `ros` node over ssh inside
`~/tatbot-ros`, or locally on that node. It never uses that node's own git
checkout, so the verbs declare no role and never hop.

**Workspace on the `ros` node** (`tatbot ros deploy` writes it):

```
~/tatbot-ros/
  repo/        rsync of this checkout's tracked ros/ config/ urdf/ scripts/ python/tatbot_contracts/ rust/visiond/config/ AGENTS.md (+ REVISION)
  build/ install/ log/   colcon, --symlink-install
  deps/        FetchContent cache: the Trossen SDK v1.8.5 (-DFETCHCONTENT_BASE_DIR)
  calib/       arm-registration-<arm>-current.json copied from the calibration conductor
  inbox/       programs sent by `tatbot ros draw`
  env.sh       from tatbot_bringup/scripts/env.sh.in: ROS, install/setup.bash, TATBOT_REPO, rmw_zenoh
  stack.env    TATBOT_ROS_ARGS=hardware:=... estop_source:=... and TATBOT_ROS_EFFECTIVE (the merged
               hardware/estop/page/arms, shown by `tatbot ros status`), written by `tatbot ros up`
  runtime.json the running tatbot_session's runtime identity, written at its start (the launch's
               `runtime_record`); `tatbot ros evidence` reads it
  .build-pending  packages synced but not yet built (deploy writes it right after rsync)
```

```bash
source /opt/ros/jazzy/setup.bash
cd ~/tatbot-ros
colcon build --base-paths repo/ros --build-base build --install-base install --symlink-install \
  --cmake-args -DCMAKE_BUILD_TYPE=Release -DFETCHCONTENT_BASE_DIR=$HOME/tatbot-ros/deps
export PYTHONPATH=$HOME/tatbot-ros/repo/python/tatbot_contracts/src:$PYTHONPATH   # env.sh's, for the tests
colcon test --base-paths repo/ros --build-base build --install-base install && colcon test-result --verbose
python3 -m pytest -q repo/ros/tatbot_estop_relay/test      # COLCON_IGNOREd, stdlib
```

Packages find repo files through `TATBOT_REPO` (`~/tatbot-ros/repo`; in a
checkout, the checkout). Nothing reads a development checkout by its path.

Run tests without `env.sh`: they need ROS, the install under test and
`tatbot_contracts` on `PYTHONPATH`, as above. `env.sh` points `ROS_DOMAIN_ID`
and zenoh at the live graph, and it sources the deployed install, so a test
from another checkout would start the deployed nodes instead of its own build.
The harnesses give every stack they start its own `rmw_zenohd` on a free port
and a random domain, and keep its logs and the session's runtime record in
their tmp dir, never `~/tatbot-ros/runtime.json`.

**Deploy** (`tatbot ros deploy`, from a worktree's `scripts/tatbot`):
1. `rsync` mirrors exactly the tracked files under `ros/ config/ urdf/ scripts/
   python/tatbot_contracts/ rust/visiond/config/` and `AGENTS.md` (working-tree
   content; `DEPLOY_DIRS` in `tatbot_ros.py`) into `repo/`, deleting anything else
   there. `AGENTS.md` is the marker by which `tatbot_runlog` finds
   `config/runlog.json`, so runs on the node land in `~/tatbot-logs`.
2. It copies `arm-registration-*-current.json` from this node's
   `~/tatbot-logs/vision/` (the calibration conductor's) into `calib/`.
3. It writes `repo/REVISION` (sha, `dirty=0|1`) and renders `env.sh`.
4. It builds only what changed: every package on the first build or with
   `--clean`, else the packages whose files `rsync` changed plus those above
   them (`colcon build --packages-above`), and nothing when no `ros/` file
   changed. The changed packages go into `.build-pending` right after `rsync`,
   and only a successful build clears it, so a `--no-build`, a failed or an
   interrupted deploy is built by the next one. Python is `--symlink-install`ed. Measured end to end
   from the operator node into an empty `~/tatbot-ros`: 56 s for the first deploy
   (SDK fetch and the nine packages, 50 s of colcon), 1.7 s for a one-file
   Python change, 0.8 s when nothing under `ros/` changed.
5. `--units` renders and installs the two units with sudo and enables the
   router. It also installs `config/udev/99-tatbot-estop.rules`, which
   `--estop serial` needs for `/dev/tatbot-estop`.

A parallel workspace on the same node (a lane, the integration run) sets the
environment: `TATBOT_ROS_ROOT=~/tatbot-ros-lanes/<task>`, `TATBOT_ROS_DOMAIN`
and `TATBOT_ROS_PORT` (defaults `~/tatbot-ros`, 0, 7450). Every verb acts on
that root. The installed units run `~/tatbot-ros` only; for any other root,
`up` starts transient units named after its directory,
`tatbot-ros-lane-<dir>-router.service` and `tatbot-ros-lane-<dir>.service`
(`sudo systemd-run`, the same limits and kill signal as the installed units, that
root's `env.sh` and `stack.env`), and `down` and `status` act on those.
`deploy`, `up`, `down` and `relay-install` write a `ros-cli` run log on the node
that ran them; `tatbot ros logs --workflow ros-cli` reads them there.

**Units** (system units running as the login user, installed by
`tatbot ros deploy --units` with sudo from `tatbot_bringup/systemd/*.in`):
- `tatbot-ros-router.service`: `rmw_zenohd` listening on `tcp/127.0.0.1:7450`.
- `tatbot-ros.service`: `ros2 launch tatbot_bringup stack.launch.py $TATBOT_ROS_ARGS`,
  `LimitRTPRIO=95`, `LimitMEMLOCK=infinity`, `KillSignal=SIGINT`; requires the router.

**Graph isolation.** `ROS_DOMAIN_ID` 0 and router port 7450 in production.
Sessions connect with `ZENOH_CONFIG_OVERRIDE='connect/endpoints=["tcp/127.0.0.1:<port>"];scouting/multicast/enabled=false'`;
the router gets `listen/endpoints=[...]` (same multicast setting) instead. The
explicit endpoint and multicast off keep the graph on 127.0.0.1. The tatbot zenoh bus (7447 on
the bus router) is a different network that only `tatbot_bridge` joins. A
parallel graph on the same host (a lane, the integration run) takes its own
domain and port, e.g. 51/7451, 52/7452, 55/7455, 60/7460, and its own
workspace `~/tatbot-ros-lanes/<task>/` with the same layout. Stop what you
started by PID: `ros2 run` and `ros2 launch` wrap the real process, so killing
the wrapper's `$!` can leave the node running. Signal the process group or run
the binary directly. Never use `pkill -f`, which matches your own shell.

**Checks.** `scripts/check ros-budget` (fast tier): ≤20k lines of code, ≤8
interfaces, ≤3 launch files. `scripts/check ros` (`--full` or by name): the
relay tests, then `colcon build` and `colcon test` of `ros/` in
`~/.cache/tatbot-check-ros/<checkout hash>/`; SKIP without `/opt/ros/jazzy`.

## 13. Names

Every cross-package name. `tatbot_description/names.py` holds them in code;
`tatbot_hardware/include/tatbot_hardware/names.hpp` mirrors the GPIO part and
a test keeps the two equal.

### Packages and executables

| Package | Build | Executables / entry points |
|---|---|---|
| `tatbot_interfaces` | ament_cmake, rosidl | the 7 interfaces of §9 |
| `tatbot_hardware` | ament_cmake, C++ | plugin `tatbot_hardware/TatbotArm` (`hardware_interface::SystemInterface`) |
| `tatbot_description` | ament_cmake + Python | `tatbot_description.robot_description()`, `tatbot_description.names` |
| `tatbot_bringup` | ament_cmake | `launch/stack.launch.py`, `config/{stack,controllers}.yaml`, `systemd/*.in`, shared CLI driver in `scripts/lib/tatbot_cli/tatbot_ros.py` |
| `tatbot_ink` | ament_python, no ROS | `python -m tatbot_ink compile …` |
| `tatbot_motion` | ament_python, no ROS | `config/motion.yaml`; `import tatbot_motion` (planners, §13 Python APIs) |
| `tatbot_session` | ament_python, rclpy | `session` (node `tatbot_session`), `client`, `keypad` (node `tatbot_keypad`) |
| `tatbot_bridge` | ament_python, rclpy + zenoh | `bridge` (node `tatbot_bridge`) |
| `tatbot_rerun` | ament_python, rclpy | `rerun_bridge` (node `tatbot_rerun`) |
| `tatbot_estop_relay` | `COLCON_IGNORE`, stdlib | `tatbot_estop_relay.py`, `tatbot-estop-relay.service.in` |

`tatbot_calib` is M2. Test file basenames are unique across `ros/`.

### Frames and joints

- `world`: the root of the description. `<arm>/world_joint` (fixed) places
  `<arm>/base_link` from `stack.yaml registration.<arm>`
  (`world_from_arm_base`); without one, at the nominal `urdf/tatbot.urdf` mount,
  which the launch refuses for a real arm on the stencil page (see Launch).
- Arm subtree: every link and joint below `<arm>/base_link` in `urdf/tatbot.urdf`,
  meshes as `file://<repo>/urdf/meshes/…`.
- `<arm>/tcp`: fixed joint `<arm>/tcp_joint` from `workspace.yaml` `<arm>.tip_frame`
  (`<arm>/tool_mount`) at `pen_tip_offset_{x,y,z}`, identity rotation. Its +z points
  along the tool toward the paper, so drawing holds tcp z = −page z.
- `page/<pattern_id>`: §3. `tatbot_bridge.page.TARGET_FROM_PAGE = diag(1, −1, −1, 1)`:
  `world_from_page = world_from_target · TARGET_FROM_PAGE`. With `page.source: fixed`, the
  pattern id is `fixed_<arm>` and the parent is `<arm>/base_link`.
- Controlled joints, in controller order: `<arm>/joint_0` … `<arm>/joint_5`
  (rad), then `<arm>/left_carriage_joint` (m, the tool rides the left finger
  carriage). The description sets the carriage limits to the arm's controller
  config (`config/arms.json` `controller_config` `joint_limits[6]`, to 1 µm):
  −0.006 … 0.040 m on the right (follower), 0 … 0.040 m on the left.

### ros2_control

One `<ros2_control name="<arm>_arm" type="system">` per arm, rendered by
`tatbot_description/urdf/ros2_control.xacro`:
- `hardware:=mock`: `mock_components/GenericSystem`, GPIO initial values
  `estop_ok` 1 and the rest 0; the joints start at the staged pose
  (`config/trossen/tatbot.yaml` `follower.staged_positions`).
- `hardware:=fake`: `tatbot_hardware/TatbotArm`, `sdk=fake`: the built-in fake
  SDK with every interlock live and no network or driver lock.
- `hardware:=real`: `tatbot_hardware/TatbotArm`, `sdk=real`: the Trossen SDK. It
  takes `/tmp/tatbot-arm-driver.lock` (one per process) and never calls
  `load_configs_from_file`.

Joints: command `position`, `velocity`; state `position`, `velocity`, `effort`.
With `sdk=real`, `effort` is the SDK's external effort (`external_efforts`:
total minus the controller's gravity and friction compensation, as
`rust/tatbot-arm` reads it); the carriage contact cap is judged on it.

`TatbotArm` hardware parameters, in the order of `names.HARDWARE_PARAMETERS`
(all text; lists comma-separated; `tatbot_description.hardware_params()` fills them):
`arm`, `sdk` (fake|real), `ip` (`real` only, from
`config/arms.json` `profile_ip_field` in `config/profiles/<profile>.json`; empty
for `fake`, which is never handed an arm address),
`end_effector` (`arms.json` `sdk_end_effector`), `estop_source`, `estop_device`,
`estop_timeout_s`, `estop_udp_port`, `estop_relay_addr`,
`estop_debounce_frames`, `rt_priority`, `rt_cpus`, `base_frame`, `tcp_frame`,
the `stack.yaml safety.*` values flattened with `_` (for example
`tip_lag_trip_m`, `carriage_retract_s`, `landing_budget_s`), the carriage cap,
deflection and retract, and `staged_positions` from `config/trossen/tatbot.yaml`
`follower` (the pink arm's qualified carriage; every `TatbotArm` reads that section),
plus `flight_path` (the `ros-stack` run directory). `fake` alone gets `fake_page_z`
after them when `stack.yaml fake.page_z` is set (it is, at `page.fixed.right`'s z,
so `tatbot ros up --hardware fake --page fixed` touches and draws on it): the base-frame z of a simulated
paper plane the fake tip cannot pass, so an offline `Touch` trips its guard. `estop_timeout_s` is
`udp_timeout_s` for `udp` and `serial_timeout_s` otherwise. `real` without an
address in the profile is an error at launch. The e-stop reader and the driver lock are process-wide,
shared by both arms' instances. SDK drivers are constructed on a thread that
first applied `SCHED_FIFO` and the CPU pin (cpp/teleop `realtime::apply`). The
error getter runs every `controller_query_period_s` (1 s) off the control thread. The tip-lag
guard gets FK from the URDF in `HardwareInfo::original_xml` (KDL,
`base_frame` → `tcp_frame`): the tip and the tcp +z axis, along which it
measures the lag.

**Inside `tatbot_hardware`** (about 1,750 lines):
- `core.{hpp,cpp}`: `ArmCore`, every §8 interlock for one arm, one tick at a
  time, with no ROS, SDK or clock. The adapter feeds it measured feedback, the
  JTC commands, the GPIO commands and the e-stop status, and sends what it
  returns. `ContactCap` is `rust/tatbot-arm/src/contact.rs`; the stall watchdog is
  `tracking_watchdog.rs` over joints 0–5; the landing is `arm_recover.cpp` run
  inside the control loop (minimum-jerk staged and sleep moves, settle 0.15 s,
  then verify).
- `tatbot_arm.{hpp,cpp}`: the ros2_control adapter (lifecycle, parameters,
  interface export in the URDF's joint order, the aux thread for TCP
  getters and the post-landing idle, the flight log).
- `sdk_backend.cpp`: the SDK calls forked from `trossen_arm_ros` (jazzy) under
  its BSD licence: `configure(wxai_v0, end_effector, ip, clear_error=true, 5 s)`,
  `get_robot_output`, `set_all_positions(q, 0, false, qd)` with the JTC velocity
  as feed-forward, `set_all_modes`. Upstream idles on deactivate and sends no
  feed-forward; this fork does neither. It never calls `load_configs_from_file`.
  Upstream pins SDK v1.9.0 only in `dependencies.repos`; every call used exists
  in v1.8.5, which CMake fetches (`TATBOT_ARM_SDK`, cached under
  `FETCHCONTENT_BASE_DIR`; a cache checked out at `v1.8.5` never touches the
  network, any other checkout is re-fetched).
- `fake_backend.cpp`: `sdk=fake`. First-order tracking (τ 20 ms) of the
  position target plus the velocity feed-forward, one 2.5 ms step per command.
  The hardware parameter `fake_page_z` (base-frame z, `sdk=fake` only) adds a
  page the tip cannot pass, so a touch trips the tip-lag guard. Tests also
  inject controller errors, carriage effort, measured velocity and a frozen arm.
- `estop.{hpp,cpp}`: the one EST1 parser and reader (below).
- `runtime.{hpp,cpp}`: `realtime::apply`, the driver lock (`DriverLease`
  exclusive mode: `O_NOFOLLOW`, owned regular file, `flock(LOCK_EX|LOCK_NB)`,
  never unlinked; `driver busy` fails configure), and the flight recorder.

Behaviour the table in §8 leaves open:
- Hold means `set_all_positions(held pose, 0, false, zeros)` every tick, motors in
  position mode. Deactivate, error, cleanup and shutdown send that hold once
  more and never idle. While the hardware is inactive, ros2_control calls
  neither read nor write, so the controller keeps that one target and the
  e-stop is not evaluated until re-activation.
- The SDK's `cleanup()` (also run by its destructor) idles the controller, so
  the SDK session is closed only after a verified landing. Cleanup or shutdown
  of an arm that has not landed (SIGINT, `tatbot ros down`, a unit restart)
  leaves the session open with the driver lock, holding the last sent pose,
  until the process exits. The controller firmware (1.8.0 and later, including the rig's 1.8.4) idles when its
  connection is lost, so an arm that has not landed can drop when the stack
  process ends. `tatbot ros down`, and `tatbot ros up` on a running stack, land
  every arm that has not landed first (`client land`; `--hold` skips it) and
  leave the stack up when a landing fails. `--hold` is refused (exit 3) while
  the session's in-cap marker says a tool may be in a palette cap: on
  2026-10-05 a held restart let the pink arm sag 9 mm and 6.7 mm aside with
  its nozzle in a cap. A cleanup → configure cycle of a held real arm in
  the same process cannot reconnect (the controller accepts one driver); restart
  the stack.
- Configure waits up to 1 s for a non-zero measurement (the SDK reports an
  all-zero pose until its daemon has robot output) and fails without one. It
  then keeps reading for `feedback_warmup_s`, so the command interfaces that
  activate seeds with the measured pose pass the first-command check at once.
  Activation first checks that clamping the measured pose stays within the existing
  arm/carriage step limits. An encoder counted a turn outside its limits is refused
  before position mode or a hold command; power-off manual repositioning is required.
  The first command counts only after `feedback_warmup_s` of non-zero feedback.
- A deactivate → activate cycle keeps a latch from before the deactivate (any
  reason but the deactivate hold's own 9) and holds the measured pose until a
  Decide; a latch that is only the deactivate hold clears on activate.
  Re-activating while a JTC goal is still running (the controller stays active
  and keeps advancing) holds with reason 10 until a Decide.
- An unlatch request while the e-stop is not released, or while the controller
  reports an error, is acked and stays latched with that reason (1, 2 or 8). A
  refused unlatch overwrites `latch_reason` (1, 2, 8 or 10) on purpose, so the
  session reads why the Decide failed; the first reason stays in the driver's
  log line, the session's `latched` Event and the flight log.
- The contact cap is judged while tracking commands; the landing is not judged
  by it (its sleep move closes the carriage, which reads as deflection). Its
  force is not judged while the carriage target moves or for
  `carriage_settle_ticks` after (a commanded lift is not contact), and then a
  `carriage_rebaseline_samples` rest median at the held target becomes its
  baseline (holding the 2 mm bias takes ~27 N more than resting on the floor);
  its deflection is judged throughout. E-stop,
  controller error, over-velocity and stall stay live during a landing.
- A landing that has not verified within `landing_budget_s` holds with reason 4.
  A land request while the e-stop is not released, while the controller reports
  an error, or during the carriage retract is acked and refused: `landing`
  stays 0 and the driver logs `land refused: <why>`. An unlatch during the
  retract is acked and refused the same way.
- A controller error latches until the aux thread reads the controller healthy
  again; an SDK exception on the control thread is sticky until the stack
  restarts (the SDK expects its process to end after one). A controller fault
  latches reason 8 and Land is refused while it lasts: restart the stack
  (`tatbot ros up`, which skips the landing of an arm reporting a controller
  error; configure clears the error with `clear_error=true`), then land or
  continue. Never use `tatbot arm recover`
  for this arm: it belongs to the LeRobot arm node, not the ROS node.
- `release()` waits 0.5 s for the aux thread. One stuck in a blocking
  TCP call (the controller gone from the network) is detached and keeps the SDK
  session, so a stop never waits for systemd's SIGKILL; activate refuses after
  0.5 s on the same stuck call.
- Every sent position is clamped 0.1 mm/mrad inside the URDF limits.

**Flight log.** `<flight_path>/<arm>-flight.bin`: one ASCII header line
`tatbot-flight 1 <arm> <record bytes>`, then one packed little-endian record
per write(): `t` (float64, s, steady clock), then float32 ×7 each of `q`, `qd`,
`effort`, the incoming command `cmd_q`, the sent `sent_q` and `sent_qd` (NaN
when nothing was sent), float32 `estop_age_s`, `rt_period_max_ms`, then uint8
`estop_ok`, `latched`, `latch_reason`, `phase` (0 inactive, 1 running, 2
latched, 3 retracting, 4 landing, 5 landed): 188 bytes. `numpy.fromfile` with
the matching dtype after skipping the header reads it; a slow disk drops whole
records, never a tick. `tatbot_bringup/scripts/flight_summary.py RUN_DIR [--arm]`
prints the bench M0 numbers from it as one JSON line: the longest period and
its jitter over 2.5 ms, each latch (reason, e-stop state and age, speed before,
time until still), the resting carriage external effort, and the signed tip
lag along the commanded tool axis while moving.

GPIO `<arm>_safety`, all doubles:

| State interface | Meaning |
|---|---|
| `estop_source` | 0 none, 1 serial, 2 udp |
| `estop_ok` | 1 released and fresh (always 1 for none) |
| `estop_age_s` | age of the last valid frame; −1 for none or before the first |
| `probe_triggered` | 0/1: the probe's latest frame reads triggered (or a broken wire) |
| `latched` | 1 holding the latched pose, ignoring commands |
| `latch_reason` | 0 none, 1 estop, 2 estop_stale, 3 carriage_contact, 4 stall, 5 over_velocity, 6 guard_tip_lag, 7 guard_probe, 8 controller_error, 9 deactivated, 10 step_refused |
| `guard_mode` | armed guard: 0 none, 1 tip_lag, 2 probe |
| `guard_tripped` | 0/1 until the next accepted unlatch |
| `trip_q0`…`trip_q6` | measured joints at the trip (rad; carriage m); NaN before any |
| `unlatch_ack`, `land_ack` | the last request id the driver processed |
| `landing`, `landed` | the landing sequence runs; verified landed and idle |
| `controller_error` | 0/1 |
| `rt_period_max_ms` | longest read-to-read period over the last second |

| Command interface | Meaning |
|---|---|
| `guard_mode` | level: 0 none, 1 tip_lag, 2 probe |
| `unlatch` | request id: a finite value different from the last one seen is one request |
| `land` | request id, the same rule |

Controllers (`tatbot_bringup/config/controllers.yaml`, controller_manager
`update_rate: 400`):
- `joint_state_broadcaster` publishes `/joint_states`.
- `<arm>_arm_controller` is `joint_trajectory_controller/JointTrajectoryController`
  with position and velocity commands, position and velocity state, splines, and
  `set_last_command_interface_value_as_state_on_activation: true`. On activate
  and on unlatch the driver sets its command interfaces to the measured pose.
  Its action is `/<arm>_arm_controller/follow_joint_trajectory`.
- `<arm>_safety_controller` is `gpio_controllers/GpioCommandController`,
  present in Jazzy 4.42, at 100 Hz. It takes `control_msgs/msg/DynamicInterfaceGroupValues`
  on `/<arm>_safety_controller/commands` and publishes the same type on
  `/<arm>_safety_controller/gpio_states`.

JTC tolerances are all 0, which Jazzy reads as unchecked (its defaults are
`stopped_velocity_tolerance` 0.01 and everything else 0): JTC never aborts or
fails a goal on tracking, because the driver's stall guard is the tracking
interlock and the arm lags about 70 ms (§1). A goal succeeds when its time
ends. `enforce_command_limits` stays at its default, false; the arm
controller enforces its own limits.

On `mock_components/GenericSystem` with this `controllers.yaml`, all three
controllers load and activate with the slash joint names, `gpio_states` carries
the 21 state interfaces in order, a `unlatch` command is accepted, and a
one-point JTC goal succeeds. `tatbot_bringup/scripts/mock_check.py` checks this
on the whole launch (below).

`tatbot_session` republishes the GPIO states as `SafetyState` on `/tatbot/safety`,
one message per arm, at 20 Hz and on change.

### E-stop

- Frames: `EST1 <seq> <state>\n`, where state 1 means released and 0 pressed, at
  100 Hz. The state is debounced over 3 frames.
- `serial`: the stack stops when the last valid frame is older than 0.10 s.
- `udp`: the driver binds `0.0.0.0:7640`. A datagram is exactly one frame from
  `estop_relay_addr` (IP match). A frame counts only if its seq is greater than
  the last one accepted, so a reordered or replayed frame is dropped. The stack
  stops past 0.15 s. After a stale stop the sequence re-seeds, so a restarted
  relay recovers after 3 frames, and the arm stays latched until a Decide.
- `none`: `estop_ok` is always 1.
- The relay sends one frame per datagram to `<ros node lan>:7640`. The source
  is `stack.yaml` `estop.source`, `udp` (the relay address comes from the
  `estop-relay` node's `lan` in `config/nodes.json`, or `--relay-addr`), or
  `tatbot ros up --estop serial|none`. Mock hardware reads no e-stop; a fake
  or real stack beside a running one needs `--estop none` or its own port.
- Garbage, malformed or merged lines and datagrams over 128 bytes are ignored;
  the status is judged by the age of the last valid frame at every tick, never
  by a latched flag. The serial source reopens the device every 0.5 s.
- **Relay** (`tatbot_estop_relay/`, on the `estop-relay` node): stdlib Python,
  `tatbot_estop_relay.py (--gpio NAME | --device PATH) --dest HOST:PORT [--bind ADDR]`.
  - `--gpio GPIO17` is the palette: the button's NC contact between GPIO17
    (header pin 11) and GND (pin 9). The relay holds the line with its pull-up
    through the kernel's GPIO character device and sends the raw level at
    100 Hz with its own sequence number: LOW (closed) is released, HIGH
    (pressed or a broken wire) a stop. A failed line is silence and is
    reopened every 0.5 s.
  - `--device` is a Pico running `firmware/estop_pico/`: the relay forwards
    every line unchanged, since the Pico already numbers its frames.
  - `--bind` fixes the source address when the Pi has several interfaces.
- `tatbot ros relay-install [--dest HOST:PORT ...] [--bind ADDR]` copies the relay,
  its unit and `config/udev/99-tatbot-estop.rules` to the `estop-relay` node over
  ssh (it needs only ssh, python3 and sudo there), installs the udev rule when
  the source is a Pico, and enables `tatbot-estop-relay.service` with `stack.yaml`
  `estop.relay_gpio` as its source (`""` = the Pico). It is the one installer.
  Without `--dest` the relay sends to every reader of the button: this node's
  driver (the ros node's `lan`, `estop.udp_port`) and, when the arm node's profile
  e-stop is the relay (`driver.estop_device` `udp://:PORT?from=estop-relay`), every
  `estop` node's `lan` at that port. Each reader accepts frames only from the
  relay's address and judges its own heartbeat, so one button stops both arms and
  a datagram lost on the way to one is a stop there alone. `--bind` pins the
  relay's source address, which must then be the stack's `--relay-addr`.
- Relay tests: `python3 -m pytest -q ros/tatbot_estop_relay/test` (pty Pico,
  emulated GPIO line, reopen, unit) and `test_hw_relay` (emulated Pico → relay
  → the driver's reader: loss, delay, reorder, wrong source, relay death,
  rebooted Pico).

### Continue and land, never a step

A goal's acceptance is awaited five seconds; one accepted after a stop or that
timeout is cancelled at once.

1. On a latch, the driver commands the latched pose with zero velocity.
   `tatbot_session` cancels the active JTC goal, writes `aborted` with the arc
   at the latch, and emits `latched`. On e-stop release it emits `released`.
2. `Decide CONTINUE` runs these steps:
   1. Require `estop_ok`.
   2. Send a one-point hold goal at the measured joints (zero velocity,
      `time_from_start` 0.1 s) and wait for it to succeed.
   3. Write `unlatch = N`.
   4. Wait at most 1 s for `unlatch_ack == N`.

   The driver unlatches only when every incoming position command is within
   `step_limit_rad` (0.05) of the latched pose, and the carriage within
   `step_limit_m` (0.005). Otherwise it keeps holding, sets `latch_reason = 10`
   and acks. `latched == 0` after the ack means accepted, and the session
   re-plans the op from the measured joints (a stroke from its arc).
3. `Decide LAND` or the `Land` action writes `land = N`. The driver runs the
   `arm_recover` sequence, which never loads configs: takeover 0.5 s, staged 4 s,
   sleep 3 s, verify ≤0.2 rad and ≤0.5 mm carriage, budget 45 s. Only then does
   it go idle with `landed = 1`, and it stays idle until it is woken. `client
   wake --arm <arm>` wakes that arm alone: its trajectory and safety controllers
   are deactivated, its hardware component alone is cycled inactive and active
   (the driver re-enters control at the rest it measures and holds it, as at the
   stack's start; the first command must start there), and the controllers come
   back, the JTC from the measured pose. The other arm keeps running, so two
   agents can share the stack (2026-09-30). The next `tatbot ros draw` or `calib
   run` wakes its landed arm this way and waits for it and, when the page comes
   from the stencil tracker, a measured page (a fixed page is never on the bus).
   `tatbot ros up` restarts the whole stack, both arms. Its JTC would still report every goal a
   success with the arm at rest, so the session refuses a Draw or Touch goal for
   a landed arm at acceptance, before any run opens, and sends it no trajectory
   at all (`client jog` and `inspect` fail too). A rejection carries no reason:
   the clients read `landed` from `/tatbot/safety` and say to run
   `tatbot ros up`. An e-stop during landing aborts it into a hold. Before
   writing `land`, the session lifts a pen within the standoff clear of the
   page when the e-stop is released (§4.4 Landing).
4. Deactivate, error or shutdown: hold (`latch_reason` 9 or 8), never idle.
5. First command: refused (held, reason 10) when it is more than 0.05 rad from
   measured or arrives before 0.10 s of non-zero feedback. Configure waits out
   that warm-up, so the measured pose seeded at activate is never refused.
6. JTC SUCCESS is not execution: every JTC tolerance is unchecked, so a goal
   "succeeds" while a latched driver holds or a landed one idles (step 3). The
   session writes `done` only when `<arm>_safety` `latched == 0` and
   `latch_reason` is unchanged at the result, otherwise `aborted` at the arc.
7. Guards (`Touch`): the session writes `guard_mode = 1` at the first slow
   (`PHASE_TOUCH`) knot of the touch plan (`tatbot_motion.guard_arm_time_s`);
   the fast leg is unguarded and the carriage contact trip is its backstop. A
   trip holds (reason 6) and records `trip_q*`. The session computes the contact
   by FK, writes `guard_mode = 0`, then unlatches as in step 2 and lifts.
   `guard_mode = 2` is written at rest before the plan and waited for. A trip
   (reason 7, `guard_tripped` 1) is unlatched the same way, and the tool backs
   out; a hold without a trip is the probe refusing.

### Launch

The only launch file is `tatbot_bringup/launch/stack.launch.py` (the budget
leaves two). Every argument defaults to `stack.yaml` when empty:

| Argument | Values |
|---|---|
| `config` | `stack.yaml` path (default: the checkout's `ros/tatbot_bringup/config/stack.yaml`, the file `robot_description()` and the CLI read) |
| `repo` | the checkout (`$TATBOT_REPO`) |
| `runtime_record` | where `tatbot_session` publishes its runtime identity (default: `runtime.json` in the workspace root holding the repo, `~/tatbot-ros/runtime.json`) |
| `arms` | comma list; `right` by default |
| `hardware` | `mock`, `fake` or `real` |
| `profile` | `config/profiles/<profile>.json` |
| `estop_source` | `none`, `serial` or `udp` |
| `estop_device` | serial device |
| `estop_udp_port` | UDP bind port |
| `estop_relay_addr` | the relay's source address |
| `probe` | `true` or `false`: read the station probe (PRB1) |
| `probe_relay_addr` | the probe relay's source address |
| `machine_addr` | the machine relay's address (`tatbot ros up` resolves it for a `pi` switch) |
| `page_source` | `stencil` or `fixed` |
| `pattern_id` | the installed print to draw on (`tatbot ros up --pattern`) |
| `touch` | `true` or `false` |
| `rerun` | `true` or `false` |
| `record` | `true` or `false`: a stack-wide MCAP |

It starts:
- `robot_state_publisher` with `robot_description(arms, hardware, registrations, …)`;
- `ros2_control_node`, with `thread_priority` from `rt.priority`;
- spawners for `joint_state_broadcaster`, `<arm>_arm_controller` and `<arm>_safety_controller`;
- `tatbot_bridge`, `tatbot_session`, and `tatbot_rerun` when `rerun` is set;
- with `record` (default `stack.yaml record: false`), `ros2 bag record` of
  `names.bag_topics(arms)` to MCAP in the run's `bag/`.

It opens a `ros-stack` run (`tatbot_runlog`, finalized on shutdown) holding:
- `stack.yaml`, the effective configuration with the arguments applied.
  `tatbot_session` gets its path as the ROS parameter `stack`, `tatbot_bridge`
  and `tatbot_rerun` as `config`, and they read everything else from it;
- `robot_description.urdf`, as given to `robot_state_publisher`;
- `ros/`, the `ROS_LOG_DIR` of every node;
- the driver's flight rings (`flight_path` is the run directory).

An arm whose `registration.<arm>` file is missing hangs at its nominal mount
and the launch logs it, except that `hardware:=real` with `page_source:=stencil`
is a configuration error: the page arrives in the camera's world, so the launch
raises and names the missing file (`tatbot ros deploy` copies it; `fixed` still
runs without one). When `rt.cpus` is set, `ros2_control_node` also gets
`cpu_affinity`; empty leaves it unpinned (the driver still puts its SDK threads
on the fastest cores).

The launcher freezes the controller parameters it supplies in `controllers.yaml`
and records its controller process in `controller-process.json` using
the process-start event. Research run metadata and evidence identify that
process's loaded executable files, including native plugins, and the launch
configuration and robot description; see
[research runtime provenance](../docs/research.md). Missing provenance affects
research comparability only. `tatbot_session`
publishes its runtime identity once at startup to the effective stack's
`runtime_record`, where `tatbot ros evidence` reads it. A session started
without one, such as `ros2 run` on the installed default, publishes none.

The source snapshot records the session/motion/description/bridge/contracts
package roots and the owner's CLI/vision source trees. It hashes a stable
recursive Python-file manifest, including membership changes during collection.
Current evidence reports whether these declared files still match startup.
Research comparison refuses absent or changed source snapshots. This is source-file
provenance, not attestation of Python objects in memory or every external
dependency. Shared runtime helpers serve the existing session and owner/research
tools; SDK and motion authority remain in the ROS stack.

The run's exit status is 1 when any process exits with a failure (not 0 and
not a SIGINT/SIGTERM death) before shutdown begins; `run.jsonl` records a
`process_failed` event for each. `controller_manager` exiting for any reason
shuts the whole launch, and so `tatbot-ros.service`, down. A launch that fails
while it is being generated finalizes its run with 1.

**Mock check.** `python3 tatbot_bringup/scripts/mock_check.py [--domain N]
[--port P|0] [--arms right] [NAME:=VALUE ...]` starts its own `rmw_zenohd` and
the launch with `hardware:=mock page_source:=fixed touch:=false`, then checks
that every controller is active, `/joint_states` arrives at ≥360 Hz, the
safety GPIO states are the 21 in order with `estop_ok` 1, and a
FollowJointTrajectory goal (0.05 rad on `joint_0` and 2 mm of carriage, then
back) succeeds, moves, and leaves `latched == 0` and `latch_reason == 0`. It
prints one JSON line and stops everything it started. Extra arguments override
the launch's, so `hardware:=fake` runs the same checks on `TatbotArm` with its
fake SDK. The session's runtime record always goes into its logs directory. `colcon test` runs it on a free port as `test_mock_stack` (mock) and
`test_fake_stack` (fake).

### Graph

| Name | Type | Owner |
|---|---|---|
| `/tatbot/draw` | action `Draw` | tatbot_session |
| `/tatbot/touch` | action `Touch` | tatbot_session |
| `/tatbot/land` | action `Land` | tatbot_session |
| `/tatbot/decide` | srv `Decide` | tatbot_session |
| `/tatbot/events` | `Event` | tatbot_session |
| `/tatbot/safety` | `SafetyState` | tatbot_session |
| `/tatbot/goals` | `trajectory_msgs/JointTrajectory`, every goal sent (for the bag) | tatbot_session |
| `/tatbot/keys/<arm>` | `std_msgs/String`: `up`, `down`, `reset`, `enter` per numpad press (§4.4) | tatbot_keypad |
| `/tatbot/page` | `Page` | tatbot_bridge |
| tf `world → page/<pattern_id>` | dynamic tf | tatbot_bridge |
| bus `tatbot/session/ros/arm/<arm>/joints` | `tatbot.arm-joints/1` (zenoh, not ROS) | tatbot_bridge |
| `/joint_states`, `/robot_description`, `/tf_static` | standard | JSB, RSP |

The bridge dials the bus router's `lan` address, `tcp/<lan>:7447`
(`tatbot_cli.nodes.bus_endpoint(address="lan")`) as a zenoh client.
It subscribes `tatbot/tracking/target/<pattern_id>` (`tatbot.target-pose/1`).
- A `measured` sample publishes `Page` (`SOURCE_MEASURED`, stamped with the
  capture time, the sigmas as a diagonal covariance, stencild's `support` as
  JSON) and the tf `world → page/<pattern_id>`.
- A `lost` sample republishes the last measured pose with `SOURCE_LOST` and no
  tf. Before any measured sample, nothing is published.
- `/tatbot/page` is reliable and transient-local with depth 1, so a late
  subscriber gets the newest page.
- `page_source: fixed` publishes `stack.yaml page.fixed.<arm>` as a static tf
  `<arm>/base_link → page/fixed_<arm>` and a `SOURCE_FIXED` `Page` per arm at
  1 Hz.
- The arm transforms are `robot_state_publisher`'s. The bridge logs the page
  in each `<arm>/base_link` through the adopted registration every 10 s.
- The one thing it puts on the bus is each arm's measured joints, for
  `stencild`'s wrist views (its only joint source). `tatbot_bridge/joints.py`
  owns the format, `tatbot.arm-joints/1`:
  - Key: `tatbot/session/ros/arm/<arm>/joints`. `stencild` subscribes to
    `tatbot/session/*/arm/*/joints`, so any session segment works.
  - Envelope: `tatbot_bus::Envelope`, with `producer {node, pid, sha, run_id: "ros"}`
    and `stamp {mono_ns, wall_ns, basis: "host"}`. `wall_ns` is the
    `/joint_states` header stamp.
  - Payload: `{arm, measured_wall_ns, joints: [6 rad in joint_0..5 order],
    carriage: {position_m, effort_n}, mode, calibration_id}`.
    - The carriage effort is the driver's external effort (N).
    - `mode` is `Position`, `Idle` when landed, or `Fault` on a controller
      error, from `/tatbot/safety`.
    - `calibration_id` is the adopted `registration.<arm>` file's, or null.
      `stencild` refuses a view whose id differs from its camera bundle.
  - Rate: at most 10 Hz, and only when the measurement advanced. `stencild` keeps 64 samples per arm and pairs each wrist
    exposure with the nearest sample within `--wrist-capture-ms` (1 s by
    default). A congested put is dropped, never queued.
  - Only with `page_source: stencil`, where the bridge holds a bus session.
    A test (`test/test_arm_joints.py`) pins the envelope, field names, joint
    order and units against `stencild`'s decode.
- While the bus router is unreachable (the rig link not up at boot, a router
  restart), the bridge logs it once and retries every 2 s.
- Parameters (strings; `''` means the `stack.yaml` value): `config`,
  `page_source`, `arms`, `pattern_id`, `bus_endpoint`.

`ros2 run tatbot_bridge page_watch [--seconds 30] [--arm right] [--json]` is the
same subscription without ROS. It prints each sample's page pose in `world`
and in `<arm>/base_link` (`base_xyz_mm`, `base_rpy_deg`, `base_z_axis`), and
ends with a summary that says whether the sample and the registration share a
camera bundle. On a flat page `base_z_axis` is close to `[0, 0, 1]`.

`tatbot_rerun` logs to the fleet viewer through `scripts/lib/tatbot_rerun.py`
`start()`, loaded by path, capped at `rerun.max_hz`. Everything is in `world`
under `draw/ros/`:
- `<arm>/tip/<n>`: the measured tip path from tf `world → <arm>/tcp`, in
  200-point chunks, cleared at each run start;
- `<arm>/tcp`: the current tip;
- `page`: the page outline and clear center (grey while lost);
- `strokes`: the program of a run, read from `~/tatbot-logs/ros-draw/<run_id>/program.json`
  at its `run_start` event;
- `events` and `<arm>/safety`: text on each event and each safety change.

Its parameters are `config`, `arms`, `max_hz`, `connect` and `output`; with
neither `connect` nor `output` set, it uses the `rerun-server` node's proxy. It
needs `rerun-sdk` 0.36 in the node's Python and exits with a message
without it; the rest of the stack runs on.

### `stack.yaml`

`tatbot_bringup/config/stack.yaml` has format `tatbot-ros-stack`, version 1. Its keys:
- `arms`, `hardware`, `fake.page_z`, `profile`, `rt.{priority, cpus}`;
- `estop.{source, serial_device, serial_timeout_s 0.10, udp_port 7640, udp_timeout_s 0.15, relay_addr, debounce_frames}`;
- `safety.*`, which the driver takes;
- `page.{source, pattern_id, size_m, clear_m, replan_*, fixed.<arm>, trim.<arm>}`;
  the launch replaces `size_m` and `clear_m` with the installed print's and adds
  `inner_edges_m`, `{left, right, bottom, top}`, its inner border edge per side
  in the page frame (m: x of left and right, y of bottom and top), converted from
  its `settings.json` `border_inner_mm` (absent, ±`clear_m`/2), and `geometry`,
  the print or `stack.yaml`;
- `registration.<arm>`;
- `machine.{switch none|pi|sim, arm, addr, gpio, port 7642, estop_port 7643, period_s 0.02, timeout_s 0.2, confirm_s 1.0}`, the tattoo machine's switch (§4.4, §7);
- `keypad.{device, arm}`, the operator's numpad (§4.4); `device: ""` runs no keypad node;
- `touch.{enabled, points}`;
- `session.{resume_timeout_s 300, knot_rate_hz}`;
- `rerun.{enabled, max_hz}`;
- `record` (stack-wide MCAP, launch `record`).

The TCP is not in it (§3). The page trim is `[x, y]` metres in the page
frame, applied as `base_from_page · T(trim)`: the program's centre lands at
`+trim`, so a centre cross drawn with trim `t` whose centre sits at `d` from the
printed centre is corrected by `t − d`. The launch records the file it read as
`config_path` in the effective `stack.yaml`, and `tatbot_session` re-reads
`page.trim` from that file at every goal, so a trim committed there and
deployed applies without a restart.

### Python APIs

- `tatbot_description.robot_description(repo=None, *, arms=("right",), hardware=None, registrations=None, ros2_control=None) -> str`
  returns URDF text. `hardware=None` gives kinematics only and needs no ROS or xacro.
  `ros2_control` is the effective `stack.yaml` dict (`None` = the checkout's);
  `registrations` maps an arm to a registration file or a 4×4 matrix.
  Helpers: `load_stack(repo=None, path=None)`, `hardware_params(repo, arm, stack, hardware)`,
  `ros2_control_xml(repo, arm, hardware, stack)`. `python3 -m tatbot_description
  [--arms right] [--hardware mock|fake|real] [--registration ARM=PATH] [-o FILE]` prints it
  (for `check_urdf`).
- `tatbot_ink.compile(design_path, *, arm="right", tool_id=None, repo=None, speed_m_s=0.0035, inks_path=None, max_segment_s=60.0, at_m=None, width_m=None) -> dict`
  returns §4.2's program and raises `tatbot_ink.CompileError` with the reason. Also
  `write_program(program, path)`, `write_preview(program, path)`, and
  `python -m tatbot_ink compile DESIGN [--arm] [--ee-tool] [--speed MM_S] [--inks] [--max-segment-s S] [--repo] [-o DIR]`
  → `DIR/program.json`, `DIR/preview.svg`, the stats as one JSON line on stdout; exit 1 on a refusal.
- `tatbot_motion.load_motion(path=None) -> dict` reads `motion.yaml`, format `tatbot-motion`, version 1.
- `tatbot_motion.Kinematics(urdf_xml, arm="right")` (or `.from_repo(repo=None, arm)`) provides
  `.joint_names`, `.lower`, `.upper`, `.fk(q) -> base_from_tcp 4×4`, `.jacobian(q) -> 6×7`
  (LOCAL_WORLD_ALIGNED, expressed in `<arm>/base_link`) and `.solve(base_from_tcp, q_seed)` (Newton IK
  for start poses and reach checks, not paths).
- `tatbot_motion.Trajectory(joint_names, t, q, qd, tip, arc_m, phase, info, axis=None, dq_dh=None)`:
  `t[0] = 0` is the seed, `qd[-1] = 0`, `tip` is FK of `q`, `phase` the Draw `PHASE_*`; `info` carries
  duration, model error, peak joint speed and any slow-downs, and a sent goal's `trim`. A stroke plan's
  `axis` (M, 3) is each knot's depth axis and `dq_dh` (M, 7) its joints per metre along it.
- `tatbot_motion.Trim(start_m, ramps)` is a pen trim over a goal's time (`value`, `rate`, `to(t0, target_m,
  cfg)` for a quintic ramp); `tatbot_motion.compose(traj, trim)` adds it along the plan's depth axis (§4.4).
- The planners, each returning a `Trajectory` at `motion.yaml knot_rate_hz` (100 Hz); each raises
  `tatbot_motion.PlanError` when a joint limit (less `clik.joint_limit_margin_rad`) is on the way or the
  tip trails its reference by more than `clik.max_model_error_m` (the tool axis by more than
  `clik.max_orientation_error_rad`). `q_seed` must lie inside the limits less that margin; the
  staged/sleep pose (joint_1 = joint_2 = 0, on the lower limit) does not, so the session's ready joint
  move comes before any plan:
  - `plan_op(op, *, base_from_page, q_seed, kin, motion, speed_m_s, from_arc_m=0.0, pen_down_at_start=False)`;
  - `plan_travel(*, q_seed, base_from_tcp, kin, motion)`;
  - `plan_lift(*, q_seed, direction, distance_m, kin, motion)`;
  - `plan_touch(*, q_seed, start_base_from_tcp, direction, max_travel_m, kin, motion, prior_distance_m=None, speed_m_s=None)`;
    arm the guard (`guard_mode = 1`) at `guard_arm_time_s(traj)`, the first `PHASE_TOUCH` knot;
  - `to_knots(traj, rate_hz)`.
- `tatbot_motion.dispatch_drift(traj, q_measured, kin, motion, pen_down=None) -> {ok, joint_rad, tip_m, pen_down}`
  compares the plan's first knot with the measured joints (the plan's carriage stands in for the reading);
  `ok` false means re-plan from `q_measured`.

How `tatbot_motion` plans (the choreography of §4.4, ported from `scripts/lib/pen_path.py`):
- Cartesian samples at `control_rate_hz` (400 Hz), a feed-forward velocity from their box-smoothed
  central differences, then the damped-least-squares CLIK of `cpp/teleop/square_probe.cpp` (λ 0.02,
  gains 4 and 2 s⁻¹, the next sample's rotation step as angular feed-forward) on pinocchio.
  `test_clik_parity` feeds identical samples through `path_plan_check` and asserts FK tip and joint
  agreement within 0.1 mm (they agree to about 10⁻¹² mm). It SKIPs, with the reason, until
  `path_plan_check` is built, and `deploy` does not carry `cpp/`; to run it on the ros node, build it
  from a checkout (`cmake -B cpp/teleop/build -S cpp/teleop -DCMAKE_BUILD_TYPE=Release && cmake --build
  cpp/teleop/build --target path_plan_check`), then `TATBOT_PATH_PLAN_CHECK=<that>/path_plan_check colcon
  test --packages-select tatbot_motion ...` and confirm `test_clik_parity` reports no skips.
- Pen orientation: tcp +z along −page z, spun as little as possible from the seed's tcp rotation
  (pen_path's `align(n_c → n) · R_c` on a flat page). A pen-up leg that turns the tool does so over
  its first chord under the 8°/s `approach.omega_max_rad_s` cap and pen_path's rotary budget
  (`approach.angular_accel_rad_s2`, 0.1 rad/s²).
- The carriage stays out of the drawing solve (`carriage.ik: false`); pen-up travel
  brings it back to `carriage.bias_m` (2 mm) at ≤ `carriage.travel_m_s`, so a trip retract is undone
  by the next travel. `carriage.ik: true` is the paper A/B's weighted seven-joint solve, bounded to its
  0.5–3.5 mm window.
- A tip below the standoff lifts straight off the page before any travel. With `pen_down_at_start` a
  tip within `dispatch_drift.start_tolerance_m` of the start (`resume_reapproach_m` when resuming from
  `from_arc_m`) joins the path pen-down; farther, it lifts and re-approaches.
- A plan over a joint-speed cap (0.375 rad/s pen down, 1.0 pen up) is slowed, leg by leg, and planned
  again; it never ships over the cap. `pause` lifts to the standoff or holds; `dip` is M4.
- Touch: settle 0.5 s at the start, 10 mm/s to 4 mm short of the prior (`PHASE_DESCEND`), then 0.2 mm/s
  (`touch.slow_m_s`, `PHASE_TOUCH`) up to `max_travel_m`, with no wiggle (`touch.wiggle_amplitude_m` is 0: a
  wiggling tip dragged the pink arm's compliant mat); without a prior the whole 25 mm search is slow.
- `tatbot_bridge.page.world_from_page(world_from_target)`.

### Runs

`tatbot_session` creates one run per `Draw` goal (`ros-draw`) or `Touch` goal
(`ros-touch`) through `$TATBOT_REPO/scripts/lib/tatbot_runlog.py`
(`init(workflow, attach_logging=False)`, `finalize`). It lives at
`~/tatbot-logs/ros-draw/<run-id>/` and holds:
- `meta.json`, with the deployed `revision` (`repo/REVISION`: sha and dirty
  flag), the program sha, tool, TCP, trim, page pattern id, registration sha,
  e-stop source and hardware (the `ros-stack` run records the same revision);
- `console.log`, `run.jsonl`;
- `program.json`;
- `ledger.jsonl` (the format is in `tatbot_session/ledger.py`: `sent`, `done`,
  `aborted` with `arc_m`, `skipped`, `decision`);
- `page.json`, with the page poses used and the touches;
- `bag/`, an MCAP (`ros2 bag record`, started by the session and stopped with
  SIGINT) of `/joint_states /tf /tf_static /tatbot/events /tatbot/safety /tatbot/page
  /tatbot/goals` and, per arm, `/<arm>_arm_controller/controller_state`, the
  `follow_joint_trajectory` feedback and status, and `/<arm>_safety_controller/gpio_states`.
  A resumed run (`Draw.run_id`) appends to its ledger and records `bag-2/`, `bag-3/` and so on;
- `drawn.svg`: the FK of the measured joints over each pen-down stretch at
  20 Hz, in page millimetres, over the planned strokes.

`console.log` holds every event line. `tatbot_session` reads one parameter,
`stack`: the effective `stack.yaml` that `stack.launch.py` writes into the
`ros-stack` run. The repo is `$TATBOT_REPO`.

### CLI

| Verb | Tier | Does |
|---|---|---|
| `tatbot ros deploy [--no-build] [--units] [--clean]` | remote | rsync + incremental colcon on the `ros` node; `--units` installs the systemd units |
| `tatbot ros up [--hardware] [--estop] [--relay-addr] [--page] [--pattern ID] [--arms] [--no-touch] [--rerun] [--hold]` | remote | `--pattern` names the print to draw on by its pattern id or the hex digits its sheet prints, resolved against the prints installed on the ros node (refused when none or several match); lands a running stack's arms (as `down`), writes `stack.env`, restarts the router and the stack, prints the effective hardware, e-stop, page and arms (flags over `stack.yaml`; flags not given fall back to `stack.yaml`, not to the last `up`), and a note when a real arm runs with `estop none` |
| `tatbot ros down [--router] [--hold]` | remote | prints each arm's landed/latched state, lands every arm that has not landed and reports no controller error (`client land`; hardware fake or real), then stops the stack; a failed landing leaves it up; `--hold` stops without landing |
| `tatbot ros status` | sensor | revision, what the stack runs with (`TATBOT_ROS_EFFECTIVE`), units, `stack.env`, newest run, 2 s of the page on the bus (`page_watch`), `SafetyState` per arm (`client status`) |
| `tatbot ros station [--arm] [--against]` | sensor | the station in an arm's frame, measured now from the palette's palette tag, never a stored pose |
| `tatbot [--ee-tool ID] ros calib run\|sweep\|apply [--run] [--arm] [--attitudes] [--heading-deg] [--visible-sides]` | motion-auto | calibrate the arm's fitted tool (a stated tool must be it) on the station probe: fresh station fix, touches, fit, closing station check, adopt; `sweep` measures it contact-free from joint-6 turns under the palette camera (§7); a tool with no probe contact model is refused before anything moves |
| `tatbot ros compile DESIGN [--arm] [--speed] [--inks] [--at] [--width] [--stencil DIR] [-o]` | offline | `tatbot_ink`, local; `--stencil` places it on that generated print's page |
| `tatbot ros ready [--arm]` | motion-auto | performs the ordinary draw wake-up before a research caller freezes the runtime; a landed arm restarts and waits for readiness |
| `tatbot ros draw PROGRAM [--arm] [--from-op] [--resume RUN] [--no-inspect] [--hold] [--no-wake]` | motion-auto | restarts the stack first when the arm has landed, unless `--no-wake`; copies the program to `inbox/` and runs `client draw` there; a complete draw is inspected, then landed (controller idle) unless `--hold` |
| `tatbot ros touch [--arm]` | motion-auto | three page touches |
| `tatbot ros jog --joint N --delta D [--arm] [--speed]` | motion-auto | `client jog`: a quintic move of one joint (0–5 rad, 6 the carriage in m) from its measured position through the JTC, 0.02 rad/s (0.002 m/s) peak by default; prints the tip displacement and the latch state; exits 1 while a draw or touch goal runs |
| `tatbot ros inspect [--arm] [--run] [--frames] [--poses]` | motion-auto | hover the wrist camera over the newest drawing and write page-frame views with the plan overlaid |
| `tatbot ros evidence --page PAGE --slot SLOT` | sensor | read a research slot claim, original run ledger, runtime and inspection evidence without motion |
| `tatbot ros decide ARM continue\|land\|skip\|redraw` | remote | `client decide` |
| `tatbot ros cancel` | remote | cancel draw/touch goals with the arm holding |
| `tatbot ros palette status\|load [SLOT=INK ...] [--level-mm SLOT=MM]` | remote | read or declare what each palette cap holds and its measured ink level (§6) |
| `tatbot ros logs [last\|list\|show\|tail] [RUN] [--workflow]` | offline | `tatbot-logs` on the `ros` node (`ros-draw`, `ros-touch`, `ros-stack`); `ros-cli` runs are read on this node, where they were written |
| `tatbot ros relay-install [--dest HOST:PORT ...] [--bind ADDR]` | remote | copies the relay and the udev rule over ssh, installs the rule and `tatbot-estop-relay.service` (source: `stack.yaml` `estop.relay_gpio`) on the `estop-relay` node; its readers default to the ros node and, with a relay profile e-stop, every `estop` node |

The global `--dry-run` prints each plan, with the backend's own dry run as
`plan` notes: every ssh, rsync, colcon and systemctl command. Exit codes: 0 ok,
2 usage (a missing program, `--estop udp` without a relay address), 4 no
single `ros` node in `config/nodes.json`, 5 the node is unreachable, otherwise
the remote command's own code. A `draw` given an `inkmap-design.json` compiles
it locally first. On the `ros` node,
`ros2 run tatbot_session client {draw,touch,jog,decide,land,status}` is the same
client.

The [DBV3 research driver](../docs/research.md) prepares immutable paired
programs and calls this stack. Research programs claim their physical page
slot before execution; `tatbot ros evidence --page ID --slot ROW:COL` reads
that claim and its existing run ledger without moving the arm.
