---
summary: Public vision pipeline and timestamp contract
tags: [vision, cameras, replay]
updated: 2026-09-16
audience: [dev, contributor]
---

# Vision

## Retained fiducial depth witness

`tatbot vision board witness` compares original RGB-D captures against an
independently measured board tag size. Provide an existing native-vision Python
interpreter, the capture manifests, the fiducial inventory, the color sensor
name, and a measurement JSON containing `target: "board"`, `edge_m`, `source`,
and optional `uncertainty_m` (metres; absent means unavailable).

```
scripts/tatbot vision board witness --python /path/to/validated/python \
  --capture /path/to/first/capture.json /path/to/last/capture.json \
  --inventory /path/to/fiducials.json --measurement /path/to/measurement.json \
  --color-sensor wrist_color --out /path/to/new-witness
```

The shared original-payload reader and surface pipeline verify the sensor pair
and native alignment. The witness solves each detected board square using the
camera's exact deprojection rays and retains every positive-depth planar pose
branch. Each branch compares an 81-point interior grid of original depth with
the size-derived RGB plane, preserving missing/saturated samples. Tags retain
their family and ID; their spacing and the whole sheet's rigidity are not assumed.
The report binds measurement, inventory, captures and payloads by SHA256.
It reports observed discrepancies without adopting calibration. Measurement
uncertainty, lens accuracy, board flatness, wrist mounting and joint timing
still contribute to physical error qualification.

For retained wrist captures, optional `--pose-inputs /path/to/snapshot` binds
the same configuration used by wrist surface fusion: `bundle.json`,
`registration.json`, `golden.json`, `config/arms.json`,
`config/workspace.yaml`, `urdf/tatbot.urdf`, and `vision.toml`. The command
verifies the capture's wrist identity, device, profile and bundle, then uses
the shared registered wrist FK to compare each common tag across views.
It retains every planar branch combination and hashes the configuration.
These discrepancies assume each individual tag stayed fixed; they do not
solve or adopt a camera mount, establish an absolute error bound, or assume
the whole board is a rigid plane.

Optional `--mount-fit-indices 0 2 4` fits unadopted hand-eye candidates from
3–8 explicitly selected captures, requiring other captures for validation.
Each common tag is solved separately through the shared flange FK using
OpenCV's Park method. Every combination of positive-depth planar branches is
retained, including unavailable/non-rigid solutions. The report checks every
observed branch against each candidate and reports differences from the nominal
mount. Excluding a capture does not make repeated captures of one held pose
independent validation views. Candidates remain unselected and unadopted;
small motion, tag geometry, registration and optics can still confound the fit.

`rust/visiond/` ingests camera streams, records synchronized frames, and
supports replay with the shared URDF. It is designed so downstream code can see
which profile and timestamp domain produced a frame.

## Frame contract

Each capture record should identify the device, requested and active profile,
timestamp domain, sequence number, calibration revision, and dropped-frame or
transport warnings. A consumer must not silently substitute a profile or
calibration.

### Robot, camera and simulator coordinates

The legacy robot-world JSON field `world_from_base` means
`world_from_URDF_root`: the solver's FK already includes the fixed arm mounts.
Use `scripts/lib/robot_world.py:root_from_world` to invert it, and require the
registration's `calibration_id` to match the camera data being transformed.
The calibrated camera pose in the robot scene is
`root_from_world @ world_from_camera`. A consumer expressed in the follower
arm base additionally applies `base_from_root`, derived from the URDF.

`root` and `rig_center` are the arm-pair midpoint. With +X forward through the
fixed cameras and +Z up, the robot's left base is at Y=+0.2675 m and its right
base is at Y=-0.2675 m. Fixed frame and nominal camera mounts are relative to
`rig_center`. Never move the frame to compensate for a registration that pairs
one wrist's markers with the other arm's encoders. During teleoperation the
joint vectors can be nearly identical, so a low reprojection error alone cannot
establish the physical arm association. Such registrations and their derived
surfaces need replacement, even if one arm's FK is numerically unchanged.

Fixed camera bodies in Rerun follow those measured optical poses. The URDF
supplies the body-to-optical geometry, not their measured rig mounting poses.
The calibration producer owns `world/calibration/camera_models`; arm producers
omit these fixed bodies, avoiding competing static transforms. Bodies without
a matching calibration are omitted. Articulated wrist camera geometry remains
on its arm's joint chain.

The cockpit uses `--robot-world` (or the current local registration) to
place world-frame tracking axes. An absent or mismatched registration makes
the tracking geometry unavailable. Raw camera previews remain in sensor axes.
Python FK refuses unknown links instead of substituting the root origin, and
resolves mimic joints from their driven joint, multiplier and offset. The
left laser attachment rides the opposite carriage and depends on this mapping.

## Developing the pipeline

```bash
cd rust/visiond
cargo test
cargo build --release
```

Use fixture streams or recorded data for tests. Keep live addresses, camera
identifiers, credentials, and calibration snapshots in private acceptance
records rather than in this page.

## Calibration

The five-camera overhead calibration pipeline (`tatbot calib` capture phases,
retained candidates, preview and apply) was removed in 2026-09 with the PoE
scene cameras it measured. Tool-tip calibration on the demo rig is the ROS 2
stack's station probe (`tatbot ros calib run`); `tatbot vision touchoff` still
solves a tip from a teleop flight log.

On the demo stack the D555 is the only fixed camera, so its colour optical
frame is the world. `tatbot ros register --arm <arm>` measures an arm in it.
The ROS stack drives the arm through 27 holds over its side of the table: the
tool points down, turned and tilted so that the wrist tag triplet shows the
camera different faces. The D555 owner captures each hold the arm reached:
its measured tool within 5 mm and 2° of the hold. Every wrist-tag corner
seen, placed in the arm base by forward kinematics and the measured tag
layout, is one correspondence of a single camera-to-base transform. The
corners are fitted to each tag's edges, since the tags span about 23 px:
OpenCV's cornerSubPix leaves most of their corners on whole pixels, and its
AprilTag refinement places them 0.44 px off in x and y. The transform is
solved by PnP, and sightings far off the fit are dropped. The
camera bundle holds the D555's colour entry and the aligned depth entry at
identity, with the optics the camera reports. A later run reuses the bundle
while the camera's profile, lens model, coefficients and serial are
unchanged, since thermal focal drift is not a new camera.

`--adopt` installs a result only when it passes its gates: tags seen from at
least 12 distinct poses (holds within 3 cm and 15° of each other count once),
every tag seen from 3 of them, a corner residual median of at most 2.5 px
(p95 6 px), and no single view moving the arm more than 5 mm or 0.5° when
left out. Holds that repeat one pose fit that view closely and leave the arm
anywhere else unmeasured. The owner then restarts so its frames carry the
bundle. A new bundle is a new world, so the other arm's registration from the
old world is moved aside until that arm is registered again.

A rigid transform cannot take up an error in the chain it sees the tags
through. The rest is modelled in `report.json`'s `chain`, two terms each
scored on holds it was not given:

- the tag set's seat on its parent frame (the target's `parent_frame`);
- offsets on joints 1–4, with the tags at FK(q + dq). Joint 0 is the camera's
  yaw and joint 5 the seat's roll, so neither is fitted.

`tatbot ros chain <run-id>...` pools several registrations of one arm and
scores each model on the runs it was not fitted to. Neither gates nor adopts
anything. Pink's layout, solved on 2026-09-19 by the retired PoE cameras,
sits 6.4, 6.8 and 2.3 mm (x, y, z of `right/gripper_left`) and 1.8° off its
seat on the four registrations of 2026-09-29. The seat takes a left-out
run's corner median from 1.47 px to 1.01, and to 0.77 px with joint offsets
of +23, +21, −24 and +1 mrad. The seat alone is written as
`wrist-layout.json`, the wrist solve that
`scripts/vision/export_wrist_tags.py <file> --write --target <target>`
adopts. Adopting it moves every registration solved through the old
layout, so the arm registers again.

With `--prior <run-id>`, the holds are planned from that run's camera. Each
candidate is solved from the pen-down joints over it. It must hold upright
as an inspection view does, and stand clear of the arm's own bodies. Of
these, the holds are chosen greedily for the information they give the
camera, the seat and joints 1–4 (D-optimal). The run prints the 1-sigma the
plan predicts. For pink, 40 holds at heights 0.10–0.22 m predict 1.5–2.2 mrad
at 0.8 px. The holds of 2026-09-29 predicted 2.3–3.9.

## Stock wrist camera geometry

Each arm carries one D405 on the original Trossen bracket. The canonical
`urdf/tatbot.urdf` contains a fixed chain from each arm's `link_6` to its
`realsense_color_optical_frame` and `realsense_depth_optical_frame`. Its joint
origins match the [Trossen WidowX AI model](https://github.com/TrossenRobotics/ManiSkill-WidowX_AI/blob/main/wxai_follower.urdf).
The bracket pose is nominal CAD geometry, not a measured correction.

The camera registry binds `wrist_left` to the left arm and `wrist_upper` to the
right arm. Moving an arm changes camera pose through that arm's measured
joint angles; it does not change the fixed mounting transform. Use color
optical coordinates for depth explicitly aligned to color. Registering either
view in the scene still needs exposure-time joint observations and the
robot-to-scene transform.

The former lower camera on the right wrist is absent from the current model.
Never register a retained `wrist_lower` image using another camera's transform;
replay it with its original geometry. The separate
single-arm simulator still contains its historical two-camera benchmark layout;
that observation contract does not qualify the present physical arrangement.

## Live viewer

Every live workflow (`tatbot live cockpit`, which subscribes to what the camera
owners publish on the bus, and the ROS 2 stack's `tatbot ros up --rerun`) streams into one fleet Rerun viewer (a
headless server on the `rerun-server` node; operator nodes attach windows) rather
than opening a viewer of its own:

```bash
tatbot viewer start            # on the server node: gRPC server on :9876 (+ web :9090)
tatbot viewer open             # attach a capped window on this node
tatbot viewer status           # server, window, version, proxy URL
tatbot viewer view file.rrd    # stream a recording (.rrd, .wxtl, or a teleop run dir) into the fleet viewer
```

`live cockpit --duration 30` runs a bounded check. It opens no camera: it
subscribes to the frames the owners already publish on the bus, so it runs
beside a LeRobot session or the ROS 2 stack. The operator needs the fleet
viewer and the bus router, but no camera-LAN address.

`viewer view` streams from the node that holds the file: a path that does not
exist where you typed it is looked for on the arm node, where flight logs
live. An unavailable fleet viewer returns exit 5 with a recovery hint. It never
opens a local window implicitly; `viewer open` is the explicit extra-window command.

Every source carries a name in the viewer's list: workflow recordings are
named from their id (`Cockpit 20:15`, `Sweep 15:02`, `Replay <file> HH:MM`,
UTC like the time panel), and the standing recording the blueprint is sent
into `Rig preview`. A producer that joins a recording sends the same name,
derived from the id, so it never flips between producers.

Producers share one application id, join a session by recording id, stamp the
`capture_time` wall-clock timeline, and log under fixed entity prefixes
(`cameras/`, `robot/`, `world/`, `teleop/`, `surface/`, `draw/`,
`session/`). One blueprint definition in `rust/visiond` always presents the
fixed **Session / Telemetry / Calibration** top-level tab bar, in that order,
with no nested tab stacks. The rerun-server installs it once per server
generation. Workflows publish data only, and neither workflow startup nor
recording selection resets the operator's selected tab.
Every viewer and producer is memory- and rate-capped;
`scripts/check rerun-caps` refuses a launcher or producer that is not. Python
producers use `scripts/lib/tatbot_rerun.py`.

`tatbot live cockpit --display-quality normal` caps the subscriber at 2 Hz;
`--display-quality showcase` raises that cap to 5 Hz, and `--fps` overrides
either. Showcase does not turn reduced upstream bus derivatives into
full-resolution sensor data. These controls affect display copies only, never
capture, tracking, policy, journal, or sensor rates.

## Evidence limits

Timestamp alignment is not geometric calibration, and a synchronized recording
is not proof of safe robot behavior. State those limits in every published
result.

### Lossless tracker diagnostics

For detector comparisons, the native `capture-poe-all --lossless-evidence`
option retains exact decoded subscriber pixels and their checksums. It requires
an existing camera-owner socket, decoded recording, and a duration of 1–30
seconds. Normal subscriber recordings retain their JPEG encoding. Use a bounded
RAM-backed destination and archive it after capture; five full-resolution luma
streams can require roughly 500 MB per second. This is diagnostic evidence,
not a production latency measurement while the extra subscriber is attached.

PoE subscriber recording drains the owner independently of encoding and disk
writes, retaining only the latest complete pending set. A slow recorder drops
superseded sets rather than blocking the camera-owner socket; the diagnostic
counter `subscriber_superseded_sets` reports those drops. Frame timestamps and
checksums remain authoritative; recording throughput is not camera throughput.

The tracker service allows up to eight malloc arenas for its parallel camera
detectors. A single arena serialized the allocation-heavy planar initializer;
this setting changes allocation contention, not detection or pose-solver math.
Camera-owner allocation settings are independent and unchanged.

### Experimental surface replay

`tatbot vision surface replay --synthetic --max-frames 300 --rerun` evaluates
sparse OpenCV tracks against known image motion, brightness changes, blank
texture, occlusion, timestamp gaps and ambiguous repeating texture. `--width 1920 --height 1080` exercises
larger frames. Reports and optional RRDs go into the `surface-replay` run log.

For retained camera pixels, use `--recording /path/to/camera/frames.jsonl
--roi X Y W H`. The reader verifies native payload checksums and reuses the
visiond pixel decoder, including JPEG evidence. It requires normalized capture
timestamps and one sensor per index. It never opens a camera or robot connection.
Tracks keep their initial identities; default loss stays latched until a new replay.

Reports separate active tracking from initialization/loss and input decoding.
Optical flow operates on a moving crop around surviving points.
CPU cost is measured offline with one OpenCV thread; synthetic image generation
and optional Rerun logging contribute to total process cost. Neither throughput
nor image-space error establishes live latency, 3D accuracy, material identity
through occlusion, or performance beside the camera services. Repeated texture,
background features and gradual drift can still produce incorrect tracks.

Add `--recover` to evaluate bounded blur handling and reference-only
reacquisition. This experimental mode withholds coordinates when image sharpness
falls or too few reference tracks survive. It retains the last accepted image
and point IDs, retries at most 10 times per second, and requires two successful
reference matches before reporting `reacquired`. It never detects replacement
features. A reference older than 750 ms ends recovery; the original 150 ms
input-gap check still applies. Large turns, repeating textures, prolonged
occlusion and nonrigid changes remain unqualified. Recovery does not establish
physical material identity or grant motion authority.

The synthetic suite includes short blur and occlusion cases. Reports include
wall and CPU processing percentiles separately for each state, so cheap paused
or lost frames cannot hide the cost of recovery attempts.

`--match orb` or `--match sift` adds a periodic advisory keyframe search. ORB is the cheaper comparison; SIFT provides a second rotation
matcher. Both reuse OpenCV, retain the initial surface patch, and search without assuming cylinder geometry.
Each attempt searches one region: a square around the last candidate (initially
the selected ROI), then a larger square after a miss, then the full image.
After three misses the cycle repeats; a candidate recenters the next local
search. This center is an advisory search hint, not a verified surface position. Matches must pass
bidirectional descriptor-ratio checks, robust local image alignment, spatial
coverage and projection checks. Output is a candidate polygon only: it never
reactivates tracking, reseeds point IDs, or certifies physical material identity.
A homography checks local image consistency; it is not a curved-surface model.

Search images are capped at 1280 pixels on the long edge and extraction requests
at most 1200 scene features, with separate quotas in a 4-by-3 image grid
to limit background competition. Reference patches retain native detail up to the same edge cap. Local search
squares span 1.5 times the candidate's longest bounding-box edge; expanded
squares span 3 times that edge. Local matches cannot rule out identical
texture outside the searched region.
Attempts are at least 500 ms apart and are spaced further using
measured CPU cost to target 10% of one core over source time. This is an offline
workload target, not a hard live CPU quota. Matching remains disabled by default.
The rotation suite measures candidate-corner error separately from tracked-point
error; Rerun records matching attempts at their source timestamps.

`--match-keyframes` (requires `--match` and `--recover`) experiments with at most
three appearance references, retaining the initial reference permanently. It
reuses the existing flow observations; it opens no cameras and computes no
additional optical-flow pyramids. Admission requires clear tracked points,
80% of original IDs (at least 24), broad geometric support, and agreement within
2 pixels between flow and a direct descriptor match to the initial reference.
Descriptors near the projected patch boundary are excluded. Each admitted
reference maps back to the original patch coordinates; candidates cannot admit
other candidates or resume robot tracking.

Admission and search share one measured CPU allowance. Searches rotate through
references from newest to oldest, one per attempt, while each reference expands its own search window.
Cheap unchanged-view checks may run every 50 ms; replay images remain capped at
5 Hz. Reports include the first tracking interruption, peak reference count, and
bank decision reasons. This is an opt-in image-space experiment: agreement on
repeating texture does not establish physical material identity, and nonrigid
surface motion remains unqualified.

The `scale_turn` synthetic case gradually changes scale and orientation before
a larger jump, with a single visible image copy. Candidate-corner errors and
flow-point errors are reported separately; the repeating-texture `alias` case
continues to document the material-identity limitation.

## Wrist camera ownership

Each D405 in the vision registry declares a physical `arm` and an `owner_role`.
The owner role resolves through the node registry and must have exactly one
owner. It describes the USB host, independently of which computer controls the
arm or which teleop role that arm takes. Device names and serials remain stable;
view roles distinguish mounting locations. The remaining right view is
`wrist_upper`; the moved left view is `wrist_left`. Its former `wrist_lower`
role is retired so it cannot silently reuse the old right-wrist extrinsic.

The D405 service resolves its node's cameras and passes explicit, repeatable
`--sensor` arguments to `capture-realsense-all --group d405`. Unknown names,
duplicate selections and names outside the selected group refuse before any
camera is opened. Node ownership and device selection use the same deployed
source snapshot. The service template gets its node identity from deployment,
so the same template runs on whichever node owns a wrist camera. Each owner keeps
its local `/tmp/tatbot-d405-frames.sock` and local LeRobot camera handoff;
starting a recording never stops another node's camera service.

A frame timeout — no frame, error or finish event from any camera worker for
15 s — is answered by the owner itself before it is answered by systemd. The
owner stops the pipeline of every device it runs, puts each through a
librealsense hardware reset, reopens it as after any transport failure, and
publishes one `tatbot.capture-health/1` record per device with `event: reset`
on its capture topic (`tatbot/vision/d405/capture/<owner-node>`;
`tatbot/vision/overhead-depth/capture` for the fixed camera) naming the last
error that worker saw. The reset is issued by the device's own worker, so it
reaches a device whose pipeline is up and delivering nothing (the SDK's frame
wait timing out); a worker that never returns from a librealsense call
(pipeline start or stop) issues none, and the owner's exit says so. A second
timeout inside one minute exits the process so systemd restarts the unit; a
reset that restored frames for a minute earns the next timeout its own reset.
A device that does not re-enumerate after its reset is an error storm (one
reopen attempt every 250 ms), not a timeout: the owner neither exits nor
resets again until the device returns. The reset re-enumerates that one USB
device and nothing else on its hub.

Live frames retain `physical_arm`, `capture_owner_role` and device serial
metadata. Camera stream topics remain `tatbot/vision/d405/<sensor>`; raw wrist
snapshot queries now require `tatbot/vision/d405/capture/<owner-node>`. There is
no unqualified wrist snapshot endpoint that might return the other arm's view.
The fixed overhead endpoint remains unchanged.

LeRobot recording selects only the receiving physical arm's local cameras and
supports a single view. Rollout compares the resulting RGB/depth keys with the
checkpoint before acquiring motion authority; a two-view checkpoint refuses
a one-view configuration. Single-camera reconstruction and the moved camera's new
extrinsic require separate implementation and calibration; this capture split
does not qualify either one. Clock alignment between hosts is also a separate
measurement, not implied by successful local capture.

Use `tatbot live cockpit` to view both capture owners.

## Fixed overhead depth camera

The fixed D555 is a separate `overhead-depth` RealSense capture group. Its
owner publishes aligned color and depth at 5 Hz on
`tatbot/vision/overhead-depth/*`, answers complete RGB-D snapshots on
`tatbot/vision/overhead-depth/capture`, and exposes its full-rate local stream
at `/tmp/tatbot-d555-frames.sock`. It is deliberately separate from the two
moving wrist D405s: opening, synchronizing, or restarting one group must not
take ownership of the other.

The cockpit includes overhead color/depth and 2 Hz wrist-depth previews at quarter
width and height. Nearest-neighbour resizing preserves the depth scale. Full
capture requests retain the original RGB-D frames. The viewer connection retains
the camera owner's JPEG bytes, avoiding raw RGB expansion without changing
image quality. Join an existing recording with
`tatbot live cockpit --recording-id <id>`. Raw camera-world wrist tracking is not overlaid on the
robot model without a measured registration between those frames.

The installed profile is 640x360 at 30 Hz, depth preset 0, laser power 150 and
a measured depth scale of 0.001 m per Z16 unit. In the installed view this
profile gave substantially better valid-pixel coverage and repeatability than
896x504 at 30 Hz or 1280x720 at 15 Hz. Preset 0 gave the best stable-pixel
coverage of presets 0-4. Laser 150 retained essentially the same coverage as
240 or 360 while using less projector power. These are measured starting
settings for this scene, not universal D555 recommendations or drawing
tolerances.

A 200-frame A/B/A check in the installed central 260x220 ROI compared the
D555 while the wrist-D405 owner was active, stopped, then active again. The
off and post-restore samples were effectively identical: 72.103% valid depth,
69.617% versus 69.638% pixels with at least 90% support, 2.965 mm temporal
robust-sigma p95, 4.000 mm per-pair delta p95, and 24.42 versus 24.51 delivered
frames/s. The pre-stop coverage was consistent at 72.046% valid and 69.587%
stable. No central-ROI interference attributable to active wrist capture was
measurable in this unattended scene. This does not qualify whole-surface
coverage, other arm/projector geometries, lighting changes or physical depth
accuracy; repeat the comparison during the fiducial-board placement study.

The Ethernet transport requires a gigabit link, DDS discovery and an MTU of
9000 from the camera owner through the PoE switch. The camera sends only
9000-byte frames, so the owner's NIC must accept them; the owner's other
routes can stay at 1500, since librealsense caps what the host sends at 1470
bytes. The manifest marks the camera `transport = "dds"`, and its owner
states the camera-LAN address discovery is bound to (`--dds-address`; the
launcher passes the owner's `lan` from `config/nodes.json`). A DDS camera's
context enables DDS for that address only and leaves USB enumeration out, so
it never touches a wrist camera; USB cameras keep the SDK's default context.
The distribution's librealsense package has no DDS. A node that owns a DDS
camera provisions the SDK built with DDS as static archives under
`~/.local/share/tatbot/toolchain/librealsense-dds` (with a `realsense2.pc`
naming every archive), and release builds there link it
(`fleet_toolchain::realsense_dds`). The SDK's file-format archive carries a
pre-0.6 xxhash whose `XXH32` symbols collide with the one the Rust `lz4-sys`
crate links, and whose state layout differs. The provisioned
`librealsense-file.a` therefore has those symbols renamed with an `rs2_`
prefix (`objcopy --redefine-syms`), and a `PROVENANCE.md` beside the archives
records it. The capture backend maps the
device hardware clock onto host Unix time, records the original device frame
number separately, suppresses repeated-timestamp DDS framesets, and assigns a
strictly increasing evidence sequence so a reconnect or repeated device frame
number cannot overwrite a retained payload.

The active bundle's two D555 entries come from guided board sweeps of the
removed five-camera calibration (see Calibration above). Those sweeps solved the
D555's native colour optical frame in the same multi-camera bundle, keeping its
native intrinsics and distortion fixed so the solve agreed with the pixel grid
the owner aligns depth to. The strict companion `overhead_depth_depth` entry
carries the same optical geometry, because the owner aligns Z16 into color
before publishing.

Once both D555 entries exist in the active bundle, its owner stamps that bundle
and `trackd` timestamp-merges the freshest overhead set with each PoE cadence
set (80 ms maximum separation) before one pose update. If the bundle lacks
either entry, the D555 stays available for camera-frame evidence and the
tracker does not subscribe to it. This makes calibration membership explicit:
an unstamped frame cannot become a world-frame tracking observation.

`stencild` is the same kind of subscriber for printed stencils: one observer
for the fleet on the camera node, reading the PoE socket where that node has
the PoE cameras and, under the same stamped bundle, the overhead socket. The
demo stack has no PoE cameras, so the D555 is the observer's only fixed view,
and at 0.43 px/mm it cannot resolve a print. A coded print's knots are about a
pixel there, so nothing is decoded or feature-matched. Where no fit measures a
print, the observer can place it by its artwork instead (`--overhead-artwork-match`,
which the launcher sets when the node has no PoE cameras). Inside each
registered arm's drawing pad, it fits the table plane to the D555's depth and
resamples the colour image onto that plane at 2 mm per pixel. It then correlates
the reference artwork, band-passed, over position and turn. The printed frame
shows as a grey ring on its sheet, and the band-pass keeps the sheet's own edge
on a dark mat from winning. Only the frame is correlated: the clear centre
(the reference's `clear_center_uv`) is where the arm draws, and a four-ink
drawing there held an uncovered page at 0.61-0.69 when the whole artwork was
correlated; on its frame alone it scored 0.79-0.82. A peak at 0.70 or more, at least 0.15 above any
other, is the page: with the drawing arm over the sheet, false peaks a quarter
turn round scored up to 0.60, and the uncovered page 0.77-0.87. The ring cannot tell a half turn apart, so the print's top
faces away from the drawing arm unless the page's last pose says otherwise.
Such a target is published `measured` with its identity unverified. It carries
the match's score and margin (`support.method` `artwork_plane_match`) and sigma
floors of 4 mm and 1.5°. The drawing stack measures the page again with its
wrist camera and touches before it draws. On the installed view the match
scored 0.84, with a margin of 0.54, in 0.4 s per turn. Its configured period is one second;
`processing_p95_ms` and `capture_to_publication_p95_ms` report the achieved
rate and age. Per-turn capture scratch uses the service's managed memory
directory; durable estimates stay in its run log. It keeps the overhead set
whose exposure the most PoE cameras share inside 40 ms. The first turn
surveys every fixed camera. Later turns keep image flow tracks current in all
fixed views and permit one rotating fixed view to run reference search every
three turns when several fixed cameras are present. A lone fixed camera can
search each turn. A lost track or changed image is searched when that view
gets its turn; this bounds the time spent searching when an arm hides a print from several
cameras at once. A still-empty view and an acquired print become due for a
full search after three minutes. Each turn projects overhead depth into the
current tracked views. A previous measured depth
footprint bounds the next projection; missing support retries the full depth
image and all current fixed views in the same turn. An anchor camera is chosen
per print (a camera named by `--exclude-anchor` can be compared but never
chosen). A valid previous anchor stays selected when another fit has at most
four more anchors and a comparable residual, limiting jumps from marginal
ranking changes; lost support switches immediately. Cross-camera pose
disagreement remains visible and needs independent calibration review. The
service publishes the print's pose as
`tatbot.target-pose/1` on `tatbot/tracking/target/<pattern_id>` with a
`support` block: the anchor camera, its measured anchors and leave-one-out
residual, both sigmas, the page corners and plane, every camera's
disagreement, the bundle id, and `motion_authority: false`.

A coded print (the default design, see [stencil frames](stencil-frames.md))
is decoded, never matched by SIFT; legacy artwork keeps SIFT. The decode
names the print and seeds the same flow track a SIFT acquisition seeds, and
every tracked frame re-reads the print's own bits at the tracked junctions,
so `physical_instance_id` is verified in the frame that carries the pose. A
rejected or ambiguous decode, or bits that do not verify (a slipped lattice,
another print put in its place), reads as `lost`, and the refusal is named
in `support.reason`. A decode of a cluttered real frame takes from seconds to
two minutes on the camera node, so it runs only inside search regions: each
registered arm's drawing pad (a square 250 mm either side of its
`config/workspace.yaml` pad pivot, through the arm's registration) projected
into the view, where a close wrist view keeps the part of the pad it sees, and
a box around a page the view already tracks. With no registered pad, a view of
at most 1.4 Mpx is searched whole and a larger one not at all. The decode runs
in one background process, one view at a time, and never holds up a turn: its
junctions are carried to the view's newest frame by optical flow and must pass
that frame's bit check, so a print that moved or went away meanwhile reads lost
and is searched again. A pad seen coarser than 1.2 px/mm is not searched. Each turn reports its pad search areas, and why an arm with a pad
has none, under `coded_search`. An installed reference the observer cannot
use (a coded manifest without its bound `coded.json`, or a coded print the
decoder cannot read) is published `lost` with the reason, listed under
`refused_references`, and never stops the other prints' tracking.

Each arm's wrist D405 is one more view in the turn. The observer reads the
D405 registry (`vision.toml`: every camera of group `d405` with a physical
`arm` and an `owner_role`), subscribes the run-agnostic
`tatbot/session/*/arm/*/joints` (`tatbot.arm-joints/1`, the arms' measured
joints, which the ROS 2 stack's `tatbot_bridge` publishes; the key keeps its
historical `session` segment) and, for an arm whose joints are on the bus,
queries its owner's `tatbot/vision/d405/capture/<node>` for the newest
RGB-D pair at most once per `--wrist-capture-ms` (one second). The view is
posed in the observer's world through the arm's own
    `tatbot.arm-registration/1` (the `arm-registration-<arm>-current.json`
beside the bundle under `--registrations`, accepted on the bundle or carried
onto it, and folded into the input fingerprint so a new registration
restarts the estimator on the new world) times the URDF chain at the joints sample
nearest the frame's exposure, with the frame's own active intrinsics rather
than a bundle entry. A joints sample farther than one capture interval from
the exposure, an arm with no joints on the bus, no registration, a launch
bound to another camera bundle, or a capture not of that arm refuses that
view by name, never the turn; a view with no pose never enters the fusion.
The tracked-wrist topic `tatbot/tracking/ee/<arm>` is never an input. A
posed view is searched every turn it arrives, its fit and disagreement listed
under `support.cameras` as `wrist_<arm>`, and every print's `support` names
the turn's `wrist_views` — each posed one with its capture stamp, joint stamp
and skew, the joints' bound bundle and the registration's carry, each refused
one with its reason. Name `wrist_<arm>` under `--exclude-anchor` to compare a
wrist without ever anchoring on it. References are
installed under the log root (`stencils/references/<name>/tracking.json`
beside its `stencil.png`, and a coded print's hash-bound `coded.json`) and
rescanned by mtime, so installing one never
restarts the service and none installed leaves it idle. A lost print is
published as `lost`. `--max-age-ms` (3 s) gates the input only: a turn whose
selected exposure is older than that, or in the future, is skipped before the
estimator runs and counted as a stale set. A measured result is published as
measured however long its turn took, stamped with its capture time, so a
consumer reads its age from the sample. A late pose is never turned into
`lost`: the ROS bridge holds the last measured pose through a `lost`, which
would be older still. The observer decides nothing about motion. Each turn
logs capture-write, estimator, publication, and observer-stage timings with
cumulative per-camera unique socket and selected frame counts, sequence gaps,
ring overwrites, unselected sets, and selected PoE-to-overhead skew. The
service metrics also expose the stale-set count and the last
capture-to-publication age.
The shared Rerun Session view includes each measured target's live page outline
under `/world/tracking/target/<pattern_id>` beside the calibrated robot. A
`lost` sample clears that outline. The Calibration view can also show the
camera-local labeled image from `tatbot vision stencil observe --connect`;
that diagnostic image does not establish a world pose.

With matching D555 camera and robot-world bundles, consumers deproject
aligned Z16 using the native color rays and the paired depth-to-color
transform. Stencil tracking uses this conversion,
including the distortion convention and the native-depth-Z to color-Z
calculation. The robot-world JSON key `world_from_base` is historical: it maps the **URDF root**
into calibration world, including the fixed arm mounts. Thus camera points use
`inverse(world_from_base) @ world_from_camera`, with no additional arm-base
translation. Fixed RGB observations and calibrated viewer frustums use this
same convention. The installed-scene input range is 0.3-1.5 m. A stale owner,
payload digest change, profile change, calibration mismatch, or missing RGB-D
member refuses the capture rather than silently falling back.

**Runtime optics and the bound model.** The camera does not report one set of
optics for the life of a process. librealsense's thermal compensation re-reads
the device calibration table as the ASIC temperature moves (the D4xx host
monitor polls every 2 s and acts on 2 °C steps), and the D555 pushes each such
update over DDS as a `calibration-changed` notification that replaces the
stream profile's intrinsics in place: same profile id, same extrinsics. The
SDK align block, by contrast, freezes the aligned profile's intrinsics when it
is created. The owner therefore reads the model before every alignment,
rebuilds the align block when the model changes, withholds an exposure whose
aligned profile disagrees with it, and publishes the model it aligned with in
both `intrinsics` and `alignment_calibration`. Consumers keep that
frame-internal match exact: rays come from the frame's own intrinsics and must
equal the alignment model's colour optics to the digit, which is what
guarantees depth is deprojected on the grid the owner aligned to. Recorded
evidence replays under the same rule and involves no bundle.

The bundle binds one snapshot of those optics, so a live model can differ from
the bound one by the thermal drift alone. Observed on the installed D555 at
640x360: two frames from one owner process 77 minutes apart on 2026-09-16
reported fx 323.392 / fy 322.901 and then fx 323.216 / fy 322.726 (0.17 px,
5.4e-4 of the focal length) with the principal point and distortion
coefficients unchanged; the cold camera two days earlier reported fx 321.988
before warming to 323.129. The scan and stencil-tracking paths report ray
drift against the bound bundle but use each frame's aligned intrinsics for
deprojection; thermal focal drift does not hold a session. Image dimensions,
the lens model and its coefficients, the stream profile, and the device serial
(when recorded in the bundle) must still match. A calibration sweep requires
one unchanged native model across its shots.

The D555 grants no motion authority until a measured camera-to-arm
registration is adopted (`tatbot ros register`, above). Timestamp alignment is
not that extrinsic. Even then a stencil page's overhead pose is a prior: the
drawing stack re-measures the page with the wrist camera and its touches
before it draws. The present mounting distance also
places part of the working surface near the lower edge of the camera's
preferred range, so placement and full-surface coverage must be rechecked
during that calibration.

## Offline RGB-D registration benchmark

`tatbot vision surface rgbd --synthetic --rerun` compares Open3D point-to-plane
and colored ICP on known 3D motion, missing depth, absent overlap, timestamp gaps,
and an intentionally ambiguous uniform cylinder. These sampled 3D controls are
not sensor simulations or evidence of physical accuracy.

For existing visiond evidence:

```bash
tatbot vision surface rgbd --recording /path/to/evidence --sensor wrist \
  --roi 100 80 240 200 --depth-range 0.1 0.4 --max-frames 300 --rerun
```

For an operator-confirmed stationary scene, replace `--rerun` with
`--stationary-depth` to measure depth coverage and repeatability without ICP.
The report separates missing depth from temporal jitter: robust per-pixel sigma
uses only pixels valid in at least 90% of evaluated frames. Flagged frames count
against coverage and break consecutive comparisons, as do gaps over 150 ms.
This measures repeatability, not absolute accuracy; object or camera motion
contaminates the result. Cropped depth storage is limited to 64 MiB; reduce the
ROI or frame count if that limit is exceeded.

Choose ROI and depth bounds for the actual working patch; these example bounds
are not operating limits. The input must contain `wrist_color/frames.jsonl` and
`wrist_depth/frames.jsonl`, aligned RGB/Z16 payloads, normalized timestamps,
active intrinsics and measured depth units for registration. Stationary-depth
mode needs the aligned stream identity, stable profiles and measured depth
units, but neither Open3D nor camera rays, so it also runs on the aarch64 camera
owner. Payload hashes are checked. No camera is opened. Nonzero inverse
Brown-Conrady distortion reuses the drawing mapper’s cached rays at its
supported 640×480 profile: a numpy port of the RealSense SDK’s deprojection,
pinned by a checked-in table of the SDK’s own answers, so no SDK wheel is
needed. Unsupported distortion models or dimensions are refused, never
approximated as pinhole.
Older captures missing this metadata cannot establish calibrated results.

RealSense records also retain `alignment_calibration`: the native depth and
color intrinsics and the SDK's depth-to-color extrinsics used for that exposure.
Rotation uses the SDK's column-major layout and translation is in metres.
The SDK maps samples onto color pixels while copying their native depth-axis
Z16 values. This metadata makes that distinction explicit; it does not alter
captured pixels or apply a fitted depth offset.

The shared `rgbd_geometry` implementation uses that transform to intersect each
color ray with its measured native-depth Z plane. Wrist and overhead scans,
stencil anchors/interiors, paper-reference landmarks and recordings carrying
this metadata use the resulting color-frame coordinates. Median captures retain
and freeze the alignment model for the whole batch. Geometry without the native
alignment metadata remains historical evidence; it cannot establish a new
calibrated live stencil surface. Calibration adoption still uses the existing
preview/apply workflow and requires a consistent camera-to-robot registration.

Run logs contain per-pair transforms, bidirectional overlap, geometric residual,
separate read/preparation/solver/output timings, CPU consumption and optional
5 Hz point-cloud RRD output through the shared writer. True material-point and
rotation errors, and false acceptance/rejection counts, are available for the
known-motion controls. Real recordings have no ground truth and are explicitly
unlabelled. A low ICP residual can still accompany an incorrect pose, as the
symmetry control demonstrates. Default thresholds are experimental comparison
settings, not qualified drawing tolerances.

Registration starts from identity for each consecutive pair. It does not chain
poses, recover persistent material identities, compensate wrist motion into robot
coordinates, or model deformation. Segmentation and camera-to-robot calibration
are separate requirements. Numerical thread pools are limited to one thread;
timing includes both comparison methods and is not a hard real-time guarantee.

Optional `--ground-truth FILE` accepts JSON with schema
`tatbot.surface-rgbd-ground-truth/1`, a `provenance` description of independent
measurement, and `frames` containing `timestamp_ns` plus rigid 4x4
`object_to_camera` matrices in meters. Every evaluated timestamp must be covered.
This enables physical-point RMS and rotation error for recordings; the benchmark
cannot verify the accuracy of supplied measurements. Do not use ICP outputs as
its own ground truth. Open3D is an optional offline dependency, pinned by the CLI;
its tests skip explicitly when unavailable in the general test environment.

`--image-check` measures RGB optical flow once per pair, filters forward/backward
and photometric inconsistencies, and compares the corresponding measured 3D
points with each registration transform. It reports residuals and unavailable
checks separately. These are correlated sensor measurements, not independent
physical ground truth, and include depth noise and nearest-pixel quantization.
They do not grant motion authority or change the existing candidate gates.
The check retains only the current/previous images and reuses cached SDK rays.

`--method colored` runs only colored registration; the default compares both
methods. `--max-points 5000` reduces the deterministic input sample cap from
20000. Compare consistency and timing on the same recording before choosing a
budget; fewer points are not automatically equivalent accuracy. The output
records method selection, point cap and consistency-check settings.

`tatbot vision depth compare` compares depth filter settings offline on a
retained raw capture burst.
