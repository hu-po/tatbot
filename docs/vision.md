---
summary: Public vision pipeline and timestamp contract
tags: [vision, cameras, replay]
updated: 2026-10-19
audience: [dev, contributor]
---

# Vision

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
place world-frame tracking axes in the URDF root, or `--camera-world` to draw
them as published, in the camera world. An absent or mismatched registration
makes the robot-root tracking geometry unavailable. Raw camera previews remain in sensor axes.
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
stack's station probe (`tatbot ros calib run`).

On the demo stack the D555 is the only fixed camera, so its colour optical
frame is the world. `tatbot ros calib overhead --arm <arm>` measures an arm in
it through the wrist D405, and is the registration the stack adopts:

- one D555 frame with the arm at rest, the wrist's holds over the palette's
  roof tag, and its straight-down views of the fiducial tile on the holder;
- both targets' corners placed in the arm base by the wrist (forward
  kinematics and the URDF's chain to the D405, the tile's corners by bundle
  adjustment with the D405's depth as a prior), fitted to where the D555 sees
  them, together with the wrist's depth of the scene on the D555's own;
- each region (the station, the holder) left out and predicted by the other:
  one region alone misses the other by about ten millimetres, so both are
  measured. A tile the D555 cannot read (the 12 mm forearm tile) has the
  holder's views aimed at the page the overhead tracks, and the holder rests
  on the wrist's depth there and on that tracked page's place: about 0.4-0.5°
  from the silicone tile's registration, where the cube's is about 2°, and
  the print within about 5 mm of where the wrist finds it. The silicone tile
  on the holder remains the better registration.

The wrist-tag registration below, `tatbot ros register --arm <arm>`, places
the camera by the cube on the carriage instead. The cube gives under the
tool's weight and its stickers stand millimetres off their measured layout,
and its fit came out tilted about 2° against the wrist camera and the
measured tool, which reads the palette station about 8 mm low; `tatbot ros
setup` runs it as the coarse first step, and `ros calib overhead` replaces it.
It measures an arm as follows.
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
bundle. A new bundle is a new world: the arm is registered again in it.

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

Each wrist D405 sits on the original Trossen bracket. The canonical
`urdf/tatbot.urdf` contains a fixed chain from each arm's `link_6` to its
`realsense_color_optical_frame` and `realsense_depth_optical_frame`. Its joint
origins match the [Trossen WidowX AI model](https://github.com/TrossenRobotics/ManiSkill-WidowX_AI/blob/main/wxai_follower.urdf).
The bracket pose is nominal CAD geometry, not a measured correction. The right
wrist carries a second bracket under it: the same chain, its mount joint rolled
180° about `link_6`'s x axis, as `right/realsense_lower_*`.

The camera registry binds `wrist_upper` and `wrist_lower` to the right arm, each
naming its own optical frames (two cameras on one arm must). The left arm is the
teleop leader alone and is not in the model. Moving an arm changes
camera pose through that arm's measured joint angles; it does not change the
fixed mounting transform. Use color optical coordinates for depth explicitly
aligned to color. Registering either view in the scene still needs
exposure-time joint observations and the robot-to-scene transform.

The lower bracket's mount joint is measured, not CAD: its pose to the upper
camera from both cameras' reads of one print (`internal/evidence/wrist_pair.py`),
1.76° and 1.96 mm off the stock bracket rolled 180°. The page filter reads it
beside the upper camera at every hover and lift. An image kept from before it
returned to the right wrist (it was the left arm's `wrist_left`, and before that
an earlier right lower camera) keeps its original geometry.

## Live viewer

Every live workflow (`tatbot live cockpit`, which subscribes to what the camera
owners publish on the bus, and the ROS 2 stack's Rerun bridge) streams into one fleet Rerun viewer (a
headless server on the `rerun-server` node; operator nodes attach windows) rather
than opening a viewer of its own:

```bash
tatbot viewer start            # on the server node: gRPC server on :9876 (+ web :9090)
tatbot viewer open             # attach a capped window on this node (--memory-limit 4GB --fullscreen on the station)
tatbot viewer status           # server, window, version, and the web viewer's ?url= address
tatbot viewer reset            # send the fixed blueprint again: every window drops its own edits
tatbot viewer view file.rrd    # stream a recording (.rrd, .wxtl, or a teleop run dir) into the fleet viewer
tatbot viewer install          # on the server node (sudo): the server unit and its load monitor
tatbot viewer station-install  # on the station (sudo): its load monitor
```

`viewer open` runs the window as the transient user unit
`tatbot-viewer-window`: the viewer drops its oldest rows at `--memory-limit`
(default 1GB), and the unit's `MemoryMax`, the cap plus 1 GiB, is the hard
ceiling. A window that already runs is left alone, so the station's launcher
can call it every time; `--restart` closes it first, because a window never
reconnects to a fleet viewer server that restarted. Once the window maps it is
maximized, or with `--fullscreen` made fullscreen (no title bar, no GNOME top
bar) and the focus handed back to the window that held it, so the station's
demo page opens behind its agent terminal and the operator alt-tabs to it.
Both are one EWMH request to the X11 window manager, no key typed; a window
already in the state is not asked again, and nothing happens off X11.
Each window keeps its own copy of the blueprint, and a tap or click edits it.
The eye beside a pane's maximize button hides that pane: an empty title strip
is left where it was, and while a pane is hidden Rerun 0.36 saves the whole
layout again on every frame. The station's page shows no blueprint panel, so
nothing on screen brings the pane back. `tatbot viewer reset` sends the fixed
blueprint again, and every attached window adopts it: hidden panes, a dragged
eye and dragged bounds return to the fixed ones, without restarting the hub or
a window.
`viewer status` prints the browser form,
`http://<server>:9090/?url=rerun%2Bhttp%3A%2F%2F<server>%3A9876%2Fproxy`,
with the server's rig address from the fleet map; it needs nothing installed
on the viewing machine.

A window pulls the heaviest stream on the rig, and the server may share its
rig port with an arm controller, whose command packets then wait behind each
depth chunk. A node with a direct cable to the server dials the server's
address on that cable instead: the server's `links` entry in the fleet map,
keyed by that node's role (`{"links": {"station": "<address>"}}`). The cable
is used only while the kernel routes its address out another interface than
the rig address and the server answers there within a second; otherwise the
node dials the rig address, so a dead cable never costs the window. Producers
on that node (its load monitor) follow the same rule. `viewer status` says
which path the window rides (`direct link to the station over <interface>` or
`rig LAN over <interface>`, with the reason the cable was passed over), and on
the server lists each client by the path it arrived on. A window keeps the path
it opened on: `viewer open --restart` moves it.

`viewer install` and `viewer station-install` refuse with exit 3 unless
`rerun --version` on that node reports the fleet's 0.36.0, and name the fix
(`uv tool install 'rerun-sdk==0.36.0'`). Each installs `tatbot-sysmon@`, which
logs the host's CPU and memory %, its hottest CPU/SoC thermal zone (deg C)
and the station's GPU % and temperature at 1 Hz under
`system/ros/` and `system/station/`: display only, about 0.1% of a core.
Each source's legend is its host's `platform` in the fleet map, else its role.
The station's monitor also reads each host whose `load_via` names the station's
role (the e-stop relay's board, which has no checkout) into
`system/estop-relay/`: one ssh session that reads `/proc/stat` and the thermal
zones every 2 s with shell builtins and a `sleep`, so nothing is installed,
started or written there. The ros node's monitor adds `system/robot/`: the
hottest motor of the arm the drawing stack drives (the controller's rotor
temperatures in the newest record of the stack's flight log) and that stack's
controller process's CPU in % of one core. It reads both from the newest
`ros-stack` run's files, never from the stack's graph or its control loop.
The Demo tab plots `cpu_percent` and `temp_c` of these four sources, each
in one colour in both panes: CPU % on a fixed 0-100 scale with the one
legend above its lines, beside TEMP, both over the last ten minutes.
The blueprint names only the entities. Each legend and colour is a static
row the monitor logs, so a blueprint sent from a newer `tatbot-visiond`
needs the monitors from the same revision, or older monitors show their
own labels in Rerun's per-entity colours.
On the ros node, `tatbot-cockpit.service` is the standing bus subscriber
(`visiond subscribe --recording-id live-cameras --duration-seconds 0
--camera-world`), installed by a fleet deploy of that node between draws. It
shows the D555's depth out to `--overhead-depth-max-m` (0.85 m: the table and
what stands on it, not the room behind; 0 shows all of it).
`tatbot rig check` reports all of these under part `station`, never under
`pink drawing`.

`live cockpit --duration 30` runs a bounded check. It opens no camera: it
subscribes to the frames the owners already publish on the bus. The operator
needs the fleet viewer and the bus router, but no camera-LAN address. It
streams into a recording of its own (`Cockpit HH:MM`), and every window on the
fleet viewer switches to a recording it has not seen, the station's demo page
included, as `viewer view` does on purpose: while the station shows the demo,
the standing `tatbot-cockpit.service` is the camera feed, and an ad hoc
cockpit is not run. It refuses `--recording-id live-cameras`, whose overlays
the standing cockpit places in camera world.

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
`session/`, `system/`). One blueprint definition in `rust/visiond` always presents the
fixed **Demo / Session / Telemetry / Calibration** top-level tab bar, in that
order, with no nested tab stacks. Demo comes first because the Rust blueprint
cannot select a tab: it is the portrait drawing page (status, the 3D scene in
the camera world, the design map beside the two upright wrist RGBs, the D555 RGB with the markers the trackers read
on it and the palette's tag, which the cockpit reads there itself once a second (`subscribe --fiducials`, the
inventory's `palette` target, outlined green and named), beside CPU and temperatures). The rerun-server installs it once per server generation, and `viewer reset` sends it again. Workflows publish data only, and neither workflow startup nor
recording selection resets the operator's selected tab.
Every viewer and producer is memory- and rate-capped;
`scripts/check rerun-caps` refuses a launcher or producer that is not. Python
producers use `scripts/lib/tatbot_rerun.py`.

`tatbot live cockpit --display-quality normal` caps the subscriber at 2 Hz;
`--display-quality showcase` raises that cap to 5 Hz, and `--fps` overrides
either. Showcase does not turn reduced upstream bus derivatives into
full-resolution sensor data. These controls affect display copies only, never
capture, tracking, policy, journal, or sensor rates.

Beneath that cap (`subscribe --max-fps`, default 5 Hz) each stream has its
own rate: the overhead colour at 5 Hz (`--overhead-color-hz`), its depth at
1 Hz (`--overhead-depth-hz`), each wrist tap's colour at 2 Hz (`--wrist-hz`).
The hub streams every row on to the station over a link the pink
controller's UDP may share, so no image the cockpit logs exceeds 60 KiB (its
chunk stays within 64 KiB, about 0.5 ms at 1 Gbps), and images go out one per
20 ms tick. Colour is re-encoded as a 640x360 JPEG of quality 70 (a lower
quality, then half the size, only for a frame that would not fit); the depth
is decimated to 320x180 by the nearest valid range in each block (zero stays
invalid) and logged RVL-compressed, lossless (`EncodedDepthImage`, the
Overhead depth pane). The Demo scene draws that depth instead as an RGB cloud
(`cloud/<band>`, `depth_cloud.rs`): every second return each way on its
pixel-centre ray under the same Pinhole, coloured with the mean of the colour
pixels it stands for in the newest colour image and a tenth of a cool blue
tint, each point 0.6 of the gap to its neighbours at its range so the cloud
reads as a surface, in four rows of at most 3,600 points so each stays within
the row cap: ~106 KB a second on the rig's table (2026-10-19), at most 230 KB.
The Scene leaves the depth image out, so a window shows the RGB cloud only
once it holds a blueprint sent since (`tatbot viewer reset`). The wrist tap's
depth is not logged: the bridge draws the wrist cloud the same way, with a
warm amber tint, so the two read apart where they overlap. The Demo tab has a
pane per wrist camera, D405-1 (RealSense 1, the lower) and D405-2 (RealSense 2,
the upper), each turned upright by its camera's display rotation (the lower is
the upper's mount turned 180°). Each keeps a draw's last frame under its age
label; a cockpit that starts
covers that frame, which it cannot vouch for, with a black one, clears its
label and shows "no frame yet" until that camera's first frame. The print's
markers each wrist reading decodes in a frame (a gauge hold's, a curved page's
locate view's, the tracker's hover and stroke-end frames) are drawn on that
frame: each marker's outline, green when the reading's accepted fit kept it,
amber when its fit kept it but was not accepted, red when read and left out,
its id at its centre, and a line naming the reading under the age label. They
come from the session's wrist tap as one `tatbot.wrist-markers/1` envelope on
`tatbot/vision/d405/<camera>/markers`, stamped as their frame, a few hundred
bytes each, about 0.6 s after it (the reading's own time); the cockpit logs
them at that stamp in the pixels it logs the frame at, and the camera's next
frame hides them. A flat page's locate views read a rectified page image and
are not drawn. At the defaults
the cockpit streams ~0.4 MB/s, ~0.5 while a draw's wrist tap sends; it
streamed ~2.6 MB/s in rows up to ~470 KB before 2026-10-18. `--camera-world` draws trackd's and
stencild's poses at identity in the camera world, the ROS `world` the Demo tab
shows. `--output FILE.rrd` also writes an offline capture, and `--connect` may
then be left out. `--duration-seconds 0` runs until SIGINT or SIGTERM. A
malformed sample is counted, reported on stderr at most every 10 s and skipped;
it never ends the subscriber.

## Evidence limits

Timestamp alignment is not geometric calibration, and a synchronized recording
is not proof of safe robot behavior. State those limits in every published
result.

### Tracker memory

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
view roles distinguish mounting locations. Both views are on the right arm:
`wrist_upper` and, under it, `wrist_lower` (the camera that was the left arm's
`wrist_left` until 2026-10-23). Both are owned by the ROS stack's node, which
opens them directly; no visiond D405 service runs for them.

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

The cockpit shows the overhead colour and depth and the wrist taps' colour as
display copies within its row cap (above). Valid-min decimation preserves the
depth scale and near edges. The overhead colour and depth each carry a Pinhole
from the owner's full-resolution intrinsics scaled to the logged size, so the
depth becomes a cloud in the camera world; the overhead 2D panes are rooted at
those pinholes. Full capture requests retain the original RGB-D frames.
Join an existing recording with
`tatbot live cockpit --recording-id <id>`. Raw camera-world wrist tracking is not overlaid on the
robot model without a measured registration between those frames.

The installed profile is colour 1280x720 and depth 640x360, both at 30 Hz,
depth preset 0, laser power 150 and a measured depth scale of 0.001 m per Z16
unit; librealsense aligns the depth to the colour, so the aligned depth is
1280x720 too. The colour's 0.86 px/mm on the page decodes skin-stencil
markers (an 18 mm marker's 3 mm cells at ~2.6 px; 12 mm markers, ~1.7 px a
cell, only upsampled twice: [the skin stencil](stencil-frames.md)), where
640x360's 0.43 px/mm decoded none. The two streams
together held 30 sets a second with no drop over a 30 s test on the camera
node. The depth stays at 640x360: in the installed view it gave substantially
better valid-pixel coverage and repeatability than 896x504 at 30 Hz or
1280x720 at 15 Hz. Preset 0 gave the best stable-pixel
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
accuracy.

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

The active bundle's two D555 entries hold the D555's native colour optical
frame, solved with its native intrinsics and distortion fixed so the solve
agrees with the pixel grid the owner aligns depth to. The strict companion `overhead_depth_depth` entry
carries the same optical geometry, because the owner aligns Z16 into color
before publishing.

Once both D555 entries exist in the active bundle, its owner stamps that bundle
and `trackd` reads the D555's stamped sets for each pose update. If the bundle lacks
either entry, the D555 stays available for camera-frame evidence and the
tracker does not subscribe to it. This makes calibration membership explicit:
an unstamped frame cannot become a world-frame tracking observation.

`stencild` is the same kind of subscriber for printed stencils: one observer
for the fleet beside the overhead camera's owner, reading its socket under the
same stamped bundle. Each turn it hands the newest overhead RGB-D set to the
observer; the D555 is its one view. The wrist camera's readings belong to the
drawing session.

Only a [fiducial-tile](stencil-frames.md) print is placed, by its markers
(`fiducials.stencil.rgbd_pose`). Each one found in the colour has each corner
lifted along its ray to the depth about its own pixel: the mean of the middle
half of a 15 x 15 px square (about 7 x 7 of the D555's own depth pixels),
taken over pixels whose mirror through the corner is valid too, so the arm
over one side does not shift it. No surface is fitted to the depth, under a
marker or across the sheet. A read is kept when it spans its printed side
within 15%, and the most reads, one per code, are kept whose corners all lie
no farther apart than printed and no nearer than their chord on the tightest
skin the layout goes on, within 4 mm (`Tile.curve_radius_m`: the forearm tile
a 35 mm forearm, the silicone tile 100 mm, its mount reading 146-191 mm): a
false read stands a pitch or more from where the others put it. Of equal
groups the one nearest its printed distances is kept, never the first by
code, so two true reads and a stray beside them keep the true pair. The
published pose is the layout laid unstretched round a cylinder through the
kept corners (`fiducials.stencil.bent`: any axis in the print's plane, convex
to the camera, no tighter than 0.75 of the layout's skin radius): its origin
the print's centre on that skin, its z the skin's normal there. Markers along
one row or column, or fewer than three, cannot show a bend, and the pose is
then the flat layout's rigid fit, the chord plane of the markers in view,
whose tilt moves with whatever part the arm leaves visible (a synthetic 40 mm
forearm, half hidden, turned it 22°). On a live forearm (2026-10-20) the chord
plane tilted 16.7° median and 25.3° p95; the bent layout read a 59 mm skin and
put the centre's normal at 14.3° median and 20.0° p95, the centre and turn
within 0.2 mm and 0.3° of the chord plane's. The drawing session reads the
pose for the page's centre and turn and for its first gauge holds and wrist
locate views; it judges a moved sheet by the kept markers' rays and measures
a curved skin at its gauge holds. The view that reads the most
markers anchors the pose. Fewer than two agreeing markers is no pose for
that turn (the arm over the sheet hides most of them), published `lost` with
the reason. The translation sigma is the
markers' residual floored at 2 mm, and the rotation sigma a fixed 1.5°: the
depth and registration bias the markers cannot see. An installed reference
that is not a tile, or fails to load, is published `lost` with the reason
under the pattern its manifest names and never stops the other prints. The
service publishes each print's pose as `tatbot.target-pose/1` on
`tatbot/tracking/target/<pattern_id>` with a `support` block: the anchor
camera, its marker count, ids and residual, both sigmas, the page corners
(`corners_m`, the pose's flat outline the viewer draws), the kept markers'
corners (`markers.ids[k]` names `anchor_points_m[4k:4k+4]`), the bundle id,
and `motion_authority: false`.

Its configured period is one second; `processing_p95_ms` and
`capture_to_publication_p95_ms` report the achieved rate and age. Per-turn
capture scratch uses the service's managed memory directory; durable
estimates stay in its run log. References are installed under the log root
(`stencils/references/<name>/tracking.json` beside its `stencil.png` and the
tile's `tile.json`) and rescanned by mtime, so installing one never restarts
the service and none installed leaves it idle. `--max-age-ms` (3 s) gates the
input only: a turn whose selected exposure is older than that, or in the
future, is skipped before the estimator runs and counted as a stale set. A
measured result is published as measured however long its turn took, stamped
with its capture time, so a consumer reads its age from the sample. A late
pose is never turned into `lost`: the ROS bridge holds the last measured pose
through a `lost`, which would be older still. The observer decides nothing
about motion. Each turn logs capture-write, estimator, publication, and
observer-stage timings with cumulative per-camera socket and selected frame
counts, sequence gaps and ring overwrites. The service metrics also expose the
stale-set count and the last capture-to-publication age.
The shared Rerun Session view includes each measured target's live page outline
under `/world/tracking/target/<pattern_id>` beside the calibrated robot. A
`lost` sample clears that outline.

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
the stencil observer's markers, the post-draw inspection's views and
recordings carrying this metadata use the resulting color-frame coordinates. Median captures retain
and freeze the alignment model for the whole batch. Geometry without the native
alignment metadata remains historical evidence; it cannot establish a new
calibrated live surface. Calibration adoption is `tatbot ros calib overhead
--adopt` and requires a consistent camera-to-robot registration.

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
