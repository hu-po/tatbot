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
silence. Not yet run: dipping. This page is the design
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
     the wrist camera's read of the print's markers corrects it in the plane.
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
     `inspect.hold_tip`): the gauge's surface plane, the print's markers read
     on it, the gauge's tip point carried into the print's frame, again with
     neither the kinematics nor the camera's mounting. Where the tip stood,
     against the gauge's skin point under the TCP (the TCP's foot, which the
     track's residual takes too, not the point sent 10 mm up), shifts the page
     onto the print by the median (`page.json` `holds`; a hold votes once, by
     its last reading, as the field and the track's seed take it: a curved
     hold reads again as it steps, from 5-10 mm higher; with
     `page.locate.cameras`, the lower wrist D405, each further camera reads
     the hold too, on its own gauge calibration and URDF pose, and the hold
     votes once: both readings averaged where both read the print by three
     markers or more (their offset kept, `offsets_m`), else the session
     camera's, else the further one's as a fallback;
     a further camera's heights join the hold's only within
     `page.height.gauge_spread_m` of the session camera's, else they are left
     out and said; its frames go to `touches/<role>/`), so the locate's far
     views no longer place the drawing. On a curved page it is turned too: by
     the median of each hold's own marker fit's turn (`holds_shift turned`,
     within 5°; over a forearm's 9-10 holds those read 0.1-0.4° apart, the
     locate's views 0.4-0.8°), then by the holds' places about their centre
     once five or more places span 5 mm both ways (holds within 5 mm of each
     other, a hold made again among them, are one; the whole turn within 5°):
     on the forearm print the places turned the page 2-3° where the markers
     read under 1°, and the track's residual follows the places. With the
     track correcting this moves no ink, its field being affine (replayed,
     every hover's prediction unchanged); it places the page wherever no
     field carries a turn (track log or off, a rigid watch, a field after a
     confirmed move), and it leaves the field less (2026-10-19: the hovers'
     median residual 1.45 -> 0.71 mm). `page.trim` adds to it. A shift past
     30 mm (a curved page's past 12 mm, `HOLD_SHIFT_CURVED_M`; the agreement
     and the confirmation below twice what they were before the owner doubled
     every stop, 2026-10-20, the curved limit 12 mm, not 18: past it nothing
     stops, the check holds cross-check the move, and that day's shifts read
     7.9 mm at most) leaves the set-up's page, except on a curved page whose
     holds agree:
     every kept reading, and every one by two markers or more (kept or
     dropped), within 3 mm of the shift, and those by two markers or more at
     two places or more (`holds_shift` `off_m`, `places`; a hold made again,
     under 5 mm from another, is the same place). Then the page moves onto it
     and the surface check's holds, each reading the print afresh at its own
     place, should put the tip within 6 mm (their median) of the moved page
     (`_confirm_holds`); read elsewhere, or by under two of them, the page is
     set up again once, afresh (`_new_page`, as mid-draw: the overhead places
     it, or the wrist from it when the overhead reads no print; the track's
     seed is that set-up's holds alone), and a page still unconfirmed then
     goes on, inking or not, the hovers' stencil watch judging moves from its
     holds' readings (refused, an inked draw ended at a mid-draw set-up; gone
     straight on, it inked where the newer check readings put the print past
     6 mm). A page they did not confirm is carried no further: no later
     set-up whose holds read nothing takes it as the last one placed, the
     next set-up's holds are made where this one's were, a mid-draw set-up
     does not keep it and a resume does not retain it; a set-up the overhead
     cannot place may start its locate from it. About 7.9 mm of a fresh set-up's shift is the tool: the
     gauge's calibrated tip stands that far across the axis from the drawing
     TCP. On a forearm (2026-10-20) four holds read +5.2 -11.7 mm, the
     farthest 1.3 mm from it at two places by 3-5 markers, and were refused;
     the tracker then moved strokes 12 mm on one marker and the page was set
     up again twice. Replayed, they move the page, three check holds confirm
     it at 1.0 mm, and the wrist's readings put the tip 0.6 mm from the plan
     (11.5 mm before); the silicone and forearm set-ups the limit passed are
     unchanged.
   - A height holds only on the IK branch it was measured on: the arm's own
     tip-height error moves ~5 mm per radian of joint 6 (2026-10-03). So the
     ready pose (the pen's heading and the joints) belongs to the page: a new
     page searches it from the rest pose and records it; a resume and the next
     run on the same page (carried with the paper) take it again, and a ready
     pose off the recorded one is named. The search takes the dips' heading
     (the station touch's, §6, when it speaks for the fitted tool) first, then
     15° steps either side of it, the first whose joint-limit margin over five
     points of the clear area reaches 0.45 rad (`ready.PREFER_MARGIN_RAD`);
     else the first heading within 0.05 rad of the best. Each dip turns the
     tool from the strokes' heading to the touch's and back at 8°/s: on a
     forearm whose strokes went at -105° against the touch's -30° that was
     ~36 s a dip (2026-10-20); at the -45° the rule takes there it is ~8 s,
     and the whole curved design plans at -45° with 0.36-0.44 rad to spare.
     A ready pose carried from another run whose heading turns past 45° from
     the dips' is not carried.
5. **The bench is the test.** Unit tests cover pure functions (compile,
   motion, fits). No mock executor stands in for the driver.
6. **Calibration is data, never code.** Tool tips, page, station and camera
   poses live in `config/workspace.yaml` and become static transforms.
7. **A code change reaches the arm in under a minute.**
8. **Budgets:**
   - about 21k lines of code (raised from 20k on 2026-10-18 for the simulator's hardware backend and batch ports,
     to 21,074 on 2026-10-20 for a curved setup's adaptive surface check and overhead-seeded first hold, and to
     21,084 the same day for curved holds that step toward the skin only on agreeing frames, checks over the field,
     to 21,103 for a curved page's holds that agree past their limit and the locate's tilt cap, to 21,113 for
     what those holds must agree on and a move no check holds confirmed carried no further, to 21,123 for a
     curved page set up afresh on the overhead's own page and one vote a hold, to 21,244 for a person's skin
     followed by the overhead's markers and set up again only for a move that lasts, in place of operator pauses,
     to 21,267 for a continue that lifts and measures the skin again and a snag trip whose needle comes out at
     once, to 21,327 for a shorter setup: one wrist camera read behind the holds' steps, the locate's view
     hint, later holds over the first's skin and one check hold where the holds agree, to 21,357 for a
     watch that reads a move off the followed page however long ago it came, a wait out of the overhead's view,
     and a curved follow checked against a print turned round the arm, and to 21,413 for a page the wrist and
     the gauge holds place when the overhead reads no print, its readings off the page left out, and a locate
     that sets aside an overhead pose two wrist views read past three sigmas, to 21,488 for the owner's doubled
     stops: a dip return judged once the arm settles and planned again instead of ending the run, a dip phase
     off its place dipped again with no key, a palette's small move carried, strokes at the dips' heading and
     the skin watch judged net of the operator's trim, and to 21,506 for a blind set-up that starts a resume
     15 mm up, rises with the skin its locate views read and carries the gauge tip's offset into its
     holds, the overhead's readings of a limb's own move up to 0.2 m past the print, to 21,510 for a page
     its check holds did not confirm set up again once instead of inked on, to 21,538 for a swing to rest
     the snag guard does not watch, a withdraw that holds when it cannot be planned, and the holds' offset
     a set-up carries turned with the pen's heading, to 21,637 for a curved inking stroke that rides
     higher by the skin its hovers read, a fresh page's first hover read and never set up for, and hovers
     kept for the skin when the print reads nowhere, to 21,649 for two unread hovers in a row going on
     over a page the wrist placed blind, to 21,658 for a stroke that lifts at its end when the charge dips
     before its continuation, to 21,670 for a lift the arm cannot follow along the skin's normal tried
     along the pen and at half the distance, to 21,686 for each wrist reading handing the markers it
     decoded to the wrist tap for the demo viewer, to 21,814 for dot-sized paths dropped where other ink
     lays them and the pen's passes over each spot counted, planned and drawn, to 21,818 for the pen
     trim's limit said with it, the skin watch's set-up for the pen's own ride and the status board's
     trim bar, to 21,899 for smoother moves between scan positions: one move up and over between
     gauge holds, joint moves under acceleration and jerk caps, and a hover travel that starts where the
     arm stands, to 21,945 for kinematics whose pinocchio data is the thread's own (a lift planned
     beside the stroke pre-planner had read its placements) and a landing's lift tried in steps along
     the pen that holds the needles off the sweep when none plans, to 21,975 for a dot dropped
     only on its own ink's line and the pen's passes counted at the tool's mark width, to 21,977
     for the skin watch judging the skin's part alike at every trim and saying what the ride lifts
     and the keys press, to 21,982 for a joint move from rest that lifts off slowly before it
     speeds up, to 21,995 for a held landing that no automatic land starts from, only a person's
     decide, and a gauge hold's hop from the stand its travel commanded, every hop not taken said,
     to 22,011 for a landing held where the driver keeps its latch or a pen's withdraw falls
     short, a snag's needle lifted on clear of the skin, and a hold said to a person before their
     land starts from it, to 22,040 for a stroke's pre-plan sent by its own ride's change along
     the depth axis, as the trim is, instead of planned again at the hover, to 22,106 for a
     keep region: a part of a design compiled on the whole design's placement, to be drawn again
     over its own lines, to 22,137 for a needle stroke's trim no deeper than skin_stop_m past
     the skin read along it, a set-up's or resume's trim eased back to it, and the depth a stroke
     holds the keys at shown and said, to 22,153 for a landing that withdraws and holds only
     over the print and LAND_CLEAR_M round it (the cameras' page standing in for a page not yet
     set up), and a session client whose exit is its verb's when the middleware fails to close,
     and to 22,233 for a locate that looks straight down first and a blind locate whose
     views read no marker searching on over the skin the overhead's depth reads, and to 22,256
     for a curved liftoff with no foot under the tip, a print read on the plane past the field,
     and a field's relief reported along the strokes, and to 22,427 for a stroke that rides the
     skin its hover read along its own path (and its blind start the page's last minute's), not one
     reading about the tip, and to 22,497 for a print placed by its markers' lifted corners in 3-D,
     read in each of a view's frames, and a search about the skin nearest a moved limb, and to 22,542
     for a transfer that gives with the skin read by each place a code reads at, every miss said, and to
     22,590 for locate views chosen by the print's markers they see past the tool, straight down from as
     high as its smallest marker keeps 40 px and aside of it, and to 22,607 for that tool's footprint standing
     on the gauge's measured tip and a marker hidden by the needle across its middle, and to 22,675 for the
     palette camera's sweep measuring the tip's length by tilts about the installed tip, the tip found in focus, and to
     23,158 for the overhead's registration through the wrist D405, not the wrist cube, to 23,161 for S1's seed
     held to the registration its touch was made in, to 23,174 for that registration's capture landing and
     waking the arm between its regions, so the holder's views plan from rest, to 23,247 for calibration views by
     one joint move where it keeps clear of the station and the stylus, the roof-tag holds and S4's yaws visited
     nearest first, and a goal's first knot sent 50 ms in, to 23,264 for the wrist gauge reading the probe's top
     square to the pen, not the body's wall, to 23,280 for its holds tried only where the camera sees that top
     across the probe, to 23,291 for a gain profile chosen when the stack starts, which calibration runs on, to
     23,299 for a registration holding one connection to the stack through its run, to 23,304 for a gauge
     correction past 3 mm refused as the known heights' error, not the gauge's, to 23,335 for the palette
     camera's tip taken where two finders agree or else the lower, a park on the ball stopped, and a height the
     overhead fix refutes refused, to 23,366 for the overhead's holder placed by the wrist's depth from views
     aimed at the bridge's page where the D555 reads no tile, to 23,369 for the palette camera's change grown
     from its sure part through the weaker, to 23,373 for a page surface's foot found when its fixed point
     runs away, and every safety command carrying the guard level asked last, to 23,389 for the print under
     the Demo's design map, to 23,401 for every wrist D405 opener taking the camera stack.yaml names by its
     vision.toml role, as a second D405 shares the ros node's USB, to 23,576 for the print's offset as one
     filter of the wrist's fits and the overhead's marker moves, `page.track.estimator`, to 23,580 for
     `tatbot ros up --estimator`, so a rehearsal runs the filter while the default stays the field, to
     23,592 for the second wrist D405's preview: opened beside the session's camera and closed with it, and
     drawn by the bridge beside the first, to 23,657 for the filter reading that second D405 at every hover
     and lift, with an offset state of its own and its calibration carried by the URDF, to 23,674 for
     each stop's error shared by the cameras that read it, so a second view never counts a stop twice, to
     23,678 for those readings' markers drawn on that D405's own pane, under a reading name the pane does not
     repeat its camera in, to 23,769 for both wrist D405s at the set-up too: each locate view and each gauge
     hold read by the lower as well, a hold's print one vote of both, its heights the lower's only where they
     agree, to 23,777 for the page filter taking whole a slide the overhead holds over three readings rather
     than weighing it down as a misread, to 23,782 for a hold's two cameras averaged only where both read the
     print by three markers or more, the offset between them kept, to 23,783 for the lower wrist D405 among
     the wrist's bodies a locate view keeps on its side and over the page, to 23,846 for the filter's wrist
     reading while a stroke draws, every half second besides its stops, to 23,850 for its line a stroke
     measured from the plan's line, not a point the pen trails, and to 23,867 for wrist readings that keep
     pulling one way taken as the sheet's move);
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
camera node:     visiond (D555) → stencild ── existing zenoh bus ──▶ tatbot_bridge (arm node)
                 page pose `world_from_target` per printed stencil          → /page (ROS)
rerun-server:    fleet Rerun hub ◀── tatbot_rerun (in the stack launch; opens its stream only through tatbot_rerun.start())
tatbot bus:      tatbot_session's wrist tap ── tatbot/vision/d405/<camera> ──▶ tatbot_rerun, the demo cockpit
                                            └─ <camera>/markers (each reading's markers) ──▶ the demo cockpit
```

Perception stays where it works for now. The camera services (`visiond`,
`stencild`) keep running unchanged on today's zenoh bus.
- `tatbot_bridge` subscribes to their page poses with zenoh-python and
  republishes them in ROS. `rmw_zenoh` cannot talk to plain zenoh
  applications, so the bridge is its own zenoh session to the existing
  router.
- Porting the cameras themselves is a later step:
  - The D555 is an Ethernet/DDS camera, so any Jazzy host can own it with
    `realsense2_camera`.
  - The capture and the stencil observer would become composable nodes in
    one process, so full-resolution frames never cross the network.
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
| `tatbot_rerun` | Python | ROS → Rerun bridge for the fleet viewer's demo page: robot, strokes, design map, page, palette, wrist cloud, status board (§13 Graph). Its pure models are in `tatbot_contracts` (`ros_scene`, `ros_status`) | 300 (about 410 since the demo viewer) |

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

Preparation places artwork on the nominal stencil page, 128 × 172 mm with a
90 × 134 mm clear centre. Both the rotated canvas and the stroke footprint
must fit. A fiducial-tile print (docs/stencil-frames.md; the silicone layout is
111 × 157.5 mm) is drawn on as it was generated: the session takes the page's
size and clear centre from the installed print (§4.3, Page) and refuses,
before the arm moves, a drawing whose lines leave that print's clear centre. Translation, rotation and
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
drawn on (its directory, with `tracking.json` and `settings.json`): the
program's page and clear centre are that print's, so a design that does not
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
   No secondary fill, simplification or scheduling occurs. One deduplication
   does: a dot-sized path (every two of its points within `--min-stroke-mm`,
   default 1) lying within a pen width of a larger path's line of its own
   resource is dropped, a descent and a lift (~10 s on the arm) and a repass
   that add no ink; a dot on bare skin or on another ink's line stays. `preparation.dropped_dots` names them, `stats.dots` counts
   them and the dots kept; `--min-stroke-mm 0` keeps every path.
   `--keep-region=X,Y;X,Y;...` (a polygon in page millimetres) then keeps only
   the placed paths inside it, as placed: a part of a design drawn again over
   its own lines on the same print. A path crossing the polygon's edge goes with
   the side holding most of its length, unless both sides hold more than 20 mm
   of it (a contour outline round the whole design): then it is cut at the
   edge, each cut on the segment it cuts, and its runs inside are kept, a
   dot-sized run dropped. Dots are judged on the whole design first. The
   compiler schedules the kept strokes' dips afresh; `preparation.keep_region`
   records the polygon and each crossing path's decision, and the preview
   draws the polygon.
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

`stats.passes` (`tatbot_ink.passes`) counts how often the pen passes over each
spot: each stroke walked at its tool's mark width on a grid of a sixth of it, a
cell passed again whenever the pen comes back over it after going more than a
mark width away (a retrace, another stroke). The mark width is
`--mark-width-mm`, else the tool's line width, else each path's generation
width (0.3 mm for the 3RL, whose mark on skin is not measured);
`stats.passes.marks` names it per resource. It gives the most and the 95th
percentile per inked cell, the mm² at 2, 3 and 5 or more passes and the five
densest 1 mm squares by mean passes; `preview.svg` shades 3 or more under the
strokes. A run records the same for what it drew (§4.5). Tracking only.

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

`substrate` names a `config/substrates.yaml` entry that every resource's tool
datasheet admits (`substrates:`): `paper_pad` or `paper_cylinder` for the
ballpoint, `silicon_skin` or `human_skin` for the needle cartridges. The
substrate is recorded in the program and changes nothing else: a needle
cartridge rides, dips and moves the same on human skin as on silicone, and its
depth is trimmed live from the station keyboard. A one-pen draw without
`--inks` names no substrate. For the 3RL on a person's forearm:

```yaml
format: tatbot-inks
version: 3
substrate: human_skin
resources:
  - id: black
    tool_id: lutin-3rl-bugpin
    ink_id: nighthawk_black
    pen_mode: ride
    slot: inkcap_medium_1
    activation: {method: manual, action: exchange}
bindings:
  - {artwork_sha256: "<exact artwork digest>", pen_id: pen-1, resource_id: black}
```

Compile with `tatbot ros compile artwork.json --inks resources.yaml -o prepared`,
then `tatbot ros draw prepared/program.json --arm right`. The first resource is
the one the run starts with: load its cartridge first. At a tool change to
another resource the arm lands and the run stops, naming the cartridge to fit;
fit it and resume with `--resume <run-id>`, which keeps the run's measured page
(§4.4): the last one the gauge holds placed on the print, while the overhead,
asked again from a cleared view when the arm hides the sheet, reads the
print's markers within 3 mm (slide and rise together) and 2.5° of where it
read them at that setup (**Page motion**). A curved page kept without those
markers is set up again. A change to a resource of the same tool and ink goes on without
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
chunking, explicit colour sequence, physical bounds, width provenance, dropped
dots and pass counts. The session tests run compiled continuations through the real
Cartesian leg builder with substituted IK/driver, preserving resume/lift
behavior. Physical qualification remains a separate DBV3 A/A drawing trial.

Draw runs record monotonic attempt durations in `run.jsonl`, with a
`draw_timing` summary in `meta.json`. Resumes aggregate all completed attempts;
missing attempt evidence leaves the total unknown. These measure action time,
not physical contact time. The research cost fields are defined in
[`docs/research.md`](../docs/research.md).

### 4.3 Register

After setting the rig up anywhere, `tatbot ros setup --hardware real --estop udp` runs the steps below in order
and lands the arm: a restart with a fixed page and the probe, the arm's registration from holds planned off the
wrist tags the D555 sees at rest (`register --prior rest`, nothing from an earlier layout), a restart on the
stencil, the D555 registered again through the wrist camera (`calib overhead --adopt`, aimed by the tags' one,
which it replaces), the station touch (`calib run --station-only`), the ball's height (`calib height`), the wrist
gauge (`calib gauge`) and the page set-up (`touch`). It prints each step's time. The steps before the page run on
the `calibration` gains (below, ros2_control), and a restart puts the role's own in before the page. A setup that
stops early lands the arm and restarts the stack on those too, so no draw starts on the calibration gains.
It always runs in its own session and the command follows its log, so a closed terminal or an agent's tool
timeout stops only the following, never the setup; `--detach` returns at once instead. `tatbot ros draw` runs
the same way, with `tatbot ros draw --progress` counting the strokes done. `tatbot ros setup
--progress` reads the newest one, running or finished, in a few lines.

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
  - The operator puts a printed fiducial-tile page (docs/stencil-frames.md)
    in front of the arm and installs its reference (`tatbot vision stencil
    reference --install`: `tracking.json`, `tile.json` and the generator's
    `settings.json`) on the overhead camera's node, then points
    `page.pattern_id` at it, or names it at `tatbot ros up --pattern <id>` (the
    hex digits its sheet prints are enough). `ros up` reads the page's size and
    clear centre from that installed print (`tatbot_session.config.print_page`),
    so a print of any size needs no other configuration; with no print installed
    under that ID, `page.size_m` and `page.clear_m` stand. It must be where the
    D555 sees it and the arm reaches, and it may move a little during a run.
  - `stencild` places it each turn by its markers in the D555's RGB-D
    (docs/vision.md): every marker names its place on the print, so the pose
    carries no identity guess and no search region. The wrist camera and the
    touches then measure it. `tatbot_bridge` publishes the page pose, its
    covariance, and the print ID. The arm resting over the page after a touch
    hides most of its markers from the D555; a draw needs it measured within
    `page.max_lost_s`. At run start it waits up to `page.wait_s` (240 s) for a lost print to be
    measured again, which covers the second side of a held pair. A set-up that
    has a page this run placed to start from (a mid-draw set-up, or a resume
    with a retained page) and the gauge's holds to place it (`page.locate.holds`,
    the gauge in use) waits 10 s, then 10 s more from rest, and goes on without
    the overhead (`_last_page`, `_blind_prior`, `_blind_placed`): the last page
    raised along its normal, 8 mm mid-draw (the page in use, whose skin the last
    hover read: the overhead's page stood 4.7-9.3 mm over the gauge's skin, so
    the first hold and views stand as high over it) and 15 mm on a goal's first
    set-up (a retained page of any age, the limb perhaps put back since; the
    first hold, 25 mm over it, still steps down to an unmoved skin in its three
    5 mm steps). The wrist's locate alone places it. When its views read no
    marker (an inked print the overhead no longer decodes, the limb perhaps
    moved since), it searches the skin the overhead's depth still reads,
    captured at rest (`_overhead_skin`, `_skin_search`, `inspect.skin_aims`):
    centred on that skin now (the median of its points within 50 mm of the
    page and twice the print's extent of its centre: a stale page put the
    views and holds on a forearm's flank, 2026-10-21), a view over that
    centre and each of the eight points the print's extent (its 10 mm margin
    included) from it where 50 depth points or more lie, the centre first,
    each planned about the page raised to stand 8 mm over its own highest skin
    point (never lowered, no surface fitted) at the locate's clearance, its
    way judged on that page, until one reads the print's markers, which place
    it as any view's do. With no such skin by the last page (a limb put back
    elsewhere on the mat) it searches about the skin nearest it within 150 mm
    (`inspect.wide_aims`: what stands 15-120 mm over the mat, the arm's own
    wrist and the mat left out); the overhead's depth is read with the arm at
    rest (`clear_view` first). A search that reads none, or no capture to aim it,
    leaves the last page standing and the refusal below. A curved page is then
    raised, never lowered, to stand 8 mm over the highest skin the locate views'
    depth planes read (less `page.height.wrist_depth_bias_m`), before the ready
    move and the first hold; with the print read, its holds stand off the
    wrist's page as the last page's holds stood off the wrist's then (`_held_off`:
    the gauge tip's ~7 mm across the pen axis, which a set-up placed afresh
    carries in its correction). That offset is the tool's: five runs' pages
    read it 85-103° from their ready heading at headings -15 to -106°
    (2026-10-20), so it is turned about the page's normal by the new heading
    less the last page's once the ready move has chosen it (`_turn_carried`;
    `_level_prior`'s carried correction alike). The holds must place it on the print, else not
    drawn. Its ready pose is a new page's, the dips' heading first (§4), never
    the last page's: its holds are made afresh at it. Nothing measures the skin before the locate views, which keep the
    pen 7.5 mm over the page they start from at least. The overhead's watch has
    no reference then; it takes one once three of its readings come within 20 s
    (`_rebase`). On 2026-10-20 a forearm's tattoo ended in such a wait: the
    overhead no longer read the inked print while the wrist read 4-7 of its
    markers at every hold and 3-5 at every hover; replayed through `setup_page`
    with the run's page messages and `page.max_lost_s` at its 5 s then, the
    set-up goes on 24.5 s after the continue, where it had waited 230 s more and
    would have refused (at 10 s, the doubled stops, the overhead's last reading
    keeps the page at once). Replayed on that tattoo's recorded locate views,
    their depth fitted again about the page a blind set-up starts from, its
    first hold starts 19.3-19.9 mm over the skin the holds then read mid-draw
    and 24.8-24.9 mm on a resume (on the overhead's page 15.9-17.4), raised
    0-3.1 mm, and its holds shift 1.1-2.1 mm, where without the carried offset
    they would shift 6.1 and 9.0 mm; one resume crosses a 29 mm move of the
    limb between set-ups. An arm not
    at rest (a cancelled draw holds where it stopped, over the print) first
    lifts clear and returns to rest, holding, so a resumed run keeps its
    controller; a rest path that would pass the pen within half the standoff
    of the page goes over the page centre at 60 mm first (from a hold over a
    raised holder the straight move swept the pen back over it 4 mm up), and
    one that still would is not taken.
  - tf2 crosses the pose into the arm's base through the arm's registration.
  - Before the touches (`page.locate.wrist`), on an overhead pose, the arm first
    stands the pen square to the page over its centre `page.locate.over_m` up
    (50, then 100 mm) at the heading and IK branch the ready move will take
    (`_ready_for`: the carried paper's ready pose on the same print, else
    `ready.pen_heading`'s search, which the ready move then takes), and both
    wrist D405s read the print there (`inspect.over_centre`: the joint margins,
    the pen's and the wrist bodies' clearance over the page and a way off the
    arm itself kept; no side line, as the ready pose under it has none). It
    rises only while fewer than `page.locate.over_enough` camera reads place the
    print, and the views below come only after: they took ~30 s of view search
    and two ~5.5 rad swings of a 160 s setup (2026-10-24). Over 37 setups'
    overhead pages (2026-10-16..24) the 50 mm view was reachable on 36 (a
    forearm's page tilted 23° at 100 mm), seeing all six of the 981a tile's
    markers in both cameras, 22 of the silicone tile's 23 and five or six of the
    forearm's twelve, the ready move under it 0.15-0.22 rad. A view that would
    see none of the print's markers is not moved to. Its depth stands a curved
    page's first hold (`_seed_hold`, `_skin_over`): the median of its points
    within 5 mm of the hold's foot, raised by the tcp's height over the gauge's
    tip, as the gauge would read the skin; round a bottle that hold had stood
    4.2 mm over the skin and been held again 6.0 mm higher. Taken from the views
    below instead (0.19 m off, another arm configuration) on two bottle setups,
    it put each hold's skin 0.4-3.9 mm low, near
    `page.height.wrist_depth_bias_m`; with too few points the overhead's markers
    stand it as before. With too few reads, and on a blind locate (no overhead
    pose), the arm points its wrist D405 at the page from a few rest-near poses
    on its own side (planned about the
    overhead pose tilted at most 10° from the base's z, `AIM_TILT_RAD`, its
    centre and turn kept: about a forearm print's 19-20° none was reachable,
    capped 22-24 were; the locate says how many views it reached), looking
    straight down first: each view is sought within 10° of the base's z from
    0.19 m over the page levelled (`inspect.top_down`, `TOP_DOWN`), and from as
    high as the print's smallest marker keeps 40 px a side (`MARKER_PX`; an 18 mm
    marker 0.27 m, `TOP_DOWN_MAX_M`), looking at its centre or 40 mm aside of it
    (`AROUND`), and only when none is reached obliquely as before (20-50°,
    0.18-0.24 m). Among the views reached, those seeing the most of the print's
    markers are taken before the least contorted (`inspect.markers_seen`: all four
    corners in the frame, none of its corners, edges' midpoints or centre on the
    tool's footprint, `tool_footprint`: the fitted tool's datasheet profile projected
    through the mount and moved onto the gauge's measured tip, which the CAD mount
    puts 11 mm off it): the pen and the machine stand on the camera's own axis, and
    over the silicone tile (2026-10-22) the markers in view read 75%, the footprint
    told the read from the missed in 91% of them, and 94% of those it left clear read. There three views 0.27 m up, 40 mm aside, saw
    22-23 of its 23 where the two 50° views taken before saw 14-18, and one read 21
    at 0.75 mm rms where a 50° view read 18 at 1.39 (3.86 at most). Their search
    solves each candidate's IK to 100 iterations: those that converge took 5, 71 at
    most, and the rest had taken 98% of a search (53 s before the first view moved,
    2026-10-22; three views now 11 s); each view says whether one was reachable and
    how long the search took. Views
    35-40° off from 0.18-0.22 m saw a forearm print's 12 mm markers 20-35 px
    and squashed and read none, where hovers facing them read 40-53 px
    (2026-10-21). The D405's ~651 px focal length puts a marker at 41 px from
    0.19 m, the tip 31 mm over the target on the camera's axis; the camera
    sitting 48° off the pen, such a view tips the pen 40-55° (the oblique
    views keep it within 30°), its wrist cube no lower than the tool mount,
    the side line 0.10 m farther (the ready pose's own cube reached 59 mm
    across it on that page), joint_0-2 within 0.8 rad of the pen-down
    reference (`max_arm_rad`); every other check as before. On that page,
    replayed: views 10° off over the centre and four of its eight
    neighbours, camera 187 mm over the target, joint_0-2 0.37-0.71 rad and
    the wrist 0.52-0.69 rad from the reference, ~6 s to choose each. The
    locate then maps the
    median-stacked frames onto the page plane through the overhead pose, and
    reads the print's markers there (`inspect.view_fit`): one rigid move that
    most markers agree with, accepted as docs/stencil-frames.md describes. A
    view that reads none adds nothing. On a curved page no image is
    rectified: the markers are read in the median frame and in each of the
    view's frames (plain and upsampled; one a median blurred, 3 of 5 frames
    read, 2026-10-21), each code's reads grouped by the place it read at (a
    false read elsewhere kept apart), each corner lifted by the pooled depth
    about it, and the print placed by those corners in 3-D
    (`track.lifted_fit`): each marker within 25% of its printed size, the most
    that keep their printed distances to one another within 2.5 mm and 5% (a
    misread or a false code stands a pitch off, or off the print), the layout
    fitted to them rigidly (laid round the skin's curve when they span both
    ways), accepted within 1.6 mm rms of that layout across each corner's ray
    (its depth noise lies along it; about a plane, a 44 mm forearm's own curve
    read 1.08 mm median against 0.88). A faded transfer on a forearm turned since it
    was laid gives with the skin: its true markers lifted 10.4-13.9 mm a side
    and 2.6 mm off their printed distances, 1.41 mm rms (2026-10-21). A view
    that places nothing says why, marker by marker, as does a gauge hold and
    an unreachable search view. Then the print's centre
    goes onto the page in use along its normal and its u axis turns into it.
    That page supplies neither shape nor tilt: carried onto a stale prior
    tilted 25° off the skin, the markers' spacing shrank by its cosine and
    misplaced the print 1.3-1.4 mm and 1.7° (synthetic; at 40° refused). No surface is fitted to the skin
    for depth: the gauge holds then measure it. Rectified onto the overhead's
    plane, 8.8 mm over a forearm's skin, a 50° view read no marker and the
    other a turn 0.5° off the holds' (2026-10-19); by depth both read. At a
    first setup the holds' own turn and shift then place the page. The
    poses combine by median, with the spread as sigma, and the result fuses
    with the overhead pose by inverse variance on x, y and yaw (`page.locate`
    sigmas). Two views or more that read the print past three sigmas of the
    two from the overhead pose (13.4 mm or 5.4° with today's) set it aside:
    the wrist's own estimate places the page (`overhead_set_aside` in the
    record). A forearm tattoo's resume (2026-10-20) took the overhead's first
    pose while its poses jumped 35-80 mm frame to frame; the fused page stood 14.5 mm and
    3.7° from where the holds put it, and their 7.8 mm shift came within
    1.2 mm of the curved limit. Replayed through `wrist_locate` on its
    recorded views, the wrist's page stands 7.7 mm and 1.3° from it, as the
    run's two other setups stood 4.8 and 7.0 mm, and the holds' shift falls
    from 8.6 to 1.3 mm by the replay's measure; those two and a silicone
    run's four fuse as they did. The run's
    `page.json` keeps both and the fused pose; `locate/` keeps the views.
    Each view is read by `page.locate.cameras` too (the lower wrist D405,
    through the session camera's preview, at its own URDF pose): its fit
    joins the views' in the median, is recorded under the locate's
    `further`, and one that reads the print counts as the view reading it
    (`locate/view<n>-<role>.png`).
    The first view is searched about the azimuth the last setup reached its
    first at (the carried paper's, `_view_hint`), and among every view only
    when none is reached there: every view took 39-43 s on the ros node
    before the arm moved (2026-10-20), three azimuths 0.6 s of a 27-32 s
    search offline. A view's azimuth is the page's, but its reach is the
    arm's, so the azimuth is carried in the arm's base too (`view_base_deg`)
    and turned onto the next page. Carried as the page's alone, a print
    turned 87-125° since its last setup missed it and searched every view,
    63-83 s, four setups running (2026-10-24). Which markers the tool hides is
    the print's, so on the same print both azimuths are searched together and
    the view seeing the most markers is kept; the page's alone when the
    base's lies within 15° of it. Replayed on the real arm model over the
    setups since 2026-10-22, the hint reached a view in every one. A search of every view that reaches none ends the locate:
    the next poses' searches, from where the arm still stands, are the same
    (three alike, 117-122 s, 2026-10-17/18/20).
  - The setup reads by one wrist D405, opened at its locate or first reading
    and kept open to the tracker (`setup_camera`; the goal's end closes it
    otherwise), each grab taking frames from now (`grab_both(fresh=True)`):
    opened at each reading it cost ~2 s (the pipeline and 30 frames of auto
    exposure) at each of a setup's 14-23 readings (2026-10-20). A reading's
    files are written behind it on one worker and the print in each camera's
    frames is read behind it on another (`_hold_tip`, `READ_WORKERS`), all
    settled (`_settle`) before the holds place the page, the tracker reads
    their frames or the goal ends; no step waits for either. On one worker
    both took ~13 s a hold on the silicone tile's 19 markers a frame, its
    holds 10 s apart, and held the setup 48 s after its holds (2026-10-24);
    the reads are mostly the markers' edge fits under the GIL, so one worker a
    camera, and the black print's pass fits only the codes the violet one
    missed (`stencil.find`, 67 to 52 ms a frame). A failed grab closes the
    camera, and the next reading opens one afresh.
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
    overhead page, the wrist D405 depth plane (a flat page's only: on a curved
    one it is a plane over the skin, and neither it nor the `planes:` report
    enters) and each touch's first contact (`page.height` sigmas). A source far from the rest is named in a warning,
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
    placed at least 5 mm inside the clear center stays on the print.
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
motion. `hover` keeps the machine off and holds the path `pen.hover.height_m`
(6 mm) over the page, touching nothing. The run's first event says which
(`pen down: …`), and `page.json` records it.

**Rehearsal in the air.** `--rehearse` draws the real program with nothing
touching the skin: the same strokes, page setup, gauge holds, hover stops,
stencil follow and set-ups again as with ink, every resource hovering
(`tatbot_contracts.ros_program.rehearsal`).

```sh
tatbot ros up --hardware real --estop udp --page stencil --arms right --probe --surface fitted
tatbot ros draw prepared/program.json --arm right --rehearse                    # 6 mm over the skin
tatbot ros draw prepared/program.json --arm right --rehearse --standoff-mm 8    # higher
for i in 1 2 3; do tatbot ros draw prepared/program.json --arm right --rehearse || break; done
```

- Dips are skipped: a rehearsal drops every `dip` op and palette cap, and a
  rinse becomes a manual change. Resources of one tool become one ink, so
  their tool changes go on without landing. A cartridge swap to another tool
  still lands.
- The tattoo machine stays off and the snag guard is not armed; the pen trim
  keys still move the path.
- The post-draw inspection is skipped. The arm lands after each run unless
  `--hold`.
- `--standoff-mm` is the tube end's height over the measured skin (3-20 mm).
  The session refuses one closer than the fitted cartridge's `needle_reach_mm`
  past its tube plus 1 mm (the 3RL: 5 mm): with the machine off the needles
  may rest anywhere in their stroke.
- Each rehearsal is its own run (`ros-draw`): its `track.jsonl`, page events
  and page rows show every reading, follow, skin reading and set-up the
  inked run would make. `rehearsal.of` in its program is the source
  program's digest.

**The pen trim** (`tatbot_motion.trim`, `motion.yaml` `pen.trim`). The
operator's drawing keys on the station keyboard move the pen-down path along its depth
axis while the arm draws: Fn+W lifts it one `step_m` (0.1 mm, lighter), Fn+S
presses it one step, Fn+X sets it back to 0, within ±`limit_m` (4 mm; the
Demo page's trim bar shows it, yellow at the limit). A held key steps once.
At −4 mm the 3RL's tube rides 1.0 mm over the gauge-measured skin and its
needles reach 3.0 mm past it by the datasheet (ride 5.0 mm, reach 4.0 mm).
The 2 mm limit held a forearm tattoo at −2 mm for 141 presses where the
wrist read the skin up to 4.06 mm under the field (2026-10-20). The trim
belongs to the cartridge on the arm (the program resource) for the whole
run: each change is a `pen_trim` ledger row and a `pen_trim` event, and a
resume starts each cartridge at its last one; a new run starts at 0.
- A needle cartridge's stroke takes the trim no deeper than
  `page.surface.rigid.skin_stop_m` (2.5 mm, the operator's deepest 3RL rung)
  past its plan over the highest skin read along it, less the `rise_m` it
  rides by: its own hover's readings and, past its ring, those near its path
  (the ride's, below); a stroke whose own hover read no skin counts the field:
  the hovers about its path may lift it, but a sink they read never takes it
  deeper (`_cap_for`; the `sent` row's `trim_cap_m`). A deep trim reaches
  only as far as the stroke's hovers read the skin sunk; the keys' own trim
  stays theirs, a press past the cap moves nothing on the arm and says where
  the stroke holds it, and a stroke it holds says so (a `pen_trim` event).
  A set-up and a resume ease a trim kept deeper than `skin_stop_m` back to
  it (a `pen_trim` row with `eased_from_m`, and an event). With one trim per
  cartridge, depth keyed for one sunk stretch of a forearm otherwise carried
  onto strokes, fresh fields and resumes it was never pressed for, up to the
  needles 3.0 mm past the gauged skin (review, 2026-10-21). Replayed over
  that tattoo's last presses at ±4 mm: 41 presses on a stroke read 0.15 mm
  under the field held at −2.65 mm instead of −4.0, the resume's −4.0 went
  back to −2.5, the strokes read 4.06 and 3.69 mm under took the keys'
  −4.0, and nine earlier strokes held 0.06-0.24 mm. A ballpoint takes the
  keys as they are.
- Every stroke plan carries, per knot, its depth axis (the page normal: a
  surface would give each knot its own) and `dq_dh`, the joints per metre
  along it with the tool's rotation and the carriage held
  (`clik.depth_sensitivity`). A trim h is added as `q + h·dq_dh` when the
  goal is sent: no new IK, and never off the plan's IK branch (§1, rule 4).
  On the pink arm's drawing poses that lands within 4 µm of the target at
  1 mm and 18 µm at 2 mm (2026-10-04), and over a forearm tattoo's 959
  drawing poses within 13 µm at 2 mm and 53 µm at 4 mm (2026-10-20); a full
  re-plan of a 40 s stroke takes 4-9 s.
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
- Fn+Enter (`enter`) answers a waiting pause or latch with `continue`. After a cartridge
  swap lands the arm, `client draw` waits for it, wakes the arm and resumes
  the run (Ctrl-C there leaves it to resume by hand).
- Fn+E (`dip`) dips now and makes every later dip a step sooner; Fn+Q
  (`fewer_dips`) makes them a step later: the dip scale of a dipping cartridge (§6).
- The keys are F13-F18, which no typing sends, so the keyboard is never
  grabbed. Fn never reaches the host: the keyboard's own keymap (VIA) puts
  F13-F18 on Fn+W, S, X, E, Q and Enter. `ros/tatbot_keys_relay` on the station node reads every keyboard's
  evdev and sends `TBK1 <key>` per press over UDP to `stack.yaml` `keypad.port`
  (7644) on the ros node (`tatbot ros relay-install --keys` installs it as
  `tatbot-keys-relay.service`). There `tatbot_session.keypad` publishes
  `/tatbot/keys/<arm>`. A held key sends once; no other key is sent. The
  keypad takes datagrams from `keypad.from_addr` alone, which `tatbot ros up`
  resolves to the station node's `lan` (unset: this host only), as the e-stop
  reader takes its relay's frames alone.

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
  `machine.confirm_s` (2 s); the pen-up travel and descent that follow give it
  time to spin up. A machine not reported running pauses the run with the
  switch's reason (the e-stop, no answer, …): `land` lands; `continue` asks
  again, a stroke from the top of its loop, so the sheet is measured again
  first (the move check, the watch, a hover): the limb may have been moved
  while the e-stop was pressed.
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
Every joint-space move (`ready.joint_move`: the locate's views, the ready move,
rest, the station's ways) is a quintic under the pen-up joint speed and
`ready.JOINT_ACCEL_RAD_S2` (1 rad/s²) and `JOINT_JERK_RAD_S3` (5 rad/s³, the
jerk a quintic starts and stops with): on the speed alone the ready move from
the last locate view peaked at 2.4-2.7 rad/s² and the tip rang 0.6-0.7 mm after
it stopped, and the drop from the high ready pose took 50 mm toward the skin in
0.5 s (2026-10-20); capped they take 2.0 and 1.3-1.6 s.
A move from rest (joints 1 and 2 on their stops) first lifts off: its largest
joint's first `ready.LIFT_RAD` (0.1 rad) is a quintic of its own peaking at
`LIFT_RAD_S` (0.1 rad/s), and the capped quintic for the rest of the way
starts at that one's peak. At rest the stops carry the arm and joint_3's
velocity integral sits at its clamp against its gravity feedforward, so lifted
off, the wrist falls ahead of its reference until the loop catches it:
10-13 mrad leaving at 0.05-0.1 rad/s, 17.6-18.2 mrad at 0.3-0.5 rad/s.
The locate's views go at the pen-up joint speed too: at a third of it joint_3,
lowering under 0.3 rad/s, chattered at 5 Hz as each move slowed, the tip
2.8-6.7 mm on the 5-8 s swing from rest (now about 3.5 s) and 3 mrad for 5 s
between views.
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

**A curved page** (`page.surface.mode: fitted`, or `tatbot ros up --surface
fitted` for one stack; `plane`, the default, is every flat page above) is set
up the same way, and the page found is the base of a
surface (`tatbot_motion.surface.PageSurface`, `tatbot_session.curved`):
- The gauge holds stand at the drawing's corners and centre
  (`curved.layout`), inside the print's clear centre less its inset
  (`page.inset_m`, 5 mm
  without one), all at the full hover, and the gauge reads the skin under
  the tip as a quadric over the plane's own support, the 8 mm about the tip
  left out (`gauge.measure(curved=True)`). One plane through a gauge frame
  sits 2.5-3 mm under a forearm's crest (synthetic, 40-60 mm radius); within
  ~8 mm of the tip the depth reads the skin 1-3 mm off either way. The
  quadric reads synthetic forearms 0.13-0.19 mm short and today's 20 flat
  holds within ±0.35 mm of the plane gauge. It reads only the surface within
  40 mm of the tip (`gauge.SUPPORT_M`): on a skin pinned over a raised holder
  the mat 50 mm under the crest filled the image box, and under window glare
  it was most of what the depth read. A hold's 25 frames are read in pooled
  fives (`page.surface.pool`, `gauge.pool`: each pixel's median where most
  frames read it); single frames on that glare spread 4 mm, pooled ones ~1 mm.
  The print's markers are read in all 25 colour frames, not the five the
  pools keep: on a faint bicep transfer (2026-10-24) a frame decoded 0-4 of
  its six markers, a different few each time, on the ros node's OpenCV 4.6,
  and three of five holds read none or one in their five; in all 25 three
  holds kept two in their printed shape. A well-read print reads the same.
- A curved page set up afresh (not kept mid-draw) starts on the overhead's
  own page: the last set-up's correction keeps only its place and turn in
  that page's plane (the tool's ~7.9 mm), never its height or tilt, nor does a
  carried paper's height move it, and the last field goes before the locate.
  The overhead reads each skin 4.7-9.3 mm over the gauge's, its own way: a
  silicone's correction carried into a live forearm's set-up stood its first
  hold 7.6 mm over it (2026-10-20), and a later silicone's would have stood
  the next forearm's ready pose and first hold 5.3-9.0 mm over it,
  the ready move's floor 0.4-4.1 mm. On the overhead's page the locate's
  clearance, the ready move and the first hold are judged as at a stack's
  first setup: replayed over the forearms' fields, the ready pose and the
  first hold 14.5-18 mm up, the floor 9.5-13 mm, and the first hold steps
  down to 10 mm on frames that agree.
- A first setup's first hold stands over the higher of its prior page and
  the skin the overhead's marker corners put under it (in 3-D, within
  12 mm), less the offset the last setup's first hold read between them and
  the skin; each setup logs that offset (silicone 8.1-9.3 mm, a live forearm
  6.1 mm, 2026-10-19/20). It never stands lower than the prior puts it. A
  hold any of whose pooled readings puts the tip more than 2 mm under the full
  hover is raised by the nearest at once, whatever the others read. One more
  than 2 mm over it goes down only on a firm reading (three pooled readings
  at least, agreeing within `gauge_spread_m`: one alone agrees with itself),
  by the nearest and 5 mm a step, three steps at most, a further step only
  after a reading within 1.5 mm of where the last aimed. On a forearm's
  flank (2026-10-20) readings 7.9-19.7 mm apart had taken the tip 8.7 mm down
  by their median, to 4.8-7.1 mm over the skin by its fields.
- The holds after a setup's first fly there open-loop and unguarded. Inside
  the patch of the first hold that read firm (its convex hull over the page)
  each starts `touch.start_above_m` and 1.5 mm (`HOLD_SKIN_M`) over that
  hold's own skin, a quadric through its patch (`_first_skin`), along its
  normal there (`_hold_start`), and steps as any hold. Elsewhere each starts
  1.5 mm over the skin the locate's over-centre views' depth puts there
  (`_skin_over`: on the silicone tile, 2026-10-24, the hold outside the first's
  patch stood 16.2 mm by the stand below and was held again, where the first,
  by that depth, read 9.96 mm). Without that depth, or after a
  first hold too few frames read to be firm, each starts 5 mm over the stand
  where the hold before it last read firm and needed no step (a step toward
  the skin is no depth until a reading there confirms it), and comes down a
  step: started there, holds read 13.7-17.1 mm and took two readings or three
  each (2026-10-20), and on skin 6-9 mm higher than under the hold before
  stood 0.9-4 mm over it (2026-10-13/20, before that rule). Replayed through
  `_gauge_point` over the last setup of every curved run since 2026-10-13
  (63, each hold's recorded patch and readings), the 940 readings became
  799 (2026-10-20's five setups: 12, 10, 10, 11 and 9 to 7, 7, 7, 7 and 6)
  with the same points, and no hold after the first started
  under 9.1 mm over its field (by the stand rule alone: 8.1; median 12.2,
  was 15.8); no step toward the skin was taken on frames that disagree.
- The skin each hold sees, carried into the base by its pose and pinned at
  the tip, is fitted as a degree-4 height field over the page set up, skin
  past the print (the mat) left out; the page record holds it, a resume
  restores it and its re-gauge refits it. Its set-up line gives the field's
  relief along the program's strokes, beside the gauge points' span, and
  over all the skin fitted: that skin reaches the forearm's flanks, and a
  14 mm heart's fields spanned 0.8 mm along its strokes where their extent
  read 18-24 mm (2026-10-21). The page plane through the gauge points is
  only the field's base frame: the field's own constant and slope terms
  carry the height and strokes are lifted off the field alone; only a
  clearance past the field's extent (a liftoff, `height_over`) measures
  from the plane. A field held to its holds' span would be wrong on a
  forearm: past the corner holds a 25-30 mm arm's skin falls 4-7 mm below
  them under a 30 mm design (synthetic, `test_curved.py`), and a field
  clamped down onto a crest between holds would put the needle deeper. Holds that read no print leave the
  page where the overhead's slide puts the last placed one, as on a flat
  page.
- A page set up again mid-draw (`_reacquire`) is held on the page in use,
  moved in its own plane as the overhead's marker readings of the last
  `max_lost_s` put the sheet (`_kept_page`): the last holds' height and normal,
  never the overhead's pose, which with the arm over the sheet tilted 11.5°
  and rose 3.8 mm (one of ten re-setups, 2026-10-19). Its holds and locate
  views are made on it, the old field on it and no earlier correction (their
  pen clearance and the views' corners are judged there), and holds that read
  nothing keep its place. Without such a reading, or when the markers read
  the sheet risen or sunk past 4 mm along the page's normal (a lifted limb:
  its holds would stand that far in the skin), it is set up as a first setup
  is, from the overhead's pose, which rose with the sheet, measured once the
  arm has cleared its view. No mid-draw set-up starts from the paper this
  run's first setup touched (`_carried_paper`, within 15 mm of the risen
  pose): on a limb lifted 12 mm it put the ready move and the first hold 2 mm
  in the skin. A run's first setup on a flat page carries that paper only
  while the cameras' page stands within 4 mm of it along its normal.
- The print's millimetres are arc lengths on the skin: a transfer wraps a
  gentle curve without stretching. The chart unfolds the surface from the
  page's centre triangle by triangle from its edge lengths (Inkmap's
  intrinsic unfold), so a design placed in print mm lands in print mm.
- `plan_op` and `plan_hover` take the surface: stroke points densified to
  0.5 mm (a chord across a curve cuts under it), lifted along each point's
  own normal, the pen turned to it along the stroke, approach and lift along
  the end points' normals, every knot's depth axis its own normal. A flat or
  absent surface plans exactly as the plane.
- A stroke point farther than `page.surface.reach_m` (20 mm) from every hold
  (its gauge point where the page in use puts it, moved by the holds' shift)
  is refused (an extrapolation), and with `check` holds at the
  corner edges' midpoints read the skin against the field before any stroke,
  each 10 mm over the field just fitted along the holds' page normal (at the
  holds' depth over a page tilted off a forearm's flank they stood 5.6-7.9 mm
  over it, 2026-10-20), raised as a hold is but never lowered, and rises
  before it travels to the next: up and over in one move (below), lifted off
  the last. Holds that each gave a point and predict
  one another within 0.5 mm (`CHECK_ONE_M`, `curved.fit`'s leave-one-out),
  on a page no check must confirm (`_confirm_holds`), are checked first at
  the midpoint by the hold the others predict worst, and at the other three
  only when it reads past half `max_loo_m` or is not counted; otherwise all
  four. Four took 67-71 s of a setup (2026-10-20); over the 98 curved setups
  of 2026-10-13..20 the 53 this checks once took 59 checks for 212 and found
  a check past the limit where the four did (the one among them read
  0.74-0.96 mm at all four: a skin that sank after its holds).
  Their skin, like every curved hover's, is turned by the camera tilt the fit
  took out of the holds' (`curved.untilt`). A check past `max_loo_m`
  (0.9 mm) means the skin moved under the holds or the field missed its
  shape, and it never refuses: inking or rehearsing alike two more holds read
  the skin again at the worst check and halfway from there to the centre (in
  no fit), and the path, the ride or a rehearsal's hover, rises off the
  holds' own field by the most any check or fresh hold put the skin over it,
  about its tip or at its point (the page record's `surface.rise_m`, kept
  through a resource change, a page set up again and a resume). The field
  stays the holds': refitted through the checks, one read 1.41 mm low pulled
  it up to 0.83 mm under them beneath the strokes (2026-10-18's 1.36 mm
  forearm, replayed), deeper there for a miss, and single readings on a still
  forearm scatter -1.6..+0.7 mm. A miss never takes the needle deeper than the
  holds' field put it; its lines come out lighter, which the depth keys trim.
  Refusing ended live forearms' setups at 1.01 and 1.36 mm (2026-10-18/20)
  and would end a tattoo at a set-up repeated mid-draw. Replayed from its
  saved frames, the 1.36 mm forearm rides 0.46 mm higher on a steady skin or
  one sunk 1.0 mm after its checks, 1.08 mm higher when it rose 1.3 mm, ink
  and rehearsal alike; the 1.01 mm forearm's checks now read within the limit
  (0.89 mm), and at a 0.8 mm limit it rides 0.89-2.19 mm higher, steady to
  risen 1.3 mm; silicone setups pass their checks and ride as before. On the
  1.01 mm forearm 0.16 mm was the check read unturned, ~0.6 mm the forearm
  rising between two checks.
  The holds and checks stand square to the page; with `turn_pen` the strokes
  turn to the skin, and the arm's own error turns with its pose. On a skin
  round a bottle (~33 mm radius, 2026-10-24) the hovers at one end of the
  drawing read it 1.4-1.8 mm over the holds' field and those at the other
  0.0, where the holds read the same skin on it: pinned at the FK tip, the
  skin moves with the arm's error, ~0.35 mm of it the camera's tip turning
  about the TCP and ~1.1 mm the arm's configuration (joint_1 57 against
  80°). So after the check, `turned_checks` (2) holds stand at the check
  point where the field turns most and the one across from it, the pen
  turned to the field's normal as a stroke's; what each reads over the
  field, less the square check's there, is a lift the strokes and hovers
  ride along the drawing (`curved.turned_lift`: linear between the two,
  held beyond, never deeper; `page.surface.turned`), and the skin watch and
  the ride take it out where they read, as they take out `rise_m`.
  `tatbot ros touch --arm right` runs the setup alone (its holds about the
  page centre).
  - Five holds carry a human forearm under a 30 x 30 mm design. Synthetic
    arms of 30-60 mm radius, tapering 3%/cm, their axis tilted 10-15° along
    the arm (`test_curved.py`): noise-free the field is within 0.1 mm, and
    with each hold 0.2 mm off at random within 0.35 mm. The checks then read
    within 0.6 mm, inside the gate. With the camera looking across the arm,
    the far flank grazes and the field is 0.6 mm off: turn the arm so the
    camera looks along it.
- **The stencil is followed rigidly**, never stroke by stroke. With
  `page.track.mode: correct` and its hover stops, the tracker watches the
  stencil on a curved page (`curved.RigidWatch`, `page.surface.rigid`).
  - Each hover's and lift's reading of the tip on the print is compared with
    what the gauge holds read at that place. Their field carries the arm's
    and the reading's own error there.
  - A reading within `follow_m` (2 mm) says the stencil stays. One past it is
    held until a second, at least 3 mm away (the hover's second view),
    agrees on one rigid move within 1.5 mm.
  - The whole page surface, its pose, height field and chart, then moves
    with the stencil (`ArmExecutor._rigid_follow`, a `page` row
    `followed: by wrist`). The hover over the stroke's start is measured
    again on the followed page. The stroke goes down there on its own points.
  - The page is set up again (fresh holds, fresh field) instead when the
    strokes would move past `max_m` (10 mm) or `max_rad` (10°) from where the
    holds measured them. It is also set up again when the print cannot tell
    the move apart (`curved.ambiguity`). A forearm rolled about its own axis
    moves the print round the curve and keeps its shape; one slid across
    its rest carries the shape with it. Both read alike in print mm. Past
    `ambiguity_m` (1 mm) or `ambiguity_rad` (0.1) apart at the design's
    points, the holds decide. Along the arm the two agree. Across a 30-40 mm
    arm, a move of two or three millimetres sets the page up again. (Each of
    these, and each stop below, twice what it was before the owner doubled
    every stop, 2026-10-20.)
  - Each curved hover also reads the skin about the tip (five pooled frames,
    the ring 8-20 mm from it) against the surface in use: the needle's ride
    off its plan if the stroke went down now, net of what the pen already
    rides higher by: the operator's trim (+ up) and the surface check's
    `rise_m`. Past `skin_m` (2 mm) it is a warning that says what the ride
    lifts and how much of it is the depth keys'. The stops judge the skin's
    part (net of `rise_m`), alike at every trim: depth keys pressed its way
    add nothing, as they are the operator's depth and a set-up keeps them;
    keys pressed against it take theirs out. Past `skin_stop_m` (2.5 mm) over
    the field the page is set up again at once, with no pause, the pen lifted
    at the hover; past `skin_stop_under_m` (5 mm) under it, once the next
    hover reads it past too. Not at a page's first reading after its holds
    (the rehover after a set-up included): fresh holds read the forearm's skin on the field where
    its hovers read it 1-1.8 mm over (2026-10-20), and a set-up for that drew
    the next stroke at the same excess. What that first reading read past a
    stop is the page's floor; a later hover sets it up again only 1 mm past it
    (`FLOOR_M`).
  - **A stroke rides its own skin.** Each hover's views keep the skin they see
    8-40 mm about the tip (`track.Tracker.wide`), and `_ride_over` reads it
    along the stroke's own path, as it is planned (`curved.skin_along`: each
    path point's skin points within 2 mm on the page's plane, their median
    off the field along its normal; no surface fitted to the skin), smoothed by
    a robust line over 3 mm of arc either side (`curved.ride_profile`). The
    stroke and the continuations it joins ride that profile point by point
    (`curved.Ridden`, a `sent` row's `ride_m` its least): rising freely,
    under its plan at most `profile_lower_m` (1 mm, owner). Where its own
    views read no skin (the 8 mm the pen hides: most of a 10 mm stroke) it
    borrows the skin the page's frames of earlier strokes read (hovers, second
    views, lifts; carried by their own poses alike), newest first, within
    `profile_keep_s` (60 s: the heart's skin drifted ~0.8 mm a minute) and
    never across a page set up again, each only when its skin agrees within
    `profile_agree_m` (1 mm: that drift and the views' 0.03-0.10 mm scatter)
    with the skin the stroke's own views read where both read (`lent_off`;
    those always read the ring about the tip), so a stroke its own hover
    reads none of rides a lent frame's profile; on a program of 7-10 mm strokes the own
    hover's views reach none of most strokes' points and those of the two
    before it nearly all (median 99%, geometry). Points still unread ride by the
    ride below, never under the plan; then raised wherever it would change faster than
    1:2 along the path, so no step goes into the skin. A stroke's line says
    the skin read along it, the ride and whether the page was set up again
    for it. The skin read past `profile_setup_m` (4 mm, owner; 2 mm until 2026-10-23) off the field
    along it sets the page up again before the stroke, without a pause (not at
    a fresh page's first hover). A stroke riding a profile is planned again at
    dispatch, never sent from its pre-plan. The tiny heart (2026-10-21): its
    hovers' rings read the skin 0.24-0.96 mm under the field while one lobe
    sank 1.2-1.3 mm against the other; replayed from its hovers' frames, its
    first stroke would have ridden ~0.3 mm under on both lobes and its second
    level on the left and 0.8 mm under (1.0 at most) on the right where its
    views saw the skin, level inside the pen's 8 mm, with no set-up at 2 mm
    (at 1 mm the second would have set the page up). The depth keys add on
    top, and the trim's caps hold.
  - An inking stroke on such a page rides higher by the skin read over the
    field, never lower than planned (`_ride_over`, the `sent` row's `ride_m`),
    less the `rise_m` it rides by already, with the operator's trim on top: its
    own hover's readings, and past 20 mm from its start (beyond that hover's
    ring) those of the page's hovers in the last 5 min within 10 mm of its
    path. The continuations it joins pen-down ride with it, and a stroke
    measured again is never lowered. Its pre-plan, made at the ride before
    (the ride is set at its own hover), is sent by the difference along each
    knot's depth axis as the trim is (the `sent` row's `fold_m`; within 21 um
    of planning it at the ride): planning it again held the needles over the
    skin 2-6 s with the machine on, on 67 of that tattoo's 112 stroke pairs.
    It is planned again when the trim and the fold would pass the trim's
    limit, or when the fold would still ease in once the tip is 3 mm more
    than any lift over its first pen-down point (`FOLD_CLEAR_M`). A stroke
    whose hover read no skin, or that had none (tracking stopped), takes the
    page's readings within 20 mm of its path; with none it rides
    `skin_stop_m` higher. On 2026-10-20's
    tattoo 17 of 34 hovers read the skin 1.0-2.3 mm over the field, under the
    stop, and each of those strokes went in that much past its plan; the
    operator's trims there (+0.7 to +1.0 mm) did by hand what this does, so a
    resume keeping them rides twice. The deeper side is not the owner's 5 mm: the
    3RL rides its tube 5 mm over the field and its needles reach 4, so a skin
    5 mm over would take the whole reach; 2.5 mm is the operator's deepest
    rung (1.5 mm in). Judged gross at 2 mm, the watch stopped a forearm
    tattoo twice at 2.29 and 2.13 mm with the trim at +0.7 and +1.0 mm (1.6
    and 1.1 mm net; 2026-10-20). Judged net of keys pressed its way too, a
    skin 0.1-0.5 mm over set the page up at once with the trim at -2.0 to
    -2.5 mm, inside a still forearm's scatter, and one press past -2.5 kept
    any skin under 2.5 mm over from setting it up (review, 2026-10-21). A
    set-up cannot change the keys'; where strokes ride by the hovers the ride
    takes the skin's part over the field, elsewhere it goes in on top of the
    keys as at trim 0. This runs with `track_correct` on too. Replayed over
    790 hover frames of nine still skins (paper, silicone, a forearm) it
    never read past 2 mm, and past 0.9 mm in up to 45 of a run's 118; a still forearm read -1.6..+0.7 mm at
    single hovers that the next one did not repeat (2026-10-20), and one that
    sank 4-5 mm (2026-10-18) read so from its first hover.
  - The off-plan check counts curved stroke ends from `follow_m`, the holds'
    residual taken out. With the print's markers kept the overhead watches the
    skin as **Page motion** says; without them it follows past 12 mm or 12°
    and sets the page up again past 24 mm or 24° (`curved.WATCH`).
  - `page.surface.track_correct` (on) moves each stroke's points by the
    page's affine field from the wrist's readings, and then the rigid follow
    stands aside. The fiducial tile's markers name their place, so a reading
    cannot land on a repeat of the print; on curved tile prints the corrected
    strokes landed within about 1 mm. The flower-of-life's band readings ran
    15-20% off across a curve, and with them this correction once carried a
    20 mm stroke 5 mm onto skin turned 18° (2026-10-08).
- Not curved yet: the post-draw
  inspection and `page.surface.mode: plane` treat the page as a plane. No
  set-up decision, follow or prior reads that pose once the print's markers are
  kept (**Page motion**): on a still forearm with the arm hovering low its
  tilt swung 0.1-18° and stopped a draw at "13.8°" (2026-10-19). The overhead
  reads the skin's rise along the page's normal from its markers' ranges (a
  re-setup keeps no risen page, the check before each stroke sets one up
  again); the wrist's skin
  watch sees it at every curved hover, and the contact cap, snag guard and
  e-stop stay.

**Stencil tracking** reads the fiducial-tile print (`docs/stencil-frames.md`)
by its AprilTag markers wherever the wrist sees it (`inspect.tile_fit`: found
in the raw frames, carried onto the page or a curved page's skin). Each
decoded marker names its place on the print, so a reading cannot land a
repeat away and needs no gate against the overhead. A hover that sees no
whole marker (a view that falls between them) looks again from the second
view, `second_view_m` (5 mm) toward the page centre, before the sheet is
taken to have left. A print the wrist camera has read nowhere in a goal is
drawn on the measured page without hover stops, said once; an inking pen on a
curved page goes on hovering for the skin alone (`_skin_alone`). A tracker
that fails after a lift is started again there.
With the print's markers kept the overhead watches the sheet by them before
every stroke (**Page motion**), with hover stops or without. A page without
markers is watched by the overhead's poses when there are no hover stops:
before each stroke descends, the median slide and turn of its five newest
poses since setup (`sheet_slide`, at the page centre), once the part the page
has not followed passes `follow_m`/`follow_rad` (1.5 mm, ~1°), moves the page
by it and writes a page row (a resume takes the page where it went); a step
that most of those poses put past `follow_max_m`/`follow_max_rad` (20 mm,
11.5°; a sheet picked up and put down) sets the page up again instead. It
compares them with the overhead's own pose for the page in use
(`page_overhead`), never with the wrist-located page: on one half-skin the two
were 7.5 mm apart. Half-skins slid 15 mm and 6° and spun 12° under needle drag on
2026-10-06, in jumps of 2-5° within single strokes, which no correction
between strokes can undo.
With `page.track.mode: log`
(`tatbot_session.track`), the wrist D405 the setup opened stays open through a
draw's strokes, and three worker processes read the holds' frames at once
(one took 6 s for a setup's 9-11, 2026-10-20), each reading logged in the order
taken. After each stroke's final lift a worker reads a fresh frame as the
gauge holds read theirs (`inspect.hold_tip`): the print's markers on the
gauge's surface, accepted when two or more agree on one rigid move (or one
read near its printed place):
- A reading the run's page cannot take is tried again from the page moved in
  its own plane by the sheet's slide the page has not followed
  (`sheet_slide`), when that passes 2 mm or 1°: the markers are read through
  the page's map, which reaches only about a centimetre past the page
  believed. No pose's tilt or height enters it. A hover that finds no print,
  nor from its second view, sets the page up again when the overhead's
  markers read the sheet moved (the watch fires or is pending, **Page
  motion**). When it has read none since the pen left the page (the arm over
  the print hides them), the arm first clears the overhead's view and waits
  up to 10 s for three readings: still none, or a moved sheet, is set up
  again; one where the page is brings the arm back (the ready move) to
  measure the hover again. On 2026-10-20 a forearm was set up twice for that
  hiding, ~4.5 min each, its markers within 0.7 mm of the page. No stroke goes
  down blind from a page the sheet has left. On a page the wrist placed while
  the overhead reads no print (no marker reference yet) nothing can tell, and
  each such hover would be set up again: two in a row go on with their skin
  read, the third is set up again (`UNREAD_SETUP`).
- A frame whose gauge reads no surface by the tip (no pen, or no skin beside
  it) is not a missing print: its markers are lifted by the frame's own depth
  (the median about each corner), carried by the camera's FK pose onto the
  page along its normal (a curved page's chart unrolls them) and taken when
  two or more agree, with no plane fitted to the skin. The goal's gauge holds
  are read the same way at setup, so the FK pose's few millimetres cancel; a
  hover whose reading puts the print within `stays_m` (10 mm) of where the
  holds put it, at its markers, keeps the page, and is not set up again. That
  reading carries no height, so `stays_m` was not doubled with the stops.
- Each measurement is a page event and a row of `<run>/track.jsonl`, with its
  frame under `<run>/track/`. It records:
  - the pen tip on the print (the gauge's tip point, read on the gauge's
    surface as the holds were) against the stroke's planned end, with the
    markers read and their rms;
  - the print's pose in the base against the run's first measurement.
- With `log` it moves nothing.
- With `correct`, every accepted reading is also a sample of where the pen inks less
  where the stack believes the TCP is: the goal's five gauge holds before the
  first stroke, and every lift after.
  - Each stroke's points move by the residual the filter (below) predicts at
    them, capped at its `cap_m` (60 mm), so the ink lands on the plan after
    the sheet slides or turns, and wherever the arm's error is. The estimator
    before it, `track.Field` (an age-weighted affine fit of the readings plus
    what that left, averaged by distance), was removed on 2026-10-24: replayed
    on 38 tracked runs the filter stood at or ahead of it on mild skin and on
    strongly curved skin (a forearm's flank, a bicep), and the field's local
    term, added to the filter, made the curved runs worse.
  - With `hover_stop`, the arm stops over each stroke's start (`plan_hover`)
    and measures there before it descends, so the stroke takes the correction
    at its own start. A sheet moved between strokes is caught before the next
    one inks. The travel is pre-planned with the stroke, which is planned from
    its end. Every descent is measured first: only a stroke the planner joins
    pen-down where the tip stands (`tatbot_motion.joins`, a continuation
    straight on from its chain) has no hover stop, never one whose pen left
    the path for a latch, a snag trip, a pause or a dip.
  - **A hover reading the filter does not take lets the overhead move the
    page** of a print whose markers are not kept (with them, the watch before
    each stroke follows, and the filter takes the markers' moves itself). Such
    a reading was not accepted, found no print, or was a jump no second view
    confirmed. The overhead's slide then moves the page
    as it does without hover stops: past `follow_m` (1.5 mm), or a turn past
    `reach_rad` (2.9°). Past `follow_max_m` the page is set up again. The
    stroke ends read before the follow are cleared; the filter carries the
    page's follows in its model.
    - On 2026-10-08 an untaped half-skin slid ~14 mm and 6° under five
      1011RL strokes. The overhead measured it throughout. The wrist read the
      inked print 4-9 mm off it, and every reading was refused or held, so
      strokes went down 4.5-9.8 mm off.
    - Replayed, the page follows at four hovers there, and the strokes start
      within 1.3 mm of the overhead's sheet (9.6 mm before).
    - A turn alone needs the reach because the overhead's yaw flickers 1-2°
      on a still sheet. Over 32 earlier flat tracked runs, the eight follows
      a smaller turn caused moved still pages 0.2-1.4 mm; at the reach, none.
      (A curved page follows its stencil rigidly instead: §4.3.)
  - **The filter** (`page_filter.PageFilter`, its keys `page.track.filter`) is
    one Kalman filter. Its state is the sheet's slide and turn since setup, the arm's error as an
    affine across the drawing, and the overhead's own slow error. A wrist
    reading goes in with its own variance, by its markers and rms. An
    overhead reading of the print's markers (the watch's
    `geometry.marks_moved`, weighted by the markers it shares with the
    setup's reference) goes in as the sheet's move plus that error. The
    page's follows are in its model (`ArmExecutor._followed`), so no reading
    carries a slide and no follow starts it again.
    - The overhead's error is a state of its own: on the forearm heart
      (2026-10-23) it read the sheet's 4.4 mm drift over 6 minutes as
      the wrist did, but between follows it erred 0.5-1 mm for tens of
      seconds, which the wrist did not see.
    - An overhead reading past its gate is weighed down by its distance as a
      likely misread, until three in a row read one place (within 1.5 mm):
      then the sheet has moved, and the filter takes it whole. In the move
      test on the forearm18 holder (2026-10-23) the mount slid 22.8 mm. The
      overhead held it every second, but by the wrist's next reading, 15 s
      later, the filter had taken only 74% of it. With the rule it holds the
      slide 3 s after it starts.
    - With `stroke_s` (0.5 s), the wrist is also read while a stroke draws.
      The tick that drives the stroke starts a grab every `stroke_s` on a
      thread of its own, one at a time and none while the readers are
      behind. Each frame carries the joints at the moment it arrived, so the
      reading uses the pose the arm had then. These readings keep no skin, are no
      stroke end, and raise an event only when held; after the lift one line
      says how many there were and how far the tip ran from the plan's line. At a
      stop the wrist reads once every 10-20 s; a stroke now gives one about
      every half second. The next stroke is pre-planned when this one starts,
      so its readings reach the stroke after next unless they move the
      correction past `replan_m`. Each weighs `moving_m` (3 mm) more than a
      stop's reading and joins no stop. On the still holder (2026-10-24) they
      scattered about 0.35 mm an axis, twice a stop's, with no lag along the
      motion; weighed as stops, their 503 put the next stop's reading 0.33 mm
      off, where stops alone put it 0.19 and the field 0.21. Weighed, it is
      0.22, and the bicep's stroke ends keep their gain (2.66 mm, stops alone
      3.52). A pull is judged and taken on a stop's noise, or that bicep's
      move is never taken.
    - A gauge hold weighs `hold_m` (3 mm) less than a stroke's reading. On
      the same heart the holds, read at the skin, showed no gradient across
      the arm, where every hover and lift ~6 mm over it read -0.25 mm per mm.
    - Readings that keep pulling one way are a move too. When the mean jump of
      the last `wrist_n` (4) checked readings lies past `pull_gate` (99.9%)
      of its own spread, the sheet opens to it and the reading is taken whole.
      On the bicep (2026-10-24), a 6.5 mm move read while its stroke drew, on
      one marker a reading, went in a little at a time over some 17 s; now in
      about 2. Over every recorded run it fires three times and moves no
      stroke end: a correction applies only at a stroke's start.
    - With `cameras` (`wrist_lower`, the D405 mounted under the pink wrist
      turned 180°), each hover and lift also reads the print through that
      camera, as the session camera's `wrist_tap.preview` has it open
      (640x360, 15 fps), and the filter takes it with an offset of its own:
      the mount stands where the URDF puts it until its gauge is calibrated
      (`wrist-gauge-<arm>-<role>.json`; until then the session camera's
      calibration, carried by the URDF). The stroke ends, the skin and the
      print's reads stay the session camera's, and a stop's verdict is the
      best of its views', so a print one camera misses on the curve but the
      other reads goes on.
    - Replayed on the 23 tracked runs of 2026-10-20..23
      (`internal/evidence/page_replay.py`), tuned on one half of them, it
      predicted the other half's next 4+ marker reading 0.28 and 0.40 mm
      median off, against the field's 0.35 and 0.51. Where the strokes' ends
      would have inked, 0.24 and 0.40 against 0.37 and 0.52 mm, its p90
      lower too. On 1-3 marker readings the two are even. Its correction moves
      a little more between strokes (median 0.25 against 0.16 mm), and 5% of
      pre-plans against 4% would be planned again. On 38 runs (2026-10-24),
      the strokes read as they drew where they were, it put the next 4+
      marker reading 0.55 mm off on strongly curved skin (the field 0.63) and
      the strokes' ends 0.66 (0.71); curved runs err 2-3 times mild ones
      whichever estimator reads them.
  - **A sheet that moves under a stroke stops it.** A stroke draws with the
    correction its plan took; the readings since (its own while it draws,
    the overhead's marker moves) reach the filter at once. Checked every
    0.5 s, once the filter's correction at the pen stands past
    `page.track.stop_m` (5 mm) from the one the stroke draws with, the stroke
    stops with the machine off and lifts, and goes on from its arc after the
    checks between strokes (the sheet followed, or the page set up again),
    with no decision asked. A silicone skin pushed ~30 mm round a bottle
    mid-stroke (2026-10-24) was read by the overhead and four wrist readings
    within 5 s and drawn on 8 s more, until the e-stop.
  - A pen-down chain keeps one snapshot of the filter. A pre-plan whose correction moved more
    than `replan_m` (1 mm) by its dispatch is planned again: at 0.3 mm the
    wrist's own scatter re-planned 3-6 strokes of each still silicone run,
    4-5 s each.
  - A reading past the filter's `gate` is held. A second confirms a real move
    when the two jumps are one rigid move of the sheet: they keep the
    readings' distance (`confirm_m`), which a misread does not. It must be
    taken at least `apart_m` (3 mm) from the held one: frames from one place
    repeat one misread. The sheet's covariance then opens and both are taken.
    A jump at a hover is measured again from the second view.
  - Each stroke's correction is in its `sent` ledger row and an event.
  - **Strokes that end off the plan are followed, never paused for.** When
    the last `off_plan_count` (2) strokes all ended past `off_plan_m` (1.5 mm)
    from the plan, one way (`track.consistently_off`), the run says which
    strokes and by how much and the page follows the overhead's markers at
    once, however little they moved; the field already holds those ends. When
    the last three ended past 6 mm one way (`OFF_SETUP`) the page is set up
    again (the pen lifted, wrist locate, gauge holds, a fresh field), and the
    stroke's hover stop then measures on it. Nothing between strokes can undo
    a slide within one. On 2026-10-20 the pause this replaced stopped a
    forearm drifting 0.5 mm/min at two ends 2.1 and 1.6 mm off, half of that
    the wrist's own read error there, and the person held still 13 min for a
    decision. A stroke the consistent offset would carry more than 2 mm past
    the print's clear area (`OFF_CLEAR_M`) sets the page up again, a second
    time too: nothing pauses for a key.
    - Only stroke ends count. A hover's reading still carries its pre-planned
      travel's older correction, and a held reading may be a misread.
    - On 2026-10-08 eight strokes ended 1.1-2.3 mm off, one way, after a
      resume while every reading said so; the follow would have come after
      the second. Over the 16 other tracked runs of 10-04 to 10-08 it never
      comes.
    - With `log` nothing is corrected, so nothing follows.

**Page motion.** Before each stroke the executor compares where the overhead
reads the print's markers now with where it read them when the run set the
page up (`geometry.sheet_watch`, `page.track.watch`). A person's forearm does
not stay where it was put, and it is not a Gaussian about one place either:
on three forearm sessions (2026-10-19/20, the overhead at 1 Hz) breathing
moved the print only 0.1-0.3 mm, but it swayed 0.4-0.9 mm (sd) over ~20 s,
drifted 0.5-1.5 mm/min one way for up to 17 min, and stepped 1-4 mm, most
steps staying put. So the page follows the drift and is set up again only for
a move that lasts, with no operator in the loop (owner, 2026-10-20: "I prefer
automatic"):
- **The drift is followed.** Before a stroke descends, its pen lifted, the
  median of the newest 20 s of readings (`follow_s`; through a gap, while the
  arm hides the print, the newest such median stands) moves the page in its own
  plane, its height field and chart with it, once the part the page has not
  followed moves the drawing's corners past 1 mm (`follow_m`) or turns it 1°
  (`_follow_marks`, a page row a resume takes). Below that the filter carries
  it: the overhead's marker moves are its readings (**Stencil tracking**). No
  holds, no motion.
- **A move that lasts is set up again**: the 10 s median (`gate_s`) 7 mm
  (`off_m`) at the drawing from where the page follows the sheet, 2.5 mm
  along the page's normal (`rise_m`), or a marker 3 mm off the others'
  rigid move (`bend_m`, three markers at least), held 10 s (`hold_s`) up to
  the newest reading. The watch reads what the sheet does now: a move the
  person took back is none, and one that stays is off the page until it is
  set up, however long ago it came (a pause, two minutes or more without a
  stroke). The page lags the sheet up to `follow_m`; at 3 mm three of 31
  references on a lively forearm fired on that lag and sway, and the owner
  doubled each of these on 2026-10-20 ("if it's 1 millimeter, make it 2")
  but `rise_m`: a rise is depth, and a hover whose gauge reads no skin has
  no other bound on it, the 3RL's needles reaching 4 mm from a tube 5 mm over
  the field (a rise alone set none of that day's runs up again). A move read but
  not yet held keeps the stroke at its top, the pen lifted and the machine
  off, for up to 15 s; with no reading in 3 s (the lifted pen over the print
  hides it) the arm clears the overhead's view and waits 10 s more there,
  and goes back to ready when the move does not last. A move still unread
  then is set up again.
- **A curved page follows only what a turned print would not put under the
  skin.** The overhead reads a forearm slid across itself and one turned about
  its own axis (the print going round a skin whose shape stays) alike, and a
  follow carries the skin's shape with the page. Before each follow the
  executor lifts the strokes left onto the followed page and onto the skin
  where the holds measured it: when an inking pen would stand 1 mm
  (`roll_m`, the wrist follow's `ambiguity_m` too) or more under that skin, the
  page is set up again instead and the holds measure which (`_rolled`). On
  the left forearm (2026-10-20) a 3 mm follow across it read 0.6 mm; its
  drift was a slide (the markers' depth moved as one), but nothing the
  overhead reads tells a slow turn from it at 2-3 mm. A rehearsal follows and
  says in the follow's message that an inked run would set up there.
- **At once**: a sheet risen or sunk past 4 mm by the newest five readings
  (`WATCH_RISE_M`, depth, so not doubled; a page kept by a set-up stays
  within `RISE_M`, 4 mm, its holds starting 10 mm over it, and a retained
  page within `RETAIN_RISE_M`, 3 mm, its slide within 6 mm),
  or one whose 20 s median has gone past `total_m`/`total_rad` (20 mm, 5.7°)
  of a curved page
  (the skin's shape under the strokes is the holds'; a 15 mm slide put a
  turned print 2 mm off) or past
  `page.replan_translation_m`/`replan_rotation_rad` (50 mm, 0.17 rad) of a
  flat one (owner, 2026-10-19: "big move should set up again").
- Replayed through the stroke loop (`refresh_page`, `_off_plan`,
  `_watch_sheet`, the tracker's field) over the 2026-10-19/20 runs: no set-up
  and no pause on still silicone (four runs). The left forearm's 9 strokes:
  one follow, no set-up; virtual strokes through its 13-minute pause: four
  follows and one set-up, when its drift across the arm reached 3.2 mm
  (0.6 mm under a turned print); the off-plan pause that stopped it does not
  come. A lively right forearm: two set-ups in ten minutes either way it is
  replayed, for turns 2.3-2.7 mm across that would put 0.5-0.6 mm under, or
  one of them and a 3 mm rise held. Asked every 30 s from each of 31-50
  reference times a run, the watch itself set up none of the left forearm's
  and one of the right one's (a rise held; the jump it replaces, none and
  two); inking, the turn check set up 8 of 50 and 10 of 31 within five
  minutes. Injected 5 mm steps were all set up, 23 s (p50) after they came and
  at most 38 s, the next stroke's check included; a 4 mm/min ramp is followed
  and fires only past 10 mm (`test_session_geometry`).
  What remains is the sway within a stroke, which no correction between
  strokes undoes: by the overhead, a stroke goes down p95 0.3-0.8 mm off the
  left forearm. Servoing within a stroke from the wrist camera is not built.
A page whose readings carry no markers (a mock) is set up again by its pose
past `page.replan_*`. A set-up:
- A pen within the standoff first lifts 20 mm along the page's normal, as
  before every mid-draw set-up.
- The arm clears the overhead's view (`clear_view`, to rest), and the pose
  the overhead measures after that places the page, as at a first setup; the
  wrist locates it and the gauge holds measure it. With the arm over the sheet
  its pose once tilted 11.5° and rose 3.8 mm, and the holds travel to 10 mm
  over it. Such a page is never kept on the last holds' height and normal
  (`_kept_page`), whichever check saw the move.
- The stroke then goes on. The check runs first in every stroke's loop,
  before the watch or a hover moves the arm over a sheet that may have risen
  toward it; dispatch makes no check of its own, and every wait between the
  check and dispatch (the machine's pause) goes back to the top of the loop.
  So does a hover stop that lasted 10 s (`HOVER_WATCH_S`: a view cleared for
  the overhead, a rehover) once the readings that came in meanwhile read a
  move, once a stroke: on 2026-10-24 a forearm moved 45 mm under the pen, its
  hover waited 41 s, the readings that came in read the move, and the stroke
  went down on the check from before the hover, 18 mm off (the mid-stroke
  stop lifted it at its first reading).
- **"Moved" is the same markers on the skin, never the overhead's pose.**
  stencild's pose takes its tilt and height from one plane over the visible
  part of the sheet; on a curved skin that plane tilts with whatever part the
  arm leaves in view. On 2026-10-19 a still forearm read "5.4 mm, 13.8°" and
  stopped a draw while its markers' pixels stayed within 1.2 px.
- The reference (`ArmExecutor._mark_reference`, the page row's
  `overhead_marks`) is each marker corner's median over the overhead's
  newest 20 s of readings during setup, kept after the last hold, with the
  markers those lacked from the setup's other readings that read the sheet
  where they do (within `follow_m`, `follow_rad` and `rise_m`,
  `geometry.marks_extended`). Past the arm the overhead reads two or three of
  a curved print's markers, a different two now and then: a forearm's
  reference kept 3 and 6 alone, and after a 45 mm move one reading in ten
  shared two markers with it (2026-10-24); its earlier readings put 16 in it,
  and replayed, the hover reads the move at its third reading. Over the
  corpus's 39 runs the extended reference reads no still sheet differently
  (p50 and p95 slides within 0.2 mm) and keeps readings the narrow one lost
  (one setup's 76 of 122 become 120). The watch
  reads from then on (the node keeps the newest 120 readings, two minutes: a
  move held longer ago still reads off the page). Each later reading carries
  every common marker's corners through the affine that maps that marker's
  setup corners onto its printed square. Only the ray through each corner
  counts (stencild's world is the D555's colour optical frame), so the depth
  it was lifted to drops out. A 2-D rigid fit in print metres gives the slide
  and turn (`geometry.marks_moved`), and how far the farthest marker's centre
  stands off it (the bend).
- Each corner's range along its ray makes it a point; the median corner's
  move along the page's normal (carried into the overhead's frame by the
  arm's registration) is the rise. A slide changes the ranges too (15 mm:
  4-6.5 mm on the overhead's oblique view) but not along the normal. The
  watch reads the slide at the drawing and the rise apart; a resume (3 mm)
  compares hypot(slide, rise). The rise is checked in every hover mode
  before the hover travels. A set-up for a rise keeps no page.
- A reading sharing under two markers with setup is no reading: the watch's
  state holds through it, as with a lost page, and a hover that found no
  print clears the overhead's view for one. Nor is one that puts any marker
  past the print's reach of the page in use's centre (its farthest printed
  corner and 0.2 m, `ArmExecutor.MARKS_REACH_M` in `_marks`), for the
  reference, the watch, a kept page and a resume alike: on a silicone run
  (2026-10-20) the overhead read a second copy of the print eight times in
  304 readings, a marker of each 0.33-0.79 m from the page's centre, while
  the forearm tattoo's own readings put one up to 85 mm from it (50 mm past
  the print, as at first, left a limb's 59-69 mm slide unread). One of the
  copy's went into a setup's reference and its page was set up again for a
  154 mm "move", and three of the newest
  five read a 128 mm rise. Replayed at every reading with references taken
  as the run took them, those two set-ups fire without the filter (a 14.5 mm
  sink, a 159 mm rise) and nothing does with it; it drops none of the
  forearm tattoo's 560 readings. A page whose messages carry no
  markers (a mock) keeps the pose check.
- Page motion stops a draw only where the sheet cannot be measured, or a
  set-up refuses as a first setup does: no fresh overhead pose within
  `page.wait_s` (the view cleared first) and no gauge holds to place the page
  from the last one, holds that place none, a ready move that would dip, a
  field that would extrapolate.
- Replayed over four forearm hover runs' page messages from the draw on (1024
  in all, 2026-10-19), the markers read the still sheets' rise at -0.8..+0.6
  mm, and a still forearm's 93 messages after setup 0.81 mm and 0.17° at most
  where its pose read 5.4 mm and 13.8°. Lifts of 6-55 mm, a 55 mm slide and
  an 11° turn injected on their skin fields read 5.3-6.3 mm for 6 mm, and a
  55 mm slide across the curve 49-53 mm.
- Its limits: a rise also moves the rays (10 mm along the normal reads about
  7 mm of slide), so under 4 mm a kept page moves up to ~3 mm across the
  print, which its holds' shift takes back. A turn on a strongly curved arm
  reads up to about 10% low. The resume check measures from setup, not from
  the slides the page followed. The watch reads the overhead at 1 Hz from
  single frames: tremor faster than 0.5 Hz folds into its 0.15-0.3 mm noise.

A moved page is never adopted as it stands: while the arm hid the print the
tracker once published a false match (2026-09-30: 112 mm and a quarter turn
away, adopted, put the pen 4 mm under the paper and dragged the sheet and its
mat), and a page that really moved needs new holds. A big move is set up
again, the wrist locating and the holds measuring it before a stroke goes
down. Smaller moves are the watch's and the hover's: the tracking jumped ~13 mm
twice while the arm was over the page (2026-09-26).
- The arm hides the page for up to 105 s. While it is hidden, the last
  measured pose stands.
- Motion during a stroke is logged, never a reason to stop: page motion is
  tracked and followed, and contact force is capped mechanically. Every op is planned from the arm's measured
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
  - An `aborted` stroke resumes from its arc position after `continue`,
    lifted and measured again first (§8). The arc is the planned sample
    nearest the measured tip at the latch. `plan_op` joins the path pen-down
    when the tip is within `resume_reapproach_m` of it; otherwise it lifts and
    re-approaches.
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
  A session that knows no page (a stack just restarted) withdraws a pen
  `approach.withdraw_m` along its own axis unless the arm is at rest or the
  cameras' page (measured within `page.max_lost_s`) puts the tip clear of the
  skin. Every landing takes only a tip over the print and 40 mm round it for
  one near the skin: past it the page's plane runs on over no skin, and the
  arm lands as it stands (on 2026-10-21 a tip 0.51 m over the skin and 0.25 m
  beside the print held, its withdraw needing joint_1 past its stop). A
  snag-tripped needle comes out the standoff along the pen, then rises to
  40 mm along the skin's normal like any landing, under the snag guard
  throughout (also on the next land after such a needle was held). A lift
  that cannot be planned goes along the pen, then half as far each way, then
  in 5 mm steps along the pen, each planned from the measured joints, also
  after a step that failed or latched. A tip that no lift took 20 mm over
  the skin (`LAND_HOLD_M`, twice the height the staged move nearly scraped
  from on 2026-10-13; with no page in use, also short of the whole withdraw) holds there
  with the machine off, and the land is refused with why, as an event too:
  the staged move would sweep the needles across the skin. So does an arm
  whose latch the driver keeps (it refuses the unlatch 0.05 rad off the
  latched pose, where a needle the work dragged stands, yet lands from the
  latch). Every automatic land holds again there: the `Land` action (`tatbot
  ros draw`'s after the draw, `ros down`, a set-up's), a wait's after
  `resume_timeout_s`, a tool change's. A person's `tatbot ros decide <arm>
  land` (the `Decide` service, also the station's land) holds too the first
  time and says why; at a hold said to a person it lands from there. The
  needles stay still meanwhile, so the person can draw away, and the arm is
  never stuck.
- **Pipelining.** While a stroke's goal runs, the next stroke is planned in
  the background from that plan's last knot. At dispatch,
  `tatbot_motion.dispatch_drift` compares its first knot with the measured
  joints, and a drifted plan is planned again from the measured joints. A
  hover travel within the drift starts from the measured joints, its offset
  faded out over its first 0.5 s (`tatbot_motion.blend_start`): planned from
  the stroke's untrimmed end, those travels began up to the trim off the arm
  (2 mm, the trim's limit then; past `dispatch_drift`'s 2 mm one is planned
  again), and JTC, which starts a goal from the measured joints, closed the
  gap within its first knot at up to 1.6 rad/s (80 of the 2026-10-20 tattoo's
  536 unstamped goals started over 0.5 mm off the arm, 32 over 1 mm).
  The background plan and the arm's own share one `Kinematics`, whose
  pinocchio data is per thread: on 2026-10-21, with one data, a lift planned
  beside a stroke's pre-plan read that plan's placements, trailed its
  reference 19 and 31 mm, and the landing swept from the skin.

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
gaps, blobs and placement. `analysis.json` holds where the print is against
the page the run used (`border`, for its readers: each view's tile markers,
`fiducials.stencil.fit`, combined over poses as §4.3's locate combines them),
the ink's shift against the plan, measured about the design centre the views
aim at (the camera mount's error plus the arm's), the coverage (the planned
path with ink within 1 mm) and the ink's placement error on the print (each
view's ink shift paired with its own marker fit, so the view's registration
error cancels). The arm ends at rest, holding; `tatbot ros draw` then lands it
(sleep pose, controller idle) unless `--hold`. The mount pose is CAD, not
fitted, so the plan overlay can be several mm off; ink against the print is
limited by image sampling and the markers' registration, and the markers
under the ink may read fewer than on a clean print.


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
- `passes.json`: `stats.passes` for the program and for the drawn lines (each
  a running mean over 5 samples first: FK of the encoders steps the tip
  0.1-0.4 mm back and forth at 20 Hz), at the program's `marks`, and the same
  in one event line; a count that fails is that line alone.

`tatbot ros logs last` finds the newest run on the `ros` node. The Rerun
bridge shows the same data live.

## 5. One arm

The stack drives the pink (right) arm alone (`tatbot_description.names.ARMS`).
The blue (left) arm is the teleop leader on the arm node (cpp/teleop), never on
this stack, and nothing here models it: no description, controller, collision
body or wrist target.

## 6. Ink and dipping

Every tool datasheet declares how it gets ink:

| Supply | Tools | In a program |
|---|---|---|
| `none` | laser prop | No ink ops. |
| `cartridge` | ballpoint | No dips. Another ink is a `tool_change`: the arm lands and the operator swaps the cartridge. |
| `dip` (`ink.mode: real`) | 3RL needle cartridge | A `dip` before its first stroke and whenever the next contact would pass its `mm_per_dip`; the operator's dip scale moves them while it draws (below). A one-pen program for the fitted tool names no cap and never dips (a dry pass). |

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
the palette moved since (2 mm, 1°; twice the fix's repeatability since the
owner doubled every stop, 2026-10-20), or the tool, joint offsets or
registration changed. A palette moved less, past the fix's 1 mm, carries the
ball across by the fixes' own shift (8 shots repeat to 0.1 mm); a tcp moved
under 5 mm (a probe calibration adopted since) carries the ball with it. The touches cannot measure the ball's top, so
its height is the palette camera's (`calib height`): a re-touch keeps the height
of the touch it replaces when the ball moved under 3 mm across with the same tool,
joint model and registration, says so when it cannot, and keeps the replaced file
beside it (`.replaced-<run>`). A dip refuses a touched ball whose height stands
more than 4 mm from the fresh overhead fix's (the fix read it 2.4-2.8 mm over the
camera's); on 2026-10-15 a re-touch's own height, 5.4 mm low, put the 3RL through
a cap's floor.

**The dip** (`tatbot_session.cap`, planned by `tatbot_motion.dip`).
1. The cap from the touched station, its bore and floor from
   `config/palette.yaml`, its ink level from the owner's palette load. The tool
   is its datasheet's body of revolution ending at the loaded TCP.
2. Approach: lift off the page to the standoff, rise until the tool's lowest
   point clears the station's top, turn to the dip's heading, cross to over the
   cap and come down its axis to `hover_m` over the rim. The heading is the
   touches' and no other: there the tip's error across the tool's axis
   cancels, and a turn moves the tool off the cap's axis by twice that error
   times the sine of half the turn (the 3RL's tip read ~10 mm off on
   2026-10-08: ~8 mm at 45°). A cap the arm cannot reach at it is refused
   before anything moves; touch the station at a heading that reaches it.
   The dip's event names its heading.
3. Descend the axis until the tool's end stands `above_ink_m` over the
   declared ink, dwell `dwell_s`, retract to the hover, and return to the page
   standoff it left. The return is judged once the arm settles there, within
   2 mm and 0.010 rad (`dip.RETURN_M`, `RETURN_RAD`): its quintic stop from
   ~118 mm/s rings joint 4 at ~6.5 Hz for ~0.5 s, and two returns read as
   their goals ended (1.09, then 1.66 mm off) ended a forearm tattoo with the
   arm holding (2026-10-20). One still off is planned again once from where
   the arm stands, and then the dip goes on from there: the next stroke plans
   from the measured joints and comes to its own standoff first. A phase
   that ends off its place (an approach or retract more than 4 mm under the
   hover, a descent or dwell more than 1 mm over its dwell height, the tool
   off the axis) withdraws and dips again once with no key; a second one, a
   latch or a tick's fault waits for continue or land.
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

**The dip scale** (the drawing keys, `tatbot_session.charge`). How much ink one dip
picks up varies from dip to dip, so a long drawing can fade before the
program's next dip. The operator answers from the drawing keys while the arm draws:
- `+` (Fn+E) asks for a dip now. A stroke goal stops where it is (the stop a land
  request makes), records `aborted` at the arc it reached (`numpad dip`),
  lifts along the page normal, and the cap is dipped: the same dip as any
  other, at the station touch's heading, with the in-cap marker and the
  withdrawal on an interruption, under the stroke's op id. The stroke then
  goes on from that arc as a resumed one does, through the hover stop when
  stencil tracking runs. Between strokes the dip comes before the next one; a
  pause or latch makes it after `continue`.
- `+` also shortens the cartridge's line per dip by a step and `-` lengthens
  it: a scale on the datasheet's `mm_per_dip` (0.1 steps, x0.2 to x2.0, from
  x1), never the datasheet's figure. Each press is a `dip_scale` ledger row
  and a `dip` event (`+: liner dips every 180 mm (x0.9); dipping now`); the
  scale belongs to the program resource for the run, and a resume starts each
  at its last one.
- Nothing changes, and a `dip` event says why, without a running draw, for a
  cartridge that takes no dips (the ballpoint), or on `+` while a dip runs or
  is already asked. `-` steps whatever runs.
- From the first press for a resource, the executor counts the line drawn
  since its last dip (any dip: the program's, the numpad's, its own) against
  `mm_per_dip` times the scale. A stroke the charge left cannot finish dips
  first, unless the charge is full; a stroke that starts a source path weighs
  the whole path, every section, as the compiler does. A stroke longer than a
  full charge stops where the charge runs out (`charge spent`) and dips there.
  A section the charge dips before (a path longer than the scaled line) is
  not joined pen-down: the section before it lifts at its end as any stroke
  does, and it is not pre-planned across the dip (2026-10-20: dipped from the
  end pen-down, a pre-plan held the dip past 30 s and ended the run, and once
  the lift off the skin could not be planned and ended it, the needle in).
  A compiled replenishing dip the charge still covers is `skipped` with the
  count as its reason; an `initial` one (a new ink) never is.
- So the program's dips are the schedule, and the count moves them only after
  the operator has asked. A numpad dip starts a new charge: the next
  scheduled dip is skipped when the charge still covers the path after it,
  and comes as compiled when it does not. At x1 the count moves no compiled
  dip, since the compiler placed them by the same count (a test checks this
  against `tatbot_ink.schedule`). Before any press a program runs exactly as
  compiled: a bench program draws past `mm_per_dip` on purpose to find where
  a line fades.
- A `+` the goal never answered (it landed or ended first) is dropped, with a
  `dip` event: the resumed goal dips before its first stroke anyway, so the
  ledger's rows never make one again.
- A cartridge that keeps fading early wants its `mm_per_dip_by_ink` measured
  again: the scale serves one run.

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
either sign). Touches cancel the tip's length for drawing, but the cameras do
not: the wrist D405 and the overhead place the skin by their own chains, so a
wrong length reads as a page height and a hold offset (2026-10-22: the 3RL's
tip 12.6 mm shorter than installed, the ballpoint's seat since 09-19). The
palette camera's sweep measures the length with its tilts (below).
`config/workspace.yaml` `joint_offsets_rad`, when present, is applied to the
joint origins by the generated description and the wrist observer; nothing
here writes it. The wrist D405 stands where the URDF puts it: over eight
set-ups' gauge holds its turn on link 6 is 0.12 deg (`gauge turn`, 2026-10-22).
Fits of the camera's seat and of joints 1-4's offsets to its views of fixed
fiducials across far-apart poses moved with the poses pooled (the seat 0.87
deg, joint 1 by degrees) and stand in neither.

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
  `probe.*`) on the ros node. The e-stop relay and its
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
- `MODE_VIEW` holds there too, by one joint move (`ready.view_move`, the
  configured pen-up joint move to the view's IK), not the way up, across and
  down; the executor refuses one that dips more than 10 mm under its lower
  end. The calibration's roof-tag and tile holds (`calib overhead`, `calib
  wristcam`) take it where `Calibration.view_way` passes that move: no dip,
  `rows_check`, and the tool's envelope off every station part and out of a
  column `probe.clearance_m` about the ball up to that far over its top (the
  height the station's way crosses at). Else they take the station's way.
  Over that way each hold turned the tool about its tip as it rose 30 mm,
  crossed and came down, three starts and stops in 11-23 s with joints 1-4
  sliding at 0.15-0.18 rad/s, and joint 3 rang about 8 mrad at 4.5 Hz
  (2026-10-22, calib overhead cf32 and its flight log). The roof-tag holds
  wanted next are visited nearest first (`reach.nearest_first`, by the largest
  joint's turn), the pool's first that succeed still the ones kept: cf32's 12
  turned the largest joint 7.8 rad between them, nearest first 4.3.
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
    (`ready.station_dip`), and a way that folds the arm onto itself (§8).
    On 2026-09-29 a view turned toward the post
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
   stands.
4. S1: the search, opposed side pairs 4 mm under the fix's ball, or 1 mm
   over a lower side height (1.8 mm up the ballpoint's tip), where the stylus
   stands for a fix up to 2 mm off in height, from starts the tool's
   envelope clears for any ball within 12 mm. A tube's reach across a pass
   line is a few millimetres, so when no pass meets the stylus, pairs run
   along lines offset across, nearer first. These contacts go to no fit.
   Edge passes over the found centre bracket the ball's top, and the floor
   rises to it; opposed pairs at the tool's side height measure.
5. S4: further upright yaws (default -60, +60, -30, +30, ... about the
   heading) until three have measured: the list's first that measure, those
   still wanted swept from the end of their span nearer the tool's yaw
   (-60, -30, +60 from S1's 0: 180 deg of wrist turns, not the list's 270;
   joints 3 and 4 rang 14-15 mrad on its 20-27 s swings, 2026-10-22). The
   first searches as S1 did; later
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
checks (the floor, the arm against itself, the camera's post, the arm against the
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
4. Tilts of -12, -6, +6 and +12 deg about the installed tip, about the
   camera's line of sight: the installed tip stays where the park put it, and
   a working tip off it swings across the camera by the tilt times its offset.
   A turn cannot see the tip along joint 6's axis, which is fixed in the mount
   whatever the park, so only these see its length. Every joint moves a
   little; their offsets enter only through the arm's Jacobian over 12 deg
   (~0.5 mm at the tip for half a degree off). They are sent only when the
   park's still leaves a tip 15 mm off the installed one room to swing down.

At each hold the CLI's node takes a full-resolution still on the palette
camera's node and copies it into the run (the ros node holds no key there).
The tip is what is in focus at the ball's distance, so a lamp, a screen or a
person behind the station drops out (on 2026-10-22 a person moving between the
lifted still and the park's defeated the change against it): the pen ends where
its in-focus width falls under 0.9 of the tool's end across (the datasheet
profile's last radius), the tube's end and not a needle standing out of it. Its
change against the lifted still checks it. Each takes the region entering the
search window's top, which stops where the pen's region breaks and never runs
past its end, so two ends more than 4 mm apart are decided by the metal's
contrast, else the lower is taken: that night the dark tube in a dim room never
stood sharp (the region in focus ended 12 mm up, and the ball's height came out
12 mm low), and the change broke at the 3RL's clear grip on half a daylight
sweep's stills. The change grows from its sure part through the weaker, so a
dark tube in a darker room (its change 20-25 over speckle, later that night)
stays one region down to the needle, where the change alone had broken mid-tube
beside the region in focus and the two agreed 6.6 mm high. A tip taken by its
change is the needle's end, up to 1 mm under the tube's. A patch about the ball's top marks the camera's drift; a park whose
ball's top stands 0.5 mm from the lifted still's has the pen on the ball or
before it, under a station prior that is off, and stops the run. `calib height`
writes nothing (exit 3) for a ball further than 8 mm from the fresh overhead
fix's, the limit every reader of the station touch holds it to. One pinhole fit gives the tip in the tool mount, its length too with
the tilts; without them the length is held at the installed one and
`report.md` gives the factor by which a length error reads across: about
(0, 1) on pink. It also gives the tip's height over the ball's top at the park,
the probe run's reference. The run
writes `holds.jsonl`, `stills/`, `candidate.json` and `report.md` under
`~/tatbot-logs/ros-calib/<run>/`; `calib apply` adopts it as above, and moves
`pen_tip_offset_z` and `mechanical_contact_z` by the measured length's change
when the tilts measured it.

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
- `tatbot ros calib overhead --arm <arm> [--adopt]` (`tatbot_calib.overhead`)
  registers it through the wrist D405 instead of the wrist's cube, whose give
  and stickers left the cube's fit tilted ~2 deg: it read the station 8.5 mm
  low against the wrist camera and the measured tool alike (2026-10-22). One
  D555 frame with the arm at rest; the wrist's holds over the palette's roof
  tag (the hand-eye's way-checked views, the tag's pose left free) and its
  straight-down views of the fiducial tile on the holder (its markers' corners
  by bundle adjustment, each read's D405 depth a 1 mm prior); one fit to the
  tag's and the tile's corners in the D555 image and to the wrist's depth on
  the D555's flat patches. A tile marker missed past 2 mm is left out once.
  Each region is left out and predicted by the other, and one alone misses
  the other by ten millimetres, so both regions are measured. A tile the D555
  cannot read (the 12 mm forearm tile: 0 of the 3 markers its views need)
  has the holder's views aimed at the bridge's page. The holder then rests on
  the wrist's depth there, whatever markers the D555's upsampled reads share,
  and stencild's page: the wrist's tile corners where that page puts them in
  the D555, one rigid observation (its sigma spread over the markers; a
  marker missing by over twice the page's median is the wrist's bad read).
  With none of these but 3 views of depth it rests on depth alone. Refitting
  2026-10-22's silicone-tile run without its markers stood 0.26 deg from the
  full fit and its station ball 0.3 mm from it in height, where the cube's
  stood 1.9 deg and 8.2 mm; the station's depth alone 2.4 deg and 3.6 mm. On
  the forearm tile depth alone stood 0.72-0.82 deg from the silicone tile's
  registration and put the print 7.8 mm from where the wrist found it; with
  stencild's page 0.42 deg and 4.6 mm, the silicone tile's own 1.9 mm. Good
  enough for the station and the page's prior, short of the silicone tile.
  `--adopt` installs it with the robot-world golden,
  keeps it on the operator's node for deploys, and restarts stencild, the EE
  tracker and the stack; the station touch is then made again.

## 8. Safety behaviour

These live in `tatbot_hardware` and act on every connected arm within one
tick.

| Condition | Response |
|---|---|
| E-stop pressed, or the last valid heartbeat is too old (checked by frame age, never a latched flag). The source is configuration:<br>- `serial`: a local Pico, 100 ms limit.<br>- `udp`: frames relayed from the palette Pi, 150 ms limit. Accepted only from the relay's address and only with an advancing sequence number.<br>- `none`: no e-stop input, for mock hardware or whenever the operator chooses. The arm's rocker switch cuts power. | Hold latch: command the pose held at the latch instant, zero velocity, motors powered. Incoming commands are ignored until unlatched. |
| Carriage contact over its cap: external effort 20 N above its 800-sample rest baseline, or 2 mm deflection, for 40 consecutive ticks; effort is judged only below 0.3 rad/s | Retract the carriage 32 mm over 0.6 s (blocking, read back), then hold latch |
| Tracking stall (error over 0.35 rad for 2 s with under 0.05 rad of progress) or measured over-velocity | Hold latch |
| Snag guard armed (a riding needle, below): the measured tip more than 1.5 mm from the commanded one, in any direction, for 0.2 s | Hold latch at the measured pose (reason 11) |
| First command more than 0.05 rad from measured, or before 100 ms of non-zero feedback after configure; activation would clamp the measured pose past the same arm/carriage step limits | Refuse before sending the position command |
| Deactivate or error | Hold, never idle. Idle is allowed only after a verified landing: staged pose 4 s, sleep pose 3 s, joints within 0.2 rad and carriage within 0.5 mm. |

**The tattoo machine** is not the driver's. The e-stop freezes the arm and
never cuts its power, and a machine left running over a frozen arm strikes one
spot: on paper a hole, on skin an injury. So the palette Pi's machine relay
cuts it on the e-stop by itself and keeps it off until it is asked off and on
again, the session turns it off at once on any latch, and 0.2 s without the
session's commands turns it off at the Pi (§4.4, §7).

**The snag guard: a needle snagged, or the work carried.** On 2026-10-08 a
3RL riding a silicone skin pinned over a raised holder went through the skin
into the holder. At the stroke's lift the holder kept the needle: the tip
stood 2.3 mm deeper and 2.1 mm aside of its command, and the next travel
dragged the holder about 20 mm before the e-stop. No guard was armed, and the
tip-lag guard counts only a tip held short of the paper.
- **The state.** The measured tip (FK of the measured joints) farther than
  `stack.yaml` `safety.snag.off_m` (1.5 mm) from the commanded tip (FK of the
  same tick's command) for `safety.snag.hold_s` (0.2 s), along the tool either
  way or across it: the work holds the needle while the arm moves on, or pulls
  on it.
- **Armed** (`guard_mode` 3) by the session for a riding pen only, from a
  draw's first stroke goal through every lift and travel after it, the lift
  before a landing included. A dip or rinse, a page setup, the driver's landing
  and the goal's end disarm it.
- **A trip** holds the measured pose (reason 11, `guard_tripped`, `trip_q*`),
  never the command, so the arm stops pulling where the needle is. The session
  turns the machine off at the latch and at once withdraws the needle a
  standoff along the pen, under the guard, its latch released as `continue`
  releases one and refused alike (the e-stop, a controller error, a moved arm,
  another tool). Until 2026-10-20 the needle stayed where it stopped, maybe in
  the skin, until someone decided. The stroke then goes on by itself as after
  any `continue` (§8 E-stop release): measured again, down from the trip's arc.
  Its second trip on that stroke sets the page up again first; its third waits
  for `continue` or `land` with the needle out. A trip in a lift or travel
  withdraws the same way and goes on the first time. A withdraw the work still
  holds trips again, and the arm holds there for a person to free the needle
  and decide. `land` first withdraws the needle a standoff along the pen under
  the guard. If the work still holds it, that lift trips, the arm holds and the
  landing is refused until the needle is freed and `tatbot ros decide land` is
  asked again.
- **The numbers.** The arm tracks its command with the velocity fed forward,
  so the two tips are compared at the same tick: allowing for a lag only widened
  the spread. On firmware 1.11, 111 strokes riding and hovering, with their
  lifts and travel (flat and curved pages, five cartridges, 38 minutes armed),
  kept the tip within 0.98 mm of its command held 0.2 s, and 1.23 mm held
  0.1 s. Lifts on the earlier firmware lagged 1.32 mm held 0.2 s. The holder
  held the needle 2.96 mm off for 0.2 s at the lift's end and 7.5 mm in the
  travel. Replayed on those recordings, the guard trips 0.1 s after that lift
  ends, 1.4 s before the travel, and nowhere else.
- **Not caught.** The needle caught in the holder mid-stroke: the tip sagged
  1.1 mm deeper for about 0.5 s (1.2 mm off held 0.2 s), and the lift lagged
  1.0 mm. Both lie inside drawing's spread. 1.1 mm held 0.3 s would have caught
  the sag, 0.08 mm from the earlier firmware's lifts.

Exclusive ownership: the driver takes the existing arm-driver lock
(`/tmp/tatbot-arm-driver.lock`) before connecting, so it can never run beside
`wxai_teleop`, `arm_recover` or LeRobot.

**The arm against itself** (`tatbot_motion.collision`, in the session, not
the driver). Every goal `ArmIO.execute` sends is checked first; a goal refused
here is a failed goal, not a latch.
- **The model.** The arm is placed in the D555's world by its registration.
  Each link is the convex hull of its URDF collision mesh (both wrist D405s
  among them), and the fitted tool is a chain of capsules along its datasheet
  profile. Distances come from pinocchio on coal.
- **The rule** (2026-09-30: an inspection view folded the pink wrist tag cube
  into its own upper arm). The model pairs the arm's own bodies whose joints
  stand more than two apart; links within two joints overlap as hulls about
  the compact wrist in every pose.
  - The wrist tag cube, which the URDF gives no collision, is a box solved
    from its three tags' calibrated frames (69-70 mm). It is paired with
    every link from link_5 up, not the parts it rides with.
  - A way is refused when two of those bodies come within `collision.self_m`
    (10 mm) and nearer than where it starts, or as near as they stand at
    rest (the landed arm folds onto itself), less 1 mm. A way that never gets
    nearer than it starts is sent, so a folded arm can unfold.
  - The whole way is checked, not only its end: a pose wherever a joint has
    moved 10 mrad (7 mm at full reach), and the last one.
  - Nothing re-plans around a refusal. A draw's move waits for `continue`
    (re-planned the same way from the measured joints) or `land`. A Touch
    goal fails with the reason. The calibration checks the same guard.
  - `Guard.self_gap(arm, q)` gives a single pose's clearance from itself, for
    choosing views.
- **The station's parts** (`Scene.parts_clearance`): the dip and the
  calibration keep the arm's bodies off the palette's parts and the overhead
  camera's post (§6, §7).
- **Not checked:**
  - the separate-process `ArmIO` of `client.py` (inspect, rest, lift, jog);
  - the driver's own retract;
  - a stack whose arm has no registration, which it logs at start.

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

**E-stop release.** The arm holds while the e-stop is pressed. Released, it
lands at once, lifted off the skin first, with no decision asked
(`session.estop_release: land`, owner 2026-10-24: the arm returns to its
rest). With `estop_release: decide`, and after every other latch, the arm
stays held and `tatbot_session` asks the operator to choose `continue` or
`land` (§9). `continue` is refused, leaving `land` as the only choice, when:
- the controller reported an error;
- the measured pose moved more than 0.05 rad or 5 mm from the latched pose;
- the fitted tool changed;
- the release has waited longer than `resume_timeout_s` (default 300 s).
  The arm then announces and lands (it holds when no lift off the skin
  plans, as §4.4's landing says).

The pose check compares the joints measured at the latch with the joints
measured now (0.05 rad on any revolute joint, or 5 mm of tip). The tool check
compares the program's `tool.id` with `config/workspace.yaml` read at the
decision. The same rules apply to a JTC goal that failed without a latch,
because the arm then holds under JTC. With nothing waiting, `decide continue`
on a latched arm runs the unlatch protocol, and `decide land` lands. A `land`
during a running goal cancels it, records `aborted` and lands.

Results of each choice:
- **`continue`** unlatches and re-plans the interrupted op from the measured
  pose. A stroke first lifts along the page's normal to the standoff and is
  measured again (the overhead's markers, which may set the page up again,
  then the hover over where it goes on, a pen-down continuation's too), so a
  person who moved while the arm waited is never met by a needle rejoining
  pen-down where it stopped; then it goes down from its arc position. A stroke
  stopped in its last `resume_reapproach_m` is drawn, and lifts too: the
  continuation after it is measured before it goes down. A machine pause's
  `continue` does the same. A dip restarts.
- **`land`** unlatches and lands to idle. The arm then takes no goal until the
  stack restarts (`tatbot ros up`).

## 9. Interfaces

Custom interfaces (`tatbot_interfaces`); everything else is standard:

| Name | Kind | Contents |
|---|---|---|
| `Draw` | action | goal `{program, arms[], from_op}`; feedback `{arm, op_id, index, total, phase}`; result `{done, skipped, uncertain}` |
| `Touch` | action | goal `{mode: SINGLE\|PAGE\|MOVE\|VIEW, arm, start_pose, direction, max_travel_m, speed_m_s, guard, prior_distance_m}`; result `{tripped, contact_poses, joints, plane, run_dir, message}` |
| `Land` | action | goal `{arms[]}` |
| `Decide` | srv | `{arm, decision: CONTINUE\|LAND\|SKIP\|REDRAW}` → `{accepted, reason}` |
| `Event` | msg | `{stamp, arm, kind, op_id, text}` |
| `SafetyState` | msg | Per arm: `{estop_source, estop_ok, estop_age_s, probe_triggered, latched, latch_reason, guard_tripped, guard_mode, trip_joints, landing, landed, controller_error, machine_off, machine_powered, machine_reply_age_s, rt_period_max_ms}`. `machine_off` is true for a passive tool too; `machine_powered` is the newest `MCS1` reply's `powered` (false before one), `machine_reply_age_s` its age (−1 with no machine switch or before the first reply) |
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
| Corner-cut addressing | the retired `scripts/lib/stroke_material.py` |
| Time laws and approach/descent/lift | the retired `scripts/lib/pen_path.py` |
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
| D555 (overhead) | Inside `stencild`: lifts the tile print's markers onto the depth under them, giving the page pose and its plane over the sheet. | Stays inside stencil tracking, for page pose and presence. Never used for height. |
| D405 (wrist) | The pink arm's D405 is this stack's own (no `visiond` unit opens it; the demo viewer sees it only through the session's wrist tap, §13 Graph). A second D405 under the wrist (2026-10-23) is display only: the tap's preview, never read. Measured before the stack: depth repeats to 0.1 mm but sat 12–13 mm above the paper on the pink arm, so touch overrides its height; a tip-over-page check read 8.6 ± 2.4 mm against 10 mm compiled. | Not needed for flat paper at M1. Later uses: (a) a page-plane prior in the arm frame, so touches start slow descent 4 mm above the page instead of searching 25 mm, about 8 s per touch instead of up to 50 s; (b) the height for the standoff laser, which should not touch the paper; (c) arm-relative stencil registration (M5); (d) curved-surface scans (M6). |

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
  deps/        FetchContent cache: the Trossen SDK v1.11.0 (-DFETCHCONTENT_BASE_DIR)
  calib/       arm-registration-<arm>-current.json copied from the calibration conductor
  inbox/       programs sent by `tatbot ros draw`
  env.sh       from tatbot_bringup/scripts/env.sh.in: ROS, install/setup.bash, TATBOT_REPO, rmw_zenoh
  stack.env    TATBOT_ROS_ARGS=hardware:=... estop_source:=... and TATBOT_ROS_EFFECTIVE (the merged
               hardware/estop/page/arms, shown by `tatbot ros status`), written by `tatbot ros up`
  runtime.json the running tatbot_session's runtime identity, written at its start (the launch's
               `runtime_record`); `tatbot ros evidence` reads it
  .build-pending  packages synced but not yet built (deploy writes it right after rsync)
  rerun-sdk/   the Rerun bridge's own rerun-sdk 0.36.0 and pyarrow 21.0.0 (`pip --target --no-deps`), on its
               PYTHONPATH alone: rerun-sdk asks for numpy 2, and the stack's cv2 needs the apt numpy 1.26
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
4. In `~/tatbot-ros` (not a lane), it installs the Rerun bridge's SDK into
   `rerun-sdk/` unless the pinned versions already import from there. Offline,
   it warns and goes on; the stack then runs without the bridge.
5. It builds only what changed: every package on the first build or with
   `--clean`, else the packages whose files `rsync` changed plus those above
   them (`colcon build --packages-above`), and nothing when no `ros/` file
   changed. The changed packages go into `.build-pending` right after `rsync`,
   and only a successful build clears it, so a `--no-build`, a failed or an
   interrupted deploy is built by the next one. Python is `--symlink-install`ed. Measured end to end
   from the operator node into an empty `~/tatbot-ros`: 56 s for the first deploy
   (SDK fetch and the nine packages, 50 s of colcon), 1.7 s for a one-file
   Python change, 0.8 s when nothing under `ros/` changed. Before a build
   that includes `tatbot_interfaces`, or a clean one, it lands and stops a
   running stack (as `down`) while the old install can still hear it: a new
   message type is a new type hash, which the running stack does not answer
   (§8).
6. `--units` renders and installs the two units with sudo and enables the
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
the bus router) is a different network: `tatbot_bridge` reads stencild's page poses from it,
`tatbot_session`'s wrist tap publishes the wrist D405's frames on it (and a blind set-up's search
queries the D555 owner's capture from it), and `tatbot_rerun` reads them (§13 Graph). A
parallel graph on the same host (a lane, the integration run) takes its own
domain and port, e.g. 51/7451, 52/7452, 55/7455, 60/7460, and its own
workspace `~/tatbot-ros-lanes/<task>/` with the same layout. Stop what you
started by PID: `ros2 run` and `ros2 launch` wrap the real process, so killing
the wrapper's `$!` can leave the node running. Signal the process group or run
the binary directly. Never use `pkill -f`, which matches your own shell.

**Checks.** `scripts/check ros-budget` (fast tier): ≤23,867 lines of code, ≤8
interfaces, ≤3 launch files. `scripts/check ros` builds in
`~/.cache/tatbot-check-ros/<checkout hash>/` and runs the fast session tier
described under Developer tests below. `scripts/check --full ros` adds the
relay tests and `colcon test` of every ROS package; SKIP without `/opt/ros/jazzy`.

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
deflection and retract, `staged_positions` and `motor_parameters` (the named
Trossen gain set) from `config/trossen/tatbot.yaml`
`follower` (the pink arm's qualified carriage; every `TatbotArm` reads that section),
`carriage_position_offset_m` (the golden's joint 6 `position_offset`), `palm_mass_kg`
and `palm_com_m` (`config/workspace.yaml <arm>.payload` when it was identified with
the arm's `tool_id` on its preset, else empty: the preset's palm),
plus `flight_path` (the `ros-stack` run directory). `fake` alone gets `fake_page_z`
after them when `stack.yaml fake.page_z` is set (it is, at `page.fixed.right`'s z,
so `tatbot ros up --hardware fake --page fixed` touches and draws on it): the base-frame z of a simulated
paper plane the fake tip cannot pass, so an offline `Touch` trips its guard. `estop_timeout_s` is
`udp_timeout_s` for `udp` and `serial_timeout_s` otherwise. `real` without an
address in the profile is an error at launch. The e-stop reader and the driver lock are process-wide,
shared by every arm instance. SDK drivers are constructed on a thread that
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
  getters and the post-landing close, the flight log).
- `sdk_backend.cpp`: the SDK calls forked from `trossen_arm_ros` (jazzy) under
  its BSD licence: `configure(wxai_v0, end_effector, ip, clear_error=true, 5 s)`
  with the dated preset and the identified palm, `set_motor_parameters` (firmware
  1.10+ boots with other gains), the carriage `position_offset` when the controller's
  differs from the golden's, `get_robot_output`, `set_all_positions(q, 0, false, qd)` with the JTC velocity
  as feed-forward, `set_all_modes`, and `cleanup()` once landed. Upstream idles
  on deactivate and sends no feed-forward; this fork does neither. It never calls `load_configs_from_file`.
  CMake fetches SDK v1.11.0 (`TATBOT_ARM_SDK`, cached under
  `FETCHCONTENT_BASE_DIR`; a cache checked out at `v1.11.0` never touches the
  network, any other checkout is re-fetched), paired with controller firmware v1.11.1.
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
  the SDK session is closed only after a verified landing, and then at once:
  the aux thread closes it within 100 ms and the control loop reads nothing
  until a wake reconnects (configure, carriage limits and feedback warm-up, as
  at the stack's start), so a landed arm's joint states are its last
  measurement, from before the idle (idled, the pink wrist sags ~1 deg and
  joints 1-2 a few mrad, which a wake then measures). A landed controller sees no traffic, and one power-cycled while
  landed comes back with a wake. On 2026-10-07 a controller kept idle with its
  session open stopped answering 1 s and 11 s after the switch, twice, and took
  no new connection until it was power-cycled. Cleanup or shutdown of an arm
  that has not landed (SIGINT, `tatbot ros down`, a unit restart) leaves the
  session open with the driver lock, holding the last sent pose,
  until the process exits. The controller firmware (1.8.0 and later, including the rig's 1.11.1) idles when its
  connection is lost, so an arm that has not landed can drop when the stack
  process ends, or when the controller drops the connection itself. Idle has no
  position loop: a velocity loop whose integral alone carries the load, so on
  Trossen's idle gains the loaded pink arm falls until that integral catches it
  (the tip 34 mm in 0.22 s from a drawing posture; three mid-draw drops on
  2026-10-08 put the 3RL into the skin). Connect scales the arm joints' idle
  velocity integrals (`config/trossen/tatbot.yaml` follower
  `idle_velocity_ki_scale`, 8 on joints 0-2 and 3 on the wrist, where 8 rang):
  the same fall stops at 5-6 mm, and a real 10 s disconnect moved the tip 4 mm.
  The restarted stack, knowing no page, withdraws a pen away from rest and near the cameras' page
  `approach.withdraw_m` (30 mm) along its axis before landing. `tatbot ros down`, and `tatbot ros up` on a running stack, land
  every arm that has not landed first (`client land`; `--hold` skips it) and
  leave the stack up when a landing fails. A running stack none of whose arms
  answers (no SafetyState: a hung session, or clients whose message types no
  longer match it; rmw_zenoh keys topics by type hash) is refused (exit 3)
  while a driver may hold an arm: real hardware with `/tmp/tatbot-arm-driver.lock`
  held or unreadable (`lslocks`). A stack whose arm never came up holds no lock and
  restarts as before; `--hold`, after seeing the arm at rest, goes on. `tatbot ros
  deploy` lands and stops a running stack (as `down`) before it rebuilds
  tatbot_interfaces or builds clean, while the old install still hears it, and
  refuses (exit 6) while a draw, touch or calibration runs. `--hold` is refused (exit 3) while
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
  latches reason 8 and Land is refused while it lasts, so the session ends a
  run's latched wait at once instead of after `resume_timeout_s`. `tatbot ros
  up` restarts the stack (its configure clears the error with
  `clear_error=true` and holds the arm where it lies) and then lands each arm
  that reported the error, the session withdrawing the pen first. A `tatbot
  ros draw` that fails with its arm reporting the error does the same by
  itself: on 2026-10-06 the controller closed its connection 49 s into a
  stroke and the 3RL's needles stayed in a half-skin until the operator came.
  The firmware idles on a lost connection, so such an arm has sagged by then.
  Never use `tatbot arm recover`
  for this arm: it belongs to the LeRobot arm node, not the ROS node.
- `release()` waits 0.5 s for the aux thread. One stuck in a blocking
  TCP call (the controller gone from the network) is detached and keeps the SDK
  session, so a stop never waits for systemd's SIGKILL; activate refuses after
  0.5 s on the same stuck call.
- Every sent position is clamped 0.1 mm/mrad inside the URDF limits.

**Flight log.** `<flight_path>/<arm>-flight.bin`: one ASCII header line
`tatbot-flight 2 <arm> realtime0=<Unix s at t 0> <record bytes>`, then one packed little-endian record
per write(): `t` (float64, s, steady clock; add `realtime0` for Unix time), then float32 ×7 each of `q`, `qd`,
`effort`, the incoming command `cmd_q`, the sent `sent_q` and `sent_qd` (NaN
when nothing was sent), the controller's `compensation` effort (its gravity and friction model) and
`rotor_temp`, then uint32 `controller_id` and uint64 `controller_us` (the SDK output header: an id counting up
since configure and the controller's µs clock, so a gap shows a lost exchange), float32 `estop_age_s`,
`rt_period_max_ms`, then uint8
`estop_ok`, `latched`, `latch_reason`, `phase` (0 inactive, 1 running, 2
latched, 3 retracting, 4 landing, 5 landed): 256 bytes (version 1, before 2026-10-14, stopped after
`sent_qd`, had no `realtime0` and was 188). `numpy.fromfile` with
the matching dtype after skipping the header reads it; a slow disk drops whole
records, never a tick. `tatbot_bringup/scripts/flight_summary.py RUN_DIR [--arm] [--since S --until S]`
prints the bench M0 numbers from it as one JSON line: the longest period and
its jitter over 2.5 ms, each latch (reason, e-stop state and age, speed before,
time until still), the resting carriage external effort, the signed tip
lag along the commanded tool axis while moving, and `tracking`: the measured tip against the sent one while
the command draws (along the path, across it, along the tool axis), per-joint errors in encoder counts (an arm
joint reports 2π/16471 rad, ~0.15 mm at the tip) and how much they grow after a joint reverses.

Each connect also appends a line to `<flight_path>/<arm>-controller.jsonl`: the driver and controller
versions, the gains in force per mode, the joint characteristics (effort corrections, friction terms, position
offsets) and joint limits, read back from the controller. `config/trossen/tatbot.yaml ros_tuning.<role>`
scales the named set's position-mode gains per arm joint and writes friction terms into the joint
characteristics (persistent) where they differ. They are set at connect only. Switching them while the arm held
(`set_motor_parameters` from the aux thread, a lower velocity integral for travel) moved nothing at five switches,
then the sixth, 28 s after the first, dropped the controller's TCP session: the arm latched on the controller error
and its feedback froze until a restart (gain-switch bench 2026-10-22).

A stack can start on a profile of them instead: `tatbot ros up --tuning NAME` (stack.yaml `tuning`, recorded in
stack.env and the ros-stack run's meta) lays `ros_tuning.profiles.NAME.<role>` over the role's keys.
- **`calibration`** is the velocity integral x1, which `tatbot ros setup` calibrates on.
- **Why:** the role's own integral (x4 on joints 0-2, x2 on 3-5) halved the strokes' error on 2026-10-14. It also
  drives a 4-5 Hz ringing of joints 1-3 while they slide. At x1 a calib wristcam's joint 3 rang 0.72 mrad median at
  0.12-0.25 rad/s against 1.9-2.1, and its fit read 1.07 px against 1.26-1.34 (bench 2026-10-22, wiki experiment
  calibration-move-shake).

GPIO `<arm>_safety`, all doubles:

| State interface | Meaning |
|---|---|
| `estop_source` | 0 none, 1 serial, 2 udp |
| `estop_ok` | 1 released and fresh (always 1 for none) |
| `estop_age_s` | age of the last valid frame; −1 for none or before the first |
| `probe_triggered` | 0/1: the probe's latest frame reads triggered (or a broken wire) |
| `latched` | 1 holding the latched pose, ignoring commands |
| `latch_reason` | 0 none, 1 estop, 2 estop_stale, 3 carriage_contact, 4 stall, 5 over_velocity, 6 guard_tip_lag, 7 guard_probe, 8 controller_error, 9 deactivated, 10 step_refused, 11 guard_snag |
| `guard_mode` | armed guard: 0 none, 1 tip_lag, 2 probe, 3 snag |
| `guard_tripped` | 0/1 until the next accepted unlatch |
| `trip_q0`…`trip_q6` | measured joints at the trip (rad; carriage m); NaN before any |
| `unlatch_ack`, `land_ack` | the last request id the driver processed |
| `landing`, `landed` | the landing sequence runs; verified landed and idle |
| `controller_error` | 0/1 |
| `rt_period_max_ms` | longest read-to-read period over the last second |

| Command interface | Meaning |
|---|---|
| `guard_mode` | level: 0 none, 1 tip_lag, 2 probe, 3 snag |
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
   it go idle with `landed = 1`, its SDK session closed, and it stays idle until
   it is woken. `client wake --arm <arm>` wakes that arm alone: its trajectory
   and safety controllers are deactivated, its hardware component alone is
   cycled inactive and active (the driver reconnects, re-enters control at the
   rest it measures and holds it, as at the stack's start; the first command
   must start there; a controller that does not answer fails the wake and the
   arm stays landed), and the controllers come back, the JTC from the measured
   pose. The next `tatbot ros draw` or `calib
   run` wakes its landed arm this way and waits for it and, when the page comes
   from the stencil tracker, a measured page (a fixed page is never on the bus).
   `tatbot ros up` restarts the whole stack. Its JTC would still report every goal a
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
8. The snag guard (§8): a riding draw writes `guard_mode = 3` and waits for it
   before its first stroke goal, and `0` before a dip, a page setup or the
   landing, and when the goal ends. A trip (reason 11) is unlatched as in step 2
   at once and the needle withdrawn along the pen (§8).

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
- `tatbot_bridge`, `tatbot_session`, and `tatbot_rerun` when `rerun` is set
  (default `stack.yaml` `rerun.enabled: true`) and its SDK is found, the
  workspace's `rerun-sdk/` first. Without it the launch says so and starts the
  rest.
  It is the only node that respawns (after 5 s), and its failure is a
  `display_failed` event, never the stack's;
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
| `/tatbot/goals` | `trajectory_msgs/JointTrajectory`, every goal sent (for the bag); `header.frame_id` is JSON: the op and where each planned Draw phase starts on the goal's time base | tatbot_session |
| `/tatbot/keys/<arm>` | `std_msgs/String`: `up`, `down`, `reset`, `dip`, `fewer_dips`, `enter` per drawing-key press (§4.4, §6) | tatbot_keypad |
| `/tatbot/page` | `Page` | tatbot_bridge |
| tf `world → page/<pattern_id>` | dynamic tf | tatbot_bridge |
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
- It only reads the bus.
- While the bus router is unreachable (the rig link not up at boot, a router
  restart), the bridge logs it once and retries every 2 s.
- Parameters (strings; `''` means the `stack.yaml` value): `config`,
  `page_source`, `arms`, `pattern_id`, `bus_endpoint`.

`ros2 run tatbot_bridge page_watch [--seconds 30] [--arm right] [--json]` is the
same subscription without ROS. It prints each sample's page pose in `world`
and in `<arm>/base_link` (`base_xyz_mm`, `base_rpy_deg`, `base_z_axis`), and
ends with a summary that says whether the sample and the registration share a
camera bundle. On a flat page `base_z_axis` is close to `[0, 0, 1]`.

`tatbot_session`'s wrist tap (`scripts/lib/wrist_tap.py`, `stack.yaml`
`wrist_tap`) republishes the frames each `grab_both` already took, the last
colour frame and raw z16 depth, as frame-set v1 (`frame_set_wire.pack`, JPEG
q90) on `tatbot/vision/d405/<camera>`, the vision.toml camera of the open
device's serial. It sends
with congestion control DROP, low priority and express, from a thread with a
one-slot queue, so the session pays a hand-off. It never grabs a frame of its
own unless `wrist_tap.poll` is set, and its first error turns it off. The
camera is open only inside a Draw goal, or a page Touch goal's set-up, on real
hardware (the locate, the gauge holds, the touches' trip frames, the tracker's
hover and stroke-end frames, a dip's frames), so frames come only then.
`wrist_tap.preview` opens display-only D405s beside it for as long as it is
open, by vision.toml role on the same arm: since 2026-10-23 the lower wrist
camera (`wrist_lower`, urdf `<arm>/realsense_lower_*`, the upper's mount
turned 180° about the wrist, on the same USB hub), at 640x360 and 15 fps.
They open on their own thread, so the session's camera waits for none of
them, and each one's tap polls it at 2 Hz; one that will not open is logged
and left out. The tracker also grabs
`page.track.cameras` (the lower D405) at every hover and lift: that reading's
markers go out on its own camera's tap, and its poll holds 8 s after the grab
(`wrist_tap.HOLD_S`), so the viewer shows the frame they were read in.
Each reading of the print's markers in a frame the tap published (a gauge
hold's, a curved page's locate view's, a tracker reading's) hands the tap the
markers it decoded, each id's corners in the frame's pixels and whether the
reading's fit kept it (`wrist_tap.show_markers`, at the stamp the tap gave
the frame); the tap sends them after the frame as `tatbot.wrist-markers/1` on
`tatbot/vision/d405/<camera>/markers`, and the cockpit draws them on that
frame (docs/vision.md). Display only, like everything below.

`tatbot_rerun` streams into the fleet viewer's standing recording,
`live-cameras`, through `scripts/lib/tatbot_rerun.py` `start()`, loaded by
path (this package has the same name). With neither `connect` nor `output` set
it dials the `rerun-server` node's proxy. Everything is in `world` under
`draw/ros/`:
- `robot/<link>/`: each visual of the ros-stack run's `robot_description.urdf`
  (meshes as `Asset3D`, a GLB without its texture samplers, and cylinders),
  logged once. The wrist tags' textured quads stand 3 mm further out of their
  faces than the URDF puts them: the measured tags lie up to 2.4 mm inside the
  nominal printed carrier, which hid them. Each link is placed from tf
  `world → <link>` on a `rerun.robot_hz` timer (12 Hz), logged again only when
  it moved more than 0.5 mm or 0.5 mrad. The measured wrist pose (trackd,
  through the cockpit) stays RGB axes under `/world/tracking/`;
- `<arm>/tip/<n>`, `<arm>/tcp`: the measured tip trail from tf on a
  `rerun.max_hz` timer (20 Hz, robot_state_publisher's rate; never the 400 Hz
  `/joint_states`), in 200-point chunks, cleared at each run start;
- `strokes/planned`, `done`, `current`: the run's program as a translucent
  ghost, each done or partial stroke as the line the pen measured (the
  ledger's done and aborted rows) in its ink, and the stroke being drawn (Draw
  feedback in `PHASE_DRAW`, cut at `arc_m`). They sit on the page in use: tf
  `world → <arm>/base_link` and the ledger's newest page row, or
  `page/<pattern_id>` before the first one, lifted onto a fitted surface, and
  moved by the newest stroke's stencil-tracking correction, so the ghost
  follows the print and a done line sits on it (each line is kept on the
  print: plus its own stroke's sent `correction_m`);
- `design/`: the same strokes on the design map in the print's millimetres
  (oriented as `drawn.svg`, whose lines stay on the page each stroke used),
  with `frame`, the design's strokes bounded with an 8% margin,
  which the Demo view fits; the tip dot shows only inside it. `stencil` is
  the print the run draws on (the run log's `pattern_id`, its installed
  `stencil.png`) under the strokes, its black a faint violet, cut to the
  frame so the fit stays the design's; none on a fixed page or with no print
  installed;
- `page`: the page's edge, clear centre and a 10 mm grid on its plane or
  fitted surface (grey while lost);
- `level` (static): the arm base's frame with z up, the Demo 3D view's origin,
  so the table reads level however the overhead camera is tilted; `table`: a
  nominal plane at the base's z = 0;
- `palette/`: the palette at the newest station fix carried onto the station
  touch, its caps and, logged once under `palette/mesh/palette_root/`, the
  print in the arm's black (the robot URDF's `black` material) and the palette
  tag's textured quad at its seat. Each cap of the deployed
  `config/palette_load.yaml` shows its state: loaded and available to dip (the
  cap there, a catalog ink and its level, as a dip admits it) in green with its
  ink as a column up to the level recorded at load, empty or no cap in grey,
  unknown in yellow, the next dip's cap in cyan, and a mark in that colour
  20 mm over its rim, which the overhead cloud lying on the caps cannot hide.
  A loaded or next cap's label (`L1 black ≥11.5 mm 85% · 2 dips`: the level is
  a lower bound recorded at load, the dips the run's) has its own row in a
  column beside the palette, laid out for the Demo eye and joined to its mark
  by a leader; the other caps share a row by what they say (`M1 S1 S2 M2 L2
  empty`). A palette load the operator changes shows within a keyframe (60 s);
- `<arm>/wrist_d405` (the camera `page.locate.camera` names; another is
  `<arm>/wrist_d405_<its role less wrist_>`, the lower one `wrist_d405_lower`,
  which the status board's wrist does not follow): each tap's depth from the
  bus at most 1 Hz, kept to
  0.07–0.50 m and at most 320 wide, placed by tf at the frame's stamp (the
  newest tf when that is gone): its `Pinhole` frustum at `depth` and, at
  `cloud/<band>`, an RGB cloud coloured from the tap's JPEG with a tenth of
  a warm amber tint (`ros_scene.wrist_cloud`; the D555's cloud leans blue),
  each point 0.6 of the gap to its neighbours at its range, in rows of at
  most 3,600 points within the cockpit's 60 KiB row;
- `status_board/<item>`: the Demo page's status board
  (`tatbot_contracts.ros_status`): run, op, phase (and the arc in a stroke),
  trim and ink over e-stop, probe, machine, latch and wrist, each a bar in its
  state's colour (green fine, yellow stale or a warning, red STOP or latched,
  grey not applicable or ended) on a fixed 860×150 canvas. The trim is the
  pen trim of the op's cartridge (its newest `pen_trim` row) of the run's
  `motion.yaml` limit, yellow at it, and yellow with the depth a needle
  stroke holds it at (its `sent` row's `trim_cap_m`) while keys pressed past
  that are held back (`trim -4.0 · held -2.65 mm`). An item is logged when it changes.
  `status` is the same as markdown, for the Session tab. The machine item
  reads `SafetyState` `machine_powered` and `machine_reply_age_s`;
- `events` and `<arm>/safety`: text on each event and each safety change.

A run is rebuilt from its files (`tatbot_contracts.ros_scene`: its
`program.json`, its `meta.json` and its ledger, tailed by offset, whose page,
dip and done rows it applies). A bridge started mid-draw
follows the newest `ros-draw` run, and each run start switches it to the new
run. A touch's events leave the scene as it is. Every layer and link is logged
again every 60 s, so a window that joins after the hub dropped a row still
gets it. Under `use_sim_time` nothing is stamped before `/clock` runs.

It is display only. A handler's first error is logged and every error
swallowed. A bridge that cannot start (no SDK, no sink) logs it and idles, so
it is not respawned every 5 s. Rerun 0.36 never redials a hub that went away,
so the bridge watches its sink (`tatbot_rerun.watch_sink`) and exits 0, and
the launch respawns it with every static row. It serves no parameter services.
Its parameters are `config`, `arms`, `max_hz`, `connect` and `output`.

### `stack.yaml`

`tatbot_bringup/config/stack.yaml` has format `tatbot-ros-stack`, version 1. Its keys:
- `arms`, `hardware`, `fake.page_z`, `profile`, `rt.{priority, cpus}`;
- `estop.{source, serial_device, serial_timeout_s 0.10, udp_port 7640, udp_timeout_s 0.15, relay_addr, debounce_frames}`;
- `safety.*`, which the driver takes;
- `page.{source, pattern_id, size_m, clear_m, replan_*, fixed.<arm>, trim.<arm>}`;
  the launch replaces `size_m` and `clear_m` with the installed print's and adds
  `geometry`, the print or `stack.yaml`;
- `registration.<arm>`;
- `machine.{switch none|pi|sim, arm, addr, gpio, port 7642, estop_port 7643, period_s 0.02, timeout_s 0.2, confirm_s 2.0, sim_endpoint}`, the tattoo machine's switch (§4.4, §7); a `sim` switch also tells `sim_endpoint` (`udp://host:port`, a simulator) each change;
- `keypad.{port 7644, arm, from_addr}`, the operator's drawing keys from the station keyboard (§4.4); `port: 0` runs no keypad node; `from_addr` "" = the station's `lan`, resolved by the CLI;
- `touch.{enabled, points}`;
- `session.{resume_timeout_s 300, estop_release land, knot_rate_hz, goal_log false}`; `goal_log` writes every goal sent into the run's `goals.jsonl`;
- `sensing.{wrist auto|real|sim|none, truth, noise_m}`: how the session senses the page from the wrist (below);
- `sim.{endpoint, keepalive_s 0.1, epoch_utc}`, read only with `hardware: sim`;
- `rerun.{enabled true, max_hz 20, robot_hz 12}`: the Rerun bridge (launch `rerun`), its tip-trail timer and
  its robot-link timer (Graph);
- `record` (stack-wide MCAP, launch `record`);
- `wrist_tap.{enabled true, poll false}`: the session's wrist D405 tap for the demo viewer (Graph); `false` is no
  tap at all. `poll` also takes a frame at 2 Hz while `grab_both` takes none, under the camera's lock: a polled
  frame is one the session does not get, so it stays off. `preview.{cameras [wrist_lower], width 640, height 360,
  fps 15}`: the display-only D405s opened beside the session's own (Graph); `[]` is none.

The TCP is not in it (§3). The page trim is `[x, y]` metres in the page
frame, applied as `base_from_page · T(trim)`: the program's centre lands at
`+trim`, so a centre cross drawn with trim `t` whose centre sits at `d` from the
printed centre is corrected by `t − d`. The launch records the file it read as
`config_path` in the effective `stack.yaml`, and `tatbot_session` re-reads
`page.trim` from that file at every goal, so a trim committed there and
deployed applies without a restart.

`sensing.wrist` decides what reads the page from the wrist: `auto` is the D405
on real hardware and nothing on mock and fake; `real` is the D405
(`gauge.WristGauge`: the wrist locate, the gauge holds, the trip frames and
stencil tracking); `none` reads nothing; `sim` is a simulator's gauge on the
truth surface `sensing.truth` (`tatbot_session.sim`, private). A curved page
(`page.surface.mode fitted`) needs a wrist that is not `none`. Real hardware
takes only `auto` or `real`: the session refuses `sim` or `none` there when it
starts.

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
  `ok` false means re-plan from `q_measured`. `blend_start(traj, q_from, kin, blend_s=0.5)` starts it from
  `q_from` instead, the offset faded out as a quintic; `hop_via(p_from, up_from, p_to, up_to, rise_m)` gives
  `plan_travel` the via points of one move up and over between two gauge holds. A goal sent from its start
  sends no knot before `ports.START_BRIDGE_S` (50 ms) but its last: JTC splines from the measured state to the
  first knot sent, and an arm one encoder count off the plan's seed crossed it in the 10 ms to the first
  100 Hz knot at 0.06 rad/s (0.4 rad/s on joint 3 at worst), on 20-40% of calibration goals (2026-10-22).

How `tatbot_motion` plans (the choreography of §4.4, ported from the retired `scripts/lib/pen_path.py`):
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
- A gauge hold after another (setup or check) is one pen-up travel up and over onto it
  (`tatbot_motion.hop_via`, `ArmExecutor._hold_travel`): a cubic Bézier leaving and arriving along the
  page's normal, its inner control points `touch.lift_m` out, 0.75 of that over the chord midway, under the
  same pen-up budgets as any travel. A lift, a ~1 s stop at its top and a travel down were three goals and
  two more of the arm's stick-slip starts and stops (the tip 0.6-1.3 mm off its command at two to three
  times its speed): 8.4-11.5 s a hold on the 2026-10-20 tattoo, replayed as one 5.0-6.7 s, ~30 s and eight goals less
  a setup. It hops from the stand the last hold's travel commanded, the pen within 2 mm of it (`HOP_AT_M`:
  after half the tattoo's holds the arm had settled a joint an encoder count, up to 0.31 mm at the tip), and a
  hop a latch stopped goes on up and over from where it stopped. A hop the planner refuses, that would pass
  nearer the page than either end, whose pen turns, or from a pen moved off that stand lifts off the hold
  first and travels as before, its event saying why (`touch N: no hop from touch M (...): lifted off it
  first`); the last hold lifts.
- A tip below the standoff lifts straight off the page before any travel. With `pen_down_at_start` a
  tip within `dispatch_drift.start_tolerance_m` of the start (`resume_reapproach_m` when resuming from
  `from_arc_m`) joins the path pen-down; farther, it lifts and re-approaches.
- A plan over a joint-speed cap (0.375 rad/s pen down, 1.0 pen up) is slowed, leg by leg, and planned
  again; it never ships over the cap. `pause` lifts to the standoff or holds; a `dip` is planned by `tatbot_motion.dip` and run by `tatbot_session.cap` (§6).
- Touch: settle 0.5 s at the start, 10 mm/s to 4 mm short of the prior (`PHASE_DESCEND`), then 0.2 mm/s
  (`touch.slow_m_s`, `PHASE_TOUCH`) up to `max_travel_m`, with no wiggle (`touch.wiggle_amplitude_m` is 0: a
  wiggling tip dragged the pink arm's compliant mat); without a prior the whole 25 mm search is slow.
- `tatbot_bridge.page.world_from_page(world_from_target)`.
- `tatbot_session.ports`: what `ArmExecutor(node, io, kin, motion, stack, repo, *, clock=None, sensing=None,
  planner=None)` reaches outside itself. `ArmPort` (`ArmIO`'s members), `NodePort`, `Clock{now, utc, sleep,
  wait}` and `Sensing{wrist, ready(arm), gauge(q, tcp, prior, *, curved, count) -> GaugeRead}`; the
  `Event.KIND_*` and `Phase.PHASE_*` values of the messages (a test pins them); `WallClock`, the rig's and
  fake hardware's clock; `goal_points`, `goal_tag` and `way_clear`. No ROS and no OpenCV: the executor imports
  without either. Every deadline, sleep and stamp in the executor, `arm.py`, `node.py`, `track.py` and
  `ledger.py` is the injected clock's (a test refuses `time.monotonic`, `time.sleep` and `time.time` there);
  with `hardware: sim` the node's clock is `arm.RosClock`, the simulator's `/clock` time.
- `tatbot_session.drawrun`: the node's draw run, step by step: `check_program`, `build_executor`,
  `wire_guard`, `draw` (the outcome), `write_page_and_drawn` and `fixed_page`.
- The simulator (`python/tatbot_sim2`, `tatbot sim2 run`) drives this executor two ways and compares them: in
  process through `tatbot_session.sim` (private), and as this whole stack in a sim lane with `hardware: sim`.

### Developer tests

Before deploying a session change, run from the checkout:

```bash
nice -n 19 scripts/check ros
```

This builds the ROS workspace in its external colcon cache, then runs
`tatbot_session/test` with `-m "not slow"`. With a warm build, the session
tests target less than three minutes on the ROS node. Other packages' tests and
extended session scenarios, including the relay tests, run with
`scripts/check --full ros`; bare `scripts/check --full` includes that same
complete ROS tier among all repository jobs. A cold
build adds compilation time. Missing Jazzy is a named SKIP.

In an already sourced **scratch** workspace, the equivalent session commands are:

```bash
python3 -m pytest -q ros/tatbot_session/test --durations=0
python3 -m pytest -q ros/tatbot_session/test -m "" --durations=0  # every session test
python3 -m pytest -q ros/tatbot_session/test -m slow --durations=0
```

The default includes one shared mock graph whose square draw takes the real
Draw action through wrist locate, five gauge holds, a fitted cylindrical
surface and its check holds, a completed stroke, MCAP/ledger/SVG output and
runtime evidence. It also tests refusal of duplicate research occupancy and
unsupported pause operations; executor unit tests retain the motion, setup,
surface, cancellation and refusal cases. The fixture supplies deterministic
decoded wrist observations and a noiseless cylinder mesh read by the production
`TruthGauge`: camera capture and marker decoding have separate image tests. Locate still solves and validates its
known top-down view over a synthetic raised skin through the real IK and travel planner, using one view
candidate instead of the full grid. The broader chooser is tested in
`test_session_inspect`.

`slow` retains fake-driver e-stop/continue/land, cancellation during a touch,
trim continuity, landing/waking and cartridge-swap lifecycle scenarios,
crash/resume, complete truth-surface simulations and live-controller clock
parity. Bringup's mock/fake whole-launch checks also carry that marker and
run in the full ROS tier. Each lifecycle scenario starts a fresh controller;
the paired draw checks wake recovery after its second side lands, using that
same controller and its carried state.

Coverage stays at its owning boundary: shared quaternion conversions are tested
in `tatbot_description`, page correction and trim in session geometry, and
charge boundaries in the pure charge tests. Compiler schedules pass through the
actual executor to check dip agreement and changed scales. The in-process
truth-gauge setup owns its mesh/surface checks; its sensor test loads the
configured factory. The mock Draw owns ROS setup integration, so another
fake-driver graph need not repeat the cylinder fit. The skin-watch trim sweep
and message cases own the keys matrix; a separate case covers surface rise.

Unit fixtures advance camera settling and marker-watch deadlines through their
injected test clock. The delayed-acceptance test advances the controller's
five-second deadline and cancellation grace without sleeping. Sensor-only and
pass-count cases prepare only the data they read. Batch repeatability, goal logs
and episode/controller agreement belong to the actual simulator batch tests
(`scripts/check sim2-core`); the session suite keeps the truth-surface setup and
pass-count output cases. Posture search uses one fixture to exercise each
independent refusal and selection of the next clear view. The curved-marker
matrix covers every pair of radius, direction and marker-set values without
repeating their full Cartesian product. Clock behavior is checked through
execution and deterministic batch runs; filesystem timestamp age checks may
use the wall clock.

All acceleration lives in `test/`: temporary motion YAML (zero settling and
faster travel on mock hardware), fewer synthetic camera frames and
a test-only node entry point that refuses physical hardware. Prepared programs
bind that same motion configuration, so the research digest refusal remains
live. Fake-driver contacts keep the original quasistatic speed and settling:
their trip-confirmation and plane-accuracy assertions depend on that model.
Production tuning and interlocks are unchanged. Graphs share a router
and controller per module where state permits; each has a private domain,
loopback router with multicast disabled, and scratch logs. Teardown stops its
clients before its router. No parallel graph workers are needed. An unmarked
session test over 60 seconds fails the job.

### Runs

`tatbot_session` creates one run per `Draw` goal (`ros-draw`) or `Touch` goal
(`ros-touch`) through `$TATBOT_REPO/scripts/lib/tatbot_runlog.py`
(`init(workflow, attach_logging=False)`, `finalize`). It lives at
`~/tatbot-logs/ros-draw/<run-id>/` and holds:
- `meta.json`, with the deployed `revision` (`repo/REVISION`: sha and dirty
  flag), the program sha, tool, TCP, trim, page pattern id, registration sha,
  e-stop source and hardware (the `ros-stack` run records the same revision);
- `console.log`, `run.jsonl` (each attempt, a resume included, logs a `stack_run` row naming the ros-stack run whose flight log and controller read-back it ran under);
- `program.json`;
- `ledger.jsonl` (the format is in `tatbot_session/ledger.py`: `sent`, `done`,
  `aborted` with `arc_m`, `skipped`, `decision`, `pen_trim`, `dip_scale`; each row's
  `t` in Unix seconds and `t_clock` on the session's clock);
- `goals.jsonl` with `session.goal_log`: every goal as sent (op, tag, the clock's
  time at sending and the goal's start, its knots);
- `page.json`, with the page poses used and the touches;
- `bag/`, an MCAP (`ros2 bag record`, started by the session and stopped with
  SIGINT) of `/joint_states /tf /tf_static /tatbot/events /tatbot/safety /tatbot/page
  /tatbot/goals` and, per arm, `/<arm>_arm_controller/controller_state`, the
  `follow_joint_trajectory` feedback and status, and `/<arm>_safety_controller/gpio_states`.
  A resumed run (`Draw.run_id`) appends to its ledger and records `bag-2/`, `bag-3/` and so on;
- `drawn.svg`: the FK of the measured joints over each pen-down stretch at
  20 Hz, in page millimetres, over the planned strokes.
- `passes.json`: how often the pen passed over each spot, planned and drawn (§4.5).

`console.log` holds every event line. `tatbot_session` reads one parameter,
`stack`: the effective `stack.yaml` that `stack.launch.py` writes into the
`ros-stack` run. The repo is `$TATBOT_REPO`.

### CLI

| Verb | Tier | Does |
|---|---|---|
| `tatbot ros deploy [--no-build] [--units] [--clean]` | remote | rsync + incremental colcon on the `ros` node, and the Rerun bridge's isolated SDK in `rerun-sdk/` (§12); `--units` installs the systemd units; a build of tatbot_interfaces, or a clean one, lands and stops a running stack first (exit 6 while a draw, touch or calibration runs) |
| `tatbot ros up [--hardware] [--estop] [--relay-addr] [--page] [--pattern ID] [--arms] [--no-touch] [--rerun] [--hold]` | remote | `--pattern` names the print to draw on by its pattern id or the hex digits its sheet prints, resolved against the prints installed on the ros node (refused when none or several match); lands a running stack's arms (as `down`), writes `stack.env`, restarts the router and the stack, prints the effective hardware, e-stop, page and arms (flags over `stack.yaml`; flags not given fall back to `stack.yaml`, not to the last `up`), and a note when a real arm runs with `estop none`; refused (exit 3) when the running stack sends no SafetyState while a driver may hold an arm (§8); a lane's stack runs no Rerun bridge unless `--rerun` |
| `tatbot ros down [--router] [--hold]` | remote | prints each arm's landed/latched state, lands every arm that has not landed and reports no controller error (`client land`; hardware fake or real), then stops the stack; a failed landing leaves it up; refused (exit 3) when the running stack sends no SafetyState while a driver may hold an arm (§8); `--hold` stops without landing |
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
