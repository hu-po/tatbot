# Release Notes

## 2026-10-06 — v0.10.0

### ROS 2 Drawing Stack

`ros/` is now public and is the only drawing path: a ROS 2 Jazzy stack on
`rmw_zenoh` that drives the right arm with a ballpoint. Its C++ ros2_control
interface is the stack's only Trossen SDK client and owns the e-stop reader,
hold latch, carriage trip-retract and stall guards; only named unsafe states
like these block motion. A Python session plans each op from the URDF with
pinocchio for 400 Hz joint trajectory controllers, and every run keeps a
resumable stroke ledger, an MCAP bag and an SVG of the executed tip path.

`tatbot ros deploy` builds the stack on the ROS node and `tatbot ros up` starts
it for a named stencil print. `tatbot ros compile` places acquired DrawingBotV3
artwork or an Inkmap design on that print; `tatbot ros draw` locates the page
from the overhead D555 and wrist D405, measures its height by guarded touches
or a wrist-camera gauge, draws, and images the result against the plan.
Stencil tracking can correct strokes for a moved sheet, a numpad trims pen
depth mid-stroke, and a multi-cartridge program lands the arm for each swap.
The stack also registers arms to the overhead camera, calibrates tool tips on
the palette's probe, checks goals against a two-arm collision model, and can
dip a needle cartridge at the palette caps.

### Retired Paths

The earlier drawing path is deleted: `tatbot draw` and the session layer that
grew behind it (`tatbot session`, `tatbot dip`, a Python orchestrator, a Rust
daemon and state machine). So are the PoE scene cameras' tools, surface
reconstruction and five-camera calibration sweep, the contact-microphone
audio stack, the rig-screen viewer kiosk and LeRobot teleop recording
(`tatbot record`).

### Rig, Artwork and Tools

The left arm runs LeRobot policies; the ROS stack never connects to it.
Each arm has one wrist D405; one overhead D555 belongs to the ROS node. Coded
flower-of-life stencil frames encode the print ID and position in the border.
The printed v11 palette (`urdf/palette.urdf`) holds six caps, a probe, a
camera, an AprilTag and the e-stop; its Raspberry Pi relays that button to both
arms as sequenced UDP frames, and a press or 150 ms of silence holds them.

`tatbot drawingbot generate` acquires DrawingBotV3 drawings as frozen metric
paths at physical size; Inkmap defaults to that artwork, and `tatbot research`
runs paired studies of it through `tatbot ros draw`.

### Release Process

Public releases are now cut by one command in the development repository. It
builds the export of a committed revision, scans it for private identifiers,
checks this repository for edits not yet ported back, shows what changes, and
pushes one export commit and its tag. Tests run in this repository's CI after
the push. This replaces the approval-gated publication described on 2026-09-06.

## 2026-09-06

### Source-Preserving Public Releases

Public releases now take the README, shared assets, ignore rules and release
history directly from the development checkout. Export verification rejects
unexpected file changes, and public contributions can coexist with later source
edits without freezing whole files. Weekly preparation produces the README,
source-derived release notes and disclosure results for review; publication
remains a separate approval-gated action.

### Shared Tattoo Artwork Across Inkmap and Simulation

Inkmap and offline simulation now draw from one shared collection of simple tattoo designs, so a design previewed in the web workspace is the same artwork the simulator renders and evaluates. Artwork previews are pinned to the collection sources, and the design gallery keeps keyboard focus while browsing.

### Self-Contained Public Export Validation

Validating a public simulator export no longer depends on deployment state: checks run against isolated fixtures with separately downloaded assets, so a fresh machine reproduces the same result without disturbing existing caches.

Commits: 4fe1ed1 9f17e90 803c81b 36c1cb5 b1a231f 7c48f93 b30bf36

## 2026-09-06

### Passive E-stop Health Reporting

`tatbot status` now reads heartbeat health from the active monitor without
opening a second serial reader. Missing or stale snapshots report unknown;
pressed and fault states report failed. Snapshot writes run separately from
heartbeat consumption, and telemetry remains outside the motion safety path.

### Public Export Approval and Simulation Metadata

Public export workflows bind publication directly to release approval checks,
preventing unverified asset variants from being uploaded. Simulation run
metadata generation now accepts schema-2 scenario specifications, resolving
serialization failures during scenario execution.

Commits: 67dec7b dbc88f5 365ed1b 4d39bc8 d431fee 951f543 5876dbd

## 2026-09-05

### Inkmap and Simulation Stability

Release approvals for Inkmap publishing are now bound to the release packet itself instead of a separate gates file, which prevents approvals from being reused incorrectly across releases. Simulation variant generation now gates perturbations by reach validity and records stencil visibility, and Inkmap pilot runs now derive their counts from the provided spec while carrying occlusion metadata per episode. These changes reduce mismatched approvals and improve the consistency of simulation replay and review outputs.

### Internal maintenance

Several internal automation commits were merged in this batch with no externally observable behavior changes.

Commits: 6f080a2 56f1926 03bcf95 48948fd 3c02a73 06391ed

## 2026-09-05

### E-Stop Safety Reliability

Emergency stop heartbeat timing is now preserved during extended uptime, preventing unexpected service restarts caused by filesystem events. Flooded test samples are rejected to maintain accurate heartbeat continuity across safety boundaries.

### 3D Inkmap Editor and Surface Mapping

The 3D mapping workspace exposes surface mapping quality metrics and preserves tool specifications when exporting simulation bundles. Target rendering retains preview styling and label metadata across synthetic perception pipelines and audited export packages.

### CLI Producer Diagnostics & Simulation Variants

CLI commands validate background producer health to surface installation failures directly. Simulation workflows support episode variants generated from Inkmap specifications, and obsolete command wrappers have been removed.

Commits: 53a6fed 350c627 7ac3942 685c550 b4322dc d0d4116 c1b8983 8313352 07fe257 f62b3a3 5c48ca9 2ed2c9f 9d4fd4e 617e8ba f25ef6a 06495a3 0160532 e9549f0 a8270b2 c99b1a6 46ff467 88c2111

## 2026-09-05

### CLI Invocation Keeps What You Typed

Commands now pass arguments through unchanged and report failures as structured errors, so a rejected run tells you what went wrong instead of swallowing the details. Generated examples quote multi-word arguments, so copying a command from the help output runs the same command you saw.

### Declared Commands, Compatible Aliases

Each command now declares its arguments, effects, and compatibility, backed by one shared validation contract instead of duplicated per-command checks. Older call shapes keep working as translating aliases, and status output distinguishes a scope that could not be collected from a lock held by someone else. Qualification checks catch stale documentation links and regressions in the selected output mode.

### Tattoo Artwork Survives Preview and Export

Previews now render the artwork faithfully rather than redrawing it with placeholder styling, and portable bundles carry the artwork together with its exact material intent so an export looks like the design it came from. Filled-coverage planning no longer strokes tessellation edges into the design and protects negative space inside filled regions.

### Saved Projects and Gated Publishing

The web workspace now persists projects across sessions, recovering unsaved edits and restoring the mobile layout on return. Publishing is gated by release approvals so direct uploads are refused with a clear message, and the end-to-end frame-rate check now measures the application itself rather than the test runner.

Commits: 841f83d ff9a985 10edebf 7d43b42 93c4fac 9891012 9b25929 7c57d36 745b192 d6358ac d8633fd 033d719 8afcc84 67b7881 d38a47c 0d36cb0 c839ad1

## 2026-09-05

### CLI Contracts and Command Discovery

The CLI preserves command-owned arguments and routing tokens, declares structured
output support, and separates local status from bounded fleet collection. Shared
body validation and draw declarations remove duplicated contracts while retaining
runtime verification, executor limits, and launcher ownership.

Simulator operations now have explicit commands: `sim sample --count`, `sim reach`,
`sim viewer`, `sim samples`, `sim eval dataset`, and `sim eval policy`. HTML viewer
generation needs only system Python. Existing mode-flag forms remain translating
aliases and warn once on stderr. Draw report destinations use `--output FILE`,
with the old leaf-local `--json FILE` retained as an alias. Removal requires at
least 30 days and two reviewed releases after September 5, plus consumer review
and a release note. Root help groups operations by purpose; `tatbot completion
bash` generates opt-in completion without editing shell startup files.

Qualification checks validate JSON plans, alias equivalence, local documentation
links and option metadata. Missing option values cannot be filled by removing a
global flag. Schema Markdown output rejects global JSON; schema freshness checks
support JSON. Remote Rerun options are inserted before the backend separator,
preserving the backend argument stream.

### Tattoo Preview Keeps the Body Visible
Placing a design that does not fit the local skin curvature no longer blanks the whole body preview. Fit is now judged by chart gates that measure limb girth and surface flattenability, so an oversized or badly curved placement is refused on its own terms while the body stays on screen.

### Every Mapped Skin Site Is a Placement Anchor
All mapped skin sites now accept tattoo placement, and the restricted six-limb initial domain is retired. More of the body is directly placeable without working around the old boundary.

Commits: 838822b 8f085ac ec9ac9b 246de97 43050c6 fb4ed7f 875c86a

## 2026-09-04

### Calibration Probe Landing & Inkmap Catalog Fixes
Calibration sweeps now output landing sentinels to the designated teleoperation controller path, ensuring reliable landing detection upon sweep completion. The 3D tattoo mapping web interface correctly processes feedback responses and resolves catalog digest verification errors during asset loading.

Commits: 6406230 902b62c c73ace7 87a4c3c fa6cef7 93d90f1

## 2026-09-04

### Calibration Sweep Shutdown and Interruption
When a calibration sweep ends, both arms now land themselves at the sleep pose instead of going idle where they stopped, so no manual guidance home is needed. Interrupting the sweep always releases teleop control cleanly, and the contact-cap rest baseline is now a 2-second measurement taken while tracking is active, making it robust to slow drift.

### Arm Recovery Takeover Guard
Arm recovery now validates the measured pose against the controller's live joint limits before taking over. Sessions that previously faulted on handover when a joint sat outside its limits are now brought into range first.

### Offline Body-Model Pipeline
The offline body-model pipeline is now cut over to a single unified representation with aligned atlas bindings, hash-verified assets, and replay manifests pinned to stable outputs. Audit and observability records are bound to their sources, while deployment, public export, powered evaluation, and human-study gates remain closed and pending.

Commits: dcd98ac d971f03 5f536dc 5761c39 c828f21 b858247 3e41779 6c64e0b 9a95ce9 9bf8858 9eb9a4d 35d8875 c237466 72bf99f

## 2026-09-04

### Guided Contact Calibration Sweep
The calibration sweep incorporates a seven-state guided touch phase that fits per-session surface contact models and waits for camera streams to initialize before proceeding. It pauses for confirmation during pen changes and releases contact-trip holds smoothly without raising stop signals.

### Teleoperation Baseline & Shutdown Safety
Contact-cap rest baselines during teleoperation are now measured dynamically while active tracking is underway rather than before the session starts. Signal handling during calibration stop handovers ignores secondary interrupt signals, guaranteeing teleoperation control is always cleanly released.

Commits: 43a195b dc29bf9 88ccb41 e66fec2 b541def a9ebc34 777ce20

## 2026-09-04

### Automatic Launch ID Ledgering
Launch IDs now replace the nonce gate with an automatic, audited system that ledgers every autonomous-motion verb on the arm node with full provenance (PID chain, SSH origin). The optional `--tag` flag adds a human-readable label to each run for reference.

### Drawing Surface Contact & Approach
The lag touch-off mechanism is now the default contact finder for drawing operations, configured with a 10 mm no-scan lift height and 3 orbital approach poses executed at 50 mm/s for consistent surface engagement.

### Guided Contact-Mic Calibration
Contact-microphone calibration replaces the audio-only approach and distinguishes electrical hum (line noise) from actual contact events. The calibration system now provides guided sessions in two forms to support various operational contexts and sensor fusion with uncertainty estimates.

Commits: 98d94a7 04391da c7d1f97 edcd31d 27d55ec 21a4712 2e31d52 7d14e44

### Draw Stack Centralization & Execution Safety
Drawing constants (caps, envelope, period, and dip speeds) are now unified under `config/draw_constants.json`, and the executor refuses runs with mismatched configuration hashes. Interactive drawing launches validate flags prior to consuming single-use nonces, restricting `--dips-only` execution strictly to dedicated dip commands. Stall-guard pauses reset servo offsets directly rather than ramping, ensuring predictable touch-off and back-off behavior.

### Human-Representation Implementation Specifications
Documentation for human-representation contracts has been updated to require
complete replacement of the retired nominal-body model across human-body
pipelines and authorize continuous implementation phases.

Commits: 3fef19a 47ff640 2048e56 bbcd3bd 99066ad cd07add 56bfe88 5363804 89f8d5e f3e0249

### Sole MHR/SOMA Body and Offline Program Pipeline

Implementation source: `6c64e0ba5a4b1a67a1430a5b0560046099cb1c60`.

Tatbot now has one nominal human-representation path: a pinned MHR identity
transferred to invariant SOMA mid topology. Inkmap, InkLang, named poses,
atlas v2, PlacementFile v5, TattooScenario v2, browser preview, and simulation
use the same model, identity, topology, rest-surface, and asset bindings. The
old body registry, selector, assets, tools, schemas, fixtures, and readers were
removed; invalid intrinsic charts fail before editor state is mutated.

The offline stack now implements strict `TattooProgram/1`,
`SurfacePlacement/1`, `InkProgram/1`, `SurfaceRegistration/1`, and
`ExecutionProgram/1` contracts. A fixed-patch Torch path supports bounded
research gradients, while deterministic hardening and the exact compiler alone
write current dense samples and ledger evidence from observed surface cells.
Synthetic registration, training, mechanics, and anatomy fixtures preserve
uncertainty and refusal denominators. No joint model, compliant mechanics,
anatomy pack, deployment, public export, powered evaluation, or human study is
admitted by this change; those gates remain separate and disabled.

### Human-Representation Release Gates

Four independent, signed-evidence gates now separate Inkmap deployment, public
release, powered evaluation, and any human-facing study. A strict checked-in
gate document keeps all external actions false and supplies a rollback that
disables the consumer or deploys only a known-good MHR/SOMA revision.

### Human-Representation Safety & Refusal Contracts
Human-representation workflows now strictly enforce refusal contracts and audit evidence retention for out-of-scope requests. Contract audit gaps have been closed to guarantee deterministic failure states and auditability across qualification and safety verification boundaries.

Commits: fa7e838 a5c7f52 1eb8b0e 2a54308 ee67877 6984303 339e0e7 8352bc9 9c8cb5f

## 2026-09-04

### Typed Placement Grounding Across Inkmap and Simulation
InkLang is now a typed, placement-only contract shared by Inkmap and offline simulation: prompts resolve against a named body’s versioned atlas to face/barycentric anchors, while ambiguous interactive requests wait for an explicit choice and batch jobs record a deterministic policy and seed. `tatbot sim resolve` preserves the request, chosen placement, actual resolved site, and resolver provenance in replayable scenarios, including the distance actually achieved by relative placements.

### More Reliable Posed-Body Simulation
Scenario sampling now projects contact onto exposed live-mesh faces, excludes clearance proxies from the contact surface, and retries incompatible body/pose/site pairings while retaining rejection reasons. Coverage promises and full-trajectory IK are checked before data is accepted; unplannable or idle outcomes are recorded instead of becoming misleading demonstrations, the GPU fixed-base root is re-pinned, and body-tattoo simulation declares its nominal tool without waiting on workstation calibration.

### Exact-Design Simulation Evaluation
Exact-design simulation is now a first-class offline workflow: `--judge` retains intended, drawn, and overlay images, and `tatbot sim eval` produces JSON, CSV, and Markdown reports with tolerance-band F1 plus coverage, distance, contact, interaction, and clamp metrics. Checkpoint closed-loop evaluation uses a local framed worker with graceful shutdown and strict provenance and held-out checks, so blank controls remain valid zero-score results while dirty, contaminated, or otherwise non-comparable inputs fail explicitly; results remain screen-only.

### Audio-Servo Touch-Off
Audio-servo drawing now establishes a hover reference at the standoff and performs touch-off itself, using a slower final approach, a small search motion, and a tracking-lag guard; recordings distinguish acoustic contact from a mechanical fallback. `tatbot draw meter` describes the active RMS contact rule and no longer exposes the obsolete free-running-level bypass, while calibrated reference distances remain available in the contact stream.

### Inkmap Atlas Review
Inkmap now includes a reproducible atlas review sheet with one card per placement phrase, showing the selected anchor, covered region, and a body-facing view for human judgment. It complements deterministic and schema checks by exposing anchors that technically resolve but face inward or appear on the wrong side of the body.

### Offline Human-Representation Contracts
Human representation now has a versioned offline contract for nominal body identity, rest-surface placement, tool intent, measured-surface binding, and execution programs, along with a pinned MHR-through-SOMA model specification and allowlisted body cache. `tatbot body bootstrap` and `tatbot body audit` verify pre-downloaded assets by hash. That audited foundation is now consumed by the sole nominal-body path; it still does not authorize deployment, motion, emissions, or human contact.

Commits: f7912b2 e513b20 565c834 9a95345 b293466 a433fa0 fdd9b73 f6efc6d e1a8170 ae7f08c 2e94fb7 ebc051b 4979091 2b384a5 9b3fb30 3f52840 2b11692 be666cc 3807f15 3152e86 2b8920f 0f9e243 24aee1d c8194e5 e98aba5 998524a 49eb2ba 5b7782d 419db17 ae243b8 9dffdf4 2ef5fe3 642b5ca 899e0b4 cfc9a50 988ea8f 8dc9960 744da55 d607efa 1ca8f2b

## 2026-09-03

### Integrated Ink Dipping & Motion Stack
Ink dip trajectories are now integrated directly into the motion stack as joint-space path segments streamed by the 400 Hz executor. Dip motions feature configurable cap slots, revised approach heights, and kinematic planning to ensure safe pen retraction on exit.

### Automated Ink & Palette Calibration
The hardware calibration pipeline introduces an automated palette calibration phase that solves ink rack layout from calibration grid contact points while holding the pen tip fixed. Calibration sweeps manage teleoperation lifecycle automatically, and end-effector tool geometry and wrist parameters have been updated from recent physical touch-off measurements.

### CLI Streamlining & Consolidation
The CLI interface has been streamlined by consolidating redundant verbs across calibration, data management, simulation, and telemetry streams. Drawing execution options now use `--scan-only`, `--no-scan`, and `--design` flags, while live vision and replay streams have been unified under cockpit and viewer verbs.

### Audio Contact Sensing & Surface Feedback
Audio contact sensing has been added as a real-time feedback mechanism during drawing operations. RMS contact scoring, calibrated signal baselines, and contact-triggered servo back-off behavior improve surface contact safety and depth tracking.

### Telemetry Visualization Blueprint
Rerun telemetry visualization has been standardized into one shared layout blueprint, enforcing a consistent contract across camera streams, URDF visualization, and sensor plots.

### Hardware Documentation & Repository Structure
A sanitized Bill of Materials (BOM) has been published in the repository documentation. Internal documentation pointers and experiment logging workflows were reorganized to reference external documentation repositories.

### One InkLang prompt-to-location contract

InkLang now means one placement-only system: description to normalized intent
to a face+barycentric point on a versioned canonical body rest surface. Inkmap
and simulation consume the same TypeScript resolver and generated atlas;
interactive ambiguity requires an explicit choice, while batch selection uses
a named deterministic policy and recorded seed. The former Python phrase
realizer and competing semantic face picker were retired. PlacementFile v5 is
the sole accepted format and pose remains a TattooScenario concern.

The shipped compliance corpus contains 146 cases on the sole MHR-through-SOMA body, all 59
leaf sites, 8 zones, 6 aspects, 3 levels, and 6 relation kinds. Inkmap's normal
check now includes a browser prompt-to-anchor scenario and atlas anchors have a
reproducible visual contact-sheet review.

Commits: 0a619d5 078cd5e 4cf4dbd debe3c0 b4543df 3b443e5 34b47f0 c434e49 119a53a 6a952d1 b3a313f 1ca082f b4358a4 9e99960 ccb144f 0967327 fc16fbe 852b3f9 9586598 b52f372 0257b23 6ab7cf9 8578906 3cac893 70079cd 4cf1453 24414f1 36200a0 2f74707 70dcd54 f46a212 7507026 2091768 dd7c528 241cae2 1d1d57b 8452cca 5c5f9c6 5987421 3e53f2e

## 2026-09-02

### Autonomous Surface-First Drawing & Path Execution
A surface-first automated drawing pipeline and path planner have been introduced, supporting 7-DOF trajectory generation, wrist-camera surface height-field mapping, isometric chart anchoring, and camera-centric orbit inspection. Execution reliability and trajectory tracking are enhanced through angular feedforward control, pen-up carriage locking, endpoint settle guards, and unpinned process scheduling to prevent SDK starvation.

### 3D Tattoo Mapping & Anatomical Simulation
3D tattoo mapping includes an updated showcase of supported body postures, sagittal-plane leg kinematics corrections, and anatomical adjustments for reclined models. Demo defaults and sidebar controls have been refined for public presentation.

### Hardware Recovery & System CLI
Arm recovery commands now default to operating on both arms and correctly lookup golden configuration profiles by arm role. Automatic reconnection and parameter application resolve hardware boot-limit fault states upon landing.

### Public Contracts & Documentation
Public documentation, repository references, and release state metadata have been updated. Setup and verification scripts were revised for portability across standalone environments outside the test rig.

Commits: 0a0f85c 60f0fb4 c35a571 5889d44 0791b49 93700af 8789f57 b8660c8 2171b47 0021bff 58861c9 7cf9b7a 62b0bb0 dbe10b4 a0e2baa 626aa57 f8c8428 846a0da 8772b3e 888977b 270e3d3 d39109f a6d5833 cfa7868 ca5482e 4b4cf5e 7da2a98 fb48add a8168ca 7cf79d3 8f5423d 3629fa7 f29c2e5 64733b4 1d215b6 16b6c57 e47f976 e667057 6df8c9d baa0702

## 2026-09-01

### Teleoperation & Leader Friction Compensation
Leader arm joint friction coefficients and effort correction factors have been retuned to improve transparency and tracking fidelity. Teleoperation CLI flags `--damping` and `--assist` now incorporate runaway protection and base-joint velocity filtering to prevent motion instability.

### CLI & Hardware Profile Interface
Hardware profile selection is now exposed via dedicated CLI verbs and flags, allowing profile configuration to be explicitly forwarded across remote execution hops. Dataset hub publishing defaults to opt-in execution, and log outputs standardise hints on CLI verb names.

### Imitation Learning & Policy Evaluation Contracts
Policy training and evaluation pipelines now enforce effort-masking contracts, preventing unmasked external effort inputs from reaching models trained on masked modalities. Policy evaluation routines fix state validation for 14-DOF policy states and guard against invalid held-out claims on training fixtures.

### Simulation & 3D Ink Mapping
Simulation contact geometry has been re-aligned with measured tool center points and scene reconfiguration start states. 3D tattoo mapping routines now define contracts for posed scenarios and align tool geometry specifications.

Commits: fed7217 8952f43 7a7727a 52988bd 7163798 c70808a 9fe6c84 fba0eff 89105dc 5595c3d e68f63a 24964dc 44f18ff 6f88de8 715deec f2cac69 726f09f 81cb9a3 0b1d133 265da2b 11cf126 16b09fb 89519de 9db6f1b bed0952 5f2af1d 10b9380 723f62a 8e28e69 81bbaf5 64839f9 49b244c 85d6ce8 459c9ee 55facda c5f9802 913a37d 88a7e3a c1315ac 767af82
