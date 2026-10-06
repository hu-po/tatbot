---
summary: Public Tatbot software architecture
tags: [architecture, components]
updated: 2026-09-09
audience: [dev, contributor]
---

# Architecture

Tatbot is intentionally split into small build roots so a contributor can
develop one surface without installing the entire robotics stack.

```text
operator or test
      │
      ├── ROS 2 stack (ros/) ─┐  drawing: one arm, ballpoint
      ├── C++ teleop ─────────┤
      ├── LeRobot adapter ────┼── robot interface
      ├── Rust visiond ───────┘
      ├── Python simulator
      └── Web Inkmap ── placement JSON

placement text ── InkLang intent + body atlas ── rest-surface anchor
motif/style text ── Inkgen ── source PNG ── DrawingBot V3 ── artwork

TattooProgram ── SurfacePlacement ── InkProgram
 body-free art      target address     tool intent

plane design + fitted tool + registered page
      └── tatbot ros compile ── program.json ── tatbot ros draw
body placement ── measured address mapping still required (refused)
```

## Boundaries

- `ros/` owns autonomous drawing: the ROS 2 Jazzy stack that compiles a design,
  registers the page, touches it and draws (`ros/README.md`).
- `cpp/teleop/` owns the high-rate leader/follower teleop loop.
- `python/lerobot_robot_tatbot/` owns the LeRobot adapter and episode API.
- `rust/visiond/` owns camera capture, synchronization, and replay data.
- `python/tatbot_sim/` owns hardware-independent fixtures and simulation.
- `web/inkmap/src/core/inklang/` owns the single placement-text to
  rest-surface-anchor implementation. Its JSON contracts are language-neutral.
- `web/inkmap/` owns browser-side placement confirmation, editing, and preview;
  Python simulation consumes canonical InkLang output rather than resolving a
  second semantic location.
- `web/inkgen/` owns artwork generation, never anatomy grounding.
- `firmware/estop_pico/` owns the e-stop heartbeat endpoint.
- `config/` and `urdf/` are shared interfaces; consumers must record their
  revision in a run manifest.

The human-representation contracts preserve two different geometry
authorities. A digest-bound SOMA mid surface may provide a nominal semantic,
pose, and preview prior. A measured local `tatbot.surface/1` remains the only
metric surface an execution program may bind. Nominal geometry cannot fill a
missing observed cell.

`TattooProgram`, `SurfacePlacement`, and `InkProgram` carry artwork,
target addresses, and ordered tool intent. The robot draws plane designs:
`tatbot_ink` compiles them into the ROS 2 stack's `program.json`. Simulation
lowers retained designs through the shared Cartesian compiler and C++ planner
into checked `tatbot.draw-samples/1`. Neither supplies a body mapping.

The old `ExecutionProgram/1` schema and synthetic fixtures remain readable
for historical inspection, but its standalone compiler has been retired.
Research and learned modules cannot write samples or import motion authority.

The pinned [MHR-through-SOMA model spec](../config/body-models/mhr-soma-v1.json)
has an explicit offline cache bootstrap and hash-only audit. It is the sole
nominal-body provider and Inkmap default: a numeric `BodyIdentity/1` is
transferred through SOMA's fixed mid topology, and named poses preserve those
face addresses. This software cutover does not authorize deployment, capture,
motion, emissions, or human contact.

## Body cache verification

`tatbot body bootstrap --spec SPEC --source-dir SOURCE --cache-dir CACHE`
copies an explicit, pre-downloaded allowlist into an immutable cache.
`tatbot body audit --spec SPEC --cache-dir CACHE` verifies it by hash without
deserializing model assets or using the network. State global `--json` before
the command for machine output; `--output DIR` on audit optionally writes an
immutable report outside the cache and repository.

The stdlib `tatbot-contracts` package owns canonical JSON and these validators.
The CLI and simulator use the same reviewed spec digest and verify assets again
at their respective input boundaries. Missing, changed, writable, symlinked,
or unlisted assets are refusals. Hash verification is not model execution or
hardware qualification.

A nominal body cannot supply a missing physical measurement.

Differentiable patches, proposal training, compliant phantom mechanics, coupled
mechanics and generic anatomy are frozen experiments. Existing tests and
historical reproducers remain, but new runtime consumers are rejected by the
stdlib body check. Reopening requires a named consumer, measured baseline
deficiency, experiment and acceptance criteria. See
[human-representation consumers and evidence](human-representation.md) for the
maintained core and the next non-human fixture experiment.

The private deployment graph, node roles, credentials, and acceptance evidence
are deliberately not part of this page.

See [InkLang](inklang.md) for the prompt-to-location contract and
[Inkmap](inkmap.md) for its browser consumer. Pose, support, world transforms,
and reach enter only when a PlacementFile is compiled into a TattooScenario.

## Change guide

Start with the smallest component README, add or update a focused test, then
run `scripts/check --light`. If an interface crosses components, document the
schema and compatibility rule before changing both sides.
