---
summary: Public Tatbot software architecture
tags: [architecture, components]
updated: 2026-08-31
audience: [dev, contributor]
---

# Architecture

Tatbot is intentionally split into small build roots so a contributor can
develop one surface without installing the entire robotics stack.

```text
operator or test
      │
      ├── C++ teleop ────────┐
      ├── LeRobot adapter ───┼── robot interface
      ├── Rust visiond ──────┘
      ├── Python simulator
      └── Web Inkmap ── placement JSON
```

## Boundaries

- `cpp/teleop/` owns the high-rate arm-control loop.
- `python/lerobot_robot_tatbot/` owns the LeRobot adapter and episode API.
- `rust/visiond/` owns camera capture, synchronization, and replay data.
- `python/tatbot_sim/` owns hardware-independent fixtures and simulation.
- `web/inkmap/` owns browser-side design placement and preview.
- `firmware/estop_pico/` owns the e-stop heartbeat endpoint.
- `config/` and `urdf/` are shared interfaces; consumers must record their
  revision in a run manifest.

The private deployment graph, node roles, credentials, and acceptance evidence
are deliberately not part of this page.

## Change guide

Start with the smallest component README, add or update a focused test, then
run `scripts/check --light`. If an interface crosses components, document the
schema and compatibility rule before changing both sides.
