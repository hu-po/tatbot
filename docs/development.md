---
summary: Public setup and developer workflows
tags: [setup, development]
updated: 2026-08-31
audience: [dev, contributor]
---

# Development

Tatbot is a collection of independently buildable components. You can work
on the web app or simulator without access to robot hardware.

## Prerequisites

- Git
- Python 3.12 and `uv`
- A C++ toolchain and CMake for teleoperation work
- Rust and Cargo for the vision service
- Node.js 22 or newer for Inkmap

Clone the repository, then choose the component you need. There is no required
root-level install.

## Component builds

| Component | Directory | Check |
| --- | --- | --- |
| Teleoperation | `cpp/teleop/` | `cmake -B build -S . && cmake --build build` |
| Vision service | `rust/visiond/` | `cargo build --release` |
| LeRobot integration | `python/lerobot_robot_tatbot/` | `uv sync` |
| Simulator | `python/tatbot_sim/` | `uv sync` in the component |
| Inkmap | `web/inkmap/` | `npm ci && npm run check` |
| E-stop firmware | `firmware/estop_pico/` | see the component README |

## Repository checks

Run the same checks used by the project before opening a change:

```bash
scripts/check --light
```

The command reports unavailable optional toolchains as `SKIP`; a `FAIL` is an
actionable defect. For docs-only work, run `scripts/check docs` as well.

## Workflow boundaries

Use the public docs for reproducible, hardware-independent work. Deployment
configuration, private topology, experiment evidence, and powered acceptance
remain in the private `internal/` tree and are not prerequisites for a public
contribution.
