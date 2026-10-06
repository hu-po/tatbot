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
| Simulator | `python/tatbot_sim/` | `uv sync --extra maniskill`; omit the extra for CPU preparation |
| Inkmap | `web/inkmap/` | `npm ci && npm run check` |
| E-stop firmware | `firmware/estop_pico/` | see the component README |

## Working on a development machine

Every session works in its own worktree on its own pushed branch, and lands
with one command; see [Work](work.md). In short:

```bash
tatbot work start --task fix-the-thing   # ~/tatbot-work/<session>, branch agent/<node>/<session>
# ... edit, commit as usual; every commit is pushed to the agent branch ...
tatbot work land                         # rebase, fast tier, push to main, delete the branch
```

The shared checkout is a mirror that refuses commits. `tatbot work status`
shows what is unlanded on this node; `tatbot work park "why"` leaves work
unlanded on purpose. `tatbot work init` sets a machine up once.

## Repository checks

Run the same checks used by the project before opening a change:

```bash
scripts/check
```

The bare command is the fast tier and finishes in minutes: lint, the
complexity and disclosure ratchets, the CLI and configuration contracts, the
docs build, and the light test profile without slow-marked tests. Pytest jobs
use every core by default, except the fast and light scripts tests, which use
four workers; `PYTEST_WORKERS=N` picks a count and `0` forces one.
Each process defaults `OMP_NUM_THREADS`, `OPENBLAS_NUM_THREADS` and
`MKL_NUM_THREADS` to one inner thread, preserving explicit settings. This
avoids multiplying pytest concurrency by a separate numerical worker pool.
`scripts/check --full` runs every job this node can except the simulator ones;
`scripts/check rust` (one workspace build, every crate's tests once) and
`scripts/check sim-fast` are the ones to run by name for a change in those
roots. A job whose inputs are
unchanged since its last PASS on this machine reports `PASS (cached ...)`;
`TATBOT_CHECK_NO_CACHE=1` reruns it.

Light scripts tests (the fast tier and `scripts/check --light tests`) default
OpenBLAS, OpenMP and MKL to one thread per native pool, since pytest workers
already run in parallel. The defaults apply before Python starts, including
collection and test subprocesses. Explicit `OPENBLAS_NUM_THREADS`,
`OMP_NUM_THREADS` and `MKL_NUM_THREADS` values take precedence. Full offline
tests and direct pytest invocations use the caller's native thread settings.

The command reports unavailable optional toolchains as `SKIP`; a `FAIL` is an
actionable defect. For docs-only work, run `scripts/check docs` as well.

A passing test whose call phase exceeds the per-test budget (60 s of wall time, set by
`TATBOT_TEST_BUDGET_S`) fails its job. Mark it `@pytest.mark.slow` -- the fast
tier then skips it and `--full` still runs it -- or make it cheaper.
Expensive shared fixture setup also belongs in the slow tier: mark every test
that consumes it. The two-arm calibration capture/solve/adoption tests use this
tier; shorter owner and fault checks remain fast. Named `scripts/check tests`
and CI include the slow tests.

## Workflow boundaries

Use the public docs for reproducible, hardware-independent work. Deployment
configuration, private topology, experiment evidence, and powered acceptance
remain in the private `internal/` tree and are not prerequisites for a public
contribution.

## Test profiles and coverage

`scripts/check --light tests` installs `scripts/tests/requirements-light.txt`
for core offline tests. `scripts/check tests` installs the full offline profile
from `scripts/tests/requirements.txt`, including vision, dataset and
training dependencies. A named profile fails if a required dependency is absent.

`scripts/check tests-integration` runs the LeRobot-dependent scripts test
modules and validates the patch catalog against copies of the installed sources
in the locked plugin environment. It is a heavy job and is omitted by
`--light`. Cloud CI runs both the offline and integration jobs. The plugin's own
suite remains `scripts/check plugin`.

Omitted collection coverage is printed even with quiet pytest output and appears
as separate `SKIP` entries in the check summary. A passed test job with omitted
coverage is not a complete acceptance run. Tool geometry and wrist-tag generation
also report separately when NumPy is unavailable.

## LeRobot patch ownership

The ordered compatibility catalog lives in `scripts/lib/lerobot_patches.py`.
`scripts/il_patch_lerobot.py` remains the launcher entry point after environment
sync; `scripts/lib/lerobot_patch_engine.py` owns application and verification.
It plans all available targets in memory, resolves historical overlapping patches,
rejects ambiguous sites and invalid Python, and only then replaces installed
files without modifying uv's shared cache. Missing targets are recorded; broken
imports inside an installed target fail validation.

A successful CLI application writes
`<venv>/share/tatbot/lerobot-patches.json`, containing the catalog hash, upstream
version, final module hashes, and omitted targets. Compare this manifest alongside
the locked dependency revision when diagnosing differences between environments.
The manifest is local source verification, not hardware qualification or a
replacement for behavior tests. Apply patches before starting consumers; replacing
multiple module files is not atomic for an already running process.

## Rust production profiles

`scripts/check rust` is one workspace build: locked Clippy (warnings are
errors) and every crate's tests once, with every feature whose native
dependency this node has. A feature whose dependency is absent is left out and
named as a `SKIP`: GStreamer (`gstreamer-1.0` and `gstreamer-app-1.0` through
`pkg-config`) for PoE capture, `realsense2` for the wrist cameras, the pinned
vendor SDK for the arm adapter. A compilation or test failure remains `FAIL`.
Tests use offline fixtures; no capture command is launched.

Under `scripts/check --full`, or when `TATBOT_RUST_PROFILES` selects a
comma-separated subset, `visiond` is also built on its own for each profile in
`rust/check-profiles.json` (`cargo build --release --bins` with only that
profile's features), so another workspace member cannot silently enable a
feature a deployment lacks. Profile builds are compile checks; nothing is
re-tested. The profiles cover core, viewer, PoE capture and wrist capture,
including the Zenoh-enabled deployment variants, and deployment inventory tests
fail when a service or the visualizer launcher introduces a feature combination
absent from the matrix.

A profile whose native dependency is missing is a named `SKIP`, unless
`TATBOT_RUST_REQUIRED_PROFILES` names it: then it fails, as does a missing Rust
toolchain. Unknown names and required profiles excluded from the selection fail
validation. For example, on a capture build host with the RealSense SDK:

```bash
TATBOT_RUST_PROFILES=vision-wrist,vision-wrist-bus \
TATBOT_RUST_REQUIRED_PROFILES=vision-wrist,vision-wrist-bus scripts/check rust
```

Passing these checks establishes build and offline-test coverage, not camera or
robot qualification.

Hosted CI builds the core, viewer and both PoE profiles once a day, and for
pull requests that touch the Rust tree, after installing the GStreamer
development libraries; it is a daily cross-check, not a per-commit gate. The
two wrist profiles run on SDK-equipped fleet hosts; they are not silently
counted as covered by the hosted job.

Cargo defaults native C compilation to GNU C17 through `.cargo/config.toml`.
The pinned AprilTag sources use pre-C23 function declarations; this keeps newer
GCC defaults from breaking their bundled build. An explicit `CFLAGS` overrides
the default.
