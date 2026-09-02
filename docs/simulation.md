---
summary: Hardware-independent Tatbot simulation workflow
tags: [simulation, testing]
updated: 2026-09-01
audience: [dev, contributor]
---

# Simulation

Use `python/tatbot_sim/` for offline development, dataset-shape checks, and
control experiments that do not connect to an arm or camera.

## Quick start

```bash
cd python/tatbot_sim
uv sync
uv run python -m tatbot_sim --help
```

Keep generated episodes and renders outside the repository. Include the source
revision and simulator configuration in any artifact manifest.

## A dataset to read without generating one

[`tatbot/sim-paper-draw-demo`](https://huggingface.co/datasets/tatbot/sim-paper-draw-demo)
is a published shard of the `paper-draw` distribution — 8 episodes, 2836
frames, wrist RGB and depth, generated with seed 0 and no flags beyond the
recipe. Use it to inspect the dataset shape, feature names and metadata that
this repository produces before running the factory yourself:

```bash
uv run --project python/lerobot_robot_tatbot python -c "
from lerobot.datasets.lerobot_dataset import LeRobotDataset
ds = LeRobotDataset('tatbot/sim-paper-draw-demo')
print(ds.meta.info['total_episodes'], 'episodes;', ds.meta.info['total_frames'], 'frames')
print(sorted(ds.meta.features))"
```

Regenerate an equivalent shard locally with:

```bash
tatbot sim generate paper-draw -- --out-dir <dir> --num-episodes 8 --num-envs 4 --seed 0
```

Each `paper-draw` invocation represents one fitted session: its seed chooses a
single persistent tip offset within the recorded calibration uncertainty, and
the run metadata records both the central calibration and the applied offset.
Across shard seeds this varies plausible seating/calibration geometry while
preserving exact agreement between the visible tip, TCP, collision, and marks.

## What simulation proves

Simulation can validate pure transforms, schema handling, deterministic replay,
and software integration. It does not prove camera calibration, contact force,
e-stop behavior on physical hardware, or safe human use.

## Posed-body scenarios

Inkmap placement files describe design intent on a canonical rest-body
surface. `config/inkmap/tattoo-scenario.schema.json` describes one resolved
offline simulation realization: body and rig identity, pose, body/world and
robot/world transforms, support fixture, fitted tool, immutable SVG, and the
derived face/barycentric stroke trace.

The split is intentional. A placement survives pose changes; a scenario is
fully replayable. Inkmap region `uv` is semantic and normalized, so it must not
be used as a metric drawing chart. The simulator maps SVG millimetres through a
surface trace and then skins those face/barycentric points into the selected
pose.

Named body poses are initially kinematic and static within an episode. This is
a geometry and planning model, not a soft-tissue, breathing, or dynamic-person
model. The checked-in `*.rig.npz` sidecars contain canonical rest vertices,
four normalized weights per vertex, named-pose vertices, and joint matrices;
the browser's Three.js skinning and Python's numerical surface loader are both
gated against the Blender-authored output. The checked-in Blender export and
loader-parity tests are the reproducible delivery and acceptance gates.

Materialize one placement without launching SAPIEN:

```bash
tatbot sim compile config/inkmap/examples/forearm-placement-v4.json -- \
  --pose reclined-left-arm-supported --seed 42 --output /tmp/forearm-scenario.json
```

Compilation is CPU-only. It verifies body, rest-surface, rig, design, and
placement identities before writing, and it fails explicitly on unsupported
SVG constructs, non-manifold/open surface exits, or topology discontinuities.
By default the patch normal is aligned to robot +Z and its +u axis uses the
validated `pi` robot-world yaw; `--target-world-m` and `--patch-yaw-rad` expose
that constrained body-to-robot placement. Dataset generation recomputes every
trajectory's FK and refuses the scenario if any target exceeds the 1 mm IK
residual gate.

Materialize a deterministic coverage suite before launching the simulator:

```bash
tatbot sim sample -- \
  --output-dir ~/tatbot-sim/scenarios/seed-42 --count 64 --seed 42
```

The sampler balances both bodies, five tattoo-session poses (supine, prone,
reclined with the legs on a chair rest, and reclined with either arm
supported), six initial atlas
sites, and the ten built-in designs. It constrains every compiled trace to the
requested face-labeled region, pairs supported-arm poses only with a tattoo on
that same arm, then probes 160 exact trajectory targets with
the 3RL CPU kinematic chain and selects the first passing robot-relative patch
yaw. `attempts.jsonl` records accepted and rejected attempts with explicit
reasons; exhausting the bounded retry budget fails the suite. The later dataset
run still recomputes FK for every target and refuses any residual over 1 mm.
`--no-reach-audit` exists for geometry-only debugging;
its output is not reach-qualified. To include locally generated artwork, save
the SVG files first and add
`--generated-design-dir /path/to/materialized-svgs`; the episode loop never
makes a network request. Add `--generated-only` to exclude the checked-in
library. Generated suites must be outside the checkout.

The compiled scenario enters the normal Tatbot expert, IK, floor-clamp, ink,
render, and LeRobot writer through a separate distribution (the existing
`skin-tattoo` silicone-pad distribution is unchanged):

```bash
tatbot sim generate body-tattoo -- \
  --scenario /path/to/one/accepted.scenario.json \
  --out-dir ~/tatbot-sim/body-forearm --num-episodes 8 --num-envs 8 --seed 0
```

The scene includes the complete posed body as a kinematic visual, a textured
drawable mesh patch, conservative bone-capsule collision proxies, and the
named support or positioning fixture. Generated OBJ caches live under
`~/.cache/tatbot/body-scenarios/`; datasets remain outside the repository.
The bed/chair proxy is authored in the named pose frame and transformed with
the body, so robot-relative patch alignment cannot leave its support behind.
Body-tattoo approach and inter-stroke hover are capped at 20 mm above the local
surface; this preserves an approach while staying inside the audited 3RL
orientation envelope.

The current body patch is a curved, kinematically projected contact surface;
the coarse body capsules are avoidance proxies, not a qualified skin contact
mesh. Body datasets are therefore stamped `kinematic-contact-v1`, target the
resolved TCP at zero working offset, and remain validation-only. Current main
also refuses the 3RL because it has no quality-gated pivot TCP of its own; the
explicitly labelled `--allow-provisional-geometry` validation override does not
make the output production-qualified. An axis-inferred body is not itself a
refusal when its axisymmetric contact tool has a qualified fixed-point
calibration.

## Contribution checklist

1. Add a deterministic fixture for the new behavior.
2. Assert units and coordinate frames at the boundary.
3. Run the simulator tests and `scripts/check --light`.
4. Label simulated results as simulated in the run manifest.
