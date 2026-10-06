# tatbot_sim

`tatbot_sim` generates deterministic synthetic episodes and fixtures for
Tatbot. It is an offline data factory, not an RL environment and not a model
of safe human operation.

## Setup

```bash
uv sync
uv run python -m tatbot_sim.factory --list
```

Private rig checkouts use their qualified workspace and arm profile. A public
checkout falls back to the clearly marked fixtures in `config/examples/` for
imports and geometry-only development. Those placeholders are not calibration,
controller limits, or powered-operation authority. Offline simulation continues
with nominal tool geometry when qualification is absent and records that basis
as a development warning.

Flat-paper generation with `--production-draw-config` uses the same resolved
arm/tool model for Python FK, Cartesian chunks and the offline C++ planner.
The checker receives the configured arm prefix and working TCP independently
of the samples CSV, verifies their agreement, and returns the bound model in
the retained joint plan. The simulator still requires 1-nm parity with its
resolved tool geometry. Nominal datasheet geometry remains unqualified; this
offline declaration does not adopt calibration or authorize hardware motion.
Calls without an explicit tool model retain the checker's compiled ballpoint
default and its original mismatch guard.

The complete `scripts/check sim` suite and offline generation run without a
fresh private calibration. Nominal/synthetic geometry is warned and stamped
as development evidence; `--require-qualified-geometry` is the explicit
strict mode.

Keep generated datasets outside the repository. Each artifact should record
the simulator revision, configuration, task, tool fixture, and simulated label.

## Contact contract

`paper-draw` uses `resolved-tool-v1` geometry and `rigid-contact-v1`
interaction. The working TCP targets zero signed distance from the surface;
pigment is permitted only from 0.25 mm below through 0.50 mm above it. The
ballpoint carries its measured-radius tip collision and a flat paper pad has a
matching collision surface. Cylindrical or displaced substrates are stamped
`kinematic-contact-v1` and retain the same audited gate until their collision
mesh is independently qualified; the default audit rejects them as production
contact evidence.

Surface profile is independent of material. `paper-draw` defaults to flat;
`skin-erase` and `skin-tattoo` use balanced flat/cylinder batches. Override any
recipe with `--dr.surface.profile flat|cylinder|balanced`. Each episode records
its selected profile and cylinder radius.

Generate model-backed designs before starting a simulator process:

```bash
tatbot sim materialize -- --output-dir ~/tatbot-sim/designs/run-42 \
  --subject "a botanical crane" --count 16 --seed 42
tatbot sim sample --count 16 -- --output-dir ~/tatbot-sim/scenarios/run-42 \
  --design-source directory --generated-design-dir ~/tatbot-sim/designs/run-42
```

The materializer is the only sim command that calls Inkgen. It records exact
PNG/SVG bytes, hashes, prompt/model/seed, and trace settings; all later stages
are offline replay.

Posed-body scenario suites use the versioned search envelope in
`config/inkmap/placement-search.json`. The resolver records all body X/Y, yaw,
and fixture candidates and accepts only placements with 1 mm IK residual,
10 mm non-tool clearance, and 5 mm tool-shaft clearance outside the fitted
tool's terminal contact section. These are simulation proxy margins, not
powered-arm or human-contact qualification.

`tatbot sim resolve "a motif on the left forearm" -- ...` calls the canonical
TypeScript InkLang batch resolver (Node.js 22+), selects only a compatible
body/pose/support, canonically resolved surface anchor, and already
materialized design, then writes the typed request, placement, scenario,
manifest, and rejection ledger. Python owns only structured distribution and
pose policy here; it has no alternate InkLang parser, realizer, or face picker.
Use explicit `spiral-v1` only for the fixed calibration spiral.

Score a served checkpoint without importing LeRobot into the ManiSkill
process:

```bash
tatbot sim eval --policy-rollout -- \
  --server 127.0.0.1:8080 --policy /models/candidate \
  --wire-scenario act_rgbd14_masked --distribution paper-draw \
  --repetitions 3 --output-dir ~/tatbot-sim/eval/candidate
```

`scripts/eval/sim_policy_eval.py` owns the deployed gRPC client and action
queue. `tatbot_sim.policy_worker` owns the one-environment simulator and judge.
Their bounded local protocol is JSON plus explicitly typed arrays; evidence
records the follower-derived feature shapes, chunk cadence/rejections,
execution filter, checkpoint digest, tool geometry basis/warnings, and exact
design overlays. The result is a screen only, never powered-motion authority.

`tatbot sim audit` refuses older air-gap datasets by default. Use
`--allow-air-gap` only to inventory history or an explicitly named negative
control, never to make it eligible for a current training mixture.
Contact qualification requires a quality-gated fixed-point/pivot TCP; ordinary
simulation does not. For an axisymmetric tool, its body may remain honestly
`axis-inferred`: the known
bore-face origin and calibrated mount-to-tip vector determine the
contact-relevant axis, while roll does not change the profile. Optional
independent body measurements remain useful for asymmetric clearance studies,
but are not a prerequisite for development data. Nominal/unqualified geometry
is stamped `qualification: development` and warned without failing. The
opt-in `--require-qualified-geometry` gate is only for a caller deliberately
requesting calibration evidence.

The `paper-draw` factory samples one seed-deterministic mount-frame tip offset
per shard inside the calibration's recorded uncertainty. It is persistent for
the shard, modeling one fitted session rather than a pen that moves between
episodes. URDF visuals/collision, IK, and tool metadata share that offset; the
auditor checks its bound and reconstruction. Use
`--no-tool-calibration-jitter` only for an explicit central-calibration control,
or `--tool-calibration-scale` to declare a bounded sensitivity study.

Current run metadata also records the full Git revision and dirty state at the
start and end of generation. The default audit rejects a current-schema shard
if the revision is missing, changes during the run, or either state is dirty.

## Development contract

- Validate task, tool, substrate, and coordinate-frame combinations before
  building a scene.
- Assert action/observation shapes and units at the writer boundary.
- Use deterministic seeds for fixtures and report when rendering is stochastic.
- Do not mix private recordings or credentials into a public dataset.

Simulation can prove software/data invariants; it cannot prove physical
calibration, contact behavior, or human-use safety.

Normal body, paper, and skin drawing use the same thirteen simple artworks as
Inkmap. The default family split is `train`; use `--artwork-split validation`
or `test` for held-out artwork. Paper/skin drawing defaults to `--task artwork`
with a two-minute cap and nominal 0.3 mm simulated footprint. `--task spiral`
is the calibration control; historical language/maze/shapes/mix tasks are explicit
replay choices. The body sampler defaults to `--design-source artwork`, compiles
typed v3 programs, and cannot substitute random curves. Run
`tatbot sim qualify-artwork -- --output-dir ~/tatbot-sim/artwork-qualification`
for the source/preview/target/deposition comparison and rejection ledger.
