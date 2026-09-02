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
controller limits, or powered-operation authority; contact-qualified generation
still refuses without real qualification unless an explicit validation-only
override is selected.

The complete `scripts/check sim` suite includes private-profile qualification
tests and therefore runs only when a locally qualified workspace and arm
profile are present. Public fixtures support imports and geometry-only commands;
they deliberately do not impersonate qualification evidence.

Keep generated datasets outside the repository. Each artifact should record
the simulator revision, configuration, task, tool fixture, and simulated label.

## Contact contract

`paper-draw` uses `resolved-tool-v1` geometry and `rigid-contact-v1`
interaction. The working TCP targets zero signed distance from the surface;
pigment is permitted only from 0.25 mm below through 0.50 mm above it. The
ballpoint carries its measured-radius tip collision and a flat paper pad has a
matching collision surface. Shaped substrates are stamped
`kinematic-contact-v1` and retain the same audited gate until their collision
mesh is independently qualified; the default audit rejects them as production
contact evidence.

`tatbot sim audit` refuses older air-gap datasets by default. Use
`--allow-air-gap` only to inventory history or an explicitly named negative
control, never to make it eligible for a current training mixture.
Contact generation requires a quality-gated fixed-point/pivot TCP. For an
axisymmetric tool, its body may remain honestly `axis-inferred`: the known
bore-face origin and calibrated mount-to-tip vector determine the
contact-relevant axis, while roll does not change the profile. Optional
independent body measurements remain useful for asymmetric clearance studies,
but are not a prerequisite for contact data. `--allow-provisional-geometry` on
generation and `--allow-provisional` on audit exist only for nominal or failed
contact calibration in labelled engineering validation.

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
