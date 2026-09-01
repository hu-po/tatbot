# tatbot_sim

`tatbot_sim` generates deterministic synthetic episodes and fixtures for
Tatbot. It is an offline data factory, not an RL environment and not a model
of safe human operation.

## Setup

```bash
uv sync
uv run python -m tatbot_sim --help
```

Keep generated datasets outside the repository. Each artifact should record
the simulator revision, configuration, task, tool fixture, and simulated label.

## Development contract

- Validate task, tool, substrate, and coordinate-frame combinations before
  building a scene.
- Assert action/observation shapes and units at the writer boundary.
- Use deterministic seeds for fixtures and report when rendering is stochastic.
- Do not mix private recordings or credentials into a public dataset.

Simulation can prove software/data invariants; it cannot prove physical
calibration, contact behavior, or human-use safety.
