---
summary: Public configuration and schema conventions
tags: [configuration, schemas]
updated: 2026-08-31
audience: [dev, contributor]
---

# Configuration

Configuration is code-adjacent API. Prefer checked-in schemas and component
defaults over undocumented environment variables.

## Rules

- Keep units and coordinate frames in the schema or a nearby comment.
- Give a format a version before changing its meaning.
- Validate configuration at the boundary and fail closed on unknown values.
- Keep machine-specific addresses, credentials, and deployment overrides out of
  public examples.
- Record the configuration revision with generated data and run logs.

## Public interfaces

- Robot geometry: `urdf/tatbot.urdf`
- Tattoo placement: `config/inkmap/placement.schema.json`
- Body-independent placement intent:
  `config/inkmap/inklang-intent.schema.json`
- Body-specific region atlas: `config/inkmap/region-atlas.schema.json`
- Grounded placement result: `config/inkmap/inklang-resolution.schema.json`
- Posed offline realization: `config/inkmap/tattoo-scenario.schema.json`
- Component defaults: the relevant directory under `config/`
- CLI behavior: [the command reference](cli.md)

When a schema changes, update its fixture, migration note, and consumer tests in
the same change.

InkLang lexicon, intent schema, atlas schema, resolver algorithm, PlacementFile,
and TattooScenario versions are independent. Do not use one version number as a
proxy for another; see [InkLang](inklang.md#intent-atlas-and-resolution).
