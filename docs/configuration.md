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
- Component defaults: the relevant directory under `config/`
- CLI behavior: [the command reference](cli.md)

When a schema changes, update its fixture, migration note, and consumer tests in
the same change.
