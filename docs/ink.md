---
summary: Public ink and consumable data model
tags: [ink, data-model]
updated: 2026-09-27
audience: [dev, contributor]
---

# Ink data model

Tatbot represents consumables as versioned data so a design or experiment can
state which material assumptions it used. The public interface is a schema and
ledger shape, not a purchasing procedure.

## Rules

- The installed palette is calibration palette v11 in `urdf/palette.urdf`,
  six caps independent of the robot URDF. See [palette geometry and
  calibration](palette.md). Its pose comes only from a fresh tag measurement;
  there is no fixed-URDF or single-point fallback.

- Use stable identifiers and explicit units.
- Record whether a value is measured, estimated, or a default.
- Keep append-only events separate from derived balances.
- Do not commit private inventory, supplier, or operator records.

## CLI

The operator surface is four `tatbot ink` verbs over `scripts/ink.py`:
`ink status` (reads; `--ledger [-n N]` for the event tail), `ink
mise-en-place` (the pre-session checklist for the stated `--ee-tool`;
`--strokes` dry-runs the dip planner), `ink edit -- <load|dump|bottle|
cartridge|caps|reconcile|weigh> …` (the one `mutates-config` verb: it rewrites
the tracked load/inventory files or appends a ledger event, and refuses any
subcommand that does not write), and `ink sync` (`remote`: copies other
nodes' ledgers here). Anything else `ink.py` offers, such as `fit` or
`plan`, passes through `tatbot ink -- …` as `offline`.
`tatbot ink <verb> --explain` states the tier and gates.

Consumers should validate the schema at read time and fail closed on an unknown
version. Private load, fit, and acceptance workflows stay in the internal repository
documentation.
