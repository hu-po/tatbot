---
summary: Public ink and consumable data model
tags: [ink, data-model]
updated: 2026-08-31
audience: [dev, contributor]
---

# Ink data model

Tatbot represents consumables as versioned data so a design or experiment can
state which material assumptions it used. The public interface is a schema and
ledger shape, not a purchasing or human-use procedure.

## Rules

- Use stable identifiers and explicit units.
- Record whether a value is measured, estimated, or a default.
- Keep append-only events separate from derived balances.
- Do not commit private inventory, supplier, or operator records.

Consumers should validate the schema at read time and fail closed on an unknown
version. Private load, fit, and acceptance workflows stay in the internal repository
documentation.
