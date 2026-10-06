---
summary: Public run-log format and debugging workflow
tags: [logs, debugging]
updated: 2026-08-31
audience: [dev, agent]
---

# Run logs

Every Tatbot workflow should produce a self-contained run record outside the
repository. A log makes a result reproducible without relying on copied
terminal output.

## Minimum record

Record the commit, command, environment, start/end time, outcome, and paths to
derived artifacts. Structured events should use stable names and include units.
Never put recordings, credentials, or private host details in a Git-tracked
public log.

## Suggested layout

```text
<run-root>/<workflow>/<run-id>/
├── meta.json       # revision, argv, environment summary, exit code
├── console.log     # complete stdout/stderr
├── run.jsonl       # timestamped structured events
└── artifacts/      # optional derived outputs, with a manifest
```

The run id should be sortable and unique. Include the execution environment in
metadata rather than encoding private infrastructure in the public filename.

## The `tatbot logs` verb

`tatbot logs` is the reader for these directories (it runs the internal
writer's index in-process; `scripts/tatbot-logs` is a shim for it). Its
subcommands: `list` (recent runs), `last <workflow>` (the most recent run of
one workflow — start here), `show <run-id>`, `tail <run-id> [-f]`,
`fetch <run-id>` (copy a remote run here, media skipped), `du`, `prune`
(retention; dry run unless `--yes`), `root`, `reindex`, `compact`, and
`selftest`. `begin` / `end` / `event` / `artifact` are what the launchers call
to write a run, not operator commands. `tatbot logs -- --help` lists them.

`list --all-nodes` sweeps the fleet: every node in the sweep set is dialed at
the ssh target and checkout `config/nodes.json` records for it, the same way
`tatbot --on <node>` reaches it. A node with no roles there is retired; the
sweep names it and does not dial it, and `show <run-id>` still resolves that
node's runs from the id. `TATBOT_NODES="a b"` sweeps exactly those nodes.

## Debugging checklist

1. Read the metadata and final structured event.
2. Check the command and repository revision.
3. Inspect the complete console log.
4. Compare artifact manifests, not only screenshots.
5. State what was measured, what is inferred, and what remains unknown.
6. Reconcile launch COUNT against the run index (`tatbot logs count <workflow>`
   before, `--expect N --before M` after) before calling any launch
   uncommanded; never judge that from notification timing.

Tatbot's internal writer and retention policy live outside the public docs.
