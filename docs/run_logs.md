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

## Debugging checklist

1. Read the metadata and final structured event.
2. Check the command and repository revision.
3. Inspect the complete console log.
4. Compare artifact manifests, not only screenshots.
5. State what was measured, what is inferred, and what remains unknown.

Tatbot's internal writer and retention policy live outside the public docs.
