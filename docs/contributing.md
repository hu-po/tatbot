---
summary: How to contribute technical and artistic work to Tatbot
tags: [contributing, workflow]
updated: 2026-08-31
audience: [dev, artist, contributor]
---

# Contributing

Contributions should be small, testable, and clear about whether they affect
software, a data format, or an experiment.

## Before editing

- Search for an existing interface or schema before adding a duplicate.
- Keep generated files tied to their generator.
- Keep private deployment details and raw experiment evidence out of public
  documentation.
- For safety-sensitive code, preserve the fail-closed behavior and add a no-arm
  regression test first.

## Submit a change

```bash
scripts/check --light
git diff --check
```

Use an imperative commit subject with a component prefix. Explain the measured
behavior, test command, and any remaining uncertainty in the change description.

## Art and data

Artists can contribute SVG designs and placement examples using the public
[design format](design-format.md). Do not commit personal data, private
recordings, credentials, or artifacts that imply consent for human operation.

## Hardware work

Hardware access is not required for a public contribution. If a change needs a
powered evaluation, keep its acceptance record in the private engineering
tree and publish only the reproducible software contract.
