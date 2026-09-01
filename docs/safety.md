---
summary: Public safety and responsible-testing boundary
tags: [safety, responsible-use]
updated: 2026-08-31
audience: [all]
---

# Safety scope

Tatbot is an experimental robotics project. The public repository is for
software and art collaboration, not for performing tattoos or medical
procedures.

## Safe development boundary

- Prefer simulation, replay, and no-arm tests.
- Use an instrumented non-human fixture for physical integration work.
- Keep an operator at the physical stop whenever motion is enabled.
- Treat an unknown calibration, tool, model, or dataset as untrusted input.
- Record the exact revision and test conditions for physical observations.

The [e-stop contract](estop.md) describes the software-side fail-closed
expectation. It does not replace a site-specific risk assessment, training, or
an owner-approved acceptance procedure.

## Claims and evidence

Separate measured results, simulations, hypotheses, and release notes. A public
example must not imply that a component has been accepted for human use merely
because it builds or passes unit tests.
