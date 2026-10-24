---
summary: Public safety and responsible-testing boundary
tags: [safety, responsible-use]
updated: 2026-10-16
audience: [all]
---

# Safety scope

Tatbot is an experimental robot tattoo arm. The ROS 2 drawing stack tattoos
silicone practice skin and human skin with a needle cartridge, and draws on
paper with a ballpoint.

## Interlocks

These stop or bound motion on every substrate:

- The [e-stop](estop.md): heartbeat silence freezes every connected arm.
- The arm controllers' limits.
- The carriage contact cap, and the retract that follows a contact trip.
- The snag guard on a riding needle.
- Exclusive driver ownership: one process drives an arm at a time.

## Development practice

- Simulation, replay and no-arm tests are the quickest way to check software.
- Keep an operator at the physical stop whenever motion is enabled.
- A verb that declares a launch id (`--explain` lists it) gets one
  automatically, written to the arm node, ledgered and audited (pid chain, SSH
  origin).
- Treat an unknown calibration, tool, model, or dataset as untrusted input.
- Record the exact revision and test conditions for physical observations.

## Claims and evidence

Separate measured results, simulations, hypotheses, and release notes. A
simulated or unit-tested result is not a physical measurement.
