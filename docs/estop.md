---
summary: Public safety scope for the hardware e-stop interface
tags: [safety, hardware]
updated: 2026-08-31
audience: [dev, contributor]
---

# Hardware e-stop

The e-stop is a fail-safe input shared by motion-capable components. A valid
heartbeat is required before a production motion entry point may command an
arm; a pressed, disconnected, malformed, or timed-out signal must stop motion.

## Public contract

- The stop path fails closed: loss of signal is treated as a stop.
- Motion consumers must hold or safely retract according to their local
  hardware contract; they must not continue stale targets.
- Recovery must re-seed command state from measured state before resuming.
- The e-stop is a motion stop, not a guarantee that power is removed.

The firmware and consumer implementations are in
`firmware/estop_pico/`, `cpp/teleop/`, and
`python/lerobot_robot_tatbot/`. Keep their protocol and timeout tests together
with code changes.

## Testing boundary

Run parser, timeout, reconnect, and launcher tests with no arm connected. A
powered test needs an explicit operator, a verified physical stop, a clear
non-human fixture, and a recorded acceptance run. The public documentation does
not authorize tattooing or human operation.
