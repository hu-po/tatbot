---
summary: Public safety scope for the hardware e-stop interface
tags: [safety, hardware]
updated: 2026-09-30
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

The active C++ or Python monitor also publishes `tatbot.estop-status/1` as an
atomic runtime snapshot. A separate writer publishes it so slow or failed
filesystem writes cannot stall heartbeat consumption. `tatbot status` reads that snapshot instead of opening
the serial device a second time. It accepts health only when the device matches,
the producer PID is alive, the state and heartbeat fields are internally consistent, and the snapshot
is at most one second old. Missing or stale telemetry is reported as unknown;
`pressed` and `fault` are reported as failed. The snapshot is observability only
and is never an input to the motion safety path.

Observers discover the same user's snapshot in the desktop runtime directory
or the login-session fallback under `/tmp`, so separate terminal environments
can share one monitor. Multiple snapshot locations are reported as ambiguous;
invalid snapshots remain rejected. An explicit `TATBOT_ESTOP_STATUS` path stays
authoritative and never falls back to another monitor.

## One button through the network relay

A single button can stop more than one arm. The palette Raspberry Pi reads the
button's normally-closed contact and sends the same `EST1 <seq> <state>` frame,
one per UDP datagram at 100 Hz, to every reader: the ROS 2 drawing driver and,
where an arm node's profile names the relay as its e-stop
(`driver.estop_device` `udp://[HOST]:PORT?from=estop-relay`), that node's Python
monitor or guarded native calibration owner. Calibration conductors resolve
the relay role through the execution owner's node map and pass a literal IPv4
sender to the native reader; unresolved or ambiguous roles are refused. Each reader:

- accepts datagrams only from the relay's address; anything else is silence,
  never a release;
- drops a reordered or replayed frame while the stream is fresh, and re-seeds
  the sequence after silence;
- decides on three agreeing frames, like the serial path;
- treats silence longer than the relay path's budget (0.15 s) as a stop.

A datagram lost on the way to one reader stops that reader's arm only. The
C++ teleoperation and recovery tools read a serial e-stop only and refuse a
relay e-stop.

The tattoo machine's power switch on the same Pi is a reader too, over the
Pi's loopback. The e-stop freezes the arms but never cuts power, and a machine
left running over a frozen arm would strike one spot, so the switch powers the
machine only while the newest frame reads released and is at most 0.15 s old,
and while the drawing session's own commands keep coming and ask for it. Once
the e-stop reads pressed or silent, releasing it does not restart the machine:
the session must ask for it off and then on again.

## Testing boundary

Run parser, timeout, reconnect, and launcher tests with no arm connected. A
motion entry point validates its live heartbeat; this document adds no separate
per-command operator confirmation. The remaining physical-test boundaries are
in [Safety scope](safety.md).
