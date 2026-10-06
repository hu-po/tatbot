# E-stop heartbeat firmware

This directory contains the small firmware endpoint used by motion consumers.
It emits a versioned heartbeat and treats an open or unreadable input as a
stop. The protocol and timeout behavior are tested by the host-side consumers.

## Protocol

```text
EST1 <sequence> <state>\n
```

`state=1` means released/healthy; `state=0` means stopped. Consumers must
debounce frames, time out a silent stream, and fail closed on malformed data.
Keep the protocol constants synchronized with the C++ and Python readers.

The 100 Hz scheduler uses integer `time.monotonic_ns()` deadlines. Absolute
floating-point uptime loses interval precision on CircuitPython boards and
can turn the heartbeat into a busy loop after long uptime. The bench checker
accepts 80–120 Hz by default and rejects both missing and flooded heartbeats.
See [CircuitPython timing](https://docs.circuitpython.org/en/stable/shared-bindings/time/).

Runtime autoreload is disabled: even host filesystem metadata writes must not
restart the heartbeat or reset its sequence. Copying an update onto CIRCUITPY
therefore stages it; an explicit board reset activates it. Perform updates and
resets only with no active arm session or heartbeat consumer, then verify the
new bytes and repeat the heartbeat and physical stop acceptance checks.

## Wiring an existing button

Use a latching button with a voltage-free NC (normally closed) contact. The
firmware does not depend on the button's housing material or brand.

| Raspberry Pi Pico | Button / host connection |
| --- | --- |
| GP2, physical pin 4 | NC terminal |
| GND, physical pin 3 | COM terminal |
| Micro-USB connector | Host USB data cable; also powers the Pico |

For a two-terminal NC contact block, connect one terminal to GP2 and the
other to GND; polarity does not matter. Leave NO and any illumination
terminals unused. Do not connect the contact to VBUS, VSYS, 3V3, or an
external supply. The internal pull-up supplies the input bias.

These pin numbers are for the official Pico footprint: with components facing
you and USB at the top, pins 3 and 4 are the third and fourth pins down the
left edge. Check the [official Pico pinout](https://datasheets.raspberrypi.com/pico/Pico-R3-A4-Pinout.pdf)
against the actual board before wiring a clone or different form factor.

With USB disconnected, check continuity across the chosen contact: closed
when released, open when pressed and latched. A normally open-only button
does not implement this wiring contract. An open wire reports STOP; a short
across the contact can appear released and is not detected by this single
input circuit.

The onboard LED is solid for a closed contact and blinks at 10 Hz for an
open contact, including when USB writes are stalled. It indicates the local
input only, not host receipt or permission to move.

## Bench workflow

Flash the board using the official CircuitPython instructions, then run the
host parser tests with no arm attached. Copy both `boot.py` and `code.py` to
CIRCUITPY and explicitly reset the board to activate them; use the
CircuitPython build for the exact board variant. Before an update or bench
check, stop any active motion session and heartbeat consumer: the serial
stream must have exactly one reader.

On the configured e-stop owner, use the existing CLI bench check:

```sh
scripts/tatbot estop check --duration 3 --expect released
scripts/tatbot estop check --duration 3 --expect stopped
scripts/tatbot estop check --duration 10 --expect cycle
```

Run the first with the button released and the second with it pressed and
latched. During the third, press and release the button. With the contact
open by disconnecting either switch wire, repeat `--expect stopped`.
Unplugging USB must fail the check rather than report a healthy input.
The checker validates frames and rate; host timeout/stop response requires
its separate consumer acceptance check. Device paths and powered acceptance
evidence are deployment-specific.

The e-stop is a motion stop, not a promise that motor power is removed. See
the public [safety contract](../../docs/estop.md).
