# tatbot e-stop — CircuitPython code.py for Raspberry Pi Pico (RP2040).
#
# Use the existing button's NC (normally closed), voltage-free contact
# between GP2 (physical pin 4) and GND (physical pin 3). For a COM/NC/NO
# switch, use COM and NC; leave NO unused. GP2 uses the internal pull-up:
#
#   button released  -> NC contact closed -> GP2 reads LOW  -> state 1 (OK)
#   button pressed   -> NC contact open   -> GP2 reads HIGH -> state 0 (STOP)
#   any broken wire  -> circuit open      -> GP2 reads HIGH -> state 0 (STOP)
#
# Frames go out the CDC data channel at 100 Hz, newline-terminated ASCII:
#
#   EST1 <seq> <state>\n      e.g.  "EST1 4711 1\n"
#
# The stream itself is the heartbeat: the host (wxai_teleop) e-stops when no
# valid frame arrives for its timeout, so unplugging the cable, killing this
# firmware, or wedging the Pico all read as STOP. seq is a monotonically
# increasing frame counter so the host can spot a rebooted or wedged sender.
#
# The onboard LED mirrors the state for at-a-glance confidence:
#   solid on    = contact closed (not proof the host receives heartbeats)
#   fast blink  = pressed / circuit open

import time

import board
import digitalio
import supervisor
import usb_cdc

# The host may write FAT metadata while inspecting CIRCUITPY. A safety sensor
# must not restart (or reset its sequence) because of host filesystem traffic.
# Intentional firmware updates require an explicit board reset while unowned.
supervisor.runtime.autoreload = False

FRAME_PERIOD_NS = 10_000_000  # 100 Hz
LED_HALF_PERIOD_NS = 50_000_000  # 10 Hz blink, independent of USB progress

nc_pin = digitalio.DigitalInOut(board.GP2)
nc_pin.direction = digitalio.Direction.INPUT
nc_pin.pull = digitalio.Pull.UP

led = digitalio.DigitalInOut(board.LED)
led.direction = digitalio.Direction.OUTPUT

serial = usb_cdc.data
serial.write_timeout = 0  # never let a disconnected/full host stall the loop
seq = 0
pending = b""
# CircuitPython floats lose sub-frame precision with uptime. At sufficiently
# large values adding 0.010 rounds back to the same timestamp and floods USB.
# Keep absolute deadlines in integer nanoseconds; convert only the small sleep.
next_frame_ns = time.monotonic_ns()

while True:
    # NC to GND: LOW = circuit closed = released/OK.
    released = not nc_pin.value
    state = 1 if released else 0

    # Host may not be reading yet (or ever); never block on a full buffer.
    # A nonblocking CDC write may accept only part of a frame, so retain the
    # unsent suffix and finish it before starting another frame. Otherwise two
    # writes can merge into malformed input such as "EST1 12EST1 13 0\n".
    if not pending:
        pending = b"EST1 %d %d\n" % (seq, state)
        seq += 1
    try:  # noqa: SIM105 — CircuitPython has no contextlib
        written = serial.write(pending)
        if written:
            pending = pending[written:]
    except Exception:
        pass

    led.value = released or (time.monotonic_ns() // LED_HALF_PERIOD_NS % 2 == 0)

    next_frame_ns += FRAME_PERIOD_NS
    delay_ns = next_frame_ns - time.monotonic_ns()
    if delay_ns > 0:
        time.sleep(delay_ns / 1_000_000_000)
    else:
        next_frame_ns = time.monotonic_ns()  # fell behind; don't burst to catch up
