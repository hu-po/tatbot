# tatbot e-stop — CircuitPython code.py for Raspberry Pi Pico (RP2040).
#
# The button is the Keymoo IP65 station's single NC (normally closed) contact
# wired between GP2 and GND. GP2 uses the internal pull-up:
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
#   solid on    = released, heartbeat flowing (normal)
#   fast blink  = pressed / circuit open

import time

import board
import digitalio
import usb_cdc

FRAME_PERIOD_S = 0.010  # 100 Hz

nc_pin = digitalio.DigitalInOut(board.GP2)
nc_pin.direction = digitalio.Direction.INPUT
nc_pin.pull = digitalio.Pull.UP

led = digitalio.DigitalInOut(board.LED)
led.direction = digitalio.Direction.OUTPUT

serial = usb_cdc.data
serial.write_timeout = 0  # never let a disconnected/full host stall the loop
seq = 0
pending = b""
next_frame = time.monotonic()

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

    led.value = released or (seq % 10 < 5)  # solid when OK, 10 Hz blink when pressed

    next_frame += FRAME_PERIOD_S
    delay = next_frame - time.monotonic()
    if delay > 0:
        time.sleep(delay)
    else:
        next_frame = time.monotonic()  # fell behind; don't burst to catch up
