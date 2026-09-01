# tatbot e-stop — CircuitPython boot.py for Raspberry Pi Pico (RP2040).
#
# Runs once at USB enumeration, before code.py. Two jobs:
#
# 1. Present a recognizable USB identity so the host's udev rule
#    (config/udev/99-tatbot-estop.rules) can pin /dev/tatbot-estop no matter
#    which physical port the box is plugged into. udev munges the product
#    string into ID_MODEL with '_' separators, so use underscores here.
#
# 2. Disable the CircuitPython REPL console CDC channel and enable only the
#    data channel. The device then enumerates as exactly ONE serial port that
#    carries nothing but heartbeat frames — no REPL banner, no traceback
#    noise, no second ttyACM to guess between.
#
# The CIRCUITPY mass-storage drive stays enabled so firmware updates remain
# drag-and-drop. If code.py ever crashes, CircuitPython idles in safe mode,
# frames stop, and the host treats the silence as an e-stop — fail-safe.

import supervisor
import usb_cdc

supervisor.set_usb_identification(
    manufacturer="tatbot",
    product="tatbot_estop",
)
usb_cdc.enable(console=False, data=True)
