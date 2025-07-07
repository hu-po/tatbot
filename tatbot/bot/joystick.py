import logging
import os

import evdev
from evdev import InputDevice, categorize, ecodes

from tatbot.utils.log import get_logger

log = get_logger('bot.joystick', '🎮')

def find_joystick():
    devices = [InputDevice(path) for path in evdev.list_devices()]
    candidates = []
    for d in devices:
        caps = d.capabilities()
        log.info(f"Device: {d.name} at {d.path} with capabilities: {caps}")
        # Broaden detection: check for common joystick/gamepad codes
        has_joystick = any(code in caps for code in [
            ecodes.ABS_X, ecodes.BTN_JOYSTICK, ecodes.BTN_GAMEPAD, ecodes.BTN_TRIGGER
        ])
        if has_joystick:
            candidates.append(d)
    # Special: print capabilities for /dev/input/event10 if it exists
    event10_path = '/dev/input/event10'
    if os.path.exists(event10_path):
        try:
            dev10 = InputDevice(event10_path)
            log.info(f"[DEBUG] /dev/input/event10 name: {dev10.name}")
            log.info(f"[DEBUG] /dev/input/event10 capabilities: {dev10.capabilities(verbose=True)}")
        except Exception as e:
            log.warning(f"[DEBUG] Could not open /dev/input/event10: {e}")
    if not candidates:
        log.warning("No joystick-like device detected.")
        return None
    if len(candidates) == 1:
        d = candidates[0]
        log.info(f"Joystick found: {d.name} at {d.path}")
        return d
    # Multiple candidates: let user select
    log.info("Multiple joystick-like devices found:")
    for idx, d in enumerate(candidates):
        log.info(f"[{idx}] {d.name} at {d.path}")
    # For now, just pick the first one (could add input() for selection if interactive)
    d = candidates[0]
    log.info(f"Using first candidate: {d.name} at {d.path}")
    return d

def main():
    dev = find_joystick()
    if not dev:
        log.warning("No joystick-like device detected.")
        return

    log.info(f"Using device: {dev.name} at {dev.path}")
    for event in dev.read_loop():
        if event.type == ecodes.EV_ABS:
            absevent = categorize(event)
            log.info(f"{absevent.event.code}: {absevent.event.value}")
        elif event.type == ecodes.EV_KEY:
            keyevent = categorize(event)
            log.info(f"{keyevent.keycode}: {keyevent.keystate}")

if __name__ == "__main__":
    main()
