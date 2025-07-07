import logging

import evdev
from evdev import InputDevice, categorize, ecodes

from tatbot.utils.log import get_logger

log = get_logger('bot.joystick', '🎮')

def find_joystick():
    devices = [InputDevice(path) for path in evdev.list_devices()]
    for d in devices:
        caps = d.capabilities()
        log.info(f"Device: {d.name} at {d.path} with capabilities: {caps}")
        # Check for joystick/gamepad by looking for ABS_X or BTN_JOYSTICK codes
        if ecodes.ABS_X in caps or ecodes.BTN_JOYSTICK in caps:
            log.info(f"Joystick found: {d.name} at {d.path}")
            return d
    log.warning("No joystick-like device detected.")
    return None

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
