import logging

import evdev
from evdev import InputDevice, categorize, ecodes

from tatbot.utils.log import get_logger

log = get_logger('bot.joystick', '🎮')

def find_joystick():
    devices = [InputDevice(path) for path in evdev.list_devices()]
    for d in devices:
        caps = d.capabilities(verbose=True)
        # check for joystick/gamepad
        if 'ABS_X' in d.capabilities() or 'BTN_JOYSTICK' in d.capabilities():
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
