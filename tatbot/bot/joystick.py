import logging
import os
import sys

import evdev
from evdev import InputDevice, categorize, ecodes

from tatbot.utils.log import get_logger

log = get_logger('bot.joystick', '🎮')

def find_joystick():
    devices = [InputDevice(path) for path in evdev.list_devices()]
    log.debug(f"Found {len(devices)} input devices:")
    for d in devices:
        try:
            caps = d.capabilities(verbose=True)
        except Exception as e:
            caps = f"[ERROR reading capabilities: {e}]"
        log.debug(f"  - {d.name} at {d.path} capabilities: {caps}")
        log.info(f"Device: {d.name} at {d.path} with capabilities: {caps}")
    # Only select the Atari joystick
    atari_name = "Retro Games LTD  Atari CX Wireless Controller"
    atari_device = None
    for d in devices:
        if d.name.strip() == atari_name:
            atari_device = d
            break
    if atari_device:
        log.debug(f"Atari joystick found: {atari_device.name} at {atari_device.path}")
        log.info(f"Atari joystick found: {atari_device.name} at {atari_device.path}")
        return atari_device
    else:
        log.error(f"Atari joystick ('{atari_name}') not found among input devices.")
        return None

def main():
    log.debug("Starting joystick main()")
    dev = find_joystick()
    if not dev:
        log.debug("No joystick-like device detected in main(). Exiting.")
        log.warning("No joystick-like device detected.")
        return

    log.debug(f"Using device: {dev.name} at {dev.path}")
    log.info(f"Using device: {dev.name} at {dev.path}")
    log.debug("Entering event read loop...")
    try:
        for event in dev.read_loop():
            log.debug(f"Event: type={event.type} code={event.code} value={event.value}")
            if event.type == ecodes.EV_ABS:
                absevent = categorize(event)
                log.debug(f"ABS event: {absevent.event.code}: {absevent.event.value}")
                log.info(f"{absevent.event.code}: {absevent.event.value}")
            elif event.type == ecodes.EV_KEY:
                keyevent = categorize(event)
                log.debug(f"KEY event: {keyevent.keycode}: {keyevent.keystate}")
                log.info(f"{keyevent.keycode}: {keyevent.keystate}")
    except Exception as e:
        log.error(f"Exception in event loop: {e}")

if __name__ == "__main__":
    main()
