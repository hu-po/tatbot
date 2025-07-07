import logging
import os
import sys

import evdev
from evdev import InputDevice, categorize, ecodes

from tatbot.utils.log import get_logger

log = get_logger('bot.joystick', '🎮')

def find_joystick():
    devices = [InputDevice(path) for path in evdev.list_devices()]
    print(f"[DEBUG] Found {len(devices)} input devices:")
    for d in devices:
        try:
            caps = d.capabilities(verbose=True)
        except Exception as e:
            caps = f"[ERROR reading capabilities: {e}]"
        print(f"  - {d.name} at {d.path} capabilities: {caps}")
        log.info(f"Device: {d.name} at {d.path} with capabilities: {caps}")
    candidates = []
    for d in devices:
        try:
            caps = d.capabilities()
        except Exception as e:
            log.warning(f"Could not read capabilities for {d.path}: {e}")
            continue
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
            print(f"[DEBUG] /dev/input/event10 name: {dev10.name}")
            print(f"[DEBUG] /dev/input/event10 capabilities: {dev10.capabilities(verbose=True)}")
            log.info(f"[DEBUG] /dev/input/event10 name: {dev10.name}")
            log.info(f"[DEBUG] /dev/input/event10 capabilities: {dev10.capabilities(verbose=True)}")
        except Exception as e:
            log.warning(f"[DEBUG] Could not open /dev/input/event10: {e}")
    if not candidates:
        print("[DEBUG] No joystick-like device detected.")
        log.warning("No joystick-like device detected.")
        return None
    if len(candidates) == 1:
        d = candidates[0]
        print(f"[DEBUG] Joystick found: {d.name} at {d.path}")
        log.info(f"Joystick found: {d.name} at {d.path}")
        return d
    # Multiple candidates: let user select
    print("[DEBUG] Multiple joystick-like devices found:")
    log.info("Multiple joystick-like devices found:")
    for idx, d in enumerate(candidates):
        print(f"  [{idx}] {d.name} at {d.path}")
        log.info(f"[{idx}] {d.name} at {d.path}")
    # For now, just pick the first one (could add input() for selection if interactive)
    d = candidates[0]
    print(f"[DEBUG] Using first candidate: {d.name} at {d.path}")
    log.info(f"Using first candidate: {d.name} at {d.path}")
    return d

def main():
    print("[DEBUG] Starting joystick main()")
    dev = find_joystick()
    if not dev:
        print("[DEBUG] No joystick-like device detected in main(). Exiting.")
        log.warning("No joystick-like device detected.")
        return

    print(f"[DEBUG] Using device: {dev.name} at {dev.path}")
    log.info(f"Using device: {dev.name} at {dev.path}")
    print("[DEBUG] Entering event read loop...")
    try:
        for event in dev.read_loop():
            print(f"[DEBUG] Event: type={event.type} code={event.code} value={event.value}")
            if event.type == ecodes.EV_ABS:
                absevent = categorize(event)
                print(f"[DEBUG] ABS event: {absevent.event.code}: {absevent.event.value}")
                log.info(f"{absevent.event.code}: {absevent.event.value}")
            elif event.type == ecodes.EV_KEY:
                keyevent = categorize(event)
                print(f"[DEBUG] KEY event: {keyevent.keycode}: {keyevent.keystate}")
                log.info(f"{keyevent.keycode}: {keyevent.keystate}")
    except Exception as e:
        print(f"[ERROR] Exception in event loop: {e}")
        log.error(f"Exception in event loop: {e}")

if __name__ == "__main__":
    main()
