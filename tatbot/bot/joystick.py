import logging
import asyncio
import evdev
from evdev import InputDevice, ecodes
from tatbot.utils.log import get_logger
from dataclasses import dataclass

log = get_logger('bot.joystick', '🎮')

RED_BUTTON_CODE = ecodes.BTN_TRIGGER  # Replace with actual code if needed
ATARI_NAME = "Retro Games LTD  Atari CX Wireless Controller"

@dataclass
class JoystickConfig:
    device_name: str = ATARI_NAME
    queue_size: int = 1

class JoystickListener:
    def __init__(self, queue: asyncio.Queue, device_name: str = ATARI_NAME):
        self.queue = queue
        self.device_name = device_name
        self.device = None
        self._task = None
        self._stop_event = asyncio.Event()

    def find_joystick(self):
        devices = [InputDevice(path) for path in evdev.list_devices()]
        for d in devices:
            if d.name.strip() == self.device_name:
                log.info(f"Joystick found: {d.name} at {d.path}")
                return d
        log.error(f"Joystick ('{self.device_name}') not found among input devices.")
        return None

    async def run(self):
        self.device = self.find_joystick()
        if not self.device:
            log.error("No joystick device found. JoystickListener exiting.")
            return
        try:
            async for event in self.device.async_read_loop():
                if self._stop_event.is_set():
                    break
                if event.type == ecodes.EV_KEY and event.value == 1:  # 1 = key down
                    if event.code == RED_BUTTON_CODE:
                        try:
                            self.queue.put_nowait("red_button")
                        except asyncio.QueueFull:
                            pass  # Drop if queue is full
        except Exception as e:
            log.error(f"Exception in joystick event loop: {e}")

    def start(self):
        self._task = asyncio.create_task(self.run())
        return self._task

    def stop(self):
        self._stop_event.set()
        if self._task:
            self._task.cancel()

# Helper function for convenience

def start_joystick_listener(queue: asyncio.Queue, device_name: str = ATARI_NAME):
    listener = JoystickListener(queue, device_name)
    listener.start()
    return listener

# Script entry point for testing
if __name__ == "__main__":
    import sys
    import argparse

    parser = argparse.ArgumentParser(description="Test JoystickListener for red button press.")
    parser.add_argument('--device_name', type=str, default=ATARI_NAME, help='Joystick device name')
    parser.add_argument('--queue_size', type=int, default=1, help='Asyncio queue size')
    args = parser.parse_args()

    config = JoystickConfig(device_name=args.device_name, queue_size=args.queue_size)

    async def main():
        queue = asyncio.Queue(maxsize=config.queue_size)
        listener = start_joystick_listener(queue, device_name=config.device_name)
        print(f"Listening for red button on device '{config.device_name}' (queue size {config.queue_size})...")
        try:
            while True:
                msg = await queue.get()
                if msg == "red_button":
                    print("Red button pressed!")
        except KeyboardInterrupt:
            print("Exiting...")
            listener.stop()

    asyncio.run(main())
