import logging
import asyncio
import evdev
from evdev import InputDevice, ecodes
from tatbot.utils.log import get_logger, print_config, setup_log_with_config
from dataclasses import dataclass

log = get_logger('bot.joystick', '🎮')

RED_BUTTON_CODE = ecodes.BTN_TRIGGER  # Replace with actual code if needed
ATARI_NAME = "Retro Games LTD  Atari CX Wireless Controller"

@dataclass
class JoystickConfig:
    debug: bool = False
    device_name: str = ATARI_NAME
    queue_size: int = 1
    axis_threshold: float = 0.5  # Only send axis event if change > threshold

# Axis codes for Atari joystick (typical for many USB gamepads)
AXES = {
    'x': ecodes.ABS_X,
    'y': ecodes.ABS_Y,
}

# These are typical min/max for ABS_X/ABS_Y, but can vary by device
AXIS_MIN = 0
AXIS_MAX = 255

class JoystickListener:
    def __init__(self, queue: asyncio.Queue, device_name: str = ATARI_NAME, axis_threshold: float = 0.5):
        self.queue = queue
        self.device_name = device_name
        self.device = None
        self._task = None
        self._stop_event = asyncio.Event()
        self.axis_threshold = axis_threshold
        self._last_axis = {'x': None, 'y': None}

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
                elif event.type == ecodes.EV_ABS:
                    for axis_name, axis_code in AXES.items():
                        if event.code == axis_code:
                            # Normalize value to -1.0..1.0
                            norm = (event.value - AXIS_MIN) / (AXIS_MAX - AXIS_MIN) * 2 - 1
                            last = self._last_axis[axis_name]
                            if last is None or abs(norm - last) > self.axis_threshold:
                                self._last_axis[axis_name] = norm
                                try:
                                    self.queue.put_nowait({"axis": axis_name, "value": norm})
                                except asyncio.QueueFull:
                                    pass
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

def start_joystick_listener(queue: asyncio.Queue, device_name: str = ATARI_NAME, axis_threshold: float = 0.5):
    listener = JoystickListener(queue, device_name, axis_threshold)
    listener.start()
    return listener

# Script entry point for testing with project style
if __name__ == "__main__":
    args = setup_log_with_config(JoystickConfig)
    print_config(args)
    if args.debug:
        log.setLevel(logging.DEBUG)

    async def main():
        queue = asyncio.Queue(maxsize=args.queue_size)
        listener = start_joystick_listener(queue, device_name=args.device_name, axis_threshold=args.axis_threshold)
        log.info(f"Listening for red button and joystick axes on device '{args.device_name}' (queue size {args.queue_size}, axis threshold {args.axis_threshold})...")
        try:
            while True:
                msg = await queue.get()
                if msg == "red_button":
                    log.info("Red button pressed!")
                elif isinstance(msg, dict) and "axis" in msg:
                    log.info(f"Axis {msg['axis']} moved to {msg['value']:.2f}")
        except KeyboardInterrupt:
            log.info("Exiting...")
            listener.stop()

    asyncio.run(main())
