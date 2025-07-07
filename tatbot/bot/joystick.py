import logging
import asyncio
import evdev
from evdev import InputDevice, ecodes
from tatbot.utils.log import get_logger, print_config, setup_log_with_config
from dataclasses import dataclass
import threading
import queue as thread_queue

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
                if event.type == ecodes.EV_KEY:
                    log.debug(f"Key event: code={event.code}, value={event.value}")
                    if event.value == 1:  # 1 = key down
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
    log.info("Starting joystick listener...")
    listener = JoystickListener(queue, device_name, axis_threshold)
    listener.start()
    return listener

def start_joystick_listener_threaded(queue, device_name: str = ATARI_NAME, axis_threshold: float = 0.5):
    """
    Starts the joystick listener in a background thread with its own event loop.
    The queue should be a thread-safe queue.Queue (not asyncio.Queue).
    Example usage in synchronous code:
        import queue as thread_queue
        joystick_queue = thread_queue.Queue(maxsize=1)
        joystick_thread = start_joystick_listener_threaded(joystick_queue)
        # In your main loop:
        if not joystick_queue.empty():
            msg = joystick_queue.get_nowait()
            # handle msg
    """
    def run_in_thread():
        import asyncio

        async def async_listener():
            internal_queue = asyncio.Queue(maxsize=1)
            listener = JoystickListener(internal_queue, device_name, axis_threshold)
            listener.start()
            while True:
                msg = await internal_queue.get()
                queue.put(msg)  # Forward to thread-safe queue

        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            loop.run_until_complete(async_listener())
        finally:
            loop.close()

    t = threading.Thread(target=run_in_thread, daemon=True)
    t.start()
    return t  # Optionally return the thread object for control

# --- Module-level polling API ---
_joystick_queue = None
_joystick_thread = None

def start_joystick_listener_polling(device_name: str = ATARI_NAME, axis_threshold: float = 0.5):
    """
    Starts the joystick listener in a background thread with internal queue management.
    Call this once at the start of your program.
    """
    global _joystick_queue, _joystick_thread
    if _joystick_queue is None:
        _joystick_queue = thread_queue.Queue(maxsize=1)
        _joystick_thread = start_joystick_listener_threaded(_joystick_queue, device_name, axis_threshold)

def get_joystick_event():
    """
    Returns the next joystick event if available, else None.
    Usage:
        event = get_joystick_event()
        if event is not None:
            # handle event
    """
    global _joystick_queue
    if _joystick_queue is not None and not _joystick_queue.empty():
        return _joystick_queue.get_nowait()
    return None

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
