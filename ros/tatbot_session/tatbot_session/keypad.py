"""`ros2 run tatbot_session keypad`: the operator's numpad (stack.yaml `keypad`) as names.KEYS presses on
/tatbot/keys/<arm>. Raw evdev, whose keypad codes ignore NumLock, grabbed so no console or desktop sees the keys;
a held key steps once; an absent pad is waited for. ros/README.md 4.4, the pen trim.
"""
from __future__ import annotations

import fcntl
import os
import select
import struct
import time

EVENT = struct.Struct("llHHi")   # struct input_event on a 64-bit kernel: timeval, type, code, value
EVIOCGRAB = 0x40044590           # _IOW('E', 0x90, int)
# KEY_KP8 and KEY_UP, KEY_KP2 and KEY_DOWN, KEY_KP5, KEY_KPENTER and KEY_ENTER (linux/input-event-codes.h)
KEYS = {72: "up", 103: "up", 80: "down", 108: "down", 76: "reset", 96: "enter", 28: "enter"}


def presses(data: bytes) -> list[tuple[int, str | None]]:
    """(code, key or None) for each key press (EV_KEY, value 1) in a read of input events."""
    whole = data[: len(data) - len(data) % EVENT.size]
    return [(code, KEYS.get(code)) for _s, _us, kind, code, value in EVENT.iter_unpack(whole) if (kind, value) == (1, 1)]


def main(argv=None) -> int:
    import rclpy
    from rclpy.node import Node
    from std_msgs.msg import String
    from tatbot_description import names

    from tatbot_session import config

    rclpy.init(args=argv)
    node = Node(names.KEYPAD_NODE)
    node.declare_parameter("stack", "")
    cfg = config.load_stack(node.get_parameter("stack").value or None)["keypad"]
    pub, log, said = node.create_publisher(String, names.keys_topic(cfg["arm"]), 10), node.get_logger(), ""
    try:   # a stop signal can arrive anywhere, the wait for an absent pad included
        while rclpy.ok():
            try:
                fd = os.open(cfg["device"], os.O_RDONLY)
                try:
                    fcntl.ioctl(fd, EVIOCGRAB, 1)
                    log.info(f"{cfg['device']}: 8 up, 2 down, 5 reset, Enter continue for the {cfg['arm']} arm")
                    said = ""
                    while rclpy.ok():
                        ready = select.select([fd], [], [], 0.5)[0]
                        for code, key in presses(os.read(fd, EVENT.size * 64) if ready else b""):
                            if key:
                                pub.publish(String(data=key))
                            else:
                                log.info(f"key code {code}: not 8, 2, 5 or Enter")
                finally:
                    os.close(fd)
            except OSError as exc:   # absent, unplugged, or another reader holds the grab
                if said != exc.strerror:
                    log.warning(f"{cfg['device']}: {exc.strerror}; waiting for the keypad")
                said = exc.strerror
                time.sleep(1.0)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.try_shutdown()
    return 0
