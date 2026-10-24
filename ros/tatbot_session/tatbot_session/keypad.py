"""`ros2 run tatbot_session keypad`: the operator's drawing keys (stack.yaml `keypad`) as names.KEYS presses on
/tatbot/keys/<arm>. The keys are pressed on the station keyboard, whose relay (ros/tatbot_keys_relay) sends each
press as one `TBK1 <key>` UDP datagram to `keypad.port` here, which takes them from `keypad.from_addr` alone (the
station's `lan`, or this host when unset), as the e-stop reader takes frames from its relay alone. ros/README.md
4.4, the pen trim, and 6, the dip scale.
"""
from __future__ import annotations

import socket

MAX_FRAME = 64


def key_of(data: bytes, keys) -> str | None:
    """The key a datagram names, or None for anything that is not one `TBK1 <key>` frame of a known key."""
    magic, _, key = data.decode("ascii", "replace").strip().partition(" ")
    return key if magic == "TBK1" and key in keys else None


def sender_ok(cfg: dict) -> str:
    """The one address keys are taken from: keypad.from_addr (the station's `lan`), else this host."""
    return cfg.get("from_addr") or "127.0.0.1"


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
    pub, log = node.create_publisher(String, names.keys_topic(cfg["arm"]), 10), node.get_logger()
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind(("0.0.0.0", int(cfg["port"])))
    sock.settimeout(0.5)
    station = sender_ok(cfg)
    log.info(f"UDP :{cfg['port']}: the station keyboard's keys for the {cfg['arm']} arm, from {station} only")
    try:   # a stop signal can arrive anywhere
        while rclpy.ok():
            try:
                data, sender = sock.recvfrom(MAX_FRAME)
            except TimeoutError:
                continue
            if sender[0] != station:   # anyone else on the rig LAN or the tailnet
                log.warning(f"{sender[0]} is not the station ({station}): ignored", throttle_duration_sec=5.0)
                continue
            key = key_of(data, names.KEYS)
            if key:
                pub.publish(String(data=key))
                log.info(f"{key} from {sender[0]}")
            else:
                log.warning(f"{sender[0]}: not a key frame: {data[:MAX_FRAME]!r}")
    except KeyboardInterrupt:
        pass
    sock.close()
    node.destroy_node()
    rclpy.try_shutdown()
    return 0
