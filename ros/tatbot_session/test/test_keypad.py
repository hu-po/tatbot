"""The keypad node's datagrams as keys: one `TBK1 <key>` frame of a key the session takes, nothing else."""
from tatbot_description import names
from tatbot_session.keypad import key_of, sender_ok


def test_each_frame_names_one_key():
    assert [key_of(f"TBK1 {key}\n".encode(), names.KEYS) for key in names.KEYS] == list(names.KEYS)


def test_anything_else_is_no_key():
    for data in (b"", b"TBK1", b"TBK1 \n", b"TBK1 land\n", b"TBK2 dip\n", b"EST1 3 1\n", b"\xff\xfe dip"):
        assert key_of(data, names.KEYS) is None


def test_keys_come_from_the_station_alone_else_this_host():
    assert sender_ok({"port": 7644, "from_addr": "192.0.2.4"}) == "192.0.2.4"
    assert sender_ok({"port": 7644, "from_addr": ""}) == sender_ok({"port": 7644}) == "127.0.0.1"


def test_the_station_keyboard_sends_every_key_the_session_takes_and_no_other():
    import importlib.util
    from pathlib import Path

    script = Path(__file__).resolve().parents[2] / "tatbot_keys_relay" / "tatbot_keys_relay.py"
    spec = importlib.util.spec_from_file_location("tatbot_keys_relay", script)
    relay = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(relay)
    assert set(relay.KEYS.values()) == set(names.KEYS)
