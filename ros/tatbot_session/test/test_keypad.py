"""The numpad's evdev reports as keys: presses only (a held key steps once), keypad codes and arrows, other codes
kept for the log; a partial event at the end of a read is left for the next."""
from tatbot_session.keypad import EVENT, presses


def _event(kind, code, value):
    return EVENT.pack(0, 0, kind, code, value)


def test_each_key_press_is_one_key():
    held_eight = [_event(4, 4, 458848), _event(1, 72, 1), _event(0, 0, 0), _event(1, 72, 2), _event(1, 72, 0)]
    others = [_event(1, code, 1) for code in (80, 108, 103, 76, 96, 28, 69)]
    data = b"".join(held_eight + others) + b"\0" * 5
    assert presses(data) == [(72, "up"), (80, "down"), (108, "down"), (103, "up"), (76, "reset"), (96, "enter"),
                             (28, "enter"), (69, None)]
