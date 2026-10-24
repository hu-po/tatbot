"""The session's machine switch against the palette Pi's machine relay (ros/tatbot_estop_relay --machine), both
over loopback UDP, the Pi's output line emulated. No ROS."""
import importlib.util
import itertools
import socket
import threading
import time
from pathlib import Path

import pytest
from tatbot_session import machine

RELAY = Path(__file__).resolve().parents[2] / "tatbot_estop_relay" / "tatbot_estop_relay.py"
spec = importlib.util.spec_from_file_location("tatbot_estop_relay", RELAY)
relay = importlib.util.module_from_spec(spec)
spec.loader.exec_module(relay)


def _free_ports(n):
    socks = [socket.socket(socket.AF_INET, socket.SOCK_DGRAM) for _ in range(n)]
    for sock in socks:
        sock.bind(("127.0.0.1", 0))
    ports = [sock.getsockname()[1] for sock in socks]
    for sock in socks:
        sock.close()
    return ports


@pytest.fixture
def pi():
    """The machine relay on loopback with an emulated line, and an e-stop relay feeding it at 100 Hz."""
    listen, estop_port = _free_ports(2)
    line, stop, released, button_on = [], threading.Event(), threading.Event(), threading.Event()
    released.set()
    button_on.set()
    threads = [threading.Thread(target=relay.run_machine, args=("GPIO22", listen, "127.0.0.1", estop_port, 0.2),
                                kwargs={"stop": stop.is_set, "open_line": lambda name: (line.append, lambda: None)},
                                daemon=True)]

    def button():
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock:
            for seq in itertools.count(1):
                if stop.is_set():
                    return
                if button_on.is_set():
                    sock.sendto(b"EST1 %d %d\n" % (seq, released.is_set()), ("127.0.0.1", estop_port))
                time.sleep(0.01)

    threads.append(threading.Thread(target=button, daemon=True))
    for thread in threads:
        thread.start()
    time.sleep(0.05)
    switch = machine.PiSwitch("127.0.0.1", listen, 0.02)
    yield switch, line, released, button_on, stop
    switch.close()
    stop.set()
    for thread in threads:
        thread.join(2)


def test_the_switch_runs_the_machine_only_while_asked_and_released_and_never_restarts_it_by_itself(pi):
    switch, line, released, button_on, stop = pi
    assert switch.wait(False, 0.5) and not any(line) and machine.reply(switch)[0] is False
    assert switch.wait(True, 0.5) and line[-1] is True
    powered, age = machine.reply(switch)
    assert powered and 0.0 <= age < 0.2
    released.clear()   # the e-stop pressed: the Pi cuts the machine by itself
    time.sleep(0.1)
    assert line[-1] is False and switch.why_off() == "the e-stop is pressed" and not switch.wait(True, 0.1)
    released.set()   # released, still asked on: it stays off until asked off and on again
    time.sleep(0.1)
    assert line[-1] is False and "holds it off" in switch.why_off()
    assert switch.wait(False, 0.5) and switch.wait(True, 0.5) and line[-1] is True
    button_on.clear()   # the e-stop relay went silent
    time.sleep(0.3)
    assert line[-1] is False and "hears no e-stop" in switch.why_off()
    button_on.set()
    assert switch.wait(False, 0.5) and switch.wait(True, 0.5)
    switch._stop.set()   # the session dies with the machine asked on
    time.sleep(0.35)
    assert line[-1] is False and "does not answer" in switch.why_off()
    assert machine.reply(switch)[1] >= 0.3   # the newest reply is the one before the silence, and it ages


def test_the_stack_names_the_switch_and_its_arm():
    assert machine.from_stack({"machine": {"switch": "none", "arm": "right"}}, "right") is None
    assert machine.from_stack({"machine": {"switch": "sim", "arm": "right"}}, "left") is None
    assert machine.from_stack({}, "right") is None
    sim = machine.from_stack({"machine": {"switch": "sim", "arm": "right"}}, "right")
    assert sim.is_off()
    sim.set(True)
    assert not sim.is_off()
    assert sim.wait(True, 0.0) and sim.on
    for bad in ({"switch": "relay", "arm": "right"}, {"switch": "pi", "arm": "right", "addr": ""}):
        with pytest.raises(ValueError):
            machine.from_stack({"machine": bad}, "right")


@pytest.mark.parametrize('condition', ['off', 'asked-on', 'reported-on', 'powered', 'old-sequence', 'stale', 'missing'])
def test_service_off_evidence_requires_a_fresh_reply_to_the_current_off_command(condition):
    switch = object.__new__(machine.PiSwitch)
    switch._cv = threading.Condition()
    switch.period_s, switch._since = 0.02, 5
    switch.on = condition == 'asked-on'
    switch.answer = None if condition == 'missing' else machine.Answer(
        4 if condition == 'old-sequence' else 5, condition == 'reported-on', condition == 'powered',
        machine.ESTOP_RELEASED, time.monotonic() - (1.0 if condition == 'stale' else 0.0))
    assert switch.is_off() is (condition == 'off')


def test_the_safety_state_shows_the_newest_reply_and_its_age_or_none():
    assert machine.reply(None) == (False, -1.0)   # no switch configured for this arm
    switch = object.__new__(machine.PiSwitch)
    switch.answer = None
    assert machine.reply(switch) == (False, -1.0)   # configured, not answered yet
    switch.answer = machine.Answer(7, True, True, machine.ESTOP_RELEASED, time.monotonic() - 0.5)
    powered, age = machine.reply(switch)
    assert powered and age == pytest.approx(0.5, abs=0.05)
    switch.answer = machine.Answer(8, True, False, machine.ESTOP_PRESSED, time.monotonic())   # cut by the e-stop
    assert machine.reply(switch)[0] is False
    sim = machine.SimSwitch()
    assert machine.reply(sim)[0] is False and 0.0 <= machine.reply(sim)[1] < 0.01
    sim.set(True)
    assert machine.reply(sim)[0] is True and sim.is_off() is False
