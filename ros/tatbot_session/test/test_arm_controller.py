"""ArmIO goals whose acceptance comes late or never, with deferred ROS transport replies."""
from concurrent.futures import Future
from threading import Thread
from types import SimpleNamespace

import numpy as np
import pytest
from tatbot_session.arm import ArmIO


class Handle:
    accepted = True

    def __init__(self):
        self.result = Future()
        self.cancels = 0

    def get_result_async(self):
        return self.result

    def cancel_goal_async(self):
        self.cancels += 1
        return SimpleNamespace(goals_canceling=[object()])


@pytest.fixture
def controller():
    io = ArmIO.__new__(ArmIO)
    sent, asked = [], []
    io.arm, io.safety, io.last_command, io.goals = 'right', {}, np.ones(7), None
    io.goal = lambda traj, first=0: traj
    io.clearance = lambda arm, q: asked.append(q) or None
    io.jtc = SimpleNamespace(wait_for_server=lambda **kwargs: True,
                             send_goal_async=lambda goal: sent.append(Future()) or sent[-1])
    io.node = SimpleNamespace(get_logger=lambda: SimpleNamespace(warning=lambda text: None))
    traj = SimpleNamespace(t=np.array([.0, .001]), q=np.zeros((2, 7)))
    return SimpleNamespace(io=io, sent=sent, asked=asked, traj=traj)


def run(r, **kwargs):
    result = []
    thread = Thread(target=lambda: result.append(r.io.execute(r.traj, **kwargs)))
    thread.start()
    return thread, result


def pending(r):
    import time

    deadline = time.monotonic() + 1.
    while (not r.sent or r.sent[-1].done()) and time.monotonic() < deadline:
        time.sleep(.001)
    assert r.sent and not r.sent[-1].done()
    return r.sent[-1]


def accepted(r):
    future = pending(r)
    handle = Handle()
    future.set_result(handle)
    return handle


def test_stop_before_acceptance_cancels_the_late_accepted_goal(controller):
    r = controller
    thread, result = run(r, should_stop=lambda: bool(r.sent))
    thread.join(3.)
    assert not thread.is_alive() and result[0][0] == 'cancelled'
    assert accepted(r).cancels == 1


def test_no_acceptance_fails_after_five_seconds_and_cancels_a_later_one(controller, monkeypatch):
    from test_ports import StepClock

    r = controller
    r.io.clock = StepClock()
    send = r.io._send

    def clocked_send(*args, **kwargs):
        done, box = send(*args, **kwargs)

        def wait(seconds):
            if not done.is_set():
                r.io.clock.sleep(seconds)
            return done.is_set()

        return SimpleNamespace(wait=wait), box

    monkeypatch.setattr(r.io, "_send", clocked_send)
    thread, result = run(r)
    thread.join(8.)
    assert not thread.is_alive() and result[0] == ('failed', 0, 'no goal acceptance within 5 s')
    assert 7.0 <= r.io.clock.now() <= 7.1   # the acceptance deadline and two-second cancellation grace both ran
    assert accepted(r).cancels == 1


def test_a_clearance_refusal_uses_the_real_arm_entry_point(controller, monkeypatch):
    r = controller
    warned, servers = [], []
    r.io.jtc.wait_for_server = lambda **kw: servers.append(kw) or True
    r.io.node.get_logger = lambda: SimpleNamespace(warning=warned.append)
    monkeypatch.setattr(r.io, "_run", lambda *a: ("ok", 1, "done"))
    r.io.clearance = lambda arm, q: f"{arm} {len(q)} rows too near"
    assert r.io.execute(r.traj) == ("failed", 0, "right 2 rows too near")
    assert warned == ["right 2 rows too near"] and servers == r.sent == []
    r.io.clearance = lambda arm, q: None
    assert r.io.execute(r.traj) == ("ok", 1, "done")


@pytest.mark.parametrize("seconds, sec, nanosec", [(14.9999999998, 15, 0), (15.0, 15, 0), (0.01, 0, 10_000_000),
                                                     (2.5, 2, 500_000_000)])
def test_a_knot_a_hair_under_a_second_does_not_wrap_back(seconds, sec, nanosec):
    """A 15 s jog's last knot, 14.9999999998 s, went out as 14.0 s, behind the knot before it (2026-10-12)."""
    from tatbot_session.arm import _duration

    d = _duration(seconds)
    assert (d.sec, d.nanosec) == (sec, nanosec)


def test_every_safety_command_carries_the_guard_level_asked_last():
    """GpioCommandController applies only the newest message at each 10 ms update: the snag guard's disarm followed
    at once by a land request was overwritten and the guard stayed armed into the next wake (2026-10-23)."""
    io = ArmIO.__new__(ArmIO)
    published = []
    io.arm = 'right'
    io.node = SimpleNamespace(get_clock=lambda: SimpleNamespace(now=lambda: SimpleNamespace(to_msg=lambda: None)))
    io.gpio = SimpleNamespace(publish=published.append)

    def newest():   # what the controller applies from the one message it holds
        values = published[-1].interface_values[0]
        return dict(zip(values.interface_names, values.values, strict=True))

    io.write(land=4.0)
    assert newest() == {'guard_mode': 0.0, 'land': 4.0}
    io.write(guard_mode=3)
    io.write(guard_mode=0)
    io.write(land=5.0)        # on the disarm's heels, as _land_now sends it
    assert newest() == {'guard_mode': 0.0, 'land': 5.0}
    io.write(guard_mode=3)
    io.write(unlatch=6.0)
    assert newest() == {'guard_mode': 3.0, 'unlatch': 6.0}
