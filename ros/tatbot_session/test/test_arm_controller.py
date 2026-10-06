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
    sent, released, asked = [], [], []
    io.arm, io.safety, io.last_command, io.goals = 'right', {}, np.ones(7), None
    io.goal = lambda traj, first=0: traj
    io.release = released.append
    io.clearance = lambda arm, q: asked.append(q) or None
    io.jtc = SimpleNamespace(wait_for_server=lambda **kwargs: True,
                             send_goal_async=lambda goal: sent.append(Future()) or sent[-1])
    io.node = SimpleNamespace(get_logger=lambda: SimpleNamespace(warning=lambda text: None))
    traj = SimpleNamespace(t=np.array([.0, .001]), q=np.zeros((2, 7)))
    return SimpleNamespace(io=io, sent=sent, released=released, asked=asked, traj=traj)


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


def test_no_acceptance_fails_after_five_seconds_and_cancels_a_later_one(controller):
    r = controller
    thread, result = run(r)
    thread.join(8.)
    assert not thread.is_alive() and result[0] == ('failed', 0, 'no goal acceptance within 5 s')
    assert accepted(r).cancels == 1
