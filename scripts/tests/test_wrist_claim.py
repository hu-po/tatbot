"""The wrist D405s between the session's idle viewer feed and a reading's camera (scripts/lib/wrist_claim.py)."""
import subprocess
import sys
import threading
import time
from pathlib import Path

LIB = Path(__file__).resolve().parents[1] / 'lib'
sys.path.insert(0, str(LIB))

import wrist_claim  # noqa: E402


class Preview:
    """wrist_tap.Preview's shape: close() and the thread that opened the cameras."""

    def __init__(self, events):
        self.events, self.thread = events, threading.Thread(target=lambda: None)
        self.thread.start()
        events.append('opened')

    def close(self):
        time.sleep(0.1)   # a pipeline stop
        self.events.append('closed')


def _until(condition, timeout_s=5.0):
    deadline = time.monotonic() + timeout_s
    while not condition():
        assert time.monotonic() < deadline, 'timed out'
        time.sleep(0.01)


def test_a_claim_waits_until_the_idle_feed_has_closed_the_cameras(tmp_path):
    events = []
    feed = wrist_claim.IdleFeed(lambda: Preview(events), root=tmp_path, quiet_s=0.0, poll_s=0.05)
    try:
        _until(lambda: events == ['opened'])
        claim = wrist_claim.Claim(root=tmp_path, wait_s=5.0)
        events.append('claimed')
        assert events == ['opened', 'closed', 'claimed'] and feed.preview is None
        time.sleep(0.3)
        assert events[-1] == 'claimed'                       # held: the feed stays off
        claim.release()
        _until(lambda: events[-1] == 'opened')               # free again: back on the viewer
    finally:
        feed.close()
    assert events[-1] == 'closed'


def test_the_feed_waits_out_the_quiet_time_and_the_rig_sleep(tmp_path):
    events, asleep = [], threading.Event()
    asleep.set()
    feed = wrist_claim.IdleFeed(lambda: Preview(events), root=tmp_path, quiet_s=0.3, poll_s=0.05,
                                asleep=asleep.is_set)
    try:
        time.sleep(0.5)
        assert events == []                                  # the rig sleeps
        asleep.clear()
        woke = time.monotonic()
        _until(lambda: events == ['opened'])
        assert time.monotonic() - woke >= 0.3                # free for quiet_s first
        asleep.set()
        _until(lambda: events == ['opened', 'closed'])
    finally:
        feed.close()


def test_a_claim_in_another_process_holds_until_that_process_ends(tmp_path):
    holder = subprocess.Popen([sys.executable, '-c', 'import sys, time; sys.path.insert(0, sys.argv[1]); '
                               'import wrist_claim; c = wrist_claim.Claim(root=__import__("pathlib").Path(sys.argv[2])); '
                               'print("held", flush=True); time.sleep(30)', str(LIB), str(tmp_path)],
                              stdout=subprocess.PIPE, text=True)
    try:
        assert holder.stdout.readline().strip() == 'held'
        assert wrist_claim.claimed(tmp_path)
    finally:
        holder.kill()                                        # a crash: no release() ran
        holder.wait()
    assert not wrist_claim.claimed(tmp_path)


def test_a_feed_that_looks_as_a_claim_comes_in_does_not_open(tmp_path):
    """The feed takes its own lock before its last look at the claims, so a claim and a feed that start together
    never both open the cameras."""
    events = []
    claim = wrist_claim.Claim(root=tmp_path, wait_s=1.0)
    feed = wrist_claim.IdleFeed(lambda: Preview(events), root=tmp_path, quiet_s=3600.0, poll_s=3600.0)
    try:
        feed._open()
        assert events == [] and feed.preview is None and feed._handle is None
    finally:
        claim.release()
        feed.close()


def test_a_claim_opens_anyway_after_a_feed_that_never_lets_go(tmp_path):
    stuck = wrist_claim._lock(tmp_path / 'feed', wrist_claim.fcntl.LOCK_SH)   # a feed whose close hangs
    said = []
    started = time.monotonic()
    claim = wrist_claim.Claim(root=tmp_path, wait_s=0.2, log=said.append)
    assert 0.2 <= time.monotonic() - started < 2.0 and 'opening anyway' in said[0]
    claim.release()
    stuck.close()
