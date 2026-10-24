"""The GPU lease on real flocks in a temporary directory; memory readings are passed in."""

from __future__ import annotations

import fcntl
import json
import os
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path

import gpu_lease
import pytest

LIB = Path(gpu_lease.__file__).resolve().parent
ROOMY = {"total_mib": 8192, "free_mib": 8000, "source": "test"}


def roomy():
    return dict(ROOMY)


def test_second_acquirer_is_refused_until_the_first_releases(tmp_path):
    gpu = {"slots": 1}
    first = gpu_lease.acquire(100, 0, "first", gpu=gpu, directory=tmp_path, memory=roomy)
    with pytest.raises(BlockingIOError, match=r"1/1 slots held \(pid \d+ first\)"):
        gpu_lease.acquire(100, 0, "second", gpu=gpu, directory=tmp_path, memory=roomy)
    first.release()
    with gpu_lease.acquire(100, 0, "second", gpu=gpu, directory=tmp_path, memory=roomy) as second:
        assert second.held and second.slot == 0


def test_a_killed_holder_frees_its_slot(tmp_path):
    gpu = {"slots": 1}
    child = subprocess.Popen(
        [sys.executable, "-c",
         "import sys; sys.path.insert(0, sys.argv[1]); import gpu_lease, time\n"
         "lease = gpu_lease.acquire(10, 0, 'child', gpu={'slots': 1}, directory=sys.argv[2],\n"
         "    memory=lambda: {'total_mib': 8192, 'free_mib': 8000, 'source': 'test'})\n"
         "print('held', flush=True); time.sleep(60)",
         str(LIB), str(tmp_path)], stdout=subprocess.PIPE, text=True)
    try:
        assert child.stdout.readline().strip() == "held"
        report = gpu_lease.status(gpu, directory=tmp_path, memory=roomy)
        assert [h["pid"] for h in report["held"]] == [child.pid]
        with pytest.raises(BlockingIOError):
            gpu_lease.acquire(10, 0, "parent", gpu=gpu, directory=tmp_path, memory=roomy)
    finally:
        child.send_signal(signal.SIGKILL)
        child.wait(10)
    # The dead holder's file is still there, but its slot is unlocked and it is ignored.
    assert (tmp_path / "slot-0.json").exists()
    assert gpu_lease.status(gpu, directory=tmp_path, memory=roomy)["held"] == []
    with gpu_lease.acquire(10, 0, "parent", gpu=gpu, directory=tmp_path, memory=roomy) as lease:
        assert lease.held
        assert json.loads((tmp_path / "slot-0.json").read_text())["pid"] == os.getpid()


def test_the_holder_file_round_trips_and_goes_with_the_lease(tmp_path):
    gpu = {"slots": 2}
    with gpu_lease.acquire(1500, 0, "evidence", run_id="run-a", gpu=gpu, directory=tmp_path,
                           memory=roomy) as lease:
        on_disk = json.loads((tmp_path / f"slot-{lease.slot}.json").read_text())
        assert on_disk == lease.holder
        assert set(on_disk) == {"pid", "run_id", "worktree", "purpose", "mib", "since"}
        assert (on_disk["purpose"], on_disk["run_id"], on_disk["mib"]) == ("evidence", "run-a", 1500)
        assert Path(on_disk["worktree"], "scripts/lib/gpu_lease.py").is_file()
        report = gpu_lease.status(gpu, directory=tmp_path, memory=roomy)
        assert report["held"] == [{"slot": lease.slot, **on_disk}]
        assert (report["free_slots"], report["granted_mib"], report["grantable_mib"]) == (1, 1500, 8192 - 1500)
    assert not (tmp_path / f"slot-{lease.slot}.json").exists()
    assert gpu_lease.status(gpu, directory=tmp_path, memory=roomy)["held"] == []


def test_free_memory_below_the_request_is_refused(tmp_path):
    gpu = {"slots": 2}
    tight = lambda: {"total_mib": 8192, "free_mib": 1000, "source": "test"}  # noqa: E731
    with pytest.raises(BlockingIOError, match="1000 MiB free < 2000 MiB requested"):
        gpu_lease.acquire(2000, 0, gpu=gpu, directory=tmp_path, memory=tight)
    assert gpu_lease.status(gpu, directory=tmp_path, memory=tight)["held"] == []
    with gpu_lease.acquire(900, 0, gpu=gpu, directory=tmp_path, memory=tight) as lease:
        assert lease.held


def test_the_reserve_is_never_handed_out(tmp_path):
    # 8192 total - 2600 reserved = 5592 MiB for the lease, however much reads as free.
    gpu = {"slots": 2, "reserve_mib": 2600}
    with pytest.raises(ValueError, match="can never be granted"):
        gpu_lease.acquire(6000, 30, gpu=gpu, directory=tmp_path, memory=roomy)
    with gpu_lease.acquire(4000, 0, gpu=gpu, directory=tmp_path, memory=roomy):
        with pytest.raises(BlockingIOError, match="4000 MiB already granted"):
            gpu_lease.acquire(2000, 0, gpu=gpu, directory=tmp_path, memory=roomy)
        with gpu_lease.acquire(1500, 0, gpu=gpu, directory=tmp_path, memory=roomy) as second:
            assert second.held
            # Every slot is held: an impossible request still fails at once and never queues.
            started = time.monotonic()
            with pytest.raises(ValueError, match="at most 5592 MiB"):
                gpu_lease.acquire(6000, 30, gpu=gpu, directory=tmp_path, memory=roomy)
            assert time.monotonic() - started < 1
            assert gpu_lease.status(gpu, directory=tmp_path, memory=roomy)["queued"] == []


def _wait_for(predicate, timeout_s=10.0):
    deadline = time.monotonic() + timeout_s
    while not predicate():
        assert time.monotonic() < deadline, "condition not reached"
        time.sleep(0.01)


def test_a_waiter_queues_first_come_and_the_queue_is_bounded(tmp_path):
    gpu = {"slots": 1}
    holder = gpu_lease.acquire(10, 0, "holder", gpu=gpu, directory=tmp_path, memory=roomy)
    got = {}

    def wait():
        got["lease"] = gpu_lease.acquire(10, 10, "waiter", gpu=gpu, directory=tmp_path, memory=roomy,
                                         poll_s=0.02, max_queue=1)

    waiter = threading.Thread(target=wait)
    waiter.start()
    _wait_for(lambda: len(gpu_lease.status(gpu, directory=tmp_path, memory=roomy)["queued"]) == 1)
    with pytest.raises(BlockingIOError, match="queue is full"):
        gpu_lease.acquire(10, 5, gpu=gpu, directory=tmp_path, memory=roomy, max_queue=1)
    holder.release()
    # A caller that does not wait cannot take the slot the waiter is first in line for.
    with pytest.raises(BlockingIOError):
        gpu_lease.acquire(10, 0, gpu=gpu, directory=tmp_path, memory=roomy)
    waiter.join(10)
    assert got["lease"].held and got["lease"].holder["purpose"] == "waiter"
    assert gpu_lease.status(gpu, directory=tmp_path, memory=roomy)["queued"] == []
    got["lease"].release()


def test_a_wait_that_runs_out_is_busy(tmp_path):
    gpu = {"slots": 1}
    with gpu_lease.acquire(10, 0, gpu=gpu, directory=tmp_path, memory=roomy):
        started = time.monotonic()
        with pytest.raises(BlockingIOError, match="slots held"):
            gpu_lease.acquire(10, 0.2, gpu=gpu, directory=tmp_path, memory=roomy, poll_s=0.05)
        assert 0.2 <= time.monotonic() - started < 5
    assert not list((tmp_path / "queue").glob("*.json"))


_CONTENDER = """
import json, sys, time
sys.path.insert(0, sys.argv[1])
import gpu_lease
role, directory, until = sys.argv[2], sys.argv[3], time.time() + float(sys.argv[4])
gpu = {"slots": 1}
memory = lambda: {"total_mib": 8192, "free_mib": 8000, "source": "test"}
counts = {"grants": 0, "refusals": [], "reads": 0}
while time.time() < until:
    if role == "acquire":
        try:
            gpu_lease.acquire(10, 0, "loop", gpu=gpu, directory=directory, memory=memory).release()
            counts["grants"] += 1
        except BlockingIOError as exc:
            counts["refusals"].append(str(exc))
    else:
        gpu_lease.status(gpu, directory=directory, memory=memory)
        counts["reads"] += 1
print(json.dumps(counts))
"""


def test_reading_status_never_makes_a_grant_fail(tmp_path):
    # One process takes and frees the only slot as fast as it can, with no
    # other acquirer, while two read the lease: every acquire must be granted.
    gpu_lease.acquire(10, 0, gpu={"slots": 1}, directory=tmp_path, memory=roomy).release()
    procs = [subprocess.Popen([sys.executable, "-c", _CONTENDER, str(LIB), role, str(tmp_path), "2.5"],
                              stdout=subprocess.PIPE, text=True)
             for role in ("acquire", "status", "status")]
    acquirer, *readers = [json.loads(proc.communicate(timeout=30)[0]) for proc in procs]
    assert acquirer["grants"] > 100 and all(reader["reads"] > 100 for reader in readers)
    assert acquirer["refusals"][:3] == []


def test_status_waits_a_bounded_time_for_an_admission(tmp_path):
    gpu_lease.acquire(10, 0, gpu={"slots": 1}, directory=tmp_path, memory=roomy).release()
    fd = os.open(tmp_path / "admit.lock", os.O_RDONLY)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX)  # an admission that never ends
        with pytest.raises(TimeoutError, match="admit.lock stayed locked"):
            gpu_lease.status({"slots": 1}, directory=tmp_path, memory=roomy, timeout_s=0.05)
    finally:
        os.close(fd)
    assert gpu_lease.status({"slots": 1}, directory=tmp_path, memory=roomy)["free_slots"] == 1


def test_memory_readings_parse_and_a_unified_card_says_so():
    assert gpu_lease.parse_nvidia_smi("8192, 5400\n") == {
        "total_mib": 8192, "free_mib": 5400, "source": "nvidia-smi"}
    with pytest.raises(ValueError, match='"unified": true'):
        gpu_lease.parse_nvidia_smi("[N/A], [N/A]\n")
    meminfo = "MemTotal:       125829120 kB\nMemFree:  1 kB\nMemAvailable:   100663296 kB\n"
    assert gpu_lease.parse_meminfo(meminfo) == {"total_mib": 122880, "free_mib": 98304, "source": "meminfo"}


def test_records_resolve_per_node_and_bad_ones_are_named():
    nmap = {"box": {"roles": [], "gpu": {"slots": 2, "reserve_mib": 512}}, "plain": {"roles": []}}
    assert gpu_lease.node_gpu(nmap, "box") == {"slots": 2, "reserve_mib": 512, "unified": False}
    assert gpu_lease.node_gpu(nmap, "plain") is None
    for bad in ({"slots": 0}, {"slots": True}, {"slots": 1, "reserve_mib": -1}, {"slots": 1, "vram": 8}):
        with pytest.raises(ValueError):
            gpu_lease.validate(bad)


def test_status_creates_nothing_and_no_record_means_no_lease(tmp_path, monkeypatch):
    report = gpu_lease.status({"slots": 2}, directory=tmp_path / "absent", memory=roomy)
    assert (report["free_slots"], report["held"], report["queued"]) == (2, [], [])
    assert not (tmp_path / "absent").exists()
    monkeypatch.setenv("TATBOT_GPU_LEASE_DIR", str(tmp_path / "env"))
    assert gpu_lease.lease_dir() == tmp_path / "env"
    monkeypatch.setattr(gpu_lease, "node_gpu", lambda: None)
    lease = gpu_lease.acquire(10)
    assert not lease.held and lease.slot is None
    assert not (tmp_path / "env").exists()


def test_hold_runs_a_command_under_a_slot_and_exits_6_when_busy(tmp_path, monkeypatch):
    monkeypatch.setenv("TATBOT_GPU_LEASE_DIR", str(tmp_path))
    monkeypatch.setattr(gpu_lease, "node_gpu", lambda: {"slots": 1, "reserve_mib": 0, "unified": True})
    probe = "import os, sys; sys.exit(0 if os.path.exists(sys.argv[1]) else 3)"
    assert gpu_lease.main(["hold", "--mib", "0", "--", sys.executable, "-c", probe,
                           str(tmp_path / "slot-0.json")]) == 0
    with gpu_lease.acquire(0, 0, gpu={"slots": 1, "unified": True}, directory=tmp_path):
        assert gpu_lease.main(["hold", "--mib", "0", "--", sys.executable, "-c", "pass"]) == 6
    # More than the node has is a usage error, never a wait or a busy.
    assert gpu_lease.main(["hold", "--mib", str(2**40), "--wait-s", "30", "--", "true"]) == 2
