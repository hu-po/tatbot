"""Real processes and kernel flocks on private temp files; no arm hardware."""
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]


@pytest.fixture(scope="module")
def lease_probe(tmp_path_factory):
    compiler = shutil.which("c++")
    if compiler is None:
        pytest.skip("C++ compiler unavailable")
    root = tmp_path_factory.mktemp("driver-takeover")
    source = root / "probe.cpp"
    source.write_text(r'''
#include "driver_lease.hpp"
#include <memory>
int main(int argc, char **argv) {
  if (argc != 3) return 2;
  try {
    std::unique_ptr<tatbot::DriverLease> lease;
    int fd = -1;
    const std::string mode = argv[2];
    if (mode == "fast") {
      fd = ::open(argv[1], O_RDWR | O_CREAT | O_NOFOLLOW, 0600);
      if (fd < 0) return 2;
      tatbot::driver_takeover::acquire(fd, std::chrono::milliseconds(150),
                                     std::chrono::milliseconds(300));
    } else {
      lease = std::make_unique<tatbot::DriverLease>(argv[1], mode == "recover"
        ? tatbot::DriverLease::Mode::recover : tatbot::DriverLease::Mode::exclusive);
    }
    std::cout << "owned" << std::endl;
    std::string line;
    std::getline(std::cin, line);
    if (fd >= 0) ::close(fd);
    return 0;
  } catch (const std::exception &error) {
    std::cerr << error.what() << std::endl;
    return 6;
  }
}
''')
    binary = root / "probe"
    subprocess.run([compiler, "-std=c++17", "-Wall", "-Wextra", "-Werror", "-pthread",
                    "-I", str(REPO / "cpp/teleop"), str(source), "-o", str(binary)],
                   check=True, capture_output=True, text=True)
    return binary


@pytest.fixture
def processes():
    children = []

    def start(argv):
        process = subprocess.Popen(argv, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                   stderr=subprocess.PIPE, text=True)
        children.append(process)
        return process

    yield start
    for process in children:
        if process.poll() is None:
            process.kill()
        process.communicate(timeout=5)


def owner(processes, path, *, ignore_term=False, lock=True):
    process = processes([sys.executable, "-c", """
import fcntl, os, signal, sys
fd = os.open(sys.argv[1], os.O_RDWR | os.O_CREAT, 0o600)
if sys.argv[2] == 'ignore':
    signal.signal(signal.SIGTERM, signal.SIG_IGN)
if sys.argv[3] == 'lock':
    fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
print('ready', flush=True)
sys.stdin.readline()
""", str(path), "ignore" if ignore_term else "terminate", "lock" if lock else "open"])
    assert process.stdout.readline() == "ready\n"
    return process


def acquire(processes, lease_probe, path, mode="fast"):
    process = processes([str(lease_probe), str(path), mode])
    assert process.stdout.readline() == "owned\n", process.stderr.read()
    return process


def test_normal_acquisition_refuses_busy_without_signaling(lease_probe, processes, tmp_path):
    path = tmp_path / "driver.lock"
    holder = owner(processes, path)
    result = subprocess.run([str(lease_probe), str(path), "exclusive"], input="",
                            capture_output=True, text=True, timeout=5)
    assert result.returncode == 6 and "driver busy" in result.stderr
    assert holder.poll() is None


def test_recovery_terminates_owner_and_retains_same_exclusive_inode(lease_probe, processes, tmp_path):
    path = tmp_path / "driver.lock"
    holder = owner(processes, path)
    inode = path.stat().st_ino
    recovery = acquire(processes, lease_probe, path, "recover")
    assert holder.wait(timeout=5) == -15
    assert path.stat().st_ino == inode
    competing = subprocess.run([str(lease_probe), str(path), "exclusive"], input="",
                               capture_output=True, text=True, timeout=5)
    assert competing.returncode == 6
    _, stderr = recovery.communicate(input="\n", timeout=5)
    assert f"SIGTERM to driver lease holder pid={holder.pid}" in stderr
    assert "SIGKILL" not in stderr
    assert recovery.returncode == 0


def test_stuck_holder_is_killed_without_signaling_unrelated_processes(lease_probe, processes, tmp_path):
    path = tmp_path / "driver.lock"
    holder = owner(processes, path, ignore_term=True)
    opener = owner(processes, path, lock=False)
    unrelated = owner(processes, tmp_path / "other.lock")
    recovery = acquire(processes, lease_probe, path)
    assert holder.wait(timeout=5) == -9
    _, stderr = recovery.communicate(input="\n", timeout=5)
    assert f"SIGKILL to driver lease holder pid={holder.pid}" in stderr
    assert str(opener.pid) not in stderr and opener.poll() is None
    assert str(unrelated.pid) not in stderr and unrelated.poll() is None


@pytest.mark.parametrize("orphan", [False, True])
def test_inherited_flock_is_found_even_after_original_owner_exits(lease_probe, processes, tmp_path, orphan):
    path = tmp_path / "driver.lock"
    # A supervisor reaps both workers. The original owner may exit before
    # recovery, leaving an inherited flock invisible in /proc/locks.
    supervisor = processes([sys.executable, "-c", """
import ctypes, fcntl, os, signal, sys
assert ctypes.CDLL(None).prctl(36, 1, 0, 0, 0) == 0  # PR_SET_CHILD_SUBREAPER
read_fd, write_fd = os.pipe()
parent = os.fork()
if parent == 0:
    os.close(read_fd)
    fd = os.open(sys.argv[1], os.O_RDWR | os.O_CREAT, 0o600)
    fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    child = os.fork()
    if child == 0:
        signal.signal(signal.SIGTERM, signal.SIG_IGN)
        os.write(write_fd, str(os.getpid()).encode())
        while True:
            signal.pause()
    if sys.argv[2] == 'orphan':
        os._exit(0)
    while True:
        signal.pause()
os.close(write_fd)
child = int(os.read(read_fd, 100))
if sys.argv[2] == 'orphan':
    os.waitpid(parent, 0)
print(f'{parent} {child}', flush=True)
try:
    sys.stdin.readline()
finally:
    for pid in ((child,) if sys.argv[2] == 'orphan' else (parent, child)):
        try:
            os.kill(pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
    while True:
        try:
            os.waitpid(-1, 0)
        except ChildProcessError:
            break
""", str(path), "orphan" if orphan else "paired"])
    parent, child = map(int, supervisor.stdout.readline().split())
    try:
        recovery = acquire(processes, lease_probe, path)
        _, stderr = recovery.communicate(input="\n", timeout=5)
        assert f"SIGKILL to driver lease holder pid={child}" in stderr
        if not orphan:
            assert f"SIGTERM to driver lease holder pid={parent}" in stderr
        assert supervisor.poll() is None
    finally:
        supervisor.communicate(input="\n", timeout=5)
    assert supervisor.returncode == 0


@pytest.mark.parametrize("kind", ["symlink", "fifo"])
def test_recovery_rejects_invalid_lock_paths(lease_probe, processes, tmp_path, kind):
    target = tmp_path / "real.lock"
    holder = owner(processes, target)
    path = tmp_path / "invalid.lock"
    if kind == "symlink":
        path.symlink_to(target)
    else:
        os.mkfifo(path)
    result = subprocess.run([str(lease_probe), str(path), "recover"], input="",
                            capture_output=True, text=True, timeout=5)
    assert result.returncode == 6
    assert "SIGTERM" not in result.stderr and holder.poll() is None


def test_free_lease_does_not_signal_file_opener(lease_probe, processes, tmp_path):
    path = tmp_path / "driver.lock"
    opener = owner(processes, path, lock=False)
    recovery = acquire(processes, lease_probe, path, "recover")
    _, stderr = recovery.communicate(input="\n", timeout=5)
    assert stderr == "" and opener.poll() is None


def test_uninspectable_owner_refuses_without_bypassing_lock(lease_probe, processes, tmp_path):
    path = tmp_path / "driver.lock"
    holder = processes([sys.executable, "-c", """
import ctypes, fcntl, os, sys
fd = os.open(sys.argv[1], os.O_RDWR | os.O_CREAT, 0o600)
fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
assert ctypes.CDLL(None).prctl(4, 0, 0, 0, 0) == 0  # PR_SET_DUMPABLE
print('ready', flush=True)
sys.stdin.readline()
""", str(path)])
    assert holder.stdout.readline() == "ready\n"
    try:
        list(Path(f"/proc/{holder.pid}/fd").iterdir())
    except PermissionError:
        pass
    else:
        pytest.skip("test process can inspect nondumpable processes")
    result = subprocess.run([str(lease_probe), str(path), "fast"], input="",
                            capture_output=True, text=True, timeout=5)
    assert result.returncode == 6 and "ownership was not released" in result.stderr
    assert "SIGTERM" not in result.stderr and "SIGKILL" not in result.stderr
    assert holder.poll() is None


def test_recovery_waits_for_graceful_cleanup(lease_probe, processes, tmp_path):
    path = tmp_path / "driver.lock"
    marker = tmp_path / "cleaned"
    holder = processes([sys.executable, "-c", """
import fcntl, os, signal, sys, time
from pathlib import Path
fd = os.open(sys.argv[1], os.O_RDWR | os.O_CREAT, 0o600)
fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
def stop(*_):
    time.sleep(0.1)
    Path(sys.argv[2]).write_text('cleanup complete')
    sys.exit(0)
signal.signal(signal.SIGTERM, stop)
print('ready', flush=True)
while True:
    signal.pause()
""", str(path), str(marker)])
    assert holder.stdout.readline() == "ready\n"
    recovery = acquire(processes, lease_probe, path, "recover")
    assert marker.read_text() == "cleanup complete"
    assert holder.wait(timeout=5) == 0
    _, stderr = recovery.communicate(input="\n", timeout=5)
    assert "SIGKILL" not in stderr
