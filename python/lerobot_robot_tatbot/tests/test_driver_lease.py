"""Real OS-lock interoperability; no SDK connection or arm command."""
import os
import shutil
import subprocess
from pathlib import Path

import pytest
from lerobot_robot_tatbot import arm_session, driver_lease


def external_lock(path):
    return subprocess.run(['flock', '-n', str(path), 'true'], check=False).returncode


def test_shared_arms_hold_until_last_release(tmp_path):
    path = tmp_path / 'driver.lock'
    first = driver_lease.acquire(path)
    second = driver_lease.acquire(path)
    inode = path.stat().st_ino
    assert external_lock(path) != 0
    first.close()
    assert external_lock(path) != 0
    second.close()
    assert external_lock(path) == 0
    assert path.stat().st_ino == inode


def test_foreign_owner_refuses_and_recovers_after_exit(tmp_path):
    path = tmp_path / 'driver.lock'
    with subprocess.Popen(['flock', '-x', str(path), 'sh', '-c', 'echo ready; read answer'],
                          stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True) as child:
        assert child.stdout.readline().strip() == 'ready'
        with pytest.raises(driver_lease.DriverBusyError):
            driver_lease.acquire(path)
        child.communicate('done\n', timeout=5)
    with driver_lease.acquire(path):
        assert external_lock(path) != 0


def test_nonregular_and_symlink_refuse_without_blocking(tmp_path):
    target = tmp_path / 'target'
    target.touch()
    link = tmp_path / 'link'
    link.symlink_to(target)
    fifo = tmp_path / 'fifo'
    os.mkfifo(fifo)
    for path in (link, fifo, tmp_path):
        with pytest.raises(driver_lease.DriverBusyError):
            driver_lease.acquire(path)


def test_cpp_python_interoperability(tmp_path):
    compiler = shutil.which('g++')
    if compiler is None:
        pytest.skip('C++ driver lease interoperability needs g++')
    repo = Path(__file__).resolve().parents[3]
    source = tmp_path / 'lease.cpp'
    source.write_text('#include "driver_lease.hpp"\nint main(int argc, char **argv) { '
                      'if(argc != 2) return 2; try {tatbot::DriverLease lease(argv[1]);}'
                      'catch (...) {return 6;} return 0;}\n')
    binary = tmp_path / 'lease'
    subprocess.run([compiler, '-std=c++17', '-Wall', '-Werror', '-I',
                    str(repo / 'cpp/teleop'), str(source), '-o', str(binary)], check=True)
    path = tmp_path / 'driver.lock'
    with driver_lease.acquire(path):
        assert subprocess.run([str(binary), str(path)], check=False).returncode == 6
    assert subprocess.run([str(binary), str(path)], check=False).returncode == 0


@pytest.mark.parametrize('name', ['follower'])
def test_busy_plugin_never_connects_or_disconnects_driver(monkeypatch, tmp_path, name):
    import importlib
    module = importlib.import_module(f'lerobot_robot_tatbot.tatbot_{name}')
    base = getattr(module, 'Tatbot' + name.title())

    class Quiet(base):
        def __del__(self):
            pass

        def _disconnect_owned(self, **kwargs):
            raise AssertionError('busy plugin must not attempt landing or cleanup')

    robot = object.__new__(Quiet)
    path = tmp_path / 'driver.lock'
    # Both arms take the lease through the shared session mixin, so that is
    # where the lock path is redirected. Patching the arm module instead
    # silently misses, and connect() then reaches for a real controller.
    monkeypatch.setattr(arm_session, 'acquire_driver_lease',
                        lambda: driver_lease.acquire(path))
    with subprocess.Popen(['flock', '-x', str(path), 'sh', '-c', 'echo ready; read answer'],
                          stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True) as child:
        assert child.stdout.readline().strip() == 'ready'
        with pytest.raises(driver_lease.DriverBusyError):
            robot.connect()
        robot.disconnect()
        child.communicate('done\n', timeout=5)
