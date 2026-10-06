"""Documented drawing commands own their dependencies and keep Python in the signal path."""
import json
import os
import selectors
import shutil
import signal
import subprocess
import sys
from pathlib import Path

import pytest
import tomllib

REPO = Path(__file__).resolve().parents[2]
HELPER = REPO/'scripts/lib/drawing_python.sh'
PROFILE = HELPER.with_suffix('.py')


def runtime_profile():
    if not shutil.which('uv'):
        pytest.skip('managed drawing runtime requires uv')
    metadata = PROFILE.read_text().split('# ///')[1].removeprefix(' script\n')
    profile = tomllib.loads('\n'.join(line.removeprefix('# ') for line in metadata.splitlines()))
    if os.environ.get('TATBOT_TEST_PROFILE') == 'offline':
        probe = subprocess.run(['uv', 'python', 'find', '--no-project', '--no-config', '--managed-python',
                                '--offline', profile['requires-python']], capture_output=True, text=True, timeout=10)
        if probe.returncode and 'No interpreter found' in probe.stderr:
            pytest.skip(f"offline image lacks managed drawing Python {profile['requires-python']}; "
                        'locked-runtime integration not checked')
        assert probe.returncode == 0, probe.stderr
    return profile


def test_research_backend_help_needs_only_stdlib():
    result = subprocess.run([sys.executable, '-S', str(REPO/'scripts/research.py'), '--help'],
                            capture_output=True, text=True, timeout=10)
    assert result.returncode == 0 and 'candidate' in result.stdout, result.stderr


def test_offline_runtime_names_the_missing_managed_interpreter(tmp_path, monkeypatch):
    if not shutil.which('uv'):
        pytest.skip('managed drawing runtime requires uv')
    monkeypatch.setenv('TATBOT_TEST_PROFILE', 'offline')
    monkeypatch.setenv('UV_PYTHON_INSTALL_DIR', str(tmp_path/'empty-managed-installation'))
    with pytest.raises(pytest.skip.Exception, match='locked-runtime integration not checked'):
        runtime_profile()


@pytest.mark.slow
@pytest.mark.parametrize('signum', [signal.SIGINT, signal.SIGTERM])
def test_locked_runtime_ignores_foreign_project_and_executes_python_directly(tmp_path, signum):
    profile = runtime_profile()
    expected = dict(requirement.split('==') for requirement in profile['dependencies'])
    (tmp_path/'pyproject.toml').write_text('[project]\nname="foreign-project"\nversion="0.0.0"\ndependencies=["not-a-real-tatbot-test-dependency"]\n')
    code = f'packages = {tuple(expected)!r}\n' + '''import importlib.metadata, json, os, platform, signal, sys
signal.signal(signal.SIGINT, lambda *_: sys.exit(42))
signal.signal(signal.SIGTERM, lambda *_: sys.exit(43))
print(json.dumps({'pid':os.getpid(), 'python':platform.python_version(),
                  'packages':{name:importlib.metadata.version(name) for name in packages}}), flush=True)
signal.pause()
'''
    errors = tmp_path/'stderr.txt'
    with errors.open('w') as stderr:
        process = subprocess.Popen([str(HELPER), '-c', code], cwd=tmp_path, stdout=subprocess.PIPE, stderr=stderr,
                                   env={**os.environ, 'VIRTUAL_ENV': str(tmp_path/'foreign-venv'),
                                        'UV_PROJECT_ENVIRONMENT': str(tmp_path/'foreign-project-env'), 'UV_NO_SYNC': '1'},
                                   text=True, start_new_session=True)
        try:
            with selectors.DefaultSelector() as selector:
                selector.register(process.stdout, selectors.EVENT_READ)
                assert selector.select(45), errors.read_text()
            output = process.stdout.readline()
            assert output, errors.read_text()
            observed = json.loads(output)
            assert observed == {'pid': process.pid, 'python': profile['requires-python'].removeprefix('=='), 'packages': expected}
            process.send_signal(signum)
            assert process.wait(timeout=10) == (42 if signum == signal.SIGINT else 43)
        finally:
            if process.poll() is None:
                process.kill()
                process.wait()
            process.stdout.close()


@pytest.mark.slow
def test_stale_runtime_lock_refuses_before_user_command(tmp_path):
    runtime_profile()
    library = tmp_path/'scripts/lib'
    library.mkdir(parents=True)
    for name in ('drawing_python.sh', 'drawing_python.py', 'drawing_python.py.lock'):
        shutil.copy2(HELPER.parent/name, library/name)
    profile = library/'drawing_python.py'
    profile.write_text(profile.read_text().replace('numpy==', 'numpy>='))
    lock = library/'drawing_python.py.lock'
    original = lock.read_bytes()
    marker = tmp_path/'unexpected-execution'
    result = subprocess.run([str(library/'drawing_python.sh'), '-c',
                             f'from pathlib import Path; Path({str(marker)!r}).touch()'],
                            capture_output=True, text=True, timeout=45)
    assert result.returncode != 0 and 'lockfile' in result.stderr.lower(), result.stderr
    assert not marker.exists() and lock.read_bytes() == original
