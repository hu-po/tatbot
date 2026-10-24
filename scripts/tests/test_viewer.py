"""The fleet viewer's recording names, fixed blueprint and data-only RRD import."""
import sys
from pathlib import Path

import pytest

# One vector for both twins: rerun_viewer::recording_name holds the same cases.
NAME_VECTOR = [
    ('live-cameras', 'Rig preview'),
    ('cockpit-20260917T201500Z', 'Cockpit 20:15'),
    ('sweep-20260917_150212', 'Sweep 15:02'),
    ('record-20260917-150212', 'Record 15:02'),
    ('draw-20260917T174304Z-arm-7e3c', 'Draw 17:43'),
    ('replay-flight_12-20260917T174304Z', 'Replay flight_12 17:43'),
    ('draw-shadow-squiggle_v3', 'Draw shadow squiggle_v3'),
    ('20260917T174304Z-arm-7e3c', None),
    ('part-1234567890123456', 'Part 1234567890123456'),
    ('x20260917T174304Z', 'X20260917T174304Z'),
    ('', None),
    (None, None),
]


@pytest.mark.parametrize(('recording_id', 'expected'), NAME_VECTOR)
def test_sources_are_named_from_their_id_and_run_ids_are_left_to_the_session(recording_id, expected):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'lib'))
    import tatbot_rerun as tr
    assert tr.recording_name(recording_id) == expected
    assert tr.STANDING_RECORDING == 'live-cameras'
    minted = tr.recording_id('cockpit')
    assert tr.recording_name(minted).startswith('Cockpit ') and minted.endswith('Z')


def test_every_viewer_server_generation_installs_one_fixed_blueprint():
    repo = Path(__file__).resolve().parents[2]
    server = (repo / 'config/systemd/tatbot-viewer@.service').read_text()
    launcher = (repo / 'scripts/viewer.sh').read_text()
    assert 'ExecStartPost=' in server and 'scripts/viewer.sh blueprint' in server
    # Producers never redial a restarted hub: every start restarts the ones beside it.
    assert ('ExecStartPost=-+/usr/bin/systemctl --no-block try-restart tatbot-cockpit.service '
            'tatbot-sysmon@%i.service') in server
    assert 'send-blueprint --connect "$proxy" --recording-id live-cameras' in launcher
    # `viewer reset` is the same send, so a window drops a pane a tap hid.
    assert '\n  blueprint|reset)\n' in launcher
    assert 'Rerun did not release ports $RERUN_PORT and ${RERUN_WEB_PORT:-9090}' in launcher


def test_rrd_import_is_data_only_before_it_reaches_the_fleet():
    repo = Path(__file__).resolve().parents[2]
    launcher = (repo / 'scripts/viewer.sh').read_text()
    assert 'sanitize-rrd --input "$target"' in launcher
    assert 'rrd verify "$import_tmp/data-only.rrd"' in launcher
    assert 'forward_file(path, proxy, rid)' in launcher
    forward = (repo / 'scripts/lib/tatbot_rerun.py').read_text().split('def forward_file', 1)[1].split('\ndef ', 1)[0]
    assert 'rr.log_file_from_path(str(path))' in forward
    assert 'rr.send_recording_name(recording_name(recording_id))' in forward


SESSION = Path(__file__).resolve().parents[2] / 'scripts/vision/rerun_session.sh'


def _session(script, env=None, path_dir=None):
    import os
    import subprocess
    full = {**os.environ, **(env or {})}
    if path_dir is not None:
        full['PATH'] = f"{path_dir}:{full['PATH']}"
    return subprocess.run(['bash', '-c', f'source "{SESSION}"; {script}'], capture_output=True, text=True,
                          timeout=30, env=full)


def _fake(directory, name, body):
    path = directory / name
    path.write_text('#!/usr/bin/env bash\n' + body + '\n')
    path.chmod(0o755)
    return path


@pytest.mark.parametrize(('cap', 'ceiling'), [('4GB', '5120M'), ('1GB', '2048M'), ('512MB', '1536M'),
                                              ('4GiB', '5120M'), ('1.5GB', '2560M'), ('50%', None),
                                              ('0GB', None), ('', None)])
def test_the_window_unit_ceiling_is_the_cap_plus_one_gib_and_needs_an_absolute_size(cap, ceiling):
    r = _session(f'rerun_session::window_memory_max "{cap}"')
    assert (r.stdout.strip() or None) == ceiling and (r.returncode == 0) == (ceiling is not None)


def test_install_refuses_with_3_unless_rerun_reports_the_fleet_pin(tmp_path):
    for version, code in (('0.36.0', 0), ('0.35.1', 3)):
        rerun = _fake(tmp_path, f'rerun-{version}', f'echo "rerun-cli {version} (native_viewer) aarch64"')
        r = _session('rerun_session::require_pin', {'RERUN_BIN': str(rerun)})
        assert r.returncode == code
    r = _session('rerun_session::require_pin', {'RERUN_BIN': str(tmp_path / 'absent')})
    assert r.returncode == 3 and "uv tool install 'rerun-sdk==0.36.0'" in r.stderr


def test_sysmon_runs_on_the_python_of_the_pinned_rerun_tool(tmp_path):
    venv = tmp_path / 'tools/rerun-sdk/bin'
    venv.mkdir(parents=True)
    _fake(venv, 'rerun', 'exit 0')
    _fake(venv, 'python', 'exit 0')
    (tmp_path / 'rerun').symlink_to(venv / 'rerun')
    r = _session('rerun_session::sdk_python', {'RERUN_BIN': str(tmp_path / 'rerun')})
    assert r.returncode == 0 and r.stdout.strip() == str(venv / 'python')
    assert _session('rerun_session::sdk_python', {'RERUN_BIN': str(tmp_path / 'absent')}).returncode == 1


def test_status_names_the_web_viewer_on_the_standing_proxy():
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'lib'))
    from tatbot_cli import nodes
    repo = Path(__file__).resolve().parents[2]
    server = nodes.rerun_server(nodes.load(repo))
    r = _session('rerun_session::web_url')
    if server is None:
        assert r.returncode == 1
        return
    lan = server[1]
    assert r.stdout.strip() == f'http://{lan}:9090/?url=rerun%2Bhttp%3A%2F%2F{lan}%3A9876%2Fproxy'


def test_open_starts_one_transient_window_unit_and_leaves_a_running_one_alone(tmp_path):
    calls = tmp_path / 'systemd-run.argv'
    stopped = tmp_path / 'stopped'
    _fake(tmp_path, 'systemctl', 'case "$*" in\n'
                                 '  "--user show-environment") echo DISPLAY=:7 ;;\n'
                                 f'  "--user is-active --quiet "*) [ -e "{stopped}" ] && exit 3; exit "${{FAKE_ACTIVE:-3}}" ;;\n'
                                 f'  "--user stop "*) echo "$@" > "{stopped}" ;;\n'
                                 'esac\nexit 0')
    _fake(tmp_path, 'systemd-run', f'printf "%s\\n" "$@" > "{calls}"')
    _fake(tmp_path, 'pgrep', 'exit 1')
    env = {'WAYLAND_DISPLAY': 'wayland-test', 'RERUN_BIN': '/opt/rerun', 'RERUN_RUNTIME_DIR': str(tmp_path / 'rt')}
    r = _session('rerun_session::open_window 4GB rerun+http://192.0.2.7:9876/proxy', env, tmp_path)
    assert r.returncode == 0, r.stderr
    argv = calls.read_text().splitlines()
    assert argv[:5] == ['--user', '--quiet', '--collect', '--unit', 'tatbot-viewer-window']
    assert argv[5:7] == ['-p', 'MemoryMax=5120M'] and '--setenv=WAYLAND_DISPLAY=wayland-test' in argv
    assert argv[argv.index('/opt/rerun'):] == ['/opt/rerun', '--connect', 'rerun+http://192.0.2.7:9876/proxy',
                                               '--memory-limit', '4GB', '--hide-welcome-screen']
    calls.unlink()
    again = _session('rerun_session::open_window 4GB rerun+http://192.0.2.7:9876/proxy',
                     {**env, 'FAKE_ACTIVE': '0'}, tmp_path)
    assert again.returncode == 0 and not calls.exists() and 'already runs' in again.stderr
    restarted = _session('rerun_session::open_window 4GB rerun+http://192.0.2.7:9876/proxy restart',
                         {**env, 'FAKE_ACTIVE': '0'}, tmp_path)
    assert restarted.returncode == 0 and calls.exists(), restarted.stderr  # closed, then opened again
    assert stopped.read_text().split() == ['--user', 'stop', 'tatbot-viewer-window']
    assert _session('rerun_session::open_window 50% x', env, tmp_path).returncode == 2


def test_every_rerun_session_helper_a_tracked_script_calls_is_defined():
    """4485e2ff2 removed view_file while the operator shell's `rerun` wrapper still called it: a sourced script
    fails only when someone types the command."""
    import re
    import subprocess
    repo = SESSION.parents[2]
    files = subprocess.run(['git', 'ls-files', '*.sh', '*.bash'], cwd=repo, capture_output=True, text=True,
                           check=True).stdout.split()
    defined = set(re.findall(r'^(rerun_session::\w+)\(\)', SESSION.read_text(), re.M))
    called = {(name, rel) for rel in files for name in re.findall(r'rerun_session::\w+', (repo / rel).read_text())}
    assert sorted(c for c in called if c[0] not in defined) == []


def test_the_operator_shell_rerun_opens_a_file_capped_and_maximized(tmp_path):
    import os
    import subprocess
    got, asked = tmp_path / 'rerun.argv', tmp_path / 'x11_window.argv'
    rerun_bin = _fake(tmp_path, 'rerun-bin', f'printf "%s\\n" "$@" > "{got}"')
    _fake(tmp_path, 'python3', f'printf "%s\\n" "$@" > "{asked}"')
    _fake(tmp_path, 'systemctl', 'exit 0')
    wrapper = SESSION.parents[1] / 'rerun_maximized.bash'
    env = {**os.environ, 'PATH': f"{tmp_path}:{os.environ['PATH']}", 'RERUN_BIN': str(rerun_bin), 'DISPLAY': ':7',
           'WAYLAND_DISPLAY': '', 'VIEWER_MEMORY_LIMIT': '2GB', 'TATBOT_REPO': str(SESSION.parents[2])}
    r = subprocess.run(['bash', '-c', f'source "{wrapper}"; rerun session.rrd --some-flag; wait'],
                       capture_output=True, text=True, timeout=30, env=env)
    assert r.returncode == 0, r.stderr
    assert got.read_text().split() == ['--memory-limit', '2GB', 'session.rrd', '--some-flag']
    assert asked.read_text().split()[1:] == ['rerun', 'maximized']


def test_open_fullscreen_also_puts_a_running_window_in_that_state_and_a_plain_open_leaves_it(tmp_path):
    asked = tmp_path / 'x11_window.argv'
    _fake(tmp_path, 'systemctl', 'exit 0')  # the window unit runs
    _fake(tmp_path, 'systemd-run', 'exit 0')  # a host without it (a CI container) opens no unit
    _fake(tmp_path, 'python3', f'printf "%s\\n" "$@" > "{asked}"')
    env = {'DISPLAY': ':7', 'WAYLAND_DISPLAY': '', 'RERUN_RUNTIME_DIR': str(tmp_path / 'rt')}
    for state, called in (('maximized', False), ('fullscreen', True)):
        r = _session(f'rerun_session::open_window 4GB rerun+http://192.0.2.7:9876/proxy "" {state}', env, tmp_path)
        assert r.returncode == 0 and 'already runs' in r.stderr and asked.exists() == called, r.stderr
    assert asked.read_text().split()[1:] == ['rerun', 'fullscreen', '--keep-focus']
    asked.unlink()
    r = _session('rerun_session::window_state fullscreen', {**env, 'WAYLAND_DISPLAY': 'wayland-0'}, tmp_path)
    assert r.returncode == 0 and not asked.exists()  # a Wayland session has no X11 window manager to ask


def test_status_names_what_the_window_dials_and_when_a_restart_would_move_it():
    import subprocess
    window = subprocess.Popen(['bash', '-c', 'sleep 30; :', 'rerun', '--connect', 'rerun+http://192.0.2.30:9876/proxy'])
    try:
        moved = _session(f'rerun_session::_window_dials {window.pid} rerun+http://198.51.100.2:9876/proxy')
        same = _session(f'rerun_session::_window_dials {window.pid} rerun+http://192.0.2.30:9876/proxy')
    finally:
        window.kill()
        window.wait()
    assert moved.stdout.splitlines()[0].startswith('        dials rerun+http://192.0.2.30:9876/proxy (')
    assert moved.stdout.splitlines()[1] == '        not the fleet path above: tatbot viewer open --restart moves it'
    assert len(same.stdout.splitlines()) == 1
    assert _session('rerun_session::_window_dials 999999999 x').stdout == ''  # no such process: nothing


def test_a_restart_waits_for_the_stopped_window_unit_to_unload(tmp_path):
    """2026-10-19: `viewer open --restart` stopped the window, systemd-run refused the name the stopped transient
    unit still held, and the station was left with no window."""
    loads = tmp_path / 'loads'
    loads.write_text('3')   # the stopped unit stays loaded for three more queries
    _fake(tmp_path, 'systemctl', 'case "$*" in\n'
                                 '  *"show -p LoadState"*) n=$(cat ' + str(loads) + '); '
                                 'if [ "$n" -gt 0 ]; then echo $((n-1)) > ' + str(loads) + '; echo loaded; '
                                 'else echo not-found; fi ;;\n'
                                 'esac\nexit 0')
    _fake(tmp_path, 'pgrep', 'exit 1')
    _fake(tmp_path, 'pkill', 'exit 0')
    r = _session('rerun_session::close_window', {'RERUN_RUNTIME_DIR': str(tmp_path / 'rt')}, tmp_path)
    assert r.returncode == 0 and loads.read_text().strip() == '0', r.stderr
