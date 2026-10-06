"""Controller-only capture bounds and honest failure reporting, without hardware."""
import json
from types import SimpleNamespace

import pytest
import teleop_network_trace as trace  # noqa: E402

DRIVER = {'leader_ip': '192.0.2.10', 'follower_ip': '192.0.2.11'}


@pytest.mark.parametrize('value', [None, '', 'example.com', '192.0.2.10 or net 0/0', '::1', '192.0.2.11'])
def test_trace_refuses_missing_nonnumeric_injected_or_duplicate_addresses(value):
    with pytest.raises(ValueError):
        trace.controller_filter({**DRIVER, 'leader_ip': value})


@pytest.mark.parametrize('seconds', [0, 14, 121, 600])
def test_capture_duration_is_bounded(seconds, tmp_path):
    with pytest.raises(ValueError, match='15..120'):
        trace.capture_command(DRIVER, tmp_path, seconds, 'tcpdump', 'timeout', 'sudo', 'operator')


def test_capture_is_passive_restricted_and_drops_privileges(tmp_path):
    command = trace.capture_command(DRIVER, tmp_path, 90, '/bin/tcpdump', '/bin/timeout', '/bin/sudo', 'operator')
    assert command[:6] == ['/bin/sudo', '-n', '/bin/timeout', '--signal=INT', '--kill-after=3', '90']
    assert command[-1] == '(host 192.0.2.10 or host 192.0.2.11) and (arp or udp port 50000 or tcp port 50001)'
    assert command[command.index('-Z') + 1] == 'operator'
    assert command[command.index('-s') + 1] == '128'
    assert command[command.index('-W') + 1] == '4'
    assert '-p' in command


@pytest.mark.parametrize(('rc', 'size', 'expected'), [(124, 24, 3), (124, 128, 0), (1, 128, 3)])
def test_empty_or_failed_trace_is_not_reported_as_success(tmp_path, monkeypatch, rc, size, expected):
    monkeypatch.setenv('TATBOT_RUN_DIR', str(tmp_path))
    monkeypatch.setattr(trace.tatbot_profile, 'load', lambda _: {'driver': DRIVER, '_sha256': 'test'})
    monkeypatch.setattr(trace.pwd, 'getpwuid', lambda _uid: SimpleNamespace(pw_name='operator'))
    monkeypatch.setattr(trace.shutil, 'which', lambda name: '/bin/' + name)
    monkeypatch.setattr(trace, 'network_snapshot', lambda *args: None)
    monkeypatch.setattr(trace.os, 'umask', lambda _: None)

    def run(command, **kwargs):
        assert kwargs == {'timeout': 25, 'check': False}
        assert command[0] == '/bin/sudo'
        (tmp_path / 'controllers.pcap0').write_bytes(bytes(size))
        return SimpleNamespace(returncode=rc)

    monkeypatch.setattr(trace.subprocess, 'run', run)
    assert trace.main(['--seconds', '15']) == expected
    result = json.loads((tmp_path / 'result.json').read_text())
    assert result['communication_qualified'] is False
    assert result['packets_retained'] is (size > 24)
