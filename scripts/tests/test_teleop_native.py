"""wxai_teleop's start and stop contracts, and the passive network trace.

Moved from the calibration-sweep tests when that pipeline was removed
(2026-09-29): these pin the native teleop and its CLI verb, not the sweep.
"""

import json
import os
import select
import signal
import subprocess
from pathlib import Path

import pytest
import tool_spec
from cli_runner import tatbot
from tatbot_cli import nodes

ROOT = Path(__file__).resolve().parents[2]


def plan(*args: str, node: str) -> dict:
    result = tatbot("--dry-run", "--json", *args, node=node)
    assert result.returncode == 0, result.stderr
    return json.loads(result.stdout)


def test_teleop_network_trace_is_passive_and_routes_to_arm_owner():
    result = plan('teleop', 'trace-network', '--seconds', '90', node=nodes.example_node('arm'))
    assert result['tier'] == 'sensor' and result['launch_id'] is None
    assert result['role'] == 'arm' and result['tool'] is None
    assert 'human_motion' not in result['effects'] and 'auto_motion' not in result['effects']
    assert result['argv'][-2:] == ['--seconds', '90']
    assert result['argv'][-3].endswith('/scripts/teleop_trace_network.sh')
    remote = plan('teleop', 'trace-network', '--seconds', '90', node=nodes.example_node('operator'))
    assert remote['hop'] is not None
    assert 'teleop trace-network --seconds 90' in remote['argv'][-1]


def test_native_startup_and_resume_never_absorb_pending_stop():
    source = (ROOT / 'cpp/teleop/wxai_teleop.cpp').read_text()
    assert 'int stop_baseline = 0;' in source
    assert 'stop_baseline = g_stop_signals.load();' not in source
    assert 'stop_baseline = resume_signals;' in source
    assert 'Alignment::wrist_calibration(opt.align_rate)' in source
    assert source.index('WXAI_TELEOP_PID=') < source.index('if (!arm_reachable(ip))')


@pytest.mark.parametrize('cancel', ['interrupt', 'eof'])
def test_real_native_supervised_start_cancels_before_arm_connection(cancel):
    binary = ROOT / 'cpp/teleop/build/wxai_teleop'
    if not binary.exists():
        pytest.skip('build the native driver first')
    # Loopback addresses are a second containment boundary if the gate regresses.
    process = subprocess.Popen(
        [str(binary), '127.0.0.1', '127.0.0.1', '--ee-tool', tool_spec.active_tool_id(ROOT),
         '--wrist-calibration', '--supervised-start'], cwd=ROOT, stdin=subprocess.PIPE,
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    output = b''
    try:
        for _ in range(30):
            ready, _, _ = select.select([process.stdout], [], [], .1)
            if ready:
                output += os.read(process.stdout.fileno(), 4096)
            if b'SUPERVISED_START_READY:' in output:
                break
        assert b'SUPERVISED_START_READY:' in output
        assert process.poll() is None
        if cancel == 'interrupt':
            process.send_signal(signal.SIGINT)
        else:
            process.stdin.close()
            process.stdin = None
        tail, _ = process.communicate(timeout=2)
        output += tail
        assert process.returncode == 130, output.decode()
        assert b'cancelled before arm connection' in output
        assert b'Connecting' not in output and b'not reachable' not in output
    finally:
        if process.poll() is None:
            process.kill()
            process.wait()
