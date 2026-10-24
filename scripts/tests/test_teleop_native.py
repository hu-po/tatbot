"""wxai_teleop's start and stop contracts, and the passive network trace."""

import json
from pathlib import Path

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
    assert 'Alignment alignment(opt.absolute, opt.align_rate);' in source
    assert source.index('WXAI_TELEOP_PID=') < source.index('if (!arm_reachable(ip))')
