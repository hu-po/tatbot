"""Launcher identity resolution and CLI admission without hardware access."""
import json
import os
import subprocess
from pathlib import Path

import pytest
from cli_runner import tatbot
from tatbot_cli import nodes

REPO = Path(__file__).resolve().parents[2]


def test_launcher_resolves_address_and_controller_from_left_registry(tmp_path):
    (tmp_path / 'config').mkdir()
    (tmp_path / 'scripts').mkdir()
    (tmp_path / 'scripts/lib').symlink_to(REPO / 'scripts/lib', target_is_directory=True)
    registry = json.loads((REPO / 'config/arms.json').read_text())
    registry['arms']['left']['profile_ip_field'] = 'policy_ip'
    registry['arms']['left']['controller_config'] = 'config/commissioned.yaml'
    (tmp_path / 'config/arms.json').write_text(json.dumps(registry))
    launcher = (REPO / 'scripts/il_rollout_async.sh').read_text()
    block = launcher.split('ROLLOUT_BINDING="$(', 1)[1].split('\n)"', 1)[0]
    script = 'set -euo pipefail\nROLLOUT_BINDING="$(' + block + '\n)"\neval "$ROLLOUT_BINDING"\n'
    script += 'printf "%s\\n" "$ROLLOUT_ARM" "$ROLLOUT_IP" "$ROLLOUT_CONFIG"\n'
    result = subprocess.run(['bash', '-c', script], text=True, capture_output=True,
                            env={**os.environ, 'REPO': str(tmp_path), 'TATBOT_POLICY_IP': '192.0.2.11'})
    assert result.returncode == 0, result.stderr
    assert result.stdout.splitlines() == ['left', '192.0.2.11', str(tmp_path / 'config/commissioned.yaml')]


def test_rollout_tool_default_comes_from_left_workspace():
    import tool_spec
    owner, = nodes.nodes_with(nodes.load(REPO), 'arm')
    result = tatbot('--dry-run', '--json', 'rollout', 'run', node=owner)
    assert result.returncode == 0, result.stderr
    plan = json.loads(result.stdout)
    assert tool_spec.active_tool_id(REPO, 'left') in plan['argv']


@pytest.mark.parametrize('override', [
    '--robot.physical_arm=right', '--robot.ip_address=192.0.2.20',
    '--robot.arm_config=other.yaml',
])
def test_launcher_refuses_overrides_of_its_physical_binding(override):
    launcher = (REPO / 'scripts/il_rollout_async.sh').read_text()
    guard = 'for arg in "$@"; do' + launcher.split('for arg in "$@"; do', 1)[1].split('\ndone', 1)[0] + '\ndone'
    result = subprocess.run(['bash', '-c', guard, 'guard', override], text=True, capture_output=True)
    assert result.returncode == 2 and 'cannot be overridden' in result.stderr
