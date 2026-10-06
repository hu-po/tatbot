"""The whole launch in its own graph: controllers active, /joint_states near 400 Hz, the safety GPIO
states, and a JTC goal that moves, succeeds and leaves the arm unlatched (scripts/mock_check.py).

TATBOT_CHECK_HARDWARE picks the hardware: mock (test_mock_stack) or fake (test_fake_stack, TatbotArm
with its fake SDK and every interlock live)."""
import json
import os
import random
import re
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

CHECK = Path(__file__).resolve().parents[1] / "scripts" / "mock_check.py"


@pytest.mark.skipif(not shutil.which("ros2"), reason="needs a sourced ROS 2 workspace")
def test_mock_stack(tmp_path):
    hardware = os.environ.get("TATBOT_CHECK_HARDWARE", "mock")
    env = dict(os.environ, TATBOT_LOG_ROOT=str(tmp_path / "logs"))
    out = subprocess.run([sys.executable, str(CHECK), "--port", "0", "--domain", str(random.randint(100, 199)),
                          "--logs", str(tmp_path / "check"), f"hardware:={hardware}"],
                         env=env, capture_output=True, text=True, timeout=200)
    lines = [line for line in out.stdout.splitlines() if line.startswith("{")]
    assert lines, out.stdout + out.stderr
    result = json.loads(lines[-1])
    assert result["ok"], json.dumps(result, indent=1) + (tmp_path / "check" / "launch.log").read_text()[-4000:]
    run = tmp_path / 'logs' / 'ros-stack' / result['ros_stack_run']
    process = json.loads((run / 'controller-process.json').read_text())
    launched_pid = re.search(r'\[ros2_control_node-\d+\]: process started with pid \[(\d+)\]',
                             (tmp_path / 'check' / 'launch.log').read_text())
    assert launched_pid and process['pid'] == int(launched_pid[1])
    assert process['process_start'] and process['boot_id']
    # This launch's session published its runtime identity into the check's logs, not beside $TATBOT_REPO.
    record = json.loads((tmp_path / 'check' / 'runtime.json').read_text())
    assert record['controller_reference'] == str(run / 'controller-process.json')
