"""The pipeline report is an offline CLI, including unknown historical evidence."""
import json
import os
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def call(*args):
    return subprocess.run([str(ROOT / 'scripts/tatbot'), 'ros', 'timing', *map(str, args)],
                          cwd=ROOT, env={**os.environ, 'TATBOT_CLI_NO_AUDIT': '1'},
                          capture_output=True, text=True)


def test_missing_timing_is_unknown_json_and_never_filled_from_run_duration(tmp_path):
    (tmp_path / 'meta.json').write_text(json.dumps({'duration_s': 123.}))
    result = call(tmp_path, '--json')
    assert result.returncode == 1, result.stderr
    report = json.loads(result.stdout)
    assert report['schema'] == 'tatbot.pipeline-timing/1'
    assert not report['recording_complete']
    assert report['metrics']['plan_to_send_s']['p95_s'] is None
    assert 'sdk_command_application_s' in report['unmeasured']


def test_dry_run_is_read_only_and_does_not_create_reports(tmp_path):
    result = call(tmp_path, '--write', '--dry-run', '--json')
    assert result.returncode == 0, result.stderr
    assert not list(tmp_path.iterdir())
    assert 'ros_pipeline.py' in result.stdout


def test_historical_report_can_be_regenerated_without_ros(tmp_path):
    result = call(tmp_path, '--write')
    assert result.returncode == 1
    assert 'unknown' in result.stdout
    assert json.loads((tmp_path / 'pipeline-timing-summary.json').read_text())['recording_complete'] is False
    assert (tmp_path / 'pipeline-timing-report.md').is_file()
