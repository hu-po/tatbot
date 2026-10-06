"""Camera-node scan works with an unexported assignment file and Python 3.10 TOML."""
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
import tag_scan  # noqa: E402


def test_credentials_literal_only_and_environment_wins(tmp_path, monkeypatch):
    path = tmp_path/'cameras.env'
    key = 'TATBOT_CAMERA_PASSWORD_CAMERA1'
    monkeypatch.delenv(key, raising=False)
    path.write_text(f'export {key}="literal $() # punctuation"\n')
    tag_scan.load_camera_credentials(path)
    assert os.environ[key] == 'literal $() # punctuation'
    path.write_text(f'{key}=replacement\n')
    tag_scan.load_camera_credentials(path)
    assert os.environ[key] == 'literal $() # punctuation'
    path.write_text('source another-file\n')
    with pytest.raises(ValueError, match='literal'):
        tag_scan.load_camera_credentials(path)


def test_scan_image_default_output_is_timestamped(tmp_path):
    proc = subprocess.run([sys.executable, str(ROOT/'scripts/vision/tag_scan.py'),
                           '--image', str(ROOT/'urdf/meshes/tags/16h5_008_56mm/tag.png')],
                          env={**os.environ, 'HOME': str(tmp_path)}, capture_output=True, text=True)
    assert proc.returncode == 0, proc.stderr
    reports = list(tmp_path.glob('tatbot-logs/vision/tag-scan/*/report.json'))
    assert len(reports) == 1
    data = json.loads(reports[0].read_text())
    assert data['utc'] and data['reports'][0]['detections'][0]['id'] == 8
