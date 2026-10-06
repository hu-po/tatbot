"""The shared material candidate module is part of the scriptlib wheel."""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import tomllib

ROOT = Path(__file__).resolve().parents[2]


def test_material_candidate_is_exported_and_imports_without_simulator():
    project = tomllib.loads((ROOT / 'scripts/lib/pyproject.toml').read_text())
    assert 'stroke_material' in project['tool']['setuptools']['py-modules']
    code = ("import sys,stroke_material; "
            "assert hasattr(stroke_material, 'MaterialCandidate'); "
            "assert not {'tatbot_sim','torch','av','mani_skill','rerun'} & set(sys.modules)")
    result = subprocess.run([sys.executable, '-c', code], env={**os.environ,
        'PYTHONPATH': str(ROOT / 'scripts/lib')}, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
