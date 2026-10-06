"""Construction must not depend on import order or a previously fitted tool."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

from tatbot_sim.config import DRConfig
from tatbot_sim.resolved import resolve


def test_resolution_and_geometry_import_need_only_standard_library():
    source = Path(__file__).resolve().parents[1] / 'src'
    result = subprocess.run(
        [sys.executable, '-S', '-c', """
import json, sys
import tatbot_sim.urdf
from tatbot_sim.resolved import resolve
print(json.dumps(resolve(tool_id='picosecond-laser-pen').metadata()))
assert not {'sapien', 'mani_skill', 'torch', 'numpy'} & sys.modules.keys()
"""], env={**os.environ, 'PYTHONPATH': os.pathsep.join((str(source), str(source.parents[1] / 'tatbot_contracts/src')))}, capture_output=True, text=True,
        check=True,
    )
    assert json.loads(result.stdout)['tool'] == 'picosecond-laser-pen'


def test_resolved_values_are_isolated_from_input_and_returned_mutations():
    dr = DRConfig()
    original = dr.camera.mount_jitter_mm
    config = resolve(tool_id='lutin-ballpoint-dot', dr=dr, seed=19)
    dr.camera.mount_jitter_mm = 50
    config.dr.camera.mount_jitter_mm = 100
    config.workspace.clear()
    assert config.dr.camera.mount_jitter_mm == original
    assert config.workspace
    assert config.seed_for('camera') == resolve(seed=19).seed_for('camera')
    assert config.seed_for('camera') != config.seed_for('lighting')
    assert config.seed_for('camera', 1) != config.seed_for('camera', 2)


def test_tool_controls_language_and_pacing_after_import(monkeypatch):
    import numpy as np
    from tatbot_sim.language import CLEAN_STYLE, pacing, sample_scene

    configs = [resolve(tool_id=tool) for tool in ('lutin-ballpoint-dot', 'picosecond-laser-pen')]
    monkeypatch.setenv('TATBOT_TOOL_ID', 'lutin-3rl-bugpin')
    grid = {'pitch_m': .006, 'xs': [], 'ys': []}
    for config in configs:
        _, program = sample_scene(np.random.default_rng(2), grid, 30, config=config, style=CLEAN_STYLE)
        assert program['tool'] == config.tool.prompt_phrase
        assert program['surface'] == config.substrate.surface_phrase
    assert pacing(configs[0])[0] < pacing(configs[1])[0]



def test_worker_hello_reports_distribution_before_world_construction(monkeypatch):
    from argparse import Namespace

    from tatbot_sim.policy_worker import WorkerServer

    monkeypatch.setenv('TATBOT_TOOL_ID', 'lutin-3rl-bugpin')
    worker = WorkerServer(Namespace(distribution='paper-draw'))
    header, arrays = worker.dispatch({'op': 'hello'}, {})
    assert header['tool_id'] == 'lutin-ballpoint-dot'
    assert arrays == {}
    assert worker.episode is None


def test_preparation_and_cached_geometry_do_not_import_a_physics_engine():
    result = subprocess.run([sys.executable, '-c', """
import importlib.abc, sys
class NoEngine(importlib.abc.MetaPathFinder):
    def find_spec(self, name, path=None, target=None):
        if name.split('.')[0] in {'mani_skill', 'sapien'}:
            raise AssertionError('engine imported by CPU preparation: ' + name)
sys.meta_path.insert(0, NoEngine())
from tatbot_sim import generate, textures
from tatbot_sim.urdf import asset_dir
assert asset_dir().name == 'data'
assert textures.TEX_DIR.is_relative_to(asset_dir())
assert not {'mani_skill', 'sapien'} & sys.modules.keys()
"""], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
