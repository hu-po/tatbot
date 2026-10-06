from __future__ import annotations

import json
from copy import deepcopy

import numpy as np
import pytest
import torch
from tatbot_sim.inkfield import InkField
from tatbot_sim.inkmap import collection
from tatbot_sim.inkmap.collection import artwork_record, collection_entries, planar_strokes
from tatbot_sim.inkmap.designs import directory_artifacts
from tatbot_sim.surface import PlanarSurface


def test_collection_is_shared_and_families_are_held_out():
    entries = collection_entries()
    assert len(entries) == 3
    assert all(e['usage'] == 'artwork' and e['source']['license'] == 'CC0-1.0' for e in entries)
    splits = [{e['family'] for e in collection_entries(split)} for split in collection.SPLITS]
    assert all(splits)
    assert all(not a & b for i, a in enumerate(splits) for b in splits[i + 1:])
    assert {e['id'] for e in entries} == {'dbv3-orbit', 'dbv3-sprout', 'dbv3-ridges'}


@pytest.mark.parametrize('split', collection.SPLITS)
def test_acquired_paths_preserve_order_direction_and_metric_size(split):
    entry = collection_entries(split)[0]
    record = artwork_record(entry)
    assert record == entry['artwork']
    strokes = planar_strokes(record)
    paths = [element for layer in record['program']['layers'] for element in layer['elements']]
    assert len(strokes) == len(paths)
    for stroke, path in zip(strokes, paths, strict=True):
        expected = np.asarray(path['points_m']) - [.015, .015]
        if path['closed']:
            expected = np.vstack([expected, expected[0]])
        assert stroke.points_m == pytest.approx(expected)
    with pytest.raises(ValueError, match='regeneration'):
        artwork_record(entry, (20, 20))
    with pytest.raises(ValueError, match='regeneration'):
        artwork_record(entry, width_mm=.5)


def test_rehashed_family_variant_cannot_cross_splits(tmp_path, monkeypatch):
    from pathlib import Path
    manifest = json.loads(collection.COLLECTION_PATH.read_text())
    duplicate = deepcopy(manifest['designs'][0])
    duplicate.update(id='same-orbit-heldout', split='test')
    manifest['designs'].append(duplicate)
    path = tmp_path / 'manifest.json'
    for entry in manifest['designs']:
        source = collection.repo_root() / 'web/inkmap/public' / entry['path']
        target = tmp_path / Path(entry['path']).parent.name / 'artwork.json'
        target.parent.mkdir(exist_ok=True)
        target.write_bytes(source.read_bytes())
        entry['path'] = str(target)
    path.write_text(json.dumps(manifest))
    monkeypatch.setattr(collection, 'COLLECTION_PATH', path)
    with pytest.raises(ValueError, match='leaks'):
        collection_entries()


def test_partial_or_traced_directory_is_not_an_acquisition(tmp_path):
    (tmp_path / 'unfinished.svg').write_text('<svg/>')
    with pytest.raises(ValueError, match='complete DBV3 acquisition'):
        directory_artifacts(tmp_path, (30, 30))
    (tmp_path / 'manifest.json').write_text(json.dumps({'schema': 'tatbot.inkgen-materialization/1', 'requested': 1, 'artifacts': []}))
    with pytest.raises(ValueError, match='complete DBV3 acquisition'):
        directory_artifacts(tmp_path, (30, 30))


def test_narrow_deposition_is_continuous_but_never_bridges_lifts_or_resets():
    surface = PlanarSurface(torch.zeros(1, 3), torch.eye(3)[None], .02, .02, 400, 400)
    field = InkField(1, surface, torch.tensor([.00015]), torch.tensor([.00015]), torch.zeros(1, 3))
    def step(x, y, down=True):
        field.deposit_segment(surface, torch.tensor([[x, y]]), torch.ones(1), torch.tensor([down]))
    step(-.005, 0.)
    step(.005, 0.)
    assert torch.all(field.field[0, 200, 103:297] > .5)
    step(.005, .005, False)
    step(-.005, .005)
    assert field.field[0, 250, 300] == 0
    field.reset()
    step(.005, -.005)
    assert field.field[0, 200, 200] == 0


def test_environment_sends_pen_up_frames_to_continuous_deposition():
    from types import SimpleNamespace

    from tatbot_sim.env import TatbotDrawEnv
    surface = PlanarSurface(torch.zeros(1, 3), torch.eye(3)[None], .02, .02, 400, 400)
    field = InkField(1, surface, torch.tensor([.00015]), torch.tensor([.00015]), torch.zeros(1, 3))
    env = SimpleNamespace(gpu_sim_enabled=False, surface=surface, _step_count=0, _dip_mask=None,
        interaction_min_m=torch.full((1,), float('inf')), interaction_max_m=torch.full((1,), -float('inf')),
        interaction_sum_m=torch.zeros(1), interaction_frames=torch.zeros(1, dtype=torch.int64),
        texture_refresh_steps=3, _pulse_emitter=lambda: None, _refresh_sheet_textures=lambda: None,
        _ink_step=lambda *_: None, _interaction_mask=lambda distance: distance.abs() < .0001,
        _apply_tool=lambda uv, incidence, touching: field.deposit_segment(surface, uv, torch.ones(1), touching))
    for position in ([0, 0, 0], [0, 0, .01], [.005, .005, 0]):
        env.agent = SimpleNamespace(tcp=SimpleNamespace(pose=SimpleNamespace(p=torch.tensor([position], dtype=torch.float32))))
        TatbotDrawEnv._after_control_step(env)
    assert field.field[0, 250, 250] == 0  # No diagonal connector across the lift.
    assert field.field[0, 300, 300] > .5
