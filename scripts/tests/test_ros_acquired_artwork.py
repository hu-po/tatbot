"""A genuine bundled acquisition through ROS preparation, without hardware."""
import copy
import json
from pathlib import Path

import numpy as np
import pytest
import tatbot_ink
from tatbot_contracts.canonical import canonical_digest
from tatbot_ink import CompileError

REPO = Path(__file__).resolve().parents[2]


def test_bundled_native_acquisition_keeps_source_recipe_pen_and_traversal(tmp_path):
    from tatbot_contracts.ros_program import validate_for_execution
    source = REPO/'web/inkmap/public/designs/dbv3-orbit/artwork.json'
    art = json.loads(source.read_text())
    result = tatbot_ink.compile(source, repo=REPO)
    validate_for_execution(result)
    assert result['preparation']['adapter'] == 'dbv3-paths-to-ros/2'
    paths = [(layer['id'], layer['ink_id'], path) for layer in art['program']['layers'] for path in layer['elements']]
    ops = [op for op in result['ops'] if op['op'] == 'stroke']
    assert len(ops) == len(paths)
    for op, (layer, pen, path) in zip(ops, paths, strict=True):
        assert op['src']['artwork_sha256'] == art['content_sha256']
        assert (op['src']['layer'], op['src']['path'], op['src']['pen']) == (layer, path['id'], pen)
        expected = np.asarray(path['points_m']) - [.015, .015]
        if path['closed'] and not np.array_equal(expected[0], expected[-1]):
            expected = np.vstack([expected, expected[0]])
        np.testing.assert_allclose(op['points_m'], expected, atol=1e-9)
    assert art['conversion']['recipe_sha256'] is not None
    with pytest.raises(CompileError, match='regenerate'):
        tatbot_ink.compile(source, repo=REPO, width_m=.035)
    legacy = copy.deepcopy(art)
    legacy['conversion'].update(adapter='tatbot-svg-paint/1', recipe_sha256=None)
    legacy['content_sha256'] = canonical_digest(legacy)
    legacy_path = tmp_path/'legacy.json'
    legacy_path.write_text(json.dumps(legacy))
    with pytest.raises(CompileError, match='DrawingBot V3'):
        tatbot_ink.compile(legacy_path, repo=REPO)
