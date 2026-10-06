"""Adversarial confidence cases: density, missing evidence, motion and edges."""
from pathlib import Path

import numpy as np
import pytest
from depth_quality import filter_depth
from surface_model import HeightFieldSurface, PlaneChart, fuse


def test_one_of_eight_and_temporally_unstable_pixels_are_not_measurements():
    raw = np.full((5, 5), 1000, np.uint16)
    support = np.full_like(raw, 8)
    support[2, 2] = 1
    mad = np.zeros_like(raw, dtype=float)
    mad[1, 1] = .01
    depth, report = filter_depth(raw, .0001, support, 8, mad)
    assert depth[2, 2] == depth[1, 1] == 0
    assert report['rejected']['low_support'] == report['rejected']['unstable'] == 1
    assert raw[2, 2] == 1000


def test_depth_discontinuity_does_not_get_smoothed_into_a_false_surface():
    raw = np.full((7, 7), 1000, np.uint16)
    raw[:, 4:] = 2000
    depth, report = filter_depth(raw, .0001)
    assert not depth[:, 3:5].any()
    assert depth[3, 1] == 1000 and depth[3, 6] == 2000
    assert report['rejected']['edge'] == 14


def test_camera_density_cannot_outvote_another_camera_and_evidence_roundtrips(tmp_path):
    chart = PlaneChart(np.zeros(3), np.eye(3))
    a = np.tile([0., 0., 0.], (300, 1))
    b = np.tile([0., 0., .001], (3, 1))
    s = fuse(np.vstack([a, b]), chart, .01, .01, .01, smooth_m=0,
             sources=['a']*len(a)+['b']*len(b))
    assert s.height[0, 0] == pytest.approx(.0005)
    assert s.source_count[0, 0] == 2
    assert s.camera_disagreement_m[0, 0] == pytest.approx(.001)
    s, _, _ = s.anchor_to([0., 0., .0005])
    s.to_npz(tmp_path/'surface.npz')
    loaded = HeightFieldSurface.from_npz(tmp_path/'surface.npz')
    assert np.array_equal(loaded.source_count, s.source_count)
    assert np.array_equal(loaded.camera_disagreement_m, s.camera_disagreement_m)
    b[:, 2] = .010
    with pytest.raises(ValueError, match='no cell'):
        fuse(np.vstack([a, b]), chart, .01, .01, .01,
             sources=['a']*len(a)+['b']*len(b))


def test_comparison_keeps_ground_truth_separate_from_stability(tmp_path):
    import importlib.util
    path = Path(__file__).resolve().parents[1]/'vision'/'depth_compare.py'
    spec = importlib.util.spec_from_file_location('depth_compare_test', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    raw = np.full((8, 5, 5), 1000, np.uint16)
    capture = tmp_path/'capture.npz'
    np.savez(capture, raw_depth_wrist_upper=raw, depth_wrist_upper=raw[0],
             units_m_wrist_upper=.0001, valid_wrist_upper=np.full((5, 5), 8, np.uint8),
             temporal_mad_m_wrist_upper=np.zeros((5, 5)),
             intrinsics_wrist_upper=[100., 100., 2., 2., 5., 5.])
    unknown = module.compare(capture, [{}])
    assert unknown['results'][0]['absolute_error'] is None
    measured = module.compare(capture, [{}], {'depth_m_wrist_upper': np.full((5, 5), .099)})
    assert measured['results'][0]['absolute_error']['signed_bias_mm'] == pytest.approx(1.)
    with pytest.raises(ValueError, match='unknown'):
        module.compare(capture, [{'misspelled_threshold': 1}])
