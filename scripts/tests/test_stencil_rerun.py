"""Display caps, measured geometry, and removal of stale surface overlays."""

from types import SimpleNamespace

import numpy as np
import pytest
from stencil_rerun import MAX_DEPTH_POINTS, ROOT, StencilRerun, camera_entity, depth_points  # noqa: E402
from stencil_surface import estimate_surface  # noqa: E402
from test_stencil_surface import geometry_fixture  # noqa: E402


def test_depth_cloud_preserves_optical_geometry_and_holes():
    _, frame = geometry_fixture()
    frame['image'][:] = [10, 20, 30]
    full, colors, _ = depth_points(frame)
    assert 0 < len(full) <= MAX_DEPTH_POINTS
    np.testing.assert_allclose(full[:, 2], .45)
    np.testing.assert_array_equal(colors[0], [30, 20, 10])
    frame['depth_m'][80:160, 100:220] = 0
    partial, _, _ = depth_points(frame)
    assert len(partial) < len(full)
    assert camera_entity('realsense1_color') == 'cameras/06_realsense1/color'
    assert camera_entity('overhead_depth_color') == 'cameras/08_overhead_depth/color'


class RecordingSpy:
    def __init__(self):
        self.logs = []

    def log(self, path, value, **kwargs):
        self.logs.append((path, value))

    def __getattr__(self, name):
        def archetype(*args, **kwargs):
            return SimpleNamespace(kind=name, args=args, kwargs=kwargs, compress=lambda **kw: name)
        return archetype


def test_lost_surface_clears_mesh_and_pose_in_the_next_display_frame(monkeypatch):
    import stencil_rerun
    stamps = []
    monkeypatch.setattr(stencil_rerun.tr, 'set_capture_time', stamps.append)
    display = StencilRerun.__new__(StencilRerun)
    display.rr, display.live, display.next_capture = RecordingSpy(), False, 0
    row, frame = geometry_fixture()
    surface, mesh = estimate_surface(row, frame)
    row.update(pattern_id='known-pattern', seed='known-seed', status='detected', surface=surface)
    display.log(frame, row, {'known-pattern': mesh})
    assert any(value.kind == 'Mesh3D' for _, value in display.rr.logs if not isinstance(value, str))
    assert any(path.endswith('/axes') for path, _ in display.rr.logs)
    count = len(display.rr.logs)
    frame['timestamp_ns'] += 100_000_000
    display.log(frame, row, {})
    assert len(display.rr.logs) == count  # at most five display frames per capture second
    frame['timestamp_ns'] += 100_000_000
    frame['color_metadata']['timestamps']['normalized_unix_ns'] = frame['timestamp_ns']
    lost = dict(row, status='lost', surface={'candidate_valid': False, 'reason': 'image_tracking_unavailable'})
    display.log(frame, lost, {})
    logs = display.rr.logs[count:]
    assert any(path == ROOT+'/stencils' and value.kind == 'Clear' for path, value in logs)
    assert not any(path.endswith(('/measured_patch', '/axes', '/center')) for path, _ in logs)
    assert stamps == [1_000_000_000, 1_200_000_000]
    count = len(display.rr.logs)
    frame['timestamp_ns'] += 200_000_000  # incompatible capture metadata must clear the old depth view
    display.log(frame, lost, {})
    logs = display.rr.logs[count:]
    assert any(path.endswith('/depth') and value.kind == 'Clear' for path, value in logs)
    assert not any(path == ROOT+'/raw_depth' and value.kind == 'Points3D' for path, value in logs)


def test_measured_mesh_serializes_with_the_pinned_rerun_sdk(tmp_path):
    pytest.importorskip('rerun')
    import warnings
    args = SimpleNamespace(connect=None, rerun=True, recording_id='fixture-surface', mode='replay')
    with warnings.catch_warnings():
        warnings.simplefilter('error')
        display = StencilRerun(args, tmp_path, 'fixture-surface')
        row, frame = geometry_fixture(curvature=4.)
        surface, mesh = estimate_surface(row, frame)
        row.update(pattern_id='fixture', seed='fixture', status='detected', surface=surface)
        display.log(frame, row, {'fixture': mesh})
        display.close()
    assert (tmp_path/'replay.rrd').stat().st_size > 1000
