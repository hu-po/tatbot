"""Owner-only median capture validation; no camera SDK or arm connection."""
import json
import shutil
import time
from pathlib import Path
from types import SimpleNamespace

import capture_owner as owner_module
import numpy as np
import pytest
import vision_capture  # noqa: E402
from capture_owner import OwnerCameras  # noqa: E402
from test_rgbd_geometry import alignment


@pytest.fixture(autouse=True)
def configured_wrist_pair(tmp_path, monkeypatch):
    """The owner tests use two separately named cameras on one configured arm."""
    repo = Path(__file__).resolve().parents[2]
    (tmp_path / 'config').mkdir()
    shutil.copyfile(repo / 'config/arms.json', tmp_path / 'config/arms.json')
    vision = tmp_path / 'rust/visiond/config/vision.toml'
    vision.parent.mkdir(parents=True)
    vision.write_text('''
[[cameras.realsense]]
name = "upper"
serial = "upper"
role = "wrist_upper"
group = "d405"
arm = "right"
owner_role = "test-owner"
[[cameras.realsense]]
name = "lower"
serial = "lower"
role = "wrist_lower"
group = "d405"
arm = "right"
owner_role = "test-owner"
[[cameras.realsense]]
name = "left"
serial = "left"
role = "wrist_left"
group = "d405"
arm = "left"
owner_role = "test-owner"
''')
    actual = owner_module.capture_arm
    monkeypatch.setattr(owner_module, 'capture_arm',
                        lambda roles, **_: actual(roles, repo=tmp_path))


class Reader:
    serials = ('upper', 'lower')

    def __init__(self):
        self.socket = SimpleNamespace(settimeout=lambda _: None)
        self.number = 0
        self.units = 0.0001
        self.closed = False
        self.mismatch = False
        self.repeat = False
        self.calls = 0

    def receive(self):
        self.calls += 1
        if not self.repeat or self.calls % 2:
            self.number += 1
        frames = {}
        intr = {'schema': 'tatbot.camera-intrinsics/1', 'fx': 385., 'fy': 385.,
                'ppx': 1., 'ppy': 1., 'width': 2, 'height': 2,
                'distortion_model': 'None', 'distortion_coefficients': [0.] * 5}
        for serial in self.serials:
            color_name = serial + '_color'
            attrs = {'device_serial': serial, 'frame_number': str(self.number),
                     'intrinsics': json.dumps(intr), 'depth_units_m': str(self.units),
                     'aligned_to': color_name, 'alignment_calibration': json.dumps(alignment(intr))}
            meta = {'sequence': self.number, 'attributes': attrs,
                    'timestamps': {'host_unix_ns': time.time_ns()}}
            frames[serial + '_depth'] = {'metadata': meta, 'depth': np.full((2, 2), 1000 + self.number, dtype=np.uint16)}
            frames[color_name] = {'metadata': {**meta, 'sequence': self.number + int(self.mismatch)},
                                  'image': np.full((2, 2, 3), 100, dtype=np.uint8)}
        return {'sequence': self.number, 'frames': frames}

    def close(self):
        self.closed = True


def test_owner_capture_preserves_metric_depth_and_distinct_provenance(tmp_path):
    reader = Reader()
    reader.repeat = True
    cams = OwnerCameras({'wrist_upper': 'upper', 'wrist_lower': 'lower'}, reader)
    try:
        path = vision_capture.capture(cams, tmp_path, 1, [0.] * 6, .002, time.time(), 8)
        with np.load(path) as arrays:
            for role in ('wrist_upper', 'wrist_lower'):
                assert arrays['depth_' + role].dtype == np.uint16
                profile = json.loads(str(arrays['owner_profile_' + role]))
                assert profile['alignment_calibration']['aligned_depth_value_axis'] == 'native_depth_z'
                assert np.all(arrays['valid_' + role] == 8)
                assert arrays['units_m_' + role] == .0001
                records = json.loads(str(arrays['owner_frames_' + role]))
                ids = [r['metadata']['attributes']['frame_number'] for r in records]
                assert len(ids) == len(set(ids)) == 8
                color_metadata = json.loads(str(arrays['owner_color_metadata_' + role]))
                assert color_metadata['sequence'] == records[-1]['metadata']['sequence']
                assert np.all(arrays['raw_depth_' + role][-1] == 1000 + color_metadata['sequence'])
                assert arrays['intrinsics_' + role].tolist() == [385., 385., 1., 1., 2., 2.]
        assert reader.calls > 9  # Repeats were discarded, not counted as median samples.
    finally:
        cams.close()
    assert reader.closed


def test_changed_owner_profile_refuses_before_done_file(tmp_path):
    reader = Reader()
    cams = OwnerCameras({'wrist_upper': 'upper', 'wrist_lower': 'lower'}, reader)
    reader.units = .001
    try:
        with pytest.raises(ValueError, match='profile changed'):
            vision_capture.capture(cams, tmp_path, 1, [0.] * 6, .002, time.time(), 8)
        assert not (tmp_path / 'capture-1.done').exists()
    finally:
        cams.close()


def test_single_manifested_camera_keeps_evidence_and_excludes_other_arm(tmp_path):
    reader = Reader()  # The source also offers a second device; it is not selected.
    cams = OwnerCameras({'wrist_upper': 'upper'}, reader)
    try:
        path = vision_capture.capture(cams, tmp_path, 1, [0.] * 6, .002, time.time(), 8)
        with np.load(path) as arrays:
            assert json.loads(str(arrays['camera_roles'])) == ['wrist_upper']
            assert str(arrays['camera_serial_wrist_upper']) == 'upper'
            assert not any('wrist_lower' in name for name in arrays.files)
            assert np.all(arrays['valid_wrist_upper'] == 8)
            assert len(json.loads(str(arrays['owner_frames_wrist_upper']))) == 8
    finally:
        cams.close()
    assert reader.closed


def test_the_left_wrist_answers_under_its_own_role_from_its_owner(tmp_path):
    """The left arm's one wrist camera (`wrist_left`, the `left-wrist-cameras`
    owner's stream) answers a capture keyed by its own role."""
    reader = Reader()
    reader.serials = ('left', 'upper')  # the stream offers another device too
    cams = OwnerCameras({'wrist_left': 'left'}, reader)
    try:
        assert cams.arm == 'left'
        path = vision_capture.capture(cams, tmp_path, 1, [0.] * 6, .002, time.time(), 8)
        with np.load(path) as arrays:
            assert json.loads(str(arrays['camera_roles'])) == ['wrist_left']
            assert str(arrays['camera_serial_wrist_left']) == 'left'
            assert np.all(arrays['valid_wrist_left'] == 8)
            assert not any('wrist_upper' in name for name in arrays.files)
    finally:
        cams.close()
    assert reader.closed


@pytest.mark.parametrize('registry', [{}, {'wrist_probe': 'probe'}, {'wrist_left': 'left', 'wrist_upper': 'upper'},
                                    {'wrist_upper': 'same', 'wrist_lower': 'same'}])
def test_invalid_selected_camera_registry_refuses(registry):
    with pytest.raises(ValueError, match='distinct manifested wrist cameras of one arm'):
        OwnerCameras(registry, Reader())


def test_persistent_mismatched_rgbd_sequence_times_out_and_closes(monkeypatch):
    reader = Reader()
    reader.mismatch = True
    clock = iter([0., 1., 6.])
    monkeypatch.setattr('lib.capture_owner.time.monotonic', lambda: next(clock))
    with pytest.raises(TimeoutError, match='unmatched RGBD sets: 1'):
        OwnerCameras({'wrist_upper': 'upper', 'wrist_lower': 'lower'}, reader)
    assert reader.closed


def test_transient_mismatch_is_discarded_without_counting_or_mixing_frames():
    class StraddledReader(Reader):
        def receive(self):
            self.mismatch = self.calls % 2 == 0
            return super().receive()

    reader = StraddledReader()
    cams = OwnerCameras({'wrist_upper': 'upper', 'wrist_lower': 'lower'}, reader)
    try:
        cams.begin_capture(8)
        assert cams.batch['unmatched_rgbd_sets'] == 8
        for cam in cams:
            ids = [r['metadata']['sequence'] for r in cam.records]
            assert ids == list(range(4, 19, 2))
            assert [int(d[0, 0]) for d in cam.depths] == [1000 + n for n in ids]
            assert json.loads(str(cam.evidence_arrays()['owner_batch_' + cam.role])) == cams.batch
    finally:
        cams.close()


def test_wrong_camera_identity_still_refuses_immediately():
    class WrongCameraReader(Reader):
        def receive(self):
            frame_set = super().receive()
            frame_set['frames']['upper_color']['metadata']['attributes'] = {'device_serial': 'other'}
            return frame_set

    reader = WrongCameraReader()
    with pytest.raises(ValueError, match='identity mismatch'):
        OwnerCameras({'wrist_upper': 'upper', 'wrist_lower': 'lower'}, reader)
    assert reader.closed


def test_newly_arrived_but_old_exposure_is_not_a_fresh_capture(monkeypatch):
    class DelayedReader(Reader):
        def receive(self):
            frame_set = super().receive()
            for entry in frame_set['frames'].values():
                entry['metadata']['timestamps']['normalized_unix_ns'] = time.time_ns()-1_000_000_000
                entry['metadata']['attributes']['actual_exposure_us'] = '1000'
            return frame_set

    reader = DelayedReader()
    clock = iter([0., 1., 6.])
    monkeypatch.setattr('lib.capture_owner.time.monotonic', lambda: next(clock))
    with pytest.raises(TimeoutError):
        OwnerCameras({'wrist_upper': 'upper', 'wrist_lower': 'lower'}, reader)
    assert reader.closed


def test_owner_restart_invalidates_the_burst_even_when_counter_increases():
    class RestartedReader(Reader):
        epoch = 'first'

        def receive(self):
            frame_set = super().receive()
            for entry in frame_set['frames'].values():
                entry['metadata']['attributes']['capture_epoch'] = self.epoch
            return frame_set

    reader = RestartedReader()
    cams = OwnerCameras({'wrist_upper': 'upper', 'wrist_lower': 'lower'}, reader)
    try:
        reader.epoch = 'restarted'
        with pytest.raises(ValueError, match='profile changed'):
            cams.begin_capture(8)
    finally:
        cams.close()


def test_owner_alignment_change_invalidates_a_median_batch():
    class ChangingAlignment(Reader):
        shifted = False

        def receive(self):
            frame_set = super().receive()
            for entry in frame_set['frames'].values():
                attrs = entry['metadata']['attributes']
                model = json.loads(attrs['alignment_calibration'])
                model['translation'][2] = .001 if self.shifted else 0.
                attrs['alignment_calibration'] = json.dumps(model)
            return frame_set

    reader = ChangingAlignment()
    cams = OwnerCameras({'wrist_upper': 'upper'}, reader)
    try:
        reader.shifted = True
        with pytest.raises(ValueError, match='profile changed'):
            cams.begin_capture(8)
    finally:
        cams.close()
