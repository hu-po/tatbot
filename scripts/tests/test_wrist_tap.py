"""The wrist D405 display tap beside the session's own grab_both, on a stub pyrealsense2 whose pipeline fails any
second thread inside it: the two never share the pipeline, the tap never grabs, a bus error only turns the tap
off, close() waits for nothing yet the tap never touches a stopped pipeline, a snapshot camera closed straight
after its grab still publishes it, and a camera dropped without close() is not kept alive by its tap. Each camera
publishes as the vision.toml camera of its serial, and stack.yaml's preview cameras open beside the session's,
polled, and close with it. The
threading tests stand a constant in for the JPEG codec (they run in the light profile); the message test encodes
with OpenCV."""
from __future__ import annotations

import collections
import contextlib
import functools
import gc
import json
import sys
import threading
import time
import types
import weakref
from pathlib import Path

import numpy as np
import pytest
import wrist_cameras
import wrist_tap
from frame_set_wire import unpack
from tatbot_session import inspect as ins

REPO = Path(__file__).resolve().parents[2]
OPENCV_JPEG = wrist_tap.encode_jpeg
UNITS = float(np.float32(0.0001))     # what get_depth_scale() returns on the D405


class Guard:
    """Records (and fails) a second thread entering the pipeline, and counts each method's calls and threads."""

    def __init__(self):
        self._inside = threading.Lock()
        self.calls = collections.Counter()
        self.threads = collections.defaultdict(set)
        self.overlaps = []

    @contextlib.contextmanager
    def enter(self, name):
        if not self._inside.acquire(blocking=False):
            self.overlaps.append(name)
            raise AssertionError(f'{name}: the pipeline was entered concurrently')
        try:
            self.calls[name] += 1
            self.threads[name].add(threading.current_thread().name)
            time.sleep(0.0002)                       # widen the window a race would need
            yield
        finally:
            self._inside.release()


class Frame:
    def __init__(self, data, stamp_ms):
        self.data, self.stamp_ms = data, stamp_ms

    def get_data(self):
        return self.data

    def get_timestamp(self):
        return self.stamp_ms

    def get_frame_timestamp_domain(self):
        return 'global_time'


class FrameSet:
    def __init__(self, color, depth, n):
        self.color, self.depth = Frame(color, n), Frame(depth, n)

    def __bool__(self):
        return True

    def get_color_frame(self):
        return self.color

    def get_depth_frame(self):
        return self.depth


class Optics:
    def __init__(self, width, height, f):
        self.width, self.height, self.fx, self.fy = width, height, f, f + 1.0
        self.ppx, self.ppy, self.coeffs, self.model = width / 2, height / 2, [0.0] * 5, 'distortion.brown_conrady'

    def as_video_stream_profile(self):
        return self

    def get_intrinsics(self):
        return self

    def get_extrinsics_to(self, _other):
        return types.SimpleNamespace(rotation=[1.0, 0, 0, 0, 1.0, 0, 0, 0, 1.0], translation=[0.0, 0.0, 0.0])


class Pipeline:
    """Frame k: colour pixels k % 256, depth k. Every other poll finds no frame ready."""

    def __init__(self, rs):
        self.rs, self.guard, self.n, self.size, self.on_stop = rs, rs.guard, 0, None, None
        rs.pipelines.append(self)

    def _frames(self):
        self.n += 1
        w, h = self.size
        return FrameSet(np.full((h, w, 3), self.n % 256, np.uint8), np.full((h, w), self.n, np.uint16), self.n)

    def start(self, cfg):
        with self.guard.enter('start'):
            self.size = cfg.size
            w, h = cfg.size
            streams = {'color': Optics(w, h, 600.0), 'depth': Optics(w, h, 400.0)}
            serial = getattr(cfg, 'serial', '000000000000')   # the device enable_device asked for, as a real one
            device = types.SimpleNamespace(get_info=lambda _key: serial, first_depth_sensor=lambda: types.
                                           SimpleNamespace(get_depth_scale=lambda: UNITS))
            return types.SimpleNamespace(get_stream=streams.__getitem__, get_device=lambda: device)

    def wait_for_frames(self, _timeout_ms):
        with self.guard.enter('wait_for_frames'):
            return self._frames()

    def poll_for_frames(self):
        with self.guard.enter('poll_for_frames'):
            return self._frames() if self.guard.calls['poll_for_frames'] % 2 else Empty()

    def stop(self):
        with self.guard.enter('stop'):
            if self.on_stop:
                self.on_stop()


class Empty:
    def __bool__(self):
        return False


def stub_realsense(monkeypatch):
    rs = types.ModuleType('pyrealsense2')
    rs.guard, rs.pipelines = Guard(), []
    rs.stream = types.SimpleNamespace(color='color', depth='depth')
    rs.format = types.SimpleNamespace(bgr8='bgr8', z16='z16')
    rs.camera_info = types.SimpleNamespace(serial_number='serial_number')

    class Config:
        def enable_device(self, serial):
            self.serial = serial

        def enable_stream(self, stream, width, height, _fmt, _fps):
            self.size = (width, height)

    class Align:
        def __init__(self, _to):
            pass

        def process(self, frames):
            with rs.guard.enter('align'):
                return frames

    rs.config, rs.align, rs.pipeline = Config, Align, lambda: Pipeline(rs)
    monkeypatch.setitem(sys.modules, 'pyrealsense2', rs)
    return rs


class Bus:
    """A zenoh session double: publishers keyed by name, puts kept (or raising `fail`)."""

    def __init__(self, fail=None, delay_s=0.0):
        self.fail, self.delay_s, self.declared, self.sent, self.closed = fail, delay_s, [], [], 0

    def declare_publisher(self, key, **options):
        self.declared.append((key, options))
        return types.SimpleNamespace(put=self._put)

    def _put(self, payload):
        time.sleep(self.delay_s)
        if self.fail:
            raise self.fail
        self.sent.append(bytes(payload))

    def close(self):
        self.closed += 1


@pytest.fixture
def rig(monkeypatch):
    zenoh = types.SimpleNamespace(CongestionControl=types.SimpleNamespace(DROP='drop'),
                                  Priority=types.SimpleNamespace(DATA_LOW='data_low'))
    monkeypatch.setitem(sys.modules, 'zenoh', zenoh)
    monkeypatch.setattr(wrist_tap, 'encode_jpeg', lambda color: b'\xff\xd8 a constant JPEG stand-in \xff\xd9')
    return stub_realsense(monkeypatch)


def camera(bus, logs, *, poll=False, width=64, height=48, aligned=False, open_bus=None):
    tap = wrist_tap.factory({'enabled': True, 'poll': poll}, repo=REPO, arm='right',
                            open_bus=open_bus or (lambda: bus), frames=(ins.CAMERA_FRAME, ins.DEPTH_FRAME),
                            log=logs.append)
    return ins.WristCamera(width=width, height=height, depth=True, aligned=aligned, tap=tap)


def wait_for(condition, timeout_s=5.0):
    deadline = time.monotonic() + timeout_s
    while not condition() and time.monotonic() < deadline:
        time.sleep(0.005)
    return condition()


def mounted():
    """The vision.toml camera a WristCamera with no serial opens (stack.yaml page.locate.camera's)."""
    serial = ins.configured_serial()
    return next(c for c in wrist_cameras.mounted_for_arm(REPO, 'right') if c['serial'] == serial)


def source(blob):
    return unpack(blob)[0]['frames'][0]['metadata']['attributes']['tap_source']


def no_router():
    raise OSError('no router')


def test_the_tap_and_grab_both_never_share_the_pipeline_and_the_tap_never_grabs(rig, monkeypatch):
    monkeypatch.setattr(wrist_tap, 'POLL_S', 0.001)
    monkeypatch.setattr(wrist_tap, 'HOLD_S', 0.0)     # polls between grabs, to race them
    bus, logs = Bus(), []
    cam = camera(bus, logs, poll=True)
    grab, callers, last = cam.grab_both, set(), set()

    def recording(*args, **kwargs):
        callers.add(threading.current_thread().name)
        return grab(*args, **kwargs)

    cam.grab_both = recording
    for i in range(300):
        cam.grab_both(2)
        last.add(int(cam.raw_depth[-1][0, 0]))
        if i % 25 == 0:
            time.sleep(0.02)                         # room for the 2 Hz (here 1 kHz) poll
    assert wait_for(lambda: any(source(blob) == 'poll' for blob in bus.sent))
    guard, polls_at_stop = rig.guard, []
    rig.pipelines[0].on_stop = lambda: polls_at_stop.append(guard.calls['poll_for_frames'])
    cam.close()
    assert wait_for(lambda: not cam.tap.thread.is_alive())

    assert guard.overlaps == [] and logs == []
    assert callers == {threading.current_thread().name}                 # only the session grabs
    assert guard.threads['wait_for_frames'] == {threading.current_thread().name}
    assert guard.threads['poll_for_frames'] == {'wrist-tap'} and guard.calls['poll_for_frames'] > 2
    assert polls_at_stop == [guard.calls['poll_for_frames']]           # no poll of a stopped pipeline
    assert guard.calls['stop'] == 1 and wait_for(lambda: bus.closed == 1)
    name = mounted()['name']
    assert bus.declared == [(f'tatbot/vision/d405/{name}',
                             {'congestion_control': 'drop', 'priority': 'data_low', 'express': True})]
    sequences = []
    for blob in bus.sent:
        header, frames = unpack(blob)
        assert set(frames) == {f'{name}_color', f'{name}_depth'}
        depth_meta, _, raw = frames[f'{name}_depth']
        if depth_meta['attributes']['tap_source'] == 'grab':
            assert int(np.frombuffer(raw, '<u2')[0]) in last       # each grab_both call's LAST frame
        sequences.append(header['sequence'])
    assert sequences == sorted(sequences) and len(set(sequences)) == len(sequences)


def test_a_grab_holds_the_poll_so_the_frame_a_reading_read_stays_shown(rig, monkeypatch):
    """A preview camera a reading grabs (the tracker's further views): no polled frame replaces the grabbed one on
    the viewer for HOLD_S, the time its markers take to follow it; then the poll goes on."""
    monkeypatch.setattr(wrist_tap, 'POLL_S', 0.001)
    monkeypatch.setattr(wrist_tap, 'HOLD_S', 0.3)
    bus, logs = Bus(), []
    cam = camera(bus, logs, poll=True)
    assert wait_for(lambda: any(source(blob) == 'poll' for blob in bus.sent))
    cam.grab_both(1)
    held = time.monotonic()
    assert wait_for(lambda: source(bus.sent[-1]) == 'grab')
    sent = len(bus.sent)
    time.sleep(max(0.0, 0.2 - (time.monotonic() - held)))
    assert len(bus.sent) == sent                     # held: nothing after the grabbed frame
    assert wait_for(lambda: source(bus.sent[-1]) == 'poll')   # and polled again after HOLD_S
    cam.close()
    assert logs == [] and rig.guard.overlaps == []


def test_the_hand_off_is_a_reference_and_the_newest_frame_wins(rig):
    gate = threading.Event()
    bus, logs = Bus(), []
    bus._put = lambda payload: (gate.wait(5), bus.sent.append(bytes(payload)))
    cam = camera(bus, logs)
    colors, _ = cam.grab_both(1)                     # the tap takes this one and blocks in put
    assert wait_for(lambda: not cam.tap._slot)
    for _ in range(3):
        colors, _ = cam.grab_both(2)
    (queued,) = cam.tap._slot
    assert queued[0] is colors[-1] and queued[1] is cam.raw_depth[-1]   # no copy in the caller's thread
    gate.set()
    assert wait_for(lambda: len(bus.sent) == 2)
    cam.close()
    assert rig.guard.calls['poll_for_frames'] == 0 and logs == []


def test_a_bus_error_turns_the_tap_off_and_never_reaches_the_session(rig):
    bus, logs = Bus(fail=RuntimeError('bus down')), []
    cam = camera(bus, logs)
    for _ in range(5):
        cam.grab_both(1)
    assert wait_for(lambda: not cam.tap.thread.is_alive())
    assert not cam.tap.enabled and len(logs) == 1 and 'bus down' in logs[0]
    for _ in range(5):
        cam.grab_both(1)                             # still grabs; the tap stays off and quiet
    cam.close()
    assert len(logs) == 1 and rig.guard.calls['stop'] == 1 and wait_for(lambda: bus.closed == 1)

    logs = []
    cam = camera(None, logs, open_bus=no_router)
    cam.grab_both(1)
    assert wait_for(lambda: not cam.tap.thread.is_alive()) and len(logs) == 1 and 'no router' in logs[0]
    cam.close()


@pytest.mark.parametrize('opening', [False, True])
def test_a_snapshot_closed_straight_after_its_grab_still_publishes_and_close_waits_for_nothing(rig, opening):
    """ArmExecutor._wrist_frames (gauge holds, trip snapshots, dip looks): open, grab_both(1), close() at once,
    with the tap thread idle or still opening its bus. close() returns before the bus is open."""
    bus, logs, gate = Bus(), [], threading.Event()
    cam = camera(bus, logs, open_bus=lambda: (gate.wait(5), bus)[1])
    if not opening:
        gate.set()
        assert wait_for(lambda: cam.tap.bus is not None)
    cam.grab_both(1)
    start = time.monotonic()
    cam.close()
    assert time.monotonic() - start < 0.25
    gate.set()
    assert wait_for(lambda: len(bus.sent) == 1) and wait_for(lambda: not cam.tap.thread.is_alive())
    header, frames = unpack(bus.sent[0])
    assert header['sequence'] == 1 and source(bus.sent[0]) == 'grab'
    raw = frames[f"{mounted()['name']}_depth"][2]
    assert np.array_equal(np.frombuffer(raw, '<u2').reshape(cam.raw_depth[-1].shape), cam.raw_depth[-1])
    assert len(bus.sent) == 1 and bus.closed == 1 and logs == [] and rig.guard.overlaps == []


def test_a_camera_dropped_without_close_is_not_kept_alive_by_its_tap(rig):
    """A goal that fails between opening the camera and closing it leaves the D405 to the garbage collector, as
    before the tap: the tap holds its camera weakly, sends what it was handed, and ends with the camera."""
    bus, logs = Bus(), []
    cam = camera(bus, logs)
    cam.grab_both(1)
    thread, ref = cam.tap.thread, weakref.ref(cam)
    del cam
    gc.collect()
    assert ref() is None and wait_for(lambda: not thread.is_alive())
    assert len(bus.sent) == 1 and bus.closed == 1 and logs == []


def test_off_in_stack_yaml_means_no_tap_at_all(rig):
    assert wrist_tap.factory({'enabled': False, 'poll': True}, repo=REPO, arm='right', open_bus=None,
                             frames=()) is None
    assert wrist_tap.factory(None, repo=REPO, arm='right', open_bus=None, frames=()) is None
    before = set(threading.enumerate())             # another test's tap may still be ending
    cam = ins.WristCamera(depth=True, tap=None)
    cam.grab_both(1)
    cam.close()
    assert cam.tap is None and not any(t.name == 'wrist-tap' for t in set(threading.enumerate()) - before)


@pytest.mark.parametrize('aligned', [False, True])
def test_a_message_carries_the_contract_metadata_at_the_d405_size(rig, aligned, monkeypatch):
    cv2 = pytest.importorskip('cv2')
    monkeypatch.setattr(wrist_tap, 'encode_jpeg', OPENCV_JPEG)
    bus, logs = Bus(), []
    cam = camera(bus, logs, width=1280, height=720, aligned=aligned)
    before = time.time_ns()
    colors, _ = cam.grab_both(1)
    after = time.time_ns()
    assert wait_for(lambda: bus.sent)
    cam.close()
    camera_entry, name = mounted(), mounted()['name']
    header, frames = unpack(bus.sent[0])
    assert header['envelope']['schema'] == 'tatbot.frame-set/1' and header['envelope']['seq'] == header['sequence']
    color_meta, color_payload, jpeg = frames[f'{name}_color']
    depth_meta, depth_payload, raw = frames[f'{name}_depth']
    for meta in (color_meta, depth_meta):
        attrs = meta['attributes']
        assert (attrs['physical_arm'], attrs['device_serial'], attrs['capture_owner_role']) == \
            ('right', camera_entry['serial'], camera_entry['owner_role'])
        assert before <= meta['timestamps']['normalized_unix_ns'] <= after
    color_optics, depth_optics = (json.loads(m['attributes']['intrinsics']) for m in (color_meta, depth_meta))
    assert color_optics['fx'] == 600.0 and (color_optics['width'], color_optics['height']) == (1280, 720)
    assert depth_optics['fx'] == (600.0 if aligned else 400.0)
    assert depth_optics['schema'] == 'tatbot.camera-intrinsics/1' and depth_optics['distortion_model'] == 'BrownConrady'
    assert color_meta['attributes']['ros_frame'] == 'right/realsense_color_optical_frame'
    assert depth_meta['attributes']['ros_frame'] == ('right/realsense_color_optical_frame' if aligned
                                                     else 'right/realsense_depth_optical_frame')
    assert depth_meta['attributes'].get('aligned_to') == (f'{name}_color' if aligned else None)
    assert depth_meta['attributes']['depth_units_m'] == '0.0001' and 'depth_units_m' not in color_meta['attributes']
    assert color_meta['profile']['format'] == 'bgr8' and depth_meta['profile']['format'] == 'z16'
    assert color_payload == {'format': 'jpeg', 'bytes': len(jpeg)}
    assert depth_payload == {'width': 1280, 'height': 720, 'bytes': 1280 * 720 * 2}
    image = cv2.imdecode(np.frombuffer(jpeg, np.uint8), cv2.IMREAD_COLOR)
    assert image.shape == (720, 1280, 3) and abs(int(image.mean()) - int(colors[-1].mean())) <= 2
    assert np.array_equal(np.frombuffer(raw, '<u2').reshape(720, 1280), cam.raw_depth[-1])


# --- two D405s on one wrist: each publishes as its own camera; a preview opens beside the session's -------------------

TWO_WRISTS = """
[[cameras.realsense]]
name = "upper"
serial = "upper-serial"
role = "wrist_upper"
group = "d405"
arm = "right"
owner_role = "ros"
optical_frame = "right/realsense_color_optical_frame"
depth_optical_frame = "right/realsense_depth_optical_frame"

[[cameras.realsense]]
name = "lower"
serial = "lower-serial"
role = "wrist_lower"
group = "d405"
arm = "right"
owner_role = "ros"
optical_frame = "right/realsense_lower_color_optical_frame"
depth_optical_frame = "right/realsense_lower_depth_optical_frame"
"""
PREVIEW = {'enabled': True, 'poll': False, 'preview': {'cameras': ['wrist_lower'], 'width': 32, 'height': 24, 'fps': 15}}


@pytest.fixture
def two_wrists(monkeypatch, tmp_path):
    """The rig's registry since 2026-10-23, the upper camera stack.yaml's: the stub's devices by these serials."""
    path = tmp_path / 'vision.toml'
    path.write_text(TWO_WRISTS)
    monkeypatch.setattr(wrist_cameras, 'registry_path', lambda _repo: path)


def own_guards(rig, monkeypatch):
    """A guard per pipeline: two cameras' pipelines are used at once, each by one thread at a time."""
    def pipeline():
        pipe = Pipeline(rig)
        pipe.guard = Guard()
        return pipe
    monkeypatch.setattr(rig, 'pipeline', pipeline)


def test_each_camera_publishes_as_the_vision_toml_camera_of_its_serial(rig, two_wrists):
    bus, logs = Bus(), []
    lower = ins.WristCamera('lower-serial', width=64, height=48, depth=True, tap=wrist_tap.factory(
        {'enabled': True}, repo=REPO, arm='right', open_bus=lambda: bus, log=logs.append,
        frames=('right/realsense_lower_color_optical_frame', 'right/realsense_lower_depth_optical_frame')))
    lower.grab_both(1)
    assert wait_for(lambda: bus.sent)
    lower.close()
    assert [key for key, _ in bus.declared] == ['tatbot/vision/d405/lower']
    frames = unpack(bus.sent[0])[1]
    assert frames['lower_color'][0]['attributes']['ros_frame'] == 'right/realsense_lower_color_optical_frame'
    assert frames['lower_depth'][0]['attributes']['device_serial'] == 'lower-serial' and logs == []

    stranger = camera(bus, logs)                     # stack.yaml's role, now on a device vision.toml lacks
    stranger.serial = 'stranger'
    stranger.tap = wrist_tap.factory({'enabled': True}, repo=REPO, arm='right', open_bus=lambda: bus,
                                     log=logs.append, frames=(ins.CAMERA_FRAME, ins.DEPTH_FRAME))(stranger)
    assert wait_for(lambda: logs) and 'no wrist camera of serial stranger' in logs[-1]
    stranger.close()


def test_a_preview_opens_beside_the_session_camera_polled_and_closes_with_it(rig, two_wrists, monkeypatch):
    own_guards(rig, monkeypatch)
    monkeypatch.setattr(wrist_tap, 'POLL_S', 0.001)
    bus, logs = Bus(), []
    shared = {'repo': REPO, 'arm': 'right', 'open_bus': lambda: bus, 'log': logs.append}
    tap = wrist_tap.factory(PREVIEW, frames=(ins.CAMERA_FRAME, ins.DEPTH_FRAME), **shared)
    preview = wrist_tap.previews(PREVIEW, open_camera=functools.partial(ins.WristCamera, depth=True), **shared)
    cam = ins.WristCamera(width=64, height=48, depth=True, tap=tap, preview=preview)
    assert cam.serial == 'upper-serial'
    assert wait_for(lambda: len(cam.preview.cameras) == 1)
    (lower,) = cam.preview.cameras
    assert (lower.serial, lower.fps, lower.depth, rig.pipelines[1].size) == ('lower-serial', 15, True, (32, 24))
    cam.grab_both(1)
    names = lambda: {name for blob in bus.sent for name in unpack(blob)[1]}   # noqa: E731
    assert wait_for(lambda: {'upper_color', 'lower_color'} <= names())
    cam.close()
    assert wait_for(lambda: not cam.tap.thread.is_alive() and not lower.tap.thread.is_alive())
    assert [pipe.guard.calls['stop'] for pipe in rig.pipelines] == [1, 1] and cam.preview.cameras == []
    assert sorted(key for key, _ in bus.declared) == ['tatbot/vision/d405/lower', 'tatbot/vision/d405/upper']
    for blob in bus.sent:
        (name, (meta, _, _)), = [(n, f) for n, f in unpack(blob)[1].items() if n.endswith('_color')]
        upper = name == 'upper_color'
        assert meta['attributes']['tap_source'] == ('grab' if upper else 'poll')
        assert meta['attributes']['ros_frame'] == ('right/realsense_color_optical_frame' if upper
                                                   else 'right/realsense_lower_color_optical_frame')
        assert (meta['profile']['width'], meta['profile']['height']) == ((64, 48) if upper else (32, 24))
    assert all(pipe.guard.overlaps == [] for pipe in rig.pipelines)
    assert rig.pipelines[0].guard.calls['poll_for_frames'] == 0 and logs == []   # the session's camera is never polled


def test_a_preview_still_opening_at_close_is_closed_once_open_and_one_not_mounted_is_logged(two_wrists):
    assert wrist_tap.previews(None, repo=REPO, arm='right', open_camera=None, open_bus=None) is None
    assert wrist_tap.previews({**PREVIEW, 'enabled': False}, repo=REPO, arm='right', open_camera=None,
                              open_bus=None) is None
    entered, gate, opened, logs = threading.Event(), threading.Event(), [], []

    def open_camera(serial, *, tap, width, height, fps):
        entered.set()
        gate.wait(5)
        opened.append(types.SimpleNamespace(serial=serial, profile=(width, height, fps), closes=[]))
        opened[-1].close = lambda: opened[-1].closes.append(1)
        return opened[-1]

    config = {'enabled': True, 'preview': {'cameras': ['wrist_probe', 'wrist_lower', 'wrist_upper'],
                                           'width': 32, 'height': 24, 'fps': 15}}
    preview = wrist_tap.previews(config, repo=REPO, arm='right', open_camera=open_camera, open_bus=None,
                                 log=logs.append)()
    assert entered.wait(5) and 'preview wrist_probe not opened' in logs[0]
    preview.close()                                  # wrist_lower is still opening
    gate.set()
    assert wait_for(lambda: not preview.thread.is_alive())
    assert [(c.serial, c.profile, c.closes) for c in opened] == [('lower-serial', (32, 24, 15), [1])]
    assert preview.cameras == [] and len(logs) == 1   # wrist_upper never opened after the close


# --- the readings' markers: tatbot.wrist-markers/1 beside the frames they were read in ------------------------------

GOLDEN_MARKERS = REPO / 'scripts/tests/fixtures/wrist_markers_golden.json'   # rust/visiond decodes it too
FOUND = {9: [[600.0, 400.0], [640.0, 400.0], [640.0, 440.0], [600.0, 440.0]],
         4: [[100.04, 200.0], [144.0, 202.0], [142.0, 246.0], [98.0, 244.0]],
         7: [[300.26, 210.0], [344.0, 211.0], [343.0, 255.0], [299.0, 254.0]]}


def test_a_markers_message_reproduces_the_golden_fixture_the_cockpit_reads():
    record = wrist_tap.marker_record(1_800_000_000_123_456_789, (720, 1280, 3), 's0075-hover', FOUND,
                                     {'ids': [4, 7], 'accepted': True})
    producer = {'node': 'example', 'pid': 4242, 'run_id': 'example-run', 'sha': '0123456789ab'}
    assert wrist_tap.markers_message({'name': 'realsense2'}, record, producer, 3) == GOLDEN_MARKERS.read_bytes()
    payload = json.loads(GOLDEN_MARKERS.read_text())['payload']
    assert [(m['id'], m['fit']) for m in payload['markers']] == [(4, True), (7, True), (9, False)]   # by id
    assert payload['markers'][0]['corners_px'][0] == [100.0, 200.0] and payload['size'] == [1280, 720]


def test_a_readings_markers_follow_its_frame_at_the_frames_own_stamp(rig):
    """A reading's markers go out after the frame they were read in, on <key>/markers, stamped as that frame: the
    cockpit draws them on it. Its publisher is declared at the first: a goal that reads no marker declares none."""
    bus, logs = Bus(), []
    cam = camera(bus, logs)
    colors, _ = cam.grab_both(2)
    stamp = cam.stamp_ns
    assert wait_for(lambda: len(bus.sent) == 1) and len(bus.declared) == 1
    wrist_tap.show_markers(cam, stamp, colors[-1].shape, 'hold 3', {4: np.full((4, 2), 1.25)},
                           {'ids': [4], 'accepted': False})
    cam.grab_both(1)                                 # a newer frame meanwhile: the markers keep their own
    assert wait_for(lambda: len(bus.sent) == 3)
    cam.close()
    name = mounted()['name']
    assert bus.declared[1] == (f'tatbot/vision/d405/{name}/markers',
                               {'congestion_control': 'drop', 'priority': 'data_low', 'express': True})
    (marks,) = [json.loads(b) for b in bus.sent if b.startswith(b'{')]
    assert marks['schema'] == 'tatbot.wrist-markers/1' and marks['seq'] == 1
    payload = marks['payload']
    assert payload['frame_ns'] == marks['stamp']['wall_ns'] == stamp == unpack(bus.sent[0])[0]['timestamp_ns']
    assert payload['sensor'] == f'{name}_color' and payload['size'] == [64, 48] and payload['reading'] == 'hold 3'
    assert payload['markers'] == [{'id': 4, 'corners_px': [[1.2, 1.2]] * 4, 'fit': True}] and not payload['accepted']
    assert logs == []


def test_markers_handed_over_just_before_close_still_go_out(rig):
    bus, logs = Bus(), []
    cam = camera(bus, logs)
    cam.grab_both(1)
    wrist_tap.show_markers(cam, cam.stamp_ns, (48, 64), 'locate', {2: FOUND[4]}, None)
    cam.close()
    assert wait_for(lambda: len(bus.sent) == 2) and wait_for(lambda: not cam.tap.thread.is_alive())
    assert json.loads(bus.sent[1])['payload']['markers'][0]['fit'] is False and logs == []


def test_showing_markers_is_display_only():
    """No tap, no stamp or no marker read: nothing. A tap that fails is logged, never raised into the session. A
    tracker reading's future: its row's markers, judged by the markers' fit, else by print_read's."""
    import concurrent.futures

    got, logs = [], []
    tap = types.SimpleNamespace(markers=got.append, log=logs.append)
    wrist_tap.show_markers(types.SimpleNamespace(tap=None), 1, (720, 1280), 'x', FOUND, None)
    wrist_tap.show_markers(types.SimpleNamespace(), 1, (720, 1280), 'x', FOUND, None)
    wrist_tap.show_markers(types.SimpleNamespace(tap=tap), None, (720, 1280), 'x', FOUND, None)
    wrist_tap.show_markers(types.SimpleNamespace(tap=tap), 1, (720, 1280), 'x', {}, None)
    assert got == [] and logs == []
    broken = types.SimpleNamespace(markers=lambda record: 1 / 0, log=logs.append)
    wrist_tap.show_markers(types.SimpleNamespace(tap=broken), 1, (720, 1280), 'x', FOUND, None)
    wrist_tap.show_markers(types.SimpleNamespace(tap=tap), 1, (), 'y', FOUND, None)    # no frame size
    assert got == [] and len(logs) == 2 and 'ZeroDivisionError' in logs[0] and 'markers of y' in logs[1]

    cam = types.SimpleNamespace(tap=tap)
    rows = [{'fit': 'markers', 'ids': [7], 'accepted': True, 'shown': FOUND},
            {'fit': None, 'print_sol': {'ids': [4, 9], 'accepted': False}, 'shown': FOUND},
            {'fit': None, 'error': 'no marker of the print seen', 'shown': {}}]
    for k, row in enumerate(rows):
        future = concurrent.futures.Future()
        future.add_done_callback(wrist_tap.when_read(cam, 100 + k, (720, 1280, 3), f's{k}'))
        future.set_result(row)
    failed = concurrent.futures.Future()
    failed.add_done_callback(wrist_tap.when_read(cam, 9, (720, 1280, 3), 'failed'))
    failed.set_exception(RuntimeError('worker died'))
    assert [(r['frame_ns'], r['reading'], r['accepted']) for r in got] == [(100, 's0', True), (101, 's1', False)]
    assert [[m['fit'] for m in r['markers']] for r in got] == [[False, True, False], [True, False, True]]
