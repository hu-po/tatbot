"""The ROS wrist D405's display tap: the frames tatbot_session's grab_both already took, republished for the demo
viewer as tatbot-vision-frame-set v1 (frame_set_wire.pack) on tatbot/vision/d405/<camera name>, the vision.toml
camera of the open device's serial (two D405s share the pink wrist since 2026-10-23).

grab_both hands its last colour frame and raw z16 depth to `WristTap.put`: references into a one-slot queue, the
newest replacing an unsent one, so the session pays a hand-off and nothing else. A daemon thread per open camera
opens the bus (the caller's opener), JPEG-encodes, packs and publishes with congestion control DROP, low priority
and express. close() waits for nothing: the thread still sends what was handed to it (the session's own copies,
so a snapshot camera closed straight after its grab publishes too) and then ends. It holds the camera weakly and
ends with it, so a camera dropped without close() is freed as before. Display only: any error turns the tap off,
logged once, and nothing raises into the session.

The print's markers a reading decoded in a frame the tap published (show_markers, when_read: the gauge holds', the
locate views' and the tracker's readings) follow it on tatbot/vision/d405/<camera name>/markers as one
tatbot.wrist-markers/1 envelope stamped at that frame's stamp: each marker's id and corners in the frame's pixels,
whether the reading's fit kept it, and whether the fit was accepted. The cockpit draws them on the frame in the
viewer's wrist pane (rust/visiond log_markers). Handed to the same thread, never waited for.

stack.yaml `wrist_tap.poll` (off by default) also takes a frame at 2 Hz while grab_both hands none, with
pipe.poll_for_frames() under the camera's lock, which grab_both holds while it reads: the two never use the
pipeline at once, but they compete for frames. A polled frame is one grab_both does not get (it waits for the
next) and grab_both waits while a poll copies its frames, which is why the poll is off by default.

stack.yaml `wrist_tap.preview` names display-only D405s (vision.toml roles on the same arm, the lower wrist camera)
opened beside the session's own, small and slow, for as long as it is open (previews, Preview). Nothing grabs them, so
each one's tap polls it at 2 Hz and the viewer shows it as it shows the session's camera. A reading may grab one
(the tracker's further views); its poll then holds HOLD_S, so the cockpit shows the frame the reading read, with the
markers it decoded, not a newer polled one.
"""
from __future__ import annotations

import collections
import json
import os
import sys
import threading
import time
import weakref
from pathlib import Path

import frame_set_wire

KEY = 'tatbot/vision/d405/{name}'
MARKERS_KEY = KEY + '/markers'
MARKERS_SCHEMA = 'tatbot.wrist-markers/1'
POLL_S = 0.5            # the optional poll's period while grab_both hands no frame
HOLD_S = 8.0            # no poll this long after a grab_both: the frame a reading read stays shown, with its markers
                        # (a gauge hold's print is read behind it, 2-4 s for both cameras' frames, 2026-10-23)
JPEG_QUALITY = 90
STOPPED = object()      # _next: stopped, with nothing handed over left to send


def _stderr(message: str) -> None:
    print(f'wrist tap: {message}', file=sys.stderr, flush=True)


def factory(config, *, repo, arm: str, open_bus, frames, run_id: str = '', log=None):
    """None when stack.yaml `wrist_tap` is off (no tap at all); else start(camera), which returns the running
    WristTap of that open camera, or None (logged) when it cannot start. `open_bus()` returns a zenoh session,
    `frames` are the colour and depth optical frames or their templates ('{arm}/...')."""
    config, log = config or {}, log or _stderr
    if not config.get('enabled'):
        return None

    def start(camera):
        try:
            return WristTap(camera, repo=Path(repo), arm=arm, open_bus=open_bus, frames=frames, run_id=run_id,
                            poll=bool(config.get('poll')), log=log)
        except Exception as error:   # display only: never into the session
            log(f'not started ({type(error).__name__}: {error})')
            return None
    return start


def previews(config, *, repo, arm: str, open_camera, open_bus, run_id: str = '', log=None):
    """None unless stack.yaml `wrist_tap` is on and its `preview` names cameras; else start(), which returns the
    Preview opening them: `preview.cameras` (vision.toml roles on `arm`) at its width, height and fps, each with a
    tap that polls it. `open_camera(serial, *, width, height, fps, tap)` opens one (tatbot_session WristCamera)."""
    config, log = config or {}, log or _stderr
    preview = config.get('preview') or {}
    if not config.get('enabled') or not preview.get('cameras'):
        return None
    roles, profile = tuple(preview['cameras']), {key: int(preview[key]) for key in ('width', 'height', 'fps')}

    def start():
        return Preview(Path(repo), arm, roles, lambda serial, tap: open_camera(serial, tap=tap, **profile),
                       lambda frames: factory({'enabled': True, 'poll': True}, repo=repo, arm=arm, open_bus=open_bus,
                                              frames=frames, run_id=run_id, log=log), log)
    return start


class Preview:
    """The display-only cameras of one start() (previews), opened one after another on a daemon thread, so the
    session's own camera waits for none of their ~2 s opens. close() closes those open, and one still opening as soon
    as it opens. A camera that cannot open is logged and left out; nothing raises into the session."""

    def __init__(self, repo: Path, arm: str, roles, open_camera, tap_for, log):
        self.cameras, self.closed, self.log, self._lock = [], False, log, threading.Lock()
        self.thread = threading.Thread(target=self._run, args=(repo, arm, roles, open_camera, tap_for),
                                       name='wrist-preview', daemon=True)
        self.thread.start()

    def _run(self, repo: Path, arm: str, roles, open_camera, tap_for) -> None:
        import wrist_cameras

        for role in roles:
            try:
                camera = {c['role']: c for c in wrist_cameras.mounted_for_arm(repo, arm)}[role]
                frames = tuple(wrist_cameras.optical_frames(repo, arm=arm, stream=s)[role] for s in ('color', 'depth'))
                if self.closed:
                    return
                opened = open_camera(camera['serial'], tap_for(frames))
            except Exception as error:   # display only: never into the session
                self.log(f'preview {role} not opened ({type(error).__name__}: {error})')
                continue
            with self._lock:
                if not self.closed:
                    self.cameras.append(opened)
                    continue
            self._close(opened)
            return

    def _close(self, camera) -> None:
        try:
            camera.close()
        except Exception as error:
            self.log(f'preview close ({type(error).__name__}: {error})')

    def close(self) -> None:
        with self._lock:
            self.closed, cameras, self.cameras = True, self.cameras, []
        for camera in cameras:
            self._close(camera)


def _revision(repo: Path) -> str:
    try:
        return (repo / 'REVISION').read_text().split()[0]
    except (OSError, IndexError):
        return ''


def describe(camera, *, arm: str, frames) -> dict:
    """What every message of this open camera carries: its streams' optics (without their size), serial, depth
    units and tf frames. Reads the camera's attributes only, never its pipeline."""
    import numpy as np

    color_frame, depth_frame = (frame.format(arm=arm) for frame in frames)
    k, aligned = camera.k, camera.align is not None
    color = {'fx': k[0, 0], 'fy': k[1, 1], 'ppx': k[0, 2], 'ppy': k[1, 2], 'coeffs': list(camera.dist),
             'model': camera.model}
    out = {'arm': arm, 'serial': str(camera.serial), 'fps': int(camera.fps), 'aligned': aligned, 'color': color,
           'color_frame': color_frame, 'depth': None}
    if camera.depth:
        out.update(depth=color if aligned else camera.depth_metadata['intrinsics'],
                   depth_frame=color_frame if aligned else depth_frame,
                   units=np.format_float_positional(np.float32(camera.depth_scale), trim='-'))
    return out


def _stream(desc: dict, sensor: str, stream: str, shape, fmt: str, item: dict, attributes: dict):
    from rgbd_geometry import camera_intrinsics

    height, width = int(shape[0]), int(shape[1])
    optics = json.dumps(camera_intrinsics(desc[stream], width, height), sort_keys=True, separators=(',', ':'))
    return frame_set_wire.frame_metadata(
        sensor, sequence=item['sequence'], stream=stream, width=width, height=height, fmt=fmt,
        unix_ns=item['stamp_ns'], monotonic_ns=item['mono_ns'], fps=desc['fps'],
        attributes={'physical_arm': desc['arm'], 'device_serial': desc['serial'],
                    'capture_owner_role': desc['owner_role'], 'intrinsics': optics, 'tap_source': item['source'],
                    **attributes})


def encode_jpeg(color) -> bytes:
    import cv2

    ok, jpeg = cv2.imencode('.jpg', color, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
    if not ok:
        raise ValueError('JPEG encoding failed')
    return jpeg.tobytes()


def message(desc: dict, item: dict, producer: dict) -> bytes:
    """One frame set from a handed-off item ({color, depth, stamp_ns, mono_ns, source, sequence}): `<name>_color`
    as JPEG of the session's bgr8 frame and, with depth, `<name>_depth` as its raw z16."""
    import numpy as np

    color, depth, name = item['color'], item['depth'], desc['name']
    data = encode_jpeg(color)
    meta = _stream(desc, f'{name}_color', 'color', color.shape, 'bgr8' if color.ndim == 3 else 'y8', item,
                   {'ros_frame': desc['color_frame']})
    frames = [(meta, frame_set_wire.jpeg(data), data)]
    if depth is not None and desc['depth'] is not None:
        raw = np.ascontiguousarray(depth, dtype='<u2').tobytes()
        extra = {'depth_units_m': desc['units'], 'ros_frame': desc['depth_frame'],
                 **({'aligned_to': f'{name}_color'} if desc['aligned'] else {})}
        meta = _stream(desc, f'{name}_depth', 'depth', depth.shape, 'z16', item, extra)
        frames.append((meta, frame_set_wire.depth(int(depth.shape[1]), int(depth.shape[0]), raw), raw))
    return frame_set_wire.pack(frames, sequence=item['sequence'], timestamp_ns=item['stamp_ns'], producer=producer)


def marker_record(stamp_ns: int, shape, reading: str, found: dict, fit: dict | None) -> dict:
    """A reading's markers in one published frame (MARKERS_SCHEMA's payload, less the sensor the tap adds): `found`
    id -> (4, 2) corners in the frame's pixels (pixel centres at integers, as the detector reports them), `fit` the
    reading's fit or None ("ids" it kept, "accepted"), `shape` the frame's (height, width, ...)."""
    kept = {int(i) for i in (fit or {}).get('ids') or ()}
    return {'frame_ns': int(stamp_ns), 'size': [int(shape[1]), int(shape[0])], 'reading': str(reading),
            'accepted': bool((fit or {}).get('accepted')),
            'markers': [{'id': int(i), 'corners_px': [[round(float(u), 1), round(float(v), 1)] for u, v in c],
                         'fit': int(i) in kept} for i, c in sorted(found.items())]}


def markers_message(desc: dict, record: dict, producer: dict, seq: int) -> bytes:
    """One MARKERS_SCHEMA envelope (tatbot_bus::Envelope's fields) at the frame's stamp, for desc's colour sensor."""
    return json.dumps({'schema': MARKERS_SCHEMA, 'producer': producer, 'seq': seq,
                       'stamp': {'mono_ns': 0, 'wall_ns': record['frame_ns'], 'basis': frame_set_wire.BASIS},
                       'payload': {**record, 'sensor': f"{desc['name']}_color"}}, separators=(',', ':')).encode()


def show_markers(camera, stamp_ns, shape, reading: str, found, fit) -> None:
    """Hand a reading's markers (marker_record) to `camera`'s tap. Nothing without a tap, a stamp or a marker read;
    display only, so nothing raises into the session."""
    tap = getattr(camera, 'tap', None)
    if tap is None or stamp_ns is None or not found:
        return
    try:
        tap.markers(marker_record(stamp_ns, shape, reading, found, fit))
    except Exception as error:   # display only: never into the session
        tap.log(f'markers of {reading} not shown ({type(error).__name__}: {error})')


def when_read(camera, stamp_ns: int, shape, reading: str):
    """A done-callback for a tracker reading's future (tatbot_session.track._measure's row: the markers read under
    "shown", the fit "ids" and "accepted" when read by the markers, else "print_sol"): show_markers on its result."""
    def shown(future) -> None:
        if not future.cancelled() and future.exception() is None:
            row = future.result()
            show_markers(camera, stamp_ns, shape, reading, row.get('shown'),
                         row if row.get('fit') else row.get('print_sol'))
    return shown


class WristTap:
    """One open camera's tap thread (see the module docstring). The camera gives k, dist, model, serial, fps,
    depth, depth_scale, depth_metadata and align here, on the caller's thread; the poll alone uses its pipe and
    lock later, through a weak reference."""

    def __init__(self, camera, *, repo: Path, arm: str, open_bus, frames, run_id: str, poll: bool, log):
        self.repo, self.arm, self.poll, self.log = repo, arm, poll, log
        self._desc, self._camera = describe(camera, arm=arm, frames=frames), weakref.ref(camera)
        self._open_bus, self._run_id = open_bus, run_id
        self.enabled, self.sequence, self.published, self.bus = True, 0, 0, None
        self._slot = collections.deque(maxlen=1)
        self._marks, self.marked = collections.deque(maxlen=16), 0   # readings' markers (show_markers), oldest first
        self._wake, self._stop = threading.Event(), threading.Event()
        self._held = 0   # monotonic ns until which the poll holds (HOLD_S after a grab)
        self.thread = threading.Thread(target=self._run, name='wrist-tap', daemon=True)
        self.thread.start()
        weakref.finalize(camera, self.close)

    def put(self, color, depth, stamp_ns: int) -> None:
        """grab_both's hand-off of its last frames: references only, never a copy."""
        if self.enabled:
            now = time.monotonic_ns()
            self._held = now + int(HOLD_S * 1e9)
            self._slot.append((color, depth, stamp_ns, now, 'grab'))
            self._wake.set()

    def markers(self, record: dict) -> None:
        """A reading's markers in a frame this tap published (marker_record), sent after the frames handed over."""
        if self.enabled:
            self._marks.append(record)
            self._wake.set()

    def close(self) -> None:
        """Stop the thread without waiting for it (nor for a bus still opening, 3 s with no router): it sends the
        frames handed over last, which are the session's own copies, then ends. The pipeline may stop at once:
        `_polled` checks the stop flag under the camera's lock, which WristCamera.close holds around pipe.stop()."""
        self._stop.set()
        self._wake.set()

    def _run(self) -> None:
        try:
            desc, publisher, producer = self._open()
            marks = None
            while (handed := self._next()) is not STOPPED:
                if handed is not None:
                    self.sequence += 1
                    item = dict(zip(('color', 'depth', 'stamp_ns', 'mono_ns', 'source'), handed, strict=True),
                                sequence=self.sequence)
                    publisher.put(message(desc, item, producer))
                    self.published += 1
                while self._marks:   # declared at the first: a goal that reads no marker publishes none
                    marks = marks or _declare(self.bus, MARKERS_KEY.format(name=desc['name']))
                    self.marked += 1
                    marks.put(markers_message(desc, self._marks.popleft(), producer, self.marked))
        except Exception as error:   # display only: the tap turns off, the session never sees it
            self.enabled = False
            self._slot.clear()
            self._marks.clear()
            if not self._stop.is_set():
                self.log(f'off after {self.published} frame sets ({type(error).__name__}: {error})')
        finally:
            try:
                if self.bus is not None:
                    self.bus.close()
            except Exception as error:
                self.log(f'bus close ({type(error).__name__}: {error})')

    def _open(self):
        import wrist_cameras
        from tatbot_cli import nodes

        serial = self._desc['serial']
        mounted = [c for c in wrist_cameras.mounted_for_arm(self.repo, self.arm) if c['serial'] == serial]
        if not mounted:
            raise LookupError(f'the vision registry mounts no wrist camera of serial {serial} on the {self.arm} arm')
        desc = {**self._desc, 'name': mounted[0]['name'], 'owner_role': mounted[0]['owner_role']}
        producer = {'node': nodes.this_node(nodes.load(self.repo)), 'pid': os.getpid(), 'run_id': self._run_id,
                    'sha': _revision(self.repo)}
        self.bus = self._open_bus()
        return desc, _declare(self.bus, KEY.format(name=desc['name'])), producer

    def _next(self):
        """The frames handed over last, else (with the poll) a polled frame, else None; STOPPED once stopped with
        nothing handed over left, frames or markers. The stop flag is read before the slot: put() comes before
        close()."""
        if not self._stop.is_set():
            self._wake.wait(POLL_S if self.poll else None)
            self._wake.clear()
        stopped = self._stop.is_set()
        if self._slot:
            return self._slot.popleft()
        if stopped:
            return None if self._marks else STOPPED   # the markers handed over still go out
        return self._polled() if self.poll and time.monotonic_ns() >= self._held else None

    def _polled(self):
        """A frame from the pipeline under the camera's lock (grab_both holds it while it reads); None without one."""
        import numpy as np

        camera = self._camera()
        if camera is None:
            return None
        with camera.lock:
            if self._stop.is_set():
                return None
            frames = camera.pipe.poll_for_frames()
            if not frames:
                return None
            if camera.align is not None:
                frames = camera.align.process(frames)
            color = np.asanyarray(frames.get_color_frame().get_data()).copy()
            depth = np.asanyarray(frames.get_depth_frame().get_data()).copy() if camera.depth else None
        return color, depth, time.time_ns(), time.monotonic_ns(), 'poll'


def _declare(bus, key: str):
    import zenoh

    return bus.declare_publisher(key, congestion_control=zenoh.CongestionControl.DROP,
                                 priority=zenoh.Priority.DATA_LOW, express=True)
