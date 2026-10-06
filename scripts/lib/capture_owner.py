"""Median-capture input from the existing D405 owner; contains no camera SDK.

Each batch uses distinct aligned frames captured after the request, carries
active intrinsics and raw depth units, and fails if identity or profile changes.
"""
from __future__ import annotations

import json
import time
from collections import deque
from pathlib import Path

import cv2
import numpy as np
from rgbd_geometry import paired_alignment
from vision.visiond_wire import UnixWireReader
from wrist_cameras import capture_arm


def aligned_color(frame_set, depth, serial):
    """Return a matching color frame, or None for a transient straddled pair."""
    color = frame_set['frames'].get(depth.get('attributes', {}).get('aligned_to'))
    if color is None or 'image' not in color:
        raise ValueError('owner depth has no aligned color frame')
    metadata = color.get('image_metadata', color['metadata'])
    if metadata.get('attributes', {}).get('device_serial') != serial:
        raise ValueError('owner RGBD identity mismatch')
    return color if metadata['sequence'] == depth['sequence'] else None


class OwnerCamera:
    def __init__(self, owner, role, serial):
        self.owner, self.role, self.serial = owner, role, serial
        self.depths = deque()
        self.color = None
        self.color_metadata = None
        self.profile = None
        self.records = []
        self.last_frame = -1
        self.units_m = None
        self.intrinsics = None

    def read_depth(self):
        if not self.depths:
            raise RuntimeError('owner capture batch is exhausted')
        return self.depths.popleft()

    def read_color(self):
        return self.color

    def evidence_arrays(self):
        return {f'owner_frames_{self.role}': np.array(json.dumps(self.records)),
                f'owner_batch_{self.role}': np.array(json.dumps(self.owner.batch)),
                f'owner_profile_{self.role}': np.array(json.dumps(self.profile)),
                f'camera_serial_{self.role}': np.array(self.serial),
                f'owner_color_metadata_{self.role}': np.array(json.dumps(self.color_metadata))}

    def close(self):
        self.owner.close()


def _depth_by_serial(frame_set):
    by_serial = {}
    for entry in frame_set['frames'].values():
        if 'depth' in entry:
            meta = entry.get('depth_metadata', entry['metadata'])
            serial = meta.get('attributes', {}).get('device_serial')
            if serial in by_serial:
                raise ValueError('duplicate D405 serial in owner set')
            by_serial[serial] = entry
    return by_serial


def _exposed_after(dm, attr, requested_ns):
    """False when the source clock says the exposure began before the request."""
    source_ns = dm['timestamps'].get('normalized_unix_ns')
    if source_ns is None:
        return True
    exposure_ns = float(attr.get('actual_exposure_us', 0))*1000
    if (type(source_ns) is not int or not np.isfinite(exposure_ns)
            or not 0 < exposure_ns <= 100_000_000):
        raise ValueError("invalid owner exposure timing")
    return source_ns - exposure_ns >= requested_ns


def _frame_profile(attr, depth, color):
    """(profile, geometry) from validated owner intrinsics, units and dimensions."""
    intr = json.loads(attr['intrinsics'])
    units = float(attr['depth_units_m'])
    geometry = np.array([intr['fx'], intr['fy'], intr['ppx'], intr['ppy'], intr['width'], intr['height']], dtype=float)
    coefficients = np.array(intr['distortion_coefficients'], dtype=float)
    if (intr.get('schema') != 'tatbot.camera-intrinsics/1'
            or not np.isfinite(geometry).all() or not np.isfinite(coefficients).all()
            or np.any(geometry[:2] <= 0) or not np.isfinite(units) or units <= 0
            or depth['depth'].shape != (intr['height'], intr['width'])
            or color['image'].shape != (intr['height'], intr['width'], 3)):
        raise ValueError('invalid owner intrinsics, units or dimensions')
    cm = color.get('image_metadata', color['metadata'])
    alignment = paired_alignment(cm.get('attributes', {}), attr, intr)
    return {'intrinsics': intr, 'units_m': units, 'capture_epoch': attr.get('capture_epoch'),
            'alignment_calibration': alignment}, geometry


class OwnerCameras(list):
    """One arm's manifested wrist cameras (`wrist_cameras.capture_roles`) read
    from the owner's stream; a registry mixing arms or unknown roles refuses."""

    def __init__(self, registry, reader=None, *, repo=None):
        super().__init__()
        try:
            self.arm = capture_arm(registry or (), **({'repo': repo} if repo is not None else {}))
        except ValueError as error:
            raise ValueError('owner capture requires distinct manifested wrist cameras of one arm') from error
        if (len(set(registry.values())) != len(registry)
                or any(not isinstance(serial, str) or not serial for serial in registry.values())):
            raise ValueError('owner capture requires distinct manifested wrist cameras of one arm')
        self.reader = reader or UnixWireReader(Path('/tmp/tatbot-d405-frames.sock'))
        self.closed = False
        self.extend(OwnerCamera(self, role, serial) for role, serial in registry.items())
        try:
            self.begin_capture(1)  # Establish active profiles for the capture report.
        except BaseException:
            self.close()
            raise

    def close(self):
        if not self.closed:
            self.closed = True
            self.reader.close()

    def _fresh_frame(self, cam, frame_set, depth, requested_ns):
        """This camera's RGBD pair from the set if it was exposed after the
        request and is newer than the last one counted; None means the set is
        not usable for this batch yet."""
        if depth is None:
            return None
        dm = depth.get('depth_metadata', depth['metadata'])
        attr = dm.get('attributes', {})
        color = aligned_color(frame_set, dm, cam.serial)
        # Enforce the atomic owner contract here too, including during
        # mixed-version deployments. Never count a straddled RGBD pair.
        if color is None:
            self.batch['unmatched_rgbd_sets'] += 1
            return None
        host_ns = int(dm['timestamps']['host_unix_ns'])
        if not _exposed_after(dm, attr, requested_ns):
            return None
        frame_number = int(attr['frame_number'])
        if host_ns < requested_ns or frame_number <= cam.last_frame:
            return None
        age = time.time_ns() - host_ns
        if age < 0 or age > 250_000_000:
            raise ValueError('owner RGBD frame is stale or from the future')
        profile, geometry = _frame_profile(attr, depth, color)
        if cam.profile is not None and cam.profile != profile:
            raise ValueError('owner camera profile changed during capture')
        return (cam, depth, color, dm, frame_number, profile, geometry)

    def begin_capture(self, count):
        if not 1 <= count <= 255:
            raise ValueError('owner median batch requires 1..255 distinct frames')
        requested_ns = time.time_ns()
        deadline = time.monotonic() + 5.0
        self.batch = {'requested_wall_ns': requested_ns, 'unmatched_rgbd_sets': 0,
                      'frames_per_camera': count}
        for cam in self:
            cam.depths.clear()
            cam.records.clear()
        while any(len(cam.depths) < count for cam in self):
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError('D405 owner did not supply a fresh complete median batch; '
                                   f'unmatched RGBD sets: {self.batch["unmatched_rgbd_sets"]}')
            self.reader.socket.settimeout(remaining)
            frame_set = self.reader.receive()
            by_serial = _depth_by_serial(frame_set)
            candidates = []
            for cam in self:
                candidate = self._fresh_frame(cam, frame_set, by_serial.get(cam.serial), requested_ns)
                if candidate is None:
                    break
                candidates.append(candidate)
            if len(candidates) != len(self):
                continue
            for cam, depth, color, dm, number, profile, geometry in candidates:
                cam.profile, cam.intrinsics = profile, geometry
                cam.units_m = profile['units_m']
                cam.last_frame = number
                cam.depths.append(depth['depth'])
                cam.color = cv2.cvtColor(color['image'], cv2.COLOR_BGR2RGB)
                cam.color_metadata = color.get('image_metadata', color['metadata'])
                cam.records.append({'set_sequence': frame_set['sequence'], 'metadata': dm})
        self.batch['elapsed_s'] = 5.0 - (deadline - time.monotonic())
