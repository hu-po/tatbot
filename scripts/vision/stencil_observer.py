"""The stencil observer: each turn, every installed fiducial-tile print placed by its markers in the fixed RGB-D views
(the overhead D555), and published with the support it rests on.

The fleet service (`stencild`) runs this on the overhead camera's node with the installed references and an identity
world. A tile print's markers are each its own code, so a marker read names its place on the print: each one found in
the colour has each corner lifted to the depth about its own pixel, the reads that lie as far apart as printed are
kept, and the tile's pose fitted to them (fiducials.stencil.rgbd_pose). A print that is not a tile is refused by name.
The wrist camera's readings are the drawing session's own. This process opens no device and holds no motion
authority: a print's pose is a number for the daemon to compare, never a permission.
"""

import hashlib
import json
import re
import sys
import time
from pathlib import Path

import cv2
import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(REPO/'scripts/lib'), str(REPO/'scripts')]
import stencil_reference  # noqa: E402
from fiducials import stencil as tile_stencil  # noqa: E402
from live_inputs import capture  # noqa: E402
from rgbd_geometry import rgbd_frame  # noqa: E402
from view_assets import read  # noqa: E402

SCHEMA = 'tatbot.stencil-observer-turn/1'
SUPPORT_SCHEMA = 'tatbot.target-support/1'
PAGE_CORNERS_UV = np.array([[0., 0.], [1., 0.], [1., 1.], [0., 1.]])
# The pose's sigma floors: the markers' own scatter is smaller than the depth and registration bias they cannot see.
TRANSLATION_SIGMA_M = 0.002
ROTATION_SIGMA_RAD = 0.026


def page_corners(pose, page_m, uv):
    """Root-frame points of a page pose at reference UVs (0..1 across, 0..1 down)."""
    uv = np.asarray(uv, float)
    local = np.column_stack(((uv[:, 0] - 0.5) * page_m[0], (uv[:, 1] - 0.5) * page_m[1], np.zeros(len(uv))))
    return local @ pose[:3, :3].T + pose[:3, 3]


def usable_references(paths):
    """({pattern_id: (manifest, Tile)} of the installed tile prints, {path: reason} of every other manifest): one
    that fails to load, or a print that is not a fiducial tile. One bad reference never stops the others; it is
    published lost with its reason (refused_target)."""
    usable, refused = {}, {}
    for path in paths:
        try:
            manifest, _ = stencil_reference.load(path)
            tile = tile_stencil.load(Path(path).parent)
            if tile is None:
                raise ValueError('not a fiducial-tile print: the observer places tile prints only')
            usable[manifest['pattern_id']] = (manifest, tile)
        except (OSError, ValueError, KeyError, TypeError) as error:
            refused[str(path)] = str(error) or type(error).__name__
    return usable, refused


def refused_target(path, reason, bundle_id):
    """A `lost` target for a refused reference whose manifest still names its pattern and
    reference id, so a consumer waiting for it reads why; None otherwise."""
    try:
        manifest = json.loads(read(path, 64_000))
    except (OSError, ValueError):
        return None
    pattern, reference_id = manifest.get('pattern_id'), manifest.get('reference_id')
    if not (isinstance(pattern, str) and re.fullmatch(r'stencil-[0-9a-f]{64}', pattern)
            and isinstance(reference_id, str) and re.fullmatch(r'[0-9a-f]{64}', reference_id)):
        return None
    reason = f'reference refused: {reason}'
    return {'pattern_id': pattern, 'reference_id': reference_id, 'reference_physical_instance_id': None,
            'physical_instance_id': None, 'physical_instance_identity_verified': False,
            'physical_instance_capture_ns': None, 'seed': str(manifest.get('seed')), 'target_frame': 'world',
            'source': 'lost', 'world_from_target': None, 'translation_sigma_m': None, 'rotation_sigma_rad': None,
            'capture_ns': int(time.time_ns()),
            'support': {'schema': SUPPORT_SCHEMA, 'anchor_camera': None, 'anchors': 0, 'cameras': {},
                        'anchor_points_m': [], 'anchor_pixels': [], 'reason': reason, 'motion_authority': False,
                        'physical_instance_identity_verified': False, 'bundle_id': bundle_id}}


def window(pixels, shape, margin_px=32):
    """The next turn's detection window about a placement's marker pixels: their box grown each way by its larger
    side and margin_px, inside the image."""
    lo, hi = pixels.min(axis=0), pixels.max(axis=0)
    pad = float((hi - lo).max()) + margin_px
    height, width = shape[:2]
    return (max(0, int(lo[0] - pad)), max(0, int(lo[1] - pad)),
            min(width, int(np.ceil(hi[0] + pad)) + 1), min(height, int(np.ceil(hi[1] + pad)) + 1))


def marker_anchor(views, tile, reading, page_m, windows=None):
    """(the tile placed by its markers in the fixed RGB-D view that reads the most of them, None) or (None, why).
    `views`: {name: (frame, root_from_camera)}; `reading`: rgbd_pose's detector arguments (overhead_reading).
    `windows` ({view: (x0, y0, x1, y1)}, updated): read each view first in the window about last turn's placement
    there, then in the whole frame when that places nothing. On the ros node the upsampled small-marker reading
    took 250 ms and 0.67 s of CPU a turn over the whole frame, 55 ms and 0.2 s in the window (the plain reading 45
    ms and 0.17 s; 2026-10-19)."""
    best, reasons = None, []
    for name, (frame, matrix) in views.items():
        try:
            rgbd, _ = rgbd_frame(frame)
        except (ValueError, KeyError, TypeError) as error:
            reasons.append(f'{name}: {error}')
            continue
        roi = None if windows is None else windows.pop(name, None)
        found = tile_stencil.rgbd_pose(frame['image'], rgbd['depth_m'], rgbd['rays'], tile, roi_xyxy=roi, **reading)
        if roi is not None and not found['found']:
            found = tile_stencil.rgbd_pose(frame['image'], rgbd['depth_m'], rgbd['rays'], tile, **reading)
        if windows is not None and found['found']:
            windows[name] = window(found['pixels'], frame['image'].shape)
        if not found['found']:
            reasons.append(f"{name}: {found['reason']}")
        elif best is None or found['markers'] > best[1]['markers']:
            best = (name, found, np.asarray(matrix, float), frame['timestamp_ns'])
    if best is None:
        return None, '; '.join(reasons) or 'no fixed RGB-D view'
    name, found, root_from_camera, stamp = best
    pose = root_from_camera @ found['camera_from_target']
    corners = page_corners(pose, page_m, PAGE_CORNERS_UV)
    return {'camera': name, 'anchors': found['markers'], 'loo_rms_m': found['rms_m'], 'pose': pose,
            'corners': corners, 'capture_ns': int(stamp), 'pixels': found['pixels'],
            'anchor_points': found['points_camera'] @ root_from_camera[:3, :3].T + root_from_camera[:3, 3],
            'translation_sigma_m': max(TRANSLATION_SIGMA_M, found['rms_m']), 'rotation_sigma_rad': ROTATION_SIGMA_RAD,
            'markers': {'read': found['markers'], 'seen': found['seen'], 'ids': found['ids'],
                        'rms_m': found['rms_m']}}, None


def target(pattern, manifest, anchor, reason, bundle_id):
    """One print's published target: measured from its anchor, else lost with the reason."""
    measured = anchor is not None
    support = {'schema': SUPPORT_SCHEMA, 'anchor_camera': anchor['camera'] if measured else None,
               'anchors': anchor['anchors'] if measured else 0, 'loo_p95_m': None,
               'loo_rms_m': anchor['loo_rms_m'] if measured else None,
               'model': 'fiducial_markers' if measured else None, 'method': 'fiducial_markers' if measured else None,
               'translation_sigma_m': anchor['translation_sigma_m'] if measured else None,
               'rotation_sigma_rad': anchor['rotation_sigma_rad'] if measured else None,
               'sigma_method': 'fiducial markers on the depth under them (registration bias floor)' if measured else None,
               'corners_m': anchor['corners'].tolist() if measured else None,
               'clear_corners_m': anchor['corners'].tolist() if measured else None,
               # markers.ids[k] names anchor_points_m[4k:4k+4], each corner on its ray at its own depth, and
               # anchor_pixels[4k:4k+4], where it was read in the anchor camera's image (pixel centres at integers)
               'anchor_points_m': anchor['anchor_points'].tolist() if measured else [],
               'anchor_pixels': np.round(anchor['pixels'], 1).tolist() if measured else [],
               'markers': anchor['markers'] if measured else None, 'cameras': {},
               'reason': reason, 'motion_authority': False, 'reference_physical_instance_id': None,
               'physical_instance_id': None, 'physical_instance_identity_verified': False,
               'physical_instance_capture_ns': None, 'bundle_id': bundle_id}
    return {'pattern_id': pattern, 'reference_id': manifest['reference_id'], 'reference_physical_instance_id': None,
            'physical_instance_id': None, 'physical_instance_identity_verified': False,
            'physical_instance_capture_ns': None, 'seed': str(manifest['seed']), 'target_frame': 'world',
            'source': 'measured' if measured else 'lost',
            'world_from_target': anchor['pose'].tolist() if measured else None,
            'translation_sigma_m': anchor['translation_sigma_m'] if measured else None,
            'rotation_sigma_rad': anchor['rotation_sigma_rad'] if measured else None,
            'capture_ns': anchor['capture_ns'] if measured else int(time.time_ns()), 'support': support}


class LiveSurface:
    """The observer one stencild service runs: its binding (references, calibration bundle, identity world) and,
    per request, the fixed views' captures in, every print's target out."""

    def __init__(self, binding, directory=None):
        self.directory = None if directory is None else Path(directory)
        self.references, self.refused_references = usable_references(binding.get('references') or ())
        self.calibration, self.robot = binding['calibration'], binding['robot_world']
        self.inventory_sha256 = hashlib.sha256(json.dumps(sorted(
            manifest['reference_id'] for manifest, _ in self.references.values())).encode()).hexdigest()
        self.readings = {pattern: tile_stencil.overhead_reading(tile) for pattern, (_, tile) in self.references.items()}
        # small-marker tiles are read in a window about their last placement (marker_anchor)
        self.windows = {pattern: {} for pattern, (_, tile) in self.references.items()
                        if tile_stencil.small_markers(tile)}
        self.provenance = {key: binding.get(key) for key in ('observer_epoch', 'evidence_kind')}
        self.provenance['calibration_id'] = self.calibration.get('bundle_id')
        self.capture_reader = capture
        self.sequence = 0

    def targets(self, views, errors):
        """One target per installed tile, then one per refused reference."""
        bundle_id = self.calibration.get('bundle_id')
        out = []
        for pattern, (manifest, tile) in self.references.items():
            page_m = np.asarray(manifest['page_mm'], float) / 1000.
            anchor, reason = marker_anchor(views, tile, self.readings[pattern], page_m, self.windows.get(pattern))
            if anchor is None and errors:
                reason += '; ' + '; '.join(f'{k}: {v}' for k, v in errors.items())
            out.append(target(pattern, manifest, anchor, None if anchor else f'markers: {reason}', bundle_id))
        known = {t['pattern_id'] for t in out}
        for path, reason in self.refused_references.items():
            refused = refused_target(path, reason, bundle_id)
            if refused is not None and refused['pattern_id'] not in known:
                known.add(refused['pattern_id'])
                out.append(refused)
        return out

    def request(self, request):
        if request.get('kind') == 'support':
            raise ValueError('stroke support is not an observer product; the observer publishes print poses')
        started = time.perf_counter()
        views, errors = {}, dict(request.get('errors') or {})
        for source, path in request['captures'].items():
            try:
                current = self.capture_reader(path, self.calibration, self.robot)
                if views.keys() & current.keys():
                    raise ValueError('duplicate camera in display captures')
                views.update(current)
            except (OSError, ValueError, KeyError, TypeError, ImportError, cv2.error) as error:
                errors[source] = str(error)
        decoded = time.perf_counter()
        depth_views = {name: view for name, view in views.items() if 'depth_m' in view[0]}
        targets = self.targets(depth_views, errors)
        self.sequence += 1
        return {'schema': SCHEMA, 'frame': 'root', 'motion_authority': False, 'targets': targets,
                'inventory_sha256': self.inventory_sha256, 'observation_sequence': self.sequence,
                'timings_ms': {'capture_decode': (decoded - started) * 1000,
                               'markers': (time.perf_counter() - decoded) * 1000,
                               'request_total': (time.perf_counter() - started) * 1000},
                **self.provenance}


def main():
    binding, directory = Path(sys.argv[1]), Path(sys.argv[2])
    worker = LiveSurface(json.loads(read(binding, 8*1024*1024)), directory)
    print(json.dumps({'ready': True}), flush=True)
    while line := sys.stdin.buffer.readline(16*1024+1):
        if len(line) > 16*1024:
            raise ValueError('live display request exceeds size limit')
        try:
            response = worker.request(json.loads(line))
        except (OSError, ValueError, KeyError, TypeError, ImportError, cv2.error) as error:
            response = {'schema': SCHEMA, 'frame': 'root', 'motion_authority': False,
                        'surfaces': [], 'targets': [], 'error': str(error)}
        print(json.dumps(response, allow_nan=False, separators=(',', ':')), flush=True)


if __name__ == '__main__':
    main()
