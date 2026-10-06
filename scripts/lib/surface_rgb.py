"""Retained scan appearance in URDF root coordinates, separate from motion geometry.

Only the newest exposure of each contributing camera is displayed. A lost
stencil clears its old points. This is a multi-camera overlay, not TSDF fusion.
"""

import copy
import json
import time
from pathlib import Path

import cv2
import numpy as np
import schemas
from stencil_coded_live import project_areas
from stencil_cross_camera import cross_camera_points, project_into_view
from stencil_interior import interior_points
from stencil_scene import SceneBank, StencilScene
from stencil_state import scan_bindings
from stencil_surface import rgbd_frame
from stencils import materialize

MAX_POINTS = 30000


def wrist_frame(capture, role):
    records = json.loads(str(capture[f'owner_frames_{role}']))
    color = json.loads(str(capture[f'owner_color_metadata_{role}']))
    depth = records[-1]['metadata']
    raw = capture[f'raw_depth_{role}']
    if len(raw) != len(records) or depth['sequence'] != color['sequence']:
        raise ValueError('retained RGB exposure differs from last raw depth')
    return {'image': np.ascontiguousarray(capture[f'color_{role}'][..., ::-1]),
            'depth_m': raw[-1].astype(float)*float(capture[f'units_m_{role}']),
            'color_metadata': color, 'depth_metadata': depth,
            'timestamp_ns': color['timestamps']['normalized_unix_ns']}


class ScanAppearance:
    """Every view's stencil tracking over one shared reference bank.

    `coded_options` bound a coded print's decode (`stencil_coded_live.CodedBank`); a turn
    starts with `begin_turn`. `search_areas` are root-frame polygons (metres) where a coded
    print may lie; each view searches their projection, or its whole image when None."""

    def __init__(self, bundle, directory, *, tracking_settings=None, coded_options=None, search_areas=None,
                 min_px_per_mm=1.2):
        cv2.setNumThreads(1)
        self.references = materialize(bundle, directory / 'stencil-references')
        self.scenes, self.batches, self.reports = {}, {}, {}
        self.pending, self.bank = {}, None
        self.views, self.direct = {}, {}
        self.tracking_settings = tracking_settings
        self.coded_options = coded_options or {}
        self.search_areas, self.min_px_per_mm = search_areas, min_px_per_mm

    def begin_turn(self):
        if self.bank is not None:
            self.bank.begin_turn()

    def coded_hint(self, source, timestamp_ns):
        """Whether this view lost a coded print it can look for where it was (StencilScene)."""
        scene = self.scenes.get(source)
        return scene is not None and scene.coded_hint(timestamp_ns)

    def search_regions(self, frame, root_from_camera):
        """Pixel boxes of the search areas in one view; None searches the whole image."""
        if self.search_areas is None:
            return None
        intrinsics = frame.get('camera_model', frame['color_metadata'].get('attributes', {}).get('intrinsics'))
        view = dict(frame, camera_model=json.loads(intrinsics) if isinstance(intrinsics, str) else intrinsics)
        try:
            return project_areas(self.search_areas, np.linalg.inv(root_from_camera),
                                 lambda points: project_into_view(points, view), frame['image'].shape,
                                 self.min_px_per_mm)
        except (ValueError, KeyError, TypeError, cv2.error):
            return []

    def unavailable(self, source, reason):
        self.pending.pop(source, None)
        self.views.pop(source, None)
        self.direct.pop(source, None)
        self.batches[source] = []
        self.reports[source] = {'status': 'unavailable', 'reason': str(reason)}

    def observe(self, frame, source, root_from_camera, keep_points=None, *, localize=True,
                periodic_search=True, search=True):
        started = time.perf_counter()
        self.unavailable(source, 'processing')
        has_depth = 'depth_m' in frame
        if has_depth:
            _, warnings = rgbd_frame(frame)
            if warnings:
                raise ValueError('; '.join(warnings))
        else:
            from stencil_pose import camera_model
            camera_model(frame['camera_model'], frame['image'].shape)
        matrix = np.asarray(root_from_camera, float)
        if (matrix.shape != (4, 4) or not np.isfinite(matrix).all()
                or not np.allclose(matrix[3], [0, 0, 0, 1])
                or not np.allclose(matrix[:3, :3].T @ matrix[:3, :3], np.eye(3), atol=1e-5)
                or not np.isclose(np.linalg.det(matrix[:3, :3]), 1)):
            raise ValueError('appearance requires a rigid root-from-camera transform')
        if self.bank is None:
            self.bank = SceneBank.from_paths(self.references, self.tracking_settings, **self.coded_options)
        if source not in self.scenes:
            self.scenes[source] = StencilScene(self.references, 'session',
                                               settings=self.tracking_settings, bank=self.bank,
                                               stable_scene_skip=self.tracking_settings is not None)
        scene = self.scenes[source]
        cm = frame['color_metadata']
        identity = ({key: cm['attributes'][key] for key in ('capture_epoch', 'device_serial', 'intrinsics', 'alignment_calibration')}
                    if has_depth else frame['source_identity'])
        if localize:
            observation = scene.observe(frame['image'], frame['timestamp_ns'],
                                        source_id=source+json.dumps(identity, sort_keys=True),
                                        periodic_search=periodic_search, search=search,
                                        regions=self.search_regions(frame, matrix))
        else:
            observation = {'stencils': [{'pattern_id': pattern, 'status': 'not_searched', 'image_tracking_valid': False}
                                       for pattern in self.bank.references]}
        localized = time.perf_counter()
        self.views[source] = (frame, observation, matrix, keep_points)
        reports = []
        for row in observation['stencils']:
            reference = scene.bank.references[row['pattern_id']]
            if has_depth:
                xyz, colors, report = interior_points(frame, row, reference)
            else:
                xyz, colors = np.empty((0, 3)), np.empty((0, 3), np.uint8)
                report = {'reason': 'tracking_only_view'}
            report.update(physical_instance_id=row.get('physical_instance_id'),
                          physical_instance_identity_verified=row.get('physical_instance_identity_verified', False))
            points = xyz @ matrix[:3, :3].T + matrix[:3, 3]
            if keep_points is not None:
                keep = keep_points(points)
                points, colors = points[keep], colors[keep]
            self.batches[source].append((points, colors, row['pattern_id']))
            reports.append({'pattern_id': row['pattern_id'], 'seed': reference['seed'],
                            'tracking': row['status'], 'tracking_reason': row.get('reason'),
                            'physical_instance_identity_verified': row.get(
                                'physical_instance_identity_verified', False),
                            'points': len(points), 'surface': report})
        self.reports[source] = {'capture_timestamp_ns': frame['timestamp_ns'], 'stencils': reports,
                                'sensor_name': cm['sensor_name'],
                                'root_from_camera': matrix.tolist(), 'source_identity': identity,
                                'owner_sequence': cm['sequence'], 'color_source_ns': cm['timestamps']['source_ns'],
                                'depth_source_ns': frame['depth_metadata']['timestamps']['source_ns'] if has_depth else None,
                                'tracking_only': not has_depth, 'local_image_search': localize,
                                'periodic_search_allowed': periodic_search, 'reference_search_allowed': search,
                                'timing_ms': {'localize': (localized-started)*1000,
                                              'surface': (time.perf_counter()-localized)*1000}}
        self.direct[source] = (list(self.batches[source]), copy.deepcopy(self.reports[source]))

    def cross_views(self, tracking_cameras=None, depth_roi=None):
        """Use another contemporaneous scan view when local tracking has no depth support.

        `depth_roi` is a projection region in one depth camera's pixels, or a
        map of them keyed by depth source when more than one depth camera
        (the overhead, a wrist) is in the turn."""
        # Recompute cross-camera support every time: a lost tracking view must
        # clear the points it previously selected in a different depth camera.
        for source in self.views:
            batches, report = self.direct[source]
            self.batches[source], self.reports[source] = list(batches), copy.deepcopy(report)
        for source, (frame, _, root_from_depth, keep_points) in self.views.items():
            if 'depth_m' not in frame:
                continue
            region = depth_roi.get(source) if isinstance(depth_roi, dict) else depth_roi
            for target, (view, observation, root_from_view, _) in self.views.items():
                if source != target and (tracking_cameras is None or target in tracking_cameras):
                    self._cross_view(source, frame, target, view, observation,
                                     root_from_depth, root_from_view, keep_points, region)

    def _cross_view(self, source, frame, target, view, observation, root_from_depth, root_from_view,
                    keep_points, depth_roi):
        intrinsics = view.get('camera_model', view['color_metadata'].get('attributes', {}).get('intrinsics'))
        view = dict(view, camera_model=json.loads(intrinsics) if isinstance(intrinsics, str) else intrinsics)
        transform = np.linalg.inv(root_from_view) @ root_from_depth
        projection_cache = {}
        for row in observation['stencils']:
            pattern = row['pattern_id']
            index = next(i for i, batch in enumerate(self.batches[source]) if batch[2] == pattern)
            # Every tracking view's fit is recorded so an observer can name
            # each camera's disagreement; the first measured support fills
            # the displayed batch and a later one never replaces it.
            filled = len(self.batches[source][index][0]) > 0
            try:
                xyz, colors, report = cross_camera_points(frame, view, row, self.bank.references[pattern],
                                                          transform, max_skew_ns=40_000_000,
                                                          projection_cache=projection_cache, depth_roi=depth_roi)
                points = xyz @ root_from_depth[:3, :3].T + root_from_depth[:3, 3]
                if keep_points is not None:
                    keep = keep_points(points)
                    points, colors = points[keep], colors[keep]
            except (ValueError, KeyError, TypeError, ImportError, cv2.error) as error:
                report, points = {'reason': str(error)}, []
            report.update(tracking_camera=target, tracking_source_identity=self.reports[target]['source_identity'],
                          tracking_camera_id=self.reports[target]['sensor_name'],
                          physical_instance_id=row.get('physical_instance_id'),
                          physical_instance_identity_verified=row.get('physical_instance_identity_verified', False),
                          view_from_depth=transform.tolist())
            existing = self.reports[source]['stencils'][index]
            existing.setdefault('cross_camera_attempts', []).append(report)
            if len(points) and not filled:
                self.batches[source][index] = (points, colors, pattern)
                existing.update(points=len(points), surface=report, tracking=row['status'])

    def defer(self, frame, source, transform, keep_points):
        # Scan viewpoints need not arrive in filename/time order. Keep a single
        # exposure per camera and run feature extraction only once per camera.
        previous = self.pending.get(source)
        if previous is None or previous[0]['timestamp_ns'] < frame['timestamp_ns']:
            self.pending[source] = (frame, source, transform, keep_points)

    def wrist(self, capture, role, chain, keep_points):
        import arm_kinematics as dk
        import capture_geometry
        try:
            arm, links = capture_geometry.capture_views(capture)
            link = links[role].replace('_depth_optical_frame', '_color_optical_frame')
            values = dk.ArmModel(arm, chain=chain).joint_map(capture['joints'], float(capture['carriage_m']))
            self.defer(wrist_frame(capture, role), role, chain.link_pose(link, values), keep_points)
        except (ValueError, KeyError, TypeError, ImportError, cv2.error, capture_geometry.StageError) as error:
            self.unavailable(role, error)

    def overhead(self, entries, read_payload, root_from_camera, keep_points):
        from visiond_wire import decode_video
        source = 'surface_overhead'
        try:
            color, depth = entries['overhead_depth_color'], entries['overhead_depth_depth']
            cm, dm = color['metadata'], depth['metadata']
            raw = np.frombuffer(read_payload(depth), '<u2').reshape(depth['height'], depth['width'])
            image = decode_video(read_payload(color), cm['profile'])
            self.defer({'image': image, 'depth_m': raw.astype(float)*float(dm['attributes']['depth_units_m']),
                          'color_metadata': cm, 'depth_metadata': dm,
                          'timestamp_ns': cm['timestamps']['normalized_unix_ns']}, source, root_from_camera, keep_points)
        except (ValueError, KeyError, TypeError, ImportError, cv2.error) as error:
            self.unavailable(source, error)

    def resolve(self):
        pending, self.pending = self.pending, {}
        for frame, source, transform, keep in pending.values():
            try:
                self.observe(frame, source, transform, keep)
            except (ValueError, KeyError, TypeError, ImportError, cv2.error) as error:
                self.unavailable(source, error)
        self.cross_views()

    def write(self, path):
        self.resolve()
        batches = [(source, xyz, rgb, pattern) for source, rows in self.batches.items()
                   for xyz, rgb, pattern in rows if len(xyz)]
        budget = max(1, MAX_POINTS//max(1, len(batches)))
        xyzs, rgbs, labels, sources = [], [], [], []
        for source, xyz, rgb, pattern in batches:
            stride = max(1, int(np.ceil(len(xyz)/budget)))
            xyzs.append(xyz[::stride])
            rgbs.append(rgb[::stride])
            labels.extend([pattern]*len(xyz[::stride]))
            sources.extend([source]*len(xyz[::stride]))
        report = {**schemas.stamp(schemas.SURFACE, 'rgb'), 'frame': 'root', 'color': 'RGB8', 'motion_authority': False,
                  'dense_material_identity_verified': False, 'cameras': self.reports,
                  'points': len(labels), 'temporal_policy': 'latest exposure per camera in this scan',
                  'fusion': 'overlay of measured points; no geometry averaging or uncertainty reduction'}
        with np.load(path, allow_pickle=False) as old:
            arrays = {key: old[key] for key in old.files}
        arrays.update(stencil_rgb_meta=np.str_(json.dumps(report, allow_nan=False)),
                      stencil_scan_bindings=np.str_(json.dumps(scan_bindings(
                          self.reports, self.bank.references if self.bank else {}), allow_nan=False)),
                      stencil_rgb_points=np.concatenate(xyzs) if xyzs else np.empty((0, 3)),
                      stencil_rgb_colors=np.concatenate(rgbs) if rgbs else np.empty((0, 3), np.uint8),
                      stencil_rgb_patterns=np.asarray(labels, dtype='U72'),
                      stencil_rgb_sources=np.asarray(sources, dtype='U64'))
        temporary = Path(path).with_suffix('.rgb.tmp')
        with temporary.open('wb') as stream:
            np.savez(stream, **arrays)
        temporary.replace(path)
        return report
