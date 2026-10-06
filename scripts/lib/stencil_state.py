"""Scan-bound stencil motion shared by Session observation and registration.

The original measured anchors define material coordinates. A pose observation
never renews the scan's geometry timestamp or changes the artwork placement.
Candidate acceptance here is geometric evidence, not native motion admission.
"""

import hashlib
import io
import json
from dataclasses import dataclass

import cv2
import numpy as np
import schemas
from bounded_npz import validate_npz_payload
from stencil_cross_camera import rigid_transform
from stencil_features import area
from surface_attachment import fit_rigid, rigid_residual

MAX_ANCHORS = 512


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                     allow_nan=False).encode()).hexdigest()


def measured_anchors(camera, stencil):
    """Use the actual depth camera's rays, including cross-camera matches."""
    report = stencil['surface']
    rows = report.get('anchors', [])
    if (not report.get('candidate_valid') or report.get('calibration_warnings')
            or not 12 <= len(rows) <= 5000):
        raise ValueError('measured stencil anchors unavailable')
    uv = np.asarray([r['reference_uv'] for r in rows], float)
    xyz = np.asarray([r['point_camera_m'] for r in rows], float)
    if (uv.shape != (len(rows), 2) or xyz.shape != (len(rows), 3)
            or not np.isfinite(uv).all() or not np.isfinite(xyz).all()
            or np.any((uv < 0) | (uv > 1)) or area(uv) < .1):
        raise ValueError('invalid measured stencil anchor geometry')
    if len(uv) > MAX_ANCHORS:
        selected = np.linspace(0, len(uv)-1, MAX_ANCHORS).astype(int)
        uv, xyz = uv[selected], xyz[selected]
    transform = rigid_transform(camera['root_from_camera'])
    return uv, xyz @ transform[:3, :3].T + transform[:3, 3]


def _duplicate_patterns(cameras):
    ambiguous = set()
    for camera in cameras.values():
        seen = set()
        for stencil in camera.get('stencils', []):
            pattern = stencil.get('pattern_id')
            if pattern in seen:
                ambiguous.add(pattern)
            seen.add(pattern)
    return ambiguous


def _conflicting_sighting(sightings, stamp, uv, xyz):
    """Contemporaneous same-pattern views must fit one physical page."""
    for earlier_stamp, earlier_uv, earlier_xyz in sightings:
        if abs(stamp-earlier_stamp) > 40_000_000:
            continue
        try:
            relative, _ = fit_motion(earlier_uv, earlier_xyz, uv, xyz, Limits())
            moved = earlier_xyz @ relative[:3, :3].T + relative[:3, 3]
            if np.max(np.linalg.norm(moved-earlier_xyz, axis=1)) > .005:
                return True
        except ValueError:
            # Separate current cameras did not establish that their
            # same-pattern sightings are one physical page.
            return True
    return False


def scan_bindings(cameras, references):
    """Bind one measured view per pattern; retain all camera evidence upstream.

    Choosing one exposure avoids treating separate views of a moving surface as
    simultaneous measurements. Two visible copies of one pattern cannot be
    bound to one physical target, even if one detection has more anchors.
    Later observations register to these originals.
    """
    candidates = {}
    sightings = {}
    ambiguous = _duplicate_patterns(cameras)
    for source, camera in cameras.items():
        for stencil in camera.get('stencils', []):
            pattern = stencil['pattern_id']
            if pattern in ambiguous:
                continue
            if stencil.get('points', 1) == 0:
                continue
            try:
                uv, xyz = measured_anchors(camera, stencil)
                stamp = int(camera['capture_timestamp_ns'])
                report = stencil['surface']
                tracking_stamp = int(report.get('tracking_timestamp_ns', stamp))
                if min(stamp, tracking_stamp) <= 0:
                    raise ValueError('missing scan exposure')
            except (ValueError, KeyError, TypeError):
                continue
            if _conflicting_sighting(sightings.get(pattern, ()), stamp, uv, xyz):
                ambiguous.add(pattern)
            sightings.setdefault(pattern, []).append((stamp, uv, xyz))
            if pattern in ambiguous:
                candidates.pop(pattern, None)
                continue
            if pattern in candidates and len(candidates[pattern]['reference_uv']) >= len(uv):
                continue
            reference = references[pattern]
            candidates[pattern] = {
                **schemas.stamp(schemas.SURFACE, 'stencil-scan-binding'),
                'pattern_id': pattern, 'reference_id': reference['reference_id'], 'seed': reference['seed'],
                'frame': 'root', 'material_frame': 'original_scan_root',
                'geometry_capture_ns': min(stamp, tracking_stamp), 'source': source,
                'camera_id': camera.get('sensor_name', source),
                'source_identity': camera['source_identity'],
                'tracking_camera': report.get('tracking_camera'),
                'tracking_camera_id': report.get('tracking_camera_id', report.get('tracking_camera')),
                'tracking_source_identity': report.get('tracking_source_identity'),
                'reference_uv': uv.tolist(), 'points_material_m': xyz.tolist(),
                'motion_authority': False,
            }
    return {pattern: binding for pattern, binding in candidates.items() if pattern not in ambiguous}


@dataclass(frozen=True)
class Limits:
    """Estimator criteria; physical Session limits remain separately bound."""
    max_age_ns: int = 3_000_000_000
    residual_m: float = .005
    min_inlier_fraction: float = .85
    min_span_m: float = .02

    def __post_init__(self):
        if not (type(self.max_age_ns) is int and 0 < self.max_age_ns <= 10_000_000_000
                and 0 < self.residual_m <= .02 and .5 <= self.min_inlier_fraction <= 1
                and 0 < self.min_span_m <= 1):
            raise ValueError('invalid stencil registration criteria')


def fit_motion(reference_uv, reference_xyz, uv, xyz, limits):
    """Match immutable artwork coordinates, then fit measured 3-D motion.

    No UV surface interpolation creates correspondences. Reacquired features
    must refer to the same template point, and each point can support one pair.
    """
    distances, indices = cv2.batchDistance(np.float32(uv), np.float32(reference_uv),
                                          cv2.CV_32F, normType=cv2.NORM_L2, K=1)
    rows = np.flatnonzero(distances[:, 0] <= 1e-6)
    _, distinct = np.unique(indices[rows, 0], return_index=True)
    rows = rows[distinct]
    original, current = reference_xyz[indices[rows, 0]], xyz[rows]
    if len(rows) < 12 or area(uv[rows]) < .1:
        raise ValueError('insufficient common scan anchors')
    if np.linalg.svd(original-original.mean(0), compute_uv=False)[1]/np.sqrt(len(rows)) < limits.min_span_m/4:
        raise ValueError('scan anchors have insufficient spatial span')
    rng = np.random.default_rng(41)
    best = np.zeros(len(rows), bool)
    for _ in range(128):
        sample = rng.choice(len(rows), 3, replace=False)
        a, b = original[sample], current[sample]
        if np.linalg.norm(np.cross(a[1]-a[0], a[2]-a[0])) < 1e-8:
            continue
        transform = fit_rigid(a, b)
        keep = rigid_residual(original, current, transform) <= limits.residual_m
        if keep.sum() > best.sum():
            best = keep
    if best.sum() < max(12, len(rows)*limits.min_inlier_fraction) or area(uv[rows][best]) < .1:
        raise ValueError('scan anchors disagree with rigid material motion; rescan required')
    transform = fit_rigid(original[best], current[best])
    residual = rigid_residual(original, current, transform)
    if np.max(residual[best]) > limits.residual_m:
        raise ValueError('refined scan registration exceeds residual bound')
    return transform, {'common_anchors': len(rows), 'inliers': int(best.sum()),
                       'residual_p95_m': float(np.percentile(residual[best], 95)),
                       'reference_indices': indices[rows[best], 0].tolist()}


class StencilSurfaceState:
    """An immutable material frame, measured geometry revisions and a fresh pose."""
    def __init__(self, binding, surface_sha256, points, colors, limits=None):
        if (not schemas.is_schema(binding, schemas.SURFACE, 'stencil-scan-binding') or binding.get('frame') != 'root'
                or binding.get('material_frame') != 'original_scan_root'
                or len(surface_sha256) != 64 or any(c not in '0123456789abcdef' for c in surface_sha256)):
            raise ValueError('invalid stencil scan binding')
        self.binding = json.loads(json.dumps(binding, allow_nan=False))
        self.uv, self.xyz = measured_anchors(
            {'root_from_camera': np.eye(4)},
            {'surface': {'candidate_valid': True, 'anchors': [
                {'reference_uv': u, 'point_camera_m': p} for u, p in zip(
                    binding['reference_uv'], binding['points_material_m'], strict=True)]}})
        self.points, self.colors = np.asarray(points, float).copy(), np.asarray(colors).copy()
        self.original_points = self.points.copy()
        if (self.points.ndim != 2 or self.points.shape[1:] != (3,) or len(self.points) > 30000
                or self.colors.shape != self.points.shape or not np.isfinite(self.points).all()
                or self.colors.dtype != np.uint8 or type(binding['geometry_capture_ns']) is not int
                or binding['geometry_capture_ns'] <= 0):
            raise ValueError('invalid retained stencil samples')
        self.limits = limits or Limits()
        self.surface_sha256, self.reference_sha256 = surface_sha256, digest(binding)
        self.geometry_capture_ns = binding['geometry_capture_ns']
        self.original_surface_sha256 = surface_sha256
        self.original_geometry_capture_ns = self.geometry_capture_ns
        self.geometry_evidence = {'method': 'original-scan-frame'}
        self.watermarks = {}
        self.identities = {binding.get('camera_id', binding['source']): digest(binding['source_identity'])}
        if binding.get('tracking_source_identity') is not None:
            self.identities[binding.get('tracking_camera_id', binding['tracking_camera'])] = digest(binding['tracking_source_identity'])
        self.invalidated = False
        self.scan_surface = None
        self.scan_from_material = np.eye(4)

    def incorporate_scan(self, other):
        """Register a new scan to the original material without resetting it.

        Rigidly consistent scans replace the sampled geometry in the ORIGINAL
        frame. Shape changes are refused pending a geometry-replanning contract.
        """
        if (other.binding['pattern_id'] != self.binding['pattern_id']
                or other.binding['reference_id'] != self.binding['reference_id']):
            raise ValueError('new scan changed the original stencil identity')
        if other.geometry_capture_ns <= self.geometry_capture_ns:
            raise ValueError('new scan must have a later geometry exposure')
        if any(self.identities[name] != identity for name, identity in other.identities.items()
               if name in self.identities):
            raise ValueError('new scan changed a bound camera identity')
        transform, evidence = fit_motion(self.uv, self.xyz, other.uv, other.xyz, self.limits)
        points = (other.points-transform[:3, 3]) @ transform[:3, :3]
        if len(self.original_points) and len(points):
            from scipy.spatial import cKDTree
            distance, _ = cKDTree(self.original_points).query(points, k=1)
            if np.quantile(distance, self.limits.min_inlier_fraction) > self.limits.residual_m:
                raise ValueError('new scan geometry differs from the original rigid patch')
        self.points, self.colors = points, other.colors.copy()
        self.surface_sha256 = other.surface_sha256
        self.geometry_capture_ns = other.geometry_capture_ns
        self.scan_surface = other.scan_surface
        self.scan_from_material = transform
        self.geometry_evidence = {'method': 'measured-original-anchor-registration', **evidence}
        self.identities.update(other.identities)

    def geometry(self):
        """Exact measured geometry revision expressed in the immutable material frame.

        This is evidence for the Session's geometry handoff, not registration
        of motion. A pose update cannot renew or change any of these fields.
        """
        return {**schemas.stamp(schemas.SURFACE, 'stencil-geometry'),
                'pattern_id': self.binding['pattern_id'], 'reference_sha256': self.reference_sha256,
                'original_surface_sha256': self.original_surface_sha256,
                'original_geometry_capture_ns': self.original_geometry_capture_ns,
                'surface_sha256': self.surface_sha256, 'geometry_capture_ns': self.geometry_capture_ns,
                'material_frame': 'original_scan_root', 'geometry_frame': 'root_at_scan',
                'scan_from_material': self.scan_from_material.tolist(),
                'registration_evidence': json.loads(json.dumps(self.geometry_evidence)),
                'registration_limits': {'residual_m': self.limits.residual_m,
                                        'min_inlier_fraction': self.limits.min_inlier_fraction,
                                        'min_span_m': self.limits.min_span_m},
                'deformation_qualified': False, 'motion_authority': False}

    def observe(self, cameras, now_ns):
        geometry = self.geometry()
        result = {**schemas.stamp(schemas.SURFACE, 'stencil-surface-state'), 'pattern_id': self.binding['pattern_id'],
                  'reference_id': self.binding['reference_id'], 'reference_sha256': self.reference_sha256,
                  'surface_sha256': self.surface_sha256, 'geometry_capture_ns': self.geometry_capture_ns,
                  'material_frame': 'original_scan_root', 'status': 'lost', 'root_from_material': None,
                  'pose_capture_ns': None, 'motion_authority': False, 'reasons': {},
                  'depth_support_scope': 'measured anchors; drawing footprint needs separate validation',
                  'dense_material_identity_verified': False}
        result.update(geometry=geometry, geometry_revision_sha256=digest(geometry),
                      geometry_revision_json=json.dumps(geometry, sort_keys=True, separators=(',', ':'), allow_nan=False))
        if self.invalidated:
            return dict(result, status='invalidated', reasons={'binding': 'camera identity changed; new binding required'})
        if any(sum(stencil.get('pattern_id') == self.binding['pattern_id']
                   for stencil in camera.get('stencils', [])) > 1 for camera in cameras.values()):
            self.invalidated = True
            return dict(result, status='invalidated', reasons={
                'identity': 'ambiguous duplicate print of the same pattern; new physical target binding required'})
        accepted = self._candidates(cameras, now_ns, result['reasons'])
        if self.invalidated:
            return dict(result, status='invalidated')
        if not accepted:
            return result
        # Other contemporaneous views must agree, rather than choosing the
        # most favorable camera when measured material placements conflict.
        source, stamp, transform, evidence = max(accepted, key=lambda row: row[1])
        for _, other_stamp, other, _ in accepted:
            if stamp-other_stamp > 40_000_000:
                continue
            disagreement = rigid_residual(self.xyz, self.xyz @ other[:3, :3].T+other[:3, 3], transform)
            if np.max(disagreement) > self.limits.residual_m:
                # Distinct cameras can each see one copy of the same print.
                # Choosing whichever view happens to survive the next turn
                # would silently transfer this job to another physical page.
                self.invalidated = True
                return dict(result, status='invalidated', reasons={
                    'identity': 'conflicting same-pattern positions across current views; '
                                'new physical target binding required'})
        return dict(result, status='tracked', source=source, pose_capture_ns=stamp,
                    root_from_material=transform.tolist(), evidence=evidence)

    def _candidates(self, cameras, now_ns, reasons):
        accepted = []
        for source, camera in cameras.items():
            for stencil in camera.get('stencils', []):
                if stencil['pattern_id'] != self.binding['pattern_id']:
                    continue
                try:
                    accepted.append(self._candidate(source, camera, stencil, now_ns))
                except (ValueError, KeyError, TypeError) as error:
                    reasons[source] = str(error)
        return accepted

    def _candidate(self, source, camera, stencil, now_ns):
        uv, xyz = measured_anchors(camera, stencil)
        report = stencil['surface']
        stamp = min(camera['capture_timestamp_ns'], report.get('tracking_timestamp_ns', camera['capture_timestamp_ns']))
        if not 0 <= now_ns-stamp <= self.limits.max_age_ns or stamp < self.geometry_capture_ns:
            raise ValueError('stale, future or pre-scan observation')
        # Scan capture labels (for example surface_overhead) and live capture
        # labels can differ for the same physical sensor. Epoch and exposure
        # guards follow the sensor, not the caller's display label.
        camera_id = camera.get('sensor_name', source)
        if stamp <= self.watermarks.get(camera_id, 0):
            raise ValueError('repeated or regressed exposure')
        self.watermarks[camera_id] = stamp
        identities = {camera_id: camera['source_identity']}
        if report.get('tracking_source_identity') is not None:
            identities[report.get('tracking_camera_id', report['tracking_camera'])] = report['tracking_source_identity']
        for name, value in identities.items():
            identity = digest(value)
            if self.identities.setdefault(name, identity) != identity:
                self.invalidated = True
                raise ValueError('camera identity changed; new binding required')
        transform, evidence = fit_motion(self.uv, self.xyz, uv, xyz, self.limits)
        return source, stamp, transform, evidence

    def displayed_points(self, observation):
        if observation['status'] != 'tracked':
            return np.empty((0, 3)), np.empty((0, 3), np.uint8)
        transform = rigid_transform(observation['root_from_material'])
        return self.points @ transform[:3, :3].T+transform[:3, 3], self.colors.copy()

    def displayed_outline(self, observation, reference=None):
        """The original material's page boundary, clear centre and measured
        anchors under the same tracked pose as its displayed points."""
        from stencil_outline import stencil_outline
        if observation['status'] != 'tracked':
            return {}
        try:
            return stencil_outline(self.uv, self.xyz, rigid_transform(observation['root_from_material']), reference)
        except (ValueError, KeyError, TypeError, cv2.error) as error:
            return {'outline_reason': (str(error) or type(error).__name__)[:200]}


def load_scan(payload, expected_sha256, limits=None):
    """The same digest-bound scan artifact used by the Session surface checker."""
    if hashlib.sha256(payload).hexdigest() != expected_sha256:
        raise ValueError('scan surface digest mismatch')
    validate_npz_payload(payload)
    with np.load(io.BytesIO(payload), allow_pickle=False) as arrays:
        if 'stencil_scan_bindings' not in arrays:
            return {}
        bindings = json.loads(str(arrays['stencil_scan_bindings']))
        if not isinstance(bindings, dict) or not 0 <= len(bindings) <= 8:
            raise ValueError('invalid stencil scan binding count')
        field = None
        if 'schema' in arrays and str(arrays['schema']) == 'tatbot.surface/1':
            from surface_model import HeightFieldSurface
            field = HeightFieldSurface.from_npz(io.BytesIO(payload))
        result = {}
        for pattern, binding in bindings.items():
            if pattern != binding['pattern_id']:
                raise ValueError('scan pattern identity mismatch')
            keep = (arrays['stencil_rgb_patterns'] == pattern) & (arrays['stencil_rgb_sources'] == binding['source'])
            result[pattern] = StencilSurfaceState(binding, expected_sha256,
                arrays['stencil_rgb_points'][keep], arrays['stencil_rgb_colors'][keep], limits)
            result[pattern].scan_surface = field
        return result
