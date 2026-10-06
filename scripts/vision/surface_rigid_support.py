"""Measured interior support under an explicit, bounded rigid-patch assumption.

Appearance is established only at the original sparse anchors. Interior drawing
points are inferred from their enclosing, connected rigid patch, never described
as independently appearance-tracked or deformation-qualified.
"""
import hashlib
import json
import sys
from dataclasses import asdict, dataclass, replace
from pathlib import Path

import cv2
import numpy as np
from surface_material_support import _measurement_mask, _validate_rgbd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "lib"))
import schemas  # noqa: E402


def _digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def _sha(value):
    return isinstance(value, str) and len(value) == 64 and all(c in '0123456789abcdef' for c in value)


@dataclass(frozen=True)
class RigidSupportProfile:
    mode: str = 'rigid-patch'
    assumption: str = 'connected-reference-patch-remains-rigid'
    min_anchors: int = 3
    min_spread_residual_ratio: float = 2.0
    anchor_request_sha256: str | None = None
    anchor_inventory_sha256: str | None = None

    def __post_init__(self):
        if (self.mode != 'rigid-patch' or self.assumption != 'connected-reference-patch-remains-rigid'
                or type(self.min_anchors) is not int or not 3 <= self.min_anchors <= 128
                or not np.isfinite(self.min_spread_residual_ratio) or self.min_spread_residual_ratio < 2
                or any(value is not None and not _sha(value) for value in
                       (self.anchor_request_sha256, self.anchor_inventory_sha256))):
            raise ValueError('invalid rigid-patch assumption or profile')


def _points(value, maximum):
    points = np.array(value, dtype=float, copy=True)
    if (points.ndim != 2 or points.shape[1:] != (3,) or not 1 <= len(points) <= maximum
            or not np.isfinite(points).all()):
        raise ValueError('invalid bounded original rigid-patch points')
    points.setflags(write=False)
    return points


class RigidPatchSupport:
    """Pure evidence: caller binds anchor_report to the exact current capture."""

    def __init__(self, reference_rgbd, reference_points, anchor_points, reference_digest,
                 anchor_request_sha256, *, residual_m=0.0015, profile=None):
        from surface_attachment import measured_xyz, project_measured_rays
        _validate_rgbd(reference_rgbd)
        if not _sha(reference_digest) or not _sha(anchor_request_sha256) or not 0 < residual_m <= .02:
            raise ValueError('invalid rigid-patch reference identity or residual')
        self._points = _points(reference_points, 4096)
        self._anchors = _points(anchor_points, 128)
        self._reference_digest = reference_digest
        self._anchor_request = anchor_request_sha256
        self._residual_m = residual_m
        base = profile or RigidSupportProfile()
        inventory = _digest(self._anchors.tolist())
        if (not isinstance(base, RigidSupportProfile)
                or base.anchor_request_sha256 not in (None, anchor_request_sha256)
                or base.anchor_inventory_sha256 not in (None, inventory)):
            raise ValueError('rigid-patch anchor inventory binding changed')
        self._profile = replace(base, anchor_request_sha256=anchor_request_sha256,
                                anchor_inventory_sha256=inventory)
        self._shape = reference_rgbd['gray'].shape
        self._depth_range = tuple(reference_rgbd['depth_range'])
        self._rays_digest = hashlib.sha256(reference_rgbd['rays'].tobytes()).hexdigest()
        combined = np.concatenate([self._points, self._anchors])
        pixels, visible = project_measured_rays(combined, reference_rgbd['rays'])
        measured, valid = measured_xyz(reference_rgbd, pixels, 2*residual_m)
        if not (visible & valid & (np.linalg.norm(measured-combined, axis=1) <= 2*residual_m)).all():
            raise ValueError('original rigid-patch measured support missing or changed')
        self._pixels, self._anchor_pixels = pixels[:len(self._points)], pixels[len(self._points):]
        self._component = self._connected_reference(reference_rgbd, pixels)
        if int(self._component.sum()) < self._profile.min_anchors:
            raise ValueError('insufficient connected original rigid-patch anchors')
        self._center = self._anchors[self._component].mean(axis=0)
        centered = self._anchors[self._component]-self._center
        _, _, basis = np.linalg.svd(centered, full_matrices=False)
        self._basis = basis[:2].T
        self._query_uv = (self._points-self._center) @ self._basis
        self._anchor_uv = (self._anchors-self._center) @ self._basis
        self._identity = _digest({'schema': 'tatbot.rigid-patch-request/1',
            'reference_digest': reference_digest, 'points_reference_m': self._points.tolist(),
            'anchors_reference_m': self._anchors.tolist(), 'profile': self.profile,
            'residual_m': residual_m, 'rays_sha256': self._rays_digest,
            'material_roi': reference_rgbd.get('material_roi')})

    @property
    def profile(self):
        return asdict(self._profile)

    @property
    def identity(self):
        return self._identity

    @property
    def reference_digest(self):
        return self._reference_digest

    @property
    def residual_m(self):
        return self._residual_m

    def _connected_reference(self, rgbd, pixels):
        valid = _measurement_mask(rgbd)
        xy = np.rint(pixels).astype(int)
        x, y = xy[0]
        mask = np.zeros((self._shape[0]+2, self._shape[1]+2), np.uint8)
        mask[1:-1, 1:-1] = ~valid
        if not valid[y, x]:
            raise ValueError('original drawing lies outside the measured material')
        depth = np.where(valid, rgbd['depth_m'], 0).astype(np.float32)
        cv2.floodFill(depth, mask, (int(x), int(y)), 0, loDiff=2*self.residual_m,
                      upDiff=2*self.residual_m, flags=8 | cv2.FLOODFILL_MASK_ONLY | (2 << 8))
        connected = mask[xy[:, 1]+1, xy[:, 0]+1] == 2
        if not connected[:len(self._points)].all():
            raise ValueError('drawing spans disconnected original measured material')
        return connected[len(self._points):]

    def _validate_current(self, rgbd, matrix):
        _validate_rgbd(rgbd)
        if (matrix.shape != (4, 4) or not np.isfinite(matrix).all()
                or not np.allclose(matrix[3], [0, 0, 0, 1], atol=1e-9, rtol=0)
                or not np.allclose(matrix[:3, :3].T @ matrix[:3, :3], np.eye(3), atol=1e-7, rtol=0)
                or not np.isclose(np.linalg.det(matrix[:3, :3]), 1., atol=1e-7, rtol=0)):
            raise ValueError('rigid-patch support requires a finite rigid pose')
        if (rgbd['gray'].shape != self._shape or tuple(rgbd['depth_range']) != self._depth_range
                or hashlib.sha256(rgbd['rays'].tobytes()).hexdigest() != self._rays_digest):
            raise ValueError('original rigid-patch camera context changed')

    def _validate_anchor_point(self, point, index):
        if (not isinstance(point, dict) or type(point.get('id')) is not int or point.get('id') != index
                or type(point.get('supported')) is not bool
                or point.get('status') not in ('supported', 'contradictory', 'unknown', 'ambiguous')
                or point['supported'] != (point['status'] == 'supported')):
            raise ValueError('malformed rigid-patch anchor evidence')
        pixel = np.asarray(point.get('reference_pixel'), dtype=float)
        if pixel.shape != (2,) or not np.allclose(pixel, self._anchor_pixels[index], atol=1e-3, rtol=0):
            raise ValueError('rigid-patch anchor reference location changed')
        if 'current_pixel' not in point:
            raise ValueError('rigid-patch anchor current location missing')
        current = point['current_pixel']
        if current is not None:
            current = np.asarray(current, dtype=float)
            if current.shape != (2,) or not np.isfinite(current).all():
                raise ValueError('malformed rigid-patch anchor current location')
        if point['status'] in ('supported', 'contradictory') and current is None:
            raise ValueError('verified rigid-patch anchor current location missing')

    def _anchor_evidence(self, report, matrix):
        if (not schemas.is_schema(report, schemas.SURFACE, 'support')
                or report.get('reference_digest') != self.reference_digest
                or report.get('request_sha256') != self._anchor_request
                or report.get('motion_authority') is not False
                or type(report.get('point_count')) is not int
                or report.get('point_count') != len(self._anchors)):
            raise ValueError('rigid-patch anchor report identity changed')
        pose = np.asarray(report.get('transform_reference_to_camera'), dtype=float)
        if pose.shape != (4, 4) or not np.array_equal(pose, matrix):
            raise ValueError('rigid-patch anchor report pose changed')
        points = report.get('points')
        if not isinstance(points, list) or len(points) != len(self._anchors):
            raise ValueError('rigid-patch anchor report inventory changed')
        supported, contradictory = [], []
        for index, point in enumerate(points):
            self._validate_anchor_point(point, index)
            supported.append(point['supported'])
            contradictory.append(point['status'] == 'contradictory')
        if (type(report.get('supported_count')) is not int
                or type(report.get('contradictory_count')) is not int
                or report.get('supported_count') != sum(supported)
                or report.get('contradictory_count') != sum(contradictory)):
            raise ValueError('rigid-patch anchor report counts changed')
        return np.array(supported) & self._component, np.array(contradictory)

    def _enclosure(self, supported):
        ids = np.flatnonzero(supported)
        if len(ids) < self._profile.min_anchors:
            return ids, np.empty((0, 2)), np.zeros(len(self._points), bool), 'insufficient_rigid_anchors'
        uv = self._anchor_uv[ids]
        spread = np.linalg.svd(uv-uv.mean(axis=0), compute_uv=False)/np.sqrt(len(uv))
        minimum = self._profile.min_spread_residual_ratio*self.residual_m
        if len(spread) < 2 or spread[1] < minimum:
            return ids, np.empty((0, 2)), np.zeros(len(self._points), bool), 'ill_conditioned_rigid_anchors'
        hull = cv2.convexHull(uv.astype(np.float32)).reshape(-1, 2)
        inside = np.array([cv2.pointPolygonTest(hull, tuple(p.astype(float)), False) >= 0
                           for p in self._query_uv])
        return ids, hull, inside, 'drawing_outside_supported_anchor_hull' if not inside.all() else None

    def evaluate(self, rgbd, camera_from_reference, anchor_report):
        from surface_attachment import measured_xyz, project_measured_rays
        matrix = np.asarray(camera_from_reference, dtype=float)
        self._validate_current(rgbd, matrix)
        supported, contradictory = self._anchor_evidence(anchor_report, matrix)
        ids, hull, enclosed, reason = self._enclosure(supported)
        if contradictory.any():
            reason = 'original_anchor_appearance_contradiction'
        expected = self._points @ matrix[:3, :3].T + matrix[:3, 3]
        pixels, visible = project_measured_rays(expected, rgbd['rays'])
        measured, valid = measured_xyz(rgbd, pixels, 2*self.residual_m)
        valid &= visible
        residual = np.linalg.norm(measured-expected, axis=1)
        disagreement = valid & (residual > 2*self.residual_m)
        inferred = valid & ~disagreement & enclosed & ~contradictory.any()
        if reason in ('insufficient_rigid_anchors', 'ill_conditioned_rigid_anchors'):
            inferred[:] = False
        if reason is None and not inferred.all():
            reason = 'rigid_patch_depth_contradiction' if disagreement.any() else 'rigid_patch_depth_unavailable'
        states = np.where(disagreement, 'contradictory', np.where(inferred, 'rigid_inferred', 'unknown'))
        points = [{'id': i, 'supported': bool(inferred[i]), 'status': str(states[i]),
                   'reference_pixel': self._pixels[i].tolist(),
                   'current_pixel': pixels[i].tolist() if valid[i] else None,
                   'measured_residual_m': float(residual[i]) if valid[i] else None}
                  for i in range(len(expected))]
        anchors = [{key: point[key] for key in ('id', 'status', 'supported', 'current_pixel')}
                   | {'reference_point_m': self._anchors[i].tolist()}
                   for i, point in enumerate(anchor_report['points'])]
        return {**schemas.stamp(schemas.SURFACE, 'rigid-patch-support'), 'reference_digest': self.reference_digest,
                'request_sha256': self.identity, 'profile': self.profile,
                'support_mode': 'rigid-patch', 'scope': 'rigid-patch-inferred-centerline',
                'rigid_assumption': self._profile.assumption, 'deformation_qualified': False,
                'accepted': bool(inferred.all()), 'reason': reason or 'rigid_patch_supported',
                'point_count': len(points), 'supported_count': int(inferred.sum()),
                'contradictory_count': int(disagreement.sum()), 'points': points,
                'anchor_evidence': {'request_sha256': self._anchor_request,
                    'point_count': len(anchors), 'supported_count': anchor_report['supported_count'],
                    'contradictory_count': int(contradictory.sum()), 'enclosing_ids': ids.tolist(),
                    'hull_reference_uv_m': hull.tolist(), 'points': anchors},
                'transform_reference_to_camera': matrix.tolist(), 'motion_authority': False}
