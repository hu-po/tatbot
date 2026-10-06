"""Measured rigid projection for appearance proposals, never RGB-D evidence.

The current image is sampled in the immutable reference pixel chart so optical
flow can propose residual motion after a rigid roll. This chart does not replace
the current measured depth or calibrated rays, and does not establish identity.
"""

from __future__ import annotations

import collections
import hashlib

import numpy as np

_MARGIN_PX = 48
_BATCH = 16384
_MAX_PIXELS = 1280 * 960
# The dense projection map is a pure function of the reference measurement,
# the current ray field, the ROI and the proposal pose. A stationary material
# seen by a stationary camera produces the same map frame after frame; a
# bounded memo keeps that from costing a full dense projection per frame.
# The map is only a search proposal, so a pose bucket of 1e-5 (metres and
# rotation-matrix entries) is far inside the search it seeds.
_MAP_CACHE = collections.OrderedDict()
_MAP_CACHE_LIMIT = 8
_POSE_BUCKET = 1e-5


def _roi_bounds(roi, shape):
    height, width = shape
    if roi is None:
        return 0, 0, width, height
    if (not isinstance(roi, (tuple, list)) or len(roi) != 4
            or any(type(value) is not int for value in roi)):
        raise ValueError('appearance warp ROI must be integer x,y,width,height')
    x, y, w, h = roi
    if min(x, y) < 0 or min(w, h) <= 0 or x+w > width or y+h > height:
        raise ValueError('appearance warp ROI is outside the reference image')
    return max(0, x-_MARGIN_PX), max(0, y-_MARGIN_PX), min(width, x+w+_MARGIN_PX), min(height, y+h+_MARGIN_PX)


def _validate_context(rgbd):
    from surface_material_support import _validate_rgbd

    _validate_rgbd(rgbd)
    height, width = rgbd['gray'].shape
    if height > 960 or width > 1280:
        raise ValueError('appearance warp exceeds bounded camera profile')
    if 'material_roi' in rgbd:
        _roi_bounds(rgbd['material_roi'], (height, width))
    rays = rgbd['rays'][..., :2]
    # A finite but folded/constant ray array is not an invertible calibration.
    if not _positive_cells(rays, 1e-12).all():
        raise ValueError('appearance warp requires invertible calibrated rays')


def _cross(a, b):
    return a[..., 0]*b[..., 1] - a[..., 1]*b[..., 0]


def _positive_cells(pixels, minimum_area):
    a, b, c, d = pixels[:-1, :-1], pixels[:-1, 1:], pixels[1:, :-1], pixels[1:, 1:]
    cells = np.ones(a.shape[:2], bool)
    # Every corner must preserve orientation. A centered derivative alone can
    # miss a fold across one corner of a quadrilateral.
    for origin, first, second in ((a, b, c), (b, d, a), (d, c, b), (c, a, d)):
        cells &= _cross(first-origin, second-origin) > minimum_area
    return cells


def _corners(pixels, shape):
    height, width = shape
    finite = np.isfinite(pixels).all(axis=1)
    safe = np.clip(np.where(finite[:, None], pixels, 0), [-1, -1], [width, height])
    # Clip before converting to integer so arbitrarily large rejected positions
    # cannot overflow an index or become an accidental valid interpolation.
    low = np.floor(safe).astype(int)
    fx, fy = (safe-low).T
    for dx, dy, weight in ((0, 0, (1-fx)*(1-fy)), (1, 0, fx*(1-fy)),
                           (0, 1, (1-fx)*fy), (1, 1, fx*fy)):
        x, y = low[:, 0]+dx, low[:, 1]+dy
        inside = finite & (x >= 0) & (x < width) & (y >= 0) & (y < height)
        yield np.clip(x, 0, width-1), np.clip(y, 0, height-1), weight, inside


def _contributing_valid(mask, pixels):
    valid = np.isfinite(pixels).all(axis=1)
    valid &= ((pixels >= 0).all(axis=1)
              & (pixels[:, 0] <= mask.shape[1]-1) & (pixels[:, 1] <= mask.shape[0]-1))
    for x, y, weight, inside in _corners(pixels, mask.shape):
        valid &= (weight <= 0) | (inside & mask[y, x])
    return valid


def _unfolded_vertices(pixels, valid):
    """Reject every vertex adjoining a folded, collapsed or unmeasured cell."""
    cells = valid[:-1, :-1] & valid[:-1, 1:] & valid[1:, :-1] & valid[1:, 1:]
    cells &= _positive_cells(pixels, 1e-8)
    vertices = np.zeros(valid.shape, bool)
    vertices[1:-1, 1:-1] = cells[:-1, :-1] & cells[:-1, 1:] & cells[1:, :-1] & cells[1:, 1:]
    return vertices


class MeasuredAppearanceWarp:
    """One bounded capture/pose proposal; returned arrays have no motion authority.

    Inputs must remain unchanged for this instance's lifetime. The cached map is
    bounded by the requested reference ROI plus 48 pixels of LK pyramid margin.
    Depth disagreement in the current frame never removes an appearance sample:
    downstream evidence must compare the actual current measurements explicitly.
    """

    def __init__(self, reference_rgbd, current_rgbd, camera_from_reference, *, roi=None, residual_m=.0015):
        from surface_material_support import _measurement_mask

        _validate_context(reference_rgbd)
        _validate_context(current_rgbd)
        matrix = np.array(camera_from_reference, dtype=float, copy=True)
        if (matrix.shape != (4, 4) or not np.isfinite(matrix).all()
                or not np.allclose(matrix[3], [0, 0, 0, 1], atol=1e-8, rtol=0)
                or not np.allclose(matrix[:3, :3].T @ matrix[:3, :3], np.eye(3), atol=1e-8, rtol=0)
                or not np.isclose(np.linalg.det(matrix[:3, :3]), 1, atol=1e-8, rtol=0)
                or not np.isfinite(residual_m) or not 0 < residual_m <= .02):
            raise ValueError('appearance warp requires a finite proper rigid pose and bounded residual')
        matrix.setflags(write=False)
        self._reference, self._current = reference_rgbd, current_rgbd
        self._matrix, self._residual_m = matrix, residual_m
        self._shape = reference_rgbd['gray'].shape
        self._bounds = _roi_bounds(roi, self._shape)
        self._source_mask = _measurement_mask(reference_rgbd)
        self._current_mask = _measurement_mask(current_rgbd)
        self._map = self._map_valid = self._tracking = None
        self._map_key = None

    def _memo_key(self):
        if self._map_key is None:
            digest = hashlib.sha256()
            for array in (self._reference['gray'], self._reference['depth_m'], self._reference['rays'],
                          self._current['rays']):
                digest.update(np.ascontiguousarray(array).tobytes())
            digest.update(np.round(self._matrix / _POSE_BUCKET).astype(np.int64).tobytes())
            digest.update(repr((self._bounds, self._residual_m, tuple(self._reference['depth_range']))).encode())
            self._map_key = digest.hexdigest()
        return self._map_key

    def _project_measured(self, pixels):
        from surface_attachment import measured_xyz, project_measured_rays

        valid = _contributing_valid(self._source_mask, pixels)
        safe = np.where(valid[:, None], pixels, 0)
        xyz, measured = measured_xyz(self._reference, safe, 2*self._residual_m)
        transformed = xyz @ self._matrix[:3, :3].T + self._matrix[:3, 3]
        projected, visible = project_measured_rays(transformed, self._current['rays'])
        valid &= measured & visible & np.isfinite(projected).all(axis=1)
        return np.where(valid[:, None], projected, 0), valid

    def requires_warp(self, reference_pixels, window_px=21):
        """Select resampling when nine measured footprint probes change shape.

        A deviation of at most 0.25 pixels from each center's pure translation
        keeps the original proposal/scorer. This is a proposal selection bound,
        not an appearance, depth, or forward/backward acceptance tolerance. The
        nine samples are not a guarantee about unsampled interior curvature;
        actual raw-image and measured-depth evidence must still pass downstream.
        """
        pixels = np.asarray(reference_pixels, dtype=float)
        if pixels.ndim != 2 or pixels.shape[1:] != (2,) or len(pixels) > _MAX_PIXELS:
            raise ValueError('appearance projection requires a bounded (N, 2) pixel array')
        if type(window_px) is not int or not 9 <= window_px <= 61 or window_px % 2 != 1:
            raise ValueError('appearance footprint requires an odd window from 9 to 61 pixels')
        if not len(pixels) or not np.isfinite(pixels).all():
            return True
        half = window_px // 2
        offsets = np.array([(x, y) for y in (-half, 0, half) for x in (-half, 0, half)], dtype=float)
        # Expand only this bounded batch, rather than all requested footprints.
        count = _BATCH // len(offsets)
        for start in range(0, len(pixels), count):
            probes = pixels[start:start+count, None, :] + offsets
            projected, valid = self._project_measured(probes.reshape(-1, 2))
            if not valid.all():
                return True
            projected = projected.reshape(-1, len(offsets), 2)
            error = projected-projected[:, 4:5, :]-offsets
            if np.any(np.linalg.norm(error, axis=2) > .25):
                return True
        return False

    def _ensure_map(self):
        if self._map is not None:
            return
        key = self._memo_key()
        cached = _MAP_CACHE.get(key)
        if cached is not None:
            _MAP_CACHE.move_to_end(key)
            self._map, self._map_valid = cached
            return
        x0, y0, x1, y1 = self._bounds
        yy, xx = np.mgrid[y0:y1, x0:x1]
        pixels = np.column_stack([xx.ravel(), yy.ravel()]).astype(float)
        projected, valid = np.zeros(pixels.shape), np.zeros(len(pixels), bool)
        for start in range(0, len(pixels), _BATCH):
            stop = min(start+_BATCH, len(pixels))
            projected[start:stop], valid[start:stop] = self._project_measured(pixels[start:stop])
        local_map = projected.reshape(y1-y0, x1-x0, 2)
        local_valid = _unfolded_vertices(local_map, valid.reshape(y1-y0, x1-x0))
        self._map = np.zeros((*self._shape, 2))
        self._map_valid = np.zeros(self._shape, bool)
        self._map[y0:y1, x0:x1] = local_map
        self._map_valid[y0:y1, x0:x1] = local_valid
        self._map.setflags(write=False)
        self._map_valid.setflags(write=False)
        _MAP_CACHE[key] = (self._map, self._map_valid)
        while len(_MAP_CACHE) > _MAP_CACHE_LIMIT:
            _MAP_CACHE.popitem(last=False)

    def project(self, reference_pixels):
        """Interpolate the measured projection proposal; masked coordinates are zero.

        Every contributing map vertex comes from original measured XYZ projected
        through the current calibrated ray field. This interpolation only guides
        appearance search; it never substitutes for final measured point geometry.
        """
        pixels = np.asarray(reference_pixels, dtype=float)
        if pixels.ndim != 2 or pixels.shape[1:] != (2,) or len(pixels) > _MAX_PIXELS:
            raise ValueError('appearance projection requires a bounded (N, 2) pixel array')
        self._ensure_map()
        output, valid = np.zeros(pixels.shape), np.zeros(len(pixels), bool)
        for start in range(0, len(pixels), _BATCH):
            stop = min(start+_BATCH, len(pixels))
            batch = pixels[start:stop]
            usable = _contributing_valid(self._map_valid, batch)
            projected = np.zeros(batch.shape)
            for x, y, weight, _ in _corners(batch, self._shape):
                projected += weight[:, None]*self._map[y, x]
            output[start:stop] = np.where(usable[:, None], projected, 0)
            valid[start:stop] = usable
        return output, valid

    def tracking_image(self):
        """Return normalized current appearance in the reference chart and its mask."""
        from surface_attachment import _tracking_image

        if self._tracking is not None:
            return self._tracking
        self._ensure_map()
        normalized = _tracking_image(self._current['gray'])
        image, mask = np.full(self._shape, 128, np.uint8), self._map_valid.copy()
        indices = np.flatnonzero(mask)
        for start in range(0, len(indices), _BATCH):
            selected = indices[start:start+_BATCH]
            pixels = self._map.reshape(-1, 2)[selected]
            valid = _contributing_valid(self._current_mask, pixels)
            sampled = np.zeros(len(pixels))
            for x, y, weight, _ in _corners(pixels, self._current_mask.shape):
                sampled += weight*normalized[y, x]
            image.ravel()[selected] = np.where(valid, np.clip(np.rint(sampled), 0, 255), 128).astype(np.uint8)
            mask.ravel()[selected] &= valid
        image.setflags(write=False)
        mask.setflags(write=False)
        self._tracking = image, mask
        return self._tracking
