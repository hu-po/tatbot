"""Bounded original-material appearance evidence at explicit measured locations.

A pose predicts a search location; it is never evidence that the texture there
belongs to the original material. Missing or competing matches remain unknown.
These development criteria confer neither physical accuracy nor motion authority.
"""
from __future__ import annotations

import copy
import hashlib
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "lib"))
import schemas  # noqa: E402


@dataclass(frozen=True)
class SupportProfile:
    scorer: str = "raw-measured-warp-connected-ncc/2"
    min_scoring_coverage: float = 0.9
    window_px: int = 21
    forward_backward_px: float = 0.75
    max_photometric_error: float = 20.0
    min_correlation: float = 0.8
    correlation_margin: float = 0.2
    min_texture_std: float = 1.0

    def __post_init__(self):
        if (self.scorer != "raw-measured-warp-connected-ncc/2"
                or not 0.9 <= self.min_scoring_coverage <= 1
                or type(self.window_px) is not int or not 9 <= self.window_px <= 61 or self.window_px % 2 != 1
                or not 0 < self.forward_backward_px <= 0.75
                or not 0 < self.max_photometric_error <= 20
                or not 0 < self.min_correlation <= 1 or not 0 < self.correlation_margin <= 1
                or not 0 < self.min_texture_std <= 255):
            raise ValueError('invalid original-appearance support profile')


def _digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def _measurement_mask(rgbd):
    depth = rgbd['depth_m']
    near, far = rgbd['depth_range']
    valid = np.isfinite(depth) & (depth > near) & (depth < far)
    if 'material_roi' in rgbd:
        rx, ry, rw, rh = rgbd['material_roi']
        iy, ix = np.ogrid[:depth.shape[0], :depth.shape[1]]
        valid &= (ix >= rx) & (ix < rx+rw) & (iy >= ry) & (iy < ry+rh)
    return valid


def _scoring_patches(rgbd, points, width, residual_m, valid=None):
    """Raw image samples whose interpolation and depth belong to the query surface."""
    half = width // 2
    yy, xx = np.mgrid[-half:half+1, -half:half+1]
    offsets = np.column_stack([xx.ravel(), yy.ravel()]).astype(np.float32)
    safe = np.nan_to_num(points, nan=-10000, posinf=-10000, neginf=-10000).astype(np.float32)
    positions = safe[:, None, :] + offsets
    return _scoring_positions(rgbd, positions, width, residual_m, valid)


def _scoring_positions(rgbd, positions, width, residual_m, valid=None, position_valid=None):
    """Score actual raw pixels with explicit interpolation and connected depth."""
    half = width//2
    positions = np.nan_to_num(positions, nan=-10000, posinf=-10000, neginf=-10000)
    x, y = positions[..., 0], positions[..., 1]
    depth = rgbd['depth_m']
    if valid is None:
        valid = _measurement_mask(rgbd)
    # Explicit bilinear weights are shared by raw image and depth. OpenCV's
    # remap weights vary by dtype/backend, so its interpolation cannot establish
    # which material pixels contributed to an appearance score.
    image, z = np.zeros(x.shape), np.zeros(x.shape, np.float32)
    x0, y0 = np.floor(x).astype(int), np.floor(y).astype(int)
    fx, fy = x-x0, y-y0
    low, high = np.full(x.shape, np.inf), np.full(x.shape, -np.inf)
    available = np.ones(x.shape, bool)
    for dx, dy, weight in ((0, 0, (1-fx)*(1-fy)), (1, 0, fx*(1-fy)),
                           (0, 1, (1-fx)*fy), (1, 1, fx*fy)):
        cx, cy, active = x0+dx, y0+dy, weight > 0
        inside = (cx >= 0) & (cx < depth.shape[1]) & (cy >= 0) & (cy < depth.shape[0])
        cx, cy = np.clip(cx, 0, depth.shape[1]-1), np.clip(cy, 0, depth.shape[0]-1)
        available &= ~active | (inside & valid[cy, cx])
        corner = depth[cy, cx]
        image += np.where(active, weight*rgbd['gray'][cy, cx], 0)
        z += weight*np.where(valid[cy, cx], corner, 0)
        low = np.minimum(low, np.where(active, corner, np.inf))
        high = np.maximum(high, np.where(active, corner, -np.inf))
    available &= high-low <= 2*residual_m
    if position_valid is not None:
        available &= position_valid
    masks = np.zeros_like(available)
    for index in range(len(positions)):
        usable = available[index].reshape(width, width)
        if not usable[half, half]:
            continue
        mask = np.zeros((width+2, width+2), np.uint8)
        mask[1:-1, 1:-1] = ~usable
        patch = z[index].reshape(width, width).copy()
        cv2.floodFill(patch, mask, (half, half), 0, loDiff=2*residual_m, upDiff=2*residual_m,
                      flags=8 | cv2.FLOODFILL_MASK_ONLY | (2 << 8))
        masks[index] = (mask[1:-1, 1:-1] == 2).ravel()
    return image, masks


def _inside(points, shape, half):
    height, width = shape
    return (np.isfinite(points).all(axis=1) & (points[:, 0] >= half) & (points[:, 1] >= half)
            & (points[:, 0] < width-half-1) & (points[:, 1] < height-half-1))


def _validate_rgbd(rgbd):
    if not isinstance(rgbd, dict) or not all(isinstance(rgbd.get(key), np.ndarray)
                                           for key in ('gray', 'depth_m', 'rays')):
        raise ValueError('material support requires measured RGB-D arrays')
    gray, depth, rays = (rgbd[key] for key in ('gray', 'depth_m', 'rays'))
    bounds = rgbd.get('depth_range')
    if (gray.ndim != 2 or gray.dtype != np.uint8 or not 0 < gray.size <= 2_000_000 or min(gray.shape) < 4
            or depth.shape != gray.shape or not np.issubdtype(depth.dtype, np.floating)
            or rays.shape != (*gray.shape, 3) or not np.issubdtype(rays.dtype, np.floating)
            or not np.isfinite(rays).all() or not np.allclose(rays[..., 2], 1., atol=1e-6, rtol=0)
            or not isinstance(bounds, (tuple, list)) or len(bounds) != 2
            or not all(isinstance(value, (int, float)) and np.isfinite(value) for value in bounds)
            or not 0 <= bounds[0] < bounds[1]):
        raise ValueError('invalid material support RGB-D camera context')


class OriginalAppearance:
    """Nine starts and a bounded score comparison, always against the original."""

    def __init__(self, rgbd, pixels, profile, residual_m):
        from surface_attachment import _tracking_image
        self.profile = profile
        self.residual_m = residual_m
        self.rgbd = {key: np.array(rgbd[key], copy=True) for key in ('gray', 'depth_m', 'rays')}
        for array in self.rgbd.values():
            array.setflags(write=False)
        self.rgbd['depth_range'] = tuple(rgbd['depth_range'])
        if 'material_roi' in rgbd:
            self.rgbd['material_roi'] = tuple(rgbd['material_roi'])
        self.gray = _tracking_image(self.rgbd['gray'])
        self.pixels = np.array(pixels, dtype=np.float32, copy=True)
        self.pixels.setflags(write=False)
        self._reference_valid = _measurement_mask(self.rgbd)
        self._reference_valid.setflags(write=False)
        self._original_patches = self._cache_original_patches()
        self.original = self._match(self.rgbd, self.pixels)

    def _cache_original_patches(self):
        count, area = len(self.pixels), self.profile.window_px**2
        # Keep large configurable requests on the existing bounded batch path.
        if count*area*(np.dtype(np.float64).itemsize + np.dtype(bool).itemsize) > 32*1024*1024:
            return None
        values, masks = np.empty((count, area)), np.empty((count, area), bool)
        for start in range(0, count, 32):
            stop = min(start+32, count)
            values[start:stop], masks[start:stop] = _scoring_patches(
                self.rgbd, self.pixels[start:stop], self.profile.window_px,
                self.residual_m, self._reference_valid)
        values.setflags(write=False)
        masks.setflags(write=False)
        return values, masks

    def _hypotheses(self, gray, pixels, starts=None):
        p = self.profile
        half = p.window_px // 2
        offsets = np.array([(x, y) for x in (-half, 0, half) for y in (-half, 0, half)], np.float32)
        starts = pixels[:, None, :] + offsets if starts is None else starts
        count = starts.shape[1]
        original = np.repeat(self.pixels, count, axis=0).reshape(-1, 1, 2)
        initial = starts.reshape(-1, 1, 2).astype(np.float32)
        tracked, forward, error = cv2.calcOpticalFlowPyrLK(
            self.gray, gray, original, initial.copy(), flags=cv2.OPTFLOW_USE_INITIAL_FLOW,
            winSize=(p.window_px, p.window_px), maxLevel=2)
        if tracked is None or forward is None or error is None:
            return initial.reshape(-1, count, 2), np.zeros((len(pixels), count), bool)
        finite = np.isfinite(tracked).all(axis=(1, 2))
        safe = np.where(finite[:, None, None], tracked, original).astype(np.float32)
        back, backward, _ = cv2.calcOpticalFlowPyrLK(
            gray, self.gray, safe, original.copy(), flags=cv2.OPTFLOW_USE_INITIAL_FLOW,
            winSize=(p.window_px, p.window_px), maxLevel=2)
        if back is None or backward is None:
            return safe.reshape(-1, count, 2), np.zeros((len(pixels), count), bool)
        valid = (finite & forward.ravel().astype(bool) & backward.ravel().astype(bool)
                 & (np.linalg.norm(back-original, axis=(1, 2)) <= p.forward_backward_px)
                 & (error.ravel() <= p.max_photometric_error)
                 & _inside(safe.reshape(-1, 2), gray.shape, half))
        return safe.reshape(-1, count, 2), valid.reshape(-1, count)

    def _warped_hypotheses(self, rgbd, pixels, warp, extra_pixels):
        from surface_attachment import _tracking_image
        legacy, legacy_valid = self._hypotheses(_tracking_image(rgbd['gray']), pixels)
        seeds = legacy if extra_pixels is None else np.concatenate([legacy, extra_pixels], axis=1)
        seed_valid = legacy_valid if extra_pixels is None else np.concatenate(
            [legacy_valid, np.isfinite(extra_pixels).all(axis=2)], axis=1)
        # A local inverse Jacobian only seeds searches in the material chart.
        # Legacy and descriptor coordinates are refined there, not discarded
        # because they disagree with the pose proposal or frozen as biased LK.
        basis = self.pixels[:, None, :] + np.array([[0, 0], [1, 0], [0, 1]])
        mapped, supported = warp.project(basis.reshape(-1, 2))
        mapped = mapped.reshape(-1, 3, 2)
        jacobian = np.stack([mapped[:, 1]-mapped[:, 0], mapped[:, 2]-mapped[:, 0]], axis=2)
        good = supported.reshape(-1, 3).all(axis=1) & (np.abs(np.linalg.det(jacobian)) > 1e-6)
        inverse = np.linalg.inv(np.where(good[:, None, None], jacobian, np.eye(2)))
        delta = np.einsum('nij,nkj->nki', inverse, np.nan_to_num(seeds)-mapped[:, :1])
        converted = self.pixels[:, None, :] + np.clip(delta, -64, 64)
        half = self.profile.window_px//2
        offsets = np.array([(x, y) for x in (-half, 0, half) for y in (-half, 0, half)])
        starts = np.concatenate([self.pixels[:, None, :]+offsets, converted], axis=1)
        chart, chart_mask = warp.tracking_image()
        hypotheses, valid = self._hypotheses(chart, self.pixels, starts)
        valid[:, 9:] &= seed_valid & good[:, None]
        current, visible = warp.project(hypotheses.reshape(-1, 2))
        current = current.reshape(hypotheses.shape)
        valid &= visible.reshape(valid.shape)
        # A chart cannot erase an independently found raw-current alternative.
        # Replace its biased translation-only location only when measured chart
        # context and a nearby successful refinement support that replacement.
        legacy_seeds = converted[:, :9]
        coverage = self._chart_coverage(chart_mask, legacy_seeds)
        replaced = (valid[:, 9:18] & (coverage >= self.profile.min_scoring_coverage)
                    & (np.linalg.norm(current[:, 9:18]-legacy, axis=2) <= 3))
        fallback = legacy_valid & ~replaced
        return (np.concatenate([current, legacy], axis=1),
                np.concatenate([valid, fallback], axis=1))

    def _chart_coverage(self, mask, seeds):
        half = self.profile.window_px//2
        yy, xx = np.mgrid[-half:half+1, -half:half+1]
        offsets = np.column_stack([xx.ravel(), yy.ravel()])
        from surface_appearance_warp import _contributing_valid
        positions = seeds[:, :, None, :] + offsets
        valid = _contributing_valid(mask, positions.reshape(-1, 2))
        return valid.reshape(*seeds.shape[:2], -1).mean(axis=2)

    def _scores(self, rgbd, hypotheses, warp=None, *, hypothesis_valid=None):
        scores = np.full(hypotheses.shape[:2], -1., np.float64)
        coverage = np.zeros(hypotheses.shape[:2])
        textured = np.zeros(hypotheses.shape[:2], bool)
        hypothesis_valid = (np.ones(hypotheses.shape[:2], bool) if hypothesis_valid is None
                            else np.asarray(hypothesis_valid, dtype=bool))
        if hypothesis_valid.shape != hypotheses.shape[:2]:
            raise ValueError('appearance hypothesis validity shape changed')
        current_valid = self._reference_valid if rgbd is self.rgbd else _measurement_mask(rgbd)
        # Cached arrays follow the exact original point inventory, including
        # pruning. Current context is rebuilt for each capture, never retained.
        for start in range(0, len(hypotheses), 32):
            stop = min(start+32, len(hypotheses))
            point_rows, candidate_rows = np.nonzero(hypothesis_valid[start:stop])
            if not len(point_rows):
                continue
            if self._original_patches is None:
                original, om = _scoring_patches(self.rgbd, self.pixels[start:stop],
                    self.profile.window_px, self.residual_m, self._reference_valid)
            else:
                original, om = (array[start:stop] for array in self._original_patches)
            original, om = original[point_rows], om[point_rows]
            if warp is None:
                current, cm = _scoring_patches(rgbd, hypotheses[start:stop][point_rows, candidate_rows],
                                                self.profile.window_px, self.residual_m, current_valid)
            else:
                current, cm = self._warped_scores(rgbd, hypotheses[start:stop], start, warp,
                                                  current_valid, point_rows, candidate_rows)
            joint = om & cm
            count = joint.sum(axis=1)
            denominator = np.maximum(count, 1)
            slots = start+point_rows, candidate_rows
            coverage[slots] = count / self.profile.window_px**2
            a = original - (original*joint).sum(axis=1)[:, None] / denominator[:, None]
            b = current - (current*joint).sum(axis=1)[:, None] / denominator[:, None]
            a, b = a*joint, b*joint
            an, bn = np.linalg.norm(a, axis=1), np.linalg.norm(b, axis=1)
            textured[slots] = ((an/np.sqrt(denominator) >= self.profile.min_texture_std)
                              & (bn/np.sqrt(denominator) >= self.profile.min_texture_std))
            scores[slots] = (a*b).sum(axis=1) / np.maximum(an*bn, 1e-6)
        return scores, textured, coverage

    def _warped_scores(self, rgbd, hypotheses, start, warp, current_valid, point_rows, candidate_rows):
        width = self.profile.window_px
        half = width//2
        yy, xx = np.mgrid[-half:half+1, -half:half+1]
        offsets = np.column_stack([xx.ravel(), yy.ravel()])
        selected_points, inverse = np.unique(point_rows, return_inverse=True)
        chart = self.pixels[start+selected_points, None, :] + offsets
        mapped, available = warp.project(chart.reshape(-1, 2))
        mapped = mapped.reshape(len(selected_points), width*width, 2)
        available = available.reshape(len(selected_points), width*width)
        center = mapped[:, width*width//2]
        selected_hypotheses = hypotheses[point_rows, candidate_rows]
        positions = mapped[inverse] + (selected_hypotheses-center[inverse])[:, None, :]
        mask = available[inverse]
        # Actual current depth determines connectivity, never agreement with
        # predicted depth. Recognized depth contradictions must survive scoring.
        return _scoring_positions(rgbd, positions.reshape(-1, width*width, 2), width,
            self.residual_m, current_valid, mask.reshape(-1, width*width))

    def _match(self, rgbd, pixels, warp=None, extra_pixels=None):
        from surface_attachment import _tracking_image
        hypotheses, valid = (self._hypotheses(_tracking_image(rgbd['gray']), pixels) if warp is None
                            else self._warped_hypotheses(rgbd, pixels, warp, extra_pixels))
        score, texture, coverage = self._scores(rgbd, hypotheses, warp, hypothesis_valid=valid)
        # Unmeasured or textureless alternatives cannot disappear to manufacture
        # a unique match. This rule also applies to original self recognition.
        unresolved = (valid & ((coverage < self.profile.min_scoring_coverage) | ~texture)).any(axis=1)
        scored = np.where(valid, score, -np.inf)
        selected = scored.argmax(axis=1)
        rows = np.arange(len(pixels))
        best = hypotheses[rows, selected]
        distance = np.linalg.norm(hypotheses-best[:, None, :], axis=2)
        same = distance <= self.profile.forward_backward_px
        alternative = np.where(valid & ~same, score, -np.inf).max(axis=1)
        best_score = scored[rows, selected]
        qualified = (valid.any(axis=1) & ~unresolved
                     & (best_score >= self.profile.min_correlation)
                     & _inside(self.pixels, self.gray.shape, self.profile.window_px//2))
        margin = np.full(len(pixels), np.inf)
        np.subtract(best_score, alternative, out=margin, where=np.isfinite(alternative))
        # Direct extra seeds do not manufacture the two-start confirmation.
        confirmed = (valid[:, :9] & same[:, :9]).sum(axis=1) >= 2
        if valid.shape[1] >= 18:
            confirmed |= (valid[:, 9:18] & same[:, 9:18]).sum(axis=1) >= 2
        unique = (qualified & confirmed
                  & (margin >= self.profile.correlation_margin))
        return {'pixels': best, 'unique': unique, 'qualified': qualified,
                'correlation': best_score, 'alternative_correlation': alternative,
                'scoring_coverage': coverage[rows, selected], 'unresolved_context': unresolved}

    def match(self, rgbd, pixels, original_window=None, *, warp=None, extra_pixels=None):
        window = (np.ones(len(pixels), dtype=bool) if original_window is None
                  else np.asarray(original_window, dtype=bool))
        if window.shape != (len(pixels),):
            raise ValueError('original appearance window shape changed')
        if extra_pixels is not None:
            extra_pixels = np.asarray(extra_pixels, dtype=float)
            if (extra_pixels.ndim != 3 or extra_pixels.shape[0] != len(pixels)
                    or extra_pixels.shape[2] != 2 or not 1 <= extra_pixels.shape[1] <= 2):
                raise ValueError('original appearance extra seeds exceed bounded inventory')
        # Original ambiguity and missing original material context cannot gain
        # current support. Keep those classifications without claiming a match.
        active = np.flatnonzero(self.original['unique'] & window)
        result = {'pixels': np.asarray(pixels).copy(), 'unique': np.zeros(len(pixels), bool),
                  'qualified': self.original['qualified'] & ~self.original['unique'] & window,
                  'correlation': np.full(len(pixels), -np.inf),
                  'alternative_correlation': np.full(len(pixels), -np.inf),
                  'scoring_coverage': np.zeros(len(pixels)),
                  'unresolved_context': self.original['unresolved_context'].copy()}
        if len(active):
            subset = copy.copy(self)
            subset.pixels = self.pixels[active]
            if self._original_patches is not None:
                subset._original_patches = tuple(array[active] for array in self._original_patches)
            observed = subset._match(rgbd, np.asarray(pixels)[active], warp,
                                     None if extra_pixels is None else extra_pixels[active])
            observed['unique'] &= self.original['unique'][active]
            observed['qualified'] &= self.original['qualified'][active]
            for key in result:
                result[key][active] = observed[key]
        return result


class MaterialSupport:
    """Immutable request for observed material, independent of surface shape."""

    def __init__(self, reference_rgbd, reference_points, reference_digest, residual_m=0.0015, profile=None):
        from surface_attachment import measured_xyz, project_measured_rays
        _validate_rgbd(reference_rgbd)
        points = np.asarray(reference_points, dtype=float)
        if (points.ndim != 2 or points.shape[1:] != (3,) or not 1 <= len(points) <= 4096
                or not np.isfinite(points).all() or not isinstance(reference_digest, str)
                or len(reference_digest) != 64 or not 0 < residual_m <= 0.02):
            raise ValueError('invalid bounded original-material support request')
        self._points = points.copy()
        self._points.setflags(write=False)
        self._reference_digest = reference_digest
        self._residual_m = residual_m
        self._profile = profile or SupportProfile()
        if not isinstance(self._profile, SupportProfile):
            raise ValueError('material support requires a validated appearance profile')
        self._shape = reference_rgbd['gray'].shape
        self._rays_digest = hashlib.sha256(reference_rgbd['rays'].tobytes()).hexdigest()
        self._depth_range = tuple(reference_rgbd['depth_range'])
        pixels, visible = project_measured_rays(points, reference_rgbd['rays'])
        safe = np.nan_to_num(pixels, nan=0, posinf=0, neginf=0)
        measured, valid = measured_xyz(reference_rgbd, safe, 2*residual_m)
        if not (visible & valid & (np.linalg.norm(measured-points, axis=1) <= 2*residual_m)).all():
            raise ValueError('original measured support missing or changed at the drawing')
        self._appearance = OriginalAppearance(reference_rgbd, pixels, self._profile, residual_m)
        self._original_window = self._window_support(reference_rgbd, pixels)
        roi = reference_rgbd.get('material_roi')
        if roi is not None:
            x, y, w, h = roi
            half = self._profile.window_px//2
            self._original_window &= ((pixels[:, 0]-half >= x) & (pixels[:, 0]+half < x+w)
                                      & (pixels[:, 1]-half >= y) & (pixels[:, 1]+half < y+h))
        self._identity = _digest({**schemas.stamp(schemas.SURFACE, 'support-request'),
            'reference_digest': reference_digest, 'points_reference_m': points.tolist(),
            'profile': self.profile, 'residual_m': residual_m,
            'material_roi': None if roi is None else list(roi)})

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

    def _window_support(self, rgbd, pixels):
        near, far = self._depth_range
        depth = rgbd['depth_m']
        valid = (np.isfinite(depth) & (depth > near) & (depth < far)).astype(np.float32)
        width = self._profile.window_px
        coverage = cv2.boxFilter(valid, -1, (width, width), normalize=True, borderType=cv2.BORDER_CONSTANT)
        x, y = np.rint(np.clip(pixels, [0, 0], [depth.shape[1]-1, depth.shape[0]-1])).astype(int).T
        supported = (coverage[y, x] >= 0.9) & _inside(pixels, depth.shape, width//2)
        half = width//2
        for index in np.flatnonzero(supported):
            cx, cy = x[index], y[index]
            patch = depth[cy-half:cy+half+1, cx-half:cx+half+1].astype(np.float32)
            usable = valid[cy-half:cy+half+1, cx-half:cx+half+1].astype(bool)
            mask = np.zeros((width+2, width+2), np.uint8)
            mask[1:-1, 1:-1] = ~usable
            # Floating-range flood fill follows measured neighboring depths,
            # allowing smooth slopes without borrowing texture across a step.
            count, _, _, _ = cv2.floodFill(patch, mask, (half, half), 0,
                loDiff=2*self.residual_m, upDiff=2*self.residual_m,
                flags=8 | cv2.FLOODFILL_MASK_ONLY)
            supported[index] = count >= 0.9*width*width
        return supported

    def evaluate(self, rgbd, camera_from_reference):
        from surface_appearance_warp import MeasuredAppearanceWarp
        from surface_attachment import measured_xyz, project_measured_rays
        matrix = np.asarray(camera_from_reference, dtype=float)
        self._validate_current(rgbd, matrix)
        expected = self._points @ matrix[:3, :3].T + matrix[:3, 3]
        pixels, visible = project_measured_rays(expected, rgbd['rays'])
        search = np.where(visible[:, None], pixels, self._appearance.pixels)
        try:
            warp = MeasuredAppearanceWarp(self._appearance.rgbd, rgbd, matrix,
                roi=self._appearance.rgbd.get('material_roi'), residual_m=self.residual_m)
            active = self._appearance.original['unique'] & self._original_window
            if active.any() and not warp.requires_warp(self._appearance.pixels[active], self._profile.window_px):
                warp = None
            evidence = self._appearance.match(rgbd, search, self._original_window, warp=warp)
        except cv2.error as error:
            raise ValueError('original material appearance computation failed') from error
        image_valid = (visible & evidence['unique'] & self._original_window
                       & self._window_support(rgbd, evidence['pixels']))
        selected = np.flatnonzero(image_valid)
        measured = np.zeros_like(expected)
        depth_valid = np.zeros(len(expected), bool)
        if len(selected):
            measured[selected], depth_valid[selected] = measured_xyz(
                rgbd, evidence['pixels'][selected], 2*self.residual_m)
        residual = np.linalg.norm(measured-expected, axis=1)
        contradiction = image_valid & depth_valid & (residual > 2*self.residual_m)
        supported = image_valid & depth_valid & ~contradiction
        states = np.where(evidence['qualified'] & ~evidence['unique'], 'ambiguous', 'unknown')
        states = np.where(contradiction, 'contradictory', np.where(supported, 'supported', states))
        points = [{'id': i, 'supported': bool(supported[i]), 'status': str(states[i]),
                   'reference_pixel': self._appearance.pixels[i].tolist(),
                   'scoring_coverage': float(evidence['scoring_coverage'][i]),
                   'unresolved_context': bool(evidence['unresolved_context'][i]),
                   'current_pixel': evidence['pixels'][i].tolist() if image_valid[i] else None,
                   'measured_residual_m': float(residual[i]) if depth_valid[i] else None}
                  for i in range(len(expected))]
        return {**schemas.stamp(schemas.SURFACE, 'support'), 'reference_digest': self.reference_digest,
                'request_sha256': self.identity, 'accepted': bool(supported.all()),
                'reason': ('original_material_supported' if supported.all() else
                           'original_material_appearance_contradiction' if contradiction.any() else
                           'original_material_appearance_unavailable'),
                'point_count': len(points), 'supported_count': int(supported.sum()),
                'contradictory_count': int(contradiction.sum()), 'points': points,
                'transform_reference_to_camera': matrix.tolist(), 'motion_authority': False}

    def _validate_current(self, rgbd, matrix):
        _validate_rgbd(rgbd)
        if (matrix.shape != (4, 4) or not np.isfinite(matrix).all()
                or not np.allclose(matrix[3], [0, 0, 0, 1], atol=1e-9, rtol=0)
                or not np.allclose(matrix[:3, :3].T @ matrix[:3, :3], np.eye(3), atol=1e-7, rtol=0)
                or not np.isclose(np.linalg.det(matrix[:3, :3]), 1., atol=1e-7, rtol=0)):
            raise ValueError('material support requires a finite rigid pose')
        if (rgbd['gray'].shape != self._shape or rgbd['gray'].dtype != np.uint8
                or rgbd['depth_m'].shape != self._shape or rgbd['rays'].shape != (*self._shape, 3)
                or tuple(rgbd['depth_range']) != self._depth_range
                or hashlib.sha256(rgbd['rays'].tobytes()).hexdigest() != self._rays_digest):
            raise ValueError('original material support camera context changed')
