"""One stencil instance: reference acquisition, bounded LK tracking and loss.

Image correspondence never certifies material geometry or robot-frame pose.
The immutable artwork bank is the only source of reacquired UV coordinates.
"""

import time

import cv2
import numpy as np
import stencil_instance
from stencil_features import ReferenceBank, fit
from surface_replay import Settings as FlowSettings
from surface_replay import SparsePatchTracker


def gray_image(image):
    if image.dtype != np.uint8 or image.ndim not in (2, 3):
        raise ValueError("stencil image must be uint8 gray or BGR")
    if min(image.shape[:2]) < 32 or max(image.shape[:2]) > 4096:
        raise ValueError("stencil image dimensions must be 32–4096")
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image


def depth_quality(depth, points):
    if depth is None:
        return {"available": False, "reason": "depth_unavailable"}
    pixels = np.rint(points).astype(int)
    x = np.clip(pixels[:, 0], 0, depth.shape[1]-1)
    y = np.clip(pixels[:, 1], 0, depth.shape[0]-1)
    values = depth[y, x]
    valid = np.isfinite(values) & (values > 0)
    return {"available": True, "sample_count": len(values), "valid_count": int(valid.sum()),
            "valid_fraction": float(valid.mean()) if len(valid) else 0.,
            "median_depth_m": float(np.median(values[valid])) if valid.any() else None,
            "reason": "sample_statistics_only_not_surface_geometry"}


class StencilTracker:
    def __init__(self, references, instance_id, settings=None, *, bank=None, pattern_id=None):
        if not instance_id or len(instance_id) > 160:
            raise ValueError("a bounded physical-instance label is required")
        self.bank = bank if bank is not None else ReferenceBank(references, settings)
        self.settings = self.bank.settings
        self.instance_id = instance_id
        self.pattern_id = pattern_id
        self.ever_acquired = False
        self.active = None
        self.last_timestamp = None
        self.last_signature = None
        self.last_search = None
        self.last_verified = None
        self.core = None
        self.anchor_uv = None
        self.decoded_instance_id = None

    def _flow(self, gray, stamp):
        observation = self.core.step(gray, stamp)
        if observation["status"] == "lost":
            return None
        uv = self.anchor_uv[np.asarray(observation["ids"], int)]
        xy = np.asarray(observation["points_px"])
        result = fit(uv, xy, gray.shape, self.settings)
        if result is not None:
            result.update(pattern_id=self.pattern_id, mirrored=self.active["mirrored"])
        return result

    def _search(self, gray, stamp):
        self.last_search = stamp
        found, reason = self.bank.detect(gray)
        if found is None:
            return None, reason
        if self.pattern_id is not None and found["pattern_id"] != self.pattern_id:
            return None, "different_pattern_visible"
        self.pattern_id = found["pattern_id"]
        self.ever_acquired = True
        self.last_verified = stamp
        self.anchor_uv = found["uv"].copy()
        self.core = SparsePatchTracker((0, 0, gray.shape[1], gray.shape[0]),
            FlowSettings(max_points=self.settings.max_features, min_points=self.settings.min_inliers,
                         max_gap_ms=self.settings.max_gap_ms, fb_limit_px=.75))
        self.core.seed_points(gray, stamp, found["pixels"])
        return found, reason

    def _advance(self, gray, stamp, *, defer_search=False):
        if self.active is None and defer_search:
            return None, "lost", "unchanged_scene_since_reference_search"
        found = self._flow(gray, stamp) if self.active is not None else None
        verify = self.last_verified is None or stamp-self.last_verified >= self.settings.verify_interval_ms*1_000_000
        if found is not None and (not verify or defer_search):
            return found, "tracked", "verification_deferred" if verify else "flow_verified"
        if defer_search:
            return None, "lost", "reference_search_deferred"
        may_search = self.last_search is None or stamp-self.last_search >= self.settings.search_interval_ms*1_000_000
        if not may_search:
            return None, "lost", "reference_search_rate_limited"
        previously_acquired = self.ever_acquired
        acquired, reason = self._search(gray, stamp)
        status = "reacquired" if previously_acquired else "detected"
        return acquired, status if acquired is not None else "lost", reason

    def _observation(self, stamp, status, reason, depth):
        reference = self.bank.references[self.pattern_id] if self.pattern_id else None
        result = {"schema": "tatbot.stencil-observation/1", "capture_timestamp_ns": stamp,
                  "instance_id": self.instance_id, "pattern_id": self.pattern_id,
                  "seed": reference["seed"] if reference else None,
                  "reference_id": reference["reference_id"] if reference else None,
                  "reference_physical_instance_id": reference.get('physical_instance_id') if reference else None,
                  "physical_instance_id": self.decoded_instance_id,
                  "status": status, "reason": reason, "image_tracking_valid": self.active is not None,
                  "geometry_valid": False, "geometry_reason": "surface_model_not_initialized",
                  "motion_authority": False,
                  "physical_instance_identity_verified": self.decoded_instance_id is not None,
                  "image_model": "planar_reference_homography", "landmarks": [],
                  "homography_uv_to_image": None, "page_polygon_px": None}
        if self.active is not None:
            value = self.active
            result.update(homography_uv_to_image=value["homography"].tolist(), page_polygon_px=value["polygon"].tolist(),
                          mirrored=value["mirrored"], inliers=value["inliers"], reference_coverage=value["coverage"],
                          reprojection_rmse_px=value["rmse_px"],
                          landmarks=[{"reference_uv": uv.tolist(), "image_px": xy.tolist()}
                                     for uv, xy in zip(value["uv"], value["pixels"], strict=True)])
        points = self.active["pixels"] if self.active is not None else np.empty((0, 2))
        result["depth_quality"] = depth_quality(depth, points)
        return result

    def _identity(self, gray, reference):
        """(decoded print-instance ID or None, reason) of the tracked page in this frame: the
        printed instance mark read through the current homography. None when this tracker reads
        no identity from the image: an unmarked print, or a coded one matched by appearance
        (the bench's SIFT comparison)."""
        if reference.get('instance_mark') is None:
            return None
        mark = dict(reference['instance_mark'], page_mm=reference['page_mm'])
        return stencil_instance.decode(gray, self.active['homography'], mark)

    def observe(self, image, timestamp_ns, *, depth_m=None, source_id="recording", defer_search=False):
        started = time.perf_counter()
        gray = gray_image(image)
        if depth_m is not None and depth_m.shape != gray.shape:
            raise ValueError("aligned depth shape must match image")
        stamp = int(timestamp_ns)
        if stamp < 0:
            raise ValueError("capture timestamp must be nonnegative")
        signature = (source_id, gray.shape)
        discontinuity = self._discontinuity(stamp, signature)
        if discontinuity:
            self.active = None
            self.last_search = None
        self.decoded_instance_id = None
        if discontinuity == "timestamp_regression":
            result = self._observation(stamp, "lost", discontinuity, depth_m)
            result["processing_ms"] = (time.perf_counter()-started)*1000
            return result
        self.last_timestamp, self.last_signature = stamp, signature
        self.active, status, reason = self._advance(gray, stamp, defer_search=defer_search)
        if self.active is not None:
            reference = self.bank.references[self.pattern_id]
            identity = self._identity(gray, reference)
            if identity is not None:
                decoded, mark_reason = identity
                if decoded == reference['physical_instance_id']:
                    self.decoded_instance_id = decoded
                else:
                    self.active, self.core, self.anchor_uv = None, None, None
                    status = 'lost'
                    reason = ('physical_instance_mismatch' if decoded is not None else mark_reason)
        result = self._observation(stamp, status, reason, depth_m)
        result["input_discontinuity"] = discontinuity
        result["processing_ms"] = (time.perf_counter()-started)*1000
        return result

    def _discontinuity(self, stamp, signature):
        if self.last_signature is not None and signature != self.last_signature:
            return "source_or_profile_changed"
        if self.last_timestamp is not None and stamp <= self.last_timestamp:
            return "timestamp_regression"
        if self.last_timestamp is not None and stamp-self.last_timestamp > self.settings.max_gap_ms*1_000_000:
            return "capture_gap"
        return None
