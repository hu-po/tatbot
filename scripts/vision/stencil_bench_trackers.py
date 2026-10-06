"""Tracker interface for the stencil bench, and the frozen SIFT baseline behind it.

A tracker sees one image and the camera intrinsics, never the scene truth. It
returns a status, the pattern it recognised, and page-UV to pixel
correspondences with optional per-point confidence. A tracker may also return a
dense `model` (UV -> pixel) when it has one; the scorer evaluates it separately.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Protocol

import numpy as np

ACCEPTED, REJECTED, AMBIGUOUS = "accepted", "rejected", "ambiguous"


@dataclass
class Located:
    status: str                      # accepted | rejected | ambiguous
    pattern_id: str | None
    uv: np.ndarray                   # (N, 2) page UV of the recognised pattern
    pixels: np.ndarray               # (N, 2) image pixels
    confidence: np.ndarray | None = None
    reason: str = ""
    model: Callable[[np.ndarray], np.ndarray] | None = None
    extra: dict = field(default_factory=dict)

    @classmethod
    def rejected(cls, reason, status=REJECTED):
        return cls(status, None, np.empty((0, 2)), np.empty((0, 2)), reason=reason)


class Tracker(Protocol):
    name: str

    def prepare(self, artworks) -> None:
        """Build any reference data from the candidate artwork(s) the bench is scoring."""

    def locate(self, image: np.ndarray, intrinsics: dict) -> Located:
        """Find the stencil in one BGR image."""


class SiftBaseline:
    """stencild's acquisition of legacy artwork: StencilScene over a scene-scale ReferenceBank
    (it also matches coded artwork by appearance here, for comparison; stencild decodes it).

    Frozen: this wraps scripts/vision/stencil_scene.py and stencil_features.py
    without changing them. Each image is a fresh first observation (detection),
    so no tracking state carries between scenes.
    """

    name = "sift"

    def __init__(self):
        self.bank = None

    def prepare(self, artworks):
        from stencil_features import ReferenceBank
        references = [a.reference for a in artworks if a.reference is not None]
        if not references:
            raise ValueError("the SIFT baseline needs a tracking.json reference per candidate artwork")
        # The comparison matches coded artwork by appearance on purpose; live tracking never does.
        self.bank = ReferenceBank(references, scene=True, allow_coded=True)

    def locate(self, image, intrinsics):
        import cv2
        from stencil_scene import StencilScene
        cv2.setRNGSeed(0)
        scene = StencilScene([], "bench", bank=self.bank)
        observation = scene.observe(image, 1_000_000_000)
        rows = [row for row in observation["stencils"] if row["image_tracking_valid"]]
        if not rows:
            reasons = {row["reason"] for row in observation["stencils"]}
            status = AMBIGUOUS if "ambiguous_reference" in reasons else REJECTED
            return Located.rejected(",".join(sorted(reasons)), status)
        row = rows[0]
        homography = np.array(row["homography_uv_to_image"])
        uv = np.array([m["reference_uv"] for m in row["landmarks"]], float).reshape(-1, 2)
        pixels = np.array([m["image_px"] for m in row["landmarks"]], float).reshape(-1, 2)

        def model(query):
            points = np.asarray(query, float).reshape(-1, 1, 2)
            return cv2.perspectiveTransform(points, homography).reshape(-1, 2)

        return Located(ACCEPTED, row["pattern_id"], uv, pixels, reason=row["reason"], model=model,
                       extra={"mirrored": row.get("mirrored"), "inliers": row.get("inliers"),
                              "physical_instance_id": row.get("physical_instance_id"),
                              "reprojection_rmse_px": row.get("reprojection_rmse_px")})


class Oracle:
    """Test fixture: reads truth through a side channel the bench sets per scene."""

    name = "oracle"

    def __init__(self, offset_mm=0.0):
        self.truth = None
        self.offset_mm = offset_mm

    def prepare(self, artworks):
        pass

    def locate(self, image, intrinsics):
        scene = self.truth
        if scene is None or scene.transfer is None:
            return Located.rejected("nothing_visible")
        uv, visible = scene.frame_cells()
        if not visible.any():
            return Located.rejected("nothing_visible")
        uv = uv[visible]
        s = scene.truth_skin_mm(uv)+self.offset_mm
        pixels, _ = scene.project(s)
        return Located(ACCEPTED, scene.truth_artwork.pattern_id, uv, pixels, reason="oracle")


def _coded():
    from stencil_coded_tracker import CodedTracker
    return CodedTracker()


def _lightglue():
    from stencil_learned_tracker import LightGlueTracker
    return LightGlueTracker()


TRACKERS = {"sift": SiftBaseline, "coded": _coded, "lightglue": _lightglue}


def make(name):
    if name not in TRACKERS:
        raise ValueError(f"unknown tracker {name!r}; known: {', '.join(sorted(TRACKERS))}")
    return TRACKERS[name]()
