"""Learned-matcher baseline for the stencil bench: ALIKED features and LightGlue (kornia), CPU only.

The reference is the candidate artwork rendered as a spread transfer (dark lines
on light) at two scales that bracket the wrist and overhead views. The query is
the image's red channel, where both inks absorb most, cropped to the skin. The
matches' MAGSAC homography picks the inliers; they are the correspondences, and
the coded tracker's plane-pose model bent by local residuals is the dense model.

Optional: torch and kornia come from the verb's plan (`uv run --with`) only when
`--tracker lightglue` is chosen, and the pretrained weights download into the
torch hub cache on first use.
"""

from __future__ import annotations

import fcntl
from pathlib import Path

import cv2
import numpy as np
from stencil_bench_trackers import ACCEPTED, Located

REFERENCE_PPM = (2.2, 4.0)      # reference scales, pixels per page mm
MAX_KEYPOINTS = 2048
MAX_SIDE = 1400                 # query crop longest side before matching
MIN_INLIERS = 30               # correct train accepts had 38 or more; a wrong one had 20
INLIER_PX = 4.0


class LightGlueTracker:
    name = "lightglue"

    def __init__(self):
        self.references = []

    def prepare(self, artworks):
        import torch
        torch.set_num_threads(1)
        # One download of the pretrained weights however many bench workers start at once.
        lock = Path(torch.hub.get_dir())/"tatbot-stencil-lightglue.lock"
        lock.parent.mkdir(parents=True, exist_ok=True)
        with lock.open("w") as handle:
            fcntl.flock(handle, fcntl.LOCK_EX)
            import kornia.feature as features
            self.extractor = features.ALIKED.from_pretrained("aliked-n16", max_num_keypoints=MAX_KEYPOINTS,
                                                             device=torch.device("cpu")).eval()
            self.matcher = features.LightGlueMatcher("aliked").eval()
        for artwork in artworks:
            for ppm in REFERENCE_PPM:
                gray = _reference_image(artwork, ppm)
                keypoints, descriptors = self._features(gray)
                uv = keypoints/ppm/np.array(artwork.page_mm)
                self.references.append((artwork, gray.shape, keypoints, descriptors, uv))

    def _features(self, gray):
        import torch
        with torch.inference_mode():
            tensor = torch.from_numpy(gray.astype(np.float32)/255.)[None, None].repeat(1, 3, 1, 1)
            found = self.extractor(tensor)[0]
        return found.keypoints.numpy(), found.descriptors

    def _match(self, query, reference):
        import kornia.feature as features
        import torch
        (q_shape, q_kp, q_desc), (_, r_shape, r_kp, r_desc, _) = query, reference
        if len(q_kp) < 8 or len(r_kp) < 8:
            return np.empty((0, 2), int)

        def lafs(kp):
            centers = torch.from_numpy(kp.astype(np.float32))[None]
            return features.laf_from_center_scale_ori(centers, torch.full((1, len(kp), 1, 1), 16.))

        with torch.inference_mode():
            _, indices = self.matcher(q_desc, r_desc, lafs(q_kp), lafs(r_kp), hw1=q_shape, hw2=r_shape)
        return indices.numpy()

    def locate(self, image, intrinsics):
        crop, origin, scale = _query(image)
        if crop is None:
            return Located.rejected("no_skin")
        keypoints, descriptors = self._features(crop)
        query = (crop.shape, keypoints, descriptors)
        best = None
        for reference in self.references:
            pairs = self._match(query, reference)
            if len(pairs) < MIN_INLIERS:
                continue
            pixels = keypoints[pairs[:, 0]]/scale+origin
            uv = reference[4][pairs[:, 1]]
            _, inliers = cv2.findHomography(uv, pixels, cv2.USAC_MAGSAC, INLIER_PX)
            count = int(inliers.sum()) if inliers is not None else 0
            if best is None or count > best[0]:
                best = (count, reference[0], uv[inliers.ravel() > 0] if count else uv[:0],
                        pixels[inliers.ravel() > 0] if count else pixels[:0])
        if best is None or best[0] < MIN_INLIERS:
            return Located.rejected("few_matches")
        count, artwork, uv, pixels = best
        from stencil_coded_tracker import LocalModel
        return Located(ACCEPTED, artwork.pattern_id, uv, pixels, reason="matched",
                       model=LocalModel(uv, pixels, artwork.page_mm, intrinsics), extra={"inliers": count})


def _reference_image(artwork, ppm):
    """The artwork as a transfer: lines spread by the fitted factor, dark on light, at `ppm`."""
    stroke = artwork.meta.get("settings", {}).get("stroke_mm", .45)
    ink = artwork.ink.astype(np.float32)
    grow = int(round(.45*stroke*artwork.ppm))
    if grow > 0:
        ink = cv2.dilate(ink, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2*grow+1,)*2))
    size = (round(artwork.page_mm[0]*ppm), round(artwork.page_mm[1]*ppm))
    ink = cv2.resize(ink, size, interpolation=cv2.INTER_AREA)
    return (255*(1-.7*ink)).astype(np.uint8)


def _query(image):
    """Red channel on the skin's bounding box, contrast-stretched, at most MAX_SIDE: (crop, origin, scale)."""
    from stencil_coded_tracker import skin_mask
    ys, xs = np.nonzero(skin_mask(image, 6.))
    if not len(xs):
        return None, None, None
    x0, y0, x1, y1 = xs.min(), ys.min(), xs.max()+1, ys.max()+1
    red = image[y0:y1, x0:x1, 2]
    scale = min(1., MAX_SIDE/max(red.shape))
    if scale < 1:
        red = cv2.resize(red, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    red = cv2.createCLAHE(2.0, (8, 8)).apply(red)
    return red, np.array([x0, y0], float), scale
