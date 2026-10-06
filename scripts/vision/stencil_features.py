"""Bounded reference-bank localization; planar image evidence, not a 3D pose."""

from dataclasses import dataclass

import cv2
import numpy as np
import stencil_instance
import stencil_reference  # noqa: E402


@dataclass(frozen=True)
class Settings:
    max_features: int = 2500
    min_inliers: int = 10
    ratio: float = .72
    min_inlier_fraction: float = .5
    min_coverage: float = .06
    max_error_px: float = 2.5
    ambiguity_ratio: float = .75
    search_interval_ms: int = 250
    verify_interval_ms: int = 1500
    max_gap_ms: int = 500


def area(points):
    points = np.asarray(points, np.float32).reshape(-1, 2)
    return float(cv2.contourArea(cv2.convexHull(points))) if len(points) >= 3 else 0.


def projected_page(homography, shape):
    corners = np.array([[0., 0.], [1., 0.], [1., 1.], [0., 1.]])
    homogeneous = np.c_[corners, np.ones(4)] @ homography.T
    denominators = homogeneous[:, 2]
    if np.min(np.abs(denominators)) < 1e-8 or np.any(denominators*denominators[0] <= 0):
        return None
    polygon = (homogeneous[:, :2]/denominators[:, None]).astype(np.float32)
    if not np.isfinite(polygon).all() or not cv2.isContourConvex(polygon):
        return None
    page_area = area(polygon)
    if not 200 <= page_area <= shape[0]*shape[1]*16:
        return None
    if np.max(np.abs(polygon)) > 8*max(shape):
        return None
    return polygon


def fit(uv, pixels, shape, settings):
    if len(uv) < settings.min_inliers:
        return None
    homography, mask = cv2.findHomography(np.asarray(uv, float), np.asarray(pixels, float),
                                 cv2.RANSAC, settings.max_error_px)
    if homography is None or mask is None or not np.isfinite(homography).all():
        return None
    keep = mask.ravel().astype(bool)
    if keep.sum() < max(settings.min_inliers, len(uv)*settings.min_inlier_fraction):
        return None
    polygon = projected_page(homography, shape)
    coverage = area(np.asarray(uv)[keep])
    if polygon is None or coverage < settings.min_coverage:
        return None
    prediction = cv2.perspectiveTransform(np.asarray(uv, float)[None], homography)[0]
    error = np.linalg.norm(prediction-np.asarray(pixels), axis=1)
    if area(np.asarray(pixels)[keep])/area(polygon) < settings.min_coverage*.5:
        return None
    rmse = float(np.sqrt(np.mean(error[keep]**2)))
    return {"homography": homography, "uv": np.asarray(uv)[keep], "pixels": np.asarray(pixels)[keep],
            "polygon": polygon, "inliers": int(keep.sum()), "coverage": coverage, "rmse_px": rmse,
            "score": float(keep.sum()*np.sqrt(coverage)/(1+rmse))}


def distinct_matches(matches, reference, scene):
    """Multiple SIFT orientations and repeated motifs cannot multiply support."""
    accepted, source_cells, target_cells = [], set(), set()
    for match in sorted(matches, key=lambda m: m.distance):
        a = tuple(np.rint(reference[match.queryIdx]/2).astype(int))
        b = tuple(np.rint(scene[match.trainIdx]/2).astype(int))
        if a in source_cells or b in target_cells:
            continue
        source_cells.add(a)
        target_cells.add(b)
        accepted.append(match)
    return accepted


def scene_features(detector, gray):
    """Limit background texture competition while keeping native pixel gates.

    A dyadic image pyramid preserves the same SIFT scale convention between
    acquisition attempts. Coordinates include resize's half-pixel offset;
    RANSAC and the later depth lookup still operate in original pixels.
    """
    image = gray
    while image.size > 2_500_000:
        image = cv2.resize(image, (max(1, image.shape[1]//2), max(1, image.shape[0]//2)),
                           interpolation=cv2.INTER_AREA)
    keys, descriptors = detector.detectAndCompute(image, None)
    if image.shape != gray.shape:
        xscale, yscale = gray.shape[1]/image.shape[1], gray.shape[0]/image.shape[0]
        keys = [cv2.KeyPoint((key.pt[0]+.5)*xscale-.5, (key.pt[1]+.5)*yscale-.5,
                            key.size*max(xscale, yscale), key.angle, key.response, key.octave, key.class_id)
                for key in keys]
    return keys, descriptors


class ReferenceBank:
    """SIFT features of one to eight references. A coded print is decoded (stencil_coded_live),
    never matched by appearance: only the bench's SIFT comparison sets `allow_coded`."""

    def __init__(self, paths, settings=None, *, scene=False, allow_coded=False):
        self.settings = settings or Settings()
        self.allow_coded = allow_coded
        if not 1 <= len(paths) <= 8:
            raise ValueError("provide one to eight stencil references")
        self.detector = cv2.SIFT_create(nfeatures=self.settings.max_features, contrastThreshold=.015)
        self.scene_detector = cv2.SIFT_create(nfeatures=6000, contrastThreshold=.015) if scene else self.detector
        self.scales = (100, 140, 180, 240, 320, 400, 640) if scene else (240, 400, 640)
        self.references = {}
        self.variants = []
        for path in paths:
            self._add(path)

    def _add(self, path):
        manifest, image_path = stencil_reference.load(path)
        pattern = manifest["pattern_id"]
        if pattern in self.references:
            raise ValueError("duplicate stencil pattern in reference bank")
        if stencil_reference.is_coded(manifest) and not self.allow_coded:
            raise ValueError("a coded print is tracked by its code, not by SIFT")
        image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
        expected = manifest["image"]
        if image is None or image.shape != (expected["height"], expected["width"]):
            raise ValueError("reference decoded dimensions disagree with manifest")
        if manifest.get('instance_mark') is not None:
            mark = dict(manifest['instance_mark'], page_mm=manifest['page_mm'])
            homography = np.diag([image.shape[1], image.shape[0], 1.])
            decoded, reason = stencil_instance.decode(image, homography, mark)
            if decoded != manifest['physical_instance_id']:
                raise ValueError(f'reference image does not contain its declared print-instance mark: {reason}')
        self.references[pattern] = manifest
        for edge in self.scales:
            for mirror in (False, True):
                self._variant(image, pattern, edge, mirror)

    def _variant(self, image, pattern, edge, mirror):
        scale = edge/max(image.shape)
        small = cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        if mirror:
            small = cv2.flip(small, 1)
        keys, desc = self.detector.detectAndCompute(small, None)
        if desc is None or len(keys) < self.settings.min_inliers:
            return
        pixels = np.asarray([k.pt for k in keys], np.float32)
        uv = (pixels+.5)/[small.shape[1], small.shape[0]]
        if mirror:
            uv[:, 0] = 1-uv[:, 0]
        self.variants.append({"pattern_id": pattern, "pixels": pixels, "uv": uv,
                              "descriptors": desc, "mirrored": mirror})

    def _candidate(self, variant, keys, matcher, shape):
        raw = matcher.knnMatch(variant["descriptors"], k=2)
        matches = [pair[0] for pair in raw if len(pair) == 2 and pair[0].distance < self.settings.ratio*pair[1].distance]
        pixels = np.asarray([k.pt for k in keys], np.float32)
        matches = distinct_matches(matches, variant["pixels"], pixels)
        uv = np.asarray([variant["uv"][m.queryIdx] for m in matches]).reshape(-1, 2)
        target = np.asarray([pixels[m.trainIdx] for m in matches]).reshape(-1, 2)
        candidate = fit(uv, target, shape, self.settings)
        if candidate is not None:
            candidate.update(pattern_id=variant["pattern_id"], mirrored=variant["mirrored"])
        return candidate

    def _ranked(self, gray):
        # The standalone observer owns this OpenCV process. Make FLANN and
        # RANSAC repeatable across acquisition attempts and replay runs.
        cv2.setRNGSeed(0)
        keys, descriptors = scene_features(self.scene_detector, gray)
        if descriptors is None or len(keys) < self.settings.min_inliers:
            return [], "insufficient_image_features"
        matcher = cv2.FlannBasedMatcher({"algorithm": 1, "trees": 4}, {"checks": 64})
        matcher.add([descriptors])
        matcher.train()
        candidates = {}
        for variant in self.variants:
            candidate = self._candidate(variant, keys, matcher, gray.shape)
            if candidate is None:
                continue
            previous = candidates.get(candidate["pattern_id"])
            if previous is None or candidate["score"] > previous["score"]:
                candidates[candidate["pattern_id"]] = candidate
        ranked = sorted(candidates.values(), key=lambda c: c["score"], reverse=True)
        return ranked, "reference_not_found"

    def detect(self, gray):
        ranked, reason = self._ranked(gray)
        if not ranked:
            return None, reason
        if len(ranked) > 1 and ranked[1]["score"] >= ranked[0]["score"]*self.settings.ambiguity_ratio:
            return None, "ambiguous_reference"
        return ranked[0], "reference_matched"

    def detect_many(self, gray):
        """One instance per pattern; only spatially competing matches conflict."""
        ranked, reason = self._ranked(gray)
        reasons = dict.fromkeys(self.references, reason)
        rejected = set()
        for index, best in enumerate(ranked):
            for other in ranked[index+1:]:
                if not competing_support(best, other):
                    continue
                rejected.add(other["pattern_id"])
                reasons[other["pattern_id"]] = "spatial_reference_conflict"
                if other["score"] >= best["score"]*self.settings.ambiguity_ratio:
                    rejected.add(best["pattern_id"])
                    reasons[best["pattern_id"]] = reasons[other["pattern_id"]] = "ambiguous_reference"
        accepted = {value["pattern_id"]: value for value in ranked if value["pattern_id"] not in rejected}
        return accepted, reasons


def competing_support(first, second):
    a = cv2.convexHull(np.asarray(first["pixels"], np.float32))
    b = cv2.convexHull(np.asarray(second["pixels"], np.float32))
    intersection, _ = cv2.intersectConvexConvex(a, b)
    return intersection > .5*min(area(a), area(b))
