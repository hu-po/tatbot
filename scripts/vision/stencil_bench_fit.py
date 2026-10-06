#!/usr/bin/env python3
"""Fit stencil-bench transfer degradation priors to one photo of a print beside its transfer.

The photo must show the printed stencil and its transfer on skin (or fake skin)
of the same artwork. Both copies are located with the frozen SIFT reference
bank, rectified to page millimetres, and aligned densely to the artwork; the
printed copy is measured with the same pipeline so camera blur cancels in the
line-spread ratio. Output is a JSON fit plus a PNG montage. The values are rough
priors for the bench's degradation model, not a calibration.
"""

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(REPO/"scripts/lib"), str(REPO/"scripts/vision")]
from tatbot_paths import bootstrap  # noqa: E402

bootstrap()
from stencil_features import ReferenceBank  # noqa: E402

PX_PER_MM = 10.0


def region_mask(hsv, select, exclude=None):
    """Convex hull of the largest connected region matching `select`."""
    count, labels, stats, _ = cv2.connectedComponentsWithStats(select.astype(np.uint8))
    if count < 2:
        raise ValueError("region not found in photo")
    largest = (labels == 1+int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))).astype(np.uint8)
    largest = cv2.morphologyEx(largest, cv2.MORPH_CLOSE, np.ones((61, 61), np.uint8))
    contours, _ = cv2.findContours(largest, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    mask = np.zeros(hsv.shape[:2], np.uint8)
    cv2.fillPoly(mask, [cv2.convexHull(max(contours, key=cv2.contourArea))], 1)
    if exclude is not None:
        mask[exclude > 0] = 0
    return mask


def locate(bank, image, mask):
    """SIFT on a crop around the copy (the bank's full-photo downsampling loses detail)."""
    x, y, w, h = cv2.boundingRect(mask)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)[y:y+h, x:x+w]
    inside = mask[y:y+h, x:x+w] > 0
    found, reason = bank.detect(np.where(inside, gray, np.uint8(np.median(gray[inside]))))
    if found is None:
        return None, reason
    offset = np.array([[1, 0, x], [0, 1, y], [0, 0, 1.]])
    return offset @ found["homography"], {"method": "sift", "inliers": int(found["inliers"])}


def _corners(contour):
    hull = cv2.convexHull(contour)
    for epsilon in np.linspace(.01, .1, 30):
        poly = cv2.approxPolyDP(hull, epsilon*cv2.arcLength(hull, True), True)
        if len(poly) == 4:
            poly = poly.reshape(4, 2).astype(np.float64)
            centre = poly.mean(axis=0)
            order = np.argsort(np.arctan2(poly[:, 1]-centre[1], poly[:, 0]-centre[0]))
            return poly[order]  # image-clockwise, starting near the top-left
    return None


def locate_band(image, mask, settings, page_mm, ink):
    """Frame-band corners when SIFT cannot see the copy; the dihedral ambiguity is scored."""
    x, y, w, h = cv2.boundingRect(mask)
    crop = image[y:y+h, x:x+w]
    inside = cv2.erode(mask[y:y+h, x:x+w], np.ones((w//30 | 1,)*2, np.uint8)) > 0
    observed, _, _ = density(crop, kernel=max(3, w//40), valid=inside)
    band = ((observed > .25) & inside).astype(np.uint8)
    close = max(3, w//25) | 1
    band = cv2.morphologyEx(band, cv2.MORPH_CLOSE, np.ones((close, close), np.uint8))
    contours, hierarchy = cv2.findContours(band, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    outer_index = max(range(len(contours)), key=lambda i: cv2.contourArea(contours[i]))
    holes = [i for i in range(len(contours)) if hierarchy[0][i][3] == outer_index]
    outer = _corners(contours[outer_index])
    inner = _corners(contours[max(holes, key=lambda i: cv2.contourArea(contours[i]))]) if holes else None
    if outer is None or inner is None:
        raise ValueError("frame band corners not found")
    m, f = settings["margin_mm"], settings["frame_mm"]
    width, height = page_mm

    def box(inset):
        return np.array([[inset/width, inset/height], [1-inset/width, inset/height],
                         [1-inset/width, 1-inset/height], [inset/width, 1-inset/height]])
    uv = np.vstack([box(m), box(m+f)])
    offset = np.array([[1, 0, x], [0, 1, y], [0, 0, 1.]])
    best, scores = None, {}
    for shift in (0, 2):
        for mirror in (False, True):
            order = [(shift+k) % 4 for k in ((0, 1, 2, 3) if not mirror else (1, 0, 3, 2))]
            pixels = np.vstack([outer[order], inner[order]])
            # The lattice is mirror- and half-turn-symmetric: only the refined fine-scale fit decides.
            homography = refine(image, offset @ cv2.findHomography(uv, pixels)[0], page_mm, ink)
            score = _agreement(image, homography, page_mm, ink, sigma_mm=.3)
            scores[f"rot{shift*90}{'_mirror' if mirror else ''}"] = round(score, 4)
            if best is None or score > best[0]:
                best = (score, homography)
    return best[1], {"method": "frame_band_corners", "dihedral_agreement_refined": scores}


def _agreement(image, homography, page_mm, ink, sigma_mm=.8):
    observed, _, _ = density(rectify(image, homography, page_mm))
    art = cv2.GaussianBlur(ink.astype(np.float32), (0, 0), sigma_mm*PX_PER_MM)
    seen = cv2.GaussianBlur(observed, (0, 0), sigma_mm*PX_PER_MM)
    return float(np.corrcoef(art.ravel(), seen.ravel())[0, 1])


def refine(image, homography, page_mm, ink):
    """ECC homography refinement of the rectified copy against the artwork, coarse to fine."""
    for sigma in (1.2, .6, .3):
        observed, _, _ = density(rectify(image, homography, page_mm))
        art = cv2.GaussianBlur(ink.astype(np.float32), (0, 0), sigma*PX_PER_MM)
        seen = cv2.GaussianBlur(observed, (0, 0), sigma*PX_PER_MM)
        warp = np.eye(3, dtype=np.float32)
        try:
            _, warp = cv2.findTransformECC(art, seen, warp, cv2.MOTION_HOMOGRAPHY,
                                           (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 100, 1e-6), None, 5)
        except cv2.error:
            break
        # seen(warp p) ~ art(p): artwork page pixel p sits at the copy's page pixel warp(p).
        uv_from_page = _uv_from_page(page_mm)
        homography = homography @ uv_from_page @ warp @ np.linalg.inv(uv_from_page)
    return homography


def _uv_from_page(page_mm):
    width, height = round(page_mm[0]*PX_PER_MM), round(page_mm[1]*PX_PER_MM)
    return np.linalg.inv(np.array([[1, 0, -.5], [0, 1, -.5], [0, 0, 1.]]) @ np.diag([width, height, 1.]))


def rectify(image, homography, page_mm):
    width, height = round(page_mm[0]*PX_PER_MM), round(page_mm[1]*PX_PER_MM)
    # uv of page pixel centres: ((x+.5)/W, (y+.5)/H), matching the reference convention.
    return cv2.warpPerspective(image, homography @ _uv_from_page(page_mm), (width, height),
                               flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP, borderValue=0)


def density(page_bgr, kernel=None, valid=None):
    """Per-pixel ink density 1 - I/white in the channel with most contrast inside `valid`."""
    image = page_bgr.astype(np.float32)
    k = (kernel or int(6*PX_PER_MM)) | 1
    white = cv2.GaussianBlur(cv2.dilate(image, np.ones((k, k), np.uint8)), (k, k), 0)
    ratio = np.clip(image/np.maximum(white, 1), 0, 1)
    absorbed = 1-ratio
    valid = (image.max(axis=2) > 8) if valid is None else valid
    channel = int(np.argmax([np.percentile(absorbed[..., c][valid], 97) for c in range(3)]))
    return absorbed[..., channel], channel, white


def artwork_mask(artwork_png, shape):
    art = cv2.imread(str(artwork_png), cv2.IMREAD_GRAYSCALE)
    return cv2.resize(art, (shape[1], shape[0]), interpolation=cv2.INTER_AREA) < 128


def align(ink, observed, band):
    """Dense smooth displacement taking artwork coordinates to the observed copy."""
    spread = cv2.GaussianBlur(ink.astype(np.float32), (0, 0), .06*PX_PER_MM*4)
    source = np.uint8(255*np.clip(spread/max(spread.max(), 1e-6), 0, 1))
    target = np.uint8(255*np.clip(observed/max(np.percentile(observed[band], 99), 1e-6), 0, 1))
    flow = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_MEDIUM).calc(source, target, None)
    # Wobble is smooth: keep only structure wider than the lattice spacing.
    weight = cv2.GaussianBlur(band.astype(np.float32), (0, 0), 4*PX_PER_MM)
    smooth = np.stack([cv2.GaussianBlur(flow[..., c]*band, (0, 0), 4*PX_PER_MM)
                       / np.maximum(weight, 1e-3) for c in range(2)], -1)
    return smooth


def warp_back(image, flow):
    grid_y, grid_x = np.mgrid[0:image.shape[0], 0:image.shape[1]].astype(np.float32)
    return cv2.remap(image, grid_x+flow[..., 0], grid_y+flow[..., 1], cv2.INTER_LINEAR)


def stroke_ridge(ink):
    """Centre-line pixels of thin artwork strokes (filled lenses and dots excluded)."""
    distance = cv2.distanceTransform(ink.astype(np.uint8), cv2.DIST_L2, 5)
    thin = ink & (cv2.dilate(distance, np.ones((int(1.2*PX_PER_MM),)*2, np.uint8)) < .45*PX_PER_MM)
    ridge = thin & (distance >= cv2.dilate(distance, np.ones((3, 3), np.uint8))-1e-3) & (distance > 0)
    return ridge


def line_width_mm(observed, ink, level):
    present = observed > .5*level
    distance = cv2.distanceTransform(present.astype(np.uint8), cv2.DIST_L2, 5)
    ridge = stroke_ridge(ink)
    # Search a small neighbourhood so a residual sub-mm misalignment does not halve the width.
    local = cv2.dilate(distance, np.ones((5, 5), np.uint8))
    values = local[ridge & (local > 0)]
    return float(2*np.median(values)/PX_PER_MM) if len(values) else None, float((local[ridge] > 0).mean())


def measure(observed, ink, band, page_bgr, white, channel):
    level = float(np.percentile(observed[ink & band], 90))
    width, present_fraction = line_width_mm(observed, ink, level)
    near = cv2.dilate(ink.astype(np.uint8), cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (int(1.2*PX_PER_MM),)*2)) > 0
    # Compare 1 mm-local ink mass with what the artwork's strokes would give at this ink level.
    local = cv2.GaussianBlur(observed, (0, 0), .5*PX_PER_MM)
    expected = cv2.GaussianBlur(ink.astype(np.float32), (0, 0), .5*PX_PER_MM)*level
    ridge = stroke_ridge(ink)
    washed = ridge & (local < .35*expected)
    stray = band & ~near & (observed > .5*level)
    far = band & ~cv2.dilate(ink.astype(np.uint8), np.ones((int(2*PX_PER_MM),)*2, np.uint8)).astype(bool)
    core = ink & band & (observed > .6*level)
    skin = np.median(page_bgr[far], axis=0)
    inked = np.median(page_bgr[core], axis=0)
    halves = {}
    for name, cols in (("left", slice(0, ink.shape[1]//2)), ("right", slice(ink.shape[1]//2, None))):
        part = np.zeros_like(ridge)
        part[:, cols] = True
        halves[name] = float(washed[part].sum()/max((ridge & part).sum(), 1))
    return {"ink_level_density": level, "line_width_mm": width,
            "stroke_present_fraction": present_fraction,
            "washed_stroke_fraction": float(washed.sum()/max(ridge.sum(), 1)),
            "washed_stroke_fraction_by_half": halves,
            "stray_ink_fraction_of_band": float(stray.sum()/max(band.sum(), 1)),
            "skin_bgr": [float(v) for v in skin], "ink_bgr": [float(v) for v in inked],
            "transmittance_bgr": [float(v) for v in np.clip(inked/np.maximum(skin, 1), 0, 1)],
            "density_channel_bgr": channel}


def fit_copy(bank, image, mask, artwork_png, settings, page_mm, band_uv):
    size = (round(page_mm[1]*PX_PER_MM), round(page_mm[0]*PX_PER_MM))
    ink = artwork_mask(artwork_png, size)
    homography, method = locate(bank, image, mask)
    sift = homography is not None
    if sift:
        homography = refine(image, homography, page_mm, ink)
    else:
        homography, method = locate_band(image, mask, settings, page_mm, ink)
    page = rectify(image, homography, page_mm)
    observed, channel, white = density(page)
    outer = np.zeros_like(ink)
    margin = band_uv["margin_px"]
    outer[margin:-margin, margin:-margin] = True
    inner = np.zeros_like(ink)
    inner[band_uv["inner"][1]:band_uv["inner"][3], band_uv["inner"][0]:band_uv["inner"][2]] = True
    band = outer & ~inner & (page.max(axis=2) > 8)
    flow = align(ink, observed, band)
    aligned = warp_back(observed, flow)
    aligned_bgr = warp_back(page, flow)
    magnitude = np.linalg.norm(flow, axis=2)[band]/PX_PER_MM
    result_p50 = float(np.median(magnitude))
    detrended = (flow - flow[band].mean(axis=0))
    residual = np.linalg.norm(detrended, axis=2)[band]/PX_PER_MM
    result = measure(aligned, ink, band, aligned_bgr, white, channel)
    result.update(located_by=method, sift_located=sift,
                  mirrored=bool(np.linalg.det(homography[:2, :2]) < 0),
                  artwork_agreement=_agreement(image, homography, page_mm, ink, sigma_mm=.3),
                  photo_px_per_mm=float(np.sqrt(abs(np.linalg.det(homography[:2, :2])/homography[2, 2]**2))
                                        / np.sqrt(page_mm[0]*page_mm[1])),
                  wobble_mm={"p50": result_p50, "mean": float(magnitude.mean()), "p95": float(np.percentile(magnitude, 95)),
                             "rms_detrended": float(np.sqrt(np.mean(residual**2))),
                             "p95_detrended": float(np.percentile(residual, 95))})
    return result, page, aligned, ink, band


def montage(parts, path):
    height = 600
    tiles = []
    for part in parts:
        image = part if part.ndim == 3 else cv2.cvtColor(np.uint8(np.clip(part, 0, 1)*255), cv2.COLOR_GRAY2BGR)
        tiles.append(cv2.resize(image, (round(image.shape[1]*height/image.shape[0]), height)))
    cv2.imwrite(str(path), np.hstack(tiles))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--photo", required=True)
    parser.add_argument("--reference", required=True, help="tracking.json of the printed artwork")
    parser.add_argument("--output", required=True, help="directory for fit.json and fit.png")
    args = parser.parse_args(argv)
    reference = Path(args.reference)
    manifest = json.loads(reference.read_text())
    settings = json.loads(reference.with_name("settings.json").read_text())
    page_mm = manifest["page_mm"]
    image = cv2.imread(args.photo)
    if image is None:
        raise SystemExit(f"cannot read {args.photo}")
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    skin_select = (hsv[..., 0] > 10) & (hsv[..., 0] < 30) & (hsv[..., 1] > 48) & (hsv[..., 1] < 120) & (hsv[..., 2] > 180)
    skin = region_mask(hsv, skin_select)
    paper = region_mask(hsv, (hsv[..., 1] < 30) & (hsv[..., 2] > 170), exclude=skin)
    # An offline appearance measurement of the paper copy: a coded print is matched by SIFT
    # here on purpose, as in the bench's SIFT comparison; live tracking decodes it.
    bank = ReferenceBank([reference], scene=True, allow_coded=True)
    inset = (settings["margin_mm"]+settings["frame_mm"])*PX_PER_MM
    size = (round(page_mm[0]*PX_PER_MM), round(page_mm[1]*PX_PER_MM))
    band_uv = {"margin_px": int(max(settings["margin_mm"]*PX_PER_MM-10, 1)),
               "inner": [int(inset+10), int(inset+10), int(size[0]-inset-10), int(size[1]-inset-10)]}
    art = reference.with_name("stencil.png")
    transfer, t_page, t_aligned, ink, band = fit_copy(bank, image, skin, art, settings, page_mm, band_uv)
    printed, p_page, p_aligned, _, _ = fit_copy(bank, image, paper, art, settings, page_mm, band_uv)
    fit = {"schema": "tatbot.stencil-bench-fit/1", "photo": Path(args.photo).name,
           "reference_pattern_id": manifest["pattern_id"], "seed": manifest["seed"],
           "stroke_mm": settings["stroke_mm"], "transfer": transfer, "print": printed,
           "line_spread_factor": (transfer["line_width_mm"]/printed["line_width_mm"]
                                  if transfer["line_width_mm"] and printed["line_width_mm"] else None),
           "transfer_density_vs_print": transfer["ink_level_density"]/printed["ink_level_density"],
           "assumptions": [
               "Both copies share the photo's blur and exposure, so width and density ratios cancel the camera.",
               "The printed copy is the artwork as printed; its measured width includes toner spread and photo blur.",
               "SIFT homography + 4 mm-smooth dense flow separate projective pose from wobble; the fake-skin pad "
               "may not be flat, so wobble includes any pad curvature.",
               "Washed = thin-stroke centre-line whose 1 mm-local ink mass is under 35% of what the artwork's "
               "strokes give at the copy's ink level.",
               "Stray ink = density over half the ink level more than 0.6 mm from any artwork ink (pooling, smudge)."]}
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    (output/"fit.json").write_text(json.dumps(fit, indent=2)+"\n")
    overlay = cv2.cvtColor(np.uint8(255-255*np.clip(t_aligned/transfer["ink_level_density"], 0, 1)), cv2.COLOR_GRAY2BGR)
    overlay[ink & band] = (overlay[ink & band]*.5 + np.array([0, 0, 255])*.5).astype(np.uint8)
    montage([p_page, t_page, overlay], output/"fit.png")
    print(json.dumps(fit, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
