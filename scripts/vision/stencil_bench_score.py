"""Stencil bench scoring: millimetres on the skin against the transferred geometry.

Per scene, each correspondence (uv, pixel) is scored by back-projecting the
pixel onto the true skin surface and measuring how far it lands from where
page point `uv` was actually transferred. Errors are surface millimetres
(unrolled arc length on a cylinder). Aggregates, curves and subtlety cost make
the scorecard; the contact sheet shows the worst scenes.
"""

from __future__ import annotations

import math

import cv2
import numpy as np

SUCCESS_P50_MM = 1.0     # accepted, right pattern, median point error at most this
GROSS_P50_MM = 5.0       # accepted with a worse median is a false accept (wrong place)
LOCALIZED_MM = 1.0       # a frame cell is localized by a correspondence this close
CURVES = {
    "washoff": ("washoff_measured", [0, .15, .3, .45, .65]),
    "radius_mm": ("radius_mm", ["flat", 30, 40, 50, 60.01]),
    "color": ("color", None),
    "camera": ("role", None),
    "px_per_mm": ("px_per_mm", [0, 2, 3, 5, 100]),
}


def score_scene(scene, located, elapsed_ms):
    """One scene's record: truth parameters, tracker outcome and errors."""
    p = scene.params
    record = {"index": p["index"], "bank": p["bank"], "kind": p["kind"], "camera": p["camera"],
              "role": p["role"], "radius_mm": p["surface"]["radius_mm"], "color": p["transfer"]["color"],
              "mirrored": p["transfer"]["mirrored"], "skin_rgb": p["transfer"]["skin_rgb"],
              **scene.summary(), "status": located.status, "reason": located.reason,
              "pattern_id": located.pattern_id, "truth_pattern_id": scene.truth_artwork.pattern_id,
              "processing_ms": round(elapsed_ms, 2), "points": int(len(located.uv))}
    cells_uv, cells_visible = scene.frame_cells()
    record["visible_frame_fraction"] = float(cells_visible.mean()) if len(cells_visible) else 0.
    errors = point_errors(scene, located)
    record["point_errors_mm"] = errors
    accepted = located.status == "accepted"
    correct_id = accepted and scene.transfer is not None and located.pattern_id == scene.truth_artwork.pattern_id
    finite = np.array([e for e in errors if e is not None and math.isfinite(e)])
    p50 = float(np.median(finite)) if len(finite) else math.inf
    record.update(p50_mm=None if not math.isfinite(p50) else round(p50, 4),
                  p95_mm=None if not len(finite) else round(float(np.percentile(finite, 95)), 4),
                  within_1mm=None if not errors else round(float(np.mean([e is not None and e <= 1 for e in errors])), 4))
    record["correct_id"] = bool(correct_id)
    record["false_accept"] = bool(accepted and (not correct_id or p50 > GROSS_P50_MM))
    record["success"] = bool(correct_id and p50 <= SUCCESS_P50_MM)
    record["localized_fraction"] = localized_fraction(scene, located, errors, cells_uv, cells_visible) if correct_id else 0.
    if correct_id and located.model is not None:
        record.update(model_errors(scene, located.model, cells_uv, cells_visible))
    return record


def point_errors(scene, located):
    """Surface-mm error per correspondence; None when the pixel sees no skin."""
    if scene.transfer is None or located.status != "accepted" or not len(located.uv):
        return []
    if located.pattern_id != scene.truth_artwork.pattern_id:
        return []
    seen = scene.backproject(np.asarray(located.pixels, float))
    truth = scene.truth_skin_mm(located.uv)
    error = np.linalg.norm(seen-truth, axis=1)
    return [None if not math.isfinite(e) else round(float(e), 4) for e in error]


def localized_fraction(scene, located, errors, cells_uv, cells_visible):
    """Visible frame cells holding at least one correspondence within LOCALIZED_MM."""
    if not cells_visible.any() or not errors:
        return 0.
    good = np.array([e is not None and e <= LOCALIZED_MM for e in errors])
    uv = np.asarray(located.uv, float)[good]
    if not len(uv):
        return 0.
    page = np.array(scene.truth_artwork.page_mm)
    pitch = 3.5
    cells = cells_uv[cells_visible]*page
    hits = np.floor(uv*page/pitch).astype(int)
    keys = {tuple(k) for k in hits}
    covered = [tuple(k) in keys for k in np.floor(cells/pitch).astype(int)]
    return round(float(np.mean(covered)), 4)


def model_errors(scene, model, cells_uv, cells_visible):
    """The tracker's dense model over visible frame cells, in surface mm."""
    if not cells_visible.any():
        return {}
    uv = cells_uv[cells_visible]
    seen = scene.backproject(model(uv))
    error = np.linalg.norm(seen-scene.truth_skin_mm(uv), axis=1)
    finite = error[np.isfinite(error)]
    if not len(finite):
        return {"model_p50_mm": None, "model_p95_mm": None, "model_localized_fraction": 0.}
    return {"model_p50_mm": round(float(np.median(finite)), 4),
            "model_p95_mm": round(float(np.percentile(finite, 95)), 4),
            "model_localized_fraction": round(float(np.mean(np.isfinite(error) & (error <= LOCALIZED_MM))), 4)}


def _stats(rows):
    positives = [r for r in rows if r["kind"] == "positive"]
    negatives = [r for r in rows if r["kind"] != "positive"]
    pooled = [e for r in positives if r["correct_id"] for e in r["point_errors_mm"] if e is not None]
    accepted = [r for r in positives if r["status"] == "accepted"]

    def rate(items, key):
        return round(float(np.mean([r[key] for r in items])), 4) if items else None

    def pct(values, q):
        return round(float(np.percentile(values, q)), 4) if len(values) else None

    return {"scenes": len(rows), "positives": len(positives), "negatives": len(negatives),
            "success_rate": rate(positives, "success"),
            "accept_rate": round(len(accepted)/len(positives), 4) if positives else None,
            "false_accept_rate": rate(rows, "false_accept"),
            "false_accepts": sum(r["false_accept"] for r in rows),
            "negative_accepts": sum(r["status"] == "accepted" for r in negatives),
            "ambiguous_rate": round(float(np.mean([r["status"] == "ambiguous" for r in rows])), 4) if rows else None,
            "point_error_mm": {"p50": pct(pooled, 50), "p95": pct(pooled, 95), "n": len(pooled)},
            "scene_p50_mm_median": pct([r["p50_mm"] for r in positives if r["correct_id"] and r["p50_mm"] is not None], 50),
            "localized_fraction_mean": rate(positives, "localized_fraction"),
            "model_error_mm": {
                "p50_median": pct([r["model_p50_mm"] for r in positives if r.get("model_p50_mm") is not None], 50),
                "p95_median": pct([r["model_p95_mm"] for r in positives if r.get("model_p95_mm") is not None], 50)},
            "model_localized_fraction_mean": (
                round(float(np.mean([r.get("model_localized_fraction", 0.) for r in positives])), 4) if positives else None),
            "processing_ms": {"p50": pct([r["processing_ms"] for r in rows], 50),
                              "p95": pct([r["processing_ms"] for r in rows], 95)}}


def _bin(value, edges):
    if edges is None:
        return str(value)
    if edges[0] == "flat":
        if value is None:
            return "flat"
        edges = edges[1:]
    for lo, hi in zip(edges[:-1], edges[1:], strict=True):
        if lo <= value < hi:
            return f"{lo:g}-{hi:g}"
    return f">={edges[-1]:g}"


def curves(rows):
    out = {}
    for name, (key, edges) in CURVES.items():
        groups = {}
        for row in rows:
            if row["kind"] != "positive" or row.get(key, "missing") == "missing":
                continue
            groups.setdefault(_bin(row[key], edges), []).append(row)
        out[name] = {label: {k: v for k, v in _stats(group).items()
                             if k in ("positives", "success_rate", "accept_rate", "false_accepts",
                                      "point_error_mm", "localized_fraction_mean", "model_error_mm")}
                     for label, group in sorted(groups.items())}
    return out


def subtlety(artwork):
    """Ink cost of the printed artwork: black fraction, frame width, largest solid blob."""
    ink = artwork.ink.astype(np.uint8)
    ppm = artwork.ppm
    area_mm2 = 1/ppm**2
    w, h = artwork.page_mm
    ys, xs = np.mgrid[0:ink.shape[0], 0:ink.shape[1]]
    band = artwork.in_frame(np.c_[(xs.ravel()+.5)/ppm, (ys.ravel()+.5)/ppm]).reshape(ink.shape)
    # Solid = survives an opening wider than a line: strokes vanish, fills and grids stay.
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (int(1.0*ppm) | 1,)*2)
    solid = cv2.morphologyEx(ink, cv2.MORPH_OPEN, kernel)
    count, _, stats, _ = cv2.connectedComponentsWithStats(solid)
    largest = float(stats[1:, cv2.CC_STAT_AREA].max()*area_mm2) if count > 1 else 0.
    rows = np.flatnonzero(ink.any(axis=1))
    cols = np.flatnonzero(ink.any(axis=0))
    return {"black_fraction_page": round(float(ink.mean()), 5),
            "black_fraction_frame": round(float(ink[band].mean()), 5) if band.any() else None,
            "black_area_mm2": round(float(ink.sum()*area_mm2), 1),
            "frame_mm": artwork.frame_mm, "margin_mm": artwork.margin_mm,
            "ink_extent_mm": [round(float(cols[0]/ppm), 2), round(float(rows[0]/ppm), 2),
                              round(float(w-(cols[-1]+1)/ppm), 2), round(float(h-(rows[-1]+1)/ppm), 2)]
            if len(rows) else None,
            "solid_fraction_of_ink": round(float(solid.sum()/max(ink.sum(), 1)), 4),
            "largest_solid_blob_mm2": round(largest, 2)}


def scorecard(rows, meta, artwork):
    return {"schema": "tatbot.stencil-bench-scorecard/1", **meta,
            "thresholds": {"success_p50_mm": SUCCESS_P50_MM, "gross_p50_mm": GROSS_P50_MM,
                           "localized_mm": LOCALIZED_MM},
            "overall": _stats(rows),
            "by_kind": {kind: _stats([r for r in rows if r["kind"] == kind])
                        for kind in sorted({r["kind"] for r in rows})},
            "curves": curves(rows), "subtlety": subtlety(artwork),
            "reasons": _reasons(rows)}


def _reasons(rows):
    counts = {}
    for row in rows:
        key = f'{row["kind"]}:{row["status"]}:{row["reason"]}'
        counts[key] = counts.get(key, 0)+1
    return dict(sorted(counts.items(), key=lambda item: -item[1]))


def badness(row):
    """Sort key: false accepts first, then misses on well-visible scenes, then large errors."""
    if row["false_accept"]:
        return (0, -(row["p50_mm"] or 99))
    if row["kind"] == "positive" and not row["success"]:
        return (1, -row["visible_frame_fraction"])
    return (2, -(row["p95_mm"] or 0))


def annotate(image, scene, located, record, width=480):
    """Thumbnail: truth frame outline (green), correspondences coloured by error, caption."""
    view = image.copy()
    if scene.transfer is not None:
        art = scene.truth_artwork
        w, h = art.page_mm
        for inset in (art.margin_mm, art.margin_mm+art.frame_mm):
            edge = np.linspace(0, 1, 60)
            loop = np.concatenate([np.c_[inset+(w-2*inset)*edge, np.full(60, inset)],
                                   np.c_[np.full(60, w-inset), inset+(h-2*inset)*edge],
                                   np.c_[w-inset-(w-2*inset)*edge, np.full(60, h-inset)],
                                   np.c_[np.full(60, inset), h-inset-(h-2*inset)*edge]])/np.array([w, h])
            pixels, visible = scene.truth_pixels(loop)
            for a, b, va, vb in zip(pixels[:-1], pixels[1:], visible[:-1], visible[1:], strict=True):
                if va and vb:
                    cv2.line(view, tuple(int(v) for v in a), tuple(int(v) for v in b), (0, 200, 0), 2)
    errors = record["point_errors_mm"] or [None]*len(located.pixels)
    for (x, y), error in zip(located.pixels, errors, strict=False):
        colour = (0, 0, 255) if error is None or error > 3 else (0, 220, 255) if error > 1 else (0, 255, 0)
        if math.isfinite(x) and math.isfinite(y):
            cv2.circle(view, (int(x), int(y)), max(2, image.shape[1]//300), colour, -1)
    scale = width/view.shape[1]
    thumb = cv2.resize(view, (width, int(view.shape[0]*scale)), interpolation=cv2.INTER_AREA)
    caption = [f'#{record["index"]} {record["kind"]} {record["camera"]} '
               f'{"flat" if record["radius_mm"] is None else "R%.0f" % record["radius_mm"]} {record["color"]}',
               f'wash {record.get("washoff_measured") or 0:.2f} {record["px_per_mm"]:.1f}px/mm '
               f'{record["status"]} p50 {record["p50_mm"]} vis {record["visible_frame_fraction"]:.2f}']
    bar = np.full((40, width, 3), 20, np.uint8)
    for k, text in enumerate(caption):
        cv2.putText(bar, text, (4, 16+18*k), cv2.FONT_HERSHEY_SIMPLEX, .42, (240, 240, 240), 1, cv2.LINE_AA)
    thumb = np.vstack([thumb, bar])
    return cv2.copyMakeBorder(thumb, 0, max(0, int(width*.75)+40-thumb.shape[0]), 0, 0, cv2.BORDER_CONSTANT)[:int(width*.75)+40]


def contact_sheet(thumbs, path, columns=4):
    if not thumbs:
        return
    blank = np.zeros_like(thumbs[0])
    while len(thumbs) % columns:
        thumbs.append(blank)
    rows = [np.hstack(thumbs[k:k+columns]) for k in range(0, len(thumbs), columns)]
    cv2.imwrite(str(path), np.vstack(rows), [cv2.IMWRITE_JPEG_QUALITY, 88] if str(path).endswith(".jpg") else [])
