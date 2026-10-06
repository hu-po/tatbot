"""Calibrated offline RGB-D inputs; generated truth is returned separately.

This module does not own a session or connect to sensors. Synthetic images are
ray intersections with metric material geometry, never 2-D warped RGB-D truth.
The ideal pinhole camera, exact depth, and textures are development controls,
not a sensor noise model or physical qualification.
"""
from __future__ import annotations

import hashlib
import json
from functools import lru_cache
from itertools import islice
from pathlib import Path

import cv2
import numpy as np
from tatbot_digest import sha256_file

PROVENANCE = "synthetic_calibrated_rgbd"


def pose(*, xyz=(0.0, 0.0, 0.0), angles_deg=(0.0, 0.0, 0.0)):
    """Rigid local-to-parent transform, Euler rotations applied X then Y then Z."""
    x, y, z = np.radians(angles_deg)
    rx = np.array([[1, 0, 0], [0, np.cos(x), -np.sin(x)], [0, np.sin(x), np.cos(x)]])
    ry = np.array([[np.cos(y), 0, np.sin(y)], [0, 1, 0], [-np.sin(y), 0, np.cos(y)]])
    rz = np.array([[np.cos(z), -np.sin(z), 0], [np.sin(z), np.cos(z), 0], [0, 0, 1]])
    result = np.eye(4)
    result[:3, :3] = rz @ ry @ rx
    result[:3, 3] = xyz
    return result


def _rigid(value):
    value = np.asarray(value, dtype=float)
    if (value.shape != (4, 4) or not np.isfinite(value).all()
            or not np.allclose(value[3], [0, 0, 0, 1])
            or not np.allclose(value[:3, :3].T @ value[:3, :3], np.eye(3), atol=1e-8)
            or not np.isclose(np.linalg.det(value[:3, :3]), 1)):
        raise ValueError("expected finite rigid 4x4 transform")
    return value


def material_height(x, y, shape="plane", deformation_m=0.0):
    """Graph z(x,y) in material metres; curved control has both principal curvatures."""
    if shape == "plane":
        height, dx, dy = np.zeros_like(x), np.zeros_like(x), np.zeros_like(y)
    elif shape == "curved":
        # Noncylindrical saddle with asymmetric waviness, max ~20 mm in the patch.
        height = 0.9*x*x - 0.5*y*y + 0.004*np.sin(35*x)*np.cos(23*y)
        dx = 1.8*x + 0.14*np.cos(35*x)*np.cos(23*y)
        dy = -y - 0.092*np.sin(35*x)*np.sin(23*y)
    else:
        raise ValueError("shape must be plane or curved")
    # Smooth local displacement preserves material UV/appearance, changes shape.
    bump = deformation_m * np.exp(-((x - 0.025)**2 + (y + 0.012)**2) / 0.0012)
    return height + bump, dx - bump*2*(x-0.025)/0.0012, dy - bump*2*(y+0.012)/0.0012


def _sparse_border_texture(image):
    """Distinct small surrounding marks; the central 130 x 84 mm stays clear."""
    centers = [(x, y) for y in (-.085, .085) for x in (-.10, 0., .10)]
    rng = np.random.default_rng(8361)
    for x, y in centers:
        center = np.array([(x/.34+.5)*1023, (y/.26+.5)*1023]).astype(int)
        for _ in range(9):
            offsets = rng.integers(-35, 36, (3, 2)).astype(np.int32)
            cv2.polylines(image, [center+offsets], False, int(rng.integers(15, 110)),
                          int(rng.integers(2, 5)), cv2.LINE_AA)
        cv2.circle(image, tuple(center+rng.integers(-15, 16, 2)), 5, 25, -1)


@lru_cache(maxsize=5)
def _texture(appearance):
    if appearance not in ("distinctive", "alternate", "repeating", "smooth", "sparse-border"):
        raise ValueError("unknown material appearance")
    image = np.full((1024, 1024), 215, np.uint8)
    if appearance in ("distinctive", "alternate"):
        rng = np.random.default_rng(2917 if appearance == "distinctive" else 7149)
        for _ in range(650):
            p = tuple(int(v) for v in rng.integers(0, 1024, 2))
            cv2.circle(image, p, int(rng.integers(2, 11)), int(rng.integers(15, 180)), -1)
        for _ in range(90):
            p = rng.integers(10, 1014, (3, 2)).astype(np.int32)
            cv2.polylines(image, [p], False, int(rng.integers(20, 160)), 2, cv2.LINE_AA)
    elif appearance == "sparse-border":
        _sparse_border_texture(image)
    elif appearance == "repeating":
        # Identical cells; boundary is deliberately outside the fixed reference ROI.
        for p in range(0, 1024, 48):
            cv2.line(image, (p, 0), (p, 1023), 35, 3)
            cv2.line(image, (0, p), (1023, p), 35, 3)
    image.setflags(write=False)
    return image


def render_fixture(*, shape="plane", world_from_material=None, world_from_camera=None,
                   appearance="distinctive", deformation_m=0.0, occlusion_fraction=0.0,
                   dark=False, blur_sigma=0.0, depth_hole_fraction=0.0,
                   stamp=1_000_000_000, sequence=0, width=320, height=240, focal_px=300.0):
    """Return (observer frame, withheld truth) with a static image-coordinate ROI.

    Frame has the observer's ``rgbd`` interface plus BGR/Z16/metadata for the
    existing dense benchmark. Ground-truth transforms, UV, and visibility are
    only in the second return value; never pass it to the observer. Surface
    shape and motion do not alter the estimator ROI, which is the whole image.
    The finite 0.34 x 0.26 m material patch exceeds the default camera view.
    """
    if not (8 <= width <= 1280 and 8 <= height <= 960 and 0 < focal_px < 10000):
        raise ValueError("invalid bounded camera geometry")
    if not all(np.isfinite(v) for v in (deformation_m, occlusion_fraction, blur_sigma,
                                        depth_hole_fraction, focal_px)):
        raise ValueError("non-finite rendering option")
    if not (0 <= occlusion_fraction <= 1 and 0 <= depth_hole_fraction <= 1
            and 0 <= blur_sigma <= 20 and abs(deformation_m) <= 0.04):
        raise ValueError("rendering option outside bounded range")
    wm = _rigid(pose(xyz=(0, 0, 0.32)) if world_from_material is None else world_from_material)
    wc = _rigid(np.eye(4) if world_from_camera is None else world_from_camera)
    mc = np.linalg.inv(wm) @ wc
    v, u = np.mgrid[:height, :width]
    rays = np.stack(((u-(width-1)/2)/focal_px, (v-(height-1)/2)/focal_px,
                     np.ones_like(u)), axis=-1)
    direction = rays @ mc[:3, :3].T
    origin = mc[:3, 3]
    t = np.full((height, width), 0.32)
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        for _ in range(15):
            points = origin + t[..., None]*direction
            z, dx, dy = material_height(points[..., 0], points[..., 1], shape, deformation_m)
            derivative = direction[..., 2] - dx*direction[..., 0] - dy*direction[..., 1]
            t -= np.clip((points[..., 2]-z)/derivative, -0.1, 0.1)
    points = origin + t[..., None]*direction
    z, _, _ = material_height(points[..., 0], points[..., 1], shape, deformation_m)
    valid = (np.isfinite(t) & (t > 0.03) & (t < 2.0)
             & (np.abs(points[..., 2]-z) < 1e-8)
             & (np.abs(points[..., 0]) < 0.17) & (np.abs(points[..., 1]) < 0.13))
    # UV lives on the material and is unchanged by pose or deformation.
    map_x = ((points[..., 0]/0.34+0.5)*1023).astype(np.float32)
    map_y = ((points[..., 1]/0.26+0.5)*1023).astype(np.float32)
    gray = cv2.remap(_texture(appearance), map_x, map_y, cv2.INTER_LINEAR,
                     borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    gray[~valid] = 0
    depth = np.where(valid, np.rint(t / 0.0001), 0).astype(np.uint16)
    if occlusion_fraction:
        cutoff = round(width*occlusion_fraction)
        gray[:, :cutoff] = 105
        depth[:, :cutoff] = 2000  # Foreground occluder at camera Z = 0.2 m.
    if dark:
        gray = np.zeros_like(gray)
    if blur_sigma:
        gray = cv2.GaussianBlur(gray, (0, 0), blur_sigma)
    if depth_hole_fraction:
        holes = np.random.default_rng(82).random(depth.shape) < depth_hole_fraction
        depth[holes] = 0
    intrinsics = {"schema": "tatbot.camera-intrinsics/1", "width": width, "height": height,
                  "fx": focal_px, "fy": focal_px, "ppx": (width-1)/2, "ppy": (height-1)/2,
                  "distortion_model": "None", "distortion_coefficients": [0.0]*5}
    calibration = f"synthetic-pinhole-{width}x{height}-{focal_px:g}"
    frame = {"stamp": int(stamp), "capture_timestamp_ns": int(stamp), "sequence": int(sequence),
             "evidence_kind": "synthetic-render", "profile_id": calibration,
             "calibration_id": calibration, "producer_id": "synthetic-v1",
             "provenance": PROVENANCE,
             "rgbd": {"gray": gray, "depth_m": depth.astype(float)*0.0001,
                      "rays": rays, "depth_range": (0.03, 2.0)},
             "bgr": cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR), "depth": depth,
             "metadata": {"profile": {"width": width, "height": height, "format": "z16"},
                          "calibration_id": calibration,
                          "attributes": {"intrinsics": intrinsics, "depth_units_m": "0.0001"}}}
    probe_y, probe_x = np.mgrid[-0.065:0.066:0.013, -0.09:0.091:0.015]
    probe_z, _, _ = material_height(probe_x, probe_y, shape, deformation_m)
    probes = np.stack((probe_x, probe_y, probe_z), axis=-1).reshape(-1, 3)
    truth = {"camera_from_material": np.linalg.inv(wc) @ wm, "world_from_material": wm.copy(),
             "world_from_camera": wc.copy(), "material_points": probes,
             "pixel_material_points": points, "surface_visible": valid,
             "shape": shape, "deformation_m": deformation_m, "appearance": appearance,
             "ground_truth_kind": "generated_metric_ray_intersection"}
    return frame, truth


def _recording_rays(intrinsics_json):
    from rgbd_geometry import camera_rays
    return camera_rays(intrinsics_json)


def _recording_entries(path):
    if path.stat().st_size > 64*1024*1024:
        raise ValueError("oversized recording index")
    with path.open() as stream:
        for _ in range(100000):
            line = stream.readline(65537)
            if not line:
                return
            if len(line) > 65536:
                raise ValueError("oversized recording metadata line")
            yield json.loads(line)
        raise ValueError("recording index exceeds row budget")


def _active_intrinsics(metadata):
    intr = metadata.get("attributes", {}).get("intrinsics")
    intr = json.loads(intr) if isinstance(intr, str) else intr
    if not isinstance(intr, dict) or any(intr[k] != metadata["profile"][k] for k in ("width", "height")):
        raise ValueError("active intrinsics/profile mismatch")
    return intr


def _recorded_profile(dm, cm, sensor):
    if dm["sensor_name"] != sensor+"_depth" or cm["sensor_name"] != sensor+"_color":
        raise ValueError("sensor identity mismatch")
    if dm.get("attributes", {}).get("aligned_to") != sensor+"_color":
        raise ValueError("depth is not explicitly aligned to selected color")
    if dm["profile"]["format"] != "z16":
        raise ValueError("depth stream is not Z16")
    active = [_active_intrinsics(metadata) for metadata in (dm, cm)]
    if active[0] != active[1]:
        raise ValueError("aligned intrinsics disagree")
    units = float(dm["attributes"]["depth_units_m"])
    if not np.isfinite(units) or not 0 < units <= .01:
        raise ValueError("invalid measured depth units")
    if cm.get("calibration_id") != dm.get("calibration_id"):
        raise ValueError("aligned calibration IDs disagree")
    signature = {"profiles": [dm["profile"], cm["profile"]], "intrinsics": active,
                 "units_m": units, "calibration_ids": [dm.get("calibration_id"), cm.get("calibration_id")],
                 'native_alignment': _recorded_alignment(dm, cm, active[0])}
    return hashlib.sha256(json.dumps(signature, sort_keys=True).encode()).hexdigest(), units, active[0]


def _recorded_alignment(dm, cm, intrinsics):
    from rgbd_geometry import paired_alignment
    ca, da = cm['attributes'], dm['attributes']
    if 'alignment_calibration' not in ca and 'alignment_calibration' not in da:
        return None  # Historical replay preserves its original, unqualified convention.
    return paired_alignment(ca, da, intrinsics)


def _capture_flags(dm, cm, previous_drops):
    drops = tuple(int(m.get("dropped_before", 0)) for m in (dm, cm))
    allowed = {"real_sense_global": "timestamp_domain=Global Time", "host_unix": "timestamp_domain=System Time"}
    warnings = [flag for m in (dm, cm) for flag in m.get("flags", [])
                if flag != allowed.get(m["timestamps"].get("source_domain"))]
    reason = None
    if previous_drops is not None and drops != previous_drops:
        reason = "capture_drop_counter_changed"
    elif warnings:
        reason = "flagged_capture"
    return drops, reason


def _bounded_payload(directory, entry):
    from visiond_wire import read_evidence_frame
    payload = (directory/entry["payload_file"]).resolve()
    if not payload.is_relative_to(directory.resolve()) or payload.stat().st_size > 8*1024*1024:
        raise ValueError("payload escapes recording or exceeds size budget")
    return read_evidence_frame(directory, entry)


def _recorded_pixels(depth_dir, color_dir, depth, color, units, intrinsics, roi, depth_range):
    rays = _recording_rays(json.dumps(intrinsics, sort_keys=True))
    raw = _bounded_payload(depth_dir, depth)["depth"]
    bgr = _bounded_payload(color_dir, color)["image"]
    if raw.shape != rays.shape[:2] or bgr.shape != (*raw.shape, 3):
        raise ValueError("aligned pixel geometry mismatch")
    depth_m = raw.astype(float)*units
    depth_m[raw == 65535] = 0
    model = _recorded_alignment(depth['metadata'], color['metadata'], intrinsics)
    if model is not None:
        from rgbd_geometry import color_depth_m
        depth_m = color_depth_m(depth_m, rays, model, intrinsics)
    h, w = raw.shape
    selected_roi = (0, 0, w, h) if roi is None else tuple(roi)
    x, y, rw, rh = selected_roi
    if min(x, y) < 0 or min(rw, rh) <= 0 or x+rw > w or y+rh > h:
        raise ValueError("ROI outside aligned image")
    return {"rgbd": {"gray": cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY), "depth_m": depth_m,
                     "rays": rays, "depth_range": depth_range}, "roi": selected_roi,
            "bgr": bgr, "depth": raw, "metadata": depth["metadata"]}


def _decode_recorded_pair(frame, depth, color, sensor, dirs, roi, depth_range, previous_drops):
    drops = previous_drops
    try:
        dm, cm = depth["metadata"], color["metadata"]
        frame["provenance"].update(color_sha256=color["sha256"], color_capture_timestamp_ns=cm["timestamps"]["normalized_unix_ns"],
                                   color_sequence=cm["sequence"])
        profile_id, units, intrinsics = _recorded_profile(dm, cm, sensor)
        frame.update(profile_id=profile_id, calibration_id=dm.get("calibration_id"))
        drops, reason = _capture_flags(dm, cm, previous_drops)
        frame["provenance"]["owner_cumulative_drops"] = list(drops)
        if reason:
            raise ValueError(reason)
        frame.update(_recorded_pixels(*dirs, depth, color, units, intrinsics, roi, depth_range))
    except (OSError, ValueError, KeyError, TypeError, ImportError, RuntimeError) as error:
        frame["invalid"] = f"recording_frame_invalid: {error}"
    return frame, drops


def _validate_recording_request(sensor, max_frames, depth_range, pair_ms):
    if (not isinstance(max_frames, int) or not 1 <= max_frames <= 10000
            or not isinstance(sensor, str) or not sensor or Path(sensor).name != sensor
            or sensor in (".", "..") or not np.isfinite(pair_ms) or not 0 <= pair_ms <= 100):
        raise ValueError("invalid bounded recording request")
    if (len(depth_range) != 2 or not np.isfinite(depth_range).all()
            or not 0 <= depth_range[0] < depth_range[1] <= 10):
        raise ValueError("invalid recording depth range")


def _recording_stamp(entry):
    stamp = entry["metadata"]["timestamps"]["normalized_unix_ns"]
    if not isinstance(stamp, int) or isinstance(stamp, bool) or stamp <= 0:
        raise ValueError("invalid recorded capture timestamp")
    return stamp


def recorded_frames(root, sensor="realsense1", *, max_frames=300, roi=None,
                    depth_range=(0.03, 2.0), pair_ms=10.0):
    """Bounded measured RGB-D adapter, using the existing evidence decoder.

    Missing producer-build SHA and calibration ID remain unknown. The producer
    identifier denotes this immutable *replay manifest*, never a guessed sensor
    process. Times and sequences retain the original depth capture values.
    Malformed/missing evidence yields a visible invalid frame and stops; corrupt
    payloads/flagged captures yield invalid frames without changing the reference.
    No camera-to-robot pose, ground truth, or physical authorization is supplied.
    """
    _validate_recording_request(sensor, max_frames, depth_range, pair_ms)
    root = Path(root)
    color_dir, depth_dir = root/(sensor+"_color"), root/(sensor+"_depth")
    indexes = [color_dir/"frames.jsonl", depth_dir/"frames.jsonl"]
    base = {"evidence_kind": "recorded-rgbd", "calibration_id": None,
            "capture_timestamp_ns": None, "stamp": None, "sequence": None,
            "producer_id": None, "profile_id": None,
            "provenance": {"recording_path": str(root.resolve()), "producer_build_sha": None,
                           "robot_camera_calibration": None, "physical_ground_truth": None,
                           "time_basis": "original_recorded_capture_clock"}}
    try:
        if any(path.stat().st_size > 64*1024*1024 for path in indexes):
            raise ValueError("oversized recording index")
        hashes = [sha256_file(p) for p in indexes]
        base["producer_id"] = "recording-replay:" + hashlib.sha256("".join(hashes).encode()).hexdigest()
        base["provenance"]["index_sha256"] = dict(zip(("color", "depth"), hashes, strict=True))
        colors = iter(_recording_entries(indexes[0]))
        color = next(colors, None)
        previous_drops = None
        for depth in islice(_recording_entries(indexes[1]), max_frames):
            stamp = _recording_stamp(depth)
            frame = base | {"capture_timestamp_ns": stamp, "stamp": stamp, "sequence": depth["metadata"]["sequence"]}
            frame["provenance"] = base["provenance"] | {"depth_sha256": depth["sha256"]}
            while color is not None and _recording_stamp(color) < stamp-pair_ms*1e6:
                color = next(colors, None)
            if color is None or abs(_recording_stamp(color)-stamp) > pair_ms*1e6:
                yield frame | {"invalid": "unpaired_rgbd"}
                continue
            frame, previous_drops = _decode_recorded_pair(
                frame, depth, color, sensor, (depth_dir, color_dir), roi, depth_range, previous_drops)
            yield frame
            color = next(colors, None)
    except (OSError, ValueError, KeyError, TypeError) as error:
        yield base | {"invalid": f"recording_manifest_invalid: {error}"}
