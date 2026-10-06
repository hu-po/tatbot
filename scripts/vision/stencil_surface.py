"""Camera-local RGB-D surface candidates anchored to measured stencil features.

The dense patch is geometric support, not verified material identity in the
blank center. No hole filling, world transform, camera fusion or motion authority.
"""

import json

import cv2
import numpy as np
from rgbd_geometry import camera_rays, color_depth_m, paired_alignment, verify_pair
from stencil_features import area
from stencil_surface_fit import bootstrap_pose, center_pose, choose_model, design
from surface_attachment import measured_xyz


def rgbd_frame(frame):
    depth = frame.get("depth_m")
    if depth is None:
        raise ValueError("depth_unavailable")
    cm, dm = frame["color_metadata"], frame["depth_metadata"]
    if cm["timestamps"]["normalized_unix_ns"] != frame["timestamp_ns"]:
        raise ValueError("rgbd_capture_timestamp_mismatch")
    ca, da = cm.get("attributes", {}), dm.get("attributes", {})
    if da.get("aligned_to") != cm["sensor_name"]:
        raise ValueError("depth_alignment_unverified")
    verify_pair(cm, dm)
    intr = da["intrinsics"]
    intr = json.loads(intr) if isinstance(intr, str) else intr
    if any(intr[key] != dm["profile"][key] for key in ("width", "height")):
        raise ValueError("depth_intrinsics_profile_mismatch")
    rays = camera_rays(json.dumps(intr, sort_keys=True))
    if depth.shape != rays.shape[:2] or depth.shape != frame["image"].shape[:2]:
        raise ValueError("aligned_depth_shape_mismatch")
    color_intr = ca["intrinsics"]
    color_intr = json.loads(color_intr) if isinstance(color_intr, str) else color_intr
    warnings = ["aligned_color_depth_intrinsics_disagree"] if color_intr != intr else []
    alignment = paired_alignment(ca, da, intr)
    depth = color_depth_m(depth, rays, alignment, intr)
    return {"depth_m": depth, "rays": rays, "depth_range": (.03, 2.)}, warnings


def measured_mesh(rgbd, observation, model):
    depth = rgbd["depth_m"]
    step = max(4, int(np.ceil(max(depth.shape)/64)))
    y, x = np.mgrid[0:depth.shape[0]:step, 0:depth.shape[1]:step]
    pixels = np.column_stack((x.ravel(), y.ravel())).astype(float)
    uv = cv2.perspectiveTransform(pixels[None], np.linalg.inv(observation["homography_uv_to_image"]))[0]
    hull = cv2.convexHull(np.asarray([row["image_px"] for row in observation["landmarks"]], np.float32))
    inside = np.array([cv2.pointPolygonTest(hull, tuple(point), False) >= 0 for point in pixels])
    xyz, valid = measured_xyz(rgbd, pixels, .006)
    residual = np.linalg.norm(xyz-design(uv, model["curved"])@model["coeff"], axis=1)
    valid &= inside & (residual <= .005)
    indices = np.full(len(pixels), -1, int)
    indices[valid] = np.arange(valid.sum())
    grid = indices.reshape(x.shape)
    faces = np.concatenate((np.stack((grid[:-1, :-1], grid[1:, :-1], grid[1:, 1:]), -1).reshape(-1, 3),
                            np.stack((grid[:-1, :-1], grid[1:, 1:], grid[:-1, 1:]), -1).reshape(-1, 3)))
    measured = np.isfinite(depth) & (depth > .03) & (depth < 2.)
    complete = cv2.erode(measured.astype(np.uint8), np.ones((step+1, step+1), np.uint8),
                         anchor=(0, 0), borderType=cv2.BORDER_CONSTANT, borderValue=0)
    cells = complete[y[:-1, :-1], x[:-1, :-1]].ravel().astype(bool)
    faces = faces[np.r_[cells, cells]]
    faces = faces[(faces >= 0).all(axis=1)]
    vertices = xyz[valid]
    if len(faces):
        edges = vertices[faces]-vertices[faces[:, [1, 2, 0]]]
        faces = faces[np.max(np.linalg.norm(edges, axis=2), axis=1) < .03]
    return {"vertices_camera_m": vertices, "triangles": faces, "image_px": pixels[valid],
            "reference_uv_hint": uv[valid]}


def _surface_report(uv, xyz, model, warnings):
    hull = cv2.convexHull(uv.astype(np.float32))
    enclosed = cv2.pointPolygonTest(hull, (.5, .5), False) >= 0
    pose = center_pose(model["coeff"]) if enclosed else None
    normal = pose[:3, 2] if pose is not None else None
    if normal is not None and np.dot(normal, pose[:3, 3]) > 0:
        normal = -normal
    return {"status": "candidate", "reason": "supported_local_fit", "candidate_valid": True,
            "model": "quadratic_uv_surface" if model["curved"] else "affine_uv_plane",
            "coefficients_camera_m": model["coeff"].tolist(), "anchor_count": len(uv),
            "reference_coverage": area(uv), "center_supported": enclosed,
            "camera_from_stencil_center": pose.tolist() if pose is not None else None,
            "normal_toward_camera": normal.tolist() if normal is not None else None,
            "loo_residual_p95_m": model["loo_p95_m"],
            "pose_axes": "origin UV(0.5,0.5); X increasing u, Y orthogonalized increasing v, Z=X cross Y",
            "uncertainty": bootstrap_pose(uv, xyz, model, pose) if pose is not None else
                           {"available": False, "reason": "center_not_enclosed_by_observed_anchors"},
            "calibration_warnings": warnings,
            "anchors": [{"reference_uv": a.tolist(), "point_camera_m": b.tolist()}
                        for a, b in zip(uv, xyz, strict=True)]}


def estimate_surface(observation, frame, reference=None):
    base = {"schema": "tatbot.stencil-surface/1", "status": "unavailable", "candidate_valid": False,
            "geometry_valid": False, "absolute_accuracy_verified": False, "motion_authority": False,
            "frame": "camera_optical", "units": "meters", "dense_material_identity_verified": False}
    if not observation["image_tracking_valid"]:
        return dict(base, reason="image_tracking_unavailable"), None
    if frame.get("depth_m") is None and reference is not None and "camera_model" in frame:
        from stencil_pose import estimate_print_pose
        try:
            return dict(base, **estimate_print_pose(observation, frame, reference)), None
        except (ValueError, KeyError, TypeError, cv2.error) as error:
            return dict(base, reason=str(error)), None
    try:
        rgbd, warnings = rgbd_frame(frame)
    except (ValueError, KeyError, ImportError, TypeError) as error:
        return dict(base, reason=str(error)), None
    uv = np.asarray([row["reference_uv"] for row in observation["landmarks"]])
    pixels = np.asarray([row["image_px"] for row in observation["landmarks"]])
    xyz, valid = measured_xyz(rgbd, pixels, .006)
    uv, xyz = uv[valid], xyz[valid]
    base.update(measured_anchor_count=len(uv), requested_anchor_count=len(pixels))
    if len(uv) < 12 or area(uv) < .1:
        return dict(base, reason="insufficient_measured_anchor_support"), None
    model = choose_model(uv, xyz)
    if model is None or model["loo_p95_m"] > .005:
        return dict(base, reason="surface_fit_inconsistent", loo_residual_p95_m=model["loo_p95_m"] if model else None), None
    report = dict(base, **_surface_report(uv, xyz, model, warnings))
    mesh = measured_mesh(rgbd, observation, model)
    report.update(mesh_vertices=len(mesh["vertices_camera_m"]), mesh_triangles=len(mesh["triangles"]))
    return report, mesh
