"""Build a private scene bank from recorded RGB-D frames and explicit image masks, without hardware."""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
from scipy.ndimage import distance_transform_edt

from tatbot_travel.camera import Intrinsics, render_plan
from tatbot_travel.kinematics import ArmKinematics
from tatbot_travel.scene import SceneBuilder


def polygon_mask(shape: tuple[int, int], vertices: list) -> np.ndarray:
    mask = np.zeros(shape, np.uint8)
    cv2.fillPoly(mask, [np.asarray(vertices, np.int32)], 255)
    return mask


def surface_mesh(xyz: np.ndarray, keep: np.ndarray, stride: int = 8,
                 max_edge_m: float = 0.15) -> str:
    """A textured surface with holes at excluded objects and depth discontinuities, rather than bridging them."""
    h, w = keep.shape
    ys, xs = np.mgrid[0:h:stride, 0:w:stride]
    points = xyz[ys, xs].reshape(-1, 3)
    ny, nx = ys.shape
    faces = []
    for y in range(ny - 1):
        for x in range(nx - 1):
            # Exclude the whole cell if even one source pixel belonged to the phantom or cradle.
            if not keep[ys[y, x]:ys[y + 1, x] + 1, xs[y, x]:xs[y, x + 1] + 1].all():
                continue
            a, b = y * nx + x, (y + 1) * nx + x
            for tri in ((a, b, b + 1), (a, b + 1, a + 1)):
                p = points[list(tri)]
                if np.max(np.linalg.norm(p - np.roll(p, 1, axis=0), axis=1)) <= max_edge_m:
                    faces.append(tri)
    if not faces:
        raise ValueError("scene masks/depth left no room surface")
    lines = [f"v {x:.7f} {y:.7f} {z:.7f}" for x, y, z in points]
    lines += [f"vt {x / (w - 1):.7f} {1 - y / (h - 1):.7f}" for y, x in zip(ys.ravel(), xs.ravel(), strict=True)]
    lines += ["f " + " ".join(f"{i + 1}/{i + 1}" for i in tri) for tri in faces]
    return "\n".join(lines) + "\n"


def _read(path: Path, flags: int = cv2.IMREAD_COLOR) -> np.ndarray:
    result = cv2.imread(str(path), flags)
    if result is None:
        raise ValueError(f"cannot read capture {path}")
    return result


def _intrinsics(attributes: dict) -> Intrinsics:
    spec = json.loads(attributes["intrinsics"])
    if spec["distortion_model"] != "BrownConradyInverse":
        raise ValueError("scene-bank expects inverse Brown-Conrady wrist captures")
    return Intrinsics(width=spec["width"], height=spec["height"], fx=spec["fx"], fy=spec["fy"],
                      cx=spec["ppx"], cy=spec["ppy"], distortion=tuple(spec["distortion_coefficients"]))


def deproject(depth: np.ndarray, intr: Intrinsics, attributes: dict) -> np.ndarray:
    """Aligned depth may store native-depth Z: solve for colour Z before using the colour rays."""
    v, u = np.mgrid[:intr.height, :intr.width]
    x, y = intr.undistort_normalized(u, v)
    rays = np.stack([x, y, np.ones_like(x)], axis=-1)
    z = depth.astype(np.float64) * float(attributes["depth_units_m"])
    alignment = json.loads(attributes.get("alignment_calibration", "{}"))
    if alignment.get("aligned_depth_value_axis") == "native_depth_z":
        rot = np.asarray(alignment["rotation"]).reshape(3, 3, order="F")
        offset = np.asarray(alignment["translation"])
        z = (z + (rot.T @ offset)[2]) / (rays @ rot)[..., 2]
    return rays * z[..., None]


def _crop_bank(captures: Path, layout: dict, out: Path) -> None:
    for kind, crops in layout["crops"].items():
        if kind not in ("wrist_bg", "scene_bg", "table", "ink"):
            raise ValueError(f"unknown bank crop kind: {kind}")
        (out / kind).mkdir()
        for i, crop in enumerate(crops):
            rgb = _read(captures / crop["file"])
            x0, y0, x1, y1 = crop["box"]
            patch = rgb[y0:y1, x0:x1].copy()
            if patch.size == 0:
                raise ValueError(f"empty {kind} crop")
            if kind == "ink":
                grey = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY)
                alpha = np.clip((crop.get("threshold", 150) - grey.astype(float)) * 4, 0, 255).astype(np.uint8)
                patch = np.dstack([patch, alpha])
                if not alpha.any():
                    raise ValueError("ink crop has no segmented ink")
            cv2.imwrite(str(out / kind / f"crop_{i:02d}.png"), patch)


def build_bank(captures: Path, layout_path: Path, out: Path) -> dict:
    """Masks/placements are explicit inputs; this tool never adopts fleet calibration or touches a driver."""
    if any((parent / ".git").exists() for parent in (out.resolve(), *out.resolve().parents)):
        raise ValueError("private scene banks must stay outside Git checkouts")
    if out.exists():
        raise ValueError(f"bank already exists: {out}; choose a new bank")
    layout = json.loads(layout_path.read_text())
    meta = json.loads((captures / layout["wrist_metadata"]).read_text())["frames"]
    colour, depth_frame = meta["realsense1_color"], meta["realsense1_depth"]
    folder = (captures / layout["wrist_metadata"]).parent
    bgr = _read(folder / colour["file"])
    depth = _read(folder / depth_frame["file"], cv2.IMREAD_UNCHANGED)
    intr = _intrinsics(colour["metadata"]["attributes"])
    if bgr.shape != (intr.height, intr.width, 3) or depth.shape != bgr.shape[:2]:
        raise ValueError("RGB-D dimensions disagree with the capture intrinsics")
    out.mkdir(parents=True)
    _crop_bank(captures, layout, out)
    alpha = polygon_mask(depth.shape, layout["selfview_polygon"])
    phantom = polygon_mask(depth.shape, layout["phantom_polygon"])
    (out / "selfview").mkdir()
    cv2.imwrite(str(out / "selfview" / "cradle.png"), np.dstack([bgr, alpha]))
    missing = depth == 0
    attributes = depth_frame["metadata"]["attributes"]
    reliable = (depth > 0) & (depth * float(attributes["depth_units_m"]) < 0.75)
    distance, nearest = distance_transform_edt(~reliable, return_indices=True)
    filled = missing & (distance <= layout.get("depth_fill_px", 16))
    depth = depth.copy()
    depth[filled] = depth[tuple(nearest[:, filled])]
    xyz_cam = deproject(depth, intr, attributes)
    unknown = depth == 0
    # D405 has no room-range depth. The far backdrop's assumed distance is recorded, not treated as measured.
    xyz_cam[unknown] = deproject(np.full_like(depth, 10000), intr,
                                {"depth_units_m": str(layout.get("far_depth_m", 2.5) / 10000)})[unknown]
    kin = ArmKinematics(SceneBuilder(plan=render_plan(intr)).compile())
    q = np.asarray(json.loads((captures / "joints.json").read_text())["before"]["positions"][:6])
    pos, rot = kin.camera_pose(q)
    xyz = xyz_cam @ rot.T + pos
    xyz[..., 2] += layout["episode"]["world"]["base_height_m"][0]
    (out / "room").mkdir()
    keep = (alpha == 0) & (phantom == 0)
    (out / "room" / "wrist.obj").write_text(surface_mesh(xyz, keep))
    cv2.imwrite(str(out / "room" / "wrist.png"), bgr)
    # A continuous distant backing closes depth holes. Remove dynamic objects from its pixels too;
    # otherwise the policy could read a static copy of the ink after the real phantom moved away.
    background = cv2.inpaint(bgr, np.maximum(alpha, phantom), 7, cv2.INPAINT_TELEA)
    far = deproject(np.ones(depth.shape), intr, {"depth_units_m": str(layout.get("far_depth_m", 2.5))})
    far = far @ rot.T + pos
    far[..., 2] += layout["episode"]["world"]["base_height_m"][0]
    (out / "room" / "backdrop.obj").write_text(surface_mesh(far, np.ones_like(keep)))
    cv2.imwrite(str(out / "room" / "backdrop.png"), background)
    # Project the photographed tabletop onto its measured plane. This supplies paper/mat pixels where
    # shiny white fixtures have no D405 depth, instead of exposing the procedural fallback table.
    rays = deproject(np.ones(depth.shape), intr, {"depth_units_m": "1"}) @ rot.T
    origin = pos + np.array([0, 0, layout["episode"]["world"]["base_height_m"][0]])
    forward = rays[..., 2] < -0.1
    distance = -origin[2] / np.where(forward, rays[..., 2], -1)
    table = origin + rays * distance[..., None]
    table[..., 2] = 0.001  # just above the procedural fallback, avoiding coincident surfaces
    (out / "room" / "table.obj").write_text(surface_mesh(table, keep & forward))
    cv2.imwrite(str(out / "room" / "table.png"), bgr)
    episode = layout["episode"]
    episode["world"].update(real_assets=".", room_assets="room", selfview_assets="selfview")
    profile = {"version": 1, "episode": episode}
    (out / "profile.json").write_text(json.dumps(profile, indent=2) + "\n")
    report = {"intrinsics": intr.__dict__, "q": q.tolist(), "camera_pos": pos.tolist(),
              "camera_rot": rot.tolist(), "far_depth_m": layout.get("far_depth_m", 2.5),
              "missing_depth_fraction": float(missing.mean()), "layout": layout,
              "filled_depth_fraction": float(filled.mean()),
              "captures": str(captures.resolve())}
    (out / "manifest.json").write_text(json.dumps(report, indent=2) + "\n")
    return report
