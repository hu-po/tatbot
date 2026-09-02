#!/usr/bin/env python3
"""The shadow of a draw dir in Rerun: surface, normals, orbit, path, captures.

    draw_shadow.py <draw-dir> [--connect rerun+http://HOST:9876/proxy] [--save]

Reads what `tatbot draw` left under the dir (docs/draw.md): surface.npz (root
frame), orbit.csv / path.csv (base frame; converted here by the one root-from-
base offset), and capture/capture-*.npz (raw D405 depth, shown in each
camera's own optical frame — the mapper owns the root-frame fusion, and this
file must not pretend to). Everything lands under one `draw/` entity tree so
it can sit beside visiond's URDF in the same viewer.

Without --connect the shadow is written to <dir>/shadow.rrd (rr.save); with
--connect it streams to the viewer, and --save adds the file. Runs in the
LeRobot venv (numpy + rerun); `write_shadow()` is importable by draw_stage.py
so the map stage can log the same picture at the hold.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

try:
    import rerun as rr
except ImportError:  # pragma: no cover - environment, not logic
    sys.exit("draw_shadow: `rerun` is not importable in this interpreter; run it in the LeRobot venv "
             "(python/lerobot_robot_tatbot/.venv/bin/python) or `pip install rerun-sdk==0.33.1`")

REPO = Path(__file__).resolve().parent.parent
ROOT_FROM_BASE = np.array([0.0, -0.2675, 0.0])  # right/base_link in root (urdf/tatbot.urdf)
MAX_CAPTURE_POINTS = 50_000
TOOL_AXIS_EVERY = 200
ROLES = ("wrist_upper", "wrist_lower")
PALETTE = np.array([[230, 90, 60], [60, 160, 230], [90, 200, 90], [230, 190, 50], [190, 90, 220],
                    [60, 210, 200], [240, 130, 40], [160, 160, 160]], dtype=np.uint8)


def _lib():
    for p in (REPO / "scripts/lib", REPO / "scripts/vision"):
        if str(p) not in sys.path:
            sys.path.insert(0, str(p))


def log_view_coordinates() -> None:
    try:
        # Cosmetic (tells the 3D view which way is up); some rerun/numpy ABI
        # pairs reject this batch type, and nothing else depends on it.
        rr.log("/", rr.ViewCoordinates.RIGHT_HAND_Z_UP, static=True)
    except Exception as error:  # noqa: BLE001
        print(f"  (view-coordinates hint skipped: {error})")


# --- surface --------------------------------------------------------------------------


def _surface_grid(surface):
    """Vertices on the surface's own count grid, so colour-by-count is exact."""
    rows, cols = np.asarray(surface.count).shape
    u = np.linspace(-surface.width_m / 2, surface.width_m / 2, cols)
    v = np.linspace(-surface.height_m / 2, surface.height_m / 2, rows)
    uu, vv = np.meshgrid(u, v)  # rows index v, cols index u (docs/draw.md)
    uv = np.stack([uu.ravel(), vv.ravel()], -1)
    out = surface.frame(uv)  # (point, d/du, d/dv, unit normal), each (N, 3)
    points, normals = np.asarray(out[0], float), np.asarray(out[-1], float)
    idx = np.arange(rows * cols).reshape(rows, cols)
    a, b, c, d = idx[:-1, :-1].ravel(), idx[:-1, 1:].ravel(), idx[1:, 1:].ravel(), idx[1:, :-1].ravel()
    faces = np.concatenate([np.stack([a, b, c], -1), np.stack([a, c, d], -1)])
    return points, faces, normals, np.asarray(surface.count).ravel()


def log_surface(surface, entity: str = "draw/surface") -> None:
    points, faces, normals, count = _surface_grid(surface)
    # Shaded by sample count: holes red, thin cells amber, well-sampled cells paper-white.
    colors = np.full((len(points), 3), [235, 235, 225], np.uint8)
    thin = (count > 0) & (count < 3)
    colors[thin] = [235, 180, 80]
    colors[count == 0] = [220, 50, 50]
    rr.log(entity, rr.Mesh3D(vertex_positions=points, triangle_indices=faces, vertex_normals=normals,
                             vertex_colors=colors), static=True)
    stride = max(1, int(round(0.005 / max(surface.width_m / max(points.shape[0] ** 0.5, 1), 1e-6))))
    rows, cols = np.asarray(surface.count).shape
    sel = np.zeros((rows, cols), bool)
    sel[::stride, ::stride] = True
    sel = sel.ravel() & (count > 0)
    rr.log(f"{entity}/normals", rr.Arrows3D(origins=points[sel], vectors=normals[sel] * 0.01,
                                            colors=[[80, 160, 255]], radii=0.0002), static=True)
    anchor = getattr(surface, "anchor_point", None)
    kind = getattr(getattr(surface, "chart", None), "kind", "?")
    holes = int((count == 0).sum())
    rr.log(f"{entity}/info", rr.TextDocument(
        f"chart {kind}; canvas {surface.width_m * 1000:.0f} x {surface.height_m * 1000:.0f} mm; "
        f"{rows}x{cols} cells; {holes} holes (red)"), static=True)
    return anchor


# --- samples files ------------------------------------------------------------------


def _runs(mask: np.ndarray) -> list[tuple[int, int]]:
    """Contiguous [start, end) index runs where mask is true."""
    runs, start = [], None
    for i, m in enumerate(mask):
        if m and start is None:
            start = i
        elif not m and start is not None:
            runs.append((start, i))
            start = None
    if start is not None:
        runs.append((start, len(mask)))
    return runs


def _tool_axis_link6() -> np.ndarray:
    """right/tool_mount +z expressed in right/link_6 (from the URDF; docs/draw.md)."""
    try:
        from urdf_kinematics import UrdfChain
        chain = UrdfChain(REPO / "urdf/tatbot.urdf")
        link6 = chain.link_pose("right/link_6")
        mount = chain.link_pose("right/tool_mount")
        return (np.linalg.inv(link6) @ mount)[:3, 2]
    except Exception as error:  # noqa: BLE001
        print(f"  (tool axis from URDF unavailable: {error}; using link-6 +x)")
        return np.array([1.0, 0.0, 0.0])


def log_path(samples: dict, entity: str = "draw/path") -> None:
    p = np.asarray(samples["p"], float) + ROOT_FROM_BASE
    pen = np.asarray(samples.get("pen", np.ones(len(p))), float) > 0.5
    down = [p[s:e + 1 if e < len(p) else e] for s, e in _runs(pen)]
    travel = [p[max(s - 1, 0):e + 1 if e < len(p) else e] for s, e in _runs(~pen)]
    if down:
        rr.log(f"{entity}/pen_down", rr.LineStrips3D(down, colors=[[20, 20, 20]], radii=0.0003), static=True)
    if travel:
        rr.log(f"{entity}/travel", rr.LineStrips3D(travel, colors=[[120, 120, 200]], radii=0.0002), static=True)
    rot = np.asarray(samples["R"], float)
    if rot.ndim == 3 and len(rot) == len(p):
        axis = _tool_axis_link6()
        sel = np.arange(0, len(p), TOOL_AXIS_EVERY)
        vectors = -(rot[sel] @ axis) * 0.015  # from the tip back up the pen body
        rr.log(f"{entity}/tool_axis", rr.Arrows3D(origins=p[sel], vectors=vectors, colors=[[255, 120, 40]],
                                                  radii=0.0003), static=True)
    length = float(np.linalg.norm(np.diff(p[pen], axis=0), axis=1).sum()) if pen.sum() > 1 else 0.0
    rr.log(f"{entity}/info", rr.TextDocument(
        f"{len(p)} samples at {samples.get('t', [0, 0.0025])[1] - samples.get('t', [0, 0])[0]:.4f} s; "
        f"pen-down length {length * 1000:.1f} mm; base->root offset {ROOT_FROM_BASE.tolist()}"), static=True)


def log_orbit(samples: dict, entity: str = "draw/orbit") -> None:
    p = np.asarray(samples["p"], float) + ROOT_FROM_BASE
    rr.log(entity, rr.LineStrips3D([p], colors=[[90, 200, 90]], radii=0.0004), static=True)
    cap = np.asarray(samples.get("capture", np.zeros(len(p))), int)
    idx = np.nonzero(cap > 0)[0]
    if len(idx):
        rr.log(f"{entity}/captures", rr.Points3D(p[idx], colors=[[90, 200, 90]], radii=0.002,
                                                 labels=[f"capture {int(cap[i])}" for i in idx]), static=True)


# --- captures ---------------------------------------------------------------------------


def deproject(depth: np.ndarray, intrinsics, units_m: float, max_points: int = MAX_CAPTURE_POINTS):
    fx, fy, ppx, ppy = (float(x) for x in np.asarray(intrinsics)[:4])
    valid = (depth > 0) & (depth < 65535)
    v, u = np.nonzero(valid)
    z = depth[v, u].astype(np.float64) * units_m
    if len(z) > max_points:
        keep = np.linspace(0, len(z) - 1, max_points).astype(int)
        u, v, z = u[keep], v[keep], z[keep]
    return np.stack([(u - ppx) / fx * z, (v - ppy) / fy * z, z], -1)


def log_captures(capture_dir: Path, entity: str = "draw/captures") -> int:
    files = sorted(capture_dir.glob("capture-*.npz"), key=lambda f: int(f.stem.split("-")[1]))
    if not files:
        return 0
    rr.log(entity, rr.TextDocument(
        "Raw D405 depth per capture, deprojected into EACH CAMERA'S OPTICAL FRAME (z forward). "
        "Not placed in root: the mapper (draw_stage.py map) owns FK + hand-eye fusion. "
        "One colour per capture k."), static=True)
    for f in files:
        z = np.load(f)
        k = int(z["k"]) if "k" in z else int(f.stem.split("-")[1])
        color = PALETTE[k % len(PALETTE)]
        for role in ROLES:
            if f"depth_{role}" not in z:
                continue
            pts = deproject(z[f"depth_{role}"], z[f"intrinsics_{role}"], float(z[f"units_m_{role}"]))
            rr.log(f"{entity}/{k}/{role}", rr.Points3D(pts, colors=[color.tolist()], radii=0.0004), static=True)
    return len(files)


# --- driver -------------------------------------------------------------------------------


def _sinks(draw_dir: Path, connect: str | None, save: bool) -> None:
    rrd = draw_dir / "shadow.rrd"
    if connect and save and hasattr(rr, "set_sinks") and hasattr(rr, "GrpcSink") and hasattr(rr, "FileSink"):
        rr.set_sinks(rr.GrpcSink(url=connect), rr.FileSink(path=rrd))
        return
    if connect:
        if hasattr(rr, "connect_grpc"):
            rr.connect_grpc(connect)
        else:  # older SDK spelling
            rr.connect(connect)
        if save:
            print("  (this rerun SDK cannot stream and save at once; shadow.rrd skipped)")
        return
    rr.save(rrd)


def write_shadow(draw_dir: str | Path, connect: str | None = None, save: bool = True) -> dict:
    """Log the shadow of `draw_dir`; returns what was found. Missing pieces are skipped, not fatal."""
    _lib()
    draw_dir = Path(draw_dir).expanduser()
    if not draw_dir.is_dir():
        raise FileNotFoundError(f"not a draw dir: {draw_dir}")
    rr.init("tatbot_draw_shadow", recording_id=f"draw-shadow-{draw_dir.name}")
    _sinks(draw_dir, connect, save)
    log_view_coordinates()
    found: dict = {"draw_dir": str(draw_dir)}

    surface_npz = draw_dir / "surface.npz"
    anchor = None
    if surface_npz.is_file():
        from draw_surface import HeightFieldSurface
        surface = HeightFieldSurface.from_npz(surface_npz)
        anchor = log_surface(surface)
        found["surface"] = str(surface_npz)
    else:
        print(f"  no surface.npz in {draw_dir} (the map stage has not run)")

    for name, logger in (("path.csv", log_path), ("orbit.csv", log_orbit)):
        f = draw_dir / name
        if not f.is_file():
            print(f"  no {name}")
            continue
        from draw_path import read_samples_csv
        samples, _header = read_samples_csv(f)
        logger({k: getattr(samples, k) for k in ("t", "p", "v", "R", "pen", "capture")})
        found[name] = str(f)

    if anchor is not None:
        rr.log("draw/anchor", rr.Points3D([np.asarray(anchor, float)], colors=[[255, 40, 40]], radii=0.0015,
                                          labels=["anchor (contact)"]), static=True)
    found["captures"] = log_captures(draw_dir / "capture")
    print(f"  shadow: {', '.join(k for k in found if k != 'draw_dir')} -> "
          f"{connect or draw_dir / 'shadow.rrd'}")
    return found


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("draw_dir", type=Path)
    ap.add_argument("--connect", help="rerun+http://HOST:9876/proxy of a running viewer")
    ap.add_argument("--save", action="store_true", help="also write <dir>/shadow.rrd (default when no --connect)")
    a = ap.parse_args(argv)
    write_shadow(a.draw_dir, connect=a.connect, save=a.save or not a.connect)
    return 0


if __name__ == "__main__":
    sys.exit(main())
