#!/usr/bin/env python3
"""`tatbot draw` stages: orbit, map, plan (docs/draw.md).

    draw_stage.py orbit <dir>                 draw.json + trigger.json -> orbit.csv
    draw_stage.py map   <dir>                 captures -> surface.npz/json, path.csv, preflight.json, shadow
    draw_stage.py plan  <dir|surface.npz> [--out DIR] [--config draw.json]
                                              offline compile + preflight on an existing surface

Exit 0 ok, 3 refusal (preflight.json names the code), 1 error. The C++
executor (`wxai_teleop --draw-dir`) runs `orbit` and `map` as subprocesses
during its holds and reads only the samples files this writes. Every motion
here is advisory geometry: the executor plans, gates and streams it.

numpy only. `scripts/lib/draw_surface.py` (the mapper's HeightFieldSurface)
and `scripts/draw_shadow.py` (the Rerun shadow) are imported lazily so the
`orbit` stage runs before either exists on a node.
"""

from __future__ import annotations

import argparse
import glob
import json
import math
import os
import subprocess
import sys
import time
import traceback
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
for sub in ("scripts/lib", "scripts/vision"):
    entry = str(REPO / sub)
    if entry not in sys.path:
        sys.path.insert(0, entry)

import draw_kinematics as dk  # noqa: E402
import draw_path as dp  # noqa: E402

CONFIG_SCHEMA = "tatbot.draw-config/1"
POSE_SCHEMA = "tatbot.draw-pose/1"
SURFACE_SCHEMA = "tatbot.surface/1"
DEPTH_MIN_M = 0.07
DEPTH_MAX_M = 0.5
MAX_POINTS = 200_000
CAMERA_LINKS = {
    "wrist_upper": "right/realsense_depth_optical_frame",
    "wrist_lower": "right/realsense_lower_depth_optical_frame",
}
EXIT_OK, EXIT_ERROR, EXIT_REFUSED = 0, 1, 3


class StageError(RuntimeError):
    pass


# --- files ------------------------------------------------------------------------

def _read_json(path: Path, schema: str) -> dict:
    if not path.is_file():
        raise StageError(f"missing {path}")
    data = json.loads(path.read_text())
    if data.get("schema") != schema:
        raise StageError(f"{path}: schema {data.get('schema')!r}, expected {schema!r}")
    return data


def _write_json(path: Path, data: dict) -> None:
    path.write_text(json.dumps(_jsonable(data), indent=2, sort_keys=True) + "\n")


def _jsonable(value):
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, np.ndarray):
        return _jsonable(value.tolist())
    if isinstance(value, (np.floating, float)):
        return None if not math.isfinite(float(value)) else float(value)
    if isinstance(value, np.integer):
        return int(value)
    return value


def _scalars(report: dict) -> dict:
    """The report keys that fit a `key,value` header line."""
    return {k: v for k, v in report.items()
            if isinstance(v, (int, float, str)) and not isinstance(v, bool)}


def _git_sha() -> str:
    try:
        return subprocess.run(["git", "-C", str(REPO), "rev-parse", "--short", "HEAD"],
                              capture_output=True, text=True, timeout=5, check=False).stdout.strip() or "unknown"
    except (OSError, subprocess.SubprocessError):
        return "unknown"


def _load_config(path: Path) -> dict:
    config = _read_json(path, CONFIG_SCHEMA)
    for key in ("design", "duration_s", "ease_s"):
        if key not in config:
            raise StageError(f"{path}: missing {key!r}")
    return config


def _period(config: dict, pose: dict | None) -> float:
    period = float((pose or {}).get("period_s") or config.get("period_s") or 0.0025)
    if not (0.0 < period <= 0.1):
        raise StageError(f"implausible control period {period}")
    return period


def _tool_constants():
    tip = dk.ballpoint_tip_in_link6_from_config()
    gap = float(np.linalg.norm(tip - dk.BALLPOINT_TIP_IN_LINK6))
    if gap > 1e-4:
        raise StageError(f"workspace.yaml tip differs from the executor's constant by {gap * 1e3:.3f} mm")
    return tip, dk.tool_axis_in_link6()


# --- orbit --------------------------------------------------------------------------

def stage_orbit(draw_dir: Path) -> int:
    config = _load_config(draw_dir / "draw.json")
    trigger = _read_json(draw_dir / "trigger.json", POSE_SCHEMA)
    period = _period(config, trigger)
    tip, axis = _tool_constants()
    samples, report = dp.orbit_samples(config, trigger, period, axis)
    pre = dp.preflight(samples, None, config, axis, trigger)
    report.update({k: v for k, v in pre.items() if v is not None})
    dp.write_samples_csv(draw_dir / "orbit.csv", samples, "orbit", tip, _scalars(report))
    refusal, executor = executor_check(draw_dir / "orbit.csv", trigger, period)
    report.update(executor)
    _write_json(draw_dir / "orbit.json", report)
    if refusal is not None:
        (draw_dir / "orbit.csv").rename(draw_dir / "orbit.refused.csv")
        print(f"orbit REFUSED by the executor's planner: {refusal}", file=sys.stderr)
        return EXIT_REFUSED
    where = (f"cameras {report['camera_distance_mm']:.0f} mm from the patch, {report['off_axis_deg']:.0f} deg off axis "
             f"(max incidence {report['camera_incidence_max_deg']:.0f} deg, tip >= {report['tip_height_min_mm']:.0f} mm up)"
             if report.get("mode") == "camera" else f"{report['standoff_mm']:.0f} mm standoff")
    print(f"orbit ({report.get('mode', 'tip')}): {report['capture_count']} viewpoints, {where}, "
          f"tilt {report['tilt_deg']:.0f} deg, {samples.n} samples over {samples.duration_s:.1f} s, "
          f"tip speed max {pre['tip_speed_max_mm_s']:.1f} mm/s -> {draw_dir / 'orbit.csv'}")
    return EXIT_OK


# --- map: deproject, fuse, anchor ---------------------------------------------------

def _deproject(depth: np.ndarray, units_m: float, intrinsics) -> np.ndarray:
    """Pinhole deprojection of a uint16 depth image into the camera optical frame (N,3)."""
    fx, fy, ppx, ppy = (float(v) for v in np.asarray(intrinsics, float)[:4])
    depth = np.asarray(depth)
    valid = (depth > 0) & (depth < 65535)
    z = depth.astype(np.float64) * float(units_m)
    valid &= (z >= DEPTH_MIN_M) & (z <= DEPTH_MAX_M)
    rows, cols = np.nonzero(valid)
    z = z[rows, cols]
    x = (cols.astype(np.float64) - ppx) / fx * z
    y = (rows.astype(np.float64) - ppy) / fy * z
    return np.stack([x, y, z], axis=1)


def _capture_cloud(npz, chain) -> tuple[np.ndarray, list[str]]:
    """All valid depth pixels of one capture in root, plus the roles that contributed."""
    joints = np.asarray(npz["joints"], float)
    carriage = float(np.asarray(npz["carriage_m"]).reshape(-1)[0])
    values = dk.joint_map(joints, carriage)
    clouds = []
    roles = []
    for role, link in CAMERA_LINKS.items():
        key = f"depth_{role}"
        if key not in npz.files:
            continue
        units = float(np.asarray(npz[f"units_m_{role}"]).reshape(-1)[0])
        points = _deproject(npz[key], units, npz[f"intrinsics_{role}"])
        if len(points) == 0:
            continue
        pose = chain.link_pose(link, values)
        clouds.append(points @ pose[:3, :3].T + pose[:3, 3])
        roles.append(role)
    if not clouds:
        return np.zeros((0, 3)), roles
    return np.concatenate(clouds), roles


def mask_tool(points: np.ndarray, tip: np.ndarray, axis: np.ndarray, radius_m: float,
              tip_radius_m: float = 0.0015, taper: float = 0.5, behind_m: float = 0.0005) -> np.ndarray:
    """Drop the fitted tool's own body from a cloud.

    A point is tool if it lies behind the tip along the tool axis
    (``s = -(p - tip) . axis > behind_m``) and within the pen's silhouette
    there: a cone of radius ``tip_radius_m + taper * s`` capped at
    ``radius_m`` (the body). Both wrist cameras look along the pen from the
    sides, so its body lies inside the map cube at ~100 mm and, unmasked,
    pulls every fit toward the cameras (live capture 2026-09-01: plane and
    cylinder rms 6-7 mm, cylinder radius 35 mm on a 45 mm bottle). The cone
    matters: with the pen leaning 37 deg off the normal, a straight 20 mm
    cylinder behind the tip also swallowed the surface to one side of the
    contact and left the design on holes.
    """
    if len(points) == 0:
        return points
    rel = points - tip
    along = rel @ axis
    radial = np.linalg.norm(rel - along[:, None] * axis[None, :], axis=1)
    s = -along
    allowed = np.minimum(radius_m, tip_radius_m + taper * s)
    tool = (s > behind_m) & (radial < allowed)
    return points[~tool]


def _subsample(points: np.ndarray, limit: int, seed: int = 0) -> np.ndarray:
    if len(points) <= limit:
        return points
    keep = np.random.default_rng(seed).choice(len(points), size=limit, replace=False)
    return points[np.sort(keep)]


def _fit_stats(surface) -> dict:
    """Fit statistics carried on a loaded surface (the sidecar json, when present)."""
    return dict(getattr(surface, "fit", None) or {})


def _chart_radius(surface) -> float:
    chart = getattr(surface, "chart", None)
    radius = getattr(chart, "radius_m", getattr(chart, "radius", float("nan")))
    return float(radius) if isinstance(radius, (int, float, np.floating)) else float("nan")


def _summary(surface, pre: dict | None, refusal: str | None, stats: dict) -> str:
    kind = getattr(getattr(surface, "chart", None), "kind", "?")
    radius = _chart_radius(surface)
    radius_text = f" radius {radius * 1e3:.1f} mm" if math.isfinite(radius) else ""
    rms = stats.get("rms_m", stats.get("fit_rms_m"))
    rms_text = f", fit rms {rms * 1e3:.2f} mm" if isinstance(rms, (int, float)) else ""
    count = np.asarray(getattr(surface, "count", np.zeros((0, 0))))
    filled = f"{int((count > 0).sum())}/{count.size} cells filled" if count.size else "no cell counts"
    head = f"Surface: {kind} chart{radius_text}{rms_text}, {filled}."
    if refusal:
        return f"{head} REFUSED: {refusal}."
    if not pre:
        return f"{head} Scan only; no path compiled."
    cruise = max((float(s.get("cruise_speed_mm_s", 0.0)) for s in pre.get("strokes", [])), default=0.0)
    executor = ""
    if pre.get("executor_model_max_error_mm") is not None:
        executor = (f" Executor: model error {float(pre['executor_model_max_error_mm']):.3f} mm (pen-down cap "
                    f"{dk.PLAN_MAX_MODEL_ERROR_DRAW_M * 1e3:g}), joint velocity "
                    f"{float(pre['executor_plan_max_joint_velocity_rad_s']):.3f} rad/s (cap "
                    f"{dk.PLAN_MAX_JOINT_VELOCITY_RAD_S:g}).")
    return (f"{head} Path: {pre['sample_count']} samples over {pre['duration_s']:.1f} s, pen-down "
            f"{pre['path_length_mm'] or 0:.1f} mm on the surface at {cruise:.2f} mm/s, arc-length ratio "
            f"{pre['arc_length_ratio'] if pre['arc_length_ratio'] is not None else float('nan'):.4f}, "
            f"lean max {pre['lean_max_deg'] or 0:.2f} deg, normal swing {pre['normal_swing_max_deg'] or 0:.1f} deg, "
            f"holes {pre['holes']}, tip speed max {pre['tip_speed_max_mm_s']:.1f} mm/s.{executor}")


def write_shadow(draw_dir: Path, connect: str | None = None) -> str | None:
    """Log the shadow through scripts/draw_shadow.py when it and `rerun` exist; never fatal.

    draw_shadow exits at import when rerun is missing (SystemExit, not ImportError).
    """
    try:
        import draw_shadow
    except (ImportError, SystemExit) as error:
        print(f"shadow: skipped ({error})", file=sys.stderr)
        return None
    try:
        draw_shadow.write_shadow(draw_dir, connect=connect or None, save=True)
    except Exception as error:  # noqa: BLE001 - the shadow is evidence, not a gate
        print(f"shadow: failed ({error})", file=sys.stderr)
        return None
    return str(draw_dir / "shadow.rrd")


PLAN_CHECK = REPO / "cpp" / "teleop" / "build" / "path_plan_check"


def executor_check(samples_path: Path, pose: dict | None, period: float) -> tuple[str | None, dict]:
    """Run the executor's own parser + planner (cpp/teleop path_plan_check) on a samples file.

    Returns (refusal, report). The refusal is None when the executor accepts
    the file or when the check cannot run here (tool not built, or a synthetic
    pose without joints) -- the report says which. The executor re-plans
    anyway; this only moves its answer to before the arms are involved.
    """
    joints = (pose or {}).get("joints")
    if joints is None:
        return None, {"executor_check": "skipped: no joints (synthetic pose)"}
    if not PLAN_CHECK.exists():
        return None, {"executor_check": f"skipped: {PLAN_CHECK.relative_to(REPO)} is not built"}
    argv = [str(PLAN_CHECK), str(samples_path), repr(float(period)),
            *[repr(float(v)) for v in joints], repr(float(pose.get("carriage_m", 0.0)))]
    try:
        run = subprocess.run(argv, capture_output=True, text=True, timeout=120)
    except (OSError, subprocess.TimeoutExpired) as error:
        return None, {"executor_check": f"skipped: {error}"}
    report = {}
    for line in (run.stdout + run.stderr).splitlines():
        key, _, value = line.partition(",")
        if key:
            report[f"executor_{key.strip()}"] = value.strip()
    if run.returncode == 0:
        report["executor_check"] = "accepted"
        return None, report
    if run.returncode == 3:
        return report.get("executor_reason", "executor refused the plan"), report
    return None, {"executor_check": f"skipped: path_plan_check exit {run.returncode}", **report}


def _compile_and_preflight(surface, config, contact, hold, period, tip, axis, out_dir: Path,
                           provenance: dict) -> tuple[int, dict | None, str | None]:
    """Compile the path, preflight it, write path.csv + preflight.json. Returns (exit, report, refusal)."""
    try:
        samples, report = dp.compile_path(surface, config, contact, hold, period)
        pre = dp.preflight(samples, surface, config, axis, hold, design_length_m=report["design_length_mm"] * 1e-3)
    except dp.DrawRefusal as refusal:
        _write_json(out_dir / "preflight.json", {
            "schema": "tatbot.draw-preflight/1", "ok": False, "code": refusal.code, "detail": refusal.detail,
            **provenance})
        return EXIT_REFUSED, None, f"{refusal.code} ({refusal.detail})"
    report.update(pre)
    # The draw keeps the carriage out of the solve (docs/draw.md): the paper A/B's seven-joint
    # mode is kept for the spiral verb only. config.path.carriage_ik = true opts back in.
    report["carriage_ik"] = 1 if config.get("path", {}).get("carriage_ik", False) else 0
    dp.write_samples_csv(out_dir / "path.csv", samples, "path", tip, _scalars(report))
    refusal, executor = executor_check(out_dir / "path.csv", hold, period)
    report.update(executor)
    if refusal is not None:
        (out_dir / "path.csv").rename(out_dir / "path.refused.csv")
        _write_json(out_dir / "preflight.json", {
            "schema": "tatbot.draw-preflight/1", "ok": False, "code": "executor_plan", "detail": refusal,
            **provenance, **report})
        return EXIT_REFUSED, None, f"executor_plan ({refusal})"
    _write_json(out_dir / "preflight.json", {
        "schema": "tatbot.draw-preflight/1", "ok": True, "code": None, **provenance, **report})
    return EXIT_OK, report, None


def stage_map(draw_dir: Path) -> int:
    started = time.time()
    config = _load_config(draw_dir / "draw.json")
    trigger = _read_json(draw_dir / "trigger.json", POSE_SCHEMA)
    hold = _read_json(draw_dir / "hold.json", POSE_SCHEMA)
    period = _period(config, hold)
    tip, axis = _tool_constants()
    try:
        import draw_surface
    except ImportError as error:
        raise StageError(f"scripts/lib/draw_surface.py is not importable: {error}") from error

    chain = dk.urdf_chain()
    capture_files = sorted(glob.glob(str(draw_dir / "capture" / "capture-*.npz")))
    if not capture_files:
        raise StageError(f"no captures under {draw_dir / 'capture'}")
    contact_root = dk.root_from_base(np.asarray(trigger["tip"], float))
    r_c = np.asarray(trigger["rotation"], float)
    axis_root = r_c @ axis
    axis_root /= np.linalg.norm(axis_root)
    tool_radius = float(config.get("map", {}).get("tool_mask_mm", 20.0)) * 1e-3
    clouds = []
    capture_ids = []
    roles_seen = set()
    masked_at_contact = 0
    for file in capture_files:
        with np.load(file) as npz:
            cloud, roles = _capture_cloud(npz, chain)
            capture_ids.append(int(np.asarray(npz["k"]).reshape(-1)[0]) if "k" in npz.files else Path(file).stem)
            joints_k = np.asarray(npz["joints"], float).reshape(-1) if "joints" in npz.files else None
            carriage_k = (float(np.asarray(npz["carriage_m"]).reshape(-1)[0]) if "carriage_m" in npz.files
                          else dk.CARRIAGE_IK_BIAS_M)
        roles_seen.update(roles)
        # The pen's own body is masked where the pen WAS for this capture (FK of the
        # capture's joints), not at the contact: during the orbit the pen is 40+ mm
        # above the surface and outside the map cube, while the fused cloud sits a
        # few mm behind the contact tip (hand-eye: capture_offsets +3..+6 mm on
        # 2026-09-02 run 9078), and the contact-pose cone then swallowed the 3x3
        # cells under the anchor -- 627 spiral samples refused as holes. A capture
        # without joints (bench `draw_capture.py once`) still masks at the contact.
        if joints_k is not None and joints_k.shape == (6,) and np.isfinite(joints_k).all():
            tip_k, r_k, _ = dk.fk_ballpoint(joints_k, carriage_k)
            axis_k = r_k @ axis
            cloud = mask_tool(cloud, dk.root_from_base(tip_k), axis_k / np.linalg.norm(axis_k), tool_radius)
        else:
            masked_at_contact += 1
            cloud = mask_tool(cloud, contact_root, axis_root, tool_radius)
        clouds.append(cloud)
    points = np.concatenate(clouds) if clouds else np.zeros((0, 3))
    if len(points) == 0:
        raise StageError("captures hold no valid depth in 0.07-0.5 m")

    extent = float(config.get("map", {}).get("extent_mm", 60.0)) * 1e-3
    inside = np.all(np.abs(points - contact_root) <= 0.5 * extent, axis=1)
    points = points[inside]
    points = _subsample(points, MAX_POINTS)
    if len(points) < 100:
        raise StageError(f"only {len(points)} depth points inside the {extent * 1e3:.0f} mm map cube")

    normal_hint = -axis_root
    u_hint = np.array([1.0, 0.0, 0.0]) - np.dot([1.0, 0.0, 0.0], normal_hint) * normal_hint
    if np.linalg.norm(u_hint) < 0.1:
        u_hint = np.array([0.0, 1.0, 0.0]) - np.dot([0.0, 1.0, 0.0], normal_hint) * normal_hint
    u_hint /= np.linalg.norm(u_hint)
    cell = float(config.get("map", {}).get("cell_mm", 1.0)) * 1e-3
    prefer = str(config.get("map", {}).get("chart", "auto"))
    chart, stats = draw_surface.choose_chart(points, normal_hint, u_hint, prefer=prefer)
    smooth = float(config.get("map", {}).get("smooth_mm", 4.0)) * 1e-3
    fused = draw_surface.fuse(points, chart, extent, extent, cell, smooth_m=smooth)
    surface, shift_m, anchor_uv = fused.anchor_to(contact_root)
    # Anchor the CHART, not just the height: a curved chart carrying a constant
    # offset is not isometric any more (docs/draw.md). Re-fuse on the offset chart
    # and anchor again; the residual shift is second order.
    chart = chart.offset(shift_m)
    fused = draw_surface.fuse(points, chart, extent, extent, cell, smooth_m=smooth)
    surface, second_shift_m, anchor_uv = fused.anchor_to(contact_root)
    filled = surface.count > 0
    # Per-capture registration: where each viewpoint's cloud sits relative to the fused
    # surface. Five viewpoints disagreeing by +-1.5-2.7 mm (first live scan) is the FK +
    # CAD hand-eye error the plane hand-eye track exists to remove; it is the number to watch.
    capture_offsets = {}
    for cid, cloud in zip(capture_ids, clouds, strict=True):
        near = cloud[np.all(np.abs(cloud - contact_root) <= 0.5 * extent, axis=1)]
        near = mask_tool(near, contact_root, axis_root, tool_radius)
        if len(near) < 50:
            continue
        _, signed = surface.project(_subsample(near, 5000))
        signed = signed[np.isfinite(signed)]
        if len(signed):
            capture_offsets[str(cid)] = {"median_mm": float(np.median(signed)) * 1e3,
                                         "mad_mm": float(np.median(np.abs(signed - np.median(signed)))) * 1e3}
    stats = dict(stats, holes=int((surface.count == 0).sum()), cells=int(surface.count.size),
                 anchor_second_shift_mm=float(second_shift_m) * 1e3,
                 height_median_mm=float(np.median(surface.height[filled])) * 1e3 if filled.any() else None,
                 anchored_radius_m=_chart_radius(surface), capture_offsets=capture_offsets)
    surface.fit = stats

    provenance = {
        "draw_dir": str(draw_dir), "captures": capture_ids, "camera_roles": sorted(roles_seen),
        "tool": config.get("tool"), "git_sha": _git_sha(), "points_used": int(len(points)),
        "anchor_shift_mm": float(shift_m) * 1e3, "anchor_uv_m": [float(v) for v in np.asarray(anchor_uv).reshape(-1)[:2]],
        "chart_kind": chart.kind, "radius_m": _chart_radius(surface),
        "cell_mm": cell * 1e3, "extent_mm": extent * 1e3, "smooth_mm": smooth * 1e3,
        "tool_mask_mm": tool_radius * 1e3, "tool_mask_at_contact_captures": masked_at_contact, "fit": stats,
        "calibration": {"tip_in_link6": tip.tolist(), "tool_axis_in_link6": axis.tolist(),
                        "urdf": str(dk.URDF_PATH.relative_to(REPO))},
    }
    surface.to_npz(draw_dir / "surface.npz", extra_json=_jsonable(provenance))  # also writes surface.json

    code, report, refusal = EXIT_OK, None, None
    if not config.get("scan_only", False):
        code, report, refusal = _compile_and_preflight(
            surface, config, trigger, hold, period, tip, axis, draw_dir, {"draw_dir": str(draw_dir)})
    shadow = write_shadow(draw_dir, config.get("rerun_connect"))
    print(_summary(surface, report, refusal, stats)
          + (f" Shadow: {shadow}." if shadow else " Shadow: not written (draw_shadow unavailable).")
          + f" ({time.time() - started:.1f} s)")
    return code


# --- plan: offline compile on an existing surface -------------------------------------

DEFAULT_CONFIG = {
    "schema": CONFIG_SCHEMA, "tool": "lutin-ballpoint-dot",
    "design": {"kind": "spiral", "radius_mm": 15, "turns": 3, "rotation_deg": 0},
    "duration_s": 120, "draw_speed_mm_s": 3.5, "ease_s": 2, "scan_only": False,
    "orbit": {"mode": "camera", "camera_distance_mm": 160, "off_axis_deg": 35, "standoff_mm": 80, "tilt_deg": 15,
              "poses": 5, "speed_mm_s": 20},
    "map": {"cell_mm": 1.0, "extent_mm": 60, "chart": "auto", "smooth_mm": 4.0, "tool_mask_mm": 20.0},
    "lean_budget_deg": 20, "lean_deadband_deg": 0,
}


def stage_plan(target: Path, out_dir: Path | None, config_path: Path | None) -> int:
    surface_path = target / "surface.npz" if target.is_dir() else target
    draw_dir = surface_path.parent
    out_dir = out_dir or draw_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    if config_path is None and (draw_dir / "draw.json").is_file():
        config_path = draw_dir / "draw.json"
    config = _load_config(config_path) if config_path else dict(DEFAULT_CONFIG)
    period = _period(config, None)
    tip, axis = _tool_constants()
    try:
        import draw_surface
    except ImportError as error:
        raise StageError(f"scripts/lib/draw_surface.py is not importable: {error}") from error
    if not surface_path.is_file():
        raise StageError(f"missing {surface_path}")
    surface = draw_surface.HeightFieldSurface.from_npz(surface_path)
    sidecar = surface_path.with_suffix(".json")
    if sidecar.is_file():
        surface.fit = (json.loads(sidecar.read_text()).get("fit") or {})
    with np.load(surface_path) as npz:
        if str(npz["schema"]) != SURFACE_SCHEMA:
            raise StageError(f"{surface_path}: schema {npz['schema']!r}")
        anchor_root = np.asarray(npz["anchor_point"], float)
        anchor_uv = np.asarray(npz["anchor_uv"], float)

    # Synthetic contact at the anchor: tool axis along -n, hold at the standoff above it.
    _, _, _, normal = surface.frame(anchor_uv[None, :])
    n_c = np.asarray(normal, float)[0]
    n_c /= np.linalg.norm(n_c)
    r_c = dk.align_rotation(axis, -n_c)
    tip_c = dk.base_from_root(anchor_root)
    standoff = float(config.get("orbit", {}).get("standoff_mm", 80.0)) * 1e-3
    contact = {"schema": POSE_SCHEMA, "frame": dp.SAMPLES_FRAME, "period_s": period, "joints": None,
               "carriage_m": dk.CARRIAGE_IK_BIAS_M, "tip": tip_c.tolist(), "rotation": r_c.tolist(),
               "tool": config.get("tool"), "t_wall": time.time(), "synthetic": True}
    hold = dict(contact, tip=(tip_c + standoff * n_c).tolist())
    _write_json(out_dir / "contact.synthetic.json", contact)
    code, report, refusal = _compile_and_preflight(
        surface, config, contact, hold, period, tip, axis, out_dir,
        {"surface": str(surface_path), "synthetic_contact": True})
    print(_summary(surface, report, refusal, _fit_stats(surface)) + f" -> {out_dir}")
    return code


# --- main -------------------------------------------------------------------------------

def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="stage", required=True)
    p_orbit = sub.add_parser("orbit", help="trigger.json -> orbit.csv")
    p_orbit.add_argument("dir", type=Path)
    p_map = sub.add_parser("map", help="captures -> surface + path + preflight + shadow")
    p_map.add_argument("dir", type=Path)
    p_plan = sub.add_parser("plan", help="offline compile + preflight on an existing surface.npz")
    p_plan.add_argument("target", type=Path, help="draw dir or surface.npz")
    p_plan.add_argument("--out", type=Path, default=None)
    p_plan.add_argument("--config", type=Path, default=None, help="draw.json (default: beside the surface)")
    args = parser.parse_args(argv)
    try:
        if args.stage == "orbit":
            return stage_orbit(args.dir.expanduser())
        if args.stage == "map":
            return stage_map(args.dir.expanduser())
        return stage_plan(args.target.expanduser(), args.out, args.config)
    except (StageError, dp.DrawRefusal, dk.PlanRefusal, OSError, ValueError, KeyError) as error:
        print(f"draw_stage {args.stage}: {error}", file=sys.stderr)
        if os.environ.get("DRAW_STAGE_TRACE"):
            traceback.print_exc()
        return EXIT_ERROR


if __name__ == "__main__":
    sys.exit(main())
