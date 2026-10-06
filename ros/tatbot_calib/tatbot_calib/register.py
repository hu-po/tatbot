"""Overhead registration from measured wrist-tag corners and shared owner captures.
Robust PnP and leave-one-view-out checks retain hold/bundle/candidate evidence.
Explicit adoption installs the validated registration; Touch travel retains existing landing semantics."""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
from tatbot_bridge import capture
from tatbot_motion.timelaw import axis_rotation


def __getattr__(name):
    return getattr(capture, name)


WORKFLOW = "ros-register"
TARGETS = {"right": "wrist", "left": "wrist_left"}
LABELS = {"right": "pink", "left": "blue"}
VISION = Path("~/tatbot-logs/vision").expanduser()
STACK_CALIB = Path("~/tatbot-ros/calib").expanduser()
# The gates a candidate must pass before --adopt installs it. A 47 mm tag spans about 23 px at 640x360 from
# the installed mount, so a corner is good to a fraction of a pixel and a layout error of a few millimetres
# shows as a pixel or two; one hold must not move the answer more than a drawing tolerance.
MIN_HOLDS = 12
MIN_HOLDS_PER_TAG = 3
MAX_MEDIAN_PX = 2.5
MAX_P95_PX = 6.0
MAX_HOLD_OUT_MM = 5.0
MAX_HOLD_OUT_DEG = 0.5
SETTLE_S = 1.0
SETTLED_RAD = 1e-3      # joints read 0.3 s apart agree this well once a hold has settled (at most 4 s)
# Joints read before and after a capture agree this well while the arm holds: a few counts of the 14-bit joint
# encoders (0.00038 rad a count; a holding arm reads one count apart).
STILL_RAD = 2e-3
# The measured tool is this near its hold once the move has succeeded: within 0.5 mm and 0.1 deg on a run whose
# arm moved, 57-111 mm and 29-70 deg on two whose arm stayed at rest through every hold, a landed arm idling
# until the stack restarted while its Touch moves still succeeded (2026-09-29). Three misses in a row stop the run.
REACHED_M = 0.005
REACHED_DEG = 2.0
MISSES_IN_A_ROW = 3
# Two holds within 3 cm and 15 deg of each other are one view of the tags (choose_holds plans them further apart).
DISTINCT_M = 0.03
DISTINCT_DEG = 15.0
# View planning from a prior camera pose: a tag counts as seen when all its corners project inside the image by
# this margin, its face turns at most this far from the camera, and it spans at least this many pixels.
VIEW_MARGIN_PX = 10.0
VIEW_MAX_ANGLE_DEG = 60.0
VIEW_MIN_SIDE_PX = 14.0


# --- the holds --------------------------------------------------------------------------------------------------
def _rz(deg: float) -> np.ndarray:
    return axis_rotation([0., 0., 1.], math.radians(deg))


def _axis(axis: str, deg: float) -> np.ndarray:
    return axis_rotation([1., 0., 0.] if axis == 'x' else [0., 1., 0.], math.radians(deg))


# Each grid point takes these tool attitudes in turn: (turn about the tool axis from the arm's own heading,
# base axis tilted about, tilt), degrees. The pen sits off the wrist axis, so its heading is far from free: on
# the pink arm at the page only 60 deg either side of its rest heading reaches, with joint margins of 0.5-0.7 rad
# at +-30 deg; a tilt about base x keeps them, one about base y costs them (2026-09-29, register --plan-only).
ATTITUDES = ((0.0, "x", 0.0), (30.0, "x", -20.0), (-30.0, "x", 20.0))


def rest_heading_deg(kin, rest=(0.0, 0.0, 0.0, 0.0, 0.0, 1.5708, 0.0)) -> float:
    """The tool's heading at rest: its x axis's direction in the base's xy plane, degrees."""
    x_axis = kin.fk(np.asarray(rest, float))[:3, 0]
    return math.degrees(math.atan2(x_axis[1], x_axis[0]))


def predicted_views(kin, arm: str, q, camera_from_base, k, dist, target, size) -> dict:
    """Each tag's predicted view at joints q from a prior camera pose: {tag: (seen, face angle deg, side px)}."""
    width, height = size
    out = {}
    for tag in target.ids:
        camera_from_tag = camera_from_base @ kin.frame(np.asarray(q, float), f"{arm}/wrist_tag{tag}")
        to_camera = -camera_from_tag[:3, 3] / np.linalg.norm(camera_from_tag[:3, 3])
        angle = math.degrees(math.acos(float(np.clip(camera_from_tag[:3, 2] @ to_camera, -1.0, 1.0))))
        pixels = _project(camera_from_base, corner_points(kin, arm, q, tag, target.edge_m), k, dist)
        side = float(np.mean(np.linalg.norm(pixels - np.roll(pixels, 1, axis=0), axis=1)))
        inside = bool(np.all((pixels >= VIEW_MARGIN_PX) & (pixels <= np.array([width, height]) - VIEW_MARGIN_PX)))
        out[tag] = (inside and angle <= VIEW_MAX_ANGLE_DEG and side >= VIEW_MIN_SIDE_PX, angle, side)
    return out


VIEW_TURNS = (-40.0, -20.0, 0.0, 20.0, 40.0)
VIEW_TILTS = (("x", 0.0), ("x", 20.0), ("x", -20.0), ("y", 15.0), ("y", -15.0))


REST = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 1.5708, 0.0])   # where a run starts


class Posture:
    """Whether the arm holds a pose as an inspection view must be held (tatbot_session.inspect.upright): the pen
    within 30 deg of straight down, each wrist joint within 40 deg of the pen-down joints over the hold
    (pen_down_reference) and the wrist cube over the tool mount; with `self_gap` (the stack's Guard.self_gap for the
    arm) inspect.VIEW_SELF_MARGIN_M clear of the arm's own bodies. Seeded from rest, the planner's IK took the base
    yaw 80 deg round with the wrist turned back, and 10 of its 27 holds stood within 15 mm of the arm's own bodies,
    one 18 mm into the guard's floor (pink, 2026-10-01)."""

    def __init__(self, kin, ready, table_z: float, self_gap=None):
        from tatbot_session import inspect

        self.kin, self.ready, self.inspect, self.self_gap = kin, ready, inspect, self_gap
        # the table, z up; pen_down_reference turns the pen through every heading itself, so its yaw is free
        self.page = np.eye(4)
        self.page[2, 3] = table_z
        self.cube = inspect.frames_of(kin, REST, inspect.CUBE)
        self.mount = (inspect.frames_of(kin, REST, ("{arm}/tool_mount",)) or [None])[0]
        self.references: dict = {}

    def reference(self, pose: np.ndarray):
        """The pen-down joints over the hold at its height (None where none solves): the planner's IK seed."""
        key = tuple(np.round(pose[:3, 3], 4))
        if key not in self.references:
            self.references[key] = self.inspect.pen_down_reference(
                self.kin, self.ready.solve_ik_seeded, self.page, pose[:2, 3], REST, float(pose[2, 3] - self.page[2, 3]))
        return self.references[key]

    def ok(self, pose: np.ndarray, q) -> bool:
        ref = self.reference(pose)
        if ref is None or not self.inspect.upright(self.kin, q, ref, self.page, self.cube, self.mount):
            return False
        return self.self_gap is None or float(self.self_gap(q)) >= self.inspect.VIEW_SELF_MARGIN_M


def planned_poses(kin, arm: str, prior: dict, target, center_xy, table_z: float, *, heading_deg: float,
                  spread_m: float, heights_m, count: int, ready, posture: Posture) -> tuple[list[np.ndarray], dict]:
    """Holds chosen for what they measure of the chain (choose_holds) from upright candidates (view_candidates),
    ordered as a nearest-neighbour tour from rest. Each hold is solved again from the one before, as the executor
    solves its way there, and one that is not upright then is dropped and said. Returns (the tour, the 1-sigma it
    predicts for the tags' seat and joints 1-4, tatbot_calib.chain.predicted_sigma)."""
    from tatbot_calib import chain

    candidates = view_candidates(kin, arm, prior, target, center_xy, table_z, heading_deg=heading_deg,
                                 spread_m=spread_m, heights_m=heights_m, ready=ready, posture=posture)
    jacobians = {}
    left = choose_holds(kin, arm, prior, target, candidates, count, jacobians)
    tour, here, q = [], kin.fk(REST)[:3, 3], REST
    while left:
        nearest = min(left, key=lambda i: float(np.linalg.norm(candidates[i][0][:3, 3] - here)))
        left.remove(nearest)
        pose = candidates[nearest][0]
        try:
            q_next = ready.solve_ik_seeded(kin, pose, q)
        except ValueError:
            q_next = None
        if q_next is None or not posture.ok(pose, q_next):
            print(f"# planned hold at {np.round(pose[:3, 3], 3).tolist()} dropped: solved from the hold before it, "
                  "it is not upright", flush=True)
            continue
        tour.append(nearest)
        here, q = pose[:3, 3], q_next
    sigma = chain.predicted_sigma([jacobians[i] for i in tour]) if tour else {"dq_mrad": None, "seat_mm": None}
    return [candidates[i][0] for i in tour], sigma


def view_candidates(kin, arm: str, prior: dict, target, center_xy, table_z: float, *, heading_deg: float,
                    spread_m: float, heights_m, ready, posture: Posture) -> list[tuple[np.ndarray, np.ndarray, list]]:
    """Every pose of a 4x4 grid at each height, five turns of the tool about its axis and five tilts that the arm
    holds upright (`posture`, solved from the pen-down joints over it) with 0.15 rad of joint margin and that shows
    at least one tag (predicted_views): (pose, joints, the tags it shows)."""
    down = np.diag([1.0, -1.0, -1.0])
    out = []
    grid = [(dx, dy, h) for dx in np.linspace(-spread_m, spread_m, 4) for dy in np.linspace(-spread_m, spread_m, 4)
            for h in heights_m]
    for (dx, dy, height), turn, (axis, tilt) in itertools.product(grid, VIEW_TURNS, VIEW_TILTS):
        pose = np.eye(4)
        pose[:3, :3] = _axis(axis, tilt) @ _rz(heading_deg + turn) @ down
        pose[:3, 3] = (center_xy[0] + dx, center_xy[1] + dy, table_z + height)
        seed = posture.reference(pose)
        if seed is None:
            continue
        try:
            q = ready.solve_ik_seeded(kin, pose, seed)
        except ValueError:
            continue
        if float(np.min(np.minimum(q[:6] - kin.lower[:6], kin.upper[:6] - q[:6]))) < 0.15 or not posture.ok(pose, q):
            continue
        views = predicted_views(kin, arm, q, prior["camera_from_base"], prior["k"], prior["dist"], target,
                                prior["size"])
        seen = [tag for tag, (ok, _, _) in views.items() if ok]
        if seen:
            out.append((pose, q, seen))
    return out


def choose_holds(kin, arm: str, prior: dict, target, candidates, count: int, jacobians: dict) -> list[int]:
    """Greedily the candidate that most raises the log determinant of what the holds tell the chain
    (tatbot_calib.chain.information: the camera, the tags' seat, joints 1-4, linearised at the prior's camera),
    among those at least DISTINCT_M or DISTINCT_DEG from every hold chosen. Today's registrations, planned for the
    tags each hold shows, would pin joints 1-4 to 3-6 mrad at 0.8 px; chosen this way 40 holds over the same
    candidates pin them to 1.2-2.0 (pink, 2026-10-01). Returns candidate indices; `jacobians` keeps each one's rows."""
    from tatbot_calib import chain

    c = chain.Chain(kin, arm, target)
    for i, (_, q, seen) in enumerate(candidates):
        jacobians[i] = chain.information(c, prior["k"], prior["dist"], prior["camera_from_base"], q, seen)
    chosen = []
    info = np.eye(6 + 6 + len(chain.FREE_JOINTS)) * 1e-9

    def distinct(i):
        pose = candidates[i][0]
        return all(np.linalg.norm(pose[:3, 3] - candidates[j][0][:3, 3]) >= DISTINCT_M
                   or _angle_deg(pose, candidates[j][0]) >= DISTINCT_DEG for j in chosen)

    while len(chosen) < count:
        best = max((i for i in jacobians if i not in chosen and distinct(i)),
                   key=lambda i: np.linalg.slogdet(info + jacobians[i].T @ jacobians[i])[1], default=None)
        if best is None:
            break
        chosen.append(best)
        info = info + jacobians[best].T @ jacobians[best]
    return chosen


def run_path(value: str) -> Path:
    """An earlier ros-register run's directory, from the directory or its run id."""
    run = Path(value).expanduser()
    return run if run.is_dir() else Path("~/tatbot-logs").expanduser() / WORKFLOW / value


def load_prior(value: str) -> dict:
    """A prior camera pose from an earlier ros-register run (its directory or run id): its fit and bundle."""
    run = run_path(value)
    report = json.loads((run / "report.json").read_text())
    optics = json.loads((run / "calibration.json").read_text())["cameras"][capture.COLOR]
    k = np.array([[optics["intrinsics"]["fx"], 0, optics["intrinsics"]["cx"]],
                  [0, optics["intrinsics"]["fy"], optics["intrinsics"]["cy"]], [0, 0, 1]], float)
    return {"camera_from_base": np.asarray(report["camera_from_base"], float), "k": k, "run": run.name,
            "dist": np.asarray(optics["distortion"]["coefficients"], float),
            "size": (optics["intrinsics"]["width"], optics["intrinsics"]["height"])}


def hold_poses(center_xy, table_z: float, *, heading_deg: float = 0.0, spread_m: float = 0.07,
               heights_m=(0.12, 0.17)) -> list[np.ndarray]:
    """TCP poses in the arm base: a 3x3 grid around `center_xy`, alternately at each height over the table,
    the tool pointing down (URDF rpy pi 0 yaw) at the arm's heading and turned or tilted a little, three
    attitudes per point, visited in a serpentine so consecutive holds are near each other."""
    down = np.diag([1.0, -1.0, -1.0])   # rpy (pi, 0, 0): the tool's z along -z of the base
    poses = []
    offsets = (-spread_m, 0.0, spread_m)
    for row, dx in enumerate(offsets):
        for col, dy in enumerate(offsets if row % 2 == 0 else offsets[::-1]):
            height = heights_m[(row * 3 + col) % len(heights_m)]
            for turn, axis, tilt in ATTITUDES:
                pose = np.eye(4)
                pose[:3, :3] = _axis(axis, tilt) @ _rz(heading_deg + turn) @ down
                pose[:3, 3] = (center_xy[0] + dx, center_xy[1] + dy, table_z + height)
                poses.append(pose)
    return poses


# --- the camera -------------------------------------------------------------------------------------------------


# --- the bundle -------------------------------------------------------------------------------------------------
MODELS = {"BrownConrady": "brown_conrady", "None": "none"}


def draft_bundle(metadata: dict, depth_profile: dict | None) -> dict:
    """A D555-only camera bundle in the colour camera's own frame: both entries at identity with the colour
    optics the owner aligned depth to, as `finalize-calibration` expects (bundle_id left empty)."""
    intrinsics = capture.intrinsics_of(metadata)
    model = MODELS.get(intrinsics["distortion_model"])
    if model is None:
        raise ValueError(f"{capture.COLOR}: unsupported distortion model {intrinsics['distortion_model']!r}")
    optics = {"width": int(intrinsics["width"]), "height": int(intrinsics["height"]),
              "fx": intrinsics["fx"], "fy": intrinsics["fy"], "cx": intrinsics["ppx"], "cy": intrinsics["ppy"]}
    identity = {"rotation": [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0], "translation_m": [0.0, 0.0, 0.0]}
    serial = str(metadata["attributes"].get("device_serial", ""))
    note = {"method": "d555_native_optics", "world": "the D555 colour optical frame",
            "device_serial": serial}
    # The bus carries colour as JPEG and says so by relabelling the profile (rgb8) and keeping the camera's own
    # format in bus_source_format; the bundle binds the stream the owner opens, so it takes the camera's own.
    color_profile = dict(metadata["profile"], format=metadata["attributes"].get("bus_source_format",
                                                                               metadata["profile"]["format"]))
    depth = dict(depth_profile or {**color_profile, "stream": "depth", "format": "z16"})

    def entry(name, profile, extra):
        return {"sensor_name": name, "profile": profile, "intrinsics": dict(optics),
                "distortion": {"model": model, "coefficients": list(intrinsics["distortion_coefficients"])},
                "world_from_camera": identity, "depth_to_color": None, "metadata": {**note, **extra}}

    return {"schema_version": 1, "bundle_id": "", "world_frame": capture.WORLD_FRAME,
            "cameras": {capture.COLOR: entry(capture.COLOR, color_profile, {}),
                        capture.DEPTH: entry(capture.DEPTH, depth, {"geometry": f"depth_aligned_to_{capture.COLOR}"})}}


def reusable(bundle: dict, draft: dict) -> bool:
    """An installed D555 bundle still describes the camera: same world, profiles, lens model, coefficients and
    device (thermal focal drift is not a new camera; docs/vision.md)."""
    try:
        if bundle.get("world_frame") != capture.WORLD_FRAME or set(bundle["cameras"]) != {capture.COLOR, capture.DEPTH}:
            return False
        for name in (capture.COLOR, capture.DEPTH):
            have, want = bundle["cameras"][name], draft["cameras"][name]
            if (have["profile"] != want["profile"] or have["distortion"]["model"] != want["distortion"]["model"]
                    or not np.allclose(have["distortion"]["coefficients"], want["distortion"]["coefficients"],
                                       rtol=0, atol=1e-6)
                    or (have["intrinsics"]["width"], have["intrinsics"]["height"])
                    != (want["intrinsics"]["width"], want["intrinsics"]["height"])
                    or have.get("metadata", {}).get("device_serial") != want["metadata"]["device_serial"]):
                return False
        return True
    except (KeyError, TypeError, ValueError):
        return False


def visiond_binary() -> Path:
    """The tatbot-visiond that stamps bundle ids: $TATBOT_VISIOND, else the newest release on this node."""
    if os.environ.get("TATBOT_VISIOND"):
        return Path(os.environ["TATBOT_VISIOND"]).expanduser()
    releases = sorted(Path("~/.local/share/tatbot/releases").expanduser().glob("*/bin/tatbot-visiond"),
                      key=lambda p: p.stat().st_mtime)
    if not releases:
        raise RuntimeError("no tatbot-visiond release on this node to stamp the bundle id (set TATBOT_VISIOND)")
    return releases[-1]


def finalize(draft: dict, run_dir: Path) -> dict:
    (run_dir / "bundle-draft.json").write_text(json.dumps(draft, indent=2) + "\n")
    visiond = visiond_binary()
    final = run_dir / "calibration.json"
    subprocess.run([str(visiond), "finalize-calibration", str(run_dir / "bundle-draft.json"), "--output", str(final)],
                   check=True)
    subprocess.run([str(visiond), "validate-calibration", str(final)], check=True)
    return json.loads(final.read_text())


# --- the solve --------------------------------------------------------------------------------------------------
def corner_points(kin, arm: str, q, tag_id: int, edge_m: float) -> np.ndarray:
    """The tag's four corners (TL, TR, BR, BL; fiducials.geometry) in the arm base at joints q."""
    from fiducials import tag_model_corners

    base_from_tag = kin.frame(np.asarray(q, float), f"{arm}/wrist_tag{tag_id}")
    model = np.c_[tag_model_corners(edge_m), np.ones(4)]
    return (base_from_tag @ model.T).T[:, :3]


def _project(camera_from_base: np.ndarray, points: np.ndarray, k, dist) -> np.ndarray:
    import cv2

    rvec, _ = cv2.Rodrigues(camera_from_base[:3, :3])
    pixels, _ = cv2.projectPoints(points.reshape(-1, 1, 3), rvec, camera_from_base[:3, 3], k, dist)
    return pixels.reshape(-1, 2)


def _pnp(points: np.ndarray, pixels: np.ndarray, k, dist, guess: np.ndarray | None = None) -> np.ndarray:
    import cv2

    if guess is None:
        ok, rvec, tvec = cv2.solvePnP(points, pixels, k, dist, flags=cv2.SOLVEPNP_SQPNP)
    else:
        rvec0, _ = cv2.Rodrigues(guess[:3, :3])
        ok, rvec, tvec = cv2.solvePnP(points, pixels, k, dist, rvec0.copy(), guess[:3, 3].reshape(3, 1).copy(),
                                      useExtrinsicGuess=True, flags=cv2.SOLVEPNP_ITERATIVE)
    if not ok:
        raise RuntimeError("PnP found no camera pose")
    rvec, tvec = cv2.solvePnPRefineLM(points, pixels, k, dist, rvec, tvec)
    out = np.eye(4)
    out[:3, :3] = cv2.Rodrigues(rvec)[0]
    out[:3, 3] = tvec.ravel()
    return out


def solve(sightings: list[dict], k, dist) -> tuple[np.ndarray, list[dict]]:
    """camera <- arm base from every sighting's four corners; a sighting whose worst corner lies past
    max(3 px, 4x the median) is dropped and the fit repeated until none is. Returns (camera_from_base, kept)."""
    kept = list(sightings)
    camera_from_base = None
    for _ in range(10):
        if len(kept) < 3:
            raise RuntimeError(f"{len(kept)} tag sightings are too few to place the camera")
        points = np.concatenate([s["points"] for s in kept])
        pixels = np.concatenate([s["pixels"] for s in kept])
        camera_from_base = _pnp(points, pixels, k, dist, camera_from_base)
        worst = np.array([np.max(np.linalg.norm(_project(camera_from_base, s["points"], k, dist) - s["pixels"],
                                                axis=1)) for s in kept])
        limit = max(3.0, 4.0 * float(np.median(worst)))
        if not np.any(worst > limit):
            return camera_from_base, kept
        kept = [s for s, w in zip(kept, worst, strict=True) if w <= limit]
    return camera_from_base, kept


def residuals(camera_from_base: np.ndarray, sightings: list[dict], k, dist) -> np.ndarray:
    return np.concatenate([np.linalg.norm(_project(camera_from_base, s["points"], k, dist) - s["pixels"], axis=1)
                           for s in sightings])


def _angle_deg(a: np.ndarray, b: np.ndarray) -> float:
    r = a[:3, :3].T @ b[:3, :3]
    return math.degrees(math.acos(max(-1.0, min(1.0, (np.trace(r) - 1.0) / 2.0))))


def off_hold(kin, q, pose: np.ndarray) -> str | None:
    """Why the arm at joints q is not at its hold `pose` (the measured tool over REACHED_M or REACHED_DEG from it),
    else None."""
    tcp = kin.fk(np.asarray(q, float))
    mm, deg = float(np.linalg.norm(tcp[:3, 3] - pose[:3, 3])) * 1000.0, _angle_deg(tcp, pose)
    if mm <= REACHED_M * 1000.0 and deg <= REACHED_DEG:
        return None
    return f"the arm is {mm:.1f} mm and {deg:.1f} deg from the hold after a move that succeeded"


def pose_views(poses) -> list[int]:
    """Each pose's view: that of the first earlier pose within DISTINCT_M and DISTINCT_DEG of it, else a new one."""
    firsts, views = [], []
    for pose in poses:
        view = next((i for i, first in enumerate(firsts) if np.linalg.norm(pose[:3, 3] - first[:3, 3]) < DISTINCT_M
                     and _angle_deg(pose, first) < DISTINCT_DEG), len(firsts))
        if view == len(firsts):
            firsts.append(pose)
        views.append(view)
    return views


def _view(sighting: dict) -> int:
    return sighting.get("view", sighting["hold"])


def hold_out(sightings: list[dict], camera_from_base: np.ndarray, k, dist) -> dict:
    """Each view left out in turn (a sighting's "view" from fit's pose_views, else its hold): how far its absence
    moves the arm base in the world (mm, deg)."""
    world_from_base = camera_from_base
    moves = []
    for view in sorted({_view(s) for s in sightings}):
        rest = [s for s in sightings if _view(s) != view]
        if len({_view(s) for s in rest}) < MIN_HOLDS - 1:
            continue
        other, _ = solve(rest, k, dist)
        moves.append({"view": view, "holds": sorted({s["hold"] for s in sightings if _view(s) == view}),
                      "mm": float(np.linalg.norm(other[:3, 3] - world_from_base[:3, 3]) * 1000),
                      "deg": _angle_deg(other, world_from_base)})
    return {"views": len(moves), "max_mm": max((m["mm"] for m in moves), default=None),
            "max_deg": max((m["deg"] for m in moves), default=None), "each": moves}


def gates(report: dict) -> list[str]:
    fit, out = report["fit"], report["hold_out"]
    why = []
    if fit["poses"] < MIN_HOLDS:
        why.append(f"{fit['holds']} holds at {fit['poses']} distinct poses saw tags (need {MIN_HOLDS} poses, "
                   f"{DISTINCT_M * 100:.0f} cm or {DISTINCT_DEG:.0f} deg apart)")
    thin = [tag for tag, row in report["per_tag"].items() if 0 < row["poses"] < MIN_HOLDS_PER_TAG]
    if thin:
        why.append(f"tags {thin} were each seen at fewer than {MIN_HOLDS_PER_TAG} distinct poses")
    if fit["median_px"] > MAX_MEDIAN_PX or fit["p95_px"] > MAX_P95_PX:
        why.append(f"corner residual median {fit['median_px']:.2f} px / p95 {fit['p95_px']:.2f} px "
                   f"(at most {MAX_MEDIAN_PX} / {MAX_P95_PX})")
    if out["max_mm"] is None or out["max_mm"] > MAX_HOLD_OUT_MM or out["max_deg"] > MAX_HOLD_OUT_DEG:
        why.append(f"one view moves the registration {out['max_mm']} mm / {out['max_deg']} deg "
                   f"(at most {MAX_HOLD_OUT_MM} / {MAX_HOLD_OUT_DEG})")
    return why


def registration_record(arm: str, bundle: dict, bundle_path: Path, world_from_base: np.ndarray,
                        root_from_base: np.ndarray, report: dict, run_id: str) -> dict:
    world_from_root = world_from_base @ np.linalg.inv(root_from_base)
    fit = report["fit"]
    return {"schema": "tatbot.arm-registration/1", "arm": arm, "arm_label": LABELS.get(arm, arm),
            "calibration_id": bundle["bundle_id"],
            "camera_bundle_sha256": hashlib.sha256(bundle_path.read_bytes()).hexdigest(),
            "camera_bundle_path": str(bundle_path),
            "frames": {"world_from_root": "D555 colour optical frame <- URDF root (historical key world_from_base)",
                       "world_from_arm_base": f"D555 colour optical frame <- {arm}/base_link"},
            "world_from_root": world_from_root.tolist(), "root_from_arm_base": root_from_base.tolist(),
            "world_from_arm_base": world_from_base.tolist(),
            "fit": {"observations": fit["sightings"], "distinct_poses": fit["poses"], "median_px": fit["median_px"],
                    "max_px": fit["max_px"], "p95_px": fit["p95_px"]},
            "withheld": {"observations": 0, "median_px": None, "max_px": None},
            "hold_out": {k: v for k, v in report["hold_out"].items() if k != "each"},
            "per_camera": {capture.COLOR: {"fit_observations": fit["sightings"], "median_px": fit["median_px"],
                                   "max_px": fit["max_px"]}},
            "per_tag": report["per_tag"], "leave_one_camera_out": {}, "camera_calibration_frozen": True,
            "rig": {"capture": run_id, "note": "one fixed camera (the D555) at its own frame; the measured wrist "
                                               "tag layout held; the camera's optics as it reports them"}}


def robot_world_record(registration: dict, kin_link: str, layout: dict) -> dict:
    """robot-world-current.json for the right arm: the URDF root in the world (historical key world_from_base)."""
    return {"world_from_base": registration["world_from_root"], "calibration_id": registration["calibration_id"],
            "link": kin_link, "link_from_tag": layout, "mode": "d555_wrist_tags",
            "observations": registration["fit"]["observations"],
            "corner_px_median": round(float(registration["fit"]["median_px"]), 3)}


# --- the run ----------------------------------------------------------------------------------------------------
def install(path: Path, destination: Path, replaced: Path) -> None:
    """Install `path` as `destination`, keeping what it replaces under `replaced`."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        replaced.mkdir(parents=True, exist_ok=True)
        shutil.copy2(destination, replaced / destination.name)
    temporary = destination.with_suffix(destination.suffix + ".next")
    shutil.copy2(path, temporary)
    temporary.replace(destination)


def adopt(run_dir: Path, arm: str, bundle: dict, reused_bundle: bool) -> list[str]:
    """The bundle, this arm's registration and (right arm) the robot-world golden where consumers read them. A
    new bundle is another world: the other arm's registration, solved in the old one, is moved aside."""
    replaced = run_dir / "replaced"
    done = []
    if not reused_bundle:
        install(run_dir / "calibration.json", VISION / "calibration-current.json", replaced)
        done.append(str(VISION / "calibration-current.json"))
    for directory in (VISION, STACK_CALIB):
        install(run_dir / f"arm-registration-{arm}.json", directory / f"arm-registration-{arm}-current.json",
                replaced / directory.name)
        done.append(str(directory / f"arm-registration-{arm}-current.json"))
        for other in directory.glob("arm-registration-*-current.json"):
            if other.name == f"arm-registration-{arm}-current.json":
                continue
            try:
                stale = json.loads(other.read_text()).get("calibration_id") != bundle["bundle_id"]
            except (OSError, ValueError):
                stale = True
            if stale:
                (replaced / directory.name).mkdir(parents=True, exist_ok=True)
                shutil.move(str(other), str(replaced / directory.name / other.name))
                done.append(f"moved aside (another camera world): {other}")
    if arm == "right":
        install(run_dir / "robot-world.json", VISION / "robot-world-current.json", replaced)
        done.append(str(VISION / "robot-world-current.json"))
    return done


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="register", description=__doc__.split("\n\n")[0])
    parser.add_argument("--arm", default="right", choices=sorted(TARGETS))
    parser.add_argument("--center", type=float, nargs=2, metavar=("X", "Y"), default=None,
                        help="the grid's centre in <arm>/base_link, m (default: the configured page's centre)")
    parser.add_argument("--table-z", dest="table_z", type=float, default=None,
                        help="the table's height in <arm>/base_link, m (default: the configured page's)")
    parser.add_argument("--spread", type=float, default=0.07, help="grid half-width, m")
    parser.add_argument("--heights", type=float, nargs="+", default=[0.12, 0.17], help="tool heights over the table, m")
    parser.add_argument("--max-holds", dest="max_holds", type=int, default=0, help="stop after this many (0: all)")
    parser.add_argument("--prior", default="",
                        help="an earlier ros-register run (directory or run id) whose fit plans the holds: upright "
                             "ones, chosen for what they measure of the camera, the tags' seat and joints 1-4")
    parser.add_argument("--holds", type=int, default=27, help="with --prior: how many holds to plan")
    parser.add_argument("--adopt", action="store_true", help="install the result when it passes the gates")
    parser.add_argument("--dry-run", dest="dry_run", action="store_true",
                        help="plan and reach-check the holds, move nothing")
    ns = parser.parse_args(argv)

    from tatbot_description import names, repo_root, robot_description
    repo = repo_root()
    capture._lib(repo)
    import tatbot_runlog
    from fiducials import load_inventory
    from fiducials.detector import DetectorConfig, FiducialDetector
    from tatbot_motion import Kinematics
    from tatbot_session import ready

    from tatbot_calib.cli import _stack

    stack = _stack(repo)
    fixed = stack["page"]["fixed"][ns.arm]["xyz"]
    center = ns.center or fixed[:2]
    table_z = fixed[2] if ns.table_z is None else ns.table_z
    kin = Kinematics(robot_description(None, arms=(ns.arm,)), ns.arm)   # the nominal mount: world = URDF root
    root_from_base = np.linalg.inv(kin.frame(np.zeros(7), names.WORLD))
    inventory = load_inventory(repo / "config" / "fiducials.json")
    target = inventory.target(TARGETS[ns.arm])
    detector = FiducialDetector.from_inventory(inventory, DetectorConfig(min_side_px=8.0, refinement="edges"),
                                               target=TARGETS[ns.arm])
    if ns.prior:
        from tatbot_motion import load_motion
        from tatbot_motion.collision import Guard

        prior = load_prior(ns.prior)
        guard, text = Guard.from_stack(repo, stack, load_motion())
        print(f"# {text}", flush=True)
        posture = Posture(kin, ready, table_z, (lambda q: guard.self_gap(ns.arm, q)) if guard is not None else None)
        poses, sigma = planned_poses(kin, ns.arm, prior, target, center, table_z, heading_deg=rest_heading_deg(kin),
                                     spread_m=ns.spread, heights_m=tuple(ns.heights), count=ns.holds, ready=ready,
                                     posture=posture)
        print(f"# {len(poses)} upright holds planned from the fit of run {prior['run']}"
              + ("" if guard is not None else ", their self clearance unchecked (no guard)")
              + f"; at 0.8 px they would pin joints 1-4 to {sigma['dq_mrad']} mrad and the tags' seat to "
              f"{sigma['seat_mm']} mm", flush=True)
    else:
        poses = hold_poses(center, table_z, heading_deg=rest_heading_deg(kin), spread_m=ns.spread,
                           heights_m=tuple(ns.heights))
    if ns.max_holds:
        poses = poses[:ns.max_holds]

    run = tatbot_runlog.init(WORKFLOW, meta={"arm": ns.arm, "holds": len(poses), "dry_run": ns.dry_run},
                             attach_logging=False, argv=["tatbot_calib", "register", *(argv or sys.argv[1:])])
    run_dir = Path(run.dir)
    (run_dir / "frames").mkdir(exist_ok=True)
    try:
        code = _run(ns, run_dir, repo, kin, root_from_base, target, detector, poses, center, table_z, ready)
    except BaseException:
        run.finalize(1, status="fail")
        raise
    run.finalize(code, status="ok" if code == 0 else "fail")
    return code


def _run(ns, run_dir: Path, repo: Path, kin, root_from_base, target, detector, poses, center, table_z, ready) -> int:
    print(f"# {len(poses)} holds around ({center[0]:.3f}, {center[1]:.3f}) over table z {table_z:.4f} in "
          f"{ns.arm}/base_link; tags {list(target.ids)} at {target.edge_m * 1000:.0f} mm; run {run_dir}", flush=True)
    if ns.dry_run:
        plan_only(kin, poses, ready)
        return 0
    sightings, first, rows = capture_holds(ns.arm, repo, kin, target, detector, poses, run_dir)
    if first is None or len({s["hold"] for s in sightings}) < 3:
        missed = sum(1 for row in rows if not row["reached"])
        why = ["fewer than 3 holds saw a wrist tag", *([f"{missed} of {len(rows)} holds were not reached"] * bool(missed))]
        print(json.dumps({"ok": False, "run_dir": str(run_dir), "why": why}))
        return 1
    bundle, reused = bundle_for(first, run_dir)
    report = fit(ns.arm, bundle, reused, sightings, target)
    record = registration_record(ns.arm, bundle, run_dir / "calibration.json",
                                 np.asarray(report["camera_from_base"]), root_from_base, report, run_dir.name)
    write_outputs(ns.arm, run_dir, kin, target, record)
    report["adopted"] = adopt(run_dir, ns.arm, bundle, reused) if ns.adopt and not report["refused"] else []
    report["chain"] = chain_report(ns.arm, kin, target, bundle, rows, report, run_dir)
    (run_dir / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    summarize(report, run_dir)
    return 0 if not report["refused"] else 3


def plan_only(kin, poses, ready) -> None:
    """Reach-check the holds from rest, the way the stack's joint fallback solves them; move nothing."""
    q = REST
    for i, pose in enumerate(poses):
        try:
            q = ready.solve_ik_seeded(kin, pose, q)
            margin = float(np.min(np.minimum(q[:6] - kin.lower[:6], kin.upper[:6] - q[:6])))
            print(f"hold {i:02d} xyz {np.round(pose[:3, 3], 3).tolist()} reach ok, joint margin {margin:.3f} rad")
        except ValueError as error:
            print(f"hold {i:02d} xyz {np.round(pose[:3, 3], 3).tolist()} unreachable: {error}")


def capture_holds(arm: str, repo: Path, kin, target, detector, poses,
                  run_dir: Path) -> tuple[list[dict], dict | None, list[dict]]:
    """Drive the holds, capture each the arm reached, detect the arm's tags; the arm lands at the end. Returns
    (sightings, the first capture, the holds' rows) and writes holds.jsonl and each captured hold's frame."""
    import cv2

    from tatbot_calib.program import Rig
    rig, camera = Rig(arm), capture.Camera(repo)
    sightings, rows, first, misses = [], [], None, 0
    try:
        for i, pose in enumerate(poses):
            try:
                rig.goal(pose, move=True, guard=0)
            except RuntimeError as error:
                print(f"hold {i:02d}: not reached ({error}); skipped", flush=True)
                rows.append({"hold": i, "reached": False, "error": str(error)})
                continue
            time.sleep(SETTLE_S)
            q_before = settled(rig)
            missed = off_hold(kin, q_before, pose)
            misses = misses + 1 if missed else 0
            if missed:
                print(f"hold {i:02d}: {missed}; skipped", flush=True)
                rows.append({"hold": i, "reached": False, "error": missed, "q": q_before.tolist()})
                if misses >= MISSES_IN_A_ROW:
                    print(f"# {misses} holds in a row not reached: the stack is not moving the arm (a landed arm "
                          "idles until it is woken); stopping", flush=True)
                    break
                continue
            shot = camera.capture(time.time_ns())
            q_after = rig.joints()
            first = first or shot
            cv2.imwrite(str(run_dir / "frames" / f"hold-{i:02d}.jpg"), shot["image"], [cv2.IMWRITE_JPEG_QUALITY, 95])
            rows.append(hold_row(i, pose, q_before, q_after, shot, detector.detect(capture.COLOR, shot["image"], shot["stamp_ns"])))
            sightings += hold_sightings(rows[-1], kin, arm, target)
    finally:
        with (run_dir / "holds.jsonl").open("w") as fh:
            fh.writelines(json.dumps(row) + "\n" for row in rows)
        try:
            print(f"# landed: {rig.land()}", flush=True)
        except RuntimeError as error:
            print(f"# landing: {error}", flush=True)
        rig.close()
        camera.close()
    return sightings, first, rows


def settled(rig, timeout_s: float = 4.0) -> np.ndarray:
    """The joints once two reads 0.3 s apart agree within SETTLED_RAD (the newest read at the timeout)."""
    q = rig.joints()
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        time.sleep(0.3)
        again = rig.joints()
        if float(np.max(np.abs(again - q))) <= SETTLED_RAD:
            return again
        q = again
    return q


def hold_row(i: int, pose, q_before, q_after, shot: dict, detections) -> dict:
    still = float(np.max(np.abs(q_after - q_before)))
    return {"hold": i, "reached": True, "tcp_target": pose.tolist(), "q": ((q_before + q_after) / 2.0).tolist(),
            "still_rad": still, "stamp_ns": shot["stamp_ns"], "intrinsics": capture.intrinsics_of(shot["metadata"]),
            "calibration_id": shot["metadata"].get("calibration_id"),
            "tags": {str(d.tag_id): d.corners_px.tolist() for d in detections}}


def hold_sightings(row: dict, kin, arm: str, target) -> list[dict]:
    """One hold's tag sightings, or none when the arm moved during its capture."""
    if row["still_rad"] > STILL_RAD:
        print(f"hold {row['hold']:02d}: the arm moved {row['still_rad']:.5f} rad during the capture; not used",
              flush=True)
        return []
    print(f"hold {row['hold']:02d}: tags {sorted(int(t) for t in row['tags'])}", flush=True)
    tcp = kin.fk(np.asarray(row["q"], float))
    return [{"hold": row["hold"], "tag": int(tag), "pixels": np.asarray(corners, float), "tcp": tcp,
             "points": corner_points(kin, arm, row["q"], int(tag), target.edge_m)} for tag, corners in row["tags"].items()]


def bundle_for(first: dict, run_dir: Path) -> tuple[dict, bool]:
    """The installed D555 bundle while it still describes the camera, else a new one from this run's optics."""
    import contextlib

    draft = draft_bundle(first["metadata"], first["depth_profile"])
    current = None
    with contextlib.suppress(OSError, ValueError):
        current = json.loads((VISION / "calibration-current.json").read_text())
    if current is not None and reusable(current, draft):
        (run_dir / "calibration.json").write_text(json.dumps(current, indent=2) + "\n")
        return current, True
    return finalize(draft, run_dir), False


def fit(arm: str, bundle: dict, reused: bool, sightings: list[dict], target) -> dict:
    optics = bundle["cameras"][capture.COLOR]
    k = np.array([[optics["intrinsics"]["fx"], 0, optics["intrinsics"]["cx"]],
                  [0, optics["intrinsics"]["fy"], optics["intrinsics"]["cy"]], [0, 0, 1]], float)
    dist = np.asarray(optics["distortion"]["coefficients"], float)
    sightings = [dict(s, view=view) for s, view in zip(sightings, pose_views([s["tcp"] for s in sightings]),
                                                       strict=True)]
    camera_from_base, kept = solve(sightings, k, dist)
    errors = residuals(camera_from_base, kept, k, dist)
    per_tag = {}
    for tag in target.ids:
        rows_t = [s for s in kept if s["tag"] == tag]
        err_t = residuals(camera_from_base, rows_t, k, dist) if rows_t else np.array([])
        per_tag[str(tag)] = {"sightings": len(rows_t), "holds": len({s["hold"] for s in rows_t}),
                             "poses": len({s["view"] for s in rows_t}),
                             "median_px": float(np.median(err_t)) if len(err_t) else None}
    report = {"arm": arm, "bundle_id": bundle["bundle_id"], "bundle_reused": reused,
              "fit": {"sightings": len(kept), "dropped": len(sightings) - len(kept),
                      "holds": len({s["hold"] for s in kept}), "poses": len({s["view"] for s in kept}),
                      "median_px": float(np.median(errors)),
                      "p95_px": float(np.percentile(errors, 95)), "max_px": float(np.max(errors))},
              "per_tag": per_tag, "hold_out": hold_out(kept, camera_from_base, k, dist),
              "camera_from_base": camera_from_base.tolist()}
    report["refused"] = gates(report)
    return report


def chain_report(arm: str, kin, target, bundle: dict, rows, report: dict, run_dir: Path) -> dict:
    """What the rigid fit leaves, modelled (tatbot_calib.chain): the tags' seat and the joints' offsets fitted to
    this run's holds and each scored on the holds it was not given; the seat's layout candidate goes to
    wrist-layout.json. Reported only: it gates and adopts nothing, and a failure in it costs the registration nothing."""
    from tatbot_calib import chain

    optics = bundle["cameras"][capture.COLOR]
    k = np.array([[optics["intrinsics"]["fx"], 0, optics["intrinsics"]["cx"]],
                  [0, optics["intrinsics"]["fy"], optics["intrinsics"]["cy"]], [0, 0, 1]], float)
    run = chain.Run(run_dir.name, k, np.asarray(optics["distortion"]["coefficients"], float), chain.sightings_of(rows),
                    np.asarray(report["camera_from_base"], float))
    try:
        return chain.report_for(chain.Chain(kin, arm, target), [run], run_dir)
    except (ValueError, RuntimeError, np.linalg.LinAlgError) as error:
        return {"error": f"{type(error).__name__}: {error}"}


def write_outputs(arm: str, run_dir: Path, kin, target, record: dict) -> None:
    (run_dir / f"arm-registration-{arm}.json").write_text(json.dumps(record, indent=2) + "\n")
    if arm != "right":
        return
    zero = np.zeros(7)
    layout = {str(t): np.linalg.solve(kin.frame(zero, target.parent_frame),
                                      kin.frame(zero, f"{arm}/wrist_tag{t}")).tolist() for t in target.ids}
    (run_dir / "robot-world.json").write_text(json.dumps(robot_world_record(record, target.parent_frame, layout),
                                                         indent=2) + "\n")


def summarize(report: dict, run_dir: Path) -> None:
    fit_, out = report["fit"], report["hold_out"]
    moved = "n/a" if out["max_mm"] is None else f"{out['max_mm']:.2f} mm / {out['max_deg']:.3f} deg"
    print(f"# fit: {fit_['holds']} holds at {fit_['poses']} distinct poses, {fit_['sightings']} tag sightings "
          f"({fit_['dropped']} dropped); corners median {fit_['median_px']:.2f} px, p95 {fit_['p95_px']:.2f} px; "
          f"one view moves it at most {moved}", flush=True)
    if report.get("chain", {}).get("error"):
        print(f"# chain: not modelled ({report['chain']['error']})", flush=True)
    elif report.get("chain"):
        from tatbot_calib import chain

        for line in chain.describe(report["chain"]):
            print(line, flush=True)
    for why in report["refused"]:
        print(f"# refused: {why}", flush=True)
    for line in report["adopted"]:
        print(f"# adopted: {line}", flush=True)
    print(json.dumps({"ok": not report["refused"], "run_dir": str(run_dir), "bundle_id": report["bundle_id"],
                      "adopted": bool(report["adopted"])}))


if __name__ == "__main__":
    sys.exit(main())
