"""The fitted tool's tip, contact-free: joint 6 turned under the palette camera (ros/README.md section 7).

The arm parks the tool upright PARK_OVER_M over the station's ball, at the yaw whose turns the camera sees best, and
holds there for small known translations (they give the camera its scale, orientation and depth) and for turns of
joint 6 alone. At each hold the operator's CLI takes a still on the palette camera's node (`ask_still`); the pen's
working tip is found in it, and the ball, which stands still, marks the camera's drift. One least-squares fit in the
image, a pinhole at the nominal focal length and the tip across its axis, then explains every still.

The pen sits 45 degrees off joint 6's axis, so a turn swings the tip on a circle of about 12 mm and tilts the pen. A
turn sees the tip's two components across that axis and none along it: with the tip's length held at the installed
one, a length error (or the detector's bias along the pen) reads in the fitted tip one for one along the factor
report.md gives. Each turn is a Touch MODE_MOVE to FK of the park's planned joints with joint 6 turned, so joints 1-5
stay where the park put them (to the planner's few mrad) and their offsets do not enter; the jog client would
re-command them at their measured, sagged values, with neither the probe guard nor the two-arm one. Every hold is
a probe-guarded move through Calibration.way_check, the tool's envelope off the station's parts on its way and
BALL_CLEAR_M off the fix's ball; the turns go only when the park's still shows the tip TURN_CLEAR_M over the ball's
top under the lowest turn. The run writes holds.jsonl, stills/, candidate.json and report.md; `tatbot ros calib
apply --run ID` adopts the tip.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation
from tatbot_description.transforms import rpy_matrix

from tatbot_calib import halo, program, reach, station
from tatbot_calib.tool import ToolRefusedError, fitted_tool

PARK_OVER_M = 0.006       # the tip over the fix's ball top at the park; the fix is good to 2 mm in height and the turns
                          # take the tip down up to 1.4 mm
SHIFT_M = 0.003           # the translations: across the camera's line of sight both ways, up, and along it
TURNS_DEG = tuple(range(-30, 31, 6))
YAWS_DEG = tuple(range(-180, 180, 15))
SETTLE_S = 1.0
BALL_CLEAR_M = 0.003      # planned: each hold keeps the tool's envelope this far off the fix's ball
TURN_CLEAR_M = 0.002      # measured: the park's still must leave the lowest turn's tip this far over the ball's top
# The palette camera's stills (rpicam-still at full resolution) and focal length: the 2 mm ball spans 125 px from
# 122 mm (2026-10-02). The fit takes the camera's depth from the translations, so this only sizes the perspective.
SIZE = (9248, 6944)
FOCAL_PX = 7560.0
LENS_FROM_OPTICAL = np.array([[0.0, 0.0, 1.0], [-1.0, 0.0, 0.0], [0.0, -1.0, 0.0]])   # x forward, z up -> z forward
# The detector's strip about the predicted tip: the body (4 mm across) above it, the background beside it, and no
# more under it than the ball's top lies (PARK_OVER_M less the prediction's millimetre).
UP_M, DOWN_M = 0.009, 0.0025
CONE_HALF_M = 0.0009      # the metal tip is 1.6 mm across where it leaves the body


@dataclass
class Camera:
    """A pinhole: base -> optical frame (R, T), focal length and principal point in pixels."""
    R: np.ndarray
    T: np.ndarray
    f: float
    c: np.ndarray

    def project(self, points) -> np.ndarray:
        p = np.atleast_2d(points) @ self.R.T + self.T
        return self.f * p[:, :2] / p[:, 2:3] + self.c

    def depth(self, point) -> float:
        return float((self.R @ point + self.T)[2])

    def moved(self, x) -> Camera:
        """Turned by the rotation vector x[:3] about its own centre, then moved x[3:6] in its own frame."""
        turn = Rotation.from_rotvec(x[:3]).as_matrix()
        return Camera(turn @ self.R, turn @ self.T + x[3:6], self.f, self.c)


def nominal_camera(base_from_palette, urdf: Path) -> Camera:
    """The palette camera where the station fix and urdf/palette.urdf put its lens frame (+x along its line of sight,
    +z up): the optical frame looks along +x, image right along -y, image down along -z."""
    import xml.etree.ElementTree as ET

    joint = next(j for j in ET.parse(urdf).getroot().iter("joint") if j.find("child").get("link") == "palette_camera")
    origin = joint.find("origin")
    lens = np.asarray(base_from_palette, float) @ rpy_matrix([float(v) for v in origin.get("xyz").split()],
                                                             [float(v) for v in origin.get("rpy", "0 0 0").split()])
    rotation = (lens[:3, :3] @ LENS_FROM_OPTICAL).T
    return Camera(rotation, -rotation @ lens[:3, 3], FOCAL_PX, np.array(SIZE, float) / 2.0)


# --- the stills ---------------------------------------------------------------------------------
def _strip(image, origin, down, px, up_m, down_m, half_m):
    """The image resampled along the pen (float BGR): row s runs `down` from `origin`, column w to its right, in
    pixels, from up_m over origin to down_m under it and half_m to each side. Returns (strip, s, w, across)."""
    import cv2

    down = np.asarray(down, float) / np.linalg.norm(down)
    across = np.array([down[1], -down[0]])
    s = np.arange(-round(up_m * px), round(down_m * px) + 1.0)
    w = np.arange(-round(half_m * px), round(half_m * px) + 1.0)
    grid = (np.asarray(origin, float) + s[:, None, None] * down + w[None, :, None] * across).astype(np.float32)
    strip = cv2.remap(image, grid[..., 0].copy(), grid[..., 1].copy(), cv2.INTER_LINEAR)
    return strip.astype(np.float32), s, w, across


def _deviation(strip, w, px, band_m):
    """Each sample's colour distance from its row's background, a line across the row through the columns past
    band_m (a lamp's or a screen's blur behind the pen drops out); and the noise, from those columns' own."""
    import cv2

    strip = cv2.GaussianBlur(strip, (0, 0), 2.0)
    outer = np.abs(w) >= band_m * px
    basis = np.stack([np.ones_like(w), w / px], axis=1)
    coef = np.einsum("ij,sjc->sic", np.linalg.pinv(basis[outer]), strip[:, outer])
    dev = np.linalg.norm(strip - np.einsum("wi,sic->swc", basis, coef), axis=2)
    rest = dev[:, outer]
    return dev, float(np.median(rest)), max(1.0, 1.4826 * float(np.median(np.abs(rest - np.median(rest)))))


def _body(image, near, down, px):
    """The pen body's axis by the predicted tip `near`: its middle across the strip 7.5 and 4.75 mm over near, both
    well up the body; (its point level with near, its image direction), or None without a pen."""
    strip, s, w, across = _strip(image, near, down, px, UP_M, 0.0, 0.0045)
    dev, floor, _ = _deviation(strip, w, px, 0.0036)
    middles = []
    for rows in (s < -0.006 * px, (s >= -0.006 * px) & (s < -0.0035 * px)):
        profile = dev[rows].mean(axis=0)
        if profile.max() < floor + 20.0:
            return None
        inside = np.flatnonzero(profile > (floor + profile.max()) / 2.0)
        middles.append((w[inside[0]] + w[inside[-1]]) / 2.0)
    slope = (middles[1] - middles[0]) / (0.00275 * px)
    down = np.asarray(down, float) / np.linalg.norm(down) + slope * across
    return np.asarray(near, float) + (middles[1] + slope * 0.00475 * px) * across, down / np.linalg.norm(down)


def find_tip(image, near, down, px) -> np.ndarray | None:
    """The pen's working tip (u, v) in a still, by the predicted tip `near`, the pen running along `down` (image)
    toward it, px pixels per metre there; None without a pen. The body sets the search's axis; the tip is where the
    metal's contrast over its row's background falls through half way to the background under it, and its middle
    across is the metal's own over the last half millimetre."""
    body = _body(image, near, down, px)
    if body is None:
        return None
    centre, down = body
    strip, s, w, across = _strip(image, centre, down, px, UP_M, DOWN_M, 0.0032)
    dev, floor, noise = _deviation(strip, w, px, 0.0026)
    cone, mm = np.abs(w) <= CONE_HALF_M * px, px / 1000.0
    central = dev[:, cone].max(axis=1)
    pen = central > floor + 10.0 * noise
    first = int(np.argmax(pen))
    if not pen[first] or first > 3.0 * mm or pen[first:].all():
        return None                      # no body near the strip's top, or no end to it inside
    low = first + int(np.argmin(pen[first:])) - 1
    upper = central[max(0, low - round(mm)):max(1, low - round(0.3 * mm))]
    level = (np.median(upper) + np.median(central[low + 1:])) / 2.0
    if np.median(upper) < floor + 12.0 * noise:
        return None
    start = max(0, low - round(0.3 * mm))
    over = np.flatnonzero(central[start:] >= level)
    if not len(over):
        return None
    i = start + int(over[-1])
    if i + 1 >= len(s):
        return None
    tip_s = s[i] + (central[i] - level) / (central[i] - central[i + 1])
    weight = np.clip(dev[(s > tip_s - 0.5 * mm) & (s <= tip_s)][:, cone] - floor, 0.0, None).sum(axis=0)
    return centre + tip_s * down + float(weight @ w[cone] / weight.sum()) * across


def find_park_tip(image, near, down, px) -> np.ndarray | None:
    """find_tip about the park's prediction, which carries the nominal camera's few millimetres: from the prediction,
    then from it moved along the pen up to 3 mm either way (the 2026-10-03 daylight park against the dark screen was
    found only from 0.5 mm higher)."""
    unit = np.asarray(down, float) / np.linalg.norm(down)
    for mm in (0.0, -0.5, 0.5, -1.0, 1.0, -1.5, 1.5, -2.0, 2.0, -2.5, 2.5, -3.0, 3.0):
        found = find_tip(image, np.asarray(near, float) + mm * 1e-3 * px * unit, down, px)
        if found is not None:
            return found
    return None


def find_ball_top(image, near, down, px) -> np.ndarray | None:
    """The probe ball's top (u, v) by its predicted centre `near`: where the ball (its dark rim, its highlight) and
    the stylus under it begin to stand out over 0.4 mm of the columns about their middle, under whatever glow the
    strip's top holds; None when nothing does."""
    strip, s, w, across = _strip(image, near, down, px, 0.003, 0.0025, 0.003)
    dev, floor, noise = _deviation(strip, w, px, 0.0022)
    on = dev > floor + 8.0 * noise
    profile = on.mean(axis=0)
    inside = np.flatnonzero(profile > profile.max() / 2.0)
    if not len(inside):
        return None
    middle = (w[inside[0]] + w[inside[-1]]) / 2.0
    width = on[:, np.abs(w - middle) <= 0.0012 * px].sum(axis=1)
    wide = width >= 0.0004 * px
    onsets = np.flatnonzero(wide[1:] & ~wide[:-1]) + 1
    if not wide[-1] or not len(onsets):
        return None                      # the stylus runs out of the strip's foot; its top must be inside
    top = onsets[-1]
    top_s = s[top] - (width[top] - 0.0004 * px) / max(1.0, float(width[top] - width[top - 1]))
    return np.asarray(near, float) + top_s * np.asarray(down, float) / np.linalg.norm(down) + middle * across


class BallMark:
    """A patch of the park's still about the ball's top; a later still's patch moved by the camera's drift."""

    def __init__(self, image, top, px):
        self.r, self.m = round(0.0013 * px), round(0.0005 * px)
        self.x, self.y = int(top[0]) - self.r, int(top[1]) - round(0.0004 * px)
        self.patch = _grey(image)[self.y:self.y + round(0.0028 * px), self.x:self.x + 2 * self.r + 1]

    def shift(self, image) -> np.ndarray | None:
        import cv2

        h, w = self.patch.shape
        area = _grey(image)[self.y - self.m:self.y + h + self.m, self.x - self.m:self.x + w + self.m]
        score = cv2.matchTemplate(area, self.patch, cv2.TM_CCOEFF_NORMED)
        _, best, _, (i, j) = cv2.minMaxLoc(score)
        if best < 0.6 or not (0 < i < score.shape[1] - 1 and 0 < j < score.shape[0] - 1):
            return None
        return np.array([i + _vertex(score[j, i - 1:i + 2]) - self.m, j + _vertex(score[j - 1:j + 2, i]) - self.m])


def _grey(image):
    return image.astype(np.float32).mean(axis=2)


def _vertex(three) -> float:
    a, b, c = (float(v) for v in three)
    return 0.5 * (a - c) / (a - 2.0 * b + c) if a - 2.0 * b + c < 0.0 else 0.0


# --- the fit ------------------------------------------------------------------------------------
def fit(mounts, uv, tip0, camera: Camera, free: bool = True):
    """The tip in the tool mount (x and y, its length held at tip0's; or tip0 itself unless `free`) and the camera
    (turned and moved from `camera`), by least squares over every still's tip: FK(q)·tip through the pinhole. Returns
    (tip, camera, misses in px, the tip's 1-sigma in m)."""
    tip0 = np.asarray(tip0, float)

    def unpack(x):
        return camera.moved(x[:6]), (np.array([x[6], x[7], tip0[2]]) if free else tip0)

    def residuals(x):
        cam, tip = unpack(x)
        return (cam.project(mounts[:, :3, :3] @ tip + mounts[:, :3, 3]) - uv).ravel()

    result = least_squares(residuals, np.concatenate([np.zeros(6), tip0[:2] if free else []]), x_scale="jac")
    cam, tip = unpack(result.x)
    scale = float(result.fun @ result.fun) / max(1, result.fun.size - result.x.size)
    cov = np.linalg.pinv(result.jac.T @ result.jac) * scale
    return tip, cam, result.fun.reshape(-1, 2), np.sqrt(np.clip(np.diag(cov)[6:], 0.0, None))


def gap(camera: Camera, tip_m, tip_px, top_px) -> float:
    """How far the tip stands over the ball's top (m) in one still: both pixels back-projected to the tip's depth
    (the ball sits by the optical axis, where its own depth hardly matters), their difference along the base's z."""
    depth = camera.depth(np.asarray(tip_m, float))
    rays = (np.array([tip_px, top_px], float) - camera.c) / camera.f
    points = (np.column_stack([rays, [1.0, 1.0]]) * depth - camera.T) @ camera.R
    return float(points[0, 2] - points[1, 2])


def ask_still(path: Path) -> None:
    """A still at `path`, taken by the operator's CLI (`tatbot ros calib sweep`), which reads this line, captures on
    the palette camera's node, copies the frame here and answers `ok` on stdin, else why not."""
    print(json.dumps({"still": str(path)}), flush=True)
    answer = sys.stdin.readline().strip()
    if answer != "ok":
        raise RuntimeError(f"no still for {path.name}: {answer or 'the CLI went away'}")


# --- the run ------------------------------------------------------------------------------------
METHOD = "palette-camera joint-6 sweep (tatbot ros calib sweep)"


class Sweep:
    """One run's holds and stills; the Calibration supplies the rig, the way checks and the records."""

    def __init__(self, cal: program.Calibration, fix: station.StationFix, tool: str, station_path: str = "",
                 ask=ask_still):
        self.cal, self.fix, self.tool, self.station_path, self.ask = cal, fix, tool, station_path, ask
        self.kin = cal.kin
        self.camera = nominal_camera(fix.base_from_palette, cal.repo / "urdf" / "palette.urdf")
        self.tip0 = np.asarray(cal.tip0, float)
        self.holds: list[dict] = []
        self.offset = np.zeros(2)        # the park's tip as seen less where the nominal camera puts it
        self.mark = self.top = None      # the ball's patch (BallMark) and its top in the park's still
        self.yaw, self.q_park = 0.0, None

    def mount(self, q) -> np.ndarray:
        return self.kin.frame(np.asarray(q, float), f"{self.cal.arm}/tool_mount")

    def tip(self, q, tip=None) -> np.ndarray:
        mount = self.mount(q)
        return mount[:3, :3] @ (self.tip0 if tip is None else tip) + mount[:3, 3]

    def park_pose(self, yaw_deg: float) -> np.ndarray:
        return halo.tool_pose(self.fix.ball + [0.0, 0.0, halo.BALL_RADIUS_M + PARK_OVER_M], math.radians(yaw_deg))

    @staticmethod
    def turned(q, deg: float) -> np.ndarray:
        q = np.array(q, float)
        q[5] += math.radians(deg)
        return q

    def seen(self, q) -> float:
        """How well the camera sees the turns from park joints q: the least singular value of the image's response
        (the nominal camera's directions) to the tip across its axis, about its mean over the turns."""
        rows = np.array([self.camera.R[:2] @ self.mount(self.turned(q, deg))[:3, :2] for deg in TURNS_DEG])
        return float(np.linalg.svd((rows - rows.mean(axis=0)).reshape(-1, 2), compute_uv=False)[-1])

    def yaws(self) -> list[float]:
        """The yaws the arm parks at from rest with the planner's joint margin, its wrist clear of the posts, the
        other arm and the station (reach._margin), and joint 6 inside its limits through the turns; best seen
        first."""
        from tatbot_session import ready

        margin = float(self.cal.motion["clik"]["joint_limit_margin_rad"])
        lower, upper = float(self.kin.lower[5]) + margin, float(self.kin.upper[5]) - margin
        ranked = []
        for yaw in YAWS_DEG:
            pose = self.park_pose(yaw)
            if reach._margin(self.kin, pose, self.cal.bodies, self.cal.posts, self.cal.guard, self.cal.arm,
                             self.cal.station_clear) < margin:
                continue
            q = ready.solve_ik_seeded(self.kin, pose, reach.REST)
            if lower <= q[5] + math.radians(min(TURNS_DEG)) and q[5] + math.radians(max(TURNS_DEG)) <= upper:
                ranked.append((self.seen(q), yaw))
        return [yaw for _, yaw in sorted(ranked, reverse=True)]

    def move(self, label: str, pose: np.ndarray) -> None:
        """One probe-guarded move, refused (RuntimeError, nothing sent) when the tool's envelope would meet a station
        part on the executor's way (up, across, down) or stand within BALL_CLEAR_M of the fix's ball, or the way fails
        Calibration.way_check (no plan, the floor, the other arm, the camera's post, the arm against the station)."""
        here = self.kin.fk(self.cal.rig.joints())[:3, 3]
        top = max(here[2], pose[2, 3] + self.cal.clearance_m)
        way = [here, np.array([here[0], here[1], top]), np.array([pose[0, 3], pose[1, 3], top]), pose[:3, 3]]
        zone = halo.meets(way, pose[:3, 2], self.cal.zones(), self.cal.halo, self.cal.wall_floor())
        clear = self.ball_gap(pose)
        if zone is not None or clear < BALL_CLEAR_M:
            raise RuntimeError(f"{label}: " + (f"its way would meet the {zone}" if zone else
                                               f"the tool would stand {clear * 1000:.1f} mm off the fix's ball")
                               + "; not sent")
        self.cal.way_check(label, [pose])
        self.cal.rig.goal(pose, move=True)

    def ball_gap(self, pose: np.ndarray) -> float:
        """The least distance (m) from the fix's ball to the tool's envelope at `pose`, up its first 30 mm."""
        h = np.arange(0.0, 0.030, 0.0005)
        axis = pose[:3, 3] - h[:, None] * pose[:3, 2]
        radius = np.array([self.cal.halo.radius_at(x) for x in h])
        return float(np.min(np.linalg.norm(axis - self.cal.ball, axis=1) - radius)) - halo.BALL_RADIUS_M

    def park(self) -> np.ndarray:
        """To the first yaw whose way passes the checks (by a staging pose when the executor plans none from where
        the arm stands: Calibration.approach); its pose."""
        from tatbot_session import ready

        refused = []
        for yaw in self.yaws():
            self.cal.heading, pose = math.radians(yaw), self.park_pose(yaw)
            try:
                self.cal.approach(pose[:3, 3])
                self.move("park", pose)
            except RuntimeError as error:
                if program._latched(self.cal):
                    raise
                refused.append(f"yaw {yaw:+.0f}: {error}")
                continue
            self.yaw, self.q_park = yaw, ready.solve_ik_seeded(self.kin, pose, self.cal.rig.joints())
            self.cal.say(f"parked at yaw {yaw:+.0f} deg, the tip {PARK_OVER_M * 1000:.0f} mm over the fix's ball top")
            return pose
        raise RuntimeError("no park is planned: " + ("; ".join(refused[:3]) or "the arm reaches no yaw over the ball"))

    def hold(self, label: str, kind: str, pose: np.ndarray) -> None:
        """A hold and its still; one the checks refuse is skipped (a latched arm stops the run)."""
        try:
            self.move(label, pose)
        except RuntimeError as error:
            if program._latched(self.cal):
                raise
            self.cal.say(f"{label}: skipped ({error})")
            return
        self.still(label, kind, pose)

    def still(self, label: str, kind: str, pose: np.ndarray) -> dict:
        """Settled, a still taken and looked at; its row goes to holds.jsonl."""
        self.cal.rig.spin(SETTLE_S)
        before = self.cal.rig.joints()
        path = self.cal.run.dir / "stills" / f"{len(self.holds):02d}-{label.replace(' ', '')}.jpg"
        path.parent.mkdir(exist_ok=True)
        self.ask(path)
        after = self.cal.rig.joints()
        q = (before + after) / 2.0
        row = {"label": label, "kind": kind, "pose": np.asarray(pose).tolist(), "q": q.tolist(),
               "q_moved_rad": float(np.max(np.abs(after - before)[:6])), "still": path.name, **self.look(path, q)}
        self.holds.append(row)
        self.cal.log("holds.jsonl", row)
        self.cal.say(f"{label}: the tip " + (f"at {np.round(row['tip_px'], 1).tolist()} px" if row["tip_px"]
                                             else "not found"))
        return row

    def look(self, path: Path, q) -> dict:
        """The tip's pixel (None when not found) and how far the ball's patch moved since the park's still."""
        import cv2

        image = cv2.imread(str(path))
        if image is None or image.shape[1::-1] != SIZE:
            raise RuntimeError(f"{path.name}: not a {SIZE[0]} x {SIZE[1]} still")
        tip, axis = self.tip(q), self.mount(q)[:3, 2]
        near, up = self.camera.project([tip, tip - 0.005 * axis]) + self.offset
        px = self.camera.f / self.camera.depth(tip)
        found = find_tip(image, near, near - up, px) if self.holds else find_park_tip(image, near, near - up, px)
        if not self.holds:
            if found is None:
                raise RuntimeError(f"{path.name}: the palette camera does not see the pen's tip at the park")
            self.first(image, found - near, px)
        shift = self.mark.shift(image) if self.mark is not None else None
        return {"tip_px": None if found is None else found.tolist(),
                "shift_px": None if shift is None else shift.tolist()}

    def first(self, image, offset, px) -> None:
        """The park's still: where the tip stands off the nominal camera's prediction places the later stills'
        search; the ball's top there, the gap's reference and the drift's patch."""
        self.offset = offset
        centre, under = self.camera.project([self.fix.ball, self.fix.ball - [0.0, 0.0, 0.001]])
        self.top = find_ball_top(image, centre, under - centre, self.camera.f / self.camera.depth(self.fix.ball))
        if self.top is None:
            self.cal.say("park: the ball's top is not found: no drift is taken off, and the turns go on the planned "
                         "clearance")
            return
        self.mark = BallMark(image, self.top, px)
        self.cal.say(f"park: the ball's top at {np.round(self.top, 1).tolist()} px")

    def shifts(self, park: np.ndarray) -> None:
        """SHIFT_M across the camera's line of sight both ways, up, and away from the camera along it."""
        across, away = self.fix.base_from_palette[:3, 1], -self.fix.base_from_palette[:3, 0]
        for label, step in (("+across", across), ("-across", -across), ("up", np.array([0.0, 0.0, 1.0])),
                            ("away", away)):
            pose = park.copy()
            pose[:3, 3] += SHIFT_M * step
            self.hold(f"shift {label}", "shift", pose)

    def room(self) -> float | None:
        """The tip's height over the ball's top in the park's still (m), through the camera fitted to the stills so
        far with the installed tip; None without the ball's top."""
        if self.top is None:
            return None
        rows = [row for row in self.holds if row["tip_px"] is not None]
        _, camera, _, _ = fit(np.array([self.mount(row["q"]) for row in rows]),
                              np.array([row["tip_px"] for row in rows]), self.tip0, self.camera, free=False)
        return gap(camera, self.tip(self.holds[0]["q"]), self.holds[0]["tip_px"], self.top)

    def turns(self) -> None:
        """Joint 6 alone through TURNS_DEG from the park's planned joints, once the park's still shows the room."""
        poses = {deg: self.kin.fk(self.turned(self.q_park, deg)) for deg in TURNS_DEG}
        drop = float(self.kin.fk(self.q_park)[2, 3] - min(pose[2, 3] for pose in poses.values()))
        room = self.room()
        if room is not None and room - drop < TURN_CLEAR_M:
            raise RuntimeError(f"the park's still shows the tip {room * 1000:.1f} mm over the ball's top and the turns "
                               f"take it {drop * 1000:.1f} mm lower, under {TURN_CLEAR_M * 1000:.0f} mm; no turn sent")
        for deg, pose in poses.items():
            self.hold(f"turn {deg:+d}", "turn", pose)

    def leak(self) -> np.ndarray:
        """What an error in the tip's length does to the fitted tip: (dx, dy) per metre of it. A turn cannot see the
        tip along joint 6's axis, n in the mount, so holding the length takes an error e along z as -e n_xy / n_z."""
        a, b = self.mount(self.q_park), self.mount(self.turned(self.q_park, 1.0))
        n = a[:3, :3].T @ Rotation.from_matrix(b[:3, :3] @ a[:3, :3].T).as_rotvec()
        return -n[:2] / n[2]

    def solve(self, rows: list[dict]) -> tuple[dict, dict, dict]:
        """The fit over `rows`, and each turn left out: (the fit's record, held-out misses, tip moves), in metres."""
        mounts = np.array([self.mount(row["q"]) for row in rows])
        uv = np.array([np.subtract(row["tip_px"], row["shift_px"] or (0.0, 0.0)) for row in rows])
        tip, camera, miss, sigma = fit(mounts, uv, self.tip0, self.camera)
        park = rows[0]
        px = camera.f / camera.depth(self.tip(park["q"], tip))
        held, moved = {}, {}
        for i in (i for i, row in enumerate(rows) if row["kind"] == "turn"):
            keep = np.arange(len(rows)) != i
            t, cam, _, _ = fit(mounts[keep], uv[keep], self.tip0, camera)
            held[rows[i]["label"]] = float(np.linalg.norm(cam.project(self.tip(rows[i]["q"], t))[0] - uv[i]) / px)
            moved[rows[i]["label"]] = float(np.linalg.norm(t - tip))
        turned = np.array([row["q"][:5] for row in rows if row["kind"] == "turn"]) - np.asarray(park["q"][:5])
        out = {"tip_m": tip.tolist(), "rms_m": {"image": float(np.sqrt(np.mean(np.sum(miss ** 2, axis=1)))) / px},
               "sigma": {"tip_m": [*sigma.tolist(), 0.0]}, "length_leak": self.leak().tolist(), "px_per_m": px,
               "camera_m": float(np.linalg.norm(camera.R.T @ camera.T + self.tip(park["q"], tip))),
               "drift_px": max([float(np.hypot(*row["shift_px"])) for row in rows if row["shift_px"]], default=None),
               "joints_1_5_moved_rad": float(np.max(np.abs(turned))), "yaw_deg": self.yaw,
               "stills": {kind: sum(row["kind"] == kind for row in rows) for kind in ("park", "shift", "turn")}}
        if self.top is not None and park["kind"] == "park":
            out["gap_m"] = gap(camera, self.tip(park["q"], tip), park["tip_px"], self.top)
            out["ball_top_z_m"] = float(self.tip(park["q"], tip)[2]) - out["gap_m"]
        return out, held, moved

    def write(self) -> int:
        """candidate.json and report.md from every still that shows the tip; 1 with fewer than three turns."""
        rows = [row for row in self.holds if row["tip_px"] is not None]
        if sum(row["kind"] == "turn" for row in rows) < 3:
            self.cal.say("fewer than three turns show the tip: nothing to fit")
            return 1
        result, held, moved = self.solve(rows)
        run = self.cal.run.dir
        (run / "candidate.json").write_text(json.dumps({
            "schema": "tatbot.probe-tip-candidate/2", "method": METHOD, "arm": self.cal.arm, "tool": self.tool,
            "fit": result, "tip0_m": self.tip0.tolist(), "held_out_rms_m": held, "tip_moved_m": moved,
            "contacts": len(rows), "station": self.station_path, "run_id": run.name,
            "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}, indent=1) + "\n")
        report = _report(run.name, self.cal.arm, result, held, moved, self.tip0)
        (run / "report.md").write_text(report)
        print(report, flush=True)
        return 0

    def run(self) -> int:
        program._preflight(self.cal)
        park = self.park()
        self.still("park", "park", park)
        self.shifts(park)
        self.turns()
        return self.write()


def _report(run: str, arm: str, out: dict, held: dict, moved: dict, tip0) -> str:
    def mm(metres):
        return metres * 1000.0

    tip, sig, leak = np.asarray(out["tip_m"]), out["sigma"]["tip_m"], out["length_leak"]
    stills = out["stills"]
    lines = [f"# Tip from the joint-6 sweep ({run})", "",
             f"- stills: the park, {stills['shift']} shifts and {stills['turn']} turns of joint 6 at yaw "
             f"{out['yaw_deg']:+.0f} deg; the image misses by rms {mm(out['rms_m']['image']):.3f} mm at the tip "
             f"({out['px_per_m'] / 1000:.1f} px/mm, the camera {mm(out['camera_m']):.0f} mm away)",
             f"- tip across its axis in the tool mount: ({mm(tip[0]):.3f}, {mm(tip[1]):.3f}) mm +- ({mm(sig[0]):.3f}, "
             f"{mm(sig[1]):.3f}); installed ({mm(tip0[0]):.3f}, {mm(tip0[1]):.3f}): moved ({mm(tip[0] - tip0[0]):+.3f}, "
             f"{mm(tip[1] - tip0[1]):+.3f}) mm",
             f"- along its axis it stays {mm(tip0[2]):.3f} mm: the turns cannot see the tip along joint 6's axis, so a "
             f"length error of 1 mm reads here as ({leak[0]:+.2f}, {leak[1]:+.2f}) mm",
             "- each turn left out: misses by (mm) " + ", ".join(f"{k} {mm(v):.3f}" for k, v in held.items()),
             f"- each turn left out moves the tip by up to {mm(max(moved.values())):.3f} mm",
             f"- joints 1-5 moved up to {mm(out['joints_1_5_moved_rad']):.2f} mrad over the turns (measured); the ball's "
             "patch drifted " + ("unseen" if out["drift_px"] is None else f"up to {out['drift_px']:.1f} px")]
    if "gap_m" in out:
        lines.append(f"- at the park the tip stood {mm(out['gap_m']):.2f} mm over the ball's top: the top at z "
                     f"{mm(out['ball_top_z_m']):.2f} mm in the base through the fitted tip and the installed length")
    return "\n".join([*lines, "", f"Adopt with `tatbot ros calib apply --arm {arm} --run {run}`."]) + "\n"


def equip(cal: program.Calibration, fix: station.StationFix, motion: dict) -> None:
    """The way checks' inputs as `calib run` sets them: the executor's motion settings, the wrist's bodies, the
    overhead camera's post, the two-arm guard, the arm's registration, the installed tip, and the station."""
    cal.motion, cal.bodies = motion, reach.wrist_bodies(cal.kin, cal.arm)
    cal.posts = [reach.overhead_post(fix.detail["overhead_camera_m"])] if fix.detail.get("overhead_camera_m") else []
    cal.guard, text = program._guard(cal.repo, motion)
    cal.world_from_base = program.world_from_base(cal.repo, cal.arm)
    cal.clearance_m = float(motion["probe"]["clearance_m"])
    mount_from_tcp = np.linalg.inv(cal.kin.frame(np.zeros(7), f"{cal.arm}/tool_mount")) @ cal.kin.fk(np.zeros(7))
    cal.tip0, cal.mount_in_tcp = mount_from_tcp[:3, 3], mount_from_tcp[:3, :3].T
    cal.say(f"S0: {text}")
    program.place_station(cal, fix)


def run(args) -> int:
    from tatbot_description import repo_root, robot_description
    from tatbot_motion import Kinematics, load_motion
    from tatbot_session.lease import PaletteLease

    repo = repo_root(None)
    try:
        tool, halo_tip = fitted_tool(repo, args.arm, args.tool)
    except ToolRefusedError as error:
        print(json.dumps({"ok": False, "message": str(error)}))
        return 3
    sys.path.insert(0, str(repo / "scripts" / "lib"))
    import tatbot_runlog

    fix = station.StationFix.from_dict(json.loads(Path(args.station).expanduser().read_text()))
    station.require_fresh(fix, args.run_started, max_age_s=args.max_age_s)
    kin = Kinematics(robot_description(None, arms=(args.arm,)), args.arm)
    run_log = tatbot_runlog.init(program.WORKFLOW, meta={"arm": args.arm, "tool": tool.tool_id, "method": METHOD,
                                                         "station": args.station},
                                 attach_logging=False, argv=["tatbot_calib", "sweep", "--arm", args.arm])
    cal = program.Calibration(None, run_log, repo, args.arm, halo_tip, 0.0, kin)
    equip(cal, fix, load_motion())
    cal.rig, code = program.Rig(args.arm), 1
    try:
        with PaletteLease(program.palette_zone(repo, args.arm, fix, run_log.dir.name)):
            code = Sweep(cal, fix, tool.tool_id, args.station).run()
    except RuntimeError as error:   # a refusal, a failed goal, or the palette held by the other arm's run
        cal.say(f"stopped: {error}")
    finally:
        if not program._latched(cal):   # lifted and landed; a latched arm stays where it holds
            program.retreat(cal.rig, kin)
        cal.rig.close()
        run_log.finalize(code, status="ok" if code == 0 else "fail")
    print(json.dumps({"ok": code == 0, "run_dir": str(run_log.dir)}), flush=True)
    return code


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="tatbot_calib sweep", description=__doc__.split("\n\n")[0])
    parser.add_argument("--arm", required=True)
    parser.add_argument("--tool", default=None, help="the tool stated to be fitted; must be config/workspace.yaml's")
    parser.add_argument("--station", required=True, help="this run's station.json (tatbot ros station)")
    parser.add_argument("--run-started", type=float, required=True, help="epoch seconds this run began")
    parser.add_argument("--max-age-s", type=float, default=station.MAX_AGE_S)
    return run(parser.parse_args(argv))


if __name__ == "__main__":
    sys.exit(main())
