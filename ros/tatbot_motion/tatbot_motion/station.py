"""Shared measured station geometry and freshness; calibration observes it, motion consumes it."""
from __future__ import annotations

import datetime as dt
import math
import time
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from tatbot_description.transforms import rpy_matrix

SCHEMA = "tatbot.station-fix/1"

MAX_AGE_S = 600.0     # a fix older than this is refused, even inside its own run

CLOCK_SKEW_S = 5.0    # the camera's owner stamps the shot; a stamp this far ahead of this node's clock is tolerated


class StaleStationError(RuntimeError):
    """The station's pose was measured before this run started, or too long ago: measure it again."""


@dataclass
class StationFix:
    arm: str
    base_from_palette: np.ndarray   # <arm>/base_link <- palette_root
    ball: np.ndarray                # the probe ball's centre in <arm>/base_link, m
    measured_utc: str               # when the fiducials were seen: the scan's capture time
    source: str                     # overhead_tag | wrist_tag
    detail: dict = field(default_factory=dict)

    def as_dict(self) -> dict:
        return {"schema": SCHEMA, "arm": self.arm, "base_from_palette": np.asarray(self.base_from_palette).tolist(),
                "ball": [float(v) for v in self.ball], "measured_utc": self.measured_utc, "source": self.source,
                "detail": self.detail}

    @classmethod
    def from_dict(cls, data: dict) -> StationFix:
        if data.get("schema") != SCHEMA:
            raise ValueError(f"not a {SCHEMA} record")
        return cls(data["arm"], np.asarray(data["base_from_palette"], float), np.asarray(data["ball"], float),
                   data["measured_utc"], data["source"], dict(data.get("detail") or {}))


def parse_utc(text: str) -> float:
    """Seconds since the epoch of an ISO-8601 UTC stamp (a trailing Z is accepted)."""
    stamp = dt.datetime.fromisoformat(str(text).replace("Z", "+00:00"))
    if stamp.tzinfo is None:
        raise ValueError(f"{text!r}: a station stamp must carry its UTC offset")
    return stamp.timestamp()


def ball_in_palette(palette_urdf: Path) -> np.ndarray:
    """The probe ball's centre in palette_root: the origin of urdf/palette.urdf's joint to palette_probe_ball."""
    for joint in ET.parse(palette_urdf).getroot().iter("joint"):
        child, parent, origin = joint.find("child"), joint.find("parent"), joint.find("origin")
        if child is not None and child.get("link") == "palette_probe_ball":
            if parent is None or parent.get("link") != "palette_root" or origin is None:
                raise ValueError(f"{palette_urdf}: palette_probe_ball must hang off palette_root with an origin")
            return np.array([float(v) for v in origin.get("xyz", "0 0 0").split()])
    raise ValueError(f"{palette_urdf}: no joint to palette_probe_ball")


def inkcap_rims(palette_urdf: Path, palette_yaml: Path, *, load=None, base_from_palette=None) -> list[tuple[str, np.ndarray, float]]:
    """Each inkcap as (slot, its rim's centre in palette_root, its outside radius): urdf/palette.urdf places the
    cap's support floor, and config/palette.yaml gives its outside dimensions.
    A tilted cap gets a conservative vertical post; an absent cap retains its support."""
    import yaml

    from tatbot_motion.dip import _positive, cap_support_frame

    palette = yaml.safe_load(Path(palette_yaml).read_text())
    out = []
    joints = list(ET.parse(palette_urdf).getroot().iter('joint'))
    for name, slot in palette['slots'].items():
        frame = cap_support_frame([j for j in joints if j.find('child') is not None
                                   and j.find('child').get('link') == name])
        size = palette["sizes"][slot["size"]]
        height = 0. if load is not None and load[name].cap_present is False else _positive(size['height_m'])
        rotation = np.eye(3) if base_from_palette is None else np.asarray(base_from_palette)[:3, :3]
        lateral = float(np.linalg.norm((rotation @ frame[:3, 2])[:2]))
        radius = _positive(size['diameter_m'])/2
        centre = (frame @ [0, 0, height, 1])[:3] + rotation.T @ [0, 0, radius*lateral]
        out.append((name, centre, radius+height*lateral))
    return out

# Part positions come from the installed standalone asset. These conservative radii
# cover the camera, v11 housing and actuator, and diamond tag seat.
# The E-stop frame is its panel; its envelope centre is halfway down the housing.
# The camera's case, plate and mount stand behind its lens, away from the probe along +X, which the lens's sphere
# misses: on v11 the case to 12 mm behind the lens and the plate and mount 16-22 mm behind, 21.5 mm to each side,
# up to 23 mm over the lens (its top at Z78.6). A post this far behind the lens frame, this high over it and this wide.
KEEP_OUT_RADII = (("palette_camera", 0.022), ("palette_estop", 0.040), ("palette_tag", 0.045))

CAMERA_MOUNT = (0.019, 0.023, 0.0235)

POSTS = (("probe_body", (0.0, 0.0, 0.030), 0.0175),
         ("probe_collar", (0.0, 0.0, 0.033), 0.005))

BODY_TOP_BELOW_BALL_M = 0.0556 - 0.030   # the probe body's white top under the ball's centre (POSTS, the URDF)

BODY_RADIUS_M = POSTS[0][2]              # the probe body's top, a white disc this wide in radius (POSTS)


def parts(base_from_palette: np.ndarray, inkcaps=(), palette_urdf: Path | None = None) -> list[tuple[str, np.ndarray, float, bool]]:
    """(name, centre in the base, radius, post) for every part: a post stands from the table up to its centre."""
    def base(xyz):
        return (base_from_palette @ np.append(xyz, 1.0))[:3]

    if palette_urdf is None:
        from tatbot_description import repo_root
        palette_urdf = repo_root(None) / "urdf" / "palette.urdf"
    frames = {}
    for joint in ET.parse(palette_urdf).getroot().iter("joint"):
        child, parent, origin = joint.find("child"), joint.find("parent"), joint.find("origin")
        if child is not None and parent is not None and parent.get("link") == "palette_root" and origin is not None:
            frames[child.get("link")] = np.array([float(v) for v in origin.get("xyz", "0 0 0").split()])
    missing = {name for name, _ in KEEP_OUT_RADII} - frames.keys()
    if missing:
        raise ValueError(f"{palette_urdf}: missing station keep-out frames {sorted(missing)}")
    frames["palette_estop"] = frames["palette_estop"] + [0.0, 0.0, -0.0262]
    back, up, width = CAMERA_MOUNT
    mount = np.add(frames["palette_camera"], (back, 0.0, up))
    return ([(name, base(frames[name]), radius, False) for name, radius in KEEP_OUT_RADII]
            + [(name, base(xyz), radius, True)
               for name, xyz, radius in (("palette_camera_mount", mount, width), *POSTS, *inkcaps)])


def require_fresh(fix: StationFix, run_started: float, *, now: float | None = None,
                  max_age_s: float = MAX_AGE_S) -> float:
    """Refuse a fix measured before the run started or more than max_age_s ago; return its age in seconds."""
    measured = parse_utc(fix.measured_utc)
    now = time.time() if now is None else float(now)
    if measured < run_started - CLOCK_SKEW_S:
        raise StaleStationError(f"the station was measured at {fix.measured_utc}, before this run started: "
                                "measure it again (the palette may have moved since)")
    age = now - measured
    if age > max_age_s:
        raise StaleStationError(f"the station was measured {age:.0f} s ago, over the {max_age_s:.0f} s a fix "
                                "is used for: measure it again")
    if age < -CLOCK_SKEW_S:
        raise StaleStationError(f"the station's stamp {fix.measured_utc} is {-age:.0f} s ahead of this node's "
                                "clock")
    return max(age, 0.0)


ZONE_MARGIN_M = 0.020   # the palette zone reaches this far past the outermost part, and this far over the tallest


def station_world_parts(zones, world_from_base: np.ndarray, table_z: float) -> list:
    """The station's parts (Calibration.zones: name, centre in the arm base, radius, post) in the collision
    scene's world: spheres, and posts standing from the table (table_z, the base) up along the base's z to their
    centre (tatbot_motion.collision.Scene.parts_clearance)."""
    world, up = np.asarray(world_from_base, float), np.asarray(world_from_base, float)[:3, 2]
    out = []
    for name, centre, radius, post in zones:
        at = (world @ np.append(centre, 1.0))[:3]
        out.append((name, "post", at, radius, max(float(centre[2]) - table_z, 1e-3), up) if post
                   else (name, "sphere", at, radius))
    return out


def palette_zone(arm, fix, run, base, zones):
    """One measured zone for calibration and drawing; base is the loaded scene's registration."""
    root = np.asarray(fix.base_from_palette)[:3, 3]
    radius = max(float(np.hypot(*(centre[:2]-root[:2])))+r for _, centre, r, _ in zones)
    top = max(float(centre[2])+(0. if post else r) for _, centre, r, post in zones)
    at = np.eye(4)
    at[:3, 3] = root
    return {'arm': arm, 'run': run, 'world_from_zone': (base @ at).tolist(),
            'radius_m': radius+ZONE_MARGIN_M, 'height_m': top-float(root[2])+ZONE_MARGIN_M}


# The post: a 20 mm extrusion beside the camera, which a bracket holds a few centimetres off the post's axis in a
# direction nothing measures (the D555 cannot see its own post). A column 60 mm in radius about the camera's
# footprint takes both; the cube met the post within 40 mm of the footprint (2026-09-29).
POST_RADIUS_M = 0.060


@dataclass(frozen=True)
class Post:
    """A vertical post about (x, y) in the arm base, from the table up to top_m."""
    xy: tuple
    radius_m: float
    top_m: float


def overhead_post(camera_in_base) -> Post:
    """The post the overhead camera stands on: about the camera's footprint (its optical centre in the arm base,
    from the arm's registration), from the table up to it."""
    x, y, z = (float(v) for v in camera_in_base)
    return Post((x, y), POST_RADIUS_M, z)


TRIM_M = 0.002
MIN_PLANE_SHARE = 0.4


def _plane_inliers(points: np.ndarray, tries: int = 80) -> np.ndarray:
    """The largest set of points within TRIM_M of one plane (RANSAC, seeded: a run repeats)."""
    rng = np.random.default_rng(0)
    best = np.zeros(len(points), bool)
    for _ in range(tries):
        a, b, c = points[rng.choice(len(points), 3, replace=False)]
        normal = np.cross(b - a, c - a)
        if np.linalg.norm(normal) < 1e-9:
            continue
        inliers = np.abs((points - a) @ (normal / np.linalg.norm(normal))) < TRIM_M
        if inliers.sum() > best.sum():
            best = inliers
    return best


def plane_depth(depth: np.ndarray, k, dist, keep: np.ndarray, px, min_px: int = 30, max_rms_m: float = 0.0015):
    """Depth along the pixel ray through a robust plane fitted to valid aligned depth.
    Return None with too few inliers, insufficient plane share or excessive residuals."""
    import cv2

    keep = keep & (depth > 0.05)
    if int(keep.sum()) < min_px:
        return None
    ys, xs = np.nonzero(keep)
    z = depth[ys, xs].astype(float)
    rays = cv2.undistortPoints(np.stack([xs, ys], 1).astype(float).reshape(-1, 1, 2), k, dist).reshape(-1, 2)
    points = np.c_[rays * z[:, None], z]
    kept = _plane_inliers(points)    # the disc: the stylus, its foot and what the outline took beside it drop out
    if int(kept.sum()) < max(min_px, MIN_PLANE_SHARE * len(points)):
        return None
    centre = points[kept].mean(axis=0)
    normal = np.linalg.svd(points[kept] - centre, full_matrices=False)[2][2]
    if float(np.sqrt(np.mean(((points[kept] - centre) @ normal) ** 2))) > max_rms_m:
        return None
    ray = np.append(cv2.undistortPoints(np.array([[px]], float), k, dist).reshape(2), 1.0)
    return float((normal @ centre) / (normal @ ray))


# A move between two fixes from the same source: over its repeatability, or a rotation over half a degree. The
# D555's fix of 8 shots repeats to 0.1 mm across and 0.3 mm in height, one shot's to 0.3 and 1.0 (2026-09-29).
MOVED_M = {"overhead_tag": 0.001, "wrist_tag": 0.001}
MOVED_RAD = math.radians(0.5)
# The tag's free pose may tilt this far from the base's vertical before it is refused: the palette stands on the
# arm's table, and its more level IPPE pose tilts 1.0-1.7 degrees there (the other one 22-23).
MAX_TILT_RAD = math.radians(8.0)
# The shots of one fix agree this well, or the palette moved during it (or a detection is wrong).
SHOTS_AGREE_M = 0.002
SHOTS_AGREE_RAD = math.radians(1.0)


def anchored(base_from_palette: np.ndarray, ball: np.ndarray, ball_in_palette_m) -> np.ndarray:
    """Move the palette origin to the measured ball, retaining its rotation and URDF ball offset."""
    out = np.array(base_from_palette, float)
    out[:3, 3] += np.asarray(ball, float) - (out @ np.append(np.asarray(ball_in_palette_m, float), 1.0))[:3]
    return out


def palette_from_tag(palette_urdf: Path, *, require_confirmed: bool = False) -> np.ndarray:
    """palette_root <- installed palette_tag from the URDF; optionally require a confirmed seat."""
    root = ET.parse(palette_urdf).getroot()
    if require_confirmed and root.get("tag_pose_status") != "confirmed":
        raise ValueError("installed palette tag orientation is unconfirmed; confirm the printed pattern's "
                         "orientation on the new seat before measuring the station")
    for joint in root.iter("joint"):
        child, parent, origin = joint.find("child"), joint.find("parent"), joint.find("origin")
        if child is not None and child.get("link") == "palette_tag":
            if parent is None or parent.get("link") != "palette_root" or origin is None:
                raise ValueError(f"{palette_urdf}: palette_tag must hang off palette_root with an origin")
            return rpy_matrix([float(v) for v in origin.get("xyz", "0 0 0").split()],
                              [float(v) for v in origin.get("rpy", "0 0 0").split()])
    raise ValueError(f"{palette_urdf}: no joint to palette_tag")


def _level(x: np.ndarray) -> np.ndarray:
    """base <- palette_root for (x, y, z, yaw): the palette level, its z along the base's."""
    return rpy_matrix(x[:3], [0, 0, x[3]])


def level_pose(corners_px, k, dist, camera_from_base: np.ndarray, model_corners, palette_tag: np.ndarray,
               max_tilt_rad: float = MAX_TILT_RAD) -> tuple[np.ndarray, float]:
    """Fit level palette position/yaw to roof-tag corners; return pose and reprojection RMS.
    Seed from the more level IPPE pose; refuse if it exceeds max_tilt_rad.
    Corner order must match the detector; palette_tag is palette_root <- palette_tag."""
    import cv2
    from scipy.optimize import least_squares

    pixels = np.asarray(corners_px, float).reshape(4, 2)
    model = np.asarray(model_corners, float).reshape(4, 3)
    base_from_camera = np.linalg.inv(camera_from_base)
    tag_from_palette = np.linalg.inv(palette_tag)
    _, rvecs, tvecs, _ = cv2.solvePnPGeneric(model, pixels, k, dist, flags=cv2.SOLVEPNP_IPPE_SQUARE)
    free = []
    for rvec, tvec in zip(rvecs, tvecs, strict=True):
        camera_from_tag = np.eye(4)
        camera_from_tag[:3, :3], camera_from_tag[:3, 3] = cv2.Rodrigues(rvec)[0], np.ravel(tvec)
        free.append(base_from_camera @ camera_from_tag @ tag_from_palette)
    if not free:
        raise ValueError("no pose of the roof tag fits its corners")
    guess = max(free, key=lambda pose: pose[2, 2])
    tilt = math.acos(max(-1.0, min(1.0, guess[2, 2])))
    if tilt > max_tilt_rad:
        raise ValueError(f"the roof tag's most level pose tilts {math.degrees(tilt):.1f} deg from the arm's vertical, "
                         f"over {math.degrees(max_tilt_rad):.0f}: not the palette standing on the table")
    corners = np.c_[model, np.ones(4)].T

    def residual(x):
        in_camera = (camera_from_base @ _level(x) @ palette_tag @ corners)[:3].T
        projected, _ = cv2.projectPoints(in_camera, np.zeros(3), np.zeros(3), k, dist)
        return (projected.reshape(4, 2) - pixels).ravel()

    fitted = least_squares(residual, np.array([*guess[:3, 3], math.atan2(guess[1, 0], guess[0, 0])]))
    return _level(fitted.x), float(np.sqrt(np.mean(fitted.fun.reshape(4, 2) ** 2) * 2.0))


TAG_DEPTH_SHARE = 0.8        # the tag's plane from the depth over this much of its quad about its centre
TAG_DEPTH_RMS_M = 0.003      # the D555's depth over the tag lies this near one plane, or the tag is not placed by it


def tag_centre_by_depth(corners_px, depth_m: np.ndarray, k, dist) -> np.ndarray | None:
    """Place the tag centre by its diagonal-crossing ray and a plane over its inner quad.
    Aligned depth avoids apparent-size bias; return None without enough planar depth."""
    import cv2


    c = np.asarray(corners_px, float).reshape(4, 2)
    d1, d2 = c[2] - c[0], c[3] - c[1]
    t = np.linalg.solve(np.column_stack([d1, -d2]), c[1] - c[0])[0]
    centre = c[0] + t * d1
    quad = centre + TAG_DEPTH_SHARE * (c - centre)
    keep = np.zeros(depth_m.shape, np.uint8)
    cv2.fillConvexPoly(keep, np.round(quad).astype(np.int32), 1)
    z = plane_depth(depth_m, k, dist, keep.astype(bool), centre, min_px=30, max_rms_m=TAG_DEPTH_RMS_M)
    if z is None:
        return None
    return np.append(cv2.undistortPoints(np.array([[centre]], float), k, dist).reshape(2), 1.0) * z


def placed_at(pose: np.ndarray, palette_tag: np.ndarray, tag_in_base) -> np.ndarray:
    """The level pose `pose` (base <- palette_root) moved so its tag's centre stands at tag_in_base; its yaw kept."""
    out = np.array(pose, float)
    out[:3, 3] += np.asarray(tag_in_base, float) - (pose @ palette_tag)[:3, 3]
    return out


def from_poses(arm: str, poses, measured_utc: str, palette_urdf: Path, source: str, detail: dict | None = None,
               *, agree_m: float = SHOTS_AGREE_M, agree_rad: float = SHOTS_AGREE_RAD) -> StationFix:
    """Mean several level palette positions/yaws; refuse disagreement beyond agree_m/agree_rad.
    Return a fix with its URDF ball position, exposure time, source and spread evidence."""
    poses = [np.asarray(pose, float) for pose in poses]
    if not poses:
        raise ValueError("no shot measured the station")
    yaws = np.array([math.atan2(pose[1, 0], pose[0, 0]) for pose in poses])
    yaw = math.atan2(float(np.mean(np.sin(yaws))), float(np.mean(np.cos(yaws))))
    position = np.mean([pose[:3, 3] for pose in poses], axis=0)
    shift = max(float(np.linalg.norm(pose[:3, 3] - position)) for pose in poses)
    turn = max(abs(math.remainder(float(y) - yaw, 2.0 * math.pi)) for y in yaws)
    if shift > agree_m or turn > agree_rad:
        raise ValueError(f"the {len(poses)} shots disagree by up to {shift * 1000:.1f} mm and {math.degrees(turn):.2f} "
                         "deg: the palette moved during the fix, or a shot saw something else")
    base_from_palette = _level(np.array([*position, yaw]))
    ball = (base_from_palette @ np.append(ball_in_palette(palette_urdf), 1.0))[:3]
    detail = dict(detail or {}, shots=len(poses), shot_spread_mm=round(shift * 1000.0, 3),
                  shot_spread_deg=round(math.degrees(turn), 3))
    return StationFix(arm, base_from_palette, ball, measured_utc, source, detail)


def station_moved(before: StationFix, after: StationFix, *, translation_m: float | None = None,
                  rotation_rad: float = MOVED_RAD) -> dict:
    """Compare one arm's fixes by ball displacement and palette rotation.
    Translation tolerance defaults to the looser source's measured repeatability."""
    if before.arm != after.arm:
        raise ValueError(f"fixes of two arms ({before.arm}, {after.arm}) are not comparable")
    tolerance = translation_m if translation_m is not None else max(MOVED_M.get(before.source, 0.003),
                                                                    MOVED_M.get(after.source, 0.003))
    shift = float(np.linalg.norm(np.asarray(after.ball) - np.asarray(before.ball)))
    turn = np.asarray(before.base_from_palette)[:3, :3].T @ np.asarray(after.base_from_palette)[:3, :3]
    angle = float(np.arccos(np.clip((np.trace(turn) - 1.0) / 2.0, -1.0, 1.0)))
    return {"moved": shift > tolerance or angle > rotation_rad, "shift_m": shift, "rotation_rad": angle,
            "tolerance_m": tolerance, "before_utc": before.measured_utc, "after_utc": after.measured_utc}


# --- the station touched -----------------------------------------------------------------------------------------
# The overhead fix places the palette through the arm's registration, which has put the probe's ball 3-8 mm from
# where the arm's touches find it (2026-09-29..10-03), against cap bores 7-14 mm across. A dip aims by the touches
# instead: the calibration's S1 measures the ball's centre in the arm's base through the tcp it commands, and the
# caps follow from the CAD offsets about it. Touches and dip share the arm's kinematics and the tcp's length, so
# those errors cancel over the 40 mm from the ball to a cap. The fix still supplies the palette's yaw and says
# whether the palette has moved since.
TOUCH_SCHEMA = "tatbot.station-touch/1"
TOUCH_TCP_CHANGE_M = 0.005   # a tcp moved further than this since the touches is another tool's seat: touch again


def touch_path(arm: str) -> Path:
    return Path(f"~/tatbot-ros/calib/station-touch-{arm}.json").expanduser()


@dataclass
class StationTouch:
    arm: str
    tool_id: str
    ball: np.ndarray            # the probe ball's centre in <arm>/base_link, as the touches' tcp met it
    rotation: np.ndarray        # the tool's rotation in the base at those touches
    tcp_m: np.ndarray           # <arm>/tool_mount -> tcp, the tcp the touches commanded
    joint_offsets: list         # config/workspace.yaml joint_offsets_rad the kinematics carried
    fix: StationFix             # the overhead fix the touches started from
    run: str
    measured_utc: str

    def as_dict(self) -> dict:
        return {"schema": TOUCH_SCHEMA, "arm": self.arm, "tool_id": self.tool_id, "ball": np.asarray(self.ball).tolist(),
                "rotation": np.asarray(self.rotation).tolist(), "tcp_m": np.asarray(self.tcp_m).tolist(),
                "joint_offsets": [float(v) for v in self.joint_offsets], "fix": self.fix.as_dict(), "run": self.run,
                "measured_utc": self.measured_utc}

    @classmethod
    def from_dict(cls, data: dict) -> StationTouch:
        if data.get("schema") != TOUCH_SCHEMA:
            raise ValueError(f"not a {TOUCH_SCHEMA} record")
        return cls(data["arm"], data["tool_id"], np.asarray(data["ball"], float), np.asarray(data["rotation"], float),
                   np.asarray(data["tcp_m"], float), list(data["joint_offsets"]), StationFix.from_dict(data["fix"]),
                   data["run"], data["measured_utc"])

    def write(self, path: Path | None = None) -> Path:
        import json

        path = path or touch_path(self.arm)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.as_dict(), indent=1) + "\n")
        return path


def joint_offsets(repo: Path, arm: str) -> list[float]:
    """config/workspace.yaml `<arm>.joint_offsets_rad`, the model correction the kinematics carry (zeros if none)."""
    import yaml

    section = yaml.safe_load((Path(repo) / "config" / "workspace.yaml").read_text()).get(arm) or {}
    return [float(v) for v in section.get("joint_offsets_rad") or [0.0] * 7]


def load_touch(arm: str, path: Path | None = None) -> StationTouch:
    """The arm's newest station touch, or StaleStationError naming how to measure one."""
    import json

    path = path or touch_path(arm)
    if not path.is_file():
        raise StaleStationError(f"the station has not been touched with the {arm} arm: tatbot ros calib run --arm "
                                f"{arm} --station-only")
    return StationTouch.from_dict(json.loads(path.read_text()))


def touched_palette(touch: StationTouch, fix: StationFix, *, tool_id: str, tcp_m, joint_offsets,
                    palette_urdf: Path) -> np.ndarray:
    """<arm>/base_link <- palette_root for aiming at the caps: the fresh fix's rotation about the touched ball.

    Refused (StaleStationError) when the touches cannot speak for now: another tool, another joint model, another
    registration (the fix would not be comparable), a tcp moved past TOUCH_TCP_CHANGE_M, or a palette the fix sees
    moved since. A tcp moved less (a probe calibration adopted since) carries the ball with it: at the touches'
    rotation the same joints put the new tcp R (t_new - t_old) further on."""
    why = None
    if touch.arm != fix.arm:
        why = f"the touch is the {touch.arm} arm's"
    elif touch.tool_id != tool_id:
        why = f"the station was touched with {touch.tool_id}, and {tool_id} is fitted"
    elif not np.allclose(np.asarray(touch.joint_offsets, float), np.asarray(joint_offsets, float), atol=1e-6):
        why = "the joint offsets changed since the touch"
    elif touch.fix.detail.get("registration_sha256") != fix.detail.get("registration_sha256"):
        why = "the arm's registration changed since the touch"
    delta = np.asarray(tcp_m, float) - touch.tcp_m
    if why is None and float(np.linalg.norm(delta)) > TOUCH_TCP_CHANGE_M:
        why = f"the tcp moved {np.linalg.norm(delta) * 1000:.1f} mm since the touch"
    if why is None:
        moved = station_moved(touch.fix, fix)
        if moved["moved"]:
            why = (f"the palette moved {moved['shift_m'] * 1000:.1f} mm, {math.degrees(moved['rotation_rad']):.2f} deg "
                   f"since the touch ({touch.fix.measured_utc})")
    if why is not None:
        raise StaleStationError(f"{why}: touch the station again (tatbot ros calib run --arm {fix.arm} --station-only)")
    ball = touch.ball + touch.rotation @ delta
    return anchored(fix.base_from_palette, ball, ball_in_palette(palette_urdf))
