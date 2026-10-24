"""The wrist D405 as a gauge of the pen's height over the paper (ros/README.md 4.3).

The pen and the wrist camera both ride link 6, so the pen's tip is one fixed point in the camera's frame, and a
depth frame of the paper beside the pen gives the paper's plane there. The tip's height over that plane needs
neither the arm's kinematics nor the camera's mounting: on 2026-10-03 sixteen contacts from three touch runs read
within 0.25 mm (1 sigma) of each other, while the arm's own tip heights for them spread 4 mm with the tool's turn.

The tip is fitted from touch snapshots (touches/<touch>-<trip>.npz and .png, which the executor saves at every
trip): the pen's points pooled over every frame and taken at their end along the paper's normal, then the offset
that puts the lowest cluster of trips, the contacts, at zero (`ros2 run tatbot_session gauge fit RUN... --write`).
It holds while the tool and the camera stay mounted as they are. Once fitted, the pen is found by its place
on the fitted axis rather than its colour, so any ink's cartridge serves (on 2026-10-03's frames of the teal
sky-blue cartridge the two ways' heights agreed within 0.12 mm, 1 sigma). Its end along the axis does not show
a moved pen: without the colour, the depth's mixed pixels between the tip and the paper run on along the axis.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import numpy as np

CALIB = Path("~/tatbot-ros/calib").expanduser()
PEN_HUE = (78, 100)        # the ballpoint's teal body in OpenCV's 0-180 hue
PEN_NEAR_M = 0.17          # the pen rides 100-150 mm from the camera; anything teal farther is the room
PAPER_BOX_PX = 150         # the paper fitted: within this of the pen in the image, and behind its middle
CONTACT_BAND_M = 0.001     # a fit's contacts: the trips within this of the lowest decile
AXIS_M = 0.003             # a fitted pen's points: within this of its axis (the cone's narrow end) ...
CONE_M = (-0.015, 0.0025)  # ... and this far along it from the fitted cone end (a hover keeps the paper ~6 mm past it)
BODY_M = 0.008             # no paper within this of the axis short of the cone end (the cartridge and machine)
SURFACE_S_MAX = 150        # the surface drawn on is bright and not strongly coloured: white paper and pink silicone
                           # (S ~75) alike; the print's dots and the arm's shadow drop out by value. Not silicone's
                           # alone: human skin of Monk tones 1-5 is S 19-77 in neutral light, 62-115 in warm, tone 6
                           # 118-147, a violet transfer on any of them S 44-100 (surface_colour, test_gauge)
SURFACE_V_MIN = 45         # ... and not dark: Monk tones 1-9 are V 58-255, tone 10 (V 41) is not read
# A small surface square to the pen (`near_m`: the probe body's top under the upright pen) is read along the pen's axis,
# as its densest band this thick past the tip, when that band holds this share of what lies near the axis. From the
# sides where the pen hides the top, that is mostly the body's wall, which runs along the pen: a plane through it read
# the top 4-13 mm low (2026-10-22: 82-94 % of such supports on the wall). The pen stands 2.9 deg off the top's normal.
SQUARE_BAND_M = 0.002
SQUARE_SHARE = 0.5


def surface_colour(hsv: np.ndarray) -> np.ndarray:
    """Which OpenCV HSV pixels (..., 3) can be the surface drawn on: not strongly coloured, not dark."""
    hsv = np.asarray(hsv).astype(int)
    return (hsv[..., 1] < SURFACE_S_MAX) & (hsv[..., 2] > SURFACE_V_MIN)


def calibration_path(arm: str, role: str = "") -> Path:
    """The wrist gauge calibration of the arm's session camera, or of its camera with vision.toml role `role`."""
    return CALIB / (f"wrist-gauge-{arm}-{role}.json" if role else f"wrist-gauge-{arm}.json")


@dataclass
class Frame:
    """One depth frame, in the colour camera's frame: the pen's points and the paper's plane (c, n toward the
    camera), with the plane's rms. Measured `curved` (a curved page), c and n are the surface's point and normal under
    the tip from a quadric over the same points (_curved), and `patch` holds the surface points it kept."""
    pen: np.ndarray
    c: np.ndarray
    n: np.ndarray
    rms: float
    patch: np.ndarray | None = None


def _points(depth_raw, meta: dict, shape) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """The depth frame's points in the colour camera's frame and the colour pixel (u, v) each falls on."""
    depth = np.asarray(depth_raw, np.float32) * float(meta["depth_units_m"])
    di = meta["intrinsics"]
    v, u = np.mgrid[0:depth.shape[0], 0:depth.shape[1]]
    ok = (depth > 0.04) & (depth < 0.40)
    z = depth[ok]
    pts = np.stack([(u[ok] - di["ppx"]) / di["fx"] * z, (v[ok] - di["ppy"]) / di["fy"] * z, z], axis=1)
    ex = meta["color_from_depth"]
    pc = pts @ np.array(ex["rotation"], float).reshape(3, 3, order="F").T + np.array(ex["translation_m"], float)
    k = np.array(meta["color_intrinsics"]["k"], float)
    uc = np.round(k[0, 0] * pc[:, 0] / pc[:, 2] + k[0, 2]).astype(int)
    vc = np.round(k[1, 1] * pc[:, 1] / pc[:, 2] + k[1, 2]).astype(int)
    inside = (uc >= 0) & (uc < shape[1]) & (vc >= 0) & (vc < shape[0])
    return pc[inside], uc[inside], vc[inside]


def on_paper(depth_raw, meta: dict, frame: Frame, shape, within_m: float = 0.003, close_px: int = 9,
             trim_px: int = 7) -> np.ndarray:
    """The colour pixels whose depth puts them on this frame's paper (within within_m of its plane), the gaps
    between the depth's samples closed and the edges trimmed: the pen, the machine and the arm, whatever their
    colour, and the depth's holes about them are not paper."""
    import cv2

    pc, uc, vc = _points(depth_raw, meta, shape)
    on = np.abs((pc - frame.c) @ frame.n) < within_m
    out = np.zeros(shape[:2], np.uint8)
    out[vc[on], uc[on]] = 1
    out = cv2.morphologyEx(out, cv2.MORPH_CLOSE, np.ones((close_px, close_px), np.uint8))
    return cv2.erode(out, np.ones((trim_px, trim_px), np.uint8)).astype(bool)


def measure(depth_raw, color_bgr, meta: dict, cal: dict | None = None, curved: bool = False,
            near_m: float | None = None) -> Frame | None:
    """The pen and the paper in one frame (WristCamera depth, unaligned, and its colour); None when either is
    missing. The pen is teal (the sky-blue cartridge the fit was made with), or with a fitted `cal` the points on
    its axis by the cone end, whatever their colour; the paper (any surface drawn on) is bright and not strongly
    coloured (the arm's shadow darkens it, the print's dots drop out by value) and off the pen; with `near_m`, only
    within that of the pen's axis through the fitted tip, read as a small surface square to the pen (_square)."""
    import cv2

    pc, uc, vc = _points(depth_raw, meta, color_bgr.shape)
    hsv = cv2.cvtColor(color_bgr, cv2.COLOR_BGR2HSV)
    teal = ((hsv[..., 0] >= PEN_HUE[0]) & (hsv[..., 0] <= PEN_HUE[1]) & (hsv[..., 1] > 90)
            & (hsv[..., 2] > 70)).astype(np.uint8)
    seen = hsv[vc, uc].astype(int)
    paper = surface_colour(seen) & ~teal[vc, uc].astype(bool)
    if cal is not None and "cone_end_cam_m" in cal:
        s, off = _along(pc, cal)
        pen = (off < AXIS_M) & (s > CONE_M[0]) & (s < CONE_M[1]) & (pc[:, 2] < PEN_NEAR_M)
        paper &= ~((off < BODY_M) & (s < 0.001))
    else:
        core = cv2.erode(teal, np.ones((7, 7), np.uint8))   # off the body's edges, where depths blend
        pen = core[vc, uc].astype(bool) & (pc[:, 2] < PEN_NEAR_M)
    if pen.sum() < 100:
        return None
    pp = pc[pen]
    box = ((uc > uc[pen].min() - PAPER_BOX_PX) & (uc < uc[pen].max() + PAPER_BOX_PX)
           & (vc > vc[pen].min() - PAPER_BOX_PX) & (vc < vc[pen].max() + PAPER_BOX_PX))
    q = pc[paper & box & (pc[:, 2] > np.median(pp[:, 2]))]
    q = _support(q, cal, curved, near_m)
    if len(q) < 300:
        return None
    if near_m is not None and cal is not None and "axis_cam" in cal and "tip_cam_m" in cal:
        square = _square(q, cal)
        return None if square is None else Frame(pp, *square)
    keep = np.ones(len(q), bool)
    c = q.mean(axis=0)
    n = np.array([0.0, 0.0, 1.0])
    r = np.zeros(len(q))
    for _ in range(4):   # a plane, refitted to the points within 3 MADs (0.8 mm at least) of it
        if keep.sum() < 3:
            return None
        c = q[keep].mean(axis=0)
        n = np.linalg.svd(q[keep] - c, full_matrices=False)[2][-1]
        r = (q - c) @ n
        mid = np.median(r[keep])
        keep = np.abs(r - mid) < max(0.0008, 3.0 * np.median(np.abs(r[keep] - mid)))
    if keep.sum() < 3:
        return None
    if n @ c > 0:
        n = -n
    if curved and cal is not None and "tip_cam_m" in cal:
        return _curved(pp, q, c, n, np.asarray(cal["tip_cam_m"], float))
    return Frame(pp, c, n, float(np.std(r[keep])))


def _support(q, cal: dict | None, curved: bool, near_m: float | None) -> np.ndarray:
    """The surface points the plane is fitted to: on a curved page the skin about the tip, not the mat beside a
    raised one; with near_m only those within it of the pen's axis through the tip (a small surface)."""
    if cal is None or "tip_cam_m" not in cal:
        return q
    if curved:
        q = q[np.linalg.norm(q - np.asarray(cal["tip_cam_m"], float), axis=1) < SUPPORT_M]
    if near_m is not None and "axis_cam" in cal:
        q = q[_along(q, {**cal, "cone_end_cam_m": cal["tip_cam_m"]})[1] < near_m]
    return q


def _square(q, cal: dict) -> tuple[np.ndarray, np.ndarray, float] | None:
    """A small surface square to the pen, from the points near its axis: its densest SQUARE_BAND_M along the axis past
    the tip, when that holds SQUARE_SHARE of them. (Its point on the axis, the normal toward the camera, the band's
    spread), or None: then most of them run along the pen."""
    axis = np.asarray(cal["axis_cam"], float)
    s = _along(q, {**cal, "cone_end_cam_m": cal["tip_cam_m"]})[0]
    lo = np.sort(s)
    count = np.searchsorted(lo, lo + SQUARE_BAND_M) - np.arange(len(lo))
    i = int(np.argmax(count))
    if count[i] < SQUARE_SHARE * len(lo):
        return None
    band = lo[i:i + count[i]]
    return np.asarray(cal["tip_cam_m"], float) + float(np.median(band)) * axis, -axis, float(np.std(band))


CORE_M = 0.008            # a curved gauge leaves out the paper this near the foot of the tip ...
CURVED_MIN_POINTS = 500   # ... and needs this many points past it
SUPPORT_M = 0.040         # ... and reads only the surface within this of the tip: a skin pinned on a raised holder
                          # (crest ~50 mm over the mat) had the mat in the image box beside it, most of the support,
                          # and frames read 7-15 mm under one hold (2026-10-08)


def pool(depths, need: float = 0.6) -> np.ndarray:
    """One raw depth frame from several taken at one hold: each pixel's median over the frames that read it, kept
    where at least `need` of them did. On a pale skin under window glare (2026-10-08) most pixels by the tip read in
    only some frames, 2-5 mm apart, and single frames' curved readings spread 4 mm; medians of five spread ~1 mm."""
    stack = np.stack([np.asarray(d) for d in depths]).astype(np.float32)
    read = stack > 0
    stack[~read] = np.nan
    out = np.zeros(stack.shape[1:], np.float32)
    keep = read.mean(axis=0) >= need
    out[keep] = np.nanmedian(stack[:, keep], axis=0)
    return np.round(out).astype(np.asarray(depths[0]).dtype)


def _curved(pen, paper, c, n, tip) -> Frame | None:
    """The surface under the tip as a quadric w = k0 + k1 a + k2 b + k3 a^2 + k4 ab + k5 b^2 over the plane (a, b, w
    along n), fitted to the plane's own points (the same box, the same rejection, max(0.8 mm, 3 MADs)) bar those within
    CORE_M of the foot of the tip. One plane through a frame's skin sits under a forearm's crest (2.5-3 mm under a
    40-60 mm radius at the tip, synthetic frames). Within ~8 mm of the foot the depth reads the skin 1-3 mm off, either
    way (the pen's edges smear into it); a quadric fitted there read flat skin up to 2.6 mm from the plane gauge,
    whose five points agree to ~0.1 mm (2026-10-06's holds). Over the plane's support with the core left out it read
    the same 20 holds within +-0.35 mm (0.58 at most), their curvature under 10/m (a 40 mm forearm's is 25). On
    synthetic forearms it reads 0.13-0.19 mm short (a quadric misses the cylinder's quartic over that much of it): the
    pen that much shallower. None with under CURVED_MIN_POINTS points."""
    foot = tip - float((tip - c) @ n) * n
    x = np.cross(n, [0.0, 1.0, 0.0] if abs(n[1]) < 0.9 else [1.0, 0.0, 0.0])
    x /= np.linalg.norm(x)
    y = np.cross(n, x)
    d = paper - foot
    a, b, w = d @ x, d @ y, d @ n
    use = (np.hypot(a, b) > CORE_M) & (np.abs(w) < 0.010)
    if use.sum() < CURVED_MIN_POINTS:
        return None
    a, b, w, pts = a[use], b[use], w[use], paper[use]
    design = np.stack([np.ones_like(a), a, b, a * a, a * b, b * b], axis=1)
    keep = np.abs(w - np.median(w)) < 0.004
    k = np.zeros(6)
    r = np.zeros_like(w)
    for _ in range(6):
        if keep.sum() < 6:
            return None
        k = np.linalg.lstsq(design[keep], w[keep], rcond=None)[0]
        r = w - design @ k
        keep = np.abs(r) < max(0.0008, 3.0 * 1.4826 * float(np.median(np.abs(r[keep]))))
    if keep.sum() < 2:
        return None
    normal = n - k[1] * x - k[2] * y
    return Frame(pen, foot + k[0] * n, normal / np.linalg.norm(normal), float(np.std(r[keep])), pts[keep])


def ring(frame: Frame, inner_m: float = CORE_M, outer_m: float = 0.020) -> np.ndarray:
    """A curved frame's kept skin (Frame.patch, camera frame) between inner_m and outer_m of the foot of its tip,
    square to its normal: the skin about the tip the camera did see."""
    if frame.patch is None or not len(frame.patch):
        return np.zeros((0, 3))
    d = frame.patch - frame.c
    lateral = np.linalg.norm(d - np.outer(d @ frame.n, frame.n), axis=1)
    return frame.patch[(lateral > inner_m) & (lateral < outer_m)]


def load_snapshot(npz: Path, cal: dict | None = None) -> Frame | None:
    import cv2

    data = np.load(npz)
    return measure(data["depth_raw"], cv2.imread(str(npz.with_suffix(".png"))), json.loads(str(data["meta"])), cal)


def _along(points, cal: dict) -> tuple[np.ndarray, np.ndarray]:
    """Each point's distance along the fitted axis past the fitted cone end (toward the paper), and from the axis."""
    end, a = np.asarray(cal["cone_end_cam_m"], float), np.asarray(cal["axis_cam"], float)
    d = np.asarray(points, float) - end
    s = d @ a
    return s, np.linalg.norm(d - np.outer(s, a), axis=1)


def height(frame: Frame, cal: dict) -> float:
    """The pen's tip over the paper (m): the fitted tip's distance along the paper's normal, less the offset."""
    return float((np.asarray(cal["tip_cam_m"], float) - frame.c) @ frame.n) - float(cal["offset_m"])


def _axis(points) -> tuple[np.ndarray, np.ndarray]:
    """A line through the pen's points (a point on it, its unit direction), refitted to those within 3 mm of it:
    by the narrow tip the depth reads the paper behind it, and those points lie well off the pen."""
    keep = np.ones(len(points), bool)
    m = points.mean(axis=0)
    d = np.array([0.0, 0.0, 1.0])
    for _ in range(6):
        if keep.sum() < 2:
            break
        m = points[keep].mean(axis=0)
        d = np.linalg.svd(points[keep] - m, full_matrices=False)[2][0]
        off = np.linalg.norm((points - m) - np.outer((points - m) @ d, d), axis=1)
        keep = off < max(0.003, 2.5 * float(np.median(off[keep])))
    return points[keep].mean(axis=0), d


def fit(frames: list[Frame]) -> tuple[dict, np.ndarray]:
    """The pen's contact point in the camera frame, and every frame's height over its paper. The pen's pooled
    points give its axis (_axis) and the cone's end on it; the contact lies on the axis past that end by the
    length that puts the contacts at zero, the contacts being the frames within CONTACT_BAND_M of the lowest
    decile (the floor a trip cannot go under). The contact is a true point on the pen (the close-up anchors the
    camera's pose on it): the lowest pooled points the first fit took were paper read through the tip, ~15 mm
    deeper along its line of sight, which kept heights right and put the camera ~10 mm off sideways."""
    pool = np.concatenate([f.pen for f in frames])
    n0 = np.mean([f.n for f in frames], axis=0)
    n0 /= np.linalg.norm(n0)
    m, a = _axis(pool)
    if a @ n0 > 0:   # toward the paper, away from the camera's side of it
        a = -a
    s = (pool - m) @ a
    near = np.linalg.norm((pool - m) - np.outer(s, a), axis=1) < 0.003
    end = m + float(np.percentile(s[near], 99.5)) * a
    # each frame's distance from the cone's end to its paper along the axis; the contacts' is the tip's length
    reach = np.array([float((f.c - end) @ f.n) / float(a @ f.n) for f in frames])
    contacts = reach[reach <= np.percentile(reach, 10) + CONTACT_BAND_M]
    length = float(np.median(contacts))
    contact = end + length * a
    heights = np.array([float((contact - f.c) @ f.n) for f in frames])
    cal = {"tip_cam_m": contact.tolist(), "offset_m": 0.0, "cone_end_cam_m": end.tolist(), "axis_cam": a.tolist(),
           "tip_length_m": length, "normal_cam": n0.tolist(), "contacts": int(len(contacts)),
           "contact_sd_m": float(np.std(heights[reach <= np.percentile(reach, 10) + CONTACT_BAND_M])),
           "frames": len(frames)}
    return cal, heights


class WristGauge:
    """The real wrist D405 as the executor's sensing (ports.Sensing, wrist real): `count` frames of the executor's
    wrist camera at a hold (ArmExecutor._wrist_frames), their heights over the paper, the point under the tip, and on
    a curved page the skin's normal there and the skin about it (curved.hold_patch). The further cameras' frames
    (ArmExecutor._further_frames) are measured with their own calibrations; their heights join the session camera's
    when their median is within page.height.gauge_spread_m of its median (a mount or calibration off moves no hold).
    Both cameras are grabbed and measured at once: one after the other a hold stood ~4.6 s from arrival to its
    reading, the two cameras' 25 frames ~0.83 s each of it (2026-10-24)."""
    wrist = "real"

    def __init__(self, ex):
        self.ex, self.cal = ex, None

    def ready(self, arm: str) -> bool:
        """True with a gauge fitted for the fitted tool; a fit measured with another tool is refused (RuntimeError)."""
        from tatbot_session import config

        path = calibration_path(arm)
        if not path.exists():
            return False
        cal, fitted = json.loads(path.read_text()), config.fitted_tool(self.ex.repo, arm)
        if cal.get("measured_with") != fitted:   # 2026-10-13: a 3RL fit relabelled for the 11RL set every page
            with_, why = cal.get("measured_with") or "an unrecorded tool", cal.get("note") or cal.get("source") or ""
            raise RuntimeError(f"the wrist gauge ({path}) was measured with {with_} ({why}), not the fitted "
                               f"{fitted}: measure it with `tatbot ros calib wristcam`")
        self.cal = cal
        return True

    def gauge(self, q, tcp, prior, *, curved: bool, count: int, name: str = ""):
        from tatbot_session import curved as surface
        from tatbot_session import inspect as ins
        from tatbot_session.ports import GaugeRead

        ex, cal = self.ex, self.cal
        pooled = int(surface.config(ex.stack)["pool"]) if curved else 1

        def beside():   # the further cameras' frames, grabbed and measured while the session camera's are
            return [(view, other, seen, self._measured(seen, view["cal"], curved)) for view, other, seen in (
                (v, o, self._pooled(s, pooled)) for v, o, s in ex._further_frames(name, q, count, every=curved))]
        try:   # opened first: the further cameras are its previews
            if ex.wrist_on():
                ex.setup_camera()
        except RuntimeError:   # _wrist_frames says why and opens it afresh next time
            pass
        with ThreadPoolExecutor(1) as pool:
            others = pool.submit(beside)
            shots = self._pooled(ex._wrist_frames(name, q, count, every=curved), pooled)
            frames = [] if shots is None else self._measured(shots, cal, curved)
            others = others.result()
        heights = [height(f, cal) for f in frames if f is not None]
        if not heights:
            return None
        further = []
        for view, other, seen, got in others:
            hs = [height(f, view["cal"]) for f in got if f is not None]
            used = bool(hs) and abs(float(np.median(hs)) - float(np.median(heights))) <= ex._gauge_limit()
            further.append({"view": view, "camera": other, "shots": seen, "frames": got, "heights_m": hs, "used": used})
        heights = heights + [h for f in further if f["used"] for h in f["heights_m"]]
        normal, patch = np.asarray(prior, float)[:3, 2], np.zeros((0, 3))
        if curved:   # the skin's own normal under the tip, and the skin about it
            turn = surface.load_turn(ex.arm)   # the wrist camera's calibrated turn (gauge turn), else None
            base_from_camera = ex.kin.frame(q, ins.camera_frame(ex.arm))
            normal = surface.surface_normal_base(next(f for f in frames if f is not None), base_from_camera, turn)
            patch = np.concatenate([surface.hold_patch(f, cal, base_from_camera, tcp, turn) for f in frames
                                    if f is not None])
        point = np.asarray(tcp, float)[:3, 3] - float(np.median(heights)) * normal
        return GaugeRead(heights, point, normal, patch, (shots, frames, cal), tuple(further))

    @staticmethod
    def _pooled(shots, pooled: int):
        """Each reading from the pooled depth of `pooled` frames (pool) and the first colour frame of them, and every
        colour frame taken, which the print's markers are read in (inspect.hold_tip `every`)."""
        if shots is None or pooled <= 1:
            return shots
        groups = range(0, len(shots[1]) - pooled + 1, pooled)
        return [shots[0][k] for k in groups], [pool(shots[1][k:k + pooled]) for k in groups], shots[2], shots[0]

    @staticmethod
    def _measured(shots, cal: dict, curved: bool) -> list:
        """Each of `shots`' (colours, raw depths, meta) frames measured with `cal` (measure; None where no pen)."""
        return [measure(d, c, shots[2], cal, **({"curved": True} if curved else {}))
                for c, d in zip(shots[0], shots[1], strict=True)]


def wrist(stack: dict) -> str:
    """stack.yaml `sensing.wrist`: auto is the real wrist on real hardware, else none. Real hardware takes auto or
    real only (ValueError): a simulator's truth would set its page up unmeasured, and none draw a curved one flat."""
    kind, real = (stack.get("sensing") or {}).get("wrist", "auto"), stack.get("hardware") == "real"
    if kind not in (("auto", "real") if real else ("auto", "real", "sim", "none")):
        allowed = "auto or real on real hardware" if real else "auto, real, sim or none"
        raise ValueError(f"stack.yaml sensing.wrist is {allowed}, not {kind!r}")
    return "real" if real else ("none" if kind == "auto" else kind)


def sensing(stack: dict, ex):
    """The executor's sensing from stack.yaml `sensing.wrist` (wrist()): real, sim (tatbot_session.sim.TruthGauge
    over sensing.truth, a simulator's surface) or none."""
    from tatbot_session.ports import NoSensing

    wrist_ = wrist(stack)
    if wrist_ == "real":
        return WristGauge(ex)
    if wrist_ == "sim":
        from tatbot_session.sim import TruthGauge  # private: the simulator's

        return TruthGauge.from_stack(stack)
    return NoSensing()


def main(argv=None) -> int:
    doc = __doc__ or ""
    parser = argparse.ArgumentParser(prog="tatbot_session gauge", description=doc.split("\n\n")[0])
    sub = parser.add_subparsers(dest="action", required=True)
    f = sub.add_parser("fit", help="fit the pen's tip in the wrist camera's frame from touch runs' snapshots")
    f.add_argument("runs", nargs="+", help="ros-touch run directories whose touches/ hold trip snapshots")
    f.add_argument("--arm", default="right")
    f.add_argument("--tool", default="", help="the fitted tool, recorded with the fit")
    f.add_argument("--seed", default="", help="a previous fit (a wrist-gauge json) whose axis finds the pen whatever "
                   "its colour: another cartridge in the same mount; without it the pen must be the teal cartridge")
    f.add_argument("--write", action="store_true", help=f"adopt: write {CALIB}/wrist-gauge-<arm>.json")
    t = sub.add_parser("turn", help="fit the wrist camera's turn from curved setups' gauge holds (curved.fit_turn)")
    t.add_argument("runs", nargs="+", help="ros-touch or ros-draw run directories of curved setups (page.json with "
                   "gauge holds, touches/gauge-<n>.npz)")
    t.add_argument("--arm", default="right")
    t.add_argument("--write", action="store_true", help="adopt: write the turn to CALIB/wrist-turn-<arm>.json")
    args = parser.parse_args(argv)
    if args.action == "turn":
        return _turn(args)
    names, frames = [], []
    seed = json.loads(Path(args.seed).expanduser().read_text()) if args.seed else None
    for run in map(Path, args.runs):
        for npz in sorted(run.glob("touches/*.npz"), key=lambda p: [int(x) for x in p.stem.split("-")]):
            frame = load_snapshot(npz, seed)
            if frame is not None:
                names.append(f"{run.name} {npz.stem}")
                frames.append(frame)
    if len(frames) < 4:
        print(json.dumps({"ok": False, "message": f"{len(frames)} usable snapshots; a fit needs 4"}))
        return 1
    cal, heights = fit(frames)
    for name, frame, h in zip(names, frames, heights, strict=True):
        print(f"  {name}: {h * 1e3:+6.2f} mm over the paper (plane rms {frame.rms * 1e3:.2f} mm)")
    cal.update(arm=args.arm, tool=args.tool, measured_with=args.tool, runs=[Path(r).name for r in args.runs],
               utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))
    print(json.dumps({"ok": True, **cal}))
    if args.write:
        CALIB.mkdir(parents=True, exist_ok=True)
        calibration_path(args.arm).write_text(json.dumps(cal, indent=1) + "\n")
    return 0


def _setup(run: Path, cal: dict, kin, frame_name: str):
    """A curved setup's holds and check holds rebuilt from its saved frames (one per hold, its last try): page.json's
    used page and each hold's record, touches/gauge-<n>.npz's depth and joints."""
    import cv2

    from tatbot_session import curved

    page = json.loads((run / "page.json").read_text())
    last = {r["touch"]: r for r in page.get("gauge", [])}
    holds, checks = [], []
    for k, record in sorted(last.items()):
        npz = run / "touches" / f"gauge-{k}.npz"
        if not npz.exists():
            continue
        data = np.load(npz)
        frame = measure(data["depth_raw"], cv2.imread(str(npz.with_suffix(".png"))), json.loads(str(data["meta"])),
                        cal, curved=True)
        if frame is None:
            continue
        q = np.asarray(data["q"], float)
        entry = {"touch": k, "xy": record["xy"], "frame": frame, "patch_cam": curved.sample_patch(frame),
                 "tip_cam": np.asarray(cal["tip_cam_m"], float), "base_from_camera": kin.frame(q, frame_name),
                 "tcp": kin.fk(q), "height_m": float(record["height_m"]),
                 "agrees": curved.disagree(record) <= 0.0005}
        (checks if k >= 100 else holds).append(entry)
    return {"name": run.name, "page": np.asarray(page["used"], float), "size_m": page.get("size_m"), "holds": holds,
            "checks": checks}


def _point(entry: dict, turn) -> np.ndarray:
    from tatbot_session import curved

    return entry["tcp"][:3, 3] - entry["height_m"] * curved.surface_normal_base(entry["frame"], entry["base_from_camera"],
                                                                                 turn)


def _misses(setup: dict, cal: dict, turn) -> list[float]:
    """The setup's check holds against its field fitted with `turn` (None: the turn fitted per setup, and taken out of
    the checks' skin too), each as a curved page's check reads it (curved.check_miss)."""
    from tatbot_session import curved

    holds = [{"xy": h["xy"], "point": _point(h, turn) if h["agrees"] else None, "tip": h["tcp"][:3, 3],
              "patch": curved.hold_patch(h["frame"], cal, h["base_from_camera"], h["tcp"], turn)} for h in setup["holds"]]
    size = setup.get("size_m") or (0.083, 0.127)
    field, report = curved.fit(setup["page"], holds, {"degree": 4}, bounds=(size[0] / 2 - 0.002, size[1] / 2 - 0.002),
                               turned=turn is not None)
    tilt = None if turn is not None else report["camera_tilt_rad"]
    return [curved.check_miss(setup["page"], field, curved.hold_patch(c["frame"], cal, c["base_from_camera"], c["tcp"],
                                                                      turn), _point(c, turn), c["tcp"][:3, 3], tilt)[0]
            for c in setup["checks"] if c["agrees"]]


def _turn(args) -> int:
    """gauge turn: the camera's turn fitted run by run and over all of them, each run's check holds against fields
    without it (fitted per setup) and with the pooled one; --write adopts the pooled one."""
    from tatbot_description import robot_description
    from tatbot_motion import Kinematics

    from tatbot_session import curved
    from tatbot_session import inspect as ins

    cal = json.loads(calibration_path(args.arm).read_text())
    kin = Kinematics(robot_description(None, arms=(args.arm,)), args.arm)
    setups = [s for s in (_setup(Path(r), cal, kin, ins.camera_frame(args.arm)) for r in args.runs) if len(s["holds"]) >= 3]
    if not setups:
        print(json.dumps({"ok": False, "message": "no run with three holds rebuilt"}))
        return 1

    def runs_of(chosen):
        return [{"page": s["page"], "holds": [{**h, "point": _point(h, None) if h["agrees"] else None}
                                              for h in s["holds"]]} for s in chosen]
    pooled = curved.fit_turn(runs_of(setups))
    per_run = {}
    for s in setups:
        alone = curved.fit_turn(runs_of([s]))
        rows = {"turn_mrad": [round(v * 1e3, 1) for v in alone], "holds": len(s["holds"]),
                "checks_fitted_mm": [round(v * 1e3, 2) for v in _misses(s, cal, None)],
                "checks_turn_mm": [round(v * 1e3, 2) for v in _misses(s, cal, pooled)]}
        per_run[s["name"]] = rows
        print(f"  {s['name']}: {rows}")
    record = {"turn_cam_rad": [float(v) for v in pooled], "turn_deg": float(np.degrees(np.linalg.norm(pooled))),
              "arm": args.arm, "runs": [s["name"] for s in setups], "per_run": per_run,
              "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    print(json.dumps({"ok": True, **{k: v for k, v in record.items() if k != "per_run"}}))
    if args.write:
        CALIB.mkdir(parents=True, exist_ok=True)
        curved.turn_path(args.arm).write_text(json.dumps(record, indent=1) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
