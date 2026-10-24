"""Stencil tracking (ros/README.md 4.4): the pen tip on the print, measured with the wrist D405.

With `page.track.mode` log or correct, the wrist D405 stays open through a draw's op loop. Worker processes read
the fiducial-tile print's markers in each frame on the gauge's surface (inspect.hold_tip, accepted when two or more
agree, or one near its printed place), so the reading's ~0.3 s never holds the executor:
- the goal's own gauge holds, before the first stroke (mode correct);
- with `hover_stop`, a fresh frame at each stroke's hover before it descends, fitted at once (at_hover);
- a fresh frame at the end of each stroke's final lift;
- with `stroke_s`, a frame every stroke_s seconds while a stroke draws (while_drawing), grabbed off the executor's
  thread with the joints the frame arrived at: the filter's alone, no skin kept, one line a stroke.
Each reading is a page event and a row of <run>/track.jsonl, its frame kept under <run>/track/. A frame whose gauge
reads no surface by the tip is read by its own depth instead (print_read), against the goal's gauge holds read so, so
a hover can tell a gauge that missed from a sheet that left (ArmExecutor._print_stays).

An accepted reading gives a residual: where the pen inks (the gauge's tip point carried onto the print) less where
the stack believes the TCP is (FK in the run's page frame). With mode correct the executor moves each stroke's
points by the residual the filter predicts at them (page_filter.PageFilter, with the overhead's marker moves), so the
ink lands on the plan however the sheet has slid or turned and wherever the arm's error is; when the last strokes
still ended off the plan one way (Tracker.off_plan), it pauses before the next one.
"""
from __future__ import annotations

import itertools
import json
import multiprocessing
import time
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from pathlib import Path

import numpy as np

from tatbot_session import curved, gauge, geometry
from tatbot_session import inspect as ins
from tatbot_session.page_filter import PageFilter
from tatbot_session.ports import Event

_TILE = None


def consistently_off(offs, limit_m: float, count: int) -> np.ndarray | None:
    """The mean of the last `count` stroke ends' offsets from the plan when each of them and their mean lie past
    limit_m: the strokes landing off the plan one way, which a misread or the readings' scatter does not do. None
    otherwise, or with fewer than `count`."""
    offs = np.asarray(offs[-count:], float).reshape(-1, 2) if count > 0 else np.zeros((0, 2))
    if count < 1 or len(offs) < count:
        return None
    mean = offs.mean(axis=0)
    return mean if (np.linalg.norm(offs, axis=1) > limit_m).all() and np.linalg.norm(mean) > limit_m else None


# A tip read while its stroke drew is measured from the plan's line: the plan's point at the arc the tick asked for runs
# ~0.4 mm ahead of the pen (2026-10-24).
def off_line(points, xy) -> float:
    """How far xy lies from the polyline through `points` (N, 2)."""
    a, b = np.asarray(points, float)[:-1], np.asarray(points, float)[1:]
    t = np.clip(np.einsum("ij,ij->i", xy - a, b - a) / np.maximum(np.einsum("ij,ij->i", b - a, b - a), 1e-18), 0.0, 1.0)
    return float(np.min(np.linalg.norm(a + t[:, None] * (b - a) - xy, axis=1)))


def point_at(op: dict, arc_m: float) -> np.ndarray:
    """The plan's point `arc_m` along a stroke (page metres)."""
    p = np.asarray(op["points_m"], float)[:, :2]
    s = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(p, axis=0), axis=1))]
    return np.array([np.interp(arc_m, s, p[:, 0]), np.interp(arc_m, s, p[:, 1])])


def _init(repo: str, pattern_id: str) -> None:
    global _TILE
    from tatbot_bridge import capture

    capture._lib(Path(repo))
    _TILE = ins.print_artwork(pattern_id)


def _measure(job: dict) -> dict:
    """In the worker: keep the frame under track/ (unless it is kept already), read it, return the log row. A
    reading the run's page cannot take is tried again from the overhead's page when it has seen the sheet move:
    the markers are read through the page's map, which reaches only a centimetre past the page believed."""
    import cv2

    if job.get("png"):
        cv2.imwrite(job["png"], job["bgr"])
        np.savez_compressed(job["npz"], depth_raw=job["depth"], q=job["q"], meta=json.dumps(job["meta"]))
    t0, shown = time.perf_counter(), {}   # the fit's own cost, not the session's time; the markers read (shown)
    out = _fit(job, job["used"], shown)
    read = out.get("surface", True)   # the gauge read a surface by the tip: no prior helps a frame where it did not
    if read and not out.get("accepted") and job.get("overhead") is not None:
        again = _fit(job, job["overhead"])
        if again.get("fit") and again["margin"] > out.get("margin", -1):
            out = {**again, "prior": "overhead"}
    if job.get("seed") or not read:   # the goal's gauge holds read it too: what a later reading is held to
        out.update(print_read(job, markers=shown))
    if not read:
        off = np.asarray(out.get("print_off_m", [np.nan] * 2)) * 1e3
        out["error"] += (f"; {out['print_none']}" if "print_none" in out else
                         f"; print {off[0]:+.1f} {off[1]:+.1f} mm off the page by {out['print_markers']} markers")
    if job.get("curved") and job.get("tcp") is not None:
        out["skin_base"], out["skin_wide_base"] = _skin(job)
    out["fit_s"], out["shown"] = round(time.perf_counter() - t0, 3), shown
    if job.get("utc"):   # a frame taken while the stroke drew (Tracker.while_drawing), at its own time
        out.update(moving=True, utc=job["utc"])
    return out


SKIN_POINTS = 200   # a curved reading's skin about the tip kept for the executor's skin watch
WIDE_POINTS = 4000  # ... and all it saw past the tip's 8 mm, for a stroke's ride along its path (curved.skin_along)
WIDE_KEEP_S = 120.0  # ... kept this long (the executor borrows them for page.surface.rigid.profile_keep_s)


def _skin(job: dict) -> tuple[list | None, list | None]:
    """A curved page's skin the frame sees 8-20 mm about the tip (gauge.ring), and 8-40 mm (the gauge's support), each
    carried into the base as the gauge holds' patches are (curved.hold_patch: the camera's pose, pinned at the arm's
    tip; turned by the camera tilt the setup's fit took out of them, curved.untilt) and thinned to SKIN_POINTS and
    WIDE_POINTS; None when the gauge reads no skin. The executor reads them against the surface in use
    (ArmExecutor._skin_watch, _ride_over)."""
    frame = gauge.measure(job["depth"], job["bgr"], job["meta"], job["cal"], curved=True)
    ring = np.zeros((0, 3)) if frame is None else gauge.ring(frame)
    if len(ring) < 50:
        return None, None
    tcp = np.asarray(job["tcp"], float)

    def carried(points, most):
        held = type("Held", (), {"patch": points[np.linspace(0, len(points) - 1, min(most, len(points))).astype(int)]})()
        return curved.untilt(job["used"], curved.hold_patch(held, job["cal"], np.asarray(job["base_from_camera"], float),
                                                            tcp, most=most), tcp[:3, 3], job.get("tilt")).tolist()
    return carried(ring, SKIN_POINTS), carried(gauge.ring(frame, outer_m=gauge.SUPPORT_M), WIDE_POINTS)


def _fit(job: dict, base_from_page, markers=None) -> dict:
    """The tip on the print by the print's markers in this frame, on the gauge's surface: inspect.hold_tip, the gauge
    holds' reading. A decoded marker names its place, so a reading needs no gate against band repeats."""
    if _TILE is None:
        return {"fit": None, "error": "no fiducial-tile print installed for the page"}
    local = {"curved": True} if job.get("curved") else {}   # a curved page reads the skin under the tip
    frame = gauge.measure(job["depth"], job["bgr"], job["meta"], job["cal"], **local)
    if frame is None:   # no pen or no surface by the tip: not a reading of the print (print_read reads it)
        return {"fit": None, "surface": False, "error": "gauge read no surface"}
    read = ins.hold_tip([job["bgr"]], [frame], job["meta"], job["tip_cam"],
                        np.linalg.inv(job["base_from_camera"]) @ base_from_page, job["page"], _TILE,
                        markers=markers, **({"surface": job["surface"]} if job.get("surface") else {}))
    if read is None:
        return {"fit": None, "error": "no marker of the print seen"}
    return {"fit": "markers", "tip_m": read["tip_m"], "cam_from_print": read["cam_from_print"],
            "inliers": read["markers"], "ids": read["ids"], "score": round(float(read["score"]), 3),
            "margin": round(float(read["margin"]), 3), "rms_m": round(float(read["rms_m"]), 6),
            "accepted": bool(read["accepted"])}


CORNER_PX = 4   # a marker corner's depth: the median over the depth pixels this near it


def print_read(job: dict, tile=None, markers=None) -> dict:
    """Where a frame's own depth puts the print's markers, the gauge aside: each corner lifted along its ray by the
    median depth about it (CORNER_PX), carried by the camera's FK pose onto the page in use along its normal and onto
    a curved page's print by its chart (the skin field's unrolling), no plane fitted to the skin. Two or more
    markers must agree, each keeping its printed shape (fiducials.stencil.fit_points). The FK pose is a few mm off,
    about alike at every hover, so a reading is held to the goal's gauge holds read the same way (Tracker.print_moved),
    not to the page. {"print_fit": the correction (tx, ty, turn), "print_at": the agreeing markers' printed centre,
    "print_off_m": where the page has that point less it, "print_markers", "print_sol": the fit} or {"print_none":
    why}. `tile`: the print's (the worker's own by default); a curved page's locate views read it so too. `markers` (a
    dict) takes every marker read: id -> corners in the frame's pixels."""
    tile = _TILE if tile is None else tile
    if tile is None:
        return {"print_none": "no fiducial-tile print installed for the page"}
    import cv2
    from fiducials import stencil  # on the path once the tile is loaded (inspect.print_artwork)

    reads = list(stencil.find(job["bgr"], tile).items())
    if job.get("frames"):   # a locate view's own frames read too: their median blurred a marker 3 of 5 read
        dets = [stencil.detector(tile), stencil.detector(tile, upsample=True)]
        for frame, det in itertools.product(job["frames"], dets):
            reads += list(stencil.find(frame, tile, det).items())
    found = _read_places(reads)   # (id, n) -> corners: one per place a code read at, a false read apart from a true
    if markers is not None:
        markers.update({k[0]: c for k, c in sorted(found.items(), key=lambda kc: -kc[0][1])})
    if not found:
        return {"print_none": "no marker of the print read"}
    k, dist = (np.asarray(job["meta"]["color_intrinsics"][key], float) for key in ("k", "coeffs"))
    pc, uc, vc = gauge._points(job["depth"], job["meta"], job["bgr"].shape)   # on the colour's pixels, undistorted
    z = np.zeros(job["bgr"].shape[:2])
    z[vc, uc] = pc[:, 2]
    keys, r = sorted(found), CORNER_PX
    rays = cv2.undistortPoints(np.concatenate([found[i] for i in keys]).reshape(-1, 1, 2).astype(float), k, dist)[:, 0]
    depth = []
    for u, v in np.round(rays * k[[0, 1], [0, 1]] + k[:2, 2]).astype(int):
        near = z[max(v - r, 0):v + r + 1, max(u - r, 0):u + r + 1]
        depth.append(float(np.median(near[near > 0])) if (near > 0).any() else np.nan)
    cam = np.c_[rays, np.ones(len(rays))] * np.asarray(depth)[:, None]
    seen = {i: cam[4 * n:4 * n + 4] for n, i in enumerate(keys) if np.isfinite(depth[4 * n:4 * n + 4]).all()}
    fit = lifted_fit(seen, tile, np.asarray(job["base_from_camera"], float), np.asarray(job["used"], float))
    if fit is None or fit["markers"] < 2 or not fit["accepted"]:
        why = [f"{len(found)} reads of {len({k[0] for k in found})} of the print's codes"]
        why += [f"{len(found) - len(seen)} with no depth at a corner"] if len(seen) < len(found) else []
        why += [fit["why"]] if fit is not None else []
        return {"print_none": "; ".join(why), "print_sol": fit}
    at = np.mean([tile.corners[i].mean(axis=0) for i in fit["ids"]], axis=0)
    move = [fit["tx_m"], fit["ty_m"], fit["theta_rad"]]
    return {"print_fit": move, "print_at": at.tolist(), "print_off_m": (seen_at(move, at) - at).tolist(),
            "print_markers": fit["markers"], "print_sol": fit}


LIFTED_AGREE_M = 0.0025   # two markers agree when every corner of one stands as far from each of the other as
# printed within this and LIFTED_STRETCH of it (stencil._consensus, the chord of a skin curved to the tile's radius
# allowed), each read within LIFTED_SIZE of its printed side, and the agreeing corners sit within LIFTED_RMS_M of the
# layout laid round the skin (stencil.bent), across each corner's ray: its depth noise lies along it. About the flat
# layout, in its plane, a forearm's curve read as error: over 57 locate views of forearm prints (2026-10-20/21, the
# skin's radius 44 mm median) the flat 1.08 mm median against 0.88 across the rays, at the cap 1.61 and 1.68 against
# 1.13 and 1.53; the silicone tile on its holder (190-220 mm) 0.89 and 0.78. A transfer on a forearm turned since it
# was laid gives with the skin: the faded heart's
# true 15 and 23 lifted 10.4-13.9 mm a side against 12 (-13..+16%), their corners -2.5..+1.9 mm off their printed
# distances (2026-10-21). A misread code stands a pitch (16.5 mm) or more from where it is printed, a false one off
# the print: either far past these.
LIFTED_SIZE, LIFTED_STRETCH = 0.25, 0.05
LIFTED_RMS_M = 0.0016
PLACE_PX = 12.0   # reads of one code within this of one another (their centres) are one place: its corners' median


def _read_places(reads) -> dict:
    """(id, n) -> (4, 2) corners: each code's reads grouped by place (PLACE_PX), each group's corners' median, the
    places most read first; a false read of a code elsewhere stays apart from the print's own (the consensus decides)."""
    out = {}
    for i in sorted({i for i, _ in reads}):
        groups: list[list] = []
        for c in (np.asarray(c, float) for j, c in reads if j == i):
            near = [g for g in groups if np.linalg.norm(np.mean(g, axis=0).mean(axis=0) - c.mean(axis=0)) < PLACE_PX]
            if near:
                near[0].append(c)
            else:
                groups.append([c])
        for n, g in enumerate(sorted(groups, key=len, reverse=True)):
            out[(i, n)] = np.median(np.stack(g), axis=0)
    return out


def lifted_fit(seen: dict, tile, base_from_camera: np.ndarray, used: np.ndarray) -> dict | None:
    """The print placed by its markers' corners as the frame's depth lifted them ((id, n) or id -> (4, 3) camera
    metres): each read kept at its printed size (LIFTED_SIZE), the most that keep their printed distances to one
    another (stencil._consensus within LIFTED_AGREE_M and LIFTED_STRETCH: a misread code stands a pitch away, a false
    one off the print), the layout fitted to them in 3-D (stencil.kabsch, laid round the skin's curve by
    stencil.bent), then put in `used`'s plane: the print's centre along its normal and its u axis turned into it. The
    page in use supplies neither shape nor tilt: projected onto a stale prior tilted off the skin, the markers' shapes
    and spacing shrank by its cosine and misplaced the print. No plane or field is fitted to the skin for depth. Keys
    as stencil.fit_points' (p_print = R(theta) q + t on the page in use) and "why" (what each check dropped); None
    with no marker of the print seen."""
    from fiducials import stencil

    seen = {(k if isinstance(k, tuple) else (k, 0)): np.asarray(c, float) for k, c in seen.items()
            if (k[0] if isinstance(k, tuple) else k) in tile.corners}
    if not seen:
        return None

    def side(c):
        return "/".join(f"{v * 1e3:.1f}" for v in np.linalg.norm(c - np.roll(c, 1, axis=0), axis=1))
    sized = {k: c for k, c in seen.items() if stencil._printed_size(c, tile.sides_m[k[0]], LIFTED_SIZE)}
    why = [f"{k[0]} off its printed size ({side(c)} mm)" for k, c in seen.items() if k not in sized]
    keep = stencil._consensus(tile, sized, LIFTED_AGREE_M, stretch=LIFTED_STRETCH) if sized else []
    why += [f"{k[0]} off the others' printed distances" for k in sized if k not in keep]
    if not keep:
        return None
    ids, points = [k[0] for k in keep], np.concatenate([sized[k] for k in keep])
    target = np.concatenate([stencil.target_corners(tile, i) for i in ids])
    flat = stencil.kabsch(target, points)
    pose, _, rms3, off = stencil.bent(tile, ids, points, flat)
    off -= points * np.sum(off * points, axis=1, keepdims=True) / np.sum(points ** 2, axis=1, keepdims=True)
    rms = float(np.sqrt(np.mean(np.sum(off ** 2, axis=1))))   # across each corner's ray (LIFTED_RMS_M)
    page_from_print = np.linalg.inv(used) @ base_from_camera @ pose @ np.diag([1.0, -1.0, -1.0, 1.0])
    yaw = float(np.arctan2(page_from_print[1, 0], page_from_print[0, 0]))
    c, s = np.cos(-yaw), np.sin(-yaw)
    rot = np.array([[c, -s], [s, c]])
    shift = -rot @ page_from_print[:2, 3]
    lone = len(ids) == 1 and stencil.lone_ok(tile, ids[0], rot, shift)
    accepted = rms <= LIFTED_RMS_M and (len(ids) >= 2 or lone)
    why += [f"{'+'.join(map(str, sorted(ids)))} agree" + (f", {rms * 1e3:.2f} mm rms past {LIFTED_RMS_M * 1e3:.1f}"
                                                         if rms > LIFTED_RMS_M else f", {rms * 1e3:.2f} mm rms")]
    why += [f"one alone, {np.hypot(*shift) * 1e3:.1f} mm from where the page puts it (a misread's reach)"] if (
        len(ids) == 1 and not lone) else []
    p = np.concatenate([tile.corners[i] for i in ids])
    spread = float(np.sqrt(np.mean(np.sum((p - p.mean(axis=0)) ** 2, axis=1))))
    sigma_xy = max(rms, 1e-4) / np.sqrt(len(p))
    return {"tx_m": float(shift[0]), "ty_m": float(shift[1]), "theta_rad": -yaw, "scale": [1.0, 1.0], "rms_m": rms,
            "sigma": [sigma_xy, sigma_xy, float(max(rms, 1e-4) / max(spread, 1e-3) / np.sqrt(len(p)))],
            "points": int(len(p)), "sides": ["markers"], "score": len(ids) / len(seen), "margin": float(accepted),
            "tone": "markers", "markers": len(ids), "seen": len(seen), "ids": sorted(ids), "accepted": accepted,
            "rms_3d_m": rms3, "tilt_deg": float(np.degrees(np.arccos(np.clip(abs(page_from_print[2, 2]), -1.0, 1.0)))),
            "why": ", ".join(why)}


def seen_at(fit, at) -> np.ndarray:
    """Where a reading puts print point `at` on the page: its correction (p_print = R q + t) undone."""
    tx, ty, turn = fit
    c, s = np.cos(turn), np.sin(turn)
    return np.array([[c, s], [-s, c]]) @ (np.asarray(at, float) - [tx, ty])


def print_offset(held, fit, at) -> float:
    """How far a reading (`fit`) puts the print, at its markers (`at`), from where the gauge holds' readings put it
    there (the median of theirs): the sheet's move under the camera, its slide and turn together."""
    return float(np.linalg.norm(seen_at(fit, at) - np.median([seen_at(h, at) for h in held], axis=0)))


def views(ex, roles, cal: dict) -> list[dict]:
    """The further wrist cameras read beside the session's (vision.toml roles on ex's arm, read through the session
    camera's wrist_tap preview: the filter's page.track.cameras, the set-up's page.locate.cameras), each with its
    serial, colour optical frame and gauge calibration: its own once measured (gauge.calibration_path with its role),
    else the session camera's (`cal`) carried onto it through the URDF, which puts the mount where its CAD does."""
    if not roles:
        return []
    import wrist_cameras  # scripts/lib, on the path with inspect

    repo, arm = Path(ex.repo), ex.arm
    mounted = {c["role"]: c for c in wrist_cameras.mounted_for_arm(repo, arm)}
    frames, out = wrist_cameras.optical_frames(repo, arm=arm, stream="color"), []
    for role in (r for r in roles if r in mounted):
        path = gauge.calibration_path(arm, role)
        own = json.loads(path.read_text()) if path.exists() else _carried(ex, cal, frames[role], role)
        out.append({"role": role, "serial": mounted[role]["serial"], "frame": frames[role], "cal": own,
                    "tip_cam": np.asarray(own["tip_cam_m"], float)})
    return out


def _carried(ex, cal: dict, frame: str, role: str) -> dict:
    """The session camera's gauge calibration in the frame of the camera at `frame`: its points and directions
    carried through the URDF's two fixed frames on the wrist."""
    q = ex.io.q.copy()
    view_from_cam = np.linalg.inv(ex.kin.frame(q, frame)) @ ex.kin.frame(q, ins.camera_frame(ex.arm))
    rot, at = view_from_cam[:3, :3], view_from_cam[:3, 3]
    points = {k: (rot @ np.asarray(cal[k], float) + at).tolist() for k in ("tip_cam_m", "cone_end_cam_m")}
    turned = {k: (rot @ np.asarray(cal[k], float)).tolist() for k in ("axis_cam", "normal_cam") if k in cal}
    return {**cal, **points, **turned, "source": f"the session camera's, carried onto {role} by the URDF"}


def view_camera(cam, view: dict):
    """A further view's camera as the session camera `cam`'s preview has it open (wrist_tap.preview), or None until
    it has."""
    preview = getattr(cam, "preview", None)
    return next((c for c in list(preview.cameras) if c.serial == view["serial"]), None) if preview else None


class Tracker:
    """One draw goal's wrist-camera tracking. `start` returns None when tracking is off or cannot run."""

    rigid: curved.RigidWatch | None = None   # a curved page's stencil watch (None: a flat page, or log only)
    skin: np.ndarray | None = None
    hover_frames = 1
    views: tuple | list = ()   # the filter's further wrist cameras (page.track.cameras, _views)
    _grabber, _grabs, drawing = None, (), None   # the filter's frames while a stroke draws (page.track.stroke_s)

    def __init__(self, ex, cam, pool, cal, cfg, centre=(0.0, 0.0)):
        self.ex, self.cam, self.pool, self.cal = ex, cam, pool, cal
        self.tip_cam = np.asarray(cal["tip_cam_m"], float)
        # a curved page's tracker moves no stroke's points unless page.surface.track_correct: across a skin pinned over a
        # holder its print readings ran 15-20% off the field's arc mm, and its correction carried a 20 mm stroke 5 mm
        # past the holds onto skin turned 18 deg, where the needle went through the skin into the holder (2026-10-08).
        # It watches the stencil instead (curved.RigidWatch), and the executor moves the whole surface with it.
        surface = curved.config(ex.stack) if ex._curved() else None
        correct = cfg.get("mode") == "correct"
        self.correcting = correct and (surface is None or bool(surface["track_correct"]))
        self.rigid = (curved.RigidWatch(**surface["rigid"]) if correct and surface is not None and not self.correcting
                      and surface["rigid"]["enabled"] else None)
        self.hover_frames = int(surface["rigid"]["hover_frames"]) if surface is not None else 1
        self.skin = None        # the newest accepted reading's skin about the tip (base, N x 3; a curved page's)
        self.wide: list = []    # strokes' readings' skin 8-40 mm about the tip: [(clock, op id, base N x 3)]
        self.views = self._views(cfg.get("cameras") or [])
        self.field = PageFilter(centre, cameras=1 + len(self.views),
                                **{k: float(v) for k, v in (cfg.get("filter") or {}).items()})
        self._fed = ex.overhead_at   # the overhead's readings fed to the filter up to this
        self.stroke_s = float(cfg.get("stroke_s", 0.0)) if self.correcting else 0.0
        self._grabber = ThreadPoolExecutor(1) if self.stroke_s > 0 else None
        self._grabs, self._grabbed, self._count, self.drawing = [], -np.inf, itertools.count(1), {}
        self.ends = []          # the stroke ends' offsets from the plan, newest last (off_plan)
        self.off_m, self.off_count = float(cfg.get("off_plan_m", 0.0015)), int(cfg.get("off_plan_count", 2))
        self.pending = []
        self.print_held, self.printed = [], []   # print_read of the gauge holds, and of the newest op's unread ones
        (ex.run_dir / "track").mkdir(exist_ok=True)

    def _views(self, roles) -> list[dict]:   # the filter's (page.track.cameras)
        return views(self.ex, roles, self.cal)

    @classmethod
    def start(cls, ex, since: float = 0.0, centre=(0.0, 0.0)) -> Tracker | None:
        """Open the camera and the worker; with mode correct, also fit the goal's gauge holds (written after
        `since`, the clock's utc) and wait for them, so the first stroke has a field. `centre`: the drawing's (page
        metres), which the filter's turn and the arm's gradient are about."""
        cfg = ex.stack["page"].get("track") or {}
        if cfg.get("mode", "off") not in ("log", "correct") or ex.sensing.wrist != "real" or ex.run_dir is None:
            return None
        path = gauge.calibration_path(ex.arm)
        if not path.exists():
            ex.event(Event.KIND_ERROR, "track: no wrist gauge calibration; not tracking")
            return None
        cal = json.loads(path.read_text())
        ex._settle()   # the holds' frames written: the seed reads them
        try:   # the setup's own camera, open already (ArmExecutor.setup_camera): ~2 s to open one
            cam = ex.take_camera() or ex.wrist_camera("")   # stack.yaml's (ins.configured_serial)
        except RuntimeError as error:
            ex.event(Event.KIND_ERROR, f"track: wrist camera: {error}; not tracking")
            return None
        # three workers read the seed's 9-11 holds at once (one took 6 s, 2026-10-20); drain logs them in order
        pool = ProcessPoolExecutor(3, mp_context=multiprocessing.get_context("spawn"), initializer=_init,
                                   initargs=(str(ex.repo), ex.stack["page"]["pattern_id"]))
        try:
            tracker = cls(ex, cam, pool, cal, cfg, centre)
            ex.event(Event.KIND_PAGE, "track: the pen tip on the print after every lift" +
                     (", each stroke corrected by it" if tracker.correcting else
                      ", the curved page following the stencil rigidly (no stroke's points move)" if tracker.rigid
                      else " (logged, no correction)"))
            for view in tracker.views:
                ex.event(Event.KIND_PAGE, f"track: {view['role']} read beside it, for the filter "
                         f"({view['cal'].get('source', 'its own calibration')})")
            if tracker.correcting or tracker.rigid:   # never a set-up's _new_page set up again (seed_since)
                tracker._seed(max(since, float(getattr(ex, "seed_since", 0.0))))
        except BaseException:   # the D405 and the worker freed now, for the next goal, not by the garbage collector
            cam.close()
            pool.shutdown(wait=False, cancel_futures=True)
            raise
        return tracker

    def _seed(self, since: float) -> None:
        import cv2

        for npz in sorted((self.ex.run_dir / "touches").glob("gauge-*.npz")):
            if npz.stat().st_mtime < since:
                continue
            z = np.load(npz, allow_pickle=True)
            self._submit(npz.stem, None, cv2.imread(str(npz.with_suffix(".png"))), z["depth_raw"],
                         json.loads(str(z["meta"])), np.asarray(z["q"], float), keep=False)
        self.drain(wait=True)

    def restart(self) -> None:
        """The stroke ends' history cleared once the page itself has followed the sheet (executor._follow_unread): the
        ends read before it stood on the page before the follow. The filter carries the page's follows in its model."""
        self.drain(wait=True)
        self.ends = []
        if self.rigid is not None:
            self.rigid.clear()

    def rebased(self) -> None:
        """The overhead's watch starts again from the sheet as it reads it now (ArmExecutor._rebase, before it
        clears the page's follows): the filter keeps what it predicts on the page in use, the sheet's part reset."""
        self.drain(wait=True)
        self.field.rebased(self.ex._followed())
        self._fed = self.ex.clock.now()

    def _submit(self, name, end, bgr, depth, meta, q, keep=True, op_id="", stamp_ns=None, camera: int = 0,
                shown=None, at=None) -> None:
        """A frame into the workers: by the session camera, or by the further view `camera` (1, ...: self.views), whose
        open camera `shown` publishes the markers read in it, as self.cam does the session camera's; `at`, a frame
        taken while the stroke drew: (its clock time, its utc), its skin not read."""
        used, view = self.ex.used(), self.views[camera - 1] if camera else None
        page = self.ex.stack["page"]
        job = {"bgr": bgr, "depth": depth, "meta": meta, "q": q, "used": used,
               "tip_cam": view["tip_cam"] if view else self.tip_cam, "cal": view["cal"] if view else self.cal,
               "page": {"size_m": page["size_m"], "clear_m": page["clear_m"]},
               "curved": self.ex._curved(), "surface": self.ex.surface(),   # a curved page's skin, read in print mm
               "overhead": self._overhead(used), "seed": end is None,
               "base_from_camera": self._camera(q) if view is None else self.ex.kin.frame(q, view["frame"]),
               "tcp": None if at else self.ex.kin.fk(q), "tilt": self.ex.skin_tilt(), "utc": at and at[1],
               "png": str(self.ex.run_dir / "track" / f"{name}.png") if keep else None,
               "npz": str(self.ex.run_dir / "track" / f"{name}.npz")}
        tcp = self.tcp_on_print(used, self.ex.kin.fk(q), job["surface"])   # print mm, as the tip's
        follow = self.ex._followed()   # the page the frame was taken on
        future = self.pool.submit(_measure, job)
        if stamp_ns is not None:   # a frame the tap published: its markers onto the viewer's wrist pane once read
            reading = name.removesuffix(f"-{view['role']}") if view else name   # the pane names the camera
            future.add_done_callback(ins.wrist_tap.when_read(shown or self.cam, stamp_ns, bgr.shape, reading))
        self.pending.append((name, op_id, end, tcp, at[0] if at else self.ex.clock.now(), follow, camera, future))

    @staticmethod
    def tcp_on_print(used: np.ndarray, tcp: np.ndarray, surface=None) -> np.ndarray:
        """Where the stack believes the TCP is on the print: its shadow on a flat page; on a curved one the print point
        at the foot of the skin's normal through it, as the reading puts the tip (inspect.hold_tip). The shadow of a pen
        tilted to a flank's normal 10 mm over the skin lies sin(tilt) x 10 mm off its foot: 3.4 mm at 20 deg."""
        if surface is None:
            return (np.linalg.inv(used) @ tcp)[:2, 3]
        return surface.foot(np.asarray(tcp, float)[:3, 3])[0]

    def _camera(self, q) -> np.ndarray:
        """The wrist camera's pose at q: its CAD pose, on a curved page turned by its calibration (curved.camera_pose),
        as the page's skin was measured through it (a further view's is its CAD pose alone)."""
        from tatbot_session import curved

        pose = self.ex.kin.frame(q, ins.camera_frame(self.ex.arm))
        return curved.camera_pose(pose, self.tip_cam, curved.load_turn(self.ex.arm)) if self.ex._curved() else pose

    def _overhead(self, used: np.ndarray) -> np.ndarray | None:
        """The run's page moved in its own plane as the overhead reads the sheet moved since setup and the page has not
        followed (ArmExecutor.sheet_slide: by the print's markers when kept), never the overhead pose's tilt or height,
        which over a curved skin are a plane's (13.7 deg on a still forearm, 2026-10-19); None without a reading or
        when it moved under 2 mm and 1 deg (the same prior, near enough)."""
        slide = self.ex.sheet_slide()
        page = None if slide is None else slide @ np.linalg.inv(self.ex.slid) @ used
        return page if page is not None and geometry.page_moved(used, page, 0.002, 0.0175) else None

    def after_lift(self, op: dict) -> None:
        """A fresh frame now (the arm at the stroke's lifted end), fitted in the worker; the readings taken while the
        stroke drew logged first, and said in one line."""
        self._take(wait=True)
        self.drain(wait=self._grabber is not None)
        if self.drawing and (reads := self.drawing.pop(op["id"], [])):
            off = [off_line(np.asarray(op["points_m"], float)[:, :2], tip) for ok, tip in reads if ok]
            self.ex.event(Event.KIND_PAGE, f"track {op['id']}: {len(reads)} readings while it drew, {len(off)} taken"
                          + (f", the tip {np.median(off) * 1e3:.2f} mm off the plan's line (median)" if off else ""),
                          op["id"])
        end = np.asarray(op["points_m"][-1 if not op.get("closed") else 0], float)
        self._grab(op["id"], op["id"], end)

    def while_drawing(self, op: dict, arc_m: float) -> None:
        """The executor's tick while a stroke draws, `arc_m` along it: a frame every stroke_s, grabbed off its thread
        with the joints as the frame arrived (inspect's observe), read like a stop's but for the filter alone; one grab
        at a time, none while the workers are behind. A failure ends it, never the stroke."""
        if self._grabber is None:
            return
        try:
            self._take()
            self.drain()
            now = self.ex.clock.now()
            if self._grabs or len(self.pending) >= 3 or now - self._grabbed < self.stroke_s:
                return
            plan = point_at(op, arc_m)   # the plan's point the pen is at
            self._grabbed = now
            self._grabs.append((f"{op['id']}-draw-{next(self._count)}", op["id"], plan,
                                self._grabber.submit(self._grab_moving)))
        except Exception as error:  # noqa: BLE001 - tracking is a log: the stops' readings go on
            self.ex.event(Event.KIND_ERROR, f"track: while drawing: {error}; read at the stops alone from here")
            self._grabber = None

    def _grab_moving(self) -> tuple:
        clock, io = self.ex.clock, self.ex.io
        colors, _ = self.cam.grab_both(1, observe=lambda: (io.q.copy(), clock.now(), clock.utc()), fresh=True)
        q, t, utc = self.cam.capture_timestamps[-1]["joints"]
        return colors[-1], self.cam.raw_depth[0], self.cam.depth_metadata, q, (t, utc), getattr(self.cam, "stamp_ns", None)

    def _take(self, wait: bool = False) -> None:
        """The frames grabbed while a stroke drew into the workers, from the executor's thread (all of them with wait)."""
        while self._grabs and (wait or self._grabs[0][-1].done()):
            name, op_id, plan, future = self._grabs.pop(0)
            try:
                bgr, depth, meta, q, at, stamp = future.result()
            except Exception as error:  # noqa: BLE001 - one frame lost, the stroke and its stops' readings go on
                self.ex.event(Event.KIND_ERROR, f"track {name}: {error}", op_id)
                continue
            self._submit(name, plan, bgr, depth, meta, q, keep=False, op_id=op_id, stamp_ns=stamp, at=at)

    def at_hover(self, name: str, op_id: str, plan_xy) -> str | None:
        """A fresh frame now (the arm at rest over plan_xy, about to descend), fitted and logged before returning:
        the filter's verdict on it ('added', 'held' or 'moved'; with further views, the best of theirs), None when the
        print was not found or refused."""
        self._take(wait=True)
        self.drain(wait=True)
        self._grab(name, op_id, np.asarray(plan_xy, float))
        return self.drain(wait=True, best=True)

    def _grab(self, name: str, op_id: str, end: np.ndarray) -> None:
        colors, _ = self.cam.grab_both(self.hover_frames, fresh=True)   # taken where the arm is now
        # a curved page's frames pooled (gauge.pool): its skin under the tip is read as well as its print (skin watch)
        depth = self.cam.raw_depth[0] if len(self.cam.raw_depth) == 1 else gauge.pool(self.cam.raw_depth)
        self._submit(name, end, colors[-1], depth, self.cam.depth_metadata, self.ex.io.q.copy(), op_id=op_id,
                     stamp_ns=getattr(self.cam, "stamp_ns", None))
        for k, view in enumerate(self.views, 1):   # where the arm stands still: the views read the same place
            cam = self._view_camera(view)
            if cam is not None:
                colors, _ = cam.grab_both(1, fresh=True)
                self._submit(f"{name}-{view['role']}", end, colors[-1], cam.raw_depth[0], cam.depth_metadata,
                             self.ex.io.q.copy(), op_id=op_id, stamp_ns=getattr(cam, "stamp_ns", None), camera=k,
                             shown=cam)

    def _view_camera(self, view: dict):
        return view_camera(self.cam, view)

    def snapshot(self) -> PageFilter:
        self.drain()
        self._feed(self.ex.clock.now())
        return self.field.snapshot(self.ex._followed())

    def _feed(self, until: float) -> None:
        """The overhead's readings of the print's markers (ArmExecutor._timed_moves: the sheet's move since setup, by
        the markers it shares with the setup's reference) since the last fed up to `until`, into the filter."""
        for t, move in self.ex._timed_moves(self._fed):
            if t <= until:
                self.field.overhead(t, move[2], move[5])
                self._fed = t

    VERDICTS = (None, "held", "added", "moved")   # a stop's views: the best of theirs is its verdict

    def drain(self, wait: bool = False, best: bool = False) -> str | None:
        """Log the finished measurements in the order taken (all of them with wait); the filter's verdict on the last,
        or with `best` the best of theirs (a stop's views)."""
        verdict = None
        while self.pending and (wait or self.pending[0][-1].done()):   # in the order taken, whichever worker read it
            item = self.pending.pop(0)
            logged = self._log(*item[:5], item[-1].result(), follow=item[5], camera=item[6])
            verdict = max(verdict, logged, key=self.VERDICTS.index) if best else logged
        return verdict

    def _log(self, name: str, op_id: str, end, tcp, t: float, row: dict, follow=None, camera: int = 0) -> str | None:
        """Log a measurement. An accepted one is the filter's reading: its residual on the page it was taken on
        (`follow`, ArmExecutor._followed). A further view's (`camera`) is the filter's alone: the stroke ends, the
        print's reads and the skin are the session camera's."""
        row = {"op": name, "op_id": op_id, "plan_end_m": None if end is None else end.tolist(),
               "tcp_page_m": tcp.tolist(), **row, **({"camera": self.views[camera - 1]["role"]} if camera else {})}
        if "print_fit" in row and end is None:
            self.print_held.append(row["print_fit"])
        elif "print_fit" in row and not row.get("fit"):
            self.printed = [p for p in self.printed if p[0] == op_id] + [(op_id, row["print_fit"], row["print_at"])]
        if row.get("surface") is False:
            text = f"track {name}: {row['error']}"
        elif not row.get("fit"):
            text = f"track {name}: the print was not found ({row.get('error', 'no marker of the print seen')})"
        else:
            tip = np.asarray(row["tip_m"][:2])   # read on the gauge's surface, as the holds were
            text = f"track {name}: tip on print ({tip[0] * 1e3:+.2f}, {tip[1] * 1e3:+.2f}) mm"
            if end is not None:
                off = tip - end
                row["off_plan_m"] = off.tolist()
                text += f", {off[0] * 1e3:+.2f} {off[1] * 1e3:+.2f} mm from the plan"
            text += f"; {row['inliers']} markers, rms {row['rms_m'] * 1e3:.2f} mm"
            text += " (from the overhead's page)" if row.get("prior") == "overhead" else ""
            if row["accepted"]:
                resid = tip - tcp
                text += f"; ink {resid[0] * 1e3:+.2f} {resid[1] * 1e3:+.2f} mm from the TCP"
                if self.rigid is not None and end is not None:   # a curved page's watch: the gauge holds' field stays
                    off = self._watch(row, tcp, tip, resid, off)
                    text += {"added": "", "held": " (held: the stencil reads moved, a second must agree)",
                             "moved": " (a second agreed: the stencil moved)"}[row["field"]]
                    text += f"; {np.linalg.norm(row['moved_m']) * 1e3:.2f} mm from where the holds read it"
                else:
                    row.update(residual_m=resid.tolist(),
                               field=self._sample(t, tcp, resid, row, follow, end is not None, camera))
                    text += {"added": "", "held": " (held: far from the field, a second must agree)",
                             "moved": " (a second agreed: the sheet moved, the filter opens to it)"}[row["field"]]
                if name == op_id and end is not None and row["field"] != "held":   # a stroke's end (after_lift)
                    self.ends = [*self.ends[-8:], (op_id, off)]
            else:
                text += " (not accepted)"
        self._keep_skin(row, op_id, t, primary=not camera and not row.get("moving"))
        self._say(row, text, op_id)
        with open(self.ex.run_dir / "track.jsonl", "a") as f:
            f.write(json.dumps(row) + "\n")
        return row.get("field")

    def _say(self, row: dict, text: str, op_id: str) -> None:
        """A reading's page event; one taken while its stroke drew is counted for after_lift's line instead, unless it
        held or moved the sheet."""
        if row.get("moving"):
            self.drawing.setdefault(op_id, []).append((row.get("accepted"), np.asarray(row.get("tip_m", [0, 0]))[:2]))
        if not row.get("moving") or row.get("field") in ("held", "moved"):
            self.ex.event(Event.KIND_PAGE, text, op_id)

    def _keep_skin(self, row: dict, op_id: str, t: float, primary: bool) -> None:
        """The depth's skin a reading carries, read whether or not the print, out of its row: the newest about the tip,
        and a stroke's wider views kept for its ride; the session camera's alone (`primary`)."""
        skin, wide, _ = row.pop("skin_base", None), row.pop("skin_wide_base", None), row.pop("shown", None)
        if not primary:
            return
        self.skin = None if skin is None else np.asarray(skin, float)
        if wide is not None and op_id:
            self.wide = [w for w in self.wide if t - w[0] <= WIDE_KEEP_S] + [(t, op_id, wide)]

    def _sample(self, t: float, tcp, resid, row: dict, follow, check: bool, camera: int = 0) -> str:
        """An accepted reading into the filter by `camera`, after the overhead's readings up to it."""
        self._feed(t)
        return self.field.add(t, tcp, resid, check, rms=row["rms_m"], markers=row["inliers"], follow=follow,
                              camera=camera, moving=bool(row.get("moving")))

    def _watch(self, row: dict, tcp, tip, resid, off) -> np.ndarray:
        """A curved page's reading for the stencil watch (curved.RigidWatch): its residual against the gauge holds'
        field there (`moved_m`: the stencil's move since setup, in print mm), the watch's verdict as row["field"], and
        the stroke end's offset with the holds' residual taken out (what off_plan counts)."""
        base = self.field.predict([tcp])[0][0] if self.field.samples else np.zeros(2)
        moved = resid - base
        surface = self.ex.surface()
        exp_uv = np.asarray(tcp, float) + base
        act, exp = (np.array([tip, exp_uv]) if surface is None else surface.chart.to_plane([tip, exp_uv]))
        row.update(residual_m=resid.tolist(), moved_m=moved.tolist(), field=self.rigid.add(act, exp, tip, exp_uv))
        return off - base

    def off_plan(self, limit_m: float | None = None, count: int | None = None) -> tuple[np.ndarray, list] | None:
        """(their mean offset, [(op id, offset)]) when the last off_plan_count (`count`) strokes all ended past
        off_plan_m (`limit_m`) from the plan one way (consistently_off), else None. A stroke's end reads where it
        inked, after its own correction was taken at its hover; a hover's reading still carries the travel's
        pre-planned, older correction."""
        self.drain(wait=True)
        count = self.off_count if count is None else count
        ends = self.ends[-count:]
        limit = self.off_m if self.rigid is None else max(self.off_m, self.rigid.follow)   # under it nothing follows
        mean = consistently_off([off for _, off in ends], limit if limit_m is None else limit_m, count)
        return None if mean is None else (mean, ends)

    def print_moved(self, op_id: str) -> list[float]:
        """How far each of op_id's readings whose gauge read no surface puts the print from where the goal's gauge
        holds put it (print_read, print_offset: the FK pose's few mm cancel); none without a hold's reading."""
        if not self.print_held:
            return []
        return [print_offset(self.print_held, fit, at) for op, fit, at in self.printed if op == op_id]

    def close(self) -> None:
        try:
            self._take(wait=True)
            self.drain(wait=True)
        finally:
            if self._grabber is not None:
                self._grabber.shutdown(wait=True)
            self.pool.shutdown(wait=True)
            self.cam.close()
