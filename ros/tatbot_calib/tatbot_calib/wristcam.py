"""The wrist camera against the tip the arm commands, measured on the palette (ros/README.md section 7).

The wrist gauge reads the skin's height under the tip from the wrist D405: the skin it sees, against where the tip
stands in that camera's view (calib/wrist-gauge-<arm>.json `tip_cam_m`). That point was a fit made with another
cartridge and relabelled (2026-10-07), and the robot model's camera mount puts the old fit's tip 4, 5 and 11 mm
from where the fit had it (2026-10-13), so neither can be trusted for the fitted cartridge.

This measures the camera's mount instead. The station touch places the probe ball in the frame the arm commands
(its height by the palette camera, `calib height`), and the palette's roof tag stands at a fixed CAD offset from the
ball. The arm holds the tool over the palette at several poses, upright and tilted; at each the wrist camera's
colour stream sees the roof tag, and one PnP gives the camera to the tag. Every hold then says where the camera sits
on the tool mount: mount <- camera = inv(base <- mount by FK) . (base <- tag) . inv(camera <- tag). Their mean is the
mount, their spread its uncertainty. The gauge's tip in the camera follows from the mount and the fitted tool's tcp,
so the gauge, dips and strokes stand in the frame the arm commands; a tcp length error cancels between them.

The holds are poses clear of what way_check refuses, whose camera sees the whole tag (HandEye.candidates).
Every hold is a probe-guarded Touch through Calibration.go_to, kept clear of the station's parts: one joint move
(MODE_VIEW) where that keeps the tool off the station and the stylus, else the station's way (MODE_MOVE), the
nearest of the holds still wanted first (reach.nearest_first). The run writes holds.jsonl, frames/, candidate.json
and report.md; --adopt installs the gauge file (the old one kept beside it as .replaced-<utc>) when the holds agree
within MAX_TIP_SPREAD_M at the tip.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np

from tatbot_calib import halo, program, reach, station
from tatbot_calib.tool import ToolRefusedError, fitted_tool

HOLDS_WANTED = 15
TRIES = 4 * HOLDS_WANTED      # candidates tried at most: a refused way costs a check, a hold that sees nothing a move
FRAMES = 5
SETTLE_S = 1.0
MIN_HOLDS = 4
OUTLIER_PX = 3.0              # a hold the fit misses by more than this (rms over its corners) is left out
MAX_TIP_SPREAD_M = 0.0008     # the holds' tips in the camera agree within this (max from their mean) to adopt
CONE_BACK_M = 0.0043          # the gauge segments the pen from its cone end this far up its axis from the tip
HEIGHTS_M = (0.05, 0.08, 0.11)        # the tcp over the roof tag's plane
OFFSETS_M = (0.0, 0.025, 0.05)        # the tcp off the tag's centre, across, at AZIMUTHS about it
AZIMUTHS = 8
# Upright over the tag the arm has IK for headings -105..+30 (joints 3 and 4 stop at 90 deg); the way checks cut
# what the overhead camera's post blocks, which moves with every fix (2026-10-19: -75 and under, much of -60).
HEADINGS_DEG = (-90, -75, -60, -45, -30, -15, 0, 15, 30)
# (tilt, its axis from the heading), deg: upright holds all turn about the vertical, so they cannot tell the camera's
# place along the pen from the tag's height; tilted ones can.
TILTS_DEG = ((0, 0), (15, 0), (15, 90), (15, 180), (15, 270))
IMAGE_MARGIN_PX = 60          # over the robot model's mount's error: its tag corners stood 21-38 px off (2026-10-19)
# What hides the tag: a dark thing on its white border (one cell round its eight-cell black square) joins the black
# square and the detector finds no outline (2026-10-19: the e-stop's red cap at 5 holds, the needle at 2).
BORDER = 10 / 8
CAP_RADIUS_M = 0.022          # the e-stop's red cap, on the top of its keep-out (tatbot_motion.station.parts)
CAP_PX = 10                   # the cap's image this far off the tag's border
PEN_M = 0.10                  # the pen, up its axis from the tip
PEN_PX = 30                   # its axis this far off the tag's border
MAX_VIEW_DEG = 45             # the tag's normal from the way to it: holds that saw it stood 25-44 deg off (2026-10-19)


def _pose(r: np.ndarray, t) -> np.ndarray:
    out = np.eye(4)
    out[:3, :3], out[:3, 3] = r, np.asarray(t, float)
    return out


def mean_pose(poses: list[np.ndarray]) -> np.ndarray:
    from scipy.spatial.transform import Rotation

    return _pose(Rotation.from_matrix([p[:3, :3] for p in poses]).mean().as_matrix(),
                 np.mean([p[:3, 3] for p in poses], axis=0))


def solve_mount(holds: list[dict], base_from_tag: np.ndarray) -> list[np.ndarray]:
    """mount <- camera from each hold (base <- mount, camera <- tag)."""
    return [np.linalg.inv(h["base_from_mount"]) @ base_from_tag @ np.linalg.inv(h["camera_from_tag"]) for h in holds]


def fit_mount(holds: list[dict], base_from_tag: np.ndarray, model, k, dist, seed: np.ndarray) -> tuple[np.ndarray, float]:
    """mount <- camera by least squares over every hold's tag corners in the image (a hold's PnP depth, ~1-2 mm at
    20-25 cm, no longer averages in as a pose): the camera that projects the tag where each hold saw it. Its
    residual's median, px."""
    import cv2
    from scipy.optimize import least_squares
    from scipy.spatial.transform import Rotation

    tag_pts = (base_from_tag @ np.c_[model, np.ones(len(model))].T).T[:, :3]

    def pose(x):
        return _pose(Rotation.from_rotvec(x[:3]).as_matrix(), x[3:])

    def residuals(x):
        out = []
        camera_from_mount = np.linalg.inv(pose(x))
        for h in holds:
            camera_from_base = camera_from_mount @ np.linalg.inv(h["base_from_mount"])
            pts = (camera_from_base @ np.c_[tag_pts, np.ones(len(tag_pts))].T).T[:, :3]
            uv, _ = cv2.projectPoints(pts, np.zeros(3), np.zeros(3), k, dist)
            out.append(uv.reshape(-1, 2) - np.asarray(h["pixels"], float))
        return np.concatenate(out).ravel()

    x0 = np.r_[Rotation.from_matrix(seed[:3, :3]).as_rotvec(), seed[:3, 3]]
    result = least_squares(residuals, x0, loss="soft_l1", f_scale=1.0)
    per_hold = np.sqrt(np.mean(result.fun.reshape(len(holds), -1) ** 2, axis=1))
    return pose(result.x), float(np.median(np.abs(result.fun))), per_hold


def gauge_record(mount_from_camera: np.ndarray, tip_mount, arm: str, tool: str) -> dict:
    """The wrist gauge's file from the camera's mount and the tcp: the tip, the pen's axis (toward the skin) and its
    cone end in the camera's frame."""
    camera_from_mount = np.linalg.inv(mount_from_camera)
    tip = camera_from_mount[:3, :3] @ np.asarray(tip_mount, float) + camera_from_mount[:3, 3]
    axis = camera_from_mount[:3, :3] @ np.array([0.0, 0.0, 1.0])   # the tool's +z, from the mount toward its tip
    return {"tip_cam_m": tip.tolist(), "offset_m": 0.0, "cone_end_cam_m": (tip - CONE_BACK_M * axis).tolist(),
            "axis_cam": axis.tolist(), "tip_length_m": CONE_BACK_M, "normal_cam": (-axis).tolist(),
            "arm": arm, "tool": tool, "measured_with": tool, "source": "wrist hand-eye on the palette's roof tag (tatbot ros calib wristcam)",
            "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}


def gauge_mount(gauge: dict, cad: np.ndarray, tip_mount) -> np.ndarray:
    """mount <- camera with the tip and the pen's axis where the gauge has them in the camera (gauge_record's inverse),
    turned about the pen as `cad` has it."""
    camera_from_mount = np.linalg.inv(cad)
    have, want = camera_from_mount[:3, 2], np.asarray(gauge["axis_cam"], float) / np.linalg.norm(gauge["axis_cam"])
    axis = np.cross(have, want)
    r = halo.axis_rotation(axis / max(np.linalg.norm(axis), 1e-12), math.atan2(np.linalg.norm(axis), have @ want)) \
        @ camera_from_mount[:3, :3]
    return np.linalg.inv(_pose(r, np.asarray(gauge["tip_cam_m"], float) - r @ np.asarray(tip_mount, float)))


def spread(poses: list[np.ndarray]) -> list[int]:
    """The poses' order, each next the farthest from those before it: its position (m) plus 0.05 m for each radian the
    tool turns between them. Rounded to 0.1 mm, ties go to the earlier pose on every node."""
    if not poses:
        return []
    xyz, rot = np.array([p[:3, 3] for p in poses]), np.array([p[:3, :3] for p in poses])

    def far(i):
        turn = np.arccos(np.clip((np.einsum("nij,ij->n", rot, rot[i]) - 1.0) / 2.0, -1.0, 1.0))
        return np.linalg.norm(xyz - xyz[i], axis=1) + 0.05 * turn

    order, near = [0], far(0)
    while len(order) < len(poses):
        near[order] = -1.0
        order.append(int(np.argmax(np.round(near, 4))))
        near = np.minimum(near, far(order[-1]))
    return order


class HandEye:
    """One run's holds over the palette."""

    def __init__(self, cal: program.Calibration, base_from_tag: np.ndarray, camera, detector, model, frame: str,
                 gauge: dict | None = None):
        self.cal, self.base_from_tag, self.camera, self.detector, self.model = cal, base_from_tag, camera, detector, model
        self.frame, self.kin, self.gauge = frame, cal.kin, gauge
        self.holds: list[dict] = []

    def cad_mount(self, q=None) -> np.ndarray:
        q = np.zeros(7) if q is None else np.asarray(q, float)   # the camera rides link 6, the mount the carriage
        return np.linalg.inv(self.kin.frame(q, f"{self.cal.arm}/tool_mount")) @ self.kin.frame(q, self.frame)

    def image(self, pts) -> np.ndarray | None:
        """Points in the camera's frame in its image (px), None when one stands within 20 mm of its plane."""
        import cv2

        pts = np.atleast_2d(np.asarray(pts, float))
        if np.any(pts[:, 2] < 0.02):
            return None
        return cv2.projectPoints(pts, np.zeros(3), np.zeros(3), self.camera.k, self.camera.dist)[0].reshape(-1, 2)

    def unseen(self, camera_from_base: np.ndarray, pen_px, cap) -> str | None:
        """Why the camera would not see the whole roof tag, or None (IMAGE_MARGIN_PX, MAX_VIEW_DEG, its white border
        PEN_PX off the pen's points pen_px in the image and CAP_PX off the e-stop's cap about `cap` in the base)."""
        import cv2

        tag = (camera_from_base @ self.base_from_tag @ np.c_[np.r_[self.model, self.model * BORDER], np.ones(8)].T).T
        uv, outline, (w, h) = self.image(tag[:4, :3]), self.image(tag[4:, :3]), self.camera.size
        if uv is None or outline is None or min(uv.min(), w - uv[:, 0].max(), h - uv[:, 1].max()) < IMAGE_MARGIN_PX:
            return "the image's edge"
        normal, centre = camera_from_base[:3, :3] @ self.base_from_tag[:3, 2], tag[:4, :3].mean(axis=0)
        if math.degrees(math.acos(min(1.0, abs(normal @ centre) / np.linalg.norm(centre)))) > MAX_VIEW_DEG:
            return "too steep"
        outline = outline.astype(np.float32).reshape(-1, 1, 2)

        def off(point) -> float:   # px from the border's outline, negative inside it
            return -cv2.pointPolygonTest(outline, (float(point[0]), float(point[1])), True)

        if pen_px is not None and min(off(p) for p in pen_px) < PEN_PX:
            return "the pen"
        cap = None if cap is None else camera_from_base[:3, :3] @ cap + camera_from_base[:3, 3]
        cap_px = None if cap is None else self.image(cap)
        if cap_px is not None and off(cap_px[0]) - self.camera.k[0, 0] * CAP_RADIUS_M / cap[2] < CAP_PX:
            return "the e-stop's cap"
        return None

    def candidates(self) -> list[np.ndarray]:
        """Tcp poses over the roof tag (HEIGHTS_M, OFFSETS_M, AZIMUTHS, HEADINGS_DEG, TILTS_DEG) where the arm, IK from
        rest, stands clear of what way_check refuses (rows_check) and its camera sees the whole tag through the
        installed gauge's mount, else the robot model's (unseen); most varied first (spread). The ways between are
        run()'s."""
        from tatbot_session import ready

        q0 = np.zeros(7)
        tcp_from_mount = np.linalg.inv(self.kin.fk(q0)) @ self.kin.frame(q0, f"{self.cal.arm}/tool_mount")
        tip_mount = np.linalg.inv(tcp_from_mount)[:3, 3]
        mount_from_camera = gauge_mount(self.gauge, self.cad_mount(), tip_mount) if self.gauge else self.cad_mount()
        camera_from_mount = np.linalg.inv(mount_from_camera)
        pen = self.image((camera_from_mount @ np.append(tip_mount, 1.0))[:3]
                         - np.linspace(0.0, PEN_M, 21)[:, None] * camera_from_mount[:3, 2])
        cap = next((centre + radius * self.cal.station_now()[:3, 2] for name, centre, radius, _ in self.cal.zones()
                    if name == "palette_estop"), None)
        centre, found = self.base_from_tag[:3, 3], []
        for h in HEIGHTS_M:
            for r in OFFSETS_M:
                for k in range(AZIMUTHS if r else 1):
                    a = 2 * math.pi * k / AZIMUTHS
                    xyz = centre + [r * math.cos(a), r * math.sin(a), h]
                    for heading in HEADINGS_DEG:
                        for tilt, about in TILTS_DEG:
                            pose = halo.tool_pose(xyz, math.radians(heading), math.radians(tilt),
                                                  math.radians(heading + about))
                            camera = pose @ tcp_from_mount @ mount_from_camera
                            if self.unseen(np.linalg.inv(camera), pen, cap):
                                continue
                            try:
                                q = ready.solve_ik_seeded(self.kin, pose, reach.REST)
                                self.cal.rows_check("candidate", [q, q])
                            except (ValueError, RuntimeError):
                                continue
                            found.append(pose)
        return [found[i] for i in spread(found)]

    def grab(self, label: str, q) -> dict | None:
        import cv2

        colors, _ = self.camera.grab_both(FRAMES)
        out = self.cal.run.dir / "frames"
        out.mkdir(exist_ok=True)
        cv2.imwrite(str(out / f"{len(self.holds):02d}-{label}.jpg"), colors[0], [cv2.IMWRITE_JPEG_QUALITY, 92])
        corners = []
        for i, image in enumerate(colors):
            found = [d for d in self.detector.detect("wrist", image, time.time_ns() + i) if int(d.tag_id) == 0]
            if found:
                corners.append(np.asarray(found[0].corners_px, float).reshape(4, 2))
        if len(corners) < FRAMES // 2 + 1:
            return None
        pixels = np.median(corners, axis=0)
        ok, rvec, tvec = cv2.solvePnP(self.model, pixels, self.camera.k, self.camera.dist,
                                      flags=cv2.SOLVEPNP_IPPE_SQUARE)
        if not ok:
            return None
        return {"label": label, "q": np.asarray(q, float).tolist(), "frames": len(corners), "pixels": pixels,
                "camera_from_tag": _pose(cv2.Rodrigues(rvec)[0], tvec.ravel()),
                "base_from_mount": self.kin.frame(q, f"{self.cal.arm}/tool_mount")}

    def run(self) -> list[dict]:
        program._preflight(self.cal)
        pool = self.candidates()
        self.cal.say(f"{len(pool)} holds clear of the station see the tag; up to {TRIES} tried for {HOLDS_WANTED}")
        for i, pose in reach.nearest_first(self.kin, self.cal.rig.joints, pool[:TRIES],
                                           lambda: HOLDS_WANTED - len(self.holds)):
            label = f"hold{i:02d}"
            try:
                self.cal.go_to(label, pose)
            except RuntimeError as error:
                if program._latched(self.cal):
                    raise
                self.cal.say(f"{label}: skipped ({error})")
                continue
            self.cal.rig.spin(SETTLE_S)
            before = self.cal.rig.joints()
            row = self.grab(label, before)
            moved = float(np.max(np.abs(self.cal.rig.joints() - before)[:6]))
            if row is None or moved > 2e-3:
                self.cal.say(f"{label}: the roof tag was not seen" if row is None else f"{label}: the arm moved")
                continue
            self.holds.append(row)
            self.cal.log("holds.jsonl", {k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in row.items()})
            self.cal.say(f"{label}: the roof tag {np.linalg.norm(row['camera_from_tag'][:3, 3]) * 1000:.0f} mm from "
                         f"the camera in {row['frames']} frames")
        return self.holds


def solve(holds: list[dict], base_from_tag: np.ndarray, hand: HandEye, tip0, old: dict | None):
    """fit_mount over the holds, those it misses past OUTLIER_PX left out (MIN_HOLDS kept), each left out once more
    for the spread: (the holds kept, report's record and lines)."""
    k, dist = hand.camera.k, hand.camera.dist
    fitted, median_px, per_hold = fit_mount(holds, base_from_tag, hand.model, k, dist,
                                            mean_pose(solve_mount(holds, base_from_tag)))
    each = ", ".join(f"{h['label']} {px:.1f}" for h, px in zip(holds, per_hold, strict=True))
    worst = sorted(range(len(holds)), key=lambda i: -per_hold[i])
    dropped = [holds[i]["label"] for i in worst if per_hold[i] > OUTLIER_PX][:len(holds) - MIN_HOLDS]
    if dropped:
        holds = [h for h in holds if h["label"] not in dropped]
        fitted, median_px, per_hold = fit_mount(holds, base_from_tag, hand.model, k, dist, fitted)
    leave_out = [fit_mount([h for h in holds if h is not left], base_from_tag, hand.model, k, dist, fitted)[0]
                 for left in holds]
    result, lines = report(fitted, leave_out, tip0, hand.cad_mount(holds[0]["q"]), old)
    result.update(median_px=median_px, dropped=dropped)
    lines.insert(0, f"- the tag's corners fit every hold to {median_px:.2f} px (median)"
                 + (f"; left out, past {OUTLIER_PX:.0f} px rms: {', '.join(dropped)}" if dropped else ""))
    lines.insert(1, f"- each hold against one fit of all of them, px rms: {each}")
    return holds, result, lines


def report(mount: np.ndarray, leave_out: list[np.ndarray], tip_mount, cad: np.ndarray,
           old: dict | None) -> tuple[dict, list[str]]:
    """The fit, and how far leaving each hold out moves the tip in the camera (its spread)."""
    mounts = leave_out
    tips = [np.linalg.inv(m)[:3, :3] @ tip_mount + np.linalg.inv(m)[:3, 3] for m in mounts]
    tip = np.linalg.inv(mount)[:3, :3] @ tip_mount + np.linalg.inv(mount)[:3, 3]
    spread = float(max(np.linalg.norm(t - tip) for t in tips))
    cad_tip = np.linalg.inv(cad)[:3, :3] @ tip_mount + np.linalg.inv(cad)[:3, 3]
    out = {"mount_from_camera": mount.tolist(), "tip_cam_m": tip.tolist(), "tip_spread_m": spread,
           "holds": len(mounts), "cad_tip_cam_m": cad_tip.tolist(),
           "camera_off_cad_m": float(np.linalg.norm(mount[:3, 3] - cad[:3, 3]))}
    lines = [f"- holds: {len(mounts)}; leaving any one out moves the tip in the camera {spread * 1000:.2f} mm at most",
             f"- the camera sits {out['camera_off_cad_m'] * 1000:.1f} mm from the robot model's mount; through the "
             f"model the tip would read {np.round((cad_tip - tip) * 1000, 1).tolist()} mm off",
             f"- the tip in the camera: {np.round(tip * 1000, 2).tolist()} mm"]
    if old:
        lines.append(f"- the installed gauge's tip ({old.get('tool')}, {old.get('note') or old.get('source', '')}) "
                     f"stands {np.round((np.asarray(old['tip_cam_m']) - tip) * 1000, 1).tolist()} mm from it")
    return out, lines


def run(args) -> int:
    from tatbot_description import repo_root, robot_description
    from tatbot_motion import Kinematics, load_motion
    from tatbot_session import inspect as ins

    from tatbot_calib import sweep

    repo = repo_root(None)
    try:
        tool, halo_tip = fitted_tool(repo, args.arm, args.tool)
    except ToolRefusedError as error:
        print(json.dumps({"ok": False, "message": str(error)}))
        return 3
    sys.path.insert(0, str(repo / "scripts" / "lib"))
    sys.path.insert(0, str(repo / "scripts" / "vision"))
    import tatbot_runlog
    from fiducials import load_inventory, tag_model_corners
    from fiducials.detector import DetectorConfig, FiducialDetector

    fix = station.StationFix.from_dict(json.loads(Path(args.station).expanduser().read_text()))
    station.require_fresh(fix, args.run_started, max_age_s=args.max_age_s)
    kin = Kinematics(robot_description(None, arms=(args.arm,)), args.arm)
    run_log = tatbot_runlog.init(program.WORKFLOW, meta={"arm": args.arm, "tool": tool.tool_id, "method": "wristcam",
                                                         "station": args.station},
                                 attach_logging=False, argv=["tatbot_calib", "wristcam", "--arm", args.arm])
    cal = program.Calibration(None, run_log, repo, args.arm, halo_tip, 0.0, kin)
    sweep.equip(cal, fix, load_motion())
    touched = program.touched_fix(cal, fix, tool.tool_id)
    if touched is fix:
        cal.say("the station has no touch for this registration and tool: tatbot ros calib run --station-only, then "
                "tatbot ros calib height")
        run_log.finalize(1, status="fail")
        print(json.dumps({"ok": False, "run_dir": str(run_log.dir)}))
        return 1
    program.place_station(cal, touched)
    urdf = repo / "urdf" / "palette.urdf"
    base_from_tag = touched.base_from_palette @ station.palette_from_tag(urdf, require_confirmed=True)
    inventory = load_inventory(repo / "config" / "fiducials.json")
    target = inventory.target("palette")
    detector = FiducialDetector.from_inventory(inventory, DetectorConfig(min_side_px=8.0, refinement="edges",
                                                                        cell_margin=0.30), target="palette")
    camera = ins.WristCamera(serial=args.serial)
    camera.size = (1280, 720)
    cal.rig, code = program.Rig(args.arm), 1
    gauge_path = Path(f"~/tatbot-ros/calib/wrist-gauge-{args.arm}.json").expanduser()
    old = json.loads(gauge_path.read_text()) if gauge_path.is_file() else None
    try:
        hand = HandEye(cal, base_from_tag, camera, detector, tag_model_corners(target.edge_m),
                       ins.camera_frame(args.arm),
                       gauge=old if old and old.get("measured_with") == tool.tool_id else None)
        holds = hand.run()
        if len(holds) < MIN_HOLDS:
            cal.say(f"only {len(holds)} holds saw the roof tag, {MIN_HOLDS} needed: nothing measured")
        else:
            holds, result, lines = solve(holds, base_from_tag, hand, cal.tip0, old)
            record = gauge_record(np.asarray(result["mount_from_camera"]), cal.tip0, args.arm, tool.tool_id)
            record.update(holds=len(holds), tip_spread_m=result["tip_spread_m"], run=run_log.dir.name)
            (run_log.dir / "candidate.json").write_text(json.dumps({**result, "gauge": record}, indent=1) + "\n")
            (run_log.dir / "report.md").write_text("# Wrist camera on the tool mount\n\n" + "\n".join(lines) + "\n")
            for line in lines:
                cal.say(line)
            good = result["tip_spread_m"] <= MAX_TIP_SPREAD_M
            if args.adopt and good:
                if gauge_path.is_file():
                    gauge_path.rename(gauge_path.with_name(gauge_path.name + time.strftime(".replaced-%Y%m%dT%H%M%SZ",
                                                                                           time.gmtime())))
                gauge_path.write_text(json.dumps(record, indent=1) + "\n")
                cal.say(f"adopted: {gauge_path}")
            elif args.adopt:
                cal.say(f"not adopted: the holds' tips spread {result['tip_spread_m'] * 1000:.2f} mm, over "
                        f"{MAX_TIP_SPREAD_M * 1000:.1f}")
            code = 0 if good else 3
    except RuntimeError as error:
        cal.say(f"stopped: {error}")
    finally:
        camera.close()
        if not program._latched(cal):
            program.retreat(cal.rig, kin)
        cal.rig.close()
        run_log.finalize(code, status="ok" if code == 0 else "fail")
    print(json.dumps({"ok": code == 0, "run_dir": str(run_log.dir)}), flush=True)
    return code


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="tatbot_calib wristcam", description=__doc__.split("\n\n")[0])
    parser.add_argument("--arm", required=True)
    parser.add_argument("--tool", default=None)
    parser.add_argument("--station", required=True)
    parser.add_argument("--run-started", type=float, required=True)
    parser.add_argument("--max-age-s", type=float, default=station.MAX_AGE_S)
    parser.add_argument("--serial", default="")
    parser.add_argument("--adopt", action="store_true")
    return run(parser.parse_args(argv))


if __name__ == "__main__":
    sys.exit(main())
