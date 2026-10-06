"""The blue arm traces ink on the practice forearm: scan it, pick a stroke, follow it, score it.

    tatbot travel trace -- scan                      # survey from the seed pose, then views above the arm
    tatbot travel trace -- plan --scan RUN/scan.npz  # offline: strokes, the chosen plan, a preview image
    tatbot travel trace -- run  --scan RUN/scan.npz [--stroke N] [--reverse] [--speed M_S]
    tatbot travel trace -- astra --scan RUN/scan.npz [--speed M_S] [--seconds S]   # Astra picks the ink
    tatbot travel trace -- collect --scan RUN/scan.npz --dataset ROOT --episodes N  # FLUX demonstrations

Everything on the arm goes through ``hardware.BlueArm`` (lease, e-stop monitor, staging, landing) and
the visiond wrist stream. A stream that would carry the arm past the wall toward the pink arm
(``pen_path.wall_crossing``) is refused before it moves. While one plays, three physical states stop
it: the e-stop, the arm not following its commands (an obstruction), and the skin closer along the pen
axis than half the hover standoff in the live wrist depth. Each ends the stream where the arm stands,
then it rises and lands.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

from tatbot_travel import assets, inkmap, pen_path, surface
from tatbot_travel.camera import Intrinsics

# A downward survey pose known to see the forearm in the 2026-10-05 layout (gap ~0.15 m above the table).
SEED_Q = (0.0689, 0.9396, 1.0641, -0.8958, 0.1627, 1.4458)
VIEW_HEIGHT_M = 0.05  # working point above the skin for scan views
TRACK_RAD, TRACK_S = 0.1, 0.15  # measured behind commanded by this much for this long: obstructed
# The controller holds position with P only, against a gravity model of the leader's handle, not the
# pen and cradle: the wrist settles ~0.01-0.06 rad short (4 mm at the pen). Each tick adds this share of
# the measured shortfall (against the command two ticks back, the arm's tracking delay) to every
# command, the sum held within +-INTEGRATOR_MAX_RAD.
INTEGRATOR_GAIN, INTEGRATOR_MAX_RAD = 0.02, 0.05
CLEARANCE_STOP_FRACTION = 0.5  # stop when the lens is within half the standoff of the skin


def standoff_m() -> float:
    pen = pen_path.load_pen()
    return pen.tcp_z - pen.lens_z


class Rig:
    """The arm and its wrist camera for one run; landing is unconditional on exit."""

    def __init__(self, out: Path, estop_required: bool = True):
        from tatbot_travel.hardware import BlueArm
        from tatbot_travel.vision_camera import VisiondWristCamera

        self.out = out
        self.camera = VisiondWristCamera()
        self.camera.wait_ready()
        self.kin = pen_path.arm_kinematics(self.camera.intrinsics)
        self.arm = BlueArm(estop_required=estop_required)
        self.robot = None  # the arm's own pixels in the wrist view, made on first use
        self.offset = np.zeros(6)  # the integrator's correction, carried from one stream to the next

    def __enter__(self) -> Rig:
        self.arm.connect(self.out / "landing.json")
        return self

    def __exit__(self, *exc) -> None:
        try:
            outcome = self.arm.land()
            print(f"trace: arm {outcome}", file=sys.stderr)
        finally:
            self.camera.close()

    def capture(self, settle_s: float = 0.4) -> tuple[surface.Frame, np.ndarray]:
        """A wrist frame exposed after the arm settled, and the joints it was taken at."""
        after = time.monotonic() + settle_s
        while True:
            frame = self.camera.newest()
            if frame is not None and frame[2] > after:
                break
            time.sleep(0.01)
        q = self.arm.measured()
        rgb, depth, _ = frame
        cam_p, cam_r = self.kin.camera_pose(q)
        return surface.Frame(rgb=np.array(rgb), depth_m=np.array(depth), intr=self.camera.intrinsics,
                             cam_p=cam_p, cam_r=cam_r), q

    def play(self, q_traj: np.ndarray, *, guard_clearance: bool = False, record: list | None = None,
             on_tick=None) -> str | None:
        """Stream joints at the control rate. Returns why it stopped early, or None.

        ``on_tick(q_cmd, q_measured, frame)`` sees every commanded tick with the newest wrist frame."""
        if why := pen_path.wall_crossing(self.kin, q_traj):
            return f"refused before moving: {why}"
        period = 1.0 / pen_path.RATE_HZ
        self.behind_since = self.close_since = None
        next_tick = time.monotonic()
        for i, q in enumerate(q_traj):
            if self.arm.estopped:
                return "e-stop"
            now = time.monotonic()
            measured = self.arm.measured()
            if why := self._obstructed(measured, q_traj[i - 1] if i else measured, now):
                return why
            clearance = np.inf
            if guard_clearance:
                clearance, why = self._clearance(i, measured, now)
                if why:
                    return why
            if i >= 2:
                self._integrate(q_traj[i - 2] - measured)
            self.arm.command(q + self.offset, 2.0 * period)
            if on_tick is not None:
                on_tick(q, measured, self.camera.newest())
            if record is not None:
                record.append({"t": now, "i": i, "q_cmd": q.tolist(), "q": measured.tolist(),
                               "offset": np.round(self.offset, 5).tolist(),
                               "clearance_m": None if not np.isfinite(clearance) else round(float(clearance), 5)})
            next_tick += period
            time.sleep(max(0.0, next_tick - time.monotonic()))
        return None

    def _obstructed(self, measured: np.ndarray, sent: np.ndarray, now: float) -> str | None:
        lag = np.abs(measured - sent).max()
        self.behind_since = (self.behind_since or now) if lag > TRACK_RAD else None
        if self.behind_since is not None and now - self.behind_since > TRACK_S:
            self.arm.freeze()
            return f"obstructed: {lag:.2f} rad behind its command"
        return None

    def _clearance(self, i: int, measured: np.ndarray, now: float) -> tuple[float, str | None]:
        frame = self.camera.newest()
        if frame is None:
            return np.inf, None
        clearance = self.lens_clearance(measured, frame[1])
        if clearance < 0.03 and i % 15 == 0:
            self.keep_frame(f"near-{i:05d}", frame, measured, clearance)
        close = clearance < CLEARANCE_STOP_FRACTION * standoff_m()
        self.close_since = (self.close_since or now) if close else None
        if self.close_since is not None and now - self.close_since > 0.2:
            self.arm.freeze()
            self.keep_frame(f"stop-{i:05d}", frame, measured, clearance)
            return clearance, f"skin {clearance * 1000:.1f} mm from the lens"
        return clearance, None

    def _integrate(self, shortfall: np.ndarray) -> None:
        self.offset = np.clip(self.offset + INTEGRATOR_GAIN * shortfall, -INTEGRATOR_MAX_RAD, INTEGRATOR_MAX_RAD)

    def keep_frame(self, name: str, frame, q: np.ndarray, clearance: float) -> None:
        """A wrist RGB-D frame and its joints into the run directory, for reading a stop afterwards."""
        rgb, depth, _ = frame
        np.savez_compressed(self.out / f"frame-{name}.npz", rgb=np.asarray(rgb), depth=np.asarray(depth), q=q,
                            clearance_m=clearance)

    def lens_clearance(self, q: np.ndarray, depth_m: np.ndarray) -> float:
        """Lens face to the skin along the pen axis, in the live wrist depth (``surface.axis_to_skin``)."""
        from tatbot_travel.scene import LENS_SITE
        from tatbot_travel.selfview import robot_mask

        if self.robot is None:
            self.robot = robot_mask(self.kin, self.camera.intrinsics)
        gap, axis = self.kin.gap_pose(q)
        lens = self.kin.data.site_xpos[self.kin.model.site(LENS_SITE).id].copy()
        cam_p, cam_r = self.kin.camera_pose(q)
        return surface.axis_to_skin(np.asarray(depth_m), self.camera.intrinsics, (lens - cam_p) @ cam_r,
                                    axis @ cam_r, (gap - cam_p) @ cam_r, self.robot)

    def go(self, q_to: np.ndarray) -> str | None:
        return self.play(pen_path.plan_move(self.arm.measured(), q_to))


# --- scan -------------------------------------------------------------------------------------------

def scan_views(kin, frame: surface.Frame, q_seed: np.ndarray, n: int = 3) -> list[np.ndarray]:
    """Joint poses looking down at the forearm the seed view found, spread along its length."""
    points, _, _ = frame.points()
    plane = surface.table_plane(points)
    n_up, d = plane
    (arm,) = surface.voxel(points[surface.above(points, plane, 0.01)], 0.003)
    if len(arm) < 200:
        raise RuntimeError("the seed view sees nothing standing on the table")
    arm = arm[surface.largest_cluster(arm, link_m=0.008)]
    centre = arm.mean(axis=0)
    along = np.linalg.svd(arm[:, :2] - centre[:2], full_matrices=False)[2][0]
    lift = (arm @ n_up).max() - centre @ n_up + VIEW_HEIGHT_M  # the working point this far over the top
    views = []
    for s in np.linspace(-0.035, 0.035, n):
        target = centre + s * np.array([*along, 0.0]) + lift * n_up
        result = kin.solve(q_seed, target, -n_up, q_seed, iters=200)
        if result.ok():
            views.append(result.q)
    if not views:
        raise RuntimeError("no scan view above the forearm is reachable")
    return views


SURVEY_YAWS = (0.0, 0.35, 0.7)  # base turns tried from the seed: only toward +y, away from the pink arm


def survey(rig: Rig, seed: np.ndarray, *, enough: float = 0.08) -> tuple[surface.Frame, np.ndarray, np.ndarray]:
    """Find the forearm: from the seed pose turn the base until the wrist view is mostly forearm (the
    operator moves it between sessions). Returns the best view, its joints and that pose as the new seed."""
    best = None
    for yaw in SURVEY_YAWS:
        pose = seed.copy()
        pose[0] += yaw
        if rig.go(pose):
            break
        frame, q = rig.capture()
        points, _, _ = frame.points()
        try:
            fraction = float(inkmap.forearm_pixels(frame, surface.table_plane(points)).mean())
        except (ValueError, IndexError):
            fraction = 0.0
        print(f"survey: base {yaw:+.2f} rad, forearm {fraction:.0%} of the view", file=sys.stderr)
        if best is None or fraction > best[0]:
            best = (fraction, frame, q, pose)
        if fraction >= enough:
            break
    if best is None or best[0] < 0.02:
        raise RuntimeError("no survey view shows the forearm; place it in front of the blue arm")
    if rig.go(best[3]):
        raise RuntimeError("stopped returning to the best survey view")
    return best[1], best[2], best[3]


def save_scan(path: Path, frames: list[surface.Frame], qs: list[np.ndarray], over: dict | None = None) -> None:
    """The wrist views and their joints; ``over`` (bgr, depth, intrinsics) adds the overhead D555's view."""
    extra = {f"overhead_{k}": v for k, v in (over or {}).items()}
    np.savez_compressed(path, rgb=np.stack([f.rgb for f in frames]), depth=np.stack([f.depth_m for f in frames]),
                        q=np.stack(qs), intrinsics=json.dumps(frames[0].intr.__dict__), **extra)


def load_overhead(path: Path) -> surface.Frame | None:
    from tatbot_travel import overhead

    data = np.load(path, allow_pickle=True)
    if "overhead_bgr" not in data.files:
        return None
    return overhead.frame_of(data["overhead_bgr"][..., ::-1], data["overhead_depth"],
                             json.loads(str(data["overhead_intrinsics"])))


def overhead_preview(path: Path, scan_path: Path, skin: surface.Surface, plane: tuple,
                     strokes: list[inkmap.Stroke], chosen: int | None) -> dict | None:
    """Align the scan's overhead view by the forearm and draw the strokes on it: the audience's picture."""
    from tatbot_travel import overhead

    over = load_overhead(scan_path)
    if over is None:
        return None
    over_arm, over_plane = overhead.forearm_cloud(over, overhead.CAMERA_UP)
    tf, rms, paired = overhead.align(skin.points, plane, over_arm, over_plane)
    over.cam_r, over.cam_p = tf[:3, :3].T, -tf[:3, :3].T @ tf[:3, 3]  # its pose in the base frame
    preview(path, over, strokes, chosen)
    return {"overhead_rms_mm": round(rms * 1000, 2), "overhead_paired": round(paired, 3),
            "overhead_preview": str(path)}


def load_scan(path: Path, kin=None) -> tuple[list[surface.Frame], np.ndarray]:
    data = np.load(path)
    values = json.loads(str(data["intrinsics"]))
    values["distortion"] = tuple(values["distortion"])
    intr = Intrinsics(**values)
    kin = kin or pen_path.arm_kinematics(intr)
    frames = [surface.Frame(rgb=rgb, depth_m=depth, intr=intr, cam_p=p, cam_r=r)
              for rgb, depth, (p, r) in zip(data["rgb"], data["depth"], (kin.camera_pose(q) for q in data["q"]),
                                            strict=True)]
    return frames, data["q"]


# --- plan -------------------------------------------------------------------------------------------

def strokes_from_scan(frames: list[surface.Frame], kin=None, robot: np.ndarray | None = None
                      ) -> tuple[surface.Surface, list[inkmap.Stroke], tuple, int]:
    """The forearm's surface from every view, and the strokes from the view that sees the most ink.

    The table plane comes from everything seen; the skin only from forearm pixels clear of the arm's own
    cradle and pen, which stand above the table too and would otherwise join the 'skin'."""
    from tatbot_travel.selfview import robot_mask

    if robot is None:
        robot = robot_mask(kin or pen_path.arm_kinematics(frames[0].intr), frames[0].intr)
    points, _ = surface.fuse(frames)
    plane = surface.table_plane(points)
    masks = [inkmap.forearm_pixels(f, plane, exclude=robot) for f in frames]
    on_arm = np.concatenate([f.points(m)[0] for f, m in zip(frames, masks, strict=True)])
    # The masks are hole-filled, which takes in table along the contact line; the skin stands above it.
    (skin_points,) = surface.voxel(on_arm[surface.above(on_arm, plane)], 0.001)
    skin = surface.Surface(skin_points)
    best, best_view = [], 0
    for i, (frame, mask) in enumerate(zip(frames, masks, strict=True)):
        found = inkmap.strokes(frame, skin, mask)[0]
        if sum(st.length_m for st in found) > sum(st.length_m for st in best):
            best, best_view = found, i
    return skin, best, plane, best_view


def preview(path: Path, frame: surface.Frame, strokes: list[inkmap.Stroke], chosen: int | None) -> None:
    import cv2

    image = cv2.cvtColor(np.ascontiguousarray(frame.rgb), cv2.COLOR_RGB2BGR)
    for i, stroke in enumerate(strokes):
        uv, _ = frame.project(stroke.points)
        colour = (0, 0, 255) if i == chosen else (255, 160, 0)
        cv2.polylines(image, [np.round(uv).astype(np.int32)], False, colour, 2 if i == chosen else 1)
        cv2.putText(image, str(i), tuple(np.round(uv[0]).astype(int)), cv2.FONT_HERSHEY_SIMPLEX, 0.4, colour, 1)
    cv2.imwrite(str(path), image)


# --- score ------------------------------------------------------------------------------------------

def score(kin, skin: surface.Surface, strokes: list[inkmap.Stroke], q_measured: np.ndarray) -> dict:
    """Cross-track distance of the measured working point to the nearest scanned ink, its height off the
    skin, and how much of the ink came within 2 mm of it."""
    tcp = np.array([kin.gap_pose(q)[0] for q in q_measured])
    a = np.concatenate([st.points[:-1] for st in strokes])
    ab = np.concatenate([np.diff(st.points, axis=0) for st in strokes])
    t = np.clip(np.einsum("nkj,kj->nk", tcp[:, None] - a[None], ab) / np.einsum("kj,kj->k", ab, ab), 0, 1)
    dist = np.linalg.norm(tcp[:, None] - (a[None] + t[..., None] * ab[None]), axis=2)
    nearest = dist.min(axis=1)
    on_skin, normals, _ = skin.project_path(tcp)
    height = ((tcp - on_skin) * normals).sum(axis=1)
    covered = np.unique(dist.argmin(axis=1)[nearest < 0.002])
    seg_len = np.linalg.norm(ab, axis=1)
    return {"cross_track_mm_mean": float(nearest.mean() * 1000),
            "cross_track_mm_p95": float(np.percentile(nearest, 95) * 1000),
            "hover_error_mm_mean": float(np.abs(height).mean() * 1000),
            "ink_covered_mm": float(seg_len[covered].sum() * 1000), "ink_mm": float(seg_len.sum() * 1000)}


# --- commands ---------------------------------------------------------------------------------------

def cmd_scan(args) -> int:
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    # The overhead first, while the arm is folded at rest and hides none of the forearm.
    over = None
    try:
        from tatbot_travel import overhead

        view = overhead.capture(assets.repo_root())
        over = {"bgr": view.rgb[..., ::-1], "depth": view.depth_m, "intrinsics": json.dumps(
            {"width": view.intr.width, "height": view.intr.height, "fx": view.intr.fx, "fy": view.intr.fy,
             "ppx": view.intr.cx, "ppy": view.intr.cy, "distortion_coefficients": list(view.intr.distortion),
             "distortion_model": "BrownConrady" if view.intr.model == "brown_conrady" else "InverseBrownConrady"})}
    except Exception as error:  # the overhead is the audience's view; the wrist scan stands without it
        print(f"scan: no overhead view ({error})", file=sys.stderr)
    with Rig(out, estop_required=not args.no_estop) as rig:
        seed = np.array(args.seed or SEED_Q, float)
        if why := rig.go(seed):
            print(f"scan: stopped moving to the seed pose: {why}", file=sys.stderr)
            return 1
        frame, q, seed = survey(rig, seed)
        frames, qs = [frame], [q]
        for view in scan_views(rig.kin, frame, seed, args.views):
            if why := rig.go(view):
                print(f"scan: stopped between views: {why}", file=sys.stderr)
                break
            frame, q = rig.capture()
            frames.append(frame)
            qs.append(q)
        rig.go(seed)
    save_scan(out / "scan.npz", frames, qs, over)
    print(json.dumps({"scan": str(out / "scan.npz"), "views": len(frames), "overhead": over is not None}))
    return 0


def plan_from(args, kin=None):
    frames, qs = load_scan(Path(args.scan), kin)
    kin = kin or pen_path.arm_kinematics(frames[0].intr)
    skin, found, _, view = strokes_from_scan(frames, kin)
    if not found:
        raise RuntimeError("no ink strokes found on the forearm")
    stroke = found[args.stroke]
    if args.reverse:
        stroke = inkmap.Stroke(stroke.points[::-1], stroke.normals[::-1], stroke.pixels[::-1])
    seed = np.array(args.seed or SEED_Q, float)
    plan = pen_path.plan_trace(kin, seed, seed, stroke.points, stroke.normals, speed_m_s=args.speed)
    return kin, frames, skin, found, view, stroke, plan


def cmd_plan(args) -> int:
    kin, frames, skin, found, view, stroke, plan = plan_from(args)
    out = Path(args.scan).parent
    preview(out / "strokes.png", frames[view], found, args.stroke)
    points, _ = surface.fuse(frames)
    over = overhead_preview(out / "overhead.png", Path(args.scan), skin, surface.table_plane(points), found,
                            args.stroke)
    print(json.dumps({"strokes_mm": [round(s.length_m * 1000, 1) for s in found], "view": view,
                      "plan_s": round(plan.duration_s, 2), "peak_joint_speed": round(plan.peak_joint_speed, 3),
                      "ik_mm": round(plan.position_error_m * 1000, 2), "preview": str(out / "strokes.png"),
                      **(over or {})}))
    return 0


def cmd_run(args) -> int:
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    record: list[dict] = []
    with Rig(out, estop_required=not args.no_estop) as rig:
        kin, frames, skin, found, view, stroke, plan = plan_from(args, rig.kin)
        preview(out / "strokes.png", frames[view], found, args.stroke)
        seed = np.array(args.seed or SEED_Q, float)
        why = rig.go(seed) or rig.play(plan.q, guard_clearance=True, record=record)
        if why:
            print(f"trace: stopped: {why}", file=sys.stderr)
            here = rig.arm.measured()
            gap, axis = rig.kin.gap_pose(here)
            rise = rig.kin.solve(here, gap - axis * pen_path.CLEARANCE_M, axis, here, iters=100)
            rig.play(pen_path.plan_move(here, rise.q, max_speed=0.15))
        rig.go(seed)
    phase = plan.phase[[r["i"] for r in record]] if record else np.array([])
    q_measured = np.array([r["q"] for r in record])[phase == 1] if record else np.empty((0, 6))
    result = {"stopped": why, "ticks": len(record),
              "score": score(kin, skin, [stroke], q_measured) if len(q_measured) else None}
    (out / "record.json").write_text(json.dumps({"stroke": args.stroke, "record": record, **result}))
    print(json.dumps(result))
    return 0 if why is None else 1


def cmd_astra(args) -> int:
    """Astra picks every stretch of ink from the wrist image; the executor only lifts and plays it."""
    from concurrent.futures import ThreadPoolExecutor

    from tatbot_travel import astra

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    decider = astra.Astra()
    frames, _ = load_scan(Path(args.scan))
    skin, scanned, _, _ = strokes_from_scan(frames)
    decisions, record, traced = [], [], []
    seed = np.array(args.seed or SEED_Q, float)
    pool = ThreadPoolExecutor(max_workers=1)

    def ask(frame: surface.Frame, anchor: np.ndarray, end: bool):
        uv_anchor = frame.project(anchor[None])[0][0]
        uv_traced = frame.project(np.array(traced))[0] if len(traced) > 1 else None
        image = astra.annotate(frame.rgb, uv_anchor, uv_traced, end=end)
        n = len(decisions)
        import cv2

        cv2.imwrite(str(out / f"astra-{n:03d}-in.jpg"), cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
        return frame, decider.decide(image)

    why = None
    with Rig(out, estop_required=not args.no_estop) as rig:
        kin = rig.kin
        started = time.monotonic()
        why = rig.go(seed)
        frame, q = rig.capture()
        future = pool.submit(ask, frame, kin.gap_pose(q)[0], False)
        plan = None
        while why is None:
            waited = time.monotonic()
            frame, decision = future.result()
            waited = time.monotonic() - waited
            decisions.append({"reason": decision.reason, "points_px": decision.points_px.tolist(),
                              "done": decision.done, "latency_s": round(decision.latency_s, 2),
                              "arm_waited_s": round(waited, 2), "usage": decision.usage})
            print(f"astra {len(decisions)}: {decision.reason} ({decision.latency_s:.1f} s)", file=sys.stderr)
            if decision.done or time.monotonic() - started > args.seconds or len(decisions) >= args.max_calls:
                break
            points, normals = astra.lift(frame, skin, decision.points_px)
            if len(points) < 3:
                print("astra: the points do not land on the forearm; asking again", file=sys.stderr)
                here = rig.arm.measured()
                frame, _ = rig.capture(settle_s=0.1)
                future = pool.submit(ask, frame, kin.gap_pose(here)[0], False)
                continue
            q_now = rig.arm.measured()
            try:
                plan = (pen_path.plan_trace(kin, q_now, seed, points, normals, speed_m_s=args.speed) if plan is None
                        else pen_path.plan_follow(kin, q_now, points, normals, speed_m_s=args.speed))
            except pen_path.PlanError as error:
                print(f"astra: cannot follow that stretch ({error}); asking again", file=sys.stderr)
                frame, _ = rig.capture(settle_s=0.1)
                future = pool.submit(ask, frame, kin.gap_pose(q_now)[0], False)
                continue
            stream = plan.q[plan.phase < 2]  # stay hovering at the end of the stretch
            end = plan.targets[plan.phase < 2][-1]
            frame, _ = rig.capture(settle_s=0.0)  # the view the next decision is made on
            future = pool.submit(ask, frame, end, True)
            why = rig.play(stream, guard_clearance=True, record=record)
            traced.extend(plan.targets[plan.phase == 1].tolist())
        if why:
            print(f"astra: stopped: {why}", file=sys.stderr)
        here = rig.arm.measured()
        gap, axis = rig.kin.gap_pose(here)
        rise = rig.kin.solve(here, gap - axis * pen_path.CLEARANCE_M, axis, here, iters=100)
        rig.play(pen_path.plan_move(here, rise.q, max_speed=0.15))
        rig.go(seed)
    pool.shutdown(wait=False, cancel_futures=True)
    q_measured = np.array([r["q"] for r in record]) if record else np.empty((0, 6))
    result = {"stopped": why, "decisions": len(decisions), "ticks": len(record),
              "moving_fraction": round(len(record) / pen_path.RATE_HZ / max(1e-9, time.monotonic() - started), 3),
              "score": score(kin, skin, scanned, q_measured) if len(q_measured) and scanned else None}
    (out / "astra.json").write_text(json.dumps({"decisions": decisions, "record": record, **result}))
    print(json.dumps(result))
    return 0 if why is None else 1


def perturbed(stroke: inkmap.Stroke, rng: np.random.Generator, max_offset_m: float = 0.006,
              decay_m: float = 0.015) -> np.ndarray:
    """The stroke's points with the start pushed sideways (along the skin) by up to ``max_offset_m``,
    the push fading over ``decay_m``: the demonstration starts off the line and settles onto it."""
    pts = stroke.points
    seg = np.linalg.norm(np.diff(pts, axis=0), axis=1)
    s = np.concatenate([[0.0], np.cumsum(seg)])
    tangent = np.gradient(pts, axis=0)
    side = np.cross(stroke.normals, tangent)
    side /= np.linalg.norm(side, axis=1, keepdims=True)
    push = rng.uniform(-max_offset_m, max_offset_m) * np.clip(1.0 - s / decay_m, 0.0, 1.0)
    return pts + push[:, None] * side


def cmd_collect(args) -> int:
    """Classical-tracer demonstrations into a LeRobot dataset (the FLUX.3 Action contract of writer.py)."""
    from types import SimpleNamespace

    from tatbot_travel.writer import DatasetSink

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed_rng)
    frames, _ = load_scan(Path(args.scan))
    skin, found, _, _ = strokes_from_scan(frames)
    found = [st for st in found if st.length_m >= args.min_stroke_mm / 1000]
    if not found:
        raise RuntimeError("no stroke long enough to demonstrate")
    sink = DatasetSink(Path(args.dataset), repo_id=args.repo_id, task=args.task)
    seed = np.array(args.seed or SEED_Q, float)
    episodes, why = [], None
    with Rig(out, estop_required=not args.no_estop) as rig:
        for n in range(args.episodes):
            stroke = found[n % len(found)]
            if rng.random() < 0.5:
                stroke = inkmap.Stroke(stroke.points[::-1], stroke.normals[::-1], stroke.pixels[::-1])
            speed = rng.uniform(0.010, 0.020)
            if why := rig.go(seed):
                break
            try:
                plan = pen_path.plan_trace(rig.kin, rig.arm.measured(), seed, perturbed(stroke, rng),
                                           stroke.normals, speed_m_s=speed)
            except pen_path.PlanError as error:
                print(f"collect {n}: skipped ({error})", file=sys.stderr)
                continue
            ticks = []

            def keep(q_cmd, q_meas, frame, ticks=ticks):
                if frame is not None:
                    ticks.append((np.asarray(frame[0]).copy(), q_meas.astype(np.float32), q_cmd.astype(np.float32)))

            why = rig.play(plan.q, guard_clearance=True, on_tick=keep)
            if why:
                print(f"collect {n}: stopped ({why}); episode dropped", file=sys.stderr)
                break
            for image, state, action in ticks:
                sink.add(SimpleNamespace(image=image, state=state, action=action, labels={}))
            index = sink.end_episode({"stroke_mm": stroke.length_m * 1000, "speed_m_s": speed, "ticks": len(ticks)})
            episodes.append(index)
            print(f"collect {n}: episode {index}, {len(ticks)} frames", file=sys.stderr)
        rig.go(seed)
    sink.finalize()
    print(json.dumps({"stopped": why, "episodes": episodes, "dataset": args.dataset}))
    return 0 if why is None else 1


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="trace", description=__doc__.split("\n")[0])
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("scan", "plan", "run", "astra", "collect"):
        p = sub.add_parser(name)
        p.add_argument("--seed", type=float, nargs=6, help="survey pose, rad (default: the 2026-10-05 layout's)")
        if name != "plan":
            p.add_argument("--out", default=os.environ.get("TATBOT_RUN_DIR"), required="TATBOT_RUN_DIR" not in os.environ,
                           help="run directory (default: the tatbot run log's)")
            p.add_argument("--no-estop", action="store_true", help="supervised: the operator holds another stop")
        if name == "scan":
            p.add_argument("--views", type=int, default=3)
        else:
            p.add_argument("--scan", required=True, help="scan.npz from `trace scan`")
            p.add_argument("--stroke", type=int, default=0, help="stroke index, longest first")
            p.add_argument("--reverse", action="store_true", help="trace the stroke from its other end")
            p.add_argument("--speed", type=float, default=pen_path.SPEED_M_S, help="m/s along the ink")
        if name == "collect":
            p.add_argument("--episodes", type=int, default=20)
            p.add_argument("--dataset", required=True, help="new LeRobot dataset root (must not exist)")
            p.add_argument("--repo-id", default="hu-po/tatbot-laser-trace")
            p.add_argument("--task", default="trace the ink on the forearm with the laser pen")
            p.add_argument("--min-stroke-mm", type=float, default=20.0)
            p.add_argument("--seed-rng", type=int, default=0)
        if name == "astra":
            p.add_argument("--seconds", type=float, default=180.0, help="stop asking after this long")
            p.add_argument("--max-calls", type=int, default=30, help="stop after this many decisions")
    args = parser.parse_args(argv)
    return {"scan": cmd_scan, "plan": cmd_plan, "run": cmd_run, "astra": cmd_astra,
            "collect": cmd_collect}[args.command](args)


if __name__ == "__main__":
    sys.exit(main())
