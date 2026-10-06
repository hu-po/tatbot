"""Dry runs on the real camera: what the travel policy would do, with nothing moving.

The first look at the real setup should not start with the arm moving. This
reads the blue arm's wrist D405 live from visiond (colour and depth), holds the
arm state at a given pose -- the pose the arm is standing in -- and runs the
policy the way the demo will: plans computed back to back in a background
thread from the last eight frames, at the node's real latency. With the arm
still, each plan is the policy's intent from that pose.

It records, per plan, the latency and the joint commands, and renders every
frame with the pen path the newest plan would take (forward kinematics of its
joints, projected through the camera's calibration) and the depth clearance
along the pen's axis: the distance from the lens face to the first surface
the pen would meet. That clearance is also the stop a moving run should
trust over the policy.

    travel shadow --checkpoint RUN/checkpoints/010000/pretrained_model_ema \\
        --dataset DATA --pose park --seconds 30 --out ~/tatbot-logs/travel_shadow
"""

from __future__ import annotations

import importlib.util
import json
import threading
import time
from pathlib import Path

import numpy as np

from tatbot_travel.camera import Intrinsics
from tatbot_travel.chunking import History

SOCKET = Path("/tmp/tatbot-d405-frames.sock")


def visiond_wire():
    """The repo's visiond reader (scripts/vision/visiond_wire.py): one decoder for every consumer."""
    from tatbot_travel import assets

    path = assets.repo_root() / "scripts" / "vision" / "visiond_wire.py"
    spec = importlib.util.spec_from_file_location("visiond_wire", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def axis_clearance(depth_m: np.ndarray, lens_cam: np.ndarray, axis_cam: np.ndarray, intr: Intrinsics, *,
                   start_m: float = 0.003, reach_m: float = 0.10, step_m: float = 0.001, run: int = 3) -> float:
    """Distance along the pen axis from the lens face to the first surface in the depth image (inf: none).

    A surface is where the axis passes behind what the camera sees and stays behind it for ``run`` samples:
    past a real surface every further point is hidden too, while one speckled depth pixel hides one sample.
    """
    r = np.arange(start_m, reach_m, step_m)
    points = lens_cam + r[:, None] * axis_cam
    uv = np.round(intr.project(points)).astype(int)
    h, w = depth_m.shape
    inside = (uv[:, 0] >= 0) & (uv[:, 0] < w) & (uv[:, 1] >= 0) & (uv[:, 1] < h) & (points[:, 2] > 0)
    surface = np.zeros(len(r))
    surface[inside] = depth_m[uv[inside, 1], uv[inside, 0]]
    hidden = inside & (surface > 0) & (surface <= points[:, 2])
    runs = np.convolve(hidden.astype(int), np.ones(run, dtype=int), mode="valid") == run
    hit = np.flatnonzero(runs)
    return float(r[hit[0]]) if len(hit) else float("inf")


class Planner:
    """Plans computed one at a time in a background thread; the newest finished one is kept."""

    def __init__(self, policy):
        self.policy = policy
        self.busy = False
        self.done: list[tuple[int, np.ndarray, float]] = []
        self.lock = threading.Lock()
        self.thread: threading.Thread | None = None

    def request(self, tick: int, window: tuple[np.ndarray, np.ndarray, np.ndarray]) -> bool:
        if self.busy:
            return False
        self.busy = True
        self.thread = threading.Thread(target=self._run, args=(tick, window), daemon=True)
        self.thread.start()
        return True

    def join(self) -> None:
        """Let a plan in flight finish: a thread killed inside CUDA aborts the process."""
        if self.thread is not None:
            self.thread.join()

    def _run(self, tick: int, window) -> None:
        started = time.monotonic()
        try:
            commands = self.policy.plan(*window)
        finally:
            with self.lock:
                self.busy = False
        with self.lock:
            self.done.append((tick, commands, time.monotonic() - started))

    def take(self) -> list[tuple[int, np.ndarray, float]]:
        with self.lock:
            done, self.done = self.done, []
        return done


class Overlay:
    """The arm model at the pinned pose: camera pose, pen axis, and planned pen paths in the image."""

    def __init__(self, q: np.ndarray):
        from tatbot_travel.camera import render_plan
        from tatbot_travel.kinematics import ArmKinematics
        from tatbot_travel.scene import SceneBuilder

        self.intr = Intrinsics.left_wrist()
        self.kin = ArmKinematics(SceneBuilder(plan=render_plan(self.intr)).compile())
        self.cam_p, self.cam_r = self.kin.camera_pose(q)
        gap, axis = self.kin.gap_pose(q)
        lens = gap - axis * 0.020  # the gap point sits 20 mm beyond the lens face
        self.lens_cam = (lens - self.cam_p) @ self.cam_r
        self.axis_cam = axis @ self.cam_r

    def gaps(self, commands: np.ndarray) -> np.ndarray:
        """World positions of the gap point over a plan's commands (forward kinematics: once per plan)."""
        return np.array([self.kin.gap_pose(q)[0] for q in commands])

    def project(self, gaps: np.ndarray) -> np.ndarray:
        """Those points in pixels, seen from the camera's current pose."""
        return self.intr.project((gaps - self.cam_p) @ self.cam_r)

    def path(self, commands: np.ndarray) -> np.ndarray:
        """Pixels of the gap point over a plan's commands, seen from the camera's current pose."""
        return self.project(self.gaps(commands))

    def draw(self, bgr: np.ndarray, path: np.ndarray | None, clearance: float, label: str) -> np.ndarray:
        import cv2

        out = bgr.copy()
        if path is not None and len(path) > 1:
            cv2.polylines(out, [np.round(path).astype(np.int32)], False, (0, 255, 0), 2, cv2.LINE_AA)
            cv2.circle(out, tuple(int(v) for v in np.round(path[0])), 5, (0, 0, 255), 2, cv2.LINE_AA)
        nose = self.intr.project(self.lens_cam[None])[0]
        cv2.circle(out, tuple(int(v) for v in np.round(nose)), 4, (255, 128, 0), -1, cv2.LINE_AA)
        text = f"{label}  clearance {clearance * 1000:.0f} mm" if np.isfinite(clearance) else f"{label}  clear"
        cv2.putText(out, text, (8, 470), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1, cv2.LINE_AA)
        return out


def run(policy, q: np.ndarray, seconds: float, out: Path, camera: str = "realsense1",
        socket_path: Path = SOCKET) -> dict:
    """Shadow the policy on the live camera for ``seconds``; writes video, plans and a summary under ``out``."""
    import cv2

    wire = visiond_wire()
    overlay = Overlay(q)
    history, planner = History(policy.n_obs_steps), Planner(policy)
    out.mkdir(parents=True, exist_ok=True)
    video = cv2.VideoWriter(str(out / "shadow.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), 30, (640, 480))
    plans_log = (out / "plans.jsonl").open("w")
    latest, latencies, clearances, tick = None, [], [], 0
    for frame_set in wire.latest_socket_sets(socket_path, duration_s=60.0):  # warm up on the first frame
        colour = frame_set["frames"].get(f"{camera}_color")
        if colour is not None and "image" in colour:
            warm = History(policy.n_obs_steps)
            warm.push(colour["image"][..., ::-1].copy(), q, q)
            started = time.monotonic()
            policy.plan(*warm.window())
            print(json.dumps({"warmup_s": round(time.monotonic() - started, 2)}), flush=True)
            break
    for frame_set in wire.latest_socket_sets(socket_path, duration_s=seconds):
        colour, depth = frame_set["frames"].get(f"{camera}_color"), frame_set["frames"].get(f"{camera}_depth")
        if colour is None or "image" not in colour:
            continue
        bgr = colour["image"]
        history.push(bgr[..., ::-1].copy(), q, q)  # the arm is pinned: state and command stay put
        clearance = float("inf")
        if depth is not None and "depth" in depth:
            clearance = axis_clearance(depth["depth"] * (depth.get("depth_units_m") or 1e-4),
                                       overlay.lens_cam, overlay.axis_cam, overlay.intr)
        clearances.append(clearance)
        planner.request(tick, history.window())
        for start, commands, latency in planner.take():
            latest = overlay.path(commands[: min(len(commands), 32)])
            latencies.append(latency)
            plans_log.write(json.dumps({"tick": start, "arrived": tick, "latency_s": round(latency, 3),
                                        "commands": np.round(commands, 4).tolist()}) + "\n")
        video.write(overlay.draw(bgr, latest, clearance, f"t={tick / 30:5.1f}s plans={len(latencies)}"))
        tick += 1
    planner.join()
    video.release()
    plans_log.close()
    finite = [c for c in clearances if np.isfinite(c)]
    summary = {"frames": tick, "plans": len(latencies),
               "latency_median_s": float(np.median(latencies)) if latencies else None,
               "latency_max_s": float(np.max(latencies)) if latencies else None,
               "clearance_min_mm": float(np.min(finite) * 1000) if finite else None,
               "pose": np.round(q, 4).tolist()}
    (out / "summary.json").write_text(json.dumps(summary, indent=1) + "\n")
    return summary
