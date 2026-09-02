#!/usr/bin/env python3
"""Preview what the data factory would generate — same code path, no dataset.

Runs ONE batch through the exact planner and expert the generator uses
(tatbot_sim.planning.plan_batch), renders it from the wrist cameras plus a
third-person and a top-down view, and writes stills and animated webp clips.
This replaces the throwaway /tmp replay scripts that hand-mirrored
generate.py's RNG order and broke whenever it changed.

Run on an x86_64 sim host in the tatbot_sim venv:
    .venv/bin/python scripts/sim_preview.py --task language --seed 7 \
        --num-envs 4 --out /tmp/preview
    # DR overrides work exactly like generate:
    #   --dr.pad.tilt-range 0.15 --dr.rgb.enabled False
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import gymnasium as gym
import numpy as np
import tatbot_sim  # noqa: F401
import torch
import tyro
from mani_skill.sensors.camera import CameraConfig
from mani_skill.utils import sapien_utils
from tatbot_sim import interaction, tools
from tatbot_sim.config import DRConfig
from tatbot_sim.depth_noise import DepthCorruptor, RGBJitter
from tatbot_sim.env import TatbotDrawEnv
from tatbot_sim.expert import (
    StrokeExpert,
    reachable_canvas_masks,
    reachable_height_ceiling,
)
from tatbot_sim.planning import plan_batch

CAMERAS = ("wrist_upper", "wrist_lower")
EXTRA_VIEWS = {
    "thirdperson": ((0.62, -0.42, 0.42), (0.29, 0, 0.05)),
    "topdown": ((0.29, 0.0, 0.72), (0.29, 0.0, 0.0)),
}


@dataclass
class Args:
    out: str = "/tmp/sim-preview"
    task: str = "language"
    seed: int = 0
    num_envs: int = 4
    horizon: int = 900
    clip_stride: int = 4
    """Capture every Nth frame for the webp clips (30/N fps)."""
    depth: bool = True
    dr: DRConfig = field(default_factory=DRConfig)
    task_name: str = field(default_factory=lambda: (
        "draw a {size_mm}mm {shape} "
        f"{tools.active_tool().prompt_phrase} on the paper pad"
    ))
    """Mirrors generate's default — see generate.Args.task_name."""
    maze_task_name: str = field(default_factory=lambda: (
        f"draw a continuous squiggle {tools.active_tool().prompt_phrase} "
        "on the grid lines of the paper pad."
    ))
    """Mirrors generate's default: the tool slot comes from the fitted tool's
    datasheet, so a tool swap moves the preview's prompt with it. The literal
    that used to sit here said "using pen tip" whatever was in the gripper."""


def main(args: Args):
    out = Path(args.out).expanduser()
    out.mkdir(parents=True, exist_ok=True)

    def with_views(self):
        return [
            CameraConfig(uid=n, pose=sapien_utils.look_at(eye=list(e), target=list(t),
                                                          up=[1, 0, 0] if n == "topdown" else [0, 0, 1]),
                         width=640, height=480, fov=0.85 if n == "thirdperson" else 0.55,
                         near=0.01, far=100)
            for n, (e, t) in EXTRA_VIEWS.items()
        ]
    TatbotDrawEnv._default_sensor_configs = property(with_views)

    rng = np.random.default_rng(args.seed)
    env = gym.make(
        "TatbotDraw-v0", num_envs=args.num_envs,
        obs_mode="rgbd" if args.depth else "rgb", control_mode="pd_joint_pos",
        sim_backend="auto", reconfiguration_freq=1, dr=args.dr,
    )
    base = env.unwrapped
    device = base.device
    robot = base.agent.robot
    env.reset(seed=args.seed)

    # The expert is built BEFORE planning because a shaped surface has to be
    # planned against what the arm can actually reach on it — the same masks
    # generate computes. Without them the preview places strokes on mound
    # flanks the wrist cannot hold the tool normal to, and renders a miss that
    # the real generator would never have planned: the preview would libel the
    # distribution it exists to show.
    active = [j.name for j in robot.active_joints]
    expert = StrokeExpert(args.num_envs, device, noise=args.dr.noise, seed=args.seed)
    idx_ik = [active.index(n) for n in expert.ik.chain.get_joint_parameter_names()]

    masks = ceiling = None
    q_now = robot.get_qpos()[:, idx_ik]
    slack = args.dr.pen_lean.max_off_base_rad
    masks = reachable_canvas_masks(expert, q_now, base.surface,
                                   interaction.WORKING_OFFSET_M,
                                   args.num_envs, max_off_base_rad=slack)
    ceiling = reachable_height_ceiling(expert, q_now, base.surface,
                                       args.num_envs, max_off_base_rad=slack)
    print(f"reachable: {np.mean([m.fraction for m in masks]):.0%} of the "
          f"{base.substrate.name}, tool ceiling {ceiling:.3f} m")

    plan = plan_batch(
        rng, base.pad_sheets, base.surface,
        task=args.task, horizon=args.horizon, num_envs=args.num_envs,
        dr=args.dr, draw_clearance=interaction.WORKING_OFFSET_M,
        task_name=args.task_name, maze_task_name=args.maze_task_name,
        reachable=masks, tool_ceiling=ceiling,
    )

    if plan.preink is not None:
        # erase episodes OPEN with the scene already inked; generate does
        # this and the preview must too, or the laser erases a blank sheet
        base.preink(plan.preink)
    q_start = expert.solve_pose(plan.targets[:, 0], robot.get_qpos()[:, idx_ik],
                                normals=plan.pen_normals[:, 0])
    full = robot.get_qpos().clone()
    full[:, idx_ik] = q_start
    robot.set_qpos(full)
    if plan.q_raised is not None:
        full = robot.get_qpos().clone()
        full[:, idx_ik] = torch.as_tensor(plan.q_raised, device=device)
        robot.set_qpos(full)
    expert.reset(plan.targets, q_start,
                 floor_plane=(plan.surface_points, plan.surface_normals),
                 pen_normals=plan.pen_normals,
                 approach_from=(plan.q_raised, plan.n_app) if plan.q_raised is not None else None)
    corruptor = DepthCorruptor(args.num_envs, device, cfg=args.dr.depth_noise, seed=args.seed) \
        if args.depth and args.dr.corrupt_depth else None
    jitter = RGBJitter(args.num_envs, device, cfg=args.dr.rgb, seed=args.seed)

    clips: dict[tuple[str, int], list] = {}
    for t in range(plan.episode_steps):
        obs, *_ = env.step(expert.act())
        if t % args.clip_stride:
            continue
        for view in list(EXTRA_VIEWS) + list(CAMERAS):
            rgb = obs["sensor_data"][view]["rgb"]
            if view in CAMERAS:
                rgb = jitter(rgb)
            f = rgb.cpu().numpy()
            for i in range(args.num_envs):
                clips.setdefault((view, i), []).append(f[i])
        if args.depth and corruptor is not None:
            d = corruptor(obs["sensor_data"]["wrist_upper"]["depth"]).cpu().numpy()
            for i in range(args.num_envs):
                mm = np.squeeze(d[i]).astype(np.float32)
                valid = mm > 0
                norm = np.clip((mm - 80.0) / 270.0, 0, 1)
                img = np.stack([norm * 255] * 3, -1).astype(np.uint8)
                img[~valid] = 16
                clips.setdefault(("depth", i), []).append(img)

    try:
        from PIL import Image
        for (view, i), frames in clips.items():
            ims = [Image.fromarray(f) for f in frames]
            ims[0].save(out / f"{view}_{i:02d}.webp", save_all=True,
                        append_images=ims[1:], duration=int(1000 * args.clip_stride / 30),
                        loop=0, quality=50, method=4)
            ims[-1].save(out / f"{view}_{i:02d}_last.png")
    except ImportError:
        import cv2
        for (view, i), frames in clips.items():
            cv2.imwrite(str(out / f"{view}_{i:02d}_last.png"), frames[-1][:, :, ::-1])
        print("PIL missing: wrote stills only")

    for i in range(args.num_envs):
        print(f"env {i}: {plan.tasks[i]}")
    print(f"wrote {len(clips)} clips to {out} ({plan.episode_steps} steps, "
          f"n_app={plan.n_app})")
    env.close()


if __name__ == "__main__":
    main(tyro.cli(Args))
