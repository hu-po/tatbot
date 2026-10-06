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

import numpy as np
import tatbot_sim  # noqa: F401
import tyro
from mani_skill.sensors.camera import CameraConfig
from mani_skill.utils import sapien_utils
from tatbot_sim import design_scene, interaction
from tatbot_sim.backends.maniskill import ManiSkillWorld
from tatbot_sim.config import DRConfig
from tatbot_sim.env import TatbotDrawEnv
from tatbot_sim.episode import Episode
from tatbot_sim.expert import (
    StrokeExpert,
    reachable_canvas_masks,
    reachable_height_ceiling,
)
from tatbot_sim.observations import ObservationBuilder
from tatbot_sim.planning import plan_batch

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
    sensor_profile: str = "deployment"
    dr: DRConfig = field(default_factory=DRConfig)
    design: str | None = None
    """A portable design to draw in place of the shared collection (--task
    artwork); the same flag as generate's, see tatbot_sim.design_scene."""
    design_placement: str = "authored"
    """authored keeps Inkmap's placement; sampled recentres and offsets it."""
    tool_id: str | None = None
    task_name: str = "draw a {size_mm}mm {shape} {tool} on the paper pad"
    maze_task_name: str = "draw a continuous squiggle {tool} on the grid lines of the paper pad."


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

    from tatbot_sim.resolved import resolve
    config = resolve(tool_id=args.tool_id, seed=args.seed, dr=args.dr,
                     sensor_profile=args.sensor_profile)
    rng = np.random.default_rng(args.seed)
    env = TatbotDrawEnv(config=config, presentation_cameras=with_views, num_envs=args.num_envs,
        obs_mode="rgbd" if args.depth else "rgb", control_mode="pd_joint_pos",
        sim_backend="auto", reconfiguration_freq=1, dr=args.dr,
        sensor_profile=args.sensor_profile,
    )
    base = env.unwrapped
    cameras = tuple(camera.role for camera in base.agent.camera_descriptions)
    device = base.device
    expert = StrokeExpert(args.num_envs, device, config=config, noise=args.dr.noise, seed=config.seed_for("noise"))
    world = ManiSkillWorld(env, config, expert)
    episode = Episode(world, ObservationBuilder(config, args.num_envs, device))
    episode.reset(seed=args.seed)

    masks = ceiling = None
    q_now = world.positions()
    slack = args.dr.pen_lean.max_off_base_rad
    masks = reachable_canvas_masks(expert, q_now, base.surface,
                                   interaction.WORKING_OFFSET_M,
                                   args.num_envs, max_off_base_rad=slack)
    ceiling = reachable_height_ceiling(expert, q_now, base.surface,
                                       args.num_envs, max_off_base_rad=slack)
    print(f"reachable: {np.mean([m.fraction for m in masks]):.0%} of the "
          f"{base.substrate.name}, tool ceiling {ceiling:.3f} m")

    design = design_scene.from_args(args, base.substrate, config=config)
    plan = plan_batch(
        rng, base.pad_sheets, base.surface, config=config,
        task=args.task, horizon=args.horizon, num_envs=args.num_envs,
        dr=args.dr, draw_clearance=interaction.WORKING_OFFSET_M,
        task_name=args.task_name, maze_task_name=args.maze_task_name,
        reachable=masks, tool_ceiling=ceiling, cap_rims=base.cap_rims_np(),
        artwork_sampler=design_scene.sampler(design),
    )
    episode.install(plan)

    clips: dict[tuple[str, int], list] = {}
    for t in range(episode.horizon):
        if episode.done:
            break
        _, observation, _ = episode.step(capture=t % args.clip_stride == 0)
        obs = episode.last_raw
        if t % args.clip_stride:
            continue
        for view in list(EXTRA_VIEWS) + list(cameras):
            rgb = observation.rgb[view] if view in cameras else obs["sensor_data"][view]["rgb"]
            f = rgb.cpu().numpy()
            for i in range(args.num_envs):
                clips.setdefault((view, i), []).append(f[i])
        if args.depth:
            d = observation.depth_mm[cameras[0]].cpu().numpy()
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
    print(f"wrote {len(clips)} clips to {out} ({episode.step_index} steps, "
          f"n_app={plan.n_app})")
    import json
    (out / "runtime.json").write_text(json.dumps({"resolved_config": config.metadata(),
                                                "runtime": episode.metadata()}, indent=2) + "\n")
    episode.close()


if __name__ == "__main__":
    main(tyro.cli(Args))
