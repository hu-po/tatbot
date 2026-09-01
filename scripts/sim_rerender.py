#!/usr/bin/env python3
"""Re-render a recorded sim dataset under fresh visual draws.

One motion batch -> N visual variants: the episode GEOMETRY (pad poses,
trajectories, physics) replays deterministically from the source dataset's
reset seeds and recorded actions, while everything drawn at scene build —
lighting rigs, environment maps, floor/table textures, camera mount jitter,
RGB response, depth corruption — redraws. The output dataset keeps the
source's actions, states, prompts and per-episode metadata VERBATIM (they
are the canonical labels) and replaces only the videos.

Also the retroactive path: when hardware changes (new tool colours, new
cameras), existing datasets can be re-rendered instead of regenerated.

    .venv/bin/python scripts/sim_rerender.py \
        --source ~/tatbot-sim/datasets/language-1k --out /tmp/language-1k-v2
    # visual DR overrides work like generate:
    #   --dr.lighting.ambient 0.02 0.3 --dr.rgb.enabled False

Episode-geometry config (pad ranges, approach, latency pairing) is FORCED
from the source dataset — overriding it would desync video from state.
Every run reports replay drift (replayed qpos vs recorded states); large
drift means the replay diverged and the output must not be trained on.
"""

from __future__ import annotations

import dataclasses
import json
import time
from dataclasses import dataclass, field
from pathlib import Path

import gymnasium as gym
import numpy as np
import pandas as pd
import tatbot_sim  # noqa: F401
import torch
import tyro
from tatbot_sim.agent import GRIP_REST, TatbotWXAI
from tatbot_sim.config import DRConfig, LatencyDR, PadDR
from tatbot_sim.depth_noise import DepthCorruptor, RGBJitter
from tatbot_sim.lerobot_writer import LeRobotWriter, quantize_depth_codes

CAMERAS = ("wrist_upper", "wrist_lower")
# Replay re-applies action[0] from state[0] (which already contains its
# effect), so the whole replay sits a fraction of one step "ahead" of the
# recording. The offset is bounded by the per-step tracking error and does
# NOT accumulate — measured ~0.6 mrad on single-stroke squiggles and up to
# ~10 mrad on language episodes, whose 0.12 m/s inter-stroke travels have
# the largest per-step motion (~4 mm EE). Labels are copied verbatim either
# way; the gate exists to catch real, growing divergence.
DRIFT_WARN_RAD = 12e-3


@dataclass
class Args:
    source: str
    out: str
    dr: DRConfig = field(default_factory=DRConfig)
    """Visual axes only — pad/approach/latency are forced from the source."""


def main(args: Args):
    src = Path(args.source).expanduser()
    meta = json.loads((src / "meta" / "run_meta.json").read_text())
    cfg = meta.get("config") or meta.get("args")
    episodes = meta["episodes"]
    src_seed, num_envs = int(cfg["seed"]), int(cfg["num_envs"])
    with_depth = bool(cfg.get("depth", True))

    # geometry must reproduce the source exactly; visuals are yours to redraw
    dr = args.dr
    dr.pad = PadDR(**{k: tuple(v) if isinstance(v, list) else v
                      for k, v in cfg["dr"]["pad"].items()})
    lat_cfg = cfg["dr"].get("latency", {"obs_delay_steps": [0, 0]})
    dr.latency = LatencyDR(obs_delay_steps=tuple(lat_cfg["obs_delay_steps"]))

    # actions/states per episode, plus the per-episode task strings
    data = pd.concat(
        [pd.read_parquet(p) for p in sorted((src / "data").glob("chunk-*/file-*.parquet"))],
        ignore_index=True,
    )
    ep_meta = pd.read_parquet(src / "meta" / "episodes" / "chunk-000" / "file-000.parquet")
    by_ep = {
        int(e): (
            np.stack(g["action"].to_numpy()).astype(np.float32),
            np.stack(g["observation.state"].to_numpy()).astype(np.float32),
        )
        for e, g in data.groupby("episode_index")
    }
    tasks_by_ep = {int(r["episode_index"]): r["tasks"][0] for _, r in ep_meta.iterrows()}

    env = gym.make(
        "TatbotDraw-v0", num_envs=num_envs, obs_mode="rgbd" if with_depth else "rgb",
        control_mode="pd_joint_pos", sim_backend=cfg.get("sim_backend", "auto"),
        reconfiguration_freq=1, dr=dr,
    )
    base = env.unwrapped
    device = base.device
    robot = base.agent.robot
    active = [j.name for j in robot.active_joints]
    idx7 = [active.index(n) for n in TatbotWXAI.joint_names]

    writer = LeRobotWriter(args.out, cameras=CAMERAS, depth=with_depth,
                           task_name=cfg.get("task_name", ""))
    corruptor = (DepthCorruptor(num_envs, device, cfg=dr.depth_noise, seed=src_seed)
                 if with_depth and dr.corrupt_depth else None)
    rgb_jitter = RGBJitter(num_envs, device, cfg=dr.rgb, seed=src_seed)

    total, n_eps = 0, len(episodes)
    drift_max, t0 = 0.0, time.time()
    while total < n_eps:
        b = min(num_envs, n_eps - total)
        ep_ids = [total + i for i in range(b)]
        env.reset(seed=src_seed + total)  # reproduces this batch's pad geometry

        acts = [by_ep[e][0] for e in ep_ids]
        states = [by_ep[e][1] for e in ep_ids]
        # episodes are variable length (recording stops where each
        # drawing ends); replay to the longest and hold the others at
        # their final action — held steps are not written back
        lens = [len(a) for a in acts]
        batch_steps = max(lens)
        # surplus envs in a short last batch replay the last episode's motion
        while len(acts) < num_envs:
            acts.append(acts[-1])
        acts = [np.concatenate([a, np.repeat(a[-1:], batch_steps - len(a), axis=0)])
                if len(a) < batch_steps else a for a in acts]
        batch_actions = torch.as_tensor(np.stack(acts), device=device)

        # start where the recording started: first recorded pose (approach
        # episodes recorded from the raised pose). A gripper-era recording
        # carries the old carriage rest (0.0144, fingers on the machine body);
        # the derived URDF has one carriage now and it sits at rest.
        full = robot.get_qpos().clone()
        init = np.stack([s[0][:7] for s in states] + [states[-1][0][:7]] * (num_envs - b))
        full[:, idx7] = torch.as_tensor(init, device=device)
        for j_i, name in enumerate(active):
            if name.endswith("carriage_joint"):
                full[:, j_i] = GRIP_REST
        robot.set_qpos(full)

        if corruptor is not None:
            corruptor.reset()
        rgb_jitter.reset()
        writer.open_batch(b, tasks=[tasks_by_ep[e] for e in ep_ids])

        delays = [int(episodes[e].get("obs_delay_steps", 0)) for e in ep_ids]
        delays += [0] * (num_envs - b)
        max_delay = max(delays)
        hist: list[dict] = []

        for t in range(batch_steps):
            obs, *_ = env.step(batch_actions[:, t])
            qpos = obs["agent"]["qpos"][:, idx7].cpu().numpy()
            # the source state stream carries the recorded obs DELAY, so the
            # fresh replay pose at t corresponds to the stored state at t+k
            for i in range(b):
                if t >= lens[i]:
                    continue
                tk = min(t + delays[i], lens[i] - 1)
                drift_max = max(drift_max, float(np.abs(qpos[i][:6] - states[i][tk][:6]).max()))
            frames = {c: rgb_jitter(obs["sensor_data"][c]["rgb"]).cpu().numpy()
                      for c in CAMERAS}
            depth = None
            if with_depth:
                depth = {}
                for c in CAMERAS:
                    d = obs["sensor_data"][c]["depth"]
                    if corruptor is not None:
                        d = corruptor(d)
                    depth[c] = (quantize_depth_codes(d.to(torch.float32))
                                .to(torch.int16).cpu().numpy().astype(np.uint16))
            hist.append({"frames": frames, "depth": depth})
            if len(hist) > max_delay + 1:
                hist.pop(0)

            d_frames = {c: f.copy() for c, f in frames.items()}
            d_depth = None if depth is None else {c: v.copy() for c, v in depth.items()}
            for i in range(num_envs):
                k = delays[i]
                if k == 0:
                    continue
                s = hist[max(0, len(hist) - 1 - k)]
                for c in CAMERAS:
                    d_frames[c][i] = s["frames"][c][i]
                    if d_depth is not None:
                        d_depth[c][i] = s["depth"][c][i]
            # actions and STATES come from the source — they are the labels;
            # only the pixels are new
            src_state = np.stack([states[i][min(t, lens[i] - 1)] for i in range(b)])
            writer.add_steps(
                np.stack([acts[i][t] for i in range(b)]), src_state,
                {c: f[:b] for c, f in d_frames.items()},
                None if d_depth is None else {c: v[:b] for c, v in d_depth.items()},
                active=[t < lens[i] for i in range(b)],
            )
        writer.close_batch()
        total += b
        print(f"[rerender] {total}/{n_eps} episodes, drift max {1000*drift_max:.2f} mrad,"
              f" {total * batch_steps / (time.time() - t0):.0f} env-steps/s", flush=True)

    writer.finalize()
    out_meta = {
        "config": cfg,
        "rerendered_from": str(src),
        "rerender_dr": dataclasses.asdict(dr),
        "replay_drift_max_rad": drift_max,
        "episodes": episodes,
    }
    with open(Path(args.out) / "meta" / "run_meta.json", "w") as f:
        json.dump(out_meta, f, indent=2)
    flag = "OK" if drift_max < DRIFT_WARN_RAD else "WARNING: EXCEEDS TOLERANCE — do not train on this output"
    print(f"[rerender] wrote {total} episodes to {args.out}; replay drift max "
          f"{1000*drift_max:.2f} mrad ({flag})")
    env.close()


if __name__ == "__main__":
    main(tyro.cli(Args))
