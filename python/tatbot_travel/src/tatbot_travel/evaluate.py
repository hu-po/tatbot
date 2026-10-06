"""Closed-loop evaluation: a trained policy drives the simulated arm; the expert keeps score.

Offline loss has not predicted how tatbot policies behave on the rig, so a
checkpoint is judged by driving: the same episodes the generator makes, on
held-out seeds, with the policy's commands going to the servo instead of the
expert's. The expert still runs every tick on the true state, so each tick
has a reference action and a verdict:

- clearance: lens face to skin, the one number that must never approach 0;
- ink error: gap point to the nearest point of any ink stroke;
- on-ink: ticks where ink was traceable (visible, reachable, still) and the
  gap point sat within ``on_ink_m`` of it.

By default the loop waits for each chunk, as the recipe's synchronous loop
does. With ``latency_ticks`` it runs the real runner's execution instead
(``chunking``): each plan is computed from the history when it is asked for,
reaches the arm that many ticks later, is smoothed (untrained joints held)
and takes over as the runner's does, and every command is clamped to the
runner's speed cap -- so a sim score measures what the rig would execute.
"""

from __future__ import annotations

import json
import time
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from tatbot_travel.chunking import ChunkScheduler, History, Plan, condition_plan
from tatbot_travel.episode import TASK, EpisodeConfig, EpisodeRunner
from tatbot_travel.expert import Mode
from tatbot_travel.flux3 import force_natten_backend, set_sampler


@dataclass
class EpisodeScore:
    seed: int
    ticks: int = 0
    traceable: int = 0
    on_ink: int = 0
    min_clearance_m: float = float("inf")
    unsafe_ticks: int = 0
    held_ticks: int = 0  # asynchronous runs: ticks with no plan command, the arm holding
    ink_errors: list[float] = field(default_factory=list)
    chunk_seconds: list[float] = field(default_factory=list)

    def summary(self) -> dict:
        errors = np.asarray(self.ink_errors) if self.ink_errors else np.array([np.nan])
        return {
            "seed": self.seed, "ticks": self.ticks,
            "on_ink_fraction": self.on_ink / max(self.traceable, 1),
            "traceable_fraction": self.traceable / max(self.ticks, 1),
            "ink_error_median_mm": float(np.nanmedian(errors) * 1000),
            "ink_error_p90_mm": float(np.nanpercentile(errors, 90) * 1000),
            "min_clearance_mm": self.min_clearance_m * 1000, "unsafe_ticks": self.unsafe_ticks,
            "held_fraction": self.held_ticks / max(self.ticks, 1),
            "chunk_latency_median_s": float(np.median(self.chunk_seconds)) if self.chunk_seconds else None,
        }


class LeRobotPolicy:
    """A trained flux3 checkpoint (adapter or full) behind LeRobot's own loaders."""

    def __init__(self, checkpoint: Path, dataset_root: Path, device: str = "cuda", peft: bool = True, *,
                 steps: int | None = None, guidance: float | None = None, merge: bool = True,
                 seed: int | None = None):
        from lerobot.configs.policies import PreTrainedConfig
        from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata
        from lerobot.policies.factory import make_policy, make_pre_post_processors

        force_natten_backend()
        cfg = PreTrainedConfig.from_pretrained(str(checkpoint))
        cfg.pretrained_path, cfg.use_peft, cfg.device = str(checkpoint), peft, device
        set_sampler(cfg, steps, guidance)
        self.sampler = {"steps": cfg.num_inference_steps, "guidance": cfg.guidance_scale}
        self.n_obs_steps = cfg.n_obs_steps
        self.cameras = [key.rsplit(".", 1)[-1] for key in cfg.camera_keys]  # e.g. ["wrist"], or ["scene", "wrist"]
        self.device = device
        meta = LeRobotDatasetMetadata("local/travel-ink", root=dataset_root)
        self.task = str(meta.tasks.index[0]) if len(meta.tasks) else TASK  # the words it was trained with
        self.policy = make_policy(cfg, ds_meta=meta).eval()
        if merge and hasattr(self.policy, "merge_and_unload"):
            # Inference only: fold the LoRA deltas (and the saved heads) into the base weights, so each
            # projection is one kernel rather than three -- a third of a Thor's chunk time at one step.
            self.policy = self.policy.merge_and_unload()
        self.pre, self.post = make_pre_post_processors(cfg, pretrained_path=str(checkpoint))
        # flux3 draws each chunk's noise from the checkpoint's fixed ``inference_seed``, as the published
        # checkpoints run (four steps with guidance). ``seed`` draws fresh noise per plan instead: at one step
        # the same noise every plan repeats one error every plan, which delta integration sums into a drift.
        self.seeds = np.random.default_rng(seed) if seed is not None else None

    def reset(self) -> None:
        self.policy.reset()
        for pipeline in (self.pre, self.post):
            if hasattr(pipeline, "reset"):
                pipeline.reset()

    def __call__(self, image: np.ndarray, state: np.ndarray) -> np.ndarray:
        import torch

        if self.cameras != ["wrist"]:
            raise ValueError(f"the checkpoint looks through {self.cameras}: evaluate it with --latency-ticks")
        observation = {
            "observation.images.wrist": torch.from_numpy(image).permute(2, 0, 1).contiguous(),
            "observation.state": torch.from_numpy(state.astype(np.float32)),
            "task": self.task,
        }
        batch = {k: v.to(self.device) if torch.is_tensor(v) else v for k, v in self.pre(observation).items()}
        with torch.no_grad():
            action = self.post(self.policy.select_action(batch))
        return np.asarray(action.detach().cpu(), dtype=float).reshape(-1)[:6]

    def plan(self, images: np.ndarray, states: np.ndarray, commands: np.ndarray,
             scenes: np.ndarray | None = None) -> np.ndarray:
        """Absolute commands from the window's last tick on, through the recipe's explicit-history path."""
        import torch

        views = {"wrist": images, "scene": scenes}
        missing = [camera for camera in self.cameras if views.get(camera) is None]
        if missing:
            raise ValueError(f"the checkpoint looks through {self.cameras}; no images for {missing}")
        observation = {
            **{f"observation.images.{camera}": torch.from_numpy(views[camera]).permute(0, 3, 1, 2).contiguous()[None]
               for camera in self.cameras},
            "observation.state": torch.from_numpy(states.astype(np.float32))[None],
            "observation.command_history": torch.from_numpy(commands.astype(np.float32))[None],
            "task": self.task,
        }
        batch = {k: v.to(self.device) if torch.is_tensor(v) else v for k, v in self.pre(observation).items()}
        if self.seeds is not None:
            self.policy.config.inference_seed = int(self.seeds.integers(2 ** 31))
        with torch.no_grad():
            chunk = self.post(self.policy.predict_action_chunk(batch))
        return np.asarray(chunk.detach().cpu(), dtype=float)[0, :, :6]


def _score_tick(score: EpisodeScore, runner: EpisodeRunner, on_ink_m: float) -> None:
    st, t = runner.expert.state, (score.ticks - 1) / runner.cfg.fps
    pose = runner.script.pose(t)
    clearance = runner.clearance(runner.servo.q, pose)
    gap, _ = runner.kin.gap_pose(runner.servo.q)
    score.min_clearance_m = min(score.min_clearance_m, clearance)
    score.unsafe_ticks += int(clearance < 0.005)
    ink_world = pose.apply(runner.world.ink.points)
    error = float(np.min(np.linalg.norm(ink_world - gap, axis=1)))
    if st.mode == Mode.TRACE:  # the expert would be tracing here: ink is traceable
        score.traceable += 1
        score.ink_errors.append(error)
        score.on_ink += int(error <= on_ink_m)


def run_policy_episode(policy, seed: int, cfg: EpisodeConfig, on_ink_m: float = 0.006) -> EpisodeScore:
    """One held-out episode under ``policy``; the expert shadows it on the true state."""
    runner = EpisodeRunner(seed, cfg)
    score = EpisodeScore(seed=seed)
    policy.reset()
    for tick in range(int(round(cfg.duration_s * cfg.fps))):
        frame = runner.step(tick, execute=False)
        started = time.time()
        command = policy(frame.image, frame.state)
        elapsed = time.time() - started
        if elapsed > 0.05:  # a chunk was computed on this tick
            score.chunk_seconds.append(elapsed)
        runner.execute(command)
        score.ticks += 1
        _score_tick(score, runner, on_ink_m)
    runner.close()
    return score


@dataclass(frozen=True)
class Execution:
    """How the real runner executes plans (``travel run``'s defaults)."""

    held: tuple[int, ...] = ()  # joints the checkpoint never learned to move
    smooth_ticks: int = 7
    max_speed: float = 0.25  # rad/s, any joint
    timing: str = "sync"


def run_async_episode(policy, seed: int, cfg: EpisodeConfig, latency_ticks: int, on_ink_m: float = 0.006,
                      execution: Execution | None = None, on_tick=None) -> EpisodeScore:
    """One held-out episode executed as the real runner would: plans late, conditioned, rebased, clamped.

    ``on_tick(tick, runner, frame, sent)`` sees every tick (a recorder, say).
    """
    from tatbot_travel.runner import Clamp, planner_window

    execution = execution or Execution()
    uses_scene = "scene" in getattr(policy, "cameras", ["wrist"])
    runner = EpisodeRunner(seed, cfg)
    score = EpisodeScore(seed=seed)
    history = History(policy.n_obs_steps)
    scheduler = ChunkScheduler(latency_ticks, back_to_back=execution.timing == "arrival",
                               sync=execution.timing == "sync", coast=0 if execution.timing == "sync" else 4)
    clamp = Clamp(runner.kin.lower, runner.kin.upper, execution.max_speed / cfg.fps)
    arriving: deque[tuple[int, Plan, np.ndarray]] = deque()  # (arrival tick, plan, the command it integrates from)
    sent = q_start = None
    for tick in range(int(round(cfg.duration_s * cfg.fps))):
        frame = runner.step(tick, execute=False)
        if sent is None:
            sent = q_start = frame.state.astype(float)
        history.push(frame.image, frame.state, sent, scene=frame.scene if uses_scene else None)
        while arriving and arriving[0][0] <= tick:
            _, plan, anchor = arriving.popleft()
            plan = Plan(plan.start, condition_plan(plan.commands, q_start, list(execution.held), execution.smooth_ticks))
            scheduler.adopt(plan.rebased(tick, anchor, sent) if execution.timing != "aligned" else plan, tick)
        if scheduler.due(tick):
            started = time.time()
            window = planner_window(history, uses_scene)
            arriving.append((tick + latency_ticks, Plan(tick, policy.plan(*window)), sent.copy()))
            score.chunk_seconds.append(time.time() - started)
            scheduler.requested()
        score.held_ticks += int(scheduler.plan is None or scheduler.plan.at(tick) is None)
        sent = clamp(sent, scheduler.command(tick, sent))
        runner.execute(sent)
        score.ticks += 1
        _score_tick(score, runner, on_ink_m)
        if on_tick is not None:
            on_tick(tick, runner, frame, sent)
    runner.close()
    return score


def evaluate(checkpoint: Path, dataset_root: Path, seeds: list[int], seconds: float, out: Path, *,
             ink_mode: str | None = None, held_out_designs: bool = False, steps: int | None = None,
             guidance: float | None = None, latency_ticks: int | None = None,
             profile: str | None = None) -> list[dict]:
    """Score ``checkpoint`` on ``seeds``; ``ink_mode`` pins what is drawn (lines, tattoos or both)."""
    from dataclasses import replace

    from tatbot_travel.profile import load_profile
    from tatbot_travel.runner import untrained_joints

    policy = LeRobotPolicy(checkpoint, dataset_root, steps=steps, guidance=guidance)
    execution = Execution(held=tuple(untrained_joints(checkpoint)))
    header = {"checkpoint": str(checkpoint), "task": policy.task, **policy.sampler, "seconds": seconds,
              "ink": ink_mode or "mix", "held_out_designs": held_out_designs, "latency_ticks": latency_ticks}
    base = load_profile(profile, seconds)
    header["profile_id"] = base.profile_id
    world = replace(base.world, held_out_designs=held_out_designs, scene_view="scene" in policy.cameras,
                    ink_modes={ink_mode: 1.0} if ink_mode else base.world.ink_modes)
    cfg = replace(base, duration_s=seconds, world=world)
    results = []
    out.parent.mkdir(parents=True, exist_ok=True)
    print(json.dumps(header))
    for seed in seeds:
        if latency_ticks is None:
            summary = run_policy_episode(policy, seed, cfg).summary()
        else:
            summary = run_async_episode(policy, seed, cfg, latency_ticks, execution=execution).summary()
        results.append(summary)
        print(json.dumps(summary))
        out.write_text(json.dumps({**header, "episodes": results}, indent=1) + "\n")
    return results
