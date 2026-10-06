"""One-environment ManiSkill worker for closed-loop checkpoint evaluation.

The worker owns every simulator object.  A separate LeRobot client process
owns the policy-server connection and exchanges only versioned JSON plus
explicit arrays over a Unix socket.  This keeps ManiSkill, LeRobot client, and
policy-serving dependency environments independent.
"""

from __future__ import annotations

import argparse
import json
import os
import socket
import sys
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
from sim_policy_protocol import (  # scripts/eval, via tatbot_sim's bootstrap
    ProtocolError,
    recv_message,
    send_message,
)
from tatbot_contracts.digest import sha256_file
from tatbot_contracts.observations import ObservationProfile

from tatbot_sim import interaction, tools
from tatbot_sim import judge as judging
from tatbot_sim.backends.maniskill import ManiSkillWorld
from tatbot_sim.distributions import DISTRIBUTIONS
from tatbot_sim.episode import Episode
from tatbot_sim.expert import (
    StrokeExpert,
    reachable_canvas_masks,
    reachable_height_ceiling,
)
from tatbot_sim.inkmap.contracts import document_sha256
from tatbot_sim.observations import ObservationBuilder
from tatbot_sim.planning import plan_batch, plan_tattoo_scenario
from tatbot_sim.repo import repo_root, source_state

WORKER_SCHEMA = "tatbot.sim-policy-worker-result/1"


def _finite_action(value: object) -> np.ndarray:
    action = np.asarray(value, dtype=np.float32)
    if action.shape != (7,) or not np.isfinite(action).all():
        raise ValueError(f"action must be a finite 7-vector, got {action.shape}")
    return action


class PolicyEpisode:
    """A single deterministic scenario controlled one 7-vector at a time."""

    def __init__(
        self,
        *,
        distribution: str,
        scenario_path: str | None,
        seed: int,
        output_dir: Path,
        surface_profile: str | None,
        record_video: bool,
        allow_expert_actions: bool,
        sensor_profile: str = "deployment",
    ):
        if distribution not in DISTRIBUTIONS:
            raise ValueError(f"unknown distribution {distribution!r}")
        dist = DISTRIBUTIONS[distribution]
        if distribution == "body-tattoo" and not scenario_path:
            raise ValueError("body-tattoo evaluation requires a compiled --scenario")
        if distribution != "body-tattoo" and scenario_path:
            raise ValueError(f"{distribution} does not accept a posed-body scenario")
        if surface_profile is not None and surface_profile not in {"flat", "cylinder", "balanced"}:
            raise ValueError(f"invalid surface profile {surface_profile!r}")
        output_dir = output_dir.expanduser().resolve()
        if output_dir.is_relative_to(repo_root().resolve()):
            raise ValueError("policy evaluation output must be outside the repository")
        if output_dir.exists() and any(output_dir.iterdir()):
            raise ValueError(f"policy evaluation episode directory is not empty: {output_dir}")
        output_dir.mkdir(parents=True, exist_ok=True)

        self.distribution = distribution
        self.scenario_path = scenario_path
        self.seed = int(seed)
        self.output_dir = output_dir
        self.record_video = record_video
        self.allow_expert_actions = allow_expert_actions
        self.source_start = source_state()
        self.finished = False
        self.closed = False
        self.step_index = 0
        self.actions: list[list[float]] = []

        args = dist.build_args()
        args.out_dir = str(output_dir)
        args.num_envs = 1
        args.num_episodes = 1
        args.seed = self.seed
        args.task = "language"
        args.scenario = scenario_path
        args.depth = True
        args.judge = True
        args.reconfigure_each_batch = False
        # Evaluation needs an exact ceiling trajectory. DART perturbations are
        # a training-data feature, not part of the requested design.
        args.dr.noise.prob = (0.0, 0.0)
        args.dr.noise.scale = (0.0, 0.0)
        args.dr.rgb.enabled = False
        args.dr.corrupt_depth = False
        if surface_profile is not None:
            args.dr.surface.profile = surface_profile
        self.args = args
        from tatbot_sim.resolved import resolve
        self.config = resolve(tool_id=dist.tool_id, sensor_profile=sensor_profile,
            seed=self.seed, dr=args.dr, scenario_path=scenario_path,
            supply=(args.supply, args.supply_ink),
            observation_profile=ObservationProfile(effort="unavailable"))
        self.geometry = self.config.geometry
        self.geometry_warnings = tools.geometry_warnings(self.config.tool, self.geometry)
        for warning in self.geometry_warnings:
            print(f"[sim-policy-worker] WARNING: {warning}", file=sys.stderr, flush=True)

        from tatbot_sim.env import TatbotDrawEnv

        self.env = TatbotDrawEnv(
            config=self.config,
            num_envs=1,
            obs_mode="rgbd",
            control_mode="pd_joint_pos",
            sim_backend=args.sim_backend,
            texture_refresh_steps=args.texture_refresh_steps,
            reconfiguration_freq=0,
        )
        self.base_env: TatbotDrawEnv = self.env.unwrapped
        self.camera_descriptions = self.base_env.agent.camera_descriptions
        self.cameras = tuple(camera.role for camera in self.camera_descriptions)
        self.expert = StrokeExpert(1, self.base_env.device, config=self.config, noise=args.dr.noise, seed=self.config.seed_for("noise"))
        self.world = ManiSkillWorld(self.env, self.config, self.expert)
        self.runtime = Episode(self.world, ObservationBuilder(self.config, 1, self.base_env.device))
        self.runtime.reset(seed=self.seed)
        self.robot, self.idx7, self.idx_ik = self.world.robot, self.world.idx7, self.world.idx_ik
        self._plan_and_place()
        field = self.base_env.ink_field
        if field is None:
            raise RuntimeError("environment has no ink field")
        self.coverage_start = float(self.runtime.coverage_start.cpu().numpy()[0])
        intended_strokes = [judging.strokes_from_plan_paths(self.plan.paths[0])]
        self.intended_field = judging.intended_field_like(
            field,
            self.base_env.surface,
            intended_strokes,
            self.base_env.ink_opacity,
        ).clone()
        self.video_writers: dict[str, cv2.VideoWriter] = {}
        if record_video:
            self._open_video_writers()
        self.runtime.measure()
        self._write_video()

    def _plan_and_place(self) -> None:
        args = self.args
        base = self.base_env
        rng = np.random.default_rng(self.seed)
        q_now = self.robot.get_qpos()[:, self.idx_ik]
        slack = args.dr.pen_lean.max_off_base_rad
        masks = reachable_canvas_masks(
            self.expert,
            q_now,
            base.surface,
            args.draw_clearance,
            1,
            max_off_base_rad=slack,
        )
        ceiling = reachable_height_ceiling(
            self.expert, q_now, base.surface, 1, max_off_base_rad=slack
        )
        if args.scenario:
            if base.body_scenario is None:
                raise RuntimeError("compiled scenario did not load into the environment")
            plan = plan_tattoo_scenario(
                rng,
                base.body_scenario,
                base.surface,
                horizon=args.horizon, config=self.config,
                num_envs=1,
                dr=args.dr,
                draw_clearance=args.draw_clearance,
                tool_ceiling=ceiling,
            )
        else:
            plan = plan_batch(
                rng,
                base.pad_sheets,
                base.surface,
                task="language",
                horizon=args.horizon, config=self.config,
                num_envs=1,
                dr=args.dr,
                draw_clearance=args.draw_clearance,
                task_name=args.task_name,
                maze_task_name=args.maze_task_name,
                reachable=masks,
                tool_ceiling=ceiling,
                cap_rims=base.cap_rims_np(),
                dip_task_name=args.dip_task_name,
            )
        self.plan = plan
        self.horizon = int(plan.lengths[0])
        if self.horizon <= 0:
            raise RuntimeError("planned policy episode has no steps")
        self.runtime.install(plan)
        _, rots = base.canvas_frame_np
        if self.expert.actions is None:
            raise RuntimeError("expert failed to materialize the reference actions")
        self.expert_actions = (
            self.expert.actions[0, : self.horizon].detach().cpu().numpy().astype(np.float32)
        )
        if self.expert_actions.shape != (self.horizon, 7):
            raise RuntimeError(
                f"expert reference shape {self.expert_actions.shape} does not match horizon"
            )
        self.surface_point = [float(value) for value in base.pad_top_center[0].cpu().numpy()]
        self.surface_normal = [float(value) for value in rots[0, :, 2]]

    def _open_video_writers(self) -> None:
        video_dir = self.output_dir / "videos"
        video_dir.mkdir(parents=True, exist_ok=True)
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        for camera in self.cameras:
            path = video_dir / f"episode_000000_{camera}.mp4"
            description = next(view for view in self.camera_descriptions if view.role == camera)
            writer = cv2.VideoWriter(str(path), fourcc, description.fps, (description.width, description.height))
            if not writer.isOpened():
                raise RuntimeError(f"could not open policy evidence video {path}")
            self.video_writers[camera] = writer

    def _write_video(self) -> None:
        observation = self.runtime.measure()
        for camera, writer in self.video_writers.items():
            rgb = observation.rgb[camera][0].detach().cpu().numpy()
            writer.write(cv2.cvtColor(np.asarray(rgb, dtype=np.uint8), cv2.COLOR_RGB2BGR))

    def observation_arrays(self) -> dict[str, np.ndarray]:
        observation = self.runtime.measure()
        arrays: dict[str, np.ndarray] = {
            "qpos": observation.qpos[0].detach().cpu().numpy().astype(np.float32),
            "external_effort": observation.external_effort[0].detach().cpu().numpy().astype(np.float32),
        }
        for camera in self.cameras:
            arrays[f"{camera}_rgb"] = observation.rgb[camera][0].detach().cpu().numpy().astype(np.uint8)
            depth = observation.depth_mm[camera][0].detach().cpu().numpy()
            if depth.ndim == 3 and depth.shape[-1] == 1:
                depth = depth[..., 0]
            arrays[f"{camera}_depth"] = depth.astype(np.uint16)
        return arrays

    def header(self) -> dict[str, Any]:
        return {
            "ok": True,
            "task": self.plan.tasks[0],
            "kind": self.plan.kinds[0],
            "horizon": self.horizon,
            "step": self.step_index,
            "done": self.runtime.done,
            "time_s": self.runtime.time_s,
            "observation_profile": self.runtime.observations.profile.metadata(),
            "distribution": self.distribution,
            "surface_profile": self.base_env.surface_profiles[0],
            "external_effort_basis": "zero-unavailable-in-sim",
            "control_freq": self.base_env.control_freq,
            "sensor_profile": {"name": self.base_env.sensor_profile, "cameras": [camera.as_dict() for camera in self.camera_descriptions]},
            "geometry_basis": tools.geometry_basis(self.geometry),
            "geometry_warnings": self.geometry_warnings,
        }

    def step(self, action: object) -> tuple[dict, dict[str, np.ndarray]]:
        if self.finished or self.runtime.done:
            raise RuntimeError("episode is already complete")
        action_np = _finite_action(action)
        self.actions.append(action_np.tolist())
        self.runtime.step(torch.as_tensor(action_np[None], device=self.base_env.device))
        self.step_index = self.runtime.step_index
        self._write_video()
        return self.header(), self.observation_arrays()

    def _close_video_writers(self) -> None:
        for writer in self.video_writers.values():
            writer.release()
        self.video_writers.clear()

    def finish(self, provenance: dict[str, Any]) -> dict:
        if self.finished:
            raise RuntimeError("episode result was already finalized")
        self._close_video_writers()
        field = self.base_env.ink_field
        if field is None:
            raise RuntimeError("episode has no ink field")
        score = judging.score_fields(
            field.field,
            self.intended_field,
            texel_per_m=self.base_env.surface.texel_per_m,
            tolerance_m=float(field.pen_radius_m.max()),
        )[0]
        eval_dir = self.output_dir / "meta" / "eval"
        eval_dir.mkdir(parents=True, exist_ok=True)
        intended_path = eval_dir / "episode_000000_intended.png"
        drawn_path = eval_dir / "episode_000000_drawn.png"
        overlay_path = eval_dir / "episode_000000_overlay.png"
        intended = self.intended_field[0].detach().cpu().numpy()
        drawn = field.field[0].detach().cpu().numpy()
        cv2.imwrite(str(intended_path), (255 * intended).astype(np.uint8))
        cv2.imwrite(str(drawn_path), (255 * drawn).astype(np.uint8))
        overlay = np.zeros((*drawn.shape, 3), dtype=np.uint8)
        overlay[..., 1] = (255 * intended).astype(np.uint8)
        overlay[..., 2] = (255 * drawn).astype(np.uint8)
        cv2.imwrite(str(overlay_path), overlay)
        artifacts = {
            "intended": str(intended_path.relative_to(self.output_dir)),
            "drawn": str(drawn_path.relative_to(self.output_dir)),
            "overlay": str(overlay_path.relative_to(self.output_dir)),
        }
        ink_stats = self.runtime.statistics()
        coverage_end = float(field.coverage().cpu().numpy()[0])
        scenario = None
        if self.base_env.body_scenario is not None:
            if self.scenario_path is None:
                raise RuntimeError("body scenario loaded without scenario_path")
            body = self.base_env.body_scenario
            scenario = {
                "source_path": str(Path(self.scenario_path).expanduser().resolve()),
                "sha256": document_sha256(body),
                "trace_sha256": body["trace"]["sha256"],
                "body": body["body"]["id"],
                "pose": body["pose"]["id"],
                "placement": body["placement"]["id"],
                "design": body["design"]["id"],
            }
        source_end = source_state()
        videos = {}
        for camera in self.cameras:
            path = self.output_dir / "videos" / f"episode_000000_{camera}.mp4"
            if path.is_file():
                videos[camera] = str(path.relative_to(self.output_dir))
        episode = {
            "episode": 0,
            "resolved_config": self.config.metadata(),
            "runtime": self.runtime.metadata(),
            "seed": self.seed,
            "kind": self.plan.kinds[0],
            "program": self.plan.programs[0],
            "strokes_canvas_m": self.plan.paths[0],
            "task": self.plan.tasks[0],
            "surface_profile": self.base_env.surface_profiles[0],
            "surface_point": self.surface_point,
            "surface_normal": self.surface_normal,
            "steps_planned": self.horizon,
            "steps_executed": self.step_index,
            "drawing_score": {**score.as_dict(), "artifacts": artifacts},
            "ink_coverage_start": self.coverage_start,
            "ink_coverage_end": coverage_end,
            "engaged": judging.engaged(self.plan.kinds[0], self.coverage_start, coverage_end, int(ink_stats["dips"][0])),
            "interaction": {
                "model": interaction.model_for(collision=self.base_env.surface_has_contact_collision),
                "frames": int(ink_stats["interaction_frames"][0]),
                "distance_min_m": _finite_or_none(ink_stats["interaction_min_m"][0]),
                "distance_mean_m": _finite_or_none(ink_stats["interaction_mean_m"][0]),
                "distance_max_m": _finite_or_none(ink_stats["interaction_max_m"][0]),
            },
            "ink": {
                "used_ul": float(ink_stats["used_ul"][0]),
                "contact_mm": float(ink_stats["contact_mm"][0]),
                "contact_s": float(ink_stats["contact_s"][0]),
            },
            "videos": videos,
        }
        checkpoint = provenance.get("checkpoint")
        client = provenance.get("client")
        if not isinstance(checkpoint, dict) or not isinstance(client, dict):
            raise ValueError("finish provenance requires checkpoint and client objects")
        run_meta = {
            "schema_version": 2,
            "worker_schema": WORKER_SCHEMA,
            "sensor_profile": self.header()["sensor_profile"],
            "config": {
                "distribution": self.distribution,
                "seed": self.seed,
                "surface_profile": self.base_env.surface_profiles[0],
                "scenario": self.scenario_path,
                "horizon": self.horizon,
            },
            "tool": {
                "id": self.config.tool.tool_id,
                "geometry_basis": tools.geometry_basis(self.geometry),
                "qualification": (
                    "qualified"
                    if self.geometry.contact_status == "pivot-calibrated"
                    else "development"
                ),
                "geometry_warnings": self.geometry_warnings,
            },
            "software": {
                "repository": self.source_start["repository"],
                "revision_start": self.source_start["revision"],
                "revision_end": source_end["revision"],
                "dirty_start": self.source_start["dirty"],
                "dirty_end": source_end["dirty"],
            },
            "scenario": scenario,
            "episodes": [episode],
            "evaluation": {
                "schema": "tatbot.sim-eval-input/1",
                "producer": "policy",
                "checkpoint_id": checkpoint.get("id"),
                "checkpoint_sha256": checkpoint.get("sha256"),
                "base_model_sha256": checkpoint.get("base_model_sha256"),
                "judge": "tatbot_sim.judge/1",
                "headline": "f1",
                "visual_artifacts": ["intended", "drawn", "overlay"],
                "videos": videos,
                "protocol": provenance.get("worker_protocol"),
                "client": client,
                "external_effort": {
                    "basis": "zero-unavailable-in-sim",
                    "warning": "Evaluation selects the unavailable-effort observation profile",
                },
                "chunk_records": provenance.get("chunk_records", []),
                "chunk_rejections": provenance.get("chunk_rejections", []),
                "disclaimer": (
                    "Simulation evaluation is a screening result only. It does not "
                    "authorize powered motion or human contact."
                ),
            },
        }
        meta_dir = self.output_dir / "meta"
        meta_dir.mkdir(parents=True, exist_ok=True)
        run_meta_path = meta_dir / "run_meta.json"
        run_meta_path.write_text(json.dumps(run_meta, indent=2, sort_keys=True) + "\n")
        completed = {
            "schema": WORKER_SCHEMA,
            "run_meta_sha256": sha256_file(run_meta_path),
            "checkpoint_id": checkpoint.get("id"),
            "checkpoint_sha256": checkpoint.get("sha256"),
            "seed": self.seed,
        }
        (meta_dir / "completed.json").write_text(
            json.dumps(completed, indent=2, sort_keys=True) + "\n"
        )
        self.finished = True
        return {
            "ok": True,
            "bundle": str(self.output_dir),
            "score": score.as_dict(),
            "steps_executed": self.step_index,
            "completed": completed,
        }

    def close(self) -> None:
        if self.closed:
            return
        self._close_video_writers()
        self.runtime.close()
        self.closed = True


def _finite_or_none(value: Any) -> float | None:
    number = float(value)
    return number if np.isfinite(number) else None


class WorkerServer:
    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.episode: PolicyEpisode | None = None

    def dispatch(self, header: dict, arrays: dict[str, np.ndarray]) -> tuple[dict, dict]:
        op = header.get("op")
        if op == "hello":
            if arrays:
                raise ValueError("hello does not accept arrays")
            return {
                "ok": True,
                "op": "hello",
                "worker_schema": WORKER_SCHEMA,
                "distribution": self.args.distribution,
                "tool_id": DISTRIBUTIONS[self.args.distribution].tool_id,
                "capabilities": ["reset", "step", "finish", "close"],
            }, {}
        if op == "reset":
            if self.episode is not None:
                raise RuntimeError("worker supports one reset per process")
            scenario = header.get("scenario")
            if scenario is not None and not isinstance(scenario, str):
                raise ValueError("scenario must be a string path or null")
            seed = header.get("seed")
            if not isinstance(seed, int) or isinstance(seed, bool):
                raise ValueError("seed must be an integer")
            self.episode = PolicyEpisode(
                distribution=self.args.distribution,
                scenario_path=scenario,
                seed=seed,
                output_dir=self.args.output_dir,
                surface_profile=self.args.surface_profile,
                record_video=self.args.record_video,
                allow_expert_actions=self.args.allow_expert_actions,
                sensor_profile=self.args.sensor_profile,
            )
            response = self.episode.header()
            response["op"] = "reset"
            return response, self.episode.observation_arrays()
        if self.episode is None:
            raise RuntimeError("reset must precede this operation")
        if op == "step":
            if set(arrays) != {"action"}:
                raise ValueError("step requires exactly one action array")
            response, result_arrays = self.episode.step(arrays["action"])
            response["op"] = "step"
            return response, result_arrays
        if op == "expert_actions":
            if not self.episode.allow_expert_actions:
                raise PermissionError("expert actions are disabled for this worker")
            return {
                "ok": True,
                "op": "expert_actions",
            }, {"actions": self.episode.expert_actions}
        if op == "finish":
            return {"op": "finish", **self.episode.finish(header)}, {}
        if op == "close":
            return {"ok": True, "op": "close"}, {}
        raise ValueError(f"unknown worker operation {op!r}")

    def serve(self, connection: socket.socket) -> None:
        while True:
            try:
                header, arrays = recv_message(connection)
                op = header.get("op")
                response, response_arrays = self.dispatch(header, arrays)
                send_message(connection, response, response_arrays)
                if op == "close":
                    return
            except (ProtocolError, ValueError, RuntimeError, PermissionError) as exc:
                send_message(
                    connection,
                    {
                        "ok": False,
                        "op": "error",
                        "code": type(exc).__name__,
                        "message": str(exc),
                    },
                )
                if isinstance(exc, ProtocolError):
                    return

    def close(self) -> None:
        if self.episode is not None:
            self.episode.close()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--socket", type=Path, required=True)
    parser.add_argument("--distribution", choices=sorted(DISTRIBUTIONS), required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--sensor-profile", choices=("deployment", "legacy-two-view"), default="deployment")
    parser.add_argument("--surface-profile", choices=("flat", "cylinder", "balanced"))
    parser.add_argument("--record-video", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--allow-expert-actions", action="store_true", help=argparse.SUPPRESS)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    path = args.socket.expanduser().resolve()
    if path.exists():
        raise SystemExit(f"worker socket already exists: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    server = WorkerServer(args)
    listener = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    try:
        listener.bind(str(path))
        os.chmod(path, 0o600)
        listener.listen(1)
        connection, _ = listener.accept()
        with connection:
            server.serve(connection)
    finally:
        server.close()
        listener.close()
        path.unlink(missing_ok=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
