#!/usr/bin/env python3
"""Run policy checkpoints closed-loop in Tatbot sim and write a screen report."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import socket
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable

import numpy as np
from sim_policy_protocol import PROTOCOL, recv_message, send_message
from tatbot_digest import sha256_file
from wrist_cameras import read_checkpoint_config

if TYPE_CHECKING:
    from wire_client import (
        ActionQueue as ActionQueueType,
    )
    from wire_client import (
        ExecutionFilter as ExecutionFilterType,
    )
    from wire_client import (
        Scenario as ScenarioType,
    )
    from wire_client import (
        WirePolicyClient as WirePolicyClientType,
    )

# The LeRobot wire client (torch, lerobot) is imported only for --client-mode
# policy: the hold control needs nothing but the worker, so it can run from
# the simulator's own interpreter on a node without the serving environment.
SCENARIOS: dict[str, ScenarioType] | None = None
ActionQueue: type[ActionQueueType] | None = None
ExecutionFilter: type[ExecutionFilterType] | None = None
WirePolicyClient: type[WirePolicyClientType] | None = None
observation_from_worker: Callable[[Any, dict[str, np.ndarray], str], dict] | None = None


def _load_wire_client() -> None:
    global SCENARIOS, ActionQueue, ExecutionFilter, WirePolicyClient, observation_from_worker
    if SCENARIOS is not None:
        return
    import wire_client

    SCENARIOS = wire_client.SCENARIOS
    ActionQueue = wire_client.ActionQueue
    ExecutionFilter = wire_client.ExecutionFilter
    WirePolicyClient = wire_client.WirePolicyClient
    observation_from_worker = wire_client.observation_from_worker

REPO = Path(__file__).resolve().parents[2]
SIM_PROJECT = REPO / "python" / "tatbot_sim"
RUN_SCHEMA = "tatbot.sim-policy-battery/1"
# The hold control is a producer with no checkpoint: the client sends the
# worker's own joint state back every step, so the arm never moves and the
# sheet stays blank. Its "digest" names the contract, so a report can never
# mistake it for a model.
HOLD_CONTROL_ID = "hold-control"
HOLD_CONTROL_SHA256 = hashlib.sha256(b"tatbot.hold-control/1").hexdigest()
TOOL_BY_DISTRIBUTION = {
    "paper-draw": "lutin-ballpoint-dot",
    "skin-erase": "picosecond-laser-pen",
    "skin-tattoo": "lutin-3rl-bugpin",
    "body-tattoo": "lutin-3rl-bugpin",
}


def _tree_sha256(path: Path) -> str:
    """Content digest for a local checkpoint file or directory."""

    path = path.expanduser().resolve()
    if path.is_file():
        return sha256_file(path)
    if not path.is_dir():
        raise ValueError(f"checkpoint is not locally readable: {path}")
    digest = hashlib.sha256()
    files = sorted(item for item in path.rglob("*") if item.is_file())
    if not files:
        raise ValueError(f"checkpoint directory contains no files: {path}")
    for file in files:
        relative = file.relative_to(path).as_posix().encode()
        digest.update(len(relative).to_bytes(4, "big"))
        digest.update(relative)
        digest.update(bytes.fromhex(sha256_file(file)))
    return digest.hexdigest()


def _valid_digest(value: str | None, label: str) -> str | None:
    if value is None:
        return None
    value = value.lower()
    if len(value) != 64 or any(char not in "0123456789abcdef" for char in value):
        raise ValueError(f"{label} must be 64 lowercase hexadecimal characters")
    return value


def _checkpoint_digest(policy: str, stated: str | None) -> str:
    stated = _valid_digest(stated, "--checkpoint-sha256")
    local = Path(policy).expanduser()
    if local.exists():
        measured = _tree_sha256(local)
        if stated is not None and stated != measured:
            raise ValueError(
                f"stated checkpoint digest {stated} differs from local bytes {measured}"
            )
        return measured
    if stated is None:
        raise ValueError(
            "the policy path is server-side and cannot be hashed here; "
            "state --checkpoint-sha256"
        )
    return stated


def _canonical_json(value: object) -> bytes:
    return json.dumps(value, separators=(",", ":"), sort_keys=True).encode()


def _run_spec(args: argparse.Namespace, checkpoint: dict) -> tuple[dict, str]:
    scenarios = []
    for path in args.scenario:
        resolved = path.expanduser().resolve()
        if not resolved.is_file():
            raise ValueError(f"scenario does not exist: {resolved}")
        scenarios.append({"path": str(resolved), "sha256": sha256_file(resolved)})
    if args.distribution == "body-tattoo" and not scenarios:
        raise ValueError("body-tattoo needs at least one --scenario")
    if args.distribution != "body-tattoo" and scenarios:
        raise ValueError(f"{args.distribution} does not accept --scenario")
    contract = None
    if args.plausibility_contract:
        path = args.plausibility_contract.expanduser().resolve()
        if not path.is_file():
            raise ValueError(f"plausibility contract does not exist: {path}")
        contract = {"path": str(path), "sha256": sha256_file(path)}
    spec = {
        "schema": RUN_SCHEMA,
        "checkpoint": checkpoint,
        "client_mode": args.client_mode,
        "server": args.server,
        "policy": args.policy,
        "policy_type": args.policy_type,
        "wire_scenario": args.wire_scenario,
        "distribution": args.distribution,
        "surface_profile": args.surface_profile,
        "sensor_profile": args.sensor_profile,
        "scenarios": scenarios,
        "repetitions": args.repetitions,
        "seed": args.seed,
        "actions_per_chunk": args.actions_per_chunk,
        "chunk_size_threshold": args.chunk_size_threshold,
        "fps": 30.0,
        "target_filter_tau_s": args.target_filter_tau_s,
        "max_joint_velocity_rad_s": args.max_joint_velocity_rad_s,
        "record_video": args.record_video,
        "plausibility_contract": contract,
        "training_seed_ranges": args.training_seed_range,
    }
    return spec, hashlib.sha256(_canonical_json(spec)).hexdigest()


def _prepare_output(root: Path, spec: dict, spec_sha256: str, resume: bool) -> None:
    root = root.expanduser().resolve()
    if root.is_relative_to(REPO.resolve()):
        raise ValueError("--output-dir must be outside the repository")
    spec_path = root / "battery.json"
    if root.exists() and any(root.iterdir()):
        if not resume:
            raise ValueError(f"output directory is not empty (use --resume): {root}")
        if not spec_path.is_file():
            raise ValueError(f"resume directory has no battery.json: {root}")
        old = json.loads(spec_path.read_text())
        if old.get("spec_sha256") != spec_sha256:
            raise ValueError("resume battery configuration differs from the existing run")
        return
    root.mkdir(parents=True, exist_ok=True)
    spec_path.write_text(
        json.dumps({**spec, "spec_sha256": spec_sha256}, indent=2, sort_keys=True) + "\n"
    )


def _response(connection: socket.socket) -> tuple[dict, dict[str, np.ndarray]]:
    header, arrays = recv_message(connection)
    if header.get("ok") is not True:
        raise RuntimeError(
            f"sim worker {header.get('code', 'error')}: {header.get('message', header)}"
        )
    return header, arrays


def _connect_worker(path: Path, process: subprocess.Popen, timeout: float) -> socket.socket:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError(f"sim worker exited before opening its socket ({process.returncode})")
        if path.exists():
            connection = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
            connection.settimeout(timeout)
            try:
                connection.connect(str(path))
                return connection
            except OSError:
                connection.close()
        time.sleep(0.05)
    raise TimeoutError(f"sim worker did not open {path} within {timeout:.1f}s")


def _worker_command(
    args: argparse.Namespace,
    socket_path: Path,
    bundle: Path,
    *,
    allow_expert_actions: bool = False,
) -> tuple[list[str], dict[str, str]]:
    sim_python = args.sim_python
    if sim_python is None:
        candidate = SIM_PROJECT / ".venv" / "bin" / "python"
        if not os.access(candidate, os.X_OK):
            raise ValueError(
                f"sim interpreter missing at {candidate}; run uv sync --project {SIM_PROJECT} --extra maniskill "
                "or state --sim-python"
            )
        sim_python = candidate
    command = [
        str(sim_python),
        "-m",
        "tatbot_sim.policy_worker",
        "--socket",
        str(socket_path),
        "--distribution",
        args.distribution,
        "--output-dir",
        str(bundle),
        "--sensor-profile",
        args.sensor_profile,
    ]
    if args.surface_profile:
        command.extend(["--surface-profile", args.surface_profile])
    command.append("--record-video" if args.record_video else "--no-record-video")
    if allow_expert_actions:
        command.append("--allow-expert-actions")
    env = dict(os.environ)
    env.update({
        "TATBOT_REPO": str(REPO),
        "TATBOT_TOOL_ID": TOOL_BY_DISTRIBUTION[args.distribution],
        "TATBOT_SIM_TIP_DELTA_M": "[0,0,0]",
    })
    env.pop("TATBOT_FACTORY_REEXEC", None)
    return command, env


def _fill_with_hold(
    connection: socket.socket,
    header: dict,
    arrays: dict[str, np.ndarray],
) -> tuple[dict, dict[str, np.ndarray]]:
    while not header["done"]:
        send_message(connection, {"op": "step"}, {"action": arrays["qpos"]})
        header, arrays = _response(connection)
    return header, arrays


def _episode_plan(spec: dict) -> list[tuple[str | None, int]]:
    scenarios = [entry["path"] for entry in spec["scenarios"]] or [None]
    result = []
    index = 0
    for scenario in scenarios:
        for _ in range(spec["repetitions"]):
            result.append((scenario, spec["seed"] + index))
            index += 1
    return result


def _validate_worker_views(header: dict, robot) -> None:
    """The worker's render profile must equal the client's declared policy views."""
    worker_cameras = header["sensor_profile"]["cameras"]
    if {c["role"] for c in worker_cameras} != set(robot.cameras):
        raise ValueError("worker camera set differs from selected policy features")
    for camera in worker_cameras:
        expected = robot.cameras[camera["role"]]
        if (camera["width"], camera["height"]) != (expected.width, expected.height):
            raise ValueError("worker camera dimensions differ from policy features")


def _run_episode(
    args: argparse.Namespace,
    policy: WirePolicyClientType | None,
    checkpoint: dict,
    bundle: Path,
    scenario_path: str | None,
    seed: int,
    log_path: Path,
) -> dict:
    if bundle.exists():
        completed = bundle / "meta" / "completed.json"
        if completed.is_file():
            record = json.loads(completed.read_text())
            if (
                record.get("checkpoint_sha256") == checkpoint["sha256"]
                and record.get("seed") == seed
            ):
                return {"status": "resumed", "bundle": str(bundle)}
        # A worker creates only this run-scoped directory. A partial episode
        # has no usable state to resume, so replay it from its deterministic
        # seed while preserving the complete siblings in the battery.
        shutil.rmtree(bundle)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="tatbot-sim-policy-") as temp:
        socket_path = Path(temp) / "worker.sock"
        command, env = _worker_command(args, socket_path, bundle)
        with log_path.open("w") as log:
            process = subprocess.Popen(
                command,
                cwd=REPO,
                env=env,
                stdout=log,
                stderr=subprocess.STDOUT,
                text=True,
            )
            connection = None
            try:
                connection = _connect_worker(socket_path, process, args.timeout)
                send_message(connection, {"op": "hello"})
                hello, _ = _response(connection)
                send_message(
                    connection,
                    {"op": "reset", "scenario": scenario_path, "seed": seed},
                )
                header, arrays = _response(connection)
                task = header["task"]
                chunks = []
                rejections = []
                execution: ExecutionFilterType | None = None
                if (
                    policy is None
                    or ActionQueue is None
                    or ExecutionFilter is None
                    or observation_from_worker is None
                ):
                    # hold control: the worker's own qpos goes straight back
                    header, arrays = _fill_with_hold(connection, header, arrays)
                else:
                    queue = ActionQueue(action_dim=7)
                    _validate_worker_views(header, policy.robot)
                    execution = ExecutionFilter(
                        arrays["qpos"],
                        fps=header["control_freq"],
                        target_filter_tau_s=args.target_filter_tau_s,
                        max_joint_velocity_rad_s=args.max_joint_velocity_rad_s,
                    )
                    while not header["done"]:
                        if queue.should_replenish(
                            policy.actions_per_chunk, args.chunk_size_threshold
                        ):
                            observation = observation_from_worker(policy.robot, arrays, task)
                            started = time.perf_counter()
                            try:
                                incoming = policy.predict(
                                    observation, max(queue.latest, 0), must_go=True
                                )
                                merge = queue.merge(incoming)
                            except (ValueError, TimeoutError) as exc:
                                rejections.append({
                                    "step": int(header["step"]),
                                    "kind": type(exc).__name__,
                                    "message": str(exc),
                                })
                                header, arrays = _fill_with_hold(connection, header, arrays)
                                break
                            timesteps = [int(item.get_timestep()) for item in incoming]
                            chunks.append({
                                "observation_step": int(header["step"]),
                                "latency_ms": (time.perf_counter() - started) * 1000.0,
                                "incoming": len(incoming),
                                "timestep_min": min(timesteps) if timesteps else None,
                                "timestep_max": max(timesteps) if timesteps else None,
                                **merge,
                            })
                        if not len(queue):
                            rejections.append({
                                "step": int(header["step"]),
                                "kind": "empty_action_queue",
                                "message": "server response contained no executable future action",
                            })
                            header, arrays = _fill_with_hold(connection, header, arrays)
                            break
                        _, requested = queue.pop()
                        executed = execution.apply(requested)
                        send_message(connection, {"op": "step"}, {"action": executed})
                        header, arrays = _response(connection)
                client_meta = {
                    "mode": args.client_mode,
                    "deployment_filter": policy is not None,
                    "wire_scenario": args.wire_scenario,
                    "policy_type": policy.policy_type if policy is not None else None,
                    "actions_per_chunk": policy.actions_per_chunk if policy is not None else None,
                    "chunk_size_threshold": args.chunk_size_threshold,
                    "aggregate_fn": "weighted_average(old=0.3,new=0.7)" if policy is not None else None,
                    "execution": execution.stats() if execution is not None else None,
                    "feature_shapes": {
                        key: value.get("shape") if isinstance(value, dict) else value
                        for key, value in policy.features.items()
                    } if policy is not None else None,
                    "plausibility_contract": (
                        None
                        if args.plausibility_contract is None
                        else {
                            "path": str(args.plausibility_contract.expanduser().resolve()),
                            "sha256": sha256_file(args.plausibility_contract.expanduser().resolve()),
                            "enforcement": "policy-server; local client records transport rejects",
                        }
                    ),
                    "worker": {
                        "schema": hello["worker_schema"],
                        "tool_id": hello["tool_id"],
                    },
                }
                send_message(connection, {
                    "op": "finish",
                    "checkpoint": checkpoint,
                    "worker_protocol": PROTOCOL,
                    "client": client_meta,
                    "chunk_records": chunks,
                    "chunk_rejections": rejections,
                })
                result, _ = _response(connection)
                send_message(connection, {"op": "close"})
                _response(connection)
                connection.close()
                connection = None
                return {
                    "status": "completed",
                    "bundle": result["bundle"],
                    "f1": result["score"]["f1"],
                    "steps": result["steps_executed"],
                    "chunk_rejections": len(rejections),
                }
            finally:
                if connection is not None:
                    connection.close()
                if process.poll() is None:
                    try:
                        # A successful close gets a chance to release Vulkan,
                        # encoders, and the Unix socket itself. On an error the
                        # connection is still live/non-None, so do not wait the
                        # full grace period for a server loop that cannot end.
                        process.wait(timeout=10 if connection is None else 0.1)
                    except subprocess.TimeoutExpired:
                        process.terminate()
                        try:
                            process.wait(timeout=10)
                        except subprocess.TimeoutExpired:
                            process.kill()
                            process.wait(timeout=10)
                else:
                    process.wait()
                if process.returncode not in (0, -15):
                    log.flush()
                    tail = log_path.read_text(errors="replace").splitlines()[-20:]
                    if sys.exc_info()[0] is None:
                        raise RuntimeError(
                            f"sim worker exited {process.returncode}; tail:\n" + "\n".join(tail)
                        )


def _write_battery_results(root: Path, results: list[dict]) -> None:
    (root / "episodes.json").write_text(
        json.dumps({"schema": RUN_SCHEMA, "episodes": results}, indent=2, sort_keys=True) + "\n"
    )


def _run_report(
    args: argparse.Namespace,
    checkpoint: dict,
    bundles: list[Path],
    report_dir: Path,
) -> None:
    if report_dir.exists():
        if (report_dir / "report.json").is_file() and args.resume:
            return
        raise ValueError(f"report output already exists: {report_dir}")
    command = [
        str(args.sim_python or (SIM_PROJECT / ".venv" / "bin" / "python")),
        "-m",
        "tatbot_sim.eval_cli",
        *(str(path) for path in bundles),
        "--output-dir",
        str(report_dir),
        "--checkpoint-id",
        checkpoint["id"],
        "--checkpoint-sha256",
        checkpoint["sha256"],
    ]
    if checkpoint.get("base_model_sha256"):
        command.extend(["--base-model-sha256", checkpoint["base_model_sha256"]])
    for low, high in args.training_seed_range:
        command.extend(["--training-seed-range", str(low), str(high)])
    env = dict(os.environ)
    env.update({
        "TATBOT_REPO": str(REPO),
        "TATBOT_TOOL_ID": TOOL_BY_DISTRIBUTION[args.distribution],
        "TATBOT_SIM_TIP_DELTA_M": "[0,0,0]",
    })
    completed = subprocess.run(command, cwd=REPO, env=env, check=False)
    if completed.returncode:
        raise RuntimeError(f"sim evaluation report failed with exit {completed.returncode}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--client-mode", choices=("policy", "hold-control"), default="policy",
        help="policy: drive the worker from a served checkpoint over the deployed gRPC wire; "
             "hold-control: send the worker's own joint state back every step (a blank-sheet "
             "control that needs no server, no checkpoint and no wire scenario)",
    )
    parser.add_argument("--server", help="existing LeRobot policy server host:port")
    parser.add_argument("--policy", help="checkpoint path as seen by the server")
    parser.add_argument("--checkpoint-config", type=Path, help="local config.json for a server-side checkpoint")
    parser.add_argument("--policy-type")
    parser.add_argument("--wire-scenario", help="follower wire scenario, e.g. act_rgbd14_masked (policy mode)")
    parser.add_argument("--distribution", choices=sorted(TOOL_BY_DISTRIBUTION), required=True)
    parser.add_argument("--scenario", type=Path, action="append", default=[])
    parser.add_argument("--surface-profile", choices=("flat", "cylinder", "balanced"))
    parser.add_argument("--sensor-profile", choices=("deployment", "legacy-two-view"), default="deployment")
    parser.add_argument("--repetitions", type=int, default=3)
    parser.add_argument("--seed", type=int, default=20260903)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--checkpoint-id")
    parser.add_argument("--checkpoint-sha256")
    parser.add_argument("--base-model-sha256")
    parser.add_argument("--actions-per-chunk", type=int)
    parser.add_argument("--chunk-size-threshold", type=float, default=0.5)
    parser.add_argument("--target-filter-tau-s", type=float, default=0.3)
    parser.add_argument("--max-joint-velocity-rad-s", type=float, default=0.25)
    parser.add_argument("--plausibility-contract", type=Path)
    parser.add_argument(
        "--training-seed-range", action="append", nargs=2, type=int, default=[],
        metavar=("MIN", "MAX"),
    )
    parser.add_argument("--record-video", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--timeout", type=float, default=90.0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--sim-python", type=Path)
    parser.add_argument("--resume", action="store_true")
    return parser


def run(args: argparse.Namespace) -> dict:
    if args.repetitions <= 0:
        raise ValueError("--repetitions must be positive")
    if not 0 <= args.chunk_size_threshold <= 1:
        raise ValueError("--chunk-size-threshold must be in [0, 1]")
    if args.client_mode == "hold-control":
        for name, value in (("--server", args.server), ("--policy", args.policy),
                            ("--wire-scenario", args.wire_scenario),
                            ("--checkpoint-id", args.checkpoint_id),
                            ("--checkpoint-sha256", args.checkpoint_sha256)):
            if value:
                raise ValueError(f"{name} has no meaning for --client-mode hold-control")
        checkpoint = {"id": HOLD_CONTROL_ID, "sha256": HOLD_CONTROL_SHA256, "base_model_sha256": None}
    else:
        if not (args.server and args.policy and args.wire_scenario):
            raise ValueError("--client-mode policy needs --server, --policy and --wire-scenario")
        base_sha = _valid_digest(args.base_model_sha256, "--base-model-sha256")
        digest = _checkpoint_digest(args.policy, args.checkpoint_sha256)
        checkpoint = {
            "id": args.checkpoint_id or Path(args.policy).name,
            "sha256": digest,
            "base_model_sha256": base_sha,
        }
        config_path, _ = read_checkpoint_config(args.policy, args.checkpoint_config)
        checkpoint["config"] = {"path": str(config_path), "sha256": sha256_file(config_path)}
    spec, spec_sha = _run_spec(args, checkpoint)
    root = args.output_dir.expanduser().resolve()
    _prepare_output(root, spec, spec_sha, args.resume)
    report_dir = root / "report"
    plan = _episode_plan(spec)
    bundles = [root / "episodes" / f"episode-{index:04d}-seed-{seed}" for index, (_, seed) in enumerate(plan)]
    if report_dir.is_dir() and (report_dir / "report.json").is_file() and args.resume:
        report = json.loads((report_dir / "report.json").read_text())
        print(f"sim policy eval already complete: {report_dir}")
        return report
    policy = None
    if args.client_mode == "policy":
        _load_wire_client()
        if SCENARIOS is None or WirePolicyClient is None:
            raise RuntimeError("failed to load wire client module")
        if args.wire_scenario not in SCENARIOS:
            raise ValueError(f"unknown --wire-scenario {args.wire_scenario!r}; one of {sorted(SCENARIOS)}")
        policy = WirePolicyClient(
            server=args.server,
            policy=args.policy,
            scenario=SCENARIOS[args.wire_scenario],
            policy_type=args.policy_type,
            actions_per_chunk=args.actions_per_chunk,
            device=args.device,
            timeout=args.timeout,
            sensor_profile=args.sensor_profile,
            checkpoint_config=args.checkpoint_config,
        )
        policy.connect()
    results = []
    try:
        for index, ((scenario_path, seed), bundle) in enumerate(zip(plan, bundles, strict=True)):
            result = _run_episode(
                args,
                policy,
                checkpoint,
                bundle,
                scenario_path,
                seed,
                root / "logs" / f"episode-{index:04d}.log",
            )
            results.append(result)
            _write_battery_results(root, results)
            print(
                f"[{index + 1}/{len(plan)}] {result['status']} seed={seed} "
                f"f1={result.get('f1', 'retained')}",
                flush=True,
            )
    finally:
        if policy is not None:
            policy.close()
    _run_report(args, checkpoint, bundles, report_dir)
    report = json.loads((report_dir / "report.json").read_text())
    print(
        f"wrote checkpoint screen {report_dir}: "
        f"f1={report['metrics']['f1']['mean']:.4f} "
        f"interpretation={report['interpretation']}"
    )
    return report


def main() -> int:
    try:
        run(build_parser().parse_args())
    except (OSError, ValueError, RuntimeError, TimeoutError) as exc:
        print(f"sim policy eval: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
