#!/usr/bin/env python3
"""Attended right-arm joint and carriage measurements with the current arm owner.

The owner retains every feedback tick. This conductor records labelled motion
and rest windows; it never applies a controller setting or touches the other arm.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import subprocess
import sys
import time
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "scripts/vision"))
sys.path.insert(0, str(REPO / "scripts/lib"))
import arm_calibration as recipe  # noqa: E402
import tatbot_runlog  # noqa: E402
from arm_guide_owner import CaptureError, Owner, validate_native_build  # noqa: E402
from tool_spec import read_workspace  # noqa: E402


def write_row(path: Path, kind: str, **body) -> None:
    with path.open("a") as stream:
        stream.write(json.dumps({"kind": kind, "wall_ns": time.time_ns(), **body}) + "\n")


def native_identity(binary: Path) -> dict:
    build = json.loads(subprocess.check_output([str(binary), "--build-info"], text=True))
    sources = recipe.native_source_digests(REPO)
    receipt = {"build": build, "binary_sha256": hashlib.sha256(binary.read_bytes()).hexdigest()}
    validate_native_build(receipt, sources, os.uname().machine)
    return receipt


def command(owner: Owner, log: Path, body: dict, expected: str, timeout: float = 40) -> dict:
    write_row(log, "command", command=body)
    owner.send(body)
    event = owner.wait(expected, timeout=timeout)
    write_row(log, "event", event=event)
    return event


def carriage_to(owner: Owner, log: Path, target_m: float) -> None:
    event = command(owner, log, {"cmd": "carriage", "carriage_m": target_m}, "carriage")
    measured = event.get("measured") or {}
    actual = (measured.get("carriage") or {}).get("position_m")
    if not isinstance(actual, (float, int)) or abs(actual - target_m) > 0.0001:
        raise CaptureError("carriage did not reach its small free-air target")


def rest(owner: Owner, log: Path, label: str, duration_s: float = 2.5) -> None:
    start_ns = time.time_ns()
    deadline = time.monotonic() + duration_s
    while time.monotonic() < deadline:
        owner.drain()
        if owner.process.poll() is not None:
            raise CaptureError("arm owner ended during rest window")
        time.sleep(0.05)
    end_ns = time.time_ns()
    event = command(owner, log, {"cmd": "status"}, "status")
    worker = event.get("worker") or {}
    if worker.get("fault") or worker.get("telemetry_dropped") or not worker.get("measured"):
        raise CaptureError(f"unhealthy rest window {label}")
    measured = worker["measured"]
    velocities = measured.get("velocities", [])
    if len(velocities) != 6 or any(not math.isfinite(v) or abs(v) > 0.075 for v in velocities):
        raise CaptureError(f"joint motion persisted in rest window {label}")
    if not (worker.get("contact") or {}).get("armed"):
        raise CaptureError(f"carriage baseline did not arm in rest window {label}")
    write_row(log, "rest", label=label, start_wall_ns=start_ns, end_wall_ns=end_ns,
              measured=measured, contact=worker.get("contact"),
              telemetry_written=event.get("telemetry_written"))


def prepare(run_dir: Path, tool: str) -> tuple[Path, Path, str, dict, list[float]]:
    if tool != "lutin-ballpoint-dot":
        raise CaptureError("this right-arm measurement expects the fitted ballpoint")
    if read_workspace(REPO).get("right", {}).get("tool_id") != tool:
        raise CaptureError("stated tool differs from the right arm workspace record")
    device = os.environ.get("TATBOT_ESTOP_DEVICE")
    if not device:
        raise CaptureError("run through the resolved arm hardware profile")
    device = recipe.native_estop_device(REPO, device)
    tool_sheet = REPO / "config/tools" / f"{tool}.yaml"
    profile = yaml.safe_load((REPO / "config/trossen/follower.yaml").read_text())
    transitions = [float(row["friction_transition_velocity"])
                   for row in profile["joint_characteristics"][:6]]
    binary = REPO / "rust/target/release/tatbot-arm-guide"
    identity = native_identity(binary)
    if hashlib.sha256(binary.read_bytes()).hexdigest() != identity["binary_sha256"]:
        raise CaptureError("native owner binary changed during preflight")
    (run_dir / "identity.json").write_text(json.dumps({"native": identity,
        "tool_datasheet_sha256": hashlib.sha256(tool_sheet.read_bytes()).hexdigest(),
        "golden_sha256": hashlib.sha256((REPO / "config/trossen/follower.yaml").read_bytes()).hexdigest(),
        "motion_constants_sha256": hashlib.sha256((REPO / "config/motion_constants.json").read_bytes()).hexdigest(),
        "transitions_rad_s": transitions}, indent=2))
    return binary, tool_sheet, device, identity, transitions


def resume_index(run_dir: Path, previous_id: str, tool_sheet: Path) -> int:
    if not re.fullmatch(r"[A-Za-z0-9_-]{8,100}", previous_id):
        raise CaptureError("invalid previous measurement run ID")
    previous = run_dir.parent / previous_id
    if not previous.is_dir() or previous.resolve().parent != run_dir.parent.resolve():
        raise CaptureError("previous measurement run is not local to this workflow")
    try:
        identity = json.loads((previous / "identity.json").read_text())
        rows = [json.loads(line) for line in (previous / "measurements.jsonl").read_text().splitlines()]
        events = [json.loads(line) for line in (previous / "owner/events.jsonl").read_text().splitlines()]
    except (OSError, ValueError) as error:
        raise CaptureError(f"previous measurement evidence unreadable: {error}") from error
    golden_sha = hashlib.sha256((REPO / "config/trossen/follower.yaml").read_bytes()).hexdigest()
    if identity.get("golden_sha256") != golden_sha or identity.get("tool_datasheet_sha256") != hashlib.sha256(tool_sheet.read_bytes()).hexdigest():
        raise CaptureError("previous measurement golden or tool identity differs")
    exits = [r for r in rows if r.get("kind") == "exit"]
    releases = [e for e in events if e.get("event") == "released"]
    if (not exits or exits[-1].get("owner_exit_code") != 0 or not exits[-1].get("released")
            or not releases or (releases[-1].get("measured") or {}).get("mode") != "Idle"):
        raise CaptureError("previous measurement has no verified idle release")
    carriage = [r for r in rows if r.get("kind") == "rest" and r.get("label", "").startswith("carriage-r")]
    if len(carriage) != 6:
        raise CaptureError("previous measurement did not complete six carriage direction rests")
    counts = {joint: sum(r.get("kind") == "return_drift" and r.get("joint_index") == joint
                         for r in rows) for joint in range(6)}
    for joint in range(5, -1, -1):
        if counts[joint] == 6:
            continue
        if counts[joint] > 6 or any(counts[lower] for lower in range(joint)):
            raise CaptureError("previous rotary measurements are not a sequential prefix")
        return joint
    raise CaptureError("previous run already completed all six rotary joints")


def skip_blocked_index(run_dir: Path, previous_id: str, blocked_id: str,
                       joint: int, tool_sheet: Path) -> int:
    if joint == 0:
        raise CaptureError("no lower rotary joint remains after the blocked joint")
    if not re.fullmatch(r"[A-Za-z0-9_-]{8,100}", blocked_id):
        raise CaptureError("invalid blocked measurement run ID")
    blocked = run_dir.parent / blocked_id
    if not blocked.is_dir() or blocked.resolve().parent != run_dir.parent.resolve():
        raise CaptureError("blocked measurement run is not local to this workflow")
    try:
        identity = json.loads((blocked / "identity.json").read_text())
        rows = [json.loads(line) for line in (blocked / "measurements.jsonl").read_text().splitlines()]
        events = [json.loads(line) for line in (blocked / "owner/events.jsonl").read_text().splitlines()]
    except (OSError, ValueError) as error:
        raise CaptureError(f"blocked measurement evidence unreadable: {error}") from error
    if (identity.get("golden_sha256") != hashlib.sha256((REPO / "config/trossen/follower.yaml").read_bytes()).hexdigest()
            or identity.get("tool_datasheet_sha256") != hashlib.sha256(tool_sheet.read_bytes()).hexdigest()):
        raise CaptureError("blocked measurement golden or tool identity differs")
    if not any(r.get("kind") == "resume" and r.get("previous_run") == previous_id
               and r.get("start_joint") == joint for r in rows):
        raise CaptureError("blocked run does not continue the stated prior joint prefix")
    faults = [e for e in events if e.get("event") == "fault"]
    if (not faults or "joint step tracking departed bound" not in faults[-1].get("reason", "")
            or f"joint_index: {joint}," not in faults[-1].get("refused_command", "")):
        raise CaptureError("blocked run does not prove a no-progress refusal at this joint")
    exits = [r for r in rows if r.get("kind") == "exit"]
    releases = [e for e in events if e.get("event") == "released"]
    if (not exits or exits[-1].get("owner_exit_code") != 0 or not exits[-1].get("released")
            or not releases or (releases[-1].get("measured") or {}).get("mode") != "Idle"):
        raise CaptureError("blocked run has no verified idle release")
    if any(r.get("kind") == "command" and r.get("command", {}).get("cmd") == "joint_step"
           and r["command"].get("joint_index", 6) < joint for r in rows):
        raise CaptureError("blocked run already commanded a lower joint")
    return joint - 1


def open_owner(run_dir: Path, binary: Path, tool_sheet: Path, device: str, identity: dict) -> Owner:
    owner_dir = run_dir / "owner"
    owner_dir.mkdir()
    owner = Owner([str(binary), "--run-dir", str(owner_dir), "--run-id", run_dir.name,
                   "--controller-role", "follower", "--backend", "trossen",
                   "--profile-dir", str(REPO / "config/trossen"),
                   "--tool-datasheet", str(tool_sheet), "--estop-device", device],
                  run_dir / "owner-client.log")
    try:
        ready = owner.wait("ready", timeout=30)
        if ready.get("build") != identity["build"]:
            raise CaptureError("native owner build changed at launch")
        if ready.get("tool_datasheet_sha256") != hashlib.sha256(tool_sheet.read_bytes()).hexdigest():
            raise CaptureError("fitted tool datasheet changed at launch")
        if ready.get("golden_sha256") != hashlib.sha256((REPO / "config/trossen/follower.yaml").read_bytes()).hexdigest():
            raise CaptureError("controller golden changed at launch")
    except (CaptureError, OSError):
        owner.close()
        raise
    return owner


def measure_carriage(owner: Owner, log: Path, carriage: float) -> None:
    # Compare rest at one interior position after opposite 0.2 mm arrivals.
    center = min(0.0314, max(0.0006, carriage))
    if abs(center - carriage) > 0.00005:
        carriage_to(owner, log, center)
    rest(owner, log, "carriage-center-initial")
    for repeat in range(3):
        for side in (-1, 1):
            carriage_to(owner, log, center + side * 0.0002)
            carriage_to(owner, log, center)
            rest(owner, log, f"carriage-r{repeat}-{side:+d}")


def measure_joints(owner: Owner, log: Path, q0: list[float], transitions: list[float],
                   start_joint: int) -> None:
    # Physical joint 6 to joint 1; only the selected joint is stepped.
    for joint in range(start_joint, -1, -1):
        speeds = [max(0.003, min(0.06, transitions[joint] * factor)) for factor in (0.5, 1, 1.5)]
        for band, speed in enumerate(speeds):
            for direction in (1, -1):
                step = {"cmd": "joint_step", "joint_index": joint,
                        "delta_rad": direction * 0.02, "speed_rad_s": speed}
                outward = command(owner, log, step, "joint_step", timeout=30)
                pair_seed = outward["before"]["joints"][joint]
                rest(owner, log, f"joint-{joint}-band-{band}-out-{direction:+d}", 1.5)
                step["delta_rad"] *= -1
                returned = command(owner, log, step, "joint_step", timeout=30)
                actual = returned["measured"]["joints"]
                if abs(actual[joint] - pair_seed) > 0.005:
                    raise CaptureError(f"joint {joint} did not return near its pair seed")
                drift = max(abs(q - seed) for q, seed in zip(actual, q0, strict=True))
                write_row(log, "return_drift", joint_index=joint, band=band,
                          direction=direction, pair_error_rad=actual[joint] - pair_seed,
                          max_drift_from_normalized_rad=drift)
                if drift > 0.03:
                    raise CaptureError("cumulative rotary drift exceeded 0.03 rad")
                rest(owner, log, f"joint-{joint}-band-{band}-return-{direction:+d}", 1.5)


def close_owner(owner: Owner, log: Path, released: bool) -> None:
    if not released and owner.process.poll() is None:
        try:
            command(owner, log, {"cmd": "release"}, "released", timeout=30)
            released = True
        except (CaptureError, OSError) as error:
            write_row(log, "release_error", reason=str(error))
    code = owner.close()
    write_row(log, "exit", owner_exit_code=code, released=released)
    if code != 0 or not released:
        raise CaptureError("arm idle release or owner exit unverified; inspect owner log")


def measure(run_dir: Path, tool: str, previous_id: str | None,
            blocked_id: str | None) -> None:
    binary, tool_sheet, device, identity, transitions = prepare(run_dir, tool)
    start_joint = resume_index(run_dir, previous_id, tool_sheet) if previous_id else 5
    if blocked_id:
        if not previous_id:
            raise CaptureError("skipping a blocked joint requires --resume-run")
        blocked_joint = start_joint
        start_joint = skip_blocked_index(run_dir, previous_id, blocked_id,
                                         blocked_joint, tool_sheet)
    log = run_dir / "measurements.jsonl"
    if previous_id:
        write_row(log, "resume", previous_run=previous_id, start_joint=start_joint,
                  carriage_from_previous=True)
    if blocked_id:
        write_row(log, "skipped_blocked_joint", blocked_run=blocked_id,
                  joint_index=blocked_joint, reason="verified no-progress refusal")
    owner = open_owner(run_dir, binary, tool_sheet, device, identity)
    released = False
    try:
        connected = command(owner, log, {"cmd": "connect"}, "connected", timeout=60)
        if not connected.get("carriage_qualified"):
            raise CaptureError("right-arm carriage is not qualified")
        start = connected["measured"]
        q0 = start["joints"]
        carriage = start["carriage"]["position_m"]
        # The controller reports a few encoder counts below nominal zero at
        # rest. Admit only a 0.5 mm feedback band around the 0..32 mm travel;
        # the first commanded target remains strictly inside that interval.
        if len(q0) != 6 or not -0.0005 <= carriage <= 0.0325:
            raise CaptureError("unexpected measured arm or carriage seed")
        rest(owner, log, "initial")
        if previous_id is None:
            measure_carriage(owner, log, carriage)
        normalized = command(owner, log, {"cmd": "normalize"}, "normalized", timeout=30)
        q0 = normalized["measured"]["joints"]
        rest(owner, log, "normalized")
        measure_joints(owner, log, q0, transitions, start_joint)
        command(owner, log, {"cmd": "release"}, "released", timeout=30)
        released = True
    finally:
        close_owner(owner, log, released)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ee-tool", required=True)
    parser.add_argument("--resume-run", help="completed local carriage/joint run ID to resume after")
    parser.add_argument("--skip-blocked-run", help="later linked run proving a no-progress joint refusal; continue at the next joint")
    args = parser.parse_args()
    try:
        with tatbot_runlog.init("calib-joint-measure", meta=vars(args)) as log:
            measure(log.dir, args.ee_tool, args.resume_run, args.skip_blocked_run)
            print(f"Joint measurement retained in {log.dir}", flush=True)
    except (CaptureError, OSError, ValueError, subprocess.CalledProcessError) as error:
        print(f"Joint measurement stopped: {error}", file=sys.stderr, flush=True)
        return 3
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
