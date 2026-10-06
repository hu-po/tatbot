"""Fast CPU kinematic probe for choosing a robot-compatible patch yaw."""

from __future__ import annotations

import hashlib
import json
import signal
import threading
from copy import deepcopy
from dataclasses import dataclass

import numpy as np
import torch

from tatbot_sim import interaction
from tatbot_sim.config import ARTWORK_HORIZON_STEPS, DRConfig
from tatbot_sim.expert import StrokeExpert
from tatbot_sim.inkmap.contracts import validate_scenario
from tatbot_sim.inkmap.mesh_patch_surface import mesh_patch_from_scenario
from tatbot_sim.inkmap.robot_clearance import (
    non_tool_clearance,
    tool_shaft_clearance,
    tool_terminal_exclusion_m,
)
from tatbot_sim.planning import plan_tattoo_scenario
from tatbot_sim.repo import repo_root
from tatbot_sim.tools import active_tool, carriage_rest_m, staged_pose

PROBE_TOLERANCE_M = 0.001
PATCH_YAW_CANDIDATES_RAD = (np.pi, 1.5 * np.pi, 0.0, 0.5 * np.pi)
PLACEMENT_SEARCH_SCHEMA = "tatbot.placement-search/1"
FULL_TRAJECTORY_HORIZON = ARTWORK_HORIZON_STEPS
PLACEMENT_SEARCH_PATH = repo_root() / "config" / "inkmap" / "placement-search.json"


def _placement_search_config() -> dict:
    config = json.loads(PLACEMENT_SEARCH_PATH.read_text())
    if config.get("schema") != "tatbot.placement-search-config/1":
        raise ReachAuditError(f"{PLACEMENT_SEARCH_PATH}: unsupported placement search schema")
    if not config.get("yaw_candidates_rad") or not config.get("offset_candidates"):
        raise ReachAuditError(f"{PLACEMENT_SEARCH_PATH}: search envelope is empty")
    return config


class ReachAuditError(ValueError):
    def __init__(self, message: str, reason: str = "placement_input"):
        super().__init__(message)
        self.reason = reason


@dataclass(frozen=True)
class PlacementSelection:
    scenario: dict
    patch_yaw_rad: float
    probe_max_residual_m: float
    candidates: tuple[dict, ...]
    audit: dict | None = None


def _yaw_variant(scenario: dict, center: np.ndarray, delta_rad: float) -> dict:
    if scenario.get("schema_version") == 3:
        return _typed_variant(scenario, delta_rad, (0, 0), (0, 0, 0))
    c, s = np.cos(delta_rad), np.sin(delta_rad)
    rotation = np.asarray([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
    transform = np.asarray(scenario["pose"]["world_from_body"], dtype=np.float64)
    transform[:3, :3] = rotation @ transform[:3, :3]
    transform[:3, 3] = center + rotation @ (transform[:3, 3] - center)
    result = deepcopy(scenario)
    result["pose"]["world_from_body"] = transform.tolist()
    validate_scenario(result)
    return result


def _typed_variant(scenario, delta_rad, xy, support):
    from tatbot_sim.human_rep.contracts import canonical_digest
    from tatbot_sim.inkmap.program_scenario import compile_simulation_bundle
    # Author a new bound request. Never mutate a compiled v3 realization.
    bundle = deepcopy(scenario["program_binding"]["bundle"])
    request = bundle["request"]
    request["patch_yaw_rad"] += float(delta_rad)
    request["target_world_m"] = [float(v) for v in np.asarray(request["target_world_m"]) + [*xy, 0]]
    request["support_offset_m"] = [float(v) for v in support]
    bundle["content_sha256"] = canonical_digest(bundle)
    ink = scenario["program_binding"]["ink_program"]
    return compile_simulation_bundle(bundle, created_at=scenario["provenance"]["created_at"],
                                     git_sha=scenario["provenance"]["git_sha"],
                                     operating_budget_s=ink.get("operating_budget_s"),
                                     initial_load_fraction=ink["initial_ink_state"]["load_fraction"])


def _placement_variant(
    scenario: dict,
    center: np.ndarray,
    delta_rad: float,
    target_offset_xy_m: tuple[float, float],
    support_offset_xyz_m: tuple[float, float, float],
) -> dict:
    if scenario.get("schema_version") == 3:
        return _typed_variant(scenario, delta_rad, target_offset_xy_m, support_offset_xyz_m)
    result = _yaw_variant(scenario, center, delta_rad)
    transform = np.asarray(result["pose"]["world_from_body"], dtype=np.float64)
    transform[:2, 3] += np.asarray(target_offset_xy_m, dtype=np.float64)
    result["pose"]["world_from_body"] = transform.tolist()
    support = np.eye(4)
    support[:3, 3] = np.asarray(support_offset_xyz_m, dtype=np.float64)
    result["support"]["world_from_nominal"] = support.tolist()
    validate_scenario(result)
    return result


class _Deadline:
    """A cooperative wall-clock bound around one expensive audit.

    The suite's own budget is checked between candidates, which is useless when
    a single candidate takes longer than the whole budget — at a 9,000-step
    horizon one full-trajectory audit can, and a measured run sat 20 minutes
    inside a 15-minute budget without ever reaching the check. SIGALRM lands
    between bytecodes, so it interrupts the audit's loop of small tensor
    operations promptly without weakening what the audit checks: a candidate
    that cannot be audited inside its share of the budget is refused with a
    reason, never silently trusted.

    Only the main thread can take the signal; elsewhere this is a no-op and the
    caller falls back to the coarser between-candidate check.
    """

    def __init__(self, seconds: float | None):
        self.seconds = seconds
        self._previous = None

    def __enter__(self):
        if not self.seconds or self.seconds <= 0:
            return self
        if threading.current_thread() is not threading.main_thread():
            return self
        self._previous = signal.signal(signal.SIGALRM, self._fire)
        signal.setitimer(signal.ITIMER_REAL, self.seconds)
        return self

    @staticmethod
    def _fire(_signum, _frame):
        raise ReachAuditError("audit exceeded its time budget", "time_budget")

    def __exit__(self, *_exc):
        if self._previous is not None:
            signal.setitimer(signal.ITIMER_REAL, 0)
            signal.signal(signal.SIGALRM, self._previous)
        return False


def _full_trajectory_residual(scenario: dict, trajectory_seed: int,
                              deadline_s: float | None = None) -> tuple[float, int, int]:
    """Solve the complete generator trajectory for one placed scenario.

    The probe above samples a few dozen targets, and a placement that passed
    it still failed ``generate`` on 185 of 5400 targets (2026-09-03): the
    misses sat between the probe's samples. So the candidate that wins the
    probe is re-checked the way the generator checks it — the same planner,
    the same ``StrokeExpert.reset`` sequential solve, FK of the joint
    reference against every target. Returns (worst residual m, targets over
    tolerance, targets).
    """
    with _Deadline(deadline_s):
        surface = mesh_patch_from_scenario(scenario)
        plan = plan_tattoo_scenario(
            np.random.default_rng(trajectory_seed),
            scenario,
            surface,
            horizon=FULL_TRAJECTORY_HORIZON,
            num_envs=1,
            dr=DRConfig(),
            draw_clearance=interaction.WORKING_OFFSET_M,
        )
        expert = StrokeExpert(1, torch.device("cpu"), noise=None, seed=trajectory_seed)
        names = expert.ik.chain.get_joint_parameter_names()
        staged = dict(zip((f"joint_{i}" for i in range(6)), staged_pose()[:6], strict=True))
        q0 = torch.tensor(
            [[staged.get(name, carriage_rest_m()) for name in names]], dtype=torch.float32,
        )
        q_start = expert.solve_pose(plan.targets[:, 0], q0, normals=plan.pen_normals[:, 0])
        expert.reset(
            plan.targets, q_start,
            floor_plane=(plan.surface_points, plan.surface_normals),
            pen_normals=plan.pen_normals,
        )
        if expert.q_ref is None:
            raise ReachAuditError("expert did not produce a joint reference", "ik_full")
        solved = expert.ik.fk(expert.q_ref.reshape(-1, len(names)))[:, :3, 3]
        desired = torch.as_tensor(plan.targets.reshape(-1, 3), dtype=solved.dtype)
        residual = torch.linalg.norm(solved - desired, dim=-1)
    return float(residual.max()), int((residual > PROBE_TOLERANCE_M).sum()), int(residual.numel())


def optimize_body_placement(
    scenario: dict,
    *,
    trajectory_seed: int,
    yaw_candidates: tuple[float, ...] | None = None,
    offset_candidates: tuple[
        tuple[tuple[float, float], tuple[float, float, float]], ...
    ] | None = None,
    probe_points: int | None = None,
    tolerance_m: float | None = None,
    non_tool_clearance_m: float | None = None,
    tool_shaft_clearance_m: float | None = None,
    audit_deadline_s: float | None = None,
) -> PlacementSelection:
    """Lexicographically select a reachable, collision-clear body placement.

    The named body pose never changes. The finite, versioned search moves the
    complete body in robot X/Y, rotates it about the tattoo patch, and permits
    a small fixture-only Y correction. Every candidate remains in the ledger.

    `audit_deadline_s` bounds the whole search, not just its last stage. The
    yaw probe and the offset sweep are the expensive part — wrapping only the
    full-trajectory re-check left a measured run 20 minutes past a 10-minute
    budget with the deadline never reached.
    """
    with _Deadline(audit_deadline_s):
        return _optimize_body_placement(
            scenario, trajectory_seed=trajectory_seed, yaw_candidates=yaw_candidates,
            offset_candidates=offset_candidates, probe_points=probe_points,
            tolerance_m=tolerance_m, non_tool_clearance_m=non_tool_clearance_m,
            tool_shaft_clearance_m=tool_shaft_clearance_m)


def _optimize_body_placement(
    scenario: dict,
    *,
    trajectory_seed: int,
    yaw_candidates: tuple[float, ...] | None = None,
    offset_candidates: tuple[
        tuple[tuple[float, float], tuple[float, float, float]], ...
    ] | None = None,
    probe_points: int | None = None,
    tolerance_m: float | None = None,
    non_tool_clearance_m: float | None = None,
    tool_shaft_clearance_m: float | None = None,
) -> PlacementSelection:
    validate_scenario(scenario)
    config = _placement_search_config()
    yaw_candidates = yaw_candidates or tuple(float(value) for value in config["yaw_candidates_rad"])
    if offset_candidates is None:
        offset_candidates = tuple(
            (
                (float(item["body_xy_m"][0]), float(item["body_xy_m"][1])),
                (
                    float(item["support_xyz_m"][0]),
                    float(item["support_xyz_m"][1]),
                    float(item["support_xyz_m"][2]),
                ),
            )
            for item in config["offset_candidates"]
        )
    probe_points = int(probe_points if probe_points is not None else config["probe_points"])
    tolerance_m = float(tolerance_m if tolerance_m is not None else config["ik_tolerance_m"])
    non_tool_clearance_m = float(
        non_tool_clearance_m
        if non_tool_clearance_m is not None
        else config["non_tool_clearance_m"]
    )
    tool_shaft_clearance_m = float(
        tool_shaft_clearance_m
        if tool_shaft_clearance_m is not None
        else config["tool_shaft_clearance_m"]
    )
    workspace_center = np.asarray(config["workspace_center_m"], dtype=np.float64)
    tool = active_tool()
    if tool.tool_id != scenario["robot"]["tool_id"]:
        raise ReachAuditError(
            f"placement optimizer needs {scenario['robot']['tool_id']!r}, "
            f"active tool is {tool.tool_id!r}",
        )
    if probe_points < 16:
        raise ReachAuditError("placement optimizer needs at least 16 trajectory samples")
    base_surface = mesh_patch_from_scenario(scenario)
    plan = plan_tattoo_scenario(
        np.random.default_rng(trajectory_seed),
        scenario,
        base_surface,
        horizon=FULL_TRAJECTORY_HORIZON,
        num_envs=1,
        dr=DRConfig(),
        draw_clearance=interaction.WORKING_OFFSET_M,
    )
    center = base_surface.origin_world_np()[0]
    indices = np.unique(np.linspace(0, plan.targets.shape[1] - 1, probe_points).astype(int))
    base_yaw = scenario.get("program_binding", {}).get("bundle", {}).get("request", {}).get("patch_yaw_rad", PATCH_YAW_CANDIDATES_RAD[0])
    specifications = [
        (yaw, xy, support)
        for xy, support in offset_candidates
        for yaw in yaw_candidates
    ]
    # Probe geometry is the already validated typed trace in the legacy
    # read-only geometry envelope. Recompiling identical paint for all search
    # candidates is unnecessary; the winner is rebound and fully validated
    # below before the complete trajectory gate or export.
    probe_source = scenario
    if scenario.get("schema_version") == 3:
        probe_source = deepcopy(scenario)
        del probe_source["program_binding"]
        probe_source["schema_version"] = 2
        probe_source["trace"]["compiler_version"] = 2
    targets, normals, variants = [], [], []
    for yaw, xy, support in specifications:
        delta = yaw - base_yaw
        c, s = np.cos(delta), np.sin(delta)
        rotation = np.asarray([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
        offset = np.asarray([xy[0], xy[1], 0.0])
        targets.append((plan.targets[0, indices] - center) @ rotation.T + center + offset)
        normals.append(plan.pen_normals[0, indices] @ rotation.T)
        variants.append(_placement_variant(probe_source, center, delta, xy, support))
    target_array = np.stack(targets)
    normal_array = np.stack(normals)
    expert = StrokeExpert(len(specifications), torch.device("cpu"), noise=None, seed=trajectory_seed)
    names = expert.ik.chain.get_joint_parameter_names()
    staged = dict(zip((f"joint_{i}" for i in range(6)), staged_pose()[:6], strict=True))
    q0 = torch.tensor(
        [[staged.get(name, carriage_rest_m()) for name in names]], dtype=torch.float32,
    ).repeat(len(specifications), 1)
    q_start = expert.solve_pose(target_array[:, 0], q0, normals=normal_array[:, 0], iters=400)
    target_tensor = torch.as_tensor(target_array.reshape(-1, 3), dtype=torch.float32)
    seed_tensor = (
        q_start[:, None, :].expand(-1, len(indices), -1).reshape(-1, len(names)).contiguous()
    )
    rotation_tensor = expert.target_rotations(normal_array.reshape(-1, 3), len(target_tensor))
    q = expert.ik.step(seed_tensor, target_tensor, rotation_tensor, iters=200)
    residual = torch.linalg.norm(expert.ik.fk(q)[:, :3, 3] - target_tensor, dim=-1)
    residual = residual.reshape(len(specifications), -1).detach().numpy()
    q_np = q.reshape(len(specifications), len(indices), -1).detach().numpy()
    records = []
    passing = []
    for index, ((yaw, xy, support), values, variant) in enumerate(
        zip(specifications, residual, variants, strict=True)
    ):
        record = {
            "candidate": index,
            "patch_yaw_rad": float(yaw),
            "target_offset_xy_m": [float(value) for value in xy],
            "support_offset_xyz_m": [float(value) for value in support],
            "max_residual_m": float(values.max()),
            "targets_over_tolerance": int((values > tolerance_m).sum()),
        }
        if record["max_residual_m"] > tolerance_m:
            record["accepted"] = False
            record["reason"] = "ik_probe"
            records.append(record)
            continue
        shaft = tool_shaft_clearance(q_np[index], variant)
        record["minimum_clearance_m"] = shaft
        if float(shaft["tool_shaft_m"]) < tool_shaft_clearance_m:
            record["accepted"] = False
            record["reason"] = "tool_shaft_clearance"
        else:
            clearances = {**non_tool_clearance(q_np[index], variant), **shaft}
            record["minimum_clearance_m"] = clearances
            if float(clearances["non_tool_robot_m"]) < non_tool_clearance_m:
                record["accepted"] = False
                record["reason"] = "robot_body_or_support_clearance"
                records.append(record)
                continue
            target_center = center + np.asarray([xy[0], xy[1], 0.0])
            record["accepted"] = True
            record["workspace_center_distance_m"] = float(np.linalg.norm(target_center - workspace_center))
            record["fixture_movement_m"] = float(np.linalg.norm(support))
            passing.append((index, record, variant))
        records.append(record)
    if not passing:
        counts: dict[str, int] = {}
        for record in records:
            reason = str(record["reason"])
            counts[reason] = counts.get(reason, 0) + 1
        summary = ", ".join(f"{key}={value}" for key, value in sorted(counts.items()))
        shaft_values = [
            item["minimum_clearance_m"]["tool_shaft_m"]
            for item in records
            if "minimum_clearance_m" in item
        ]
        best = f"; best shaft={max(shaft_values) * 1000:.1f} mm" if shaft_values else ""
        reason = "ik_probe" if set(counts) == {"ik_probe"} else "placement_clearance"
        raise ReachAuditError(
            f"no placement candidate passed {tolerance_m * 1000:.0f} mm IK, "
            f"{non_tool_clearance_m * 1000:.0f} mm robot, and "
            f"{tool_shaft_clearance_m * 1000:.0f} mm shaft gates ({summary}{best})",
            reason,
        )
    ranked = sorted(
        passing,
        key=lambda item: (
            -item[1]["minimum_clearance_m"]["non_tool_robot_m"],
            item[1]["workspace_center_distance_m"],
            item[1]["fixture_movement_m"],
            item[0],
        ),
    )
    chosen_index = chosen = result = None
    for index, record, variant in ranked:
        if scenario.get("schema_version") == 3:
            variant = _typed_variant(scenario, record["patch_yaw_rad"] - base_yaw,
                                     record["target_offset_xy_m"], record["support_offset_xyz_m"])
        worst, over, total = _full_trajectory_residual(variant, trajectory_seed)
        record["full_trajectory"] = {
            "max_residual_m": worst,
            "targets_over_tolerance": over,
            "targets": total,
            "horizon": FULL_TRAJECTORY_HORIZON,
        }
        if worst <= tolerance_m:
            chosen_index, chosen, result = index, record, variant
            break
        record["accepted"] = False
        record["reason"] = "ik_full"
    if result is None:
        detail = ", ".join(
            f"candidate {index}: {record['full_trajectory']['max_residual_m'] * 1000:.1f} mm "
            f"over {record['full_trajectory']['targets_over_tolerance']} targets"
            for index, record, _ in ranked
        )
        raise ReachAuditError(
            f"every probe-passing placement failed the full-trajectory "
            f"{tolerance_m * 1000:.0f} mm gate ({detail})",
            "ik_full",
        )
    assert chosen_index is not None and chosen is not None
    body_transform = result["pose"]["world_from_body"]
    support_transform = result["support"]["world_from_nominal"]
    audit = {
        "schema": PLACEMENT_SEARCH_SCHEMA,
        "search": {
            "probe_points": int(len(indices)),
            "full_trajectory_gate": True,
            "full_trajectory_horizon": FULL_TRAJECTORY_HORIZON,
            "ik_tolerance_m": float(tolerance_m),
            "non_tool_clearance_m": float(non_tool_clearance_m),
            "tool_shaft_clearance_m": float(tool_shaft_clearance_m),
            "tool_terminal_exclusion_m": tool_terminal_exclusion_m(tool),
            "workspace_center_m": workspace_center.tolist(),
            "configuration": str(PLACEMENT_SEARCH_PATH.relative_to(repo_root())),
            "configuration_sha256": hashlib.sha256(PLACEMENT_SEARCH_PATH.read_bytes()).hexdigest(),
        },
        "selected": {
            "candidate": int(chosen_index),
            "world_from_body": body_transform,
            "world_from_support": support_transform,
            "patch_yaw_rad": chosen["patch_yaw_rad"],
            "target_offset_xy_m": chosen["target_offset_xy_m"],
            "support_offset_xyz_m": chosen["support_offset_xyz_m"],
            "max_residual_m": chosen["max_residual_m"],
            "full_trajectory": chosen["full_trajectory"],
            "minimum_clearance_m": chosen["minimum_clearance_m"],
            "workspace_center_distance_m": chosen["workspace_center_distance_m"],
            "fixture_movement_m": chosen["fixture_movement_m"],
        },
        "candidates": records,
    }
    if result.get("schema_version") != 3:
        result["placement_optimization"] = audit
    validate_scenario(result)
    return PlacementSelection(
        scenario=result,
        patch_yaw_rad=float(chosen["patch_yaw_rad"]),
        probe_max_residual_m=float(chosen["max_residual_m"]),
        candidates=tuple(records),
        audit=audit,
    )


