"""Fast CPU kinematic probe for choosing a robot-compatible patch yaw."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass

import numpy as np
import torch

from tatbot_sim import interaction
from tatbot_sim.config import DRConfig
from tatbot_sim.expert import StrokeExpert
from tatbot_sim.inkmap.contracts import validate_scenario
from tatbot_sim.inkmap.mesh_patch_surface import mesh_patch_from_scenario
from tatbot_sim.planning import plan_tattoo_scenario
from tatbot_sim.tools import active_tool, carriage_rest_m, staged_pose

PROBE_TOLERANCE_M = 0.001
PROBE_POINTS = 160
PATCH_YAW_CANDIDATES_RAD = (np.pi, 1.5 * np.pi, 0.0, 0.5 * np.pi)


class ReachAuditError(ValueError):
    pass


@dataclass(frozen=True)
class ReachSelection:
    scenario: dict
    patch_yaw_rad: float
    probe_max_residual_m: float
    candidates: tuple[dict, ...]


def _yaw_variant(scenario: dict, center: np.ndarray, delta_rad: float) -> dict:
    c, s = np.cos(delta_rad), np.sin(delta_rad)
    rotation = np.asarray([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
    transform = np.asarray(scenario["pose"]["world_from_body"], dtype=np.float64)
    transform[:3, :3] = rotation @ transform[:3, :3]
    transform[:3, 3] = center + rotation @ (transform[:3, 3] - center)
    result = deepcopy(scenario)
    result["pose"]["world_from_body"] = transform.tolist()
    validate_scenario(result)
    return result


def select_reachable_patch_yaw(
    scenario: dict,
    *,
    trajectory_seed: int,
    yaw_candidates: tuple[float, ...] = PATCH_YAW_CANDIDATES_RAD,
    probe_points: int = PROBE_POINTS,
    tolerance_m: float = PROBE_TOLERANCE_M,
) -> ReachSelection:
    """Choose the first yaw whose subsampled exact trajectory passes 1 mm.

    The probe is deliberately a fast rejection gate: final dataset generation
    recomputes FK for every target and refuses any miss over the same limit.
    """
    validate_scenario(scenario)
    tool = active_tool()
    if tool.tool_id != scenario["robot"]["tool_id"]:
        raise ReachAuditError(
            f"reach probe needs {scenario['robot']['tool_id']!r}, active tool is {tool.tool_id!r}",
        )
    if probe_points < 16:
        raise ReachAuditError("reach probe needs at least 16 trajectory samples")
    surface = mesh_patch_from_scenario(scenario)
    plan = plan_tattoo_scenario(
        np.random.default_rng(trajectory_seed),
        scenario,
        surface,
        horizon=1800,
        num_envs=1,
        dr=DRConfig(),
        draw_clearance=interaction.WORKING_OFFSET_M,
    )
    center = surface.origin_world_np()[0]
    indices = np.unique(np.linspace(0, plan.targets.shape[1] - 1, probe_points).astype(int))
    base_yaw = PATCH_YAW_CANDIDATES_RAD[0]
    targets, normals = [], []
    for yaw in yaw_candidates:
        delta = yaw - base_yaw
        c, s = np.cos(delta), np.sin(delta)
        rotation = np.asarray([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
        targets.append((plan.targets[0, indices] - center) @ rotation.T + center)
        normals.append(plan.pen_normals[0, indices] @ rotation.T)
    target_array = np.stack(targets)
    normal_array = np.stack(normals)
    expert = StrokeExpert(len(yaw_candidates), torch.device("cpu"), noise=None, seed=trajectory_seed)
    names = expert.ik.chain.get_joint_parameter_names()
    staged = dict(zip((f"joint_{i}" for i in range(6)), staged_pose()[:6], strict=True))
    q0 = torch.tensor(
        [[staged.get(name, carriage_rest_m()) for name in names]], dtype=torch.float32,
    ).repeat(len(yaw_candidates), 1)
    q_start = expert.solve_pose(target_array[:, 0], q0, normals=normal_array[:, 0], iters=400)
    target_tensor = torch.as_tensor(target_array.reshape(-1, 3), dtype=torch.float32)
    seed_tensor = q_start[:, None, :].expand(-1, len(indices), -1).reshape(-1, 6).contiguous()
    rotation_tensor = expert.target_rotations(normal_array.reshape(-1, 3), len(target_tensor))
    q = expert.ik.step(seed_tensor, target_tensor, rotation_tensor, iters=200)
    residual = torch.linalg.norm(expert.ik.fk(q)[:, :3, 3] - target_tensor, dim=-1)
    residual = residual.reshape(len(yaw_candidates), -1).detach().numpy()
    records = tuple({
        "patch_yaw_rad": float(yaw),
        "max_residual_m": float(values.max()),
        "targets_over_tolerance": int((values > tolerance_m).sum()),
    } for yaw, values in zip(yaw_candidates, residual, strict=True))
    for index, record in enumerate(records):
        if record["max_residual_m"] <= tolerance_m:
            variant = _yaw_variant(scenario, center, yaw_candidates[index] - base_yaw)
            return ReachSelection(
                scenario=variant,
                patch_yaw_rad=float(yaw_candidates[index]),
                probe_max_residual_m=record["max_residual_m"],
                candidates=records,
            )
    summary = ", ".join(
        f"yaw={item['patch_yaw_rad']:.3f}:{item['max_residual_m'] * 1000:.1f}mm"
        for item in records
    )
    raise ReachAuditError(f"no patch yaw passed the {tolerance_m * 1000:.0f} mm probe ({summary})")
