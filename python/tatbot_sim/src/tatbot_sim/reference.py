"""Shared reference refinement and quality measurements for simulation clients.

These are synthetic-data checks, not physical robot motion prerequisites.
"""

from dataclasses import dataclass

import numpy as np
import torch

from tatbot_sim import interaction, tools

REACH_TOLERANCE_M = 0.001

CONTACT_REFERENCE_TOLERANCE_M = 0.5 * interaction.CONTACT_ABOVE_TOLERANCE_M

DIP_AXIS_TOLERANCE_RAD = np.radians(15.0)
"""How far the tool may lean off a cap's entry axis at the credited step.

Deliberately looser than anything on the sheet: the dip plan enters along the
palette normal (scripts/lib/dip_motion.py) while the simulator drives the tip
down and leaves the axis to IK, so this is a sanity bound on that difference rather than
a contact tolerance. Containment is the real gate — a cap is 8-15 mm across,
and a reference that misses it was never a dip.
"""

def _reference_contact_error_m(expert, plan, num_envs: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Per step, how far above its intended clearance the solved reference
    sits along the surface normal, and which steps are meant to touch.

    Returns (error (B, T) in metres, positive = too high; pen_down (B, T)).
    """
    q_reference = expert.q_ref
    if q_reference is None:
        raise RuntimeError("expert did not produce a joint reference")
    n_app = plan.n_app if plan.q_raised is not None else 0
    solved = expert.ik.fk(q_reference.reshape(-1, expert.ik.n_joints))[:, :3, 3].reshape(num_envs, -1, 3)[:, n_app:]
    device, dtype = solved.device, solved.dtype
    targets = torch.as_tensor(plan.targets, dtype=dtype, device=device)
    points = torch.as_tensor(plan.surface_points, dtype=dtype, device=device)
    normals = torch.as_tensor(plan.surface_normals, dtype=dtype, device=device)
    intended = ((targets - points) * normals).sum(-1)
    actual = ((solved - points) * normals).sum(-1)
    pen_down = intended <= interaction.CONTACT_ABOVE_TOLERANCE_M
    return actual - intended, pen_down


def _contact_worst_mm(error_np: np.ndarray, pen_np: np.ndarray) -> np.ndarray:
    """Worst pen-down contact error per env, in mm. Off-contact steps score 0."""
    return np.where(pen_np, np.abs(error_np), 0.0).max(axis=1) * 1000.0


def _settle_contact_reference(expert, plan, q_start, reset_kwargs: dict, num_envs: int,
                              rounds: int = 3) -> np.ndarray:
    """Re-target the reference along the surface normal until pen-down steps
    sit inside the contact band, then return the worst remaining error (mm).

    The damped IK trades position against the requested tool lean, and its
    converged reference sat 0.5-2 mm ABOVE the sheet on later strokes while
    passing the 1 mm residual gate (2026-09-03, seed 4200): the sim tip then
    hovered a hair outside the 0.5 mm contact band and never marked, and the
    dataset called that a demonstration. A residual gate cannot see it, so the
    reference is closed on contact instead: each round moves every offending
    target by its measured error and re-solves.
    """
    targets = np.array(plan.targets, dtype=np.float32, copy=True)
    normals = np.asarray(plan.surface_normals, dtype=np.float32)
    for _ in range(rounds):
        error, pen_down = _reference_contact_error_m(expert, plan, num_envs)
        error_np = error.cpu().numpy()
        pen_np = pen_down.cpu().numpy()
        offending = pen_np & (np.abs(error_np) > CONTACT_REFERENCE_TOLERANCE_M)
        if not offending.any():
            return _contact_worst_mm(error_np, pen_np)
        targets[offending] -= (error_np[offending, None] * normals[offending])
        expert.reset(targets, q_start, **reset_kwargs, resolve=True)
    # The last round re-solved and nothing has measured that reference yet.
    # Reporting the error it was asked to correct credited only `rounds - 1`
    # of the corrections: a batch fixed on the final round was still dropped as
    # `ik_reference`, and the number written to run_meta described a joint
    # reference that no longer existed.
    error, pen_down = _reference_contact_error_m(expert, plan, num_envs)
    return _contact_worst_mm(error.cpu().numpy(), pen_down.cpu().numpy())


def _reference_residual_mm(expert, plan, num_envs: int) -> np.ndarray:
    """Worst distance per env between the expert's solved joint reference and
    the targets it was asked to reach, in mm. The same FK the body-scenario
    gate used; now every distribution is held to it."""
    q_reference = expert.q_ref
    if q_reference is None:
        raise RuntimeError("expert did not produce a joint reference")
    n_app = plan.n_app if plan.q_raised is not None else 0
    solved = expert.ik.fk(q_reference.reshape(-1, expert.ik.n_joints))[:, :3, 3].reshape(num_envs, -1, 3)
    solved = solved[:, n_app:]
    desired = torch.as_tensor(plan.targets, dtype=solved.dtype, device=solved.device)
    residual = torch.linalg.norm(solved - desired, dim=-1)
    return (residual.max(dim=1).values * 1000.0).cpu().numpy()


def _dip_reference_error(expert, plan, cap_rims, num_envs: int, *, palette=None):
    """Per env, the worst dip in the solved reference: how far the tip sits
    from the cap's axis at the credited step, against that cap's own radius,
    and how far the tool leans off the cap's entry axis.

    The residual gate above is 1 mm against the commanded point, and it passes
    a dip that misses the cap completely — the commanded point is inside the
    cap, so a reference that lands 16 mm away still satisfies every other
    check. Nothing downstream notices either: env.set_dip_schedule credits the
    charge by step index and drains the cap without looking at the tip
    (measured on this bench 2026-09-09, every cap 5-25 mm out and 17-26 deg
    off). Envs with no dips score zero.
    """
    lateral_mm = np.zeros(num_envs)
    axis_deg = np.zeros(num_envs)
    outside = np.zeros(num_envs, dtype=bool)
    if not plan.dip_credits or cap_rims is None:
        return lateral_mm, axis_deg, outside
    q_reference = expert.q_ref
    if q_reference is None:
        raise RuntimeError("expert did not produce a joint reference")
    n_app = plan.n_app if plan.q_raised is not None else 0
    pose = expert.ik.fk(q_reference.reshape(-1, expert.ik.n_joints))
    tips = pose[:, :3, 3].reshape(num_envs, -1, 3)[:, n_app:].cpu().numpy().astype(np.float64)
    rots = pose[:, :3, :3].reshape(num_envs, -1, 3, 3)[:, n_app:].cpu().numpy().astype(np.float64)
    # the bore, not the outer envelope: a wall thickness of slack is a rim strike
    radii = {slot: spec.size.bore_diameter_m / 2.0 for slot, spec in (palette if palette is not None else tools.palette()).items()}
    for i in range(num_envs):
        steps = list(plan.dip_credits[i]) if i < len(plan.dip_credits) else []
        entries = list(plan.dips[i]) if plan.dips and i < len(plan.dips) else []
        for k, step in enumerate(steps):
            slot = entries[k]["slot"] if k < len(entries) else None
            if slot is None or slot not in radii or slot not in cap_rims:
                continue
            # the cap's own axis, which dipping.dip_segment records as the
            # floor normal for every step of the segment
            axis = np.asarray(plan.surface_normals[i, step], dtype=np.float64)
            axis = axis / max(np.linalg.norm(axis), 1e-12)
            delta = tips[i, step] - np.asarray(plan.targets[i, step], dtype=np.float64)
            lateral = float(np.linalg.norm(delta - np.dot(delta, axis) * axis))
            commanded = np.asarray(plan.pen_normals[i, step], dtype=np.float64)
            commanded = commanded / max(np.linalg.norm(commanded), 1e-12)
            local = expert.target_rotations(commanded[None, :], 1).cpu().numpy()[0].T @ commanded
            pointing = rots[i, step] @ local
            pointing /= max(np.linalg.norm(pointing), 1e-12)
            angle = float(np.arccos(np.clip(np.dot(pointing, commanded), -1.0, 1.0)))
            lateral_mm[i] = max(lateral_mm[i], lateral * 1000.0)
            axis_deg[i] = max(axis_deg[i], np.degrees(angle))
            if lateral >= radii[slot] or angle > DIP_AXIS_TOLERANCE_RAD:
                outside[i] = True
    return lateral_mm, axis_deg, outside


@dataclass(frozen=True)
class ReferenceQuality:
    residual_mm: np.ndarray
    contact_mm: np.ndarray
    dip_lateral_mm: np.ndarray
    dip_axis_deg: np.ndarray
    dip_missed: np.ndarray

    def metadata(self):
        return {name: getattr(self, name).tolist() for name in self.__dataclass_fields__}


def refine(expert, plan, q_start, reset_kwargs, *, num_envs, count, cap_rims, palette):
    """Use the same solve budget and final reference checks in every client."""
    kwargs = dict(reset_kwargs)
    residual = _reference_residual_mm(expert, plan, num_envs)
    if (residual[:count] > REACH_TOLERANCE_M * 1000).any():
        kwargs |= {"batch_iters": 240, "sweeps": 2, "sweep_iters": 12}
        expert.reset(plan.targets, q_start, **kwargs, resolve=True)
    contact = _settle_contact_reference(expert, plan, q_start, kwargs, num_envs)
    # Settling replaces the reference, so report the final solve everywhere.
    residual = _reference_residual_mm(expert, plan, num_envs)
    lateral, axis, missed = _dip_reference_error(expert, plan, cap_rims, num_envs, palette=palette)
    return ReferenceQuality(residual, contact, lateral, axis, missed)


def measure(expert, plan, *, cap_rims, palette):
    """Measure a native reference without changing any of its commands."""
    count = len(plan.targets)
    residual = _reference_residual_mm(expert, plan, count)
    error, down = _reference_contact_error_m(expert, plan, count)
    contact = _contact_worst_mm(error.cpu().numpy(), down.cpu().numpy())
    lateral, axis, missed = _dip_reference_error(expert, plan, cap_rims, count, palette=palette)
    return ReferenceQuality(residual, contact, lateral, axis, missed)
