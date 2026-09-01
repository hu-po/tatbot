"""Scripted stroke-following expert: batched damped-least-squares IK.

Emits native ``pd_joint_pos`` actions (7-dim: 6 arm joints + carriage), so
datasets need no control-mode conversion. IK runs on the serial chain
base->tattoo_needle (6 joints; the pen is welded to link_6 where the mount
sits at carriage rest, so the carriage is not an EE ancestor and is
commanded to its rest value — the safety layer's, never the policy's).

Motion is continuous by decision: the expert never idles, even though 60% of
real teleop control steps hold every joint perfectly still. Empty frames are
not worth generating — action chunking is expected to absorb the difference in
pacing — so do not "fix" this by adding pauses.

The whole reference trajectory is known before the episode runs, so IK is
solved once for every (env, timestep) as a single flat batch rather than once
per control step. Per-step solving was 57% of generation wall-clock: the
solve is tiny but launch-bound, and paying that 200x per episode dwarfed both
simulation and video encoding. Every timestep is seeded from the episode's
start pose and refined together, then swept sequentially so neighbouring
timesteps agree on an IK branch.

DART-style noise: decaying joint-space perturbation bursts are added on top of
the reference trajectory, so the commanded pose is knocked off the stroke and
then converges back to it — the recovery behaviour plain scripted replays lack.

The commanded trajectory is clamped so it never asks for the needle below the
pad surface, mirroring the follower's z-floor on the real rig. Nothing here
simulates contact (see the env's contact note), so before this clamp a noise
burst could command straight through the pad and the data taught exactly that:
the 99%-sim policy of 2026-08-21 drove ~40 mm through the real paper. The
clamp keeps the recovery behaviour that matters: bursts still pin the needle
onto the surface (below the nominal draw plane) and the decaying burst then
commands back up — a "too deep, come up" demonstration. States below the
*surface* disappear from the data entirely, and measurement says that is all
they were: in unclamped data every achieved-below step was downstream of a
commanded-below step, and on the real rig contact plus the follower's z-floor
make such states unreachable anyway. audit_depth.py checks both properties on
the written dataset.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pytorch_kinematics as pk
import torch

from tatbot_sim.agent import PEN_GRIP
from tatbot_sim.urdf import build_tatbot_urdf

if TYPE_CHECKING:  # config imports nothing from here; keep it that way
    from tatbot_sim.config import NoiseDR

URDF_PATH = build_tatbot_urdf()
EE_LINK = "tattoo_needle"

# How the real robot holds the pen, measured from teleop rather than derived
# from frames: hu-po/draw-square-fm2_20260831_121554 (new fixed EE, measured
# ballpoint tip, 2,827 drawing-phase steps through this same chain) shows the
# operator holds the BORE axis 2.9 deg from vertical and lets the crooked tip
# lean (tip axis 9.7 deg off), with the needle-frame x (link_6 z, the camera
# axis) at CAM_AXIS_WORLD below. The previous derivation held the TIP vector
# vertical with needle x -> world -y — that basis choice put the wrist a
# different IK branch from the real arm (sim joint_5 median -0.91 vs real
# +1.56) and, once the tip was measured crooked, shrank the reachable
# envelope to a corner the real robot does not live in.
#
# So the base orientation is now built from two facts, not a convention:
# the fitted tool's bore points straight down, and the camera axis points
# where the real recordings put it. For a straight tool (tip along the bore,
# e.g. the laser or an untouched-off 3RL) the bore IS the tip axis and only
# the roll differs from the old constant. Fit quality: geodesic distance to
# the real drawing frames p50/p95 = 3.6/7.5 deg, within the data's own
# spread about its mean (p95 4.8 deg).
# Per tool: the roll is a fact about how the arm carries THAT tool, and only
# the Lutin pen body has real new-EE recordings to fit it from. The fm2 roll
# is measured; the laser keeps the prior derived convention (needle x ->
# world -y), which its reach depends on — under the fm2 roll the 130 mm body
# cannot hold its bore vertical anywhere over the skin (13-20 mm residuals,
# multi-seed). Refit from the laser's own recordings when they exist.
_CAM_AXIS_BY_TOOL = {
    # ONLY the ballpoint: the fit is ballpoint data, and applying it to the
    # 3RL's nominal straight tip costs the pre-flight corner 6.1 mm — the
    # crooked measured seat is part of why the fm2 roll reaches. The 3RL gets
    # its own fit when it is touched off and recorded (operator-deferred).
    "lutin-ballpoint-dot": np.array([-0.9697, 0.1842, 0.1605]),
}
CAM_AXIS_WORLD = _CAM_AXIS_BY_TOOL.get(
    __import__("tatbot_sim.tools", fromlist=["active_tool"]).active_tool().tool_id,
    np.array([0.0, -1.0, 0.0]),
)
"""Needle-frame +x in world while drawing — fm2 mean for the pen body."""


def _pen_down_matrix() -> torch.Tensor:
    """(3, 3) base pen-down orientation: bore down, cameras as recorded.

    Built as a pair of orthonormal triads so the bore constraint is exact and
    the camera axis takes whatever component of CAM_AXIS_WORLD is left
    perpendicular to it."""
    from tatbot_sim.urdf import tool_tcp_m

    reg = __import__("tatbot_sim.tools", fromlist=["registry"]).registry()
    tcp = np.asarray(tool_tcp_m(), dtype=np.float64)
    _, pitch, yaw = reg.axis_rpy(tcp)
    cy, sy, cp, sp = np.cos(yaw), np.sin(yaw), np.cos(pitch), np.sin(pitch)
    r_mp = (np.array([[cy, -sy, 0.0], [sy, cy, 0.0], [0.0, 0.0, 1.0]])
            @ np.array([[cp, 0.0, sp], [0.0, 1.0, 0.0], [-sp, 0.0, cp]]))
    bore_in_needle = r_mp.T @ np.array([0.0, 0.0, 1.0])

    def triad(primary, secondary_hint):
        a1 = primary / np.linalg.norm(primary)
        a2 = secondary_hint - (secondary_hint @ a1) * a1
        a2 /= np.linalg.norm(a2)
        return np.stack([a1, a2, np.cross(a1, a2)], axis=1)

    needle = triad(bore_in_needle, np.array([1.0, 0.0, 0.0]))
    world = triad(np.array([0.0, 0.0, -1.0]), CAM_AXIS_WORLD)
    return torch.tensor(world @ needle.T, dtype=torch.float32)


PEN_DOWN = _pen_down_matrix()


class BatchedIK:
    """Damped least-squares IK over a pytorch_kinematics serial chain."""

    def __init__(self, device: torch.device, damping: float = 0.05, ori_weight: float = 0.3):
        with open(URDF_PATH, "rb") as f:
            urdf = f.read()
        self.chain = pk.build_serial_chain_from_urdf(urdf, end_link_name=EE_LINK).to(
            device=device, dtype=torch.float32
        )
        lim = self.chain.get_joint_limits()
        self.q_lo = torch.as_tensor(lim[0], dtype=torch.float32, device=device)
        self.q_hi = torch.as_tensor(lim[1], dtype=torch.float32, device=device)
        self.damping = damping
        self.ori_weight = ori_weight
        self.device = device
        self.n_joints = len(self.chain.get_joint_parameter_names())

    def fk(self, q: torch.Tensor) -> torch.Tensor:
        """(B, n) joints -> (B, 4, 4) EE pose in base frame."""
        return self.chain.forward_kinematics(q).get_matrix()

    def step(
        self, q: torch.Tensor, target_pos: torch.Tensor, target_rot: torch.Tensor, iters: int = 3
    ) -> torch.Tensor:
        """Iterate DLS from ``q`` toward (target_pos (B,3), target_rot (B,3,3))."""
        for _ in range(iters):
            mat = self.fk(q)
            pos_err = target_pos - mat[:, :3, 3]
            r_cur = mat[:, :3, :3]
            # orientation error: 0.5 * sum of cross products of frame axes
            rot_err = 0.5 * (
                torch.cross(r_cur[:, :, 0], target_rot[:, :, 0], dim=1)
                + torch.cross(r_cur[:, :, 1], target_rot[:, :, 1], dim=1)
                + torch.cross(r_cur[:, :, 2], target_rot[:, :, 2], dim=1)
            )
            twist = torch.cat([pos_err, self.ori_weight * rot_err], dim=1)  # (B, 6)
            jac = self.chain.jacobian(q)  # (B, 6, n)
            jjt = jac @ jac.transpose(1, 2)
            jjt += (self.damping**2) * torch.eye(6, device=self.device)
            dq = jac.transpose(1, 2) @ torch.linalg.solve(jjt, twist.unsqueeze(-1))
            q = torch.clamp(q + dq.squeeze(-1), self.q_lo, self.q_hi)
        return q


class ReachMask:
    """Where on a canvas the fitted tool can be held normal to the surface.

    A mound is not uniformly workable: its summit is locally flat and its
    margins lie flat, but the flanks between them ask the wrist for a lean it
    cannot make while a 130 mm tool is in the gripper. A scalar reach radius
    cannot say that -- the reachable set is not a disc -- so this is a coarse
    boolean over canvas coordinates, sampled from the surface the episode will
    actually use.

    Nearest-node lookup, and deliberately coarse: it decides where a stroke may
    be PLACED, and a stroke placed a few millimetres inside a boundary the IK
    was going to miss anyway is not a distinction worth the samples.
    """

    def __init__(self, mask: np.ndarray, width_m: float, height_m: float):
        self.mask = np.asarray(mask, dtype=bool)
        self.rows, self.cols = self.mask.shape
        self.width_m, self.height_m = float(width_m), float(height_m)

    @property
    def fraction(self) -> float:
        return float(self.mask.mean())

    def _index(self, xy: np.ndarray):
        xy = np.asarray(xy, dtype=np.float64).reshape(-1, 2)
        j = np.rint((xy[:, 0] + self.width_m / 2) / self.width_m * (self.cols - 1))
        i = np.rint((xy[:, 1] + self.height_m / 2) / self.height_m * (self.rows - 1))
        return (np.clip(i, 0, self.rows - 1).astype(int),
                np.clip(j, 0, self.cols - 1).astype(int))

    def ok(self, xy) -> bool:
        """True when EVERY point is somewhere the tool can be held normal."""
        i, j = self._index(xy)
        return bool(self.mask[i, j].all())

    def node_ok(self, x: float, y: float) -> bool:
        i, j = self._index([[x, y]])
        return bool(self.mask[i[0], j[0]])


def reachable_canvas_masks(expert, q0_arm, surface, clearance: float, num_envs: int,
                           cols: int = 21, rows: int = 27, tol: float = 0.001,
                           iters: int = 120, max_off_base_rad: float = 0.0) -> list[ReachMask]:
    """Ask the IK where, on each env's canvas, the tool can work at all.

    The tool is held along the LOCAL normal, which is the whole point: a pose
    the arm reaches pointing straight down can be unreachable pointing thirty
    degrees off it, and the flanks of a mound are exactly that case.

    ``max_off_base_rad`` must match what the PLANNER will do. The map has to
    answer the question the episode will actually ask: if the tool is going to
    be held no further than twenty degrees off the pad, asking whether it can
    be held at thirty-five condemns ground that would have worked.

    Every env solves in ONE batched call. The IK is launch-bound, so a loop
    over environments costs several seconds a batch for the same answer.

    Checked 2026-08-31, after the reach pre-flight turned out to be
    basin-bound under the fixed-EE staged pose (see reach_residual_at):
    these masks are NOT. Re-solving every failed grid point with the
    pre-flight's perturbed-seed retry converted 0 of 725 failures across
    {3RL, laser} x {seed 3, 11} on the draped skin — median leftover
    residual 5-6 mm, i.e. genuine lean-bound flank points, which is what
    the warm start from the canvas centre exists to buy. Do not add a
    retry ladder here without a measurement that says otherwise, and do
    not read a low fraction as a solver bug: fractions swing 74-94% with
    the sampled surface draw alone, so masks are only comparable on the
    same draw.
    """
    us = np.linspace(-surface.width_m / 2, surface.width_m / 2, cols)
    vs = np.linspace(-surface.height_m / 2, surface.height_m / 2, rows)
    vv, uu = np.meshgrid(vs, us, indexing="ij")
    uv = np.stack([uu.ravel(), vv.ravel()], 1).astype(np.float32)
    n = len(uv)

    pts, nrm, seeds, seed_axes = [], [], [], []
    for i in range(num_envs):
        p_i, n_i = surface.frame_np(i, uv)
        pts.append(p_i + clearance * n_i)
        nrm.append(n_i)
        # Warm-started the way an episode starts, from this env's canvas centre
        # held along its average normal: seeding from rest would measure the
        # solver's basin of attraction rather than the arm's reach.
        seeds.append(np.repeat((p_i.mean(0) + clearance * n_i.mean(0))[None], n, 0))
        seed_axes.append(np.repeat(n_i.mean(0)[None], n, 0))
    if max_off_base_rad > 0:
        from tatbot_sim.planning import cap_lean
        nrm = [cap_lean(a.astype(np.float64), surface.base_normal_np(i), max_off_base_rad)
               for i, a in enumerate(nrm)]

    tgt = torch.as_tensor(np.concatenate(pts), dtype=torch.float32, device=expert.device)
    axes = np.concatenate(nrm).astype(np.float64)
    total = len(tgt)
    q = q0_arm[:1].expand(total, 6).contiguous().to(expert.device)
    q = expert.ik.step(
        q, torch.as_tensor(np.concatenate(seeds), dtype=torch.float32, device=expert.device),
        expert.target_rotations(np.concatenate(seed_axes), total), iters=iters,
    )
    q = expert.ik.step(q, tgt, expert.target_rotations(axes, total), iters=iters)
    res = torch.linalg.norm(expert.ik.fk(q)[:, :3, 3] - tgt, dim=-1).cpu().numpy()
    ok = (res <= tol).reshape(num_envs, rows, cols)
    return [ReachMask(ok[i], surface.width_m, surface.height_m) for i in range(num_envs)]


def reachable_height_ceiling(expert, q0_arm, surface, num_envs: int,
                             candidates=(0.006, 0.012, 0.020, 0.030, 0.045, 0.060),
                             keep: float = 0.85, max_off_base_rad: float = 0.0,
                             **kw) -> float:
    """Highest the tool can be held above the surface and still work most of it.

    Drawing happens a few millimetres off the skin, but a trajectory also
    HOVERS between strokes and STARTS well clear of the surface, and those are
    the poses a mound makes impossible: height is exactly what the arm is short
    of. Measured over a 25 mm mound, the laser reaches all of a skin at 4 mm
    and none of it at 50 -- so an episode that starts at 50 asks for a pose the
    arm cannot make, the sequential solve walks forward from that bad answer,
    and everything after it inherits the miss.

    Returns the tallest candidate that still holds ``keep`` of the ground the
    drawing height itself can reach. It is a ladder rather than a bisection
    because the answer only has to be right to the nearest few millimetres and
    each rung costs an IK batch.
    """
    def frac(c):
        ms = reachable_canvas_masks(expert, q0_arm, surface, c, num_envs,
                                    max_off_base_rad=max_off_base_rad, **kw)
        return float(np.mean([m.fraction for m in ms]))

    base = frac(candidates[0])
    if base <= 0.0:
        return candidates[0]
    for c in reversed(candidates[1:]):
        if frac(c) >= keep * base:
            return float(c)
    return float(candidates[0])


def _per_step(v: torch.Tensor, b: int, t_len: int) -> torch.Tensor:
    """(B, 3) or (B, T, 3) -> (B, t_len, 3).

    One plane per episode is broadcast; a plane per step is padded at the FRONT
    with its first entry, because the only thing prepended to a trajectory is
    the approach, and that descends toward the first drawing pose.
    """
    if v.ndim == 2:
        return v.unsqueeze(1).expand(b, t_len, 3)
    if v.shape[1] == t_len:
        return v
    if v.shape[1] > t_len:
        raise ValueError(f"floor plane has {v.shape[1]} steps for a {t_len}-step trajectory")
    head = v[:, :1].expand(b, t_len - v.shape[1], 3)
    return torch.cat([head, v], dim=1)


class StrokeExpert:
    """Tracks per-env EE target trajectories, emitting 7-dim joint-pos actions."""

    def __init__(
        self,
        num_envs: int,
        device: torch.device,
        pen_grip: float = PEN_GRIP,
        noise: "NoiseDR | None" = None,
        seed: int | None = None,
    ):
        from tatbot_sim.config import NoiseDR

        self.ik = BatchedIK(device)
        self.num_envs = num_envs
        self.device = device
        self.pen_grip = pen_grip
        self.noise = noise or NoiseDR()
        # one stream for the whole run: re-seeding per batch replayed the
        # SAME burst timing in every batch of a dataset
        self._nrng = np.random.default_rng(seed)
        self.targets: torch.Tensor | None = None  # (B, T, 3) world-frame EE positions
        self.q_ref: torch.Tensor | None = None  # (B, T, 6) solved joint reference
        self.t = 0
        self.clamped_fraction = 0.0  # of the last reset's steps, how many hit the floor

    def target_rotations(self, normals: np.ndarray | None, batch: int) -> torch.Tensor:
        """(N, 3, 3) pen-down orientations along the given axis directions.

        ``normals`` are pen-axis directions, one per row — per env, or per
        (env, timestep) flattened; None means level. The base pen-down pose is
        tilted by the minimal rotation taking world +z onto each direction, so
        the wrist twist (camera orientation) stays put while the pen leans.
        """
        base = PEN_DOWN.to(self.device)
        if normals is None:
            return base.unsqueeze(0).expand(batch, 3, 3).contiguous()
        align = torch.as_tensor(
            _rotations_z_to(np.asarray(normals, dtype=np.float64)),
            dtype=torch.float32, device=self.device,
        )
        return (align @ base.unsqueeze(0)).contiguous()

    def solve_pose(
        self,
        targets_world: np.ndarray,
        q0_arm: torch.Tensor,
        normals: np.ndarray | None = None,
        iters: int = 300,
    ):
        """Solve joints for a single (B, 3) target — used to place the arm at the
        start of an episode, since the surface pose varies per environment."""
        t = torch.as_tensor(targets_world, dtype=torch.float32, device=self.device)
        rot_b = self.target_rotations(normals, t.shape[0])
        return self.ik.step(q0_arm.clone(), t, rot_b, iters=iters)

    def reset(
        self,
        targets_world: np.ndarray,
        q0_arm: torch.Tensor,
        floor_plane: tuple[np.ndarray, np.ndarray] | None = None,
        pen_normals: np.ndarray | None = None,
        approach_from: tuple[np.ndarray, int] | None = None,
        batch_iters: int = 60,
        sweeps: int = 1,
        sweep_iters: int = 4,
    ):
        """Solve the whole episode's joint trajectory. targets_world: (B, T, 3).

        ``approach_from`` is (q_raised (B,6), steps): prepend a joint-space
        min-jerk descent from a raised pose (the robot's staged position on
        the real rig) to the first drawing pose. Real sessions record this
        arc whenever an episode starts before the arm is down — sim covers
        it the same way, in joint space, exactly like the hardware moves.
        ``floor_plane`` is (points (B,3), normals (B,3)) — each environment's
        pad surface; when given, no commanded step may put the needle on the
        far side of it (see module docstring). The pen is held along
        ``pen_normals`` when given — (B,3) for a constant lean or (B,T,3) for
        a lean that evolves over the path (the controlled, continuous handle
        that flicks and stipple will drive) — else perpendicular to the floor
        plane.
        """
        targets = torch.as_tensor(targets_world, dtype=torch.float32, device=self.device)
        b, t_len, _ = targets.shape
        if pen_normals is not None:
            pn = np.asarray(pen_normals, dtype=np.float64)
            if pn.ndim == 2:
                pn = np.repeat(pn[:, None, :], t_len, axis=1)
            assert pn.shape == (b, t_len, 3), pn.shape
            rot_seq = self.target_rotations(pn.reshape(-1, 3), b * t_len).reshape(
                b, t_len, 3, 3
            )
        else:
            normals = floor_plane[1] if floor_plane is not None else None
            if normals is not None and np.asarray(normals).ndim == 3:
                # a shaped surface supplies a normal per step; holding the tool
                # perpendicular to the first one for the whole path would lean
                # it further off the skin the further it travelled
                nrm = np.asarray(normals, dtype=np.float64)
                rot_seq = self.target_rotations(nrm.reshape(-1, 3), b * t_len).reshape(
                    b, t_len, 3, 3
                )
            else:
                rot_seq = (
                    self.target_rotations(normals, b)
                    .unsqueeze(1)
                    .expand(b, t_len, 3, 3)
                    .contiguous()
                )

        flat_tgt = targets.reshape(b * t_len, 3)
        flat_rot = rot_seq.reshape(b * t_len, 3, 3).contiguous()
        q = q0_arm.unsqueeze(1).expand(b, t_len, 6).reshape(b * t_len, 6).clone()
        q = self.ik.step(q, flat_tgt, flat_rot, iters=batch_iters)
        q = q.reshape(b, t_len, 6)

        # Sequential sweeps: re-seed each timestep from its predecessor's
        # solution so adjacent steps settle into the same IK branch and the
        # commanded trajectory stays continuous. Seed from the batch solution
        # rather than the start pose, and refine with several iterations — a
        # single iteration per step cannot track the trajectory and silently
        # replaces the converged batch solve with a worse one.
        # One sweep suffices: across language and maze batches a second
        # sweep reproduces the first to float32 noise (max 5e-7 rad,
        # measured 2026-08-25) while costing 8-17 s per batch on the
        # generation node — the loop is launch-bound, not compute-bound.
        for _ in range(sweeps):
            prev = q[:, 0]
            cols = []
            for i in range(t_len):
                prev = self.ik.step(prev, targets[:, i], rot_seq[:, i], iters=sweep_iters)
                cols.append(prev)
            q = torch.stack(cols, dim=1)

        # Precompute the full action tensor. act() is called inside the hot
        # loop while dozens of encode threads compete for the CPU, so any
        # host-side RNG or host->device copy there is disproportionately
        # expensive; doing it once per batch keeps the loop to one slice.
        # Burst frequency and size draw per env per batch from the NoiseDR
        # ranges, so episodes span near-clean to moderately perturbed.
        nrng = self._nrng
        n_prob = nrng.uniform(*self.noise.prob, b).astype(np.float32)[:, None]
        n_scale = nrng.uniform(*self.noise.scale, b).astype(np.float32)[:, None]
        noise = np.zeros((b, t_len, 6), dtype=np.float32)
        cur = np.zeros((b, 6), dtype=np.float32)
        for i in range(t_len):
            fires = (nrng.random((b, 1)) < n_prob).astype(np.float32)
            burst = nrng.standard_normal((b, 6)).astype(np.float32) * n_scale
            cur = cur * self.noise.decay + burst * fires
            noise[:, i] = cur
        noise_t = torch.as_tensor(noise, device=self.device)

        if approach_from is not None:
            q_raised, n_app = approach_from
            qr = torch.as_tensor(q_raised, dtype=torch.float32, device=self.device)
            u = torch.linspace(0, 1, n_app + 1, device=self.device)[:-1]
            blend = (10 * u**3 - 15 * u**4 + 6 * u**5).view(1, -1, 1)  # min-jerk
            seg = qr.unsqueeze(1) + (q[:, :1] - qr.unsqueeze(1)) * blend
            q = torch.cat([seg, q], dim=1)
            t_len += n_app
            # extend the noise stream over the approach with the same process
            extra = np.zeros((b, n_app, 6), dtype=np.float32)
            cur = np.zeros((b, 6), dtype=np.float32)
            for i in range(n_app):
                fires = (nrng.random((b, 1)) < n_prob).astype(np.float32)
                burst = nrng.standard_normal((b, 6)).astype(np.float32) * n_scale
                cur = cur * self.noise.decay + burst * fires
                extra[:, i] = cur
            noise_t = torch.cat([torch.as_tensor(extra, device=self.device), noise_t], dim=1)

        q_cmd = torch.clamp(q + noise_t, self.ik.q_lo, self.ik.q_hi)
        self.clamped_fraction = 0.0
        if floor_plane is not None:
            pts, nms = floor_plane
            pt_t = torch.as_tensor(pts, dtype=torch.float32, device=self.device)
            nm_t = torch.as_tensor(nms, dtype=torch.float32, device=self.device)
            # Per-step floor. On and near the stroke (reference inside the
            # 5.5 mm ink-deposit band) it is the surface itself: noise may
            # press the pen down to the pad and recover — the demonstrations
            # that matter. Anywhere the reference is clear of the band —
            # travel, hover, and the upper part of the descend ramp — the
            # band is off-limits: a burst pressing through it stamps a stray
            # disconnected mark on the sheet that the recorded path never
            # explains (measured ~1 per episode before this clamp). Ramp
            # steps whose reference sits between the band and the floor keep
            # the reference itself (fraction-0 fallback), which never inks.
            t_draw = targets.shape[1]
            pt_t = _per_step(pt_t, b, t_draw)
            nm_t = _per_step(nm_t, b, t_draw)
            ref_dist = ((targets - pt_t) * nm_t).sum(-1)
            offset = torch.where(
                ref_dist > 0.006, torch.full_like(ref_dist, 0.0075),
                torch.zeros_like(ref_dist),
            )
            if approach_from is not None:
                # the approach descends from the raised pose, always well clear
                offset = torch.cat(
                    [torch.full((b, approach_from[1]), 0.0075, device=self.device), offset],
                    dim=1,
                )
            q_cmd = self._clamp_to_floor(q, q_cmd, pt_t, nm_t, offset)
        grip = torch.full((b, t_len, 1), self.pen_grip, device=self.device)
        self.actions = torch.cat([q_cmd, grip], dim=2)  # (B, T, 7)

        self.q_ref = q
        self.targets = targets
        self.t = 0

    def _clamp_to_floor(
        self, q_ref: torch.Tensor, q_cmd: torch.Tensor,
        pt: torch.Tensor, nm: torch.Tensor, offset: torch.Tensor,
    ) -> torch.Tensor:
        """Scale noise back wherever it commands the needle below the per-step
        floor: ``offset`` metres above the surface's own tangent plane AT THAT
        STEP (0 = the surface itself; see reset for how the travel steps raise
        it above the ink band). ``pt``/``nm`` are (B, 3) for one plane per
        episode or (B, T, 3) for a plane per step, which is what a surface
        that is not flat needs. Offending steps are those whose commanded needle sits below
        that floor along the plane's outward normal. For each, bisection finds
        the largest fraction of the noise that stays legal — distance is not
        linear in joint angles. The reference itself respects every floor it
        is checked against, so fraction 0 is always safe; steps already legal
        are untouched.
        """
        b, t_len, _ = q_cmd.shape
        pt_flat = _per_step(pt, b, t_len).reshape(-1, 3)
        nm_flat = _per_step(nm, b, t_len).reshape(-1, 3)
        off_flat = offset.reshape(-1)

        flat_ref = q_ref.reshape(-1, 6)
        flat_cmd = q_cmd.reshape(-1, 6).clone()
        pos = self.ik.fk(flat_cmd)[:, :3, 3]
        below = ((pos - pt_flat) * nm_flat).sum(-1) < off_flat
        self.clamped_fraction = float(below.float().mean())
        if not bool(below.any()):
            return q_cmd
        ref = flat_ref[below]
        delta = flat_cmd[below] - ref
        pt_b, nm_b, off_b = pt_flat[below], nm_flat[below], off_flat[below]
        lo = torch.zeros(len(ref), device=self.device)
        hi = torch.ones_like(lo)
        for _ in range(12):
            mid = (lo + hi) / 2
            pos = self.ik.fk(ref + mid.unsqueeze(1) * delta)[:, :3, 3]
            ok = ((pos - pt_b) * nm_b).sum(-1) >= off_b
            lo = torch.where(ok, mid, lo)
            hi = torch.where(ok, hi, mid)
        # both endpoints sit inside the joint-limit box, so the blend does too
        flat_cmd[below] = ref + lo.unsqueeze(1) * delta
        return flat_cmd.reshape(b, t_len, 6)

    @property
    def horizon(self) -> int:
        return self.targets.shape[1]

    def act(self) -> torch.Tensor:
        """Return the next (B, 7) pd_joint_pos action."""
        t = min(self.t, self.horizon - 1)
        self.t += 1
        return self.actions[:, t]


def reach_residual_at(
    expert: "StrokeExpert",
    q_rest: torch.Tensor,
    pad_center: np.ndarray,
    top_z: float,
    draw_clearance: float,
    reach: float = 0.06,
    retries: int = 8,
    good_enough_m: float = 5e-4,
) -> float:
    """Worst IK residual (m) across one pad height, centre to reach limit.

    Each target is solved from ``q_rest`` and, when that misses, from
    ``retries`` deterministic perturbations of it, keeping the best. A DLS
    solve is basin-bound, not workspace-bound: the 2026-08-31 fixed-EE
    validation measured 73-152 mm "residuals" at pad targets that solve to
    0.0 mm from a nudged seed — the staged pose the EE change moved (wrist
    +pi/2) seeds a bad basin for the short tools, and this gate refused two
    distributions the arm reaches fine. Best-of-seeds stays sound because a
    low residual only ever *proves* reachability: ``BatchedIK.step`` clamps
    every iterate to joint limits, so a converged retry is as feasible as a
    converged first try. The perturbations are drawn from a fixed generator,
    so the gate's verdict cannot flap between runs.
    """
    normal = np.array([[0.0, 0.0, 1.0]])
    rot = expert.target_rotations(normal, 1)
    gen = torch.Generator().manual_seed(0)
    worst = 0.0
    for off in (0.0, reach):
        tgt = torch.tensor(
            [[pad_center[0] + off, 0.0, top_z + draw_clearance]],
            dtype=torch.float32, device=expert.device,
        )
        best = float("inf")
        for attempt in range(retries + 1):
            seed = q_rest[:1].clone()
            if attempt:
                seed += (torch.randn(seed.shape, generator=gen) * 0.3).to(expert.device)
            q = expert.ik.step(seed, tgt, rot, iters=400)
            best = min(best, float(torch.linalg.norm(expert.ik.fk(q)[:, :3, 3] - tgt, dim=-1)))
            if best <= good_enough_m:
                break
        worst = max(worst, best)
    return worst


def worst_reach_residual(
    expert: "StrokeExpert",
    q_rest: torch.Tensor,
    pad_center: np.ndarray,
    z_range: tuple[float, float],
    draw_clearance: float,
    reach: float = 0.06,
) -> tuple[float, float]:
    """Worst IK residual over the sampled drawing envelope, and where.

    Returns (residual_m, pad_top_z). MAX_TOOL_Z_CENTER encodes this ceiling
    for the tattoo pen as a measured constant, but it is a per-TOOL fact and a
    long tool has a much lower one: a tool the arm cannot hold perpendicular
    does not fail loudly, it returns a best-effort pose tens of millimetres
    away and the episode quietly marks the wrong place. Probing the corners of
    the envelope before generating is cheap and turns that into an error.
    """
    worst, worst_z = 0.0, float(z_range[0])
    for top_z in (float(z_range[0]), float(z_range[1])):
        res = reach_residual_at(expert, q_rest, pad_center, top_z, draw_clearance, reach)
        if res > worst:
            worst, worst_z = res, top_z
    return worst, worst_z


def highest_reachable_z(
    expert: "StrokeExpert",
    q_rest: torch.Tensor,
    pad_center: np.ndarray,
    z_range: tuple[float, float],
    draw_clearance: float,
    tolerance_m: float,
    reach: float = 0.06,
    steps: int = 12,
) -> float | None:
    """Highest pad top the fitted tool can still work to within ``tolerance_m``.

    Bisected, so an unreachable envelope can say what WOULD work instead of
    leaving the operator to sweep for it by hand. None when even the floor of
    the range is out of reach — then the tool, not the pad, is the problem.
    """
    lo, hi = float(z_range[0]), float(z_range[1])
    if reach_residual_at(expert, q_rest, pad_center, lo, draw_clearance, reach) > tolerance_m:
        return None
    for _ in range(steps):
        mid = 0.5 * (lo + hi)
        if reach_residual_at(expert, q_rest, pad_center, mid, draw_clearance, reach) <= tolerance_m:
            lo = mid
        else:
            hi = mid
    return lo


def _rotations_z_to(n: np.ndarray) -> np.ndarray:
    """(N, 3) unit vectors -> (N, 3, 3) minimal rotations taking +z onto each
    (batched Rodrigues; per-timestep orientation targets make N large)."""
    n = n / np.linalg.norm(n, axis=-1, keepdims=True)
    axis = np.cross(np.array([0.0, 0.0, 1.0]), n)
    s_ = np.linalg.norm(axis, axis=-1)
    c = n[:, 2]
    out = np.tile(np.eye(3), (len(n), 1, 1))
    out[c < 0] = np.diag([1.0, -1.0, -1.0])  # straight down: flip about x
    ok = s_ > 1e-9
    a = axis[ok] / s_[ok, None]
    k = np.zeros((ok.sum(), 3, 3))
    k[:, 0, 1], k[:, 0, 2] = -a[:, 2], a[:, 1]
    k[:, 1, 0], k[:, 1, 2] = a[:, 2], -a[:, 0]
    k[:, 2, 0], k[:, 2, 1] = -a[:, 1], a[:, 0]
    out[ok] = np.eye(3) + s_[ok, None, None] * k + (1 - c[ok])[:, None, None] * (k @ k)
    return out


def _quat_to_matrix(q: torch.Tensor) -> torch.Tensor:
    w, x, y, z = q
    return torch.tensor(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
            [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
            [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
        ],
        dtype=torch.float32,
        device=q.device,
    )
