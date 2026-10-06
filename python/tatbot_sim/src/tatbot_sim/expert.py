"""Scripted stroke-following expert: batched damped-least-squares IK.

Emits native ``pd_joint_pos`` actions (7-dim: 6 arm joints + carriage), so
datasets need no control-mode conversion. IK runs on the serial chain
base->tattoo_needle, which since 2026-09-08 is SEVEN joints: the tool hangs
off the carriage as it does on the real arm, so the prismatic axis is an EE
ancestor and can be solved.

Whether it *is* solved is ``carriage_ik``, the executor's own per-path flag
(``cpp/teleop/square_probe.cpp``). Off, the carriage holds the rest value the
follower position-holds it at and the expert reproduces its six-axis
behaviour exactly. On, it serves the component of tip position error along the
local surface normal -- see :mod:`tatbot_sim.carriage` for why that rather
than its share of the twist, and for the envelope and rate caps, which are
read from ``config/motion_constants.json`` rather than restated here. The DART
bursts never touch it either way: it is the safety layer's axis on the real
follower, and a recovery demonstration that jogs it is not one the hardware
would produce.

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

The commanded trajectory is clamped to the contact contract: at the stroke it
permits only 0.25 mm of numerical penetration, while travel stays at least
1 mm clear. The env adds rigid tip/pad contact for a flat qualified substrate;
before these bounds a noise burst could command straight through the pad and
the data taught exactly that:
the 99%-sim policy of 2026-08-21 drove ~40 mm through the real paper. The
clamp keeps the recovery behaviour that matters: bursts still pin the needle
onto the surface and the decaying burst then
commands back up — a "too deep, come up" demonstration. States below the
declared penetration band disappear from the data, and measurement says that is all
they were: in unclamped data every achieved-below step was downstream of a
commanded-below step, and on the real rig contact plus the follower's z-floor
make such states unreachable anyway. The 2026-08-21 depth audit measured both
properties on the written dataset; its script is retired (git history has it).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pytorch_kinematics as pk
import torch

from tatbot_sim import interaction
from tatbot_sim.carriage import CARRIAGE_JOINT, CarriagePolicy
from tatbot_sim.resolved import ResolvedConfig, resolve
from tatbot_sim.urdf import build_tatbot_urdf

if TYPE_CHECKING:  # config imports nothing from here; keep it that way
    from tatbot_sim.config import NoiseDR

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

def _camera_axis_world(config):
    # Recorded ballpoint roll includes the measured seat's lean. Applying it
    # to a straight nominal tip creates an incompatible IK orientation.
    if config.geometry.measured:
        return _CAM_AXIS_BY_TOOL.get(config.tool.tool_id, np.array([0.0, -1.0, 0.0]))
    return np.array([0.0, -1.0, 0.0])


def _pen_down_matrix(config) -> torch.Tensor:
    """(3, 3) base pen-down orientation: bore down, cameras as recorded.

    Built as a pair of orthonormal triads so the bore constraint is exact and
    the camera axis takes whatever component of CAM_AXIS_WORLD is left
    perpendicular to it."""
    reg = __import__("tatbot_sim.tools", fromlist=["registry"]).registry()
    tcp = np.asarray(config.geometry.tcp_offset_m, dtype=np.float64)
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
    world = triad(np.array([0.0, 0.0, -1.0]), _camera_axis_world(config))
    return torch.tensor(world @ needle.T, dtype=torch.float32)


class BatchedIK:
    """Damped least-squares IK over a pytorch_kinematics serial chain."""

    def __init__(self, device: torch.device, damping: float = 0.05, ori_weight: float = 0.3,
                 carriage: CarriagePolicy | None = None, roll_weight: float = 0.0,
                 config: ResolvedConfig | None = None):
        self.config = config or resolve()
        self.urdf_path = build_tatbot_urdf(config=self.config)
        self.pen_down = _pen_down_matrix(self.config).to(device)
        with open(self.urdf_path, "rb") as f:
            urdf = f.read()
        self.chain = pk.build_serial_chain_from_urdf(urdf, end_link_name=EE_LINK).to(
            device=device, dtype=torch.float32
        )
        lim = self.chain.get_joint_limits()
        self.q_lo = torch.as_tensor(lim[0], dtype=torch.float32, device=device)
        self.q_hi = torch.as_tensor(lim[1], dtype=torch.float32, device=device)
        self.damping = damping
        self.ori_weight = ori_weight
        self.roll_weight = roll_weight
        # Which way the tool points in the EE frame. target_rotations builds
        # its targets as align @ PEN_DOWN with align carrying world +z onto the
        # wanted axis, so this is the local direction that lands on it.
        self.tool_axis = (self.pen_down.T @ torch.tensor(
            [0.0, 0.0, 1.0], device=device)).contiguous()
        self.device = device
        names = self.chain.get_joint_parameter_names()
        self.n_joints = len(names)
        # Since the tool hangs off the carriage the chain is seven-dimensional.
        # Found by name: the action layout is arm-then-carriage, but nothing
        # here should assume the index.
        self.carriage_index = names.index(CARRIAGE_JOINT) if CARRIAGE_JOINT in names else None
        self.carriage = carriage
        if carriage is not None and self.carriage_index is None:
            raise ValueError(
                f"a carriage policy was given but {CARRIAGE_JOINT!r} is not in the IK "
                f"chain ({names}); the tool is welded above the carriage"
            )

    def fk(self, q: torch.Tensor) -> torch.Tensor:
        """(B, n) joints -> (B, 4, 4) EE pose in base frame."""
        return self.chain.forward_kinematics(q).get_matrix()

    def step(
        self, q: torch.Tensor, target_pos: torch.Tensor, target_rot: torch.Tensor, iters: int = 3,
        *, normals: torch.Tensor | None = None, carriage_from: torch.Tensor | None = None,
        max_carriage_step_m: float | None = None, centering: float = 0.0,
    ) -> torch.Tensor:
        """Iterate DLS from ``q`` toward (target_pos (B,3), target_rot (B,3,3)).

        The carriage is never left to the arm solve. Given ``normals`` (B,3),
        the outward unit surface normal at each target, and a carriage policy,
        it serves the component of position error along that normal, bounded by
        the guarded envelope and -- with ``carriage_from`` and
        ``max_carriage_step_m`` -- by the executor's rate cap against the
        previous control frame; the arm then solves the residual twist, so a
        curved surface's lever-arm swing stays with the arm. Without
        ``normals`` the carriage holds the value ``q`` arrives with.
        """
        ci = self.carriage_index
        drive = normals is not None and self.carriage is not None and ci is not None
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
            # The tip is a body of revolution, so spin about its own axis is not
            # part of the contact pose. Even a small camera-roll preference can
            # drive a reachable target into a joint limit when the requested
            # camera frame is far from the arm's current roll. Solve position
            # and tool-axis tilt without spending either on that preference.
            if self.roll_weight != 1.0:
                axis = torch.nn.functional.normalize(r_cur @ self.tool_axis, dim=1)
                spin = (rot_err * axis).sum(dim=1, keepdim=True) * axis
                rot_err = rot_err - spin + self.roll_weight * spin
            twist = torch.cat([pos_err, self.ori_weight * rot_err], dim=1)  # (B, 6)
            jac = self.chain.jacobian(q)  # (B, 6, n)
            if ci is not None:
                if drive:
                    column = jac[:, :, ci]                      # (B, 6) its whole twist
                    align = (column[:, :3] * normals).sum(-1)   # how much of it is normal
                    normal_err = (pos_err * normals).sum(-1)
                    # Projection, not inversion: as the tool leans off the
                    # normal the carriage does proportionally less and the arm
                    # takes the rest, instead of dividing by a vanishing term.
                    want = q[:, ci] + normal_err * align \
                        + centering * (self.carriage.bias_m - q[:, ci])
                    want = want.clamp(self.carriage.min_m, self.carriage.max_m)
                    if carriage_from is not None and max_carriage_step_m is not None:
                        want = torch.clamp(want, carriage_from - max_carriage_step_m,
                                           carriage_from + max_carriage_step_m)
                    twist = twist - column * (want - q[:, ci]).unsqueeze(-1)
                    q = q.clone()
                    q[:, ci] = want
                # Whether or not it was driven, the carriage is not the arm
                # solve's to spend: zeroing its column keeps dq[ci] at zero, so
                # a locked carriage stays exactly where it was put.
                jac = jac.clone()
                jac[:, :, ci] = 0.0
            jjt = jac @ jac.transpose(1, 2)
            jjt += (self.damping**2) * torch.eye(6, device=self.device)
            dq = jac.transpose(1, 2) @ torch.linalg.solve(jjt, twist.unsqueeze(-1))
            q = torch.clamp(q + dq.squeeze(-1), self.q_lo, self.q_hi)
        return q


class ReachMask:
    """Where on a canvas the fitted tool can be held normal to the surface.

    A curved profile is not uniformly workable: its crest is locally flat and its
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
    degrees off it, and the flanks of a cylinder are exactly that case.

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
    {3RL, laser} x {seed 3, 11} on the former draped-skin model — median leftover
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
    # Pinned at carriage rest whether or not the carriage is a solved axis.
    # This mask decides where strokes may be placed, so letting it move with a
    # solver flag would re-sample the scene: two runs differing only in
    # carriage_ik drew different designs in 2 of 4 flat pairs (2026-09-08),
    # which silently re-qualifies a mask every reach and clearance gate here
    # was accepted against. carriage_ik chooses how an accepted plan is
    # executed, never which plan is accepted.
    q = expert.seed_pose(q0_arm[:1], 1, carriage_m=expert.carriage_rest_m) \
              .expand(total, expert.ik.n_joints).contiguous()
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
    the poses a curved profile makes impossible: height is exactly what the
    arm is short of. Cylindrical profiles are audited directly by the same
    mask, so an episode that starts too high does not ask for a pose the
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
        pen_grip: float | None = None,
        noise: "NoiseDR | None" = None,
        seed: int | None = None,
        carriage_ik: bool = False,
        control_hz: float | None = None,
        config: ResolvedConfig | None = None,
    ):
        from tatbot_sim.config import NoiseDR

        # `carriage_ik` mirrors the executor's per-path flag of the same name:
        # off pins the carriage at rest and reproduces the six-axis behaviour
        # exactly, which is what every A/B against it needs.
        #
        # It defaults OFF because turning it on moves the start pose to the
        # carriage's 2 mm drawing bias, and every reach mask, clearance margin
        # and accepted body scenario in this repo was qualified against the
        # pinned pose. Those gates have to be re-run, not silently reinterpreted
        # -- and as of 2026-09-08 the axis buys no measured reference accuracy
        # here, because this expert solves an idealised open-loop trajectory
        # that the six-axis arm already tracks to a few microns. The precision
        # the carriage is actually for is a tracking property of the real arm.
        self.carriage_policy = CarriagePolicy.from_repo() if carriage_ik else None
        self.config = config or resolve()
        self.control_hz = float(self.config.timing.control_hz if control_hz is None else control_hz)
        pen_grip = self.config.carriage_rest_m if pen_grip is None else pen_grip
        self.ik = BatchedIK(device, carriage=self.carriage_policy, config=self.config)
        self.carriage_rest_m = float(pen_grip)
        self.num_envs = num_envs
        self.device = device
        self.pen_grip = pen_grip
        self.noise = noise or NoiseDR()
        # one stream for the whole run: re-seeding per batch replayed the
        # SAME burst timing in every batch of a dataset
        self._nrng = np.random.default_rng(self.config.seed_for("noise") if seed is None else seed)
        self._noise: torch.Tensor | None = None
        """This batch's DART bursts, kept so a re-solve reuses them rather than
        advancing the stream and making the run depend on solver effort."""
        self.targets: torch.Tensor | None = None  # (B, T, 3) world-frame EE positions
        self.q_ref: torch.Tensor | None = None  # (B, T, n_joints) solved joint reference
        self.actions: torch.Tensor | None = None
        self.t = 0
        self.clamped_fraction = 0.0  # of the last reset's steps, how many hit the floor

    def target_rotations(self, normals: np.ndarray | None, batch: int) -> torch.Tensor:
        """(N, 3, 3) pen-down orientations along the given axis directions.

        ``normals`` are pen-axis directions, one per row — per env, or per
        (env, timestep) flattened; None means level. The base pen-down pose is
        tilted by the minimal rotation taking world +z onto each direction, so
        the wrist twist (camera orientation) stays put while the pen leans.
        """
        base = self.ik.pen_down
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
        return self.ik.step(self.seed_pose(q0_arm, t.shape[0]).clone(), t, rot_b, iters=iters)

    def begin_episode(self, seed: int):
        """Bind perturbations to an episode, independent of earlier plans."""
        self._nrng = np.random.default_rng(self.config.seed_for('noise', seed))
        self._noise = None

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
        resolve: bool = False,
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
        plane. ``resolve`` marks a re-solve of the SAME batch -- the residual
        gate's longer retry and every contact-settle round -- so it keeps that
        batch's DART bursts instead of drawing new ones. Without it the noise
        depended on how many times the solver happened to run, and two runs
        differing only in a solver flag were not comparable.
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

        n_dof = self.ik.n_joints
        ci = self.ik.carriage_index
        flat_tgt = targets.reshape(b * t_len, 3)
        flat_rot = rot_seq.reshape(b * t_len, 3, 3).contiguous()
        q0 = self.seed_pose(q0_arm, b)
        # The batch stage solves every timestep independently, so it has no
        # previous frame to rate-limit the carriage against and leaves it on its
        # seed; the sequential sweep below is where the axis is actually driven.
        q = q0.unsqueeze(1).expand(b, t_len, n_dof).reshape(b * t_len, n_dof).clone()
        q = self.ik.step(q, flat_tgt, flat_rot, iters=batch_iters)
        q = q.reshape(b, t_len, n_dof)

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
        # The sweep is the only place with a previous control frame in hand, so
        # it is where the carriage's rate cap can mean anything. The surface's
        # own normals drive it -- plane, deformed cylinder or posed body patch
        # alike -- and without a surface the axis simply stays put.
        surface_normals = None
        if self.carriage_policy is not None and floor_plane is not None:
            surface_normals = _per_step(
                torch.as_tensor(np.asarray(floor_plane[1], dtype=np.float32),
                                dtype=torch.float32, device=self.device),
                b, t_len,
            )
        max_carriage_step_m = (self.carriage_policy.max_step_m(self.control_hz)
                               if self.carriage_policy is not None else None)
        centering = (self.carriage_policy.centering_per_iteration(self.control_hz, sweep_iters)
                     if self.carriage_policy is not None else 0.0)
        for _ in range(sweeps):
            prev = q[:, 0]
            cols = []
            for i in range(t_len):
                prev = self.ik.step(
                    prev, targets[:, i], rot_seq[:, i], iters=sweep_iters,
                    normals=None if surface_normals is None else surface_normals[:, i],
                    carriage_from=None if ci is None else prev[:, ci],
                    max_carriage_step_m=max_carriage_step_m,
                    centering=centering,
                )
                cols.append(prev)
            q = torch.stack(cols, dim=1)

        # Precompute the full action tensor. act() is called inside the hot
        # loop while dozens of encode threads compete for the CPU, so any
        # host-side RNG or host->device copy there is disproportionately
        # expensive; doing it once per batch keeps the loop to one slice.
        # Burst frequency and size draw per env per batch from the NoiseDR
        # ranges, so episodes span near-clean to moderately perturbed.
        # Drawn once per BATCH, not once per reset(). A re-solve is the same
        # episode being solved again, so it keeps the episode's own bursts.
        n_app = approach_from[1] if approach_from is not None else 0
        if resolve and self._noise is not None:
            noise_t = self._noise
        else:
            noise_t = self._draw_noise(b, t_len, n_app, n_dof, ci)
            self._noise = noise_t

        if approach_from is not None:
            q_raised, n_app = approach_from
            qr = self.seed_pose(q_raised, b)
            u = torch.linspace(0, 1, n_app + 1, device=self.device)[:-1]
            blend = (10 * u**3 - 15 * u**4 + 6 * u**5).view(1, -1, 1)  # min-jerk
            seg = qr.unsqueeze(1) + (q[:, :1] - qr.unsqueeze(1)) * blend
            q = torch.cat([seg, q], dim=1)
            t_len += n_app

        q_cmd = torch.clamp(q + noise_t, self.ik.q_lo, self.ik.q_hi)
        self.clamped_fraction = 0.0
        if floor_plane is not None:
            pts, nms = floor_plane
            pt_t = torch.as_tensor(pts, dtype=torch.float32, device=self.device)
            nm_t = torch.as_tensor(nms, dtype=torch.float32, device=self.device)
            # Per-step floor. On and near the stroke (reference inside the
            # contact band), noise may use only the declared sub-millimetre
            # penetration allowance. Anywhere the reference is clear —
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
                ref_dist > interaction.CONTACT_ABOVE_TOLERANCE_M,
                torch.full_like(ref_dist, interaction.TRAVEL_FLOOR_M),
                torch.full_like(ref_dist, -interaction.MAX_PENETRATION_M),
            )
            if approach_from is not None:
                # the approach descends from the raised pose, always well clear
                offset = torch.cat(
                    [torch.full((b, approach_from[1]), interaction.TRAVEL_FLOOR_M,
                                device=self.device), offset],
                    dim=1,
                )
            q_cmd = self._clamp_to_floor(q, q_cmd, pt_t, nm_t, offset)
        # The carriage is a solved column of q_cmd now, not a constant appended
        # after the fact. With carriage_ik off it holds pen_grip for the whole
        # episode, which is the tensor the six-axis expert used to build.
        self.actions = q_cmd  # (B, T, 7)

        self.q_ref = q
        self.targets = targets
        self.t = 0

    def _draw_noise(self, b: int, t_len: int, n_app: int, n_dof: int,
                    ci: int | None) -> torch.Tensor:
        """One batch's DART bursts, (b, n_app + t_len, n_dof), approach first.

        Every reset() used to draw from the running stream, and the contact
        settle re-solves a batch up to three times while the residual gate can
        add a fourth. So the bursts depended on how hard the solve happened to
        be: two runs differing only in a solver flag got different noise, and
        no solver A/B in this factory was comparable (measured 2026-09-08,
        burst magnitude 30.11 against 26.74 for one extra re-solve). Drawing
        per batch advances the stream exactly once either way, so a seed
        reproduces the bursts it always did on a run that never re-solved.

        The carriage is the safety layer's axis on the real follower, so the
        bursts stay off it: a recovery demonstration that jogs it is not one
        the hardware would ever produce. Arm draws keep their (b, 6) shape so
        the stream is the one existing seeds already produced.
        """
        nrng = self._nrng
        n_prob = nrng.uniform(*self.noise.prob, b).astype(np.float32)[:, None]
        n_scale = nrng.uniform(*self.noise.scale, b).astype(np.float32)[:, None]
        arm_cols = [j for j in range(n_dof) if j != ci]

        def stream(steps: int) -> np.ndarray:
            out = np.zeros((b, steps, n_dof), dtype=np.float32)
            cur = np.zeros((b, len(arm_cols)), dtype=np.float32)
            for i in range(steps):
                fires = (nrng.random((b, 1)) < n_prob).astype(np.float32)
                burst = nrng.standard_normal((b, len(arm_cols))).astype(np.float32) * n_scale
                cur = cur * self.noise.decay + burst * fires
                out[:, i, arm_cols] = cur
            return out

        main = stream(t_len)
        block = main if not n_app else np.concatenate([stream(n_app), main], axis=1)
        return torch.as_tensor(block, device=self.device)

    def seed_pose(self, q_arm: torch.Tensor, b: int,
                  carriage_m: float | None = None) -> torch.Tensor:
        """(B, n_joints) start pose, with the carriage seeded if it is missing.

        Callers hand us the six arm joints, because that is what a staged pose
        and a reach probe are. A solved carriage starts at its drawing bias so
        it has authority in both directions; a locked one starts at rest, where
        the follower position-holds it.
        """
        q = torch.as_tensor(q_arm, dtype=torch.float32, device=self.device)
        if q.ndim == 1:
            q = q.unsqueeze(0)
        if q.shape[0] == 1 and b > 1:
            q = q.expand(b, -1)
        ci = self.ik.carriage_index
        if ci is None:
            return q.contiguous()
        arm = [j for j in range(self.ik.n_joints) if j != ci]
        if q.shape[1] == self.ik.n_joints:
            q = q[:, arm]
        elif q.shape[1] != len(arm):
            raise ValueError(
                f"start pose has {q.shape[1]} joints; this chain has {self.ik.n_joints} "
                f"({len(arm)} arm + carriage)"
            )
        # The expert owns the carriage for the whole episode, so it sets the
        # start too -- a caller's live qpos carries the rest value, which is
        # outside the guarded drawing envelope and is exactly the start the
        # executor refuses (square_probe.cpp). Taking only the arm columns keeps
        # a six- and a seven-wide caller meaning the same thing.
        seed = carriage_m if carriage_m is not None else (
            self.carriage_policy.bias_m if self.carriage_policy is not None
            else self.carriage_rest_m)
        column = torch.full((q.shape[0], 1), seed, dtype=torch.float32, device=self.device)
        return torch.cat([q, column], dim=1).contiguous()

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

        flat_ref = q_ref.reshape(-1, self.ik.n_joints)
        flat_cmd = q_cmd.reshape(-1, self.ik.n_joints).clone()
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
        return flat_cmd.reshape(b, t_len, self.ik.n_joints)

    @property
    def horizon(self) -> int:
        return 0 if self.targets is None else self.targets.shape[1]

    def install_reference(self, reference, targets):
        """Use native positions unchanged; IK remains available for measurement.

        No perturbation, floor clamp or reference refinement is applied here.
        The engine's float32 command boundary is the sole numeric conversion.
        """
        from tatbot_contracts.observations import FOLLOWER_JOINTS

        if abs(reference.period_s * self.control_hz - 1) > 1e-12:
            raise ValueError('native reference period differs from world control period')
        order = [FOLLOWER_JOINTS.index(name) for name in self.ik.chain.get_joint_parameter_names()]
        self.actions = torch.tensor(reference.positions, dtype=torch.float32, device=self.device)
        self.q_ref = self.actions[:, :, order].clone()
        self.targets = torch.as_tensor(targets, dtype=torch.float32, device=self.device)
        self._noise = None
        self.clamped_fraction = 0.0
        self.t = 0
        return torch.tensor(reference.seed[:, order], dtype=torch.float32, device=self.device)

    def act(self) -> torch.Tensor:
        """Return the next (B, 7) pd_joint_pos action."""
        if self.actions is None:
            raise RuntimeError("StrokeExpert.reset() must be called before act()")
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
            [[pad_center[0] + off, pad_center[1], top_z + draw_clearance]],
            dtype=torch.float32, device=expert.device,
        )
        best = float("inf")
        for attempt in range(retries + 1):
            seed = q_rest[:1].clone()
            if attempt:
                jitter = torch.randn(seed.shape, generator=gen) * 0.3
                # 0.3 is radians for a revolute joint; on the metre-scale
                # carriage it is 300 mm, which the joint-limit clamp turns into
                # a random draw across the axis's whole 44 mm travel and a
                # reach verdict that depends on it.
                if expert.ik.carriage_index is not None:
                    jitter[:, expert.ik.carriage_index] = 0.0
                seed += jitter.to(expert.device)
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
