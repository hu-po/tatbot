"""ManiSkill state, controller and capture operations used by episode execution."""

from __future__ import annotations

from dataclasses import replace
from importlib.metadata import version

from tatbot_contracts.observations import FOLLOWER_JOINTS


class ManiSkillWorld:
    """Bind to the current articulation after every reconfiguring reset.

    Engine objects remain here; positions/actions are device arrays. The
    environment continues to own material updates at each control step.
    """

    def __init__(self, env, config, expert):
        self.env = env
        self.base = env.unwrapped
        self.config = config
        self.expert = expert
        self.versions = {name: version(name) for name in ("mani-skill-nightly", "sapien", "torch", "numpy")}
        if self.base.control_freq != config.timing.control_hz:
            raise ValueError('world control frequency differs from resolved timing')
        self.bind()

    def bind(self):
        self.ik_names = tuple(self.expert.ik.chain.get_joint_parameter_names())
        self.robot, self.idx7, self.idx_ik = bind_articulation(self.base, self.ik_names)

    def reset(self, *, seed, options=None):
        result = self.env.reset(seed=seed, options=options)
        self.bind()
        return result

    def positions(self):
        return self.robot.get_qpos()[:, self.idx_ik].clone()

    def place(self, positions):
        full = self.robot.get_qpos().clone()
        full[:, self.idx_ik] = positions
        self.robot.set_qpos(full)

    def synchronize(self):
        if self.base.gpu_sim_enabled:
            self.base.scene._gpu_apply_all()
            self.base.scene.px.gpu_update_articulation_kinematics()
            self.base.scene._gpu_fetch_all()
        self.base.agent.controller.reset()

    def initialize_material(self, plan, *, stencil=None):
        self.base.set_stencil(stencil)
        if plan.preink is not None:
            self.base.preink(plan.preink)
        self.base.set_dip_schedule(plan)

    def coverage(self):
        return self.base.ink_field.coverage().clone()

    def statistics(self):
        return self.base.ink_episode_stats()

    def cap_rims(self):
        return self.base.cap_rims_np()

    def measurements(self, calibration):
        """Measured follower state and contact, in the contract's joint order."""
        from tatbot_sim.contact_force import external_joint_efforts

        _, distance, _ = self.base.surface.project(self.base.agent.tcp.pose.p)
        contact = external_joint_efforts(
            self.base, self.expert.ik, self.positions(), calibration=calibration,
            tool_id=self.config.tool.tool_id, in_contact=self.base._interaction_mask(distance))
        order = [self.ik_names.index(name) for name in FOLLOWER_JOINTS]
        return (self.robot.get_qpos()[:, self.idx7].clone(),
                replace(contact, joint_efforts=contact.joint_efforts[:, order]))

    def observe(self):
        return self.base.get_obs()

    def advance(self, action, *, capture=True):
        mode = self.base.obs_mode
        try:
            if not capture:
                self.base._obs_mode = 'state'
            return self.env.step(action)
        finally:
            self.base._obs_mode = mode

    def advance_joints(self, positions, velocities, *, capture=True):
        """Apply unnormalized named-order position and velocity references.

        This is the session worker's feed-forward boundary. Synthetic policy
        observations/actions retain their existing seven-position contract.
        """
        import torch

        if self.base._control_mode != 'pd_joint_pos_vel':
            raise ValueError('position/velocity references require pd_joint_pos_vel')
        shape = (self.base.num_envs, len(FOLLOWER_JOINTS))
        positions = torch.as_tensor(positions, dtype=torch.float32, device=self.base.device)
        velocities = torch.as_tensor(velocities, dtype=torch.float32, device=self.base.device)
        if (tuple(positions.shape) != shape or tuple(velocities.shape) != shape
                or not torch.isfinite(positions).all() or not torch.isfinite(velocities).all()):
            raise ValueError('joint references must be finite follower arrays')
        return self.advance(torch.cat((positions, velocities), dim=-1), capture=capture)

    def metadata(self):
        return {"backend": "maniskill", "versions": self.versions, "episode_seeds": self.base.episode_seed,
                "controller": self.base._control_mode,
                "command_channels": (["position", "velocity"] if self.base._control_mode == 'pd_joint_pos_vel'
                                     else ["position"]),
                "scene_seeds": self.base.scene_seed, "camera_seed": self.base.agent.camera_seed,
                "camera_mounts": self.base.agent.camera_mounts,
                "lighting": self.base.lighting_sample,
                "contact_collision": bool(self.base.surface_has_contact_collision)}

    @staticmethod
    def completed(terminated, truncated):
        return bool((terminated | truncated).all())

    def close(self):
        self.env.close()


def bind_articulation(base, ik_names):
    """Resolve named channels again after a reset replaces the articulation."""
    robot = base.agent.robot
    names = [joint.name for joint in robot.active_joints]
    try:
        return (robot, [names.index(name) for name in FOLLOWER_JOINTS],
                [names.index(name) for name in ik_names])
    except ValueError as exc:
        raise RuntimeError(f'articulation does not satisfy the follower joint contract: {names}') from exc
