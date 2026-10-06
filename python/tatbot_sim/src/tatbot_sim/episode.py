"""Shared reset, reference installation, ticking and completion for sim clients.

The caller supplies a world adapter, an action source and observation builder.
This is an episode lifecycle for synthetic consumers, not a drawing executor:
physical drawing is the ROS 2 stack in ros/.
"""

from __future__ import annotations

from tatbot_sim.reference import measure, refine


class Episode:
    def __init__(self, world, observations):
        self.world = world
        self.observations = observations
        self.expert = world.expert
        self.plan = None
        self.step_index = 0
        self.horizon = 0
        self.last_raw = None
        self.last_observation = None
        self.backend_complete = False

    def reset(self, *, seed, options=None):
        self.plan = None
        self.step_index = 0
        self.horizon = 0
        self.last_observation = None
        self.backend_complete = False
        self.expert.begin_episode(seed)
        self.observations.reset(seed)
        self.last_raw, info = self.world.reset(seed=seed, options=options)
        return self.last_raw, info

    def install(self, plan, *, count=None, clamp_floor=True, stencil=None):
        """Place every client at the same initialized pose and material state."""
        batch = len(plan.targets)
        if count is not None and not 0 < count <= batch:
            raise ValueError('episode count must select at least one member of the batch')
        horizon = int(plan.lengths[:count].max())
        if horizon <= 0:
            raise ValueError('episode plan must have a positive duration')
        self.plan = None
        self.world.initialize_material(plan, stencil=stencil)
        self.coverage_start = self.world.coverage()
        native = getattr(plan, 'native_reference', None)
        if native is not None:
            if plan.n_app or native.positions.shape[:2] != plan.targets.shape[:2]:
                raise ValueError('native reference must include every approach and drawing tick')
            q_start = self.expert.install_reference(native, plan.targets)
            self.world.place(q_start)
            self.world.synchronize()
            self.quality = measure(self.expert, plan, cap_rims=self.world.cap_rims(),
                                   palette=self.world.config.palette)
            self._installed(plan, q_start, horizon)
            return q_start
        q_start = self.expert.solve_pose(plan.targets[:, 0], self.world.positions(),
                                         normals=plan.pen_normals[:, 0])
        initial = q_start
        if plan.q_raised is not None:
            initial = self.expert.seed_pose(plan.q_raised, q_start.shape[0]).to(q_start.device)
        self.world.place(initial)
        self.world.synchronize()
        self.reset_kwargs = {
            'floor_plane': (plan.surface_points, plan.surface_normals) if clamp_floor else None,
            'pen_normals': plan.pen_normals,
            'approach_from': (plan.q_raised, plan.n_app) if plan.q_raised is not None else None,
        }
        self.expert.reset(plan.targets, q_start, **self.reset_kwargs)
        self.quality = refine(
            self.expert, plan, q_start, self.reset_kwargs,
            num_envs=len(plan.targets), count=count,
            cap_rims=self.world.cap_rims(), palette=self.world.config.palette)
        self._installed(plan, q_start, horizon)
        return q_start

    def _installed(self, plan, q_start, horizon):
        self.plan = plan
        self.q_start = q_start
        self.step_index = 0
        self.horizon = horizon
        self.backend_complete = False
        self.last_raw = None
        self.last_observation = None

    @property
    def done(self):
        return self.plan is not None and (self.backend_complete or self.step_index >= self.horizon)

    @property
    def time_s(self):
        return self.step_index / self.world.config.timing.control_hz

    def measure(self):
        if self.last_observation is None:
            raw = self.last_raw if self.last_raw is not None else self.world.observe()
            self.last_raw = raw
            self.last_observation = self.observations.build(
                self.world, raw, step=self.step_index)
        return self.last_observation

    def step(self, action=None, *, capture=True):
        if self.plan is None or self.done:
            raise RuntimeError('episode must have an installed, unfinished plan')
        command = self.expert.act() if action is None else action
        self.last_raw, reward, terminated, truncated, info = self.world.advance(command, capture=capture)
        self.backend_complete = self.world.completed(terminated, truncated)
        self.step_index += 1
        self.last_observation = None
        return command, self.measure(), (reward, terminated, truncated, info)

    def statistics(self):
        return self.world.statistics()

    def metadata(self):
        return {"lifecycle": "tatbot.sim-episode/1", "step": self.step_index,
                "time_s": self.time_s, "horizon": self.horizon, "done": self.done,
                "completion": "backend" if self.backend_complete else "horizon" if self.done else None,
                "observation_profile": self.observations.profile.metadata(),
                "world": self.world.metadata(),
                "sensor_randomization": self.observations.metadata(),
                "action_noise_seed": self.world.config.seed_for("noise", self.observations.episode_seed),
                "motion": (self.plan.native_reference.metadata()
                           if self.plan is not None and getattr(self.plan, 'native_reference', None) is not None
                           else {"planner": "synthetic-stroke-expert"}),
                "reference_quality": self.quality.metadata() if self.plan is not None else None}

    def close(self):
        self.world.close()
