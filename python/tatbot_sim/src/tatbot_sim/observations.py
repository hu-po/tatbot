"""Build one device-resident observation per tick for all episode consumers.

Sensor corruption is evaluated once, before any output is selected. Video,
dataset depth and reduced contact features therefore see the same samples.
Presentation cameras stay outside policy measurements. Contact truth remains
an explicit separate result; it is never added to the policy feature vector.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

import torch
from tatbot_contracts.observations import ObservationProfile

from tatbot_sim.contact_force import ContactEffort, EffortCalibration
from tatbot_sim.depth_noise import DepthCorruptor, RGBJitter


@dataclass(frozen=True)
class Observation:
    step: int
    time_s: float
    qpos: torch.Tensor
    external_effort: torch.Tensor
    effort_available: tuple[bool, ...]
    rgb: dict[str, torch.Tensor]
    depth_mm: dict[str, torch.Tensor]
    contact: ContactEffort
    profile: ObservationProfile

    @property
    def state(self) -> torch.Tensor:
        return torch.cat((self.qpos, self.external_effort), dim=-1)

    def effort_metadata(self) -> dict:
        return {**self.contact.as_metadata(), "profile": self.profile.metadata(),
                "available": self.effort_available}


class ObservationBuilder:
    """Instance-local measurement/noise state, shared by every output sink."""

    def __init__(self, config, num_envs, device):
        self.config = config
        self.cameras = tuple(camera.role for camera in config.cameras)
        self.profile = config.observation_profile
        payload = config.effort_calibration
        self.calibration = EffortCalibration.from_dict(payload) if payload is not None else None
        self.num_envs, self.device = num_envs, device
        self.reset(config.seed)

    def reset(self, episode_seed):
        self.episode_seed = episode_seed
        dr = self.config.dr
        self.depth = {}
        self.rgb = {}
        for camera in self.cameras:
            self.depth[camera] = (DepthCorruptor(
                self.num_envs, self.device, cfg=dr.depth_noise,
                seed=self.config.seed_for(f'depth-profile:{camera}', episode_seed))
                if dr.corrupt_depth else None)
            self.rgb[camera] = RGBJitter(
                self.num_envs, self.device, cfg=dr.rgb,
                seed=self.config.seed_for(f'rgb-profile:{camera}', episode_seed))

    def _frame_seed(self, camera, kind, step):
        return self.config.seed_for(f'{kind}:{camera}:{self.episode_seed}', step)

    def build(self, world, raw, *, step) -> Observation:
        qpos, contact = world.measurements(self.calibration)
        available = tuple(contact.simulated and self.profile.effort == 'contact' and enabled
                          for enabled in self.profile.effort_mask)
        effort = contact.joint_efforts.clone()
        effort[:, ~torch.tensor(available, device=effort.device)] = 0
        rgb, depth = {}, {}
        sensors = raw.get('sensor_data', {}) if isinstance(raw, Mapping) else {}
        if isinstance(raw, Mapping) and 'sensor_data' in raw and not sensors:
            raise ValueError('captured observation contains no configured cameras')
        for camera in self.cameras:
            if not sensors:
                continue  # a physics-only tick has no image sample
            if camera not in sensors:
                raise ValueError(f'captured observation is missing configured camera {camera}')
            sensor = sensors[camera]
            self._validate_sensor(camera, sensor)
            if 'rgb' in sensor:
                rgb[camera] = self.rgb[camera](sensor['rgb'], seed=self._frame_seed(camera, 'rgb', step))
            if 'depth' in sensor:
                sample = sensor['depth']
                corruptor = self.depth[camera]
                sample = (corruptor(sample, seed=self._frame_seed(camera, 'depth', step))
                          if corruptor is not None else sample)
                depth[camera] = torch.nan_to_num(sample.float(), nan=0, posinf=0, neginf=0).round().clamp(0, 65535).to(torch.int32)
        return Observation(step, step / self.config.timing.control_hz, qpos, effort,
                           available, rgb, depth, contact, self.profile)

    def _validate_sensor(self, role, sensor):
        description = next(camera for camera in self.config.cameras if camera.role == role)
        shape = (self.num_envs, description.height, description.width)
        if not {'rgb', 'depth'} & sensor.keys():
            raise ValueError(f'camera {role} contains no RGB or depth sample')
        for channel, channels in (('rgb', 3), ('depth', 1)):
            if channel in sensor and tuple(sensor[channel].shape) != (*shape, channels):
                raise ValueError(f'camera {role} {channel} shape differs from resolved profile')

    def metadata(self):
        """Resolved per-camera response, with seeds for its spatial fields."""
        result = {}
        for camera in self.cameras:
            rgb, depth = self.rgb[camera], self.depth[camera]
            rgb_values = {name: getattr(rgb, name).reshape(self.num_envs, -1).cpu().tolist()
                          for name in ('exposure', 'wb', 'gamma', 'noise')}
            depth_values = None if depth is None else {
                name: getattr(depth, name).flatten().cpu().tolist()
                for name in ('sigma', 'sigma_corr', 'sigma_iid', 'warp_amp', 'min_z', 'edge_p', 'blob_f', 'graze_w', 'range_w')}
            result[camera] = {"rgb_enabled": rgb.cfg.enabled, "rgb": rgb_values,
                              "depth": depth_values, "depth_edge_kernel": depth.edge_k if depth else None,
                              "rgb_seed": self.config.seed_for(f'rgb-profile:{camera}', self.episode_seed),
                              "depth_seed": self.config.seed_for(f'depth-profile:{camera}', self.episode_seed)}
        return {"schema": "tatbot.sensor-randomization/1", "episode_seed": self.episode_seed,
                "frame_seed_basis": "camera, channel, episode seed, control tick", "cameras": result}
