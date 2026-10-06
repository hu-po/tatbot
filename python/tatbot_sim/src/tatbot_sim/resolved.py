"""Immutable construction inputs assembled from the existing robot registries.

No engine, renderer, model build or device access belongs in resolution. Mutable
registry payloads and DR ranges are retained privately and exposed as copies;
one runtime cannot change another runtime's resolved inputs.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
import os
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path

import ink_spec
import tool_spec
from tatbot_contracts.observations import ObservationProfile
from wrist_cameras import CameraDescription, describe, registry_path

from tatbot_sim import palette as sim_palette
from tatbot_sim import tools
from tatbot_sim.config import DRConfig


@dataclass(frozen=True)
class Timing:
    physics_hz: int = 120
    control_hz: int = 30

    def __post_init__(self):
        if (type(self.physics_hz) is not int or type(self.control_hz) is not int
                or self.control_hz <= 0 or self.physics_hz % self.control_hz
                or self.physics_hz < self.control_hz):
            raise ValueError('physics frequency must be a positive multiple of control frequency')


@dataclass(frozen=True)
class SurfacePose:
    """Top-surface frame in the follower base, fixed across resets.

    Dimensions and material continue to come from the substrate datasheet.
    A prepared scene supplies this pose explicitly instead of sampling PadDR.
    """

    position_m: tuple[float, float, float]
    quaternion_wxyz: tuple[float, float, float, float]

    def __post_init__(self):
        for name, size in (('position_m', 3), ('quaternion_wxyz', 4)):
            values = tuple(getattr(self, name))
            if (len(values) != size or any(type(v) not in (int, float) or not math.isfinite(v)
                                          for v in values)):
                raise ValueError('surface pose requires finite position and quaternion')
            object.__setattr__(self, name, values)
        if abs(sum(v * v for v in self.quaternion_wxyz) - 1) > 1e-8:
            raise ValueError('surface pose requires a unit quaternion')


@dataclass(frozen=True)
class ResolvedConfig:
    repo: Path
    sensor_profile: str
    observation_profile: ObservationProfile
    cameras: tuple[CameraDescription, ...]
    geometry: tool_spec.ResolvedToolGeometry
    staged_pose: tuple[float, ...]
    carriage_rest_m: float
    timing: Timing
    seed: int
    scenario_path: str | None
    supply: tuple[str, str | None]
    ink_policy: ink_spec.InkPolicy
    sources: tuple[tuple[str, str], ...]
    _tool: tool_spec.ToolSpec = field(repr=False)
    _substrate: tool_spec.Substrate = field(repr=False)
    _workspace: dict = field(repr=False)
    _dr: DRConfig = field(repr=False)
    _palette: tuple = field(repr=False)
    _palette_load: tuple = field(repr=False)
    _palette_scene: sim_palette.PaletteScene = field(repr=False)
    _effort_calibration: dict | None = field(repr=False)
    surface_pose: SurfacePose | None = None

    @property
    def tool(self):
        return copy.deepcopy(self._tool)

    @property
    def substrate(self):
        return copy.deepcopy(self._substrate)

    @property
    def workspace(self):
        return copy.deepcopy(self._workspace)

    @property
    def dr(self):
        return copy.deepcopy(self._dr)

    @property
    def palette(self):
        return dict(self._palette)

    @property
    def palette_load(self):
        return dict(self._palette_load)

    @property
    def palette_scene(self):
        return self._palette_scene

    @property
    def effort_calibration(self):
        return copy.deepcopy(self._effort_calibration)

    def with_dr(self, dr: DRConfig):
        return replace(self, _dr=copy.deepcopy(dr).resolve_for(self.substrate))

    def with_supply(self, kind: str, ink_id: str | None = None):
        load = ink_spec.supply_load(kind, self.palette, ink_id, self.repo)
        return replace(self, supply=(kind, ink_id if kind == 'wet' else None),
                       _palette_load=tuple(load.items()))

    def validate_sources(self):
        for name, digest in self.sources:
            path = self.repo / name
            if hashlib.sha256(path.read_bytes()).hexdigest() != digest:
                raise ValueError(f'simulation input changed after resolution: {name}')

    def seed_for(self, stream: str, episode: int = 0) -> int:
        value = f'tatbot.sim-rng/1:{self.seed}:{episode}:{stream}'.encode()
        return int.from_bytes(hashlib.sha256(value).digest()[:8], 'big')

    def metadata(self) -> dict:
        return {
            'schema': 'tatbot.sim-config/1',
            'tool': self._tool.tool_id, 'substrate': self._substrate.name,
            'geometry': asdict(self.geometry), 'dr': asdict(self._dr),
            'sensor_profile': {'name': self.sensor_profile,
                               'cameras': [camera.as_dict() for camera in self.cameras]},
            'observation_profile': self.observation_profile.metadata(),
            'effort_calibration': self.effort_calibration,
            'timing': asdict(self.timing), 'seed': self.seed,
            'rng_streams': {name: self.seed_for(name) for name in ('layout', 'camera', 'lighting', 'noise', 'depth_noise', 'rgb_noise')},
            'scenario_path': self.scenario_path, 'supply': self.supply,
            'palette': {'revision': self._palette_scene.revision,
                        'rim_basis': self._palette_scene.rim_basis,
                        'motion_authority': False},
            'surface_pose': ({'frame': 'right/base_link', **asdict(self.surface_pose)}
                             if self.surface_pose is not None else None),
            'sources': dict(self.sources),
        }


def _sources(tool, scenario_path) -> tuple[tuple[str, str], ...]:
    paths = [tools.REPO / 'urdf/tatbot.urdf', tools.workspace_path(),
             tools.arm_golden_path(), Path(tool.source), registry_path(tools.REPO),
             tools.REPO / 'config/substrates.yaml']
    paths += [tools.REPO / name for name in (ink_spec.INKS_RELPATH, ink_spec.PALETTE_RELPATH,
              ink_spec.LOAD_RELPATH) if (tools.REPO / name).is_file()]
    scene = sim_palette.load(tools.REPO)
    palette_inputs = (sim_palette.GEOMETRY_RELPATH, scene.urdf, scene.mesh, scene.collision_mesh,
                      scene.tag_mesh, scene.tag_inventory)
    paths += [tools.REPO / name for name in dict.fromkeys(filter(None, palette_inputs))]
    effort_path = tools.REPO / 'config/contact_effort.json'
    if effort_path.is_file():
        paths.append(effort_path)
    if scenario_path:
        paths.append(Path(scenario_path))
    return tuple((str(path.relative_to(tools.REPO)) if path.is_relative_to(tools.REPO) else str(path),
                  hashlib.sha256(path.read_bytes()).hexdigest()) for path in paths)


def resolve(*, tool_id: str | None = None, substrate_name: str | None = None,
            sensor_profile: str = 'deployment', seed: int = 0, dr: DRConfig | None = None,
            tip_delta_m: tuple[float, float, float] | None = None,
            timing: Timing | None = None, scenario_path: str | None = None,
            surface_pose: SurfacePose | None = None,
            supply: tuple[str, str | None] | None = None,
            observation_profile: ObservationProfile | None = None) -> ResolvedConfig:
    """Resolve once at a CLI/runtime boundary; explicit values override defaults."""
    if type(seed) is not int or seed < 0:
        raise ValueError('simulation seed must be a nonnegative integer')
    if surface_pose is not None and (not isinstance(surface_pose, SurfacePose) or scenario_path):
        raise ValueError('an explicit surface pose requires SurfacePose and excludes a body scenario')
    delta = tools.calibration_delta_m() if tip_delta_m is None else tuple(tip_delta_m)
    if len(delta) != 3 or not all(math.isfinite(value) for value in delta):
        raise ValueError('tip calibration delta must be a finite three-vector')
    tool = tool_spec.load_tool(tool_id, tools.REPO) if tool_id else tools.active_tool()
    workspace = tools.workspace()
    substrate = tool_spec.substrate_for(tool, tools.REPO,
        name=substrate_name or os.environ.get(tools.SUBSTRATE_ENV) or None)
    palette_scene = sim_palette.load(tools.REPO)
    palette = palette_scene.palette
    selected_supply = tools.supply() if supply is None else supply
    load = ink_spec.supply_load(selected_supply[0], palette, selected_supply[1], tools.REPO)
    effort_path = tools.REPO / 'config/contact_effort.json'
    effort_calibration = json.loads(effort_path.read_text()) if effort_path.is_file() else None
    return ResolvedConfig(
        repo=tools.REPO, sensor_profile=sensor_profile, cameras=describe(tools.REPO, profile=sensor_profile),
        observation_profile=observation_profile or ObservationProfile(),
        geometry=tool_spec.resolved_tool_geometry(tool, workspace, 'right', tools.REPO, tip_delta_m=delta),
        staged_pose=tuple(tools.staged_pose()), carriage_rest_m=tools.carriage_rest_m(),
        timing=timing or Timing(), seed=seed, scenario_path=scenario_path,
        supply=selected_supply, ink_policy=ink_spec.policy_for(tool), sources=_sources(tool, scenario_path),
        _tool=copy.deepcopy(tool), _substrate=copy.deepcopy(substrate), _workspace=copy.deepcopy(workspace),
        _dr=copy.deepcopy(dr or DRConfig()).resolve_for(substrate),
        _palette=tuple(palette.items()), _palette_load=tuple(load.items()),
        _palette_scene=palette_scene,
        _effort_calibration=copy.deepcopy(effort_calibration),
        surface_pose=surface_pose,
    )
