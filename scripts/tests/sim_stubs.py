"""Fake modules for importing scripts/sim_preview.py and scripts/sim_cinematic.py
without the render stack — scoped so they never outlive the module that asked.

Both scripts import ManiSkill, SAPIEN, gymnasium, tyro and a dozen tatbot_sim
submodules at module scope, none of which the offline profile installs, so
their tests fake the lot. The fakes used to be written straight into
sys.modules at collection time and never taken out again. Pytest collects the
whole directory in one process, alphabetically, so every module collected
after test_sim_* saw a ``tatbot_sim`` package with no files in it (and, for a
while, a ``torch`` that was not a package): test_sim_dataset, test_surface_rgbd
and test_train_tb_bridge failed at collection — but only in the full run, since
each of them alone was green, which is how it stayed broken.

The contract now: a :class:`SimStubs` holds the fakes, :meth:`load_script`
installs them only for as long as the script's own import takes, and the
module-scoped fixture from :meth:`fixture` re-installs the same objects for
the span of that test module's tests — sim_cinematic imports tatbot_sim.tasks
& co. inside main(), and the tests read ``sys.modules["tatbot_sim.tools"]``
back to assert on the calls. Both are ``pytest.MonkeyPatch`` contexts, so
whatever sys.modules held before (a real package, or nothing) is what it holds
after. The script keeps its own references to the fakes it bound at import.
"""

from __future__ import annotations

import contextlib
import importlib.util
import sys
import types
from collections.abc import Iterator
from pathlib import Path

import pytest


class SimStubs:
    def __init__(self) -> None:
        self.modules: dict[str, types.ModuleType] = {}

    def module(self, name: str, **attrs: object) -> types.ModuleType:
        """Register a fake module. A dotted name is also bound as an attribute
        of its registered parent, so ``from tatbot_sim import tools`` resolves
        the same object ``import tatbot_sim.tools`` does."""
        mod = types.ModuleType(name)
        for attr, value in attrs.items():
            setattr(mod, attr, value)
        self.modules[name] = mod
        parent, _, child = name.rpartition(".")
        if parent in self.modules:
            setattr(self.modules[parent], child, mod)
        return mod

    def package(self, name: str, **attrs: object) -> types.ModuleType:
        """A fake package: ``__path__`` is empty, so a submodule the test did not
        register fails at import instead of searching the filesystem."""
        mod = self.module(name, **attrs)
        mod.__path__ = []
        return mod

    @contextlib.contextmanager
    def installed(self) -> Iterator[None]:
        """Every registered fake shadows sys.modules until the block ends."""
        with pytest.MonkeyPatch.context() as patch:
            for name, mod in self.modules.items():
                patch.setitem(sys.modules, name, mod)
            yield

    def load_script(self, name: str, path: Path) -> types.ModuleType:
        """Import a scripts/*.py file by path against the fakes.

        The script itself stays registered under ``name``: its Args is a
        dataclass under postponed annotations, which dataclasses resolves
        through sys.modules, and the tests construct one."""
        spec = importlib.util.spec_from_file_location(name, path)
        assert spec is not None and spec.loader is not None, path
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        with self.installed():
            spec.loader.exec_module(module)
        return module

    def fixture(self):
        """Module-scoped autouse fixture: bind it to a module-level name in the
        test file, and the fakes are in place for that module's tests only."""

        @pytest.fixture(scope="module", autouse=True)
        def sim_stubs_installed() -> Iterator[None]:
            with self.installed():
                yield

        return sim_stubs_installed


def install_episode_stubs(stubs):
    """Supply fixture observations through the shared runtime client interface.

    Physics and reference setup are tested in the simulator's integration
    suite; presentation tests only inspect plan handoff and output selection.
    """
    from unittest.mock import MagicMock

    class World:
        def __init__(self, env, config, expert):
            self.env, self.config = env, config

        def positions(self):
            return self.env.unwrapped.agent.robot.get_qpos()[:, :1]

    class Runtime:
        def __init__(self, world, observations):
            self.world = world
            self.step_index = 0
            self.done = False

        def reset(self, **kwargs):
            self.world.env.reset(**kwargs)

        def install(self, plan):
            self.plan = plan
            self.horizon = plan.episode_steps

        def step(self, **kwargs):
            self.last_raw, *rest = self.world.env.step({})
            sensors = self.last_raw['sensor_data']
            observation = types.SimpleNamespace(
                rgb={key: value['rgb'] for key, value in sensors.items() if 'rgb' in value},
                depth_mm={key: value['depth'] for key, value in sensors.items() if 'depth' in value})
            self.step_index += 1
            return {}, observation, rest

        def metadata(self):
            return {}

        def close(self):
            self.world.env.close()

    stubs.package('tatbot_sim.backends')
    stubs.module('tatbot_sim.backends.maniskill', ManiSkillWorld=World)
    instances = []
    def runtime(world, observations):
        instance = Runtime(world, observations)
        instances.append(instance)
        return instance
    constructor = MagicMock(side_effect=runtime)
    constructor.instances = instances
    stubs.module('tatbot_sim.episode', Episode=constructor)
    stubs.module('tatbot_sim.observations', ObservationBuilder=MagicMock())
