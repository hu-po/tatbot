"""Validate the catalog against locked upstream sources without editing the environment."""

from __future__ import annotations

import ast
import json
import types
from pathlib import Path

import lerobot_patch_engine as engine  # noqa: E402
import pytest
from lerobot_patches import PATCHES  # noqa: E402


def test_locked_patch_set_applies_to_copies_and_preserves_action_features(tmp_path, monkeypatch):
    pytest.importorskip("lerobot")
    originals, planned, targets, skipped = engine.plan_patches(PATCHES)
    assert not skipped, f"integration environment is missing patch targets: {sorted(skipped)}"
    assert set(targets) == {name for name, _, _ in PATCHES}
    modules = {}
    for name, source in targets.items():
        path = tmp_path / f"{name}.py"
        path.write_text(originals[source])
        modules[name] = types.SimpleNamespace(__file__=str(path))
    real_import = engine.importlib.import_module
    monkeypatch.setattr(engine.importlib, "import_module",
                        lambda name: modules[name] if name in modules else real_import(name))
    receipt = tmp_path / "manifest.json"
    assert engine.apply_patches(PATCHES, receipt_path=receipt) == 0
    first = receipt.read_bytes()
    assert len(json.loads(first)["modules"]) == len(targets)
    assert engine.apply_patches(PATCHES, receipt_path=receipt) == 0
    assert receipt.read_bytes() == first
    for name, source in targets.items():
        assert Path(modules[name].__file__).read_text() == planned[source]
        assert source.read_text() == originals[source]

    # Exercise the actual patched upstream comprehensions. Observation effort
    # belongs in policy inputs; it must never become an action feature.
    context = ast.parse(planned[targets["lerobot.rollout.context"]])
    expressions = {
        target.id: node.value
        for node in ast.walk(context) if isinstance(node, ast.Assign)
        for target in node.targets if isinstance(target, ast.Name)
        and target.id in {"observation_features_hw", "action_features_hw"}
    }
    features = {"joint.pos": float, "joint.vel": float, "joint.eff": float,
                "joint.ext_eff": float, "camera": (3, 32, 32), "unrelated": str}
    scope = {"all_obs_features": features, "robot": types.SimpleNamespace(action_features=features)}
    observation = eval(compile(ast.Expression(expressions["observation_features_hw"]), "context", "eval"), scope)
    action = eval(compile(ast.Expression(expressions["action_features_hw"]), "context", "eval"), scope)
    assert set(observation) == {"joint.pos", "joint.vel", "joint.eff", "joint.ext_eff", "camera"}
    assert set(action) == {"joint.pos", "joint.vel"}
