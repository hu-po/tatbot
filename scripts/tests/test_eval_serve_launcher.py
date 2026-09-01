from __future__ import annotations

import hashlib
import json
import os
import runpy
import subprocess
import sys
import types
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]


def test_patcher_accepts_final_form_supplied_by_later_overlap(
    tmp_path: Path, monkeypatch
) -> None:
    namespace = runpy.run_path(str(REPO / "scripts" / "il_patch_lerobot.py"))
    source = tmp_path / "fake_patch_target.py"
    source.write_text("intermediate")
    module = types.ModuleType("tatbot_fake_patch_target")
    module.__file__ = str(source)
    monkeypatch.setitem(sys.modules, module.__name__, module)
    namespace["main"].__globals__["PATCHES"] = [
        (module.__name__, ["missing-predecessor"], "final"),
        (module.__name__, ["intermediate"], "final"),
    ]

    assert namespace["main"]() == 0
    assert source.read_text() == "final"


def test_patcher_rerun_is_noop_and_returns_zero_when_already_patched(
    tmp_path: Path, monkeypatch
) -> None:
    namespace = runpy.run_path(str(REPO / "scripts" / "il_patch_lerobot.py"))
    source = tmp_path / "fake_patch_target_noop.py"
    source.write_text("already_patched_content")
    module = types.ModuleType("tatbot_fake_patch_target_noop")
    module.__file__ = str(source)
    monkeypatch.setitem(sys.modules, module.__name__, module)
    namespace["main"].__globals__["PATCHES"] = [
        (module.__name__, ["old_content"], "already_patched_content"),
    ]

    assert namespace["main"]() == 0
    assert source.read_text() == "already_patched_content"


def test_groot_server_rejects_unavailable_saved_base_model(tmp_path: Path) -> None:
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    (checkpoint / "config.json").write_text(
        json.dumps(
            {
                "type": "groot",
                "base_model_path": "/training-host/path/that/is/not-portable",
            }
        )
    )
    env_root = tmp_path / "env"
    (env_root / ".venv" / "bin").mkdir(parents=True)
    os.symlink(sys.executable, env_root / ".venv" / "bin" / "python")

    result = subprocess.run(
        [
            str(REPO / "scripts" / "eval" / "serve.sh"),
            "--policy",
            str(checkpoint),
            "--env-root",
            str(env_root),
        ],
        cwd=REPO,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 2
    assert "pass --base-model PATH" in result.stderr


def test_groot_server_rejects_contract_from_different_postprocessor(tmp_path: Path) -> None:
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    base_model = tmp_path / "base-model"
    base_model.mkdir()
    (checkpoint / "config.json").write_text(
        json.dumps(
            {
                "type": "groot",
                "base_model_path": str(base_model),
                "n_action_steps": 16,
                "output_features": {"action": {"shape": [7]}},
            }
        )
    )
    (checkpoint / "policy_postprocessor.json").write_text("{}")
    contract = tmp_path / "contract.json"
    contract.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "kind": "demonstration-derived no-arm trajectory plausibility contract",
                "horizon": 16,
                "joints": 7,
                "postprocessor_sha256": "not-this-checkpoint",
            }
        )
    )
    env_root = tmp_path / "env"
    (env_root / ".venv" / "bin").mkdir(parents=True)
    os.symlink(sys.executable, env_root / ".venv" / "bin" / "python")

    result = subprocess.run(
        [
            str(REPO / "scripts" / "eval" / "serve.sh"),
            "--policy",
            str(checkpoint),
            "--plausibility-contract",
            str(contract),
            "--env-root",
            str(env_root),
        ],
        cwd=REPO,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode != 0
    assert "postprocessor hash does not match checkpoint" in result.stderr


def test_act_server_accepts_contract_option_and_checks_binding(tmp_path: Path) -> None:
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    (checkpoint / "config.json").write_text(
        json.dumps(
            {
                "type": "act",
                "n_action_steps": 16,
                "output_features": {"action": {"shape": [7]}},
            }
        )
    )
    (checkpoint / "policy_postprocessor.json").write_text("{}")
    contract = tmp_path / "contract.json"
    contract.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "kind": "demonstration-derived no-arm trajectory plausibility contract",
                "horizon": 16,
                "joints": 7,
                "postprocessor_sha256": "not-this-checkpoint",
            }
        )
    )
    env_root = tmp_path / "env"
    (env_root / ".venv" / "bin").mkdir(parents=True)
    os.symlink(sys.executable, env_root / ".venv" / "bin" / "python")

    result = subprocess.run(
        [
            str(REPO / "scripts" / "eval" / "serve.sh"),
            "--policy",
            str(checkpoint),
            "--plausibility-contract",
            str(contract),
            "--env-root",
            str(env_root),
        ],
        cwd=REPO,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode != 0
    assert "currently requires a GR00T policy" not in result.stderr
    assert "postprocessor hash does not match checkpoint" in result.stderr


def test_act_server_requires_checkpoint_bound_processor_state(tmp_path: Path) -> None:
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    (checkpoint / "config.json").write_text(
        json.dumps(
            {
                "type": "act",
                "n_action_steps": 16,
                "output_features": {"action": {"shape": [7]}},
            }
        )
    )
    postprocessor = checkpoint / "policy_postprocessor.json"
    postprocessor.write_text("{}")
    contract = tmp_path / "contract.json"
    contract.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "kind": "demonstration-derived no-arm trajectory plausibility contract",
                "horizon": 16,
                "joints": 7,
                "postprocessor_sha256": hashlib.sha256(postprocessor.read_bytes()).hexdigest(),
            }
        )
    )
    env_root = tmp_path / "env"
    (env_root / ".venv" / "bin").mkdir(parents=True)
    os.symlink(sys.executable, env_root / ".venv" / "bin" / "python")

    result = subprocess.run(
        [
            str(REPO / "scripts" / "eval" / "serve.sh"),
            "--policy",
            str(checkpoint),
            "--plausibility-contract",
            str(contract),
            "--env-root",
            str(env_root),
        ],
        cwd=REPO,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode != 0
    assert "lacks processor artifact hashes" in result.stderr


def test_policy_server_reconnect_patch_is_exact_and_fail_closed() -> None:
    namespace = runpy.run_path(str(REPO / "scripts" / "il_patch_lerobot.py"))
    replacement = next(
        new
        for module, _old, new in namespace["PATCHES"]
        if module == "lerobot.async_inference.policy_server"
        and "Reusing loaded checkpoint for matching client reconnect" in new
    )

    assert replacement.index("if self.policy is not None:") < replacement.index(
        "self.policy = policy_class.from_pretrained"
    )
    for field in (
        "checkpoint",
        "policy_type",
        "device",
        "features",
        "actions_per_chunk",
        "rename_map",
    ):
        assert f'mismatches.append("{field}")' in replacement
    assert 'reset = getattr(self.policy, "reset", None)' in replacement
    assert "return services_pb2.Empty()" in replacement


def test_groot_predecode_diagnostic_is_opt_in_and_before_postprocess() -> None:
    namespace = runpy.run_path(str(REPO / "scripts" / "il_patch_lerobot.py"))
    patch_tuple = next(
        (module, old_candidates, new)
        for module, old_candidates, new in namespace["PATCHES"]
        if module == "lerobot.async_inference.policy_server"
        and "Tatbot predecode normalized output" in new
    )
    _module, old_candidates, replacement = patch_tuple

    assert 'environ.get("TATBOT_TRACE_PREDECODE") == "1"' in replacement
    assert replacement.index("action_tensor.detach()") < replacement.index("self.postprocessor")

    # Verify both relative and absolute action postprocessing branches exist and retain predecode tensor
    assert 'if getattr(self.policy.config, "use_relative_actions", False):' in replacement
    assert "self.postprocessor(action_tensor).squeeze(0)" in replacement
    assert replacement.count("_tatbot_normalized_action = action_tensor.detach().clone()") == 2

    # Verify candidate patch sites accept both un-postprocessed and postprocessed relative-action layers
    assert any("start_postprocess = time.perf_counter()" in old for old in old_candidates)


def test_groot_fixed_noise_seed_diagnostic_is_opt_in() -> None:
    namespace = runpy.run_path(str(REPO / "scripts" / "il_patch_lerobot.py"))
    replacement = next(
        new
        for module, _old, new in namespace["PATCHES"]
        if module == "lerobot.async_inference.policy_server"
        and "TATBOT_GROOT_FIXED_NOISE_SEED" in new
        and "torch.manual_seed" in new
    )

    assert replacement.index('environ.get("TATBOT_GROOT_FIXED_NOISE_SEED")') < replacement.index(
        "self.policy.predict_action_chunk"
    )
    assert 'environ.get("TATBOT_GROOT_NOISE_SEED_BASE")' in replacement
    assert "int(timestep)" in replacement
    assert "paired GR00T noise requires an observation timestep" in replacement
    assert "set only one of" in replacement

    call = next(
        new
        for module, _old, new in namespace["PATCHES"]
        if module == "lerobot.async_inference.policy_server"
        and "timestep=observation_t.get_timestep()" in new
        and "self._get_action_chunk" in new
    )
    assert "self._get_action_chunk" in call


def test_groot_denoising_grid_controls_are_opt_in_and_bounded() -> None:
    namespace = runpy.run_path(str(REPO / "scripts" / "il_patch_lerobot.py"))
    replacements = [
        new
        for module, _old, new in namespace["PATCHES"]
        if module == "lerobot.policies.groot.groot_n1_7"
    ]

    setup = next(new for new in replacements if "TATBOT_GROOT_INFERENCE_TIMESTEPS" in new)
    loop = next(new for new in replacements if "_tatbot_u0" in new)
    assert "else self.num_inference_timesteps" in setup
    assert '{"linear", "early_dense", "late_dense"}' in setup
    assert "_tatbot_u0**2" in loop
    assert "_tatbot_u0**0.5" in loop
    assert "dt = 1.0 / _tatbot_steps" in loop


def test_groot_server_rejects_invalid_denoising_controls(tmp_path: Path) -> None:
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    base_model = tmp_path / "base-model"
    base_model.mkdir()
    (checkpoint / "config.json").write_text(
        json.dumps({"type": "groot", "base_model_path": str(base_model)})
    )
    env_root = tmp_path / "env"
    (env_root / ".venv" / "bin").mkdir(parents=True)
    os.symlink(sys.executable, env_root / ".venv" / "bin" / "python")

    bad_steps = subprocess.run(
        [
            str(REPO / "scripts" / "eval" / "serve.sh"),
            "--policy",
            str(checkpoint),
            "--groot-inference-steps",
            "0",
            "--env-root",
            str(env_root),
        ],
        cwd=REPO,
        capture_output=True,
        text=True,
        check=False,
    )
    bad_schedule = subprocess.run(
        [
            str(REPO / "scripts" / "eval" / "serve.sh"),
            "--policy",
            str(checkpoint),
            "--groot-flow-schedule",
            "cosine",
            "--env-root",
            str(env_root),
        ],
        cwd=REPO,
        capture_output=True,
        text=True,
        check=False,
    )

    assert bad_steps.returncode == 2
    assert "positive integer" in bad_steps.stderr
    assert bad_schedule.returncode == 2
    assert "linear, early_dense, or late_dense" in bad_schedule.stderr


def test_groot_embodiment_mapping_is_canonical_and_fail_closed() -> None:
    namespace = runpy.run_path(str(REPO / "scripts" / "il_patch_lerobot.py"))
    replacements = [
        new
        for module, _old, new in namespace["PATCHES"]
        if module == "lerobot.policies.groot.processor_groot"
    ]
    merge = next(new for new in replacements if "projector_id in N1_7_EMBODIMENT_MAPPING" in new)
    fail_closed = next(new for new in replacements if "refusing to fall back to projector 0" in new)

    assert 'embodiment_mapping.get("new_embodiment") != 10' in merge
    assert "embodiment_mapping.setdefault(tag, projector_id)" in merge
    assert "self.embodiment_mapping[self.embodiment_tag]" in fail_closed
    assert ".get(self.embodiment_tag, 0)" not in fail_closed


def test_groot_state_dropout_is_applied_only_inside_the_action_head() -> None:
    namespace = runpy.run_path(str(REPO / "scripts" / "il_patch_lerobot.py"))
    replacement = next(
        new
        for module, _old, new in namespace["PATCHES"]
        if module == "lerobot.policies.groot.processor_groot"
        and "do not independently drop raw state here" in new
    )

    assert "state_dropout_prob=0.0" in replacement
    assert "checkpoint_assets.state_dropout_prob" not in replacement


def test_async_wire_uses_zero_copy_image_wrap_and_highest_pickle_protocol() -> None:
    namespace = runpy.run_path(str(REPO / "scripts" / "il_patch_lerobot.py"))
    patches = namespace["PATCHES"]

    image_replacement = next(
        new
        for module, _old, new in patches
        if module == "lerobot.async_inference.helpers" and "torch.as_tensor" in new
    )
    assert "torch.tensor" not in image_replacement

    pickle_replacements = [
        new
        for module, _old, new in patches
        if module in {
            "lerobot.async_inference.policy_server",
            "lerobot.async_inference.robot_client",
        }
        and "pickle.HIGHEST_PROTOCOL" in new
    ]
    assert len(pickle_replacements) == 3


def test_chunk_guard_is_before_action_serialization_and_opt_in() -> None:
    namespace = runpy.run_path(str(REPO / "scripts" / "il_patch_lerobot.py"))
    replacement = next(
        new
        for module, _old, new in namespace["PATCHES"]
        if module == "lerobot.async_inference.policy_server"
        and "Tatbot chunk passed demonstration plausibility contract" in new
    )

    assert 'environ.get(\n            "TATBOT_PLAUSIBILITY_CONTRACT"' in replacement
    assert "from chunk_guard import enforce_chunk" in replacement
    assert "raise RuntimeError" in replacement
    assert replacement.index("enforce_chunk(") < replacement.index("action_tensor.detach().cpu()")
    assert "plausibility guard requires relative-action policy" not in replacement
    absolute_capture = next(
        new
        for module, _old, new in namespace["PATCHES"]
        if module == "lerobot.async_inference.policy_server"
        and "processors historically accepted only" in new
        and "_tatbot_normalized_action = action_tensor.detach().clone()" in new
    )
    assert absolute_capture.index("_tatbot_normalized_action") < absolute_capture.index(
        "self.postprocessor(single_action)"
    )

    abort_replacement = next(
        new
        for module, _old, new in namespace["PATCHES"]
        if module == "lerobot.async_inference.policy_server"
        and "plausibility-guarded inference failed" in new
    )
    assert "grpc.StatusCode.FAILED_PRECONDITION" in abort_replacement
    assert 'environ.get("TATBOT_PLAUSIBILITY_CONTRACT")' in abort_replacement


def test_overlapping_policy_server_patches_have_idempotent_final_forms() -> None:
    namespace = runpy.run_path(str(REPO / "scripts" / "il_patch_lerobot.py"))
    replacements = [
        new
        for module, _old, new in namespace["PATCHES"]
        if module == "lerobot.async_inference.policy_server"
    ]
    history = next(new for new in replacements if "TATBOT_OBS_HISTORY bundle" in new)
    guard_state = next(new for new in replacements if "_tatbot_guard_state = observation.get" in new)
    predecode = next(new for new in replacements if "Tatbot predecode normalized output" in new)
    normalized_capture = next(
        new for new in replacements if "_tatbot_normalized_action = action_tensor.detach" in new
    )

    assert guard_state in history
    assert normalized_capture in predecode


def test_act_depth_expansion_expands_1channel_depth_to_3channel_and_leaves_3channel_rgb_untouched(
    tmp_path: Path, monkeypatch
) -> None:
    namespace = runpy.run_path(str(REPO / "scripts" / "il_patch_lerobot.py"))
    patch_tuple = next(
        (module, old_candidates, new)
        for module, old_candidates, new in namespace["PATCHES"]
        if module == "lerobot.policies.act.modeling_act"
    )
    _module_name, old_candidates, replacement = patch_tuple

    # 1. Verify patch application to candidate source code
    original_code = "def process(self, batch):\n" + old_candidates[0]
    source = tmp_path / "fake_modeling_act.py"
    source.write_text(original_code)
    module = types.ModuleType("tatbot_fake_modeling_act")
    module.__file__ = str(source)
    monkeypatch.setitem(sys.modules, module.__name__, module)

    fake_patches = [(module.__name__, old_candidates, replacement)]
    monkeypatch.setitem(namespace["main"].__globals__, "PATCHES", fake_patches)
    assert namespace["main"]() == 0

    # 2. Verify behavioral execution of the patched code
    passed_shapes: list[tuple[int, ...]] = []

    class FakeBackbone:
        def __call__(self, img: FakeTensor) -> dict[str, None]:
            passed_shapes.append(img.shape)
            return {"feature_map": None}

    class FakeModel:
        def __init__(self) -> None:
            self.backbone = FakeBackbone()

    class FakeTensor:
        def __init__(self, shape: tuple[int, ...]) -> None:
            self.shape = shape

        def expand(self, *dims: int) -> FakeTensor:
            new_shape = list(self.shape)
            for i, d in enumerate(dims):
                if d != -1:
                    new_shape[i] = d
            return FakeTensor(tuple(new_shape))

    img_1ch = FakeTensor((2, 1, 480, 640))
    img_3ch = FakeTensor((2, 3, 480, 640))
    batch = {"observation.images": [img_1ch, img_3ch]}

    patched_source = source.read_text()
    scope: dict[str, object] = {"OBS_IMAGES": "observation.images"}
    exec(patched_source, scope)
    model = FakeModel()
    scope["process"](model, batch)

    # Single-channel depth image expanded to 3 channels; 3-channel RGB image left untouched
    assert passed_shapes == [(2, 3, 480, 640), (2, 3, 480, 640)]
