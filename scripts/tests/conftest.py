"""Collection guard: the suite must run from a bare clone on any node.

Nine modules in here import a heavy optional dependency (directly, or through
the script they exercise). Without a guard they raise at *collection* time, and
one missing wheel takes down the whole run — `pytest -q scripts/tests/` reported
"9 errors during collection" and skipped every other test in the directory,
including ones that need nothing but the standard library.

Collection errors are indistinguishable from a red suite at a glance, which is
how the command printed in AGENTS.md stayed broken without anyone noticing. So
a module whose dependency is absent is *ignored and named* instead: the run goes
green on what it could actually check, and the header says what it could not.

Install the full set to run everything (see `scripts/tests/requirements.txt`):

    uvx --with-requirements scripts/tests/requirements.txt pytest -q scripts/tests/

or just `scripts/check tests`, which does that for you.
"""

from __future__ import annotations

import os
from importlib.util import find_spec
from pathlib import Path

import pytest

# test module stem -> every import that must resolve for it to be collectable.
# Lists, not single names: a module can clear its first missing import and hit
# another one behind it. Mapping only the first is how this table was wrong on
# its first draft -- installing opencv/pyarrow/torch simply moved the collection
# error to `lerobot` and `tensorboard`.
#
# Keep in sync with scripts/tests/requirements.txt. `lerobot` is deliberately
# NOT in that file: it is an entire robotics stack pulled from a git fork with
# a pinned trossen override, and dragging it into this suite's throwaway
# environment is not worth it. The three modules that need it are collected
# when the suite is run from the plugin's own venv, and skipped here.
OPTIONAL_DEPS: dict[str, tuple[str, ...]] = {
    "test_calibrate_board_session": ("cv2",),
    "test_ee_fiducial": ("cv2",),
    "test_ee_tracker_io": ("cv2",),
    "test_fiducial_inventory": ("cv2",),
    "test_il_analyze_rollout": ("numpy",),
    "test_ink_hook": ("numpy",),
    "test_palette_cal": ("numpy",),
    "test_train_dataset": ("pyarrow", "lerobot"),
    "test_train_depth_encoding": ("pyarrow", "lerobot"),
    "test_train_offline_eval": ("torch", "lerobot"),
    "test_train_sampler": ("torch",),
    "test_train_tb_bridge": ("torch", "tensorboard"),
}


def _absent(mods: tuple[str, ...]) -> list[str]:
    return [m for m in mods if find_spec(m) is None]


_missing = {stem: absent for stem, mods in OPTIONAL_DEPS.items() if (absent := _absent(mods))}

collect_ignore = [f"{stem}.py" for stem in _missing]

# --- deployment-configured tests --------------------------------------------
# Some modules exercise behaviour that only exists once a deployment has been
# described: its fleet (config/nodes.json), its hardware profile, its measured
# arm goldens, its ink/palette inventory. A public checkout has none of those,
# and a test that cannot run is a SKIP, not a failure — the same contract
# scripts/check keeps for a node that lacks a toolchain.
DEPLOYMENT_FILES: dict[str, tuple[str, ...]] = {
    "test_audio_route": ("config/audio/ee-input1.asoundrc",),
    "test_ee_fiducial": ("config/fiducials.json",),
    "test_estop_launchers": ("config/nodes.json",),
    "test_eval_checkpoint_contract": ("config/nodes.json",),
    "test_fiducial_inventory": ("config/fiducials.json",),
    "test_fuse_session": ("config/fiducials.json",),
    "test_il_dip": ("config/inks.yaml", "config/palette.yaml"),
    "test_ink_hook": ("config/inks.yaml", "config/palette.yaml"),
    "test_ink_spec": ("config/inks.yaml", "config/palette.yaml"),
    "test_profile": ("config/profiles/tatbot.json",),
    "test_runlog": ("config/nodes.json",),
    "test_staged_pose_single_source": ("config/trossen/tatbot.yaml",),
    "test_tool_spec": ("config/workspace.yaml",),
}

_repo_root = Path(__file__).resolve().parents[2]
_unconfigured = {
    stem: absent
    for stem, files in DEPLOYMENT_FILES.items()
    if (absent := [f for f in files if not (_repo_root / f).is_file()])
}
collect_ignore += [f"{stem}.py" for stem in _unconfigured]

# --- fleet-dependent tests --------------------------------------------------
# Most of the CLI suite is about grammar, gates, tiers and refusals, and runs
# anywhere. These exercise node routing, ssh targets or profile addresses, so
# they need config/nodes.json. Named explicitly: a new fleet test that forgets
# to list itself fails loudly in a public checkout rather than skipping
# silently, which is the right direction for that mistake.
NEEDS_FLEET = {
    "test_autonomous_verbs_need_a_literal_nonce",
    "test_dashdash_passthrough_reaches_the_launcher_untouched",
    "test_dip_plan_and_connect_only_command_nothing_and_need_no_nonce",
    "test_dip_refuses_a_tool_that_never_dips_and_a_bare_call",
    "test_dip_rehearse_is_a_moving_dip_with_a_rehearsal_tool",
    "test_dip_yes_with_a_real_needle_needs_allow_real",
    "test_dry_run_never_writes_the_arm_token",
    "test_hop_is_refused_for_autonomous_motion",
    "test_hop_refuses_a_node_without_a_checkout",
    "test_hop_uses_the_canonical_ssh_target_and_no_hop",
    "test_hostname_alias_resolves_to_the_node_name",
    "test_hub_python_prefers_the_plugin_venv_then_il_train_then_uv",
    "test_inkgen_ctl_verbs_construct_argv",
    "test_inkgen_role_auto_hop_and_sync",
    "test_inkgen_serve_options_pass_through",
    "test_inkmap_dev_and_build_options_pass_through",
    "test_live_cockpit_requires_viewer_role_and_passes_flags",
    "test_motion_verbs_refuse_estop_overrides_with_exit_3",
    "test_offline_verbs_run_anywhere",
    "test_record_dip_writes_the_nonce_the_dip_will_consume",
    "test_record_with_dip_is_autonomous_motion",
    "test_safe_passthrough_is_not_refused",
    "test_teleop_start_is_the_canonical_bare_teleop",
    "test_tool_from_environment_is_accepted",
    "test_tool_must_be_stated_for_motion_verbs",
    "test_train_manifest_render_uses_the_wrapped_tools_default_mode",
    "test_train_offline_eval_uses_the_pinned_training_environment",
    "test_unknown_on_node_is_exit_2",
    "test_write_nonce_is_what_the_launcher_reads",
    "test_wrong_node_is_exit_4_with_the_on_form",
}


# Verbs whose backing scripts are private (fleet deploy, dataset hub) do not
# exist in a public checkout, and the CLI hides them there on purpose. Tests
# that assert those verbs — or read private CLI bookkeeping — only make sense
# where that tooling is present.
NEEDS_PRIVATE_TOOLING = {
    "test_cli_shim_is_not_orphan_or_entry_point",
    "test_data_push_dry_run_names_the_interpreter",
    "test_ee_tool_is_the_flag_every_python_tool_accepts",
    "test_every_verb_has_one_tier_and_an_example_that_dry_runs",
    "test_inkgen_deploy_constructs_deploy_script_argv",
    "test_inkgen_verbs_have_honest_tiers",
    "test_inkmap_deploy_constructs_deploy_script_argv",
    "test_inkmap_verbs_have_honest_tiers",
    "test_shims_delegate_to_the_cli",
}


def pytest_collection_modifyitems(config, items):  # noqa: ANN001 - pytest hook
    no_fleet = not (_repo_root / "config" / "nodes.json").is_file()
    no_private = not (_repo_root / "config" / "cli-orphans.txt").is_file()
    fleet_skip = pytest.mark.skip(reason="needs a described fleet (config/nodes.json)")
    tooling_skip = pytest.mark.skip(reason="needs the private fleet/deploy tooling")
    for item in items:
        name = item.name.split("[")[0]
        if no_fleet and name in NEEDS_FLEET:
            item.add_marker(fleet_skip)
        elif no_private and name in NEEDS_PRIVATE_TOOLING:
            item.add_marker(tooling_skip)



# Fiducial tests parse an inventory; the deployment's own (config/fiducials.json)
# is not in a public checkout, so fall back to the synthetic example. This is a
# TEST-ONLY fallback: runtime never substitutes an example for measured data.
_repo = Path(__file__).resolve().parents[2]
if not (_repo / "config" / "fiducials.json").is_file():
    os.environ.setdefault(
        "TATBOT_FIDUCIAL_CONFIG", str(_repo / "config" / "examples" / "fiducials.json"))



def pytest_report_header(config) -> str | None:  # noqa: ANN001 - pytest hook signature
    """Name what was not checked, so a green run is never mistaken for a full one."""
    lines = []
    if _unconfigured:
        files = sorted({f for absent in _unconfigured.values() for f in absent})
        lines.append(
            f"tatbot: skipping {len(_unconfigured)} module(s) that need a configured "
            f"deployment ({', '.join(files)}) -- see config/examples/")
    if not _missing:
        return "\n".join(lines) or None
    by_dep: dict[str, list[str]] = {}
    for stem, mods in sorted(_missing.items()):
        for mod in mods:
            by_dep.setdefault(mod, []).append(stem)
    parts = [f"{mod} ({len(stems)})" for mod, stems in sorted(by_dep.items())]
    hint = "see scripts/tests/requirements.txt"
    if all("lerobot" in mods for mods in _missing.values()):
        # requirements.txt deliberately omits lerobot; pointing at it would be
        # advice that cannot be followed.
        hint = "run from python/lerobot_robot_tatbot/.venv to include these"
    lines.append(
        f"tatbot: skipping {len(_missing)} module(s), missing {', '.join(parts)} -- {hint}"
    )
    return "\n".join(lines)
