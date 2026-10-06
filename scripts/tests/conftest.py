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

    uvx --python 3.12 --with-requirements scripts/tests/requirements.txt pytest -q scripts/tests/

or just `scripts/check tests`, which does that for you.
"""

from __future__ import annotations

import os
import sys
from functools import cache
from importlib.util import find_spec
from pathlib import Path

import pytest

# The suite's one path setup. Test modules import repo modules by bare name
# (`import pen_path`), which needs every script root on sys.path; pytest
# loads this conftest before any of them, so no test file carries its own
# sys.path arithmetic. scripts/lib itself has to be found by hand, once.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "lib"))
from tatbot_paths import bootstrap  # noqa: E402

bootstrap()
_repo_root = Path(__file__).resolve().parents[2]
# These ROS build roots provide pure compiler/geometry libraries to root tests;
# loading their submitted source needs neither a ROS install nor an executor.
_SOURCE_ROOTS = [
    _repo_root / "python/tatbot_contracts/src",
    *(_repo_root / "ros" / name for name in ("tatbot_ink", "tatbot_motion", "tatbot_description")),
]
for root in _SOURCE_ROOTS:
    sys.path.insert(0, str(root))

# Every import a test module needs at collection time is read from its source,
# transitively through the repo's own modules (scripts/lib, scripts/vision, the
# python/ roots ...), and checked with find_spec. A hand-kept table lived here
# before: it had to be edited for every new test module, and three modules added
# in one afternoon were not, so the light profile failed collection on `yaml`
# and `cv2` for tests that had nothing to do with either. Derived, it cannot go
# stale. Imports inside `try:` and `if TYPE_CHECKING:` are not collection
# requirements and are not counted.
_SEARCH_ROOTS = [
    _repo_root / "scripts" / "tests",
    _repo_root / "scripts" / "lib",
    _repo_root / "scripts" / "vision",
    _repo_root / "scripts" / "train",
    _repo_root / "scripts" / "eval",
    _repo_root / "scripts",
    *sorted((_repo_root / "python").glob("*/src")),
    *_SOURCE_ROOTS,
]


@cache
def _resolve_repo_module(name: str, here: Path | None = None) -> Path | None:
    """The repo file `import name` would load, searching the roots tests add to sys.path.

    A directory without `__init__.py` (scripts/lib is one) is a namespace
    package and resolves to itself; only files are read for further imports.
    """
    parts = name.split(".")
    roots = ([here] if here else []) + _SEARCH_ROOTS
    for root in roots:
        # Walk the dotted path as far as it names modules; a trailing segment
        # that is a symbol rather than a module resolves to its package.
        found = None
        base = root
        for part in parts:
            if (base / f"{part}.py").is_file():
                found = base / f"{part}.py"
            elif (base / part / "__init__.py").is_file():
                found = base / part / "__init__.py"
            elif (base / part).is_dir() and any((base / part).glob("*.py")):
                found = base / part
            else:
                break
            base = base / part
        if found is not None:
            return found
    return None


@cache
def _top_level_imports(path: Path) -> list[tuple[str, int]]:
    """(name, level) for each import that runs when the module is loaded."""
    import ast

    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    out: list[tuple[str, int]] = []

    def walk(stmts):
        for node in stmts:
            if isinstance(node, ast.Import):
                out.extend((alias.name, 0) for alias in node.names)
            elif isinstance(node, ast.ImportFrom):
                if node.level:
                    for alias in node.names:
                        out.append(((node.module + "." if node.module else "") + alias.name, node.level))
                else:
                    out.extend((f"{node.module}.{alias.name}", 0) for alias in node.names)
            elif isinstance(node, ast.If):
                if "TYPE_CHECKING" not in ast.dump(node.test):
                    walk(node.body)
                walk(node.orelse)
            elif isinstance(node, (ast.With, ast.For, ast.While)):
                walk(node.body)
            # ast.Try: a guarded import is optional by construction.
            # Function and class bodies run later than collection.

    walk(tree.body)
    return out


def _collection_imports(path: Path, seen: set[Path]) -> set[str]:
    """Third-party top-level modules `path` imports when loaded, through repo modules."""
    if path in seen:
        return set()
    seen.add(path)
    needed: set[str] = set()
    for name, level in _top_level_imports(path):
        if level:
            pkg = path.parent
            for _ in range(level - 1):
                pkg = pkg.parent
            target = _resolve_repo_module(name, pkg)
            if target is not None and target.is_file():
                needed |= _collection_imports(target, seen)
            continue
        top = name.split(".")[0]
        if top in sys.stdlib_module_names or top == "pytest":
            continue
        target = _resolve_repo_module(name)
        if target is None:
            needed.add(top)
        elif target.is_file():
            needed |= _collection_imports(target, seen)
    return needed


def _runtime_imports(path: Path) -> set[str]:
    """Modules a test declares with `pytest.importorskip(...)` anywhere in its body.

    Those are run-time needs -- the test skips itself, one at a time, with a
    reason -- but a profile that promises full coverage must still install them.
    """
    import ast

    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    names = {
        node.args[0].value
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "importorskip"
        and node.args
        and isinstance(node.args[0], ast.Constant)
        and isinstance(node.args[0].value, str)
    }
    # A repo module skipped this way is private tooling absent from the public
    # export, not a dependency any profile installs.
    return {name.split(".")[0] for name in names if _resolve_repo_module(name) is None}


def _absent(mods) -> list[str]:
    return [m for m in mods if find_spec(m) is None]


_TEST_MODULES = sorted(Path(__file__).resolve().parent.glob("test_*.py"))
_COLLECTION_IMPORTS = {path.stem: _collection_imports(path, set()) for path in _TEST_MODULES}
_RUNTIME_IMPORTS = {path.stem: _runtime_imports(path) for path in _TEST_MODULES}
# Shared imports recur across test modules. Cache only this discovery pass:
# tests can create files or change search roots after collection.
_top_level_imports.cache_clear()
_resolve_repo_module.cache_clear()

_missing = {stem: absent for stem, mods in _COLLECTION_IMPORTS.items() if (absent := _absent(sorted(mods)))}

collect_ignore = [f"{stem}.py" for stem in _missing]

# --- deployment-configured tests --------------------------------------------
# Some modules exercise behaviour that only exists once a deployment has been
# described: its fleet (config/nodes.json), its hardware profile, its measured
# arm goldens, its ink/palette inventory. A public checkout has none of those,
# and a test that cannot run is a SKIP, not a failure — the same contract
# scripts/check keeps for a node that lacks a toolchain.
DEPLOYMENT_FILES: dict[str, tuple[str, ...]] = {
    "test_ee_fiducial": ("config/fiducials.json",),
    "test_estop_launchers": ("config/nodes.json",),
    "test_eval_checkpoint_contract": ("config/nodes.json",),
    "test_fiducial_inventory": ("config/fiducials.json",),
    "test_palette_scan_entry": ("config/fiducials.json",),
    "test_ink_hook": ("config/inks.yaml", "config/palette.yaml"),
    "test_ink_spec": ("config/inks.yaml", "config/palette.yaml"),
    "test_arm_context": ("config/trossen/leader.yaml",),
    "test_draw_research": ("config/workspace.yaml",),
    "test_fleet_service_units": ("config/nodes.json", "config/systemd/tatbot-trackd.service"),
    "test_native_registration": ("config/workspace.yaml", "config/fiducials.json"),
    "test_palette_geometry": ("config/palette_geometry.json", "config/palette.yaml"),
    "test_probe_calibration_adopt": ("config/workspace.yaml",),
    "test_ros_palette": ("config/palette.yaml",),
    "test_teleop_poses": ("config/workspace.yaml",),
    "test_wrist_mount_geometry": ("rust/visiond/config/vision.toml",),
    "test_profile": ("config/profiles/tatbot.json",),
    "test_runlog": ("config/nodes.json",),
    "test_staged_pose_single_source": ("config/trossen/tatbot.yaml",),
    "test_tool_spec": ("config/workspace.yaml",),
}

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
    "test_arm_recover_defaults_to_both_and_preserves_explicit_single_arm_forms",
    "test_arm_recover_ip_override_requires_an_explicit_role",
    "test_autonomous_verbs_get_a_launch_id_and_tag_is_an_optional_label",
    "test_dashdash_passthrough_reaches_the_launcher_untouched",
    "test_draw_and_compile_dry_run",
    "test_draw_and_dip_are_gone_and_their_kept_tools_moved",
    "test_dry_run_never_writes_the_arm_token",
    "test_hop_carries_autonomous_motion_over_a_tty",
    "test_hop_refuses_a_node_without_a_checkout",
    "test_hop_uses_the_canonical_ssh_target_and_no_hop",
    "test_hostname_alias_resolves_to_the_node_name",
    "test_hub_python_prefers_the_plugin_venv_then_il_train_then_uv",
    "test_inkgen_ctl_verbs_construct_argv",
    "test_inkgen_role_auto_hop_and_sync",
    "test_inkgen_serve_options_pass_through",
    "test_inkmap_dev_options_pass_through",
    "test_node_is_list_and_run",
    "test_node_list_one_node_and_run_interactive",
    "test_live_cockpit_requires_viewer_role_and_passes_flags",
    "test_live_cockpit_surface_and_poe_only_flags",
    "test_viewer_view_streams_from_the_node_that_holds_the_file",
    "test_vision_deploy_reaches_every_deploy_target",
    "test_vision_track_and_monitor_verify_hop_to_the_camera_node",
    "test_motion_verbs_refuse_estop_overrides_with_exit_3",
    "test_offline_verbs_run_anywhere",
    "test_pattern_names_one_installed_print_by_the_digits_its_sheet_prints",
    "test_teleop_network_trace_is_passive_and_routes_to_arm_owner",
    "test_teleop_check_is_read_only_and_routes_to_the_arm_owner",
    "test_teleop_check_reports_observations_without_opening_any_device",
    "test_omitting_the_tool_uses_the_configured_one_and_names_its_source",
    "test_an_intentional_environment_tool_crosses_the_hop_as_an_explicit_flag",
    "test_a_remote_default_is_the_owners_and_planning_never_asks_that_node",
    "test_a_mistyped_tool_is_refused_before_an_ssh_is_spent",
    "test_cli_inspection_records_go_and_has_no_pose_target",
    "test_cli_overhead_capture_is_an_explicit_authorized_native_hold",
    "test_record_is_human_motion_with_an_ink_opt_out_and_no_scripted_dip",
    "test_remote_motion_json_refusal_is_one_object_before_any_exec",
    "test_safe_passthrough_is_not_refused",
    "test_sim_compile_placement_takes_the_stated_tool",
    "test_sim_compile_sample_delegates_to_bounded_offline_materializer",
    "test_sim_is_six_verbs",
    "test_sim_preview_reach_cinematic_viewer_audit_samples",
    "test_tags_scan_hops_to_the_camera_node",
    "test_teleop_start_is_the_canonical_bare_teleop",
    "test_tool_from_environment_is_accepted",
    "test_tool_must_be_stated_for_motion_verbs",
    "test_train_manifest_render_uses_the_wrapped_tools_default_mode",
    "test_train_offline_eval_uses_the_pinned_training_environment",
    "test_unknown_on_node_is_exit_2",
    "test_uv_setup_and_remote_native_effects_are_visible_in_plans",
    "test_write_launch_id_is_what_the_launcher_reads",
    "test_wrong_node_is_exit_4_with_the_on_form",
}


# Verbs whose backing scripts are private (fleet deploy, dataset hub) do not
# exist in a public checkout, and the CLI hides them there on purpose. Tests
# that assert those verbs, read private CLI bookkeeping, or read one measured
# file the rest of their module does not need, only make sense where it exists.
NEEDS_PRIVATE_TOOLING: dict[str, tuple[str, ...]] = {
    "test_data_hub_dry_run_names_the_interpreter": ("scripts/dataset_hub.py",),
    "test_inkgen_deploy_constructs_deploy_script_argv": ("scripts/inkgen_deploy.sh",),
    "test_inkmap_deploy_constructs_deploy_script_argv": ("scripts/inkmap_deploy.sh",),
    "test_identities_resolve_each_label_to_its_arm_controller_and_tag_triplet": ("config/fiducials.json",),
    "test_no_tags_is_retained_without_a_calibration_claim": ("config/fiducials.json",),
    "test_camera_unit_uses_the_selected_manifest_node": ("config/systemd/tatbot-visiond-d405.service",),
    "test_every_viewer_server_generation_installs_one_fixed_blueprint": ("config/systemd/tatbot-viewer@.service",),
    "test_replaced_carriers_use_the_current_measured_tag_frames": ("config/wrist_tags_measured_left.json",),
    "test_both_installed_attachments_follow_their_own_carriage_and_camera": ("cad/leader-laser-v1/build.py",),
    "test_session_capture_names_the_arm_whose_wrist_cameras_answer": ("rust/visiond/config/vision.toml",),
    "test_live_manifest_assigns_distinct_wrist_arms_and_owners": ("rust/visiond/config/vision.toml",),
}

# These compare generated paths or compiled constants with the deployment's
# measured touch-off. The example workspace is intentionally not substituted:
# doing that would turn placeholder geometry into apparent qualification.
NEEDS_WORKSPACE = {
    "test_ballpoint_tip_matches_urdf_tool_mount_plus_pen_offset",
    "test_tip_constant_matches_workspace_derivation",
    "test_right_arm_model_reproduces_the_module_constants",
    "test_left_arm_model_is_the_mirrored_leader_from_the_urdf",
    "test_leader_laser_is_modelled_as_a_standoff_tool_at_its_measured_face",
    "test_rollout_tool_default_comes_from_left_workspace",
    "test_bundled_native_acquisition_keeps_source_recipe_pen_and_traversal",
    "test_precedence_is_flag_then_environment_then_configured",
    "test_posed_witness_refuses_unbound_camera_and_configuration",
}

_no_fleet = not (_repo_root / "config" / "nodes.json").is_file()
_no_workspace = not (_repo_root / "config" / "workspace.yaml").is_file()
_missing_tooling = {
    name: [path for path in paths if not (_repo_root / path).is_file()]
    for name, paths in NEEDS_PRIVATE_TOOLING.items()
}
_missing_tooling = {name: paths for name, paths in _missing_tooling.items() if paths}


def pytest_collection_modifyitems(config, items):  # noqa: ANN001 - pytest hook
    fleet_skip = pytest.mark.skip(reason="needs a described fleet (config/nodes.json)")
    workspace_skip = pytest.mark.skip(reason="needs a measured touch-off (config/workspace.yaml)")
    for item in items:
        name = item.name.split("[")[0]
        if _no_fleet and name in NEEDS_FLEET:
            item.add_marker(fleet_skip)
        elif name in _missing_tooling:
            missing = ", ".join(_missing_tooling[name])
            item.add_marker(pytest.mark.skip(reason=f"needs files a public checkout omits ({missing})"))
        elif _no_workspace and name in NEEDS_WORKSPACE:
            item.add_marker(workspace_skip)



# Fiducial tests parse an inventory; the deployment's own (config/fiducials.json)
# is not in a public checkout, so fall back to the synthetic example. This is a
# TEST-ONLY fallback: runtime never substitutes an example for measured data.
_repo = Path(__file__).resolve().parents[2]
if not (_repo / "config" / "fiducials.json").is_file():
    os.environ.setdefault(
        "TATBOT_FIDUCIAL_CONFIG", str(_repo / "config" / "examples" / "fiducials.json"))

# The rig's power marker (~/tatbot-logs/rig/state.json) is live fleet state on
# the node running the suite, and every CLI test spawns `tatbot` with a copy of
# this environment. With the rig asleep, the rig gate answered first — exit 5,
# "tatbot rig wake" — where 16 tests expected the estop, profile or tool gate
# they were exercising, and the suite went red overnight and green at breakfast.
# Point the marker at a path that does not exist: no test here may see the
# real one, and the tests OF the marker (test_cli_rig) set their own.
os.environ["TATBOT_RIG_STATE"] = "/nonexistent/tatbot-rig-state.json"



def pytest_report_header(config) -> str | None:  # noqa: ANN001 - pytest hook signature
    """Name what was not checked, so a green run is never mistaken for a full one."""
    def selected(stem):
        target = _repo_root / "scripts" / "tests" / f"{stem}.py"
        return any(
            (path := Path(arg.split("::", 1)[0]).resolve()) == target or path in target.parents
            for arg in config.args
        )

    missing = {stem: deps for stem, deps in _missing.items() if selected(stem)}
    unconfigured = {stem: files for stem, files in _unconfigured.items() if selected(stem)}
    lines = []
    if unconfigured:
        files = sorted({f for absent in unconfigured.values() for f in absent})
        lines.append(
            f"tatbot: skipping {len(unconfigured)} module(s) that need a configured "
            f"deployment ({', '.join(files)}) -- see config/examples/")
    if _no_fleet:
        lines.append(f"tatbot: skipping {len(NEEDS_FLEET)} item(s) that need config/nodes.json")
    if _missing_tooling:
        files = sorted({f for absent in _missing_tooling.values() for f in absent})
        lines.append(
            f"tatbot: skipping {len(_missing_tooling)} item(s) for omitted deployment tooling "
            f"({', '.join(files)})")
    if _no_workspace:
        lines.append(
            f"tatbot: skipping {len(NEEDS_WORKSPACE)} item(s) that compare measured touch-off geometry")
    if not missing:
        return "\n".join(lines) or None
    by_dep: dict[str, list[str]] = {}
    for stem, mods in sorted(missing.items()):
        for mod in mods:
            by_dep.setdefault(mod, []).append(stem)
    parts = [f"{mod} ({len(stems)})" for mod, stems in sorted(by_dep.items())]
    hint = "see scripts/tests/requirements.txt"
    if all("lerobot" in mods for mods in missing.values()):
        # requirements.txt deliberately omits lerobot; pointing at it would be
        # advice that cannot be followed.
        hint = "run from python/lerobot_robot_tatbot/.venv to include these"
    lines.append(
        f"tatbot: skipping {len(missing)} module(s), missing {', '.join(parts)} -- {hint}"
    )
    return "\n".join(lines)


def pytest_terminal_summary(terminalreporter, exitstatus, config) -> None:
    """Quiet pytest runs still disclose modules omitted during collection."""
    report = pytest_report_header(config)
    if report:
        for line in report.splitlines():
            terminalreporter.write_line(line)
    for line in _budget_lines():
        terminalreporter.write_line(line)


_OWNED_LOG_ROOT = "_TATBOT_TEST_LOG_ROOT"


def pytest_configure(config) -> None:  # noqa: ANN001 - pytest hook
    config.addinivalue_line(
        "markers", "slow: over the per-test budget; the fast tier (bare scripts/check) skips it")
    # Tests that drive real scripts (the calibration driver, the run-log
    # writers) land in log_root() unless they set TATBOT_LOG_ROOT themselves.
    # On a rig that is ~/tatbot-logs: shared by every worker of a parallel run
    # and by concurrent runs, and the calibration solve lock there made two
    # runs fail each other (2026-09-16). Every pytest process gets its own
    # root; a worker re-derives one because it inherits the controller's.
    if not os.environ.get("TATBOT_LOG_ROOT") or os.environ.get(_OWNED_LOG_ROOT):
        import shutil
        import tempfile

        worker = getattr(config, "workerinput", {}).get("workerid", "main")
        root = tempfile.mkdtemp(prefix=f"tatbot-test-logs-{worker}-")
        os.environ["TATBOT_LOG_ROOT"] = root
        os.environ[_OWNED_LOG_ROOT] = "1"
        config.add_cleanup(lambda: shutil.rmtree(root, ignore_errors=True))


def pytest_sessionstart(session) -> None:
    """A named profile must install its required dependencies, or fail."""
    profile = os.environ.get("TATBOT_TEST_PROFILE")
    if profile == "offline":
        required = set().union(*_COLLECTION_IMPORTS.values(), *_RUNTIME_IMPORTS.values()) - {"lerobot"}
    elif profile == "integration":
        required = {"lerobot", "pyarrow", "torch"}
    elif profile == "light":
        required = {"numpy", "scipy"}
    elif profile is None:
        return
    else:
        raise pytest.UsageError(f"Unknown TATBOT_TEST_PROFILE: {profile}")
    missing = _absent(sorted(required))
    if missing:
        raise pytest.UsageError(f"{profile} profile missing required dependencies: {', '.join(missing)}")


# --- per-test time budget ---------------------------------------------------
# Nothing governed what one test may cost while the suite grew 3.5x in the
# first half of September 2026 (786 -> 2,750 functions), and the default check
# drifted from 7 to 14 minutes without anyone deciding that. Like the
# complexity ratchet, a passing test whose call phase exceeds the budget fails
# the job: mark it `slow` (the fast tier then skips it; --full and CI still
# run it) or make it cheaper. Wall time, measured on the controller only, so
# a parallel run counts each test once -- and inflates it: a 30 s test reads
# ~55 s beside seven busy workers, which is why the budget is 60. TATBOT_TEST_BUDGET_S=0 disables it; scripts/check
# does that on hosted runners, whose shared vCPUs are not this budget's clock.
TEST_BUDGET_S = float(os.environ.get("TATBOT_TEST_BUDGET_S", "60"))
_over_budget: list[tuple[float, str]] = []


def _is_worker(config) -> bool:  # noqa: ANN001 - pytest config
    return hasattr(config, "workerinput")


def pytest_runtest_logreport(report) -> None:  # noqa: ANN001 - pytest hook
    if (TEST_BUDGET_S > 0 and report.when == "call" and report.passed
            and report.duration > TEST_BUDGET_S and "slow" not in report.keywords):
        _over_budget.append((report.duration, report.nodeid))


def _budget_lines() -> list[str]:
    if not _over_budget:
        return []
    lines = [f"tatbot: {len(_over_budget)} passing test(s) over the {TEST_BUDGET_S:.0f} s budget -- "
             "mark @pytest.mark.slow or make cheaper (TATBOT_TEST_BUDGET_S=0 disables):"]
    lines += [f"  {duration:6.1f}s {nodeid}" for duration, nodeid in sorted(_over_budget, reverse=True)]
    return lines


def pytest_sessionfinish(session, exitstatus) -> None:  # noqa: ANN001 - pytest hook
    if _over_budget and not _is_worker(session.config):
        session.exitstatus = max(int(exitstatus), 1)
