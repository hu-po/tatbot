"""Explicit public simulator configuration, isolated from deployment files."""
from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
from pathlib import Path

import pytest

# Some test modules import scripts/vision modules (surface_attachment, ...) by
# bare name before anything has imported tatbot_sim, whose package __init__ is
# what puts the script roots on sys.path. Do it here, before collection.
from tatbot_paths import bootstrap  # noqa: E402

bootstrap(Path(os.environ.get("TATBOT_REPO") or Path(__file__).resolve().parents[3]))


# File-level partitions prevent pytest importing an unavailable engine during
# collection. New numerical tests default to geometry; explicit runtime tests
# belong in one of the engine/render sets below.
CONTRACT_TESTS = {
    'test_repo.py', 'test_resolved_config.py', 'test_distributions.py',
    'test_temporal_labels.py',
    'test_shared_artifact_helpers.py',
    'test_human_rep_boundary.py', 'test_human_rep_contracts.py', 'test_inkmap_contracts.py',
    'test_public_profile.py', 'test_palette_scene.py',
    'test_writer_discard.py', 'test_sheet_cache.py',
}
ENGINE_TESTS = {
    'test_artwork_collection.py', 'test_ink_dips.py', 'test_substrate.py',
    'test_episode_variants.py',
}
RENDER_TESTS = {
    'test_approach_start_pose.py', 'test_design_dataset.py', 'test_dynamic_texture.py',
    'test_example_dataset.py', 'test_root_pin.py', 'test_backend_construction.py',
    'test_episode_runtime.py', 'test_native_episode.py', 'test_palette_render.py',
}


def capability(path):
    name = Path(path).name
    if name in CONTRACT_TESTS:
        return 'contracts'
    if name in RENDER_TESTS:
        return 'render'
    if name in ENGINE_TESTS:
        return 'engine'
    return 'geometry'


def pytest_ignore_collect(collection_path, config):
    selected = config.getoption('--sim-capability')
    if selected != 'all' and collection_path.name.startswith('test_') and collection_path.suffix == '.py':
        return capability(collection_path) != selected
    return None


def _prepare_public_checkouts(source: Path, root: Path, *, build_planner: bool) -> None:
    """Give disposable public trees the checkout and planner artifacts tests expect."""
    for checkout in (source, root):
        if (checkout / ".git").exists():
            continue
        subprocess.run(["git", "init", "-q", str(checkout)], check=True, timeout=30)
        subprocess.run(["git", "-C", str(checkout), "add", "-A"], check=True, timeout=60)
        subprocess.run(["git", "-C", str(checkout), "-c", "user.name=Tatbot CI",
                        "-c", "user.email=ci@localhost", "commit", "-q", "-m",
                        "public validation snapshot"], check=True, timeout=60)
    if not build_planner:
        return
    subprocess.run(["cmake", "-S", str(root / "cpp/teleop"), "-B",
                    str(root / "cpp/teleop/build")], check=True, timeout=60)
    subprocess.run(["cmake", "--build", str(root / "cpp/teleop/build"),
                    "--target", "path_plan_check", "-j2"], check=True, timeout=120)


def _pending_wrist_layout(inventory: Path) -> str:
    """A wrist-tag layout that has never been calibrated, pinned to `inventory`.

    What tatbot_sim.urdf accepts without tag poses: the inventory's raw-file
    hash, ids, edge and parent, and `pending_recalibration` with no tags.
    """
    import hashlib
    import json

    raw = inventory.read_bytes()
    wrist = json.loads(raw)["targets"]["wrist"]
    return json.dumps({
        "//": "Synthetic: the public validation copy has never calibrated its wrist. Written by conftest.py.",
        "schema_version": 2, "calibration_status": "pending_recalibration",
        "generated_utc": "1970-01-01T00:00:00Z", "inventory_hash": hashlib.sha256(raw).hexdigest(),
        "target_ids": list(wrist["ids"]), "edge_m": wrist["edge_m"], "parent_frame": wrist["parent_frame"],
        "source": "synthetic", "source_link": wrist["parent_frame"], "tags": {},
    }, indent=2) + "\n"


def pytest_addoption(parser):
    parser.addoption("--sim-capability", choices=('all', 'contracts', 'geometry', 'engine', 'render'),
                     default='all', help="collect only this dependency boundary before importing tests")
    parser.addoption("--sim-profile", choices=("checkout", "public"), default="checkout",
                     help="public uses isolated synthetic configuration; checkout preserves deployment tests")
    parser.addoption("--sim-render", action="store_true", help="run tests requiring a render device")


def pytest_configure(config):
    config.addinivalue_line("markers", "field_calibration: requires the deployment's measured evidence")
    config.addinivalue_line("markers", "slow: over the per-test budget; excluded by scripts/check sim-fast")
    # pytest-timeout is a runtime dependency of the sim job rather than the
    # package, so register the mark to keep an ad-hoc pytest run warning-free.
    config.addinivalue_line("markers", "timeout: per-test timeout (pytest-timeout)")
    if config.getoption("--sim-render") and config.getoption("--sim-capability") in ('all', 'render'):
        # SAPIEN must enable GPU PhysX before the first CPU scene/material too.
        # A full render pass includes CPU worlds before the batched GPU cases;
        # engine-free partitions must keep their existing dependency boundary.
        import torch

        if torch.cuda.is_available():
            from sapien import physx

            physx.enable_gpu()
    if config.getoption("--sim-profile") != "public":
        return
    source = Path(__file__).resolve().parents[3]
    temporary = tempfile.TemporaryDirectory(prefix="tatbot-public-sim-")
    root = Path(temporary.name)
    config.add_cleanup(temporary.cleanup)
    _copy_public_source(source, root)
    _seed_public_config(source, root)
    # A public checkout has Git metadata and builds the portable C++ planner.
    # Recreate both facts only in disposable validation copies; neither the
    # candidate nor private calibration is modified or imported.
    _prepare_public_checkouts(source, root, build_planner=config.getoption("--sim-capability") != "contracts")
    patch = pytest.MonkeyPatch()
    config.add_cleanup(patch.undo)
    patch.syspath_prepend(str(root / "scripts/lib"))
    patch.syspath_prepend(str(root / "python/tatbot_contracts/src"))
    patch.syspath_prepend(str(root / "python/tatbot_sim/src"))
    patch.setenv("PYTHONPATH", os.pathsep.join([
        str(root / "python/tatbot_sim/src"), str(root / "python/tatbot_contracts/src"),
        str(root / "scripts/lib"),
        os.environ.get("PYTHONPATH", ""),
    ]))
    patch.setenv("TATBOT_REPO", str(root))
    patch.setenv("TATBOT_TOOL_ID", "lutin-ballpoint-dot")
    patch.delenv("TATBOT_SIM_TIP_DELTA_M", raising=False)


def _copy_public_source(source: Path, root: Path) -> None:
    """The source trees a public checkout has, plus the environments it borrows."""
    # Copy source bytes so compiler provenance and __file__ resolve within this
    # isolated root too. Never let tests write through links into the checkout.
    ignored = shutil.ignore_patterns(".git", ".venv", "node_modules", "__pycache__",
                                     "build", "_build", "target", "dist", ".pytest_cache", ".ruff_cache")
    for name in ("python", "scripts", "urdf", "web", "cpp", "rust", "firmware", "docs",
                 "LICENSE", "README.md", "THIRD_PARTY.md", "pyproject.toml"):
        child = source / name
        if not child.exists():
            continue
        if child.is_dir():
            shutil.copytree(child, root / child.name, ignore=ignored)
        else:
            shutil.copy2(child, root / child.name)
    dependencies = source / "web/inkmap/node_modules"
    if dependencies.is_dir():
        (root / "web/inkmap/node_modules").symlink_to(dependencies, target_is_directory=True)
    # The copy borrows the checkout's python/tatbot_sim/.venv beside it.
    environment = source / "python/tatbot_sim/.venv"
    if environment.is_dir():
        (root / "python/tatbot_sim/.venv").symlink_to(environment, target_is_directory=True)


def _seed_public_config(source: Path, root: Path) -> None:
    """The config a public checkout has: portable inputs, examples, nothing measured."""
    cfg = root / "config"
    cfg.mkdir()
    # Allowlist portable inputs; never inherit private bench configuration.
    for name in ("arms.json", "arm-labels.json", "body-models", "examples", "human-representation", "inkmap",
                 "tools", "substrates.yaml", "motion_constants.json"):
        src = source / "config" / name
        if src.is_dir():
            shutil.copytree(src, cfg / name)
        else:
            shutil.copy2(src, cfg / name)
    fixtures = source / "config/examples/simulator"
    for name in ("workspace.yaml", "palette.yaml", "inks.yaml", "palette_load.yaml",
                 "palette_geometry.json"):
        shutil.copy2(fixtures / name, cfg / name)
    # The Rust workspace compiles two private files in: tatbot_bus::fleet reads
    # config/nodes.json (the export manifest's public example is what a checkout
    # without a fleet builds against) and the simulator the wrist-tag layout. The
    # simulator refuses a layout that disagrees with its inventory, so the copy
    # gets an empty layout pending calibration, pinned to the inventory the copy
    # resolves -- exactly a checkout that has never calibrated its wrist.
    shutil.copy2(source / "config/examples/nodes.json", cfg / "nodes.json")
    (cfg / "wrist_tags_measured.json").write_text(_pending_wrist_layout(cfg / "examples/fiducials.json"))
    (cfg / "trossen").mkdir()
    shutil.copy2(fixtures / "arm.yaml", cfg / "trossen/tatbot.yaml")
    shutil.copy2(fixtures / "follower.yaml", cfg / "trossen/follower.yaml")


def pytest_report_header(config):
    return f"sim profile: {config.getoption('--sim-profile')} (public fixtures are never calibration evidence)"


def pytest_collection_modifyitems(config, items):
    if config.getoption("--sim-profile") != "public":
        return
    for item in items:
        if item.get_closest_marker("field_calibration"):
            item.add_marker(pytest.mark.skip(reason="public profile has no measured field calibration"))
        if not config.getoption("--sim-render") and capability(item.path) == "render":
            item.add_marker(pytest.mark.skip(reason="render-device validation requires --sim-render"))


@pytest.fixture
def sim_profile(request):
    return request.config.getoption("--sim-profile")


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


def pytest_terminal_summary(terminalreporter, exitstatus, config) -> None:  # noqa: ANN001 - pytest hook
    for line in _budget_lines():
        terminalreporter.write_line(line)
