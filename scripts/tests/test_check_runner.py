"""Exercise check reporting and the body scanner with independent fixtures."""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import subprocess
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]


def function(name):
    source = (REPO / "scripts/check").read_text()
    return re.search(rf"^{name}\(\).*?^}}\n(?=\n|$)", source, re.M | re.S).group()


# Exactly the fields web/inkmap/tools/export-soma.py carries from the authoring
# sheet into the catalog; everything else on either side is derived or consumed.
BAKED_POSE = {
    "body_rotation_xyzw": [0, 0, 0, 1],
    "constraints": ["posterior surface faces support"],
    "label": "supine on tattoo bed",
    "support_id": "tattoo-bed-v1",
}


def body_checkout(tmp_path):
    # The scan loads the frozen-import boundary module by path, so a checkout
    # without it cannot run at all. Copy the real module rather than stubbing
    # it: scan_frozen_imports then runs for real over the fixture, and these
    # tests keep asserting on what the scan reports rather than on how it died.
    boundary = Path("python/tatbot_sim/src/tatbot_sim/human_rep/boundary.py")
    (tmp_path / boundary).parent.mkdir(parents=True)
    shutil.copyfile(REPO / boundary, tmp_path / boundary)
    assets = tmp_path / "web/inkmap/public/bodies"
    assets.mkdir(parents=True)
    for suffix in ("exclusions.bin", "glb", "poses.bin", "regions.json"):
        (assets / f"mhr-soma-v1.{suffix}").touch()
    specs = tmp_path / "config/body-models"
    specs.mkdir(parents=True)
    (specs / "mhr-soma-v1.json").write_text(json.dumps({
        "name": "mhr-soma-v1", "model": {"identity_model_type": "mhr"}}))
    # The catalog is baked from the authoring sheet, and the scan compares the
    # fields the bake copies across. Both sides have to exist and agree here, or
    # every test below fails on the parity gate instead of on what it asserts.
    catalog = tmp_path / "config/inkmap/body-poses.json"
    catalog.parent.mkdir(parents=True)
    catalog.write_text(json.dumps({
        "model_spec_id": "mhr-soma-v1", "schema": "tatbot.body-pose-catalog/2",
        "identity_sha256": "a" * 64, "model_spec_sha256": "b" * 64,
        "pose_ids": ["supine"],
        "poses": {"supine": dict(BAKED_POSE, surface_sha256="c" * 64)}}))
    (specs / "mhr-soma-v1").mkdir()
    (specs / "mhr-soma-v1/poses.json").write_text(json.dumps({
        "schema": "tatbot.soma-pose-authoring/1", "rig_id": "mhr-soma-v1",
        "identity_sha256": "a" * 64, "model_spec_sha256": "b" * 64,
        "pose_ids": ["supine"],
        "poses": {"supine": dict(BAKED_POSE, joint_rotations_euler_xyz_deg={"LeftArm": [0, 0, -55]})}}))
    subprocess.run(["git", "init", "-q", str(tmp_path)], check=True)
    return catalog, specs / "mhr-soma-v1/poses.json"


def test_body_scan_ignores_incidental_binary_token_but_rejects_text(tmp_path):
    body_checkout(tmp_path)
    token = bytes([72, 66, 77])
    asset = tmp_path / "palette.stl"
    asset.write_bytes(b"\0" * 80 + token + b"\0" * 50)
    code = function("body_single_path_impl") + "\nbody_single_path_impl"
    result = subprocess.run(["bash", "-c", code], cwd=tmp_path, capture_output=True, text=True)
    assert result.returncode == 0, result.stdout
    (tmp_path / "retired.txt").write_bytes(token)
    result = subprocess.run(["bash", "-c", code], cwd=tmp_path, capture_output=True, text=True)
    assert result.returncode == 1
    assert "retired.txt: retired body token" in result.stdout


def test_body_scan_rejects_a_new_consumer_of_a_frozen_module(tmp_path):
    """The research freeze is a live gate, not just a module the scan loads.

    Until the fixture carried boundary.py the scan died before reaching this
    check, so nothing here proved it still fires.
    """
    body_checkout(tmp_path)
    consumer = tmp_path / "python/tatbot_sim/src/tatbot_sim/human_rep/consumer.py"
    consumer.write_text("from tatbot_sim.human_rep import mechanics\n")
    code = function("body_single_path_impl") + "\nbody_single_path_impl"
    result = subprocess.run(["bash", "-c", code], cwd=tmp_path, capture_output=True, text=True)
    assert result.returncode == 1
    assert "frozen research dependency tatbot_sim.human_rep.mechanics" in result.stderr


def test_body_scan_reports_an_absent_boundary_module_by_name(tmp_path):
    """A missing boundary module must say so rather than raise a traceback."""
    body_checkout(tmp_path)
    (tmp_path / "python/tatbot_sim/src/tatbot_sim/human_rep/boundary.py").unlink()
    code = function("body_single_path_impl") + "\nbody_single_path_impl"
    result = subprocess.run(["bash", "-c", code], cwd=tmp_path, capture_output=True, text=True)
    assert result.returncode == 1
    assert "frozen-import boundary module is missing" in result.stderr
    assert "Traceback" not in result.stderr


def test_body_scan_rejects_renamed_binary_by_actual_hash(tmp_path):
    body_checkout(tmp_path)
    asset = b"\0retired binary fixture\0"
    (tmp_path / "renamed.bin").write_bytes(asset)
    code = function("body_single_path_impl")
    original_digest = re.search(r"[a-f0-9]{64}", code).group()
    code = code.replace(original_digest, hashlib.sha256(asset).hexdigest())
    result = subprocess.run(["bash", "-c", code + "\nbody_single_path_impl"],
                            cwd=tmp_path, capture_output=True, text=True)
    assert result.returncode == 1
    assert "renamed.bin: retired binary asset digest" in result.stdout


def test_successful_job_surfaces_omitted_pytest_coverage():
    code = "C_DIM= C_OFF= C_PASS= C_FAIL= C_SKIP=\n"
    code += "declare -a R_NAME=() R_STATE=() R_NOTE=()\n"
    code += 'record() { R_NAME+=("$1"); R_STATE+=("$2"); R_NOTE+=("${3:-}"); }\n'
    code += function("skip") + "\n" + function("run_job")
    code += '\nrun_job tests printf "%s\\n" "tatbot: skipping 3 module(s)"\n'
    code += '[[ "${R_STATE[*]}" == "PASS SKIP" ]]'
    result = subprocess.run(["bash", "-c", code], capture_output=True, text=True)
    assert result.returncode == 0, result.stdout
    assert "tests-coverage" in result.stdout and "3 module(s)" in result.stdout


def test_named_profile_refuses_missing_dependency(monkeypatch):
    import runpy

    import pytest

    namespace = runpy.run_path(str(REPO / "scripts/tests/conftest.py"))
    start = namespace["pytest_sessionstart"]
    monkeypatch.setenv("TATBOT_TEST_PROFILE", "integration")
    monkeypatch.setitem(start.__globals__, "_absent", lambda required: ["lerobot"])
    with pytest.raises(pytest.UsageError, match="integration profile missing required dependencies: lerobot"):
        start(None)


def test_offline_profile_requires_metric_rgbd_dependencies(monkeypatch):
    import runpy

    import pytest

    namespace = runpy.run_path(str(REPO / "scripts/tests/conftest.py"))
    start = namespace["pytest_sessionstart"]
    monkeypatch.setenv("TATBOT_TEST_PROFILE", "offline")
    for dependency in ("open3d", "threadpoolctl"):
        monkeypatch.setitem(start.__globals__, "_absent",
                            lambda required, dep=dependency: [dep] if dep in required else [])
        with pytest.raises(pytest.UsageError, match=f"missing required dependencies: {dependency}"):
            start(None)


def test_offline_profile_resolves_source_compiler_and_checks_external_dependencies(monkeypatch):
    import runpy

    import pytest

    namespace = runpy.run_path(str(REPO / "scripts/tests/conftest.py"))
    dependencies = namespace["_COLLECTION_IMPORTS"]["test_ros_acquired_artwork"]
    assert {"numpy", "yaml"} <= dependencies
    assert not {"tatbot_ink", "tatbot_motion", "tatbot_description"} & dependencies
    start = namespace["pytest_sessionstart"]
    monkeypatch.setenv("TATBOT_TEST_PROFILE", "offline")
    monkeypatch.setitem(start.__globals__, "_absent", lambda required: ["yaml"] if "yaml" in required else [])
    with pytest.raises(pytest.UsageError, match="offline profile missing required dependencies: yaml"):
        start(None)


def test_light_profile_discloses_unavailable_attachment_modules(monkeypatch):
    import importlib.util
    import runpy

    original = importlib.util.find_spec
    monkeypatch.setattr(importlib.util, "find_spec",
                        lambda name: None if name == "cv2" else original(name))
    namespace = runpy.run_path(str(REPO / "scripts/tests/conftest.py"))
    for stem in ("test_surface_attachment", "test_surface_attachment_inputs",
                 "test_surface_attachment_benchmark"):
        assert "cv2" in namespace["_missing"][stem]
        assert f"{stem}.py" in namespace["collect_ignore"]
    monkeypatch.setenv("TATBOT_TEST_PROFILE", "light")
    namespace["pytest_sessionstart"](None)


def test_coverage_summary_only_reports_selected_modules(monkeypatch):
    import runpy
    from types import SimpleNamespace

    namespace = runpy.run_path(str(REPO / "scripts/tests/conftest.py"))
    report = namespace["pytest_report_header"]
    monkeypatch.setitem(report.__globals__, "_missing", {"test_selected": ["lerobot"], "test_other": ["torch"]})
    config = SimpleNamespace(args=[str(REPO / "scripts/tests/test_selected.py")])
    result = report(config)
    assert "lerobot (1)" in result and "torch" not in result
    lines = []
    namespace["pytest_terminal_summary"](SimpleNamespace(write_line=lines.append), 0, config)
    assert result.splitlines() == lines


def run_body_scan(tmp_path):
    code = function("body_single_path_impl") + "\nbody_single_path_impl"
    return subprocess.run(["bash", "-c", code], cwd=tmp_path, capture_output=True, text=True)


def test_body_scan_accepts_a_catalog_that_matches_its_authoring_sheet(tmp_path):
    body_checkout(tmp_path)
    assert run_body_scan(tmp_path).returncode == 0


def test_body_scan_catches_an_authored_pose_field_that_was_never_re_baked(tmp_path):
    """The drift the catalog cannot show on its own.

    export-soma.py copies four authored fields per pose into the catalog and
    derives the rest. Its own --check re-bakes and compares bytes, but that needs
    the MHR/SOMA source assets and so never runs here. Edit the sheet, skip the
    bake, and web/inkmap and the simulator keep serving the stale text.
    """
    catalog_path, authoring_path = body_checkout(tmp_path)
    authoring = json.loads(authoring_path.read_text())
    authoring["poses"]["supine"]["label"] = "supine on a different bed"
    authoring_path.write_text(json.dumps(authoring))

    result = run_body_scan(tmp_path)
    assert result.returncode == 1
    assert "pose 'supine' field 'label' was edited in the authoring sheet" in result.stderr
    assert "export-soma.py" in result.stderr  # says how to fix it, not just that it broke


def test_body_scan_catches_a_pose_added_to_only_one_side(tmp_path):
    catalog_path, authoring_path = body_checkout(tmp_path)
    catalog = json.loads(catalog_path.read_text())
    catalog["pose_ids"] = ["supine", "prone"]
    catalog_path.write_text(json.dumps(catalog))

    result = run_body_scan(tmp_path)
    assert result.returncode == 1
    assert "disagree on pose_ids" in result.stderr


def test_body_scan_catches_a_catalog_baked_from_a_different_identity(tmp_path):
    catalog_path, authoring_path = body_checkout(tmp_path)
    catalog = json.loads(catalog_path.read_text())
    catalog["identity_sha256"] = "d" * 64
    catalog_path.write_text(json.dumps(catalog))

    result = run_body_scan(tmp_path)
    assert result.returncode == 1
    assert "disagree on identity_sha256" in result.stderr


def test_body_scan_ignores_the_fields_the_bake_derives(tmp_path):
    """A derived field differing is normal: only the copied four are compared."""
    catalog_path, authoring_path = body_checkout(tmp_path)
    catalog = json.loads(catalog_path.read_text())
    catalog["poses"]["supine"]["surface_sha256"] = "e" * 64
    catalog["poses"]["supine"]["quality"] = {"triangle_area_ratio_p99": 1.2}
    catalog_path.write_text(json.dumps(catalog))

    assert run_body_scan(tmp_path).returncode == 0


def schema_checkout(tmp_path, source):
    """A checkout with the repository's schema table and one session source file."""
    table = tmp_path / "config/schemas.json"
    table.parent.mkdir(parents=True)
    shutil.copyfile(REPO / "config/schemas.json", table)
    path = tmp_path / "rust/tatbot-arm/src/witness.rs"
    path.parent.mkdir(parents=True)
    path.write_text(source)
    return table


def run_schemas(root):
    code = f'ROOT="{REPO}"\n' + function("schemas_impl") + f"\nschemas_impl {root}"
    return subprocess.run(["bash", "-c", code], capture_output=True, text=True)


def test_schemas_fails_a_literal_outside_the_table_and_names_the_file(tmp_path):
    table = schema_checkout(tmp_path, 'let s = "tatbot.never-written/1";\n#[cfg(test)]\nmod tests { const T: &str = "tatbot.only-in-a-test/1"; }\n')
    result = run_schemas(tmp_path)
    assert result.returncode == 1
    assert "rust/tatbot-arm/src/witness.rs writes tatbot.never-written/1" in result.stderr
    assert "only-in-a-test" not in result.stderr, "a test module is not scanned"
    # Listing the file under pending with its lane admits it; a family needs no row.
    t = json.loads(table.read_text())
    t["pending"] = {"tatbot.never-written/1": {"disposition": "delete", "sites": {"rust/tatbot-arm/src/witness.rs": "F step 9"}}}
    table.write_text(json.dumps(t))
    assert run_schemas(tmp_path).returncode == 0
    (tmp_path / "rust/tatbot-arm/src/witness.rs").write_text('let s = "tatbot.receipt/1";\n')
    t["pending"] = {}
    table.write_text(json.dumps(t))
    assert run_schemas(tmp_path).returncode == 0


def test_schemas_fails_a_stale_pending_row_so_the_table_only_shrinks(tmp_path):
    table = schema_checkout(tmp_path, 'let s = "tatbot.receipt/1";\n')
    t = json.loads(table.read_text())
    t["pending"] = {"tatbot.scan-settled/1": {"disposition": "rename", "sites": {"rust/tatbot-arm/src/witness.rs": "B step 4a"}}}
    table.write_text(json.dumps(t))
    result = run_schemas(tmp_path)
    assert result.returncode == 1
    assert "pending['tatbot.scan-settled/1'].sites['rust/tatbot-arm/src/witness.rs'] is stale" in result.stderr
    assert "remove the row" in result.stderr
    t["pending"] = {"tatbot.scan-settled/1": {"disposition": "rename", "sites": {"rust/tatbot-arm/src/witness.rs": ""}}}
    (tmp_path / "rust/tatbot-arm/src/witness.rs").write_text("let s = 'tatbot.scan-settled/1';\n")
    table.write_text(json.dumps(t))
    result = run_schemas(tmp_path)
    assert result.returncode == 1 and "names no lane" in result.stderr


def run_cache_harness(tmp_path, *, recorded, no_cache=False, state="PASS"):
    """Drive scripts/check's PASS cache with a throwaway cache directory."""
    code = 'C_DIM= C_OFF= C_PASS= C_FAIL= C_SKIP=\ndeclare -a R_NAME=() R_STATE=() R_NOTE=()\n'
    code += 'record() { R_NAME+=("$1"); R_STATE+=("$2"); R_NOTE+=("${3:-}"); }\n'
    code += f'source "{REPO}/scripts/lib/sim_fingerprint.sh"\n'
    for name in ("cached", "passed"):
        code += function(name) + "\n"
    code += 'fp=deadbeef\n'
    if recorded:
        code += 'sim_fingerprint::record "$fp" tests-fast\n'
    code += 'if cached tests "$fp" tests-fast; then echo HIT; else echo MISS; fi\n'
    code += f'record tests {state}\n'
    code += 'passed tests && sim_fingerprint::record "$fp" tests-fast\n'
    code += 'ls "$(sim_fingerprint::cache_dir)" 2>/dev/null || true\n'
    env = {**os.environ, "XDG_CACHE_HOME": str(tmp_path)}
    env.pop("TATBOT_CHECK_NO_CACHE", None)
    if no_cache:
        env["TATBOT_CHECK_NO_CACHE"] = "1"
    result = subprocess.run(["bash", "-c", code], capture_output=True, text=True, env=env)
    assert result.returncode == 0, result.stderr
    return result.stdout


def test_fast_test_cache_key_includes_native_build_inputs(tmp_path):
    lock = tmp_path / 'rust/Cargo.lock'
    lock.parent.mkdir()
    lock.write_text('old lock')
    subprocess.run(['git', 'init', '-q'], cwd=tmp_path, check=True)
    subprocess.run(['git', 'add', 'rust/Cargo.lock'], cwd=tmp_path, check=True)
    command = ('source "$1"; sim_fingerprint::compute "$2" '
               'sim_fingerprint::guarded_tests')

    def key():
        return subprocess.run(['bash', '-c', command, 'bash',
                               str(REPO / 'scripts/lib/sim_fingerprint.sh'), str(tmp_path)],
                              capture_output=True, text=True, check=True).stdout.strip()

    baseline = key()
    lock.write_text('new lock')
    after_lock = key()
    assert after_lock != baseline
    source = tmp_path / 'rust/trossen-arm-sys/src/lib.rs'
    source.parent.mkdir(parents=True)
    source.write_text('new source')
    subprocess.run(['git', 'add', 'rust/trossen-arm-sys/src/lib.rs'], cwd=tmp_path, check=True)
    assert key() != after_lock


def test_pass_cache_hits_only_a_recorded_pass_of_the_same_scope(tmp_path):
    assert "MISS" in run_cache_harness(tmp_path, recorded=False)
    out = run_cache_harness(tmp_path, recorded=True)
    assert "HIT" in out and "cached: unchanged since" in out


def test_pass_cache_is_bypassed_on_request_and_records_only_a_pass(tmp_path):
    assert "MISS" in run_cache_harness(tmp_path / "bypass", recorded=True, no_cache=True)
    out = run_cache_harness(tmp_path / "fail", recorded=False, state="FAIL")
    assert "MISS" in out and "deadbeef" not in out
