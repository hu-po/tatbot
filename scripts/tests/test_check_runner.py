"""Exercise check reporting and the body scanner with independent fixtures."""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]


def function(name):
    source = (REPO / "scripts/check").read_text()
    return re.search(rf"^{name}\(\).*?^}}\n(?=\n|$)", source, re.M | re.S).group()


# Exactly the fields the pose bake (web/inkmap/tools/export-soma.py, in git
# history) carried from the authoring sheet into the catalog; everything else on either side is derived or consumed.
BAKED_POSE = {
    "body_rotation_xyzw": [0, 0, 0, 1],
    "constraints": ["posterior surface faces support"],
    "label": "supine on tattoo bed",
    "support_id": "tattoo-bed-v1",
}


def body_checkout(tmp_path):
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


def test_ros_fast_and_full_select_their_real_test_tiers(tmp_path):
    """The full selector reaches colcon as arguments: its Python test task replaces PYTEST_ADDOPTS."""
    tools = tmp_path / "bin"
    tools.mkdir()
    setup = tmp_path / "setup.bash"
    setup.touch()
    recorder = tools / "recorder"
    recorder.write_text('''#!/bin/bash
printf '%s|%s|%s\\n' "${0##*/}" "$*" "${PYTEST_ADDOPTS:-}" >> "$CALLS"
if [[ "${0##*/}" == colcon && "$1" == build ]]; then
  mkdir -p "$TATBOT_ROS_CHECK_DIR/install"
  touch "$TATBOT_ROS_CHECK_DIR/install/setup.bash"
fi
''')
    recorder.chmod(0o755)
    for command in ("python3", "colcon"):
        (tools / command).symlink_to(recorder)
    job = function("job_ros").replace("/opt/ros/jazzy/setup.bash", str(setup))
    for full in (0, 1):
        calls = tmp_path / f"calls-{full}"
        env = dict(os.environ, PATH=f"{tools}:{os.environ['PATH']}", CALLS=str(calls),
                   TATBOT_ROS_CHECK_DIR=str(tmp_path / f"cache-{full}"), PYTEST_ADDOPTS="",
                   TATBOT_ROS_PYTHON=str(tools / "python3"))   # the session tier's interpreter, recorded
        code = f'FULL={full}\nhave() {{ return 0; }}\nrun_job() {{ shift; "$@"; }}\n'
        subprocess.run(["bash", "-c", code + job + "\njob_ros"], cwd=REPO, env=env, check=True)
        recorded = calls.read_text()
        assert "colcon|build" in recorded
        if full:
            assert "colcon|test " in recorded and "colcon|test-result" in recorded
            assert "--return-code-on-test-failure" in recorded
            assert "--pytest-args -m slow or not slow --durations=0" in " ".join(recorded.split())
            assert "tatbot_estop_relay/test" in recorded
            assert "tatbot_session/test -m not slow" not in recorded
        else:
            assert "tatbot_session/test -m not slow" in recorded and "colcon|test " not in recorded
            # private CI reads this report as the developer tier's ROS coverage
            assert f"--junitxml {tmp_path}/cache-0/build/tatbot_session/pytest.xml" in recorded
            assert "tatbot_estop_relay/test" not in recorded


def test_ros_session_budget_fails_only_unmarked_overruns(monkeypatch):
    import runpy
    from types import SimpleNamespace

    path = REPO / "ros/tatbot_session/test/conftest.py"
    for keywords, budget, expected in (({}, "60", 1), ({"slow": True}, "60", 0), ({}, "0", 0)):
        hooks = runpy.run_path(str(path))
        monkeypatch.setenv("TATBOT_TEST_BUDGET_S", budget)
        hooks["pytest_runtest_logreport"](SimpleNamespace(when="call", passed=True, keywords=keywords,
                                                         duration=61.0, nodeid="fixture-test"))
        reporter = SimpleNamespace(write_line=lambda text: None)
        session = SimpleNamespace(exitstatus=0, config=SimpleNamespace(
            pluginmanager=SimpleNamespace(get_plugin=lambda name, reporter=reporter: reporter)))
        hooks["pytest_sessionfinish"](session, 0)
        assert session.exitstatus == expected


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


# A job pack (python/<project>/check-jobs.sh) driven through a copy of the real
# runner, so these exercise the hook, --list and the drive loop as shipped.
PACK = '''\
# shellcheck shell=bash
ALL_JOBS+=(probe-pack)
JOB_DESC[probe-pack]="a job a pack adds"
job_probe_pack() {
  [[ -z "${PROBE_SKIP:-}" ]] || { skip probe-pack "asked to skip"; return; }
  run_job probe-pack touch "$PROBE_MARK"
}
'''


def check_copy(root, pack=None):
    for rel in ("scripts/check", "scripts/lib/sim_fingerprint.sh"):
        (root / rel).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(REPO / rel, root / rel)
    if pack is not None:
        (root / "python/probe").mkdir(parents=True)
        (root / "python/probe/check-jobs.sh").write_text(pack)
    subprocess.run(["git", "init", "-q", str(root)], check=True)
    return root


def run_check(root, *args, **env):
    return subprocess.run(["bash", str(root / "scripts/check"), "--no-hooks", *args], cwd=root,
                          capture_output=True, text=True, env={**os.environ, **env}, timeout=60)


def test_a_job_pack_adds_one_listed_job_and_no_pack_changes_nothing(tmp_path):
    declared = re.search(r"^ALL_JOBS=\((.*?)\)$", (REPO / "scripts/check").read_text(), re.M).group(1).split()
    bare = run_check(check_copy(tmp_path / "bare"), "--list")
    assert bare.returncode == 0, bare.stderr
    assert [line.split()[0] for line in bare.stdout.splitlines()] == declared
    packed = run_check(check_copy(tmp_path / "packed", PACK), "--list")
    assert packed.returncode == 0, packed.stderr
    assert packed.stdout.splitlines() == [*bare.stdout.splitlines(), "  probe-pack a job a pack adds"]


def test_a_pack_job_runs_by_name_and_can_skip(tmp_path):
    root = check_copy(tmp_path / "tree", PACK)
    mark = tmp_path / "ran"
    ran = run_check(root, "probe-pack", PROBE_MARK=str(mark))
    assert ran.returncode == 0, ran.stdout + ran.stderr
    assert mark.is_file() and ">> probe-pack PASS" in ran.stdout and "1 passed, 0 failed, 0 skipped" in ran.stdout
    skipped = run_check(root, "probe-pack", PROBE_MARK=str(mark), PROBE_SKIP="1")
    assert skipped.returncode == 0, skipped.stdout + skipped.stderr
    assert "SKIP (asked to skip)" in skipped.stdout and "0 passed, 0 failed, 1 skipped" in skipped.stdout


SKIP_EXIT_PACK = PACK + '''\
ALL_JOBS+=(probe-exit)
JOB_DESC[probe-exit]="a body that may find, running, that it cannot run here"
job_probe_exit() {
  local RUN_JOB_SKIP_EXIT=3
  run_job probe-exit bash -c 'echo "some detail"; echo "the shared resource was taken"; exit "$PROBE_RC"'
}
'''


def test_a_job_that_names_its_skip_exit_reports_it_as_a_skip_with_its_reason(tmp_path):
    root = check_copy(tmp_path / "tree", SKIP_EXIT_PACK.replace(
        'run_job probe-pack touch "$PROBE_MARK"', 'run_job probe-pack bash -c \'exit "$PROBE_RC"\''))
    for rc, exit_job, plain_job in (("3", "SKIP (the shared resource was taken)", "FAIL"), ("1", "FAIL", "FAIL"),
                                    ("0", "PASS", "PASS")):
        ran = run_check(root, "probe-exit", "probe-pack", PROBE_RC=rc)
        lines = {line.split()[1]: " ".join(line.split()[2:]) for line in ran.stdout.splitlines()
                 if line.startswith(">> ")}
        assert lines["probe-exit"].startswith(exit_job), ran.stdout
        assert lines["probe-pack"].startswith(plain_job), ran.stdout  # exit 3 is a FAIL unless a job says so
        assert ran.returncode == (0 if rc == "0" else 1), ran.stdout


def test_a_pack_job_without_a_description_is_refused_by_name(tmp_path):
    root = check_copy(tmp_path, PACK.replace('JOB_DESC[probe-pack]="a job a pack adds"\n', ""))
    result = run_check(root, "--list")
    assert result.returncode == 2
    assert "python/probe/check-jobs.sh: job 'probe-pack' needs a JOB_DESC entry" in result.stderr


def test_pass_markers_follow_the_directory_override(tmp_path):
    code = (f'source "{REPO}/scripts/lib/sim_fingerprint.sh"\n'
            'sim_fingerprint::record deadbeef probe\n'
            'sim_fingerprint::is_fresh deadbeef probe && echo FRESH\n'
            'TATBOT_SIM_PASS_DIR= sim_fingerprint::is_fresh deadbeef probe && echo SHARED\n'
            'true\n')
    env = {**os.environ, "XDG_CACHE_HOME": str(tmp_path / "xdg"), "TATBOT_SIM_PASS_DIR": str(tmp_path / "own")}
    result = subprocess.run(["bash", "-c", code], capture_output=True, text=True, env=env)
    assert result.returncode == 0, result.stderr
    assert result.stdout.split() == ["FRESH"]
    assert (tmp_path / "own/deadbeef.probe").is_file() and not (tmp_path / "xdg").exists()


def test_the_test_suite_key_ignores_only_the_tests_a_job_pack_runs(tmp_path):
    rels = ("python/plain/a.py", "python/probe/tests/test_b.py", "python/probe/check-jobs.sh",
            "python/probe/src/probe/capabilities/cap.py")
    for rel in rels:
        (tmp_path / rel).parent.mkdir(parents=True, exist_ok=True)
        (tmp_path / rel).write_text("v1\n")
    subprocess.run(["git", "init", "-q"], cwd=tmp_path, check=True)
    subprocess.run(["git", "add", "python"], cwd=tmp_path, check=True)
    command = 'source "$1"; sim_fingerprint::compute "$2" sim_fingerprint::guarded_tests'

    def key():  # from this checkout's cwd: the guard must glob the repo it hashes
        return subprocess.run(["bash", "-c", command, "bash", str(REPO / "scripts/lib/sim_fingerprint.sh"),
                               str(tmp_path)], cwd=REPO, capture_output=True, text=True, check=True).stdout.strip()

    baseline = key()
    (tmp_path / "python/probe/tests/test_b.py").write_text("v2\n")
    (tmp_path / "python/probe/tests/test_c.py").write_text("untracked\n")
    assert key() == baseline
    # The scripts tests reach a pack project's source through the CLI (a verb
    # wraps its entry point and scans its metadata), so source edits re-run them.
    previous = baseline
    for rel, text in (("python/probe/src/probe/capabilities/cap.py", "v2\n"),
                      ("python/probe/src/probe/capabilities/new.py", "untracked\n"),
                      ("python/probe/tests_helper.py", "beside tests/, not in it\n"),
                      ("python/probe/check-jobs.sh", "v2\n"),
                      ("python/plain/a.py", "v2\n")):
        (tmp_path / rel).write_text(text)
        current = key()
        assert current != previous, rel
        previous = current


# What git exports to a hook or a `git bisect run` step. A job that inherited
# them would run a test's scratch-repository git against the repository they
# name; the runner drops them and keeps only a commit hook's index for --staged.
GIT_PROBE_PACK = '''\
# shellcheck shell=bash
ALL_JOBS+=(probe-git)
JOB_DESC[probe-git]="what a job sees of git"
job_probe_git() {
  sources '*.py' > "$PROBE_OUT/staged"
  env | grep -E '^GIT_(DIR|WORK_TREE|INDEX_FILE|COMMON_DIR|OBJECT_DIRECTORY|ALTERNATE_OBJECT_DIRECTORIES)=' \
    > "$PROBE_OUT/env" || true
  git rev-parse --absolute-git-dir > "$PROBE_OUT/git-dir"
  run_job probe-git true
}
'''


def test_a_job_never_sees_the_repository_git_handed_the_runner(tmp_path):
    root = check_copy(tmp_path / "tree", GIT_PROBE_PACK)
    decoy = tmp_path / "decoy"
    subprocess.run(["git", "init", "-q", str(decoy)], check=True)
    decoy_config = (decoy / ".git/config").read_text()
    # A commit hook's own index holds a file the tree's index does not.
    (root / "staged.py").write_text("x = 1\n")
    index = tmp_path / "hook-index"
    subprocess.run(["git", "add", "staged.py"], cwd=root, check=True, env={**os.environ, "GIT_INDEX_FILE": str(index)})
    out = tmp_path / "out"
    out.mkdir()
    env = {"GIT_DIR": str(decoy / ".git"), "GIT_WORK_TREE": str(decoy), "GIT_COMMON_DIR": str(decoy / ".git"),
           "GIT_OBJECT_DIRECTORY": str(decoy / ".git/objects"), "GIT_INDEX_FILE": str(index),
           "GIT_ALTERNATE_OBJECT_DIRECTORIES": str(decoy / ".git/objects"), "PROBE_OUT": str(out)}
    ran = run_check(root, "--staged", "probe-git", **env)
    assert ran.returncode == 0, ran.stdout + ran.stderr
    assert (out / "env").read_text() == ""
    assert Path((out / "git-dir").read_text().strip()) == (root / ".git").resolve()
    assert (out / "staged").read_text().split() == ["staged.py"]
    assert (decoy / ".git/config").read_text() == decoy_config


def test_this_session_carries_none_of_gits_repository_variables():
    location = ("GIT_DIR", "GIT_WORK_TREE", "GIT_INDEX_FILE", "GIT_COMMON_DIR", "GIT_OBJECT_DIRECTORY",
                "GIT_ALTERNATE_OBJECT_DIRECTORIES")
    assert [name for name in location if name in os.environ] == []


def test_a_session_started_by_git_drops_its_repository_variables(tmp_path):
    """As a hook or a `git bisect run` step starts it: the conftest clears them before any test."""
    decoy = tmp_path / "decoy"
    subprocess.run(["git", "init", "-q", str(decoy)], check=True)
    env = {**os.environ, "GIT_DIR": str(decoy / ".git"), "GIT_WORK_TREE": str(decoy),
           "GIT_INDEX_FILE": str(decoy / ".git/index"), "GIT_COMMON_DIR": str(decoy / ".git")}
    probe = f"{Path(__file__).resolve()}::test_this_session_carries_none_of_gits_repository_variables"
    out = subprocess.run([sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider", "-p", "no:xdist", probe],
                         cwd=REPO, env=env, capture_output=True, text=True, timeout=120)
    assert out.returncode == 0 and "1 passed" in out.stdout, out.stdout + out.stderr
