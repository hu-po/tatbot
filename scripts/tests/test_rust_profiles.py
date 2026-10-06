"""Production feature coverage, selection and failure classification."""

from __future__ import annotations

import json
import os
import re
import subprocess
from pathlib import Path

import pytest
from test_check_runner import function

REPO = Path(__file__).resolve().parents[2]
from rust_checks import profile_rows  # noqa: E402


def test_profiles_cover_deployed_services_and_visualizer():
    features = (REPO / "rust/visiond/Cargo.toml").read_text().split("[features]", 1)[1].split("\n[", 1)[0]
    assert re.search(r"^default\s*=\s*\[\s*\]", features, re.M), (
        "deployment defaults changed; update the isolated profile matrix to cover their effective features")
    profiles = json.loads((REPO / "rust/check-profiles.json").read_text())
    covered = {frozenset(row["features"]) for row in profiles.values()}
    assert frozenset() in covered and frozenset({"rerun"}) in covered
    nodes = REPO / "config/nodes.json"
    if not nodes.exists():
        pytest.skip("deployment node inventory is absent")
    for node in json.loads(nodes.read_text()).values():
        if not isinstance(node, dict):
            continue
        for service in node.get("services", []):
            if service.get("package") == "tatbot-visiond":
                assert frozenset(service["features"]) in covered, service["features"]
    launcher = REPO / "scripts/vision/deploy_visualizer.sh"
    if launcher.exists():
        deployed = re.findall(r"cargo build[^\n]*--features '([^']+)'", launcher.read_text())
        assert deployed
        assert all(frozenset(features.split(",")) in covered for features in deployed)


def test_selection_and_required_profiles_are_validated(monkeypatch):
    monkeypatch.setenv("TATBOT_RUST_PROFILES", "vision-wrist,vision-core")
    monkeypatch.setenv("TATBOT_RUST_REQUIRED_PROFILES", "vision-wrist")
    assert profile_rows(REPO).splitlines() == [
        "vision-wrist|realsense,rerun|realsense2|1", "vision-core|||0"]
    monkeypatch.setenv("TATBOT_RUST_REQUIRED_PROFILES", "vision-poe")
    with pytest.raises(ValueError, match="included"):
        profile_rows(REPO)
    monkeypatch.setenv("TATBOT_RUST_PROFILES", "typo")
    with pytest.raises(ValueError, match="unknown"):
        profile_rows(REPO)


def run_profile(tmp_path, *, native=True, required=False, compile_failure=False):
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    (tmp_path / "rust").mkdir()
    trace = tmp_path / "cargo.log"
    cargo = bin_dir / "cargo"
    cargo.write_text('#!/bin/sh\nprintf "%s\\n" "$*" >> "$RUST_CHECK_TRACE"\n' +
                     ('exit 17\n' if compile_failure else 'exit 0\n'))
    cargo.chmod(0o755)
    pkg_config = bin_dir / "pkg-config"
    pkg_config.write_text('#!/bin/sh\nexit ' + ('0' if native else '1') + '\n')
    pkg_config.chmod(0o755)
    code = 'C_DIM= C_OFF= C_PASS= C_FAIL= C_SKIP=\ndeclare -a R_NAME=() R_STATE=() R_NOTE=()\n'
    code += 'record() { R_NAME+=("$1"); R_STATE+=("$2"); R_NOTE+=("${3:-}"); }\n'
    code += 'have() { command -v "$1" >/dev/null 2>&1; }\n'
    for name in ("skip", "run_job", "rust_profile_unavailable", "rust_vision_profile"):
        code += function(name) + "\n"
    code += f'rust_vision_profile vision-wrist realsense,rerun realsense2 {int(required)}\n'
    code += 'printf "STATE=%s\\n" "${R_STATE[*]}"\n'
    result = subprocess.run(["bash", "-c", code], cwd=tmp_path, capture_output=True, text=True,
                            env={**os.environ, "PATH": f'{bin_dir}:{os.environ["PATH"]}',
                                 "RUST_CHECK_TRACE": str(trace)})
    assert result.returncode == 0, result.stderr
    return result.stdout, trace.read_text().splitlines() if trace.exists() else []


def test_profile_builds_the_isolated_locked_release_binary_and_nothing_else(tmp_path):
    # The matrix answers one question -- does each deployed profile compile on
    # its own -- so it builds and never re-runs clippy or the tests, which the
    # unified workspace job already ran once for every crate.
    output, commands = run_profile(tmp_path)
    assert "STATE=PASS" in output
    assert commands == [
        "build --locked -p tatbot-visiond --no-default-features --features realsense,rerun --release --bins --quiet",
    ]


@pytest.mark.parametrize("native", [True, False])
def test_unified_features_name_what_this_node_cannot_compile(tmp_path, native):
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    pkg_config = bin_dir / "pkg-config"
    pkg_config.write_text('#!/bin/sh\nexit ' + ('0' if native else '1') + '\n')
    pkg_config.chmod(0o755)
    code = 'C_DIM= C_OFF= C_PASS= C_FAIL= C_SKIP=\ndeclare -a R_NAME=() R_STATE=() R_NOTE=()\n'
    code += 'record() { R_NAME+=("$1"); R_STATE+=("$2"); R_NOTE+=("${3:-}"); }\n'
    code += 'have() { command -v "$1" >/dev/null 2>&1; }\n'
    code += f'TROSSEN_ARM_SDK_ROOT={tmp_path}/no-sdk\n'
    for name in ("skip", "rust_features"):
        code += function(name) + "\n"
    code += 'rust_features\nprintf "FEATURES=%s\\nSKIPS=%s\\n" "$RUST_FEATURES" "${R_NAME[*]}"\n'
    result = subprocess.run(["bash", "-c", code], cwd=tmp_path, capture_output=True, text=True,
                            env={**os.environ, "PATH": f'{bin_dir}:{os.environ["PATH"]}'})
    assert result.returncode == 0, result.stderr
    features = result.stdout.split("FEATURES=", 1)[1].splitlines()[0].split(",")
    skips = result.stdout.split("SKIPS=", 1)[1].splitlines()[0].split()
    assert {"tatbot-visiond/rerun", "tatbot-visiond/zenoh", "tatbot-bus/zenoh"} <= set(features)
    assert "rust-trossen" in skips and "tatbot-arm/trossen" not in features
    assert ("tatbot-visiond/gstreamer" in features) is native
    assert ("tatbot-visiond/realsense" in features) is native
    assert ("rust-gstreamer" in skips) is not native


@pytest.mark.parametrize("required,state", [(False, "SKIP"), (True, "FAIL")])
def test_native_dependency_absence_is_explicit(tmp_path, required, state):
    output, commands = run_profile(tmp_path, native=False, required=required)
    assert f"STATE={state}" in output and "realsense2" in output
    assert not commands


def test_compile_failure_is_never_downgraded_to_missing_dependency(tmp_path):
    output, commands = run_profile(tmp_path, compile_failure=True)
    assert "STATE=FAIL" in output and len(commands) == 1
