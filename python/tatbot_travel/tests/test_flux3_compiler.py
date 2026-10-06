"""Compiled inference must reject unsupported assemblers before policy startup."""

import os
import sys
from types import SimpleNamespace

import pytest
from tatbot_travel import flux3

KEYS = ("TRITON_PTXAS_PATH", "TRITON_PTXAS_BLACKWELL_PATH")


@pytest.fixture
def assemblers(monkeypatch, tmp_path):
    for key in KEYS:
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setenv("CUDA_HOME", str(tmp_path))
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(
        cuda=SimpleNamespace(get_device_capability=lambda: (11, 0))))
    monkeypatch.setattr(flux3.shutil, "which", lambda _: None)
    support = {}

    def run(args, **kwargs):
        return SimpleNamespace(returncode=0, stdout=support.get(args[0], "sm_100a"))

    monkeypatch.setattr(flux3.subprocess, "run", run)
    return str(tmp_path / "bin/ptxas"), support


def test_installed_supported_assembler_replaces_both_unset_triton_defaults(assemblers):
    path, support = assemblers
    support[path] = "Allowed GPU targets: sm_100a, sm_110a"
    flux3._configure_flex_compiler()
    assert [os.environ[key] for key in KEYS] == [path, path]


def test_explicit_supported_compilers_are_preserved(monkeypatch, assemblers):
    _, support = assemblers
    for index, key in enumerate(KEYS):
        path = f"/custom/compiler-{index}"
        monkeypatch.setenv(key, path)
        support[path] = "sm_110a"
    flux3._configure_flex_compiler()
    assert [os.environ[key] for key in KEYS] == ["/custom/compiler-0", "/custom/compiler-1"]


@pytest.mark.parametrize("key", KEYS)
def test_explicit_unsupported_compiler_is_refused_without_partial_env_change(monkeypatch, assemblers, key):
    path, support = assemblers
    support[path] = "sm_110a"
    monkeypatch.setenv(key, "/custom/unsupported")
    before = {name: os.environ.get(name) for name in KEYS}
    with pytest.raises(RuntimeError, match=key):
        flux3._configure_flex_compiler()
    assert {name: os.environ.get(name) for name in KEYS} == before


def test_missing_compatible_toolkit_fails_before_vae_import(monkeypatch, assemblers):
    monkeypatch.setenv("F3_NATTEN_BACKEND", "flex-fna")
    monkeypatch.setenv("F3_NATTEN_COMPILE", "1")
    with pytest.raises(RuntimeError, match="supporting sm_110a"):
        flux3.force_natten_backend()
    assert all(key not in os.environ for key in KEYS)


def test_other_gpu_keeps_its_existing_triton_compiler_selection(monkeypatch, assemblers):
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(
        cuda=SimpleNamespace(get_device_capability=lambda: (10, 0))))
    monkeypatch.setenv(KEYS[0], "/existing/compiler")
    flux3._configure_flex_compiler()
    assert os.environ[KEYS[0]] == "/existing/compiler"
    assert KEYS[1] not in os.environ
