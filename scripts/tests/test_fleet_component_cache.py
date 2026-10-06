"""Cross-revision planner reuse keeps its source and executable boundaries."""
from __future__ import annotations

import hashlib
import json

import fleet_component_cache as cache
import fleet_release
import pytest


def _file(source, name, body):
    path = source / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(body)
    return {'kind': 'file', 'sha256': hashlib.sha256(body).hexdigest(),
            'executable': False}


def _stage(root, sha, *, config=b'limits', docs=b'old'):
    stage = root / 'releases' / sha
    source = stage / 'source'
    source.mkdir(parents=True)
    files = {
        'cpp/teleop/CMakeLists.txt': _file(source, 'cpp/teleop/CMakeLists.txt', b'planner'),
        'config/motion_constants.json': _file(source, 'config/motion_constants.json', config),
        'docs/session.md': _file(source, 'docs/session.md', docs),
    }
    for name in cache.BUILD_RECIPES:
        files[name] = _file(source, name, name.encode())
    (stage / 'source-manifest.json').write_text(json.dumps({
        'schema': 'tatbot.receipt/1', 'kind': 'source-manifest', 'source_sha': sha,
        'archive_sha256': 'a' * 64, 'files': files,
    }))
    (stage / 'build.sha').write_text(sha)
    return stage


def _old_build(stage, monkeypatch):
    binary = stage / cache.PLANNER
    binary.parent.mkdir(parents=True)
    binary.write_bytes(b'#!/bin/sh\nexit 0\n')
    binary.chmod(0o755)
    monkeypatch.setattr(cache, '_environment',
                        lambda path: {'compiler_reusable': True, 'toolchain': 'same', 'abi': 'same'})
    monkeypatch.setattr(cache, '_cmake_compiler_matches', lambda stage, compiler: True)
    header = stage.parent.parent / 'header.hpp'
    header.write_bytes(b'first')
    monkeypatch.setattr(cache, '_compile_dependencies',
                        lambda stage, manifest: {str(header): cache.sha256_file(header)})
    cache._record(stage)
    receipt = fleet_release.build(stage, stage.name, 'arm-owner', [],
                                    ['path_plan_check'],
                                    components=['source/planner-component.json'])
    (stage / 'build.json').write_text(json.dumps(receipt))
    return receipt


def test_docs_only_new_source_reuses_verified_planner_in_own_stage(tmp_path, monkeypatch):
    old = _stage(tmp_path, 'a' * 40)
    receipt = _old_build(old, monkeypatch)
    new = _stage(tmp_path, 'b' * 40, docs=b'updated')
    checkout = tmp_path / 'checkout'
    checkout.mkdir()
    (checkout / fleet_release.STAMP).write_text(json.dumps(receipt))

    recorded = cache.reuse(new, checkout, 'arm-owner')
    assert (new / cache.PLANNER).read_bytes() == (old / cache.PLANNER).read_bytes()
    assert recorded['source_sha'] == new.name
    assert recorded['reused_from'] == old.name
    assert (new / 'source' / cache.RECORD).is_file()
    assert cache._record(new) == recorded


def test_reused_release_carries_header_proof_to_next_revision(tmp_path, monkeypatch):
    old = _stage(tmp_path, 'a' * 40)
    first_receipt = _old_build(old, monkeypatch)
    checkout = tmp_path / 'checkout'
    checkout.mkdir()
    (checkout / fleet_release.STAMP).write_text(json.dumps(first_receipt))
    middle = _stage(tmp_path, 'b' * 40, docs=b'middle')
    middle_record = cache.reuse(middle, checkout, 'arm-owner')
    assert middle_record['compile_dependencies'] == {
        str(tmp_path / 'header.hpp'): cache.sha256_file(tmp_path / 'header.hpp')}
    assert not (middle / 'source/cpp/teleop/build/CMakeCache.txt').exists()
    middle_receipt = fleet_release.build(middle, middle.name, 'arm-owner', [],
                                           ['path_plan_check'],
                                           components=['source/planner-component.json'])
    (middle / 'build.json').write_text(json.dumps(middle_receipt))
    (checkout / fleet_release.STAMP).write_text(json.dumps(middle_receipt))
    newest = _stage(tmp_path, 'c' * 40, docs=b'newest')
    newest_record = cache.reuse(newest, checkout, 'arm-owner')
    assert newest_record['reused_from'] == middle.name
    assert newest_record['compile_dependencies'] == middle_record['compile_dependencies']


def test_reused_planner_record_refuses_later_runtime_drift(tmp_path, monkeypatch):
    old = _stage(tmp_path, 'a' * 40)
    receipt = _old_build(old, monkeypatch)
    checkout = tmp_path / 'checkout'
    checkout.mkdir()
    (checkout / fleet_release.STAMP).write_text(json.dumps(receipt))
    new = _stage(tmp_path, 'b' * 40, docs=b'updated')
    cache.reuse(new, checkout, 'arm-owner')
    monkeypatch.setattr(cache, '_environment',
                        lambda path: {'compiler_reusable': True, 'toolchain': 'changed'})
    with pytest.raises(ValueError, match='toolchain or runtime differs'):
        cache._record(new)


@pytest.mark.parametrize('change', ('config', 'binary', 'record', 'unbound_record', 'toolchain',
                                    'runtime_relocation', 'stamp', 'external_header'))
def test_changed_input_or_untrusted_prior_release_refuses_reuse(tmp_path, monkeypatch, change):
    old = _stage(tmp_path, 'a' * 40)
    receipt = _old_build(old, monkeypatch)
    new = _stage(tmp_path, 'b' * 40, config=b'changed' if change == 'config' else b'limits')
    checkout = tmp_path / 'checkout'
    checkout.mkdir()
    if change == 'binary':
        (old / cache.PLANNER).write_bytes(b'#!/bin/sh\nexit 1\n')
    elif change == 'record':
        path = old / 'source' / cache.RECORD
        record = json.loads(path.read_text())
        record['binary_sha256'] = '0' * 64
        path.write_text(json.dumps(record))
    elif change == 'unbound_record':
        receipt['components'] = {}
        (old / 'build.json').write_text(json.dumps(receipt))
    elif change == 'toolchain':
        monkeypatch.setattr(cache, '_environment', lambda path: {'toolchain': 'changed'})
    elif change == 'external_header':
        (tmp_path / 'header.hpp').write_bytes(b'changed')
    elif change == 'runtime_relocation':
        monkeypatch.setattr(cache, '_environment',
                            lambda path: ({'compiler_reusable': True, 'toolchain': 'same',
                                           'abi': 'changed'} if new.name in str(path) else
                                          {'compiler_reusable': True, 'toolchain': 'same',
                                           'abi': 'same'}))
    elif change == 'stamp':
        receipt['node'] = 'another-owner'
    (checkout / fleet_release.STAMP).write_text(json.dumps(receipt))

    with pytest.raises((OSError, ValueError)):
        cache.reuse(new, checkout, 'arm-owner')
    assert not (new / cache.PLANNER).exists()


def test_actual_cmake_compiler_and_external_header_bytes_bound_reuse(tmp_path, monkeypatch):
    stage = _stage(tmp_path, 'a' * 40)
    compiler = cache._command_binary('c++')
    cache_file = stage / 'source/cpp/teleop/build/CMakeCache.txt'
    cache_file.parent.mkdir(parents=True)
    cache_file.write_text('CMAKE_CXX_COMPILER:FILEPATH=/usr/bin/c++\n')
    assert cache._cmake_compiler_matches(stage, compiler)
    cache_file.write_text('CMAKE_CXX_COMPILER:FILEPATH=/bin/false\n')
    assert not cache._cmake_compiler_matches(stage, compiler)
    cache_file.write_text('CMAKE_CXX_COMPILER:FILEPATH=/usr/bin/c++\n'
                          'CMAKE_CXX_COMPILER_LAUNCHER:STRING=ccache\n')
    assert not cache._cmake_compiler_matches(stage, compiler)
    cache_file.write_text('CMAKE_CXX_COMPILER:FILEPATH=/usr/bin/c++\n')
    monkeypatch.setenv('CMAKE_TOOLCHAIN_FILE', '/outside/toolchain.cmake')
    assert not cache._cmake_compiler_matches(stage, compiler)
    monkeypatch.delenv('CMAKE_TOOLCHAIN_FILE')

    header = tmp_path / 'system-header.hpp'
    header.write_bytes(b'first')
    source = stage / 'source/cpp/teleop/CMakeLists.txt'
    for name in cache.DEPFILES:
        depfile = stage / 'source/cpp/teleop/build/CMakeFiles' / name
        depfile.parent.mkdir(parents=True, exist_ok=True)
        depfile.write_text(f'CMakeFiles/{name.removesuffix(".d")}: \\\n {source} \\\n {header}\n')
    manifest = cache._manifest(stage)
    first = cache._compile_dependencies(stage, manifest)
    assert first == {str(header): cache.sha256_file(header)}
    header.write_bytes(b'second')
    assert cache._compile_dependencies(stage, manifest) != first
    rogue = stage / 'source/cpp/teleop/build/generated.hpp'
    rogue.write_bytes(b'not in source manifest')
    depfile.write_text(f'CMakeFiles/{name.removesuffix(".d")}: {source} {rogue}\n')
    with pytest.raises(ValueError, match='untracked generated'):
        cache._compile_dependencies(stage, manifest)


def test_build_script_attempts_cache_before_configure_and_receipts_its_proof():
    from fleet_deploy import component_plan, native_build_steps

    lines = native_build_steps('/release/source', '/cache/sdk', '/release', '/checkout', 'arm-owner')
    reuse = next(i for i, line in enumerate(lines) if 'fleet_component_cache.py reuse' in line)
    configure = next(i for i, line in enumerate(lines) if 'cmake -S ' in line)
    build = next(i for i, line in enumerate(lines) if 'cmake --build' in line)
    record = next(i for i, line in enumerate(lines) if 'fleet_component_cache.py record' in line)
    assert reuse < configure < build < record
    # A reused planner skips only its own target; teleop and recovery always build.
    assert lines[build].startswith('if [ "$PLAN_REUSED" = 0 ]')
    reused_branch = lines[build].split('; else ', 1)[1]
    assert 'path_plan_check' not in reused_branch and 'wxai_teleop' in reused_branch
    plan = component_plan({'fleetctl': {'features': []}}, [], True, True)
    assert 'source/planner-component.json' in plan['components']
    assert plan['launch_binaries'] == ['fleetctl', 'path_plan_check']
