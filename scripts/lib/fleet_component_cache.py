"""Reuse the offline C++ planner only when its complete declared inputs match.

The planner is a standalone executable. Its source closure is deliberately
conservative: all tracked C++ teleop and config files, plus the local C++
compiler CMake actually chose, its external headers, CMake, build flags and
resolved runtime libraries. Other native targets still build normally. A
copied planner carries that proof into the new release's build receipt.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import re
import shlex
import shutil
import subprocess
import sys
from pathlib import Path

import fleet_release
import fleet_source
from tatbot_digest import sha256_file

SCHEMA = 'tatbot.receipt/1'
KIND = 'planner-component'
RECORD = 'planner-component.json'
PLANNER = Path('source/cpp/teleop/build/path_plan_check')
SOURCE_PREFIXES = ('cpp/teleop/', 'config/')
BUILD_RECIPES = ('scripts/lib/fleet_component_cache.py',
                 'scripts/lib/fleet_deploy.py',
                 'scripts/lib/fleet_toolchain.sh')
BUILD_FLAGS = frozenset({'CC', 'CXX', 'CPPFLAGS', 'LDFLAGS', 'CPATH',
                         'CPLUS_INCLUDE_PATH', 'LIBRARY_PATH', 'TATBOT_NO_ARM_SDK'})
DEPFILES = ('path_plan_check.dir/path_plan_check.cpp.o.d',
            'square_probe.dir/square_probe.cpp.o.d')
SHA1 = re.compile(r'[0-9a-f]{40}\Z')


def _manifest(stage: Path) -> dict:
    manifest = json.loads((stage / 'source-manifest.json').read_bytes())
    fleet_source.verify_manifest_tree(stage / 'source', manifest,
                                        manifest['source_sha'])
    return manifest


def _source_inputs(manifest: dict) -> dict:
    selected = {name: row for name, row in manifest['files'].items()
                if name.startswith(SOURCE_PREFIXES) or name in BUILD_RECIPES}
    if (not any(name.startswith('cpp/teleop/') for name in selected)
            or not set(BUILD_RECIPES) <= selected.keys()):
        raise ValueError('planner source closure is absent')
    return selected


def _input_digest(manifest: dict) -> str:
    selected = _source_inputs(manifest)
    return hashlib.sha256(json.dumps(selected, sort_keys=True,
                                     separators=(',', ':')).encode()).hexdigest()


def _command_binary(name: str) -> dict:
    command = shutil.which(name)
    if not command:
        raise ValueError(f'{name} is unavailable')
    resolved = Path(command).resolve(strict=True)
    return {'path': str(resolved), 'sha256': sha256_file(resolved)}


def _linked_libraries(binary: Path) -> dict:
    lines = subprocess.check_output(['ldd', str(binary)], text=True).splitlines()
    if any('not found' in line for line in lines):
        raise ValueError('planner has a missing runtime library')
    paths = set()
    for line in lines:
        for token in line.split():
            if token.startswith('/'):
                paths.add(Path(token).resolve(strict=True))
    if not paths:
        raise ValueError('planner runtime library set is unavailable')
    return {str(path): sha256_file(path) for path in sorted(paths)}


def _environment(binary: Path) -> dict:
    compiler = shlex.split(os.environ.get('CXX', 'c++'))
    if len(compiler) == 1:
        selected = compiler[0]
        compiler_target = subprocess.check_output([selected, '-dumpmachine'], text=True).strip()
        compiler_binary = _command_binary(selected)
        reusable = True
    else:
        # CMake accepts wrappers and compound compiler commands. They can be
        # built and receipted, but this cache cannot prove their byte closure.
        compiler_target = None
        compiler_binary = None
        reusable = False
    return {
        'system': platform.system(), 'machine': platform.machine(),
        'compiler_target': compiler_target, 'compiler': compiler_binary,
        'compiler_reusable': reusable, 'cmake': _command_binary('cmake'),
        'library_probe': _command_binary('ldd'),
        'flags': {name: value for name, value in os.environ.items()
                  if name in BUILD_FLAGS or name.startswith(('CMAKE_', 'CXX', 'LD_'))},
        'runtime_libraries': _linked_libraries(binary),
    }


def _cmake_compiler_matches(stage: Path, compiler: dict | None) -> bool:
    """The compiler CMake actually selected must be the one we fingerprinted."""
    if compiler is None or any(name.startswith('CMAKE_') and name != 'CMAKE_BUILD_PARALLEL_LEVEL'
                               for name in os.environ):
        return False
    cache = stage / 'source/cpp/teleop/build/CMakeCache.txt'
    if cache.is_symlink() or not cache.is_file() or cache.stat().st_size > 2_000_000:
        return False
    entries = {}
    for line in cache.read_text().splitlines():
        if line.startswith(('#', '//')) or '=' not in line:
            continue
        key, value = line.split('=', 1)
        entries.setdefault(key.split(':', 1)[0], []).append(value)
    selected = entries.get('CMAKE_CXX_COMPILER', [])
    if (len(selected) != 1 or not selected[0].startswith('/')
            or any(entries.get(key, ['']) != [''] for key in
                   ('CMAKE_CXX_COMPILER_LAUNCHER', 'CMAKE_TOOLCHAIN_FILE'))):
        return False
    try:
        return str(Path(selected[0]).resolve(strict=True)) == compiler['path']
    except (OSError, KeyError, TypeError):
        return False


def _compile_dependencies(stage: Path, manifest: dict) -> dict:
    """Hash external headers from both planner object depfiles.

    An untracked/generated input inside the source tree or a depfile syntax
    this bounded reader cannot prove makes the component nonreusable.
    """
    source = (stage / 'source').resolve(strict=True)
    tracked = _source_inputs(manifest)
    external = {}
    for name in DEPFILES:
        depfile = stage / 'source/cpp/teleop/build/CMakeFiles' / name
        if (depfile.is_symlink() or not depfile.is_file()
                or depfile.stat().st_size > 1_000_000):
            raise ValueError('planner compiler dependency file is absent or redirected')
        raw = depfile.read_text()
        merged = raw.replace('\\\n', ' ')
        target, separator, paths = merged.partition(':')
        if (not separator or '\\' in merged or not target.endswith(name.removesuffix('.d'))
                or not paths.split()):
            raise ValueError('planner compiler dependency file has unknown syntax')
        for token in paths.split():
            path = Path(token)
            if not path.is_absolute() or path.is_symlink():
                raise ValueError('planner compiler dependency is not a plain absolute path')
            resolved = path.resolve(strict=True)
            if resolved.is_relative_to(source):
                relative = resolved.relative_to(source).as_posix()
                if relative not in tracked or tracked[relative].get('kind') != 'file':
                    raise ValueError('planner has an untracked generated source dependency')
            else:
                if not resolved.is_file():
                    raise ValueError('planner external dependency is not a file')
                external[str(resolved)] = sha256_file(resolved)
    if not external:
        raise ValueError('planner compiler dependency closure is empty')
    return dict(sorted(external.items()))


def _current_external_dependencies(recorded: dict) -> dict:
    """Rehash a prior compiler's external headers without needing its depfiles."""
    if not isinstance(recorded, dict) or not recorded or len(recorded) > 10_000:
        raise ValueError('planner external dependency record is invalid')
    current = {}
    for name, digest in recorded.items():
        if not isinstance(name, str) or not isinstance(digest, str):
            raise ValueError('planner external dependency record is invalid')
        path = Path(name)
        if (not path.is_absolute() or path.is_symlink() or not path.is_file()
                or str(path.resolve(strict=True)) != name
                or re.fullmatch(r'[0-9a-f]{64}', digest) is None):
            raise ValueError('planner external dependency is absent or redirected')
        current[name] = sha256_file(path)
    return current


def _bound_component(stage: Path, manifest: dict, binary: Path,
                     receipt: dict) -> dict:
    """Read a prior proof only if its installed build receipt binds its bytes."""
    path = stage / 'source' / RECORD
    if path.is_symlink() or not path.is_file():
        raise ValueError('installed planner component is absent or redirected')
    named = receipt.get('binaries', {}).get('path_plan_check')
    if (receipt.get('source_sha') != stage.name or receipt.get('release') != str(stage)
            or (stage / 'build.sha').read_text() != stage.name
            or not named or named.get('sha256') != sha256_file(binary)
            or receipt.get('components', {}).get('source/' + RECORD) != sha256_file(path)):
        raise ValueError('installed planner component differs from its build receipt')
    recorded = json.loads(path.read_bytes())
    if (not isinstance(recorded, dict) or recorded.get('schema') != SCHEMA
            or recorded.get('kind') != KIND or recorded.get('source_sha') != stage.name
            or recorded.get('source_inputs_sha256') != _input_digest(manifest)
            or recorded.get('binary_sha256') != named['sha256']
            or recorded.get('source_closure_reusable') is not True
            or recorded.get('compile_closure_reusable') is not True
            or not isinstance(recorded.get('environment'), dict)
            or recorded['environment'].get('compiler_reusable') is not True
            or recorded.get('compile_dependencies') !=
            _current_external_dependencies(recorded.get('compile_dependencies'))):
        raise ValueError('installed planner compile proof differs')
    return recorded


def _qualified_environment(compiler_stage: Path, binary: Path) -> dict:
    environment = _environment(binary)
    proved = _cmake_compiler_matches(compiler_stage, environment.get('compiler'))
    environment['compiler_reusable'] = bool(environment.get('compiler_reusable') and proved)
    return environment


def _record_compile_proof(stage: Path, manifest: dict, binary: Path,
                          reused_from: str | None) -> tuple[dict, dict | None, bool]:
    if reused_from is None:
        environment = _qualified_environment(stage, binary)
        try:
            dependencies = _compile_dependencies(stage, manifest)
        except (OSError, ValueError, UnicodeError):
            dependencies = None
        return environment, dependencies, bool(environment['compiler_reusable'])

    previous = stage.parent / reused_from
    previous_manifest = _manifest(previous)
    if _input_digest(previous_manifest) != _input_digest(manifest):
        raise ValueError('reused planner source differs')
    proof = _bound_component(previous, previous_manifest, previous / PLANNER,
                             fleet_release.load(previous / 'build.json'))
    if sha256_file(previous / PLANNER) != sha256_file(binary):
        raise ValueError('reused planner executable differs')
    environment = _environment(binary)
    if environment != proof['environment']:
        raise ValueError('reused planner toolchain or runtime differs')
    return environment, proof['compile_dependencies'], True


def _record(stage: Path, *, reused_from: str | None = None) -> dict:
    manifest = _manifest(stage)
    binary = stage / PLANNER
    if binary.is_symlink() or not binary.is_file() or not os.access(binary, os.X_OK):
        raise ValueError('planner executable is absent')
    path = stage / 'source' / RECORD
    existing = json.loads(path.read_bytes()) if path.exists() else None
    if existing is not None and reused_from is None:
        if not isinstance(existing, dict):
            raise ValueError('invalid planner component record')
        reused_from = existing.get('reused_from')
    if reused_from is not None and not SHA1.fullmatch(reused_from):
        raise ValueError('invalid planner reuse source')
    environment, dependencies, compiler_proved = _record_compile_proof(
        stage, manifest, binary, reused_from)
    result = {'schema': SCHEMA, 'kind': KIND,
              'source_sha': manifest['source_sha'],
              'source_inputs_sha256': _input_digest(manifest),
              'source_closure_reusable': all(row.get('kind') == 'file'
                                             for row in _source_inputs(manifest).values()),
              'compile_dependencies': dependencies,
              'compile_closure_reusable': dependencies is not None and compiler_proved,
              'binary_sha256': sha256_file(binary),
              'environment': environment,
              'reused_from': reused_from}
    if path.exists():
        if existing != result:
            raise ValueError('planner component changed after preparation')
    else:
        path.write_text(json.dumps(result, sort_keys=True, indent=2) + '\n')
    return result


def _previous_stage(stage: Path, checkout: Path, node: str) -> Path:
    stamp = json.loads((checkout / fleet_release.STAMP).read_bytes())
    if not isinstance(stamp, dict) or not isinstance(stamp.get('release'), str):
        raise ValueError('installed build stamp is invalid')
    old = Path(stamp['release'])
    if (not old.is_absolute() or old.is_symlink() or old == stage
            or old.parent.resolve() != stage.parent.resolve()
            or not SHA1.fullmatch(old.name)
            or stamp.get('node') != node or stamp.get('source_sha') != old.name):
        raise ValueError('installed planner release is not a prior release on this node')
    return old


def reuse(stage: Path, checkout: Path, node: str) -> dict:
    """Copy a verified prior planner into a new staged source or raise a miss."""
    new = _manifest(stage)
    old_stage = _previous_stage(stage, checkout, node)
    if (old_stage / fleet_release.INCOMPLETE).exists():
        raise ValueError('installed planner build was interrupted')
    old_receipt = fleet_release.verify(old_stage / 'source')
    if (old_receipt.get('node') != node or old_receipt['source_sha'] != old_stage.name
            or (old_stage / 'build.sha').read_text() != old_stage.name):
        raise ValueError('installed planner release identity differs')
    old_binary = old_stage / PLANNER
    if old_binary.is_symlink():
        raise ValueError('installed planner component is redirected')
    old_manifest = _manifest(old_stage)
    recorded = _bound_component(old_stage, old_manifest, old_binary, old_receipt)
    if (recorded['source_inputs_sha256'] != _input_digest(new)
            or any(row.get('kind') != 'file' for row in _source_inputs(new).values())
            or recorded['environment'] != _environment(old_binary)):
        raise ValueError('planner input, binary or toolchain closure differs')
    new_binary = stage / PLANNER
    if new_binary.exists() or new_binary.is_symlink():
        raise ValueError('new planner stage is not empty')
    new_binary.parent.mkdir(parents=True, exist_ok=True)
    temporary = new_binary.with_name(new_binary.name + '.next')
    shutil.copy2(old_binary, temporary)
    temporary.replace(new_binary)
    try:
        if _environment(new_binary) != recorded['environment']:
            raise ValueError('planner runtime ABI changes at the new release path')
        return _record(stage, reused_from=old_stage.name)
    except Exception:
        new_binary.unlink(missing_ok=True)
        raise


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    for name in ('reuse', 'record'):
        command = commands.add_parser(name)
        command.add_argument('--stage', type=Path, required=True)
        if name == 'reuse':
            command.add_argument('--checkout', type=Path, required=True)
            command.add_argument('--node', required=True)
    args = parser.parse_args(argv)
    try:
        if args.command == 'reuse':
            result = reuse(args.stage, args.checkout, args.node)
        else:
            result = _record(args.stage)
        print(json.dumps({'planner': 'reused' if result['reused_from'] else 'built',
                          'binary_sha256': result['binary_sha256']}))
        return 0
    except (OSError, ValueError, TypeError, AttributeError, KeyError,
            subprocess.SubprocessError) as error:
        print(f'planner component {"miss" if args.command == "reuse" else "refused"}: {error}',
              file=sys.stderr)
        return 1 if args.command == 'reuse' else 3


if __name__ == '__main__':
    raise SystemExit(main())
