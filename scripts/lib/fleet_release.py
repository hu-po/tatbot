"""A release's build receipt: what a deploy built, what a launcher may run.

`build` runs on the build node as the last line of a full deploy's remote build
and writes `<stage>/build.json`, a `tatbot.receipt/1` of kind `build` naming the
source sha and the sha256 of every binary a checkout resolves through it: the
bus controller and, on the arm node, the offline planner. The receipt also
hashes staged service executables for an exact same-source build reuse check. `verify` runs where a launcher runs -- the
mirror checkout a full deploy stamped with `.tatbot-build.json`, a Gitless
viewer overlay verified against its source manifest, or a release's own
`source/` -- and refuses when the checkout has moved past its receipt,
carries tracked edits, or any named binary no longer has the digest the deploy
recorded. Neither command runs cmake or cargo. A reuse probe hashes generated
build trees in full, so its cost scales with their bytes rather than just the
named binaries.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import stat
import subprocess
from pathlib import Path

from fleet_source import verify_archive, verify_overlay
from tatbot_digest import sha256_file

SCHEMA = 'tatbot.receipt/1'
KIND = 'build'
RECEIPT = 'build.json'
STAMP = '.tatbot-build.json'

# What a checkout resolves through its receipt, relative to the release
# directory. The bus controller is staged into bin/; the planner is built in
# the release's source tree. Keep the stamped release protected from pruning
# for as long as a checkout uses it.
LAUNCH_BINARIES = {
    'fleetctl': 'bin/fleetctl',
    'path_plan_check': 'source/cpp/teleop/build/path_plan_check',
}
SHA1 = re.compile(r'[0-9a-f]{40}\Z')
SHA256 = re.compile(r'[0-9a-f]{64}\Z')
INCOMPLETE = 'build.incomplete'


def _component_path(stage: Path, relative: str) -> Path:
    """Resolve a staged artifact without accepting an escaping receipt path."""
    if not isinstance(relative, str):
        raise ValueError('invalid component path type')
    path = Path(relative)
    if path.is_absolute() or not path.parts or '..' in path.parts:
        raise ValueError(f'invalid component path: {relative}')
    result = stage / path
    # uv's venv interpreter is normally a leaf symlink into its pinned Python
    # toolchain. Hash the executable it actually runs, while refusing a parent
    # symlink that could redirect an arbitrary component tree out of a release.
    if not result.parent.resolve().is_relative_to(stage.resolve()):
        raise ValueError(f'component path escapes release: {relative}')
    return result


def _tree_path(stage: Path, relative: str) -> Path:
    root = _component_path(stage, relative)
    if root.is_symlink() or not root.is_dir():
        raise ValueError(f'generated tree missing or redirected: {relative}')
    return root


def _linked_content(path: Path, root: Path) -> list:
    target = os.readlink(path)
    resolved = path.resolve(strict=True)
    if resolved.is_dir():
        if not resolved.is_relative_to(root):
            raise ValueError(f'generated tree directory symlink escapes: {path}')
        bound = ['directory', stat.S_IMODE(resolved.stat().st_mode)]
    elif resolved.is_file():
        bound = ['file', stat.S_IMODE(resolved.stat().st_mode),
                 resolved.stat().st_size, sha256_file(resolved)]
    else:
        raise ValueError(f'unsupported generated tree symlink: {path}')
    return ['symlink', stat.S_IMODE(path.lstat().st_mode), target, bound]


def tree_contents(root: Path) -> dict:
    """Hash every generated path, file byte, mode and symlink in a tree.

    Directory symlinks are not traversed; an external directory symlink is
    refused. External file symlinks bind both the link text and target bytes.
    This is deliberately a full scan on build/reuse, never a cached mtime key.
    """
    if root.is_symlink() or not root.is_dir():
        raise ValueError(f'generated tree missing or redirected: {root}')
    digest = hashlib.sha256()
    entries = 0
    file_bytes = 0
    resolved_root = root.resolve()
    digest.update(json.dumps(['.', 'directory', stat.S_IMODE(root.stat().st_mode)],
                             separators=(',', ':')).encode() + b'\n')

    def visit(directory):
        nonlocal entries, file_bytes
        for path in sorted(directory.iterdir(), key=lambda item: item.name):
            relative = path.relative_to(root).as_posix()
            mode = path.lstat().st_mode
            if stat.S_ISDIR(mode):
                record = [relative, 'directory', stat.S_IMODE(mode)]
                digest.update(json.dumps(record, separators=(',', ':')).encode() + b'\n')
                entries += 1
                visit(path)
            elif stat.S_ISREG(mode):
                size = path.stat().st_size
                record = [relative, 'file', stat.S_IMODE(mode), size, sha256_file(path)]
                digest.update(json.dumps(record, separators=(',', ':')).encode() + b'\n')
                entries += 1
                file_bytes += size
            elif stat.S_ISLNK(mode):
                record = [relative, *_linked_content(path, resolved_root)]
                digest.update(json.dumps(record, separators=(',', ':')).encode() + b'\n')
                entries += 1
            else:
                raise ValueError(f'unsupported generated tree member: {path}')

    visit(root)
    if not entries:
        raise ValueError(f'empty generated tree: {root}')
    return {'sha256': digest.hexdigest(), 'entries': entries, 'file_bytes': file_bytes}


def _python_toolchain(venv: Path) -> Path | None:
    """Find a bounded uv-managed base Python tree, if this venv uses one.

    A system interpreter commonly resolves under /usr. Scanning all of /usr
    would turn a build receipt into an expensive, unstable OS snapshot; such
    a venv can be built but is ineligible for generated-tree build reuse.
    """
    interpreter = (venv / 'bin/python').resolve(strict=True)
    if not interpreter.is_file() or interpreter.parent.name != 'bin':
        return None
    toolchain = interpreter.parent.parent
    if toolchain.is_relative_to(venv.resolve()):
        return None
    if (toolchain.parent.name != 'python' or toolchain.parent.parent.name != 'uv'
            or not toolchain.name.startswith('cpython-') or not (toolchain / 'lib').is_dir()):
        return None
    return toolchain


def generated_contents(stage: Path, trees) -> tuple[dict, dict]:
    """Receipt all requested build trees and each venv's external stdlib."""
    named = {}
    toolchains = {}
    for relative in sorted(set(trees)):
        root = _tree_path(stage, relative)
        named[relative] = tree_contents(root)
        if root.name in ('.venv', 'observer-venv'):
            toolchain = _python_toolchain(root)
            if toolchain is not None:
                toolchains[relative] = {'path': str(toolchain), **tree_contents(toolchain)}
    return named, toolchains


def build(stage: Path, source_sha: str, node: str, features, binaries, components=(), trees=()) -> dict:
    """The receipt for `stage`; every named binary must exist and be executable."""
    if not stage.is_absolute():
        raise ValueError('release root must be absolute')
    if not SHA1.match(source_sha):
        raise ValueError('invalid source identity')
    unknown = sorted(set(binaries) - set(LAUNCH_BINARIES))
    if unknown:
        raise ValueError('not a launch binary: ' + ', '.join(unknown))
    named = {}
    for name in sorted(set(binaries)):
        path = _component_path(stage, LAUNCH_BINARIES[name])
        if not path.is_file() or not (path.stat().st_mode & 0o111):
            raise ValueError(f'launch binary missing or not executable: {path}')
        named[name] = {'path': str(path), 'sha256': sha256_file(path)}
    if not named:
        raise ValueError('a build receipt names at least one launch binary')
    named_components = {}
    for relative in sorted(set(components)):
        path = _component_path(stage, relative)
        if not path.is_file():
            raise ValueError(f'component missing: {path}')
        named_components[relative] = sha256_file(path)
    named_trees, toolchains = generated_contents(stage, trees)
    return {'schema': SCHEMA, 'kind': KIND, 'source_sha': source_sha, 'node': node,
            'release': str(stage), 'features': sorted(set(features)), 'binaries': named,
            'components': named_components, 'generated_trees': named_trees,
            'python_toolchains': toolchains}


def load(path: Path) -> dict:
    """The receipt at `path`, refused unless it has the build receipt's shape."""
    try:
        receipt = json.loads(path.read_bytes())
    except FileNotFoundError:
        raise ValueError(f'no build receipt at {path}; deploy first') from None
    if not _receipt_shape(receipt):
        raise ValueError(f'invalid build receipt at {path}')
    _validate_binaries(receipt, path)
    _validate_components(receipt, path)
    _validate_trees(receipt, path)
    return receipt


def _validate_binaries(receipt, path):
    for name, record in receipt['binaries'].items():
        if (name not in LAUNCH_BINARIES or not isinstance(record, dict)
                or not isinstance(record.get('path'), str)
                or not SHA256.match(str(record.get('sha256', '')))):
            raise ValueError(f'invalid build receipt binary {name!r} at {path}')


def _validate_components(receipt, path):
    for relative, digest in receipt.get('components', {}).items():
        _component_path(Path(receipt['release']), relative)
        if not SHA256.match(str(digest)):
            raise ValueError(f'invalid build receipt component {relative!r} at {path}')


def _validate_trees(receipt, path):
    for relative, record in receipt.get('generated_trees', {}).items():
        _component_path(Path(receipt['release']), relative)
        if not _tree_record(record):
            raise ValueError(f'invalid generated tree receipt {relative!r} at {path}')
    for relative, record in receipt.get('python_toolchains', {}).items():
        if (relative not in receipt.get('generated_trees', {})
                or not isinstance(record, dict)
                or not isinstance(record.get('path'), str)
                or not Path(record['path']).is_absolute()
                or not _tree_record(record)):
            raise ValueError(f'invalid Python toolchain receipt {relative!r} at {path}')


def _receipt_shape(receipt) -> bool:
    if not isinstance(receipt, dict):
        return False
    release = receipt.get('release')
    if not isinstance(release, str):
        return False
    return all((receipt.get('schema') == SCHEMA, receipt.get('kind') == KIND,
                SHA1.match(str(receipt.get('source_sha', ''))) is not None,
                Path(release).is_absolute(), isinstance(receipt.get('features'), list),
                isinstance(receipt.get('binaries'), dict), bool(receipt.get('binaries')),
                isinstance(receipt.get('components', {}), dict),
                isinstance(receipt.get('generated_trees', {}), dict),
                isinstance(receipt.get('python_toolchains', {}), dict)))


def _tree_record(record):
    return (isinstance(record, dict) and SHA256.match(str(record.get('sha256', ''))) is not None
            and type(record.get('entries')) is int and record['entries'] > 0
            and type(record.get('file_bytes')) is int and record['file_bytes'] >= 0)


def checkout_identity(checkout: Path, expected: str, release: Path | None = None) -> str:
    """Verify the selected checkout form against its retained source identity."""
    if (checkout / '.git').exists():
        head = subprocess.check_output(['git', '-C', str(checkout), 'rev-parse', 'HEAD'], text=True).strip()
        if subprocess.check_output(['git', '-C', str(checkout), 'status', '--porcelain',
                                    '--untracked-files=no'], text=True).strip():
            raise ValueError('checkout has tracked edits')
        return head
    if checkout.name == 'source' and (checkout.parent / RECEIPT).is_file():
        return verify_archive(checkout, expected)
    overlay = checkout / '.tatbot-deploy.json'
    if overlay.is_file() and release is not None:
        if json.loads(overlay.read_bytes()).get('source_commit') != expected:
            raise ValueError('viewer overlay source differs from its build receipt')
        return verify_overlay(checkout, release / 'source-manifest.json', expected)
    raise ValueError('neither a git checkout nor a built release')


def verify(checkout: Path) -> dict:
    """The receipt that binds `checkout`, or a ValueError naming what moved."""
    receipt_path = (checkout.parent / RECEIPT if checkout.name == 'source' and not (checkout / '.git').exists()
                    else checkout / STAMP)
    receipt = load(receipt_path)
    if (Path(receipt['release']) / INCOMPLETE).exists():
        raise ValueError('previous build did not finish')
    head = checkout_identity(checkout, receipt['source_sha'], Path(receipt['release']))
    if head != receipt['source_sha']:
        raise ValueError(f'checkout is at {head[:12]}, the build receipt names {receipt["source_sha"][:12]}; '
                         'deploy this revision or check out the built one')
    if not Path(receipt['release']).is_dir():
        raise ValueError(f'the built release is gone: {receipt["release"]}')
    for name, record in receipt['binaries'].items():
        path = Path(record['path'])
        expected = _component_path(Path(receipt['release']), LAUNCH_BINARIES[name])
        if path != expected:
            raise ValueError(f'launch binary path differs: {name} at {path}')
        if not path.is_file() or not (path.stat().st_mode & 0o111):
            raise ValueError(f'launch binary missing or not executable: {name} at {path}')
        if sha256_file(path) != record['sha256']:
            raise ValueError(f'launch binary changed since its build: {name} at {path}')
    for relative, digest in receipt.get('components', {}).items():
        path = _component_path(Path(receipt['release']), relative)
        if not path.is_file() or sha256_file(path) != digest:
            raise ValueError(f'component changed since its build: {relative} at {path}')
    return receipt


def reusable(stage: Path, source_sha: str, node: str, features, binaries,
             components, manifest_sha256: str, trees=()) -> dict:
    """Prove a complete same-source build before skipping all build commands."""
    if (stage / INCOMPLETE).exists():
        raise ValueError('previous build did not finish')
    if (stage / 'build.sha').read_text() != source_sha:
        raise ValueError('build source differs')
    manifest = (stage / 'source-manifest.json').read_bytes()
    if hashlib.sha256(manifest).hexdigest() != manifest_sha256:
        raise ValueError('source manifest differs')
    receipt = verify(stage / 'source')
    if (receipt['source_sha'] != source_sha or receipt.get('node') != node
            or receipt['release'] != str(stage)
            or receipt['features'] != sorted(set(features))
            or set(receipt['binaries']) != set(binaries)
            or set(receipt.get('components', {})) != set(components)
            or set(receipt.get('generated_trees', {})) != set(trees)):
        raise ValueError('build component set differs')
    for name, record in receipt['binaries'].items():
        if record['path'] != str(stage / LAUNCH_BINARIES[name]):
            raise ValueError(f'launch binary path differs: {name}')
    named_trees, toolchains = generated_contents(stage, trees)
    venvs = {relative for relative in trees if Path(relative).name in ('.venv', 'observer-venv')}
    if set(toolchains) != venvs:
        raise ValueError('generated environment has no bounded Python toolchain receipt')
    if (receipt.get('generated_trees', {}) != named_trees
            or receipt.get('python_toolchains', {}) != toolchains):
        raise ValueError('generated runtime tree or Python toolchain changed since build')
    return receipt


def planner_binary(checkout: Path) -> Path:
    """The offline planner an offline rung (a compile's executor preflight)
    runs: on a stamped checkout the build receipt's `path_plan_check`, so a
    trace's planner digest is the receipt's, and the checkout's own build
    otherwise (a development worktree, or a node whose receipt names no
    planner)."""
    own = checkout / 'cpp/teleop/build/path_plan_check'
    if not (checkout / STAMP).is_file():
        return own
    record = verify(checkout)['binaries'].get('path_plan_check')
    return Path(record['path']) if record else own


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    builder = commands.add_parser('build', help='write <stage>/build.json after a deploy build')
    builder.add_argument('--stage', type=Path, required=True)
    builder.add_argument('--sha', required=True)
    builder.add_argument('--node', required=True)
    builder.add_argument('--features', default='', help='comma-separated build features to record')
    builder.add_argument('--binary', action='append', default=[], choices=sorted(LAUNCH_BINARIES),
                         help='a launch binary the receipt must name; repeatable')
    builder.add_argument('--component', action='append', default=[],
                         help='release-relative staged service artifact; repeatable')
    builder.add_argument('--tree', action='append', default=[],
                         help='generated release tree whose full contents are bound; repeatable')
    verifier = commands.add_parser('verify', help='check a checkout against its build receipt')
    verifier.add_argument('--checkout', type=Path, required=True)
    verifier.add_argument('--binary', choices=sorted(LAUNCH_BINARIES),
                          help='print this binary\'s verified path instead of the receipt')
    verifier.add_argument('--feature', action='append', default=[],
                          help='a daemon feature the receipt must carry; repeatable')
    reuse = commands.add_parser('reusable', help='verify a complete staged build for exact reuse')
    reuse.add_argument('--stage', type=Path, required=True)
    reuse.add_argument('--sha', required=True)
    reuse.add_argument('--node', required=True)
    reuse.add_argument('--features', default='')
    reuse.add_argument('--manifest-sha256', required=True)
    reuse.add_argument('--binary', action='append', default=[], choices=sorted(LAUNCH_BINARIES))
    reuse.add_argument('--component', action='append', default=[])
    reuse.add_argument('--tree', action='append', default=[])
    args = parser.parse_args(argv)
    try:
        if args.command == 'build':
            stage = args.stage.resolve()
            receipt = build(stage, args.sha, args.node,
                            [f for f in args.features.split(',') if f], args.binary, args.component,
                            args.tree)
            temporary = stage / (RECEIPT + '.next')
            temporary.write_text(json.dumps(receipt, indent=2) + '\n')
            temporary.replace(stage / RECEIPT)
            (stage / INCOMPLETE).unlink(missing_ok=True)
            print(stage / RECEIPT)
            return
        if args.command == 'reusable':
            reusable(args.stage.resolve(), args.sha, args.node,
                     [f for f in args.features.split(',') if f], args.binary,
                     args.component, args.manifest_sha256, args.tree)
            print(args.sha)
            return
        receipt = verify(args.checkout.resolve())
        absent = sorted(set(args.feature) - set(receipt['features']))
        if absent:
            raise ValueError('the daemon was built without ' + ', '.join(absent)
                             + '; the release carries ' + (', '.join(receipt['features']) or 'no features'))
        if args.binary:
            if args.binary not in receipt['binaries']:
                raise ValueError(f'the build receipt names no {args.binary}')
            print(receipt['binaries'][args.binary]['path'])
        else:
            print(json.dumps(receipt, indent=2))
    except (OSError, ValueError, subprocess.SubprocessError) as error:
        parser.exit(3, f'fleet release refused: {error}\n')


if __name__ == '__main__':
    main()
