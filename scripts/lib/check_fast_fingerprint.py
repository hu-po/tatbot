"""Content key for completed, read-only jobs in the bare fast check.

The key follows the working tree, not mtimes. Explicit and push-time checks
still execute normally; this only lets a second bare check reuse a full PASS
when its repository, relevant environment and checker are unchanged.
"""

from __future__ import annotations

import hashlib
import os
import shutil
import stat
import subprocess
import sys
from pathlib import Path

JOBS = frozenset({'lint-sh', 'cli', 'body-single-path', 'export'})
ENV_NAMES = frozenset({'PATH', 'HOME', 'LANG', 'CI', 'BASH_ENV'})
ENV_PREFIXES = ('TATBOT_', 'PYTHON', 'UV_', 'XDG_', 'LC_', 'GIT_', 'LD_', 'SHELLCHECK_')
TOOLS = ('python3', 'bash', 'git', 'grep', 'sha256sum')


def _git_bytes(root: Path, *args: str) -> bytes:
    return subprocess.check_output(('git', *args), cwd=root)


def _field(hasher, *items: bytes) -> None:
    for item in items:
        hasher.update(len(item).to_bytes(8, 'big'))
        hasher.update(item)


def _file(hasher, root: Path, relative: Path) -> None:
    path = root / relative
    info = path.lstat()
    if not stat.S_ISREG(info.st_mode):
        # A symlink may point outside the tree and change without its link
        # text changing. Leave this run uncached instead of guessing its closure.
        raise ValueError(f'non-regular fast-check input: {relative}')
    digest = hashlib.sha256()
    with path.open('rb') as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b''):
            digest.update(chunk)
    _field(hasher, os.fsencode(relative), str(stat.S_IMODE(info.st_mode)).encode(), digest.digest())


def _tool(hasher, name: str) -> None:
    resolved = shutil.which(name)
    if resolved is None:
        raise ValueError(f'checker tool is unavailable: {name}')
    path = Path(resolved).resolve(strict=True)
    _field(hasher, name.encode(), os.fsencode(path))
    _file(hasher, path.parent, Path(path.name))


def _tracked_inputs(hasher, root: Path) -> None:
    # The index contributes tracked membership, modes and staged blob IDs.
    # Some checks also inspect Git's view of the tree, so staging different
    # bytes must invalidate a completed PASS even if working bytes stay put.
    index = _git_bytes(root, 'ls-files', '--stage', '-z')
    for row in index.split(b'\0'):
        if row:
            mode_stage, separator, name = row.partition(b'\t')
            if not separator:
                raise ValueError('invalid Git index row')
            mode, object_id, stage = mode_stage.split(b' ')
            _field(hasher, b'index', name, mode, object_id, stage)
    names = sorted(set(filter(None, _git_bytes(root, 'ls-files', '-z', '--cached',
                                          '--others', '--exclude-standard').split(b'\0'))))
    for name in names:
        _file(hasher, root, Path(os.fsdecode(name)))


def _generated_body(hasher, root: Path) -> None:
    # The body freeze also scans this ignored generated tree if it exists.
    dist = root / 'web/inkmap/dist'
    if dist.exists() or dist.is_symlink():
        if dist.is_symlink() or not dist.is_dir():
            raise ValueError('generated Inkmap tree is redirected')
        _field(hasher, b'inkmap-dist-present')
        for path in sorted(dist.rglob('*')):
            relative = path.relative_to(root)
            if path.is_symlink():
                raise ValueError('generated Inkmap tree contains a symlink')
            if path.is_dir():
                _field(hasher, b'directory', os.fsencode(relative),
                       str(stat.S_IMODE(path.stat().st_mode)).encode())
            else:
                _file(hasher, root, relative)
    else:
        _field(hasher, b'inkmap-dist-absent')


def _environment(hasher) -> None:
    for name, value in sorted(os.environ.items()):
        if name in ENV_NAMES or name.startswith(ENV_PREFIXES):
            _field(hasher, b'env', os.fsencode(name), os.fsencode(value))


def _checker_tools(hasher, job: str) -> None:
    for name in TOOLS:
        _tool(hasher, name)
    if job != 'lint-sh':
        return
    if shutil.which('shellcheck'):
        _tool(hasher, 'shellcheck')
        command = ('shellcheck', '--version')
    elif shutil.which('uvx'):
        _tool(hasher, 'uvx')
        command = ('uvx', '--from', 'shellcheck-py', 'shellcheck', '--version')
    else:
        raise ValueError('ShellCheck is unavailable')
    _field(hasher, b'shellcheck', subprocess.check_output(command))


def fingerprint(root: Path, job: str) -> str:
    if job not in JOBS:
        raise ValueError(f'unknown fast-check job: {job}')
    root = root.resolve(strict=True)
    hasher = hashlib.sha256()
    _field(hasher, b'tatbot.fast-check-closure/1', job.encode())
    _tracked_inputs(hasher, root)
    _generated_body(hasher, root)
    if job == 'export':
        # The candidate inventory records the source commit even if its file
        # bytes are identical across revisions.
        _field(hasher, b'head', _git_bytes(root, 'rev-parse', 'HEAD').strip())
    _environment(hasher)
    _checker_tools(hasher, job)
    return hasher.hexdigest()


if __name__ == '__main__':
    if len(sys.argv) != 2:
        raise SystemExit('usage: check_fast_fingerprint.py JOB')
    try:
        print(fingerprint(Path.cwd(), sys.argv[1]))
    except (OSError, ValueError, subprocess.CalledProcessError):
        # Ineligibility is not a PASS. The caller runs the actual job.
        raise SystemExit(1) from None
