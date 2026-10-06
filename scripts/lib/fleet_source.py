"""Verify native fleet source identity in a checkout or immutable release."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import stat
import subprocess
import tarfile
from pathlib import Path, PurePosixPath

# This module runs alone on a deploy stage (`fleet_deploy.py` invokes the
# staged copy), so it names its receipt's family and kind here
# instead of importing the schema table; the writer and the reader of a stage's
# manifest are always the same release.
MANIFEST_SCHEMA = 'tatbot.receipt/1'
MANIFEST_KIND = 'source-manifest'


def sha256(data):
    return hashlib.sha256(data).hexdigest()


def archive_manifest(archive: Path, source_sha: str) -> dict:
    if not re.fullmatch(r'[0-9a-f]{40}', source_sha):
        raise ValueError('invalid archive source identity')
    files = {}
    with tarfile.open(archive, 'r:') as stream:
        for member in stream:
            name = PurePosixPath(member.name)
            if name.is_absolute() or '..' in name.parts:
                raise ValueError('archive path escapes source')
            if member.isdir():
                continue
            if member.name in files:
                raise ValueError('duplicate archive path')
            if member.issym():
                files[member.name] = {'kind': 'symlink', 'target': member.linkname}
            elif member.isfile():
                files[member.name] = {'kind': 'file', 'sha256': sha256(stream.extractfile(member).read()),
                                      'executable': bool(member.mode & 0o111)}
            else:
                raise ValueError('unsupported source archive member')
    if not files:
        raise ValueError('empty source archive')
    return {'schema': MANIFEST_SCHEMA, 'kind': MANIFEST_KIND, 'source_sha': source_sha,
            'archive_sha256': sha256(archive.read_bytes()), 'files': files}


def verify_archive(repo: Path, expected: str | None = None) -> str:
    if repo.name != 'source' or (repo / '.git').exists():
        raise ValueError('not a staged source release')
    receipt = json.loads((repo.parent / 'source-manifest.json').read_bytes())
    return verify_manifest_tree(repo, receipt, expected)


def verify_manifest_tree(repo: Path, receipt: dict, expected: str | None = None) -> str:
    """Verify only archived paths, leaving generated and node-local files alone."""
    source = receipt.get('source_sha', '')
    if ((receipt.get('schema'), receipt.get('kind')) != (MANIFEST_SCHEMA, MANIFEST_KIND)
            or not re.fullmatch(r'[0-9a-f]{40}', source)
            or not re.fullmatch(r'[0-9a-f]{64}', receipt.get('archive_sha256', ''))
            or not isinstance(receipt.get('files'), dict) or not receipt['files']):
        raise ValueError('invalid release source manifest')
    if expected is not None and source != expected:
        raise ValueError('release source identity differs')
    for name, record in receipt['files'].items():
        if not isinstance(name, str) or not isinstance(record, dict):
            raise ValueError('invalid release manifest entry')
        relative = PurePosixPath(name)
        if relative.is_absolute() or '..' in relative.parts or not relative.parts:
            raise ValueError('release manifest path escapes source')
        path = repo / name
        # No parent symlink may redirect a tracked file outside the source tree.
        if not path.parent.resolve().is_relative_to(repo.resolve()):
            raise ValueError(f'release source path escapes: {name}')
        mode = path.lstat().st_mode
        if record.get('kind') == 'symlink':
            good = stat.S_ISLNK(mode) and os.readlink(path) == record.get('target')
        else:
            good = (record.get('kind') == 'file' and stat.S_ISREG(mode)
                    and bool(mode & 0o111) == record.get('executable')
                    and sha256(path.read_bytes()) == record.get('sha256'))
        if not good:
            raise ValueError(f'release source changed: {name}')
    return source


def verify_overlay(repo: Path, manifest: Path, expected: str) -> str:
    """Verify a Gitless viewer checkout against its deployed source manifest."""
    if (repo / '.git').exists():
        raise ValueError('viewer overlay is a git checkout')
    return verify_manifest_tree(repo, json.loads(manifest.read_bytes()), expected)


def source_identity(repo: Path, *, require_clean=False, expected=None) -> str:
    if (repo / '.git').exists():
        source = subprocess.check_output(['git', '-C', str(repo), 'rev-parse', 'HEAD'], text=True).strip()
        if require_clean and subprocess.check_output(
                ['git', '-C', str(repo), 'status', '--porcelain', '--untracked-files=no'], text=True).strip():
            raise ValueError('a fleet build requires a clean source checkout')
    else:
        if repo.name != 'source':
            raise ValueError('source has neither checkout metadata nor a release layout')
        identity = repo.parent / 'build.sha'
        if not identity.is_file():
            identity = repo.parent / 'source.sha'
        source = identity.read_text().strip()
        verify_archive(repo, source)
    if not re.fullmatch(r'[0-9a-f]{40}', source) or (expected is not None and source != expected):
        raise ValueError('source identity differs or is invalid')
    return source


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repo', type=Path, required=True)
    parser.add_argument('--require-clean', action='store_true')
    parser.add_argument('--expected')
    parser.add_argument('--verify-archive', action='store_true', help='verify extraction before build.sha exists')
    args = parser.parse_args()
    try:
        result = (verify_archive(args.repo, args.expected) if args.verify_archive else
                  source_identity(args.repo, require_clean=args.require_clean, expected=args.expected))
        print(result)
    except (OSError, ValueError, subprocess.SubprocessError) as error:
        parser.exit(3, f'fleet source refused: {error}\n')


if __name__ == '__main__':
    main()
