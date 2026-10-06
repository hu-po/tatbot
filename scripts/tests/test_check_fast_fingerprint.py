"""A fast-tier PASS key changes with every input these read-only jobs inspect."""

import os
import subprocess

import pytest
from check_fast_fingerprint import fingerprint


def git(root, *args):
    return subprocess.run(['git', *args], cwd=root, capture_output=True,
                          text=True, check=True).stdout.strip()


def checkout(tmp_path):
    git(tmp_path, 'init', '-q')
    git(tmp_path, 'config', 'user.name', 'Fixture')
    git(tmp_path, 'config', 'user.email', 'fixture@example.com')
    (tmp_path / '.gitignore').write_text('web/inkmap/dist/\n')
    script = tmp_path / 'run.sh'
    script.write_text('#!/bin/sh\nexit 0\n')
    script.chmod(0o755)
    git(tmp_path, 'add', '.gitignore', 'run.sh')
    git(tmp_path, 'commit', '-qm', 'fixture')
    return script


def test_fast_check_key_tracks_working_bytes_modes_membership_and_generated_body(tmp_path):
    script = checkout(tmp_path)
    baseline = fingerprint(tmp_path, 'body-single-path')
    script.write_text('#!/bin/sh\nexit 1\n')
    changed = fingerprint(tmp_path, 'body-single-path')
    assert changed != baseline
    git(tmp_path, 'add', 'run.sh')
    staged = fingerprint(tmp_path, 'body-single-path')
    assert staged != changed
    script.write_text('#!/bin/sh\nexit 0\n')
    assert fingerprint(tmp_path, 'body-single-path') != baseline
    git(tmp_path, 'reset', '-q', 'HEAD', '--', 'run.sh')
    assert fingerprint(tmp_path, 'body-single-path') == baseline
    script.chmod(0o644)
    assert fingerprint(tmp_path, 'body-single-path') != baseline
    script.chmod(0o755)
    extra = tmp_path / 'new.txt'
    extra.write_text('untracked input')
    assert fingerprint(tmp_path, 'body-single-path') != baseline
    extra.unlink()
    dist = tmp_path / 'web/inkmap/dist'
    dist.mkdir(parents=True)
    (dist / 'bundle.js').write_text('generated body')
    generated = fingerprint(tmp_path, 'body-single-path')
    assert generated != baseline
    (dist / 'bundle.js').write_text('changed generated body')
    assert fingerprint(tmp_path, 'body-single-path') != generated
    (dist / 'bundle.js').unlink()
    dist.rmdir()
    assert fingerprint(tmp_path, 'body-single-path') == baseline


def test_fast_check_key_binds_environment_and_export_revision(tmp_path, monkeypatch):
    checkout(tmp_path)
    monkeypatch.delenv('TATBOT_NODE', raising=False)
    cli = fingerprint(tmp_path, 'cli')
    export = fingerprint(tmp_path, 'export')
    monkeypatch.setenv('TATBOT_NODE', 'different-fixture-node')
    assert fingerprint(tmp_path, 'cli') != cli
    monkeypatch.delenv('TATBOT_NODE')
    git(tmp_path, 'commit', '--allow-empty', '-qm', 'new source revision')
    assert fingerprint(tmp_path, 'cli') == cli
    assert fingerprint(tmp_path, 'export') != export


def test_fast_check_does_not_cache_external_symlink_inputs(tmp_path):
    checkout(tmp_path)
    (tmp_path / 'outside').symlink_to(os.devnull)
    with pytest.raises(ValueError, match='non-regular fast-check input'):
        fingerprint(tmp_path, 'cli')
