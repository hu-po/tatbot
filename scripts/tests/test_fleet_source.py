"""Archive verification is hardware-free and must detect changed tracked source."""
import io
import json
import subprocess
import tarfile

import pytest
from fleet_source import archive_manifest, source_identity, verify_archive  # noqa: E402


@pytest.fixture
def release(tmp_path):
    archive = tmp_path / 'source.tar'
    with tarfile.open(archive, 'w') as stream:
        for name, data, mode in [('scripts/start.sh', b'echo measured', 0o755), ('config.json', b'{}', 0o644)]:
            member = tarfile.TarInfo(name)
            member.size = len(data)
            member.mode = mode
            stream.addfile(member, io.BytesIO(data))
    source = tmp_path / 'release/source'
    source.mkdir(parents=True)
    with tarfile.open(archive) as stream:
        stream.extractall(source, filter='data')
    manifest = archive_manifest(archive, 'a' * 40)
    (source.parent / 'source-manifest.json').write_text(json.dumps(manifest))
    return source


def test_verify_before_build_and_launch_after_build_without_parent_git(release, monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail('archive verification cannot query Git')
    monkeypatch.setattr(subprocess, 'check_output', forbidden)
    assert verify_archive(release, 'a' * 40) == 'a' * 40
    with pytest.raises(FileNotFoundError):
        source_identity(release, require_clean=True)
    (release.parent / 'build.sha').write_text('a' * 40)
    assert source_identity(release, require_clean=True) == 'a' * 40
    (release / 'rust/target').mkdir(parents=True)
    (release / 'rust/target/generated').write_text('build outputs are allowed')
    assert source_identity(release, require_clean=True) == 'a' * 40


@pytest.mark.parametrize('change', ['bytes', 'mode', 'missing', 'symlink', 'identity'])
def test_modified_tracked_source_or_receipt_refuses(release, change):
    (release.parent / 'build.sha').write_text('a' * 40)
    path = release / 'config.json'
    if change == 'bytes':
        path.write_text('{"changed":true}')
    elif change == 'mode':
        path.chmod(0o755)
    elif change == 'missing':
        path.unlink()
    elif change == 'symlink':
        path.unlink()
        path.symlink_to('/dev/null')
    else:
        (release.parent / 'build.sha').write_text('c' * 40)
    with pytest.raises((ValueError, OSError)):
        source_identity(release, require_clean=True)


def test_git_checkout_still_refuses_tracked_changes(tmp_path):
    subprocess.run(['git', 'init', '-q', str(tmp_path)], check=True)
    (tmp_path / 'tracked').write_text('original')
    subprocess.run(['git', '-C', str(tmp_path), 'add', 'tracked'], check=True)
    subprocess.run(['git', '-C', str(tmp_path), '-c', 'user.name=test', '-c', 'user.email=test@example.org',
                    'commit', '-qm', 'fixture'], check=True)
    source = source_identity(tmp_path, require_clean=True)
    assert len(source) == 40
    (tmp_path / 'tracked').write_text('changed')
    with pytest.raises(ValueError, match='clean source checkout'):
        source_identity(tmp_path, require_clean=True)
