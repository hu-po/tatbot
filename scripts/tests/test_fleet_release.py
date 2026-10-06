"""The build receipt binds a checkout to what a deploy built; verify never builds."""
from __future__ import annotations

import hashlib
import json
import shutil
import subprocess

import fleet_release as release  # noqa: E402
import pytest

SHA = 'a' * 40


def make_release(root, *, arm=True):
    stage = root / 'releases' / SHA
    binaries = {'fleetctl': b'#!/bin/sh\nexit 2\n'}
    if arm:
        binaries['path_plan_check'] = b'#!/bin/sh\nexit 3\n'
    paths = {stage / release.LAUNCH_BINARIES[name]: body for name, body in binaries.items()}
    if arm:
        # A generated environment tree the arm build may receipt.
        paths[stage / 'source/python/tatbot_sim/.venv/bin/python'] = b'#!/bin/sh\nexit 4\n'
    for path, body in paths.items():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(body)
        path.chmod(0o755)
    return stage


def git(repo, *args):
    return subprocess.check_output(['git', '-C', str(repo), *args], text=True).strip()


def make_checkout(root):
    repo = root / 'checkout'
    repo.mkdir()
    subprocess.run(['git', 'init', '-q', str(repo)], check=True)
    (repo / 'tracked').write_text('original')
    subprocess.run(['git', '-C', str(repo), 'add', 'tracked'], check=True)
    subprocess.run(['git', '-C', str(repo), '-c', 'user.name=t', '-c', 'user.email=t@example.org',
                    'commit', '-qm', 'fixture'], check=True)
    return repo


def test_build_names_the_features_and_every_launch_binary_digest(tmp_path):
    stage = make_release(tmp_path)
    receipt = release.build(stage, SHA, 'arm-node', ['trossen', 'rerun'],
                            ['fleetctl', 'path_plan_check'])
    assert receipt['schema'] == 'tatbot.receipt/1' and receipt['kind'] == 'build'
    assert receipt['source_sha'] == SHA and receipt['release'] == str(stage)
    assert receipt['features'] == ['rerun', 'trossen']
    assert set(receipt['binaries']) == set(release.LAUNCH_BINARIES)
    for name, record in receipt['binaries'].items():
        path = stage / release.LAUNCH_BINARIES[name]
        assert record['path'] == str(path)
        assert record['sha256'] == hashlib.sha256(path.read_bytes()).hexdigest()
    # A named binary the build did not produce fails the build, not a launch.
    with pytest.raises(ValueError, match='launch binary missing'):
        release.build(make_release(tmp_path / 'viewer', arm=False), SHA, 'viewer', [],
                      ['fleetctl', 'path_plan_check'])
    with pytest.raises(ValueError, match='not a launch binary'):
        release.build(stage, SHA, 'arm-node', [], ['wxai_teleop'])
    with pytest.raises(ValueError, match='release root must be absolute'):
        release.build(stage.relative_to(tmp_path), SHA, 'arm-node', [], ['fleetctl'])


def test_release_root_in_receipt_is_absolute(tmp_path):
    stage = make_release(tmp_path)
    receipt = release.build(stage, SHA, 'arm-node', [], ['fleetctl'])
    path = stage / release.RECEIPT
    receipt['release'] = 'relative/release'
    path.write_text(json.dumps(receipt))
    with pytest.raises(ValueError, match='invalid build receipt'):
        release.load(path)


def test_checkout_switches_to_a_relocated_build_only_with_its_new_receipt(tmp_path):
    checkout = make_checkout(tmp_path)
    source_sha = git(checkout, 'rev-parse', 'HEAD')
    old = make_release(tmp_path / 'old')
    moved = make_release(tmp_path / 'moved')
    old_receipt = release.build(old, source_sha, 'arm-node', [], list(release.LAUNCH_BINARIES))
    new_receipt = release.build(moved, source_sha, 'arm-node', [], list(release.LAUNCH_BINARIES))
    stamp = checkout / release.STAMP
    stamp.write_text(json.dumps(old_receipt))
    assert release.verify(checkout)['release'] == str(old)
    stamp.write_text(json.dumps(new_receipt))
    assert release.verify(checkout)['release'] == str(moved)
    stamp.write_text(json.dumps(old_receipt))
    (old / 'bin/fleetctl').unlink()
    with pytest.raises(ValueError, match='launch binary missing'):
        release.verify(checkout)


def test_verify_binds_each_launch_path_to_the_receipt_release(tmp_path):
    stage = make_release(tmp_path)
    checkout = make_checkout(tmp_path)
    receipt = release.build(stage, git(checkout, 'rev-parse', 'HEAD'), 'arm-node', [],
                            list(release.LAUNCH_BINARIES))
    receipt['binaries']['path_plan_check'] = dict(receipt['binaries']['fleetctl'])
    (checkout / release.STAMP).write_text(json.dumps(receipt))
    with pytest.raises(ValueError, match='launch binary path differs: path_plan_check'):
        release.verify(checkout)


def test_build_and_verify_refuse_a_launch_parent_symlink_outside_release(tmp_path):
    stage = make_release(tmp_path)
    checkout = make_checkout(tmp_path)
    receipt = release.build(stage, git(checkout, 'rev-parse', 'HEAD'), 'arm-node', [],
                            ['fleetctl'])
    (checkout / release.STAMP).write_text(json.dumps(receipt))
    outside = tmp_path / 'outside-bin'
    shutil.move(str(stage / 'bin'), outside)
    (stage / 'bin').symlink_to(outside, target_is_directory=True)
    with pytest.raises(ValueError, match='component path escapes release'):
        release.build(stage, SHA, 'arm-node', [], ['fleetctl'])
    with pytest.raises(ValueError, match='component path escapes release'):
        release.verify(checkout)


def test_complete_staged_service_build_is_reusable_only_while_every_binding_matches(tmp_path):
    stage = make_release(tmp_path, arm=False)
    component = stage / 'bin/tatbot-visiond'
    component.write_bytes(b'#!/bin/sh\nexit 5\n')
    component.chmod(0o755)
    source = stage / 'source'
    source.mkdir()
    (source / 'tracked').write_text('same archived source\n')
    manifest = {'schema': 'tatbot.receipt/1', 'kind': 'source-manifest',
                'source_sha': SHA, 'archive_sha256': 'f' * 64,
                'files': {'tracked': {'kind': 'file', 'executable': False,
                                      'sha256': hashlib.sha256((source / 'tracked').read_bytes()).hexdigest()}}}
    manifest_bytes = json.dumps(manifest).encode()
    (stage / 'source-manifest.json').write_bytes(manifest_bytes)
    (stage / 'build.sha').write_text(SHA)
    (stage / release.INCOMPLETE).touch()
    release.main(['build', '--stage', str(stage), '--sha', SHA, '--node', 'camera-node',
                  '--binary', 'fleetctl', '--component', 'bin/tatbot-visiond'])
    assert not (stage / release.INCOMPLETE).exists()
    manifest_digest = hashlib.sha256(manifest_bytes).hexdigest()
    def expected():
        return release.reusable(stage, SHA, 'camera-node', [],
                                ['fleetctl'],
                                ['bin/tatbot-visiond'], manifest_digest)
    assert expected()['components']['bin/tatbot-visiond'] == hashlib.sha256(component.read_bytes()).hexdigest()
    with pytest.raises(ValueError, match='component set differs'):
        release.reusable(stage, SHA, 'camera-node', [], ['fleetctl'], [], manifest_digest)
    (stage / release.INCOMPLETE).touch()
    with pytest.raises(ValueError, match='did not finish'):
        expected()
    with pytest.raises(ValueError, match='did not finish'):
        release.verify(source)
    (stage / release.INCOMPLETE).unlink()
    component.write_bytes(b'changed service executable')
    with pytest.raises(ValueError, match='component changed'):
        expected()
    with pytest.raises(ValueError, match='component changed'):
        release.verify(source)
    component.write_bytes(b'#!/bin/sh\nexit 5\n')
    (stage / 'bin/fleetctl').write_bytes(b'changed launch executable')
    with pytest.raises(ValueError, match='launch binary changed'):
        expected()
    (stage / 'bin/fleetctl').write_bytes(b'#!/bin/sh\nexit 2\n')
    (source / 'tracked').write_text('changed source')
    with pytest.raises(ValueError, match='release source changed'):
        expected()
    (source / 'tracked').write_text('same archived source\n')
    (stage / 'source-manifest.json').write_bytes(manifest_bytes + b' ')
    with pytest.raises(ValueError, match='source manifest differs'):
        expected()
    (stage / 'source-manifest.json').write_bytes(manifest_bytes)
    outside = tmp_path / 'other-fleetctl'
    outside.write_bytes((stage / 'bin/fleetctl').read_bytes())
    receipt_path = stage / release.RECEIPT
    receipt = json.loads(receipt_path.read_text())
    receipt['binaries']['fleetctl']['path'] = str(outside)
    receipt_path.write_text(json.dumps(receipt))
    with pytest.raises(ValueError, match='launch binary path differs'):
        expected()


def test_arm_reuse_requires_exact_generated_trees_and_base_python(tmp_path):
    stage = make_release(tmp_path)
    source = stage / 'source'
    (source / 'tracked').write_text('source')
    venv = source / 'python/tatbot_sim/.venv'
    (venv / 'lib').mkdir()
    (venv / 'lib/package.py').write_text('module = 1\n')
    interpreter = venv / 'bin/python'
    interpreter.unlink()
    toolchain = tmp_path / 'uv/python/cpython-test'
    (toolchain / 'bin').mkdir(parents=True)
    (toolchain / 'lib').mkdir()
    (toolchain / 'bin/python3.11').write_bytes(b'base interpreter')
    (toolchain / 'bin/python3.11').chmod(0o755)
    (toolchain / 'lib/os.py').write_text('stdlib = 1\n')
    interpreter.symlink_to(toolchain / 'bin/python3.11')
    manifest = {'schema': 'tatbot.receipt/1', 'kind': 'source-manifest',
                'source_sha': SHA, 'archive_sha256': 'f' * 64,
                'files': {'tracked': {'kind': 'file', 'executable': False,
                                      'sha256': hashlib.sha256(b'source').hexdigest()}}}
    manifest_bytes = json.dumps(manifest).encode()
    (stage / 'source-manifest.json').write_bytes(manifest_bytes)
    (stage / 'build.sha').write_text(SHA)
    trees = ['source/cpp/teleop/build', 'source/python/tatbot_sim/.venv']
    binaries = list(release.LAUNCH_BINARIES)
    receipt = release.build(stage, SHA, 'arm-node', ['rerun', 'trossen'], binaries, trees=trees)
    (stage / release.RECEIPT).write_text(json.dumps(receipt))
    manifest_digest = hashlib.sha256(manifest_bytes).hexdigest()

    def reused():
        return release.reusable(stage, SHA, 'arm-node', ['rerun', 'trossen'], binaries,
                                [], manifest_digest, trees)

    assert reused() == receipt
    assert set(receipt['generated_trees']) == set(trees)
    assert receipt['python_toolchains']['source/python/tatbot_sim/.venv']['path'] == str(toolchain)
    with pytest.raises(ValueError, match='component set differs'):
        release.reusable(stage, SHA, 'arm-node', ['rerun', 'trossen'], binaries,
                         [], manifest_digest, trees[:1])
    (venv / 'lib/package.py').write_text('module = 2\n')
    with pytest.raises(ValueError, match='generated runtime tree'):
        reused()
    (venv / 'lib/package.py').write_text('module = 1\n')
    (venv / 'lib/extra.py').write_text('unreceipted')
    with pytest.raises(ValueError, match='generated runtime tree'):
        reused()
    (venv / 'lib/extra.py').unlink()
    (toolchain / 'lib/os.py').write_text('stdlib = 2\n')
    with pytest.raises(ValueError, match='Python toolchain changed'):
        reused()
    (toolchain / 'lib/os.py').write_text('stdlib = 1\n')
    assert reused() == receipt
    (venv / 'lib').chmod(0o700)
    with pytest.raises(ValueError, match='generated runtime tree'):
        reused()
    (venv / 'lib').chmod(0o755)
    venv.chmod(0o700)
    with pytest.raises(ValueError, match='generated runtime tree'):
        reused()
    venv.chmod(0o755)


def test_unbounded_system_python_is_built_but_never_reused(tmp_path):
    stage = make_release(tmp_path)
    venv = stage / 'source/python/tatbot_sim/.venv'
    interpreter = venv / 'bin/python'
    interpreter.unlink()
    system = tmp_path / 'system-python'
    (system / 'bin').mkdir(parents=True)
    (system / 'lib').mkdir()
    (system / 'bin/python3').write_bytes(b'system interpreter')
    (system / 'lib/os.py').write_bytes(b'system standard library')
    interpreter.symlink_to(system / 'bin/python3')
    trees = ['source/python/tatbot_sim/.venv']
    receipt = release.build(stage, SHA, 'arm-node', [], ['fleetctl'], trees=trees)
    assert receipt['generated_trees'] and receipt['python_toolchains'] == {}
    (stage / release.RECEIPT).write_text(json.dumps(receipt))
    source = stage / 'source'
    (source / 'tracked').write_text('source')
    manifest = {'schema': 'tatbot.receipt/1', 'kind': 'source-manifest',
                'source_sha': SHA, 'archive_sha256': 'f' * 64,
                'files': {'tracked': {'kind': 'file', 'executable': False,
                                      'sha256': hashlib.sha256(b'source').hexdigest()}}}
    manifest_bytes = json.dumps(manifest).encode()
    (stage / 'source-manifest.json').write_bytes(manifest_bytes)
    (stage / 'build.sha').write_text(SHA)
    with pytest.raises(ValueError, match='no bounded Python toolchain receipt'):
        release.reusable(stage, SHA, 'arm-node', [], ['fleetctl'], [],
                         hashlib.sha256(manifest_bytes).hexdigest(), trees)


def test_component_receipt_rejects_path_escape_and_missing_artifact(tmp_path):
    stage = make_release(tmp_path, arm=False)
    with pytest.raises(ValueError, match='component missing'):
        release.build(stage, SHA, 'camera-node', [], ['fleetctl'], ['bin/absent'])
    with pytest.raises(ValueError, match='invalid component path'):
        release.build(stage, SHA, 'camera-node', [], ['fleetctl'], ['../outside'])
    escaped = stage / 'escaped'
    escaped.symlink_to(tmp_path, target_is_directory=True)
    with pytest.raises(ValueError, match='component path escapes release'):
        release.build(stage, SHA, 'camera-node', [], ['fleetctl'], ['escaped/outside'])


def test_component_receipt_hashes_a_venv_interpreter_leaf_symlink(tmp_path):
    stage = make_release(tmp_path, arm=False)
    external_python = tmp_path / 'python-runtime'
    external_python.write_bytes(b'python v1')
    external_python.chmod(0o755)
    venv = stage / 'source/python/lerobot_robot_tatbot/.venv/bin'
    venv.mkdir(parents=True)
    (venv / 'python').symlink_to(external_python)
    relative = 'source/python/lerobot_robot_tatbot/.venv/bin/python'
    receipt = release.build(stage, SHA, 'arm-node', [], ['fleetctl'], [relative])
    assert receipt['components'][relative] == hashlib.sha256(b'python v1').hexdigest()
    (stage / release.RECEIPT).write_text(json.dumps(receipt))
    source = stage / 'source'
    (source / 'tracked').write_text('source')
    manifest = {'schema': 'tatbot.receipt/1', 'kind': 'source-manifest', 'source_sha': SHA,
                'archive_sha256': 'f' * 64,
                'files': {'tracked': {'kind': 'file', 'executable': False,
                                      'sha256': hashlib.sha256(b'source').hexdigest()}}}
    manifest_bytes = json.dumps(manifest).encode()
    (stage / 'source-manifest.json').write_bytes(manifest_bytes)
    (stage / 'build.sha').write_text(SHA)
    def reusable():
        return release.reusable(stage, SHA, 'arm-node', [], ['fleetctl'], [relative],
                                hashlib.sha256(manifest_bytes).hexdigest())
    assert reusable() == receipt
    external_python.write_bytes(b'python v2')
    with pytest.raises(ValueError, match='component changed'):
        reusable()


def test_the_cli_writes_the_receipt_beside_the_stage_and_verify_reads_it_back(tmp_path, capsys):
    stage = make_release(tmp_path)
    release.main(['build', '--stage', str(stage), '--sha', SHA, '--node', 'arm-node',
                  '--features', 'trossen,rerun', '--binary', 'fleetctl',
                  '--binary', 'path_plan_check'])
    written = json.loads((stage / 'build.json').read_text())
    assert written['features'] == ['rerun', 'trossen']
    assert capsys.readouterr().out.strip() == str(stage / 'build.json')
    repo = make_checkout(tmp_path)
    written['source_sha'] = git(repo, 'rev-parse', 'HEAD')
    (repo / '.tatbot-build.json').write_text(json.dumps(written))
    release.main(['verify', '--checkout', str(repo), '--binary', 'fleetctl'])
    assert capsys.readouterr().out.strip() == str(stage / 'bin/fleetctl')
    release.main(['verify', '--checkout', str(repo)])
    assert json.loads(capsys.readouterr().out)['release'] == str(stage)
    # A caller that needs a build feature names it; a receipt without it refuses.
    release.main(['verify', '--checkout', str(repo), '--binary', 'fleetctl',
                  '--feature', 'trossen', '--feature', 'rerun'])
    assert capsys.readouterr().out.strip() == str(stage / 'bin/fleetctl')
    with pytest.raises(SystemExit) as refused:
        release.main(['verify', '--checkout', str(repo), '--binary', 'fleetctl', '--feature', 'realsense'])
    assert refused.value.code == 3
    assert 'built without realsense' in capsys.readouterr().err


@pytest.mark.parametrize('moved', ['head', 'edits', 'binary', 'release', 'unstamped'])
def test_verify_refuses_a_checkout_that_moved_past_its_receipt(tmp_path, moved, capsys):
    stage = make_release(tmp_path)
    repo = make_checkout(tmp_path)
    receipt = release.build(stage, git(repo, 'rev-parse', 'HEAD'), 'arm-node', ['trossen', 'rerun'],
                            list(release.LAUNCH_BINARIES))
    (repo / '.tatbot-build.json').write_text(json.dumps(receipt))
    assert release.verify(repo) == receipt
    if moved == 'head':
        (repo / 'tracked').write_text('later')
        subprocess.run(['git', '-C', str(repo), '-c', 'user.name=t', '-c', 'user.email=t@example.org',
                        'commit', '-qam', 'later'], check=True)
        expected = 'checkout is at'
    elif moved == 'edits':
        (repo / 'tracked').write_text('edited')
        expected = 'tracked edits'
    elif moved == 'binary':
        (stage / 'bin/fleetctl').write_bytes(b'#!/bin/sh\nexit 9\n')
        expected = 'changed since its build'
    elif moved == 'release':
        (stage / 'bin/fleetctl').unlink()
        expected = 'launch binary missing'
    else:
        (repo / '.tatbot-build.json').unlink()
        expected = 'no build receipt'
    with pytest.raises(ValueError, match=expected):
        release.verify(repo)
    with pytest.raises(SystemExit) as refused:
        release.main(['verify', '--checkout', str(repo), '--binary', 'fleetctl'])
    assert refused.value.code == 3
    assert expected in capsys.readouterr().err
    assert capsys.readouterr().out == ''


def test_verify_never_runs_cmake_or_cargo(tmp_path, monkeypatch):
    stage = make_release(tmp_path)
    repo = make_checkout(tmp_path)
    receipt = release.build(stage, git(repo, 'rev-parse', 'HEAD'), 'arm-node', ['trossen'],
                            list(release.LAUNCH_BINARIES))
    (repo / '.tatbot-build.json').write_text(json.dumps(receipt))
    seen = []
    real = subprocess.Popen

    class Recorded(real):
        def __init__(self, argv, *args, **kwargs):
            seen.append(argv[0])
            super().__init__(argv, *args, **kwargs)

    monkeypatch.setattr(subprocess, 'Popen', Recorded)
    release.verify(repo)
    assert set(seen) == {'git'}
    # A release's own source tree verifies its archive instead of asking git.
    source = stage / 'source'
    (source / 'scripts').mkdir(parents=True, exist_ok=True)
    (source / 'scripts/x').write_text('x')
    (stage / 'build.json').write_text(json.dumps(receipt))
    seen.clear()
    monkeypatch.setattr(release, 'verify_archive', lambda checkout, expected: expected)
    assert release.verify(source) == receipt
    assert seen == []


def test_gitless_viewer_overlay_verifies_only_manifested_source(tmp_path):
    stage = make_release(tmp_path, arm=False)
    source = stage / 'source'
    checkout = tmp_path / 'viewer'
    relative = 'scripts/lib/tatbot_digest.py'
    for root in (source, checkout):
        (root / 'scripts/lib').mkdir(parents=True)
        (root / relative).write_text('same archived source\n')
    manifest = {'schema': 'tatbot.receipt/1', 'kind': 'source-manifest',
                'source_sha': SHA, 'archive_sha256': 'f' * 64,
                'files': {relative: {'kind': 'file', 'executable': False,
                                     'sha256': hashlib.sha256((source / relative).read_bytes()).hexdigest()}}}
    (stage / 'source-manifest.json').write_text(json.dumps(manifest))
    receipt = release.build(stage, SHA, 'viewer', [], ['fleetctl'])
    (checkout / '.tatbot-build.json').write_text(json.dumps(receipt))
    (checkout / '.tatbot-deploy.json').write_text(json.dumps({'source_commit': SHA}))
    (checkout / 'scripts/lib/__pycache__').mkdir()
    (checkout / 'scripts/lib/__pycache__/tatbot_digest.pyc').write_bytes(b'generated here')
    assert release.verify(checkout) == receipt
    (checkout / relative).write_text('local tracked edit\n')
    with pytest.raises(ValueError, match='release source changed'):
        release.verify(checkout)
    (checkout / relative).write_text('same archived source\n')
    (checkout / '.tatbot-deploy.json').write_text(json.dumps({'source_commit': 'b' * 40}))
    with pytest.raises(ValueError, match='viewer overlay source differs'):
        release.verify(checkout)


def test_an_offline_rung_plans_with_the_receipts_planner_on_a_stamped_checkout(tmp_path):
    """The compile's executor preflight runs the receipt's planner, never a
    checkout build left over from an earlier
    constants sha; an unstamped worktree keeps its own build, and a node whose
    receipt names no planner falls back to the checkout's."""
    stage = make_release(tmp_path)
    repo = make_checkout(tmp_path)
    own = repo / 'cpp/teleop/build/path_plan_check'
    assert release.planner_binary(repo) == own
    receipt = release.build(stage, git(repo, 'rev-parse', 'HEAD'), 'arm-node', ['trossen', 'rerun'],
                            list(release.LAUNCH_BINARIES))
    (repo / '.tatbot-build.json').write_text(json.dumps(receipt))
    assert release.planner_binary(repo) == stage / release.LAUNCH_BINARIES['path_plan_check']
    viewer = release.build(make_release(tmp_path / 'viewer', arm=False), git(repo, 'rev-parse', 'HEAD'), 'viewer', [],
                           ['fleetctl'])
    (repo / '.tatbot-build.json').write_text(json.dumps(viewer))
    assert release.planner_binary(repo) == own
    (repo / 'tracked').write_text('edited')
    (repo / '.tatbot-build.json').write_text(json.dumps(receipt))
    with pytest.raises(ValueError, match='tracked edits'):
        release.planner_binary(repo)
