"""The unattended camera handoff must not grant general service control."""
import hashlib
import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location('fleet_install', ROOT / 'scripts/lib/fleet_install.py')
installer = importlib.util.module_from_spec(spec)
spec.loader.exec_module(installer)


def test_handoff_rule_contains_only_exact_camera_start_stop():
    rule = installer.d405_handoff_rule('robot-host')
    commands = rule.split('NOPASSWD: ', 1)[1].strip().split(', ')
    assert commands == ['/usr/bin/systemctl start tatbot-visiond-d405.service',
                        '/usr/bin/systemctl stop tatbot-visiond-d405.service']
    assert rule.startswith('robot-host ALL=(root) ')


@pytest.mark.parametrize('user', ['root ALL=(ALL) ALL', 'robot-host\nroot', 'robot-host*', '', 'robot-host,root'])
def test_handoff_rule_rejects_policy_injection(user):
    with pytest.raises(ValueError, match='invalid service account'):
        installer.d405_handoff_rule(user)


@pytest.mark.parametrize('targeted', [False, True])
def test_camera_unit_uses_the_selected_manifest_node(tmp_path, monkeypatch, targeted):
    stage = tmp_path / 'release'
    units = stage / 'source/config/systemd'
    units.mkdir(parents=True)
    unit = 'tatbot-visiond-d405.service'
    (units / unit).write_text((ROOT / 'config/systemd' / unit).read_text())
    (stage / 'source/config/nodes.json').write_text(json.dumps({
        'capture-left': {'ssh': 'robot@192.0.2.10', 'services': [
            {'binary': 'tatbot-visiond', 'unit': unit},
            *([{'binary': 'trackd', 'unit': 'tatbot-trackd.service'}] if targeted else [])]} }))
    (stage / 'bin').mkdir()
    binary = stage / 'bin/tatbot-visiond'
    binary.write_text('#!/bin/sh\nexit 99\n')
    binary.chmod(0o755)
    sha = 'a' * 40
    (stage / 'build.sha').write_text(sha)
    monkeypatch.setattr(Path, 'home', lambda: Path('/home') / 'robot')
    monkeypatch.setattr(sys, 'argv', ['install', '--stage', str(stage), '--node',
                                     'capture-left', '--sha', sha, '--verify-only',
                                     *(['--service', unit] if targeted else [])])
    calls = []
    monkeypatch.setattr(installer, 'run', lambda *argv: calls.append(argv))
    installer.main()
    rendered = (stage / 'units' / unit).read_text()
    assert 'Environment=TATBOT_NODE=capture-left' in rendered
    assert '@NODE@' not in rendered
    assert [call[0] for call in calls] == ['systemd-analyze', '/usr/sbin/visudo']


@pytest.fixture
def installed_arm_release(tmp_path, monkeypatch):
    """Offline installed-release state, including a process in the unit cgroup."""
    stage, repo = tmp_path / 'release', tmp_path / 'checkout'
    sha, node, user = 'a' * 40, 'arm-node', 'tester'
    service = {'unit': 'tatbot-trackd.service', 'binary': 'trackd'}
    template = stage / 'source/config/systemd' / service['unit']
    template.parent.mkdir(parents=True)
    template.write_text('ExecStart=@RELEASE@/source/scripts/fleet_service.sh trackd\n')
    staged_unit = stage / 'units' / service['unit']
    staged_unit.parent.mkdir()
    staged_unit.write_text(installer._unit_text(stage, stage / 'source', service['unit'],
                                               user, tmp_path, node))
    unit_root = tmp_path / 'systemd'
    unit_root.mkdir()
    (unit_root / service['unit']).write_bytes(staged_unit.read_bytes())
    monkeypatch.setattr(installer, 'SYSTEMD_UNIT_ROOT', unit_root)
    binaries = {}
    for name in ('fleetctl', 'trackd'):
        binary = stage / 'bin' / name
        binary.parent.mkdir(exist_ok=True)
        binary.write_bytes(f'#!/bin/sh\n# {name}\n'.encode())
        binary.chmod(0o755)
        copied = repo / 'rust/target/release' / name
        copied.parent.mkdir(parents=True, exist_ok=True)
        copied.write_bytes(binary.read_bytes())
        copied.chmod(0o755)
        binaries[name] = installer.fleet_release.sha256_file(binary)
    receipt = {'source_sha': sha, 'node': node, 'release': str(stage),
               'binaries': {name: {'path': str(stage / 'bin' / name), 'sha256': binaries[name]}
                            for name in ('fleetctl',)},
               'components': {'bin/trackd': binaries['trackd']}}
    monkeypatch.setattr(installer.fleet_release, 'verify', lambda path: receipt)
    (stage / 'installed.json').write_text(json.dumps({
        'schema': 'tatbot.receipt/1', 'kind': 'deployment', 'sha': sha, 'node': node,
        'units': [service['unit']], 'mode': 'services'}))
    project = repo / 'web/inkmap'
    project.mkdir(parents=True)
    (project / 'package-lock.json').write_text('pinned\n')
    (project / 'node_modules/.tatbot-artwork-runtime.sha256').parent.mkdir()
    (project / 'node_modules/.tatbot-artwork-runtime.sha256').write_text(
        installer.fleet_release.sha256_file(project / 'package-lock.json') + '\n')
    for name in installer.artwork_runtime.MODULES:
        package = project / 'node_modules' / name / 'package.json'
        package.parent.mkdir(parents=True)
        package.write_text('{}')
    installed_receipt = stage / 'installed.json'
    installed = json.loads(installed_receipt.read_text())
    installed['artwork_runtime_sha256'] = installer.fleet_release.tree_contents(
        project / 'node_modules')['sha256']
    installed_receipt.write_text(json.dumps(installed))
    launcher = repo / 'scripts/tatbot'
    launcher.parent.mkdir(parents=True)
    launcher.write_text('#!/bin/sh\n')
    launcher.chmod(0o755)
    link = tmp_path / 'bin/tatbot'
    link.parent.mkdir()
    link.symlink_to(launcher)
    monkeypatch.setattr(installer.tatbot_cli_install, 'DESTINATION', link)
    cgroup_root = tmp_path / 'cgroup'
    group = cgroup_root / 'system.slice' / service['unit']
    group.mkdir(parents=True)
    (group / 'cgroup.procs').write_text('123\n124\n')
    monkeypatch.setattr(installer, 'CGROUP_ROOT', cgroup_root)
    proc_root = tmp_path / 'proc'
    for pid, target in ((123, Path('/usr/bin/sh')), (124, stage / 'bin/trackd')):
        process = proc_root / str(pid)
        process.mkdir(parents=True)
        (process / 'exe').symlink_to(target)
    monkeypatch.setattr(installer, 'PROC_ROOT', proc_root)
    state = {'ActiveState': 'active', 'NeedDaemonReload': 'no'}

    def run(argv, **kwargs):
        if argv[0] != 'systemctl':
            return subprocess.CompletedProcess(argv, 0, stdout='tatbot 1\n', stderr='')
        fields = {'Id': service['unit'], 'SubState': 'running', 'MainPID': '123',
                  'ExecMainStartTimestampMonotonic': '400', 'InvocationID': 'instance',
                  'FragmentPath': str(unit_root / service['unit']), 'UnitFileState': 'enabled',
                  'ControlGroup': '/system.slice/' + service['unit'],
                  'ExecStart': f'{{ path={stage}/source/scripts/fleet_service.sh ; argv[]=... }}',
                  **state}
        return subprocess.CompletedProcess(argv, 0,
                                           stdout=''.join(f'{key}={value}\n' for key, value in fields.items()),
                                           stderr='')

    monkeypatch.setattr(installer.subprocess, 'run', run)
    return stage, repo, sha, node, service, user, state, proc_root, unit_root


def test_unchanged_installed_arm_release_proves_no_restart(installed_arm_release):
    stage, repo, sha, node, service, user, *_ = installed_arm_release
    installer.verify_installed(stage, repo, sha, node, [service], user, stage.parent)


@pytest.mark.parametrize('change', ['checkout-binary', 'unit', 'staged-unit', 'inactive', 'daemon-reload',
                                    'running-binary', 'cli', 'artwork', 'artwork-content', 'receipt'])
def test_installed_arm_release_refuses_incomplete_or_changed_state(installed_arm_release, change):
    stage, repo, sha, node, service, user, state, proc_root, unit_root = installed_arm_release
    if change == 'checkout-binary':
        (repo / 'rust/target/release/fleetctl').write_bytes(b'changed')
    elif change == 'unit':
        (unit_root / service['unit']).write_text('changed')
    elif change == 'staged-unit':
        (stage / 'units' / service['unit']).write_text('changed')
    elif change == 'inactive':
        state['ActiveState'] = 'inactive'
    elif change == 'daemon-reload':
        state['NeedDaemonReload'] = 'yes'
    elif change == 'running-binary':
        (proc_root / '124/exe').unlink()
        (proc_root / '124/exe').symlink_to(repo / 'rust/target/release/trackd')
    elif change == 'cli':
        installer.tatbot_cli_install.DESTINATION.unlink()
    elif change == 'artwork':
        (repo / 'web/inkmap/node_modules/.tatbot-artwork-runtime.sha256').unlink()
    elif change == 'artwork-content':
        (repo / 'web/inkmap/node_modules/ajv/package.json').write_text('{"changed":true}')
    else:
        (stage / 'installed.json').write_text('{}')
    with pytest.raises((OSError, ValueError), match='differs|unavailable|receipt|executable|CLI|artwork|release'):
        installer.verify_installed(stage, repo, sha, node, [service], user, stage.parent)


def test_verify_installed_cli_does_not_render_or_install(tmp_path, monkeypatch):
    stage = tmp_path / 'release'
    node = {'roles': ['arm'], 'checkout': str(tmp_path / 'checkout')}
    observed = []
    monkeypatch.setattr(sys, 'argv', ['install', '--stage', str(stage), '--node', 'arm-node',
                                     '--sha', 'a' * 40, '--verify-installed'])
    monkeypatch.setattr(installer, '_install_context',
                        lambda *args: (stage, stage / 'source', node, [], 'tester', tmp_path))
    monkeypatch.setattr(installer, '_render_units', lambda *args: pytest.fail('rendered staged unit'))
    monkeypatch.setattr(installer, '_install_release', lambda *args: pytest.fail('installed service'))
    monkeypatch.setattr(installer, 'verify_installed', lambda *args: observed.append(args))
    installer.main()
    assert observed == [(stage, tmp_path / 'checkout', 'a' * 40, 'arm-node', [], 'tester', tmp_path)]


# --------------------------------------------------------------------------
# Release retention. Nothing pruned superseded trees: 31 had accumulated on
# one node at ~150 MB each, and the deploy that needed the space was the one
# that could not get it.
# --------------------------------------------------------------------------

def make_releases(root, names):
    for index, name in enumerate(names):
        tree = root / name
        tree.mkdir(parents=True)
        (tree / 'build.sha').write_text(name)
        os.utime(tree, (1_700_000_000 + index, 1_700_000_000 + index))
    return root


SHAS = [f'{n:040x}' for n in range(6)]


def test_prune_keeps_the_installed_sha_the_recent_and_nothing_else(tmp_path):
    releases = make_releases(tmp_path / 'releases', SHAS)
    removed = installer.prune_releases(releases, keep_sha=SHAS[0], referenced=set(), keep=2)
    surviving = sorted(p.name for p in releases.iterdir())
    # SHAS[0] is the oldest by mtime, so it survives only because it is the
    # sha just installed -- the case a plain "newest N" rule would delete.
    assert surviving == sorted([SHAS[0], SHAS[4], SHAS[5]])
    assert set(removed) == {SHAS[1], SHAS[2], SHAS[3]}


def test_prune_never_removes_a_tree_an_installed_unit_still_points_at(tmp_path):
    releases = make_releases(tmp_path / 'releases', SHAS)
    removed = installer.prune_releases(releases, keep_sha=SHAS[5],
                                       referenced={SHAS[1]}, keep=1)
    assert SHAS[1] not in removed
    assert (releases / SHAS[1]).is_dir()


def test_prune_only_considers_release_shaped_names(tmp_path):
    releases = make_releases(tmp_path / 'releases', SHAS[:2])
    for stray in ('scratch', 'previous-source', SHAS[0][:39], SHAS[0] + 'f'):
        (releases / stray).mkdir()
    installer.prune_releases(releases, keep_sha=SHAS[1], referenced=set(), keep=1)
    for stray in ('scratch', 'previous-source', SHAS[0][:39], SHAS[0] + 'f'):
        assert (releases / stray).is_dir(), f'{stray} is not a release tree'


def test_referenced_releases_reads_the_shas_units_point_at(tmp_path):
    units = tmp_path / 'systemd'
    units.mkdir()
    # No literal home path here: the export scanner reads these as a
    # disclosure, and only the /releases/<sha> segment is under test.
    (units / 'tatbot-visiond-poe.service').write_text(
        f'ExecStart={tmp_path}/releases/{SHAS[3]}/source/scripts/fleet_service.sh visiond-poe\n')
    (units / 'unrelated.service').write_text(
        f'ExecStart={tmp_path}/releases/{SHAS[4]}/bin/other\n')
    assert installer.referenced_releases(units) == {SHAS[3]}


# --------------------------------------------------------------------------
# The build receipt. A full deploy brings the command path (the checkout
# config/nodes.json names) to the built revision and stamps it with the
# release's receipt; a launcher verifies against that stamp instead of
# building. A --service deploy carries no launch binary and leaves both alone.
# --------------------------------------------------------------------------

def make_receipt(stage, sha):
    receipt = {'schema': 'tatbot.receipt/1', 'kind': 'build', 'source_sha': sha, 'node': 'arm-node',
               'release': str(stage), 'features': [],
               'binaries': {'fleetctl': {'path': str(stage / 'bin/fleetctl'), 'sha256': 'f' * 64}}}
    stage.mkdir(parents=True, exist_ok=True)
    (stage / 'build.json').write_text(json.dumps(receipt, indent=2) + '\n')
    return receipt


def git_checkout(monkeypatch, root, sha):
    repo = root / 'checkout'
    (repo / '.git').mkdir(parents=True)
    monkeypatch.setattr(installer.subprocess, 'check_output',
                        lambda argv, **kwargs: sha + '\n' if argv[3] == 'rev-parse' else '')
    return repo


def test_a_full_install_brings_the_checkout_to_the_release_and_stamps_its_receipt(tmp_path, monkeypatch):
    sha = SHAS[2]
    stage = tmp_path / 'releases' / sha
    receipt = make_receipt(stage, sha)
    repo = git_checkout(monkeypatch, tmp_path, sha)
    calls = []
    monkeypatch.setattr(installer, 'run', lambda *argv: calls.append(argv))
    parser = installer.argparse.ArgumentParser()
    installer.update_checkout(parser, stage, repo, sha, 'arm-node', [])
    assert ('git', '-C', str(repo), 'merge', '--ff-only', sha) in calls
    assert json.loads((repo / '.tatbot-build.json').read_text()) == receipt
    assert not (repo / '.tatbot-build.json.fleet-deploy').exists()
    # The stamp is ignored like the viewer's overlay receipt: a full deploy
    # leaves the checkout clean, so the mirror sweep neither stops
    # fast-forwarding it nor rescues the stamp into a worktree.
    ignored = subprocess.run(['git', '-C', str(ROOT), 'check-ignore', '-q', '--no-index', '.tatbot-build.json'])
    assert ignored.returncode == 0
    # Without a receipt, or with one naming another source, the install is
    # refused before the merge: the checkout neither moves nor loses the
    # stamp a launcher already trusts.
    merged = len(calls)
    other = tmp_path / 'releases' / SHAS[3]
    make_receipt(other, SHAS[3])
    with pytest.raises(SystemExit) as refused:
        installer.update_checkout(parser, other, repo, sha, 'arm-node', [])
    assert refused.value.code == 3
    (other / 'build.json').unlink()
    with pytest.raises(SystemExit) as refused:
        installer.update_checkout(parser, other, repo, sha, 'arm-node', [])
    assert refused.value.code == 3
    assert len(calls) == merged
    assert json.loads((repo / '.tatbot-build.json').read_text()) == receipt


def test_a_full_install_prepares_the_arm_checkouts_artwork_runtime(tmp_path, monkeypatch):
    """The design stage of `session new` runs the shared SVG adapter with the
    checkout's `web/inkmap/node_modules`, gitignored and so empty after the
    ff-merge that brings the checkout to the built revision: a fresh deploy's
    first `session new` failed `artwork_runtime_unavailable` until an operator
    ran npm ci by hand. The install prepares it (`artwork_runtime.prepare`, once
    per lock digest) with npm resolved as the build resolved it — the release's
    private runtime first — so the first session pays no install; a camera
    node or a `--service` deploy leaves the checkout's runtime alone."""
    import artwork_runtime
    sha = SHAS[2]
    stage = tmp_path / 'releases' / sha
    make_receipt(stage, sha)
    (stage / 'source/config/systemd').mkdir(parents=True)
    (stage / 'bin').mkdir()
    (stage / 'build.sha').write_text(sha)
    checkout = tmp_path / 'checkout'
    (checkout / '.git').mkdir(parents=True)
    (checkout / artwork_runtime.PROJECT).mkdir(parents=True)
    (checkout / artwork_runtime.PROJECT / artwork_runtime.LOCK_FILE).write_text('{"lockfileVersion": 3}')
    nodes = {'arm-node': {'ssh': 'robot@192.0.2.12', 'roles': ['arm'], 'checkout': str(checkout), 'services': []},
             'capture-left': {'ssh': 'robot@192.0.2.10', 'roles': ['capture'], 'checkout': str(checkout), 'services': []}}
    (stage / 'source/config/nodes.json').write_text(json.dumps(nodes))
    private = stage / 'toolchain/node/bin'
    private.mkdir(parents=True)
    (private / 'npm').write_text('#!/bin/sh\nexit 97\n')
    (private / 'npm').chmod(0o755)
    monkeypatch.setattr(Path, 'home', lambda: Path('/home') / 'robot')
    monkeypatch.setenv('XDG_CACHE_HOME', str(tmp_path / 'cache'))
    monkeypatch.setattr(installer.subprocess, 'check_output',
                        lambda argv, **kwargs: sha + '\n' if argv[3] == 'rev-parse' else '')
    calls = []
    monkeypatch.setattr(installer, 'run', lambda *argv: calls.append(argv))
    npm = []

    def npm_ci(argv, cwd=None, **kwargs):
        npm.append((argv, Path(cwd)))
        for name in artwork_runtime.MODULES:
            (Path(cwd) / 'node_modules' / name).mkdir(parents=True)
            (Path(cwd) / 'node_modules' / name / 'package.json').write_text('{}')
        return subprocess.CompletedProcess(argv, 0)
    monkeypatch.setattr(subprocess, 'run', npm_ci)

    def install(node, *extra):
        calls.clear()
        monkeypatch.setattr(sys, 'argv', ['install', '--stage', str(stage), '--node', node, '--sha', sha, *extra])
        installer.main()
    install('arm-node')
    [(argv, cwd)] = npm
    assert argv == [str(private / 'npm'), 'ci', '--ignore-scripts', '--omit=dev', '--no-audit', '--no-fund']
    assert cwd == checkout / artwork_runtime.PROJECT
    merged = next(i for i, call in enumerate(calls) if call[:1] == ('git',) and 'merge' in call)
    cli = next(i for i, call in enumerate(calls) if call[0] == 'python3' and call[1].endswith('tatbot_cli_install.py'))
    assert merged < cli, 'the runtime is prepared in the merged checkout, before the CLI install'
    assert (checkout / artwork_runtime.PROJECT / 'node_modules' / artwork_runtime.RECEIPT).read_text().strip() == \
        artwork_runtime.sha256_file(checkout / artwork_runtime.PROJECT / artwork_runtime.LOCK_FILE)
    # Prepared once: the next deploy of the same lock runs no npm; a camera
    # node never does; a --service deploy moved no checkout.
    install('arm-node')
    install('capture-left')
    (checkout / artwork_runtime.PROJECT / 'node_modules' / artwork_runtime.RECEIPT).unlink()
    assert installer.prepare_artwork_runtime(installer.argparse.ArgumentParser(), stage, checkout,
                                             nodes['arm-node'], ['tatbot-visiond-d405.service']) is False
    assert len(npm) == 1
    # A failing npm stops the install by name; the checkout keeps its revision.
    monkeypatch.setattr(subprocess, 'run', lambda argv, **kwargs: subprocess.CompletedProcess(argv, 1))
    with pytest.raises(SystemExit) as stopped:
        install('arm-node')
    assert stopped.value.code == 3
    assert json.loads((checkout / '.tatbot-build.json').read_text())['source_sha'] == sha


def test_a_service_install_neither_merges_nor_restamps_the_checkout(tmp_path, monkeypatch):
    old, new = SHAS[1], SHAS[2]
    stamped = make_receipt(tmp_path / 'releases' / old, old)
    stage = tmp_path / 'releases' / new
    make_receipt(stage, new)
    repo = git_checkout(monkeypatch, tmp_path, old)
    (repo / '.tatbot-build.json').write_text(json.dumps(stamped))
    monkeypatch.setattr(installer, 'run', lambda *argv: pytest.fail(f'a service install ran {argv}'))
    monkeypatch.setattr(installer.subprocess, 'check_output',
                        lambda *a, **k: pytest.fail('a service install queried the checkout'))
    installer.update_checkout(installer.argparse.ArgumentParser(), stage, repo, new, 'arm-node',
                              ['tatbot-visiond-d405.service'])
    assert json.loads((repo / '.tatbot-build.json').read_text()) == stamped


def test_viewer_overlay_copies_only_archived_source_and_preserves_local_bytecode(tmp_path):
    old, new = SHAS[1], SHAS[2]
    stage = tmp_path / 'releases' / new
    source = stage / 'source'
    baseline = stage / 'previous-source'
    checkout = tmp_path / 'viewer'
    relative = Path('scripts/lib/tatbot_digest.py')
    pycache = Path('scripts/lib/__pycache__/tatbot_digest.cpython-312.pyc')
    for root in (source, baseline, checkout):
        (root / relative).parent.mkdir(parents=True)
    (source / relative).write_text('new source\n')
    (baseline / relative).write_text('old source\n')
    (checkout / relative).write_text('old source\n')
    for root, content in ((source, b'new generated bytecode'), (checkout, b'local bytecode')):
        (root / pycache).parent.mkdir(parents=True)
        (root / pycache).write_bytes(content)
    (stage / 'previous.sha').write_text(old)
    (checkout / '.tatbot-deploy.json').write_text(json.dumps({'source_commit': old}))
    (stage / 'source-manifest.json').write_text(json.dumps({
        'schema': 'tatbot.receipt/1', 'kind': 'source-manifest', 'source_sha': new,
        'archive_sha256': 'f' * 64,
        'files': {str(relative): {'kind': 'file',
                                  'sha256': hashlib.sha256((source / relative).read_bytes()).hexdigest(),
                                  'executable': False}}}))
    parser = installer.argparse.ArgumentParser()
    installer.overlay_viewer_source(parser, checkout, stage, source, new, 'viewer')
    assert (checkout / relative).read_text() == 'new source\n'
    assert (checkout / pycache).read_bytes() == b'local bytecode'
    assert json.loads((checkout / '.tatbot-deploy.json').read_text())['source_commit'] == new
    (checkout / '.tatbot-deploy.json').write_text(json.dumps({'source_commit': old}))
    (checkout / relative).write_text('local source edit\n')
    with pytest.raises(SystemExit) as refused:
        installer.overlay_viewer_source(parser, checkout, stage, source, new, 'viewer')
    assert refused.value.code == 3
    assert (checkout / relative).read_text() == 'local source edit\n'


def test_referenced_releases_protects_the_release_the_checkout_receipt_names(tmp_path):
    units = tmp_path / 'systemd'
    units.mkdir()
    (units / 'tatbot-visiond-poe.service').write_text(
        f'ExecStart={tmp_path}/releases/{SHAS[3]}/source/scripts/fleet_service.sh visiond-poe\n')
    checkout = tmp_path / 'checkout'
    checkout.mkdir()
    stamped = make_receipt(tmp_path / 'releases' / SHAS[1], SHAS[1])
    (checkout / '.tatbot-build.json').write_text(json.dumps(stamped))
    assert installer.referenced_releases(units, checkout) == {SHAS[3], SHAS[1]}
    # An unstamped or unreadable checkout protects nothing extra.
    assert installer.referenced_releases(units, tmp_path / 'nowhere') == {SHAS[3]}
    (checkout / '.tatbot-build.json').write_text('{"schema": "other"}')
    assert installer.referenced_releases(units, checkout) == {SHAS[3]}
    # The stamp keeps its release across a prune of everything else: the
    # stamped tree is the oldest by mtime, neither installed nor recent.
    (checkout / '.tatbot-build.json').write_text(json.dumps(stamped))
    releases = make_releases(tmp_path / 'releases', [SHAS[0], SHAS[4], SHAS[5]])
    os.utime(releases / SHAS[1], (1_600_000_000, 1_600_000_000))
    removed = installer.prune_releases(releases, keep_sha=SHAS[5],
                                       referenced=installer.referenced_releases(units, checkout), keep=1)
    assert set(removed) == {SHAS[0], SHAS[4]}
    assert (releases / SHAS[1]).is_dir()
