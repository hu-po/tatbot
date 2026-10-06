"""Install already-built fleet services on their node. No arm process is touched."""
from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
from pathlib import Path

import artwork_runtime
import fleet_release
import fleet_source
import schemas
import tatbot_cli_install


def run(*argv):
    print('+', *argv, flush=True)
    return subprocess.run(argv, check=True)


RELEASE_SHA = re.compile(r'[0-9a-f]{40}\Z')
RELEASES_KEPT = 3
SYSTEMD_UNIT_ROOT = Path('/etc/systemd/system')
CGROUP_ROOT = Path('/sys/fs/cgroup')
PROC_ROOT = Path('/proc')
HANDOFF_RULE = Path('/etc/sudoers.d/tatbot-d405-owner')


def referenced_releases(units_dir=Path('/etc/systemd/system'), checkout: Path | None = None) -> set[str]:
    """Release shas an installed unit still points at, plus the one the
    checkout's build receipt names: its binaries live in that release tree,
    so pruning it would strand the checkout. Never prune these."""
    referenced = set()
    for unit in sorted(units_dir.glob('tatbot-*.service')):
        try:
            text = unit.read_text()
        except OSError:
            continue
        referenced.update(re.findall(r'/releases/([0-9a-f]{40})', text))
    if checkout is not None:
        try:
            release = fleet_release.load(checkout / fleet_release.STAMP)['release']
        except ValueError:
            release = ''
        referenced.update(re.findall(r'/releases/([0-9a-f]{40})\Z', release))
    return referenced


def prune_releases(releases: Path, keep_sha: str, referenced: set[str],
                   keep: int = RELEASES_KEPT) -> list[str]:
    """Delete superseded release trees.

    Nothing pruned them before: 31 had accumulated on one node at ~150 MB
    each, and the deploy that needed the space was the one that could not
    get it. Protected are the sha just installed, every sha an installed
    unit still references, and the `keep` most recent by mtime so a rollback
    target survives. Only 40-hex names are ever considered, so an unrelated
    directory under the release root is never a candidate.
    """
    if not releases.is_dir():
        return []
    trees = [p for p in releases.iterdir() if p.is_dir() and RELEASE_SHA.match(p.name)]
    recent = sorted(trees, key=lambda p: p.stat().st_mtime, reverse=True)[:keep]
    protected = {keep_sha, *referenced, *(p.name for p in recent)}
    removed = []
    for tree in sorted(trees):
        if tree.name in protected:
            continue
        shutil.rmtree(tree, ignore_errors=True)
        removed.append(tree.name)
    return removed


def d405_handoff_rule(user: str) -> str:
    """Only start/stop of the camera owner; no shell, wildcard or arm unit."""
    if re.fullmatch(r'[a-z_][a-z0-9_-]*', user) is None:
        raise ValueError('invalid service account')
    return (f'{user} ALL=(root) NOPASSWD: /usr/bin/systemctl start tatbot-visiond-d405.service, '
            '/usr/bin/systemctl stop tatbot-visiond-d405.service\n')


def select_services(services, requested):
    """Select exact manifested units; refuse typos before any remote mutation."""
    if not requested:
        return services
    if len(requested) != len(set(requested)):
        raise ValueError('duplicate service selector')
    unknown = set(requested) - {service['unit'] for service in services}
    if unknown:
        raise ValueError('service not manifested on selected node: ' + ', '.join(sorted(unknown)))
    return [service for service in services if service['unit'] in requested]


def overlay_viewer_source(p, repo, stage, source, sha, node):
    """Tar-overlay viewer: only replace files whose old content matches the
    previous deployment manifest. A changed or untracked collision fails."""
    previous = repo / '.tatbot-deploy.json'
    if not previous.exists():
        p.exit(3, 'install refused: viewer overlay has no source receipt\n')
    # The caller supplies an archive of the prior pushed source for comparison.
    baseline = stage / 'previous-source'
    if not baseline.is_dir() or (stage / 'previous.sha').read_text() != json.loads(previous.read_text())['source_commit']:
        p.exit(3, 'install refused: prior viewer source must be staged for comparison\n')
    # Build and verification may create bytecode or other untracked files in
    # the staged tree. The source manifest names exactly what git archived;
    # only those files belong in the viewer's checkout overlay.
    try:
        fleet_source.verify_archive(source, sha)
    except (OSError, ValueError) as error:
        p.exit(3, f'install refused: staged source changed: {error}\n')
    manifest = json.loads((stage / 'source-manifest.json').read_text())
    files = [source / relative for relative in manifest['files']]
    for file in files:
        relative = file.relative_to(source)
        target = repo / relative
        old = baseline / relative
        if any(parent.is_symlink() for parent in (target, *target.parents)) or file.is_symlink():
            p.exit(3, f'install refused: symlink overlay path {relative}\n')
        if target.exists() and (not target.is_file() or target.read_bytes() != file.read_bytes()) and (not old.is_file() or target.read_bytes() != old.read_bytes()):
            p.exit(3, f'install refused: preserve changed overlay file {relative}\n')
    for file in files:
        target = repo / file.relative_to(source)
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary = target.with_name(target.name + '.fleet-deploy')
        shutil.copy2(file, temporary)
        temporary.replace(target)
    previous.write_text(json.dumps({'source_commit': sha, 'target': node}) + '\n')


def checkout_of(node: dict, home: Path) -> Path:
    """The node's command path: the checkout `config/nodes.json` names, which
    the CLI hop runs in and a full deploy brings to the built revision."""
    checkout = node.get('checkout') or '~/tatbot'
    return home / checkout[2:] if checkout.startswith('~/') else Path(checkout)


def update_checkout(p, stage: Path, repo: Path, sha: str, node: str, service: list[str]) -> None:
    """Bring the command path to the built revision and stamp it with the
    release's build receipt. A `--service` deploy carries no launch binary:
    it neither merges nor re-stamps, so the mirror keeps the sha and receipt
    a launcher already verifies against."""
    if service:
        return
    try:
        receipt = fleet_release.load(stage / fleet_release.RECEIPT)
    except ValueError as error:
        p.exit(3, f'install refused: {error}\n')
    if receipt['source_sha'] != sha:
        p.exit(3, 'install refused: build receipt names another source\n')
    source = stage / 'source'
    if (repo / '.git').exists():
        dirty = subprocess.check_output(['git', '-C', str(repo), 'status', '--porcelain', '--untracked-files=no'], text=True)
        if dirty.strip():
            p.exit(3, 'install refused: target has tracked edits; preserved\n')
        run('git', '-C', str(repo), 'fetch', 'origin', 'main')
        run('git', '-C', str(repo), 'merge', '--ff-only', sha)
        actual = subprocess.check_output(['git', '-C', str(repo), 'rev-parse', 'HEAD'], text=True).strip()
        if actual != sha:
            p.exit(3, 'install refused: target is ahead of the staged source; preserved\n')
    else:
        overlay_viewer_source(p, repo, stage, source, sha, node)
    stamp = repo / fleet_release.STAMP
    temporary = stamp.with_name(stamp.name + '.fleet-deploy')
    shutil.copy2(stage / fleet_release.RECEIPT, temporary)
    temporary.replace(stamp)


def prepare_artwork_runtime(p, stage: Path, repo: Path, node: dict, service: list[str]) -> bool:
    """The design stage's Node dependencies in the checkout the CLI hop runs
    in, so the first design compile after a deploy pays no install. Only the
    arm node runs that stage; a
    `--service` deploy moved no checkout. npm resolves as the build did: the
    release's private runtime, the user's provisioned one, then PATH."""
    if service or 'arm' not in node.get('roles', []):
        return False
    npm = artwork_runtime.find_npm((stage / 'toolchain/node/bin',))
    try:
        return artwork_runtime.prepare(repo, npm=npm)
    except ValueError as error:
        p.exit(3, f'install stopped: {error}; the checkout is at the built revision and its first '
                  'design compile retries the install\n')


def _install_context(p, args):
    """Resolve the selected release and refuse mismatched node or bundle."""
    stage = args.stage.resolve()
    source = stage / 'source'
    if (stage / 'build.sha').read_text() != args.sha:
        p.exit(3, 'install refused: build receipt differs from requested source\n')
    nodes = json.loads((source / 'config/nodes.json').read_text())
    if args.node not in nodes:
        p.exit(3, 'install refused: unknown node in staged manifest\n')
    node = nodes[args.node]
    try:
        services = select_services(node['services'], args.service)
    except ValueError as error:
        p.exit(3, f'install refused: {error}\n')
    user = node['ssh'].split('@', 1)[0]
    home = Path.home()
    if home != Path('/home') / user:
        p.exit(4, 'install refused: node account mismatch\n')
    return stage, source, node, services, user, home


def _unit_text(stage, source, unit, user, home, node):
    template = (source / 'config/systemd' / unit).read_text()
    rendered = (template.replace('@USER@', user).replace('@HOME@', str(home))
                .replace('@RELEASE@', str(stage)).replace('@NODE@', node))
    if any(token in rendered for token in ('@USER@', '@HOME@', '@RELEASE@', '@NODE@')):
        raise ValueError('unresolved unit template')
    return rendered


def _render_units(p, args, stage, source, services, user, home):
    """Render and validate selected units before any installed unit changes."""
    # Only these explicitly manifested camera/bus/tracker units can be changed.
    allowed = {'tatbot-zenohd.service', 'tatbot-zenoh-presence.service', 'tatbot-visiond-poe.service',
               'tatbot-visiond-d405.service', 'tatbot-visiond-d555.service',
               'tatbot-trackd.service', 'tatbot-trackd-left.service', 'tatbot-stencild.service'}
    for service in services:
        if service['unit'] not in allowed:
            p.exit(3, 'install refused: unapproved service in manifest\n')
        if not os.access(stage / 'bin' / service['binary'], os.X_OK):
            p.exit(3, 'install refused: required binary absent\n')
    units = stage / 'units'
    units.mkdir(exist_ok=True)
    for service in services:
        name = service['unit']
        try:
            rendered = _unit_text(stage, source, name, user, home, args.node)
        except ValueError as error:
            p.exit(3, f'{error}\n')
        (units / name).write_text(rendered)
    # Validate all files before writing a system unit; a FAIL never starts one.
    if services:
        run('systemd-analyze', 'verify', *(str(units / s['unit']) for s in services))
    handoff_rule = None
    if any(s['unit'] == 'tatbot-visiond-d405.service' for s in services):
        handoff_rule = stage / 'tatbot-d405-owner.sudoers'
        handoff_rule.write_text(d405_handoff_rule(user))
        run('/usr/sbin/visudo', '-cf', str(handoff_rule))
    return units, handoff_rule


def _copy_checkout_binaries(stage, repo, services, args):
    binaries = repo / 'rust/target/release'
    binaries.mkdir(parents=True, exist_ok=True)
    selected = {service['binary'] for service in services}
    for binary in (stage / 'bin').iterdir():
        if args.service and binary.name not in selected:
            continue
        temporary = binaries / (binary.name + '.fleet-deploy')
        shutil.copy2(binary, temporary)
        temporary.replace(binaries / binary.name)


def _activate_services(services, units, handoff_rule):
    for service in services:
        run('sudo', '-n', 'install', '-m', '644', str(units / service['unit']), '/etc/systemd/system/' + service['unit'])
    if handoff_rule is not None:
        run('sudo', '-n', 'install', '-o', 'root', '-g', 'root', '-m', '440',
            str(handoff_rule), '/etc/sudoers.d/tatbot-d405-owner')
    if services:
        run('sudo', '-n', 'systemctl', 'daemon-reload')
        for service in services:
            run('sudo', '-n', 'systemctl', 'enable', service['unit'])
            run('sudo', '-n', 'systemctl', 'restart', service['unit'])


def _installed_binaries(stage: Path, repo: Path, receipt: dict) -> dict[str, str]:
    """Prove every copied executable still equals the verified release."""
    digests = {name: record['sha256'] for name, record in receipt['binaries'].items()
               if Path(record['path']).parent == stage / 'bin'}
    for relative, digest in receipt.get('components', {}).items():
        if Path(relative).parent == Path('bin'):
            name = Path(relative).name
            if name in digests and digests[name] != digest:
                raise ValueError(f'conflicting executable receipt: {name}')
            digests[name] = digest
    staged = stage / 'bin'
    if ({path.name for path in staged.iterdir()} != set(digests)
            or any(path.is_symlink() or not path.is_file() for path in staged.iterdir())):
        raise ValueError('staged executable set differs from its receipt')
    for name, digest in digests.items():
        installed = repo / 'rust/target/release' / name
        if (installed.is_symlink() or not installed.is_file() or not os.access(installed, os.X_OK)
                or fleet_release.sha256_file(installed) != digest):
            raise ValueError(f'checkout executable differs from release: {name}')
    return digests


def _loaded_service_matches(stage: Path, unit: str, binary: str, digest: str) -> None:
    """Prove the loaded unit and its one service executable, not just disk files."""
    rendered = stage / 'units' / unit
    installed = SYSTEMD_UNIT_ROOT / unit
    if (installed.is_symlink() or not installed.is_file()
            or installed.read_bytes() != rendered.read_bytes()):
        raise ValueError(f'installed unit differs from release: {unit}')
    command = next((line.removeprefix('ExecStart=') for line in rendered.read_text().splitlines()
                    if line.startswith('ExecStart=')), '')
    result = subprocess.run(
        ['systemctl', 'show', '--property=Id,ActiveState,SubState,MainPID,'
         'ExecMainStartTimestampMonotonic,InvocationID,NeedDaemonReload,FragmentPath,'
         'ExecStart,UnitFileState,ControlGroup', unit],
        capture_output=True, text=True, timeout=10, check=False)
    if result.returncode:
        raise ValueError(f'service generation unavailable: {unit}')
    fields = dict(line.split('=', 1) for line in result.stdout.splitlines() if '=' in line)
    if (fields.get('Id') != unit or fields.get('ActiveState') != 'active'
            or fields.get('SubState') != 'running' or fields.get('NeedDaemonReload') != 'no'
            or fields.get('UnitFileState') != 'enabled'
            or fields.get('FragmentPath') != str(installed)
            or not command.startswith(str(stage / 'source/scripts/fleet_service.sh'))
            or f'path={command.split()[0]}' not in fields.get('ExecStart', '')
            or not fields.get('InvocationID')):
        raise ValueError(f'loaded service differs from installed release: {unit}')
    main_pid = int(fields.get('MainPID', '0'))
    started = int(fields.get('ExecMainStartTimestampMonotonic', '0'))
    group = fields.get('ControlGroup', '')
    if (main_pid <= 0 or started <= 0 or not group.startswith('/')
            or '..' in Path(group).parts or Path(group).name != unit):
        raise ValueError(f'service process generation is unproved: {unit}')
    cgroup = (CGROUP_ROOT / group.lstrip('/')).resolve()
    if not cgroup.is_relative_to(CGROUP_ROOT.resolve()):
        raise ValueError(f'service cgroup escapes its root: {unit}')
    pids = (cgroup / 'cgroup.procs').read_text().split()
    if not 1 <= len(pids) <= 64 or str(main_pid) not in pids or any(not pid.isdecimal() for pid in pids):
        raise ValueError(f'service process set is unproved: {unit}')
    expected = str(stage / 'bin' / binary)
    running = [pid for pid in pids if os.readlink(PROC_ROOT / pid / 'exe') == expected]
    if (len(running) != 1
            or fleet_release.sha256_file(PROC_ROOT / running[0] / 'exe') != digest):
        raise ValueError(f'active executable differs from release: {unit}')


def _installed_artwork_and_cli(repo: Path, installed: dict) -> None:
    """Check the exact prepared artwork tree and launcher without writing either."""
    project = repo / artwork_runtime.PROJECT
    if (not artwork_runtime.current(project,
                                    fleet_release.sha256_file(project / artwork_runtime.LOCK_FILE))
            or installed.get('artwork_runtime_sha256') !=
            fleet_release.tree_contents(project / 'node_modules')['sha256']):
        raise ValueError('checkout artwork runtime differs from its lock')
    launcher = repo.resolve() / 'scripts/tatbot'
    link = tatbot_cli_install.DESTINATION
    if (not link.is_symlink() or os.readlink(link) != str(launcher)
            or not os.access(launcher, os.X_OK)
            or subprocess.run([str(link), '--version'], capture_output=True,
                              text=True, timeout=10, check=False,
                              env={**os.environ, 'PYTHONDONTWRITEBYTECODE': '1'}).returncode):
        raise ValueError('installed Tatbot CLI differs from checkout')


def verify_installed(stage: Path, repo: Path, sha: str, node: str,
                     services: list[dict], user: str, home: Path) -> None:
    """Read-only same-release proof; any gap requires the normal install."""
    expected = fleet_release.verify(stage / 'source')
    if (fleet_release.verify(repo) != expected or expected['source_sha'] != sha
            or expected.get('node') != node or expected['release'] != str(stage)):
        raise ValueError('checkout receipt differs from selected release')
    installed = json.loads((stage / 'installed.json').read_bytes())
    if (installed.get('schema') != schemas.RECEIPT or installed.get('kind') != 'deployment'
            or installed.get('sha') != sha or installed.get('node') != node
            or installed.get('mode') != 'services'
            or installed.get('units') != [service['unit'] for service in services]):
        raise ValueError('previous installation receipt differs')
    for service in services:
        unit = stage / 'units' / service['unit']
        if unit.is_symlink() or unit.read_text() != _unit_text(stage, stage / 'source', service['unit'],
                                                                user, home, node):
            raise ValueError(f'staged unit differs from source: {service["unit"]}')
    if any(service['unit'] == 'tatbot-visiond-d405.service' for service in services):
        expected_rule = stage / 'tatbot-d405-owner.sudoers'
        if (expected_rule.is_symlink() or expected_rule.read_text() != d405_handoff_rule(user)
                or HANDOFF_RULE.is_symlink() or not HANDOFF_RULE.is_file()
                or HANDOFF_RULE.read_bytes() != expected_rule.read_bytes()):
            raise ValueError('installed camera handoff rule differs from release')
    digests = _installed_binaries(stage, repo, expected)
    _installed_artwork_and_cli(repo, installed)
    for service in services:
        name = service['binary']
        if name not in digests:
            raise ValueError(f'service executable lacks a build receipt: {name}')
        _loaded_service_matches(stage, service['unit'], name, digests[name])


def _install_release(p, args, stage, node, services, home, units, handoff_rule):
    run('sudo', '-n', 'true')
    repo = checkout_of(node, home)
    update_checkout(p, stage, repo, args.sha, args.node, args.service)
    artwork_digest = None
    prepare_artwork_runtime(p, stage, repo, node, args.service)
    if not args.service and 'arm' in node.get('roles', []):
        artwork_digest = fleet_release.tree_contents(
            repo / artwork_runtime.PROJECT / 'node_modules')['sha256']
    # A deployed checkout is also an operator endpoint in non-login PATHs.
    run('python3', str(repo / 'scripts/lib/tatbot_cli_install.py'), '--repo', str(repo))
    _copy_checkout_binaries(stage, repo, services, args)
    _activate_services(services, units, handoff_rule)
    receipt = {**schemas.stamp(schemas.RECEIPT, 'deployment'), 'sha': args.sha, 'node': args.node,
               'units': [s['unit'] for s in services], 'mode': 'services'}
    if artwork_digest is not None:
        receipt['artwork_runtime_sha256'] = artwork_digest
    (stage / 'installed.json').write_text(json.dumps(receipt, indent=2) + '\n')
    for name in prune_releases(stage.parent, args.sha, referenced_releases(checkout=repo)):
        print('+ pruned superseded release', name, flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--stage', type=Path, required=True)
    p.add_argument('--node', required=True)
    p.add_argument('--sha', required=True)
    p.add_argument('--verify-only', action='store_true', help='render and verify staged units without installing')
    p.add_argument('--verify-installed', action='store_true',
                   help='prove an unchanged installed arm release without restarting services')
    p.add_argument('--service', action='append', default=[], help='install only this exact manifested unit; repeatable')
    args = p.parse_args()
    if args.verify_installed and (args.verify_only or args.service):
        p.error('--verify-installed requires an unfiltered full-service release')
    stage, source, node, services, user, home = _install_context(p, args)
    if args.verify_installed and 'arm' not in node.get('roles', []):
        p.exit(3, 'install refused: installed-state proof is arm-only\n')
    if args.verify_installed:
        try:
            verify_installed(stage, checkout_of(node, home), args.sha, args.node,
                             services, user, home)
        except (OSError, ValueError, KeyError, TypeError, subprocess.SubprocessError) as error:
            p.exit(3, f'install proof unavailable: {error}\n')
        print('Installed release and selected service generations verified; no service restarted.')
        return
    units, handoff_rule = _render_units(p, args, stage, source, services, user, home)
    if args.verify_only:
        print('Staged unit files verified; no checkout or system service changed.')
        return
    _install_release(p, args, stage, node, services, home, units, handoff_rule)


if __name__ == '__main__':
    main()
