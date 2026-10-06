"""Deploy a pushed fleet source archive; never start or stop an arm process."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shlex
import subprocess
import sys
import time
from contextlib import contextmanager, nullcontext
from pathlib import Path

from fleet_install import checkout_of, select_services
from fleet_source import archive_manifest

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "scripts/lib"))
from tatbot_paths import bootstrap  # noqa: E402

bootstrap()
from tatbot_cli import nodes as _nodes  # noqa: E402

# Deploy targets are ROLES, resolved through config/nodes.json, so no node name
# is frozen here and a clone describing a different fleet deploys to its own
# nodes: the fleet viewer, the camera nodes (PoE and the overhead D555), the
# arm node.
DEPLOY_ROLES = ('rerun-server', 'poe-cameras', 'overhead-depth', 'arm')


def _role_node(role):
    """The single node carrying `role`, or None where the map does not say."""
    names = _nodes.nodes_with(_nodes.load(REPO), role)
    return names[0] if len(names) == 1 else None


DEPLOY_TARGETS = tuple(dict.fromkeys(
    n for n in (_role_node(r) for r in DEPLOY_ROLES) if n))
VIEWER_NODE = _role_node('rerun-server')


# A release build needs several GB. Nothing checked before, so a deploy would
# start a cargo build on a node at 99% and fill the rootfs of a live capture
# owner -- the deploy that needs the space being the one that cannot get it.
# The release tree and the build cache are measured separately because they
# are not always the same filesystem: a node may symlink ~/.cache onto a
# second disk, so a roomy build volume can hide a full system volume.
RELEASE_FREE_BYTES = 1 * 2**30
BUILD_FREE_BYTES = 5 * 2**30

FREE_PROBE = """import shutil, sys
from pathlib import Path
for path in sys.argv[1:]:
    where = Path(path)
    while not where.exists() and where != where.parent:
        where = where.parent
    print(shutil.disk_usage(where).free, where)
"""

# Diagnostics only. The release's build/source receipts remain the authority
# for identity and integrity; failure to measure bytes cannot authorize or
# refuse a deploy. File sizes here are apparent bytes, not network framing or
# allocated disk blocks.
BYTE_PROBE = """import json, os, sys
from pathlib import Path
mode, location, *extra = sys.argv[1:]
path = Path(location)
if mode == 'build':
    receipt = json.loads((path / 'build.json').read_text())
    artifacts = [Path(row['path']) for row in receipt['binaries'].values()]
    artifacts += [path / name for name in receipt.get('components', {})]
    file_bytes = sum((Path(root) / name).lstat().st_size
                     for root, _, files in os.walk(path) for name in files)
    counts = {'release_file_bytes': file_bytes,
              'receipt_artifact_bytes': sum(item.stat().st_size for item in artifacts)}
elif mode == 'install-full':
    stage, unit_root, *units = extra
    names = [file.name for file in (Path(stage) / 'bin').iterdir() if file.is_file()]
    counts = {'checkout_binary_bytes': sum((path / 'rust/target/release' / name).stat().st_size
                                             for name in names),
              'installed_unit_bytes': sum((Path(unit_root) / name).stat().st_size
                                          for name in units),
              'stamp_bytes': (path / '.tatbot-build.json').stat().st_size}
else:
    raise ValueError('unknown diagnostic byte probe')
print(json.dumps(counts))
"""

BYTE_PROBE_KEYS = {
    'build': ('release_file_bytes', 'receipt_artifact_bytes'),
    'install-full': ('checkout_binary_bytes', 'installed_unit_bytes', 'stamp_bytes'),
}

# Read-only generation evidence. MainPID is the runlog shell wrapper, not proof
# of its child executable; these fields are diagnostics, never restart gates.
GENERATION_PROBE = """import hashlib, json, subprocess, sys
from pathlib import Path
stage = Path(sys.argv[1])
names = sys.argv[2:]
properties = 'Id,ActiveState,SubState,MainPID,ExecMainStartTimestampMonotonic,InvocationID,NeedDaemonReload,FragmentPath,ExecStart'
observed = {}
for name in names:
    result = subprocess.run(['systemctl', 'show', '--property=' + properties, name],
                            capture_output=True, text=True, check=True)
    fields = dict(line.split('=', 1) for line in result.stdout.splitlines() if '=' in line)
    if fields.get('Id') != name:
        raise ValueError('systemd returned another unit')
    fragment = Path(fields['FragmentPath'])
    rendered = stage / 'units' / name
    fields['fragment_sha256'] = hashlib.sha256(fragment.read_bytes()).hexdigest()
    fields['rendered_sha256'] = (hashlib.sha256(rendered.read_bytes()).hexdigest()
                                 if rendered.is_file() else None)
    observed[name] = fields
print(json.dumps(observed))
"""


def selected_unit_generations(target, stage, services):
    """Best-effort loaded-unit/process-generation observation, never a gate."""
    names = [service['unit'] for service in services]
    if not names:
        return {}
    try:
        observed = json.loads(remote_output(target, ['python3', '-c', GENERATION_PROBE,
                                                    stage, *names]))
        if (not isinstance(observed, dict) or set(observed) != set(names)
                or not all(isinstance(row, dict) for row in observed.values())):
            return None
        return observed
    except (OSError, ValueError, TypeError, subprocess.SubprocessError):
        return None

# Verify and execute staged attester bytes with code supplied by this pushed
# deploy. Loading the two imports from verified bytes prevents a staged .pyc
# or an untracked module from shadowing even a genuine attester source file.
REUSE_HELPERS = ('scripts/lib/fleet_release.py', 'scripts/lib/fleet_source.py',
                 'scripts/lib/tatbot_digest.py')
TRUSTED_REUSE_RUNNER = """import hashlib, sys, types
from pathlib import Path
stage = Path(sys.argv[1])
source = stage / 'source'
if stage.is_symlink() or source.is_symlink():
    sys.exit(3)
names = ('scripts/lib/fleet_release.py', 'scripts/lib/fleet_source.py',
         'scripts/lib/tatbot_digest.py')
if tuple(sys.argv[3:9:2]) != names:
    sys.exit(3)
paths = [stage / 'source-manifest.json'] + [source / name for name in names]
digests = [sys.argv[2], *sys.argv[4:9:2]]
payloads = {}
for path, digest in zip(paths, digests, strict=True):
    if (path.is_symlink() or not path.is_file()
            or not path.parent.resolve().is_relative_to(source.resolve() if path != paths[0] else stage.resolve())):
        sys.exit(3)
    data = path.read_bytes()
    if hashlib.sha256(data).hexdigest() != digest:
        sys.exit(3)
    payloads[path] = data
if len(sys.argv) == 9:
    print('verified')
    sys.exit(0)
if sys.argv[9] != '--':
    sys.exit(3)
for module, path in (('tatbot_digest', paths[3]), ('fleet_source', paths[2])):
    loaded = types.ModuleType(module)
    loaded.__file__ = str(path)
    sys.modules[module] = loaded
    exec(compile(payloads[path], str(path), 'exec'), loaded.__dict__)
script = paths[1]
sys.argv = [str(script), *sys.argv[10:]]
exec(compile(payloads[script], str(script), 'exec'),
     {'__name__': '__main__', '__file__': str(script), '__package__': None})
"""


def _reuse_runner_argv(stage, manifest, manifest_digest, release_args=()):
    helpers = []
    for relative in REUSE_HELPERS:
        record = manifest['files'].get(relative)
        if not isinstance(record, dict) or record.get('kind') != 'file':
            raise ValueError(f'source archive lacks deploy verifier: {relative}')
        helpers.extend((relative, record['sha256']))
    return ['python3', '-I', '-S', '-c', TRUSTED_REUSE_RUNNER, stage, manifest_digest,
            *helpers, *(['--', *release_args] if release_args else [])]


def remote_byte_counts(target, mode, location, *extra):
    """Best-effort byte measurement, deliberately separate from release checks."""
    try:
        counts = json.loads(remote_output(target, ['python3', '-c', BYTE_PROBE,
                                                  mode, location, *extra]))
        keys = BYTE_PROBE_KEYS[mode]
        if set(counts) != set(keys) or any(type(counts[key]) is not int or counts[key] < 0
                                           for key in keys):
            return None
        return counts
    except (OSError, ValueError, TypeError, subprocess.SubprocessError):
        return None


class DeployMetrics:
    """Per-run arm deploy costs, bound to the selected source and release.

    These are diagnostics in the existing run directory, never a launch or
    reuse receipt. An interrupted run retains completed phases and the failed
    phase; absent phases have not run.
    """

    def __init__(self, run_dir, sha, manifest_sha256, node, release):
        self.path = run_dir / 'deploy-metrics-full-arm.json'
        self.record = {'schema': 'tatbot.receipt/1', 'kind': 'deploy-metrics',
                       'mode': 'full-arm', 'source_sha': sha,
                       'source_manifest_sha256': manifest_sha256,
                       'node': node, 'release': release, 'phases': {}}
        self._write()

    def _write(self):
        try:
            temporary = self.path.with_suffix('.json.next')
            temporary.write_text(json.dumps(self.record, indent=2) + '\n')
            temporary.replace(self.path)
        except OSError as error:
            print(f'deploy diagnostics unavailable: {error}', file=sys.stderr)

    @contextmanager
    def phase(self, name):
        values = {'byte_counts': None, 'end_ns': None}
        start = time.monotonic_ns()
        status = 'complete'
        try:
            yield values
        except BaseException:
            status = 'failed'
            raise
        finally:
            elapsed_ms = round(((values['end_ns'] if values['end_ns'] is not None
                                 else time.monotonic_ns()) - start) / 1_000_000, 3)
            self.record['phases'][name] = {'status': status, 'elapsed_ms': elapsed_ms,
                                           'byte_counts': values['byte_counts']}
            self._write()
            print(f'deploy {name}: {status}, {elapsed_ms:.3f} ms, '
                  f'bytes={values["byte_counts"] if values["byte_counts"] is not None else "unavailable"}',
                  flush=True)


def disk_shortfalls(measured, required):
    """Human-readable shortfalls; empty when every path has its headroom.

    Pure so the arithmetic is testable without a node: `measured` is
    (label, free_bytes, filesystem_path) and `required` the matching needs.
    """
    shortfalls = []
    for (label, free, where), need in zip(measured, required, strict=True):
        if free < need:
            shortfalls.append(
                f'{label}: {free // 2**20} MiB free on {where}, needs {need // 2**20} MiB')
    return shortfalls


def remote_output(target, argv):
    return output(['ssh', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=10', target,
                   shlex.join(str(x) for x in argv)])


def preflight_disk(parser, target, release_path, build_path):
    """Refuse before staging rather than fail with a full disk mid-build."""
    rows = remote_output(target, ['python3', '-c', FREE_PROBE,
                                  release_path, build_path]).splitlines()
    if len(rows) != 2:
        parser.exit(5, f'deploy refused: could not measure disk on {target}\n')
    labels = ('release tree', 'build cache')
    measured = []
    for label, row in zip(labels, rows, strict=True):
        free, where = row.split(None, 1)
        measured.append((label, int(free), where))
    shortfalls = disk_shortfalls(measured, (RELEASE_FREE_BYTES, BUILD_FREE_BYTES))
    if shortfalls:
        parser.exit(5, 'deploy refused: not enough disk on the build node\n  '
                    + '\n  '.join(shortfalls) + '\n')


def arguments(parser):
    """Shared operator arguments; importing this module performs no deployment."""
    parser.add_argument('node', choices=('all', *DEPLOY_TARGETS))
    parser.add_argument('--service', action='append', default=[],
                        help='build and deploy only this exact manifested unit on one node; repeatable')
    parser.add_argument('--build-only', action='store_true',
                        help='stage and compile without installing system configuration or restarting services')


def run(argv, **kwargs):
    print('+ ' + shlex.join(str(x) for x in argv), flush=True)
    return subprocess.run(argv, check=True, **kwargs)


def output(argv):
    return subprocess.check_output(argv, text=True).strip()


def remote(target, argv, **kwargs):
    return run(['ssh', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=10', target,
                shlex.join(str(x) for x in argv)], **kwargs)


def stage_previous_viewer_source(parser, target, checkout, stage, baseline):
    """The viewer's currently deployed source goes beside the release so its
    build can diff against what it replaces. Only a tar-overlay viewer needs
    that: the install fast-forwards a git checkout (fleet_install.update_checkout),
    which has no overlay receipt to read on its first viewer deploy."""
    if subprocess.run(['ssh', '-o', 'BatchMode=yes', target, f'test -d {checkout}/.git']).returncode == 0:
        print(f'{target}: the viewer checkout is git, so no overlay baseline is staged', flush=True)
        return
    old = subprocess.check_output(['ssh', '-o', 'BatchMode=yes', target,
        f"cat {checkout}/.tatbot-deploy.json"], text=True)
    previous_sha = json.loads(old)['source_commit']
    if len(previous_sha) != 40 or any(c not in '0123456789abcdef' for c in previous_sha):
        parser.exit(3, 'invalid previous viewer source receipt\n')
    with baseline.open('xb') as file:
        run(['git', 'archive', previous_sha], stdout=file)
    remote(target, ['mkdir', '-p', stage + '/previous-source'])
    with baseline.open('rb') as file:
        remote(target, ['tar', '-xf', '-', '-C', stage + '/previous-source'], stdin=file)
    remote(target, ['bash', '-c', 'printf %s ' + shlex.quote(previous_sha) + ' > ' + shlex.quote(stage + '/previous.sha')])


def native_build_steps(source, fetchcontent, stage, checkout, node):
    """Shell lines that build the arm node's native pieces beside the Rust binaries:
    the offline samples planner (path_plan_check), teleop and recovery."""
    component = source + '/scripts/lib/fleet_component_cache.py'
    reuse = shlex.join(['python3', component, 'reuse', '--stage', stage,
                        '--checkout', checkout, '--node', node])
    record = shlex.join(['python3', component, 'record', '--stage', stage])
    build_dir = source + '/cpp/teleop/build'
    configured = ('cmake -S ' + shlex.quote(source + '/cpp/teleop') + ' -B '
                  + shlex.quote(build_dir) + ' "${TROSSEN_FETCH_ARGS[@]}"')
    additional = ['wxai_teleop', 'arm_recover']
    build_new = shlex.join(['cmake', '--build', build_dir, '--target',
                            'path_plan_check', *additional, '-j2'])
    build_other = shlex.join(['cmake', '--build', build_dir, '--target',
                              *additional, '-j2'])
    return [
        'command -v cmake >/dev/null || { echo "arm staging requires cmake" >&2; exit 5; }',
        shlex.join(['mkdir', '-p', fetchcontent]),
        'TROSSEN_FETCH_ARGS=(' + shlex.quote('-DFETCHCONTENT_BASE_DIR=' + fetchcontent) + ')',
        'if [ -d ' + shlex.quote(fetchcontent + '/trossen_arm_sdk-src/include') + ' ]; then '
        'TROSSEN_FETCH_ARGS+=(' + shlex.quote('-DFETCHCONTENT_SOURCE_DIR_TROSSEN_ARM_SDK=' + fetchcontent + '/trossen_arm_sdk-src') + '); fi',
        'export TROSSEN_ARM_SDK_ROOT=' + shlex.quote(fetchcontent + '/trossen_arm_sdk-src'),
        'if ' + reuse + '; then PLAN_REUSED=1; else PLAN_REUSED=0; fi',
        configured,
        'if [ "$PLAN_REUSED" = 0 ]; then ' + build_new + '; else ' + build_other + '; fi',
        record,
    ]


# The stencil observer's interpreter: a venv in the release, built from the
# observer's own requirements on the node that runs the observer service (the
# `track` role's node manifests it). The camera node has no simulator or
# LeRobot environment, and the observer must not inherit either.
OBSERVER_UNIT = 'tatbot-stencild.service'
OBSERVER_REQUIREMENTS = 'scripts/vision/requirements-observer.txt'

# These are the service branches of fleet_service.sh. A new unit needs an
# explicit dependency decision before a deploy may stage or switch it.
SERVICE_ENVIRONMENTS = {
    'tatbot-zenohd.service': (),
    'tatbot-zenoh-presence.service': (),
    'tatbot-visiond-poe.service': (),
    'tatbot-visiond-d405.service': (),
    'tatbot-visiond-d555.service': (),
    'tatbot-trackd.service': (),
    'tatbot-trackd-left.service': (),
    OBSERVER_UNIT: ('observer-venv',),
}


def service_environments(services):
    units = {service['unit'] for service in services}
    unknown = units - SERVICE_ENVIRONMENTS.keys()
    if unknown:
        raise ValueError('unknown service runtime dependency: ' + ', '.join(sorted(unknown)))
    return sorted({environment for unit in units for environment in SERVICE_ENVIRONMENTS[unit]})


def observer_venv_steps(stage, services):
    """Shell lines that build the observer venv when the observer is selected."""
    if not any(service['unit'] == OBSERVER_UNIT for service in services):
        return []
    venv = stage + '/observer-venv'
    return [
        'command -v uv >/dev/null || { echo "observer staging requires uv" >&2; exit 5; }',
        shlex.join(['uv', 'venv', '--allow-existing', venv]),
        shlex.join(['uv', 'pip', 'install', '--python', venv + '/bin/python',
                    '-r', stage + '/source/' + OBSERVER_REQUIREMENTS]),
        shlex.join([venv + '/bin/python', '-c', 'import cv2, numpy']),
    ]


def component_plan(binaries, services, arm_node, receipt):
    """The selected build closure reported beside the per-node reuse result.

    A full deploy (`receipt`) stages the bus controller the liveliness check
    runs; the arm node also carries the offline samples planner
    (path_plan_check), with teleop and recovery beside it.
    """
    service_envs = service_environments(services)
    units = {service['unit'] for service in services}
    return {
        'launch_binaries': ((['fleetctl'] + (['path_plan_check'] if arm_node else []))
                            if receipt and 'fleetctl' in binaries else []),
        'service_units': [service['unit'] for service in services],
        'components': sorted({'bin/' + service['binary'] for service in services}
                             | ({'observer-venv/bin/python'} if OBSERVER_UNIT in units else set())
                             | ({'source/cpp/teleop/build/wxai_teleop',
                                 'source/cpp/teleop/build/arm_recover',
                                 'source/planner-component.json'} if arm_node else set())),
        'native_targets': ['path_plan_check', 'wxai_teleop', 'arm_recover'] if arm_node else [],
        'environments': service_envs,
        'generated_trees': ['source/cpp/teleop/build'] if arm_node else [],
    }


def build_receipt_steps(stage, sha, node, binaries, arm_node, services, *, receipt=True):
    """The remote build's last line: the receipt naming what was built.

    Every node's full deploy stages the bus controller; the arm node also
    carries the planner. A binary the receipt must name and cannot find fails
    the build. A `--service` deploy writes no receipt.
    """
    plan = component_plan(binaries, services, arm_node, receipt)
    if not plan['launch_binaries']:
        return []
    return [shlex.join(['python3', stage + '/source/scripts/lib/fleet_release.py', 'build',
                        '--stage', stage, '--sha', sha, '--node', node,
                        *(arg for name in plan['launch_binaries'] for arg in ('--binary', name)),
                        *(arg for path in plan['components'] for arg in ('--component', path)),
                        *(arg for path in plan['generated_trees'] for arg in ('--tree', path))])]


def reusable_build(target, stage, sha, node_name, node, binaries, services, manifest):
    """Reuse only the complete Rust/service build with its exact source archive.

    The arm CMake tree requires a whole-tree receipt. Generated service
    environments remain outside the no-op path until their closure is
    explicitly bound.
    """
    arm_node = 'arm' in node.get('roles', [])
    if service_environments(services):
        return False
    plan = component_plan(binaries, services, arm_node, True)
    if not plan['launch_binaries']:
        return False
    digest = hashlib.sha256(json.dumps(manifest).encode()).hexdigest()
    release_args = ['reusable', '--stage', stage, '--sha', sha, '--node', node_name,
                    '--manifest-sha256', digest,
                    *(arg for name in plan['launch_binaries'] for arg in ('--binary', name)),
                    *(arg for path in plan['components'] for arg in ('--component', path)),
                    *(arg for path in plan['generated_trees'] for arg in ('--tree', path))]
    argv = _reuse_runner_argv(stage, manifest, digest, release_args)
    try:
        return remote_output(target, argv) == sha
    except subprocess.CalledProcessError:
        return False


def cargo_build_steps(binaries, services, stage):
    """Build each selected crate into the stage's bin/, plus zenohd when a service needs it."""
    build = []
    for binary, spec in binaries.items():
        command = ['cargo', 'build', '--locked', '--release', '-p', spec['package']]
        if spec['features']:
            command += ['--features', ','.join(spec['features'])]
        build += [shlex.join(command),
                  f'install -m 755 "$CARGO_TARGET_DIR/release/{binary}" {shlex.quote(stage + "/bin/" + binary + ".next")}',
                  shlex.join(["mv", "-f", stage + "/bin/" + binary + ".next", stage + "/bin/" + binary])]
    if any(s['binary'] == 'zenohd' for s in services):
        build += ['cargo install zenohd --version 1.10.0 --locked --no-default-features --features zenoh/transport_tcp --root ' + shlex.quote(stage)]
    return build


def _deploy_selection(parser, args):
    if args.service and args.node == 'all':
        parser.error('--service requires a specific node')
    os.chdir(Path(__file__).resolve().parents[2])
    run(['git', 'fetch', 'origin', 'main'])
    sha = output(['git', 'rev-parse', 'HEAD'])
    if sha != output(['git', 'rev-parse', 'origin/main']):
        parser.exit(3, 'deploy refused: HEAD must equal pushed origin/main\n')
    nodes = json.loads(output(['git', 'show', f'{sha}:config/nodes.json']))
    if args.service:
        try:
            select_services(nodes[args.node]['services'], args.service)
        except ValueError as error:
            parser.exit(3, f'deploy refused: {error}\n')
    selected = DEPLOY_TARGETS if args.node == 'all' else (args.node,)
    try:
        for name in selected:
            service_environments(select_services(nodes[name]['services'], args.service))
    except (KeyError, ValueError) as error:
        parser.exit(3, f'deploy refused: {error}\n')
    return sha, nodes, selected


def _source_archive(run_dir, sha):
    archive = run_dir / 'source.tar'
    with archive.open('xb') as file:
        run(['git', 'archive', sha], stdout=file)
    manifest = archive_manifest(archive, sha)
    (run_dir / 'source-manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    payload = json.dumps(manifest).encode()
    return archive, manifest, payload, hashlib.sha256(payload).hexdigest()


def _stage_context(parser, args, name, node, sha, digest, run_dir):
    target = node['ssh']
    services = select_services(node['services'], args.service)
    user = target.split('@', 1)[0]
    stage = f'/home/{user}/.local/share/tatbot/releases/{sha}'
    arm_node = 'arm' in node.get('roles', []) and not args.service
    # A full deploy stages the bus controller the liveliness check runs.
    binaries = {binary: {'package': binary, 'features': []}
                for binary in (() if args.service else ('fleetctl',))}
    for service in services:
        if service.get('package'):
            binaries[service['binary']] = service
    try:
        plan = {'staged_manifest_sha256': digest,
                **component_plan(binaries, services, arm_node, not args.service)}
    except ValueError as error:
        parser.exit(3, f'deploy refused: {error}\n')
    metrics = DeployMetrics(run_dir, sha, digest, name, stage) if arm_node else None
    return {'name': name, 'node': node, 'target': target, 'services': services, 'user': user,
            'stage': stage, 'arm_node': arm_node, 'binaries': binaries,
            'plan': plan, 'metrics': metrics}


def _reuse_stage(args, context, sha, manifest):
    metrics = context['metrics']
    started = time.perf_counter()
    reused = not args.service and reusable_build(
        context['target'], context['stage'], sha, context['name'], context['node'],
        context['binaries'], context['services'], manifest)
    if metrics:
        metrics.record['reuse_check'] = {'elapsed_ms': round((time.perf_counter() - started) * 1000, 3),
                                         'reused': reused}
        metrics._write()
    return reused


def _stage_previous_viewer(parser, args, context, run_dir):
    if context['name'] == VIEWER_NODE and not args.build_only:
        stage_previous_viewer_source(
            parser, context['target'], context['node'].get('checkout') or '~/tatbot',
            context['stage'], run_dir / f"previous-{context['name']}.tar")


def _transfer_source(context, archive, payload, sha):
    target, stage, metrics = context['target'], context['stage'], context['metrics']
    with (metrics.phase('transfer') if metrics else nullcontext({})) as phase:
        remote(target, ['mkdir', '-p', stage + '/source', stage + '/bin'])
        remote(target, ['touch', stage + '/build.incomplete'])
        with archive.open('rb') as file:
            remote(target, ['tar', '-xf', '-', '-C', stage + '/source'], stdin=file)
        remote(target, ['python3', '-c',
                        'import pathlib,sys; p=pathlib.Path(sys.argv[1]); t=p.with_suffix(".next"); t.write_bytes(sys.stdin.buffer.read()); t.replace(p)',
                        stage + '/source-manifest.json'], input=payload)
        remote(target, ['python3', stage + '/source/scripts/lib/fleet_source.py',
                        '--repo', stage + '/source', '--verify-archive', '--expected', sha])
        if metrics:
            phase['byte_counts'] = {'source_archive_bytes': archive.stat().st_size,
                                    'source_manifest_bytes': len(payload)}


def _claim_fresh_stage(parser, target, stage):
    """Never extract over a complete or partial same-SHA release tree."""
    remote(target, ['mkdir', '-p', str(Path(stage).parent)])
    try:
        remote(target, ['mkdir', stage])
    except subprocess.CalledProcessError:
        parser.exit(3, 'deploy refused: existing same-SHA stage failed reuse verification; '
                       'preserve it for inspection and use a fresh release identity\n')


def _build_commands(args, context, sha):
    stage, name, user = context['stage'], context['name'], context['user']
    services, binaries = context['services'], context['binaries']
    build = ['set -eu', 'export PATH="$HOME/.cargo/bin:$HOME/.local/bin:$PATH"',
             f'cd {shlex.quote(stage + "/source/rust")}',
             f'export TATBOT_SOURCE_COMMIT={shlex.quote(sha)}',
             f'export CARGO_TARGET_DIR="$HOME/.cache/tatbot-fleet-build/{name}"',
             'command -v cargo >/dev/null || { echo "missing cargo toolchain on build node" >&2; exit 5; }',
             shlex.join(['source', stage + '/source/scripts/lib/fleet_toolchain.sh']),
             shlex.join(['fleet_toolchain::use', stage + '/source']),
             'export CARGO_BUILD_JOBS=2 CMAKE_BUILD_PARALLEL_LEVEL=2 UV_CONCURRENT_BUILDS=2']
    if context['arm_node']:
        # Build/install-to-stage only: no driver, serial, lease, or motor action.
        checkout = str(checkout_of(context['node'], Path('/home') / user))
        build += native_build_steps(stage + '/source', f'/home/{user}/.cache/tatbot-fetchcontent',
                                    stage, checkout, name)
    build += cargo_build_steps(binaries, services, stage)
    build += observer_venv_steps(stage, services)
    build += [shlex.join(['python3', stage + '/source/scripts/lib/fleet_source.py',
                           '--repo', stage + '/source', '--verify-archive', '--expected', sha]),
              'printf %s ' + shlex.quote(sha) + ' > ' + shlex.quote(stage + '/build.sha')]
    receipt = build_receipt_steps(stage, sha, name, binaries, context['arm_node'], services,
                                  receipt=not args.service)
    return build + (receipt or [shlex.join(['rm', stage + '/build.incomplete'])])


def _run_stage_build(args, context, sha):
    metrics, target, stage = context['metrics'], context['target'], context['stage']
    build = _build_commands(args, context, sha)
    with (metrics.phase('build') if metrics else nullcontext({})) as phase:
        remote(target, ['bash', '-c', '\n'.join(build)])
        if metrics:
            phase['end_ns'] = time.monotonic_ns()
            phase['byte_counts'] = remote_byte_counts(target, 'build', stage)


def _stage_target(parser, args, name, node, sha, manifest, digest, run_dir, archive, payload):
    context = _stage_context(parser, args, name, node, sha, digest, run_dir)
    target, stage = context['target'], context['stage']
    reused = _reuse_stage(args, context, sha, manifest)
    if reused:
        print(f'Reusing verified fleet build on {name}: {sha[:12]}')
        _stage_previous_viewer(parser, args, context, run_dir)
    else:
        preflight_disk(parser, target, stage,
                       f"/home/{context['user']}/.cache/tatbot-fleet-build/{name}")
        _claim_fresh_stage(parser, target, stage)
        _transfer_source(context, archive, payload, sha)
        _stage_previous_viewer(parser, args, context, run_dir)
        try:
            _run_stage_build(args, context, sha)
        except subprocess.CalledProcessError as error:
            return None, context['metrics'], error.returncode
    row = {'node': name, 'target': target, 'stage': stage, 'sha': sha,
           'controller': stage + '/bin/fleetctl' if 'fleetctl' in context['binaries'] else None,
           'reused': reused, 'component_plan': context['plan']}
    return row, context['metrics'], None


def _install_service_row(row, node, sha, service_args, services, metrics):
    command = ['python3', row['stage'] + '/source/scripts/lib/fleet_install.py',
               '--stage', row['stage'], '--node', row['node'], '--sha', sha, *service_args]
    if not metrics or row['node'] != metrics.record['node']:
        remote(row['target'], command)
        return
    checkout = str(checkout_of(node, Path('/home') / row['target'].split('@', 1)[0]))
    with metrics.phase('install') as phase:
        remote(row['target'], command)
        phase['end_ns'] = time.monotonic_ns()
        phase['byte_counts'] = remote_byte_counts(
            row['target'], 'install-full', checkout, row['stage'], '/etc/systemd/system',
            *(service['unit'] for service in services))


def _verified_install_noop(row, sha, metrics):
    """Try the read-only installed-state proof for a reused full arm build."""
    if not metrics or row['node'] != metrics.record['node'] or not row.get('reused'):
        return False
    command = ['python3', row['stage'] + '/source/scripts/lib/fleet_install.py',
               '--stage', row['stage'], '--node', row['node'], '--sha', sha,
               '--verify-installed']
    started = time.perf_counter()
    try:
        remote(row['target'], command)
        accepted = True
    except (OSError, subprocess.SubprocessError):
        accepted = False
    metrics.record['installed_state_probe'] = {
        'elapsed_ms': round((time.perf_counter() - started) * 1000, 3),
        'accepted': accepted}
    metrics._write()
    return accepted


def _service_query_byte_counts(run_dir):
    """Sizes of saved liveliness responses, excluding systemctl and SSH framing."""
    try:
        return {label: (run_dir / name).stat().st_size if (run_dir / name).is_file() else 0
                for label, name in (('service_query_stdout_bytes', 'services.json'),
                                    ('service_query_stderr_bytes', 'services.stderr'))}
    except OSError:
        return None


def _verify_service_health(parser, stages, nodes, requested, run_dir, sha, metrics):
    with (metrics.phase('health') if metrics else nullcontext({})) as phase:
        try:
            verify_deployment(parser, stages, nodes, requested, run_dir, sha)
        finally:
            if metrics:
                phase['byte_counts'] = _service_query_byte_counts(run_dir)


def _install_or_skip_service_row(row, nodes, sha, service_args, selected, noops, metrics):
    if row['node'] in noops:
        metrics.record['install_action'] = 'verified_noop'
        with metrics.phase('install') as phase:
            phase['byte_counts'] = {'checkout_binary_bytes': 0,
                                    'installed_unit_bytes': 0, 'stamp_bytes': 0}
        return
    if metrics and row['node'] == metrics.record['node']:
        metrics.record['install_action'] = 'installed'
        metrics._write()
    _install_service_row(row, nodes[row['node']], sha, service_args, selected, metrics)


def _install_service_bundles(parser, args, stages, nodes, run_dir, sha, metrics=None):
    service_args = [arg for unit in args.service for arg in ('--service', unit)]
    # Probe before the normal preflight: a proven repeat leaves even the staged
    # unit files untouched. Any missing proof takes the existing install path.
    noops = ({row['node'] for row in stages if _verified_install_noop(row, sha, metrics)}
             if not args.build_only and not args.service else set())
    for row in stages:
        if row['node'] in noops:
            continue
        remote(row['target'], ['python3', row['stage'] + '/source/scripts/lib/fleet_install.py',
                              '--stage', row['stage'], '--node', row['node'], '--sha', sha,
                              '--verify-only', *service_args])
    if args.build_only:
        print('All selected builds completed; no checkout, unit or process changed.')
        return
    # Privilege is checked only after concrete binaries and units exist.
    for row in stages:
        if row['node'] not in noops:
            remote(row['target'], ['sudo', '-n', 'true'])
    arm_stage = next((row for row in stages if metrics and row['node'] == metrics.record['node']), None)
    if arm_stage:
        metrics.record['service_generations'] = {
            'before': selected_unit_generations(arm_stage['target'], arm_stage['stage'],
                                                select_services(nodes[arm_stage['node']]['services'], args.service)),
            'after': None}
        metrics._write()
    try:
        for row in stages:
            _install_or_skip_service_row(
                row, nodes, sha, service_args,
                select_services(nodes[row['node']]['services'], args.service), noops, metrics)
        _verify_service_health(parser, stages, nodes, args.service, run_dir, sha, metrics)
    finally:
        if arm_stage:
            metrics.record['service_generations']['after'] = selected_unit_generations(
                arm_stage['target'], arm_stage['stage'],
                select_services(nodes[arm_stage['node']]['services'], args.service))
            metrics._write()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    arguments(parser)
    args = parser.parse_args()
    sha, nodes, selected = _deploy_selection(parser, args)
    run_dir = Path(os.environ['TATBOT_RUN_DIR'])
    archive, manifest, payload, digest = _source_archive(run_dir, sha)
    stages, build_errors, metrics = [], [], None
    # Build every selected target before installing any selected service.
    for name in selected:
        row, staged_metrics, error = _stage_target(parser, args, name, nodes[name], sha, manifest,
                                                   digest, run_dir, archive, payload)
        metrics = staged_metrics or metrics
        if error is not None:
            build_errors.append({'node': name, 'exit': error})
            continue
        stages.append(row)
        (run_dir / 'built.json').write_text(json.dumps(stages, indent=2) + '\n')
    if build_errors:
        (run_dir / 'build-errors.json').write_text(json.dumps(build_errors, indent=2) + '\n')
        print('Build failures recorded; no selected service installed', file=sys.stderr)
        sys.exit(1)
    _install_service_bundles(parser, args, stages, nodes, run_dir, sha, metrics)


def bus_controller(parser, stages, nodes):
    """The `fleetctl` the liveliness check runs, on the bus-router node.

    The controller comes from a build receipt: the
    stage this deploy built and installed on the router when it staged one
    there, else the release the router's checkout is stamped with (a
    `--service` deploy of another node builds no controller anywhere, and the
    stamp's release is protected from pruning). Nothing is compiled here.
    Returns the router's ssh target, its bus endpoint and the binary's path.
    """
    name, router = next((name, node) for name, node in nodes.items()
                        if isinstance(node, dict) and 'bus-router' in node.get('roles', []))
    target = router['ssh']
    endpoint = 'tcp/' + router['lan'] + ':7447'
    staged = next((row['controller'] for row in stages
                   if row['node'] == name and row.get('controller')), None)
    if staged:
        return target, endpoint, staged
    checkout = str(checkout_of(router, Path('/home') / target.split('@', 1)[0]))
    try:
        path = remote_output(target, ['python3', checkout + '/scripts/lib/fleet_release.py',
                                      'verify', '--checkout', checkout, '--binary', 'fleetctl'])
    except subprocess.CalledProcessError:
        path = ''
    if len(path.splitlines()) != 1 or not path.endswith('/bin/fleetctl'):
        parser.exit(3, 'deploy: the bus-router node has no verified controller for the liveliness check; '
                    'a full deploy of that node stamps its checkout with one\n')
    return target, endpoint, path


def verify_deployment(parser, stages, nodes, requested, run_dir, sha):
    import time
    # Display subscribers intentionally publish no sensor liveliness: doing so
    # would impersonate the camera owner. Verify their units independently.
    for row in stages:
        units = [s['unit'] for s in select_services(nodes[row['node']]['services'], requested)
                 if not s.get('liveliness')]
        if not units:
            continue
        first = remote_output(row['target'], ['systemctl', 'show', '--property=Id,ActiveState,NRestarts', *units])
        time.sleep(5)
        second = remote_output(row['target'], ['systemctl', 'show', '--property=Id,ActiveState,NRestarts', *units])
        if first != second or any('ActiveState=active' not in block for block in second.split('\n\n')):
            parser.exit(5, f'deploy: selected services are not stable on {row["node"]}\n{second}\n')
    requirements = [f"{row['node']}/{s['liveliness']}" for row in stages
                    for s in select_services(nodes[row['node']]['services'], requested) if s.get('liveliness')]
    if not requirements:
        print('Selected subscriber services are active; inspect viewer status for frame freshness.')
        return
    # The liveliness query runs on the bus-router node with a deployed
    # controller: the workstation builds nothing and needs no bus route.
    target, endpoint, controller = bus_controller(parser, stages, nodes)
    query = [controller, '--connect', endpoint, 'services', '--expected-sha', sha]
    for service in requirements:
        query += ['--require', service]
    command = ['ssh', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=10', target, shlex.join(query)]
    deadline = time.monotonic() + 60
    while True:
        result = subprocess.run(command, capture_output=True, text=True)
        (run_dir / 'services.json').write_text(result.stdout)
        (run_dir / 'services.stderr').write_text(result.stderr)
        if result.returncode == 0:
            print(result.stdout)
            break
        if result.returncode != 5 or time.monotonic() >= deadline:
            print(result.stdout, result.stderr, file=sys.stderr)
            sys.exit(result.returncode)
        time.sleep(2)
    print('Selected services have matching source liveliness; hardware phase acceptance is separate.')


if __name__ == '__main__':
    try:
        main()
    except subprocess.CalledProcessError as error:
        sys.exit(error.returncode)
