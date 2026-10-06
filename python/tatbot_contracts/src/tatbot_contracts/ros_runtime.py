"""Runtime evidence for paired research: imported Python and the launched controller's mapped code.

This is provenance, not a motion interlock. Missing or changing evidence makes a research comparison
inadmissible; ordinary drawing keeps its existing rules.
"""
from __future__ import annotations

import hashlib
import json
import os
import xml.etree.ElementTree as ET
from pathlib import Path

from tatbot_contracts.canonical import canonical_digest, parse_json
from tatbot_contracts.process import PROC, alive, process_record


def record_controller(path, pid):
    """Called by the launcher for this specific controller process, never by PID search."""
    publish(path, process_record(pid))


def publish(path, value):
    path = Path(path).expanduser()
    temporary = path.with_suffix('.pending')
    temporary.write_text(json.dumps(value, sort_keys=True)+'\n')
    temporary.replace(path)


def _executable_files(pid, *, proc=PROC):
    files = {}
    for line in (proc/str(pid)/'maps').read_text().splitlines():
        fields = line.split(maxsplit=5)
        if 'x' not in fields[1]:
            continue
        path = fields[5] if len(fields) == 6 else ''
        if path in ('[vdso]', '[vsyscall]'):
            continue
        if not path.startswith('/') or path.endswith(' (deleted)'):
            raise ValueError('controller has unidentifiable executable mappings')
        major, minor = (int(n, 16) for n in fields[3].split(':'))
        files[path] = (os.makedev(major, minor), int(fields[4]))
    if not files:
        raise ValueError('controller has no file-backed executable mappings')
    return files


def _file_identity(path, mapped):
    with Path(path).open('rb') as stream:
        before = os.fstat(stream.fileno())
        if (before.st_dev, before.st_ino) != mapped:
            raise ValueError(f'loaded controller file was replaced: {path}')
        digest = hashlib.sha256()
        for chunk in iter(lambda: stream.read(1024*1024), b''):
            digest.update(chunk)
        after = os.fstat(stream.fileno())
        if (before.st_size, before.st_mtime_ns, before.st_ctime_ns) != (after.st_size, after.st_mtime_ns, after.st_ctime_ns):
            raise ValueError(f'controller file changed while hashing: {path}')
    return {'path': path, 'device': before.st_dev, 'inode': before.st_ino, 'sha256': digest.hexdigest()}


def controller_identity(reference, *, proc=PROC):
    result = {'complete': False, 'process': None, 'files': [], 'files_sha256': None, 'reason': None}
    try:
        if not reference:
            raise ValueError('launcher did not identify its controller process')
        stamp = parse_json(Path(reference).read_bytes())
        result['process'] = stamp
        if not alive(stamp, proc=proc):
            raise ValueError('recorded controller process is no longer live')
        executable = (proc/str(stamp['pid'])/'exe').readlink()
        if executable.name != 'ros2_control_node':
            raise ValueError('recorded process is not ros2_control_node')
        mapped = _executable_files(stamp['pid'], proc=proc)
        files = [_file_identity(path, mapped[path]) for path in sorted(mapped)]
        if not alive(stamp, proc=proc) or mapped != _executable_files(stamp['pid'], proc=proc):
            raise ValueError('controller process or mappings changed while collecting evidence')
        result.update(complete=True, files=files, files_sha256=canonical_digest({'files': files}))
    except (OSError, ValueError, KeyError, IndexError, TypeError, AttributeError) as error:
        result['reason'] = str(error)
    return result


def configuration_digest(stack, description, controllers):
    robot = ET.fromstring(description)
    for param in robot.findall('.//ros2_control/hardware/param[@name="flight_path"]'):
        param.text = ''
    dynamic = {'flight_path', 'controller_process_file', 'config_path', 'runtime_record', 'runtime_configuration_sha256',
               'service_refresh_id', 'runtime_workspace_sha256'}
    return canonical_digest({'stack': {k: v for k, v in stack.items() if k not in dynamic},
                             'robot_description': ET.tostring(robot, encoding='unicode'),
                             'controllers_sha256': hashlib.sha256(controllers).hexdigest()})


def workspace_description(repo, render):
    path = Path(repo)/'config/workspace.yaml'
    raw = path.read_bytes()
    description = render()
    if path.read_bytes() != raw:
        raise ValueError('workspace changed while constructing the launched robot description')
    return description, hashlib.sha256(raw).hexdigest()


def workspace_record(repo):
    return Path(repo).parent/'runtime.json'


def software_digest(identity):
    """Compare completed pairs across restarts while retaining process evidence separately."""
    try:
        if identity['schema'] != 'tatbot.ros-runtime/2' or identity['controller']['complete'] is not True:
            raise ValueError('incomplete research runtime')
        files = [{'path': item['path'], 'sha256': item['sha256']} for item in identity['controller']['files']]
        if not files or not identity['sources_sha256'] or not identity['configuration_sha256']:
            raise ValueError('missing research code identity')
        return canonical_digest({'schema': 'tatbot.ros-software/1',
                                 **{k: identity[k] for k in ('sources_sha256', 'configuration_sha256', 'python', 'numpy')},
                                 'controller_files': sorted(files, key=lambda item: item['path'])})
    except (KeyError, TypeError) as error:
        raise ValueError('missing research code identity') from error


def _sources(roots):
    if not isinstance(roots, dict) or not roots:
        raise ValueError('runtime has no source roots')
    files = {}
    for name, value in sorted(roots.items()):
        if not isinstance(name, str) or not name:
            raise ValueError('runtime source root is missing or invalid')
        root = _source_root(value)
        candidates = sorted(root.rglob('*.py'))
        if not candidates:
            raise ValueError(f'runtime source root {name} contains no Python source')
        for path in candidates:
            files[f'{name}/{path.relative_to(root).as_posix()}'] = path
    return files


def _source_root(value):
    root = Path(value)
    if not root.is_absolute() or not root.is_dir():
        raise ValueError('runtime source root is missing or invalid')
    return root


def source_digest(roots):
    """Hash one stable source manifest, including additions/removals during collection."""
    files = _sources(roots)
    before = {name: _source_stamp(path) for name, path in files.items()}
    hashes = {name: hashlib.sha256(path.read_bytes()).hexdigest() for name, path in files.items()}
    if _sources(roots) != files or any(_source_stamp(path) != before[name] for name, path in files.items()):
        raise ValueError('runtime source changed while collecting evidence')
    return canonical_digest(hashes)


def _source_stamp(path):
    info = path.stat()
    return info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns, info.st_ctime_ns


def source_status(identity):
    result = {'complete': False, 'sha256': None, 'reason': None}
    try:
        result['sha256'] = source_digest(identity.get('source_roots'))
        if result['sha256'] != identity.get('sources_sha256'):
            raise ValueError('runtime source differs from its startup snapshot')
        result['complete'] = True
    except (OSError, ValueError, TypeError) as error:
        result['reason'] = str(error)
    return result
