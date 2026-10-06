"""Durable research slot ownership and read-only run evidence on the ROS execution owner.

This supplements the existing ledger. It never chooses or dispatches a stroke.
An admitted slot remains occupied even when execution fails or its client vanishes.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
from pathlib import Path

from tatbot_contracts.canonical import canonical_digest, parse_json


def _claim_path(log_root, page, slot):
    if not isinstance(page, str) or not re.fullmatch(r'[a-zA-Z0-9][a-zA-Z0-9_-]{0,79}', page):
        raise ValueError('research page requires a bounded physical page identifier')
    if not isinstance(slot, str) or not re.fullmatch(r'\d{1,3}:[01]', slot):
        raise ValueError('research slot requires row:column')
    return Path(log_root)/'research-pages'/page/(slot.replace(':', '-')+'.json')


def claim_slot(log_root, program, run_id, *, resume=False):
    context = program.get('research')
    if context is None:
        return
    if set(context) != {'study_sha256', 'trial', 'side', 'page', 'slot', 'slot_m'} or context['side'] not in ('a', 'b'):
        raise ValueError('invalid research execution context')
    path = _claim_path(log_root, context['page'], context['slot'])
    value = {'schema': 'tatbot.research-slot/1', **context, 'run_id': run_id, 'program_sha256': canonical_digest(program)}
    if path.exists():
        existing = parse_json(path.read_bytes())
        if not resume or existing != value:
            differences = sorted(key for key in set(existing) | set(value) if existing.get(key) != value.get(key))
            reason = f'changed {", ".join(differences)}' if differences else 'new draw request'
            raise ValueError(f'research slot is occupied by run {existing.get("run_id")} ({reason}); '
                             'reconcile its ledger before resuming')
        return
    if resume:
        raise ValueError('research run has no durable slot claim; inspect its evidence before proceeding')
    path.parent.mkdir(parents=True, exist_ok=True)
    # O_EXCL arbitrates clients/workspaces on this execution owner. A torn claim
    # remains occupied and unreadable; it is never interpreted as a blank slot.
    with path.open('x') as stream:
        json.dump(value, stream, sort_keys=True, allow_nan=False)
        stream.write('\n')
        stream.flush()
        os.fsync(stream.fileno())
    directory = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(directory)
    finally:
        os.close(directory)


def validate_motion(program, motion_text):
    """A research program must execute with the preparation's recorded motion law."""
    if not program.get('research'):
        return
    import yaml

    actual = hashlib.sha256(json.dumps(yaml.safe_load(motion_text), sort_keys=True).encode()).hexdigest()
    if actual != program['preparation']['motion_sha256']:
        raise ValueError('research motion configuration changed after preparation')


def _inspection(run):
    folders = sorted((run/'inspect').glob('*/meta.json'))
    if not folders:
        return None
    folder = folders[-1].parent
    files = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in folder.iterdir() if p.is_file()}
    return {'directory': str(folder), 'meta': parse_json((folder/'meta.json').read_bytes()),
            'analysis': json.loads((folder/'analysis.json').read_text(), parse_constant=lambda _: None), 'files': files}


def evidence(runtime_record, log_root, page, slot):
    path = _claim_path(log_root, page, slot)
    runtime_path = Path(runtime_record)
    from tatbot_session import runtime

    identity = runtime.current(parse_json(runtime_path.read_bytes())) if runtime_path.exists() else None
    live = identity is not None and runtime.alive(identity)
    result = {'schema': 'tatbot.research-evidence/1', 'runtime': identity, 'runtime_live': live, 'claim': None}
    if not path.exists():
        return result
    claim = parse_json(path.read_bytes())
    run_id = claim['run_id']
    if Path(run_id).name != run_id or run_id in ('.', '..'):
        raise ValueError('invalid run identity in slot claim')
    run = Path(log_root)/'ros-draw'/run_id
    from tatbot_contracts.ros_progress import read

    from tatbot_session import timing

    result.update(claim=claim, program=parse_json((run/'program.json').read_bytes()),
                  meta=json.loads((run/'meta.json').read_text()), ledger=read(run/'ledger.jsonl'),
                  motion_yaml=(run/'motion.yaml').read_text(), inspection=_inspection(run),
                  timing=timing.summarize(run/'run.jsonl'))
    return result


def main():
    from tatbot_session import config, runtime

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--page', required=True)
    parser.add_argument('--slot', required=True)
    parser.add_argument('--runtime-record', help="the stack's runtime record; default the workspace root's, "
                        '$TATBOT_REPO/../runtime.json')
    args = parser.parse_args()
    repo = Path(os.environ['TATBOT_REPO'])
    runlog = config.runlog(repo)
    record = args.runtime_record or runtime.workspace_record(repo)
    print(json.dumps(evidence(record, runlog.log_root(runlog.load_config()), args.page, args.slot), allow_nan=False))


if __name__ == '__main__':
    main()
