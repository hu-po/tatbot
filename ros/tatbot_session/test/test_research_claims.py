"""Lost clients cannot allocate another draw to an occupied physical slot."""
import copy
import json
from concurrent.futures import ThreadPoolExecutor

import pytest
from tatbot_session.research import claim_slot


def program():
    return {'format': 'tatbot-program', 'version': 1, 'ops': [], 'research': {
        'study_sha256': 'a'*64, 'trial': '0000', 'side': 'a', 'page': 'physical-sheet-001', 'slot': '0:0', 'slot_m': [.026, .026]}}


def test_claim_is_persistent_and_only_exact_original_run_can_resume(tmp_path):
    original = program()
    claim_slot(tmp_path, original, 'run-one')
    claim_slot(tmp_path, original, 'run-one', resume=True)
    for run, resume in [('run-one', False), ('run-two', False), ('run-two', True)]:
        with pytest.raises(ValueError, match='occupied'):
            claim_slot(tmp_path, original, run, resume=resume)
    changed = copy.deepcopy(original)
    changed['ops'] = [{'op': 'pause', 'id': 'p0'}]
    with pytest.raises(ValueError, match='occupied'):
        claim_slot(tmp_path, changed, 'run-one', resume=True)
    assert json.loads((tmp_path/'research-pages/physical-sheet-001/0-0.json').read_text())['run_id'] == 'run-one'


def test_racing_clients_only_one_can_claim(tmp_path):
    def try_claim(index):
        try:
            claim_slot(tmp_path, program(), f'run-{index}')
            return True
        except (ValueError, FileExistsError):
            return False
    with ThreadPoolExecutor(max_workers=8) as pool:
        assert sum(pool.map(try_claim, range(8))) == 1


def test_torn_claim_and_unadmitted_resume_are_refusals(tmp_path):
    with pytest.raises(ValueError, match='no durable'):
        claim_slot(tmp_path, program(), 'run-one', resume=True)
    path = tmp_path/'research-pages/physical-sheet-001/0-0.json'
    path.parent.mkdir(parents=True)
    path.write_text('{"run_id":')
    with pytest.raises(ValueError):
        claim_slot(tmp_path, program(), 'run-two')
    assert path.read_text() == '{"run_id":'
    claim_slot(tmp_path, {'ops': []}, 'ordinary-run')


def test_motion_admission_refuses_drift_before_claim(tmp_path):
    import hashlib

    from tatbot_session.research import validate_motion

    p = program()
    p['preparation'] = {'motion_sha256': hashlib.sha256(json.dumps({'speed': 1}, sort_keys=True).encode()).hexdigest()}
    validate_motion(p, 'speed: 1\n')
    with pytest.raises(ValueError, match='motion configuration changed'):
        validate_motion(p, 'speed: 2\n')
    validate_motion({'ops': []}, 'speed: 2\n')
