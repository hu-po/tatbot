"""Separate fidelity, placement and measured cost; promotion requires A/A and holdout evidence."""
from __future__ import annotations

from pathlib import Path

import numpy as np
from tatbot_session.fidelity import identity
from tatbot_session.runtime import software_digest

from draw_research.model import ResearchError, event, frozen, load_study, locked, read, seal, write
from draw_research.prepare import load_trial
from draw_research.run import _execution_context, _receipt_state

LOWER = {'missing_fraction', 'spill_fraction', 'spill_area_mm2', 'draw_session_s'}
METRICS = LOWER | {'iou'}


def score(root, trial_id):
    root = Path(root)
    with locked(root):
        trial = load_trial(root, trial_id)
        directory = root/'trials'/trial_id
        state = read(directory/'state.json')
        if any(side['status'] != 'complete' for side in state['sides'].values()):
            raise ResearchError('both sides must finish without skipped or uncertain operations before scoring')
        sides = {}
        for name, side in trial['sides'].items():
            receipt = read(directory/name/'evidence.json')
            program = read(directory/name/'program.json')
            if _receipt_state(receipt, program, side) != 'complete' or receipt['meta']['runtime'] != state['runtime']:
                raise ResearchError('execution evidence no longer matches the paired trial')
            _execution_context(state, receipt, program)
            fidelity = receipt['inspection']['analysis'].get('fidelity')
            if not fidelity or fidelity['scorer'] != trial['scorer'] or trial['scorer'] != identity() or fidelity['metrics'] is None:
                raise ResearchError('both sides need valid slot-isolated inspection from the same current scorer')
            # Physical contact has no instrumented duration; metadata alone cannot establish it.
            sides[name] = {'fidelity': fidelity,
                           'draw_session_s': _draw_seconds(receipt), 'timing': receipt.get('timing'),
                           'client_wall_s': _elapsed(state['sides'][name]['attempts']),
                           'planned_s': side['modeled_s'], 'pen_down_s': None,
                           'run_id': receipt['claim']['run_id'], 'inspection': receipt['inspection']['directory'],
                           'inspection_files': receipt['inspection']['files']}
        result = seal({'schema': 'tatbot.dbv3-score/1', 'trial_sha256': trial['content_sha256'],
                       'runtime': state['runtime'], 'execution_context': state['execution_context'], 'scorer': identity(), 'sides': sides,
                       'human_preference': None})
        path = directory/'score.json'
        if path.exists() and frozen(path, 'tatbot.dbv3-score/1') != result:
            raise ResearchError('score identity changed; preserve this result and record a separate reanalysis')
        write(path, result)
        if state['stage'] != 'decided':
            state['stage'] = 'scored'
        write(directory/'state.json', state)
        event(root, 'scored', trial=trial_id, score_sha256=result['content_sha256'])
        return result


def _elapsed(attempts):
    return sum(v['elapsed_s'] for v in attempts) if attempts and all('elapsed_s' in v for v in attempts) else None


def _draw_seconds(receipt):
    timing = receipt.get('timing') or {}
    value = timing.get('duration_s')
    if (timing.get('schema') != 'tatbot.draw-timing/1' or timing.get('complete') is not True
            or isinstance(value, bool) or not isinstance(value, (int, float)) or not np.isfinite(value) or value <= 0):
        return None
    return value


def _require_measured_times(evidence):
    for trial, scored in evidence:
        for side in ('a', 'b'):
            seconds = scored['sides'][side].get('draw_session_s')
            if (isinstance(seconds, bool) or not isinstance(seconds, (int, float))
                    or not np.isfinite(seconds) or seconds <= 0):
                raise ResearchError(f"trial {trial['id']} side {side} needs a positive measured drawing-session duration "
                                    "for baseline qualification or promotion; retain incomplete evidence as inconclusive")


def _evidence(root, trial_ids, *, allow_unscored=False):
    output = []
    if len(set(trial_ids)) != len(trial_ids):
        raise ResearchError('decision evidence must name distinct trials')
    for trial_id in trial_ids:
        trial = load_trial(root, trial_id)
        path = root/'trials'/trial_id/'score.json'
        if allow_unscored and not path.exists():
            output.append((trial, None))
            continue
        scored = frozen(path, 'tatbot.dbv3-score/1')
        if scored['trial_sha256'] != trial['content_sha256'] or (not allow_unscored and scored['scorer'] != identity()):
            raise ResearchError('decision requires exact trials scored with the current algorithm')
        output.append((trial, scored))
    if not output or (not allow_unscored and len({software_digest(s['runtime']) for _, s in output}) != 1):
        raise ResearchError('decision evidence requires one consistent deployed runtime')
    if not allow_unscored and any(s['execution_context'] != output[0][1]['execution_context'] for _, s in output):
        raise ResearchError('decision evidence requires consistent tool, registration and execution configuration')
    return output


def decide(root, trial_ids, *, decision, reason, metric='missing_fraction', min_gain=0., max_time_ratio=1.25):
    root = Path(root)
    if decision not in ('qualify', 'promote', 'keep', 'inconclusive') or not reason.strip():
        raise ResearchError('record a decision and its reasoning')
    if metric not in METRICS or not np.isfinite(min_gain) or min_gain < 0 or not np.isfinite(max_time_ratio) or max_time_ratio <= 0:
        raise ResearchError('declare a supported metric, nonnegative minimum gain and positive time ratio')
    with locked(root):
        evidence = _evidence(root, trial_ids, allow_unscored=decision == 'inconclusive')
        state = read(root/'state.json')
        if decision in ('qualify', 'promote') and any(s['execution_context']['hardware'] != 'real' for _, s in evidence):
            raise ResearchError('physical DBV3 qualification/promotion requires real-hardware ink evidence; mock runs are software checks')
        if decision in ('qualify', 'promote'):
            _require_measured_times(evidence)
        request = seal({'schema': 'tatbot.dbv3-decision-request/1', 'decision': decision, 'reason': reason,
                        'trials': [{'id': t['id'], 'sha256': t['content_sha256'], 'score_sha256': s['content_sha256'] if s else None} for t, s in evidence],
                        'metric': metric, 'min_gain': min_gain, 'max_time_ratio': max_time_ratio})
        folder = root/'decisions'
        folder.mkdir(exist_ok=True)
        path = folder/(request['content_sha256']+'.json')
        if path.exists():
            record = frozen(path, 'tatbot.dbv3-decision/1')
        else:
            policy = load_study(root)['promotion']
            target = None
            if decision == 'qualify':
                target = _qualify(evidence, state, policy)
            elif decision == 'promote':
                target = _promote(evidence, state, policy, metric, min_gain, max_time_ratio)
            record = seal({'schema': 'tatbot.dbv3-decision/1', 'request': request,
                           'previous_baseline': state['baseline'], 'baseline': target})
            write(path, record)
        # The immutable decision is the write-ahead record. Retrying repairs an
        # interrupted pointer/state update without undoing a later promotion.
        if record['baseline'] is not None and state['baseline'] == record['previous_baseline']:
            state['baseline'] = record['baseline']
            write(root/'state.json', state)
        for trial, _ in evidence:
            path = root/'trials'/trial['id']/'state.json'
            progress = read(path)
            progress['stage'] = 'decided'
            write(path, progress)
        event(root, 'decided', decision=decision, decision_sha256=record['content_sha256'])
        return record


def _qualified_target(trial, side):
    return {'candidate_sha256': trial['sides'][side]['candidate_sha256'],
            'policy_sha256': trial['sides'][side]['policy_sha256'], 'evidence_trial': trial['id'], 'side': side}


def _qualify(evidence, state, policy):
    target = _qualified_target(evidence[0][0], 'a')
    if state['baseline'] is not None and state['baseline'] != target:
        raise ResearchError('a qualified baseline already exists; use paired promotion')
    if len(evidence) < policy['aa_pairs'] or any(t['kind'] != 'aa' or t['sides']['a']['candidate_sha256'] != target['candidate_sha256'] for t, _ in evidence):
        raise ResearchError('initial DBV3 qualification requires at least two valid A/A pairs on the same frozen candidate')
    return target


def _promote(evidence, state, policy, metric, min_gain, max_time_ratio):
    if state['baseline'] is None:
        raise ResearchError('qualify a DBV3 baseline with A/A first')
    baseline = state['baseline']['candidate_sha256']
    if any(t['sides']['a']['candidate_sha256'] != baseline for t, _ in evidence):
        raise ResearchError('promotion evidence must compare against the current frozen baseline')
    noise = [(t, s) for t, s in evidence if t['kind'] == 'aa']
    pairs = [(t, s) for t, s in evidence if t['kind'] == 'paired']
    train = [t for t, _ in pairs if t['source_split'] == 'train']
    validation = [t for t, _ in pairs if t['source_split'] == 'validation']
    if len(noise) < policy['aa_pairs'] or len(train) < policy['train_pairs'] or len(validation) < policy['validation_pairs']:
        raise ResearchError('promotion needs two A/A pairs, two training pairs and a held-out validation pair')
    target = _qualified_target(pairs[0][0], 'b')
    if any(t['sides']['b']['candidate_sha256'] != target['candidate_sha256'] for t, _ in pairs):
        raise ResearchError('promotion must refer to one exact candidate across training and validation')
    spread = {m: max(abs(_metric(s['sides']['a'], m) - _metric(s['sides']['b'], m)) for _, s in noise)
              for m in {metric, 'missing_fraction', 'spill_fraction'}}
    for _, scored in pairs:
        a, b = (scored['sides'][k] for k in ('a', 'b'))
        first, second = a['fidelity']['metrics'], b['fidelity']['metrics']
        gain = _metric(a, metric)-_metric(b, metric) if metric in LOWER else _metric(b, metric)-_metric(a, metric)
        if gain <= max(min_gain, spread[metric]):
            raise ResearchError('candidate gain does not exceed observed A/A variation on every training/validation repeat')
        if any(second[m]-first[m] > spread[m] for m in ('missing_fraction', 'spill_fraction')):
            raise ResearchError('candidate trades a fidelity loss beyond the A/A spread; retain it as a separate tradeoff')
        if b['draw_session_s'] is None or a['draw_session_s'] is None or b['draw_session_s'] > max_time_ratio*a['draw_session_s']:
            raise ResearchError('candidate exceeds the declared measured drawing-session time ratio')
    return target


def _metric(side, name):
    value = side.get(name) if name == 'draw_session_s' else side['fidelity']['metrics'][name]
    if value is None or not np.isfinite(value):
        raise ResearchError(f'promotion needs a measured finite {name} on both sides')
    return value
