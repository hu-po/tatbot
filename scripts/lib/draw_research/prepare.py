"""Freeze candidates and prepare paired programs before reserving any robot execution."""
from __future__ import annotations

import copy
import shutil
from pathlib import Path

import numpy as np
from drawingbot.recipe import verify_bundle
from tatbot_contracts.artwork import validate_path_artwork
from tatbot_contracts.ros_program import validate_for_execution
from tatbot_ink import compile, write_preview, write_program
from tatbot_session.fidelity import identity as scorer_identity

from draw_research.model import (
    ResearchError,
    case_map,
    differences,
    digest,
    event,
    frozen,
    identifier,
    layout_from_mm,
    load_study,
    locked,
    read,
    seal,
    write,
)


def init(corpus_path, output, *, layout_path=None):
    corpus_path, output = Path(corpus_path), Path(output)
    corpus = read(corpus_path)
    cases = case_map(corpus)
    layout = layout_from_mm(read(layout_path) if layout_path is not None else
                            {'slot_mm': [26, 26], 'rows_mm': [42, 14, -14, -42], 'columns_mm': [-15, 15]})
    for case in cases.values():
        source = (corpus_path.parent / case['file']).resolve()
        if digest(source) != case['sha256']:
            raise ResearchError(f"source bytes differ for {case['id']}")
    output.mkdir(parents=True, exist_ok=False)
    (output / 'sources').mkdir()
    (output / 'trials').mkdir()
    (output / 'pages').mkdir()
    for case in cases.values():
        shutil.copyfile((corpus_path.parent / case['file']).resolve(), output / 'sources' / (case['id'] + '.png'))
        case['file'] = f"sources/{case['id']}.png"
    study = seal({'schema': 'tatbot.dbv3-study/1', 'corpus': corpus,
                  'layout': layout,
                  'promotion': {'aa_pairs': 2, 'train_pairs': 2, 'validation_pairs': 1}})
    write(output / 'study.json', study)
    write(output / 'state.json', {'baseline': None, 'next_iteration': 0})
    event(output, 'study_created', study_sha256=study['content_sha256'])
    return study


def _acquired(directory, case):
    directory = Path(directory).resolve()
    result = read(directory / 'result.json')
    if result.get('schema') != 'tatbot.dbv3-acquisition/1':
        raise ResearchError('candidate requires a completed native DBV3 acquisition')
    for name, expected in result['outputs'].items():
        path = (directory / name).resolve()
        if not path.is_relative_to(directory) or digest(path) != expected:
            raise ResearchError(f'acquisition artifact changed: {name}')
    if digest(directory/'job.json') != result['job_sha256'] or digest(directory/'recipe/recipe.json') != result['recipe_sha256']:
        raise ResearchError('acquisition job or recipe changed')
    recipe = verify_bundle(directory/'recipe')
    art, job = read(directory/'artwork.json'), read(directory/'job.json')
    validate_path_artwork(art)
    if art['source_sha256'] != case['sha256'] or job['source']['sha256'] != case['sha256'] or job['size_mm'] != case['size_mm']:
        raise ResearchError(f"acquisition does not match corpus case {case['id']}")
    if art['conversion']['recipe_sha256'] != result['recipe_sha256']:
        raise ResearchError('artwork binds a different native recipe')
    decoding = read(directory/'decoding.json')
    generation = {'software': recipe['software'], 'decoder': {k: decoding[k] for k in ('adapter', 'python', 'sources', 'chord_error_m')}}
    return art, job, generation


def candidate(root, acquisitions_path, output, *, repo, tool_id=None, speed_m_s=.0035, max_chunk_s=60):
    """One policy, acquired for named source cases; held-out test input is excluded from tuning."""
    study, output = load_study(root), Path(output)
    cases, inputs = case_map(study['corpus']), read(acquisitions_path)
    if not isinstance(inputs, dict) or set(inputs) != {k for k, c in cases.items() if c['split'] != 'test'}:
        raise ResearchError('acquisitions must cover every train and validation source case, without test sources')
    verified, policy, generation_identity = {}, None, None
    for key, directory in inputs.items():
        if key not in cases or cases[key]['split'] == 'test':
            raise ResearchError('candidate tuning accepts named train/validation cases only')
        directory = (Path(acquisitions_path).parent / directory).resolve()
        art, job, generation = _acquired(directory, cases[key])
        if generation_identity is not None and generation != generation_identity:
            raise ResearchError("all candidate cases must share the DBV3 application and decoder runtime")
        generation_identity = generation
        this_policy = {k: v for k, v in job.items() if k not in ('schema', 'id', 'source', 'size_mm')}
        if policy is not None and this_policy != policy:
            raise ResearchError('all candidate cases must use one declared recipe policy')
        policy = this_policy
        verified[key] = (directory, art)
    output.mkdir(parents=True, exist_ok=False)
    record = {'schema': 'tatbot.dbv3-candidate/1', 'study_sha256': study['content_sha256'], 'generation_identity': generation_identity, 'cases': {},
              'policy': {'recipe': policy, 'preparation': {'tool_id': tool_id, 'speed_m_s': speed_m_s, 'max_chunk_s': max_chunk_s}}}
    identities = []
    for key, (directory, art) in verified.items():
        folder = output / key
        folder.mkdir()
        for name in ('artwork.json', 'job.json', 'result.json', 'decoding.json', 'db-settings.json'):
            shutil.copyfile(directory/name, folder/name)
        shutil.copytree(directory/'recipe', folder/'recipe')
        if read(folder/'artwork.json') != art:
            raise ResearchError('acquisition changed during candidate freeze')
        reference = compile(folder/'artwork.json', repo=repo, tool_id=tool_id, speed_m_s=speed_m_s, max_segment_s=max_chunk_s)
        identity = _preparation_identity(reference)
        identities.append(identity)
        record['cases'][key] = {'artwork_sha256': art['content_sha256'], 'file_sha256': digest(folder/'artwork.json'),
                                'recipe_sha256': art['conversion']['recipe_sha256'], 'source_sha256': art['source_sha256']}
    if any(value != identities[0] for value in identities):
        raise ResearchError('preparation identity changed while freezing candidate')
    record['preparation_identity'] = identities[0]
    record['scorer'] = scorer_identity()
    record['policy']['preparation']['tool_id'] = identities[0]['resources'][0]['tool']['id']
    record['policy_sha256'] = seal(copy.deepcopy(record['policy']))['content_sha256']
    write(output/'candidate.json', seal(record))
    return record


def load_candidate(path, study):
    path = Path(path).resolve()
    value = frozen(path/'candidate.json', 'tatbot.dbv3-candidate/1')
    if value['scorer'] != scorer_identity():
        raise ResearchError('scorer changed; freeze a new candidate before preparing a trial')
    if value['study_sha256'] != study['content_sha256']:
        raise ResearchError('candidate belongs to another source corpus/study')
    for key, case in value['cases'].items():
        identifier(key)
        if digest(path/key/'recipe/recipe.json') != case['recipe_sha256']:
            raise ResearchError(f'candidate recipe changed: {key}')
        verify_bundle(path/key/'recipe')
        if digest(path/key/'artwork.json') != case['file_sha256']:
            raise ResearchError(f'candidate artwork changed: {key}')
    return value


def _single_resource(program):
    """Paired research currently freezes one configured assembly, including its unknown pigment."""
    validate_for_execution(program)
    if len(program['resources']) != 1 or program['resources'][0].get('identity_source') != 'owner_fitted':
        raise ResearchError('paired research requires one owner-fitted resource; freeze a new candidate for another assembly')
    return program['resources'][0]


def _preparation_identity(program):
    _single_resource(program)
    return {**{key: program['preparation'][key] for key in ('implementation', 'motion_sha256')},
            'resources': program['resources']}


def _prepare_side(candidate_path, candidate_doc, case, at, out, *, repo, slot_m):
    params = candidate_doc['policy']['preparation']
    source = Path(candidate_path)/case/'artwork.json'
    program = compile(source, repo=repo, tool_id=params['tool_id'], speed_m_s=params['speed_m_s'],
                      max_segment_s=params['max_chunk_s'], at_m=at)
    identity = _preparation_identity(program)
    if identity != candidate_doc['preparation_identity']:
        raise ResearchError('preparation code, runtime, tool or motion changed; freeze a new candidate')
    half = np.asarray(slot_m)/2
    size = program['design']['placements'][0]['size_m']
    if np.any(np.asarray(size)/2 > half + 1e-10):
        raise ResearchError('both acquired canvas dimensions must fit the study slot')
    physical_width = _single_resource(program)['tool']['line_width_m']
    for op in program['ops']:
        if op['op'] == 'stroke':
            radius = max(physical_width or 0, op['generation_width_m'])/2
            if np.any(np.abs(np.asarray(op['points_m'])-at)+radius > half+1e-10):
                raise ResearchError('stroke footprint leaves the study slot')
    out.mkdir()
    shutil.copyfile(source, out/'artwork.json')
    write_program(program, out/'program.json')
    write_preview(program, out/'preview.svg')
    return program


def _comparison(a, b, case, kind, factor):
    if case not in a['cases'] or case not in b['cases']:
        raise ResearchError('both candidates must contain this source case')
    changes = differences(a['policy'], b['policy'])
    if a['generation_identity'] != b['generation_identity']:
        raise ResearchError('paired candidates require the same DBV3 application and decoder runtime')
    if a['preparation_identity'] != b['preparation_identity']:
        raise ResearchError('paired candidates require the same preparation/runtime/tool/motion identity')
    if kind == 'aa':
        if changes or a['cases'][case] != b['cases'][case]:
            raise ResearchError('A/A requires the exact same acquired artwork and policy')
    elif changes != [factor] or not factor.startswith(('recipe.', 'preparation.speed_m_s', 'preparation.max_chunk_s')):
        raise ResearchError(f'paired trial must change only the declared factor; changed: {changes}')
    if factor.startswith('preparation.') and a['cases'][case] != b['cases'][case]:
        raise ResearchError('motion/preparation trials must reuse the exact acquired artwork')
    return changes


def _slot(layout, row, column):
    if type(row) is not int or not 0 <= row < len(layout['rows_m']):
        raise ResearchError('row is outside the study layout')
    return [layout['columns_m'][column], layout['rows_m'][row]]


def prepare(root, a_path, b_path, case, *, page_id, row, hypothesis, kind='aa', factor='', repo,
            max_modeled_s=600):
    """Prepare two frozen sides, then reserve both slots. Never erase an attempt to retry."""
    root, page_id = Path(root), identifier(page_id)
    if not isinstance(hypothesis, str) or not hypothesis.strip():
        raise ResearchError('record the hypothesis before preparing a trial')
    if kind not in ('aa', 'paired'):
        raise ResearchError('trial kind must be aa or paired')
    with locked(root):
        study = load_study(root)
        cases = case_map(study['corpus'])
        if case not in cases or cases[case]['split'] == 'test':
            raise ResearchError('routine research cannot use held-out test sources')
        a, b = load_candidate(a_path, study), load_candidate(b_path, study)
        changes = _comparison(a, b, case, kind, factor)
        state = read(root/'state.json')
        if kind != 'aa' and (state['baseline'] is None or state['baseline']['candidate_sha256'] != a['content_sha256']):
            raise ResearchError('A must be the qualified DBV3 baseline; qualify A/A first')
        page_path = root/'pages'/f'{page_id}.json'
        page = read(page_path) if page_path.exists() else {'id': page_id, 'confirmed': False, 'confirmation': None, 'slots': {}}
        slots = [f'{row}:{column}' for column in (0, 1)]
        if any(slot in page['slots'] for slot in slots):
            raise ResearchError('a reserved, partial, completed or unresolved slot remains occupied; choose another row/page')
        iteration = state['next_iteration']
        while (root/'trials'/f'{iteration:04d}').exists():
            iteration += 1
        trial_id = f'{iteration:04d}'
        directory = root/'trials'/trial_id
        directory.mkdir()
        event(root, 'preparation_started', trial=trial_id, page=page_id, row=row)
        try:
            record = _pair(study, directory, [a_path, b_path], [a, b], case, iteration, page_id, row, repo)
            _screen_time(record, max_modeled_s)
            record.update(hypothesis=hypothesis, kind=kind, factor=factor, changes=changes,
                          source_case=case, source_split=cases[case]['split'], source_family=cases[case]['family'])
            write(directory/'trial.json', seal(record))
            for slot in slots:
                page['slots'][slot] = {'trial': trial_id, 'status': 'reserved'}
            write(page_path, page)
            write(directory/'state.json', {'stage': 'prepared', 'sides': {s: {'status': 'prepared', 'attempts': []} for s in ('a', 'b')}})
            state['next_iteration'] = iteration + 1
            write(root/'state.json', state)
            event(root, 'prepared', trial=trial_id, trial_sha256=record['content_sha256'])
            return record
        except BaseException as error:
            event(root, 'preparation_failed', trial=trial_id, error=str(error))
            raise


def _pair(study, directory, paths, candidates, case, iteration, page_id, row, repo):
    record = {'schema': 'tatbot.dbv3-trial/1', 'study_sha256': study['content_sha256'], 'id': directory.name,
              'scorer': candidates[0]['scorer'], 'page': page_id, 'row': row, 'order': ['a', 'b'] if (iteration//2) % 2 == 0 else ['b', 'a'], 'sides': {}}
    for index, (name, path, candidate_doc) in enumerate(zip(('a', 'b'), paths, candidates, strict=True)):
        column = (iteration + index) % 2
        at = _slot(study['layout'], row, column)
        out = directory/name
        program = _prepare_side(path, candidate_doc, case, at, out, repo=repo, slot_m=study['layout']['slot_m'])
        program['research'] = {'study_sha256': study['content_sha256'], 'trial': directory.name, 'side': name,
                               'page': page_id, 'slot': f'{row}:{column}', 'slot_m': study['layout']['slot_m']}
        write_program(program, out/'program.json')
        write(out/'candidate.json', candidate_doc)
        record['sides'][name] = {'candidate_sha256': candidate_doc['content_sha256'], 'policy_sha256': candidate_doc['policy_sha256'],
                                'program_sha256': digest(out/'program.json'), 'program_content_sha256': seal(copy.deepcopy(program))['content_sha256'],
                                'artwork_file_sha256': digest(out/'artwork.json'), 'at_m': at, 'slot': f'{row}:{column}',
                                'modeled_s': program['stats']['time_estimate']['modeled_s']}
    return record


def load_trial(root, trial_id):
    root = Path(root)
    identifier(trial_id)
    study = load_study(root)
    directory = root/'trials'/trial_id
    trial = frozen(directory/'trial.json', 'tatbot.dbv3-trial/1')
    if trial['study_sha256'] != study['content_sha256']:
        raise ResearchError('trial belongs to another study')
    for name, side in trial['sides'].items():
        if name not in ('a', 'b') or digest(directory/name/'program.json') != side['program_sha256']:
            raise ResearchError('frozen trial program changed')
        if digest(directory/name/'artwork.json') != side['artwork_file_sha256']:
            raise ResearchError('frozen trial artwork changed')
        candidate_doc = frozen(directory/name/'candidate.json', 'tatbot.dbv3-candidate/1')
        if candidate_doc['content_sha256'] != side['candidate_sha256']:
            raise ResearchError('trial candidate changed')
    return trial


def _screen_time(record, limit):
    if not np.isfinite(limit) or limit <= 0:
        raise ResearchError('preparation-time budget must be finite and positive')
    if any(side['modeled_s'] > limit for side in record['sides'].values()):
        raise ResearchError('candidate exceeds the study preparation-time screening budget')
