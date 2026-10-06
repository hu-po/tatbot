"""Paired DBV3 state recovery using real preparation and an explicit offline ROS double."""
from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest
import research  # owns the offline production package path setup

REPO = research.REPO


def save(path, value):
    path.write_text(json.dumps(value, allow_nan=False))
    return path


@pytest.fixture
def study(tmp_path):
    pytest.importorskip('numpy')
    pytest.importorskip('yaml')
    from draw_research import prepare
    from draw_research.model import digest
    from drawingbot.job import SCHEMA
    from drawingbot.pens import verify_pen_readback
    from drawingbot.recipe import DEFAULTS, read_json, save_bundle
    from tatbot_contracts.artwork import freeze_artwork
    from tatbot_contracts.paths import freeze_program

    cases, acquisitions = [], {}
    for split, name in [('train', 'flower'), ('validation', 'bird'), ('test', 'animal')]:
        source = tmp_path/f'{name}.png'
        source.write_bytes(f'synthetic unit-test source {name}'.encode())
        provenance = {'kind': 'fixture', 'identifier': name, 'license': None, 'attribution': None, 'generation': None}
        case = {'id': name, 'family': name, 'split': split, 'file': source.name, 'sha256': digest(source),
                'size_mm': [20, 20], 'provenance': provenance}
        cases.append(case)
        if split == 'test':
            continue
        folder = tmp_path/f'acquire-{name}'
        folder.mkdir()
        requested = read_json(DEFAULTS / 'drawing-set.json')
        actual = {**requested, 'native_set_id': 0, 'pens': [{**requested['pens'][0], 'native_row_id': 0,
                  'name': 'black', 'export_group': 'Ballpoint_black'}]}
        effective = {'drawing': {'size_mm': [20, 20], 'pen_width_mm': .5},
                     'drawing_set': verify_pen_readback(requested, actual, .5)}
        job = {'schema': SCHEMA, 'id': name, 'source': {'file': str(source), 'sha256': case['sha256'],
               'provenance': provenance}, 'size_mm': [20, 20], 'pfm': 'fixture', 'settings': {'Random Seed': 42},
               'pen_width_mm': .5, 'drawing_set': requested, 'state': read_json(DEFAULTS / 'state.json')}
        save(folder/'job.json', job)
        project = save(folder/'project.json', {'data': {'settings': {}}})
        save_bundle(folder/'recipe', project, source, job, job['state'], effective, {'fixture': True})
        recipe_hash = digest(folder/'recipe/recipe.json')
        geometry = {'canvas_m': {'width': .02, 'height': .02}, 'inks': [{'id': 'black', 'color_srgb': [0, 0, 0]}],
                    'layers': [{'id': 'L1', 'ink_id': 'black', 'elements': [{'id': 'P1', 'kind': 'path', 'fill': False,
                      'closed': False, 'deposition': 1, 'width_m': .0005, 'points_m': [[.004, .004], [.016, .01], [.009, .016]]}]}],
                    'negative_space_masks': []}
        paths, _ = freeze_program(geometry, source_sha256=case['sha256'], name=name, adapter='dbv3-batik-paths/1')
        art = freeze_artwork(paths, name=name, source=provenance,
                             conversion={'adapter': 'dbv3-batik-paths/1', 'chord_error_m': .000005, 'recipe_sha256': recipe_hash})
        save(folder/'artwork.json', art)
        save(folder/'decoding.json', {'adapter': 'fixture', 'python': 'fixture', 'sources': {}, 'chord_error_m': .000005})
        save(folder/'db-settings.json', {})
        save(folder/'result.json', {'schema': 'tatbot.dbv3-acquisition/1', 'job_sha256': digest(folder/'job.json'),
                                   'recipe_sha256': recipe_hash, 'outputs': {n: digest(folder/n) for n in ('artwork.json', 'decoding.json')}})
        acquisitions[name] = str(folder)
    manifest = save(tmp_path/'corpus.json', {'schema': 'tatbot.dbv3-corpus/1', 'cases': cases})
    root = tmp_path/'study'
    prepare.init(manifest, root)
    inputs = save(tmp_path/'acquisitions.json', acquisitions)
    candidate = tmp_path/'candidate'
    prepare.candidate(root, inputs, candidate, repo=REPO)
    return root, candidate, inputs


def trial(study, row=0, *, page='sheet-001', **kwargs):
    from draw_research.prepare import prepare

    root, candidate, _ = study
    return prepare(root, candidate, candidate, 'flower', page_id=page, row=row, hypothesis='Estimate A/A repeatability', repo=REPO, **kwargs)


def test_default_layout_preserves_frozen_study(study):
    from draw_research import prepare
    from draw_research.model import read

    root = study[0]
    layout = save(root.parent/'layout.json', {'slot_mm': [26, 26], 'rows_mm': [42, 14, -14, -42], 'columns_mm': [-15, 15]})
    result = prepare.init(root.parent/'corpus.json', root.parent/'explicit-layout', layout_path=layout)
    assert result == read(root/'study.json')


def test_custom_layout_places_unchanged_artwork_and_limits_rows(study):
    import numpy as np
    from draw_research import prepare
    from draw_research.model import ResearchError, read

    root, _, inputs = study
    original = trial(study)
    layout = save(root.parent/'layout.json', {'slot_mm': [24, 32], 'rows_mm': [36, 0, -36], 'columns_mm': [-15, 15]})
    custom, candidate = root.parent/'custom-study', root.parent/'custom-candidate'
    result = prepare.init(root.parent/'corpus.json', custom, layout_path=layout)
    assert result['layout'] == {'slot_m': [.024, .032], 'rows_m': [.036, 0, -.036], 'columns_m': [-.015, .015]}
    prepare.candidate(custom, inputs, candidate, repo=REPO)
    for row, y in enumerate((.036, 0, -.036)):
        pair = trial((custom, candidate, inputs), row)
        for side in ('a', 'b'):
            program = read(custom/'trials'/pair['id']/side/'program.json')
            old = read(root/'trials'/original['id']/side/'program.json')
            x = (-.015, .015)[int(pair['sides'][side]['slot'].split(':')[1])]
            assert pair['sides'][side]['at_m'] == [x, y]
            assert program['research']['slot_m'] == [.024, .032]
            for before, after in zip(old['ops'], program['ops'], strict=True):
                if before['op'] == 'stroke':
                    np.testing.assert_allclose(np.asarray(before['points_m'])-original['sides'][side]['at_m'],
                                               np.asarray(after['points_m'])-[x, y], atol=1e-12)
                    assert after['generation_width_m'] == before['generation_width_m']
    with pytest.raises(ResearchError, match='row is outside'):
        trial((custom, candidate, inputs), 3)
    assert set(read(custom/'pages/sheet-001.json')['slots']) == {f'{r}:{c}' for r in range(3) for c in range(2)}


@pytest.mark.parametrize('change', [
    {'slot_mm': [0, 26]}, {'slot_mm': [-1, 26]}, {'slot_mm': [26]}, {'slot_mm': [True, 26]},
    {'rows_mm': []}, {'rows_mm': '42,14'}, {'rows_mm': [0]*1001},
    {'columns_mm': [15, -15]}, {'columns_mm': [-15, 0, 15]},
    {'rows_mm': [42, 16]}, {'rows_mm': [42, 17]}, {'columns_mm': [-13, 13]},
    {'rows_mm': [44]}, {'columns_mm': [-20, 20]}, {'slot_mm': [26, '26']}, {'units': 'mm'},
])
def test_invalid_layout_refuses_before_creating_study(study, change):
    from draw_research import prepare
    from draw_research.model import ResearchError

    root = study[0]
    value = {'slot_mm': [26, 26], 'rows_mm': [42, 14, -14, -42], 'columns_mm': [-15, 15], **change}
    layout = save(root.parent/'invalid-layout.json', value)
    output = root.parent/'invalid-study'
    with pytest.raises(ResearchError):
        prepare.init(root.parent/'corpus.json', output, layout_path=layout)
    assert not output.exists()


def test_imported_overlapping_layout_refuses_before_reserving_or_creating_trial(study):
    from draw_research.model import ResearchError, read, seal

    root = study[0]
    imported = read(root/'study.json')
    imported['layout']['rows_m'] = [.014, .014]
    save(root/'study.json', seal(imported))
    with pytest.raises(ResearchError, match='separation'):
        trial(study)
    assert not list((root/'pages').iterdir()) and not list((root/'trials').iterdir())
    assert read(root/'state.json')['next_iteration'] == 0


def test_smaller_slot_requires_new_acquisition_without_rescaling(study):
    from draw_research import prepare
    from draw_research.model import ResearchError

    root, _, inputs = study
    layout = save(root.parent/'small-layout.json', {'slot_mm': [18, 26], 'rows_mm': [0], 'columns_mm': [-15, 15]})
    custom, candidate = root.parent/'small-study', root.parent/'small-candidate'
    prepare.init(root.parent/'corpus.json', custom, layout_path=layout)
    prepare.candidate(custom, inputs, candidate, repo=REPO)
    with pytest.raises(ResearchError, match='canvas dimensions'):
        trial((custom, candidate, inputs))
    assert not list((custom/'pages').iterdir())


def test_family_split_leakage_and_corrupt_recipe_are_refused(study):
    from draw_research.model import ResearchError, case_map, read
    from draw_research.prepare import load_candidate

    root, candidate, _ = study
    source = read(root/'study.json')
    corrupt = copy.deepcopy(source['corpus'])
    corrupt['cases'][1]['family'] = corrupt['cases'][0]['family']
    with pytest.raises(ResearchError, match='one split'):
        case_map(corrupt)
    (candidate/'flower/recipe/source.png').write_bytes(b'corrupt source')
    with pytest.raises(ValueError, match='input differs'):
        load_candidate(candidate, source)


def test_prepare_both_before_motion_counterbalance_and_preserve_failed_attempt(study):
    from draw_research.model import ResearchError, read
    from draw_research.prepare import load_trial

    root, _, _ = study
    for i in range(4):
        result = trial(study, i)
        assert result['sides']['a']['slot'] == f'{i}:{i%2}'
        assert result['order'][0] == ('a' if i < 2 else 'b')
        assert load_trial(root, result['id']) == result
        assert all((root/'trials'/result['id']/side/'preview.svg').exists() for side in ('a', 'b'))
    with pytest.raises(ResearchError, match='occupied'):
        trial(study)
    with pytest.raises(ResearchError, match='budget'):
        trial(study, page='sheet-002', max_modeled_s=.01)
    assert (root/'trials/0004/a/program.json').exists()
    assert trial(study, page='sheet-002')['id'] == '0005'
    assert read(root/'pages/sheet-001.json')['confirmed'] is False


def test_declared_motion_factor_reuses_artwork_and_requires_baseline(study):
    from draw_research import prepare
    from draw_research.model import ResearchError, read, write

    root, a, inputs = study
    b = root.parent/'speed-candidate'
    record = prepare.candidate(root, inputs, b, repo=REPO, speed_m_s=.004)
    old = read(a/'candidate.json')
    assert old['cases'] == record['cases']
    kwargs = {'page_id': 'sheet-001', 'row': 0, 'hypothesis': 'Faster traversal reduces time', 'kind': 'paired', 'repo': REPO}
    with pytest.raises(ResearchError, match='declared factor'):
        prepare.prepare(root, a, b, 'flower', factor='recipe.seed', **kwargs)
    with pytest.raises(ResearchError, match='qualify'):
        prepare.prepare(root, a, b, 'flower', factor='preparation.speed_m_s', **kwargs)
    write(root/'state.json', {'baseline': {'candidate_sha256': old['content_sha256']}, 'next_iteration': 0})
    result = prepare.prepare(root, a, b, 'flower', factor='preparation.speed_m_s', **kwargs)
    assert result['changes'] == ['preparation.speed_m_s']
    with pytest.raises(ResearchError, match='held-out'):
        prepare.prepare(root, a, b, 'animal', factor='preparation.speed_m_s', **kwargs)


class FakeRos:
    """No controller, network or robot. Receipts mirror existing ROS run artifacts."""
    def __init__(self):
        self.receipts = {}
        self.calls = []
        self.runtime = {'schema': 'tatbot.ros-runtime/2', 'sources_sha256': 'a'*64, 'configuration_sha256': 'c'*64, 'python': '3.12', 'numpy': '2.0',
                        'pid': 100, 'process_start': '1', 'boot_id': 'fixture',
                        'controller': {'complete': True, 'files_sha256': 'a'*64,
                                       'files': [{'path': '/test/controller', 'sha256': 'a'*64, 'inode': 1}]}}
        self.ready_calls = []
        self.holds = []
        self.is_landed = True
        self.fail_landing = False
        self.hardware = 'real'  # synthetic receipt; this test double cannot contact hardware
        self.crash_after = None
        self.outcome = 'done'

    def ready(self, log):
        Path(log).write_text('offline startup fixture\n')
        self.ready_calls.append(str(log))
        self.is_landed = False
        if self.runtime:
            self.runtime = {**self.runtime, 'pid': self.runtime.get('pid', 0)+1}
        return 0

    def landed(self):
        return self.is_landed

    def page_measured(self):
        self.page_polls = getattr(self, 'page_polls', 0) + 1
        return self.page_polls > getattr(self, 'page_lost_polls', 0)

    def evidence(self, page, slot):
        receipt = copy.deepcopy(self.receipts.get((page, slot), {'schema': 'tatbot.research-evidence/1', 'claim': None}))
        receipt.update(runtime=copy.deepcopy(self.runtime), runtime_live=True)
        return receipt

    def draw(self, program, log, *, resume=None, hold=False):
        from draw_research.model import read
        from tatbot_contracts.canonical import canonical_digest
        from tatbot_motion import motion_path
        from tatbot_session.fidelity import identity

        p = read(program)
        self.calls.append((str(program), resume))
        self.holds.append(hold)
        slot = p['research']
        Path(log).write_text('offline test double\n')
        self.receipts[slot['page'], slot['slot']] = {
            'schema': 'tatbot.research-evidence/1', 'runtime': self.runtime, 'runtime_live': True,
            'claim': {'program_sha256': canonical_digest(p), 'run_id': resume or f'fixture-run-{len(self.calls)}'},
            'program': p, 'motion_yaml': motion_path().read_text(),
            'timing': {'schema': 'tatbot.draw-timing/1', 'complete': True, 'duration_s': 12.5, 'attempts': [], 'reason': None},
            'meta': {'status': 'ok' if self.outcome == 'done' else 'fail', 'runtime': self.runtime, 'duration_s': 12.5,
                     'hardware': self.hardware, 'estop_source': 'none', 'page_source': 'fixed', 'touch': True,
                     'program': {'resources': copy.deepcopy(p['resources'])},
                     'arms': {'right': {'tool': p['resources'][0]['tool']['id'], 'trim': [0, 0], 'registration_sha256': 'fixture'}}},
            'ledger': [{'arm': 'right', 'op': o['id'], 'event': self.outcome} for o in p['ops']],
            'inspection': {'directory': 'fixture-inspection', 'files': {}, 'analysis': {
                'fidelity': {'scorer': identity(), 'metrics': {'missing_fraction': .1, 'spill_fraction': .1,
                                                             'spill_area_mm2': .1, 'iou': .8}}}}}
        if self.outcome == 'done' and not hold and not self.fail_landing:
            self.is_landed = True
        if len(self.calls) == self.crash_after:
            raise KeyboardInterrupt('client vanished after the arm finished')
        return 0


def confirmed(study):
    from draw_research.run import confirm_page

    result = trial(study)
    confirm_page(study[0], result['page'], observation='Synthetic unit test page placement')
    return result


def test_lost_client_recovers_completed_side_without_redraw(study):
    from draw_research import run
    from draw_research.model import read

    root = study[0]
    result = confirmed(study)
    ros = FakeRos()
    ros.crash_after = 1
    with pytest.raises(KeyboardInterrupt):
        run.run(root, result['id'], ros=ros)
    assert read(root/'trials/0000/state.json')['sides']['a']['status'] == 'dispatching'
    final = run.run(root, result['id'], ros=ros)
    assert final['stage'] == 'drawn' and len(ros.calls) == 2
    run.run(root, result['id'], ros=ros)
    assert len(ros.calls) == 2


def test_a_signal_that_stops_the_runner_is_named_in_the_journal(study):
    """The runner turns SIGTERM into a bare KeyboardInterrupt; the pause event said error '' (trial 0004)."""
    from draw_research import run

    result, ros = confirmed(study), FakeRos()

    def cancelled(*args, **kwargs):
        raise KeyboardInterrupt

    ros.draw = cancelled
    with pytest.raises(KeyboardInterrupt):
        run.run(study[0], result['id'], ros=ros)
    paused = [json.loads(line) for line in (study[0]/'events.jsonl').read_text().splitlines()][-1]
    assert paused['event'] == 'execution_paused' and paused['error'] == 'KeyboardInterrupt'


def test_each_side_waits_for_the_page_outside_its_draw_attempt(study, monkeypatch):
    """The second side of trial 0006 counted 230 s of page waiting as drawing time; the runner now waits
    between attempts, before the draw, and journals how long."""
    from draw_research import run

    result, ros = confirmed(study), FakeRos()
    ros.page_lost_polls = 2
    waits = []
    monkeypatch.setattr(run.time, 'sleep', waits.append)
    draws = []
    real_draw = ros.draw
    ros.draw = lambda *args, **kwargs: draws.append(ros.page_polls) or real_draw(*args, **kwargs)
    assert run.run(study[0], result['id'], ros=ros)['stage'] == 'drawn'
    assert draws[0] == 3 and len(waits) == 2          # dispatched only once the page was measured
    awaited = [json.loads(line) for line in (study[0]/'events.jsonl').read_text().splitlines()
               if '"page_awaited"' in line]
    assert [row['side'] for row in awaited] == result['order']


def test_tool_selection_cannot_override_frozen_pair(study):
    from draw_research import run
    from draw_research.model import ResearchError, read

    result, ros = confirmed(study), FakeRos()
    with pytest.raises(ResearchError, match='selected tool differs'):
        run.run(study[0], result['id'], ros=ros, tool_id='another-tool')
    assert not ros.calls and not ros.receipts
    chosen = read(study[0]/'trials'/result['id']/'a/program.json')['resources'][0]['tool']['id']
    assert run.run(study[0], result['id'], ros=ros, tool_id=chosen)['stage'] == 'drawn'


@pytest.mark.parametrize('missing', [True, False])
def test_execution_must_preserve_frozen_resource_constraints(study, missing):
    from draw_research import run
    from draw_research.model import ResearchError

    result, ros = confirmed(study), FakeRos()
    draw = ros.draw
    def changed(*args, **kwargs):
        code = draw(*args, **kwargs)
        for receipt in ros.receipts.values():
            if missing:
                receipt['meta'].pop('program')
            else:
                receipt['meta']['program']['resources'][0]['tool']['datasheet_sha256'] = 'a'*64
        return code
    ros.draw = changed
    with pytest.raises(ResearchError, match='resource constraints'):
        run.run(study[0], result['id'], ros=ros)
    assert len(ros.calls) == 1


@pytest.mark.parametrize('outcome,message', [('sent', 'uncertain'), ('skipped', 'skipped'), ('aborted', 'interrupted')])
def test_partial_or_unresolved_slot_is_not_redrawn(study, outcome, message):
    from draw_research import run
    from draw_research.model import ResearchError

    result, ros = confirmed(study), FakeRos()
    ros.outcome = outcome
    with pytest.raises(ResearchError, match=message):
        run.run(study[0], result['id'], ros=ros)
    with pytest.raises(ResearchError, match=message):
        run.run(study[0], result['id'], ros=ros)
    assert len(ros.calls) == 1
    if outcome == 'aborted':
        ros.outcome = 'done'
        run.run(study[0], result['id'], ros=ros, resume=True)
        assert ros.calls[1][1] == 'fixture-run-1'
        assert len(ros.calls) == 3


def test_unconfirmed_page_and_changed_runtime_refuse_before_next_side(study):
    from draw_research import run
    from draw_research.model import ResearchError

    result, ros = trial(study), FakeRos()
    with pytest.raises(ResearchError, match='fresh physical'):
        run.run(study[0], result['id'], ros=ros)
    assert not ros.calls
    run.confirm_page(study[0], result['page'], observation='Synthetic fresh sheet observation')
    ros.crash_after = 1
    with pytest.raises(KeyboardInterrupt):
        run.run(study[0], result['id'], ros=ros)
    # A process restart cannot silently become B of an unfinished pair.
    ros.runtime = {**ros.runtime, 'pid': ros.runtime['pid']+1}
    ros.receipts[('sheet-001', '0:0')]['runtime'] = ros.runtime
    with pytest.raises(ResearchError, match='runtime changed'):
        run.run(study[0], result['id'], ros=ros)
    assert len(ros.calls) == 1


@pytest.mark.parametrize('identity', [None, {'schema': 'tatbot.ros-runtime/1'},
                                        {'schema': 'tatbot.ros-runtime/2', 'controller': {'complete': False}}])
def test_unidentified_runtime_refuses_before_dispatch(study, identity):
    from draw_research import run
    from draw_research.model import ResearchError

    result, ros = confirmed(study), FakeRos()
    ros.runtime = identity
    with pytest.raises(ResearchError, match='runtime is required|identified loaded'):
        run.run(study[0], result['id'], ros=ros)
    assert not ros.calls


def test_score_qualification_is_repeatable_and_does_not_regress_state(study):
    from draw_research import assess, run
    from draw_research.model import read

    root, ros = study[0], FakeRos()
    first = confirmed(study)
    second = trial(study, 1)
    for value in (first, second):
        run.run(root, value['id'], ros=ros)
        assert assess.score(root, value['id'])['human_preference'] is None
    result = assess.decide(root, ['0000', '0001'], decision='qualify', reason='Synthetic repeated A/A evidence')
    assert read(root/'state.json')['baseline'] == result['baseline']
    assert assess.decide(root, ['0000', '0001'], decision='qualify', reason='Synthetic repeated A/A evidence') == result
    assess.score(root, '0000')
    assert read(root/'trials/0000/state.json')['stage'] == 'decided'


def test_promotion_requires_noise_margin_validation_and_no_smear_tradeoff():
    from draw_research.assess import _promote
    from draw_research.model import ResearchError

    def pair(kind, split, gain=0, spill=.1):
        trial = {'id': 'fixture', 'kind': kind, 'source_split': split,
                 'sides': {'a': {'candidate_sha256': 'A'}, 'b': {'candidate_sha256': 'A' if kind == 'aa' else 'B', 'policy_sha256': 'policy'}}}
        a = {'fidelity': {'metrics': {'missing_fraction': .2, 'spill_fraction': .1, 'iou': .6}}, 'draw_session_s': 10}
        b = {'fidelity': {'metrics': {'missing_fraction': .2-gain, 'spill_fraction': spill, 'iou': .6+gain}}, 'draw_session_s': 8}
        return trial, {'sides': {'a': a, 'b': b}}
    policy = {'aa_pairs': 2, 'train_pairs': 2, 'validation_pairs': 1}
    state = {'baseline': {'candidate_sha256': 'A'}}
    evidence = [pair('aa', 'train', .01), pair('aa', 'train', -.01)] + [pair('paired', split, .1) for split in ('train', 'train', 'validation')]
    assert _promote(evidence, state, policy, 'missing_fraction', 0, 1.25)['candidate_sha256'] == 'B'
    with pytest.raises(ResearchError, match='held-out'):
        _promote(evidence[:-1], state, policy, 'missing_fraction', 0, 1.25)
    evidence[-1] = pair('paired', 'validation', .005)
    with pytest.raises(ResearchError, match='variation'):
        _promote(evidence, state, policy, 'missing_fraction', 0, 1.25)
    evidence[-1] = pair('paired', 'validation', .1, spill=.5)
    with pytest.raises(ResearchError, match='fidelity loss'):
        _promote(evidence, state, policy, 'missing_fraction', 0, 1.25)


def test_failed_trial_can_be_recorded_inconclusive_without_score(study):
    from draw_research import assess
    from draw_research.model import read

    trial(study)
    result = assess.decide(study[0], ['0000'], decision='inconclusive', reason='Runtime changed before execution')
    assert result['baseline'] is None
    assert result['request']['trials'][0]['score_sha256'] is None
    assert read(study[0]/'trials/0000/state.json')['stage'] == 'decided'


def test_resumed_run_does_not_misreport_last_attempt_as_total_time(study):
    from draw_research import assess, run
    from draw_research.model import read, write

    result, ros = confirmed(study), FakeRos()
    run.run(study[0], result['id'], ros=ros)
    path = study[0]/'trials/0000/a/evidence.json'
    receipt = read(path)
    receipt['meta']['resumed_at'] = 123
    receipt.pop('timing')
    write(path, receipt)
    scored = assess.score(study[0], '0000')
    assert scored['sides']['a']['draw_session_s'] is None


def test_resumed_run_uses_measured_total_across_attempts(study):
    from draw_research import assess, run
    from draw_research.model import read, write

    result, ros = confirmed(study), FakeRos()
    run.run(study[0], result['id'], ros=ros)
    path = study[0]/'trials/0000/a/evidence.json'
    receipt = read(path)
    receipt['meta']['resumed_at'] = 123
    receipt['timing'].update(duration_s=20., attempts=[{'id': 'first', 'duration_s': 7.5}, {'id': 'resume', 'duration_s': 12.5}])
    write(path, receipt)
    assert assess.score(study[0], '0000')['sides']['a']['draw_session_s'] == 20.


def test_missing_attempt_timing_cannot_qualify_or_promote_but_remains_inconclusive(study):
    from draw_research import assess, run
    from draw_research.model import ResearchError, read, write

    root, ros = study[0], FakeRos()
    confirmed(study)
    trial(study, 1)
    for value in ('0000', '0001'):
        run.run(root, value, ros=ros)
    path = root/'trials/0001/b/evidence.json'
    receipt = read(path)
    receipt.pop('timing')
    # Neither last-attempt wall time nor an uninstrumented metadata field is measurement evidence.
    receipt['meta'].update(duration_s=99, pen_down_s=55)
    write(path, receipt)
    assess.score(root, '0000')
    scored = assess.score(root, '0001')
    assert scored['sides']['b']['draw_session_s'] is None
    assert scored['sides']['b']['pen_down_s'] is None
    for decision in ('qualify', 'promote'):
        with pytest.raises(ResearchError, match='trial 0001 side b needs a positive measured'):
            assess.decide(root, ['0000', '0001'], decision=decision, reason='Synthetic timing gap')
        assert read(root/'state.json')['baseline'] is None
        assert read(root/'trials/0001/state.json')['stage'] == 'scored'
    assert not list((root/'decisions').glob('*.json'))
    result = assess.decide(root, ['0000', '0001'], decision='inconclusive', reason='Retain the missing timing evidence')
    assert result['baseline'] is None
    assert read(root/'trials/0001/score.json') == scored


@pytest.mark.parametrize('seconds', [None, 0, -1, True, float('nan'), float('inf'), '12.5'])
def test_baseline_timing_requires_a_positive_finite_number_for_every_side(seconds):
    from draw_research.assess import _draw_seconds, _require_measured_times
    from draw_research.model import ResearchError

    evidence = [({'id': 'late-aa'}, {'sides': {'a': {'draw_session_s': 12.5}, 'b': {'draw_session_s': seconds}}})]
    with pytest.raises(ResearchError, match='late-aa side b'):
        _require_measured_times(evidence)
    assert _draw_seconds({'timing': {'schema': 'tatbot.draw-timing/1', 'complete': True, 'duration_s': seconds}}) is None


def test_promotion_accepts_time_gain_with_quality_preserved():
    from draw_research.assess import _promote

    evidence = []
    for index, (kind, split) in enumerate([('aa', 'train'), ('aa', 'train'), ('paired', 'train'), ('paired', 'train'), ('paired', 'validation')]):
        trial = {'id': str(index), 'kind': kind, 'source_split': split,
                 'sides': {'a': {'candidate_sha256': 'A'}, 'b': {'candidate_sha256': 'A' if kind == 'aa' else 'B', 'policy_sha256': 'speed'}}}
        a = {'draw_session_s': 10., 'fidelity': {'metrics': {'missing_fraction': .1, 'spill_fraction': .1}}}
        b = {'draw_session_s': 10.1 if kind == 'aa' else 8., 'fidelity': {'metrics': {'missing_fraction': .1, 'spill_fraction': .1}}}
        evidence.append((trial, {'sides': {'a': a, 'b': b}}))
    result = _promote(evidence, {'baseline': {'candidate_sha256': 'A'}},
                      {'aa_pairs': 2, 'train_pairs': 2, 'validation_pairs': 1}, 'draw_session_s', 0., 1.)
    assert result['candidate_sha256'] == 'B'


def test_incomplete_decision_pointer_update_is_repaired(study, monkeypatch):
    from draw_research import assess, run
    from draw_research.model import read

    root, ros = study[0], FakeRos()
    confirmed(study)
    trial(study, 1)
    for value in ('0000', '0001'):
        run.run(root, value, ros=ros)
        assess.score(root, value)
    original = assess.write
    def fail_pointer(path, value):
        if path == root/'state.json':
            raise OSError('power loss at baseline pointer')
        original(path, value)
    monkeypatch.setattr(assess, 'write', fail_pointer)
    with pytest.raises(OSError, match='power loss'):
        assess.decide(root, ['0000', '0001'], decision='qualify', reason='Synthetic evidence')
    assert len(list((root/'decisions').glob('*.json'))) == 1 and read(root/'state.json')['baseline'] is None
    monkeypatch.setattr(assess, 'write', original)
    result = assess.decide(root, ['0000', '0001'], decision='qualify', reason='Synthetic evidence')
    assert read(root/'state.json')['baseline'] == result['baseline']


def test_mock_receipts_cannot_qualify_a_physical_baseline(study):
    from draw_research import assess, run
    from draw_research.model import ResearchError

    root, ros = study[0], FakeRos()
    ros.hardware = 'mock'
    confirmed(study)
    trial(study, 1)
    for value in ('0000', '0001'):
        run.run(root, value, ros=ros)
        assess.score(root, value)
    with pytest.raises(ResearchError, match='real-hardware ink'):
        assess.decide(root, ['0000', '0001'], decision='qualify', reason='Simulation is not physical qualification')


def review_bundle(study, *, split='train'):
    from draw_research import review
    from draw_research.model import read

    root, _, inputs = study
    mapping = read(inputs)
    selected = {'flower': mapping['flower']} if split == 'train' else {'bird': mapping['bird']}
    source = save(root.parent/f'{split}-review-inputs.json', selected)
    result = review.build(root, [source], root.parent/f'{split}-review', repo=REPO, split=split)
    return read(result['review'])


def test_review_training_subset_uses_frozen_native_art_and_production_cost(study):
    from draw_research.model import read
    from tatbot_contracts.canonical import canonical_digest

    bundle = review_bundle(study)
    assert len(bundle['entries']) == 1
    entry = bundle['entries'][0]
    assert entry['source_case'] == 'flower' and entry['source_split'] == 'train'
    program = read(study[0].parent/'train-review/programs'/f"{entry['id']}.json")
    assert entry['preparation']['program_sha256'] == canonical_digest(program)
    assert entry['preparation']['stats'] == program['stats']
    assert 'source' not in entry['recipe']
    assert set(bundle['artworks']) == {entry['artwork_sha256']}
    assert program['design']['placements'][0]['artwork_sha256'] == entry['artwork_sha256']
    assert read(study[0]/'state.json')['baseline'] is None
    assert read(study[0]/'reviews'/f"{bundle['content_sha256']}.json") == bundle


def test_review_split_admission_and_corrupt_acquisition(study):
    from draw_research import review
    from draw_research.model import ResearchError, read

    root, _, inputs = study
    with pytest.raises(ResearchError, match='only train'):
        review.build(root, [inputs], root.parent/'mixed-review', repo=REPO)
    bundle = review_bundle(study, split='validation')
    assert bundle['entries'][0]['source_case'] == 'bird'
    with pytest.raises(ResearchError, match='train or validation'):
        review.build(root, [inputs], root.parent/'test-review', repo=REPO, split='test')
    mapping = read(inputs)
    Path(mapping['flower'], 'artwork.json').write_text('{}')
    train = save(root.parent/'train.json', {'flower': mapping['flower']})
    with pytest.raises(ResearchError, match='artifact changed'):
        review.build(root, [train], root.parent/'corrupt-review', repo=REPO)


def review_feedback(bundle):
    from draw_research.model import seal

    entry = bundle['entries'][0]
    return seal({'schema': 'tatbot.artwork-feedback/1', 'review_sha256': bundle['content_sha256'],
                 'study_sha256': bundle['study_sha256'], 'observations': [{
                     'id': 'human-observation-fixture', 'entry_id': entry['id'], 'artwork_sha256': entry['artwork_sha256'],
                     'program_sha256': entry['preparation']['program_sha256'], 'preview_sha256': 'a'*64,
                     'renderer': 'inkmap-metric-svg/1', 'preference': 'like', 'observed_at': '2026-09-28T00:00:00Z'}]})


def test_feedback_identity_idempotency_and_conflicting_history(study):
    from draw_research import review
    from draw_research.model import ResearchError, read, seal

    bundle = review_bundle(study)
    feedback = review_feedback(bundle)
    path = save(study[0].parent/'feedback.json', feedback)
    assert review.ingest(study[0], path)['already_imported'] is False
    assert review.ingest(study[0], path)['already_imported'] is True
    assert read(study[0]/'state.json')['baseline'] is None
    feedback['observations'][0]['preference'] = 'dislike'
    save(path, seal(feedback))
    with pytest.raises(ResearchError, match='conflicts'):
        review.ingest(study[0], path)


@pytest.mark.parametrize('field,value', [('artwork_sha256', 'b'*64), ('program_sha256', 'b'*64),
                                        ('preview_sha256', 'bad'), ('renderer', 'unrecognized'),
                                        ('preference', 'promote'), ('observed_at', 'yesterday')])
def test_feedback_rejects_mismatched_and_invalid_observations(study, field, value):
    from draw_research import review
    from draw_research.model import ResearchError, seal

    bundle = review_bundle(study)
    feedback = review_feedback(bundle)
    feedback['observations'][0][field] = value
    path = save(study[0].parent/'bad-feedback.json', seal(feedback))
    with pytest.raises(ResearchError):
        review.ingest(study[0], path)
    assert not list((study[0]/'feedback').glob('*.json'))


def test_inkmap_feedback_roundtrip_uses_python_native_review(study):
    import shutil
    import subprocess

    from draw_research import review
    from draw_research.model import read

    node = shutil.which('node')
    if node is None:
        pytest.skip('Node is unavailable for the shared browser contract')
    bundle = review_bundle(study)
    path = study[0]/'reviews'/f"{bundle['content_sha256']}.json"
    module = (REPO/'web/inkmap/src/core/study-review.ts').as_uri()
    code = f'''import {{ readFileSync }} from 'node:fs';
import {{ loadReview, observe, feedbackDocument }} from {json.dumps(module)};
const review = await loadReview(readFileSync(process.argv[1], 'utf8'));
const observation = observe(review, review.bundle.entries[0], 'like');
process.stdout.write(JSON.stringify(await feedbackDocument(review, [observation])));
'''
    result = subprocess.run([node, '--experimental-strip-types', '--input-type=module', '-e', code, str(path)],
                            capture_output=True, text=True, timeout=30, check=True)
    feedback = save(study[0].parent/'browser-feedback.json', json.loads(result.stdout))
    receipt = review.ingest(study[0], feedback)
    assert receipt['observations'] == 1
    assert read(receipt['feedback'])['observations'][0]['preference'] == 'like'


def test_review_limit_applies_to_the_file_the_browser_opens(study, monkeypatch):
    from draw_research import review
    from draw_research.model import ResearchError

    bundle = review_bundle(study)
    compact_bytes = len(json.dumps(bundle).encode())
    on_disk_bytes = (study[0].parent/'train-review/review.json').stat().st_size
    assert on_disk_bytes > compact_bytes
    monkeypatch.setattr(review, 'MAX_BYTES', (compact_bytes+on_disk_bytes)//2)
    output = study[0].parent/'oversize-review'
    with pytest.raises(ResearchError, match='browser import limit'):
        review.build(study[0], [study[0].parent/'train-review-inputs.json'], output, repo=REPO)
    assert not (output/'review.json').exists()


def test_pair_lifecycle_wakes_once_holds_first_and_lands_last_in_either_order(study):
    from draw_research import assess, run
    from draw_research.model import read

    root, ros = study[0], FakeRos()
    confirmed(study)
    trial(study, 1)
    second = trial(study, 2)
    assert second['order'] == ['b', 'a']
    for trial_id in ('0000', '0002'):
        assert run.run(root, trial_id, ros=ros)['stage'] == 'drawn'
        assert ros.is_landed
        assess.score(root, trial_id)
    assert ros.holds == [True, False, True, False] and len(ros.ready_calls) == 2
    a, b = (read(root/f'trials/{trial_id}/state.json') for trial_id in ('0000', '0002'))
    assert a['runtime']['pid'] != b['runtime']['pid']
    # Different completed operating windows with identical software are comparable.
    assess.decide(root, ['0000', '0002'], decision='qualify', reason='Synthetic completed pair restarts')


@pytest.mark.parametrize('change', ['native-code', 'launch-configuration'])
def test_loaded_code_change_between_completed_pairs_cannot_qualify(study, change):
    from draw_research import assess, run
    from draw_research.model import ResearchError

    root, ros = study[0], FakeRos()
    confirmed(study)
    trial(study, 1)
    run.run(root, '0000', ros=ros)
    assess.score(root, '0000')
    ros.runtime = copy.deepcopy(ros.runtime)
    if change == 'native-code':
        ros.runtime['controller']['files'][0]['sha256'] = 'b'*64
    else:
        ros.runtime['configuration_sha256'] = 'b'*64
    run.run(root, '0001', ros=ros)
    assess.score(root, '0001')
    with pytest.raises(ResearchError, match='consistent deployed runtime'):
        assess.decide(root, ['0000', '0001'], decision='qualify', reason='Changed native code must not qualify')


def test_lost_last_client_and_landing_failure_do_not_redraw(study):
    from draw_research import run
    from draw_research.model import ResearchError

    root, ros = study[0], FakeRos()
    confirmed(study)
    ros.crash_after, ros.fail_landing = 2, True
    with pytest.raises(KeyboardInterrupt):
        run.run(root, '0000', ros=ros)
    with pytest.raises(ResearchError, match='final landing is unverified'):
        run.run(root, '0000', ros=ros)
    assert len(ros.calls) == 2 and len(ros.ready_calls) == 1
    ros.is_landed = True  # independent ROS landing reconciliation, no drawing retry
    assert run.run(root, '0000', ros=ros)['stage'] == 'drawn'
    ros.runtime = None  # completed state stays readable after the operating window closes
    assert run.run(root, '0000', ros=ros)['stage'] == 'drawn'
    assert len(ros.calls) == 2


def test_ready_failure_preserves_log_and_does_not_dispatch(study):
    from draw_research import run
    from draw_research.model import ResearchError, read

    class FailsOnce(FakeRos):
        def ready(self, log):
            super().ready(log)
            return int(len(self.ready_calls) == 1)

    root, ros = study[0], FailsOnce()
    confirmed(study)
    with pytest.raises(ResearchError, match='startup did not complete'):
        run.run(root, '0000', ros=ros)
    assert not ros.calls
    assert read(root/'trials/0000/state.json')['ready_attempts'][0]['exit_code'] == 1
    assert run.run(root, '0000', ros=ros)['stage'] == 'drawn'
    assert len(ros.ready_calls) == 2 and (root/'trials/0000/ready-000.log').is_file()


def test_research_ros_draw_keeps_existing_inspection_and_disables_implicit_wake(monkeypatch, tmp_path):
    from draw_research.run import Ros

    ros, calls = Ros(REPO), []
    monkeypatch.setattr(ros, '_execute', lambda args, log: calls.append(args) or 0)
    ros.ready(tmp_path/'ready.log')
    ros.draw(tmp_path/'program.json', tmp_path/'a.log', hold=True)
    ros.draw(tmp_path/'program.json', tmp_path/'b.log', resume='same-run')
    assert calls[0] == ['ready', '--arm', 'right']
    assert '--hold' in calls[1] and '--hold' not in calls[2]
    assert all('--no-wake' in c and '--no-inspect' not in c for c in calls[1:])
    assert calls[2][-2:] == ['--resume', 'same-run']


def test_research_ros_status_reads_a_page_pose_holding_negative_zero(monkeypatch):
    """Status is live telemetry; a -0.0 in the page pose stopped a real pair after its first side."""
    import subprocess

    from draw_research.run import Ros

    status = ('{"page": {"summary": {"measured": 2, "last": {"base_from_page": [[-0.0, 0.99, 0.0, 0.28]]}}},'
              ' "stack": {"safety": {"right": {"landed": true}}}}')
    monkeypatch.setattr(subprocess, 'run', lambda *a, **k: subprocess.CompletedProcess(a, 0, stdout=status, stderr=''))
    ros = Ros(REPO)
    assert ros.page_measured() and ros.landed()


def test_research_ros_command_runs_the_real_cli_launcher():
    """The launcher is a bash shim; handing it to a Python interpreter failed every real run."""
    import subprocess

    from draw_research.run import Ros

    result = subprocess.run([*Ros(REPO).command, 'ros', 'evidence', '--help'], capture_output=True, text=True,
                            timeout=30, stdin=subprocess.DEVNULL)
    assert result.returncode == 0, result.stderr
    assert '--slot' in result.stdout


def test_occupied_second_slot_is_checked_before_startup_or_first_draw(study):
    from draw_research import run
    from draw_research.model import ResearchError

    result, ros = confirmed(study), FakeRos()
    ros.receipts[result['page'], result['sides']['b']['slot']] = {
        'schema': 'tatbot.research-evidence/1', 'claim': {'program_sha256': 'different-program'}}
    with pytest.raises(ResearchError, match='different program'):
        run.run(study[0], result['id'], ros=ros)
    assert not ros.ready_calls and not ros.calls
