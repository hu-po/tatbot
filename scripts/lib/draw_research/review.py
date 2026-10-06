"""Portable Inkmap review of native acquisitions; preferences never dispatch motion."""
from __future__ import annotations

import hashlib
import json
import re
from datetime import datetime
from pathlib import Path

from tatbot_contracts.canonical import canonical_digest
from tatbot_ink import compile, write_program

from draw_research.model import ResearchError, case_map, event, frozen, load_study, locked, read, seal, write
from draw_research.prepare import _acquired, _single_resource

REVIEW = 'tatbot.artwork-review/1'
FEEDBACK = 'tatbot.artwork-feedback/1'
MAX_BYTES = 20 * 1024 * 1024


def _entry(directory, case, output, *, repo, tool_id, speed_m_s, max_chunk_s):
    art, job, _ = _acquired(directory, case)
    source = output/'artworks'/f"{art['content_sha256']}.json"
    source.parent.mkdir(exist_ok=True)
    write(source, art)
    program = compile(source, repo=repo, tool_id=tool_id,
                      speed_m_s=speed_m_s, max_segment_s=max_chunk_s, at_m=(0, 0))
    program_hash = canonical_digest(program)
    key = f"{case['id']}-{art['content_sha256'][:12]}-{program_hash[:12]}"
    write_program(program, output/'programs'/f'{key}.json')
    recipe = {k: v for k, v in job.items() if k not in ('schema', 'id', 'source')}
    preparation = {'program_sha256': program_hash, 'tool': _single_resource(program)['tool'],
                   'speed_m_s': program['draw_speed_m_s'], 'stats': program['stats'],
                   'identity': program['preparation']}
    return art, {'id': key, 'source_case': case['id'], 'source_split': case['split'],
                 'artwork_sha256': art['content_sha256'], 'recipe': recipe, 'preparation': preparation}


def _sources(paths, cases, split):
    selected = []
    for path in paths:
        path = Path(path).resolve()
        inputs = read(path)
        if not isinstance(inputs, dict) or not inputs:
            raise ResearchError('review requires nonempty case-to-acquisition maps')
        for key, directory in inputs.items():
            if key not in cases or cases[key]['split'] != split:
                raise ResearchError(f'review accepts only {split} cases; refused {key}')
            selected.append((cases[key], (path.parent/directory).resolve()))
    if not 1 <= len(selected) <= 64:
        raise ResearchError('review requires 1..64 acquisitions')
    return selected


def build(root, acquisitions, output, *, repo, title='DBV3 study review', split='train', tool_id=None,
          speed_m_s=.0035, max_chunk_s=60):
    """Acquisition maps may cover a training subset; validation is explicitly opt-in."""
    if split not in ('train', 'validation') or not isinstance(title, str) or not 1 <= len(title.strip()) <= 200:
        raise ResearchError('review requires a title and a train or validation split')
    root, output = Path(root), Path(output)
    study = load_study(root)
    cases = case_map(study['corpus'])
    selected = _sources(acquisitions, cases, split)
    output.mkdir(parents=True, exist_ok=False)
    entries, artworks = {}, {}
    for case, directory in selected:
        art, entry = _entry(directory, case, output, repo=repo, tool_id=tool_id,
                            speed_m_s=speed_m_s, max_chunk_s=max_chunk_s)
        if entry['id'] in entries:
            raise ResearchError('duplicate review entry; each acquired preparation appears once')
        entries[entry['id']] = entry
        artworks[art['content_sha256']] = art
    bundle = seal({'schema': REVIEW, 'name': title.strip(), 'study_sha256': study['content_sha256'],
                   'artworks': artworks, 'entries': list(entries.values())})
    if len(json.dumps(bundle, indent=2, sort_keys=True, allow_nan=False).encode()) + 1 > MAX_BYTES:
        raise ResearchError('review exceeds the 20 MiB browser import limit')
    with locked(root):
        if load_study(root)['content_sha256'] != study['content_sha256']:
            raise ResearchError('study changed during review preparation')
        (root/'reviews').mkdir(exist_ok=True)
        write(output/'review.json', bundle)
        write(root/'reviews'/f"{bundle['content_sha256']}.json", bundle)
        event(root, 'review_prepared', review_sha256=bundle['content_sha256'], entries=len(entries), split=split)
    return {'review': str(output/'review.json'), 'content_sha256': bundle['content_sha256'], 'entries': len(entries)}


def _sha(value):
    if not isinstance(value, str) or not re.fullmatch('[0-9a-f]{64}', value):
        raise ResearchError('feedback identity requires SHA-256')
    return value


def _observation(value, entries):
    fields = {'id', 'entry_id', 'artwork_sha256', 'program_sha256', 'preview_sha256', 'renderer', 'preference', 'observed_at'}
    if not isinstance(value, dict) or set(value) != fields:
        raise ResearchError('feedback observation has missing or unknown fields')
    if not isinstance(value['id'], str) or not re.fullmatch('[a-zA-Z0-9_-]{1,80}', value['id']):
        raise ResearchError('invalid observation ID')
    if not isinstance(value['entry_id'], str):
        raise ResearchError('invalid review entry ID')
    entry = entries.get(value['entry_id'])
    if entry is None or value['artwork_sha256'] != entry['artwork_sha256'] or value['program_sha256'] != entry['preparation']['program_sha256']:
        raise ResearchError('feedback does not identify the reviewed artwork and preparation')
    _sha(value['preview_sha256'])
    if value['renderer'] != 'inkmap-metric-svg/1' or value['preference'] not in ('like', 'dislike', 'unrated'):
        raise ResearchError('unsupported renderer or preference')
    try:
        timestamp = datetime.fromisoformat(value['observed_at'].replace('Z', '+00:00'))
        if timestamp.utcoffset() is None:
            raise ValueError('timezone missing')
    except (ValueError, TypeError, AttributeError) as error:
        raise ResearchError('observation requires a timestamp with timezone') from error


def _validate_observations(observations, review, observed):
    if not isinstance(observations, list) or not 1 <= len(observations) <= 10000:
        raise ResearchError('feedback requires 1..10000 observations')
    entries = {entry['id']: entry for entry in review['entries']}
    seen = set()
    for observation in observations:
        _observation(observation, entries)
        key = observation['id']
        if key in seen:
            raise ResearchError('duplicate observation ID')
        seen.add(key)
        previous = observed/f'{key}.json'
        record = {'review_sha256': review['content_sha256'], 'observation': observation}
        if previous.exists() and read(previous) != record:
            raise ResearchError('observation ID conflicts with previously imported feedback')


def ingest(root, path):
    """Preserve human observations verbatim; imports do not qualify a baseline."""
    root, path = Path(root), Path(path)
    if path.stat().st_size > MAX_BYTES:
        raise ResearchError('feedback exceeds 20 MiB')
    value = frozen(path, FEEDBACK)
    if set(value) != {'schema', 'content_sha256', 'review_sha256', 'study_sha256', 'observations'}:
        raise ResearchError('feedback has missing or unknown fields')
    review_hash = _sha(value['review_sha256'])
    with locked(root):
        study = load_study(root)
        review = frozen(root/'reviews'/f'{review_hash}.json', REVIEW)
        if review['content_sha256'] != review_hash or value['study_sha256'] != study['content_sha256'] or review['study_sha256'] != value['study_sha256']:
            raise ResearchError('feedback belongs to another review or study')
        observations = value['observations']
        folder = root/'feedback'
        observed = folder/'observations'
        _validate_observations(observations, review, observed)
        observed.mkdir(parents=True, exist_ok=True)
        destination = folder/f"{value['content_sha256']}.json"
        # An import receipt is also the deduplication key after a lost CLI response.
        if destination.exists():
            if read(destination) != value:
                raise ResearchError('feedback receipt differs')
            return {'feedback': str(destination), 'already_imported': True}
        for observation in observations:
            write(observed/f"{observation['id']}.json", {'review_sha256': review_hash, 'observation': observation})
        write(destination, value)
        event(root, 'human_feedback_imported', feedback_sha256=value['content_sha256'], review_sha256=review_hash,
              observations=len(observations), file_sha256=hashlib.sha256(path.read_bytes()).hexdigest())
    return {'feedback': str(destination), 'already_imported': False, 'observations': len(observations)}
