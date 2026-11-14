"""Runtime provenance: a run's record of its sources and loaded controller code, and what makes it incomplete."""
from pathlib import Path

from tatbot_session import runtime


def test_a_missing_controller_record_is_incomplete(tmp_path):
    assert not runtime.controller_identity(None)['complete']
    assert not runtime.controller_identity(tmp_path/'missing.json')['complete']


def test_session_snapshot_covers_shared_and_owner_source_roots(tmp_path, monkeypatch):
    import tatbot_bridge
    import tatbot_contracts
    import tatbot_description
    import tatbot_motion
    import tatbot_session

    for module in (tatbot_session, tatbot_motion, tatbot_description, tatbot_bridge, tatbot_contracts):
        root = tmp_path/module.__name__
        root.mkdir()
        entry = root/'__init__.py'
        entry.write_text('VALUE = 1\n')
        monkeypatch.setattr(module, '__file__', str(entry))
    for folder in ('lib', 'vision'):
        root = tmp_path/'scripts'/folder
        root.mkdir(parents=True)
        (root/'source.py').write_text('VALUE = 1\n')
    identity = runtime.session_identity(tmp_path)
    assert set(identity['source_roots']) == {'tatbot_session', 'tatbot_motion', 'tatbot_description',
                                           'tatbot_bridge', 'tatbot_contracts', 'scripts/lib', 'scripts/vision'}
    assert runtime.current(identity)['source_status']['complete']
    for root in ('tatbot_bridge', 'tatbot_contracts', 'scripts/lib', 'scripts/vision'):
        path = next(Path(identity['source_roots'][root]).glob('*.py'))
        before = path.read_bytes()
        path.write_bytes(before+b'# changed dependency\n')
        assert not runtime.current(identity)['source_status']['complete']
        assert identity['sources_sha256'] != runtime.session_identity(tmp_path)['sources_sha256']
        path.write_bytes(before)
    assert runtime.current(identity)['source_status']['complete']
