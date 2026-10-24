"""The artifact schema table: one JSON both languages read, legacy rows as data."""

import json

import schemas as schema


def test_is_schema_accepts_the_pair_and_its_legacy_name_and_refuses_the_rest(monkeypatch):
    program = 'tatbot.example-program/1'
    table = schema.load()
    table['families'][program] = {'keyed_by': 'kind', 'default_kind': 'program'}
    table['kinds'][program] = ['prepared', 'program']
    table['legacy'] = {'tatbot.scan-settled/1': [schema.RECEIPT, 'scan-settled'],
                       'tatbot.example-prepared/1': [program, 'prepared']}
    monkeypatch.setattr(schema, '_TABLE', table)
    receipt = {'schema': schema.RECEIPT, 'kind': 'scan-settled', 'quiet_s': 1.0}
    assert schema.is_schema(receipt, schema.RECEIPT, 'scan-settled')
    assert not schema.is_schema(receipt, schema.RECEIPT, 'stream-result')
    assert not schema.is_schema(receipt, schema.SURFACE, 'scan-settled')
    legacy = {'schema': 'tatbot.scan-settled/1', 'quiet_s': 1.0}
    assert schema.is_schema(legacy, schema.RECEIPT, 'scan-settled')
    assert not schema.is_schema(legacy, schema.RECEIPT, 'stream-result')
    assert not schema.is_schema({'schema': 'tatbot.never-written/1'}, schema.RECEIPT, 'scan-settled')
    assert not schema.is_schema({'kind': 'scan-settled'}, schema.RECEIPT, 'scan-settled')
    assert not schema.is_schema('tatbot.scan-settled/1', schema.RECEIPT, 'scan-settled')
    # A program document carries no kind: the family's default names it.
    assert schema.is_schema({'schema': program, 'tool': 'pen'}, program, 'program')
    assert schema.is_schema({'schema': 'tatbot.example-prepared/1'}, program, 'prepared')
    assert schema.resolve('tatbot.mock-dynamics/1') is None
    assert schema.is_schema(schema.stamp(schema.RECEIPT, 'capture-failure'), schema.RECEIPT, 'capture-failure')
    assert schema.is_schema(schema.stamp(schema.RECEIPT, 'standoff-shape-binding'),
                            schema.RECEIPT, 'standoff-shape-binding')


def test_the_table_is_consistent_and_every_pending_site_names_its_lane():
    t = schema.load()
    assert set(t) == {'families', 'kinds', 'program_versions', 'bus', 'external', 'legacy', 'pending'}
    assert set(t['families']) == {schema.SURFACE, schema.RECEIPT}
    for name, (family, kind) in t['legacy'].items():
        assert family in t['families'], name
        assert kind in t['kinds'][family], name
        assert name not in t['bus'] and name not in t['external'], name
    for family, kinds in t['kinds'].items():
        assert kinds == sorted(set(kinds)), family
    for name, row in t['pending'].items():
        assert row['sites'] and all(row['sites'].values()), name
        assert row['disposition'] in ('rename', 'delete'), name
        assert (name in t['legacy']) == (row['disposition'] == 'rename'), name
    # The tree agrees with the table: the same check the fast tier runs.
    assert schema.check() == []
    assert json.loads(schema.PATH.read_text()) == t
