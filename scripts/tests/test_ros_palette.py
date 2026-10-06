"""`tatbot ros palette`: cap declarations with a measured ink level, in a copy of the config."""
from __future__ import annotations

import json
import shutil
from pathlib import Path

import ink_spec
import pytest
from tatbot_cli import ros_palette

REPO = Path(__file__).resolve().parents[2]


@pytest.fixture
def owner(tmp_path):
    repo = tmp_path/'repo'
    (repo/'config').mkdir(parents=True)
    for name in ('palette.yaml', 'palette_load.yaml', 'inks.yaml', 'arms.json'):
        shutil.copy(REPO/'config'/name, repo/'config'/name)
    return repo


def test_declared_caps_record_presence_and_level_and_no_volume(owner):
    initial = ros_palette.status(owner)
    result = ros_palette.declare(owner, ['M1=nighthawk_black', 'L1=none', 'S1=absent', 'S2=unknown'],
                                 ['inkcap_medium_1=6'])
    assert result['slots']['inkcap_medium_1'] == {'ink_id': 'nighthawk_black', 'cap_present': True,
                                                  'level_lower_bound_m': 0.006, 'fill_ul': None}
    assert result['slots']['inkcap_large_1']['cap_present'] is True
    assert result['slots']['inkcap_small_1']['cap_present'] is False
    assert result['slots']['inkcap_small_2']['cap_present'] is None
    assert result['slots']['inkcap_medium_2'] == initial['slots']['inkcap_medium_2']
    assert ink_spec.load_palette_load(owner)['inkcap_medium_1'].dry is False


@pytest.mark.parametrize('caps,levels', [
    ([], []), (['M1=missing-ink'], ['M1=6']), (['M1=nighthawk_black'], []),
    (['M1=nighthawk_black'], ['M1=nan']), (['M1=nighthawk_black'], ['M1=0']),
    (['M1=nighthawk_black'], ['M1=9.5']), (['M1=none'], ['M1=6']),
    (['M1=unknown'], ['M1=6']), (['M1=absent'], ['M1=6']),
    (['M1=none', 'inkcap_medium_1=absent'], []), (['M1=none'], ['M2=6']), (['M9=none'], []),
])
def test_invalid_declarations_leave_the_file_alone(owner, caps, levels):
    before = (owner/ink_spec.LOAD_RELPATH).read_bytes()
    with pytest.raises(ValueError):
        ros_palette.declare(owner, caps, levels)
    assert (owner/ink_spec.LOAD_RELPATH).read_bytes() == before


def test_a_volume_write_keeps_the_measured_level(owner):
    ros_palette.declare(owner, ['M1=nighthawk_black'], ['M1=6'])
    load = ink_spec.load_palette_load(owner)
    load['inkcap_large_1'] = ink_spec.SlotLoad('inkcap_large_1', 'nighthawk_black', 10)
    ink_spec.write_palette_load(load, owner)
    after = ink_spec.load_palette_load(owner)['inkcap_medium_1']
    assert after.level_lower_bound_m == 0.006 and after.fill_known is False


def test_cli_prints_the_declarations(owner, monkeypatch, capsys):
    monkeypatch.setenv('TATBOT_REPO', str(owner))
    assert ros_palette.main(['load', 'M1=nighthawk_black', '--level-mm', 'M1=6']) == 0
    assert json.loads(capsys.readouterr().out)['slots']['inkcap_medium_1']['fill_ul'] is None
    assert ros_palette.main(['status', 'M1=none']) == 2
    assert json.loads(capsys.readouterr().out)['ok'] is False
