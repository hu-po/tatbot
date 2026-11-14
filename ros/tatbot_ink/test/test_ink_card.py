"""A cartridge whose datasheet says its dip values are another's is refused for a design until its card is read."""
from __future__ import annotations

import shutil

import pytest
import tatbot_ink
import yaml
from ink_designs import REPO, artwork, element, program, write
from resource_fixtures import fixture_repo, manifest, resource, save
from tatbot_ink.__main__ import main
from tatbot_ink.errors import CompileError
from tatbot_ink.input import read_input


@pytest.fixture
def design(tmp_path):
    art = artwork(program([('black', [element([[.003, .01], [.013, .01]])])]))
    repo = fixture_repo(tmp_path / 'repo')
    motion = repo / 'ros/tatbot_motion/config/motion.yaml'
    motion.parent.mkdir(parents=True)
    shutil.copyfile(REPO / 'ros/tatbot_motion/config/motion.yaml', motion)
    inks = manifest(tmp_path / 'inks.yaml', read_input(art), [resource('N', tool='round-3', mode='ride')],
                    [('black', 'N')], substrate='practice')
    return write(tmp_path, art), repo, inks


def card(repo, value):
    path = repo / 'config/tools/round-3.yaml'
    sheet = yaml.safe_load(path.read_text())
    sheet['dip']['card'] = value
    save(path, sheet)


def test_borrowed_dip_values_are_refused_for_a_design_and_drawn_only_when_allowed(design, tmp_path):
    path, repo, inks = design
    plain = tatbot_ink.compile(path, repo=repo, inks_path=inks)
    assert 'dip_card' not in plain['resources'][0]
    card(repo, {'status': 'borrowed', 'from': ['round-5']})
    with pytest.raises(CompileError, match=r"round-3: the dip values .* are another cartridge's.*--dip-card"):
        tatbot_ink.compile(path, repo=repo, inks_path=inks)
    allowed = tatbot_ink.compile(path, repo=repo, inks_path=inks, allow_borrowed_dip=True)
    assert allowed['resources'][0]['dip_card'] == 'borrowed' and allowed['resources'][0]['dip'] == plain['resources'][0]['dip']
    args = ['compile', str(path), '--repo', str(repo), '--arm', 'right', '--inks', str(inks), '-o', str(tmp_path / 'out')]
    assert main(args) == 1 and main([*args, '--allow-borrowed-dip']) == 0
    card(repo, {'status': 'read', 'study': 'dip-round-3'})
    assert 'dip_card' not in tatbot_ink.compile(path, repo=repo, inks_path=inks)['resources'][0]
    card(repo, {'status': 'guessed'})
    with pytest.raises(CompileError, match='card.status is borrowed or read'):
        tatbot_ink.compile(path, repo=repo, inks_path=inks)


def test_a_borrowed_cartridge_drawn_dry_is_not_refused(design):
    """Without a cap it never dips, so what its dip values are does not matter."""
    path, repo, inks = design
    card(repo, {'status': 'borrowed'})
    manifest(inks, read_input(artwork(program([('black', [element([[.003, .01], [.013, .01]])])]))),
             [resource('N', tool='round-3', mode='hover', slot=None)], [('black', 'N')], substrate='practice')
    assert tatbot_ink.compile(path, repo=repo, inks_path=inks)['stats']['dips'] == 0


def test_the_1007rl_is_the_first_borrowed_cartridge_and_no_other_datasheet_claims_a_card():
    """Its depth and dwell are the 11RL's and its mm_per_dip the 3RL's first figure (2026-11-10's red)."""
    cards = {path.stem: (yaml.safe_load(path.read_text()).get('dip') or {}).get('card')
             for path in sorted((REPO / 'config/tools').glob('*.yaml'))}
    assert {tool: value['status'] for tool, value in cards.items() if value} == {'lutin-7rl': 'borrowed'}
