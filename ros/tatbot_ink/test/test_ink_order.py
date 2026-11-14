"""A lighter ink is not drawn over a darker one's mark: within a program, and after the programs drawn before it."""
from __future__ import annotations

import copy
import json
import shutil
from pathlib import Path

import pytest
import tatbot_ink
from ink_designs import REPO, artwork, element, program, write
from resource_fixtures import fixture_repo, manifest, resource
from tatbot_contracts.ros_program import validate_for_execution
from tatbot_ink import order
from tatbot_ink.__main__ import main
from tatbot_ink.errors import CompileError
from tatbot_ink.input import read_input

PAIR = Path(__file__).parent / 'fixtures/chicano_pair.json'


@pytest.fixture(scope='module')
def pair():
    """A black lettering and its red fill as they were prepared for one print: the red separated to reach 0.4 mm
    under the black, on the understanding that it is drawn first. The black was drawn first."""
    return json.loads(PAIR.read_text())


def test_lightness_orders_the_catalogue_inks():
    assert order.lightness([0, 0, 0]) == 0 and order.lightness([255, 255, 255]) == pytest.approx(100)
    assert order.lightness([255, 0, 0]) - order.lightness([60, 60, 63]) > order.LIGHTER_L
    assert order.lightness(None) is None


def test_the_red_after_the_black_it_reaches_under_is_refused_and_the_black_after_the_red_is_not(pair):
    black, red = pair['black'], pair['red']
    with pytest.raises(CompileError, match=r'bright_red \(from s0002\) draws after true_black, a darker ink of an '
                                           r'earlier run, and their marks overlap by 8\d\d(\.\d+)? mm2'):
        order.check(red, [(black, False)])
    assert order.check(black, [(red, False)]) == {'order': ['true_black'], 'after': ['bright_red'], 'pairs': []}
    assert order.check(red) == {'order': ['bright_red'], 'after': [], 'pairs': []}


def test_a_wipe_after_the_earlier_run_lets_the_lighter_ink_follow(pair):
    report = order.check(pair['red'], [(pair['black'], True)])
    assert [(p['darker'], p['lighter'], p['verdict'], p['earlier_run']) for p in report['pairs']] == [
        ('true_black', 'bright_red', 'wiped', True)]
    # a wipe before the black was drawn took nothing of the black off
    with pytest.raises(CompileError, match='ink order'):
        order.check(pair['red'], [(pair['red'], True), (pair['black'], False)])


def test_both_inks_in_one_program_are_refused_dark_first(pair):
    """The two as one program, the black's ops then the red's: the same refusal, with no earlier run."""
    black, red = copy.deepcopy(pair['black']), copy.deepcopy(pair['red'])
    for op in red['ops']:
        op['id'] = 'r' + op['id']
    both = {**black, 'resources': black['resources'] + red['resources'], 'ops': black['ops'] + red['ops'],
            'stats': {'passes': {'marks': {**black['stats']['passes']['marks'], **red['stats']['passes']['marks']}}}}
    with pytest.raises(CompileError, match=r'bright_red \(from rs0002\) draws after true_black, a darker ink, and'):
        order.check(both)
    change = next(op for op in both['ops'] if op['id'] == 'rt0000')
    change['wipe'] = True
    assert order.check(both)['pairs'][0]['verdict'] == 'wiped'
    light_first = {**both, 'ops': red['ops'] + black['ops']}
    assert order.check(light_first) == {'order': ['bright_red', 'true_black'], 'after': [], 'pairs': []}


# --- through the compile ---------------------------------------------------------------------------------------

def two_inks(tmp_path, first, then, *, gap=0.0, resources=None):
    """A 10 mm bar of `first` ink and, drawn after it, one of `then`, their 0.5 mm marks sharing an edge strip
    unless `gap` (m) parts them."""
    y = {first: .0100, then: .0100 + .0004 + gap}
    layers = [(ink, [element([[.003, y[ink]], [.013, y[ink]]])]) for ink in (first, then)]
    art = artwork(program(layers))
    design = read_input(art)
    repo = fixture_repo(tmp_path / 'repo')
    motion = repo / 'ros/tatbot_motion/config/motion.yaml'
    motion.parent.mkdir(parents=True)
    shutil.copyfile(REPO / 'ros/tatbot_motion/config/motion.yaml', motion)
    inks = manifest(tmp_path / 'inks.yaml', design,
                    resources or [resource('K', ink='black'), resource('R', ink='red')],
                    [('black', 'K'), ('red', 'R')])
    return write(tmp_path, art), repo, inks


def test_compile_refuses_red_after_black_where_they_touch_and_takes_red_first(tmp_path):
    path, repo, inks = two_inks(tmp_path, 'black', 'red')
    with pytest.raises(CompileError, match=r'ink order: red \(from s\d+\) draws after black, a darker ink, and their '
                                           'marks overlap by'):
        tatbot_ink.compile(path, repo=repo, inks_path=inks)
    path, repo, inks = two_inks(tmp_path / 'light-first', 'red', 'black')
    result = tatbot_ink.compile(path, repo=repo, inks_path=inks)
    assert result['stats']['ink_order'] == {'order': ['red', 'black'], 'after': [], 'pairs': []}
    assert 'drawn_after' not in result['preparation']


def test_a_design_separated_with_a_gap_compiles_dark_first(tmp_path):
    path, repo, inks = two_inks(tmp_path, 'black', 'red', gap=.0003)   # 0.7 mm between 0.5 mm marks
    result = tatbot_ink.compile(path, repo=repo, inks_path=inks)
    assert [(p['darker'], p['lighter'], p['overlap_mm2'], p['verdict']) for p in result['stats']['ink_order']['pairs']] \
        == [('black', 'red', 0.0, 'clear')]


def test_wipe_before_puts_a_wipe_on_the_tool_change_and_lets_the_lighter_ink_follow(tmp_path):
    wiped = [resource('K', ink='black'), resource('R', ink='red', wipe_before=True)]
    path, repo, inks = two_inks(tmp_path, 'black', 'red', resources=wiped)
    result = tatbot_ink.compile(path, repo=repo, inks_path=inks)
    changes = [op for op in result['ops'] if op['op'] == 'tool_change']
    assert ['wipe' in op for op in changes] == [False, True] and changes[1]['wipe'] is True
    assert result['stats']['ink_order']['pairs'][0]['verdict'] == 'wiped'
    validate_for_execution(result)
    first = copy.deepcopy(result)
    first['ops'][0]['wipe'] = True
    with pytest.raises(ValueError, match='a wipe is'):
        validate_for_execution(first)
    falsy = copy.deepcopy(result)
    next(op for op in falsy['ops'] if op['op'] == 'tool_change' and not op['initial'])['wipe'] = False
    with pytest.raises(ValueError, match='a wipe is'):
        validate_for_execution(falsy)


@pytest.mark.parametrize(('resources', 'message'), [
    ([resource('K', ink='black', wipe_before=True), resource('R', ink='red')], 'wipes nothing this program drew'),
    ([resource('K', ink='black'), resource('R', ink='red', wipe_before='yes')], 'wipe_before is true or false'),
])
def test_a_wipe_that_cannot_happen_is_refused(tmp_path, resources, message):
    path, repo, inks = two_inks(tmp_path, 'black', 'red', resources=resources)
    with pytest.raises(CompileError, match=message):
        tatbot_ink.compile(path, repo=repo, inks_path=inks)


def test_after_names_the_programs_already_on_the_print(tmp_path):
    """The red alone compiles; after the black's program it is refused, and with --wiped it is prepared and the
    program records what it was drawn after."""
    black_art, repo, black_inks = two_inks(tmp_path / 'k', 'black', 'red')
    art = json.loads(black_art.read_text())

    def one(ink, folder):
        solo = copy.deepcopy(art)
        layers = [(layer['ink_id'], layer['elements']) for layer in solo['program']['layers'] if layer['ink_id'] == ink]
        doc = artwork(program(layers))
        design = read_input(doc)
        inks = manifest(folder / 'inks.yaml', design, [resource('X', ink=ink)], [(ink, 'X')])
        return write(folder, doc), inks

    (tmp_path / 'black').mkdir()
    (tmp_path / 'red').mkdir()
    black_path, black_inks = one('black', tmp_path / 'black')
    red_path, red_inks = one('red', tmp_path / 'red')
    earlier = tatbot_ink.write_program(tatbot_ink.compile(black_path, repo=repo, inks_path=black_inks),
                                       tmp_path / 'black/program.json')
    assert tatbot_ink.compile(red_path, repo=repo, inks_path=red_inks)['stats']['ink_order']['after'] == []
    with pytest.raises(CompileError, match='draws after black, a darker ink of an earlier run'):
        tatbot_ink.compile(red_path, repo=repo, inks_path=red_inks, after=[earlier])
    result = tatbot_ink.compile(red_path, repo=repo, inks_path=red_inks, after=[earlier], wiped=True)
    assert result['stats']['ink_order']['pairs'][0]['verdict'] == 'wiped'
    assert [(row['wiped'], len(row['sha256'])) for row in result['preparation']['drawn_after']] == [(True, 64)]
    with pytest.raises(CompileError, match='name its program with --after'):
        tatbot_ink.compile(red_path, repo=repo, inks_path=red_inks, wiped=True)
    with pytest.raises(CompileError, match='--after .*artwork.json'):
        tatbot_ink.compile(red_path, repo=repo, inks_path=red_inks, after=[red_path])
    assert main(['compile', str(red_path), '--repo', str(repo), '--arm', 'right', '--inks', str(red_inks),
                                     '--after', str(earlier), '-o', str(tmp_path / 'out')]) == 1
    assert main(['compile', str(red_path), '--repo', str(repo), '--arm', 'right', '--inks', str(red_inks),
                                     '--after', str(earlier), '--wiped', '-o', str(tmp_path / 'out')]) == 0
