"""Production draw startup/hold/landing, a draw after a land and a wake, and a draw interrupted while it lands for a
cartridge swap, each on a fresh isolated fake driver; no physical arm or camera."""
import importlib.util
import json
import shlex
import signal
import socket
from types import SimpleNamespace

import pytest
import session_graph as sg
from test_session_fake_e2e import Pico, _fake_urdf, _status

pytestmark = pytest.mark.skipif(not sg.ZENOHD.exists(), reason='needs ROS 2 Jazzy with rmw_zenoh_cpp')


@pytest.fixture
def graph(tmp_path):
    """A fake-driver graph of the test's own, unlatched: the arm starts awake at its staged pose."""
    port = sg.free_port(socket.SOCK_DGRAM)
    stack = sg.base_stack(hardware='fake', flight_path=str(tmp_path))
    stack['estop'].update(source='udp', udp_port=port, relay_addr='127.0.0.1')
    urdf = _fake_urdf(stack)
    pico, graph = Pico(port), sg.Graph(tmp_path)
    try:
        graph.up(urdf, stack)
        if _status(graph)['latched']:
            assert sg.last_json(graph.client('decide', 'right', 'continue').stdout)['accepted']
        yield graph
    finally:
        graph.stop()
        pico.running = False


def test_production_draw_holds_between_research_sides_and_lands_after_pair(graph, tmp_path, monkeypatch):
    spec = importlib.util.spec_from_file_location('paired_ros_backend', sg.repo()/'scripts/lib/tatbot_cli/tatbot_ros.py')
    backend = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(backend)
    env_keys = ('TATBOT_REPO', 'TATBOT_LOG_ROOT', 'ROS_LOG_DIR', 'ROS_DOMAIN_ID', 'RMW_IMPLEMENTATION', 'ZENOH_CONFIG_OVERRIDE')
    (tmp_path/'env.sh').write_text('\n'.join(f'export {key}={shlex.quote(graph.env[key])}' for key in env_keys)+'\n')
    target = SimpleNamespace(root=str(tmp_path), local=True, node='fixture', units=('fixture-router', 'fixture-stack'),
                             dest=lambda path: str(tmp_path/path), shell=lambda command: ['bash', '-c', command],
                             in_env=lambda command: f'source {tmp_path}/env.sh && {command}')
    monkeypatch.setattr(backend, 'ros_target', lambda: target)
    # Graph owns processes directly, not system units. All status and action traffic is real ROS.
    monkeypatch.setattr(backend, 'UNIT_UP', 'true')
    monkeypatch.setattr(backend, 'restart', lambda *args: pytest.fail('a pair must not restart its active controller'))

    def execute(target, command, runner):
        result = graph.run(['bash', '-c', command], timeout=180)
        print(result.stdout, result.stderr)
        return result.returncode

    monkeypatch.setattr(backend, 'run_cancellable', execute)
    runner = backend.Runner(False)
    backend.ready(SimpleNamespace(arm='right'), runner)
    runtimes = []
    for index in range(2):
        program = sg.square_program(tmp_path, name=f'pair-{index}', sides=(.004,))
        document = json.loads(program.read_text())
        document['research'] = {'study_sha256': 'a'*64, 'trial': '0000', 'side': 'ab'[index],
                                'page': 'synthetic-pair', 'slot': f'0:{index}', 'slot_m': [.026, .026]}
        program.write_text(json.dumps(document))
        # Camera inspection needs the physical rig; this software test explicitly omits it.
        args = ['draw', str(program), '--no-wake', '--no-inspect'] + (['--hold'] if index == 0 else [])
        assert backend.draw(backend.parser().parse_args(args), runner) == 0
        assert _status(graph)['landed'] is (index == 1)
        receipt = graph.evidence('synthetic-pair', f'0:{index}')
        assert receipt.returncode == 0, receipt.stderr
        evidence = json.loads(receipt.stdout)
        assert evidence['meta']['hardware'] == 'fake' and evidence['meta']['status'] == 'ok'
        assert evidence['runtime_live'] and evidence['runtime'] == evidence['meta']['runtime']
        assert evidence['inspection'] is None
        assert [r['event'] for r in evidence['ledger'] if r['event'] in ('sent', 'done') and r['op'][0] == 's'] == ['sent', 'done']
        (tmp_path/f'evidence-{index}.json').write_text(json.dumps(evidence, indent=2)+'\n')
        runtimes.append(evidence['runtime'])
    assert runtimes[0] == runtimes[1]


def test_a_woken_arm_draws_again_after_landing_from_a_draw(graph):
    """Land after a draw, wake, draw: the keypad cartridge swap's land -> Enter -> wake -> resume. The first goal
    after the wake latched over_velocity 7 ms in, at the sleep pose (fake driver, 2026-10-04)."""
    for name in ('before-land', 'after-wake'):
        draw = graph.client('draw', str(sg.square_program(graph.tmp, name=name, sides=(.004,))), '--arm', 'right',
                            timeout=240)
        assert draw.returncode == 0, draw.stdout + draw.stderr
        if name == 'before-land':
            land = graph.client('land', '--arm', 'right', timeout=150)
            assert land.returncode == 0, land.stdout + land.stderr
            wake = graph.client('wake', '--arm', 'right', timeout=60)
            assert wake.returncode == 0, wake.stdout + wake.stderr
    right = _status(graph)
    assert right['latched'] is False and right['landed'] is False, right


def _swap_program(graph):
    """Two squares, the second in another cartridge's ink: the draw lands for the swap between them."""
    path = sg.square_program(graph.tmp, name='swap', sides=(.004, .004))
    program = json.loads(path.read_text())
    program['resources'].append({**program['resources'][0], 'id': 'swap', 'ink_id': 'swap'})
    at = next(i for i, op in enumerate(program['ops']) if op['op'] == 'stroke' and op['src']['placement'] == 'p1')
    for op in program['ops'][at:]:
        op.update(resource_id='swap', ink='swap')
    program['ops'].insert(at, {**program['ops'][0], 'id': 't9002', 'resource_id': 'swap', 'initial': False})
    path.write_text(json.dumps(program))
    return path


def test_a_draw_interrupted_landing_for_a_swap_does_not_wait_for_the_keypad(graph):
    """SIGTERM while a draw lands for a cartridge swap: the client cancels it and exits once the arm has landed. It
    used to go on to wait for the keypad's Enter with every signal ignored after the cancel."""
    keypad = sg.base_stack()['keypad']   # the committed stack.yaml, which the client reads through the install
    assert keypad.get('device') and keypad.get('arm') == 'right', keypad
    log = 'swap-sigterm.log'
    proc = graph.client_bg('draw', str(_swap_program(graph)), '--arm', 'right', log=log)
    try:
        assert graph.wait_for(log, ' pause t9002 ', 120), (graph.tmp/log).read_text()
        proc.send_signal(signal.SIGTERM)
        assert proc.wait(timeout=45) == 1
    finally:
        graph.end(proc)
    text = (graph.tmp/log).read_text()
    assert 'cancelling; the arm holds' in text and 'on the keypad' not in text, text
    rows = sg.ledger(sg.last_json(text)['run_dir'])
    assert [r['event'] for r in rows if r.get('op') == 't9002'] == ['sent']
    assert _status(graph)['landed'] is True
