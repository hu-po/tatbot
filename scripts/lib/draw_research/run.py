"""One resumable paired iteration using the existing ROS draw and inspect workflow."""
from __future__ import annotations

import contextlib
import hashlib
import json
import signal
import subprocess
import time
from pathlib import Path

import yaml
from tatbot_contracts.canonical import canonical_digest, parse_json
from tatbot_contracts.ros_progress import last_events
from tatbot_contracts.ros_runtime import software_digest

from draw_research.model import ResearchError, event, identifier, locked, read, write
from draw_research.prepare import _single_resource, load_trial


class Ros:
    """The production CLI is the only execution route; this class also permits offline test doubles."""
    def __init__(self, repo):
        # The CLI launcher is a bash shim that execs the stdlib CLI in the same process,
        # so a signal sent to this child still reaches `ros draw` and cancels its goal.
        self.command = [str(Path(repo)/'scripts/tatbot')]

    def evidence(self, page, slot):
        result = subprocess.run([*self.command, 'ros', 'evidence', '--page', page, '--slot', slot],
                                check=True, capture_output=True, text=True, timeout=30)
        return parse_json(result.stdout)

    def ready(self, log):
        return self._execute(['ready', '--arm', 'right'], log)

    def page_measured(self):
        return ((self._status(60).get('page') or {}).get('summary') or {}).get('measured', 0) > 0

    def landed(self):
        return self._status(30).get('stack', {}).get('safety', {}).get('right', {}).get('landed') is True

    def _status(self, timeout):
        """Live telemetry, not digest material: plain JSON. The page pose can hold -0.0, which the canonical
        reader refuses; that stopped a pair after its first side (2026-09-30)."""
        result = subprocess.run([*self.command, 'ros', 'status', '--json'],
                                check=True, capture_output=True, text=True, timeout=timeout)
        return json.loads(result.stdout)

    def draw(self, program, log, *, resume=None, hold=False):
        command = ['draw', str(program), '--no-wake']
        if resume:
            command += ['--resume', resume]
        if hold:
            command += ['--hold']
        return self._execute(command, log)

    def _execute(self, command, log):
        with Path(log).open('x') as stream:
            process = subprocess.Popen([*self.command, 'ros', *command], stdout=stream, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL)
            previous = signal.signal(signal.SIGTERM, _interrupt)
            try:
                return process.wait()
            except KeyboardInterrupt:
                process.send_signal(signal.SIGINT)  # existing ros CLI cancels its goal
                # The durable claim/ledger remains authoritative after a timeout.
                with contextlib.suppress(subprocess.TimeoutExpired):
                    process.wait(timeout=60)
                raise
            finally:
                signal.signal(signal.SIGTERM, previous)


def _interrupt(signum, frame):
    raise KeyboardInterrupt


PAGE_WAIT_S = 600.0   # a coded print was measured again 1-4 min after the arm uncovered it (2026-09-28)


def _await_page(root, trial, name, ros, *, timeout_s=PAGE_WAIT_S, poll_s=10.0):
    """Wait between attempts, not inside one, for the cameras to measure the page again. The first side and
    its held inspection hide the print; a draw that waited for it counted 230 s of waiting as drawing time on
    the second side of trial 0006. The draw's own freshness rule still decides; this only moves the wait."""
    started = time.monotonic()
    while not ros.page_measured() and time.monotonic() - started < timeout_s:
        time.sleep(poll_s)
    event(root, 'page_awaited', trial=trial['id'], side=name, seconds=round(time.monotonic() - started, 1))


def confirm_page(root, page_id, *, observation):
    """Record a concrete new-sheet placement observation; never called by the loop itself."""
    root, page_id = Path(root), identifier(page_id)
    if not isinstance(observation, str) or len(observation.strip()) < 10:
        raise ResearchError('record the operator observation establishing a fresh physical sheet')
    with locked(root):
        path = root/'pages'/f'{page_id}.json'
        if not path.exists():
            raise ResearchError('prepare the concrete trial and its intended page first')
        page = read(path)
        if page['confirmed']:
            raise ResearchError('this page was already confirmed; confirmation never resets occupied slots')
        page.update(confirmed=True, confirmation={'observation': observation})
        write(path, page)
        event(root, 'physical_page_confirmed', page=page_id, **page['confirmation'])


def _runtime(evidence):
    runtime = evidence.get('runtime')
    if evidence.get('schema') != 'tatbot.research-evidence/1' or not evidence.get('runtime_live') or runtime is None:
        raise ResearchError('a deployed, running research-capable ROS runtime is required')
    if runtime.get('schema') != 'tatbot.ros-runtime/2' or not runtime.get('controller', {}).get('complete'):
        raise ResearchError('research requires identified loaded controller code')
    try:
        software_digest(runtime)
    except ValueError as error:
        raise ResearchError(str(error)) from error
    return runtime


def _receipt_state(receipt, program, side):
    claim = receipt.get('claim')
    if claim is None:
        return 'unclaimed'
    if claim['program_sha256'] != side['program_content_sha256'] or canonical_digest(receipt['program']) != claim['program_sha256']:
        raise ResearchError('slot belongs to a different program; never redraw it')
    expected = program['preparation']['motion_sha256']
    actual = hashlib.sha256(json.dumps(yaml.safe_load(receipt['motion_yaml']), sort_keys=True).encode()).hexdigest()
    if actual != expected:
        raise ResearchError('executed motion configuration differs from the prepared candidate')
    events = last_events(receipt['ledger'], 'right')
    states = [events.get(op['id'], {}).get('event') for op in program['ops']]
    if any(value == 'skipped' for value in states):
        raise ResearchError('a skipped operation invalidates this research drawing')
    if all(value == 'done' for value in states):
        return 'complete' if receipt.get('inspection') else 'needs_inspection'
    if receipt['meta'].get('status') == 'running':
        return 'running'
    return 'uncertain' if 'sent' in states else 'interrupted'


def run(root, trial_id, *, ros, resume=False, tool_id=None):
    root = Path(root)
    with locked(root):
        trial = load_trial(root, trial_id)
        directory = root/'trials'/trial_id
        if tool_id and any(_single_resource(read(directory/name/'program.json'))['tool']['id'] != tool_id for name in trial['order']):
            raise ResearchError('selected tool differs from the frozen trial; prepare a new candidate rather than overriding its tool')
        state = read(directory/'state.json')
        page = read(root/'pages'/f"{trial['page']}.json")
        if not page['confirmed']:
            raise ResearchError('a fresh physical sheet has not been established; tracker reacquisition is not a page change')
        if state['stage'] in ('drawn', 'scored', 'decided'):
            return state
        try:
            _ready(root, trial, state, ros)
            for name in trial['order']:
                _side(root, trial, state, name, ros, resume)
                write(directory/'state.json', state)
            if not ros.landed():
                raise ResearchError('both drawings are complete but final landing is unverified; reconcile with ROS before retrying')
        except BaseException as error:
            # A signal arrives as a bare KeyboardInterrupt; name it rather than journal an empty reason.
            event(root, 'execution_paused', trial=trial_id, error=str(error) or type(error).__name__)
            raise
        state['stage'] = 'drawn'
        write(directory/'state.json', state)
        event(root, 'pair_drawn', trial=trial_id)
        return state


def _ready(root, trial, state, ros):
    """Only a genuinely unstarted pair may use ordinary startup/restart behavior."""
    if state.get('runtime') is not None or any(side['attempts'] for side in state['sides'].values()):
        return
    directory = root/'trials'/trial['id']
    statuses = [_receipt_state(ros.evidence(trial['page'], side['slot']), read(directory/name/'program.json'), side)
                for name, side in trial['sides'].items()]
    if any(status != 'unclaimed' for status in statuses):
        return
    attempts = state.setdefault('ready_attempts', [])
    log = f'ready-{len(attempts):03d}.log'
    attempts.append({'log': log})
    write(directory/'state.json', state)
    attempts[-1]['exit_code'] = ros.ready(directory/log)
    write(directory/'state.json', state)
    event(root, 'pair_readied', trial=trial['id'], **attempts[-1])
    if attempts[-1]['exit_code'] != 0:
        raise ResearchError('ROS startup did not complete; no drawing was dispatched')


def _side(root, trial, state, name, ros, resume):
    directory = root/'trials'/trial['id']
    side, progress = trial['sides'][name], state['sides'][name]
    program = read(directory/name/'program.json')
    receipt = ros.evidence(trial['page'], side['slot'])
    runtime = _runtime(receipt)
    if state.get('runtime') is not None and state['runtime'] != runtime:
        raise ResearchError('ROS runtime changed during the pair; the comparison is inconclusive')
    state['runtime'] = runtime
    status = _receipt_state(receipt, program, side)
    if receipt.get('claim') is not None:
        _execution_context(state, receipt, program)
    if progress['status'] == 'complete' and status != 'complete':
        raise ResearchError('previously complete execution evidence is now missing or inconsistent')
    if status == 'complete':
        _complete(directory, state, name, receipt)
        return
    if status in ('running', 'uncertain', 'needs_inspection'):
        raise ResearchError(f"side {name}: {status}; preserve its slot and reconcile run {receipt['claim']['run_id']}")
    if status == 'interrupted' and not resume:
        raise ResearchError('known interrupted run: use --resume to continue its existing ledger')
    _await_page(root, trial, name, ros)
    # Persist intent before dispatch. If the process dies here, the execution-owner
    # claim distinguishes an unstarted side from one already sent to the arm.
    attempt = len(progress['attempts'])
    log = directory/name/f'draw-{attempt:03d}.log'
    progress['attempts'].append({'log': log.name, 'resume': receipt.get('claim', {}).get('run_id') if receipt.get('claim') else None})
    progress['status'] = 'dispatching'
    write(directory/'state.json', state)
    event(root, 'side_dispatching', trial=trial['id'], side=name, attempt=attempt)
    started = time.monotonic()
    result = ros.draw(directory/name/'program.json', log, resume=progress['attempts'][-1]['resume'], hold=name != trial['order'][-1])
    progress['attempts'][-1]['elapsed_s'] = time.monotonic() - started
    receipt = ros.evidence(trial['page'], side['slot'])
    write(directory/name/f'receipt-{attempt:03d}.json', receipt)
    progress['attempts'][-1]['exit_code'] = result
    status = _receipt_state(receipt, program, side)
    progress['status'] = status
    write(directory/'state.json', state)
    if status != 'complete':
        raise ResearchError(f'side {name}: {status}; its slot stays occupied')
    if _runtime(receipt) != state['runtime'] or receipt['meta'].get('runtime') != state['runtime']:
        raise ResearchError('executed runtime drift makes the trial inconclusive')
    _execution_context(state, receipt, program)
    _complete(directory, state, name, receipt)
    write(directory/'state.json', state)
    event(root, 'side_completed', trial=trial['id'], side=name, run_id=receipt['claim']['run_id'])
    if result != 0:
        raise ResearchError(f'side {name}: ROS completion workflow returned {result}; reconcile its inspection/landing before retrying')


def _complete(directory, state, name, receipt):
    write(directory/name/'evidence.json', receipt)
    state['sides'][name].update(status='complete', run_id=receipt['claim']['run_id'])
    state['stage'] = 'first_completed'


def _execution_context(state, receipt, program):
    meta = receipt['meta']
    if meta.get('runtime') != state['runtime']:
        raise ResearchError('recorded execution runtime differs from this pair')
    context = {k: meta[k] for k in ('hardware', 'estop_source', 'page_source', 'touch', 'arms')}
    if context['arms']['right']['tool'] != _single_resource(program)['tool']['id']:
        raise ResearchError('executed tool differs from the prepared candidate')
    if (meta.get('program') or {}).get('resources') != program['resources']:
        raise ResearchError('executed resource constraints differ from the prepared candidate')
    context['resources'] = meta['program']['resources']
    if state.get('execution_context') is not None and context != state['execution_context']:
        raise ResearchError('tool, registration, page trim or execution configuration changed during the pair')
    state['execution_context'] = context
