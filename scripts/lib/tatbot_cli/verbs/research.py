"""Paired DBV3 trials through the existing preparation and ROS execution path."""
from functools import partial

from draw_research.cli import SUMMARIES, arguments

from tatbot_cli.registry import MOTION_AUTO, OFFLINE, verb
from tatbot_cli.verbs._common import sh

SCRIPT = 'scripts/research.sh'
WRAPS = (SCRIPT, 'scripts/research.py', 'scripts/lib/drawing_python.sh', 'scripts/lib/drawing_python.py',
         'scripts/lib/draw_research/cli.py', 'scripts/lib/draw_research/model.py',
         'scripts/lib/draw_research/review.py', 'scripts/lib/draw_research/prepare.py', 'scripts/lib/draw_research/run.py', 'scripts/lib/draw_research/assess.py')
EXAMPLES = {
    'init': ('/tmp/corpus.json', '--out', '/tmp/study'),
    'candidate': ('/tmp/study', '/tmp/acquisitions.json', '--out', '/tmp/candidate'),
    'review': ('/tmp/study', '/tmp/acquisitions.json', '--out', '/tmp/review'),
    'feedback': ('/tmp/study', '/tmp/feedback.json'),
    'prepare': ('/tmp/study', '--a', '/tmp/candidate', '--b', '/tmp/candidate', '--case', 'flower', '--page', 'sheet-001',
                '--row', '0', '--hypothesis', 'Measure repeatability with identical DBV3 inputs'),
    'page': ('/tmp/study', 'sheet-001', '--observation', 'Operator replaced the completed sheet with a fresh print'),
    'run': ('/tmp/study', '0000'), 'score': ('/tmp/study', '0000'),
    'decide': ('/tmp/study', 'qualify', '0000', '0001', '--reason', 'Repeated A/A drawings inspected and accepted'),
}


def _dispatch(ctx, ns, rest):
    tokens = ctx.invocation.command_tokens[1:]
    tool = ['--ee-tool', ctx.ee_tool] if tokens[0] in ('candidate', 'review', 'run') and ctx.ee_tool else []
    return sh(ctx, SCRIPT, *tokens, *tool, *rest)


for _name, _summary in SUMMARIES.items():
    _effects = ('read_files', 'write_files', 'start_process')
    if _name == 'run':
        _effects += ('network', 'remote_exec', 'autonomous_motion')
    verb(noun='research', verb=_name, tier=MOTION_AUTO if _name == 'run' else OFFLINE,
         effects=_effects, args=partial(arguments, command=_name), wraps=WRAPS, doc='docs/research.md',
         summary=_summary, example=EXAMPLES[_name])(_dispatch)
