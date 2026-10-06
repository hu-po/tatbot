"""Shared command grammar for the thin CLI facade and the logged research driver."""
from __future__ import annotations

import argparse
import json
import os

SUMMARIES = {
    'init': 'freeze a source-family corpus and create an empty DBV3 study',
    'candidate': 'freeze one DBV3 recipe and preparation policy across train and validation sources',
    'review': 'prepare native artwork and production costs for visual review in Inkmap',
    'feedback': 'import exact-artifact human preferences from Inkmap into a study',
    'prepare': 'prepare both sides and reserve an unused paired row without moving the robot',
    'page': 'record the concrete observation establishing a fresh physical sheet',
    'run': 'continue one prepared DBV3 pair through ROS draw and inspect, reconciling its ledger',
    'score': 'record slot-isolated fidelity, placement and measured cost for a completed pair',
    'decide': 'record a reasoned decision and qualify or promote a DBV3 baseline from evidence',
}


def arguments(p, command):
    if command == 'init':
        p.add_argument('corpus', help='tatbot.dbv3-corpus/1 JSON with source bytes and family splits')
        p.add_argument('--out', required=True, help='new study directory outside the repository')
        p.add_argument('--layout', help='JSON with slot_mm [width,height], rows_mm centres and two columns_mm centres')
        return
    p.add_argument('study', help='frozen DBV3 study directory')
    CONFIGURE[command](p)


def _candidate(p):
    p.add_argument('acquisitions', help='JSON object mapping each train/validation case ID to its acquisition directory')
    p.add_argument('--out', required=True, help='new immutable candidate directory')
    p.add_argument('--speed', type=float, default=3.5, help='drawing speed, mm/s')
    p.add_argument('--max-chunk-s', type=float, default=60, help='planned contact-time bound per computational chunk')


def _review(p):
    p.add_argument('acquisitions', nargs='+', help='case-to-acquisition maps, all from the selected split')
    p.add_argument('--out', required=True, help='new review directory with review.json and prepared programs')
    p.add_argument('--title', default='DBV3 study review')
    p.add_argument('--split', choices=('train', 'validation'), default='train')
    p.add_argument('--speed', type=float, default=3.5, help='drawing speed, mm/s')
    p.add_argument('--max-chunk-s', type=float, default=60)


def _feedback(p):
    p.add_argument('feedback', help='feedback JSON downloaded from Inkmap')


def _prepare(p):
    p.add_argument('--a', required=True, help='frozen baseline candidate directory')
    p.add_argument('--b', required=True, help='frozen candidate directory (same as A for A/A)')
    p.add_argument('--case', required=True, help='named train or validation source')
    p.add_argument('--page', required=True, help='unique physical sheet identifier')
    p.add_argument('--row', required=True, type=int, help='unused zero-based row in study layout')
    p.add_argument('--hypothesis', required=True, help='predicted effect and why this trial is useful')
    p.add_argument('--kind', choices=('aa', 'paired'), default='aa')
    p.add_argument('--factor', default='', help='one changed policy leaf, e.g. preparation.speed_m_s')
    p.add_argument('--max-modeled-s', type=float, default=600, help='per-side screening budget; no motion')


def _page(p):
    p.add_argument('page', help='physical sheet ID already reserved by prepare')
    p.add_argument('--observation', required=True, help='concrete operator observation establishing a fresh sheet')


def _run(p):
    p.add_argument('trial', help='prepared trial ID')
    p.add_argument('--resume', action='store_true', help='allow continuation of a known interrupted run through its existing ledger')


def _score(p):
    p.add_argument('trial', help='completed trial ID')


def _decide(p):
    p.add_argument('decision', choices=('qualify', 'promote', 'keep', 'inconclusive'))
    p.add_argument('trials', nargs='+', help='distinct evidence trial IDs')
    p.add_argument('--reason', required=True, help='observed result, decision and next step')
    p.add_argument('--metric', choices=('missing_fraction', 'spill_fraction', 'spill_area_mm2', 'iou', 'draw_session_s'), default='missing_fraction')
    p.add_argument('--min-gain', type=float, default=0, help='minimum absolute improvement, also exceeding observed A/A spread')
    p.add_argument('--max-time-ratio', type=float, default=1.25, help='largest acceptable measured B/A drawing-session time ratio')


CONFIGURE = {'review': _review, 'feedback': _feedback, 'candidate': _candidate, 'prepare': _prepare, 'page': _page, 'run': _run, 'score': _score, 'decide': _decide}


def main(repo, argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    for name, summary in SUMMARIES.items():
        command = commands.add_parser(name, help=summary)
        arguments(command, name)
        if name in ('candidate', 'review', 'run'):
            # The facade owns this global flag; the standalone backend also
            # honours its environment spelling, as normal tool resolution does.
            command.add_argument('--ee-tool', default=os.environ.get('TATBOT_EE_TOOL') or None)
    ns = parser.parse_args(argv)
    from draw_research import assess, prepare, review, run

    actions = {
        'init': lambda: prepare.init(ns.corpus, ns.out, layout_path=ns.layout),
        'candidate': lambda: prepare.candidate(ns.study, ns.acquisitions, ns.out, repo=repo, tool_id=ns.ee_tool,
                                               speed_m_s=ns.speed/1000, max_chunk_s=ns.max_chunk_s),
        'review': lambda: review.build(ns.study, ns.acquisitions, ns.out, repo=repo, title=ns.title, split=ns.split,
                                       tool_id=ns.ee_tool, speed_m_s=ns.speed/1000, max_chunk_s=ns.max_chunk_s),
        'feedback': lambda: review.ingest(ns.study, ns.feedback),
        'prepare': lambda: prepare.prepare(ns.study, ns.a, ns.b, ns.case, page_id=ns.page, row=ns.row,
                                           hypothesis=ns.hypothesis, kind=ns.kind, factor=ns.factor, repo=repo,
                                           max_modeled_s=ns.max_modeled_s),
        'page': lambda: run.confirm_page(ns.study, ns.page, observation=ns.observation),
        'run': lambda: run.run(ns.study, ns.trial, ros=run.Ros(repo), resume=ns.resume, tool_id=ns.ee_tool),
        'score': lambda: assess.score(ns.study, ns.trial),
        'decide': lambda: assess.decide(ns.study, ns.trials, decision=ns.decision, reason=ns.reason,
                                      metric=ns.metric, min_gain=ns.min_gain, max_time_ratio=ns.max_time_ratio),
    }
    print(json.dumps(actions[ns.command](), indent=2, allow_nan=False))
    return 0
