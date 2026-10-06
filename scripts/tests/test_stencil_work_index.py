"""New print issuance is durable and carried by the ordinary session view."""

import argparse
import json

import pytest
import stencil_frame
import stencil_work_index
import stencils


def _frame(output):
    parser = argparse.ArgumentParser()
    stencil_frame.add_arguments(parser)
    args = parser.parse_args(['--output', str(output)])
    stencil_frame.validate(args)
    args.output = output
    return args


def test_new_mark_is_issued_before_session_view_retains_exact_receipt(tmp_path):
    pytest.importorskip('PIL')
    root = tmp_path/'state'/'original-work'
    output = tmp_path/'frame'
    with stencil_work_index.IssueIndex(root) as index:
        with pytest.raises(ValueError, match='busy with an active session'), stencil_work_index.IssueIndex(root):
            pass
        settings = stencil_frame.render(_frame(output), stencil_frame.build(_frame(output)),
                                        issue_index=index)
        assert settings['issue_receipt'] == str(output/'issue-receipt.json')
        issued = root/'issued'/f'{settings["physical_instance_id"]}.json'
        assert issued.read_bytes() == (output/'issue-receipt.json').read_bytes()
    assert json.loads((root/'state.json').read_text())['enrollments'] == {}
    with stencil_work_index.IssueIndex(root):
        view = stencils.bundle([output/'tracking.json'])
        assert len(view['references']) == 1
        assert 'issue_receipt_base64' in view['references'][0]
        copied = stencils.materialize(view, tmp_path/'retained')
        assert (copied[0].parent/'issue-receipt.json').read_bytes() == issued.read_bytes()
        assert copied[0].read_bytes() == (output/'tracking.json').read_bytes()


def test_older_mark_and_interrupted_issue_cannot_gain_an_enrollment(tmp_path):
    pytest.importorskip('PIL')
    root = tmp_path/'state'/'original-work'
    output = tmp_path/'unissued'
    stencil_frame.render(_frame(output), stencil_frame.build(_frame(output)))
    assert 'issue_receipt_base64' not in stencils.bundle([output/'tracking.json'])['references'][0]
    with stencil_work_index.IssueIndex(root):
        pass
    (root/'issued'/'.tmp-interrupted').write_bytes(b'incomplete')
    with pytest.raises(ValueError, match='interrupted print issue'), stencil_work_index.IssueIndex(root):
        pass


def test_receipt_tamper_is_refused_before_session_retain(tmp_path):
    pytest.importorskip('PIL')
    root = tmp_path/'state'/'original-work'
    output = tmp_path/'frame'
    with stencil_work_index.IssueIndex(root) as index:
        stencil_frame.render(_frame(output), stencil_frame.build(_frame(output)), issue_index=index)
    receipt = json.loads((output/'issue-receipt.json').read_text())
    receipt['image_sha256'] = '0' * 64
    (output/'issue-receipt.json').write_text(json.dumps(receipt))
    with pytest.raises(ValueError, match='differs'):
        stencils.bundle([output/'tracking.json'])


def test_two_issued_copies_of_one_artwork_bind_separate_jobs(tmp_path):
    pytest.importorskip('PIL')
    root = tmp_path/'state'/'original-work'
    first, second = tmp_path/'first', tmp_path/'second'
    with stencil_work_index.IssueIndex(root) as index:
        stencil_frame.render(_frame(first), stencil_frame.build(_frame(first)), issue_index=index)
        stencil_frame.render(_frame(second), stencil_frame.build(_frame(second)), issue_index=index)
    one = stencils.bundle([first/'tracking.json'])
    two = stencils.bundle([second/'tracking.json'])
    first_mark = one['references'][0]['reference']['physical_instance_id']
    second_mark = two['references'][0]['reference']['physical_instance_id']
    assert first_mark != second_mark
    assert one['references'][0]['reference']['pattern_id'] == two['references'][0]['reference']['pattern_id']
    assert one['references'][0]['issue_receipt_base64'] != two['references'][0]['issue_receipt_base64']
    with pytest.raises(ValueError, match='patterns must be distinct'):
        stencils.bundle([first/'tracking.json', second/'tracking.json'])
