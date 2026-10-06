"""Print registration must separate placement from faint ink and camera pose error."""
import json

import cv2
import numpy as np
import pytest
import stencil_coded
import stencil_coded_live as coded
from tatbot_session import inspect as ins


@pytest.fixture(scope='module')
def drawing(tmp_path_factory):
    reference = stencil_coded.generate('inspect-ink', tmp_path_factory.mktemp('inspect')) / 'tracking.json'
    image = np.full((480, 640, 3), 205, np.uint8)
    image[90:390, 220:420] = cv2.resize(cv2.imread(str(reference.with_name('stencil.png'))), (200, 300))
    plan = np.array([[-.005, -.025], [.005, -.025], [.005, -.015], [-.005, -.015], [-.005, -.025]])
    pixels = np.c_[(plan[:, 0] + .005 + .05) * 2000 + 220, (.075 - plan[:, 1] + .006) * 2000 + 90]
    cv2.polylines(image, [np.rint(pixels).astype(np.int32)], False, (45, 45, 45), 2)
    program = {'ops': [{'op': 'stroke', 'closed': True, 'points_m': plan.tolist()}],
               'page': {'clear_m': [.062, .112]}}
    return reference, image, program


def test_decoded_print_reports_placement_shape_and_sampling(drawing):
    reference, image, program = drawing
    measured, page = coded.coded_print_measure(image, program, reference, ins.ink_alignment, decode_s=10)
    assert measured['valid'] and measured['identity_verified'] and page is not None
    assert measured['bits_disagree'] == 0 and measured['bits_agree'] >= 8
    assert measured['placement_m'] == pytest.approx([.005, -.006], abs=2 * max(measured['native_pixel_mm']) / 1000)
    assert measured['ink']['p95_gap_m'] < .001
    assert measured['uncovered_plan_fraction'] < .1
    assert not measured['dimensions_measured']
    assert max(measured['native_pixel_mm']) > measured['raster_pixel_mm']


def test_wrong_print_is_unavailable_before_scoring(drawing):
    reference, image, program = drawing
    measured, page = coded.coded_print_measure(image, {**program, 'diagnostic': {'physical_print_id': 'different'}},
                                              reference, ins.ink_alignment)
    assert not measured['valid'] and page is None
    assert 'identity differs' in measured['reason']


def test_conflicting_bits_are_unavailable_even_with_a_decode(drawing, monkeypatch):
    reference, image, program = drawing
    monkeypatch.setattr('stencil_coded_tracker.verify_bits', lambda *args: (100, 1))
    measured, page = coded.coded_print_measure(image, program, reference, ins.ink_alignment, decode_s=10)
    assert not measured['valid'] and page is None
    assert 'conflicts' in measured['reason']


def test_dark_occlusion_cannot_turn_a_forward_match_into_measured_ink(drawing):
    reference, image, program = drawing
    blocked = image.copy()
    blocked[260:315, 302:358] = 20

    def false_match(*args):
        return {'found': True, 'dx_m': .005, 'dy_m': -.006, 'coverage': 1.0}

    measured, page = coded.coded_print_measure(blocked, program, reference, false_match, decode_s=10)
    assert measured['identity_verified'] and page is not None
    assert measured['ink']['found'] and not measured['valid']
    assert 'brightness' in measured['reason'] or 'shadow' in measured['reason']


def test_faint_complete_square_is_detected_separately_from_placement(drawing):
    reference, image, program = drawing
    faint = image.copy()
    region = faint[270:310, 310:350]
    region[(region == 45).all(axis=2)] = 190
    measured, _ = coded.coded_print_measure(faint, program, reference, ins.ink_alignment, decode_s=10)
    assert measured['valid'] and measured['uncovered_plan_fraction'] < .1
    assert measured['ink']['p95_gap_m'] < .001


def test_disagreeing_views_preserve_candidates_without_claiming_placement(drawing, tmp_path, monkeypatch):
    reference, image, program = drawing
    manifest = json.loads(reference.read_text())
    monkeypatch.setattr('stencil_reference.observer_references', lambda: reference.parent.parent)
    pattern = reference.parent.name
    monkeypatch.setattr('stencil_reference.load', lambda path: (manifest, reference.with_name('stencil.png')))
    rows = iter([{'valid': True, 'placement_m': [0., 0.], 'heldout_p95_mm': .3},
                 {'valid': True, 'placement_m': [0., .006], 'heldout_p95_mm': .3}])
    monkeypatch.setattr(coded, 'coded_print_measure', lambda *args, **kwargs: (next(rows), None))
    for n in range(2):
        cv2.imwrite(str(tmp_path / f'pose{n}-0.png'), image)
    result = coded.inspect_ink(tmp_path, [{'frames': 1}, {'frames': 1}], program, pattern, ins.ink_alignment)
    assert all(not row['valid'] for row in result['print_coordinates'])
    assert 'unavailable' in result['coded_summary']
