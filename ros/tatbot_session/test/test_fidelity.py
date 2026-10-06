"""Synthetic observations distinguish registration, missing paths, spill and unseen paper."""
import copy

import cv2
import numpy as np
import pytest
from tatbot_session.fidelity import _expected, measure

EXTENT = (-.04, -.04, .04, .04)
PPM = 4
BORDER = {'scale': [1, 1], 'theta_rad': 0, 'tx_m': 0, 'ty_m': 0, 'sigma': [.00002, .00002, .0001]}


@pytest.fixture
def drawing():
    p = {'resources': [{'id': 'B', 'tool': {'line_width_m': .0005, 'line_width_status': 'measured'}}], 'design': {'at_m': [0, 0]},
         'research': {'slot_m': [.026, .026]}, 'ops': [{'op': 'stroke', 'resource_id': 'B', 'closed': False, 'generation_width_m': .0005,
          'src': {'placement': 'p', 'path': 'one'}, 'points_m': [[-.007, -.006], [.006, -.003], [.004, .008], [-.004, .003]]}]}
    mask, _, _ = _expected(p, (320, 320), EXTENT, PPM)
    image = np.full((320, 320), 255, np.uint8)
    image[mask] = 0
    return p, image


def score(drawing, image=None, border=BORDER):
    program, original = drawing
    return measure(original if image is None else image, program, border, [0, 0], EXTENT, PPM)


def test_perfect_and_shifted_shape_separate_placement_from_fidelity(drawing):
    good = score(drawing)
    assert good['valid'] and good['missing_fraction'] == good['spill_fraction'] == 0
    assert good['iou'] == 1 and good['placement_error_m'] == [0, 0]
    shifted = cv2.warpAffine(drawing[1], np.float32([[1, 0, 4], [0, 1, -2]]), (320, 320), borderValue=255)
    result = score(drawing, shifted)
    assert result['valid'] and result['iou'] == 1
    np.testing.assert_allclose(result['placement_error_m'], [.001, .0005])
    assert result['width_resolved']


def test_missing_ink_and_thick_smear_do_not_win(drawing):
    missing = drawing[1].copy()
    missing[130:143, 148:180] = 255
    result = score(drawing, missing)
    assert result['missing_fraction'] > 0 and result['iou'] < 1
    thick = cv2.erode(drawing[1], np.ones((7, 7), np.uint8))
    smear = score(drawing, thick)
    assert smear['spill_fraction'] > .3 and smear['spill_area_mm2'] > 0 and smear['iou'] < .6
    assert smear['observed_width_proxy_mm'] > result['observed_width_proxy_mm']


def test_other_slot_marks_are_excluded_and_unseen_paper_is_invalid(drawing):
    other = drawing[1].copy()
    other[230:260, 220:255] = 0
    assert score(drawing, other) == score(drawing)
    unseen = drawing[1].copy()
    unseen[:, :150] = 0
    assert not score(drawing, unseen)['valid']
    assert not score(drawing, border=None)['valid']
    blank = np.full_like(drawing[1], 255)
    assert not score(drawing, blank)['valid']


def test_independent_border_translation_corrects_camera_error(drawing):
    shifted = cv2.warpAffine(drawing[1], np.float32([[1, 0, 4], [0, 1, 0]]), (320, 320), borderValue=255)
    border = {**BORDER, 'tx_m': -.001}
    measured = score(drawing, shifted, border)
    assert measured['valid'] and measured['iou'] == 1
    assert measured['placement_error_m'] == [0, 0]


def test_computational_chunks_are_not_counted_as_repeated_paths(drawing):
    program, _ = drawing
    split = copy.deepcopy(program)
    op = split['ops'][0]
    tail = copy.deepcopy(op)
    tail['points_m'] = op['points_m'][1:]
    op['points_m'] = op['points_m'][:2]
    split['ops'].append(tail)
    first = _expected(program, (320, 320), EXTENT, PPM)
    second = _expected(split, (320, 320), EXTENT, PPM)
    np.testing.assert_array_equal(first[0], second[0])
    assert first[2] == second[2] == 0
    tail['src']['path'] = 'repeat'
    assert _expected(split, (320, 320), EXTENT, PPM)[2] > 0


def test_degenerate_registration_is_inconclusive(drawing):
    assert not score(drawing, border={**BORDER, 'scale': [0, 0]})['valid']
    assert not score(drawing, border={**BORDER, 'tx_m': float('nan')})['valid']


def _shade(image, floor):
    """A soft shadow edge across the slot, as the arm cast over a wrist view (2026-09-28): full light on the
    right, `floor` of it on the left, a 6 mm penumbra between."""
    x = np.arange(image.shape[1], dtype=np.float32)
    light = floor + (1 - floor) * np.clip((x - 160) / (6 * PPM) + .5, 0, 1)
    return (image.astype(np.float32) * light[None, :]).astype(np.uint8)


def test_a_shadow_across_the_slot_is_not_ink_and_a_dark_one_is_refused(drawing):
    clean = score(drawing)
    shaded = score(drawing, _shade(drawing[1], .75))
    assert shaded['valid'] and shaded['spill_area_mm2'] == clean['spill_area_mm2'] == 0
    assert shaded['missing_fraction'] == clean['missing_fraction'] == 0
    dark = score(drawing, _shade(drawing[1], .5))
    assert not dark['valid'] and 'shadow' in dark['reason']


def test_placement_beyond_two_millimetres_is_measured_not_mistaken_for_missing_ink(drawing):
    """The right arm drew 3.7 mm low on the print; a 2 mm search reported 61% of that drawing missing."""
    moved = cv2.warpAffine(drawing[1], np.float32([[1, 0, 6], [0, 1, 16]]), (320, 320), borderValue=255)
    result = score(drawing, moved)
    assert result['valid'] and result['missing_fraction'] == result['spill_fraction'] == 0
    np.testing.assert_allclose(result['placement_error_m'], [.0015, -.004])


def test_blurred_mid_grey_ballpoint_lines_are_ink(drawing):
    """At ~3.7 px/mm a 0.5 mm line reads 0.54-0.79 of the paper around it, not black."""
    faint = cv2.GaussianBlur(np.where(drawing[1] == 0, 140, 235).astype(np.uint8), (0, 0), 1.0)
    result = score(drawing, faint)
    assert result['valid'] and result['missing_fraction'] < .05 and result['spill_fraction'] < .05


def test_resource_geometry_uses_one_alignment_without_claiming_pigment(drawing):
    program, image = copy.deepcopy(drawing)
    program['resources'].append({**program['resources'][0], 'id': 'R'})
    second = copy.deepcopy(program['ops'][0])
    second['resource_id'] = 'R'
    second['points_m'] = [[x*.5+.005, y] for x, y in second['points_m']]
    program['ops'][0]['points_m'] = [[x*.5-.005, y] for x, y in program['ops'][0]['points_m']]
    program['ops'].append(second)
    mask, _, overlap = _expected(program, image.shape, EXTENT, PPM)
    assert overlap == 0
    image.fill(255)
    image[mask] = 0
    measured = score((program, image))
    for item in measured['per_resource'].values():
        assert item['color_accuracy'] is None
        assert item['metrics']['placement_error_m'] == measured['placement_error_m']


def test_overlapping_resources_remain_unattributable(drawing):
    program, image = copy.deepcopy(drawing)
    program['resources'].append({**program['resources'][0], 'id': 'R'})
    program['ops'].append({**copy.deepcopy(program['ops'][0]), 'resource_id': 'R'})
    assert _expected(program, image.shape, EXTENT, PPM)[2] > 0
    result = score((program, image))
    assert result['valid'] and result['iou'] == 1
    assert all(v['metrics'] is None and v['unique_footprint_fraction'] == 0 for v in result['per_resource'].values())


def test_resource_inspection_without_research_uses_page_clear_area(drawing):
    program, image = copy.deepcopy(drawing)
    del program['research']
    program['page'] = {'clear_m': [.026, .026]}
    assert score((program, image))['iou'] == 1
