"""The physical print code is decoded from pixels, never inferred from a label."""

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import pytest
import stencil_frame
import stencil_instance
import stencil_reference
from stencil_tracking import StencilTracker


def marked(tmp_path):
    parser = argparse.ArgumentParser()
    stencil_frame.add_arguments(parser)
    args = parser.parse_args(['--output', str(tmp_path), '--instance-id',
                              '0123456789abcdef01234567'])
    stencil_frame.validate(args)
    args.output = Path(args.output)
    stencil_frame.render(args, stencil_frame.build(args))
    reference = args.output/'tracking.json'
    manifest, image_path = stencil_reference.load(reference)
    image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    mark = dict(manifest['instance_mark'], page_mm=manifest['page_mm'])
    transform = np.diag([image.shape[1], image.shape[0], 1.])
    return reference, manifest, image, mark, transform


def test_instance_code_refuses_ambiguous_missing_and_too_small_image(tmp_path):
    reference, manifest, image, mark, transform = marked(tmp_path)
    expected = manifest['physical_instance_id']
    assert stencil_instance.decode(image, transform, mark) == (expected, 'instance_mark_decoded')

    mirrored = cv2.flip(image, 1)
    reflection = np.array([[-image.shape[1], 0, image.shape[1]-1],
                           [0, image.shape[0], 0], [0, 0, 1.]])
    assert stencil_instance.decode(mirrored, reflection, mark) == (expected, 'instance_mark_decoded')

    low_resolution = cv2.resize(image, (100, 150), interpolation=cv2.INTER_AREA)
    decoded, reason = stencil_instance.decode(low_resolution, np.diag([100., 150., 1.]), mark)
    assert decoded is None and reason == 'instance_mark_too_small'

    module_px = mark['module_mm']*image.shape[1]/mark['page_mm'][0]
    x = round((mark['x_mm']+1.5*mark['module_mm'])*image.shape[1]/mark['page_mm'][0])
    y = round((mark['y_mm']+1.5*mark['module_mm'])*image.shape[0]/mark['page_mm'][1])
    ambiguous = image.copy()
    radius = round(module_px*.35)
    ambiguous[y-radius:y+radius+1, x-radius:x+radius+1] = 127
    decoded, reason = stencil_instance.decode(ambiguous, transform, mark)
    assert decoded is None and reason == 'instance_mark_ambiguous'

    missing = image.copy()
    x0 = round(mark['x_mm']*image.shape[1]/mark['page_mm'][0])
    y0 = round(mark['y_mm']*image.shape[0]/mark['page_mm'][1])
    missing[y0:y0+round(6*module_px), x0:x0+round(34*module_px)] = 255
    assert stencil_instance.decode(missing, transform, mark)[0] is None

    forged = dict(manifest, physical_instance_id='fedcba9876543210fedcba98')
    forged['reference_id'] = stencil_reference.reference_digest(forged)
    reference.write_text(json.dumps(forged))
    with pytest.raises(ValueError, match='declared print-instance mark'):
        StencilTracker([reference], 'fixture')
