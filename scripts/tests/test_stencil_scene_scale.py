"""Whole-scene acquisition preserves native UV correspondence after resizing."""
import json

import cv2
import numpy as np
from stencil_features import scene_features  # noqa: E402
from stencil_scene import StencilScene  # noqa: E402
from test_stencil_tracking import reference, render  # noqa: E402


def test_odd_size_resize_maps_pixel_centers_without_relaxing_geometry():
    descriptor = np.ones((1, 128), np.float32)
    class Detector:
        def detectAndCompute(self, image, mask):  # noqa: N802 - OpenCV detector protocol
            assert image.shape == (900, 1500)
            return [cv2.KeyPoint(120., 170., 8.)], descriptor
    keys, actual = scene_features(Detector(), np.zeros((1801, 3001), np.uint8))
    np.testing.assert_allclose(keys[0].pt, [(120.5)*3001/1500-.5, (170.5)*1801/900-.5], atol=1e-4)
    assert actual is descriptor


def test_two_small_patterns_in_large_cluttered_scene_keep_native_coordinates():
    rng = np.random.default_rng(51)
    image = np.clip(rng.normal(195, 15, (1668, 2960)), 0, 255).astype(np.uint8)
    truth = {}
    for seed, x, y in [('tatbot-42', 1800, 200), ('tatbot-43', 450, 950)]:
        patch, h = render(seed)
        if seed == 'tatbot-43':
            patch = np.clip(patch*.4+125, 0, 255).astype(np.uint8)
        image[y:y+480, x:x+640] = patch
        transform = np.array([[1., 0, x], [0, 1, y], [0, 0, 1]])
        truth[seed] = transform @ h
    scene = StencilScene([reference('tatbot-42'), reference('tatbot-43')], 'two-surfaces')
    observation = scene.observe(cv2.cvtColor(image, cv2.COLOR_GRAY2BGR), 1_000_000_000)
    for row in observation['stencils']:
        assert row['image_tracking_valid'], row['reason']
        assert row['pattern_id'] == json.loads(reference(row['seed']).read_bytes())['pattern_id']
        uv = np.asarray([p['reference_uv'] for p in row['landmarks']], np.float64)
        pixels = np.asarray([p['image_px'] for p in row['landmarks']])
        expected = cv2.perspectiveTransform(uv[None], truth[row['seed']])[0]
        assert np.percentile(np.linalg.norm(pixels-expected, axis=1), 95) < 3
        assert not row['geometry_valid'] and not row['motion_authority']
    lost = scene.observe(cv2.cvtColor(np.full_like(image, 180), cv2.COLOR_GRAY2BGR), 1_500_000_000)
    assert not any(row['image_tracking_valid'] for row in lost['stencils'])
