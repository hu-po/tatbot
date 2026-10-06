from __future__ import annotations

import numpy as np
import pytest
from tatbot_sim.inkmap.parity_evidence import build_surface_cases, mask_metrics


def test_surface_evidence_cases_have_fixed_denominators_and_wrap_refusal():
    cases = build_surface_cases()
    assert len(cases) == 9
    assert sum(item["expected_status"] == "accepted" for item in cases) == 8
    assert sum(len(item["points_m"]) for item in cases if item["expected_status"] == "accepted") == 10_847
    assert cases[-1]["id"] == "complete-wrap-left-forearm"
    assert cases[-1]["expected_status"] == "rejected"


def test_mask_metrics_cover_exact_empty_and_one_pixel_boundary_cases():
    empty = np.zeros((9, 9), dtype=bool)
    assert mask_metrics(empty, empty)["status"] == "empty_match"
    square = empty.copy()
    square[2:7, 2:7] = True
    exact = mask_metrics(square, square)
    assert exact["mask_iou"] == 1
    assert exact["boundary_distance_p95_px"] == 0
    shifted = np.roll(square, 1, axis=1)
    moved = mask_metrics(square, shifted)
    assert moved["mask_iou"] == pytest.approx(20 / 30)
    assert moved["boundary_distance_p95_px"] == pytest.approx(1)
