from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
import torch
from tatbot_sim.env import TatbotDrawEnv
from tatbot_sim.generate import _occlude_observations
from tatbot_sim.planning import (
    EPISODE_VARIANT_OUTCOMES,
    BatchPlan,
    apply_episode_variant,
)


def _plan() -> BatchPlan:
    batch, steps = 2, 10
    targets = np.zeros((batch, steps, 3), dtype=np.float32)
    targets[..., 0] = np.linspace(0, 0.09, steps)
    surface_points = targets.copy()
    normals = np.zeros_like(targets)
    normals[..., 2] = 1
    return BatchPlan(
        n_app=2,
        q_raised=None,
        draw_horizon=steps,
        targets=targets,
        pen_normals=normals.copy(),
        surface_points=surface_points,
        surface_normals=normals,
        lean_profiles=[np.zeros((steps, 2)) for _ in range(batch)],
        kinds=["language"] * batch,
        tasks=["draw"] * batch,
        paths=[[] for _ in range(batch)],
        programs=[None] * batch,
        lengths=np.asarray([12, 10], dtype=np.int32),
    )


@pytest.mark.parametrize("variant,outcome", EPISODE_VARIANT_OUTCOMES.items())
def test_variants_preserve_answer_key_and_declare_outcome(variant, outcome):
    plan = _plan()
    original_targets = plan.targets.copy()
    original_lengths = plan.lengths.copy()
    result = apply_episode_variant(plan, variant)
    np.testing.assert_array_equal(result.intended_targets, original_targets)
    np.testing.assert_array_equal(result.intended_lengths, original_lengths)
    assert result.variant_ids == [variant, variant]
    assert result.expected_outcomes == [outcome, outcome]


def test_failure_variants_change_only_the_declared_execution_dimension():
    missed = apply_episode_variant(_plan(), "missed-stroke")
    distance = np.linalg.norm(missed.targets - missed.intended_targets, axis=-1)
    np.testing.assert_allclose(distance, 0.006, atol=1e-7)
    np.testing.assert_array_equal(missed.lengths, missed.intended_lengths)

    partial = apply_episode_variant(_plan(), "partial-coverage")
    lift = partial.targets[..., 2] - partial.intended_targets[..., 2]
    np.testing.assert_allclose(lift[:, :3], 0)
    np.testing.assert_allclose(lift[:, 3:7], 0.015)
    np.testing.assert_allclose(lift[:, 7:], 0)

    interrupted = apply_episode_variant(_plan(), "interrupted")
    np.testing.assert_array_equal(interrupted.lengths, np.asarray([7, 6]))
    np.testing.assert_array_equal(interrupted.targets, interrupted.intended_targets)


def test_unknown_variant_is_refused_without_mutating_plan():
    plan = _plan()
    original = plan.targets.copy()
    with pytest.raises(ValueError, match="unknown episode variant"):
        apply_episode_variant(plan, "invented")
    np.testing.assert_array_equal(plan.targets, original)
    assert plan.intended_targets is None


def test_stencil_is_a_separate_appearance_layer():
    fake = SimpleNamespace(
        device=torch.device("cpu"),
        num_envs=2,
        _sheet_base=torch.ones((2, 4, 5, 3)),
        _stencil_base=None,
        _stencil_fraction=np.zeros(2, dtype=np.float32),
        ink_field=None,
    )
    coverage = np.zeros((2, 4, 5), dtype=np.float32)
    coverage[0, :2] = 1
    coverage[1, :, :1] = 0.5
    TatbotDrawEnv.set_stencil(fake, coverage)
    assert fake._stencil_base is not None
    np.testing.assert_allclose(fake._stencil_fraction, [0.5, 0.1])
    assert torch.all(fake._stencil_base <= fake._sheet_base)
    # visible = stencil x (1 - ink): pigment laid over the guide hides it,
    # pigment elsewhere does not, and the guide area itself never changes.
    ink = torch.zeros((2, 4, 5))
    ink[0, :2] = 1.0          # env 0: fully inked over its stencil rows
    ink[1, :, 4] = 1.0        # env 1: inked away from its stencil column
    np.testing.assert_allclose(fake._stencil_fraction, [0.5, 0.1])
    np.testing.assert_allclose(TatbotDrawEnv.stencil_visible_fraction(fake, ink), [0.0, 0.1], atol=1e-7)
    with pytest.raises(ValueError, match="differs from stencil"):
        TatbotDrawEnv.stencil_visible_fraction(fake, torch.zeros((2, 3, 5)))
    TatbotDrawEnv.set_stencil(fake, None)
    assert fake._stencil_base is None
    np.testing.assert_array_equal(fake._stencil_fraction, [0, 0])
    np.testing.assert_array_equal(TatbotDrawEnv.stencil_visible_fraction(fake, ink), [0, 0])


def test_partial_coverage_lift_is_checked_against_the_validated_ceiling():
    plan = _plan()
    with pytest.raises(ValueError, match="exceeds the 10 mm tool ceiling"):
        apply_episode_variant(plan, "partial-coverage", tool_ceiling=0.010)
    assert plan.intended_targets is None
    lifted = apply_episode_variant(_plan(), "partial-coverage", tool_ceiling=0.020)
    assert lifted.expected_outcomes == ["partial", "partial"]


def test_camera_occlusion_is_exact_labeled_and_non_aliasing():
    source = np.full((2, 100, 200, 3), 255, dtype=np.uint8)
    frames = {"upper": source, "lower": source.copy()}
    actual = _occlude_observations(frames, 0.15)
    assert actual == pytest.approx(0.15015)
    quarter = {"upper": source.copy()}
    assert _occlude_observations(quarter, 0.25) == pytest.approx(0.25)
    assert np.all(source == 255)
    for values in frames.values():
        hidden = np.all(values == [20, 23, 28], axis=-1)
        assert hidden.mean() == pytest.approx(actual)


def test_camera_occlusion_refuses_one_ambiguous_cross_camera_label():
    frames = {
        "upper": np.zeros((1, 100, 200, 3), dtype=np.uint8),
        "lower": np.zeros((1, 101, 200, 3), dtype=np.uint8),
    }
    with pytest.raises(RuntimeError, match="one labelable area"):
        _occlude_observations(frames)
