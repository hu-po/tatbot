from __future__ import annotations

import numpy as np
from tatbot_sim.human_rep.training import (
    PRIMARY_METRIC,
    admission_decision,
    build_dataset,
    dataset_manifest,
    evaluate_model,
    evaluate_predictions,
    heldout_combination_split,
    heuristic_predictions,
    sensitivity_by_axis,
    split_manifest,
    train_joint,
    train_modular,
)


def test_combination_split_is_disjoint_and_keeps_refusals():
    dataset = build_dataset(384, seed=17)
    split = heldout_combination_split(dataset)
    manifest = dataset_manifest(dataset, split, seed=17)
    splits = split_manifest(dataset, split)
    assert sum(manifest["split_counts"].values()) == 384
    assert manifest["refused_rows"] > 0
    assert splits["pairwise_combination_overlap"] == 0
    keys = {
        name: {dataset.records[int(index)]["combination_sha256"] for index in indices}
        for name, indices in split.items()
    }
    assert not keys["train"] & keys["dev"]
    assert not keys["train"] & keys["test"]
    assert not keys["dev"] & keys["test"]
    assert all(splits["splits"][name]["refused"] > 0 for name in split)


def test_refused_examples_remain_in_every_metric_denominator():
    dataset = build_dataset(256, seed=19)
    test = heldout_combination_split(dataset)["test"]
    predictions = heuristic_predictions(dataset, test)
    metrics = evaluate_predictions(dataset, test, predictions)
    assert metrics["rows"] == len(test)
    assert metrics["accepted_targets"] + metrics["refused_targets"] == len(test)
    assert metrics["refused_targets"] > 0
    assert metrics["all_rows_in_denominator"] is True
    assert metrics[PRIMARY_METRIC] == 1.0
    assert metrics["refusal_precision"] == 1.0
    assert metrics["refusal_recall"] == 1.0


def test_modular_and_joint_training_are_deterministic_and_joint_is_not_auto_admitted():
    dataset = build_dataset(384, seed=23)
    split = heldout_combination_split(dataset)
    modular = train_modular(dataset, split["train"])
    joint = train_joint(dataset, split["train"], rank=8)
    heuristic_metrics = evaluate_predictions(
        dataset,
        split["test"],
        heuristic_predictions(dataset, split["test"]),
    )
    heuristic_metrics["deterministic_repeat"] = True
    modular_metrics = evaluate_model(dataset, split["test"], modular)
    joint_metrics = evaluate_model(dataset, split["test"], joint)
    assert modular_metrics["deterministic_repeat"] is True
    assert joint_metrics["deterministic_repeat"] is True
    assert np.array_equal(
        modular.predict(dataset.features[split["test"]]),
        modular.predict(dataset.features[split["test"]]),
    )
    decision = admission_decision(
        {"heuristic": heuristic_metrics, "modular": modular_metrics, "joint": joint_metrics}
    )
    assert decision["joint_admitted"] is False
    assert decision["admitted_model"] == "heuristic"
    assert decision["disposition"] == "retain_strongest_simpler_baseline"


def test_sensitivity_retains_identity_pose_registration_and_support_axes():
    dataset = build_dataset(256, seed=29)
    test = heldout_combination_split(dataset)["test"]
    predictions = heuristic_predictions(dataset, test)
    sensitivity = sensitivity_by_axis(dataset, test, predictions)
    assert set(sensitivity) == {"identity", "pose", "registration", "support"}
    assert all(values for values in sensitivity.values())
    assert all(score == 1.0 for values in sensitivity.values() for score in values.values())
