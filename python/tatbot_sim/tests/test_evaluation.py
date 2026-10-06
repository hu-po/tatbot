from __future__ import annotations

import json

import pytest
from tatbot_sim.evaluation import (
    DISCLAIMER,
    SimEvalError,
    build_evaluation_report,
    evaluate_dataset,
    write_evaluation_report,
)


def _dataset(tmp_path, *, dirty: bool = False, omit_score: bool = False):
    root = tmp_path / ("dirty" if dirty else "clean")
    eval_dir = root / "meta" / "eval"
    eval_dir.mkdir(parents=True)
    episodes = []
    for index, f1 in enumerate((0.9, 0.8, 0.7)):
        artifacts = {}
        for label in ("intended", "drawn", "overlay"):
            path = eval_dir / f"episode_{index:06d}_{label}.png"
            path.write_bytes(f"{index}:{label}".encode())
            artifacts[label] = str(path.relative_to(root))
        score = {
            "f1": f1,
            "precision": f1 - 0.02,
            "recall": f1 + 0.02,
            "iou": f1 - 0.1,
            "chamfer_drawn_to_intended_mm": 1.0 + index,
            "chamfer_intended_to_drawn_mm": 1.5 + index,
            "coverage_ratio": 1.0,
            "blank": False,
            "degenerate": False,
            "artifacts": artifacts,
        }
        episodes.append({
            "episode": index,
            "kind": "shape",
            "engaged": True,
            "interaction": {"frames": 100 + index},
            "ink": {"contact_s": 3.0 + index},
            "surface_profile": "flat" if index < 2 else "cylinder",
            "task": "draw generated design",
            **({} if omit_score and index == 1 else {"drawing_score": score}),
        })
    run_meta = {
        "schema_version": 2,
        "config": {"seed": 4200, "distribution": "paper-draw"},
        "tool": {"id": "lutin-ballpoint-dot"},
        "software": {
            "revision_start": "abc1234", "revision_end": "abc1234",
            "dirty_start": dirty, "dirty_end": dirty,
        },
        "episodes": episodes,
        "evaluation": {
            "schema": "tatbot.sim-eval-input/1", "producer": "expert",
        },
        "floor_clamped_step_fraction": 0.01,
    }
    (root / "meta" / "run_meta.json").write_text(json.dumps(run_meta))
    return root


def test_eval_report_aggregates_scores_ci_evidence_and_split(monkeypatch, tmp_path):
    monkeypatch.setattr(
        "tatbot_sim.evaluation.source_state",
        lambda: {"revision": "abc1234", "dirty": False},
    )
    dataset = evaluate_dataset(_dataset(tmp_path))
    report = build_evaluation_report(
        [dataset],
        checkpoint_id="expert",
        checkpoint_sha256=None,
        base_model_sha256=None,
        training_seed_ranges=((0, 1000),),
        created_at="2026-09-03T00:00:00Z",
    )
    assert report["disclaimer"] == DISCLAIMER
    assert report["interpretation"] == "screen-only"
    assert report["split"]["status"] == "held-out"
    assert report["metrics"]["f1"]["mean"] == pytest.approx(0.8)
    assert report["metrics"]["f1"]["ci95"] is not None
    assert len(report["episodes"][0]["artifacts"]["overlay"]["sha256"]) == 64
    assert report["datasets"][0]["surface_profiles"] == ["cylinder", "flat"]
    assert report["mechanical"]["chunk_rejections"] == 0
    output = tmp_path / "report"
    write_evaluation_report(output, report)
    assert (output / "report.json").is_file()
    assert (output / "scores.csv").read_text().count("\n") == 4
    assert "screening result only" in (output / "report.md").read_text()
    assert "## Evidence" in (output / "report.md").read_text()


def test_eval_blocks_contaminated_or_dirty_runs(monkeypatch, tmp_path):
    monkeypatch.setattr(
        "tatbot_sim.evaluation.source_state",
        lambda: {"revision": "abc1234", "dirty": False},
    )
    contaminated = build_evaluation_report(
        [evaluate_dataset(_dataset(tmp_path))],
        checkpoint_id="expert",
        checkpoint_sha256=None,
        base_model_sha256=None,
        training_seed_ranges=((4200, 4300),),
    )
    assert contaminated["split"]["status"] == "contaminated"
    assert not contaminated["comparison_eligible"]
    assert "training_seed_overlap" in contaminated["comparison_blocks"]

    other = tmp_path / "other"
    dirty = build_evaluation_report(
        [evaluate_dataset(_dataset(other, dirty=True))],
        checkpoint_id="expert", checkpoint_sha256=None, base_model_sha256=None,
    )
    assert "dirty_dataset_source" in dirty["comparison_blocks"]


def test_eval_fails_closed_when_scores_or_visual_evidence_are_missing(tmp_path):
    with pytest.raises(SimEvalError, match="missing_drawing_score"):
        evaluate_dataset(_dataset(tmp_path, omit_score=True))

    root = _dataset(tmp_path / "other")
    (root / "meta" / "eval" / "episode_000000_overlay.png").unlink()
    with pytest.raises(SimEvalError, match="missing_visual_artifact"):
        evaluate_dataset(root)


@pytest.mark.parametrize("split,forged,eligible", [("train", False, False), ("test", False, True), ("test", True, False)])
def test_artwork_evaluation_uses_frozen_families_not_just_new_seeds(tmp_path, monkeypatch, split, forged, eligible):
    from tatbot_sim.inkmap.collection import collection_entries
    monkeypatch.setattr("tatbot_sim.evaluation.source_state", lambda: {"revision": "abc1234", "dirty": False})
    entry = collection_entries(split)[0]
    root = _dataset(tmp_path)
    path = root / "meta/run_meta.json"
    meta = json.loads(path.read_text())
    for episode in meta["episodes"]:
        episode["artwork"] = {"design_id": entry["id"], "family": entry["family"], "split": split,
                              "source_sha256": "0" * 64 if forged else entry["sha256"]}
    path.write_text(json.dumps(meta))
    report = build_evaluation_report([evaluate_dataset(root)], checkpoint_id="expert",
                                     checkpoint_sha256=None, base_model_sha256=None)
    assert report["comparison_eligible"] is eligible
    if forged:
        assert "artwork_provenance_unverified" in report["comparison_blocks"]
