"""Aggregate exact-design simulation scores into a reproducible screen report."""

from __future__ import annotations

import csv
import json
import math
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from tatbot_contracts.digest import sha256_file

from tatbot_sim.repo import repo_root, source_state

EVAL_SCHEMA = "tatbot.sim-eval/1"
DISCLAIMER = (
    "Simulation evaluation is a screening result only. It does not authorize "
    "powered motion or human contact."
)
SCORE_FIELDS = (
    "f1",
    "precision",
    "recall",
    "iou",
    "chamfer_drawn_to_intended_mm",
    "chamfer_intended_to_drawn_mm",
    "coverage_ratio",
)


class SimEvalError(ValueError):
    def __init__(self, code: str, message: str):
        super().__init__(f"{code}: {message}")
        self.code = code


@dataclass(frozen=True)
class DatasetEvaluation:
    root: Path
    run_meta: dict
    run_meta_sha256: str
    rows: tuple[dict, ...]


def _safe_artifact(root: Path, value: object, label: str) -> tuple[str, str]:
    if not isinstance(value, str) or not value:
        raise SimEvalError("missing_visual_artifact", f"{label} path is missing")
    relative = Path(value)
    if relative.is_absolute() or ".." in relative.parts:
        raise SimEvalError("unsafe_visual_artifact", f"{label} must be relative to the dataset")
    path = root / relative
    if not path.is_file():
        raise SimEvalError("missing_visual_artifact", f"{path} does not exist")
    return relative.as_posix(), sha256_file(path)


def evaluate_dataset(root: Path) -> DatasetEvaluation:
    root = Path(root).expanduser().resolve()
    meta_path = root / "meta" / "run_meta.json"
    if not meta_path.is_file():
        raise SimEvalError("missing_run_meta", f"{meta_path} does not exist")
    try:
        run_meta = json.loads(meta_path.read_text())
    except json.JSONDecodeError as exc:
        raise SimEvalError("invalid_run_meta", f"{meta_path}: {exc}") from exc
    episodes = run_meta.get("episodes")
    if not isinstance(episodes, list) or not episodes:
        raise SimEvalError("missing_episodes", f"{meta_path} has no episodes")
    evaluation = run_meta.get("evaluation")
    if not isinstance(evaluation, dict) or evaluation.get("schema") != "tatbot.sim-eval-input/1":
        raise SimEvalError(
            "missing_eval_provenance",
            f"{meta_path} has no tatbot.sim-eval-input/1 producer contract",
        )
    rows = []
    for episode in episodes:
        if episode.get("kind") in ("erase", "dip"):
            continue
        score = episode.get("drawing_score")
        if not isinstance(score, dict):
            raise SimEvalError(
                "missing_drawing_score",
                f"{root}: episode {episode.get('episode')} was not generated with --judge",
            )
        numeric = {}
        for field in SCORE_FIELDS:
            value = score.get(field)
            if value is None and bool(score.get("blank")) and field.startswith("chamfer_"):
                numeric[field] = None
                continue
            if not isinstance(value, (int, float)):
                raise SimEvalError(
                    "invalid_drawing_score",
                    f"{root}: episode {episode.get('episode')} has invalid {field}",
                )
            if math.isfinite(value):
                numeric[field] = float(value)
            elif bool(score.get("blank")) and field.startswith("chamfer_"):
                numeric[field] = None
            else:
                raise SimEvalError(
                    "invalid_drawing_score",
                    f"{root}: episode {episode.get('episode')} has invalid {field}",
                )
        artifacts = score.get("artifacts")
        if not isinstance(artifacts, dict):
            raise SimEvalError(
                "missing_visual_artifact",
                f"{root}: episode {episode.get('episode')} has no intended/drawn/overlay record",
            )
        artifact_records = {}
        for label in ("intended", "drawn", "overlay"):
            path, digest = _safe_artifact(root, artifacts.get(label), label)
            artifact_records[label] = {"path": path, "sha256": digest}
        video_records = {}
        videos = episode.get("videos", {})
        if not isinstance(videos, dict):
            raise SimEvalError(
                "invalid_visual_artifact",
                f"{root}: episode {episode.get('episode')} videos must be an object",
            )
        for camera, value in sorted(videos.items()):
            path, digest = _safe_artifact(root, value, f"{camera} video")
            video_records[camera] = {"path": path, "sha256": digest}
        interaction = episode.get("interaction", {})
        rows.append({
            "dataset": str(root),
            "episode": int(episode["episode"]),
            "kind": str(episode.get("kind")),
            **numeric,
            "blank": bool(score.get("blank")),
            "engaged": bool(episode.get("engaged")),
            "contact_s": float(episode.get("ink", {}).get("contact_s", 0.0)),
            "interaction_frames": int(interaction.get("frames", 0)),
            "surface_profile": episode.get("surface_profile"),
            "task": episode.get("task"),
            "steps_planned": episode.get("steps_planned"),
            "steps_executed": episode.get("steps_executed"),
            "artifacts": artifact_records,
            "videos": video_records,
        })
    if not rows:
        raise SimEvalError("no_scorable_episodes", f"{root} contains no deposition episodes")
    return DatasetEvaluation(root, run_meta, sha256_file(meta_path), tuple(rows))


def _bootstrap_summary(values: list[float], *, seed: int) -> dict:
    if not values:
        return {
            "n": 0, "mean": None, "stddev": None,
            "minimum": None, "maximum": None, "ci95": None,
        }
    array = np.asarray(values, dtype=np.float64)
    result = {
        "n": len(values),
        "mean": float(array.mean()),
        "stddev": float(array.std(ddof=1)) if len(array) > 1 else 0.0,
        "minimum": float(array.min()),
        "maximum": float(array.max()),
        "ci95": None,
    }
    if len(array) >= 3:
        rng = np.random.default_rng(seed)
        indices = rng.integers(0, len(array), size=(10_000, len(array)))
        means = array[indices].mean(axis=1)
        result["ci95"] = [float(value) for value in np.quantile(means, [0.025, 0.975])]
    return result


def _seed_values(run_meta: dict) -> set[int]:
    values = set()
    seed = run_meta.get("config", {}).get("seed")
    if isinstance(seed, int) and not isinstance(seed, bool):
        values.add(seed)
    for episode in run_meta.get("episodes", []):
        episode_seed = episode.get("seed")
        if isinstance(episode_seed, int) and not isinstance(episode_seed, bool):
            values.add(episode_seed)
    return values


def _is_contaminated(seed_values: set[int], training_seed_ranges: tuple[tuple[int, int], ...]) -> bool:
    return any(low <= seed <= high for seed in seed_values for low, high in training_seed_ranges)


def build_evaluation_report(
    datasets: list[DatasetEvaluation],
    *,
    checkpoint_id: str,
    checkpoint_sha256: str | None,
    base_model_sha256: str | None,
    training_seed_ranges: tuple[tuple[int, int], ...] = (),
    created_at: str | None = None,
) -> dict:
    if not datasets:
        raise SimEvalError("missing_dataset", "at least one dataset is required")
    if any(low > high for low, high in training_seed_ranges):
        raise SimEvalError("invalid_training_seed_range", "training seed range minimum exceeds maximum")
    for label, value in (
        ("checkpoint_sha256", checkpoint_sha256), ("base_model_sha256", base_model_sha256),
    ):
        if value is not None and (
            len(value) != 64 or any(character not in "0123456789abcdef" for character in value)
        ):
            raise SimEvalError(
                "invalid_checkpoint_digest", f"{label} must be 64 lowercase hex characters",
            )
    producers = {dataset.run_meta["evaluation"].get("producer") for dataset in datasets}
    if len(producers) != 1 or None in producers:
        raise SimEvalError("mixed_eval_producer", f"dataset producers differ: {sorted(map(str, producers))}")
    producer = next(iter(producers))
    if producer == "expert" and checkpoint_id != "expert":
        raise SimEvalError(
            "checkpoint_provenance_mismatch",
            "expert-generated inputs must use checkpoint-id expert",
        )
    if producer == "policy":
        recorded = {dataset.run_meta["evaluation"].get("checkpoint_id") for dataset in datasets}
        if recorded != {checkpoint_id}:
            raise SimEvalError(
                "checkpoint_provenance_mismatch",
                f"report checkpoint {checkpoint_id!r} does not match dataset metadata {recorded!r}",
            )
        recorded_sha = {
            dataset.run_meta["evaluation"].get("checkpoint_sha256") for dataset in datasets
        }
        if checkpoint_sha256 is None or recorded_sha != {checkpoint_sha256}:
            raise SimEvalError(
                "checkpoint_provenance_mismatch",
                "policy input checkpoint digest is missing or differs from the report",
            )
    tools = {
        dataset.run_meta.get("tool", {}).get("tool_id")
        or dataset.run_meta.get("tool", {}).get("id")
        for dataset in datasets
    }
    distributions = {
        dataset.run_meta.get("config", {}).get("distribution") for dataset in datasets
    }
    if len(tools) != 1:
        raise SimEvalError("mixed_tools", f"dataset tools differ: {sorted(map(str, tools))}")
    if len(distributions) != 1:
        raise SimEvalError(
            "mixed_distributions", f"dataset distributions differ: {sorted(map(str, distributions))}",
        )
    rows = [row for dataset in datasets for row in dataset.rows]
    seed_values = set().union(*(_seed_values(dataset.run_meta) for dataset in datasets))
    contaminated = _is_contaminated(seed_values, training_seed_ranges)
    dataset_dirty = any(
        dataset.run_meta.get("software", {}).get(field) is not False
        for dataset in datasets
        for field in ("dirty_start", "dirty_end")
    )
    if created_at is None:
        created_at = datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")
    metrics = {
        field: _bootstrap_summary(
            [row[field] for row in rows if row[field] is not None], seed=index,
        )
        for index, field in enumerate(SCORE_FIELDS)
    }
    datasets_record = []
    for dataset in datasets:
        run_meta = dataset.run_meta
        scenario = run_meta.get("scenario")
        datasets_record.append({
            "path": str(dataset.root),
            "run_meta_sha256": dataset.run_meta_sha256,
            "episodes": len(dataset.rows),
            "distribution": run_meta.get("config", {}).get("distribution"),
            "seed": run_meta.get("config", {}).get("seed"),
            "tool_id": (
                run_meta.get("tool", {}).get("tool_id") or run_meta.get("tool", {}).get("id")
            ),
            "tool_geometry_basis": run_meta.get("tool", {}).get("geometry_basis"),
            "tool_qualification": run_meta.get("tool", {}).get("qualification"),
            "tool_geometry_warnings": run_meta.get("tool", {}).get("geometry_warnings", []),
            "software": run_meta.get("software"),
            "scenario": scenario,
            "surface_profiles": sorted({
                str(episode.get("surface_profile"))
                for episode in run_meta.get("episodes", [])
                if episode.get("surface_profile") is not None
            }),
            "policy_client": run_meta.get("evaluation", {}).get("client"),
            "worker_protocol": run_meta.get("evaluation", {}).get("protocol"),
            "chunk_rejections": run_meta.get("evaluation", {}).get("chunk_rejections", []),
        })
    harness_state = source_state()
    harness_dirty = harness_state["dirty"] is not False
    artworks = [ep["artwork"] for dataset in datasets for ep in dataset.run_meta.get("episodes", []) if "artwork" in ep]
    artwork_splits = {art.get("split") for art in artworks}
    training_artwork = bool(artworks) and bool(artwork_splits - {"validation", "test", "calibration"})
    calibration_control = "calibration" in artwork_splits
    invalid_artwork = False
    if artworks:
        from tatbot_sim.inkmap.collection import collection_entries
        collection = {e["id"]: e for e in collection_entries()}
        for artwork in artworks:
            if artwork.get("split") == "calibration":
                continue
            entry = collection.get(artwork.get("design_id"))
            if entry is None or any(artwork.get(key) != entry[field] for key, field in (
                ("source_sha256", "sha256"), ("family", "family"), ("split", "split"),
            )):
                invalid_artwork = True
    eligible = (not contaminated and not dataset_dirty and not harness_dirty and len(rows) >= 3
                and not training_artwork and not calibration_control and not invalid_artwork)
    return {
        "schema": EVAL_SCHEMA,
        "created_at": created_at,
        "disclaimer": DISCLAIMER,
        "interpretation": "screen-only" if eligible else "not-comparable",
        "comparison_eligible": eligible,
        "comparison_blocks": [
            reason for reason, blocked in (
                ("training_seed_overlap", contaminated),
                ("artwork_not_held_out", training_artwork),
                ("artwork_provenance_unverified", invalid_artwork),
                ("calibration_control_separate_from_artwork", calibration_control),
                ("dirty_dataset_source", dataset_dirty),
                ("dirty_eval_harness", harness_dirty),
                ("fewer_than_3_episodes", len(rows) < 3),
            ) if blocked
        ],
        "checkpoint": {
            "producer": producer,
            "id": checkpoint_id,
            "sha256": checkpoint_sha256,
            "base_model_sha256": base_model_sha256,
        },
        "split": {
            "status": "calibration" if calibration_control else "contaminated" if contaminated or training_artwork else "held-out",
            "artwork_families": sorted({a["family"] for a in artworks if a.get("family")}),
            "artwork_splits": sorted(str(v) for v in artwork_splits),
            "evaluation_seeds": sorted(seed_values),
            "training_seed_ranges": [list(value) for value in training_seed_ranges],
        },
        "datasets": datasets_record,
        "harness": {
            "revision": harness_state["revision"],
            "dirty": harness_state["dirty"],
            "evaluation_module_sha256": sha256_file(Path(__file__)),
            "repository": str(repo_root()),
        },
        "metrics": metrics,
        "mechanical": {
            "episodes": len(rows),
            "engaged_fraction": float(np.mean([row["engaged"] for row in rows])),
            "blank_fraction": float(np.mean([row["blank"] for row in rows])),
            "contact_s": _bootstrap_summary([row["contact_s"] for row in rows], seed=100),
            "interaction_frames": _bootstrap_summary(
                [float(row["interaction_frames"]) for row in rows], seed=101,
            ),
            "floor_clamped_step_fraction": [
                dataset.run_meta.get("floor_clamped_step_fraction") for dataset in datasets
            ],
            "chunk_rejections": sum(
                len(dataset.run_meta.get("evaluation", {}).get("chunk_rejections", []))
                for dataset in datasets
            ),
        },
        "episodes": rows,
    }


def _write_markdown(path: Path, report: dict) -> None:
    lines = [
        "# Tatbot simulation evaluation",
        "",
        f"> {report['disclaimer']}",
        "",
        f"- Checkpoint: `{report['checkpoint']['id']}`",
        f"- Checkpoint SHA-256: `{report['checkpoint']['sha256'] or 'not applicable'}`",
        f"- Harness revision: `{report['harness']['revision'] or 'unknown'}`",
        f"- Interpretation: **{report['interpretation']}**",
        f"- Split: **{report['split']['status']}**",
        f"- Episodes: {report['mechanical']['episodes']}",
        f"- Chunk rejections: {report['mechanical']['chunk_rejections']}",
        "",
        "| metric | mean | 95% bootstrap CI | min | max |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for field in SCORE_FIELDS:
        value = report["metrics"][field]
        ci = "n/a" if value["ci95"] is None else f"{value['ci95'][0]:.4f} - {value['ci95'][1]:.4f}"
        if value["mean"] is None:
            lines.append(f"| {field} | n/a | {ci} | n/a | n/a |")
        else:
            lines.append(
                f"| {field} | {value['mean']:.4f} | {ci} | "
                f"{value['minimum']:.4f} | {value['maximum']:.4f} |",
            )
    if report["comparison_blocks"]:
        lines.extend(["", "Comparison blocks: " + ", ".join(report["comparison_blocks"]) + "."])
    lines.extend(["", "## Evidence", ""])
    for episode in report["episodes"]:
        overlay = episode["artifacts"]["overlay"]["path"]
        videos = ", ".join(
            f"{camera}: `{record['path']}`" for camera, record in episode["videos"].items()
        ) or "none"
        lines.append(
            f"- Episode {episode['episode']} ({episode['surface_profile'] or 'unspecified surface'}): "
            f"overlay `{overlay}`; videos {videos}."
        )
    path.write_text("\n".join(lines) + "\n")


def write_evaluation_report(output_dir: Path, report: dict) -> None:
    output_dir = Path(output_dir)
    if output_dir.exists() and any(output_dir.iterdir()):
        raise SimEvalError("output_not_empty", f"output directory is not empty: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "report.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    with (output_dir / "scores.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=("dataset", "episode", "kind", *SCORE_FIELDS, "blank", "engaged", "contact_s", "interaction_frames"),
        )
        writer.writeheader()
        for row in report["episodes"]:
            writer.writerow({field: row[field] for field in writer.fieldnames})
    _write_markdown(output_dir / "report.md", report)
