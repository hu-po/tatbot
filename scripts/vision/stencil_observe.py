#!/usr/bin/env python3
"""Reference-backed stencil image tracking from recordings or an existing owner."""

import argparse
import json
import platform
import shutil
import sys
import time
from collections import Counter
from itertools import islice
from pathlib import Path

import cv2
import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "scripts/lib"))
from tatbot_paths import bootstrap  # noqa: E402

bootstrap()
import stencil_arguments  # noqa: E402
import stencil_reference  # noqa: E402
import tatbot_runlog  # noqa: E402
from stencil_coded_live import CodedStencilTracker  # noqa: E402
from stencil_evaluate import score  # noqa: E402
from stencil_inputs import image_frames, owner_frames, visiond_frames  # noqa: E402
from stencil_scene import StencilScene, objects  # noqa: E402
from stencil_surface import estimate_surface  # noqa: E402
from stencil_tracking import StencilTracker  # noqa: E402


def _sources(args):
    if args.mode == "observe":
        return owner_frames(args.socket, args.sensor, args.duration_s)
    if args.frames:
        return image_frames(args.frames)
    return visiond_frames(args.recording)


def _references(tracker, paths, output):
    for path in paths:
        manifest_path = Path(path).expanduser().resolve()
        manifest = json.loads(manifest_path.read_text())
        destination = output/"references"/manifest["pattern_id"]
        destination.mkdir(parents=True)
        shutil.copy2(manifest_path, destination/"tracking.json")
        shutil.copy2(manifest_path.with_name("stencil.png"), destination/"stencil.png")
        if manifest_path.with_name("coded.json").is_file():
            shutil.copy2(manifest_path.with_name("coded.json"), destination/"coded.json")
    return list(tracker.bank.references.values())


def _overlay(image, observation, destination):
    shown = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR) if image.ndim == 2 else image.copy()
    for index, row in enumerate(objects(observation)):
        color = ((0, 200, 0), (255, 180, 0), (200, 0, 200))[index % 3]
        if row["image_tracking_valid"]:
            polygon = np.rint(row["page_polygon_px"]).astype(np.int32)
            cv2.polylines(shown, [polygon], True, color, 2)
            for point in row["landmarks"]:
                cv2.circle(shown, tuple(np.rint(point["image_px"]).astype(int)), 2, color, -1)
        label = f"{row.get('seed')}: {row['status']} | surface: {row.get('surface', {}).get('status', 'off')}"
        cv2.putText(shown, label, (8, 24+24*index), cv2.FONT_HERSHEY_SIMPLEX, .5, color, 1)
    cv2.imwrite(str(destination), shown)


def _surfaces(result, frame, output, index, references):
    meshes = {}
    directory = output/"surfaces"
    directory.mkdir(exist_ok=True)
    for row in objects(result):
        started = time.perf_counter()
        row["surface"], mesh = estimate_surface(row, frame, references.get(row["pattern_id"]))
        row["geometry_reason"] = ("surface_candidate_not_qualified" if row["surface"]["candidate_valid"]
                                  else row["surface"]["reason"])
        row["surface_processing_ms"] = (time.perf_counter()-started)*1000
        if mesh is not None:
            meshes[row['pattern_id']] = mesh
            path = directory/f"{index:06}-{row['pattern_id'][-12:]}.npz"
            np.savez_compressed(path, **mesh)
            row["surface"]["mesh_file"] = str(path.relative_to(output))
    return meshes


def _compact(result):
    reduced = {key: value for key, value in result.items() if key not in ("landmarks", "input", "stencils", "surface")}
    if "surface" in result:
        reduced["surface"] = {key: value for key, value in result["surface"].items() if key != "anchors"}
    if "stencils" in result:
        reduced["stencils"] = [_compact(row) for row in result["stencils"]]
    return reduced


def evaluate(frames, tracker, output, limit, live=False, surface=False, display=None, regions=None):
    observations = []
    search = {"regions": regions} if regions is not None else {}
    with (output/"observations.jsonl").open("w") as stream:
        for index, frame in enumerate(islice(frames, limit)):
            started = time.perf_counter()
            result = tracker.observe(frame["image"], frame["timestamp_ns"],
                                     depth_m=frame.get("depth_m"), source_id=frame["source_id"], **search)
            result.update(frame_index=index, input=frame.get("provenance", {}))
            meshes = _surfaces(result, frame, output, index, tracker.bank.references) if surface else {}
            result["image_processing_ms"] = result["processing_ms"]
            result["processing_ms"] = (time.perf_counter()-started)*1000
            if live:
                result["capture_age_ms"] = (time.time_ns()-frame["timestamp_ns"])/1e6
                result["freshness_verified"] = False  # Host-clock agreement is not established here.
            stream.write(json.dumps(result, allow_nan=False)+"\n")
            stream.flush()
            observations.append(_compact(result))
            if display is not None:
                display.log(frame, result, meshes)
        if observations:
            _overlay(frame["image"], result, output/"last-frame.png")
    return observations


def _report(args, tracker, observations, output):
    costs = [o["processing_ms"] for o in observations if "processing_ms" in o]
    rows = [row for observation in observations for row in objects(observation)]
    report = {"schema": "tatbot.stencil-replay-report/1", "mode": args.mode,
              "opencv": cv2.__version__, "python": platform.python_version(),
              "references": _references(tracker, args.reference, output),
              "frames": len(observations), "status_counts": dict(Counter(o["status"] for o in rows)),
              "image_valid_frames": sum(o["image_tracking_valid"] for o in observations),
              "geometry_valid_frames": 0, "motion_authority": False,
              "processing_ms": {f"p{p}": float(np.percentile(costs, p)) if costs else None for p in (50, 95)},
              "processing_by_status_ms": {
                  status: {f"p{p}": float(np.percentile([o["processing_ms"] for o in rows
                                                        if o["status"] == status], p)) for p in (50, 95)}
                  for status in sorted({o["status"] for o in rows})},
              "per_pattern": {pattern: {
                  "seed": reference["seed"],
                  "image_valid_frames": sum(row["image_tracking_valid"] for row in rows if row["pattern_id"] == pattern),
                  "surface_candidates": sum(row.get("surface", {}).get("candidate_valid", False)
                                            for row in rows if row["pattern_id"] == pattern)}
                  for pattern, reference in tracker.bank.references.items()},
              "latency_scope": "processing only; excludes camera acquisition, transport, reference-bank initialization and Rerun display",
              "evaluation": None}
    if getattr(args, "truth", None):
        labels = [json.loads(line) for line in Path(args.truth).expanduser().read_text().splitlines() if line.strip()]
        if args.all_references:
            from stencil_evaluate import score_scene
            report["evaluation"] = score_scene(observations, labels)
        else:
            report["evaluation"] = score(observations, labels)
    (output/"report.json").write_text(json.dumps(report, indent=2, allow_nan=False)+"\n")
    return report


def _tracker(args):
    """Every supplied pattern (`--all-references`), else one instance: a coded print is decoded,
    legacy artwork matched by SIFT; one instance cannot mix the two."""
    if args.all_references:
        return StencilScene(args.reference, args.instance)
    coded = [stencil_reference.is_coded(stencil_reference.load(path)[0]) for path in args.reference]
    if any(coded) and not all(coded):
        raise ValueError("one instance tracks coded prints or legacy artwork, not both; use --all-references")
    if args.region and not all(coded):
        raise ValueError("--region bounds a coded print's search")
    return (CodedStencilTracker if all(coded) else StencilTracker)(args.reference, args.instance)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_subparsers(dest="mode", required=True)
    stencil_arguments.replay(modes.add_parser("replay"))
    stencil_arguments.observe(modes.add_parser("observe"))
    args = parser.parse_args(argv)
    cv2.setNumThreads(1)
    try:
        output = stencil_arguments.validate(args, REPO)
        tracker = _tracker(args)
    except (ValueError, OSError, KeyError) as error:
        parser.error(str(error))
    output.mkdir(parents=True)
    with tatbot_runlog.init("stencil-"+args.mode, prune_first=False) as run:
        frames = _sources(args)
        display = None
        try:
            if args.rerun or args.connect:
                from stencil_rerun import StencilRerun
                display = StencilRerun(args, output, run.run_id, references=tracker.bank.references)
            observations = evaluate(frames, tracker, output, args.max_frames,
                                    live=args.mode == "observe", surface=args.surface, display=display,
                                    regions=[args.region] if args.region else None)
        finally:
            frames.close()
            if display is not None:
                display.close()
        report = _report(args, tracker, observations, output)
        run.artifact(output/"observations.jsonl")
        run.artifact(output/"report.json")
        if args.rerun:
            run.artifact(output/"replay.rrd")
        print(json.dumps(report, indent=2))
        print(f"Report: {output/'report.json'}")
        run.finalize(0 if observations else 5)
    return 0 if observations else 5


if __name__ == "__main__":
    raise SystemExit(main())
