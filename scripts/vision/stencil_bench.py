#!/usr/bin/env python3
"""Stencil bench tier 0: score one stencil design and tracker on seeded 2-D transfer scenes.

Renders a deterministic bank of wrist and overhead views of the candidate's
transfer on flat or cylindrical skin (washed off, spread, pooled, smudged,
wobbled; violet or light blue), runs the tracker on each image, and scores its
correspondences in millimetres against the transferred geometry. Writes
scorecard.json, scenes.jsonl, a worst-case contact sheet and a sample sheet to
the run directory. CPU only; no camera, arm or network.
"""

import json
import os
import shutil
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(REPO/"scripts/lib"), str(REPO/"scripts/vision")]
from tatbot_paths import bootstrap  # noqa: E402

bootstrap()
# One thread per worker process: the bench parallelises over scenes.
for _variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_variable, "1")
import cv2  # noqa: E402
import numpy as np  # noqa: E402
import stencil_bench_arguments  # noqa: E402
import stencil_bench_candidates as candidates  # noqa: E402
import stencil_bench_scene as bench  # noqa: E402
import stencil_bench_score as scoring  # noqa: E402
import stencil_bench_trackers as trackers  # noqa: E402
import tatbot_runlog  # noqa: E402

_WORKER = {}


def cameras(args):
    root = tatbot_runlog.log_root()
    calibration = Path(args.calibration).expanduser() if args.calibration else root/"vision/calibration-current.json"
    robot_world = Path(args.robot_world).expanduser() if args.robot_world else root/"vision/robot-world-current.json"
    overhead, views = bench.usable_overhead(bench.overhead_models(REPO, calibration, robot_world))
    return bench.wrist_models(REPO)+overhead, views


def prepare_artwork(args, output):
    """(candidate, distractor) artwork directories under the run's output."""
    settings = candidates.parse_settings(args.set)
    if args.artwork:
        candidate_dir = output/"artwork/candidate"
        shutil.copytree(Path(args.artwork).expanduser(), candidate_dir)
        distractor_dir = candidates.generate("flower-of-life", "bench-distractor", output/"artwork/distractor")
    else:
        candidate_dir = candidates.generate(args.candidate, args.seed, output/"artwork/candidate",
                                            marked=args.marked, settings=settings)
        distractor_dir = candidates.generate(args.candidate, f"{args.seed}-distractor", output/"artwork/distractor",
                                             marked=args.marked, settings=settings)
    return candidate_dir, distractor_dir


def _init(tracker_name, candidate_dir, distractor_dir, camera_models, oracle=None):
    cv2.setNumThreads(1)
    candidate = bench.load_artwork(candidate_dir)
    distractor = bench.load_artwork(distractor_dir)
    tracker = oracle if oracle is not None else trackers.make(tracker_name)
    tracker.prepare([candidate])
    _WORKER.update(tracker=tracker, candidate=candidate, distractor=distractor,
                   cameras={c.name: c for c in camera_models})


def run_scene(params, thumbnail=True):
    w = _WORKER
    camera = w["cameras"][params["camera"]]
    scene = bench.Scene(params, w["candidate"], camera, distractor=w["distractor"])
    image = scene.render()
    if hasattr(w["tracker"], "truth"):
        w["tracker"].truth = scene
    started = time.perf_counter()
    located = w["tracker"].locate(image, camera.intrinsics())
    elapsed = (time.perf_counter()-started)*1000
    record = scoring.score_scene(scene, located, elapsed)
    thumb = None
    if thumbnail:
        ok, encoded = cv2.imencode(".jpg", scoring.annotate(image, scene, located, record), [cv2.IMWRITE_JPEG_QUALITY, 85])
        thumb = encoded.tobytes() if ok else None
    return record, thumb


def evaluate(params_list, workers, init_args):
    if workers <= 1:
        _init(*init_args)
        return [run_scene(p) for p in params_list]
    with ProcessPoolExecutor(workers, initializer=_init, initargs=init_args) as pool:
        return list(pool.map(run_scene, params_list, chunksize=1))


def write_outputs(output, results, card):
    rows = [record for record, _ in results]
    with (output/"scenes.jsonl").open("w") as stream:
        for row in rows:
            stream.write(json.dumps(row)+"\n")
    (output/"scorecard.json").write_text(json.dumps(card, indent=2)+"\n")
    decode = [cv2.imdecode(np.frombuffer(thumb, np.uint8), cv2.IMREAD_COLOR) for _, thumb in results]
    order = sorted(range(len(rows)), key=lambda i: scoring.badness(rows[i]))
    worst = [decode[i] for i in order[:card["worst_shown"]] if decode[i] is not None]
    scoring.contact_sheet(worst, output/"worst.jpg")
    scoring.contact_sheet([d for d in decode[:8] if d is not None], output/"sample.jpg")


def main(argv=None):
    parser = stencil_bench_arguments.parser(__doc__)
    args = parser.parse_args(argv)
    try:
        stencil_bench_arguments.validate(args)
    except ValueError as error:
        parser.error(str(error))
    with tatbot_runlog.init("stencil-bench") as run:
        output = Path(args.output).expanduser() if args.output else run.dir
        output.mkdir(parents=True, exist_ok=True)
        started = time.perf_counter()
        candidate_dir, distractor_dir = prepare_artwork(args, output)
        camera_models, views = cameras(args)
        params = [bench.sample_params(args.bank, i, camera_models, args.camera, args.degradation) for i in range(args.scenes)]
        workers = args.workers or min(8, os.cpu_count() or 1)
        results = evaluate(params, workers, (args.tracker, candidate_dir, distractor_dir, camera_models))
        candidate = bench.load_artwork(candidate_dir)
        meta = {"run_id": run.run_id, "tracker": args.tracker, "bank": args.bank, "scenes": args.scenes,
                "camera_mix": args.camera, "degradation": args.degradation, "workers": workers, "worst_shown": args.worst,
                "candidate": {"name": "artwork" if args.artwork else args.candidate, "seed": candidate.meta["seed"],
                              "pattern_id": candidate.pattern_id, "marked": candidate.meta["marked"],
                              "settings": candidates.parse_settings(args.set)},
                "cameras": {c.name: c.intrinsics() for c in camera_models}, "overhead_views": views,
                "photo_fit": bench.PHOTO_FIT, "elapsed_s": None}
        if args.bank == "holdout":
            meta["holdout_note"] = "holdout bank: report it, never tune on it"
        card = scoring.scorecard([r for r, _ in results], meta, candidate)
        card["elapsed_s"] = round(time.perf_counter()-started, 1)
        write_outputs(output, results, card)
        for name in ("scorecard.json", "scenes.jsonl", "worst.jpg", "sample.jpg"):
            run.artifact(output/name)
        overall = card["overall"]
        print(json.dumps({"run_id": run.run_id, "output": str(output), "elapsed_s": card["elapsed_s"],
                          **{k: overall[k] for k in ("success_rate", "accept_rate", "false_accepts",
                                                     "ambiguous_rate", "point_error_mm",
                                                     "localized_fraction_mean", "model_error_mm")}}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
