"""``travel``: preview, generate, merge and summarise synthetic ink-tracing episodes.

    travel preview --seed 3 --seconds 20 --out /tmp/ep3      # mp4 + contact sheet, no LeRobot needed
    travel generate --root DIR --episodes 400 --workers 12    # LeRobot v3 shards under DIR/shards
    travel merge --root DIR                                   # shards -> DIR/merged
    travel stats --root DIR                                   # expert mode fractions from the labels
    travel evaluate --checkpoint CKPT --dataset DS --out r.json  # closed loop on held-out seeds

Rendering is headless EGL on GPU 0 unless the environment already says otherwise.
"""

from __future__ import annotations

import argparse
import json
import os
import signal
import sys
import time
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
os.environ.setdefault("MUJOCO_EGL_DEVICE_ID", "0")
# Only NVIDIA's EGL: with Mesa's vendor also installed, a GPU context that fails under memory pressure
# falls back to software rendering -- 27 s a frame -- without an error; better to fail loudly.
_NVIDIA_EGL = Path("/usr/share/glvnd/egl_vendor.d/10_nvidia.json")
if _NVIDIA_EGL.exists():
    os.environ.setdefault("__EGL_VENDOR_LIBRARY_FILENAMES", str(_NVIDIA_EGL))
# Generation parallelises across worker processes; per-process thread pools only oversubscribe.
for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_var, "1")

MODES = ("park", "approach", "trace", "backoff")


def _episode_config(seconds: float, profile: str | None = None):
    from tatbot_travel.profile import load_profile

    return load_profile(profile, seconds)


def preview(args: argparse.Namespace) -> int:
    from dataclasses import replace

    if args.appearance_seeds or args.episode_seeds or args.skin_seeds:
        return preview_variants(args)
    import cv2
    import numpy as np

    from tatbot_travel.episode import EpisodeRunner
    from tatbot_travel.preview import inspect_views

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    cfg = _episode_config(args.seconds, args.profile)
    if args.appearance_seed is not None:
        cfg = replace(cfg, world=replace(cfg.world, appearance_seed=args.appearance_seed))
    if args.skin_seed is not None:
        cfg = replace(cfg, world=replace(cfg.world, skin_seed=args.skin_seed))
    if args.skin_reference:
        cfg = replace(cfg, world=replace(cfg.world, skin=replace(cfg.world.skin, randomize=False)))
    run = EpisodeRunner(args.seed, cfg)
    width = 640 + (run.world.scene_intrinsics.width if run.scene is not None else 0)
    writer = cv2.VideoWriter(str(out.with_suffix(".mp4")), cv2.VideoWriter_fourcc(*"mp4v"), 30, (width, 480))
    frames = []

    def on_frame(frame) -> None:
        if not frames and args.views:
            inspect_views(run, out, frame.image, geometry_only=args.geometry_only)
        views = [frame.image] + ([frame.scene] if frame.scene is not None else [])
        image = cv2.cvtColor(np.concatenate(views, axis=1), cv2.COLOR_RGB2BGR)
        frames.append(image)
        label = f"t={len(frames) / 30:5.1f}s {MODES[frame.labels['mode']]:8s} vis={frame.labels['visible']:.2f}"
        cv2.putText(image, label, (8, 470), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
        writer.write(image)

    started = time.time()
    n = run.run(on_frame)
    writer.release()
    run.close()
    picks = np.linspace(0, len(frames) - 1, 12).astype(int)
    tiles = [cv2.resize(frames[i], (width // 2, 240)) for i in picks]
    sheet = np.concatenate([np.concatenate(tiles[r * 4:(r + 1) * 4], axis=1) for r in range(3)], axis=0)
    cv2.imwrite(str(out.with_suffix(".png")), sheet)
    print(f"{n} frames in {time.time() - started:.1f}s -> {out.with_suffix('.mp4')}, {out.with_suffix('.png')}")
    return 0


def preview_variants(args: argparse.Namespace) -> int:
    from tatbot_travel.preview import comparison_sheet

    seeds = args.episode_seeds or args.appearance_seeds or args.skin_seeds
    if not 1 <= len(seeds) <= 8 or len(set(seeds)) != len(seeds):
        raise ValueError("preview comparison needs 1-8 distinct seeds")
    out, variants = Path(args.out), []
    kind = "seed" if args.episode_seeds else ("skin" if args.skin_seeds else "look")
    seeds = [None, *seeds] if args.skin_seeds else seeds
    for seed in seeds:
        variant = out.with_name(f'{out.name}-{kind}{"reference" if seed is None else seed}')
        field = {"seed": "seed", "look": "appearance_seed", "skin": "skin_seed"}[kind]
        override = {field: seed, "skin_reference": seed is None}
        preview(argparse.Namespace(**{**vars(args), "out": str(variant), "views": True,
                                     "appearance_seeds": None, "episode_seeds": None,
                                     "skin_seeds": None, **override}))
        variants.append(variant)
    comparison_sheet(variants, out, episodes=bool(args.episode_seeds), skins=bool(args.skin_seeds))
    return 0


def _generate_shard(job: tuple[str, str, int, list[int], float, str | None]) -> dict:
    root, repo_id, shard, seeds, seconds, profile = job
    import cv2

    from tatbot_travel.camera import SCENE_STREAM_SIZE
    from tatbot_travel.episode import TASK, EpisodeRunner
    from tatbot_travel.writer import DatasetSink

    cv2.setNumThreads(1)
    # A stopped worker (SIGTERM as well as SIGINT) still finalizes its shard: every finished episode is kept.
    # Once the shard is done, SIGTERM kills again: raised in an idle worker's garbage collection, the
    # interrupt would be swallowed and leave the pool's terminate waiting on it forever.
    previous = signal.signal(signal.SIGTERM, signal.default_int_handler)
    cfg = _episode_config(seconds, profile)
    scene_hw = (SCENE_STREAM_SIZE[1], SCENE_STREAM_SIZE[0]) if cfg.world.scene_view else None
    sink = DatasetSink(Path(root) / "shards" / f"shard_{shard:03d}", f"{repo_id}_shard{shard:03d}", TASK,
                       image_writer_threads=2, scene_hw=scene_hw)
    done = []
    try:
        for seed in seeds:
            run = EpisodeRunner(seed, cfg)
            run.run(sink.add)
            meta = {"seed": seed, "draw": run.draw, "traceable_at_start": bool(run.traceable), **run.world.meta,
                    "profile_id": cfg.profile_id, "start_pose_rad": cfg.start_pose_rad,
                    "stale_frame": bool(run.stale), "servo_tau_s": run.servo.tau,
                    "servo_delay_ticks": run.servo.delay, "camera_look": run.sensor.look.__dict__}
            if run.scene is not None:
                meta |= {"scene_look": run.scene.look.__dict__, "scene_latency_s": run.scene.latency,
                         "scene_period_s": run.scene.period}
            sink.end_episode(meta, run.world.ink.surface_arrays())
            run.close()
            done.append(seed)
    finally:
        sink.finalize()
        signal.signal(signal.SIGTERM, previous)
    return {"shard": shard, "episodes": len(done)}


def generate(args: argparse.Namespace) -> int:
    import multiprocessing as mp

    root = Path(args.root)
    if (root / "shards").exists() and any((root / "shards").iterdir()):
        print(f"{root / 'shards'} is not empty; choose a new root", file=sys.stderr)
        return 2
    seeds = [args.seed_base + i for i in range(args.episodes)]
    cfg = _episode_config(args.seconds, args.profile)  # validate before starting workers or writing shards
    jobs = [(str(root), args.repo_id, k, seeds[k::args.workers], args.seconds, args.profile)
            for k in range(args.workers)]
    root.mkdir(parents=True, exist_ok=True)
    (root / "generation.json").write_text(json.dumps({**vars(args), "profile_id": cfg.profile_id},
                                                   indent=1, default=str) + "\n")
    started = time.time()
    with mp.get_context("spawn").Pool(args.workers) as pool:
        for result in pool.imap_unordered(_generate_shard, jobs):
            print(f"shard {result['shard']:03d}: {result['episodes']} episodes ({time.time() - started:.0f}s)",
                  flush=True)
    return 0


def merge(args: argparse.Namespace) -> int:
    """Aggregate the shards (in shard order) and carry their labels over to global episode indices."""
    import shutil

    from lerobot.datasets.aggregate import aggregate_datasets

    root = Path(args.root)
    out = root / "merged"
    if out.exists():
        print(f"{out} exists", file=sys.stderr)
        return 2
    shards = sorted(p for p in (root / "shards").iterdir() if (p / "meta" / "info.json").exists())
    aggregate_datasets(repo_ids=[f"{args.repo_id}-{p.name}" for p in shards], aggr_repo_id=args.repo_id,
                       roots=shards, aggr_root=out)
    offset = 0
    (out / "labels").mkdir(parents=True, exist_ok=True)
    for shard in shards:
        count = json.loads((shard / "meta" / "info.json").read_text())["total_episodes"]
        for local in range(count):
            for suffix in (".npz", ".json", ".surface.npz"):
                src = shard / "labels" / f"episode_{local:06d}{suffix}"
                if src.exists():
                    shutil.copy2(src, out / "labels" / f"episode_{offset + local:06d}{suffix}")
        offset += count
    print(f"merged {len(shards)} shards, {offset} episodes -> {out}")
    return 0


def stats(args: argparse.Namespace) -> int:
    import numpy as np

    files = sorted(path for path in Path(args.root).rglob("labels/episode_*.npz")
                   if not path.name.endswith(".surface.npz"))
    counts = np.zeros(len(MODES))
    for f in files:
        counts += np.bincount(np.load(f)["mode"], minlength=len(MODES))[: len(MODES)]
    total = max(counts.sum(), 1)
    print(f"{len(files)} episodes, {int(total)} frames: "
          + ", ".join(f"{m} {c / total:.2f}" for m, c in zip(MODES, counts, strict=True)))
    return 0


def scene_bank(args: argparse.Namespace) -> int:
    from tatbot_travel.scene_bank import build_bank

    report = build_bank(Path(args.captures), Path(args.layout), Path(args.out))
    print(json.dumps({"out": args.out, "missing_depth_fraction": report["missing_depth_fraction"]}))
    return 0


def evaluate(args: argparse.Namespace) -> int:
    from tatbot_travel.evaluate import evaluate as run

    seeds = [args.seed_base + i for i in range(args.episodes)]
    run(Path(args.checkpoint), Path(args.dataset), seeds, args.seconds, Path(args.out), ink_mode=args.ink,
        held_out_designs=args.held_out_designs, steps=args.steps, guidance=args.guidance,
        latency_ticks=args.latency_ticks, profile=args.profile)
    return 0


def _pose(spec: str):
    """A named pose (park, rest) or six comma-separated joint angles in radians."""
    import numpy as np

    if spec == "rest":  # the arm's own rest: folded, the tool rolled 90 deg
        return np.array([0.0, 0.0, 0.0, 0.0, 0.0, np.pi / 2])
    if spec == "park":
        from tatbot_travel.camera import Intrinsics, render_plan
        from tatbot_travel.episode import park_pose
        from tatbot_travel.kinematics import ArmKinematics
        from tatbot_travel.scene import SceneBuilder

        return park_pose(ArmKinematics(SceneBuilder(plan=render_plan(Intrinsics.left_wrist())).compile()))
    q = np.array([float(v) for v in spec.split(",")])
    if q.shape != (6,):
        raise SystemExit("--pose takes park, rest, or six comma-separated radians")
    return q


def shadow(args: argparse.Namespace) -> int:
    from tatbot_travel.evaluate import LeRobotPolicy
    from tatbot_travel.shadow import run

    out = Path(args.out).expanduser() / time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    policy = LeRobotPolicy(Path(args.checkpoint), Path(args.dataset), steps=args.steps, guidance=args.guidance)
    summary = run(policy, _pose(args.pose), args.seconds, out, camera=args.camera, socket_path=Path(args.socket))
    print(json.dumps({"out": str(out), **summary}))
    return 0


def run_arm(args: argparse.Namespace) -> int:
    """The policy on the real blue arm, on the arm node: a shadow run unless --move."""
    import logging

    from tatbot_travel.hardware import BlueArm, SceneCamera, WristCamera
    from tatbot_travel.planner import ProcessPlanner
    from tatbot_travel.runner import Gates, run, untrained_joints

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
    if (args.move or args.hold) and args.ceiling_m is None:
        raise SystemExit("--move and --hold need --ceiling-m: the highest any part of the arm may go on this rig "
                         "(the lab's overhead cameras hang about 0.55 m over the blue arm's base)")
    gates = Gates(max_speed=args.max_speed, stop_mm=args.stop_mm, abort_mm=args.abort_mm,
                  ceiling_m=args.ceiling_m if args.ceiling_m is not None else Gates.ceiling_m)
    planner = ProcessPlanner(Path(args.checkpoint), Path(args.dataset), steps=args.steps, guidance=args.guidance,
                             seed=0 if args.fresh_seed else None)
    if "scene" in planner.cameras and not args.scene_camera:
        planner.close()
        raise SystemExit("this checkpoint looks through a scene camera too: pass --scene-camera (camera5)")
    scene = SceneCamera(args.scene_camera) if args.scene_camera else None
    arm = BlueArm(estop_required=not args.no_estop) if (args.move or args.hold) else None
    held = untrained_joints(Path(args.checkpoint))
    if held:
        logging.getLogger("travel.runner").warning(
            "the checkpoint never saw joints %s move in training: holding them at the start pose", held)
    result = run(planner, WristCamera(), arm, _pose(args.pose), args.seconds, gates, Path(args.out),
                 drive=not args.hold, held=held, smooth_ticks=args.smooth_ticks, timing=args.timing, scene=scene)
    print(json.dumps(result))
    landed = result.get("landing") in (None, "landed", "recovered")  # None: the arm was never taken
    return 0 if result.get("ended") in ("time", "interrupted") and landed else 1


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="travel", description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("scene-bank", help="build private room/cradle/ink assets from recorded RGB-D captures")
    p.add_argument("--captures", required=True)
    p.add_argument("--layout", required=True, help="JSON with explicit crop boxes, masks and episode overrides")
    p.add_argument("--out", required=True, help="new bank directory outside the repository")
    p.set_defaults(func=scene_bank)
    p = sub.add_parser("preview", help="render one episode to mp4 and a contact sheet")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--seconds", type=float, default=20.0)
    p.add_argument("--out", required=True, help="output path without extension")
    p.add_argument("--profile", help="episode profile JSON beside a private scene bank")
    p.add_argument("--views", action="store_true", help="also save exterior and hand views with surface-address labels")
    p.add_argument("--geometry-only", action="store_true", help="hide captured room meshes in exterior inspections")
    appearance = p.add_mutually_exclusive_group()
    appearance.add_argument("--appearance-seed", type=int, help="vary surfaces and lights with fixed episode geometry/ink")
    appearance.add_argument("--appearance-seeds", type=int, nargs="+", help="compare up to eight appearance seeds")
    appearance.add_argument("--episode-seeds", type=int, nargs="+", help="compare full scenes, including placement, from fixed cameras")
    appearance.add_argument("--skin-seed", type=int, help="vary only skin appearance with fixed scene, lighting, pose and ink")
    appearance.add_argument("--skin-seeds", type=int, nargs="+", help="compare up to eight skin seeds, preceded by the reference appearance")
    appearance.add_argument("--skin-reference", action="store_true", help="render skin tone and properties at their configured centers")
    p.set_defaults(func=preview)
    p = sub.add_parser("generate", help="write LeRobot v3 shards")
    p.add_argument("--root", required=True)
    p.add_argument("--repo-id", default="local/travel-ink")
    p.add_argument("--episodes", type=int, required=True)
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--seconds", type=float, default=60.0)
    p.add_argument("--seed-base", type=int, default=0)
    p.add_argument("--profile", help="episode profile JSON beside a private scene bank")
    p.set_defaults(func=generate)
    p = sub.add_parser("merge", help="aggregate shards into one dataset")
    p.add_argument("--root", required=True)
    p.add_argument("--repo-id", default="local/travel-ink")
    p.set_defaults(func=merge)
    p = sub.add_parser("evaluate", help="drive held-out sim episodes with a trained checkpoint")
    p.add_argument("--checkpoint", required=True,
                   help="a pretrained_model (or pretrained_model_ema) directory from lerobot-train")
    p.add_argument("--dataset", required=True, help="the dataset root it was trained on (for metadata)")
    p.add_argument("--episodes", type=int, default=10)
    p.add_argument("--seconds", type=float, default=60.0)
    p.add_argument("--seed-base", type=int, default=900000, help="held out: training seeds start at 100000")
    p.add_argument("--out", required=True, help="results JSON path")
    p.add_argument("--ink", choices=["lines", "tattoos", "both"], help="draw only this (default: the training mix)")
    p.add_argument("--held-out-designs", action="store_true", help="tattoo with the validation/test flash")
    p.add_argument("--steps", type=int, help="sampler steps (default: the checkpoint's)")
    p.add_argument("--guidance", type=float, help="video guidance; <= 1 is one pass per step (default: the checkpoint's)")
    p.add_argument("--latency-ticks", type=int,
                   help="run the demo's asynchronous chunking, each plan arriving this many ticks late")
    p.add_argument("--profile", help="use the same private scene profile as generation")
    p.set_defaults(func=evaluate)
    p = sub.add_parser("shadow", help="run a checkpoint on the live wrist camera with the arm pinned (no motion)")
    p.add_argument("--checkpoint", required=True, help="a pretrained_model (or _ema) directory")
    p.add_argument("--dataset", required=True, help="the dataset root it was trained on (for metadata)")
    p.add_argument("--pose", default="park", help="the pose the arm stands in: park, rest, or six radians")
    p.add_argument("--seconds", type=float, default=30.0)
    p.add_argument("--steps", type=int)
    p.add_argument("--guidance", type=float)
    p.add_argument("--camera", default="realsense1", help="visiond sensor name of the blue arm's wrist D405")
    p.add_argument("--socket", default="/tmp/tatbot-d405-frames.sock")
    p.add_argument("--out", default="~/tatbot-logs/travel_shadow")
    p.set_defaults(func=shadow)
    p = sub.add_parser("run", help="the policy on the real blue arm (arm node); a shadow run unless --move")
    p.add_argument("--checkpoint", required=True, help="a pretrained_model (or _ema) directory")
    p.add_argument("--dataset", required=True, help="the dataset root it was trained on (for metadata)")
    p.add_argument("--pose", default="park", help="start pose: park (the sim's), rest, or six radians")
    p.add_argument("--ceiling-m", type=float, help="required with --move: no part of the arm above this (base frame, m)")
    p.add_argument("--seconds", type=float, default=60.0)
    p.add_argument("--steps", type=int, help="sampler steps (default: the checkpoint's -- 4 for the published ones)")
    p.add_argument("--guidance", type=float, help="guidance; <= 1 is one pass per step (default: the checkpoint's)")
    p.add_argument("--fresh-seed", action="store_true", help="new sampling noise every plan (default: the fixed seed)")
    p.add_argument("--scene-camera", help="a fleet PoE camera (e.g. camera5): recorded, and fed to policies "
                   "trained on a scene view")
    p.add_argument("--move", action="store_true", help="take the arm and move it (default: shadow, arm untouched)")
    p.add_argument("--hold", action="store_true",
                   help="take the arm to the start pose and hold it there: the policy only draws what it would do")
    p.add_argument("--max-speed", type=float, default=0.25, help="rad/s, any joint")
    p.add_argument("--smooth-ticks", type=int, default=7, help="moving average over each plan (1: raw)")
    p.add_argument("--timing", choices=["sync", "aligned", "arrival"], default="sync",
                   help="sync: plan, execute 32, hold while planning (the recipe's loop); aligned/arrival overlap "
                        "planning with motion (fast sampling only)")
    p.add_argument("--no-estop", action="store_true",
                   help="supervised runs only: no e-stop monitor (the operator holds another stop)")
    p.add_argument("--stop-mm", type=float, default=10.0, help="hold while the pen axis meets a surface this close")
    p.add_argument("--abort-mm", type=float, default=4.0, help="end the run if something stays this close")
    p.add_argument("--out", default="~/tatbot-logs/travel_run")
    p.set_defaults(func=run_arm)
    p = sub.add_parser("stats", help="expert mode fractions over a generated root")
    p.add_argument("--root", required=True)
    p.set_defaults(func=stats)
    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
