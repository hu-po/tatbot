"""Time one FLUX 3 Action chunk on this GPU: the demo's control loop waits for exactly this.

    python bench_latency.py --policy black-forest-labs/flux-3-action-so101 --runs 8
    python bench_latency.py --policy ~/travel/runs/v0-lora/base --steps 4 2 1 --guidance 3 1

Observations are synthetic but shaped by the package's own input features
(cameras, state width, history), so the numbers are the model's cost at the
package's canvas. Results go to stdout as JSON lines, one per setting.
"""

from __future__ import annotations

import argparse
import json
import statistics
import time

import torch
from tatbot_travel.flux3 import force_natten_backend, set_sampler


def load(path: str, device: str):
    from lerobot.policies.factory import make_pre_post_processors
    from lerobot.policies.flux3 import Flux3Policy

    policy = Flux3Policy.from_pretrained(path).to(device).eval()
    pre, post = make_pre_post_processors(policy.config, pretrained_path=path)
    return policy, pre, post


def observation(policy, generator: torch.Generator) -> dict:
    obs = {"task": "trace the ink on the arm with the laser"}
    for key, feature in policy.config.input_features.items():
        shape = tuple(feature.shape)
        if len(shape) == 3:  # camera: uint8 CHW
            obs[key] = torch.randint(0, 256, shape, dtype=torch.uint8, generator=generator)
        else:
            obs[key] = torch.randn(shape, generator=generator)
    return obs


def on_device(batch: dict, device: str) -> dict:
    """Processors saved by export_base carry its CPU device; the model runs on the GPU."""
    return {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()}


def time_chunks(policy, pre, runs: int, generator: torch.Generator, device: str) -> dict:
    times = []
    torch.cuda.reset_peak_memory_stats()
    for i in range(runs + 2):  # two warm-ups: allocator, text cache, lazy compilation
        batch = on_device(pre(observation(policy, generator)), device)
        torch.cuda.synchronize()
        started = time.perf_counter()
        with torch.inference_mode():
            chunk = policy.predict_action_chunk(batch)
        torch.cuda.synchronize()
        if i >= 2:
            times.append(time.perf_counter() - started)
    first = chunk[0, 0, :3].float().tolist()  # same seeded inputs every run: compare across settings
    return {"median_s": statistics.median(times), "min_s": min(times), "chunk": list(chunk.shape),
            "peak_gib": torch.cuda.max_memory_allocated() / 2**30,
            "fingerprint": [round(v, 4) for v in first] + [round(float(chunk.float().abs().mean()), 5)]}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--policy", required=True, help="Hub id or local flux3 package")
    parser.add_argument("--steps", type=int, nargs="+", default=[4, 2, 1])
    parser.add_argument("--guidance", type=float, nargs="+", default=[None, 1.0],
                        help="video guidance scales to try; omit a value to keep the package's")
    parser.add_argument("--compile", action="store_true", help="also time with torch.compile")
    parser.add_argument("--runs", type=int, default=6)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    force_natten_backend()
    policy, pre, _ = load(args.policy, args.device)
    cfg = policy.config
    base = {"sampler": cfg.sampler, "steps": cfg.num_inference_steps, "guidance": cfg.guidance_scale,
            "guidance_action": cfg.guidance_scale_action, "canvas": list(cfg.canvas_hw),
            "layout": cfg.camera_layout, "chunk": cfg.chunk_size, "device": torch.cuda.get_device_name()}
    print(json.dumps({"package": args.policy, **base}))
    generator = torch.Generator().manual_seed(0)
    for compile_model in ([False, True] if args.compile else [False]):
        cfg.compile_model = compile_model
        for guidance in args.guidance:
            for steps in args.steps:
                set_sampler(cfg, steps, base["guidance"] if guidance is None else guidance)
                result = time_chunks(policy, pre, args.runs, generator, args.device)
                print(json.dumps({"steps": steps, "guidance": cfg.guidance_scale, "compile": compile_model,
                                  **result}), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
