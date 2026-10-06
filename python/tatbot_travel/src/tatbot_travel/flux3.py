"""FLUX 3 Action at run time: what the evaluator and the demo set before loading a policy."""

from __future__ import annotations

import functools
import os
import shutil
import subprocess
from pathlib import Path


def _configure_flex_compiler() -> None:
    """Select a compatible assembler before compiled Flex reaches a real-arm session."""
    import torch

    if torch.cuda.get_device_capability() != (11, 0):
        return
    keys = ("TRITON_PTXAS_PATH", "TRITON_PTXAS_BLACKWELL_PATH")
    cuda_home = os.environ.get("CUDA_HOME")
    candidates = ([str(Path(cuda_home) / "bin/ptxas")] if cuda_home else [])
    candidates += [shutil.which("ptxas"), "/usr/local/cuda/bin/ptxas"]
    supported = {}

    def supports_target(path):
        if path not in supported:
            try:
                result = subprocess.run([path, "--help"], capture_output=True, text=True, timeout=3)
                supported[path] = result.returncode == 0 and "sm_110a" in result.stdout
            except (OSError, subprocess.TimeoutExpired):
                supported[path] = False
        return supported[path]

    selected = {}
    for key in keys:
        explicit = os.environ.get(key)
        choices = [explicit] if explicit else candidates
        selected[key] = next((path for path in choices if path and supports_target(path)), None)
        if selected[key] is None:
            raise RuntimeError(f"compiled Flex requires a ptxas supporting sm_110a; configure {key} "
                               "or CUDA_HOME with a compatible CUDA toolkit")
    os.environ.update(selected)


def force_natten_backend() -> None:
    """Honour $F3_NATTEN_BACKEND in the flux3 video VAE.

    The VAE hands its chosen backend to NATTEN inside ``attention_kwargs``,
    which NATTEN applies only to its additional-KV attention, so the
    neighbourhood attention itself always takes NATTEN's default (CUTLASS)
    path -- and the public aarch64 wheels carry no Thor (sm_110) kernels for
    it. Passing ``backend`` at the top level makes ``flex-fna`` (pure torch)
    usable there; $F3_NATTEN_COMPILE=1 also has NATTEN compile its flex
    attention, which otherwise runs unfused. NATTEN discourages that (it cannot
    test compiled flex everywhere), so check a compiled run's chunks against an
    uncompiled one (``bench_latency.py`` prints a fingerprint) before trusting it.
    """
    backend = os.environ.get("F3_NATTEN_BACKEND")
    if not backend:
        return
    compiled = os.environ.get("F3_NATTEN_COMPILE") == "1"
    if compiled and backend == "flex-fna":
        _configure_flex_compiler()
    from lerobot.policies.flux3.f3 import video_vae

    if compiled:
        from natten import allow_flex_compile

        allow_flex_compile()

    for name in ("na2d", "na3d"):
        original = getattr(video_vae, name)
        if getattr(original, "_travel_backend", None) == backend:
            continue

        @functools.wraps(original)
        def patched(*args, _original=original, **kwargs):
            kwargs.pop("attention_kwargs", None)
            if backend.startswith("cutlass") and len(args) >= 3:
                # The VAE hands NATTEN fp32 queries and keys with bf16 values; flex casts, CUTLASS refuses.
                dtype = args[2].dtype
                args = (args[0].to(dtype), args[1].to(dtype), *args[2:])
            return _original(*args, backend=backend, torch_compile=compiled, **kwargs)

        patched._travel_backend = backend
        setattr(video_vae, name, patched)


def fp32_trainable() -> None:
    """Give the parameters a run trains from scratch fp32 master weights.

    The flux3 policy loads its trunk in bf16, and PEFT copies the fresh action
    heads (``modules_to_save``) in that dtype, so AdamW updated them -- and kept
    their moments -- in bf16: at the heads' peak learning rate 38-88 % of the
    updates rounded away, 91-99.5 % in the cooldown. The LoRA weights were
    already fp32. Casting every trainable bf16 parameter to fp32 just before
    LeRobot builds its optimizer fixes that; the forward still runs under
    bf16 autocast, and a saved checkpoint loads into a bf16 policy as before.
    """
    import logging

    import torch
    from lerobot.scripts import lerobot_train

    original = lerobot_train.make_optimizer_and_scheduler
    if getattr(original, "_travel_fp32", False):
        return

    @functools.wraps(original)
    def make(cfg, policy, *args, **kwargs):
        cast = [name for name, p in policy.named_parameters() if p.requires_grad and p.dtype == torch.bfloat16]
        for _, p in policy.named_parameters():
            if p.requires_grad and p.dtype == torch.bfloat16:
                p.data = p.data.float()
        logging.getLogger("travel.train").info("fp32 master weights for %d trainable bf16 tensors: %s",
                                               len(cast), ", ".join(cast))
        return original(cfg, policy, *args, **kwargs)

    make._travel_fp32 = True
    lerobot_train.make_optimizer_and_scheduler = make


def train_main() -> int:
    """``lerobot-train`` with the travel fixes: fp32 trainable weights, and the NATTEN shim on a Thor."""
    force_natten_backend()
    fp32_trainable()
    from lerobot.scripts.lerobot_train import main

    return main()


def set_sampler(cfg, steps: int | None = None, guidance: float | None = None) -> None:
    """Override a flux3 config's sampler; guidance <= 1 also drops action guidance (one pass per step)."""
    if steps is not None:
        cfg.num_inference_steps = steps
    if guidance is not None:
        cfg.guidance_scale = guidance
        if guidance <= 1.0:
            cfg.guidance_scale_action = 1.0
