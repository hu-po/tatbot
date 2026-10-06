"""The model, and nothing else.

No Gradio, no FastAPI, no fleet routing, no quota policy, no simulator: those
belong to whoever is serving. This module owns one thing — the lifecycle of a
diffusion pipeline and one image per normalized request — so the Space, a local
worker and a batch job all draw the same picture from the same words.

Importing it loads no weights and touches no CUDA. `Engine.load()` is the only
step that does, and it is explicit because on ZeroGPU *when* the model is placed
on the device is part of the platform's contract, not an implementation detail.
"""
from __future__ import annotations

import hashlib
import io
import os
import threading
import time

from contracts import (
    PRODUCER,
    PRODUCER_VERSION,
    GenerationRequest,
    GenerationSettings,
    RequestError,
    result_document,
)

__all__ = ["Engine", "EngineError", "check_request", "resolve_revision"]


class EngineError(RuntimeError):
    """The model could not be loaded or could not draw."""


def check_request(loaded: GenerationSettings, request: GenerationRequest) -> None:
    """Refuse a request this engine cannot truthfully produce.

    A caller may name a model and a revision; the engine draws with the
    weights it has. Labelling the output as produced by something else would
    be a lie in every sidecar downstream, so a different model, or a pinned
    revision that differs from — or cannot be confirmed against — the loaded
    one, is refused before any GPU time is spent. Switching models is a
    deliberate restart with INKGEN_MODEL, never a per-request surprise.
    """
    wanted = request.settings
    if wanted.model != loaded.model:
        raise RequestError(f"this generator serves {loaded.model!r}, not {wanted.model!r}; "
                           "start one with INKGEN_MODEL set to that model")
    if wanted.model_revision and wanted.model_revision != loaded.model_revision:
        if loaded.model_revision is None:
            raise RequestError(f"this generator could not resolve the revision of {loaded.model!r}, so it "
                               f"cannot confirm the requested revision {wanted.model_revision}; "
                               "start it with INKGEN_MODEL_REVISION set to that commit")
        raise RequestError(f"this generator serves {loaded.model!r} at revision {loaded.model_revision}, "
                           f"not {wanted.model_revision}")


def resolve_revision(model: str, revision: str | None = None) -> str | None:
    """The immutable commit a moving model reference points at, right now.

    Resolved from the Hub's metadata, which is a small HTTP call and no
    download. `None` means it could not be determined — offline, a local path,
    or no Hub credentials — and a batch that requires a pin must refuse rather
    than record a name that will move underneath it.
    """
    if revision:
        return revision
    if os.path.isdir(model):
        return None
    try:
        from huggingface_hub import model_info

        return str(model_info(model).sha or "") or None
    except Exception:  # noqa: BLE001 - any Hub or network failure is "unknown"
        return None


class Engine:
    """One model instance. One worker owns one of these."""

    def __init__(self, settings: GenerationSettings | None = None, *, device: str | None = None,
                 dtype: object | None = None) -> None:
        self.settings = settings or GenerationSettings.from_env()
        self._device = device
        self._dtype = dtype
        self._pipe = None
        self._lock = threading.Lock()

    # ---- identity, without loading anything -------------------------------
    @property
    def device(self) -> str:
        if self._device is None:
            self._device = default_device()
        return self._device

    @property
    def loaded(self) -> bool:
        return self._pipe is not None

    def identity(self) -> dict:
        """What this engine is, for a health document or a job manifest."""
        return {"model": self.settings.model, "model_revision": self.settings.model_revision,
                "settings": self.settings.as_json(), "device": self.device,
                "loaded": self.loaded, "producer": PRODUCER, "producer_version": PRODUCER_VERSION}

    def pin(self, *, required: bool = False) -> str | None:
        """Resolve and adopt an immutable revision before work starts."""
        revision = resolve_revision(self.settings.model, self.settings.model_revision)
        if revision is None and required:
            raise EngineError(
                f"could not resolve an immutable revision for {self.settings.model!r}; "
                "set INKGEN_MODEL_REVISION to the exact commit this batch should use")
        self.settings = self.settings.pinned(revision)
        return revision

    # ---- lifecycle --------------------------------------------------------
    def load(self):
        """Build the pipeline and place it on the device. Idempotent."""
        with self._lock:
            if self._pipe is not None:
                return self._pipe
            try:
                import torch
                from diffusers import ZImagePipeline
            except ImportError as exc:  # pragma: no cover - depends on the host
                raise EngineError(f"the generation dependencies are not installed: {exc}") from exc
            dtype = self._dtype if self._dtype is not None else torch.bfloat16
            try:
                pipe = ZImagePipeline.from_pretrained(
                    self.settings.model, torch_dtype=dtype,
                    **({"revision": self.settings.model_revision} if self.settings.model_revision else {}))
                pipe.to(self.device)
            except Exception as exc:  # noqa: BLE001 - surfaced as one refusal
                raise EngineError(f"could not load {self.settings.model}: {exc}") from exc
            pipe.set_progress_bar_config(disable=True)
            self._pipe = pipe
            return pipe

    def close(self) -> None:
        """Release only what this engine owns; never another process's memory."""
        with self._lock:
            self._pipe = None
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:  # noqa: BLE001 - releasing a cache must never fail a job
            pass

    # ---- drawing ----------------------------------------------------------
    def check_request(self, request: GenerationRequest) -> None:
        """Refuse a model or revision these weights are not."""
        check_request(self.settings, request)

    def render(self, request: GenerationRequest):
        """One PIL image for one normalized request, with the loaded weights only."""
        if not isinstance(request, GenerationRequest):
            raise RequestError("render takes a normalized GenerationRequest")
        self.check_request(request)
        pipe = self.load()
        import torch

        generator_device = self.device if self.device == "cuda" else "cpu"
        generator = torch.Generator(device=generator_device).manual_seed(int(request.seed))
        settings = request.settings
        try:
            return pipe(prompt=request.prompt, height=settings.height, width=settings.width,
                        num_inference_steps=settings.steps, guidance_scale=settings.guidance,
                        generator=generator).images[0]
        except Exception as exc:  # noqa: BLE001 - one refusal, not a traceback
            raise EngineError(f"generation failed: {exc}") from exc

    def generate(self, request: GenerationRequest) -> tuple[bytes, dict]:
        """PNG bytes and the metadata describing exactly what produced them."""
        started = time.time()
        # An unpinned request drawn by pinned weights is reported as what it
        # was: the loaded revision, never null, and never a caller's guess.
        if request.settings.model_revision is None and self.settings.model_revision:
            request = request.pinned(self.settings.model_revision)
        image = self.render(request)
        buffer = io.BytesIO()
        image.save(buffer, format="PNG")
        png = buffer.getvalue()
        meta = result_document(request, png_sha256=hashlib.sha256(png).hexdigest(),
                               seconds=time.time() - started, device=self.device)
        return png, meta


def default_device() -> str:
    """cuda when a card is actually visible; ZeroGPU emulates one at import."""
    if os.environ.get("SPACE_ID") and os.environ.get("ZEROGPU", "1") != "0":
        try:
            import spaces  # noqa: F401

            return "cuda"
        except ImportError:
            pass
    try:
        import torch

        return "cuda" if torch.cuda.is_available() else "cpu"
    except ImportError:
        return "cpu"
