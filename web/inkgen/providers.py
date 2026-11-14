"""Explicit paid text-to-image providers. Keys are arguments, never provenance.

The local reproducible batch contract stays in contracts.py. Cloud results use
v2 because pinned revisions and seeds are not available from every service.
"""
from __future__ import annotations

import base64
import binascii
import hashlib
import io
import time
from dataclasses import dataclass

import provider_http as http
from contracts import PRODUCER, PRODUCER_VERSION, RequestError, canonical_digest, normalize_request

MODELS = {
    "fal": ("fal-ai/flux/schnell", "fal-ai/flux-2-pro"),
    "gemini": ("gemini-3.1-flash-image", "gemini-3-pro-image"),
    "openai": ("gpt-image-2.5-flare", "gpt-image-2.5-sunburst"),
}
LABELS = {"inkgen": "Inkgen — default, no key", "fal": "fal", "gemini": "Google Gemini", "openai": "OpenAI"}
TIMEOUT_S = 180


@dataclass(frozen=True)
class RemoteRequest:
    provider: str
    model: str
    prompt: str
    seed: int | None
    seed_requested: bool

    def parameters(self) -> dict:
        if self.provider == "fal":
            parameters = {"prompt": self.prompt, "seed": self.seed,
                          "image_size": {"width": 1024, "height": 1024}, "output_format": "png"}
            if self.model == "fal-ai/flux/schnell":
                parameters["num_images"] = 1
            return parameters
        if self.provider == "openai":
            return {"model": self.model, "prompt": self.prompt, "n": 1, "size": "1024x1024",
                    "quality": "medium", "output_format": "png"}
        return {"model": self.model, "input": [{"type": "text", "text": self.prompt}], "store": False,
                "response_format": {"type": "image", "mime_type": "image/jpeg",
                                    "aspect_ratio": "1:1", "image_size": "1K"}}

    @property
    def digest(self) -> str:
        return canonical_digest({"provider": self.provider, "model": self.model, "parameters": self.parameters()})


def normalize_remote(payload: dict) -> RemoteRequest:
    provider = payload.get("provider")
    if provider not in MODELS:
        raise RequestError("provider must be inkgen, fal, gemini or openai")
    if set(payload) - {"provider", "model", "subject", "style", "seed", "turnstile"}:
        raise RequestError("unsupported cloud generation setting; use subject, style, model and an optional fal seed")
    model = payload.get("model") or MODELS[provider][0]
    if model not in MODELS[provider]:
        raise RequestError("choose a supported model for this provider")
    if provider != "fal" and payload.get("seed") is not None:
        raise RequestError("this provider does not support a generation seed; select Random seed")
    request = normalize_request({key: payload[key] for key in ("subject", "style", "seed") if key in payload})
    return RemoteRequest(provider, model, request.prompt, request.seed if provider == "fal" else None,
                         provider == "fal" and request.seed_requested)


@dataclass(frozen=True)
class RemoteResult:
    png: bytes
    original: bytes
    media_type: str
    metadata: dict


def _decode(value: str) -> bytes:
    try:
        return base64.b64decode(value, validate=True)
    except (ValueError, TypeError, binascii.Error):
        raise http.ProviderError("provider returned invalid image data") from None


def _fal(request: RemoteRequest, key: str, send) -> tuple[bytes, dict]:
    headers = {"Authorization": f"Key {key}"}
    queued = http.document(send("POST", f"https://queue.fal.run/{request.model}", headers, request.parameters()))
    status_url = http.checked_url(queued["status_url"], ("queue.fal.run",))
    response_url = http.checked_url(queued["response_url"], ("queue.fal.run",))
    deadline = time.monotonic() + TIMEOUT_S
    while time.monotonic() < deadline:
        status = http.document(send("GET", status_url, headers, timeout=min(30, max(1, deadline - time.monotonic()))))
        if status.get("status") == "COMPLETED":
            result = http.document(send("GET", response_url, headers))
            url = http.checked_url(result["images"][0]["url"], ("fal.media",), subdomains=True)
            return send("GET", url, {}), {"seed": result.get("seed"), "request_id": queued.get("request_id")}
        if status.get("status") not in ("IN_QUEUE", "IN_PROGRESS"):
            raise http.ProviderError("fal did not complete this generation; check its request history")
        time.sleep(0.75)
    raise http.ProviderError("fal timed out; it may still finish and bill this request; check its request history", 504)


def _openai(request: RemoteRequest, key: str, send) -> tuple[bytes, dict]:
    result = http.document(send("POST", "https://api.openai.com/v1/images/generations",
                                {"Authorization": f"Bearer {key}"}, request.parameters()))
    return _decode(result["data"][0]["b64_json"]), {"usage": result.get("usage")}


def _gemini(request: RemoteRequest, key: str, send) -> tuple[bytes, dict]:
    result = http.document(send("POST", "https://generativelanguage.googleapis.com/v1beta/interactions",
                                {"x-goog-api-key": key, "Api-Revision": "2026-05-20"}, request.parameters()))
    images = [part for step in result.get("steps", []) if step.get("type") == "model_output"
              for part in step.get("content", []) if part.get("type") == "image" and part.get("data")]
    if not images:
        raise http.ProviderError("Gemini returned no image; the request may have been refused")
    return _decode(images[-1]["data"]), {"usage": result.get("usage")}


def _image(data: bytes) -> tuple[bytes, str, tuple[int, int]]:
    from PIL import Image, UnidentifiedImageError

    if len(data) > http.MAX_RESPONSE_BYTES:
        raise http.ProviderError("provider image exceeds the size limit")
    try:
        with Image.open(io.BytesIO(data)) as image:
            if image.format not in ("PNG", "JPEG", "WEBP") or image.width * image.height > 4096 * 4096:
                raise http.ProviderError("provider returned an unsupported image format or dimensions")
            media_type = Image.MIME[image.format]
            image.load()
            buffer = io.BytesIO()
            image.convert("RGBA" if "A" in image.getbands() else "RGB").save(buffer, format="PNG")
            png = data if image.format == "PNG" else buffer.getvalue()
            return png, media_type, image.size
    except (UnidentifiedImageError, OSError, ValueError, Image.DecompressionBombError):
        raise http.ProviderError("provider returned an unreadable image") from None


def _usage(value: object) -> dict:
    """Only numeric usage counters; never echo arbitrary upstream strings."""
    if not isinstance(value, dict):
        return {}
    names = ("input_tokens", "output_tokens", "total_tokens", "total_input_tokens", "total_output_tokens")
    return {name: value[name] for name in names if type(value.get(name)) in (int, float)}


def generate(request: RemoteRequest, key: str, *, send=None) -> RemoteResult:
    started = time.monotonic()
    try:
        original, facts = {"fal": _fal, "openai": _openai, "gemini": _gemini}[request.provider](
            request, key, send or http.request)
        png, media_type, size = _image(original)
    except (KeyError, IndexError, TypeError, AttributeError, ValueError):
        raise http.ProviderError("provider returned an incomplete generation response") from None
    seed = facts.get("seed") if type(facts.get("seed")) is int else None
    metadata = {
        "schema": "tatbot.inkgen-result/2", "provider": request.provider, "model": request.model,
        "model_revision": None, "prompt": request.prompt, "seed": seed, "seed_requested": request.seed_requested,
        "settings": {"model": request.model, "width": size[0], "height": size[1]},
        "parameters": request.parameters(), "request_sha256": request.digest,
        "png_sha256": hashlib.sha256(png).hexdigest(), "source_sha256": hashlib.sha256(original).hexdigest(),
        "source_media_type": media_type, "usage": _usage(facts.get("usage")),
        "seconds": round(time.monotonic() - started, 3), "device": "provider",
        "producer": PRODUCER, "producer_version": PRODUCER_VERSION,
    }
    return RemoteResult(png, original, media_type, metadata)
