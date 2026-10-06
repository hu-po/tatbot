"""inkgen — tattoo artwork from a few words, as source images for DrawingBot V3.

One process, three homes: a Hugging Face ZeroGPU Space (Gradio SDK, `python
app.py`), a GPU node in the fleet (`tatbot inkgen serve`), or any machine with
a CPU and patience. The frontend only needs the base URL.

    POST /api/generate  {"subject": "...", "seed": 123?, "style": "..."?, "turnstile": "..."?}
                        -> {"png_base64", "seed", "prompt", "seconds", "model", "model_revision",
                            "settings", "request_sha256", "png_sha256", ...}
    GET  /api/health    -> JSON

This file is the *public serving adapter*: quotas, Turnstile, CORS, the Gradio
page and the ZeroGPU lifecycle. What a request means and how an image is drawn
live in `contracts.py` and `engine.py`; who is admitted and how many draw at
once live in `serving.py`. None of them know about any of this, so the batch
worker and this Space cannot drift apart in what they ask for.

Both public entries — the HTTP API and the page — pass through the same
admission policy and the same inference gate: one render at a time, a bounded
line of callers waiting for it, and health probes answered while it draws.
The HTTP handler hands the render to a worker thread with its context copied,
which is what keeps the ZeroGPU scheduler's request context intact.

Off the Hub the process stops itself after INKGEN_IDLE_STOP_S (900) with no
generation, so `tatbot design generate` can start one on demand and forget it.
`INKGEN_FAKE_ENGINE=1` serves the whole thing without weights (see
`fake_engine.py`), for adapter tests and for driving the page on a laptop.

The default look is fixed in the prompt builder; an optional `style` phrase
replaces the look descriptors.
Model: Z-Image-Turbo (Apache-2.0), 6B, 8 steps, no guidance.
"""
from __future__ import annotations

import asyncio
import base64
import contextlib
import hashlib
import io
import json
import os
import tempfile
import threading
import time
from pathlib import Path

import gradio as gr
import httpx
from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import JSONResponse, Response

try:  # effect-free outside ZeroGPU Spaces (the decorator becomes a no-op)
    import spaces
except ImportError:  # pragma: no cover - local runs
    class _Spaces:
        @staticmethod
        def gpu(*_a, **_k):
            return (lambda f: f) if not _a or not callable(_a[0]) else _a[0]

        GPU = gpu  # the real package spells it this way
    spaces = _Spaces()  # type: ignore[assignment]

# Stdlib-only, model-free, so all of these are testable without a GPU or this module.
from contracts import (  # noqa: E402
    PRODUCER_VERSION,
    GenerationRequest,
    GenerationSettings,
    RequestError,
    normalize_request,
    normalize_seed,
    result_document,
    tattoo_prompt,
)
from engine import Engine, EngineError  # noqa: E402
from idle import IdleStop, idle_stop_seconds  # noqa: E402
from serving import AdmissionPolicy, InferenceGate, RefusalError  # noqa: E402

SETTINGS = GenerationSettings.from_env()
GPU_SECONDS_PER_CALL = int(os.environ.get("INKGEN_GPU_SECONDS", "12"))
ZERO_GPU = bool(os.environ.get("SPACE_ID")) and "spaces" in globals() and not isinstance(spaces, type)
# Visitor quotas exist to protect the owner's shared ZeroGPU allowance from
# strangers. They are a property of *how this process is served*, not of the
# request: a private worker a batch was pointed at owns its own job limits and
# must not inherit a six-per-minute public cap, or a 24-image job dies at
# image seven with HTTP 429. Off the Hub the caps default to off; any deploy
# that wants them states the numbers explicitly.
PUBLIC_SERVICE = bool(os.environ.get("SPACE_ID"))
FAKE_ENGINE = bool(os.environ.get("INKGEN_FAKE_ENGINE"))
TURNSTILE_SECRET = os.environ.get("TURNSTILE_SECRET", "")
EXAMPLE_SUBJECTS = ["a swallow carrying a rose", "a dagger through a heart", "a lunar moth"]


def _build_id() -> str:
    """Which payload this is, so a deploy can tell a new revision from the old.

    The stamp is written into the payload at upload time and travels with it;
    the Hub's own SPACE_COMMIT_SHA names the Space repo, not the source, and an
    unstamped local run says so rather than guessing.
    """
    stamp = os.path.join(os.path.dirname(os.path.abspath(__file__)), "build.json")
    try:
        with open(stamp) as handle:
            return str(json.load(handle)["build"])
    except (OSError, ValueError, KeyError):
        return os.environ.get("INKGEN_BUILD") or os.environ.get("SPACE_COMMIT_SHA") or "unstamped"


BUILD = _build_id()
BOOT = time.time()

IDLE = IdleStop(idle_stop_seconds(os.environ, zero_gpu=ZERO_GPU))
ADMISSION = AdmissionPolicy.from_env(public=PUBLIC_SERVICE)
GATE = InferenceGate.from_env()


def _engine():
    if FAKE_ENGINE:
        from fake_engine import FakeEngine

        return FakeEngine(SETTINGS, delay_s=float(os.environ.get("INKGEN_FAKE_DELAY_S") or 0))
    return Engine(SETTINGS, device="cuda" if ZERO_GPU else None)


ENGINE = _engine()
# The revision is resolved before the weights are, so the reply can say which
# commit drew the image even on the first request. A Hub outage leaves it null
# rather than failing the service: the public page is not a reproducible batch.
ENGINE.pin()
MODEL = ENGINE.settings.model
DEVICE = ENGINE.device
SIZE = ENGINE.settings.width
STEPS = ENGINE.settings.steps

# ---- model (placed on cuda at import: ZeroGPU emulates CUDA here and optimises
# the later transfer, so this stays an import-time load and not a lazy one).
ENGINE.load()


@spaces.GPU(duration=GPU_SECONDS_PER_CALL)
def render(request):
    return ENGINE.render(request)


def _cost_s() -> float:
    return float(GPU_SECONDS_PER_CALL if DEVICE == "cuda" else 0)


def serve(request: GenerationRequest, ip: str) -> tuple[bytes, dict]:
    """One admitted render, the same for every public entry.

    The line is joined before the quota is charged, so a caller refused for
    quota never touches the GPU and one refused for a full line is never
    charged. Work in flight holds the idle timer open.
    """
    ENGINE.check_request(request)
    t0 = time.time()
    with GATE.acquire():
        ADMISSION.admit(ip, _cost_s())
        with IDLE.hold():  # a render slower than the idle timeout is never cut in half
            image = render(request)
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    png = buffer.getvalue()
    # `seconds` stays wall-clock at this boundary (queueing included) and the
    # older keys keep their exact meaning, so an existing client reads this
    # reply unchanged and a new one gets the settings that were actually used.
    meta = result_document(request, png_sha256=hashlib.sha256(png).hexdigest(),
                           seconds=time.time() - t0, device=DEVICE)
    return png, meta


def _client_ip(req: Request) -> str:
    xff = req.headers.get("x-forwarded-for", "")
    return (xff.split(",")[0].strip() if xff else None) or (req.client.host if req.client else "?")


async def _verify_turnstile(token: str | None, ip: str) -> None:
    if not TURNSTILE_SECRET:
        return
    if not token:
        raise HTTPException(400, "missing turnstile token")
    async with httpx.AsyncClient(timeout=10) as c:
        r = await c.post("https://challenges.cloudflare.com/turnstile/v0/siteverify",
                         data={"secret": TURNSTILE_SECRET, "response": token, "remoteip": ip})
    if not r.json().get("success"):
        raise HTTPException(403, "turnstile verification failed")


# ---- HTTP API. Routes are prepended to Gradio's own FastAPI app after launch()
# (ZeroGPU only detects @spaces.GPU functions through Gradio's launch), so CORS is
# done by hand here rather than by middleware, and errors are returned, not raised.
router = APIRouter()
CORS = {"Access-Control-Allow-Origin": "*", "Access-Control-Allow-Methods": "GET, POST, OPTIONS",
        "Access-Control-Allow-Headers": "Content-Type", "Access-Control-Max-Age": "86400"}


def _json(data, status: int = 200, headers: dict[str, str] | None = None) -> JSONResponse:
    return JSONResponse(data, status_code=status, headers={**CORS, **(headers or {})})


def _refused(exc: RefusalError) -> JSONResponse:
    body: dict = {"error": exc.detail}
    headers: dict[str, str] = {}
    if exc.retry_after_s is not None:
        body["retry_after_s"] = exc.retry_after_s
        headers["Retry-After"] = str(exc.retry_after_s)
    return _json(body, exc.status, headers)


@router.options("/api/{rest:path}")
def preflight(rest: str):
    return Response(status_code=204, headers=CORS)


def health_document() -> dict:
    # A health probe is not use: it reports the idle countdown, never extends it.
    # Build and model identity are here so a deployment can be verified against
    # the revision that was uploaded; no private topology or credential is.
    return {"ok": True, "model": MODEL, "device": DEVICE, "zero_gpu": ZERO_GPU, "steps": STEPS, "size": SIZE,
            "turnstile": bool(TURNSTILE_SECRET), "public_service": PUBLIC_SERVICE, "fake_engine": FAKE_ENGINE,
            **ADMISSION.describe(), **GATE.state(),
            "uptime_s": round(time.time() - BOOT), "build": BUILD, "api_version": PRODUCER_VERSION,
            "model_revision": ENGINE.settings.model_revision, "settings": ENGINE.settings.as_json(),
            "loaded": ENGINE.loaded, **IDLE.state()}


@router.get("/api/health")
def health():
    return _json(health_document())


@router.post("/api/generate")
async def generate(req: Request):
    try:
        body = await req.json()
        if not isinstance(body, dict):
            raise RequestError("request must be a JSON object")
        request = normalize_request(body, defaults=ENGINE.settings)
        # A model or revision these weights are not is refused here, before
        # the line and before any quota is charged.
        ENGINE.check_request(request)
        ip = _client_ip(req)
        await _verify_turnstile(body.get("turnstile"), ip)
    except HTTPException as exc:
        return _json({"error": exc.detail}, exc.status_code)
    except RequestError as exc:
        return _json({"error": str(exc)}, 400)
    except (ValueError, TypeError):
        return _json({"error": "bad request"}, 400)
    t0 = time.time()
    try:
        # A worker thread, with this task's context copied into it: the event
        # loop keeps answering health while the engine draws, and the ZeroGPU
        # wrapper still finds the Gradio request it schedules against.
        png, meta = await asyncio.to_thread(serve, request, ip)
    except RefusalError as exc:
        return _refused(exc)
    except (EngineError, RequestError) as exc:
        return _json({"error": str(exc)}, 500 if isinstance(exc, EngineError) else 400)
    return _json({"png_base64": base64.b64encode(png).decode("ascii"), **meta,
                  "seconds": round(time.time() - t0, 1)})


# ---- Gradio page: the standalone product, and a manual bench for the Space.
DOWNLOADS = Path(tempfile.mkdtemp(prefix="inkgen-"))
_downloads_lock = threading.Lock()
_downloads: list[Path] = []


def _keep_download(name: str, data: bytes, keep: int = 24) -> str:
    """Write one downloadable file; the oldest are released past `keep`."""
    path = DOWNLOADS / name
    path.write_bytes(data)
    with _downloads_lock:
        _downloads.append(path)
        while len(_downloads) > keep:
            old = _downloads.pop(0)
            with contextlib.suppress(OSError):
                old.unlink()
    return str(path)


def _page_ip(request: gr.Request | None) -> str:
    """The page visitor's address, the way the API sees it, for the same quota."""
    if request is None:
        return "page"
    headers = getattr(request, "headers", None)
    forwarded = headers.get("x-forwarded-for", "") if headers is not None and hasattr(headers, "get") else ""
    if forwarded:
        return forwarded.split(",")[0].strip() or "page"
    client = getattr(request, "client", None)
    return getattr(client, "host", None) or "page"


def _status_text(doc: dict | None = None) -> str:
    """One line a person can act on, from evidence the service actually has."""
    doc = doc or health_document()
    where = "a stand-in engine (no model)" if doc["fake_engine"] else f"{doc['model']} on {doc['device']}"
    parts = [f"Ready — {where}" if doc["loaded"] else f"Loading {doc['model']}…"]
    if doc["budget_s"]:
        parts.append(f"daily GPU budget {doc['budget_spent_s']:.0f}/{doc['budget_s']} s")
    if doc["inference_in_flight"]:
        parts.append(f"drawing now ({doc['inference_waiting']} waiting)")
    if doc["idle_stop_in_s"] is not None:
        parts.append(f"stops after {doc['idle_stop_s']} s idle ({doc['idle_stop_in_s']:.0f} s left; a generation resets it)")
    return " · ".join(parts)


def page_generate(subject: str, random_seed: bool, seed: float | None, request: gr.Request | None = None):
    """The page's one action; the same admission and gate as the API."""
    # Same normalization as the API, so the page and the programmatic path
    # cannot disagree about what a subject, a style or seed zero mean. A
    # random seed is chosen once and reported; zero is a seed like any other.
    try:
        normalized = normalize_request(
            {"subject": subject or "", "seed": None if random_seed or seed is None else normalize_seed(seed)},
            defaults=ENGINE.settings)
        png, meta = serve(normalized, _page_ip(request))
    except RefusalError as exc:
        wait = f" Try again in about {exc.retry_after_s} s." if exc.retry_after_s else ""
        raise gr.Error(f"{exc.detail}.{wait}") from None
    except (RequestError, EngineError) as exc:
        raise gr.Error(str(exc)) from None
    stem = f"inkgen-{meta['seed']}-{meta['png_sha256'][:8]}"
    png_path = _keep_download(f"{stem}.png", png)
    meta_path = _keep_download(f"{stem}.json", json.dumps(meta, indent=2, sort_keys=True).encode())
    revision = f"@ {meta['model_revision'][:12]}" if meta["model_revision"] else "(revision unresolved)"
    caption = (f"seed {meta['seed']} · {meta['seconds']:.1f} s · {meta['model']} {revision}"
               + (" · stand-in engine" if FAKE_ENGINE else ""))
    return (png_path, caption, meta, gr.DownloadButton(value=png_path, visible=True),
            gr.DownloadButton(value=meta_path, visible=True), gr.Button(visible=True), _status_text())


def page_variation(subject: str, request: gr.Request | None = None):
    """The same subject, a seed nobody chose: that is what makes it a variation."""
    return page_generate(subject, True, None, request)


with gr.Blocks(title="Inkgen — tattoo artwork") as demo:
    gr.Markdown(
        "# Inkgen\n"
        "**Inkgen draws tattoo artwork from a few words.** Type a subject and get black tattoo-flash "
        "linework on white. The look is fixed; the subject and the seed are yours.")
    with gr.Row():
        subject = gr.Textbox(label="Subject", placeholder=EXAMPLE_SUBJECTS[0], max_lines=1, scale=4,
                             value=EXAMPLE_SUBJECTS[0])
        generate_button = gr.Button("Generate artwork", variant="primary", scale=1)
    gr.Examples(examples=[[example] for example in EXAMPLE_SUBJECTS], inputs=[subject], label="Example subjects")
    with gr.Row():
        random_seed = gr.Checkbox(label="Random seed (default)", value=True, scale=1)
        seed = gr.Number(label="Seed", value=0, precision=0, minimum=0, visible=False, scale=1,
                         info="Zero is a seed. The seed that was used is shown under the result.")
    random_seed.change(lambda random: gr.Number(visible=not random), inputs=[random_seed], outputs=[seed])
    result = gr.Image(label="Artwork", type="filepath", height=512, interactive=False)
    caption = gr.Markdown("")
    with gr.Row():
        variation_button = gr.Button("New variation", visible=False)
        png_download = gr.DownloadButton("Download PNG", visible=False)
        meta_download = gr.DownloadButton("Download generation metadata", visible=False)
    gr.Markdown("A variation is the same subject with a new seed. The same seed reproduces the same image on "
                "this generator; a different GPU or runtime may not draw identical pixels.")
    with gr.Row():
        status = gr.Markdown(_status_text())
        check_button = gr.Button("Check service", size="sm", scale=0)
    check_button.click(lambda: _status_text(), outputs=[status])
    with gr.Accordion("Details of the last generation", open=False):
        details = gr.JSON(label="Generation metadata", value=None)
    with gr.Accordion("API", open=False):
        gr.Markdown("```\nPOST /api/generate   {\"subject\": \"a swallow carrying a rose\", \"seed\": 42}   "
                    "→ {png_base64, seed, prompt, model, model_revision, settings, request_sha256, png_sha256, seconds}\n"
                    "GET  /api/health\n```\n"
                    "Past the per-address and daily limits the service answers 429/503 with `retry_after_s` "
                    "rather than spending anything. A model or revision other than the loaded one is refused (400).")
    outputs = [result, caption, details, png_download, meta_download, variation_button, status]
    generate_button.click(page_generate, inputs=[subject, random_seed, seed], outputs=outputs)
    subject.submit(page_generate, inputs=[subject, random_seed, seed], outputs=outputs)
    variation_button.click(page_variation, inputs=[subject], outputs=outputs)


if __name__ == "__main__":
    # One launch path everywhere. Gradio's launch() is what registers @spaces.GPU functions
    # with ZeroGPU on the Hub; locally it is just a server. Our API routes go in FRONT of
    # Gradio's routes so its catch-all does not swallow /api/*.
    port = int(os.environ.get("INKGEN_PORT") or os.environ.get("GRADIO_SERVER_PORT") or "7860")
    # Bind localhost by default (plan Phase 6: no unexpected network
    # listener); a Space must serve its container interface, and a deploy
    # that wants LAN exposure states INKGEN_HOST explicitly.
    host = os.environ.get("INKGEN_HOST") or (
        "0.0.0.0" if os.environ.get("SPACE_ID") else "127.0.0.1")
    gradio_app, _local, _share = demo.launch(server_name=host, server_port=port,
                                             prevent_thread_lock=True, ssr_mode=False,
                                             allowed_paths=[str(DOWNLOADS)])
    for route in reversed(router.routes):
        gradio_app.router.routes.insert(0, route)
    IDLE.start()
    idle_note = f"idle stop {IDLE.timeout_s} s" if IDLE.enabled else "no idle stop"
    print(f"inkgen: API on http://{host}:{port}/api/generate (model={MODEL}, device={DEVICE}, "
          f"zero_gpu={ZERO_GPU}, fake_engine={FAKE_ENGINE}, {idle_note})", flush=True)
    demo.block_thread()


__all__ = ["ENGINE", "demo", "router", "serve", "tattoo_prompt"]
