"""inkgen — the design generator behind Inkmap's "Generate a design" panel.

One process, three homes: a Hugging Face ZeroGPU Space (Gradio SDK, `python
app.py`), a GPU node in the fleet (`tatbot inkgen serve`), or any machine with
a CPU and patience. The frontend only needs the base URL.

    POST /api/generate  {"subject": "...", "seed": 123?, "style": "..."?, "turnstile": "..."?}
                        -> {"png_base64", "seed", "prompt", "seconds"}
    GET  /api/health    -> JSON

The default look is fixed here (prompt blurb); an optional `style` phrase —
Inkmap derives it from the inklang style slots (config/inkmap/styles.json) —
replaces the look descriptors. The browser traces the PNG to black
ink and SVG. Model: Z-Image-Turbo (Apache-2.0), 6B, 8 steps, no guidance.
"""
from __future__ import annotations

import base64
import io
import os
import threading
import time
from collections import defaultdict, deque

import gradio as gr
import httpx
import torch
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

MODEL = os.environ.get("INKGEN_MODEL", "Tongyi-MAI/Z-Image-Turbo")
STEPS = int(os.environ.get("INKGEN_STEPS", "8"))
SIZE = int(os.environ.get("INKGEN_SIZE", "768"))
GPU_SECONDS_PER_CALL = int(os.environ.get("INKGEN_GPU_SECONDS", "12"))
ZERO_GPU = bool(os.environ.get("SPACE_ID")) and "spaces" in globals() and not isinstance(spaces, type)
DEVICE = "cuda" if (torch.cuda.is_available() or ZERO_GPU) else "cpu"
DTYPE = torch.bfloat16
PER_IP_PER_MIN = int(os.environ.get("INKGEN_PER_IP_PER_MIN", "6"))
PER_IP_PER_DAY = int(os.environ.get("INKGEN_PER_IP_PER_DAY", "60"))
DAILY_BUDGET_S = int(os.environ.get("INKGEN_DAILY_BUDGET_S", "1800"))  # GPU-seconds/day for the whole service
TURNSTILE_SECRET = os.environ.get("TURNSTILE_SECRET", "")
ALLOW_ORIGINS = [o for o in os.environ.get("INKGEN_ALLOW_ORIGINS", "*").split(",") if o]
BOOT = time.time()


def tattoo_prompt(subject: str, style: str | None = None) -> str:
    """Must match web/inkmap/src/core/gen.ts. `style` replaces the default look."""
    s = " ".join(subject.split())
    look = " ".join((style or "").split())[:160] or \
        "clean black linework suitable for vector tracing"
    return (f"tattoo flash design of {s}, {look}, "
            "isolated on a plain white background, centered, no text")


# ---- model (placed on cuda at import: ZeroGPU emulates CUDA here and optimises the later transfer)
from diffusers import ZImagePipeline  # noqa: E402

pipe = ZImagePipeline.from_pretrained(MODEL, torch_dtype=DTYPE)
pipe.to(DEVICE)
pipe.set_progress_bar_config(disable=True)


@spaces.GPU(duration=GPU_SECONDS_PER_CALL)
def render(prompt: str, seed: int):
    g = torch.Generator(device=DEVICE if DEVICE == "cuda" else "cpu").manual_seed(seed)
    return pipe(prompt=prompt, height=SIZE, width=SIZE, num_inference_steps=STEPS, guidance_scale=0.0, generator=g).images[0]


# ---- limits: cheap, in-memory, per replica. The point is protecting the owner's ZeroGPU quota.
_lock = threading.Lock()
_per_ip: dict[str, deque[float]] = defaultdict(deque)
_budget: deque[tuple[float, float]] = deque()  # (t, seconds)


def _client_ip(req: Request) -> str:
    xff = req.headers.get("x-forwarded-for", "")
    return (xff.split(",")[0].strip() if xff else None) or (req.client.host if req.client else "?")


def _check_limits(ip: str, cost_s: float) -> None:
    now = time.time()
    with _lock:
        q = _per_ip[ip]
        while q and now - q[0] > 86400:
            q.popleft()
        if sum(1 for t in q if now - t < 60) >= PER_IP_PER_MIN:
            raise HTTPException(429, "slow down — a few per minute is plenty")
        if len(q) >= PER_IP_PER_DAY:
            raise HTTPException(429, "daily limit for this address reached")
        while _budget and now - _budget[0][0] > 86400:
            _budget.popleft()
        if sum(s for _, s in _budget) + cost_s > DAILY_BUDGET_S:
            raise HTTPException(503, "the generator's daily budget is spent — try again tomorrow")
        q.append(now)
        _budget.append((now, cost_s))


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


def _json(data, status: int = 200) -> JSONResponse:
    return JSONResponse(data, status_code=status, headers=CORS)


@router.options("/api/{rest:path}")
def preflight(rest: str):
    return Response(status_code=204, headers=CORS)


@router.get("/api/health")
def health():
    with _lock:
        spent = sum(s for _, s in _budget)
    return _json({"ok": True, "model": MODEL, "device": DEVICE, "zero_gpu": ZERO_GPU, "steps": STEPS, "size": SIZE,
                  "turnstile": bool(TURNSTILE_SECRET), "budget_s": DAILY_BUDGET_S, "budget_spent_s": round(spent, 1),
                  "uptime_s": round(time.time() - BOOT)})


@router.post("/api/generate")
async def generate(req: Request):
    try:
        body = await req.json()
        subject = str(body.get("subject", "")).strip()
        if not 1 <= len(subject) <= 120:
            raise HTTPException(400, "subject must be 1-120 characters")
        seed = int(body.get("seed") or int.from_bytes(os.urandom(4), "big") % 1_000_000)
        style = str(body.get("style") or "").strip() or None
        if style is not None and len(style) > 160:
            raise HTTPException(400, "style must be at most 160 characters")
        ip = _client_ip(req)
        await _verify_turnstile(body.get("turnstile"), ip)
        _check_limits(ip, float(GPU_SECONDS_PER_CALL if DEVICE == "cuda" else 0))
    except HTTPException as exc:
        return _json({"error": exc.detail}, exc.status_code)
    except (ValueError, TypeError):
        return _json({"error": "bad request"}, 400)
    prompt = tattoo_prompt(subject, style)
    t0 = time.time()
    img = render(prompt, seed)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return _json({"png_base64": base64.b64encode(buf.getvalue()).decode("ascii"), "seed": seed, "prompt": prompt,
                  "seconds": round(time.time() - t0, 1), "model": MODEL})


# ---- Gradio page for the Space (also usable as a manual test bench)
def demo_generate(subject: str, seed: float | None):
    s = int(seed) if seed else int.from_bytes(os.urandom(4), "big") % 1_000_000
    return render(tattoo_prompt(subject or "a swallow"), s), s


demo = gr.Interface(
    fn=demo_generate,
    inputs=[gr.Textbox(label="subject", value="a swallow carrying a rose"), gr.Number(label="seed (blank = random)", precision=0)],
    outputs=[gr.Image(label="flash", type="pil"), gr.Number(label="seed used", precision=0)],
    title="inkgen",
    description=("The generator behind Inkmap. Fixed style: black tattoo flash linework. "
                 "API: POST /api/generate {subject, seed} → PNG; GET /api/health."),
    flagging_mode="never",
)

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
                                             prevent_thread_lock=True, ssr_mode=False)
    for route in reversed(router.routes):
        gradio_app.router.routes.insert(0, route)
    print(f"inkgen: API on http://{host}:{port}/api/generate (model={MODEL}, device={DEVICE}, zero_gpu={ZERO_GPU})", flush=True)
    demo.block_thread()
