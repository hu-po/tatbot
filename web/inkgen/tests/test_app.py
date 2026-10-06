"""The serving adapter, driven with a stand-in engine: no weights, no GPU.

Skipped where gradio/fastapi are not installed (the stdlib policy tests in
`test_serving.py` still run there); `scripts/inkgen_serve.sh`'s venv has both,
and that is where these are run for evidence.
"""
from __future__ import annotations

import asyncio
import base64
import os
import threading
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

pytest.importorskip("gradio")
pytest.importorskip("fastapi")
httpx = pytest.importorskip("httpx")

os.environ.update({"INKGEN_FAKE_ENGINE": "1", "INKGEN_PER_IP_PER_MIN": "2", "INKGEN_PER_IP_PER_DAY": "0",
                   "INKGEN_DAILY_BUDGET_S": "0", "INKGEN_MAX_WAITING": "2", "INKGEN_QUEUE_WAIT_S": "5",
                   "INKGEN_IDLE_STOP_S": "0", "INKGEN_MODEL_REVISION": "c" * 40})
os.environ.pop("SPACE_ID", None)

import app  # noqa: E402
import gradio as gr  # noqa: E402
from fastapi import FastAPI  # noqa: E402


def api() -> FastAPI:
    application = FastAPI()
    application.include_router(app.router)
    return application


def client() -> httpx.AsyncClient:
    return httpx.AsyncClient(transport=httpx.ASGITransport(app=api()), base_url="http://inkgen.test")


def page_request(ip: str) -> SimpleNamespace:
    return SimpleNamespace(headers={"x-forwarded-for": ip}, client=SimpleNamespace(host="127.0.0.1"))


def run(coroutine):
    return asyncio.run(coroutine)


def test_health_answers_while_a_render_is_in_flight():
    """The render runs off the event loop; the loop keeps serving probes."""
    inside, release = threading.Event(), threading.Event()
    app.ENGINE.render_hook = lambda _r: (inside.set(), release.wait(10))

    async def scenario():
        async with client() as c:
            pending = asyncio.ensure_future(c.post("/api/generate", json={"subject": "a swallow", "seed": 3},
                                                   headers={"x-forwarded-for": "198.51.100.1"}))
            assert await asyncio.to_thread(inside.wait, 10)
            probe = await asyncio.wait_for(c.get("/api/health"), 5)
            document = probe.json()
            assert document["inference_in_flight"] == 1
            assert document["requests_in_flight"] == 1, "the idle timer must be held while drawing"
            assert document["loaded"] is True and document["fake_engine"] is True
            release.set()
            reply = await pending
            assert reply.status_code == 200
            body = reply.json()
            assert base64.b64decode(body["png_base64"]).startswith(b"\x89PNG")
            assert body["seed"] == 3 and body["seed_requested"] is True
            assert body["model_revision"] == "c" * 40 and body["request_sha256"] and body["png_sha256"]
            assert body["settings"]["model"] == body["model"]
            after = (await c.get("/api/health")).json()
            assert after["inference_in_flight"] == 0 and after["requests_in_flight"] == 0
    try:
        run(scenario())
    finally:
        app.ENGINE.render_hook = None


def test_api_and_page_share_one_admission_policy():
    """Two per minute per address, whichever door the visitor uses."""
    async def scenario():
        async with client() as c:
            first = await c.post("/api/generate", json={"subject": "a swallow", "seed": 1},
                                 headers={"x-forwarded-for": "198.51.100.2"})
            assert first.status_code == 200
            # The page's generation is the quota's second of two.
            outputs = app.page_generate("a swallow", False, 5, page_request("198.51.100.2"))
            assert outputs[2]["seed"] == 5
            third = await c.post("/api/generate", json={"subject": "a swallow", "seed": 1},
                                 headers={"x-forwarded-for": "198.51.100.2"})
            assert third.status_code == 429
            assert third.json()["retry_after_s"] >= 1
            assert third.headers["retry-after"] == str(third.json()["retry_after_s"])
            with pytest.raises(gr.Error, match="slow down"):
                app.page_generate("a swallow", True, None, page_request("198.51.100.2"))
            other = await c.post("/api/generate", json={"subject": "a swallow", "seed": 1},
                                 headers={"x-forwarded-for": "198.51.100.3"})
            assert other.status_code == 200
    run(scenario())


def test_a_model_or_revision_these_weights_are_not_is_refused():
    calls = len(app.ENGINE.calls)

    async def scenario():
        async with client() as c:
            reply = await c.post("/api/generate", json={"subject": "a swallow", "seed": 1, "model": "x/y"},
                                 headers={"x-forwarded-for": "198.51.100.4"})
            assert reply.status_code == 400
            assert "serves" in reply.json()["error"] and "x/y" in reply.json()["error"]
            reply = await c.post("/api/generate", json={"subject": "a swallow", "seed": 1, "model_revision": "d" * 40},
                                 headers={"x-forwarded-for": "198.51.100.4"})
            assert reply.status_code == 400
            assert "revision" in reply.json()["error"]
    run(scenario())
    assert len(app.ENGINE.calls) == calls, "a refused identity must not draw"
    assert app.ADMISSION.describe()["budget_spent_s"] == 0


def test_page_seed_zero_random_seed_and_variation():
    zero = app.page_generate("a moth", False, 0, page_request("198.51.100.5"))
    assert zero[2]["seed"] == 0 and zero[2]["seed_requested"] is True
    assert "seed 0 ·" in zero[1]
    assert Path(zero[0]).read_bytes().startswith(b"\x89PNG")
    chosen = app.page_generate("a moth", True, 0, page_request("198.51.100.6"))
    assert chosen[2]["seed_requested"] is False  # random even though the hidden field says 0
    variation = app.page_variation("a moth", page_request("198.51.100.7"))
    assert variation[2]["seed_requested"] is False
    assert variation[2]["png_sha256"] != zero[2]["png_sha256"] or variation[2]["seed"] == 0
    assert variation[2]["model_revision"] == "c" * 40


def test_renders_are_serialised_across_concurrent_api_calls():
    app.ENGINE.delay_s = 0.05
    app.ENGINE.max_concurrent = 0

    async def scenario():
        async with client() as c:
            replies = await asyncio.gather(*(
                c.post("/api/generate", json={"subject": "a swallow", "seed": n}, headers={"x-forwarded-for": f"203.0.113.{n}"})
                for n in range(3)))
            return [r.status_code for r in replies]
    try:
        started = time.monotonic()
        codes = run(scenario())
    finally:
        app.ENGINE.delay_s = 0
    assert codes == [200, 200, 200]
    assert app.ENGINE.max_concurrent == 1
    assert time.monotonic() - started >= 0.15


def test_a_full_line_is_refused_busy_not_parked():
    inside, release = threading.Event(), threading.Event()
    app.ENGINE.render_hook = lambda _r: (inside.set(), release.wait(10))

    async def scenario():
        async with client() as c:
            post = lambda n: c.post("/api/generate", json={"subject": "a swallow", "seed": n},  # noqa: E731
                                    headers={"x-forwarded-for": f"203.0.113.1{n}"})
            running = asyncio.ensure_future(post(1))
            assert await asyncio.to_thread(inside.wait, 10)
            waiting = [asyncio.ensure_future(post(2)), asyncio.ensure_future(post(3))]
            for _ in range(200):
                if (await c.get("/api/health")).json()["inference_waiting"] == 2:
                    break
                await asyncio.sleep(0.01)
            refused = await post(4)
            assert refused.status_code == 503 and "busy" in refused.json()["error"]
            assert refused.headers["retry-after"]
            release.set()
            assert (await running).status_code == 200
            assert [r.status_code for r in await asyncio.gather(*waiting)] == [200, 200]
    try:
        run(scenario())
    finally:
        app.ENGINE.render_hook = None
