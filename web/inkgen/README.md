---
title: inkgen
emoji: 🖋️
colorFrom: gray
colorTo: pink
sdk: gradio
sdk_version: 5.49.1
python_version: "3.12"
app_file: app.py
pinned: false
short_description: Tattoo-flash artwork from a few words (Z-Image-Turbo).
---

# inkgen

Inkgen draws tattoo artwork from a few words: type a subject, get black
tattoo-flash linework on white, a source image for a DrawingBot V3
acquisition. Model:
[Z-Image-Turbo](https://huggingface.co/Tongyi-MAI/Z-Image-Turbo)
(Apache-2.0), 8 steps, no guidance, on ZeroGPU. The page keeps the last image
while another draws; a random seed is the default, zero is a seed, the seed
used is shown, and **New variation** is the same subject with a new seed. The
same seed reproduces the same image on this generator; a different GPU or
runtime may not draw identical pixels.

    POST /api/generate   {"subject": "a swallow carrying a rose", "seed": 42}
                         → {png_base64, seed, prompt, model, model_revision, settings,
                            request_sha256, png_sha256, seconds}
    GET  /api/health

Limits: a few requests per minute per address, a daily per-address cap, and a
daily GPU-seconds budget for the whole service, applied to the page and the
API alike; past those it answers 429/503 with `retry_after_s` rather than
spending anything. One image draws at a time, with a short line behind it; a
caller past the line is answered *busy* at once. A request naming a model or
revision other than the loaded one is refused (400) — the reply's identity is
what drew the image.

The same program runs off the Hub on the fleet's GPU node — from any
checkout: `tatbot inkgen ctl -- start|stop|status|logs` (the CLI hops to the
node with the `inkgen` role and manages the background process there),
`tatbot inkgen status` probes its health from any node, and `tatbot inkgen
deploy` publishes this directory to the Space. Details: `docs/inkmap.md`.
