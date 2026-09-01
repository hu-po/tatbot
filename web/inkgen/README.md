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
short_description: Tattoo-flash generator behind Inkmap (Z-Image-Turbo).
---

# inkgen

The design generator behind [Inkmap](https://huggingface.co/spaces/tatbot/inkmap).
Type a subject, get black tattoo-flash linework; Inkmap then traces it to
vector ink in the browser. Model: [Z-Image-Turbo](https://huggingface.co/Tongyi-MAI/Z-Image-Turbo)
(Apache-2.0), 8 steps, no guidance, on ZeroGPU.

    POST /api/generate   {"subject": "a swallow carrying a rose", "seed": 42}   → image/png
    GET  /api/health

Limits: a few requests per minute per address, a daily per-address cap, and a
daily GPU-seconds budget for the whole service; past those it answers 429/503
rather than spending anything.

The same program runs off the Hub on the fleet's GPU node — from any
checkout: `tatbot inkgen start` / `status` / `logs` / `stop` (the CLI hops to
the node with the `inkgen` role), and `tatbot inkgen deploy` publishes this
directory to the Space. Details: `docs/inkmap.md`.
