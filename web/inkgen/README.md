---
title: inkgen
emoji: 🖋️
colorFrom: gray
colorTo: pink
sdk: gradio
sdk_version: 6.30.0
python_version: "3.12"
app_file: app.py
pinned: false
short_description: Tattoo art with Inkgen, fal, Gemini or OpenAI.
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

## Choose a provider

**Inkgen** remains the default and requires no paid-provider key. The page also
offers **fal**, **Google Gemini** and **OpenAI**, with a reviewed model picker.
This interface generates one source image from a subject; image editing,
reference-image uploads and arbitrary model endpoints are not part of this API.
All source images still require DrawingBot V3 acquisition before Inkmap placement.

For a cloud provider, paste the key itself into **Your provider API key**.
The password field masks it on screen. The generator backend receives it and
uses it only for that request's selected provider; using a hosted generator
therefore means trusting its operator with the key. It is never written to a
file, metadata, downloads, browser storage, or shared environment variables.
Gradio run history and monitoring are disabled. The field clears on provider
change, **Clear key**, refresh, and every 15 minutes. An already submitted
request keeps its credential until it finishes; clearing the field does not
cancel or refund a paid request. The API does not automatically retry paid jobs.

Hosted visitors always supply their own key, even when an operator has configured
Space Secrets. A missing or refused key never falls back to an operator key or
another provider. New variations on a cloud provider are additional paid requests.

Supported models:

| Provider | Models |
| --- | --- |
| fal | `fal-ai/flux/schnell`, `fal-ai/flux-2-pro` |
| Google Gemini | `gemini-3.1-flash-image`, `gemini-3-pro-image` |
| OpenAI | `gpt-image-2.5-flare`, `gpt-image-2.5-sunburst` |

fal requests one 1024 × 1024 PNG with a seed. OpenAI requests one 1024 × 1024
PNG at medium quality. Gemini requests a square 1K JPEG through Interactions,
with server-side interaction storage disabled; its original JPEG is retained
alongside the PNG preview. Availability, billing and content
rules belong to the selected provider and account. Seed and immutable model
revision are reported as unavailable when the service does not provide them.
The returned image's actual dimensions are recorded.

## Local configuration

The public repository includes `.env.example`. Copy it to the git-ignored
`.env` beside it, preserving any existing credentials. Fill only the optional
`FAL_KEY`, `GEMINI_API_KEY` or `OPENAI_API_KEY` you use. The backend reads these
literal assignments, including optional `export`; it does not execute shell
commands or expand variables. A `.env` beside `app.py` is also supported.
Process environment variables take precedence over both files.

`INKGEN_PROVIDER=inkgen` is the default. A local operator may explicitly set
`INKGEN_PROVIDER=fal`, `gemini` or `openai`; the service then skips loading local
weights and the launcher installs `requirements.txt.api`, with no GPU libraries.
To serve the local Inkgen model, use the normal `requirements.txt.local` install.
The CLI resolves configuration on the serving node. The optional browser key
overrides only that request's provider environment key.

For a server-backed Hugging Face Space, configure deployment keys under
**Settings → Secrets**. Environment variables are used directly and `.env` files
are ignored on Spaces. Keys must never be placed in static frontend build
variables: static Spaces expose configuration to browser JavaScript. Our public
Space starts with Inkgen regardless of `INKGEN_PROVIDER` and never lends its
deployment keys to visitors.

    POST /api/generate   {"subject": "a swallow carrying a rose", "seed": 42}
                         → {png_base64, seed, prompt, model, model_revision, settings,
                            request_sha256, png_sha256, seconds}
    GET  /api/health

Cloud requests add `provider` and optionally a model from the table above.
Supply the credential in the **`X-Provider-Key` HTTP header**, never the URL or
request JSON. `seed` is accepted for fal; omit it for Gemini and OpenAI.
`style` remains an optional replacement for the default look. Diffusion settings
such as `steps` and `guidance` belong to the local Inkgen request contract.

Local Inkgen replies retain `tatbot.inkgen-result/1` and the deterministic batch
workflow. Cloud replies use `tatbot.inkgen-result/2`: `provider`, `parameters`,
nullable `seed` and `model_revision`, numeric `usage` counters when available,
actual image dimensions, request digest and byte hashes. Credentials never enter
the request digest. `png_base64` is the PNG preview; `source_base64` retains the
exact provider output, identified by `source_media_type` and `source_sha256`.
The page offers both PNG and original-source downloads plus the metadata file.
Cloud results are not accepted as reproducible local batch evidence.

Provider interfaces: [fal](https://fal.ai/models/fal-ai/flux/schnell/api),
[Gemini](https://aistudio.google.com/docs/image-generation),
[OpenAI](https://developers.openai.com/api/docs/guides/image-generation).

Limits: a few requests per minute per address, a daily per-address cap, and a
daily GPU-seconds budget for the whole service, applied to the page and the
API alike; past those it answers 429/503 with `retry_after_s` rather than
spending anything. One image draws at a time, with a short line behind it; a
caller past the line is answered *busy* at once. A request naming a model or
revision other than the loaded one on a local Inkgen request is refused (400) — the reply's identity is
what drew the image.

The same program runs off the Hub on the fleet's GPU node — from any
checkout: `tatbot inkgen ctl -- start|stop|status|logs` (the CLI hops to the
node with the `inkgen` role and manages the background process there),
`tatbot inkgen status` probes its health from any node, and `tatbot inkgen
deploy` publishes this directory to the Space. Details: `docs/inkmap.md`.
