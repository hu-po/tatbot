---
name: tatbot-flow
description: Generate tattoo-design source images with Google Flow (Nano Banana) in the operator's logged-in Chrome, and file them with provenance sidecars for the art pipeline. Use when asked to make design images with Flow, run a batch of art prompts, or collect Flow results into the art corpus.
---

# Google Flow for the art studio

Flow is the art studio's primary image generator. It has no API, so it is driven through Claude in
Chrome in the operator's signed-in browser. Image generation costs no credits; it is capped per day
(on Ultra, about 500 Nano Banana Pro images). Video costs credits, so this skill never makes video.

Hard rules:

- **Never offer Flow on the demo station.** The station is offline and has its own generator.
- **No unofficial API wrappers or cookie scraping outside the page.** Everything goes through the
  Flow tab.
- **Generate nothing that is not asked for.** Publish nothing.
- **Record what Flow actually received**, read back from the page, not what was meant to be typed.

## Setup

1. Load the `anthropic-skills:chrome-browser` skill. Then call `list_connected_browsers`.
2. If the downloads must land on this machine, prove the browser is local:
   - navigate a tab to `https://example.com/?<unique-marker>`;
   - search `~/.config/google-chrome/*/Sessions/*` for the marker bytes with a short `python3` loop
     (Chrome journals within about 10 s).
   A browser name such as "Browser 1" proves nothing.
   If `list_connected_browsers` is empty, the extension has dropped its connection, which happens after a
   quiet spell; `switch_browser` cannot reach it either. Ask the owner to open the Claude side panel in
   Chrome (signed in), then check again. The same device id comes back, so a browser verified earlier
   stays verified.
3. Open `https://flow.google.com/`. An `ULTRA` badge means the plan's caps apply. Make a **New project**
   per study, so its grid holds only that study.
4. The settings chip at the bottom right of the prompt box sets:
   - **Image** (not Video);
   - aspect: the plan's `aspect`, 1:1 for flash and 9:16 for a forearm18 tile round, otherwise the canvas's
     (Flow offers 16:9, 4:3, 1:1, 3:4 and 9:16 for images, 2026-11-08);
   - model: **Nano Banana Pro** follows prompts best; 2.1 and 2 Lite are the cheaper caps;
   - outputs: x1–x4.
   Read the chip back after every change. A click that did not take looks the same as one that did.

## Submitting prompts: the merge race

Observed 2026-10-30:
- Return in the prompt box does not reliably submit.
- With about four or more batches running, Flow refuses a new submission silently and leaves the
  text in the box.
- Clicking into a non-empty box drops the caret mid-text, so the next prompt typed is *inserted
  inside* the old one. Three batches came back generated from two merged prompts.

Submit each prompt like this:

1. Click the prompt box, press `ctrl+a`, then `Delete`.
2. Type the prompt.
3. Click the **send arrow** (bottom right of the box), not Return.
4. Wait about 4 s. In `javascript_tool`, confirm both:
   - the box (`[contenteditable="true"]`) is empty;
   - the first `div.batch-container`'s text, after scrolling the grid to the top, starts with your
     prompt.
5. If the box still holds the text, wait 10–15 s and click send again. Keep at most two batches in
   flight.

Write the planned prompts first to `<corpus>/flow/prompts.json`:

```json
{"generator": "flow-ui", "model": "...", "aspect": "...", "project": "<url>",
 "prompts": [{"id": "", "style": "", "subject": "", "text": "", "variant": null}]}
```

## Collecting the results

- The grid is virtualized: only tiles near the viewport exist in the DOM.
- Each image is a signed `flow-content.google/image/<id>` URL, since 2026-11-06. It already is the
  original, a 1024×1024 JPEG for Nano Banana Pro at 1:1. Fetch it with `credentials: 'omit'`, because
  `include` fails its CORS check.
- Chrome throttles bursts of programmatic downloads, so download one zip, not many files.

1. Run [collect_zip.js](collect_zip.js) in the project tab with `javascript_tool`. It scrolls the grid,
   reads each batch's prompt and images, fetches the originals and downloads one zip
   (`batches.json` and `bNN-K.jpg`).
2. Confirm the zip landed in `~/Downloads`. Extract it into a new, empty directory with
   `python3 -I` and `zipfile`, checking that no member name has a `/`.
3. File it:

   ```bash
   python3 -I .claude/skills/tatbot-flow/file_batches.py EXTRACTED CORPUS/flow CORPUS/flow/prompts.json
   ```

   Each image gets a `tatbot.art-generation/1` sidecar: generator, model, aspect, project, the read-back
   prompt, prompt id, style, subject, references and the image's sha256. Unmatched (merged) batches
   are filed apart and flagged.
4. Keep the corpus outside the repository. Images are never committed.

## Prompting for plottable flash

What held on 2026-10-30, with Nano Banana Pro, 1:1, ten prompts × 3 images. On-palette is the share
of inked pixels within ΔE00 12 of a requested ink.

| Pattern | Result |
| --- | --- |
| Name the style, "bold even black outlines of a single weight", "flat solid colour fills, no gradients, no blending, no soft shading", "isolated on a plain white background, centered" | American traditional came back flat, outlined, centred and 72–87% on-palette |
| Name the inks: "Use only five inks: black, bright red, …" | Followed. Off-palette pixels were mostly antialiasing and a grey rock or two |
| Add hex codes for each ink | The same share on-palette, but the median colour error fell from 6.8 to 4.0 ΔE00 (3 + 3 images). Worth adding |
| "Linocut / relief print" | Comes back as a **photograph of a print**: off-white paper, shadow, once a wooden frame edge. Adding "Flat digital artwork, not a photograph: pure white background, no paper texture, no shadow, no frame" fixed 3 of 3 |
| "No text" on a vehicle | Signage still appears ("POWELL & HYDE"). Say "no signs, no lettering anywhere", and check |
| "Seed-of-life circle grid" | The grid dominates and the subject turns abstract. Colours are obeyed exactly, but thin construction lines antialias to grey (24–51% on-palette by that measure). "The {subject}'s silhouette is bold and clearly readable; the grid stays light behind it" fixed legibility 3 of 3 |

- Style-specific templates live in the `tatbot-style-*` skills.
- Every result still goes through palette snapping and separation (`tatbot art`). Flow's colours are
  suggestions to the model, and the bottles decide.
- Ingredients (reference images) are dragged into the prompt box or added with `+`. The palette sheet
  from `tatbot art swatch` is the intended palette reference. Its effect is not yet measured, so test
  it against a no-reference batch before relying on it.

## When the UI changes

Flow changes often. If a selector or position fails, take a screenshot, adapt, and update this skill
with the date and what changed.
