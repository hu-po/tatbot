#!/usr/bin/env python3
"""Build a local viewer for a directory of sim_cinematic renders.

The clips are 1080x1920 h264 and the point of looking at them is to see them
as they will be posted, so this writes a page that plays the actual files —
no re-encoding, no data URIs, native seeking.

    python scripts/render_viewer.py ~/renders/social-v2
    xdg-open ~/renders/social-v2/index.html

It writes index.html INTO the render directory and inlines the metadata from
each take's JSON sidecar. Inlining is not an optimisation: Chrome blocks
fetch() on file:// URLs, so a page that loaded its own sidecars would come up
empty when opened directly, and opening it directly is what gives the browser
real file access and working seek. (Python's http.server does not answer Range
requests, so serving it is the worse option, not the better one.)
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

HEAD = r"""<!doctype html>
<html lang="en" data-theme="dark">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Screening room — tatbot sim renders, 2026-08-27</title>
<style>
:root {
  --ground: #0A0C0F;
  --panel: #13161B;
  --panel-2: #191D23;
  --line: #262B33;
  --line-2: #333A44;
  --ink: #E9E7E3;
  --muted: #8B929C;
  --paper: #6E9BE8;
  --erase: #DDAB43;
  --tattoo: #E4738F;
  --sans: ui-sans-serif, system-ui, -apple-system, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;
  --mono: ui-monospace, "SF Mono", "JetBrains Mono", "DejaVu Sans Mono", Menlo, Consolas, monospace;
}
* { box-sizing: border-box; }
html, body { margin: 0; background: var(--ground); color: var(--ink); }
body { font-family: var(--sans); font-size: 15px; line-height: 1.55; -webkit-font-smoothing: antialiased; }
.wrap { max-width: 1500px; margin: 0 auto; padding: 0 28px 100px; }

.eyebrow {
  font-family: var(--mono); font-size: 10.5px; letter-spacing: .16em;
  text-transform: uppercase; color: var(--muted);
}

header { padding: 56px 0 26px; border-bottom: 1px solid var(--line); }
h1 {
  margin: 14px 0 0; font-size: clamp(30px, 4.6vw, 52px); font-weight: 650;
  letter-spacing: -0.03em; line-height: 1.02;
}
h1 em { font-style: normal; color: var(--muted); font-weight: 400; }
.sub { margin: 16px 0 0; color: var(--muted); max-width: 74ch; }
.sub b { color: var(--ink); font-weight: 550; }

/* ------------------------------------------------------------- controls */
.bar {
  position: sticky; top: 0; z-index: 50; display: flex; gap: 10px;
  align-items: center; flex-wrap: wrap; padding: 12px 0;
  background: rgba(10, 12, 15, .93); backdrop-filter: blur(10px);
  border-bottom: 1px solid var(--line);
}
button {
  font-family: var(--mono); font-size: 11.5px; letter-spacing: .06em;
  color: var(--ink); background: var(--panel); border: 1px solid var(--line-2);
  padding: 8px 14px; border-radius: 3px; cursor: pointer;
}
button:hover { border-color: var(--muted); }
button:focus-visible { outline: 2px solid var(--paper); outline-offset: 2px; }
button[aria-pressed="true"] { background: var(--ink); color: var(--ground); border-color: var(--ink); }
.bar .spacer { flex: 1; }
.bar .hint { font-family: var(--mono); font-size: 11px; color: var(--muted); }

/* --------------------------------------------------------------- takes */
section { padding: 44px 0 0; }
.take-head { display: flex; align-items: baseline; gap: 16px; flex-wrap: wrap; }
.take-head h2 {
  margin: 0; font-size: 21px; font-weight: 600; letter-spacing: -0.01em;
  padding-left: 14px; border-left: 3px solid var(--accent);
}
.take-head .who { font-family: var(--mono); font-size: 12px; color: var(--accent); }
.prompt {
  margin: 14px 0 0; font-family: var(--mono); font-size: 13px; color: var(--ink);
  background: var(--panel); border: 1px solid var(--line); border-left: 3px solid var(--accent);
  padding: 12px 16px; border-radius: 3px; max-width: 92ch;
}
.facts {
  display: flex; flex-wrap: wrap; gap: 6px 22px; margin-top: 12px;
  font-family: var(--mono); font-size: 11.5px; color: var(--muted);
}
.facts b { color: var(--ink); font-weight: 500; }

.row { display: grid; grid-template-columns: repeat(3, 1fr); gap: 18px; margin-top: 20px; }
figure { margin: 0; background: var(--panel); border: 1px solid var(--line); border-radius: 4px; overflow: hidden; }
.vidwrap { position: relative; background: #000; cursor: zoom-in; }
video { display: block; width: 100%; height: auto; }
.badge {
  position: absolute; top: 10px; left: 10px; font-family: var(--mono); font-size: 10px;
  letter-spacing: .12em; text-transform: uppercase; color: var(--ground);
  background: var(--accent); padding: 4px 8px; border-radius: 2px; font-weight: 600;
}
figcaption {
  display: flex; justify-content: space-between; align-items: center; gap: 10px;
  padding: 11px 14px; font-family: var(--mono); font-size: 11px; color: var(--muted);
  border-top: 1px solid var(--line);
}
figcaption a { color: var(--muted); text-decoration: none; border-bottom: 1px solid var(--line-2); }
figcaption a:hover { color: var(--ink); }

/* ------------------------------------------------------------- lightbox */
dialog {
  border: none; padding: 0; background: transparent; max-width: 100vw; max-height: 100vh;
}
dialog::backdrop { background: rgba(5, 6, 8, .93); }
dialog video { max-height: 92vh; width: auto; border-radius: 4px; }
.lb { display: flex; flex-direction: column; align-items: center; gap: 12px; }
.lb .meta { font-family: var(--mono); font-size: 12px; color: var(--muted); }

footer {
  margin-top: 66px; padding-top: 20px; border-top: 1px solid var(--line);
  font-family: var(--mono); font-size: 11.5px; color: var(--muted);
  display: flex; flex-wrap: wrap; gap: 8px 26px;
}

@media (max-width: 1100px) { .row { grid-template-columns: repeat(2, 1fr); } }
@media (max-width: 720px)  { .row { grid-template-columns: 1fr; } .wrap { padding: 0 16px 60px; } }
@media (prefers-reduced-motion: reduce) { * { animation: none !important; transition: none !important; } }
</style>
</head>
<body>
<div class="wrap">

<header>
  <div class="eyebrow">tatbot sim · path-traced takes</div>
  <h1>Screening room <em>— nine takes, three distributions</em></h1>
  <p class="sub">Every clip is <b>1080 × 1920, ray-traced</b> (SAPIEN <span class="mono">rt-med</span>,
  4 samples per pixel, OptiX-denoised), rendered from the same recipes the data factory generates
  training data from — but under a fixed studio rig instead of the randomized one, on studio
  surfaces instead of the bench's deliberately noisy textures. These are the files as they would
  be posted; nothing here has been re-compressed for the page.</p>
</header>

<div class="bar">
  <button id="playall" aria-pressed="true">❙❙ pause all</button>
  <button id="restart">↺ restart all</button>
  <button data-rate="0.25">0.25×</button>
  <button data-rate="0.5">0.5×</button>
  <button data-rate="1" aria-pressed="true">1×</button>
  <button data-rate="2">2×</button>
  <span class="spacer"></span>
  <span class="hint">click any clip to enlarge · space plays/pauses · silent by design</span>
</div>

<div id="takes"></div>

<footer>
  <span>9 clips · 63 MB · h264 yuv420p</span>
  <span>rendered by scripts/sim_cinematic.py</span>
  <span>source files sit beside this page</span>
</footer>

</div>

<dialog id="lightbox"><div class="lb">
  <video id="lbvideo" loop muted playsinline controls></video>
  <div class="meta" id="lbmeta"></div>
</div></dialog>

<script>
"""

TAIL = r"""
const ACCENT = { "paper-draw": "var(--paper)", "skin-erase": "var(--erase)", "skin-tattoo": "var(--tattoo)" };
const SHOT_NOTE = {
  orbit: "camera arcs 95° around the work",
  pov:   "the robot's own upper wrist camera",
  macro: "rides the tool tip, smoothed",
};
const TITLE = {
  "paper-draw":  "Ballpoint on ruled paper",
  "skin-erase":  "Laser removing ink from silicone",
  "skin-tattoo": "3RL liner depositing on silicone",
};

function load() {
  const takes = Object.entries(TAKES).map(([stem, meta]) => ({ stem, meta }));
  document.getElementById("takes").innerHTML = takes.map(({ stem, meta }) => {
    const a = ACCENT[meta.distribution] || "var(--muted)";
    const speed = meta.speed === 1 ? "real time" : meta.speed + "× speed";
    const secs = (meta.frames / meta.fps).toFixed(1);
    return `
    <section style="--accent:${a}">
      <div class="take-head">
        <h2>${TITLE[meta.distribution] || meta.distribution}</h2>
        <span class="who">${meta.distribution} · ${meta.tool} · ${meta.substrate}</span>
      </div>
      <p class="prompt">“${meta.prompt}”</p>
      <div class="facts">
        <span>seed <b>${meta.seed}</b></span>
        <span><b>${meta.size[0]}×${meta.size[1]}</b></span>
        <span><b>${meta.fps}</b> fps · <b>${speed}</b></span>
        <span><b>${secs}s</b> · ${meta.frames} frames</span>
        <span>shader <b>${meta.shader}</b></span>
        <span>exposure <b>${meta.exposure}</b></span>
        <span>ink on surface at end <b>${(100 * meta.ink_coverage_end).toFixed(2)}%</b></span>
      </div>
      <div class="row">
        ${meta.shots.map(shot => {
          const file = `${stem}-${shot}.mp4`;
          return `<figure>
            <div class="vidwrap" data-file="${file}" data-label="${meta.distribution} · ${shot}">
              <span class="badge">${shot}</span>
              <video src="${file}" poster="${stem}-${shot}.poster.jpg"
                     loop muted playsinline autoplay preload="auto"></video>
            </div>
            <figcaption><span>${SHOT_NOTE[shot] || ""}</span><a href="${file}" download>save</a></figcaption>
          </figure>`;
        }).join("")}
      </div>
    </section>`;
  }).join("");
  wire();
}

function wire() {
  const vids = [...document.querySelectorAll("section video")];
  const playall = document.getElementById("playall");
  let playing = true;

  function setPlaying(on) {
    playing = on;
    vids.forEach(v => on ? v.play().catch(() => {}) : v.pause());
    playall.setAttribute("aria-pressed", String(on));
    playall.textContent = on ? "❙❙ pause all" : "▶ play all";
  }
  playall.onclick = () => setPlaying(!playing);
  document.getElementById("restart").onclick = () => {
    vids.forEach(v => { v.currentTime = 0; });
    setPlaying(true);
  };
  document.querySelectorAll("[data-rate]").forEach(btn => {
    btn.onclick = () => {
      const rate = parseFloat(btn.dataset.rate);
      vids.forEach(v => { v.playbackRate = rate; });
      document.querySelectorAll("[data-rate]").forEach(b =>
        b.setAttribute("aria-pressed", String(b === btn)));
    };
  });

  const lb = document.getElementById("lightbox");
  const lbv = document.getElementById("lbvideo");
  const lbm = document.getElementById("lbmeta");
  document.querySelectorAll(".vidwrap").forEach(w => {
    w.onclick = () => {
      lbv.src = w.dataset.file;
      lbm.textContent = w.dataset.label + " — " + w.dataset.file;
      lb.showModal();
      lbv.play().catch(() => {});
    };
  });
  lb.addEventListener("click", e => { if (e.target === lb) lb.close(); });
  lb.addEventListener("close", () => { lbv.pause(); lbv.removeAttribute("src"); lbv.load(); });

  addEventListener("keydown", e => {
    if (e.code === "Space" && !lb.open) { e.preventDefault(); setPlaying(!playing); }
    if (e.key === "Escape" && lb.open) lb.close();
  });
}

load();
</script>
</body>
</html>
"""


def main(argv: list[str]) -> None:
    if not argv:
        raise SystemExit("usage: render_viewer.py <render-dir> [more-dirs...]")
    for arg in argv:
        root = Path(arg).expanduser()
        if not root.is_dir():
            raise SystemExit(f"not a directory: {root}")
        takes = {}
        for sidecar in sorted(root.glob("*.json")):
            meta = json.loads(sidecar.read_text())
            stem = sidecar.stem
            # only keep shots whose file actually landed — a killed render
            # leaves a sidecar describing clips that were never encoded
            shots = [s for s in meta.get("shots", []) if (root / f"{stem}-{s}.mp4").is_file()]
            if not shots:
                print(f"  skip {stem}: no clips on disk")
                continue
            meta["shots"] = shots
            takes[stem] = meta
        if not takes:
            raise SystemExit(f"no renders with clips in {root}")
        total = sum(f.stat().st_size for f in root.glob("*.mp4"))
        n_clips = sum(len(m["shots"]) for m in takes.values())
        page = (HEAD + "const TAKES = " + json.dumps(takes, indent=1) + ";\n" + TAIL)
        page = page.replace("9 clips · 63 MB · h264 yuv420p",
                            f"{n_clips} clips · {total / 1e6:.0f} MB · h264 yuv420p")
        page = page.replace("nine takes, three distributions",
                            f"{n_clips} clips, {len(takes)} takes")
        (root / "index.html").write_text(page)
        print(f"{root}/index.html — {len(takes)} takes, {n_clips} clips, "
              f"{total / 1e6:.0f} MB")
        print(f"  open: file://{(root / 'index.html').resolve()}")


if __name__ == "__main__":
    main(sys.argv[1:])
