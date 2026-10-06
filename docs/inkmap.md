---
summary: Inkmap tattoo design placement and MHR/SOMA preview
tags: [web, design, preview]
updated: 2026-09-05
audience: [artist, dev, contributor]
---

# Inkmap

`web/inkmap/` is a browser application for placing a vector design on a 3D
body preview. It produces a versioned placement record; it does not command a
robot or upload a design by itself.

Inkmap uses [InkLang](inklang.md) to turn placement descriptions into
canonical rest-surface anchors. Inkmap is the editor and visualizer; it does
not define a second placement grammar or independently choose semantic
regions.

Type a placement-only phrase such as `on the left forearm`. Inkmap shows the
normalized phrase and resolves it before a design is committed. If the phrase
omits a required side, names a multi-site zone, or leaves a relative direction
open, Inkmap shows concrete candidates and waits for a choice. Artwork
generation cannot start from that request until the location is resolved. A
manual click is the inverse input mode: the atlas describes and validates the
chosen anchor through the same contract.

## Develop

```bash
cd web/inkmap
npm ci
npm run check
npm run dev
```

The app should work with local fixtures and no private service. Keep generated
images and personal designs outside Git unless their license permits sharing.
**Simulation export**, under the top bar menu's **Advanced** section in the body
workspace, downloads an immutable bundle with all accepted placements, faithful artwork, pose/support, seed,
camera/appearance, and nominal tool ID. Compile it offline with
`tatbot sim compile FILE -- --output SCENARIO`; select `--placement-id ID` for
multi-placement drawing input. Body bytes resolve only through the pinned local
cache. The typed v3 compiler and deterministic simulator target-label
materializer are available; the latter emits soft coverage/color and discrete
IDs without pre-inking the episode. `tatbot sim perception SCENARIO -- \
--output-dir DIR` adds audited dense camera-space labels as privileged NPZ
sidecars; they are not deployed-policy observations. **Open compiled preview** revalidates the
portable bundle/program/trace bindings before loading the exact placement and
named pose. It is a visual preview, not the CLI's local execution validation or
motion approval. Automated mapping/mask parity passes; human and GPU rendered
scene review remain pending.

Before assigning production compute, `tatbot sim pilot-plan -- --output-dir
DIR --seed 42` writes the audited pilot ledger without launching a renderer.
It fixes the cross-product at six site/pose/support cells, three acquired DBV3
artworks, two appearances, and three views: 36 reference scenes and
108 reference images. The selected three-identity expansion is 108 scenes and
324 images, but its two non-reference identities stay blocked until their
visual review is accepted. The same ledger assigns 12 drawing episodes across
every cell and artwork class while leaving reach/clearance, compute, stencil,
GPU, and human-review gates pending; unrun episodes have no success label.

See [design format](design-format.md) for the exact contract and current gates.
Hugging Face upload is separately gated by the recorded `inkmap_deployment`
and `public_release` approvals. The deployment script enforces both before
reading credentials or building/uploading; CI skips upload while either is
pending. `tatbot inkmap deploy -- --check-gates` is a read-only gate check.

Prepare a portable candidate before requesting either approval:

```bash
tatbot inkmap release-prepare --output-dir /tmp/inkmap-release
tatbot inkmap release-audit /tmp/inkmap-release/release-manifest.json \
  --require-current-source
```

Preparation runs the full Inkmap typecheck/unit/build/development-browser/
production-browser suite, then copies the complete static bundle outside the
repository. Its manifest hashes every bundle file plus the source revision,
schemas, body/design assets, build inputs, runtime endpoint, and
disable-consumer rollback. The release-gates file is recorded as a snapshot
digest, not a bound input, because approval edits that file by design. It
remains `prepared_not_authorized`; it neither edits approvals nor deploys.
An approval binds the packet by writing the manifest's `content_sha256` into
its `evidence_sha256`; `--require-current-source` then accepts the gates file
only if it is byte-identical to the snapshot or differs by approvals that bind
exactly this packet. Deployment runs that same audit against the current
checkout before it reads credentials, so a packet whose bundle, schemas, or
body assets no longer match the source cannot be uploaded:
`tatbot inkmap deploy --artifact-dir /tmp/inkmap-release`.
### The editor

The editor is canvas-first: the body (or the paper/cylinder chart) fills the
stage, and one contextual panel beside it — a dock on desktop, a stacked tray
on touch screens narrower than 768 px; a desktop window dragged narrow keeps
its dock — shows only the step at hand:

- **Choose** — *Library* (six recent or curated thumbnails, **Browse all**,
  **Import artwork**) or *Generate* (native DBV3 acquisition instructions). Artwork
  made in this session comes first; a fresh project shows the acquired examples in
  manifest order, not an invented recency.
- **Place** — the chosen artwork and one instruction: click or tap the body,
  or describe a location. The ghost under the pointer is judged by the same
  atlas and geometry rules commit applies, and turns red where a spot is
  unsupported, so nothing looks placeable and is then refused.
- **Adjust** — width in mm, rotation in degrees, mirror, **Accept** and
  **Cancel** in one toolbar (under the body on desktop, in the tray on
  phones), beside dragging and the on-body ↻ ↔ handles. Reopening an accepted
  tattoo is the same state; Cancel restores its accepted values exactly and
  one Accept is one undo step.
- **Ready** — the placement list, **Add artwork**, **Export design**.

Either order works: artwork first and then a place, or a location sentence
first ("on the left forearm") and then artwork. An ambiguous sentence ("on the
thigh") lists concrete candidates and waits for a choice; nothing places until
one is taken. The stock body path is three primary actions — choose, place,
accept — with the body and the primary action on screen together at every
tested viewport (390×844, 522×900, 1440×900, and short heights).

**File** holds the project (name, local save status, **Open project…**,
**Open design…**) and every download, named by what it holds: **Portable
design (JSON)** — the accepted body placements, or the paper/cylinder draft;
**Artwork SVG** — the *selected* artwork as this editor draws it, a preview and
not a stencil, available before any placement; **Project backup (JSON)**; and,
under *Advanced*, the v6 **placement file**, loading one, the simulation bundle
and the compiled preview. A refusal is reported beside the action that produced
it, in the stage's one message region and in the menu. **View** holds the
camera presets, pose, skin tone, atlas, grid, neutral light and the quality
overlay; Undo/Redo and Fit/Focus stay in the bar. **New project** under File
replaces the current project with an empty one after a confirmation that
names what it holds; the library, pose and skin tone stay.

**Body / Paper / Cylinder** are the workspaces, and each is its own draft
in the project: the body's placements, the pad's, and the cylinder's never
mix. A tab shows one surface and parks the other; a pending edit is accepted
if it still fits, otherwise cancelled, before the switch. Paper and Cylinder
are shown in 3D as the bench's fixtures (`config/substrates.yaml`): the
paper pad, 7.5 × 11 in and 1 cm thick, and the paper cylinder, ⌀85 mm and
7.5 in long, both white with a faint blue ¼ in grid. Each starts at its
fixture's measured size; typed dimensions stay as typed. The cylinder's drawable area is its whole outer
surface except the bottom quarter it rests on and the end caps; an arc can
never close the full circumference.

Placing on paper is the same act as placing on the body: choose artwork, a
ghost follows the pointer over the paper (red where it would run off the
drawable area), click to place; then drag to move, the ↻ ↔ handles and A/D,
W/S rotate and resize, and the one toolbar carries width in mm, rotation,
mirror, **Accept** (Enter) and **Cancel** (Esc), which restores the accepted
placement exactly. Placing lands in Adjust, Accept returns to the placement
list, and Undo/Redo step through accepted edits on whichever editor is showing.
The chart keeps its own history and edit snapshot, so nothing on the body is
touched by any of it. Location sentences are the body's; the chart has no
anatomy to name.

Keyboard: Enter accepts (never from inside a field or on a focused button),
Escape closes a menu first, then an expanded tray, then cancels the edit;
A/D rotate and W/S resize; Ctrl/Cmd+Z undoes. Closing a menu returns focus to
its button.

The built-in library contains three small native DBV3 acquisitions: `dbv3-orbit`,
`dbv3-sprout` and `dbv3-ridges`, each generated at 30 × 30 mm with a 0.3 mm
black generation pen. `web/inkmap/public/designs/manifest.json` names only
acquired JSON records. A fresh project opens the library without placing
anything. The same records supply the simulator and five-pose showcase.

Each example includes its original CC0 raster source, version 3 job and frozen
recipe, effective pen table, native export and decoding evidence. The app loads
`artwork.json`, checks its identity, and derives the preview from the frozen
metric program. It never reconstructs paths from a preview SVG. Missing or
changed catalogue bytes produce an explicit error. Retired finished SVGs and
hidden IDs have been removed.

### Projects, editing, and recovery

The editor autosaves the current project in this browser's IndexedDB, including
artwork, placements, pose, skin tone, camera, pending edits, up to 50 undo
steps, and both surface drafts, the one showing and the one parked. The saved/saving/failed
indicator describes local storage, not a cloud backup. **Project backup**
downloads a portable `tatbot.inkmap-project/3` recovery copy; **Open project**
restores it. Storage failures leave the in-memory project available for
download and explicit retry, announced beside the work with both actions at
hand. Concurrent tabs use revision-checked transactions: a stale tab cannot
overwrite newer saved work. Download that tab's recovery copy before reloading
the saved version.

The project stores shared `tatbot.inkmap-artwork/2` records directly for body
placements and both chart drafts. Artwork includes the frozen metric program,
source provenance, recipe digest when available and decoding tolerance. Readers
validate its identity without reconstructing it from SVG. Earlier project and
placement versions require explicit migration. Reopening preserves the saved
canvas dimensions. Projects retain custom artwork and all undo dependencies;
unused catalogue entries load from the library rather than inflating the backup.

The private licensed DrawingBotV3 worker emits this shared record directly.
**Import artwork** accepts its JSON; the public app can view, place and export it
without a DBV3 license. Imports, catalogue loading, saved-project recovery and
compiled-preview loading require `dbv3-batik-paths/1` with a frozen recipe
identity. Legacy projects give an explicit Generate/Import action; they are never
silently retraced. **Import artwork** accepts JSON only. Source SVG and raster
images must first go through the native acquisition workflow. DBV3 exports,
derived preview SVGs and UI icons remain valid SVG uses.

Artwork dimensions and pen width are separate metric quantities. The editor's
millimetre dimensions export as `physical_scale_m` in metres: 50 × 75 mm becomes
`[0.05, 0.075]`. Resizing a recipe acquisition previews its paths with fixed pen
width and marks the placement as needing regeneration at the requested size.
Translation and rotation reuse the acquisition. Preview width is a generation
assumption; measured deposition width belongs to the physical tool profile.

The pinned reference body also uses metres (Z up); its rest-mesh height is about
1.738 m. Placement dimensions therefore have a real metric meaning on that
model. This is one nominal body, not a size estimate for the person being drawn
on. Mapping to a person's measured surface is a separate input boundary.

A DBV3 drawing was optimized for its acquisition dimensions and pen width.
Scaling its paths changes line spacing and visual density even when the pen
stays fixed. To adapt that density, rerun DBV3 at the chosen final dimensions;
a geometric resize alone does not regenerate its paths.

**Portable design** and **Open design** are the
[`tatbot.inkmap-design/1`](design-format.md) half, and Open design is one
control: which editor can show a file is a property of the file, so opening one
lands in the body or the chart workspace by what it actually places artwork on,
leaving the other draft as it was. A file neither editor can represent —
placements mixed across body and chart targets, or a surface warp — names the
unsupported feature, keeps the working draft, and offers the original bytes
back unchanged. Nothing is reconstructed as an approximation and handed back
under the same name. Opening a design and saving it again without touching it
returns the identity it arrived with, including one written by the headless
[`tatbot design place`](design.md): each placement keeps the review and
provenance it came with, and an edit changes only the placement it touched.

Rotation is stored in radians, as the placement contract has it; the degree box
is a display at the edge. (A degrees round trip is lossy — 30° becomes
29.999999999999996 — which would have moved a design nobody edited.)

Accept commits an edit. Cancel (or Escape) restores an existing tattoo's prior
state, or removes an unaccepted new tattoo. Delete remains a distinct action.
Undo/redo use the buttons or Ctrl/Cmd+Z and Ctrl/Cmd+Shift+Z; one accepted edit
is one history step. Width and rotation accept numeric values; drawing order is
editable. The **placement file** export refuses pending edits and contains only
accepted placements. Project recovery files may intentionally contain pending
edits.

Selecting a tattoo frames its surface site. Drag the tattoo itself to move its
canonical anchor; the yellow on-body ruler and adjacent handles resize or
rotate it. Camera orbiting is disabled for the duration of those gestures so a
single drag has one meaning. The same size/rotation operations remain available
through millimeter/degree inputs and W/S/A/D keys. Reset, front, back, patient
left, patient right, and selection-focus camera actions use the documented
`front=-y`, `left=+x`, `up=+z` body frame.

The optional quality panel reports the intrinsic chart's mapped area, posed
area stretch, duplicate-edge disagreement, covered face count, and atlas site.
It is diagnostic only: the existing chart-area, seam, edge, and supported-site
refusals still decide whether an edit can be accepted. The yellow ruler is the
authored physical width, not a pixel scale or a claim about reachable tool
motion.

On phones and tablets up to 768 px the panel is a tray stacked under the stage
rather than a drawer over it, so the two never overlap: its working height
follows its content up to about 45% of the viewport, it expands to most of the
screen for browsing a full library, and it collapses to a handle to give the
body the room. Each new step reopens it to its working size, so Accept is never
behind a collapsed handle. The body stands alone in the viewport: no floor
grid and no bed, chair or armrest props. The simulator's nominal scenario
families still carry their named supports; the editor only names the pose's
support in the bundle it exports. Neutral inspection lighting is a view toggle.

### Procedural simulation showcase

The guided milestone view uses the production Three.js renderer and checked-in
scenario fixtures; it is not a painted mockup. It switches among all five
tattoo-session poses on the fixed MHR/SOMA body, names each pose's intended
support, projects each scenario's real SVG placement, and draws the
compiled face/barycentric surface trace in cyan:

```bash
tatbot inkmap dev
# open http://127.0.0.1:4180/?showcase=1
```

The evidence panel records the deterministic CPU sampling and reach-audit run
behind the five examples. Its last pipeline stage remains visibly pending:
the showcase does not claim a GPU-rendered ManiSkill episode, deformable skin,
MediaPipe tracking, powered-arm behavior, or safe human contact.

## The design generator

Source images can come from the separate text-to-image generator over HTTP
(`web/inkgen/`). `tatbot inkgen serve` runs one in the foreground on the node
that carries the `inkgen` role (the CLI hops there); `tatbot inkgen ctl --
start|stop|status|logs` manages a background one on that node; `tatbot inkgen
status` is the health probe from any node (`--space` for the hosted one); and
`tatbot inkgen deploy` publishes `web/inkgen` to its Space. The editor does not
trace or place responses from that service.

Inkgen is also a small product of its own: its page draws tattoo artwork from
a subject, with a random seed by default (zero is a seed like any other, and
the seed that was used is shown), keeps the last image while another draws,
and offers **New variation** (the same subject, a seed nobody chose),
**Download PNG** and the generation metadata as a file. The same seed
reproduces the same image on that generator; a different GPU or runtime is not
promised identical pixels.

### Generation in the editor

**Generate** explains the native [DrawingBot V3 acquisition](drawingbot.md):
run a version 3 job against an installed, activated DBV3 application, then import
its `artwork.json`. There is no browser tracer or image-service fallback.
Acquisition runs outside the browser. The bundled examples and previously
acquired projects remain usable offline.

### Serving policy

Both public entries — the HTTP API and the page — pass through one admission
policy and one inference gate (`web/inkgen/serving.py`, stdlib, model-free):

- **Admission** is the visitor quota: per address per minute
  (`INKGEN_PER_IP_PER_MIN`, 6 on a Space), per address per day
  (`INKGEN_PER_IP_PER_DAY`, 60) and a daily GPU-seconds budget for the whole
  service (`INKGEN_DAILY_BUDGET_S`, 1800). Off the Hub all three default to
  off, because a private worker a batch was pointed at owns its own job
  limits; a stated number wins anywhere. A refusal answers 429/503 with
  `retry_after_s` and a `Retry-After` header.
- **The gate** runs one render at a time per engine with a bounded line
  (`INKGEN_MAX_WAITING`, 3; `INKGEN_QUEUE_WAIT_S`, 120): a caller past the line
  is refused *busy* (503) at once rather than parked. The line is joined
  before the quota is charged, so a quota refusal never touches the GPU and a
  busy refusal is never charged.
- The HTTP handler hands the render to a worker thread with its context
  copied, so `/api/health` keeps answering while the engine draws (it reports
  `inference_in_flight`, `inference_waiting` and the idle hold) and the
  ZeroGPU wrapper still finds the Gradio request it schedules against.
- A request naming a **model or revision these weights are not** is refused
  (400) before the line: the reply's `model`/`model_revision` are what drew
  the image, never a caller's wish. Switching models is a restart with
  `INKGEN_MODEL`, not a per-request surprise.

`INKGEN_FAKE_ENGINE=1` serves all of this without weights (`fake_engine.py`,
a deterministic PNG per seed): the adapter tests in `web/inkgen/tests/` drive
the real routes and page callbacks with it.

A fleet generator is meant to come and go. It stops itself after 15 minutes
with no generation (`ctl -- start --idle-minutes N`, `0` keeps it up;
`INKGEN_IDLE_STOP_S` is the same number in seconds), and `/api/health` reports
`last_request_unix` and `idle_stop_in_s` so `tatbot status` can show the
countdown. Health probes read that countdown, they never extend it, and a
generation still running when the timer expires finishes first. On the Hub the
Space owns the lifecycle and the timer is off. Anything that needs a generator
starts one through the same idempotent `ctl -- start`, so a stale pidfile after
an idle stop is normal rather than a crash. `tatbot design generate` and
`tatbot sim materialize` do exactly that (`--no-autostart` refuses instead),
and both refuse before starting when the GPU has less than
`INKGEN_VRAM_MIN_MB` (14000) free, naming what holds it — the generator's node
also carries the `sim` role. On a node with the role, `tatbot status` shows
whether one is running and how long until it stops itself.

Which generator answers is an explicit choice, never a search. There are four
kinds — the managed **fleet** worker (the only one anything here starts), an
**endpoint** a caller named with `--api-url` (probed, never started, never
woken), the public **Space** (`--space`, or the interactive default when no
node carries the role), and none at all. Bulk work does not accept that last
fallback: `tatbot inkgen batch` refuses (exit 5) rather than pointing a
thousand-image job at a shared public GPU.

### Deploying the generator

`tatbot inkgen deploy` prepares, uploads and *verifies* the Space:

```bash
scripts/inkgen_deploy.sh --prepare-dir ~/tatbot-release/inkgen-rc   # no upload
tatbot inkgen deploy -- --artifact-dir ~/tatbot-release/inkgen-rc
```

Preparation copies exactly the payload the Space runs — `app.py`, `contracts.py`,
`engine.py`, `serving.py`, `fake_engine.py`, `batch.py`, `idle.py`,
`requirements.txt`, `README.md` — stamps it
with `build.json` naming the source commit, imports the model-free half from the
copy with this repository off `sys.path` entirely, hashes everything into a
`tatbot.inkgen-release-candidate/1` manifest, and stops. A payload that only
imports inside the checkout is not a Space, and this is where that is decided.

The upload is no longer piped through anything: a failed `hf upload` fails the
deploy. It used to end in `| grep -v '^Hint' || true`, which reports grep's
status and then discards it, so a failed upload printed a successful deploy and
the wait loop "verified" the revision that was already live. `/api/health` now
reports the payload's build stamp and the script waits for *that* stamp — a
healthy previous revision is a failure, not a pass.

### Batch generation

`tatbot inkgen batch` runs — or resumes — one artwork generation job:

```bash
tatbot inkgen batch --output-dir ~/tatbot-artwork/flash-v1 \
  --subjects-file subjects.txt --count 24 --seed 7 --replacement-budget 8
tatbot inkgen batch-status --output-dir ~/tatbot-artwork/flash-v1
```

Interrupting it is safe: rerun the same command. A request's identity comes from
its content — `<subject-slug>-<nth occurrence>`, and a seed derived from that
key — so inserting, reordering or sharding a job leaves every existing request
where it was. Cache keys bind the resolved model revision and every generation
setting, so the same words on different weights are different work. The raster
is persisted before anything is traced, so a tracer failure never costs a second
GPU request. Resume verifies retained bytes against the ledger and regenerates
what does not match as a new counted attempt. One worker holds the job; a second
gets a busy refusal.

`manifest.json` appears only when every requested slot ended accepted, so
these are source-image results, not Inkmap or simulator artwork. Native DBV3
acquisition is required before placement. A job that falls short
writes `selection.json` instead, with requested/accepted/refused/failed/
duplicate on its face, and exits 1. Identical output is deduplicated: both
requests are kept, one artwork is published. `--replacement-budget` lets a
refused or duplicate slot spend a fresh candidate on the same slot — a retry
budget, not extra dataset items.

The job module is also a standalone program with no Tatbot checkout, node map or
fleet configuration involved, which is what the Space payload needs:

```bash
python web/inkgen/batch.py run ./job --subject "a swallow" --count 4 --api http://127.0.0.1:8600
```

Source generation and placement of acquired DBV3 artwork are reachable without
a browser as [`tatbot design`](design.md). The former `design trace` command
refuses with a native acquisition action.
Drawing it on paper is `tatbot ros compile` and `tatbot ros draw` on the ROS 2
stack (`ros/README.md`, section 4).

## Placement record

The sole public schema is PlacementFile v6 at
`config/inkmap/placement.schema.json`. A record binds the fixed model spec,
identity, topology, rest surface, and browser asset to the design, exact
anchor, physical scale, rotation, mirror state, original placement request,
normalized intent, and resolution provenance. Older placement versions have
no reader or migration path. See [design format](design-format.md).

Preview geometry is a design aid, not a claim that a physical tool can safely
reach the same surface.

## Named body poses

The only checked-in body is generated by the pinned MHR-through-SOMA provider
on SOMA's indexed mid topology. Standing neutral is the editor default and the
rest-surface reference. The tattoo-session pose set is supine on a bed, prone on a bed,
reclined with the legs on the chair rest, and reclined with either arm lowered
onto an armrest. The
pose control bakes the pose into the geometry used for display, raycasts, and
decals. The editor's pose picker, in the body panel and under View, offers
standing and reclined seated; the other session poses stay in the catalog for
the showcase and the simulator, and a project that already holds one keeps it.
Skin tone sits beside the pose picker. No support prop is drawn. InkLang resolution and region
charts remain tied to the canonical rest surface. Placement anchors name that
unchanged rest-surface face and barycentric coordinates; the preview applies
the same anchor to the posed geometry.

Named joint/support constraints live in
`config/body-models/mhr-soma-v1/poses.json`. The deterministic reference GLB,
expanded posed-face cache, upstream not-skin exclusion mask, and
`config/inkmap/body-poses.json` catalog are checked in so normal browser
development does not need licensed source assets. Regeneration is an explicit
offline build in the locked SOMA environment and exact reviewed cache:

```bash
python web/inkmap/tools/export-soma.py \
  --cache-dir /path/to/reviewed/body-cache
```

Generation fails before model construction on an absent/unlisted/mismatched
asset or software lock, and fails if topology, units, axes, identity, face
order, pose names, or expected surface digests drift. The browser renders
chart-clipped triangles from an intrinsic rest-surface unfold and replays the
same face/barycentric addresses on posed vertices; a chart overlap or
insufficient girth refuses instead of projecting through the far side.
`npm --prefix web/inkmap run check`
independently checks all named poses against the generated cache. These
numerical gates complement, rather than replace, browser review of every pose,
site, support, and retained showcase.

## Acquired artwork study review

**File → Review study…** opens a portable review JSON generated by
`tatbot research review`. It uses Inkmap's shared metric artwork renderer,
shows preparation costs and physical-width assumptions, and records Like,
Dislike or Clear observations against the exact artwork, prepared program and
rendered SVG. Review does not replace the current project or placement draft.
**Add to library** makes a reviewed artwork available to the ordinary editor.

Preferences persist in this browser and merge across tabs. Download feedback
for a durable copy and import it with `tatbot research feedback`; a save failure
is shown explicitly with retry and recovery download. The browser never invokes
the private licensed worker. See [the research workflow](research.md).
