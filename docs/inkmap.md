---
summary: Inkmap places tattoo source art on an MHR/SOMA body and exports a brief for the art pipeline
tags: [web, design, preview]
updated: 2026-11-02
audience: [artist, dev, contributor]
---

# Inkmap

`web/inkmap/` is a browser application for choosing tattoo art and placing it
on a 3D body. It works with the **source image**, not robot strokes: a design
is a PNG or JPEG, placed and sized freely, and the finished placement exports
as a **tattoo brief** that names each design's art and the width to draw it at.
The art pipeline separates the image into the robot's inks at that width, and
DrawingBot V3 makes the strokes there (`tatbot art separate --brief`,
[art pipeline](art.md)). Inkmap does not command a robot, and the public
Space carries no DBV3 artwork. What comes back is a picture: the pipeline's
[robot render](#the-robot-render), to see on the body before anyone approves.

Inkmap uses [InkLang](inklang.md) to turn placement descriptions into
canonical rest-surface anchors. Inkmap is the editor and visualizer; it does
not define a second placement grammar or independently choose semantic
regions.

Type a placement-only phrase such as `on the left forearm`. Inkmap shows the
normalized phrase and resolves it before a design is committed. If the phrase
omits a required side, names a multi-site zone, or leaves a relative direction
open, Inkmap shows concrete candidates and waits for a choice. A manual click
is the inverse input mode: the atlas describes and validates the chosen anchor
through the same contract.

## Develop

```bash
cd web/inkmap
npm ci
npm run check
npm run dev
```

The app should work with local fixtures and no private service. Keep generated
images and personal designs outside Git unless their license permits sharing.
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

The editor is canvas-first: the body fills the stage, and one contextual panel beside it — a dock on desktop, a stacked tray
on touch screens narrower than 768 px; a desktop window dragged narrow keeps
its dock — shows only the step at hand:

- **Choose** — the library (six recent or curated thumbnails, **Browse all**,
  **Import art**). Art imported in this session comes first; a fresh project
  shows the stock collection in manifest order, not an invented recency.
- **Place** — the chosen art and one instruction: click or tap the body,
  or describe a location. The ghost under the pointer is judged by the same
  atlas and geometry rules commit applies, and turns red where a spot is
  unsupported, so nothing looks placeable and is then refused.
- **Adjust** — width in mm, rotation in degrees, mirror, **Accept** and
  **Cancel** in one toolbar (under the body on desktop, in the tray on
  phones), beside dragging and the on-body ↻ ↔ handles. Reopening an accepted
  tattoo is the same state; Cancel restores its accepted values exactly and
  one Accept is one undo step. **Draw earlier** and **Draw later** set the
  order, and **Download source image** returns the art's own bytes.
- **Ready** — the placement list, **Add artwork**, **Export brief**.

**Adjust** also says what the chosen size means downstream ([below](#fit-and-cost)).

Either order works: art first and then a place, or a location sentence
first ("on the left forearm") and then art. An ambiguous sentence ("on the
thigh") lists concrete candidates and waits for a choice; nothing places until
one is taken. The stock body path is three primary actions — choose, place,
accept — with the body and the primary action on screen together at every
tested viewport (390×844, 522×900, 1440×900, and short heights).

**File** holds the project (name, local save status, **Open project…**) and
every download, named by what it holds: **Tattoo brief (JSON)**, each placed
design's art and size for the art pipeline; **Source image**, the selected
design's original file; **Placement card (HTML)**; **Mockup image (JPEG)**, the
current view of the body; **Project backup (JSON)**; and, under *Advanced*, the
v7 **placement file** and loading one. A refusal is reported beside the action
that produced it, in the stage's one message region and in the menu. **View**
holds the camera presets, pose, skin tone, atlas, grid, neutral light and the
quality overlay; Undo/Redo and Fit/Focus stay in the bar. **New project**
under File replaces the current project with an empty one after a confirmation
that names what it holds; the library, pose and skin tone stay.

Inkmap edits body placements only. The paper pad and cylinder editors were
removed (2026-11-02), and so were stroke artwork, layer sets, Generate and the
study review: Inkmap holds no strokes.

Keyboard: Enter accepts (never from inside a field or on a focused button),
Escape closes a menu first, then an expanded tray, then cancels the edit;
A/D rotate and W/S resize; Ctrl/Cmd+Z undoes. Closing a menu returns focus to
its button.

### Source art

A design is a `tatbot.inkmap-art/1` record ([design format](design-format.md#source-art)):
the image bytes, their digest and pixel size, `art_px` (the rectangle the art
occupies), where it came from and on what licence, and optionally the inks and
style it was drawn for. `art_px` follows the art pipeline's own rule: the box
around every pixel that is not paper (light and neutral, CIELAB L\* above 85 and
chroma under 12), padded by 3% of the longer side. Inkmap draws that crop with
the paper made transparent, so the skin shows through where the separation
leaves no ink, and the placed width is the crop's width.

Inkmap and the art pipeline read an image alike, or the art a person approves is
not the art drawn: transparency lies over white paper, and a photo stands as its
EXIF says. A browser turns a JPEG as it decodes it, so import keeps a turned
photo's pixels upright, as a new JPEG.

The stock collection (`flash-art-v1`, `web/inkmap/public/designs/manifest.json`)
is three designs kept from the art wall, generated with Google Flow:
**Swallow and poppies**, **Sun and moon** and **Snake and dagger**. Each entry
names its JPEG, digest, size, `art_px` (computed by the art pipeline) and the
inks its separation used. They are not dedicated to the public domain
([LICENSE.txt](../web/inkmap/public/designs/LICENSE.txt)). The same designs
supply the five-pose showcase. The former DBV3 defaults, `dbv3-orbit`,
`dbv3-sprout` and `dbv3-ridges`, are simulator fixtures in
`config/sim2/acquired/`.

**Import art** takes PNG and JPEG files, several at once. A JSON file with the
same name stem beside an image is read as its `tatbot.art-generation/1` sidecar
(the art corpus writes one for every generated image): its prompt, model and
generator become the record's source, and its subject becomes the name. An
image without one is recorded as imported, with no licence. Either way its inks
are suggested from its colours ([below](#fit-and-cost)). Import says when the box
misleads: a small mark away from the design stretches it, so the design is less
than the width set, and a pale background that did not read as paper would be
drawn as ink. These are notices, as the pipeline's rule is what sizes the art.
SVG and DBV3
`artwork.json` are refused by name. A record over 8 MB is refused.

Size is free up to 300 mm on the longer side. A placement's width is the art's
drawn width in millimetres, kept to 0.1 mm (the pipeline separates at 10 px/mm),
and its height follows the art's aspect. A design starts 60 mm wide, or 120 mm
tall if that is smaller, so a strip starts on the limb. Resizing changes nothing but that number: the strokes are
made later, at the chosen size, so line spacing and pen width stay right at
any size. Rotation and mirror are part of the placement.

The pinned reference body uses metres (Z up); its rest-mesh height is about
1.738 m, so a placement's millimetres have a real metric meaning on that model.
It is one nominal body, not a size estimate for the person being drawn on.
Mapping to a person's measured surface is a separate input boundary.

### Fit and cost

Everything here is advice. Nothing in it stops a size being accepted or exported: the compile is what refuses a
design its page cannot hold.

- **Stencil tile.** Each fiducial tile ([the skin stencil](stencil-frames.md)) says whether the design fits it, and
  by how much it does not. Their sizes are `config/stencil-tiles.json`, which the tile generator reads too. The
  compile centres a design on its page and never turns it, so a design fits by its own width and height. The chosen
  tile is drawn on the body round the design, and **Fit to …** sets the widest size that fits.
- **Time and dips.** An estimate from the art's inked area and its inks, before any strokes exist: 0.072 min per
  inked mm² and 2.6 min per ink, 0.04 dips per inked mm² and one per ink. Fitted to 52 corpus separations, its median
  error is 10% and one in ten is over a third out. A [robot render](#the-robot-render) replaces it with the real figure.
- **Wraps.** How far the skin turns across the design along its more curved axis, on the rest surface.
- **Detail.** A colour may be lost when its fattest spot is under 1 mm in radius at the chosen width: the pipeline
  opens thin colour and then fills inside a half line's margin. The darkest ink is not judged, since its lines are
  widened to one drawn line. Over the corpus's 124 colour-ink cases this caught 7 of the 8 inks that drew nothing and
  flagged 4 that drew, so it is a warning.
- **Overlaps.** Other designs whose art rectangles share this one's skin, here and in the placement list. Each
  design is its own separation and stencil, so two on the same skin are two sessions drawn over each other.
- **Inks.** What the art pipeline will load: the art record's `palette`. Imported art starts with the catalogue inks
  nearest its colours, at most five; change them here. The list is `config/inkmap/inks.json`: every catalogue ink's id,
  name and colour, which a test holds to `config/inks.yaml`. It is what could be loaded, not what is. The lining blacks
  are offered but never suggested, or one black design would suggest three blacks.

### The robot render

Source art is what was asked for. A **robot render** is what the robot is predicted to draw: the heavier lines, the
muted colour and whatever a size lost. The strokes do not exist until DBV3 makes them, so it takes a round trip:

```bash
tatbot art separate --brief inkmap-brief.json --out sep --app /opt/drawingbotv3
```

That writes `sep/inkmap-render.json` (a numbered folder each for several placements). **Load render**, in the Adjust
panel or under File, shows it on the body in place of the art, and the panel gives the pipeline's minutes, dips and
lost inks in place of the estimate, whether or not the render is showing. The placement card prints it too.

- It is a picture, never strokes: the program's predicted ink over white, at the design's canvas.
- Ink filters what lies under it, so the picture multiplies the lit skin. One render serves any skin tone, and what
  an ink does on deep skin shows. White and silver are opaque, and cannot show this way.
- A render holds only for its art, width and mirror, since strokes are made at one size. At another width the art
  returns and the panel says which width it was made for. New inks are new art, and need a new one.
- It is kept in this browser beside the project, not in it: it is made from the project, so a backup leaves it out.

It is a prediction, with [the render's limits](art.md#predicted-render-and-minutes).

The stock designs ship their own, so a visitor with no pipeline can see one: at the default width and at the widest
the forearm tile holds, fetched once the design is placed. They come with the page, so nothing saves them. A new
visitor meets the source art: **Show it on the body** turns a render on, loading one does too, and this browser
remembers the choice. `tatbot art
stock-renders --app /opt/drawingbotv3 --out DIR` makes them again, which the separation changing calls for: nothing
detects that. A test does fail on one whose art or inks have moved on.

### The placement card

The brief's strokes are made flat, and the robot draws on a page: nothing reads a placement's anchor or rotation to
move the arm. They reach the skin through whoever lays the stencil, and the **Placement card** is what they read. It
is one self-contained HTML page per placement, to print: the placement on the body, its size, tile, inks and
estimate, how far it lies from the two nearest landmarks, how the tile is turned, and the design and tile at actual
size with a 50 mm bar to check the print against. A design past the page is shown scaled, and says so.

The distances lie along the skin, straight in the rest surface unfolded about the design, to landmarks a tape can
start from, such as the tip of the elbow and the top of the shoulder. The turn is measured from the design's real
direction on the surface to the top of the shoulder on an arm, the hip on a leg and the head elsewhere. Both are on
one nominal body: a starting point to measure against the person. The card takes the selected stencil tile, and
refuses pending edits as the brief does.

### The tattoo brief

**Export brief** writes `tatbot.inkmap-brief/1`
([design format](design-format.md#tattoo-brief)): every accepted placement in
drawing order, each with its art id, `width_mm`, height, rotation, mirror,
anchor, site and sentence, and the art records it uses, byte for byte. It
refuses pending edits. `tatbot art separate --brief brief.json --out DIR`
separates one placement's image at its width, with the inks and style the art
records when the command names none.

### Projects, editing, and recovery

The editor autosaves the current project in this browser's IndexedDB, including
art, placements, pose, skin tone, camera, pending edits and up to 50 undo
steps. The saved/saving/failed indicator describes local storage, not a cloud
backup. **Project backup** downloads a portable `tatbot.inkmap-project/5`
recovery copy; **Open project** restores it. Storage failures leave the
in-memory project available for download and explicit retry, announced beside
the work with both actions at hand. Concurrent tabs use revision-checked
transactions: a stale tab cannot overwrite newer saved work. Download that
tab's recovery copy before reloading the saved version.

A project keeps the art records of its placements and every imported design;
stock designs it does not use load from the collection rather than inflating
the backup. A project of another version has no migration: one saved in the
browser gives way to a new project, and **Open project** refuses such a file.

Rotation is stored in radians, as the placement contract has it; the degree box
is a display at the edge. (A degrees round trip is lossy — 30° becomes
29.999999999999996 — which would have moved a design nobody edited.)

The stored rotation is measured from the body's up on the rest surface, which on
an arm is nearly square to the skin and of no use to a person. So the box reads
from upright: 0 stands the art straight on a hanging arm. A new design starts
upright, and one dragged keeps its turn, the store adding the difference
between the two up directions. Nothing stored changes meaning: a saved placement
draws as it always did, and the box simply shows how it is turned.

Saved words can outlive the lexicon or resolver that wrote them. A placement's
anchor and rotation are its record, so loading one from an older version keeps
them and makes its site and sentence again from the current atlas.

Accept commits an edit. Cancel (or Escape) restores an existing tattoo's prior
state, or removes an unaccepted new tattoo. Delete remains a distinct action.
Undo/redo use the buttons or Ctrl/Cmd+Z and Ctrl/Cmd+Shift+Z; one accepted edit
is one history step. Width and rotation accept numeric values; drawing order is
editable. The brief and the **placement file** refuse pending edits and contain
only accepted placements. Project recovery files may intentionally contain
pending edits.

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
grid and no bed, chair or armrest props. Neutral inspection lighting is a view
toggle.

### Pose showcase

The guided view uses the production Three.js renderer and the tracked
`public/showcase/manifest.json`; it is not a painted mockup. It switches among
all five tattoo-session poses on the fixed MHR/SOMA body, names each pose's
intended support, and places the slide's stock design at its face/barycentric
anchor with the editor's own decals:

```bash
tatbot inkmap dev
# open http://127.0.0.1:4180/?showcase=1
```

Each slide names the exact image bytes it shows, and loading refuses a
collection that differs. Its last pipeline stage remains visibly pending: the
showcase claims no reach, deformable skin, tracking, or powered-arm behavior.

## The design generator

Source images can come from the separate text-to-image generator over HTTP
(`web/inkgen/`). `tatbot inkgen serve` runs one in the foreground on the node
that carries the `inkgen` role (the CLI hops there); `tatbot inkgen ctl --
start|stop|status|logs` manages a background one on that node; `tatbot inkgen
status` is the health probe from any node (`--space` for the hosted one); and
`tatbot inkgen deploy` publishes `web/inkgen` to its Space. A downloaded
image is art Inkmap can import; the editor does not call the service itself.

Inkgen is also a small product of its own: its page draws tattoo artwork from
a subject, with a random seed by default (zero is a seed like any other, and
the seed that was used is shown), keeps the last image while another draws,
and offers **New variation** (the same subject, a seed nobody chose),
**Download PNG** and the generation metadata as a file. The same seed
reproduces the same image on that generator; a different GPU or runtime is not
promised identical pixels.

### Optional image providers

Inkgen's source-image page and `/api/generate` also support fal, Google Gemini
and OpenAI. Inkgen remains the no-key default. A provider and model picker
exposes a masked key field only for cloud providers; it clears on provider
change, **Clear key**, refresh, and every 15 minutes. The key passes through
the generator backend to that provider. It is not saved in projects, downloads,
logs, browser storage, or shared environment variables; Gradio run history is
disabled. Hosted requests always require the visitor's key and never fall back
to deployment credentials or another provider.

For a local installation, copy the public `.env.example` to a private,
git-ignored `.env` and fill only `FAL_KEY`, `GEMINI_API_KEY` or `OPENAI_API_KEY`
as needed. The backend loads literal assignments; process environment wins.
`INKGEN_PROVIDER=inkgen` remains the default. An explicit local cloud default
uses API-only dependencies and skips loading the local model. Configuration
belongs on the execution owner. Server-backed Space deployments use Hugging Face
Secrets; static frontend configuration must not contain credentials.

HTTP clients choose `provider` and send their key in `X-Provider-Key`. Local
Inkgen keeps its existing request/result and reproducible batch contracts.
Cloud results use `tatbot.inkgen-result/2`, with provider/model, actual dimensions,
request parameters, available numeric usage, hashes and exact original output
alongside the PNG preview. Seeds and immutable revisions can be unavailable.
There is no automatic retry or paid-provider fallback. See
`web/inkgen/README.md` for models, key handling and
response details. This is text-to-image generation; acquisition and placement
keep their existing boundaries.

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
Space owns the lifecycle and the timer is off. Starting one goes through the
same idempotent `ctl -- start`, so a stale pidfile after an idle stop is normal
rather than a crash. On a node with the role, `tatbot status` shows whether
one is running and how long until it stops itself.

Which generator answers is an explicit choice, never a search. There are four
kinds — the managed **fleet** worker (the only one anything here starts), an
**endpoint** a caller named (`tatbot inkgen status --url`, batch's `--api`;
probed, never started, never woken), the public **Space** (`--space`, or the interactive default when no
node carries the role), and none at all.

### Deploying the generator

`tatbot inkgen deploy` prepares, uploads and *verifies* the Space:

```bash
scripts/inkgen_deploy.sh --prepare-dir ~/tatbot-release/inkgen-rc   # no upload
tatbot inkgen deploy -- --artifact-dir ~/tatbot-release/inkgen-rc
```

Preparation copies exactly the payload the Space runs — `app.py`, `contracts.py`,
`configuration.py`, `provider_http.py`, `providers.py`, `engine.py`,
`serving.py`, `fake_engine.py`, `batch.py`, `idle.py`,
`requirements.txt`, `README.md` — stamps it
with `build.json` naming the source commit, imports the model-free half from the
copy with this repository off `sys.path` entirely, hashes everything into a
`tatbot.inkgen-release-candidate/1` manifest, and stops. A payload that only
imports inside the checkout is not a Space, and this is where that is decided.

A failed `hf upload` fails the deploy: nothing pipes its exit status away.
`/api/health` reports the payload's build stamp and the script waits for
*that* stamp — a healthy previous revision is a failure, not a pass.

### Batch generation

`web/inkgen/batch.py` runs — or resumes — one image generation job. Interrupting
it is safe: rerun the same command. A request's identity comes from its
content — `<subject-slug>-<nth occurrence>`, and a seed derived from that key —
so inserting, reordering or sharding a job leaves every existing request where
it was. Cache keys bind the resolved model revision and every generation
setting. Resume verifies retained bytes against the ledger and regenerates
what does not match as a new counted attempt. One worker holds the job; a
second gets a busy refusal. The results are source images: Inkmap places
them, and the art pipeline separates them for the robot.

It is a standalone program with no Tatbot checkout, node map or fleet
configuration involved, which is what the Space payload needs:

```bash
python web/inkgen/batch.py run ./job --subject "a swallow" --count 4 --api http://127.0.0.1:8600
```

Acquired DBV3 artwork (`tatbot art separate`, [DrawingBot V3](drawingbot.md))
is drawn without a browser: `tatbot ros compile artwork.json` and
`tatbot ros draw` on the ROS 2 stack (`ros/README.md`, section 4).

## Placement record

The placement schema is PlacementFile v7 at
`config/inkmap/placement.schema.json`. A record binds the fixed model spec,
identity, topology, rest surface, and browser asset to the source art, exact
anchor, physical size, rotation, mirror state, original placement request,
normalized intent, and resolution provenance. Older placement versions, which
carried stroke artwork, have no reader or migration path. See
[design format](design-format.md).

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
standing, reclined seated and reclined with either arm supported, the session pose
for an arm tattoo; the other session poses stay in the catalog for
the showcase and offline consumers, and a project that already holds one keeps it.
Skin tone sits beside the pose picker. No support prop is drawn. InkLang resolution and region
charts remain tied to the canonical rest surface. Placement anchors name that
unchanged rest-surface face and barycentric coordinates; the preview applies
the same anchor to the posed geometry.

Named joint/support constraints live in
`config/body-models/mhr-soma-v1/poses.json`. The deterministic reference GLB,
expanded posed-face cache, upstream not-skin exclusion mask, and
`config/inkmap/body-poses.json` catalog are checked in so normal browser
development does not need licensed source assets. There is no regeneration
tool in the tree: the bake (`web/inkmap/tools/export-soma.py`) ran on the
deleted ManiSkill simulator's body model and is in git history, and
`scripts/check body-single-path` still refuses an authoring-sheet edit the
catalog does not carry. The browser renders
chart-clipped triangles from an intrinsic rest-surface unfold and replays the
same face/barycentric addresses on posed vertices; a chart overlap or
insufficient girth refuses instead of projecting through the far side.
`npm --prefix web/inkmap run check`
independently checks all named poses against the generated cache. These
numerical gates complement, rather than replace, browser review of every pose,
site, support, and retained showcase.
