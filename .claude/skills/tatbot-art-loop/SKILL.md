---
name: tatbot-art-loop
description: Run a round of tatbot's art voting loop. Read the owner's votes, stars and asks from the Flash Wall, promote kept designs up the pyramid (robot paths, fill studies), propose and generate the next round in Google Flow, separate it into the robot's inks, put it back on the wall and refresh its map. Use when asked to iterate on the art, run the next round, read the votes or asks, or add designs to the voting wall.
---

# The art voting loop

The owner votes Keep or Drop, shortlists designs for the rig, and asks for more of what they like on a
private claude.ai page, the Flash Wall. Each round, the
agent reads the votes, works out which choices are winning, generates the next round from them, and
puts it back on the wall. The page's source is `python/tatbot_art/gallery.html`. Its URL, and the ids
of every uploaded image, are in the corpus at `~/tatbot-art-corpus/wall/assets.json`. The corpus
lives outside the repository and is never committed.

## The wall's database

- `designs/<id>`: one document per design.
  - `round`, `order`, `style`, `subject`, `prompt`, `variant`, `fill`, `separation`
  - `inks`: `[{id, name, hex}]`
  - `views`: `{flow, ours, drawn}`, each an asset URL `/_blob/<id>`
  - `genes`: `{style, subject, palette, variant, fill}`
  - `parents`: the kept designs it was built on. A fill-study variant also carries `study_of`.
  - from separation: `minutes`, `dips`, `colours`, `mixes`, `fidelity`
- `rounds/<n>`: `{round, title, note, created}`, shown above the grid.
- `votes/<viewer id>`: `{v: {<design id>: 1 | -1}, s: {<design id>: 1}, at}`. `s` is the stars.
  Each viewer writes only their own document.
- `asks/<id>`: `{kind, anchors, n, steer, label?, at, status}`, where `kind` is near, branch, blend or
  cluster. The page writes them with status `open`; the agent sets `done` once a round answers them.
- `map/current`: the output of `tatbot art map`.

Only the owner writes `designs`, `rounds`, `asks` and `map`; the agent writes them as the owner, with
the `ArtifactData` tool.

## One round

1. **Read the wall.**
   - `ArtifactData list votes`, saved raw as `votes.json`: `{viewer: {v, s, at}}`.
   - `ArtifactData list asks`, saved as `asks.json`: `{id: document}`.
   - `ArtifactData list designs` (with `out_dir`), merged into `designs.json` as `{id: document}`. Keep
     the listed versions: every update needs its `if_version`.
2. **Promote.** Spend DBV3 time where the votes earned it, before generating anything new.
   ```bash
   tatbot art promote --designs designs.json --votes votes.json --corpus ~/tatbot-art-corpus --out promote.json --app /opt/drawingbotv3
   ```
   - The pass draws kept sketches, then studies kept drawings in every other contained fill
     (`loop.FILLS`). `--budget` (default 40) counts separations, about 1–2 minutes each, shared across
     styles.
   - DBV3 runs one job at a time. Run the pass in the background, after any other separation batch.
   - **Bundle the stars.** A ★ is the shortlist. Turn it into what the rig takes; this runs no DBV3:
     ```bash
     tatbot art bundle --votes votes.json --designs designs.json --corpus ~/tatbot-art-corpus --out ~/tatbot-art-corpus/bundles
     ```
     A starred sketch with no robot paths is listed as waiting. Draw it in this pass, then bundle again.
     Report each bundle's minutes and inks to the owner. Drawing it stays the owner's call. To try a
     design on a body, **Import art** with its Flow image (and the `.json` sidecar beside it) in Inkmap;
     the brief Inkmap exports goes to `tatbot art separate --brief` at the width chosen there.
3. **Propose.**
   ```bash
   tatbot art propose --designs designs.json --votes votes.json --asks asks.json --round N --out ~/tatbot-art-corpus/flow/prompts-rN.json
   ```
   - Open asks take the first prompts. The plan lists the ones it answered in `asks_used`.
   - A gene's score is its designs' summed votes over (count + 2). The genes are style, subject, palette
     and fill.
   - `--explore` (default 0.34) is the share of the remaining prompts on untried genes, with random
     palettes from every owned ink. When the owner asks for wide exploration, use 0.7 and 15 prompts.
     Pick the `--seed` whose plan covers the most styles and fills.
   - No style takes more than its share of a round. Every prompt records its `parents`.
   - The output also records `gene_scores`. Quote the leaders in the round note.
4. **Generate.**
   - Follow `tatbot-flow`: a new Flow project per round, Nano Banana Pro, the plan's `aspect` (1:1 unless the round
     is a tile's), x3.
   - Submit with the send arrow, one prompt at a time, checking each was accepted.
   - Collect with `collect_zip.js`, and file with `file_batches.py` against `prompts-rN.json`.
5. **Separate, plan only.** A new design is a sketch: it needs its separation into the load's inks,
   not its robot paths.
   ```bash
   tatbot art separate IMG --inks <the prompt's inks> --style <the prompt's separation> --fill <the prompt's fill> --width-mm 60 --out ~/tatbot-art-corpus/separations/<id>
   ```
   Add `--app /opt/drawingbotv3` only when the owner wants every new design drawn at once.
6. **Upload.**
   - Make 640 px JPEG thumbnails, white-padded to a square:
     - `<id>-flow` from the Flow image;
     - `<id>-ours` from `target.png`;
     - `<id>-drawn` from `drawn.png`, for every design and fill-study variant the promote pass drew.
   - For rounds with recorded stencil placements, run the wall command in step 7 with
     `--stencil-previews DIR` before upload. Upload the exported PNGs under their filename stems
     alongside the square thumbnails. Rerun the wall command with the returned asset IDs in step 7.
   - Upload them as assets of the wall, up to 25 per call, from the scratchpad.
   - Add the returned ids to `assets.json`.
7. **Post.**
   ```bash
   tatbot art wall --round N --plan prompts-rN.json --corpus ~/tatbot-art-corpus --assets assets.json --out rN-designs.json
   ```
   - Rerun `wall` for every earlier round the promote pass touched. Their new drawn views and fill-study
     variants come out of the same command.
   - Write the documents with `ArtifactData batch` calls of at most 50 writes each. Use `set` for new
     documents and `update` with `if_version` for existing ones. Add `rounds/N` with a note.
   - Set each answered ask to `status: done` and `round: N`.
   - Say in the note what the votes favoured, which asks this round answers, and what it explores.
8. **Map.** Re-read `designs`, then run:
   ```bash
   tatbot art map --designs designs.json --votes votes.json --corpus ~/tatbot-art-corpus --out map.json
   ```
   Set `map/current` from the output, with `if_version`.
9. Tell the owner the round is up: how many designs, which asks it answered, and what the promote pass
   drew and studied.

## A tile round

The lab draws one design on each small stencil tile (forearm18: 40 x 62 mm over six 18 mm markers). Compose a
round for the tile; never fit a design made at another size down to one.

- Propose with `--tile forearm18`. Every prompt then asks for a small tattoo on the tile, mostly outline and bare
  skin with ink on no more than a third of it. Loads hold black and at most two colours, dotwork is left out
  (programs have no dot), and the plan's aspect is 9:16.
- Generate as above, at 9:16, and read the aspect chip back.
- Instead of step 5, separate the round at the tile's size onto a print of that tile:
  ```bash
  tatbot art tiles --plan ~/tatbot-art-corpus/flow/prompts-rN.json --corpus ~/tatbot-art-corpus \
    --stencil <a forearm18 print> --out ~/tatbot-art-corpus/tiles-rN.json
  ```
  Add `--app /opt/drawingbotv3` for the programs the lab draws, one a tile, with renders and minutes.
- Post as any round. In the round note, say it is composed for the tile, and quote from the index the designs whose
  colour is mostly under 2 mm, the inks that drew nothing and, once acquired, the minutes and how many are over the
  tile's budget (51 min for forearm18). Report the budget beside the votes; never drop a design for it.

## Judging the loop

- The votes are the objective. Never infer preference from the fidelity score or the minutes.
- Do report minutes beside the votes. A favourite that takes three hours is worth knowing about
  before it reaches the rig.
- Keep every round's designs on the wall. The round selector filters them, and old votes keep scoring the
  genes.
