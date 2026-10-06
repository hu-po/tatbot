---
summary: InkLang tattoo-placement language and body-surface grounding
tags: [inklang, inkmap, placement, schema]
updated: 2026-09-03
audience: [artist, dev, contributor]
---

# InkLang

InkLang is Tatbot's system for turning a human description of a tattoo-site
placement into an actual location on Tatbot's fixed MHR-through-SOMA body. The
location is a face index plus barycentric coordinates on its canonical
rest surface; a surface digest identifies exactly which geometry those
coordinates mean.

InkLang is placement-only. Motif, visual style, color, SVG generation, body
pose, world and robot transforms, reachability, inverse kinematics, contact,
and robot motion belong to other contracts. There is no body selector;
identity variation is an explicit `BodyIdentity/1`, never inferred from
anatomy words.

```text
placement description
        |
        v
body-independent intent -- needs_choice / rejected
        |
        | + fixed model/identity digests and versioned region atlas
        v
rest-surface resolution = face + barycentric coordinates
        |
        +--> Inkmap preview / manual adjustment
        +--> PlacementFile v6
        +--> TattooScenario (pose and simulation are downstream)
```

## Examples

| Kind | Input or action | Result |
| --- | --- | --- |
| plain | `on the left forearm` | one default anchor inside `forearm:left` |
| refined | `left upper inner forearm` | one anchor inside the requested level and aspect partition |
| relative | `two inches below the left collarbone` | a bounded two-inch surface walk from the collarbone anchor |
| ambiguous | `forearm` | `needs_choice` with left and right candidates; nothing is placed yet |
| rejected | `on the left sternum` | `INKLANG_INVALID_LATERALITY`; the midline site cannot be left-sided |
| reverse | manually click a labeled face | an actual leaf site, region UV, and canonical placement phrase for that anchor |

The older full tattoo sentence remains compatible, for example “a fine line
octopus on the left knee ditch.” Only its placement clause is InkLang. The
motif and style fields are handled by the surrounding legacy TattooRequest
adapter and may feed Inkgen; they never change how the location is grounded.

Resolve one description offline:

```bash
tatbot inkmap resolve \
  --prompt "left upper inner forearm"
```

The command emits canonical JSON. Use `--input FILE` for a JSON batch.
Interactive policy never guesses a missing side, multi-site zone member, or
`beside` direction: it returns concrete candidates. A non-interactive caller
must explicitly select `--policy seeded-v1 --seed N`; the policy and seed are
part of the resolution. Unknown or impossible descriptions return a structured
`rejected` result.

## Intent, atlas, and resolution

A successful path records three independently versioned contracts:

- The **intent** preserves the exact input in `description`, its normalized
  `canonical_phrase`, the structured site/laterality/aspect/level/relation,
  and any ambiguities, unknown terms, or parse issues.
- The **region atlas** binds InkLang vocabulary to the fixed model-spec,
  identity, topology, rest-surface, and browser-asset digests. It contains
  every face label, per-region chart, and reviewed default anchor.
- The **resolution** records resolver name/version/policy, body and surface,
  the concrete `anchor.face` and three barycentric weights, actual leaf site
  and `region_uv`, any explicit `choice`, and — for a relative placement — the
  `relative` walk record. An unresolved result carries candidates or named
  issues instead of an invented anchor.

`actual` always describes the placement that was made, never the wording that
asked for it. When a seeded policy or an accepted candidate resolves an
ambiguity, `actual.canonical_phrase` names the side and leaf site that were
chosen, so it re-resolves to the same anchor; the original wording stays in
`intent.canonical_phrase` and `intent.description`.

Versions have separate meanings:

- InkLang lexicon 0.3 defines 59 stable leaf sites plus 8 zones, aliases,
  laterality, 6 aspects, 3 levels, and 6 relative relation kinds.
- intent schema 1 describes body-independent meaning.
- atlas schema 2 describes the sole SOMA-specific grounding dataset.
- resolver version 1 and resolution schema 2 describe the grounding behavior
  and its result. Schema 2 adds the optional `relative` walk record.
- PlacementFile version 6 stores all model, identity, topology, and
  rest-surface bindings with canonical rest-surface anchors;
  pose belongs in TattooScenario, not PlacementFile.

Changing a body mesh, vocabulary, atlas representation, resolver algorithm,
or placement format is therefore a different change and must not be hidden
under another component's version.

## Relative placement

Relative distance is surface distance, not a Cartesian jump. The resolver
starts at the referenced default anchor, walks the connected labeled
rest-surface face graph using centroid-to-centroid edge costs, restricts the
search to the requested body-frame direction, and selects the deterministic
face nearest the requested path length. `between` selects the midpoint of the
shortest connected surface path. A direction that cannot reach the requested
distance fails with `INKLANG_OFFSET_OUT_OF_BOUNDS`.

A resolved relative placement reports what the walk actually did:

```json
"relative": {
  "kind": "below",
  "requested_m": 0.0508,
  "achieved_m": 0.049796991458917020,
  "reference_face": 6540
}
```

`achieved_m` is surface path length, not straight-line distance, so it exceeds
the chord between the two anchors wherever the body curves. The walk cannot
land between faces, so its accuracy is bounded by the mesh: across every leaf
site, all five directional kinds, and offsets from 1 cm to 30 cm on the fixed
MHR-through-SOMA body, `|achieved_m - requested_m|` stays within twice the
atlas's longest centroid-to-centroid step (`AtlasIndex.surfaceStepLimit()`). A
requested offset near or below that step is therefore satisfied only coarsely
— check `achieved_m` and the atlas's reported step limit rather than assuming
the request was met exactly.

## Anchor precision

An anchor is a face index plus barycentric weights, but the resolver only ever
places anchors at the face centroid, `[1/3, 1/3, 1/3]`. Placement precision
from a description is therefore quantized to one face. Interior barycentric
weights are reserved for manual placement in Inkmap, where a person positions
the design directly. A consumer must accept any valid barycentric triple, and
must not assume a resolved anchor is finer than its face.

## Responsibilities

| Component | Owns | Does not own |
| --- | --- | --- |
| InkLang | placement parsing, normalization, grounding, reverse description, and semantic validation | artwork, pose, robot transforms, reach, or motion |
| Inkmap | browser preview/editor, ambiguity confirmation, manual placement, and PlacementFile authoring | another grammar or resolver |
| Inkgen | motif/style prompt to a source image for DBV3 acquisition | body location |
| PlacementFile | design plus canonical rest-surface anchor and provenance | pose or executable behavior |
| TattooScenario | one resolved pose/support/world/tool/surface-trace realization | changing the placement's rest-surface meaning |

Simulation invokes the same TypeScript resolver and consumes its complete JSON;
there is no Python grammar or competing semantic face chooser. A downstream
sampling policy may filter canonically resolved region-UV candidates for a
pose, but it cannot redefine the requested site.

## Vocabulary and extension rules

The vocabulary source is
[`config/inkmap/sites.json`](../config/inkmap/sites.json). Site identifiers are
stable machine keys. Add colloquial wording as an alias when meaning is
unchanged; add a leaf only when it represents a distinct commercial placement,
and bump the lexicon version when the accepted meaning changes. Aspects and
levels refine an existing leaf instead of multiplying near-duplicate site ids.
A zone is explicitly non-unique and therefore resolves to candidates or a
named deterministic policy.

Every vocabulary or geometry change must regenerate the normative corpus and
the SOMA atlas, pass exhaustive containment and determinism checks, and receive a
visual anchor review. The checked-in
[`config/inkmap/examples/inklang/corpus-v1.json`](../config/inkmap/examples/inklang/corpus-v1.json) contains
146 cases covering the fixed body and all grounding axes. Structured issue meanings are in
[`config/inkmap/inklang-errors.json`](../config/inkmap/inklang-errors.json).

## Schemas, implementation, and checks

- [`config/inkmap/inklang-intent.schema.json`](../config/inkmap/inklang-intent.schema.json)
- [`config/inkmap/region-atlas.schema.json`](../config/inkmap/region-atlas.schema.json)
- [`config/inkmap/inklang-resolution.schema.json`](../config/inkmap/inklang-resolution.schema.json)
- reference core: `web/inkmap/src/core/inklang/`
- legacy full-sentence adapter: `web/inkmap/src/core/lang.ts`
- offline CLI: `web/inkmap/tools/resolve.ts`
- generated atlases: `web/inkmap/public/bodies/*.regions.json`

```bash
npm --prefix web/inkmap run check
uvx pytest -q scripts/tests/test_inklang.py
```

InkLang and Inkmap are design and simulation tools. A valid resolution does not
demonstrate physical reachability, authorize powered motion, qualify contact,
model deformable human tissue, or establish readiness for human use.
