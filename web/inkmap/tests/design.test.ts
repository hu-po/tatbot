import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";
import { SvgDOMParser } from "../tools/svg-dom.ts";
import { designFromBodyFile } from "../src/core/body-design.ts";
import { makeDesign, parseDesign, validateDesign } from "../src/core/design.ts";
import { makeSurfacePlacement } from "../src/core/surface-placement.ts";
import { canonicalDigest } from "../src/core/human-representation/schema.ts";
import { makeSimulationBundle, simulationRequest } from "../src/core/sim-bundle.ts";
import type { AtlasData } from "../src/core/atlas.ts";
import type { PlacementFile } from "../src/core/schema.ts";

Object.assign(globalThis, { DOMParser: SvgDOMParser });
const atlas = JSON.parse(readFileSync(new URL("../public/bodies/mhr-soma-v1.regions.json", import.meta.url), "utf8")) as AtlasData;
const file = JSON.parse(readFileSync(new URL("../../../config/inkmap/examples/forearm-placement-v6.json", import.meta.url), "utf8")) as PlacementFile;

test("body design and simulation export use identical frozen artwork and placement intent", async () => {
  const design = await designFromBodyFile("Body artwork", file, atlas);
  const bundle = await makeSimulationBundle(file, atlas, simulationRequest("supine", "#c07f57", null, "lutin-3rl-bugpin", 42));
  assert.deepEqual(design.artworks, bundle.artworks);
  assert.deepEqual(design.placements.map(item => item.id), bundle.surface_placements.map(item => item.id));
  const placed = design.placements[0].placement;
  const old = bundle.surface_placements[0].placement;
  assert.equal(placed.target.kind, "body");
  assert.deepEqual(placed.target.anchor, old.anchor);
  assert.deepEqual(placed.target.supported_domain, old.supported_domain);
  assert.deepEqual(placed.physical_scale_m, old.physical_scale_m);
  assert.deepEqual(await parseDesign(JSON.stringify(design)), design);
});

test("plane and cylinder use the same ordered portable design without body assets", async () => {
  const body = await designFromBodyFile("Source", file, atlas);
  for (const kind of ["plane", "cylinder"] as const) {
    const original = body.placements[0];
    const placed = await makeSurfacePlacement({ ...original.placement,
      target: { canvas_m: [.1, .1], anchor_uv_m: [0, 0], margin_m: .003,
        ...(kind === "cylinder" ? { kind, radius_m: .03 } : { kind }) }, physical_scale_m: [.02, .03],
    });
    const design = await makeDesign({ name: kind, artworks: body.artworks, placements: [
      { ...original, placement: placed }, { ...original, id: "second", placement: placed },
    ] });
    assert.deepEqual(await parseDesign(JSON.stringify(design)), design);
    assert.deepEqual(design.placements.map(item => item.id), [original.id, "second"]);
    assert.equal(JSON.stringify(design).includes('"body_identity_sha256"'), false);
    const changed = structuredClone(design); changed.placements.reverse();
    await assert.rejects(validateDesign(changed), /digest/);
  }
});

test("rehashed designs cannot detach source artwork, omit bindings or duplicate placement IDs", async () => {
  const original = await designFromBodyFile("Source", file, atlas);
  for (const mutate of [
    (d: typeof original) => { d.placements[0].artwork_id = "missing"; },
    (d: typeof original) => { d.placements.push(structuredClone(d.placements[0])); },
    (d: typeof original) => { d.artworks.unused = Object.values(d.artworks)[0]; },
    (d: typeof original) => { d.placements[0].placement.tattoo_program_sha256 = "a".repeat(64); },
    (d: typeof original) => { Object.values(d.artworks)[0].name += " "; },
    (d: typeof original) => { Object.assign(d, { hardware_authority: true }); },
  ]) {
    const changed = structuredClone(original); mutate(changed);
    changed.placements[0].placement.content_sha256 = await canonicalDigest(changed.placements[0].placement);
    changed.content_sha256 = await canonicalDigest(changed);
    await assert.rejects(validateDesign(changed));
  }
  await assert.rejects(parseDesign('{"schema":1,"schema":2}'), /duplicate/);
});

// The Python design builder (tatbot_sim.inkmap.design_build, `tatbot design place`)
// writes designs this reader must accept. The fixture is regenerated and compared
// byte for byte by python/tatbot_sim/tests/test_design_build.py, so the two
// builders cannot drift apart without one of these two tests failing.
test("a cylinder design built by the Python CLI opens in this reader", async () => {
  const raw = readFileSync(new URL("fixtures/python-cylinder-design.json", import.meta.url), "utf8");
  const design = await parseDesign(raw);
  assert.equal(design.schema, "tatbot.inkmap-design/1");
  assert.equal(design.content_sha256, await canonicalDigest(design));
  assert.deepEqual(await validateDesign(design), design);
  const placed = design.placements[0].placement;
  assert.equal(placed.target.kind, "cylinder");
  assert.equal(placed.target.radius_m, 0.04);
  assert.deepEqual(placed.target.canvas_m, [0.08, 0.11]);
  // Arc length must stay inside the chart's injective region: short of closing on itself.
  assert.ok(placed.target.canvas_m[1] < 2 * Math.PI * placed.target.radius_m);
  const artwork = design.artworks[design.placements[0].artwork_id];
  assert.equal(placed.tattoo_program_sha256, artwork.program.content_sha256);
  assert.equal(artwork.conversion.adapter, "dbv3-batik-paths/1");
});
