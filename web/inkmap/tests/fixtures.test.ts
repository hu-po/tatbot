import test from "node:test";
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { atFixtureDimensions, fixtureCaption, fixtureDimensions, GRID_PITCH_MM, PAPER_CYLINDER, PAPER_PAD } from "../src/core/fixtures.ts";
import { EMPTY_DRAFT } from "../src/core/chart-draft.ts";

/** The bench's substrate registry, read the simple way: `key: value` lines under a named block. */
function substrate(name: string): Record<string, string> {
  const text = readFileSync(fileURLToPath(new URL("../../../config/substrates.yaml", import.meta.url)), "utf8");
  const out: Record<string, string> = {};
  let inside = false;
  for (const line of text.split("\n")) {
    if (/^\S/.test(line)) { inside = line.startsWith(`${name}:`); continue; }
    const match = inside ? /^\s+([a-z_]+):\s*(.+?)\s*$/.exec(line) : null;
    if (match) out[match[1]] = match[2];
  }
  assert.ok(Object.keys(out).length, `no substrate ${name} in config/substrates.yaml`);
  return out;
}

const mm = (metres: string) => Number((Number(metres) * 1000).toFixed(3));

test("the editor's paper pad is the bench's paper pad", () => {
  const pad = substrate("paper_pad");
  assert.deepEqual(PAPER_PAD.canvas_mm, [mm(pad.width_m), mm(pad.height_m)]);
  assert.equal(PAPER_PAD.thickness_mm, mm(pad.thickness_m));
  assert.equal(PAPER_PAD.grid_pitch_mm, mm(pad.grid_pitch_m));
  assert.equal(pad.shape, undefined);
  assert.deepEqual(PAPER_PAD.canvas_mm, [190.5, 279.4]);   // 7.5 × 11 in
  assert.equal(GRID_PITCH_MM, 6.35);                        // 1/4 in
});

test("the editor's paper cylinder is the bench's paper cylinder", () => {
  const tube = substrate("paper_cylinder");
  assert.equal(tube.shape, "cylinder");
  assert.equal(PAPER_CYLINDER.radius_mm, mm(tube.diameter_m) / 2);
  assert.equal(PAPER_CYLINDER.thickness_mm, mm(tube.thickness_m));
  // the chart's u is the axis (the substrate's height), v the band's arc (its width)
  assert.equal(PAPER_CYLINDER.canvas_mm[0], mm(tube.height_m));
  assert.ok(Math.abs(PAPER_CYLINDER.canvas_mm[1] - mm(tube.width_m)) < 0.05, `${PAPER_CYLINDER.canvas_mm[1]} vs ${mm(tube.width_m)}`);
  assert.equal(PAPER_CYLINDER.grid_pitch_mm, mm(tube.grid_pitch_m));
  assert.equal(PAPER_CYLINDER.radius_mm, 42.5);             // 85 mm across
  assert.equal(PAPER_CYLINDER.canvas_mm[0], 190.5);          // 7.5 in long
  // the band is the outer surface less the bottom quarter, short of the full circumference the contract refuses
  assert.ok(Math.abs(PAPER_CYLINDER.canvas_mm[1] - 1.5 * Math.PI * PAPER_CYLINDER.radius_mm) < 0.05);
  assert.ok(PAPER_CYLINDER.canvas_mm[1] < 2 * Math.PI * PAPER_CYLINDER.radius_mm);
});

test("a fresh chart is the pad, and a switch to the cylinder adopts the cylinder", () => {
  assert.equal(EMPTY_DRAFT.kind, "plane");
  assert.deepEqual([EMPTY_DRAFT.width, EMPTY_DRAFT.height], PAPER_PAD.canvas_mm);
  assert.equal(EMPTY_DRAFT.radius, PAPER_CYLINDER.radius_mm);
  assert.ok(atFixtureDimensions(EMPTY_DRAFT));
  assert.ok(atFixtureDimensions({ kind: "cylinder", ...fixtureDimensions("cylinder") }));
  assert.ok(!atFixtureDimensions({ ...EMPTY_DRAFT, width: 100 }));
  // a typed cylinder radius is a change; a typed pad radius is not a pad dimension
  assert.ok(!atFixtureDimensions({ kind: "cylinder", ...fixtureDimensions("cylinder"), radius: 60 }));
  assert.ok(atFixtureDimensions({ ...EMPTY_DRAFT, radius: 60 }));
  // Older canvas dimensions are still explicit dimensions, not permission to resize.
  assert.ok(!atFixtureDimensions({ kind: "cylinder", width: 190.5, height: 66.76, radius: 42.5 }));
  assert.ok(!atFixtureDimensions({ kind: "plane", width: 190.5, height: 66.76, radius: 42.5 }));
});

test("the caption names the fixture in bench units", () => {
  assert.match(fixtureCaption("plane"), /190\.5 × 279\.4 × 10 mm \(7\.5 in × 11 in\) · ¼ in grid/);
  assert.match(fixtureCaption("cylinder"), /⌀85 × 190\.5 mm long \(7\.5 in\) · ¼ in grid/);
});
