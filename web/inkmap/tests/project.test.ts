import test from "node:test";
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { makeProject, validateProject, parseProject } from "../src/core/project.ts";
import { canonicalDigest } from "../src/core/human-representation/schema.ts";
import type { PlacementFile } from "../src/core/schema.ts";

const fixture = JSON.parse(readFileSync(new URL("../../../config/inkmap/examples/forearm-placement-v7.json", import.meta.url), "utf8")) as PlacementFile;
const make = () => makeProject({ name: "Fixture", placement_file: fixture,
  editor: { pose_id: "supine", skin_tone: "#804030", show_atlas: true, camera: { position: [1, -2, 1], target: [0, 0, 1], fov: 38 } },
  history: { past: [[]], future: [] }, edit_before: null, selected_id: null });

test("strict project round-trip retains body, artwork, history, and editor settings", async () => {
  const p = await make();
  assert.deepEqual(await parseProject(JSON.stringify(p)), p);
  const changed = structuredClone(p); changed.editor.skin_tone = "#ffffff";
  changed.content_sha256 = await canonicalDigest(changed);
  assert.notEqual(changed.content_sha256, p.content_sha256);
  assert.deepEqual((await validateProject(changed)).placement_file, p.placement_file);
});

test("rehashed malformed histories, missing art, bad bodies, versions, and drafts fail closed", async () => {
  const p = await make();
  for (const mutate of [
    (r: any) => { r.unknown = true; },
    (r: any) => { r.schema = "tatbot.inkmap-project/99"; },
    (r: any) => { r.chart = null; },
    (r: any) => { r.placement_file.designs.ring.schema = "tatbot.inkmap-artwork/2"; },
    (r: any) => { r.placement_file.designs.ring.image.data_base64 = r.placement_file.designs.ring.image.data_base64.slice(4); },
    (r: any) => { r.editor.pose_id = "unknown"; },
    (r: any) => { r.placement_file.body.identity_sha256 = "0".repeat(64); },
    (r: any) => { r.history.past = [Array(2).fill(fixture.placements[0])]; },
    (r: any) => { r.history.past = [[{ ...fixture.placements[0], design_id: "missing" }]]; },
    (r: any) => { r.selected_id = fixture.placements[0].id; },
    (r: any) => { r.placement_file.placements[0].size_mm[0] = Infinity; },
  ]) {
    const broken = structuredClone(p); mutate(broken);
    await assert.rejects(async () => { broken.content_sha256 = await canonicalDigest(broken); await validateProject(broken); });
  }
  await assert.rejects(parseProject('{"schema":1,"schema":2}'), /duplicate_key/);
});

test("a project of another version is refused in one line, with no migration", async () => {
  const current = await make();
  for (const schema of ["tatbot.inkmap-project/1", "tatbot.inkmap-project/2", "tatbot.inkmap-project/3", "tatbot.inkmap-project/4"]) {
    const old = { ...current, schema }; old.content_sha256 = await canonicalDigest(old);
    await assert.rejects(validateProject(old), /^Error: project_invalid: this project is from another version of Inkmap and cannot open here$/);
    await assert.rejects(parseProject(JSON.stringify(old)), /another version of Inkmap/);
  }
});

test("a current project is version 5 and carries its art", async () => {
  const saved = await make();
  assert.equal(saved.schema, "tatbot.inkmap-project/5");
  assert.equal(saved.placement_file.schema_version, 7);
  assert.equal(saved.placement_file.designs!.ring.schema, "tatbot.inkmap-art/1");
});
