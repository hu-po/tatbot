// The editor shell's pure helpers: what the short library shows, and how the
// body editor's mode is read off the placement store.
import { test } from "node:test";
import assert from "node:assert/strict";
import { shortLibrary } from "../src/library.ts";
import { bodyMode } from "../src/ui.ts";

test("the short library shows session artwork first, then the curated stock in manifest order", () => {
  const stock = (id: string) => ({ id, name: id, path: `data:,${id}`, default_size_mm: [30, 30] as [number, number], sourcePath: `designs/${id}.svg` });
  const session = (id: string) => ({ id, name: id, path: `data:,${id}`, default_size_mm: [30, 30] as [number, number] });
  const designs = [stock("a"), stock("b"), stock("c"), stock("d"), stock("e"), stock("f"), stock("g"), session("gen-1"), session("gen-2")];
  assert.deepEqual(shortLibrary(designs).map((d) => d.id), ["gen-2", "gen-1", "a", "b", "c", "d"]);
  assert.deepEqual(shortLibrary(designs.slice(0, 3)).map((d) => d.id), ["a", "b", "c"], "a fresh project shows the curated defaults, not an invented recency");
});

test("the body editor's mode is read off the store", () => {
  assert.equal(bodyMode(null, null, 0, false), "choose");
  assert.equal(bodyMode("leaf", null, 0, false), "place");
  assert.equal(bodyMode(null, "p-1", 1, false), "adjust");
  assert.equal(bodyMode(null, null, 1, false), "ready");
  assert.equal(bodyMode(null, null, 1, true), "choose");
});
