import { test } from "node:test";
import assert from "node:assert/strict";
import { createHash } from "node:crypto";
import { sha256 } from "../src/core/sha256.ts";

const hex = (a: Uint8Array) => Buffer.from(a).toString("hex");

test("pure-JS sha256 matches known vectors and node:crypto on odd sizes", () => {
  assert.equal(hex(sha256(new Uint8Array(0))), "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855");
  assert.equal(hex(sha256(new TextEncoder().encode("abc"))), "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad");
  for (const n of [55, 56, 63, 64, 65, 1000, 70001]) {
    const buf = new Uint8Array(n);
    for (let i = 0; i < n; i++) buf[i] = (i * 31 + 7) & 0xff;
    assert.equal(hex(sha256(buf)), createHash("sha256").update(buf).digest("hex"), `n=${n}`);
  }
});
