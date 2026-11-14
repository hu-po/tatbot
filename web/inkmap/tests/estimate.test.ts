import assert from "node:assert/strict";
import test from "node:test";
import { estimateCost, formatMinutes } from "../src/core/estimate.ts";

test("the estimate lands near what DBV3 and the compile measured for the swallow at three sizes", () => {
  // tatbot art separate, Swallow and poppies, five inks; its art rectangle is 40% inked.
  const measured: [number, number, number, number][] = [[30, 30.8, 43.6, 20], [45, 46.2, 82.0, 38], [60, 61.6, 136.6, 68]];
  for (const [width, height, minutes, dips] of measured) {
    const cost = estimateCost(0.4, [width, height], 5);
    assert.ok(Math.abs(cost.minutes - minutes) / minutes < 0.2, `${width} mm: ${cost.minutes} min against ${minutes}`);
    assert.ok(Math.abs(cost.dips - dips) / dips < 0.1, `${width} mm: ${cost.dips} dips against ${dips}`);
  }
});

test("each ink adds time and inked area scales both", () => {
  const one = estimateCost(0.5, [40, 40], 1), five = estimateCost(0.5, [40, 40], 5), big = estimateCost(0.5, [80, 80], 1);
  assert.ok(five.minutes > one.minutes && five.dips === one.dips + 4, "an ink starts with its own dip");
  assert.ok(Math.abs(big.dips - 1 - 4 * (one.dips - 1)) < 1e-9, "four times the area is four times the area's dips");
});

test("minutes read as a person says them", () => {
  assert.equal(formatMinutes(2), "5 min");
  assert.equal(formatMinutes(43), "45 min");
  assert.equal(formatMinutes(60), "1 h");
  assert.equal(formatMinutes(136.6), "2 h 15 min");
});
