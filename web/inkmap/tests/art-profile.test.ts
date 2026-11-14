import assert from "node:assert/strict";
import test from "node:test";
import { artNotes, profileArt, suggestInks, thinInks } from "../src/core/art-profile.ts";
import { inkById } from "../src/core/inks.ts";

const ink = (id: string) => inkById.get(id)!;

/** An RGBA image of paper with rectangles of colour: [x, y, width, height, rgb]. */
function image(width: number, height: number, blocks: [number, number, number, number, [number, number, number]][]): Uint8ClampedArray {
  const rgba = new Uint8ClampedArray(width * height * 4).fill(255);
  for (const [x0, y0, w, h, rgb] of blocks) {
    for (let y = y0; y < y0 + h; y++) for (let x = x0; x < x0 + w; x++) rgba.set([...rgb, 255], 4 * (y * width + x));
  }
  return rgba;
}

const BLACK: [number, number, number] = [10, 10, 12], GREEN: [number, number, number] = [50, 150, 95], RED: [number, number, number] = [240, 70, 50];

test("coverage is the inked share of the rectangle; transparent and paper pixels are not ink", () => {
  const rgba = image(100, 100, [[0, 0, 50, 100, BLACK]]);
  for (let pixel = 0; pixel < 100; pixel++) rgba[4 * pixel + 3] = 0;   // a transparent row, half of it over the black
  assert.ok(Math.abs(profileArt(rgba, 100, 100, [ink("true_black")]).coverage - 0.495) < 1e-9);
});

test("an ink's radius is its fattest spot's, whatever its area", () => {
  // A 40 px square against a long 6 px stripe of the same area class.
  const { inks } = profileArt(image(200, 100, [[10, 10, 40, 40, GREEN], [60, 60, 130, 6, RED]]), 200, 100, [ink("light_green"), ink("bright_red")]);
  const [green, red] = inks;
  assert.ok(Math.abs(green.radiusOfWidth * 200 - 20) < 2.5, `square: ${green.radiusOfWidth * 200} px`);
  assert.ok(Math.abs(red.radiusOfWidth * 200 - 3) < 1, `stripe: ${red.radiusOfWidth * 200} px`);
});

test("a thin colour is at risk below the width that makes it 1 mm fat, and the darkest ink never is", () => {
  const inks = [ink("true_black"), ink("light_green")];
  const profile = profileArt(image(100, 100, [[0, 0, 4, 100, BLACK], [20, 20, 8, 60, GREEN]]), 100, 100, inks);
  const [risk] = thinInks(profile, inks, 20);
  assert.equal(risk.ink.id, "light_green");
  assert.ok(Math.abs(risk.minWidthMm - 25) < 4, `an 8 px stripe, radius 4 of 100 px, is 1 mm fat at 25 mm: ${risk.minWidthMm}`);
  assert.deepEqual(thinInks(profile, inks, 30), []);
  assert.ok(thinInks(profile, inks, 5).every(({ ink: item }) => item.id !== "true_black"), "a black line is widened, not opened");
});

test("an ink the art barely holds is its antialiasing, not a colour to lose", () => {
  const inks = [ink("true_black"), ink("bright_red")];
  const profile = profileArt(image(100, 100, [[0, 0, 60, 60, BLACK], [70, 70, 3, 3, RED]]), 100, 100, inks);
  assert.deepEqual(thinInks(profile, inks, 10), []);
});

test("suggested inks are the design's colours, most used first, a load at most", () => {
  assert.deepEqual(suggestInks(image(100, 100, [[0, 0, 60, 60, BLACK], [70, 70, 25, 25, RED]]), 100, 100), ["true_black", "bright_red"]);
  const many = suggestInks(image(120, 20, [[0, 0, 20, 20, [10, 10, 12]], [20, 0, 20, 20, [250, 200, 70]], [40, 0, 20, 20, [240, 70, 50]],
    [60, 0, 20, 20, [50, 150, 95]], [80, 0, 20, 20, [120, 200, 250]], [100, 0, 20, 20, [100, 80, 160]]]), 120, 20);
  assert.equal(many.length, 5);
  assert.equal(suggestInks(image(10, 10, []), 10, 10).length, 1, "a blank rectangle still names one ink");
  const greys: [number, number, number, number, [number, number, number]][] = [[0, 0, 30, 30, [0, 0, 0]], [30, 0, 30, 30, [10, 10, 12]], [60, 0, 30, 30, [45, 45, 48]]];
  assert.deepEqual(suggestInks(image(90, 30, greys), 90, 30), ["true_black"], "shades of one black are one ink, not a load of them");
});

test("a speck away from the design is noticed, with how much of the width the design really is", () => {
  const [note] = artNotes(image(200, 200, [[10, 10, 130, 130, BLACK], [190, 190, 3, 3, BLACK]]), 200, 200, 0);
  assert.match(note, /small mark away from the design/);
  assert.match(note, /about 65% of the width/);
});

test("a design with its parts spread out, or a margin the crop added, is not a speck", () => {
  assert.deepEqual(artNotes(image(200, 200, [[10, 10, 80, 80, BLACK], [110, 110, 80, 80, RED]]), 200, 200, 0), []);
  assert.deepEqual(artNotes(image(120, 120, [[10, 10, 100, 100, BLACK]]), 120, 120, 10), [], "the 3% crop margin is the box's own");
});

test("a pale background that did not read as paper is noticed, a dark one is not", () => {
  assert.match(artNotes(image(100, 100, [[0, 0, 100, 100, [222, 210, 188]]]), 100, 100, 0)[0], /pale background/);
  assert.deepEqual(artNotes(image(100, 100, [[0, 0, 100, 100, BLACK]]), 100, 100, 0), []);
});
