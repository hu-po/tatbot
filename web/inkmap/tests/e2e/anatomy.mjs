import assert from "node:assert/strict";
import { chromium } from "playwright";

// The arm as a person sees it: on a body standing with its arms down, not the rest surface's T-pose.
const url = process.argv[2] ?? "http://127.0.0.1:4186/";
const browser = await chromium.launch({ channel: "chrome", headless: true, args: ["--use-gl=swiftshader"] });
const errors = [];

/** How far a placement's art is turned from upright against gravity on the posed body, degrees. */
const tilt = () => {
  const s = window.__inkmap.getState(), p = s.placements.find(item => item.id === s.selected), rest = s.body.restSkin.geometry, posed = s.body.skin.geometry;
  const at = (g, k) => { const i = g.index ? g.index.getX(3 * p.anchor.face + k) : 3 * p.anchor.face + k, a = g.attributes.position; return [a.getX(i), a.getY(i), a.getZ(i)]; };
  const sub = (a, b) => a.map((v, i) => v - b[i]), dot = (a, b) => a[0] * b[0] + a[1] * b[1] + a[2] * b[2], scale = (a, t) => a.map(v => v * t);
  const cross = (a, b) => [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]], unit = a => scale(a, 1 / Math.hypot(...a));
  const [r0, r1, r2] = [0, 1, 2].map(k => at(rest, k)), [q0, q1, q2] = [0, 1, 2].map(k => at(posed, k));
  const normal = unit(cross(sub(r1, r0), sub(r2, r0))), up = [0, 0, 1];
  const v0 = unit(sub(up, scale(normal, dot(up, normal)))), side = cross(normal, v0);
  const v = v0.map((c, i) => c * Math.cos(p.rotation_rad) + side[i] * Math.sin(p.rotation_rad));
  const [e1, e2] = [sub(r1, r0), sub(r2, r0)], [g11, g12, g22, b1, b2] = [dot(e1, e1), dot(e1, e2), dot(e2, e2), dot(v, e1), dot(v, e2)], det = g11 * g22 - g12 * g12;
  const [f1, f2] = [sub(q1, q0), sub(q2, q0)], carried = f1.map((c, i) => c * (b1 * g22 - b2 * g12) / det + f2[i] * (b2 * g11 - b1 * g12) / det);
  const pn = unit(cross(f1, f2)), top = unit(sub(carried, scale(pn, dot(carried, pn)))), plumb = unit(sub(up, scale(pn, dot(up, pn))));
  return Math.atan2(dot(cross(plumb, top), pn), dot(plumb, top)) * 180 / Math.PI;
};

try {
  const page = await browser.newPage({ viewport: { width: 1440, height: 900 } });
  page.on("pageerror", error => errors.push(error.message));
  await page.goto(url);
  await page.waitForFunction(() => { const s = window.__inkmap?.getState(); return s?.body && s.atlas && s.projectReady && s.designs.length; }, null, { timeout: 120_000 });
  const describe = async text => {
    await page.getByRole("textbox", { name: "Describe a location", exact: true }).fill(text);
    await page.getByRole("button", { name: "Resolve", exact: true }).click();
  };
  const rotation = page.getByRole("spinbutton", { name: "Rotation in degrees", exact: true });

  await describe("on the outer left biceps");
  await page.getByRole("button", { name: "Swallow and poppies", exact: true }).click();
  await rotation.waitFor();
  const placed = await page.evaluate(() => { const s = window.__inkmap.getState(), p = s.placements[0]; return { site: p.site, sentence: p.language.sentence }; });
  assert.equal(placed.site.aspect, "outer");
  assert.match(placed.sentence, /left .*outer bicep/);

  // Upright as it lands, and still upright wherever it is moved round the arm, the box reading 0 throughout.
  assert.equal(await rotation.inputValue(), "0");
  assert.ok(Math.abs(await page.evaluate(tilt)) < 25, `art ${await page.evaluate(tilt)} deg from plumb`);
  for (const face of [6749, 6529, 6345, 6383]) {
    await page.evaluate(id => { const s = window.__inkmap.getState(); s.update(s.selected, { anchor: { face: id, barycentric: [1 / 3, 1 / 3, 1 / 3] } }); }, face);
    assert.equal(await rotation.inputValue(), "0", `face ${face}`);
    assert.ok(Math.abs(await page.evaluate(tilt)) < 25, `face ${face}: art ${await page.evaluate(tilt)} deg from plumb`);
  }

  // A turn the person gave it is kept across the same move.
  await rotation.fill("30");
  await page.evaluate(() => { const s = window.__inkmap.getState(); s.update(s.selected, { anchor: { face: 6749, barycentric: [1 / 3, 1 / 3, 1 / 3] } }); });
  assert.equal(await rotation.inputValue(), "30");
  await rotation.fill("0");

  // The real outer arm is named for what it is.
  assert.match(await page.evaluate(() => { const s = window.__inkmap.getState(); return s.placements[0].language.sentence; }), /outer/);
  await page.getByRole("button", { name: /Accept/ }).click();

  // A saved placement keeps its anchor and its look; only the words an older atlas gave it are made again.
  const before = await page.evaluate(() => {
    const s = window.__inkmap.getState(), file = s.toFile();
    const old = structuredClone(file);
    const p = old.placements[0];
    p.site.lexicon = "0.3"; p.site.aspect = null; p.language.resolution.resolver.version = 1; p.language.resolution.actual.aspect = null; p.language.sentence = "a swallow on the left tricep";
    s.loadFile(old);
    return { anchor: file.placements[0].anchor, rotation: file.placements[0].rotation_rad };
  });
  const after = await page.evaluate(() => { const p = window.__inkmap.getState().placements[0]; return { anchor: p.anchor, rotation: p.rotation_rad, lexicon: p.site.lexicon, aspect: p.site.aspect, sentence: p.language.sentence }; });
  assert.deepEqual([after.anchor, after.rotation], [before.anchor, before.rotation]);
  assert.equal(after.lexicon, "0.4");
  assert.equal(after.aspect, "outer");
  assert.match(after.sentence, /outer/);

  // A size in the sentence sets the width, and the art is found without it.
  await page.getByRole("button", { name: "Add artwork", exact: true }).click();
  await describe("a 3 inch sun and moon on my left outer bicep");
  await rotation.waitFor();
  assert.equal(await page.evaluate(() => { const s = window.__inkmap.getState(); return s.placements.find(p => p.id === s.selected).size_mm[0]; }), 76.2);
  assert.deepEqual(errors, []);
  console.log("anatomy: ok");
} finally {
  await browser.close();
}
