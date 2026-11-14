import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { chromium } from "playwright";

// What a chosen size means, read in the real editor: the stencil tiles, the fit action and the wrap.
const url = process.argv[2] ?? "http://127.0.0.1:4186/";
const browser = await chromium.launch({ channel: "chrome", headless: true, args: ["--use-gl=swiftshader"] });
const errors = [];
try {
  const page = await browser.newPage({ viewport: { width: 1440, height: 900 } });
  page.on("pageerror", error => errors.push(error.message));
  await page.route("**/designs/renders/**", route => route.abort());   // the estimate and Load render, without the stock's own
  await page.goto(url);
  await page.waitForFunction(() => { const s = window.__inkmap?.getState(); return s?.body && s.atlas && s.projectReady && s.designs.length; }, null, { timeout: 120_000 });
  await page.getByRole("textbox", { name: "Describe a location", exact: true }).fill("on the left forearm");
  await page.getByRole("button", { name: "Resolve", exact: true }).click();
  await page.getByRole("button", { name: "Swallow and poppies", exact: true }).click();
  const fit = page.getByRole("region", { name: "Fit and cost" });
  await fit.waitFor();

  // The default width is past the skin tile: said here, before anything is exported.
  assert.match(await fit.innerText(), /Forearm\s+45 × 63\s+15\.0 mm too wide/);
  assert.match(await fit.innerText(), /Silicone\s+111 × 157\.5\s+fits/);
  assert.match(await fit.innerText(), /Wraps\s+\d+° of the skin's curve/);

  // The estimate is near the separation's real 136.6 min and 68 dips, and at this size the leaves hold.
  const estimate = await fit.innerText();
  const [, hours, minutes, dips] = estimate.match(/about (\d) h(?: (\d+) min)?, (\d+) dips/);
  const total = 60 * Number(hours) + Number(minutes ?? 0);
  assert.ok(total >= 105 && total <= 165, `about ${total} min at 60 mm`);
  assert.ok(Number(dips) >= 50 && Number(dips) <= 85, `about ${dips} dips at 60 mm`);
  assert.doesNotMatch(estimate, /may be lost/);

  // Numbers are typed one key at a time: a first digit under the smallest width is a value still being written.
  const widthField = page.getByRole("spinbutton", { name: "Width in millimeters", exact: true });
  const rotationField = page.getByRole("spinbutton", { name: "Rotation in degrees", exact: true });
  const typed = async (field, keys) => { await field.click(); await page.keyboard.press("Control+a"); await page.keyboard.type(keys, { delay: 20 }); };
  const size = () => page.evaluate(() => window.__inkmap.getState().placements[0].size_mm[0]);
  await typed(widthField, "95");
  assert.equal(await size(), 95);
  await typed(widthField, "47.5");
  assert.equal(await size(), 47.5);
  await typed(widthField, "350");
  assert.equal(await widthField.getAttribute("aria-invalid"), "true");
  assert.equal(await size(), 35, "the last value in range holds while the field is past it");
  await page.keyboard.press("Tab");
  assert.equal(await widthField.inputValue(), "35", "leaving the field shows the placement's own value");
  const stored = () => page.evaluate(() => window.__inkmap.getState().placements[0].rotation_rad);
  const upright = await stored();
  await typed(rotationField, "-30");
  const turned = await stored() - upright;
  assert.ok(Math.abs(Math.atan2(Math.sin(turned), Math.cos(turned)) + Math.PI / 6) < 1e-9, "a negative number types as one");
  await typed(rotationField, "0");
  await page.keyboard.press("Enter");
  assert.equal(await page.evaluate(() => window.__inkmap.getState().selected !== null), true, "Enter in a field finishes the number, not the edit");
  await typed(widthField, "60");
  await page.keyboard.press("Tab");

  // A wider tile needs no action; back on the forearm one, the action fits the design and keeps its aspect.
  await fit.getByRole("radio", { name: /Silicone/ }).check();
  assert.equal(await fit.getByRole("button", { name: /Fit to/ }).count(), 0);
  await fit.getByRole("radio", { name: /^Forearm 45/ }).check();
  await fit.getByRole("button", { name: "Fit to Forearm (45.0 mm)" }).click();
  await page.waitForFunction(() => window.__inkmap.getState().placements[0].size_mm[0] === 45);
  const [width, height] = await page.evaluate(() => window.__inkmap.getState().placements[0].size_mm);
  assert.ok(Math.abs(width / height - 60 / 61.595) < 1e-3);
  assert.match(await fit.innerText(), /Forearm\s+45 × 63\s+fits/);

  // At 45 mm DBV3 really does drop the leaves: the panel says so, and the width that would hold them.
  const [, needed] = (await fit.innerText()).match(/Light Green may be lost below about (\d+) mm/);
  assert.ok(Number(needed) > 45 && Number(needed) <= 60, `the leaves need ${needed} mm`);

  // The inks are the art's, and a change reaches the record the brief carries.
  const inks = fit.getByRole("group", { name: "Inks" });
  assert.match(await inks.innerText(), /Golden Yellow[\s\S]*True Black/);
  await inks.getByRole("button", { name: "Remove Light Green" }).click();
  await page.waitForFunction(() => { const s = window.__inkmap.getState(); return !s.designs[0].embedded.palette.inks.includes("light_green"); });
  await page.waitForFunction(() => !document.querySelector(".fit").innerText.includes("Light Green may be lost"));
  await inks.getByRole("combobox", { name: "Add an ink" }).selectOption({ label: "Light Green" });
  await page.waitForFunction(() => window.__inkmap.getState().designs[0].embedded.palette.inks.at(-1) === "light_green");

  // The tile is remembered, and it is advice: an oversize design can still be accepted.
  await page.evaluate(() => { const s = window.__inkmap.getState(); s.update(s.selected, { size_mm: [100, 102.658] }); });
  await page.getByRole("button", { name: /Accept/ }).click();
  assert.equal(await page.evaluate(() => window.__inkmap.getState().placements.length), 1);
  assert.equal(await page.evaluate(() => localStorage.getItem("inkmap.tile")), "forearm");

  // A second design on the same skin is noticed, in the panel and in the list, and it is allowed.
  await page.getByRole("button", { name: "Add artwork", exact: true }).click();
  await page.getByRole("button", { name: "Sun and moon", exact: true }).click();
  await page.getByRole("textbox", { name: "Describe a location", exact: true }).fill("on the left forearm");
  await page.getByRole("button", { name: "Resolve", exact: true }).click();
  await page.waitForFunction(() => document.querySelector(".fit")?.innerText.includes("Overlaps"), null, { timeout: 10_000 });
  assert.match(await fit.innerText(), /Overlaps\s+Swallow and poppies/);
  await page.getByRole("button", { name: /Accept/ }).click();
  await page.getByRole("status").filter({ hasText: "Swallow and poppies overlaps Sun and moon" }).waitFor();

  // The card carries what the brief cannot: where to lay the tile, how to turn it, and the design at size.
  const [card] = await Promise.all([page.waitForEvent("download"), page.getByRole("button", { name: "Placement card", exact: true }).click()]);
  assert.equal(card.suggestedFilename(), "inkmap-card.html");
  const html = readFileSync(await card.path(), "utf8");
  assert.equal((html.match(/<section>/g) ?? []).length, 2, "one page per placement");
  assert.match(html, /about \d+ mm from the (wrist|tip of the elbow)/);
  assert.match(html, /Tile top (points at|turned \d+° (anti)?clockwise from pointing at) the top of the shoulder/);
  assert.match(html, /the design does not fit it/);
  const mockups = [...html.matchAll(/class="mockup" src="data:image\/jpeg;base64,([^"]+)"/g)];
  assert.equal(mockups.length, 2);
  assert.ok(mockups.every(([, data]) => data.length > 5000), "a real render of the body, not an empty buffer");
  assert.equal(await page.evaluate(() => window.__inkmap.getState().selected), null, "exporting changes nothing");

  await page.getByRole("combobox", { name: "Pose" }).selectOption({ label: "Reclined, left arm supported" });
  await page.waitForFunction(() => window.__inkmap.getState().body?.poseId === "reclined-left-arm-supported", null, { timeout: 60_000 });
  assert.deepEqual(errors, []);
  console.log("fit: ok");
} finally {
  await browser.close();
}
