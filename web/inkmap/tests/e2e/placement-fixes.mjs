import assert from "node:assert/strict";
import { mkdirSync, mkdtempSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import * as THREE from "three";
import { chromium } from "playwright";

const url = process.argv[2] ?? "http://127.0.0.1:4186/";
const out = process.env.INKMAP_E2E_EVIDENCE ? join(process.env.INKMAP_E2E_EVIDENCE, "placement-fixes") : mkdtempSync(join(tmpdir(), "inkmap-placement-fixes-"));
mkdirSync(out, { recursive: true });
const browser = await chromium.launch({ channel: "chrome", headless: true, args: ["--use-gl=swiftshader"] });
const errors = [], checks = [];
const ready = page => page.waitForFunction(() => { const s = window.__inkmap?.getState(); return s?.body && s.atlas && s.projectReady && s.designs.length; }, null, { timeout: 120_000 });
const describe = async (page, text) => {
  await page.getByRole("textbox", { name: "Describe a location", exact: true }).fill(text);
  await page.getByRole("button", { name: "Resolve", exact: true }).click();
};
async function cameraVisible(page) {
  await page.waitForFunction(() => {
    const s = window.__inkmap.getState(), p = s.placements.find(p => p.id === s.selected), g = s.body.skin.geometry;
    if (!p || !s.cameraSnapshot) return false;
    return s.cameraSnapshot.target.every((v, axis) => Math.abs(v - p.anchor.barycentric.reduce((sum, w, i) => {
      const vertex = g.index ? g.index.array[p.anchor.face * 3 + i] : p.anchor.face * 3 + i;
      return sum + w * g.attributes.position.array[vertex * 3 + axis];
    }, 0)) < 1e-6);
  });
  const data = await page.evaluate(() => { const s = window.__inkmap.getState(), g = s.body.skin.geometry;
    return { camera: s.cameraSnapshot, positions: Array.from(g.attributes.position.array), indices: g.index ? Array.from(g.index.array) : null }; });
  const geometry = new THREE.BufferGeometry().setAttribute("position", new THREE.Float32BufferAttribute(data.positions, 3));
  if (data.indices) geometry.setIndex(data.indices);
  const material = new THREE.MeshBasicMaterial({ side: THREE.DoubleSide });
  const eye = new THREE.Vector3(...data.camera.position), target = new THREE.Vector3(...data.camera.target);
  const hit = new THREE.Raycaster(eye, target.clone().sub(eye).normalize()).intersectObject(new THREE.Mesh(geometry, material))[0];
  assert.ok(hit && hit.point.distanceTo(target) < .002, "the live camera must see the selected skin point, without the torso in front");
  geometry.dispose(); material.dispose();
}
async function reachable(page, name, height) {
  const button = page.getByRole("button", { name, exact: true }), box = await button.boundingBox();
  assert.ok(box && box.y >= 0 && box.y + box.height <= height, `${name} is off screen after scrolling`);
  assert.ok(await button.evaluate(el => { const r = el.getBoundingClientRect(); const hit = document.elementFromPoint(r.x + r.width / 2, r.y + r.height / 2); return hit === el || el.contains(hit); }), `${name} is obscured`);
}
try {
  const desktop = await browser.newPage({ viewport: { width: 1440, height: 1000 } });
  desktop.on("pageerror", e => errors.push(e.message));
  await desktop.goto(url); await ready(desktop);
  await desktop.getByRole("button", { name: "dbv3-sprout", exact: true }).click();
  for (const prompt of ["not on my left forearm", "don't put it on my left forearm"]) {
    await describe(desktop, prompt);
    assert.equal(await desktop.evaluate(() => window.__inkmap.getState().placements.length), 0);
    await desktop.locator(".sentence [role=alert]").waitFor();
  }
  await describe(desktop, "on the inside of my left forearm");
  await desktop.waitForFunction(() => window.__inkmap.getState().selected);
  await cameraVisible(desktop);
  // A normal stock placement at a canonical forearm point whose normal faces
  // through the torso. Focus must work without any composition metadata.
  await desktop.evaluate(() => { const s = window.__inkmap.getState(); s.update(s.selected, {
    anchor: { face: 8327, barycentric: [.3201246437648309, .1227676776669629, .5571076785682062] },
    rotation_rad: 2.032666503737368, size_mm: [28, 44],
  }); });
  assert.equal(await desktop.evaluate(() => window.__inkmap.getState().error), null);
  await desktop.getByRole("toolbar", { name: "Adjust tattoo" }).getByRole("button", { name: "Focus", exact: true }).click();
  await cameraVisible(desktop);
  await desktop.screenshot({ path: join(out, "desktop-forearm-focus.png") });
  await desktop.getByRole("button", { name: "✓ Accept", exact: true }).click();
  await desktop.waitForFunction(() => window.__inkmap.getState().saveStatus === "saved");
  await desktop.reload(); await ready(desktop);
  assert.equal(await desktop.evaluate(() => window.__inkmap.getState().placements[0].design_id), "dbv3-sprout");
  checks.push("natural possessives, negation refusal, visible stock-artwork focus, reload");
  for (const height of [844, 600]) {
    const page = await browser.newPage({ viewport: { width: 390, height }, hasTouch: true });
    page.on("pageerror", e => errors.push(e.message));
    await page.goto(url); await ready(page);
    for (const action of ["✕ Cancel", "✓ Accept"]) {
      await page.getByRole("button", { name: "dbv3-sprout", exact: true }).tap();
      await describe(page, "on my left shin");
      await page.waitForFunction(() => window.__inkmap.getState().selected);
      await page.locator(".provenance > summary").tap();
      await page.locator(".dock-scroll").evaluate(el => { el.scrollTop = el.scrollHeight; });
      await reachable(page, "✓ Accept", height); await reachable(page, "✕ Cancel", height);
      assert.equal(await page.evaluate(() => document.documentElement.scrollWidth), 390);
      await page.screenshot({ path: join(out, `mobile-${height}-${action.includes("Accept") ? "accept" : "cancel"}.png`) });
      await page.getByRole("button", { name: action, exact: true }).tap();
      await page.waitForFunction(() => !window.__inkmap.getState().selected);
      assert.equal(await page.evaluate(() => window.__inkmap.getState().placements.length), action.includes("Accept") ? 1 : 0);
    }
    checks.push(`mobile 390×${height}: Accept and Cancel remain visible, unobscured and usable after scrolling`);
    await page.close();
  }
  assert.deepEqual(errors, []);
  writeFileSync(join(out, "report.json"), JSON.stringify({ url, result: "pass", checked_at: new Date().toISOString(), checks, errors }, null, 2));
  console.log("PASS placement language, ordinary focus and scrolled mobile actions", out);
} finally { await browser.close(); }
