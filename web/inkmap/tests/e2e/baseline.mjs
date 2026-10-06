// Read-only server review; all edits live in this disposable browser context.
import { chromium } from 'playwright';
import { mkdir, writeFile, readFile } from 'node:fs/promises';
import { createHash } from 'node:crypto';
import { execFileSync } from 'node:child_process';
import path from 'node:path';

const [url, output] = process.argv.slice(2);
if (!url || !output || !path.isAbsolute(output)) throw new Error('usage: baseline.mjs URL ABSOLUTE_EVIDENCE_DIR');
const root = execFileSync('git', ['rev-parse', '--show-toplevel'], { encoding: 'utf8' }).trim();
const designIds = JSON.parse(await readFile(new URL('../../public/designs/manifest.json', import.meta.url), 'utf8')).designs.map(design => design.id);
if (output === root || output.startsWith(root + '/')) throw new Error('evidence must be outside the repository');
await mkdir(output, { recursive: true });
const browser = await chromium.launch({ headless: true, args: ['--ignore-gpu-blocklist', '--use-gl=swiftshader'] });
const result = { schema: 'tatbot.inkmap-baseline/1', checked_at: new Date().toISOString(), url,
  vantage: 'local', revision: execFileSync('git', ['rev-parse', 'HEAD'], { encoding: 'utf8' }).trim(),
  dirty: Boolean(execFileSync('git', ['status', '--porcelain'], { encoding: 'utf8' }).trim()), errors: [], fixtures: {} };
try {
  const page = await browser.newPage({ viewport: { width: 1440, height: 1000 } });
  page.on('pageerror', error => result.errors.push(error.message));
  await page.goto(url);
  await page.waitForFunction(ids => { const s = window.__inkmap?.getState(); return s?.atlas
    && s.designs.length === ids.length && ids.every(id => s.designs.some(design => design.id === id)); }, designIds);
  result.initial_text = await page.locator('body').innerText();
  await page.screenshot({ path: path.join(output, 'desktop.png') });
  await page.locator('.sentence input').fill('on the thigh');
  await page.locator('.sentence button.primary').click();
  result.ambiguity = await page.locator('.resolution').innerText();
  await page.locator('.sentence input').fill('on the left forearm');
  await page.locator('.sentence button.primary').click();
  await page.locator('.picker button').first().click();
  result.export = await page.evaluate(() => window.__inkmap.getState().toFile());
  await page.screenshot({ path: path.join(output, 'placement.png') });
  await page.locator('.acceptbar .accept').click();
  await page.locator('.poses select').selectOption('reclined-left-arm-supported');
  await page.waitForFunction(() => window.__inkmap.getState().body?.poseId === 'reclined-left-arm-supported');
  result.posed = await page.evaluate(() => { const s = window.__inkmap.getState(); return { pose: s.poseId, count: s.placements.length, error: s.error }; });
  await page.screenshot({ path: path.join(output, 'pose.png') });
  await page.reload();
  await page.waitForFunction(() => window.__inkmap?.getState().atlas);
  result.reload = await page.evaluate(() => ({ count: window.__inkmap.getState().placements.length, pose: window.__inkmap.getState().poseId }));
  await page.setViewportSize({ width: 390, height: 844 });
  result.mobile = await page.evaluate(() => ({ width: innerWidth, scrollWidth: document.documentElement.scrollWidth,
    canvas: document.querySelector('canvas').getBoundingClientRect().toJSON() }));
  await page.screenshot({ path: path.join(output, 'mobile.png') });
  for (const name of ['linework', 'blackwork', 'negative-space', 'stipple', 'color-layers']) {
    const bytes = await readFile(new URL(`../fixtures/artwork/${name}.svg`, import.meta.url));
    result.fixtures[name] = createHash('sha256').update(bytes).digest('hex');
  }
  await writeFile(path.join(output, 'manifest.json'), JSON.stringify(result, null, 2) + '\n');
  console.log(JSON.stringify(result, null, 2));
} finally { await browser.close(); }
