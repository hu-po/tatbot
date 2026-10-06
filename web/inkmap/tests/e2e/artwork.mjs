// Compare the canonical artwork adapter with the browser's independent SVG rasterizer.
import { chromium } from 'playwright';
import { mkdir, readFile, writeFile } from 'node:fs/promises';
import { spawn } from 'node:child_process';
import path from 'node:path';
import { stripVTControlCharacters } from 'node:util';

const output = process.env.INKMAP_E2E_EVIDENCE;
if (!output || !path.isAbsolute(output)) throw new Error('set INKMAP_E2E_EVIDENCE to an absolute directory outside the checkout');
const repo = path.resolve(new URL('../../../../', import.meta.url).pathname);
if (path.resolve(output) === repo || path.resolve(output).startsWith(repo + '/')) throw new Error('evidence must be outside the checkout');
await mkdir(output, { recursive: true });
const pngBytes = dataUrl => Buffer.from(dataUrl.slice(dataUrl.indexOf(',') + 1), 'base64');
const pixelsPerMm = Number(process.env.INKMAP_ARTWORK_PIXELS_PER_MM ?? '4');
if (!Number.isInteger(pixelsPerMm) || pixelsPerMm < 4 || pixelsPerMm > 16) throw new Error('INKMAP_ARTWORK_PIXELS_PER_MM must be an integer from 4 through 16');
const server = spawn(process.execPath, ['node_modules/vite/bin/vite.js', '--host', '127.0.0.1', '--port', '4182', '--strictPort'], { stdio: ['ignore', 'pipe', 'pipe'] });
let serverLog = '';
server.stdout.on('data', chunk => { serverLog += chunk; });
server.stderr.on('data', chunk => { serverLog += chunk; });
let browser;
try {
  let ready = false;
  for (let i = 0; i < 120; i++) {
    if (server.exitCode !== null) throw new Error(serverLog);
    try { ready = (await fetch('http://127.0.0.1:4182/')).ok; } catch { /* starting */ }
    ready = ready && stripVTControlCharacters(serverLog).includes('http://127.0.0.1:4182/');
    if (ready) break;
    await new Promise(resolve => setTimeout(resolve, 250));
  }
  if (!ready) throw new Error('artwork test server did not start');
  browser = await chromium.launch({ headless: true, args: ['--ignore-gpu-blocklist', '--use-gl=swiftshader'] });
  const page = await browser.newPage({ viewport: { width: 1100, height: 1600 } });
  await page.goto('http://127.0.0.1:4182/tests/e2e/artwork.html', { waitUntil: 'networkidle' });
  await page.waitForFunction(() => window.artworkContract);
  const results = [];
  for (const name of ['linework', 'blackwork', 'negative-space', 'stipple', 'color-layers']) {
    const source = await readFile(new URL(`../fixtures/artwork/${name}.svg`, import.meta.url), 'utf8');
    const result = await page.evaluate(async ({ name, source, pixelsPerMm }) => {
      const { tattooProgramFromSvg, tattooProgramToSvg } = window.artworkContract;
      const box = source.match(/viewBox="([^"]+)"/)[1].split(/\s+/).map(Number);
      const options = { canvas_m: [box[2] / 1000, box[3] / 1000], semantic_intent: name,
        width_m: 0.0003, deposition: 1, chord_error_m: 0.000025 };
      const program = await tattooProgramFromSvg(source, options);
      const preview = await tattooProgramToSvg(program);
      const width = box[2] * pixelsPerMm, height = box[3] * pixelsPerMm;
      const render = async svg => {
        const canvas = document.createElement('canvas'); canvas.width = width; canvas.height = height;
        const ctx = canvas.getContext('2d');
        const img = new Image(); img.src = 'data:image/svg+xml;charset=utf-8,' + encodeURIComponent(svg); await img.decode();
        ctx.drawImage(img, 0, 0, width, height);
        return { pixels: ctx.getImageData(0, 0, width, height), png: canvas.toDataURL() };
      };
      const original = await render(source), actual = await render(preview);
      let union = 0, intersection = 0, rgbError = 0;
      const delta = document.createElement('canvas'); delta.width = width; delta.height = height;
      const ctx = delta.getContext('2d'); const pixels = ctx.createImageData(width, height);
      for (let i = 0; i < original.pixels.data.length; i += 4) {
        const a = original.pixels.data, b = actual.pixels.data;
        const insideA = a[i + 3] >= 128, insideB = b[i + 3] >= 128;
        if (insideA || insideB) union++;
        if (insideA && insideB) { intersection++; rgbError += Math.abs(a[i] - b[i]) + Math.abs(a[i + 1] - b[i + 1]) + Math.abs(a[i + 2] - b[i + 2]); }
        pixels.data[i] = insideA && !insideB ? 255 : 0;
        pixels.data[i + 1] = insideB && !insideA ? 255 : 0;
        pixels.data[i + 3] = 255;
      }
      ctx.putImageData(pixels, 0, 0);
      return { name, width, height, pixels_per_mm: pixelsPerMm, mask_iou: union ? intersection / union : 1,
        overlap_rgb_mae: intersection ? rgbError / intersection / 3 : 0, program,
        original: original.png, compiled: actual.png, difference: delta.toDataURL(), preview };
    }, { name, source, pixelsPerMm });
    results.push(result);
    await writeFile(path.join(output, `${name}.program.json`), JSON.stringify(result.program, null, 2) + '\n');
    await writeFile(path.join(output, `${name}.preview.svg`), result.preview);
    await writeFile(path.join(output, `${name}.original.png`), pngBytes(result.original));
    await writeFile(path.join(output, `${name}.browser.png`), pngBytes(result.compiled));
    await writeFile(path.join(output, `${name}.difference.png`), pngBytes(result.difference));
  }
  const widths = await page.evaluate(async () => {
    const { artworkFromSvg, embeddedFromArtwork, artworkTexture } = window.artworkContract;
    const art = await artworkFromSvg({ name: 'Metric line',
      original_svg: '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 50 75"><path d="M5 37.5L45 37.5" stroke="black" fill="none"/></svg>',
      source: { kind: 'fixture', identifier: 'metric-line', license: null, attribution: null, generation: null },
      conversion: { adapter: 'tatbot-svg-paint/1', canvas_m: [.05, .075], semantic_intent: 'Metric line',
        width_m: .0005, deposition: 1, chord_error_m: .00001, strokes: 'centerline' } });
    const body = { id: 'metric-line', name: art.name, path: 'unused', default_size_mm: [50, 75], embedded: embeddedFromArtwork(art) };
    const result = [];
    for (const [surface, source] of [['chart', art], ['body', body]]) {
      for (const scale of [.4, 1, 2]) {
        const size = [50 * scale, 75 * scale];
        const texture = await artworkTexture(source, size), canvas = texture.image;
        const pixels = canvas.getContext('2d').getImageData(Math.floor(canvas.width / 2), 0, 1, canvas.height).data;
        let coverage = 0;
        for (let row = 0; row < canvas.height; row++) coverage += pixels[row * 4 + 3] / 255;
        const widthMm = coverage * size[1] / canvas.height, toleranceMm = size[1] / canvas.height;
        result.push({ surface, size_mm: size, measured_width_mm: widthMm, tolerance_mm: toleranceMm });
        texture.dispose();
        if (Math.abs(widthMm - .5) > toleranceMm) throw new Error(`${surface} preview pen width changed at ${scale}x: ${widthMm} mm`);
      }
    }
    return result;
  });
  await writeFile(path.join(output, 'metric-widths.json'), JSON.stringify(widths, null, 2) + '\n');
  console.log(JSON.stringify(widths, null, 2));
  const html = '<!doctype html><meta charset="utf-8"><style>body{font:16px sans-serif;background:#ddd}article{display:flex;gap:15px}img{background:white}h2{margin-bottom:5px}</style><h1>Original / canonical programme / mask difference</h1>' + results.map(r => `<h2>${r.name} · IoU ${r.mask_iou.toFixed(4)} · RGB MAE ${r.overlap_rgb_mae.toFixed(3)}</h2><article>${[r.original, r.compiled, r.difference].map(src => `<img src="${src}">`).join('')}</article>`).join('');
  await writeFile(path.join(output, 'contact-sheet.html'), html);
  await page.setContent(html); await page.screenshot({ path: path.join(output, 'contact-sheet.png'), fullPage: true });
  const metrics = results.map(({ name, mask_iou, overlap_rgb_mae, program, pixels_per_mm }) => ({ name, mask_iou, overlap_rgb_mae, pixels_per_mm, program_sha256: program.content_sha256 }));
  await writeFile(path.join(output, 'metrics.json'), JSON.stringify(metrics, null, 2) + '\n');
  console.log(JSON.stringify(metrics, null, 2));
  if (metrics.some(result => result.mask_iou < 0.98 || result.overlap_rgb_mae > 2)) throw new Error('artwork fidelity gate failed');
} finally {
  if (browser) await browser.close();
  server.kill('SIGTERM');
  if (server.exitCode === null) await new Promise(resolve => server.once('exit', resolve));
}
