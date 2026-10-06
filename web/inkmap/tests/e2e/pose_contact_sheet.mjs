import { chromium } from "playwright";
import { mkdirSync } from "node:fs";

const out = new URL("./out/", import.meta.url).pathname;
mkdirSync(out, { recursive: true });
const poses = [
  "supine",
  "prone",
  "reclined-seated",
  "reclined-left-arm-supported",
  "reclined-right-arm-supported",
];
const browser = await chromium.launch({
  channel: "chrome",
  headless: true,
  args: ["--ignore-gpu-blocklist", "--use-gl=swiftshader"],
});
const page = await (await browser.newContext({ viewport: { width: 1000, height: 850 } })).newPage();
await page.goto(process.argv[2] ?? "http://127.0.0.1:4180/");
await page.waitForFunction(() => window.__inkmap?.getState().body, null, { timeout: 60_000 });
for (const poseId of poses) {
  await page.evaluate((pose) => window.__inkmap.getState().setPoseId(pose), poseId);
  await page.waitForFunction((pose) => {
    const state = window.__inkmap?.getState();
    return state?.body?.spec.id === "mhr-soma-v1" && state.body.poseId === pose;
  }, poseId, { timeout: 60_000 });
  await page.waitForTimeout(300);
  await page.screenshot({
    path: `${out}/pose-mhr-soma-v1-${poseId}.png`,
    clip: { x: 300, y: 0, width: 700, height: 850 },
  });
}
const browserErrors = await page.evaluate(() => (window.__logs ?? []).filter((line) => line.startsWith("error:") || line.startsWith("uncaught:") || line.startsWith("unhandled:")));
if (browserErrors.length) throw new Error(browserErrors.join("\n"));
await browser.close();
console.log(`wrote ${poses.length} pose renders to ${out}`);
