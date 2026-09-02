import { chromium } from "playwright";
import { mkdirSync } from "node:fs";

const out = new URL("./out/", import.meta.url).pathname;
mkdirSync(out, { recursive: true });
const browser = await chromium.launch({
  channel: "chrome",
  headless: true,
  args: ["--ignore-gpu-blocklist", "--use-gl=swiftshader"],
});
const page = await (await browser.newContext({ viewport: { width: 1440, height: 1000 }, deviceScaleFactor: 1 })).newPage();
await page.goto(process.argv[2] ?? "http://127.0.0.1:4180/?showcase=1");
await page.waitForFunction(() => {
  const state = window.__inkmap?.getState();
  return state?.body && state.showcaseScenario && state.body.poseId === state.showcaseScenario.pose.id;
}, null, { timeout: 60_000 });
await page.waitForTimeout(900);
if (await page.locator(".scenario-card").count() !== 5) throw new Error("showcase must expose five scenarios");
await page.screenshot({ path: `${out}/procedural-body-showcase.png`, fullPage: true });
await page.locator(".scenario-card").last().click();
await page.waitForFunction(() => window.__inkmap?.getState().body?.poseId === "reclined-right-arm-supported", null, { timeout: 60_000 });
await page.waitForTimeout(900);
await page.screenshot({ path: `${out}/procedural-body-supported-arm.png`, fullPage: true });
await page.getByRole("button", { name: "focus tattoo site" }).click();
await page.waitForTimeout(700);
await page.screenshot({ path: `${out}/procedural-body-toolpath-detail.png`, fullPage: true });
const browserErrors = await page.evaluate(() => (window.__logs ?? []).filter((line) => line.startsWith("error:") || line.startsWith("uncaught:") || line.startsWith("unhandled:")));
if (browserErrors.length) throw new Error(browserErrors.join("\n"));
await browser.close();
console.log(`wrote ${out}procedural-body-showcase.png`);
console.log(`wrote ${out}procedural-body-supported-arm.png`);
console.log(`wrote ${out}procedural-body-toolpath-detail.png`);
