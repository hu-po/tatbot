/** Rasterize original/compiled artwork with the same browser SVG renderer. */
import { readFileSync } from "node:fs";
import { chromium } from "playwright";
const jobs = JSON.parse(readFileSync(process.argv[2], "utf8"));
const browser = await chromium.launch({ headless: true });
try {
  const page = await browser.newPage();
  for (const job of jobs) {
    const svg = readFileSync(job.svg, "utf8");
    await page.setViewportSize({ width: job.width, height: job.height });
    await page.setContent('<style>html,body{margin:0;background:transparent}img{display:block;width:100vw;height:100vh}</style><img>');
    await page.locator("img").evaluate((img, text) => { img.src = `data:image/svg+xml;base64,${text}`; }, Buffer.from(svg).toString("base64"));
    await page.locator("img").evaluate(img => img.decode());
    await page.screenshot({ path: job.png, omitBackground: true });
  }
} finally { await browser.close(); }
