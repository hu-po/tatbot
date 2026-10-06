// Materialize the ignored scenarios required by the normal web tests.
import { existsSync, readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { resolve } from "node:path";
import { spawnSync } from "node:child_process";

const repo = fileURLToPath(new URL("../../../", import.meta.url));
const output = resolve(repo, "web/inkmap/public/showcase");
const manifestPath = resolve(output, "manifest.json");
const original = readFileSync(manifestPath);
const manifest = JSON.parse(original);
const catalog = JSON.parse(readFileSync(resolve(repo, "config/inkmap/body-poses.json")));
const catalogDigest = JSON.parse(readFileSync(resolve(repo, "config/inkmap/body-poses.digest.json")));
const missing = manifest.slides.filter((slide) => {
  const path = resolve(output, slide.scenario);
  if (!existsSync(path)) return true;
  const scenario = JSON.parse(readFileSync(path));
  const artwork = JSON.parse(readFileSync(resolve(repo, `web/inkmap/public/designs/${slide.artwork_id}/artwork.json`)));
  return scenario.body.pose_asset_sha256 !== catalog.pose_asset.sha256
    || scenario.pose.catalog_sha256 !== catalogDigest.sha256
    || scenario.design.id !== slide.artwork_id
    || scenario.design.sha256 !== artwork.source_sha256
    || scenario.program_binding?.bundle?.artworks?.[slide.artwork_id]?.content_sha256 !== artwork.content_sha256;
});
if (missing.length) {
  const python = resolve(repo, "python/tatbot_sim/.venv/bin/python");
  if (!existsSync(python)) {
    throw new Error("Showcase scenarios are missing. From the repository root run: uv sync --project python/tatbot_sim; then rerun npm test.");
  }
  console.log(`Preparing ${missing.length} missing or stale showcase scenario(s) from the retained manifest.`);
  const result = spawnSync(python, ["-m", "tatbot_sim.inkmap.showcase", "--output-dir", output,
    "--install", "--scenarios-only"], {
    cwd: repo, stdio: "inherit",
    env: { ...process.env, TATBOT_REPO: repo, PYTHONPATH: resolve(repo, "python/tatbot_sim/src") },
  });
  if (result.error) throw result.error;
  if (result.status !== 0) throw new Error(`Showcase preparation failed (${result.status}); see compiler output above.`);
  if (!readFileSync(manifestPath).equals(original)) throw new Error("Showcase preparation changed the tracked manifest.");
  for (const slide of manifest.slides) {
    if (!existsSync(resolve(output, slide.scenario))) throw new Error(`Showcase compiler omitted ${slide.scenario}`);
  }
}
