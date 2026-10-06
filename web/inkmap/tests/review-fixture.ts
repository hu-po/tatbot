import { readFileSync } from "node:fs";
import { acquiredArtwork } from "./acquired-artwork.ts";
import { canonicalDigest } from "../src/core/human-representation/schema.ts";
import type { StudyReview } from "../src/core/study-review.ts";

export async function fixtureReview(): Promise<StudyReview> {
  const art = await acquiredArtwork();
  const recipe = JSON.parse(readFileSync(new URL("../public/designs/dbv3-orbit/recipe/recipe.json", import.meta.url), "utf8"));
  const bundle = { schema: "tatbot.artwork-review/1" as const, content_sha256: "", name: "Acquired artwork comparison", study_sha256: "a".repeat(64),
    artworks: { [art.content_sha256]: art }, entries: [0, 1].map(index => ({
      id: `orbit-${index}`, source_case: "dbv3-orbit", source_split: "train" as const, artwork_sha256: art.content_sha256,
      recipe: { pfm: recipe.variant.pfm, settings: recipe.effective.settings },
      preparation: { program_sha256: String(index).repeat(64), speed_m_s: .0035 + index * .0005, identity: {},
        tool: { id: "fixture-tool", line_width_m: null, line_width_status: "unknown" as const },
        stats: { paths: 1, strokes: 1, contact_m: .042, travel_m: 0, notes: ["Tool width is unknown"],
          time_estimate: { modeled_s: 20 - index, scope: "Fixture estimate; excludes setup", unknown_operations: {} } } }
    })) };
  bundle.content_sha256 = await canonicalDigest(bundle);
  return bundle;
}
