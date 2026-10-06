#!/usr/bin/env node
// Regenerate the normative InkLang prompt-to-surface corpus.
import { writeFileSync } from "node:fs";
import { pathToFileURL } from "node:url";
import { MODEL_SPEC_ID } from "../src/core/body.ts";
import {
  SITES,
  ZONES,
  realizePlacement,
  type SitePhrase,
} from "../src/core/inklang/index.ts";
import { canonicalJson, resolvePrompt, type ResolveRequest } from "./resolve.ts";

interface CorpusCase {
  id: string;
  request: ResolveRequest;
  expected: Awaited<ReturnType<typeof resolvePrompt>>;
}

function slug(value: string): string {
  return value.replace(/[^a-z0-9]+/g, "-").replace(/^-|-$/g, "");
}

function add(
  requests: { id: string; request: ResolveRequest }[],
  id: string,
  body: string,
  prompt: string,
  policy?: "interactive" | "seeded-v1",
  seed?: number,
): void {
  requests.push({
    id: `${body}:${id}`,
    request: { prompt, ...(policy ? { policy } : {}), ...(seed === undefined ? {} : { seed }) },
  });
}

export async function buildCorpus(): Promise<{ corpus_schema_version: 1; cases: CorpusCase[] }> {
  const requests: { id: string; request: ResolveRequest }[] = [];
  for (const body of [MODEL_SPEC_ID]) {
    for (const [siteId, site] of Object.entries(SITES)) {
      const lateralities = site.laterality === "sided" ? (["left", "right"] as const) : ([null] as const);
      for (const laterality of lateralities) {
        const phrase: SitePhrase = { id: siteId, laterality, aspect: null, level: null };
        add(requests, `leaf:${laterality ?? "center"}:${siteId}`, body, realizePlacement(phrase));
      }
    }

    for (const [zoneId, zone] of Object.entries(ZONES)) {
      const laterality = zone.laterality === "sided" ? "left" : null;
      const phrase: SitePhrase = { id: zoneId, laterality, aspect: null, level: null };
      const prompt = realizePlacement(phrase);
      add(requests, `zone:interactive:${zoneId}`, body, prompt);
      add(requests, `zone:seeded:${zoneId}`, body, prompt, "seeded-v1", 903);
    }

    const aspects: SitePhrase[] = [
      { id: "forearm", laterality: "left", aspect: "inner", level: null },
      { id: "forearm", laterality: "right", aspect: "outer", level: null },
      { id: "shoulder_cap", laterality: "left", aspect: "front", level: null },
      { id: "shoulder_cap", laterality: "right", aspect: "back", level: null },
      { id: "neck", laterality: null, aspect: "side", level: null },
      { id: "hand", laterality: "left", aspect: "top", level: null },
    ];
    for (const phrase of aspects) add(requests, `aspect:${phrase.aspect}:${phrase.id}`, body, realizePlacement(phrase));

    for (const level of ["upper", "mid", "lower"] as const) {
      const phrase: SitePhrase = { id: "forearm", laterality: "left", aspect: null, level };
      add(requests, `level:${level}`, body, realizePlacement(phrase));
    }

    const relatives: SitePhrase[] = [
      { id: "collarbone", laterality: "left", aspect: null, level: null, relation: { kind: "above", offset_m: 0.03, render: "3 cm" } },
      { id: "collarbone", laterality: "left", aspect: null, level: null, relation: { kind: "below", offset_m: 0.03, render: "3 cm" } },
      { id: "ear", laterality: "left", aspect: null, level: null, relation: { kind: "behind", offset_m: 0.03, render: "3 cm" } },
      { id: "ear", laterality: "right", aspect: null, level: null, relation: { kind: "in_front", offset_m: 0.03, render: "3 cm" } },
      { id: "collarbone", laterality: "left", aspect: null, level: null, relation: { kind: "beside", offset_m: 0.03, render: "3 cm" } },
      { id: "shoulder_blade", laterality: "left", aspect: null, level: null, relation: { kind: "between", other: { id: "shoulder_blade", laterality: "right" } } },
    ];
    for (const phrase of relatives) {
      add(requests, `relation:${phrase.relation!.kind}`, body, realizePlacement(phrase));
    }

    for (const prompt of [
      "popliteal fossa",
      "outside of the left shin",
      "left shoulder blade",
      "right upper inner arm",
      "center chest",
      "nape of the neck",
      "left love handle",
      "right kneecap",
    ]) add(requests, `alias:${slug(prompt)}`, body, prompt);

    for (const prompt of ["forearm", "arm", "leg", "beside the left collarbone"]) {
      add(requests, `ambiguity:${slug(prompt)}`, body, prompt);
    }
    for (const prompt of ["flux capacitor", "left sternum", "lower outer leg"]) {
      add(requests, `rejection:${slug(prompt)}`, body, prompt);
    }
    add(
      requests,
      "legacy-full-sentence",
      body,
      "a fine line octopus on the left knee ditch",
    );
  }

  const cases: CorpusCase[] = [];
  for (const item of requests) {
    cases.push({ ...item, expected: await resolvePrompt(item.request) });
  }
  return { corpus_schema_version: 1, cases };
}

async function main(): Promise<void> {
  const output = new URL("../../../config/inkmap/examples/inklang/corpus-v1.json", import.meta.url);
  const corpus = await buildCorpus();
  writeFileSync(output, `${canonicalJson(corpus)}\n`);
  process.stderr.write(`wrote ${corpus.cases.length} InkLang corpus cases\n`);
}

if (process.argv[1] && import.meta.url === pathToFileURL(process.argv[1]).href) await main();
