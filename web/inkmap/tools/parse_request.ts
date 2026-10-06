#!/usr/bin/env node
import { intentFromSite } from "../src/core/inklang/index.ts";
import { parseSentence, stylePrompt } from "../src/core/lang.ts";

const sentence = process.argv.slice(2).join(" ").trim();
if (sentence.length === 0) {
  process.stdout.write(JSON.stringify({ ok: false, code: "empty_prompt", error: "prompt is empty" }));
  process.exitCode = 2;
} else {
  try {
    const program = parseSentence(sentence);
    process.stdout.write(JSON.stringify({
      ok: true,
      program,
      placement_intent: intentFromSite(sentence, program.site),
      style_prompt: stylePrompt(program) ?? null,
    }));
  } catch (error) {
    process.stdout.write(JSON.stringify({
      ok: false,
      code: "semantic_parse",
      error: error instanceof Error ? error.message : String(error),
    }));
    process.exitCode = 2;
  }
}
