import { test } from "node:test";
import assert from "node:assert/strict";
import {
  INKLANG_VERSION, SITES, ZONES, STYLES, TECHNIQUES, COLORS,
  parseSentence, realize, validateProgram, isZone,
  type TattooProgram,
} from "../src/core/lang.ts";

test("the lexicon has the 59 leaf sites of lexicon v0.3", () => {
  assert.equal(Object.keys(SITES).length, 59);
  assert.equal(INKLANG_VERSION, "0.3");
});

test("every site is well-formed and every declared aspect is a known aspect", () => {
  const aspectIds = new Set(["inner", "outer", "front", "back", "side", "top"]);
  for (const [id, s] of Object.entries(SITES)) {
    assert.match(id, /^[a-z][a-z_]*$/, `${id}: ids are snake_case`);
    assert.ok(s.name.length > 0 && s.group.length > 0 && s.ncic_site.length > 0, `${id}: name/group/ncic_site`);
    assert.ok(["sided", "midline", "any"].includes(s.laterality), `${id}: laterality`);
    assert.ok(["flat", "wrap", "crease"].includes(s.geometry), `${id}: geometry`);
    for (const a of s.aspects ?? []) assert.ok(aspectIds.has(a), `${id}: unknown aspect ${a}`);
  }
  for (const [id, z] of Object.entries(ZONES)) {
    for (const m of z.members) assert.ok(m in SITES, `zone ${id}: unknown member ${m}`);
  }
});

test("no phrase resolves to two different things", () => {
  const seen = new Map<string, string>();
  const claim = (phrase: string, owner: string) => {
    const k = phrase.toLowerCase().replace(/-/g, " ").trim();
    const prev = seen.get(k);
    assert.ok(prev === undefined || prev === owner, `"${k}" claimed by both ${prev} and ${owner}`);
    seen.set(k, owner);
  };
  for (const [id, s] of Object.entries(SITES)) for (const p of [id, s.name, ...(s.aliases ?? [])]) claim(p, `site:${id}`);
  for (const [id, z] of Object.entries(ZONES)) for (const p of [id, z.name, ...(z.aliases ?? [])]) claim(p, `zone:${id}`);
  for (const table of [STYLES, TECHNIQUES, COLORS]) {
    for (const [id, e] of Object.entries(table)) for (const p of [id, e.name, ...e.aliases]) claim(p, `term:${id}`);
  }
});

test("the two founding sentences parse exactly", () => {
  const p1 = parseSentence("A fine line octopus on the left knee ditch");
  assert.deepEqual(p1, {
    inklang: "0.3", motif: "octopus", style: "fine-line", secondary: [],
    technique: null, color: null,
    site: { id: "knee_ditch", laterality: "left", aspect: null, level: null },
  } satisfies TattooProgram);

  const p2 = parseSentence("a watercolor skull with tribal themes on the left inside biceps");
  assert.deepEqual(p2, {
    inklang: "0.3", motif: "skull", style: "watercolor", secondary: ["tribal"],
    technique: null, color: null,
    site: { id: "bicep", laterality: "left", aspect: "inner", level: null },
  } satisfies TattooProgram);
});

test("realize is canonical and parse inverts it", () => {
  const programs: TattooProgram[] = [
    { inklang: "0.3", motif: "octopus", style: "fine-line", secondary: [], technique: null, color: null, site: { id: "knee_ditch", laterality: "left", aspect: null, level: null } },
    { inklang: "0.3", motif: "skull", style: "watercolor", secondary: ["tribal"], technique: null, color: null, site: { id: "bicep", laterality: "left", aspect: "inner", level: null } },
    { inklang: "0.3", motif: "snake wrapped around a dagger", style: null, secondary: [], technique: "dotwork", color: "black-and-grey", site: { id: "forearm", laterality: "right", aspect: "outer", level: null } },
    { inklang: "0.3", motif: "rose", style: "american-traditional", secondary: ["geometric", "ornamental"], technique: null, color: "full-color", site: { id: "sternum", laterality: null, aspect: null, level: null } },
    { inklang: "0.3", motif: "koi fish", style: "japanese", secondary: [], technique: null, color: null, site: { id: "full_sleeve", laterality: "right", aspect: null, level: null } },
  ];
  for (const p of programs) {
    const s = realize(p);
    assert.deepEqual(parseSentence(s), p, `roundtrip failed for "${s}"`);
    assert.equal(realize(parseSentence(s)), s, `canonical form unstable for "${s}"`);
  }
  assert.equal(realize(programs[0]), "a fine line octopus on the left knee ditch");
  assert.equal(realize(programs[1]), "a watercolor skull with tribal themes on the left inner bicep");
});

test("aliases, compounds, and precedence", () => {
  // Colloquial site aliases land on the canonical id.
  assert.equal(parseSentence("a rose on the back of the knee").site.id, "knee_ditch");
  assert.equal(parseSentence("a rose on the popliteal fossa").site.id, "knee_ditch");
  assert.equal(parseSentence("a rose on the left antecubital fossa").site.id, "elbow_ditch");
  // Compound aliases carry an implied aspect.
  assert.deepEqual(parseSentence("a moth on the throat").site, { id: "throat", laterality: null, aspect: null, level: null });
  assert.deepEqual(parseSentence("a moth on the nape of the neck").site, { id: "nape", laterality: null, aspect: null, level: null });
  // A laterality word that actually opens a site alias is given back.
  assert.deepEqual(parseSentence("a sun on the middle of the back").site, { id: "mid_back", laterality: null, aspect: null, level: null });
  // Longest match: "watercolor wash" is a color, bare "watercolor" a style.
  const wash = parseSentence("a watercolor wash hummingbird on the right ribs");
  assert.equal(wash.color, "watercolor-wash");
  assert.equal(wash.style, null);
  // "with … themes" only strips when the clause is really styles.
  assert.equal(parseSentence("a skull with roses on the forearm").motif, "skull with roses");
  const mixed = parseSentence("a skull with roses and thorns on the forearm");
  assert.equal(mixed.motif, "skull with roses and thorns");
  // Motifs containing "on the" split on the last occurrence.
  assert.equal(parseSentence("a skull on the throne on the left forearm").motif, "skull on the throne");
  // Zones parse and are marked.
  assert.ok(isZone(parseSentence("a japanese dragon on the right leg sleeve").site.id));
});

test("lies are rejected: laterality and aspect rules, unknown terms", () => {
  assert.throws(() => parseSentence("a rose on the left sternum"), /midline/);
  assert.throws(() => parseSentence("a rose on the inner sternum"), /not allowed|aspect/);
  assert.throws(() => parseSentence("a rose on the flux capacitor"), /unknown site/);
  assert.throws(() => parseSentence("a rose"), /on the/);
  assert.throws(() => parseSentence("a fine line on the wrist"), /no motif/);
  const bad: TattooProgram = { inklang: "0.3", motif: "rose", style: "vaporwave" as string, secondary: [], technique: null, color: null, site: { id: "wrist", laterality: null, aspect: null, level: null } };
  assert.throws(() => validateProgram(bad), /unknown style/);
  assert.throws(() => realize({ ...bad, style: null, site: { id: "knee_ditch", laterality: "center", aspect: null, level: null } }), /sided/);
});

test("midline sites realize without laterality; sided zones carry it", () => {
  assert.equal(
    realize({ inklang: "0.3", motif: "serpent", style: null, secondary: [], technique: null, color: null, site: { id: "spine", laterality: null, aspect: null, level: null } }),
    "a serpent on the spine",
  );
  assert.equal(
    realize({ inklang: "0.3", motif: "wave", style: null, secondary: [], technique: null, color: null, site: { id: "anklet", laterality: "left", aspect: null, level: null } }),
    "a wave on the left anklet",
  );
});

test("levels: upper/lower/mid combine with aspects and roundtrip", () => {
  const p = parseSentence("an owl on the left upper inner forearm");
  assert.deepEqual(p.site, { id: "forearm", laterality: "left", aspect: "inner", level: "upper" });
  assert.equal(realize(p), "an owl on the left upper inner forearm");
  assert.deepEqual(parseSentence("a star on the lower back").site, { id: "lower_back", laterality: null, aspect: null, level: null });
  assert.deepEqual(parseSentence("a star on the lower spine").site, { id: "spine", laterality: null, aspect: null, level: "lower" });
  assert.throws(() => parseSentence("a star on the upper lower thigh"), /conflicting levels/);
  assert.throws(() => realize({ inklang: "0.3", motif: "wave", style: null, secondary: [], technique: null, color: null, site: { id: "anklet", laterality: "left", aspect: null, level: "upper" } }), /zones take no level/);
});

test("v0.3 leaves: forehead, throat, nape are real sites now", () => {
  const p = parseSentence("a classic american third eye on the forehead");
  assert.deepEqual(p, {
    inklang: "0.3", motif: "third eye", style: "american-traditional", secondary: [],
    technique: null, color: null,
    site: { id: "forehead", laterality: null, aspect: null, level: null },
  } satisfies TattooProgram);
  assert.equal(realize(p), "an american traditional third eye on the forehead");
  assert.equal(realize(parseSentence("a moth on the throat")), "a moth on the throat");
  assert.equal(parseSentence("a spider on the brow").site.id, "eyebrow");
  assert.equal(parseSentence("a sun on the tailbone").site.id, "sacrum");
  assert.equal(parseSentence("a wave on the achilles").site.id, "achilles");
  assert.deepEqual(parseSentence("a tiger on the hamstring").site, { id: "thigh", laterality: null, aspect: "back", level: null });
});

test("relative phrases: measures, relations, between", () => {
  const p = parseSentence("a rose two inches below the left collarbone");
  assert.deepEqual(p.site, {
    id: "collarbone", laterality: "left", aspect: null, level: null,
    rel: { kind: "below", offset_m: 2 * 0.0254, render: "two inches" },
  });
  assert.equal(realize(p), "a rose two inches below the left collarbone");
  assert.equal(realize(parseSentence(realize(p))), realize(p));

  const j = parseSentence("a star just above the navel");
  assert.equal(j.site.rel!.offset_m, 0.03);
  assert.equal(realize(j), "a star just above the navel");

  const cm = parseSentence("a dot 3 cm behind the left ear");
  assert.ok(Math.abs(cm.site.rel!.offset_m! - 0.03) < 1e-9);

  const b = parseSentence("a moth between the shoulder blades");
  assert.equal(b.site.id, "shoulder_blade");
  assert.deepEqual(b.site.rel, { kind: "between", other: { id: "shoulder_blade", laterality: "right" } });
  assert.equal(realize(b), "a moth between the left shoulder blade and the right shoulder blade");

  const b2 = parseSentence("a sun between the navel and the sternum");
  assert.equal(b2.site.id, "navel");
  assert.equal(b2.site.rel!.other!.id, "sternum");

  // "behind the ear" is a site, not a relation.
  assert.equal(parseSentence("a star behind the ear").site.id, "behind_ear");
  assert.equal(parseSentence("a star behind the left ear").site.id, "behind_ear");
  assert.throws(() => parseSentence("a rose 90 inches below the collarbone"), /offset/);
});

test("aspect-first phrasing: 'outside of the left shin'", () => {
  const p = parseSentence("an illustrative giant squid on the outside of the left shin");
  assert.deepEqual(p.site, { id: "shin", laterality: "left", aspect: "outer", level: null });
  assert.equal(p.style, "illustrative");
  assert.equal(p.motif, "giant squid");
  assert.equal(realize(p), "an illustrative giant squid on the left outer shin");
  assert.equal(realize(parseSentence(realize(p))), realize(p));
  assert.deepEqual(parseSentence("a koi on the inside of the right calf").site, { id: "calf", laterality: "right", aspect: "inner", level: null });
  assert.deepEqual(parseSentence("a sun on the back of the left thigh").site, { id: "thigh", laterality: "left", aspect: "back", level: null });
});
