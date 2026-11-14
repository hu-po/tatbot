---
name: tatbot-style-linocut
description: Linocut and woodcut (relief print) style for tatbot. Black masses with carved white lines plus one or two flat colour blocks, mapped to the owned inks and DBV3 recipes, with sizes, a tested Flow prompt and its known failure (Flow returns photographs of prints). Use when a design is asked for as linocut, woodcut, block print or relief print, or when high-contrast carved texture is wanted.
---

# Linocut and woodcut

A relief print is a black mass with white lines carved out of it, and usually one flat colour block
printed under it, sometimes a little off-register. On a robot the white lines are free (negative
space), but the black mass is area, and area costs line and dips. The art is in letting the hatch
itself read as the carving. Read `tatbot-color` and `tatbot-dbv3-recipes`.

## The look

- **Black.** Big black shapes describing form. White gouge lines cut into them follow the form: fur,
  water, wood grain, sky.
- **Colour.** One accent block, flat, sitting under the black, so the black overlaps it. A slight
  offset is part of the charm.
- **Values.** None in between: no grey, no gradient, no halftone. Tone comes from the density of the
  gouge lines.
- **Motifs.** Ships, animals, cityscapes, machines, waves, botanicals. San Francisco subjects in the
  2026-10-30 study: a cable car on a hill, a sea lion on a pier post, a shipyard crane.

## Palette on the owned inks

Black plus one accent, at most two:

| Accent | Mood |
| --- | --- |
| `bright_red` | Classic, urgent: vehicles, hearts |
| `bright_orange` | Warm: sunsets, animals |
| `blue_sky` | Water and sky |
| `golden_yellow` | Light, sun, gold |

A second accent works only if it touches a different area: sky blue for sky with orange for the
subject. The colour block is drawn first, the black second.

## Translating it to strokes

- **Let the hatch be the gouge.** Fill the black mass with Sketch Lines at default density (75). The
  hatching reads as carved texture instead of fighting it.
  - A dense linocut ship held at Line Density 50 (41.75 × 86 mm).
  - At 30 a detailed drawing falls apart.
- **The mass edge is the outline.** Do not add a separate ECS outline layer around black masses. It
  doubles contact for no visible gain. Outline only where colour meets paper with no black.
- **White lines must be at least 1 mm wide at the drawn size,** or ink spread fills them. On skin,
  render with `--spread-mm 0.2` to check.
- **The colour block** is a solid fill (ECS, edges off) or an angled hatch at the ink's screen angle,
  eroded so the black overlaps its edge.
- **9M swaths** suit large black areas only when the design is 60 mm or more and its white lines run
  vertically in the print. Otherwise the swath erases them.

## Sizes

| Size | What survives |
| --- | --- |
| 35 mm | A silhouette with 3–5 gouge lines; the texture is gone |
| 60 mm | Only if the gouges are bold. A Flow linocut sea lion at 60 mm lost almost all of its gouges: they were about 0.2 mm, under one line. It separated as an 83% black mass of 8 m of line, 89 dips, about 3 h. Ask for "few bold gouge lines at least 1 mm wide" |
| 85–100 mm | Full texture: the long-format piece |

## Flow prompt (tested 2026-10-30, Nano Banana Pro, 1:1)

> Linocut relief print of {subject}, designed as a tattoo. A carved black ink mass with crisp white gouge
> lines that describe form and texture, plus one flat second colour block in {accent} printed under the
> black. Only two inks: black and {accent}. Bold and high contrast, no gradients, no grey, no halftone
> dots. Flat digital artwork, not a photograph: pure white background, no paper texture, no shadow, no
> frame, no signs, no text, no border. Centered, isolated.

Without the "Flat digital artwork, not a photograph" sentence, all nine first-round images came back as
**photographs of prints**: off-white paper, a soft shadow, once a wooden frame edge, and signage text
on the cable car. On-palette scores were 56–80%, pulled down mostly by the paper tint. With it, the
same sea lion came back flat on pure white three times out of three (on-palette 68% to 74%, median
error 7.7 to 5.1 ΔE00).

## Failure modes

| Failure | Fix |
| --- | --- |
| Paper tint, shadow or frame | Crop to the artwork, set pixels with L* above about 80 to white, then snap to the palette. Or re-roll with the fixed prompt |
| Gouge texture finer than 1 mm at the drawn size | Draw larger, or ask for "few bold gouge lines" |
| The colour block fills the whole background (sky) | A big area means many dips. Crop it to a shape or drop it |
| Lettering on signs | Remove it in separation |
