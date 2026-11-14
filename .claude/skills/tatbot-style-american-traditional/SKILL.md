---
name: tatbot-style-american-traditional
description: American traditional (old school) tattoo flash for tatbot. The look, its palette on the owned inks, how its layers map to DBV3 recipes and tools, time budgets, a tested Flow prompt, and the failure modes. Use when a design is asked for in traditional, old-school, sailor or flash style, or when bold outlined flat-colour work is wanted.
---

# American traditional

Bold will hold. Traditional flash is the style the machine draws most honestly: thick even outlines,
flat colour and solid black shadows, with no blending to fake. Read `tatbot-color` for the palette
rules and `tatbot-dbv3-recipes` for settings.

## The look

- **Outlines.** One heavy, even black outline around every shape. It is the heaviest line in the
  piece.
- **Shadows.** Solid black shapes, not gradients. Black often covers 25–40% of the inked area.
- **Fill.** Flat fills in a few saturated colours. Bare skin or paper is a colour too: leave
  highlights and the subject's light side unfilled.
- **Detail.** Few, large, readable shapes. No detail smaller than about 2 mm at the drawn size.
- **Motifs.** Swallows, roses, daggers, ships, panthers, eagles, snakes, hearts, anchors and banners.
  For a place: its animals, vehicles and flowers drawn as icons. Generate original designs; never copy
  a named artist's flash.

## Palette on the owned inks

| Role | Inks |
| --- | --- |
| Core load | `true_black` (or `nighthawk_black`), `bright_red`, `golden_yellow`, `light_green`, `blue_sky` |
| Warm variant | swap `blue_sky` for `bright_orange`, or `brown` for wood and fur |
| Accent only | `lemon_yellow` for highlights inside golden yellow; `snow_white` highlight dots last (not on pale skin or the tile) |

Use 60-30-10: one dominant colour, usually yellow or red, a secondary, and an accent.

## Layers, tools and recipes

Draw lightest first, black last:

1. **Yellow, then green or blue, then red.** Fill each region with an angled hatch: Sketch Lines with
   start angle = the ink's screen angle, Even weighted. Erode each region by half the line width so
   it stays inside its outline.
2. **Black shadow shapes.** Use a solid fill (ECS, edges and contours off). At 60 mm or more, use 9M
   swaths for large black (Sketch Lines at 90°, pen 3 mm, regions eroded 1.6 mm).
3. **Black outline.** ECS edges-only, last, so it covers the fills' ragged edges.

The outline layer is traced out and back, so its contact length is about twice the edge length. It is
often the single largest time item.

## Budgets

Rules of thumb from the October pieces. Scale them by area, and always compile and render:

| Size (wide) | Inks | Strokes | Dips | Minutes |
| --- | --- | --- | --- | --- |
| 35 mm | black + 1–2 | ≤ 60 | ≤ 15 | about 15–25 |
| 60 mm | black + 2–3 | ≤ 120 | ≤ 40 | about 45–75 |
| 95 mm | black + 2–3 | about 250 | about 80–90 | about 150–200 |

Over budget? Cut in this order: fill area (leave more skin), then a colour, then black shadow area,
and only then shrink.

## Flow prompt (tested 2026-10-30, Nano Banana Pro, 1:1)

> American traditional tattoo flash of {subject}. Bold, even black outlines of a single weight, flat
> solid colour fills with no gradients, no blending and no soft shading; shadows are solid black
> shapes. Use only these five exact colours and no others: #000000 black, #FA5032 red, #FDC949 golden
> yellow, #3B9560 green, #79C7FB sky blue. Large simple shapes with generous gaps between lines, no tiny
> details. Isolated on a plain white background, centered, no text, no lettering, no signs, no border.

Results: flat, outlined and centred, with 72–87% of inked pixels on the requested inks. The hex codes
tightened the hue (median error 4.0 against 6.8 ΔE00 with names alone). Sea lion, cable car and
swallow-with-poppies all read clearly.

## Failure modes

| Failure | Fix |
| --- | --- |
| Grey appears for rocks, metal or shadow | Snap it to black, a black tint or `brown`; never add a grey cap for it |
| Signage or lettering on vehicles despite "no text" | Remove it in separation or re-roll. Lettering is a separate job |
| Whiskers, foam or thin waves come out thinner than two line widths | Merge them into the outline or drop them |
| A colour region narrower than about 2 mm | Becomes outline. Do not give it its own ink |
| Too many outline strokes | Simplify the silhouette, and render to check the outline minutes |
