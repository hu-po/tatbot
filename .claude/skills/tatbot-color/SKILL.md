---
name: tatbot-color
description: Colour theory for tatbot's physical ink palette. Use when choosing which inks to load for a design, writing a colour brief or image prompt, planning tints and two-ink hatch mixes, or judging a predicted colour render on paper, silicone or skin.
---

# Colour for a robot with five caps

Tatbot draws colour with real ink, one cartridge at a time. Colour is a physical, measured and
budgeted quantity, not a word in a prompt. Decide it with the tools, then write the prompt.

## The constraints that shape every palette

- **At most five inks per load.** The palette has six caps, and a one-cartridge piece keeps one cap of
  water for rinses. Choose the inks before generating anything.
- **Line costs time.** A colour ink carries about 60 mm of line per dip, black about 130 mm. A dip gap
  costs about 48 s, a rinse about 40 s, and each stroke about 3 s. Large filled areas and many inks
  are what make a piece long.
- **No blending.** The machine cannot make a smooth gradient. Tone comes from line pitch (a tint), from
  two inks hatched over each other (a mix), or from black laid over a colour (a shade).
- **Evidence is uneven.** `config/inks.yaml` records each ink's appearance evidence: nominal <
  vendor-photo < photo < swatch. Every `tatbot art` output names the weakest evidence it used. Until a
  drawn swatch card is measured, rank designs against each other and never promise an exact colour.

## Tools

```bash
tatbot art swatch --inks golden_yellow,bright_red,light_green,light_purple,true_black --out /tmp/sw
tatbot art separate flash.jpg --inks golden_yellow,bright_red,light_green,blue_sky,true_black \
    --width-mm 60 --out /tmp/sep --app /opt/drawingbotv3
tatbot art render program.json --substrate skin-3 --out /tmp/render
```

- `swatch` lists everything a load can draw: tints at 1/3, 2/3 and full coverage, plus every two-ink
  overlap, each with its hatch angles and mm of line per cm². Its `swatches.png` is also the picture to
  give an image generator as a palette reference.
- `separate` turns a flat-colour image into the load's colours and per-ink DBV3 layers, then compiles
  and renders it. Its report lists the colours used, the minutes and a fidelity score.
- `render` predicts a compiled program on `paper`, `silicone` or `skin-1`…`skin-6`, with fading
  toward each dip, plus minutes, dips and rinses. Add `--spread-mm 0.3` to preview a healed or aged
  look on skin.

## The owned inks

| Role | Catalogue ids |
| --- | --- |
| Black: outlines, shadows, value anchor | `true_black`, `nighthawk_black`, `kuro_sumi_black` |
| Warm | `bright_red`, `rose_pink`, `bright_orange`, `golden_yellow`, `lemon_yellow`, `brown` |
| Cool | `lime_green`, `light_green`, `blue_sky`, `light_purple` |
| Opaque (covers what is under it) | `silver`, `snow_white` |

`fuchsia` and `true_blue` are catalogued with nominal primaries, and which bottles they are is
unconfirmed. Check `tatbot art swatch` output before relying on them. The set's sky blue bottle is
`blue_sky` (vendor-photo colour); `true_blue` may be the same bottle.

## Choosing a load

Work in this order. Each step can veto the next.

1. **Value first.** Squint at the source, or blur it, and count value groups. A readable tattoo has
   three or four: paper or skin, a light colour, a mid colour, black. Two colours of the same value
   read as one, however different their hues. Black carries the value contrast, so keep it.
2. **Pick the hues from what is achievable.** Choose a harmony in OKLCh hue (`tatbot_art.color.oklch`)
   among the owned inks and their mixes, not among imagined colours:

   | Harmony | Load (plus black) | Notes |
   | --- | --- | --- |
   | Traditional primaries | `bright_red`, `golden_yellow`, `light_green`, `blue_sky` | The flash standard. Black outline everything. |
   | Complementary, warm-cool | `bright_orange`, `light_purple` | Maximum pop. Keep one dominant. |
   | Complementary, red-green | `bright_red`, `light_green` | Classic, but strong at equal area. Use 60-30-10. |
   | Analogous warm | `lemon_yellow`, `golden_yellow`, `bright_orange`, `bright_red` | Sunsets, fire, poppies. Needs black for value. |
   | Analogous cool | `lime_green`, `light_green`, `light_purple` | Water, foliage, night. |
   | Earth | `brown`, `golden_yellow`, `bright_orange` | Animals, wood, leather. Reads soft on skin, so outline it. |

3. **Proportion: 60-30-10.** A dominant ink, a secondary and an accent. Equal areas of complementary
   colours vibrate and look muddy.
4. **Mixes before more inks.** A sixth colour usually exists as a mix of two loaded inks:
   - red over yellow gives orange;
   - yellow over sky blue gives green;
   - black at 1/3 coverage over a colour gives its shade.
   Mixes cost line but no extra cap or rinse. At most two inks overlap in one region, and each ink
   keeps its own hatch angle (the swatch tool assigns them, at least 15 degrees apart) to avoid moiré.
5. **Opaque inks go last.** White and silver cover what is under them, so they serve as highlights over
   colour. White is nearly invisible on the pale silicone tile and on light skin. Never use white as a
   line.
6. **Price it.** Compile, render and read the minutes. If a piece is over budget, cut first a fill
   area, then a mix, then an ink, and only then shrink it.

## Skin

- Render on the skin tone the piece is for. On deeper tones (`skin-4` to `skin-6`), value contrast
  matters more than hue. Yellow, pastels and white lose almost all contrast. Black outlines with
  saturated red, green or violet hold.
- On light skin (`skin-1` to `skin-2`), yellow next to white or paper-coloured areas disappears. Give
  yellow a black or red edge.
- Longevity: black lasts. Yellows, whites and pastels fade first. Fine lines spread with time, so keep
  gaps of at least 1 mm between lines meant to stay separate, and check `--spread-mm`.

## Replacing what the machine cannot do

| Wanted | Do instead |
| --- | --- |
| Smooth gradient | Two or three tint steps of one ink, or a whip ramp: hatch pitch widening across the region |
| Soft shadow | A black tint (1/3 coverage) over the colour, or solid black shapes (traditional) |
| Many similar hues | One hue in tints, or a mix, before another cap |
| Stipple shading | Only for small accents. Each dot is a stroke, about 3 s, so 1,000 dots is about 50 min |
| Fine colour detail | Black linework. Colour fills need regions at least two line widths across |

## Before handing a design on

- The load is five inks or fewer, in order lightest first, with black and opaque inks last.
- The value structure reads when blurred.
- `tatbot art render` was run on the target substrate. The minutes fit the budget, and the
  weakest evidence source is stated in the hand-off.
