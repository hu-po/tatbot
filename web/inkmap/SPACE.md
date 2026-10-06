---
title: Inkmap
emoji: 🐙
colorFrom: gray
colorTo: pink
sdk: static
pinned: false
short_description: Try a tattoo on a 3D body before it touches skin.
---

# Inkmap — try a tattoo on before it's real

Pick a design, click anywhere on the body, and it wraps onto the skin right
there. Turn it, size it, mirror it, move on to the next one. Inkmap is the
sketchpad in front of [Tatbot](https://tatbot.ai), an open-source tattoo robot
being built in public in Austin: the place where a tattoo gets *decided* —
which design, where on the body, how big, which way up — in a form a machine
can read back.

## How to use it

| do this | to |
| --- | --- |
| describe a placement, such as **on the left forearm** | resolve an exact spot with InkLang; choose from buttons if the wording is ambiguous |
| click a design, then click the body | place it — a ghost follows your pointer until you click |
| **A** / **D** | rotate on the skin (hold **Shift** for bigger steps) |
| **W** / **S** | make it larger / smaller |
| **Enter** or ✓ Accept | keep it and pick the next one |
| **Delete** or ✕ Discard | throw it away |
| click a placed tattoo | select it again — sliders and mirror are in the sidebar |
| ♂ / ♀ and the colour dots | switch body, change skin tone (natural or otherwise) |
| drag the background | orbit; scroll to zoom |
| download JSON / load JSON | save your layout and bring it back later |

Placing and saving happen in your browser; the JSON file is written to your
own computer.

InkLang is the placement system behind the description box. It normalizes the
words, combines them with the selected body, and records one exact point on the
body's canonical surface. It does not generate artwork. A phrase like
**forearm** needs a left/right choice and nothing is generated or placed until
you make it. The older “a fine line octopus on the left knee ditch” sentence
also works; its design words and placement clause stay separate internally.

## What is in a saved layout

Each tattoo is stored as a **surface anchor** — a spot on the body mesh — plus
a rotation and a size in millimetres, and the file records exactly which body
it was made on. The original placement description, normalized meaning, and
body-surface resolution travel with that anchor. That is deliberately not a
picture. A layout is a sketch, not a robot instruction, reachability result, or
contact qualification. Nothing here operates a machine or tattoos a person.

## Credits

- Body: a synthetic nominal body derived from
  [SOMA-X v0.3.0](https://github.com/NVlabs/SOMA-X/tree/v0.3.0) (Copyright (c)
  2026 NVIDIA CORPORATION & AFFILIATES) and
  [MHR v1.0.1](https://github.com/facebookresearch/MHR/tree/v1.0.1) (Copyright
  (c) Meta Platforms, Inc. and affiliates), both Apache-2.0. It is not a scan
  of anyone.
- Rendering: [three.js](https://threejs.org/) via React Three Fiber.
- Designs: a handful of placeholder line drawings; the point is the placement,
  not the flash.

Made by [Tatbot](https://tatbot.ai) · code on
[GitHub](https://github.com/hu-po/tatbot) · hello@tatbot.ai
