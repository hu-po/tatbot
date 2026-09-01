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
| click a design, then click the body | place it — a ghost follows your pointer until you click |
| type a subject under **Generate a design** | have a new flash design drawn for you (a few seconds; it queues on a shared GPU) |
| **A** / **D** | rotate on the skin (hold **Shift** for bigger steps) |
| **W** / **S** | make it larger / smaller |
| **Enter** or ✓ Accept | keep it and pick the next one |
| **Delete** or ✕ Discard | throw it away |
| click a placed tattoo | select it again — sliders and mirror are in the sidebar |
| ♂ / ♀ and the colour dots | switch body, change skin tone (natural or otherwise) |
| drag the background | orbit; scroll to zoom |
| download JSON / load JSON | save your layout and bring it back later |

Placing, tracing and saving happen in your browser; the JSON file is written
to your own computer. Only the subject you type is sent to the generator,
which draws with the open-weights [Z-Image-Turbo](https://huggingface.co/Tongyi-MAI/Z-Image-Turbo)
model and keeps nothing.

## What is in a saved layout

Each tattoo is stored as a **surface anchor** — a spot on the body mesh — plus
a rotation and a size in millimetres, and the file records exactly which body
it was made on. That is deliberately not a picture: it is the information a
robot would need to draw the same design in the same place. A layout is a
sketch, not an instruction. Nothing here operates a machine or tattoos a
person.

## Credits

- Bodies: the stylized male and female base meshes from Blender Studio's
  [Human Base Meshes](https://www.blender.org/download/demo-files/) bundle,
  released under CC0. They are deliberately not anyone in particular.
- Rendering: [three.js](https://threejs.org/) via React Three Fiber.
- Designs: a handful of placeholder line drawings; the point is the placement,
  not the flash.

Made by [Tatbot](https://tatbot.ai) · code on
[GitHub](https://github.com/hu-po/tatbot) · hello@tatbot.ai
