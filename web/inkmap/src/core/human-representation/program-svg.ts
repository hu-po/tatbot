/** Deterministic review image of TattooProgram/1. Filled regions have no
 * outline: width_m is deposition planning intent, not an extra paint border.
 */
import { validateTattooProgram, type TattooProgram } from "./tattoo-program.ts";

function n(value: number): string { return String(Number(value.toPrecision(14))); }

/** A placed preview changes geometry but retains metric conversion widths.
 * Omitting size_m produces the canonical, content-addressed artwork preview.
 */
export async function tattooProgramToSvg(value: TattooProgram, size_m?: readonly [number, number]): Promise<string> {
  return renderTattooProgramSvg(await validateTattooProgram(value), size_m);
}

/** Render already-validated frozen geometry; callers never parse this back into artwork. */
export function renderTattooProgramSvg(program: TattooProgram, size_m?: readonly [number, number]): string {
  const [width, height] = size_m ?? [program.canvas_m.width, program.canvas_m.height];
  if (![width, height].every(value => Number.isFinite(value) && value > 0)) throw new Error("preview size must be positive finite metres");
  const sx = width / program.canvas_m.width, sy = height / program.canvas_m.height;
  const point = ([x, y]: number[]) => [n(x * sx), n(height - y * sy)];
  const inks = new Map(program.inks.map(ink => [ink.id, ink.color_srgb]));
  const fragments: string[] = [];
  for (const layer of program.layers) {
    const rgb = inks.get(layer.ink_id)!;
    const color = `rgb(${rgb.map(channel => n(channel * 255)).join(",")})`;
    const filled: string[] = [];
    for (const element of layer.elements) {
      const stroke = `stroke="${color}" stroke-width="${n(element.width_m)}" stroke-linecap="round" stroke-linejoin="round"`;
      if (element.kind === "cubic_bezier") {
        const points = element.control_points_m!.map(p => point(p).join(" "));
        fragments.push(`<path d="M${points[0]} C${points.slice(1).join(" ")}" fill="none" ${stroke}/>`);
      } else if (element.kind === "dots" || element.kind === "stipple") {
        for (const p of element.points_m!) {
          const [x, y] = point(p);
          fragments.push(`<circle cx="${x}" cy="${y}" r="${n(element.width_m / 2)}" fill="${color}"/>`);
        }
      } else {
        if (element.fill) {
          const polygon = [...element.points_m!];
          const area = polygon.reduce((sum, [x, y], i) => {
            const next = polygon[(i + 1) % polygon.length];
            return sum + x * next[1] - next[0] * y;
          }, 0);
          if (area < 0) polygon.reverse();
          filled.push("M" + polygon.map(p => point(p).join(" ")).join("L") + "Z");
          continue;
        }
        const points = element.points_m!.map(p => point(p).join(",")).join(" ");
        const tag = element.closed ? "polygon" : "polyline";
        fragments.push(`<${tag} points="${points}" ${element.fill ? `fill="${color}" stroke="none"` : `fill="none" ${stroke}`}/>`);
      }
    }
    // One nonzero paint operation unions adjacent triangles without AA cracks.
    if (filled.length) fragments.push(`<path d="${filled.join("")}" fill="${color}" fill-rule="nonzero" stroke="none"/>`);
  }
  let paint = fragments.join("");
  if (program.negative_space_masks.length) {
    const masks = program.negative_space_masks.map(mask => `<polygon points="${mask.points_m.map(p => point(p).join(",")).join(" ")}" fill="black"/>`).join("");
    paint = `<defs><mask id="negative-space" maskUnits="userSpaceOnUse" x="0" y="0" width="${n(width)}" height="${n(height)}"><rect width="${n(width)}" height="${n(height)}" fill="white"/>${masks}</mask></defs><g mask="url(#negative-space)">${paint}</g>`;
  }
  return `<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 ${n(width)} ${n(height)}" width="${n(width * 1000)}mm" height="${n(height * 1000)}mm">${paint}</svg>`;
}
