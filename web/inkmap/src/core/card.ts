/** The placement card: one printable page per placement, for whoever lays the stencil. It is where a placement's
 *  anchor and rotation reach the skin, since the brief's strokes are made flat and the robot draws on a page. */
import type { PlacementGuide } from "./landmarks.ts";

export interface CardSection {
  name: string;
  sentence: string | null;
  sizeMm: readonly [number, number];
  mirror: boolean;
  /** Flip the picture when printing it: source art is drawn unflipped, and a robot render already is. */
  flipArt: boolean;
  tile: { name: string; sizeMm: readonly [number, number]; fits: boolean };
  guide: PlacementGuide | null;
  inks: { name: string; rgb: readonly [number, number, number] | null }[];
  estimate: string | null;
  detail: string[];
  /** The art, paper transparent, and the placement seen on the body: data URLs. */
  artUrl: string;
  mockupUrl: string;
}

/** The most of a page a drawing may take, mm, leaving room for the words on the same sheet. */
const FIGURE_MM: readonly [number, number] = [170, 110];

const escapeHtml = (text: string) => text.replace(/[&<>"']/g, (c) => `&#${c.charCodeAt(0)};`);
const mm = (value: number) => `${Number(value.toFixed(2))}mm`;

/** How the tile's top is turned, in words for the person holding it. */
export function turnWords({ towards, clockwiseDeg }: PlacementGuide["top"]): string {
  const turn = ((clockwiseDeg + 540) % 360) - 180, by = Math.abs(turn);
  if (by < 5) return `Tile top points at the ${towards}`;
  return `Tile top turned ${by}° ${turn > 0 ? "clockwise" : "anticlockwise"} from pointing at the ${towards}, seen from outside the skin`;
}

function figure({ sizeMm: [width, height], tile, artUrl, flipArt }: CardSection): string {
  const [tileWidth, tileHeight] = tile.sizeMm;
  const scale = Math.min(1, FIGURE_MM[0] / Math.max(width, tileWidth), FIGURE_MM[1] / Math.max(height, tileHeight));
  const note = scale === 1 ? "Actual size when printed at 100%. Check the bar with a ruler." : `Shown at ${Math.round(scale * 100)}%, not actual size.`;
  return `<figure style="height:${mm(Math.max(height, tileHeight) * scale)}">
    <div class="tile" style="width:${mm(tileWidth * scale)};height:${mm(tileHeight * scale)}"></div>
    <img src="${artUrl}" alt="" style="width:${mm(width * scale)};height:${mm(height * scale)}${flipArt ? ";transform:translate(-50%,-50%) scaleX(-1)" : ""}">
  </figure>
  <p class="note">${note}</p>
  ${scale === 1 ? `<div class="bar" style="width:50mm">50 mm</div>` : ""}`;
}

function section(card: CardSection, index: number, total: number): string {
  const { guide, tile, sizeMm } = card;
  const rows: [string, string][] = [
    ["Size", `${sizeMm[0].toFixed(1)} × ${sizeMm[1].toFixed(1)} mm${card.mirror ? ", mirrored" : ""}`],
    ["Stencil", `${tile.name} tile, ${tile.sizeMm[0]} × ${tile.sizeMm[1]} mm${tile.fits ? "" : " (the design does not fit it)"}`],
    ...guide ? [
      ["Lay it", guide.landmarks.map(({ name, distanceMm }) => `about ${distanceMm} mm from the ${name}`).join(", ") || "by eye"],
      ["Turn", turnWords(guide.top)],
    ] as [string, string][] : [],
    ...card.estimate ? [["Estimate", card.estimate]] as [string, string][] : [],
    ...card.detail.length ? [["Detail", card.detail.join("; ")]] as [string, string][] : [],
  ];
  const inks = card.inks.map(({ name, rgb }) => `<span><i style="background:${rgb ? `rgb(${rgb.join(" ")})` : "none"}"></i>${escapeHtml(name)}</span>`).join("");
  return `<section>
  <header><h1>${escapeHtml(card.name)}</h1><p>${index + 1} of ${total}${card.sentence ? ` · ${escapeHtml(card.sentence)}` : ""}</p></header>
  <div class="top"><img class="mockup" src="${card.mockupUrl}" alt="The placement on the body">
  <dl>${rows.map(([key, value]) => `<dt>${key}</dt><dd>${escapeHtml(value)}</dd>`).join("")}<dt>Inks</dt><dd class="inks">${inks || "none named"}</dd></dl></div>
  ${figure(card)}
</section>`;
}

/** The whole card as one self-contained page. */
export function cardHtml(title: string, reference: string, cards: CardSection[]): string {
  return `<!doctype html>
<html lang="en"><head><meta charset="utf-8"><title>${escapeHtml(title)}: placement card</title>
<style>
@page { margin: 12mm; }
* { box-sizing: border-box; }
body { margin: 0; font: 11pt/1.35 system-ui, sans-serif; color: #111; background: #fff; }
section { break-after: page; padding: 0 0 6mm; }
section:last-of-type { break-after: auto; }
header h1 { margin: 0; font-size: 16pt; }
header p { margin: 1mm 0 4mm; color: #555; }
.top { display: flex; gap: 6mm; align-items: flex-start; }
.mockup { width: 72mm; border-radius: 2mm; }
dl { flex: 1; margin: 0; display: grid; grid-template-columns: 18mm 1fr; gap: 1.5mm 3mm; }
dt { color: #555; }
dd { margin: 0; }
.inks { display: flex; flex-wrap: wrap; gap: 1mm 3mm; }
.inks i { display: inline-block; width: 3mm; height: 3mm; margin-right: 1mm; border-radius: 50%; border: 0.2mm solid #888; vertical-align: -0.3mm; }
figure { position: relative; margin: 6mm auto 0; }
figure > * { position: absolute; left: 50%; top: 50%; transform: translate(-50%, -50%); }
.tile { border: 0.4mm dashed #0a8; }
.note { margin: 3mm 0 1mm; text-align: center; color: #555; font-size: 9pt; }
.bar { margin: 0 auto; border-top: 0.5mm solid #111; text-align: center; font-size: 9pt; }
footer { color: #777; font-size: 8pt; }
</style></head><body>
${cards.map((card, index) => section(card, index, cards.length)).join("\n")}
<footer>Inkmap brief ${escapeHtml(reference.slice(0, 12))}. Distances are on one nominal body: measure against the person.</footer>
</body></html>
`;
}
