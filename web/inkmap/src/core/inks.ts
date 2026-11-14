/** The inks Inkmap offers (config/inkmap/inks.json): every ink in the catalogue, by the id `tatbot art separate` reads,
 *  with the display colour the separation matches against. It is what could be loaded, not what is. */
import inksJson from "../../../../config/inkmap/inks.json" with { type: "json" };

export interface Ink {
  id: string;
  name: string;
  rgb: [number, number, number];
  /** False keeps an ink out of suggestions: the lining blacks, or one black design would suggest all three. */
  suggest?: boolean;
}

export const INKS = inksJson.inks as Ink[];
export const inkById = new Map(INKS.map((ink) => [ink.id, ink]));
export const MAX_INKS = 5;
