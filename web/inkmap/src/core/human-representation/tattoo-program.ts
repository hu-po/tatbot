import { validateContract, type JsonObject } from "./schema.ts";

export interface TattooElement {
  id: string;
  kind: "path" | "region" | "dots" | "stipple" | "cubic_bezier";
  closed: boolean;
  fill: boolean;
  width_m: number;
  deposition: number;
  points_m?: [number, number][];
  control_points_m?: [number, number][];
}

export interface TattooProgram extends JsonObject {
  schema: "tatbot.tattoo-program/1";
  content_sha256: string;
  canvas_m: { width: number; height: number };
  inks: { id: string; color_srgb: [number, number, number] }[];
  layers: { id: string; ink_id: string; elements: TattooElement[] }[];
  negative_space_masks: { id: string; points_m: [number, number][] }[];
  semantic_intent: string;
  preview_sha256: string;
  provenance: JsonObject;
}

export async function validateTattooProgram(value: unknown): Promise<TattooProgram> {
  return await validateContract(value, {
    expectedSchema: "tatbot.tattoo-program/1",
  }) as TattooProgram;
}
