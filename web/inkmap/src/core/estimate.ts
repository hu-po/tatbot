/** What a design at a size is likely to cost the robot. Fitted, not computed: the strokes do not exist yet, and DBV3
 *  makes them from the brief. Over 52 of the art corpus's separations at 60 mm the median error is 10% and one in ten
 *  is over a third out; a real figure comes from `tatbot art separate`. Inked area drives both: dips scale with line,
 *  line with area. An ink costs a rinse, its colour's first strokes and at least its first dip. */
const MINUTES_PER_INKED_MM2 = 0.072;
const MINUTES_PER_INK = 2.6;
const DIPS_PER_INKED_MM2 = 0.04;

export interface CostEstimate { minutes: number; dips: number }

export function estimateCost(coverage: number, [width, height]: readonly [number, number], inkCount: number): CostEstimate {
  const inked = coverage * width * height;
  return { minutes: MINUTES_PER_INKED_MM2 * inked + MINUTES_PER_INK * inkCount, dips: DIPS_PER_INKED_MM2 * inked + inkCount };
}

/** Minutes as a person says them, to the nearest five. */
export function formatMinutes(minutes: number): string {
  const rounded = Math.max(5, Math.round(minutes / 5) * 5);
  const hours = Math.floor(rounded / 60), rest = rounded % 60;
  return hours ? `${hours} h${rest ? ` ${rest} min` : ""}` : `${rest} min`;
}
