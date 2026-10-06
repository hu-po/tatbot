/** The two gridded paper fixtures on the bench, as config/substrates.yaml
 * records them (a test holds the two equal).
 *
 * The pad is 7.5 × 11 in and 1 cm thick; the cylinder is 85 mm across and
 * 7.5 in long. Both are white paper printed with a faint blue 1/4 in square
 * grid. These are the editor's starting surfaces and what its 3D preview
 * draws; they are nominal authoring geometry, not a measured registration,
 * and session preparation binds the measured setup separately.
 */
export const INCH_MM = 25.4;
export const GRID_PITCH_MM = INCH_MM / 4;

export interface PaperFixture {
  kind: "plane" | "cylinder";
  id: "paper_pad" | "paper_cylinder";
  label: string;
  /** Chart canvas: u across (plane width / cylinder axis), v up (plane
   *  height / cylinder arc), millimetres. */
  canvas_mm: [number, number];
  /** Cylinder radius; null for the pad. */
  radius_mm: number | null;
  /** Pad thickness, or the cylinder's diameter. */
  thickness_mm: number;
  grid_pitch_mm: number;
  /** Paper and rule colours as CSS. */
  paper: string;
  rule: string;
}

export const PAPER_PAD: PaperFixture = {
  kind: "plane", id: "paper_pad", label: "Paper pad",
  canvas_mm: [7.5 * INCH_MM, 11 * INCH_MM], radius_mm: null, thickness_mm: 10,
  grid_pitch_mm: GRID_PITCH_MM, paper: "#ffffff", rule: "#9ebde6",
};

export const PAPER_CYLINDER_RADIUS_MM = 42.5;
/** The drawable band: the whole outer surface except the bottom quarter the
 *  cylinder rests on — three quarters of the circumference, 135° either side
 *  of the crest. The end caps are never drawn on. */
export const PAPER_CYLINDER_BAND_MM = Number((2 * Math.PI * PAPER_CYLINDER_RADIUS_MM * 0.75).toFixed(2));

export const PAPER_CYLINDER: PaperFixture = {
  kind: "cylinder", id: "paper_cylinder", label: "Paper cylinder",
  canvas_mm: [7.5 * INCH_MM, PAPER_CYLINDER_BAND_MM], radius_mm: PAPER_CYLINDER_RADIUS_MM,
  thickness_mm: 2 * PAPER_CYLINDER_RADIUS_MM,
  grid_pitch_mm: GRID_PITCH_MM, paper: "#ffffff", rule: "#9ebde6",
};

export const FIXTURES: Record<PaperFixture["kind"], PaperFixture> = { plane: PAPER_PAD, cylinder: PAPER_CYLINDER };

export const fixtureFor = (kind: PaperFixture["kind"]): PaperFixture => FIXTURES[kind];

/** The fixture's dimensions as the chart draft carries them. */
export function fixtureDimensions(kind: PaperFixture["kind"]): { width: number; height: number; radius: number } {
  const fixture = fixtureFor(kind);
  return { width: fixture.canvas_mm[0], height: fixture.canvas_mm[1], radius: fixture.radius_mm ?? PAPER_CYLINDER_RADIUS_MM };
}

/** Whether a draft's surface is still exactly one fixture, untouched. */
export function atFixtureDimensions(draft: { kind: PaperFixture["kind"]; width: number; height: number; radius: number }): boolean {
  const same = (want: { width: number; height: number; radius: number }) => draft.width === want.width && draft.height === want.height
    && (draft.kind === "plane" || draft.radius === want.radius);
  return same(fixtureDimensions(draft.kind));
}

export const mmToIn = (mm: number): string => {
  const inches = mm / INCH_MM;
  return Number.isInteger(inches * 4) ? `${inches} in` : `${inches.toFixed(2)} in`;
};

/** One line naming the fixture in the units the bench uses. */
export function fixtureCaption(kind: PaperFixture["kind"]): string {
  const fixture = fixtureFor(kind);
  return kind === "plane"
    ? `${fixture.label} · ${fixture.canvas_mm[0]} × ${fixture.canvas_mm[1]} × ${fixture.thickness_mm} mm (${mmToIn(fixture.canvas_mm[0])} × ${mmToIn(fixture.canvas_mm[1])}) · ¼ in grid`
    : `${fixture.label} · ⌀${fixture.thickness_mm} × ${fixture.canvas_mm[0]} mm long (${mmToIn(fixture.canvas_mm[0])}) · ¼ in grid`;
}
