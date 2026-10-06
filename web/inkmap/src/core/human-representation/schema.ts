import { sha256Hex } from "../sha256.ts"

export const HUMAN_REP_SCHEMAS = [
  "tatbot.body-identity/1",
  "tatbot.body-state/1",
  "tatbot.tattoo-program/1",
  "tatbot.surface-coordinate/1",
  "tatbot.surface-placement/1",
  "tatbot.surface-placement/2",
  "tatbot.surface-curve/1",
  "tatbot.surface-curve/2",
  "tatbot.ink-program/1",
  "tatbot.ink-program/2",
  "tatbot.surface-registration/1",
  "tatbot.execution-program/1",
] as const

export type HumanRepSchema = (typeof HUMAN_REP_SCHEMAS)[number]
export type JsonObject = Record<string, unknown>
const MID_FACE_COUNT = 36_108

export class ContractError extends Error {
  readonly code: string
  readonly path: string

  constructor(code: string, path: string, detail: string) {
    super(`${code} at ${path}: ${detail}`)
    this.name = "ContractError"
    this.code = code
    this.path = path
  }
}

class StrictJsonParser {
  private offset = 0
  private readonly source: string

  constructor(source: string) {
    this.source = source
  }

  parse(): unknown {
    const result = this.value("$")
    this.space()
    if (this.offset !== this.source.length) this.fail("invalid_json", "$", "trailing input")
    return result
  }

  private fail(code: string, path: string, detail: string): never {
    throw new ContractError(code, path, detail)
  }

  private space(): void {
    while (
      this.offset < this.source.length &&
      (this.source[this.offset] === " " ||
        this.source[this.offset] === "\t" ||
        this.source[this.offset] === "\r" ||
        this.source[this.offset] === "\n")
    ) this.offset += 1
  }

  private value(path: string): unknown {
    this.space()
    const first = this.source[this.offset]
    if (first === "{") return this.object(path)
    if (first === "[") return this.array(path)
    if (first === '"') return this.string(path)
    for (const [token, value] of [["true", true], ["false", false], ["null", null]] as const) {
      if (this.source.startsWith(token, this.offset)) {
        this.offset += token.length
        return value
      }
    }
    for (const token of ["NaN", "Infinity", "-Infinity"]) {
      if (this.source.startsWith(token, this.offset)) this.fail("non_finite", path, token)
    }
    const match = this.source.slice(this.offset).match(/^-?(?:0|[1-9]\d*)(?:\.\d+)?(?:[eE][+-]?\d+)?/)
    if (!match) this.fail("invalid_json", path, `unexpected token at byte ${this.offset}`)
    this.offset += match[0].length
    const number = Number(match[0])
    if (!Number.isFinite(number)) this.fail("non_finite", path, match[0])
    if (Object.is(number, -0)) this.fail("negative_zero", path, match[0])
    if (/^-?\d+$/.test(match[0]) && !Number.isSafeInteger(number)) {
      this.fail("unsafe_integer", path, match[0])
    }
    return number
  }

  private string(path: string): string {
    const start = this.offset
    this.offset += 1
    let escaped = false
    while (this.offset < this.source.length) {
      const char = this.source[this.offset++]
      if (escaped) escaped = false
      else if (char === "\\") escaped = true
      else if (char === '"') {
        try {
          return wellFormedUnicode(JSON.parse(this.source.slice(start, this.offset)) as string, path)
        } catch (error) {
          this.fail("invalid_json", path, String(error))
        }
      } else if (char.charCodeAt(0) < 0x20) {
        this.fail("invalid_json", path, "unescaped control character")
      }
    }
    this.fail("invalid_json", path, "unterminated string")
  }

  private object(path: string): JsonObject {
    this.offset += 1
    const result: JsonObject = {}
    const keys = new Set<string>()
    this.space()
    if (this.source[this.offset] === "}") {
      this.offset += 1
      return result
    }
    while (true) {
      this.space()
      if (this.source[this.offset] !== '"') this.fail("invalid_json", path, "object key is not a string")
      const key = this.string(path)
      if (keys.has(key)) this.fail("duplicate_key", path, key)
      keys.add(key)
      this.space()
      if (this.source[this.offset++] !== ":") this.fail("invalid_json", path, "missing colon")
      Object.defineProperty(result, key, {
        value: this.value(`${path}.${key}`),
        enumerable: true,
        configurable: true,
        writable: true,
      })
      this.space()
      const next = this.source[this.offset++]
      if (next === "}") return result
      if (next !== ",") this.fail("invalid_json", path, "expected comma or closing brace")
    }
  }

  private array(path: string): unknown[] {
    this.offset += 1
    const result: unknown[] = []
    this.space()
    if (this.source[this.offset] === "]") {
      this.offset += 1
      return result
    }
    while (true) {
      result.push(this.value(`${path}[${result.length}]`))
      this.space()
      const next = this.source[this.offset++]
      if (next === "]") return result
      if (next !== ",") this.fail("invalid_json", path, "expected comma or closing bracket")
    }
  }
}

export function parseJsonStrict(source: string): unknown {
  return new StrictJsonParser(source).parse()
}

function object(value: unknown, path: string): JsonObject {
  if (value === null || Array.isArray(value) || typeof value !== "object") {
    throw new ContractError("wrong_type", path, "expected object")
  }
  return value as JsonObject
}

function wellFormedUnicode(value: string, path: string): string {
  for (let index = 0; index < value.length; index += 1) {
    const unit = value.charCodeAt(index)
    if (unit >= 0xd800 && unit <= 0xdbff) {
      const next = value.charCodeAt(index + 1)
      if (!(next >= 0xdc00 && next <= 0xdfff)) {
        throw new ContractError("invalid_json", path, "lone Unicode surrogate")
      }
      index += 1
    } else if (unit >= 0xdc00 && unit <= 0xdfff) {
      throw new ContractError("invalid_json", path, "lone Unicode surrogate")
    }
  }
  return value
}

function canonicalPart(value: unknown, path: string): string {
  if (value === null) return "null"
  if (typeof value === "boolean") return value ? "true" : "false"
  if (typeof value === "string") return JSON.stringify(wellFormedUnicode(value, path))
  if (typeof value === "number") {
    if (!Number.isFinite(value)) throw new ContractError("non_finite", path, String(value))
    if (Object.is(value, -0)) throw new ContractError("negative_zero", path, "-0")
    return JSON.stringify(value)
  }
  if (Array.isArray(value)) {
    return "[" + value.map((item, index) => canonicalPart(item, path + "[" + index + "]")).join(",") + "]"
  }
  const item = object(value, path)
  return "{" + Object.keys(item).map((key) => wellFormedUnicode(key, path + " key")).sort().map(
    (key) => JSON.stringify(key) + ":" + canonicalPart(item[key], path + "." + key),
  ).join(",") + "}"
}

export function canonicalJson(value: unknown, omitDigest = false): string {
  if (!omitDigest) return canonicalPart(value, "$")
  const copy = { ...object(value, "$") }
  delete copy.content_sha256
  return canonicalPart(copy, "$")
}

export async function canonicalDigest(value: unknown): Promise<string> {
  const bytes = new TextEncoder().encode(canonicalJson(value, true))
  return sha256Hex(bytes.buffer)
}

/** Digest any complete canonical JSON value without omitting contract fields. */
export async function canonicalDocumentDigest(value: unknown): Promise<string> {
  const bytes = new TextEncoder().encode(canonicalJson(value))
  return sha256Hex(bytes.buffer)
}

function keys(value: JsonObject, path: string, required: string[], optional: string[] = []): void {
  const missing = required.filter((key) => !Object.hasOwn(value, key))
  const allowed = new Set([...required, ...optional])
  const unknown = Object.keys(value).filter((key) => !allowed.has(key)).sort()
  if (missing.length) throw new ContractError("missing_field", path, missing.join(", "))
  if (unknown.length) throw new ContractError("unknown_field", path, unknown.join(", "))
}

function string(value: unknown, path: string): string {
  if (typeof value !== "string" || !value) throw new ContractError("wrong_type", path, "expected non-empty string")
  return wellFormedUnicode(value, path)
}

function utcTimestamp(value: unknown, path: string): string {
  const result = string(value, path)
  const parsed = new Date(result)
  if (!/^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z$/.test(result) ||
      Number.isNaN(parsed.getTime()) ||
      parsed.toISOString() !== result.replace("Z", ".000Z")) {
    throw new ContractError("wrong_time", path, "expected an RFC 3339 UTC timestamp")
  }
  return result
}

function number(value: unknown, path: string): number {
  if (typeof value !== "number" || !Number.isFinite(value)) throw new ContractError("wrong_type", path, "expected finite number")
  if (Object.is(value, -0)) throw new ContractError("negative_zero", path, "-0")
  return value
}

function integer(value: unknown, path: string, maximum?: number): number {
  const result = number(value, path)
  if (!Number.isSafeInteger(result) || result < 0) {
    throw new ContractError("wrong_type", path, "expected nonnegative safe integer")
  }
  if (maximum !== undefined && result > maximum) {
    throw new ContractError("out_of_range", path, `must be at most ${maximum}`)
  }
  return result
}

function boolean(value: unknown, path: string): boolean {
  if (typeof value !== "boolean") throw new ContractError("wrong_type", path, "expected boolean")
  return value
}

function array(value: unknown, path: string, length?: number): unknown[] {
  if (!Array.isArray(value)) throw new ContractError("wrong_type", path, "expected array")
  if (length !== undefined && value.length !== length) {
    throw new ContractError("wrong_length", path, `expected ${length}, got ${value.length}`)
  }
  return value
}

function numbers(value: unknown, path: string, length?: number): number[] {
  return array(value, path, length).map((item, index) => number(item, `${path}[${index}]`))
}

function canvasPoint(value: unknown, path: string, width: number, height: number): void {
  const point = numbers(value, path, 2)
  if (!(point[0] >= 0 && point[0] <= width && point[1] >= 0 && point[1] <= height)) {
    throw new ContractError("out_of_range", path, "point lies outside canvas_m")
  }
}

function orderedPair(
  value: unknown,
  path: string,
  minimum?: number,
  maximum?: number,
): number[] {
  const pair = numbers(value, path, 2)
  if (pair[0] > pair[1]) throw new ContractError("out_of_range", path, "lower bound exceeds upper bound")
  if (minimum !== undefined && pair[0] < minimum) {
    throw new ContractError("out_of_range", path, `lower bound must be at least ${minimum}`)
  }
  if (maximum !== undefined && pair[1] > maximum) {
    throw new ContractError("out_of_range", path, `upper bound must be at most ${maximum}`)
  }
  return pair
}

function sha(value: unknown, path: string): string {
  const result = string(value, path)
  if (!/^[0-9a-f]{64}$/.test(result)) throw new ContractError("wrong_hash", path, "expected lowercase SHA-256")
  return result
}

function matrix(value: unknown, path: string): void {
  const rows = array(value, path, 4)
  rows.forEach((row, index) => numbers(row, `${path}[${index}]`, 4))
  if ((rows[3] as number[]).some((entry, index) => entry !== [0, 0, 0, 1][index])) {
    throw new ContractError("wrong_transform", path, "last row must be [0,0,0,1]")
  }
}

function provenance(value: unknown, path: string): void {
  const result = object(value, path)
  keys(result, path, ["producer", "version", "created_utc", "source_sha256"], ["seed", "checkpoint_sha256", "prompt"])
  string(result.producer, `${path}.producer`)
  string(result.version, `${path}.version`)
  utcTimestamp(result.created_utc, `${path}.created_utc`)
  sha(result.source_sha256, `${path}.source_sha256`)
  if (result.seed !== undefined) integer(result.seed, `${path}.seed`)
  if (result.checkpoint_sha256 !== undefined) sha(result.checkpoint_sha256, `${path}.checkpoint_sha256`)
  if (result.prompt !== undefined) string(result.prompt, `${path}.prompt`)
}

export interface ReadOptions {
  expectedSchema?: HumanRepSchema
  expectedTopologySha256?: string
}

const fields: Record<HumanRepSchema, string[]> = {
  "tatbot.surface-curve/2": ["schema","content_sha256","coordinates","rest_surface_arc_length_m","width_m","deposition","direction","source_primitive_sha256","compiler_sha256"],
  "tatbot.ink-program/2": ["schema","content_sha256","tattoo_program_sha256","surface_placement_sha256","compiler_sha256","initial_ink_state","predicted_ink_state","events","total_material_path_length_m","uncertainty","provenance"],
  "tatbot.surface-placement/2": ["schema","content_sha256","tattoo_program_sha256","target","physical_scale_m","rotation_rad","mirrored","warp","review","provenance"],
  "tatbot.body-identity/1": ["schema","content_sha256","model_spec_sha256","coefficients","scales","bounds_prior","rest_surface_sha256","provenance"],
  "tatbot.body-state/1": ["schema","content_sha256","model_spec_sha256","body_identity_sha256","joint_rotations_xyzw","named_pose","tracked_source","tatbot_from_body","confidence","correctives_enabled","posed_surface_sha256","provenance"],
  "tatbot.tattoo-program/1": ["schema","content_sha256","canvas_m","inks","layers","negative_space_masks","semantic_intent","preview_sha256","provenance"],
  "tatbot.surface-coordinate/1": ["schema","content_sha256","topology_sha256","face_index","barycentric"],
  "tatbot.surface-placement/1": ["schema","content_sha256","tattoo_program_sha256","body_identity_sha256","rest_surface_sha256","semantic_site","laterality","anchor","tangent_frame_rule","physical_scale_m","rotation_rad","mirrored","warp","supported_domain","review","provenance"],
  "tatbot.surface-curve/1": ["schema","content_sha256","coordinates","rest_surface_arc_length_m","width_m","deposition","direction","source_primitive_sha256","compiler_sha256"],
  "tatbot.ink-program/1": ["schema","content_sha256","tattoo_program_sha256","surface_placement_sha256","compiler_sha256","initial_ink_state","predicted_ink_state","events","total_material_path_length_m","uncertainty","provenance"],
  "tatbot.surface-registration/1": ["schema","content_sha256","source_frame","target_frame","body_state_sha256","measured_surface_sha256","method","correspondences","observed_patch_from_body","supported_cells","covariance","confidence","capture_sha256","calibration_sha256","provenance"],
  "tatbot.execution-program/1": ["schema","content_sha256","ink_program","body_state","surface_registration","measured_surface","tool","robot","support","palette","calibration","exact_events","uncertainty","preflight","samples_manifest","provenance"],
}

const optionalFields: Partial<Record<HumanRepSchema, string[]>> = {
  "tatbot.ink-program/1": ["operating_budget_s", "fill_style"],
  "tatbot.ink-program/2": ["operating_budget_s", "fill_style"],
}

// The paint planners a compiled program may name; absent means concentric.
const fillStyles = ["concentric", "hatch"]

function coordinate(value: unknown, path: string, options: ReadOptions, envelope = false): void {
  const result = object(value, path)
  keys(result, path, envelope ? fields["tatbot.surface-coordinate/1"] : ["topology_sha256", "face_index", "barycentric"])
  const topology = sha(result.topology_sha256, `${path}.topology_sha256`)
  if (options.expectedTopologySha256 && topology !== options.expectedTopologySha256) {
    throw new ContractError("wrong_topology", `${path}.topology_sha256`, topology)
  }
  integer(result.face_index, `${path}.face_index`, MID_FACE_COUNT - 1)
  const barycentric = numbers(result.barycentric, `${path}.barycentric`, 3)
  if (Math.abs(barycentric.reduce((sum, entry) => sum + entry, 0) - 1) > 1e-6 ||
      barycentric.some((entry) => entry < -1e-6 || entry > 1 + 1e-6)) {
    throw new ContractError("bad_barycentric", `${path}.barycentric`, "must be normalized and in range")
  }
}

function curve(value: unknown, path: string, options: ReadOptions, envelope = false, chartAllowed = false): void {
  const result = object(value, path)
  const required = ["coordinates","rest_surface_arc_length_m","width_m","deposition","direction","source_primitive_sha256","compiler_sha256"]
  keys(result, path, envelope ? fields["tatbot.surface-curve/1"] : required)
  const coordinates = array(result.coordinates, `${path}.coordinates`)
  if (coordinates.length < 2) throw new ContractError("wrong_length", `${path}.coordinates`, "at least two required")
  let topology: string | undefined
  const chart = chartAllowed && Object.hasOwn(object(coordinates[0], `${path}.coordinates[0]`), "chart_uv_m")
  const chartPoints: number[][] = []
  coordinates.forEach((entry, index) => {
    const coordinatePath = `${path}.coordinates[${index}]`
    if (chart) {
      const point = object(entry, coordinatePath)
      keys(point, coordinatePath, ["target_sha256", "chart_uv_m"])
      const binding = sha(point.target_sha256, `${coordinatePath}.target_sha256`)
      chartPoints.push(numbers(point.chart_uv_m, `${coordinatePath}.chart_uv_m`, 2))
      if (topology !== undefined && topology !== binding) throw new ContractError("wrong_target", coordinatePath, "curve mixes target bindings")
      topology = binding
      return
    }
    coordinate(entry, coordinatePath, options)
    const actual = sha(object(entry, coordinatePath).topology_sha256, `${coordinatePath}.topology_sha256`)
    if (topology === undefined) topology = actual
    else if (actual !== topology) {
      throw new ContractError("wrong_topology", `${coordinatePath}.topology_sha256`, "surface curve mixes topology digests")
    }
  })
  if (number(result.rest_surface_arc_length_m, `${path}.rest_surface_arc_length_m`) <= 0 ||
      number(result.width_m, `${path}.width_m`) <= 0) {
    throw new ContractError("out_of_range", path, "lengths must be positive")
  }
  if (chart) {
    const length = chartPoints.slice(1).reduce((sum, p, i) => sum + Math.hypot(p[0] - chartPoints[i][0], p[1] - chartPoints[i][1]), 0)
    if (!Number.isFinite(length) || Math.abs(length - Number(result.rest_surface_arc_length_m)) > 1e-9) throw new ContractError("wrong_length", `${path}.rest_surface_arc_length_m`, "differs from metric chart curve")
  }
  const deposition = number(result.deposition, `${path}.deposition`)
  if (deposition < 0 || deposition > 1) throw new ContractError("out_of_range", `${path}.deposition`, "expected [0,1]")
  if (!["forward", "reverse"].includes(string(result.direction, `${path}.direction`))) {
    throw new ContractError("wrong_enum", `${path}.direction`, String(result.direction))
  }
  sha(result.source_primitive_sha256, `${path}.source_primitive_sha256`)
  sha(result.compiler_sha256, `${path}.compiler_sha256`)
}

function identity(result: JsonObject, path: string): void {
  sha(result.model_spec_sha256, `${path}.model_spec_sha256`)
  const coefficients = numbers(result.coefficients, `${path}.coefficients`, 45)
  const scales = numbers(result.scales, `${path}.scales`, 68)
  const prior = object(result.bounds_prior, `${path}.bounds_prior`)
  keys(prior, `${path}.bounds_prior`, ["name", "version", "max_abs_coefficient", "max_abs_scale"])
  string(prior.name, `${path}.bounds_prior.name`)
  string(prior.version, `${path}.bounds_prior.version`)
  const limit = number(prior.max_abs_coefficient, `${path}.bounds_prior.max_abs_coefficient`)
  if (limit <= 0 || coefficients.some((entry) => Math.abs(entry) > limit)) {
    throw new ContractError("out_of_range", `${path}.coefficients`, "coefficient exceeds prior")
  }
  const scaleLimit = number(prior.max_abs_scale, `${path}.bounds_prior.max_abs_scale`)
  if (scaleLimit <= 0 || scales.some((entry) => Math.abs(entry) > scaleLimit)) {
    throw new ContractError("out_of_range", `${path}.scales`, "scale exceeds prior")
  }
  sha(result.rest_surface_sha256, `${path}.rest_surface_sha256`)
  provenance(result.provenance, `${path}.provenance`)
}

function bodyState(result: JsonObject, path: string): void {
  sha(result.model_spec_sha256, `${path}.model_spec_sha256`)
  sha(result.body_identity_sha256, `${path}.body_identity_sha256`)
  array(result.joint_rotations_xyzw, `${path}.joint_rotations_xyzw`, 77).forEach((entry, index) => {
    const rotation = numbers(entry, `${path}.joint_rotations_xyzw[${index}]`, 4)
    if (Math.abs(rotation.reduce((sum, item) => sum + item * item, 0) - 1) > 1e-5) {
      throw new ContractError("bad_rotation", `${path}.joint_rotations_xyzw[${index}]`, "quaternion is not unit")
    }
  })
  if ((result.named_pose === null) === (result.tracked_source === null)) {
    throw new ContractError("wrong_pose_source", path, "exactly one of named_pose and tracked_source must be populated")
  }
  if (result.named_pose !== null) string(result.named_pose, `${path}.named_pose`)
  else {
    const tracked = object(result.tracked_source, `${path}.tracked_source`)
    keys(tracked, `${path}.tracked_source`, ["tracker","sample_time_utc","capture_sha256","source_frame"])
    string(tracked.tracker, `${path}.tracked_source.tracker`)
    utcTimestamp(tracked.sample_time_utc, `${path}.tracked_source.sample_time_utc`)
    sha(tracked.capture_sha256, `${path}.tracked_source.capture_sha256`)
    const sourceFrame = string(tracked.source_frame, `${path}.tracked_source.source_frame`)
    if (!/^[a-z][a-z0-9_]*$/.test(sourceFrame)) {
      throw new ContractError("wrong_frame", `${path}.tracked_source.source_frame`, sourceFrame)
    }
  }
  matrix(result.tatbot_from_body, `${path}.tatbot_from_body`)
  const confidence = number(result.confidence, `${path}.confidence`)
  if (confidence < 0 || confidence > 1) throw new ContractError("out_of_range", `${path}.confidence`, "expected [0,1]")
  if (boolean(result.correctives_enabled, `${path}.correctives_enabled`) !== true) {
    throw new ContractError("wrong_value", `${path}.correctives_enabled`, "canonical MHR/SOMA mode requires correctives")
  }
  sha(result.posed_surface_sha256, `${path}.posed_surface_sha256`)
  provenance(result.provenance, `${path}.provenance`)
}

function tattoo(result: JsonObject, path: string): void {
  const canvas = object(result.canvas_m, `${path}.canvas_m`)
  keys(canvas, `${path}.canvas_m`, ["width", "height"])
  const width = number(canvas.width, `${path}.canvas_m.width`)
  const height = number(canvas.height, `${path}.canvas_m.height`)
  if (width <= 0 || height <= 0) {
    throw new ContractError("out_of_range", `${path}.canvas_m`, "dimensions must be positive")
  }
  const inkIds = new Set<string>()
  const inks = array(result.inks, `${path}.inks`)
  if (!inks.length) throw new ContractError("wrong_length", `${path}.inks`, "at least one ink is required")
  inks.forEach((raw, index) => {
    const ink = object(raw, `${path}.inks[${index}]`)
    keys(ink, `${path}.inks[${index}]`, ["id", "color_srgb"])
    const id = string(ink.id, `${path}.inks[${index}].id`)
    if (inkIds.has(id)) throw new ContractError("duplicate_id", `${path}.inks[${index}].id`, id)
    inkIds.add(id)
    if (numbers(ink.color_srgb, `${path}.inks[${index}].color_srgb`, 3).some((entry) => entry < 0 || entry > 1)) {
      throw new ContractError("out_of_range", `${path}.inks[${index}].color_srgb`, "expected [0,1]")
    }
  })
  const layers = array(result.layers, `${path}.layers`)
  if (!layers.length) throw new ContractError("wrong_length", `${path}.layers`, "at least one layer is required")
  const layerIds = new Set<string>()
  const elementIds = new Set<string>()
  layers.forEach((raw, index) => {
    const layer = object(raw, `${path}.layers[${index}]`)
    keys(layer, `${path}.layers[${index}]`, ["id", "ink_id", "elements"])
    const layerId = string(layer.id, `${path}.layers[${index}].id`)
    if (layerIds.has(layerId)) throw new ContractError("duplicate_id", `${path}.layers[${index}].id`, layerId)
    layerIds.add(layerId)
    if (!inkIds.has(string(layer.ink_id, `${path}.layers[${index}].ink_id`))) {
      throw new ContractError("unknown_reference", `${path}.layers[${index}].ink_id`, String(layer.ink_id))
    }
    const elementsPath = `${path}.layers[${index}].elements`
    const elements = array(layer.elements, elementsPath)
    if (!elements.length) throw new ContractError("wrong_length", elementsPath, "at least one element is required")
    elements.forEach((value, elementIndex) => {
      const elementPath = `${elementsPath}[${elementIndex}]`
      const element = object(value, elementPath)
      const common = ["id","kind","closed","fill","width_m","deposition"]
      keys(element, elementPath, common, ["points_m","control_points_m"])
      const elementId = string(element.id, `${elementPath}.id`)
      if (elementIds.has(elementId)) throw new ContractError("duplicate_id", `${elementPath}.id`, elementId)
      elementIds.add(elementId)
      const kind = string(element.kind, `${elementPath}.kind`)
      if (["path","region","dots","stipple"].includes(kind)) {
        keys(element, elementPath, [...common, "points_m"])
        const pointsPath = `${elementPath}.points_m`
        const points = array(element.points_m, pointsPath)
        const minimum = kind === "region" ? 3 : kind === "path" ? 2 : 1
        if (points.length < minimum) {
          throw new ContractError("wrong_length", pointsPath, `${kind} requires at least ${minimum} point(s)`)
        }
        points.forEach((point, pointIndex) => canvasPoint(point, `${pointsPath}[${pointIndex}]`, width, height))
      } else if (kind === "cubic_bezier") {
        keys(element, elementPath, [...common, "control_points_m"])
        array(element.control_points_m, `${elementPath}.control_points_m`, 4).forEach(
          (point, pointIndex) => canvasPoint(point, `${elementPath}.control_points_m[${pointIndex}]`, width, height),
        )
      } else {
        throw new ContractError("unsupported_element", `${elementPath}.kind`, kind)
      }
      boolean(element.closed, `${elementPath}.closed`)
      boolean(element.fill, `${elementPath}.fill`)
      if (number(element.width_m, `${elementPath}.width_m`) <= 0) {
        throw new ContractError("out_of_range", `${elementPath}.width_m`, "must be positive")
      }
      const deposition = number(element.deposition, `${elementPath}.deposition`)
      if (deposition < 0 || deposition > 1) throw new ContractError("out_of_range", `${elementPath}.deposition`, "expected [0,1]")
    })
  })
  const maskIds = new Set<string>()
  array(result.negative_space_masks, `${path}.negative_space_masks`).forEach((raw, index) => {
    const mask = object(raw, `${path}.negative_space_masks[${index}]`)
    keys(mask, `${path}.negative_space_masks[${index}]`, ["id", "points_m"])
    const maskId = string(mask.id, `${path}.negative_space_masks[${index}].id`)
    if (maskIds.has(maskId)) throw new ContractError("duplicate_id", `${path}.negative_space_masks[${index}].id`, maskId)
    maskIds.add(maskId)
    const points = array(mask.points_m, `${path}.negative_space_masks[${index}].points_m`)
    if (points.length < 3) {
      throw new ContractError("wrong_length", `${path}.negative_space_masks[${index}].points_m`, "at least three points are required")
    }
    points.forEach((point, pointIndex) => canvasPoint(
      point,
      `${path}.negative_space_masks[${index}].points_m[${pointIndex}]`,
      width,
      height,
    ))
  })
  string(result.semantic_intent, `${path}.semantic_intent`)
  sha(result.preview_sha256, `${path}.preview_sha256`)
  provenance(result.provenance, `${path}.provenance`)
}

function targetPlacement(result: JsonObject, path: string, options: ReadOptions): void {
  const target = object(result.target, `${path}.target`)
  const kind = string(target.kind, `${path}.target.kind`)
  if (kind === "body") {
    const names = ["body_identity_sha256","rest_surface_sha256","semantic_site","laterality","anchor","tangent_frame_rule","supported_domain"]
    keys(target, `${path}.target`, ["kind", ...names])
    placement({ ...result, ...Object.fromEntries(names.map(name => [name, target[name]])) }, path, options)
    return
  }
  if (kind !== "plane" && kind !== "cylinder") throw new ContractError("wrong_enum", `${path}.target.kind`, kind)
  keys(target, `${path}.target`, ["kind","canvas_m","anchor_uv_m","margin_m", ...(kind === "cylinder" ? ["radius_m"] : [])])
  const canvas = numbers(target.canvas_m, `${path}.target.canvas_m`, 2)
  const anchor = numbers(target.anchor_uv_m, `${path}.target.anchor_uv_m`, 2)
  const scale = numbers(result.physical_scale_m, `${path}.physical_scale_m`, 2)
  const margin = number(target.margin_m, `${path}.target.margin_m`)
  if ([...canvas, ...scale].some(v => v <= 0) || margin < 0) throw new ContractError("out_of_range", `${path}.target`, "positive dimensions and nonnegative margin required")
  if (kind === "cylinder") {
    const radius = number(target.radius_m, `${path}.target.radius_m`)
    if (radius <= 0 || canvas[1] >= 2 * Math.PI * radius) throw new ContractError("girth", `${path}.target.canvas_m`, "chart must be smaller than one circumference")
  }
  const angle = number(result.rotation_rad, `${path}.rotation_rad`)
  boolean(result.mirrored, `${path}.mirrored`)
  if (result.warp !== null) throw new ContractError("unsupported_warp", `${path}.warp`, "analytic placements require an unwarped chart")
  const c = Math.abs(Math.cos(angle)), s = Math.abs(Math.sin(angle))
  const extent = [(c * scale[0] + s * scale[1]) / 2, (s * scale[0] + c * scale[1]) / 2]
  if (canvas.some((size, i) => Math.abs(anchor[i]) + extent[i] + margin > size / 2 + 1e-12)) throw new ContractError("placement_outside_domain", `${path}.target`, "rotated artwork canvas exceeds target margin")
  sha(result.tattoo_program_sha256, `${path}.tattoo_program_sha256`)
  const review = object(result.review, `${path}.review`)
  keys(review, `${path}.review`, ["status","reviewer","evidence_sha256"])
  if (!["pending","accepted","rejected"].includes(string(review.status, `${path}.review.status`))) throw new ContractError("wrong_enum", `${path}.review.status`, String(review.status))
  string(review.reviewer, `${path}.review.reviewer`)
  sha(review.evidence_sha256, `${path}.review.evidence_sha256`)
  provenance(result.provenance, `${path}.provenance`)
}

function placement(result: JsonObject, path: string, options: ReadOptions): void {
  for (const name of ["tattoo_program_sha256","body_identity_sha256","rest_surface_sha256"]) sha(result[name], `${path}.${name}`)
  string(result.semantic_site, `${path}.semantic_site`)
  if (!["left","right","midline","not_applicable"].includes(string(result.laterality, `${path}.laterality`))) {
    throw new ContractError("wrong_enum", `${path}.laterality`, String(result.laterality))
  }
  coordinate(result.anchor, `${path}.anchor`, options)
  const anchor = object(result.anchor, `${path}.anchor`)
  string(result.tangent_frame_rule, `${path}.tangent_frame_rule`)
  if (numbers(result.physical_scale_m, `${path}.physical_scale_m`, 2).some((entry) => entry <= 0)) {
    throw new ContractError("out_of_range", `${path}.physical_scale_m`, "must be positive")
  }
  number(result.rotation_rad, `${path}.rotation_rad`)
  boolean(result.mirrored, `${path}.mirrored`)
  if (result.warp !== null) {
    const warp = object(result.warp, `${path}.warp`)
    keys(warp, `${path}.warp`, ["kind","max_displacement_m","parameters"])
    string(warp.kind, `${path}.warp.kind`)
    if (number(warp.max_displacement_m, `${path}.warp.max_displacement_m`) < 0) {
      throw new ContractError("out_of_range", `${path}.warp.max_displacement_m`, "must be nonnegative")
    }
    numbers(warp.parameters, `${path}.warp.parameters`)
  }
  const domain = object(result.supported_domain, `${path}.supported_domain`)
  keys(domain, `${path}.supported_domain`, ["face_indices","margin_m"])
  const facesPath = `${path}.supported_domain.face_indices`
  const faces = array(domain.face_indices, facesPath).map(
    (entry, index) => integer(entry, `${facesPath}[${index}]`, MID_FACE_COUNT - 1),
  )
  if (!faces.length) throw new ContractError("wrong_length", facesPath, "at least one face is required")
  if (!faces.includes(integer(anchor.face_index, `${path}.anchor.face_index`, MID_FACE_COUNT - 1))) {
    throw new ContractError("anchor_outside_domain", `${path}.anchor.face_index`, "anchor face is absent from supported_domain.face_indices")
  }
  if (number(domain.margin_m, `${path}.supported_domain.margin_m`) < 0) {
    throw new ContractError("out_of_range", `${path}.supported_domain.margin_m`, "must be nonnegative")
  }
  const review = object(result.review, `${path}.review`)
  keys(review, `${path}.review`, ["status","reviewer","evidence_sha256"])
  if (!["pending", "accepted", "rejected"].includes(string(review.status, `${path}.review.status`))) {
    throw new ContractError("wrong_enum", `${path}.review.status`, String(review.status))
  }
  string(review.reviewer, `${path}.review.reviewer`)
  sha(review.evidence_sha256, `${path}.review.evidence_sha256`)
  provenance(result.provenance, `${path}.provenance`)
}

function inkProgram(result: JsonObject, path: string, options: ReadOptions): void {
  for (const name of ["tattoo_program_sha256","surface_placement_sha256","compiler_sha256"]) sha(result[name], `${path}.${name}`)
  if (result.operating_budget_s !== undefined) {
    // A cartridge program carries the budget it was compiled against.
    const budget = number(result.operating_budget_s, `${path}.operating_budget_s`)
    if (!(budget > 0)) throw new ContractError("out_of_range", `${path}.operating_budget_s`, "expected a positive finite number of seconds")
  }
  if (result.fill_style !== undefined && !fillStyles.includes(result.fill_style as string)) {
    throw new ContractError("wrong_enum", `${path}.fill_style`, String(result.fill_style))
  }
  for (const name of ["initial_ink_state","predicted_ink_state"]) {
    const state = object(result[name], `${path}.${name}`)
    keys(state, `${path}.${name}`, ["ink_id","load_fraction"])
    string(state.ink_id, `${path}.${name}.ink_id`)
    const load = number(state.load_fraction, `${path}.${name}.load_fraction`)
    if (load < 0 || load > 1) throw new ContractError("out_of_range", `${path}.${name}.load_fraction`, "expected [0,1]")
  }
  const events = array(result.events, `${path}.events`)
  if (!events.length) throw new ContractError("wrong_length", `${path}.events`, "at least one event is required")
  let materialLength = 0
  events.forEach((raw, index) => {
    const event = object(raw, `${path}.events[${index}]`)
    const kind = string(event.kind, `${path}.events[${index}].kind`)
    if (kind === "stroke") {
      keys(event, `${path}.events[${index}]`, ["kind","curve","start_choice","allowed_tool_class","speed_m_s","orientation_tolerance_rad","contact_envelope_m","research_depth_band_m","ordering_rationale"])
      curve(event.curve, `${path}.events[${index}].curve`, options, false, result.schema === "tatbot.ink-program/2")
      for (const rawPoint of array(object(event.curve, path).coordinates, path)) {
        const point = object(rawPoint, path)
        if (point.target_sha256 !== undefined && point.target_sha256 !== result.surface_placement_sha256) throw new ContractError("wrong_target", `${path}.events[${index}].curve`, "curve differs from program placement")
      }
      materialLength += number(
        object(event.curve, `${path}.events[${index}].curve`).rest_surface_arc_length_m,
        `${path}.events[${index}].curve.rest_surface_arc_length_m`,
      )
      if (!["start", "end"].includes(string(event.start_choice, `${path}.events[${index}].start_choice`))) {
        throw new ContractError("wrong_enum", `${path}.events[${index}].start_choice`, String(event.start_choice))
      }
      string(event.allowed_tool_class, `${path}.events[${index}].allowed_tool_class`)
      if (number(event.speed_m_s, `${path}.events[${index}].speed_m_s`) <= 0) {
        throw new ContractError("out_of_range", `${path}.events[${index}].speed_m_s`, "must be positive")
      }
      if (number(event.orientation_tolerance_rad, `${path}.events[${index}].orientation_tolerance_rad`) < 0) {
        throw new ContractError("out_of_range", `${path}.events[${index}].orientation_tolerance_rad`, "must be nonnegative")
      }
      orderedPair(event.contact_envelope_m, `${path}.events[${index}].contact_envelope_m`)
      if (event.research_depth_band_m !== null) {
        orderedPair(event.research_depth_band_m, `${path}.events[${index}].research_depth_band_m`)
      }
      string(event.ordering_rationale, `${path}.events[${index}].ordering_rationale`)
    } else if (kind === "pen_transition") {
      keys(event, `${path}.events[${index}]`, ["kind","state"])
      if (!["lift", "approach", "contact_intent", "retract"].includes(string(event.state, `${path}.events[${index}].state`))) {
        throw new ContractError("wrong_enum", `${path}.events[${index}].state`, String(event.state))
      }
    } else if (kind === "dip") {
      keys(event, `${path}.events[${index}]`, ["kind","ink_id","target_load","trigger_reason","dependency","expected_load_after"])
      string(event.ink_id, `${path}.events[${index}].ink_id`)
      orderedPair(event.target_load, `${path}.events[${index}].target_load`, 0, 1)
      string(event.trigger_reason, `${path}.events[${index}].trigger_reason`)
      string(event.dependency, `${path}.events[${index}].dependency`)
      const expectedLoad = number(event.expected_load_after, `${path}.events[${index}].expected_load_after`)
      if (expectedLoad < 0 || expectedLoad > 1) {
        throw new ContractError("out_of_range", `${path}.events[${index}].expected_load_after`, "expected [0,1]")
      }
    } else if (kind === "tool_change") {
      keys(event, `${path}.events[${index}]`, ["kind","required_tool_class","transition_intent"])
      string(event.required_tool_class, `${path}.events[${index}].required_tool_class`)
      string(event.transition_intent, `${path}.events[${index}].transition_intent`)
    } else if (kind === "barrier") {
      keys(event, `${path}.events[${index}]`, ["kind","intent"])
      string(event.intent, `${path}.events[${index}].intent`)
    } else throw new ContractError("wrong_enum", `${path}.events[${index}].kind`, kind)
  })
  const declaredLength = number(result.total_material_path_length_m, `${path}.total_material_path_length_m`)
  if (declaredLength < 0) {
    throw new ContractError("out_of_range", `${path}.total_material_path_length_m`, "must be nonnegative")
  }
  if (Math.abs(declaredLength - materialLength) > Math.max(1e-12, Math.abs(materialLength) * 1e-12)) {
    throw new ContractError(
      "wrong_value",
      `${path}.total_material_path_length_m`,
      `declared ${declaredLength}; stroke sum is ${materialLength}`,
    )
  }
  const uncertainty = object(result.uncertainty, `${path}.uncertainty`)
  keys(uncertainty, `${path}.uncertainty`, ["length_sigma_m","load_sigma"])
  if (number(uncertainty.length_sigma_m, `${path}.uncertainty.length_sigma_m`) < 0 ||
      number(uncertainty.load_sigma, `${path}.uncertainty.load_sigma`) < 0) {
    throw new ContractError("out_of_range", `${path}.uncertainty`, "uncertainty must be nonnegative")
  }
  provenance(result.provenance, `${path}.provenance`)
}

function registration(result: JsonObject, path: string, options: ReadOptions): void {
  if (result.source_frame !== "body") throw new ContractError("wrong_frame", `${path}.source_frame`, String(result.source_frame))
  if (result.target_frame !== "observed_patch") throw new ContractError("wrong_frame", `${path}.target_frame`, String(result.target_frame))
  for (const name of ["body_state_sha256","measured_surface_sha256","capture_sha256","calibration_sha256"]) sha(result[name], `${path}.${name}`)
  string(result.method, `${path}.method`)
  const correspondences = array(result.correspondences, `${path}.correspondences`)
  if (!correspondences.length) throw new ContractError("wrong_length", `${path}.correspondences`, "at least one required")
  correspondences.forEach((raw, index) => {
    const entry = object(raw, `${path}.correspondences[${index}]`)
    keys(entry, `${path}.correspondences[${index}]`, ["canonical","observed_xyz_m","error_m"])
    coordinate(entry.canonical, `${path}.correspondences[${index}].canonical`, options)
    numbers(entry.observed_xyz_m, `${path}.correspondences[${index}].observed_xyz_m`, 3)
    if (number(entry.error_m, `${path}.correspondences[${index}].error_m`) < 0) {
      throw new ContractError("out_of_range", `${path}.correspondences[${index}].error_m`, "must be nonnegative")
    }
  })
  matrix(result.observed_patch_from_body, `${path}.observed_patch_from_body`)
  const cells = array(result.supported_cells, `${path}.supported_cells`)
  if (!cells.length) throw new ContractError("wrong_length", `${path}.supported_cells`, "at least one required")
  cells.forEach((entry, index) => integer(entry, `${path}.supported_cells[${index}]`))
  numbers(result.covariance, `${path}.covariance`, 36)
  const confidence = number(result.confidence, `${path}.confidence`)
  if (confidence < 0 || confidence > 1) throw new ContractError("out_of_range", `${path}.confidence`, "expected [0,1]")
  provenance(result.provenance, `${path}.provenance`)
}

async function nested(value: unknown, schema: HumanRepSchema, path: string, options: ReadOptions): Promise<void> {
  const result = object(value, path)
  if (result.schema !== schema) throw new ContractError("wrong_schema", `${path}.schema`, String(result.schema))
  keys(result, path, fields[schema], schema.startsWith("tatbot.ink-program/") ? optionalFields[schema] : [])
  sha(result.content_sha256, `${path}.content_sha256`)
  if (schema === "tatbot.body-identity/1") identity(result, path)
  else if (schema === "tatbot.body-state/1") bodyState(result, path)
  else if (schema === "tatbot.tattoo-program/1") tattoo(result, path)
  else if (schema === "tatbot.surface-coordinate/1") coordinate(result, path, options, true)
  else if (schema === "tatbot.surface-placement/1") placement(result, path, options)
  else if (schema === "tatbot.surface-placement/2") targetPlacement(result, path, options)
  else if (schema === "tatbot.surface-curve/1") curve(result, path, options, true)
  else if (schema === "tatbot.surface-curve/2") curve(result, path, options, true, true)
  else if (schema === "tatbot.ink-program/1") inkProgram(result, path, options)
  else if (schema === "tatbot.ink-program/2") inkProgram(result, path, options)
  else if (schema === "tatbot.surface-registration/1") registration(result, path, options)
  else if (schema === "tatbot.execution-program/1") {
    await nested(result.ink_program, "tatbot.ink-program/1", `${path}.ink_program`, options)
    await nested(result.body_state, "tatbot.body-state/1", `${path}.body_state`, options)
    await nested(result.surface_registration, "tatbot.surface-registration/1", `${path}.surface_registration`, options)
    const inkProgramValue = object(result.ink_program, `${path}.ink_program`)
    const bodyStateValue = object(result.body_state, `${path}.body_state`)
    const registrationValue = object(result.surface_registration, `${path}.surface_registration`)
    if (registrationValue.body_state_sha256 !== bodyStateValue.content_sha256) {
      throw new ContractError(
        "wrong_hash",
        `${path}.surface_registration.body_state_sha256`,
        "does not bind the embedded body state",
      )
    }
    const measured = object(result.measured_surface, `${path}.measured_surface`)
    keys(measured, `${path}.measured_surface`, ["schema","path","sha256"])
    if (measured.schema !== "tatbot.surface/1") throw new ContractError("wrong_schema", `${path}.measured_surface.schema`, String(measured.schema))
    string(measured.path, `${path}.measured_surface.path`)
    const measuredDigest = sha(measured.sha256, `${path}.measured_surface.sha256`)
    if (registrationValue.measured_surface_sha256 !== measuredDigest) {
      throw new ContractError(
        "wrong_hash",
        `${path}.surface_registration.measured_surface_sha256`,
        "does not bind the measured surface",
      )
    }

    const tool = object(result.tool, `${path}.tool`)
    keys(tool, `${path}.tool`, ["id","datasheet_sha256","class"])
    string(tool.id, `${path}.tool.id`)
    sha(tool.datasheet_sha256, `${path}.tool.datasheet_sha256`)
    const toolClass = string(tool.class, `${path}.tool.class`)
    const inkEvents = array(inkProgramValue.events, `${path}.ink_program.events`).map(
      (event, index) => object(event, `${path}.ink_program.events[${index}]`),
    )
    inkEvents.forEach((inkEvent, index) => {
      let requiredClass: unknown
      if (inkEvent.kind === "stroke") requiredClass = inkEvent.allowed_tool_class
      else if (inkEvent.kind === "tool_change") requiredClass = inkEvent.required_tool_class
      if (requiredClass !== undefined && requiredClass !== toolClass) {
        throw new ContractError(
          "execution_binding_mismatch",
          `${path}.ink_program.events[${index}]`,
          `requires tool class ${String(requiredClass)}; execution binds ${toolClass}`,
        )
      }
    })

    const robot = object(result.robot, `${path}.robot`)
    keys(robot, `${path}.robot`, ["id","urdf_sha256"])
    string(robot.id, `${path}.robot.id`)
    sha(robot.urdf_sha256, `${path}.robot.urdf_sha256`)

    const support = object(result.support, `${path}.support`)
    keys(support, `${path}.support`, ["kind","configuration_sha256"])
    string(support.kind, `${path}.support.kind`)
    sha(support.configuration_sha256, `${path}.support.configuration_sha256`)

    const palette = object(result.palette, `${path}.palette`)
    keys(palette, `${path}.palette`, ["snapshot_sha256","load_state_sha256","resolved_caps"])
    sha(palette.snapshot_sha256, `${path}.palette.snapshot_sha256`)
    sha(palette.load_state_sha256, `${path}.palette.load_state_sha256`)
    const caps = array(palette.resolved_caps, `${path}.palette.resolved_caps`)
    if (!caps.length) throw new ContractError("wrong_length", `${path}.palette.resolved_caps`, "at least one required")
    const capInks = new Set<string>()
    const capSlots = new Set<number>()
    caps.forEach((raw, index) => {
      const cap = object(raw, `${path}.palette.resolved_caps[${index}]`)
      keys(cap, `${path}.palette.resolved_caps[${index}]`, ["ink_id","slot","robot_base_from_cap"])
      const inkId = string(cap.ink_id, `${path}.palette.resolved_caps[${index}].ink_id`)
      const slot = integer(cap.slot, `${path}.palette.resolved_caps[${index}].slot`)
      if (capInks.has(inkId) || capSlots.has(slot)) {
        throw new ContractError(
          "execution_binding_mismatch",
          `${path}.palette.resolved_caps[${index}]`,
          "ink IDs and cap slots must be unique",
        )
      }
      capInks.add(inkId)
      capSlots.add(slot)
      matrix(cap.robot_base_from_cap, `${path}.palette.resolved_caps[${index}].robot_base_from_cap`)
    })
    const requiredInks = new Set<string>()
    for (const name of ["initial_ink_state", "predicted_ink_state"]) {
      const state = object(inkProgramValue[name], `${path}.ink_program.${name}`)
      requiredInks.add(string(state.ink_id, `${path}.ink_program.${name}.ink_id`))
    }
    inkEvents.forEach((inkEvent, index) => {
      if (inkEvent.kind === "dip") {
        requiredInks.add(string(inkEvent.ink_id, `${path}.ink_program.events[${index}].ink_id`))
      }
    })
    const missingInks = [...requiredInks].filter((inkId) => !capInks.has(inkId)).sort()
    if (missingInks.length) {
      throw new ContractError(
        "execution_binding_mismatch",
        `${path}.palette.resolved_caps`,
        `missing ink IDs: ${missingInks.join(", ")}`,
      )
    }

    const calibration = object(result.calibration, `${path}.calibration`)
    keys(calibration, `${path}.calibration`, ["sha256","captured_utc"])
    const calibrationDigest = sha(calibration.sha256, `${path}.calibration.sha256`)
    utcTimestamp(calibration.captured_utc, `${path}.calibration.captured_utc`)
    if (registrationValue.calibration_sha256 !== calibrationDigest) {
      throw new ContractError(
        "execution_binding_mismatch",
        `${path}.surface_registration.calibration_sha256`,
        "does not bind the execution calibration",
      )
    }

    const exactEvents = array(result.exact_events, `${path}.exact_events`)
    if (!exactEvents.length) throw new ContractError("wrong_length", `${path}.exact_events`, "at least one required")
    const actualEventIndices: number[] = []
    let previousStop = 0
    exactEvents.forEach((raw, index) => {
      const event = object(raw, `${path}.exact_events[${index}]`)
      keys(event, `${path}.exact_events[${index}]`, ["ink_event_index","kind","sample_range"])
      const inkEventIndex = integer(event.ink_event_index, `${path}.exact_events[${index}].ink_event_index`)
      const kind = string(event.kind, `${path}.exact_events[${index}].kind`)
      if (inkEventIndex >= inkEvents.length || inkEvents[inkEventIndex].kind !== kind) {
        throw new ContractError(
          "execution_binding_mismatch",
          `${path}.exact_events[${index}]`,
          "does not identify the same embedded ink event",
        )
      }
      actualEventIndices.push(inkEventIndex)
      const sampleRange = array(event.sample_range, `${path}.exact_events[${index}].sample_range`, 2).map(
        (entry, rangeIndex) => integer(entry, `${path}.exact_events[${index}].sample_range[${rangeIndex}]`),
      )
      if (sampleRange[1] <= sampleRange[0]) {
        throw new ContractError("out_of_range", `${path}.exact_events[${index}].sample_range`, "range must be nonempty")
      }
      if (index > 0 && sampleRange[0] < previousStop) {
        throw new ContractError(
          "trajectory_discontinuous",
          `${path}.exact_events[${index}].sample_range`,
          "sample ranges overlap or run backward",
        )
      }
      previousStop = sampleRange[1]
    })
    const expectedEventIndices = inkEvents.flatMap((inkEvent, index) => inkEvent.kind === "stroke" ? [index] : [])
    if (actualEventIndices.length !== expectedEventIndices.length ||
        actualEventIndices.some((eventIndex, index) => eventIndex !== expectedEventIndices[index])) {
      throw new ContractError(
        "execution_binding_mismatch",
        `${path}.exact_events`,
        "must map every stroke exactly once in ink-program order",
      )
    }

    const uncertainty = object(result.uncertainty, `${path}.uncertainty`)
    keys(uncertainty, `${path}.uncertainty`, ["registration_sigma_m","surface_sigma_m"])
    if (number(uncertainty.registration_sigma_m, `${path}.uncertainty.registration_sigma_m`) < 0 ||
        number(uncertainty.surface_sigma_m, `${path}.uncertainty.surface_sigma_m`) < 0) {
      throw new ContractError("out_of_range", `${path}.uncertainty`, "uncertainty must be nonnegative")
    }

    const preflight = object(result.preflight, `${path}.preflight`)
    keys(preflight, `${path}.preflight`, ["observed_cells_only","max_surface_age_s","policy_sha256"])
    if (boolean(preflight.observed_cells_only, `${path}.preflight.observed_cells_only`) !== true) {
      throw new ContractError("wrong_value", `${path}.preflight.observed_cells_only`, "must be true")
    }
    if (number(preflight.max_surface_age_s, `${path}.preflight.max_surface_age_s`) <= 0) {
      throw new ContractError("out_of_range", `${path}.preflight.max_surface_age_s`, "must be positive")
    }
    sha(preflight.policy_sha256, `${path}.preflight.policy_sha256`)

    const samples = object(result.samples_manifest, `${path}.samples_manifest`)
    keys(samples, `${path}.samples_manifest`, ["schema","path","sha256","sidecar_path","sidecar_sha256"])
    if (samples.schema !== "tatbot.draw-samples/1") throw new ContractError("wrong_schema", `${path}.samples_manifest.schema`, String(samples.schema))
    string(samples.path, `${path}.samples_manifest.path`)
    sha(samples.sha256, `${path}.samples_manifest.sha256`)
    string(samples.sidecar_path, `${path}.samples_manifest.sidecar_path`)
    sha(samples.sidecar_sha256, `${path}.samples_manifest.sidecar_sha256`)
    provenance(result.provenance, `${path}.provenance`)
  }
  const actual = await canonicalDigest(result)
  if (result.content_sha256 !== actual) {
    throw new ContractError("wrong_hash", `${path}.content_sha256`, `declared ${result.content_sha256}, computed ${actual}`)
  }
}

export async function validateContract(value: unknown, options: ReadOptions = {}): Promise<JsonObject> {
  const result = object(value, "$")
  const schema = string(result.schema, "$.schema")
  if (!(HUMAN_REP_SCHEMAS as readonly string[]).includes(schema)) {
    throw new ContractError("unknown_schema", "$.schema", schema)
  }
  if (options.expectedSchema && options.expectedSchema !== schema) {
    throw new ContractError("wrong_schema", "$.schema", `expected ${options.expectedSchema}, got ${schema}`)
  }
  await nested(result, schema as HumanRepSchema, "$", options)
  return result
}

export async function readContractJson(source: string, options: ReadOptions = {}): Promise<JsonObject> {
  return validateContract(parseJsonStrict(source), options)
}
