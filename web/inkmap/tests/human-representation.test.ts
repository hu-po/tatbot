import assert from "node:assert/strict"
import { readFileSync } from "node:fs"
import { test } from "node:test"

import {
  ContractError,
  canonicalDigest,
  canonicalJson,
  parseJsonStrict,
  readContractJson,
} from "../src/core/human-representation/schema.ts"
import { validatePlacementFile } from "../src/core/schema.ts"

const root = new URL("../../../config/human-representation/examples/", import.meta.url)
const fixture = JSON.parse(readFileSync(new URL("fixture-set.json", root), "utf8"))
const topology = "e0ca7ee25dc0b4c8d841bb2626e364bb88b7af7fae037e30854728842e320a18"

test("the complete fixture chain matches the Python canonical digest goldens", async () => {
  for (const entry of fixture.files) {
    const source = readFileSync(new URL(entry.path, root), "utf8")
    const parsed = await readContractJson(source, {
      expectedSchema: entry.schema,
      expectedTopologySha256: topology,
    })
    assert.equal(await canonicalDigest(parsed), entry.content_sha256)
    assert.deepEqual(
      await readContractJson(canonicalJson(parsed), {
        expectedSchema: entry.schema,
        expectedTopologySha256: topology,
      }),
      parsed,
    )
  }

  const artwork = JSON.parse(readFileSync(new URL("artwork-matrix.json", root), "utf8"))
  const elements = artwork.layers.flatMap((layer: { elements: Record<string, unknown>[] }) => layer.elements)
  assert.deepEqual(
    new Set(elements.map((element: { kind: string }) => element.kind)),
    new Set(["path", "cubic_bezier", "region", "stipple"]),
  )
  const cubic = elements.find((element: { kind: string }) => element.kind === "cubic_bezier") as {
    control_points_m: unknown[]
  } | undefined
  assert.equal(cubic?.control_points_m.length, 4)
  assert.ok(artwork.negative_space_masks.length)
  assert.ok(artwork.inks.length > 1)
})

test("canonical numbers have one cross-language spelling", () => {
  for (const [value, canonical] of [
    [1.0, "1"],
    [1e-7, "1e-7"],
    [1e-6, "0.000001"],
    [1e20, "100000000000000000000"],
    [1e21, "1e+21"],
  ] as const) {
    assert.equal(canonicalJson(value), canonical)
  }
  assert.throws(
    () => parseJsonStrict('{"value":9007199254740992}'),
    (error: unknown) => error instanceof ContractError && error.code === "unsafe_integer",
  )
})

test("strict JSON accepts only RFC whitespace and preserves prototype-like own keys", () => {
  assert.throws(
    () => parseJsonStrict('\u00a0{"value":1}'),
    (error: unknown) => error instanceof ContractError && error.code === "invalid_json",
  )
  const parsed = parseJsonStrict('{"__proto__":{"hidden":true}}') as Record<string, unknown>
  assert.deepEqual(Object.keys(parsed), ["__proto__"])
  assert.equal(Object.hasOwn(parsed, "__proto__"), true)
  assert.equal(Object.getPrototypeOf(parsed), Object.prototype)
  assert.equal(canonicalJson(parsed), '{"__proto__":{"hidden":true}}')
})

test("canonical keys use UTF-16 order and nested execution objects are closed", async () => {
  assert.equal(canonicalJson({ "\ue000": 1, "\u{10000}": 2 }), '{"𐀀":2,"":1}')
  const execution = JSON.parse(readFileSync(new URL("execution-program.json", root), "utf8"))
  execution.tool.unexpected = true
  await assert.rejects(
    readContractJson(JSON.stringify(execution)),
    (error: unknown) => error instanceof ContractError && error.code === "unknown_field",
  )
})

test("body states select one named or tracked pose source", async () => {
  const bodyState = JSON.parse(readFileSync(new URL("body-state.json", root), "utf8"))
  bodyState.named_pose = null
  bodyState.tracked_source = {
    tracker: "synthetic-fixture",
    sample_time_utc: "2026-09-04T01:09:37Z",
    capture_sha256: "7".repeat(64),
    source_frame: "body_tracker",
  }
  bodyState.content_sha256 = await canonicalDigest(bodyState)
  await assert.doesNotReject(readContractJson(JSON.stringify(bodyState)))

  bodyState.named_pose = "ambiguous"
  bodyState.content_sha256 = await canonicalDigest(bodyState)
  await assert.rejects(
    readContractJson(JSON.stringify(bodyState)),
    (error: unknown) => error instanceof ContractError && error.code === "wrong_pose_source",
  )
})

test("readers enforce the pinned mid topology, supported domain, and correctives mode", async () => {
  const cases: [string, (value: Record<string, any>) => void, string][] = [
    ["surface-coordinate.json", (value) => { value.face_index = 36_108 }, "out_of_range"],
    ["surface-curve.json", (value) => { value.coordinates[1].topology_sha256 = "1".repeat(64) }, "wrong_topology"],
    ["surface-placement.json", (value) => {
      value.supported_domain.face_indices = [value.anchor.face_index + 1]
    }, "anchor_outside_domain"],
    ["body-state.json", (value) => { value.correctives_enabled = false }, "wrong_value"],
    ["tattoo-program.json", (value) => { value.layers[0].elements = [] }, "wrong_length"],
  ]
  for (const [name, mutate, code] of cases) {
    const value = JSON.parse(readFileSync(new URL(name, root), "utf8"))
    mutate(value)
    value.content_sha256 = await canonicalDigest(value)
    await assert.rejects(
      readContractJson(JSON.stringify(value)),
      (error: unknown) => error instanceof ContractError && error.code === code,
    )
  }
})

test("ink and execution readers enforce envelopes, derived totals, and session bindings", async () => {
  const inkCases: [(value: Record<string, any>) => void, string][] = [
    [(value) => { value.events[1].contact_envelope_m = [0.001, -0.001] }, "out_of_range"],
    [(value) => { value.events[3].target_load = [0.9, 0.7] }, "out_of_range"],
    [(value) => { value.total_material_path_length_m = 0.02 }, "wrong_value"],
  ]
  for (const [mutate, code] of inkCases) {
    const value = JSON.parse(readFileSync(new URL("ink-program.json", root), "utf8"))
    mutate(value)
    value.content_sha256 = await canonicalDigest(value)
    await assert.rejects(
      readContractJson(JSON.stringify(value)),
      (error: unknown) => error instanceof ContractError && error.code === code,
    )
  }

  const executionCases: ((value: Record<string, any>) => void)[] = [
    (value) => { value.calibration.sha256 = "1".repeat(64) },
    (value) => { value.tool.class = "marker" },
    (value) => { value.palette.resolved_caps[0].ink_id = "blue" },
    (value) => { value.exact_events[0].ink_event_index = 0 },
  ]
  for (const mutate of executionCases) {
    const value = JSON.parse(readFileSync(new URL("execution-program.json", root), "utf8"))
    mutate(value)
    value.content_sha256 = await canonicalDigest(value)
    await assert.rejects(
      readContractJson(JSON.stringify(value)),
      (error: unknown) => error instanceof ContractError && error.code === "execution_binding_mismatch",
    )
  }
})

test("tool changes require a tool class and transition intent", async () => {
  const inkProgram = JSON.parse(readFileSync(new URL("ink-program.json", root), "utf8"))
  const toolChange = inkProgram.events.find((event: { kind: string }) => event.kind === "tool_change")
  delete toolChange.required_tool_class
  inkProgram.content_sha256 = await canonicalDigest(inkProgram)
  await assert.rejects(
    readContractJson(JSON.stringify(inkProgram)),
    (error: unknown) => error instanceof ContractError && error.code === "missing_field",
  )
})

test("duplicate keys, non-finite numbers, and negative zero fail before validation", () => {
  for (const [name, code] of [
    ["duplicate-key.json", "duplicate_key"],
    ["nonfinite.json", "non_finite"],
    ["negative-zero.json", "negative_zero"],
    ["lone-surrogate.json", "invalid_json"],
    ["malformed.json", "invalid_json"],
  ]) {
    assert.throws(
      () => parseJsonStrict(readFileSync(new URL(`invalid/${name}`, root), "utf8")),
      (error: unknown) => error instanceof ContractError && error.code === code,
    )
  }
})

test("unknown fields, units, frames, topology, and hashes are refused by name", async () => {
  for (const [name, code] of [
    ["unknown-field.json", "unknown_field"],
    ["wrong-unit.json", "unknown_field"],
    ["wrong-frame.json", "wrong_frame"],
    ["wrong-topology.json", "wrong_topology"],
    ["wrong-hash.json", "wrong_hash"],
    ["unsupported-artwork.json", "unsupported_element"],
  ]) {
    await assert.rejects(
      readContractJson(readFileSync(new URL(`invalid/${name}`, root), "utf8"), {
        expectedTopologySha256: topology,
      }),
      (error: unknown) => error instanceof ContractError && error.code === code,
    )
  }
})

test("only the SOMA placement v5 contract is readable", () => {
  const current = JSON.parse(readFileSync(
    new URL("../../../config/inkmap/examples/forearm-placement-v6.json", import.meta.url),
    "utf8",
  ))
  assert.doesNotThrow(() => validatePlacementFile(current))
  for (const version of [1, 2, 3, 4]) {
    assert.throws(
      () => validatePlacementFile({ schema_version: version }),
      /unsupported schema\/model/,
    )
  }
})
