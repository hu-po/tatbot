# tatbot_surface

Torch-free readers for the one reference body surface and the records placed on
it. numpy and scipy only, plus `tatbot-contracts`; no torch, shapely, physics
engine, renderer or Node bridge.

| Module | Reads |
|---|---|
| `gltf_surface` | the checked-in SOMA mid GLB as canonical faces and vertices, digest-checked |
| `rig` | `load_body_rig()`: the rest surface plus the baked named poses; `PosedBody.points(face, barycentric)` |
| `surface_trace` | unfolds a patch around an anchor, checks a design footprint, compiles metric strokes to face/barycentric anchors |
| `contracts` | the v6 placement file and the shared body-binding and anchor checks |
| `program_contract` | one complete `tatbot.tattoo-program/1` |
| `artwork` | one artwork record: its program and the shared envelope |

Paths resolve against `TATBOT_REPO`, else this checkout. Traces keep the
compiler id `tatbot_sim.surface_trace` they were first recorded under.

Tests: `python/tatbot_surface/tests`, run by `scripts/check surface` (fast
tier, numpy and scipy only) and the `surface` CI suite.
