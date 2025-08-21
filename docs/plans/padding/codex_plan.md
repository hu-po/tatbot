# Distance-Based, Padded Strokes — Consolidated Plan

Goal: Adopt distance-based resampling with padding so each stroke’s point spacing is consistent in meters while preserving GPU batch efficiency. Support only two modes: legacy fixed-count and explicit distance-based sampling.

## Scope & Assumptions
- Preserve backward compatibility: legacy fixed-count remains available.
- GPU IK requires uniform tensor shapes; padding provides this without resampling to a fixed count.
- Hover behavior: padded tail should be hover, not “stuck” at last contact.

## Current Status (Implemented)
- G-code resampling supports both fixed-count and explicit distance spacing.
- Distance mode pads to design-wide max length (bounded by `stroke_max_points`) and updates `scene.stroke_length`.
- Hover application on padded tail; depth/EE offsets unchanged.
- Config includes `stroke_sample_mode` and `stroke_point_spacing_m`.

## Improvements After Reviewing Other Plans
- Sampling modes: `stroke_sample_mode: fixed | distance` (no auto mode).
- Limits: add `stroke_max_points` to cap padding length and avoid excessive memory in distance mode.
- Metadata: plan for `actual_points`, `pad_mask`, `arc_length_m`, `point_spacing_m` on `Stroke`.
- Execution: plan to skip padded indices during robot execution based on `pad_mask`/`actual_points`.
- Mapping: preserve spacing through surface mapping (3D arc-length), then re-pad.
- Viz: show padded vs. real points and spacing/length stats.

## Detailed Design
-- Config
  - Add to Scene (backward compatible):
    - `stroke_sample_mode`: default `fixed`.
    - `stroke_point_spacing_m`: meters; used when mode is `distance`.
    - `stroke_max_points`: hard cap for padded length; default 256 (distance mode).
- Resampling
  - If mode `fixed`: use legacy fixed-count.
  - If mode `distance`: use explicit spacing; variable length per stroke.
- Padding
  - For distance mode: set `scene.stroke_length = min(max_len, stroke_max_points)` where `max_len` is the design max.
  - For fixed mode: do not change `scene.stroke_length`.
  - Populate `actual_points` and `pad_mask` in `Stroke` (future patch) to enable skipping padded steps.
- Hover behavior
  - First index hover as before.
  - Padded tail `[n_real..l-1]` moved to last-real + hover; when no padding, only last index hovered.
- Surface mapping
  - Map to 3D; when in distance mode, resample by 3D arc-length using spacing; re-pad to `scene.stroke_length`.

## Implementation Steps
1) Config wiring
   - Extend `Scene` with the sampling mode and limits (non-breaking defaults).
   - Thread mode/spacing into G-code and mapping stages.
2) Stroke metadata (future)
   - Add `actual_points: int`, `pad_mask: np.ndarray[bool]`, `arc_length_m: float`, `point_spacing_m: float` into `Stroke`.
   - Ensure YAML I/O persists these fields or re-computable where large.
3) Execution
   - Update robot stroke loop to stop at `actual_points - 1` (or mask) instead of full padded length when desirable.
   - Keep dataset shape uniform; optionally mark padded frames.
4) Visualization
   - Display counts and padding; highlight non-padded region differently.
5) Surface mapping
   - Update `map.py` to preserve spacing on 3D arc-length and then pad (distance mode).

## Validation & Safety
- Validate `stroke_point_spacing_m` > 0; apply optional guardrails in configs if needed.
- Enforce `stroke_max_points` cap to prevent OOM.
- Degenerate strokes -> ensure min 2 points before padding.
- Verify shapes across tools (viz, GPU convert, robot tool) remain consistent.

## Test Plan
- Unit-like checks on spacing function: expected count ≈ len/spacing + 1.
- End-to-end on a multi-stroke design: `scene.stroke_length` equals min(max_len, cap) and matches `strokebatch.joints.shape[1]`.
- Viz sanity: slider bounds follow padded length; padding highlighted.
- Robot dry-run: confirm padded tail hovers; optional skip padded points in loop.

## Future Work
- Masked compute: optional mask for IK/execution to reduce compute on padded indices.
- Velocity profile based on per-step distance to bound EE velocity.
- Persist chosen spacing and caps in outputs if needed.
---
orphan: true
---
