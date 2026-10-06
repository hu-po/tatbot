# Ordered matte vinyl fiducials

The operator supplied the [order layout image](assets/ordered-fiducial-sheet.png)
on 2026-09-09 and confirmed that the four large stickers are intended for the
palette and end effector. The installed assignment is 47 mm wrist markers and
a small 46 mm palette marker; `config/fiducials.json` owns the exact family, IDs
and sizes. The order-sheet measurements
below remain dimension evidence, not a detector calibration.

## Evidence and dimensions

- The sheet is labelled 27.4 × 27.4 cm.
- The sheet outline is approximately 1216 pixels square in the supplied image.
- Each large sticker outline is approximately 450 pixels square, giving
  274 × 450 / 1216 = 101.4 mm including its white border.
- Its black square is approximately 360 pixels wide, giving 81.1 mm.
- These are image-derived estimates, consistent with a nominal 4-inch sticker;
  the order's exact physical dimensions and manufacturing tolerance are unknown.
- OpenCV's AprilTag 36h11 dictionary decodes the large stickers as IDs 20, 21,
  22 and 23. The 16h5 dictionary detects none. The small stickers decode as
  36h11 IDs 0–5 and 30. The installed palette uses small ID 0. The pink wrist
  uses small IDs 2, 3, 4; the blue wrist uses 5, 30, 1. No large sticker is
  installed.

The compact wrist carriers provide three 68 mm faces for 64 mm white stickers
with 47 mm black squares. The palette's diamond-shaped base seat is 60 mm square and carries
a small sticker. Detector sizes use the installed dimensions.

## Current compatibility

The installed calibration palette v11 reuses small ID 0 on its 60 mm
base-mounted diamond seat. The 46 mm black edge is the retained sticker's
inventory value. The old v10 pose does not transfer: the new printed-pattern
orientation and station placement need confirmation and a fresh ROS fix.
See [palette](palette.md).

The compact rear-facing revision-3 wrist carriers replace the large V17/V5
carrier panels. Their assembly meshes and nominal face datums are retained under
`urdf/meshes/ee/fiducial-cubes-47mm-rear/`. The original tool cradles remain.

The installed `config/fiducials.json` is authoritative. Both wrist layouts were
measured again after the replacement; nominal CAD faces do not qualify a
tracking layout. The replacement family is 36h11, including palette ID 0.
