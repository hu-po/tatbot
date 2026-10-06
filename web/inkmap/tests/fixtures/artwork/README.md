# Synthetic artwork acceptance fixtures

Original Tatbot test artwork, authored 2026-09-05; CC0-1.0. These five files
are fixed regression inputs, not additions to the visitor picker. Keep their
bytes stable once an evidence manifest records their SHA-256 hashes.

| File | Expected appearance and semantics |
| --- | --- |
| `linework.svg` | Two open lines, 1 and 3 canvas units wide, round caps/joins, translated group, 80×50 canvas |
| `blackwork.svg` | Solid rotated five-sided black polygon; its interior must be painted |
| `negative-space.svg` | Annulus with transparent center; even-odd compound path, not a white circle or two filled disks |
| `stipple.svg` | Nine filled dots with five radii, separated by transparent negative space |
| `color-layers.svg` | Red rounded rectangle behind a blue circle; mirrored group; blue wins at overlap |

Physical fixture dimensions in millimeters equal the viewBox dimensions.
Mirroring and additional placement rotation are independent scenario variants.
Malformed XML, external references, filters, text, unknown paint operations,
hash mismatches, and unsupported transformations are rejection cases in the
artwork contract tests, not acceptable approximations of these fixtures.
