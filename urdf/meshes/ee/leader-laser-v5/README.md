# Left laser attachment (leader-laser-v5)

The cradle and clamp remain installed. The raised carrier in this directory
is historical: the robot now uses the compact rear-facing carrier in
`../fiducial-cubes-47mm-rear/`, with the same attachment placement.

Assembly-coordinate exports from `cad/leader-laser-v1/` (revision V5), in
millimetres: the cradle designed to replace the thumb-side leader finger, the
oval clamp cap sized for the handheld laser pen's 35 mm grip, and the mirrored
raised tag carrier. The meshes are authored in `left/carriage_right` (the CAD
frame, `provenance.json` `frame`), but the print is **installed on
`left/carriage_left`, rolled 180 degrees about the carriage x axis** — the
2x3 M3 pattern is symmetric under that roll, so it bolts on either way.
`urdf/tatbot.urdf` attaches the meshes under `left/carriage_left` with
`rpy="pi 0 0"` and URDF scale `0.001 0.001 0.001` (`provenance.json`
`installed_placement`). These are the assembly files, not the print
orientations. `provenance.json` binds the source and mesh hashes, the CAD
tool datum and the unassigned CAD face datums.

The placement is measured, not assumed: the 2026-09-16 wrist_left fiducial
layout (solved in `left/carriage_right`, the frozen parent of that capture,
without reference to any CAD) lands on the carrier's three faces only in that
placement, and the CAD pen axis then points straight down in every palette
contact hold of the same session. The identity placement under
`left/carriage_right` that the URDF carried until 2026-09-17 drew the mount
upside down on the wrong carriage and the pen 90 degrees off its true axis.
The operator identifies the left arm as carrying the laser and the pink-taped
right arm as carrying the Lutin; the print revision has not been re-surveyed.

The pen itself is not a mesh here: `left/tool_mount` is the fat end of the
grip cavity with +z along the pen axis toward the nose (the CAD clamp datum
carried through the installed placement; in link 6 it points the same way as
the follower's datum), and `scripts/gen_tool_urdf.py` renders the laser from
its datasheet on that frame at the measured lens-face touch-off
(`config/workspace.yaml`).

The installed compact carrier has 47 mm 36h11 IDs 2, 3 and 4
(`config/fiducials.json`, target `wrist_left`, parent `left/carriage_left`).
Its measured layout is pending recalibration. The earlier V5 carrier's
measured tag poses are historical and cannot be transferred to the new
carrier. CAD artwork does not determine marker identity; face letters are
placeholders.

These visual meshes do not supply a laser TCP, collision qualification,
optical focus or measured fiducial transforms.

To reproduce, run the existing CAD builder in an environment with CadQuery
2.8.0, trimesh and manifold3d 3.5.3:

```sh
python cad/leader-laser-v1/build.py --output /tmp/leader-laser-v5
```

These three `assembly/*.stl` files reproduce the original V5 assembly,
including its historical carrier. Never copy `print/*.stl` into the robot model: their
origins and orientations are for the printer bed.
