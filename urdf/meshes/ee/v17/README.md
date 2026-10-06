# V17 cradle and clamp source meshes

The operator reported successful installation and fit on 2026-09-07. These
three STL files are the unchanged **assembly-coordinate** meshes of the printed
V17 package, not its rotated/translated bed meshes. Units are millimetres;
identity origin under `right/gripper_left`, URDF scale `0.001 0.001 0.001`.
`provenance.json` pins each mesh's SHA256 and the unassigned CAD face datums.

The cradle and clamp remain installed; the raised carrier is historical.
The real URDF imports the compact rear-facing carrier from
`../fiducial-cubes-47mm-rear/` alongside those two retained visuals. The
carriage screw pattern and fixed camera mounts are unchanged. The nominal tool
axis and CAD tip are preserved: carriage tip `(115.539, -11.521, -0.564)` mm.
`right/tool_mount` is now the first clamp bore band, station 73 mm behind that
tip; its +z points toward the tip. The datasheet uses the same datum.

The pink/right compact carrier carries 47 mm 36h11 IDs 2/3/4. The blue/left
arm carries IDs 5/30/1 and the palette carries 81 mm ID 22, all from `config/fiducials.json`.
CAD artwork was placeholder artwork and does not determine marker identity.
Fresh calibration must assign measured tag poses. Both wrist layout records
are pending with no transforms; existing tool touch-offs and unchanged palette
contact registration are retained. No collision mass,
inertia, contact accuracy or motion acceptance is claimed by these visual meshes.

Run the normal wrist calibration workflow before tracking the new stickers.

The adjacent `../pen_tip_v17.stl` is a nominal cartridge frustum in metres:
32 mm base diameter, 2 mm tip diameter, 35 mm length, 96 radial segments.
It replaces the old 20 mm-wide display cone; it is not a printed component.

## Ordered replacement stickers

The palette uses a large matte vinyl sticker and the wrists use smaller
stickers. Their dimensions and decoded family/IDs are recorded in the shared
[ordered sticker specification](../../../../docs/ordered_fiducials.md).
