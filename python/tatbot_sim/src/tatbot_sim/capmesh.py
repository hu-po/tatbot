"""An ink cap as a mesh: a thin-walled cup with a floor, rim at z = 0.

Sized from the cap's specification (``ink_spec.CapSize``, config/palette.yaml):
the outside is ``diameter_m`` across the rim by ``height_m`` tall with a
``wall_m`` shell, so the bore the tool enters is ``bore_diameter_m`` and the
inside floor sits ``depth_m`` below the rim. Built from two trimesh primitives
(wall annulus, floor disc) concatenated — no boolean needed, nothing overlaps —
and cached as OBJ per cap size. The rim is at z = 0 and the cup hangs below it
to the support floor the scene seats it on (palette.py: rim = floor + outside
height).
"""

from __future__ import annotations

from pathlib import Path

SECTIONS = 40
# A brimming cap shows its surface just under the rim, not a disc hovering over the rack.
BRIM_M = 0.0005


def cap_mesh_path(out_dir: Path, size) -> Path:
    """Write (once) and return the OBJ for a cap of this ``ink_spec.CapSize``."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    dims = f"{size.diameter_m * 1000:.1f}x{size.height_m * 1000:.1f}x{size.wall_m * 1000:.2f}"
    path = out_dir / f"inkcap_{size.size_id}_{dims}mm.obj"
    if path.is_file():
        return path
    import trimesh

    r_in = size.bore_diameter_m / 2
    wall = trimesh.creation.annulus(r_min=r_in, r_max=size.diameter_m / 2, height=size.height_m,
                                    sections=SECTIONS)
    wall.apply_translation([0, 0, -size.height_m / 2])
    floor = trimesh.creation.cylinder(radius=r_in, height=size.wall_m, sections=SECTIONS)
    floor.apply_translation([0, 0, -size.height_m + size.wall_m / 2])
    trimesh.util.concatenate([wall, floor]).export(path)
    return path


def ink_level_z(size, fill_ul: float) -> float:
    """Where the ink surface sits relative to the rim for ``fill_ul`` in this
    cap: ink_spec's surface depth, the one the dip plunges below
    (ink_spec.dip_plunge_m)."""
    return -max(BRIM_M, size.surface_depth_m(fill_ul))
