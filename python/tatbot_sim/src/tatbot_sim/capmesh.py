"""An ink cap as a mesh: hollow, with a floor, and a flat flange at the rim.

A real cap is a thin-walled cup with an annular lip that rests on the
palette and a hole in the middle the needle goes down; a solid cylinder
reads as a plug. Built from three trimesh primitives (wall annulus, floor
disc, flange annulus) concatenated — no boolean needed, nothing overlaps —
and cached as OBJ per cap size. The rim is at z = 0 and the cup hangs
below it, matching the URDF's inkcap_* frames (the rim IS the frame).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

WALL_M = 0.0006
FLOOR_M = 0.0006
FLANGE_W_M = 0.0015
FLANGE_T_M = 0.0005
SECTIONS = 40


def cap_mesh_path(out_dir: Path, size_id: str, diameter_m: float, depth_m: float) -> Path:
    """Write (once) and return the OBJ for a cap of this size."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"inkcap_{size_id}_{diameter_m * 1000:.1f}x{depth_m * 1000:.1f}mm.obj"
    if path.is_file():
        return path
    import trimesh

    r_in = diameter_m / 2
    r_out = r_in + WALL_M
    wall = trimesh.creation.annulus(r_min=r_in, r_max=r_out, height=depth_m, sections=SECTIONS)
    wall.apply_translation([0, 0, -depth_m / 2])
    floor = trimesh.creation.cylinder(radius=r_out, height=FLOOR_M, sections=SECTIONS)
    floor.apply_translation([0, 0, -depth_m + FLOOR_M / 2])
    flange = trimesh.creation.annulus(r_min=r_in, r_max=r_out + FLANGE_W_M, height=FLANGE_T_M,
                                      sections=SECTIONS)
    flange.apply_translation([0, 0, -FLANGE_T_M / 2])
    cap = trimesh.util.concatenate([wall, floor, flange])
    cap.export(path)
    return path


def ink_level_z(depth_m: float, diameter_m: float, fill_ul: float) -> float:
    """Where the ink surface sits below the rim for ``fill_ul`` in this cup."""
    area = np.pi * (diameter_m / 2) ** 2
    level = -depth_m + FLOOR_M + (fill_ul * 1e-9) / area
    # never above the flange: an over-declared fill is a brimming cap, not a
    # disc hovering over the rack
    return min(-FLANGE_T_M, max(-(depth_m - FLOOR_M), level))
