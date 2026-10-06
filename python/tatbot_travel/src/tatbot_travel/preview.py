"""Repeatable camera inspections of the same world used by data generation."""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import mujoco
import numpy as np

from tatbot_travel.scene import SCENE_ONLY_GROUP


def inspect_views(run, out: Path, wrist: np.ndarray, *, geometry_only: bool = False) -> list[str]:
    """Save wrist, exterior and hand views without changing episode geometry."""
    world = run.world
    pose = run.script.pose(0.0)
    vertices = pose.apply(world.phantom.vertices)
    hand = vertices[world.phantom.vertices[:, 0] > world.phantom.forearm[1]]
    target = vertices.mean(axis=0)
    hand_target = hand.mean(axis=0) if len(hand) else target
    forward = pose.rot.apply([1, 0, 0])
    bearing = float(np.degrees(np.arctan2(forward[1], forward[0])))
    views = [("workspace", np.array([0.35, 0.0, 0.0]), 135, -55, 1.4),
             ("workspace-top", np.array([0.35, 0.0, 0.0]), 90, -89, 1.55),
             ("overview", target, bearing + 50, -35, 0.70),
             ("top", target, bearing, -89, 0.65),
             ("side", target, bearing + 90, -12, 0.60),
             ("hand-palm", hand_target, bearing + 45, -55, 0.24),
             ("hand-side", hand_target, bearing + 90, -15, 0.24),
             ("hand-end", hand_target, bearing + 180, -25, 0.24)]
    option = mujoco.MjvOption()
    option.geomgroup[SCENE_ONLY_GROUP] = 1
    original_groups = run.model.geom_group.copy()
    if geometry_only:
        for geom in range(run.model.ngeom):
            material = int(run.model.geom_matid[geom])
            name = run.model.mat(material).name if material >= 0 else ""
            if name and name.startswith("captured_room"):
                run.model.geom_group[geom] = 5
        option.geomgroup[5] = 0
    images = [("wrist", wrist)]
    renderer = mujoco.Renderer(run.model, height=480, width=640)
    try:
        for name, lookat, azimuth, elevation, distance in views:
            camera = mujoco.MjvCamera()
            camera.type = mujoco.mjtCamera.mjCAMERA_FREE
            camera.lookat[:] = lookat
            camera.azimuth, camera.elevation, camera.distance = azimuth, elevation, distance
            renderer.update_scene(run.data, camera=camera, scene_option=option)
            images.append((name, renderer.render().copy()))
    finally:
        renderer.close()
        run.model.geom_group[:] = original_groups
    tiles, paths = [], []
    for name, rgb in images:
        path = out.with_name(f"{out.name}-{name}.png")
        bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        cv2.imwrite(str(path), bgr)
        tile = cv2.resize(bgr, (480, 360))
        cv2.rectangle(tile, (0, 330), (480, 360), (20, 20, 20), -1)
        cv2.putText(tile, name, (10, 351), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
        tiles.append(tile)
        paths.append(str(path))
    tiles += [np.zeros_like(tiles[0])] * (9 - len(tiles))
    gallery = np.concatenate([np.concatenate(tiles[row * 3:row * 3 + 3], axis=1) for row in range(3)])
    cv2.imwrite(str(out.with_name(f"{out.name}-views.png")), gallery)
    np.savez_compressed(out.with_suffix(".surface.npz"), **world.ink.surface_arrays())
    meta = {"seed": run.seed, "draw": run.draw, "profile_id": run.cfg.profile_id,
            "inspection_time_s": 0.0, "geometry_only": geometry_only,
            "start_state_rad": run.history[0].tolist(),
            "phantom_pose": {"position_m": pose.pos.tolist(), "quaternion_wxyz": pose.quat_wxyz().tolist(),
                             "heading_deg": bearing}, **world.meta, "view_images": paths}
    out.with_suffix(".json").write_text(json.dumps(meta, indent=2) + "\n")
    return paths


def comparison_sheet(variants: list[Path], out: Path, *, episodes: bool = False, skins: bool = False) -> None:
    """Compare appearance alone, or full scenes from fixed workspace cameras."""
    rows, metadata = [], []
    for variant in variants:
        meta = json.loads(variant.with_suffix(".json").read_text())
        appearance = meta["appearance"]
        views = ("workspace", "workspace-top", "wrist") if episodes else ("overview", "hand-palm", "wrist")
        tiles = [cv2.imread(str(variant.with_name(f"{variant.name}-{view}.png"))) for view in views]
        row = np.concatenate([cv2.resize(tile, (480, 360)) for tile in tiles], axis=1)
        light = appearance["lighting"]
        label = (f'look {appearance["seed"]} | {appearance["workspace_surface"]["style"]} | '
                 f'light energy {light["energy"]:.2f}, warmth {light["warmth"]:+.2f}')
        if episodes:
            pose = meta["phantom_pose"]
            x, y = pose["position_m"][:2]
            items = appearance["tabletop"]
            mats, papers = sum(i["kind"] == "mat" for i in items), sum(i["kind"] == "paper" for i in items)
            layout = next((i["layout"] for i in items if i["kind"] == "paper"), "none")
            label = (f'episode {meta["seed"]} | {appearance["workspace_surface"]["style"]} | '
                     f'{mats} mats, {papers} papers ({layout}) | xy ({x:.2f}, {y:+.2f})m, heading {pose["heading_deg"]:.0f} deg')
        if skins:
            skin = appearance["skin"]
            rgb = ",".join(str(round(c)) for c in skin["rgb"])
            label = (f'skin {skin["branch"]} / {skin["seed"]} | RGB {rgb} | '
                     f'specular {skin["specular"]:.2f}, shine {skin["shininess"]:.2f} | '
                     f'mottling {skin["mottling"]:.3f}, fine {skin["fine_texture"]:.3f}')
        cv2.rectangle(row, (0, 330), (row.shape[1], 360), (20, 20, 20), -1)
        cv2.putText(row, label, (10, 352), cv2.FONT_HERSHEY_SIMPLEX, .6, (255, 255, 255), 1)
        rows.append(row)
        metadata.append(meta)
    suffix = "episodes" if episodes else ("skins" if skins else "appearances")
    cv2.imwrite(str(out.with_name(f"{out.name}-{suffix}.png")), np.concatenate(rows, axis=0))
    out.with_name(f"{out.name}-{suffix}.json").write_text(json.dumps(metadata, indent=2) + "\n")
