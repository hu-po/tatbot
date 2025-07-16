import logging
import os

import numpy as np
import polyscope as ps
import potpourri3d as pp3d
import trimesh
from PIL import Image

from tatbot.map.base import (
    AppState,
    ViewConfig,
    cleanup_ini_files,
    polyscope_view,
    remove_polyscope_ini_files,
    symlink_ini_files,
)
from tatbot.utils.log import get_logger, print_config, setup_log_with_config

log = get_logger('map.view_obj', "📦")

def view_obj(config: ViewConfig):
    if not config.mesh_path:
        raise ValueError("mesh_path must be set in ViewConfig")
    mesh_path = os.path.expanduser(config.mesh_path)
    log.info(f"Reading mesh from {mesh_path}")

    # Load mesh with trimesh to get UVs and texture
    mesh = trimesh.load(mesh_path, force='mesh')
    if not isinstance(mesh, trimesh.Trimesh):
        raise ValueError("Loaded mesh is not a trimesh.Trimesh instance")

    # Optionally use the trimesh viewer for true texture visualization
    if getattr(config, 'use_trimesh_viewer', False):
        log.info("Launching trimesh viewer for true texture visualization.")
        scene = trimesh.Scene(mesh)
        scene.show()
        return

    V = mesh.vertices
    F = mesh.faces

    # Try to get UVs
    uvs = None
    if hasattr(mesh.visual, 'uv') and mesh.visual.uv is not None:
        uvs = mesh.visual.uv
        log.info(f"Loaded {uvs.shape[0]} UV coordinates")
    else:
        log.warning("No UV coordinates found in mesh.")

    # Try to get texture image
    texture_img = None
    if hasattr(mesh.visual, 'material') and hasattr(mesh.visual.material, 'image') and mesh.visual.material.image is not None:
        texture_img = mesh.visual.material.image
        log.info(f"Loaded texture image from material: {texture_img.size}")
        texture_np = np.array(texture_img.convert('RGB')) / 255.0  # shape (H, W, 3), float32
    else:
        log.warning("No texture image found in mesh material.")
        texture_np = None

    # Fallback: try to find texture image in the same directory as the OBJ
    if texture_np is None and hasattr(mesh.visual, 'material') and hasattr(mesh.visual.material, 'image_path'):
        tex_path = mesh.visual.material.image_path
        if tex_path and os.path.exists(tex_path):
            texture_img = Image.open(tex_path)
            texture_np = np.array(texture_img.convert('RGB')) / 255.0
            log.info(f"Loaded texture image from path: {tex_path}")
        else:
            log.warning(f"Texture image path not found or does not exist: {tex_path}")

    # Load mesh for geometry processing (potpourri3d)
    V_pp3d, F_pp3d = pp3d.read_mesh(mesh_path)

    ps.init()
    ps_mesh = ps.register_surface_mesh("My Skin Mesh", V_pp3d, F_pp3d)

    # Add UVs and texture if available
    baked_texture = False
    if uvs is not None and texture_np is not None:
        # Bake texture to per-vertex colors using UVs
        try:
            H, W, _ = texture_np.shape
            u = np.clip(uvs[:, 0], 0, 1)
            v = np.clip(uvs[:, 1], 0, 1)
            v = 1.0 - v  # OBJ UV origin is bottom-left, image is top-left
            x = (u * (W - 1)).astype(int)
            y = (v * (H - 1)).astype(int)
            vertex_colors = texture_np[y, x]  # shape (n_vertices, 3)
            ps_mesh.add_vertex_color_quantity("texture", vertex_colors)
            log.info("Baked texture to vertex colors and added to Polyscope mesh.")
            baked_texture = True
        except Exception as e:
            log.warning(f"Failed to bake texture to vertex colors: {e}")
    else:
        log.warning("Skipping texture baking: missing UVs or texture image.")

    solver = pp3d.MeshVectorHeatSolver(V_pp3d, F_pp3d)
    basisX, basisY, basisN = solver.get_tangent_frames()

    state = AppState()

    polyscope_view(
        state=state,
        solver=solver,
        structure=ps_mesh,
        basisX=basisX,
        basisY=basisY,
        basisN=basisN,
        get_name=ps_mesh.get_name,
        add_scalar_quantity=ps_mesh.add_scalar_quantity,
        add_vector_quantity=ps_mesh.add_vector_quantity,
        selection_label="Vertex",
    )

if __name__ == "__main__":
    args = setup_log_with_config(ViewConfig)
    print_config(args)
    if args.debug:
        log.setLevel(logging.DEBUG)
    try:
        symlink_ini_files(args)
        view_obj(args)
    except Exception as e:
        log.error(f"❌ View Exit with Error:\n{e}")
    except KeyboardInterrupt:
        log.info("🛑⌨️ Keyboard interrupt detected")
    finally:
        log.info("🛑 Disconnecting...")
        cleanup_ini_files(args)
        remove_polyscope_ini_files(args)