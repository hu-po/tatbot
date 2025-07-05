import logging
import os

import polyscope as ps
import potpourri3d as pp3d

from tatbot.utils.log import get_logger, print_config, setup_log_with_config
from tatbot.map.base import ViewConfig, AppState, symlink_ini_files, cleanup_ini_files, polyscope_view, remove_polyscope_ini_files

log = get_logger('map.view_obj', "📐")

def view_obj(config: ViewConfig):
    if not config.mesh_path:
        raise ValueError("mesh_path must be set in ViewConfig")
    mesh_path = os.path.expanduser(config.mesh_path)
    log.info(f"Reading mesh from {mesh_path}")
    V, F = pp3d.read_mesh(mesh_path)

    ps.init()
    ps_mesh = ps.register_surface_mesh("My Skin Mesh", V, F)

    solver = pp3d.MeshVectorHeatSolver(V, F)
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