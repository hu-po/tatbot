import logging
import os

import polyscope as ps
import potpourri3d as pp3d

from tatbot.utils.log import get_logger, print_config, setup_log_with_config
from tatbot.map.base import ViewConfig, AppState, symlink_ini_files, cleanup_ini_files, polyscope_view, remove_polyscope_ini_files

log = get_logger('map.view_ply', "📐")

def view_pointcloud(config: ViewConfig):
    if not config.pointcloud_path:
        raise ValueError("pointcloud_path must be set in ViewConfig")
    pointcloud_path = os.path.expanduser(config.pointcloud_path)
    log.info(f"Reading pointcloud from {pointcloud_path}")
    P = pp3d.read_point_cloud(pointcloud_path)

    ps.init()
    ps_pointcloud = ps.register_point_cloud("My Pointcloud", P)

    solver = pp3d.PointCloudHeatSolver(P)
    basisX, basisY, basisN = solver.get_tangent_frames()

    state = AppState()

    polyscope_view(
        state=state,
        solver=solver,
        structure=ps_pointcloud,
        basisX=basisX,
        basisY=basisY,
        basisN=basisN,
        get_name=ps_pointcloud.get_name,
        add_scalar_quantity=ps_pointcloud.add_scalar_quantity,
        add_vector_quantity=ps_pointcloud.add_vector_quantity,
        selection_label="Point",
    )

if __name__ == "__main__":
    args = setup_log_with_config(ViewConfig)
    print_config(args)
    if args.debug:
        log.setLevel(logging.DEBUG)
    try:
        symlink_ini_files(args)
        view_pointcloud(args)
    except Exception as e:
        log.error(f"❌ View Exit with Error:\n{e}")
    except KeyboardInterrupt:
        log.info("🛑⌨️ Keyboard interrupt detected")
    finally:
        log.info("🛑 Disconnecting...")
        cleanup_ini_files(args)
        remove_polyscope_ini_files(args)