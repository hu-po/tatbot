"""

https://github.com/nmwsharp/potpourri3d
https://x.com/nmwsharp/status/1940293930400326147
https://github.com/nmwsharp/vector-heat-demo
https://polyscope.run/py/
https://geometry-central.net/surface/algorithms/vector_heat_method/#logarithmic-map

"""
import logging
import os
from dataclasses import dataclass

import polyscope as ps
import potpourri3d as pp3d
import numpy as np

from tatbot.utils.log import get_logger, print_config, setup_log_with_config

log = get_logger('map.view', "📐")

@dataclass
class ViewConfig:
    debug: bool = False
    """Enable debug logging."""

    config_dir: str = "~/tatbot/config/polyscope"
    """Directory containing the polyscope config files."""
    ini_files: tuple[str] = (".polyscope.ini", "imgui.ini")
    """Polyscope config files."""

    mesh_path: str = "~/tatbot/nfs/3d/fakeskin-lowpoly/fakeskin-lowpoly.obj"
    """Path to the mesh file."""
    pointcloud_path: str = "~/tatbot/nfs/3d/skin.ply"
    """Path to the pointcloud file."""
    source_vertex: int = 0
    """Source vertex for the geodesic distance."""

def view_mesh(config: ViewConfig):
    config_dir = os.path.expanduser(config.config_dir)

    for ini_file in config.ini_files:
        target_path = os.path.join(config_dir, ini_file)
        link_path = os.path.join(os.getcwd(), ini_file)
        # Remove existing file if not already symlink
        if os.path.exists(link_path) and not os.path.islink(link_path):
            os.remove(link_path)
        # Create symlink if missing or incorrect
        if not os.path.exists(link_path):
            try:
                os.symlink(target_path, link_path)
            except OSError as e:
                log.error(f"Failed to create symlink for {ini_file}: {e}")

    mesh_path = os.path.expanduser(config.mesh_path)
    log.info(f"Reading mesh from {mesh_path}")
    V, F = pp3d.read_mesh(mesh_path) 

    # Compute geodesic distance from vertex 0
    dists = pp3d.compute_distance(V, F, 0)

    # --- 3. Visualize with Polyscope ---
    ps.init()

    # Register the mesh
    ps_mesh = ps.register_surface_mesh("My Skin Mesh", V, F)

    # Add the computed distance as a scalar quantity to the mesh
    ps_mesh.add_scalar_quantity("Geodesic Distance", dists, enabled=True)

    # Show the interactive viewer
    ps.show()

    # solver = pp3d.MeshVectorHeatSolver(V,F)

    # # Extend the value `0.` from vertex 12 and `1.` from vertex 17. Any vertex 
    # # geodesically closer to 12. will take the value 0., and vice versa 
    # # (plus some slight smoothing)
    # ext = solver.extend_scalar([12, 17], [0.,1.])

    # # Get the tangent frames which are used by the solver to define tangent data
    # # at each vertex
    # basisX, basisY, basisN = solver.get_tangent_frames()

    # # Parallel transport a vector along the surface
    # # (and map it to a vector in 3D)
    # sourceV = 22
    # ext = solver.transport_tangent_vector(sourceV, [6., 6.])
    # ext3D = ext[:,0,np.newaxis] * basisX +  ext[:,1,np.newaxis] * basisY

    # # Compute the logarithmic map
    # logmap = solver.compute_log_map(sourceV)

    # MeshVectorHeatSolver.compute_log_map(v_ind, strategy='AffineLocal')

def view_pointcloud(config: ViewConfig):
    
    pointcloud_path = os.path.expanduser(config.pointcloud_path)
    log.info(f"Reading pointcloud from {pointcloud_path}")
    P = pp3d.read_point_cloud(pointcloud_path)

    # = Stateful solves
    solver = pp3d.PointCloudHeatSolver(P)

    # Compute the geodesic distance to point 4
    dists = solver.compute_distance(4)

    # Extend the value `0.` from point 12 and `1.` from point 17. Any point 
    # geodesically closer to 12. will take the value 0., and vice versa 
    # (plus some slight smoothing)
    ext = solver.extend_scalar([12, 17], [0.,1.])

    # Get the tangent frames which are used by the solver to define tangent data
    # at each point
    basisX, basisY, basisN = solver.get_tangent_frames()

    # Parallel transport a vector along the surface
    # (and map it to a vector in 3D)
    sourceP = 22
    ext = solver.transport_tangent_vector(sourceP, [6., 6.])
    ext3D = ext[:,0,np.newaxis] * basisX +  ext[:,1,np.newaxis] * basisY

    # Compute the logarithmic map
    logmap = solver.compute_log_map(sourceP)

    # Signed distance to the oriented curve(s) denoted by a point sequence.
    curves = [
            [9, 10, 12, 13, 51, 48], 
            [79, 93, 12, 30, 78, 18, 92], 
            [90, 84, 19, 91, 82, 81, 83]
            ]
    signed_dist = solver.compute_signed_distance(curves, basisN)

    ps.init()
    ps_pointcloud = ps.register_point_cloud("My Pointcloud", P)
    ps_pointcloud.add_scalar_quantity("Geodesic Distance", dists, enabled=True)
    ps.show()


if __name__ == "__main__":
    args = setup_log_with_config(ViewConfig)
    print_config(args)
    if args.debug:
        log.setLevel(logging.DEBUG)
    try:
        log.info("🔗 Symlink .ini files from config to current directory")
        for ini_file in args.ini_files:
            target_path = os.path.join(args.config_dir, ini_file)
            link_path = os.path.join(os.getcwd(), ini_file)
            # Remove existing file if not already symlink
            if os.path.exists(link_path) and not os.path.islink(link_path):
                os.remove(link_path)
            # Create symlink if missing or incorrect
            if not os.path.exists(link_path):
                try:
                    os.symlink(target_path, link_path)
                except OSError as e:
                    log.error(f"Failed to create symlink for {ini_file}: {e}")
        view_pointcloud(args)
    except Exception as e:
        log.error(f"❌ View Exit with Error:\n{e}")
    except KeyboardInterrupt:
        log.info("🛑⌨️ Keyboard interrupt detected")
    finally:
        log.info("🛑 Disconnecting...")
        # remove the symlinks
        for ini_file in args.ini_files:
            link_path = os.path.join(os.getcwd(), ini_file)
            if os.path.exists(link_path):
                log.info(f"🗑️ Removing symlink {link_path}")
                os.remove(link_path)