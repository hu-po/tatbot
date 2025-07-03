"""

https://github.com/nmwsharp/potpourri3d
https://x.com/nmwsharp/status/1940293930400326147
https://github.com/nmwsharp/vector-heat-demo
https://polyscope.run/py/
https://geometry-central.net/surface/algorithms/vector_heat_method/#logarithmic-map
https://polyscope.run/structures/point_cloud/basics/

"""
import logging
import os
from dataclasses import dataclass, field

import polyscope as ps
import polyscope.imgui as psim
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

def view_mesh(config: ViewConfig):

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

    ps.init()
    ps_pointcloud = ps.register_point_cloud("My Pointcloud", P)

    # = Stateful solves
    solver = pp3d.PointCloudHeatSolver(P)

    # Get the tangent frames which are used by the solver to define tangent data
    # at each point
    basisX, basisY, basisN = solver.get_tangent_frames()

    # State for the GUI
    @dataclass
    class AppState:
        geodesic_source_point: int = -1
        extension_source_points: list = field(default_factory=list)
        transport_source_point: int = -1
        transport_vector: list = field(default_factory=lambda: [6., 6.])
        logmap_source_point: int = -1
        curves: list = field(default_factory=lambda: [[]])
    
    state = AppState()
    ui_scale_set = False

    def callback():
        nonlocal ui_scale_set
        if not ui_scale_set:
            psim.GetIO().FontGlobalScale = 1.5
            ui_scale_set = True

        log.debug("--- GUI callback ---")

        # By default, a window will be opened for the user callback.
        # We can start adding UI elements immediately.
        
        # Geodesic Distance
        if psim.TreeNode("Geodesic Distance"):
            log.debug("Rendering 'Geodesic Distance' UI")
            if psim.Button("Set Source Point"):
                log.debug("Button 'Set Source Point' clicked")
                if ps.have_selection():
                    selection = ps.get_selection()
                    log.debug(f"Got selection: {selection}")
                    name = selection.structure_name
                    idx = selection.local_index
                    if name == ps_pointcloud.get_name():
                        state.geodesic_source_point = idx
                        log.info(f"Set geodesic source to {idx}")
                else:
                    log.warning("'Set Source Point' clicked but no selection was made.")

            psim.TextUnformatted(f"Source: {state.geodesic_source_point if state.geodesic_source_point != -1 else 'None'}")
            if psim.Button("Compute Geodesic Distance"):
                log.debug("Button 'Compute Geodesic Distance' clicked")
                if state.geodesic_source_point != -1:
                    dists = solver.compute_distance(state.geodesic_source_point)
                    ps_pointcloud.add_scalar_quantity("Geodesic Distance", dists, enabled=True)
                    ps.screenshot()
                else:
                    log.warning("No source point selected for geodesic distance.")
            psim.TreePop()
        
        # Scalar Extension
        if psim.TreeNode("Scalar Extension"):
            log.debug("Rendering 'Scalar Extension' UI")
            if psim.Button("Add Source"):
                log.debug("Button 'Add Source' clicked")
                if ps.have_selection():
                    selection = ps.get_selection()
                    log.debug(f"Got selection: {selection}")
                    name = selection.structure_name
                    idx = selection.local_index
                    if name == ps_pointcloud.get_name():
                        state.extension_source_points.append(idx)
                        log.info(f"Added extension source {idx}")
                else:
                    log.warning("'Add Source' clicked but no selection was made.")
            
            if psim.Button("Clear Sources"):
                log.debug("Button 'Clear Sources' clicked")
                state.extension_source_points.clear()
                log.info("Cleared extension sources")

            psim.TextUnformatted(f"Sources: {state.extension_source_points}")

            if psim.Button("Compute Scalar Extension"):
                log.debug("Button 'Compute Scalar Extension' clicked")
                if len(state.extension_source_points) >= 2:
                    points = state.extension_source_points
                    values = np.linspace(0., 1., len(points)).tolist()
                    
                    ext = solver.extend_scalar(points, values)
                    ps_pointcloud.add_scalar_quantity("Scalar Extension", ext, enabled=True)
                else:
                    log.warning("Need at least 2 source points for scalar extension.")
            psim.TreePop()

        # Vector Transport
        if psim.TreeNode("Vector Transport"):
            log.debug("Rendering 'Vector Transport' UI")
            if psim.Button("Set Source"):
                log.debug("Button 'Set Source' for Vector Transport clicked")
                if ps.have_selection():
                    selection = ps.get_selection()
                    log.debug(f"Got selection: {selection}")
                    name = selection.structure_name
                    idx = selection.local_index
                    if name == ps_pointcloud.get_name():
                        state.transport_source_point = idx
                        log.info(f"Set transport source to {idx}")
                else:
                    log.warning("'Set Source' clicked but no selection was made.")

            psim.TextUnformatted(f"Source: {state.transport_source_point if state.transport_source_point != -1 else 'None'}")
            
            _, state.transport_vector[0] = psim.InputFloat("vx", state.transport_vector[0])
            _, state.transport_vector[1] = psim.InputFloat("vy", state.transport_vector[1])

            if psim.Button("Compute Vector Transport"):
                log.debug("Button 'Compute Vector Transport' clicked")
                if state.transport_source_point != -1:
                    ext = solver.transport_tangent_vector(state.transport_source_point, state.transport_vector)
                    ext3D = ext[:,0,np.newaxis] * basisX +  ext[:,1,np.newaxis] * basisY
                    ps_pointcloud.add_vector_quantity("Transported Vector", ext3D, enabled=True)
                else:
                    log.warning("No source point selected for vector transport.")
            psim.TreePop()

        # Log Map
        if psim.TreeNode("Log Map"):
            log.debug("Rendering 'Log Map' UI")
            if psim.Button("Set Source##LogMap"):
                log.debug("Button 'Set Source##LogMap' clicked")
                if ps.have_selection():
                    selection = ps.get_selection()
                    log.debug(f"Got selection: {selection}")
                    name = selection.structure_name
                    idx = selection.local_index
                    if name == ps_pointcloud.get_name():
                        state.logmap_source_point = idx
                        log.info(f"Set logmap source to {idx}")
                else:
                    log.warning("'Set Source##LogMap' clicked but no selection was made.")
            
            psim.TextUnformatted(f"Source: {state.logmap_source_point if state.logmap_source_point != -1 else 'None'}")
            
            if psim.Button("Compute Log Map"):
                log.debug("Button 'Compute Log Map' clicked")
                if state.logmap_source_point != -1:
                    logmap = solver.compute_log_map(state.logmap_source_point)
                    logmap3D = logmap[:,0,np.newaxis] * basisX +  logmap[:,1,np.newaxis] * basisY
                    ps_pointcloud.add_vector_quantity("Log Map", logmap3D, enabled=True)
                else:
                    log.warning("No source point selected for log map.")
            psim.TreePop()
        
        # Signed Distance
        if psim.TreeNode("Signed Distance"):
            log.debug("Rendering 'Signed Distance' UI")
            for i, curve in enumerate(state.curves):
                psim.TextUnformatted(f"Curve {i}: {curve}")

            if psim.Button("Add Point to Last Curve"):
                log.debug("Button 'Add Point to Last Curve' clicked")
                if ps.have_selection():
                    selection = ps.get_selection()
                    log.debug(f"Got selection: {selection}")
                    name = selection.structure_name
                    idx = selection.local_index
                    if name == ps_pointcloud.get_name():
                        if not state.curves:
                            state.curves.append([])
                        state.curves[-1].append(idx)
                else:
                    log.warning("'Add Point to Last Curve' clicked but no selection was made.")

            if psim.Button("Add New Curve"):
                log.debug("Button 'Add New Curve' clicked")
                state.curves.append([])

            if psim.Button("Clear Curves"):
                log.debug("Button 'Clear Curves' clicked")
                state.curves = [[]]
            
            if psim.Button("Compute Signed Distance"):
                log.debug("Button 'Compute Signed Distance' clicked")
                valid_curves = [c for c in state.curves if c]
                if valid_curves:
                    signed_dist = solver.compute_signed_distance(valid_curves, basisN)
                    ps_pointcloud.add_scalar_quantity("Signed Distance", signed_dist, enabled=True)
                else:
                    log.warning("No valid curves to compute signed distance.")
            psim.TreePop()


    ps.set_user_callback(callback)
    ps.show()


if __name__ == "__main__":
    args = setup_log_with_config(ViewConfig)
    print_config(args)
    if args.debug:
        log.setLevel(logging.DEBUG)
    try:
        log.info("🔗 Symlink .ini files from config to current directory")
        for ini_file in args.ini_files:
            target_path = os.path.expanduser(os.path.join(args.config_dir, ini_file))
            link_path = os.path.join(os.getcwd(), ini_file)

            # Re-create symlink if it's broken or points to the wrong place
            if os.path.lexists(link_path):
                if not os.path.islink(link_path) or os.readlink(link_path) != target_path:
                    os.remove(link_path)
                    try:
                        os.symlink(target_path, link_path)
                        log.info(f"Updated symlink for {ini_file}")
                    except OSError as e:
                        log.error(f"Failed to create symlink for {ini_file}: {e}")
            else:
                try:
                    os.symlink(target_path, link_path)
                    log.info(f"Created symlink for {ini_file}")
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