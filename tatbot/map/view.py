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

from tatbot.utils.log import get_logger, print_config, setup_log_with_config

log = get_logger('map.view', "📐")

@dataclass
class ViewConfig:
    debug: bool = False
    """Enable debug logging."""

    mesh_path: str = "~/tatbot/nfs/3d/fakeskin-lowpoly/fakeskin-lowpoly.obj"
    config_dir: str = "~/tatbot/config/polyscope"
    ini_files: tuple[str] = (".polyscope.ini", "imgui.ini")
    source_vertex: int = 0


def view(config: ViewConfig):
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

if __name__ == "__main__":
    args = setup_log_with_config(ViewConfig)
    print_config(args)
    if args.debug:
        log.setLevel(logging.DEBUG)
    try:
      view(args)
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