from tatbot.utils.log import get_logger

log = get_logger('map.wrap')

# https://github.com/nmwsharp/potpourri3d
# https://x.com/nmwsharp/status/1940293930400326147

# https://github.com/nmwsharp/vector-heat-demo

# https://polyscope.run/py/

# https://geometry-central.net/surface/algorithms/vector_heat_method/#logarithmic-map

import potpourri3d as pp3d
import polyscope as ps
import numpy as np

# --- 1. & 2. Load mesh and compute with potpourri3d ---
# For this example, we'll generate a sample mesh
V, F = pp3d.read_mesh("path/to/your/mesh.obj") 

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