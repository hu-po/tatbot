# 2D to 3D Mapping

To tattoo on a non-flat surface like a practice arm, the 2D artwork must be accurately "wrapped" onto the 3D surface. This is a complex geometric problem solved using a technique called geodesic tracing.

## Implementation

The core logic for this process is in `src/tatbot/gen/map.py`. The pipeline is as follows:

1.  **3D Surface Reconstruction**: A 3D mesh of the target surface is first created from Realsense point clouds using `open3d`'s Poisson surface reconstruction (`src/tatbot/utils/plymesh.py`).
2.  **Projection**: The flat 2D points of a stroke (from G-code) are transformed into 3D space based on the desired position and orientation of the design. These 3D points are then projected to find the closest vertices on the target mesh.
3.  **Geodesic Tracing**: Instead of simply connecting the projected points with straight lines (which would go *through* the surface), we trace the shortest path *along the surface* between each point. This is known as a geodesic path. We use the [`potpourri3d`](https://github.com/nmwsharp/potpourri3d) library for its efficient `GeodesicTracer`.
4.  **Resampling & Normals**: The resulting 3D path is resampled to have a uniform density of points. At each point, the surface normal of the mesh is calculated. This normal is crucial for orienting the tattoo needle to be perpendicular to the skin.
5.  **Debugging**: The `src/tatbot/viz/map.py` tool provides an interactive visualization for debugging this entire process.

## Misc Links

- https://github.com/nmwsharp/potpourri3d
- https://x.com/nmwsharp/status/1940293930400326147
- https://github.com/nmwsharp/vector-heat-demo
- https://polyscope.run/py/
- https://geometry-central.net/surface/algorithms/vector_heat_method/#logarithmic-map
- https://polyscope.run/structures/point_cloud/basics/
- https://github.com/DanuserLab/u-unwrap3D
- https://github.com/mhogg/pygeodesic
- https://www.open3d.org/docs/latest/tutorial/Advanced/surface_reconstruction.html#Poisson-surface-reconstruction
- https://github.com/nmwsharp/potpourri3d?tab=readme-ov-file#mesh-geodesic-paths
- https://www.open3d.org/docs/release/tutorial/geometry/mesh.html#