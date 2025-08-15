# Vision System

The `tatbot` system uses a combination of Amcrest IP cameras for wide-angle scene coverage and Intel Realsense depth cameras mounted on the robot arms for precise 3D sensing. AprilTags are used for object tracking and calibration.

## Cameras

- `src/conf/cams/default.yaml`: The main configuration file for all cameras.
- `src/tatbot/data/cams.py`: The Pydantic data model for camera configurations.
- `src/tatbot/cam/`: The source module for all camera-related code, including calibration and capture.

### Calibration
Camera calibration is a critical step. While intrinsic parameters (focal length, sensor size) are relatively static, extrinsic parameters (the 3D pose of each camera in the world) must be calculated for the specific setup.

The `src/conf/cams/default.yaml` file contains **placeholder** extrinsics. To get real values, you must run the calibration script:
```bash
uv run python -m tatbot.cam.extrinsics
```
This script uses AprilTags to find the precise location of each camera relative to the world origin.

### IP PoE Cameras
Amcrest 5MP Turret POE Camera, UltraHD Outdoor IP Camera POE with Mic/Audio, 5-Megapixel Security Surveillance Cameras, 98ft NightVision, 132° FOV, MicroSD (256GB), (IP5M-T1179EW-AI-V3)

the cameras are currently set at:

- resolution: 1920x1080
- fps: 5
- bitrate CBR max 2048
- frameinterval: 10
- no substream, all watermarks off 

### RealSense Depth Cameras
tatbot uses two [D405 Intel Realsense cameras](https://www.intelrealsense.com/depth-camera-d405/).

- [`pyrealsense2`](https://github.com/IntelRealSense/librealsense)
- both realsense cameras are connected to `hog` via usb3 port
- Follow the [calibration guide](https://dev.intelrealsense.com/docs/self-calibration-for-depth-cameras).
- Use the `rs-enumerate-devices` command to verify that both realsenses are connected. If this doesn't work, unplug and replug the realsense cameras.
- Should be calibrated out of the box, but can be recalibrated
  - [example1](https://github.com/IntelRealSense/librealsense/blob/master/wrappers/python/examples/depth_auto_calibration_example.py)
  - [example2](https://github.com/IntelRealSense/librealsense/blob/master/wrappers/python/examples/depth_ucal_example.py)
- [FOV differs for depth and rgb cameras](https://www.intel.com/content/www/us/en/support/articles/000030385/emerging-technologies/intel-realsense-technology.html)
- TODO: these will somewhat randomly fail, need to create robust exception handling

## AprilTags

AprilTags are used to track objects (i.e. palette) in the scene and for camera calibration.

- [`pupil-apriltags`](https://github.com/pupil-labs/apriltags)
- [png generator](https://chaitanyantr.github.io/apriltag.html).
- [3d generator](https://lyehe.github.io/aruco_3d/)

see:

- `src/conf/tags/default.yaml`
- `src/tatbot/data/tags.py`
- `src/tatbot/cam/tracker.py`

## 2D to 3D Mapping

To tattoo on a non-flat surface like a practice arm, the 2D artwork must be accurately "wrapped" onto the 3D surface. This is a complex geometric problem solved using a technique called geodesic tracing.

### Implementation

The core logic for this process is in `src/tatbot/gen/map.py`. The pipeline is as follows:

1.  **3D Surface Reconstruction**: A 3D mesh of the target surface is first created from Realsense point clouds using `open3d`'s Poisson surface reconstruction (`src/tatbot/utils/plymesh.py`).
2.  **Projection**: The flat 2D points of a stroke (from G-code) are transformed into 3D space based on the desired position and orientation of the design. These 3D points are then projected to find the closest vertices on the target mesh.
3.  **Geodesic Tracing**: Instead of simply connecting the projected points with straight lines (which would go through the surface), we trace the shortest path along the surface between each point. This is known as a geodesic path. We use the [`potpourri3d`](https://github.com/nmwsharp/potpourri3d) library for its efficient `GeodesicTracer`.
4.  **Resampling & Normals**: The resulting 3D path is resampled to have a uniform density of points. At each point, the surface normal of the mesh is calculated. This normal is crucial for orienting the tattoo needle to be perpendicular to the skin.
5.  **Debugging**: The `src/tatbot/viz/map.py` tool provides an interactive visualization for debugging this entire process.

### Misc Links

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
