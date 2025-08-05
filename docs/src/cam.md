# Camera Module (`src/tatbot/cam`)

This module is responsible for handling all camera-related operations, including intrinsic and extrinsic calibration, AprilTag tracking, and capturing point cloud data.

## Core Abstractions

-   **`Intrinsics`**: A data structure representing the internal parameters of a camera, such as focal length and principal point.
-   **`Pose`**: Represents the 3D position and orientation of an object, used for camera extrinsics and tag poses.
-   **`TagTracker`**: A class that detects AprilTags in images and calculates their 3D poses.
-   **`DepthCamera`**: A class to interface with a RealSense depth camera to capture and process point cloud data.

## Key Files and Functionality

### `intrinsics_rs.py`

-   **Purpose**: Manages RealSense camera intrinsics.
-   **Functionality**:
    -   Detects connected RealSense devices and their serial numbers.
    -   `print_intrinsics`: Connects to a specified RealSense camera and prints its intrinsic parameters.
    -   `match_realsense_devices`: Compares connected hardware with cameras defined in the configuration files.
-   **Usage**: Run as a script to query and verify the intrinsics of connected RealSense cameras.

### `intrinsics_ip.py`

-   **Purpose**: A placeholder for IP camera intrinsic calibration.
-   **Functionality**: Currently, this is a TODO and does not implement intrinsic calculation. It is intended to house the logic for calibrating IP cameras, possibly using a checkerboard pattern.

### `extrinsics.py`

-   **Purpose**: Calculates the extrinsic parameters (position and orientation) of all cameras in the system.
-   **`get_extrinsics`**: The core function that implements a multi-camera, multi-tag optimization process:
    1.  It uses `TagTracker` to detect AprilTags in a set of images from different cameras.
    2.  It then iteratively refines the camera extrinsics by minimizing the reprojection error of the detected tags. It computes an average world pose for each tag and then updates each camera's pose to better align with these averages.
    3.  One camera is treated as the reference (anchor), and its pose is fixed to the world origin.
-   **Usage**: This is a key calibration step to determine where all the cameras are in relation to each other.

### `tracker.py`

-   **Purpose**: Detects and tracks AprilTags.
-   **`TagTracker`**:
    -   Initializes an AprilTag detector.
    -   `track_tags`: Takes an image and camera intrinsics, detects tags, and calculates their 6-DOF pose in the world frame. It can also draw the detections on the image for visualization.

### `depth.py`

-   **Purpose**: Interfaces with RealSense cameras to capture depth data.
-   **`DepthCamera`**:
    -   Initializes a connection to a RealSense camera.
    -   `get_pointcloud`: Captures color and depth frames, applies filters (decimation, clipping), and generates a 3D point cloud. The points are transformed into the world coordinate frame using the camera's extrinsic pose. It can also save the point cloud as a PLY file.

## How It Works and How to Use It

1.  **Intrinsic Calibration**: First, ensure the intrinsic parameters for each camera are known. For RealSense cameras, `intrinsics_rs.py` can be used to query these values. For IP cameras, a procedure would need to be implemented in `intrinsics_ip.py`.
2.  **Extrinsic Calibration**: Once intrinsics are set, run the `get_extrinsics` function from `extrinsics.py` with a series of images from all cameras viewing known AprilTags. This will compute the 3D pose of each camera.
3.  **Tracking and Perception**: With calibrated cameras, you can use `TagTracker` to find objects in the world or `DepthCamera` to capture 3D information about the scene.
