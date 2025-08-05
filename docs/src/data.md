# Data Module (`src/tatbot/data`)

This module defines the core data structures used throughout the `tatbot` application. It uses `pydantic` to create typed data models, ensuring that all parts of the system communicate with well-defined and validated data.

## Core Abstractions

-   **`BaseCfg`**: The foundation of all data models in this module. It provides YAML serialization and deserialization (`to_yaml`, `from_yaml`), along with helpful string representations. It also handles the conversion of `numpy` arrays.

-   **`Scene`**: The master data structure that encapsulates the entire state of a robotic task. It aggregates all other configuration objects like `Arms`, `Cams`, `Skin`, `Inks`, etc. It also loads and validates all necessary sub-configurations, such as robot poses and pen configurations, making it the central point of access for all scene-related data.

-   **`Stroke`**: Represents a single continuous motion of a robot arm. It contains the geometric data for the stroke in various coordinate systems (pixels, meters, world).
    -   **Array Handling**: To keep the main configuration files clean, large `numpy` arrays (like `meter_coords`, `ee_pos`, etc.) are saved to separate `.npy` files, and the `Stroke` object only holds file references.
    -   **`StrokeList`**: Manages a list of bimanual stroke pairs and handles the coordinated saving and loading of their associated array files.
    -   **`StrokeBatch`**: A `jax_dataclasses.pytree_dataclass` designed for batch processing of strokes, likely for machine learning or simulation purposes. It's optimized for performance and can be easily saved and loaded using `safetensors`.

## Key Data Models

-   **`Pose` (`pose.py`)**: Defines 3D position (`Pos`) and orientation (`Rot` as a quaternion). This is a fundamental building block for representing the location of objects and robot links in the world.
    -   **`ArmPose`**: A specialized pose that represents the joint angles of a robot arm.

-   **`Arms` (`arms.py`)**: Configuration for the bimanual robot arms, including IP addresses, config file paths, movement speeds, and end-effector offsets.

-   **`Cams` (`cams.py`)**: Configuration for all cameras.
    -   It defines distinct classes for different camera types (`RealSenseCameraConfig`, `IPCameraConfig`).
    -   Includes intrinsic and extrinsic camera parameters.

-   **`URDF` (`urdf.py`)**: Holds the path to the robot's URDF file and the names of important links within it (e.g., end-effectors, cameras, tags). This allows the system to be independent of hardcoded link names.

-   **`Skin` (`skin.py`)**: Defines the properties of the surface on which the robot will be working. This includes the physical dimensions of the design area and the location of a "zone" used for cropping point clouds.

-   **`Inks` (`inks.py`)**: Defines the properties of the inks and ink caps available to the robot, including their color, size, and pose in the world.

-   **`Tags` (`tags.py`)**: Configuration for the AprilTag detection system, including the tag family, size, and which specific tag IDs to look for.

-   **`Node` (`node.py`)**: Represents a computing node in the distributed `tatbot` system, with properties like IP address and user for SSH.

## How It Works and How to Use It

1.  **Configuration as Code**: The system is designed to be configured through YAML files that are parsed into these Pydantic models. This provides strong validation and makes the configuration easy to read and edit.
2.  **Centralized Scene**: The `Scene` object is the main entry point. You create a `Scene` by loading a top-level YAML file. The `Scene` model's validators will then automatically load all other required data, such as poses from the `poses` directory, pen configurations, and URDF link poses.
3.  **Data Flow**: Higher-level modules (like `ops` or `mcp`) will typically operate on a `Scene` object to get all the information they need about the robot, cameras, and the task at hand. The data objects themselves are generally passive, holding data and providing serialization/deserialization logic.
4.  **Stroke Management**: For tasks involving drawing or tracing, paths are represented as `Stroke` objects. The `StrokeList` provides a convenient way to manage sequences of strokes and their large associated data arrays.
