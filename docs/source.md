# Source Code Documentation

This document provides a comprehensive overview of the `tatbot` source code structure, organized by module. Each module serves a specific purpose in the robotic tattoo system, from hardware control to data generation and visualization.

## Table of Contents

- [Main Scripts](#main-scripts-srctatbot)
- [Bot Module](#bot-module-srctatbotbot)
- [Camera Module](#camera-module-srctatbotcam)
- [Data Module](#data-module-srctatbotdata)
- [Generation Module](#generation-module-srctatbotgen)
- [MCP Module](#mcp-module-srctatbotmcp)
- [Operations Module](#operations-module-srctatbotops)
- [Utilities Module](#utilities-module-srctatbotutils)
- [Visualization Module](#visualization-module-srctatbotviz)

---

## Main Scripts (`src/tatbot/`)

This document describes the main entry points and configuration schema files located directly in the `src/tatbot` directory. These files are central to the application's configuration loading and validation process, bridging the gap between raw YAML files and the strongly-typed Pydantic data models used throughout the system.

### Core Abstractions

-   **Hydra-Powered Configuration**: The application uses the [Hydra](https://hydra.cc/) library for managing its complex configuration. This allows for a modular approach where different parts of the configuration (arms, cameras, scenes, etc.) are defined in separate YAML files and composed at runtime.
-   **Pydantic Schema Validation**: While Hydra is excellent for composing configurations, Pydantic is used to provide an additional layer of validation and type safety. The raw, dictionary-like configuration object from Hydra is parsed into a Pydantic model, which ensures that all required fields are present and have the correct types.

### Key Files and Functionality

#### `main.py`

-   **Purpose**: Serves as the primary entry point for loading and validating the application's configuration. It demonstrates how to use Hydra to compose a full `Scene` object.
-   **`main` function**:
    -   This is a `@hydra.main`-decorated function, which means it will be the entry point when the script is run.
    -   It receives the composed `cfg: DictConfig` object from Hydra.
    -   Its main job is to call `load_scene_from_config` to parse and validate this configuration.
    -   It then prints a summary of the loaded scene to confirm that everything was loaded correctly.
    -   Now includes specific exception handling for `ConfigurationError` with descriptive error messages.
-   **`compose_and_validate_scene`**:
    -   This is a crucial utility function used by many other modules (like `ops` and `viz`) to load a specific `Scene` configuration by name.
    -   It handles the complexity of initializing Hydra if it hasn't been started yet (e.g., when running a visualization script) or using the existing Hydra context if it has (e.g., when called from an MCP server).
    -   It uses Hydra's `compose` API to load the base configuration and then apply an override to select a specific scene (e.g., `scenes=tatbotlogo`).
-   **Configuration**: Uses `AppConstants` class for centralized management of configuration paths and default values.

#### `config_schema.py`

-   **Purpose**: Defines the top-level Pydantic model for the entire application configuration.
-   **`AppConfig`**:
    -   This Pydantic model mirrors the structure of the composed Hydra configuration (`config.yaml`). It expects fields like `arms`, `cams`, `scenes`, etc., which correspond to the different configuration groups.
    -   **Type Safety**: All configuration fields now use proper type hints (`Dict[str, Any]`) for better validation and development experience.
    -   **`create_scene` validator**: This is the most important part of the file. It's a `@model_validator` that runs after the initial fields have been parsed. Its job is to take the raw dictionary configurations for each component (e.g., `self.arms`, which is a `Dict[str, Any]`) and instantiate them into their corresponding Pydantic objects (e.g., `Arms(**self.arms)`).
    -   Finally, it assembles all these component objects into a single, fully-validated `Scene` object.

### How It Works and How to Use It

1.  **Hydra Composition**: When any part of the application that uses Hydra starts (e.g., running `main.py` or the MCP server), Hydra reads the `conf/config.yaml` file. This file tells Hydra to look for configurations in various subdirectories (`arms`, `cams`, `scenes`, etc.).
2.  **Scene Selection**: The `scenes` group is special. The `config.yaml` is set up to default to `scenes/default.yaml`, but this can be overridden from the command line or programmatically. The selected scene file (e.g., `scenes/tatbotlogo.yaml`) contains the high-level parameters for a specific task.
3.  **Pydantic Validation**: The composed `DictConfig` object from Hydra is then passed to the `AppConfig` model in `config_schema.py`.
4.  **Object Instantiation**: The `create_scene` validator in `AppConfig` takes over, systematically building the final `Scene` object by instantiating all of its dependencies (`Arms`, `Cams`, `Skin`, etc.) from the raw configuration data.
5.  **Ready to Use**: The result is a single, easy-to-use, and fully-validated `Scene` object that can be passed to the `ops`, `viz`, or other modules to perform their tasks. The `compose_and_validate_scene` function in `main.py` is the standard way to get access to this object.

---

## Bot Module (`src/tatbot/bot`)

This module is responsible for controlling the robot's hardware, specifically the Trossen arms. It handles configuration, homing, and URDF-based kinematic calculations.

### Core Abstractions

-   **`TrossenConfig`**: A data class that holds configuration parameters for the Trossen arms, such as IP addresses, config file paths, and test poses.
-   **`yourdfpy.URDF` and `pyroki.Robot`**: These are used to load and represent the robot's model from a URDF file, enabling forward kinematics calculations.

### Key Files and Functionality

#### `trossen_config.py`

-   **Purpose**: Manages the configuration of the Trossen arms using YAML files.
-   **`driver_from_arms`**: A key function that takes an `Arms` configuration object and returns a pair of configured `trossen_arm.TrossenArmDriver` instances, one for each arm.
-   **`configure_arm`**: A function to configure a single arm, load its settings from a file, and run a test to verify the connection and pose control.
-   **Execution**: When run as a script, it configures both the left and right arms based on the provided arguments.

#### `trossen_homing.py`

-   **Purpose**: Provides a step-by-step, interactive process for calibrating and homing a Trossen arm.
-   **Process**: The script guides the user through a series of physical and software steps to ensure the arm is correctly positioned and calibrated. This includes:
    1.  Placing the arm in calibration jigs.
    2.  Setting the home position via a direct TCP socket command.
    3.  Verifying joint positions.
    4.  Rebooting the controller.
    5.  Testing gravity compensation.
-   **Usage**: This is intended to be run as a standalone script when setting up a new arm or when re-calibration is necessary.

#### `urdf.py`

-   **Purpose**: Handles URDF loading and forward kinematics.
-   **`load_robot`**: Caches and loads a URDF file into `yourdfpy` and `pyroki` robot objects.
-   **`get_link_indices`**: Retrieves the numerical indices for given link names from the URDF.
-   **`get_link_poses`**: Calculates the poses (position and rotation) of specified links given a set of joint positions, using forward kinematics.

### How It Works and How to Use It

1.  **Configuration**: The `trossen_config.py` script is the primary entry point for setting up the arms. It reads configuration from YAML files and establishes a connection with the arm controllers.
2.  **Homing**: Before the arms can be used, they must be homed. The `trossen_homing.py` script provides a guided process for this critical step.
3.  **Kinematics**: The `urdf.py` module is used by other parts of the system to understand the robot's physical structure and to calculate the positions of its various parts.

This module encapsulates the low-level hardware control and setup for the Trossen arms, providing a foundation for higher-level operations.

---

## Camera Module (`src/tatbot/cam`)

This module is responsible for handling all camera-related operations, including intrinsic and extrinsic calibration, AprilTag tracking, and capturing point cloud data.

### Core Abstractions

-   **`Intrinsics`**: A data structure representing the internal parameters of a camera, such as focal length and principal point.
-   **`Pose`**: Represents the 3D position and orientation of an object, used for camera extrinsics and tag poses.
-   **`TagTracker`**: A class that detects AprilTags in images and calculates their 3D poses.
-   **`DepthCamera`**: A class to interface with a RealSense depth camera to capture and process point cloud data.

### Key Files and Functionality

#### `intrinsics_rs.py`

-   **Purpose**: Manages RealSense camera intrinsics.
-   **Functionality**:
    -   Detects connected RealSense devices and their serial numbers.
    -   `print_intrinsics`: Connects to a specified RealSense camera and prints its intrinsic parameters.
    -   `match_realsense_devices`: Compares connected hardware with cameras defined in the configuration files.
-   **Usage**: Run as a script to query and verify the intrinsics of connected RealSense cameras.

#### `intrinsics_ip.py`

-   **Purpose**: A placeholder for IP camera intrinsic calibration.
-   **Functionality**: Currently, this is a TODO and does not implement intrinsic calculation. It is intended to house the logic for calibrating IP cameras, possibly using a checkerboard pattern.

#### `extrinsics.py`

-   **Purpose**: Calculates the extrinsic parameters (position and orientation) of all cameras in the system.
-   **`get_extrinsics`**: The core function that implements a multi-camera, multi-tag optimization process:
    1.  It uses `TagTracker` to detect AprilTags in a set of images from different cameras.
    2.  It then iteratively refines the camera extrinsics by minimizing the reprojection error of the detected tags. It computes an average world pose for each tag and then updates each camera's pose to better align with these averages.
    3.  One camera is treated as the reference (anchor), and its pose is fixed to the world origin.
-   **Usage**: This is a key calibration step to determine where all the cameras are in relation to each other.

#### `tracker.py`

-   **Purpose**: Detects and tracks AprilTags.
-   **`TagTracker`**:
    -   Initializes an AprilTag detector.
    -   `track_tags`: Takes an image and camera intrinsics, detects tags, and calculates their 6-DOF pose in the world frame. It can also draw the detections on the image for visualization.

#### `depth.py`

-   **Purpose**: Interfaces with RealSense cameras to capture depth data.
-   **`DepthCamera`**:
    -   Initializes a connection to a RealSense camera.
    -   `get_pointcloud`: Captures color and depth frames, applies filters (decimation, clipping), and generates a 3D point cloud. The points are transformed into the world coordinate frame using the camera's extrinsic pose. It can also save the point cloud as a PLY file.

### How It Works and How to Use It

1.  **Intrinsic Calibration**: First, ensure the intrinsic parameters for each camera are known. For RealSense cameras, `intrinsics_rs.py` can be used to query these values. For IP cameras, a procedure would need to be implemented in `intrinsics_ip.py`.
2.  **Extrinsic Calibration**: Once intrinsics are set, run the `get_extrinsics` function from `extrinsics.py` with a series of images from all cameras viewing known AprilTags. This will compute the 3D pose of each camera.
3.  **Tracking and Perception**: With calibrated cameras, you can use `TagTracker` to find objects in the world or `DepthCamera` to capture 3D information about the scene.

---

## Data Module (`src/tatbot/data`)

This module defines the core data structures used throughout the `tatbot` application. It uses `pydantic` to create typed data models, ensuring that all parts of the system communicate with well-defined and validated data.

### Core Abstractions

-   **`BaseCfg`**: The foundation of all data models in this module. It provides YAML serialization and deserialization (`to_yaml`, `from_yaml`), along with helpful string representations. It also handles the conversion of `numpy` arrays.

-   **`Scene`**: The master data structure that encapsulates the entire state of a robotic task. It aggregates all other configuration objects like `Arms`, `Cams`, `Skin`, `Inks`, etc. It also loads and validates all necessary sub-configurations, such as robot poses and pen configurations, making it the central point of access for all scene-related data.

-   **`Stroke`**: Represents a single continuous motion of a robot arm. It contains the geometric data for the stroke in various coordinate systems (pixels, meters, world).
    -   **Array Handling**: To keep the main configuration files clean, large `numpy` arrays (like `meter_coords`, `ee_pos`, etc.) are saved to separate `.npy` files, and the `Stroke` object only holds file references.
    -   **`StrokeList`**: Manages a list of bimanual stroke pairs and handles the coordinated saving and loading of their associated array files.
    -   **`StrokeBatch`**: A `jax_dataclasses.pytree_dataclass` designed for batch processing of strokes, likely for machine learning or simulation purposes. It's optimized for performance and can be easily saved and loaded using `safetensors`.

### Key Data Models

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

### How It Works and How to Use It

1.  **Configuration as Code**: The system is designed to be configured through YAML files that are parsed into these Pydantic models. This provides strong validation and makes the configuration easy to read and edit.
2.  **Centralized Scene**: The `Scene` object is the main entry point. You create a `Scene` by loading a top-level YAML file. The `Scene` model's validators will then automatically load all other required data, such as poses from the `poses` directory, pen configurations, and URDF link poses.
3.  **Data Flow**: Higher-level modules (like `ops` or `mcp`) will typically operate on a `Scene` object to get all the information they need about the robot, cameras, and the task at hand. The data objects themselves are generally passive, holding data and providing serialization/deserialization logic.
4.  **Stroke Management**: For tasks involving drawing or tracing, paths are represented as `Stroke` objects. The `StrokeList` provides a convenient way to manage sequences of strokes and their large associated data arrays.

---

## Generation Module (`src/tatbot/gen`)

This module is responsible for generating the data required for the robot to perform its tasks. It takes high-level representations, such as G-code or simple alignment descriptions, and converts them into concrete `Stroke` and `StrokeBatch` objects that contain the precise end-effector poses and joint configurations needed for execution.

### Core Abstractions

-   **Pipeline Philosophy**: The modules here form a processing pipeline. You start with a high-level description (G-code, alignment sequence), which is then progressively transformed into more detailed and hardware-specific data.
    -   `gcode.py`/`align.py` -> `map.py` -> `batch.py` -> `ik.py`
-   **`StrokeList` as intermediate representation**: The `StrokeList` is the common data structure that is passed between the different generation steps.

### Key Files and Functionality

#### `gcode.py`

-   **Purpose**: To parse `.gcode` files and convert them into a `StrokeList`.
-   **`parse_gcode_file`**: Reads a G-code file, where `G0` commands indicate "pen up" (a new stroke) and `G1` commands indicate "pen down" (part of the current stroke). It converts the G-code's millimeter coordinates into meter coordinates within the design frame. Each stroke is resampled to a uniform length (`scene.stroke_length`).
-   **`make_gcode_strokes`**: This is the main entry point. It finds all `.gcode` files in the scene's design directory, parses them, and then orchestrates the sequence of strokes for both arms. It intelligently interleaves drawing strokes with ink-dipping strokes, ensuring an arm gets new ink before it starts a new path. It also generates preview images for each stroke.

#### `align.py`

-   **Purpose**: To generate a predefined sequence of strokes used for visual alignment and calibration.
-   **`make_align_strokes`**: Creates a `StrokeList` for a sequence of simple movements, such as hovering over the calibration point and ink caps. This is useful for verifying the robot's setup before running a real task.

#### `inkdip.py`

-   **Purpose**: To create the specific trajectory for dipping a pen into an ink cap.
-   **`make_inkdip_func`**: A factory function that returns a new function for generating inkdip strokes. The returned function is cached and, when called, creates a `Stroke` that moves the end-effector down into the ink cap, waits, and then moves back up.

#### `map.py`

-   **Purpose**: To project the flat, 2D strokes generated from G-code onto a 3D mesh of the skin.
-   **`map_strokes_to_mesh`**: This is a sophisticated function that "wraps" the 2D design onto the 3D surface. It uses `potpourri3d` to trace geodesic paths (the shortest path along the surface) between points of the stroke. This ensures that the drawing appears correct on the curved surface. It also calculates the surface normals at each point of the new 3D stroke, which is crucial for orienting the pen correctly.

#### `ik.py`

-   **Purpose**: To perform inverse kinematics (IK), which is the process of calculating the required robot joint angles to achieve a desired end-effector pose.
-   **`ik`**: A JAX-jitted function that solves the IK problem for a single pose using the `pyroki` library. It's a complex optimization problem that tries to match the target pose while respecting joint limits and staying close to a "rest" pose.
-   **`batch_ik`**: A `vmap`'d version of the `ik` function that can solve for a whole batch of poses very efficiently on a GPU.

#### `batch.py`

-   **Purpose**: To convert a `StrokeList` into a `StrokeBatch`. This is the final step in the generation pipeline.
-   **`strokebatch_from_strokes`**: This function takes the `StrokeList` (which by this point contains the 3D, mesh-mapped end-effector poses) and:
    1.  Applies various offsets (hovering, depth, and general end-effector slop).
    2.  Calls `batch_ik` to calculate the joint angles for every single point in every stroke for both arms.
    3.  Packages everything into a `StrokeBatch` object, which is a PyTree of JAX arrays, ready for efficient execution or simulation.

### How It Works and How to Use It

1.  **Start with a design**: Create a G-code file representing your 2D design.
2.  **Generate Strokes**: Use `make_gcode_strokes` to parse the G-code and create an initial `StrokeList`. This list will represent the sequence of drawing and ink-dipping actions.
3.  **Map to Surface**: If you are working on a 3D surface, pass the `StrokeList` to `map_strokes_to_mesh` along with the mesh data. This will update the strokes with 3D end-effector positions and surface normals.
4.  **Create Batch**: Finally, pass the (potentially mapped) `StrokeList` to `strokebatch_from_strokes`. This will perform the final IK calculations and produce a `StrokeBatch` that contains all the data needed to execute the trajectory on the real robot.
5.  **Alignment**: Separately, you can use `make_align_strokes` to generate a simple set of strokes for calibration and verification, which can also be processed through the `batch.py` module.

---

## MCP Module (`src/tatbot/mcp`)

This module implements the server for the **M**ulti-agent **C**ommand **P**latform (MCP). It acts as the primary API endpoint for controlling the `tatbot` system. It's built on the `mcp-server` library and uses Hydra for configuration. The server exposes a set of "tools" (RPC-like functions) that can be called by clients to perform various operations.

### Core Abstractions

-   **`FastMCP`**: The underlying server implementation from the `mcp-server` package. It handles the low-level communication protocol.
-   **Tools as Functions**: The server's functionality is exposed as a series of functions (tools). Each tool is a Python function decorated to register it with the MCP server. This makes it easy to add new capabilities.
-   **Hydra for Configuration**: The entire server, including which tools to enable and network settings, is configured via Hydra. This allows for flexible deployments and easy management of different setups (e.g., a "head" node vs. an "arm" node).
-   **Pydantic Models**: All tool inputs and outputs are defined using Pydantic models. This ensures that all communication with the server is strongly typed and validated, reducing errors and making the API self-documenting.

### Key Files and Functionality

#### `server.py`

-   **Purpose**: The main entry point for the MCP server.
-   **`main`**: The Hydra-decorated main function. It parses the configuration, creates a `FastMCP` instance, registers the appropriate tools, and starts the server.
-   **`_register_tools`**: A helper function that dynamically registers tool functions from `handlers.py` with the MCP instance. Tools are registered with their original names, and node differentiation is handled through the MCP server configuration where each node has a unique server name.

#### `handlers.py`

-   **Purpose**: Contains the implementation of the actual tool functions that the server exposes.
-   **`@mcp_handler`**: A decorator used to register a function as an available tool in a central registry.
-   **Key Tools**:
    -   `run_op`: The most important tool. It executes a high-level operation (from the `tatbot.ops` module), such as `stroke` or `align`. It's an `async` generator, allowing it to stream progress and log messages back to the client as the operation runs. Now includes robust error handling with specific exception types.
    -   `ping_nodes`: Checks the network connectivity of other nodes.
    -   `list_scenes`, `list_nodes`: Tools for discovering available scenes and nodes, which is very useful for UIs and command-line clients.
-   **Error Handling**: Implements specific exception handling for different failure modes (configuration errors, network issues, file operations, etc.) with appropriate error messages and recovery strategies.

#### `models.py`

-   **Purpose**: Defines all the Pydantic models for the requests and responses of the tools in `handlers.py`.
-   **Input Models** (e.g., `RunOpInput`): Define the expected parameters for a tool call. They include validators to ensure, for example, that a requested scene or operation actually exists before the handler logic is even run.
-   **Response Models** (e.g., `RunOpResult`): Define the structure of the data that a tool will return.
-   **Configuration Constants**: Uses `MCPConstants` class for default values (host, port, transport, etc.) instead of hardcoded values, improving maintainability.
-   **`NumpyEncoder`**: A custom JSON encoder is provided to handle the serialization of `numpy` arrays, which are common in the data structures but not natively supported by JSON.

#### `__init__.py`

-   This file makes the `handlers` and `models` easily importable and also uses `__all__` to explicitly define the public API of the module, which primarily consists of the tool handler functions.

### How It Works and How to Use It

1.  **Launch the Server**: The server is started by running `python -m tatbot.mcp.server`. Hydra takes over and loads the configuration from the `conf` directory.
2.  **Configuration**: The behavior of the server is controlled by `conf/mcp/default.yaml`. You can specify the host, port, and, most importantly, which `tools` should be enabled for this specific server instance.
3.  **Client Interaction**: A client (like a UI or a script) connects to the server's host and port. It can then call the registered tools by name, passing parameters as a JSON object that conforms to the corresponding input model in `models.py`.
4.  **Running an Operation**: To run a robot task, a client would call the `run_op` tool, specifying an `op_name` and a `scene_name`. The server then instantiates the appropriate `Op` from the `tatbot.ops` module and executes it, streaming back progress.
5.  **Distributed System**: In a typical setup, you would run an MCP server on each node of the `tatbot` system (e.g., one on the main "head" computer, one on each arm controller). Tools are registered with their original names (e.g., `run_op`) and nodes are distinguished by their MCP server names in the client configuration (e.g., `tatbot.ook`, `tatbot.oop`), allowing you to call the same tool on different nodes through their respective servers.

---

## Operations Module (`src/tatbot/ops`)

This module defines high-level, stateful operations that the robot can perform. Each "Op" is a class that encapsulates the logic for a specific task, such as recording a dataset, running a stroking motion, or resetting the robot. These operations are the main entry points for the MCP server's `run_op` tool.

### Core Abstractions

-   **`BaseOp`**: The abstract base class for all operations. It provides a common structure, including:
    -   An `__init__` method that loads the specified `scene` configuration.
    -   A `run` method, which is an `async` generator. This allows an operation to be long-running and stream progress updates and log messages back to the client.
    -   A `cleanup` method for any necessary teardown logic.

-   **`RecordOp`**: A specialized base class that inherits from `BaseOp` and adds functionality for recording data into a `LeRobotDataset`. It handles the creation of the dataset, connecting to the robot, and saving the final data. Most other operations inherit from this class.

-   **Op Configuration**: Each operation has a corresponding Pydantic/dataclass model for its configuration (e.g., `StrokeOpConfig`). This allows for type-safe and validated configuration for each operation.

### Key Files and Functionality

#### `__init__.py`

-   **Purpose**: Acts as a factory and registry for all available operations.
-   **`NODE_AVAILABLE_OPS`**: A dictionary that maps a node name (e.g., "ook", "trossen-ai") to a list of operation names that are permitted to run on that node. This provides a simple but effective way to control which hardware can perform which tasks.
-   **`get_op`**: A factory function that takes an `op_name` and `node_name`, validates that the operation is allowed on that node, and returns the corresponding `Op` class and its configuration class.

#### `base.py`

-   **Purpose**: Defines the `BaseOp` class, which all other operations inherit from.

#### `record.py`

-   **Purpose**: Defines the `RecordOp` class, which provides the core functionality for recording `LeRobotDataset`s. It handles:
    -   Creating the dataset directory structure.
    -   Initializing the `LeRobotDataset` object.
    -   Connecting to the robot hardware via the `lerobot` library.
    -   The main `run` loop, which calls a `_run` method (to be implemented by subclasses) and then handles the final saving and optional pushing of the dataset to the Hugging Face Hub.

#### `reset.py`

-   **`ResetOp`**: A very simple operation to reset the robot. It connects to the robot and then immediately disconnects, which typically causes the `lerobot` driver to send the arms to their home/sleep position.

#### `record_align.py`

-   **`AlignOp`**: An operation for running a pre-defined alignment sequence.
-   **Functionality**:
    1.  It uses `make_align_strokes` from the `gen` module to create a `StrokeList`.
    2.  It converts this into a `StrokeBatch` using `strokebatch_from_strokes`.
    3.  It then iterates through this alignment sequence, moving the robot to each pose.
    4.  Because it inherits from `RecordOp`, it also records the data from this alignment procedure into a `LeRobotDataset`.

#### `record_sense.py`

-   **`SenseOp`**: An operation for capturing sensory data from the environment, specifically images and point clouds.
-   **Functionality**:
    1.  It uses the `lerobot` interface to capture standard RGB images from all configured cameras.
    2.  It then disconnects the `lerobot` camera interfaces and uses the `tatbot.cam.DepthCamera` class to capture and save a series of high-resolution point clouds (PLY files).

#### `record_stroke.py`

-   **`StrokeOp`**: The most complex operation, responsible for executing a drawing/stroking task based on G-code.
-   **Functionality**:
    1.  **Stroke Generation**: If not resuming, it calls `make_gcode_strokes` and `strokebatch_from_strokes` to generate the full `StrokeBatch` for the scene's design.
    2.  **Teleoperation**: It connects to a gamepad (`AtariTeleoperator`) which allows a human operator to make real-time adjustments to the needle depth (`offset_idx_l`/`r`) during the operation.
    3.  **Execution Loop**: It iterates through every stroke and every pose within each stroke, sending the calculated joint angles to the robot.
    4.  **Recording**: As it executes the motion, it records the robot's state (observations and actions) into a `LeRobotDataset`, creating one "episode" per stroke. It also saves detailed logs for each episode.

### How It Works and How to Use It

1.  A client calls the `run_op` tool on the MCP server, providing an `op_name` (e.g., "stroke") and a `scene_name`.
2.  The `mcp.handlers.run_op` function calls `ops.get_op` to get the correct `Op` class (e.g., `StrokeOp`).
3.  An instance of `StrokeOp` is created. Its `__init__` method loads the full `Scene` configuration.
4.  The `run` method of the `StrokeOp` instance is called.
5.  The operation proceeds with its logic, for example, generating strokes from G-code, and then iterating through them, sending commands to the robot.
6.  Throughout the process, it `yield`s progress dictionaries, which are streamed back to the client.
7.  Because it inherits from `RecordOp`, all the data generated during the operation is saved into a `LeRobotDataset` for later use in training a policy.

---

## Utilities Module (`src/tatbot/utils`)

This module provides a collection of helper functions and classes that are used across the entire `tatbot` application. These utilities cover a range of functionalities, from logging and network management to color conversion and data validation.

### Key Files and Functionality

#### `log.py`

-   **Purpose**: To provide a standardized logging setup for the application.
-   **`get_logger`**: The core function that creates and configures a `logging.Logger` instance. It adds a custom formatter that includes an emoji, making the logs easy to read and identify the source module.
-   **`setup_log_with_config`**: A helper function that initializes the logging system based on command-line arguments parsed by `tyro`. It sets the global logging level (e.g., to `DEBUG` if `--debug` is passed) and can enable debug logging for specific submodules.

#### `net.py`

-   **Purpose**: To manage all network-related tasks, particularly SSH connections to the various nodes in the `tatbot` system.
-   **`NetworkManager`**: A comprehensive class that handles:
    -   Loading node configurations from `nodes.yaml`.
    -   Generating and distributing SSH keys to all nodes to enable passwordless login.
    -   Writing a local `~/.ssh/config` file to make it easy to connect to nodes by their names (e.g., `ssh ook`).
    -   Testing the connectivity to all nodes in parallel.
-   **Usage**: The `setup_network` method is the main entry point for a first-time setup. The `NetworkManager` is also used by the `mcp.handlers.ping_nodes` tool.

#### `mode_toggle.py`

-   **Purpose**: To switch the network configuration of the entire `tatbot` system between two modes: "home" and "edge". This is a highly specialized utility for managing DNS settings in a specific network environment.
-   **`NetworkToggler`**:
    -   **Home Mode**: Assumes a local DNS server is running on `rpi1`. It configures all other nodes and the robot arms to use `rpi1` as their DNS server.
    -   **Edge Mode**: Configures all nodes to use the main LAN router for DNS.
    -   It works by SSHing into each node and modifying system configuration files (`/etc/dhcpcd.conf`).

#### `plymesh.py`

-   **Purpose**: Provides powerful utilities for working with 3D point clouds and meshes, primarily using the `Open3D` library.
-   **`create_mesh_from_ply_files`**: This is the main function. It takes one or more PLY point cloud files and performs a series of operations to create a clean, watertight 3D mesh:
    1.  Loads and merges the point clouds.
    2.  Optionally clips the point cloud to a specific "zone".
    3.  Cleans the point cloud using voxel downsampling and outlier removal.
    4.  Uses Poisson surface reconstruction to create an initial mesh.
    5.  Trims vertices from low-density areas of the mesh.
    6.  Applies Laplacian smoothing to reduce noise.
    7.  Performs a final cleaning pass to ensure the mesh is manifold and has no degenerate faces.
-   **`save_ply`, `load_ply`**: Helper functions for saving and loading point clouds.

#### `colors.py`

-   **Purpose**: Defines a dictionary of standard color names to BGR tuples and provides a color conversion utility.
-   **`COLORS`**: A dictionary of human-readable color names (e.g., "red", "blue") mapped to their `(B, G, R)` color values, suitable for use with OpenCV.
-   **`argb_to_bgr`**: A function to convert a color from a single 32-bit ARGB integer (as used in some design software) to a BGR tuple.

#### `validation.py` & `jnp_types.py`

-   **Purpose**: Provide small, focused utility functions for data validation and type conversion.
-   **`expand_user_path`**: A simple helper to expand the `~` in a path string.
-   **`ensure_numpy_array`**: Converts JAX arrays into NumPy arrays, which is often necessary before serialization or when interfacing with libraries that don't directly support JAX arrays.

---

## Visualization Module (`src/tatbot/viz`)

This module provides a suite of interactive 3D visualization tools for the `tatbot` system, built on the `viser` library. These tools are invaluable for debugging, calibration, and understanding the complex geometric data and robot motions involved in the tasks. Each visualization is a standalone application that can be run from the command line.

### Core Abstractions

-   **`viser.ViserServer`**: The core component from the `viser` library that runs a WebSocket server, allowing real-time, interactive 3D scenes to be viewed in a web browser.
-   **`BaseViz`**: An abstract base class for all visualization applications in this module. It handles the common setup tasks:
    -   Initializing a `ViserServer`.
    -   Loading a `Scene` configuration.
    -   Loading the robot's URDF model and displaying it in the scene using `ViserUrdf`.
    -   Creating GUI elements for displaying joint values.
    -   Optionally connecting to the real robot hardware to mirror the visualized state.
    -   Displaying camera frustums and other static scene elements.
-   **Interactive GUIs**: Each visualization uses `viser`'s GUI components (sliders, buttons, text boxes) to allow for real-time interaction with the scene and the underlying data.

### Key Files and Functionality

#### `base.py`

-   **Purpose**: Defines the `BaseViz` class, which provides the foundational structure for all other visualization tools.
-   **`run` loop**: The `BaseViz` class has a main `run` loop that continuously updates the visualization. In each iteration, it:
    1.  Updates the visualized robot's joint positions.
    2.  If connected to the real robot, sends the current joint positions to the hardware.
    3.  Updates the poses of camera frustums based on the robot's kinematics.
    4.  If depth cameras are enabled, captures and displays point clouds.
    5.  Calls a `step` method, which is intended to be overridden by subclasses to implement specific animation or interaction logic.

#### `stroke.py`

-   **`VizStroke`**: A tool for visualizing the execution of a `StrokeBatch`.
-   **Functionality**:
    -   It first generates a `StrokeList` and a `StrokeBatch` using the functions from the `gen` module.
    -   It displays the entire path of all strokes as a point cloud.
    -   It provides a GUI with play/pause controls and sliders to scrub through the timeline of strokes and poses.
    -   As the animation plays, it highlights the current path and pose in the 3D view and in the 2D design image.
    -   The `step` method in this class is responsible for advancing the animation frame by frame.

#### `teleop.py`

-   **`TeleopViz`**: An interactive tool for teleoperating the robot's end-effectors using inverse kinematics (IK).
-   **Functionality**:
    -   It displays `viser`'s "transform controls" (interactive 3D gizmos) at the position of each end-effector.
    -   The user can drag these controls in the browser to set a new target pose for the end-effector.
    -   In its `step` method, it calls the `ik` function from the `gen` module to solve for the joint angles that will achieve the target pose.
    -   The robot model (and the real robot, if connected) then updates to the new configuration.
    -   It includes GUI buttons to save the current arm pose to a YAML file, which is extremely useful for defining key poses like "home" or "ready".

#### `map.py`

-   **`VizMap`**: A tool specifically for visualizing and debugging the process of mapping 2D strokes onto a 3D skin mesh.
-   **Functionality**:
    -   It displays the raw point clouds that represent the skin.
    -   It provides a GUI button to trigger the `create_mesh_from_ply_files` utility, which builds a 3D mesh from the point clouds and displays it.
    -   It displays the 2D strokes in a "design plane," which can be moved and rotated by the user with a transform control.
    -   It provides a GUI button to trigger the `map_strokes_to_mesh` function, which projects the 2D strokes onto the 3D mesh. The resulting mapped strokes are then displayed.
-   **Usage**: This tool is essential for calibrating the `design_pose` (the position and orientation of the 2D design relative to the 3D skin) and for verifying that the surface mapping algorithm is working correctly.

### How It Works and How to Use It

Each visualization is a script that can be run directly (e.g., `python -m tatbot.viz.stroke --scene=myscene`). When run, it starts a `viser` server, and a URL is printed to the console. Opening this URL in a web browser will show the interactive 3D scene. The user can then interact with the GUI elements to control the visualization, and the 3D view will update in real time.
