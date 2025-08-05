# Operations Module (`src/tatbot/ops`)

This module defines high-level, stateful operations that the robot can perform. Each "Op" is a class that encapsulates the logic for a specific task, such as recording a dataset, running a stroking motion, or resetting the robot. These operations are the main entry points for the MCP server's `run_op` tool.

## Core Abstractions

-   **`BaseOp`**: The abstract base class for all operations. It provides a common structure, including:
    -   An `__init__` method that loads the specified `scene` configuration.
    -   A `run` method, which is an `async` generator. This allows an operation to be long-running and stream progress updates and log messages back to the client.
    -   A `cleanup` method for any necessary teardown logic.

-   **`RecordOp`**: A specialized base class that inherits from `BaseOp` and adds functionality for recording data into a `LeRobotDataset`. It handles the creation of the dataset, connecting to the robot, and saving the final data. Most other operations inherit from this class.

-   **Op Configuration**: Each operation has a corresponding Pydantic/dataclass model for its configuration (e.g., `StrokeOpConfig`). This allows for type-safe and validated configuration for each operation.

## Key Files and Functionality

### `__init__.py`

-   **Purpose**: Acts as a factory and registry for all available operations.
-   **`NODE_AVAILABLE_OPS`**: A dictionary that maps a node name (e.g., "ook", "trossen-ai") to a list of operation names that are permitted to run on that node. This provides a simple but effective way to control which hardware can perform which tasks.
-   **`get_op`**: A factory function that takes an `op_name` and `node_name`, validates that the operation is allowed on that node, and returns the corresponding `Op` class and its configuration class.

### `base.py`

-   **Purpose**: Defines the `BaseOp` class, which all other operations inherit from.

### `record.py`

-   **Purpose**: Defines the `RecordOp` class, which provides the core functionality for recording `LeRobotDataset`s. It handles:
    -   Creating the dataset directory structure.
    -   Initializing the `LeRobotDataset` object.
    -   Connecting to the robot hardware via the `lerobot` library.
    -   The main `run` loop, which calls a `_run` method (to be implemented by subclasses) and then handles the final saving and optional pushing of the dataset to the Hugging Face Hub.

### `reset.py`

-   **`ResetOp`**: A very simple operation to reset the robot. It connects to the robot and then immediately disconnects, which typically causes the `lerobot` driver to send the arms to their home/sleep position.

### `record_align.py`

-   **`AlignOp`**: An operation for running a pre-defined alignment sequence.
-   **Functionality**:
    1.  It uses `make_align_strokes` from the `gen` module to create a `StrokeList`.
    2.  It converts this into a `StrokeBatch` using `strokebatch_from_strokes`.
    3.  It then iterates through this alignment sequence, moving the robot to each pose.
    4.  Because it inherits from `RecordOp`, it also records the data from this alignment procedure into a `LeRobotDataset`.

### `record_sense.py`

-   **`SenseOp`**: An operation for capturing sensory data from the environment, specifically images and point clouds.
-   **Functionality**:
    1.  It uses the `lerobot` interface to capture standard RGB images from all configured cameras.
    2.  It then disconnects the `lerobot` camera interfaces and uses the `tatbot.cam.DepthCamera` class to capture and save a series of high-resolution point clouds (PLY files).

### `record_stroke.py`

-   **`StrokeOp`**: The most complex operation, responsible for executing a drawing/stroking task based on G-code.
-   **Functionality**:
    1.  **Stroke Generation**: If not resuming, it calls `make_gcode_strokes` and `strokebatch_from_strokes` to generate the full `StrokeBatch` for the scene's design.
    2.  **Teleoperation**: It connects to a gamepad (`AtariTeleoperator`) which allows a human operator to make real-time adjustments to the needle depth (`offset_idx_l`/`r`) during the operation.
    3.  **Execution Loop**: It iterates through every stroke and every pose within each stroke, sending the calculated joint angles to the robot.
    4.  **Recording**: As it executes the motion, it records the robot's state (observations and actions) into a `LeRobotDataset`, creating one "episode" per stroke. It also saves detailed logs for each episode.

## How It Works and How to Use It

1.  A client calls the `run_op` tool on the MCP server, providing an `op_name` (e.g., "stroke") and a `scene_name`.
2.  The `mcp.handlers.run_op` function calls `ops.get_op` to get the correct `Op` class (e.g., `StrokeOp`).
3.  An instance of `StrokeOp` is created. Its `__init__` method loads the full `Scene` configuration.
4.  The `run` method of the `StrokeOp` instance is called.
5.  The operation proceeds with its logic, for example, generating strokes from G-code, and then iterating through them, sending commands to the robot.
6.  Throughout the process, it `yield`s progress dictionaries, which are streamed back to the client.
7.  Because it inherits from `RecordOp`, all the data generated during the operation is saved into a `LeRobotDataset` for later use in training a policy.
