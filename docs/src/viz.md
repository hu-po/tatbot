# Visualization Module (`src/tatbot/viz`)

This module provides a suite of interactive 3D visualization tools for the `tatbot` system, built on the `viser` library. These tools are invaluable for debugging, calibration, and understanding the complex geometric data and robot motions involved in the tasks. Each visualization is a standalone application that can be run from the command line.

## Core Abstractions

-   **`viser.ViserServer`**: The core component from the `viser` library that runs a WebSocket server, allowing real-time, interactive 3D scenes to be viewed in a web browser.
-   **`BaseViz`**: An abstract base class for all visualization applications in this module. It handles the common setup tasks:
    -   Initializing a `ViserServer`.
    -   Loading a `Scene` configuration.
    -   Loading the robot's URDF model and displaying it in the scene using `ViserUrdf`.
    -   Creating GUI elements for displaying joint values.
    -   Optionally connecting to the real robot hardware to mirror the visualized state.
    -   Displaying camera frustums and other static scene elements.
-   **Interactive GUIs**: Each visualization uses `viser`'s GUI components (sliders, buttons, text boxes) to allow for real-time interaction with the scene and the underlying data.

## Key Files and Functionality

### `base.py`

-   **Purpose**: Defines the `BaseViz` class, which provides the foundational structure for all other visualization tools.
-   **`run` loop**: The `BaseViz` class has a main `run` loop that continuously updates the visualization. In each iteration, it:
    1.  Updates the visualized robot's joint positions.
    2.  If connected to the real robot, sends the current joint positions to the hardware.
    3.  Updates the poses of camera frustums based on the robot's kinematics.
    4.  If depth cameras are enabled, captures and displays point clouds.
    5.  Calls a `step` method, which is intended to be overridden by subclasses to implement specific animation or interaction logic.

### `stroke.py`

-   **`VizStrokes`**: A tool for visualizing the execution of a `StrokeBatch`.
-   **Functionality**:
    -   It first generates a `StrokeList` and a `StrokeBatch` using the functions from the `gen` module.
    -   It displays the entire path of all strokes as a point cloud.
    -   It provides a GUI with play/pause controls and sliders to scrub through the timeline of strokes and poses.
    -   As the animation plays, it highlights the current path and pose in the 3D view and in the 2D design image.
    -   The `step` method in this class is responsible for advancing the animation frame by frame.

### `teleop.py`

-   **`TeleopViz`**: An interactive tool for teleoperating the robot's end-effectors using inverse kinematics (IK).
-   **Functionality**:
    -   It displays `viser`'s "transform controls" (interactive 3D gizmos) at the position of each end-effector.
    -   The user can drag these controls in the browser to set a new target pose for the end-effector.
    -   In its `step` method, it calls the `ik` function from the `gen` module to solve for the joint angles that will achieve the target pose.
    -   The robot model (and the real robot, if connected) then updates to the new configuration.
    -   It includes GUI buttons to save the current arm pose to a YAML file, which is extremely useful for defining key poses like "home" or "ready".

### `map.py`

-   **`VizMap`**: A tool specifically for visualizing and debugging the process of mapping 2D strokes onto a 3D skin mesh.
-   **Functionality**:
    -   It displays the raw point clouds that represent the skin.
    -   It provides a GUI button to trigger the `create_mesh_from_ply_files` utility, which builds a 3D mesh from the point clouds and displays it.
    -   It displays the 2D strokes in a "design plane," which can be moved and rotated by the user with a transform control.
    -   It provides a GUI button to trigger the `map_strokes_to_mesh` function, which projects the 2D strokes onto the 3D mesh. The resulting mapped strokes are then displayed.
-   **Usage**: This tool is essential for calibrating the `design_pose` (the position and orientation of the 2D design relative to the 3D skin) and for verifying that the surface mapping algorithm is working correctly.

## How It Works and How to Use It

Each visualization is a script that can be run directly (e.g., `python -m tatbot.viz.stroke --scene=myscene`). When run, it starts a `viser` server, and a URL is printed to the console. Opening this URL in a web browser will show the interactive 3D scene. The user can then interact with the GUI elements to control the visualization, and the 3D view will update in real time.
