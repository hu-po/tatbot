# Generation Module (`src/tatbot/gen`)

This module is responsible for generating the data required for the robot to perform its tasks. It takes high-level representations, such as G-code or simple alignment descriptions, and converts them into concrete `Stroke` and `StrokeBatch` objects that contain the precise end-effector poses and joint configurations needed for execution.

## Core Abstractions

-   **Pipeline Philosophy**: The modules here form a processing pipeline. You start with a high-level description (G-code, alignment sequence), which is then progressively transformed into more detailed and hardware-specific data.
    -   `gcode.py`/`align.py` -> `map.py` -> `batch.py` -> `ik.py`
-   **`StrokeList` as intermediate representation**: The `StrokeList` is the common data structure that is passed between the different generation steps.

## Key Files and Functionality

### `gcode.py`

-   **Purpose**: To parse `.gcode` files and convert them into a `StrokeList`.
-   **`parse_gcode_file`**: Reads a G-code file, where `G0` commands indicate "pen up" (a new stroke) and `G1` commands indicate "pen down" (part of the current stroke). It converts the G-code's millimeter coordinates into meter coordinates within the design frame. Each stroke is resampled to a uniform length (`scene.stroke_length`).
-   **`make_gcode_strokes`**: This is the main entry point. It finds all `.gcode` files in the scene's design directory, parses them, and then orchestrates the sequence of strokes for both arms. It intelligently interleaves drawing strokes with ink-dipping strokes, ensuring an arm gets new ink before it starts a new path. It also generates preview images for each stroke.

### `align.py`

-   **Purpose**: To generate a predefined sequence of strokes used for visual alignment and calibration.
-   **`make_align_strokes`**: Creates a `StrokeList` for a sequence of simple movements, such as hovering over the calibration point and ink caps. This is useful for verifying the robot's setup before running a real task.

### `inkdip.py`

-   **Purpose**: To create the specific trajectory for dipping a pen into an ink cap.
-   **`make_inkdip_func`**: A factory function that returns a new function for generating inkdip strokes. The returned function is cached and, when called, creates a `Stroke` that moves the end-effector down into the ink cap, waits, and then moves back up.

### `map.py`

-   **Purpose**: To project the flat, 2D strokes generated from G-code onto a 3D mesh of the skin.
-   **`map_strokes_to_mesh`**: This is a sophisticated function that "wraps" the 2D design onto the 3D surface. It uses `potpourri3d` to trace geodesic paths (the shortest path along the surface) between points of the stroke. This ensures that the drawing appears correct on the curved surface. It also calculates the surface normals at each point of the new 3D stroke, which is crucial for orienting the pen correctly.

### `ik.py`

-   **Purpose**: To perform inverse kinematics (IK), which is the process of calculating the required robot joint angles to achieve a desired end-effector pose.
-   **`ik`**: A JAX-jitted function that solves the IK problem for a single pose using the `pyroki` library. It's a complex optimization problem that tries to match the target pose while respecting joint limits and staying close to a "rest" pose.
-   **`batch_ik`**: A `vmap`'d version of the `ik` function that can solve for a whole batch of poses very efficiently on a GPU.

### `batch.py`

-   **Purpose**: To convert a `StrokeList` into a `StrokeBatch`. This is the final step in the generation pipeline.
-   **`strokebatch_from_strokes`**: This function takes the `StrokeList` (which by this point contains the 3D, mesh-mapped end-effector poses) and:
    1.  Applies various offsets (hovering, depth, and general end-effector slop).
    2.  Calls `batch_ik` to calculate the joint angles for every single point in every stroke for both arms.
    3.  Packages everything into a `StrokeBatch` object, which is a PyTree of JAX arrays, ready for efficient execution or simulation.

## How It Works and How to Use It

1.  **Start with a design**: Create a G-code file representing your 2D design.
2.  **Generate Strokes**: Use `make_gcode_strokes` to parse the G-code and create an initial `StrokeList`. This list will represent the sequence of drawing and ink-dipping actions.
3.  **Map to Surface**: If you are working on a 3D surface, pass the `StrokeList` to `map_strokes_to_mesh` along with the mesh data. This will update the strokes with 3D end-effector positions and surface normals.
4.  **Create Batch**: Finally, pass the (potentially mapped) `StrokeList` to `strokebatch_from_strokes`. This will perform the final IK calculations and produce a `StrokeBatch` that contains all the data needed to execute the trajectory on the real robot.
5.  **Alignment**: Separately, you can use `make_align_strokes` to generate a simple set of strokes for calibration and verification, which can also be processed through the `batch.py` module.
