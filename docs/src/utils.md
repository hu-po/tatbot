# Utilities Module (`src/tatbot/utils`)

This module provides a collection of helper functions and classes that are used across the entire `tatbot` application. These utilities cover a range of functionalities, from logging and network management to color conversion and data validation.

## Key Files and Functionality

### `log.py`

-   **Purpose**: To provide a standardized logging setup for the application.
-   **`get_logger`**: The core function that creates and configures a `logging.Logger` instance. It adds a custom formatter that includes an emoji, making the logs easy to read and identify the source module.
-   **`setup_log_with_config`**: A helper function that initializes the logging system based on command-line arguments parsed by `tyro`. It sets the global logging level (e.g., to `DEBUG` if `--debug` is passed) and can enable debug logging for specific submodules.

### `net.py`

-   **Purpose**: To manage all network-related tasks, particularly SSH connections to the various nodes in the `tatbot` system.
-   **`NetworkManager`**: A comprehensive class that handles:
    -   Loading node configurations from `nodes.yaml`.
    -   Generating and distributing SSH keys to all nodes to enable passwordless login.
    -   Writing a local `~/.ssh/config` file to make it easy to connect to nodes by their names (e.g., `ssh ook`).
    -   Testing the connectivity to all nodes in parallel.
-   **Usage**: The `setup_network` method is the main entry point for a first-time setup. The `NetworkManager` is also used by the `mcp.handlers.ping_nodes` tool.

### `mode_toggle.py`

-   **Purpose**: To switch the network configuration of the entire `tatbot` system between two modes: "home" and "edge". This is a highly specialized utility for managing DNS settings in a specific network environment.
-   **`NetworkToggler`**:
    -   **Home Mode**: Assumes a local DNS server is running on `rpi1`. It configures all other nodes and the robot arms to use `rpi1` as their DNS server.
    -   **Edge Mode**: Configures all nodes to use the main LAN router for DNS.
    -   It works by SSHing into each node and modifying system configuration files (`/etc/dhcpcd.conf`).

### `plymesh.py`

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

### `colors.py`

-   **Purpose**: Defines a dictionary of standard color names to BGR tuples and provides a color conversion utility.
-   **`COLORS`**: A dictionary of human-readable color names (e.g., "red", "blue") mapped to their `(B, G, R)` color values, suitable for use with OpenCV.
-   **`argb_to_bgr`**: A function to convert a color from a single 32-bit ARGB integer (as used in some design software) to a BGR tuple.

### `validation.py` & `jnp_types.py`

-   **Purpose**: Provide small, focused utility functions for data validation and type conversion.
-   **`expand_user_path`**: A simple helper to expand the `~` in a path string.
-   **`ensure_numpy_array`**: Converts JAX arrays into NumPy arrays, which is often necessary before serialization or when interfacing with libraries that don't directly support JAX arrays.
