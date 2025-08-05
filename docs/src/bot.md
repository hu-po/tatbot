# Bot Module (`src/tatbot/bot`)

This module is responsible for controlling the robot's hardware, specifically the Trossen arms. It handles configuration, homing, and URDF-based kinematic calculations.

## Core Abstractions

-   **`TrossenConfig`**: A data class that holds configuration parameters for the Trossen arms, such as IP addresses, config file paths, and test poses.
-   **`yourdfpy.URDF` and `pyroki.Robot`**: These are used to load and represent the robot's model from a URDF file, enabling forward kinematics calculations.

## Key Files and Functionality

### `trossen_config.py`

-   **Purpose**: Manages the configuration of the Trossen arms using YAML files.
-   **`driver_from_arms`**: A key function that takes an `Arms` configuration object and returns a pair of configured `trossen_arm.TrossenArmDriver` instances, one for each arm.
-   **`configure_arm`**: A function to configure a single arm, load its settings from a file, and run a test to verify the connection and pose control.
-   **Execution**: When run as a script, it configures both the left and right arms based on the provided arguments.

### `trossen_homing.py`

-   **Purpose**: Provides a step-by-step, interactive process for calibrating and homing a Trossen arm.
-   **Process**: The script guides the user through a series of physical and software steps to ensure the arm is correctly positioned and calibrated. This includes:
    1.  Placing the arm in calibration jigs.
    2.  Setting the home position via a direct TCP socket command.
    3.  Verifying joint positions.
    4.  Rebooting the controller.
    5.  Testing gravity compensation.
-   **Usage**: This is intended to be run as a standalone script when setting up a new arm or when re-calibration is necessary.

### `urdf.py`

-   **Purpose**: Handles URDF loading and forward kinematics.
-   **`load_robot`**: Caches and loads a URDF file into `yourdfpy` and `pyroki` robot objects.
-   **`get_link_indices`**: Retrieves the numerical indices for given link names from the URDF.
-   **`get_link_poses`**: Calculates the poses (position and rotation) of specified links given a set of joint positions, using forward kinematics.

## How It Works and How to Use It

1.  **Configuration**: The `trossen_config.py` script is the primary entry point for setting up the arms. It reads configuration from YAML files and establishes a connection with the arm controllers.
2.  **Homing**: Before the arms can be used, they must be homed. The `trossen_homing.py` script provides a guided process for this critical step.
3.  **Kinematics**: The `urdf.py` module is used by other parts of the system to understand the robot's physical structure and to calculate the positions of its various parts.

This module encapsulates the low-level hardware control and setup for the Trossen arms, providing a foundation for higher-level operations.
