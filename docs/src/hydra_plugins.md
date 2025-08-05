# Hydra Plugins Module (`src/tatbot/hydra_plugins`)

This module contains custom plugins for the [Hydra](https://hydra.cc/) configuration framework. Hydra is used to manage the complex configurations required by the `tatbot` system, and these plugins extend its functionality.

## Core Abstractions

-   **`SearchPathPlugin`**: A class provided by Hydra that allows for programmatically adding new paths to Hydra's configuration search path.

## Key Files and Functionality

### `search_path_plugin.py`

-   **Purpose**: To allow external, third-party Python packages to provide their own configuration files to the `tatbot` application without needing to modify the core `tatbot` repository.
-   **`TatbotSearchPathPlugin`**:
    -   This class implements the `manipulate_search_path` method.
    -   It adds two new search paths: `pkg://tatbot_external.conf` and `pkg://tatbot_custom.conf`. The `pkg://` prefix tells Hydra to look for configurations within installed Python packages.
-   **Usage**: This plugin is automatically discovered by Hydra at runtime. If you have a separate package (e.g., `tatbot_external`) installed in the same Python environment, and that package has a `conf` directory, Hydra will be able to discover and use the YAML files within it. This is a powerful mechanism for creating modular and extensible configurations. For example, a new set of arms or a different camera setup could be entirely defined in an external package.
