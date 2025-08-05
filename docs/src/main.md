# Main Scripts (`src/tatbot/`)

This document describes the main entry points and configuration schema files located directly in the `src/tatbot` directory. These files are central to the application's configuration loading and validation process, bridging the gap between raw YAML files and the strongly-typed Pydantic data models used throughout the system.

## Core Abstractions

-   **Hydra-Powered Configuration**: The application uses the [Hydra](https://hydra.cc/) library for managing its complex configuration. This allows for a modular approach where different parts of the configuration (arms, cameras, scenes, etc.) are defined in separate YAML files and composed at runtime.
-   **Pydantic Schema Validation**: While Hydra is excellent for composing configurations, Pydantic is used to provide an additional layer of validation and type safety. The raw, dictionary-like configuration object from Hydra is parsed into a Pydantic model, which ensures that all required fields are present and have the correct types.

## Key Files and Functionality

### `main.py`

-   **Purpose**: Serves as the primary entry point for loading and validating the application's configuration. It demonstrates how to use Hydra to compose a full `Scene` object.
-   **`main` function**:
    -   This is a `@hydra.main`-decorated function, which means it will be the entry point when the script is run.
    -   It receives the composed `cfg: DictConfig` object from Hydra.
    -   Its main job is to call `load_scene_from_config` to parse and validate this configuration.
    -   It then prints a summary of the loaded scene to confirm that everything was loaded correctly.
-   **`compose_and_validate_scene`**:
    -   This is a crucial utility function used by many other modules (like `ops` and `viz`) to load a specific `Scene` configuration by name.
    -   It handles the complexity of initializing Hydra if it hasn't been started yet (e.g., when running a visualization script) or using the existing Hydra context if it has (e.g., when called from an MCP server).
    -   It uses Hydra's `compose` API to load the base configuration and then apply an override to select a specific scene (e.g., `scenes=tatbotlogo`).

### `config_schema.py`

-   **Purpose**: Defines the top-level Pydantic model for the entire application configuration.
-   **`AppConfig`**:
    -   This Pydantic model mirrors the structure of the composed Hydra configuration (`config.yaml`). It expects fields like `arms`, `cams`, `scenes`, etc., which correspond to the different configuration groups.
    -   **`create_scene` validator**: This is the most important part of the file. It's a `@model_validator` that runs after the initial fields have been parsed. Its job is to take the raw dictionary configurations for each component (e.g., `self.arms`, which is a `dict`) and instantiate them into their corresponding Pydantic objects (e.g., `Arms(**self.arms)`).
    -   Finally, it assembles all these component objects into a single, fully-validated `Scene` object.

## How It Works and How to Use It

1.  **Hydra Composition**: When any part of the application that uses Hydra starts (e.g., running `main.py` or the MCP server), Hydra reads the `conf/config.yaml` file. This file tells Hydra to look for configurations in various subdirectories (`arms`, `cams`, `scenes`, etc.).
2.  **Scene Selection**: The `scenes` group is special. The `config.yaml` is set up to default to `scenes/default.yaml`, but this can be overridden from the command line or programmatically. The selected scene file (e.g., `scenes/tatbotlogo.yaml`) contains the high-level parameters for a specific task.
3.  **Pydantic Validation**: The composed `DictConfig` object from Hydra is then passed to the `AppConfig` model in `config_schema.py`.
4.  **Object Instantiation**: The `create_scene` validator in `AppConfig` takes over, systematically building the final `Scene` object by instantiating all of its dependencies (`Arms`, `Cams`, `Skin`, etc.) from the raw configuration data.
5.  **Ready to Use**: The result is a single, easy-to-use, and fully-validated `Scene` object that can be passed to the `ops`, `viz`, or other modules to perform their tasks. The `compose_and_validate_scene` function in `main.py` is the standard way to get access to this object.
