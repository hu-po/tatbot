# MCP Module (`src/tatbot/mcp`)

This module implements the server for the **M**ulti-agent **C**ommand **P**latform (MCP). It acts as the primary API endpoint for controlling the `tatbot` system. It's built on the `mcp-server` library and uses Hydra for configuration. The server exposes a set of "tools" (RPC-like functions) that can be called by clients to perform various operations.

## Core Abstractions

-   **`FastMCP`**: The underlying server implementation from the `mcp-server` package. It handles the low-level communication protocol.
-   **Tools as Functions**: The server's functionality is exposed as a series of functions (tools). Each tool is a Python function decorated to register it with the MCP server. This makes it easy to add new capabilities.
-   **Hydra for Configuration**: The entire server, including which tools to enable and network settings, is configured via Hydra. This allows for flexible deployments and easy management of different setups (e.g., a "head" node vs. an "arm" node).
-   **Pydantic Models**: All tool inputs and outputs are defined using Pydantic models. This ensures that all communication with the server is strongly typed and validated, reducing errors and making the API self-documenting.

## Key Files and Functionality

### `server.py`

-   **Purpose**: The main entry point for the MCP server.
-   **`main`**: The Hydra-decorated main function. It parses the configuration, creates a `FastMCP` instance, registers the appropriate tools, and starts the server.
-   **`_register_tools`**: A helper function that dynamically registers tool functions from `handlers.py` with the MCP instance. It supports "namespacing" tools, where a tool's name is prefixed with the node's name (e.g., `ook_run_op`). This is crucial for avoiding name collisions in a multi-node system where different nodes might expose a tool with the same base name.

### `handlers.py`

-   **Purpose**: Contains the implementation of the actual tool functions that the server exposes.
-   **`@mcp_handler`**: A decorator used to register a function as an available tool in a central registry.
-   **Key Tools**:
    -   `run_op`: The most important tool. It executes a high-level operation (from the `tatbot.ops` module), such as `stroke` or `align`. It's an `async` generator, allowing it to stream progress and log messages back to the client as the operation runs.
    -   `ping_nodes`: Checks the network connectivity of other nodes.
    -   `list_scenes`, `list_nodes`, `list_ops`: Tools for discovering available scenes, nodes, and operations, which is very useful for UIs and command-line clients.

### `models.py`

-   **Purpose**: Defines all the Pydantic models for the requests and responses of the tools in `handlers.py`.
-   **Input Models** (e.g., `RunOpInput`): Define the expected parameters for a tool call. They include validators to ensure, for example, that a requested scene or operation actually exists before the handler logic is even run.
-   **Response Models** (e.g., `RunOpResult`): Define the structure of the data that a tool will return.
-   **`NumpyEncoder`**: A custom JSON encoder is provided to handle the serialization of `numpy` arrays, which are common in the data structures but not natively supported by JSON.

### `__init__.py`

-   This file makes the `handlers` and `models` easily importable and also uses `__all__` to explicitly define the public API of the module, which primarily consists of the tool handler functions.

## How It Works and How to Use It

1.  **Launch the Server**: The server is started by running `python -m tatbot.mcp.server`. Hydra takes over and loads the configuration from the `conf` directory.
2.  **Configuration**: The behavior of the server is controlled by `conf/mcp/default.yaml`. You can specify the host, port, and, most importantly, which `tools` should be enabled for this specific server instance.
3.  **Client Interaction**: A client (like a UI or a script) connects to the server's host and port. It can then call the registered tools by name, passing parameters as a JSON object that conforms to the corresponding input model in `models.py`.
4.  **Running an Operation**: To run a robot task, a client would call the `run_op` tool, specifying an `op_name` and a `scene_name`. The server then instantiates the appropriate `Op` from the `tatbot.ops` module and executes it, streaming back progress.
5.  **Distributed System**: In a typical setup, you would run an MCP server on each node of the `tatbot` system (e.g., one on the main "head" computer, one on each arm controller). The `namespace_tools` feature ensures that you can, for example, call `ook_run_op` to run a stroke on the "ook" arm and `eek_run_op` to run one on the "eek" arm, even though both are handled by the same underlying `run_op` function on their respective nodes.
