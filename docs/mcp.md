# MCP (Model Context Protocol)

- **Unified Server**: A single `tatbot.mcp.server` now runs on all nodes.
- **Hydra Configuration**: Node-specific behavior (host, port, tools) is defined in YAML files under `conf/mcp/`.
- **Unified Tools Architecture**: Tools are now in `tatbot.tools` with decorator-based registration (see [Tools Documentation](tools.md)).
- **Pydantic Models**: All requests and responses are strongly typed with Pydantic for validation and clarity.

## Starting Servers
To start an MCP server on a specific node, use the unified launcher script:
```bash
# Start the server on the 'ook' node
./scripts/run_mcp.sh ook

# Start on 'eek' with debug mode enabled
./scripts/run_mcp.sh eek mcp.debug=true
```

## Server Logs
Logs are written to `/nfs/tatbot/mcp-logs/<node_name>.log`. For example, the `ook` server's log is at `/nfs/tatbot/mcp-logs/ook.log`.

## Node Configuration
The behavior of each MCP server is defined by a corresponding YAML file in `src/conf/mcp/`. For example, the `ook` node is configured by `src/conf/mcp/ook.yaml`.

These files control:
- **`host` and `port`**: Network settings for the server.
- **`tools`**: A list of which tools are enabled on this node (tools auto-register from `tatbot.tools`).
- **`extras`**: A list of optional dependency groups (from `pyproject.toml`) that should be installed on this node. The `run_mcp.sh` script automatically reads this list and uses `uv pip install` to ensure the correct dependencies are present before starting the server.

## Available Tools

Tools are now defined in the unified `tatbot.tools` module using decorator-based registration. See the [Tools Documentation](tools.md) for detailed information.

**System Tools:**
- `list_nodes` (rpi1, rpi2, eek, ook, oop): List all configured tatbot nodes
- `ping_nodes` (rpi1, rpi2, eek, ook, oop): Test connectivity to tatbot nodes  
- `list_scenes` (eek): List available scene configurations
- `list_recordings` (eek): List available recordings from the recordings directory

**Robot Tools:**
- `align` (hog, oop): Generate and execute alignment strokes for calibration
- `reset` (hog, oop): Reset robot to safe/ready position
- `sense` (hog): Capture environmental data (cameras, sensors)
- `stroke` (hog): Execute artistic strokes on paper/canvas

**GPU Tools** (available on GPU-enabled nodes only):
- `convert_strokelist_to_batch` (ook, oop): GPU-accelerated stroke trajectory conversion

Tools specify their node availability and requirements directly in their decorator, eliminating the need for separate mapping files.

## Cross-Node GPU Processing

The MCP system enables transparent cross-node GPU acceleration for stroke trajectory conversion. This allows robot operations on non-GPU nodes (like `hog`) to automatically leverage GPU-accelerated inverse kinematics solving on GPU-enabled nodes (like `ook`).

### Architecture

**GPU Detection & Routing**
- Robot operations (`align`, `stroke`) automatically detect local GPU availability via `check_local_gpu()`
- When no local GPU is available, operations route conversion requests to remote GPU nodes
- `GPUProxy` handles node discovery, load balancing, and communication with GPU-enabled nodes

**NFS Path Translation**
- All nodes share NFS storage mounted at the canonical path `/nfs/tatbot/`
- Files remain on shared NFS throughout the entire process - no data transfer required

**MCP Protocol Communication**
- Cross-node requests use JSON-RPC 2.0 over StreamableHTTP transport
- Proper session establishment with `initialize` → `notifications/initialized` handshake
- Nodes are distinguished by their MCP server names (e.g., `ook`) with tools using original names
- Retry logic with exponential backoff for fault tolerance

### Workflow Example

1. **Operation Start**: User runs `align` operation on `hog`
2. **GPU Detection**: `check_local_gpu()` returns `False` on `hog`
3. **File Creation**: Strokes saved to `/nfs/tatbot/recordings/align-*/strokes.yaml`
4. **Remote Conversion**: MCP call to `convert_strokelist_to_batch` on `ook` server with translated paths
5. **GPU Processing**: `ook` performs JAX-accelerated inverse kinematics solving
6. **Result Storage**: Strokebatch saved to shared NFS at translated output path
7. **Operation Continue**: `hog` loads strokebatch and continues robot operation

### Configuration

**GPU Node Setup**
```yaml
# conf/mcp/ook.yaml
host: "0.0.0.0"
port: 8000
extras: ["gpu"]  # Enables GPU dependencies
tools:
  - run_op
  - convert_strokelist_to_batch  # GPU conversion tool
```

**Non-GPU Node Setup**
```yaml  
# conf/mcp/hog.yaml
host: "0.0.0.0"
port: 8000
extras: []  # No GPU dependencies
tools:
  - run_op  # Will use GPUProxy for remote conversion
```

### Error Handling

- **Connection Failures**: Automatic retry with exponential backoff
- **GPU Node Unavailable**: Falls back to CPU conversion (if enabled)
- **NFS Sync Issues**: Built-in timeout and file existence verification
- **Path Translation**: Robust handling of different mount point formats

### Benefits

- **Transparent**: Robot operations work identically regardless of node GPU capability
- **Efficient**: No data transfer - files remain on shared NFS throughout
- **Scalable**: Easy to add new GPU nodes to the processing pool
- **Fault Tolerant**: Multiple retry mechanisms and fallback options
- **Zero Configuration**: Automatic detection and routing based on node capabilities

## Cursor IDE Integration
The workflow in Cursor remains similar:
- Use `ctrl-shift-p` > "View: OpenMCP Settings" to toggle servers.
- If you change the server config (e.g., enable a new tool), you must restart the server and re-toggle it in Cursor.
- If you encounter "Bad Request" errors, a restart of the server and a re-toggle in Cursor usually resolves it.

## Inspector

manually call tools on `hog` node:

```bash
cd ~/tatbot
npx @modelcontextprotocol/inspector --config .cursor/mcp.json --server hog
```