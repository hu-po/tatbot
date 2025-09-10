---
summary: Stroke → GPU conversion codepath walkthrough
tags: [mcp, gpu, stroke, jax]
updated: 2025-09-10
audience: [dev]
---

# Stroke → GPU Conversion Codepath

This document maps the end-to-end flow when the `stroke` MCP tool on a non-GPU node offloads inverse kinematics (IK) conversion to a GPU node. It is intended as a reference for debugging issues such as lingering processes or memory pressure on GPU nodes.

## Components

- `src/tatbot/tools/robot/stroke.py` → `stroke_tool(...)`
  - Entry point for executing strokes on the robot (node: `hog`).
  - Generates `StrokeList` and produces `strokes.yaml` under `/nfs/tatbot/recordings/...`.
  - If no local GPU: uses `GPUConversionService` to call a remote GPU tool.

- `src/tatbot/utils/gpu_conversion.py` → `GPUConversionService`
  - Discovers GPU nodes from `src/conf/mcp/*.yaml` via `extras: [gpu]`.
  - Establishes FastMCP session (`initialize`, `notifications/initialized`).
  - Calls remote tool `convert_strokelist_to_batch` with NFS paths.
  - Retries across GPU nodes with exponential backoff.

- `src/tatbot/mcp/client.py` → `MCPClient`
  - Thin JSON-RPC client over `aiohttp` for FastMCP servers.
  - Handles SSE or JSON responses for tool completion.

- `src/tatbot/mcp/server.py` → Hydra-driven FastMCP server
  - Started with `./scripts/mcp_run.sh <node>` on GPU node (e.g., `ook`).
  - Registers tools via `tatbot.tools.registry` based on node config.

- `src/tatbot/tools/gpu/convert_strokes.py` → `convert_strokes(...)`
  - GPU tool: loads `StrokeList`, composes scene, runs JAX IK, saves `strokebatch.safetensors`.
  - Validates GPU availability via node `extras`.
  - Emits progress via MCP context; returns success with output path.

- `src/tatbot/gen/batch.py` → `strokebatch_from_strokes(...)`
  - Performs JAX-accelerated IK (uses `jax`, `jax.numpy`, `jaxlie`).
  - Computes end-effector positions/rotations, applies offsets, runs `batch_ik`.
  - Returns `StrokeBatch` (saved by the tool).

- `src/tatbot/main.py` → `compose_and_validate_scene(...)`
  - Composes Hydra config for `Scene`; used by both `stroke` and GPU tool.

## Configuration

- GPU nodes: `src/conf/mcp/ook.yaml`, `src/conf/mcp/oop.yaml`
  - `extras: [bot, dev, gen, img, viz, gpu]`
  - `tools: [convert_strokelist_to_batch, ...]`

- Non-GPU node: `src/conf/mcp/hog.yaml`
  - `tools: [stroke, align, ...]`

- Server defaults: `src/conf/mcp/default.yaml` (host, port, transport, debug)

## Runtime Flow

1) Client (hog): `stroke_tool` detects no local GPU via `check_local_gpu()` and writes `strokes.yaml` to NFS.
2) Client (hog): `GPUConversionService.convert_strokelist_remote(...)` discovers GPU nodes and calls `MCPClient.establish_session(...)` to `http://<gpu-host>:<port>/mcp`.
3) Server (gpu): `tatbot.mcp.server` accepts the session and exposes registered tools.
4) Client→Server: `tools/call` JSON-RPC for `convert_strokelist_to_batch` with NFS input/output paths and scene/meta.
5) Server (gpu): Tool `convert_strokes` loads `StrokeList` and calls `compose_and_validate_scene`.
6) Server (gpu): Tool calls `strokebatch_from_strokes` to run JAX IK; then saves `strokebatch.safetensors` to NFS.
7) Client (hog): Receives success, loads `StrokeBatch`, proceeds to execute strokes on robot.

## Logging & Observability

- GPU server logs: `/nfs/tatbot/mcp-logs/<node>.log` (started by `./scripts/mcp_run.sh <node>`).
- Key loggers:
  - `tools.convert_strokes` (convert tool)
  - `gen.batch` (JAX IK batching)
  - `mcp.server`, `mcp.client` (session and RPC)
- Enable server debug: pass `mcp.debug=true` to `mcp_run.sh` (Hydra flag).

## Process & Threading Notes

- The GPU conversion tool runs inside the long-lived FastMCP server process (`python3 -m tatbot.mcp.server`).
- The conversion tool itself does not spawn subprocesses; it uses JAX/XLA within the process.
- Visualization tools (`start_*_viz`) start background threads (daemon) but not new processes.
- Other OS-level process spawns in repo:
  - `utils/net.py` uses `subprocess.run` for SSH setup (not used in conversion path).
  - `scripts/mcp_run.sh` uses `nohup` to launch the server; `scripts/kill.sh` is a manual nuke.

## Known Risk Areas (for investigation)

- JAX/XLA GPU memory retention across repeated tool invocations in a long-lived process (server) causing memory pressure.
- FastMCP server implementation details (external) could spawn worker processes; parent must `wait()` children to avoid zombies.
- Hydra re-composition within the tool (`hydra.compose`) under an already Hydra-initialized process — interplay is handled by `compose_and_validate_scene`, but concurrency needs attention.
- NFS sync waits: tool polls for files; timeouts and retries are logged.

## Primary Files & Functions

- GPU tool: `src/tatbot/tools/gpu/convert_strokes.py::convert_strokes`
- IK logic: `src/tatbot/gen/batch.py::strokebatch_from_strokes`
- MCP server: `src/tatbot/mcp/server.py::main`
- MCP client: `src/tatbot/mcp/client.py::{establish_session, call_tool}`
- Routing service: `src/tatbot/utils/gpu_conversion.py::GPUConversionService`
- Stroke orchestrator: `src/tatbot/tools/robot/stroke.py::stroke_tool`

