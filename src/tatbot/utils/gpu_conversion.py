"""GPU conversion service built on top of the MCP client.

This provides domain-specific operations for cross-node GPU stroke conversion
without exposing MCP protocol details to tool code.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import yaml

from tatbot.mcp.client import MCPClient
from tatbot.utils.constants import NFS_DIR
from tatbot.utils.log import get_logger

log = get_logger("services.gpu_conversion", "🎯")


class GPUConversionService:
    def __init__(self, request_timeout_s: int = 60) -> None:
        self.client = MCPClient(request_timeout_s)

    def _get_gpu_nodes(self) -> List[str]:
        # TODO: make dynamic by reading from conf/nodes.yaml or pinging
        return ["ook"]

    def _load_node_host_port(self, node_name: str) -> Tuple[str, int]:
        # Config files live under tatbot/src/conf/mcp, not tatbot/src/tatbot/conf/mcp
        config_dir = Path(__file__).resolve().parents[2] / "conf" / "mcp"
        node_config_file = config_dir / f"{node_name}.yaml"
        if not node_config_file.exists():
            raise FileNotFoundError(f"Node config file not found: {node_config_file}")
        with open(node_config_file, "r") as f:
            node_config = yaml.safe_load(f)
        host = node_config.get("host", "localhost")
        port = int(node_config.get("port", 8000))
        return host, port

    async def convert_strokelist_remote(
        self,
        strokes_file_path: str,
        strokebatch_file_path: str,
        scene_name: str,
        first_last_rest: bool = True,
        use_ee_offsets: bool = True,
        preferred_node: Optional[str] = None,
        max_retries: int = 2,
    ) -> Tuple[bool, Optional[bytes]]:
        gpu_nodes = self._get_gpu_nodes()
        if not gpu_nodes:
            log.error("No GPU nodes available")
            return False, None

        attempt_nodes: List[str] = []
        if preferred_node and preferred_node in gpu_nodes:
            attempt_nodes.append(preferred_node)
            gpu_nodes.remove(preferred_node)
        attempt_nodes.extend(gpu_nodes)

        # Validate that provided paths are canonical NFS paths
        if not strokes_file_path.startswith(str(NFS_DIR)):
            raise ValueError(f"strokes_file_path must be under {NFS_DIR}: {strokes_file_path}")
        if not strokebatch_file_path.startswith(str(NFS_DIR)):
            raise ValueError(f"strokebatch_file_path must be under {NFS_DIR}: {strokebatch_file_path}")

        for retry in range(max_retries):
            for node_name in attempt_nodes:
                try:
                    host, port = self._load_node_host_port(node_name)
                except Exception as e:
                    log.error(f"Error loading node config for {node_name}: {e}")
                    continue

                success, session_id, url = await self.client.establish_session(host, port)
                if not success or not session_id or not url:
                    log.error(f"Failed to establish MCP session with {node_name}")
                    continue

                target_strokes_path = strokes_file_path
                target_strokebatch_path = strokebatch_file_path

                tool_name = "convert_strokelist_to_batch"
                arguments: Dict[str, Any] = {
                    "input_data": {
                        "strokes_file_path": target_strokes_path,
                        "strokebatch_file_path": target_strokebatch_path,
                        "scene_name": scene_name,
                        "first_last_rest": first_last_rest,
                        "use_ee_offsets": use_ee_offsets,
                    },
                    # FastMCP wrapper requires this placeholder field
                    "ignored_kwargs": "{}",
                }

                ok, response = await self.client.call_tool(url, session_id, tool_name, arguments)
                if ok and isinstance(response, dict):
                    # The MCP server wraps tool output; look for tool result success
                    # Tool returns a JSON-compatible dict already
                    if response.get("content"):
                        # If content-wrapped, extract first text item
                        for item in response["content"]:
                            if item.get("type") == "text":
                                import json as _json
                                try:
                                    payload = _json.loads(item["text"])  # tool's JSON
                                except Exception:
                                    payload = {}
                                if payload.get("success"):
                                    return True, None
                    else:
                        if response.get("success"):
                            return True, None

                log.warning(f"Conversion failed on {node_name}: {response}")

            if retry < max_retries - 1:
                wait_s = 2 ** retry
                log.info(f"Waiting {wait_s}s before retry...")
                await asyncio.sleep(wait_s)

        log.error("All GPU nodes failed to convert strokelist")
        return False, None

    async def ping_remote_nodes(self, nodes: Optional[List[str]] = None) -> Dict[str, bool]:
        from tatbot.mcp.client import aiohttp as _aiohttp  # type: ignore

        results: Dict[str, bool] = {}
        config_dir = Path(__file__).resolve().parent.parent / "conf" / "mcp"
        available_nodes = [f.stem for f in config_dir.glob("*.yaml") if f.name != "default.yaml"]
        target_nodes = nodes or available_nodes

        async def _ping(node_name: str) -> bool:
            try:
                host, port = self._load_node_host_port(node_name)
                url = f"http://{host}:{port}/mcp/"
                async with _aiohttp.ClientSession() as session:
                    async with session.get(url, timeout=_aiohttp.ClientTimeout(total=5)) as resp:
                        return resp.status == 200
            except Exception:
                return False

        coros = [_ping(n) for n in target_nodes]
        responses = await asyncio.gather(*coros, return_exceptions=True)
        for n, r in zip(target_nodes, responses, strict=False):
            results[n] = False if isinstance(r, Exception) else bool(r)
        return results


