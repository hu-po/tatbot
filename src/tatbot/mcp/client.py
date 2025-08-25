"""Generic MCP client utilities for establishing sessions and invoking tools.

This module encapsulates the MCP protocol plumbing (session initialization,
JSON-RPC requests, and optional SSE parsing) so domain code can remain clean.
"""

from __future__ import annotations

import json
from typing import Any, Dict, Optional, Tuple

import aiohttp

from tatbot.utils.log import get_logger

log = get_logger("mcp.client", "🔗")


class MCPClient:
    """Lightweight MCP client for calling remote tools.

    The typical flow is:
    1) establish_session(host, port) -> (success, session_id, url)
    2) call_tool(url, session_id, tool_name, arguments)
    """

    def __init__(self, request_timeout_s: int = 60) -> None:
        self.request_timeout_s = request_timeout_s

    async def establish_session(self, host: str, port: int) -> Tuple[bool, Optional[str], Optional[str]]:
        """Establish an MCP session and return (success, session_id, base_url)."""
        url = f"http://{host}:{port}/mcp"
        headers = {
            "Content-Type": "application/json",
            "Accept": "application/json, text/event-stream",
        }

        try:
            async with aiohttp.ClientSession() as session:
                init_request = {
                    "jsonrpc": "2.0",
                    "id": "init",
                    "method": "initialize",
                    "params": {
                        "protocolVersion": "2025-06-18",
                        "clientInfo": {"name": "tatbot-mcp-client", "version": "1.0.0"},
                        "capabilities": {},
                    },
                }

                async with session.post(
                    url, json=init_request, headers=headers, timeout=aiohttp.ClientTimeout(total=10)
                ) as response:
                    session_id = response.headers.get("mcp-session-id")
                    if not session_id:
                        log.error("No MCP session ID returned")
                        return False, None, None
                    log.info(f"Initialized MCP session {session_id}")

                # Complete handshake
                init_complete_request = {
                    "jsonrpc": "2.0",
                    "method": "notifications/initialized",
                    "params": {},
                }
                headers_with_session = {**headers, "mcp-session-id": session_id}
                async with session.post(
                    url, json=init_complete_request, headers=headers_with_session, timeout=aiohttp.ClientTimeout(total=10)
                ) as response:
                    if response.status in [200, 202]:
                        log.info(
                            f"Completed MCP initialization for session {session_id} (status: {response.status})"
                        )
                        return True, session_id, url
                    log.error(f"Failed to complete MCP initialization: {response.status}")
                    return False, None, None
        except Exception as e:  # pragma: no cover - network errors
            log.error(f"Failed to establish MCP session: {e}")
            return False, None, None

    async def call_tool(
        self,
        url: str,
        session_id: str,
        tool_name: str,
        arguments: Dict[str, Any],
    ) -> Tuple[bool, Dict[str, Any]]:
        """Call an MCP tool and return (success, response_data)."""
        import uuid

        rpc_request = {
            "jsonrpc": "2.0",
            "id": str(uuid.uuid4()),
            "method": "tools/call",
            "params": {"name": tool_name, "arguments": arguments},
        }
        headers = {
            "Content-Type": "application/json",
            "Accept": "application/json, text/event-stream",
            "mcp-session-id": session_id,
        }

        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    url, json=rpc_request, headers=headers, timeout=aiohttp.ClientTimeout(total=self.request_timeout_s)
                ) as response:
                    if response.status != 200:
                        error_text = await response.text()
                        log.error(f"Remote tool call failed: {response.status} - {error_text}")
                        return False, {"error": error_text, "status": response.status}

                    content_type = response.headers.get("content-type", "")
                    if "text/event-stream" in content_type:
                        # Parse simple SSE stream
                        response_text = await response.text()
                        for line in response_text.strip().split("\n"):
                            if line.startswith("data: "):
                                data_content = line[6:]
                                if not data_content.strip():
                                    continue
                                try:
                                    rpc_response = json.loads(data_content)
                                except json.JSONDecodeError as e:
                                    log.error(f"Failed to parse SSE data: {e}")
                                    continue
                                if "error" in rpc_response:
                                    return False, {"error": rpc_response["error"]}
                                if "result" in rpc_response:
                                    return True, rpc_response["result"]
                        return False, {"error": "No valid response in event stream"}

                    # Regular JSON
                    rpc_response = await response.json()
                    if "error" in rpc_response:
                        return False, {"error": rpc_response["error"]}
                    if "result" in rpc_response:
                        return True, rpc_response["result"]
                    return False, {"error": "Invalid JSON-RPC response"}
        except Exception as e:  # pragma: no cover - network errors
            log.error(f"Error calling remote tool: {e}")
            return False, {"error": str(e)}


