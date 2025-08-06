"""GPU proxy for cross-node stroke conversion."""

import asyncio
import base64
import json
import os
import random
from typing import Any, Dict, List, Optional, Tuple

import aiohttp
import yaml

from tatbot.utils.log import get_logger

log = get_logger("mcp.gpu_proxy", "🎯🔗")


class GPUProxy:
    """Proxy for GPU-accelerated stroke conversion on remote nodes."""
    
    def __init__(self, timeout: int = 60):
        """Initialize GPU proxy.
        
        Args:
            timeout: Request timeout in seconds
        """
        self.timeout = timeout
        self._mcp_config = None
        
    def _get_mcp_config(self):
        """Get MCP configuration."""
        if self._mcp_config is None:
            import hydra
            try:
                self._mcp_config = hydra.compose(config_name="config")
            except Exception as e:
                log.warning(f"Failed to load hydra config: {e}")
                # Return a minimal config structure for testing
                self._mcp_config = type('Config', (), {'mcp': {}})()
        return self._mcp_config
    
    def _get_gpu_nodes(self) -> List[str]:
        """Get list of nodes with GPU support that have active MCP servers.
        
        Returns:
            List of node names with GPU extras and running MCP servers
        """
        # Based on your info: MCP servers are running on ook and trossen-ai
        # But only ook has GPU support, so return only ook for now
        return ["ook"]
    
    def _select_gpu_node(self, preferred: Optional[str] = None) -> Optional[str]:
        """Select a GPU node for processing.
        
        Args:
            preferred: Preferred node name if available
            
        Returns:
            Selected node name or None if no GPU nodes available
        """
        gpu_nodes = self._get_gpu_nodes()
        
        if not gpu_nodes:
            log.error("No GPU nodes available")
            return None
            
        if preferred and preferred in gpu_nodes:
            return preferred
            
        # Random selection for load balancing
        selected = random.choice(gpu_nodes)
        log.info(f"Selected GPU node: {selected}")
        return selected
    
    async def _establish_mcp_session(
        self,
        node_name: str,
        host: str,
        port: int
    ) -> Tuple[bool, Optional[str]]:
        """Establish MCP session with a remote node.
        
        Args:
            node_name: Target node name
            host: Node host address
            port: Node port
            
        Returns:
            Tuple of (success, session_id)
        """
        url = f"http://{host}:{port}/mcp/"
        
        headers = {
            "Content-Type": "application/json",
            "Accept": "application/json, text/event-stream"
        }
        
        try:
            async with aiohttp.ClientSession() as session:
                # Step 1: Initialize the session
                init_request = {
                    "jsonrpc": "2.0",
                    "id": "init",
                    "method": "initialize",
                    "params": {
                        "protocolVersion": "2024-11-05",
                        "clientInfo": {
                            "name": "tatbot-gpu-proxy",
                            "version": "1.0.0"
                        },
                        "capabilities": {}
                    }
                }
                
                async with session.post(
                    url,
                    json=init_request,
                    headers=headers,
                    timeout=aiohttp.ClientTimeout(total=10)
                ) as response:
                    session_id = response.headers.get("mcp-session-id")
                    if not session_id:
                        log.error(f"No session ID returned from {node_name}")
                        return False, None
                    
                    log.info(f"Initialized MCP session {session_id} with {node_name}")
                
                # Step 2: Send initialized notification to complete handshake
                init_complete_request = {
                    "jsonrpc": "2.0",
                    "method": "notifications/initialized",
                    "params": {}
                }
                
                headers_with_session = {
                    **headers,
                    "mcp-session-id": session_id
                }
                
                async with session.post(
                    url,
                    json=init_complete_request,
                    headers=headers_with_session,
                    timeout=aiohttp.ClientTimeout(total=10)
                ) as response:
                    if response.status in [200, 202]:  # Accept both OK and Accepted
                        log.info(f"Completed MCP initialization for session {session_id} (status: {response.status})")
                        return True, session_id
                    else:
                        log.error(f"Failed to complete initialization: {response.status}")
                        return False, None
                        
        except Exception as e:
            log.error(f"Failed to establish MCP session with {node_name}: {e}")
            return False, None

    async def _call_remote_tool(
        self, 
        node_name: str,
        tool_name: str,
        input_data: Dict[str, Any]
    ) -> Tuple[bool, Dict[str, Any]]:
        """Call an MCP tool on a remote node.
        
        Args:
            node_name: Target node name
            tool_name: MCP tool name
            input_data: Tool input parameters
            
        Returns:
            Tuple of (success, response_data)
        """
        try:
            from pathlib import Path
            
            # Load the specific node's MCP config
            config_dir = Path(__file__).parent.parent.parent / "conf" / "mcp"
            node_config_file = config_dir / f"{node_name}.yaml"
            
            if not node_config_file.exists():
                log.error(f"Node config file not found: {node_config_file}")
                return False, {"error": f"Node {node_name} config not found"}
            
            with open(node_config_file, 'r') as f:
                node_config = yaml.safe_load(f)
            
            host = node_config.get("host", "localhost")
            port = node_config.get("port", 8000)
            
            # MCP servers expect JSON-RPC calls, not REST endpoints
            url = f"http://{host}:{port}/mcp/"
        except Exception as e:
            log.error(f"Error getting node config for {node_name}: {e}")
            return False, {"error": f"Config error: {str(e)}"}
        
        # Establish MCP session first
        session_success, session_id = await self._establish_mcp_session(node_name, host, port)
        if not session_success or not session_id:
            return False, {"error": "Failed to establish MCP session"}
        
        # Create JSON-RPC request
        import uuid
        # Tools are namespaced with node name (e.g., "ook_convert_strokelist_to_batch")
        namespaced_tool_name = f"{node_name}_{tool_name}"
        
        rpc_request = {
            "jsonrpc": "2.0",
            "id": str(uuid.uuid4()),
            "method": "tools/call",
            "params": {
                "name": namespaced_tool_name,
                "arguments": input_data
            }
        }
        
        headers = {
            "Content-Type": "application/json",
            "Accept": "application/json, text/event-stream",
            "mcp-session-id": session_id
        }
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    url,
                    json=rpc_request,
                    headers=headers,
                    timeout=aiohttp.ClientTimeout(total=self.timeout)
                ) as response:
                    if response.status == 200:
                        content_type = response.headers.get('content-type', '')
                        
                        if 'text/event-stream' in content_type:
                            # Handle streaming response
                            response_text = await response.text()
                            
                            # Parse SSE (Server-Sent Events) format
                            for line in response_text.strip().split('\n'):
                                if line.startswith('data: '):
                                    data_content = line[6:]  # Remove 'data: ' prefix
                                    if data_content.strip():
                                        try:
                                            rpc_response = json.loads(data_content)
                                            if "error" in rpc_response:
                                                log.error(f"JSON-RPC error: {rpc_response['error']}")
                                                return False, {"error": rpc_response['error']}
                                            elif "result" in rpc_response:
                                                # Extract the tool result from JSON-RPC response
                                                tool_result = rpc_response["result"]
                                                if isinstance(tool_result, dict) and "content" in tool_result:
                                                    # MCP tool responses are wrapped in content
                                                    for content_item in tool_result["content"]:
                                                        if content_item.get("type") == "text":
                                                            try:
                                                                # Parse the JSON result from the text content
                                                                result_data = json.loads(content_item["text"])
                                                                return True, result_data
                                                            except json.JSONDecodeError:
                                                                return True, {"raw_response": content_item["text"]}
                                                return True, tool_result
                                        except json.JSONDecodeError as e:
                                            log.error(f"Failed to parse SSE data: {e}, data: {data_content}")
                                            continue
                            
                            log.error("No valid JSON-RPC response found in event stream")
                            return False, {"error": "No valid response in event stream"}
                        else:
                            # Handle regular JSON response
                            rpc_response = await response.json()
                            if "error" in rpc_response:
                                log.error(f"JSON-RPC error: {rpc_response['error']}")
                                return False, {"error": rpc_response['error']}
                            elif "result" in rpc_response:
                                # Extract the tool result from JSON-RPC response
                                tool_result = rpc_response["result"]
                                if isinstance(tool_result, dict) and "content" in tool_result:
                                    # MCP tool responses are wrapped in content
                                    for content_item in tool_result["content"]:
                                        if content_item.get("type") == "text":
                                            try:
                                                # Parse the JSON result from the text content
                                                result_data = json.loads(content_item["text"])
                                                return True, result_data
                                            except json.JSONDecodeError:
                                                return True, {"raw_response": content_item["text"]}
                                return True, tool_result
                            else:
                                log.error(f"Unexpected JSON-RPC response format: {rpc_response}")
                                return False, {"error": "Invalid JSON-RPC response"}
                    else:
                        error_text = await response.text()
                        log.error(f"Remote tool call failed: {response.status} - {error_text}")
                        return False, {"error": error_text, "status": response.status}
                        
        except asyncio.TimeoutError:
            log.error(f"Timeout calling {tool_name} on {node_name}")
            return False, {"error": "Request timeout"}
        except Exception as e:
            log.error(f"Error calling remote tool: {e}")
            return False, {"error": str(e)}
    
    async def convert_strokelist_remote(
        self,
        strokes_yaml: str,
        scene_name: str,
        first_last_rest: bool = True,
        use_ee_offsets: bool = True,
        preferred_node: Optional[str] = None,
        max_retries: int = 2
    ) -> Tuple[bool, Optional[bytes]]:
        """Convert StrokeList to StrokeBatch using a remote GPU node with retry logic.
        
        Args:
            strokes_yaml: YAML content of StrokeList
            scene_name: Scene name for conversion
            first_last_rest: Apply first/last rest positions
            use_ee_offsets: Apply end-effector offsets
            preferred_node: Preferred GPU node name
            max_retries: Maximum number of retry attempts
            
        Returns:
            Tuple of (success, strokebatch_bytes)
        """
        gpu_nodes = self._get_gpu_nodes()
        if not gpu_nodes:
            log.error("No GPU nodes available")
            return False, None
        
        # Prepare input data with the wrapper format expected by MCP tools
        input_data = {
            "input_data": {
                "strokes_yaml": strokes_yaml,
                "scene_name": scene_name,
                "first_last_rest": first_last_rest,
                "use_ee_offsets": use_ee_offsets
            }
        }
        
        # Try preferred node first, then others
        attempt_nodes = []
        if preferred_node and preferred_node in gpu_nodes:
            attempt_nodes.append(preferred_node)
            gpu_nodes.remove(preferred_node)
        attempt_nodes.extend(gpu_nodes)
        
        for retry in range(max_retries):
            for node_name in attempt_nodes:
                log.info(f"Attempt {retry + 1}/{max_retries}: Calling convert_strokelist_to_batch on {node_name}")
                
                # Call remote tool
                success, response = await self._call_remote_tool(
                    node_name,
                    "convert_strokelist_to_batch",
                    input_data
                )
                
                if success and response.get("success"):
                    # Decode base64 response
                    try:
                        strokebatch_bytes = base64.b64decode(response["strokebatch_base64"])
                        log.info(f"Successfully received strokebatch from {node_name}")
                        return True, strokebatch_bytes
                    except Exception as e:
                        log.error(f"Failed to decode strokebatch from {node_name}: {e}")
                        continue
                else:
                    log.warning(f"Conversion failed on {node_name}: {response}")
                    continue
            
            # Wait before next retry
            if retry < max_retries - 1:
                wait_time = 2 ** retry  # Exponential backoff
                log.info(f"Waiting {wait_time}s before retry...")
                await asyncio.sleep(wait_time)
        
        log.error("All GPU nodes failed to convert strokelist")
        return False, None
    
    async def ping_remote_nodes(self, nodes: Optional[List[str]] = None) -> Dict[str, bool]:
        """Ping remote nodes to check connectivity.
        
        Args:
            nodes: List of node names to ping, or None for all nodes
            
        Returns:
            Dict mapping node name to connectivity status
        """
        try:
            from pathlib import Path
            
            # Get available nodes from MCP config files
            config_dir = Path(__file__).parent.parent.parent / "conf" / "mcp"
            available_nodes = [
                f.stem for f in config_dir.glob("*.yaml") 
                if f.name != "default.yaml"
            ]
            target_nodes = nodes or available_nodes
        except Exception as e:
            log.error(f"Error getting nodes for ping: {e}")
            return {}
        
        results = {}
        
        tasks = []
        for node_name in target_nodes:
            tasks.append(self._ping_single_node(node_name))
        
        responses = await asyncio.gather(*tasks, return_exceptions=True)
        
        for node_name, response in zip(target_nodes, responses, strict=False):
            if isinstance(response, Exception):
                results[node_name] = False
            else:
                results[node_name] = response
        
        return results
    
    async def _ping_single_node(self, node_name: str) -> bool:
        """Ping a single node.
        
        Args:
            node_name: Node to ping
            
        Returns:
            True if node is reachable
        """
        try:
            from pathlib import Path
            
            # Load the specific node's MCP config
            config_dir = Path(__file__).parent.parent.parent / "conf" / "mcp"
            node_config_file = config_dir / f"{node_name}.yaml"
            
            if not node_config_file.exists():
                return False
            
            with open(node_config_file, 'r') as f:
                node_config = yaml.safe_load(f)
            
            host = node_config.get("host", "localhost")
            port = node_config.get("port", 8000)
            
            # Use a simple health check or try the MCP endpoint
            url = f"http://{host}:{port}/mcp/"
            
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    url,
                    timeout=aiohttp.ClientTimeout(total=5)
                ) as response:
                    return response.status == 200
        except:
            return False


def check_local_gpu() -> bool:
    """Check if the current node has GPU support.
    
    Returns:
        True if current node has GPU extras configured
    """
    try:
        import socket

        import hydra
        
        from pathlib import Path
        
        hostname = socket.gethostname()
        node_name = hostname.lower()
        
        # Load the current node's MCP config
        config_dir = Path(__file__).parent.parent.parent / "conf" / "mcp"
        node_config_file = config_dir / f"{node_name}.yaml"
        
        if node_config_file.exists():
            with open(node_config_file, 'r') as f:
                node_config = yaml.safe_load(f)
            return "gpu" in node_config.get("extras", [])
        else:
            # Fallback to current hydra config
            cfg = hydra.compose(config_name="config")
            return "gpu" in cfg.mcp.get("extras", [])
    except Exception as e:
        log.warning(f"Failed to check local GPU support: {e}")
        return False


async def get_or_convert_strokebatch(
    strokes_yaml: str,
    scene_name: str,
    first_last_rest: bool = True,
    use_ee_offsets: bool = True,
    cache_path: Optional[str] = None,
    fallback_to_cpu: bool = True
) -> Tuple[bool, Optional[bytes]]:
    """Get or convert StrokeBatch with comprehensive error handling and fallbacks.
    
    Args:
        strokes_yaml: YAML content of StrokeList
        scene_name: Scene name for conversion
        first_last_rest: Apply first/last rest positions
        use_ee_offsets: Apply end-effector offsets
        cache_path: Optional path to cache the result
        fallback_to_cpu: Whether to fallback to CPU conversion if GPU fails
        
    Returns:
        Tuple of (success, strokebatch_bytes)
    """
    # Check for cached result
    if cache_path and os.path.exists(cache_path):
        try:
            log.info(f"Loading cached strokebatch from {cache_path}")
            with open(cache_path, "rb") as f:
                return True, f.read()
        except Exception as e:
            log.warning(f"Failed to load cached strokebatch: {e}")
    
    # Check if local node has GPU
    if check_local_gpu():
        log.info("Using local GPU for conversion")
        try:
            import io

            import safetensors.numpy

            from tatbot.data.stroke import StrokeList
            from tatbot.gen.batch import strokebatch_from_strokes

            # Local conversion
            strokes_data = yaml.safe_load(strokes_yaml)
            strokes = StrokeList.model_validate(strokes_data)
            from tatbot.main import compose_and_validate_scene
            scene = compose_and_validate_scene(scene_name)
            
            strokebatch = strokebatch_from_strokes(
                scene, 
                strokes,
                first_last_rest=first_last_rest,
                use_ee_offsets=use_ee_offsets
            )
            
            # Save to bytes
            buffer = io.BytesIO()
            safetensors.numpy.save({
                "ee_pos_l": strokebatch.ee_pos_l,
                "ee_pos_r": strokebatch.ee_pos_r,
                "ee_rot_l": strokebatch.ee_rot_l,
                "ee_rot_r": strokebatch.ee_rot_r,
                "joints": strokebatch.joints,
            }, buffer)
            
            strokebatch_bytes = buffer.getvalue()
            
            # Cache if requested
            if cache_path:
                try:
                    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
                    with open(cache_path, "wb") as f:
                        f.write(strokebatch_bytes)
                    log.info(f"Cached strokebatch to {cache_path}")
                except Exception as e:
                    log.warning(f"Failed to cache strokebatch: {e}")
            
            return True, strokebatch_bytes
            
        except Exception as e:
            log.error(f"Local GPU conversion failed: {e}")
            # Continue to try remote or CPU fallback
    
    # Use remote GPU node
    log.info("Using remote GPU node for conversion")
    gpu_proxy = GPUProxy()
    success, strokebatch_bytes = await gpu_proxy.convert_strokelist_remote(
        strokes_yaml,
        scene_name,
        first_last_rest,
        use_ee_offsets
    )
    
    if success and strokebatch_bytes:
        # Cache if requested
        if cache_path:
            try:
                os.makedirs(os.path.dirname(cache_path), exist_ok=True)
                with open(cache_path, "wb") as f:
                    f.write(strokebatch_bytes)
                log.info(f"Cached strokebatch to {cache_path}")
            except Exception as e:
                log.warning(f"Failed to cache strokebatch: {e}")
        
        return True, strokebatch_bytes
    
    # Final fallback to CPU conversion if enabled
    if fallback_to_cpu:
        log.warning("GPU conversion failed, falling back to CPU conversion")
        try:
            import io

            import safetensors.numpy

            from tatbot.data.stroke import StrokeList
            from tatbot.gen.batch import strokebatch_from_strokes

            # CPU conversion
            strokes_data = yaml.safe_load(strokes_yaml)
            strokes = StrokeList.model_validate(strokes_data)
            from tatbot.main import compose_and_validate_scene
            scene = compose_and_validate_scene(scene_name)
            
            # Use smaller batch size for CPU
            strokebatch = strokebatch_from_strokes(
                scene, 
                strokes,
                batch_size=32,  # Smaller batch for CPU
                first_last_rest=first_last_rest,
                use_ee_offsets=use_ee_offsets
            )
            
            # Save to bytes
            buffer = io.BytesIO()
            safetensors.numpy.save({
                "ee_pos_l": strokebatch.ee_pos_l,
                "ee_pos_r": strokebatch.ee_pos_r,
                "ee_rot_l": strokebatch.ee_rot_l,
                "ee_rot_r": strokebatch.ee_rot_r,
                "joints": strokebatch.joints,
            }, buffer)
            
            strokebatch_bytes = buffer.getvalue()
            
            # Cache if requested
            if cache_path:
                try:
                    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
                    with open(cache_path, "wb") as f:
                        f.write(strokebatch_bytes)
                    log.info(f"Cached CPU-converted strokebatch to {cache_path}")
                except Exception as e:
                    log.warning(f"Failed to cache strokebatch: {e}")
            
            log.info("CPU fallback conversion successful")
            return True, strokebatch_bytes
            
        except Exception as e:
            log.error(f"CPU fallback conversion failed: {e}")
    
    log.error("All conversion methods failed")
    return False, None