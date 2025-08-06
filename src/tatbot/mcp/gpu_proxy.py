"""GPU proxy for cross-node stroke conversion."""

import asyncio
import base64
import os
import random
from typing import Any, Dict, List, Optional, Tuple

import aiohttp
import yaml

from tatbot.utils.log import get_logger
from tatbot.utils.net import NetworkManager

log = get_logger("mcp.gpu_proxy", "🎯🔗")


class GPUProxy:
    """Proxy for GPU-accelerated stroke conversion on remote nodes."""
    
    def __init__(self, timeout: int = 60):
        """Initialize GPU proxy.
        
        Args:
            timeout: Request timeout in seconds
        """
        self.timeout = timeout
        self.net = NetworkManager()
        
    def _get_gpu_nodes(self) -> List[str]:
        """Get list of nodes with GPU support.
        
        Returns:
            List of node names with GPU extras
        """
        gpu_nodes = []
        for node in self.net.nodes:
            if "gpu" in node.extras:
                gpu_nodes.append(node.name)
        return gpu_nodes
    
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
        node = next((n for n in self.net.nodes if n.name == node_name), None)
        if not node:
            log.error(f"Node {node_name} not found")
            return False, {"error": f"Node {node_name} not found"}
        
        url = f"http://{node.host}:{node.port}/mcp/tool/{tool_name}"
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    url,
                    json=input_data,
                    timeout=aiohttp.ClientTimeout(total=self.timeout)
                ) as response:
                    if response.status == 200:
                        data = await response.json()
                        return True, data
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
        
        # Prepare input data
        input_data = {
            "strokes_yaml": strokes_yaml,
            "scene_name": scene_name,
            "first_last_rest": first_last_rest,
            "use_ee_offsets": use_ee_offsets
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
        target_nodes = nodes or [n.name for n in self.net.nodes]
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
        node = next((n for n in self.net.nodes if n.name == node_name), None)
        if not node:
            return False
        
        url = f"http://{node.host}:{node.port}/health"
        
        try:
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
        
        cfg = hydra.compose(config_name="config")
        hostname = socket.gethostname()
        
        # Map hostname to node name
        node_name = hostname.lower()
        if node_name in cfg.mcp:
            node_cfg = cfg.mcp[node_name]
        else:
            node_cfg = cfg.mcp.get("default", {})
        
        return "gpu" in node_cfg.get("extras", [])
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