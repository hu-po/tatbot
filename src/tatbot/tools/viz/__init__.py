"""Visualization tools for remote viser server control via MCP."""

import threading
from typing import Any, Dict, Optional, Tuple

from tatbot.utils.log import get_logger

log = get_logger("tools.viz", "🎨")

# Import all viz tools to register them
__all__ = ["stroke_viz", "teleop_viz", "map_viz", "control"]

# Server registry to track running viser servers
# Format: {name: (viz_instance, thread)}
_VIZ_SERVERS: Dict[str, Tuple[Any, threading.Thread]] = {}
_REGISTRY_LOCK = threading.Lock()


def get_server(name: str) -> Optional[Any]:
    """Get a running viz server by name."""
    with _REGISTRY_LOCK:
        if name in _VIZ_SERVERS:
            return _VIZ_SERVERS[name][0]  # Return the viz instance
        return None


def register_server(name: str, server: Any, thread: threading.Thread) -> bool:
    """Register a running viz server with its thread. Returns False if already exists."""
    with _REGISTRY_LOCK:
        if name in _VIZ_SERVERS:
            return False
        _VIZ_SERVERS[name] = (server, thread)
    log.info(f"Registered viz server: {name}")
    return True


def stop_server(name: str) -> bool:
    """Stop and unregister a viz server."""
    # Extract server info under lock
    with _REGISTRY_LOCK:
        if name not in _VIZ_SERVERS:
            return False
        viz_instance, thread = _VIZ_SERVERS.pop(name)
    
    # Stop operations outside lock to avoid blocking other operations
    # Stop the viz instance
    try:
        if hasattr(viz_instance, "stop"):
            viz_instance.stop()
        else:
            log.warning(f"Viz instance {name} has no stop method")
    except Exception as e:
        log.error(f"Error stopping viz instance {name}: {e}")
    
    # Wait for thread to finish
    try:
        if thread.is_alive():
            thread.join(timeout=5.0)  # Wait up to 5 seconds
            if thread.is_alive():
                log.warning(f"Thread for {name} did not stop within timeout")
    except Exception as e:
        log.error(f"Error joining thread for {name}: {e}")
    
    log.info(f"Stopped viz server: {name}")
    return True


def list_servers() -> list:
    """List all running viz servers."""
    with _REGISTRY_LOCK:
        return list(_VIZ_SERVERS.keys())


def get_server_url(name: str, node_name: str) -> Optional[str]:
    """Get the URL for a running server."""
    with _REGISTRY_LOCK:
        if name not in _VIZ_SERVERS:
            return None
        viz_instance, _ = _VIZ_SERVERS[name]
    
    # Load node config to get host
    from tatbot.tools.registry import _load_node_config
    node_config = _load_node_config(node_name)
    host = node_config.get("host", "localhost")
    
    return f"http://{host}:{viz_instance.server.get_port()}"


def get_server_status(name: str, node_name: str) -> dict:
    """Get detailed status information for a server."""
    with _REGISTRY_LOCK:
        if name not in _VIZ_SERVERS:
            return {
                "server_name": name,
                "running": False,
                "server_url": None,
                "host": None,
                "port": None,
                "thread_alive": None,
                "started_at": None,
            }
        
        viz_instance, thread = _VIZ_SERVERS[name]
    
    # Load node config to get host
    from tatbot.tools.registry import _load_node_config
    node_config = _load_node_config(node_name)
    host = node_config.get("host", "localhost")
    
    # Get server details
    port = viz_instance.server.get_port() if hasattr(viz_instance, 'server') else None
    server_url = f"http://{host}:{port}" if port else None
    thread_alive = thread.is_alive() if thread else None
    
    return {
        "server_name": name,
        "running": True,
        "server_url": server_url,
        "host": host,
        "port": port,
        "thread_alive": thread_alive,
        "started_at": None,  # Could add timestamp tracking later
    }