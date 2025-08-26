#!/usr/bin/env python3
"""Utility to extract node configuration, particularly extras dependencies."""

import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml


def get_node_extras(node: str, config_root: Optional[Path] = None) -> List[str]:
    """Get the extras list for a given node from its MCP configuration.
    
    Args:
        node: The node name (e.g., 'ook', 'oop', 'eek')
        config_root: Root path to config directory. If None, attempts to find it automatically.
        
    Returns:
        List of extras (e.g., ['bot', 'cam', 'gen'])
        
    Raises:
        FileNotFoundError: If config file doesn't exist
        ValueError: If config is malformed
    """
    if config_root is None:
        # Try to find config directory relative to this file
        current_dir = Path(__file__).parent
        # Go up to tatbot/src/tatbot/utils -> tatbot/src -> tatbot -> tatbot/src/conf
        config_root = current_dir.parent.parent / "conf"
        
        # If that doesn't work, try from current working directory
        if not config_root.exists():
            config_root = Path.cwd() / "src" / "conf"
            
        # Last resort: relative to home
        if not config_root.exists():
            config_root = Path.home() / "tatbot" / "src" / "conf"
    
    config_path = config_root / "mcp" / f"{node}.yaml"
    
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file for node '{node}' not found at {config_path}")
    
    with open(config_path) as f:
        config = yaml.safe_load(f)
    
    # Handle defaults inheritance
    if 'defaults' in config:
        # Load default config first
        default_path = config_root / "mcp" / "default.yaml"
        if default_path.exists():
            with open(default_path) as f:
                default_config = yaml.safe_load(f)
            # Merge configs (node config overrides defaults)
            extras = default_config.get('extras', [])
            extras.extend(config.get('extras', []))
            # Remove duplicates while preserving order
            seen = set()
            final_extras = []
            for extra in extras:
                if extra not in seen:
                    seen.add(extra)
                    final_extras.append(extra)
            return final_extras
    
    return config.get('extras', [])


def get_current_node() -> Optional[str]:
    """Get the current node name based on username or hostname.
    
    Returns:
        Node name if detected, None otherwise
    """
    # Try username first (as used in mcp_run.sh)
    username = os.environ.get('USER', '')
    if username:
        # Check if config exists for this username
        try:
            get_node_extras(username)
            return username
        except FileNotFoundError:
            pass
    
    # Fallback to hostname
    try:
        import socket
        hostname = socket.gethostname()
        get_node_extras(hostname)
        return hostname
    except (FileNotFoundError, ImportError):
        pass
    
    return None


def load_node_ips(config_root: Optional[Path] = None) -> Dict[str, str]:
    """Load node IP addresses from nodes.yaml configuration.
    
    Args:
        config_root: Root path to config directory. If None, attempts to find it automatically.
        
    Returns:
        Dictionary mapping node names to IP addresses
        
    Raises:
        FileNotFoundError: If nodes.yaml doesn't exist
    """
    if config_root is None:
        # Try to find config directory relative to this file
        current_dir = Path(__file__).parent
        # Go up to tatbot/src/tatbot/utils -> tatbot/src -> tatbot -> tatbot/src/conf
        config_root = current_dir.parent.parent / "conf"
        
        # If that doesn't work, try from current working directory
        if not config_root.exists():
            config_root = Path.cwd() / "src" / "conf"
            
        # Last resort: relative to home
        if not config_root.exists():
            config_root = Path.home() / "tatbot" / "src" / "conf"
    
    nodes_path = config_root / "nodes.yaml"
    
    if not nodes_path.exists():
        raise FileNotFoundError(f"Nodes configuration not found at {nodes_path}")
    
    with open(nodes_path) as f:
        config = yaml.safe_load(f)
    
    node_ips = {}
    for node in config.get('nodes', []):
        node_ips[node['name']] = node['ip']
    
    return node_ips






def main():
    """CLI interface for getting node extras."""
    if len(sys.argv) < 2:
        # Try to auto-detect node
        node = get_current_node()
        if not node:
            print("Usage: python -m tatbot.utils.node_config <node_name>", file=sys.stderr)
            print("Available nodes: ook, oop, rpi1, rpi2, ojo, eek, hog", file=sys.stderr)
            sys.exit(1)
    else:
        node = sys.argv[1]
    
    try:
        extras = get_node_extras(node)
        print(','.join(extras))
    except FileNotFoundError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"Error loading config for node '{node}': {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
