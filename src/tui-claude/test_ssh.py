#!/usr/bin/env python3

import os
import sys
from pathlib import Path

# Add current directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from node_monitor import NodeMonitor

def test_ssh_connectivity():
    """Test SSH connectivity to all nodes"""
    print("Testing SSH connectivity to all nodes...\n")
    
    monitor = NodeMonitor()
    
    for node in monitor.nodes:
        print(f"Testing {node.name} ({node.ip}):")
        
        # Test SSH connection
        client = monitor.get_ssh_client(node)
        if client:
            print(f"  ✓ SSH connection successful")
            
            # Test a simple command
            output = monitor.run_ssh_command(node, "whoami")
            if output:
                print(f"  ✓ Command execution successful: {output}")
            else:
                print(f"  ✗ Command execution failed")
        else:
            print(f"  ✗ SSH connection failed")
            # Check if SSH keys exist
            for key_path in ['~/.ssh/id_rsa', '~/.ssh/id_ed25519', '~/.ssh/id_ecdsa']:
                expanded_path = os.path.expanduser(key_path)
                if os.path.exists(expanded_path):
                    print(f"    Found SSH key: {key_path}")
                    break
            else:
                print(f"    No SSH keys found in ~/.ssh/")
        
        print()
    
    monitor.close()

if __name__ == "__main__":
    test_ssh_connectivity()