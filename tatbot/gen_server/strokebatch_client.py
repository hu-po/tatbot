#!/usr/bin/env python3
"""
Strokebatch client for tatbot.

This client calls the strokebatch server on ojo to compute strokebatches.
It has minimal dependencies and can be used from the robot.

Usage:
    python scripts/strokebatch_client.py <scene_path> <strokelist_path> [--output OUTPUT] [--server SERVER]
"""

import os
import sys
import argparse
from pathlib import Path
from typing import Optional

try:
    import requests
except ImportError:
    print("❌ Requests not installed. Install with: pip install requests")
    sys.exit(1)

def call_strokebatch_server(
    scene_path: str,
    strokelist_path: str,
    server_url: str = "http://ojo:5000",
    output_path: Optional[str] = None
) -> str:
    """
    Call the strokebatch server to compute a strokebatch.
    
    Args:
        scene_path: Path to the scene YAML file
        strokelist_path: Path to the strokelist YAML file
        server_url: URL of the strokebatch server
        output_path: Path to save the result (default: strokebatch.safetensors)
    
    Returns:
        Path to the saved strokebatch file
    
    Raises:
        Exception: If the server request fails
    """
    # Validate input files
    if not os.path.exists(scene_path):
        raise FileNotFoundError(f"Scene file not found: {scene_path}")
    if not os.path.exists(strokelist_path):
        raise FileNotFoundError(f"Strokelist file not found: {strokelist_path}")
    
    # Set default output path
    if output_path is None:
        output_path = "strokebatch.safetensors"
    
    print(f"📤 Sending request to {server_url}/compute_strokebatch")
    print(f"   Scene: {scene_path}")
    print(f"   Strokelist: {strokelist_path}")
    print(f"   Output: {output_path}")
    
    try:
        # Prepare files for upload
        with open(scene_path, 'rb') as scene_file, open(strokelist_path, 'rb') as strokelist_file:
            files = {
                'scene': (os.path.basename(scene_path), scene_file, 'application/x-yaml'),
                'strokelist': (os.path.basename(strokelist_path), strokelist_file, 'application/x-yaml')
            }
            
            # Make the request
            response = requests.post(
                f"{server_url}/compute_strokebatch",
                files=files,
                timeout=300  # 5 minute timeout for computation
            )
        
        # Check response
        if response.status_code == 200:
            # Save the result
            with open(output_path, 'wb') as f:
                f.write(response.content)
            
            print(f"✅ Strokebatch saved to {output_path}")
            return output_path
            
        else:
            # Try to get error details
            try:
                error_data = response.json()
                error_msg = error_data.get('error', 'Unknown error')
                details = error_data.get('details', '')
                if details:
                    error_msg += f": {details}"
            except:
                error_msg = f"HTTP {response.status_code}: {response.text}"
            
            raise Exception(f"Server error: {error_msg}")
    
    except requests.exceptions.Timeout:
        raise Exception("Request timed out (server took too long to respond)")
    except requests.exceptions.ConnectionError:
        raise Exception(f"Could not connect to server at {server_url}")
    except requests.exceptions.RequestException as e:
        raise Exception(f"Request failed: {e}")

def health_check(server_url: str = "http://ojo:5000") -> bool:
    """
    Check if the strokebatch server is healthy.
    
    Args:
        server_url: URL of the strokebatch server
    
    Returns:
        True if server is healthy, False otherwise
    """
    try:
        response = requests.get(f"{server_url}/health", timeout=10)
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Server healthy: {data}")
            return True
        else:
            print(f"❌ Server unhealthy: HTTP {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Cannot reach server: {e}")
        return False

def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description='Strokebatch client')
    parser.add_argument('scene_path', help='Path to scene YAML file')
    parser.add_argument('strokelist_path', help='Path to strokelist YAML file')
    parser.add_argument('--output', '-o', help='Output path for strokebatch (default: strokebatch.safetensors)')
    parser.add_argument('--server', '-s', default='http://ojo:5000', help='Server URL (default: http://ojo:5000)')
    parser.add_argument('--health', action='store_true', help='Check server health and exit')
    
    args = parser.parse_args()
    
    if args.health:
        success = health_check(args.server)
        sys.exit(0 if success else 1)
    
    try:
        result_path = call_strokebatch_server(
            scene_path=args.scene_path,
            strokelist_path=args.strokelist_path,
            server_url=args.server,
            output_path=args.output
        )
        print(f"🎉 Success! Strokebatch saved to: {result_path}")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        sys.exit(1)

if __name__ == '__main__':
    main() 