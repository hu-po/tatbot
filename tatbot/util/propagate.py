"""

Copy the .env file to [all] tatbot compute nodes:

> python scripts/oop/propagate.py --file-paths cfg/.env

"""

import yaml
import os
import subprocess
import sys
import logging
import tyro
from typing import List
from dataclasses import dataclass
from dotenv import load_dotenv

# Configure logger
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('copy-to-compute')

@dataclass
class CopyArgs:
    """Arguments for copying files to compute nodes."""
    file_paths: List[str]
    """List of file paths to copy to compute nodes."""

def copy_files_to_compute_nodes(file_paths: List[str]):
    # Path to the config file (relative to the script)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    config_path = os.path.join(project_root, 'cfg', 'compute.yaml')
    
    # Load the compute configuration
    try:
        with open(config_path, 'r') as f:
            compute_config = yaml.safe_load(f)
    except Exception as e:
        logger.error(f"Error loading compute.yaml: {e}")
        sys.exit(1)
    
    # Check if any files exist
    for file_path in file_paths:
        if not os.path.exists(file_path):
            logger.error(f"File not found: {file_path}")
            sys.exit(1)
    
    # Load environment variables from .env file (for passwords)
    env_path = os.path.join(project_root, 'cfg', '.env')
    if os.path.exists(env_path):
        load_dotenv(env_path)
    
    nodes_processed_count = 0
    # Copy each file to each enabled compute node
    for node_id, node_config in compute_config.items():
        logger.info(f"Processing node: {node_id}") # Log which node is being considered
        if node_config: # Check if node_config is not None or empty, REMOVED: and node_config.get('enabled', False)
            nodes_processed_count += 1
            ip = node_config.get('ip')
            username = node_config.get('username')
            device_name = node_config.get('device_name')
            
            # Get the corresponding password from environment variables
            password_var = f"{node_id.upper().replace('-', '_')}_PASSWORD"
            password = os.environ.get(password_var)
            
            if ip and username:
                logger.info(f"Attempting to connect to node {node_id} at IP: {ip}")
                for file_path in file_paths:
                    # Preserve the relative path structure instead of just using basename
                    # This assumes file_paths are relative to the project root
                    rel_path = os.path.relpath(os.path.abspath(file_path), project_root)
                    
                    # Destination on the compute node (preserving directory structure)
                    destination = f"{username}@{ip}:/home/{username}/dev/tatbot-dev/{rel_path}"
                    logger.info(f"Copying {rel_path} to {device_name} ({destination})...")
                    try:
                        if password:
                            # Use sshpass with password for authentication
                            result = subprocess.run(
                                ['sshpass', '-p', password, 'scp', file_path, destination], 
                                check=True,
                                stdout=subprocess.PIPE,
                                stderr=subprocess.PIPE
                            )
                        else:
                            # Fallback to regular scp (using SSH keys)
                            logger.warning(f"No password found for {node_id}, using SSH key authentication")
                            result = subprocess.run(
                                ['scp', file_path, destination], 
                                check=True,
                                stdout=subprocess.PIPE,
                                stderr=subprocess.PIPE
                            )
                        logger.info(f"Successfully copied {rel_path} to {device_name}")
                    except subprocess.CalledProcessError as e:
                        logger.error(f"Error copying {rel_path} to {device_name}: {e}")
                        logger.error(f"Error details: {e.stderr.decode('utf-8')}")
            else:
                 # Log why the node is being skipped more specifically
                 if not ip:
                     logger.warning(f"Skipping node {node_id}: Missing IP address in config.")
                 elif not username:
                     logger.warning(f"Skipping node {node_id}: Missing username in config.")
                 else: # Should not happen based on the 'if ip and username' check, but added for completeness
                     logger.warning(f"Skipping node {node_id}: Unknown reason (IP: {ip}, Username: {username})")
        else:
            # Node configuration itself is missing or invalid
             logger.warning(f"Skipping node {node_id}: Invalid or empty configuration in compute.yaml.")
            # REMOVED: Logging for skipping disabled nodes as the check is gone.

    if nodes_processed_count == 0:
        logger.warning("No valid compute nodes found or processed in the configuration.") # Updated message slightly

if __name__ == "__main__":
    args = tyro.cli(CopyArgs)
    copy_files_to_compute_nodes(args.file_paths)