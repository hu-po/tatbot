"""
SSH into each enabled compute node and shut it down.

> python scripts/oop/shutdown.py
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
logger = logging.getLogger('shutdown-compute')

@dataclass
class ShutdownArgs:
    """Arguments for shutting down compute nodes."""
    # No arguments needed for now, but keep the structure for consistency
    pass

def shutdown_compute_nodes():
    logger.info("Starting compute node shutdown process...")
    # Path to the config file (relative to the script)
    project_root = os.path.expanduser("~/dev/tatbot-dev")
    config_path = os.path.join(project_root, 'cfg', 'compute.yaml')
    logger.info(f"Loading compute configuration from: {config_path}")

    # Load the compute configuration
    try:
        with open(config_path, 'r') as f:
            compute_config = yaml.safe_load(f)
    except Exception as e:
        logger.error(f"Error loading compute.yaml: {e}")
        sys.exit(1)

    # Load environment variables from .env file (for passwords)
    env_path = os.path.join(project_root, 'cfg', '.env')
    if os.path.exists(env_path):
        load_dotenv(env_path)
    else:
        logger.warning(f".env file not found at {env_path}. Proceeding without passwords (assuming SSH key auth).")

    # Shutdown each enabled compute node
    for node_id, node_config in compute_config.items():
        logger.info(f"Processing node: {node_id}")
        ip = node_config.get('ip')
        username = node_config.get('username')
        device_name = node_config.get('device_name', node_id) # Use node_id as fallback for name

        # Get the corresponding password from environment variables
        password_var = f"{node_id.upper().replace('-', '_')}_PASSWORD"
        password = os.environ.get(password_var)

        if ip and username:
            logger.info(f"Attempting shutdown on {device_name} ({username}@{ip})...")
            try:
                # Use -t -t to force pseudo-terminal allocation for sudo
                ssh_command_base = ['ssh', '-t', '-t', f"{username}@{ip}"]
                logger.debug(f"Base SSH command: {' '.join(ssh_command_base)}")
                # Command to execute remotely. Pass password to sudo via stdin.
                remote_command_sudo = f"echo '{password}' | sudo -S poweroff"
                remote_command_nosudo = "sudo poweroff" # For key-based auth

                if password:
                    # Use sshpass with password for authentication
                    # Execute the remote command requiring sudo password
                    logger.info(f"Using password authentication for {device_name}.")
                    logger.debug(f"Executing remote command (sudo): {remote_command_sudo}")
                    result = subprocess.run(
                        ['sshpass', '-p', password] + ssh_command_base + [remote_command_sudo],
                        check=True,
                        stdout=subprocess.PIPE,
                        stderr=subprocess.PIPE,
                        text=True # Capture output as text
                    )
                else:
                    # Fallback to regular ssh (using SSH keys)
                    # Sudo might require passwordless setup on the remote machine
                    logger.warning(f"No password found for {node_id}. Using SSH key authentication.")
                    logger.warning(f"  Ensure sudo is configured for passwordless 'poweroff' for user '{username}' on {device_name}, or this will fail.")
                    logger.debug(f"Executing remote command (no sudo password): {remote_command_nosudo}")
                    result = subprocess.run(
                        ssh_command_base + [remote_command_nosudo],
                        check=True,
                        stdout=subprocess.PIPE,
                        stderr=subprocess.PIPE,
                        text=True # Capture output as text
                    )

                # Check stdout/stderr for sudo prompts or success messages (may vary)
                logger.info(f"Shutdown command successfully sent to {device_name}.")
                logger.debug(f"Result stdout: {result.stdout.strip()}")
                logger.debug(f"Result stderr: {result.stderr.strip()}")

            except subprocess.CalledProcessError as e:
                logger.error(f"Error shutting down {device_name}: {e}")
                logger.error(f"Command: {' '.join(e.cmd)}")
                logger.error(f"Stderr: {e.stderr.strip()}")
                # Continue to the next node
            except FileNotFoundError:
                logger.error(f"Error: 'sshpass' command not found. Please install sshpass or use SSH key authentication.")
                # Stop if sshpass is needed but not found
                if password:
                    sys.exit(1)
        else:
            logger.warning(f"Skipping node {node_id}: Missing IP or username")

    logger.info("Compute node shutdown process finished.")

if __name__ == "__main__":
    # Parse arguments even if none are defined, for consistency
    args = tyro.cli(ShutdownArgs)
    shutdown_compute_nodes() 