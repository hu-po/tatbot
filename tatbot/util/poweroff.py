import logging
import os
import subprocess
import sys
from typing import List
from dataclasses import dataclass, field

import yaml

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s|%(name)s|%(levelname)s|%(message)s',
    datefmt='%H:%M:%S',
)
log = logging.getLogger(__name__)
log.setLevel(logging.INFO)

@dataclass
class PoweroffConfig:
    """Configuration for shutting down compute nodes."""
    root_dir: str = os.environ['TATBOT_ROOT']
    config_path: str = f"{root_dir}/config/compute.yaml"

def run(config: PoweroffConfig):
    log.info("Starting compute node shutdown process...")
    log.info(f"Loading compute configuration from: {config.config_path}")

    # Load the compute configuration
    try:
        with open(config.config_path, 'r') as f:
            compute_config = yaml.safe_load(f)
    except Exception as e:
        log.error(f"Error loading compute.yaml: {e}")
        sys.exit(1)

    nodes_processed_count = 0
    for node_id, node_config in compute_config.items():
        log.info(f"Processing node: {node_id}")
        if node_config:
            nodes_processed_count += 1
            ip = node_config.get('ip')
            username = node_config.get('username')
            device_name = node_config.get('device_name', node_id)

            # Get the corresponding password from environment variables
            password_var = f"{node_id.upper().replace('-', '_')}_PASSWORD"
            password = os.environ.get(password_var)

            if ip and username:
                log.info(f"Attempting shutdown on {device_name} ({username}@{ip})...")
                try:
                    # Use -t -t to force pseudo-terminal allocation for sudo
                    ssh_command_base = ['ssh', '-t', '-t', f"{username}@{ip}"]
                    log.debug(f"Base SSH command: {' '.join(ssh_command_base)}")
                    
                    # Command to execute remotely
                    remote_command_sudo = f"echo '{password}' | sudo -S poweroff"
                    remote_command_nosudo = "sudo poweroff"

                    if password:
                        log.info(f"Using password authentication for {device_name}.")
                        log.debug(f"Executing remote command (sudo): {remote_command_sudo}")
                        try:
                            result = subprocess.run(
                                ['sshpass', '-p', password] + ssh_command_base + [remote_command_sudo],
                                check=True,
                                stdout=subprocess.PIPE,
                                stderr=subprocess.PIPE,
                                text=True
                            )
                        except subprocess.CalledProcessError as e:
                            # Check if this is a connection termination (exit code 255)
                            if e.returncode == 255 and "Connection to" in e.stderr and "closed" in e.stderr:
                                log.info(f"Connection closed by {device_name} (expected during shutdown)")
                                log.debug(f"Connection details: {e.stderr.strip()}")
                            else:
                                raise  # Re-raise if it's a different type of error
                    else:
                        log.warning(f"No password found for {node_id}. Using SSH key authentication.")
                        log.warning(f"  Ensure sudo is configured for passwordless 'poweroff' for user '{username}' on {device_name}, or this will fail.")
                        log.debug(f"Executing remote command (no sudo password): {remote_command_nosudo}")
                        try:
                            result = subprocess.run(
                                ssh_command_base + [remote_command_nosudo],
                                check=True,
                                stdout=subprocess.PIPE,
                                stderr=subprocess.PIPE,
                                text=True
                            )
                        except subprocess.CalledProcessError as e:
                            # Check if this is a connection termination (exit code 255)
                            if e.returncode == 255 and "Connection to" in e.stderr and "closed" in e.stderr:
                                log.info(f"Connection closed by {device_name} (expected during shutdown)")
                                log.debug(f"Connection details: {e.stderr.strip()}")
                            else:
                                raise  # Re-raise if it's a different type of error

                    log.info(f"Shutdown command successfully sent to {device_name}.")
                    log.debug(f"Result stdout: {result.stdout.strip() if 'result' in locals() else ''}")
                    log.debug(f"Result stderr: {result.stderr.strip() if 'result' in locals() else ''}")

                except subprocess.CalledProcessError as e:
                    log.error(f"Error shutting down {device_name}: {e}")
                    log.error(f"Command: {' '.join(e.cmd)}")
                    log.error(f"Stderr: {e.stderr.strip()}")
                except FileNotFoundError:
                    log.error(f"Error: 'sshpass' command not found. Please install sshpass or use SSH key authentication.")
                    if password:
                        sys.exit(1)
            else:
                if not ip:
                    log.warning(f"Skipping node {node_id}: Missing IP address in config.")
                elif not username:
                    log.warning(f"Skipping node {node_id}: Missing username in config.")
                else:
                    log.warning(f"Skipping node {node_id}: Unknown reason (IP: {ip}, Username: {username})")
        else:
            log.warning(f"Skipping node {node_id}: Invalid or empty configuration in compute.yaml.")

    if nodes_processed_count == 0:
        log.warning("No valid compute nodes found or processed in the configuration.")

    log.info("Compute node shutdown process finished.")

if __name__ == "__main__":
    run(PoweroffConfig()) 