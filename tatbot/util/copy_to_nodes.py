import argparse
import yaml
import os
import subprocess
import sys
import logging
from typing import List
from dataclasses import dataclass, field

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('copy-to-compute')

@dataclass
class CopyConfig:
    file_paths: List[str] = field(default_factory=list)
    root_dir: str = os.environ['TATBOT_ROOT']
    config_path: str = f"{root_dir}/config/compute.yaml"

def run(config: CopyConfig):
    try:
        with open(config.config_path, 'r') as f:
            compute_config = yaml.safe_load(f)
    except Exception as e:
        logger.error(f"Error loading compute.yaml: {e}")
        sys.exit(1)
    
    for file_path in config.file_paths:
        if not os.path.exists(file_path):
            logger.error(f"File not found: {file_path}")
            sys.exit(1)
    
    nodes_processed_count = 0
    for node_id, node_config in compute_config.items():
        logger.info(f"Processing node: {node_id}")
        if node_config:
            nodes_processed_count += 1
            ip = node_config.get('ip')
            username = node_config.get('username')
            device_name = node_config.get('device_name')
            
            password_var = f"{node_id.upper().replace('-', '_')}_PASSWORD"
            password = os.environ.get(password_var)
            
            if ip and username:
                logger.info(f"Attempting to connect to node {node_id} at IP: {ip}")
                for file_path in config.file_paths:
                    rel_path = os.path.relpath(os.path.abspath(file_path), config.root_dir)
                    
                    destination = f"{username}@{ip}:/home/{username}/tatbot/{rel_path}"
                    logger.info(f"Copying {rel_path} to {device_name} ({destination})...")
                    try:
                        scp_command = ['scp']
                        if os.path.isdir(file_path):
                            scp_command.append('-r')
                        
                        if password:
                            result = subprocess.run(
                                ['sshpass', '-p', password] + scp_command + [file_path, destination],
                                check=True,
                                stdout=subprocess.PIPE,
                                stderr=subprocess.PIPE
                            )
                        else:
                            logger.warning(f"No password found for {node_id}, using SSH key authentication")
                            result = subprocess.run(
                                scp_command + [file_path, destination],
                                check=True,
                                stdout=subprocess.PIPE,
                                stderr=subprocess.PIPE
                            )
                        logger.info(f"Successfully copied {rel_path} to {device_name}")
                    except subprocess.CalledProcessError as e:
                        logger.error(f"Error copying {rel_path} to {device_name}: {e}")
                        logger.error(f"Error details: {e.stderr.decode('utf-8')}")
            else:
                 if not ip:
                     logger.warning(f"Skipping node {node_id}: Missing IP address in config.")
                 elif not username:
                     logger.warning(f"Skipping node {node_id}: Missing username in config.")
                 else:
                     logger.warning(f"Skipping node {node_id}: Unknown reason (IP: {ip}, Username: {username})")
        else:
             logger.warning(f"Skipping node {node_id}: Invalid or empty configuration in compute.yaml.")

    if nodes_processed_count == 0:
        logger.warning("No valid compute nodes found or processed in the configuration.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Copy files to compute nodes')
    parser.add_argument('file_paths', nargs='+', help='List of file paths to copy to compute nodes')
    args = parser.parse_args()
    run(CopyConfig(file_paths=args.file_paths))