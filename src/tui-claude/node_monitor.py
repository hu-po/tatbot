import concurrent.futures
import logging
import os
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import paramiko
import yaml

# Setup logging
LOG_DIR = Path(__file__).parent / "logs"
LOG_DIR.mkdir(exist_ok=True)
LOG_PATH = LOG_DIR / "tui.log"

# Configure logger (avoid duplicates)
logger = logging.getLogger('NodeMonitor')
if not logger.handlers:
    logger.setLevel(logging.DEBUG)
    file_handler = logging.FileHandler(LOG_PATH)
    file_handler.setLevel(logging.DEBUG)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

# Reduce paramiko logging verbosity
paramiko_logger = logging.getLogger('paramiko')
paramiko_logger.setLevel(logging.WARNING)


@dataclass
class NodeStats:
    """Simplified stats container"""
    online: bool = False
    cpu_load: Optional[float] = None
    cpu_cores: Optional[int] = None
    memory_used_gb: Optional[float] = None
    memory_total_gb: Optional[float] = None
    memory_percent: Optional[float] = None
    gpu_used_gb: Optional[float] = None
    gpu_total_gb: Optional[float] = None
    gpu_count: Optional[int] = None
    gpu_temp: Optional[float] = None
    gpu_util: Optional[float] = None
    data_source: str = "remote"  # "remote" or "meta"
    error: Optional[str] = None
    last_update: Optional[float] = None


@dataclass
class NodeInfo:
    """Node configuration and current stats"""
    name: str
    emoji: str
    ip: str
    user: str
    stats: NodeStats = None
    
    def __post_init__(self):
        if self.stats is None:
            self.stats = NodeStats()


class NodeMonitor:
    def __init__(self, config_path: str = "/home/oop/tatbot/src/conf/nodes.yaml"):
        logger.info(f"Initializing NodeMonitor with config: {config_path}")
        self.config_path = Path(config_path)
        self.nodes: List[NodeInfo] = []
        self.ssh_clients: Dict[str, paramiko.SSHClient] = {}
        self.metadata: Dict[str, Dict] = {}
        
        try:
            self.load_config()
            self.load_metadata()
            logger.info(f"NodeMonitor initialized successfully with {len(self.nodes)} nodes")
        except Exception as e:
            logger.error(f"Failed to initialize NodeMonitor: {e}", exc_info=True)
            raise
    
    def load_config(self):
        """Load node configuration"""
        try:
            logger.debug(f"Loading config from: {self.config_path}")
            if not self.config_path.exists():
                raise FileNotFoundError(f"Config file not found: {self.config_path}")
            
            with open(self.config_path, 'r') as f:
                config = yaml.safe_load(f)
            
            logger.debug(f"Loaded config: {config}")
            
            if 'nodes' not in config:
                raise ValueError("Config file missing 'nodes' section")
            
            self.nodes = []
            for i, node in enumerate(config['nodes']):
                try:
                    node_info = NodeInfo(
                        name=node['name'],
                        emoji=node['emoji'],
                        ip=node.get('hostname', node.get('ip', node['name'])),
                        user=node['user']
                    )
                    self.nodes.append(node_info)
                    logger.debug(f"Loaded node {i+1}: {node_info.name} ({node_info.ip})")
                except KeyError as e:
                    logger.error(f"Node {i+1} missing required field: {e}")
                    raise
            
            logger.info(f"Successfully loaded {len(self.nodes)} nodes")
        except Exception as e:
            logger.error(f"Failed to load config: {e}", exc_info=True)
            raise
    
    def load_metadata(self):
        """Load hardware metadata for fallback"""
        meta_path = Path(__file__).parent / "nodes_meta.yaml"
        logger.debug(f"Looking for metadata at: {meta_path}")
        
        if not meta_path.exists():
            logger.warning(f"Metadata file not found: {meta_path}")
            return
        
        try:
            with open(meta_path, 'r') as f:
                meta_config = yaml.safe_load(f)
            
            self.metadata = meta_config.get('nodes_meta', {})
            logger.info(f"Loaded metadata for {len(self.metadata)} nodes")
            
            for name, meta in self.metadata.items():
                logger.debug(f"Metadata for {name}: {meta}")
        except Exception as e:
            logger.error(f"Failed to load metadata: {e}", exc_info=True)
    
    def get_ssh_client(self, node: NodeInfo) -> Optional[paramiko.SSHClient]:
        """Get or create SSH client with environment variable support"""
        if node.name in self.ssh_clients:
            # Check if connection is still alive
            try:
                transport = self.ssh_clients[node.name].get_transport()
                if transport and transport.is_active():
                    return self.ssh_clients[node.name]
            except Exception as e:
                logger.debug(f"Connection to {node.name} is dead: {e}")
            # Remove dead connection
            del self.ssh_clients[node.name]
        
        try:
            logger.debug(f"Establishing SSH connection to {node.name} ({node.ip})")
            client = paramiko.SSHClient()
            client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
            
            connect_kwargs = {
                'hostname': node.ip,
                'username': node.user,
                'port': int(os.environ.get('TATBOT_TUI_SSH_PORT', '22')),
                'timeout': 5,  # Increased from 3 to 5 for better reliability
                'look_for_keys': True,
                'allow_agent': True,
                'banner_timeout': 10,  # Add banner timeout
                'auth_timeout': 10     # Add auth timeout
            }
            
            ssh_key = os.environ.get('TATBOT_TUI_SSH_KEY')
            if ssh_key:
                connect_kwargs['key_filename'] = ssh_key
                logger.debug(f"Using SSH key: {ssh_key}")
            else:
                # Try common SSH key locations (skip problematic DSA keys)
                for key_path in ['~/.ssh/id_rsa', '~/.ssh/id_ed25519', '~/.ssh/id_ecdsa']:
                    expanded_path = os.path.expanduser(key_path)
                    if os.path.exists(expanded_path):
                        connect_kwargs['key_filename'] = expanded_path
                        logger.debug(f"Using default SSH key: {expanded_path}")
                        break
                
                # Disable problematic algorithms
                connect_kwargs['disabled_algorithms'] = {
                    'pubkeys': ['ssh-dss']  # Disable DSA which causes issues
                }
            
            client.connect(**connect_kwargs)
            self.ssh_clients[node.name] = client
            logger.debug(f"SSH connection to {node.name} established")
            return client
            
        except paramiko.AuthenticationException as e:
            logger.debug(f"SSH authentication to {node.name} failed: {e}")
            return None
        except (paramiko.SSHException, OSError, ValueError) as e:
            logger.debug(f"SSH connection to {node.name} failed: {e}")
            return None
        except Exception as e:
            logger.warning(f"Unexpected SSH error for {node.name}: {e}")
            return None
    
    def run_ssh_command(self, node: NodeInfo, command: str) -> Optional[str]:
        """Execute SSH command and return output"""
        client = self.get_ssh_client(node)
        if not client:
            logger.debug(f"No SSH client available for {node.name}")
            return None
        
        try:
            logger.debug(f"Running command on {node.name}: {command}")
            stdin, stdout, stderr = client.exec_command(command, timeout=3)
            output = stdout.read().decode().strip()
            error = stderr.read().decode().strip()
            
            if error:
                logger.debug(f"Command stderr on {node.name}: {error}")
            
            return output
        except Exception as e:
            logger.debug(f"Command failed on {node.name}: {e}")
            return None
    
    def check_online(self, node: NodeInfo) -> bool:
        """Check online status - try SSH first, fallback to ping"""
        try:
            # Try a quick SSH connection test first (more reliable than ping)
            client = self.get_ssh_client(node)
            if client:
                logger.debug(f"{node.name} online via SSH")
                return True
            
            # Fallback to ping if SSH fails
            logger.debug(f"SSH failed, trying ping for {node.name} ({node.ip})")
            result = subprocess.run(
                ['ping', '-c', '1', '-W', '1', node.ip],
                capture_output=True, timeout=2
            )
            online = result.returncode == 0
            logger.debug(f"Ping {node.name}: {'SUCCESS' if online else 'FAILED'}")
            return online
        except Exception as e:
            logger.debug(f"Online check for {node.name} failed: {e}")
            return False
    
    def get_cpu_stats(self, node: NodeInfo) -> Tuple[Optional[float], Optional[int]]:
        """Get CPU load and core count"""
        # Try physical cores first
        cores_cmd = ("lscpu -p=CORE,SOCKET 2>/dev/null | egrep -v '^#' | "
                    "awk -F, '{print $1\"-\"$2}' | sort -u | wc -l")
        cores_output = self.run_ssh_command(node, cores_cmd)
        
        cpu_cores = None
        if cores_output and cores_output.isdigit() and int(cores_output) > 0:
            cpu_cores = int(cores_output)
        else:
            # Fallback to logical cores
            nproc_output = self.run_ssh_command(node, 'nproc --all')
            if nproc_output and nproc_output.isdigit():
                cpu_cores = int(nproc_output)
        
        # Get load average
        load_output = self.run_ssh_command(node, 'uptime')
        if not load_output:
            return None, cpu_cores
        
        try:
            load_avg = load_output.split('load average:')[1].strip()
            load_1min = float(load_avg.split(',')[0])
            
            if cpu_cores:
                cpu_load = (load_1min / cpu_cores) * 100
            else:
                cpu_load = load_1min * 100
            
            return cpu_load, cpu_cores
        except:
            return None, cpu_cores
    
    def get_memory_stats(self, node: NodeInfo) -> Tuple[Optional[float], Optional[float], Optional[float]]:
        """Get memory usage in GB"""
        output = self.run_ssh_command(node, 'free -b')
        if not output:
            return None, None, None
        
        try:
            lines = output.split('\n')
            mem_line = lines[1].split()
            
            total_bytes = int(mem_line[1])
            used_bytes = int(mem_line[2])
            
            total_gb = total_bytes / (1024**3)
            used_gb = used_bytes / (1024**3)
            percent = (used_bytes / total_bytes) * 100 if total_bytes > 0 else 0
            
            return used_gb, total_gb, percent
        except:
            return None, None, None
    
    def get_gpu_stats(self, node: NodeInfo) -> Tuple[Optional[float], Optional[float], Optional[int], Optional[float], Optional[float]]:
        """Get GPU stats: used_gb, total_gb, count, temp, util"""
        cmd = ('nvidia-smi --query-gpu=memory.used,memory.total,temperature.gpu,utilization.gpu '
               '--format=csv,noheader,nounits 2>/dev/null')
        output = self.run_ssh_command(node, cmd)
        
        if not output:
            return None, None, None, None, None
        
        try:
            lines = output.strip().split('\n')
            total_used_mb = 0
            total_memory_mb = 0
            gpu_count = 0
            max_temp = 0
            avg_util = 0
            
            for line in lines:
                if not line or line.startswith('No devices'):
                    continue
                
                parts = [p.strip() for p in line.split(',')]
                if len(parts) >= 4:
                    used_mb = float(parts[0])
                    total_mb = float(parts[1])
                    temp = float(parts[2]) if parts[2] else 0
                    util = float(parts[3]) if parts[3] else 0
                    
                    total_used_mb += used_mb
                    total_memory_mb += total_mb
                    max_temp = max(max_temp, temp)
                    avg_util += util
                    gpu_count += 1
            
            if gpu_count == 0:
                return None, None, None, None, None
            
            return (
                total_used_mb / 1024,  # GB
                total_memory_mb / 1024,  # GB
                gpu_count,
                max_temp,
                avg_util / gpu_count
            )
        except:
            return None, None, None, None, None
    
    def apply_metadata_fallback(self, node: NodeInfo):
        """Apply metadata when remote data unavailable"""
        meta = self.metadata.get(node.name, {})
        if not meta:
            return
        
        stats = node.stats
        
        # Apply CPU metadata if not available from remote
        if stats.cpu_cores is None and meta.get('cpu_cores'):
            stats.cpu_cores = meta['cpu_cores']
            if stats.data_source == "remote" and not stats.online:
                stats.data_source = "meta"
        
        # Apply GPU metadata if not available from remote  
        if stats.gpu_total_gb is None and meta.get('gpu_total_mb'):
            stats.gpu_total_gb = meta['gpu_total_mb'] / 1024
            stats.gpu_count = meta.get('gpu_count', 1)
            if stats.data_source == "remote" and not stats.online:
                stats.data_source = "meta"
    
    def update_node(self, node: NodeInfo):
        """Update a single node's stats"""
        try:
            logger.debug(f"Updating node: {node.name}")
            stats = node.stats
            stats.online = self.check_online(node)
            stats.data_source = "remote"
            stats.last_update = time.time()
            
            if stats.online:
                logger.debug(f"{node.name} is online, gathering stats")
                # Get all stats
                stats.cpu_load, stats.cpu_cores = self.get_cpu_stats(node)
                stats.memory_used_gb, stats.memory_total_gb, stats.memory_percent = self.get_memory_stats(node)
                stats.gpu_used_gb, stats.gpu_total_gb, stats.gpu_count, stats.gpu_temp, stats.gpu_util = self.get_gpu_stats(node)
                
                stats.error = None
                logger.debug(f"{node.name} stats: CPU={stats.cpu_load}%, MEM={stats.memory_used_gb}GB, GPU={stats.gpu_used_gb}GB")
            else:
                logger.debug(f"{node.name} is offline")
                # Reset remote stats but keep metadata
                stats.cpu_load = None
                stats.memory_used_gb = None
                stats.memory_total_gb = None
                stats.memory_percent = None
                stats.gpu_used_gb = None
                stats.gpu_temp = None
                stats.gpu_util = None
                stats.error = "Offline"
            
            # Apply metadata fallback
            self.apply_metadata_fallback(node)
            logger.debug(f"Updated {node.name} successfully")
            
        except Exception as e:
            logger.error(f"Failed to update node {node.name}: {e}", exc_info=True)
            if node.stats:
                node.stats.error = f"Update failed: {e}"
    
    def update_all_nodes(self):
        """Update all nodes in parallel with reduced connection overhead"""
        try:
            logger.debug(f"Updating all {len(self.nodes)} nodes in parallel")
            # Limit concurrent SSH connections to prevent "MaxStartups exceeded"
            max_workers = min(3, len(self.nodes))  # Reduced from 8 to 3
            with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
                futures = [executor.submit(self.update_node, node) for node in self.nodes]
                concurrent.futures.wait(futures)
            logger.debug("All nodes updated successfully")
        except Exception as e:
            logger.error(f"Failed to update nodes: {e}", exc_info=True)
    
    def close(self):
        """Close all SSH connections"""
        for client in self.ssh_clients.values():
            try:
                client.close()
            except:
                pass
        self.ssh_clients.clear()