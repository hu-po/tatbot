import concurrent.futures
import os
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import paramiko
import yaml
from ssh_pool import get_pool


@dataclass
class GpuDevice:
    index: int
    name: str
    memory_used_mb: float
    memory_total_mb: float
    utilization_percent: float
    temperature_c: float
    memory_percent: float

@dataclass
class NodeMeta:
    cpu_cores: Optional[int] = None
    gpu_total_mb: Optional[int] = None
    gpu_count: Optional[int] = None

@dataclass
class NodeInfo:
    name: str
    emoji: str
    ip: str
    user: str
    is_online: bool = False
    cpu_load: Optional[float] = None
    cpu_count: Optional[int] = None
    cpu_count_source: str = ""
    memory_usage: Optional[Dict[str, float]] = None
    gpu_info: Optional[Dict[str, Any]] = None
    gpu_source: str = ""
    last_update: Optional[float] = None
    error: Optional[str] = None
    meta: Optional[NodeMeta] = None


class NodeMonitor:
    def __init__(self, config_path: str = "/home/oop/tatbot/src/conf/nodes.yaml", 
                 meta_path: str = "/home/oop/tatbot/src/tui-opencode/nodes_meta.yaml"):
        self.config_path = Path(config_path)
        self.meta_path = Path(meta_path)
        self.nodes = []
        self.ssh_clients = {}
        self.metas = {}
        self.load_config()
        self.load_metadata()
        self.initial_probe()
        
    def load_config(self):
        with open(self.config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        self.nodes = [
            NodeInfo(
                name=node['name'],
                emoji=node['emoji'],
                ip=node.get('hostname', node.get('ip', node['name'])),
                user=node['user']
            )
            for node in config['nodes']
        ]
    
    def load_metadata(self):
        if not self.meta_path.exists():
            return
        
        try:
            with open(self.meta_path, 'r') as f:
                meta_config = yaml.safe_load(f)
            
            if 'nodes_meta' in meta_config:
                for name, data in meta_config['nodes_meta'].items():
                    self.metas[name] = NodeMeta(
                        cpu_cores=data.get('cpu_cores'),
                        gpu_total_mb=data.get('gpu_total_mb'),
                        gpu_count=data.get('gpu_count')
                    )
            
            for node in self.nodes:
                if node.name in self.metas:
                    node.meta = self.metas[node.name]
        except Exception as e:
            print(f"Failed to load metadata: {e}")
    
    def initial_probe(self):
        print("🔍 Probing nodes for initial capabilities...")
        for node in self.nodes:
            print(f"  📡 Checking {node.name}...")
            self.update_node(node)
        print("✅ Initial probe complete!")
    
    def check_node_online(self, node: NodeInfo) -> bool:
        try:
            result = subprocess.run(
                ['ping', '-c', '1', '-W', '1', node.ip],
                capture_output=True,
                text=True,
                timeout=2
            )
            return result.returncode == 0
        except (subprocess.TimeoutExpired, Exception):
            return False
    
    def run_ssh_command(self, node: NodeInfo, cmd: str, timeout: float = 4.0) -> Tuple[int, str, str]:
        try:
            pool = get_pool()
            return pool.exec(node.ip, node.user, cmd, timeout=timeout)
        except Exception as e:
            return 255, "", str(e)
    
    def get_cpu_info(self, node: NodeInfo) -> Tuple[Optional[float], Optional[int], str]:
        try:
            code, physical_cores, _ = self.run_ssh_command(
                node,
                "lscpu -p=CORE,SOCKET 2>/dev/null | egrep -v '^#' | "
                "awk -F, '{print $1 \"-\" $2}' | sort -u | wc -l"
            )
            
            cpu_count = None
            if code == 0 and physical_cores.strip().isdigit() and int(physical_cores.strip()) > 0:
                cpu_count = int(physical_cores.strip())
            else:
                code, logical_cores, _ = self.run_ssh_command(node, 'nproc --all')
                if code == 0 and logical_cores.strip().isdigit():
                    cpu_count = int(logical_cores.strip())
            
            code, uptime_line, _ = self.run_ssh_command(node, 'uptime')
            if code != 0:
                return None, cpu_count, "remote" if cpu_count else ""
                
            load_avg = uptime_line.split('load average:')[1].strip()
            load_1min = float(load_avg.split(',')[0])
            
            if cpu_count:
                cpu_load_percent = (load_1min / cpu_count) * 100
                return cpu_load_percent, cpu_count, "remote"
            else:
                return load_1min * 100, None, "remote"
            
        except Exception as e:
            node.error = f"CPU info error: {str(e)}"
            return None, None, ""
    
    def get_memory_info(self, node: NodeInfo) -> Optional[Dict[str, float]]:
        try:
            code, output, _ = self.run_ssh_command(node, 'free -b')
            if code != 0:
                return None
                
            lines = output.strip().split('\n')
            if len(lines) < 2:
                return None
                
            mem_line = lines[1].split()
            if len(mem_line) < 4:
                return None
                
            total = int(mem_line[1])
            used = int(mem_line[2])
            free = int(mem_line[3])
            
            return {
                'total_gb': total / (1024**3),
                'used_gb': used / (1024**3),
                'free_gb': free / (1024**3),
                'percent': (used / total) * 100 if total > 0 else 0
            }
        except Exception as e:
            node.error = f"Memory info error: {str(e)}"
            return None
    
    def get_gpu_info(self, node: NodeInfo) -> Tuple[Optional[Dict[str, Any]], str]:
        try:
            code, output, _ = self.run_ssh_command(
                node,
                'nvidia-smi --query-gpu=index,name,memory.total,memory.used,memory.free,utilization.gpu,temperature.gpu '
                '--format=csv,noheader,nounits 2>/dev/null'
            )
            
            if code != 0 or not output.strip():
                return None, ""
            
            gpu_info = {'gpus': [], 'devices': []}
            total_used = 0
            total_memory = 0
            
            for line in output.strip().split('\n'):
                if line and not line.startswith('No devices'):
                    parts = [p.strip() for p in line.split(',')]
                    if len(parts) >= 7:
                        try:
                            idx = int(parts[0])
                            mem_total = float(parts[2])
                            mem_used = float(parts[3])
                            
                            device = GpuDevice(
                                index=idx,
                                name=parts[1],
                                memory_total_mb=mem_total,
                                memory_used_mb=mem_used,
                                memory_free_mb=float(parts[4]),
                                utilization_percent=float(parts[5]),
                                temperature_c=float(parts[6]),
                                memory_percent=(mem_used / mem_total) * 100 if mem_total > 0 else 0
                            )
                            
                            gpu_info['devices'].append(device)
                            total_used += mem_used
                            total_memory += mem_total
                            
                            gpu_info['gpus'].append({
                                'index': idx,
                                'name': device.name,
                                'memory_total_mb': device.memory_total_mb,
                                'memory_used_mb': device.memory_used_mb,
                                'memory_free_mb': device.memory_free_mb,
                                'utilization_percent': device.utilization_percent,
                                'temperature_c': device.temperature_c,
                                'memory_percent': device.memory_percent
                            })
                        except (ValueError, IndexError):
                            continue
            
            if gpu_info['gpus']:
                gpu_info['total_used_mb'] = total_used
                gpu_info['total_memory_mb'] = total_memory
                gpu_info['gpu_count'] = len(gpu_info['gpus'])
                return gpu_info, "remote"
            
            return None, ""
            
        except Exception as e:
            print(f"GPU info error for {node.name}: {str(e)}")
            return None, ""
    
    def update_node(self, node: NodeInfo):
        node.is_online = self.check_node_online(node)
        
        if node.is_online:
            cpu_load, cpu_count, cpu_source = self.get_cpu_info(node)
            node.cpu_load = cpu_load
            
            if cpu_count:
                node.cpu_count = cpu_count
                node.cpu_count_source = cpu_source
            elif node.meta and node.meta.cpu_cores:
                node.cpu_count = node.meta.cpu_cores
                node.cpu_count_source = "meta"
            
            node.memory_usage = self.get_memory_info(node)
            
            gpu_info, gpu_source = self.get_gpu_info(node)
            node.gpu_info = gpu_info
            node.gpu_source = gpu_source
            
            if not gpu_info and node.meta and node.meta.gpu_total_mb:
                node.gpu_info = {
                    'gpus': [],
                    'total_memory_mb': node.meta.gpu_total_mb,
                    'gpu_count': node.meta.gpu_count or 0,
                    'meta_only': True
                }
                node.gpu_source = "meta"
            
            node.last_update = time.time()
            node.error = None
        else:
            node.cpu_load = None
            if node.meta and node.meta.cpu_cores:
                node.cpu_count = node.meta.cpu_cores
                node.cpu_count_source = "meta"
            else:
                node.cpu_count = None
                node.cpu_count_source = ""
            
            node.memory_usage = None
            
            if node.meta and node.meta.gpu_total_mb:
                node.gpu_info = {
                    'gpus': [],
                    'total_memory_mb': node.meta.gpu_total_mb,
                    'gpu_count': node.meta.gpu_count or 0,
                    'meta_only': True
                }
                node.gpu_source = "meta"
            else:
                node.gpu_info = None
                node.gpu_source = ""
            
            node.error = "Node offline"
    
    def update_all_nodes(self):
        with concurrent.futures.ThreadPoolExecutor(max_workers=min(8, max(2, len(self.nodes)))) as executor:
            futures = [executor.submit(self.update_node, node) for node in self.nodes]
            concurrent.futures.wait(futures)
    
    def close(self):
        pool = get_pool()
        pool.close_all()