from __future__ import annotations

import subprocess
import os
from dataclasses import dataclass
from typing import Optional, List


@dataclass
class CpuStats:
    load_1: float
    load_5: float
    load_15: float
    cores: Optional[int] = None


@dataclass
class GpuDeviceStats:
    index: int
    mem_used_mb: int
    mem_total_mb: int


@dataclass
class GpuStats:
    devices: List[GpuDeviceStats]
    mem_used_mb: int
    mem_total_mb: int
    gpu_count: int


@dataclass
class NodeStats:
    online: bool
    cpu: Optional[CpuStats]
    gpu: Optional[GpuStats]


def run_local(command: list[str], timeout: float = 3.0) -> subprocess.CompletedProcess:
    return subprocess.run(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=timeout,
        text=True,
    )


def _ssh_base_cmd(host: str, user: str) -> list[str]:
    ssh_cmd = [
        "ssh",
        "-o", "BatchMode=yes",
        "-o", "StrictHostKeyChecking=accept-new",
        "-o", "UserKnownHostsFile=/dev/null",
        "-o", "ConnectTimeout=3",
    ]
    identity = os.environ.get("TATBOT_TUI_SSH_KEY")
    if identity:
        ssh_cmd += ["-i", identity]
    port = os.environ.get("TATBOT_TUI_SSH_PORT")
    if port:
        ssh_cmd += ["-p", port]
    ssh_cmd += [f"{user}@{host}"]
    return ssh_cmd


def run_ssh(host: str, user: str, remote_cmd: str, timeout: float = 4.0) -> subprocess.CompletedProcess:
    ssh_cmd = _ssh_base_cmd(host, user) + [remote_cmd]
    return run_local(ssh_cmd, timeout=timeout)


def check_ssh_online(host: str, user: str) -> bool:
    try:
        result = run_ssh(host, user, "true", timeout=2.5)
        return result.returncode == 0
    except Exception:
        return False


def ping_host(host: str, count: int = 1, timeout: float = 1.0) -> bool:
    try:
        # Linux ping: -c count, -W timeout(seconds)
        result = run_local(["ping", "-c", str(count), "-W", str(int(timeout)), host], timeout=timeout + 0.5)
        return result.returncode == 0
    except Exception:
        return False


def get_remote_cpu_stats(host: str, user: str) -> Optional[CpuStats]:
    # Read load averages
    load_cmd = "cat /proc/loadavg | awk '{print $1,\"\",$2,\"\",$3}'"
    load_1 = load_5 = load_15 = None
    try:
        result = run_ssh(host, user, load_cmd)
        if result.returncode == 0:
            parts = result.stdout.strip().split()
            if len(parts) >= 3:
                load_1 = float(parts[0])
                load_5 = float(parts[1])
                load_15 = float(parts[2])
    except Exception:
        pass

    if load_1 is None or load_5 is None or load_15 is None:
        return None

    # Try to fetch physical core count via lscpu; fall back to nproc
    cores = None
    try:
        # Count unique (core,socket) pairs for physical cores
        lscpu = run_ssh(
            host,
            user,
            "lscpu -p=CORE,SOCKET 2>/dev/null | egrep -v '^#' | awk -F, '{print $1 "-" $2}' | sort -u | wc -l",
        )
        if lscpu.returncode == 0 and lscpu.stdout.strip():
            cores = int(lscpu.stdout.strip())
        if not cores:
            # Fallback: nproc (logical CPUs)
            nproc = run_ssh(host, user, "nproc --all")
            if nproc.returncode == 0:
                cores = int(nproc.stdout.strip().splitlines()[0])
        if not cores:
            # Last resort: count processors entries
            cpuinfo = run_ssh(host, user, "grep -c ^processor /proc/cpuinfo || true")
            if cpuinfo.returncode == 0 and cpuinfo.stdout.strip():
                cores = int(cpuinfo.stdout.strip().splitlines()[0])
    except Exception:
        pass

    return CpuStats(load_1=load_1, load_5=load_5, load_15=load_15, cores=cores)


def get_remote_gpu_stats(host: str, user: str) -> Optional[GpuStats]:
    # Use nvidia-smi if available; collect per-GPU used/total in MiB
    cmd = (
        "nvidia-smi --query-gpu=index,memory.used,memory.total "
        "--format=csv,noheader,nounits 2>/dev/null"
    )
    try:
        result = run_ssh(host, user, cmd)
        if result.returncode == 0 and result.stdout.strip():
            devices: List[GpuDeviceStats] = []
            used_sum = 0
            total_sum = 0
            for line in result.stdout.strip().splitlines():
                parts = [p.strip() for p in line.split(",")]
                if len(parts) >= 3:
                    try:
                        idx = int(parts[0])
                        used = int(parts[1])
                        total = int(parts[2])
                        devices.append(GpuDeviceStats(index=idx, mem_used_mb=used, mem_total_mb=total))
                        used_sum += used
                        total_sum += total
                    except ValueError:
                        continue
            if devices:
                return GpuStats(devices=devices, mem_used_mb=used_sum, mem_total_mb=total_sum, gpu_count=len(devices))
    except Exception:
        pass
    return None


def get_node_stats(host: str, user: str) -> NodeStats:
    # Attempt remote probes regardless of ICMP reachability (some networks block ping)
    cpu = get_remote_cpu_stats(host, user)
    gpu = get_remote_gpu_stats(host, user)
    online = (cpu is not None) or (gpu is not None)
    if not online:
        online = check_ssh_online(host, user) or ping_host(host)
    return NodeStats(online=online, cpu=cpu, gpu=gpu)
