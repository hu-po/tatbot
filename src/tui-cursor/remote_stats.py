from __future__ import annotations

import subprocess
from dataclasses import dataclass
from typing import Optional


@dataclass
class CpuStats:
    load_1: float
    load_5: float
    load_15: float
    cores: Optional[int] = None


@dataclass
class GpuStats:
    mem_used_mb: int
    mem_total_mb: int
    gpu_count: int = 1


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


def run_ssh(host: str, user: str, remote_cmd: str, timeout: float = 3.5) -> subprocess.CompletedProcess:
    ssh_cmd = [
        "ssh",
        "-o", "BatchMode=yes",
        "-o", "StrictHostKeyChecking=accept-new",
        "-o", "UserKnownHostsFile=/dev/null",
        "-o", "ConnectTimeout=2",
        f"{user}@{host}",
        remote_cmd,
    ]
    return run_local(ssh_cmd, timeout=timeout)


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

    # Try to fetch core count
    cores = None
    try:
        nproc = run_ssh(host, user, "nproc --all")
        if nproc.returncode == 0:
            cores = int(nproc.stdout.strip().splitlines()[0])
        else:
            # Fallback: count processors in /proc/cpuinfo
            cpuinfo = run_ssh(host, user, "grep -c ^processor /proc/cpuinfo || true")
            if cpuinfo.returncode == 0 and cpuinfo.stdout.strip():
                cores = int(cpuinfo.stdout.strip().splitlines()[0])
    except Exception:
        pass

    return CpuStats(load_1=load_1, load_5=load_5, load_15=load_15, cores=cores)


def get_remote_gpu_stats(host: str, user: str) -> Optional[GpuStats]:
    # Use nvidia-smi if available; otherwise None. Sum all GPUs if multiple.
    cmd = (
        "nvidia-smi --query-gpu=memory.used,memory.total "
        "--format=csv,noheader,nounits 2>/dev/null"
    )
    try:
        result = run_ssh(host, user, cmd)
        if result.returncode == 0 and result.stdout.strip():
            used_sum = 0
            total_sum = 0
            count = 0
            for line in result.stdout.strip().splitlines():
                parts = [p.strip() for p in line.split(",")]
                if len(parts) >= 2:
                    try:
                        used_sum += int(parts[0])
                        total_sum += int(parts[1])
                        count += 1
                    except ValueError:
                        continue
            if total_sum > 0:
                return GpuStats(mem_used_mb=used_sum, mem_total_mb=total_sum, gpu_count=max(1, count))
    except Exception:
        pass
    return None


def get_node_stats(host: str, user: str) -> NodeStats:
    online = ping_host(host)
    if not online:
        return NodeStats(online=False, cpu=None, gpu=None)
    cpu = get_remote_cpu_stats(host, user)
    gpu = get_remote_gpu_stats(host, user)
    return NodeStats(online=True, cpu=cpu, gpu=gpu)
