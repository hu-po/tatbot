from __future__ import annotations

import subprocess
import os
from dataclasses import dataclass
from typing import Optional, List

# Support both package and direct-script imports
try:
    from .ssh_pool import get_pool  # type: ignore
except Exception:
    import importlib.util
    import sys
    _BASE_DIR_RS = os.path.dirname(__file__)
    _SSH_PATH = os.path.join(_BASE_DIR_RS, "ssh_pool.py")
    _spec = importlib.util.spec_from_file_location("ssh_pool", _SSH_PATH)
    if _spec is None or _spec.loader is None:
        raise ImportError("Cannot import ssh_pool")
    _mod = importlib.util.module_from_spec(_spec)
    sys.modules["ssh_pool"] = _mod
    _spec.loader.exec_module(_mod)
    get_pool = getattr(_mod, "get_pool")

# Optional logging path via environment
_LOG_PATH = os.environ.get("TATBOT_TUI_LOG")


def _log(msg: str) -> None:
    path = _LOG_PATH
    if not path:
        return
    try:
        with open(path, "a", encoding="utf-8") as f:
            from time import strftime, localtime
            ts = strftime('%Y-%m-%d %H:%M:%S', localtime())
            f.write(f"[{ts}] {msg}\n")
    except Exception:
        pass


@dataclass
class CpuStats:
    load_1: float
    load_5: float
    load_15: float
    physical_cores: Optional[int] = None
    logical_cores: Optional[int] = None
    percent_1m: Optional[float] = None


@dataclass
class GpuDeviceStats:
    index: int
    name: Optional[str]
    util_percent: Optional[float]
    temp_c: Optional[float]
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
    memory: Optional["MemoryStats"]


@dataclass
class MemoryStats:
    total_bytes: int
    used_bytes: int
    free_bytes: int
    percent: float


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
    # Prefer persistent Paramiko connection for performance
    try:
        pool = get_pool()
        code, out, err = pool.exec(host, user, remote_cmd, timeout=timeout)
        # Emulate CompletedProcess
        cp = subprocess.CompletedProcess(args=["paramiko", host, remote_cmd], returncode=code, stdout=out, stderr=err)
        if code != 0:
            _log(f"ssh_exec_failed host={host} user={user} code={code} cmd={remote_cmd!r} err={err.strip()[:200]}")
        return cp
    except Exception:
        ssh_cmd = _ssh_base_cmd(host, user) + [remote_cmd]
        cp = run_local(ssh_cmd, timeout=timeout)
        if cp.returncode != 0:
            _log(f"ssh_subprocess_failed host={host} user={user} code={cp.returncode} cmd={remote_cmd!r} err={cp.stderr.strip()[:200]}")
        return cp


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

    # Try to fetch physical and logical core counts
    physical = None
    logical = None
    try:
        # Count unique (core,socket) pairs for physical cores
        lscpu = run_ssh(
            host,
            user,
            "lscpu -p=CORE,SOCKET 2>/dev/null | egrep -v '^#' | awk -F, '{print $1 "-" $2}' | sort -u | wc -l",
        )
        if lscpu.returncode == 0 and lscpu.stdout.strip():
            physical = int(lscpu.stdout.strip())
        # Logical CPUs
        nproc = run_ssh(host, user, "nproc --all")
        if nproc.returncode == 0 and nproc.stdout.strip():
            logical = int(nproc.stdout.strip().splitlines()[0])
        if logical is None:
            # Last resort: count processors entries
            cpuinfo = run_ssh(host, user, "grep -c ^processor /proc/cpuinfo || true")
            if cpuinfo.returncode == 0 and cpuinfo.stdout.strip():
                logical = int(cpuinfo.stdout.strip().splitlines()[0])
    except Exception:
        pass

    percent = None
    try:
        if logical and logical > 0:
            percent = (load_1 / float(logical)) * 100.0
    except Exception:
        percent = None

    return CpuStats(
        load_1=load_1,
        load_5=load_5,
        load_15=load_15,
        physical_cores=physical,
        logical_cores=logical,
        percent_1m=percent,
    )


def get_remote_gpu_stats(host: str, user: str) -> Optional[GpuStats]:
    # Use nvidia-smi if available; collect per-GPU stats in MiB
    cmd = (
        "nvidia-smi --query-gpu=index,name,utilization.gpu,temperature.gpu,memory.used,memory.total "
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
                if len(parts) >= 6:
                    try:
                        idx = int(parts[0])
                        name = parts[1]
                        util = float(parts[2]) if parts[2] else None
                        temp = float(parts[3]) if parts[3] else None
                        used = int(parts[4])
                        total = int(parts[5])
                        devices.append(GpuDeviceStats(index=idx, name=name, util_percent=util, temp_c=temp, mem_used_mb=used, mem_total_mb=total))
                        used_sum += used
                        total_sum += total
                    except ValueError:
                        continue
            if devices:
                return GpuStats(devices=devices, mem_used_mb=used_sum, mem_total_mb=total_sum, gpu_count=len(devices))
    except Exception:
        pass
    return None


def get_remote_memory_stats(host: str, user: str) -> Optional[MemoryStats]:
    # Use free -b for bytes
    cmd = "free -b | awk '/^Mem:/ {print $2,\" \",$3,\" \",$4}'"
    try:
        result = run_ssh(host, user, cmd)
        if result.returncode == 0 and result.stdout.strip():
            parts = result.stdout.strip().split()
            if len(parts) >= 3:
                total = int(parts[0])
                used = int(parts[1])
                free = int(parts[2])
                percent = (used / total) * 100.0 if total > 0 else 0.0
                return MemoryStats(total_bytes=total, used_bytes=used, free_bytes=free, percent=percent)
    except Exception:
        pass
    return None


def get_node_stats(host: str, user: str) -> NodeStats:
    # Attempt remote probes regardless of ICMP reachability (some networks block ping)
    cpu = get_remote_cpu_stats(host, user)
    gpu = get_remote_gpu_stats(host, user)
    mem = get_remote_memory_stats(host, user)
    online = any(v is not None for v in (cpu, gpu, mem))
    if not online:
        online = check_ssh_online(host, user) or ping_host(host)
    return NodeStats(online=online, cpu=cpu, gpu=gpu, memory=mem)
