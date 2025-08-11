"""Shared constants for tatbot."""

from pathlib import Path

# NFS mount point - canonical path for shared storage across all nodes
NFS_DIR = Path("/nfs/tatbot")

# Common subdirectories
NFS_RECORDINGS_DIR = NFS_DIR / "recordings"
NFS_DESIGNS_DIR = NFS_DIR / "designs"
NFS_LOGS_DIR = NFS_DIR / "mcp-logs"