from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Dict, Optional


@dataclass
class NodeMeta:
    cpu_cores: Optional[int] = None
    gpu_total_mb: Optional[int] = None  # Sum across all GPUs
    gpu_count: Optional[int] = None


def parse_nodes_meta(meta_path: str) -> Dict[str, NodeMeta]:
    """
    Parse a minimal YAML-like file with structure:

    nodes_meta:
      <node_name>:
        cpu_cores: 16
        gpu_total_mb: 24576
        gpu_count: 1

    Returns a mapping from node name to NodeMeta.
    """
    if not os.path.exists(meta_path):
        return {}

    metas: Dict[str, NodeMeta] = {}
    in_section = False
    current_name: Optional[str] = None
    current: dict[str, str] = {}

    def finalize():
        nonlocal current_name, current
        if current_name is None:
            return
        meta = NodeMeta(
            cpu_cores=int(current["cpu_cores"]) if "cpu_cores" in current else None,
            gpu_total_mb=int(current["gpu_total_mb"]) if "gpu_total_mb" in current else None,
            gpu_count=int(current["gpu_count"]) if "gpu_count" in current else None,
        )
        metas[current_name] = meta
        current_name = None
        current = {}

    with open(meta_path, "r", encoding="utf-8") as f:
        for raw in f:
            line = raw.rstrip("\n")
            s = line.strip()
            if not s or s.startswith("#"):
                continue
            if s == "nodes_meta:" or s.startswith("nodes_meta:"):
                in_section = True
                continue
            if not in_section:
                continue
            if not line.startswith(" ") and s.endswith(":"):
                # New top-level outside section; stop
                break
            # node key e.g., "  ook:"
            if s.endswith(":") and not s.startswith("-"):
                finalize()
                current_name = s[:-1].strip()
                current = {}
                continue
            # key-value lines
            if ":" in s and current_name is not None:
                key, value = s.split(":", 1)
                current[key.strip()] = value.strip()
        finalize()

    return metas
