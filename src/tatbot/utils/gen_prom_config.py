#!/usr/bin/env python3
"""Render Prometheus config from inventory.yml.

Outputs to config/monitoring/prometheus/prometheus.yml
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

import yaml


def _repo_root() -> Path:
    """Best-effort detection of the repository root.

    Walk upwards until a directory containing both `config/` and `src/` is found.
    Fallback to two levels up from this file.
    """
    here = Path(__file__).resolve()
    for parent in [here] + list(here.parents):
        try:
            if (parent / "config").is_dir() and (parent / "src").is_dir():
                return parent
        except Exception:
            continue
    # Fallbacks: support both old (utils/) and new (src/tatbot/utils/) locations
    # Old: parents[1] from utils/ -> repo root
    # New: parents[3] from src/tatbot/utils/ -> repo root
    return here.parents[3] if len(here.parents) >= 4 else here.parents[1]


REPO_ROOT = _repo_root()
INVENTORY = REPO_ROOT / "config/monitoring/inventory.yml"
OUT = REPO_ROOT / "config/monitoring/prometheus/prometheus.yml"
NODES_YAML = REPO_ROOT / "src/conf/nodes.yaml"


def _load_node_emojis() -> Dict[str, str]:
    """Load node -> emoji mapping from src/conf/nodes.yaml.

    Returns an empty dict if file missing or malformed to avoid hard failures.
    """
    try:
        if not NODES_YAML.exists():
            return {}
        with NODES_YAML.open("r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        mapping: Dict[str, str] = {}
        for entry in data.get("nodes", []) or []:
            name = entry.get("name")
            emoji = entry.get("emoji")
            if isinstance(name, str) and isinstance(emoji, str) and emoji:
                mapping[name] = emoji
        return mapping
    except Exception:
        return {}


def _targets_from_inventory(inv: Dict[str, Any]) -> Dict[str, List[Tuple[str, str]]]:
    nodes: Dict[str, Any] = inv.get("nodes", {})

    node_targets: List[Tuple[str, str]] = []  # (name, addr)
    jetson_targets: List[Tuple[str, str]] = []
    nvidia_targets: List[Tuple[str, str]] = []
    intel_targets: List[Tuple[str, str]] = []
    rpi_targets: List[Tuple[str, str]] = []

    for name, cfg in nodes.items():
        # Node exporter
        if "addr" in cfg:
            node_targets.append((name, cfg["addr"]))

        # Jetson exporter (on 9100)
        if "jetson" in cfg and isinstance(cfg["jetson"], dict) and "addr" in cfg["jetson"]:
            jetson_targets.append((name, cfg["jetson"]["addr"]))

        # NVIDIA DCGM
        gpu = cfg.get("gpu", {})
        if isinstance(gpu, dict) and "nvidia_dcgm" in gpu:
            dcgm = gpu["nvidia_dcgm"]
            if isinstance(dcgm, dict) and "addr" in dcgm:
                nvidia_targets.append((name, dcgm["addr"]))

        # Intel GPU exporter
        if isinstance(gpu, dict) and "intel" in gpu:
            intel = gpu["intel"]
            if isinstance(intel, dict) and "addr" in intel:
                intel_targets.append((name, intel["addr"]))

        # rpi exporter
        if "rpi" in cfg and isinstance(cfg["rpi"], dict) and "addr" in cfg["rpi"]:
            rpi_targets.append((name, cfg["rpi"]["addr"]))

    return {
        "nodes": sorted(set(node_targets)),
        "jetson": sorted(set(jetson_targets)),
        "nvidia_dgpu": sorted(set(nvidia_targets)),
        "intel_gpu": sorted(set(intel_targets)),
        "rpi_soc": sorted(set(rpi_targets)),
    }


def render(inv: Dict[str, Any]) -> str:
    si = inv.get("scrape_interval", "15s")
    t = _targets_from_inventory(inv)
    emoji_map = _load_node_emojis()
    # Assemble YAML with stable ordering
    doc: Dict[str, Any] = {
        "global": {
            "scrape_interval": si,
            "evaluation_interval": si,
        },
        "scrape_configs": [],
    }

    def add_job(name: str, targets: List[Tuple[str, str]]) -> None:
        if not targets:
            return
        # Each target gets its own static_config so we can attach a per-target 'node' label
        static_configs: List[Dict[str, Any]] = []
        for node_name, addr in targets:
            emoji = emoji_map.get(node_name, "")
            friendly = f"{emoji} {node_name}".strip() if emoji else node_name
            static_configs.append({
                "targets": [addr],
                # 'node' is a friendly label used for the legend (emoji + name)
                # 'name' preserves the plain node name
                "labels": {"node": friendly, "name": node_name},
            })

        job: Dict[str, Any] = {
            "job_name": name,
            "static_configs": static_configs,
            # Relabel 'instance' to the friendly node name for nicer legends
            "relabel_configs": [
                {"source_labels": ["node"], "target_label": "instance"},
                {"source_labels": ["__address__"], "target_label": "addr"},
            ],
        }
        doc["scrape_configs"].append(job)

    add_job("nodes", t["nodes"]) 
    add_job("jetson", t["jetson"]) 
    add_job("nvidia_dgpu", t["nvidia_dgpu"]) 
    add_job("intel_gpu", t["intel_gpu"]) 
    add_job("rpi_soc", t["rpi_soc"]) 

    return yaml.safe_dump(doc, sort_keys=False)


def main() -> int:
    if not INVENTORY.exists():
        print(f"Inventory not found: {INVENTORY}", file=sys.stderr)
        return 1

    with INVENTORY.open("r", encoding="utf-8") as f:
        inv = yaml.safe_load(f)

    out_yaml = render(inv)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(out_yaml, encoding="utf-8")
    print(f"Wrote {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

