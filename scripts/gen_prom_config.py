#!/usr/bin/env python3
"""Render Prometheus config from inventory.yml.

Outputs to config/monitoring/prometheus/prometheus.yml
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict, List

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
INVENTORY = REPO_ROOT / "config/monitoring/inventory.yml"
OUT = REPO_ROOT / "config/monitoring/prometheus/prometheus.yml"


def _targets_from_inventory(inv: Dict[str, Any]) -> Dict[str, List[str]]:
    nodes: Dict[str, Any] = inv.get("nodes", {})

    node_targets: List[str] = []
    jetson_targets: List[str] = []
    nvidia_targets: List[str] = []
    intel_targets: List[str] = []
    rpi_targets: List[str] = []

    for name, cfg in nodes.items():
        # Node exporter
        if "addr" in cfg:
            node_targets.append(cfg["addr"])

        # Jetson exporter (on 9100)
        if "jetson" in cfg and isinstance(cfg["jetson"], dict) and "addr" in cfg["jetson"]:
            jetson_targets.append(cfg["jetson"]["addr"])

        # NVIDIA DCGM
        gpu = cfg.get("gpu", {})
        if isinstance(gpu, dict) and "nvidia_dcgm" in gpu:
            dcgm = gpu["nvidia_dcgm"]
            if isinstance(dcgm, dict) and "addr" in dcgm:
                nvidia_targets.append(dcgm["addr"])

        # Intel GPU exporter
        if isinstance(gpu, dict) and "intel" in gpu:
            intel = gpu["intel"]
            if isinstance(intel, dict) and "addr" in intel:
                intel_targets.append(intel["addr"])

        # rpi exporter
        if "rpi" in cfg and isinstance(cfg["rpi"], dict) and "addr" in cfg["rpi"]:
            rpi_targets.append(cfg["rpi"]["addr"])

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
    # Assemble YAML with stable ordering
    doc: Dict[str, Any] = {
        "global": {
            "scrape_interval": si,
            "evaluation_interval": si,
        },
        "scrape_configs": [],
    }

    def add_job(name: str, targets: List[str]) -> None:
        if not targets:
            return
        doc["scrape_configs"].append({
            "job_name": name,
            "static_configs": [{"targets": targets}],
        })

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

