from __future__ import annotations

import os
from dataclasses import dataclass
from typing import List


@dataclass
class Node:
    name: str
    emoji: str
    ip: str
    user: str


def parse_nodes_config(config_path: str) -> List[Node]:
    """
    Parse a minimal YAML-like file at `config_path` that looks like:

    nodes:
      - name: ook
        emoji: 🦧
        ip: 192.168.1.90
        user: ook
      - name: ...

    This parser is intentionally minimal and tailored to the structure above to avoid
    adding external YAML dependencies. It ignores unknown keys and blank lines.
    """
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config not found: {config_path}")

    nodes: List[Node] = []
    current: dict[str, str] | None = None
    in_nodes_section = False

    with open(config_path, "r", encoding="utf-8") as f:
        for raw_line in f:
            line = raw_line.rstrip("\n")
            stripped = line.strip()

            if not stripped:
                # Blank line: if we were collecting a node and have all fields, finalize it
                if current and all(k in current for k in ("name", "emoji", "ip", "user")):
                    nodes.append(Node(**current))
                    current = None
                continue

            if stripped.startswith("#"):
                continue

            if stripped == "nodes:" or stripped.startswith("nodes:"):
                in_nodes_section = True
                continue

            if not in_nodes_section:
                continue

            # Start of a new node list item
            if stripped.startswith("- name:"):
                # Finalize previous node if complete
                if current and all(k in current for k in ("name", "emoji", "ip", "user")):
                    nodes.append(Node(**current))
                # Start new
                name_value = stripped.split(":", 1)[1].strip()
                current = {"name": name_value}
                continue

            # If we are currently building a node, capture expected keys
            if current is not None and \
               (stripped.startswith("emoji:") or stripped.startswith("ip:") or stripped.startswith("user:")):
                key, value = stripped.split(":", 1)
                current[key.strip()] = value.strip()
                continue

        # End of file: finalize last node
        if current and all(k in current for k in ("name", "emoji", "ip", "user")):
            nodes.append(Node(**current))

    return nodes
