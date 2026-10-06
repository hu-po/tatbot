"""Generate the small, reviewable TattooProgram class matrix."""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
from typing import Any

from tatbot_sim.human_rep.contracts import canonical_bytes, canonical_digest, validate_contract
from tatbot_sim.human_rep.tattoo_program import tattoo_program_to_svg
from tatbot_sim.repo import repo_root

CREATED_UTC = "2026-09-04T13:00:00Z"


def _element(identifier: str, kind: str, *, points=None, controls=None, closed=False, fill=False, width=0.0008, deposition=0.7):
    result = {
        "id": identifier,
        "kind": kind,
        "closed": closed,
        "fill": fill,
        "width_m": width,
        "deposition": deposition,
    }
    if points is not None:
        result["points_m"] = points
    if controls is not None:
        result["control_points_m"] = controls
    return result


def _program(name: str, inks: list[dict], layers: list[dict], masks: list[dict]) -> tuple[dict[str, Any], bytes]:
    source = hashlib.sha256(f"tatbot-program-example:{name}:1".encode()).hexdigest()
    document = {
        "schema": "tatbot.tattoo-program/1",
        "content_sha256": "0" * 64,
        "canvas_m": {"width": 0.06, "height": 0.06},
        "inks": inks,
        "layers": layers,
        "negative_space_masks": masks,
        "semantic_intent": f"Synthetic {name} contract fixture; no person data.",
        "preview_sha256": "0" * 64,
        "provenance": {
            "producer": "tatbot-human-representation-examples",
            "version": "1",
            "created_utc": CREATED_UTC,
            "source_sha256": source,
            "seed": 9042026,
        },
    }
    preview = (tattoo_program_to_svg({
        **document,
        "preview_sha256": source,
        "content_sha256": canonical_digest({**document, "preview_sha256": source}),
    }) + "\n").encode()
    document["preview_sha256"] = hashlib.sha256(preview).hexdigest()
    document["content_sha256"] = canonical_digest(document)
    validate_contract(document, expected_schema="tatbot.tattoo-program/1")
    return document, preview


def examples() -> dict[str, tuple[dict[str, Any], bytes]]:
    black = {"id": "black", "color_srgb": [0, 0, 0]}
    red = {"id": "red", "color_srgb": [0.75, 0.05, 0.04]}
    blue = {"id": "blue", "color_srgb": [0.03, 0.18, 0.8]}
    return {
        "linework": _program("linework", [black], [{
            "id": "lines", "ink_id": "black", "elements": [
                _element("line", "path", points=[[0.008, 0.012], [0.022, 0.045], [0.052, 0.018]]),
                _element("curve", "cubic_bezier", controls=[[0.01, 0.05], [0.02, 0.02], [0.04, 0.02], [0.05, 0.05]]),
            ],
        }], []),
        "blackwork": _program("blackwork", [black], [{
            "id": "fill", "ink_id": "black", "elements": [
                _element("diamond", "region", points=[[0.03, 0.006], [0.054, 0.03], [0.03, 0.054], [0.006, 0.03]], closed=True, fill=True, deposition=1.0),
            ],
        }], [{"id": "center-cutout", "points_m": [[0.03, 0.02], [0.04, 0.03], [0.03, 0.04], [0.02, 0.03]]}]),
        "stipple": _program("stipple", [black], [{
            "id": "dots", "ink_id": "black", "elements": [
                _element("field", "stipple", points=[[0.015, 0.015], [0.03, 0.012], [0.045, 0.015], [0.02, 0.03], [0.04, 0.03], [0.03, 0.045]], width=0.0012, deposition=0.45),
            ],
        }], []),
        "color": _program("multi-color", [red, blue], [
            {"id": "red-lines", "ink_id": "red", "elements": [_element("red-wave", "path", points=[[0.006, 0.02], [0.02, 0.04], [0.035, 0.02], [0.052, 0.04]])]},
            {"id": "blue-region", "ink_id": "blue", "elements": [_element("blue-triangle", "region", points=[[0.018, 0.01], [0.042, 0.01], [0.03, 0.032]], closed=True, fill=True, deposition=0.9)]},
        ], []),
    }


def write_examples(root: Path, *, check: bool) -> None:
    for name, (program, preview) in examples().items():
        destination = root / name
        outputs = {
            destination / "program.json": canonical_bytes(program) + b"\n",
            destination / "preview.svg": preview,
        }
        for path, expected in outputs.items():
            if check:
                if not path.is_file() or path.read_bytes() != expected:
                    raise ValueError(f"generated example differs: {path}")
            else:
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(expected)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=repo_root() / "config/human-representation/examples")
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    write_examples(args.output, check=args.check)


if __name__ == "__main__":
    main()
