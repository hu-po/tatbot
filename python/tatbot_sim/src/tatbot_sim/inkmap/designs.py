"""Immutable design artifacts for simulator scenarios.

Normal suites consume the shared reviewed artwork collection. The Archimedean
spiral is an explicit calibration control; random curves are test fixtures only.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from tatbot_sim.inkmap.svg_strokes import compile_svg_strokes

SPIRAL_ID = "spiral-v1"
SPIRAL_RADIUS_MM = 15.0
SPIRAL_TURNS = 3.0
SPIRAL_STEP_MM = 0.25
PROCEDURAL_MODEL = "tatbot-sim-procedural-v1"


@dataclass(frozen=True)
class DesignArtifact:
    """Exact vector bytes plus enough provenance to replay their creation."""

    id: str
    name: str
    svg: str
    size_mm: tuple[float, float]
    source: dict
    artwork: dict | None = None

    @property
    def sha256(self) -> str:
        return hashlib.sha256(self.svg.encode()).hexdigest()

    def embedded(self, size_mm: tuple[float, float] | None = None) -> dict:
        """Freeze public simulation SVG input once at its explicit import boundary."""
        from tatbot_sim.inkmap.artwork import make_artwork_record
        if self.artwork is not None:
            from copy import deepcopy

            from tatbot_contracts.artwork import canvas_m, validate_path_artwork
            record = validate_path_artwork(self.artwork)
            if size_mm is not None and not np.allclose(size_mm, np.asarray(canvas_m(record)) * 1000, rtol=0, atol=1e-6):
                raise ValueError("physical resizing requires DBV3 regeneration")
            return deepcopy(record)
        if self.source.get("collection"):
            from tatbot_sim.inkmap.collection import artwork_record, collection_entries
            entry = next(e for e in collection_entries() if e["id"] == self.id)
            return artwork_record(entry, tuple(size_mm or self.size_mm))
        if self.source.get("kind") == "generated":
            source = {"kind": "generated", "identifier": self.id, "license": None, "attribution": None,
                      "generation": {"prompt": self.source.get("prompt", self.name),
                                     "model": self.source["model"], "model_revision": self.source.get("model_revision"),
                                     "seed": self.source.get("seed"), "tracing": self.source.get("trace", {}).get("algorithm")}}
        else:
            source = {"kind": "fixture", "identifier": self.id, "license": None, "attribution": None, "generation": None}
        return make_artwork_record(name=self.name, original_svg=self.svg, source=source, conversion={
            "canvas_m": [v / 1000 for v in (size_mm or self.size_mm)],
            "semantic_intent": self.name, "width_m": .0003, "deposition": 1, "chord_error_m": .000005,
            **({"strokes": "centerline"} if self.id == SPIRAL_ID else {}),
        })


def _fmt(value: float) -> str:
    text = f"{value:.6f}".rstrip("0").rstrip(".")
    return "0" if text in ("", "-0") else text


def spiral_polyline(
    radius_mm: float = SPIRAL_RADIUS_MM,
    turns: float = SPIRAL_TURNS,
    step_mm: float = SPIRAL_STEP_MM,
) -> np.ndarray:
    """The calibration Archimedean spiral, sampled uniformly in arc length."""
    if not (radius_mm > 0 and turns > 0 and step_mm > 0):
        raise ValueError("spiral radius, turns and step must be positive")
    total_angle = 2.0 * math.pi * turns
    scale = radius_mm / total_angle
    length = 0.5 * scale * (
        total_angle * math.sqrt(1.0 + total_angle * total_angle) + math.asinh(total_angle)
    )
    count = max(2, int(math.ceil(length / step_mm)) + 1)
    distance = np.linspace(0.0, length, count)
    angle = total_angle * distance / length
    for _ in range(6):
        root = np.sqrt(1.0 + angle * angle)
        integrated = 0.5 * scale * (angle * root + np.arcsinh(angle))
        angle -= (integrated - distance) / (scale * root)
        angle = np.clip(angle, 0.0, total_angle)
    radius = scale * angle
    return np.stack([radius * np.cos(angle), radius * np.sin(angle)], axis=1)


def spiral_svg() -> str:
    points = spiral_polyline()
    encoded = " ".join(f"{_fmt(x)},{_fmt(-y)}" for x, y in points)
    radius = _fmt(SPIRAL_RADIUS_MM)
    diameter = _fmt(2.0 * SPIRAL_RADIUS_MM)
    return (
        '<svg xmlns="http://www.w3.org/2000/svg" '
        f'viewBox="-{radius} -{radius} {diameter} {diameter}" fill="none" '
        'stroke="#111" stroke-width="0.5">'
        f'<polyline points="{encoded}"/></svg>'
    )


def spiral_artifact() -> DesignArtifact:
    svg = spiral_svg()
    artifact = DesignArtifact(
        id=SPIRAL_ID,
        name="Spiral",
        svg=svg,
        size_mm=(2.0 * SPIRAL_RADIUS_MM, 2.0 * SPIRAL_RADIUS_MM),
        source={
            "kind": "spiral",
            "generator": "tatbot draw",
            "radius_mm": SPIRAL_RADIUS_MM,
            "turns": SPIRAL_TURNS,
            "step_mm": SPIRAL_STEP_MM,
        },
    )
    compile_svg_strokes(artifact.svg, artifact.size_mm)
    return artifact


def procedural_artifact(seed: int, size_mm: tuple[float, float] = (50.0, 50.0)) -> DesignArtifact:
    """Generate bounded, changing vector linework without a checked-in motif list.

    Legacy fuzz and replay fixture only. Normal simulation selects the shared
    artwork collection or complete materializations; this is not a CLI provider.
    """
    rng = np.random.default_rng(seed)
    paths = []
    for _ in range(int(rng.integers(1, 4))):
        point = rng.uniform(18.0, 82.0, size=2)
        commands = [f"M{_fmt(point[0])} {_fmt(point[1])}"]
        for _ in range(int(rng.integers(1, 4))):
            controls = rng.uniform(8.0, 92.0, size=(3, 2))
            point = controls[-1]
            commands.append(
                "C" + " ".join(f"{_fmt(x)} {_fmt(y)}" for x, y in controls),
            )
        paths.append(f'<path d="{" ".join(commands)}"/>')
    svg = (
        '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 100 100" '
        'fill="none" stroke="#111" stroke-width="2">'
        + "".join(paths)
        + "</svg>"
    )
    digest = hashlib.sha256(svg.encode()).hexdigest()
    artifact = DesignArtifact(
        id=f"gen-{digest[:16]}",
        name=f"Generated design {digest[:8]}",
        svg=svg,
        size_mm=(float(size_mm[0]), float(size_mm[1])),
        source={"kind": "generated", "model": PROCEDURAL_MODEL, "seed": int(seed)},
    )
    compile_svg_strokes(artifact.svg, artifact.size_mm)
    return artifact


def directory_artifacts(path: Path, size_mm: tuple[float, float]) -> tuple[DesignArtifact, ...]:
    """Read native acquisition outputs, with no SVG reconstruction or resizing."""
    from tatbot_contracts.artwork import canvas_m, validate_path_artwork
    from tatbot_contracts.paths import render_paths
    if not path.is_dir():
        raise ValueError(f"acquisition directory does not exist: {path}")
    directories = [path] if (path / "result.json").is_file() else sorted(p for p in path.iterdir() if p.is_dir())
    result = []
    for directory in directories:
        receipt_path = directory / "result.json"
        if not receipt_path.is_file():
            raise ValueError("requires a complete DBV3 acquisition with result.json and artwork.json")
        receipt = json.loads(receipt_path.read_text())
        raw = (directory / "artwork.json").read_bytes()
        record = validate_path_artwork(json.loads(raw))
        if (receipt.get("schema") != "tatbot.dbv3-acquisition/1"
                or receipt["outputs"]["artwork.json"] != hashlib.sha256(raw).hexdigest()
                or receipt["recipe_sha256"] != record["conversion"]["recipe_sha256"]
                or record["conversion"]["adapter"] != "dbv3-batik-paths/1"
                or record["program"]["provenance"]["producer"] != "dbv3-batik-paths/1"):
            raise ValueError("requires hash-bound acquired DBV3 artwork")
        recipe = (directory / "recipe/recipe.json").read_bytes()
        if hashlib.sha256(recipe).hexdigest() != receipt['recipe_sha256']:
            raise ValueError("acquisition recipe identity differs")
        geometry = {key: record["program"][key] for key in ("canvas_m", "inks", "layers", "negative_space_masks")}
        result.append(DesignArtifact(id=record["name"], name=record["name"], svg=render_paths(geometry),
                                     size_mm=tuple(v * 1000 for v in canvas_m(record)),
                                     source=record["source"], artwork=record))
    if not result:
        raise ValueError("requires a complete DBV3 acquisition")
    return tuple(result)
