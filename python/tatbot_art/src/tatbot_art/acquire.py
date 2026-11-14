"""From separated layers to a program: DBV3 jobs, acquisition, a tatbot-inks/3 draft and the compile.

Every path comes from DrawingBotV3 (`tatbot drawingbot generate`); compile is `tatbot ros compile`. The
inks file binds each ink to an assumed cap, water in a large one and the rest by how much line each
draws. It is a draft: the cap that really holds each ink is the palette's to say when the piece is
prepared for a run.
"""
from __future__ import annotations

import copy
import hashlib
import json
import re
import subprocess
from pathlib import Path

import numpy as np
import yaml

from .png import write_png

REPO = Path(__file__).resolve().parents[4]
DEFAULTS = REPO / "scripts" / "lib" / "drawingbot" / "defaults"
WATER_SLOT = "inkcap_large_1"
INK_SLOTS = ("inkcap_large_2", "inkcap_medium_1", "inkcap_medium_2", "inkcap_small_1", "inkcap_small_2")
FILLS = {  # fill -> (PFM, distribution); settings come from `settings`
    "hatch": ("Streamlines Flow Field", "Luminance weighted"),
    "solid": ("ECS Drawing", "Single pen"),
    "edges": ("ECS Drawing", "Single pen"),
}
# How a solid region is filled: (PFM, distribution, settings, source grey). `packed` is the measured default;
# the rest are textures the voting loop explores, and the drawn view shows whether each stays in its region.
SOLID_FILLS = {
    "packed": ("ECS Drawing", "Single pen", {"Plotting Resolution": 2.0, "Draw Edges": False, "Draw Contours": False}, 70),
    "contour": ("Streamlines Edge Field", "Luminance weighted", {"Threshold": 250.0, "Min Spacing": 1.0, "Max Spacing": 4.0}, 0),
    "flow": ("Streamlines Flow Field", "Luminance weighted", {"Threshold": 250.0}, 0),
    "sweep": ("Sketch Sweeping Curves", "Even weighted", {}, 40),
    "maze": ("Voronoi TSP", "Single pen", {"Ignore White": True}, 0),
    "waves": ("Sketch Waves", "Even weighted", {}, 40),
    # Added after a single-shape test. Over real designs only squares stays inside its layer's regions;
    # curves and superformula, like sweep, waves and maze, cross the white between them. The drawn view
    # shows that line, and the recipes skill records the measurement.
    "squares": ("Sketch Squares", "Even weighted", {}, 40),
    "curves": ("Sketch Curves", "Even weighted", {}, 40),
    "superformula": ("Sketch Superformula", "Even weighted", {}, 40),
}


def dbv3_angle(angle_deg: float) -> float:
    """DBV3's Start Angle turns clockwise in the artwork's y-up frame: 45 draws lines at 135 degrees."""
    return float((-angle_deg) % 180.0)


def settings(fill: str, angle_deg: float, seed: int, width_mm: float = 0.6) -> dict:
    """`hatch` is a flattened flow field: parallel streamlines at one angle, spaced by tone, ending at white."""
    if fill == "hatch":
        return {"Random Seed": seed, "Threshold": 250.0, "Start Angle": int(round(dbv3_angle(angle_deg))),
                "Amplitude": 0.001, "X Frequency": 0.001, "Y Frequency": 0.001, "Min Spacing": width_mm,
                "Max Spacing": round(10 * width_mm, 3), "Min Length": 5.0, "Max Length": 2000.0}
    if fill == "solid":
        return {"Random Seed": seed, **SOLID_FILLS["packed"][2]}
    if fill.startswith("solid:"):
        return {"Random Seed": seed, **SOLID_FILLS[fill.split(":", 1)[1]][2]}
    return {"Random Seed": seed, "Edge Blur": 5, "Plotting Resolution": 2.0, "Draw Contours": False,
            "Draw Shading": False}


def pen(look) -> dict:
    """A pen in the ink's colour, no channel at 255: DBV3 exports those as names the decoder refuses."""
    rgba = [min(254, int(round(c * 255))) for c in look.srgb] + [255]
    return {"id": look.ink_id.replace("_", "-"), "name": look.name, "type": "Liner", "enabled": True,
            "rgba": rgba, "weight": 100, "stroke_factor": 1.0}


def provenance(job_id: str, sidecar: dict | None, image_name: str) -> dict:
    if not sidecar:
        return {"kind": "imported", "identifier": job_id, "license": None, "generation": None,
                "attribution": f"separated by tatbot art from {image_name}"}
    return {"kind": "generated", "identifier": job_id, "license": None,
            "attribution": f"generated with {sidecar.get('generator')}, separated by tatbot art",
            "generation": {"prompt": sidecar["prompt"], "model": sidecar.get("model"), "model_revision": None,
                           "seed": None, "tracing": "tatbot art separate"}}


def job_id(name: str, index: int, layer) -> str:
    slug = re.sub(r"[^a-z0-9-]+", "-", f"{name}-{index:02d}-{layer.ink_id}-{layer.fill}".lower()).strip("-")
    return slug[:100]


def write_job(root: Path, jid: str, layer, look, canvas_mm, width_mm: float, seed: int, source_meta: dict) -> Path:
    d = root / jid
    (d / "input").mkdir(parents=True, exist_ok=True)
    png = write_png(d / "input" / "source.png", np.repeat(layer.source[..., None], 3, axis=2))
    pfm, distribution = (SOLID_FILLS[layer.fill.split(":", 1)[1]][:2] if layer.fill.startswith("solid:")
                         else FILLS[layer.fill])
    drawing_set = json.loads((DEFAULTS / "drawing-set.json").read_text())
    drawing_set.update({"distribution_type": distribution, "pens": [pen(look)]})
    job = {"schema": "tatbot.dbv3-job/3", "id": jid,
           "source": {"file": "input/source.png", "sha256": hashlib.sha256(png.read_bytes()).hexdigest(),
                      "provenance": copy.deepcopy(source_meta)},
           "size_mm": [round(float(v), 3) for v in canvas_mm], "pen_width_mm": width_mm, "pfm": pfm,
           "settings": settings(layer.fill, layer.angle_deg, seed, width_mm), "drawing_set": drawing_set,
           "state": json.loads((DEFAULTS / "state.json").read_text())}
    job["source"]["provenance"]["identifier"] = jid
    (d / "job.json").write_text(json.dumps(job, indent=1) + "\n")
    return d


def generate(job_dir: Path, out: Path, app: Path) -> Path | None:
    """One DBV3 acquisition through the logged CLI, reused while it exists; None when DBV3 refused it.

    A refusal (no drawable paths, a setting that did not read back) is the layer's, not the piece's: the
    caller records it and compiles the rest. Its reason is in the drawingbot run's log.
    """
    if not (out / "artwork.json").exists():
        run = subprocess.run([str(REPO / "scripts" / "drawingbot.sh"), "generate", str(job_dir / "job.json"),
                              "--out", str(out), "--app", str(app)], capture_output=True, text=True)
        if run.returncode:
            reason = [x for x in (run.stdout + run.stderr).splitlines() if x.startswith("drawingbot:")]
            (job_dir / "refused.txt").write_text((reason[-1] if reason else f"exit {run.returncode}") + "\n")
            return None
    return out / "artwork.json"


def fitted_tool(arm: str | None = None) -> str:
    """`arm`'s fitted tool; the drawing arm's (the first of stack.yaml arms), which `tatbot ros compile` compiles for."""
    if arm is None:
        from tatbot_description import drawing_arm

        arm = drawing_arm(REPO)
    return yaml.safe_load((REPO / "config" / "workspace.yaml").read_text())[arm]["tool_id"]


def inks_file(path: Path, layers_with_art, line_mm: dict[str, float], tool_id: str | dict[str, str],
              substrate: str = "silicon_skin") -> Path:
    """A tatbot-inks/3 draft: water in a large cap, the inks drawing the most line in the next largest.

    `tool_id` is the one cartridge, or each ink's. An ink on the cartridge the ink drawn before it used is taken up by
    a rinse; the first ink on another cartridge by a manual exchange, the arm landed while the operator swaps it."""
    inks = sorted({lay.ink_id for lay, _ in layers_with_art}, key=lambda i: -line_mm.get(i, 0.0))
    if len(inks) > len(INK_SLOTS):
        raise ValueError(f"{len(inks)} inks need more caps than the palette's {len(INK_SLOTS)} beside water")
    tool = tool_id.__getitem__ if isinstance(tool_id, dict) else (lambda _ink: tool_id)
    drawn = list(dict.fromkeys(lay.ink_id for lay, _ in layers_with_art))
    one_tool = len({tool(ink) for ink in drawn}) == 1
    rinsed = {ink for before, ink in zip(drawn, drawn[1:], strict=False) if tool(before) == tool(ink)}
    lines = ["format: tatbot-inks", "version: 3", f"substrate: {substrate}", "resources:"]
    for ink, slot in zip(inks, INK_SLOTS, strict=False):
        lines += [f"- id: {ink}", f"  tool_id: {tool(ink)}", f"  ink_id: {ink}", "  pen_mode: ride", f"  slot: {slot}"]
        lines += (["  activation:", "    method: rinse", f"    slot: {WATER_SLOT}", "    ink_id: water"]
                  if one_tool or ink in rinsed else ["  activation:", "    method: manual", "    action: exchange"])
    lines.append("bindings:")
    for lay, art in layers_with_art:
        digest = json.loads(art.read_text())["content_sha256"]
        lines += [f"- artwork_sha256: {digest}", f"  pen_id: {lay.ink_id.replace('_', '-')}", f"  resource_id: {lay.ink_id}"]
    path.write_text("\n".join(lines) + "\n")
    return path


def line_mm(artwork: Path) -> float:
    program = json.loads(artwork.read_text())["program"]
    return float(sum(np.linalg.norm(np.diff(np.asarray(el["points_m"]), axis=0), axis=1).sum()
                     for layer in program["layers"] for el in layer["elements"]) * 1000)


def _drawable(stencil: Path | None, at: str | None) -> tuple[np.ndarray, np.ndarray]:
    """(half the drawable width and height, the canvas centre), mm, on the page `tatbot ros compile` would place a
    canvas on: the print `stencil` names, else the nominal page. `at` is the compile's centre, X,Y mm."""
    from tatbot_ink import place

    page = place.print_page(stencil, REPO) if stencil else place.stencil_page()
    centre_mm = np.array([float(v) for v in at.split(",")]) if at else np.zeros(2)
    return place.drawable_half(page) * 1000, centre_mm


def page_room(stencil: Path | None = None, at: str | None = None) -> tuple[float, float]:
    """The largest canvas, width and height in mm, that the page holds centred at `at`."""
    half_mm, centre_mm = _drawable(stencil, at)
    room = 2 * (half_mm - np.abs(centre_mm))
    if (room <= 0).any():
        raise ValueError(f"--at {at} lies off the page's drawable area")
    return float(room[0]), float(room[1])


def page_fit(canvas_mm, stencil: Path | None = None, at: str | None = None) -> dict:
    """Whether a canvas fits the page `tatbot ros compile` would place it on, without DBV3 having drawn it: the
    print `stencil` names, else the nominal page. `at` is the compile's canvas centre, X,Y mm in the page frame."""
    half_mm, centre_mm = _drawable(stencil, at)
    fits = bool(np.all(np.abs(centre_mm) + np.asarray(canvas_mm) / 2 <= half_mm + 1e-6))
    return {"canvas_mm": [round(float(v), 2) for v in canvas_mm], "at_mm": centre_mm.tolist(),
            "drawable_mm": [round(float(v), 2) for v in 2 * half_mm], "stencil": str(stencil) if stencil else None,
            "fits": fits}


def compile_program(artworks: list[Path], inks: Path, out: Path, extra: list[str]) -> dict:
    run = subprocess.run([str(REPO / "scripts" / "tatbot"), "ros", "compile", *map(str, artworks), "--inks", str(inks),
                          "-o", str(out), *extra], capture_output=True, text=True)
    stats = next((json.loads(x) for x in reversed(run.stdout.splitlines()) if x.startswith("{")), None)
    if run.returncode or stats is None:
        raise RuntimeError(f"tatbot ros compile failed: {(run.stderr or run.stdout).strip()[-600:]}")
    return stats


def canvas_program(layers_with_art, canvas_mm) -> dict:
    """The acquired paths as a program on a page the size of the canvas, for comparing with the target."""
    w, h = canvas_mm
    resources, ops = {}, []
    for lay, art in layers_with_art:
        resources.setdefault(lay.ink_id, {"id": lay.ink_id, "ink_id": lay.ink_id, "tool": {}})
        for layer in json.loads(art.read_text())["program"]["layers"]:
            for el in layer["elements"]:
                pts = (np.asarray(el["points_m"], float) - [w / 2000, h / 2000]).tolist()
                ops.append({"op": "stroke", "resource_id": lay.ink_id, "points_m": pts,
                            "generation_width_m": el.get("width_m"), "deposition": el.get("deposition", 1.0)})
    return {"page": {"size_m": [w / 1000, h / 1000]}, "resources": list(resources.values()), "ops": ops}
