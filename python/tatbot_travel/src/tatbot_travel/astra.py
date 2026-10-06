"""Astra decides where the pen goes: it looks at the wrist image and names the ink to follow next.

Each call sends one wrist frame with a labelled pixel grid, a crosshair where the pen points now and
the path already traced, and asks for the next stretch of ink line as image points in order. Astra
alone chooses those points -- which line, which way, where to stop. Local code only lifts them onto
the measured skin through the frame's own depth and plays them through the shared executor; no ink
detector is consulted. Calls are pipelined: the next is sent when a stretch starts moving, anchored
at that stretch's planned end, so the arm keeps moving while the model thinks.
"""

from __future__ import annotations

import base64
import json
import os
import time
from dataclasses import dataclass, field

import cv2
import numpy as np

from tatbot_travel.surface import Frame, Surface

MODEL = os.environ.get("TATBOT_ASTRA_MODEL", "gpt-6-astra")
EFFORT = os.environ.get("TATBOT_ASTRA_EFFORT", "low")
GRID_PX = 40

INSTRUCTIONS = """You are steering a robot arm that holds a laser pen a few millimetres above a silicone \
practice forearm with ink drawn on it. The image comes from a camera on the robot's wrist.

Marks drawn on the image:
- cyan crosshair: where the pen points right now (or, when labelled "END", where the current move will end);
- green line: the ink already traced; do not trace it again;
- white grid lines every 40 pixels, labelled with pixel coordinates (u to the right, v down).

Trace the ink. Return the next stretch of ink line to follow, as 10 to 30 pixel points [u, v] in order \
along the centre of one ink line, about 150 to 250 pixels long. Start at the crosshair if it is on a \
line; otherwise start on the nearest untraced line. Points must be on ink, on the forearm. When the \
line ends or branches, pick the continuation that keeps tracing untraced ink. If no untraced ink is \
visible, return no points and done=true. Give a one-sentence reason a spectator would understand."""

SCHEMA = {
    "type": "object",
    "properties": {
        "reason": {"type": "string"},
        "points": {"type": "array", "items": {"type": "array", "items": {"type": "integer"},
                                              "minItems": 2, "maxItems": 2}, "maxItems": 40},
        "done": {"type": "boolean"},
    },
    "required": ["reason", "points", "done"],
    "additionalProperties": False,
}


@dataclass
class Decision:
    reason: str
    points_px: np.ndarray  # (N, 2) u, v
    done: bool
    latency_s: float
    usage: dict = field(default_factory=dict)


def annotate(rgb: np.ndarray, pen_px: np.ndarray, traced_px: np.ndarray | None = None, *,
             end: bool = False) -> np.ndarray:
    """The image Astra sees: grid, crosshair and the traced path (RGB in, RGB out)."""
    out = np.ascontiguousarray(rgb).copy()
    h, w = out.shape[:2]
    grid = out.copy()
    for u in range(0, w, GRID_PX):
        cv2.line(grid, (u, 0), (u, h - 1), (255, 255, 255), 1)
    for v in range(0, h, GRID_PX):
        cv2.line(grid, (0, v), (w - 1, v), (255, 255, 255), 1)
    out = cv2.addWeighted(grid, 0.35, out, 0.65, 0)
    for u in range(0, w, 2 * GRID_PX):
        cv2.putText(out, str(u), (u + 2, 12), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1)
    for v in range(0, h, 2 * GRID_PX):
        cv2.putText(out, str(v), (2, v + 12), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1)
    if traced_px is not None and len(traced_px) > 1:
        cv2.polylines(out, [np.round(traced_px).astype(np.int32)], False, (0, 255, 0), 2)
    c = tuple(int(round(x)) for x in pen_px)
    cv2.drawMarker(out, c, (0, 255, 255), cv2.MARKER_CROSS, 22, 2)
    if end:
        cv2.putText(out, "END", (c[0] + 8, c[1] - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1)
    return out


class Astra:
    """One model, one conversation-free call per stretch (each call is a fresh look)."""

    def __init__(self, model: str = MODEL, effort: str = EFFORT):
        from openai import OpenAI

        if not os.environ.get("OPENAI_API_KEY"):
            raise RuntimeError("OPENAI_API_KEY is not set")
        self.client, self.model, self.effort = OpenAI(), model, effort

    def decide(self, image_rgb: np.ndarray) -> Decision:
        ok, jpeg = cv2.imencode(".jpg", cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 90])
        if not ok:
            raise RuntimeError("could not encode the wrist image")
        url = "data:image/jpeg;base64," + base64.b64encode(jpeg.tobytes()).decode()
        started = time.monotonic()
        response = self.client.responses.create(
            model=self.model, instructions=INSTRUCTIONS, reasoning={"effort": self.effort},
            input=[{"role": "user", "content": [{"type": "input_text", "text": "Next stretch of ink to trace."},
                                                {"type": "input_image", "image_url": url}]}],
            text={"format": {"type": "json_schema", "name": "next_stretch", "schema": SCHEMA, "strict": True}})
        latency = time.monotonic() - started
        answer = json.loads(response.output_text)
        points = np.array(answer["points"], float).reshape(-1, 2)
        usage = response.usage.model_dump() if getattr(response, "usage", None) else {}
        return Decision(reason=answer["reason"], points_px=points, done=bool(answer["done"]) or len(points) < 2,
                        latency_s=latency, usage=usage)


def lift(frame: Frame, skin: Surface, points_px: np.ndarray, *, step_m: float = 0.001) -> tuple[np.ndarray, np.ndarray]:
    """Astra's pixels onto the skin: each through the frame's own depth (a 5x5 median, so a pixel on a
    line's dark centre with no depth still lands), then onto the measured surface; resampled per mm."""
    from tatbot_travel.inkmap import resample

    h, w = frame.depth_m.shape
    base = []
    for u, v in np.round(points_px).astype(int):
        if not (0 <= u < w and 0 <= v < h):
            continue
        patch = frame.depth_m[max(0, v - 2):v + 3, max(0, u - 2):u + 3]
        patch = patch[patch > 0]
        if not len(patch):
            continue
        x, y = frame.intr.undistort_normalized(np.array([float(u)]), np.array([float(v)]))
        z = float(np.median(patch))
        base.append(np.array([x[0] * z, y[0] * z, z]) @ frame.cam_r.T + frame.cam_p)
    if len(base) < 2:
        return np.empty((0, 3)), np.empty((0, 3))
    on_skin, normals, supported = skin.project_path(resample(np.array(base), step_m))
    return on_skin[supported], normals[supported]
