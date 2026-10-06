"""Geometry the Rerun bridge logs, as plain numpy (no ROS, no rerun): page outlines, program strokes
and the measured tip path, all in `world`."""
from __future__ import annotations

import numpy as np
from tatbot_description.transforms import quat_matrix as matrix  # noqa: F401 -- existing geometry API


def rectangle(width_m: float, height_m: float) -> np.ndarray:
    """A closed rectangle centred on the page origin, in the page plane (z = 0), 5 x 3."""
    w, h = width_m / 2, height_m / 2
    return np.array([[-w, -h, 0], [w, -h, 0], [w, h, 0], [-w, h, 0], [-w, -h, 0]], float)


def transform(matrix, points) -> np.ndarray:  # noqa: F811 -- preserve the existing keyword argument
    points = np.asarray(points, float)
    return points @ np.asarray(matrix, float)[:3, :3].T + np.asarray(matrix, float)[:3, 3]


def page_outlines(world_from_page, size_m, clear_m) -> list[np.ndarray]:
    """The page edge and its clear center, in world."""
    return [transform(world_from_page, rectangle(*size_m)), transform(world_from_page, rectangle(*clear_m))]


def program_strokes(program: dict, world_from_page) -> list[np.ndarray]:
    """Every stroke op's points_m (page-frame x, y) on the page plane, in world, in program order."""
    strokes = []
    for op in program.get("ops", []):
        if op.get("op") == "stroke" and op.get("points_m"):
            xy = np.asarray(op["points_m"], float).reshape(-1, 2)
            strokes.append(transform(world_from_page, np.column_stack([xy, np.zeros(len(xy))])))
    return strokes


class TipPath:
    """The measured tip path of one arm, in chunks so each log carries at most `chunk` points: a point is
    added once the tip moved `min_step_m` from the last one; a full chunk is closed and the next starts
    at its last point."""

    def __init__(self, min_step_m: float = 0.0002, chunk: int = 200):
        self.min_step_m, self.chunk = min_step_m, chunk
        self.index = 0
        self.points: list[np.ndarray] = []

    def add(self, point) -> bool:
        p = np.asarray(point, float)
        if self.points and np.linalg.norm(p - self.points[-1]) < self.min_step_m:
            return False
        if len(self.points) >= self.chunk:
            self.index, self.points = self.index + 1, self.points[-1:]
        self.points.append(p)
        return True

    def clear(self) -> None:
        self.index, self.points = 0, []

    def array(self) -> np.ndarray:
        """The open chunk (number `index`)."""
        return np.array(self.points, float).reshape(-1, 3)
