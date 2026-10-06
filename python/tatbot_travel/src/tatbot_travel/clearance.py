"""How close the pen is to the skin.

The pen must never touch the phantom. The generator measures the clearance
of the pen's nose (the lens face and a few stations behind it) to the skin
every tick, so the expert can flinch before contact and evaluations can
count near misses. Distances come from dense surface samples in the
phantom's own frame, so a moving phantom costs one inverse transform.
"""

from __future__ import annotations

import numpy as np
import trimesh
from scipy.spatial import cKDTree

from tatbot_travel.motion import Pose
from tatbot_travel.phantom import Phantom

NOSE_STATIONS_M = (0.0, -0.01, -0.02, -0.035)  # along the pen axis from the lens face, toward the tail


class SkinDistance:
    def __init__(self, phantom: Phantom, spacing_m: float = 0.0015, seed: int = 0):
        mesh = trimesh.Trimesh(phantom.vertices, np.vstack([phantom.faces, phantom.cap_faces]), process=False)
        count = max(1000, int(mesh.area / spacing_m ** 2))
        samples, _ = trimesh.sample.sample_surface_even(mesh, count, seed=seed)
        self.points = np.vstack([samples, phantom.vertices])
        self.tree = cKDTree(self.points)

    def __call__(self, pose: Pose, points: np.ndarray) -> np.ndarray:
        """Distance (m) from world ``points`` to the phantom skin at ``pose``."""
        return self.tree.query(pose.rot.inv().apply(points - pose.pos))[0]

    def pen(self, pose: Pose, lens: np.ndarray, axis: np.ndarray) -> float:
        """Smallest clearance over the pen's nose: the lens face and stations behind it."""
        points = lens[None] + np.asarray(NOSE_STATIONS_M)[:, None] * axis[None]
        return float(self(pose, points).min())

    def escape(self, pose: Pose, lens: np.ndarray, axis: np.ndarray) -> np.ndarray:
        """World unit direction that moves the closest nose station straight away from the skin."""
        points = lens[None] + np.asarray(NOSE_STATIONS_M)[:, None] * axis[None]
        local = pose.rot.inv().apply(points - pose.pos)
        dist, index = self.tree.query(local)
        k = int(np.argmin(dist))
        away = pose.rot.apply(local[k] - self.points[index[k]])
        norm = np.linalg.norm(away)
        return away / norm if norm > 1e-9 else -axis
