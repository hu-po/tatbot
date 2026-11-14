"""A flat sheet wrapped onto a developable cone with an asymmetric spline section.

The printed half-skin mounts (``cad/half-skin-mount-*/design.json``) are this surface:
straight lines from a common apex outside the mount, through a cross-section of two
splines that meet at a crest. A cone unrolls without stretch, so the sheet's
mid-surface keeps its flat rectangle; the faces either side of it stretch and
compress with the bend. The mount's CAD exporter and the simulator's fixture both
build from this one class, so the simulated skin lies on the surface that is printed.

Lengths are the design's, times ``unit`` (the designs are in millimetres; ``unit=1e-3``
gives metres). Frame: the developed sheet's centre over the origin, x along its short
edge, z up from the mount's flat bottom. numpy and scipy only.
"""

from __future__ import annotations

import numpy as np
from scipy.interpolate import CubicSpline, PchipInterpolator

SAMPLES = 8193
"""Section samples the flat angle is integrated over."""


class Developable:
    def __init__(self, design: dict, unit: float = 1.0):
        width, length, thickness = (unit * float(v) for v in design["skin_mm"])
        angle = np.deg2rad(design["crest_skew_deg"])
        c, s = np.cos(angle), np.sin(angle)
        self.rotation = np.array([[c, -s], [s, c]])
        self.half_skin = np.array([width, length]) / 2
        self.half_body = self.half_skin + unit * np.asarray(design["border_mm"], float)
        self.thickness = thickness
        self.crest = unit * float(design["crest_offset_mm"])
        self.height = unit * float(design["crest_height_mm"])
        self.apex = unit * float(design["apex_along_mm"])
        if self.apex <= np.abs(self.rotation[1]) @ self.half_body:
            raise ValueError("the cone's apex lies inside the mount")
        left, right = design["left_profile_mm"], design["right_profile_mm"]
        if not left[-1] == right[0] == [0, 0]:
            raise ValueError("both profiles meet at the crest, [0, 0]")
        x, z = unit * np.array(left + right[1:], float).T
        self.profile = PchipInterpolator(x, z, extrapolate=False)
        self.profile_slope = self.profile.derivative()
        # Preserve polar distances by matching the flat angle to the arc
        # length traced by the unit rays from the apex on the unit sphere.
        sample = np.unique(np.r_[np.linspace(x[0], x[-1], SAMPLES), x])
        rays, cross = self.rays(sample)
        angular_speed = np.linalg.norm(cross, axis=-1) / np.sum(rays**2, axis=-1)
        arc = CubicSpline(sample, angular_speed).antiderivative()
        self.ray_at_angle = PchipInterpolator(arc(sample) - arc(0), sample, extrapolate=False)

    def rays(self, x):
        z, slope = self.profile(x), self.profile_slope(x)
        rays = np.stack((x, np.full_like(x, -self.apex), z), axis=-1)
        # Cross product of the ray and its lateral derivative gives the upward normal.
        cross = np.stack((-self.apex * slope, z - x * slope, np.full_like(x, self.apex)), axis=-1)
        return rays, cross

    def wrap(self, flat, offset=0):
        """Points and outward normals of flat sheet points ``(..., 2)``. ``offset`` is measured outward
        from the printed backing: half a thickness is the sheet's mid-surface, a whole one its outer face."""
        across, along = np.moveaxis(np.asarray(flat, float) @ self.rotation.T, -1, 0)
        lateral, longitudinal = across - self.crest, self.apex - along
        radius = np.hypot(lateral, longitudinal)
        section_x = self.ray_at_angle(np.arctan2(lateral, longitudinal))
        rays, cross = self.rays(section_x)
        normals = cross / np.linalg.norm(cross, axis=-1, keepdims=True)
        apex = np.array([self.crest, self.apex, self.height + self.thickness / 2])
        points = apex + rays * (radius[..., None] / np.linalg.norm(rays, axis=-1, keepdims=True))
        points += normals * (np.asarray(offset) - self.thickness / 2)[..., None]
        points[..., :2] = points[..., :2] @ self.rotation
        normals[..., :2] = normals[..., :2] @ self.rotation
        return points, normals
