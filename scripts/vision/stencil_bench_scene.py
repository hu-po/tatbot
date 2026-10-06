"""Stencil bench tier 0: seeded 2-D scenes of a stencil transfer on skin, with dense truth.

A scene is a camera looking at a transferred stencil on a flat or cylindrical
skin surface. Page millimetres `s` are the surface's own (unrolled, arc-length)
coordinates; the transfer puts artwork point `p` at `s = mirror(p) + wobble(p)`,
so the truth for a page-UV point is where its *transferred* ink landed, a dense
field rather than one homography. Units: millimetres in the surface frame,
pixels in the image. The surface frame has its origin at the page centre, X
along page u, Y along page v (down) and Z into the skin.

Nothing here reads a camera or touches hardware. Camera models come from the
checked-in camera registry and, when present, the fleet calibration bundle.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

# Rough fit of the transfer degradation to one phone photo of seed "1" printed
# beside its violet transfer on cream fake skin (scripts/vision/stencil_bench_fit.py).
# The photo and the full fit are kept with the experiment record, not in this repository.
PHOTO_FIT = {
    "line_spread_factor": 1.91,          # transfer line width / printed line width (0.84 / 0.44 mm)
    "density_vs_print": 0.54,            # transfer ink density relative to toner on paper
    "violet_transmittance_rgb": (0.70, 0.68, 1.0),
    "cream_skin_rgb": (243, 218, 186),
    "wobble_p50_mm": 0.43, "wobble_rms_detrended_mm": 0.97,
    "stray_ink_fraction_of_band": 0.05,  # pooling and smudge outside the spread artwork
    "faded_stroke_fraction": 0.21,       # stroke centre-line under half the ink level
    "mirrored": False,
}

INKS = {"violet": PHOTO_FIT["violet_transmittance_rgb"],
        # Light-blue transfer: absorbs red most, a little green. Assumed, not yet photographed.
        "light-blue": (0.58, 0.82, 0.97)}
SKIN_TONES = (((243, 218, 186), .30), ((236, 198, 172), .25), ((205, 158, 122), .20),
              ((168, 118, 88), .15), ((112, 76, 56), .10))
BANKS = {"train": 0x57E4C11, "holdout": 0x40D0F7}
WRIST_PROFILES = ((640, 480, .75), (1280, 720, .25))  # visiond stream; ROS inspect stream
PAD_MARGIN_MM = 40.0


# --- artwork ---------------------------------------------------------------------------------

@dataclass
class Artwork:
    pattern_id: str
    page_mm: tuple[float, float]
    ink: np.ndarray          # bool, page raster at `ppm` pixels per mm, True = black
    ppm: float
    margin_mm: float
    frame_mm: float
    reference: Path | None   # tracking.json beside stencil.png, for reference-bank trackers
    meta: dict

    def in_frame(self, s_mm):
        """True where page point `s_mm` lies in the printed frame band."""
        s = np.asarray(s_mm, float).reshape(-1, 2)
        w, h = self.page_mm
        m, f = self.margin_mm, self.frame_mm
        outer = (s[:, 0] >= m) & (s[:, 0] <= w-m) & (s[:, 1] >= m) & (s[:, 1] <= h-m)
        inner = (s[:, 0] > m+f) & (s[:, 0] < w-m-f) & (s[:, 1] > m+f) & (s[:, 1] < h-m-f)
        return outer & ~inner


def load_artwork(directory):
    directory = Path(directory).expanduser()
    manifest = json.loads((directory/"tracking.json").read_text())
    settings = json.loads((directory/"settings.json").read_text())
    image = cv2.imread(str(directory/"stencil.png"), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError(f"no stencil.png in {directory}")
    page = tuple(float(v) for v in manifest["page_mm"])
    return Artwork(pattern_id=manifest["pattern_id"], page_mm=page, ink=image < 128,
                   ppm=image.shape[1]/page[0], margin_mm=float(settings["margin_mm"]),
                   frame_mm=float(settings["frame_mm"]), reference=directory/"tracking.json",
                   meta={"seed": manifest.get("seed"), "settings": settings,
                         "marked": manifest.get("instance_mark") is not None})


# --- cameras ---------------------------------------------------------------------------------

@dataclass
class CameraModel:
    name: str
    role: str                # wrist | overhead
    width: int
    height: int
    k: tuple[float, float, float, float]
    dist: tuple[float, ...]
    basis: str
    surface_from_camera_candidates: tuple = ()   # overhead: table-frame poses (base frame, mm)

    def intrinsics(self):
        fx, fy, cx, cy = self.k
        return {"width": self.width, "height": self.height, "fx": fx, "fy": fy, "cx": cx, "cy": cy,
                "distortion": list(self.dist), "model": "brown_conrady", "basis": self.basis}


def _workspace_page(repo):
    """The drawing pivot and paper height in the right arm base frame, metres."""
    from stencil_coded_live import drawing_pads
    return drawing_pads(repo)["right"]


def wrist_models(repo):
    """D405 colour models from the camera registry (declared intrinsics, else nominal FOV)."""
    try:
        import wrist_cameras
        described = wrist_cameras.describe(Path(repo), arms=("right",))[0]
        fx, fy, cx, cy = described.intrinsic
        width, height, basis = described.width, described.height, described.intrinsic_basis
    except Exception:  # noqa: BLE001 - a bare checkout still benches with the nominal D405
        focal = 480/(2*math.tan(.96/2))
        fx = fy = focal
        cx, cy, width, height, basis = 320., 240., 640, 480, "nominal-fov"
    models = []
    for w, h, _ in WRIST_PROFILES:
        # Same vertical field of view in the 16:9 inspect stream.
        scale = h/height
        models.append(CameraModel(f"wrist-{w}x{h}", "wrist", w, h,
                                  (fx*scale, fy*scale, w/2+(cx-width/2)*scale, h/2+(cy-height/2)*scale),
                                  (0., 0., 0., 0., 0.), basis))
    return models


def overhead_models(repo, calibration=None, robot_world=None):
    """Overhead PoE models posed over the drawing pivot from the calibration bundle; a
    nominal camera when the bundle is absent (bare clone, CI).

    The robot-world record registers the URDF root (its `world_from_base` is a legacy name,
    robot_world.py) and the pivot is in the right arm's base frame, so the arm's mount is
    crossed between them; without it every camera sat off by that mount offset."""
    from ink_spec import base_from_root_matrix
    from robot_world import root_from_world
    try:
        bundle = json.loads(Path(calibration).expanduser().read_text())
        base_from_world = base_from_root_matrix(repo, "right") @ root_from_world(
            json.loads(Path(robot_world).expanduser().read_text()))
        page_base = _workspace_page(repo)
    except (OSError, TypeError, KeyError, ValueError):
        return [_nominal_overhead()]
    models = []
    for name, entry in sorted(bundle["cameras"].items()):
        intr = entry["intrinsics"]
        rotation = np.array(entry["world_from_camera"]["rotation"]).reshape(3, 3)
        world_from_camera = np.eye(4)
        world_from_camera[:3, :3], world_from_camera[:3, 3] = rotation, entry["world_from_camera"]["translation_m"]
        base_from_camera = base_from_world @ world_from_camera
        base_from_camera[:3, 3] = (base_from_camera[:3, 3]-page_base)*1000  # mm, relative to the pivot
        coefficients = tuple(float(c) for c in entry.get("distortion", {}).get("coefficients", [0]*5))
        models.append(CameraModel(name, "overhead", int(intr["width"]), int(intr["height"]),
                                  (intr["fx"], intr["fy"], intr["cx"], intr["cy"]),
                                  (coefficients+(0.,)*5)[:5], "calibration-bundle", (base_from_camera,)))
    return models


def _nominal_overhead():
    """0.8 m away, 35 degrees off the table normal; stated nominal, never measured."""
    tilt = math.radians(35)
    camera_z = np.array([0, math.sin(tilt), -math.cos(tilt)])      # looks down and forward (base frame)
    camera_x = np.array([1., 0, 0])
    camera_y = np.cross(camera_z, camera_x)
    pose = np.eye(4)
    pose[:3, :3] = np.stack([camera_x, camera_y, camera_z], 1)
    pose[:3, 3] = -camera_z*800
    return CameraModel("overhead-nominal", "overhead", 2960, 1668, (1640., 1645., 1480., 834.),
                       (0.,)*5, "nominal", (pose,))


def pivot_view(model):
    """Pixels per mm, obliquity and image position of the drawing pivot for an overhead model."""
    base_from_camera = model.surface_from_camera_candidates[0]
    rotation, origin = base_from_camera[:3, :3], base_from_camera[:3, 3]
    points = (np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0.]])-origin) @ rotation
    fx, fy, cx, cy = model.k
    pixels = np.c_[fx*points[:, 0]/points[:, 2]+cx, fy*points[:, 1]/points[:, 2]+cy]
    ray = -origin/np.linalg.norm(origin)
    return {"distance_mm": round(float(np.linalg.norm(origin)), 1),
            "obliquity_deg": round(math.degrees(math.acos(min(1., abs(ray[2])))), 1),
            "px_per_mm": round(float(np.sqrt(abs(cross2(pixels[1]-pixels[0], pixels[2]-pixels[0])))), 3),
            "pivot_px": [round(float(v), 1) for v in pixels[0]],
            "in_view": bool(points[0, 2] > 0 and 0 <= pixels[0, 0] < model.width and 0 <= pixels[0, 1] < model.height),
            "basis": model.basis}


def usable_overhead(models, max_obliquity_deg=70.):
    """Cameras that see the drawing pivot at a usable angle, and the per-camera view record."""
    views = {model.name: pivot_view(model) for model in models}
    keep = [m for m in models if views[m.name]["in_view"] and views[m.name]["obliquity_deg"] <= max_obliquity_deg]
    return keep or models[:1], views


# --- surfaces --------------------------------------------------------------------------------

def _rot2(angle):
    c, s = math.cos(angle), math.sin(angle)
    return np.array([[c, -s], [s, c]])


class Surface:
    """Flat (radius None) or a cylinder bulging toward the camera, crest near the page centre."""

    def __init__(self, page_mm, radius_mm=None, axis_deg=0., crest_offset_mm=0.):
        self.centre = np.array(page_mm, float)/2
        self.radius = radius_mm
        self.alpha = math.radians(axis_deg)
        self.rot = _rot2(self.alpha)
        self.offset = self.rot @ np.array([crest_offset_mm, 0.])

    def _wrap_coords(self, s):
        return (np.asarray(s, float).reshape(-1, 2)-self.centre-self.offset) @ self.rot  # (across, along)

    def point(self, s):
        """Surface-frame 3-D point and outward normal (toward -Z) for page mm `s`."""
        q = self._wrap_coords(s)
        if self.radius is None:
            local = np.c_[q, np.zeros(len(q))]
            normal = np.tile([0., 0., -1.], (len(q), 1))
        else:
            r, phi = self.radius, q[:, 0]/self.radius
            local = np.c_[r*np.sin(phi), q[:, 1], r*(1-np.cos(phi))]
            normal = np.c_[np.sin(phi), np.zeros(len(q)), -np.cos(phi)]
        rot3 = np.eye(3)
        rot3[:2, :2] = self.rot
        point = local @ rot3.T
        point[:, :2] += self.offset
        return point, normal @ rot3.T

    def intersect(self, origin, directions):
        """First skin hit of rays `origin + t*d`: (page mm, 3-D point, normal, hit mask)."""
        rot3 = np.eye(3)
        rot3[:2, :2] = self.rot
        o = (np.asarray(origin, float)-np.r_[self.offset, 0.]) @ rot3
        d = directions @ rot3
        if self.radius is None:
            with np.errstate(divide="ignore", invalid="ignore"):
                t = -o[2]/d[:, 2]
            hit = np.isfinite(t) & (t > 0)
            local = o+t[:, None]*d
            q = local[:, :2]
        else:
            r = self.radius
            a = d[:, 0]**2+d[:, 2]**2
            b = 2*(o[0]*d[:, 0]+(o[2]-r)*d[:, 2])
            c = o[0]**2+(o[2]-r)**2-r*r
            disc = b*b-4*a*c
            with np.errstate(divide="ignore", invalid="ignore"):
                t = (-b-np.sqrt(np.maximum(disc, 0)))/(2*a)
            hit = (disc > 0) & (t > 0)
            local = o+t[:, None]*d
            q = np.c_[r*np.arctan2(local[:, 0], r-local[:, 2]), local[:, 1]]
        s = q @ self.rot.T+self.centre+self.offset
        point, normal = self.point(s)
        return s, point, normal, hit

    def table_z(self):
        """The table plane under the skin, along +Z (pad thickness or arm diameter)."""
        return 3. if self.radius is None else 2*self.radius


# --- transfer --------------------------------------------------------------------------------

def smooth_noise(rng, shape, corr_px):
    """Zero-mean unit-variance noise with a Gaussian correlation length of `corr_px`."""
    small = (max(2, int(shape[0]/max(corr_px, 1))+3), max(2, int(shape[1]/max(corr_px, 1))+3))
    field = cv2.resize(rng.standard_normal(small).astype(np.float32), (shape[1], shape[0]),
                       interpolation=cv2.INTER_CUBIC)
    field = cv2.GaussianBlur(field, (0, 0), max(corr_px/3, .5))
    return (field-field.mean())/max(float(field.std()), 1e-6)


def cross2(a, b):
    return a[..., 0]*b[..., 1]-a[..., 1]*b[..., 0]


def sample_points(image, xy, border=cv2.BORDER_CONSTANT, value=0.):
    """Bilinear samples of `image` at pixel coordinates `xy` (N,2); remap caps each axis at 32767."""
    xy = np.asarray(xy, np.float32).reshape(-1, 2)
    width = 4096
    rows = max(1, -(-len(xy)//width))
    padded = np.zeros((rows*width, 2), np.float32)
    padded[:len(xy)] = xy
    grid = padded.reshape(rows, width, 2)
    out = cv2.remap(image, grid[..., 0], grid[..., 1], cv2.INTER_LINEAR, borderMode=border, borderValue=value)
    return out.reshape(rows*width, *image.shape[2:])[:len(xy)]


def _disk(radius_px):
    size = max(1, int(round(2*radius_px)) | 1)
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size))


class Transfer:
    """Ink density on the skin, in page mm: spread, wash-off, pooling, smudge, wobble, mirror."""

    def __init__(self, artwork, params, rng):
        self.artwork, self.params = artwork, params
        ppm, ink = artwork.ppm, artwork.ink
        density = ink.astype(np.float32)
        stroke = artwork.meta.get("settings", {}).get("stroke_mm", .45)
        grow = (params["spread_factor"]-1)*stroke/2
        if grow > 0:
            density = cv2.dilate(density, _disk(grow*ppm))
        density = cv2.GaussianBlur(density, (0, 0), max(params["edge_blur_mm"]*ppm, .3))
        keep, self.washed_fraction = self._washoff(rng, ink, ppm)
        fade = np.clip(1+params["fade"]*smooth_noise(rng, ink.shape, 3*ppm), .35, 1.15)
        density = density*keep*fade
        density = self._pools(rng, density, ink, ppm)
        density = self._smudges(rng, density, ink, ppm)
        self.density = np.clip(density, 0, 1)*params["density"]
        self._wobble_field(rng)

    def _washoff(self, rng, ink, ppm):
        p = self.params
        field = smooth_noise(rng, ink.shape, p["washoff_corr_mm"]*ppm)
        band = ink & self.artwork.in_frame(self._grid_mm()).reshape(ink.shape)
        values = field[band] if band.any() else field.ravel()
        cut = float(np.quantile(values, p["washoff"])) if p["washoff"] > 0 else -np.inf
        soft = .15
        keep = np.clip(.5+(field-cut)/(2*soft), 0, 1) if np.isfinite(cut) else np.ones_like(field)
        if p["speckle"] > 0:
            fine = smooth_noise(rng, ink.shape, .6*ppm)
            keep = keep*(fine > np.quantile(fine, p["speckle"]))
        washed = float((keep[band] < .5).mean()) if band.any() else 0.
        return keep.astype(np.float32), washed

    def _grid_mm(self):
        h, w = self.artwork.ink.shape
        ys, xs = np.mgrid[0:h, 0:w]
        return np.c_[(xs.ravel()+.5)/self.artwork.ppm, (ys.ravel()+.5)/self.artwork.ppm]

    def _ink_points(self, rng, ink, count):
        rows, cols = np.nonzero(ink)
        if count == 0 or not len(rows):
            return np.empty((0, 2), int)
        pick = rng.integers(0, len(rows), count)
        return np.c_[cols[pick], rows[pick]]

    def _pools(self, rng, density, ink, ppm):
        pools = np.zeros_like(density)
        for x, y in self._ink_points(rng, ink, rng.poisson(self.params["pools"])):
            axes = (int(rng.uniform(.4, 1.6)*ppm), int(rng.uniform(.4, 1.6)*ppm))
            cv2.ellipse(pools, (int(x), int(y)), axes, float(rng.uniform(0, 180)), 0, 360,
                        float(rng.uniform(.7, 1.)), -1)
        pools = cv2.GaussianBlur(pools, (0, 0), .15*ppm)
        return np.maximum(density, pools)

    def _smudges(self, rng, density, ink, ppm):
        for x, y in self._ink_points(rng, ink, rng.poisson(self.params["smudges"])):
            length = max(3, int(rng.uniform(2, 6)*ppm))
            kernel = np.zeros((length, length), np.float32)
            kernel[length//2, :] = 1/length
            rot = cv2.getRotationMatrix2D((length/2-.5, length/2-.5), float(rng.uniform(0, 180)), 1)
            kernel = cv2.warpAffine(kernel, rot, (length, length))
            kernel /= max(kernel.sum(), 1e-6)
            half = int(rng.uniform(2, 5)*ppm)
            y0, y1 = max(0, y-half), min(density.shape[0], y+half)
            x0, x1 = max(0, x-half), min(density.shape[1], x+half)
            patch = density[y0:y1, x0:x1]
            smear = cv2.filter2D(patch, -1, kernel)
            window = cv2.GaussianBlur(np.pad(np.ones((max(1, y1-y0-2*half//3), max(1, x1-x0-2*half//3)),
                                                     np.float32), half//3), (0, 0), half/4)
            window = cv2.resize(window, (x1-x0, y1-y0))
            density[y0:y1, x0:x1] = np.maximum(patch, smear*1.6*window)
        return density

    def _wobble_field(self, rng):
        """Displacement (mm) on a 1 mm grid over the page, scaled to the sampled RMS."""
        w, h = self.artwork.page_mm
        shape = (int(math.ceil(h))+1, int(math.ceil(w))+1)
        corr = self.params["wobble_corr_mm"]
        field = np.stack([smooth_noise(rng, shape, corr), smooth_noise(rng, shape, corr)], -1)
        field *= self.params["wobble_mm"]/max(float(np.sqrt((field**2).sum(-1).mean())), 1e-6)
        field[~np.isfinite(field)] = 0
        self.wobble = field.astype(np.float32)

    def displacement(self, p_mm):
        return sample_points(self.wobble, p_mm, cv2.BORDER_REPLICATE).astype(float)

    def mirror(self, p_mm):
        p = np.asarray(p_mm, float).reshape(-1, 2).copy()
        if self.params["mirrored"]:
            p[:, 0] = self.artwork.page_mm[0]-p[:, 0]
        return p

    def to_skin(self, p_mm):
        """Where artwork point p landed on the skin (page mm): the truth."""
        return self.mirror(p_mm)+self.displacement(p_mm)

    def to_artwork(self, s_mm):
        """Inverse of to_skin by fixed-point iteration (wobble is smooth and sub-2 mm)."""
        s = np.asarray(s_mm, float).reshape(-1, 2)
        p = self.mirror(s)
        for _ in range(3):
            p = self.mirror(s-self.displacement(p))
        return p

    def sample(self, s_mm, blur_px=0.):
        """Ink density at skin points `s_mm` (N,2), prefiltered by `blur_px` artwork pixels."""
        density = cv2.GaussianBlur(self.density, (0, 0), blur_px) if blur_px > .3 else self.density
        return sample_points(density, self.to_artwork(s_mm)*self.artwork.ppm-.5)


# --- scenes ----------------------------------------------------------------------------------

def scene_rng(bank, index, salt="scene"):
    digest = hashlib.sha256(f"{bank}:{index}:{salt}".encode()).digest()
    return np.random.default_rng([BANKS[bank], int.from_bytes(digest[:8], "big")])


def scene_kind(index):
    """Every eighth scene is blank skin and every eighth another print (negatives)."""
    return {3: "blank", 7: "other_print"}.get(index % 8, "positive")


CLEAN_TRANSFER = {"transmittance_rgb": [.25, .25, .3], "density": 1.0, "spread_factor": 1.0, "edge_blur_mm": .05,
                  "washoff": 0.0, "speckle": 0.0, "fade": 0.0, "pools": 0.0, "smudges": 0.0,
                  "wobble_mm": 0.0, "mirrored": False}


def sample_params(bank, index, cameras, camera_mix="mix", degradation="full"):
    """Deterministic scene parameters for scene `index` of `bank` (JSON-serialisable)."""
    rng = scene_rng(bank, index)
    wrist = [c for c in cameras if c.role == "wrist"]
    overhead = [c for c in cameras if c.role == "overhead"]
    role = {"wrist": "wrist", "overhead": "overhead"}.get(camera_mix) or ("wrist" if rng.random() < .6 else "overhead")
    pool = wrist if role == "wrist" else overhead
    if role == "wrist":
        weights = np.array([p[2] for p in WRIST_PROFILES][:len(pool)])
        camera = pool[int(rng.choice(len(pool), p=weights/weights.sum()))]
    else:
        camera = pool[int(rng.integers(len(pool)))]
    flat = rng.random() < .4
    color = "violet" if rng.random() < .5 else "light-blue"
    skins, weights = zip(*SKIN_TONES, strict=True)
    skin = skins[int(rng.choice(len(skins), p=np.array(weights)/sum(weights)))]
    params = {
        "bank": bank, "index": index, "kind": scene_kind(index), "seed": int(rng.integers(2**31)),
        "camera": camera.name, "role": role,
        "surface": {"radius_mm": None if flat else float(rng.uniform(30, 60)),
                    "axis_deg": float(rng.normal(0, 8) + (90 if rng.random() < .15 else 0)),
                    "crest_offset_mm": float(rng.uniform(-12, 12))},
        "transfer": {"color": color, "transmittance_rgb": list(INKS[color]),
                     "density": float(rng.uniform(.65, 1.15) if color == "violet" else rng.uniform(.5, 1.0)),
                     "skin_rgb": list(skin), "spread_factor": float(rng.uniform(1.5, 2.4)),
                     "edge_blur_mm": float(rng.uniform(.05, .15)), "washoff": float(rng.uniform(0, .6)),
                     "washoff_corr_mm": float(rng.uniform(4, 12)), "speckle": float(rng.uniform(0, .12)),
                     "fade": float(rng.uniform(.1, .35)), "pools": float(rng.uniform(3, 12)),
                     "smudges": float(rng.uniform(0, 3)), "wobble_mm": float(rng.uniform(.5, 1.5)),
                     "wobble_corr_mm": float(rng.uniform(12, 30)), "mirrored": bool(rng.random() < .2)},
        "imaging": {"exposure": float(rng.uniform(.75, 1.1)), "gradient": float(rng.uniform(-.3, .3)),
                    "gradient_deg": float(rng.uniform(0, 360)), "ambient": float(rng.uniform(.35, .7)),
                    "blur_px": float(rng.uniform(.3, 1.2)),
                    "motion_px": int(rng.choice([0, 0, 0, 3, 5])), "motion_deg": float(rng.uniform(0, 180)),
                    "noise": float(rng.uniform(1, 4) if role == "overhead" else rng.uniform(2, 5)),
                    "jpeg": int(rng.integers(55, 86) if role == "overhead" else rng.integers(80, 99))},
    }
    params["pose"] = _sample_pose(rng, camera, params)
    if degradation == "clean":
        # Control: the same scenes with a crisp dark transfer, to separate transfer damage from view.
        params["transfer"].update(CLEAN_TRANSFER, color="clean")
    params["degradation"] = degradation
    return params


def _look_at(camera_position, target, roll):
    z = target-camera_position
    z /= np.linalg.norm(z)
    helper = np.array([0., 0., 1.]) if abs(z[2]) < .9 else np.array([1., 0., 0.])
    x = np.cross(helper, z)
    x /= np.linalg.norm(x)
    y = np.cross(z, x)
    c, s = math.cos(roll), math.sin(roll)
    x, y = c*x+s*y, -s*x+c*y
    return np.stack([x, y, z], 1)   # columns: camera axes in the surface frame


def _sample_pose(rng, camera, params):
    """surface_from_camera (4x4, mm) plus the sampled pose variables."""
    target_uv = rng.uniform(0, 1, 2)
    if camera.role == "wrist":
        distance = float(rng.uniform(100, 200))
        tilt = math.radians(rng.uniform(0, 35))
        azimuth = rng.uniform(0, 2*math.pi)
        roll = rng.uniform(0, 2*math.pi)
        return {"kind": "wrist", "target_uv": target_uv.tolist(), "distance_mm": distance,
                "tilt_deg": math.degrees(tilt), "azimuth_deg": math.degrees(azimuth),
                "roll_deg": math.degrees(roll)}
    yaw = rng.uniform(0, 2*math.pi)
    shift = rng.uniform(-30, 30, 2)
    return {"kind": "overhead", "page_yaw_deg": math.degrees(yaw), "page_shift_mm": shift.tolist(),
            "target_uv": [.5, .5]}


class Scene:
    """One rendered scene and its dense truth. Construct from `sample_params` output."""

    def __init__(self, params, artwork, camera, *, distractor=None):
        self.params, self.camera = params, camera
        self.truth_artwork = distractor if params["kind"] == "other_print" else artwork
        self.artwork = artwork
        rng = np.random.default_rng(params["seed"])
        s = params["surface"]
        self.surface = Surface(self.truth_artwork.page_mm, s["radius_mm"], s["axis_deg"], s["crest_offset_mm"])
        self.transfer = (Transfer(self.truth_artwork, params["transfer"], rng)
                         if params["kind"] != "blank" else None)
        self.rotation, self.origin = self._pose()
        self._rng = rng

    # geometry -----------------------------------------------------------------
    def _pose(self):
        pose = self.params["pose"]
        page = np.array(self.truth_artwork.page_mm)
        if pose["kind"] == "wrist":
            target_s = np.array(pose["target_uv"])*page
            point, normal = self.surface.point(target_s)
            point, normal = point[0], normal[0]
            tilt, azimuth = math.radians(pose["tilt_deg"]), math.radians(pose["azimuth_deg"])
            helper = np.array([1., 0, 0]) if abs(normal[0]) < .9 else np.array([0, 1., 0])
            e1 = np.cross(normal, helper)
            e1 /= np.linalg.norm(e1)
            e2 = np.cross(normal, e1)
            direction = math.cos(tilt)*normal+math.sin(tilt)*(math.cos(azimuth)*e1+math.sin(azimuth)*e2)
            origin = point+pose["distance_mm"]*direction
            return _look_at(origin, point, math.radians(pose["roll_deg"])), origin
        # Overhead: the page lies on the table at the drawing pivot with a sampled yaw and shift.
        base_from_camera = self.camera.surface_from_camera_candidates[0]
        yaw = math.radians(pose["page_yaw_deg"])
        base_from_surface = np.eye(4)
        x = np.array([math.cos(yaw), math.sin(yaw), 0.])
        z = np.array([0., 0., -1.])
        base_from_surface[:3, :3] = np.stack([x, np.cross(z, x), z], 1)
        base_from_surface[:2, 3] = pose["page_shift_mm"]
        surface_from_camera = np.linalg.inv(base_from_surface) @ base_from_camera
        return surface_from_camera[:3, :3], surface_from_camera[:3, 3]

    def _normalized(self, pixels):
        """Undistorted normalised coordinates of `pixels`: Newton on the radius, then a few
        fixed-point steps for the tangential terms. NaN outside the model's monotonic range."""
        fx, fy, cx, cy = self.camera.k
        xd, yd = (pixels[:, 0]-cx)/fx, (pixels[:, 1]-cy)/fy
        if not any(self.camera.dist):
            return np.c_[xd, yd]
        k1, k2, p1, p2, k3 = self.camera.dist
        rd = np.hypot(xd, yd)
        r = rd.copy()
        for _ in range(30):
            r2 = r*r
            f = 1+k1*r2+k2*r2*r2+k3*r2**3
            slope = f+2*r2*(k1+2*k2*r2+3*k3*r2*r2)
            r = np.clip(r-(r*f-rd)/np.where(np.abs(slope) > 1e-6, slope, 1e-6), 0, None)
        scale = np.where(rd > 1e-12, r/np.maximum(rd, 1e-12), 1.)
        x, y = xd*scale, yd*scale
        for _ in range(5):
            r2 = x*x+y*y
            radial = 1+k1*r2+k2*r2*r2+k3*r2**3
            x = (xd-2*p1*x*y-p2*(r2+2*x*x))/radial
            y = (yd-p1*(r2+2*y*y)-2*p2*x*y)/radial
        r2 = x*x+y*y
        radial = 1+k1*r2+k2*r2*r2+k3*r2**3
        residual = np.hypot(x*radial+2*p1*x*y+p2*(r2+2*x*x)-xd, y*radial+p1*(r2+2*y*y)+2*p2*x*y-yd)
        bad = (residual*fx > .05) | (r2 > self._max_r2())
        x[bad], y[bad] = np.nan, np.nan
        return np.c_[x, y]

    def _max_r2(self):
        """Squared radius where the radial model stops increasing (it folds beyond)."""
        k1, k2, _, _, k3 = self.camera.dist
        r = np.linspace(0, 3, 3001)
        r2 = r*r
        slope = 1+3*k1*r2+5*k2*r2*r2+7*k3*r2**3
        folds = np.flatnonzero(slope <= 0)
        return float(r2[folds[0]-1]) if len(folds) else 9.

    def rays(self, pixels):
        n = self._normalized(np.asarray(pixels, float).reshape(-1, 2))
        directions = np.c_[n, np.ones(len(n))] @ self.rotation.T
        return directions/np.linalg.norm(directions, axis=1, keepdims=True)

    def skin_mask(self, s, hit):
        page = np.array(self.truth_artwork.page_mm)
        if self.surface.radius is None:
            inside = np.all((s > -PAD_MARGIN_MM) & (s < page+PAD_MARGIN_MM), axis=1)
        else:
            along = self.surface._wrap_coords(s)[:, 1]
            inside = np.abs(along) < page.max()/2+2*PAD_MARGIN_MM
        return hit & inside

    def backproject(self, pixels):
        """Skin page mm seen at `pixels` (NaN where the ray misses skin)."""
        s, _, _, hit = self.surface.intersect(self.origin, self.rays(pixels))
        s = s.copy()
        s[~self.skin_mask(s, hit)] = np.nan
        return s

    def project(self, s_mm):
        """Image pixels of skin points `s_mm` and whether each is visible (front-facing, in image)."""
        point, normal = self.surface.point(s_mm)
        camera = (point-self.origin) @ self.rotation
        facing = np.einsum("ij,ij->i", normal, self.origin-point) > 0
        z = camera[:, 2]
        with np.errstate(divide="ignore", invalid="ignore"):
            x, y = camera[:, 0]/z, camera[:, 1]/z
        k1, k2, p1, p2, k3 = self.camera.dist
        r2 = x*x+y*y
        radial = 1+k1*r2+k2*r2*r2+k3*r2**3
        xd = x*radial+2*p1*x*y+p2*(r2+2*x*x)
        yd = y*radial+p1*(r2+2*y*y)+2*p2*x*y
        fx, fy, cx, cy = self.camera.k
        pixels = np.c_[fx*xd+cx, fy*yd+cy]
        inside = ((pixels[:, 0] >= 0) & (pixels[:, 0] <= self.camera.width-1)
                  & (pixels[:, 1] >= 0) & (pixels[:, 1] <= self.camera.height-1))
        # Beyond the model's monotonic range distortion folds points back into the image.
        visible = facing & (z > 1) & inside & (r2 < self._max_r2())
        return pixels, visible

    def truth_skin_mm(self, uv):
        """Where page-UV point `uv` of the truth artwork landed on the skin (mm)."""
        return self.transfer.to_skin(np.asarray(uv, float).reshape(-1, 2)*np.array(self.truth_artwork.page_mm))

    def truth_pixels(self, uv):
        s = self.truth_skin_mm(uv)
        pixels, visible = self.project(s)
        return pixels, visible

    def px_per_mm(self):
        uv = np.array(self.params["pose"]["target_uv"])
        s = uv*np.array(self.truth_artwork.page_mm)
        pixels, _ = self.project(np.array([s, s+[1, 0], s+[0, 1]]))
        return float(np.sqrt(abs(cross2(pixels[1]-pixels[0], pixels[2]-pixels[0]))))

    # rendering ----------------------------------------------------------------
    def render(self):
        w, h = self.camera.width, self.camera.height
        ys, xs = np.mgrid[0:h, 0:w]
        pixels = np.c_[xs.ravel(), ys.ravel()].astype(np.float32)
        directions = self._ray_grid(w, h)
        s, point, normal, hit = self.surface.intersect(self.origin, directions)
        skin = self.skin_mask(s, hit)
        image = self._table(directions, skin)
        colour = self._skin(s[skin], normal[skin], point[skin])
        image[skin] = colour
        image = image.reshape(h, w, 3)
        return self._imaging(image, pixels)

    def _ray_grid(self, w, h):
        """Per-pixel rays; distortion is inverted on a coarse grid and interpolated (sub-0.01 px)."""
        step = 8 if any(self.camera.dist) else 1
        gx, gy = np.arange(0, w+step, step, dtype=np.float64), np.arange(0, h+step, step, dtype=np.float64)
        mesh = np.stack(np.meshgrid(gx, gy), -1).reshape(-1, 2)
        coarse = self._normalized(mesh).reshape(len(gy), len(gx), 2).astype(np.float32)
        ys, xs = np.mgrid[0:h, 0:w].astype(np.float32)/step
        coarse = cv2.remap(coarse, xs, ys, cv2.INTER_LINEAR)
        n = coarse.reshape(-1, 2).astype(np.float64)
        directions = np.c_[n, np.ones(len(n))] @ self.rotation.T
        return directions/np.linalg.norm(directions, axis=1, keepdims=True)

    def _table(self, directions, skin):
        """Cutting-mat table (10 mm grid, 50 mm major lines, clutter blobs) under the skin."""
        with np.errstate(divide="ignore", invalid="ignore"):
            t = (self.surface.table_z()-self.origin[2])/directions[:, 2]
        hit = np.isfinite(t) & (t > 0) & ~skin
        image = np.full((len(directions), 3), 28., np.float32)
        xy = self.origin[:2]+t[hit, None]*directions[hit, :2]
        texture, ppm, extent = self._table_texture()
        value = sample_points(texture, (xy+extent)*ppm-.5, cv2.BORDER_CONSTANT, 28.)
        image[hit] = value[:, None]*np.array([1., 1.02, .98], np.float32)
        return image

    def _table_texture(self, ppm=3., extent=450.):
        rng = np.random.default_rng(self.params["seed"]+1)
        size = int(2*extent*ppm)
        texture = np.full((size, size), rng.uniform(55, 95), np.float32)
        for pitch, value, width in ((10, 35, 1), (50, 45, 2)):
            for k in np.arange(-extent, extent+1, pitch):
                pixel = int(round((k+extent)*ppm))
                cv2.line(texture, (pixel, 0), (pixel, size-1), float(texture[0, 0]+value), width)
                cv2.line(texture, (0, pixel), (size-1, pixel), float(texture[0, 0]+value), width)
        for _ in range(int(rng.integers(4, 12))):
            centre = ((rng.uniform(-250, 250, 2)+extent)*ppm).astype(int)
            axes = (rng.uniform(8, 40, 2)*ppm).astype(int)
            cv2.ellipse(texture, tuple(int(v) for v in centre), tuple(int(v) for v in axes),
                        float(rng.uniform(0, 180)), 0, 360, float(rng.uniform(15, 230)), -1)
        return cv2.GaussianBlur(texture, (0, 0), .7), ppm, extent

    def _skin(self, s, normal, point):
        t = self.params["transfer"]
        imaging = self.params["imaging"]
        rng = np.random.default_rng(self.params["seed"]+2)
        albedo = np.array(t["skin_rgb"][::-1], np.float32)  # BGR
        # Skin texture: mottling (~6 mm) and pores (~0.4 mm), in page mm so they scale with view.
        grid = 4.
        span = np.array(self.truth_artwork.page_mm)+2*PAD_MARGIN_MM
        shape = (int(span[1]*grid), int(span[0]*grid))
        mottle = smooth_noise(rng, shape, 6*grid)*.02+smooth_noise(rng, shape, .5*grid)*.015
        texture = sample_points(mottle, (s+PAD_MARGIN_MM)*grid-.5, cv2.BORDER_REFLECT)
        colour = albedo[None]*(1+texture[:, None])
        if self.transfer is not None:
            ratio = self.transfer.artwork.ppm/max(self.px_per_mm(), 1e-3)
            density = self.transfer.sample(s, blur_px=.5*ratio)
            absorb = 1-np.array(t["transmittance_rgb"][::-1], np.float32)
            colour *= np.clip(1-density[:, None]*absorb[None], 0, 1)
        light = self.rotation[:, 2]*-1+np.array([.2, -.3, -.6])
        light /= np.linalg.norm(light)
        lambert = np.clip(normal @ light, 0, 1)
        shade = imaging["ambient"]+(1-imaging["ambient"])*lambert
        return colour*shade[:, None]

    def _imaging(self, image, pixels):
        p = self.params["imaging"]
        rng = self._rng
        h, w = image.shape[:2]
        angle = math.radians(p["gradient_deg"])
        ramp = ((pixels[:, 0]-w/2)*math.cos(angle)+(pixels[:, 1]-h/2)*math.sin(angle))/math.hypot(w, h)
        image = image*(p["exposure"]*(1+2*p["gradient"]*ramp)).reshape(h, w, 1)
        image = cv2.GaussianBlur(image, (0, 0), p["blur_px"])
        if p["motion_px"]:
            kernel = np.zeros((p["motion_px"], p["motion_px"]), np.float32)
            kernel[p["motion_px"]//2, :] = 1
            rot = cv2.getRotationMatrix2D(((p["motion_px"]-1)/2, (p["motion_px"]-1)/2), p["motion_deg"], 1)
            kernel = cv2.warpAffine(kernel, rot, kernel.shape[::-1])
            image = cv2.filter2D(image, -1, kernel/max(kernel.sum(), 1e-6))
        image = image+rng.normal(0, p["noise"], image.shape).astype(np.float32)
        image = np.clip(image, 0, 255).astype(np.uint8)
        ok, encoded = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, p["jpeg"]])
        if not ok:
            raise RuntimeError("JPEG encoding failed")
        return cv2.imdecode(encoded, cv2.IMREAD_COLOR)

    # truth summaries ----------------------------------------------------------
    def frame_cells(self, pitch_mm=3.5):
        """Frame-band cell centres (uv of the truth artwork) and which are visible in the image."""
        art = self.truth_artwork
        w, h = art.page_mm
        xs = np.arange(art.margin_mm+pitch_mm/2, w-art.margin_mm, pitch_mm)
        ys = np.arange(art.margin_mm+pitch_mm/2, h-art.margin_mm, pitch_mm)
        grid = np.stack(np.meshgrid(xs, ys), -1).reshape(-1, 2)
        grid = grid[art.in_frame(grid)]
        uv = grid/np.array(art.page_mm)
        if self.transfer is None:
            return uv, np.zeros(len(uv), bool)
        s = self.truth_skin_mm(uv)
        pixels, visible = self.project(s)
        seen = self.backproject(pixels[visible])
        ok = np.zeros(len(uv), bool)
        ok[np.flatnonzero(visible)] = np.linalg.norm(seen-s[visible], axis=1) < .5
        return uv, ok

    def summary(self):
        return {"washoff_measured": None if self.transfer is None else round(self.transfer.washed_fraction, 4),
                "px_per_mm": round(self.px_per_mm(), 3)}
