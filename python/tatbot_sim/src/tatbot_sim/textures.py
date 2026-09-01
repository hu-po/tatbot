"""Procedural grid paper for the drawing surface — texture, geometry, metadata.

The real sessions draw on ruled paper, so a blank pad leaves the wrist views
without the structure the policy actually sees. A handful of sheets are
generated once into the ManiSkill asset directory and dealt out across
environments, with tint, pitch and rule colour varying between them.

Each sheet is three files that travel together:

- ``grid_NN.png`` — the ruled paper image;
- ``grid_NN.obj``/``.mtl`` — a UV'd quad the size of the sheet. The pad's box
  primitive cannot carry the texture: SAPIEN wraps box UVs around all six
  faces (verified with a striped probe — the top face samples an interior
  region, not 0..1), so the top face is an explicit quad whose UVs we control.
- ``grid_NN.json`` — the ruling's geometry in millimetres, so the maze
  generator can put strokes ON the printed lines rather than near them.

The ruling is a single line weight on an exact 6 mm pitch (the operator's
paper — one grid size, no major/minor distinction). 6 mm is not an integer
pixel count at this resolution, so lines are drawn anti-aliased at true
millimetre positions and the metadata is exact by construction; only the
grid's phase (where the first line falls), the rule colour/strength, paper
tint and grain vary between sheets.

The pixel↔canvas mapping is fixed by the quad's UVs (verified by probe
render): texture column 0 sits at canvas x = −W/2 with +u along +x, and
texture row 0 sits at canvas y = −H/2 with rows running along +y. A line at
pixel index k therefore lies at canvas ``(k + 0.5) * mm_per_px − half``.

Kept deliberately small: the scene is a pad, ink dots and lights — resist
growing a procedural world here.
"""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
from mani_skill import ASSET_DIR

TEX_DIR = Path(ASSET_DIR) / "robots/widowxai/tatbot_textures"
# Letter sheet (216 x 279.4 mm); the env sizes its pad from these so texture
# and geometry cannot drift apart. ~2.4 px/mm on both axes.
SHEET_W_M = 0.2159
SHEET_H_M = 0.2794
SIZE_X = 512
SIZE_Y = 662
GRID_PITCH_M = 0.006  # the operator's paper: a single 6 mm grid


def _line_profile(n_px: int, span_m: float, offset_m: float, half_w_m: float) -> np.ndarray:
    """(n_px,) line coverage in [0,1]: anti-aliased rules at exact mm positions."""
    mm_per_px = span_m / n_px
    centers_m = (np.arange(n_px) + 0.5) * mm_per_px
    phase = (centers_m - offset_m) % GRID_PITCH_M
    dist = np.minimum(phase, GRID_PITCH_M - phase)
    return np.clip(1.0 - dist / half_w_m, 0.0, 1.0)


def _make_sheet(rng: np.random.Generator, path: Path) -> dict:
    paper = rng.uniform([0.86, 0.86, 0.83], [0.99, 0.99, 0.97])
    rule = rng.uniform([0.45, 0.55, 0.62], [0.70, 0.78, 0.85])
    img = np.ones((SIZE_Y, SIZE_X, 3)) * paper

    strength = rng.uniform(0.3, 0.7)
    half_w = rng.uniform(0.00025, 0.0005)  # rule half-width, m (0.5-1 mm lines)
    off_x = float(rng.uniform(0, GRID_PITCH_M))
    off_y = float(rng.uniform(0, GRID_PITCH_M))
    cov_x = _line_profile(SIZE_X, SHEET_W_M, off_x, half_w) * strength  # columns
    cov_y = _line_profile(SIZE_Y, SHEET_H_M, off_y, half_w) * strength  # rows
    cov = np.maximum(cov_y[:, None], cov_x[None, :])
    img = img * (1 - cov[..., None]) + rule * cov[..., None]

    img += rng.normal(0, 0.006, img.shape)  # grain, so it is not perfectly flat
    cv2.imwrite(str(path), np.clip(img[..., ::-1] * 255, 0, 255).astype(np.uint8))
    return {"pitch_m": GRID_PITCH_M, "offset_x_m": off_x, "offset_y_m": off_y}


def _make_skin(rng: np.random.Generator, path: Path, sub, size_x: int, size_y: int) -> dict:
    """A blank silicone practice skin: pink, faintly mottled, nothing printed.

    There is no ruling to trace, which is why a skin episode's target is ink
    that is already on it rather than a printed line. What varies is the batch
    variation a box of practice skins actually has -- tone, saturation, and a
    slow mottle from the casting -- plus a fine grain so the wrist views have
    something to lock onto on an otherwise featureless surface.
    """
    base = np.array([float(v) for v in (sub.base_color or "0.91 0.71 0.65").split()])
    base = base * rng.uniform(0.93, 1.06) + rng.uniform(-0.02, 0.02, 3)
    # desaturate or deepen the whole sheet a little, around its own grey
    grey = float(base.mean())
    base = grey + (base - grey) * rng.uniform(0.85, 1.15)

    # slow mottle: a few low-frequency waves, the casting's unevenness
    ys = np.linspace(0, 1, size_y)[:, None]
    xs = np.linspace(0, 1, size_x)[None, :]
    mottle = np.zeros((size_y, size_x))
    for _ in range(4):
        fx, fy = rng.uniform(1.0, 3.5, 2)
        ph = rng.uniform(0, 2 * np.pi, 2)
        mottle += np.sin(2 * np.pi * fx * xs + ph[0]) * np.sin(2 * np.pi * fy * ys + ph[1])
    mottle *= rng.uniform(0.006, 0.018) / 4.0

    img = np.clip(base[None, None, :] * (1.0 + mottle[..., None]), 0, 1)
    img += rng.normal(0, rng.uniform(0.004, 0.009), img.shape)  # grain
    cv2.imwrite(str(path), np.clip(img[..., ::-1] * 255, 0, 255).astype(np.uint8))
    return {"base_color": base.tolist()}


def skin_sheets(count: int, sub, seed: int = 0) -> list[dict]:
    """Ensure ``count`` silicone skins exist and return their metadata.

    Same shape of entry as the paper sheets, so the planner does not care which
    substrate it is laying strokes on. ``xs``/``ys`` are a placement LATTICE,
    not a ruling: motifs that want to sit on a grid still need somewhere to sit
    when nothing is printed, and on a skin their absolute position only has to
    look deliberate, since the episode removes what it finds rather than
    tracing a line that is drawn there.
    """
    TEX_DIR.mkdir(parents=True, exist_ok=True)
    size_x, size_y = int(sub.texel_cols), int(sub.texel_rows)
    sheets = []
    for i in range(count):
        rng = np.random.default_rng((seed << 8) + 4096 + i)
        stem = TEX_DIR / f"skin_{sub.name}_{i:02d}"
        png = stem.with_suffix(".png")
        stale = png.exists() and cv2.imread(str(png)).shape[:2] != (size_y, size_x)
        if not png.exists() or stale:
            _make_skin(rng, png, sub, size_x, size_y)
        if not stem.with_suffix(".obj").exists() or stale:
            _write_quad(stem, sub.width_m, sub.height_m)
        sheets.append({
            "png": str(png),
            "obj": str(stem.with_suffix(".obj")),
            "pitch_m": GRID_PITCH_M,
            "xs": _line_coords_m(float(rng.uniform(0, GRID_PITCH_M)), sub.width_m),
            "ys": _line_coords_m(float(rng.uniform(0, GRID_PITCH_M)), sub.height_m),
            "ruled": False,
        })
    return sheets


def _write_quad(stem: Path, width_m: float = SHEET_W_M, height_m: float = SHEET_H_M):
    """UV'd quad for the sheet's top face; see module docstring for mapping."""
    hx, hy = width_m / 2, height_m / 2
    stem.with_suffix(".mtl").write_text(
        f"newmtl paper\nKd 1 1 1\nmap_Kd {stem.name}.png\n"
    )
    # rows run along +y and image row 0 is the TOP of the png, which OBJ
    # convention puts at v=1 — hence v = 1 at y = -hy.
    stem.with_suffix(".obj").write_text(
        f"mtllib {stem.name}.mtl\n"
        f"v {-hx} {-hy} 0\nv {hx} {-hy} 0\nv {hx} {hy} 0\nv {-hx} {hy} 0\n"
        "vt 0 1\nvt 1 1\nvt 1 0\nvt 0 0\n"
        "vn 0 0 1\n"
        "usemtl paper\n"
        "f 1/1/1 2/2/1 3/3/1\nf 1/1/1 3/3/1 4/4/1\n"
    )


def write_surface_mesh(stem: Path, mtl_stem: str, verts, normals, rows: int, cols: int,
                       thickness_m: float | None = None) -> str:
    """Write a UV'd, smooth-shaded grid mesh for a shaped sheet; return its path.

    ``verts`` and ``normals`` are (rows*cols, 3) in the pad's local frame, laid
    out row-major with rows running along +y — the same order ``canvas_to_px``
    uses, so the UVs below are the flat quad's mapping unchanged and the ruling
    lands where it always did.

    The caller computes them from the Surface itself rather than from a height
    formula here, which is what keeps the picture and the ink model the same
    shape: a mesh derived independently would be free to drift.

    With ``thickness_m`` the sheet becomes a SOLID: the same surface offset
    down by its thickness, plus a wall around the rim. That is what a shaped
    substrate needs — a slab whose top is a 25 mm mound and whose body is a
    flat box would show its box through the mound, and a separate body
    modelled underneath is more scene than the shape is worth.
    """
    us = np.arange(cols) / (cols - 1)
    vs = 1.0 - np.arange(rows) / (rows - 1)  # v=1 at y=-hy, as the quad had it
    uv = np.stack(np.meshgrid(vs, us, indexing="ij")[::-1], axis=-1).reshape(-1, 2)

    i, j = np.meshgrid(np.arange(rows - 1), np.arange(cols - 1), indexing="ij")
    a = (i * cols + j + 1).ravel()  # OBJ is 1-indexed
    b, c, d = a + 1, a + cols + 1, a + cols
    tri = np.concatenate([np.stack([a, b, c], 1), np.stack([a, c, d], 1)], 0)

    if thickness_m:
        n = rows * cols
        under = np.asarray(verts, dtype=np.float64).copy()
        under[:, 2] -= float(thickness_m)
        verts = np.concatenate([np.asarray(verts, dtype=np.float64), under], 0)
        normals = np.concatenate([np.asarray(normals), -np.asarray(normals)], 0)
        uv = np.concatenate([uv, uv], 0)
        # the underside, wound the other way so it faces down
        tri = np.concatenate([tri, np.stack([tri[:, 0], tri[:, 2], tri[:, 1]], 1) + n], 0)
        # and a wall around the rim, joining the two skins edge for edge
        rim = np.concatenate([
            np.arange(cols),                                   # front row
            np.arange(cols - 1, rows * cols, cols),            # right column
            np.arange(rows * cols - 1, (rows - 1) * cols - 1, -1),   # back row
            np.arange((rows - 1) * cols, -1, -cols),           # left column
        ])
        a, b = rim[:-1] + 1, rim[1:] + 1
        tri = np.concatenate([tri, np.stack([a, b, b + n], 1), np.stack([a, b + n, a + n], 1)], 0)

    out = [f"mtllib {mtl_stem}.mtl"]
    out += [f"v {x:.6f} {y:.6f} {z:.6f}" for x, y, z in verts]
    out += [f"vt {u:.6f} {v:.6f}" for u, v in uv]
    out += [f"vn {x:.6f} {y:.6f} {z:.6f}" for x, y, z in normals]
    out.append("usemtl paper")
    out += [f"f {p}/{p}/{p} {q}/{q}/{q} {r}/{r}/{r}" for p, q, r in tri]
    stem.with_suffix(".obj").write_text("\n".join(out) + "\n")
    return str(stem.with_suffix(".obj"))


def _line_coords_m(offset_m: float, span_m: float) -> list[float]:
    """Canvas coordinates (metres, centred) of the rules along one axis."""
    out = []
    x = offset_m
    while x < span_m:
        out.append(x - span_m / 2)
        x += GRID_PITCH_M
    return out


def _rule_px(offset_m: float, span_m: float, n_px: int) -> list[int]:
    """Pixel columns/rows of the printed rules along one axis."""
    out, x = [], offset_m
    while x < span_m:
        out.append(int(round(x / span_m * n_px)))
        x += GRID_PITCH_M
    return out


def _apply_wear(rng: np.random.Generator, img: np.ndarray, meta: dict) -> np.ndarray:
    """A used sheet: ghost strokes ON the ruling (previous drawings, faded),
    smudges, and uneven yellowing. ``img`` is float RGB in [0, 1]."""
    h, w = img.shape[:2]
    # uneven yellowing: a low-frequency field pushing blue down where it dips
    f = cv2.resize(rng.uniform(0.0, 1.0, (4, 4)), (w, h), interpolation=cv2.INTER_CUBIC)
    depth = rng.uniform(0.0, 0.12)
    img = img * (1 - f[..., None] * depth * np.array([0.1, 0.35, 1.0]))
    # ghost strokes: short lattice walks in faded grey, like erased/old ink
    xs = _rule_px(meta["offset_x_m"], SHEET_W_M, w)
    ys = _rule_px(meta["offset_y_m"], SHEET_H_M, h)
    overlay = img.copy()
    for _ in range(int(rng.integers(1, 4))):
        i, j = int(rng.integers(1, len(xs) - 1)), int(rng.integers(1, len(ys) - 1))
        pts = [(xs[i], ys[j])]
        for _seg in range(int(rng.integers(3, 9))):
            di, dj = ((1, 0), (-1, 0), (0, 1), (0, -1))[int(rng.integers(4))]
            run = int(rng.integers(1, 4))
            i = int(np.clip(i + di * run, 0, len(xs) - 1))
            j = int(np.clip(j + dj * run, 0, len(ys) - 1))
            pts.append((xs[i], ys[j]))
        grey = float(rng.uniform(0.45, 0.75))
        cv2.polylines(overlay, [np.array(pts)], False, (grey, grey, grey),
                      int(rng.integers(2, 5)), lineType=cv2.LINE_AA)
    img = img + (overlay - img) * rng.uniform(0.25, 0.6)
    # smudges: soft dark ellipses, barely-there
    for _ in range(int(rng.integers(0, 4))):
        mask = np.zeros((h, w), np.float32)
        c = (int(rng.integers(0, w)), int(rng.integers(0, h)))
        axes = (int(rng.integers(10, 60)), int(rng.integers(6, 30)))
        cv2.ellipse(mask, c, axes, float(rng.uniform(0, 180)), 0, 360, 1.0, -1)
        mask = cv2.GaussianBlur(mask, (0, 0), 9)
        img = img * (1 - mask[..., None] * rng.uniform(0.04, 0.12))
    return np.clip(img, 0, 1)


def grid_paper_sheets(count: int, seed: int = 0, wear_variants: int = 0) -> list[dict]:
    """Ensure ``count`` sheets exist and return their metadata.

    Each dict: ``png``, ``obj`` (paths), ``xs``/``ys`` (rule canvas
    coordinates in metres along x and y), ``pitch_m``. With
    ``wear_variants`` > 0, each base sheet also contributes that many worn
    variants (same ruling geometry, used-sheet texture) to the pool; the
    pristine base sheets always stay in it.
    """
    TEX_DIR.mkdir(parents=True, exist_ok=True)
    sheets = []
    for i in range(count):
        # per-sheet stream: params stay stable no matter which files exist
        rng = np.random.default_rng((seed << 8) + i)
        stem = TEX_DIR / f"grid_{i:02d}"
        png, meta_p = stem.with_suffix(".png"), stem.with_suffix(".json")
        meta = json.loads(meta_p.read_text()) if meta_p.exists() else {}
        stale = (
            (png.exists() and cv2.imread(str(png)).shape[:2] != (SIZE_Y, SIZE_X))
            or "pitch_m" not in meta  # pre-6mm-ruling sheet
        )
        if not png.exists() or stale:
            meta = _make_sheet(rng, png)
            meta_p.write_text(json.dumps(meta))
        if not stem.with_suffix(".obj").exists():
            _write_quad(stem)
        entry = {
            "png": str(png),
            "obj": str(stem.with_suffix(".obj")),
            "pitch_m": meta["pitch_m"],
            # columns run along canvas x, rows along canvas y (probe-verified)
            "xs": _line_coords_m(meta["offset_x_m"], SHEET_W_M),
            "ys": _line_coords_m(meta["offset_y_m"], SHEET_H_M),
        }
        sheets.append(entry)
        for v in range(1, wear_variants + 1):
            wstem = TEX_DIR / f"grid_{i:02d}_w{v}"
            wpng = wstem.with_suffix(".png")
            if not wpng.exists() or stale:
                wrng = np.random.default_rng((seed << 8) + i * 97 + v)
                base = cv2.imread(str(png))[..., ::-1].astype(np.float64) / 255.0
                worn = _apply_wear(wrng, base, meta)
                cv2.imwrite(str(wpng), (worn[..., ::-1] * 255).astype(np.uint8))
            if not wstem.with_suffix(".obj").exists():
                _write_quad(wstem)
            sheets.append({**entry, "png": str(wpng), "obj": str(wstem.with_suffix(".obj"))})
    return sheets



# --- backgrounds: procedural environment cube maps and floor textures -------
# No downloaded HDRIs: six generated faces per set give the same two things an
# HDRI gives the renderer — image-based ambience and a non-void background —
# from a palette we control. Faces are LDR PNGs; SAPIEN accepts them directly.

ENV_FACE_PX = 256


def _env_face(rng, palette, kind: str) -> "np.ndarray":
    top, horizon, ground = palette
    n = ENV_FACE_PX
    v = np.linspace(0, 1, n)[:, None, None]  # 0 = top of image
    if kind == "up":
        img = np.ones((n, n, 3)) * top
    elif kind == "down":
        img = np.ones((n, n, 3)) * ground
    else:  # side: sky over ground with a soft horizon band
        img = top * (1 - v) + horizon * np.clip(1 - np.abs(v - 0.55) * 4, 0, 1) * 0.5
        img = np.where(v < 0.55, img, ground * (v * 0.6 + 0.4))
    # large soft blobs: fake clutter/windows/shadows so the background is not flat
    for _ in range(int(rng.integers(2, 6))):
        cy, cx = rng.integers(0, n, 2)
        r = int(rng.integers(n // 8, n // 2))
        yy, xx = np.ogrid[:n, :n]
        mask = np.clip(1 - ((yy - cy) ** 2 + (xx - cx) ** 2) / (r * r), 0, 1)[..., None]
        img = img * (1 - 0.35 * mask) + rng.uniform(0, 1, 3) * 0.35 * mask
    img += rng.normal(0, 0.01, img.shape)
    return np.clip(img * 255, 0, 255).astype(np.uint8)


def environment_face_sets(count: int, seed: int = 0) -> list[tuple[str, ...]]:
    """``count`` procedural cube maps; each entry is (px,nx,py,ny,pz,nz) paths.

    Overall level varies a lot between maps, because this is the scene's
    ambience floor and a wrist camera's whole dynamic range sits on top of it.
    A set that is uniformly bright lights every crevice from every direction:
    measured against the bench on 2026-08-26 the sim put NO pixel below 0.05
    where the real stream puts 26% of them, and no shadow the lights cast could
    reach the floor this holds up. Dim rooms belong in the draw as much as
    studio panels do.
    """
    out_dir = TEX_DIR / "envmaps"
    out_dir.mkdir(parents=True, exist_ok=True)
    sets = []
    for i in range(count):
        rng = np.random.default_rng((seed << 16) + 7 + i)
        base_h = rng.uniform(0, 1)
        # a dim room is as ordinary as a bright one; skewed low because the
        # lights are what should light the work, not the walls
        dim = float(rng.uniform(0.12, 1.0) ** 1.4)
        def tone(level, sat, base_h=base_h, dim=dim):  # this iteration's hue, not the loop's
            c = np.array([abs(base_h * 6 - 3) - 1, 2 - abs(base_h * 6 - 2), 2 - abs(base_h * 6 - 4)])
            c = np.clip(c, 0, 1)
            return (1 - sat + sat * c) * level * dim
        palette = (tone(rng.uniform(0.2, 0.75), rng.uniform(0.1, 0.7)),
                   tone(rng.uniform(0.25, 0.8), rng.uniform(0.2, 0.8)),
                   tone(rng.uniform(0.05, 0.35), rng.uniform(0.1, 0.5)))
        paths = []
        # sapien slot order px,nx,py,ny,pz,nz; z is up in its world frame
        for slot, kind in (("px", "side"), ("nx", "side"), ("py", "side"),
                           ("ny", "side"), ("pz", "up"), ("nz", "down")):
            # v2 in the name: the cache is keyed by filename and the level
            # range changed, so the old bright faces must not be reused
            f = out_dir / f"env2_{i:02d}_{slot}.png"
            if not f.exists():
                cv2.imwrite(str(f), _env_face(rng, palette, kind)[..., ::-1])
            else:
                rng.integers(0, 2, 8)  # keep the stream aligned
            paths.append(str(f))
        sets.append(tuple(paths))
    return sets


def floor_textures(count: int, seed: int = 0) -> list[str]:
    """Procedural floor/tabletop textures: base tone + one of a few motifs.

    Tone is skewed DARK and reaches near-black, because the surface a wrist
    camera actually sees most of is the bench, and this bench is a black
    self-healing cutting mat. Held above 0.15 -- as these were until the
    2026-08-26 comparison -- no lighting makes a wrist frame's dark tail: the
    real stream puts 26% of its pixels below 0.05 and the sim put none there,
    and most of that gap is the mat's albedo rather than any shadow.
    """
    out_dir = TEX_DIR / "floors"
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for i in range(count):
        # v3: the cache is keyed by name and the tone range changed under it
        f = out_dir / f"floor3_{i:02d}.png"
        paths.append(str(f))
        if f.exists():
            continue
        rng = np.random.default_rng((seed << 16) + 91 + i)
        n = 512
        # cubed draw: about half the set lands under 0.12, because the bench
        # this has to cover is a black mat and a randomiser whose every draw is
        # a mid grey never contains it. The pale tail stays -- a policy that
        # only ever saw a dark bench would be as brittle the other way.
        level = 0.02 + 0.83 * float(rng.uniform(0, 1)) ** 3
        base = level * rng.uniform(0.85, 1.15, 3)   # one tone, faintly tinted
        img = np.ones((n, n, 3)) * base
        motif = rng.integers(0, 3)
        if motif == 0:  # planks
            w = int(rng.integers(24, 90))
            for k in range(0, n, w):
                img[:, k : k + 2] *= 0.6
                img[:, k:k + w] *= rng.uniform(0.85, 1.1)
        elif motif == 1:  # tiles
            w = int(rng.integers(40, 128))
            img[::w, :] *= 0.6
            img[:, ::w] *= 0.6
        img += rng.normal(0, rng.uniform(0.01, 0.05), img.shape)
        cv2.imwrite(str(f), np.clip(img[..., ::-1] * 255, 0, 255).astype(np.uint8))
    return paths
