"""The marker budget: which of a fiducial-tile print's markers a program's ink leaves readable.

A design drawn over the tile covers the markers the overhead and the wrist cameras track the print by. `predict`
draws the program's strokes over the print's stencil.png as the cameras would see them and reads the result with the
stack's own detector (scripts/vision/fiducials), at the overhead's scale after every ink and at the wrist's every
few strokes. `refusals` holds that against the budget (stack.yaml page.markers).

The picture is a model, fitted to recorded frames (page.markers.model in stack.yaml gives each value and what it
was fitted to): the skin and the transfer as flat colours, each stroke a line of its resource's colour, as wide as
its mark plus the ink smeared beside it, seen through a blur at a camera's pixels per millimetre.
Needs OpenCV; the rest of tatbot_ink does not.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import yaml

from tatbot_ink import passes

RENDER_PX_MM = 8.0      # the page is drawn at this, then resampled to a camera's
EDGE_M = 0.008          # bare skin drawn past the print's edge: an edge marker's quad needs skin round it
PLACED = 0.5            # a read counts when its centre is within this share of the marker's side of its printed place


def settings(repo) -> dict:
    """stack.yaml's page.markers: the budget and the model."""
    stack = yaml.safe_load((Path(repo) / "ros/tatbot_bringup/config/stack.yaml").read_text())
    return stack["page"]["markers"]


def _fiducials(repo):
    path = str(Path(repo) / "scripts/vision")
    if path not in sys.path:
        sys.path.insert(0, path)
    from fiducials import stencil

    return stencil


class Print:
    """A fiducial-tile print's folder: its markers (fiducials.stencil.Tile) and its stencil.png."""

    def __init__(self, folder, repo):
        import cv2

        self.reader = _fiducials(repo)
        folder = Path(folder).expanduser()
        self.folder = folder if folder.is_dir() else folder.parent
        self.tile = self.reader.load(self.folder)
        if self.tile is None:
            return
        self.pattern_id = json.loads((self.folder / "tracking.json").read_text()).get("pattern_id", "")
        art = cv2.imread(str(self.folder / "stencil.png"), cv2.IMREAD_GRAYSCALE)
        if art is None:
            raise ValueError(f"{self.folder}: no stencil.png beside tile.json")
        self.size_m = np.asarray(self.tile.size_m)
        shape = tuple(int(round(v * 1000 * RENDER_PX_MM)) for v in self.size_m)
        self.transfer = 1.0 - cv2.resize(art, shape, interpolation=cv2.INTER_AREA).astype(np.float32) / 255.0
        self.centres = {i: c.mean(axis=0) for i, c in self.tile.corners.items()}


class Scene:
    """The print on skin as a camera sees it, the ink laid on it so far."""

    def __init__(self, sheet: Print, model: dict):
        import cv2

        self.sheet, self.model = sheet, model
        pad = int(round(EDGE_M * 1000 * RENDER_PX_MM))
        transfer = cv2.copyMakeBorder(sheet.transfer, pad, pad, pad, pad, cv2.BORDER_CONSTANT, value=0.0)
        skin, violet = (np.asarray(model[key][::-1], np.float32) for key in ("skin_rgb", "transfer_rgb"))   # BGR
        self.image = skin + transfer[..., None] * (violet - skin)
        self.origin = -sheet.size_m / 2 - EDGE_M   # the page point (x, y) of the image's lower-left corner

    def pixels(self, points_m) -> np.ndarray:
        """Page metres -> this image's pixel coordinates (row 0 at the print's top)."""
        p = (np.asarray(points_m, float).reshape(-1, 2) - self.origin) * 1000 * RENDER_PX_MM
        return np.c_[p[:, 0] - 0.5, self.image.shape[0] - p[:, 1] - 0.5]

    def lay(self, points_m, mark_m: float, rgb) -> None:
        """One stroke's ink: its mark and the ink smeared beside it (model ink_bleed_m), at model ink_opacity."""
        import cv2

        width = max(1, int(round((mark_m + self.model["ink_bleed_m"]) * 1000 * RENDER_PX_MM)))
        mask = np.zeros(self.image.shape[:2], np.uint8)
        line = np.round(self.pixels(points_m) * 16).astype(np.int32)
        cv2.polylines(mask, [line], False, 255, width, cv2.LINE_AA, shift=4)
        cover = (mask.astype(np.float32) / 255.0 * self.model["ink_opacity"])[..., None]
        self.image += cover * (np.asarray(rgb[::-1], np.float32) - self.image)

    def lay_area(self, mask, extent_m, rgb) -> None:
        """An inked region: `mask` (rows from the top) covering extent_m = (xmin, ymin, xmax, ymax) in page metres."""
        import cv2

        (left, top), (right, bottom) = self.pixels([[extent_m[0], extent_m[3]], [extent_m[2], extent_m[1]]]) + 0.5
        x0, y0, x1, y1 = (int(round(v)) for v in (left, top, right, bottom))
        patch = cv2.resize(np.asarray(mask, np.float32), (x1 - x0, y1 - y0), interpolation=cv2.INTER_AREA)
        cover = np.zeros(self.image.shape[:2], np.float32)
        height, width = cover.shape
        cx0, cy0, cx1, cy1 = max(x0, 0), max(y0, 0), min(x1, width), min(y1, height)
        cover[cy0:cy1, cx0:cx1] = patch[cy0 - y0:cy1 - y0, cx0 - x0:cx1 - x0]
        cover = (cover * self.model["ink_opacity"])[..., None]
        self.image += cover * (np.asarray(rgb[::-1], np.float32) - self.image)

    def view(self, camera: dict) -> tuple[np.ndarray, float]:
        """(the scene at a camera's px_per_mm through its blur_px, uint8 BGR; its pixels per millimetre)."""
        import cv2

        scale = camera["px_per_mm"] / RENDER_PX_MM
        small = cv2.resize(self.image, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        if camera["blur_px"] > 0:
            small = cv2.GaussianBlur(small, (0, 0), camera["blur_px"])
        return np.clip(np.round(small), 0, 255).astype(np.uint8), camera["px_per_mm"]

    def _placed(self, reads: dict, px_mm: float) -> list[int]:
        """The codes read where they are printed: a false decode stands elsewhere."""
        height = self.image.shape[0] / RENDER_PX_MM * px_mm
        out = []
        for tag_id, corners in reads.items():
            centre = corners.mean(axis=0)
            page = np.array([centre[0] + 0.5, height - centre[1] - 0.5]) / px_mm / 1000 + self.origin
            if np.linalg.norm(page - self.sheet.centres[tag_id]) < PLACED * self.sheet.tile.sides_m[tag_id]:
                out.append(int(tag_id))
        return sorted(out)

    def overhead(self) -> list[int]:
        """The markers the overhead reads (the stencil observer's reading, fiducials.stencil.overhead_reading)."""
        reader, tile = self.sheet.reader, self.sheet.tile
        image, px_mm = self.view(self.model["overhead"])
        how = reader.overhead_reading(tile)
        reads = reader._largest(reader.find_all(image, tile, how["det"], contrast=how["contrast"]))
        return self._placed({i: corners for i, (corners, _) in reads.items()}, px_mm)

    def wrist(self) -> list[int]:
        """The markers a wrist camera reads with the whole print in view (fiducials.stencil.find)."""
        image, px_mm = self.view(self.model["wrist"])
        return self._placed(self.sheet.reader.find(image, self.sheet.tile), px_mm)


def _strokes(program: dict):
    """[(op, points with a closed path's return, mark width m, rgb)] of a program's strokes, in order."""
    marks = {key: row["width_mm"] and row["width_mm"] / 1000 for key, row in passes.marks(program).items()}
    rgb = {row["id"]: row.get("rgb") or [0, 0, 0] for row in program["resources"]}
    return [(op, op["points_m"] + ([op["points_m"][0]] if op["closed"] else []),
             marks[op["resource_id"]] or op["generation_width_m"], rgb[op["resource_id"]])
            for op in program["ops"] if op["op"] == "stroke"]


def _in_view(sheet: Print, readable, point_m, view_m: float) -> list[int]:
    return [i for i in readable if np.linalg.norm(sheet.centres[i] - np.asarray(point_m[:2])) <= view_m]


def predict(program: dict, stencil, repo, config: dict | None = None, *, drawn=()) -> dict | None:
    """What a program's ink leaves of the print in `stencil` (its folder) for the cameras, or None when that is no
    fiducial-tile print (nothing to track by its markers). `drawn`: programs already drawn on this print, their
    ink under this one's. `config`: page.markers (default: this checkout's stack.yaml).

    {"pattern_id", "markers", "anchors", "clean": the bare print's counts, "inks": per ink in drawing order what
    the overhead reads once it is drawn ("overhead", "anchors", "lost": the codes it read before the program and
    not then), "wrist": {"min", "at": the stroke with the fewest markers in a wrist view of its start, "under":
    strokes under the budget, "lost", "curve": [strokes drawn, markers readable anywhere, fewest in view until the
    next reading]}, "model", "budget"}."""
    config = config or settings(repo)
    sheet = Print(stencil, repo)
    if sheet.tile is None:
        return None
    scene = Scene(sheet, config["model"])
    for earlier in drawn:
        for _, points, mark, rgb in _strokes(earlier):
            scene.lay(points, mark, rgb)
    strokes = _strokes(program)
    budget = {key: config[key] for key in ("overhead_min", "anchors_min", "wrist_min")}
    clean = {"overhead": scene.overhead(), "wrist": scene.wrist()}
    inks, curve, under, readable = [], [], 0, clean["wrist"]
    worst = (len(readable), None, [])   # (markers in view, the stroke, the codes lost there)
    for n, (op, points, mark, rgb) in enumerate(strokes):
        if n and n % config["every"] == 0:
            readable = scene.wrist()
        near = _in_view(sheet, readable, points[0], config["wrist_view_m"])
        if n % config["every"] == 0:
            curve.append([n, len(readable), len(near)])
        curve[-1][2] = min(curve[-1][2], len(near))
        under += len(near) < budget["wrist_min"]
        if worst[1] is None or len(near) < worst[0]:
            lost = sorted(set(_in_view(sheet, clean["wrist"], points[0], config["wrist_view_m"])) - set(near))
            worst = (len(near), op["id"], lost)
        scene.lay(points, mark, rgb)
        if n + 1 == len(strokes) or strokes[n + 1][0]["resource_id"] != op["resource_id"]:
            seen = scene.overhead()
            inks.append({"resource": op["resource_id"], "after": op["id"], "overhead": len(seen),
                         "anchors": len(sheet.tile.anchors & set(seen)), "lost": sorted(set(clean["overhead"]) - set(seen))})
    return {"pattern_id": sheet.pattern_id, "markers": len(sheet.tile.ids), "anchors": len(sheet.tile.anchors),
            "clean": {key: len(value) for key, value in clean.items()}, "inks": inks,
            "wrist": {"min": worst[0], "at": worst[1], "under": under, "lost": worst[2], "curve": curve},
            "model": config["model"], "budget": budget, "every": config["every"], "wrist_view_m": config["wrist_view_m"]}


def _crossed(points_m, tile, reach_m: float) -> frozenset:
    """The markers a path's ink reaches: those whose square, grown by reach_m, holds a point of the line."""
    points = np.asarray(points_m, float)[:, :2]
    steps = np.maximum(1, np.ceil(np.linalg.norm(np.diff(points, axis=0), axis=1) / 0.001).astype(int))
    dense = np.vstack([np.linspace(a, b, n, endpoint=False) for a, b, n in zip(points[:-1], points[1:], steps, strict=True)]
                      + [points[-1:]])
    return frozenset(i for i, c in tile.corners.items()
                     if ((dense >= c.min(axis=0) - reach_m) & (dense <= c.max(axis=0) + reach_m)).all(axis=1).any())


def spare_order(strokes, ink, width, stencil, repo, config: dict | None = None):
    """The placed paths (place.Stroke) with each ink's reordered to spare the print's markers: of the paths left,
    the one whose ink reaches the fewest markers no path has reached yet, the nearest to the last path's end among
    equals. The markers a design must cover are covered late, so the strokes before read more of them.
    `ink(stroke)`: its resource, `width(stroke)`: its mark. Each ink's paths stay together, in the inks' order;
    a print without markers leaves the order as it is.

    On the chicano TATBOT's black (2026-11-10) the model's strokes with under 3 markers in a wrist view fell from
    14 to 5 for 0.9 m more travel; the lab's farthest-first order (tatbot_lab.tile.fill_order: a cell costs one
    marker whatever the order) raised them to 79, each path reaching markers all over the sheet early."""
    tile = _fiducials(repo).load(Path(stencil).expanduser() if Path(stencil).expanduser().is_dir()
                                 else Path(stencil).expanduser().parent)
    if tile is None:
        return list(strokes)
    bleed = (config or settings(repo))["model"]["ink_bleed_m"]
    reached, out, at = set(), [], np.zeros(2)
    runs = []   # consecutive paths of one ink
    for stroke in strokes:
        if runs and ink(runs[-1][0]) == ink(stroke):
            runs[-1].append(stroke)
        else:
            runs.append([stroke])
    for run in runs:
        left = [(stroke, _crossed(stroke.points_m, tile, (width(stroke) + bleed) / 2)) for stroke in run]
        while left:
            best = min(range(len(left)), key=lambda k: (len(left[k][1] - reached),
                                                        float(np.linalg.norm(left[k][0].points_m[0] - at))))
            stroke, crossed = left.pop(best)
            out.append(stroke)
            reached |= crossed
            at = stroke.points_m[-1]
    return out


AREA_POINTS = 200   # an inked region's wrist views are counted at this many of its points


def predict_areas(layers, stencil, repo, config: dict | None = None) -> dict | None:
    """predict for a design not yet drawn as paths: `layers`, in drawing order, are (resource id, rgb, mask, extent_m)
    as Scene.lay_area takes them, each region taken as inked all over (a hatch covers less: the count errs low).
    The wrist's count is the fewest readable markers within a view of any inked point, each ink read once it is
    drawn. The report is predict's, without the wrist's curve."""
    config = config or settings(repo)
    sheet = Print(stencil, repo)
    if sheet.tile is None:
        return None
    scene = Scene(sheet, config["model"])
    budget = {key: config[key] for key in ("overhead_min", "anchors_min", "wrist_min")}
    clean = {"overhead": scene.overhead(), "wrist": scene.wrist()}
    inks, worst = [], (len(clean["wrist"]), None, [])
    for resource, rgb, mask, extent_m in layers:
        mask = np.asarray(mask, bool)
        if not mask.any():
            continue
        scene.lay_area(mask, extent_m, rgb)
        seen, readable = scene.overhead(), scene.wrist()
        inks.append({"resource": resource, "after": resource, "overhead": len(seen),
                     "anchors": len(sheet.tile.anchors & set(seen)), "lost": sorted(set(clean["overhead"]) - set(seen))})
        rows, cols = np.nonzero(mask)
        pick = np.linspace(0, len(rows) - 1, min(AREA_POINTS, len(rows))).astype(int)
        step = np.array([(extent_m[2] - extent_m[0]) / mask.shape[1], (extent_m[3] - extent_m[1]) / mask.shape[0]])
        points = np.c_[extent_m[0] + (cols[pick] + 0.5) * step[0], extent_m[3] - (rows[pick] + 0.5) * step[1]]
        for point in points:
            near = _in_view(sheet, readable, point, config["wrist_view_m"])
            if worst[1] is None or len(near) < worst[0]:
                lost = sorted(set(_in_view(sheet, clean["wrist"], point, config["wrist_view_m"])) - set(near))
                worst = (len(near), resource, lost)
    return {"pattern_id": sheet.pattern_id, "markers": len(sheet.tile.ids), "anchors": len(sheet.tile.anchors),
            "clean": {key: len(value) for key, value in clean.items()}, "inks": inks,
            "wrist": {"min": worst[0], "at": worst[1], "under": int(worst[0] < budget["wrist_min"]), "lost": worst[2]},
            "model": config["model"], "budget": budget, "wrist_view_m": config["wrist_view_m"]}


def _codes(ids) -> str:
    return ", ".join(str(i) for i in ids) or "none"


def refusals(report: dict | None) -> list[str]:
    """What of a prediction is under its budget, one line a stage, naming the stage and the markers lost; [] within
    it (or for a print without markers)."""
    if report is None or not report.get("inks"):
        return []
    budget, out = report["budget"], []
    anchors_min = min(budget["anchors_min"], report["anchors"])
    for ink in report["inks"]:
        if ink["overhead"] < budget["overhead_min"] or ink["anchors"] < anchors_min:
            needs = f"{budget['overhead_min']}" + (f", {anchors_min} of them anchors" if anchors_min else "")
            stage = ink["resource"] + (f" (stroke {ink['after']})" if ink["after"] != ink["resource"] else "")
            out.append(f"after {stage} the overhead would read {ink['overhead']} of "
                       f"the print's {report['markers']} markers ({ink['anchors']} anchors; the budget is {needs}): "
                       f"lost {_codes(ink['lost'])}")
    wrist = report["wrist"]
    if wrist["min"] < budget["wrist_min"]:
        where = (f"at stroke {wrist['at']}", f", and at {wrist['under']} strokes fewer than the budget"
                 ) if "curve" in wrist else (f"over {wrist['at']}", "")
        out.append(f"{where[0]} a wrist view would hold {wrist['min']} readable markers (the budget is "
                   f"{budget['wrist_min']}){where[1]}: lost there {_codes(wrist['lost'])}")
    return out


def summary(report: dict | None) -> str:
    """One line for a person: what the cameras keep."""
    if report is None:
        return "no fiducial-tile print: no marker budget"
    last = report["inks"][-1] if report["inks"] else {"overhead": report["clean"]["overhead"], "anchors": report["anchors"]}
    wrist = report["wrist"]
    return (f"markers: the overhead reads {report['clean']['overhead']} of {report['markers']} before and "
            f"{last['overhead']} after ({last['anchors']} anchors); a wrist view holds {wrist['min']} at its fewest"
            + (f" ({'stroke ' if 'curve' in wrist else ''}{wrist['at']})" if wrist["at"] else ""))
