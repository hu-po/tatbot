"""What the ROS stack's Rerun bridge (ros/tatbot_rerun) shows of a drawing run, as plain data.

A run's state comes from its files alone (program.json and ledger.jsonl, tailed by byte offset), so a bridge started
mid-draw, or one replaying a bag, rebuilds the same scene:
  strokes   planned: the program's stroke ops; done: the ledger's done/aborted rows' measured `line` (a partial stroke
            too), over the planned ghost; current: Draw feedback in PHASE_DRAW, the planned stroke cut at arc_m. Never
            from op_sent/op_done events: op_sent fires before a dip, op_done for tool changes too. All three are kept
            in the print's frame: a measured line plus its stroke's sent `correction_m` (stencil tracking drew the
            stroke there to follow a print that moved, so the ink sits on the print where the design has it), and on
            the page in use less the newest stroke's correction (DrawState.shift), where the print is now
  placement tf world -> <arm>/base_link and the newest page row's `used`; tf world -> page/<pattern> only before the
            first page row (the touches correct it by millimetres). A page.surface.mode fitted page lifts points onto
            its height field (tatbot_motion.surface.PageSurface); the page's own grid and edges hold the field's edge
            height past its extent, where the polynomial is an extrapolation
  design    the print's millimetres oriented like drawn.svg (ros_drawn: x right, the print's top up the image); unlike
            drawn.svg, whose lines are on the page each stroke used, a corrected stroke's line here is on the print,
            whose installed stencil.png lies faintly under the strokes, cut to the design's frame
  palette   the station touch over the newest station fix, the run's own or a ros-station run's, whichever is newer
            (tatbot_motion.station.touched_palette), cap rims from config/palette.yaml, each cap's state (loaded and
            available to dip, empty, unknown, no cap) and level from the deployed config/palette_load.yaml, dips
            counted from the ledger; never the simulator's ink volume model. Each cap carries a mark over its rim, and
            the loaded ones a row of their own in a label column laid out for the Demo eye, the rest grouped. The
            print takes the arm's colour, and its tag is drawn as the tag (its textured quad)
  robot     the URDF's visuals, per link; the wrist tags' quads stand out of their carrier's faces
numpy; no ROS and no rerun. tatbot_description (its transforms), tatbot_motion and scripts/lib's ink_spec are imported
lazily by the functions that need them, from the bridge's own path.
"""
from __future__ import annotations

import datetime as dt
import json
import xml.etree.ElementTree as ET
from collections import Counter
from pathlib import Path
from urllib.parse import unquote, urlparse

import numpy as np

from tatbot_contracts.ros_progress import counts, next_index

PLANNED_RGBA = (154, 170, 187, 64)   # drawn.svg's planned #9ab, translucent
CURRENT_RGB = (255, 60, 60)
PAGE_RGB, LOST_RGB = (40, 160, 220), (160, 160, 160)
STENCIL_RGBA = (150, 120, 220, 40)   # the print's black under the design map: a transfer's violet, faint (Rerun blends
                                     # in linear light, so 40 reads as a mid violet) under a lifted black ink line
NEXT_RGB = (0, 220, 255)             # the next dip's cap and its label (a cap's own ink colour can be near black)
# A cap by its state (caps), in the status board's colours (ros_status.RGB): loaded and available to dip green, empty
# or no cap grey, unknown yellow. Never its ink's colour: black ink would not read on the dark palette.
CAP_RGB = {"loaded": (0x2E, 0xCC, 0x71), "empty": (0x7F, 0x8C, 0x8D), "absent": (0x7F, 0x8C, 0x8D),
           "unknown": (0xF1, 0xC4, 0x0F)}
ARM_MATERIAL = "black"               # the arm links' URDF material, which the palette (a black print, no colour) takes too
TAG_LIFT_M = 0.003                   # a tag's quad out of its face: the right arm's measured wrist tags lie up to 2.4 mm
                                     # inside its nominal printed carrier (2026-10-19), which hid all but a sliver of them


# --- geometry -------------------------------------------------------------------------------------------------------
def transform(matrix, points) -> np.ndarray:
    points = np.asarray(points, float)
    return points @ np.asarray(matrix, float)[:3, :3].T + np.asarray(matrix, float)[:3, 3]


def moved(previous, pose, tolerance: float = 5e-4) -> bool:
    """Whether a 4x4 pose differs from the one logged before (None: never logged) by more than tolerance in any
    element: 0.5 mm, or about 0.5 mrad of turn, under a pixel of the Demo scene. A landed arm's encoder noise mostly
    stays under it, so a still robot logs (almost) nothing."""
    return previous is None or float(np.abs(np.asarray(pose) - previous).max()) > tolerance


def rectangle(width_m: float, height_m: float, per_side: int = 1) -> np.ndarray:
    """A closed rectangle centred on the page origin, in the page plane (z = 0), each side cut into per_side segments
    (lifted onto a curved page, its edges then follow the surface instead of cutting chords), (4 per_side + 1) x 3."""
    w, h = width_m / 2, height_m / 2
    corners = np.array([[-w, -h, 0], [w, -h, 0], [w, h, 0], [-w, h, 0], [-w, -h, 0]], float)
    share = np.arange(per_side)[:, None] / per_side
    return np.vstack([a + share * (b - a) for a, b in zip(corners[:-1], corners[1:], strict=True)] + [corners[-1:]])


def cut(points, arc_m: float, closed: bool = False) -> np.ndarray:
    """The polyline's first arc_m metres (a closed stroke runs back to its start), (M, 2)."""
    pts = np.asarray(points, float).reshape(-1, 2)
    if closed and len(pts) > 1:
        pts = np.vstack([pts, pts[:1]])
    run = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(pts, axis=0), axis=1))]
    if arc_m >= run[-1]:
        return pts
    k = int(np.searchsorted(run, max(arc_m, 0.0), side="right"))
    share = (arc_m - run[k - 1]) / max(run[k] - run[k - 1], 1e-12)
    return np.vstack([pts[:k], pts[k - 1] + share * (pts[k] - pts[k - 1])])


def lift(rgb) -> tuple[int, int, int]:
    """An ink colour moved 30% toward white, so black ink still reads on the viewer's dark background."""
    rgb = np.asarray(rgb, float)[:3]
    return tuple(int(v) for v in rgb + (255.0 - rgb) * 0.3)


def design_mm(points, size_m) -> np.ndarray:
    """Page metres -> the design map's millimetres, drawn.svg's convention (ros_drawn._xy)."""
    pts = np.asarray(points, float).reshape(-1, 2)
    return np.column_stack([(pts[:, 0] + size_m[0] / 2) * 1000.0, (size_m[1] / 2 - pts[:, 1]) * 1000.0])


def design_frame(strokes_mm, margin: float = 0.08, pad_mm: float = 2.0) -> np.ndarray | None:
    """A closed rectangle round the design's strokes (design-map mm) with a margin: the Demo's Design view fits to
    it, so the design sits centred in its pane instead of wherever it lies on the print. None without strokes."""
    pts = [np.asarray(s, float).reshape(-1, 2) for s in strokes_mm if len(s)]
    if not pts:
        return None
    lo, hi = np.min([q.min(0) for q in pts], 0), np.max([q.max(0) for q in pts], 0)
    m = margin * float(np.max(hi - lo)) + pad_mm
    (x0, y0), (x1, y1) = lo - m, hi + m
    return np.array([[x0, y0], [x1, y0], [x1, y1], [x0, y1], [x0, y0]])


def within(frame, point) -> bool:
    """Whether a design-map point lies inside design_frame's rectangle."""
    lo, hi = np.asarray(frame)[:4].min(0), np.asarray(frame)[:4].max(0)
    return bool(np.all(np.asarray(point, float).reshape(2) >= lo) and np.all(np.asarray(point, float).reshape(2) <= hi))


def stencil_underlay(gray, page_mm, size_m, frame, max_px: int = 640, rgba=STENCIL_RGBA) -> dict | None:
    """The print under the design map: its stencil.png (greyscale, spanning page_mm) cut to design_frame's rectangle,
    so the Design view still fits the frame, block-averaged to at most max_px a side, its ink `rgba` with alpha by
    darkness and its paper clear. {rgba (H, W, 4) uint8, translation, scale}: a Transform3D putting the image's pixel
    corners in design-map mm. None without a frame or where the frame misses the print."""
    if frame is None:
        return None
    gray = np.asarray(gray, float)
    mm_per_px = np.asarray(page_mm, float) / gray.shape[1::-1]
    origin = design_mm([[-page_mm[0] / 2000.0, page_mm[1] / 2000.0]], size_m)[0]   # the print's top left
    lo, hi = np.asarray(frame)[:4].min(0), np.asarray(frame)[:4].max(0)
    (x0, y0), (x1, y1) = (np.clip(f((v - origin) / mm_per_px), 0, gray.shape[1::-1]).astype(int)
                          for f, v in ((np.ceil, lo), (np.floor, hi)))
    k = max(1, -(-max(x1 - x0, y1 - y0) // max_px))
    x1, y1 = x0 + (x1 - x0) // k * k, y0 + (y1 - y0) // k * k   # whole blocks, inside the frame
    if x1 <= x0 or y1 <= y0:
        return None
    block = gray[y0:y1, x0:x1].reshape((y1 - y0) // k, k, (x1 - x0) // k, k).mean(axis=(1, 3))
    out = np.empty(block.shape + (4,), np.uint8)
    out[..., :3], out[..., 3] = rgba[:3], np.round((1.0 - block / 255.0) * rgba[3])
    return {"rgba": out, "translation": [*(origin + np.array([x0, y0]) * mm_per_px), 0.0],
            "scale": [*(mm_per_px * k), 1.0]}


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


# --- the run --------------------------------------------------------------------------------------------------------
class LedgerTail:
    """ledger.jsonl read incrementally by byte offset: each poll returns the complete lines appended since the last;
    a partial last line waits for its newline, and a torn line (a crash the ledger later ended) is skipped."""

    def __init__(self, path):
        self.path, self.offset = Path(path), 0

    def poll(self) -> list[dict]:
        try:
            with self.path.open("rb") as stream:
                stream.seek(self.offset)
                data = stream.read()
        except OSError:
            return []
        end = data.rfind(b"\n") + 1
        self.offset += end
        rows = []
        for line in data[:end].splitlines():
            try:
                rows.append(json.loads(line))
            except ValueError:
                continue
        return rows


def stamp_s(text) -> float | None:
    """Seconds since the epoch of a run id's or a fix's leading UTC stamp (20261018T173448Z...) or of an ISO-8601 UTC
    time (meta.json's started_at), else None."""
    for form, size in (("%Y%m%dT%H%M%SZ", 16), ("%Y-%m-%dT%H:%M:%S", 19)):
        try:
            return dt.datetime.strptime(str(text)[:size], form).replace(tzinfo=dt.timezone.utc).timestamp()
        except ValueError:
            continue
    return None


def newest_run(workflow_dir, now_s: float | None = None) -> Path | None:
    """The newest run directory of a workflow (run ids lead with their UTC stamp), or None; with now_s, the newest
    one started by then (a bag replayed on its own clock never meets a later run)."""
    runs = sorted(p for p in Path(workflow_dir).glob("*T*Z-*") if p.is_dir() and not p.is_symlink()
                  and (now_s is None or (stamp_s(p.name) or 0.0) <= now_s))
    return runs[-1] if runs else None


def palette_fix_path(log_root, run_dir=None, now_s: float | None = None) -> Path | None:
    """The newest station fix by its UTC stamp: a draw run's own (<run>/station/<op>-<stamp>/station.json, one per dip)
    or the newest ros-station run's (<stamp>-<node>-<n>/station.json), so a station measured after a draw wins; with
    now_s, only fixes made by then."""
    own = Path(run_dir).glob("station/*/station.json") if run_dir else ()
    fixes = [(path.parent.name.rsplit("-", 1)[-1], path) for path in own]
    run = newest_run(Path(log_root) / "ros-station", now_s)
    if run is not None and (run / "station.json").is_file():
        fixes.append((run.name.split("-", 1)[0], run / "station.json"))
    fixes = [fix for fix in fixes if now_s is None or (stamp_s(fix[0]) or 0.0) <= now_s]
    return max(fixes, key=lambda fix: fix[0])[1] if fixes else None


class DrawState:
    """One ros-draw run of one arm, from its program and its ledger's rows. Rows are applied once the node's clock
    reaches their `t` (a bag replayed on its own clock meets them in order; a live run's are always due)."""

    def __init__(self, run_dir, program: dict, arm: str):
        self.run_dir, self.arm = Path(run_dir), arm
        self.ops = program.get("ops", [])
        self.by_id = {op["id"]: op for op in self.ops}
        self.resources = {r["id"]: r for r in program.get("resources", [])}
        self.planned = {op["id"]: np.asarray(op["points_m"], float).reshape(-1, 2) for op in self.ops
                        if op.get("op") == "stroke" and len(op.get("points_m") or []) > 1}
        self.tail = LedgerTail(self.run_dir / "ledger.jsonl")
        self.rows: list[dict] = []
        self.pending: list[dict] = []
        self.lines: dict[str, list] = {}   # op id -> its measured lines (N, 2), in the print's frame
        self.corrections: dict[str, np.ndarray] = {}   # stroke op id -> its newest sent row's correction_m
        self.shift = np.zeros(2)   # the newest stroke's correction: the page in use has print point p at p - shift
        self.page: dict | None = None
        self.dips: Counter = Counter()
        self.trims: dict[str, float] = {}   # resource id -> its pen trim (m), the newest pen_trim row's
        self.trim_cap: tuple = (None, None)   # (op, m): the newest sent row's trim_cap_m, its deepest trim
        self.t: float | None = None
        self.meta: dict = {}
        self.meta_stamp = self.motion_stamp = self.trim_limit_m = None

    @classmethod
    def load(cls, run_dir) -> DrawState | None:
        path = Path(run_dir) / "program.json"
        if not path.is_file():
            return None
        program = json.loads(path.read_text())
        return cls(run_dir, program, program.get("arm") or "right")

    def poll(self, now_s: float | None = None) -> set[str]:
        """Apply the rows due by now_s (None: all); the layers they change (design, planned, done, page, palette)."""
        self._read_meta()
        rows = self.pending + self.tail.poll()
        due = [row for row in rows if now_s is None or float(row.get("t", 0.0)) <= now_s]
        self.pending = [row for row in rows if now_s is not None and float(row.get("t", 0.0)) > now_s]
        changed = set()
        for row in due:
            changed |= self._apply(row)
        return changed

    def _read_meta(self) -> None:
        """meta.json (the run log's status and start, its end once finalized), read again when it changes."""
        try:
            stamp = (self.run_dir / "meta.json").stat().st_mtime_ns
            if stamp != self.meta_stamp:
                self.meta, self.meta_stamp = json.loads((self.run_dir / "meta.json").read_text()), stamp
        except (OSError, ValueError):
            pass

    def summary(self) -> dict:
        """The run for the status (ros_status.items): {id, t, index, total, done, started_s, ended_s, status}."""
        status = self.meta.get("status")
        return {"id": self.run_dir.name, "t": self.t, "index": self.next_op(), "total": len(self.ops),
                "done": sum(counts(self.rows, self.arm).values()), "status": status,
                "started_s": stamp_s(self.meta.get("started_at") or self.run_dir.name),
                "ended_s": stamp_s(self.meta.get("ended_at")) if status not in (None, "running") else None}

    def _apply(self, row: dict) -> set[str]:
        if row.get("arm") != self.arm:
            return set()
        self.rows.append(row)
        self.t = float(row.get("t", self.t or 0.0))
        event = row.get("event")
        if event == "page" and row.get("page"):
            self.page, self.shift = row["page"], np.zeros(2)   # set up again: the page in use is on the print
            return {"page", "planned", "done"}
        if event == "pen_trim":
            self.trims[row.get("resource")] = float(row.get("trim_m", 0.0))
            return set()
        if event == "sent":
            self.trim_cap = (row.get("op"), row.get("trim_cap_m"))
            if row.get("op") in self.planned:
                c = self.corrections[row["op"]] = np.asarray(row.get("correction_m") or (0.0, 0.0), float)
                if not np.array_equal(c, self.shift):
                    self.shift = c
                    return {"planned", "done"}
        if event == "dip_phase" and row.get("phase") == "dwell" and row.get("status") == "ok":
            self.dips[row.get("slot")] += 1
            return {"palette"}
        if event in ("done", "aborted") and row.get("line"):
            line = np.asarray(row["line"], float).reshape(-1, 2) + self.corrections.get(row["op"], np.zeros(2))
            self.lines.setdefault(row["op"], []).append(line)
            return {"done", "design"}
        return set()

    @property
    def pattern_id(self) -> str | None:
        """The print the run draws on (its run log's arms.<arm>.pattern_id); None before meta.json says."""
        return ((self.meta.get("arms") or {}).get(self.arm) or {}).get("pattern_id")

    def colour(self, op_id: str) -> tuple[int, int, int]:
        """The stroke's ink colour, lifted (black ink reads on a dark view)."""
        return lift(self.resources.get(self.by_id[op_id].get("resource_id"), {}).get("rgb") or (0, 0, 0))

    def next_op(self) -> int | None:
        """Index of the op the run is on, or does next, from the ledger."""
        return next_index(self.ops, self.rows, self.arm)

    def ink(self, index: int | None) -> dict | None:
        """{ink, slot} of the op at index (its resource), or None."""
        if index is None or not 0 <= index < len(self.ops):
            return None
        resource = self.resources.get(self.ops[index].get("resource_id")) or {}
        return {"ink": resource.get("ink_id"), "slot": resource.get("slot")}

    def trim(self, index: int | None) -> dict | None:
        """{m, limit_m, used_m} of the op at index's resource (ros_status.trim_item): its pen trim (0 before its first
        pen_trim row), motion.yaml's pen.trim limit_m as the run directory holds it (each draw goal writes it; None
        unread), and the trim that op takes once sent: a needle cartridge's no deeper than its sent row's trim_cap_m
        (the skin read along it), or None."""
        if index is None or not 0 <= index < len(self.ops):
            return None
        import yaml
        path = self.run_dir / "motion.yaml"
        try:
            stamp = path.stat().st_mtime_ns
            if stamp != self.motion_stamp:
                self.trim_limit_m = float(yaml.safe_load(path.read_text())["pen"]["trim"]["limit_m"])
                self.motion_stamp = stamp
        except (OSError, ValueError, KeyError, TypeError, yaml.YAMLError):
            pass
        op, m = self.ops[index], self.trims.get(self.ops[index].get("resource_id"), 0.0)
        cap = self.trim_cap[1] if self.trim_cap[0] == op.get("id") else None
        return {"m": m, "limit_m": self.trim_limit_m, "used_m": m if cap is None else max(m, float(cap))}

    def dip_slot(self, index: int | None) -> str | None:
        """The slot of the dip at or after op `index`, the one the palette highlights."""
        if index is None:
            return None
        return next((op.get("slot") for op in self.ops[index:] if op.get("op") == "dip"), None)

    def current(self, op_id: str, arc_m: float) -> np.ndarray | None:
        """The stroke the pen is drawing, cut at arc_m, in the print's frame (on_page puts it where the pen is)."""
        op = self.by_id.get(op_id)
        if op_id not in self.planned or op is None:
            return None
        return cut(self.planned[op_id], arc_m, bool(op.get("closed")))

    def on_page(self, points) -> np.ndarray:
        """Print-frame points on the page in use: less the newest stroke's correction, where the print is now."""
        return np.asarray(points, float).reshape(-1, 2) - self.shift


# --- placement ------------------------------------------------------------------------------------------------------
def page_surface(page: dict | None, fitted: bool):
    """The page row's curved surface (tatbot_motion.surface.PageSurface over `used`) on a fitted page, else None."""
    field = ((page or {}).get("surface") or {}).get("field")
    if not fitted or not field or "used" not in (page or {}):
        return None
    from tatbot_motion.surface import HeightField, PageSurface

    return PageSurface(page["used"], HeightField.from_dict(field))


def page_local(points, surface=None, *, chart: bool = True, within_fit: bool = False) -> np.ndarray:
    """(N, 3) page-frame points of print (u, v) points (chart) or base-plane (x, y) points (a measured line): the
    plane's z = 0, or on a curved surface its height there (PageSurface.lift without the base). within_fit: past the
    field's extent the height of its nearest edge, not the polynomial's extrapolation (centimetres off at a print's
    corners); a stroke keeps the lift the arm follows."""
    pts = np.asarray(points, float).reshape(-1, 2)
    if surface is None:
        return np.column_stack([pts, np.zeros(len(pts))])
    xy = surface.chart.to_plane(pts) if chart else pts
    x0, y0, x1, y1 = surface.height.extent if within_fit else (-np.inf, -np.inf, np.inf, np.inf)
    return np.column_stack([xy, surface.height(np.column_stack([np.clip(xy[:, 0], x0, x1), np.clip(xy[:, 1], y0, y1)]))])


def page_grid(size_m, surface=None, step_m: float = 0.01) -> list[np.ndarray]:
    """The page as lines of constant u and v every step_m over its size, page frame (on the surface when curved,
    within its fit)."""
    w, h = size_m[0] / 2, size_m[1] / 2
    us, vs = np.linspace(-w, w, max(2, round(2 * w / step_m) + 1)), np.linspace(-h, h, max(2, round(2 * h / step_m) + 1))
    fine_u, fine_v = np.linspace(-w, w, 24), np.linspace(-h, h, 32)
    lines = [np.column_stack([np.full(len(fine_v), u), fine_v]) for u in us]
    lines += [np.column_stack([fine_u, np.full(len(fine_u), v)]) for v in vs]
    return [page_local(line, surface, within_fit=True) for line in lines]


def page_lines(size_m, clear_m, surface=None) -> list[np.ndarray]:
    """The page layer, page frame: the print's edge and its clear centre (each side cut in 24), then page_grid."""
    edges = [page_local(rectangle(*s, per_side=24)[:, :2], surface, within_fit=True) for s in (size_m, clear_m)]
    return edges + page_grid(size_m, surface)


def table_grid(span_x=(-0.10, 0.60), span_y=(-0.45, 0.45), step_m: float = 0.05) -> list[np.ndarray]:
    """A nominal table plane, the arm base's z = 0, as a line grid in the base frame."""
    xs, ys = np.arange(span_x[0], span_x[1] + 1e-9, step_m), np.arange(span_y[0], span_y[1] + 1e-9, step_m)
    lines = [np.array([[x, span_y[0], 0.0], [x, span_y[1], 0.0]]) for x in xs]
    return lines + [np.array([[span_x[0], y, 0.0], [span_x[1], y, 0.0]]) for y in ys]


# --- the robot ------------------------------------------------------------------------------------------------------
def mesh_asset(path) -> tuple[bytes, str | None]:
    """A visual mesh's bytes for Rerun's Asset3D, and its media type (None: from the file name). A binary glTF loses
    its texture samplers: Rerun 0.36 always samples mesh textures repeating and trilinear and warns once per
    primitive about any other setting (the wrist tags ask for nearest and clamp-to-edge), so the look is the same
    and only the warnings go."""
    import struct

    data = Path(path).read_bytes()
    if data[:4] != b"glTF" or len(data) < 20 or data[16:20] != b"JSON":
        return data, None
    size = struct.unpack_from("<I", data, 12)[0]
    doc = json.loads(data[20:20 + size])
    if "samplers" not in doc:
        return data, None
    doc.pop("samplers")
    for texture in doc.get("textures", []):
        texture.pop("sampler", None)
    text = json.dumps(doc, separators=(",", ":")).encode()
    text += b" " * (-len(text) % 4)
    rest = data[20 + size:]
    head = data[:8] + struct.pack("<I", 20 + len(text) + len(rest)) + struct.pack("<I", len(text)) + b"JSON"
    return head + text + rest, "model/gltf-binary"


def mesh_path(uri: str, base_dir, packages=None) -> Path | None:
    """A URDF mesh filename as a local path: file://, package://<pkg>/ (through `packages(pkg) -> share dir`), or
    relative to the URDF's directory."""
    parsed = urlparse(uri)
    if parsed.scheme == "file":
        return Path(unquote(parsed.path))
    if parsed.scheme == "package":
        share = packages(parsed.netloc) if packages else None
        return Path(share) / unquote(parsed.path).lstrip("/") if share else None
    return Path(base_dir) / uri if not parsed.scheme else None


def tag_asset(family: str, tag_id: int, edge_m: float) -> str:
    """A tag's textured quad under urdf/, as scripts/vision/generate_wrist_tags.py --viewer-assets writes it: the
    marker and its one-module white quiet zone, the black square edge_m wide, in the tag frame (+z out of the face)."""
    return f"meshes/tags/{family.removeprefix('apriltag_')}_{int(tag_id):03d}_{round(edge_m * 1000)}mm/tag.glb"


def is_tag_quad(path) -> bool:
    """Whether a visual's mesh is a tag's quad (tag_asset)."""
    return Path(path).name == "tag.glb" and Path(path).parent.parent.name == "tags"


def _out_of_face(pose) -> np.ndarray:
    """A tag quad's pose TAG_LIFT_M along its own +z, out of the face it is stuck on."""
    lifted = np.array(pose, float)
    lifted[:3, 3] += lifted[:3, 2] * TAG_LIFT_M
    return lifted


def _materials(root) -> dict[str, str]:
    return {m.get("name"): m.find("color").get("rgba") for m in root.findall("material") if m.find("color") is not None}


def _rgba(text: str | None):
    return tuple(int(round(float(v) * 255)) for v in text.split()) if text else None


def _visual_rgba(visual, materials: dict):
    material = visual.find("material")
    if material is None:
        return None
    color = material.find("color")
    return _rgba(color.get("rgba") if color is not None else materials.get(material.get("name")))


def material_rgba(urdf_path, name: str = ARM_MATERIAL):
    """A URDF's named top-level material as 0-255 RGBA, or None."""
    return _rgba(_materials(ET.parse(urdf_path).getroot()).get(name))


def _visual(link: str, k: int, visual, materials: dict, base_dir, packages) -> dict | None:
    from tatbot_description.transforms import rpy_matrix

    geometry = visual.find("geometry")
    shape = geometry[0] if geometry is not None and len(geometry) else None
    origin = visual.find("origin")
    pose = rpy_matrix(*([float(v) for v in (origin.get(key) or "0 0 0").split()] for key in ("xyz", "rpy"))) \
        if origin is not None else np.eye(4)
    out = {"link": link, "name": f"visual_{k}", "link_from_visual": pose, "rgba": _visual_rgba(visual, materials)}
    if shape is not None and shape.tag == "mesh":
        path = mesh_path(shape.get("filename", ""), base_dir, packages)
        scale = [float(v) for v in (shape.get("scale") or "1 1 1").split()]
        if path and is_tag_quad(path):
            out["link_from_visual"] = _out_of_face(pose)
        return {**out, "kind": "mesh", "path": path, "scale": scale} if path and path.is_file() else None
    if shape is not None and shape.tag == "cylinder":
        return {**out, "kind": "cylinder", "radius": float(shape.get("radius")), "length": float(shape.get("length"))}
    return None


def urdf_visuals(urdf_path, packages=None, rgba=None) -> list[dict]:
    """Every link's mesh and cylinder visuals (collision geometry ignored; a mesh not found on disk is left out):
    {link, name, kind, link_from_visual, rgba, path + scale | radius + length}. rgba: the colour of a visual that names
    none. A tag's quad (is_tag_quad) is moved out of its face (TAG_LIFT_M)."""
    root = ET.parse(urdf_path).getroot()
    materials = _materials(root)
    out = []
    for link in root.findall("link"):
        for k, visual in enumerate(link.findall("visual")):
            item = _visual(link.get("name"), k, visual, materials, Path(urdf_path).parent, packages)
            if item is not None:
                out.append({**item, "rgba": item["rgba"] or rgba})
    return out


def cylinders(visuals) -> dict[str, dict]:
    """A link's cylinder visuals batched, in its frame: {link: {centers, quaternions, radii, lengths, colors}}."""
    from tatbot_description.transforms import matrix_quat

    out: dict[str, dict] = {}
    for v in visuals:
        if v["kind"] != "cylinder":
            continue
        batch = out.setdefault(v["link"], {"centers": [], "quaternions": [], "radii": [], "lengths": [], "colors": []})
        batch["centers"].append(v["link_from_visual"][:3, 3])
        batch["quaternions"].append(matrix_quat(v["link_from_visual"]))
        batch["radii"].append(v["radius"])
        batch["lengths"].append(v["length"])
        batch["colors"].append(v["rgba"] or (200, 200, 200, 255))
    return out


# --- the palette ----------------------------------------------------------------------------------------------------
def palette_visuals(repo, robot_urdf) -> list[dict]:
    """The palette's visuals in palette_root, as urdf_visuals gives them: its URDF's print in the arm's colour (robot_urdf's
    ARM_MATERIAL), then the palette tag's quad (fiducials.json's palette target; the seat holds its first id) at the
    URDF's palette_tag, TAG_LIFT_M out of its seat; no quad when the repo has no asset for it."""
    from tatbot_motion import station

    repo = Path(repo)
    urdf = repo / "urdf" / "palette.urdf"
    target = json.loads((repo / "config" / "fiducials.json").read_text())["targets"]["palette"]
    path = repo / "urdf" / tag_asset(target["family"], target["ids"][0], target["edge_m"])
    tag = {"link": "palette_root", "name": f"tag_{target['ids'][0]}", "kind": "mesh", "rgba": None, "path": path,
           "scale": [1.0, 1.0, 1.0], "link_from_visual": _out_of_face(station.palette_from_tag(urdf))}
    return urdf_visuals(urdf, rgba=material_rgba(robot_urdf)) + ([tag] if path.is_file() else [])


def palette_pose(repo, arm: str, fix_path, touch_path=None) -> dict:
    """{base_from_palette, measured_s, how}: the station touch carried over the fix (touched_palette with the touch's own
    tool and tcp), else the fix alone, saying why."""
    from tatbot_motion import station

    fix = station.StationFix.from_dict(json.loads(Path(fix_path).read_text()))
    urdf = Path(repo) / "urdf" / "palette.urdf"
    try:
        touch = station.load_touch(arm, touch_path)
        pose = station.touched_palette(touch, fix, tool_id=touch.tool_id, tcp_m=touch.tcp_m,
                                       joint_offsets=touch.joint_offsets, palette_urdf=urdf)
        how = "touched"
    except (station.StaleStationError, ValueError, KeyError, OSError) as why:
        pose, how = np.asarray(fix.base_from_palette, float), f"fix only: {why}"
    return {"base_from_palette": pose, "measured_s": station.parse_utc(fix.measured_utc), "how": how}


def _cap_state(load, inks: dict) -> tuple[str, str]:
    """(state, what the label says of a cap that is not loaded). loaded: what a dip admits (tatbot_session.cap): the
    cap there, a catalog ink and its level; empty: a cap declared with no ink; absent: no cap; unknown: anything else,
    a cap never declared or declared unknown, an ink with no level, or an ink the catalog does not hold."""
    if load is None or load.cap_present is None:
        return "unknown", "unknown"
    if load.cap_present is False:
        return "absent", "no cap"
    if not load.ink_id:
        return "empty", "empty"
    if load.ink_id not in inks:
        return "unknown", f"{load.ink_id}? not an ink"
    return ("loaded", "") if load.level_lower_bound_m is not None else ("unknown", f"{ink_word(load.ink_id)}, level?")


def ink_word(ink_id: str) -> str:
    """An ink id's last word, the colour in every catalog id (nighthawk_black -> black, snow_white -> white)."""
    return str(ink_id).rsplit("_", 1)[-1]


def _cap(slot: str, rim, radius: float, size: dict, load, inks: dict, dips: int) -> dict:
    state, said = _cap_state(load, inks)
    loaded = state == "loaded"
    wall, outside = float(size.get("wall_m", 0.0005)), float(size["height_m"])
    height = outside if state != "absent" else 0.0
    rim = np.asarray(rim, float) + [0.0, 0.0, outside - height]   # no cap: where its rim would stand, for the ring
    level = float(load.level_lower_bound_m) if loaded else None
    alias = slot.removeprefix("inkcap_")
    alias = f"{alias[0].upper()}{alias.rsplit('_', 1)[-1]}"   # inkcap_large_1 -> L1, as `tatbot ros palette load` takes
    took = f" · {dips} dip{'' if dips == 1 else 's'}" if dips else ""
    fill = None if level is None else min(level / (outside - wall), 1.0)
    if loaded:
        brief = f"{alias} {ink_word(load.ink_id)} ≥{fill:.0%}"
        text = f"{alias} {ink_word(load.ink_id)} ≥{level * 1000:.1f} mm {fill:.0%}{took}"
    else:
        brief = text = f"{alias} {said}{took}"
    return {"slot": slot, "alias": alias, "state": state, "ink": load.ink_id if loaded else None,
            "rgb": lift(inks[load.ink_id].rgb) if loaded else None, "rim": rim, "radius": float(radius),
            "height": height, "inner_radius": float(radius) - wall, "floor": rim - [0.0, 0.0, outside - wall],
            "level": level, "fill": fill, "dips": dips, "text": text, "brief": brief}


def caps(repo, base_from_palette, dips: Counter) -> list[dict]:
    """Each cap in palette_root, in its slot's order: {slot, alias (L1), state (loaded, empty, unknown, absent; see
    _cap_state), ink, rgb (lifted ink colour, loaded caps only), rim (where it would stand without a cap), radius
    (outside), height (0 without a cap), inner_radius, floor (inside), level (m of ink over the floor, a lower bound
    recorded at load), fill (level over the inner depth), dips (the run's from it), text (its label), brief (the
    status board's)}. The DEPLOYED repo's palette load; never the simulator's ink volume model."""
    import ink_spec
    import yaml
    from tatbot_motion import station

    repo = Path(repo)
    load = ink_spec.load_palette_load(repo)
    inks = ink_spec.load_inks(repo)
    sizes = yaml.safe_load((repo / "config" / "palette.yaml").read_text())
    rims = station.inkcap_rims(repo / "urdf" / "palette.urdf", repo / "config" / "palette.yaml", load=load,
                               base_from_palette=base_from_palette)
    return [_cap(slot, rim, radius, sizes["sizes"][sizes["slots"][slot]["size"]], load.get(slot), inks, dips[slot])
            for slot, rim, radius in rims]


# The Demo scene's default eye (rust/visiond/src/rerun_viewer.rs DEMO_EYE and DEMO_LOOK_AT, z up; keep in step) in the
# arm base, Rerun's 55° vertical field of view, and the station's Scene pane under its title bar, pixels.
DEMO_VIEW = {"eye": (0.44, 0.68, 0.60), "look_at": (0.24, 0.10, 0.15), "fov_y_deg": 55.0, "pane_px": (860.0, 642.0)}
# The caps' labels, laid out in the Demo eye's pixels as the station draws a label (1.35 px/pt: Xft.dpi 144 times the
# viewer's zoom 0.9): Rerun centres a label's box on its point, the box's top `top_px` under it, `box_px` high, its text
# `em_px` an em (label_px) and `pad_px` a side. At the Scene's scale six caps fit in about 60 px, so labels on the caps
# pile up: they stand in a column `gap_px` left of the caps instead, centred on one line, rows `pitch_px` apart (half
# again a box), so they stay apart and off the caps while a touch turns the eye up to 30° or tilts it 15° at the same
# zoom; `tatbot viewer reset` brings the eye back. Each cap's mark stands `lift_m` over its rim, clear of the D555
# cloud, which lies on the caps' tops; the column stands `toward_eye_m` nearer the eye, so its leaders clear it too.
CAP_LABELS = {"lift_m": 0.02, "toward_eye_m": 0.06, "em_px": 17.0, "pad_px": 6.0, "top_px": 1.5, "box_px": 30.0,
              "gap_px": 48.0, "pitch_px": 46.0, "end_px": 6.0, "stub_px": 12.0, "margin_px": 8.0}
CAP_FOOTNOTE = "≥ level as recorded at load"
MARK_RADII = {"next": -4.0, "loaded": -3.5, "unknown": -3.0, "empty": -2.0, "absent": -2.0}   # UI points


def demo_camera(view=DEMO_VIEW) -> tuple[np.ndarray, np.ndarray, float]:
    """(eye, rows [right, down, forward] of the view, focal length in pixels)."""
    eye = np.asarray(view["eye"], float)
    forward = np.asarray(view["look_at"], float) - eye
    forward /= np.linalg.norm(forward)
    right = np.cross(forward, [0.0, 0.0, 1.0])
    right /= np.linalg.norm(right)
    return eye, np.array([right, np.cross(forward, right), forward]), view["pane_px"][1] / 2 / np.tan(
        np.radians(view["fov_y_deg"]) / 2)


def to_screen(points, view=DEMO_VIEW) -> tuple[np.ndarray, np.ndarray]:
    """Points in the arm base -> (their pixels in the Scene pane, x right and y down; their depth from the eye)."""
    eye, axes, focal = demo_camera(view)
    local = (np.asarray(points, float) - eye) @ axes.T
    return focal * local[:, :2] / local[:, 2:] + np.asarray(view["pane_px"]) / 2, local[:, 2]


def from_screen(pixels, depth, view=DEMO_VIEW) -> np.ndarray:
    """Pixels in the Scene pane at a depth from the eye -> points in the arm base (to_screen's inverse)."""
    eye, axes, focal = demo_camera(view)
    xy = (np.asarray(pixels, float) - np.asarray(view["pane_px"]) / 2) * depth / focal
    return eye + np.column_stack([xy, np.full(len(xy), depth)]) @ axes


def label_px(text: str, layout=CAP_LABELS) -> float:
    """A label's box width on the station, pixels, from its characters' widths in the viewer's font (Inter), in ems."""
    em = sum(0.3 if ch in " .,'·!ijlrtfI1" else 0.85 if ch in "mwMW%" else 0.63 if ch.isupper() or ch.isdigit()
             or ch == "≥" else 0.55 for ch in text)
    return em * layout["em_px"] + 2 * layout["pad_px"]


def label_column(marks, led, rest, footnote: str | None, view=DEMO_VIEW, layout=CAP_LABELS):
    """The label column at the Demo eye, in the arm base: (rows top to bottom as (text, mark index or None), each row's
    label point, {mark index: its leader's elbow and end}). marks: every cap's mark; led: (text, mark index) for each
    cap with its own row and leader, ordered as the eye sees their marks so no leader crosses another, centred on
    them; rest: texts with no cap, then the footnote, under them (the least needed rows are the first a moved eye
    loses past the pane's foot, near which the palette sits). Rows stand `pitch_px` apart, the column moved into the
    pane; the labels centred on one line (a label's width does not move its row when the eye turns), the widest
    `gap_px` left of the leftmost mark; a leader runs from its mark to `stub_px` right of the widest label, then level
    to `end_px` from its own label's edge, never through a label."""
    pixels, depth = to_screen(marks, view)
    led = sorted(led, key=lambda row: pixels[row[1], 1])
    rows = led + [(text, None) for text in rest] + [(footnote, None)] * bool(footnote and led)
    centred = len(led) or len(rows)   # the led rows on their marks, else every row on every mark
    y = pixels[[k for _, k in led] or slice(None), 1].mean() + (np.arange(len(rows)) - (centred - 1) / 2) * layout[
        "pitch_px"]
    inside = layout["box_px"] / 2 + layout["margin_px"]
    y += max(min(0.0, view["pane_px"][1] - inside - y.max()), inside - y.min())   # into the pane, the top first
    widths = np.array([label_px(text, layout) for text, _ in rows])
    edge, near = pixels[:, 0].min() - layout["gap_px"], depth.mean() - layout["toward_eye_m"]
    points = from_screen(np.column_stack([np.full(len(rows), edge - widths.max() / 2),
                                          y - layout["top_px"] - layout["box_px"] / 2]), near, view)
    ends = [[edge + layout["end_px"] + layout["stub_px"], edge - (widths.max() - width) / 2 + layout["end_px"]]
            for width in widths[:len(led)]]
    return rows, points, {k: from_screen(np.column_stack([ends[n], [y[n]] * 2]), near, view)
                          for n, (_, k) in enumerate(led)}


def _rows(caps: list[dict], highlight: str | None) -> tuple[list, dict[str, str]]:
    """(led: (text, cap index) per cap loaded or next; rest: {text: state} per group of the other caps by what their
    labels say, in slot order)."""
    led = [(("next dip: " if c["slot"] == highlight else "") + c["text"], k) for k, c in enumerate(caps)
           if c["state"] == "loaded" or c["slot"] == highlight]
    groups: dict[str, list[dict]] = {}
    for c in caps:
        if c["state"] != "loaded" and c["slot"] != highlight:
            groups.setdefault(c["text"].removeprefix(c["alias"] + " "), []).append(c)
    return led, {" ".join(c["alias"] for c in group) + f" {said}": group[0]["state"] for said, group in groups.items()}


def palette_layers(caps: list[dict], base_from_palette, highlight: str | None = None) -> dict[str, tuple[str, dict]]:
    """{entity under the palette: (rerun archetype, its arguments)}, in palette_root, each cap by its state (CAP_RGB;
    widths in UI points, so they hold at the scene's scale):
      caps    a ring round each cap's outside, full height: loaded bold, the current or next dip's slot bolder in
              NEXT_RGB; no cap a flat ring where its rim would be
      ink     each loaded cap's ink as a solid column from its inner floor up to the level recorded at load
      marks   each cap's mark `lift_m` over its rim (the ring and ink can hide under the D555 cloud; this cannot)
      labels  in a column (label_column): each loaded or next cap's own row (alias, ink, level, fill, the run's dips),
              the others grouped by what they say (`M1 S1 S2 M2 L2 empty`), then CAP_FOOTNOTE
      leaders each cap's stem from its rim to its mark, and on from a loaded or next cap's mark to its row"""
    def rgb(cap):
        return NEXT_RGB if cap["slot"] == highlight else CAP_RGB[cap["state"]]

    pose = np.asarray(base_from_palette, float)
    marks = [c["rim"] + [0.0, 0.0, CAP_LABELS["lift_m"]] for c in caps]
    led, rest = _rows(caps, highlight)
    rows, points, leaders = label_column(transform(pose, marks), led, list(rest), CAP_FOOTNOTE)
    to_palette = np.linalg.inv(pose)
    loaded = [c for c in caps if c["state"] == "loaded"]
    return {
        "caps": ("Cylinders3D", {
            "centers": [c["rim"] - [0.0, 0.0, c["height"] / 2] for c in caps],
            "radii": [c["radius"] + 0.0005 for c in caps], "lengths": [max(c["height"], 0.0002) for c in caps],
            "colors": [rgb(c) for c in caps], "fill_mode": "majorwireframe",
            "line_radii": [-1.5 if c["slot"] == highlight else -1.0 if c["state"] == "loaded" else -0.6 for c in caps]}),
        "ink": ("Cylinders3D", {
            "centers": [c["floor"] + [0.0, 0.0, c["level"] / 2] for c in loaded], "lengths": [c["level"] for c in loaded],
            "radii": [c["inner_radius"] for c in loaded], "colors": [c["rgb"] for c in loaded], "fill_mode": "solid"}),
        "marks": ("Points3D", {
            "positions": marks, "colors": [rgb(c) for c in caps],
            "radii": [MARK_RADII["next" if c["slot"] == highlight else c["state"]] for c in caps]}),
        "labels": ("Points3D", {
            "positions": transform(to_palette, points), "radii": -0.5, "labels": [text for text, _ in rows],
            "colors": [rgb(caps[k]) if k is not None else CAP_RGB[rest.get(text, "empty")] for text, k in rows],
            "show_labels": True}),
        "leaders": ("LineStrips3D", {
            "strips": [[c["rim"], mark, *transform(to_palette, leaders[k])] if k in leaders else [c["rim"], mark]
                       for k, (c, mark) in enumerate(zip(caps, marks, strict=True))],
            "radii": -0.75, "colors": [(*rgb(c), 160) for c in caps]}),
    }


# --- the wrist camera -----------------------------------------------------------------------------------------------
WRIST_RANGE_M = (0.07, 0.50)   # the D405's working range: nearer is noise, farther the room beyond the page


def wrist_depth(frames: dict, max_width: int = 320, range_m=WRIST_RANGE_M) -> dict | None:
    """The depth stream of one wrist-tap frame set (frame_set_wire.unpack's frames; '<camera>_depth'), every n-th pixel
    to at most max_width, a pixel outside range_m set to 0 (invalid, not drawn): {stamp_ns, ros_frame, depth (H, W) raw
    units, focal, principal (scaled to it), meter}."""
    name = next((key for key in frames if key.endswith("_depth")), None)
    if name is None:
        return None
    metadata, payload, data = frames[name]
    attributes, width, height = metadata["attributes"], int(payload["width"]), int(payload["height"])
    step = max(1, -(-width // max_width))
    k = json.loads(attributes["intrinsics"])
    scale, units = width / float(k.get("width") or width) / step, float(attributes["depth_units_m"])
    depth = np.frombuffer(data, "<u2").reshape(height, width)[::step, ::step].copy()
    depth[(depth < range_m[0] / units) | (depth > range_m[1] / units)] = 0
    return {"stamp_ns": int(metadata["timestamps"]["normalized_unix_ns"]), "ros_frame": attributes["ros_frame"],
            "depth": depth, "focal": [k["fx"] * scale, k["fy"] * scale],
            "principal": [k["ppx"] * scale, k["ppy"] * scale], "meter": 1.0 / units}


# The wrist cloud's look beside the overhead one (rust/visiond/src/depth_cloud.rs, keep in step): measured RGB with a
# tenth of a warm tint (the D555's is cool), each point `radius_share` of the gap to its neighbours at its range, so
# the cloud reads as a surface. `band` points per Points3D row keep each row the hub streams on within the cockpit's
# 60 KiB (12 B of position and 4 of colour a point); `max_points` bounds the cloud (a 320x180 plane at every 2nd pixel).
WRIST_CLOUD = {"tint": (255, 150, 40), "tint_share": 0.1, "radius_share": 0.6, "band": 3600, "max_points": 14400}


def wrist_cloud(frames: dict, decode=None, max_width: int = 320, range_m=WRIST_RANGE_M, style=WRIST_CLOUD) -> dict | None:
    """wrist_depth's frame plus its returns as an RGB cloud in the camera's optical frame: `points` (N, 3) float32
    metres (Rerun's pixel-centre rays under the same pinhole), `colors` (N, 3) uint8 sampled from the tap's colour at
    the same image position and tinted, and `bands`, (points, colors, radius m) rows of at most style['band'] points (a
    fixed count, so a smaller cloud's leftover rows come back empty and clear the last one's), each radius
    style['radius_share'] of the gap between neighbours at the row's mean range. `decode(jpeg bytes)` returns
    an (H, W, 3) RGB array or None; without it, or without a colour frame, every point takes the tint alone. A display
    derivative: no lens distortion (scripts/lib/rgbd_geometry.py owns deprojection)."""
    frame = wrist_depth(frames, max_width, range_m)
    if frame is None:
        return None
    depth = frame["depth"]
    height, width = depth.shape
    stride = max(1, int(np.ceil(np.sqrt(np.count_nonzero(depth) / style["max_points"]))))
    ys, xs = np.nonzero(depth[::stride, ::stride])
    ys, xs = ys[:style["max_points"]] * stride, xs[:style["max_points"]] * stride   # a lumpy plane can still overshoot
    z = depth[ys, xs].astype(np.float64) / frame["meter"]
    (fx, fy), (px, py) = frame["focal"], frame["principal"]
    points = np.stack([(xs + 0.5 - px) / fx * z, (ys + 0.5 - py) / fy * z, z], axis=1).astype(np.float32)
    tint = np.asarray(style["tint"], np.float64)
    rgb = _wrist_rgb(frames, decode)
    if rgb is None:
        measured = np.broadcast_to(tint, (len(z), 3))
    else:
        ch, cw = rgb.shape[:2]
        cx = np.minimum(((xs + 0.5) * cw / width).astype(int), cw - 1)
        cy = np.minimum(((ys + 0.5) * ch / height).astype(int), ch - 1)
        measured = rgb[cy, cx].astype(np.float64)
    share = float(style["tint_share"])
    colors = np.round(measured * (1.0 - share) + tint * share).astype(np.uint8)
    band, gap_per_m = int(style["band"]), stride * 2.0 / (fx + fy)
    radius = [float(style["radius_share"] * gap_per_m * z[i * band:(i + 1) * band].mean()) if i * band < len(z) else 0.0
              for i in range(-(-int(style["max_points"]) // band))]
    bands = [(points[i * band:(i + 1) * band], colors[i * band:(i + 1) * band], r) for i, r in enumerate(radius)]
    return {**frame, "points": points, "colors": colors, "bands": bands}


def _wrist_rgb(frames: dict, decode):
    """The tap's colour as (H, W, 3) RGB, or None: no decoder, no '<camera>_color' JPEG, or one that will not decode."""
    name = next((key for key in frames if key.endswith("_color")), None)
    if decode is None or name is None:
        return None
    try:
        rgb = decode(bytes(frames[name][2]))
    except Exception:  # noqa: BLE001  (a bad JPEG costs the colours, never the cloud)
        return None
    if rgb is None or getattr(rgb, "ndim", 0) != 3 or rgb.shape[2] != 3 or 0 in rgb.shape[:2]:
        return None
    return rgb
