"""Coded prints in live observation and replay.

A coded flower-of-life print (`scripts/lib/stencil_coded.py`) is found by
decoding its lattice, never by matching its appearance. The decode is the
anchor: it names the print and seeds the same optical-flow track a SIFT
acquisition seeds. Between decodes, every tracked frame re-reads the print's
own bits at the tracked junctions, so the print's identity is verified in the
frame that carries its pose, as a printed instance mark is. A decode that is
rejected or ambiguous, or bits that do not verify, read as lost.

A decode costs seconds on a laptop and tens of seconds on the camera node on a
cluttered real frame, so it runs only inside search regions (the drawing areas
projected into a view, and boxes around prints the view already tracks). In
the live observer it runs in one background process, one view at a time, and
never holds up a turn: its junctions are carried to the view's newest frame by
optical flow and must pass that frame's bit check, so a print that moved or
went away meanwhile reads lost and is searched again. Replay decodes in line.
"""

from __future__ import annotations

import json
import multiprocessing
import time
from concurrent.futures import ProcessPoolExecutor
from concurrent.futures.process import BrokenProcessPool
from dataclasses import replace
from pathlib import Path

import cv2
import numpy as np
import stencil_reference
from stencil_bench_trackers import ACCEPTED, AMBIGUOUS
from stencil_coded_tracker import CodedTracker, log10_tail, verify_bits
from stencil_features import Settings, competing_support, fit
from stencil_tracking import StencilTracker

# Per-frame bit verification: decisive edges read, their significance against chance (as a
# decode's attachment), and the share that must agree with the page code.
VERIFY_MIN_EDGES = 8
VERIFY_SIGNIFICANCE = -3.0
VERIFY_AGREEMENT = .85
# A tracked page's search box grows by this share of its size on every side.
TRACKED_PAD = .25
MAX_REGIONS = 3
CODED_DEFERRED = "coded_decode_deferred"
CODED_PENDING = "coded_decode_pending"
# Reasons a view waits for a decode rather than having searched and missed.
WAITING = frozenset({CODED_DEFERRED, CODED_PENDING})
# A background decode older than this is dropped unused: its view moved on or went away.
STALE_JOB_S = 120.
# A print a view tracked and lost (an arm over it for its draw) is looked for first where it was:
# that box alone decodes in seconds, where the drawing pads' squares take 1-2 min a view and one
# view at a time, so a full sweep of the views took 20-40 min (bench 2026-09-27). Every
# HINT_EVERY-th search of such a view is still the full one, for a print moved elsewhere, and a
# hint older than HINT_TTL_NS is dropped. A hint search goes before any other view's full one.
HINT_TTL_NS = 30*60*10**9
HINT_EVERY = 4
URGENT_HOLD_S = 5.
# Refusals a consumer should see by name: the print was seen but not safely identified.
REFUSALS = frozenset({"coded_bits_disagree", "coded_contested_decode", "coded_components_disagree",
                      "coded_page_bits_disagree", "coded_spatial_conflict", "coded_duplicate_decode"})


class _Artwork:
    """What `CodedTracker.prepare` reads: the pattern and the verified tracking.json path."""

    def __init__(self, pattern_id, reference):
        self.pattern_id, self.reference = pattern_id, Path(reference)


class CodedBank:
    """Decoders for the coded references of one scene. Prints whose decoder settings differ
    (lattice spacing, knot size, bit style) get separate decoders; each searched image runs
    every decoder once over its search regions.

    `decodes_per_turn` bounds the in-line decodes between `begin_turn` calls (None:
    unbounded); `whole_image_px` bounds the frame size searched whole when no region is given
    (None: any size); `decode_s` bounds one image's decode in wall-clock seconds (None:
    unbounded): past it no new component is searched for, and what was found is still decoded.
    `background` decodes in one worker process instead (`request`)."""

    def __init__(self, paths, settings=None, *, decodes_per_turn=None, whole_image_px=None, decode_s=None,
                 background=False):
        self.settings = settings or Settings()
        self.references, self.books, self.modes = {}, {}, {}
        self.decoders = []
        groups = {}
        for path in paths:
            manifest, _ = stencil_reference.load(path)
            if not stencil_reference.is_coded(manifest):
                raise ValueError("a coded bank holds coded prints only")
            pattern = manifest["pattern_id"]
            if pattern in self.references:
                raise ValueError("duplicate stencil pattern in reference bank")
            code = json.loads(stencil_reference.coded_path(path).read_text())
            key = json.dumps({"geometry": code["geometry"], "style": code.get("style")}, sort_keys=True)
            groups.setdefault(key, []).append(_Artwork(pattern, path))
            self.references[pattern] = manifest
        for artworks in groups.values():
            decoder = CodedTracker(substrate="any")
            decoder.prepare(artworks)
            self.decoders.append(decoder)
            for book in decoder.codebooks:
                self.books[book.pattern_id], self.modes[book.pattern_id] = book, decoder.mode
        self.decodes_per_turn, self.whole_image_px, self.decode_s = decodes_per_turn, whole_image_px, decode_s
        self.spent = 0
        self.paths, self.background = [str(path) for path in paths], background
        self.pool, self.jobs = None, {}
        self.urgent_at = -1e9

    def begin_turn(self):
        self.spent = 0

    def affordable(self):
        return self.decodes_per_turn is None or self.spent < self.decodes_per_turn

    def decode(self, image, regions):
        """{pattern: (found | None, reason)} for one image, decoded in line. `regions` are
        pixel boxes (x0, y0, x1, y1), or None for the whole image. Only a decode that runs
        counts against the turn's budget."""
        crops = self.crops(image, regions)
        if not crops:
            return dict.fromkeys(self.references, (None, "coded_no_search_region"))
        self.spent += 1
        return self._found(*self.locate_crops(crops), image.shape)

    def crops(self, image, regions):
        """[((x0, y0), crop)] to decode: the regions clipped and merged, or the whole image when
        `regions` is None and it is small enough."""
        h, w = image.shape[:2]
        if regions is None:
            if self.whole_image_px is not None and h*w > self.whole_image_px:
                return []
            regions = [(0, 0, w, h)]
        regions = merge_boxes([clip_box(box, w, h) for box in regions if clip_box(box, w, h)])[:MAX_REGIONS]
        return [((x0, y0), np.ascontiguousarray(image[y0:y1, x0:x1])) for x0, y0, x1, y1 in regions]

    def locate_crops(self, crops):
        """The decode itself: ({pattern: [(uv, image pixels, extra)]}, {pattern: refusal})."""
        deadline = None if self.decode_s is None else time.perf_counter()+self.decode_s
        results = {pattern: [] for pattern in self.references}
        refused = {}
        for offset, crop in crops:
            if crop.ndim == 2:
                crop = cv2.cvtColor(crop, cv2.COLOR_GRAY2BGR)
            for decoder in self.decoders:
                found, failed = decoder.locate_all(crop, None, deadline)
                for book in decoder.codebooks:
                    located = found.get(book.pattern_id)
                    if located is not None and located.status == ACCEPTED:
                        results[book.pattern_id].append((located.uv, located.pixels+offset, dict(located.extra)))
                    elif located is not None and located.status == AMBIGUOUS:
                        refused[book.pattern_id] = "coded_"+located.reason
                    else:
                        source = located if located is not None else failed
                        refused.setdefault(book.pattern_id, "coded_"+(source.reason if source else "not_found"))
        return results, refused

    def request(self, image, regions, key, urgent=False):
        """The background decode for one view (`key`): its finished result, carried to this
        frame, else (None, reason) per print — pending while this view's decode runs, deferred
        while another view's does. Starting a decode costs this turn only the crops' copy.
        An `urgent` request (a lost print's own box) goes before the other views' searches:
        they defer while one asked within URGENT_HOLD_S."""
        now = time.monotonic()
        if urgent:
            self.urgent_at = now
        job = self.jobs.get(key)
        if job is not None and job["future"].done():
            del self.jobs[key]
            return self._finished(job, image)
        if job is not None:
            return dict.fromkeys(self.references, (None, CODED_PENDING))
        self._expire()
        if any(not other["future"].done() for other in self.jobs.values()):
            return dict.fromkeys(self.references, (None, CODED_DEFERRED))
        if not urgent and now-self.urgent_at < URGENT_HOLD_S:
            return dict.fromkeys(self.references, (None, CODED_DEFERRED))
        crops = self.crops(image, regions)
        if not crops:
            return dict.fromkeys(self.references, (None, "coded_no_search_region"))
        try:
            if self.pool is None:
                self.pool = ProcessPoolExecutor(1, mp_context=multiprocessing.get_context("spawn"),
                                                initializer=_worker_start, initargs=(self.paths, self.decode_s))
            future = self.pool.submit(_worker_locate, crops)
        except BrokenProcessPool:
            # The worker died (it is started again for the next search); this view reads lost.
            self.pool = None
            return dict.fromkeys(self.references, (None, "coded_decode_failed: BrokenProcessPool"))
        self.jobs[key] = {"future": future, "gray": gray_of(image), "started": time.monotonic()}
        return dict.fromkeys(self.references, (None, CODED_PENDING))

    def _expire(self):
        now = time.monotonic()
        for key in [key for key, job in self.jobs.items() if job["future"].done() and now-job["started"] > STALE_JOB_S]:
            del self.jobs[key]

    def _finished(self, job, image):
        try:
            results, refused = job["future"].result()
        except Exception as error:  # noqa: BLE001 - a failed worker reads lost and is searched again
            if isinstance(error, BrokenProcessPool):
                self.pool = None
            return dict.fromkeys(self.references, (None, f"coded_decode_failed: {type(error).__name__}"))
        results = {pattern: carried(hits, job["gray"], gray_of(image)) for pattern, hits in results.items()}
        return self._found(results, refused, image.shape)

    def close(self):
        if self.pool is not None:
            self.pool.shutdown(wait=False, cancel_futures=True)
            self.pool = None

    def _found(self, results, refused, shape):
        out = {}
        for pattern, hits in results.items():
            if pattern in refused and refused[pattern] in REFUSALS:
                out[pattern] = (None, refused[pattern])
            elif not hits:
                out[pattern] = (None, refused.get(pattern, "coded_not_found"))
            else:
                out[pattern] = self._fit(pattern, hits, shape)
        accepted = [pattern for pattern, (found, _) in out.items() if found is not None]
        for index, first in enumerate(accepted):
            for second in accepted[index+1:]:
                if competing_support(out[first][0], out[second][0]):
                    out[first] = out[second] = (None, "coded_spatial_conflict")
        return out

    def _fit(self, pattern, hits, shape):
        """The decoded junctions as a tracking acquisition: `fit`'s homography, gates and
        inliers, with a reprojection gate scaled to the decoded lattice (the decoder's own)."""
        hits = [hit for hit in hits if len(hit[0])]
        if not hits:
            return None, "coded_decode_moved"
        uv = np.round(np.concatenate([uv for uv, _, _ in hits]), 9)
        pixels = np.concatenate([pixels for _, pixels, _ in hits])
        # Search regions never overlap, so one page junction decoded in two of them is two
        # physical copies of this print.
        uv, unique = np.unique(uv, axis=0, return_index=True)
        if len(unique) < len(pixels):
            return None, "coded_duplicate_decode"
        pixels = pixels[unique]
        spacing = lattice_spacing_px(pixels)
        error_px = max(self.settings.max_error_px, .3*spacing)
        found = fit(uv, pixels, shape, replace(self.settings, max_error_px=error_px))
        if found is None:
            return None, "coded_decode_unsupported_geometry"
        if len(hits) > 1 and found["inliers"] < .9*len(uv):
            return None, "coded_duplicate_decode"
        best = min((extra for _, _, extra in hits), key=lambda extra: extra.get("log10_chance", 0.))
        found.update(pattern_id=pattern, mirrored=bool(best.get("mirrored")), error_px=error_px,
                     print_id=best.get("print_id"),
                     decode={key: best.get(key) for key in ("log10_chance", "page_bits_agree",
                                                            "page_bits_disagree", "components")})
        return found, "coded_decoded"


_WORKER = {}


def _worker_start(paths, decode_s):
    cv2.setNumThreads(1)
    _WORKER["bank"] = CodedBank(paths, decode_s=decode_s)


def _worker_locate(crops):
    return _WORKER["bank"].locate_crops(crops)


def gray_of(image):
    return image if image.ndim == 2 else cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


def carried(hits, before, after):
    """Decoded junctions moved from the frame the decode read to this one by forward-backward
    Lucas–Kanade; a junction flow cannot follow is dropped (its UV with it)."""
    if before.shape != after.shape:
        return [(uv[:0], pixels[:0], extra) for uv, pixels, extra in hits]
    if before is after or np.array_equal(before, after):
        return hits
    moved = []
    for uv, pixels, extra in hits:
        start = np.asarray(pixels, np.float32).reshape(-1, 1, 2)
        forward, ok, _ = cv2.calcOpticalFlowPyrLK(before, after, start, None, winSize=(21, 21), maxLevel=3)
        back, ok_back, _ = cv2.calcOpticalFlowPyrLK(after, before, forward, None, winSize=(21, 21), maxLevel=3)
        keep = (ok.ravel() == 1) & (ok_back.ravel() == 1) & (np.linalg.norm((back-start).reshape(-1, 2), axis=1) < .75)
        moved.append((np.asarray(uv)[keep], forward.reshape(-1, 2)[keep].astype(float), extra))
    return moved


class SharedCodedSearch:
    """One decode per image for every coded pattern tracker that needs it."""

    def __init__(self, bank):
        self.bank = bank
        self.key, self.image, self.regions, self.result, self.source = None, None, None, None, None
        self.urgent = False

    def begin(self, image, timestamp_ns, regions, source=None, urgent=False):
        key = (id(image), int(timestamp_ns))
        if key != self.key:
            self.key, self.image, self.regions, self.result, self.source = key, image, regions, None, source
            self.urgent = urgent

    def affordable(self):
        return self.result is not None or self.bank.background or self.bank.affordable()

    def detect(self, pattern):
        if self.result is None:
            self.result = (self.bank.request(self.image, self.regions, self.source, self.urgent)
                           if self.bank.background else self.bank.decode(self.image, self.regions))
        if pattern is not None:
            return self.result[pattern]
        found = [value for value in self.result.values() if value[0] is not None]
        if len(found) > 1:
            return None, "coded_spatial_conflict"
        if found:
            return found[0]
        reasons = sorted({reason for _, reason in self.result.values()})
        return None, reasons[0] if len(reasons) == 1 else ",".join(reasons)


class CodedPatternBank:
    """The per-tracker view of a shared coded search (`pattern` None: any coded print)."""

    def __init__(self, search, pattern=None):
        self.search, self.pattern = search, pattern
        self.references, self.settings = search.bank.references, search.bank.settings

    def detect(self, _gray):
        return self.search.detect(self.pattern)


class CodedStencilTracker(StencilTracker):
    """One coded print instance. Its decode seeds the flow track; every tracked frame
    re-reads the print's bits at the tracked junctions."""

    def __init__(self, references, instance_id, settings=None, *, bank=None, pattern_id=None):
        if bank is None:
            bank = CodedPatternBank(SharedCodedSearch(CodedBank(references, settings)))
        super().__init__([], instance_id, bank=bank, pattern_id=pattern_id)
        # The bits re-read every frame verify the print; a live track is not re-decoded.
        self.settings = self.base_settings = replace(self.settings, verify_interval_ms=10**15)
        self.image, self.regions = None, None
        self.last_polygon, self.bits, self.decoded = None, None, None
        self.hint_polygon, self.hint_ns, self.hint_searches = None, None, 0

    def tracked_box(self):
        """The search box around this print's last tracked page, or None."""
        return polygon_box(self.last_polygon, TRACKED_PAD) if self.last_polygon is not None else None

    def hint_box(self, timestamp_ns):
        """The box around where this view last tracked the print, while it is lost: its next
        search is that box alone (None: search the view's regions, every HINT_EVERY-th search
        of a lost print, or once the hint is HINT_TTL_NS old)."""
        if self.active is not None or self.hint_polygon is None:
            return None
        if int(timestamp_ns)-self.hint_ns > HINT_TTL_NS:
            self.hint_polygon = None
            return None
        if self.hint_searches % HINT_EVERY == HINT_EVERY-1:
            return None
        return polygon_box(self.hint_polygon, TRACKED_PAD)

    def observe(self, image, timestamp_ns, *, regions=None, **kwargs):
        """`regions`: pixel boxes a decode may search besides this print's own last page (None:
        the whole image). A scene begins its shared search with every tracker's box first."""
        self.image, self.bits, self.decoded = image, None, None
        own = self.tracked_box()
        self.bank.search.begin(image, timestamp_ns, None if regions is None else [*regions, *([own] if own else [])],
                               kwargs.get("source_id"))
        result = super().observe(image, timestamp_ns, **kwargs)
        if self.active is not None:
            self.last_polygon = np.asarray(self.active["polygon"], float)
            self.hint_polygon, self.hint_ns, self.hint_searches = self.last_polygon, int(timestamp_ns), 0
        result["image_model"] = "coded_junction_homography"
        result["coded"] = {"bits_agree": self.bits[0] if self.bits else None,
                           "bits_disagree": self.bits[1] if self.bits else None,
                           "decoded_this_frame": self.decoded is not None, "decode": self.decoded}
        return result

    def _advance(self, gray, stamp, *, defer_search=False):
        deferred = not defer_search and not self.bank.search.affordable()
        active, status, reason = super()._advance(gray, stamp, defer_search=defer_search or deferred)
        if deferred and active is None:
            reason = CODED_DEFERRED
        return active, status, reason

    def _search(self, gray, stamp):
        found, reason = super()._search(gray, stamp)
        if found is not None:
            self.settings = replace(self.base_settings, max_error_px=found["error_px"])
            self.decoded = found["decode"]
        elif reason not in WAITING:
            if reason != "coded_no_search_region":
                self.last_polygon = None    # its old page was searched and it is not there
            self.hint_searches += 1     # the hint stays: an arm over the print hides it for minutes
        return found, reason

    def _identity(self, gray, reference):
        pattern = self.pattern_id
        search = self.bank.search.bank
        agree, disagree = verify_bits(self.image, self.active["uv"], self.active["pixels"], search.books[pattern],
                                      search.modes[pattern], self.active["homography"])
        self.bits = (int(agree), int(disagree))
        read = agree+disagree
        if read < VERIFY_MIN_EDGES:
            return None, "coded_bits_unreadable"
        if log10_tail(read, agree) > VERIFY_SIGNIFICANCE or agree < VERIFY_AGREEMENT*read:
            return None, "coded_bits_disagree"
        return search.books[pattern].print_id, None


def lattice_spacing_px(pixels):
    """Median distance from each junction to its nearest other junction: one lattice step."""
    pixels = np.asarray(pixels, np.float32)
    if len(pixels) < 2:
        return 0.
    distance = cv2.batchDistance(pixels, pixels, cv2.CV_32F, normType=cv2.NORM_L2, K=2)[0]
    return float(np.median(distance[:, 1]))


def polygon_box(polygon, pad):
    polygon = np.asarray(polygon, float)
    low, high = polygon.min(0), polygon.max(0)
    grow = (high-low)*pad
    return tuple(int(v) for v in (*np.floor(low-grow), *np.ceil(high+grow)))


def clip_box(box, width, height):
    x0, y0, x1, y1 = (int(round(v)) for v in box)
    x0, y0, x1, y1 = max(0, x0), max(0, y0), min(width, x1), min(height, y1)
    return (x0, y0, x1, y1) if x1-x0 >= 32 and y1-y0 >= 32 else None


def merge_boxes(boxes):
    """Overlapping boxes as their union, largest first, so no pixel is decoded twice."""
    merged = []
    for box in sorted(boxes, key=lambda b: -(b[2]-b[0])*(b[3]-b[1])):
        for index, other in enumerate(merged):
            if box[0] < other[2] and other[0] < box[2] and box[1] < other[3] and other[1] < box[3]:
                merged[index] = (min(box[0], other[0]), min(box[1], other[1]),
                                 max(box[2], other[2]), max(box[3], other[3]))
                break
        else:
            merged.append(box)
    return merged


def project_areas(areas, view_from_root, project, shape, min_px_per_mm, samples=13):
    """Pixel boxes of search areas (root-frame quadrilaterals, metres) in one view: `project`
    maps camera-frame points to pixels, NaN where the camera model has no ray. Each area is
    sampled on a grid, so a view that sees part of it (a close wrist camera) keeps that part;
    an area wholly unseen, or seen too coarsely for a decode (adjacent samples under
    `min_px_per_mm` apart), gives no box."""
    boxes = []
    h, w = shape[:2]
    t = np.linspace(0., 1., samples)
    for polygon in areas:
        corners = np.asarray(polygon, float)
        grid = (corners[0]+t[:, None, None]*(corners[1]-corners[0])+t[None, :, None]*(corners[3]-corners[0]))
        step_mm = 1000*np.linalg.norm(corners[1]-corners[0])/(samples-1)
        points = grid.reshape(-1, 3)@view_from_root[:3, :3].T+view_from_root[:3, 3]
        pixels = np.full((len(points), 2), np.nan)
        ahead = points[:, 2] > 1e-3
        if ahead.any():
            pixels[ahead] = np.asarray(project(points[ahead]), float)
        pixels = pixels.reshape(samples, samples, 2)
        seen = np.isfinite(pixels).all(-1) & (pixels[..., 0] >= 0) & (pixels[..., 0] < w) \
            & (pixels[..., 1] >= 0) & (pixels[..., 1] < h)
        if not seen.any():
            continue
        # Pixel distance between adjacent samples both in view: the local resolution.
        gaps = np.concatenate([np.linalg.norm(np.diff(pixels, axis=0), axis=-1)[seen[1:] & seen[:-1]],
                               np.linalg.norm(np.diff(pixels, axis=1), axis=-1)[seen[:, 1:] & seen[:, :-1]]])
        if not len(gaps) or np.median(gaps)/step_mm < min_px_per_mm:
            continue
        inside = pixels[seen]
        pad = float(np.median(gaps))
        box = clip_box((*(inside.min(0)-pad), *(inside.max(0)+pad)), w, h)
        if box is not None:
            boxes.append(box)
    return boxes


def drawing_pads(repo):
    """{arm: [pivot x, pivot y, paper plane z]} in each arm's base frame, metres, for the arms
    whose `config/workspace.yaml` section has a pad touch-off pivot and a paper plane."""
    from tool_spec import read_workspace

    keys = ("pivot_point_x", "pivot_point_y", "paper_plane_z")
    pads = {}
    for arm, section in read_workspace(repo).items():
        if not isinstance(section, dict):
            continue
        values = [section.get(key) for key in keys]
        if all(type(value) in (int, float) for value in values):
            pads[arm] = np.asarray(values, dtype=float)
    return pads


def pad_area(world_from_base, pad, half_m):
    """The square `half_m` either side of a pad pivot on its paper plane, in the world."""
    corners = pad+half_m*np.array([[-1., -1., 0.], [1., -1., 0.], [1., 1., 0.], [-1., 1., 0.]])
    return corners@world_from_base[:3, :3].T+world_from_base[:3, 3]


def coded_print_measure(image, program, reference, ink_alignment, *, px_per_mm=10.0, decode_s=45.0, regions=None):
    """Measure ink in independently decoded print coordinates using the existing ink scorer.

    The camera/TCP pose does not register this image. Nominal print dimensions,
    held-out lattice error and raster sampling bound interpretation; forward gaps
    do not measure tip accuracy or penalize every stray mark. A failed decode or
    ink search is unavailable evidence, never proof that the pen left no ink.
    The caller supplies the existing vision modules on its import path.
    """
    import cv2
    from stencil_coded_tracker import verify_bits

    bank = CodedBank([reference], decode_s=decode_s)
    pattern, manifest = next(iter(bank.references.items()))
    expected = (program.get("diagnostic") or {}).get("physical_print_id")
    if expected and expected != manifest["physical_instance_id"]:
        return {"valid": False, "reason": "physical print identity differs from program"}, None
    found, reason = bank.decode(image, regions)[pattern]
    result = {"valid": False, "reason": reason, "pattern_id": pattern,
              "print_id": manifest["physical_instance_id"], "reference_id": manifest["reference_id"],
              "dimensions_measured": manifest["dimensions_measured"]}
    if found is None:
        return result, None
    uv, pixels, homography = found["uv"], found["pixels"], found["homography"]
    agree, disagree = verify_bits(image, uv, pixels, bank.books[pattern], bank.modes[pattern], homography)
    result.update(bits_agree=int(agree), bits_disagree=int(disagree), inliers=int(found["inliers"]))
    if agree < 8 or disagree:
        result["reason"] = "print bits not independently verified without conflicts"
        return result, None
    page_mm = np.asarray(manifest["page_mm"], float)
    errors = []
    for fold in range(4):
        held = np.arange(len(uv)) % 4 == fold
        if (~held).sum() < 4 or not held.any():
            continue
        fit, _ = cv2.findHomography(uv[~held], pixels[~held], 0)
        if fit is not None:
            predicted = cv2.perspectiveTransform(pixels[held][None], np.linalg.inv(fit))[0]
            errors.extend(np.linalg.norm((predicted - uv[held]) * page_mm, axis=1).tolist())
    size = np.rint(page_mm * px_per_mm).astype(int)
    uv_from_pixel = np.array([[1 / size[0], 0, .5 / size[0]],
                              [0, 1 / size[1], .5 / size[1]], [0, 0, 1.]])
    page = cv2.warpPerspective(image, homography @ uv_from_pixel, tuple(size),
                               flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP, borderValue=0)
    half = page_mm / 2000
    ink = ink_alignment(page, program, (-half[0], -half[1], half[0], half[1]),
                        px_per_mm, stencil_reference.page_geometry(manifest)["clear_m"])
    centre_px = cv2.perspectiveTransform(np.array([[[.5, .5]]]), homography)[0, 0]
    pixel_axes = np.array([[centre_px, centre_px + [1., 0.], centre_px + [0., 1.]]])
    pixel_uv = cv2.perspectiveTransform(pixel_axes, np.linalg.inv(homography))[0]
    native_pixel_mm = np.linalg.norm((pixel_uv[1:] - pixel_uv[0]) * page_mm, axis=1).tolist()
    result.update(homography=homography.tolist(), heldout_p95_mm=float(np.percentile(errors, 95)) if errors else None,
                  raster_pixel_mm=1 / px_per_mm, native_pixel_mm=native_pixel_mm, ink=ink, identity_verified=True)
    result["valid"] = bool(ink and ink["found"])
    result["reason"] = "ink alignment candidate; review occlusion and existing marks" if result["valid"] else "ink alignment unavailable on verified print"
    if result['valid']:
        from tatbot_session import fidelity, inspect

        plan = inspect.plan_samples(program) + [ink['dx_m'], ink['dy_m']]
        lo, hi = plan.min(axis=0) - .002, plan.max(axis=0) + .002
        gx, gy = inspect.page_grid((-half[0], -half[1], half[0], half[1]), px_per_mm)
        roi = (gx >= lo[0]) & (gx <= hi[0]) & (gy >= lo[1]) & (gy <= hi[1])
        _, refusal = fidelity._observed(page, roi, px_per_mm)
        if refusal:
            result.update(valid=False, reason=refusal)
    if result["valid"]:
        result["placement_m"] = [ink["dx_m"], ink["dy_m"]]
        result["uncovered_plan_fraction"] = 1 - ink["coverage"]
        result["coverage_tolerance_mm"] = 1.0
    return result, page


def inspect_ink(out_dir, poses, program, pattern, ink_alignment, px_per_mm=10.0):
    """Add coded-print registration to the ROS inspector's existing ink analysis."""
    reference = stencil_reference.observer_references() / pattern / 'tracking.json'
    if not reference.is_file():
        return {}
    rows = []
    try:
        manifest, _ = stencil_reference.load(reference)
        if not stencil_reference.is_coded(manifest):
            return {}
        for n, pose in enumerate(poses):
            frames = [cv2.imread(str(out_dir / f'pose{n}-{j}.png')) for j in range(pose['frames'])]
            raw = np.median(np.stack(frames), axis=0).astype(np.uint8)
            measured, printed = coded_print_measure(raw, program, reference, ink_alignment, px_per_mm=px_per_mm)
            rows.append(measured)
            if printed is not None:
                cv2.imwrite(str(out_dir / f'print{n}.png'), printed)
    except (ValueError, OSError, ImportError, cv2.error) as error:
        rows = [{'valid': False, 'reason': f'coded inspection unavailable: {error}'}]
    good = [row for row in rows if row['valid']]
    if len(good) > 1:
        placements = np.array([row['placement_m'] for row in good])
        noise_mm = max(2.0, 2 * max(row['heldout_p95_mm'] or 0.0 for row in good))
        if np.max(np.linalg.norm(placements[:, None] - placements[None, :], axis=2)) * 1000 > noise_mm:
            for row in good:
                row.update(valid=False, reason='placement candidates disagree between views; inspect occlusion or old ink')
            good = []
    if good:
        d = np.median([row['placement_m'] for row in good], axis=0) * 1e3
        gap = np.median([row['ink']['p95_gap_m'] for row in good]) * 1e3
        missing = np.median([row['uncovered_plan_fraction'] for row in good])
        summary = (f'decoded print: placement candidate {d[0]:+.1f} {d[1]:+.1f} mm; aligned forward gap p95 {gap:.2f} mm; '
                   f'uncovered plan {missing:.0%} at 1 mm ({len(good)} views, nominal print scale)')
        summary += '; single view, visual review required' * (len(good) == 1)
    else:
        summary = 'decoded-print measurement unavailable: ' + '; '.join(sorted({row['reason'] for row in rows}))
    return {'print_coordinates': rows, 'coded_summary': summary}


def write_inspection(out_dir, poses, program, pattern, views, analysis, scorer, clear, extent, px_per_mm):
    """Save both camera-model and decoded-print views of the same inspection."""
    for n, view in enumerate(views):
        cv2.imwrite(str(out_dir / f'page{n}.png'), view)
        cv2.imwrite(str(out_dir / f'overlay{n}.png'), scorer.overlay(view, program, clear, extent, px_per_mm))
    analysis.update(inspect_ink(out_dir, poses, program, pattern, scorer.ink_alignment, px_per_mm))
    (out_dir / 'analysis.json').write_text(json.dumps(analysis, indent=1))
    print(scorer.summary(analysis), flush=True)
