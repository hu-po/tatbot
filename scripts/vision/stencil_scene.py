"""Independent stencil identities sharing one reference search per image.

Legacy artwork is found by SIFT; a coded print (`stencil_coded_live`) by its
decode. Each installed reference is tracked by exactly one of the two.
"""

import time

import cv2
import numpy as np
import stencil_reference
from stencil_coded_live import WAITING, CodedBank, CodedPatternBank, CodedStencilTracker, SharedCodedSearch
from stencil_features import ReferenceBank, Settings
from stencil_tracking import StencilTracker, gray_image

IDLE_RESCAN_NS = 180_000_000_000


class SharedSearch:
    def __init__(self, bank):
        self.bank = bank
        self.result = None

    def detect(self, gray, pattern):
        if self.result is None:
            self.result = self.bank.detect_many(gray)
        matches, reasons = self.result
        if pattern in matches:
            return matches[pattern], "reference_matched"
        return None, reasons[pattern]


class PatternBank:
    def __init__(self, search, pattern):
        self.search, self.pattern = search, pattern
        self.references, self.settings = search.bank.references, search.bank.settings

    def detect(self, gray):
        return self.search.detect(gray, self.pattern)


class SceneBank:
    """Every reference of one scene: legacy artwork matched by SIFT (`sift`), coded prints
    decoded (`coded`). `coded_options` go to the `CodedBank` (its decode budget)."""

    def __init__(self, sift=None, coded=None):
        self.sift, self.coded = sift, coded
        self.settings = (sift or coded).settings
        self.references = {**(sift.references if sift else {}), **(coded.references if coded else {})}

    @classmethod
    def from_paths(cls, paths, settings=None, **coded_options):
        if not 1 <= len(paths) <= 8:
            raise ValueError("provide one to eight stencil references")
        coded = [path for path in paths if stencil_reference.is_coded(stencil_reference.load(path)[0])]
        legacy = [path for path in paths if path not in coded]
        settings = settings or Settings()
        bank = cls(ReferenceBank(legacy, settings, scene=True) if legacy else None,
                   CodedBank(coded, settings, **coded_options) if coded else None)
        if len(bank.references) != len(paths):
            raise ValueError("duplicate stencil pattern in reference bank")
        return bank

    def begin_turn(self):
        if self.coded is not None:
            self.coded.begin_turn()


class StencilScene:
    """One physical copy per supplied pattern; copies cannot be distinguished."""

    def __init__(self, references, instance_id, settings=None, *, bank=None, stable_scene_skip=False):
        if not instance_id or len(instance_id) > 80:
            raise ValueError("multi-stencil instance namespace must be 1–80 characters")
        if bank is None:
            bank = SceneBank.from_paths(references, settings)
        elif isinstance(bank, ReferenceBank):
            bank = SceneBank(sift=bank)
        self.bank = bank
        self.search = SharedSearch(bank.sift) if bank.sift is not None else None
        self.coded_search = SharedCodedSearch(bank.coded) if bank.coded is not None else None
        self.stable_scene_skip = stable_scene_skip
        self.trackers = {}
        for pattern in bank.sift.references if bank.sift is not None else ():
            self.trackers[pattern] = StencilTracker([], instance_id+"/"+pattern, bank=PatternBank(self.search, pattern),
                                                    pattern_id=pattern)
        for pattern in bank.coded.references if bank.coded is not None else ():
            self.trackers[pattern] = CodedStencilTracker(
                [], instance_id+"/"+pattern, bank=CodedPatternBank(self.coded_search, pattern), pattern_id=pattern)
        self.search_thumbnail = None
        self.search_stamp = None
        self.search_source = None

    def coded_hint(self, timestamp_ns):
        """Whether a coded print this view lost has a place to be looked for first: such a view
        searches every turn, its small box going before the other views' full searches."""
        return any(tracker.hint_box(timestamp_ns) is not None for tracker in self.trackers.values()
                   if isinstance(tracker, CodedStencilTracker))

    def observe(self, image, timestamp_ns, *, periodic_search=True, search=True, regions=None, **kwargs):
        """Track every pattern; run the shared reference search only when the
        image changed (or its periodic rescan is due) and `search` allows it —
        a caller defers the search for a print it knows is lost, keeping the
        flow tracks of the others. A coded print's decode searches `regions`
        (pixel boxes; None: the whole image) and every tracked coded page."""
        started = time.perf_counter()
        if self.search is not None:
            self.search.result = None
        if self.coded_search is not None:
            coded = [tracker for tracker in self.trackers.values() if isinstance(tracker, CodedStencilTracker)]
            boxes = [box for box in (tracker.tracked_box() for tracker in coded) if box]
            hints = [box for box in (tracker.hint_box(timestamp_ns) for tracker in coded) if box]
            if hints:   # a print this view lost is looked for where it was, first
                self.coded_search.begin(image, timestamp_ns, [*boxes, *hints], kwargs.get('source_id'), urgent=True)
            else:
                self.coded_search.begin(image, timestamp_ns, None if regions is None else [*regions, *boxes],
                                        kwargs.get('source_id'))
        gray = gray_image(image)
        thumbnail = cv2.resize(gray, (160, 90), interpolation=cv2.INTER_AREA)
        source = (kwargs.get('source_id'), gray.shape)
        stable = (self.stable_scene_skip and self.search_thumbnail is not None and self.search_source == source
                  and timestamp_ns > self.search_stamp
                  and np.count_nonzero(cv2.absdiff(thumbnail, self.search_thumbnail) > 40)
                  <= thumbnail.size*.0025)
        periodic_due = stable and timestamp_ns-self.search_stamp >= IDLE_RESCAN_NS
        defer_search = (stable and (not periodic_due or not periodic_search)) or not search
        searched = (self.search_thumbnail, self.search_stamp, self.search_source)
        if not defer_search:
            self.search_thumbnail, self.search_stamp, self.search_source = thumbnail, timestamp_ns, source
        rows = [tracker.observe(image, timestamp_ns, defer_search=defer_search, regions=regions, **kwargs)
                if isinstance(tracker, CodedStencilTracker) else
                tracker.observe(gray, timestamp_ns, defer_search=defer_search, **kwargs)
                for tracker in self.trackers.values()]
        if any(row['reason'] in WAITING for row in rows):
            # A decode budget or a decode still running, not this image, held the search
            # back: an unchanged image must still be searched next turn.
            self.search_thumbnail, self.search_stamp, self.search_source = searched
        return {"schema": "tatbot.stencil-scene/1", "capture_timestamp_ns": int(timestamp_ns),
                "stencils": rows, "image_tracking_valid": any(row["image_tracking_valid"] for row in rows),
                "geometry_valid": False, "motion_authority": False,
                "processing_ms": (time.perf_counter()-started)*1000}


def objects(observation):
    return observation.get("stencils", [observation])
