"""Ink, inkcaps, palette, and consumables — the registry and the charge model.

    ink catalog     config/inks.yaml          hand-written
    palette         config/palette.yaml       hand-written, rarely
    palette load    config/palette_load.yaml  written by scripts/ink.py (session fact)
    inventory       config/inventory.yaml     written by scripts/ink.py (reconciled snapshot)
    ledger          ~/tatbot-logs/ink/ledger.jsonl   append-only, every dip/stroke/fill/weigh

Stdlib-only for the same reason ``tool_spec.py`` is: the sim is a separate uv
project and loads this file by path. The tool's ink POLICY lives in its
datasheet as an ``ink:`` block and is read from ``ToolSpec.raw`` here, so
``tool_spec`` does not import this module and nothing is circular.

Design:

* The tool carries a CHARGE in microlitres. Contact debits it —
  ``deposit_ul_per_mm * contact_mm + bleed_ul_per_s * contact_s``, where
  contact time counts whether or not the tip is moving, because a needle
  parked on skin still loses ink. A dip credits it and debits the cap.
* A dip is planned when the next stroke's need exceeds the charge, on colour
  change, or at session start, and the plan says WHY.
* ``mode`` decides what the same arithmetic touches: ``real`` debits caps and
  stock; ``rehearsal`` (the ballpoint) runs the identical path with every event
  tagged and nothing real mutated, dry caps allowed; ``none`` (the laser)
  refuses to plan a dip at all.
* Dip depth follows the cap's fill level, so an emptying cap gets a deeper
  plunge rather than a fixed one.
"""

from __future__ import annotations

import importlib.util
import json
import math
import os
import socket
import sys
import uuid
from dataclasses import dataclass, field, replace
from datetime import datetime, timezone
from pathlib import Path

# --- sibling import that also works when loaded by path ------------------------

try:  # scripts/lib on sys.path (scripts, tests)
    import tool_spec as _tool_spec
except ImportError as err:  # loaded by path from the sim package
    _tool_spec = sys.modules.get("tatbot_tool_spec")
    if _tool_spec is None:
        _p = Path(__file__).resolve().with_name("tool_spec.py")
        _s = importlib.util.spec_from_file_location("tatbot_tool_spec", _p)
        if _s is None or _s.loader is None:
            raise ImportError(f"Cannot load module spec for {_p}") from err
        _tool_spec = importlib.util.module_from_spec(_s)
        sys.modules["tatbot_tool_spec"] = _tool_spec
        _s.loader.exec_module(_tool_spec)

parse_simple_yaml = _tool_spec.parse_simple_yaml
REPO = _tool_spec.REPO

SCHEMA_VERSION = 1
INKS_RELPATH = "config/inks.yaml"
PALETTE_RELPATH = "config/palette.yaml"
LOAD_RELPATH = "config/palette_load.yaml"
INVENTORY_RELPATH = "config/inventory.yaml"
PALETTE_CAL_RELPATH = "config/palette_calibration.yaml"
MODES = ("real", "rehearsal", "none")
DIP_REASONS = ("session_start", "low_charge", "color_change", "operator")
NO_INK = "none"


def _utc() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _node() -> str:
    return os.environ.get("TATBOT_NODE") or socket.gethostname().split(".")[0]


def _read(relpath: str, repo: Path | str) -> dict:
    path = Path(repo) / relpath
    if not path.is_file():
        raise FileNotFoundError(f"missing {path}")
    data = parse_simple_yaml(path.read_text())
    version = data.get("schema_version", SCHEMA_VERSION)
    if version != SCHEMA_VERSION:
        raise ValueError(f"{path}: schema_version {version}, this code reads {SCHEMA_VERSION}")
    return data


# --- the registries -------------------------------------------------------------

@dataclass(frozen=True)
class Ink:
    ink_id: str
    display_name: str
    rgb: tuple[int, int, int]
    prompt_phrase: str
    vendor: str | None = None
    product: str | None = None
    bottle_ml: float | None = None
    viscosity: str | None = None
    dip: dict = field(default_factory=dict)
    """Per-ink overrides of the tool's ``ink:`` block (a ``dip:`` map in
    inks.yaml): any of DIP_OVERRIDE_KEYS. A thin liner takes up less than a
    thick opaque; the tool's datasheet is the default, the ink refines it."""


DIP_OVERRIDE_KEYS = ("uptake_ul", "deposit_ul_per_mm", "bleed_ul_per_s", "dip_depth_m", "dip_dwell_s")


@dataclass(frozen=True)
class CapSize:
    size_id: str
    diameter_m: float
    depth_m: float
    capacity_ul: float
    usable_frac: float = 0.7
    product: str | None = None

    @property
    def area_m2(self) -> float:
        return math.pi * (self.diameter_m / 2) ** 2

    def surface_depth_m(self, fill_ul: float) -> float:
        """How far below the rim the ink surface sits for a given fill."""
        fill_m3 = max(0.0, fill_ul) * 1e-9
        return max(0.0, self.depth_m - fill_m3 / self.area_m2)


@dataclass(frozen=True)
class PaletteSlot:
    slot_id: str      # == URDF frame name
    size: CapSize
    arm: str


@dataclass(frozen=True)
class SlotLoad:
    slot_id: str
    ink_id: str | None       # None = dry cap
    fill_ul: float = 0.0
    bottle: str | None = None
    utc: str | None = None

    @property
    def dry(self) -> bool:
        return self.ink_id is None or self.fill_ul <= 0


@dataclass(frozen=True)
class InkPolicy:
    """The ``ink:`` block of a tool datasheet."""

    mode: str = "none"
    charge_capacity_ul: float = 0.0
    uptake_ul: float = 0.0
    deposit_ul_per_mm: float = 0.0
    bleed_ul_per_s: float = 0.0
    dip_depth_m: float = 0.0
    dip_dwell_s: float = 0.0
    min_fill_frac: float = 0.0

    @property
    def dips(self) -> bool:
        return self.mode in ("real", "rehearsal")

    @property
    def touches_stock(self) -> bool:
        return self.mode == "real"

    def stroke_ul(self, contact_mm: float, contact_s: float) -> float:
        return self.deposit_ul_per_mm * max(0.0, contact_mm) + self.bleed_ul_per_s * max(0.0, contact_s)


def load_inks(repo: Path | str = REPO) -> dict[str, Ink]:
    data = _read(INKS_RELPATH, repo)
    out = {}
    for ink_id, d in data.items():
        if not isinstance(d, dict):
            continue
        rgb = d.get("rgb")
        if not (isinstance(rgb, list) and len(rgb) == 3):
            raise ValueError(f"{INKS_RELPATH}: {ink_id}: rgb must be [r, g, b]")
        out[ink_id] = Ink(
            ink_id=ink_id,
            display_name=d.get("display_name") or ink_id,
            rgb=(int(rgb[0]), int(rgb[1]), int(rgb[2])),
            prompt_phrase=d.get("prompt_phrase") or "with ink",
            vendor=d.get("vendor"),
            product=d.get("product"),
            bottle_ml=d.get("bottle_ml"),
            viscosity=d.get("viscosity"),
            dip=_dip_overrides(ink_id, d.get("dip")),
        )
    if NO_INK in out:
        raise ValueError(f"{INKS_RELPATH}: {NO_INK!r} is reserved for a dry cap")
    return out


def _dip_overrides(ink_id: str, block) -> dict:
    if not block:
        return {}
    if not isinstance(block, dict):
        raise ValueError(f"{INKS_RELPATH}: {ink_id}: dip must be a map")
    bad = sorted(set(block) - set(DIP_OVERRIDE_KEYS))
    if bad:
        raise ValueError(f"{INKS_RELPATH}: {ink_id}: dip keys {bad} not in {DIP_OVERRIDE_KEYS}")
    out = {}
    for k, v in block.items():
        if v is None:
            continue
        v = float(v)
        if v < 0:
            raise ValueError(f"{INKS_RELPATH}: {ink_id}: dip.{k} must be >= 0")
        out[k] = v
    return out


def policy_with_ink(policy: "InkPolicy", ink: Ink | None) -> "InkPolicy":
    """The tool's policy refined by the ink it is about to carry: the ink's
    ``dip:`` overrides replace the datasheet's numbers, uptake still capped
    at the tool's capacity. No ink (a dry rehearsal cap) is the datasheet."""
    if ink is None or not ink.dip:
        return policy
    fields = {k: v for k, v in ink.dip.items() if k in DIP_OVERRIDE_KEYS}
    if "uptake_ul" in fields:
        fields["uptake_ul"] = min(fields["uptake_ul"], policy.charge_capacity_ul)
    return replace(policy, **fields)


def load_palette(repo: Path | str = REPO) -> dict[str, PaletteSlot]:
    data = _read(PALETTE_RELPATH, repo)
    sizes = {}
    for size_id, d in (data.get("sizes") or {}).items():
        cs = CapSize(
            size_id=size_id,
            diameter_m=float(d["diameter_m"]),
            depth_m=float(d["depth_m"]),
            capacity_ul=float(d["capacity_ul"]),
            usable_frac=float(d.get("usable_frac", 0.7)),
            product=d.get("product"),
        )
        geometric = cs.area_m2 * cs.depth_m * 1e9
        if abs(geometric - cs.capacity_ul) / geometric > 0.05:
            raise ValueError(
                f"{PALETTE_RELPATH}: size {size_id}: capacity_ul {cs.capacity_ul:.0f} disagrees "
                f"with pi r^2 h = {geometric:.0f} uL by more than 5%")
        sizes[size_id] = cs
    slots = {}
    for slot_id, d in (data.get("slots") or {}).items():
        size = sizes.get(d.get("size"))
        if size is None:
            raise ValueError(f"{PALETTE_RELPATH}: slot {slot_id}: unknown size {d.get('size')!r}")
        arm = d.get("arm")
        if arm not in ("left", "right"):
            raise ValueError(f"{PALETTE_RELPATH}: slot {slot_id}: arm must be left|right")
        slots[slot_id] = PaletteSlot(slot_id=slot_id, size=size, arm=arm)
    if not slots:
        raise ValueError(f"{PALETTE_RELPATH}: no slots")
    return slots


def load_palette_load(repo: Path | str = REPO,
                      palette: dict[str, PaletteSlot] | None = None) -> dict[str, SlotLoad]:
    data = _read(LOAD_RELPATH, repo)
    palette = palette or load_palette(repo)
    out = {}
    for slot_id, d in (data.get("slots") or {}).items():
        if slot_id not in palette:
            raise ValueError(f"{LOAD_RELPATH}: {slot_id} is not a slot in {PALETTE_RELPATH}")
        ink = d.get("ink")
        ink_id = None if ink in (None, NO_INK, "") else str(ink)
        out[slot_id] = SlotLoad(
            slot_id=slot_id,
            ink_id=ink_id,
            fill_ul=float(d.get("fill_ul") or 0.0),
            bottle=d.get("bottle"),
            utc=d.get("utc"),
        )
    for slot_id in palette:
        out.setdefault(slot_id, SlotLoad(slot_id=slot_id, ink_id=None))
    return out


SUPPLIES = ("bench", "wet", "dry")


def supply_load(kind: str, palette: dict[str, PaletteSlot], ink_id: str | None = None,
                repo: Path | str = REPO, arm: str = "right") -> dict[str, SlotLoad]:
    """A palette load that is NOT the bench's: ``wet`` fills every ``arm`` cap
    to its usable volume with ``ink_id`` (the sim's default — a simulator is
    not the bench, and a batch should not be refused because nobody poured
    this morning); ``dry`` empties every cap; ``bench`` is what
    config/palette_load.yaml says right now."""
    if kind not in SUPPLIES:
        raise ValueError(f"supply {kind!r} not one of {SUPPLIES}")
    if kind == "bench":
        return load_palette_load(repo, palette)
    if kind == "dry":
        return {s: SlotLoad(s, None) for s in palette}
    if not ink_id:
        raise ValueError("a wet supply needs an ink_id")
    inks = load_inks(repo)
    if ink_id not in inks:
        raise ValueError(f"unknown ink {ink_id!r}; have {', '.join(inks)}")
    return {
        s: (SlotLoad(s, ink_id, p.size.capacity_ul * p.size.usable_frac, bottle=f"{kind}-supply")
            if p.arm == arm else SlotLoad(s, None))
        for s, p in palette.items()
    }


def load_inventory(repo: Path | str = REPO) -> dict:
    """Bottles, cartridges, caps — as nested dicts. Human-scale; see reconcile."""
    data = _read(INVENTORY_RELPATH, repo)
    return {k: (data.get(k) or {}) for k in ("bottles", "cartridges", "caps")} | {
        "utc": data.get("utc")}


def policy_for(tool) -> InkPolicy:
    """The ink policy of a loaded ToolSpec (or anything with ``.raw``/``.tool_id``)."""
    block = (getattr(tool, "raw", None) or {}).get("ink") or {}
    mode = block.get("mode", "none")
    if mode not in MODES:
        raise ValueError(f"{getattr(tool, 'tool_id', tool)}: ink.mode {mode!r} not one of {MODES}")
    pol = InkPolicy(
        mode=mode,
        charge_capacity_ul=float(block.get("charge_capacity_ul") or 0.0),
        uptake_ul=float(block.get("uptake_ul") or 0.0),
        deposit_ul_per_mm=float(block.get("deposit_ul_per_mm") or 0.0),
        bleed_ul_per_s=float(block.get("bleed_ul_per_s") or 0.0),
        dip_depth_m=float(block.get("dip_depth_m") or 0.0),
        dip_dwell_s=float(block.get("dip_dwell_s") or 0.0),
        min_fill_frac=float(block.get("min_fill_frac") or 0.0),
    )
    if pol.dips:
        for name in ("charge_capacity_ul", "uptake_ul", "dip_depth_m"):
            if getattr(pol, name) <= 0:
                raise ValueError(
                    f"{getattr(tool, 'tool_id', tool)}: ink.mode {mode} needs ink.{name} > 0")
        if pol.uptake_ul > pol.charge_capacity_ul:
            raise ValueError(
                f"{getattr(tool, 'tool_id', tool)}: ink.uptake_ul exceeds charge_capacity_ul")
    return pol


# --- validation: the fourth leg of (task, tool, substrate, ink) -------------------

class InkSupplyError(ValueError):
    """The fitted tool and the palette load cannot supply what the task needs."""


def usable_slots(policy: InkPolicy, palette: dict[str, PaletteSlot],
                 load: dict[str, SlotLoad], arm: str = "right",
                 ink_id: str | None = None) -> list[SlotLoad]:
    """Slots this tool may dip in, in palette order. Rehearsal accepts dry caps
    (and ignores ink identity, since a dry cap has none); real needs the ink and
    a fill above ``min_fill_frac`` of the cap's usable volume."""
    out = []
    for slot_id, slot in palette.items():
        if slot.arm != arm:
            continue
        sl = load[slot_id]
        if policy.mode == "rehearsal":
            if sl.ink_id is None or ink_id is None or sl.ink_id == ink_id:
                out.append(sl)
            continue
        if policy.mode != "real":
            continue
        if sl.dry or (ink_id is not None and sl.ink_id != ink_id):
            continue
        usable = slot.size.capacity_ul * slot.size.usable_frac
        if sl.fill_ul < policy.min_fill_frac * usable:
            continue
        out.append(sl)
    return out


def require_supply(policy: InkPolicy, palette: dict[str, PaletteSlot],
                   load: dict[str, SlotLoad], needs_ink: bool, arm: str = "right",
                   ink_id: str | None = None, tool_id: str = "tool") -> None:
    """Refuse a run whose ink story cannot be true. Raises InkSupplyError."""
    if not needs_ink:
        return
    if policy.mode == "none":
        raise InkSupplyError(
            f"this task needs an ink supply and {tool_id!r} has ink.mode none — fit a "
            "tool that dips (lutin-3rl-bugpin, or lutin-ballpoint-dot to rehearse)")
    if not usable_slots(policy, palette, load, arm, ink_id):
        want = f" of {ink_id}" if ink_id else ""
        raise InkSupplyError(
            f"{tool_id!r} ({policy.mode}) has no usable {arm}-arm cap{want} in "
            f"{LOAD_RELPATH}: fill one with `scripts/ink.py load <slot> <ink> --ul <n>`")


# --- the charge model and dip planner --------------------------------------------

@dataclass(frozen=True)
class StrokeNeed:
    """What one planned stroke will cost, before it is drawn."""
    contact_mm: float
    contact_s: float
    ink_id: str | None = None


@dataclass(frozen=True)
class DipPlan:
    before_stroke: int     # index into the stroke list; len(strokes) never happens
    slot_id: str
    reason: str
    charge_before_ul: float
    charge_after_ul: float
    ink_id: str | None
    cap_fill_ul: float = 0.0   # what the cap holds as this dip begins (real: drains dip by dip)
    why_slot: str = ""         # select_slot's reason


@dataclass
class Charge:
    """Mutable ink on the tool. One per tool per session."""
    ul: float
    capacity_ul: float
    ink_id: str | None = None

    @property
    def frac(self) -> float:
        return 0.0 if self.capacity_ul <= 0 else max(0.0, min(1.0, self.ul / self.capacity_ul))

    def debit(self, ul: float) -> float:
        """Take ink off the tool; returns what was actually available."""
        taken = min(self.ul, max(0.0, ul))
        self.ul -= taken
        return taken

    def credit(self, ul: float, ink_id: str | None = None) -> float:
        room = max(0.0, self.capacity_ul - self.ul)
        added = min(room, max(0.0, ul))
        self.ul += added
        if ink_id is not None:
            self.ink_id = ink_id
        return added


# Which cap size an ink wants, best first. A liner takes little ink per
# session and a small cap keeps it fresh; colour and opaque inks are poured
# for fills and want room. Unknown viscosity behaves like colour.
# Medium first for everything: small caps are narrow enough that the robot's
# needle sometimes misses them (operator, 2026-08-28), so they are the last
# resort even for a liner that would not need the volume.
CAP_SIZE_PREFERENCE: dict[str | None, tuple[str, ...]] = {
    "lining": ("medium", "large", "small"),
    "color": ("medium", "large", "small"),
    "opaque": ("large", "medium", "small"),
    None: ("medium", "large", "small"),
}


@dataclass(frozen=True)
class SlotChoice:
    slot_id: str
    reason: str


def select_slot(policy: InkPolicy, palette: dict[str, PaletteSlot], load: dict[str, SlotLoad],
                arm: str, ink_id: str | None, inks: dict[str, Ink] | None = None,
                fills: dict[str, float] | None = None,
                need_ul: float | None = None) -> SlotChoice | None:
    """Which cap to dip in, and why. The rules, in order:

    1. only usable caps: this arm, this ink, above ``min_fill_frac`` (``usable_slots``);
       ``fills`` is a planner's running view after the dips it already placed,
       so a cap drained mid-plan stops being offered;
    2. rehearsal: the first usable cap — the choreography is the point;
    3. a cap that can cover ``need_ul`` (the rest of the session) beats one
       that cannot, so a session does not hop caps mid-way if one will do;
    4. the ink's preferred cap size (CAP_SIZE_PREFERENCE by viscosity);
    5. the fullest.
    """
    fills = fills or {}
    current = {s: replace(sl, fill_ul=fills.get(s, sl.fill_ul)) for s, sl in load.items()}
    cands = usable_slots(policy, palette, current, arm, ink_id)
    if not cands:
        return None
    if policy.mode == "rehearsal":
        return SlotChoice(cands[0].slot_id, "rehearsal: first usable cap")
    ink = (inks or {}).get(ink_id) if ink_id else None
    if ink is None and ink_id is None and cands:
        cand_ink_id = cands[0].ink_id
        ink = (inks or {}).get(cand_ink_id) if cand_ink_id else None
    viscosity = ink.viscosity if ink is not None else None
    pref = CAP_SIZE_PREFERENCE.get(viscosity, CAP_SIZE_PREFERENCE[None])
    reasons = []
    pool = cands
    if need_ul is not None:
        enough = [c for c in pool if c.fill_ul >= need_ul]
        if enough:
            pool = enough
            reasons.append(f"covers the remaining {need_ul:.1f} uL")
        else:
            reasons.append(f"no cap covers the remaining {need_ul:.1f} uL")
    by_size = sorted(pool, key=lambda c: (
        pref.index(palette[c.slot_id].size.size_id) if palette[c.slot_id].size.size_id in pref else 9,
        -c.fill_ul))
    best = by_size[0]
    size = palette[best.slot_id].size.size_id
    reasons.append(f"{getattr(ink, 'viscosity', None) or 'ink'} prefers {pref[0]}; took {size}")
    same = [c for c in pool if palette[c.slot_id].size.size_id == size]
    if len(same) > 1:
        reasons.append(f"fullest of {len(same)} {size} caps")
    return SlotChoice(best.slot_id, "; ".join(reasons))


def pick_slot(policy: InkPolicy, palette: dict[str, PaletteSlot], load: dict[str, SlotLoad],
              arm: str, ink_id: str | None, fills: dict[str, float] | None = None) -> str | None:
    """Compatibility shim over ``select_slot``: the slot id, or None."""
    choice = select_slot(policy, palette, load, arm, ink_id, fills=fills)
    return choice.slot_id if choice else None


# --- what a planned program will cost ---------------------------------------------------

def need_from_polylines(polylines_m, policy: InkPolicy, speed_m_s: float,
                        settle_s: float = 0.2, ink_id: str | None = None) -> float:
    """Microlitres a planned program spends: every stroke's length in contact
    and its time on the surface (the draw plus the settle before the lift),
    through the same stroke_ul the ledger uses. ``polylines_m`` is a list of
    strokes, each a list of [x, y] canvas points in metres — the shape
    run_meta's ``strokes_canvas_m`` / ``path_canvas_m`` and a language
    program's strokes already have."""
    total = 0.0
    for pts in polylines_m:
        length = 0.0
        for a, b in zip(pts, pts[1:], strict=False):
            length += math.dist(a[:2], b[:2])
        total += policy.stroke_ul(length * 1000.0, length / max(speed_m_s, 1e-6) + settle_s)
    return total


def program_polylines(doc) -> list:
    """Pull stroke polylines out of the JSON shapes a program comes in: a bare
    list of polylines; a run_meta.json (every episode's strokes); a single
    episode entry; or a language program dict with ``strokes``."""
    if isinstance(doc, list):
        if doc and isinstance(doc[0], dict):
            out = []
            for ep in doc:
                out += program_polylines(ep)
            return out
        if doc and doc[0] and isinstance(doc[0][0], (int, float)):
            return [doc]  # one polyline
        return doc
    if isinstance(doc, dict):
        if "episodes" in doc:
            return program_polylines(doc["episodes"])
        for key in ("strokes_canvas_m", "strokes", "polylines"):
            if key in doc:
                return program_polylines(doc[key])
        if "path_canvas_m" in doc:
            return [doc["path_canvas_m"]]
    return []


# --- mise en place: what a human sets up before a session ---------------------------

@dataclass(frozen=True)
class Prep:
    kind: str       # fill | refill | dump | cartridge | bottle | weigh | ok | info
    text: str
    slot_id: str | None = None
    ink_id: str | None = None
    ul: float | None = None


def _open_bottle(inventory: dict, ink_id: str) -> str | None:
    best = None
    for bid, b in (inventory.get("bottles") or {}).items():
        if b.get("ink") != ink_id or b.get("retired"):
            continue
        if b.get("opened") and (best is None or (b.get("remaining_ml") or 0) > 0):
            return bid
        best = best or bid
    return best


def mise_en_place(policy: InkPolicy, palette: dict[str, PaletteSlot], load: dict[str, SlotLoad],
                  inks: dict[str, Ink], inventory: dict, needs: dict[str, float],
                  tool_id: str = "tool", arm: str = "right") -> list[Prep]:
    """The checklist for a session that will spend ``needs`` (ink_id -> uL).

    Pure: reads state, proposes actions, changes nothing. Fills are proposed
    on dry caps of the ink's preferred size, largest need first; a cap below
    the floor with the right ink is a refill; caps holding inks the session
    does not need are pointed out, not dumped."""
    out: list[Prep] = []
    if policy.mode == "none":
        return [Prep("ok", f"{tool_id} carries no ink — nothing to pour")]
    if policy.mode == "rehearsal":
        out.append(Prep("info", f"{tool_id} rehearses: leave every cap DRY (water and ink are messy)"))
        wet = [s for s, sl in load.items() if not sl.dry and palette[s].arm == arm]
        for s in wet:
            out.append(Prep("dump", f"{s} holds {load[s].ink_id}; dump it or the rehearsal dips into pigment",
                            slot_id=s, ink_id=load[s].ink_id))
        return out or [Prep("ok", "every cap is dry")]

    for cid, c in (inventory.get("cartridges") or {}).items():
        if c.get("fits") == tool_id and not c.get("retired"):
            n = c.get("count")
            if n is None:
                out.append(Prep("cartridge", f"count the {c.get('needle_code')} box {cid} (never counted)"))
            elif n <= 0:
                out.append(Prep("cartridge", f"{cid} is empty — no fresh {c.get('needle_code')} to fit"))
            else:
                out.append(Prep("cartridge", f"fit a fresh {c.get('needle_code')} from {cid} ({n} left)"))
            break
    else:
        out.append(Prep("cartridge", f"no cartridge box in inventory fits {tool_id}"))

    taken: set[str] = set()
    for ink_id, need in sorted(needs.items(), key=lambda kv: -kv[1]):
        if ink_id not in inks:
            out.append(Prep("info", f"{ink_id} is not in config/inks.yaml"))
            continue
        pref = CAP_SIZE_PREFERENCE.get(inks[ink_id].viscosity, CAP_SIZE_PREFERENCE[None])
        have = 0.0
        for s, sl in load.items():
            if palette[s].arm != arm or sl.ink_id != ink_id:
                continue
            usable = palette[s].size.capacity_ul * palette[s].size.usable_frac
            if sl.fill_ul >= policy.min_fill_frac * usable:
                have += sl.fill_ul
                taken.add(s)
            else:
                top = usable - sl.fill_ul
                out.append(Prep("refill", f"top up {s} with {ink_id}: {top:.0f} uL (below the floor)",
                                slot_id=s, ink_id=ink_id, ul=top))
                have += usable
                taken.add(s)
        # the uptake per dip is the unit the caps are spent in; ask for a
        # little more than the need so the last dip is not refused
        want = need * 1.15 + policy.uptake_ul
        while have < want:
            dry = [s for s, sl in load.items()
                   if palette[s].arm == arm and sl.dry and s not in taken]
            if not dry:
                out.append(Prep("info", f"{ink_id}: {want - have:.0f} uL short and no dry cap left to fill"))
                break
            dry.sort(key=lambda s: (pref.index(palette[s].size.size_id)
                                    if palette[s].size.size_id in pref else 9))
            s = dry[0]
            usable = palette[s].size.capacity_ul * palette[s].size.usable_frac
            ul = min(usable, max(want - have, policy.min_fill_frac * usable + policy.uptake_ul * 2))
            bottle = _open_bottle(inventory, ink_id)
            src = f" from {bottle}" if bottle else " — no bottle of it in inventory.yaml"
            out.append(Prep("fill", f"fill {s} ({palette[s].size.size_id}) with {ink_id}: {ul:.0f} uL{src}",
                            slot_id=s, ink_id=ink_id, ul=ul))
            have += ul
            taken.add(s)
        if have >= want and not any(o.kind in ("fill", "refill") and o.ink_id == ink_id for o in out):
            out.append(Prep("ok", f"{ink_id}: {have:.0f} uL loaded covers {need:.0f} uL", ink_id=ink_id))
    for s, sl in load.items():
        if palette[s].arm == arm and not sl.dry and sl.ink_id not in needs:
            out.append(Prep("info", f"{s} holds {sl.ink_id}, which this session does not use",
                            slot_id=s, ink_id=sl.ink_id))
    out.append(Prep("weigh", "weigh each filled cap: `ink.py weigh <slot> <g> --when before`"))
    return out


def plan_dips(strokes: list[StrokeNeed], policy: InkPolicy, palette: dict[str, PaletteSlot],
              load: dict[str, SlotLoad], arm: str = "right",
              initial_charge_ul: float = 0.0, initial_ink: str | None = None,
              tool_id: str = "tool", inks: dict[str, Ink] | None = None) -> list[DipPlan]:
    """Where dips go and why. Pure: touches nothing.

    A stroke that costs more than a full charge gets one dip before it and is
    drawn on what it has — a mid-stroke dip is a different stroke, which the
    planner that made the stroke list should have produced.
    """
    if not policy.dips or not strokes:
        return []
    charge = Charge(ul=min(initial_charge_ul, policy.charge_capacity_ul),
                    capacity_ul=policy.charge_capacity_ul, ink_id=initial_ink)
    fills = {s: load[s].fill_ul for s in load}
    plans: list[DipPlan] = []
    for i, need in enumerate(strokes):
        want = need.ink_id
        reason = None
        if i == 0 and charge.ul <= 0:
            reason = "session_start"
        elif want is not None and charge.ink_id is not None and want != charge.ink_id:
            reason = "color_change"
        elif policy.stroke_ul(need.contact_mm, need.contact_s) > charge.ul:
            reason = "low_charge"
        if reason is not None:
            remaining = sum(policy.stroke_ul(n.contact_mm, n.contact_s) for n in strokes[i:])
            choice = select_slot(policy, palette, load, arm, want, inks, fills, need_ul=remaining)
            if choice is None:
                raise InkSupplyError(
                    f"stroke {i} needs {want or 'ink'} and {tool_id!r} has no usable "
                    f"{arm}-arm cap for it")
            slot_id = choice.slot_id
            before = charge.ul
            if reason == "color_change":
                charge.ul = 0.0  # a colour change starts from a wiped needle
                before = 0.0
            fill_before = fills[slot_id]
            taken = charge.credit(policy.uptake_ul, load[slot_id].ink_id or want)
            if policy.touches_stock:
                fills[slot_id] = max(0.0, fills[slot_id] - taken)
            plans.append(DipPlan(i, slot_id, reason, before, charge.ul, charge.ink_id,
                                 cap_fill_ul=fill_before, why_slot=choice.reason))
        charge.debit(policy.stroke_ul(need.contact_mm, need.contact_s))
    return plans


def dip_plunge_m(policy: InkPolicy, slot: PaletteSlot, fill_ul: float) -> float:
    """How far below the cap RIM the tip goes: the ink surface plus dip_depth_m,
    never past the floor. A dry cap (rehearsal) plunges to dip_depth_m below the
    rim, which is the same choreography without the pigment."""
    surface = slot.size.surface_depth_m(fill_ul) if fill_ul > 0 else 0.0
    return min(slot.size.depth_m, surface + policy.dip_depth_m)


# --- where the caps are ----------------------------------------------------------------

URDF_RELPATH = "urdf/tatbot.urdf"


def palette_layout_from_urdf(repo: Path | str = REPO) -> dict[str, tuple[float, float, float]]:
    """Each slot's offset from ``palette_root``, metres, read from the URDF's
    fixed ``inkcap_*`` joints. The rack's ARC is real hardware and this is its
    one source; where the rack SITS is a measured pose (config/poses.yaml
    ``palette_center``), corrected per session by the ``palette_tag8``
    observation — the URDF's ``palette_root`` origin is 1.0-era design intent
    and is deliberately not used for that."""
    import xml.etree.ElementTree as ET

    root = ET.parse(Path(repo) / URDF_RELPATH).getroot()
    out = {}
    for joint in root.findall("joint"):
        child = joint.find("child")
        parent = joint.find("parent")
        if child is None or parent is None:
            continue
        name = child.get("link", "")
        if not name.startswith("inkcap_") or parent.get("link") != "palette_root":
            continue
        origin = joint.find("origin")
        raw_xyz = origin.get("xyz") if origin is not None else None
        xyz = [float(v) for v in (raw_xyz or "0 0 0").split()]
        out[name] = (xyz[0], xyz[1], xyz[2])
    if not out:
        raise ValueError(f"{URDF_RELPATH}: no inkcap_* joints under palette_root")
    return out


def _urdf_fixed_origin(root, child: str):
    for joint in root.findall("joint"):
        c = joint.find("child")
        if c is not None and c.get("link") == child:
            if joint.get("type") != "fixed":
                raise ValueError(f"{URDF_RELPATH}: {child} is not on a fixed joint")
            o = joint.find("origin")
            xyz = [float(v) for v in (o.get("xyz") if o is not None else "0 0 0").split()]
            rpy = [float(v) for v in (o.get("rpy") if o is not None else "0 0 0").split()]
            return xyz, rpy, joint.find("parent").get("link")
    raise ValueError(f"{URDF_RELPATH}: no joint has child {child!r}")


def _rpy_matrix(r, p, y):
    cr, sr, cp, sp, cy, sy = math.cos(r), math.sin(r), math.cos(p), math.sin(p), math.cos(y), math.sin(y)
    return [
        [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
        [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
        [-sp, cp * sr, cp * cr],
    ]


def tag8_in_palette_root(repo: Path | str = REPO) -> tuple[float, float, float]:
    """Where the palette's AprilTag (``palette_tag8``) sits relative to
    ``palette_root``, from the URDF — so an observed tag centre (a tip planted
    on it, or a fused tag pose) gives the rack's root back."""
    import xml.etree.ElementTree as ET

    root = ET.parse(Path(repo) / URDF_RELPATH).getroot()
    xyz, _rpy, parent = _urdf_fixed_origin(root, "palette_tag8")
    if parent != "palette_root":
        raise ValueError(f"{URDF_RELPATH}: palette_tag8 must hang off palette_root")
    return (xyz[0], xyz[1], xyz[2])


def base_from_root_matrix(repo: Path | str = REPO, arm: str = "right"):
    """4x4 rigid transform ROOT -> the arm's base frame, from the URDF. The
    driver's Cartesian API and FK-derived poses live in the arm base; the
    robot-world calibration bundle solves in the rig ROOT (see
    docs/ee_fiducial_tracking.md), so a vision observation must cross this to
    land in the same frame as il_dip's caps."""
    import xml.etree.ElementTree as ET

    root = ET.parse(Path(repo) / URDF_RELPATH).getroot()
    base_xyz, base_rpy, base_parent = _urdf_fixed_origin(root, f"{arm}/base_link")
    if base_parent != "root":
        raise ValueError(f"{URDF_RELPATH}: {arm}/base_link must hang off root")
    rot = _rpy_matrix(*base_rpy)                 # base axes expressed in root
    root_from_base = [[rot[i][j] for j in range(3)] + [base_xyz[i]] for i in range(3)] + [[0, 0, 0, 1]]
    import numpy as _np
    return _np.linalg.inv(_np.array(root_from_base, float))


def palette_root_in_base(repo: Path | str = REPO, arm: str = "right") -> tuple[float, float, float]:
    """``palette_root`` expressed in the ARM's base frame — the frame the
    driver's Cartesian API and the sim both work in. Both the rack and the arm
    mount are fixed joints off ``root`` in the URDF, so this is the rig's own
    geometry. Verified against the measured palette hold on 2026-08-28: the
    tip recorded "on the palette tag" (config/poses.yaml, ROOT frame) lands
    ~3 cm from this point, on the rack, not 27 cm away — poses.yaml's ee_xyz_m
    is root-frame, which is easy to misread as base-frame."""
    import xml.etree.ElementTree as ET

    root = ET.parse(Path(repo) / URDF_RELPATH).getroot()
    pal_xyz, pal_rpy, pal_parent = _urdf_fixed_origin(root, "palette_root")
    base_xyz, base_rpy, base_parent = _urdf_fixed_origin(root, f"{arm}/base_link")
    if pal_parent != "root" or base_parent != "root":
        raise ValueError(f"{URDF_RELPATH}: palette_root and {arm}/base_link must both hang off root")
    rot = _rpy_matrix(*base_rpy)  # base in root; rot^T maps root vectors into base
    d = [pal_xyz[i] - base_xyz[i] for i in range(3)]
    res = tuple(float(sum(rot[j][i] * d[j] for j in range(3))) for i in range(3))
    return (res[0], res[1], res[2])


# --- palette calibration: where the rack ACTUALLY is (config/palette_calibration.yaml) ---
#
# palette_root_in_base above is the rig's nominal geometry from the URDF. The
# rack is not bolted to that pose — it is set on the bench and moves. A palette
# calibration measures the real palette_root in the arm base frame, from either
# source, and il_dip prefers it over the URDF nominal:
#   tip     the ballpoint tip planted on palette_tag8 and rolled — the pivot
#           solve gives the tag centre in the base frame directly (FK, no
#           camera error), so this is authoritative. Needs the arm.
#   vision  palette_tag8 observed by the cameras and carried into the base
#           frame through the robot-world bundle — hands-off but only as good
#           as that bundle (~7 mm), so it is the quick check, not the truth.
# Both are dated; il_dip takes the freshest un-stale tip, else vision, else the
# URDF, and folds the source's residual into the cap-clearance budget.

def load_palette_cal(repo: Path | str = REPO) -> dict:
    data = _read(PALETTE_CAL_RELPATH, repo)
    return {k: v for k, v in data.items() if k in ("tip", "vision") and isinstance(v, dict)}


def _cal_age_h(rec: dict, now: datetime | None = None) -> float | None:
    utc = rec.get("utc")
    if not utc:
        return None
    try:
        t = datetime.strptime(utc, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc)
    except ValueError:
        return None
    return (( now or datetime.now(timezone.utc)) - t).total_seconds() / 3600.0


def choose_palette_root(cal: dict, urdf_root, max_age_h: float = 168.0,
                        now: datetime | None = None) -> dict:
    """Which palette_root il_dip should use, and why. Precedence: a fresh tip
    calibration, then a fresh vision one, then the URDF nominal. 'fresh' is
    younger than max_age_h — the rack moves, and a month-old measurement is a
    guess. Returns {root, source, residual_mm, age_h, note}."""
    for source in ("tip", "vision"):
        rec = cal.get(source) or {}
        root = rec.get("root_xyz_m")
        age = _cal_age_h(rec, now)
        if not root or len(root) != 3:
            continue
        if age is not None and age > max_age_h:
            continue
        return {"root": tuple(float(x) for x in root), "source": source,
                "residual_mm": float(rec.get("residual_mm") or 0.0),
                "age_h": age, "note": rec.get("note")}
    # nothing fresh: say if there is a STALE one worth re-measuring
    stale = next((src for src in ("tip", "vision")
                  if (cal.get(src) or {}).get("root_xyz_m")), None)
    note = f"{stale} calibration is stale (> {max_age_h:.0f} h) — re-measure it" if stale else None
    return {"root": tuple(float(x) for x in urdf_root), "source": "urdf",
            "residual_mm": 0.0, "age_h": None, "note": note}


def write_palette_cal(source: str, rec: dict, repo: Path | str = REPO) -> Path:
    """Merge one source's record into config/palette_calibration.yaml, leaving
    the other source untouched."""
    if source not in ("tip", "vision"):
        raise ValueError(f"palette cal source must be tip|vision, not {source!r}")
    path = Path(repo) / PALETTE_CAL_RELPATH
    body = {"schema_version": SCHEMA_VERSION}
    existing = load_palette_cal(repo)
    existing[source] = rec
    for src in ("tip", "vision"):
        if src in existing:
            body[src] = existing[src]
    path.write_text(_preserve_header(path) + "\n".join(_emit(body)) + "\n")
    return path


# --- the ledger ---------------------------------------------------------------------

def ledger_path() -> Path:
    override = os.environ.get("TATBOT_INK_LEDGER")
    if override:
        return Path(override).expanduser()
    root = os.environ.get("TATBOT_LOG_ROOT")
    if not root:
        try:
            import tatbot_runlog  # scripts/lib sibling
            root = str(tatbot_runlog.log_root())
        except Exception:
            root = "~/tatbot-logs"
    return Path(root).expanduser() / "ink" / "ledger.jsonl"


def remote_ledger_dir(path: Path | None = None) -> Path:
    """Copies of OTHER nodes' ledgers (``ink.py sync``), one file per node,
    read alongside the local one. Sessions dip on the arm node while the
    console runs elsewhere; without this the shared truth was only the committed
    palette_load.yaml after a by-hand reconcile."""
    return (path or ledger_path()).parent / "remote"


def append_event(kind: str, mode: str, path: Path | None = None,
                 mirror: Path | None = None, **fields) -> dict:
    """Append one event. Every event carries an ``id`` so the same event read
    from a synced copy and the local file is counted once. ``mirror`` is a
    second file to write the same line to — a run directory's ``ink.jsonl``,
    so the run carries its own ink story next to its flight log."""
    if mode not in ("real", "rehearsal", "sim"):
        raise ValueError(f"ledger mode {mode!r} must be real|rehearsal|sim")
    ev = {"id": uuid.uuid4().hex, "utc": _utc(), "node": _node(), "kind": kind, "mode": mode,
          **fields}
    path = path or ledger_path()
    line = json.dumps(ev, sort_keys=True) + "\n"
    for target in (path, mirror):
        if target is None:
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        with open(target, "a") as f:
            f.write(line)
    return ev


def _read_lines(path: Path) -> list[dict]:
    out = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if line:
            out.append(json.loads(line))
    return out


def read_events(path: Path | None = None, since: str | None = None,
                mode: str | None = None, include_remote: bool = True) -> list[dict]:
    """The ledger, oldest first: the local file plus every synced remote copy,
    de-duplicated by event ``id`` (events older than ids fall back to their
    whole line)."""
    path = path or ledger_path()
    files = [path] if path.is_file() else []
    if include_remote:
        rdir = remote_ledger_dir(path)
        if rdir.is_dir():
            files += sorted(rdir.glob("*.jsonl"))
    seen: set[str] = set()
    out = []
    for f in files:
        for ev in _read_lines(f):
            key = ev.get("id") or json.dumps(ev, sort_keys=True)
            if key in seen:
                continue
            seen.add(key)
            if since and ev.get("utc", "") < since:
                continue
            if mode and ev.get("mode") != mode:
                continue
            out.append(ev)
    out.sort(key=lambda e: e.get("utc", ""))
    return out


@dataclass
class Replay:
    """What the REAL events say happened to caps and stock. Rehearsal and sim
    events are counted but never applied."""
    cap_fill_ul: dict[str, float] = field(default_factory=dict)
    cap_ink: dict[str, str | None] = field(default_factory=dict)
    bottle_used_ul: dict[str, float] = field(default_factory=dict)
    cartridge_used: dict[str, int] = field(default_factory=dict)
    caps_used: dict[str, int] = field(default_factory=dict)
    stroke_ul: float = 0.0
    contact_mm: float = 0.0
    contact_s: float = 0.0
    dips: int = 0
    ignored: dict[str, int] = field(default_factory=dict)
    weighs: list[dict] = field(default_factory=list)


def replay(events: list[dict]) -> Replay:
    r = Replay()
    for ev in events:
        mode = ev.get("mode")
        kind = ev.get("kind")
        if not isinstance(mode, str) or mode != "real":
            if isinstance(mode, str):
                r.ignored[mode] = r.ignored.get(mode, 0) + 1
            continue
        if kind == "cap.fill":
            slot = ev["slot"]
            r.cap_fill_ul[slot] = r.cap_fill_ul.get(slot, 0.0) + float(ev.get("ul", 0))
            r.cap_ink[slot] = ev.get("ink_id")
            bottle = ev.get("bottle")
            if isinstance(bottle, str):
                r.bottle_used_ul[bottle] = r.bottle_used_ul.get(bottle, 0.0) + float(ev.get("ul", 0))
            if ev.get("cap_stock"):
                r.caps_used[ev["cap_stock"]] = r.caps_used.get(ev["cap_stock"], 0) + 1
        elif kind == "cap.dump":
            r.cap_fill_ul[ev["slot"]] = 0.0
            r.cap_ink[ev["slot"]] = None
        elif kind == "dip":
            slot = ev["slot"]
            r.cap_fill_ul[slot] = max(0.0, r.cap_fill_ul.get(slot, 0.0) - float(ev.get("uptake_ul", 0)))
            r.dips += 1
        elif kind == "stroke":
            r.stroke_ul += float(ev.get("ul", 0))
            r.contact_mm += float(ev.get("contact_mm", 0))
            r.contact_s += float(ev.get("contact_s", 0))
        elif kind == "cartridge.fit":
            cid = ev["cartridge_id"]
            r.cartridge_used[cid] = r.cartridge_used.get(cid, 0) + int(ev.get("n", 1))
        elif kind == "weigh":
            r.weighs.append(ev)
    return r


# --- writers for the two script-owned files ------------------------------------------

def _yaml_scalar(v) -> str:
    if v is None:
        return "null"
    if isinstance(v, bool):
        return "true" if v else "false"
    if isinstance(v, (int, float)):
        return repr(v) if isinstance(v, float) else str(v)
    s = str(v)
    if s in (NO_INK,) or (s and all(c.isalnum() or c in "_-." for c in s) and not s[0].isdigit()):
        return s
    return json.dumps(s)


def _emit(d: dict, indent: int = 0) -> list[str]:
    pad = "  " * indent
    out = []
    for k, v in d.items():
        if isinstance(v, dict):
            out.append(f"{pad}{k}:")
            out.extend(_emit(v, indent + 1))
        elif isinstance(v, (list, tuple)):
            out.append(f"{pad}{k}: {json.dumps(list(v))}")
        else:
            out.append(f"{pad}{k}: {_yaml_scalar(v)}")
    return out


def _preserve_header(path: Path) -> str:
    """The leading comment block of a file we are about to rewrite."""
    if not path.is_file():
        return ""
    head = []
    for line in path.read_text().splitlines():
        if line.startswith("#") or not line.strip():
            head.append(line)
        else:
            break
    return "\n".join(head).rstrip("\n") + "\n\n" if head else ""


def write_palette_load(load: dict[str, SlotLoad], repo: Path | str = REPO,
                       note: str | None = None) -> Path:
    path = Path(repo) / LOAD_RELPATH
    body = {
        "schema_version": SCHEMA_VERSION,
        "utc": _utc(),
        "note": note if note is not None else _existing_note(path),
        "slots": {
            s.slot_id: {
                "ink": s.ink_id or NO_INK,
                "fill_ul": round(s.fill_ul, 1),
                "bottle": s.bottle,
                "utc": s.utc,
            } for s in load.values()
        },
    }
    path.write_text(_preserve_header(path) + "\n".join(_emit(body)) + "\n")
    return path


def _existing_note(path: Path) -> str | None:
    try:
        return parse_simple_yaml(path.read_text()).get("note")
    except Exception:
        return None


def write_inventory(inv: dict, repo: Path | str = REPO) -> Path:
    path = Path(repo) / INVENTORY_RELPATH
    body = {"schema_version": SCHEMA_VERSION, "utc": _utc()}
    for section in ("bottles", "cartridges", "caps"):
        body[section] = inv.get(section) or {}
    path.write_text(_preserve_header(path) + "\n".join(_emit(body)) + "\n")
    return path


# --- dataset stamp ---------------------------------------------------------------------

def dataset_ink_metadata(tool, repo: Path | str = REPO, arm: str = "right",
                         load: dict[str, SlotLoad] | None = None) -> dict:
    """What a dataset carries next to meta/tool.json: the policy and the palette
    load at recording time, inlined so it stays readable after both change.
    ``load`` is the supply the run actually used (a sim's synthetic wet rack,
    say); None is the bench's palette_load.yaml."""
    pol = policy_for(tool)
    palette = load_palette(repo)
    load = load if load is not None else load_palette_load(repo, palette)
    inks = load_inks(repo)
    slots = {}
    for slot_id, slot in palette.items():
        if slot.arm != arm:
            continue
        sl = load[slot_id]
        ink_id = sl.ink_id
        ink = inks.get(ink_id) if ink_id else None
        slots[slot_id] = {
            "size": slot.size.size_id,
            "ink": sl.ink_id or NO_INK,
            "rgb": list(ink.rgb) if ink else None,
            "fill_ul": sl.fill_ul,
            "bottle": sl.bottle,
        }
    return {
        "schema_version": SCHEMA_VERSION,
        "tool_id": getattr(tool, "tool_id", None),
        "policy": pol.__dict__,
        "arm": arm,
        "slots": slots,
    }
