"""New trials: written by the researcher, or the standard characterization card for a cartridge.

A proposal is {technique, cartridge, inks, settings} with an optional hypothesis and `repeat_of` (an A/A repeat of
an earlier trial, settings and all). Every one is checked against the study's techniques, cartridges, inks and
bounds before any is written.

The card (docs/lab.md) measures a cartridge in one sheet: a depth ladder of lines, how far one
dip carries at the deepest and a middle depth, and an A/A repeat of the middle line in each ink; a speed ladder, a
whip, corners and a solid patch in the first. A carry cell dips once when the study bounds `mm_per_dip`: it is drawn
at the bound's top, so one dip has the whole cell to run out in; otherwise at the datasheet's cadence, read per dip. In two inks that is 19 cells, one silicone print's grid markers. Its
depths and speeds are spread over the study's bounds and shared between the parts, so the card needs few runs (each
run is one page set-up).

The dip card measures how a cartridge takes ink, for one whose datasheet dip is borrowed from another (`dip.card.status:
borrowed`, which the compile refuses for a design): its mark and a dip's flood at the datasheet's dip, twice (the
noise floor); how far one dip carries; and a ladder each of the dwell, the depth in the cap and the drain, over
whichever of them the study bounds. Every cell starts on a fresh dip, so each reads its own dip's flood. One depth
and one speed, so one run.
"""
from __future__ import annotations

import numpy as np
from tatbot_contracts import lab

from tatbot_lab import space, store

DEPTHS = 4   # rungs of the card's depth ladder
SPEEDS = 3   # rungs of its speed ladder
LADDER = 3   # rungs of each of the dip card's ladders
# the dip card's ladders: setting -> what its cells are read for
DIP_LADDERS = {"dip_dwell_s": "dwell ladder: the flood at the dip's first stroke against the time in the cap",
               "dip_above_ink_mm": "depth ladder: the flood against how far over the ink the tube's end holds",
               "dip_drain_s": "drain ladder: the flood against the hold over the cap after the dwell"}


def trial(study: dict, trial_id: str, proposal: dict) -> dict:
    """One proposal as a proposed trial record."""
    allowed = {"technique", "cartridge", "inks", "settings", "hypothesis", "repeat_of", "instance"}
    if stray := set(proposal) - allowed:
        raise ValueError(f"proposal: unknown {sorted(stray)}")
    settings = dict(proposal.get("settings") or {})
    space.resolve(proposal["technique"], settings)
    record = {"schema": lab.TRIAL, "study": study["id"], "id": trial_id, "created": store.now(), "status": "proposed",
              "technique": proposal["technique"],
              "cartridge": {"type": proposal["cartridge"], **({"instance": proposal["instance"]}
                                                              if proposal.get("instance") else {})},
              "inks": list(proposal["inks"]), "settings": settings}
    for name in ("hypothesis", "repeat_of"):
        if proposal.get(name):
            record[name] = proposal[name]
    return lab.trial(record, study)


def add(study: dict, proposals: list[dict]) -> list[dict]:
    """Proposed trial records for these proposals, validated together, then appended."""
    existing = store.trials(study["id"])
    ids = store.next_ids(existing, len(proposals), "t")
    records = []
    for trial_id, proposal in zip(ids, proposals, strict=True):
        if (rep := proposal.get("repeat_of")) and rep not in existing and rep not in {r["id"] for r in records}:
            raise ValueError(f"proposal for {trial_id}: repeat_of {rep} is no trial of this study")
        records.append(trial(study, trial_id, proposal))
    store.append(study["id"], records)
    return records


def _spread(span, count: int) -> list[float]:
    lo, hi = span
    return [round(float(v), 3) for v in np.linspace(lo, hi, count)] if hi > lo else [float(lo)]


def card(study: dict, cartridge: str, inks: list[str]) -> list[dict]:
    """The characterization card's proposals for one cartridge, in the given inks (the first is the card's main)."""
    for name in space.REQUIRED:
        if name not in study["bounds"]:
            raise ValueError(f"the card spreads {name} over the study's bounds; the study gives it none")
    depths = sorted(_spread(study["bounds"]["trim_mm"], DEPTHS), reverse=True)   # shallow to deep
    speeds = _spread(study["bounds"]["speed_mm_s"], SPEEDS)
    mid_depth, mid_speed = depths[len(depths) // 2], speeds[len(speeds) // 2]
    want = set(study["techniques"])
    out: list[dict] = []
    one_dip = {"mm_per_dip": study["bounds"]["mm_per_dip"][1]} if "mm_per_dip" in study["bounds"] else {}

    def add_one(technique, ink, trim, speed, why, repeat=False):
        if technique in want:
            out.append({"technique": technique, "cartridge": cartridge, "inks": [ink],
                        "settings": {"trim_mm": trim, "speed_mm_s": speed,
                                     **(one_dip if technique == "carry" else {})},
                        "hypothesis": why, **({"repeat": True} if repeat else {})})

    for ink in inks:
        for trim in depths:
            add_one("line", ink, trim, mid_speed, "depth ladder: the marking threshold, saturation and blowout")
        for trim in sorted({depths[-1], mid_depth}):
            add_one("carry", ink, trim, mid_speed, "how far one dip carries at this depth")
        add_one("line", ink, mid_depth, mid_speed, "A/A repeat of the middle line: the noise floor", repeat=True)
    main = inks[0]
    for speed in speeds:
        if speed != mid_speed:
            add_one("line", main, mid_depth, speed, "speed ladder at the middle depth: "
                    f"{space.dose_per_mm(study, {'speed_mm_s': speed}):.1f} pokes/mm in a stroke's middle")
    add_one("whip", main, mid_depth, mid_speed, "a whip's fade from full to none")
    add_one("corners", main, mid_depth, mid_speed, "what stops, turns and lifts leave")
    add_one("solid", main, mid_depth, mid_speed, "a fill's darkness and its bands")
    return out


def dip_card(study: dict, cartridge: str, ink: str) -> list[dict]:
    """The dip card's proposals for one cartridge in one ink, at the middle of the study's depth and speed bounds."""
    for name in space.REQUIRED:
        if name not in study["bounds"]:
            raise ValueError(f"the dip card draws at the middle of the study's {name} bounds; the study gives it none")
    ladders = [name for name in DIP_LADDERS if name in study["bounds"]]
    if not ladders and "mm_per_dip" not in study["bounds"]:
        raise ValueError(f"the dip card varies the dip: bound mm_per_dip or any of {', '.join(DIP_LADDERS)}")
    depths = sorted(_spread(study["bounds"]["trim_mm"], DEPTHS), reverse=True)
    speeds = _spread(study["bounds"]["speed_mm_s"], SPEEDS)
    base = {"trim_mm": depths[len(depths) // 2], "speed_mm_s": speeds[len(speeds) // 2]}

    def one(technique, why, repeat=False, **dip):
        return {"technique": technique, "cartridge": cartridge, "inks": [ink], "settings": {**base, **dip},
                "hypothesis": why, **({"repeat": True} if repeat else {})}

    out = [one("line", "the mark and a dip's flood at the datasheet's dip"),
           one("line", "A/A repeat at the datasheet's dip: the noise floor", repeat=True)]
    if "mm_per_dip" in study["bounds"]:
        out.append(one("carry", "how far one dip carries: mm_per_dip", mm_per_dip=study["bounds"]["mm_per_dip"][1]))
    for name in ladders:
        rungs = _spread(study["bounds"][name], LADDER)
        # no drain is the datasheet's dip, which the first two cells already draw
        out += [one("line", DIP_LADDERS[name], **{name: value}) for value in rungs
                if not (name == "dip_drain_s" and value == 0)]
    return [p for p in out if p["technique"] in study["techniques"]]


def add_card(study: dict, cartridge: str, inks: list[str]) -> list[dict]:
    """The card proposed: each A/A repeat names the first trial with its settings."""
    return _add_repeats(study, card(study, cartridge, inks))


def add_dip_card(study: dict, cartridge: str, ink: str) -> list[dict]:
    """The dip card proposed, its A/A repeat naming the first cell."""
    return _add_repeats(study, dip_card(study, cartridge, ink))


def _add_repeats(study: dict, proposals: list[dict]) -> list[dict]:
    existing = store.trials(study["id"])
    ids = store.next_ids(existing, len(proposals), "t")
    first: dict[tuple, str] = {}
    for trial_id, proposal in zip(ids, proposals, strict=True):
        key = (proposal["technique"], proposal["inks"][0], tuple(sorted(proposal["settings"].items())))
        if proposal.pop("repeat", False):
            proposal["repeat_of"] = first[key]
        else:
            first.setdefault(key, trial_id)
    records = [trial(study, trial_id, p) for trial_id, p in zip(ids, proposals, strict=True)]
    store.append(study["id"], records)
    return records
