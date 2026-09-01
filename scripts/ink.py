#!/usr/bin/env python3
"""Ink, caps, palette and consumables — the operator's console.

    ink.py status                              palette load, stock, ledger totals
    ink.py load <slot> <ink_id> --ul <n> [--bottle <id>] [--cap-stock <id>]
    ink.py dump <slot>                         cap emptied and discarded
    ink.py bottle add <id> --ink <ink_id> --ml <n> [--purchased YYYY-MM-DD]
    ink.py bottle open|retire <id>
    ink.py cartridge add <id> --spec "<text>" --needle-code 1003RL --count <n> [--fits <tool_id>]
    ink.py cartridge fit <id> [--n 1]          one taken from the box and fitted
    ink.py cartridge count <id> <n>            recount on the shelf
    ink.py cartridge retire <id>
    ink.py caps count <id> <n>                 blank ink caps recount
    ink.py weigh <slot|bottle_id> <grams> --when before|after
    ink.py ledger [--since ISO] [--mode real|rehearsal|sim] [-n N]
    ink.py reconcile [--write]                 fold ledger use into inventory.yaml
    ink.py plan --ee-tool <id> --strokes <mm,s>[,<ink>] ...   dry-run the dip planner
    ink.py fit                                 refit uptake/deposit/bleed from weigh events
    ink.py mise-en-place --ee-tool <id> --need black=800   the setup checklist for a session
    ink.py mise-en-place --ee-tool <id> --ink black --program run_meta.json   need from the planned program
    ink.py session [status]                    the open ink session on this node
    ink.py session start --ee-tool <id> [--need-ul N | --program <json>] [--force]
    ink.py session end [--note "..."]          close it, totals into the ledger
    ink.py session rebuild <session_id>        recover a session's state from the ledger
    ink.py sync <node|user@host> ...           pull other nodes' ledgers into ink/remote/

Nothing here moves the arm. `load`, `dump` and the inventory verbs rewrite the
two script-owned files under config/ (commit them — they are session facts)
and append a matching `real` event to the ledger, so the ledger can always be
replayed into the same state. Rehearsal and sim events are written by the
code that dips, not by hand.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts" / "lib"))

import ink_session  # noqa: E402
import ink_spec  # noqa: E402
import tool_spec  # noqa: E402


def _die(msg: str) -> int:
    print(f"ink: {msg}", file=sys.stderr)
    return 2


# --- status -------------------------------------------------------------------------

def cmd_status(args) -> int:
    inks = ink_spec.load_inks(REPO)
    palette = ink_spec.load_palette(REPO)
    load = ink_spec.load_palette_load(REPO, palette)
    inv = ink_spec.load_inventory(REPO)
    try:
        tool = tool_spec.load_active_tool(REPO)
        pol = ink_spec.policy_for(tool)
        print(f"fitted tool: {tool.tool_id}  ink.mode={pol.mode}")
    except Exception as exc:  # no workspace on a dev box is fine
        print(f"fitted tool: unknown ({exc})")
    print(f"\npalette ({ink_spec.LOAD_RELPATH}):")
    for slot_id, slot in palette.items():
        sl = load[slot_id]
        usable = slot.size.capacity_ul * slot.size.usable_frac
        if sl.dry:
            state = "dry"
        else:
            name = inks[sl.ink_id].display_name if sl.ink_id in inks else f"?{sl.ink_id}"
            state = f"{name} {sl.fill_ul:.0f}/{usable:.0f} uL ({100 * sl.fill_ul / usable:.0f}%)"
        print(f"  {slot_id:24s} {slot.size.size_id:6s} {slot.arm:5s} {state}")
    print("\nbottles:")
    for bid, b in inv["bottles"].items():
        rem = "?" if b.get("remaining_ml") is None else f"{b['remaining_ml']:.0f}"
        flag = " (retired)" if b.get("retired") else ""
        print(f"  {bid:24s} {b.get('ink')}: {rem}/{b.get('ml') or '?'} mL, opened {b.get('opened') or '?'}{flag}")
    print("\ncartridges:")
    for cid, c in inv["cartridges"].items():
        cnt = "?" if c.get("count") is None else c["count"]
        flag = " (retired)" if c.get("retired") else ""
        print(f"  {cid:24s} {c.get('needle_code')}: {cnt}/{c.get('initial_count') or '?'} fits {c.get('fits') or '-'}{flag}")
    print("\nblank caps:")
    for kid, c in inv["caps"].items():
        cnt = "?" if c.get("count") is None else c["count"]
        print(f"  {kid:24s} {c.get('size')}: {cnt}/{c.get('initial_count') or '?'}")
    evs = ink_spec.read_events()
    r = ink_spec.replay(evs)
    print(f"\nledger {ink_spec.ledger_path()}: {len(evs)} events; real dips {r.dips}, "
          f"real strokes {r.contact_mm:.0f} mm / {r.contact_s:.0f} s = {r.stroke_ul:.2f} uL; "
          f"ignored {r.ignored or 'none'}")
    return 0


# --- palette load ---------------------------------------------------------------------

def cmd_load(args) -> int:
    inks = ink_spec.load_inks(REPO)
    palette = ink_spec.load_palette(REPO)
    load = ink_spec.load_palette_load(REPO, palette)
    if args.slot not in palette:
        return _die(f"unknown slot {args.slot!r} (known: {', '.join(palette)})")
    if args.ink not in inks:
        return _die(f"unknown ink {args.ink!r} (known: {', '.join(inks)})")
    slot = palette[args.slot]
    usable = slot.size.capacity_ul * slot.size.usable_frac
    if args.ul <= 0 or args.ul > usable:
        return _die(f"{args.slot} ({slot.size.size_id}) takes at most {usable:.0f} uL usable, not {args.ul}")
    inv = ink_spec.load_inventory(REPO)
    if args.bottle and args.bottle not in inv["bottles"]:
        return _die(f"unknown bottle {args.bottle!r}")
    if args.cap_stock and args.cap_stock not in inv["caps"]:
        return _die(f"unknown cap stock {args.cap_stock!r}")
    prev = load[args.slot]
    if not prev.dry and prev.ink_id != args.ink:
        return _die(f"{args.slot} still holds {prev.ink_id}; `ink.py dump {args.slot}` first")
    ev = ink_spec.append_event("cap.fill", "real", slot=args.slot, ink_id=args.ink, ul=float(args.ul),
                        bottle=args.bottle, cap_stock=args.cap_stock)
    load[args.slot] = ink_spec.SlotLoad(args.slot, args.ink, prev.fill_ul + float(args.ul),
                                 args.bottle or prev.bottle, ev["utc"])
    path = ink_spec.write_palette_load(load, REPO)
    if args.bottle and inv["bottles"][args.bottle].get("remaining_ml") is not None:
        inv["bottles"][args.bottle]["remaining_ml"] = round(
            inv["bottles"][args.bottle]["remaining_ml"] - args.ul / 1000.0, 3)
        ink_spec.write_inventory(inv, REPO)
    if args.cap_stock and inv["caps"][args.cap_stock].get("count") is not None:
        inv["caps"][args.cap_stock]["count"] -= 1
        ink_spec.write_inventory(inv, REPO)
    print(f"{args.slot}: {args.ink} {load[args.slot].fill_ul:.0f} uL -> {path}")
    return 0


def cmd_dump(args) -> int:
    palette = ink_spec.load_palette(REPO)
    load = ink_spec.load_palette_load(REPO, palette)
    if args.slot not in palette:
        return _die(f"unknown slot {args.slot!r}")
    prev = load[args.slot]
    ev = ink_spec.append_event("cap.dump", "real", slot=args.slot, ink_id=prev.ink_id,
                        ul_discarded=prev.fill_ul)
    load[args.slot] = ink_spec.SlotLoad(args.slot, None, 0.0, None, ev["utc"])
    ink_spec.write_palette_load(load, REPO)
    print(f"{args.slot}: dumped {prev.fill_ul:.0f} uL of {prev.ink_id or 'nothing'}")
    return 0


# --- inventory ------------------------------------------------------------------------

def cmd_bottle(args) -> int:
    inv = ink_spec.load_inventory(REPO)
    bottles = inv["bottles"]
    if args.verb == "add":
        if args.id in bottles:
            return _die(f"bottle {args.id!r} exists")
        if args.ink not in ink_spec.load_inks(REPO):
            return _die(f"unknown ink {args.ink!r}")
        bottles[args.id] = {"ink": args.ink, "ml": args.ml, "purchased": args.purchased,
                            "opened": None, "remaining_ml": args.ml, "retired": False,
                            "note": args.note}
        ink_spec.append_event("bottle.add", "real", bottle_id=args.id, ink_id=args.ink, ml=args.ml)
    else:
        if args.id not in bottles:
            return _die(f"unknown bottle {args.id!r}")
        b = bottles[args.id]
        if args.verb == "open":
            b["opened"] = ink_spec._utc()[:10]
            if b.get("remaining_ml") is None:
                b["remaining_ml"] = b.get("ml")
            ink_spec.append_event("bottle.open", "real", bottle_id=args.id, ink_id=b.get("ink"))
        elif args.verb == "retire":
            b["retired"] = True
            ink_spec.append_event("bottle.retire", "real", bottle_id=args.id, ink_id=b.get("ink"),
                           remaining_ml=b.get("remaining_ml"))
    ink_spec.write_inventory(inv, REPO)
    print(f"bottle {args.id}: {bottles[args.id]}")
    return 0


def cmd_cartridge(args) -> int:
    inv = ink_spec.load_inventory(REPO)
    carts = inv["cartridges"]
    if args.verb == "add":
        if args.id in carts:
            return _die(f"cartridge {args.id!r} exists")
        if args.fits and args.fits not in tool_spec.list_tools(REPO):
            return _die(f"unknown tool {args.fits!r}")
        carts[args.id] = {"spec": args.spec, "needle_code": args.needle_code, "fits": args.fits,
                          "initial_count": args.count, "count": args.count,
                          "opened": None, "expires": args.expires, "retired": False, "note": None}
        ink_spec.append_event("cartridge.add", "real", cartridge_id=args.id, spec=args.spec,
                       n=args.count, tool_id=args.fits)
    else:
        if args.id not in carts:
            return _die(f"unknown cartridge {args.id!r}")
        c = carts[args.id]
        if args.verb == "fit":
            if c.get("count") is not None:
                c["count"] = max(0, c["count"] - args.n)
            c["opened"] = c.get("opened") or ink_spec._utc()[:10]
            ink_spec.append_event("cartridge.fit", "real", cartridge_id=args.id, n=args.n,
                           tool_id=c.get("fits"))
        elif args.verb == "count":
            c["count"] = args.n
            ink_spec.append_event("cartridge.count", "real", cartridge_id=args.id, n=args.n)
        elif args.verb == "retire":
            c["retired"] = True
            ink_spec.append_event("cartridge.retire", "real", cartridge_id=args.id, n=c.get("count"))
    ink_spec.write_inventory(inv, REPO)
    print(f"cartridge {args.id}: {carts[args.id]}")
    return 0


def cmd_caps(args) -> int:
    inv = ink_spec.load_inventory(REPO)
    if args.id not in inv["caps"]:
        return _die(f"unknown cap stock {args.id!r}")
    inv["caps"][args.id]["count"] = args.n
    ink_spec.append_event("caps.count", "real", cap_stock=args.id, n=args.n)
    ink_spec.write_inventory(inv, REPO)
    print(f"caps {args.id}: {inv['caps'][args.id]}")
    return 0


def cmd_weigh(args) -> int:
    palette = ink_spec.load_palette(REPO)
    inv = ink_spec.load_inventory(REPO)
    if args.target in palette:
        fields = {"slot": args.target}
    elif args.target in inv["bottles"]:
        fields = {"bottle_id": args.target}
    else:
        return _die(f"{args.target!r} is neither a slot nor a bottle")
    ev = ink_spec.append_event("weigh", "real", grams=float(args.grams), when=args.when, **fields)
    print(ev)
    return 0


# --- ledger ---------------------------------------------------------------------------

def cmd_ledger(args) -> int:
    evs = ink_spec.read_events(since=args.since, mode=args.mode)
    if args.n:
        evs = evs[-args.n:]
    for ev in evs:
        rest = {k: v for k, v in ev.items() if k not in ("utc", "node", "kind", "mode")}
        print(f"{ev['utc']} {ev['node']:8s} {ev['mode']:9s} {ev['kind']:15s} {rest}")
    print(f"{len(evs)} events from {ink_spec.ledger_path()}")
    return 0


def cmd_reconcile(args) -> int:
    """Real events since the snapshot -> what the caps and bottles should read."""
    palette = ink_spec.load_palette(REPO)
    load = ink_spec.load_palette_load(REPO, palette)
    inv = ink_spec.load_inventory(REPO)
    r = ink_spec.replay(ink_spec.read_events())
    drift = 0
    print("caps (palette_load vs ledger replay):")
    for slot_id in palette:
        replayed = r.cap_fill_ul.get(slot_id)
        if replayed is None:
            continue
        cur = load[slot_id].fill_ul
        mark = "" if abs(cur - replayed) < 1.0 else "   <-- drift"
        drift += bool(mark)
        print(f"  {slot_id:24s} file {cur:7.1f}  replay {replayed:7.1f} uL{mark}")
        if args.write and mark:
            load[slot_id] = ink_spec.SlotLoad(slot_id, r.cap_ink.get(slot_id), replayed,
                                       load[slot_id].bottle, ink_spec._utc())
    print("bottles (uL poured into caps per the ledger):")
    for bid, used in r.bottle_used_ul.items():
        b = inv["bottles"].get(bid)
        if not b:
            print(f"  {bid}: NOT in inventory.yaml")
            continue
        print(f"  {bid:24s} poured {used:.0f} uL; remaining_ml {b.get('remaining_ml')}")
    if r.weighs:
        print(f"weigh events on record: {len(r.weighs)} (`ink.py fit` uses these)")
    unknown = [c for c, v in inv["cartridges"].items() if v.get("count") is None and not v.get("retired")]
    if unknown:
        print(f"cartridge boxes never counted: {', '.join(unknown)} — `ink.py cartridge count <id> <n>`")
    if args.write and drift:
        ink_spec.write_palette_load(load, REPO, note="reconciled from ledger")
        print(f"wrote {ink_spec.LOAD_RELPATH}")
    return 0


# --- refit the constants from weighings --------------------------------------------------

def _pairs(weighs: list[dict]) -> list[tuple[dict, dict]]:
    """before/after weigh events on the same cap or bottle, in order."""
    open_: dict[str, dict] = {}
    out = []
    for ev in weighs:
        key = ev.get("slot") or ev.get("bottle_id")
        if ev.get("when") == "before":
            open_[key] = ev
        elif ev.get("when") == "after" and key in open_:
            out.append((open_.pop(key), ev))
    return out


def cmd_fit(args) -> int:
    """Estimate the datasheet's ink constants from what the scale said.

    A before/after weighing of a cap brackets some number of dips into it:
    the mass lost (1 g ~ 1000 uL for a water-based ink) over those dips is
    the uptake. Between consecutive dips the tool is driven to low charge
    by the planner, so what it took up is what it spent — and the ledger has
    the mm and s it spent it on. Two-parameter least squares over those
    intervals gives deposit_ul_per_mm and bleed_ul_per_s. Prints the edit;
    never writes the datasheet.
    """
    # ledger ORDER, not the clock: several events land in one second, and
    # `<=` on equal timestamps would drop or double-count them
    evs = [dict(e, _i=i) for i, e in enumerate(ink_spec.read_events(mode="real"))]
    weighs = [e for e in evs if e.get("kind") == "weigh"]
    pairs = _pairs(weighs)
    if not pairs:
        return _die("no before/after weigh pairs in the real ledger — "
                    "`ink.py weigh <slot> <g> --when before` … `--when after`")
    density = args.density_g_per_ml
    uptakes = []
    for before, after in pairs:
        key = before.get("slot") or before.get("bottle_id")
        between = [e for e in evs if e.get("kind") == "dip" and e.get("slot") == key
                   and before["_i"] < e["_i"] < after["_i"]]
        if not between:
            continue
        lost_ul = (float(before["grams"]) - float(after["grams"])) / density * 1000.0
        uptakes.append(lost_ul / len(between))
        print(f"{key}: {len(between)} dips, {lost_ul:.1f} uL lost -> {lost_ul / len(between):.3f} uL/dip")
    if not uptakes:
        return _die("weigh pairs bracket no dips")
    uptake = sum(uptakes) / len(uptakes)
    # intervals between consecutive dips of one tool: spent ~= uptake
    dips = [e for e in evs if e.get("kind") == "dip"]
    strokes = [e for e in evs if e.get("kind") == "stroke"]
    rows = []
    for a, b in zip(dips, dips[1:], strict=False):
        mm = sum(float(s.get("contact_mm", 0)) for s in strokes if a["_i"] < s["_i"] < b["_i"])
        s_ = sum(float(s.get("contact_s", 0)) for s in strokes if a["_i"] < s["_i"] < b["_i"])
        if mm > 0 or s_ > 0:
            rows.append((mm, s_, uptake))
    print(f"\nuptake_ul: {uptake:.3f}  (from {len(uptakes)} weigh pair(s))")
    if len(rows) >= 2:
        # normal equations for [a, b] in a*mm + b*s = ul
        sxx = sum(r[0] * r[0] for r in rows)
        sxy = sum(r[0] * r[1] for r in rows)
        syy = sum(r[1] * r[1] for r in rows)
        sxz = sum(r[0] * r[2] for r in rows)
        syz = sum(r[1] * r[2] for r in rows)
        det = sxx * syy - sxy * sxy
        if abs(det) > 1e-12:
            a = (sxz * syy - syz * sxy) / det
            b = (sxx * syz - sxy * sxz) / det
            print(f"deposit_ul_per_mm: {max(a, 0):.5f}\nbleed_ul_per_s: {max(b, 0):.5f}  "
                  f"(least squares over {len(rows)} dip intervals)")
        else:
            print("dip intervals are collinear in (mm, s); vary stroke speed to separate deposit from bleed")
    else:
        print("fewer than two dip-to-dip intervals with strokes; deposit/bleed not separable yet")
    print("\nedit the `ink:` block of the tool's datasheet by hand if these look right")
    return 0


# --- mise en place --------------------------------------------------------------------

def cmd_mise(args) -> int:
    tool = tool_spec.load_tool(args.tool_id, REPO)
    pol = ink_spec.policy_for(tool)
    palette = ink_spec.load_palette(REPO)
    load = ink_spec.load_palette_load(REPO, palette)
    inks = ink_spec.load_inks(REPO)
    inv = ink_spec.load_inventory(REPO)
    needs = {}
    for spec in args.need or []:
        ink, _, ul = spec.partition("=")
        needs[ink] = float(ul or 0)
    if args.strokes_mm and not needs:
        return _die("--strokes-mm needs --need <ink>=<uL> or --ink <ink>")
    if args.ink and args.strokes_mm:
        needs[args.ink] = pol.stroke_ul(args.strokes_mm, args.strokes_mm / max(args.speed_mm_s, 1e-6))
    if args.program:
        if not args.ink:
            return _die("--program needs --ink <ink_id>: a program says where, not what colour")
        import json as _json
        doc = _json.loads(Path(args.program).read_text())
        polys = ink_spec.program_polylines(doc)
        if not polys:
            return _die(f"no strokes found in {args.program}")
        need = ink_spec.need_from_polylines(polys, pol, args.speed_mm_s / 1000.0)
        needs[args.ink] = needs.get(args.ink, 0.0) + need
        print(f"program {args.program}: {len(polys)} strokes -> {need:.0f} uL of {args.ink}")
    items = ink_spec.mise_en_place(pol, palette, load, inks, inv, needs, tool_id=tool.tool_id, arm=args.arm)
    print(f"mise en place — {tool.tool_id} ({pol.mode}), needs "
          + (", ".join(f"{k} {v:.0f} uL" for k, v in needs.items()) or "nothing declared"))
    marks = {"fill": "[ ]", "refill": "[ ]", "dump": "[ ]", "cartridge": "[ ]", "weigh": "[ ]",
             "bottle": "[ ]", "ok": " ok", "info": " ~ "}
    for it in items:
        print(f"  {marks.get(it.kind, '[ ]')} {it.text}")
    return 0


# --- planner dry run -------------------------------------------------------------------

def cmd_plan(args) -> int:
    tool = tool_spec.load_tool(args.tool_id, REPO)
    pol = ink_spec.policy_for(tool)
    palette = ink_spec.load_palette(REPO)
    load = ink_spec.load_palette_load(REPO, palette)
    needs = []
    for spec in args.strokes:
        parts = spec.split(",")
        mm, s = float(parts[0]), float(parts[1])
        ink = parts[2] if len(parts) > 2 else None
        needs.append(ink_spec.StrokeNeed(mm, s, ink))
    try:
        ink_spec.require_supply(pol, palette, load, needs_ink=True, arm=args.arm, tool_id=tool.tool_id)
        plans = ink_spec.plan_dips(needs, pol, palette, load, arm=args.arm, tool_id=tool.tool_id)
    except ink_spec.InkSupplyError as exc:
        return _die(str(exc))
    total = sum(pol.stroke_ul(n.contact_mm, n.contact_s) for n in needs)
    print(f"{tool.tool_id} ({pol.mode}): {len(needs)} strokes cost {total:.2f} uL, {len(plans)} dips")
    for d in plans:
        depth = ink_spec.dip_plunge_m(pol, palette[d.slot_id], load[d.slot_id].fill_ul)
        print(f"  before stroke {d.before_stroke:3d}: {d.slot_id:24s} {d.reason:13s} "
              f"{d.charge_before_ul:.2f} -> {d.charge_after_ul:.2f} uL, plunge {depth * 1000:.1f} mm")
    return 0


# --- argparse -----------------------------------------------------------------------------

# --- the session and the other nodes' ledgers -------------------------------------------

def cmd_session(args) -> int:
    verb = args.verb or "status"
    if verb == "status":
        sess = ink_session.current() or ink_session.load()
        policy = None
        if sess is not None:
            try:
                policy = ink_spec.policy_for(tool_spec.load_tool(sess.tool_id, REPO))
            except Exception:
                policy = None
        print(ink_session.describe(sess, policy))
        return 0
    if verb == "start":
        why = _require_tool(args, "session start")
        if why:
            return _die(why)
        tool = tool_spec.load_tool(args.tool_id, REPO)
        policy = ink_spec.policy_for(tool)
        need = args.need_ul
        if args.program:
            import json

            doc = json.loads(Path(args.program).expanduser().read_text())
            need = ink_spec.need_from_polylines(ink_spec.program_polylines(doc), policy,
                                                args.speed_mm_s / 1000.0)
        try:
            sess = ink_session.start(tool, policy, need_ul=need, note=args.note, force=args.force)
        except ValueError as exc:
            return _die(str(exc))
        print(ink_session.describe(sess, policy))
        return 0
    if verb == "end":
        sess = ink_session.current()
        if sess is None:
            return _die("no open session")
        ink_session.end(sess, note=args.note)
        print(ink_session.describe(sess))
        return 0
    if verb == "rebuild":
        if not args.id:
            return _die("session rebuild needs the session id")
        events = ink_spec.read_events()
        start_ev = next((e for e in events if e.get("session_id") == args.id and e.get("kind") == "session.start"), None)
        capacity = None
        if start_ev and start_ev.get("tool_id"):
            try:
                capacity = ink_spec.policy_for(tool_spec.load_tool(start_ev["tool_id"], REPO)).charge_capacity_ul
            except Exception:
                capacity = None
        sess = ink_session.rebuild(args.id, events, capacity_ul=capacity)
        if sess is None:
            return _die(f"no session.start for {args.id} in the ledger")
        print(ink_session.describe(sess))
        if args.write:
            ink_session.save(sess)
            print(f"wrote {ink_session.session_path()}")
        return 0
    return _die(f"unknown session verb {verb}")


def cmd_sync(args) -> int:
    """Copy each node's ledger into <ledger dir>/remote/<node>.jsonl. The
    ledger is append-only and every event carries an id, so a copy is safe
    to re-pull and read_events counts each event once."""
    import subprocess

    rdir = ink_spec.remote_ledger_dir()
    rdir.mkdir(parents=True, exist_ok=True)
    rc = 0
    for target in args.nodes:
        node = target.split("@")[-1].split(".")[0]
        dest = rdir / f"{node}.jsonl"
        src = f"{target}:{args.remote_path}"
        cmd = ["scp", "-q", "-o", "BatchMode=yes", "-o", "ConnectTimeout=8", src, str(dest)]
        r = subprocess.run(cmd, check=False, capture_output=True, text=True)
        if r.returncode != 0:
            print(f"  {target}: {r.stderr.strip() or 'copy failed'}", file=sys.stderr)
            rc = 1
            continue
        n = len(ink_spec.read_events(dest, include_remote=False))
        print(f"  {target}: {n} event(s) -> {dest}")
    evs = ink_spec.read_events()
    print(f"ledger now sees {len(evs)} event(s) across local + {len(list(rdir.glob('*.jsonl')))} remote file(s)")
    return rc


def _tool_arg(p: argparse.ArgumentParser) -> None:
    """The tool in the gripper — stated on the command line, or by the CLI
    through TATBOT_EE_TOOL (`tatbot --ee-tool <id> ink …`); never inferred."""
    p.add_argument("--ee-tool", "--tool-id", dest="tool_id", default=os.environ.get("TATBOT_EE_TOOL") or None,
                   help="the tool in the gripper (default: $TATBOT_EE_TOOL)")


def _require_tool(args, what: str) -> str | None:
    if not args.tool_id:
        return f"{what} needs --ee-tool <id> (or TATBOT_EE_TOOL): name the tool in the gripper"
    return None


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    sub.add_parser("status").set_defaults(fn=cmd_status)

    p = sub.add_parser("load")
    p.set_defaults(fn=cmd_load)
    p.add_argument("slot")
    p.add_argument("ink")
    p.add_argument("--ul", type=float, required=True)
    p.add_argument("--bottle")
    p.add_argument("--cap-stock")

    p = sub.add_parser("dump")
    p.set_defaults(fn=cmd_dump)
    p.add_argument("slot")

    p = sub.add_parser("bottle")
    p.set_defaults(fn=cmd_bottle)
    p.add_argument("verb", choices=["add", "open", "retire"])
    p.add_argument("id")
    p.add_argument("--ink")
    p.add_argument("--ml", type=float)
    p.add_argument("--purchased")
    p.add_argument("--note")

    p = sub.add_parser("cartridge")
    p.set_defaults(fn=cmd_cartridge)
    p.add_argument("verb", choices=["add", "fit", "count", "retire"])
    p.add_argument("id")
    p.add_argument("n", nargs="?", type=int, default=1)
    p.add_argument("--spec")
    p.add_argument("--needle-code")
    p.add_argument("--count", type=int)
    p.add_argument("--fits")
    p.add_argument("--expires")

    p = sub.add_parser("caps")
    p.set_defaults(fn=cmd_caps)
    p.add_argument("verb", choices=["count"])
    p.add_argument("id")
    p.add_argument("n", type=int)

    p = sub.add_parser("weigh")
    p.set_defaults(fn=cmd_weigh)
    p.add_argument("target")
    p.add_argument("grams", type=float)
    p.add_argument("--when", choices=["before", "after"], required=True)

    p = sub.add_parser("ledger")
    p.set_defaults(fn=cmd_ledger)
    p.add_argument("--since")
    p.add_argument("--mode", choices=["real", "rehearsal", "sim"])
    p.add_argument("-n", type=int, default=0)

    p = sub.add_parser("reconcile")
    p.set_defaults(fn=cmd_reconcile)
    p.add_argument("--write", action="store_true")

    p = sub.add_parser("mise-en-place")
    p.set_defaults(fn=cmd_mise)
    _tool_arg(p)
    p.add_argument("--arm", default="right")
    p.add_argument("--need", nargs="*", metavar="INK=UL", help="session need per ink, microlitres")
    p.add_argument("--ink", help="with --strokes-mm: the one ink the session draws with")
    p.add_argument("--strokes-mm", type=float, help="estimate the need from planned stroke length")
    p.add_argument("--program", help="planned strokes as JSON (run_meta.json, an episode entry, "
                                     "a language program, or a list of [x,y] polylines in metres)")
    p.add_argument("--speed-mm-s", type=float, default=30.0)

    p = sub.add_parser("fit")
    p.set_defaults(fn=cmd_fit)
    p.add_argument("--density-g-per-ml", type=float, default=1.0)

    p = sub.add_parser("plan")
    p.set_defaults(fn=cmd_plan)
    _tool_arg(p)
    p.add_argument("--arm", default="right")
    p.add_argument("--strokes", nargs="+", required=True, metavar="MM,S[,INK]")

    p = sub.add_parser("session")
    p.set_defaults(fn=cmd_session)
    p.add_argument("verb", nargs="?", choices=["status", "start", "end", "rebuild"])
    p.add_argument("id", nargs="?")
    _tool_arg(p)
    p.add_argument("--need-ul", type=float)
    p.add_argument("--program")
    p.add_argument("--speed-mm-s", type=float, default=30.0)
    p.add_argument("--note")
    p.add_argument("--force", action="store_true", help="end an open session first")
    p.add_argument("--write", action="store_true", help="rebuild: also write session.json")

    p = sub.add_parser("sync")
    p.set_defaults(fn=cmd_sync)
    p.add_argument("nodes", nargs="+", help="ssh targets (node name or user@host)")
    p.add_argument("--remote-path", default="tatbot-logs/ink/ledger.jsonl")

    args = ap.parse_args(argv)
    if args.cmd in ("mise-en-place", "plan"):
        why = _require_tool(args, args.cmd)
        if why:
            return _die(why)
    if args.cmd == "bottle" and args.verb == "add" and (not args.ink or args.ml is None):
        return _die("bottle add needs --ink and --ml")
    if args.cmd == "cartridge" and args.verb == "add" and (not args.spec or not args.needle_code or args.count is None):
        return _die("cartridge add needs --spec, --needle-code and --count")
    return args.fn(args)


if __name__ == "__main__":
    sys.exit(main())
