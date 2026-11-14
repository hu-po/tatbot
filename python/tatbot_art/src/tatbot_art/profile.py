"""Explain a saved program's time as a standalone report, without rerunning preparation."""
from __future__ import annotations

import hashlib
import json
import math
from collections import Counter
from html import escape
from pathlib import Path

from tatbot_contracts.ros_program import validate_for_execution

from .timing import minutes, workflow

LABELS = {"drawing": "Drawing", "descent": "Descent", "settle": "Settling", "lift": "Lifting",
          "travel": "Travel between strokes", "dip": "Dipping", "rinse": "Rinsing", "swap": "Cartridge swaps",
          "pen_change": "Resource changes (compiler)", "compiler_subtotal": "Compiler subtotal (unitemized)"}


def _length_mm(op: dict) -> float:
    points = op["points_m"]
    length = sum(math.dist(a, b) for a, b in zip(points, points[1:], strict=False))
    if op.get("closed"):
        length += math.dist(points[-1], points[0])
    return length * 1000


def _resources(program: dict, steps: list[dict]) -> list[dict]:
    rows = []
    for resource in program["resources"]:
        rid = resource["id"]
        strokes = [o for o in program["ops"] if o["op"] == "stroke" and o["resource_id"] == rid]
        counts = Counter(s["kind"] for s in steps if s["resource_id"] == rid)
        rows.append({"id": rid, "ink": resource["ink_id"], "tool": resource["tool"]["id"],
                     "strokes": len(strokes), "touchdowns": sum(not s.get("continues", False) for s in strokes),
                     "contact_mm": round(sum(_length_mm(s) for s in strokes), 2),
                     **{k: counts[k] for k in ("dip", "rinse", "swap", "selection")}})
    return rows


def build(program: dict, *, gaps_s: dict | None = None) -> dict:
    """Use the saved duration/2 components, never current motion config or a second timing model."""
    validate_for_execution(program)
    estimate = (program.get("stats") or {}).get("time_estimate") or {}
    if program.get("rehearsal") or estimate.get("model") != "tatbot-duration/2":
        raise ValueError("profile needs an original compiled program with its tatbot-duration/2 estimate")
    counts = Counter("pen_change" if o["op"] == "tool_change" else o["op"]
                     for o in program["ops"] if o["op"] != "stroke")
    if dict(counts) != estimate.get("workflow_counts"):
        raise ValueError("program operations differ from its recorded timing; compile it again")
    steps = workflow(program)
    time = minutes(program, gaps_s)
    rows = [{"id": k, "label": LABELS.get(k, k), "seconds": v,
             "share": round(v / time["estimated_s"], 6) if time["estimated_s"] else 0,
             "basis": time["basis"][k]} for k, v in time["components_s"].items()]
    resources = _resources(program, steps)
    return {"schema": "tatbot.art-time-profile/1", "time": time,
            "components": sorted(rows, key=lambda r: -r["seconds"]), "resources": resources,
            "totals": {k: sum(r[k] for r in resources) for k in ("strokes", "touchdowns", "contact_mm")},
            "motion_sha256": estimate.get("motion_sha256"),
            "scope": estimate.get("scope", "Recorded compiler estimate"),
            "excludes": "Page setup, IK retiming, tracking holds, interruptions and recovery; not a live ETA."}


def text(report: dict) -> str:
    time = report["time"]
    qualifier = "" if time["complete"] else "at least "
    lines = [f"Estimated program time: {qualifier}{time['minutes']:.1f} min", "",
             f"{'Component':<32} {'Minutes':>9} {'Share':>8}  Basis"]
    for row in report["components"]:
        lines.append(f"{row['label']:<32} {row['seconds']/60:>9.2f} {row['share']:>7.1%}  {row['basis']}")
    lines += ["", f"{report['totals']['strokes']} stroke chunks, {report['totals']['touchdowns']} touchdowns; "
              f"{time['dips']} dips, {time['rinses']} rinses, {time['swaps']} swaps",
              f"Scope: {report['scope']}", report["excludes"]]
    if time["unknown_operations"]:
        lines.append(f"Unpriced operations: {time['unknown_operations']}")
    return "\n".join(lines)


def html(report: dict, name: str) -> str:
    def safe(value):
        return escape(str(value), quote=True)

    components = "".join(
        f"<tr><th scope='row'>{safe(r['label'])}<span class='bar' style='width:{r['share']*100:.3f}%'></span></th>"
        f"<td>{r['seconds']/60:.2f}</td><td>{r['share']:.1%}</td><td>{safe(r['basis'])}</td></tr>"
        for r in report["components"])
    resources = "".join(
        f"<tr><th scope='row'>{safe(r['id'])}<small>{safe(r['ink'])} · {safe(r['tool'])}</small></th>"
        f"<td>{r['contact_mm']:.1f}</td><td>{r['strokes']}</td><td>{r['touchdowns']}</td>"
        f"<td>{r['dip']}</td><td>{r['rinse']}</td><td>{r['swap']}</td></tr>" for r in report["resources"])
    time = report["time"]
    missing = f"<p>Unpriced operations: {safe(time['unknown_operations'])}</p>" if not time["complete"] else ""
    prefix = "" if time["complete"] else "At least "
    return f"""<!doctype html>
<html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Drawing time · {safe(name)}</title>
<style>
*{{box-sizing:border-box}}body{{margin:0;background:#f4f2ec;color:#20352f;font:16px/1.55 system-ui,sans-serif}}
main{{max-width:1100px;margin:48px auto;padding:0 24px}}h1{{font-size:36px;margin:8px 0}}h2{{margin-top:36px}}
.eyebrow{{text-transform:uppercase;letter-spacing:.15em;font-size:12px}}.total{{font-size:64px;font-weight:650}}
.total small{{font-size:18px;line-height:1.4}}
.panel{{background:white;padding:24px;border-radius:12px;margin-top:24px}}.scroll{{overflow-x:auto}}
table{{border-collapse:collapse;width:100%;font-variant-numeric:tabular-nums}}th,td{{padding:12px;text-align:right;
border-bottom:1px solid #e0e7e3}}th:first-child{{text-align:left;min-width:220px}}td:last-child{{text-align:left}}
thead th{{font-size:12px;color:#52655b;text-transform:uppercase}}small{{display:block;font-weight:400;color:#52655b}}
.bar{{display:block;background:#4f866c;height:5px;border-radius:4px;margin-top:7px;min-width:0}}
.note{{color:#52655b}}code{{overflow-wrap:anywhere;font-size:12px}}footer{{margin:24px 0;font-size:13px}}
@media(max-width:600px){{main{{margin:24px auto;padding:0 16px}}h1{{font-size:28px}}.panel{{padding:16px}}}}
</style><main><div class="eyebrow">Tatbot · offline drawing-time profile</div>
<h1>{safe(name)}</h1><div class="total">{prefix}{time['minutes']:.1f} <small>estimated minutes</small></div>
<p>{report['totals']['contact_mm']/1000:.2f} m of ink paths · {report['totals']['strokes']} stroke chunks ·
{time['dips']} dips · {time['rinses']} rinses · {time['swaps']} cartridge swaps</p>
<p class="note">{safe(report['excludes'])}</p>{missing}
<section class="panel"><h2>Where the time goes</h2><div class="scroll"><table>
<thead><tr><th>Component</th><th>Minutes</th><th>Share</th><th>Basis</th></tr></thead>
<tbody>{components}</tbody></table></div></section>
<section class="panel"><h2>What drives the costs</h2><p class="note">Dips and rinses use fixed gaps only where
the compiler left their costs out. Initial selections and changes keeping the same cartridge and ink add no
rinse or swap. Touchdowns exclude chunks that continue pen-down. Counts explain costs; they do not change paths.</p>
<div class="scroll"><table><thead><tr><th>Resource / ink / cartridge</th><th>Path mm</th><th>Chunks</th>
<th>Touchdowns</th><th>Dips</th><th>Rinses</th><th>Swaps</th></tr></thead><tbody>{resources}</tbody></table></div></section>
<footer><p>Model scope: {safe(report['scope'])}</p><p>Program SHA-256: <code>{safe(report['source']['sha256'])}</code>
<br>Motion SHA-256: <code>{safe(report['motion_sha256'])}</code></p></footer></main></html>"""


def write(source: Path, out: Path, gaps_s: dict | None = None) -> dict:
    if source.resolve() in {(out / f"profile.{ext}").resolve() for ext in ("json", "html")}:
        raise ValueError("profile output would overwrite the input program")
    raw = source.read_bytes()
    report = build(json.loads(raw), gaps_s=gaps_s)
    report["source"] = {"name": source.name, "sha256": hashlib.sha256(raw).hexdigest()}
    out.mkdir(parents=True, exist_ok=True)
    (out / "profile.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    (out / "profile.html").write_text(html(report, source.parent.parent.name + " / " + source.name))
    return report
