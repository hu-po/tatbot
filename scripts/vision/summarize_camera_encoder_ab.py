#!/usr/bin/env python3
"""Summarize the structured latency lines from camera_encoder_ab.sh logs."""

from __future__ import annotations

import json
import pathlib
import statistics
import sys

FIELDS = {
    "pipeline_capture_age_ms": None,
    "pipeline_stage_latency_ms": "source_pts_to_rtsp_rtp_first",
    "capture_event_channel_wait_ms": None,
    "synchronizer_wait_ms": None,
    "fiducial_processing_ms": None,
}


def structured_line(log: pathlib.Path, key: str):
    prefix = f"{key}="
    for line in log.read_text(errors="replace").splitlines():
        if line.startswith(prefix):
            return json.loads(line[len(prefix) :])
    return None


def phase_values(root: pathlib.Path, phase: str, camera: str):
    rows = []
    for log in sorted((root / phase).glob("run-*.log")):
        row = {"run": log.name}
        for field, stage in FIELDS.items():
            value = structured_line(log, field)
            for statistic in ("median", "p95"):
                key = f"{field}.{statistic}"
                if value is None:
                    row[key] = None
                elif field in {"pipeline_capture_age_ms", "capture_event_channel_wait_ms"}:
                    row[key] = value.get(camera, {}).get(statistic)
                elif field == "pipeline_stage_latency_ms":
                    row[key] = value.get(camera, {}).get(stage, {}).get(statistic)
                else:
                    row[key] = value.get(statistic) if value else None
        rows.append(row)
    return rows


def median(values):
    values = [value for value in values if value is not None]
    return statistics.median(values) if values else None


def fmt(value):
    return "missing" if value is None else f"{value:.2f}"


def main() -> int:
    if len(sys.argv) != 3:
        raise SystemExit("usage: summarize_camera_encoder_ab.py EXPERIMENT CAMERA")
    root, camera = pathlib.Path(sys.argv[1]), sys.argv[2]
    phases = ["a_before", "b_variant", "a_restored"]
    data = {phase: phase_values(root, phase, camera) for phase in phases}
    print(f"# Camera encoder A/B/A summary: {camera}\n")
    print("Medians are medians across per-run medians, in milliseconds.\n")
    print("| Metric | A before | B variant | A restored | B vs paired A |")
    print("| --- | ---: | ---: | ---: | ---: |")
    for field in FIELDS:
        for statistic in ("median", "p95"):
            key = f"{field}.{statistic}"
            values = {phase: median([row[key] for row in rows]) for phase, rows in data.items()}
            paired_a = median(
                [
                    value
                    for phase in ("a_before", "a_restored")
                    for value in [values[phase]]
                    if value is not None
                ]
            )
            delta = (
                None if paired_a is None or values["b_variant"] is None else values["b_variant"] - paired_a
            )
            print(
                f"| `{key}` | {fmt(values['a_before'])} | {fmt(values['b_variant'])} | "
                f"{fmt(values['a_restored'])} | {fmt(delta)} |"
            )
    print(
        "\nAcceptance remains manual: require at least 20 ms improvement in both "
        "median and p95, no errors/drops/regressions, stable phone-event skew, "
        "and unchanged pose quality."
    )
    (root / "summary.json").write_text(json.dumps(data, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
