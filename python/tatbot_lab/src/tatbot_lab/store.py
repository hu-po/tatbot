"""The lab's records on disk, outside the repo: one directory per study under the lab root.

    <root>/<study>/study.json          the study (tatbot.lab-study/1), written once
    <root>/<study>/trials.jsonl        trial records, appended; the last line for an id is that trial now
    <root>/<study>/plans/<plan>/       a plan's runs: programs, cells manifests, its summary plan.json

The root is TATBOT_LAB_ROOT, else ~/tatbot-lab on the node running the lab.
"""
from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path

from tatbot_contracts import lab


def root() -> Path:
    return Path(os.environ.get("TATBOT_LAB_ROOT") or Path.home() / "tatbot-lab").expanduser()


def now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def study_dir(study_id: str) -> Path:
    return root() / lab._id(study_id, "study id")


def create(record: dict) -> Path:
    lab.study(record)
    directory = study_dir(record["id"])
    if (directory / "study.json").exists():
        raise ValueError(f"study {record['id']} exists at {directory}; a study is written once")
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "study.json").write_text(json.dumps(record, indent=1) + "\n")
    return directory


def load(study_id: str) -> dict:
    path = study_dir(study_id) / "study.json"
    if not path.exists():
        raise ValueError(f"no study {study_id} under {root()}")
    return lab.study(json.loads(path.read_text()))


def trials(study_id: str) -> dict[str, dict]:
    """id -> the trial's latest record, in the order trials were first written."""
    path = study_dir(study_id) / "trials.jsonl"
    out: dict[str, dict] = {}
    if path.exists():
        for n, line in enumerate(path.read_text().splitlines(), 1):
            if line.strip():
                record = json.loads(line)
                try:
                    out[record["id"]] = lab.trial(record)
                except (ValueError, KeyError) as error:
                    raise ValueError(f"{path}:{n}: {error}") from error
    return out


def append(study_id: str, records: list[dict]) -> None:
    """Write new or updated trial records; every one is validated before any is written."""
    study = load(study_id)
    for record in records:
        lab.trial(record, study)
    with (study_dir(study_id) / "trials.jsonl").open("a") as f:
        for record in records:
            f.write(json.dumps(record, separators=(",", ":")) + "\n")


def next_ids(existing, count: int, prefix: str) -> list[str]:
    taken = {int(i[len(prefix):]) for i in existing if i.startswith(prefix) and i[len(prefix):].isdigit()}
    start = max(taken, default=0) + 1
    return [f"{prefix}{n:04d}" for n in range(start, start + count)]


def plans_dir(study_id: str) -> Path:
    return study_dir(study_id) / "plans"
