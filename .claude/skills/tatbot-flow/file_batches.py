"""File a Flow zip's images under CORPUS/<prompt id>/ with one generation sidecar each.

usage: python3 -I file_batches.py EXTRACTED_DIR CORPUS_DIR PROMPTS_JSON

PROMPTS_JSON is the plan the prompts were typed from: {generator, model, aspect, project, prompts:
[{id, style, subject, text, variant?}]}. A batch whose read-back prompt matches no planned prompt
exactly is filed as `unmatched-<n>` with `merged_prompt: true`: what Flow actually generated from is
the page's text, never the text meant to be typed. Images already in the corpus (same sha256) are
skipped, so collecting a whole project again files only what is new.
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path


def _norm(text: str) -> str:
    return " ".join(text.split())


def main(raw: Path, corpus: Path, plan_path: Path) -> int:
    plan = json.loads(plan_path.read_text())
    batches = json.loads((raw / "batches.json").read_text())
    by_text = {_norm(p["text"]): p for p in plan["prompts"]}
    have = {json.loads(f.read_text())["image"]["sha256"] for f in corpus.glob("*/*.json")}
    unmatched = 0
    for batch in batches["batches"]:
        prompt = by_text.get(_norm(batch["prompt"]))
        images = [(im, (raw / im["file"]).read_bytes()) for im in batch["images"]]
        images = [(im, data) for im, data in images if hashlib.sha256(data).hexdigest() not in have]
        if not images:
            continue
        if prompt is None:
            unmatched += 1
        pid = prompt["id"] if prompt else f"unmatched-{unmatched}"
        out = corpus / pid
        out.mkdir(parents=True, exist_ok=True)
        start = len(list(out.glob("*.jpg")))
        for k, (_image, data) in enumerate(images):
            name = f"{pid}-{start + k + 1}.jpg"
            (out / name).write_bytes(data)
            sidecar = {
                "schema": "tatbot.art-generation/1", "generator": plan.get("generator", "flow-ui"),
                "model": plan.get("model"), "aspect": plan.get("aspect"), "project": plan.get("project"),
                "prompt": batch["prompt"], "prompt_id": prompt and prompt["id"],
                "style": prompt and prompt.get("style"), "subject": prompt and prompt.get("subject"),
                "variant": prompt and prompt.get("variant"), "merged_prompt": prompt is None,
                "created": batch.get("created"), "fetched_utc": batches.get("fetched_utc"),
                "references": plan.get("references", []),
                "image": {"file": name, "sha256": hashlib.sha256(data).hexdigest(), "format": "jpeg"},
            }
            (out / f"{name[:-4]}.json").write_text(json.dumps(sidecar, indent=1) + "\n")
        print(pid, len(images), "MERGED" if prompt is None else "")
    return 0


if __name__ == "__main__":
    sys.exit(main(Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])))
