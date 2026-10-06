"""Bounded subprocess bridge to the browser's canonical SVG paint adapter."""

from __future__ import annotations

import json
import shutil
import subprocess
from typing import Any

from tatbot_sim.human_rep.contracts import ContractError
from tatbot_sim.repo import repo_root


def artwork_request(document: dict[str, Any]) -> dict[str, Any]:
    executable = shutil.which("node")
    if executable is None:
        raise ContractError("artwork_runtime_unavailable", "$.svg", "Node.js 22 and Inkmap dependencies are required")
    payload = json.dumps(document, allow_nan=False)
    if len(payload.encode()) > 30_000_000:
        raise ContractError("tattoo_program_unsupported", "$.svg", "artwork request exceeds 30 MB")
    try:
        completed = subprocess.run(
            [executable, "--experimental-strip-types", str(repo_root() / "web/inkmap/tools/artwork.ts")],
            input=payload, capture_output=True, text=True, timeout=60, check=False,
            cwd=repo_root(),
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise ContractError("artwork_runtime_unavailable", "$.svg", str(exc)) from exc
    if completed.returncode:
        for line in completed.stderr.splitlines():
            try:
                failure = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(failure, dict) and "code" in failure:
                raise ContractError(failure["code"], failure.get("path", "$.svg"), failure.get("detail", "artwork refused"))
        if "ERR_NO_TYPESCRIPT" in completed.stderr:
            # Named separately because `npm ci` cannot fix it: Debian/Ubuntu build
            # node 22 without Amaro, so --experimental-strip-types is accepted and
            # then refuses to strip. Reported as a missing adapter otherwise.
            raise ContractError(
                "artwork_runtime_unavailable", "$.svg",
                "this node's Node.js is built without TypeScript support "
                "(ERR_NO_TYPESCRIPT); the shared SVG adapter is a .ts module, so it "
                "needs a Node.js 22 build with type stripping",
            )
        raise ContractError(
            "artwork_runtime_unavailable", "$.svg",
            "shared SVG adapter failed; run npm ci in web/inkmap: " + completed.stderr[:1000],
        )
    if len(completed.stdout) > 30_000_000:
        raise ContractError("tattoo_program_unsupported", "$.svg", "artwork exceeds output budget")
    try:
        result = json.loads(completed.stdout)
    except json.JSONDecodeError as exc:
        raise ContractError("artwork_runtime_unavailable", "$.svg", "adapter returned invalid JSON") from exc
    if not isinstance(result, dict):
        raise ContractError("artwork_runtime_unavailable", "$.svg", "adapter returned a non-object")
    return result
