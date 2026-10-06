"""Materialize Inkgen rasters as immutable SVG design artifacts.

This is an explicit pre-simulation stage. Network and model work end here;
scenario compilation and the environment consume only the SVG bytes written
to disk.

The bookkeeping — request identity, retained rasters, resume, retries, the job
lock — belongs to `web/inkgen/batch.py`, which knows nothing about artwork. What
lives here is the half that does: Inkmap's own tracer, the artwork record the
editor's reader validates, and the `tatbot.inkgen-materialization/1` manifest
the source-image workflow records. Native DBV3 acquisition is required before simulation. That manifest is still published only when every
requested slot ended with one accepted artwork; a short job finalizes as an
explicit selection instead, and the source-selection reader refuses it.
"""

from __future__ import annotations

import base64
import hashlib
import json
import sys
import urllib.error
import urllib.request
from collections.abc import Callable, Sequence
from pathlib import Path

import numpy as np

from tatbot_sim.inkmap.design_build import (
    MAX_IMAGE_BYTES,
    TRACE_MODEL,
    DesignBuildError,
    decode_image,
    fit_size_mm,
    trace_raster_svg,
)
from tatbot_sim.inkmap.designs import DesignArtifact
from tatbot_sim.repo import repo_root

MATERIALIZATION_SCHEMA = "tatbot.inkgen-materialization/1"
MAX_PNG_BYTES = MAX_IMAGE_BYTES
DEFAULT_MAX_ATTEMPTS = 3

__all__ = ["InkgenMaterializationError", "materialize_inkgen_designs", "materialize_job",
           "trace_raster_svg", "TRACE_MODEL", "MATERIALIZATION_SCHEMA"]


class InkgenMaterializationError(DesignBuildError):
    """A remote reply was not usable as a design.

    Tracing refusals come from the shared builder as DesignBuildError; this
    names the part that is still this module's own, the Inkgen exchange.
    """


def batch_module():
    """The generator's own job module, which is stdlib-only and standalone.

    Imported by path rather than vendored: the Space ships the same file, and a
    second copy of resume semantics is exactly the drift this plan set out to
    remove.
    """
    path = repo_root() / "web/inkgen"
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))
    import batch

    return batch


def _post_json(url: str, payload: dict, timeout_s: float) -> dict:
    request = urllib.request.Request(
        url.rstrip("/") + "/api/generate",
        data=json.dumps(payload, separators=(",", ":")).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout_s) as response:  # noqa: S310
            raw = response.read(MAX_PNG_BYTES * 2)
    except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError) as exc:
        raise InkgenMaterializationError(f"Inkgen request failed: {exc}") from exc
    try:
        body = json.loads(raw)
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise InkgenMaterializationError("Inkgen returned invalid JSON") from exc
    if not isinstance(body, dict) or body.get("error"):
        raise InkgenMaterializationError(f"Inkgen rejected the request: {body.get('error', body)!r}")
    return body


def _decode_png(value: object) -> tuple[bytes, np.ndarray]:
    if not isinstance(value, str):
        raise InkgenMaterializationError("Inkgen reply has no png_base64 string")
    try:
        encoded = base64.b64decode(value, validate=True)
    except (ValueError, TypeError) as exc:
        raise InkgenMaterializationError("Inkgen reply contains invalid base64") from exc
    try:
        return encoded, decode_image(encoded)
    except DesignBuildError as exc:
        raise InkgenMaterializationError(f"Inkgen payload is not usable: {exc}") from exc


def http_generator(api_url: str, *, timeout_s: float = 120.0,
                   request_json: Callable[[str, dict, float], dict] = _post_json,
                   ensure: Callable[[], str] | None = None):
    """Ask an already-selected generator for one image per normalized request.

    `ensure` runs at the first request that actually needs the generator, not
    before: a job whose rasters are all cached — a retrace at a new artwork
    size, say — has no reason to demand a live one.
    """
    state = {"url": api_url}

    def generate(request):
        if ensure is not None and state.get("ensured") is None:
            state["url"] = ensure()
            state["ensured"] = True
        api_url = state["url"]
        payload = {"subject": request.subject, "seed": request.seed}
        if request.style:
            payload["style"] = request.style
        reply = request_json(api_url, payload, timeout_s)
        png, _ = _decode_png(reply.get("png_base64"))
        return png, {"seed": int(reply.get("seed", request.seed)),
                     "prompt": str(reply.get("prompt") or request.prompt),
                     "model": str(reply.get("model") or request.settings.model),
                     "model_revision": reply.get("model_revision") or request.settings.model_revision,
                     "settings": reply.get("settings") or request.settings.as_json(),
                     "seconds": reply.get("seconds")}
    return generate


def artwork_converter(size_mm: tuple[float, float] = (50.0, 50.0)):
    """Trace one retained raster into the artifact triple a library holds."""
    batch = batch_module()

    def convert(png: bytes, meta: dict, item) -> dict:
        try:
            _, image = _decode_png(base64.b64encode(png).decode())
            svg, trace = trace_raster_svg(image)
        except DesignBuildError as exc:
            # The bytes are retained and the words will not change: another
            # identical request would draw the same unusable picture.
            raise batch.ConversionRefusedError(str(exc)) from exc
        used_seed = int(meta.get("seed", item.seed))
        model = str(meta.get("model") or "inkgen")
        source = {
            "kind": "generated",
            "model": model,
            "model_revision": meta.get("model_revision"),
            "subject": item.subject,
            "style": item.style,
            "seed": used_seed,
            "prompt": str(meta.get("prompt") or item.subject),
            "png_sha256": hashlib.sha256(png).hexdigest(),
            "trace": trace,
        }
        digest = hashlib.sha256(svg.encode()).hexdigest()
        artifact = DesignArtifact(id=f"gen-{digest[:16]}", name=f"{item.subject} {used_seed}",
                                  svg=svg, size_mm=fit_size_mm(tuple(trace["size_px"]), size_mm),
                                  source=source)
        from tatbot_sim.inkmap.artwork import make_artwork_record
        make_artwork_record(
            name=artifact.name, original_svg=svg,
            source={"kind": "generated", "identifier": artifact.id, "license": None, "attribution": None,
                    "generation": {"prompt": source["prompt"], "model": model,
                                   "model_revision": source["model_revision"], "seed": used_seed,
                                   "tracing": TRACE_MODEL}},
            conversion={"adapter": "tatbot-svg-paint/1", "canvas_m": [v / 1000 for v in artifact.size_mm],
                        "semantic_intent": item.subject, "width_m": .0003, "deposition": 1,
                        "chord_error_m": .000005})
        stem = f"{item.ordinal:04d}-{artifact.id}"
        record = {"schema": MATERIALIZATION_SCHEMA, "id": artifact.id, "name": artifact.name,
                  "sha256": artifact.sha256, "size_mm": list(artifact.size_mm), "source": source,
                  "svg": f"{stem}.svg", "png": f"{stem}.png"}
        return {"artifacts": {f"{stem}.svg": svg, f"{stem}.png": png,
                              f"{stem}.json": json.dumps(record, indent=2, sort_keys=True) + "\n"},
                "record": record}
    return convert


def _publish_manifest(job, api_url: str) -> dict:
    """The complete-library manifest, written once and only when it is true."""
    records = [{key: value for key, value in row.items() if key not in {"key", "ordinal"}}
               for row in job.library()]
    manifest = {"schema": MATERIALIZATION_SCHEMA, "api_url": api_url,
                "seed": int(job.request["seed"]), "requested": len(records),
                "job_id": job.request["job_id"], "settings": job.request["settings"],
                "report": job.report(), "artifacts": records}
    path = job.root / "manifest.json"
    temporary = path.with_name(f".manifest.{job.request['job_id'][:12]}.tmp")
    temporary.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)
    return manifest


def materialize_job(
    output_dir: Path,
    subjects: Sequence[str],
    *,
    count: int,
    seed: int,
    api_url: str,
    style: str | None = None,
    size_mm: tuple[float, float] = (50.0, 50.0),
    timeout_s: float = 120.0,
    request_json: Callable[[str, dict, float], dict] = _post_json,
    ensure: Callable[[], str] | None = None,
    model: str | None = None,
    model_revision: str | None = None,
    replacement_budget: int = 0,
    max_attempts: int = DEFAULT_MAX_ATTEMPTS,
    backend: str = "endpoint",
    stop: Callable[[], bool] | None = None,
    on_event: Callable[..., None] | None = None,
) -> dict:
    """Run (or resume) a generation job and publish its artwork library."""
    batch = batch_module()
    from contracts import GenerationSettings

    if not api_url.startswith(("http://", "https://")):
        raise InkgenMaterializationError("Inkgen API URL must start with http:// or https://")
    if not all(np.isfinite(size_mm)) or min(size_mm) <= 0:
        raise InkgenMaterializationError(f"design size must be positive and finite, got {size_mm!r}")
    output_dir = Path(output_dir)
    if (output_dir / ".job/request.json").is_file():
        # The frozen job already records the weights it was cut against. Taking
        # them from there is what makes a resume a resume: rebuilding them from
        # this process's environment would mint a different job id and the
        # directory would refuse its own continuation.
        settings = GenerationSettings(**batch.load_job(output_dir)["settings"])
    else:
        defaults = GenerationSettings.from_env()
        settings = GenerationSettings(model=model or defaults.model,
                                      model_revision=model_revision or defaults.model_revision,
                                      width=defaults.width, height=defaults.height,
                                      steps=defaults.steps, guidance=defaults.guidance)
    try:
        items = batch.plan_items(subjects, count, seed=seed, style=style)
        request = batch.freeze_request(items, settings=settings, seed=seed, backend=backend,
                                       replacement_budget=replacement_budget)
    except ValueError as exc:
        raise InkgenMaterializationError(str(exc)) from exc
    # What the conversion is, as opposed to what was asked of the model. A job
    # rerun at a different artwork size retraces from the rasters it already
    # holds; the generation cache is keyed by the request and stays valid.
    from contracts import canonical_digest

    conversion_key = canonical_digest({"adapter": "tatbot-svg-paint/1", "tracer": TRACE_MODEL,
                                       "size_mm": [float(value) for value in size_mm],
                                       "width_mm": 0.3})
    with batch.Job.open(output_dir, request) as job:
        status = job.run(generate=http_generator(api_url, timeout_s=timeout_s,
                                                 request_json=request_json, ensure=ensure),
                         convert=artwork_converter(size_mm), conversion_key=conversion_key,
                         max_attempts=max_attempts, stop=stop, on_event=on_event)
        if job.is_complete():
            return _publish_manifest(job, api_url)
        return {"schema": batch.SELECTION_SCHEMA, "api_url": api_url,
                **job.finalize_selection(reason="job did not fill every requested slot"),
                "cancelled": status.get("cancelled", False)}


def materialize_inkgen_designs(
    output_dir: Path,
    subjects: Sequence[str],
    *,
    count: int,
    seed: int,
    api_url: str,
    style: str | None = None,
    size_mm: tuple[float, float] = (50.0, 50.0),
    timeout_s: float = 120.0,
    autostart: bool = False,
    request_json: Callable[[str, dict, float], dict] = _post_json,
    **job_options,
) -> dict:
    """Generate, trace, validate, and write a bounded artifact directory.

    The compatible spelling: same arguments, same manifest, now resumable. An
    all-or-nothing caller still gets all or nothing — a job that does not fill
    every slot raises rather than handing back a short library that looks whole.
    """
    if autostart:
        # The generator stops itself when idle, so a batch that needs it starts
        # one through the same idempotent path `tatbot design generate` uses,
        # and only when that address is the fleet worker's.
        from tatbot_sim.inkmap.inkgen_client import ensure_backend, resolve_backend

        backend, _ = ensure_backend(resolve_backend(api_url, require_configured=True))
        api_url = backend.url
    result = materialize_job(output_dir, subjects, count=count, seed=seed, api_url=api_url,
                             style=style, size_mm=size_mm, timeout_s=timeout_s,
                             request_json=request_json, **job_options)
    if result.get("schema") != MATERIALIZATION_SCHEMA:
        raise InkgenMaterializationError(
            f"job filled {result.get('accepted', 0)} of {result.get('requested', count)} requested "
            f"slots ({result.get('refused', 0)} refused, {result.get('failed', 0)} failed, "
            f"{result.get('duplicate', 0)} duplicate); resume it or accept the subset explicitly")
    return result
