"""A generation job that survives being interrupted.

A batch of a thousand images is not one long request; it is a thousand small
ones with a bookkeeping problem. This module owns the bookkeeping and nothing
else — it does not know how to draw (that is `engine.py`), and it does not know
what an artwork is (the caller supplies a converter). Stdlib only, so it runs on
the Space payload, on a fleet worker, and in a test with no GPU anywhere.

Three separate things live under one job root, and conflating them is the bug
this layout exists to prevent:

    .job/request.json   the frozen request manifest — what was asked for, once
    .job/ledger.json    the mutable ledger — where each item actually got to
    .job/raw/           retained model output, keyed by request digest
    <root>/*.svg|png    the accepted artwork library, published atomically
    <root>/manifest.json only written when the whole job succeeded

A raw cache is not a library. A completed item is not an accepted artwork. A
23-of-24 job is not a success — `manifest.json` appears only when every item
was accepted, and an explicitly chosen subset is `selection.json`, a different
document with the rejected and failed counts written into it.
"""
from __future__ import annotations

import fcntl
import hashlib
import json
import os
import re
import time
import uuid
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, field
from pathlib import Path

from contracts import (
    PRODUCER,
    PRODUCER_VERSION,
    SEED_MAX,
    GenerationRequest,
    GenerationSettings,
    RequestError,
    canonical_digest,
    canonical_json,
)

JOB_SCHEMA = "tatbot.inkgen-job/1"
LEDGER_SCHEMA = "tatbot.inkgen-job-ledger/1"
SELECTION_SCHEMA = "tatbot.inkgen-selection/1"

MAX_ITEMS = 10_000
MAX_ERROR_CHARS = 400
DEFAULT_MAX_ATTEMPTS = 3

# pending  -> nothing has been asked of the generator yet
# running  -> a worker claimed it; a crash leaves it here and resume reclaims it
# generated-> raster retained and verified; conversion has not run or not finished
# accepted -> converted, validated and published into the library
# duplicate-> byte-identical to an earlier accepted item; kept, not published twice
# refused  -> the output was not usable as artwork, and retrying it will not help
# failed   -> attempts exhausted against a transport or engine error
TERMINAL = {"accepted", "duplicate", "refused", "failed"}
STATES = {"pending", "running", "generated", *TERMINAL}


class BatchError(RuntimeError):
    """The job could not proceed, with the reason a caller can act on."""


class JobBusyError(BatchError):
    """Another worker owns this job right now."""


class ConversionRefusedError(BatchError):
    """The raster was retained, but it cannot become artwork. Not retryable."""


def _slug(text: str) -> str:
    return re.sub(r"-+", "-", re.sub(r"[^a-z0-9]+", "-", text.lower())).strip("-")[:40] or "item"


def _derive_seed(job_seed: int, key: str) -> int:
    """A seed that belongs to the item, not to its position in a list."""
    raw = hashlib.sha256(f"{job_seed}:{key}".encode()).digest()
    return int.from_bytes(raw[:4], "big") % (SEED_MAX + 1)


@dataclass(frozen=True, slots=True)
class JobItem:
    """One requested image, identified by something stable."""

    key: str
    subject: str
    seed: int
    ordinal: int
    style: str | None = None
    replaces: str | None = None

    def request(self, settings: GenerationSettings) -> GenerationRequest:
        return GenerationRequest(subject=self.subject, seed=self.seed, style=self.style,
                                 settings=settings, seed_requested=True)

    def as_json(self) -> dict:
        return {"key": self.key, "subject": self.subject, "seed": self.seed,
                "ordinal": self.ordinal, "style": self.style, "replaces": self.replaces}

    @classmethod
    def from_json(cls, document: dict) -> JobItem:
        return cls(key=str(document["key"]), subject=str(document["subject"]),
                   seed=int(document["seed"]), ordinal=int(document["ordinal"]),
                   style=document.get("style"), replaces=document.get("replaces"))


def plan_items(subjects: Sequence[str], count: int, *, seed: int,
               style: str | None = None) -> list[JobItem]:
    """Turn a subject pool and a count into stable, content-derived items.

    A key is `<subject-slug>-<nth time this subject appears>`, so inserting or
    reordering another subject leaves every existing item's key — and therefore
    its seed and its request digest — exactly where it was.
    """
    clean = [" ".join(value.split()) for value in subjects if value.strip()]
    if not clean:
        raise RequestError("at least one non-empty subject is required")
    if not 0 < count <= MAX_ITEMS:
        raise RequestError(f"count must be 1-{MAX_ITEMS}, got {count}")
    style = " ".join(style.split()) if style else None
    occurrences: dict[str, int] = {}
    items = []
    for ordinal in range(count):
        subject = clean[ordinal % len(clean)]
        slug = _slug(subject)
        occurrences[slug] = occurrences.get(slug, 0) + 1
        key = f"{slug}-{occurrences[slug]:03d}"
        items.append(JobItem(key=key, subject=subject, seed=_derive_seed(seed, key),
                             ordinal=ordinal, style=style))
    return items


def freeze_request(items: Sequence[JobItem], *, settings: GenerationSettings, seed: int,
                   backend: str, replacement_budget: int = 0, label: str = "") -> dict:
    """The immutable half of a job: what was asked for, and on which weights."""
    if not items:
        raise RequestError("a job needs at least one item")
    if len(items) > MAX_ITEMS:
        raise RequestError(f"a job holds at most {MAX_ITEMS} items")
    keys = [item.key for item in items]
    if len(set(keys)) != len(keys):
        raise RequestError("job item keys must be unique")
    if not 0 <= replacement_budget <= max(8, len(items)):
        raise RequestError("replacement budget must be between 0 and the job size")
    document = {"schema": JOB_SCHEMA, "label": label, "seed": int(seed), "backend": backend,
                "settings": settings.as_json(), "replacement_budget": int(replacement_budget),
                "producer": PRODUCER, "producer_version": PRODUCER_VERSION,
                # Sorted by key, so two callers who listed the same work in a
                # different order freeze the same job and share its identity.
                "items": [item.as_json() for item in sorted(items, key=lambda i: i.key)]}
    document["job_id"] = canonical_digest(document)
    return document


@dataclass
class LedgerEntry:
    key: str
    request_sha256: str
    state: str = "pending"
    attempts: int = 0
    error: str | None = None
    png_sha256: str | None = None
    conversion_sha256: str | None = None
    conversion_key: str | None = None
    artifacts: dict[str, str] = field(default_factory=dict)
    duplicate_of: str | None = None
    replaced_by: str | None = None
    seconds: float | None = None

    def as_json(self) -> dict:
        return {"key": self.key, "request_sha256": self.request_sha256, "state": self.state,
                "attempts": self.attempts, "error": self.error, "png_sha256": self.png_sha256,
                "conversion_sha256": self.conversion_sha256, "conversion_key": self.conversion_key,
                "artifacts": dict(self.artifacts),
                "duplicate_of": self.duplicate_of, "replaced_by": self.replaced_by,
                "seconds": self.seconds}

    @classmethod
    def from_json(cls, document: dict) -> LedgerEntry:
        state = str(document.get("state", "pending"))
        if state not in STATES:
            raise BatchError(f"ledger has an unknown item state {state!r}")
        return cls(key=str(document["key"]), request_sha256=str(document["request_sha256"]),
                   state=state, attempts=int(document.get("attempts", 0)),
                   error=document.get("error"), png_sha256=document.get("png_sha256"),
                   conversion_sha256=document.get("conversion_sha256"),
                   conversion_key=document.get("conversion_key"),
                   artifacts=dict(document.get("artifacts") or {}),
                   duplicate_of=document.get("duplicate_of"),
                   replaced_by=document.get("replaced_by"), seconds=document.get("seconds"))


def _atomic_write_bytes(path: Path, data: bytes) -> None:
    """Publish a file only once it is complete; a reader never sees a half one."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    with open(temporary, "wb") as handle:
        handle.write(data)
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


def _atomic_write_json(path: Path, document: object) -> None:
    _atomic_write_bytes(path, (json.dumps(document, indent=2, sort_keys=True,
                                          allow_nan=False) + "\n").encode())


def _digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


class Job:
    """One worker's exclusive handle on one job directory."""

    def __init__(self, root: Path, request: dict) -> None:
        self._run_attempts: dict[str, int] = {}
        self._conversion_key: str | None = None
        self.root = Path(root)
        self.request = request
        self.settings = GenerationSettings(**request["settings"])
        self.items = {item["key"]: JobItem.from_json(item) for item in request["items"]}
        self.entries: dict[str, LedgerEntry] = {}
        self._lock_handle = None

    # ---- paths -----------------------------------------------------------
    @property
    def job_dir(self) -> Path:
        return self.root / ".job"

    @property
    def raw_dir(self) -> Path:
        return self.job_dir / "raw"

    @property
    def ledger_path(self) -> Path:
        return self.job_dir / "ledger.json"

    @property
    def manifest_path(self) -> Path:
        return self.root / "manifest.json"

    # ---- opening ---------------------------------------------------------
    @classmethod
    def open(cls, root: Path, request: dict) -> Job:
        """Create or resume a job, refusing to resume a different one here."""
        root = Path(root)
        job = cls(root, request)
        job.job_dir.mkdir(parents=True, exist_ok=True)
        job.raw_dir.mkdir(parents=True, exist_ok=True)
        frozen = job.job_dir / "request.json"
        if frozen.is_file():
            existing = json.loads(frozen.read_text())
            if existing.get("job_id") != request.get("job_id"):
                raise BatchError(
                    f"{root} holds job {existing.get('job_id', '?')[:12]} and this is "
                    f"{request.get('job_id', '?')[:12]}; resume the original or choose a new directory")
            job.request = existing
            job.settings = GenerationSettings(**existing["settings"])
            job.items = {item["key"]: JobItem.from_json(item) for item in existing["items"]}
        else:
            if job.manifest_path.exists():
                raise BatchError(f"{root} already holds a completed artwork library")
            _atomic_write_json(frozen, request)
        return job

    def acquire(self) -> Job:
        """One writer. A second gets a refusal, not a corrupted ledger."""
        self.job_dir.mkdir(parents=True, exist_ok=True)
        handle = (self.job_dir / "lock").open("a")
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as exc:
            handle.close()
            raise JobBusyError(f"another worker owns {self.root}; wait for it or use a different job") from exc
        self._lock_handle = handle
        self._load_ledger()
        return self

    def release(self) -> None:
        if self._lock_handle is not None:
            fcntl.flock(self._lock_handle, fcntl.LOCK_UN)
            self._lock_handle.close()
            self._lock_handle = None

    def __enter__(self) -> Job:
        return self.acquire()

    def __exit__(self, *_exc) -> None:
        self.release()

    # ---- ledger ----------------------------------------------------------
    def _load_ledger(self) -> None:
        if self.ledger_path.is_file():
            document = json.loads(self.ledger_path.read_text())
            if document.get("schema") != LEDGER_SCHEMA or document.get("job_id") != self.request["job_id"]:
                raise BatchError("ledger belongs to a different job")
            self.entries = {row["key"]: LedgerEntry.from_json(row) for row in document["items"]}
        for key, item in self.items.items():
            digest = item.request(self.settings).digest
            entry = self.entries.get(key)
            if entry is None:
                self.entries[key] = LedgerEntry(key=key, request_sha256=digest)
            elif entry.request_sha256 != digest:
                # The frozen request cannot change under a job, so this means the
                # ledger was hand-edited or copied. Refuse; do not reconcile.
                raise BatchError(f"ledger item {key} records a different request than the frozen manifest")
        self._reconcile()
        self._save_ledger()

    def _reconcile(self) -> None:
        """Crash recovery: verify retained bytes, reclaim abandoned work.

        Completed work is never discarded, and bytes that no longer hash to what
        the ledger says are never silently accepted: the item goes back to
        pending and its next attempt is a new attempt, counted.
        """
        for entry in self.entries.values():
            if entry.state == "running":
                entry.state = "pending"  # its worker is gone; the attempt still counted
            if entry.state == "failed":
                # Opening the job again is the operator deciding to try again.
                # `refused` is not promoted: identical words on identical weights
                # with an identical seed will draw the identical unusable picture.
                entry.state = "pending"
            if entry.state in {"generated", "accepted"} and entry.png_sha256:
                raw = self.raw_dir / f"{entry.request_sha256}.png"
                if not raw.is_file() or _digest(raw.read_bytes()) != entry.png_sha256:
                    self._invalidate(entry, "retained raster is missing or corrupt")
                    continue
            if entry.state == "accepted":
                for name, expected in entry.artifacts.items():
                    path = self.root / name
                    if not path.is_file() or _digest(path.read_bytes()) != expected:
                        self._invalidate(entry, f"published artifact {name} is missing or corrupt")
                        break

    def _invalidate(self, entry: LedgerEntry, reason: str) -> None:
        for name in entry.artifacts:
            (self.root / name).unlink(missing_ok=True)
        (self.raw_dir / f"{entry.request_sha256}.png").unlink(missing_ok=True)
        (self.raw_dir / f"{entry.request_sha256}.json").unlink(missing_ok=True)
        entry.state = "pending"
        entry.error = reason[:MAX_ERROR_CHARS]
        entry.png_sha256 = None
        entry.conversion_sha256 = None
        entry.artifacts = {}

    def _save_ledger(self) -> None:
        _atomic_write_json(self.ledger_path, {
            "schema": LEDGER_SCHEMA, "job_id": self.request["job_id"],
            "items": [self.entries[key].as_json() for key in sorted(self.entries)]})

    # ---- the work --------------------------------------------------------
    def pending_keys(self) -> list[str]:
        """Work still to do, in frozen-manifest order so progress is legible."""
        return [key for key in sorted(self.items, key=lambda k: (self.items[k].ordinal, k))
                if self.entries[key].state not in TERMINAL]

    def counts(self) -> dict[str, int]:
        tally = dict.fromkeys(sorted(STATES), 0)
        for entry in self.entries.values():
            tally[entry.state] += 1
        return tally

    def accepted_keys(self) -> list[str]:
        return [key for key in sorted(self.items, key=lambda k: (self.items[k].ordinal, k))
                if self.entries[key].state == "accepted"]

    def run(self, *, generate: Callable[[GenerationRequest], tuple[bytes, dict]],
            convert: Callable[[bytes, dict, JobItem], dict],
            conversion_key: str | None = None,
            max_attempts: int = DEFAULT_MAX_ATTEMPTS,
            stop: Callable[[], bool] | None = None,
            on_event: Callable[[str, JobItem, LedgerEntry], None] | None = None) -> dict:
        """Work the pending items until they are done, or until asked to stop.

        `generate` gets a normalized request and returns (png, metadata).
        `convert` gets retained bytes and returns {'artifacts': {name: bytes},
        'record': {...}}, or raises ConversionRefusedError for output that no retry
        will fix. Everything else is retried up to `max_attempts`.

        `conversion_key` identifies the conversion itself — its dimensions, its
        tracer and its options. An item accepted under a different key is
        reconverted from the raster already on disk, which costs nothing: the
        generation cache is keyed by the request and stays valid. Without this,
        rerunning a job at a new artwork size silently kept the old one.
        """
        if self._lock_handle is None:
            raise BatchError("run() needs the job lock; use `with Job.open(...) as job:`")
        cancelled = False
        # The attempt budget bounds this run. `entry.attempts` keeps accumulating
        # across resumes because it is what the report is about, but it must not
        # make a resumed job refuse to start.
        self._run_attempts: dict[str, int] = {}
        self._reconvert(conversion_key)
        while True:
            pending = self.pending_keys()
            if not pending:
                break
            if stop is not None and stop():
                cancelled = True
                break
            key = pending[0]
            item, entry = self.items[key], self.entries[key]
            if self._run_attempts.get(key, 0) >= max_attempts:
                entry.state = "failed"
                entry.error = (entry.error or "attempts exhausted")[:MAX_ERROR_CHARS]
                self._save_ledger()
                self._mint_replacement(item, entry)
                continue
            self._conversion_key = conversion_key
            self._work_one(item, entry, generate=generate, convert=convert, on_event=on_event)
            if entry.state in {"refused", "failed", "duplicate"}:
                self._mint_replacement(item, entry)
        return {**self.status(), "cancelled": cancelled}

    def _reconvert(self, conversion_key: str | None) -> None:
        """Drop published artifacts whose conversion no longer describes them."""
        if conversion_key is None:
            return
        changed = False
        for entry in self.entries.values():
            if entry.state != "accepted" or entry.conversion_key == conversion_key:
                continue
            for name in entry.artifacts:
                (self.root / name).unlink(missing_ok=True)
            entry.artifacts = {}
            entry.conversion_sha256 = None
            # The raster stays: only the tracing has to happen again.
            entry.state = "generated" if entry.png_sha256 else "pending"
            changed = True
        if changed:
            self._save_ledger()

    def _work_one(self, item: JobItem, entry: LedgerEntry, *, generate, convert, on_event) -> None:
        request = item.request(self.settings)
        entry.state = "running"
        entry.attempts += 1
        self._run_attempts[item.key] = self._run_attempts.get(item.key, 0) + 1
        self._save_ledger()
        raw = self.raw_dir / f"{entry.request_sha256}.png"
        try:
            if entry.png_sha256 is None or not raw.is_file():
                # The raster is persisted before anything is traced, so a
                # conversion failure never costs another GPU request.
                started = time.monotonic()
                png, meta = generate(request)
                if not isinstance(png, bytes) or not png:
                    raise BatchError("generator returned no image bytes")
                digest = _digest(png)
                if meta.get("png_sha256") not in (None, digest):
                    raise BatchError("generator reported a digest that differs from its own bytes")
                _atomic_write_bytes(raw, png)
                _atomic_write_json(raw.with_suffix(".json"),
                                   {**meta, "png_sha256": digest, "key": item.key,
                                    "request": request.as_json()})
                entry.png_sha256 = digest
                entry.seconds = round(meta.get("seconds") or (time.monotonic() - started), 3)
                entry.state = "generated"
                entry.error = None
                self._save_ledger()
                self._emit(on_event, "generated", item, entry)
            png = raw.read_bytes()
            meta = json.loads(raw.with_suffix(".json").read_text())
            duplicate = self._first_accepted_with(entry.png_sha256, exclude=item.key)
            if duplicate is not None:
                # Both requests are kept; only one artwork enters the library.
                entry.state, entry.duplicate_of, entry.error = "duplicate", duplicate, None
                self._save_ledger()
                self._emit(on_event, "duplicate", item, entry)
                return
            result = convert(png, meta, item)
            self._publish(item, entry, result, conversion_key=self._conversion_key)
            self._emit(on_event, "accepted", item, entry)
        except ConversionRefusedError as exc:
            entry.state, entry.error = "refused", str(exc)[:MAX_ERROR_CHARS]
            self._save_ledger()
            self._emit(on_event, "refused", item, entry)
        except Exception as exc:  # noqa: BLE001 - one retryable failure, recorded
            entry.error = f"{type(exc).__name__}: {exc}"[:MAX_ERROR_CHARS]
            entry.state = "generated" if entry.png_sha256 else "pending"
            self._save_ledger()
            self._emit(on_event, "error", item, entry)

    def _emit(self, on_event, kind: str, item: JobItem, entry: LedgerEntry) -> None:
        if on_event is not None:
            on_event(kind, item, entry)

    def _first_accepted_with(self, png_sha256: str | None, *, exclude: str) -> str | None:
        if not png_sha256:
            return None
        for key in self.accepted_keys():
            if key != exclude and self.entries[key].png_sha256 == png_sha256:
                return key
        return None

    def _publish(self, item: JobItem, entry: LedgerEntry, result: dict,
                 *, conversion_key: str | None = None) -> None:
        """Write the converted artifacts, then mark the item accepted."""
        artifacts = result.get("artifacts") or {}
        if not artifacts:
            raise BatchError("converter published no artifacts")
        written: dict[str, str] = {}
        try:
            for name, data in artifacts.items():
                if Path(name).name != name or name.startswith("."):
                    raise BatchError(f"artifact name must be a plain basename, got {name!r}")
                payload = data.encode() if isinstance(data, str) else bytes(data)
                _atomic_write_bytes(self.root / name, payload)
                written[name] = _digest(payload)
        except BaseException:
            for name in written:
                (self.root / name).unlink(missing_ok=True)
            raise
        record = result.get("record") or {}
        _atomic_write_json(self.raw_dir / f"{entry.request_sha256}.record.json", record)
        entry.artifacts = written
        entry.conversion_sha256 = canonical_digest(record)
        entry.conversion_key = conversion_key
        entry.state = "accepted"
        entry.error = None
        entry.duplicate_of = None
        self._save_ledger()

    def _mint_replacement(self, item: JobItem, entry: LedgerEntry) -> None:
        """Spend a unit of the retry budget on a fresh candidate for one slot.

        A replacement is a different seed for the same intent — not an extra
        dataset item. It inherits the original's ordinal so the published
        library keeps the order the job was asked for.
        """
        budget = int(self.request.get("replacement_budget", 0))
        used = sum(1 for value in self.items.values() if value.replaces)
        if used >= budget:
            return
        origin = item.replaces or item.key
        key = f"{origin}#r{used + 1}"
        if key in self.items:
            return
        seed = _derive_seed(int(self.request["seed"]), key)
        fresh = JobItem(key=key, subject=item.subject, seed=seed, ordinal=item.ordinal,
                        style=item.style, replaces=origin)
        self.items[key] = fresh
        self.entries[key] = LedgerEntry(key=key, request_sha256=fresh.request(self.settings).digest)
        entry.replaced_by = key
        self._save_ledger()

    # ---- reporting and finalization --------------------------------------
    def status(self) -> dict:
        counts = self.counts()
        return {"schema": "tatbot.inkgen-job-status/1", "job_id": self.request["job_id"],
                "root": str(self.root), "requested": len(self.request["items"]),
                "candidates": len(self.items), "counts": counts,
                "accepted": counts["accepted"], "complete": self.is_complete(),
                "attempts": sum(entry.attempts for entry in self.entries.values()),
                "generation_seconds": round(sum(entry.seconds or 0 for entry in self.entries.values()), 3)}

    def is_complete(self) -> bool:
        """Every originally requested slot ended with one accepted artwork."""
        accepted = {self.items[key].replaces or key for key in self.accepted_keys()}
        return accepted >= {item["key"] for item in self.request["items"]}

    def library(self) -> list[dict]:
        """The accepted records, in the order the job asked for them."""
        rows = []
        for key in self.accepted_keys():
            item, entry = self.items[key], self.entries[key]
            record = json.loads((self.raw_dir / f"{entry.request_sha256}.record.json").read_text())
            rows.append({"key": key, "ordinal": item.ordinal, **record})
        return rows

    def report(self) -> dict:
        """Requested, accepted, refused, failed, duplicate — never one number."""
        counts = self.counts()
        return {"requested": len(self.request["items"]), "candidates": len(self.items),
                "accepted": counts["accepted"], "refused": counts["refused"],
                "failed": counts["failed"], "duplicate": counts["duplicate"],
                "pending": counts["pending"] + counts["running"] + counts["generated"],
                "attempts": sum(entry.attempts for entry in self.entries.values()),
                "generation_seconds": round(sum(entry.seconds or 0 for entry in self.entries.values()), 3)}

    def finalize_selection(self, *, reason: str) -> dict:
        """Publish an explicitly chosen subset — never as a completion manifest."""
        document = {"schema": SELECTION_SCHEMA, "job_id": self.request["job_id"],
                    "reason": reason, "complete": self.is_complete(), **self.report(),
                    "accepted_keys": self.accepted_keys()}
        _atomic_write_json(self.root / "selection.json", document)
        return document


def load_job(root: Path) -> dict:
    """The frozen request of an existing job, for resume and for reporting."""
    path = Path(root) / ".job" / "request.json"
    if not path.is_file():
        raise BatchError(f"{root} is not a generation job directory")
    document = json.loads(path.read_text())
    if document.get("schema") != JOB_SCHEMA:
        raise BatchError(f"{path} is not a {JOB_SCHEMA}")
    return document


def shard(items: Iterable[JobItem], *, index: int, of: int) -> list[JobItem]:
    """Split work across workers without changing any item's identity."""
    if not 0 <= index < of or of < 1:
        raise RequestError(f"shard {index} of {of} is not a valid split")
    ordered = sorted(items, key=lambda item: item.key)
    return [item for position, item in enumerate(ordered) if position % of == index]


def replan(request: dict, *, settings: GenerationSettings) -> dict:
    """The same job on different weights is a different job, and says so."""
    items = [JobItem.from_json(row) for row in request["items"]]
    return freeze_request(items, settings=settings, seed=int(request["seed"]),
                          backend=str(request["backend"]),
                          replacement_budget=int(request.get("replacement_budget", 0)),
                          label=str(request.get("label", "")))


__all__ = ["JOB_SCHEMA", "SELECTION_SCHEMA", "BatchError", "ConversionRefusedError", "Job", "JobBusyError",
           "JobItem", "LedgerEntry", "canonical_json", "freeze_request", "load_job", "plan_items",
           "replan", "shard"]


# ---- standalone entry point -------------------------------------------------
# `python batch.py ...` from this directory, with no Tatbot checkout, no fleet
# configuration and no node map. It generates and retains rasters; turning them
# into artwork needs Inkmap's tracer, which is a separate, larger dependency.
def _png_converter(png: bytes, meta: dict, item: JobItem) -> dict:
    stem = f"{item.ordinal:04d}-{item.key}"
    return {"artifacts": {f"{stem}.png": png,
                          f"{stem}.json": json.dumps({**meta, "key": item.key}, indent=2,
                                                     sort_keys=True) + "\n"},
            "record": {"key": item.key, "png": f"{stem}.png", "png_sha256": meta["png_sha256"]}}


def _local_generator():
    from engine import Engine

    engine = Engine()
    engine.pin(required=False)
    return engine


def _http_generator(url: str, timeout_s: float):
    import urllib.request

    def generate(request: GenerationRequest) -> tuple[bytes, dict]:
        import base64

        body = json.dumps({"subject": request.subject, "seed": request.seed,
                           **({"style": request.style} if request.style else {})}).encode()
        post = urllib.request.Request(url.rstrip("/") + "/api/generate", data=body,
                                      headers={"Content-Type": "application/json"}, method="POST")
        with urllib.request.urlopen(post, timeout=timeout_s) as response:  # noqa: S310
            reply = json.loads(response.read(64 << 20))
        if reply.get("error"):
            raise BatchError(str(reply["error"]))
        return base64.b64decode(reply["png_base64"], validate=True), {
            "seed": int(reply.get("seed", request.seed)), "prompt": reply.get("prompt"),
            "model": reply.get("model"), "model_revision": reply.get("model_revision"),
            "seconds": reply.get("seconds")}
    return generate


def main(argv: Sequence[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(prog="inkgen-batch", description=__doc__.split("\n")[0])
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("plan", "run"):
        p = sub.add_parser(name, help="freeze a job manifest" if name == "plan" else "run or resume a job")
        p.add_argument("root", type=Path)
        p.add_argument("--subject", action="append", default=[], required=True)
        p.add_argument("--count", type=int, required=True)
        p.add_argument("--seed", type=int, default=0)
        p.add_argument("--style")
        p.add_argument("--replacement-budget", type=int, default=0)
        if name == "run":
            p.add_argument("--api", help="an HTTP generator; omit to load the model in this process")
            p.add_argument("--timeout-s", type=float, default=300.0)
            p.add_argument("--max-attempts", type=int, default=DEFAULT_MAX_ATTEMPTS)
    status = sub.add_parser("status", help="read an existing job's ledger; contacts nothing")
    status.add_argument("root", type=Path)
    args = parser.parse_args(argv)

    if args.command == "status":
        request = load_job(args.root)
        with Job.open(args.root, request) as job:
            print(json.dumps(job.report(), indent=2, sort_keys=True))
        return 0

    settings = GenerationSettings.from_env()
    items = plan_items(args.subject, args.count, seed=args.seed, style=args.style)
    request = freeze_request(items, settings=settings, seed=args.seed,
                             backend="local" if getattr(args, "api", None) is None else "endpoint",
                             replacement_budget=args.replacement_budget)
    if args.command == "plan":
        Job.open(args.root, request)
        print(json.dumps({"job_id": request["job_id"], "root": str(args.root),
                          "items": len(request["items"])}, indent=2))
        return 0

    if args.api:
        generate = _http_generator(args.api, args.timeout_s)
    else:
        engine = _local_generator()
        request = replan(request, settings=engine.settings)
        generate = engine.generate
    with Job.open(args.root, request) as job:
        status_document = job.run(generate=generate, convert=_png_converter,
                                  max_attempts=args.max_attempts)
        print(json.dumps({**status_document, "report": job.report()}, indent=2, sort_keys=True))
        return 0 if job.is_complete() else 1


if __name__ == "__main__":
    raise SystemExit(main())
