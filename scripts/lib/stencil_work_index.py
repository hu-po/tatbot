"""Issue a new stencil mark into the arm owner's durable original-work index.

An issue receipt proves that this software minted the printable mark after the
index epoch began. It does not prove a print was placed or authorize motion;
the native placement path must later co-observe the exact mark and enroll it.
"""

from __future__ import annotations

import fcntl
import hashlib
import json
import os
import secrets
import stat
from pathlib import Path

import stencil_instance
import tatbot_paths

SCHEMA = "tatbot.original-work-index/1"
ISSUE_SCHEMA = "tatbot.original-work-issue/1"
STATE = "state.json"
PENDING = "pending.json"


class IndexBusyError(ValueError):
    """Another live original-work writer owns the stable lock inode."""


class IndexEvidenceError(ValueError):
    """The durable index cannot establish an issue cutover safely."""


def validate_issue_bytes(reference: dict, reference_bytes: bytes, raw: bytes) -> dict:
    """Check the portable receipt against its exact printed reference bytes."""
    if len(raw) > 8192:
        raise ValueError("print issue receipt exceeds size limit")
    value = json.loads(raw)
    fields = {"schema", "index_epoch", "physical_instance_id", "pattern_id",
              "reference_id", "reference_sha256", "image_sha256", "svg_sha256",
              "artwork_sha256", "instance_mark"}
    def hex_digest(text, length):
        return (isinstance(text, str) and len(text) == length
                and all(char in "0123456789abcdef" for char in text))
    if (not isinstance(value, dict) or set(value) != fields
            or value["schema"] != ISSUE_SCHEMA
            or not hex_digest(value["index_epoch"], 32)
            or not stencil_instance.valid_id(value["physical_instance_id"])
            or any(not hex_digest(value[field], 64)
                   for field in ("reference_id", "reference_sha256", "image_sha256",
                                 "svg_sha256", "artwork_sha256"))
            or value["physical_instance_id"] != reference.get("physical_instance_id")
            or value["pattern_id"] != reference.get("pattern_id")
            or value["reference_id"] != reference.get("reference_id")
            or value["reference_sha256"] != hashlib.sha256(reference_bytes).hexdigest()
            or value["image_sha256"] != reference["image"]["sha256"]
            or value["artwork_sha256"] != reference["generator"]["svg_sha256"]
            or value["instance_mark"] != reference.get("instance_mark")):
        raise ValueError("print issue receipt differs from its exact mark and artwork")
    return value


def default_root() -> Path:
    return tatbot_paths.state_root() / "session" / "original-work"


def _sync_dir(path: Path) -> None:
    fd = os.open(path, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def _private(path: Path, *, directory: bool) -> None:
    info = path.lstat()
    wanted = stat.S_ISDIR if directory else stat.S_ISREG
    if not wanted(info.st_mode) or info.st_uid != os.getuid() or info.st_mode & 0o077:
        raise IndexEvidenceError(f"original-work index needs an owned private {'directory' if directory else 'file'}: {path}")
    if not directory and info.st_nlink != 1:
        raise IndexEvidenceError(f"original-work index has a linked file: {path}")


def _create_private(path: Path) -> None:
    path.mkdir(parents=True, mode=0o700)
    path.chmod(0o700)


class IssueIndex:
    """One exclusive writer; the lock inode remains stable across snapshots."""

    def __init__(self, root: Path):
        self.root = Path(root)
        self.lock_fd: int | None = None
        self.epoch: str | None = None

    def _prepare_root(self) -> bool:
        if not self.root.is_absolute():
            raise IndexEvidenceError("original-work index root must be absolute")
        created_root = False
        if not (self.root.is_symlink() or self.root.exists()):
            try:
                self.root.mkdir(parents=True, mode=0o700)
                created_root = True
            except FileExistsError:
                # Another issuer may have created it while this process was
                # waiting. The state decision belongs under the lock below.
                pass
        _private(self.root, directory=True)
        return created_root

    def _read_snapshot(self, created_root: bool) -> dict:
        issued = self.root / "issued"
        state = self.root / STATE
        initializing = created_root and not (state.is_symlink() or state.exists())
        if initializing:
            _create_private(issued)
        _private(issued, directory=True)
        if any(child.name == PENDING or child.name.startswith("state.json.tmp-")
               for child in self.root.iterdir()):
            raise ValueError("interrupted original-work index update requires reconciliation")
        if any(not child.name.endswith(".json") or not stencil_instance.valid_id(child.stem)
               for child in issued.iterdir()):
            raise ValueError("interrupted print issue requires reconciliation")
        if not initializing:
            _private(state, directory=False)
            fd = os.open(state, os.O_RDONLY | os.O_NOFOLLOW | os.O_CLOEXEC)
            with os.fdopen(fd, "rb") as file:
                raw = file.read(64 * 1024 * 1024 + 1)
            if len(raw) > 64 * 1024 * 1024:
                raise ValueError("original-work index state exceeds issue-reader limit")
            return json.loads(raw)
        snapshot = {"schema": SCHEMA, "epoch": secrets.token_hex(16),
                    "seq": 0, "enrollments": {}, "work": {}}
        self._write_initial(snapshot)
        return snapshot

    def __enter__(self):
        created_root = self._prepare_root()
        flags = os.O_RDWR | os.O_CREAT | os.O_NOFOLLOW | os.O_CLOEXEC
        self.lock_fd = os.open(self.root / ".lock", flags, 0o600)
        try:
            _private(self.root / ".lock", directory=False)
            try:
                fcntl.flock(self.lock_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as error:
                raise IndexBusyError("original-work index is busy with an active session") from error
            snapshot = self._read_snapshot(created_root)
            if (not isinstance(snapshot, dict)
                    or set(snapshot) != {"schema", "epoch", "seq", "enrollments", "work"}
                    or snapshot["schema"] != SCHEMA
                    or not isinstance(snapshot["epoch"], str)
                    or len(snapshot["epoch"]) != 32
                    or any(char not in "0123456789abcdef" for char in snapshot["epoch"])
                    or not isinstance(snapshot["seq"], int) or snapshot["seq"] < 0
                    or not isinstance(snapshot["enrollments"], dict)
                    or not isinstance(snapshot["work"], dict)):
                raise ValueError("invalid original-work index state for print issuance")
            self.epoch = snapshot["epoch"]
            return self
        except IndexBusyError:
            self.__exit__(None, None, None)
            raise
        except (ValueError, KeyError, OSError) as error:
            self.__exit__(None, None, None)
            raise IndexEvidenceError(str(error)) from error
        except BaseException:
            self.__exit__(None, None, None)
            raise

    def __exit__(self, _kind, _value, _traceback):
        if self.lock_fd is not None:
            os.close(self.lock_fd)
            self.lock_fd = None

    def _write_initial(self, snapshot: dict) -> None:
        raw = (json.dumps(snapshot, sort_keys=True, separators=(",", ":")) + "\n").encode()
        marker = self.root / PENDING
        fd = os.open(marker, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
        with os.fdopen(fd, "wb") as file:
            file.write(json.dumps({"schema": SCHEMA, "next_seq": 0,
                                   "next_sha256": hashlib.sha256(raw).hexdigest()}).encode())
            file.flush()
            os.fsync(file.fileno())
        _sync_dir(self.root)
        temp = self.root / f"state.json.tmp-{os.getpid()}-{secrets.token_hex(8)}"
        fd = os.open(temp, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
        with os.fdopen(fd, "wb") as file:
            file.write(raw)
            file.flush()
            os.fsync(file.fileno())
        os.replace(temp, self.root / STATE)
        _sync_dir(self.root)
        marker.unlink()
        _sync_dir(self.root)

    def issue(self, settings: dict, reference: dict, reference_bytes: bytes) -> Path:
        """Retain one newly generated image/mark/artwork before returning it."""
        if self.lock_fd is None or self.epoch is None:
            raise RuntimeError("print issue index is not locked")
        instance_id = reference["physical_instance_id"]
        if not stencil_instance.valid_id(instance_id) or settings["physical_instance_id"] != instance_id:
            raise IndexEvidenceError("print issue has an invalid or changed mark")
        if (reference["physical_instance_encoded"] is not True
                or reference["instance_mark"] != settings["instance_mark"]
                or reference["image"]["sha256"] != settings["files"]["stencil.png"]
                or reference["generator"]["svg_sha256"] != settings["artwork_svg_sha256"]):
            raise IndexEvidenceError("print issue does not bind the rendered mark and artwork")
        receipt = {"schema": ISSUE_SCHEMA, "index_epoch": self.epoch,
                   "physical_instance_id": instance_id,
                   "pattern_id": reference["pattern_id"],
                   "reference_id": reference["reference_id"],
                   "reference_sha256": hashlib.sha256(reference_bytes).hexdigest(),
                   "image_sha256": settings["files"]["stencil.png"],
                   "svg_sha256": settings["files"]["stencil.svg"],
                   "artwork_sha256": settings["artwork_svg_sha256"],
                   "instance_mark": settings["instance_mark"]}
        raw = (json.dumps(receipt, sort_keys=True, separators=(",", ":")) + "\n").encode()
        try:
            validate_issue_bytes(reference, reference_bytes, raw)
        except ValueError as error:
            raise IndexEvidenceError(str(error)) from error
        issued = self.root / "issued"
        temp = issued / f".tmp-{os.getpid()}-{secrets.token_hex(8)}"
        final = issued / f"{instance_id}.json"
        if final.exists():
            raise IndexEvidenceError("print instance was already issued")
        fd = os.open(temp, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
        with os.fdopen(fd, "wb") as file:
            file.write(raw)
            file.flush()
            os.fsync(file.fileno())
        _sync_dir(issued)
        os.link(temp, final, follow_symlinks=False)
        _sync_dir(issued)
        temp.unlink()
        _sync_dir(issued)
        return final
