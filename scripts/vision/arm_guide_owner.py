"""The conductor side of one `tatbot-arm-guide` owner: its JSON-line protocol and native build check.

`wrist_pose.py` (`tatbot calib pose`) and `joint_payload_measure.py`
(`tatbot calib joint-measure`) each start one owner beside them and drive it
through this client. The wire format is `rust/tatbot-arm/src/guide.rs`; the
owner's own lease, E-stop monitor, telemetry and terminal idle release are
unchanged by anything here.
"""
from __future__ import annotations

import contextlib
import json
import os
import select
import subprocess
import threading
import time
from pathlib import Path

EVENT_TIMEOUT_S = 40.0


class CaptureError(RuntimeError):
    """Refused or aborted; the retained run says why."""


class OwnerFaultError(CaptureError):
    """The arm owner latched a fault: the arm holds where it can; recovery may follow."""

    def __init__(self, event: dict):
        super().__init__(f"arm owner fault: {event.get('reason')}")
        self.reason = str(event.get("reason"))
        self.owner_event = event


class Owner:
    """One `tatbot-arm-guide` process."""

    def __init__(self, argv, log_path):
        self.send_lock = threading.Lock()
        self.read_buffer = b""
        self.log = Path(log_path).open("a")  # noqa: SIM115 -- close() owns it with the process
        # Its own session and stderr file: a holding owner must outlive this
        # conductor and its terminal without keeping their pipes open.
        self.stderr = Path(str(log_path) + ".stderr").open("a")  # noqa: SIM115 -- closed with the log
        self.process = subprocess.Popen(argv, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=self.stderr,
                                        text=True, bufsize=1, start_new_session=True)

    def send(self, command: dict) -> None:
        with self.send_lock:
            self._send(command)

    def _send(self, command: dict) -> None:
        line = json.dumps(command)
        self.log.write("> " + line + "\n")
        self.log.flush()
        if self.process.poll() is not None:
            raise CaptureError("arm owner exited before the command could be sent")
        self.process.stdin.write(line + "\n")
        self.process.stdin.flush()

    def poll(self, timeout: float) -> dict | None:
        """The next event within `timeout` seconds, or None."""
        deadline = time.monotonic() + timeout
        while b"\n" not in self.read_buffer:
            ready, _, _ = select.select([self.process.stdout], [], [], max(0, deadline - time.monotonic()))
            if not ready:
                return None
            chunk = os.read(self.process.stdout.fileno(), 65536)
            if not chunk:
                raise CaptureError("arm owner closed its event stream")
            self.read_buffer += chunk
        raw, self.read_buffer = self.read_buffer.split(b"\n", 1)
        line = raw.decode("utf-8", errors="replace")
        self.log.write(line + "\n")
        self.log.flush()
        if not line.startswith("{"):
            return None  # retain SDK diagnostics without treating them as events
        try:
            event = json.loads(line)
        except ValueError as error:
            raise CaptureError(f"malformed arm owner event: {line[:160]}") from error
        if not isinstance(event, dict) or not isinstance(event.get("event"), str):
            return None
        return event

    def drain(self) -> None:
        """Consume queued events; a fault already reported refuses the next command."""
        while (event := self.poll(0)) is not None:
            if event.get("event") == "fault":
                raise OwnerFaultError(event)
            if event.get("event") in ("refused", "conductor_lost", "recorder_failed"):
                raise CaptureError(f"arm owner {event['event']}: {event.get('reason')}")

    def wait(self, name: str, timeout: float = EVENT_TIMEOUT_S) -> dict:
        """The next `name` event; a fault or refusal before it is an error."""
        deadline = time.monotonic() + timeout
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise CaptureError(f"arm owner did not report {name} within {timeout:.0f} s; state unknown")
            event = self.poll(min(0.2, remaining))
            if event is None:
                continue
            kind = event.get("event")
            if kind == name:
                return event
            if kind == "fault":
                raise OwnerFaultError(event)
            if kind in ("refused", "release_refused", "conductor_lost", "recorder_failed"):
                raise CaptureError(f"arm owner {kind}: {event.get('reason')}")

    def close(self) -> int | None:
        """EOF requests terminal idle release; drain events until the owner exits."""
        if self.process.stdin and not self.process.stdin.closed:
            self.process.stdin.close()
        deadline = time.monotonic() + 60
        while time.monotonic() < deadline:
            try:
                self.poll(0.1)
            except CaptureError:
                if self.process.poll() is not None:
                    break
                time.sleep(0.01)
            if self.process.poll() is not None and not self.read_buffer:
                break
        with contextlib.suppress(subprocess.TimeoutExpired):
            self.process.wait(timeout=1)
        self.log.close()
        self.stderr.close()
        return self.process.returncode


def validate_native_build(receipt: dict, expected: dict, arch: str | None) -> None:
    """The owner binary was built here, for this machine, from exactly these native sources."""
    build = receipt.get("build") or {}
    if build.get("schema") != "tatbot.arm-guide-build/1" or build.get("native_backend_compiled") is not True:
        raise CaptureError("arm owner binary lacks verified native build metadata")
    if build.get("sources") != expected:
        raise CaptureError("arm owner binary was built from different native sources; rebuild before capture")
    if not arch or not build.get("target", "").startswith(arch + "-"):
        raise CaptureError("arm owner binary target does not match its configured architecture")
    if build.get("profile") != "release":
        raise CaptureError("arm owner needs the release native build for its control loop")
    sdk = build.get("sdk") or {}
    if (sdk.get("version") != "1.8.5" or sdk.get("revision") != "fdfd9f68f57b3bd05c4e85011fa5c11296525b2f"
            or len(sdk.get("library_sha256", "")) != 64 or len(receipt.get("binary_sha256", "")) != 64):
        raise CaptureError("arm owner binary lacks the pinned SDK/library identity")
