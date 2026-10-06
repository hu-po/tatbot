"""Talk to the design generator, and start one when we are allowed to.

The generator is meant to come and go: it holds ~13 GB of VRAM on a node that
also runs simulation, and it stops itself when idle (`web/inkgen/idle.py`). So
anything that needs it probes health, starts one through the CLI's own
idempotent `tatbot inkgen ctl -- start` when the target is the fleet generator,
and refuses rather than starting anything when the GPU is already spoken for.

Which generator answers is an explicit choice, not a search. Four kinds:

    fleet     the managed worker on the node holding the `inkgen` role — the
              only one this process may start
    endpoint  an address the caller named; probed, never started, never woken
    space     the public Hugging Face Space; ours to ask, not ours to run
    (none)    no role configured and nothing named

A person exploring interactively may fall through to the Space, because that is
what the public app does anyway. Bulk work may not: `resolve_backend(...,
require_configured=True)` refuses instead, so a missing line in nodes.json can
never quietly point a thousand-image job at a shared public GPU.
"""
from __future__ import annotations

import base64
import hashlib
import json
import os
import shutil
import subprocess
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any

from tatbot_sim.inkmap.design_build import MAX_IMAGE_BYTES, DesignBuildError
from tatbot_sim.repo import repo_root

DEFAULT_PORT = 8600
SPACE_URL = "https://hu-po-inkgen.hf.space"
# The model is bf16 on a 6B transformer plus its text encoder; below this a
# start would either fail or evict whatever else holds the card.
VRAM_MIN_MB = int(os.environ.get("INKGEN_VRAM_MIN_MB", "14000"))
HEALTH_TIMEOUT_S = 5.0
START_TIMEOUT_S = 900.0


class InkgenUnreachableError(DesignBuildError):
    """No generator answered, and none could be started."""


class InkgenBusyError(DesignBuildError):
    """The GPU is held by something else; starting would evict it."""


class InkgenBackendError(DesignBuildError):
    """No generator was configured, and this caller does not accept a fallback."""


@dataclass(frozen=True)
class Backend:
    """Which generator, and whether this process may start it."""

    kind: str  # fleet | endpoint | space
    url: str

    @property
    def managed(self) -> bool:
        """Only the fleet worker is ours to start; nothing else is."""
        return self.kind == "fleet"

    def describe(self) -> str:
        return f"{self.kind} generator at {self.url}"


def fleet_endpoint(node_config: dict[str, Any] | None = None) -> str | None:
    """Where the fleet generator answers, or None when no node holds the role."""

    from tatbot_cli import nodes as node_table

    nmap = node_config or node_table.load(repo_root())
    here = os.uname().nodename.split(".")[0]
    for node in node_table.nodes_with(nmap, "inkgen"):
        if node == here or node_table.host_of(nmap, node) in {here, "127.0.0.1"}:
            return f"http://127.0.0.1:{DEFAULT_PORT}"
        host = node_table.host_of(nmap, node)
        if host:
            return f"http://{host}:{DEFAULT_PORT}"
    return None


def resolve_backend(api_url: str | None = None, *, space: bool = False,
                    require_configured: bool = False,
                    node_config: dict[str, Any] | None = None) -> Backend:
    """Name the generator this call will use, before anything is contacted."""
    if space:
        if api_url and api_url.rstrip("/") != SPACE_URL:
            raise InkgenBackendError("choose either the hosted Space or an explicit --api-url, not both")
        return Backend("space", SPACE_URL)
    fleet = fleet_endpoint(node_config)
    if api_url:
        url = api_url.rstrip("/")
        if not url.startswith(("http://", "https://")):
            raise InkgenBackendError("generator URL must start with http:// or https://")
        if url == SPACE_URL:
            return Backend("space", SPACE_URL)
        # An address that happens to be the fleet worker's is the fleet worker;
        # any other address is somebody else's server and is never started here.
        return Backend("fleet" if fleet and url == fleet.rstrip("/") else "endpoint", url)
    if fleet:
        return Backend("fleet", fleet)
    if require_configured:
        raise InkgenBackendError(
            "no generator is configured: no node carries the `inkgen` role in config/nodes.json. "
            "Name one with --api-url, or state --space to use the public generator on purpose. "
            "Bulk work never falls back to the public Space by itself.")
    return Backend("space", SPACE_URL)


def fleet_url(node_config: dict[str, Any] | None = None) -> str:
    """Compatible spelling of the interactive default: role first, Space last."""
    return resolve_backend(node_config=node_config).url


def health(api_url: str, *, timeout_s: float = HEALTH_TIMEOUT_S) -> dict[str, Any] | None:
    """The generator's health document, or None when nothing answers."""
    try:
        with urllib.request.urlopen(api_url.rstrip("/") + "/api/health", timeout=timeout_s) as response:  # noqa: S310
            return json.loads(response.read(1 << 20))
    except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError, ValueError, OSError):
        return None


def free_vram_mb() -> int | None:
    """Free memory on the least-loaded visible GPU, or None without nvidia-smi."""
    if not shutil.which("nvidia-smi"):
        return None
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits"],
                             capture_output=True, text=True, timeout=20, check=True).stdout
    except (OSError, subprocess.SubprocessError):
        return None
    values = [int(line.strip()) for line in out.splitlines() if line.strip().isdigit()]
    return max(values) if values else None


def gpu_holders(limit: int = 5) -> list[str]:
    """What currently holds the GPU, so a refusal can name it.

    Program name, pid and memory only. A process's full argv can run to
    kilobytes (a browser's GPU process does), and a refusal a person has to
    read must stay one line per holder.
    """
    if not shutil.which("nvidia-smi"):
        return []
    try:
        out = subprocess.run(["nvidia-smi", "--query-compute-apps=pid,process_name,used_memory",
                              "--format=csv,noheader"], capture_output=True, text=True, timeout=20,
                             check=True).stdout
    except (OSError, subprocess.SubprocessError):
        return []
    holders = []
    for line in out.splitlines():
        fields = [field.strip() for field in line.split(",")]
        if len(fields) < 3:
            continue
        pid, name, used = fields[0], fields[1], fields[-1]
        program = os.path.basename(name.split()[0]) if name.split() else name
        megabytes = int("".join(c for c in used if c.isdigit()) or 0)
        holders.append((megabytes, f"{program} (pid {pid}, {used})"))
    return [text for _, text in sorted(holders, key=lambda item: -item[0])[:limit]]


def start(*, timeout_s: float = START_TIMEOUT_S, check_vram: bool = True) -> None:
    """Start the fleet generator through the CLI's own idempotent verb.

    `tatbot inkgen ctl -- start` hops to the node with the role, is idempotent,
    and waits for `/api/health` itself, so there is one start path, not two.
    """
    if check_vram:
        free = free_vram_mb()
        if free is not None and free < VRAM_MIN_MB:
            holders = gpu_holders()
            raise InkgenBusyError(
                f"only {free} MB of GPU memory free, the generator needs {VRAM_MIN_MB} MB"
                + (f"; held by: {'; '.join(holders)}" if holders else "")
                + " — stop that work or use --api-url with the hosted generator")
    argv = [str(repo_root() / "scripts/tatbot"), "inkgen", "ctl", "--", "start"]
    try:
        completed = subprocess.run(argv, timeout=timeout_s, check=False)
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise InkgenUnreachableError(f"could not start a generator: {exc}") from exc
    if completed.returncode:
        raise InkgenUnreachableError(
            f"`tatbot inkgen ctl -- start` exited {completed.returncode}; "
            "see `tatbot inkgen ctl -- logs`")


def ensure_backend(backend: Backend, *, autostart: bool = True,
                   timeout_s: float = START_TIMEOUT_S) -> tuple[Backend, dict[str, Any]]:
    """Return the named backend once it answers, starting it only if it is ours.

    A backend that is not the fleet worker is never started, never woken and
    never replaced by another one: a caller who named an address and got no
    answer is told that, not handed a different generator's output.
    """
    document = health(backend.url)
    if document is not None:
        return backend, document
    if not autostart:
        raise InkgenUnreachableError(f"no generator answering at {backend.url} (--no-autostart)")
    if not backend.managed:
        raise InkgenUnreachableError(
            f"the {backend.describe()} did not answer; it is not ours to start. "
            "Start it yourself, or select the fleet generator.")
    start(timeout_s=timeout_s)
    document = health(backend.url, timeout_s=30)
    if document is None:
        raise InkgenUnreachableError(f"started a generator but {backend.url}/api/health still does not answer")
    return backend, document


def ensure_running(api_url: str | None = None, *, autostart: bool = True,
                   timeout_s: float = START_TIMEOUT_S,
                   require_configured: bool = False) -> tuple[str, dict[str, Any]]:
    """Compatible spelling of `ensure_backend`, returning the answering URL."""
    backend, document = ensure_backend(resolve_backend(api_url, require_configured=require_configured),
                                       autostart=autostart, timeout_s=timeout_s)
    return backend.url, document


def generate(subject: str, *, seed: int | None = None, style: str | None = None,
             api_url: str | None = None, timeout_s: float = 180.0, autostart: bool = True,
             space: bool = False) -> dict[str, Any]:
    """One image. The reply carries the prompt, seed and model that produced it."""
    subject = " ".join(subject.split())
    if not 1 <= len(subject) <= 120:
        raise DesignBuildError("subject must be 1-120 characters")
    backend, _ = ensure_backend(resolve_backend(api_url, space=space), autostart=autostart)
    url = backend.url
    payload: dict[str, Any] = {"subject": subject}
    if seed is not None:
        # Seed zero is a seed. `is not None` is the whole point of this line.
        payload["seed"] = int(seed)
    if style:
        payload["style"] = " ".join(style.split())
    request = urllib.request.Request(url.rstrip("/") + "/api/generate",
                                     data=json.dumps(payload, separators=(",", ":")).encode(),
                                     headers={"Content-Type": "application/json"}, method="POST")
    try:
        with urllib.request.urlopen(request, timeout=timeout_s) as response:  # noqa: S310
            body = json.loads(response.read(MAX_IMAGE_BYTES * 2))
    except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError) as exc:
        raise InkgenUnreachableError(f"generator request failed: {exc}") from exc
    except (ValueError, UnicodeDecodeError) as exc:
        raise DesignBuildError("generator returned invalid JSON") from exc
    if not isinstance(body, dict) or body.get("error"):
        raise DesignBuildError(f"generator refused the request: {body.get('error', body)!r}")
    try:
        png = base64.b64decode(body["png_base64"], validate=True)
    except (KeyError, TypeError, ValueError) as exc:
        raise DesignBuildError("generator reply has no usable png_base64") from exc
    reported = body.get("seed")
    return {"png": png, "png_sha256": hashlib.sha256(png).hexdigest(),
            "seed": int(reported if reported is not None else (seed if seed is not None else 0)),
            "prompt": str(body.get("prompt") or subject),
            "model": str(body.get("model") or "inkgen"),
            "model_revision": body.get("model_revision"), "settings": body.get("settings"),
            "seconds": body.get("seconds"), "api_url": url, "backend": backend.kind}
