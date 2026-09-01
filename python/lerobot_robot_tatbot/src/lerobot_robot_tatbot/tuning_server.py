"""In-process tuning server: HTTP + SSE cockpit for teleop/inference tuning.

Runs as a daemon thread inside whatever process owns the arms (lerobot-record,
lerobot rollout, or the standalone `python -m lerobot_robot_tatbot.tune`
session). One TrossenArmDriver owns each arm's UDP session, so this is the
only place a tuning UI can live — a separate daemon could never touch the
driver. The server itself never calls the driver: parameter changes are
enqueued into TuningShared and drained by the control loop (see params.py),
and telemetry is read from the loop-published snapshot.

Endpoints
  GET  /              cockpit page (static, rendered from the registry)
  GET  /api/registry  parameter definitions + current/golden/pending values
  GET  /api/stream    SSE: telemetry + pending/dirty state at ~10 Hz
  POST /api/param     {"name": ..., "value": ...} or {"name", "joint", "value"}
  POST /api/save      write live values to the golden YAMLs
  POST /api/revert    queue every dirty parameter back to its golden value
  POST /api/capture_pose  staged_positions := follower's current pose
  POST /api/recover   staged-pose recovery (standalone tune sessions only)
"""

from __future__ import annotations

import contextlib
import json
import logging
import socket
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from importlib import resources
from pathlib import Path

from lerobot_robot_tatbot import goldens
from lerobot_robot_tatbot.params import SESSION, TuningShared

logger = logging.getLogger(__name__)

_singleton_lock = threading.Lock()
_singleton: "TuningServer | None" = None


def get_shared_server(port: int) -> "TuningServer":
    """Process-wide server: leader and follower plugins connect in either
    order and register with the same instance."""
    global _singleton
    with _singleton_lock:
        if _singleton is None:
            _singleton = TuningServer(port)
        return _singleton


class TuningServer:
    def __init__(self, port: int):
        self.port = port
        self.shared = TuningShared()
        self.cfg_dir = goldens.config_dir()
        self._rigs: dict[str, dict] = {}  # arm -> {config, robot, driver}
        self._httpd: ThreadingHTTPServer | None = None
        self._lock = threading.Lock()
        self.standalone = False  # set by tune.py; gates /api/recover

    # ------------------------------------------------------------------
    # Rig registration (called from plugin configure(), loop thread)
    # ------------------------------------------------------------------

    def register(self, arm: str, params: list, config, robot=None) -> TuningShared:
        with self._lock:
            self._rigs[arm] = {"config": config, "robot": robot}
            for p in params:
                self.shared.registry[p.name] = p
            # Golden = the values just loaded from the YAMLs at connect.
            # get_fns are guarded: they return None instead of raising when
            # the arm link is down, so a dying TCP channel yields ONE
            # aggregate warning here, not a traceback per parameter.
            missing = []
            for p in params:
                if p.get_fn is not None and p.apply != SESSION:
                    try:
                        value = p.get_fn()
                    except Exception as e:
                        logger.error("golden read failed for %s: %s", p.name, e)
                        value = None
                    if value is None:
                        missing.append(p.name)
                    else:
                        self.shared.golden[p.name] = value
            if missing:
                logger.warning(
                    "%s: no golden value for %d params (%s…) — arm config "
                    "link problem? They will show empty in the cockpit until "
                    "a read succeeds.",
                    arm, len(missing), ", ".join(missing[:3]),
                )
            self._ensure_listening()
        return self.shared

    def start(self) -> None:
        """Begin serving now (before any rig registers) so the cockpit is
        reachable during the connect phase and shows connection problems."""
        with self._lock:
            self._ensure_listening()

    def unregister(self, arm: str) -> None:
        with self._lock:
            self._rigs.pop(arm, None)
            for name in [n for n, p in self.shared.registry.items() if p.arm == arm]:
                self.shared.registry.pop(name, None)
                self.shared.golden.pop(name, None)
                with self.shared.lock:
                    self.shared.pending.pop(name, None)
                    self.shared.waiting.discard(name)

    def _ensure_listening(self) -> None:
        if self._httpd is not None:
            return
        try:
            self._httpd = ThreadingHTTPServer(
                ("0.0.0.0", self.port), _make_handler(self)
            )
        except OSError as e:
            logger.warning(
                "tuning server: port %d unavailable (%s) — cockpit disabled "
                "for this process", self.port, e,
            )
            return
        t = threading.Thread(
            target=self._httpd.serve_forever, name="tuning-http", daemon=True
        )
        t.start()
        logger.info(
            "tuning cockpit: http://%s:%d/", socket.gethostname(), self.port
        )

    def close(self) -> None:
        global _singleton
        if self._httpd is not None:
            self._httpd.shutdown()
            self._httpd = None
        with _singleton_lock:
            if _singleton is self:
                _singleton = None

    # ------------------------------------------------------------------
    # API implementations (server threads — no driver calls!)
    # ------------------------------------------------------------------

    def api_registry(self) -> dict:
        shared = self.shared
        vals = shared.values()
        with shared.lock:
            pending = dict(shared.pending)
            waiting = sorted(shared.waiting)
        params = []
        for name, p in shared.registry.items():
            d = p.describe()
            d["value"] = pending.get(name, vals.get(name))
            d["golden"] = shared.golden.get(name)
            d["pending"] = name in pending
            d["waiting"] = name in waiting
            params.append(d)
        session = {}
        for arm, rig in self._rigs.items():
            cfg = rig["config"]
            info = {"ip_address": getattr(cfg, "ip_address", "?")}
            for attr in ("loop_rate", "include_velocity", "include_effort",
                         "include_external_effort", "estop_device",
                         "estop_required", "flight_log_dir", "staged_positions",
                         "id"):
                if hasattr(cfg, attr):
                    v = getattr(cfg, attr)
                    info[attr] = list(v) if isinstance(v, tuple) else v
            session[arm] = info
        return {
            "params": params,
            "session": session,
            "arms": sorted(self._rigs),
            "standalone": self.standalone,
            "config_dir": str(self.cfg_dir),
        }

    def api_param(self, body: dict) -> dict:
        name = body["name"]
        if "joint" in body and body["joint"] is not None:
            return self.shared.request_joint(name, int(body["joint"]), body["value"])
        return self.shared.request(name, body["value"])

    def api_save(self) -> dict:
        vals = self.shared.values()
        saved = []
        if "leader" in self._rigs:
            saved.append(str(goldens.update_arm_golden(
                self.cfg_dir / "leader.yaml", vals, "leader")))
        if "follower" in self._rigs:
            saved.append(str(goldens.update_arm_golden(
                self.cfg_dir / "follower.yaml", vals, "follower")))
            leader_rig = self._rigs.get("leader")
            saved.append(str(goldens.save_tatbot_yaml(
                self._rigs["follower"]["config"],
                {"enabled": True, "port": self.port},
                self.cfg_dir,
                leader_config=leader_rig["config"] if leader_rig else None,
            )))
        # What we just wrote is the new golden.
        for name in self.shared.registry:
            if name in vals:
                self.shared.golden[name] = vals[name]
        return {"saved": saved}

    def api_revert(self) -> dict:
        reverted = []
        for name, gold in self.shared.golden.items():
            p = self.shared.registry.get(name)
            if p is None or p.apply == SESSION:
                continue
            cur = self.shared.values().get(name)
            if cur is not None and not _differs(cur, gold):
                continue
            self.shared.request(name, gold)
            reverted.append(name)
        return {"reverted": reverted}

    def api_capture_pose(self) -> dict:
        rig = self._rigs.get("follower")
        if rig is None:
            raise RuntimeError("no follower connected")
        with self.shared.lock:
            telem = self.shared.snapshot.get("arms", {}).get("follower")
        if not telem or "positions" not in telem:
            raise RuntimeError("no follower telemetry yet — is the loop running?")
        pose = [round(float(v), 5) for v in telem["positions"]]
        rig["config"].staged_positions = pose
        return {
            "staged_positions": pose,
            "note": "applied to the live config; Save to golden persists it "
                    "in tatbot.yaml (used from the next session start)",
        }

    def api_recover(self) -> dict:
        if not self.standalone:
            raise PermissionError(
                "recover is only available in a standalone tune session — "
                "during lerobot record/rollout use scripts/il_recover_arm.sh"
            )
        self.shared.recover_requested = True
        return {"queued": True}

    def api_stream_payload(self) -> dict:
        shared = self.shared
        now = time.time()
        with shared.lock:
            arms = {
                name: {**telem, "age": round(now - telem.get("t", now), 2)}
                for name, telem in shared.snapshot.get("arms", {}).items()
            }
            snap = {
                "t": shared.snapshot.get("t"),
                "arms": arms,
                "pending": sorted(shared.pending),
                "waiting": sorted(shared.waiting),
                "last_error": shared.snapshot.get("last_error"),
            }
        snap["dirty"] = sorted(shared.dirty())
        return snap


def _differs(a, b, tol=1e-6):
    from lerobot_robot_tatbot.params import _differs as d
    return d(a, b, tol)


# ---------------------------------------------------------------------------
# HTTP plumbing
# ---------------------------------------------------------------------------


def _cockpit_html() -> bytes:
    try:
        ref = resources.files("lerobot_robot_tatbot") / "cockpit.html"
        return ref.read_bytes()
    except Exception:
        fallback = Path(__file__).with_name("cockpit.html")
        return fallback.read_bytes()


def _make_handler(server: TuningServer):
    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, fmt, *args):  # route into logging, not stderr
            logger.debug("http: " + fmt, *args)

        def _json(self, obj, status=200):
            data = json.dumps(obj).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self):
            try:
                if self.path in ("/", "/index.html"):
                    body = _cockpit_html()
                    self.send_response(200)
                    self.send_header("Content-Type", "text/html; charset=utf-8")
                    self.send_header("Content-Length", str(len(body)))
                    self.end_headers()
                    self.wfile.write(body)
                elif self.path == "/api/registry":
                    self._json(server.api_registry())
                elif self.path == "/api/stream":
                    self._sse()
                else:
                    self._json({"error": "not found"}, 404)
            except (BrokenPipeError, ConnectionResetError):
                pass
            except Exception as e:
                logger.exception("GET %s failed", self.path)
                with contextlib.suppress(Exception):
                    self._json({"error": str(e)}, 500)

        def _sse(self):
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Cache-Control", "no-cache")
            self.end_headers()
            while True:
                payload = json.dumps(server.api_stream_payload())
                self.wfile.write(f"data: {payload}\n\n".encode())
                self.wfile.flush()
                time.sleep(0.1)

        def do_POST(self):
            try:
                length = int(self.headers.get("Content-Length") or 0)
                body = json.loads(self.rfile.read(length) or b"{}")
                if self.path == "/api/param":
                    self._json(server.api_param(body))
                elif self.path == "/api/save":
                    self._json(server.api_save())
                elif self.path == "/api/revert":
                    self._json(server.api_revert())
                elif self.path == "/api/capture_pose":
                    self._json(server.api_capture_pose())
                elif self.path == "/api/recover":
                    self._json(server.api_recover())
                else:
                    self._json({"error": "not found"}, 404)
            except KeyError as e:
                self._json({"error": f"unknown parameter: {e}"}, 404)
            except PermissionError as e:
                self._json({"error": str(e)}, 409)
            except (ValueError, RuntimeError) as e:
                self._json({"error": str(e)}, 400)
            except (BrokenPipeError, ConnectionResetError):
                pass
            except Exception as e:
                logger.exception("POST %s failed", self.path)
                with contextlib.suppress(Exception):
                    self._json({"error": str(e)}, 500)

    return Handler
