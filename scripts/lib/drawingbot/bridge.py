"""One licensed JavaFX acquisition per bounded process group, with explicit replay state."""
from __future__ import annotations

import contextlib
import fcntl
import json
import os
import selectors
import signal
import subprocess
import time
import xml.etree.ElementTree as ET
from pathlib import Path

from drawingbot.artifacts import digest, write_json
from drawingbot.pens import verify_pen_readback
from drawingbot.recipe import DEFAULTS, read_json, runtime_project, save_bundle, verify_bundle


class Bridge:
    def __init__(self, app: Path, work: Path):
        jar = app / "lib/app/DrawingBotV3-Premium-1.6.22-stable-all.jar"
        if not jar.is_file():
            raise ValueError("this bridge requires a licensed Premium 1.6.22 installation")
        work.mkdir(parents=True, exist_ok=True)
        self.work, self.original_export, self.used = work, None, False
        self.process = self.selector = self.log = None
        self.pending = b""
        lock = Path.home() / ".cache/tatbot/drawingbot.lock"
        lock.parent.mkdir(parents=True, exist_ok=True)
        self.lock = lock.open("a")
        try:
            fcntl.flock(self.lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            self._start(app, jar)
        except BaseException:
            self.close()
            raise

    def _start(self, app, jar):
        source = Path(__file__).with_name("Dbv3Bridge.java")
        classpath = f"{self.work}:{app / 'lib/app/*'}"
        subprocess.run(["javac", "-cp", str(jar), "-d", str(self.work), str(source)], check=True, timeout=60)
        self.log = (self.work / "jvm.log").open("w")
        self.process = subprocess.Popen(
            ["xvfb-run", "-a", "-s", "-screen 0 1280x1024x24 -extension GLX", "java", "-Xlog:all=warning:stderr", "-Dprism.order=sw",
             f"-Duser.home={Path.home()}", "-cp", classpath, "Dbv3Bridge", str(self.work / "application.log")],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=self.log, start_new_session=True)
        self.selector = selectors.DefaultSelector()
        self.selector.register(self.process.stdout, selectors.EVENT_READ)
        self.identity = {"edition": "Premium", "version": "1.6.22", "jar_sha256": digest(jar),
                         "bridge_sha256": digest(source), "mode": "JavaFX with Xvfb, one job per process",
                         "runtime_flags": ["prism.order=sw", "xvfb:GLX=off"],
                         "java_version": subprocess.run(["java", "-Xlog:all=warning:stderr", "-version"], capture_output=True,
                                                        text=True, check=True, timeout=10).stderr.strip()}
        self._read(100)
        self._wait(lambda: self.call({"op": "status"})["loaded"], 90,
                   "Premium did not load; complete normal app activation first")
        self.original_export = self.call({"op": "exportSettings"})

    def _read(self, timeout: float):
        deadline = time.monotonic() + timeout
        while b"\n" not in self.pending:
            remaining = deadline - time.monotonic()
            if remaining <= 0 or not self.selector.select(remaining):
                raise TimeoutError("DrawingBotV3 did not respond; inspect the worker logs")
            data = os.read(self.process.stdout.fileno(), 65536)
            if not data:
                raise RuntimeError("DrawingBotV3 exited; inspect the worker logs")
            self.pending += data
            if len(self.pending) > 2_000_000:
                raise RuntimeError("DrawingBotV3 reply exceeded the protocol limit")
        line, self.pending = self.pending.split(b"\n", 1)
        reply = json.loads(line)
        if isinstance(reply, dict) and "error" in reply:
            raise RuntimeError(reply["error"])
        return reply

    def call(self, command: dict, timeout: float = 40):
        self.process.stdin.write((json.dumps(command) + "\n").encode())
        self.process.stdin.flush()
        return self._read(timeout)

    @staticmethod
    def _wait(predicate, timeout, message):
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if predicate():
                return
            time.sleep(.1)
        raise TimeoutError(message)

    def _apply_state(self, state):
        for name, value in state["area"].items():
            actual = self.call({"op": "areaChoice", "control": "clip" if name == "clipping" else name, "value": value})
            if actual["value"] != value:
                raise RuntimeError(f"drawing-area readback differs for {name}")
        actual = self.call({"op": "setExportPreferences", "values": state["export"]})
        if actual != state["export"]:
            raise RuntimeError("export preference readback differs from the recipe")

    def export(self, variant: dict, input_dir: Path, output_dir: Path, *, replay: Path | None = None,
               migrate_runtime: bool = False) -> dict:
        if self.used:
            raise RuntimeError("a worker accepts one acquisition; start a fresh Bridge")
        self.used = True
        output_dir.mkdir(parents=True, exist_ok=True)
        if any(output_dir.iterdir()):
            raise ValueError("acquisition output must be empty")
        try:
            return self._export(variant, input_dir, output_dir, replay, migrate_runtime)
        except BaseException as exc:
            write_json(self.work / "failure.json", {"variant": variant["id"], "error": str(exc),
                                                   "type": type(exc).__name__})
            raise

    def _export(self, variant, input_dir, output_dir, replay, migrate_runtime):
        source = input_dir / "source.png"
        self.call({"op": "validateImage", "file": str(source)})
        project = DEFAULTS / "project.json"
        state = variant.get("state", read_json(DEFAULTS / "state.json"))
        if replay is not None:
            recipe = verify_bundle(replay, self.identity, migrate_runtime=migrate_runtime)
            if digest(source) != recipe["files"]["source.png"]:
                raise ValueError("replay source differs from recipe")
            project, state = replay / "project.json", read_json(replay / "state.json")
        path = runtime_project(project, source, self.work / "project-runtime.json",
                               drawing_set=variant["drawing_set"] if replay is None else None)
        self.call({"op": "loadProject", "file": str(path)})
        self._wait(lambda: self.call({"op": "imageLoaded"}), 30, "project image did not load")
        self._apply_state(state)
        if replay is None:
            self.call({"op": "configure", "pfm": variant["pfm"],
                       "size_mm": variant["size_mm"], "pen_width_mm": variant["pen_width_mm"],
                       "settings": variant.get("settings", variant.get("overrides", {}))})
        settings = {"settings": self.call({"op": "pfmSettings"}),
                    "drawing": self.call({"op": "drawingSettings"}), **state}
        settings["drawing_set"] = verify_pen_readback(variant["drawing_set"],
            self.call({"op": "drawingSetSettings"}), settings["drawing"]["pen_width_mm"])
        if replay is None:
            expected = {"size_mm": variant["size_mm"], "pen_width_mm": variant["pen_width_mm"]}
            if settings["drawing"] != expected:
                raise RuntimeError(f"drawing settings readback differs: {settings['drawing']} != {expected}")
            for key, value in variant.get("settings", variant.get("overrides", {})).items():
                if settings["settings"].get(key) != value:
                    raise RuntimeError(f"PFM setting readback differs: {key}")
        if replay is not None and settings != recipe["effective"]:
            raise RuntimeError("replayed settings differ from the saved effective state")
        self.call({"op": "prepareBatch", "input": str(input_dir), "output": str(output_dir)})
        native = self.work / "native-project.json"
        self.call({"op": "saveProject", "file": str(native)})
        save_bundle(output_dir.parent / "recipe", native, source, variant, state, settings, self.identity)
        self.call({"op": "batch", "input": str(input_dir), "output": str(output_dir)})
        self._wait_export(output_dir / "source.svg", 300)
        final_pens = verify_pen_readback(variant["drawing_set"], self.call({"op": "drawingSetSettings"}),
                                        settings["drawing"]["pen_width_mm"])
        if final_pens != settings["drawing_set"]:
            raise RuntimeError("native pen settings changed during generation")
        return settings

    def _wait_export(self, path, timeout):
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            status = self.call({"op": "status"}, timeout=min(40, max(.01, deadline - time.monotonic())))
            if status.get("taskError"):
                raise RuntimeError(f"DBV3 task failed: {status['taskError']}")
            if path.is_file() and not status["batchDisabled"] and _complete_svg(path):
                return
            time.sleep(.1)
        raise TimeoutError("DBV3 export deadline exceeded")

    def close(self):
        if self.process is not None:
            if self.process.poll() is None:
                with contextlib.suppress(OSError, ValueError, RuntimeError, TimeoutError):
                    if self.original_export is not None:
                        self.call({"op": "setExportPreferences", "values": self.original_export}, timeout=2)
                    self.call({"op": "quit"}, timeout=2)
            stop_process_group(self.process)
            self.process.stdin.close()
            self.process.stdout.close()
        if self.selector is not None:
            self.selector.close()
        if self.log is not None:
            self.log.close()
        self.lock.close()


def stop_process_group(process):
    """Bound shutdown of our session, including descendants of an exited xvfb-run wrapper."""
    for sig in (signal.SIGTERM, signal.SIGKILL):
        with contextlib.suppress(ProcessLookupError):
            os.killpg(process.pid, sig)
        try:
            process.wait(timeout=2)
        except subprocess.TimeoutExpired:
            continue
        # The wrapper can exit before its children; still send the final group kill.
    process.wait(timeout=2)


def _complete_svg(path: Path) -> bool:
    try:
        ET.fromstring(path.read_bytes())
        return True
    except ET.ParseError:
        return False
