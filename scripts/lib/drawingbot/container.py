"""Private mounted-app acquisition container; build context contains only the public adapter."""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import tempfile
import uuid
from pathlib import Path

from drawingbot.artifacts import REPO, digest, write_json
from drawingbot.bridge import stop_process_group


def build(engine: str, image: str, output: Path) -> None:
    output.mkdir(parents=True, exist_ok=False)
    with tempfile.TemporaryDirectory(prefix="tatbot-dbv3-build-") as temp:
        context = Path(temp)
        shutil.copy2(REPO / "containers/drawingbot/Containerfile", context / "Containerfile")
        adapter = Path(__file__).parent
        sources = [*adapter.glob("*.py"), *adapter.glob("*.java"), *adapter.glob("defaults/*.json")]
        for source in sources:
            target = context / "scripts/lib/drawingbot" / source.relative_to(adapter)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target)
        shutil.copy2(REPO / "scripts/drawingbot.py", context / "scripts/drawingbot.py")
        shutil.copytree(REPO / "python/tatbot_contracts/src", context / "python/tatbot_contracts/src",
                        ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
        subprocess.run([engine, "build", "--platform", "linux/amd64", "-t", image, str(context)], check=True)
    identity = json.loads(subprocess.check_output([engine, "image", "inspect", image], text=True))[0]
    write_json(output / "image.json", {"id": identity["Id"], "image": image,
                                      "containerfile_sha256": digest(REPO / "containers/drawingbot/Containerfile")})
    packages = subprocess.check_output([engine, "run", "--rm", "--network=none", "--entrypoint", "cat",
                                        identity["Id"], "/opt/tatbot/runtime-packages.txt"], text=True)
    (output / "runtime-packages.txt").write_text(packages)


def run(engine: str, image: str, app: Path, output: Path, *, recipe: Path | None = None,
        state: Path | None = None, migrate_runtime: bool = False) -> None:
    """Smoke runs without license state or network; export takes a dedicated, normally activated home."""
    if recipe is not None and (state is None or not state.is_dir()):
        raise ValueError("container export needs --state: its own normally activated container home")
    if state is not None and state.resolve() in (Path.home().resolve(), (Path.home() / ".DrawingBotV3").resolve()):
        raise ValueError("use a dedicated container home; do not mount host activation identity")
    output.mkdir(parents=True, exist_ok=False)
    name = f"tatbot-dbv3-{uuid.uuid4().hex}"
    argv = [engine, "run", "--rm", "--init", "--name", name, "--read-only", "--cap-drop=ALL", "--security-opt=no-new-privileges",
            "--pids-limit=256", "--tmpfs", "/tmp:rw,size=512m", "--env", "HOME=/state"]
    if engine == "podman":
        argv += ["--userns=keep-id"]
    argv += ["--user", f"{os.getuid()}:{os.getgid()}", "-v", f"{app}:/app:ro", "-v", f"{output}:/out:rw"]
    if recipe is None:
        argv += ["--network=none", "--tmpfs", "/state:rw,size=128m,mode=1777", image,
                 "runtime-probe", "--app", "/app", "--out", "/out/probe"]
    else:
        argv += ["-v", f"{state}:/state:rw", "-v", f"{recipe}:/recipe:ro", image,
                 "export", "/recipe", "--app", "/app", "--out", "/out/job"]
        if migrate_runtime:
            argv += ["--migrate-runtime"]
    identity = json.loads(subprocess.check_output([engine, "image", "inspect", image], text=True))[0]
    # Execute the inspected immutable image, even if another build retags the name.
    argv[argv.index(image)] = identity["Id"]
    write_json(output / "container.json", {"image_id": identity["Id"], "mode": "export" if recipe else "runtime-probe",
                                          "app_jar_sha256": digest(app / "lib/app/DrawingBotV3-Premium-1.6.22-stable-all.jar")})
    try:
        subprocess.run(argv, check=True, timeout=600)
    finally:
        subprocess.run([engine, "rm", "-f", name], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=20)


def runtime_probe(app: Path, output: Path) -> None:
    """Compile the actual adapter and render JavaFX controls without launching Premium or activating."""
    output.mkdir(parents=True, exist_ok=False)
    jar = app / "lib/app/DrawingBotV3-Premium-1.6.22-stable-all.jar"
    here = Path(__file__).parent
    subprocess.run(["javac", "-cp", str(jar), "-d", str(output), str(here / "Dbv3Bridge.java"),
                    str(here / "FxRuntimeProbe.java")], check=True, timeout=60)
    command = ["xvfb-run", "-a", "-s", "-screen 0 1280x1024x24 -extension GLX", "java", "-Xlog:all=warning:stderr", "-Dprism.order=sw",
               f"-Duser.home={Path.home()}", "-cp", f"{output}:{jar}", "FxRuntimeProbe", str(output / "probe.png")]
    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
                               start_new_session=True)
    try:
        stdout, stderr = process.communicate(timeout=45)
        (output / "runtime.log").write_text(stderr)
        if process.returncode:
            raise RuntimeError(f"JavaFX runtime probe failed; inspect {output / 'runtime.log'}")
        (output / "stdout.log").write_text(stdout)
        records = [line for line in stdout.splitlines() if line.startswith('{"javafx_controls":')]
        if len(records) != 1:
            raise RuntimeError("JavaFX probe did not emit one result; inspect stdout.log")
        result = json.loads(records[0])
        result.update(scope="JavaFX runtime and adapter compilation only; no Premium launch or licensed export",
                      jar_sha256=digest(jar), preview_sha256=digest(output / "probe.png"))
        write_json(output / "result.json", result)
    finally:
        stop_process_group(process)
