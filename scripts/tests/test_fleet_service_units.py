"""Every systemd unit a manifested service supervises must actually exist.

`fleet_service.sh` names units it watches. A template unit's instance is a
runtime fact — the login user of the node that runs it — so it has to be
resolved, not written down. On 2026-09-09 an export-scrub refactor replaced the
live instance `tatbot-viewer@<user>.service` with the placeholder
`tatbot-viewer@viewer-host.service`, which reads as a node-free name and
supervises nothing: the reporter exited 5 with "supervised service is not
active" and systemd restarted it in a loop. Nothing caught it until the service
was deployed, so this is the check that would have.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
SERVICE = REPO / "scripts/fleet_service.sh"
UNITS = REPO / "config/systemd"

# `--unit <name>` occurrences, in the order the script declares them.
UNIT_ARGUMENT = re.compile(r'--unit\s+"?([^\s"]+)"?')


def declared_units() -> list[str]:
    return UNIT_ARGUMENT.findall(SERVICE.read_text())


def manifested_units() -> set[str]:
    nodes = json.loads((REPO / "config/nodes.json").read_text())
    return {service["unit"] for node, record in nodes.items()
            if not node.startswith(("__", "//"))
            for service in record.get("services", [])}


def test_every_supervised_unit_is_a_real_unit_or_resolved_at_runtime():
    for unit in declared_units():
        if "$" in unit:
            continue  # resolved at runtime; covered by the tests below
        assert (UNITS / unit).is_file(), (
            f"{unit} is supervised but config/systemd/{unit} does not exist")


def test_a_template_instance_is_resolved_and_never_written_down():
    """`name@instance.service` where the instance is a host fact, not a constant."""
    for unit in declared_units():
        if "@" not in unit:
            continue
        instance = unit.split("@", 1)[1].removesuffix(".service")
        assert "$" in instance, (
            f"{unit} freezes a template instance; resolve it on the host that runs it")


def test_no_supervised_unit_name_carries_a_fleet_node_name():
    """The reason the placeholder was introduced is still a real requirement."""
    nodes = [n for n in json.loads((REPO / "config/nodes.json").read_text())
             if not n.startswith(("__", "//"))]
    for unit in declared_units():
        for node in nodes:
            assert not re.search(rf"(?<![a-z0-9]){re.escape(node)}(?![a-z0-9])", unit), \
                f"{unit} names node {node}; resolve it through config/nodes.json instead"


@pytest.mark.parametrize("service", ["zenohd", "zenoh-presence"])
def test_each_manifested_service_has_a_branch_that_runs_something(service):
    text = SERVICE.read_text()
    line = next((line for line in text.splitlines() if line.strip().startswith(f"{service})")), None)
    assert line is not None, f"{service} is manifested but fleet_service.sh has no branch"
    assert "runlog::run" in line, f"{service} must run under a run log"


def test_the_manifest_and_the_script_agree_on_what_exists():
    units = manifested_units()
    assert units, "no node manifests any service"
    for unit in units:
        assert (UNITS / unit).is_file(), f"manifested {unit} has no template"


def test_trackd_uses_overhead_only_with_the_shared_calibration_bundle():
    text = SERVICE.read_text()
    assert '{"overhead_depth_color", "overhead_depth_depth"} <= set(cameras)' in text
    assert 'd555_calibrated && D555_CALIBRATION=(--calibration "$CALIBRATION")' in text
    assert 'd555_calibrated && TRACKD_SOCKETS+=(--socket /tmp/tatbot-d555-frames.sock)' in text
    assert 'has_role poe-cameras && TRACKD_SOCKETS+=(--socket /tmp/tatbot-poe-frames.sock)' in text
    assert (directive("tatbot-trackd.service", "Requires") or "").split() == ["tatbot-visiond-d555.service"]
    assert (directive("tatbot-trackd.service", "Wants") or "").split() == ["tatbot-visiond-poe.service"]


def test_the_d555_owner_binds_dds_discovery_to_its_own_rig_address():
    """The D555 is an Ethernet camera found over DDS; its owner states the
    address discovery answers on, and that address is the owner's `lan`."""
    text = SERVICE.read_text()
    line = next(line for line in text.splitlines() if "--group overhead-depth" in line)
    assert '--dds-address "$DDS_ADDRESS"' in line
    assert 'address = load(Path(sys.argv[1])).get(sys.argv[2], {}).get("lan")' in text
    nodes = json.loads((REPO / "config/nodes.json").read_text())
    owners = [name for name, record in nodes.items() if not name.startswith(("__", "//"))
              and "tatbot-visiond-d555.service" in {s["unit"] for s in record.get("services", [])}]
    assert len(owners) <= 1, owners
    for owner in owners:
        assert "overhead-depth" in nodes[owner]["roles"]
        assert nodes[owner].get("lan"), f"{owner} owns the D555 but states no rig address"


def test_stencild_subscribes_to_both_owners_with_installed_references_and_no_node_name():
    """The fleet stencil observer is one subscriber beside the two trackers:
    the PoE socket always, the overhead socket only under the shared
    calibration bundle, references from the log root, the release's own
    interpreter, and a unit whose node is rendered at install time."""
    text = SERVICE.read_text()
    line = next(line for line in text.splitlines() if '"$RELEASE/bin/stencild"' in line)
    assert 'runlog::run' in line
    # the PoE cameras only where the node owns them (the demo stack has none),
    # the overhead under the shared bundle, and at least one of the two
    assert 'has_role poe-cameras && STENCILD_SOCKETS+=(--poe-socket /tmp/tatbot-poe-frames.sock)' in text
    assert 'd555_calibrated && STENCILD_SOCKETS+=(--overhead-socket /tmp/tatbot-d555-frames.sock)' in text
    assert '[ "${#STENCILD_SOCKETS[@]}" -gt 0 ] ||' in text
    # the lone D555 places a print by its artwork where no PoE camera can read it
    assert 'has_role poe-cameras || STENCILD_ARTWORK=(--overhead-artwork-match)' in text
    assert '"${STENCILD_ARTWORK[@]}"' in line
    assert '--calibration "$CALIBRATION"' in line
    assert '--references "$STENCIL_REFERENCES"' in line
    assert 'STENCIL_REFERENCES="$(tatbot_paths::log_root)/stencils/references"' in text
    assert '--python "$RELEASE/observer-venv/bin/python"' in line
    assert '--observer "$ROOT/scripts/vision/stencil_observer.py"' in line
    assert '--work "$STENCIL_WORK"' in line
    assert 'STENCIL_WORK="${RUNTIME_DIRECTORY:-$RUN_DIR}/captures"' in text
    assert '--output "$RUN_DIR/estimates.jsonl"' in line
    assert '--exclude-anchor' not in line   # every fixed camera may anchor a print
    # The wrist views: the registry names each arm's D405 and its owner, the
    # registrations beside the bundle pose them, the release's URDF is the FK.
    assert '--vision-config "$VISION_CONFIG"' in line
    assert '--registrations "$(dirname "$CALIBRATION")"' in line
    assert '--urdf "$ROOT/urdf/tatbot.urdf"' in line
    unit = "tatbot-stencild.service"
    assert (directive(unit, "Requires") or "").split() == ["tatbot-visiond-d555.service"]
    assert "tatbot-visiond-poe.service" in (directive(unit, "Wants") or "").split()
    assert directive(unit, "RuntimeDirectory") == "tatbot-stencild"
    assert directive(unit, "RuntimeDirectoryMode") == "0700"
    environment = [value for key, value in directives(unit) if key == "Environment"]
    assert "TATBOT_NODE=@NODE@" in environment, "the observer's node is an install-time fact"
    nodes = json.loads((REPO / "config/nodes.json").read_text())
    owners = [name for name, record in nodes.items() if not name.startswith(("__", "//"))
              and unit in {s["unit"] for s in record.get("services", [])}]
    # No node manifests it while the D555 waits for a demo-stack owner; at most one ever does.
    assert len(owners) <= 1, owners
    for owner in owners:
        assert "track" in nodes[owner]["roles"]
        service = next(s for s in nodes[owner]["services"] if s["unit"] == unit)
        assert service == {"binary": "stencild", "package": "trackd", "features": [],
                           "unit": unit, "env_file": None, "liveliness": "stencild"}


# --------------------------------------------------------------------------
# Readiness. `After=` orders process *starts*, so a Type=simple owner is
# "started" the moment it is exec'd -- ordering that reads like readiness and
# is not. A unit whose socket another unit orders against must therefore be
# Type=notify. On 2026-09-09 trackd exited 1 on ConnectionRefused against the
# capture owner's socket and only recovered on Restart=, ~6 s later.
# --------------------------------------------------------------------------

def directives(unit: str) -> list[tuple[str, str]]:
    """(key, value) for every `Key=value` line, comments and sections dropped."""
    pairs = []
    for line in (UNITS / unit).read_text().splitlines():
        line = line.strip()
        if not line or line.startswith(("#", ";", "[")) or "=" not in line:
            continue
        key, value = line.split("=", 1)
        pairs.append((key.strip(), value.strip()))
    return pairs


def directive(unit: str, key: str) -> str | None:
    values = [value for name, value in directives(unit) if name == key]
    return values[-1] if values else None


def wrapper_units() -> list[str]:
    """Units whose ExecStart is the session wrapper rather than a binary."""
    return [unit.name for unit in sorted(UNITS.glob("*.service"))
            if "fleet_service.sh" in (directive(unit.name, "ExecStart") or "")]


def test_a_notify_wrapper_unit_accepts_the_child_s_readiness():
    """The wrapper never exec's, so READY=1 cannot come from MAINPID.

    `runlog::run` deliberately runs the binary as a child so the run log's
    EXIT trap survives. systemd's default NotifyAccess=main therefore drops
    the datagram, the unit never reaches READY, and it is killed at
    TimeoutStartSec -- a unit that fails to start at all, which is worse than
    the race it was meant to fix. Verified both ways against systemd 259.
    """
    checked = 0
    for unit in wrapper_units():
        if directive(unit, "Type") != "notify":
            continue
        checked += 1
        assert directive(unit, "NotifyAccess") == "all", (
            f"{unit} is Type=notify but its ExecStart is the session wrapper, "
            "which runs the binary as a child; NotifyAccess=all or systemd "
            "drops the readiness datagram and the unit times out")
        assert directive(unit, "TimeoutStartSec") is not None, (
            f"{unit} is Type=notify without an explicit TimeoutStartSec; the "
            "90 s default can kill a capture that is merely slow to start")
    assert checked, "no wrapper unit declares Type=notify; this check went blind"


def test_a_unit_ordered_after_a_socket_owner_requires_a_ready_owner():
    """Ordering only means "connectable" if the owner declares readiness."""
    checked = 0
    for unit in sorted(UNITS.glob("*.service")):
        for after in (directive(unit.name, "After") or "").split():
            if not after.endswith(".service") or not (UNITS / after).is_file():
                continue
            if not _publishes_a_socket(after):
                continue
            checked += 1
            assert directive(after, "Type") == "notify", (
                f"{unit.name} is ordered after {after}, which binds a frame "
                f"socket, but {after} is Type={directive(after, 'Type')}: systemd "
                "releases the consumer as soon as the owner forks, so this "
                "ordering cannot mean the socket is connectable. Whether this "
                "consumer subscribes today is not the point -- the next one will")
    assert checked, "no unit is ordered after a socket owner; this check went blind"


def _publishes_a_socket(unit: str) -> bool:
    """True when the unit's fleet_service.sh branch binds a frame socket.

    A branch runs from its `name)` or `a|name)` label to its `;;`, often over
    several lines, so the whole branch is searched, not the label line."""
    exec_start = directive(unit, "ExecStart") or ""
    if "fleet_service.sh" not in exec_start:
        return False
    service = exec_start.rsplit(None, 1)[-1]
    branch: list[str] = []
    for line in SERVICE.read_text().splitlines():
        label = re.match(r"\s*([\w|-]+)\)", line)
        if not branch and not (label and service in label.group(1).split("|")):
            continue
        branch.append(line)
        if line.rstrip().endswith(";;"):
            break
    return "--socket " in "\n".join(branch)
