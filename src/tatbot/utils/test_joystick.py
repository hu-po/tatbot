"""Utility to inspect and validate the Atari CX Wireless Controller on this node.

Features:
- Enumerates evdev input devices and prints name, phys, vendor/product, path
- Compares exact and substring matches against expected Atari controller name
- Shows permissions/ownership for relevant /dev/input nodes
- Optionally attempts a connection using the Teleoperator implementation
- Optionally reads a few events to validate functionality (non-blocking)

Usage examples:
  uv run python src/tatbot/utils/test_joystick.py
  uv run python src/tatbot/utils/test_joystick.py --contains Atari --read-seconds 2 --connect
  uv run python src/tatbot/utils/test_joystick.py --name "Retro Games LTD  Atari CX Wireless Controller"
"""

from __future__ import annotations

import argparse
import os
import stat as statlib
import time
from pathlib import Path
import glob
from typing import Optional

import evdev
from evdev import InputDevice

from lerobot.teleoperators.gamepad import (
    AtariTeleoperator,
    AtariTeleoperatorConfig,
)


def format_mode(mode: int) -> str:
    return statlib.filemode(mode)


def print_header(title: str) -> None:
    print("\n" + title)
    print("-" * len(title))


def list_devices() -> list[str]:
    # Sort for stability
    paths = sorted(evdev.list_devices())
    if not paths:
        # Fallback: plain glob on /dev/input/event*
        paths = sorted(glob.glob("/dev/input/event*"))
    return paths


def describe_device(path: str) -> dict:
    d = InputDevice(path)
    info = {
        "path": path,
        "name": d.name,
        "name_stripped": d.name.strip() if isinstance(d.name, str) else d.name,
        "phys": getattr(d, "phys", None),
        "vendor": getattr(d.info, "vendor", None),
        "product": getattr(d.info, "product", None),
        "version": getattr(d.info, "version", None),
    }
    return info


def enumerate_evdev(expected_name: str, contains: Optional[str]) -> list[dict]:
    print_header("Evdev devices")
    try:
        import evdev as _ev  # noqa: F401
        print(f"evdev module: {evdev.__file__}")
    except Exception:
        pass
    infos: list[dict] = []
    for path in list_devices():
        try:
            info = describe_device(path)
        except Exception as exc:  # Permission or hotplug errors
            # Try to read sysfs name without opening the device
            sys_name = None
            try:
                ev = Path(path).name  # eventX
                sys_name_path = Path(f"/sys/class/input/{ev}/device/name")
                if sys_name_path.exists():
                    sys_name = sys_name_path.read_text(errors="ignore").strip()
            except Exception:
                pass
            info = {
                "path": path,
                "name": sys_name or "<unavailable>",
                "name_stripped": (sys_name or "").strip() if sys_name else None,
                "phys": None,
                "vendor": None,
                "product": None,
                "version": None,
                "error": str(exc),
            }
            print(f"{path}: name={info['name']!r} <error opening device: {exc}>")
        infos.append(info)
        name_stripped = info.get("name_stripped")
        vendor = info.get("vendor")
        product = info.get("product")
        vendor_str = f"0x{vendor:04x}" if isinstance(vendor, int) else "N/A"
        product_str = f"0x{product:04x}" if isinstance(product, int) else "N/A"
        print(
            f"{path}: name={info['name']!r} strip={name_stripped!r} "
            f"phys={info['phys']!r} id=({vendor_str},{product_str})"
        )

    exact_exists = any(i["name_stripped"] == expected_name for i in infos)
    print(f"\nDefault expected name: {expected_name!r}")
    print(f"Exact match exists? {exact_exists}")
    if contains:
        substr_matches = [i for i in infos if i["name_stripped"] and contains in i["name_stripped"]]
        print(f"Substring '{contains}' matches: {[i['name_stripped'] for i in substr_matches]}")
    return infos


def print_permissions_for(path: Path) -> None:
    try:
        st = path.stat()
    except FileNotFoundError:
        print(f"{path} -> <missing>")
        return
    except PermissionError as exc:
        print(f"{path} -> <permission error: {exc}>")
        return
    mode = format_mode(st.st_mode)
    try:
        import pwd, grp  # noqa: PLC0415

        owner = pwd.getpwuid(st.st_uid).pw_name
        group = grp.getgrgid(st.st_gid).gr_name
    except Exception:
        owner = st.st_uid
        group = st.st_gid
    print(f"{path} -> {owner}:{group} {mode}")


def show_permissions(infos: list[dict]) -> None:
    print_header("/dev/input permissions")
    # Print high-level nodes we care about
    for p in ["/dev/input/js0"]:
        print_permissions_for(Path(p))

    # Print event nodes that correspond to Atari or all when small
    for info in infos:
        name = (info.get("name_stripped") or "")
        if "Atari" in name or name == "":
            print_permissions_for(Path(info["path"]))

    # by-id and by-path symlinks for joystick
    print_header("/dev/input symlinks")
    for root in ["/dev/input/by-id", "/dev/input/by-path"]:
        root_path = Path(root)
        if not root_path.exists():
            continue
        for symlink in sorted(root_path.iterdir()):
            try:
                target = symlink.resolve()
            except Exception:
                target = "<unresolved>"
            print(f"{symlink} -> {target}")


def resolve_by_id_symlink() -> Optional[str]:
    # Prefer an explicit -event-joystick symlink if available
    for p in sorted(Path("/dev/input/by-id").glob("*-event-joystick")):
        try:
            target = str(p.resolve())
            return target
        except Exception:
            continue
    return None


def try_connect_via_teleop(name: str, read_seconds: float) -> None:
    print_header("Teleoperator connection test")
    cfg = AtariTeleoperatorConfig(device_name=name)
    teleop = AtariTeleoperator(cfg)
    try:
        teleop.connect()
        print(f"Connected: {teleop.is_connected}")
        if read_seconds > 0:
            print(f"Reading actions for ~{read_seconds:.1f}s...")
            end = time.time() + read_seconds
            while time.time() < end:
                action = teleop.get_action()
                # Only print when something changes or button press
                if action.get("red_button") or action.get("x") or action.get("y"):
                    print(f"action: {action}")
                time.sleep(0.05)
    except Exception as exc:
        print(f"Teleop connect failed: {exc}")
        print("Tip: ensure your user is in the 'input' group or run this once with sudo to test.")
    finally:
        try:
            teleop.disconnect()
        except Exception:
            pass


def read_events_direct(path: str, seconds: float) -> None:
    print_header(f"Direct evdev read from {path}")
    try:
        dev = InputDevice(path)
        print(f"Opened {path} as {dev.name!r}")
    except Exception as exc:
        print(f"Open failed: {exc}")
        return
    end = time.time() + seconds
    dev.grab = getattr(dev, "grab", None)  # type: ignore[attr-defined]
    try:
        while time.time() < end:
            event = dev.read_one()
            if event is not None:
                print(event)
            time.sleep(0.02)
    except PermissionError as exc:
        print(f"Permission error reading events: {exc}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", help="Exact evdev name to expect/use")
    parser.add_argument("--contains", default="Atari", help="Substring to search for when picking a device")
    parser.add_argument("--connect", action="store_true", help="Attempt teleoperator connect using detected name")
    parser.add_argument("--read-seconds", type=float, default=0.0, help="Read actions or events for N seconds")
    parser.add_argument("--device-path", help="Direct /dev/input/eventX path to read events from")
    parser.add_argument("--dump-proc", action="store_true", help="Print /proc/bus/input/devices if available")
    parser.add_argument("--prefer-by-id", action="store_true", help="Resolve and use by-id *-event-joystick path for direct reading")
    args = parser.parse_args()

    expected_name = args.name or AtariTeleoperatorConfig().device_name
    infos = enumerate_evdev(expected_name, args.contains)
    show_permissions(infos)

    if args.dump_proc:
        print_header("/proc/bus/input/devices")
        try:
            print(Path("/proc/bus/input/devices").read_text())
        except Exception as exc:
            print(f"<error reading proc devices: {exc}>")

    # Pick a candidate name by substring match first
    candidate_name: Optional[str] = None
    for info in infos:
        name = info.get("name_stripped")
        if name and args.contains and args.contains in name:
            candidate_name = name
            break
    if candidate_name is None:
        # Fallback to exact expected name if present
        for info in infos:
            if info.get("name_stripped") == expected_name:
                candidate_name = expected_name
                break

    print_header("Candidate selection")
    print(f"Candidate name: {candidate_name!r}")

    if args.connect and candidate_name:
        try_connect_via_teleop(candidate_name, read_seconds=max(0.0, args.read_seconds))

    if args.prefer_by_id and args.read_seconds > 0 and not args.device_path:
        by_id = resolve_by_id_symlink()
        if by_id:
            print_header("by-id direct read")
            print(f"Resolved by-id event device: {by_id}")
            read_events_direct(by_id, seconds=args.read_seconds)

    if args.device_path and args.read_seconds > 0:
        read_events_direct(args.device_path, seconds=args.read_seconds)

    print("\nDone.")


if __name__ == "__main__":
    main()
