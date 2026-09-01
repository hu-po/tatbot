#!/usr/bin/env python3
"""Dip the fitted tool into the palette's ink caps — scripted, monitored, ledgered.

    il_dip.py --ee-tool <id> [--slots inkcap_right_large ...] [--dry-run | --connect-only | --yes]
              [--ip <follower-ip>] [--allow-real] [--land]

The real-robot half of the ink system. A dip is
a scripted motion for now (learned dips are a later item): for each cap,
hover above the rim along the tool's own axis, plunge to the depth the cap's
fill level asks for (ink_spec.dip_plunge_m), dwell, retract, and move on.
The charge model runs alongside and every dip lands in the ledger with its
mode — `rehearsal` for the ballpoint, `real` for the 3RL.

Where the caps are: the rack's arc is the URDF's inkcap_* frames and the
rack itself sits at palette_root, both fixed joints off the rig root, mapped
into the arm's base frame (ink_spec.palette_root_in_base). The tool is
held VERTICAL over the rack (operator, 2026-08-28) — "vertical" being the
MEASURED tool axis, the line from the mount's bore face to the solved tip,
straight down — with the wrist yaw taken
from the recorded palette_center hold (config/poses.yaml) so the pose stays
near one the arm is known to make there; --as-held keeps that hold's tilt.

Gates, in order: the stated tool must match the calibration
(tool_spec.require_stated_tool); its ink policy must dip; a REAL tool is
refused until the ledger shows the choreography has been rehearsed with the
ballpoint and --allow-real is given; every target is sanity-checked against
the reach envelope; and nothing moves without --yes. The hardware e-stop is
honoured throughout: an engaged button freezes the arm at its measured pose
and the motion resumes on release. Ctrl+C is shielded during motion.

The palette's yaw about its own axis is taken as the rig's (the URDF's).
Where the rack SITS comes from --palette-from: `urdf` (palette_root off the
rig root, the default) or `hold` (the operator's measured palette_center
hold in config/poses.yaml — the tip planted on the palette's tag, so the
tag centre is known and the rack's root follows from the URDF's tag offset;
refused when the hold is older than --max-hold-age-h, because the rig is
not static). --palette-offset is the manual correction on top of either.

A dip belongs to a SESSION (scripts/lib/ink_session.py): the charge on the
needle carries from one rollout to the next, so a second `--dip` in the same
session is a top-up (or, with --if-needed, nothing at all), not a fresh
start. Without --slots the cap is chosen by ink_spec.select_slot with the
session's remaining need; --slots dips exactly those caps, reason "operator".
"""

from __future__ import annotations

import argparse
import contextlib
import math
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts" / "lib"))
sys.path.insert(0, str(REPO / "scripts" / "vision"))

import ink_session  # noqa: E402
import ink_spec  # noqa: E402
import numpy as np  # noqa: E402
import tool_spec  # noqa: E402
from urdf_kinematics import UrdfChain  # noqa: E402

TIP_LINK = tool_spec.tip_frame("right")  # right/tool_mount: the tip offset's frame
CARRIAGE_JOINT = "right/left_carriage_joint"
CARRIAGE_REST_M = 0.0
URDF = REPO / "urdf" / "tatbot.urdf"
POSES = REPO / "config" / "poses.yaml"
DEFAULT_IP = os.environ.get("TATBOT_FOLLOWER_IP", "")
DEFAULT_ESTOP = os.environ.get("TATBOT_ESTOP_DEVICE", "")
# The pad sits at 0.29 m; the rack at ~0.30 m radius. Anything past this is
# not the rack, whatever the arithmetic says.
MAX_REACH_M = 0.45
RIM_Z_RANGE_M = (0.03, 0.20)
ESTOP_POLL_S = 0.02


# --- geometry (pure; tested without hardware) --------------------------------------

@dataclass(frozen=True)
class HoldFrame:
    """The tool as the operator held it over the rack: EE rotation in the
    arm base frame and the unit tool axis (EE -> tip) in that frame."""

    rot_ee: np.ndarray      # (3, 3) base <- ee
    axis: np.ndarray        # (3,) unit, base frame, points from EE to tip
    tip_offset: np.ndarray  # (3,) tip in the EE frame (workspace.yaml)

    def ee_for_tip(self, tip_base: np.ndarray) -> np.ndarray:
        """EE position that puts the tip at ``tip_base`` with this rotation."""
        return np.asarray(tip_base, dtype=np.float64) - self.rot_ee @ self.tip_offset


@dataclass(frozen=True)
class CapTargets:
    slot_id: str
    rim: np.ndarray       # tip point at the rim centre, base frame
    above: np.ndarray     # hover, along the tool axis
    transit: np.ndarray   # above + a vertical lift, for moving between caps
    bottom: np.ndarray    # plunge, along the tool axis
    plunge_m: float
    fill_ul: float
    dwell_s: float = 0.0      # the ink's own dwell (inks.yaml dip:), else the tool's
    uptake_ul: float = 0.0    # what this dip credits, likewise
    ink_id: str | None = None
    reason: str = "operator"
    why_slot: str = ""


def rotation_to_axis_angle(rot: np.ndarray) -> np.ndarray:
    """(3,3) -> (3,) rotation vector, the driver's angle-axis convention."""
    rot = np.asarray(rot, dtype=np.float64)
    angle = math.acos(max(-1.0, min(1.0, (np.trace(rot) - 1.0) / 2.0)))
    if angle < 1e-9:
        return np.zeros(3)
    if abs(math.pi - angle) < 1e-6:
        # symmetric case: axis from the largest diagonal term
        m = (rot + np.eye(3)) / 2.0
        i = int(np.argmax(np.diag(m)))
        axis = np.sqrt(np.maximum(m[:, i], 0.0))
        axis[i] = math.sqrt(max(m[i, i], 0.0))
        for j in range(3):
            if j != i and m[i, j] < 0:
                axis[j] = -axis[j]
        axis /= np.linalg.norm(axis)
        return axis * angle
    axis = np.array([rot[2, 1] - rot[1, 2], rot[0, 2] - rot[2, 0], rot[1, 0] - rot[0, 1]])
    return axis / (2.0 * math.sin(angle)) * angle


def axis_angle_to_rotation(vec: np.ndarray) -> np.ndarray:
    vec = np.asarray(vec, dtype=np.float64)
    angle = float(np.linalg.norm(vec))
    if angle < 1e-12:
        return np.eye(3)
    k = vec / angle
    kx = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    return np.eye(3) + math.sin(angle) * kx + (1 - math.cos(angle)) * (kx @ kx)


def hold_frame_from_joints(chain: UrdfChain, joints, tip_offset: np.ndarray,
                           arm: str = "right") -> HoldFrame:
    """FK of a recorded hold -> the EE rotation in the ARM BASE frame.

    UrdfChain poses are in the rig root; the driver's Cartesian API is in the
    arm base, so the base's own fixed mount is divided out."""
    names = chain.arm_joint_names(arm)
    ee_root = chain.link_pose(TIP_LINK if arm == "right" else f"{arm}/ee_gripper_link",
                              {**dict(zip(names, joints, strict=False)),
                               CARRIAGE_JOINT: CARRIAGE_REST_M})
    base_root = chain.link_pose(f"{arm}/base_link")
    ee_base = np.linalg.inv(base_root) @ ee_root
    rot = ee_base[:3, :3]
    tip = np.asarray(tip_offset, dtype=np.float64)
    axis = rot @ tip
    axis = axis / np.linalg.norm(axis)
    return HoldFrame(rot_ee=rot, axis=axis, tip_offset=tip)


def vertical_frame(frame: HoldFrame) -> HoldFrame:
    """The same EE rotation turned so the tool axis points straight down —
    the minimal rotation from the held axis to -z, so the wrist keeps the
    yaw it had. Dips are vertical (operator, 2026-08-28): the old tilt came
    from an EE whose tool sat askew; the tool is aligned with the grippers
    now."""
    a = frame.axis / np.linalg.norm(frame.axis)
    down = np.array([0.0, 0.0, -1.0])
    v = np.cross(a, down)
    s, c = float(np.linalg.norm(v)), float(a @ down)
    if s < 1e-9:
        rot = np.eye(3) if c > 0 else np.diag([1.0, -1.0, -1.0])
    else:
        k = v / s
        kx = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
        rot = np.eye(3) + s * kx + (1 - c) * (kx @ kx)
    rot_ee = rot @ frame.rot_ee
    return HoldFrame(rot_ee=rot_ee, axis=rot_ee @ frame.tip_offset / np.linalg.norm(frame.tip_offset),
                     tip_offset=frame.tip_offset)


def read_hold_joints(poses_path: Path = POSES, name: str = "palette_center") -> list[float]:
    data = ink_spec.parse_simple_yaml(poses_path.read_text())
    pose = (data.get("poses") or {}).get(name)
    if not pose or "joints" not in pose:
        raise ValueError(f"{poses_path}: no measured pose {name!r} with joints")
    return [float(v) for v in pose["joints"]]


def tip_offset_from_workspace(workspace: dict, arm: str = "right") -> np.ndarray:
    tip = tool_spec.tip_offset_m(workspace, arm)
    if tip is None:
        raise ValueError(f"{tool_spec.WORKSPACE_RELPATH}: no tip offset in "
                         f"{tool_spec.tip_frame(arm)} — run a touch-off first (a "
                         "gripper-era file names another frame and does not count)")
    return np.array(tip, dtype=np.float64)


def cap_targets(slots: list[str], palette: dict, load: dict, layout: dict,
                palette_root_base: np.ndarray, frame: HoldFrame, policy,
                hover_m: float, lift_m: float,
                palette_offset: np.ndarray | None = None,
                inks: dict | None = None, reasons: dict | None = None) -> list[CapTargets]:
    """One CapTargets per slot. The ink in the cap refines the tool's dip
    (ink_spec.policy_with_ink): depth, dwell and uptake are the ink's where
    inks.yaml says so, the datasheet's otherwise."""
    out = []
    root = np.asarray(palette_root_base, dtype=np.float64)
    if palette_offset is not None:
        root = root + np.asarray(palette_offset, dtype=np.float64)
    for slot_id in slots:
        slot = palette[slot_id]
        rim = root + np.asarray(layout[slot_id], dtype=np.float64)
        fill = load[slot_id].fill_ul
        ink_id = load[slot_id].ink_id
        pol = ink_spec.policy_with_ink(policy, (inks or {}).get(ink_id) if ink_id else None)
        plunge = ink_spec.dip_plunge_m(pol, slot, fill)
        above = rim - frame.axis * hover_m
        reason, why = (reasons or {}).get(slot_id, ("operator", "named on the command line"))
        out.append(CapTargets(
            slot_id=slot_id, rim=rim, above=above,
            transit=above + np.array([0.0, 0.0, lift_m]),
            bottom=rim + frame.axis * plunge, plunge_m=plunge, fill_ul=fill,
            dwell_s=pol.dip_dwell_s, uptake_ul=pol.uptake_ul, ink_id=ink_id,
            reason=reason, why_slot=why))
    return out


def palette_root_from_hold(chain: UrdfChain, joints, tip_offset: np.ndarray,
                           repo: Path = REPO, arm: str = "right") -> np.ndarray:
    """The rack's root in the ARM BASE frame from the operator's measured
    hold: the tip was planted on the palette's AprilTag (poses.yaml
    palette_center, "on the palette tag"), so the tip's FK is the tag
    centre and the URDF's tag offset gives palette_root. The yaw is the
    rig's. Only as good as the tip offset the hold was recorded with —
    workspace.yaml's, which is the fitted tool's today."""
    names = chain.arm_joint_names(arm)
    ee_root = chain.link_pose(TIP_LINK if arm == "right" else f"{arm}/ee_gripper_link",
                              {**dict(zip(names, joints, strict=False)),
                               CARRIAGE_JOINT: CARRIAGE_REST_M})
    base_root = chain.link_pose(f"{arm}/base_link")
    tip_root = ee_root @ np.array([*np.asarray(tip_offset, dtype=np.float64), 1.0])
    tip_base = (np.linalg.inv(base_root) @ tip_root)[:3]
    return tip_base - np.asarray(ink_spec.tag8_in_palette_root(repo), dtype=np.float64)


def read_hold_utc(poses_path: Path = POSES, name: str = "palette_center") -> str | None:
    data = ink_spec.parse_simple_yaml(poses_path.read_text())
    pose = (data.get("poses") or {}).get(name) or {}
    return pose.get("utc")


def hold_age_h(utc: str | None, now: float | None = None) -> float | None:
    if not utc:
        return None
    from datetime import datetime, timezone

    t = datetime.strptime(str(utc), "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc).timestamp()
    return ((now if now is not None else time.time()) - t) / 3600.0


CAP_CLEARANCE_SIGMA = 1.5   # a cap is dippable when its radius covers this many tip-uncertainties


def tip_sigma_m(workspace: dict, arm: str = "right") -> float:
    """What the fitted tool's tip position is good to: the touch-off's rms
    residual, which il_touchoff writes next to the constants for exactly this
    reason. 0 when the workspace has no touch-off record (nothing known)."""
    rec = ((workspace.get(arm) or {}).get("touchoff") or {})
    try:
        return max(0.0, float(rec.get("residual_mm") or 0.0)) / 1000.0
    except (TypeError, ValueError):
        return 0.0


def caps_wide_enough(palette: dict, slots: list[str], sigma_m: float,
                     k: float = CAP_CLEARANCE_SIGMA) -> tuple[list[str], list[str]]:
    """(keep, refused): a cap whose radius is under k * sigma is a rim strike
    waiting to happen — with a 4 mm calibration the 8 mm and 11 mm caps are
    out and only the 15 mm one is honest."""
    keep, refused = [], []
    for slot in slots:
        radius = palette[slot].size.diameter_m / 2.0
        (keep if radius >= k * sigma_m else refused).append(slot)
    return keep, refused


def check_targets(targets: list[CapTargets]) -> list[str]:
    """Refusals, not warnings: a target outside the envelope is a wrong frame."""
    problems = []
    for t in targets:
        for label, p in (("above", t.above), ("bottom", t.bottom), ("transit", t.transit)):
            if np.linalg.norm(p[:2]) > MAX_REACH_M:
                problems.append(f"{t.slot_id}.{label} is {np.linalg.norm(p[:2]):.3f} m out — not the rack")
        if not (RIM_Z_RANGE_M[0] <= t.rim[2] <= RIM_Z_RANGE_M[1]):
            problems.append(f"{t.slot_id} rim z {t.rim[2]:.3f} m outside {RIM_Z_RANGE_M}")
        if t.plunge_m <= 0:
            problems.append(f"{t.slot_id} plunge {t.plunge_m * 1000:.1f} mm")
    return problems


@dataclass(frozen=True)
class Move:
    label: str
    tip: np.ndarray      # tip target, base frame
    seconds: float
    dwell_s: float = 0.0


def dip_moves(targets: list[CapTargets], policy, travel_s: float, plunge_speed: float,
              settle_s: float = 0.3) -> list[Move]:
    """The whole session as tip-space moves: transit -> above -> bottom (dwell)
    -> above -> transit, per cap, so every cap is entered and left along the
    tool axis and the moves between caps stay lifted."""
    moves = []
    for t in targets:
        plunge_s = max(0.8, (t.plunge_m + np.linalg.norm(t.above - t.rim)) / max(plunge_speed, 1e-4))
        moves.append(Move(f"{t.slot_id}: transit", t.transit, travel_s))
        moves.append(Move(f"{t.slot_id}: above rim", t.above, max(1.0, travel_s / 2), settle_s))
        moves.append(Move(f"{t.slot_id}: plunge {t.plunge_m * 1000:.1f} mm", t.bottom, plunge_s,
                          t.dwell_s if t.dwell_s > 0 else policy.dip_dwell_s))
        moves.append(Move(f"{t.slot_id}: retract", t.above, plunge_s))
        moves.append(Move(f"{t.slot_id}: lift", t.transit, max(1.0, travel_s / 2)))
    return moves


def real_gate(policy, events: list[dict], allow_real: bool) -> str | None:
    """Why a real tool may not dip yet, or None. The choreography has to have
    been rehearsed on this rig with the ballpoint before pigment is involved."""
    if not policy.touches_stock:
        return None
    rehearsed = [e for e in events if e.get("kind") == "dip" and e.get("mode") == "rehearsal"]
    if not rehearsed:
        return ("no rehearsal dips in the ledger — run this with the ballpoint "
                "(lutin-ballpoint-dot) into dry caps first")
    if not allow_real:
        return f"{len(rehearsed)} rehearsal dips on record; pass --allow-real to dip a real needle"
    return None


# --- hardware --------------------------------------------------------------------------

def tip_from_joints(chain: UrdfChain, joints, tip_offset: np.ndarray, arm: str = "right") -> tuple[np.ndarray, np.ndarray]:
    """FK of live joints -> (tip, ee) positions in the ARM BASE frame."""
    names = chain.arm_joint_names(arm)
    ee_root = chain.link_pose(TIP_LINK if arm == "right" else f"{arm}/ee_gripper_link",
                              {**dict(zip(names, joints, strict=False)),
                               CARRIAGE_JOINT: CARRIAGE_REST_M})
    base_root = chain.link_pose(f"{arm}/base_link")
    ee_base = np.linalg.inv(base_root) @ ee_root
    ee = ee_base[:3, 3]
    tip = ee + ee_base[:3, :3] @ np.asarray(tip_offset, dtype=np.float64)
    return tip, ee


def connect_only(args, chain: UrdfChain, frame: HoldFrame, targets: list[CapTargets], tool, log=print) -> int:
    """Everything run_hardware does BEFORE its first command, and nothing
    after: the e-stop is acquired, the driver configured, the controller's
    error state read, the joints read and FK'd, and the tip's distance to
    every planned point reported. No mode is set, no position commanded —
    the arm stays exactly as it is. This is the unattended check between a
    dry run and the first rehearsal (docs/ink.md)."""
    import trossen_arm
    from lerobot_robot_tatbot import recovery
    from lerobot_robot_tatbot.estop import acquire_estop, release_estop

    estop = acquire_estop(args.estop_device, required=True)
    estop.wait_for_initial_state()
    log(f"e-stop {args.estop_device}: {estop.state.value}" + (" — ENGAGED (a real dip would wait here)" if estop.engaged else ""))
    driver = trossen_arm.TrossenArmDriver()
    try:
        driver.configure(trossen_arm.Model.wxai_v0,
                         trossen_arm.StandardEndEffector.wxai_v0_follower,
                         args.ip, True, recovery.CONFIGURE_TIMEOUT_S)
        err = recovery.controller_error(driver)
        if err:
            log(f"controller reports {err!r} — power-cycle the arm")
            return 3
        joints = list(driver.get_all_positions())
        log(f"connected to {args.ip}; controller clean; joints {np.round(joints, 3).tolist()}")
        tip, ee = tip_from_joints(chain, joints, frame.tip_offset)
        log(f"  tip now {np.round(tip, 3)}  ee {np.round(ee, 3)}  (base frame, {tool.tool_id} tip offset)")
        for t in targets:
            log(f"  {t.slot_id:24s} to hover {np.linalg.norm(t.above - tip) * 1000:6.0f} mm   "
                f"to rim {np.linalg.norm(t.rim - tip) * 1000:6.0f} mm   rim below tip {(tip[2] - t.rim[2]) * 1000:6.0f} mm")
        log("  nothing commanded; the arm is where it was")
        return 0
    finally:
        with contextlib.suppress(Exception):
            driver.cleanup()
        release_estop(estop)


def _pose6(frame: HoldFrame, tip: np.ndarray) -> list[float]:
    ee = frame.ee_for_tip(tip)
    return [*map(float, ee), *map(float, rotation_to_axis_angle(frame.rot_ee))]


def _wait_phase(driver, seconds: float, estop, log) -> None:
    """Poll the clock with the e-stop in hand: an engaged button freezes the
    arm where it is (position hold at the measured pose) and the wait resumes
    when it is released — the same contract recovery._run_monitored_phase keeps."""
    import trossen_arm

    started = time.monotonic()
    while time.monotonic() - started < seconds + 0.15:
        if estop is not None and getattr(estop, "engaged", False):
            log("E-STOP engaged: holding the arm at its measured pose")
            present = list(driver.get_all_positions())
            driver.set_all_modes(trossen_arm.Mode.position)
            driver.set_all_positions(present, 0.0, False)
            while getattr(estop, "engaged", False):
                time.sleep(ESTOP_POLL_S)
            log("E-stop released: the remaining motion is re-issued")
            return
        time.sleep(ESTOP_POLL_S)


def run_hardware(args, frame: HoldFrame, targets: list[CapTargets], moves: list[Move],
                 policy, tool, log=print) -> int:
    import trossen_arm
    from lerobot_robot_tatbot import recovery
    from lerobot_robot_tatbot.estop import acquire_estop, release_estop

    estop = acquire_estop(args.estop_device, required=True)
    estop.wait_for_initial_state()
    if estop.engaged:
        release_estop(estop)
        log(f"e-stop engaged ({estop.state.value}) — release it and retry")
        return 3
    driver = trossen_arm.TrossenArmDriver()
    ledger = ink_spec.ledger_path()
    mirror = args.mirror
    sess = args.session
    dips_done = 0
    try:
        driver.configure(trossen_arm.Model.wxai_v0,
                         trossen_arm.StandardEndEffector.wxai_v0_follower,
                         args.ip, True, recovery.CONFIGURE_TIMEOUT_S)
        err = recovery.controller_error(driver)
        if err:
            log(f"controller reports {err!r} — power-cycle the arm")
            return 3
        start = list(driver.get_all_positions())
        driver.set_all_modes(trossen_arm.Mode.position)
        driver.set_all_positions(start, recovery.TAKEOVER_S, False)
        time.sleep(recovery.TAKEOVER_S + 0.1)
        with recovery.SigintShield():
            for move in moves:
                pose6 = _pose6(frame, move.tip)
                log(f"  {move.label:34s} tip {np.round(move.tip, 3)}  {move.seconds:.1f} s")
                for _attempt in range(3):
                    driver.set_cartesian_positions(
                        pose6, trossen_arm.InterpolationSpace.cartesian, move.seconds, False)
                    before = time.monotonic()
                    _wait_phase(driver, move.seconds, estop, log)
                    if time.monotonic() - before >= move.seconds:
                        break
                err = recovery.controller_error(driver)
                if err:
                    log(f"controller error mid-dip: {err!r}")
                    return 3
                if move.dwell_s > 0:
                    _wait_phase(driver, move.dwell_s, estop, log)
                if move.label.split(": ")[1].startswith("plunge"):
                    slot_id = move.label.split(":")[0]
                    t = next(t for t in targets if t.slot_id == slot_id)
                    record_dip(sess, policy, t, tool, args.run_id, ledger, mirror, ip=args.ip)
                    dips_done += 1
            log("returning to the start pose")
            driver.set_all_positions(start, max(2.0, args.travel_s), False)
            _wait_phase(driver, max(2.0, args.travel_s), estop, log)
        if args.land:
            ok = recovery.land_arms_together(
                [("follower", driver, list(args.staged), 6)], estop=estop)
            if not ok:
                log("landing did not verify — use il_recover_arm.sh")
                return 4
        return 0
    finally:
        log(f"{dips_done} dip(s) written to {ledger} as {policy.mode}"
            + (f" (session {sess.session_id}: charge {sess.charge_ul:.2f}/{sess.capacity_ul:.2f} uL)"
               if sess is not None else " (no session)"))
        release_estop(estop)


def record_dip(sess, policy, t: CapTargets, tool, run_id, ledger, mirror, **extra) -> dict:
    """The arm has plunged: the session credits the charge and writes the
    event; without a session (--session none) the event is still written,
    charge unknown."""
    if sess is not None:
        return ink_session.apply_dip(
            sess, policy, t.slot_id, t.ink_id, t.uptake_ul, t.reason, depth_m=t.plunge_m,
            why_slot=t.why_slot, run_id=run_id, ledger=ledger, mirror=mirror, **extra)
    return ink_spec.append_event(
        "dip", policy.mode, path=ledger, mirror=mirror, slot=t.slot_id, ink_id=t.ink_id,
        uptake_ul=t.uptake_ul, reason=t.reason, why_slot=t.why_slot, charge_before=None,
        charge_after=None, depth_m=t.plunge_m, tool_id=tool.tool_id, run_id=run_id,
        session_id=None, **extra)


def planned_need_ul(args, policy) -> float | None:
    """What the session is expected to spend: --need-ul, or the planned
    program's strokes through the charge arithmetic (ink_spec.need_from_polylines)."""
    if args.need_ul is not None:
        return float(args.need_ul)
    if args.program:
        import json

        doc = json.loads(Path(args.program).expanduser().read_text())
        return ink_spec.need_from_polylines(ink_spec.program_polylines(doc), policy,
                                            args.speed_mm_s / 1000.0)
    return None


# --- main ------------------------------------------------------------------------------

def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ee-tool", "--tool-id", dest="tool_id", required=True, help="the tool in the gripper, stated")
    ap.add_argument("--slots", nargs="*", help="palette slots (default: every usable right-arm cap)")
    ap.add_argument("--ip", default=DEFAULT_IP)
    ap.add_argument("--estop-device", default=DEFAULT_ESTOP)
    ap.add_argument("--hover-m", type=float, default=0.02)
    ap.add_argument("--lift-m", type=float, default=0.03, help="extra vertical lift between caps")
    ap.add_argument("--plunge-speed", type=float, default=0.02, help="m/s into and out of a cap")
    ap.add_argument("--travel-s", type=float, default=3.0, help="seconds per transit move")
    ap.add_argument("--as-held", action="store_true",
                    help="dip along the tool axis of the recorded palette hold instead of vertically")
    ap.add_argument("--palette-offset", nargs=3, type=float, metavar=("DX", "DY", "DZ"),
                    help="manual correction of palette_root in the base frame, metres")
    ap.add_argument("--palette-from", choices=["cal", "urdf", "hold"], default="cal",
                    help="where the rack sits: the URDF's palette_root, or the measured "
                         "palette_center hold (tip on the palette tag) in config/poses.yaml")
    ap.add_argument("--max-palette-age-h", type=float, default=168.0,
                    help="a palette calibration older than this is stale; fall back to the next source")
    ap.add_argument("--max-hold-age-h", type=float, default=24.0,
                    help="refuse --palette-from hold when the hold is older than this")
    ap.add_argument("--allow-stale-hold", action="store_true")
    ap.add_argument("--allow-real", action="store_true", help="permit a `real` ink tool")
    ap.add_argument("--dry-run", action="store_true", help="print the plan; touch nothing")
    ap.add_argument("--connect-only", action="store_true",
                    help="after the plan: acquire the e-stop, connect to the arm, read it, report the tip's "
                         "distance to every planned point, disconnect — the arm is never commanded")
    ap.add_argument("--yes", action="store_true", help="actually move the arm")
    ap.add_argument("--land", action="store_true", help="staged -> sleep -> idle afterwards")
    ap.add_argument("--run-id", default=None)
    ap.add_argument("--run-dir", default=None,
                    help="a run directory: events are mirrored into its ink.jsonl and "
                         "--run-id defaults to its name")
    ap.add_argument("--session", choices=["auto", "none"], default="auto",
                    help="auto: dip into the open ink session (opening one if none); "
                         "none: write ledger events only, no session state")
    ap.add_argument("--ink", default=None, help="which ink to dip (default: the session's, or any)")
    ap.add_argument("--need-ul", type=float, default=None,
                    help="what the session will spend, for the cap choice and --if-needed")
    ap.add_argument("--program", default=None,
                    help="planned strokes as JSON (run_meta.json, episode entry, language "
                         "program, or polylines in metres) — the need comes from them")
    ap.add_argument("--speed-mm-s", type=float, default=30.0)
    ap.add_argument("--if-needed", action="store_true",
                    help="dip only if the session's charge will not cover the need; "
                         "otherwise print why not and exit 0")
    args = ap.parse_args(argv)
    if args.run_dir and not args.run_id:
        args.run_id = Path(args.run_dir).expanduser().name
    args.mirror = (Path(args.run_dir).expanduser() / "ink.jsonl") if args.run_dir else None

    try:
        tool = tool_spec.require_stated_tool(args.tool_id, REPO, context="this dip")
    except Exception as exc:
        print(f"il_dip: refused: {exc}", file=sys.stderr)
        return 2
    policy = ink_spec.policy_for(tool)
    if not policy.dips:
        print(f"il_dip: {tool.tool_id} has ink.mode {policy.mode}; it never dips", file=sys.stderr)
        return 2
    why = real_gate(policy, ink_spec.read_events(), args.allow_real)
    if why:
        print(f"il_dip: refused: {why}", file=sys.stderr)
        return 2

    palette = ink_spec.load_palette(REPO)
    load = ink_spec.load_palette_load(REPO, palette)
    inks = ink_spec.load_inks(REPO)
    usable = [s.slot_id for s in ink_spec.usable_slots(policy, palette, load, "right", args.ink)]
    if not usable:
        want = f" of {args.ink}" if args.ink else ""
        print(f"il_dip: no usable right-arm cap{want} for {tool.tool_id} ({policy.mode}) — "
              "`scripts/ink.py load` one first", file=sys.stderr)
        return 2
    # The calibration's precision decides which caps the tip can be trusted into:
    # the tool's tip uncertainty and the rack's own placement uncertainty add.
    workspace = tool_spec.read_workspace(REPO)
    cal_for_budget = ink_spec.load_palette_cal(REPO)
    chosen_for_budget = ink_spec.choose_palette_root(
        cal_for_budget, ink_spec.palette_root_in_base(REPO), max_age_h=args.max_palette_age_h)
    sigma_m = math.hypot(tip_sigma_m(workspace), chosen_for_budget["residual_mm"] / 1000.0)
    usable, too_narrow = caps_wide_enough(palette, usable, sigma_m)
    if too_narrow:
        print(f"  tip calibration is ±{sigma_m * 1000:.1f} mm (touch-off rms); caps narrower than "
              f"{CAP_CLEARANCE_SIGMA:g}x that are out: {', '.join(too_narrow)}")
    if not usable:
        print(f"il_dip: refused: no cap is wide enough for a ±{sigma_m * 1000:.1f} mm tip — "
              "re-do the touch-off (or fix the tool's play in the gripper) before dipping", file=sys.stderr)
        return 2
    palette_ok = {k: v for k, v in palette.items() if k in usable or v.arm != "right"}

    # --- the session: whose charge is this dip topping up?
    sess = None
    need_ul = planned_need_ul(args, policy)
    if args.session == "auto":
        sess = ink_session.current()
        if sess is not None and sess.tool_id != tool.tool_id:
            print(f"il_dip: refused: session {sess.session_id} is {sess.tool_id}'s, not "
                  f"{tool.tool_id}'s — `scripts/ink.py session end` it first", file=sys.stderr)
            return 2
        if sess is not None and need_ul is not None and sess.need_ul != need_ul:
            sess.need_ul = need_ul
            ink_session.save(sess)
    fresh = sess is None
    charge_ul = 0.0 if fresh else sess.charge_ul
    remaining = need_ul if fresh else (sess.remaining_need_ul() if need_ul is None else need_ul)
    ink_now = None if fresh else sess.ink_id

    # --- which cap, and why
    reasons: dict[str, tuple[str, str]] = {}
    if args.slots:
        bad = [s for s in args.slots if s not in usable]
        if bad:
            print(f"il_dip: not usable for {tool.tool_id} ({policy.mode}): {bad}; usable: {usable}",
                  file=sys.stderr)
            return 2
        slots = list(args.slots)
    else:
        want = args.ink or ink_now
        if fresh or (sess.dips == 0 and charge_ul <= 0):
            why = "session_start"
        elif want is not None and ink_now is not None and want != ink_now:
            why = "color_change"
        else:
            why = sess.needs_dip(policy, remaining)
        if why is None:
            msg = (f"il_dip: session {sess.session_id} carries {charge_ul:.2f}/{sess.capacity_ul:.2f} uL"
                   f" of {ink_now}; " + (f"the remaining {remaining:.2f} uL is covered"
                                          if remaining is not None else "no need was stated"))
            if args.if_needed:
                print(msg + " — not dipping (--if-needed)")
                return 0
            why = "operator"
            print(msg + " — dipping anyway (pass --if-needed to skip)")
        choice = ink_spec.select_slot(policy, palette_ok, load, "right", want, inks, need_ul=remaining)
        if choice is None:
            print(f"il_dip: no usable right-arm cap of {want}", file=sys.stderr)
            return 2
        slots = [choice.slot_id]
        reasons[choice.slot_id] = (why, choice.reason)

    # --- where the rack is
    chain = UrdfChain(str(URDF))
    try:
        tip = tip_offset_from_workspace(workspace)
    except ValueError as exc:
        print(f"il_dip: refused: {exc}", file=sys.stderr)
        return 2
    hold_joints = read_hold_joints()
    frame = hold_frame_from_joints(chain, hold_joints, tip)
    if not args.as_held:
        frame = vertical_frame(frame)
    root_urdf = np.array(ink_spec.palette_root_in_base(REPO))
    root_hold = palette_root_from_hold(chain, hold_joints, tip)
    hold_utc = read_hold_utc()
    age_h = hold_age_h(hold_utc)
    cal = ink_spec.load_palette_cal(REPO)
    chosen = ink_spec.choose_palette_root(cal, root_urdf, max_age_h=args.max_palette_age_h)
    if args.palette_from == "hold":
        if age_h is None or age_h > args.max_hold_age_h:
            if not args.allow_stale_hold:
                print(f"il_dip: refused: the palette_center hold is "
                      f"{'undated' if age_h is None else f'{age_h:.0f} h old'} (> {args.max_hold_age_h:.0f} h);"
                      " the rig is not static — re-measure it, or --allow-stale-hold", file=sys.stderr)
                return 2
            print(f"il_dip: WARNING: using a {age_h:.0f} h old palette hold", file=sys.stderr)
        root_base, root_source = root_hold, "hold"
    elif args.palette_from == "urdf":
        root_base, root_source = root_urdf, "urdf"
    else:  # cal (default): the measured rack, tip authoritative, then vision, then urdf
        root_base, root_source = np.array(chosen["root"]), chosen["source"]
        age = "" if chosen["age_h"] is None else f", {chosen['age_h']:.0f} h old"
        print(f"  palette from {root_source}"
              + (f" (±{chosen['residual_mm']:.1f} mm{age})" if root_source != "urdf" else " (nominal; no calibration)"))
        if chosen.get("note"):
            print(f"    {chosen['note']}")
    layout = ink_spec.palette_layout_from_urdf(REPO)
    targets = cap_targets(slots, palette, load, layout, root_base, frame, policy,
                          args.hover_m, args.lift_m, args.palette_offset, inks, reasons)
    problems = check_targets(targets)
    if problems:
        for p in problems:
            print(f"il_dip: refused: {p}", file=sys.stderr)
        return 2
    moves = dip_moves(targets, policy, args.travel_s, args.plunge_speed)

    tilt = math.degrees(math.acos(max(-1.0, min(1.0, -float(frame.axis[2])))))
    print(f"{tool.tool_id} ({policy.mode}) — {len(slots)} cap(s), tool axis {tilt:.0f}° from vertical "
          f"({'as held at poses.yaml palette_center' if args.as_held else 'vertical; yaw from the palette_center hold'}), "
          f"palette_root {np.round(root_base, 3)} in the base frame ({root_source})")
    print(f"  palette_root: urdf {np.round(root_urdf, 3)}  hold {np.round(root_hold, 3)}"
          f" ({'undated' if age_h is None else f'{age_h:.0f} h old'}; "
          f"{np.linalg.norm(root_urdf - root_hold) * 1000:.0f} mm apart)")
    print("  " + ink_session.describe(sess, policy).replace("\n", "\n  ") if sess is not None
          else "  no open session — this dip opens one" if args.session == "auto" else "  no session (--session none)")
    for t in targets:
        state = "dry" if load[t.slot_id].dry else f"{load[t.slot_id].ink_id} {t.fill_ul:.0f} uL"
        print(f"  {t.slot_id:24s} rim {np.round(t.rim, 3)}  plunge {t.plunge_m * 1000:.1f} mm  ({state})"
              f"  {t.reason}: {t.why_slot}")
    total_s = sum(m.seconds + m.dwell_s for m in moves)
    print(f"  {len(moves)} moves, ~{total_s:.0f} s; ledger {ink_spec.ledger_path()}"
          + (f", mirrored to {args.mirror}" if args.mirror else ""))
    if args.dry_run:
        return 0
    if args.connect_only:
        return connect_only(args, chain, frame, targets, tool)
    if not args.yes:
        print("il_dip: pass --yes to move the arm (or --dry-run / --connect-only to stop here)", file=sys.stderr)
        return 2
    if args.session == "auto" and sess is None:
        sess = ink_session.start(tool, policy, need_ul=need_ul, mirror=args.mirror,
                                 note=f"opened by il_dip{' for ' + args.run_id if args.run_id else ''}")
        print(f"  opened session {sess.session_id}")
    args.session = sess
    args.load = load
    args.staged = tool_spec.staged_positions(REPO)  # the golden, not a copy
    return run_hardware(args, frame, targets, moves, policy, tool)


if __name__ == "__main__":
    sys.exit(main())
