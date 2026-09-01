"""Standalone teleop tuning session: `python -m lerobot_robot_tatbot.tune`.

Runs the minimal leader→follower mirroring loop (the same absolute-position
semantics lerobot-record uses, minus cameras and dataset) with the tuning
cockpit attached, so arm feel and tracking can be tuned without recording.
The cockpit's Recover button works here: it pauses mirroring, moves the
follower to the staged pose, and re-baselines.

Usage:
    python -m lerobot_robot_tatbot.tune [--leader-ip IP] [--follower-ip IP]
        [--rate HZ] [--port PORT] [--leader-only]

Ctrl+C retracts BOTH arms at once (staged pose, then sleep, then idle).
"""

from __future__ import annotations

import argparse
import logging
import signal
import sys
import time

from lerobot_robot_tatbot import paths

logger = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--leader-ip", default=paths.driver_default("leader_ip", "TATBOT_LEADER_IP"))
    parser.add_argument("--follower-ip", default=paths.driver_default("follower_ip", "TATBOT_FOLLOWER_IP"))
    parser.add_argument("--rate", type=float, default=30.0, help="loop Hz")
    parser.add_argument("--port", type=int, default=8899, help="cockpit port")
    parser.add_argument(
        "--leader-only", action="store_true",
        help="tune leader feel without connecting the follower",
    )
    parser.add_argument(
        "--no-estop", action="store_true",
        help="hardware-free bench only: explicitly disable the e-stop monitor",
    )
    parser.add_argument(
        "--ee-tool", dest="ee_tool", default=None,
        help="the tool in the mount, stated (a datasheet name in config/tools/); "
             "the follower refuses to connect without it unless --no-tool-registry",
    )
    parser.add_argument(
        "--no-tool-registry", action="store_true",
        help="bench with nothing in the mount: skip the tool cross-check",
    )
    args = parser.parse_args()
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(name)s %(message)s"
    )

    from lerobot_robot_tatbot.config_tatbot_follower import TatbotFollowerConfig
    from lerobot_robot_tatbot.config_tatbot_leader import TatbotLeaderTeleopConfig
    from lerobot_robot_tatbot.tatbot_follower import TatbotFollower
    from lerobot_robot_tatbot.tatbot_leader import TatbotLeader
    from lerobot_robot_tatbot.tuning_server import get_shared_server

    # lerobot reconfigures the root logger when imported, which silently
    # dropped every INFO line above (cockpit URL, connect progress, landing
    # narration). Re-assert it for the root and our own package.
    logging.getLogger().setLevel(logging.INFO)
    logging.getLogger("lerobot_robot_tatbot").setLevel(logging.INFO)
    if not logging.getLogger().handlers:
        logging.basicConfig(level=logging.INFO)

    server = get_shared_server(args.port)
    server.standalone = True
    # Serve immediately — the cockpit exists only while this process runs,
    # so make it reachable from the first moment and say where it is.
    server.start()
    import socket

    logger.info(
        "cockpit: http://%s:%d/ (from this machine: http://localhost:%d/)",
        socket.gethostname(), args.port, args.port,
    )

    estop_device = "" if args.no_estop else paths.driver_default(
        "estop_device", "TATBOT_ESTOP_DEVICE")
    leader = TatbotLeader(TatbotLeaderTeleopConfig(
        id="tune_leader", ip_address=args.leader_ip, tuning_port=args.port,
        estop_device=estop_device, estop_required=not args.no_estop,
    ))
    follower = None
    if not args.leader_only:
        follower = TatbotFollower(TatbotFollowerConfig(
            id="tune_follower", ip_address=args.follower_ip,
            tuning_port=args.port, loop_rate=int(args.rate),
            flight_log_dir="auto:tune",
            estop_device=estop_device, estop_required=not args.no_estop,
            ee_tool=args.ee_tool, use_tool_registry=not args.no_tool_registry,
        ))

    # SIGTERM (kill, systemd stop) parks exactly like Ctrl+C.
    def _sigterm(signum, frame):
        raise KeyboardInterrupt

    signal.signal(signal.SIGTERM, _sigterm)

    period = 1.0 / args.rate
    shared = server.shared
    exit_code = 0
    leader_started = follower_started = False
    try:
        from lerobot_robot_tatbot import recovery

        # Arms lift together via recovery.arm_group (config.coordinated_arms,
        # on by default) — the same mechanism lerobot-record and
        # lerobot-teleoperate get, so every teleop entry point behaves alike.
        logger.info("connecting leader (%s)…", args.leader_ip)
        leader.connect()
        leader_started = True
        if follower is not None:
            logger.info("connecting follower (%s)…", args.follower_ip)
            follower.connect()
            follower_started = True
        # Lift every connected arm at once, then finish each one's setup.
        recovery.arm_group.stage_pending()
        logger.info(
            "tuning session live at %.0f Hz — cockpit on port %d, "
            "Ctrl+C to end", args.rate, args.port,
        )
        while True:
            t0 = time.monotonic()
            action = leader.get_action()
            if follower is not None:
                if shared.recover_requested:
                    _recover(follower)
                    shared.recover_requested = False
                else:
                    follower.send_action(action)
            elif shared.recover_requested:
                logger.warning("recover requested but no follower connected")
                shared.recover_requested = False
            elapsed = time.monotonic() - t0
            if elapsed < period:
                time.sleep(period - elapsed)
    except KeyboardInterrupt:
        logger.info("ending tuning session — parking arms…")
    except Exception as e:
        # Driver fault (TCP/UDP loss, firmware trip) or any other failure:
        # abort the session instead of hammering a dead link. disconnect()
        # below parks each arm (home → sleep → idle), reconnecting with
        # clear_error if the current connection is wedged.
        logger.error("session aborted: %s", e)
        from lerobot_robot_tatbot import recovery

        for name, rig in (("leader", leader), ("follower", follower)):
            if rig is None:
                continue
            fw_err = recovery.controller_error(rig.driver)
            if fw_err:
                logger.error("%s firmware error: %s", name, fw_err)
        started = [
            name for name, ok in
            (("leader", leader_started), ("follower", follower_started)) if ok
        ]
        if started:
            logger.error(
                "landing %s together (staged → sleep → idle)…", " and ".join(started)
            )
        else:
            logger.error(
                "no arm was under our control when this failed — nothing to "
                "park, the arms were not moved"
            )
        exit_code = 1
    finally:
        # Land only arms this session actually took control of — a failed
        # connect() must not drive an arm we never touched.
        rigs = []
        if leader_started:
            rigs.append(("leader", leader))
        if follower is not None and follower_started:
            rigs.append(("follower", follower))
        if rigs:
            from lerobot_robot_tatbot import recovery

            with recovery.SigintShield():
                # The first disconnect lands the whole fleet together; the
                # rest then only release their hardware.
                for name, rig in rigs:
                    try:
                        rig.disconnect()
                    except Exception:
                        logger.exception("%s landing/teardown failed", name)
        server.close()
    if exit_code and (leader_started or follower_started):
        logger.error(
            "session ended on a fault — if an arm is not resting at the "
            "sleep pose, power-cycle it and run scripts/il_recover_arm.sh"
        )
    sys.exit(exit_code)


def _recover(follower) -> None:
    """Cockpit Recover: slow staged-pose move, then back to bounded-grip
    mirroring with fresh baselines. Mirrors scripts/il_recover_arm.sh."""
    import trossen_arm

    from lerobot_robot_tatbot import recovery

    if follower._estop is not None and follower._estop.engaged:
        logger.warning("RECOVER refused: e-stop engaged — release it first")
        return
    logger.warning("RECOVER: moving follower to staged pose")
    drv = follower.driver
    drv.set_all_modes(trossen_arm.Mode.position)
    # Hold the gripper where it is: recovering with a gripped tool must not
    # grind it against the position-mode effort saturation.
    staged = list(follower.config.staged_positions)
    staged[follower.GRIPPER] = drv.get_all_positions()[follower.GRIPPER]
    recovery.raise_arms_together(
        [("follower", drv, staged, follower.GRIPPER)],
        goal_time=4.0,
        estop=follower._estop,
    )
    # Re-enter bounded-grip mode and drop stale smoothing/slew state so the
    # next mirrored action starts from the recovered pose.
    modes = [trossen_arm.Mode.position] * len(follower.config.joint_names)
    modes[follower.GRIPPER] = trossen_arm.Mode.external_effort
    drv.set_joint_modes(modes)
    drv.set_joint_external_effort(follower.GRIPPER, 0.0, 0.0, False)
    follower._cmd_target = None
    follower._cmd_time = None
    follower._filt_target = None
    follower._scale_state = None
    follower._watchdog = recovery.TrackingWatchdog()
    logger.warning("RECOVER done — mirroring resumes")


if __name__ == "__main__":
    main()
