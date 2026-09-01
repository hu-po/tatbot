import contextlib
import logging
import time
from pathlib import Path

import trossen_arm
from lerobot_teleoperator_trossen.widowxai_leader import WidowXAILeaderTeleop

from lerobot_robot_tatbot import goldens, params, recovery, runlog_shim
from lerobot_robot_tatbot.config_tatbot_leader import TatbotLeaderTeleopConfig
from lerobot_robot_tatbot.estop import EstopMonitor, acquire_estop, release_estop

logger = logging.getLogger(__name__)

VALUES_REFRESH_S = 5.0

# The firmware idles a motor (killing the session) above ~9.42 rad/s. Friction
# assist tuned past a joint's static friction makes the arm self-drive toward
# exactly that. Catch it with margin: above this speed, if a tuning change
# landed in the last few seconds, put it back — that is a runaway, not the
# operator (and with no recent change we only warn, so fast hand motion is
# never punished).
RUNAWAY_VELOCITY = 6.0  # rad/s

# Software viscous damping (see TatbotLeaderTeleopConfig.leader_damping).
# Sign determined ON HARDWARE 2026-08-20: -1.0 made the arm markedly more
# lurchy, i.e. it was ADDING energy with velocity. So in this firmware's
# external-effort convention a POSITIVE commanded effort opposes positive
# velocity. Do not "fix" this back without re-testing on the arm.
DAMPING_SIGN = 1.0
# Hard cap per joint so a wrong sign or a fast swing stays bounded and the
# runaway watchdog has room to act.
DAMPING_EFFORT_CAP = 2.0  # Nm
ARM_JOINTS = 6


class TatbotLeader(WidowXAILeaderTeleop):
    """WidowX AI leader with golden-config load and live leader-feel tuning.

    In the IL path the leader runs pure gravity compensation, so the
    friction-compensation and effort-correction tables loaded here ARE the
    entire feel of the arm in the artist's hand. get_action() additionally
    drains tuning-cockpit changes on the control-loop thread (the only thread
    allowed to touch the driver) and publishes leader telemetry.
    """

    config_class = TatbotLeaderTeleopConfig
    name = "tatbot_leader_teleop"

    def __init__(self, config: TatbotLeaderTeleopConfig):
        if config.use_tatbot_yaml:
            goldens.apply_section(config, goldens.load_tatbot_yaml().get("leader"))
        super().__init__(config)
        self.config = config
        self._tuning = None
        self._tuning_server = None
        self._values_published = 0.0
        self._damping_written = False
        self._estop: EstopMonitor | None = None
        self._estop_holding = False
        # Set by a coordinated startup so every arm lifts at once.
        self.defer_staging = False
        self._pending_stage: list[float] | None = None

    def configure(self) -> None:
        # Acquire the shared reader before any staged motion. The follower
        # acquires the same object, so serial bytes always have one consumer.
        if self._estop is not None:
            release_estop(self._estop)
            self._estop = None
        if self.config.estop_required and not self.config.estop_device:
            # Same refusal as the follower: profile failure must never
            # degrade a required e-stop into a silent skip.
            raise RuntimeError(
                "estop_required=True but no e-stop device resolved from the "
                "hardware profile — fix TATBOT_PROFILE/config/profiles, or "
                "set estop_required=False for an explicit hardware-free bench")
        if self.config.estop_device:
            self._estop = acquire_estop(
                self.config.estop_device, required=self.config.estop_required
            )
            if self._estop is not None:
                self._estop.wait_for_initial_state()
                if self._estop.engaged:
                    state = self._estop.state.value
                    release_estop(self._estop)
                    self._estop = None
                    raise RuntimeError(
                        f"e-stop engaged ({state}) — twist-release the button "
                        "(or reconnect the box) and retry"
                    )
        # Golden arm config first (controller state is scratch RAM), applied
        # field-by-field via setters (see TatbotFollower.configure for why
        # not load_configs_from_file), then the SDK leader end-effector
        # preset — gravity comp must match the hardware.
        if self.config.arm_config != "-":
            arm_yaml = (
                Path(self.config.arm_config).expanduser()
                if self.config.arm_config
                else goldens.config_dir() / "leader.yaml"
            )
            if arm_yaml.exists():
                applied = goldens.apply_arm_golden(
                    self.driver, trossen_arm, arm_yaml
                )
                self.driver.set_end_effector(
                    trossen_arm.StandardEndEffector.wxai_v0_leader
                )
                logger.info(
                    f"{self} applied golden {arm_yaml} ({', '.join(applied)})"
                )
            else:
                logger.warning(
                    f"{self} golden arm config {arm_yaml} missing — running "
                    "on the controller's current state"
                )
        err = recovery.controller_error(self.driver)
        if err:
            raise RuntimeError(
                f"{self} controller reports a firmware error after connect: "
                f"{err!r} — power-cycle the arm"
            )
        # Upstream's configure() is inlined here (rather than super()) so the
        # staged-pose check lands BETWEEN the move and the gravity-comp mode
        # switch: once the arm is in external_effort it is back-driveable by
        # design — it may sag, or the operator may already be holding it —
        # so a position check there would abort healthy sessions.
        staged = list(self.config.staged_positions)
        self.driver.set_all_modes(trossen_arm.Mode.position)
        if self.defer_staging or self.config.coordinated_arms:
            # Pin the target to where the arm stands before we step away:
            # position mode with an uncommanded target can jerk toward a
            # stale one when torque enables.
            with contextlib.suppress(Exception):
                self.driver.set_all_positions(
                    list(self.driver.get_all_positions()),
                    recovery.TAKEOVER_S, False,
                )
            recovery.arm_group.register(
                "leader", self, self.driver, staged, ARM_JOINTS,
                estop=self._estop,
            )
            # A coordinated startup will move every arm at once and then call
            # finish_staging(); blocking here is what makes the arms lift one
            # after another.
            self._pending_stage = staged
            return
        self._pending_stage = staged
        recovery.raise_arms_together(
            [("leader", self.driver, staged, ARM_JOINTS)],
            goal_time=2.0,
            estop=self._estop,
        )
        self.finish_staging()

    def finish_staging(self) -> None:
        """Verify the staged move and complete configuration."""
        staged = self._pending_stage or list(self.config.staged_positions)
        self._pending_stage = None
        recovery.assert_controller_healthy(self.driver, staged, str(self))
        self.driver.set_all_modes(trossen_arm.Mode.external_effort)
        self.driver.set_all_external_efforts(
            [0.0] * len(self.config.joint_names), goal_time=0.0, blocking=True
        )
        if self.config.tuning_enabled and self.config.tuning_port > 0:
            from lerobot_robot_tatbot.tuning_server import get_shared_server

            self._tuning_server = get_shared_server(self.config.tuning_port)
            self._tuning = self._tuning_server.register(
                "leader",
                params.build_leader_params(self.driver, trossen_arm, self.config),
                self.config,
            )
            self._tuning.publish_values()
            self._values_published = time.monotonic()

    def get_action(self) -> dict[str, float]:
        if self._estop is not None and self._estop.engaged:
            return self._enter_estop_hold()
        if self._estop_holding:
            self._leave_estop_hold()
        self._ensure_staged()
        action = super().get_action()
        if self._tuning is not None:
            velocities = [float(v) for v in self.driver.get_all_velocities()]
            worst = max((abs(v) for v in velocities[:6]), default=0.0)
            if worst > RUNAWAY_VELOCITY:
                self._tuning.note_instability(
                    "leader", f"leader joint hit {worst:.1f} rad/s"
                )
            self._tuning.apply_pending("leader", velocities)
            now_mono = time.monotonic()
            if now_mono - self._values_published > VALUES_REFRESH_S:
                self._tuning.publish_values()
                self._values_published = now_mono
            self._apply_damping(velocities)
            self._tuning.publish("leader", {
                "joints": list(self.config.joint_names),
                "positions": [
                    float(action[f"{j}.pos"]) for j in self.config.joint_names
                ],
                "velocities": velocities,
                "external_efforts": [
                    float(e) for e in self.driver.get_all_external_efforts()
                ],
            })
        return action

    def _ensure_staged(self) -> None:
        """First use lifts every deferred arm in the group, together."""
        if self._pending_stage is not None and not (
            self._estop is not None and self._estop.engaged
        ):
            recovery.arm_group.stage_pending()

    def _enter_estop_hold(self) -> dict[str, float]:
        positions = list(self.driver.get_all_positions())
        if not self._estop_holding:
            self.driver.set_all_modes(trossen_arm.Mode.position)
            self.driver.set_all_positions(positions, 0.0, False)
            self._estop_holding = True
            self._damping_written = False
            estop_state = self._estop.state.value if self._estop is not None else "engaged"
            logger.warning(
                "E-STOP engaged (%s): leader frozen",
                estop_state,
            )
            runlog_shim.event(
                "estop", state=estop_state, arm="leader"
            )
        return {
            f"{joint}.pos": float(position)
            for joint, position in zip(
                self.config.joint_names, positions, strict=True
            )
        }

    def _leave_estop_hold(self) -> None:
        # A pending coordinated stage owns the next mode change. During an
        # active session restore gravity compensation at the held pose.
        if self._pending_stage is None:
            self.driver.set_all_modes(trossen_arm.Mode.external_effort)
            self.driver.set_all_external_efforts(
                [0.0] * len(self.config.joint_names), 0.0, False
            )
        self._estop_holding = False
        logger.warning("E-stop released: leader tracking resumed")
        runlog_shim.event("estop", state="released", arm="leader")

    def _apply_damping(self, velocities) -> None:
        """Command effort opposing joint velocity — the damping the firmware's
        friction model cannot provide (all three of its terms are assist).

        Arm joints only: the leader's gripper carriage is left free for the
        hand (the follower ignores it since 2026-08-30 — its carriage is the
        tool's contact axis, owned by the safety layer). When damping returns to 0 the efforts are zeroed once,
        restoring plain gravity compensation.
        """
        damping = float(getattr(self.config, "leader_damping", 0.0) or 0.0)
        if damping <= 0.0:
            if self._damping_written:
                self.driver.set_arm_external_efforts(
                    [0.0] * ARM_JOINTS, 0.0, False
                )
                self._damping_written = False
            return
        efforts = [
            max(-DAMPING_EFFORT_CAP,
                min(DAMPING_EFFORT_CAP, DAMPING_SIGN * damping * v))
            for v in velocities[:ARM_JOINTS]
        ]
        self.driver.set_arm_external_efforts(efforts, 0.0, False)
        self._damping_written = True

    def disconnect(self, land: bool = True) -> None:
        """land=False: already brought down by a coordinated multi-arm
        landing; skip the motion and just release the hardware."""
        try:
            if self._estop is not None and self._estop.engaged:
                logger.warning(
                    "leader disconnect requested with e-stop engaged (%s): "
                    "holding until twist-release",
                    self._estop.state.value,
                )
                while self._estop.engaged:
                    time.sleep(recovery.ESTOP_POLL_S)
            if land and self.config.coordinated_arms and \
                    recovery.arm_group.names():
                recovery.arm_group.land()  # lands the whole fleet, once
                land = False
            if not land:
                with contextlib.suppress(Exception):
                    self.driver.set_all_modes(trossen_arm.Mode.idle)
                with contextlib.suppress(Exception):
                    self.driver.cleanup()
                return
            super().disconnect()
        except Exception as e:
            # Never leave the arm frozen in gravity comp: reconnect with
            # clear_error and bring it home → sleep → idle.
            logger.error(
                "graceful leader disconnect failed (%s) — attempting "
                "emergency park", e,
            )
            recovery.land_arm(
                self.config.ip_address,
                trossen_arm.StandardEndEffector.wxai_v0_leader,
                list(self.config.staged_positions),
                name="leader",
                estop=self._estop,
            )
        finally:
            recovery.arm_group.unregister("leader")
            release_estop(self._estop)
            self._estop = None
            if self._tuning_server is not None:
                self._tuning_server.unregister("leader")
                self._tuning_server = None
                self._tuning = None


# lerobot >= 0.6 resolves a device class by stripping "Config" off the config
# class name, so TatbotLeaderTeleopConfig sends it looking for
# "TatbotLeaderTeleop". Alias rather than rename: the class has been
# TatbotLeader since the plugin was written, and the config name is what
# register_subclass keys on.
TatbotLeaderTeleop = TatbotLeader
