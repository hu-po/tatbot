from dataclasses import dataclass, field

from lerobot.robots.config import RobotConfig
from lerobot_robot_trossen.config_widowxai_follower import WidowXAIFollowerConfig

from lerobot_robot_tatbot import paths


def _golden_staged_positions() -> list[float]:
    """staged_positions from config/trossen/tatbot.yaml, or upstream's shape if
    the golden is unreadable (bench without a checkout)."""
    try:
        from lerobot_robot_tatbot import goldens

        pose = goldens.load_tatbot_yaml().get("follower", {}).get("staged_positions")
        if pose and len(pose) == 7:
            return [float(v) for v in pose]
    except Exception:  # noqa: BLE001 - a missing golden must not break construction
        pass
    return [0.0] * 7


@RobotConfig.register_subclass("tatbot_follower")
@dataclass
class TatbotFollowerConfig(WidowXAIFollowerConfig):
    """WidowX AI follower with tatbot defaults and bounded-grip gripper control.

    The gripper runs in external_effort mode under a soft proportional law
    toward the commanded gripper position, saturating at ``grip_force`` when
    closing. Constants mirror cpp/teleop/wxai_teleop.cpp, where they were
    derived from measured contact stiffness (~51 N/mm) and validated on the
    tattoo pen; keep the two in sync.
    """

    # Follower address from the hardware profile (TATBOT_FOLLOWER_IP env
    # override > profile driver stanza > empty, which fails at connect).
    ip_address: str = field(default_factory=lambda: paths.driver_default(
        "follower_ip", "TATBOT_FOLLOWER_IP"))

    # Bounds each commanded joint step (rad). Demo steps at 30 Hz are
    # <=0.05 rad so this never clips demonstrations; at rollout it caps a
    # misbehaving policy (or a stale-observation replan) at ~5 rad/s, under
    # the firmware's ~9.4 rad/s velocity trip. (Upstream default was 5.0 —
    # that allowed the velocity fault seen on the first rollout.)
    max_relative_target: float | None = 0.5

    # Slew limit for the SENT arm target (rad/s): send_action moves its
    # internal command toward the requested action by at most this rate, so
    # no policy/chunking misbehavior can command a violent step (firmware
    # velocity trip is ~9.4 rad/s). Demo tracking runs well below this;
    # raise toward ~4 if faithful replay of the fastest demo motions
    # (joint_3 peaked 3.9 rad/s) ever matters more than the safety margin.
    max_joint_velocity: float = 2.0
    # Optional controller-side velocity_max for all six arm joints. Unlike the
    # software target slew above, this limit is enforced by the arm controller.
    # Powered policy launchers commission at 0.75 rad/s; ordinary teleop keeps
    # the role golden unless it opts in.
    controller_velocity_limit: float | None = None

    # Flight recorder: every send_action appends commanded vs measured state
    # to a timestamped CSV under this directory (empty string disables).
    # Cheap (buffered writes at loop rate) and invaluable for diagnosing
    # rollout motion artifacts offline. "auto:<workflow>" resolves through
    # the log root (TATBOT_LOG_ROOT > config/runlog.json > XDG state); an
    # explicit path is used as-is.
    flight_log_dir: str = "auto:rollout"

    # Optional low-pass (EMA time constant, s) on the requested arm target,
    # applied before the slew limiter. 0 disables. Keep 0 for recording
    # (leader signal is already smooth; lag would degrade demos); rollout
    # scripts set ~0.3 to average away chunk-aggregation tremor — the
    # execution-side analogue of ACT temporal ensembling.
    target_filter_tau: float = 0.0

    # External effort is a policy observation (contact/force sensing).
    include_external_effort: bool = True
    # Keep the seven external-effort features on the policy wire but replace
    # their values with zero. Co-trained policies whose simulation source has
    # no force signal use this to preserve their 14-wide state normalizer
    # without introducing live-force train/serve skew. This affects only the
    # observation dictionary: motion safety continues to read measured effort
    # directly from the driver.
    mask_external_effort: bool = False

    # Empty keeps the camera's raw one-channel millimetre depth, which is the
    # recording contract. ``depth-v1`` replaces each live <camera>_depth value
    # with Tatbot's three-channel policy encoding while retaining the feature
    # name. Rollout launchers set this only for checkpoints trained on that
    # exact view; it never changes the safety sensors or recorded source data.
    depth_policy_encoding: str = ""

    # THE CARRIAGE IS THE CONTACT AXIS (2026-08-30). The tool sits in a bore
    # on the mount bolted to the left finger carriage, and the carriage moves
    # it along its own axis: 0.0 (the closed hard stop) is the pen at rest,
    # opening retracts it. Nothing is gripped, so there is no grip law; the
    # carriage runs in POSITION mode like the arm joints, held at the rest
    # value, and its external effort is the pen's contact force. Three
    # numbers replace the four grip-law constants; tatbot.yaml and the C++
    # teleop carry copies, and scripts/check_tool_sync.py fails if they drift.
    #
    # Where the carriage rests, metres. The closed hard stop is the only
    # repeatable datum; the touch-off is solved with the carriage here.
    carriage_rest_m: float = 0.0
    # Where a safety trip sends it. Firmware caps the carriage at 0.040 m
    # (config/trossen/follower.yaml), so this is the full 40 mm of lift the
    # mechanism has — the pen leaves the work before the arm freezes.
    carriage_retract_m: float = 0.040
    # Contact force above which the pen is retracted and the arm retreats:
    # the carriage effort's departure from its rest baseline, assessed only
    # while the arm moves at drawing speeds (tatbot_follower.CONTACT_STILL_RAD_S)
    # and sustained for carriage_cap_debounce steps. Bench 2026-08-30: the
    # estimate wanders -12..+10 N at rest and swung +-18 N during fast free
    # moves; a hand push read +10.9 N. 20 N therefore catches a HARD press,
    # not light contact — and deflection cannot substitute (0.02 mm under
    # that push; the hold is too stiff). Finer sensing needs a compliant
    # carriage (lower position kp on joint 6) or a real sensor.
    carriage_contact_cap_n: float = 20.0
    # Carriage deflection OPEN off its target that also counts as contact, m.
    # Bench 2026-08-30: the lead screw is self-locking — a hard push moved the
    # carriage 0.19 mm and it stayed — so this only catches a gross event.
    # The overforce guard (joint torques) is the contact detector; it
    # retracts the carriage too.
    carriage_contact_deflect_m: float = 0.002
    # Consecutive control steps over the cap before tripping (30 Hz -> 0.1 s).
    carriage_cap_debounce: int = 3
    # Interpolation horizon for carriage moves that are not trips, s.
    carriage_goal_time_s: float = 0.5
    # WHICH TOOL IS IN THE MOUNT. Required whenever use_tool_registry is on,
    # which is the default — tip offset, prompt phrase and ink policy all come
    # from it, so a wrong answer here is a wrong answer everywhere downstream.
    #
    # It used to be read out of config/workspace.yaml. That file records the
    # tool the last TOUCH-OFF used, so after a physical swap it names the
    # previous tool and hands over a complete, plausible, wrong set of
    # constants. Stating it makes the swap visible; tool_registry then checks
    # the statement against workspace.yaml and refuses if they disagree,
    # because the geometry under `right:` belongs to whatever is named there.
    ee_tool: str | None = None
    # Cross-check the STATED tool against the registry and the calibration in
    # workspace.yaml at connect: a tool with no mount (the laser pen) or one
    # that is not what the touch-off measured refuses to connect. False skips
    # the check for bench work with nothing in the mount; ee_tool is then
    # ignored.
    use_tool_registry: bool = True

    # Overforce guard: if any ARM joint's |external effort| exceeds
    # overforce_limit (Nm), send_action latches the present pose and ignores
    # incoming actions until the load falls below overforce_release for
    # overforce_hold_s. Demos draw at <=~4 Nm (p95) on the loaded joints; an
    # uncontrolled press escalates far beyond that and stalls the controller's
    # comm loop (observed: telemetry stream halt mid-press, 2026-08-20).
    # Applies identically to demos and rollouts (pairing preserved). 0 disables.
    # 2026-08-21: at 12.0 the guard fired at 12.1 Nm and the end effector was
    # still torn off. Two things were wrong and both are fixed here and in the
    # follower: the trip was far too late, and the response latched the pose the
    # arm was ALREADY pressed into, which holds the load rather than relieving
    # it. It now backs off to where the arm was before the press developed.
    #
    # Tuned on measured data, not intuition. What separates a runaway press from
    # ordinary drawing is not peak force but how long it lasts: on 2026-08-21 a
    # healthy rollout's longest excursion above 4 Nm was 0.17 s, while the
    # runaway held above it for 5.83 s. Trip counts over a healthy rollout, the
    # human teleop demonstrations, and that runaway:
    #
    #     4.0 Nm / 0.10 s ->  56 /  24 /  22   fires constantly on good runs
    #     4.0 Nm / 0.33 s ->   0 /   4 /   7   still fights human demos
    #     5.0 Nm / 0.33 s ->   0 /   0 /   7   clean separation
    #
    # So: sustained load, not a spike. A guard that fires during normal drawing
    # is worse than useless — it makes rollouts unusable and trains people to
    # raise the limit. The workspace floor (z_floor_m) is the primary defence;
    # this is the backstop for whatever the floor does not catch.
    #
    # RAISED 5.0 -> 9.0 on 2026-08-26 (operator), and the paragraph above is
    # exactly the trap this has to avoid, so here is the reasoning rather than
    # the intuition. That whole table was measured with the ballpoint, whose
    # tip is 63.7 mm past the fingertips. The fitted laser pen's is 136.3 mm.
    # Joint torque is tip force times lever, so the SAME tip force shows up as
    # ~2.1x the joint effort with this tool: 9.0 Nm here is a smaller tip force
    # than 5.0 Nm was with the ballpoint, not a larger one. Scaling by lever
    # alone would have allowed ~10.7 Nm; 9.0 keeps margin under that.
    #
    # It was firing on ordinary handling: three trips in 45 s of teleop
    # (5.3, 6.5, 6.5 Nm) with nobody pressing into anything, which is the
    # "fires during normal drawing" failure the table warns about — the tool
    # simply got longer and the old number stopped meaning what it meant.
    #
    # 9.0 stays clear of the 12.0 that fired too late on 2026-08-21. Note that
    # incident had TWO causes and only one was the number: the response also
    # latched the pose the arm was already pressed into. It retreats now.
    #
    # This should really be per-tool, derived from protrusion the way
    # il_touchoff's residual gate is — a constant tuned for one pen silently
    # means something different for the next one. Until then, RE-EXAMINE THIS
    # WHEN THE TOOL CHANGES.
    #
    # RE-EXAMINED 2026-08-30 (fixed mount): the ballpoint's tip now sits
    # ~60 mm past the mount bore face, which itself is ~40 mm out along the
    # finger, so the lever from the wrist is back in the ballpoint's class
    # (~100 mm), not the laser's 136. 9.0 Nm therefore reads as a LARGER tip
    # force than it did with the laser fitted; it is left in place as the
    # backstop only because the carriage contact cap (carriage_contact_cap_n)
    # now sees the tip force directly and trips first. Re-pick on the bench
    # with the cap, not before.
    overforce_limit: float = 9.0
    overforce_release: float = 2.5
    # Consecutive over-limit control steps before retreating (30 Hz -> ~0.33 s).
    overforce_debounce: int = 10
    # How far back to retreat when the guard trips, in seconds of pose history.
    overforce_retreat_s: float = 0.4
    overforce_hold_s: float = 0.5
    # A rolling criterion catches alternating dynamic loads that never reach
    # the consecutive debounce above. The 2026-08-27 incident spent 86.7% of
    # a 0.5 s window above 9 Nm; 57 successful arm-only flight logs stayed
    # below 5.2 Nm (the apparent outlier was carriage grip effort).
    overforce_window_s: float = 0.5
    overforce_window_fraction: float = 0.5
    overforce_window_min_samples: int = 8

    # Measured-motion aborts. These bound physical response, unlike the target
    # slew limiter above. Successful rollout logs peaked at 1.59 rad/s, 40.5
    # rad/s^2 and one fast reversal/s; the incident reached 5.31, 217 and 18.
    measured_velocity_abort: float = 2.5
    measured_acceleration_abort: float = 80.0
    reversal_window_s: float = 1.0
    reversal_min_velocity: float = 0.2
    reversal_abort_count: int = 4

    # The staged-to-policy-pose traverse clamps for about 1-2 s. After that
    # grace, clamping on most ticks is an out-of-envelope policy and aborts.
    clamp_abort_grace_s: float = 2.5
    clamp_abort_window_s: float = 1.0
    clamp_abort_fraction: float = 0.8
    clamp_abort_min_samples: int = 20

    # Hard workspace floor: the commanded pose is rejected if it would put the
    # tool below this height (metres, arm base frame, ee_gripper_link origin).
    # Unlike the overforce guard this is preventive rather than reactive — it
    # never lets the press happen, instead of reacting once the load is real.
    #
    # It exists because the policy is not responsible for depth: sim simulates
    # no contact at all, so a co-trained policy has never met a consequence for
    # driving through the surface (measured 2026-08-21: a co-trained checkpoint
    # drove ~60 mm deeper than a real-only one). Depth belongs to the
    # controller, and this is that control.
    #
    # Real draw-square recordings put ee_gripper_link at z ~= 0.082 m with the
    # pen on the paper, so 0.060 m leaves ordinary drawing untouched while
    # stopping a runaway descent. RAISE THIS if the table is raised; lower it
    # only with the tool clear of the work.
    #
    # This constant is a stand-in for a measured quantity: il_touchoff.py
    # solves the actual surface plane (and pen tip) into config/workspace.yaml,
    # and the fitted tool's own length is in config/tools/. Run
    # scripts/check_tool_sync.py to see the floor those two imply. Since
    # 2026-08-26 it does give one — that session planted its tip on the pad, so
    # paper_plane_z is the paper rather than the palette tag it used to be.
    #
    # Take the derived number as a prompt, not an answer. It is
    # `plane - reach - margin` with reach the largest component of the tip
    # offset, which assumes the tool hangs straight down from the gripper. The
    # laser's tip is mostly along +X (134.8 of 136.3 mm), so a scalar height on
    # ee_gripper_link does not bound where its tip actually is; the honest
    # bound depends on wrist orientation, which this floor cannot see.
    #
    # AND IT IS STILL TUNED FOR THE BALLPOINT. 0.060 came from draw-square
    # recordings that put ee_gripper_link at ~0.082 m with a 63.7 mm pen on the
    # paper. The fitted tool is now the laser at 136.3 mm — 73 mm longer — so
    # the same floor lets its tip sit far deeper than the margin this number
    # was chosen to express. Under teleop the operator's hand is the depth
    # control and this is only a backstop; a policy rollout has no such hand,
    # which is the case this floor exists for. Re-pick it deliberately before
    # rolling out a policy with a long tool fitted.
    # ABSOLUTE floor on the TOOL TIP, metres in the arm base frame. Used when
    # no measured surface is available, and as the deliberate override.
    #
    # It bounded ee_gripper_link until 2026-08-26, which is why 0.060 was the
    # number: with a 63.7 mm ballpoint, recordings put the gripper base at
    # ~0.082 m with the pen on paper, so the constant silently carried the
    # tool's length inside it. Swap to a 136.3 mm laser and the same 0.060
    # lets the tip go ~73 mm deeper — and by a different amount at every wrist
    # angle, because the laser's tip is 134.8 mm along +X, not down the z axis.
    # The floor now measures the tip itself through the URDF, so tool length
    # and orientation are handled by geometry rather than by a constant.
    #
    # None disables the floor entirely.
    z_floor_m: float | None = None
    # How far the tip may go BELOW the measured work surface, metres. This is
    # the real safety knob and the one worth arguing about: it is the only
    # quantity here that does NOT change with the table or the tool.
    #
    # Preferred over z_floor_m because the surface moves. The table can sit
    # above or below the robot base, pads and skins have thickness, and the
    # touch-off measures wherever it actually is (config/workspace.yaml
    # paper_plane_z, which may be negative). Resolving floor = surface - this
    # at connect means moving the table and re-running the touch-off updates
    # the floor, instead of leaving a stale constant that reads as protection.
    #
    # 10 mm matches tool_spec.derive_z_floor_m's margin. It has to clear the
    # calibration's own uncertainty — the laser's tip is good to a few mm —
    # without clipping legitimate contact, while still stopping the runaway
    # case this exists for (a co-trained checkpoint drove ~60 mm deep on
    # 2026-08-21). On a draped substrate the measured plane is the LOW ground,
    # so the real clearance over the mound is larger, not smaller.
    #
    # None falls back to z_floor_m.
    z_floor_below_surface_m: float | None = 0.010
    # URDF used for the floor's forward kinematics, relative to the repo root.
    # Must carry the FITTED tool's generated block (scripts/gen_tool_urdf.py):
    # the floor reads the tip out of it, so a stale block would measure the
    # previous tool's tip and quietly under-protect. Checked at connect.
    z_floor_urdf: str = "urdf/tatbot.urdf"
    # Production policy launchers set this true. It resolves the touch-off,
    # fitted-tool URDF and floor before staging, and refuses motion if any part
    # is missing or stale. False preserves hardware-free/bench workflows.
    require_z_floor: bool = False
    # A powered policy may only use a touch-off from the current work session.
    # Four hours permits setup/inspection without letting yesterday's table
    # position silently become today's safety plane. <=0 disables the age gate.
    z_floor_max_age_s: float = 4 * 60 * 60

    # Hardware e-stop (firmware/estop_pico/): while engaged, send_action
    # freezes the arm at the poses measured at engage time and ignores
    # incoming actions. Empty string disables the monitor entirely. With
    # Production defaults fail closed. Hardware-free bench code must opt out
    # explicitly with estop_device="".
    estop_device: str = field(default_factory=lambda: paths.driver_default(
        "estop_device", "TATBOT_ESTOP_DEVICE"))
    estop_required: bool = True
    # Policy sessions set this true so a press freezes the arm and terminates
    # inference; teleop/recording may keep the historical hold-and-resume path.
    abort_on_estop: bool = False

    # --- golden configs & tuning cockpit (docs/teleop_tuning.md) ---------

    # Arm golden YAML written into the controller at every connect (controller
    # state is scratch RAM that reverts on power cycle). Empty = auto:
    # config/trossen/follower.yaml resolved from the repo checkout or
    # $TATBOT_CONFIG_DIR. "-" disables loading (previous behavior).
    arm_config: str = ""

    # Load config/trossen/tatbot.yaml (grip law, smoothing, slew, poses) at
    # startup. Fields explicitly overridden on the CLI keep their CLI value.
    use_tatbot_yaml: bool = True

    # Move every connected arm TOGETHER: each plugin defers its staged
    # move until the first arm is used, and the first arm disconnected
    # lands the whole fleet. lerobot connects and disconnects the robot and
    # the teleoperator one at a time, so without this you watch the arms
    # rise and retract one after the other on every session. Set False to
    # restore strictly per-arm behaviour.
    coordinated_arms: bool = True

    # In-process tuning cockpit (HTTP + SSE). 0 or tuning_enabled=False
    # disables. One server per process; leader and follower share it.
    tuning_enabled: bool = True
    tuning_port: int = 8899

    # EXPERIMENTAL per-arm-joint leader→follower motion scaling (gripper
    # excluded). 1.0 everywhere = stock 1:1 mapping. Scaling anchors at the
    # pose where the scale was last changed; see docs/teleop_tuning.md.
    motion_scale: list[float] = field(default_factory=lambda: [1.0] * 6)

    # Staged/idle pose. The golden (config/trossen/tatbot.yaml) is the ONE
    # source — read at construction so the dataclass default mirrors it
    # without a literal copy (scripts/tests/test_staged_pose_single_source.py
    # forbids one). Upstream's default was the un-rolled gripper-era pose.
    staged_positions: list[float] = field(default_factory=lambda: _golden_staged_positions())

    def __post_init__(self):
        if self.mask_external_effort and not self.include_external_effort:
            raise ValueError(
                "mask_external_effort requires include_external_effort=True "
                "so the policy state remains 14-wide"
            )
        self.carriage_rest_m = min(max(float(self.carriage_rest_m), 0.0), 0.040)
        self.carriage_retract_m = min(max(float(self.carriage_retract_m), 0.005), 0.040)
        self.carriage_contact_cap_n = min(max(float(self.carriage_contact_cap_n), 2.0), 40.0)
        self.carriage_contact_deflect_m = min(max(float(self.carriage_contact_deflect_m), 0.0003), 0.02)
        self.carriage_cap_debounce = max(int(self.carriage_cap_debounce), 1)
        if self.depth_policy_encoding not in ("", "depth-v1"):
            raise ValueError(
                "depth_policy_encoding must be empty or 'depth-v1', got "
                f"{self.depth_policy_encoding!r}"
            )
        fractions = {
            "overforce_window_fraction": self.overforce_window_fraction,
            "clamp_abort_fraction": self.clamp_abort_fraction,
        }
        for name, value in fractions.items():
            if not 0.0 < value <= 1.0:
                raise ValueError(f"{name} must be in (0, 1], got {value}")
