"""Parameter registry for the teleop tuning cockpit.

Every tunable knob — whether it lives in the arm controller (driver ``set_*``
calls) or in our plugin code (grip law, smoothing, slew) — is declared ONCE
here with its range, units, apply-safety class, and which golden YAML it
persists to. The tuning server renders its UI from this registry and the
control loop applies changes through it, so adding a knob later is one
declaration and zero frontend work.

Apply-safety classes:
  LIVE       — safe to change while the arm moves (clamps, filters, friction).
  HELD_STILL — applied only when every joint is at rest (stiffness, limits);
               a pending change waits and the UI shows the gate state.
  SESSION    — read-only in the UI; set via config file / CLI, needs a new
               session (observation schema, rates, rig identity).

Driver-side writes MUST run on the control-loop thread (one TrossenArmDriver
owns the arm's UDP session and is not thread-safe): the server enqueues
changes into TuningShared.pending and the plugins drain the queue once per
loop via apply_pending().
"""

from __future__ import annotations

import logging
import math
import threading
import time
from dataclasses import dataclass
from typing import Any, Callable

logger = logging.getLogger(__name__)

LIVE = "live"
HELD_STILL = "held_still"
SESSION = "session"

DRIVER = "driver"
PLUGIN = "plugin"

# A joint counts as "at rest" below this speed (rad/s, m/s for the gripper).
REST_VELOCITY = 0.05
ARM_JOINTS = 6  # joints 0-5; index 6 is the gripper carriage


@dataclass
class Param:
    """One tunable parameter. ``get_fn``/``set_fn`` are bound by build_registry."""

    name: str
    group: str  # leader-feel | follower-tracking | carriage | safety | session
    label: str
    doc: str
    units: str = ""
    minimum: float | None = None
    maximum: float | None = None
    step: float = 0.01
    per_joint: int = 0  # 0 = scalar, else vector length (6 arm / 7 all)
    apply: str = LIVE
    owner: str = PLUGIN
    arm: str = "follower"  # leader | follower
    persist: str | None = None  # leader.yaml | follower.yaml | tatbot.yaml
    # Per-joint [min, max] pairs. Required wherever joints have different
    # physical scales — the arm joints carry Nm and the gripper carriage N,
    # so one shared range would let a slider drag put an arm-joint value
    # dozens of times past anything sane (this is what self-drove joint 4
    # into the firmware velocity trip on 2026-08-20).
    limits: list[tuple[float, float]] | None = None
    # Seconds to traverse a joint's full range when a change is applied. Set
    # on DESTABILIZING parameters (friction assist, gravity-comp scaling,
    # stiffness): the control loop walks toward the target instead of
    # stepping, so the operator feels the approach to instability rather
    # than being thrown by it. None = apply immediately.
    ramp_seconds: float | None = None
    get_fn: Callable[[], Any] | None = None
    set_fn: Callable[[Any], None] | None = None

    def joint_limits(self, index: int) -> tuple[float | None, float | None]:
        if self.limits is not None:
            return self.limits[index]
        return self.minimum, self.maximum

    def clamp(self, value):
        def one(v, lo, hi):
            v = float(v)
            if lo is not None:
                v = max(lo, v)
            if hi is not None:
                v = min(hi, v)
            return v

        if self.per_joint:
            vals = list(value)
            if len(vals) != self.per_joint:
                raise ValueError(
                    f"{self.name}: expected {self.per_joint} values, got {len(vals)}"
                )
            return [one(v, *self.joint_limits(i)) for i, v in enumerate(vals)]
        return one(value, self.minimum, self.maximum)

    def ramp_steps(self, dt: float) -> list[float] | float:
        """Max change allowed this tick — a list per joint, or a scalar."""
        if self.ramp_seconds is None or self.ramp_seconds <= 0:
            return [float("inf")] * self.per_joint if self.per_joint else float("inf")

        ramp_sec = self.ramp_seconds

        def span(lo, hi):
            if lo is None or hi is None:
                return 1.0
            return max(abs(hi - lo), 1e-9)

        if self.per_joint:
            return [
                span(*self.joint_limits(i)) / ramp_sec * dt
                for i in range(self.per_joint)
            ]
        return span(self.minimum, self.maximum) / ramp_sec * dt

    def describe(self) -> dict:
        return {
            "name": self.name,
            "group": self.group,
            "label": self.label,
            "doc": self.doc,
            "units": self.units,
            "min": self.minimum,
            "max": self.maximum,
            "limits": (
                [list(pair) for pair in self.limits] if self.limits else None
            ),
            "step": self.step,
            "per_joint": self.per_joint,
            "apply": self.apply,
            "arm": self.arm,
            "ramped": self.ramp_seconds is not None,
        }


class TuningShared:
    """State shared between the tuning server thread and the control loop.

    The server only ever touches ``pending`` (under the lock) and reads
    ``snapshot``/``golden``; every driver/plugin write happens on the control
    loop thread in apply_pending().
    """

    def __init__(self):
        self.lock = threading.Lock()
        self.registry: dict[str, Param] = {}
        self.pending: dict[str, Any] = {}
        self.waiting: set[str] = set()  # HELD_STILL params gated on motion
        self.golden: dict[str, Any] = {}
        self.snapshot: dict[str, Any] = {}
        self.session_info: dict[str, Any] = {}
        self.recover_requested = False
        self.revert_requested = False
        self._appliers: list[Callable[[], None]] = []
        self._last_apply: dict[str, float] = {}   # arm -> monotonic of last tick
        self._ramping: set[str] = set()           # params mid-ramp
        self._last_change: dict[str, tuple] = {}  # arm -> (name, prev, when)

    # -- server side ------------------------------------------------------

    def request(self, name: str, value) -> dict:
        with self.lock:
            param = self.registry.get(name)
            if param is None:
                raise KeyError(name)
            if param.apply == SESSION:
                raise PermissionError(
                    f"{name} is a session parameter — set it in tatbot.yaml or "
                    "via CLI flags and restart the session"
                )
            value = param.clamp(value)
            self.pending[name] = value
            if param.apply == HELD_STILL:
                self.waiting.add(name)
            return {"name": name, "value": value, "queued": True}

    def request_joint(self, name: str, index: int, value: float) -> dict:
        """Edit one element of a per-joint parameter, merging with current."""
        with self.lock:
            param = self.registry.get(name)
            if param is None:
                raise KeyError(name)
            if not param.per_joint:
                raise ValueError(f"{name} is not per-joint")
            if not 0 <= index < param.per_joint:
                raise ValueError(f"{name}: joint index {index} out of range")
            known = (
                self.pending.get(name)
                or self.snapshot.get("values", {}).get(name)
                or (param.get_fn() if param.get_fn is not None else None)
            )
        if known is None:
            raise RuntimeError(
                f"{name} has no known value yet (arm link down?) — "
                "cannot edit a single joint"
            )
        base = list(known)
        base[index] = value
        return self.request(name, base)

    def values(self) -> dict[str, Any]:
        """Current registry values, read on the server thread from the last
        loop-published snapshot (never touches the driver)."""
        with self.lock:
            return dict(self.snapshot.get("values", {}))

    def dirty(self) -> dict[str, Any]:
        vals = self.values()
        out = {}
        for name, gold in self.golden.items():
            cur = vals.get(name)
            if cur is None:
                continue
            if _differs(cur, gold):
                out[name] = {"value": cur, "golden": gold}
        return out

    # -- control-loop side ------------------------------------------------

    def publish(self, arm: str, telemetry: dict) -> None:
        """Called by a plugin once per loop with its telemetry dict."""
        with self.lock:
            telemetry["t"] = time.time()
            self.snapshot.setdefault("arms", {})[arm] = telemetry
            self.snapshot["t"] = telemetry["t"]

    def publish_values(self) -> None:
        """Refresh registry values into the snapshot (loop thread only).

        Driver-side get_fns are guarded (_DriverGuard): cached after one
        success, damped after repeated failure, returning None while
        unknown — so this never raises and never floods the log.
        """
        vals = {}
        for name, p in self.registry.items():
            if p.get_fn is None:
                continue
            try:
                v = p.get_fn()
            except Exception as e:
                logger.error("reading %s failed: %s", name, e)
                continue
            if v is not None:
                vals[name] = v
        with self.lock:
            self.snapshot["values"] = vals

    def apply_pending(self, arm: str, velocities, now: float | None = None) -> None:
        """Drain queued changes for ``arm``'s params. Loop thread only.

        Parameters carrying ``ramp_seconds`` are walked toward their target a
        slice per tick and stay pending until they arrive, so a slider drag
        becomes a smooth approach instead of a torque step.
        """
        now = time.monotonic() if now is None else now
        last = self._last_apply.get(arm)
        self._last_apply[arm] = now
        dt = min(max(now - last, 1e-3), 0.2) if last is not None else 1 / 30
        with self.lock:
            if not self.pending:
                return
            items = [
                (n, v)
                for n, v in self.pending.items()
                if self.registry[n].arm == arm
            ]
        if not items:
            return
        at_rest = velocities is not None and all(
            abs(v) < REST_VELOCITY for v in velocities
        )
        for name, value in items:
            param = self.registry[name]
            if param.apply == HELD_STILL and not at_rest:
                continue  # stays pending; UI shows the gate
            try:
                previous = param.get_fn() if param.get_fn else None
                target, arrived = self._ramped(param, value, dt, previous)
                if param.set_fn is not None:
                    param.set_fn(target)
                if previous is not None and name not in self._ramping:
                    # Remember the pre-change value so an instability within
                    # the next few seconds can put it back (note_instability).
                    self._last_change[arm] = (name, previous, now)
                if not arrived:
                    self._ramping.add(name)
                    continue  # still travelling; stays pending
                self._ramping.discard(name)
                logger.info("tuning: %s = %s", name, target)
            except Exception as e:
                logger.error("applying %s failed: %s", name, e)
                self._ramping.discard(name)
                with self.lock:
                    self.snapshot["last_error"] = f"applying {name} failed: {e}"
            with self.lock:
                self.pending.pop(name, None)
                self.waiting.discard(name)

    def _ramped(self, param: Param, value, dt: float, current):
        """(value_to_write, arrived?) for this tick."""
        if param.ramp_seconds is None or current is None:
            return value, True
        steps = param.ramp_steps(dt)
        if isinstance(steps, (int, float)):
            delta = float(value) - float(current)
            if abs(delta) <= steps:
                return value, True
            return float(current) + math.copysign(steps, delta), False
        out, arrived = [], True
        for i, (c, t) in enumerate(zip(current, value, strict=True)):
            delta = float(t) - float(c)
            if abs(delta) <= steps[i]:
                out.append(float(t))
            else:
                out.append(float(c) + math.copysign(steps[i], delta))
                arrived = False
        return out, arrived

    def note_instability(self, arm: str, reason: str, now: float | None = None,
                         window_s: float = 5.0) -> str | None:
        """Undo the last parameter change on ``arm`` after an instability.

        Raising friction assist (or gravity-comp scaling) past a joint's
        static friction makes the arm self-drive: it accelerates until the
        firmware trips its velocity limit and idles the motor, which kills
        the session. When the caller sees that runaway starting, this puts
        the offending value back and cancels any in-flight ramp toward it.
        Returns the reverted parameter name, or None if nothing recent.
        """
        now = time.monotonic() if now is None else now
        entry = self._last_change.get(arm)
        if entry is None:
            return None
        name, previous, when = entry
        if now - when > window_s:
            return None
        param = self.registry.get(name)
        if param is None or previous is None:
            return None
        self._last_change.pop(arm, None)
        with self.lock:  # cancel the ramp before it resumes next tick
            self.pending.pop(name, None)
            self.waiting.discard(name)
        self._ramping.discard(name)
        try:
            if param.set_fn is not None:
                param.set_fn(previous)
        except Exception as e:
            logger.error("reverting %s failed: %s", name, e)
            return None
        message = f"{reason} — reverted {name} to {previous}"
        logger.warning("INSTABILITY: %s", message)
        with self.lock:
            self.snapshot["last_error"] = message
        return name


def _differs(a, b, tol=1e-6) -> bool:
    if isinstance(a, (list, tuple)):
        return len(a) != len(b) or any(_differs(x, y, tol) for x, y in zip(a, b, strict=True))
    try:
        return abs(float(a) - float(b)) > tol
    except (TypeError, ValueError):
        return a != b


# ---------------------------------------------------------------------------
# Registry construction
# ---------------------------------------------------------------------------


class _DriverGuard:
    """Caching + failure damping for driver-side (TCP config) bindings.

    Config values only change through our own writes, so after one
    successful read the cache serves every later get — no periodic TCP
    polling of the controller (repeated get_motor_parameters/get_joint_limits
    reads were the log flood in the 2026-08-20 follower TCP incident).
    Reads stop being attempted after MAX_FAILS consecutive failures and
    resume after the next successful write; get() returns None while the
    value is unknown instead of raising.
    """

    MAX_FAILS = 3

    def __init__(self, name: str, get_raw, set_raw):
        self.name = name
        self.get_raw = get_raw
        self.set_raw = set_raw
        self.cache = None
        self.fails = 0

    def get(self):
        if self.cache is not None:
            return self.cache
        if self.fails >= self.MAX_FAILS:
            return None
        try:
            self.cache = self.get_raw()
            self.fails = 0
        except Exception as e:
            self.fails += 1
            log = logger.error if self.fails in (1, self.MAX_FAILS) else logger.debug
            gave_up = (" — giving up until a successful write"
                       if self.fails >= self.MAX_FAILS else "")
            log("reading %s failed (%d/%d): %s%s",
                self.name, self.fails, self.MAX_FAILS, e, gave_up)
            return None
        return self.cache

    def set(self, vals):
        self.set_raw(vals)  # raises on failure; apply_pending reports it
        self.cache = [float(v) for v in vals]
        self.fails = 0


def _guarded(name, get_raw, set_raw):
    guard = _DriverGuard(name, get_raw, set_raw)
    return guard.get, guard.set


def _driver_vector_param(driver, name: str, getter: str, setter: str):
    get = getattr(driver, getter)
    set_ = getattr(driver, setter)
    return _guarded(
        name, lambda: [float(v) for v in get()], lambda vals: set_(list(vals))
    )


def _motor_pid_param(driver, trossen_arm_mod, name: str, loop: str, field_name: str):
    """Bind one field of the position-mode PID tables (per joint).

    pybind returns copies on attribute access, so mutate defensively:
    pull the PID struct out, edit it, and push the whole table back.
    """
    mode = trossen_arm_mod.Mode.position

    def get():
        mp = driver.get_motor_parameters()
        return [float(getattr(getattr(mp[j][mode], loop), field_name)) for j in range(len(mp))]

    def set_(vals):
        mp = driver.get_motor_parameters()
        for j, v in enumerate(vals):
            motor = mp[j][mode]
            pid = getattr(motor, loop)
            setattr(pid, field_name, float(v))
            setattr(motor, loop, pid)
            mp[j][mode] = motor
        driver.set_motor_parameters(mp)

    return _guarded(name, get, set_)


def _joint_limit_param(driver, name: str, field_name: str):
    def get():
        return [float(getattr(jl, field_name)) for jl in driver.get_joint_limits()]

    def set_(vals):
        limits = driver.get_joint_limits()
        for jl, v in zip(limits, vals, strict=True):
            setattr(jl, field_name, float(v))
        driver.set_joint_limits(limits)

    return _guarded(name, get, set_)


def _config_param(config, attr: str, post: Callable | None = None):
    def get():
        v = getattr(config, attr)
        return list(v) if isinstance(v, (list, tuple)) else v

    def set_(value):
        setattr(config, attr, value)
        if post is not None:
            post(value)

    return get, set_


def build_leader_params(driver, trossen_arm_mod, config=None) -> list[Param]:
    """Leader feel: in the IL path the leader runs pure gravity comp, so the
    friction/characteristic tables ARE the entire feel of the arm in hand."""

    def p(name, getter, setter, **kw):
        get_fn, set_fn = _driver_vector_param(driver, name, getter, setter)
        return Param(
            name=name, arm="leader", owner=DRIVER, persist="leader.yaml",
            per_joint=7, get_fn=get_fn, set_fn=set_fn, **kw,
        )

    ec_get, ec_set = _driver_vector_param(
        driver, "leader_effort_correction",
        "get_effort_corrections", "set_effort_corrections",
    )
    extra: list[Param] = []
    if config is not None and hasattr(config, "leader_damping"):
        d_get, d_set = _config_param(config, "leader_damping")
        extra.append(Param(
            name="leader_damping", group="leader-feel", label="Damping",
            doc="EXPERIMENTAL. Software viscous damping: effort opposing "
                "joint velocity, the term the firmware's friction model has "
                "no way to provide (all three of its terms push WITH motion). "
                "This is the knob for a leader that sticks then lurches "
                "during slow, fine moves. Raise in small steps — it should "
                "feel progressively heavier and smoother. If it gets MORE "
                "lurchy instead, the sign is inverted: zero it and tell me.",
            units="Nm/(rad/s)", minimum=0.0, maximum=0.5, step=0.01,
            apply=LIVE, owner=PLUGIN, arm="leader", persist="tatbot.yaml",
            ramp_seconds=2.0, get_fn=d_get, set_fn=d_set,
        ))
    return extra + [
        p("leader_friction_constant_term",
          "get_friction_constant_terms", "set_friction_constant_terms",
          group="leader-feel", label="Friction constant",
          doc="Breakout assist per joint. Raise SLOWLY until the joint moves "
              "freely — past that point the joint drives itself and runs away "
              "(there is no damping in gravity comp until viscous is tuned). "
              "Changes ramp in over ~3 s; a runaway auto-reverts this value.",
          units="Nm | N", step=0.01,
          # Arm joints are tuned at 0.04–0.24 Nm, the gripper carriage at 7 N:
          # they need separate ranges or a drag on a wrist joint reaches a
          # value dozens of times past self-drive (the 2026-08-20 J4 trip).
          limits=[(-0.5, 1.0)] * ARM_JOINTS + [(0.0, 12.0)],
          ramp_seconds=3.0),
        p("leader_friction_coulomb_coef",
          "get_friction_coulomb_coefs", "set_friction_coulomb_coefs",
          group="leader-feel", label="Friction coulomb",
          doc="Assist proportional to the joint's gravity/inertial load. "
              "Raise if the joint feels heavier when the arm is stretched "
              "out than when it is balanced, at low speed. Destabilizing — "
              "ramped and auto-reverted.",
          units="Nm/Nm", minimum=0.0, maximum=0.5, step=0.005,
          ramp_seconds=3.0),
        p("leader_friction_viscous_coef",
          "get_friction_viscous_coefs", "set_friction_viscous_coefs",
          group="leader-feel", label="Friction viscous",
          doc="Assist proportional to speed (NOT damping — every friction "
              "term pushes in the direction of motion). Raise only if the "
              "joint gets harder to move the FASTER you move it. "
              "Destabilizing at speed: ramped and auto-reverted.",
          units="Nm/(rad/s)", minimum=0.0, maximum=0.5, step=0.005,
          ramp_seconds=3.0),
        p("leader_friction_transition_velocity",
          "get_friction_transition_velocities", "set_friction_transition_velocities",
          group="leader-feel", label="Friction transition vel",
          doc="Speed at which assist reaches full value (it ramps linearly "
              "from 0 at standstill, which is why breakout feels stickiest). "
              "LOWER = assist engages sooner = less stiction, but risks "
              "oscillation at rest; RAISE for quiet operation at the cost "
              "of more stiction.",
          units="rad/s", minimum=1e-4, maximum=0.2, step=0.001),
        Param(
            name="leader_effort_correction", arm="leader", owner=DRIVER,
            persist="leader.yaml", per_joint=7, apply=HELD_STILL,
            group="leader-feel", label="Effort correction",
            doc="Gravity-comp scaling per joint (firmware range 0.2–5). Raise "
                "a joint's value if the links beyond it sag; too high and the "
                "arm lifts itself. Applied at rest, ramped over ~3 s.",
            minimum=0.2, maximum=5.0, step=0.01, ramp_seconds=3.0,
            get_fn=ec_get, set_fn=ec_set,
        ),
    ]


def build_follower_params(driver, config, trossen_arm_mod, robot=None) -> list[Param]:
    params: list[Param] = []

    # -- firmware stiffness / inner loop (held-still) ----------------------
    for loop, field_name, name, label, doc, mx, step in [
        ("position", "kp", "follower_position_kp", "Position kp",
         "The stiffness table — resistance to skin-drag deflection.", 200.0, 1.0),
        ("velocity", "kp", "follower_velocity_kp", "Velocity kp",
         "Inner-loop damping/tracking. The subtle knob behind hum vs sluggish.",
         20.0, 0.1),
        ("velocity", "ki", "follower_velocity_ki", "Velocity ki",
         "Inner-loop integral gain (position mode).", 0.1, 0.001),
    ]:
        get_fn, set_fn = _motor_pid_param(driver, trossen_arm_mod, name, loop, field_name)
        params.append(Param(
            name=name, group="follower-tracking", label=label, doc=doc,
            per_joint=7, apply=HELD_STILL, owner=DRIVER, arm="follower",
            persist="follower.yaml", minimum=0.0, maximum=mx, step=step,
            ramp_seconds=2.0,  # a stiffness step under load kicks the arm
            get_fn=get_fn, set_fn=set_fn,
        ))

    # -- plugin-side tracking (live) ---------------------------------------
    def set_goal_time(mult):
        if robot is not None:
            robot.min_time_to_move = mult / config.loop_rate

    for attr, name, label, doc, units, mn, mx, step, post in [
        ("min_time_to_move_multiplier", "goal_time_multiplier",
         "Goal-time multiplier",
         "Firmware interpolation horizon = multiplier / loop_rate "
         "(3.0 @ 30 Hz = 100 ms). The dominant smoothness-vs-lag knob.",
         "× loop period", 0.5, 6.0, 0.1, set_goal_time),
        ("target_filter_tau", "target_filter_tau", "Target filter τ",
         "EMA low-pass on requested targets. Keep 0 for recording; "
         "~0.3 for rollouts to average chunk-aggregation tremor.",
         "s", 0.0, 1.0, 0.01, None),
        ("max_joint_velocity", "max_joint_velocity", "Max joint velocity",
         "Slew clamp on sent targets — the guard between chunk-handoff "
         "lunges and the ~9.4 rad/s firmware trip.",
         "rad/s", 0.5, 6.0, 0.1, None),
        ("max_relative_target", "max_relative_target", "Max relative target",
         "Per-step leash of goal vs measured position.",
         "rad", 0.1, 2.0, 0.05, None),
    ]:
        get_fn, set_fn = _config_param(config, attr, post)
        params.append(Param(
            name=name, group="follower-tracking", label=label, doc=doc,
            units=units, minimum=mn, maximum=mx, step=step,
            apply=LIVE, owner=PLUGIN, arm="follower", persist="tatbot.yaml",
            get_fn=get_fn, set_fn=set_fn,
        ))

    # -- motion scale (experimental) ---------------------------------------
    get_fn, set_fn = _config_param(config, "motion_scale")
    params.append(Param(
        name="motion_scale", group="follower-tracking", label="Motion scale",
        doc="EXPERIMENTAL per-joint leader→follower scale (gripper excluded). "
            "1.0 = 1:1. Scaling anchors at the pose where it was last "
            "changed; expect workspace offset until re-centered.",
        per_joint=ARM_JOINTS, minimum=0.25, maximum=1.0, step=0.05,
        apply=LIVE, owner=PLUGIN, arm="follower", persist="tatbot.yaml",
        get_fn=get_fn, set_fn=set_fn,
    ))

    # -- carriage contact axis (live: a cap and a lift, not gains) ----------
    for attr, label, doc, units, mn, mx, step in [
        ("carriage_contact_cap_n", "Contact cap",
         "Pen contact force (carriage external effort) above which the pen "
         "retracts and the arm retreats. A clamp, safe to adjust under load.",
         "N", 2.0, 40.0, 0.5),
        ("carriage_retract_m", "Retract lift",
         "How far the carriage lifts the pen on a trip or e-stop.", "m",
         0.005, 0.040, 0.001),
    ]:
        get_fn, set_fn = _config_param(config, attr)
        params.append(Param(
            name=attr, group="carriage", label=label, doc=doc, units=units,
            minimum=mn, maximum=mx, step=step,
            apply=LIVE, owner=PLUGIN, arm="follower", persist="tatbot.yaml",
            get_fn=get_fn, set_fn=set_fn,
        ))

    # -- safety envelope (held-still, deliberate) --------------------------
    for field_name, label, doc, mx, step in [
        ("velocity_max", "Velocity max",
         "Firmware velocity trip per joint (the 'velocity fault').", 10.0, 0.1),
        ("velocity_tolerance", "Velocity tolerance",
         "Fault margin above velocity_max (0 today; docs/robot.md TODO "
         "suggests 0.2 × velocity_max).", 5.0, 0.1),
        ("position_tolerance", "Position tolerance",
         "Following-error trip threshold.", 1.0, 0.01),
        ("effort_max", "Effort max",
         "Torque/force ceiling per joint.", 250.0, 1.0),
        ("effort_tolerance", "Effort tolerance",
         "Fault margin above effort_max.", 400.0, 1.0),
    ]:
        get_fn, set_fn = _joint_limit_param(driver, f"follower_{field_name}", field_name)
        params.append(Param(
            name=f"follower_{field_name}", group="safety", label=label,
            doc=doc, per_joint=7, apply=HELD_STILL, owner=DRIVER,
            arm="follower", persist="follower.yaml",
            minimum=0.0, maximum=mx, step=step,
            get_fn=get_fn, set_fn=set_fn,
        ))

    return params


def session_info(config) -> dict:
    """Read-only session facts shown in the cockpit's Session & rig panel."""
    return {
        "ip_address": config.ip_address,
        "loop_rate": config.loop_rate,
        "include_velocity": config.include_velocity,
        "include_effort": config.include_effort,
        "include_external_effort": config.include_external_effort,
        "estop_device": config.estop_device,
        "estop_required": config.estop_required,
        "flight_log_dir": config.flight_log_dir,
        "staged_positions": list(config.staged_positions),
    }
