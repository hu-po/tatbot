//! Mock and feature-gated native arm adapters. No hardware CLI is enabled here.
pub mod contact;
pub mod estop;
pub mod guide;
pub mod joint_stream;
pub mod lease;
pub use joint_stream::JointSample;
pub mod motion_guard;
pub mod offline_clock;
pub mod profile;
pub mod recovery;
pub mod tracking_watchdog;
#[cfg(feature = "trossen")]
pub mod trossen;
pub mod worker;
use serde::{Deserialize, Serialize};
use std::{
    path::Path,
    sync::{
        Arc, Mutex,
        atomic::{AtomicI32, Ordering},
    },
};
#[derive(Debug, thiserror::Error)]
#[error("arm refused: {0}")]
pub struct Error(pub String);
pub type Result<T> = std::result::Result<T, Error>;
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub enum Mode {
    Idle,
    Position,
    Fault,
    /// Exactly six rotary joints in zero-commanded external effort with the
    /// carriage position-controlled. Every other mixed vector is `Fault`.
    HandGuiding,
}
impl Mode {
    /// The telemetry code shared with scripts/lib/arm_calibration.py.
    pub fn code(self) -> u8 {
        match self {
            Self::Idle => 0,
            Self::Position => 1,
            Self::Fault => 2,
            Self::HandGuiding => 3,
        }
    }
}
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CarriageMeasured {
    pub position_m: f64,
    /// Last successfully submitted target; unknown before measured re-seed.
    pub target_m: Option<f64>,
    pub effort_n: f64,
}
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Measured {
    pub carriage: Option<CarriageMeasured>,
    pub joints: Vec<f64>,
    pub velocities: Vec<f64>,
    pub efforts: Vec<f64>,
    /// Additional raw SDK channels for the owner's diagnostic sidecar; keep
    /// existing status and publication JSON schemas unchanged.
    #[serde(skip)]
    pub dynamics: Option<JointDynamics>,
    pub mode: Mode,
    pub error: String,
}
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct JointDynamics {
    /// Seven axes: rotary acceleration in rad/s², carriage in m/s².
    pub accelerations: Vec<f64>,
    /// Seven axes: SDK joint effort in Nm, carriage in N.
    pub efforts: Vec<f64>,
    /// Seven axes: SDK compensation effort in Nm, carriage in N.
    pub compensation_efforts: Vec<f64>,
}
#[derive(Debug, Clone)]
pub struct ArmConfig {
    pub joints: usize,
}
#[derive(Debug, Clone)]
pub struct Sample {
    pub joints: Vec<f64>,
    pub dt_s: f64,
    pub contact: f64,
}
/// What the fitted tool does at the work, read from its datasheet's
/// `contact:` key (absent means contact). A contact tool touches the work,
/// so its carriage's effort channel may count as contact and a trip
/// retracts along the carriage; a standoff tool works in free space, never
/// touches, and a trip holds. Neither is a property of the arm: whether the
/// carriage's effort channel is assessable at all is the profile's
/// `carriage_qualified`, and both decide together
/// (`effort_assessable = policy == Contact && carriage_qualified`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ContactPolicy {
    Contact,
    Standoff,
}
impl ContactPolicy {
    /// The datasheet's existing `contact:` key: absent or `true` is a contact
    /// tool, `false` a standoff tool, anything else is refused.
    pub fn from_datasheet(datasheet: &serde_json::Value) -> Result<Self> {
        match datasheet.get("contact") {
            None | Some(serde_json::Value::Null) | Some(serde_json::Value::Bool(true)) => {
                Ok(Self::Contact)
            }
            Some(serde_json::Value::Bool(false)) => Ok(Self::Standoff),
            Some(other) => Err(Error(format!(
                "tool datasheet `contact:` must be a boolean, not {other}"
            ))),
        }
    }
}
/// The refusal every backend returns for a carriage command on an
/// unqualified carriage; it names the key that would qualify it.
pub fn unqualified_carriage(action: &str, role: Option<&str>) -> Error {
    Error(format!(
        "{action} refused: this carriage is not qualified (config/trossen/tatbot.yaml {}.carriage_qualified is false)",
        role.unwrap_or("<role>")
    ))
}
/// The controllers' configured position tolerance (config/trossen/*.yaml):
/// measured feedback may rest this far past a nominal bound while the
/// controller still reports it healthy. Judgements of *measured* joints
/// against the nominal envelope allow it; commanded targets never do.
pub const FEEDBACK_POSITION_TOLERANCE_RAD: f64 = 0.2;
/// What the last `connect` waited for after the controller was configured:
/// whether a live measurement settled inside the backend's patience, and how
/// long that took. A backend reports it; the worker carries it in its status
/// so the session journals it as the `connect_wait` event. A controller that
/// stayed silent is reported here and never commanded blind on its account:
/// the callers' own freshness checks still judge every command.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct ConnectWait {
    /// A live, non-default measurement arrived and settled.
    pub settled: bool,
    /// Seconds from the configured controller to the settled measurement,
    /// or to the end of the backend's patience.
    pub elapsed_s: f64,
    /// The settle the backend requires after configure and the patience it
    /// allows, in seconds, as it applied them.
    pub settle_s: f64,
    pub patience_s: f64,
    /// The backend's connect ordinal (1 for its first), so a reader tells a
    /// new connect's wait from a repeat of the last one.
    pub connect: u64,
}
impl ConnectWait {
    /// The journal's word for the outcome.
    pub fn outcome(&self) -> &'static str {
        if self.settled {
            "settle_reached"
        } else {
            "patience_exhausted"
        }
    }
}
pub trait ArmBackend {
    fn connect(&mut self, cfg: &ArmConfig) -> Result<()>;
    /// The wait the last `connect` made for a live measurement, when this
    /// backend waits for one. `None` for a backend that has none to report.
    fn connect_wait(&self) -> Option<ConnectWait> {
        None
    }
    fn measured(&mut self) -> Result<Measured>;
    /// Seven live axes, with carriage limits in metres. Missing limits refuse.
    fn recovery_limits(&mut self) -> Result<[recovery::PositionLimit; 7]> {
        Err(Error("controller recovery limits unavailable".into()))
    }
    /// Whether measured feedback requires configuration recovery. Native
    /// backends may recognize their live rotary feedback tolerance here;
    /// this never widens the limits used for commanded recovery targets.
    fn recovery_needs_configuration(&mut self, measured: &[f64; 7]) -> Result<bool> {
        recovery::needs_golden(measured, &self.recovery_limits()?)
    }
    fn retract_carriage(&mut self, target_m: f64, goal_time_s: f64) -> Result<()>;
    /// Timed carriage-only move with every rotary target held at its measured
    /// value: an attended calibration returning a tripped carriage to its
    /// datum, or parking it clear before release. Position mode only.
    fn move_carriage(&mut self, _target_m: f64, _goal_time_s: f64) -> Result<()> {
        Err(Error(
            "carriage moves are unsupported by this backend".into(),
        ))
    }
    /// The fitted tool's policy, as the owner configured this backend.
    fn contact_policy(&self) -> ContactPolicy {
        ContactPolicy::Contact
    }
    /// Whether this carriage's effort channel and trip retract are qualified
    /// (`config/trossen/tatbot.yaml <role>.carriage_qualified`). An
    /// unqualified carriage is judged by its deflection screen alone and
    /// refuses `retract_carriage` and `move_carriage`.
    fn carriage_qualified(&self) -> bool {
        true
    }
    fn set_mode(&mut self, mode: Mode) -> Result<()>;
    /// Establish position control at measured state. A native backend may
    /// return a monitored interval for a legal target just inside its nominal
    /// limits. Emergency `hold` stays an immediate measured freeze.
    fn reseed_hold(&mut self, _max_velocity: f64, _envelope: f64) -> Result<std::time::Duration> {
        self.set_mode(Mode::Position)?;
        self.hold()?;
        Ok(std::time::Duration::ZERO)
    }
    /// Terminal release for an attended, supported arm. This removes drive
    /// effort even with the stop asserted; no further command may follow it.
    fn shutdown_idle(&mut self) -> Result<()> {
        self.set_mode(Mode::Idle)
    }
    /// Apply a freshly prepared seven-axis hold over the legacy takeover interval.
    fn takeover(&mut self, _target: &[f64; 7]) -> Result<()> {
        Err(Error(
            "measured takeover is unsupported by this backend".into(),
        ))
    }
    fn move_j(&mut self, target: &[f64], goal_time_s: f64) -> Result<()>;
    fn recovery_move(&mut self, _target: &[f64; 7], _phase: recovery::Phase) -> Result<()> {
        Err(Error(
            "seven-axis recovery move is unsupported by this backend".into(),
        ))
    }
    fn stream(&mut self, sample: &Sample) -> Result<()>;
    fn stream_joint(&mut self, _sample: &JointSample) -> Result<()> {
        Err(Error(
            "seven-axis feed-forward streaming unsupported".into(),
        ))
    }
    /// Measured freeze: submit the current pose as a position target. From
    /// hand guiding this also switches every joint back to position control
    /// (fresh measurement, then modes, then that measured target).
    fn hold(&mut self) -> Result<()>;
    /// Finish a healthy joint stream without re-seeding its carriage target
    /// from encoder noise. Fault and physical-stop paths use measured hold.
    fn finish_joint_stream(&mut self) -> Result<()> {
        self.hold()
    }
    /// Six rotary joints to zero-commanded external effort while the carriage
    /// stays position-controlled at `carriage_m`, the confirmed measured datum.
    /// The caller verifies the mixed mode vector by fresh readback afterwards.
    fn enter_hand_guiding(&mut self, _carriage_m: f64) -> Result<()> {
        Err(Error("hand guiding is unsupported by this backend".into()))
    }
    /// Feedback with controller mode and error queried now rather than from
    /// the periodic status cache: for mode-transition readback.
    fn measured_fresh(&mut self) -> Result<Measured> {
        self.measured()
    }
    /// The end-effector payload the controller compensates, as configured,
    /// for the evidence record. None where the backend has no such notion.
    fn payload_readback(&mut self) -> Result<Option<serde_json::Value>> {
        Ok(None)
    }
    fn clear_error(&mut self) -> Result<()>;
    fn load_config(&mut self, path: &Path) -> Result<()>;
    fn inject_faults(&mut self, _: Faults) -> Result<()> {
        Err(Error("fault injection is mock-only".into()))
    }
}

/// An offline-only world which advances exactly one controller period on
/// request. Native hardware adapters deliberately do not implement this trait.
/// Backend I/O, including advancement and destruction, must have independent
/// bounded wall timeouts: closing a clock cannot cancel an in-flight call.
pub trait OfflineArmBackend: ArmBackend {
    fn backend_id(&self) -> &'static str;
    fn advance_world(&mut self, period: std::time::Duration) -> Result<()>;
}
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct Faults {
    pub trip_at: Option<u64>,
    pub error_after: Option<u64>,
    pub lag_ms: u64,
    pub mode_flip: Option<u64>,
    pub freeze: bool,
    pub out_of_envelope: bool,
    pub carriage_effort_n: Option<f64>,
    pub carriage_deflection_m: Option<f64>,
    pub carriage_stuck: bool,
    /// Hand-guiding only: the six-joint pose a scripted operator moves the
    /// mock toward at a bounded rate. Cleared by leaving hand guiding.
    pub guide_pose: Option<[f64; 6]>,
    /// The controller leaves the rotary modes unchanged on entry: the mixed
    /// mode readback then fails.
    pub guide_entry_refused: bool,
    /// The controller keeps the rotary joints in effort mode on stop: the
    /// position readback then fails.
    pub guide_stop_refused: bool,
    /// Every measurement returns the identical tuple: the freshness screen
    /// must refuse it before it is trusted for a hold.
    pub frozen_feedback: bool,
    /// Hand guiding only: the carriage reads this far from its datum.
    pub guide_carriage_drift_m: Option<f64>,
    /// Hand guiding only: the scripted operator's approach speed in rad/s
    /// (default 0.8, below the profile velocity guard).
    pub guide_speed_rad_s: Option<f64>,
}
pub struct MockArm {
    pub contact_policy: ContactPolicy,
    pub carriage_qualified: bool,
    pub faults: Faults,
    state: Measured,
    tick: u64,
    connected: bool,
    pub commands: u64,
    recovery_limits: Option<[recovery::PositionLimit; 7]>,
    reads: u64,
    /// The mock's measurement is live the moment it connects: its wait
    /// settles at once, so a mock session journals the same event a native
    /// one does, with no time in it.
    connect_wait: Option<ConnectWait>,
    connects: u64,
    /// Every mode transition and motion command the mock was asked for, in
    /// order, shared so a test can read the exact trace after moving the
    /// backend onto its worker thread.
    pub trace: Arc<Mutex<Vec<String>>>,
}
impl Default for MockArm {
    fn default() -> Self {
        Self {
            contact_policy: ContactPolicy::Contact,
            carriage_qualified: true,
            faults: Faults::default(),
            state: Measured {
                carriage: Some(CarriageMeasured {
                    position_m: 0.002,
                    target_m: Some(0.002),
                    effort_n: 0.0,
                }),
                joints: vec![0.0; 6],
                velocities: vec![0.0; 6],
                efforts: vec![0.0; 6],
                dynamics: None,
                mode: Mode::Idle,
                error: String::new(),
            },
            tick: 0,
            connected: false,
            commands: 0,
            recovery_limits: None,
            reads: 0,
            connect_wait: None,
            connects: 0,
            trace: Arc::new(Mutex::new(Vec::new())),
        }
    }
}
impl MockArm {
    /// Cold offline controller with an explicit seven-axis measured seed. No
    /// command or connection is made to initialize these synthetic measurements.
    pub fn recovery_seed(limits: [recovery::PositionLimit; 7], seed: [f64; 7]) -> Result<Self> {
        if seed.iter().any(|q| !q.is_finite()) {
            return Err(Error("non-finite recovery simulation seed".into()));
        }
        let mut arm = Self::with_recovery_limits(limits)?;
        arm.state.joints.copy_from_slice(&seed[..6]);
        let carriage = arm.state.carriage.as_mut().unwrap();
        carriage.position_m = seed[6];
        carriage.target_m = Some(seed[6]);
        Ok(arm)
    }
    /// Cold mock controller resting at an explicit seven-axis pose inside a
    /// symmetric joint envelope: a reproducible fixture for offline planning,
    /// never measured feedback or a hardware limit.
    pub fn seeded(seed: [f64; 7], envelope_rad: f64) -> Result<Self> {
        if seed.iter().any(|q| !q.is_finite())
            || !envelope_rad.is_finite()
            || envelope_rad <= 0.0
            || seed[..6].iter().any(|q| q.abs() > envelope_rad)
        {
            return Err(Error("non-finite or out-of-envelope mock seed".into()));
        }
        let mut limits = [recovery::PositionLimit {
            min: -envelope_rad,
            max: envelope_rad,
        }; 7];
        limits[6] = recovery::PositionLimit {
            min: -0.006,
            max: 0.04,
        };
        let mut arm = Self::with_recovery_limits(limits)?;
        arm.state.joints.copy_from_slice(&seed[..6]);
        let carriage = arm.state.carriage.as_mut().unwrap();
        carriage.position_m = seed[6];
        carriage.target_m = Some(seed[6]);
        Ok(arm)
    }
    /// Explicit controller model for offline profile integration. Defaults are
    /// unchanged; these values never configure a physical controller.
    pub fn with_recovery_limits(limits: [recovery::PositionLimit; 7]) -> Result<Self> {
        recovery::needs_golden(&[0.0; 7], &limits)?;
        Ok(Self {
            recovery_limits: Some(limits),
            ..Self::default()
        })
    }
}
impl MockArm {
    fn record(&self, entry: impl Into<String>) {
        self.trace.lock().unwrap().push(entry.into());
    }
    /// A scripted operator: the rotary joints approach `guide_pose` at
    /// 0.8 rad/s (below the profile velocity guard) while a tiny
    /// deterministic effort ripple keeps the feedback tuple changing, as a
    /// live controller's does. `frozen_feedback` suppresses the ripple.
    fn advance_guided(&mut self) {
        self.reads += 1;
        let step_rad = self.faults.guide_speed_rad_s.unwrap_or(0.8) * 0.00125;
        let target = self.faults.guide_pose;
        for (index, q) in self.state.joints.iter_mut().enumerate() {
            let delta = target.map_or(0.0, |pose| (pose[index] - *q).clamp(-step_rad, step_rad));
            *q += delta;
            self.state.velocities[index] = delta / 0.00125;
            self.state.efforts[index] = if self.faults.frozen_feedback {
                0.0
            } else {
                0.01 * ((self.reads % 7) as f64 - 3.0)
            };
        }
    }
}
impl ArmBackend for MockArm {
    fn contact_policy(&self) -> ContactPolicy {
        self.contact_policy
    }
    fn carriage_qualified(&self) -> bool {
        self.carriage_qualified
    }
    fn enter_hand_guiding(&mut self, carriage_m: f64) -> Result<()> {
        if !self.connected || self.state.mode != Mode::Position || !self.state.error.is_empty() {
            return Err(Error(
                "hand guiding needs a healthy position-mode controller".into(),
            ));
        }
        if !carriage_m.is_finite() {
            return Err(Error("carriage datum must be finite".into()));
        }
        let carriage = self
            .state
            .carriage
            .as_mut()
            .ok_or_else(|| Error("carriage missing".into()))?;
        carriage.target_m = Some(carriage_m);
        self.record(format!("enter_hand_guiding({carriage_m})"));
        if !self.faults.guide_entry_refused {
            self.state.mode = Mode::HandGuiding;
        }
        Ok(())
    }
    fn payload_readback(&mut self) -> Result<Option<serde_json::Value>> {
        Ok(Some(
            serde_json::json!({"backend": "mock", "palm_mass_kg": 0.0}),
        ))
    }
    fn connect(&mut self, cfg: &ArmConfig) -> Result<()> {
        if cfg.joints != self.state.joints.len() {
            return Err(Error("joint count".into()));
        }
        self.connected = true;
        self.connects += 1;
        self.connect_wait = Some(ConnectWait {
            settled: true,
            elapsed_s: 0.0,
            settle_s: 0.0,
            patience_s: 0.0,
            connect: self.connects,
        });
        Ok(())
    }
    fn connect_wait(&self) -> Option<ConnectWait> {
        self.connect_wait
    }
    fn recovery_limits(&mut self) -> Result<[recovery::PositionLimit; 7]> {
        if let Some(limits) = self.recovery_limits {
            return Ok(limits);
        }
        // Synthetic limits for the mock fixture only, not a hardware envelope.
        let mut limits = [recovery::PositionLimit {
            min: -1.0,
            max: 1.0,
        }; 7];
        limits[6] = recovery::PositionLimit {
            min: -0.006,
            max: 0.04,
        };
        Ok(limits)
    }
    fn measured(&mut self) -> Result<Measured> {
        if !self.connected {
            return Err(Error("not connected".into()));
        }
        if self.state.mode == Mode::HandGuiding {
            self.advance_guided();
        }
        let mut m = self.state.clone();
        if let Some(carriage) = &mut m.carriage {
            if self.state.mode == Mode::HandGuiding
                && let Some(drift) = self.faults.guide_carriage_drift_m
            {
                carriage.position_m += drift;
            }
            if let Some(effort) = self.faults.carriage_effort_n {
                carriage.effort_n = effort;
            }
            if let Some(deflection) = self.faults.carriage_deflection_m
                && let Some(target) = carriage.target_m
                && target < 0.032
            {
                carriage.position_m = target + deflection;
            }
        }
        if self.faults.out_of_envelope {
            m.joints[0] = 100.0;
        }
        if self.faults.error_after.is_some_and(|t| self.tick >= t) {
            m.error = "injected controller error".into();
        }
        if self.faults.mode_flip.is_some_and(|t| self.tick >= t) {
            m.mode = Mode::Fault;
        }
        Ok(m)
    }
    fn retract_carriage(&mut self, target_m: f64, goal_time_s: f64) -> Result<()> {
        if !self.carriage_qualified {
            return Err(unqualified_carriage("trip retract", None));
        }
        if target_m != 0.032 || goal_time_s != 0.6 {
            return Err(Error("unqualified mock retract".into()));
        }
        self.record(format!("retract_carriage({target_m})"));
        let carriage = self
            .state
            .carriage
            .as_mut()
            .ok_or_else(|| Error("carriage absent".into()))?;
        carriage.target_m = Some(target_m);
        if !self.faults.carriage_stuck {
            carriage.position_m = target_m;
        }
        Ok(())
    }
    fn move_carriage(&mut self, target_m: f64, goal_time_s: f64) -> Result<()> {
        if !self.carriage_qualified {
            return Err(unqualified_carriage("carriage move", None));
        }
        if !self.connected {
            return Err(Error("not connected".into()));
        }
        if self.state.mode != Mode::Position || !self.state.error.is_empty() {
            return Err(Error(
                "carriage move needs a healthy position-mode arm".into(),
            ));
        }
        if !(1.0..=5.0).contains(&goal_time_s) || !target_m.is_finite() {
            return Err(Error("unqualified mock carriage move".into()));
        }
        self.record(format!("move_carriage({target_m})"));
        let carriage = self
            .state
            .carriage
            .as_mut()
            .ok_or_else(|| Error("carriage absent".into()))?;
        carriage.target_m = Some(target_m);
        if !self.faults.carriage_stuck {
            carriage.position_m = target_m;
        }
        Ok(())
    }
    fn set_mode(&mut self, m: Mode) -> Result<()> {
        if !self.connected {
            return Err(Error("not connected".into()));
        }
        self.record(format!("set_mode({m:?})"));
        self.state.mode = m;
        if m != Mode::HandGuiding {
            self.faults.guide_pose = None;
        }
        Ok(())
    }
    fn move_j(&mut self, target: &[f64], goal_time_s: f64) -> Result<()> {
        self.stream(&Sample {
            joints: target.to_vec(),
            dt_s: goal_time_s,
            contact: 0.0,
        })
    }
    fn recovery_move(&mut self, target: &[f64; 7], phase: recovery::Phase) -> Result<()> {
        if recovery::needs_golden(target, &self.recovery_limits()?)? {
            return Err(Error("recovery move target outside limits".into()));
        }
        self.move_j(&target[..6], phase.duration())?;
        let carriage = self
            .state
            .carriage
            .as_mut()
            .ok_or_else(|| Error("carriage missing".into()))?;
        carriage.target_m = Some(target[6]);
        if !self.faults.carriage_stuck {
            carriage.position_m = target[6];
        }
        Ok(())
    }
    fn takeover(&mut self, target: &[f64; 7]) -> Result<()> {
        let limits = self.recovery_limits()?;
        if recovery::needs_golden(target, &limits)? {
            return Err(Error("takeover target outside limits".into()));
        }
        self.set_mode(Mode::Position)?;
        self.move_j(&target[..6], recovery::TAKEOVER_S)?;
        let carriage = self
            .state
            .carriage
            .as_mut()
            .ok_or_else(|| Error("carriage missing".into()))?;
        carriage.position_m = target[6];
        carriage.target_m = Some(target[6]);
        Ok(())
    }
    fn stream(&mut self, s: &Sample) -> Result<()> {
        let m = self.measured()?;
        if m.mode != Mode::Position || !m.error.is_empty() {
            return Err(Error("controller unhealthy".into()));
        }
        if s.joints.len() != m.joints.len()
            || !s.dt_s.is_finite()
            || s.dt_s <= 0.0
            || s.joints.iter().any(|v| !v.is_finite())
        {
            return Err(Error("invalid sample".into()));
        }
        if self.faults.trip_at == Some(self.tick) {
            self.faults.trip_at = None;
            return Err(Error("injected contact trip".into()));
        }
        self.tick += 1;
        self.commands += 1;
        self.record("stream");
        let fraction = if self.faults.freeze {
            0.0
        } else {
            s.dt_s / (s.dt_s + self.faults.lag_ms as f64 / 1000.0)
        };
        for ((q, v), target) in self
            .state
            .joints
            .iter_mut()
            .zip(&mut self.state.velocities)
            .zip(&s.joints)
        {
            let delta = (target - *q) * fraction;
            *q += delta;
            *v = delta / s.dt_s;
        }
        Ok(())
    }
    fn stream_joint(&mut self, s: &JointSample) -> Result<()> {
        self.stream(&s.angular())?;
        let carriage = self
            .state
            .carriage
            .as_mut()
            .ok_or_else(|| Error("missing carriage".into()))?;
        carriage.target_m = Some(s.positions[6]);
        if !self.faults.freeze && !self.faults.carriage_stuck {
            let fraction = s.dt_s / (s.dt_s + self.faults.lag_ms as f64 / 1000.0);
            carriage.position_m += (s.positions[6] - carriage.position_m) * fraction;
        }
        Ok(())
    }
    fn hold(&mut self) -> Result<()> {
        if !self.connected {
            return Err(Error("not connected".into()));
        }
        if self.state.mode == Mode::HandGuiding {
            // Modes first, then the measured target: the qualified ordering.
            self.record("hold_from_hand_guiding(set_mode(Position), positions(measured))");
            self.faults.guide_pose = None;
            if !self.faults.guide_stop_refused {
                self.state.mode = Mode::Position;
            }
        } else {
            self.record("hold");
        }
        self.state.velocities.fill(0.0);
        if let Some(carriage) = &mut self.state.carriage {
            carriage.target_m = Some(carriage.position_m);
        }
        Ok(())
    }
    fn finish_joint_stream(&mut self) -> Result<()> {
        self.state.velocities.fill(0.0);
        Ok(())
    }
    fn clear_error(&mut self) -> Result<()> {
        self.faults.error_after = None;
        self.faults.mode_flip = None;
        self.state.error.clear();
        Ok(())
    }
    fn inject_faults(&mut self, faults: Faults) -> Result<()> {
        self.faults = faults;
        Ok(())
    }
    fn load_config(&mut self, _: &Path) -> Result<()> {
        Err(Error("mock cannot load hardware configuration".into()))
    }
}
/// How far a commanded joint target may lead the measured joint before the
/// stream is refused as out of sync with the arm. A position controller
/// follows a stream with lag: at 0.085 rad/s the follower measured 6 mrad
/// behind its target, which the previous rule (one tick of the recovery
/// velocity cap, 5 mrad) refused the moment the control loop ran at real
/// time. At the planner's 1.0 rad/s pen-up cap that lag is about 70 mrad;
/// a stalled controller is the tracking watchdog's finding (0.35 rad for
/// 2 s), not this bound's. Commanded step sizes are bounded by the planner
/// and the stream validation, not here. This bound only has to catch a
/// stream that lost its arm entirely, so it is generous.
pub const TRACKING_ERROR_LIMIT_RAD: f64 = 0.3;

/// Independent of FSM, bus, journal and viewer. Every command reads the monitor atomic.
pub struct Control<B: ArmBackend> {
    pub backend: B,
    pub estop: Arc<AtomicI32>,
    pub max_velocity: f64,
    /// Bound on how far a target may lead the measured joint; production
    /// uses TRACKING_ERROR_LIMIT_RAD, tests of the slower watchdogs loosen it.
    pub max_tracking_error: f64,
    pub max_contact: f64,
    pub envelope: f64,
}
impl<B: ArmBackend> Control<B> {
    fn check_sample(&mut self, s: &Sample) -> Result<Measured> {
        if self.estop.load(Ordering::SeqCst) != estop::OK {
            self.backend.hold()?;
            return Err(Error("e-stop".into()));
        }
        let m = self.backend.measured()?;
        let valid = s.joints.len() == m.joints.len()
            && s.dt_s.is_finite()
            && s.dt_s > 0.0
            && self.max_velocity.is_finite()
            && self.max_velocity > 0.0
            && self.max_tracking_error.is_finite()
            && self.max_tracking_error > 0.0
            && self.max_contact.is_finite()
            && self.max_contact > 0.0
            && self.envelope.is_finite()
            && self.envelope > 0.0
            && s.contact.is_finite()
            && s.contact >= 0.0
            && s.contact <= self.max_contact
            && m.error.is_empty()
            && m.mode == Mode::Position
            && m.carriage
                .as_ref()
                .is_none_or(|carriage| carriage.target_m.is_some_and(f64::is_finite))
            && s.joints.iter().zip(&m.joints).all(|(q, old)| {
                q.is_finite()
                    && old.is_finite()
                    && q.abs() <= self.envelope
                    && old.abs() <= self.envelope + FEEDBACK_POSITION_TOLERANCE_RAD
                    && (q - old).abs() <= self.max_tracking_error
            });
        if !valid {
            self.backend.hold()?;
            return Err(Error(
                "control invariant; retract requires qualified primitive".into(),
            ));
        }
        // Recheck after measurements. There is no bus or disk operation in this path.
        if self.estop.load(Ordering::SeqCst) != estop::OK {
            self.backend.hold()?;
            return Err(Error("e-stop".into()));
        }
        Ok(m)
    }
    pub fn stream(&mut self, s: &Sample) -> Result<()> {
        self.check_sample(s)?;
        self.backend.stream(s)
    }
    pub fn stream_joint(&mut self, s: &JointSample) -> Result<()> {
        self.stream_joint_mode(s, false)
    }
    pub(crate) fn stream_carriage(&mut self, s: &JointSample) -> Result<()> {
        self.stream_joint_mode(s, true)
    }
    fn stream_joint_mode(&mut self, s: &JointSample, transfer: bool) -> Result<()> {
        let measured = self.check_sample(&s.angular())?;
        let mut previous = s.clone();
        previous.positions[6] = measured
            .carriage
            .as_ref()
            .and_then(|c| c.target_m)
            .unwrap_or(f64::NAN);
        let checked = if transfer {
            s.validate_transfer(Some(&previous))
        } else {
            s.validate(Some(&previous))
        };
        let valid = checked.is_ok()
            && measured
                .carriage
                .as_ref()
                .is_some_and(|c| c.position_m.is_finite())
            && s.velocities.iter().all(|v| v.is_finite())
            && s.positions.iter().all(|q| q.is_finite())
            && s.velocities[..6]
                .iter()
                .all(|v| v.abs() <= self.max_velocity);
        if !valid {
            self.backend.hold()?;
            return Err(Error("seven-axis control invariant".into()));
        }
        if self.estop.load(Ordering::SeqCst) != estop::OK {
            self.backend.hold()?;
            return Err(Error("e-stop".into()));
        }
        self.backend.stream_joint(s)
    }
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn an_unknown_carriage_command_requires_reseed_before_streaming() {
        let mut arm = MockArm::default();
        arm.connect(&ArmConfig { joints: 6 }).unwrap();
        arm.set_mode(Mode::Position).unwrap();
        arm.state.carriage.as_mut().unwrap().target_m = None;
        let mut control = Control {
            backend: arm,
            estop: Arc::new(AtomicI32::new(estop::OK)),
            max_velocity: 1.0,
            max_tracking_error: TRACKING_ERROR_LIMIT_RAD,
            max_contact: 1.0,
            envelope: 1.0,
        };
        assert!(
            control
                .stream(&Sample {
                    joints: vec![0.0; 6],
                    dt_s: 0.0025,
                    contact: 0.0
                })
                .is_err()
        );
        assert_eq!(control.backend.commands, 0);
        // The refusal issued a measured hold, rather than pretending the old
        // target was equal to feedback before any command was submitted.
        assert_eq!(
            control
                .backend
                .measured()
                .unwrap()
                .carriage
                .unwrap()
                .target_m,
            Some(0.002)
        );
    }
    #[test]
    fn seeded_lag_freeze_mode_and_error_faults_cannot_escape_control_bounds() {
        for seed in 1..65_u64 {
            for kind in 0..5 {
                let mut arm = MockArm::default();
                arm.connect(&ArmConfig { joints: 6 }).unwrap();
                arm.set_mode(Mode::Position).unwrap();
                arm.faults = Faults {
                    lag_ms: seed % 8,
                    freeze: kind == 0,
                    mode_flip: (kind == 1).then_some(seed % 20),
                    error_after: (kind == 2).then_some(seed % 20),
                    out_of_envelope: kind == 3,
                    ..Default::default()
                };
                let mut c = Control {
                    backend: arm,
                    estop: Arc::new(AtomicI32::new(estop::OK)),
                    max_velocity: 1.0,
                    max_tracking_error: TRACKING_ERROR_LIMIT_RAD,
                    max_contact: 1.0,
                    envelope: 1.0,
                };
                let mut refused = false;
                // A frozen arm is refused once its target leads the
                // measurement by TRACKING_ERROR_LIMIT_RAD (tick 301 here).
                for tick in 1..=350 {
                    let before = c.backend.commands;
                    if c.stream(&Sample {
                        joints: vec![tick as f64 * 0.001; 6],
                        dt_s: 0.0025,
                        contact: 0.0,
                    })
                    .is_err()
                    {
                        assert_eq!(c.backend.commands, before);
                        refused = true;
                        break;
                    }
                    assert!(
                        c.backend
                            .state
                            .joints
                            .iter()
                            .all(|q| q.is_finite() && q.abs() <= 1.0)
                    );
                }
                if kind < 4 {
                    assert!(refused, "fault {kind}, seed {seed}");
                }
            }
        }
    }
    #[test]
    fn control_refuses_estop_nonfinite_velocity_contact_and_envelope() {
        let mut arm = MockArm::default();
        arm.connect(&ArmConfig { joints: 6 }).unwrap();
        arm.set_mode(Mode::Position).unwrap();
        let mut c = Control {
            backend: arm,
            estop: Arc::new(AtomicI32::new(estop::FAULT)),
            max_velocity: 1.0,
            max_tracking_error: TRACKING_ERROR_LIMIT_RAD,
            max_contact: 1.0,
            envelope: 1.0,
        };
        let s = Sample {
            joints: vec![0.01; 6],
            dt_s: 0.1,
            contact: 0.0,
        };
        assert!(c.stream(&s).is_err());
        assert_eq!(c.backend.commands, 0);
        c.estop.store(estop::OK, Ordering::SeqCst);
        assert!(c.stream(&s).is_ok());
        for bad in [
            Sample {
                contact: 2.0,
                ..s.clone()
            },
            Sample {
                joints: vec![2.0; 6],
                ..s.clone()
            },
            Sample {
                dt_s: f64::NAN,
                ..s.clone()
            },
            Sample {
                joints: vec![0.5; 6],
                ..s.clone()
            },
        ] {
            assert!(c.stream(&bad).is_err());
        }
        assert_eq!(c.backend.commands, 1);
    }
}
