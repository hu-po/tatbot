//! Native seven-axis SDK adapter. Construct with Worker::spawn_with so the
//! vendor object never crosses threads. This feature does not enable CLI motion.
use crate::{
    ArmBackend, ArmConfig, CarriageMeasured, ConnectWait, Error, JointDynamics, Measured, Mode,
    Result, Sample, estop, lease::HardwareLease, recovery,
};
use std::{
    net::Ipv4Addr,
    path::{Path, PathBuf},
    sync::{
        Arc,
        atomic::{AtomicI32, Ordering},
    },
};
use trossen_arm_sys::ffi;

#[derive(Clone, Copy, Debug)]
pub enum Role {
    Leader,
    Follower,
}
#[derive(Clone, Debug)]
pub struct Config {
    pub address: Ipv4Addr,
    pub role: Role,
    pub golden: PathBuf,
    /// The fitted tool's policy from its datasheet (`ContactPolicy`).
    pub contact: crate::ContactPolicy,
    /// `config/trossen/tatbot.yaml <role>.carriage_qualified`.
    pub carriage_qualified: bool,
}
impl Role {
    fn name(self) -> &'static str {
        match self {
            Self::Leader => "leader",
            Self::Follower => "follower",
        }
    }
}

pub struct TrossenArm {
    // Drop the vendor object before releasing the shared process lease.
    driver: Option<trossen_arm_sys::DriverPtr>,
    config: Config,
    limits: Vec<ffi::JointLimit>,
    carriage_target: Option<f64>,
    stream_target: Option<[f64; 7]>,
    /// The rotary joints were commanded into external effort and no hold,
    /// idle or reconnect has taken them out since.
    guiding: bool,
    stop: Arc<AtomicI32>,
    _lease: Arc<HardwareLease>,
    /// What the last `connect` waited for after `configure()`.
    connect_wait: Option<ConnectWait>,
    connects: u64,
}
/// Exactly the hand-guiding mode vector: six external-effort joints, carriage in position.
const HAND_GUIDING_MODES: [u8; 7] = [3, 3, 3, 3, 3, 3, 1];

fn sdk<T>(value: std::result::Result<T, impl std::fmt::Display>) -> Result<T> {
    value.map_err(|error| Error(format!("Trossen SDK: {error}")))
}
/// Connection-time diagnostics only; no extra feedback query or control-loop log.
/// A failed stderr write cannot prevent the existing hold or its error path.
fn reseed_marker(stage: &str, started: std::time::Instant, detail: serde_json::Value) {
    use std::io::Write;
    let wall_ns = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map_or(0, |d| u64::try_from(d.as_nanos()).unwrap_or(u64::MAX));
    let _ = writeln!(
        std::io::stderr().lock(),
        "native-reseed {}",
        serde_json::json!({
            "schema": "tatbot.receipt/1", "kind": "native-reseed", "stage": stage,
            "wall_ns": wall_ns, "elapsed_ns": started.elapsed().as_nanos(),
            "detail": detail,
        })
    );
}
fn checked_measurement(raw: ffi::Measurement, target: Option<f64>) -> Result<Measured> {
    for values in [
        &raw.positions,
        &raw.velocities,
        &raw.accelerations,
        &raw.efforts,
        &raw.external_efforts,
        &raw.compensation_efforts,
    ] {
        if values.len() != 7 || values.iter().any(|q| !q.is_finite()) {
            return Err(Error(
                "native measurement requires seven finite axes".into(),
            ));
        }
    }
    if raw.modes.len() != 7 || target.is_some_and(|q| !q.is_finite()) {
        return Err(Error("native mode/target observation is invalid".into()));
    }
    let mode = if raw.modes.iter().all(|m| *m == 0) {
        Mode::Idle
    } else if raw.modes.iter().all(|m| *m == 1) {
        Mode::Position
    } else if raw.modes == HAND_GUIDING_MODES {
        Mode::HandGuiding
    } else {
        Mode::Fault
    };
    Ok(Measured {
        joints: raw.positions[..6].to_vec(),
        velocities: raw.velocities[..6].to_vec(),
        efforts: raw.external_efforts[..6].to_vec(),
        dynamics: Some(JointDynamics {
            accelerations: raw.accelerations,
            efforts: raw.efforts,
            compensation_efforts: raw.compensation_efforts,
        }),
        carriage: Some(CarriageMeasured {
            position_m: raw.positions[6],
            target_m: target,
            effort_n: raw.external_efforts[6],
        }),
        mode,
        error: recovery::controller_error(&raw.error).to_owned(),
    })
}
/// Rotary targets for a carriage-only move are the measured pose clamped
/// into the nominal bounds, as takeover and the landing routine hold it:
/// feedback may rest a few millirad past a bound inside the controller's
/// tolerance, and a nominal target must not be refused for that.
fn clamp_rotary_into_limits(q: &mut [f64], limits: &[ffi::JointLimit]) -> Result<()> {
    if limits.len() != 7 || q.len() != 7 {
        return Err(Error("live controller limits unavailable".into()));
    }
    for (value, limit) in q[..6].iter_mut().zip(limits) {
        if !limit.position_min.is_finite()
            || !limit.position_max.is_finite()
            || limit.position_min >= limit.position_max
        {
            return Err(Error("invalid live controller limits".into()));
        }
        *value = value.clamp(
            limit.position_min + recovery::LIMIT_MARGIN,
            limit.position_max - recovery::LIMIT_MARGIN,
        );
    }
    Ok(())
}
fn checked_target(q: &[f64], limits: &[ffi::JointLimit]) -> Result<()> {
    checked_positions(q, limits, false)
}
/// Rotary feedback may occupy the controller's configured tolerance band.
/// The carriage is a fixed calibration datum and remains a nominal target.
fn checked_hold_pose(q: &[f64], limits: &[ffi::JointLimit]) -> Result<()> {
    checked_positions(q, limits, true)
}
/// Legal startup hold, with the measured carriage unchanged. Clipping
/// tolerated rotary feedback is a timed correction, never an immediate seed.
fn reseed_target(
    q: &[f64],
    limits: &[ffi::JointLimit],
    max_velocity: f64,
    envelope: f64,
) -> Result<Vec<f64>> {
    checked_hold_pose(q, limits)?;
    let mut target = q.to_vec();
    clamp_rotary_into_limits(&mut target, limits)?;
    if !max_velocity.is_finite()
        || max_velocity <= 0.0
        || !envelope.is_finite()
        || envelope <= 0.0
        || target[..6].iter().zip(&q[..6]).any(|(target, measured)| {
            target.abs() > envelope
                || 2.0 * (target - measured).abs() / recovery::TAKEOVER_S > max_velocity
        })
    {
        return Err(Error(
            "measured re-seed violates control envelope/velocity".into(),
        ));
    }
    checked_target(&target, limits)?;
    Ok(target)
}

fn position_mode_needed(measured: &Measured) -> Result<bool> {
    if !measured.error.is_empty() {
        return Err(Error(measured.error.clone()));
    }
    Ok(measured.mode != Mode::Position)
}
/// Match the rotary feedback band already accepted by measured takeover.
/// Reapplying configuration disconnects the controller, so resting encoder
/// feedback inside that band must not trigger it at every recovery increment.
/// Carriage feedback retains nominal bounds; targets use the original limits.
fn feedback_needs_configuration(q: &[f64; 7], limits: &[ffi::JointLimit]) -> Result<bool> {
    if limits.len() != 7 {
        return Err(Error("native recovery requires seven live limits".into()));
    }
    let nominal = std::array::from_fn(|i| recovery::PositionLimit {
        min: limits[i].position_min,
        max: limits[i].position_max,
    });
    recovery::needs_golden(q, &nominal)?; // validate finite feedback and nominal bounds
    let mut feedback = nominal;
    for (i, limit) in limits.iter().enumerate() {
        if !limit.position_tolerance.is_finite() || limit.position_tolerance < 0.0 {
            return Err(Error("invalid native feedback tolerance".into()));
        }
        if i < 6 {
            feedback[i].min -= limit.position_tolerance;
            feedback[i].max += limit.position_tolerance;
        }
    }
    recovery::needs_golden(q, &feedback)
}
fn checked_positions(q: &[f64], limits: &[ffi::JointLimit], measured: bool) -> Result<()> {
    if q.len() != 7 || limits.len() != 7 {
        return Err(Error(format!(
            "native target requires seven axes and live limits; got {} axes and {} limits",
            q.len(),
            limits.len()
        )));
    }
    for (index, (q, limit)) in q.iter().zip(limits).enumerate() {
        let tolerance = if measured && index < 6 {
            limit.position_tolerance
        } else {
            0.0
        };
        let min = limit.position_min - tolerance;
        let max = limit.position_max + tolerance;
        if !q.is_finite()
            || !limit.position_min.is_finite()
            || !limit.position_max.is_finite()
            || limit.position_min >= limit.position_max
            || !tolerance.is_finite()
            || tolerance < 0.0
            || !min.is_finite()
            || !max.is_finite()
            || *q < min
            || *q > max
        {
            let unit = if index == 6 { "m (carriage)" } else { "rad" };
            let kind = if measured {
                "measured hold pose"
            } else {
                "target"
            };
            return Err(Error(format!(
                "native {kind} outside live controller limits: joint {} (index {index}), target {q:.9}, range [{min:.9}, {max:.9}] {unit} (position tolerance {tolerance:.9})",
                index + 1
            )));
        }
    }
    Ok(())
}
/// Controller mode and error are polled over TCP at most this often from the
/// control loop; a fault the firmware has already acted on is seen within it.
const STATUS_MAX_AGE_S: f64 = 0.1;
/// After `configure()` the controller has delivered no robot output yet; the
/// landing tool (`arm_recover.cpp`) waits for a live measurement before its
/// first command and never lost a session, while the daemon's profile load
/// 14 ms after configure had the follower close the TCP session twice on
/// 2026-09-17. Same wait, same numbers.
const MEASUREMENT_SETTLE_S: f64 = 0.1;
const MEASUREMENT_WAIT_S: f64 = 1.0;

/// A sampled trajectory already owns its interpolation. Keep its interval for
/// the independent lead/velocity checks, but submit the target immediately to
/// the SDK, as the drawing executor does. Timed recovery retains SDK interpolation.
enum CommandTiming {
    Timed(f64),
    Sample(f64),
}
impl CommandTiming {
    fn interval(&self) -> f64 {
        match *self {
            Self::Timed(seconds) | Self::Sample(seconds) => seconds,
        }
    }
    fn sdk_goal_time(&self) -> f64 {
        match *self {
            Self::Timed(seconds) => seconds,
            Self::Sample(_) => 0.0,
        }
    }
}

fn measured_axes(m: &Measured) -> Result<Vec<f64>> {
    let carriage = m
        .carriage
        .as_ref()
        .ok_or_else(|| Error("native carriage missing".into()))?;
    let mut q = m.joints.clone();
    q.push(carriage.position_m);
    Ok(q)
}
impl TrossenArm {
    /// No SDK object or network connection is created here. The physical
    /// monitor and the real hardware lease must outlive this backend.
    pub fn new(
        mut config: Config,
        stop: Arc<AtomicI32>,
        lease: Arc<HardwareLease>,
    ) -> Result<Self> {
        config.golden = config
            .golden
            .canonicalize()
            .map_err(|e| Error(format!("golden: {e}")))?;
        if !config.golden.is_file() || config.golden.to_str().is_none() {
            return Err(Error("golden must be a UTF-8 regular file".into()));
        }
        Ok(Self {
            driver: None,
            config,
            limits: Vec::new(),
            carriage_target: None,
            stream_target: None,
            guiding: false,
            stop,
            _lease: lease,
            connect_wait: None,
            connects: 0,
        })
    }
    fn released(&self) -> Result<()> {
        if self.stop.load(Ordering::SeqCst) != estop::OK {
            return Err(Error("physical e-stop prevents native command".into()));
        }
        Ok(())
    }
    fn driver(&mut self) -> Result<std::pin::Pin<&mut ffi::Driver>> {
        self.driver
            .as_mut()
            .map(|d| d.pin_mut())
            .ok_or_else(|| Error("native driver not connected".into()))
    }
    /// Feedback with the controller's mode and error queried now: for the
    /// health checks that precede a mode change or follow a fault clear.
    fn measured_fresh(&mut self) -> Result<Measured> {
        let raw = sdk(self.driver()?.measure(0.0))?;
        checked_measurement(raw, self.carriage_target)
    }
    fn refresh_limits(&mut self) -> Result<()> {
        self.limits = sdk(self.driver()?.limits())?;
        Ok(())
    }
    /// A non-default measurement at least MEASUREMENT_SETTLE_S after
    /// configure, or MEASUREMENT_WAIT_S of patience; a controller that stays
    /// silent is reported (`connect_wait`, journaled by the session) and left
    /// to the callers' own freshness checks.
    fn await_live_measurement(&mut self) -> ConnectWait {
        let started = std::time::Instant::now();
        self.connects += 1;
        let connect = self.connects;
        let wait = loop {
            let live = self
                .measured_fresh()
                .is_ok_and(|m| m.joints.iter().any(|q| *q != 0.0));
            let elapsed = started.elapsed().as_secs_f64();
            if live && elapsed >= MEASUREMENT_SETTLE_S {
                break ConnectWait {
                    settled: true,
                    elapsed_s: elapsed,
                    settle_s: MEASUREMENT_SETTLE_S,
                    patience_s: MEASUREMENT_WAIT_S,
                    connect,
                };
            }
            if elapsed >= MEASUREMENT_WAIT_S {
                eprintln!(
                    "native connect: no live measurement within {MEASUREMENT_WAIT_S} s of configure; proceeding with the controller's report"
                );
                break ConnectWait {
                    settled: false,
                    elapsed_s: elapsed,
                    settle_s: MEASUREMENT_SETTLE_S,
                    patience_s: MEASUREMENT_WAIT_S,
                    connect,
                };
            }
            std::thread::sleep(std::time::Duration::from_millis(10));
        };
        self.connect_wait = Some(wait);
        wait
    }
    fn command(&mut self, q: &[f64], seconds: f64) -> Result<()> {
        self.command_with_velocity(q, &[0.0; 7], CommandTiming::Timed(seconds))
    }
    fn command_with_velocity(
        &mut self,
        q: &[f64],
        velocity: &[f64; 7],
        timing: CommandTiming,
    ) -> Result<()> {
        let seconds = timing.interval();
        self.released()?;
        checked_target(q, &self.limits)?;
        let m = self.measured()?;
        if m.mode != Mode::Position || !m.error.is_empty() {
            return Err(Error(
                "native controller unhealthy or not in position mode".into(),
            ));
        }
        let actual = measured_axes(&m)?;
        if velocity
            .iter()
            .zip(&self.limits)
            .any(|(v, l)| !v.is_finite() || v.abs() > l.velocity_max)
            || !seconds.is_finite()
            || seconds <= 0.0
            || seconds > 60.0
            || self
                .limits
                .iter()
                .any(|limit| !limit.velocity_max.is_finite() || limit.velocity_max <= 0.0)
        {
            return Err(Error("native command exceeds live velocity limits".into()));
        }
        // A target may lead the measured joint by the controller's tracking
        // lag plus whatever a move of `seconds` covers at the controller's
        // velocity limit: a 2.5 ms streamed sample gets the lag allowance, a
        // 4 s recovery move gets its full travel. One tick of the velocity
        // limit alone refused the lag at real time; the lag allowance alone
        // refused every landing from a scan viewpoint.
        if q[..6]
            .iter()
            .zip(&actual[..6])
            .zip(&self.limits)
            .any(|((target, current), limit)| {
                (target - current).abs()
                    > crate::TRACKING_ERROR_LIMIT_RAD + limit.velocity_max * seconds
            })
        {
            return Err(Error(
                "native target leads measured joints beyond the tracking limit and the move's own travel".into(),
            ));
        }
        // Measurement may block. Read the monitor again immediately before SDK command.
        self.released()?;
        let result = sdk(self
            .driver()?
            .positions(q, velocity, &[0.0; 7], timing.sdk_goal_time()));
        self.carriage_target = if result.is_ok() { Some(q[6]) } else { None };
        self.stream_target = if result.is_ok() {
            q.try_into().ok()
        } else {
            None
        };
        result
    }
}
impl ArmBackend for TrossenArm {
    fn connect(&mut self, cfg: &ArmConfig) -> Result<()> {
        if cfg.joints != 6 {
            return Err(Error(
                "native arm has six rotational joints plus carriage".into(),
            ));
        }
        self.released()?;
        if let Some(mut old) = self.driver.take() {
            // Recover.Freeze owns the freeze attempt. A wedged old session
            // must not have to accept another hold before a fresh connection
            // can clear its firmware fault (the legacy landing contract).
            if let Err(error) = sdk(old.pin_mut().cleanup()) {
                eprintln!("recovery: old controller cleanup failed; reconnecting fresh: {error}");
            }
        }
        self.carriage_target = None;
        self.stream_target = None;
        self.guiding = false;
        self.limits.clear();
        let mut driver = sdk(ffi::make_driver())?;
        self.released()?;
        sdk(driver.pin_mut().configure(
            &self.config.address.to_string(),
            matches!(self.config.role, Role::Follower),
            true,
            recovery::CONFIGURE_TIMEOUT_S,
        ))?;
        self.driver = Some(driver);
        if let Err(stop) = self.released() {
            return Err(Error(format!("{stop}; hold={:?}", self.hold())));
        }
        self.await_live_measurement();
        self.refresh_limits()
    }
    fn connect_wait(&self) -> Option<ConnectWait> {
        self.connect_wait
    }
    fn recovery_limits(&mut self) -> Result<[recovery::PositionLimit; 7]> {
        self.refresh_limits()?;
        if self.limits.len() != 7 {
            return Err(Error("native recovery requires seven live limits".into()));
        }
        Ok(std::array::from_fn(|i| recovery::PositionLimit {
            min: self.limits[i].position_min,
            max: self.limits[i].position_max,
        }))
    }
    fn recovery_needs_configuration(&mut self, measured: &[f64; 7]) -> Result<bool> {
        self.refresh_limits()?;
        feedback_needs_configuration(measured, &self.limits)
    }
    /// Per-tick feedback: joint state from the SDK's cached controller
    /// output, controller mode and error re-read at most every
    /// STATUS_MAX_AGE_S. Those two are TCP queries of about 1.5 ms each on
    /// the arm node; read on every 2.5 ms tick, twice per streamed sample,
    /// they held the control loop below half real time.
    fn measured(&mut self) -> Result<Measured> {
        let raw = sdk(self.driver()?.measure(STATUS_MAX_AGE_S))?;
        checked_measurement(raw, self.carriage_target)
    }
    fn contact_policy(&self) -> crate::ContactPolicy {
        self.config.contact
    }
    fn carriage_qualified(&self) -> bool {
        self.config.carriage_qualified
    }
    fn retract_carriage(&mut self, target_m: f64, goal_time_s: f64) -> Result<()> {
        if !self.config.carriage_qualified {
            return Err(crate::unqualified_carriage(
                "trip retract",
                Some(self.config.role.name()),
            ));
        }
        if target_m != 0.032 || goal_time_s != 0.6 {
            return Err(Error("unqualified native trip retract".into()));
        }
        self.released()?;
        let measured = self.measured()?;
        let mut q = measured_axes(&measured)?;
        // The retract must never be refused for feedback resting a few
        // millirad past a nominal bound: hold the rotary joints at the measured
        // pose clamped into the live limits (sweep-arm-pink-20260916_122412
        // reported PEN NOT RETRACTED for a 0.0002 rad excursion).
        clamp_rotary_into_limits(&mut q, &self.limits)?;
        q[6] = target_m;
        self.command(&q, goal_time_s)
    }
    fn move_carriage(&mut self, target_m: f64, goal_time_s: f64) -> Result<()> {
        if !self.config.carriage_qualified {
            return Err(crate::unqualified_carriage(
                "carriage move",
                Some(self.config.role.name()),
            ));
        }
        if !target_m.is_finite() || !(1.0..=5.0).contains(&goal_time_s) {
            return Err(Error("unqualified native carriage move".into()));
        }
        self.released()?;
        self.refresh_limits()?;
        let measured = self.measured()?;
        let mut q = measured_axes(&measured)?;
        if (target_m - q[6]).abs() / goal_time_s > crate::worker::CARRIAGE_MOVE_MAX_M_PER_S {
            return Err(Error("carriage move faster than 50 mm/s".into()));
        }
        clamp_rotary_into_limits(&mut q, &self.limits)?;
        q[6] = target_m;
        self.command(&q, goal_time_s)
    }
    fn recovery_move(&mut self, target: &[f64; 7], phase: recovery::Phase) -> Result<()> {
        self.released()?;
        self.refresh_limits()?;
        self.command(target, phase.duration())
    }
    fn takeover(&mut self, target: &[f64; 7]) -> Result<()> {
        self.released()?;
        self.refresh_limits()?;
        checked_target(target, &self.limits)?;
        // Unlike normal mode selection, recovery may hold just inside a limit
        // when the measured carriage rests beyond it. The prepared command,
        // not an out-of-bounds measured position, is sent after mode selection.
        self.released()?;
        if position_mode_needed(&self.measured_fresh()?)? {
            self.released()?;
            sdk(self.driver()?.position_mode())?;
        }
        self.command(target, recovery::TAKEOVER_S)?;
        self.guiding = false;
        Ok(())
    }
    fn reseed_hold(&mut self, max_velocity: f64, envelope: f64) -> Result<std::time::Duration> {
        let started = std::time::Instant::now();
        self.released()?;
        self.refresh_limits()?;
        let measured = self.measured_fresh()?;
        let needs_mode = position_mode_needed(&measured)?;
        let target = reseed_target(
            &measured_axes(&measured)?,
            &self.limits,
            max_velocity,
            envelope,
        )?;
        reseed_marker(
            "prepared",
            started,
            serde_json::json!({"measured": measured, "target_axes": target,
                "mode_change_required": needs_mode, "goal_time_s": recovery::TAKEOVER_S,
                "max_velocity_rad_s": max_velocity, "envelope_rad": envelope}),
        );
        self.released()?;
        if needs_mode {
            reseed_marker("position_mode_begin", started, serde_json::json!({}));
            let result = sdk(self.driver()?.position_mode());
            reseed_marker(
                "position_mode_returned",
                started,
                serde_json::json!({"ok": result.is_ok(), "error": result.as_ref().err().map(ToString::to_string)}),
            );
            result?;
        }
        // These bracket the existing validated command wrapper, including its
        // feedback/stop checks, rather than claiming an SDK send timestamp.
        reseed_marker("target_begin", started, serde_json::json!({}));
        let result = self.command(&target, recovery::TAKEOVER_S);
        reseed_marker(
            "target_returned",
            started,
            serde_json::json!({"ok": result.is_ok(), "error": result.as_ref().err().map(ToString::to_string)}),
        );
        result?;
        self.guiding = false;
        Ok(std::time::Duration::from_secs_f64(recovery::TAKEOVER_S))
    }
    fn set_mode(&mut self, mode: Mode) -> Result<()> {
        self.released()?;
        match mode {
            Mode::Position => {
                let measured = self.measured_fresh()?;
                if !measured.error.is_empty() {
                    return Err(Error(measured.error));
                }
                let q = measured_axes(&measured)?;
                checked_hold_pose(&q, &self.limits)?;
                for (index, (q, limit)) in q[..6].iter().zip(&self.limits).enumerate() {
                    let target = q.clamp(limit.position_min, limit.position_max);
                    if target != *q {
                        eprintln!(
                            "native takeover: joint {} measured {q:.9} rad is within feedback tolerance; controller clips hold target to {target:.9} rad",
                            index + 1
                        );
                    }
                }
                self.released()?;
                sdk(self.driver()?.position_mode())
            }
            Mode::Idle => {
                self.carriage_target = None;
                self.stream_target = None;
                self.guiding = false;
                sdk(self.driver()?.idle_mode())
            }
            Mode::Fault => Err(Error("fault is an observation, not a command mode".into())),
            Mode::HandGuiding => Err(Error(
                "hand guiding is entered through enter_hand_guiding with a carriage datum".into(),
            )),
        }
    }
    fn shutdown_idle(&mut self) -> Result<()> {
        // Terminal teardown only: never re-enable position/effort control to
        // release a supported arm. Ordinary mode changes retain their stop gate.
        self.carriage_target = None;
        self.stream_target = None;
        self.guiding = false;
        sdk(self.driver()?.idle_mode())
    }
    fn move_j(&mut self, target: &[f64], goal_time_s: f64) -> Result<()> {
        self.released()?;
        if target.len() != 6 {
            return Err(Error(
                "native arm target needs six rotational joints".into(),
            ));
        }
        let carriage = self
            .carriage_target
            .ok_or_else(|| Error("measured re-seed required before native motion".into()))?;
        let mut q = target.to_vec();
        q.push(carriage);
        self.command(&q, goal_time_s)
    }
    fn stream_joint(&mut self, sample: &crate::JointSample) -> Result<()> {
        self.command_with_velocity(
            &sample.positions,
            &sample.velocities,
            CommandTiming::Sample(sample.dt_s),
        )
    }
    fn stream(&mut self, sample: &Sample) -> Result<()> {
        if !sample.contact.is_finite() || sample.contact < 0.0 {
            return Err(Error("invalid contact input".into()));
        }
        let carriage = self
            .carriage_target
            .ok_or_else(|| Error("measured re-seed required before native motion".into()))?;
        let mut q = sample.joints.clone();
        q.push(carriage);
        self.command_with_velocity(&q, &[0.0; 7], CommandTiming::Sample(sample.dt_s))
    }
    fn hold(&mut self) -> Result<()> {
        // A measured freeze is permitted while STOP is latched. It never
        // substitutes an old commanded pose or retracts. From hand guiding it
        // does select position mode on every joint first, in the order the
        // qualified teleop executor uses: set_all_modes(position), then
        // set_all_positions(get_all_positions(), 0, non-blocking).
        self.carriage_target = None;
        self.stream_target = None;
        if self.guiding {
            sdk(self.driver()?.position_mode())?;
            self.guiding = false;
        }
        let q = sdk(self.driver()?.hold())?;
        if q.len() != 7 || q.iter().any(|q| !q.is_finite()) {
            return Err(Error("invalid native hold receipt".into()));
        }
        self.carriage_target = Some(q[6]);
        self.stream_target = q.as_slice().try_into().ok();
        Ok(())
    }
    fn finish_joint_stream(&mut self) -> Result<()> {
        self.released()?;
        let q = self
            .stream_target
            .ok_or_else(|| Error("completed stream has no retained target".into()))?;
        // Normal completion holds the exact last command with zero velocity.
        // Fresh feedback, live limits and the stop are still checked below.
        // Emergency Hold separately freezes the measured pose.
        self.command_with_velocity(&q, &[0.0; 7], CommandTiming::Sample(0.0025))
    }
    fn enter_hand_guiding(&mut self, carriage_m: f64) -> Result<()> {
        self.released()?;
        if self.guiding {
            return Err(Error("already hand guiding".into()));
        }
        let measured = self.measured_fresh()?;
        if measured.mode != Mode::Position || !measured.error.is_empty() {
            return Err(Error(
                "hand guiding needs a healthy position-mode controller".into(),
            ));
        }
        let mut q = measured_axes(&measured)?;
        q[6] = carriage_m;
        checked_hold_pose(&q, &self.limits)?;
        self.released()?;
        // Modes first (six external effort, carriage position), then zero
        // commanded effort on the arm joints, then the carriage datum as its
        // immediate position target; the caller reads the modes back fresh.
        sdk(self.driver()?.joint_modes(&HAND_GUIDING_MODES))?;
        self.guiding = true;
        self.stream_target = None;
        self.carriage_target = None;
        sdk(self.driver()?.zero_arm_external_efforts())?;
        sdk(self.driver()?.joint_position(6, carriage_m, 0.0))?;
        self.carriage_target = Some(carriage_m);
        Ok(())
    }
    fn measured_fresh(&mut self) -> Result<Measured> {
        Self::measured_fresh(self)
    }
    fn payload_readback(&mut self) -> Result<Option<serde_json::Value>> {
        let ee = sdk(self.driver()?.end_effector())?;
        Ok(Some(serde_json::json!({
            "backend": "trossen",
            "sdk_version": trossen_arm_sys::SDK_VERSION,
            "standard_end_effector": match self.config.role {
                Role::Follower => "wxai_v0_follower",
                Role::Leader => "wxai_v0_leader",
            },
            "palm_mass_kg": ee.palm_mass_kg,
            "palm_origin_xyz_m": ee.palm_origin_xyz_m,
            "palm_inertia": ee.palm_inertia,
            "finger_left_mass_kg": ee.finger_left_mass_kg,
            "finger_right_mass_kg": ee.finger_right_mass_kg,
            "offset_finger_left_m": ee.offset_finger_left_m,
            "offset_finger_right_m": ee.offset_finger_right_m,
            "pitch_circle_radius_m": ee.pitch_circle_radius_m,
            "t_flange_tool": ee.t_flange_tool,
            "note": "vendor standard end effector reapplied after configuration loading; the fitted mount, tool, camera and cable load are not represented in it",
        })))
    }
    fn clear_error(&mut self) -> Result<()> {
        self.released()?;
        let golden = self.config.golden.clone();
        self.load_config(&golden)?;
        self.connect(&ArmConfig { joints: 6 })?;
        let error = self.measured_fresh()?.error;
        if !error.is_empty() {
            return Err(Error(format!(
                "fault persists after golden/reconnect: {error}"
            )));
        }
        Ok(())
    }
    fn load_config(&mut self, path: &Path) -> Result<()> {
        self.released()?;
        let file = path.canonicalize().map_err(|e| Error(e.to_string()))?;
        if file != self.config.golden {
            return Err(Error("native config differs from selected golden".into()));
        }
        self.released()?;
        self.carriage_target = None;
        self.stream_target = None;
        self.guiding = false;
        sdk(self.driver()?.load_config(file.to_str().unwrap()))?;
        self.refresh_limits()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    fn raw() -> ffi::Measurement {
        ffi::Measurement {
            positions: vec![0.1; 7],
            velocities: vec![0.2; 7],
            accelerations: vec![0.25; 7],
            efforts: vec![0.35; 7],
            external_efforts: vec![0.3; 7],
            compensation_efforts: vec![0.4; 7],
            modes: vec![1; 7],
            error: "No error".into(),
        }
    }
    #[test]
    fn native_mapping_separates_carriage_and_unknown_command_from_arm_axes() {
        let m = checked_measurement(raw(), None).unwrap();
        assert_eq!(m.joints.len(), 6);
        assert_eq!(m.velocities.len(), 6);
        assert_eq!(m.efforts.len(), 6);
        assert_eq!(m.dynamics.as_ref().unwrap().accelerations, vec![0.25; 7]);
        assert_eq!(m.dynamics.as_ref().unwrap().efforts, vec![0.35; 7]);
        assert_eq!(
            m.dynamics.as_ref().unwrap().compensation_efforts,
            vec![0.4; 7]
        );
        assert_eq!(m.carriage.unwrap().target_m, None);
        assert_eq!(m.mode, Mode::Position);
        assert_eq!(m.error, "");
        let m = checked_measurement(raw(), Some(0.032)).unwrap();
        assert_eq!(m.carriage.unwrap().target_m, Some(0.032));
        let mut broken = raw();
        broken.modes[6] = 0;
        assert_eq!(checked_measurement(broken, None).unwrap().mode, Mode::Fault);
        let mut guiding = raw();
        guiding.modes = HAND_GUIDING_MODES.to_vec();
        assert_eq!(
            checked_measurement(guiding, Some(0.0)).unwrap().mode,
            Mode::HandGuiding
        );
        // Any other mixed vector, including a carriage in effort mode or a
        // single rotary joint left in position control, stays a fault.
        for (index, value) in [(6, 3), (0, 1), (2, 0), (5, 2)] {
            let mut mixed = raw();
            mixed.modes = HAND_GUIDING_MODES.to_vec();
            mixed.modes[index] = value;
            assert_eq!(checked_measurement(mixed, None).unwrap().mode, Mode::Fault);
        }
        let mut broken = raw();
        broken.positions[6] = f64::NAN;
        assert!(checked_measurement(broken, None).is_err());
    }
    #[test]
    fn carriage_limits_use_metres_and_never_the_rotational_joint_box() {
        let mut limits: Vec<_> = (0..7)
            .map(|_| ffi::JointLimit {
                position_min: -3.0,
                position_max: 3.0,
                position_tolerance: 0.0,
                velocity_max: 1.0,
                velocity_tolerance: 0.0,
                effort_max: 20.0,
                effort_tolerance: 0.0,
            })
            .collect();
        limits[6].position_min = -0.006;
        limits[6].position_max = 0.04;
        assert!(checked_target(&[0.0; 7], &limits).is_ok());
        let mut q = [0.0; 7];
        q[6] = 0.032;
        assert!(checked_target(&q, &limits).is_ok());
        for bad in [0.041, -0.0061, 1.0, f64::NAN] {
            q[6] = bad;
            let error = checked_target(&q, &limits).unwrap_err().to_string();
            assert!(error.contains("joint 7 (index 6)"), "{error}");
            assert!(error.contains("m (carriage)"), "{error}");
        }
        assert!(checked_target(&[0.0; 6], &limits).is_err());
        // A parked shoulder can read slightly below zero without a firmware
        // fault. Its position tolerance does not widen the command range.
        limits[1].position_min = 0.0;
        limits[1].position_tolerance = 0.2;
        q = [0.0; 7];
        q[1] = -0.0032425420358777046;
        q[6] = -0.0047224219888448715;
        assert!(checked_hold_pose(&q, &limits).is_ok());
        let error = checked_target(&q, &limits).unwrap_err().to_string();
        assert!(error.contains("joint 2 (index 1)"), "{error}");
        assert!(error.contains("target -0.003242542"), "{error}");
        assert!(
            error.contains("range [0.000000000, 3.000000000] rad"),
            "{error}"
        );
        q[1] = 0.0;
        assert!(checked_target(&q, &limits).is_ok());
        for value in [-0.199, 3.199] {
            q[1] = value;
            assert!(checked_hold_pose(&q, &limits).is_ok());
            assert!(checked_target(&q, &limits).is_err());
        }
        for value in [-0.201, 3.201, f64::NAN] {
            q[1] = value;
            assert!(checked_hold_pose(&q, &limits).is_err());
        }
        q[1] = 0.0;
        for tolerance in [-0.1, f64::NAN, f64::INFINITY] {
            limits[1].position_tolerance = tolerance;
            assert!(checked_hold_pose(&q, &limits).is_err());
        }
        limits[1].position_tolerance = 0.2;
        limits[6].position_tolerance = 0.004;
        q[6] = -0.0061;
        assert!(checked_hold_pose(&q, &limits).is_err());
    }
    #[test]
    fn reseed_corrects_rotary_tolerance_within_cap_and_preserves_carriage() {
        let mut limits: Vec<_> = (0..7)
            .map(|_| ffi::JointLimit {
                position_min: 0.0,
                position_max: 3.0,
                position_tolerance: 0.1,
                velocity_max: 1.0,
                velocity_tolerance: 0.0,
                effort_max: 20.0,
                effort_tolerance: 0.0,
            })
            .collect();
        limits[6].position_max = 0.04;
        let mut q = [0.1; 7];
        q[6] = 0.002;
        assert_eq!(reseed_target(&q, &limits, 0.1, 3.3).unwrap(), q);
        q[1] = -0.003;
        let target = reseed_target(&q, &limits, 0.1, 3.3).unwrap();
        assert_eq!(target[1], recovery::LIMIT_MARGIN);
        assert_eq!(target[6], q[6]);
        assert!(checked_target(&q, &limits).is_err());
        checked_target(&target, &limits).unwrap();
        for value in [-0.026, -0.101, f64::NAN] {
            q[1] = value;
            assert!(reseed_target(&q, &limits, 0.1, 3.3).is_err());
        }
        q[1] = 0.1;
        for value in [-0.0001, 0.0401] {
            q[6] = value;
            assert!(reseed_target(&q, &limits, 0.1, 3.3).is_err());
        }
        q[6] = 0.002;
        assert!(reseed_target(&q, &limits, 0.1, 0.09).is_err());
        for bad in [0.0, -0.1, f64::NAN, f64::INFINITY] {
            assert!(reseed_target(&q, &limits, bad, 3.3).is_err());
            assert!(reseed_target(&q, &limits, 0.1, bad).is_err());
        }
    }
    #[test]
    fn healthy_position_control_does_not_reselect_modes() {
        let mut measured = checked_measurement(raw(), None).unwrap();
        assert!(!position_mode_needed(&measured).unwrap());
        measured.mode = Mode::Idle;
        assert!(position_mode_needed(&measured).unwrap());
        measured.mode = Mode::HandGuiding;
        assert!(position_mode_needed(&measured).unwrap());
        measured.error = "controller fault".into();
        assert!(position_mode_needed(&measured).is_err());
    }
    #[test]
    fn recovery_feedback_tolerance_avoids_reload_without_widening_targets() {
        let mut limits: Vec<_> = (0..7)
            .map(|_| ffi::JointLimit {
                position_min: 0.0,
                position_max: 3.0,
                position_tolerance: 0.01,
                velocity_max: 1.0,
                velocity_tolerance: 0.0,
                effort_max: 20.0,
                effort_tolerance: 0.0,
            })
            .collect();
        limits[6].position_max = 0.04;
        let mut q = [0.0; 7];
        q[6] = 0.002;
        for (axis, value) in [(1, -0.003), (2, -0.0002), (5, 3.009)] {
            q[axis] = value;
            assert!(!feedback_needs_configuration(&q, &limits).unwrap());
            assert!(checked_target(&q, &limits).is_err());
            let nominal = std::array::from_fn(|i| recovery::PositionLimit {
                min: limits[i].position_min,
                max: limits[i].position_max,
            });
            let targets = recovery::Targets::from_measured(q, [0.0; 7], &nominal, None).unwrap();
            checked_target(&targets.takeover, &limits).unwrap();
            q[axis] = 0.0;
        }
        for (axis, value) in [(1, -0.011), (5, 3.011), (6, -0.0002), (6, 0.0401)] {
            q[axis] = value;
            assert!(feedback_needs_configuration(&q, &limits).unwrap());
            q[axis] = 0.002;
        }
        q[1] = f64::NAN;
        assert!(feedback_needs_configuration(&q, &limits).is_err());
        q[1] = 0.0;
        assert!(feedback_needs_configuration(&q, &limits[..6]).is_err());
        for bad in [f64::NAN, f64::INFINITY, -0.01] {
            limits[1].position_tolerance = bad;
            assert!(feedback_needs_configuration(&q, &limits).is_err());
        }
        limits[1].position_tolerance = 0.01;
        limits[1].position_max = 0.0;
        assert!(feedback_needs_configuration(&q, &limits).is_err());
    }
    #[test]
    fn constructor_and_latched_controls_never_connect() {
        let temp = tempfile::tempdir().unwrap();
        let golden = temp.path().join("follower.yaml");
        std::fs::write(&golden, "test-only: not loaded\n").unwrap();
        let lease = Arc::new(HardwareLease::for_test(
            temp.path().join("driver.lock").as_path(),
        ));
        let stop = Arc::new(AtomicI32::new(estop::FAULT));
        let mut arm = TrossenArm::new(
            Config {
                address: Ipv4Addr::LOCALHOST,
                role: Role::Follower,
                golden,
                contact: crate::ContactPolicy::Contact,
                carriage_qualified: true,
            },
            stop.clone(),
            lease,
        )
        .unwrap();
        assert!(arm.connect(&ArmConfig { joints: 6 }).is_err());
        assert!(arm.set_mode(Mode::Position).is_err());
        assert!(
            arm.reseed_hold(0.1, 3.3)
                .unwrap_err()
                .to_string()
                .contains("e-stop")
        );
        assert!(arm.retract_carriage(0.032, 0.6).is_err());
        assert!(arm.move_j(&[0.0; 6], 0.5).is_err());
        assert!(
            arm.finish_joint_stream()
                .unwrap_err()
                .to_string()
                .contains("e-stop")
        );
        assert!(
            arm.stream_joint(&crate::JointSample {
                positions: [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.002],
                velocities: [0.0; 7],
                dt_s: 0.0025,
                pen: false,
            })
            .unwrap_err()
            .to_string()
            .contains("e-stop")
        );
        assert!(arm.measured().is_err());
        assert!(arm.hold().is_err());
        assert!(
            arm.enter_hand_guiding(0.0)
                .unwrap_err()
                .to_string()
                .contains("e-stop")
        );
        assert!(arm.driver.is_none());
        stop.store(estop::OK, Ordering::SeqCst);
        assert!(
            arm.enter_hand_guiding(0.0)
                .unwrap_err()
                .to_string()
                .contains("not connected")
        );
        assert!(arm.set_mode(Mode::HandGuiding).is_err());
        assert!(
            arm.move_j(&[0.0; 6], 0.5)
                .unwrap_err()
                .to_string()
                .contains("re-seed")
        );
        assert!(arm.connect(&ArmConfig { joints: 7 }).is_err());
        assert!(arm.driver.is_none());
        assert_eq!(arm.contact_policy(), crate::ContactPolicy::Contact);
        assert!(arm.carriage_qualified());
    }
    /// The carriage refusals are keyed on the profile's measurement, not the
    /// role: an unqualified carriage of either role refuses its retract and
    /// timed move naming the key, before any driver call.
    #[test]
    fn an_unqualified_carriage_refuses_retract_and_move_naming_the_profile_key() {
        let temp = tempfile::tempdir().unwrap();
        let golden = temp.path().join("leader.yaml");
        std::fs::write(&golden, "test-only: not loaded\n").unwrap();
        let lease = Arc::new(HardwareLease::for_test(
            temp.path().join("driver.lock").as_path(),
        ));
        let stop = Arc::new(AtomicI32::new(estop::OK));
        for role in [Role::Leader, Role::Follower] {
            let mut arm = TrossenArm::new(
                Config {
                    address: Ipv4Addr::LOCALHOST,
                    role,
                    golden: golden.clone(),
                    contact: crate::ContactPolicy::Standoff,
                    carriage_qualified: false,
                },
                stop.clone(),
                lease.clone(),
            )
            .unwrap();
            assert_eq!(arm.contact_policy(), crate::ContactPolicy::Standoff);
            assert!(!arm.carriage_qualified());
            for error in [
                arm.retract_carriage(0.032, 0.6).unwrap_err().to_string(),
                arm.move_carriage(0.0, 2.0).unwrap_err().to_string(),
            ] {
                assert!(
                    error.contains(&format!("tatbot.yaml {}.carriage_qualified", role.name())),
                    "{error}"
                );
            }
            assert!(arm.driver.is_none());
        }
    }
}
