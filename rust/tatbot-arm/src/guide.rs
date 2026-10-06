//! Hand-guiding owner protocol: the state machine, telemetry record and
//! JSON-lines wire format shared with scripts/vision/arm_guide_owner.py.
//!
//! The owner process (`tatbot-arm-guide`) keeps one selected arm's worker,
//! reads commands from stdin, writes events to stdout and retains every
//! control tick's feedback in `telemetry.bin`. Nothing here touches an SDK:
//! the state machine is pure so its transitions are exact and testable, and
//! the binary applies them to a `Worker` through the existing primitives.
use crate::{Measured, Mode, worker::TelemetrySample};
use serde::{Deserialize, Serialize};
use std::io::Write;

/// Byte layout of one retained tick, little-endian, matching
/// `arm_calibration.TELEMETRY_RECORD` ("<QQQ21dBiB"): tick, wall ns,
/// monotonic ns since worker start, positions[7], velocities[7], external
/// efforts[7], mode code, e-stop word, flags (bit 0 guiding, bit 1 latched).
pub const TELEMETRY_MAGIC: &[u8; 8] = b"TBGUIDE1";
pub const TELEMETRY_RECORD_BYTES: usize = 8 * 3 + 21 * 8 + 1 + 4 + 1;
/// Optional parallel SDK diagnostic channels. A matching tick and wall time
/// join this record to TBGUIDE1 without changing historical calibration logs.
/// Layout: tick, wall ns, monotonic ns, acceleration[7], joint effort[7],
/// compensation effort[7]. Offline backends encode missing values as NaN.
pub const DYNAMICS_MAGIC: &[u8; 8] = b"TBDYNA01";
pub const DYNAMICS_RECORD_BYTES: usize = 8 * 3 + 21 * 8;

/// Inspection uses the configured parked targets, with optional bounded base
/// yaw. Feedback at a hard stop is not a legal new target. A conservative
/// twice-mean interpolation bound leaves room for arrival readback error.
pub fn wrist_target(
    measured: &[f64],
    staged: &[f64; 7],
    angle: f64,
    base: Option<f64>,
) -> crate::Result<(Vec<f64>, f64)> {
    if measured.len() != 6
        || measured.iter().chain(staged).any(|v| !v.is_finite())
        || !(-std::f64::consts::PI..=std::f64::consts::PI).contains(&angle)
    {
        return Err(crate::Error("invalid wrist pose".into()));
    }
    if measured[0].abs() > std::f64::consts::PI / 6.0 + 0.1
        || measured[1..5]
            .iter()
            .zip(&staged[1..5])
            .any(|(q, rest)| (q - rest).abs() > 0.1)
    {
        return Err(crate::Error(
            "wrist positioning requires the supported parked posture".into(),
        ));
    }
    let delta = (angle - measured[5]).abs();
    if delta > std::f64::consts::FRAC_PI_2 + 0.05 {
        return Err(crate::Error("wrist move exceeds a quarter turn".into()));
    }
    let base = base.unwrap_or(measured[0]);
    if !base.is_finite()
        || base.abs() > std::f64::consts::PI / 6.0
        || (base - measured[0]).abs() > std::f64::consts::PI / 6.0 + 0.05
    {
        return Err(crate::Error(
            "base inspection yaw exceeds 30 degrees".into(),
        ));
    }
    let mut target = staged[..6].to_vec();
    target[0] = base;
    target[5] = angle;
    let travel = target
        .iter()
        .zip(measured)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0_f64, f64::max);
    Ok((
        target,
        (2.0 * travel / 0.08).max(crate::recovery::STAGED_POSE_S),
    ))
}

pub fn telemetry_header() -> [u8; 16] {
    let mut header = [0u8; 16];
    header[..8].copy_from_slice(TELEMETRY_MAGIC);
    header[8..].copy_from_slice(&(TELEMETRY_RECORD_BYTES as u64).to_le_bytes());
    header
}

pub fn dynamics_header() -> [u8; 16] {
    let mut header = [0u8; 16];
    header[..8].copy_from_slice(DYNAMICS_MAGIC);
    header[8..].copy_from_slice(&(DYNAMICS_RECORD_BYTES as u64).to_le_bytes());
    header
}

pub fn encode_dynamics_sample(sample: &TelemetrySample) -> [u8; DYNAMICS_RECORD_BYTES] {
    let mut out = [0u8; DYNAMICS_RECORD_BYTES];
    let mut cursor = std::io::Cursor::new(&mut out[..]);
    let mono_ns = u64::try_from(sample.since_start.as_nanos()).unwrap_or(u64::MAX);
    for word in [sample.tick, sample.wall_ns, mono_ns] {
        cursor.write_all(&word.to_le_bytes()).unwrap();
    }
    let missing = [f64::NAN; 7];
    let diagnostics = sample.measured.dynamics.as_ref();
    for values in [
        diagnostics.map_or(missing.as_slice(), |d| d.accelerations.as_slice()),
        diagnostics.map_or(missing.as_slice(), |d| d.efforts.as_slice()),
        diagnostics.map_or(missing.as_slice(), |d| d.compensation_efforts.as_slice()),
    ] {
        for axis in 0..7 {
            cursor
                .write_all(&values.get(axis).copied().unwrap_or(f64::NAN).to_le_bytes())
                .unwrap();
        }
    }
    out
}

pub fn encode_sample(sample: &TelemetrySample) -> [u8; TELEMETRY_RECORD_BYTES] {
    let mut out = [0u8; TELEMETRY_RECORD_BYTES];
    let mut cursor = std::io::Cursor::new(&mut out[..]);
    let m = &sample.measured;
    let carriage = m.carriage.as_ref();
    let axes = |values: &[f64], extra: f64| -> [f64; 7] {
        let mut row = [f64::NAN; 7];
        for (slot, value) in row
            .iter_mut()
            .zip(values.iter().chain(std::iter::once(&extra)))
        {
            *slot = *value;
        }
        row
    };
    let positions = axes(&m.joints, carriage.map_or(f64::NAN, |c| c.position_m));
    let velocities = axes(&m.velocities, 0.0);
    let efforts = axes(&m.efforts, carriage.map_or(f64::NAN, |c| c.effort_n));
    let mono_ns = u64::try_from(sample.since_start.as_nanos()).unwrap_or(u64::MAX);
    for word in [sample.tick, sample.wall_ns, mono_ns] {
        cursor.write_all(&word.to_le_bytes()).unwrap();
    }
    for value in positions.iter().chain(&velocities).chain(&efforts) {
        cursor.write_all(&value.to_le_bytes()).unwrap();
    }
    cursor.write_all(&[m.mode.code()]).unwrap();
    cursor.write_all(&sample.estop.to_le_bytes()).unwrap();
    cursor
        .write_all(&[u8::from(sample.guiding) | (u8::from(sample.latched) << 1)])
        .unwrap();
    out
}

/// Commands the conductor may send, one JSON object per line.
#[derive(Clone, Debug, Deserialize, PartialEq)]
#[serde(tag = "cmd", rename_all = "snake_case")]
pub enum Command {
    /// Open the driver and take over at the measured pose (position hold).
    Connect,
    /// Enter hand guiding at the confirmed carriage datum (the measured
    /// carriage when omitted; a stated datum must match the measurement).
    Guide {
        #[serde(default)]
        carriage_m: Option<f64>,
    },
    /// Record a settled hold window while guiding continues.
    Hold {
        id: String,
        settle_s: f64,
        capture_s: f64,
    },
    /// Leave hand guiding for a measured position hold.
    Stop,
    /// After a latched guard trip: clear the latch with a measured position
    /// re-seed so the attended capture can continue. The carriage stays where
    /// the trip left it until `Carriage` returns it.
    Recover,
    /// Timed carriage-only move from a position hold (datum return before
    /// re-entering hand guiding, or parking clear before release).
    Carriage {
        carriage_m: f64,
    },
    /// Bring feedback resting just beyond a nominal rotary limit into the
    /// legal command range through the measured recovery path.
    Normalize,
    /// One small, slow relative rotary move from the measured position hold.
    /// The conductor pairs equal and opposite moves and retains every tick.
    JointStep {
        joint_index: usize,
        delta_rad: f64,
        speed_rad_s: f64,
    },
    /// Attended wrist and optional base positioning at the parked posture.
    Wrist {
        angle_rad: f64,
        #[serde(default)]
        base_rad: Option<f64>,
    },
    /// Conductor-side acquisition failure: latch a stop without releasing.
    Abort {
        reason: String,
    },
    /// Supported release to idle from a position hold.
    Release,
    /// Mock backends only: fault injection for regression tests.
    Inject(crate::Faults),
    Status,
}

/// Where the owner is; every transition is decided by `Session`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum State {
    /// Preflight done, no driver open.
    Ready,
    /// Driver open, all joints in position control at the measured pose.
    Holding,
    Guiding,
    /// A hold window is being recorded while guiding continues.
    Recording,
    /// Motors idle, driver still owned.
    Released,
    /// A fault, stop or terminal loss latched the worker; the driver stays
    /// owned and the arm holds where it can. Only Release leaves this state.
    Faulted,
    Closed,
}

#[derive(Debug, PartialEq, Eq)]
pub enum Refusal {
    WrongState(State),
    Invalid(&'static str),
}
impl std::fmt::Display for Refusal {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::WrongState(state) => write!(f, "not allowed in state {state:?}"),
            Self::Invalid(reason) => write!(f, "{reason}"),
        }
    }
}

/// Pure transition table. `admit` says whether a command may start from the
/// current state; the owner then reports `completed`/`faulted` outcomes.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Session {
    pub state: State,
}
impl Session {
    pub fn new() -> Self {
        Self {
            state: State::Ready,
        }
    }
    pub fn admit(&self, command: &Command) -> Result<State, Refusal> {
        use State::*;
        let next = match (self.state, command) {
            (Ready, Command::Connect) => Holding,
            (Holding, Command::Guide { .. }) => Guiding,
            (
                Guiding,
                Command::Hold {
                    settle_s,
                    capture_s,
                    id,
                },
            ) => {
                if id.is_empty()
                    || id.len() > 64
                    || !id
                        .bytes()
                        .all(|b| b.is_ascii_alphanumeric() || b == b'-' || b == b'_')
                {
                    return Err(Refusal::Invalid("hold id must be a short identifier"));
                }
                if !(0.0..=5.0).contains(settle_s) || !(0.5..=30.0).contains(capture_s) {
                    return Err(Refusal::Invalid(
                        "hold windows must be 0..5 s settle and 0.5..30 s capture",
                    ));
                }
                Recording
            }
            (Guiding, Command::Stop) => Holding,
            (Faulted, Command::Recover) => Holding,
            (Holding, Command::Carriage { carriage_m }) => {
                if !carriage_m.is_finite() {
                    return Err(Refusal::Invalid("carriage target must be finite"));
                }
                Holding
            }
            (Holding, Command::Normalize) => Holding,
            (
                Holding,
                Command::JointStep {
                    joint_index,
                    delta_rad,
                    speed_rad_s,
                },
            ) => {
                if *joint_index >= 6
                    || !delta_rad.is_finite()
                    || !speed_rad_s.is_finite()
                    || delta_rad.abs() < 0.002
                    || delta_rad.abs() > 0.02
                    || *speed_rad_s < 0.003
                    || *speed_rad_s > 0.06
                {
                    return Err(Refusal::Invalid(
                        "joint step requires index 0..5, 0.002..0.02 rad travel and 0.003..0.06 rad/s speed",
                    ));
                }
                Holding
            }
            (Holding, Command::Wrist { angle_rad, .. }) => {
                if !(-std::f64::consts::PI..=std::f64::consts::PI).contains(angle_rad) {
                    return Err(Refusal::Invalid("wrist angle must be within +/- pi"));
                }
                Holding
            }
            (Holding, Command::Release) => Released,
            (Faulted, Command::Release) => Released,
            (Faulted | Released | Holding | Guiding | Ready, Command::Status) => self.state,
            (Ready | Holding | Guiding | Recording | Faulted, Command::Abort { .. }) => Faulted,
            (Faulted | Released | Holding | Guiding | Ready, Command::Inject(_)) => self.state,
            (state, _) => return Err(Refusal::WrongState(state)),
        };
        Ok(next)
    }
    /// A hold window finished (accepted or not): guiding continues.
    pub fn recorded(&mut self) {
        debug_assert_eq!(self.state, State::Recording);
        self.state = State::Guiding;
    }
    /// A latched worker fault, physical stop, or conductor loss.
    pub fn fault(&mut self) {
        if self.state != State::Closed {
            self.state = State::Faulted;
        }
    }
    pub fn close(&mut self) {
        self.state = State::Closed;
    }
}
impl Default for Session {
    fn default() -> Self {
        Self::new()
    }
}

/// One accepted or rejected hold window, computed from the retained ticks
/// that fell inside it. The owner reports raw statistics; the conductor
/// applies the tool's declared accuracy profile in tool-tip millimetres.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct HoldWindow {
    pub id: String,
    pub start_wall_ns: u64,
    pub end_wall_ns: u64,
    pub ticks: usize,
    pub median_positions: [f64; 7],
    /// Largest excursion of any rotary joint from its window median, rad.
    pub joint_motion_max_rad: f64,
    /// Up to 64 evenly spaced seven-axis positions across the window, so the
    /// conductor can model the tool tip's own spread rather than a joint bound.
    #[serde(default)]
    pub sampled_positions: Vec<[f64; 7]>,
    /// Largest carriage excursion from its window median, m.
    pub carriage_motion_max_m: f64,
    pub carriage_median_m: f64,
    pub effort_median_nm: [f64; 7],
    pub effort_range_nm: [f64; 7],
    pub max_gap_ns: u64,
    pub healthy: bool,
    pub reason: Option<String>,
}

pub fn summarize_window(
    id: &str,
    samples: &[TelemetrySample],
    start_wall_ns: u64,
    end_wall_ns: u64,
) -> HoldWindow {
    let mut window: Vec<&TelemetrySample> = samples
        .iter()
        .filter(|s| (start_wall_ns..=end_wall_ns).contains(&s.wall_ns))
        .collect();
    window.sort_by_key(|s| s.tick);
    let mut healthy = !window.is_empty();
    let mut reason = None;
    let mut columns: Vec<Vec<f64>> = (0..21).map(|_| Vec::with_capacity(window.len())).collect();
    for sample in &window {
        let m = &sample.measured;
        if m.mode != Mode::HandGuiding
            || sample.estop != crate::estop::OK
            || sample.latched
            || !sample.guiding
        {
            healthy = false;
            reason
                .get_or_insert_with(|| "window was not entirely healthy hand guiding".to_string());
        }
        for (index, value) in m
            .joints
            .iter()
            .chain(std::iter::once(
                &m.carriage.as_ref().map_or(f64::NAN, |c| c.position_m),
            ))
            .chain(&m.velocities)
            .chain(std::iter::once(&0.0))
            .chain(&m.efforts)
            .chain(std::iter::once(
                &m.carriage.as_ref().map_or(f64::NAN, |c| c.effort_n),
            ))
            .enumerate()
        {
            columns[index].push(*value);
        }
    }
    let median = |values: &[f64]| -> f64 {
        if values.is_empty() {
            return f64::NAN;
        }
        let mut sorted = values.to_vec();
        sorted.sort_by(f64::total_cmp);
        sorted[sorted.len() / 2]
    };
    let mut median_positions = [f64::NAN; 7];
    let mut joint_motion_max_rad = 0.0_f64;
    for axis in 0..7 {
        median_positions[axis] = median(&columns[axis]);
        if axis < 6 {
            for value in &columns[axis] {
                joint_motion_max_rad =
                    joint_motion_max_rad.max((value - median_positions[axis]).abs());
            }
        }
    }
    let carriage_motion_max_m = columns[6]
        .iter()
        .map(|v| (v - median_positions[6]).abs())
        .fold(0.0, f64::max);
    let mut effort_median_nm = [f64::NAN; 7];
    let mut effort_range_nm = [f64::NAN; 7];
    for axis in 0..7 {
        let column = &columns[14 + axis];
        effort_median_nm[axis] = median(column);
        effort_range_nm[axis] = column.iter().cloned().fold(f64::MIN, f64::max)
            - column.iter().cloned().fold(f64::MAX, f64::min);
    }
    let mut max_gap_ns = 0;
    for pair in window.windows(2) {
        let gap = pair[1].since_start.saturating_sub(pair[0].since_start);
        max_gap_ns = max_gap_ns.max(u64::try_from(gap.as_nanos()).unwrap_or(u64::MAX));
    }
    if window.len() < 250 {
        healthy = false;
        reason.get_or_insert_with(|| format!("only {} ticks retained in the window", window.len()));
    }
    if max_gap_ns > 25_000_000 {
        healthy = false;
        reason.get_or_insert_with(|| {
            format!(
                "telemetry gap of {} ms inside the window",
                max_gap_ns / 1_000_000
            )
        });
    }
    let stride = window.len().div_ceil(64).max(1);
    let sampled_positions = window
        .iter()
        .step_by(stride)
        .map(|sample| {
            let mut q = [f64::NAN; 7];
            for (index, value) in sample.measured.joints.iter().take(6).enumerate() {
                q[index] = *value;
            }
            q[6] = sample
                .measured
                .carriage
                .as_ref()
                .map_or(f64::NAN, |c| c.position_m);
            q
        })
        .collect();
    HoldWindow {
        id: id.to_string(),
        start_wall_ns,
        end_wall_ns,
        ticks: window.len(),
        median_positions,
        joint_motion_max_rad,
        sampled_positions,
        carriage_motion_max_m,
        carriage_median_m: median_positions[6],
        effort_median_nm,
        effort_range_nm,
        max_gap_ns,
        healthy,
        reason,
    }
}

/// A compact per-tick summary for the operator terminal (about 5 Hz).
#[derive(Clone, Debug, Serialize)]
pub struct Tick {
    pub positions: [f64; 7],
    pub mode: Mode,
    pub estop: i32,
    pub guiding: bool,
    pub fault: Option<String>,
    pub late_ticks: u64,
    pub telemetry_dropped: u64,
}
impl Tick {
    pub fn from_measured(
        m: &Measured,
        estop: i32,
        guiding: bool,
        fault: Option<String>,
        late_ticks: u64,
        telemetry_dropped: u64,
    ) -> Self {
        let mut positions = [f64::NAN; 7];
        positions[..m.joints.len().min(6)].copy_from_slice(&m.joints[..m.joints.len().min(6)]);
        positions[6] = m.carriage.as_ref().map_or(f64::NAN, |c| c.position_m);
        Self {
            positions,
            mode: m.mode,
            estop,
            guiding,
            fault,
            late_ticks,
            telemetry_dropped,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::CarriageMeasured;
    use std::time::Duration;

    fn sample(tick: u64, mode: Mode, guiding: bool) -> TelemetrySample {
        TelemetrySample {
            tick,
            wall_ns: 1_000_000 * tick,
            since_start: Duration::from_micros(2500 * tick),
            measured: Measured {
                carriage: Some(CarriageMeasured {
                    position_m: 0.002,
                    target_m: Some(0.002),
                    effort_n: 1.0,
                }),
                joints: vec![0.1 + 0.0001 * (tick % 3) as f64; 6],
                velocities: vec![0.0; 6],
                efforts: vec![0.5; 6],
                dynamics: None,
                mode,
                error: String::new(),
            },
            estop: crate::estop::OK,
            guiding,
            latched: false,
        }
    }

    #[test]
    fn wrist_position_uses_legal_parked_targets_and_bounds_motion() {
        let parked = [0.0, 0.0, 0.0, 0.0, 0.0, std::f64::consts::FRAC_PI_2, 0.0];
        let mut measured = parked[..6].to_vec();
        measured[0] = 0.012;
        let (target, seconds) =
            wrist_target(&measured, &parked, std::f64::consts::PI, None).unwrap();
        assert_eq!(target[0], measured[0]);
        assert_eq!(&target[1..5], &parked[1..5]);
        assert!(2.0 * (target[5] - measured[5]).abs() / seconds <= 0.08 + 1e-12);
        assert!(wrist_target(&measured, &parked, -std::f64::consts::PI, None).is_err());
        assert!(wrist_target(&measured, &parked, f64::NAN, None).is_err());
        measured[1] = 0.2;
        assert!(wrist_target(&measured, &parked, 0.0, None).is_err());
        measured[1] = -0.004;
        let (target, _) = wrist_target(&measured, &parked, 0.0, Some(-0.35)).unwrap();
        assert_eq!(target[1], 0.0);
        assert_eq!(target[0], -0.35);
        assert!(wrist_target(&measured, &parked, 0.0, Some(0.6)).is_err());
        assert!(wrist_target(&measured, &parked, 0.0, Some(f64::NAN)).is_err());
        let command = Command::Wrist {
            angle_rad: 0.0,
            base_rad: None,
        };
        let mut session = Session::new();
        assert!(session.admit(&command).is_err());
        session.state = State::Holding;
        assert_eq!(session.admit(&command), Ok(State::Holding));
        session.state = State::Guiding;
        assert!(session.admit(&command).is_err());
    }

    #[test]
    fn transitions_follow_the_supported_progression_only() {
        let mut session = Session::new();
        assert_eq!(
            session.admit(&Command::Guide { carriage_m: None }),
            Err(Refusal::WrongState(State::Ready))
        );
        assert_eq!(session.admit(&Command::Connect), Ok(State::Holding));
        session.state = State::Holding;
        assert_eq!(session.admit(&Command::Release), Ok(State::Released));
        assert_eq!(
            session.admit(&Command::Guide {
                carriage_m: Some(0.0)
            }),
            Ok(State::Guiding)
        );
        session.state = State::Guiding;
        assert_eq!(
            session.admit(&Command::Release),
            Err(Refusal::WrongState(State::Guiding))
        );
        assert_eq!(
            session.admit(&Command::Connect),
            Err(Refusal::WrongState(State::Guiding))
        );
        let hold = Command::Hold {
            id: "contact-0-left_tip".into(),
            settle_s: 0.5,
            capture_s: 3.0,
        };
        assert_eq!(session.admit(&hold), Ok(State::Recording));
        session.state = State::Recording;
        assert_eq!(
            session.admit(&Command::Stop),
            Err(Refusal::WrongState(State::Recording))
        );
        session.recorded();
        assert_eq!(session.admit(&Command::Stop), Ok(State::Holding));
        session.fault();
        // Recovery from a trip: re-seed to a hold, return the carriage, guide
        // again. Neither Recover nor Carriage is admitted while guiding.
        assert_eq!(session.admit(&Command::Recover), Ok(State::Holding));
        assert_eq!(
            session.admit(&Command::Carriage { carriage_m: 0.002 }),
            Err(Refusal::WrongState(State::Faulted))
        );
        session.state = State::Holding;
        assert_eq!(
            session.admit(&Command::Carriage { carriage_m: 0.002 }),
            Ok(State::Holding)
        );
        assert_eq!(
            session.admit(&Command::Carriage {
                carriage_m: f64::NAN
            }),
            Err(Refusal::Invalid("carriage target must be finite"))
        );
        assert_eq!(
            session.admit(&Command::Recover),
            Err(Refusal::WrongState(State::Holding))
        );
        session.state = State::Guiding;
        assert_eq!(
            session.admit(&Command::Carriage { carriage_m: 0.002 }),
            Err(Refusal::WrongState(State::Guiding))
        );
        session.fault();
        assert_eq!(
            session.admit(&Command::Guide { carriage_m: None }),
            Err(Refusal::WrongState(State::Faulted))
        );
        assert_eq!(session.admit(&Command::Release), Ok(State::Released));
        session.close();
        assert_eq!(
            session.admit(&Command::Release),
            Err(Refusal::WrongState(State::Closed))
        );
    }

    #[test]
    fn joint_step_only_admits_small_slow_single_axis_moves_from_holding() {
        let mut session = Session {
            state: State::Holding,
        };
        assert_eq!(session.admit(&Command::Normalize), Ok(State::Holding));
        let command = Command::JointStep {
            joint_index: 4,
            delta_rad: -0.02,
            speed_rad_s: 0.01,
        };
        assert_eq!(session.admit(&command), Ok(State::Holding));
        for command in [
            Command::JointStep {
                joint_index: 6,
                delta_rad: 0.01,
                speed_rad_s: 0.01,
            },
            Command::JointStep {
                joint_index: 4,
                delta_rad: 0.03,
                speed_rad_s: 0.01,
            },
            Command::JointStep {
                joint_index: 4,
                delta_rad: 0.01,
                speed_rad_s: 0.1,
            },
            Command::JointStep {
                joint_index: 4,
                delta_rad: f64::NAN,
                speed_rad_s: 0.01,
            },
        ] {
            assert!(session.admit(&command).is_err());
        }
        session.state = State::Guiding;
        assert_eq!(
            session.admit(&Command::Normalize),
            Err(Refusal::WrongState(State::Guiding))
        );
        assert_eq!(
            session.admit(&command),
            Err(Refusal::WrongState(State::Guiding))
        );
    }

    #[test]
    fn hold_requests_validate_their_windows_and_ids() {
        let session = Session {
            state: State::Guiding,
        };
        for (id, settle, capture) in [
            ("", 0.5, 3.0),
            ("bad id", 0.5, 3.0),
            ("ok", -1.0, 3.0),
            ("ok", 0.5, 0.1),
            ("ok", 0.5, 31.0),
        ] {
            assert!(matches!(
                session.admit(&Command::Hold {
                    id: id.into(),
                    settle_s: settle,
                    capture_s: capture
                }),
                Err(Refusal::Invalid(_))
            ));
        }
    }

    #[test]
    fn telemetry_record_round_trips_the_shared_layout() {
        let encoded = encode_sample(&sample(7, Mode::HandGuiding, true));
        assert_eq!(encoded.len(), TELEMETRY_RECORD_BYTES);
        assert_eq!(u64::from_le_bytes(encoded[..8].try_into().unwrap()), 7);
        assert_eq!(
            u64::from_le_bytes(encoded[8..16].try_into().unwrap()),
            7_000_000
        );
        assert_eq!(
            u64::from_le_bytes(encoded[16..24].try_into().unwrap()),
            17_500_000
        );
        let position6 = f64::from_le_bytes(encoded[24 + 6 * 8..24 + 7 * 8].try_into().unwrap());
        assert_eq!(position6, 0.002);
        assert_eq!(
            encoded[TELEMETRY_RECORD_BYTES - 6],
            Mode::HandGuiding.code()
        );
        assert_eq!(encoded[TELEMETRY_RECORD_BYTES - 1], 0b01);
        assert_eq!(&telemetry_header()[..8], TELEMETRY_MAGIC);
    }

    #[test]
    fn dynamics_sidecar_matches_tick_and_marks_missing_sdk_channels() {
        let mut tick = sample(7, Mode::Position, false);
        let missing = encode_dynamics_sample(&tick);
        assert_eq!(&dynamics_header()[..8], DYNAMICS_MAGIC);
        assert_eq!(
            u64::from_le_bytes(dynamics_header()[8..].try_into().unwrap()) as usize,
            DYNAMICS_RECORD_BYTES
        );
        assert_eq!(missing[..24], encode_sample(&tick)[..24]);
        assert!(f64::from_le_bytes(missing[24..32].try_into().unwrap()).is_nan());
        tick.measured.dynamics = Some(crate::JointDynamics {
            accelerations: vec![1.0; 7],
            efforts: vec![2.0; 7],
            compensation_efforts: vec![3.0; 7],
        });
        let record = encode_dynamics_sample(&tick);
        for (offset, expected) in [(24, 1.0), (24 + 7 * 8, 2.0), (24 + 14 * 8, 3.0)] {
            assert_eq!(
                f64::from_le_bytes(record[offset..offset + 8].try_into().unwrap()),
                expected
            );
        }
    }

    #[test]
    fn window_summary_flags_unhealthy_short_and_gapped_windows() {
        let good: Vec<_> = (1..=400)
            .map(|t| sample(t, Mode::HandGuiding, true))
            .collect();
        let window = summarize_window("h", &good, 1_000_000, 400_000_000);
        assert!(window.healthy, "{:?}", window.reason);
        assert_eq!(window.ticks, 400);
        assert!(window.joint_motion_max_rad <= 0.0002);
        assert_eq!(window.carriage_median_m, 0.002);
        let short = summarize_window("h", &good[..100], 1_000_000, 100_000_000);
        assert!(!short.healthy);
        let mut mixed = good.clone();
        mixed[200].measured.mode = Mode::Position;
        assert!(!summarize_window("h", &mixed, 1_000_000, 400_000_000).healthy);
        let mut stopped = good.clone();
        stopped[200].estop = crate::estop::FAULT;
        assert!(!summarize_window("h", &stopped, 1_000_000, 400_000_000).healthy);
        let mut gapped = good.clone();
        gapped.remove(200);
        for s in &mut gapped[200..] {
            s.since_start += Duration::from_millis(30);
        }
        let window = summarize_window("h", &gapped, 1_000_000, 400_000_000);
        assert!(!window.healthy && window.max_gap_ns > 25_000_000);
    }
}
