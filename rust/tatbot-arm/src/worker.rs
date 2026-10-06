//! One backend owner per arm. A bounded command mailbox cannot stall heartbeat
//! checks; results use a bounded latest-value cell, never a blocking publisher.
use crate::{
    ArmBackend, Control, Error, Measured, OfflineArmBackend, Result, Sample, estop,
    motion_guard::MotionGuard,
    offline_clock::{CONTROL_PERIOD, Clock, OfflineClock},
};
use std::sync::{
    Arc, Mutex,
    atomic::{AtomicBool, AtomicU8, AtomicUsize, Ordering},
    mpsc::{self, SyncSender, TryRecvError},
};
use std::thread::{self, JoinHandle};
use std::time::{Duration, Instant};

/// One-use authorization for a prefix of one continuous stream. The caller
/// grants only after graph guards pass. A missed boundary faults; it never waits
/// with a resumable queue or silently inserts a hold into a planned trajectory.
enum GateDecision {
    Grant(usize),
    Hold,
}
type PhaseGate = Box<dyn FnMut(usize, &Status) -> Result<GateDecision> + Send>;
type StartGuard = Box<dyn Fn(&Status) -> Result<()> + Send + Sync>;

/// The most samples one stream or permit may carry: the planner's tick budget
/// (`planner.max_ticks` in config/motion_constants.json, 625 s at 400 Hz). Every
/// task or hover plan passes `path_plan_check` before dispatch. E-stop,
/// contact and joint watchdogs bound it during execution. The former
/// 24000 (60 s) refused the native scan's 66 s first leg after every guard had
/// passed (run 9616).
pub const MAX_STREAM_SAMPLES: usize = 250_000;
/// While the six rotary joints are in zero-commanded external effort the
/// measured-velocity guard only bounds what a hand can do to the arm; the
/// profile limit tuned for commanded motion tripped on a wrist flick at
/// 2.01 rad/s. A dropped arm still crosses twice that limit within a fraction
/// of a radian, so the trip remains a protective catch.
pub const HAND_GUIDING_VELOCITY_FACTOR: f64 = 2.0;
/// Attended carriage moves: at most 50 mm/s, verified against the target
/// within this tolerance after the goal time.
pub const CARRIAGE_MOVE_TOLERANCE_M: f64 = 0.0005;
/// The bound every timed carriage move honours (`trossen::move_carriage`
/// refuses faster): a 32 mm return to rest takes 0.64 s at this rate, and
/// the 3 s Sleep phase moves it at a third of it.
pub const CARRIAGE_MOVE_MAX_M_PER_S: f64 = 0.05;

pub struct StreamPermit {
    total: usize,
    granted: AtomicUsize,
    claimed: AtomicBool,
    gate: Option<Mutex<PhaseGate>>,
    start_guard: Option<StartGuard>,
    start_guard_refused: AtomicBool,
    stopped_at: AtomicUsize,
    check_every_sample: bool,
    starts_at: Option<Instant>,
}
impl StreamPermit {
    pub fn new(total: usize) -> Result<Arc<Self>> {
        if total == 0 || total > MAX_STREAM_SAMPLES {
            return Err(Error("stream permit length".into()));
        }
        Ok(Arc::new(Self {
            total,
            granted: AtomicUsize::new(0),
            claimed: AtomicBool::new(false),
            gate: None,
            start_guard: None,
            start_guard_refused: AtomicBool::new(false),
            stopped_at: AtomicUsize::new(usize::MAX),
            check_every_sample: false,
            starts_at: None,
        }))
    }
    /// The owner invokes this bounded, nonblocking pure guard at a prefix edge.
    /// It must not perform I/O, wait for a journal, or authorize a later edge.
    pub fn with_phase_gate(
        total: usize,
        mut gate: impl FnMut(usize, &Status) -> Result<usize> + Send + 'static,
    ) -> Result<Arc<Self>> {
        if total == 0 || total > MAX_STREAM_SAMPLES {
            return Err(Error("stream permit length".into()));
        }
        Ok(Arc::new(Self {
            total,
            granted: AtomicUsize::new(0),
            claimed: AtomicBool::new(false),
            gate: Some(Mutex::new(Box::new(move |index, status| {
                gate(index, status).map(GateDecision::Grant)
            }))),
            start_guard: None,
            start_guard_refused: AtomicBool::new(false),
            stopped_at: AtomicUsize::new(usize::MAX),
            check_every_sample: false,
            starts_at: None,
        }))
    }
    /// Evaluate a pure measured stop condition before every sample. A true
    /// result discards the remaining queue and holds measured state. The owner
    /// publishes the stopping index only after that hold succeeds. This does
    /// not classify contact; the caller retains its own measured evidence.
    pub fn with_stop_guard(
        total: usize,
        guard: impl FnMut(usize, &Status) -> Result<bool> + Send + 'static,
    ) -> Result<Arc<Self>> {
        Self::scheduled_stop_guard(total, None, guard)
    }
    /// A shared monotonic start for preloaded streams. Only the initial sample
    /// waits; later missed permits still fault instead of pausing trajectories.
    pub fn scheduled_stop_guard(
        total: usize,
        starts_at: Option<Instant>,
        mut guard: impl FnMut(usize, &Status) -> Result<bool> + Send + 'static,
    ) -> Result<Arc<Self>> {
        let now = Instant::now();
        if starts_at.is_some_and(|start| {
            start <= now || start.duration_since(now) > Duration::from_secs(30)
        }) {
            return Err(Error(
                "scheduled stream start must be within the next 30 seconds".into(),
            ));
        }
        if total == 0 || total > MAX_STREAM_SAMPLES {
            return Err(Error("stream permit length".into()));
        }
        Ok(Arc::new(Self {
            total,
            granted: AtomicUsize::new(0),
            claimed: AtomicBool::new(false),
            gate: Some(Mutex::new(Box::new(move |index, status| {
                Ok(if guard(index, status)? {
                    GateDecision::Hold
                } else {
                    GateDecision::Grant(index + 1)
                })
            }))),
            start_guard: None,
            start_guard_refused: AtomicBool::new(false),
            stopped_at: AtomicUsize::new(usize::MAX),
            check_every_sample: true,
            starts_at,
        }))
    }
    pub fn stopped_at(&self) -> Option<usize> {
        let index = self.stopped_at.load(Ordering::Acquire);
        (index != usize::MAX).then_some(index)
    }
    /// Bind a one-shot, bounded, nonblocking validity check before sharing the
    /// permit. The worker evaluates it at sample zero, even when the initial
    /// approach prefix was already granted by the session owner.
    pub fn bind_start_guard(
        permit: &mut Arc<Self>,
        guard: impl Fn(&Status) -> Result<()> + Send + Sync + 'static,
    ) -> Result<()> {
        let unique = Arc::get_mut(permit)
            .ok_or_else(|| Error("stream start guard requires unshared permit".into()))?;
        if unique.start_guard.is_some() || unique.claimed.load(Ordering::Acquire) {
            return Err(Error("stream start guard already bound or claimed".into()));
        }
        unique.start_guard = Some(Box::new(guard));
        Ok(())
    }
    fn authorize(&self, index: usize, status: &Status) -> Result<bool> {
        if index == 0
            && let Some(guard) = &self.start_guard
            && let Err(error) = guard(status)
        {
            self.start_guard_refused.store(true, Ordering::Release);
            return Err(error);
        }
        if !self.check_every_sample && index < self.granted.load(Ordering::Acquire) {
            return Ok(true);
        }
        if let Some(gate) = &self.gate {
            let mut gate = gate
                .try_lock()
                .map_err(|_| Error("phase gate busy".into()))?;
            let next = match gate(index, status)? {
                GateDecision::Grant(next) => next,
                GateDecision::Hold => return Ok(false),
            };
            if next <= index {
                return Err(Error("phase gate did not authorize next prefix".into()));
            }
            self.grant(next)?;
        }
        Ok(true)
    }
    pub fn grant(&self, prefix: usize) -> Result<()> {
        if prefix > self.total {
            return Err(Error("stream permit exceeds plan".into()));
        }
        self.granted
            .fetch_update(Ordering::Release, Ordering::Relaxed, |old| {
                (prefix >= old).then_some(prefix)
            })
            .map_err(|_| Error("stream permit cannot move backward".into()))?;
        Ok(())
    }
}
#[derive(Clone, Copy)]
enum TimedHold {
    Measured(Instant),
    Commanded(Instant),
}
impl TimedHold {
    fn due(self, now: Instant) -> bool {
        match self {
            Self::Measured(until) | Self::Commanded(until) => now >= until,
        }
    }
    fn finish<B: ArmBackend>(self, backend: &mut B) -> Result<()> {
        match self {
            Self::Measured(_) => backend.hold(),
            Self::Commanded(_) => backend.finish_joint_stream(),
        }
    }
}
pub enum Primitive {
    TripRetract,
    Reconnect(crate::ArmConfig),
    /// Load the frozen controller profile while idle, before attended takeover.
    LoadConfig(std::path::PathBuf),
    ClearError,
    Reseed,
    Takeover,
    RecoveryMove(crate::recovery::Phase),
    PrepareRecovery {
        staged: [f64; 7],
    },
    /// Release only after landing; ordinary motion requires a new measured re-seed.
    Idle,
    Inject(crate::Faults),
    Stream(Vec<Sample>),
    StreamJoint {
        seed: [f64; 7],
        expected_sequence: u64,
        permit: Option<Arc<StreamPermit>>,
        samples: Vec<crate::JointSample>,
    },
    /// Air-only carriage positioning with all rotary targets held fixed.
    /// The session must separately admit the complete tool sweep in free air.
    StreamCarriage {
        seed: [f64; 7],
        expected_sequence: u64,
        samples: Vec<crate::JointSample>,
    },
    MoveJ {
        target: Vec<f64>,
        goal_time_s: f64,
    },
    Hold,
    Freeze,
    /// Six rotary joints to zero-commanded external effort, carriage held in
    /// position control at the confirmed measured datum. Requires a healthy
    /// position-mode arm with an armed carriage contact baseline.
    HandGuide {
        carriage_m: f64,
    },
    /// Leave hand guiding: measured hold, then fresh position-mode readback.
    GuideStop,
    /// Timed carriage-only move in position control (rotary targets held at
    /// their measured values), verified by readback after the goal time.
    /// Refused while hand guiding or latched.
    CarriageTo {
        target_m: f64,
        goal_time_s: f64,
    },
    /// Read the configured end-effector payload into `Status::payload`.
    ReadPayload,
}
enum QueuedSample {
    Angular(Sample),
    Joint(crate::JointSample, Option<Arc<StreamPermit>>, usize, bool),
}
impl QueuedSample {
    fn joints(&self) -> Vec<f64> {
        match self {
            Self::Angular(s) => s.joints.clone(),
            Self::Joint(s, ..) => s.positions[..6].to_vec(),
        }
    }
    fn execute<B: ArmBackend>(&self, control: &mut Control<B>) -> Result<()> {
        match self {
            Self::Angular(s) => control.stream(s),
            Self::Joint(s, permit, index, transfer) => {
                if permit
                    .as_ref()
                    .is_some_and(|p| *index >= p.granted.load(Ordering::Acquire))
                {
                    return Err(Error(format!(
                        "stream phase not authorized before sample {index}"
                    )));
                }
                if *transfer {
                    control.stream_carriage(s)
                } else {
                    control.stream_joint(s)
                }
            }
        }
    }
}
#[derive(Clone, Debug, serde::Serialize)]
pub struct StreamSeedReceipt {
    pub requested: [f64; 7],
    pub measured: Option<Measured>,
    pub expected_sequence: u64,
    pub previous_sequence: u64,
    pub accepted: bool,
    pub error: Option<String>,
}
/// Counts successful backend submissions, not measured arrival. The sequence
/// binds the counter to one request; a stopped stream never credits its tail.
#[derive(Clone, Debug, serde::Serialize)]
pub struct StreamProgress {
    pub sequence: u64,
    pub total: usize,
    pub submitted: usize,
    pub last_pen: Option<bool>,
    /// Deadline misses only while this stream is executing, including its
    /// final submission and hold. Startup and recovery are counted separately.
    pub late_ticks: u64,
    pub max_late_us: u64,
}
/// An invalid material target stops the stream inside its backend owner. This
/// receipt is retained in Status for the session journal to persist off the
/// control thread; no stroke completion or resumption is implied.
#[derive(Clone, Copy, Debug, PartialEq, Eq, serde::Serialize)]
#[serde(rename_all = "snake_case")]
pub enum TargetAbortOutcome {
    Held,
    Retracted,
    Uncertain,
}
#[derive(Clone, Debug, serde::Serialize)]
pub struct TargetAbortReceipt {
    pub schema: &'static str,
    pub kind: &'static str,
    pub sequence: u64,
    pub stream_progress: Option<StreamProgress>,
    pub requested_wall_ns: u64,
    pub completed_wall_ns: Option<u64>,
    pub outcome: TargetAbortOutcome,
    pub hold_confirmed: bool,
    pub retract_commanded: bool,
    pub contact_armed: bool,
    pub contact_assessable: bool,
    pub contact_age_ms: Option<f64>,
    pub stop_measured: Option<Measured>,
    pub final_measured: Option<Measured>,
    pub reason: String,
    pub motion_authority: bool,
}
/// What the hand-guiding entry and exit actually did, from fresh readbacks.
#[derive(Clone, Debug, serde::Serialize)]
pub struct GuideReceipt {
    pub carriage_datum_m: f64,
    pub entered_wall_ns: u64,
    pub entered_measured: Measured,
    pub stopped_wall_ns: Option<u64>,
    /// The measured pose submitted as the position target on stop.
    pub stop_submitted: Option<[f64; 7]>,
    pub stop_measured: Option<Measured>,
    pub stop_error: Option<String>,
}
/// One control tick's feedback, forwarded to a recorder off the owner thread.
#[derive(Clone, Debug)]
pub struct TelemetrySample {
    pub tick: u64,
    pub wall_ns: u64,
    pub since_start: Duration,
    pub measured: Measured,
    pub estop: i32,
    pub guiding: bool,
    pub latched: bool,
}
#[derive(Clone, Debug, Default)]
pub struct Status {
    /// The owner believes the six rotary joints are in hand guiding.
    pub guiding: bool,
    pub guide: Option<GuideReceipt>,
    /// Telemetry samples the recorder could not accept in time.
    pub telemetry_dropped: u64,
    /// The controller's configured end-effector payload, when read.
    pub payload: Option<serde_json::Value>,
    /// Freshness uses the worker's source clock. Live hardware always uses
    /// monotonic host time; offline world time pauses between requested steps.
    pub clock: Clock,
    /// Mock fault-injection acknowledgement, retained independently of the
    /// controller fault it deliberately causes. Reset on the next request.
    pub injection_result: Option<std::result::Result<(), String>>,
    pub contact: Option<crate::contact::ContactEvaluation>,
    pub contact_at: Option<Instant>,
    pub watchdog_ok: bool,
    pub retract_verified: Option<bool>,
    /// One-shot target-invalid abort outcome. The session persists this
    /// serializable receipt after observing it; the worker performs no I/O.
    pub target_abort: Option<TargetAbortReceipt>,
    /// What the backend's last connect waited for (`Reconnect`, and the
    /// reconnect inside `ClearError`); the session journals it.
    pub connect_wait: Option<crate::ConnectWait>,
    pub recovery_targets: Option<crate::recovery::Targets>,
    pub stream_seed: Option<StreamSeedReceipt>,
    pub stream_progress: Option<StreamProgress>,
    /// The one-shot sample-zero validity check refused after queue admission.
    /// This is an explicit owner signal, separate from ordinary phase faults.
    pub stream_start_guard_refused: bool,
    pub commands: u64,
    pub sequence: u64,
    pub completed: bool,
    pub fault: Option<String>,
    /// The exact worker fault for which a recovery class was established.
    /// Replacing the fault text (for example with a failed retract) invalidates
    /// this class rather than lending the old trip's retry policy to a new fault.
    pub recovery_fault: Option<(String, RecoveryFaultCategory)>,
    pub measured: Option<Measured>,
    /// Host receipt time of this measured sample, never the subscriber poll time.
    pub measured_wall_ns: u64,
    /// Start of the SDK read corresponding to measured_wall_ns.
    pub measured_started_wall_ns: u64,
    /// Offline world timestamp; absent on live workers. Host receipt timestamps
    /// above remain wall time and must not be used as simulation source time.
    pub measured_simulation_ns: Option<u64>,
    /// Local freshness cannot be extended by a wall-clock adjustment.
    pub measured_at: Option<Instant>,
    /// Control ticks that started after their 2.5 ms slot. A stream is paced
    /// by the loop, so late ticks stretch it; the runtime judges completion by
    /// progress and retains these counts as evidence of a loop that could not
    /// keep the period (an unoptimised build on the arm node did not).
    pub late_ticks: u64,
    pub max_late_us: u64,
}
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RecoveryFaultCategory {
    Trip,
}
impl Status {
    pub fn recovery_category(&self) -> Option<&'static str> {
        match self.recovery_fault.as_ref() {
            Some((message, RecoveryFaultCategory::Trip))
                if self.fault.as_ref() == Some(message) && self.retract_verified != Some(false) =>
            {
                Some("trip")
            }
            _ => None,
        }
    }
    fn record_lateness(&mut self, late_us: u64, joint_stream_tick: bool) {
        self.late_ticks += 1;
        self.max_late_us = self.max_late_us.max(late_us);
        if joint_stream_tick && let Some(progress) = &mut self.stream_progress {
            progress.late_ticks += 1;
            progress.max_late_us = progress.max_late_us.max(late_us);
        }
    }
    /// A missing, unarmed, unassessable or stale contact meter never grants a
    /// contact transition. This is carriage evidence, not a calibrated pen-tip
    /// contact classification or proof of fixture contact.
    pub fn contact_is_valid(&self, max_age: Duration) -> bool {
        self.fault.is_none()
            && self.measurement_is_fresh(max_age)
            && self.contact.is_some_and(|c| c.armed && c.assessable)
            && self
                .contact_at
                .and_then(|stamp| self.clock.now().checked_duration_since(stamp))
                .is_some_and(|age| age <= max_age)
    }
    pub fn measurement_is_fresh(&self, max_age: Duration) -> bool {
        self.clock.running()
            && self.measured.is_some()
            && self
                .measured_at
                .and_then(|stamp| self.clock.now().checked_duration_since(stamp))
                .is_some_and(|age| age <= max_age)
    }
    /// Why `contact_is_valid(max_age)` does not hold, naming the channel, or
    /// None when it holds. A consumer that refuses on staleness says this
    /// instead of "stale": on 2026-09-18 a paper search faulted at 49 mm with
    /// every measurement under 17 ms old, and the fault did not say that it
    /// was the contact estimate that had lapsed.
    pub fn staleness(&self, max_age: Duration) -> Option<String> {
        let limit_ms = max_age.as_secs_f64() * 1e3;
        let age_ms = |stamp: Option<Instant>| {
            stamp
                .and_then(|stamp| self.clock.now().checked_duration_since(stamp))
                .map(|age| age.as_secs_f64() * 1e3)
        };
        if let Some(fault) = &self.fault {
            return Some(format!("worker fault: {fault}"));
        }
        if !self.clock.running() {
            return Some("worker clock not running".into());
        }
        if self.measured.is_none() {
            return Some("no measurement yet".into());
        }
        match age_ms(self.measured_at) {
            None => return Some("measurement receipt time missing or in the future".into()),
            Some(age) if age > limit_ms => {
                return Some(format!(
                    "measurement {age:.1} ms old, limit {limit_ms:.0} ms"
                ));
            }
            Some(_) => {}
        }
        let Some(contact) = self.contact else {
            return Some("no contact meter".into());
        };
        if !contact.armed {
            return Some("contact meter not armed (baseline re-arming)".into());
        }
        if !contact.assessable {
            return Some("contact meter not assessable (arm moving)".into());
        }
        match age_ms(self.contact_at) {
            None => Some("contact estimate receipt time missing or in the future".into()),
            Some(age) if age > limit_ms => Some(format!(
                "contact estimate {age:.1} ms old, limit {limit_ms:.0} ms"
            )),
            Some(_) => None,
        }
    }
}
struct Request {
    sequence: u64,
    primitive: Primitive,
    joint_queue: Option<std::collections::VecDeque<QueuedSample>>,
}
pub struct Worker {
    pub software_stop: Arc<AtomicBool>,
    target_abort: Arc<AtomicU8>,
    sender: SyncSender<Request>,
    status: Arc<Mutex<Status>>,
    shutdown: Arc<AtomicBool>,
    idle_on_shutdown: Arc<AtomicBool>,
    shutdown_receipt: Arc<Mutex<Option<Result<crate::Measured>>>>,
    thread: Option<JoinHandle<()>>,
    sequence: u64,
    native_shutdown_timeout: Option<Duration>,
    clock: Clock,
}
impl Worker {
    pub fn spawn<B: ArmBackend + Send + 'static>(
        control: Control<B>,
        period: Duration,
        guard: MotionGuard,
    ) -> Result<Self> {
        Self::spawn_with(move || Ok(control), period, guard)
    }

    /// Construct and destroy a backend on its sole owning thread. Vendor
    /// objects need not be Send, and never receive an unsafe Send promise.
    /// Construction failures are published as completed faults; no tick runs.
    /// Construction must not arm motion: the tick-level protections begin
    /// only after it returns. Connection timeouts remain the factory's duty.
    pub fn spawn_with<B, F>(construct: F, period: Duration, guard: MotionGuard) -> Result<Self>
    where
        B: ArmBackend + 'static,
        F: FnOnce() -> Result<Control<B>> + Send + 'static,
    {
        Self::spawn_timed(
            construct,
            period,
            guard,
            Clock::Live,
            None,
            Arc::new(AtomicBool::new(false)),
            None,
        )
    }

    /// A live worker that forwards every tick's feedback to `telemetry` with
    /// a non-blocking send: the hand-guiding owner records it off this thread.
    pub fn spawn_recording<B, F>(
        construct: F,
        guard: MotionGuard,
        telemetry: SyncSender<TelemetrySample>,
    ) -> Result<Self>
    where
        B: ArmBackend + 'static,
        F: FnOnce() -> Result<Control<B>> + Send + 'static,
    {
        Self::spawn_timed(
            construct,
            CONTROL_PERIOD,
            guard,
            Clock::Live,
            None,
            Arc::new(AtomicBool::new(false)),
            Some(telemetry),
        )
    }

    /// Run the same guards and commands against one explicitly offline world.
    /// Its caller grants 2.5 ms ticks through the clock and receives an
    /// acknowledgement only after world advancement and feedback publication.
    /// Hardware adapters cannot enter this path through `ArmBackend` alone.
    pub fn spawn_offline_with<B, F>(
        construct: F,
        clock: Arc<OfflineClock>,
        guard: MotionGuard,
    ) -> Result<Self>
    where
        B: OfflineArmBackend + 'static,
        F: FnOnce() -> Result<Control<B>> + Send + 'static,
    {
        Self::spawn_offline_with_stop(construct, clock, guard, Arc::new(AtomicBool::new(false)))
    }

    /// Retain the launcher's stop token while replacing its unstarted mock
    /// worker with an offline world. Signals can cancel world construction and
    /// still stop the selected worker after construction succeeds.
    pub fn spawn_offline_with_stop<B, F>(
        construct: F,
        clock: Arc<OfflineClock>,
        guard: MotionGuard,
        software_stop: Arc<AtomicBool>,
    ) -> Result<Self>
    where
        B: OfflineArmBackend + 'static,
        F: FnOnce() -> Result<Control<B>> + Send + 'static,
    {
        clock.claim_worker()?;
        Self::spawn_timed(
            construct,
            CONTROL_PERIOD,
            guard,
            Clock::Offline(clock),
            Some(B::advance_world),
            software_stop,
            None,
        )
    }

    fn spawn_timed<B, F>(
        construct: F,
        period: Duration,
        mut guard: MotionGuard,
        clock: Clock,
        advance_world: Option<fn(&mut B, Duration) -> Result<()>>,
        software_stop: Arc<AtomicBool>,
        telemetry: Option<SyncSender<TelemetrySample>>,
    ) -> Result<Self>
    where
        B: ArmBackend + 'static,
        F: FnOnce() -> Result<Control<B>> + Send + 'static,
    {
        if period != CONTROL_PERIOD {
            return Err(Error(
                "control period must be 2.5ms for the carried contact-cap contract".into(),
            ));
        }
        let (sender, receiver) = mpsc::sync_channel::<Request>(1);
        let status = Arc::new(Mutex::new(Status {
            guiding: false,
            guide: None,
            telemetry_dropped: 0,
            payload: None,
            clock: clock.clone(),
            injection_result: None,
            contact: None,
            contact_at: None,
            watchdog_ok: false,
            retract_verified: None,
            target_abort: None,
            connect_wait: None,
            recovery_targets: None,
            stream_seed: None,
            stream_progress: None,
            stream_start_guard_refused: false,
            commands: 0,
            sequence: 0,
            completed: false,
            fault: None,
            recovery_fault: None,
            measured: None,
            measured_wall_ns: 0,
            measured_started_wall_ns: 0,
            measured_simulation_ns: None,
            measured_at: None,
            late_ticks: 0,
            max_late_us: 0,
        }));
        let published = status.clone();
        let shutdown = Arc::new(AtomicBool::new(false));
        let idle_on_shutdown = Arc::new(AtomicBool::new(false));
        let terminal_idle = idle_on_shutdown.clone();
        let shutdown_receipt = Arc::new(Mutex::new(None));
        let terminal_receipt = shutdown_receipt.clone();
        let stop = shutdown.clone();
        let emergency = software_stop.clone();
        let target_abort = Arc::new(AtomicU8::new(0));
        let target_abort_requested = target_abort.clone();
        let owner_clock = clock.clone();
        let thread = thread::spawn(move || {
            let clock = owner_clock;
            let _offline_owner = match &clock {
                Clock::Offline(clock) => Some(clock.owner_guard()),
                Clock::Live => None,
            };
            let mut control = match construct() {
                Ok(control) => control,
                Err(error) => {
                    let mut state = published.lock().unwrap();
                    state.fault = Some(format!("backend construction: {error}"));
                    state.completed = true;
                    state.watchdog_ok = false;
                    if let Clock::Offline(clock) = &clock {
                        clock.close(state.fault.as_ref().unwrap().clone());
                    }
                    return;
                }
            };
            let start = clock.now();
            let mut next = clock.now();
            let mut pending = std::collections::VecDeque::new();
            let mut state = Status {
                guiding: false,
                guide: None,
                telemetry_dropped: 0,
                payload: None,
                clock: clock.clone(),
                injection_result: None,
                contact: None,
                contact_at: None,
                watchdog_ok: true,
                retract_verified: None,
                target_abort: None,
                connect_wait: None,
                recovery_targets: None,
                stream_seed: None,
                stream_progress: None,
                stream_start_guard_refused: false,
                commands: 0,
                sequence: 0,
                completed: true,
                fault: None,
                recovery_fault: None,
                measured: None,
                measured_wall_ns: 0,
                measured_started_wall_ns: 0,
                measured_simulation_ns: None,
                measured_at: None,
                late_ticks: 0,
                max_late_us: 0,
            };
            let mut latched = false;
            let mut emergency_seen = false;
            let mut parked = false;
            let mut watchdog = crate::tracking_watchdog::TrackingWatchdog::default();
            let mut commanded: Option<Vec<f64>> = None;
            let mut contact = crate::contact::ContactCap::default();
            let mut retract_started: Option<Instant> = None;
            let mut carriage_move: Option<(Instant, f64, f64)> = None;
            // An air carriage transfer (`StreamCarriage`: rotary targets held,
            // the free-air sweep admitted by the session with a clearance
            // receipt) is judged by its deflection screen only, like the
            // hand-guide's timed carriage move: at 1 mm/s the carriage's own
            // stick-slip reads 17-24 N against a baseline armed elsewhere,
            // the cap's own magnitude, with no displacement (seven transfers,
            // 2026-09-13 to 09-16). Half a second after its last sample the
            // baseline re-arms where the carriage now rests.
            let mut transfer_until: Option<Instant> = None;
            let mut motion_until: Option<TimedHold> = None;
            let mut guiding = false;
            let base_velocity_limit = guard.velocity_limit;
            // The carriage's effort channel counts as contact, and a trip
            // retracts along it, only for a contact tool on a qualified
            // carriage; a standoff tool or an unqualified carriage keeps the
            // deflection screen and holds on a trip.
            let carriage_contact = control.backend.contact_policy()
                == crate::ContactPolicy::Contact
                && control.backend.carriage_qualified();
            let mut feedback = FeedbackChange::default();
            let mut tick: u64 = 0;
            while !stop.load(Ordering::Acquire) {
                tick += 1;
                if let Clock::Offline(clock) = &clock
                    && !clock.wait_tick(&stop, || {
                        let active = control.estop.load(Ordering::SeqCst) != estop::OK
                            || emergency.load(Ordering::Acquire)
                            || target_abort_requested.load(Ordering::Acquire) == 1;
                        if !active {
                            emergency_seen = false;
                        }
                        active && !emergency_seen
                    })
                {
                    break;
                }
                let target_abort_now = target_abort_requested
                    .compare_exchange(1, 2, Ordering::AcqRel, Ordering::Acquire)
                    .is_ok();
                let mut joint_stream_tick = state.stream_progress.is_some() && !state.completed;
                // No FSM, journal, bus or viewer code executes on this thread.
                emergency_seen = control.estop.load(Ordering::SeqCst) != estop::OK
                    || emergency.load(Ordering::Acquire);
                if emergency_seen {
                    if let Some(receipt) = &mut state.target_abort
                        && receipt.retract_commanded
                        && receipt.completed_wall_ns.is_none()
                    {
                        receipt.outcome = TargetAbortOutcome::Uncertain;
                        receipt.completed_wall_ns = Some(wall_ns());
                        receipt.reason =
                            "physical or software stop interrupted target retract".into();
                    }
                    parked = false;
                    pending.clear();
                    retract_started = None;
                    carriage_move = None;
                    motion_until = None;
                    latched = true;
                    let hold = control.backend.hold();
                    if hold.is_ok() {
                        guiding = false;
                    }
                    state.fault = Some(match hold {
                        Ok(()) => "e-stop".into(),
                        Err(e) => format!("e-stop hold failed: {e}"),
                    });
                    state.completed = true;
                }
                if target_abort_now {
                    let prior_latch = latched;
                    let prior_retract = retract_started.take().is_some();
                    let contact_age_ms = state
                        .contact_at
                        .and_then(|at| clock.now().checked_duration_since(at))
                        .map(|age| age.as_secs_f64() * 1e3);
                    let contact_armed = state.contact.is_some_and(|c| c.armed);
                    let contact_assessable = state.contact.is_some_and(|c| c.assessable);
                    let prior_measurement_fresh =
                        state.measurement_is_fresh(Duration::from_millis(20));
                    let pen_submitted =
                        state.stream_progress.as_ref().and_then(|p| p.last_pen) == Some(true);
                    let active_stream = !state.completed
                        && state
                            .stream_progress
                            .as_ref()
                            .is_some_and(|p| p.submitted > 0);
                    let mut receipt = TargetAbortReceipt {
                        schema: "tatbot.receipt/1",
                        kind: "target-abort",
                        sequence: state.sequence,
                        stream_progress: state.stream_progress.clone(),
                        requested_wall_ns: wall_ns(),
                        completed_wall_ns: None,
                        outcome: TargetAbortOutcome::Uncertain,
                        hold_confirmed: false,
                        retract_commanded: false,
                        contact_armed,
                        contact_assessable,
                        contact_age_ms,
                        stop_measured: None,
                        final_measured: None,
                        reason: "target abort pending measured hold".into(),
                        motion_authority: false,
                    };
                    pending.clear();
                    motion_until = None;
                    carriage_move = None;
                    commanded = None;
                    watchdog.reset();
                    latched = true;
                    parked = false;
                    state.completed = true;
                    state.retract_verified = None;
                    // The higher-priority stop already issued its hold above.
                    // Never add a controller query or command to that path.
                    let hold = (!emergency_seen).then(|| control.backend.hold());
                    let stopped = hold
                        .as_ref()
                        .filter(|result| result.is_ok())
                        .and_then(|_| control.backend.measured_fresh().ok());
                    let healthy = stopped.as_ref().is_some_and(|m| {
                        m.mode == crate::Mode::Position
                            && m.error.is_empty()
                            && m.joints.len() == 6
                            && m.joints.iter().all(|q| q.is_finite())
                            && m.velocities.len() >= 6
                            && m.velocities[..6]
                                .iter()
                                .all(|speed| speed.is_finite() && speed.abs() < 0.3)
                            && m.carriage.as_ref().is_some_and(|c| {
                                c.position_m.is_finite()
                                    && c.target_m.is_some_and(f64::is_finite)
                                    && c.effort_n.is_finite()
                            })
                    });
                    receipt.hold_confirmed = hold.as_ref().is_some_and(Result::is_ok) && healthy;
                    receipt.stop_measured = stopped.clone();
                    let basic_retract_ready = !emergency_seen
                        && !prior_latch
                        && !prior_retract
                        && receipt.hold_confirmed
                        && carriage_contact
                        && active_stream
                        && pen_submitted
                        && prior_measurement_fresh
                        && contact_armed
                        && contact_age_ms.is_some_and(|age| age <= 20.0);
                    let carriage_in_limits = basic_retract_ready
                        && stopped
                            .as_ref()
                            .and_then(|m| m.carriage.as_ref())
                            .zip(control.backend.recovery_limits().ok())
                            .is_some_and(|(carriage, limits)| {
                                let limit = limits[6];
                                carriage.position_m >= limit.min
                                    && carriage.position_m <= limit.max
                                    && (limit.min..=limit.max).contains(&0.032)
                                    && carriage.position_m <= 0.032
                            });
                    // A backend hold, fresh readback, or live-limit query may
                    // block. The pre-stop evidence must still be current at
                    // the actual retract decision, not only at abort entry.
                    let decision_contact_age_ms = state
                        .contact_at
                        .and_then(|at| clock.now().checked_duration_since(at))
                        .map(|age| age.as_secs_f64() * 1e3);
                    receipt.contact_age_ms = decision_contact_age_ms;
                    let feedback_still_fresh = state
                        .measurement_is_fresh(Duration::from_millis(20))
                        && decision_contact_age_ms.is_some_and(|age| age <= 20.0);
                    let stop_now = control.estop.load(Ordering::SeqCst) != estop::OK
                        || emergency.load(Ordering::Acquire);
                    let refusal = if emergency_seen || stop_now {
                        Some("physical or software stop has priority")
                    } else if prior_latch || prior_retract {
                        Some("another fault or retract was already latched")
                    } else if !receipt.hold_confirmed {
                        Some("measured position hold is unverified")
                    } else if !carriage_contact {
                        Some("contact carriage retract is unqualified")
                    } else if !active_stream || !pen_submitted {
                        Some("no active submitted pen-contact stream")
                    } else if !prior_measurement_fresh {
                        Some("pre-stop controller feedback is stale")
                    } else if !contact_armed || contact_age_ms.is_none_or(|age| age > 20.0) {
                        Some("armed contact feedback is missing or stale")
                    } else if !carriage_in_limits {
                        Some("carriage feedback outside live recovery limits")
                    } else if !feedback_still_fresh {
                        Some("contact or controller feedback aged during target abort")
                    } else {
                        None
                    };
                    if let Some(reason) = refusal {
                        receipt.outcome = if receipt.hold_confirmed
                            && !prior_retract
                            && !emergency_seen
                            && !stop_now
                        {
                            TargetAbortOutcome::Held
                        } else {
                            TargetAbortOutcome::Uncertain
                        };
                        receipt.completed_wall_ns = Some(wall_ns());
                        receipt.final_measured = stopped;
                        receipt.reason = reason.into();
                    } else {
                        // The fixed trip-retract command is already qualified by
                        // this backend. The worker verifies its readback below;
                        // neither the caller nor the session clears a stop.
                        if control.estop.load(Ordering::SeqCst) != estop::OK
                            || emergency.load(Ordering::Acquire)
                        {
                            receipt.completed_wall_ns = Some(wall_ns());
                            receipt.reason = "stop arrived before target retract".into();
                        } else {
                            match control.backend.retract_carriage(0.032, 0.6) {
                                Ok(()) => {
                                    retract_started = Some(clock.now());
                                    state.completed = false;
                                    receipt.retract_commanded = true;
                                    receipt.reason =
                                        "retract commanded; awaiting measured readback".into();
                                }
                                Err(error) => {
                                    state.retract_verified = Some(false);
                                    receipt.completed_wall_ns = Some(wall_ns());
                                    receipt.reason =
                                        format!("target retract command failed: {error}");
                                }
                            }
                        }
                    }
                    if !emergency_seen {
                        state.fault = Some(format!("target invalid; {}", receipt.reason));
                    }
                    state.target_abort = Some(receipt);
                }
                let measured_started_wall_ns = wall_ns();
                match control.backend.measured() {
                    Ok(m) => {
                        if parked && (m.mode != crate::Mode::Idle || !m.error.is_empty()) {
                            parked = false;
                            state.fault =
                                Some(format!("idle mode lost; hold={:?}", control.backend.hold()));
                        }
                        let expected_mode = if guiding {
                            crate::Mode::HandGuiding
                        } else {
                            crate::Mode::Position
                        };
                        // Measured joints may rest inside the controller's
                        // feedback tolerance past the nominal envelope (a wrist
                        // rolled to 3.148 rad against a pi bound while guiding).
                        let measured_envelope =
                            control.envelope + crate::FEEDBACK_POSITION_TOLERANCE_RAD;
                        if !latched
                            && (m.mode != expected_mode
                                || !m.error.is_empty()
                                || m.joints.is_empty()
                                || m.joints
                                    .iter()
                                    .any(|q| !q.is_finite() || q.abs() > measured_envelope))
                        {
                            pending.clear();
                            latched = true;
                            state.completed = true;
                            let hold = control.backend.hold();
                            guiding = guiding && hold.is_err();
                            state.fault = Some(format!(
                                "controller/envelope (mode {:?}, expected {:?}); hold={hold:?}",
                                m.mode, expected_mode
                            ));
                        }
                        // The SDK returns cached UDP output after feedback stops.
                        // An entirely unchanged seven-axis tuple for 250 ms is
                        // refused while hand guiding; this is a conservative
                        // screen, not a measured packet age.
                        if !latched
                            && guiding
                            && feedback.observe(&m, clock.now()) >= Duration::from_millis(250)
                        {
                            pending.clear();
                            latched = true;
                            state.completed = true;
                            let hold = control.backend.hold();
                            guiding = guiding && hold.is_err();
                            state.fault = Some(format!(
                                "feedback unchanged for 250 ms; controller feedback unverified; hold={hold:?}"
                            ));
                        }
                        if !latched
                            && let Some(target) = &commanded
                            && let Err(error) = watchdog.observe(
                                target,
                                &m.joints,
                                clock.since(start).as_secs_f64(),
                            )
                        {
                            pending.clear();
                            latched = true;
                            state.watchdog_ok = false;
                            state.completed = true;
                            let hold = control.backend.hold();
                            state.fault = Some(format!("{error}; hold={hold:?}"));
                            state.recovery_fault = hold.is_ok().then(|| {
                                (state.fault.clone().unwrap(), RecoveryFaultCategory::Trip)
                            });
                        }
                        let mut trip = None;
                        state.contact = None;
                        state.contact_at = None;
                        guard.velocity_limit = if guiding {
                            base_velocity_limit * HAND_GUIDING_VELOCITY_FACTOR
                        } else {
                            base_velocity_limit
                        };
                        if !latched {
                            trip = guard
                                .observe(
                                    clock.since(start).as_secs_f64(),
                                    &m.velocities,
                                    &m.efforts,
                                )
                                .map(|t| {
                                    format!(
                                        "{}: joint={} observed={} limit={}",
                                        t.reason, t.joint, t.observed, t.limit
                                    )
                                });
                            // A commanded carriage move is not contact: its
                            // position lags its target by the whole travel. The
                            // cap resumes on the measured hold that ends the move.
                            // An unqualified carriage's effort is never a
                            // contact signal, guided or streamed: about -110 N
                            // at rest, swinging 20 N with posture in free space
                            // (measured while guiding; streamed motion changes
                            // posture the same way). Only its deflection screen
                            // judges contact, and its trip holds; so does a
                            // standoff tool's, which never touches the work.
                            if let Some(until) = transfer_until
                                && clock.now() >= until
                            {
                                transfer_until = None;
                                contact.rearm();
                            }
                            if let Some(carriage) = &m.carriage
                                && let Some(target_m) = carriage.target_m
                                && carriage_move.is_none()
                            {
                                let speed =
                                    m.velocities.iter().map(|v| v.abs()).fold(0.0_f64, f64::max);
                                match contact.observe(crate::contact::ContactObservation {
                                    effort_n: carriage.effort_n,
                                    position_m: carriage.position_m,
                                    target_m,
                                    arm_speed_rad_s: speed,
                                    aligning: motion_until.is_some()
                                        || !pending.is_empty()
                                        || speed >= 0.3,
                                    effort_assessable: carriage_contact && transfer_until.is_none(),
                                }) {
                                    Ok(e) => {
                                        state.contact = Some(e);
                                        state.contact_at = Some(clock.now());
                                        if e.trip {
                                            trip = Some(format!(
                                                "carriage_contact_cap: contact_n={} cap_n={} deflection_m={} limit_m={} effort_n={} position_m={} target_m={} baseline_n={:?}",
                                                e.contact_n,
                                                contact.cap_n,
                                                e.deflection_m,
                                                contact.deflect_m,
                                                carriage.effort_n,
                                                carriage.position_m,
                                                target_m,
                                                e.baseline_n
                                            ));
                                        }
                                    }
                                    Err(e) => trip = Some(e.to_string()),
                                }
                            }
                        }
                        if let Some(reason) = trip {
                            pending.clear();
                            latched = true;
                            state.completed = true;
                            let hold = control.backend.hold();
                            guiding = guiding && hold.is_err();
                            state.fault = Some(format!("{reason}; hold={hold:?}"));
                            state.recovery_fault = hold.is_ok().then(|| {
                                (state.fault.clone().unwrap(), RecoveryFaultCategory::Trip)
                            });
                            // A trip retract never outranks the physical stop atomic.
                            if control.estop.load(Ordering::SeqCst) == estop::OK
                                && !emergency.load(Ordering::Acquire)
                                && carriage_contact
                            {
                                match control.backend.retract_carriage(0.032, 0.6) {
                                    Ok(()) => {
                                        retract_started = Some(clock.now());
                                        state.completed = false;
                                        state.retract_verified = None;
                                    }
                                    Err(e) => {
                                        state.retract_verified = Some(false);
                                        state.fault =
                                            Some(format!("{reason}; retract failed: {e}"));
                                    }
                                }
                            }
                        }
                        if let Some((started, target_m, goal_time_s)) = carriage_move
                            && clock.since(started).as_secs_f64() >= goal_time_s + 0.3
                        {
                            carriage_move = None;
                            motion_until = None;
                            match &m.carriage {
                                Some(carriage)
                                    if (carriage.position_m - target_m).abs()
                                        <= CARRIAGE_MOVE_TOLERANCE_M =>
                                {
                                    state.completed = true;
                                    if let Err(error) = control.backend.hold() {
                                        latched = true;
                                        state.fault = Some(format!(
                                            "hold after carriage move failed: {error}"
                                        ));
                                    }
                                }
                                other => {
                                    latched = true;
                                    state.completed = true;
                                    state.fault = Some(format!(
                                        "carriage did not reach {target_m} m: measured {:?}; hold={:?}",
                                        other.as_ref().map(|c| c.position_m),
                                        control.backend.hold()
                                    ));
                                }
                            }
                        }
                        if let Some(started) = retract_started {
                            let target_retract =
                                state.target_abort.as_ref().is_some_and(|receipt| {
                                    receipt.retract_commanded && receipt.completed_wall_ns.is_none()
                                });
                            if let Some(carriage) = &m.carriage
                                && (carriage.position_m - 0.032).abs() <= 0.002
                                && (!target_retract
                                    || (m.mode == crate::Mode::Position
                                        && m.error.is_empty()
                                        && m.joints.len() == 6
                                        && m.joints.iter().all(|q| q.is_finite())))
                            {
                                state.retract_verified = Some(true);
                                state.completed = true;
                                retract_started = None;
                                if target_retract && let Some(receipt) = &mut state.target_abort {
                                    receipt.outcome = TargetAbortOutcome::Retracted;
                                    receipt.completed_wall_ns = Some(wall_ns());
                                    receipt.final_measured = Some(m.clone());
                                    receipt.reason =
                                        "qualified target retract verified by carriage readback"
                                            .into();
                                }
                            } else if clock.since(started) > Duration::from_millis(800) {
                                state.retract_verified = Some(false);
                                state.completed = true;
                                retract_started = None;
                                state.fault = Some(format!(
                                    "{}; PEN NOT RETRACTED",
                                    state.fault.as_deref().unwrap_or("trip")
                                ));
                                if target_retract && let Some(receipt) = &mut state.target_abort {
                                    receipt.outcome = TargetAbortOutcome::Uncertain;
                                    receipt.completed_wall_ns = Some(wall_ns());
                                    receipt.final_measured = Some(m.clone());
                                    receipt.reason =
                                        "target retract not verified before 800 ms deadline".into();
                                }
                            }
                        }
                        state.measured_started_wall_ns = measured_started_wall_ns;
                        state.measured_wall_ns = wall_ns();
                        state.measured_at = Some(clock.now());
                        state.measured_simulation_ns = clock.timestamp_ns();
                        if let Some(sender) = &telemetry
                            && sender
                                .try_send(TelemetrySample {
                                    tick,
                                    wall_ns: state.measured_wall_ns,
                                    since_start: clock.since(start),
                                    measured: m.clone(),
                                    estop: control.estop.load(Ordering::SeqCst),
                                    guiding,
                                    latched,
                                })
                                .is_err()
                        {
                            state.telemetry_dropped += 1;
                        }
                        state.measured = Some(m);
                    }
                    Err(e) => {
                        pending.clear();
                        latched = true;
                        retract_started = None;
                        if let Some(receipt) = &mut state.target_abort
                            && receipt.retract_commanded
                            && receipt.completed_wall_ns.is_none()
                        {
                            receipt.outcome = TargetAbortOutcome::Uncertain;
                            receipt.completed_wall_ns = Some(wall_ns());
                            receipt.reason = format!("target retract feedback failed: {e}");
                        }
                        let hold = control.backend.hold();
                        guiding = guiding && hold.is_err();
                        state.fault = Some(format!("measurement: {e}; hold={hold:?}"));
                        state.completed = true;
                        if let Clock::Offline(clock) = &clock {
                            state.watchdog_ok = false;
                            *published.lock().unwrap() = state.clone();
                            clock.close(state.fault.as_ref().unwrap().clone());
                            break;
                        }
                    }
                }
                // A target abort's measured retract owns the backend until
                // its receipt is final. Even a queued recovery request waits;
                // it cannot hide an incomplete retract behind a new sequence.
                let request = if state.target_abort.as_ref().is_some_and(|receipt| {
                    receipt.retract_commanded && receipt.completed_wall_ns.is_none()
                }) {
                    Err(TryRecvError::Empty)
                } else {
                    receiver.try_recv()
                };
                match request {
                    Ok(request) => {
                        let previous_sequence = state.sequence;
                        state.stream_seed = None;
                        state.stream_progress = None;
                        state.stream_start_guard_refused = false;
                        state.sequence = request.sequence;
                        state.injection_result = None;
                        match request.primitive {
                            // While hand guiding only the stop, freeze, trip and
                            // mock-injection primitives are operations of this
                            // mode. Anything else is a conductor error: latch and
                            // take the ordered effort-to-position freeze.
                            ref other
                                if guiding
                                    && !matches!(
                                        other,
                                        Primitive::TripRetract
                                            | Primitive::Inject(_)
                                            | Primitive::ReadPayload
                                            | Primitive::Hold
                                            | Primitive::Freeze
                                            | Primitive::GuideStop
                                    ) =>
                            {
                                pending.clear();
                                latched = true;
                                state.completed = true;
                                let hold = control.backend.hold();
                                if hold.is_ok() {
                                    guiding = false;
                                }
                                state.fault = Some(format!(
                                    "primitive refused while hand guiding; hold={hold:?}"
                                ));
                            }
                            Primitive::TripRetract => {
                                motion_until = None;
                                pending.clear();
                                latched = true;
                                state.completed = true;
                                if control.estop.load(Ordering::SeqCst) != estop::OK
                                    || emergency.load(Ordering::Acquire)
                                {
                                    state.fault = Some("e-stop prevents retract".into());
                                    state.retract_verified = Some(false);
                                } else if !carriage_contact {
                                    state.fault = Some(format!(
                                        "trip hold without retract ({:?} tool, carriage_qualified={}); hold={:?}",
                                        control.backend.contact_policy(),
                                        control.backend.carriage_qualified(),
                                        control.backend.hold()
                                    ));
                                    state.retract_verified = None;
                                } else {
                                    let _ = control.backend.hold();
                                    match control.backend.retract_carriage(0.032, 0.6) {
                                        Ok(()) => {
                                            retract_started = Some(clock.now());
                                            state.completed = false;
                                            state.retract_verified = None;
                                        }
                                        Err(e) => {
                                            state.fault = Some(e.to_string());
                                            state.retract_verified = Some(false);
                                        }
                                    }
                                }
                            }
                            Primitive::Reconnect(config) => {
                                motion_until = None;
                                state.recovery_targets = None;
                                parked = false;
                                guiding = false;
                                pending.clear();
                                contact = crate::contact::ContactCap::default();
                                state.contact = None;
                                state.contact_at = None;
                                state.completed = true;
                                state.fault = control
                                    .backend
                                    .connect(&config)
                                    .err()
                                    .map(|e| e.to_string());
                                state.connect_wait = control.backend.connect_wait();
                                if state.fault.is_none() {
                                    watchdog.reset();
                                    commanded = None;
                                    state.watchdog_ok = true;
                                }
                            }
                            Primitive::LoadConfig(path) => {
                                state.completed = true;
                                let result = (|| {
                                    if control.estop.load(Ordering::SeqCst) != estop::OK
                                        || emergency.load(Ordering::Acquire)
                                        || control.backend.measured()?.mode != crate::Mode::Idle
                                    {
                                        return Err(Error(
                                            "profile loading requires a released stop and idle arm"
                                                .into(),
                                        ));
                                    }
                                    control.backend.load_config(&path)
                                })();
                                state.fault = result.err().map(|error| error.to_string());
                            }
                            Primitive::ClearError => {
                                pending.clear();
                                state.completed = true;
                                state.fault =
                                    control.backend.clear_error().err().map(|e| e.to_string());
                                state.connect_wait = control.backend.connect_wait();
                            }
                            Primitive::Inject(faults) => {
                                state.completed = true;
                                let result = control
                                    .backend
                                    .inject_faults(faults)
                                    .map_err(|e| e.to_string());
                                state.fault = result.as_ref().err().cloned();
                                state.injection_result = Some(result);
                            }
                            Primitive::ReadPayload => {
                                state.completed = true;
                                match control.backend.payload_readback() {
                                    Ok(payload) => state.payload = payload,
                                    Err(e) => state.fault = Some(format!("payload readback: {e}")),
                                }
                            }
                            Primitive::PrepareRecovery { staged } => {
                                let busy = !pending.is_empty() || motion_until.is_some();
                                state.completed = true;
                                state.recovery_targets = None;
                                let result = (|| {
                                    if control.estop.load(Ordering::SeqCst) != estop::OK
                                        || emergency.load(Ordering::Acquire)
                                    {
                                        return Err(Error(
                                            "e-stop prevents recovery preparation".into(),
                                        ));
                                    }
                                    if busy {
                                        return Err(Error(
                                            "busy during recovery preparation".into(),
                                        ));
                                    }
                                    let axes = |m: Measured| -> Result<[f64; 7]> {
                                        if !m.error.is_empty() || m.joints.len() != 6 {
                                            return Err(Error(
                                                "unhealthy recovery measurement".into(),
                                            ));
                                        }
                                        let carriage = m.carriage.ok_or_else(|| {
                                            Error("recovery carriage unavailable".into())
                                        })?;
                                        let mut q = [0.0; 7];
                                        q[..6].copy_from_slice(&m.joints);
                                        q[6] = carriage.position_m;
                                        Ok(q)
                                    };
                                    let mut measured = axes(control.backend.measured()?)?;
                                    if control.backend.recovery_needs_configuration(&measured)? {
                                        // Native clear_error applies its golden then reconnects.
                                        // Re-read both pose and limits; never retain boot limits.
                                        control.backend.clear_error()?;
                                        measured = axes(control.backend.measured()?)?;
                                    }
                                    let limits = control.backend.recovery_limits()?;
                                    if control.estop.load(Ordering::SeqCst) != estop::OK
                                        || emergency.load(Ordering::Acquire)
                                    {
                                        return Err(Error(
                                            "e-stop during recovery preparation".into(),
                                        ));
                                    }
                                    crate::recovery::Targets::from_measured(
                                        measured,
                                        staged,
                                        &limits,
                                        carriage_rest(&control.backend, &staged),
                                    )
                                })();
                                match result {
                                    Ok(targets) => {
                                        // Preparation may reload the golden and reconnect
                                        // in idle. No ordinary motion is admitted until
                                        // Takeover explicitly verifies position control.
                                        latched = true;
                                        commanded = None;
                                        watchdog.reset();
                                        state.recovery_targets = Some(targets);
                                        state.fault = None;
                                        state.recovery_fault = None;
                                    }
                                    Err(error) => {
                                        pending.clear();
                                        motion_until = None;
                                        state.fault = Some(format!(
                                            "{error}; hold={:?}",
                                            control.backend.hold()
                                        ));
                                        latched = true;
                                    }
                                }
                            }
                            Primitive::Takeover => {
                                parked = false;
                                guiding = false;
                                pending.clear();
                                motion_until = None;
                                state.completed = true;
                                let result = (|| {
                                    if control.estop.load(Ordering::SeqCst) != estop::OK
                                        || emergency.load(Ordering::Acquire)
                                    {
                                        return Err(Error("e-stop prevents takeover".into()));
                                    }
                                    let prepared =
                                        state.recovery_targets.as_ref().ok_or_else(|| {
                                            Error("prepare recovery before takeover".into())
                                        })?;
                                    let staged = prepared.staged;
                                    // The prepared Sleep carriage is the profile's rest
                                    // for a qualified carriage (its staged pose carries
                                    // the measured value); an unqualified one is re-read.
                                    let rest = control
                                        .backend
                                        .carriage_qualified()
                                        .then_some(prepared.sleep[6]);
                                    let measured = control.backend.measured()?;
                                    if !measured.error.is_empty() || measured.joints.len() != 6 {
                                        return Err(Error("unhealthy takeover measurement".into()));
                                    }
                                    let mut q = [0.0; 7];
                                    q[..6].copy_from_slice(&measured.joints);
                                    q[6] = measured
                                        .carriage
                                        .ok_or_else(|| Error("takeover carriage missing".into()))?
                                        .position_m;
                                    let limits = control.backend.recovery_limits()?;
                                    let targets = crate::recovery::Targets::from_measured(
                                        q, staged, &limits, rest,
                                    )?;
                                    if !control.envelope.is_finite()
                                        || control.envelope <= 0.0
                                        || !control.max_velocity.is_finite()
                                        || control.max_velocity <= 0.0
                                        || targets.takeover[..6].iter().zip(&q[..6]).any(
                                            |(target, current)| {
                                                target.abs() > control.envelope
                                                    || 2.0 * (target - current).abs()
                                                        / crate::recovery::TAKEOVER_S
                                                        > control.max_velocity
                                            },
                                        )
                                    {
                                        return Err(Error(
                                            "takeover violates control envelope/velocity".into(),
                                        ));
                                    }
                                    if control.estop.load(Ordering::SeqCst) != estop::OK
                                        || emergency.load(Ordering::Acquire)
                                    {
                                        return Err(Error(
                                            "e-stop during takeover preparation".into(),
                                        ));
                                    }
                                    control.backend.takeover(&targets.takeover)?;
                                    Ok(targets)
                                })();
                                match result {
                                    Ok(targets) => {
                                        commanded = Some(targets.takeover[..6].to_vec());
                                        state.recovery_targets = Some(targets);
                                        latched = false;
                                        guard.reset();
                                        watchdog.reset();
                                        state.watchdog_ok = true;
                                        state.fault = None;
                                        state.recovery_fault = None;
                                        state.commands += 1;
                                        state.completed = false;
                                        motion_until = Some(TimedHold::Measured(
                                            clock.now()
                                                + Duration::from_secs_f64(
                                                    crate::recovery::TAKEOVER_S,
                                                ),
                                        ));
                                    }
                                    Err(error) => {
                                        latched = true;
                                        state.fault = Some(format!(
                                            "{error}; hold={:?}",
                                            control.backend.hold()
                                        ));
                                    }
                                }
                            }
                            Primitive::Reseed => {
                                motion_until = None;
                                parked = false;
                                guiding = false;
                                pending.clear();
                                state.completed = true;
                                let result = (|| {
                                    if control.estop.load(Ordering::SeqCst) != estop::OK
                                        || emergency.load(Ordering::Acquire)
                                    {
                                        return Err(Error("e-stop".into()));
                                    }
                                    let measured = control.backend.measured()?;
                                    if !measured.error.is_empty() {
                                        return Err(Error("controller unhealthy".into()));
                                    }
                                    let duration = control
                                        .backend
                                        .reseed_hold(control.max_velocity, control.envelope)?;
                                    if duration > Duration::from_secs(5) {
                                        return Err(Error(
                                            "re-seed hold interval exceeds five seconds".into(),
                                        ));
                                    }
                                    let measured = control.backend.measured()?;
                                    // The backend seeded all axes. Validate readback
                                    // without treating tolerated encoder feedback
                                    // beyond the envelope as a commanded target.
                                    let envelope = control.envelope;
                                    control.check_sample(&Sample {
                                        joints: measured
                                            .joints
                                            .iter()
                                            .map(|q| q.clamp(-envelope, envelope))
                                            .collect(),
                                        dt_s: period.as_secs_f64(),
                                        contact: 0.0,
                                    })?;
                                    Ok(duration)
                                })();
                                match result {
                                    Ok(duration) => {
                                        latched = false;
                                        guard.reset();
                                        watchdog.reset();
                                        commanded = None;
                                        state.watchdog_ok = true;
                                        state.fault = None;
                                        state.recovery_fault = None;
                                        state.commands += 1;
                                        if !duration.is_zero() {
                                            state.completed = false;
                                            motion_until =
                                                Some(TimedHold::Commanded(clock.now() + duration));
                                        }
                                    }
                                    Err(e) => {
                                        latched = true;
                                        state.fault = Some(e.to_string());
                                    }
                                }
                            }
                            Primitive::Hold | Primitive::Freeze => {
                                motion_until = None;
                                pending.clear();
                                commanded = None;
                                watchdog.reset();
                                state.completed = true;
                                match control.backend.hold() {
                                    Ok(()) => guiding = false,
                                    Err(e) => {
                                        state.fault = Some(e.to_string());
                                        latched = true;
                                    }
                                }
                            }
                            _ if latched => {
                                state.completed = true;
                                state.fault = Some("Recover with measured re-seed required".into());
                            }
                            Primitive::CarriageTo {
                                target_m,
                                goal_time_s,
                            } => {
                                pending.clear();
                                commanded = None;
                                state.completed = true;
                                let result = (|| {
                                    if control.estop.load(Ordering::SeqCst) != estop::OK
                                        || emergency.load(Ordering::Acquire)
                                    {
                                        return Err(Error("e-stop prevents carriage move".into()));
                                    }
                                    if !target_m.is_finite() || !(1.0..=5.0).contains(&goal_time_s)
                                    {
                                        return Err(Error("unqualified carriage move".into()));
                                    }
                                    let measured = control.backend.measured()?;
                                    if measured.mode != crate::Mode::Position
                                        || !measured.error.is_empty()
                                    {
                                        return Err(Error(format!(
                                            "carriage move needs a healthy position-mode arm: {:?} {}",
                                            measured.mode, measured.error
                                        )));
                                    }
                                    control.backend.move_carriage(target_m, goal_time_s)
                                })();
                                match result {
                                    Ok(()) => {
                                        state.completed = false;
                                        state.fault = None;
                                        state.recovery_fault = None;
                                        state.commands += 1;
                                        carriage_move = Some((clock.now(), target_m, goal_time_s));
                                        // The moving carriage is not a contact; the cap
                                        // treats it as alignment until it settles. The
                                        // readback check clears this itself; the expiry
                                        // only backs it up.
                                        motion_until = Some(TimedHold::Measured(
                                            clock.now()
                                                + Duration::from_secs_f64(goal_time_s + 0.5),
                                        ));
                                    }
                                    Err(error) => {
                                        latched = true;
                                        state.fault = Some(format!(
                                            "{error}; hold={:?}",
                                            control.backend.hold()
                                        ));
                                    }
                                }
                            }
                            Primitive::GuideStop => {
                                // Order: fresh measurement, all joints to position
                                // control, that measured pose as the target; then a
                                // fresh mode readback decides whether the arm is held.
                                motion_until = None;
                                pending.clear();
                                commanded = None;
                                watchdog.reset();
                                state.completed = true;
                                let result = (|| {
                                    if !guiding {
                                        return Err(Error("not hand guiding".into()));
                                    }
                                    let before = control.backend.measured()?;
                                    control.backend.hold()?;
                                    let after = control.backend.measured_fresh()?;
                                    if after.mode != crate::Mode::Position
                                        || !after.error.is_empty()
                                    {
                                        return Err(Error(format!(
                                            "position mode not confirmed after hand-guiding stop: {:?} {}",
                                            after.mode, after.error
                                        )));
                                    }
                                    let mut submitted = [0.0; 7];
                                    submitted[..6].copy_from_slice(&before.joints);
                                    submitted[6] =
                                        before.carriage.as_ref().map_or(f64::NAN, |c| c.position_m);
                                    Ok((submitted, after))
                                })();
                                if let Some(receipt) = &mut state.guide {
                                    receipt.stopped_wall_ns = Some(wall_ns());
                                    receipt.stop_error =
                                        result.as_ref().err().map(ToString::to_string);
                                    if let Ok((submitted, after)) = &result {
                                        receipt.stop_submitted = Some(*submitted);
                                        receipt.stop_measured = Some(after.clone());
                                    }
                                }
                                match result {
                                    Ok(_) => {
                                        guiding = false;
                                        state.guiding = false;
                                        state.fault = None;
                                        state.recovery_fault = None;
                                        state.commands += 1;
                                    }
                                    Err(error) => {
                                        // Mode unknown: keep the owner alive and latched.
                                        latched = true;
                                        state.fault = Some(format!(
                                            "{error}; hold={:?}",
                                            control.backend.hold()
                                        ));
                                    }
                                }
                            }
                            _ if !pending.is_empty() || motion_until.is_some() => {
                                // Never attach a new acknowledgement sequence to
                                // an old stream that continues moving invisibly.
                                pending.clear();
                                latched = true;
                                state.completed = true;
                                state.fault =
                                    Some(format!("busy; hold={:?}", control.backend.hold()));
                            }
                            Primitive::HandGuide { carriage_m } => {
                                state.completed = true;
                                let result = (|| {
                                    if control.estop.load(Ordering::SeqCst) != estop::OK
                                        || emergency.load(Ordering::Acquire)
                                    {
                                        return Err(Error("e-stop prevents hand guiding".into()));
                                    }
                                    let measured = control.backend.measured_fresh()?;
                                    let carriage = measured.carriage.as_ref().ok_or_else(|| {
                                        Error("carriage measurement missing".into())
                                    })?;
                                    if measured.mode != crate::Mode::Position
                                        || !measured.error.is_empty()
                                        || measured.joints.len() != 6
                                    {
                                        return Err(Error(format!(
                                            "hand guiding needs a healthy position-mode arm: {:?} {}",
                                            measured.mode, measured.error
                                        )));
                                    }
                                    if !carriage_m.is_finite()
                                        || (carriage.position_m - carriage_m).abs() > 0.0005
                                    {
                                        return Err(Error(format!(
                                            "carriage datum {carriage_m} differs from measured {}",
                                            carriage.position_m
                                        )));
                                    }
                                    if !state.contact.is_some_and(|c| c.armed) {
                                        return Err(Error(
                                            "carriage contact baseline not armed; rest the unloaded arm first".into(),
                                        ));
                                    }
                                    control.backend.enter_hand_guiding(carriage_m)?;
                                    let after = control.backend.measured_fresh()?;
                                    if after.mode != crate::Mode::HandGuiding
                                        || !after.error.is_empty()
                                    {
                                        return Err(Error(format!(
                                            "mixed hand-guiding mode not confirmed by readback: {:?} {}",
                                            after.mode, after.error
                                        )));
                                    }
                                    Ok(after)
                                })();
                                match result {
                                    Ok(after) => {
                                        guiding = true;
                                        commanded = None;
                                        watchdog.reset();
                                        feedback = FeedbackChange::default();
                                        state.guiding = true;
                                        state.guide = Some(GuideReceipt {
                                            carriage_datum_m: carriage_m,
                                            entered_wall_ns: wall_ns(),
                                            entered_measured: after,
                                            stopped_wall_ns: None,
                                            stop_submitted: None,
                                            stop_measured: None,
                                            stop_error: None,
                                        });
                                        state.fault = None;
                                        state.recovery_fault = None;
                                        state.commands += 1;
                                    }
                                    Err(error) => {
                                        latched = true;
                                        state.fault = Some(format!(
                                            "{error}; hold={:?}",
                                            control.backend.hold()
                                        ));
                                    }
                                }
                            }
                            Primitive::RecoveryMove(phase) => {
                                state.completed = true;
                                let result = (|| {
                                    if control.estop.load(Ordering::SeqCst) != estop::OK
                                        || emergency.load(Ordering::Acquire)
                                    {
                                        return Err(Error("e-stop prevents recovery move".into()));
                                    }
                                    let targets =
                                        state.recovery_targets.as_ref().ok_or_else(|| {
                                            Error("prepare recovery before move".into())
                                        })?;
                                    let target = phase.target(targets);
                                    let measured = control.backend.measured()?;
                                    if measured.mode != crate::Mode::Position
                                        || !measured.error.is_empty()
                                        || measured.joints.len() != 6
                                        || measured.joints.iter().any(|q| !q.is_finite())
                                    {
                                        return Err(Error(
                                            "unhealthy recovery move measurement".into(),
                                        ));
                                    }
                                    if crate::recovery::needs_golden(
                                        &target,
                                        &control.backend.recovery_limits()?,
                                    )? {
                                        return Err(Error(
                                            "recovery move outside live limits".into(),
                                        ));
                                    }
                                    if !control.envelope.is_finite()
                                        || control.envelope <= 0.0
                                        || !control.max_velocity.is_finite()
                                        || control.max_velocity <= 0.0
                                        || target[..6].iter().zip(&measured.joints).any(
                                            |(q, current)| {
                                                q.abs() > control.envelope
                                                    || current.abs() > control.envelope
                                                    || 2.0 * (q - current).abs() / phase.duration()
                                                        > control.max_velocity
                                            },
                                        )
                                    {
                                        return Err(Error(
                                            "recovery move violates control envelope/velocity"
                                                .into(),
                                        ));
                                    }
                                    // The carriage rides at its measured value through
                                    // every phase but a qualified carriage's return to
                                    // rest at Sleep: that is a timed carriage move under
                                    // the same bound as `CarriageTo`, judged by its
                                    // readback, with the contact estimator sitting it out.
                                    let carriage_m = measured
                                        .carriage
                                        .as_ref()
                                        .ok_or_else(|| {
                                            Error("recovery move carriage missing".into())
                                        })?
                                        .position_m;
                                    let carriage_travel_m = (target[6] - carriage_m).abs();
                                    let moves_carriage =
                                        carriage_travel_m > CARRIAGE_MOVE_TOLERANCE_M;
                                    if moves_carriage
                                        && (!control.backend.carriage_qualified()
                                            || carriage_travel_m / phase.duration()
                                                > CARRIAGE_MOVE_MAX_M_PER_S)
                                    {
                                        return Err(Error(format!(
                                            "recovery move carriage travel {carriage_travel_m:.4} m refused (carriage_qualified={})",
                                            control.backend.carriage_qualified()
                                        )));
                                    }
                                    if control.estop.load(Ordering::SeqCst) != estop::OK
                                        || emergency.load(Ordering::Acquire)
                                    {
                                        return Err(Error(
                                            "e-stop during recovery move preparation".into(),
                                        ));
                                    }
                                    control.backend.recovery_move(&target, phase)?;
                                    Ok((target, moves_carriage))
                                })();
                                match result {
                                    Ok((target, moves_carriage)) => {
                                        commanded = Some(target[..6].to_vec());
                                        watchdog.reset();
                                        state.commands += 1;
                                        state.completed = false;
                                        state.fault = None;
                                        state.recovery_fault = None;
                                        if moves_carriage {
                                            carriage_move =
                                                Some((clock.now(), target[6], phase.duration()));
                                            // The readback check completes the move; the
                                            // expiry only backs it up, as for `CarriageTo`.
                                            motion_until = Some(TimedHold::Measured(
                                                clock.now()
                                                    + Duration::from_secs_f64(
                                                        phase.duration() + 0.5,
                                                    ),
                                            ));
                                        } else {
                                            motion_until = Some(TimedHold::Measured(
                                                clock.now()
                                                    + Duration::from_secs_f64(phase.duration()),
                                            ));
                                        }
                                    }
                                    Err(error) => {
                                        latched = true;
                                        state.fault = Some(format!(
                                            "{error}; hold={:?}",
                                            control.backend.hold()
                                        ));
                                    }
                                }
                            }
                            Primitive::Idle => {
                                state.completed = true;
                                let result = (|| {
                                    if control.estop.load(Ordering::SeqCst) != estop::OK
                                        || emergency.load(Ordering::Acquire)
                                    {
                                        return Err(Error("e-stop prevents idle release".into()));
                                    }
                                    control.backend.set_mode(crate::Mode::Idle)?;
                                    // Mode changes may be acknowledged before feedback catches up.
                                    // Keep polling the physical stop while waiting for measured idle.
                                    let deadline = Instant::now() + Duration::from_millis(500);
                                    loop {
                                        if control.estop.load(Ordering::SeqCst) != estop::OK
                                            || emergency.load(Ordering::Acquire)
                                        {
                                            return Err(Error(
                                                "e-stop prevents idle release".into(),
                                            ));
                                        }
                                        let measured = control.backend.measured()?;
                                        if !measured.error.is_empty() {
                                            return Err(Error(format!(
                                                "idle release controller error: {}",
                                                measured.error
                                            )));
                                        }
                                        if measured.mode == crate::Mode::Idle {
                                            return Ok(());
                                        }
                                        if Instant::now() >= deadline {
                                            return Err(Error(format!(
                                                "idle mode not confirmed by controller: {:?}",
                                                measured.mode
                                            )));
                                        }
                                        thread::sleep(Duration::from_millis(2));
                                    }
                                })();
                                latched = true;
                                commanded = None;
                                watchdog.reset();
                                state.contact = None;
                                state.contact_at = None;
                                parked = result.is_ok();
                                state.fault = result.err().map(|e| e.to_string());
                                if !parked {
                                    let _ = control.backend.hold();
                                }
                            }
                            Primitive::StreamJoint {
                                seed,
                                expected_sequence,
                                ..
                            }
                            | Primitive::StreamCarriage {
                                seed,
                                expected_sequence,
                                ..
                            } => {
                                let mut receipt = StreamSeedReceipt {
                                    requested: seed,
                                    measured: None,
                                    expected_sequence,
                                    previous_sequence,
                                    accepted: false,
                                    error: None,
                                };
                                let result = (|| {
                                    if expected_sequence != previous_sequence {
                                        return Err(Error(format!(
                                            "stream preflight sequence changed: expected {expected_sequence}, actual {previous_sequence}"
                                        )));
                                    }
                                    let queue = request.joint_queue.ok_or_else(|| {
                                        Error("missing validated joint queue".into())
                                    })?;
                                    let first = match queue.front() {
                                        Some(QueuedSample::Joint(s, ..)) => s,
                                        _ => return Err(Error("missing joint seed sample".into())),
                                    };
                                    let measured = control.backend.measured()?;
                                    receipt.measured = Some(measured.clone());
                                    first.check_seed(&seed, &measured)?;
                                    if control.estop.load(Ordering::SeqCst) != estop::OK
                                        || emergency.load(Ordering::Acquire)
                                    {
                                        return Err(Error(
                                            "e-stop prevents stream dispatch".into(),
                                        ));
                                    }
                                    state.stream_progress = Some(StreamProgress {
                                        sequence: request.sequence,
                                        total: queue.len(),
                                        submitted: 0,
                                        last_pen: None,
                                        late_ticks: 0,
                                        max_late_us: 0,
                                    });
                                    pending = queue;
                                    Ok(())
                                })();
                                receipt.accepted = result.is_ok();
                                receipt.error = result.as_ref().err().map(ToString::to_string);
                                if result.is_ok()
                                    && matches!(
                                        pending.front(),
                                        Some(QueuedSample::Joint(_, _, _, true))
                                    )
                                {
                                    // The admitted air transfer starts now; every
                                    // executed sample extends the window.
                                    transfer_until = Some(clock.now() + Duration::from_millis(500));
                                }
                                state.stream_seed = Some(receipt);
                                state.completed = result.is_err();
                                state.fault = result
                                    .err()
                                    .map(|e| format!("{e}; hold={:?}", control.backend.hold()));
                                if state.fault.is_some() {
                                    latched = true;
                                    pending.clear();
                                }
                            }
                            Primitive::Stream(samples) => {
                                if samples.is_empty()
                                    || samples.len() > MAX_STREAM_SAMPLES
                                    || samples
                                        .iter()
                                        .any(|s| (s.dt_s - period.as_secs_f64()).abs() > 1e-9)
                                {
                                    state.completed = true;
                                    state.fault = Some("stream sample period mismatch".into());
                                } else {
                                    pending.extend(samples.into_iter().map(QueuedSample::Angular));
                                    state.completed = false;
                                    state.fault = None;
                                    state.recovery_fault = None;
                                }
                            }
                            Primitive::MoveJ {
                                target,
                                goal_time_s,
                            } => {
                                if !goal_time_s.is_finite()
                                    || goal_time_s <= 0.0
                                    || goal_time_s > 60.0
                                {
                                    state.completed = true;
                                    state.fault = Some("invalid MoveJ duration".into());
                                } else if let Some(m) = &state.measured {
                                    if target.len() != m.joints.len() {
                                        state.completed = true;
                                        state.fault = Some("MoveJ width".into());
                                    } else {
                                        let ticks =
                                            (goal_time_s / period.as_secs_f64()).ceil() as usize;
                                        for i in 1..=ticks {
                                            let t = i as f64 / ticks as f64;
                                            let blend = t * t * (3.0 - 2.0 * t);
                                            pending.push_back(QueuedSample::Angular(Sample {
                                                joints: m
                                                    .joints
                                                    .iter()
                                                    .zip(&target)
                                                    .map(|(a, b)| a + (b - a) * blend)
                                                    .collect(),
                                                dt_s: period.as_secs_f64(),
                                                contact: 0.0,
                                            }));
                                        }
                                        state.completed = false;
                                        state.fault = None;
                                        state.recovery_fault = None;
                                    }
                                }
                            }
                        }
                    }
                    Err(TryRecvError::Disconnected) => break,
                    Err(TryRecvError::Empty) => {}
                }
                joint_stream_tick |= state.stream_progress.is_some() && !state.completed;
                let awaiting_start = matches!(pending.front(),
                    Some(QueuedSample::Joint(_, Some(permit), index, _))
                    if (*index == 0).then_some(permit.starts_at).flatten()
                        .is_some_and(|start| clock.now() < start));
                if !latched
                    && !awaiting_start
                    && let Some(sample) = pending.pop_front()
                {
                    let authorization = match &sample {
                        QueuedSample::Joint(_, Some(permit), index, _) => {
                            let seed_check = if *index == 0 && permit.starts_at.is_some() {
                                match (&sample, &state.measured) {
                                    (QueuedSample::Joint(first, ..), Some(measured)) => {
                                        first.check_seed(&first.positions, measured).map(|_| ())
                                    }
                                    _ => Err(Error(
                                        "scheduled stream lacks fresh measured seed".into(),
                                    )),
                                }
                            } else {
                                Ok(())
                            };
                            seed_check.and_then(|_| permit.authorize(*index, &state))
                        }
                        _ => Ok(true),
                    };
                    let guarded_stop = matches!(authorization, Ok(false));
                    if guarded_stop {
                        pending.clear();
                        state.completed = true;
                        match control.backend.hold() {
                            Ok(()) => {
                                commanded = None;
                                watchdog.reset();
                                if let QueuedSample::Joint(_, Some(permit), index, _) = &sample {
                                    permit.stopped_at.store(*index, Ordering::Release);
                                }
                            }
                            Err(error) => {
                                latched = true;
                                state.fault = Some(format!("measured stop hold failed: {error}"));
                            }
                        }
                    } else if let Err(e) = authorization.and_then(|_| sample.execute(&mut control))
                    {
                        if let QueuedSample::Joint(_, Some(permit), 0, _) = &sample {
                            state.stream_start_guard_refused =
                                permit.start_guard_refused.load(Ordering::Acquire);
                        }
                        pending.clear();
                        latched = true;
                        state.fault = Some(format!("{e}; hold={:?}", control.backend.hold()));
                    } else {
                        commanded = Some(sample.joints());
                        state.commands += 1;
                        if let (QueuedSample::Joint(sample, ..), Some(progress)) =
                            (&sample, &mut state.stream_progress)
                        {
                            progress.submitted += 1;
                            progress.last_pen = Some(sample.pen);
                        }
                        if let QueuedSample::Joint(_, _, _, true) = &sample {
                            transfer_until = Some(clock.now() + Duration::from_millis(500));
                        }
                    }
                    state.completed = pending.is_empty();
                    if state.completed && !latched && !guarded_stop {
                        let result = match sample {
                            QueuedSample::Joint(..) => control.backend.finish_joint_stream(),
                            QueuedSample::Angular(..) if matches!(clock, Clock::Offline(_)) => {
                                control.backend.finish_joint_stream()
                            }
                            QueuedSample::Angular(..) => control.backend.hold(),
                        };
                        if let Err(error) = result {
                            latched = true;
                            state.fault = Some(format!(
                                "stream completion hold failed: {error}; hold={:?}",
                                control.backend.hold()
                            ));
                        }
                    }
                }
                if let (Clock::Offline(offline), Some(advance)) = (&clock, advance_world)
                    && let Err(error) =
                        advance(&mut control.backend, period).and_then(|()| offline.stepped())
                {
                    state.fault = Some(format!("offline world step: {error}"));
                    state.completed = true;
                    state.watchdog_ok = false;
                    *published.lock().unwrap() = state.clone();
                    offline.close(state.fault.as_ref().unwrap().clone());
                    break;
                }
                if latched {
                    motion_until = None;
                } else if motion_until.is_some_and(|hold| hold.due(clock.now())) {
                    let hold = motion_until.take().unwrap();
                    state.completed = true;
                    if let Err(error) = hold.finish(&mut control.backend) {
                        latched = true;
                        state.fault = Some(format!("timed recovery hold failed: {error}"));
                    }
                }
                // Completion includes a post-command measurement; acknowledgements
                // must never expose the pre-clear/pre-reseed controller snapshot.
                let measured_started_wall_ns = wall_ns();
                match control.backend.measured() {
                    Ok(m) => {
                        state.measured_started_wall_ns = measured_started_wall_ns;
                        state.measured_wall_ns = wall_ns();
                        state.measured_at = Some(clock.now());
                        state.measured_simulation_ns = clock.timestamp_ns();
                        // There may be no next offline tick. A fault arising in
                        // the final world step must be visible with completion,
                        // rather than waiting for another caller permit.
                        if matches!(clock, Clock::Offline(_))
                            && !latched
                            && (m.mode != crate::Mode::Position
                                || !m.error.is_empty()
                                || m.joints.is_empty()
                                || m.joints
                                    .iter()
                                    .any(|q| !q.is_finite() || q.abs() > control.envelope))
                        {
                            pending.clear();
                            latched = true;
                            state.completed = true;
                            state.fault = Some(format!(
                                "controller/envelope after offline step; hold={:?}",
                                control.backend.hold()
                            ));
                        }
                        state.measured = Some(m);
                    }
                    Err(e) => {
                        state.fault = Some(e.to_string());
                        latched = true;
                        pending.clear();
                        state.completed = true;
                        retract_started = None;
                        if let Some(receipt) = &mut state.target_abort
                            && receipt.retract_commanded
                            && receipt.completed_wall_ns.is_none()
                        {
                            receipt.outcome = TargetAbortOutcome::Uncertain;
                            receipt.completed_wall_ns = Some(wall_ns());
                            receipt.reason = format!("target retract final feedback failed: {e}");
                        }
                        let _ = control.backend.hold();
                        if let Clock::Offline(clock) = &clock {
                            state.watchdog_ok = false;
                            *published.lock().unwrap() = state.clone();
                            clock.close(state.fault.as_ref().unwrap().clone());
                            break;
                        }
                    }
                }
                // Published as the controller reports it after this tick's
                // commands, not as intended: a failed transition leaves the
                // flag telling the truth.
                state.guiding = state
                    .measured
                    .as_ref()
                    .is_some_and(|m| m.mode == crate::Mode::HandGuiding);
                if let Clock::Offline(clock) = &clock {
                    // Offline callers consume a completed world step, so publish
                    // before acknowledging it. The live loop below retains its
                    // nonblocking latest-value publication and wall pacing.
                    *published.lock().unwrap() = state.clone();
                    clock.acknowledge();
                    continue;
                }
                next += period;
                let now = Instant::now();
                let delay = next.checked_duration_since(now);
                if delay.is_none() {
                    state.record_lateness(
                        (now - next).as_micros().min(u128::from(u64::MAX)) as u64,
                        joint_stream_tick,
                    );
                    next = now;
                }
                // Publish timing with completion so the final tick is retained.
                // A reader can hold the status lock arbitrarily long. Drop this
                // publication rather than putting the control loop behind it.
                if let Ok(mut destination) = published.try_lock() {
                    *destination = state.clone();
                }
                if delay.is_some()
                    && let Some(remaining) = next.checked_duration_since(Instant::now())
                {
                    thread::sleep(remaining);
                }
            }
            if let Clock::Offline(clock) = &clock {
                // Invalidate source freshness before a backend's final hold or
                // destructor can wait on its transport. The owner guard also
                // closes the clock when construction or a backend panics.
                clock.close("offline arm worker stopped");
            }
            // A successfully released arm must not receive a position hold
            // during normal worker destruction. Fault/e-stop paths still hold.
            if terminal_idle.load(Ordering::Acquire) {
                let result = (|| {
                    control.backend.shutdown_idle()?;
                    let deadline = Instant::now() + Duration::from_millis(500);
                    loop {
                        let measured = control.backend.measured()?;
                        if measured.mode == crate::Mode::Idle {
                            return Ok(measured);
                        }
                        if Instant::now() >= deadline {
                            return Err(Error(format!(
                                "shutdown idle not confirmed: {:?}",
                                measured.mode
                            )));
                        }
                        thread::sleep(Duration::from_millis(2));
                    }
                })();
                *terminal_receipt.lock().unwrap() = Some(result);
            } else if !parked {
                let _ = control.backend.hold();
            }
        });
        Ok(Self {
            software_stop,
            target_abort,
            sender,
            status,
            shutdown,
            idle_on_shutdown,
            shutdown_receipt,
            thread: Some(thread),
            sequence: 0,
            native_shutdown_timeout: None,
            clock,
        })
    }
    /// End the worker permanently and remove drive effort from the supported
    /// arm. This terminal path cannot clear the stop and resume motion.
    pub fn shutdown_idle(&mut self) -> Result<crate::Measured> {
        self.idle_on_shutdown.store(true, Ordering::Release);
        self.shutdown.store(true, Ordering::Release);
        let deadline = Instant::now() + Duration::from_secs(5);
        while self
            .thread
            .as_ref()
            .is_some_and(|thread| !thread.is_finished())
        {
            if Instant::now() >= deadline {
                return Err(Error("shutdown worker timed out; idle unverified".into()));
            }
            thread::sleep(Duration::from_millis(5));
        }
        if let Some(thread) = self.thread.take() {
            thread
                .join()
                .map_err(|_| Error("shutdown worker panicked; idle unverified".into()))?;
        }
        self.shutdown_receipt
            .lock()
            .unwrap()
            .take()
            .unwrap_or_else(|| Err(Error("shutdown produced no idle receipt".into())))
    }
    /// A native SDK call can block beyond its advertised timeout. Once the
    /// runtime has failed and is destroying its worker, never leave that owner
    /// alive indefinitely with the hardware lease and old commands. A process
    /// exit closes every SDK socket/thread; it does not claim a verified hold.
    #[cfg(feature = "trossen")]
    pub(crate) fn bound_native_shutdown(mut self) -> Self {
        self.native_shutdown_timeout = Some(Duration::from_secs(5));
        self
    }
    /// One-shot, nonblocking abort of the current target-bound execution.
    /// The owner consumes this before another stream sample and retains the
    /// outcome in `Status::target_abort`. A second request never rearms it.
    pub fn abort_target_invalid(&self) -> bool {
        self.target_abort
            .compare_exchange(0, 1, Ordering::AcqRel, Ordering::Acquire)
            .is_ok()
    }
    pub fn submit(&mut self, primitive: Primitive) -> Result<u64> {
        // Validate and allocate on the caller thread, never in the 400 Hz
        // owner loop. The private mailbox carries the finished queue by move.
        let (primitive, joint_queue) = match primitive {
            Primitive::StreamJoint {
                seed,
                expected_sequence,
                permit,
                samples,
            } => {
                if matches!(self.clock, Clock::Offline(_))
                    && permit.as_ref().is_some_and(|p| p.starts_at.is_some())
                {
                    return Err(Error(
                        "offline streams cannot use a live scheduled start".into(),
                    ));
                }
                if samples.is_empty() || samples.len() > MAX_STREAM_SAMPLES {
                    return Err(Error("seven-axis stream length".into()));
                }
                for (i, sample) in samples.iter().enumerate() {
                    sample.validate(i.checked_sub(1).map(|j| &samples[j]))?;
                }
                if permit
                    .as_ref()
                    .is_some_and(|p| p.starts_at.is_some_and(|start| start <= Instant::now()))
                {
                    return Err(Error(
                        "scheduled stream start elapsed before submission".into(),
                    ));
                }
                if let Some(permit) = &permit
                    && (permit.total != samples.len()
                        || permit.claimed.swap(true, Ordering::AcqRel))
                {
                    return Err(Error(
                        "stream permit length mismatch or already used".into(),
                    ));
                }
                let queue = samples
                    .into_iter()
                    .enumerate()
                    .map(|(index, sample)| {
                        QueuedSample::Joint(sample, permit.clone(), index, false)
                    })
                    .collect();
                (
                    Primitive::StreamJoint {
                        seed,
                        expected_sequence,
                        permit: None, // authorization is carried by the private queue
                        samples: Vec::new(),
                    },
                    Some(queue),
                )
            }
            Primitive::StreamCarriage {
                seed,
                expected_sequence,
                samples,
            } => {
                if samples.is_empty()
                    || samples.len() > MAX_STREAM_SAMPLES
                    || samples[0].positions != seed
                    || samples.last().unwrap().velocities != [0.0; 7]
                {
                    return Err(Error(
                        "air carriage stream length, seed or terminal velocity".into(),
                    ));
                }
                for (i, sample) in samples.iter().enumerate() {
                    sample.validate_transfer(i.checked_sub(1).map(|j| &samples[j]))?;
                }
                // A transfer must finish in the unchanged contact reserve.
                let mut endpoint = samples.last().unwrap().clone();
                endpoint.pen = true;
                endpoint.validate(None)?;
                let queue = samples
                    .into_iter()
                    .enumerate()
                    .map(|(index, sample)| QueuedSample::Joint(sample, None, index, true))
                    .collect();
                (
                    Primitive::StreamCarriage {
                        seed,
                        expected_sequence,
                        samples: Vec::new(),
                    },
                    Some(queue),
                )
            }
            other => (other, None),
        };
        let sequence = self.sequence + 1;
        self.sender
            .try_send(Request {
                sequence,
                primitive,
                joint_queue,
            })
            .map_err(|e| Error(format!("control mailbox: {e}")))?;
        self.sequence = sequence;
        Ok(sequence)
    }
    pub fn status(&self) -> Status {
        self.status.lock().unwrap().clone()
    }
    /// Observe final command/feedback receipts without owning or mutating the worker.
    /// A retained reader does not keep the worker or its backend thread alive.
    pub fn status_reader(&self) -> impl Fn() -> Status + Send + 'static {
        let status = self.status.clone();
        move || status.lock().unwrap().clone()
    }
}
impl Drop for Worker {
    fn drop(&mut self) {
        self.shutdown.store(true, Ordering::Release);
        if let Some(thread) = self.thread.take() {
            if let Some(timeout) = self.native_shutdown_timeout {
                let deadline = Instant::now() + timeout;
                while !thread.is_finished() {
                    if Instant::now() >= deadline {
                        eprintln!(
                            "native controller worker did not stop within {timeout:?}; \
                             terminating its process and sockets; landing is unverified"
                        );
                        // Do not detach an SDK thread that may later resume
                        // sending commands, or release its hardware lease early.
                        std::process::exit(5);
                    }
                    std::thread::sleep(Duration::from_millis(5));
                }
            }
            let _ = thread.join();
        }
    }
}

type FeedbackTuple = (Vec<f64>, Vec<f64>, Vec<f64>, Option<f64>);
/// Longest interval during which the complete measured tuple did not change.
#[derive(Default)]
struct FeedbackChange {
    previous: Option<FeedbackTuple>,
    changed_at: Option<Instant>,
}
impl FeedbackChange {
    fn observe(&mut self, m: &Measured, now: Instant) -> Duration {
        let current = (
            m.joints.clone(),
            m.velocities.clone(),
            m.efforts.clone(),
            m.carriage.as_ref().map(|c| c.position_m),
        );
        if self.previous.as_ref() != Some(&current) {
            self.previous = Some(current);
            self.changed_at = Some(now);
        }
        self.changed_at
            .map_or(Duration::ZERO, |at| now.saturating_duration_since(at))
    }
}

/// Where a qualified carriage returns at Sleep: the staged pose's carriage,
/// the profile's rest. An unqualified carriage is never commanded away from
/// its measured value, so its landing carries none.
fn carriage_rest<B: crate::ArmBackend>(backend: &B, staged: &[f64; 7]) -> Option<f64> {
    backend.carriage_qualified().then_some(staged[6])
}
fn wall_ns() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_nanos() as u64)
        .unwrap_or(0)
}

#[cfg(test)]
#[path = "worker_offline_tests.rs"]
mod offline_tests;

#[cfg(test)]
#[path = "worker_guide_tests.rs"]
mod guide_tests;

#[cfg(test)]
mod tests {
    use super::*;
    /// The search's freshness fault names the channel that lapsed, and the
    /// report agrees with `contact_is_valid` at every step.
    #[test]
    fn staleness_names_the_channel_that_lapsed() {
        let limit = Duration::from_millis(25);
        let mut status = Status::default();
        assert_eq!(
            status.staleness(limit).as_deref(),
            Some("no measurement yet")
        );
        status.measured = Some(Measured {
            carriage: None,
            joints: vec![0.0; 6],
            velocities: vec![0.0; 6],
            efforts: vec![0.0; 6],
            dynamics: None,
            mode: Mode::Position,
            error: String::new(),
        });
        status.measured_at = Some(Instant::now() - Duration::from_millis(40));
        let stale = status.staleness(limit).unwrap();
        assert!(
            stale.starts_with("measurement ") && stale.ends_with("limit 25 ms"),
            "{stale}"
        );
        status.measured_at = Some(Instant::now());
        assert_eq!(status.staleness(limit).as_deref(), Some("no contact meter"));
        let mut meter = crate::contact::ContactEvaluation {
            armed: false,
            assessable: false,
            effort_assessable: false,
            deflection_exceeded: false,
            contact_n: 0.0,
            baseline_n: None,
            deflection_m: 0.0,
            trip: false,
        };
        status.contact = Some(meter);
        assert!(status.staleness(limit).unwrap().contains("not armed"));
        meter.armed = true;
        status.contact = Some(meter);
        assert!(status.staleness(limit).unwrap().contains("not assessable"));
        meter.assessable = true;
        status.contact = Some(meter);
        status.contact_at = Some(Instant::now() - Duration::from_millis(40));
        assert!(
            status
                .staleness(limit)
                .unwrap()
                .starts_with("contact estimate ")
        );
        assert!(!status.contact_is_valid(limit));
        status.contact_at = Some(Instant::now());
        assert_eq!(status.staleness(limit), None);
        assert!(status.contact_is_valid(limit));
        status.fault = Some("boom".into());
        assert_eq!(
            status.staleness(limit).as_deref(),
            Some("worker fault: boom")
        );
        assert!(!status.contact_is_valid(limit));
    }
    #[test]
    fn stream_timing_excludes_startup_and_retains_its_completed_tick() {
        let mut status = Status::default();
        status.record_lateness(140_000, false);
        status.stream_progress = Some(StreamProgress {
            sequence: 1,
            total: 1,
            submitted: 1,
            last_pen: Some(false),
            late_ticks: 0,
            max_late_us: 0,
        });
        status.completed = true;
        status.record_lateness(300, true);
        status.record_lateness(90_000, false);
        let stream = status.stream_progress.unwrap();
        assert_eq!((stream.late_ticks, stream.max_late_us), (1, 300));
        assert_eq!((status.late_ticks, status.max_late_us), (3, 140_000));
    }
    use crate::{ArmConfig, MockArm, Mode};
    use std::sync::atomic::AtomicI32;
    struct ReseedProbe {
        arm: MockArm,
        calls: Arc<Mutex<Vec<&'static str>>>,
        entered: mpsc::SyncSender<(f64, f64)>,
    }
    impl ArmBackend for ReseedProbe {
        fn connect(&mut self, config: &ArmConfig) -> Result<()> {
            self.arm.connect(config)
        }
        fn measured(&mut self) -> Result<Measured> {
            self.arm.measured()
        }
        fn set_mode(&mut self, mode: Mode) -> Result<()> {
            self.arm.set_mode(mode)
        }
        fn reseed_hold(&mut self, max_velocity: f64, envelope: f64) -> Result<Duration> {
            self.arm.set_mode(Mode::Position)?;
            self.arm.hold()?;
            self.entered.send((max_velocity, envelope)).unwrap();
            Ok(Duration::from_millis(120))
        }
        fn move_j(&mut self, target: &[f64], seconds: f64) -> Result<()> {
            self.arm.move_j(target, seconds)
        }
        fn retract_carriage(&mut self, target: f64, seconds: f64) -> Result<()> {
            self.arm.retract_carriage(target, seconds)
        }
        fn stream(&mut self, sample: &Sample) -> Result<()> {
            self.arm.stream(sample)
        }
        fn hold(&mut self) -> Result<()> {
            self.calls.lock().unwrap().push("measured hold");
            self.arm.hold()
        }
        fn finish_joint_stream(&mut self) -> Result<()> {
            self.calls.lock().unwrap().push("retain target");
            self.arm.finish_joint_stream()
        }
        fn clear_error(&mut self) -> Result<()> {
            self.arm.clear_error()
        }
        fn load_config(&mut self, path: &std::path::Path) -> Result<()> {
            self.arm.load_config(path)
        }
    }
    #[test]
    fn timed_reseed_waits_for_completion_and_stop_cancels_target_retention() {
        for stopped in [false, true] {
            let mut arm = MockArm::default();
            arm.connect(&ArmConfig { joints: 6 }).unwrap();
            arm.set_mode(Mode::Position).unwrap();
            let stop = Arc::new(AtomicI32::new(estop::OK));
            let calls = Arc::new(Mutex::new(Vec::new()));
            let (entered, ready) = mpsc::sync_channel(1);
            let mut worker = Worker::spawn(
                Control {
                    backend: ReseedProbe {
                        arm,
                        calls: calls.clone(),
                        entered,
                    },
                    estop: stop.clone(),
                    max_velocity: 0.1,
                    max_tracking_error: crate::TRACKING_ERROR_LIMIT_RAD,
                    max_contact: 1.0,
                    envelope: 3.3,
                },
                Duration::from_micros(2500),
                MotionGuard::new(1.0, 9.0, 0.5, 0.5, 8),
            )
            .unwrap();
            let sequence = worker.submit(Primitive::Reseed).unwrap();
            assert_eq!(
                ready.recv_timeout(Duration::from_secs(1)).unwrap(),
                (0.1, 3.3)
            );
            let deadline = Instant::now() + Duration::from_secs(1);
            while worker.status().sequence != sequence {
                assert!(Instant::now() < deadline);
                thread::sleep(Duration::from_millis(1));
            }
            assert!(!worker.status().completed);
            assert!(calls.lock().unwrap().is_empty());
            if stopped {
                stop.store(estop::PRESSED, Ordering::SeqCst);
            }
            loop {
                let status = worker.status();
                if status.sequence == sequence && status.completed {
                    assert_eq!(status.fault.is_some(), stopped);
                    break;
                }
                assert!(Instant::now() < deadline);
                thread::sleep(Duration::from_millis(2));
            }
            if stopped {
                thread::sleep(Duration::from_millis(150));
                stop.store(estop::OK, Ordering::SeqCst);
                thread::sleep(Duration::from_millis(10));
                let calls = calls.lock().unwrap();
                assert!(!calls.is_empty());
                assert!(calls.iter().all(|call| *call == "measured hold"));
                assert!(worker.status().fault.is_some());
            } else {
                assert_eq!(*calls.lock().unwrap(), vec!["retain target"]);
            }
            assert_eq!(worker.shutdown_idle().unwrap().mode, Mode::Idle);
        }
    }
    #[test]
    fn native_shutdown_kills_blocked_worker_process() {
        const CHILD: &str = "TATBOT_TEST_BLOCKED_NATIVE_SHUTDOWN";
        if std::env::var_os(CHILD).is_some() {
            let (entered, ready) = std::sync::mpsc::channel();
            let mut worker = Worker::spawn_with::<MockArm, _>(
                move || {
                    entered.send(()).unwrap();
                    loop {
                        std::thread::park();
                    }
                },
                Duration::from_micros(2500),
                MotionGuard::new(2.0, 15.0, 0.5, 0.5, 8),
            )
            .unwrap();
            ready.recv_timeout(Duration::from_secs(1)).unwrap();
            worker.native_shutdown_timeout = Some(Duration::from_millis(50));
            drop(worker);
            panic!("blocked native worker was detached");
        }
        let mut child = std::process::Command::new(std::env::current_exe().unwrap())
            .args([
                "--exact",
                "worker::tests::native_shutdown_kills_blocked_worker_process",
                "--nocapture",
            ])
            .env(CHILD, "1")
            .stdout(std::process::Stdio::piped())
            .stderr(std::process::Stdio::piped())
            .spawn()
            .unwrap();
        let deadline = Instant::now() + Duration::from_secs(3);
        while child.try_wait().unwrap().is_none() {
            if Instant::now() >= deadline {
                child.kill().unwrap();
                child.wait().unwrap();
                panic!("native worker shutdown hung its process");
            }
            std::thread::sleep(Duration::from_millis(10));
        }
        let output = child.wait_with_output().unwrap();
        assert_eq!(output.status.code(), Some(5));
        assert!(String::from_utf8_lossy(&output.stderr).contains("landing is unverified"));
    }
    struct StreamProbe {
        arm: MockArm,
        seen: Arc<Mutex<Vec<crate::JointSample>>>,
    }
    #[test]
    fn air_carriage_stream_crosses_reserve_boundary_and_estop_discards_tail() {
        for stopped in [false, true] {
            let mut arm = MockArm::default();
            arm.connect(&ArmConfig { joints: 6 }).unwrap();
            arm.set_mode(Mode::Position).unwrap();
            let mut seed = [0.0; 7];
            seed[6] = 0.0036;
            arm.state.carriage.as_mut().unwrap().position_m = seed[6];
            arm.state.carriage.as_mut().unwrap().target_m = Some(seed[6]);
            let stop = Arc::new(AtomicI32::new(estop::OK));
            let mut worker = Worker::spawn(
                Control {
                    backend: arm,
                    estop: stop.clone(),
                    max_velocity: 1.0,
                    max_tracking_error: crate::TRACKING_ERROR_LIMIT_RAD,
                    max_contact: 1.0,
                    envelope: 1.0,
                },
                Duration::from_micros(2500),
                MotionGuard::new(1.0, 9.0, 0.5, 0.5, 8),
            )
            .unwrap();
            let samples = crate::JointSample::carriage_transfer(seed, 0.0034).unwrap();
            let total = samples.len();
            let sequence = worker
                .submit(Primitive::StreamCarriage {
                    seed,
                    expected_sequence: 0,
                    samples,
                })
                .unwrap();
            let deadline = Instant::now() + Duration::from_secs(3);
            loop {
                let status = worker.status();
                if stopped && status.commands >= 10 {
                    stop.store(estop::PRESSED, Ordering::SeqCst);
                }
                if status.sequence == sequence && status.completed {
                    let progress = status.stream_progress.unwrap();
                    if stopped {
                        assert_eq!(status.fault.as_deref(), Some("e-stop"));
                        assert!(progress.submitted < total);
                    } else {
                        assert!(status.fault.is_none(), "{:?}", status.fault);
                        assert_eq!(progress.submitted, total);
                        assert!(
                            (status.measured.unwrap().carriage.unwrap().position_m - 0.0034).abs()
                                < 1e-12
                        );
                    }
                    break;
                }
                assert!(Instant::now() < deadline);
                thread::sleep(Duration::from_millis(1));
            }
        }
    }
    #[test]
    fn air_carriage_transfer_is_judged_by_deflection_and_rearms_where_it_settles() {
        // 24 N of stick-slip effort against a -7 N baseline (2026-09-16's
        // transfer) must not trip an air transfer; once it settles, the
        // baseline re-arms at the new effort and strokes see no phantom.
        let mut arm = MockArm::default();
        arm.connect(&ArmConfig { joints: 6 }).unwrap();
        arm.set_mode(Mode::Position).unwrap();
        arm.faults.carriage_effort_n = Some(-7.0);
        let mut seed = [0.0; 7];
        seed[6] = 0.0006;
        arm.state.carriage.as_mut().unwrap().position_m = seed[6];
        arm.state.carriage.as_mut().unwrap().target_m = Some(seed[6]);
        let stop = Arc::new(AtomicI32::new(estop::OK));
        let mut worker = Worker::spawn(
            Control {
                backend: arm,
                estop: stop.clone(),
                max_velocity: 1.0,
                max_tracking_error: crate::TRACKING_ERROR_LIMIT_RAD,
                max_contact: 1.0,
                envelope: 1.0,
            },
            Duration::from_micros(2500),
            MotionGuard::new(1.0, 9.0, 0.5, 0.5, 8),
        )
        .unwrap();
        let armed = Instant::now() + Duration::from_secs(4);
        while !worker.status().contact.is_some_and(|c| c.armed) {
            assert!(Instant::now() < armed, "baseline never armed");
            thread::sleep(Duration::from_millis(5));
        }
        assert!((worker.status().contact.unwrap().baseline_n.unwrap() + 7.0).abs() < 1e-9);
        // The transfer streams while the carriage's effort winds up 31 N past the baseline.
        let injected = worker
            .submit(Primitive::Inject(crate::Faults {
                carriage_effort_n: Some(24.0),
                ..Default::default()
            }))
            .unwrap();
        while worker.status().sequence != injected {
            thread::sleep(Duration::from_millis(1));
        }
        let samples = crate::JointSample::carriage_transfer(seed, 0.0025).unwrap();
        let total = samples.len();
        let sequence = worker
            .submit(Primitive::StreamCarriage {
                seed,
                expected_sequence: injected,
                samples,
            })
            .unwrap();
        let deadline = Instant::now() + Duration::from_secs(4);
        loop {
            let status = worker.status();
            // The tick that accepts the seed evaluates contact before it
            // streams the first sample, so that one status still carries an
            // assessable evaluation; from the next tick the effort channel is
            // off and nothing trips (a trip needs 40 consecutive ticks).
            let submitted = status.stream_progress.as_ref().map_or(0, |p| p.submitted);
            if let Some(contact) = status.contact
                && status.sequence == sequence
                && submitted >= 2
                && !status.completed
            {
                assert!(!contact.effort_assessable && !contact.trip, "{contact:?}");
            }
            if status.sequence == sequence && status.completed {
                assert!(status.fault.is_none(), "{:?}", status.fault);
                assert_eq!(submitted, total);
                break;
            }
            assert!(Instant::now() < deadline);
            thread::sleep(Duration::from_millis(1));
        }
        // Settled: the baseline re-arms at the effort the carriage now holds.
        let rearmed = Instant::now() + Duration::from_secs(4);
        loop {
            let contact = worker.status().contact.unwrap();
            if contact.armed
                && contact.effort_assessable
                && (contact.baseline_n.unwrap() - 24.0).abs() < 1e-9
            {
                assert!(contact.contact_n.abs() < 1e-9 && !contact.trip);
                break;
            }
            assert!(
                Instant::now() < rearmed,
                "baseline did not re-arm where the carriage settled: {contact:?}"
            );
            thread::sleep(Duration::from_millis(5));
        }
    }
    #[test]
    fn completed_joint_stream_keeps_carriage_target_but_stop_freezes_measurement() {
        let mut arm = MockArm::default();
        arm.connect(&ArmConfig { joints: 6 }).unwrap();
        arm.set_mode(Mode::Position).unwrap();
        arm.state.carriage.as_mut().unwrap().position_m = 0.00198;
        arm.faults.carriage_stuck = true;
        let stop = Arc::new(AtomicI32::new(estop::OK));
        let mut worker = Worker::spawn(
            Control {
                backend: arm,
                estop: stop.clone(),
                max_velocity: 1.0,
                max_tracking_error: crate::TRACKING_ERROR_LIMIT_RAD,
                max_contact: 1.0,
                envelope: 1.0,
            },
            Duration::from_micros(2500),
            MotionGuard::new(1.0, 9.0, 0.5, 0.5, 8),
        )
        .unwrap();
        let seed = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.00198];
        let mut target = seed;
        target[6] = 0.002;
        for previous in 0..3 {
            let sequence = worker
                .submit(Primitive::StreamJoint {
                    seed,
                    expected_sequence: previous,
                    permit: None,
                    samples: vec![
                        crate::JointSample {
                            positions: target,
                            velocities: [0.0; 7],
                            dt_s: 0.0025,
                            pen: false,
                        };
                        8
                    ],
                })
                .unwrap();
            let deadline = Instant::now() + Duration::from_secs(1);
            while worker.status().sequence != sequence || !worker.status().completed {
                assert!(Instant::now() < deadline);
                thread::sleep(Duration::from_millis(1));
            }
            let status = worker.status();
            assert!(status.fault.is_none(), "{:?}", status.fault);
            assert_eq!(
                status.measured.unwrap().carriage.unwrap().target_m,
                Some(0.002)
            );
        }
        stop.store(estop::PRESSED, Ordering::SeqCst);
        let deadline = Instant::now() + Duration::from_secs(1);
        while worker.status().fault.is_none() {
            assert!(Instant::now() < deadline);
            thread::sleep(Duration::from_millis(1));
        }
        let status = worker.status();
        assert_eq!(status.commands, 24);
        assert_eq!(
            status.measured.unwrap().carriage.unwrap().target_m,
            Some(0.00198)
        );
    }
    impl ArmBackend for StreamProbe {
        fn connect(&mut self, c: &ArmConfig) -> Result<()> {
            self.arm.connect(c)
        }
        fn measured(&mut self) -> Result<Measured> {
            self.arm.measured()
        }
        fn set_mode(&mut self, m: Mode) -> Result<()> {
            self.arm.set_mode(m)
        }
        fn move_j(&mut self, q: &[f64], t: f64) -> Result<()> {
            self.arm.move_j(q, t)
        }
        fn retract_carriage(&mut self, q: f64, t: f64) -> Result<()> {
            self.arm.retract_carriage(q, t)
        }
        fn stream(&mut self, s: &Sample) -> Result<()> {
            self.arm.stream(s)
        }
        fn stream_joint(&mut self, s: &crate::JointSample) -> Result<()> {
            self.arm.stream_joint(s)?;
            self.seen.lock().unwrap().push(s.clone());
            Ok(())
        }
        fn hold(&mut self) -> Result<()> {
            self.arm.hold()
        }
        fn clear_error(&mut self) -> Result<()> {
            self.arm.clear_error()
        }
        fn load_config(&mut self, p: &std::path::Path) -> Result<()> {
            self.arm.load_config(p)
        }
    }
    struct DelayedSeedProbe {
        probe: StreamProbe,
        pause: Arc<AtomicBool>,
        entered: mpsc::SyncSender<()>,
        release: mpsc::Receiver<()>,
    }
    impl ArmBackend for DelayedSeedProbe {
        fn connect(&mut self, config: &ArmConfig) -> Result<()> {
            self.probe.connect(config)
        }
        fn measured(&mut self) -> Result<Measured> {
            if self.pause.swap(false, Ordering::AcqRel) {
                self.entered.send(()).unwrap();
                self.release.recv().unwrap();
            }
            self.probe.measured()
        }
        fn set_mode(&mut self, mode: Mode) -> Result<()> {
            self.probe.set_mode(mode)
        }
        fn move_j(&mut self, target: &[f64], goal_time_s: f64) -> Result<()> {
            self.probe.move_j(target, goal_time_s)
        }
        fn retract_carriage(&mut self, target_m: f64, goal_time_s: f64) -> Result<()> {
            self.probe.retract_carriage(target_m, goal_time_s)
        }
        fn stream(&mut self, sample: &Sample) -> Result<()> {
            self.probe.stream(sample)
        }
        fn stream_joint(&mut self, sample: &crate::JointSample) -> Result<()> {
            self.probe.stream_joint(sample)
        }
        fn hold(&mut self) -> Result<()> {
            self.probe.hold()
        }
        fn clear_error(&mut self) -> Result<()> {
            self.probe.clear_error()
        }
        fn load_config(&mut self, path: &std::path::Path) -> Result<()> {
            self.probe.load_config(path)
        }
    }
    #[test]
    fn invalid_target_during_delayed_controller_handoff_refuses_sample_zero() {
        let mut arm = MockArm::default();
        arm.connect(&ArmConfig { joints: 6 }).unwrap();
        arm.set_mode(Mode::Position).unwrap();
        let seen = Arc::new(Mutex::new(Vec::new()));
        let pause = Arc::new(AtomicBool::new(false));
        let valid = Arc::new(AtomicBool::new(true));
        let (entered_tx, entered_rx) = mpsc::sync_channel(1);
        let (release_tx, release_rx) = mpsc::sync_channel(1);
        let mut worker = Worker::spawn(
            Control {
                backend: DelayedSeedProbe {
                    probe: StreamProbe {
                        arm,
                        seen: seen.clone(),
                    },
                    pause: pause.clone(),
                    entered: entered_tx,
                    release: release_rx,
                },
                estop: Arc::new(AtomicI32::new(estop::OK)),
                max_velocity: 1.0,
                max_tracking_error: crate::TRACKING_ERROR_LIMIT_RAD,
                max_contact: 1.0,
                envelope: 1.0,
            },
            Duration::from_micros(2500),
            MotionGuard::new(1.0, 9.0, 0.5, 0.5, 8),
        )
        .unwrap();
        let deadline = Instant::now() + Duration::from_secs(1);
        while worker.status().measured.is_none() {
            assert!(Instant::now() < deadline);
            thread::sleep(Duration::from_millis(1));
        }
        let prior_commands = worker.status().commands;
        pause.store(true, Ordering::Release);
        entered_rx.recv_timeout(Duration::from_secs(1)).unwrap();
        let mut permit =
            StreamPermit::with_phase_gate(8, |_, _| Err(Error("unexpected phase boundary".into())))
                .unwrap();
        StreamPermit::bind_start_guard(&mut permit, {
            let valid = valid.clone();
            move |_| {
                if valid.load(Ordering::Acquire) {
                    Ok(())
                } else {
                    Err(Error("material target moved".into()))
                }
            }
        })
        .unwrap();
        permit.grant(8).unwrap();
        let seed = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.002];
        let sequence = worker
            .submit(Primitive::StreamJoint {
                seed,
                expected_sequence: 0,
                permit: Some(permit),
                samples: vec![
                    crate::JointSample {
                        positions: seed,
                        velocities: [0.0; 7],
                        dt_s: 0.0025,
                        pen: false,
                    };
                    8
                ],
            })
            .unwrap();
        // The session already submitted the queue, but feedback held the
        // worker before its sample-zero authorization.
        valid.store(false, Ordering::Release);
        release_tx.send(()).unwrap();
        let deadline = Instant::now() + Duration::from_secs(1);
        let status = loop {
            let status = worker.status();
            if status.sequence == sequence && status.completed {
                break status;
            }
            assert!(Instant::now() < deadline);
            thread::sleep(Duration::from_millis(1));
        };
        assert!(status.stream_seed.as_ref().unwrap().accepted);
        assert!(status.stream_start_guard_refused);
        assert_eq!(status.commands, prior_commands);
        assert_eq!(status.stream_progress.unwrap().submitted, 0);
        assert!(status.fault.unwrap().contains("material target moved"));
        assert!(seen.lock().unwrap().is_empty());
        valid.store(true, Ordering::Release);
        thread::sleep(Duration::from_millis(25));
        assert!(seen.lock().unwrap().is_empty(), "discarded queue resumed");
    }
    #[test]
    fn owner_phase_gate_preserves_samples_while_observer_is_delayed() {
        for refuse in [false, true] {
            let mut arm = MockArm::default();
            arm.connect(&ArmConfig { joints: 6 }).unwrap();
            arm.set_mode(Mode::Position).unwrap();
            let seen = Arc::new(Mutex::new(Vec::new()));
            let mut worker = Worker::spawn(
                Control {
                    backend: StreamProbe {
                        arm,
                        seen: seen.clone(),
                    },
                    estop: Arc::new(AtomicI32::new(estop::OK)),
                    max_velocity: 1.0,
                    max_tracking_error: crate::TRACKING_ERROR_LIMIT_RAD,
                    max_contact: 1.0,
                    envelope: 1.0,
                },
                Duration::from_micros(2500),
                MotionGuard::new(1.0, 9.0, 0.5, 0.5, 8),
            )
            .unwrap();
            let (sender, receiver) = mpsc::sync_channel(2);
            let permit = StreamPermit::with_phase_gate(8, move |index, status| {
                assert!(status.measurement_is_fresh(Duration::from_millis(250)));
                assert_eq!(status.stream_progress.as_ref().unwrap().submitted, index);
                if refuse {
                    return Err(Error("test contact refusal".into()));
                }
                sender.try_send(index).unwrap();
                match index {
                    3 => Ok(6),
                    6 => Ok(8),
                    _ => Err(Error("unexpected phase".into())),
                }
            })
            .unwrap();
            permit.grant(3).unwrap();
            let seed = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.002];
            let samples = (0..8)
                .map(|i| crate::JointSample {
                    positions: seed,
                    velocities: [0.0; 7],
                    dt_s: 0.0025,
                    pen: (3..6).contains(&i),
                })
                .collect::<Vec<_>>();
            worker
                .submit(Primitive::StreamJoint {
                    seed,
                    expected_sequence: 0,
                    permit: Some(permit),
                    samples: samples.clone(),
                })
                .unwrap();
            // The journal/observer does not run at either 2.5 ms phase edge.
            thread::sleep(Duration::from_millis(150));
            let captured = seen.lock().unwrap();
            assert_eq!(captured.len(), if refuse { 3 } else { 8 });
            for (a, b) in captured.iter().zip(&samples) {
                assert_eq!(a.positions, b.positions);
                assert_eq!(a.velocities, b.velocities);
                assert_eq!(a.pen, b.pen);
                assert_eq!(a.dt_s, b.dt_s);
            }
            let status = worker.status();
            assert_eq!(status.stream_progress.unwrap().submitted, captured.len());
            assert_eq!(status.fault.is_some(), refuse);
            if !refuse {
                assert_eq!(receiver.try_iter().collect::<Vec<_>>(), vec![3, 6]);
            }
        }
    }

    #[test]
    fn measured_stop_holds_at_exact_sample_without_replaying_the_tail() {
        for stop_at in [0, 3, 7, 8] {
            let mut arm = MockArm::default();
            arm.connect(&ArmConfig { joints: 6 }).unwrap();
            arm.set_mode(Mode::Position).unwrap();
            let seen = Arc::new(Mutex::new(Vec::new()));
            let mut worker = Worker::spawn(
                Control {
                    backend: StreamProbe {
                        arm,
                        seen: seen.clone(),
                    },
                    estop: Arc::new(AtomicI32::new(estop::OK)),
                    max_velocity: 1.0,
                    max_tracking_error: crate::TRACKING_ERROR_LIMIT_RAD,
                    max_contact: 1.0,
                    envelope: 1.0,
                },
                Duration::from_micros(2500),
                MotionGuard::new(1.0, 9.0, 0.5, 0.5, 8),
            )
            .unwrap();
            let permit = StreamPermit::with_stop_guard(8, move |index, status| {
                assert!(status.measurement_is_fresh(Duration::from_millis(250)));
                assert_eq!(status.stream_progress.as_ref().unwrap().submitted, index);
                Ok(index == stop_at)
            })
            .unwrap();
            let seed = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.002];
            let samples = vec![
                crate::JointSample {
                    positions: seed,
                    velocities: [0.0; 7],
                    dt_s: 0.0025,
                    pen: false,
                };
                8
            ];
            worker
                .submit(Primitive::StreamJoint {
                    seed,
                    expected_sequence: 0,
                    permit: Some(permit.clone()),
                    samples: samples.clone(),
                })
                .unwrap();
            thread::sleep(Duration::from_millis(100));
            let status = worker.status();
            assert!(
                status.completed && status.fault.is_none(),
                "{:?}",
                status.fault
            );
            assert_eq!(seen.lock().unwrap().len(), stop_at);
            assert_eq!(status.stream_progress.unwrap().submitted, stop_at);
            assert_eq!(permit.stopped_at(), (stop_at < 8).then_some(stop_at));
            permit.grant(8).unwrap();
            thread::sleep(Duration::from_millis(30));
            assert_eq!(seen.lock().unwrap().len(), stop_at);
            assert!(
                worker
                    .submit(Primitive::StreamJoint {
                        seed,
                        expected_sequence: 1,
                        permit: Some(permit),
                        samples,
                    })
                    .is_err()
            );
        }
    }

    #[test]
    fn phase_permit_faults_at_exact_boundary_and_late_grant_cannot_resume() {
        for granted in [0, 3, 8] {
            let mut arm = MockArm::default();
            arm.connect(&ArmConfig { joints: 6 }).unwrap();
            arm.set_mode(Mode::Position).unwrap();
            let seen = Arc::new(Mutex::new(Vec::new()));
            let mut worker = Worker::spawn(
                Control {
                    backend: StreamProbe {
                        arm,
                        seen: seen.clone(),
                    },
                    estop: Arc::new(AtomicI32::new(estop::OK)),
                    max_velocity: 1.0,
                    max_tracking_error: crate::TRACKING_ERROR_LIMIT_RAD,
                    max_contact: 1.0,
                    envelope: 1.0,
                },
                Duration::from_micros(2500),
                MotionGuard::new(1.0, 9.0, 0.5, 0.5, 8),
            )
            .unwrap();
            let permit = StreamPermit::new(8).unwrap();
            permit.grant(granted).unwrap();
            assert!(permit.grant(9).is_err());
            if granted > 0 {
                assert!(permit.grant(granted - 1).is_err());
            }
            let seed = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.002];
            let samples = vec![
                crate::JointSample {
                    positions: seed,
                    velocities: [0.0; 7],
                    dt_s: 0.0025,
                    pen: false,
                };
                8
            ];
            let sequence = worker
                .submit(Primitive::StreamJoint {
                    seed,
                    expected_sequence: 0,
                    permit: Some(permit.clone()),
                    samples: samples.clone(),
                })
                .unwrap();
            let deadline = Instant::now() + Duration::from_secs(1);
            while worker.status().sequence != sequence || !worker.status().completed {
                assert!(Instant::now() < deadline);
                thread::sleep(Duration::from_millis(1));
            }
            let status = worker.status();
            assert_eq!(seen.lock().unwrap().len(), granted);
            assert_eq!(status.stream_progress.unwrap().submitted, granted);
            if granted < 8 {
                assert!(status.fault.unwrap().contains(&format!("sample {granted}")));
                permit.grant(8).unwrap();
                thread::sleep(Duration::from_millis(20));
                assert_eq!(seen.lock().unwrap().len(), granted);
            } else {
                assert!(status.fault.is_none());
            }
            assert!(
                worker
                    .submit(Primitive::StreamJoint {
                        seed,
                        expected_sequence: sequence,
                        permit: Some(permit),
                        samples,
                    })
                    .unwrap_err()
                    .to_string()
                    .contains("already used")
            );
        }
    }
    #[test]
    fn seven_axis_feedforward_reaches_backend_and_stop_discards_queue() {
        let mut arm = MockArm::default();
        arm.connect(&ArmConfig { joints: 6 }).unwrap();
        arm.set_mode(Mode::Position).unwrap();
        let seen = Arc::new(Mutex::new(Vec::new()));
        let stop = Arc::new(AtomicI32::new(estop::OK));
        let mut worker = Worker::spawn(
            Control {
                backend: StreamProbe {
                    arm,
                    seen: seen.clone(),
                },
                estop: stop.clone(),
                max_velocity: 1.0,
                max_tracking_error: crate::TRACKING_ERROR_LIMIT_RAD,
                max_contact: 1.0,
                envelope: 1.0,
            },
            Duration::from_micros(2500),
            MotionGuard::new(1.0, 9.0, 0.5, 0.5, 8),
        )
        .unwrap();
        let samples: Vec<_> = (1..=200)
            .map(|i| crate::JointSample {
                positions: [
                    i as f64 * 0.0001,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.002 + i as f64 * 0.00001 * 0.0025,
                ],
                velocities: [0.04, 0.0, 0.0, 0.0, 0.0, 0.0, 0.00001],
                dt_s: 0.0025,
                pen: true,
            })
            .collect();
        worker
            .submit(Primitive::StreamJoint {
                seed: [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.002],
                expected_sequence: 0,
                permit: None,
                samples: samples.clone(),
            })
            .unwrap();
        let deadline = Instant::now() + Duration::from_secs(2);
        while seen.lock().unwrap().len() < 10 {
            assert!(Instant::now() < deadline, "{:?}", worker.status());
            thread::sleep(Duration::from_millis(2));
        }
        stop.store(estop::PRESSED, Ordering::SeqCst);
        let pressed = Instant::now();
        while worker.status().fault.is_none() {
            assert!(pressed.elapsed() < Duration::from_millis(100));
            thread::sleep(Duration::from_millis(1));
        }
        let accepted = seen.lock().unwrap().clone();
        assert!(accepted.len() < 200);
        let progress = worker.status().stream_progress.unwrap();
        assert_eq!(progress.total, 200);
        assert_eq!(progress.submitted, accepted.len());
        assert_eq!(progress.sequence, worker.status().sequence);
        assert_eq!(progress.last_pen, Some(true));
        for (actual, expected) in accepted.iter().zip(samples.iter()) {
            assert_eq!(actual.positions, expected.positions);
            assert_eq!(actual.velocities, expected.velocities);
        }
        assert!(
            worker
                .status()
                .measured
                .unwrap()
                .carriage
                .unwrap()
                .position_m
                > 0.002
        );
        stop.store(estop::OK, Ordering::SeqCst);
        thread::sleep(Duration::from_millis(20));
        assert_eq!(seen.lock().unwrap().len(), accepted.len());
        assert_eq!(
            worker.status().stream_progress.unwrap().submitted,
            accepted.len()
        );
        let seq = worker
            .submit(Primitive::StreamJoint {
                seed: [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.002],
                expected_sequence: 0,
                permit: None,
                samples: vec![samples[0].clone()],
            })
            .unwrap();
        while worker.status().sequence != seq {
            assert!(Instant::now() < deadline);
            thread::sleep(Duration::from_millis(2));
        }
        assert!(worker.status().fault.unwrap().contains("re-seed required"));
        assert!(worker.status().stream_progress.is_none());
        assert_eq!(seen.lock().unwrap().len(), accepted.len());
    }
    #[test]
    fn stream_seed_rejects_intervening_command_and_joint_or_carriage_drift() {
        for case in 0..3 {
            let mut arm = MockArm::default();
            arm.connect(&ArmConfig { joints: 6 }).unwrap();
            arm.set_mode(Mode::Position).unwrap();
            let seen = Arc::new(Mutex::new(Vec::new()));
            let mut worker = Worker::spawn(
                Control {
                    backend: StreamProbe {
                        arm,
                        seen: seen.clone(),
                    },
                    estop: Arc::new(AtomicI32::new(estop::OK)),
                    max_velocity: 1.0,
                    max_tracking_error: crate::TRACKING_ERROR_LIMIT_RAD,
                    max_contact: 1.0,
                    envelope: 1.0,
                },
                Duration::from_micros(2500),
                MotionGuard::new(1.0, 9.0, 0.5, 0.5, 8),
            )
            .unwrap();
            let deadline = Instant::now() + Duration::from_secs(2);
            if case == 0 {
                let sequence = worker.submit(Primitive::Hold).unwrap();
                while worker.status().sequence != sequence || !worker.status().completed {
                    assert!(Instant::now() < deadline);
                    thread::sleep(Duration::from_millis(1));
                }
            }
            let mut seed = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.002];
            if case == 1 {
                // Twice the pen-down joint seed tolerance (10 mrad).
                seed[0] = 0.02;
            }
            if case == 2 {
                // 0.3 mm past the pen-down carriage seed tolerance (0.5 mm).
                seed[6] = 0.0028;
            }
            let sequence = worker
                .submit(Primitive::StreamJoint {
                    seed,
                    expected_sequence: 0,
                    permit: None,
                    samples: vec![crate::JointSample {
                        positions: [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.002],
                        velocities: [0.0; 7],
                        dt_s: 0.0025,
                        pen: true,
                    }],
                })
                .unwrap();
            while worker.status().sequence != sequence || !worker.status().completed {
                assert!(Instant::now() < deadline);
                thread::sleep(Duration::from_millis(1));
            }
            let status = worker.status();
            let receipt = status.stream_seed.unwrap();
            assert!(!receipt.accepted);
            assert_eq!(receipt.requested, seed);
            assert!(receipt.error.unwrap().contains(if case == 0 {
                "sequence changed"
            } else {
                "seed moved"
            }));
            assert_eq!(status.commands, 0);
            assert!(seen.lock().unwrap().is_empty());
            assert!(status.fault.is_some());
        }
    }
    struct ThreadBoundArm {
        arm: MockArm,
        // Rc intentionally makes this backend !Send and !Sync.
        owner: std::rc::Rc<std::thread::ThreadId>,
        dropped: Arc<Mutex<Option<std::thread::ThreadId>>>,
        holds: Arc<std::sync::atomic::AtomicUsize>,
        ignore_idle: bool,
        idle_delay: Option<Instant>,
        delay_idle: bool,
    }
    impl Drop for ThreadBoundArm {
        fn drop(&mut self) {
            assert_eq!(*self.owner, std::thread::current().id());
            *self.dropped.lock().unwrap() = Some(std::thread::current().id());
        }
    }
    impl ArmBackend for ThreadBoundArm {
        fn connect(&mut self, c: &ArmConfig) -> Result<()> {
            self.arm.connect(c)
        }
        fn measured(&mut self) -> Result<Measured> {
            assert_eq!(*self.owner, std::thread::current().id());
            if self
                .idle_delay
                .is_some_and(|at| at.elapsed() >= Duration::from_millis(80))
            {
                self.idle_delay = None;
                self.arm.set_mode(Mode::Idle)?;
            }
            self.arm.measured()
        }
        fn retract_carriage(&mut self, q: f64, t: f64) -> Result<()> {
            self.arm.retract_carriage(q, t)
        }
        fn set_mode(&mut self, mode: Mode) -> Result<()> {
            if self.delay_idle && mode == Mode::Idle {
                self.idle_delay = Some(Instant::now());
                return Ok(());
            }
            if mode == Mode::Position {
                self.idle_delay = None;
            }
            if self.ignore_idle && mode == Mode::Idle {
                return Ok(());
            }
            self.arm.set_mode(mode)
        }
        fn move_j(&mut self, q: &[f64], t: f64) -> Result<()> {
            self.arm.move_j(q, t)
        }
        fn stream(&mut self, s: &Sample) -> Result<()> {
            self.arm.stream(s)
        }
        fn hold(&mut self) -> Result<()> {
            self.holds.fetch_add(1, Ordering::SeqCst);
            self.arm.hold()
        }
        fn clear_error(&mut self) -> Result<()> {
            self.arm.clear_error()
        }
        fn load_config(&mut self, p: &std::path::Path) -> Result<()> {
            self.arm.load_config(p)
        }
    }
    #[test]
    fn non_send_backend_is_created_used_and_dropped_on_the_owner_thread() {
        let caller = std::thread::current().id();
        let dropped = Arc::new(Mutex::new(None));
        let receipt = dropped.clone();
        let worker = Worker::spawn_with(
            move || {
                let mut arm = MockArm::default();
                arm.connect(&ArmConfig { joints: 6 })?;
                arm.set_mode(Mode::Position)?;
                Ok(Control {
                    backend: ThreadBoundArm {
                        arm,
                        owner: std::rc::Rc::new(std::thread::current().id()),
                        dropped: receipt,
                        holds: Arc::new(std::sync::atomic::AtomicUsize::new(0)),
                        ignore_idle: false,
                        idle_delay: None,
                        delay_idle: false,
                    },
                    estop: Arc::new(AtomicI32::new(estop::OK)),
                    max_velocity: 1.0,
                    max_tracking_error: crate::TRACKING_ERROR_LIMIT_RAD,
                    max_contact: 1.0,
                    envelope: 1.0,
                })
            },
            Duration::from_micros(2500),
            MotionGuard::new(1.0, 9.0, 0.5, 0.5, 8),
        )
        .unwrap();
        let deadline = Instant::now() + Duration::from_secs(2);
        while worker.status().measured.is_none() {
            assert!(Instant::now() < deadline);
            std::thread::sleep(Duration::from_millis(1));
        }
        assert!(worker.status().fault.is_none());
        drop(worker);
        let owner = dropped.lock().unwrap().unwrap();
        assert_ne!(owner, caller);
    }
    #[test]
    fn terminal_idle_releases_under_stop_without_reseed_or_rehold() {
        for ignore_idle in [false, true] {
            let holds = Arc::new(std::sync::atomic::AtomicUsize::new(0));
            let held = holds.clone();
            let mut worker = Worker::spawn_with(
                move || {
                    let mut arm = MockArm::default();
                    arm.connect(&ArmConfig { joints: 6 })?;
                    arm.set_mode(Mode::Position)?;
                    Ok(Control {
                        backend: ThreadBoundArm {
                            arm,
                            owner: std::rc::Rc::new(thread::current().id()),
                            dropped: Arc::new(Mutex::new(None)),
                            holds: held,
                            ignore_idle,
                            idle_delay: None,
                            delay_idle: !ignore_idle,
                        },
                        estop: Arc::new(AtomicI32::new(estop::PRESSED)),
                        max_velocity: 1.0,
                        max_tracking_error: crate::TRACKING_ERROR_LIMIT_RAD,
                        max_contact: 1.0,
                        envelope: 1.0,
                    })
                },
                Duration::from_micros(2500),
                MotionGuard::new(1.0, 9.0, 0.5, 0.5, 8),
            )
            .unwrap();
            let result = worker.shutdown_idle();
            if ignore_idle {
                assert!(
                    result
                        .unwrap_err()
                        .to_string()
                        .contains("idle not confirmed")
                );
            } else {
                assert_eq!(result.unwrap().mode, Mode::Idle);
            }
            assert!(worker.thread.is_none());
            assert!(
                worker.submit(Primitive::Reseed).is_err(),
                "terminal release cannot restart"
            );
            let before = holds.load(Ordering::SeqCst);
            drop(worker);
            assert_eq!(holds.load(Ordering::SeqCst), before);
        }
    }

    #[test]
    fn idle_release_is_measured_latches_motion_and_shutdown_does_not_hold() {
        let holds = Arc::new(std::sync::atomic::AtomicUsize::new(0));
        let held = holds.clone();
        let mut worker = Worker::spawn_with(
            move || {
                let mut arm = MockArm::default();
                arm.connect(&ArmConfig { joints: 6 })?;
                arm.set_mode(Mode::Position)?;
                Ok(Control {
                    backend: ThreadBoundArm {
                        arm,
                        owner: std::rc::Rc::new(thread::current().id()),
                        dropped: Arc::new(Mutex::new(None)),
                        holds: held,
                        ignore_idle: false,
                        idle_delay: None,
                        delay_idle: true,
                    },
                    estop: Arc::new(AtomicI32::new(estop::OK)),
                    max_velocity: 1.0,
                    max_tracking_error: crate::TRACKING_ERROR_LIMIT_RAD,
                    max_contact: 1.0,
                    envelope: 1.0,
                })
            },
            Duration::from_micros(2500),
            MotionGuard::new(1.0, 9.0, 0.5, 0.5, 8),
        )
        .unwrap();
        let await_command = |worker: &Worker, sequence| {
            let deadline = Instant::now() + Duration::from_secs(1);
            loop {
                let status = worker.status();
                if status.sequence == sequence && status.completed {
                    return status;
                }
                assert!(Instant::now() < deadline);
                thread::sleep(Duration::from_millis(2));
            }
        };
        let sequence = worker.submit(Primitive::Idle).unwrap();
        let status = await_command(&worker, sequence);
        assert!(status.fault.is_none());
        assert_eq!(status.measured.unwrap().mode, Mode::Idle);
        thread::sleep(Duration::from_millis(30));
        assert!(worker.status().fault.is_none());
        let sequence = worker
            .submit(Primitive::MoveJ {
                target: vec![0.1; 6],
                goal_time_s: 0.1,
            })
            .unwrap();
        let status = await_command(&worker, sequence);
        assert!(status.fault.unwrap().contains("re-seed required"));
        assert_eq!(status.commands, 0);
        let sequence = worker.submit(Primitive::Reseed).unwrap();
        let status = await_command(&worker, sequence);
        assert!(status.fault.is_none());
        assert_eq!(status.measured.unwrap().mode, Mode::Position);
        let sequence = worker.submit(Primitive::Idle).unwrap();
        assert!(await_command(&worker, sequence).fault.is_none());
        let before = holds.load(Ordering::SeqCst);
        drop(worker);
        assert_eq!(holds.load(Ordering::SeqCst), before);
    }

    /// Session teardown after a verified landing: a viewer or status reader
    /// holding the published status cell must not delay the worker's own
    /// destruction, the released backend must not be re-held, and the backend
    /// is still destroyed on its owning thread with no command in flight.
    #[test]
    fn shutdown_completes_while_a_status_reader_is_blocked_and_does_not_rehold() {
        let holds = Arc::new(std::sync::atomic::AtomicUsize::new(0));
        let dropped = Arc::new(Mutex::new(None));
        let (held, destroyed) = (holds.clone(), dropped.clone());
        let mut worker = Worker::spawn_with(
            move || {
                let mut arm = MockArm::default();
                arm.connect(&ArmConfig { joints: 6 })?;
                arm.set_mode(Mode::Position)?;
                Ok(Control {
                    backend: ThreadBoundArm {
                        arm,
                        owner: std::rc::Rc::new(thread::current().id()),
                        dropped: destroyed,
                        holds: held,
                        ignore_idle: false,
                        idle_delay: None,
                        delay_idle: false,
                    },
                    estop: Arc::new(AtomicI32::new(estop::OK)),
                    max_velocity: 1.0,
                    max_tracking_error: crate::TRACKING_ERROR_LIMIT_RAD,
                    max_contact: 1.0,
                    envelope: 1.0,
                })
            },
            Duration::from_micros(2500),
            MotionGuard::new(1.0, 9.0, 0.5, 0.5, 8),
        )
        .unwrap();
        let sequence = worker.submit(Primitive::Idle).unwrap();
        let deadline = Instant::now() + Duration::from_secs(1);
        loop {
            let status = worker.status();
            if status.sequence == sequence && status.completed {
                assert!(status.fault.is_none());
                assert_eq!(status.measured.unwrap().mode, Mode::Idle);
                break;
            }
            assert!(Instant::now() < deadline);
            thread::sleep(Duration::from_millis(2));
        }
        // A reader that never returns: the guard is leaked on another thread
        // so the status lock stays held for the rest of the test.
        let blocked = worker.status.clone();
        let reader = thread::spawn(move || {
            let guard = blocked.lock().unwrap();
            std::mem::forget(guard);
        });
        reader.join().unwrap();
        assert!(
            worker.status.try_lock().is_err(),
            "reader still holds status"
        );
        let before = holds.load(Ordering::SeqCst);
        let started = Instant::now();
        drop(worker);
        assert!(
            started.elapsed() < Duration::from_secs(1),
            "worker destruction waited on a blocked status reader"
        );
        assert_eq!(
            holds.load(Ordering::SeqCst),
            before,
            "released arm was re-held"
        );
        let owner = dropped
            .lock()
            .unwrap()
            .expect("backend destroyed on shutdown");
        assert_ne!(owner, thread::current().id());
    }

    #[test]
    fn estop_interrupts_pending_idle_feedback_confirmation() {
        let stop = Arc::new(AtomicI32::new(estop::OK));
        let input = stop.clone();
        let mut worker = Worker::spawn_with(
            move || {
                let mut arm = MockArm::default();
                arm.connect(&ArmConfig { joints: 6 })?;
                arm.set_mode(Mode::Position)?;
                Ok(Control {
                    backend: ThreadBoundArm {
                        arm,
                        owner: std::rc::Rc::new(thread::current().id()),
                        dropped: Arc::new(Mutex::new(None)),
                        holds: Arc::new(AtomicUsize::new(0)),
                        ignore_idle: true,
                        idle_delay: None,
                        delay_idle: false,
                    },
                    estop: input,
                    max_velocity: 1.,
                    max_tracking_error: crate::TRACKING_ERROR_LIMIT_RAD,
                    max_contact: 1.,
                    envelope: 1.,
                })
            },
            Duration::from_micros(2500),
            MotionGuard::new(1., 9., 0.5, 0.5, 8),
        )
        .unwrap();
        let seq = worker.submit(Primitive::Idle).unwrap();
        thread::sleep(Duration::from_millis(30));
        let pressed = Instant::now();
        stop.store(estop::PRESSED, Ordering::Release);
        loop {
            let s = worker.status();
            if s.sequence == seq
                && s.completed
                && s.fault.as_ref().is_some_and(|f| f.contains("e-stop"))
            {
                break;
            }
            assert!(pressed.elapsed() < Duration::from_millis(200));
            thread::sleep(Duration::from_millis(2));
        }
    }
    #[test]
    fn idle_release_refuses_unexecuted_mode_command_and_latched_estop() {
        for stop in [estop::OK, estop::PRESSED] {
            let mut worker = Worker::spawn_with(
                move || {
                    let mut arm = MockArm::default();
                    arm.connect(&ArmConfig { joints: 6 })?;
                    arm.set_mode(Mode::Position)?;
                    Ok(Control {
                        backend: ThreadBoundArm {
                            arm,
                            owner: std::rc::Rc::new(thread::current().id()),
                            dropped: Arc::new(Mutex::new(None)),
                            holds: Arc::new(std::sync::atomic::AtomicUsize::new(0)),
                            ignore_idle: stop == estop::OK,
                            idle_delay: None,
                            delay_idle: false,
                        },
                        estop: Arc::new(AtomicI32::new(stop)),
                        max_velocity: 1.0,
                        max_tracking_error: crate::TRACKING_ERROR_LIMIT_RAD,
                        max_contact: 1.0,
                        envelope: 1.0,
                    })
                },
                Duration::from_micros(2500),
                MotionGuard::new(1.0, 9.0, 0.5, 0.5, 8),
            )
            .unwrap();
            let sequence = worker.submit(Primitive::Idle).unwrap();
            let deadline = Instant::now() + Duration::from_secs(1);
            loop {
                let status = worker.status();
                if status.sequence == sequence && status.completed {
                    assert!(status.fault.is_some());
                    assert_eq!(status.measured.unwrap().mode, Mode::Position);
                    assert_eq!(status.commands, 0);
                    break;
                }
                assert!(Instant::now() < deadline);
                thread::sleep(Duration::from_millis(2));
            }
        }
    }

    /// A standoff tool's trip, and a contact tool's on an unqualified
    /// carriage, latch and hold: no retract is commanded and none is claimed.
    #[test]
    fn a_standoff_trip_still_latches_and_holds_without_retract() {
        for (explicit, policy, qualified) in [
            (false, crate::ContactPolicy::Standoff, true),
            (true, crate::ContactPolicy::Standoff, true),
            (false, crate::ContactPolicy::Contact, false),
            (true, crate::ContactPolicy::Contact, false),
        ] {
            let mut arm = MockArm {
                contact_policy: policy,
                carriage_qualified: qualified,
                ..Default::default()
            };
            arm.connect(&ArmConfig { joints: 6 }).unwrap();
            arm.set_mode(Mode::Position).unwrap();
            if !explicit {
                arm.faults.carriage_deflection_m = Some(0.003);
            }
            let mut worker = Worker::spawn(
                Control {
                    backend: arm,
                    estop: Arc::new(AtomicI32::new(estop::OK)),
                    max_velocity: 1.0,
                    max_tracking_error: crate::TRACKING_ERROR_LIMIT_RAD,
                    max_contact: 1.0,
                    envelope: 1.0,
                },
                Duration::from_micros(2500),
                MotionGuard::new(1.0, 9.0, 0.5, 0.5, 8),
            )
            .unwrap();
            if explicit {
                worker.submit(Primitive::TripRetract).unwrap();
            }
            let deadline = Instant::now() + Duration::from_secs(2);
            loop {
                let status = worker.status();
                if status.fault.is_some() && status.completed {
                    assert_eq!(status.retract_verified, None);
                    assert_eq!(
                        status.measured.unwrap().carriage.unwrap().position_m,
                        if explicit { 0.002 } else { 0.005 }
                    );
                    break;
                }
                assert!(Instant::now() < deadline);
                thread::sleep(Duration::from_millis(1));
            }
        }
    }
    #[test]
    fn recovery_preparation_retains_retracted_carriage_and_refuses_invalid_targets() {
        let mut arm = MockArm::default();
        arm.connect(&ArmConfig { joints: 6 }).unwrap();
        arm.set_mode(Mode::Position).unwrap();
        arm.retract_carriage(0.032, 0.6).unwrap();
        let stop = Arc::new(AtomicI32::new(estop::OK));
        let mut worker = Worker::spawn(
            Control {
                backend: arm,
                estop: stop.clone(),
                max_velocity: 1.0,
                max_tracking_error: crate::TRACKING_ERROR_LIMIT_RAD,
                max_contact: 1.0,
                envelope: 1.0,
            },
            Duration::from_micros(2500),
            MotionGuard::new(1.0, 9.0, 0.5, 0.5, 8),
        )
        .unwrap();
        let await_command = |worker: &Worker, sequence| {
            let deadline = Instant::now() + Duration::from_secs(1);
            loop {
                let status = worker.status();
                if status.sequence == sequence && status.completed {
                    return status;
                }
                assert!(Instant::now() < deadline);
                thread::sleep(Duration::from_millis(2));
            }
        };
        let sequence = worker
            .submit(Primitive::PrepareRecovery {
                staged: [0.0, 0.2, -0.3, 0.0, 0.0, 0.5, 0.002],
            })
            .unwrap();
        let status = await_command(&worker, sequence);
        assert!(status.fault.is_none());
        let targets = status.recovery_targets.unwrap();
        assert_eq!(targets.staged[6], 0.032);
        // The mock's carriage is qualified: it returns to the staged rest at Sleep.
        assert_eq!(targets.sleep, [0.0, 0.0, 0.0, 0.0, 0.0, 0.5, 0.002]);
        assert_eq!(status.commands, 0);
        let sequence = worker
            .submit(Primitive::PrepareRecovery { staged: [4.0; 7] })
            .unwrap();
        let status = await_command(&worker, sequence);
        assert!(status.recovery_targets.is_none());
        assert!(status.fault.unwrap().contains("outside controller limits"));
        assert_eq!(status.commands, 0);
        stop.store(estop::PRESSED, Ordering::SeqCst);
        let sequence = worker
            .submit(Primitive::PrepareRecovery { staged: [0.0; 7] })
            .unwrap();
        let status = await_command(&worker, sequence);
        assert!(status.recovery_targets.is_none());
        assert!(status.fault.unwrap().contains("e-stop"));
        assert_eq!(status.commands, 0);
    }

    struct IdleAfterRecoveryClear(MockArm, bool, Arc<std::sync::atomic::AtomicUsize>);
    impl ArmBackend for IdleAfterRecoveryClear {
        fn connect(&mut self, c: &ArmConfig) -> Result<()> {
            self.0.connect(c)
        }
        fn measured(&mut self) -> Result<Measured> {
            self.0.measured()
        }
        fn recovery_limits(&mut self) -> Result<[crate::recovery::PositionLimit; 7]> {
            self.0.recovery_limits()
        }
        fn recovery_needs_configuration(&mut self, measured: &[f64; 7]) -> Result<bool> {
            let mut feedback = self.0.recovery_limits()?;
            if self.1 {
                feedback[1].min -= 0.01;
            }
            crate::recovery::needs_golden(measured, &feedback)
        }
        fn retract_carriage(&mut self, q: f64, t: f64) -> Result<()> {
            self.0.retract_carriage(q, t)
        }
        fn set_mode(&mut self, mode: Mode) -> Result<()> {
            self.0.set_mode(mode)
        }
        fn move_j(&mut self, q: &[f64], t: f64) -> Result<()> {
            self.0.move_j(q, t)
        }
        fn stream(&mut self, s: &Sample) -> Result<()> {
            self.0.stream(s)
        }
        fn hold(&mut self) -> Result<()> {
            self.0.hold()
        }
        fn clear_error(&mut self) -> Result<()> {
            self.2.fetch_add(1, Ordering::SeqCst);
            self.0.clear_error()?;
            self.0.set_mode(Mode::Idle)
        }
        fn load_config(&mut self, p: &std::path::Path) -> Result<()> {
            self.0.load_config(p)
        }
        fn takeover(&mut self, q: &[f64; 7]) -> Result<()> {
            self.0.takeover(q)
        }
        fn inject_faults(&mut self, f: crate::Faults) -> Result<()> {
            self.0.inject_faults(f)
        }
    }

    #[test]
    fn recovery_reconnect_idle_requires_takeover_before_motion() {
        let mut limits = [crate::recovery::PositionLimit {
            min: -1.0,
            max: 1.0,
        }; 7];
        limits[1].min = 0.0;
        limits[6] = crate::recovery::PositionLimit {
            min: 0.0,
            max: 0.04,
        };
        let mut arm =
            MockArm::recovery_seed(limits, [0.0, -0.004, 0.0, 0.0, 0.0, 0.0, 0.002]).unwrap();
        arm.connect(&ArmConfig { joints: 6 }).unwrap();
        arm.set_mode(Mode::Position).unwrap();
        let reloads = Arc::new(std::sync::atomic::AtomicUsize::new(0));
        let mut worker = Worker::spawn(
            Control {
                backend: IdleAfterRecoveryClear(arm, false, reloads.clone()),
                estop: Arc::new(AtomicI32::new(estop::OK)),
                max_velocity: 1.0,
                max_tracking_error: crate::TRACKING_ERROR_LIMIT_RAD,
                max_contact: 1.0,
                envelope: 1.0,
            },
            Duration::from_micros(2500),
            MotionGuard::new(1.0, 9.0, 0.5, 0.5, 8),
        )
        .unwrap();
        let wait = |worker: &Worker, sequence| {
            let deadline = Instant::now() + Duration::from_secs(2);
            loop {
                let s = worker.status();
                if s.sequence == sequence && s.completed {
                    return s;
                }
                assert!(Instant::now() < deadline);
                thread::sleep(Duration::from_millis(2));
            }
        };
        let id = worker
            .submit(Primitive::PrepareRecovery { staged: [0.0; 7] })
            .unwrap();
        assert!(wait(&worker, id).fault.is_none());
        thread::sleep(Duration::from_millis(30));
        assert!(worker.status().fault.is_none());
        assert_eq!(worker.status().measured.unwrap().mode, Mode::Idle);
        assert_eq!(reloads.load(Ordering::SeqCst), 1);
        let id = worker
            .submit(Primitive::MoveJ {
                target: vec![0.0; 6],
                goal_time_s: 1.0,
            })
            .unwrap();
        assert!(
            wait(&worker, id)
                .fault
                .unwrap()
                .contains("re-seed required")
        );
        let id = worker.submit(Primitive::Takeover).unwrap();
        let s = wait(&worker, id);
        assert!(s.fault.is_none(), "{:?}", s.fault);
        assert_eq!(s.measured.unwrap().mode, Mode::Position);
        let id = worker
            .submit(Primitive::Inject(crate::Faults {
                mode_flip: Some(0),
                ..Default::default()
            }))
            .unwrap();
        wait(&worker, id);
        thread::sleep(Duration::from_millis(30));
        assert!(
            worker
                .status()
                .fault
                .unwrap()
                .contains("controller/envelope")
        );
    }

    #[test]
    fn repeated_recovery_preparation_uses_backend_feedback_band_and_nominal_targets() {
        let await_command = |worker: &Worker, sequence| {
            let deadline = Instant::now() + Duration::from_secs(2);
            loop {
                let status = worker.status();
                if status.sequence == sequence && status.completed {
                    return status;
                }
                assert!(Instant::now() < deadline);
                thread::sleep(Duration::from_millis(2));
            }
        };
        let mut limits = [crate::recovery::PositionLimit {
            min: -1.0,
            max: 1.0,
        }; 7];
        limits[1].min = 0.0;
        let mut arm =
            MockArm::recovery_seed(limits, [0.0, -0.003, 0.0, 0.0, 0.0, 0.0, 0.002]).unwrap();
        arm.connect(&ArmConfig { joints: 6 }).unwrap();
        arm.set_mode(Mode::Position).unwrap();
        let reloads = Arc::new(std::sync::atomic::AtomicUsize::new(0));
        let mut worker = Worker::spawn(
            Control {
                backend: IdleAfterRecoveryClear(arm, true, reloads.clone()),
                estop: Arc::new(AtomicI32::new(estop::OK)),
                max_velocity: 1.0,
                max_tracking_error: crate::TRACKING_ERROR_LIMIT_RAD,
                max_contact: 1.0,
                envelope: 1.0,
            },
            Duration::from_micros(2500),
            MotionGuard::new(1.0, 9.0, 0.5, 0.5, 8),
        )
        .unwrap();
        for _ in 0..2 {
            let id = worker
                .submit(Primitive::PrepareRecovery { staged: [0.0; 7] })
                .unwrap();
            let status = await_command(&worker, id);
            assert!(status.fault.is_none(), "{:?}", status.fault);
            assert_eq!(status.measured.unwrap().mode, Mode::Position);
            let targets = status.recovery_targets.unwrap();
            assert_eq!(targets.takeover[1], crate::recovery::LIMIT_MARGIN);
            assert_eq!(targets.staged[1], 0.0);
            assert_eq!(reloads.load(Ordering::SeqCst), 0);
        }
        let id = worker
            .submit(Primitive::MoveJ {
                target: vec![0.0; 6],
                goal_time_s: 1.0,
            })
            .unwrap();
        assert!(
            await_command(&worker, id)
                .fault
                .unwrap()
                .contains("re-seed required")
        );
        let id = worker
            .submit(Primitive::PrepareRecovery { staged: [0.0; 7] })
            .unwrap();
        assert!(await_command(&worker, id).fault.is_none());
        let id = worker.submit(Primitive::Takeover).unwrap();
        let status = await_command(&worker, id);
        assert!(status.fault.is_none(), "{:?}", status.fault);
        assert_eq!(status.measured.unwrap().mode, Mode::Position);
        assert_eq!(reloads.load(Ordering::SeqCst), 0);
        let mut invalid = [0.0; 7];
        invalid[1] = -0.003;
        let id = worker
            .submit(Primitive::PrepareRecovery { staged: invalid })
            .unwrap();
        assert!(
            await_command(&worker, id)
                .fault
                .unwrap()
                .contains("outside controller limits")
        );
    }

    #[test]
    fn takeover_waits_for_interval_and_estop_interrupts_without_resuming() {
        let mut arm = MockArm::default();
        arm.connect(&ArmConfig { joints: 6 }).unwrap();
        arm.set_mode(Mode::Position).unwrap();
        let stop = Arc::new(AtomicI32::new(estop::OK));
        let mut worker = Worker::spawn(
            Control {
                backend: arm,
                estop: stop.clone(),
                max_velocity: 1.0,
                max_tracking_error: crate::TRACKING_ERROR_LIMIT_RAD,
                max_contact: 1.0,
                envelope: 1.0,
            },
            Duration::from_micros(2500),
            MotionGuard::new(1.0, 9.0, 0.5, 0.5, 8),
        )
        .unwrap();
        let await_command = |worker: &Worker, sequence| {
            let deadline = Instant::now() + Duration::from_secs(2);
            loop {
                let status = worker.status();
                if status.sequence == sequence && status.completed {
                    return status;
                }
                assert!(Instant::now() < deadline);
                thread::sleep(Duration::from_millis(2));
            }
        };
        let sequence = worker.submit(Primitive::Takeover).unwrap();
        assert!(
            await_command(&worker, sequence)
                .fault
                .unwrap()
                .contains("prepare recovery")
        );
        assert_eq!(worker.status().commands, 0);
        let sequence = worker
            .submit(Primitive::PrepareRecovery { staged: [0.0; 7] })
            .unwrap();
        assert!(await_command(&worker, sequence).fault.is_none());
        let start = Instant::now();
        let sequence = worker.submit(Primitive::Takeover).unwrap();
        thread::sleep(Duration::from_millis(40));
        assert!(!worker.status().completed);
        let status = await_command(&worker, sequence);
        assert!(start.elapsed() >= Duration::from_secs_f64(crate::recovery::TAKEOVER_S));
        assert!(status.fault.is_none());
        let sequence = worker
            .submit(Primitive::MoveJ {
                target: vec![0.1; 6],
                goal_time_s: 0.4,
            })
            .unwrap();
        assert!(await_command(&worker, sequence).fault.is_none());
        let sequence = worker.submit(Primitive::Takeover).unwrap();
        // A new takeover must refresh the old prepared pose after the move.
        thread::sleep(Duration::from_millis(40));
        assert_eq!(worker.status().recovery_targets.unwrap().takeover[0], 0.1);
        let before = Instant::now();
        stop.store(estop::PRESSED, Ordering::SeqCst);
        let status = await_command(&worker, sequence);
        assert!(before.elapsed() < Duration::from_millis(100));
        assert!(status.fault.unwrap().contains("e-stop"));
        let commands = status.commands;
        stop.store(estop::OK, Ordering::SeqCst);
        let sequence = worker
            .submit(Primitive::MoveJ {
                target: vec![0.2; 6],
                goal_time_s: 0.4,
            })
            .unwrap();
        assert!(
            await_command(&worker, sequence)
                .fault
                .unwrap()
                .contains("re-seed required")
        );
        assert_eq!(worker.status().commands, commands);
    }

    #[test]
    fn takeover_cannot_clamp_an_arbitrary_pose_past_the_control_velocity_cap() {
        let mut arm = MockArm::default();
        arm.connect(&ArmConfig { joints: 6 }).unwrap();
        arm.set_mode(Mode::Position).unwrap();
        arm.inject_faults(crate::Faults {
            out_of_envelope: true,
            ..Default::default()
        })
        .unwrap();
        let mut worker = Worker::spawn(
            Control {
                backend: arm,
                estop: Arc::new(AtomicI32::new(estop::OK)),
                max_velocity: 1.0,
                max_tracking_error: crate::TRACKING_ERROR_LIMIT_RAD,
                max_contact: 1.0,
                envelope: 1.0,
            },
            Duration::from_micros(2500),
            MotionGuard::new(1.0, 9.0, 0.5, 0.5, 8),
        )
        .unwrap();
        for primitive in [
            Primitive::PrepareRecovery { staged: [0.0; 7] },
            Primitive::Takeover,
        ] {
            let sequence = worker.submit(primitive).unwrap();
            let deadline = Instant::now() + Duration::from_secs(1);
            while worker.status().sequence != sequence || !worker.status().completed {
                assert!(Instant::now() < deadline);
                thread::sleep(Duration::from_millis(2));
            }
        }
        assert!(
            worker
                .status()
                .fault
                .unwrap()
                .contains("control envelope/velocity")
        );
        assert_eq!(worker.status().commands, 0);
    }

    #[test]
    fn seven_axis_recovery_moves_preserve_carriage_and_are_stop_monitored() {
        let mut arm = MockArm::default();
        arm.connect(&ArmConfig { joints: 6 }).unwrap();
        arm.set_mode(Mode::Position).unwrap();
        arm.retract_carriage(0.032, 0.6).unwrap();
        let stop = Arc::new(AtomicI32::new(estop::OK));
        let mut worker = Worker::spawn(
            Control {
                backend: arm,
                estop: stop.clone(),
                max_velocity: 1.0,
                max_tracking_error: crate::TRACKING_ERROR_LIMIT_RAD,
                max_contact: 1.0,
                envelope: 1.0,
            },
            Duration::from_micros(2500),
            MotionGuard::new(1.0, 9.0, 0.5, 0.5, 8),
        )
        .unwrap();
        let await_command = |worker: &Worker, sequence| {
            let deadline = Instant::now() + Duration::from_secs(5);
            loop {
                let status = worker.status();
                if status.sequence == sequence && status.completed {
                    return status;
                }
                assert!(Instant::now() < deadline);
                thread::sleep(Duration::from_millis(2));
            }
        };
        let staged = [0.0, 0.2, -0.3, 0.0, 0.0, 0.5, 0.002];
        for primitive in [Primitive::PrepareRecovery { staged }, Primitive::Takeover] {
            let sequence = worker.submit(primitive).unwrap();
            assert!(await_command(&worker, sequence).fault.is_none());
        }
        let commands = worker.status().commands;
        let sequence = worker
            .submit(Primitive::RecoveryMove(crate::recovery::Phase::Staged))
            .unwrap();
        thread::sleep(Duration::from_millis(40));
        let status = worker.status();
        assert!(!status.completed);
        assert_eq!(status.commands, commands + 1);
        assert_eq!(
            status.measured.unwrap().carriage.unwrap().target_m,
            Some(0.032)
        );
        let pressed = Instant::now();
        stop.store(estop::PRESSED, Ordering::SeqCst);
        assert!(
            await_command(&worker, sequence)
                .fault
                .unwrap()
                .contains("e-stop")
        );
        assert!(pressed.elapsed() < Duration::from_millis(100));
        stop.store(estop::OK, Ordering::SeqCst);
        let sequence = worker
            .submit(Primitive::RecoveryMove(crate::recovery::Phase::Sleep))
            .unwrap();
        assert!(
            await_command(&worker, sequence)
                .fault
                .unwrap()
                .contains("re-seed required")
        );
        for primitive in [Primitive::PrepareRecovery { staged }, Primitive::Takeover] {
            let sequence = worker.submit(primitive).unwrap();
            assert!(await_command(&worker, sequence).fault.is_none());
        }
        let commands = worker.status().commands;
        let start = Instant::now();
        let sequence = worker
            .submit(Primitive::RecoveryMove(crate::recovery::Phase::Sleep))
            .unwrap();
        let status = await_command(&worker, sequence);
        assert!(status.fault.is_none());
        assert!(start.elapsed() >= Duration::from_secs_f64(crate::recovery::SLEEP_POSE_S));
        assert_eq!(status.commands, commands + 1);
        let measured = status.measured.unwrap();
        assert_eq!(measured.joints, vec![0.0, 0.0, 0.0, 0.0, 0.0, 0.5]);
        // The retracted, qualified carriage returned to the staged rest at Sleep.
        assert_eq!(measured.carriage.unwrap().position_m, 0.002);
    }

    /// Landing after a trip: a qualified carriage rides retracted to the
    /// staged pose and returns to the profile's rest at Sleep as a timed move
    /// the readback judges, with the contact estimator sitting it out; an
    /// unqualified carriage keeps its measured value through both phases; a
    /// carriage that does not reach its rest latches the landing.
    #[test]
    fn a_qualified_carriage_returns_to_rest_at_sleep_and_an_unreached_rest_latches() {
        let staged = [0.0, 0.2, -0.3, 0.0, 0.0, 0.5, 0.002];
        for (qualified, stuck) in [(true, false), (false, false), (true, true)] {
            // The mock's default carriage is qualified: retract it as a trip
            // would, then state this case's qualification.
            let mut arm = MockArm::default();
            arm.connect(&ArmConfig { joints: 6 }).unwrap();
            arm.set_mode(Mode::Position).unwrap();
            arm.retract_carriage(0.032, 0.6).unwrap();
            arm.carriage_qualified = qualified;
            arm.faults.carriage_stuck = stuck;
            let stop = Arc::new(AtomicI32::new(estop::OK));
            let mut worker = Worker::spawn(
                Control {
                    backend: arm,
                    estop: stop.clone(),
                    max_velocity: 1.0,
                    max_tracking_error: crate::TRACKING_ERROR_LIMIT_RAD,
                    max_contact: 1.0,
                    envelope: 1.0,
                },
                Duration::from_micros(2500),
                MotionGuard::new(1.0, 9.0, 0.5, 0.5, 8),
            )
            .unwrap();
            let await_command = |worker: &Worker, sequence| {
                let deadline = Instant::now() + Duration::from_secs(8);
                loop {
                    let status = worker.status();
                    if status.sequence == sequence && status.completed {
                        return status;
                    }
                    assert!(Instant::now() < deadline);
                    thread::sleep(Duration::from_millis(2));
                }
            };
            for primitive in [
                Primitive::PrepareRecovery { staged },
                Primitive::Takeover,
                Primitive::RecoveryMove(crate::recovery::Phase::Staged),
            ] {
                let sequence = worker.submit(primitive).unwrap();
                let status = await_command(&worker, sequence);
                assert!(status.fault.is_none(), "{:?}", status.fault);
                assert_eq!(status.measured.unwrap().carriage.unwrap().position_m, 0.032);
            }
            let targets = worker.status().recovery_targets.unwrap();
            assert_eq!(targets.sleep[6], if qualified { 0.002 } else { 0.032 });
            let submitted = Instant::now();
            let sequence = worker
                .submit(Primitive::RecoveryMove(crate::recovery::Phase::Sleep))
                .unwrap();
            thread::sleep(Duration::from_millis(200));
            let moving = worker.status();
            assert!(!moving.completed);
            if qualified {
                // The estimator sat out the commanded carriage move: no
                // contact evaluation after the tick that admitted it.
                assert!(
                    moving
                        .contact_at
                        .is_none_or(|at| at <= submitted + Duration::from_millis(20)),
                    "{:?}",
                    moving.contact_at.map(|at| at.duration_since(submitted))
                );
            }
            let status = await_command(&worker, sequence);
            let carriage = status.measured.unwrap().carriage.unwrap().position_m;
            match (qualified, stuck) {
                (true, false) => {
                    assert!(status.fault.is_none(), "{:?}", status.fault);
                    assert!(
                        submitted.elapsed()
                            >= Duration::from_secs_f64(crate::recovery::SLEEP_POSE_S)
                    );
                    assert_eq!(carriage, 0.002);
                }
                (false, _) => {
                    assert!(status.fault.is_none(), "{:?}", status.fault);
                    assert_eq!(carriage, 0.032);
                }
                (true, true) => {
                    assert!(
                        status
                            .fault
                            .as_deref()
                            .unwrap()
                            .contains("carriage did not reach 0.002 m"),
                        "{:?}",
                        status.fault
                    );
                    assert_eq!(carriage, 0.032);
                }
            }
        }
    }

    /// A reconnect carries what the backend waited for into the status the
    /// session reads: the mock's measurement is live at once, so its wait
    /// settles with no time in it. A backend with no wait reports none.
    #[test]
    fn a_reconnect_carries_the_backends_connect_wait_into_the_status() {
        let mut arm = MockArm::default();
        arm.connect(&ArmConfig { joints: 6 }).unwrap();
        arm.set_mode(Mode::Position).unwrap();
        let mut worker = Worker::spawn(
            Control {
                backend: arm,
                estop: Arc::new(AtomicI32::new(estop::OK)),
                max_velocity: 1.0,
                max_tracking_error: crate::TRACKING_ERROR_LIMIT_RAD,
                max_contact: 1.0,
                envelope: 1.0,
            },
            Duration::from_micros(2500),
            MotionGuard::new(1.0, 9.0, 0.5, 0.5, 8),
        )
        .unwrap();
        assert!(worker.status().connect_wait.is_none());
        let sequence = worker
            .submit(Primitive::Reconnect(ArmConfig { joints: 6 }))
            .unwrap();
        let deadline = Instant::now() + Duration::from_secs(3);
        let status = loop {
            let status = worker.status();
            if status.sequence == sequence && status.completed {
                break status;
            }
            assert!(Instant::now() < deadline, "reconnect never completed");
            thread::sleep(Duration::from_millis(2));
        };
        assert!(status.fault.is_none(), "{:?}", status.fault);
        let wait = status.connect_wait.expect("the mock reports its wait");
        assert!(wait.settled);
        assert_eq!(wait.elapsed_s, 0.0);
        assert_eq!(wait.outcome(), "settle_reached");
        // The mock connected once at spawn and once here.
        assert_eq!(wait.connect, 2);
        assert_eq!(
            crate::ConnectWait {
                settled: false,
                elapsed_s: 1.0,
                settle_s: 0.1,
                patience_s: 1.0,
                connect: 1,
            }
            .outcome(),
            "patience_exhausted"
        );
    }
    #[test]
    fn constructor_failure_publishes_no_healthy_measurement() {
        let worker = Worker::spawn_with::<MockArm, _>(
            || Err(Error("connection refused".into())),
            Duration::from_micros(2500),
            MotionGuard::new(1.0, 9.0, 0.5, 0.5, 8),
        )
        .unwrap();
        let deadline = Instant::now() + Duration::from_secs(2);
        while worker.status().fault.is_none() {
            assert!(Instant::now() < deadline);
            std::thread::sleep(Duration::from_millis(1));
        }
        let status = worker.status();
        assert!(status.fault.unwrap().contains("connection refused"));
        assert!(!status.watchdog_ok);
        assert!(status.measured.is_none());
        assert_eq!(status.commands, 0);
        assert!(status.completed);
    }
    #[test]
    fn startup_move_finishes_before_collecting_the_rest_baseline() {
        let mut arm = MockArm::default();
        arm.connect(&ArmConfig { joints: 6 }).unwrap();
        arm.set_mode(Mode::Position).unwrap();
        let mut worker = Worker::spawn(
            Control {
                backend: arm,
                estop: Arc::new(AtomicI32::new(estop::OK)),
                max_velocity: 1.0,
                max_tracking_error: crate::TRACKING_ERROR_LIMIT_RAD,
                max_contact: 1.0,
                envelope: 1.0,
            },
            Duration::from_micros(2500),
            MotionGuard::new(1.0, 9.0, 0.5, 0.5, 8),
        )
        .unwrap();
        let await_command = |worker: &Worker, sequence| {
            let deadline = Instant::now() + Duration::from_secs(6);
            loop {
                let status = worker.status();
                assert!(status.fault.is_none(), "{:?}", status.fault);
                assert!(!status.contact.is_some_and(|c| c.armed));
                if status.sequence == sequence && status.completed {
                    break;
                }
                assert!(Instant::now() < deadline);
                thread::sleep(Duration::from_millis(2));
            }
        };
        for primitive in [
            // The staged pose's carriage is the rest a qualified carriage
            // returns to at Sleep, so it must lie inside the mock's limits.
            Primitive::PrepareRecovery {
                staged: [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.002],
            },
            Primitive::Takeover,
            Primitive::RecoveryMove(crate::recovery::Phase::Staged),
        ] {
            let sequence = worker.submit(primitive).unwrap();
            await_command(&worker, sequence);
        }
        let settled = Instant::now();
        loop {
            let status = worker.status();
            assert!(status.fault.is_none());
            if status.contact_is_valid(Duration::from_millis(250)) {
                break;
            }
            assert!(settled.elapsed() < Duration::from_secs(4));
            thread::sleep(Duration::from_millis(2));
        }
        assert!(settled.elapsed() >= Duration::from_millis(1800));
    }

    #[test]
    fn contact_evidence_requires_a_fresh_armed_assessment() {
        let mut arm = MockArm::default();
        arm.connect(&ArmConfig { joints: 6 }).unwrap();
        let limit = Duration::from_millis(250);
        let mut status = Status {
            measured: Some(arm.measured().unwrap()),
            measured_at: Some(Instant::now()),
            ..Status::default()
        };
        assert!(!status.contact_is_valid(limit));
        let mut cap = crate::contact::ContactCap::default();
        let observation = crate::contact::ContactObservation {
            effort_n: 0.0,
            position_m: 0.002,
            target_m: 0.002,
            arm_speed_rad_s: 0.0,
            aligning: false,
            effort_assessable: true,
        };
        for _ in 0..799 {
            status.contact = Some(cap.observe(observation).unwrap());
        }
        status.contact_at = Some(Instant::now());
        assert!(!status.contact_is_valid(limit));
        status.contact = Some(cap.observe(observation).unwrap());
        assert!(status.contact_is_valid(limit));
        status.contact_at = Some(Instant::now() - Duration::from_secs(1));
        assert!(!status.contact_is_valid(limit));
        status.contact_at = Some(Instant::now() + Duration::from_secs(1));
        assert!(!status.contact_is_valid(limit));
        status.contact_at = Some(Instant::now());
        status.contact = Some(
            cap.observe(crate::contact::ContactObservation {
                arm_speed_rad_s: 0.3,
                ..observation
            })
            .unwrap(),
        );
        assert!(!status.contact_is_valid(limit));
        status.contact = Some(cap.observe(observation).unwrap());
        status.fault = Some("controller fault".into());
        assert!(!status.contact_is_valid(limit));
        status.fault = None;
        status.measured_at = None;
        assert!(!status.contact_is_valid(limit));
    }

    #[test]
    fn recovery_fault_class_applies_only_to_the_exact_worker_fault() {
        let mut status = Status {
            fault: Some("measured trip".into()),
            recovery_fault: Some(("measured trip".into(), RecoveryFaultCategory::Trip)),
            ..Status::default()
        };
        assert_eq!(status.recovery_category(), Some("trip"));
        status.retract_verified = Some(false);
        assert_eq!(status.recovery_category(), None);
        status.retract_verified = Some(true);
        status.fault = Some("retract failed".into());
        assert_eq!(status.recovery_category(), None);
        status.fault = None;
        assert_eq!(status.recovery_category(), None);
    }
    #[test]
    fn measurement_freshness_uses_monotonic_receipt_not_wall_clock() {
        let mut arm = MockArm::default();
        arm.connect(&ArmConfig { joints: 6 }).unwrap();
        let mut status = Status::default();
        let limit = Duration::from_millis(250);
        assert!(!status.measurement_is_fresh(limit));
        status.measured = Some(arm.measured().unwrap());
        status.measured_at = Some(Instant::now());
        assert!(status.measurement_is_fresh(limit));
        status.measured_at = Some(Instant::now() - Duration::from_secs(1));
        status.measured_wall_ns = u64::MAX;
        assert!(!status.measurement_is_fresh(limit));
        status.measured_at = Some(Instant::now() + Duration::from_secs(1));
        assert!(!status.measurement_is_fresh(limit));
    }
    #[test]
    fn injection_ack_survives_the_fault_and_a_delayed_status_reader() {
        let mut arm = MockArm::default();
        arm.connect(&ArmConfig { joints: 6 }).unwrap();
        arm.set_mode(Mode::Position).unwrap();
        let mut worker = Worker::spawn(
            Control {
                backend: arm,
                estop: Arc::new(AtomicI32::new(estop::OK)),
                max_velocity: 1.0,
                max_tracking_error: crate::TRACKING_ERROR_LIMIT_RAD,
                max_contact: 1.0,
                envelope: 1.0,
            },
            Duration::from_micros(2500),
            MotionGuard::new(1.0, 9.0, 0.5, 0.5, 8),
        )
        .unwrap();
        let shared_status = worker.status.clone();
        let locked = shared_status.lock().unwrap();
        let sequence = worker
            .submit(Primitive::Inject(crate::Faults {
                error_after: Some(0),
                ..Default::default()
            }))
            .unwrap();
        // The observer misses the successful-injection tick. Subsequent
        // controller-error ticks must not overwrite that acknowledgement.
        thread::sleep(Duration::from_millis(40));
        drop(locked);
        let deadline = Instant::now() + Duration::from_secs(1);
        loop {
            let status = worker.status();
            if status.sequence == sequence
                && let Some(fault) = &status.fault
            {
                assert_eq!(status.injection_result, Some(Ok(())));
                assert!(fault.contains("controller/envelope"));
                assert_eq!(status.commands, 0);
                break;
            }
            assert!(Instant::now() < deadline);
            thread::sleep(Duration::from_millis(2));
        }
        let sequence = worker
            .submit(Primitive::MoveJ {
                target: vec![0.1; 6],
                goal_time_s: 1.0,
            })
            .unwrap();
        while worker.status().sequence != sequence {
            assert!(Instant::now() < deadline);
            thread::sleep(Duration::from_millis(2));
        }
        let status = worker.status();
        assert!(status.injection_result.is_none());
        assert!(status.fault.unwrap().contains("re-seed required"));
        assert_eq!(status.commands, 0);
    }
    #[test]
    fn stalled_command_latches_watchdog_and_requires_reseed() {
        let mut arm = MockArm::default();
        arm.connect(&ArmConfig { joints: 6 }).unwrap();
        arm.set_mode(Mode::Position).unwrap();
        arm.faults.freeze = true;
        let mut worker = Worker::spawn(
            Control {
                backend: arm,
                estop: Arc::new(AtomicI32::new(estop::OK)),
                // Isolate the no-progress detector from the stricter production
                // tracking bound; this backend is exclusively a frozen mock.
                max_velocity: 1000.0,
                max_tracking_error: 1000.0,
                max_contact: 1.0,
                envelope: 1.0,
            },
            Duration::from_micros(2500),
            MotionGuard::new(1.0, 9.0, 0.5, 0.5, 8),
        )
        .unwrap();
        let stream_sequence = worker
            .submit(Primitive::Stream(vec![
                Sample {
                    joints: vec![0.5; 6],
                    dt_s: 0.0025,
                    contact: 0.0
                };
                1200
            ]))
            .unwrap();
        let deadline = Instant::now() + Duration::from_secs(4);
        // Startup is deliberately unready (watchdog_ok=false), not a trip.
        // Wait for completion of this particular stream before assessing it.
        while Instant::now() < deadline {
            let status = worker.status();
            if status.sequence == stream_sequence && status.completed {
                break;
            }
            thread::sleep(Duration::from_millis(5));
        }
        let status = worker.status();
        assert_eq!(status.sequence, stream_sequence);
        assert!(!status.watchdog_ok, "{:?}", status.fault);
        assert!(status.completed);
        assert!(status.fault.unwrap().contains("tracking watchdog"));
        let sequence = worker.submit(Primitive::Reseed).unwrap();
        let deadline = Instant::now() + Duration::from_secs(1);
        while worker.status().sequence != sequence && Instant::now() < deadline {
            thread::sleep(Duration::from_millis(5));
        }
        assert_eq!(worker.status().sequence, sequence);
        assert!(worker.status().watchdog_ok);
        assert!(worker.status().fault.is_none());
    }
    #[test]
    fn controller_faults_latch_even_without_a_pending_stream() {
        for faults in [
            crate::Faults {
                error_after: Some(0),
                ..Default::default()
            },
            crate::Faults {
                mode_flip: Some(0),
                ..Default::default()
            },
            crate::Faults {
                out_of_envelope: true,
                ..Default::default()
            },
        ] {
            let mut arm = MockArm::default();
            arm.connect(&ArmConfig { joints: 6 }).unwrap();
            arm.set_mode(Mode::Position).unwrap();
            arm.faults = faults;
            let worker = Worker::spawn(
                Control {
                    backend: arm,
                    estop: Arc::new(AtomicI32::new(estop::OK)),
                    max_velocity: 1.0,
                    max_tracking_error: crate::TRACKING_ERROR_LIMIT_RAD,
                    max_contact: 1.0,
                    envelope: 1.0,
                },
                Duration::from_micros(2500),
                MotionGuard::new(1.0, 9.0, 0.5, 0.5, 8),
            )
            .unwrap();
            let deadline = Instant::now() + Duration::from_secs(1);
            loop {
                let status = worker.status();
                if let Some(reason) = status.fault {
                    assert!(reason.contains("controller/envelope"));
                    assert_eq!(status.commands, 0);
                    break;
                }
                assert!(Instant::now() < deadline);
                thread::sleep(Duration::from_millis(5));
            }
        }
    }
    #[test]
    fn stalled_observer_does_not_stall_estop_and_release_does_not_resume() {
        let mut arm = MockArm::default();
        arm.connect(&ArmConfig { joints: 6 }).unwrap();
        arm.set_mode(Mode::Position).unwrap();
        let atomic = Arc::new(AtomicI32::new(estop::OK));
        let control = Control {
            backend: arm,
            estop: atomic.clone(),
            max_velocity: 1.0,
            max_tracking_error: crate::TRACKING_ERROR_LIMIT_RAD,
            max_contact: 1.0,
            envelope: 1.0,
        };
        let mut worker = Worker::spawn(
            control,
            Duration::from_micros(2500),
            MotionGuard::new(1.0, 9.0, 0.5, 0.5, 8),
        )
        .unwrap();
        worker
            .submit(Primitive::MoveJ {
                target: vec![0.1; 6],
                goal_time_s: 1.0,
            })
            .unwrap();
        thread::sleep(Duration::from_millis(30));
        let status = worker.status.clone();
        let locked = status.lock().unwrap();
        atomic.store(estop::PRESSED, Ordering::SeqCst);
        thread::sleep(Duration::from_millis(30));
        drop(locked);
        thread::sleep(Duration::from_millis(10));
        let stopped = worker.status();
        assert!(stopped.completed);
        assert_eq!(stopped.fault.as_deref(), Some("e-stop"));
        let joints = stopped.measured.unwrap().joints;
        atomic.store(estop::OK, Ordering::SeqCst);
        worker
            .submit(Primitive::MoveJ {
                target: vec![0.2; 6],
                goal_time_s: 1.0,
            })
            .unwrap();
        thread::sleep(Duration::from_millis(20));
        let after = worker.status();
        assert!(after.fault.unwrap().contains("Recover"));
        assert_eq!(after.measured.unwrap().joints, joints);
    }
    #[test]
    fn contact_trip_retract_is_measured_and_stuck_carriage_is_failure() {
        for stuck in [false, true] {
            let mut arm = MockArm::default();
            arm.connect(&ArmConfig { joints: 6 }).unwrap();
            arm.set_mode(Mode::Position).unwrap();
            arm.faults.carriage_deflection_m = Some(0.003);
            arm.faults.carriage_stuck = stuck;
            let control = Control {
                backend: arm,
                estop: Arc::new(AtomicI32::new(estop::OK)),
                max_velocity: 1.0,
                max_tracking_error: crate::TRACKING_ERROR_LIMIT_RAD,
                max_contact: 1.0,
                envelope: 1.0,
            };
            let worker = Worker::spawn(
                control,
                Duration::from_micros(2500),
                MotionGuard::new(1.0, 9.0, 0.5, 0.5, 8),
            )
            .unwrap();
            let deadline = Instant::now() + Duration::from_secs(2);
            loop {
                let status = worker.status();
                if let Some(verified) = status.retract_verified {
                    assert_eq!(verified, !stuck);
                    assert_eq!(status.recovery_category(), (!stuck).then_some("trip"));
                    let fault = status.fault.unwrap();
                    assert!(fault.contains("carriage_contact_cap"));
                    assert!(fault.contains("contact_n=") && fault.contains("deflection_m="));
                    assert!(fault.contains("cap_n=20") && fault.contains("limit_m=0.002"));
                    break;
                }
                assert!(Instant::now() < deadline, "trip did not finish");
                thread::sleep(Duration::from_millis(5));
            }
        }
    }

    fn target_abort_stream(
        policy: crate::ContactPolicy,
        qualified: bool,
        carriage_stuck: bool,
        arm_contact_baseline: bool,
    ) -> (Worker, Arc<std::sync::atomic::AtomicI32>, u64) {
        let mut arm = MockArm {
            contact_policy: policy,
            carriage_qualified: qualified,
            ..Default::default()
        };
        arm.connect(&ArmConfig { joints: 6 }).unwrap();
        arm.set_mode(Mode::Position).unwrap();
        arm.faults.carriage_stuck = carriage_stuck;
        let estop = Arc::new(std::sync::atomic::AtomicI32::new(estop::OK));
        let mut worker = Worker::spawn(
            Control {
                backend: arm,
                estop: estop.clone(),
                max_velocity: 1.0,
                max_tracking_error: crate::TRACKING_ERROR_LIMIT_RAD,
                max_contact: 1.0,
                envelope: 1.0,
            },
            CONTROL_PERIOD,
            MotionGuard::new(1.0, 9.0, 0.5, 0.5, 8),
        )
        .unwrap();
        if arm_contact_baseline {
            let deadline = Instant::now() + Duration::from_secs(4);
            while !worker.status().contact_is_valid(Duration::from_millis(250)) {
                assert!(Instant::now() < deadline, "contact baseline did not arm");
                thread::sleep(Duration::from_millis(2));
            }
        }
        let seed = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.002];
        let sequence = worker
            .submit(Primitive::StreamJoint {
                seed,
                expected_sequence: 0,
                permit: None,
                samples: vec![
                    crate::JointSample {
                        positions: seed,
                        velocities: [0.0; 7],
                        dt_s: 0.0025,
                        pen: true,
                    };
                    400
                ],
            })
            .unwrap();
        let deadline = Instant::now() + Duration::from_secs(1);
        while worker
            .status()
            .stream_progress
            .as_ref()
            .map(|p| p.submitted)
            < Some(10)
        {
            assert!(
                Instant::now() < deadline,
                "stream never submitted ten samples"
            );
            thread::sleep(Duration::from_millis(1));
        }
        (worker, estop, sequence)
    }

    struct SlowLimitsArm {
        arm: MockArm,
        delay: Duration,
        retracts: Arc<std::sync::atomic::AtomicUsize>,
    }
    impl ArmBackend for SlowLimitsArm {
        fn connect(&mut self, config: &ArmConfig) -> Result<()> {
            self.arm.connect(config)
        }
        fn measured(&mut self) -> Result<Measured> {
            self.arm.measured()
        }
        fn recovery_limits(&mut self) -> Result<[crate::recovery::PositionLimit; 7]> {
            thread::sleep(self.delay);
            self.arm.recovery_limits()
        }
        fn retract_carriage(&mut self, target_m: f64, goal_time_s: f64) -> Result<()> {
            self.retracts.fetch_add(1, Ordering::AcqRel);
            self.arm.retract_carriage(target_m, goal_time_s)
        }
        fn set_mode(&mut self, mode: Mode) -> Result<()> {
            self.arm.set_mode(mode)
        }
        fn move_j(&mut self, target: &[f64], goal_time_s: f64) -> Result<()> {
            self.arm.move_j(target, goal_time_s)
        }
        fn stream(&mut self, sample: &Sample) -> Result<()> {
            self.arm.stream(sample)
        }
        fn stream_joint(&mut self, sample: &crate::JointSample) -> Result<()> {
            self.arm.stream_joint(sample)
        }
        fn hold(&mut self) -> Result<()> {
            self.arm.hold()
        }
        fn clear_error(&mut self) -> Result<()> {
            self.arm.clear_error()
        }
        fn load_config(&mut self, path: &std::path::Path) -> Result<()> {
            self.arm.load_config(path)
        }
    }

    #[test]
    fn delayed_live_limits_cannot_retract_from_aged_contact_evidence() {
        let mut arm = MockArm::default();
        arm.connect(&ArmConfig { joints: 6 }).unwrap();
        arm.set_mode(Mode::Position).unwrap();
        let retracts = Arc::new(std::sync::atomic::AtomicUsize::new(0));
        let mut worker = Worker::spawn(
            Control {
                backend: SlowLimitsArm {
                    arm,
                    delay: Duration::from_millis(40),
                    retracts: retracts.clone(),
                },
                estop: Arc::new(AtomicI32::new(estop::OK)),
                max_velocity: 1.0,
                max_tracking_error: crate::TRACKING_ERROR_LIMIT_RAD,
                max_contact: 1.0,
                envelope: 1.0,
            },
            CONTROL_PERIOD,
            MotionGuard::new(1.0, 9.0, 0.5, 0.5, 8),
        )
        .unwrap();
        let deadline = Instant::now() + Duration::from_secs(4);
        while !worker.status().contact_is_valid(Duration::from_millis(250)) {
            assert!(Instant::now() < deadline);
            thread::sleep(Duration::from_millis(2));
        }
        let seed = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.002];
        worker
            .submit(Primitive::StreamJoint {
                seed,
                expected_sequence: 0,
                permit: None,
                samples: vec![
                    crate::JointSample {
                        positions: seed,
                        velocities: [0.0; 7],
                        dt_s: 0.0025,
                        pen: true,
                    };
                    400
                ],
            })
            .unwrap();
        while worker
            .status()
            .stream_progress
            .as_ref()
            .map(|p| p.submitted)
            < Some(10)
        {
            assert!(Instant::now() < deadline);
            thread::sleep(Duration::from_millis(1));
        }
        assert!(worker.abort_target_invalid());
        let receipt = loop {
            if let Some(receipt) = worker.status().target_abort
                && receipt.completed_wall_ns.is_some()
            {
                break receipt;
            }
            assert!(Instant::now() < deadline);
            thread::sleep(Duration::from_millis(1));
        };
        assert_eq!(receipt.outcome, TargetAbortOutcome::Held);
        assert!(!receipt.retract_commanded);
        assert!(receipt.reason.contains("aged during target abort"));
        assert!(receipt.contact_age_ms.unwrap() > 20.0);
        assert_eq!(retracts.load(Ordering::Acquire), 0);
    }

    #[test]
    fn target_abort_retracts_only_qualified_fresh_contact_and_keeps_stream_discarded() {
        for (policy, qualified, stuck, outcome) in [
            (
                crate::ContactPolicy::Contact,
                true,
                false,
                TargetAbortOutcome::Retracted,
            ),
            (
                crate::ContactPolicy::Standoff,
                true,
                false,
                TargetAbortOutcome::Held,
            ),
            (
                crate::ContactPolicy::Contact,
                false,
                false,
                TargetAbortOutcome::Held,
            ),
            (
                crate::ContactPolicy::Contact,
                true,
                true,
                TargetAbortOutcome::Uncertain,
            ),
        ] {
            let (mut worker, _estop, sequence) =
                target_abort_stream(policy, qualified, stuck, true);
            assert!(worker.abort_target_invalid());
            assert!(!worker.abort_target_invalid());
            let deadline = Instant::now() + Duration::from_secs(2);
            let status = loop {
                let status = worker.status();
                if status
                    .target_abort
                    .as_ref()
                    .is_some_and(|receipt| receipt.completed_wall_ns.is_some())
                {
                    break status;
                }
                assert!(Instant::now() < deadline, "target abort did not finish");
                thread::sleep(Duration::from_millis(1));
            };
            let receipt = status.target_abort.as_ref().unwrap();
            assert_eq!(receipt.schema, "tatbot.receipt/1");
            assert_eq!(receipt.kind, "target-abort");
            assert_eq!(receipt.sequence, sequence);
            assert_eq!(receipt.outcome, outcome);
            assert!(receipt.stream_progress.as_ref().unwrap().submitted < 400);
            assert!(!receipt.motion_authority);
            assert_eq!(
                receipt.retract_commanded,
                qualified && policy == crate::ContactPolicy::Contact
            );
            assert!(receipt.contact_armed && receipt.contact_age_ms.unwrap() <= 20.0);
            assert!(status.completed);
            assert_eq!(
                status.retract_verified,
                match outcome {
                    TargetAbortOutcome::Retracted => Some(true),
                    TargetAbortOutcome::Uncertain => Some(false),
                    TargetAbortOutcome::Held => None,
                }
            );
            let retained = serde_json::to_value(receipt).unwrap();
            assert_eq!(retained["outcome"], serde_json::to_value(outcome).unwrap());
            assert_eq!(retained["motion_authority"], false);
            let count = status.commands;
            thread::sleep(Duration::from_millis(20));
            assert_eq!(worker.status().commands, count, "aborted queue resumed");
            worker
                .submit(Primitive::MoveJ {
                    target: vec![0.1; 6],
                    goal_time_s: 0.4,
                })
                .unwrap();
            let deadline = Instant::now() + Duration::from_secs(1);
            while worker.status().sequence == sequence {
                assert!(Instant::now() < deadline);
                thread::sleep(Duration::from_millis(1));
            }
            assert_eq!(
                worker.status().commands,
                count,
                "target abort admitted new motion"
            );
        }
    }

    #[test]
    fn higher_priority_stop_supersedes_target_abort_retract() {
        for physical in [true, false] {
            let (worker, estop, sequence) =
                target_abort_stream(crate::ContactPolicy::Contact, true, false, true);
            if physical {
                estop.store(estop::PRESSED, Ordering::SeqCst);
            } else {
                worker.software_stop.store(true, Ordering::Release);
            }
            assert!(worker.abort_target_invalid());
            let deadline = Instant::now() + Duration::from_secs(1);
            let status = loop {
                let status = worker.status();
                if status.target_abort.is_some() {
                    break status;
                }
                assert!(Instant::now() < deadline);
                thread::sleep(Duration::from_millis(1));
            };
            let receipt = status.target_abort.unwrap();
            assert_eq!(receipt.sequence, sequence);
            assert_eq!(receipt.outcome, TargetAbortOutcome::Uncertain);
            assert!(!receipt.retract_commanded);
            assert_eq!(status.retract_verified, None);
            assert_eq!(status.fault.as_deref(), Some("e-stop"));
        }
    }

    #[test]
    fn target_abort_without_armed_contact_holds_without_retract() {
        let (worker, _, _) = target_abort_stream(crate::ContactPolicy::Contact, true, false, false);
        assert!(worker.abort_target_invalid());
        let deadline = Instant::now() + Duration::from_secs(1);
        loop {
            let status = worker.status();
            if let Some(receipt) = status.target_abort {
                assert_eq!(receipt.outcome, TargetAbortOutcome::Held);
                assert!(!receipt.retract_commanded);
                assert!(!receipt.contact_armed);
                assert!(receipt.reason.contains("contact feedback"));
                break;
            }
            assert!(Instant::now() < deadline);
            thread::sleep(Duration::from_millis(1));
        }
    }

    #[test]
    fn physical_stop_interrupts_in_progress_target_retract() {
        let (worker, estop, _) =
            target_abort_stream(crate::ContactPolicy::Contact, true, true, true);
        assert!(worker.abort_target_invalid());
        let deadline = Instant::now() + Duration::from_secs(1);
        while !worker
            .status()
            .target_abort
            .as_ref()
            .is_some_and(|receipt| receipt.retract_commanded)
        {
            assert!(Instant::now() < deadline);
            thread::sleep(Duration::from_millis(1));
        }
        estop.store(estop::PRESSED, Ordering::SeqCst);
        let deadline = Instant::now() + Duration::from_secs(1);
        loop {
            let status = worker.status();
            if let Some(receipt) = status.target_abort
                && receipt.completed_wall_ns.is_some()
            {
                assert_eq!(receipt.outcome, TargetAbortOutcome::Uncertain);
                assert!(receipt.reason.contains("interrupted"));
                assert_eq!(status.retract_verified, None);
                assert_eq!(status.fault.as_deref(), Some("e-stop"));
                break;
            }
            assert!(Instant::now() < deadline);
            thread::sleep(Duration::from_millis(1));
        }
    }
}
