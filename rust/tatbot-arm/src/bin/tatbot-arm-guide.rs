//! Hand-guiding owner for one selected physical arm.
//!
//! Started by the calibration conductor on the arm node (or locally with the
//! mock backend for the offline loop). Reads JSON-line commands on stdin,
//! writes JSON-line events on stdout, and retains every control tick in
//! `<run-dir>/telemetry.bin`. It owns exactly one driver behind the fleet
//! hardware lease and the physical E-stop monitor; the other arm is never
//! opened. Termination stops acquisition, then idles and releases the supported
//! arm with measured confirmation. No orphaned holding owner is retained.
use serde::Deserialize;
use sha2::{Digest, Sha256};
use std::{
    collections::VecDeque,
    io::{BufRead, BufWriter, Write},
    os::fd::FromRawFd,
    path::{Path, PathBuf},
    sync::{
        Arc, Mutex,
        atomic::{AtomicBool, AtomicI32, Ordering},
        mpsc,
    },
    thread,
    time::{Duration, Instant},
};
use tatbot_arm::{
    ArmConfig, ContactPolicy, Error, Result, estop,
    guide::{
        Command, HoldWindow, Session, State, Tick, dynamics_header, encode_dynamics_sample,
        encode_sample, summarize_window, telemetry_header,
    },
    lease::{DriverLease, HardwareLease},
    profile::NativeProfile,
    worker::{Primitive, Status, TelemetrySample, Worker},
};

const TICK_EVENT_PERIOD: Duration = Duration::from_millis(200);
const RING_SECONDS: u64 = 120;
const RELEASE_REQUEST: &str = "release.request";
/// A qualified carriage's trip-retract endpoint doubles as its parked
/// position before a terminal release.
const PARK_CARRIAGE_M: f64 = 0.032;

fn build_info() -> serde_json::Value {
    serde_json::from_str(include_str!(concat!(env!("OUT_DIR"), "/native-build.json")))
        .expect("compiled native build record")
}

struct Args {
    run_dir: PathBuf,
    controller_role: String,
    backend: String,
    profile_dir: PathBuf,
    /// The fitted tool's datasheet: its `contact:` key is the policy this
    /// owner's worker runs under, and its digest is in the owner record.
    tool_datasheet: PathBuf,
    estop_device: Option<PathBuf>,
    lease_path: Option<PathBuf>,
    mock_seed: [f64; 7],
    mock_estop_file: Option<PathBuf>,
    run_id: Option<String>,
}

fn usage() -> ! {
    eprintln!(
        "usage: tatbot-arm-guide --run-dir DIR --controller-role leader|follower --backend trossen|mock \
         --profile-dir config/trossen --tool-datasheet config/tools/<tool>.yaml [--run-id ID] [--estop-device PATH] \
         [--lease PATH] [--mock-seed JSON7] [--mock-estop-file PATH]"
    );
    std::process::exit(2);
}

fn parse_args() -> Args {
    let mut args = std::env::args().skip(1);
    let mut run_dir = None;
    let mut controller_role = None;
    let mut backend = None;
    let mut profile_dir = None;
    let mut tool_datasheet = None;
    let mut estop_device = None;
    let mut lease_path = None;
    let mut mock_seed = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];
    let mut mock_estop_file = None;
    let mut run_id = None;
    while let Some(flag) = args.next() {
        let value = args.next().unwrap_or_else(|| usage());
        match flag.as_str() {
            "--run-dir" => run_dir = Some(PathBuf::from(value)),
            "--controller-role" => controller_role = Some(value),
            "--backend" => backend = Some(value),
            "--profile-dir" => profile_dir = Some(PathBuf::from(value)),
            "--tool-datasheet" => tool_datasheet = Some(PathBuf::from(value)),
            "--estop-device" => estop_device = Some(PathBuf::from(value)),
            "--lease" => lease_path = Some(PathBuf::from(value)),
            "--mock-seed" => {
                let seed: Vec<f64> = serde_json::from_str(&value).unwrap_or_else(|_| usage());
                mock_seed = seed.try_into().unwrap_or_else(|_| usage());
            }
            "--mock-estop-file" => mock_estop_file = Some(PathBuf::from(value)),
            "--run-id" => run_id = Some(value),
            _ => usage(),
        }
    }
    let (
        Some(run_dir),
        Some(controller_role),
        Some(backend),
        Some(profile_dir),
        Some(tool_datasheet),
    ) = (
        run_dir,
        controller_role,
        backend,
        profile_dir,
        tool_datasheet,
    )
    else {
        usage()
    };
    if backend != "mock" && backend != "trossen" {
        usage();
    }
    Args {
        run_dir,
        controller_role,
        backend,
        profile_dir,
        tool_datasheet,
        estop_device,
        lease_path,
        mock_seed,
        mock_estop_file,
        run_id,
    }
}

fn wall_ns() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_nanos() as u64)
        .unwrap_or(0)
}

/// Event log: stdout for the conductor, events.jsonl for the run.
struct Events {
    protocol: std::fs::File,
    log: BufWriter<std::fs::File>,
}
impl Events {
    fn open(run_dir: &Path) -> std::io::Result<Self> {
        // Reserve the protocol pipe before constructing the SDK. Its C++
        // logger writes to fd 1; those diagnostics belong on stderr.
        let fd = unsafe { libc::fcntl(libc::STDOUT_FILENO, libc::F_DUPFD_CLOEXEC, 3) };
        if fd < 0 {
            return Err(std::io::Error::last_os_error());
        }
        let protocol = unsafe { std::fs::File::from_raw_fd(fd) };
        if unsafe { libc::dup2(libc::STDERR_FILENO, libc::STDOUT_FILENO) } < 0 {
            return Err(std::io::Error::last_os_error());
        }
        Ok(Self {
            protocol,
            log: BufWriter::new(
                std::fs::OpenOptions::new()
                    .create(true)
                    .append(true)
                    .open(run_dir.join("events.jsonl"))?,
            ),
        })
    }
    fn emit(&mut self, event: &str, mut body: serde_json::Value) {
        let object = body.as_object_mut().expect("event body is an object");
        object.insert("event".into(), event.into());
        object.insert("wall_ns".into(), wall_ns().into());
        let line = serde_json::to_string(&body).unwrap();
        // A conductor whose pipe is gone must not take the owner down with it;
        // stdin's EOF is what turns that into a stop.
        let _ = writeln!(self.protocol, "{line}");
        let _ = self.protocol.flush();
        let _ = writeln!(self.log, "{line}");
        let _ = self.log.flush();
    }
    fn record_command(&mut self, line: &str) {
        let _ = writeln!(
            self.log,
            "{{\"stdin\":{}}}",
            serde_json::to_string(line).unwrap()
        );
        let _ = self.log.flush();
    }
}

/// Every tick written to telemetry.bin, and the recent ticks kept for hold windows.
struct Recorder {
    ring: Arc<Mutex<VecDeque<TelemetrySample>>>,
    written: Arc<Mutex<u64>>,
    failed: Arc<AtomicBool>,
    error: Arc<Mutex<Option<String>>>,
    _thread: thread::JoinHandle<()>,
}
impl Recorder {
    fn written(&self) -> u64 {
        *self.written.lock().unwrap()
    }
}
impl Recorder {
    fn start(run_dir: &Path, receiver: mpsc::Receiver<TelemetrySample>) -> std::io::Result<Self> {
        let file = BufWriter::new(std::fs::File::create_new(run_dir.join("telemetry.bin"))?);
        let dynamics = BufWriter::new(std::fs::File::create_new(run_dir.join("dynamics.bin"))?);
        Self::with_writers(file, Some(Box::new(dynamics)), receiver)
    }
    #[cfg(test)]
    fn with_writer(
        file: impl Write + Send + 'static,
        receiver: mpsc::Receiver<TelemetrySample>,
    ) -> std::io::Result<Self> {
        Self::with_writers(file, None, receiver)
    }
    fn with_writers(
        mut file: impl Write + Send + 'static,
        mut dynamics: Option<Box<dyn Write + Send>>,
        receiver: mpsc::Receiver<TelemetrySample>,
    ) -> std::io::Result<Self> {
        file.write_all(&telemetry_header())?;
        file.flush()?;
        if let Some(writer) = &mut dynamics {
            writer.write_all(&dynamics_header())?;
            writer.flush()?;
        }
        let ring = Arc::new(Mutex::new(VecDeque::new()));
        let written = Arc::new(Mutex::new(0u64));
        let shared_ring = ring.clone();
        let shared_written = written.clone();
        let failed = Arc::new(AtomicBool::new(false));
        let error = Arc::new(Mutex::new(None));
        let shared_failed = failed.clone();
        let shared_error = error.clone();
        let thread = thread::spawn(move || {
            let failure = |e: std::io::Error| {
                *shared_error.lock().unwrap() = Some(e.to_string());
                shared_failed.store(true, Ordering::Release);
            };
            let mut last_flush = Instant::now();
            while let Ok(sample) = receiver.recv() {
                if let Err(e) = file.write_all(&encode_sample(&sample)) {
                    failure(e);
                    return;
                }
                if let Some(writer) = &mut dynamics
                    && let Err(e) = writer.write_all(&encode_dynamics_sample(&sample))
                {
                    failure(e);
                    return;
                }
                *shared_written.lock().unwrap() += 1;
                let mut ring = shared_ring.lock().unwrap();
                let horizon = sample
                    .since_start
                    .saturating_sub(Duration::from_secs(RING_SECONDS));
                while ring
                    .front()
                    .is_some_and(|s: &TelemetrySample| s.since_start < horizon)
                {
                    ring.pop_front();
                }
                ring.push_back(sample);
                drop(ring);
                if last_flush.elapsed() > Duration::from_millis(100) {
                    if let Err(e) = file.flush() {
                        failure(e);
                        return;
                    }
                    if let Some(writer) = &mut dynamics
                        && let Err(e) = writer.flush()
                    {
                        failure(e);
                        return;
                    }
                    last_flush = Instant::now();
                }
            }
            if let Err(e) = file.flush() {
                failure(e);
                return;
            }
            if let Some(writer) = &mut dynamics
                && let Err(e) = writer.flush()
            {
                failure(e);
            }
        });
        Ok(Self {
            ring,
            written,
            failed,
            error,
            _thread: thread,
        })
    }
    fn window(&self, start_wall_ns: u64, end_wall_ns: u64) -> Vec<TelemetrySample> {
        self.ring
            .lock()
            .unwrap()
            .iter()
            .filter(|s| (start_wall_ns..=end_wall_ns).contains(&s.wall_ns))
            .cloned()
            .collect()
    }
}

/// The e-stop word for the mock backend follows a plain file the test edits:
/// `ok` releases, anything else (or a missing file) stops.
fn mock_estop_monitor(path: PathBuf, state: Arc<AtomicI32>, stop: Arc<AtomicBool>) {
    thread::spawn(move || {
        while !stop.load(Ordering::Acquire) {
            let word = match std::fs::read_to_string(&path) {
                Ok(text) if text.trim() == "ok" => estop::OK,
                _ => estop::FAULT,
            };
            state.store(word, Ordering::SeqCst);
            thread::sleep(Duration::from_millis(5));
        }
    });
}

struct Owner {
    args: Args,
    events: Events,
    session: Session,
    stop_word: Arc<AtomicI32>,
    worker: Option<Worker>,
    recorder: Option<Recorder>,
    telemetry_sender: Option<mpsc::SyncSender<TelemetrySample>>,
    profile: NativeProfile,
    policy: ContactPolicy,
    _hardware_lease: Option<Arc<HardwareLease>>,
    _mock_lease: Option<DriverLease>,
    _estop_monitor: Option<estop::Monitor>,
    mock_stop: Arc<AtomicBool>,
    acquisition_stop: Arc<AtomicBool>,
    releasing: Arc<AtomicBool>,
    last_tick_event: Instant,
    /// The fault the worker reported, published exactly once.
    reported_fault: Option<String>,
    reported_recorder_failure: bool,
}

impl Owner {
    fn status(&self) -> Option<Status> {
        self.worker.as_ref().map(Worker::status)
    }

    fn await_sequence(&self, sequence: u64, timeout: Duration) -> Result<Status> {
        let worker = self
            .worker
            .as_ref()
            .ok_or_else(|| Error("no driver".into()))?;
        let deadline = Instant::now() + timeout;
        loop {
            let status = worker.status();
            if status.sequence == sequence && status.completed {
                return Ok(status);
            }
            if Instant::now() >= deadline {
                return Err(Error(format!(
                    "worker did not complete sequence {sequence} within {timeout:?}; state unknown"
                )));
            }
            thread::sleep(Duration::from_millis(2));
        }
    }

    fn submit(&mut self, primitive: Primitive, timeout: Duration) -> Result<Status> {
        let sequence = self
            .worker
            .as_mut()
            .ok_or_else(|| Error("no driver".into()))?
            .submit(primitive)?;
        let status = self.await_sequence(sequence, timeout)?;
        match &status.fault {
            Some(fault) => Err(Error(fault.clone())),
            None => Ok(status),
        }
    }

    /// Open the selected driver behind the lease and the physical monitor,
    /// then a measured position hold: supported takeover, no motion target.
    fn connect(&mut self) -> Result<serde_json::Value> {
        let (sender, receiver) = mpsc::sync_channel(8192);
        let recorder = Recorder::start(&self.args.run_dir, receiver)
            .map_err(|e| Error(format!("telemetry recorder: {e}")))?;
        self.recorder = Some(recorder);
        self.telemetry_sender = Some(sender.clone());
        let worker = if self.args.backend == "mock" {
            let path = self
                .args
                .lease_path
                .clone()
                .ok_or_else(|| Error("mock backend needs --lease".into()))?;
            self._mock_lease = Some(DriverLease::acquire(&path)?);
            let estop_file = self
                .args
                .mock_estop_file
                .clone()
                .ok_or_else(|| Error("mock backend needs --mock-estop-file".into()))?;
            mock_estop_monitor(estop_file, self.stop_word.clone(), self.mock_stop.clone());
            let mut worker = self.profile.spawn_mock_guide(
                self.args.mock_seed,
                self.stop_word.clone(),
                self.policy,
                sender.clone(),
            )?;
            let sequence = worker.submit(Primitive::Reconnect(ArmConfig { joints: 6 }))?;
            self.worker = Some(worker);
            self.await_sequence(sequence, Duration::from_secs(5))?;
            self.worker.take().unwrap()
        } else {
            #[cfg(feature = "trossen")]
            {
                let device = self
                    .args
                    .estop_device
                    .clone()
                    .ok_or_else(|| Error("native backend needs --estop-device".into()))?;
                let lease = Arc::new(HardwareLease::acquire()?);
                let monitor = estop::Monitor::bind(
                    device,
                    self.args.run_dir.join("estop-snapshot.json"),
                    self.stop_word.clone(),
                )
                .map_err(|e| Error(format!("e-stop monitor: {e}")))?;
                self._estop_monitor = Some(monitor);
                self._hardware_lease = Some(lease.clone());
                let mut worker = self.profile.clone().spawn_native(
                    self.stop_word.clone(),
                    lease,
                    self.policy,
                    Some(sender.clone()),
                )?;
                // The heartbeat must be seen released before any command.
                let deadline = Instant::now() + Duration::from_secs(3);
                while self.stop_word.load(Ordering::SeqCst) != estop::OK {
                    if Instant::now() >= deadline {
                        return Err(Error(
                            "physical e-stop is not released; no driver command sent".into(),
                        ));
                    }
                    thread::sleep(Duration::from_millis(10));
                }
                let sequence = worker.submit(Primitive::Reconnect(ArmConfig { joints: 6 }))?;
                self.worker = Some(worker);
                self.await_sequence(sequence, Duration::from_secs(30))?;
                self.worker.take().unwrap()
            }
            #[cfg(not(feature = "trossen"))]
            {
                return Err(Error(
                    "this build has no native backend; rebuild with --features trossen".into(),
                ));
            }
        };
        self.worker = Some(worker);
        // Independent of stdin command handling and hold-window waits. A
        // recorder error or conductor loss reaches the worker's existing
        // software stop without waiting for the owner to finish a command.
        let stop = self.worker.as_ref().unwrap().software_stop.clone();
        let acquisition_stop = self.acquisition_stop.clone();
        let recorder_failed = self.recorder.as_ref().unwrap().failed.clone();
        let recorder_error = self.recorder.as_ref().unwrap().error.clone();
        let read_status = self.worker.as_ref().unwrap().status_reader();
        let shutdown = self.mock_stop.clone();
        let releasing = self.releasing.clone();
        thread::spawn(move || {
            while !shutdown.load(Ordering::Acquire) {
                if !releasing.load(Ordering::Acquire)
                    && !recorder_failed.load(Ordering::Acquire)
                    && read_status().telemetry_dropped > 0
                {
                    *recorder_error.lock().unwrap() =
                        Some("telemetry recorder could not retain every control tick".into());
                    recorder_failed.store(true, Ordering::Release);
                }
                if !releasing.load(Ordering::Acquire)
                    && (acquisition_stop.load(Ordering::Acquire)
                        || recorder_failed.load(Ordering::Acquire)
                        || STOP_SIGNAL.load(Ordering::Acquire))
                {
                    stop.store(true, Ordering::Release);
                }
                thread::sleep(Duration::from_millis(5));
            }
        });
        if self.args.backend == "trossen" {
            self.submit(
                Primitive::LoadConfig(self.profile.golden_path.clone()),
                Duration::from_secs(20),
            )?;
        }
        if let Some(fault) = self.status().and_then(|s| s.fault) {
            return Err(Error(format!("driver connection: {fault}")));
        }
        // Measured position hold at whatever pose the supported arm is in.
        let status = self.submit(Primitive::Reseed, Duration::from_secs(5))?;
        let measured = status
            .measured
            .clone()
            .ok_or_else(|| Error("no measurement after takeover".into()))?;
        let payload = self
            .submit(Primitive::ReadPayload, Duration::from_secs(5))?
            .payload;
        let _ = std::fs::write(
            self.args.run_dir.join("payload-readback.json"),
            serde_json::to_string_pretty(&payload).unwrap(),
        );
        Ok(serde_json::json!({
            "measured": measured,
            "carriage_datum_m": measured.carriage.as_ref().map(|c| c.position_m),
            "carriage_qualified": self.profile.carriage_qualified,
            "contact_policy": self.policy,
            "payload": payload,
            "contact_armed": status.contact.is_some_and(|c| c.armed),
        }))
    }

    fn wait_armed(&self, timeout: Duration) -> Result<()> {
        let deadline = Instant::now() + timeout;
        loop {
            let status = self.status().ok_or_else(|| Error("no driver".into()))?;
            if let Some(fault) = status.fault {
                return Err(Error(fault));
            }
            if status.contact.is_some_and(|c| c.armed) {
                return Ok(());
            }
            if Instant::now() >= deadline {
                return Err(Error(
                    "carriage contact baseline did not arm; keep the unloaded arm at rest".into(),
                ));
            }
            thread::sleep(Duration::from_millis(10));
        }
    }

    fn guide(&mut self, carriage_m: Option<f64>) -> Result<serde_json::Value> {
        self.wait_armed(Duration::from_secs(6))?;
        let status = self.status().unwrap();
        let measured = status
            .measured
            .and_then(|m| m.carriage)
            .map(|c| c.position_m)
            .ok_or_else(|| Error("no carriage measurement".into()))?;
        let datum = carriage_m.unwrap_or(measured);
        if (datum - measured).abs() > 0.0005 {
            return Err(Error(format!(
                "confirmed carriage datum {datum} differs from measured {measured}"
            )));
        }
        let status = self.submit(
            Primitive::HandGuide { carriage_m: datum },
            Duration::from_secs(5),
        )?;
        Ok(serde_json::json!({"receipt": status.guide, "measured": status.measured}))
    }

    fn hold(&mut self, id: &str, settle_s: f64, capture_s: f64) -> Result<HoldWindow> {
        let watch = |owner: &Self, until: Instant| -> Result<()> {
            while Instant::now() < until {
                if let Some(fault) = owner.status().and_then(|s| s.fault) {
                    return Err(Error(fault));
                }
                thread::sleep(Duration::from_millis(10));
            }
            Ok(())
        };
        watch(self, Instant::now() + Duration::from_secs_f64(settle_s))?;
        let start = wall_ns();
        self.events.emit(
            "capture_started",
            serde_json::json!({"id": id, "start_wall_ns": start}),
        );
        watch(self, Instant::now() + Duration::from_secs_f64(capture_s))?;
        let end = wall_ns();
        // Let the recorder drain the last ticks of the window.
        thread::sleep(Duration::from_millis(30));
        let samples = self.recorder.as_ref().unwrap().window(start, end);
        Ok(summarize_window(id, &samples, start, end))
    }

    fn stop(&mut self) -> Result<serde_json::Value> {
        let status = self.submit(Primitive::GuideStop, Duration::from_secs(5))?;
        Ok(serde_json::json!({"receipt": status.guide, "measured": status.measured}))
    }

    /// Clear a latched guard trip for an attended continuation: the arm is
    /// re-seeded in position control at its measured pose (a still-running
    /// trip retract finishes first). The carriage is left where the trip put
    /// it; `Carriage` returns it once the operator has guided the tool clear.
    fn recover(&mut self) -> Result<serde_json::Value> {
        if self.stop_word.load(Ordering::SeqCst) != tatbot_arm::estop::OK {
            return Err(Error(
                "e-stop asserted; release it before recovering".into(),
            ));
        }
        if let Some(status) = self.status()
            && !status.completed
        {
            let _ = self.await_sequence(status.sequence, Duration::from_secs(2));
        }
        let before = self.status().ok_or_else(|| Error("no driver".into()))?;
        if before.retract_verified == Some(false) {
            return Err(Error("trip retract unverified; release instead".into()));
        }
        let tripped = before.fault.clone();
        if before
            .measured
            .as_ref()
            .is_some_and(|m| !m.error.is_empty())
        {
            self.submit(Primitive::ClearError, Duration::from_secs(5))?;
        }
        let status = self.submit(Primitive::Reseed, Duration::from_secs(5))?;
        self.reported_fault = None;
        Ok(
            serde_json::json!({"cleared": tripped, "measured": status.measured,
                              "retract_verified": before.retract_verified}),
        )
    }

    /// Timed carriage-only move from a position hold, verified by readback.
    fn carriage(&mut self, carriage_m: f64) -> Result<serde_json::Value> {
        let measured = self
            .status()
            .and_then(|s| s.measured)
            .and_then(|m| m.carriage)
            .map(|c| c.position_m)
            .ok_or_else(|| Error("no carriage measurement".into()))?;
        // 50 mm/s cap with a floor of one second: a 37 mm datum return takes 1.5 s.
        let goal_time_s = ((carriage_m - measured).abs() / 0.025).clamp(1.0, 5.0);
        let status = self.submit(
            Primitive::CarriageTo {
                target_m: carriage_m,
                goal_time_s,
            },
            Duration::from_secs_f64(goal_time_s + 3.0),
        )?;
        Ok(
            serde_json::json!({"carriage_m": carriage_m, "goal_time_s": goal_time_s, "measured": status.measured}),
        )
    }

    /// A single bounded rotary axis step, with the other five measured axes
    /// held. The worker supplies its existing limits, contact cap and stop.
    fn joint_step(
        &mut self,
        joint_index: usize,
        delta_rad: f64,
        speed_rad_s: f64,
    ) -> Result<serde_json::Value> {
        let before = self
            .status()
            .and_then(|s| s.measured)
            .ok_or_else(|| Error("no arm measurement".into()))?;
        if before.joints.len() != 6 || before.joints.iter().any(|q| !q.is_finite()) {
            return Err(Error("invalid measured rotary pose".into()));
        }
        if before.velocities.len() != 6
            || before
                .velocities
                .iter()
                .any(|v| !v.is_finite() || v.abs() > 0.075)
        {
            return Err(Error("joint step requires a quiet measured hold".into()));
        }
        let mut target = before.joints.clone();
        target[joint_index] += delta_rad;
        let limit = self.profile.position_limits[joint_index];
        // Keep a measured-arrival margin inside the configured command range.
        if target[joint_index] < limit.min + 0.01 || target[joint_index] > limit.max - 0.01 {
            return Err(Error(
                "joint step too close to configured position limit".into(),
            ));
        }
        // The worker freezes each completed angular stream at its *measured*
        // endpoint, so waiting cannot close a following error. Up to two
        // smaller commands may approach the same original target. Neither
        // the target nor speed grows, and every intermediate result is checked.
        let mut current = before.clone();
        let mut attempts = Vec::new();
        for attempt in 0..=2 {
            let mut command_target = current.joints.clone();
            command_target[joint_index] = target[joint_index];
            let requested_rad = (target[joint_index] - current.joints[joint_index]).abs();
            let goal_time_s = (1.5 * requested_rad / speed_rad_s).max(0.5);
            let status = self.submit(
                Primitive::MoveJ {
                    target: command_target.clone(),
                    goal_time_s,
                },
                Duration::from_secs_f64(goal_time_s + 5.0),
            )?;
            if status.telemetry_dropped != 0 || !status.watchdog_ok {
                return Err(Error("joint step lost telemetry or watchdog health".into()));
            }
            let measured = status
                .measured
                .ok_or_else(|| Error("no joint arrival measurement".into()))?;
            if measured.mode != tatbot_arm::Mode::Position
                || measured.joints.len() != 6
                || measured.joints.iter().any(|q| !q.is_finite())
            {
                return Err(Error("invalid joint arrival measurement".into()));
            }
            let selected_error = (measured.joints[joint_index] - target[joint_index]).abs();
            let other_error = measured
                .joints
                .iter()
                .zip(&before.joints)
                .enumerate()
                .filter(|(index, _)| *index != joint_index)
                .map(|(_, (q, seed))| (q - seed).abs())
                .fold(0.0_f64, f64::max);
            let low = before.joints[joint_index].min(target[joint_index]) - 0.005;
            let high = before.joints[joint_index].max(target[joint_index]) + 0.005;
            let progress = requested_rad - selected_error;
            let receipt = serde_json::json!({
                "attempt": attempt, "command_target": command_target,
                "goal_time_s": goal_time_s, "measured": measured,
                "selected_error_rad": selected_error, "other_error_rad": other_error,
                "progress_rad": progress,
            });
            self.events.emit("joint_step_progress", receipt.clone());
            attempts.push(receipt);
            if measured.joints[joint_index] < low
                || measured.joints[joint_index] > high
                || other_error > 0.005
                || progress < 0.001
            {
                return Err(Error(format!(
                    "joint step tracking departed bound: selected error {selected_error:.4} rad, other error {other_error:.4} rad, progress {progress:.4} rad"
                )));
            }
            if selected_error <= 0.005 {
                return Ok(serde_json::json!({
                    "before": before, "target": target, "measured": measured,
                    "joint_index": joint_index, "delta_rad": delta_rad,
                    "speed_rad_s": speed_rad_s, "max_error_rad": selected_error.max(other_error),
                    "attempts": attempts,
                }));
            }
            current = measured;
        }
        let error = (current.joints[joint_index] - target[joint_index]).abs();
        Err(Error(format!(
            "joint step arrival error {error:.4} rad after three bounded commands"
        )))
    }

    /// A parked feedback value can sit just beyond a nominal hard stop. The
    /// recovery path clamps the takeover target, then travels only the small
    /// amount needed to get those axes inside the legal command range.
    fn normalize(&mut self) -> Result<serde_json::Value> {
        use tatbot_arm::recovery::{Phase, STAGED_POSE_S};
        let before = self
            .status()
            .and_then(|s| s.measured)
            .ok_or_else(|| Error("no arm measurement".into()))?;
        let carriage = before
            .carriage
            .as_ref()
            .ok_or_else(|| Error("no carriage measurement".into()))?;
        if before.joints.len() != 6 || before.joints.iter().any(|q| !q.is_finite()) {
            return Err(Error("invalid measured rotary pose".into()));
        }
        if before.velocities.len() != 6
            || before
                .velocities
                .iter()
                .any(|v| !v.is_finite() || v.abs() > 0.075)
        {
            return Err(Error("normalization requires a quiet measured hold".into()));
        }
        let mut staged = [0.0; 7];
        staged[..6].copy_from_slice(&before.joints);
        staged[6] = carriage.position_m;
        let mut moved = Vec::new();
        for (index, limit) in self.profile.position_limits[..6].iter().enumerate() {
            let target = staged[index].clamp(limit.min + 0.02, limit.max - 0.02);
            if (target - staged[index]).abs() > 1e-6 {
                if (target - staged[index]).abs() > 0.03 {
                    return Err(Error(
                        "normalization would exceed 0.03 rad on one axis".into(),
                    ));
                }
                moved.push(index);
                staged[index] = target;
            }
        }
        if moved.len() > 2 {
            return Err(Error(
                "normalization would move more than two rotary axes".into(),
            ));
        }
        if moved.is_empty() {
            return Ok(serde_json::json!({"before": before, "measured": before,
                "target": staged, "moved_indices": moved, "no_motion": true}));
        }
        self.submit(
            Primitive::PrepareRecovery { staged },
            Duration::from_secs(10),
        )?;
        self.submit(Primitive::Takeover, Duration::from_secs(3))?;
        let status = self.submit(
            Primitive::RecoveryMove(Phase::Staged),
            Duration::from_secs_f64(STAGED_POSE_S + 3.0),
        )?;
        let measured = status
            .measured
            .ok_or_else(|| Error("no normalization arrival readback".into()))?;
        let max_error = measured
            .joints
            .iter()
            .zip(&staged[..6])
            .map(|(q, expected)| (q - expected).abs())
            .fold(0.0_f64, f64::max);
        if measured.joints.len() != 6 || max_error > 0.005 {
            return Err(Error(format!(
                "normalization arrival error {max_error:.4} rad"
            )));
        }
        Ok(serde_json::json!({"before": before, "target": staged,
            "measured": measured, "moved_indices": moved,
            "max_error_rad": max_error, "no_motion": false}))
    }

    fn wrist(&mut self, angle_rad: f64, base_rad: Option<f64>) -> Result<serde_json::Value> {
        use tatbot_arm::recovery::{Phase, STAGED_POSE_S, TAKEOVER_S};
        let before = self
            .status()
            .and_then(|s| s.measured)
            .ok_or_else(|| Error("no arm measurement".into()))?;
        let (target, travel_s) = tatbot_arm::guide::wrist_target(
            &before.joints,
            &self.profile.recovery.staged_positions,
            angle_rad,
            base_rad,
        )?;
        for (q, limit) in target.iter().zip(&self.profile.position_limits) {
            if *q < limit.min || *q > limit.max {
                return Err(Error("inspection target exceeds controller limits".into()));
            }
        }
        let steps = (travel_s / STAGED_POSE_S).ceil() as usize;
        let mut arrivals = Vec::new();
        for index in 1..=steps {
            let mut staged = self.profile.recovery.staged_positions;
            staged[..6].copy_from_slice(&target);
            for axis in [0, 5] {
                staged[axis] = before.joints[axis]
                    + (target[axis] - before.joints[axis]) * index as f64 / steps as f64;
            }
            // The existing recovery path prepares a legal measured takeover,
            // validates live limits, and monitors the timed vendor move. Do not
            // stream negative hard-stop feedback as a new commanded target.
            self.submit(
                Primitive::PrepareRecovery { staged },
                Duration::from_secs(10),
            )?;
            self.submit(Primitive::Takeover, Duration::from_secs(3))?;
            let status = self.submit(
                Primitive::RecoveryMove(Phase::Staged),
                Duration::from_secs_f64(STAGED_POSE_S + 3.0),
            )?;
            let measured = status
                .measured
                .ok_or_else(|| Error("no inspection arrival readback".into()))?;
            let worst = measured
                .joints
                .iter()
                .zip(&staged[..6])
                .map(|(a, b)| (a - b).abs())
                .fold(0.0_f64, f64::max);
            if measured.joints.len() != 6 || worst > 0.02 {
                return Err(Error(format!(
                    "inspection pose not reached: maximum rotary error {worst} rad"
                )));
            }
            arrivals.push(serde_json::json!({"step": index, "target": staged,
                "measured": measured, "max_error_rad": worst}));
        }
        let final_pose = arrivals
            .last()
            .ok_or_else(|| Error("no inspection steps".into()))?;
        Ok(serde_json::json!({"before": before, "target": target,
            "goal_time_s": steps as f64 * (STAGED_POSE_S + TAKEOVER_S),
            "measured": final_pose["measured"], "max_error_rad": final_pose["max_error_rad"],
            "arrivals": arrivals}))
    }

    /// Terminal idle release: no re-seed, position target or future motion.
    fn release(&mut self) -> Result<serde_json::Value> {
        if self.releasing.load(Ordering::Acquire) && self.worker.is_none() {
            return Err(Error("previous terminal idle release failed".into()));
        }
        self.releasing.store(true, Ordering::Release);
        // Let an already-triggered carriage retract reach its existing
        // qualified endpoint before removing drive effort.
        if let Some(status) = self.status()
            && !status.completed
        {
            let _ = self.await_sequence(status.sequence, Duration::from_secs(2));
            self.tick_event();
        }
        let retract_verified = self.status().and_then(|status| status.retract_verified);
        let Some(mut worker) = self.worker.take() else {
            return Ok(serde_json::json!({"no_driver": true}));
        };
        let result = worker.shutdown_idle();
        if let Err(error) = &result {
            self.events.emit(
                "release_failed",
                serde_json::json!({
                    "reason": error.to_string(), "idle_verified": false,
                }),
            );
        }
        // Native destruction has its own deadline if an SDK call is blocked.
        drop(worker);
        let measured = result?;
        Ok(serde_json::json!({"measured": measured, "retract_verified": retract_verified}))
    }

    /// Faults are published as soon as the owner sees them; ticks at 5 Hz.
    fn tick_event(&mut self) {
        if let Some(recorder) = &self.recorder
            && recorder.failed.load(Ordering::Acquire)
            && !self.reported_recorder_failure
        {
            self.reported_recorder_failure = true;
            self.events.emit(
                "recorder_failed",
                serde_json::json!({
                    "reason": recorder.error.lock().unwrap().clone(),
                }),
            );
        }
        let Some(status) = self.status() else { return };
        let due = self.last_tick_event.elapsed() >= TICK_EVENT_PERIOD;
        if !due && status.fault.is_none() {
            return;
        }
        if due {
            self.last_tick_event = Instant::now();
        }
        if let Some(measured) = &status.measured
            && due
        {
            let tick = Tick::from_measured(
                measured,
                self.stop_word.load(Ordering::SeqCst),
                status.guiding,
                status.fault.clone(),
                status.late_ticks,
                status.telemetry_dropped,
            );
            self.events
                .emit("tick", serde_json::to_value(tick).unwrap());
        }
        if let Some(fault) = status.fault
            && self.reported_fault.as_ref() != Some(&fault)
        {
            self.reported_fault = Some(fault.clone());
            self.session.fault();
            self.events.emit(
                "fault",
                serde_json::json!({"reason": fault, "state": self.session.state, "guiding": status.guiding,
                                    "estop": self.stop_word.load(Ordering::SeqCst),
                                    "retract_verified": status.retract_verified}),
            );
        }
    }

    fn handle(&mut self, command: Command) {
        let next = match self.session.admit(&command) {
            Ok(next) => next,
            Err(refusal) => {
                self.events.emit(
                    "refused",
                    serde_json::json!({"reason": refusal.to_string(), "state": self.session.state}),
                );
                return;
            }
        };
        let outcome: Result<(&str, serde_json::Value)> = match &command {
            Command::Connect => self.connect().map(|body| ("connected", body)),
            Command::Guide { carriage_m } => self.guide(*carriage_m).map(|body| ("guiding", body)),
            Command::Hold {
                id,
                settle_s,
                capture_s,
            } => {
                self.session.state = State::Recording;
                let result = self.hold(id, *settle_s, *capture_s);
                self.session.recorded();
                match result {
                    Ok(window) => {
                        self.events
                            .emit("hold", serde_json::to_value(window).unwrap());
                        return;
                    }
                    Err(error) => Err(error),
                }
            }
            Command::Stop => self.stop().map(|body| ("holding", body)),
            Command::Recover => self.recover().map(|body| ("recovered", body)),
            Command::Carriage { carriage_m } => {
                self.carriage(*carriage_m).map(|body| ("carriage", body))
            }
            Command::Normalize => self.normalize().map(|body| ("normalized", body)),
            Command::JointStep {
                joint_index,
                delta_rad,
                speed_rad_s,
            } => self
                .joint_step(*joint_index, *delta_rad, *speed_rad_s)
                .map(|body| ("joint_step", body)),
            Command::Wrist {
                angle_rad,
                base_rad,
            } => self
                .wrist(*angle_rad, *base_rad)
                .map(|body| ("wrist", body)),
            Command::Abort { reason } => {
                self.acquisition_stop.store(true, Ordering::Release);
                self.conductor_lost(reason);
                return;
            }
            Command::Release => self.release().map(|body| ("released", body)),
            Command::Inject(faults) => {
                if self.args.backend != "mock" {
                    Err(Error("fault injection is mock-only".into()))
                } else {
                    self.submit(Primitive::Inject(faults.clone()), Duration::from_secs(2))
                        .map(|_| ("injected", serde_json::json!({})))
                }
            }
            Command::Status => Ok((
                "status",
                serde_json::json!({"state": self.session.state,
                "telemetry_written": self.recorder.as_ref().map(Recorder::written),
                "worker": self.status().map(|s| serde_json::json!({
                    "fault": s.fault, "guiding": s.guiding, "measured": s.measured, "late_ticks": s.late_ticks,
                    "max_late_us": s.max_late_us, "telemetry_dropped": s.telemetry_dropped,
                    "contact": s.contact.map(|c| serde_json::json!({"armed": c.armed, "contact_n": c.contact_n, "baseline_n": c.baseline_n})),
                    "guide": s.guide}))}),
            )),
        };
        match outcome {
            Ok((event, mut body)) => {
                if !matches!(command, Command::Status | Command::Inject(_)) {
                    self.session.state = next;
                }
                body.as_object_mut().unwrap().insert(
                    "state".into(),
                    serde_json::to_value(self.session.state).unwrap(),
                );
                self.events.emit(event, body);
                if next == State::Released {
                    self.session.close();
                    self.events
                        .emit("closed", serde_json::json!({"state": self.session.state}));
                }
            }
            Err(error) => {
                if !matches!(command, Command::Status | Command::Inject(_)) {
                    self.session.fault();
                }
                self.reported_fault = self
                    .status()
                    .and_then(|s| s.fault)
                    .or(self.reported_fault.clone());
                self.events.emit(
                    "fault",
                    serde_json::json!({"reason": error.to_string(), "refused_command": format!("{command:?}"), "state": self.session.state,
                                        "guiding": self.status().map(|s| s.guiding), "estop": self.stop_word.load(Ordering::SeqCst)}),
                );
            }
        }
    }

    /// Stop acquisition before terminal idle release of the supported arm.
    fn conductor_lost(&mut self, reason: &str) {
        let guiding = self.status().is_some_and(|s| s.guiding);
        let stop = if guiding {
            self.submit(Primitive::GuideStop, Duration::from_secs(5))
                .map(|_| ())
        } else if self
            .status()
            .and_then(|s| s.measured)
            .is_some_and(|m| m.mode == tatbot_arm::Mode::Position)
        {
            self.submit(Primitive::Hold, Duration::from_secs(5))
                .map(|_| ())
        } else {
            Ok(())
        };
        // A terminated capture leaves a qualified carriage's tool retracted
        // from the fixture before its motors idle, as a trip would have.
        let retract = if stop.is_ok() {
            self.park_carriage()
        } else {
            Ok(())
        };
        self.session.fault();
        self.events.emit(
            "conductor_lost",
            serde_json::json!({"reason": reason, "stop": stop.as_ref().err().map(ToString::to_string),
                                "stop_measured": self.status().and_then(|s| s.measured),
                                "retract": retract.as_ref().err().map(ToString::to_string),
                                "state": self.session.state, "run_id": self.run_id(),
                                "release_request": self.args.run_dir.join(RELEASE_REQUEST)}),
        );
    }

    /// Qualified carriages only: the retract endpoint, from a position hold,
    /// unless the carriage is already there. Never latches on refusal.
    fn park_carriage(&mut self) -> Result<()> {
        if !self.profile.carriage_qualified {
            return Ok(());
        }
        let Some(status) = self.status() else {
            return Ok(());
        };
        if status.fault.is_some() || !status.completed {
            return Ok(());
        }
        let carriage = status
            .measured
            .and_then(|m| m.carriage)
            .ok_or_else(|| Error("no carriage measurement".into()))?;
        if carriage.position_m.is_finite() && (carriage.position_m - PARK_CARRIAGE_M).abs() <= 0.002
        {
            return Ok(());
        }
        self.carriage(PARK_CARRIAGE_M).map(|_| ())
    }

    fn run_id(&self) -> String {
        self.args.run_id.clone().unwrap_or_else(|| {
            self.args
                .run_dir
                .file_name()
                .and_then(|s| s.to_str())
                .unwrap_or("run")
                .to_string()
        })
    }

    /// A `release.request` naming this run asks the owner to release the arm
    /// it is holding; the operator writes `{"run_id": "<this run>"}` to the
    /// path `owner.json` names once the cause of a fault is cleared and the
    /// arm is supported. Returns true once the arm is released and the
    /// session closed.
    fn poll_release_request(&mut self) -> bool {
        let path = self.args.run_dir.join(RELEASE_REQUEST);
        let Ok(text) = std::fs::read_to_string(&path) else {
            return false;
        };
        #[derive(Deserialize)]
        struct Request {
            run_id: String,
        }
        let accepted =
            serde_json::from_str::<Request>(&text).is_ok_and(|r| r.run_id == self.run_id());
        let _ = std::fs::remove_file(&path);
        if !accepted {
            self.events.emit(
                "refused",
                serde_json::json!({"reason": "release request names another run"}),
            );
            return false;
        }
        let outcome = self.release();
        let receipt = serde_json::json!({"released": outcome.is_ok(), "error": outcome.as_ref().err().map(ToString::to_string), "wall_ns": wall_ns()});
        let _ = std::fs::write(
            self.args.run_dir.join("release.receipt"),
            serde_json::to_string_pretty(&receipt).unwrap(),
        );
        self.events.emit(
            if outcome.is_ok() {
                "released"
            } else {
                "release_refused"
            },
            serde_json::json!({"local": true, "outcome": receipt,
                               "reason": outcome.as_ref().err().map(ToString::to_string)}),
        );
        if outcome.is_ok() {
            self.session.close();
            self.events
                .emit("closed", serde_json::json!({"state": self.session.state}));
            return true;
        }
        false
    }
}

static STOP_SIGNAL: AtomicBool = AtomicBool::new(false);
extern "C" fn on_stop_signal(_: libc::c_int) {
    STOP_SIGNAL.store(true, Ordering::Release);
}

fn main() {
    // Read-only identity query: no profile, run directory, lease, serial
    // monitor or SDK object is constructed on this path.
    if std::env::args().skip(1).eq(["--build-info"]) {
        println!("{}", build_info());
        return;
    }
    let args = parse_args();
    std::fs::create_dir_all(&args.run_dir).expect("run directory");
    let mut events = Events::open(&args.run_dir).expect("events log");
    let role = match NativeProfile::role_named(&args.controller_role) {
        Ok(role) => role,
        Err(error) => {
            events.emit(
                "fault",
                serde_json::json!({"reason": error.to_string(), "state": "preflight"}),
            );
            std::process::exit(3);
        }
    };
    // Preflight before any lease, serial or SDK access: the exact controller
    // profile and golden this owner will use are snapshotted into the run.
    let profile = match NativeProfile::load_role(&args.profile_dir, role)
        .and_then(|profile| profile.snapshot(&args.run_dir.join("profile")))
    {
        Ok(profile) => profile,
        Err(error) => {
            events.emit(
                "fault",
                serde_json::json!({"reason": error.to_string(), "state": "preflight"}),
            );
            std::process::exit(3);
        }
    };
    // The fitted tool's datasheet decides the worker's contact policy; its
    // digest binds the owner record to the exact sheet read.
    let (policy, datasheet_sha256) = match std::fs::read(&args.tool_datasheet)
        .map_err(|e| {
            Error(format!(
                "tool datasheet {}: {e}",
                args.tool_datasheet.display()
            ))
        })
        .and_then(|bytes| {
            let datasheet: serde_json::Value = serde_yaml::from_slice(&bytes).map_err(|e| {
                Error(format!(
                    "tool datasheet {}: {e}",
                    args.tool_datasheet.display()
                ))
            })?;
            let policy = ContactPolicy::from_datasheet(&datasheet)?;
            Ok((policy, format!("{:x}", Sha256::digest(&bytes))))
        }) {
        Ok(read) => read,
        Err(error) => {
            events.emit(
                "fault",
                serde_json::json!({"reason": error.to_string(), "state": "preflight"}),
            );
            std::process::exit(3);
        }
    };
    // SAFETY: the handler only stores an atomic flag.
    unsafe {
        let handler = on_stop_signal as extern "C" fn(libc::c_int) as libc::sighandler_t;
        libc::signal(libc::SIGTERM, handler);
        libc::signal(libc::SIGINT, handler);
        libc::signal(libc::SIGHUP, handler);
    }
    let owner_record = serde_json::json!({
        "schema": "tatbot.arm-guide-owner/1",
        "run_id": args.run_id.clone().unwrap_or_else(|| args.run_dir.file_name().and_then(|s| s.to_str()).unwrap_or("run").to_string()),
        "pid": std::process::id(),
        "backend": args.backend,
        "controller_role": args.controller_role,
        "profile_sha256": profile.profile_sha256,
        "golden_sha256": profile.golden_sha256,
        "carriage_qualified": profile.carriage_qualified,
        "tool_datasheet": args.tool_datasheet,
        "tool_datasheet_sha256": datasheet_sha256,
        "contact_policy": policy,
        "sdk_version": option_env!("TATBOT_SDK_VERSION").unwrap_or("1.8.5"),
        "source_commit": option_env!("TATBOT_SOURCE_COMMIT").unwrap_or("development"),
        "native_backend_compiled": cfg!(feature = "trossen"),
        "build": build_info(),
        "estop_device": args.estop_device,
        "started_wall_ns": wall_ns(),
        "telemetry": "telemetry.bin (TBGUIDE1, every control tick)",
        "dynamics": "dynamics.bin (TBDYNA01, matched by tick and wall ns)",
        "release_request": RELEASE_REQUEST,
    });
    std::fs::write(
        args.run_dir.join("owner.json"),
        serde_json::to_string_pretty(&owner_record).unwrap(),
    )
    .expect("owner record");
    let mut owner = Owner {
        session: Session::new(),
        stop_word: Arc::new(AtomicI32::new(estop::FAULT)),
        worker: None,
        recorder: None,
        telemetry_sender: None,
        profile,
        policy,
        _hardware_lease: None,
        _mock_lease: None,
        _estop_monitor: None,
        mock_stop: Arc::new(AtomicBool::new(false)),
        acquisition_stop: Arc::new(AtomicBool::new(false)),
        releasing: Arc::new(AtomicBool::new(false)),
        last_tick_event: Instant::now(),
        reported_fault: None,
        reported_recorder_failure: false,
        events,
        args,
    };
    owner.events.emit("ready", owner_record.clone());

    // stdin on its own thread so ticks and faults are published while idle.
    let (lines, incoming) = mpsc::channel::<Option<String>>();
    let acquisition_stop = owner.acquisition_stop.clone();
    thread::spawn(move || {
        let stdin = std::io::stdin();
        for line in stdin.lock().lines() {
            match line {
                Ok(line) => {
                    if matches!(
                        serde_json::from_str::<Command>(&line),
                        Ok(Command::Abort { .. })
                    ) {
                        acquisition_stop.store(true, Ordering::Release);
                    }
                    if lines.send(Some(line)).is_err() {
                        return;
                    }
                }
                Err(_) => break,
            }
        }
        acquisition_stop.store(true, Ordering::Release);
        let _ = lines.send(None);
    });
    while owner.session.state != State::Closed {
        owner.tick_event();
        // A fault holds the arm where the worker stopped it and waits for the
        // conductor's decision: Recover continues the attended capture,
        // Release idles. Conductor loss (EOF, signal) or a release request
        // still ends in terminal idle release; nothing holds unattended.
        if matches!(owner.session.state, State::Holding | State::Faulted)
            && owner.poll_release_request()
        {
            break;
        }
        if STOP_SIGNAL.swap(false, Ordering::AcqRel) {
            if owner.worker.is_some() {
                owner.conductor_lost("stop signal");
            }
            break;
        }
        match incoming.recv_timeout(Duration::from_millis(50)) {
            Ok(Some(line)) => {
                let line = line.trim().to_string();
                if line.is_empty() {
                    continue;
                }
                owner.events.record_command(&line);
                match serde_json::from_str::<Command>(&line) {
                    Ok(command) => owner.handle(command),
                    Err(error) => owner.events.emit(
                        "refused",
                        serde_json::json!({"reason": format!("unparseable command: {error}")}),
                    ),
                }
            }
            Ok(None) => {
                if owner.session.state != State::Closed && owner.worker.is_some() {
                    owner.conductor_lost("conductor stdin closed");
                }
                break;
            }
            Err(mpsc::RecvTimeoutError::Timeout) => {}
            Err(mpsc::RecvTimeoutError::Disconnected) => break,
        }
    }
    if owner.session.state != State::Closed {
        match owner.release() {
            Ok(body) => {
                owner.events.emit("released", body);
                owner.session.close();
                owner
                    .events
                    .emit("closed", serde_json::json!({"state": owner.session.state}));
            }
            Err(error) => owner.events.emit(
                "release_failed",
                serde_json::json!({
                    "reason": error.to_string(), "idle_verified": false,
                }),
            ),
        }
    }
    owner.mock_stop.store(true, Ordering::Release);
    let mut exit = if owner.session.state == State::Closed {
        0
    } else {
        3
    };
    // Dropping the worker joins its thread; a released arm receives no hold.
    drop(owner.worker.take());
    drop(owner.telemetry_sender.take());
    if let Some(recorder) = owner.recorder.take() {
        let deadline = Instant::now() + Duration::from_secs(2);
        while !recorder._thread.is_finished() && Instant::now() < deadline {
            thread::sleep(Duration::from_millis(10));
        }
        if !recorder._thread.is_finished() || recorder.failed.load(Ordering::Acquire) {
            exit = 3;
            owner.events.emit("recorder_failed", serde_json::json!({
                "reason": recorder.error.lock().unwrap().clone().unwrap_or_else(|| "recorder did not finish".into()),
            }));
        }
    }
    owner.events.emit(
        "exit",
        serde_json::json!({"code": exit, "state": owner.session.state}),
    );
    std::process::exit(exit);
}

#[cfg(test)]
mod recorder_tests {
    use super::*;

    struct SharedBytes(Arc<Mutex<Vec<u8>>>);
    impl Write for SharedBytes {
        fn write(&mut self, bytes: &[u8]) -> std::io::Result<usize> {
            self.0.lock().unwrap().extend_from_slice(bytes);
            Ok(bytes.len())
        }
        fn flush(&mut self) -> std::io::Result<()> {
            Ok(())
        }
    }

    #[test]
    fn recorder_writes_paired_legacy_and_dynamics_ticks() {
        let legacy = Arc::new(Mutex::new(Vec::new()));
        let dynamics = Arc::new(Mutex::new(Vec::new()));
        let (sender, receiver) = mpsc::channel();
        let recorder = Recorder::with_writers(
            SharedBytes(legacy.clone()),
            Some(Box::new(SharedBytes(dynamics.clone()))),
            receiver,
        )
        .unwrap();
        sender
            .send(TelemetrySample {
                tick: 42,
                wall_ns: 100,
                since_start: Duration::from_millis(1),
                estop: estop::OK,
                guiding: false,
                latched: false,
                measured: tatbot_arm::Measured {
                    joints: vec![0.0; 6],
                    velocities: vec![0.0; 6],
                    efforts: vec![0.0; 6],
                    dynamics: Some(tatbot_arm::JointDynamics {
                        accelerations: vec![1.0; 7],
                        efforts: vec![2.0; 7],
                        compensation_efforts: vec![3.0; 7],
                    }),
                    mode: tatbot_arm::Mode::Position,
                    error: String::new(),
                    carriage: Some(tatbot_arm::CarriageMeasured {
                        position_m: 0.002,
                        target_m: Some(0.002),
                        effort_n: 0.0,
                    }),
                },
            })
            .unwrap();
        drop(sender);
        recorder._thread.join().unwrap();
        let legacy = legacy.lock().unwrap();
        let dynamics = dynamics.lock().unwrap();
        assert_eq!(legacy.len(), 16 + tatbot_arm::guide::TELEMETRY_RECORD_BYTES);
        assert_eq!(
            dynamics.len(),
            16 + tatbot_arm::guide::DYNAMICS_RECORD_BYTES
        );
        assert_eq!(legacy[16..40], dynamics[16..40]);
        assert_eq!(dynamics[40..48], 1.0f64.to_le_bytes());
        assert_eq!(dynamics[40 + 7 * 8..48 + 7 * 8], 2.0f64.to_le_bytes());
    }

    #[test]
    fn sdk_stdout_is_separate_from_protocol_events() {
        const CHILD: &str = "TATBOT_TEST_PROTOCOL_CHILD";
        if std::env::var_os(CHILD).is_some() {
            let directory = tempfile::tempdir().unwrap();
            let mut events = Events::open(directory.path()).unwrap();
            std::io::stdout()
                .write_all(b"SDK diagnostic on fd 1\n")
                .unwrap();
            std::io::stdout().flush().unwrap();
            events.emit("test_protocol", serde_json::json!({}));
            return;
        }
        let output = std::process::Command::new(std::env::current_exe().unwrap())
            .args([
                "--exact",
                "recorder_tests::sdk_stdout_is_separate_from_protocol_events",
                "--nocapture",
            ])
            .env(CHILD, "1")
            .output()
            .unwrap();
        assert!(output.status.success());
        let stdout = String::from_utf8(output.stdout).unwrap();
        let stderr = String::from_utf8(output.stderr).unwrap();
        assert!(stdout.contains("test_protocol"));
        assert!(!stdout.contains("SDK diagnostic"));
        assert!(stderr.contains("SDK diagnostic"));
    }

    struct FailingWriter {
        bytes: usize,
        fail_flush: bool,
    }
    impl Write for FailingWriter {
        fn write(&mut self, bytes: &[u8]) -> std::io::Result<usize> {
            if self.bytes >= 16 && !self.fail_flush {
                return Err(std::io::Error::other("injected disk full"));
            }
            self.bytes += bytes.len();
            Ok(bytes.len())
        }
        fn flush(&mut self) -> std::io::Result<()> {
            if self.fail_flush && self.bytes > 16 {
                Err(std::io::Error::other("injected flush failure"))
            } else {
                Ok(())
            }
        }
    }

    #[test]
    fn recorder_write_and_flush_errors_are_retained_as_stop_signals() {
        for fail_flush in [false, true] {
            let (sender, receiver) = mpsc::channel();
            let recorder = Recorder::with_writer(
                FailingWriter {
                    bytes: 0,
                    fail_flush,
                },
                receiver,
            )
            .unwrap();
            sender
                .send(TelemetrySample {
                    tick: 1,
                    wall_ns: 1,
                    since_start: Duration::from_millis(1),
                    estop: estop::OK,
                    guiding: true,
                    latched: false,
                    measured: tatbot_arm::Measured {
                        joints: vec![0.0; 6],
                        velocities: vec![0.0; 6],
                        efforts: vec![0.0; 6],
                        dynamics: None,
                        mode: tatbot_arm::Mode::HandGuiding,
                        error: String::new(),
                        carriage: Some(tatbot_arm::CarriageMeasured {
                            position_m: 0.002,
                            target_m: Some(0.002),
                            effort_n: 0.0,
                        }),
                    },
                })
                .unwrap();
            drop(sender);
            let deadline = Instant::now() + Duration::from_secs(2);
            while !recorder.failed.load(Ordering::Acquire) && Instant::now() < deadline {
                thread::sleep(Duration::from_millis(1));
            }
            assert!(recorder.failed.load(Ordering::Acquire));
            assert!(
                recorder
                    .error
                    .lock()
                    .unwrap()
                    .as_ref()
                    .unwrap()
                    .contains("injected")
            );
            if !fail_flush {
                assert_eq!(recorder.written(), 0);
            }
            recorder._thread.join().unwrap();
        }
    }
}
