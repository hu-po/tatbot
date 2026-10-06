use super::*;
use crate::{ArmConfig, Faults, JointSample, MockArm, Mode};
use std::path::Path;
use std::sync::atomic::{AtomicI32, AtomicU64};

const EPOCH: u64 = 1_000_000_000;
const WALL_LIMIT: Duration = Duration::from_secs(3);

/// Commands become measurements only when the world advances. This catches an
/// acknowledgement of a command before its physical step/feedback has happened.
struct WorldProbe {
    arm: MockArm,
    pending: Option<JointSample>,
    steps: Arc<AtomicU64>,
    delay: Duration,
    fail_step: Option<u64>,
    fail_read: Option<u64>,
    fault_read: Option<u64>,
}

impl ArmBackend for WorldProbe {
    fn connect(&mut self, cfg: &ArmConfig) -> Result<()> {
        self.arm.connect(cfg)
    }
    fn measured(&mut self) -> Result<Measured> {
        if self.fail_read == Some(self.steps.load(Ordering::Acquire)) {
            return Err(Error("probe measurement transport lost".into()));
        }
        let mut measured = self.arm.measured()?;
        if self.fault_read == Some(self.steps.load(Ordering::Acquire)) {
            measured.mode = Mode::Fault;
        }
        Ok(measured)
    }
    fn recovery_limits(&mut self) -> Result<[crate::recovery::PositionLimit; 7]> {
        self.arm.recovery_limits()
    }
    fn retract_carriage(&mut self, target: f64, duration: f64) -> Result<()> {
        self.arm.retract_carriage(target, duration)
    }
    fn set_mode(&mut self, mode: Mode) -> Result<()> {
        self.arm.set_mode(mode)
    }
    fn takeover(&mut self, target: &[f64; 7]) -> Result<()> {
        self.arm.takeover(target)
    }
    fn move_j(&mut self, target: &[f64], duration: f64) -> Result<()> {
        self.arm.move_j(target, duration)
    }
    fn recovery_move(&mut self, target: &[f64; 7], phase: crate::recovery::Phase) -> Result<()> {
        self.arm.recovery_move(target, phase)
    }
    fn stream(&mut self, sample: &Sample) -> Result<()> {
        let mut positions = [0.; 7];
        positions[..6].copy_from_slice(&sample.joints);
        positions[6] = self.arm.measured()?.carriage.unwrap().target_m.unwrap();
        self.pending = Some(JointSample {
            positions,
            velocities: [0.; 7],
            dt_s: sample.dt_s,
            pen: false,
        });
        Ok(())
    }
    fn stream_joint(&mut self, sample: &JointSample) -> Result<()> {
        self.pending = Some(sample.clone());
        Ok(())
    }
    fn hold(&mut self) -> Result<()> {
        self.pending = None;
        self.arm.hold()
    }
    fn finish_joint_stream(&mut self) -> Result<()> {
        Ok(())
    }
    fn clear_error(&mut self) -> Result<()> {
        self.arm.clear_error()
    }
    fn load_config(&mut self, path: &Path) -> Result<()> {
        self.arm.load_config(path)
    }
    fn inject_faults(&mut self, faults: Faults) -> Result<()> {
        self.arm.inject_faults(faults)
    }
}

impl OfflineArmBackend for WorldProbe {
    fn backend_id(&self) -> &'static str {
        "offline-probe"
    }
    fn advance_world(&mut self, period: Duration) -> Result<()> {
        assert_eq!(period, CONTROL_PERIOD);
        thread::sleep(self.delay);
        let step = self.steps.load(Ordering::Acquire) + 1;
        if self.fail_step == Some(step) {
            return Err(Error("probe physics transport lost".into()));
        }
        if let Some(sample) = self.pending.take() {
            self.arm.stream_joint(&sample)?;
        }
        self.steps.store(step, Ordering::Release);
        Ok(())
    }
}

fn probe() -> WorldProbe {
    let mut arm = MockArm::default();
    arm.connect(&ArmConfig { joints: 6 }).unwrap();
    arm.set_mode(Mode::Position).unwrap();
    WorldProbe {
        arm,
        pending: None,
        steps: Arc::new(AtomicU64::new(0)),
        delay: Duration::ZERO,
        fail_step: None,
        fail_read: None,
        fault_read: None,
    }
}

fn spawn(probe: WorldProbe) -> (Worker, Arc<OfflineClock>) {
    spawn_with_stop(probe, Arc::new(AtomicBool::new(false)))
}

fn spawn_with_stop(probe: WorldProbe, stop: Arc<AtomicBool>) -> (Worker, Arc<OfflineClock>) {
    let clock = OfflineClock::new(EPOCH).unwrap();
    let worker = Worker::spawn_offline_with_stop(
        move || {
            Ok(Control {
                backend: probe,
                estop: Arc::new(AtomicI32::new(estop::OK)),
                max_velocity: 1.0,
                max_tracking_error: crate::TRACKING_ERROR_LIMIT_RAD,
                max_contact: 1.0,
                envelope: 1.0,
            })
        },
        clock.clone(),
        MotionGuard::new(1.0, 9.0, 0.5, 0.5, 8),
        stop,
    )
    .unwrap();
    (worker, clock)
}

fn samples() -> Vec<JointSample> {
    (0..4)
        .map(|i| JointSample {
            positions: [i as f64 * 0.00001, 0., 0., 0., 0., 0., 0.002],
            velocities: [0.; 7],
            dt_s: 0.0025,
            pen: false,
        })
        .collect()
}

#[test]
fn final_world_step_reports_controller_fault_without_another_permit() {
    let mut world = probe();
    world.fault_read = Some(4);
    let (mut worker, clock) = spawn(world);
    let refs = samples();
    worker
        .submit(Primitive::StreamJoint {
            seed: refs[0].positions,
            expected_sequence: 0,
            permit: None,
            samples: refs,
        })
        .unwrap();
    clock
        .advance(Duration::from_millis(10), WALL_LIMIT)
        .unwrap();
    let status = worker.status();
    assert!(status.completed);
    assert!(
        status
            .fault
            .as_deref()
            .unwrap()
            .contains("after offline step")
    );
    assert_eq!(status.measured.unwrap().mode, Mode::Fault);
    assert_eq!(clock.elapsed(), Duration::from_millis(10));
}

#[test]
fn offline_move_j_applies_its_last_worker_reference_before_completion() {
    let (mut worker, clock) = spawn(probe());
    clock.advance(CONTROL_PERIOD, WALL_LIMIT).unwrap();
    worker
        .submit(Primitive::MoveJ {
            target: vec![0.0001, 0., 0., 0., 0., 0.],
            goal_time_s: 0.01,
        })
        .unwrap();
    clock
        .advance(Duration::from_millis(10), WALL_LIMIT)
        .unwrap();
    let status = worker.status();
    assert!(
        status.completed && status.fault.is_none(),
        "{:?}",
        status.fault
    );
    assert_eq!(
        status.measured.unwrap().joints,
        [0.0001, 0., 0., 0., 0., 0.]
    );
    assert_eq!(clock.elapsed(), Duration::from_micros(12_500));
}

#[test]
fn slow_world_publishes_each_command_at_exact_simulation_ticks() {
    let mut world = probe();
    world.delay = Duration::from_millis(8); // slower than the live 2.5 ms budget
    let steps = world.steps.clone();
    let (mut worker, clock) = spawn(world);
    let refs = samples();
    let sequence = worker
        .submit(Primitive::StreamJoint {
            seed: refs[0].positions,
            expected_sequence: 0,
            permit: None,
            samples: refs.clone(),
        })
        .unwrap();
    thread::sleep(Duration::from_millis(20));
    assert_eq!(
        steps.load(Ordering::Acquire),
        0,
        "host time cannot run a queued stream"
    );
    for (i, sample) in refs.iter().enumerate() {
        clock.advance(CONTROL_PERIOD, WALL_LIMIT).unwrap();
        let status = worker.status();
        assert_eq!(status.sequence, sequence);
        assert_eq!(status.stream_progress.unwrap().submitted, i + 1);
        assert_eq!(status.completed, i + 1 == refs.len());
        assert_eq!(status.measured.unwrap().joints, sample.positions[..6]);
        assert_eq!(
            status.measured_simulation_ns,
            Some(EPOCH + (i as u64 + 1) * 2_500_000)
        );
        assert_eq!(status.late_ticks, 0);
        assert!(status.fault.is_none(), "{:?}", status.fault);
        assert!(status.measured_wall_ns >= status.measured_started_wall_ns);
        assert_ne!(Some(status.measured_wall_ns), status.measured_simulation_ns);
    }
    let old = worker.status();
    thread::sleep(Duration::from_millis(20));
    assert!(old.measurement_is_fresh(Duration::from_millis(1)));
    assert_eq!(steps.load(Ordering::Acquire), 4);
    clock
        .advance(Duration::from_micros(2501), WALL_LIMIT)
        .unwrap();
    assert_eq!(
        steps.load(Ordering::Acquire),
        6,
        "round upward to whole ticks"
    );
    assert!(
        !old.measurement_is_fresh(Duration::from_millis(1)),
        "age follows world time"
    );
    assert!(worker.status().measurement_is_fresh(Duration::ZERO));
    drop(worker);
    assert!(
        !old.measurement_is_fresh(Duration::from_secs(100)),
        "closed worlds cannot grant freshness"
    );
    assert!(clock.advance(CONTROL_PERIOD, WALL_LIMIT).is_err());
}

#[test]
fn physics_and_feedback_failures_never_acknowledge_success() {
    for failure in 0..3 {
        let mut world = probe();
        match failure {
            0 => world.fail_step = Some(1),
            1 => world.fail_read = Some(0),
            _ => world.fail_read = Some(1),
        }
        let (worker, clock) = spawn(world);
        let error = clock.advance(CONTROL_PERIOD, WALL_LIMIT).unwrap_err();
        assert!(error.to_string().contains("transport lost"), "{error}");
        // A failed read after stepping retains elapsed physics time but does
        // not publish an acknowledged/fresh observation of that world state.
        assert_eq!(
            clock.elapsed(),
            if failure == 2 {
                CONTROL_PERIOD
            } else {
                Duration::ZERO
            }
        );
        let status = worker.status();
        assert!(status.completed && !status.watchdog_ok && status.fault.is_some());
        assert!(!status.measurement_is_fresh(Duration::from_secs(1)));
        assert!(clock.advance(CONTROL_PERIOD, WALL_LIMIT).is_err());
    }
}

#[test]
fn paused_software_stop_discards_the_stream_tail_without_free_running() {
    let world = probe();
    let steps = world.steps.clone();
    let stop = Arc::new(AtomicBool::new(false));
    let (mut worker, clock) = spawn_with_stop(world, stop.clone());
    let refs = samples();
    worker
        .submit(Primitive::StreamJoint {
            seed: refs[0].positions,
            expected_sequence: 0,
            permit: None,
            samples: refs,
        })
        .unwrap();
    clock.advance(CONTROL_PERIOD, WALL_LIMIT).unwrap();
    stop.store(true, Ordering::Release);
    let limit = Instant::now() + WALL_LIMIT;
    loop {
        let status = worker.status();
        if status.completed && status.fault.as_deref() == Some("e-stop") {
            break;
        }
        assert!(
            Instant::now() < limit,
            "paused worker failed to handle stop"
        );
        thread::sleep(Duration::from_millis(1));
    }
    assert_eq!(steps.load(Ordering::Acquire), 2);
    thread::sleep(Duration::from_millis(20));
    assert_eq!(
        steps.load(Ordering::Acquire),
        2,
        "latched stop must not run virtual time"
    );
    stop.store(false, Ordering::Release);
    clock
        .advance(Duration::from_millis(10), WALL_LIMIT)
        .unwrap();
    assert_eq!(
        worker.status().commands,
        1,
        "release cannot resume the discarded tail"
    );
    assert!(worker.status().fault.is_some());
}

#[test]
fn offline_clock_rejects_live_scheduling_and_duplicate_world_owners() {
    let (mut worker, clock) = spawn(probe());
    let duplicate = Worker::spawn_offline_with::<WorldProbe, _>(
        || panic!("second world constructed"),
        clock.clone(),
        MotionGuard::new(1.0, 9.0, 0.5, 0.5, 8),
    );
    assert!(duplicate.err().unwrap().to_string().contains("world owner"));
    let refs = samples();
    let permit = StreamPermit::scheduled_stop_guard(
        refs.len(),
        Some(Instant::now() + Duration::from_secs(2)),
        |_, _| Ok(false),
    )
    .unwrap();
    let error = worker
        .submit(Primitive::StreamJoint {
            seed: refs[0].positions,
            expected_sequence: 0,
            permit: Some(permit),
            samples: refs,
        })
        .unwrap_err();
    assert!(error.to_string().contains("live scheduled"));
    assert_eq!(clock.elapsed(), Duration::ZERO);
}

#[test]
fn failed_construction_closes_clock_without_advancing() {
    let clock = OfflineClock::new(EPOCH).unwrap();
    let worker = Worker::spawn_offline_with::<WorldProbe, _>(
        || Err(Error("fixture setup refused".into())),
        clock.clone(),
        MotionGuard::new(1.0, 9.0, 0.5, 0.5, 8),
    )
    .unwrap();
    let error = clock.advance(CONTROL_PERIOD, WALL_LIMIT).unwrap_err();
    assert!(error.to_string().contains("fixture setup refused"));
    assert!(worker.status().completed);
    assert_eq!(clock.elapsed(), Duration::ZERO);
}

#[test]
fn absent_worker_has_a_wall_timeout_without_advancing_simulated_time() {
    let clock = OfflineClock::new(EPOCH).unwrap();
    let start = Instant::now();
    let error = clock
        .advance(CONTROL_PERIOD, Duration::from_millis(15))
        .unwrap_err();
    assert!(error.to_string().contains("timed out"));
    assert!(start.elapsed() >= Duration::from_millis(15));
    assert_eq!(clock.elapsed(), Duration::ZERO);
    assert!(!clock.running());
}

#[test]
fn contact_trip_keeps_debounce_and_retract_deadline_in_controller_time() {
    for stuck in [false, true] {
        let mut world = probe();
        world.arm.faults.carriage_deflection_m = Some(0.003);
        world.arm.faults.carriage_stuck = stuck;
        let (worker, clock) = spawn(world);
        clock.advance(CONTROL_PERIOD * 39, WALL_LIMIT).unwrap();
        assert!(worker.status().fault.is_none());
        thread::sleep(Duration::from_millis(15));
        assert!(
            worker.status().fault.is_none(),
            "host wait is not another cap tick"
        );
        clock.advance(CONTROL_PERIOD, WALL_LIMIT).unwrap();
        let tripped = worker.status();
        assert!(!tripped.completed);
        let reason = tripped.fault.unwrap();
        assert!(reason.contains("carriage_contact_cap"));
        assert!(reason.contains("cap_n=20") && reason.contains("limit_m=0.002"));
        if stuck {
            clock
                .advance(Duration::from_millis(800), WALL_LIMIT)
                .unwrap();
            assert!(!worker.status().completed);
            clock.advance(CONTROL_PERIOD, WALL_LIMIT).unwrap();
            let status = worker.status();
            assert!(status.completed && status.retract_verified == Some(false));
            assert!(status.fault.unwrap().contains("PEN NOT RETRACTED"));
        } else {
            clock.advance(CONTROL_PERIOD, WALL_LIMIT).unwrap();
            let status = worker.status();
            assert!(status.completed && status.retract_verified == Some(true));
            assert_eq!(status.measured.unwrap().carriage.unwrap().position_m, 0.032);
        }
    }
}

#[test]
fn recovery_takeover_waits_for_its_full_simulated_interval() {
    let (mut worker, clock) = spawn(probe());
    worker
        .submit(Primitive::PrepareRecovery {
            staged: [0.0, 0.2, -0.3, 0.0, 0.0, 0.5, 0.002],
        })
        .unwrap();
    clock.advance(CONTROL_PERIOD, WALL_LIMIT).unwrap();
    assert!(
        worker.status().fault.is_none(),
        "{:?}",
        worker.status().fault
    );
    let sequence = worker.submit(Primitive::Takeover).unwrap();
    let duration = Duration::from_secs_f64(crate::recovery::TAKEOVER_S);
    clock
        .advance(duration - CONTROL_PERIOD, WALL_LIMIT)
        .unwrap();
    let status = worker.status();
    assert_eq!(status.sequence, sequence);
    assert!(!status.completed && status.fault.is_none(), "{status:?}");
    clock.advance(CONTROL_PERIOD, WALL_LIMIT).unwrap();
    let status = worker.status();
    assert!(status.completed && status.fault.is_none(), "{status:?}");
    assert_eq!(clock.elapsed(), duration + CONTROL_PERIOD);
}

#[test]
fn timed_out_in_flight_step_cannot_start_another_or_grant_freshness() {
    let mut world = probe();
    world.delay = Duration::from_millis(40);
    let steps = world.steps.clone();
    let (worker, clock) = spawn(world);
    let error = clock
        .advance(Duration::from_millis(100), Duration::from_millis(15))
        .unwrap_err();
    assert!(error.to_string().contains("timed out"));
    // Destruction joins the already-started bounded backend operation. Closing
    // a clock does not claim to cancel physics that is already in flight.
    let snapshot = worker.status();
    drop(worker);
    assert!(steps.load(Ordering::Acquire) <= 1);
    assert!(!snapshot.measurement_is_fresh(Duration::from_secs(100)));
    assert!(clock.advance(CONTROL_PERIOD, WALL_LIMIT).is_err());
}

#[test]
fn stop_supersedes_a_latched_trip_retract_while_paused() {
    let mut world = probe();
    world.arm.faults.carriage_deflection_m = Some(0.003);
    world.arm.faults.carriage_stuck = true;
    let (worker, clock) = spawn(world);
    clock.advance(CONTROL_PERIOD * 40, WALL_LIMIT).unwrap();
    assert!(!worker.status().completed);
    assert!(
        worker
            .status()
            .fault
            .unwrap()
            .contains("carriage_contact_cap")
    );
    worker.software_stop.store(true, Ordering::Release);
    let limit = Instant::now() + WALL_LIMIT;
    loop {
        let status = worker.status();
        if status.fault.as_deref() == Some("e-stop") {
            assert!(status.completed);
            assert_eq!(status.retract_verified, None);
            break;
        }
        assert!(Instant::now() < limit);
        thread::sleep(Duration::from_millis(1));
    }
    assert_eq!(clock.elapsed(), CONTROL_PERIOD * 41);
    clock.advance(Duration::from_secs(1), WALL_LIMIT).unwrap();
    assert_eq!(worker.status().fault.as_deref(), Some("e-stop"));
    assert_eq!(worker.status().retract_verified, None);
}
