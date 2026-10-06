//! Hand-guiding lifecycle on the mock backend: exact command traces, mode
//! readback, the tool's and carriage's trip responses and the parked-effort
//! regression.
//! Command and state-machine evidence only; no controller behaviour is claimed.
use super::*;
use crate::{ArmConfig, ContactPolicy, Faults, MockArm, Mode, estop};
use std::sync::atomic::AtomicI32;

struct Rig {
    worker: Worker,
    trace: Arc<Mutex<Vec<String>>>,
    stop: Arc<AtomicI32>,
    telemetry: Option<mpsc::Receiver<TelemetrySample>>,
}

fn spawn(configure: impl FnOnce(&mut MockArm) + Send + 'static, recording: bool) -> Rig {
    let stop = Arc::new(AtomicI32::new(estop::OK));
    let estop_atomic = stop.clone();
    let trace = Arc::new(Mutex::new(Vec::new()));
    let shared = trace.clone();
    let construct = move || {
        let mut arm = MockArm::default();
        configure(&mut arm);
        arm.trace = shared;
        Ok(Control {
            backend: arm,
            estop: estop_atomic,
            max_velocity: 1.0,
            max_tracking_error: crate::TRACKING_ERROR_LIMIT_RAD,
            max_contact: 1.0,
            envelope: 3.5,
        })
    };
    let guard = MotionGuard::new(1.0, crate::profile::OVERFORCE_NM, 0.5, 0.5, 8);
    let (worker, telemetry) = if recording {
        let (sender, receiver) = mpsc::sync_channel(4096);
        (
            Worker::spawn_recording(construct, guard, sender).unwrap(),
            Some(receiver),
        )
    } else {
        (
            Worker::spawn_with(construct, Duration::from_micros(2500), guard).unwrap(),
            None,
        )
    };
    Rig {
        worker,
        trace,
        stop,
        telemetry,
    }
}

fn wait_for(worker: &Worker, sequence: u64) -> Status {
    let deadline = Instant::now() + Duration::from_secs(3);
    loop {
        let status = worker.status();
        if status.sequence == sequence && status.completed {
            return status;
        }
        assert!(
            Instant::now() < deadline,
            "sequence {sequence} never completed"
        );
        thread::sleep(Duration::from_millis(2));
    }
}

fn wait_until(worker: &Worker, predicate: impl Fn(&Status) -> bool, what: &str) -> Status {
    let deadline = Instant::now() + Duration::from_secs(4);
    loop {
        let status = worker.status();
        if predicate(&status) {
            return status;
        }
        assert!(Instant::now() < deadline, "{what}: {:?}", status.fault);
        thread::sleep(Duration::from_millis(2));
    }
}

/// Connect, re-seed and rest until the carriage contact baseline arms.
fn prepare(rig: &mut Rig) {
    let sequence = rig
        .worker
        .submit(Primitive::Reconnect(ArmConfig { joints: 6 }))
        .unwrap();
    assert!(wait_for(&rig.worker, sequence).fault.is_none());
    let sequence = rig.worker.submit(Primitive::Reseed).unwrap();
    assert!(wait_for(&rig.worker, sequence).fault.is_none());
    wait_until(
        &rig.worker,
        |s| s.contact.is_some_and(|c| c.armed),
        "contact baseline armed at rest",
    );
}

fn trace(rig: &Rig) -> Vec<String> {
    rig.trace.lock().unwrap().clone()
}

fn enter_guiding(rig: &mut Rig, carriage_m: f64) -> Status {
    let sequence = rig
        .worker
        .submit(Primitive::HandGuide { carriage_m })
        .unwrap();
    wait_for(&rig.worker, sequence)
}

#[test]
fn hand_guiding_enters_records_operator_motion_and_stops_with_an_exact_trace() {
    let mut rig = spawn(|_| {}, true);
    prepare(&mut rig);
    let status = enter_guiding(&mut rig, 0.002);
    assert!(status.fault.is_none(), "{:?}", status.fault);
    assert!(status.guiding);
    let receipt = status.guide.clone().unwrap();
    assert_eq!(receipt.carriage_datum_m, 0.002);
    assert_eq!(receipt.entered_measured.mode, Mode::HandGuiding);
    assert!(receipt.stopped_wall_ns.is_none());
    // A scripted operator moves the arm; the worker commands nothing.
    let sequence = rig
        .worker
        .submit(Primitive::Inject(Faults {
            guide_pose: Some([0.3, -0.2, 0.1, 0.0, 0.2, -0.1]),
            ..Default::default()
        }))
        .unwrap();
    assert!(wait_for(&rig.worker, sequence).fault.is_none());
    let moved = wait_until(
        &rig.worker,
        |s| {
            s.measured
                .as_ref()
                .is_some_and(|m| (m.joints[0] - 0.3).abs() < 1e-6)
        },
        "operator motion reached the scripted pose",
    );
    assert_eq!(moved.measured.as_ref().unwrap().mode, Mode::HandGuiding);
    assert!(moved.fault.is_none());
    let sequence = rig.worker.submit(Primitive::GuideStop).unwrap();
    let status = wait_for(&rig.worker, sequence);
    assert!(status.fault.is_none(), "{:?}", status.fault);
    assert!(!status.guiding);
    let receipt = status.guide.unwrap();
    assert!(receipt.stopped_wall_ns.is_some());
    assert_eq!(receipt.stop_measured.as_ref().unwrap().mode, Mode::Position);
    let submitted = receipt.stop_submitted.unwrap();
    assert!((submitted[0] - 0.3).abs() < 1e-6 && submitted[6] == 0.002);
    // The exact command sequence: the re-seed's own position hold, then one
    // mixed-mode entry, then one ordered effort-to-position stop. No stream,
    // move, retract or idle was ever sent to the arm.
    assert_eq!(
        trace(&rig),
        vec![
            "set_mode(Position)",
            "hold",
            "enter_hand_guiding(0.002)",
            "hold_from_hand_guiding(set_mode(Position), positions(measured))",
        ]
    );
    // Retained telemetry carries the guiding flag and the mixed mode code.
    let samples: Vec<_> = rig.telemetry.take().unwrap().try_iter().collect();
    assert!(
        samples
            .iter()
            .any(|s| s.guiding && s.measured.mode == Mode::HandGuiding)
    );
    assert!(
        samples
            .iter()
            .any(|s| !s.guiding && s.measured.mode == Mode::Position)
    );
    assert!(samples.windows(2).all(|pair| pair[1].tick > pair[0].tick));
    assert_eq!(rig.worker.status().telemetry_dropped, 0);
    // Supported release happens from the position hold, never from guiding.
    let sequence = rig.worker.submit(Primitive::Idle).unwrap();
    let status = wait_for(&rig.worker, sequence);
    assert!(status.fault.is_none(), "{:?}", status.fault);
    assert_eq!(status.measured.unwrap().mode, Mode::Idle);
}

#[test]
fn entry_is_refused_before_the_carriage_baseline_arms_or_at_a_wrong_datum() {
    let mut rig = spawn(|_| {}, false);
    let sequence = rig
        .worker
        .submit(Primitive::Reconnect(ArmConfig { joints: 6 }))
        .unwrap();
    wait_for(&rig.worker, sequence);
    let sequence = rig.worker.submit(Primitive::Reseed).unwrap();
    wait_for(&rig.worker, sequence);
    let status = enter_guiding(&mut rig, 0.002);
    assert!(
        status
            .fault
            .as_deref()
            .unwrap()
            .contains("baseline not armed")
    );
    assert!(!status.guiding);
    assert!(
        !trace(&rig)
            .iter()
            .any(|c| c.starts_with("enter_hand_guiding"))
    );
    // Refusal latched a measured hold; recovery is the existing re-seed.
    let sequence = rig.worker.submit(Primitive::Reseed).unwrap();
    assert!(wait_for(&rig.worker, sequence).fault.is_none());
    wait_until(&rig.worker, |s| s.contact.is_some_and(|c| c.armed), "armed");
    let status = enter_guiding(&mut rig, 0.010);
    assert!(status.fault.as_deref().unwrap().contains("carriage datum"));
    assert!(
        !trace(&rig)
            .iter()
            .any(|c| c.starts_with("enter_hand_guiding"))
    );
}

#[test]
fn a_nonzero_start_pose_keeps_its_carriage_datum_and_never_resets_it() {
    let mut rig = spawn(
        |arm| {
            *arm = MockArm::seeded([0.4, -0.3, 0.2, 0.1, -0.2, 0.9, 0.010], 3.0).unwrap();
        },
        false,
    );
    prepare(&mut rig);
    let status = enter_guiding(&mut rig, 0.010);
    assert!(status.fault.is_none(), "{:?}", status.fault);
    let measured = status.measured.unwrap();
    assert_eq!(measured.carriage.unwrap().target_m, Some(0.010));
    assert!((measured.joints[0] - 0.4).abs() < 1e-9);
    assert!(
        !trace(&rig)
            .iter()
            .any(|c| c.starts_with("retract") || c.starts_with("set_mode(Idle"))
    );
    assert!(trace(&rig).contains(&"enter_hand_guiding(0.01)".to_string()));
}

#[test]
fn mode_readback_failure_on_entry_latches_a_measured_hold() {
    let mut rig = spawn(
        |arm| {
            arm.faults.guide_entry_refused = true;
        },
        false,
    );
    prepare(&mut rig);
    let status = enter_guiding(&mut rig, 0.002);
    let fault = status.fault.unwrap();
    assert!(fault.contains("not confirmed by readback"), "{fault}");
    assert!(!status.guiding);
    let last = trace(&rig);
    assert_eq!(last[last.len() - 2], "enter_hand_guiding(0.002)");
    assert_eq!(last[last.len() - 1], "hold");
    // Latched: nothing else runs until a measured re-seed.
    let sequence = rig.worker.submit(Primitive::GuideStop).unwrap();
    let fault = wait_for(&rig.worker, sequence).fault.unwrap();
    assert!(fault.contains("re-seed"), "{fault}");
}

#[test]
fn mode_readback_failure_on_stop_reports_unknown_state_and_refuses_release() {
    let mut rig = spawn(
        |arm| {
            arm.faults.guide_stop_refused = true;
        },
        false,
    );
    prepare(&mut rig);
    assert!(enter_guiding(&mut rig, 0.002).fault.is_none());
    let sequence = rig.worker.submit(Primitive::GuideStop).unwrap();
    let status = wait_for(&rig.worker, sequence);
    let fault = status.fault.clone().unwrap();
    assert!(fault.contains("position mode not confirmed"), "{fault}");
    assert_eq!(
        status
            .guide
            .unwrap()
            .stop_error
            .as_deref()
            .map(|e| e.contains("not confirmed")),
        Some(true)
    );
    // The owner stays alive; the rotary joints may still be in effort mode,
    // so idle is refused (with another ordered freeze attempt), never sent.
    assert!(status.guiding, "controller still reports the mixed mode");
    let sequence = rig.worker.submit(Primitive::Idle).unwrap();
    let fault = wait_for(&rig.worker, sequence).fault.unwrap();
    assert!(fault.contains("refused while hand guiding"), "{fault}");
    assert!(!trace(&rig).iter().any(|c| c == "set_mode(Idle)"));
}

#[test]
fn physical_stop_during_guiding_freezes_through_the_ordered_transition() {
    let mut rig = spawn(|_| {}, false);
    prepare(&mut rig);
    assert!(enter_guiding(&mut rig, 0.002).fault.is_none());
    rig.stop.store(estop::FAULT, Ordering::SeqCst);
    let status = wait_until(&rig.worker, |s| s.fault.is_some(), "e-stop latched");
    assert_eq!(status.fault.as_deref(), Some("e-stop"));
    assert!(!status.guiding);
    assert_eq!(
        trace(&rig).last().unwrap(),
        "hold_from_hand_guiding(set_mode(Position), positions(measured))"
    );
    // Neither release nor a fresh entry is possible while STOP is latched.
    let stopped_trace = trace(&rig);
    let sequence = rig
        .worker
        .submit(Primitive::HandGuide { carriage_m: 0.002 })
        .unwrap();
    let status = wait_for(&rig.worker, sequence);
    assert!(status.fault.is_some());
    assert!(!status.guiding);
    assert!(
        trace(&rig)[stopped_trace.len()..]
            .iter()
            .all(|command| command == "hold"),
        "STOP may repeat the protective hold but must not re-enter guiding"
    );
    // A live STOP can overwrite the diagnostic with "e-stop" on any tick.
    // Once it clears, the separate recovery latch must still refuse entry.
    rig.stop.store(estop::OK, Ordering::SeqCst);
    let sequence = rig
        .worker
        .submit(Primitive::HandGuide { carriage_m: 0.002 })
        .unwrap();
    let fault = wait_for(&rig.worker, sequence).fault.unwrap();
    assert!(fault.contains("re-seed"), "{fault}");
    assert!(
        trace(&rig)[stopped_trace.len()..]
            .iter()
            .all(|command| command == "hold"),
        "release must not resume guiding"
    );
    assert!(!trace(&rig).iter().any(|c| c == "set_mode(Idle)"));
}

#[test]
fn motion_recovery_and_release_primitives_are_refused_while_guiding() {
    for primitive in [
        Primitive::Idle,
        Primitive::MoveJ {
            target: vec![0.0; 6],
            goal_time_s: 1.0,
        },
        Primitive::Stream(vec![Sample {
            joints: vec![0.0; 6],
            dt_s: 0.0025,
            contact: 0.0,
        }]),
        Primitive::Reseed,
        Primitive::PrepareRecovery { staged: [0.0; 7] },
        Primitive::ClearError,
    ] {
        let mut rig = spawn(|_| {}, false);
        prepare(&mut rig);
        assert!(enter_guiding(&mut rig, 0.002).fault.is_none());
        let sequence = rig.worker.submit(primitive).unwrap();
        let status = wait_for(&rig.worker, sequence);
        assert!(status.fault.is_some(), "refused");
        let commands = trace(&rig);
        assert!(!commands.iter().any(|c| c == "set_mode(Idle)"));
        assert_eq!(
            commands.iter().filter(|c| *c == "stream").count(),
            0,
            "measured re-seed never streams a motion target"
        );
        assert!(!commands.iter().any(|c| c.starts_with("retract")));
    }
}

#[test]
fn unchanged_feedback_faults_within_the_conservative_screen() {
    let mut rig = spawn(|_| {}, false);
    prepare(&mut rig);
    assert!(enter_guiding(&mut rig, 0.002).fault.is_none());
    let sequence = rig
        .worker
        .submit(Primitive::Inject(Faults {
            frozen_feedback: true,
            ..Default::default()
        }))
        .unwrap();
    wait_for(&rig.worker, sequence);
    let started = Instant::now();
    let status = wait_until(
        &rig.worker,
        |s| s.fault.is_some(),
        "frozen feedback refused",
    );
    assert!(status.fault.unwrap().contains("feedback unchanged"));
    assert!(started.elapsed() < Duration::from_millis(600));
    wait_until(
        &rig.worker,
        |s| !s.guiding,
        "ordered freeze reported by the controller",
    );
}

/// The trip response follows the tool's policy and the carriage's
/// qualification, not the arm: a contact tool on a qualified carriage trips
/// on effort and retracts; a standoff tool, or an unqualified carriage,
/// never counts effort as contact while guiding and never retracts.
#[test]
fn carriage_trip_while_guiding_keeps_each_tools_response_on_its_carriage() {
    for (policy, qualified) in [
        (ContactPolicy::Contact, true),
        (ContactPolicy::Standoff, true),
        (ContactPolicy::Contact, false),
    ] {
        let mut rig = spawn(
            move |arm| {
                arm.contact_policy = policy;
                arm.carriage_qualified = qualified;
            },
            false,
        );
        prepare(&mut rig);
        assert!(enter_guiding(&mut rig, 0.002).fault.is_none());
        let sequence = rig
            .worker
            .submit(Primitive::Inject(Faults {
                carriage_effort_n: Some(25.0),
                ..Default::default()
            }))
            .unwrap();
        wait_for(&rig.worker, sequence);
        if policy == ContactPolicy::Standoff || !qualified {
            // The effort is not a contact signal while guiding: it never
            // trips, guiding continues, no retract is ever commanded. (The
            // deflection screen still is; see the worker trip tests.)
            thread::sleep(Duration::from_millis(200));
            let status = rig.worker.status();
            assert!(
                status.fault.is_none() && status.guiding,
                "{:?}",
                status.fault
            );
            assert!(
                !trace(&rig)
                    .iter()
                    .any(|c| c.starts_with("retract_carriage"))
            );
            continue;
        }
        let status = wait_until(&rig.worker, |s| s.fault.is_some(), "contact cap tripped");
        assert!(
            status
                .fault
                .as_deref()
                .unwrap()
                .contains("carriage_contact_cap")
        );
        let commands = trace(&rig);
        let stop = commands
            .iter()
            .position(|c| c == "hold_from_hand_guiding(set_mode(Position), positions(measured))")
            .expect("ordered stop before any retract");
        let retracted = commands.iter().position(|c| c == "retract_carriage(0.032)");
        assert!(retracted.is_some_and(|r| r > stop));
        assert_eq!(
            wait_until(&rig.worker, |s| s.retract_verified.is_some(), "retract").retract_verified,
            Some(true)
        );
    }
}

/// The withdrawn teacher read a raw -25 N parked carriage effort as contact.
/// The carried protection forms its baseline at rest first: a constant offset
/// is not contact, while a real change after arming still trips.
#[test]
fn a_raw_parked_carriage_effort_is_not_contact_but_a_change_after_arming_is() {
    let mut rig = spawn(
        |arm| {
            arm.faults.carriage_effort_n = Some(-24.996);
        },
        false,
    );
    prepare(&mut rig);
    let status = enter_guiding(&mut rig, 0.002);
    assert!(status.fault.is_none(), "{:?}", status.fault);
    let evaluation = status.contact.unwrap();
    assert!(evaluation.armed && evaluation.contact_n.abs() < 1.0);
    assert!((evaluation.baseline_n.unwrap() + 24.996).abs() < 1e-9);
    thread::sleep(Duration::from_millis(300));
    assert!(rig.worker.status().fault.is_none());
    // Now the effort actually changes by more than the cap: that is contact.
    let sequence = rig
        .worker
        .submit(Primitive::Inject(Faults {
            carriage_effort_n: Some(-24.996 + 21.0),
            ..Default::default()
        }))
        .unwrap();
    wait_for(&rig.worker, sequence);
    let status = wait_until(&rig.worker, |s| s.fault.is_some(), "debounced trip");
    assert!(status.fault.unwrap().contains("carriage_contact_cap"));
}

/// An unqualified carriage carries the pen's weight and its clamp preload:
/// about -110 N at rest, swinging 20 N with posture in free space. That effort
/// is never contact, streamed or guided; its deflection screen is the
/// carriage's interlock and its trip holds without a retract.
#[test]
fn an_unqualified_carriage_effort_swing_is_not_contact_in_native_motion_but_deflection_is() {
    let mut rig = spawn(
        |arm| {
            arm.contact_policy = ContactPolicy::Contact;
            arm.carriage_qualified = false;
            arm.faults.carriage_effort_n = Some(-110.0);
        },
        false,
    );
    prepare(&mut rig);
    let armed = rig.worker.status().contact.unwrap();
    assert!(armed.armed && armed.assessable && !armed.effort_assessable);
    assert!((armed.baseline_n.unwrap() + 110.0).abs() < 1e-9);
    let sequence = rig
        .worker
        .submit(Primitive::Inject(Faults {
            carriage_effort_n: Some(-85.0),
            ..Default::default()
        }))
        .unwrap();
    wait_for(&rig.worker, sequence);
    thread::sleep(Duration::from_millis(300));
    let status = rig.worker.status();
    assert!(status.fault.is_none(), "{:?}", status.fault);
    let evaluation = status.contact.unwrap();
    assert!(evaluation.contact_n > 20.0 && !evaluation.effort_assessable && !evaluation.trip);
    assert!(
        !trace(&rig)
            .iter()
            .any(|c| c.starts_with("retract_carriage"))
    );
    let sequence = rig
        .worker
        .submit(Primitive::Inject(Faults {
            carriage_deflection_m: Some(0.003),
            ..Default::default()
        }))
        .unwrap();
    wait_for(&rig.worker, sequence);
    let status = wait_until(&rig.worker, |s| s.fault.is_some(), "deflection trip");
    assert!(
        status
            .fault
            .as_deref()
            .unwrap()
            .contains("carriage_contact_cap")
    );
    assert_eq!(status.retract_verified, None);
    assert!(
        !trace(&rig)
            .iter()
            .any(|c| c.starts_with("retract_carriage"))
    );
}

#[test]
fn hand_guiding_is_unavailable_to_backends_without_it() {
    struct NoGuide(MockArm);
    impl ArmBackend for NoGuide {
        fn connect(&mut self, cfg: &ArmConfig) -> Result<()> {
            self.0.connect(cfg)
        }
        fn measured(&mut self) -> Result<Measured> {
            self.0.measured()
        }
        fn retract_carriage(&mut self, t: f64, d: f64) -> Result<()> {
            self.0.retract_carriage(t, d)
        }
        fn set_mode(&mut self, m: Mode) -> Result<()> {
            self.0.set_mode(m)
        }
        fn move_j(&mut self, t: &[f64], d: f64) -> Result<()> {
            self.0.move_j(t, d)
        }
        fn stream(&mut self, s: &Sample) -> Result<()> {
            self.0.stream(s)
        }
        fn hold(&mut self) -> Result<()> {
            self.0.hold()
        }
        fn clear_error(&mut self) -> Result<()> {
            self.0.clear_error()
        }
        fn load_config(&mut self, p: &std::path::Path) -> Result<()> {
            self.0.load_config(p)
        }
    }
    let stop = Arc::new(AtomicI32::new(estop::OK));
    let mut worker = Worker::spawn_with(
        {
            let stop = stop.clone();
            move || {
                Ok(Control {
                    backend: NoGuide(MockArm::default()),
                    estop: stop,
                    max_velocity: 1.0,
                    max_tracking_error: crate::TRACKING_ERROR_LIMIT_RAD,
                    max_contact: 1.0,
                    envelope: 3.5,
                })
            }
        },
        Duration::from_micros(2500),
        MotionGuard::new(1.0, crate::profile::OVERFORCE_NM, 0.5, 0.5, 8),
    )
    .unwrap();
    let sequence = worker
        .submit(Primitive::Reconnect(ArmConfig { joints: 6 }))
        .unwrap();
    wait_for(&worker, sequence);
    let sequence = worker.submit(Primitive::Reseed).unwrap();
    wait_for(&worker, sequence);
    wait_until(&worker, |s| s.contact.is_some_and(|c| c.armed), "armed");
    let sequence = worker
        .submit(Primitive::HandGuide { carriage_m: 0.002 })
        .unwrap();
    assert!(
        wait_for(&worker, sequence)
            .fault
            .unwrap()
            .contains("unsupported")
    );
}

/// A guard trip while guiding is recoverable for an attended capture: the
/// measured re-seed unlatches, the carriage returns to its datum through the
/// timed carriage move, and hand guiding re-enters at that datum. The move is
/// refused while guiding and while latched.
#[test]
fn a_tripped_capture_recovers_reseeds_returns_the_carriage_and_guides_again() {
    let mut rig = spawn(
        |arm| {
            arm.contact_policy = ContactPolicy::Contact;
            arm.carriage_qualified = true;
        },
        false,
    );
    prepare(&mut rig);
    assert!(enter_guiding(&mut rig, 0.002).fault.is_none());
    let refused = rig
        .worker
        .submit(Primitive::CarriageTo {
            target_m: 0.002,
            goal_time_s: 1.0,
        })
        .unwrap();
    let status = wait_for(&rig.worker, refused);
    assert!(
        status
            .fault
            .as_deref()
            .unwrap()
            .contains("primitive refused while hand guiding")
    );
    // The refusal froze the arm; a re-seed clears the latch.
    let sequence = rig.worker.submit(Primitive::Reseed).unwrap();
    assert!(wait_for(&rig.worker, sequence).fault.is_none());
    wait_until(
        &rig.worker,
        |s| s.contact.is_some_and(|c| c.armed),
        "contact baseline armed",
    );
    assert!(enter_guiding(&mut rig, 0.002).fault.is_none());
    let sequence = rig
        .worker
        .submit(Primitive::Inject(Faults {
            carriage_effort_n: Some(25.0),
            ..Default::default()
        }))
        .unwrap();
    wait_for(&rig.worker, sequence);
    wait_until(&rig.worker, |s| s.fault.is_some(), "contact cap tripped");
    let status = wait_until(&rig.worker, |s| s.retract_verified.is_some(), "retract");
    assert_eq!(status.retract_verified, Some(true));
    let blocked = rig
        .worker
        .submit(Primitive::CarriageTo {
            target_m: 0.002,
            goal_time_s: 1.0,
        })
        .unwrap();
    assert!(
        wait_for(&rig.worker, blocked)
            .fault
            .as_deref()
            .unwrap()
            .contains("re-seed required")
    );
    let sequence = rig
        .worker
        .submit(Primitive::Inject(Faults {
            carriage_effort_n: Some(0.0),
            ..Default::default()
        }))
        .unwrap();
    wait_for(&rig.worker, sequence);
    let sequence = rig.worker.submit(Primitive::Reseed).unwrap();
    let status = wait_for(&rig.worker, sequence);
    assert!(status.fault.is_none(), "{:?}", status.fault);
    assert!(!status.guiding);
    let sequence = rig
        .worker
        .submit(Primitive::CarriageTo {
            target_m: 0.002,
            goal_time_s: 1.0,
        })
        .unwrap();
    let status = wait_for(&rig.worker, sequence);
    assert!(status.fault.is_none(), "{:?}", status.fault);
    let carriage = status.measured.unwrap().carriage.unwrap();
    assert!((carriage.position_m - 0.002).abs() <= CARRIAGE_MOVE_TOLERANCE_M);
    assert!(trace(&rig).iter().any(|c| c == "move_carriage(0.002)"));
    let status = enter_guiding(&mut rig, 0.002);
    assert!(status.fault.is_none(), "{:?}", status.fault);
    assert!(status.guiding);
}

/// A stuck carriage never reports the move as complete: the readback check
/// latches after the goal time.
#[test]
fn an_unreached_carriage_target_latches_after_its_goal_time() {
    let mut rig = spawn(|arm| arm.faults.carriage_stuck = true, false);
    prepare(&mut rig);
    let sequence = rig
        .worker
        .submit(Primitive::CarriageTo {
            target_m: 0.030,
            goal_time_s: 1.0,
        })
        .unwrap();
    let status = wait_for(&rig.worker, sequence);
    assert!(
        status
            .fault
            .as_deref()
            .unwrap()
            .contains("carriage did not reach"),
        "{:?}",
        status.fault
    );
}

/// While guiding, the measured-velocity guard allows twice the profile limit:
/// a hand moving the arm at 1.5 rad/s under a 1 rad/s profile is not a trip,
/// 2.5 rad/s still is.
#[test]
fn the_velocity_guard_doubles_while_guiding_and_still_catches_a_fast_arm() {
    for (speed, trips) in [(1.5, false), (2.5, true)] {
        let mut rig = spawn(move |arm| arm.faults.guide_speed_rad_s = Some(speed), false);
        prepare(&mut rig);
        assert!(enter_guiding(&mut rig, 0.002).fault.is_none());
        let sequence = rig
            .worker
            .submit(Primitive::Inject(Faults {
                guide_pose: Some([0.5, 0.0, 0.0, 0.0, 0.0, 0.0]),
                guide_speed_rad_s: Some(speed),
                ..Default::default()
            }))
            .unwrap();
        wait_for(&rig.worker, sequence);
        thread::sleep(Duration::from_millis(150));
        let status = rig.worker.status();
        if trips {
            assert!(
                status
                    .fault
                    .as_deref()
                    .is_some_and(|f| f.contains("measured_velocity")),
                "{:?}",
                status.fault
            );
        } else {
            assert!(
                status.fault.is_none() && status.guiding,
                "{:?}",
                status.fault
            );
        }
    }
}
