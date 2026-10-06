//! Rigid-frame subscriber. This process contains no camera backend or arm driver.
use anyhow::{Result, ensure};
use clap::Parser;
use std::{
    collections::{BTreeMap, BTreeSet},
    path::PathBuf,
    time::{Duration, Instant},
};
use tatbot_bus::{Envelope, Producer, Stamp, service::ServiceLease, transport::Bus};
use tatbot_visiond::{
    CalibrationBundle, ReceivedFrameSet,
    fiducials::{
        AprilTagDetectorFactory, EstimatorConfig, FiducialInventory, RustEeTracker, WristLayout,
    },
};
use trackd::{evidence, merge_auxiliary};

#[derive(Parser)]
struct Args {
    #[arg(long, required = true)]
    connect: Vec<String>,
    #[arg(long)]
    /// Camera-owner sockets. The first is the cadence source; fresh auxiliary
    /// sets are merged by capture timestamp before one tracker update.
    socket: Vec<PathBuf>,
    #[arg(long)]
    node: String,
    #[arg(long)]
    inventory: PathBuf,
    #[arg(long)]
    calibration: PathBuf,
    #[arg(long)]
    wrist_layout: PathBuf,
    /// Inventory target; non-wrist entries must be measured rigid paper layouts.
    #[arg(long, default_value = "wrist")]
    target: String,
    #[arg(long)]
    output: PathBuf,
    #[arg(long, default_value_t = 0.3)]
    reacquire_scale: f64,
    /// Reuse recent measured and fresh multiview initializers while tracking.
    #[arg(long)]
    temporal_initializers: bool,
    /// Explicit quad-search trial setting; preserves full-resolution edge refinement.
    #[arg(long)]
    quad_decimate: Option<f64>,
    #[arg(long, default_value_t = 67_108_864)]
    evidence_bytes: u64,
    #[arg(long, default_value_t = 4)]
    evidence_files: usize,
    #[arg(long, default_value_t = 20)]
    full_scan_period: u64,
    #[arg(long, default_value_t = 80)]
    roi_margin_px: usize,
    #[arg(long, default_value_t = 250.0)]
    max_age_ms: f64,
    #[arg(long, default_value_t = 80.0)]
    auxiliary_sync_ms: f64,
    #[arg(long, default_value_t = 0)]
    max_sets: u64,
}

/// The physical arm whose end effector a wrist target rides; other targets
/// are tracked scene objects. `wrist` predates per-arm targets and keeps its
/// unsuffixed service name and topic.
fn tracked_arm(target: &str) -> Option<&'static str> {
    match target {
        "wrist" => Some("right"),
        "wrist_left" => Some("left"),
        _ => None,
    }
}

fn main() -> Result<()> {
    let args = Args::parse();
    ensure!(
        !args.target.is_empty()
            && args.target.len() <= 64
            && args
                .target
                .bytes()
                .all(|b| b.is_ascii_alphanumeric() || b == b'-' || b == b'_'),
        "invalid target name"
    );
    ensure!(
        args.full_scan_period > 0
            && args.max_age_ms.is_finite()
            && args.max_age_ms > 0.0
            && args.auxiliary_sync_ms.is_finite()
            && args.auxiliary_sync_ms > 0.0,
        "invalid tracker limits"
    );
    ensure!(
        !args.socket.is_empty()
            && args.socket.len() <= 4
            && args.socket.iter().collect::<BTreeSet<_>>().len() == args.socket.len(),
        "tracker needs one to four distinct owner sockets"
    );
    let _owner = tatbot_visiond::ownership::CameraLease::acquire(
        &std::env::temp_dir().join(format!("tatbot-trackd-{}.lock", args.target)),
    )?;
    let inventory = FiducialInventory::load(&args.inventory)?;
    let calibration = CalibrationBundle::load(&args.calibration)?;
    let layout = WristLayout::load_target(&args.wrist_layout, &inventory, &args.target, false)?;
    let detector =
        AprilTagDetectorFactory::new(&inventory, Some(&args.target), Some(args.reacquire_scale))?;
    let detector = if let Some(value) = args.quad_decimate {
        detector.with_quad_decimate(value)?
    } else {
        detector
    };
    let mut tracker =
        RustEeTracker::new(&calibration, &inventory, layout, EstimatorConfig::default())?;
    let bus = Bus::open(&args.connect, &[]).map_err(|e| anyhow::anyhow!("{e}"))?;
    let producer = Producer {
        node: args.node,
        pid: std::process::id(),
        sha: option_env!("TATBOT_SOURCE_COMMIT")
            .unwrap_or("development")
            .into(),
        run_id: std::env::var("TATBOT_RUN_ID")
            .unwrap_or_else(|_| format!("trackd-{}", std::process::id())),
    };
    let lease = ServiceLease::declare(
        &bus,
        producer.clone(),
        &match tracked_arm(&args.target) {
            Some("right") => "trackd".into(),
            Some(arm) => format!("trackd-{arm}"),
            None => format!("trackd-target-{}", args.target),
        },
        vec![if tracked_arm(&args.target).is_some() {
            "tatbot.tracking-pose/1".into()
        } else {
            "tatbot.target-pose/1".into()
        }],
    )
    .map_err(|e| anyhow::anyhow!("{e}"))?;
    // The capture owner is ordered first by the unit and by the deploy
    // manifest, and its `Type=notify` unit reports ready only once bound; an
    // owner started by hand or restarted may still be binding, so absorb that
    // gap here rather than exiting 1 and leaning on `Restart=`.
    let latest = std::sync::Arc::new(std::sync::Mutex::new(
        BTreeMap::<usize, ReceivedFrameSet>::new(),
    ));
    let reader_errors =
        std::sync::Arc::new(std::sync::Mutex::new(BTreeMap::<usize, String>::new()));
    for (source, socket) in args.socket.iter().enumerate() {
        let mut client = tatbot_visiond::UnixFrameClient::connect_within(
            socket,
            tatbot_visiond::CONNECT_WINDOW,
        )?;
        client.set_read_timeout(Duration::from_secs(2))?;
        let incoming = latest.clone();
        let errors = reader_errors.clone();
        std::thread::spawn(move || {
            loop {
                match client.recv_if_ready() {
                    Ok(None) => continue,
                    Ok(Some(set)) => {
                        if let Ok(mut slots) = incoming.try_lock() {
                            slots.insert(source, set);
                        }
                    }
                    Err(e) => {
                        errors.lock().unwrap().insert(source, e.to_string());
                        break;
                    }
                }
            }
        });
    }
    let mut output = evidence::RollingEvidence::new(
        args.output.clone(),
        args.evidence_bytes,
        args.evidence_files,
    )?;
    let mut tracking_lost = true;
    let mut last_measured_stamp_ns = None;
    let mut processed = 0_u64;
    let mut window_started = Instant::now();
    let mut window_processed = 0_u64;
    let mut window_measured = 0_u64;
    let mut stale_sets = 0_u64;
    let mut last_age_ms = 0.0;
    let mut last_processed_unix_ms = 0.0;
    let mut last_stale_log: Option<Instant> = None;
    let mut dropped = 0_u64;
    let mut last_sequences = BTreeMap::new();
    let mut auxiliary_used = 0_u64;
    let mut auxiliary_missed = 0_u64;
    let mut latencies = std::collections::VecDeque::new();
    loop {
        if let Some((source, error)) = reader_errors.lock().unwrap().iter().next() {
            anyhow::bail!(
                "camera owner socket {}: {error}",
                args.socket[*source].display()
            );
        }
        if window_started.elapsed() >= Duration::from_secs(1) {
            let mut sorted = latencies.iter().copied().collect::<Vec<_>>();
            sorted.sort_by(f64::total_cmp);
            *lease.metrics.lock().unwrap() = BTreeMap::from([
                (
                    "pose_fps".into(),
                    window_processed as f64 / window_started.elapsed().as_secs_f64(),
                ),
                ("processed_sets".into(), processed as f64),
                ("dropped_sets".into(), dropped as f64),
                ("auxiliary_used_sets".into(), auxiliary_used as f64),
                ("auxiliary_missed_sets".into(), auxiliary_missed as f64),
                (
                    "processing_p95_ms".into(),
                    sorted
                        .get(sorted.len().saturating_sub(1) * 95 / 100)
                        .copied()
                        .unwrap_or(0.0),
                ),
                ("capture_age_ms".into(), last_age_ms),
                (
                    "updated_unix_ms".into(),
                    std::time::SystemTime::now()
                        .duration_since(std::time::UNIX_EPOCH)?
                        .as_secs_f64()
                        * 1000.0,
                ),
                ("last_processed_unix_ms".into(), last_processed_unix_ms),
                ("stale_sets".into(), stale_sets as f64),
                (
                    "measured_pose_fps".into(),
                    window_measured as f64 / window_started.elapsed().as_secs_f64(),
                ),
                ("processing_samples".into(), sorted.len() as f64),
            ]);
            window_started = Instant::now();
            window_processed = 0;
            window_measured = 0;
        }
        let received = {
            let mut pending = latest.lock().unwrap();
            pending.remove(&0).map(|primary| {
                merge_auxiliary(
                    primary,
                    &mut pending,
                    (args.auxiliary_sync_ms * 1e6) as u128,
                )
            })
        };
        let Some(received) = received else {
            std::thread::sleep(Duration::from_millis(2));
            continue;
        };
        let (received, used_sources, source_sequences) = received?;
        if args.socket.len() > 1 {
            if used_sources > 1 {
                auxiliary_used += 1;
            } else {
                auxiliary_missed += 1;
            }
        }
        let started = Instant::now();
        let set = tatbot_visiond::SynchronizedFrameSet {
            sequence: received.sequence,
            timestamp_basis: received.timestamp_basis,
            timestamp_ns: received.timestamp_ns,
            maximum_skew_ns: received.maximum_skew_ns,
            frames: received
                .frames
                .into_iter()
                .map(|f| (f.metadata.sensor_name.clone(), f))
                .collect(),
        };
        ensure!(
            set.frames
                .values()
                .all(|f| f.metadata.calibration_id.as_deref()
                    == Some(calibration.bundle_id.as_str())),
            "capture calibration differs from tracker"
        );
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)?
            .as_nanos() as i128;
        let age_ms = (now - set.timestamp_ns) as f64 / 1e6;
        last_age_ms = age_ms;
        if age_ms < 0.0 || age_ms > args.max_age_ms {
            stale_sets += 1;
            if last_stale_log.is_none_or(|last| last.elapsed() >= Duration::from_secs(5)) {
                eprintln!(
                    "stale_frame sequence={} age_ms={age_ms} rejected_total={stale_sets}",
                    set.sequence
                );
                last_stale_log = Some(Instant::now());
            }
            continue;
        }
        let initializer_recent =
            initializer_is_recent(last_measured_stamp_ns, set.timestamp_ns, args.max_age_ms);
        let mut selected = if tracking_lost || !initializer_recent {
            BTreeMap::new()
        } else {
            tracker.predicted_rois(&set, args.roi_margin_px)
        };
        // Stagger full searches across cameras to avoid a periodic five-camera
        // CPU spike. Lost tracking always requests full-frame reacquisition.
        let stagger = (args.full_scan_period / set.frames.len().max(1) as u64).max(1);
        for (index, name) in set.frames.keys().enumerate() {
            if (processed + index as u64 * stagger).is_multiple_of(args.full_scan_period) {
                selected.remove(name);
            }
        }
        let pose_candidates = !args.temporal_initializers || tracking_lost || !initializer_recent;
        // A merged overhead set carries the aligned depth plane next to its
        // color frame. Depth never holds a tag; keep it out of the detector
        // rather than failing the whole cadence set.
        let undetectable = set
            .frames
            .iter()
            .filter(|(_, frame)| !tatbot_visiond::fiducials::detectable_frame(frame))
            .map(|(name, _)| name.clone())
            .collect::<BTreeSet<_>>();
        let detections = detector.detect_set_with_pose_candidates(
            &calibration,
            &set,
            &undetectable,
            &selected,
            pose_candidates,
        )?;
        let detection_ms = started.elapsed().as_secs_f64() * 1000.0;
        let mut estimate = tracker.update_constrained(
            processed,
            set.timestamp_ns,
            set.maximum_skew_ns,
            detections.detections,
            age_ms,
            detection_ms,
            started,
            if set.frames.len() < calibration.cameras.len() {
                2
            } else {
                0
            },
        );
        tracking_lost = estimate.status != "measured";
        if !tracking_lost {
            last_measured_stamp_ns = Some(set.timestamp_ns);
        }
        window_measured += u64::from(!tracking_lost);
        estimate.input_cameras = set.frames.keys().cloned().collect();
        estimate.partial_input = Some(set.frames.len() < calibration.cameras.len());
        estimate.image_prep_latency_ms = detections.image_prep_latency_ms;
        estimate.apriltag_latency_ms = detections.apriltag_latency_ms;
        estimate.quad_detection_latency_ms = detections.quad_detection_latency_ms;
        estimate.pose_candidate_latency_ms = detections.pose_candidate_latency_ms;
        estimate.roi_camera_count = detections.roi_camera_count;
        let mut record = serde_json::to_value(&estimate)?;
        record["detector_quad_decimate"] = serde_json::json!(detector.quad_decimate());
        record["detector_reacquire_quad_decimate"] =
            serde_json::json!(detector.reacquire_quad_decimate());
        record["detector_reacquire_scale"] = serde_json::json!(args.reacquire_scale);
        record["pose_candidates_computed"] = serde_json::json!(pose_candidates);
        record["input_sequences"] = serde_json::json!(source_sequences);
        record["input_sockets"] = serde_json::json!(args.socket);

        let message = Envelope {
            schema: "tatbot.tracking-pose/1".into(),
            producer: producer.clone(),
            stamp: Stamp {
                mono_ns: 0,
                wall_ns: u64::try_from(set.timestamp_ns)?,
                basis: set.timestamp_basis,
            },
            seq: processed,
            payload: serde_json::json!({"target":args.target,"tracking_frame":estimate.tracking_frame,"source":estimate.status,"tag_ids":estimate.used_tags,"inventory_hash":estimate.inventory_hash,"wrist_layout_hash":estimate.wrist_layout_hash,"confidence":if estimate.status=="measured" {1.0}else{0.0},"calibration_id":estimate.calibration_id,"pose":estimate.world_from_ee,"cameras":estimate.used_cameras,"estimate":estimate}),
        };
        let mut message = message;
        let topic = if let Some(arm) = tracked_arm(&args.target) {
            format!("tatbot/tracking/ee/{arm}")
        } else {
            message.schema = "tatbot.target-pose/1".into();
            message.payload = serde_json::json!({
                "target_id":args.target, "target_frame":estimate.tracking_frame,
                "source":estimate.status, "world_from_target":estimate.world_from_ee,
                "calibration_id":estimate.calibration_id,
                "inventory_sha256":estimate.inventory_hash, "layout_sha256":estimate.wrist_layout_hash,
                "translation_sigma_m":estimate.translation_sigma_mm.map(|v| v / 1000.0),
                "rotation_sigma_rad":estimate.rotation_sigma_deg.map(f64::to_radians),
                "tag_ids":estimate.used_tags
            });
            format!("tatbot/tracking/target/{}", args.target)
        };
        output.write_record(&serde_json::to_vec(&if tracked_arm(&args.target).is_some() {
            record
        } else {
            serde_json::to_value(&message)?
        })?)?;
        bus.publish(&topic, &message)
            .map_err(|e| anyhow::anyhow!("{e}"))?;
        last_processed_unix_ms = now as f64 / 1e6;
        processed += 1;
        window_processed += 1;
        for (source, sequence) in source_sequences {
            if let Some(previous) = last_sequences.insert(source, sequence) {
                dropped += sequence.saturating_sub(previous + 1);
            }
        }
        latencies.push_back(started.elapsed().as_secs_f64() * 1000.0);
        if latencies.len() > 100 {
            latencies.pop_front();
        }
        if args.max_sets > 0 && processed >= args.max_sets {
            break;
        }
    }
    output.finish()?;
    Ok(())
}

// A previous measured pose is only an initializer while its capture clock is
// monotonic and recent. Reacquire after gaps, even if no lost estimate was emitted.
fn initializer_is_recent(previous: Option<i128>, current: i128, max_age_ms: f64) -> bool {
    previous
        .and_then(|previous| current.checked_sub(previous))
        .is_some_and(|age| age >= 0 && age as f64 / 1e6 <= max_age_ms)
}

#[cfg(test)]
mod initializer_tests {
    use super::initializer_is_recent;

    #[test]
    fn requires_recent_monotonic_measured_capture() {
        assert!(!initializer_is_recent(None, 1_000_000_000, 250.0));
        assert!(initializer_is_recent(
            Some(1_000_000_000),
            1_250_000_000,
            250.0
        ));
        assert!(!initializer_is_recent(
            Some(1_000_000_000),
            1_250_000_001,
            250.0
        ));
        assert!(!initializer_is_recent(
            Some(1_000_000_000),
            999_999_999,
            250.0
        ));
        assert!(!initializer_is_recent(Some(i128::MIN), i128::MAX, 250.0));
    }
}
