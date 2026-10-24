//! One fleet stencil observer on the overhead camera's node. Each turn it
//! hands the newest overhead RGB-D set to the Python estimator as a
//! digest-bound capture, and publishes every visible print's pose as
//! `tatbot.target-pose/1` on `tatbot/tracking/target/<pattern_id>` with the
//! support it rests on. No camera backend, no arm, no motion authority: a
//! print's pose is a number the daemon compares, never a permission.
use anyhow::{Result, ensure};
use clap::Parser;
use serde::Serialize;
use sha2::{Digest, Sha256};
use std::{
    collections::{BTreeMap, BTreeSet, VecDeque},
    path::{Path, PathBuf},
    sync::{Arc, Mutex, atomic::AtomicBool},
    time::{Duration, Instant},
};
use tatbot_bus::{Envelope, Producer, Stamp, service::ServiceLease, transport::Bus};
use tatbot_visiond::{CalibrationBundle, FrameRecord, ReceivedFrameSet};
use trackd::{evidence::RollingEvidence, worker::Worker};

/// Recent sets kept per owner socket while a turn is paced.
const RING: usize = 8;
const SCHEMA: &str = "tatbot.target-pose/1";
const OVERHEAD: &str = "overhead-depth";

#[derive(Parser)]
struct Args {
    #[arg(long, required = true)]
    connect: Vec<String>,
    #[arg(long)]
    node: String,
    /// The overhead RGB-D owner's frame socket: the observer's one view.
    #[arg(long)]
    overhead_socket: PathBuf,
    #[arg(long)]
    calibration: PathBuf,
    /// Installed stencil references, one directory each holding `tracking.json`
    /// beside its `stencil.png`; rescanned by mtime every turn, idle when empty.
    #[arg(long)]
    references: PathBuf,
    /// The observer venv's interpreter and the observer script it runs.
    #[arg(long)]
    python: PathBuf,
    #[arg(long)]
    observer: PathBuf,
    /// Scratch directory for the per-turn captures the estimator reads.
    #[arg(long)]
    work: PathBuf,
    #[arg(long)]
    output: PathBuf,
    /// One estimator turn per period.
    #[arg(long, default_value_t = 1000)]
    turn_ms: u64,
    #[arg(long, default_value_t = 3000.0)]
    max_age_ms: f64,
    #[arg(long, default_value_t = 67_108_864)]
    evidence_bytes: u64,
    #[arg(long, default_value_t = 4)]
    evidence_files: usize,
    #[arg(long, default_value_t = 0)]
    max_turns: u64,
}

fn frame_stamp(frame: &tatbot_visiond::FrameRecord) -> i128 {
    frame
        .metadata
        .timestamps
        .normalized_unix_ns
        .unwrap_or(frame.metadata.timestamps.host_unix_ns)
}

/// A file's identity for the input fingerprint: path, size and mtime.
fn fingerprint_file(digest: &mut Sha256, file: &Path) -> Result<()> {
    let meta = std::fs::metadata(file)?;
    let modified = meta
        .modified()?
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_nanos())
        .unwrap_or(0);
    digest.update(format!("{}\t{}\t{modified}\n", file.display(), meta.len()));
    Ok(())
}

/// Installed references and a fingerprint that changes when any is added,
/// removed or rewritten: the manifest, its image and the tile's `tile.json`
/// by path, size and mtime. The observer's loader verifies the
/// files against the manifest; this only decides when to restart it.
fn scan_references(directory: &Path) -> Result<(Vec<PathBuf>, String)> {
    let mut paths = Vec::new();
    let mut digest = Sha256::new();
    if directory.is_dir() {
        let mut entries = std::fs::read_dir(directory)?
            .collect::<std::io::Result<Vec<_>>>()?
            .into_iter()
            .map(|entry| entry.path())
            .filter(|path| path.is_dir())
            .collect::<Vec<_>>();
        entries.sort();
        for entry in entries {
            let manifest = entry.join("tracking.json");
            let image = entry.join("stencil.png");
            if !(manifest.is_file() && image.is_file()) {
                continue;
            }
            for file in [&manifest, &image] {
                fingerprint_file(&mut digest, file)?;
            }
            let tile = entry.join("tile.json");
            if tile.is_file() {
                fingerprint_file(&mut digest, &tile)?;
            }
            paths.push(manifest);
        }
    }
    Ok((paths, format!("{:x}", digest.finalize())))
}

/// Every frame of one role written digest-bound beside its manifest entry,
/// with the exposure window the frames span.
fn frame_entries(
    directory: &Path,
    role: &str,
    frames: &[FrameRecord],
) -> Result<(serde_json::Map<String, serde_json::Value>, (i128, i128))> {
    ensure!(
        !frames.is_empty() && frames.len() <= 5,
        "live capture frame count"
    );
    let mut window: Option<(i128, i128)> = None;
    let mut entries = serde_json::Map::new();
    for (index, frame) in frames.iter().enumerate() {
        let name = &frame.metadata.sensor_name;
        ensure!(!entries.contains_key(name), "duplicate sensor {name}");
        let stamp = frame
            .metadata
            .timestamps
            .normalized_unix_ns
            .ok_or_else(|| anyhow::anyhow!("camera timestamp missing"))?;
        window = Some(window.map_or((stamp, stamp), |(a, b)| (a.min(stamp), b.max(stamp))));
        let filename = format!("{role}-{index}.pixels");
        let bytes = frame.payload.bytes();
        std::fs::write(directory.join(&filename), bytes)?;
        entries.insert(name.clone(), serde_json::json!({"metadata":frame.metadata,
            "payload_file":filename, "payload_bytes":bytes.len(), "sha256":format!("{:x}", Sha256::digest(bytes))}));
    }
    Ok((entries, window.expect("at least one frame")))
}

/// The overhead set as the estimator reads it: `tatbot.session-surface/1` kind `live-capture`,
/// every payload digest-bound, the exposure window the frames span, the
/// aligned RGB-D pair the bundle names.
fn write_capture(
    directory: &Path,
    set: &ReceivedFrameSet,
    producer: &serde_json::Value,
    calibration: &str,
) -> Result<PathBuf> {
    for frame in &set.frames {
        let name = &frame.metadata.sensor_name;
        ensure!(
            frame.metadata.calibration_id.as_deref() == Some(calibration),
            "camera calibration identity differs"
        );
        ensure!(
            ["overhead_depth_color", "overhead_depth_depth"].contains(&name.as_str()),
            "unexpected overhead sensor"
        );
    }
    let (frames, (after_ns, before_ns)) = frame_entries(directory, OVERHEAD, &set.frames)?;
    tatbot_visiond::frame_bus::validate_capture_geometry(&tatbot_visiond::SynchronizedFrameSet {
        sequence: set.sequence,
        timestamp_basis: set.timestamp_basis.clone(),
        timestamp_ns: set.timestamp_ns,
        maximum_skew_ns: set.maximum_skew_ns,
        frames: set
            .frames
            .iter()
            .map(|f| (f.metadata.sensor_name.clone(), f.clone()))
            .collect(),
    })?;
    let path = directory.join(format!("{OVERHEAD}.json"));
    std::fs::write(
        &path,
        serde_json::to_vec(&serde_json::json!({
            "schema":"tatbot.session-surface/1", "kind":"live-capture", "producer":producer,
            "geometry_calibration_id":calibration,
            "wrist_capture_window":{"after_ns":after_ns, "before_ns":before_ns}, "frames":frames,
        }))?,
    )?;
    Ok(path)
}

fn hex64(value: &serde_json::Value) -> bool {
    value.as_str().is_some_and(|s| {
        s.len() == 64
            && s.bytes()
                .all(|b| b.is_ascii_hexdigit() && !b.is_ascii_uppercase())
    })
}

fn finite_matrix(value: &serde_json::Value) -> bool {
    let Some(rows) = value.as_array() else {
        return false;
    };
    rows.len() == 4
        && rows.iter().all(|row| {
            row.as_array().is_some_and(|cells| {
                cells.len() == 4
                    && cells
                        .iter()
                        .all(|cell| cell.as_f64().is_some_and(f64::is_finite))
            })
        })
}

/// Every target the estimator reported as one `tatbot.target-pose/1`
/// envelope on its print's topic. A malformed target refuses the whole turn:
/// a consumer never sees a pose whose shape the observer did not check.
fn publications(
    reply: &serde_json::Value,
    bundle_id: &str,
    producer: &Producer,
    seq: u64,
    basis: &str,
) -> Result<Vec<(String, Envelope<serde_json::Value>)>> {
    ensure!(
        hex64(&reply["inventory_sha256"]),
        "estimator reply lacks the reference inventory digest"
    );
    let targets = reply["targets"]
        .as_array()
        .ok_or_else(|| anyhow::anyhow!("estimator reply lacks targets"))?;
    let observer_epoch = reply["observer_epoch"]
        .as_str()
        .filter(|epoch| !epoch.is_empty())
        .ok_or_else(|| anyhow::anyhow!("estimator reply lacks its observer epoch"))?;
    let mut out = Vec::new();
    let mut patterns = BTreeSet::new();
    for target in targets {
        let pattern = target["pattern_id"]
            .as_str()
            .ok_or_else(|| anyhow::anyhow!("target without a pattern id"))?;
        ensure!(
            pattern
                .strip_prefix("stencil-")
                .is_some_and(|rest| hex64(&serde_json::json!(rest))),
            "invalid pattern id {pattern}"
        );
        ensure!(
            patterns.insert(pattern),
            "ambiguous duplicate physical print for pattern {pattern}"
        );
        ensure!(
            hex64(&target["reference_id"]),
            "target without its reference id"
        );
        let source = target["source"].as_str().unwrap_or("");
        let measured = match source {
            "measured" => true,
            "lost" => false,
            _ => anyhow::bail!("target source must be measured or lost"),
        };
        let pose = &target["world_from_target"];
        let sigma = |key: &str| {
            target[key]
                .as_f64()
                .is_some_and(|v| v.is_finite() && v >= 0.0)
        };
        if measured {
            ensure!(
                finite_matrix(pose) && sigma("translation_sigma_m") && sigma("rotation_sigma_rad"),
                "measured target lacks a finite pose or its sigmas"
            );
        } else {
            ensure!(pose.is_null(), "a lost target carries no pose");
        }
        let capture_ns = target["capture_ns"]
            .as_u64()
            .ok_or_else(|| anyhow::anyhow!("target without a capture time"))?;
        let mut support = target["support"].clone();
        ensure!(support.is_object(), "target without support");
        support["motion_authority"] = serde_json::json!(false);
        support["bundle_id"] = serde_json::json!(bundle_id);
        support["observer_epoch"] = serde_json::json!(observer_epoch);
        let payload = serde_json::json!({
            "target_id": pattern, "target_frame": "world", "source": source,
            "world_from_target": if measured { pose.clone() } else { serde_json::Value::Null },
            "calibration_id": bundle_id,
            "inventory_sha256": reply["inventory_sha256"], "layout_sha256": target["reference_id"],
            "translation_sigma_m": if measured { target["translation_sigma_m"].clone() } else { serde_json::Value::Null },
            "rotation_sigma_rad": if measured { target["rotation_sigma_rad"].clone() } else { serde_json::Value::Null },
            "tag_ids": [], "support": support,
        });
        out.push((
            format!("tatbot/tracking/target/{pattern}"),
            Envelope {
                schema: SCHEMA.into(),
                producer: producer.clone(),
                stamp: Stamp {
                    mono_ns: 0,
                    wall_ns: capture_ns,
                    basis: basis.into(),
                },
                seq,
                payload,
            },
        ));
    }
    Ok(out)
}

fn now_unix_ns() -> Result<i128> {
    Ok(std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)?
        .as_nanos() as i128)
}

type Ring = Arc<Mutex<VecDeque<ReceivedFrameSet>>>;

#[derive(Clone, Default, Serialize)]
struct CameraDelivery {
    socket_frames: u64,
    unique_socket_frames: u64,
    repeated_socket_frames: u64,
    sequence_skips: u64,
    selected_frames: u64,
    unique_selected_frames: u64,
    last_socket_sequence: Option<u64>,
    last_socket_stamp_ns: Option<i128>,
    last_selected_stamp_ns: Option<i128>,
}

#[derive(Clone, Default, Serialize)]
struct SocketDelivery {
    socket_sets: u64,
    ring_overwrites: u64,
    unselected_sets: u64,
    cameras: BTreeMap<String, CameraDelivery>,
}

impl SocketDelivery {
    fn received(&mut self, set: &ReceivedFrameSet) {
        self.socket_sets += 1;
        for frame in &set.frames {
            let row = self
                .cameras
                .entry(frame.metadata.sensor_name.clone())
                .or_default();
            let stamp = frame_stamp(frame);
            row.socket_frames += 1;
            if row.last_socket_stamp_ns.is_none_or(|last| stamp > last) {
                row.unique_socket_frames += 1;
            } else {
                row.repeated_socket_frames += 1;
            }
            if let Some(last) = row.last_socket_sequence {
                row.sequence_skips += frame
                    .metadata
                    .sequence
                    .saturating_sub(last.saturating_add(1));
            }
            row.last_socket_sequence = Some(frame.metadata.sequence);
            row.last_socket_stamp_ns = Some(stamp);
        }
    }

    fn selected(&mut self, set: &ReceivedFrameSet, skipped: usize) {
        self.unselected_sets += skipped as u64;
        for frame in &set.frames {
            let row = self
                .cameras
                .entry(frame.metadata.sensor_name.clone())
                .or_default();
            let stamp = frame_stamp(frame);
            row.selected_frames += 1;
            if row.last_selected_stamp_ns.is_none_or(|last| stamp > last) {
                row.unique_selected_frames += 1;
            }
            row.last_selected_stamp_ns = Some(stamp);
        }
    }
}

fn subscribe(
    socket: &Path,
    errors: Arc<Mutex<BTreeMap<String, String>>>,
) -> Result<(Ring, Arc<Mutex<SocketDelivery>>)> {
    // The capture owner is ordered first by the unit, but `After=` only
    // sequences process starts: absorb its binding window here rather than
    // exiting and leaning on `Restart=`.
    let mut client =
        tatbot_visiond::UnixFrameClient::connect_within(socket, tatbot_visiond::CONNECT_WINDOW)?;
    client.set_read_timeout(Duration::from_secs(2))?;
    let ring: Ring = Arc::new(Mutex::new(VecDeque::new()));
    let incoming = ring.clone();
    let delivery = Arc::new(Mutex::new(SocketDelivery::default()));
    let received = delivery.clone();
    let name = socket.display().to_string();
    std::thread::spawn(move || {
        loop {
            match client.recv_if_ready() {
                Ok(None) => continue,
                Ok(Some(set)) => {
                    received
                        .lock()
                        .unwrap_or_else(|e| e.into_inner())
                        .received(&set);
                    let mut slots = incoming.lock().unwrap_or_else(|e| e.into_inner());
                    slots.push_back(set);
                    while slots.len() > RING {
                        slots.pop_front();
                        received
                            .lock()
                            .unwrap_or_else(|e| e.into_inner())
                            .ring_overwrites += 1;
                    }
                }
                Err(e) => {
                    errors
                        .lock()
                        .unwrap_or_else(|e| e.into_inner())
                        .insert(name, e.to_string());
                    break;
                }
            }
        }
    });
    Ok((ring, delivery))
}

/// Take a set out of its ring, dropping anything older than the chosen
/// exposure so a later turn never observes the past. Called under the lock
/// the set was chosen in: an owner push between the two cannot shift the
/// index.
fn take(slots: &mut VecDeque<ReceivedFrameSet>, index: usize) -> (ReceivedFrameSet, usize) {
    let chosen = slots.remove(index).expect("index from the same lock scope");
    let before = slots.len();
    slots.retain(|set| set.timestamp_ns > chosen.timestamp_ns);
    (chosen, before - slots.len())
}

fn main() -> Result<()> {
    let args = Args::parse();
    ensure!(
        args.turn_ms > 0 && args.max_age_ms.is_finite() && args.max_age_ms > 0.0,
        "invalid observer limits"
    );
    let _owner = tatbot_visiond::ownership::CameraLease::acquire(
        &std::env::temp_dir().join("tatbot-stencild.lock"),
    )?;
    let calibration = CalibrationBundle::load(&args.calibration)?;
    std::fs::create_dir_all(&args.work)?;
    let bus = Bus::open(&args.connect).map_err(|e| anyhow::anyhow!("{e}"))?;
    let producer = Producer::this_process(args.node.clone(), "stencild");
    let lease = ServiceLease::declare(&bus, producer.clone(), "stencild", vec![SCHEMA.into()])
        .map_err(|e| anyhow::anyhow!("{e}"))?;
    let reader_errors = Arc::new(Mutex::new(BTreeMap::<String, String>::new()));
    let (overhead, overhead_delivery) = subscribe(&args.overhead_socket, reader_errors.clone())?;
    let mut output = RollingEvidence::new(
        args.output.clone(),
        args.evidence_bytes,
        args.evidence_files,
    )?;
    let stop = AtomicBool::new(false);
    let owner = serde_json::json!({"role": OVERHEAD, "node": tatbot_bus::fleet::node_with_role(OVERHEAD),
        "via": "frame-socket", "observer": producer});
    let mut worker: Option<Worker> = None;
    let mut fingerprint = String::new();
    let mut references = 0_usize;
    let mut turns = 0_u64;
    let mut published = 0_u64;
    let mut measured = 0_u64;
    let mut stale = 0_u64;
    let mut restarts = 0_u64;
    let mut last_age_ms = 0.0;
    let mut last_publication_age_ms = 0.0;
    let mut publication_ages = VecDeque::new();
    let mut last_capture_write_ms = 0.0;
    let mut last_estimator_ms = 0.0;
    let mut last_publish_ms = 0.0;
    let mut last_observer_timings = serde_json::Value::Null;
    let mut latencies = VecDeque::new();
    loop {
        let turn_started = Instant::now();
        if let Some((socket, error)) = reader_errors
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .iter()
            .next()
        {
            anyhow::bail!("camera owner socket {socket}: {error}");
        }
        let (paths, current) = scan_references(&args.references)?;
        if current != fingerprint {
            // A changed reference set restarts the estimator with the new
            // bank; an emptied one leaves the observer idle, never exited.
            worker = None;
            fingerprint = current;
            references = paths.len();
            eprintln!("references changed: {references} installed");
        }
        if worker.is_none() && references > 0 {
            let binding = args.work.join("binding.json");
            std::fs::write(
                &binding,
                serde_json::to_vec(&serde_json::json!({
                    "references": paths, "calibration": calibration,
                    "robot_world": {"calibration_id": calibration.bundle_id, "frame": "world",
                        "world_from_base": [[1.0,0.0,0.0,0.0],[0.0,1.0,0.0,0.0],[0.0,0.0,1.0,0.0],[0.0,0.0,0.0,1.0]]},
                    "observer_epoch": format!("{}-{}", std::process::id(), now_unix_ns()?),
                    "evidence_kind": "live-rgbd",
                    "publish_targets_only": true,
                }))?,
            )?;
            match Worker::start(&args.python, &args.observer, &[&binding, &args.work], &stop) {
                Ok(started) => worker = Some(started),
                Err(error) => {
                    restarts += 1;
                    eprintln!("estimator start: {error:#}");
                }
            }
        }
        {
            let mut sorted = latencies.iter().copied().collect::<Vec<f64>>();
            sorted.sort_by(f64::total_cmp);
            let mut ages = publication_ages.iter().copied().collect::<Vec<f64>>();
            ages.sort_by(f64::total_cmp);
            *lease.metrics.lock().unwrap() = BTreeMap::from([
                ("references".into(), references as f64),
                ("turns".into(), turns as f64),
                ("published_targets".into(), published as f64),
                ("measured_targets".into(), measured as f64),
                ("stale_sets".into(), stale as f64),
                ("estimator_restarts".into(), restarts as f64),
                ("capture_age_ms".into(), last_age_ms),
                ("capture_to_publication_ms".into(), last_publication_age_ms),
                (
                    "capture_to_publication_p95_ms".into(),
                    ages.get(ages.len().saturating_sub(1) * 95 / 100)
                        .copied()
                        .unwrap_or(0.0),
                ),
                ("capture_write_ms".into(), last_capture_write_ms),
                ("estimator_ms".into(), last_estimator_ms),
                ("publish_ms".into(), last_publish_ms),
                (
                    "processing_p95_ms".into(),
                    sorted
                        .get(sorted.len().saturating_sub(1) * 95 / 100)
                        .copied()
                        .unwrap_or(0.0),
                ),
                ("estimator_up".into(), f64::from(u8::from(worker.is_some()))),
                ("updated_unix_ms".into(), now_unix_ns()? as f64 / 1e6),
            ]);
        }
        let Some(estimator) = worker.as_mut() else {
            std::thread::sleep(Duration::from_millis(args.turn_ms));
            continue;
        };
        // Choose and take under the same lock: the index chosen is the one
        // removed, whatever the owner pushes meanwhile.
        let taken = {
            let mut sets = overhead.lock().unwrap_or_else(|e| e.into_inner());
            (0..sets.len())
                .max_by_key(|&i| sets[i].timestamp_ns)
                .map(|index| take(&mut sets, index))
        };
        let Some((set, skipped)) = taken else {
            std::thread::sleep(Duration::from_millis(5));
            continue;
        };
        overhead_delivery
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .selected(&set, skipped);
        let newest = set.timestamp_ns;
        let age_ms = (now_unix_ns()? - newest) as f64 / 1e6;
        last_age_ms = age_ms;
        if age_ms < 0.0 || age_ms > args.max_age_ms {
            stale += 1;
            continue;
        }
        let capture = args.work.join(format!("capture-{turns}"));
        let capture_started = Instant::now();
        std::fs::create_dir(&capture)?;
        let mut captures = serde_json::Map::new();
        let mut errors = serde_json::Map::new();
        match write_capture(&capture, &set, &owner, &calibration.bundle_id) {
            Ok(path) => {
                captures.insert(OVERHEAD.into(), serde_json::json!(path));
            }
            Err(error) => {
                errors.insert(OVERHEAD.into(), serde_json::json!(format!("{error:#}")));
            }
        }
        let basis = set.timestamp_basis.clone();
        let request = serde_json::json!({"captures": captures, "errors": errors});
        last_capture_write_ms = capture_started.elapsed().as_secs_f64() * 1000.0;
        let estimator_started = Instant::now();
        let reply = estimator.request_within(&request, &stop, Duration::from_secs(30));
        last_estimator_ms = estimator_started.elapsed().as_secs_f64() * 1000.0;
        let _ = std::fs::remove_dir_all(&capture);
        let reply = match reply {
            Ok(reply) => reply,
            Err(error) => {
                eprintln!("estimator turn {turns}: {error:#}");
                worker = None;
                restarts += 1;
                continue;
            }
        };
        if let Some(error) = reply.get("error").and_then(|e| e.as_str()) {
            eprintln!("estimator turn {turns}: {error}");
        } else {
            last_observer_timings = reply["timings_ms"].clone();
            match publications(&reply, &calibration.bundle_id, &producer, turns, &basis) {
                Ok(messages) => {
                    let publish_started = Instant::now();
                    for (topic, message) in messages {
                        let publish_ns = now_unix_ns()?;
                        let age_ms = (publish_ns - i128::from(message.stamp.wall_ns)) as f64 / 1e6;
                        last_publication_age_ms = age_ms;
                        publication_ages.push_back(age_ms);
                        if publication_ages.len() > 100 {
                            publication_ages.pop_front();
                        }
                        output.write_record(&serde_json::to_vec(&message)?)?;
                        bus.publish(&topic, &message)
                            .map_err(|e| anyhow::anyhow!("{e}"))?;
                        published += 1;
                        measured += u64::from(message.payload["source"] == "measured");
                    }
                    last_publish_ms = publish_started.elapsed().as_secs_f64() * 1000.0;
                }
                Err(error) => eprintln!("estimator turn {turns} refused: {error:#}"),
            }
        }
        turns += 1;
        latencies.push_back(turn_started.elapsed().as_secs_f64() * 1000.0);
        eprintln!(
            "turn-profile {}",
            serde_json::json!({
                "turn": turns, "capture_write_ms": last_capture_write_ms,
                "estimator_ms": last_estimator_ms, "publish_ms": last_publish_ms,
                "capture_to_publication_ms": last_publication_age_ms,
                "observer": last_observer_timings,
                "inputs": {
                    "overhead": overhead_delivery.lock().unwrap_or_else(|e| e.into_inner()).clone(),
                },
            })
        );
        if latencies.len() > 100 {
            latencies.pop_front();
        }
        if args.max_turns > 0 && turns >= args.max_turns {
            break;
        }
        // One turn per period, however fast the estimator answered.
        let period = Duration::from_millis(args.turn_ms);
        if let Some(rest) = period.checked_sub(turn_started.elapsed()) {
            std::thread::sleep(rest);
        }
    }
    output.finish()?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use tatbot_visiond::{
        FrameMetadata, FrameRecord, FrameTimestamps, PixelFormat, RecordedPayload, SensorKind,
        StreamProfile, time::TimestampDomain,
    };

    fn frame(name: &str, kind: SensorKind, stamp: i128, payload: RecordedPayload) -> FrameRecord {
        let (format, stream) = match &payload {
            RecordedPayload::Depth { .. } => (PixelFormat::Z16, "depth"),
            _ => (PixelFormat::Bgr8, "color"),
        };
        FrameRecord {
            metadata: FrameMetadata {
                sensor_name: name.into(),
                sensor_kind: kind,
                sequence: 1,
                profile: StreamProfile {
                    stream: stream.into(),
                    format,
                    width: 2,
                    height: 2,
                    fps_num: 30,
                    fps_den: 1,
                },
                timestamps: FrameTimestamps {
                    source_ns: Some(stamp),
                    source_domain: TimestampDomain::HostUnix,
                    rtp_timestamp: None,
                    pipeline_pts_ns: None,
                    pipeline_dts_ns: None,
                    host_monotonic_ns: 1,
                    host_unix_ns: stamp,
                    normalized_unix_ns: Some(stamp),
                },
                dropped_before: 0,
                calibration_id: Some("bundle".into()),
                flags: vec![],
                attributes: Default::default(),
            },
            payload,
        }
    }

    fn video() -> RecordedPayload {
        RecordedPayload::Video {
            format: PixelFormat::Bgr8,
            width: 2,
            height: 2,
            bytes: vec![7; 12],
        }
    }

    fn set(sequence: u64, stamp: i128, frames: Vec<FrameRecord>) -> ReceivedFrameSet {
        ReceivedFrameSet {
            envelope: None,
            sequence,
            timestamp_basis: "normalized_source".into(),
            timestamp_ns: stamp,
            maximum_skew_ns: 0,
            frames,
        }
    }

    #[test]
    fn delivery_counts_unique_camera_exposures_and_skipped_sets() {
        let mut delivery = SocketDelivery::default();
        let mut first = set(
            1,
            100,
            vec![frame("camera1", SensorKind::RealSense, 100, video())],
        );
        first.frames[0].metadata.sequence = 10;
        let mut repeated = set(
            2,
            100,
            vec![frame("camera1", SensorKind::RealSense, 100, video())],
        );
        repeated.frames[0].metadata.sequence = 10;
        let mut next = set(
            3,
            300,
            vec![frame("camera1", SensorKind::RealSense, 300, video())],
        );
        next.frames[0].metadata.sequence = 13;
        for set in [&first, &repeated, &next] {
            delivery.received(set);
        }
        delivery.selected(&next, 2);
        let camera = &delivery.cameras["camera1"];
        assert_eq!(delivery.socket_sets, 3);
        assert_eq!(delivery.unselected_sets, 2);
        assert_eq!(camera.socket_frames, 3);
        assert_eq!(camera.unique_socket_frames, 2);
        assert_eq!(camera.repeated_socket_frames, 1);
        assert_eq!(camera.sequence_skips, 2);
        assert_eq!(camera.unique_selected_frames, 1);
    }

    fn overhead_set(sequence: u64, stamp: i128) -> ReceivedFrameSet {
        let mut color = frame(
            "overhead_depth_color",
            SensorKind::RealSense,
            stamp,
            video(),
        );
        let mut depth = frame(
            "overhead_depth_depth",
            SensorKind::RealSense,
            stamp,
            RecordedPayload::Depth {
                width: 2,
                height: 2,
                bytes: vec![0; 8],
            },
        );
        for record in [&mut color, &mut depth] {
            record
                .metadata
                .attributes
                .insert("device_serial".into(), "d555".into());
            record
                .metadata
                .attributes
                .insert("capture_epoch".into(), "e".into());
        }
        depth
            .metadata
            .attributes
            .insert("aligned_to".into(), "overhead_depth_color".into());
        depth
            .metadata
            .attributes
            .insert("depth_units_m".into(), "0.0001".into());
        set(sequence, stamp, vec![color, depth])
    }

    #[test]
    fn capture_manifest_binds_frames_exposure_window_and_payload_digests() {
        let directory = tempfile::tempdir().unwrap();
        let producer = serde_json::json!({"role": OVERHEAD});
        let mut overhead = overhead_set(3, 1_000_000_000);
        overhead.frames[1].metadata.timestamps.normalized_unix_ns = Some(1_010_000_000);
        let path = write_capture(directory.path(), &overhead, &producer, "bundle").unwrap();
        let manifest: serde_json::Value =
            serde_json::from_slice(&std::fs::read(&path).unwrap()).unwrap();
        assert_eq!(manifest["schema"], "tatbot.session-surface/1");
        assert_eq!(manifest["kind"], "live-capture");
        assert_eq!(manifest["geometry_calibration_id"], "bundle");
        assert_eq!(
            manifest["wrist_capture_window"],
            serde_json::json!({"after_ns": 1_000_000_000_i64, "before_ns": 1_010_000_000_i64})
        );
        let entry = &manifest["frames"]["overhead_depth_color"];
        let bytes = std::fs::read(
            directory
                .path()
                .join(entry["payload_file"].as_str().unwrap()),
        )
        .unwrap();
        assert_eq!(bytes, vec![7; 12]);
        assert_eq!(entry["payload_bytes"], 12);
        assert_eq!(entry["sha256"], format!("{:x}", Sha256::digest(&bytes)));
        assert_eq!(entry["metadata"]["sensor_name"], "overhead_depth_color");
        // Another camera's frame and a foreign bundle are refused.
        let mut foreign = overhead_set(5, 1_000_000_000);
        foreign.frames[0].metadata.sensor_name = "camera1".into();
        assert!(write_capture(directory.path(), &foreign, &producer, "bundle").is_err());
        assert!(
            write_capture(
                directory.path(),
                &overhead_set(6, 1_000_000_000),
                &producer,
                "other"
            )
            .is_err()
        );
    }

    #[test]
    fn references_are_rescanned_by_mtime_and_none_means_idle() {
        let directory = tempfile::tempdir().unwrap();
        let (paths, idle) = scan_references(directory.path()).unwrap();
        assert!(paths.is_empty());
        assert_eq!(
            scan_references(&directory.path().join("absent")).unwrap().1,
            idle
        );
        // An incomplete install (no image beside the manifest) is not a reference.
        let seed = directory.path().join("tatbot-43");
        std::fs::create_dir(&seed).unwrap();
        std::fs::write(seed.join("tracking.json"), b"{}").unwrap();
        assert_eq!(scan_references(directory.path()).unwrap().1, idle);
        std::fs::write(seed.join("stencil.png"), b"png").unwrap();
        let (paths, installed) = scan_references(directory.path()).unwrap();
        assert_eq!(paths, vec![seed.join("tracking.json")]);
        assert_ne!(installed, idle);
        assert_eq!(scan_references(directory.path()).unwrap().1, installed);
        // A rewritten manifest (a later mtime) is a new bank.
        let later = std::time::SystemTime::now() + Duration::from_secs(5);
        std::fs::File::options()
            .write(true)
            .open(seed.join("tracking.json"))
            .unwrap()
            .set_modified(later)
            .unwrap();
        let (_, rewritten) = scan_references(directory.path()).unwrap();
        assert_ne!(rewritten, installed);
        // A tile print's layout beside the manifest is part of the bank.
        std::fs::write(seed.join("tile.json"), b"{}").unwrap();
        let (paths, tiled) = scan_references(directory.path()).unwrap();
        assert_eq!(paths, vec![seed.join("tracking.json")]);
        assert_ne!(tiled, rewritten);
        std::fs::remove_dir_all(&seed).unwrap();
        assert_eq!(scan_references(directory.path()).unwrap().1, idle);
    }

    #[test]
    fn an_observation_becomes_one_target_pose_per_print_with_support_and_no_motion_authority() {
        let producer = Producer {
            node: "observer".into(),
            pid: 1,
            sha: "a".repeat(40),
            run_id: "test".into(),
        };
        let pattern = format!("stencil-{}", "b".repeat(64));
        let lost = format!("stencil-{}", "c".repeat(64));
        let identity = serde_json::json!([
            [1.0, 0.0, 0.0, 0.1],
            [0.0, 1.0, 0.0, 0.2],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0]
        ]);
        let reply = serde_json::json!({"inventory_sha256": "d".repeat(64), "observer_epoch": "test-epoch-1",
            "targets": [
            {"pattern_id": pattern, "reference_id": "e".repeat(64), "source": "measured",
             "world_from_target": identity, "translation_sigma_m": 0.001, "rotation_sigma_rad": 0.01,
             "capture_ns": 1_000_000_000_u64,
             "support": {"anchor_camera": "overhead_depth_color", "anchors": 12, "motion_authority": true}},
            {"pattern_id": lost, "reference_id": "f".repeat(64), "source": "lost",
             "world_from_target": null, "capture_ns": 1_000_000_000_u64,
             "support": {"anchor_camera": null, "anchors": 0, "reason": "no measured anchors"}}]});
        let messages = publications(&reply, "bundle", &producer, 7, "normalized_source").unwrap();
        assert_eq!(messages.len(), 2);
        let (topic, message) = &messages[0];
        assert_eq!(topic, &format!("tatbot/tracking/target/{pattern}"));
        assert_eq!(message.schema, SCHEMA);
        assert_eq!(message.seq, 7);
        assert_eq!(message.stamp.wall_ns, 1_000_000_000);
        let payload = &message.payload;
        assert_eq!(payload["target_id"], pattern);
        assert_eq!(payload["target_frame"], "world");
        assert_eq!(payload["source"], "measured");
        assert_eq!(payload["calibration_id"], "bundle");
        assert_eq!(payload["layout_sha256"], "e".repeat(64));
        assert_eq!(payload["inventory_sha256"], "d".repeat(64));
        assert_eq!(payload["support"]["observer_epoch"], "test-epoch-1");
        assert_eq!(payload["world_from_target"], identity);
        assert_eq!(payload["translation_sigma_m"], 0.001);
        assert_eq!(payload["tag_ids"], serde_json::json!([]));
        // The estimator cannot grant what the observer never holds.
        assert_eq!(payload["support"]["motion_authority"], false);
        assert_eq!(payload["support"]["bundle_id"], "bundle");
        assert_eq!(payload["support"]["anchor_camera"], "overhead_depth_color");
        let (_, lost_message) = &messages[1];
        assert_eq!(lost_message.payload["source"], "lost");
        assert!(lost_message.payload["world_from_target"].is_null());
        assert!(lost_message.payload["translation_sigma_m"].is_null());
        // A malformed target refuses the turn: a measured print without a
        // pose, a pattern id that is not a stencil identity, a reply without
        // its inventory digest.
        let mut broken = reply.clone();
        broken["targets"][0]["world_from_target"] = serde_json::Value::Null;
        assert!(publications(&broken, "bundle", &producer, 8, "normalized_source").is_err());
        let mut broken = reply.clone();
        broken["targets"][0]["pattern_id"] = serde_json::json!("wrist");
        assert!(publications(&broken, "bundle", &producer, 8, "normalized_source").is_err());
        let mut broken = reply.clone();
        broken["inventory_sha256"] = serde_json::Value::Null;
        assert!(publications(&broken, "bundle", &producer, 8, "normalized_source").is_err());
        let mut ambiguous = reply.clone();
        ambiguous["targets"]
            .as_array_mut()
            .unwrap()
            .push(reply["targets"][0].clone());
        assert!(
            publications(&ambiguous, "bundle", &producer, 8, "normalized_source")
                .unwrap_err()
                .to_string()
                .contains("ambiguous duplicate physical print")
        );
    }
}
