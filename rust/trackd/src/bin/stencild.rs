//! One fleet stencil observer on the camera node. Each turn it pairs the
//! overhead RGB-D set with the PoE set sharing its exposure, adds each arm's
//! wrist RGB-D from its owner's capture queryable posed by the measured
//! joints on the bus, hands them to the Python estimator as digest-bound
//! captures, and publishes every visible print's pose as
//! `tatbot.target-pose/1` on `tatbot/tracking/target/<pattern_id>` with the
//! support it rests on. No camera backend, no arm, no motion authority: a
//! print's pose is a number the daemon compares, never a permission.
use anyhow::{Result, ensure};
use clap::Parser;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::{
    collections::{BTreeMap, BTreeSet, VecDeque},
    path::{Path, PathBuf},
    sync::{Arc, Mutex, atomic::AtomicBool},
    time::{Duration, Instant},
};
use tatbot_bus::{
    Envelope, Producer, Stamp, capture::configured_arms, service::ServiceLease, transport::Bus,
};
use tatbot_visiond::{
    CalibrationBundle, FrameRecord, ReceivedFrameSet, RecordedPayload, SensorKind, VisionConfig,
};
use trackd::{evidence::RollingEvidence, worker::Worker};
use zenoh::Wait;

/// Recent sets kept per owner socket while a turn is paced.
const RING: usize = 8;
const SCHEMA: &str = "tatbot.target-pose/1";
const POE: &str = "poe-cameras";
const OVERHEAD: &str = "overhead-depth";
/// The measured-joint topic every arm's runtime publishes (`tatbot.arm-joints/1`);
/// subscribed run-agnostically, so the subscription outlives runs.
const JOINTS_SCHEMA: &str = "tatbot.arm-joints/1";
const JOINTS_KEY: &str = "tatbot/session/*/arm/*/joints";
/// Joint samples kept per arm: 6.4 s at the topic's 10 Hz bound.
const JOINTS_RING: usize = 64;

#[derive(Parser)]
struct Args {
    #[arg(long, required = true)]
    connect: Vec<String>,
    #[arg(long)]
    node: String,
    /// The PoE owner's frame socket: every fixed view each turn. A fleet
    /// without PoE cameras (the demo stack) names none.
    #[arg(long)]
    poe_socket: Option<PathBuf>,
    /// The overhead RGB-D owner's frame socket; without it no print is
    /// measured. With no PoE socket it is the only fixed view.
    #[arg(long)]
    overhead_socket: Option<PathBuf>,
    #[arg(long)]
    calibration: PathBuf,
    /// The D405 registry (`vision.toml`): every camera of group `d405` with a
    /// physical `arm` and an `owner_role` is a wrist view, queried from its
    /// owner's `tatbot/vision/d405/capture/<node>`. Without it no wrist view
    /// enters a turn.
    #[arg(long)]
    vision_config: Option<PathBuf>,
    /// The directory holding each arm's own `arm-registration-<arm>-current.json`
    /// (and the golden `robot-world-current.json` a carried registration
    /// needs): the observer poses a wrist view through it. An arm without one
    /// is refused by name.
    #[arg(long)]
    registrations: Option<PathBuf>,
    /// The URDF the observer's forward kinematics reads (the release's own).
    #[arg(long)]
    urdf: Option<PathBuf>,
    /// One wrist capture per owner per period at most; a joints sample
    /// farther than this from the exposure refuses that view, never the turn.
    #[arg(long, default_value_t = 1000)]
    wrist_capture_ms: u64,
    /// A wrist owner that answers no capture inside this bound refuses its
    /// view this turn.
    #[arg(long, default_value_t = 500)]
    wrist_query_ms: u64,
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
    /// A camera whose fit is reported beside the others but never chosen as
    /// a print's anchor.
    #[arg(long)]
    exclude_anchor: Vec<String>,
    /// Where no fit measures a print, place it by its artwork on the table
    /// plane in the fixed RGB-D view (identity unverified): for a fleet whose
    /// only fixed camera cannot resolve the print (the demo stack's D555).
    #[arg(long)]
    overhead_artwork_match: bool,
    /// One estimator turn per period: SIFT per view is bounded by it.
    #[arg(long, default_value_t = 1000)]
    turn_ms: u64,
    /// A PoE frame pairs with the overhead exposure inside this window.
    #[arg(long, default_value_t = 40.0)]
    pair_window_ms: f64,
    #[arg(long, default_value_t = 3000.0)]
    max_age_ms: f64,
    #[arg(long, default_value_t = 67_108_864)]
    evidence_bytes: u64,
    #[arg(long, default_value_t = 4)]
    evidence_files: usize,
    #[arg(long, default_value_t = 0)]
    max_turns: u64,
}

/// Which recent sets one turn observes.
#[derive(Debug, PartialEq, Eq)]
struct Pairing {
    poe: Option<usize>,
    overhead: Option<usize>,
    /// PoE frames whose exposure lies inside the window of the overhead's.
    paired: usize,
}

fn frame_stamp(frame: &tatbot_visiond::FrameRecord) -> i128 {
    frame
        .metadata
        .timestamps
        .normalized_unix_ns
        .unwrap_or(frame.metadata.timestamps.host_unix_ns)
}

/// The overhead set whose exposure the most PoE cameras share inside the
/// window, with that PoE set; ties go to the newest overhead exposure. With
/// no overhead set the newest PoE set is observed alone; with no PoE owner
/// (`poe` None) the newest overhead set is. Fusion skew is per camera pair,
/// so PoE frames are counted one by one.
fn pair(
    poe: Option<&VecDeque<ReceivedFrameSet>>,
    overhead: &VecDeque<ReceivedFrameSet>,
    window_ns: u128,
) -> Option<Pairing> {
    let Some(poe) = poe else {
        let newest = (0..overhead.len()).max_by_key(|&i| overhead[i].timestamp_ns)?;
        return Some(Pairing {
            poe: None,
            overhead: Some(newest),
            paired: 0,
        });
    };
    if overhead.is_empty() {
        let newest = (0..poe.len()).max_by_key(|&i| poe[i].timestamp_ns)?;
        return Some(Pairing {
            poe: Some(newest),
            overhead: None,
            paired: 0,
        });
    }
    let mut best: Option<(usize, i128, Pairing)> = None;
    for (o, top) in overhead.iter().enumerate() {
        for (p, fixed) in poe.iter().enumerate() {
            let paired = fixed
                .frames
                .iter()
                .filter(|frame| frame_stamp(frame).abs_diff(top.timestamp_ns) <= window_ns)
                .count();
            let key = (paired, top.timestamp_ns);
            if best
                .as_ref()
                .is_none_or(|(count, stamp, _)| key > (*count, *stamp))
            {
                best = Some((
                    paired,
                    top.timestamp_ns,
                    Pairing {
                        poe: Some(p),
                        overhead: Some(o),
                        paired,
                    },
                ));
            }
        }
    }
    best.map(|(_, _, pairing)| pairing)
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

/// The registration files the observer reads at start, folded into the
/// input fingerprint so an adopted registration restarts the estimator on the
/// new world; an absent file is fingerprinted as absent.
fn fingerprint_registrations(digest: &mut Sha256, registrations: Option<&Path>) -> Result<()> {
    let Some(directory) = registrations else {
        return Ok(());
    };
    let arms = configured_arms().map_err(anyhow::Error::msg)?;
    let golden = directory.join("robot-world-current.json");
    for file in arms
        .ids()
        .map(|arm| registration_path(directory, arm))
        .chain(std::iter::once(golden))
    {
        if file.is_file() {
            fingerprint_file(digest, &file)?;
        } else {
            digest.update(format!("{}\tabsent\n", file.display()));
        }
    }
    Ok(())
}

/// Installed references and a fingerprint that changes when any is added,
/// removed or rewritten: the manifest, its image and a coded print's
/// `coded.json` by path, size and mtime. The observer's loader verifies the
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
            let code = entry.join("coded.json");
            if code.is_file() {
                fingerprint_file(&mut digest, &code)?;
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

/// One role's frames as the estimator reads them: `tatbot.session-surface/1` kind `live-capture`,
/// every payload digest-bound, the exposure window the frames span. The PoE
/// role carries decoded fixed images only; the overhead role the aligned
/// RGB-D pair the bundle names.
fn write_capture(
    directory: &Path,
    role: &str,
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
        if role == POE {
            ensure!(
                frame.metadata.sensor_kind == SensorKind::PoE
                    && matches!(frame.payload, RecordedPayload::Video { .. }),
                "fixed tracking camera is not a decoded PoE image"
            );
        } else {
            ensure!(
                ["overhead_depth_color", "overhead_depth_depth"].contains(&name.as_str()),
                "unexpected overhead sensor"
            );
        }
    }
    let (frames, (after_ns, before_ns)) = frame_entries(directory, role, &set.frames)?;
    if role == OVERHEAD {
        tatbot_visiond::frame_bus::validate_capture_geometry(
            &tatbot_visiond::SynchronizedFrameSet {
                sequence: set.sequence,
                timestamp_basis: set.timestamp_basis.clone(),
                timestamp_ns: set.timestamp_ns,
                maximum_skew_ns: set.maximum_skew_ns,
                frames: set
                    .frames
                    .iter()
                    .map(|f| (f.metadata.sensor_name.clone(), f.clone()))
                    .collect(),
            },
        )?;
    }
    let path = directory.join(format!("{role}.json"));
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

/// One measured-joint sample as `tatbot.arm-joints/1` carries it: the driver's
/// six joints, the carriage, the runtime's bound robot-world (null on a launch
/// that bound none) and the worker's measurement stamp.
#[derive(Clone, Debug, Deserialize, PartialEq)]
struct JointsSample {
    arm: String,
    measured_wall_ns: u64,
    joints: Vec<f64>,
    carriage: Option<Carriage>,
    #[serde(default)]
    mode: serde_json::Value,
    calibration_id: Option<String>,
}
#[derive(Clone, Debug, Deserialize, PartialEq)]
struct Carriage {
    position_m: f64,
    effort_n: f64,
}
impl JointsSample {
    fn parse(payload: &serde_json::Value) -> Result<Self> {
        let sample: Self = serde_json::from_value(payload.clone())?;
        configured_arms()
            .map_err(anyhow::Error::msg)?
            .binding(&sample.arm)
            .map_err(anyhow::Error::msg)?;
        ensure!(
            sample.measured_wall_ns > 0
                && sample.joints.len() == 6
                && sample.joints.iter().all(|v| v.is_finite())
                && sample
                    .carriage
                    .as_ref()
                    .is_none_or(|c| c.position_m.is_finite() && c.effort_n.is_finite()),
            "joints sample is not six finite joints with a finite carriage"
        );
        Ok(sample)
    }
}

type JointsRing = Arc<Mutex<BTreeMap<String, VecDeque<JointsSample>>>>;

/// Keep the newest `JOINTS_RING` samples of an arm; a replayed older stamp
/// is kept too (the pairing picks by stamp, never by arrival).
fn retain_joints(ring: &JointsRing, sample: JointsSample) {
    let mut ring = ring.lock().unwrap_or_else(|e| e.into_inner());
    let samples = ring.entry(sample.arm.clone()).or_default();
    samples.push_back(sample);
    while samples.len() > JOINTS_RING {
        samples.pop_front();
    }
}

/// The run-agnostic subscription to every arm's measured joints. A sample
/// under another schema or an unparseable one is counted, never fatal.
fn subscribe_joints(
    bus: &Bus,
    ring: JointsRing,
    refused: Arc<std::sync::atomic::AtomicU64>,
) -> Result<zenoh::pubsub::Subscriber<()>> {
    bus.session
        .declare_subscriber(JOINTS_KEY)
        .callback(move |sample| {
            let bytes = sample.payload().to_bytes();
            match tatbot_bus::transport::decode::<serde_json::Value>(&bytes, JOINTS_SCHEMA)
                .map_err(|e| anyhow::anyhow!("{e}"))
                .and_then(|envelope| JointsSample::parse(&envelope.payload))
            {
                Ok(sample) => retain_joints(&ring, sample),
                Err(_) => {
                    refused.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                }
            }
        })
        .wait()
        .map_err(|e| anyhow::anyhow!("{e}"))
}

/// One wrist camera the observer may pose, from the registry: the arm it is
/// mounted on and the capture owner its frames come from.
#[derive(Clone, Debug, PartialEq)]
struct WristView {
    arm: String,
    camera: String,
    owner_role: String,
    owner: String,
    key: String,
}
impl WristView {
    fn role(&self) -> String {
        self.key.clone()
    }
    fn topic(&self) -> String {
        format!("tatbot/vision/d405/capture/{}", self.owner)
    }
}

/// Every D405 with a physical arm and a capture owner; an owner role the
/// fleet map does not carry refuses the observer at start, never a turn.
fn wrist_views(config: &VisionConfig) -> Result<Vec<WristView>> {
    let mut views = Vec::new();
    let arms = configured_arms().map_err(anyhow::Error::msg)?;
    for camera in &config.cameras.realsense {
        if camera.group != "d405" {
            continue;
        }
        let (Some(arm), Some(owner_role)) = (&camera.arm, &camera.owner_role) else {
            continue;
        };
        arms.binding(arm).map_err(anyhow::Error::msg)?;
        let owner = tatbot_bus::fleet::node_with_role(owner_role).ok_or_else(|| {
            anyhow::anyhow!(
                "{}: no fleet node carries capture role {owner_role}",
                camera.name
            )
        })?;
        views.push(WristView {
            arm: arm.clone(),
            camera: camera.name.clone(),
            owner_role: owner_role.clone(),
            owner: owner.to_owned(),
            key: String::new(),
        });
    }
    let mut counts = BTreeMap::new();
    for view in &views {
        *counts.entry(view.arm.clone()).or_insert(0_usize) += 1;
    }
    for view in &mut views {
        view.key = if counts[&view.arm] == 1 {
            format!("wrist-{}", view.arm)
        } else {
            format!("wrist-{}-{}", view.arm, view.camera)
        };
    }
    Ok(views)
}

/// The joints that pose one wrist exposure: the arm's sample nearest the
/// exposure inside the tolerance, carrying a carriage, bound to no other
/// camera bundle than the observer's. Anything else refuses the view.
fn pose_joints(
    samples: &VecDeque<JointsSample>,
    arm: &str,
    exposure_ns: i128,
    tolerance_ns: i128,
    bundle_id: &str,
) -> Result<(JointsSample, f64)> {
    let nearest = samples
        .iter()
        .min_by_key(|s| i128::from(s.measured_wall_ns).abs_diff(exposure_ns))
        .ok_or_else(|| anyhow::anyhow!("no measured joints of the {arm} arm on the bus"))?;
    let skew_ns = i128::from(nearest.measured_wall_ns).abs_diff(exposure_ns);
    ensure!(
        skew_ns <= tolerance_ns as u128,
        "nearest joints of the {arm} arm are {:.1} ms from the exposure, over {} ms",
        skew_ns as f64 / 1e6,
        tolerance_ns / 1_000_000
    );
    ensure!(
        nearest.carriage.is_some(),
        "joints of the {arm} arm carry no carriage"
    );
    if let Some(bound) = &nearest.calibration_id {
        ensure!(
            bound == bundle_id,
            "the {arm} arm's launch binds camera bundle {bound}, the observer runs {bundle_id}"
        );
    }
    Ok((nearest.clone(), skew_ns as f64 / 1e6))
}

/// The wrist owner's newest RGB-D set, the producer held to the owner role.
fn query_wrist(bus: &Bus, view: &WristView, limit: Duration) -> Result<ReceivedFrameSet> {
    let replies = bus
        .session
        .get(view.topic())
        .timeout(limit)
        .wait()
        .map_err(|e| anyhow::anyhow!("{e}"))?;
    let reply = replies
        .recv_timeout(limit)
        .map_err(|_| anyhow::anyhow!("{}: no reply within {} ms", view.topic(), limit.as_millis()))?
        .ok_or_else(|| anyhow::anyhow!("{}: no capture owner answered", view.topic()))?;
    let sample = reply
        .result()
        .map_err(|e| anyhow::anyhow!("{}: {e}", view.topic()))?;
    let (producer, set) = tatbot_visiond::frame_bus::decode(&sample.payload().to_bytes())?;
    ensure!(
        tatbot_bus::fleet::is_role(&producer.node, &view.owner_role),
        "{}: capture produced by {}, not the {} owner",
        view.topic(),
        producer.node,
        view.owner_role
    );
    Ok(ReceivedFrameSet {
        envelope: None,
        sequence: set.sequence,
        timestamp_basis: set.timestamp_basis,
        timestamp_ns: set.timestamp_ns,
        maximum_skew_ns: set.maximum_skew_ns,
        frames: set.frames.into_values().collect(),
    })
}

/// The wrist capture as the estimator reads it: the arm's colour/depth pair
/// (the owner's active intrinsics ride in the frame metadata), the joints
/// paired with its exposure and the registration file the observer poses it
/// through. Refused by name without the pair, the arm or the registration.
#[allow(clippy::too_many_arguments)]
fn write_wrist_capture(
    directory: &Path,
    view: &WristView,
    set: &ReceivedFrameSet,
    producer: &serde_json::Value,
    calibration: &str,
    joints: &JointsSample,
    skew_ms: f64,
    registration: &Path,
) -> Result<PathBuf> {
    ensure!(
        registration.is_file(),
        "the {} arm has no registration installed beside the observer ({})",
        view.arm,
        registration.display()
    );
    let names = [
        format!("{}_color", view.camera),
        format!("{}_depth", view.camera),
    ];
    let mut frames = Vec::new();
    for name in &names {
        let frame = set
            .frames
            .iter()
            .find(|f| f.metadata.sensor_name == *name)
            .ok_or_else(|| anyhow::anyhow!("wrist capture carries no {name} frame"))?;
        ensure!(
            frame.metadata.sensor_kind == SensorKind::RealSense
                && frame.metadata.attributes.get("physical_arm") == Some(&view.arm),
            "{name}: capture is of the {:?} arm, not {:?}",
            frame.metadata.attributes.get("physical_arm"),
            view.arm
        );
        frames.push(frame.clone());
    }
    ensure!(
        matches!(frames[0].payload, RecordedPayload::Video { .. })
            && matches!(frames[1].payload, RecordedPayload::Depth { .. }),
        "wrist capture is not a decoded colour image and a depth plane"
    );
    let role = view.role();
    let (entries, (after_ns, before_ns)) = frame_entries(directory, &role, &frames)?;
    let carriage = joints
        .carriage
        .as_ref()
        .expect("pose_joints requires a carriage");
    let path = directory.join(format!("{role}.json"));
    std::fs::write(
        &path,
        serde_json::to_vec(&serde_json::json!({
            "schema":"tatbot.session-surface/1", "kind":"live-capture", "producer":producer,
            "geometry_calibration_id":calibration,
            "wrist_capture_window":{"after_ns":after_ns, "before_ns":before_ns}, "frames":entries,
            "wrist":{"arm":view.arm, "camera":view.camera, "joints":joints.joints,
                "carriage_m":carriage.position_m, "carriage_effort_n":carriage.effort_n,
                "measured_wall_ns":joints.measured_wall_ns, "joints_skew_ms":skew_ms,
                "joints_calibration_id":joints.calibration_id, "mode":joints.mode,
                "registration":registration},
        }))?,
    )?;
    Ok(path)
}

/// The registration file the observer poses `arm` through.
fn registration_path(registrations: &Path, arm: &str) -> PathBuf {
    registrations.join(format!("arm-registration-{arm}-current.json"))
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
        // Which wrist views the turn posed (their joint skew and
        // registration provenance) and which it refused, by name.
        if reply["wrist_views"].is_object() {
            support["wrist_views"] = reply["wrist_views"].clone();
        }
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

/// Take a paired set out of its ring, dropping anything older than the
/// chosen exposure so a later turn never observes the past. Called under the
/// lock the pairing was chosen in: an owner push between the two cannot
/// shift the index.
fn take(slots: &mut VecDeque<ReceivedFrameSet>, index: usize) -> (ReceivedFrameSet, usize) {
    let chosen = slots.remove(index).expect("index from the same lock scope");
    let before = slots.len();
    slots.retain(|set| set.timestamp_ns > chosen.timestamp_ns);
    (chosen, before - slots.len())
}

fn main() -> Result<()> {
    let args = Args::parse();
    let arms = configured_arms().map_err(anyhow::Error::msg)?;
    ensure!(
        args.turn_ms > 0
            && args.pair_window_ms.is_finite()
            && args.pair_window_ms > 0.0
            && args.max_age_ms.is_finite()
            && args.max_age_ms > 0.0,
        "invalid observer limits"
    );
    ensure!(
        args.poe_socket.is_some() || args.overhead_socket.is_some(),
        "no fixed camera owner socket: name the PoE socket, the overhead socket or both"
    );
    ensure!(
        args.poe_socket.is_none() || args.overhead_socket != args.poe_socket,
        "the PoE and overhead owners publish distinct sockets"
    );
    ensure!(
        args.wrist_capture_ms > 0 && args.wrist_query_ms > 0,
        "invalid wrist capture limits"
    );
    let _owner = tatbot_visiond::ownership::CameraLease::acquire(
        &std::env::temp_dir().join("tatbot-stencild.lock"),
    )?;
    let calibration = CalibrationBundle::load(&args.calibration)?;
    let wrists = match &args.vision_config {
        Some(path) => wrist_views(&VisionConfig::load(path)?)?,
        None => Vec::new(),
    };
    ensure!(
        wrists.is_empty() || args.registrations.is_some(),
        "wrist views need --registrations (the arms' own registration files)"
    );
    std::fs::create_dir_all(&args.work)?;
    let bus = Bus::open(&args.connect, &[]).map_err(|e| anyhow::anyhow!("{e}"))?;
    let joints: JointsRing = Arc::new(Mutex::new(BTreeMap::new()));
    let joints_refused = Arc::new(std::sync::atomic::AtomicU64::new(0));
    let _joints_subscription = if wrists.is_empty() {
        None
    } else {
        Some(subscribe_joints(
            &bus,
            joints.clone(),
            joints_refused.clone(),
        )?)
    };
    for view in &wrists {
        eprintln!(
            "wrist view: {} arm, {} from the {} owner ({}), poses through {}",
            view.arm,
            view.camera,
            view.owner_role,
            view.topic(),
            registration_path(
                args.registrations.as_deref().unwrap_or(Path::new("")),
                &view.arm
            )
            .display()
        );
    }
    let producer = Producer {
        node: args.node.clone(),
        pid: std::process::id(),
        sha: option_env!("TATBOT_SOURCE_COMMIT")
            .unwrap_or("development")
            .into(),
        run_id: std::env::var("TATBOT_RUN_ID")
            .unwrap_or_else(|_| format!("stencild-{}", std::process::id())),
    };
    let lease = ServiceLease::declare(&bus, producer.clone(), "stencild", vec![SCHEMA.into()])
        .map_err(|e| anyhow::anyhow!("{e}"))?;
    let reader_errors = Arc::new(Mutex::new(BTreeMap::<String, String>::new()));
    let poe = match &args.poe_socket {
        Some(socket) => Some(subscribe(socket, reader_errors.clone())?),
        None => None,
    };
    let poe_delivery = poe.as_ref().map(|(_, delivery)| delivery.clone());
    let poe = poe.map(|(ring, _)| ring);
    let overhead = match &args.overhead_socket {
        Some(socket) => Some(subscribe(socket, reader_errors.clone())?),
        None => None,
    };
    let overhead_delivery = overhead.as_ref().map(|(_, delivery)| delivery.clone());
    let overhead = overhead.map(|(ring, _)| ring);
    let mut output = RollingEvidence::new(
        args.output.clone(),
        args.evidence_bytes,
        args.evidence_files,
    )?;
    let stop = AtomicBool::new(false);
    let owner_json = |role: &str| {
        serde_json::json!({"role": role, "node": tatbot_bus::fleet::node_with_role(role),
            "via": "frame-socket", "observer": producer})
    };
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
    let mut last_paired = 0_usize;
    let mut latencies = VecDeque::new();
    let mut wrist_posed = 0_u64;
    let mut wrist_refused = 0_u64;
    let mut last_wrist_ms = 0.0;
    let mut last_wrist_query: BTreeMap<String, Instant> = BTreeMap::new();
    let wrist_period = Duration::from_millis(args.wrist_capture_ms.max(args.turn_ms));
    let wrist_tolerance_ns = i128::from(args.wrist_capture_ms) * 1_000_000;
    let window_ns = (args.pair_window_ms * 1e6) as u128;
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
        let current = {
            let mut digest = Sha256::new();
            digest.update(current.as_bytes());
            fingerprint_registrations(&mut digest, args.registrations.as_deref())?;
            format!("{:x}", digest.finalize())
        };
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
            // The registration files present now; the fingerprint restarts
            // the estimator when one appears, changes or goes.
            let registrations = args.registrations.as_deref().map(|directory| {
                arms.ids()
                    .map(|arm| (arm.to_owned(), registration_path(directory, arm)))
                    .filter(|(_, path)| path.is_file())
                    .collect::<BTreeMap<_, _>>()
            });
            let golden = args
                .registrations
                .as_deref()
                .map(|directory| directory.join("robot-world-current.json"))
                .filter(|path| path.is_file());
            std::fs::write(
                &binding,
                serde_json::to_vec(&serde_json::json!({
                    "references": paths, "calibration": calibration, "excluded_anchors": args.exclude_anchor,
                    "overhead_artwork_match": args.overhead_artwork_match,
                    "robot_world": {"calibration_id": calibration.bundle_id, "frame": "world",
                        "world_from_base": [[1.0,0.0,0.0,0.0],[0.0,1.0,0.0,0.0],[0.0,0.0,1.0,0.0],[0.0,0.0,0.0,1.0]]},
                    "observer_epoch": format!("{}-{}", std::process::id(), now_unix_ns()?),
                    "evidence_kind": "live-rgbd",
                    "publish_targets_only": true,
                    "wrist": {"registrations": registrations, "robot_world_golden": golden,
                        "urdf": args.urdf, "vision_config": args.vision_config},
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
                ("paired_poe_cameras".into(), last_paired as f64),
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
                ("wrist_views".into(), wrists.len() as f64),
                ("wrist_views_posed".into(), wrist_posed as f64),
                ("wrist_views_refused".into(), wrist_refused as f64),
                ("wrist_query_ms".into(), last_wrist_ms),
                (
                    "joints_samples_refused".into(),
                    joints_refused.load(std::sync::atomic::Ordering::Relaxed) as f64,
                ),
                ("updated_unix_ms".into(), now_unix_ns()? as f64 / 1e6),
            ]);
        }
        let Some(estimator) = worker.as_mut() else {
            std::thread::sleep(Duration::from_millis(args.turn_ms));
            continue;
        };
        // Pair and take under the same locks: the indices the pairing chose
        // are the ones removed, whatever the owners push meanwhile.
        let taken = {
            let mut poe_sets = poe
                .as_ref()
                .map(|ring| ring.lock().unwrap_or_else(|e| e.into_inner()));
            let empty = VecDeque::new();
            let mut top = overhead
                .as_ref()
                .map(|ring| ring.lock().unwrap_or_else(|e| e.into_inner()));
            pair(
                poe_sets.as_deref(),
                top.as_deref().unwrap_or(&empty),
                window_ns,
            )
            .map(|pairing| {
                let poe_set = pairing
                    .poe
                    .zip(poe_sets.as_deref_mut())
                    .map(|(index, ring)| take(ring, index));
                let overhead_set = pairing
                    .overhead
                    .zip(top.as_deref_mut())
                    .map(|(index, ring)| take(ring, index));
                (pairing, poe_set, overhead_set)
            })
        };
        let Some((pairing, poe_set, overhead_set)) = taken else {
            std::thread::sleep(Duration::from_millis(5));
            continue;
        };
        let poe_set = poe_set.map(|(set, skipped)| {
            if let Some(delivery) = &poe_delivery {
                delivery
                    .lock()
                    .unwrap_or_else(|e| e.into_inner())
                    .selected(&set, skipped);
            }
            set
        });
        let overhead_set = overhead_set.map(|(set, skipped)| {
            if let Some(delivery) = &overhead_delivery {
                delivery
                    .lock()
                    .unwrap_or_else(|e| e.into_inner())
                    .selected(&set, skipped);
            }
            set
        });
        let pair_skew_by_camera_ms = overhead_set.as_ref().zip(poe_set.as_ref()).map(|(top, poe_set)| {
            poe_set
                .frames
                .iter()
                .map(|frame| {
                    (
                        frame.metadata.sensor_name.clone(),
                        frame_stamp(frame).abs_diff(top.timestamp_ns) as f64 / 1e6,
                    )
                })
                .collect::<BTreeMap<_, _>>()
        });
        last_paired = pairing.paired;
        let Some(newest) = overhead_set
            .as_ref()
            .or(poe_set.as_ref())
            .map(|set| set.timestamp_ns)
        else {
            continue;
        };
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
        let mut roles = Vec::new();
        match poe_set.as_ref() {
            Some(set) => roles.push((POE, set)),
            None => {
                errors.insert(POE.into(), "no PoE owner socket".into());
            }
        }
        match overhead_set.as_ref() {
            Some(set) => roles.push((OVERHEAD, set)),
            None => {
                errors.insert(OVERHEAD.into(), "no overhead owner socket".into());
            }
        }
        for (role, set) in roles {
            match write_capture(
                &capture,
                role,
                set,
                &owner_json(role),
                &calibration.bundle_id,
            ) {
                Ok(path) => {
                    captures.insert(role.into(), serde_json::json!(path));
                }
                Err(error) => {
                    errors.insert(role.into(), serde_json::json!(format!("{error:#}")));
                }
            }
        }
        // Each arm's wrist view: one query per owner per period at most,
        // only for an arm whose joints are on the bus, posed by the sample
        // nearest the exposure. A refusal names the view, never the turn.
        let wrist_started = Instant::now();
        let mut wrist_turn = BTreeMap::new();
        for view in &wrists {
            if last_wrist_query.get(&view.key).is_some_and(|last| {
                last.elapsed() + Duration::from_millis(args.turn_ms / 2) < wrist_period
            }) {
                continue;
            }
            last_wrist_query.insert(view.key.clone(), Instant::now());
            let registration = registration_path(
                args.registrations.as_deref().unwrap_or(Path::new("")),
                &view.arm,
            );
            let posed = (|| -> Result<PathBuf> {
                ensure!(
                    registration.is_file(),
                    "the {} arm has no registration installed beside the observer ({})",
                    view.arm,
                    registration.display()
                );
                let live = {
                    let ring = joints.lock().unwrap_or_else(|e| e.into_inner());
                    ring.get(&view.arm).is_some_and(|samples| {
                        samples.back().is_some_and(|newest| {
                            now_unix_ns().is_ok_and(|now| {
                                i128::from(newest.measured_wall_ns).abs_diff(now)
                                    <= wrist_tolerance_ns as u128
                            })
                        })
                    })
                };
                ensure!(
                    live,
                    "no measured joints of the {} arm on the bus inside {} ms",
                    view.arm,
                    args.wrist_capture_ms
                );
                let set = query_wrist(&bus, view, Duration::from_millis(args.wrist_query_ms))?;
                let (sample, skew_ms) = {
                    let ring = joints.lock().unwrap_or_else(|e| e.into_inner());
                    let empty = VecDeque::new();
                    pose_joints(
                        ring.get(&view.arm).unwrap_or(&empty),
                        &view.arm,
                        set.timestamp_ns,
                        wrist_tolerance_ns,
                        &calibration.bundle_id,
                    )?
                };
                write_wrist_capture(
                    &capture,
                    view,
                    &set,
                    &serde_json::json!({"role": view.owner_role, "node": view.owner,
                        "via": "capture-queryable", "topic": view.topic(), "observer": producer}),
                    &calibration.bundle_id,
                    &sample,
                    skew_ms,
                    &registration,
                )
            })();
            match posed {
                Ok(path) => {
                    wrist_posed += 1;
                    captures.insert(view.role(), serde_json::json!(path));
                    wrist_turn.insert(view.role(), serde_json::json!({"posed": true}));
                }
                Err(error) => {
                    wrist_refused += 1;
                    let reason = format!("{error:#}");
                    wrist_turn.insert(view.role(), serde_json::json!({"refused": reason}));
                    errors.insert(view.role(), serde_json::json!(reason));
                }
            }
        }
        last_wrist_ms = wrist_started.elapsed().as_secs_f64() * 1000.0;
        let basis = overhead_set
            .as_ref()
            .or(poe_set.as_ref())
            .map(|s| s.timestamp_basis.clone())
            .unwrap_or_default();
        let request =
            serde_json::json!({"captures": captures, "errors": errors, "accepted_scan": null});
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
                    "poe": poe_delivery.as_ref().map(|d| d.lock().unwrap_or_else(|e| e.into_inner()).clone()),
                    "overhead": overhead_delivery.as_ref().map(|d| d.lock().unwrap_or_else(|e| e.into_inner()).clone()),
                    "pair_skew_by_camera_ms": pair_skew_by_camera_ms,
                    "paired_poe_cameras": pairing.paired,
                    "wrist": wrist_turn, "wrist_query_ms": last_wrist_ms,
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
        FrameMetadata, FrameRecord, FrameTimestamps, PixelFormat, StreamProfile,
        time::TimestampDomain,
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

    fn poe_set(sequence: u64, stamps: &[i128]) -> ReceivedFrameSet {
        let frames = stamps
            .iter()
            .enumerate()
            .map(|(i, &stamp)| frame(&format!("camera{}", i + 1), SensorKind::PoE, stamp, video()))
            .collect();
        set(sequence, stamps[0], frames)
    }

    #[test]
    fn delivery_counts_unique_camera_exposures_and_skipped_sets() {
        let mut delivery = SocketDelivery::default();
        let mut first = set(
            1,
            100,
            vec![frame("camera1", SensorKind::PoE, 100, video())],
        );
        first.frames[0].metadata.sequence = 10;
        let mut repeated = set(
            2,
            100,
            vec![frame("camera1", SensorKind::PoE, 100, video())],
        );
        repeated.frames[0].metadata.sequence = 10;
        let mut next = set(
            3,
            300,
            vec![frame("camera1", SensorKind::PoE, 300, video())],
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
    fn pairing_keeps_the_overhead_set_with_the_most_poe_cameras_inside_the_window() {
        let window = 40_000_000;
        let poe = VecDeque::from([
            poe_set(1, &[1_000_000_000, 1_010_000_000, 1_020_000_000]),
            poe_set(2, &[1_100_000_000, 1_150_000_000, 1_200_000_000]),
        ]);
        // The older overhead exposure sits inside 40 ms of all three fixed
        // frames of the first set; the newer only of one frame of the second.
        let overhead = VecDeque::from([
            overhead_set(1, 1_030_000_000),
            overhead_set(2, 1_110_000_000),
        ]);
        assert_eq!(
            pair(Some(&poe), &overhead, window),
            Some(Pairing {
                poe: Some(0),
                overhead: Some(0),
                paired: 3
            })
        );
        // Equal pairing goes to the newest overhead exposure.
        let poe = VecDeque::from([
            poe_set(1, &[1_000_000_000, 1_010_000_000, 1_020_000_000]),
            poe_set(2, &[1_120_000_000, 1_150_000_000, 1_180_000_000]),
        ]);
        let overhead = VecDeque::from([
            overhead_set(1, 1_020_000_000),
            overhead_set(2, 1_150_000_000),
        ]);
        assert_eq!(
            pair(Some(&poe), &overhead, window),
            Some(Pairing {
                poe: Some(1),
                overhead: Some(1),
                paired: 3
            })
        );
        // No overhead: the newest PoE set alone, nothing paired.
        assert_eq!(
            pair(Some(&poe), &VecDeque::new(), window),
            Some(Pairing {
                poe: Some(1),
                overhead: None,
                paired: 0
            })
        );
        // A PoE owner that has sent nothing yet holds the turn.
        assert_eq!(pair(Some(&VecDeque::new()), &overhead, window), None);
        // No PoE owner at all (the demo stack): the newest overhead set alone.
        assert_eq!(
            pair(None, &overhead, window),
            Some(Pairing {
                poe: None,
                overhead: Some(1),
                paired: 0
            })
        );
        assert_eq!(pair(None, &VecDeque::new(), window), None);
    }

    #[test]
    fn capture_manifest_binds_role_frames_exposure_window_and_payload_digests() {
        let directory = tempfile::tempdir().unwrap();
        let producer = serde_json::json!({"role": POE});
        let path = write_capture(
            directory.path(),
            POE,
            &poe_set(3, &[1_000_000_000, 1_010_000_000]),
            &producer,
            "bundle",
        )
        .unwrap();
        let manifest: serde_json::Value =
            serde_json::from_slice(&std::fs::read(&path).unwrap()).unwrap();
        assert_eq!(manifest["schema"], "tatbot.session-surface/1");
        assert_eq!(manifest["kind"], "live-capture");
        assert_eq!(manifest["geometry_calibration_id"], "bundle");
        assert_eq!(
            manifest["wrist_capture_window"],
            serde_json::json!({"after_ns": 1_000_000_000_i64, "before_ns": 1_010_000_000_i64})
        );
        let entry = &manifest["frames"]["camera2"];
        let bytes = std::fs::read(
            directory
                .path()
                .join(entry["payload_file"].as_str().unwrap()),
        )
        .unwrap();
        assert_eq!(bytes, vec![7; 12]);
        assert_eq!(entry["payload_bytes"], 12);
        assert_eq!(entry["sha256"], format!("{:x}", Sha256::digest(&bytes)));
        assert_eq!(entry["metadata"]["sensor_name"], "camera2");
        // The overhead pair passes its geometry check; a depth plane on the
        // PoE role and a foreign bundle are refused.
        write_capture(
            directory.path(),
            OVERHEAD,
            &overhead_set(4, 1_000_000_000),
            &producer,
            "bundle",
        )
        .unwrap();
        let mut wrong_role = overhead_set(5, 1_000_000_000);
        wrong_role.frames[0].metadata.sensor_kind = SensorKind::PoE;
        assert!(write_capture(directory.path(), POE, &wrong_role, &producer, "bundle").is_err());
        assert!(
            write_capture(
                directory.path(),
                POE,
                &poe_set(6, &[1_000_000_000]),
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
        // A coded print's code beside the manifest is part of the bank.
        std::fs::write(seed.join("coded.json"), b"{}").unwrap();
        let (paths, coded) = scan_references(directory.path()).unwrap();
        assert_eq!(paths, vec![seed.join("tracking.json")]);
        assert_ne!(coded, rewritten);
        std::fs::remove_dir_all(&seed).unwrap();
        assert_eq!(scan_references(directory.path()).unwrap().1, idle);
    }

    fn joints(arm: &str, stamp: u64, calibration: Option<&str>) -> JointsSample {
        JointsSample::parse(&serde_json::json!({
            "arm": arm, "measured_wall_ns": stamp, "joints": [0.1, -0.2, 0.3, -0.4, 0.5, -0.6],
            "carriage": {"position_m": 0.002, "effort_n": 0.5}, "mode": "Position",
            "calibration_id": calibration,
        }))
        .unwrap()
    }

    fn wrist_set(arm: &str, stamp: i128) -> ReceivedFrameSet {
        let mut color = frame("realsense1_color", SensorKind::RealSense, stamp, video());
        let mut depth = frame(
            "realsense1_depth",
            SensorKind::RealSense,
            stamp,
            RecordedPayload::Depth {
                width: 2,
                height: 2,
                bytes: vec![0; 8],
            },
        );
        for record in [&mut color, &mut depth] {
            record.metadata.calibration_id = None;
            for (key, value) in [
                ("physical_arm", arm),
                ("device_serial", "d405"),
                ("capture_epoch", "e"),
                ("intrinsics", "{}"),
            ] {
                record.metadata.attributes.insert(key.into(), value.into());
            }
        }
        depth
            .metadata
            .attributes
            .insert("aligned_to".into(), "realsense1_color".into());
        set(9, stamp, vec![color, depth])
    }

    #[test]
    fn two_wrist_cameras_on_one_arm_keep_distinct_capture_roles() {
        let mut config = VisionConfig::load(
            Path::new(env!("CARGO_MANIFEST_DIR")).join("../visiond/config/vision.example.toml"),
        )
        .unwrap();
        let mut second = config.cameras.realsense[0].clone();
        second.name = "realsense2".into();
        second.serial = "second-device".into();
        config.cameras.realsense.push(second);
        config.validate().unwrap();
        let views = wrist_views(&config).unwrap();
        assert_eq!(views.len(), 2);
        assert_eq!(views[0].arm, "right");
        assert_eq!(views[1].arm, "right");
        assert_eq!(views[0].role(), "wrist-right-realsense1");
        assert_eq!(views[1].role(), "wrist-right-realsense2");
    }

    #[test]
    fn a_wrist_view_enters_the_turn_only_with_fresh_joints_and_the_arms_registration() {
        // The registry names the wrist views: the arm, the camera and the
        // owner the fleet map resolves; the example registry has one.
        let config = VisionConfig::load(
            Path::new(env!("CARGO_MANIFEST_DIR")).join("../visiond/config/vision.example.toml"),
        )
        .unwrap();
        let views = wrist_views(&config).unwrap();
        assert_eq!(views.len(), 1);
        let view = &views[0];
        assert_eq!(
            (view.arm.as_str(), view.camera.as_str()),
            ("right", "realsense1")
        );
        assert_eq!(view.owner_role, "realsense");
        assert_eq!(
            view.owner,
            tatbot_bus::fleet::node_with_role("realsense").unwrap()
        );
        assert_eq!(view.role(), "wrist-right");
        assert_eq!(
            view.topic(),
            format!("tatbot/vision/d405/capture/{}", view.owner)
        );
        // A sample is six finite joints, a carriage and an arm; anything
        // else never enters the ring.
        assert!(
            JointsSample::parse(&serde_json::json!({"arm": "centre", "measured_wall_ns": 1,
            "joints": vec![0.0; 6], "carriage": null, "calibration_id": null}))
            .is_err()
        );
        assert!(
            JointsSample::parse(&serde_json::json!({"arm": "right", "measured_wall_ns": 1,
            "joints": vec![0.0; 7], "carriage": null, "calibration_id": null}))
            .is_err()
        );
        let ring: JointsRing = Arc::new(Mutex::new(BTreeMap::new()));
        for stamp in [1_000_000_000, 1_100_000_000, 1_200_000_000] {
            retain_joints(&ring, joints("right", stamp, Some("bundle")));
        }
        retain_joints(&ring, joints("left", 1_150_000_000, None));
        let ring = ring.lock().unwrap();
        // The sample nearest the exposure poses it, its skew stated.
        let (sample, skew_ms) = pose_joints(
            &ring["right"],
            "right",
            1_130_000_000,
            1_000_000_000,
            "bundle",
        )
        .unwrap();
        assert_eq!((sample.measured_wall_ns, skew_ms), (1_100_000_000, 30.0));
        // Joints farther than one capture interval refuse the view, by name;
        // an arm with no joints on the bus too; a launch bound to another
        // camera bundle too, naming both; one bound to none is admitted.
        let stale = pose_joints(
            &ring["right"],
            "right",
            3_000_000_000,
            1_000_000_000,
            "bundle",
        )
        .unwrap_err()
        .to_string();
        assert_eq!(
            stale,
            "nearest joints of the right arm are 1800.0 ms from the exposure, over 1000 ms"
        );
        assert!(
            pose_joints(
                &VecDeque::new(),
                "left",
                1_000_000_000,
                1_000_000_000,
                "bundle"
            )
            .unwrap_err()
            .to_string()
            .contains("no measured joints of the left arm")
        );
        let other = pose_joints(
            &ring["right"],
            "right",
            1_130_000_000,
            1_000_000_000,
            "other",
        )
        .unwrap_err()
        .to_string();
        assert!(other.contains("binds camera bundle bundle") && other.contains("runs other"));
        assert_eq!(
            pose_joints(
                &ring["left"],
                "left",
                1_130_000_000,
                1_000_000_000,
                "bundle"
            )
            .unwrap()
            .0
            .calibration_id,
            None
        );
        let mut no_carriage = joints("right", 1_130_000_000, Some("bundle"));
        no_carriage.carriage = None;
        assert!(
            pose_joints(
                &VecDeque::from([no_carriage]),
                "right",
                1_130_000_000,
                1_000_000_000,
                "bundle"
            )
            .unwrap_err()
            .to_string()
            .contains("carry no carriage")
        );
        // The capture the estimator reads: the pair, the joints that pose it
        // and the registration it is posed through; refused without the
        // registration, with the other arm's camera, or without the depth.
        let directory = tempfile::tempdir().unwrap();
        let registrations = directory.path().join("vision");
        std::fs::create_dir(&registrations).unwrap();
        let registration = registration_path(&registrations, "right");
        let producer = serde_json::json!({"role": "realsense"});
        let missing = write_wrist_capture(
            directory.path(),
            view,
            &wrist_set("right", 1_130_000_000),
            &producer,
            "bundle",
            &sample,
            skew_ms,
            &registration,
        )
        .unwrap_err()
        .to_string();
        assert!(
            missing.starts_with("the right arm has no registration installed"),
            "{missing}"
        );
        std::fs::write(&registration, b"{}").unwrap();
        let path = write_wrist_capture(
            directory.path(),
            view,
            &wrist_set("right", 1_130_000_000),
            &producer,
            "bundle",
            &sample,
            skew_ms,
            &registration,
        )
        .unwrap();
        let manifest: serde_json::Value =
            serde_json::from_slice(&std::fs::read(&path).unwrap()).unwrap();
        assert_eq!(manifest["schema"], "tatbot.session-surface/1");
        assert_eq!(manifest["kind"], "live-capture");
        assert_eq!(manifest["geometry_calibration_id"], "bundle");
        assert_eq!(
            manifest["wrist_capture_window"],
            serde_json::json!({"after_ns": 1_130_000_000_i64, "before_ns": 1_130_000_000_i64})
        );
        let wrist = &manifest["wrist"];
        assert_eq!(wrist["arm"], "right");
        assert_eq!(wrist["camera"], "realsense1");
        assert_eq!(
            wrist["joints"],
            serde_json::json!([0.1, -0.2, 0.3, -0.4, 0.5, -0.6])
        );
        assert_eq!(wrist["carriage_m"], 0.002);
        assert_eq!(wrist["measured_wall_ns"], 1_100_000_000_u64);
        assert_eq!(wrist["joints_skew_ms"], 30.0);
        assert_eq!(wrist["joints_calibration_id"], "bundle");
        assert_eq!(wrist["registration"], serde_json::json!(registration));
        let entries = manifest["frames"].as_object().unwrap();
        assert_eq!(
            entries.keys().collect::<Vec<_>>(),
            ["realsense1_color", "realsense1_depth"]
        );
        assert_eq!(
            entries["realsense1_color"]["metadata"]["attributes"]["intrinsics"],
            "{}"
        );
        assert!(
            write_wrist_capture(
                directory.path(),
                view,
                &wrist_set("left", 1_130_000_000),
                &producer,
                "bundle",
                &sample,
                skew_ms,
                &registration,
            )
            .unwrap_err()
            .to_string()
            .contains("not \"right\"")
        );
        let mut colour_only = wrist_set("right", 1_130_000_000);
        colour_only.frames.pop();
        assert!(
            write_wrist_capture(
                directory.path(),
                view,
                &colour_only,
                &producer,
                "bundle",
                &sample,
                skew_ms,
                &registration,
            )
            .unwrap_err()
            .to_string()
            .contains("no realsense1_depth frame")
        );
        // The registration files are part of the input fingerprint: a
        // rewritten one restarts the estimator on the new world.
        let before = {
            let mut digest = Sha256::new();
            fingerprint_registrations(&mut digest, Some(&registrations)).unwrap();
            format!("{:x}", digest.finalize())
        };
        std::fs::File::options()
            .write(true)
            .open(&registration)
            .unwrap()
            .set_modified(std::time::SystemTime::now() + Duration::from_secs(5))
            .unwrap();
        let after = {
            let mut digest = Sha256::new();
            fingerprint_registrations(&mut digest, Some(&registrations)).unwrap();
            format!("{:x}", digest.finalize())
        };
        assert_ne!(before, after);
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
            "wrist_views": {"wrist_right": {"joints_skew_ms": 30.0}, "wrist_left": {"refused": "no measured joints of the left arm"}},
            "targets": [
            {"pattern_id": pattern, "reference_id": "e".repeat(64), "source": "measured",
             "world_from_target": identity, "translation_sigma_m": 0.001, "rotation_sigma_rad": 0.01,
             "capture_ns": 1_000_000_000_u64,
             "support": {"anchor_camera": "camera4", "anchors": 40, "motion_authority": true}},
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
        assert_eq!(payload["support"]["anchor_camera"], "camera4");
        // The turn's wrist views ride every print's support: posed with
        // their joint skew, or refused by name.
        assert_eq!(
            payload["support"]["wrist_views"]["wrist_right"]["joints_skew_ms"],
            30.0
        );
        assert_eq!(
            payload["support"]["wrist_views"]["wrist_left"]["refused"],
            "no measured joints of the left arm"
        );
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
