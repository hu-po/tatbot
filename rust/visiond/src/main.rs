use std::path::PathBuf;

#[cfg(feature = "rerun")]
use std::net::UdpSocket;

use anyhow::Result;
use clap::{Parser, Subcommand};
#[cfg(any(feature = "rerun", feature = "fiducials"))]
use serde::Deserialize;
#[cfg(any(
    feature = "gstreamer",
    feature = "realsense",
    feature = "rerun",
    feature = "fiducials"
))]
use serde::Serialize;
use tatbot_visiond::{
    CalibrationBundle, VisionConfig, pairwise_sync_report, read_recording_entries,
};

#[cfg(any(feature = "gstreamer", feature = "realsense"))]
use tatbot_visiond::FrameSynchronizer;
#[cfg(any(feature = "gstreamer", feature = "realsense"))]
use tatbot_visiond::UnixFramePublisher;
use tracing_subscriber::EnvFilter;

#[cfg(feature = "gstreamer")]
use std::env;

#[cfg(any(feature = "rerun", feature = "fiducials"))]
use std::fs::File;
#[cfg(any(feature = "rerun", feature = "fiducials"))]
use std::io::{BufRead, BufReader};
#[cfg(any(feature = "gstreamer", feature = "realsense"))]
use std::sync::mpsc;
#[cfg(any(feature = "gstreamer", feature = "realsense", feature = "fiducials"))]
use std::{
    fs::OpenOptions,
    io::{BufWriter, Write},
};

#[cfg(any(
    feature = "gstreamer",
    feature = "realsense",
    feature = "rerun",
    feature = "fiducials"
))]
use std::collections::BTreeMap;
#[cfg(any(feature = "rerun", feature = "gstreamer"))]
use std::time::{SystemTime, UNIX_EPOCH};
#[cfg(any(feature = "gstreamer", feature = "realsense", feature = "rerun"))]
use std::{thread, time::Duration};

#[cfg(any(
    feature = "gstreamer",
    feature = "realsense",
    feature = "rerun",
    feature = "fiducials"
))]
use std::time::Instant;

use anyhow::Context;

#[cfg(feature = "gstreamer")]
use tatbot_visiond::gstreamer_backend::{PoeRtspCapture, PoeStream};

#[cfg(any(feature = "gstreamer", feature = "realsense"))]
use tatbot_visiond::EvidenceRecorder;

#[cfg(feature = "realsense")]
use tatbot_visiond::realsense_backend::RealsenseCapture;

#[cfg(all(feature = "fiducials", feature = "gstreamer"))]
use tatbot_visiond::DetectionRoi;
#[cfg(all(feature = "rerun", any(feature = "gstreamer", feature = "realsense")))]
use tatbot_visiond::RerunSink;
#[cfg(any(feature = "rerun", feature = "fiducials"))]
use tatbot_visiond::SensorKind;
#[cfg(any(feature = "gstreamer", feature = "rerun", feature = "fiducials"))]
use tatbot_visiond::SynchronizedFrameSet;
#[cfg(all(feature = "fiducials", feature = "gstreamer"))]
use tatbot_visiond::expanded_detection_roi;
#[cfg(any(feature = "rerun", feature = "fiducials"))]
use tatbot_visiond::read_recording_frame;
#[cfg(feature = "fiducials")]
use tatbot_visiond::{
    AprilTagDetectorFactory, EstimatorConfig, FiducialDetection, FiducialInventory, RustEeTracker,
    WristLayout,
};
#[cfg(feature = "rerun")]
use tatbot_visiond::{LiveTeleopTick, RerunLayout, RerunViewer, TeleopSetup};
#[cfg(any(feature = "gstreamer", feature = "rerun"))]
use tatbot_visiond::{PixelFormat, RecordedPayload};

#[derive(Debug, Parser)]
#[command(name = "tatbot-visiond", about = "Tatbot 2.0 vision capture service")]
struct Cli {
    #[arg(long, default_value = "info", env = "RUST_LOG")]
    log_filter: String,
    #[command(subcommand)]
    command: Command,
}

#[derive(Debug, Subcommand)]
enum Command {
    /// Parse and validate a vision configuration without opening hardware.
    ValidateConfig { config: PathBuf },
    /// Print the configured sensor names and profiles.
    DescribeConfig { config: PathBuf },
    /// Parse, validate, and verify a versioned calibration bundle.
    ValidateCalibration { bundle: PathBuf },
    /// Compute and stamp the content-addressed bundle_id of a draft
    /// calibration bundle (written by external calibration tooling), then
    /// fully validate the result.
    FinalizeCalibration {
        draft: PathBuf,
        /// Output path; defaults to overwriting the draft in place.
        #[arg(long)]
        output: Option<PathBuf>,
    },
    /// Compare two recorded JSONL streams using their retained timestamps.
    AnalyzeSync { reference: PathBuf, other: PathBuf },
    /// Detect configured AprilTags in an existing decoded evidence capture.
    #[cfg(feature = "fiducials")]
    DetectFiducials {
        evidence: PathBuf,
        #[arg(long)]
        inventory: PathBuf,
        #[arg(long)]
        calibration: PathBuf,
        #[arg(long)]
        output: PathBuf,
        /// Restrict to one configured target (e.g. wrist); default is all mounted ids.
        #[arg(long)]
        target: Option<String>,
        #[arg(long)]
        scale: Option<f64>,
        #[arg(long, default_value_t = 0)]
        max_sets: usize,
    },
    /// Re-run the Rust EE pose solver on retained live detections without images or hardware.
    #[cfg(feature = "fiducials")]
    ReplayEeDetections {
        /// estimates.jsonl emitted by capture-poe-all --fiducial-output.
        input: PathBuf,
        #[arg(long)]
        inventory: PathBuf,
        #[arg(long)]
        calibration: PathBuf,
        #[arg(long)]
        wrist_layout: PathBuf,
        #[arg(long)]
        output: PathBuf,
        /// Remove a camera before solving; repeat for leave-many-out studies.
        #[arg(long)]
        exclude_camera: Vec<String>,
        #[arg(long)]
        max_source_rmse_px: Option<f64>,
        #[arg(long)]
        max_total_rmse_px: Option<f64>,
        #[arg(long)]
        max_translation_sigma_mm: Option<f64>,
        #[arg(long)]
        max_rotation_sigma_deg: Option<f64>,
    },
    /// Replay a synchronized evidence capture into a Rerun recording or viewer.
    #[cfg(feature = "rerun")]
    ReplayRerun {
        /// Evidence roots containing synchronized_frames.jsonl and sensor directories.
        /// Optional when --teleop-log is given.
        #[arg(value_name = "RECORDING_ROOT")]
        recording_roots: Vec<PathBuf>,
        /// Reconstruction dataset containing pointclouds/ and metadata/.
        #[arg(long, value_name = "DATASET_DIR")]
        reconstruction_dir: Option<PathBuf>,
        /// URDF whose visual meshes should be added to the 3D scene.
        #[arg(long, value_name = "URDF")]
        urdf: Option<PathBuf>,
        /// Calibration bundle whose camera frustums should be drawn in 3D.
        #[arg(long, value_name = "BUNDLE")]
        calibration: Option<PathBuf>,
        /// URDF link the calibration world frame is anchored to.
        #[arg(long, default_value = "palette_tag8")]
        calibration_anchor: String,
        /// robot_world.json from solve_robot_world.py: places the calibration
        /// frame by the MEASURED world_from_base, overriding the anchor guess.
        #[arg(long, value_name = "JSON")]
        robot_world: Option<PathBuf>,
        /// A wxai_teleop flight log (.wxtl) to replay: animates the URDF arms
        /// and logs teleop timing/tracking time series.
        #[arg(long, value_name = "WXTL")]
        teleop_log: Option<PathBuf>,
        /// Which URDF arm the teleop leader drove ("left" or "right").
        #[arg(long, default_value = "left")]
        teleop_leader: String,
        /// Rate at which animated link transforms are logged; scalar series
        /// keep the full recorded tick rate.
        #[arg(long, default_value_t = 60.0)]
        teleop_fps: f64,
        /// Write an .rrd file instead of spawning the local viewer.
        #[arg(long)]
        output: Option<PathBuf>,
        /// Spawn/connect to a local `rerun` viewer.
        #[arg(long)]
        spawn: bool,
        /// Pace replay according to capture timestamps.
        #[arg(long)]
        realtime: bool,
        #[arg(long, default_value_t = 1.0)]
        speed: f64,
        /// Per evidence source visualization rate. 0 retains every set.
        #[arg(long, default_value_t = 0.0)]
        max_fps: f64,
        /// Uniform color-image scale for the visualization derivative.
        #[arg(long, default_value_t = 1.0)]
        image_scale: f64,
        #[arg(long, default_value_t = 85)]
        jpeg_quality: u8,
        /// Explicit viewer workflow. By default the recorded sensor families
        /// and presence of teleop data select a layout with no empty panels.
        #[arg(long, value_enum)]
        rerun_layout: Option<RerunLayout>,
    },
    /// Bridge decimated, nonblocking wxai_teleop UDP telemetry into Rerun.
    /// This process never opens an arm connection or participates in control.
    #[cfg(feature = "rerun")]
    StreamTeleop {
        // Localhost by default (plan Phase 3): receiving joint state from
        // another node is an explicit deployment decision (--bind 0.0.0.0:9878).
        #[arg(long, default_value = "127.0.0.1:9878")]
        bind: String,
        #[arg(long)]
        connect: Option<String>,
        /// Save a standalone live joint recording instead of connecting.
        #[arg(long, value_name = "RRD")]
        output: Option<PathBuf>,
        #[arg(long)]
        recording_id: String,
        #[arg(long, value_name = "URDF")]
        urdf: PathBuf,
        /// Draw calibrated camera frustums and align their world frame.
        #[arg(long, value_name = "BUNDLE")]
        calibration: Option<PathBuf>,
        #[arg(long, default_value = "palette_tag8")]
        calibration_anchor: String,
        /// Measured robot/world alignment, preferred over the URDF anchor.
        #[arg(long, value_name = "JSON")]
        robot_world: Option<PathBuf>,
        /// Omit when another producer owns the shared recording blueprint.
        #[arg(long, value_enum)]
        rerun_layout: Option<RerunLayout>,
        #[arg(long, default_value = "left")]
        leader_prefix: String,
        #[arg(long, default_value = "right")]
        follower_prefix: String,
        /// Stop after this duration; 0 runs until interrupted.
        #[arg(long, default_value_t = 0)]
        duration_seconds: u64,
        /// Report missing telemetry at this age; 0 disables the warning.
        #[arg(long, default_value_t = 3.0)]
        idle_timeout_seconds: f64,
    },
    /// Capture one live PoE stream into the evidence format.
    #[cfg(feature = "gstreamer")]
    CapturePoe {
        config: PathBuf,
        #[arg(long)]
        sensor: String,
        #[arg(long, default_value = "main")]
        stream: String,
        #[arg(long, default_value_t = 10)]
        duration_seconds: u64,
        /// Decode H.264 into BGR pixels before recording or transport.
        #[arg(long)]
        decoded: bool,
        /// Drop delta frames before decode; ~2 Hz I-frames at GOP=10.
        #[arg(long)]
        keyframes_only: bool,
        #[arg(long)]
        output: Option<PathBuf>,
        #[arg(long)]
        calibration: Option<PathBuf>,
    },
    /// Capture the configured PoE cameras concurrently into one evidence set.
    #[cfg(feature = "gstreamer")]
    CapturePoeAll {
        config: PathBuf,
        #[arg(long, default_value = "main")]
        stream: String,
        #[arg(long, default_value_t = 10)]
        duration_seconds: u64,
        /// Decode H.264 into BGR pixels before recording or transport.
        #[arg(long)]
        decoded: bool,
        /// Drop delta frames before decode; ~2 Hz I-frames at GOP=10.
        #[arg(long)]
        keyframes_only: bool,
        #[arg(long)]
        output: Option<PathBuf>,
        #[arg(long)]
        calibration: Option<PathBuf>,
        /// Canonical fiducial inventory. Enables in-process AprilTag detection.
        #[cfg(feature = "fiducials")]
        #[arg(long)]
        fiducial_inventory: Option<PathBuf>,
        /// Calibrated wrist layout. Enables EE pose estimation; omitted means detection-only.
        #[cfg(feature = "fiducials")]
        #[arg(long)]
        wrist_layout: Option<PathBuf>,
        /// Write detection batches or EE pose estimates as JSONL.
        #[cfg(feature = "fiducials")]
        #[arg(long)]
        fiducial_output: Option<PathBuf>,
        /// Optional detector image scale in (0, 1].
        #[cfg(feature = "fiducials")]
        #[arg(long)]
        fiducial_scale: Option<f64>,
        /// Cap expensive fiducial passes per second. 0 processes every synchronized set.
        #[cfg(feature = "fiducials")]
        #[arg(long, default_value_t = 0.0)]
        fiducial_max_fps: f64,
        /// Minimum fresh cameras in a bounded partial tracker set. Complete
        /// sets still emit immediately. Applies only to tracker-only no-record runs.
        #[cfg(feature = "fiducials")]
        #[arg(long, default_value_t = 3)]
        fiducial_min_cameras: usize,
        /// Maximum wait for a complete tracker set before a fresh partial set
        /// may emit. Applies only to tracker-only no-record runs.
        #[cfg(feature = "fiducials")]
        #[arg(long, default_value_t = 60)]
        fiducial_max_sync_wait_ms: u64,
        /// Tracker synchronization tolerance in milliseconds. Zero uses the
        /// calibrated session tolerance. Applies only to tracker-only runs.
        #[cfg(feature = "fiducials")]
        #[arg(long, default_value_t = 0.0)]
        fiducial_sync_tolerance_ms: f64,
        /// Refuse a measured tracker update when capture-to-processing age
        /// exceeds this bound. The output becomes predicted/unavailable.
        #[cfg(feature = "fiducials")]
        #[arg(long, default_value_t = 350.0)]
        fiducial_max_capture_age_ms: f64,
        /// Omit a camera from fiducial detection/pose only; repeat as needed.
        #[cfg(feature = "fiducials")]
        #[arg(long)]
        fiducial_exclude_camera: Vec<String>,
        /// Track detections inside the previous full-resolution bounds plus this margin. 0 disables.
        #[cfg(feature = "fiducials")]
        #[arg(long, default_value_t = 0)]
        fiducial_roi_margin_px: usize,
        /// Staggered full-frame reacquisition interval per camera. 0 disables.
        #[cfg(feature = "fiducials")]
        #[arg(long, default_value_t = 50)]
        fiducial_full_scan_period: usize,
        /// Consecutive empty ROI scans before falling back to full-frame search.
        #[cfg(feature = "fiducials")]
        #[arg(long, default_value_t = 3)]
        fiducial_roi_hold_frames: usize,
        /// Full-frame search cadence for a camera with no active ROI. 0 scans every set.
        #[cfg(feature = "fiducials")]
        #[arg(long, default_value_t = 5)]
        fiducial_reacquire_period: usize,
        #[arg(long)]
        socket: Option<PathBuf>,
        /// Cap synchronized sets sent to the local socket. 0 sends every set.
        #[arg(long, default_value_t = 0.0)]
        socket_max_fps: f64,
        /// Uniformly scale decoded BGR/RGB frames before local socket transport.
        /// Metadata dimensions are updated; consumers must scale calibrated
        /// intrinsics explicitly and may not treat this as a calibrated profile.
        #[arg(long, default_value_t = 1.0)]
        socket_scale: f64,
        /// Crop a decoded camera before local socket transport, as
        /// CAMERA=X,Y,WIDTH,HEIGHT in source pixels. Repeat once for every
        /// configured PoE camera; partial crop sets are refused so an omitted
        /// camera cannot silently restore full-frame copying and backpressure.
        #[arg(long, value_name = "CAMERA=X,Y,WIDTH,HEIGHT")]
        socket_crop: Vec<String>,
        /// Write synchronized decoded frames to an Rerun recording.
        #[cfg(feature = "rerun")]
        #[arg(long)]
        rerun_output: Option<PathBuf>,
        /// Stream synchronized decoded frames to a local Rerun Viewer.
        #[cfg(feature = "rerun")]
        #[arg(long)]
        rerun_spawn: bool,
        /// Stream synchronized decoded frames to a Rerun viewer elsewhere on
        /// the network, e.g. rerun+http://192.0.2.90:9876/proxy (color is
        /// JPEG-encoded for the wire).
        #[cfg(feature = "rerun")]
        #[arg(long)]
        rerun_connect: Option<String>,
        /// Cap how many synchronized sets per second are logged to Rerun.
        /// A viewer keeps every frame it is sent, so an unthrottled live
        /// view will exhaust the viewer host's RAM. 0 disables the cap.
        #[cfg(feature = "rerun")]
        #[arg(long, default_value_t = 4.0)]
        rerun_max_fps: f64,
        /// Share this Rerun recording with other producers (e.g. the Python
        /// surface reconstruction) so their data overlays this stream.
        #[cfg(feature = "rerun")]
        #[arg(long)]
        rerun_recording_id: Option<String>,
        /// Viewer workflow; shared recording ids never imply a layout.
        #[cfg(feature = "rerun")]
        #[arg(long, value_enum, default_value_t = RerunLayout::Poe)]
        rerun_layout: RerunLayout,
        /// Add the robot model to the live 3D scene.
        #[cfg(feature = "rerun")]
        #[arg(long, value_name = "URDF")]
        urdf: Option<PathBuf>,
        /// Draw the calibrated camera frustums in the live 3D scene.
        #[cfg(feature = "rerun")]
        #[arg(long, value_name = "BUNDLE")]
        rerun_calibration: Option<PathBuf>,
        /// URDF link the calibration world frame is anchored to.
        #[cfg(feature = "rerun")]
        #[arg(long, default_value = "palette_tag8")]
        calibration_anchor: String,
        /// robot_world.json from solve_robot_world.py: places the calibration
        /// frame by the MEASURED world_from_base, overriding the anchor guess.
        #[cfg(feature = "rerun")]
        #[arg(long, value_name = "JSON")]
        robot_world: Option<PathBuf>,
        /// Live-view mode: do not write evidence or a sync index to disk.
        #[arg(long)]
        no_record: bool,
    },
    /// Continuously monitor the PoE cameras (substream by default) and serve
    /// per-camera health as Prometheus metrics. Runs until stopped.
    #[cfg(feature = "gstreamer")]
    MonitorPoe {
        config: PathBuf,
        #[arg(long, default_value = "sub")]
        stream: String,
        /// Bind host for the /metrics endpoint. Localhost by default; a
        /// deployment that wants network scraping states it explicitly
        /// (plan Phase 3: no default network listener).
        #[arg(long, default_value = "127.0.0.1")]
        bind_host: String,
        /// TCP port for the /metrics endpoint.
        #[arg(long, default_value_t = 9099)]
        port: u16,
        /// Stop after this many seconds; 0 means run forever.
        #[arg(long, default_value_t = 0)]
        duration_seconds: u64,
    },
    /// Capture one live RealSense device into the evidence format.
    #[cfg(feature = "realsense")]
    CaptureRealsense {
        config: PathBuf,
        #[arg(long)]
        sensor: String,
        #[arg(long, default_value_t = 10)]
        duration_seconds: u64,
        #[arg(long)]
        output: Option<PathBuf>,
        #[arg(long)]
        calibration: Option<PathBuf>,
    },
    /// Capture both configured RealSense devices into synchronized color/depth sets.
    #[cfg(feature = "realsense")]
    CaptureRealsenseAll {
        config: PathBuf,
        #[arg(long, default_value_t = 10)]
        duration_seconds: u64,
        #[arg(long)]
        output: Option<PathBuf>,
        #[arg(long)]
        calibration: Option<PathBuf>,
        #[arg(long)]
        socket: Option<PathBuf>,
        /// Write synchronized color/depth frames to an Rerun recording.
        #[cfg(feature = "rerun")]
        #[arg(long)]
        rerun_output: Option<PathBuf>,
        /// Stream synchronized color/depth frames to a local Rerun Viewer.
        #[cfg(feature = "rerun")]
        #[arg(long)]
        rerun_spawn: bool,
        /// Stream to a Rerun viewer elsewhere on the network, e.g.
        /// rerun+http://192.0.2.90:9876/proxy (color is JPEG-encoded for
        /// the wire; depth stays raw Z16, so scale it).
        #[cfg(feature = "rerun")]
        #[arg(long)]
        rerun_connect: Option<String>,
        /// Cap how many synchronized sets per second are logged to Rerun.
        /// A viewer keeps every frame it is sent, so an unthrottled live
        /// view will exhaust the viewer host's RAM. 0 disables the cap.
        #[cfg(feature = "rerun")]
        #[arg(long, default_value_t = 4.0)]
        rerun_max_fps: f64,
        /// Uniformly scale color AND depth before logging to Rerun (raw
        /// 640x480 Z16 is 0.6 MB per frame per camera). Evidence on disk and
        /// the socket transport are never scaled by this.
        #[cfg(feature = "rerun")]
        #[arg(long, default_value_t = 1.0)]
        rerun_image_scale: f64,
        /// Share this Rerun recording with other producers (the PoE-camera
        /// node, stream-teleop, live audio) so everything lands in one viewer.
        #[cfg(feature = "rerun")]
        #[arg(long)]
        rerun_recording_id: Option<String>,
        #[cfg(feature = "rerun")]
        #[arg(long, value_enum, default_value_t = RerunLayout::Realsense)]
        rerun_layout: RerunLayout,
        /// Live-view mode: do not write evidence or a sync index to disk.
        #[arg(long)]
        no_record: bool,
    },
}

#[cfg(feature = "fiducials")]
#[derive(Debug, Deserialize)]
struct EeDetectionReplayRow {
    sequence: u64,
    timestamp_ns: i128,
    #[serde(default)]
    maximum_skew_ns: u128,
    #[serde(default)]
    queue_latency_ms: f64,
    #[serde(default)]
    detection_latency_ms: f64,
    #[serde(default)]
    detections: BTreeMap<String, Vec<FiducialDetection>>,
}

#[cfg(feature = "fiducials")]
fn apply_positive_override(target: &mut f64, value: Option<f64>, name: &str) -> Result<()> {
    if let Some(value) = value {
        if !value.is_finite() || value <= 0.0 {
            anyhow::bail!("--{name} must be finite and positive");
        }
        *target = value;
    }
    Ok(())
}

fn main() -> Result<()> {
    let cli = Cli::parse();
    tracing_subscriber::fmt()
        .with_env_filter(EnvFilter::new(cli.log_filter))
        .with_target(false)
        .init();

    match cli.command {
        Command::ValidateConfig { config } => {
            let config = VisionConfig::load(config)?;
            println!(
                "valid vision config schema {} with {} sensors",
                config.schema_version,
                config.sensor_names().count()
            );
        }
        Command::DescribeConfig { config } => {
            let config = VisionConfig::load(config)?;
            println!("schema_version={}", config.schema_version);
            println!("ntp_server={}", config.sync.ntp_server);
            for camera in &config.cameras.poe {
                println!(
                    "poe {} {} main={}x{}@{:.3} {:?}",
                    camera.name,
                    camera.address,
                    camera.main.width,
                    camera.main.height,
                    camera.main.fps(),
                    camera.main.format
                );
                if let Some(sub) = &camera.sub {
                    println!(
                        "poe {} sub={}x{}@{:.3} {:?}",
                        camera.name,
                        sub.width,
                        sub.height,
                        sub.fps(),
                        sub.format
                    );
                }
            }
            for camera in &config.cameras.realsense {
                println!(
                    "realsense {} serial={} color={}x{}@{:.3} {:?} depth={}x{}@{:.3} {:?}",
                    camera.name,
                    camera.serial,
                    camera.color.width,
                    camera.color.height,
                    camera.color.fps(),
                    camera.color.format,
                    camera.depth.width,
                    camera.depth.height,
                    camera.depth.fps(),
                    camera.depth.format
                );
            }
        }
        Command::ValidateCalibration { bundle } => {
            let bundle = CalibrationBundle::load(&bundle)?;
            println!(
                "valid calibration bundle {} with {} cameras",
                bundle.bundle_id,
                bundle.cameras.len()
            );
        }
        Command::FinalizeCalibration { draft, output } => {
            let text = std::fs::read_to_string(&draft)
                .with_context(|| format!("reading draft bundle {}", draft.display()))?;
            let bundle: CalibrationBundle = serde_json::from_str(&text)
                .with_context(|| format!("parsing draft bundle {}", draft.display()))?;
            let bundle = bundle.with_computed_id()?;
            let output = output.unwrap_or(draft);
            bundle.write(&output)?;
            println!(
                "finalized calibration bundle {} with {} cameras -> {}",
                bundle.bundle_id,
                bundle.cameras.len(),
                output.display()
            );
        }
        Command::AnalyzeSync { reference, other } => {
            let reference_entries = read_recording_entries(&reference)?;
            let other_entries = read_recording_entries(&other)?;
            let reference_metadata: Vec<_> = reference_entries
                .iter()
                .map(|entry| entry.metadata.clone())
                .collect();
            let other_metadata: Vec<_> = other_entries
                .iter()
                .map(|entry| entry.metadata.clone())
                .collect();
            let report = pairwise_sync_report(&reference_metadata, &other_metadata)
                .map_err(anyhow::Error::msg)?;
            println!("{}", serde_json::to_string_pretty(&report)?);
        }
        #[cfg(feature = "fiducials")]
        Command::DetectFiducials {
            evidence,
            inventory,
            calibration,
            output,
            target,
            scale,
            max_sets,
        } => {
            let inventory = FiducialInventory::load(inventory)?;
            let calibration = CalibrationBundle::load(calibration)?;
            let detector = AprilTagDetectorFactory::new(&inventory, target.as_deref(), scale)?;
            let source = ReplaySource::load(evidence)?;
            let mut writer = BufWriter::new(
                OpenOptions::new()
                    .create_new(true)
                    .write(true)
                    .open(&output)
                    .with_context(|| format!("opening {}", output.display()))?,
            );
            let limit = if max_sets == 0 {
                source.index.len()
            } else {
                max_sets.min(source.index.len())
            };
            let mut detection_count = 0_usize;
            for row in source.index.iter().take(limit) {
                let set = source.frame_set(row)?;
                let started = Instant::now();
                let detections = detector.detect_set(&calibration, &set)?;
                let detection_latency_ms = started.elapsed().as_secs_f64() * 1000.0;
                detection_count += detections.len();
                let batch = FiducialDetectionBatch::new(
                    &set,
                    &inventory.inventory_hash,
                    &calibration.bundle_id,
                    0.0,
                    detection_latency_ms,
                    0.0,
                    detection_latency_ms,
                    0,
                    detection_latency_ms,
                    "offline_processing_only",
                    detections,
                );
                serde_json::to_writer(&mut writer, &batch)?;
                writer.write_all(b"\n")?;
            }
            writer.flush()?;
            println!(
                "detected {detection_count} configured tags in {limit} synchronized sets -> {}",
                output.display()
            );
        }
        #[cfg(feature = "fiducials")]
        Command::ReplayEeDetections {
            input,
            inventory,
            calibration,
            wrist_layout,
            output,
            exclude_camera,
            max_source_rmse_px,
            max_total_rmse_px,
            max_translation_sigma_mm,
            max_rotation_sigma_deg,
        } => {
            let inventory = FiducialInventory::load(inventory)?;
            let calibration = CalibrationBundle::load(calibration)?;
            let layout = WristLayout::load(wrist_layout, &inventory, false)?;
            let mut config = EstimatorConfig::default();
            apply_positive_override(
                &mut config.max_source_rmse_px,
                max_source_rmse_px,
                "max-source-rmse-px",
            )?;
            apply_positive_override(
                &mut config.max_total_rmse_px,
                max_total_rmse_px,
                "max-total-rmse-px",
            )?;
            apply_positive_override(
                &mut config.max_translation_sigma_mm,
                max_translation_sigma_mm,
                "max-translation-sigma-mm",
            )?;
            apply_positive_override(
                &mut config.max_rotation_sigma_deg,
                max_rotation_sigma_deg,
                "max-rotation-sigma-deg",
            )?;
            let excluded: std::collections::BTreeSet<_> = exclude_camera.into_iter().collect();
            let mut tracker = RustEeTracker::new(&calibration, &inventory, layout, config)?;
            let reader = BufReader::new(
                File::open(&input).with_context(|| format!("opening {}", input.display()))?,
            );
            let mut writer = BufWriter::new(
                OpenOptions::new()
                    .create_new(true)
                    .write(true)
                    .open(&output)
                    .with_context(|| format!("opening {}", output.display()))?,
            );
            let mut statuses = BTreeMap::<String, usize>::new();
            let mut rows = 0_usize;
            for (line_index, line) in reader.lines().enumerate() {
                let line = line.with_context(|| {
                    format!("reading {} line {}", input.display(), line_index + 1)
                })?;
                if line.trim().is_empty() {
                    continue;
                }
                let row: EeDetectionReplayRow = serde_json::from_str(&line).with_context(|| {
                    format!("parsing {} line {}", input.display(), line_index + 1)
                })?;
                let detections = row
                    .detections
                    .into_values()
                    .flatten()
                    .filter(|detection| !excluded.contains(&detection.camera))
                    .collect();
                let detector_age =
                    std::time::Duration::from_secs_f64(row.detection_latency_ms.max(0.0) / 1000.0);
                let mut estimate = tracker.update(
                    row.sequence,
                    row.timestamp_ns,
                    row.maximum_skew_ns,
                    detections,
                    row.queue_latency_ms,
                    row.detection_latency_ms,
                    Instant::now() - detector_age,
                );
                estimate.latency_basis = "retained_capture_detection_plus_replay_solver".into();
                *statuses.entry(estimate.status.clone()).or_default() += 1;
                serde_json::to_writer(&mut writer, &estimate)?;
                writer.write_all(b"\n")?;
                rows += 1;
            }
            writer.flush()?;
            println!(
                "replayed {rows} EE detection rows statuses={} -> {}",
                serde_json::to_string(&statuses)?,
                output.display()
            );
        }
        #[cfg(feature = "rerun")]
        Command::ReplayRerun {
            recording_roots,
            reconstruction_dir,
            urdf,
            calibration,
            calibration_anchor,
            robot_world,
            teleop_log,
            teleop_leader,
            teleop_fps,
            output,
            spawn,
            realtime,
            speed,
            max_fps,
            image_scale,
            jpeg_quality,
            rerun_layout,
        } => {
            if output.is_some() && spawn {
                anyhow::bail!("choose either --output or --spawn, not both");
            }
            if realtime && !(speed.is_finite() && speed > 0.0) {
                anyhow::bail!("--speed must be finite and positive");
            }
            if !max_fps.is_finite() || max_fps < 0.0 {
                anyhow::bail!("--max-fps must be finite and non-negative");
            }
            if !image_scale.is_finite() || !(0.0..=1.0).contains(&image_scale) || image_scale == 0.0
            {
                anyhow::bail!("--image-scale must be in (0, 1]");
            }
            if jpeg_quality == 0 || jpeg_quality > 100 {
                anyhow::bail!("--jpeg-quality must be between 1 and 100");
            }
            if recording_roots.is_empty() && teleop_log.is_none() && calibration.is_none() {
                anyhow::bail!(
                    "provide at least one RECORDING_ROOT, --teleop-log, or --calibration"
                );
            }
            let follower_prefix = match teleop_leader.as_str() {
                "left" => "right",
                "right" => "left",
                other => anyhow::bail!("--teleop-leader must be 'left' or 'right', got {other}"),
            };
            if !(teleop_fps.is_finite() && teleop_fps > 0.0) {
                anyhow::bail!("--teleop-fps must be finite and positive");
            }
            let teleop = teleop_log
                .map(|path| -> Result<TeleopSetup> {
                    Ok(TeleopSetup {
                        log: tatbot_visiond::TeleopLog::read_file(path)?,
                        leader_prefix: teleop_leader.clone(),
                        follower_prefix: follower_prefix.to_owned(),
                        transform_fps: teleop_fps,
                    })
                })
                .transpose()?;
            let sources = recording_roots
                .into_iter()
                .map(ReplaySource::load)
                .collect::<Result<Vec<_>>>()?;
            let has_poe = sources
                .iter()
                .any(|source| source.has_sensor_kind(SensorKind::PoE));
            let has_realsense = sources
                .iter()
                .any(|source| source.has_sensor_kind(SensorKind::RealSense));
            let layout =
                rerun_layout.unwrap_or_else(|| match (has_poe, has_realsense, teleop.is_some()) {
                    (true, true, true) => RerunLayout::Full,
                    (true, true, false) => RerunLayout::Cameras,
                    (true, false, true) => RerunLayout::PoeTeleop,
                    (false, true, true) => RerunLayout::RealsenseTeleop,
                    (true, false, false) => RerunLayout::Poe,
                    (false, true, false) => RerunLayout::Realsense,
                    (false, false, true) => RerunLayout::Teleop,
                    (false, false, false) => RerunLayout::Calibration,
                });
            let calibration_bundle = calibration
                .as_deref()
                .map(CalibrationBundle::load)
                .transpose()?;
            let mut viewer = if spawn {
                RerunViewer::spawn(layout)?
            } else {
                let output =
                    output.with_context(|| "ReplayRerun needs --output PATH or --spawn")?;
                RerunViewer::save(output, layout)?
            };
            // Offline conversion trades CPU for ~20-50x smaller recordings.
            viewer.set_jpeg_quality(Some(jpeg_quality));
            viewer.log_session_metadata(
                "replay",
                None,
                urdf.as_deref(),
                calibration_bundle
                    .as_ref()
                    .map(|bundle| bundle.bundle_id.as_str()),
            )?;
            viewer.log_scene(
                reconstruction_dir.as_deref(),
                urdf.as_deref(),
                teleop.as_ref(),
            )?;
            if let Some(bundle) = &calibration_bundle {
                viewer.log_calibration(
                    bundle,
                    urdf.as_deref(),
                    Some(calibration_anchor.as_str()),
                    robot_world.as_deref(),
                )?;
            }
            if let Some(setup) = &teleop {
                println!(
                    "replayed {} teleop ticks ({} joints, {:.1} s) into Rerun",
                    setup.log.ticks.len(),
                    setup.log.num_joints,
                    setup
                        .log
                        .ticks
                        .last()
                        .map(|tick| tick.t_wake)
                        .unwrap_or(0.0)
                );
            }
            let mut replay_rows = Vec::new();
            for (source_index, source) in sources.iter().enumerate() {
                for (row_index, row) in source.index.iter().enumerate() {
                    replay_rows.push((row.timestamp_ns, source_index, row_index));
                }
            }
            replay_rows.sort_by_key(|(timestamp_ns, source_index, row_index)| {
                (*timestamp_ns, *source_index, *row_index)
            });
            let input_rows = replay_rows.len();
            decimate_replay_rows(&mut replay_rows, sources.len(), max_fps);
            if replay_rows.is_empty() && teleop.is_none() && calibration.is_none() {
                anyhow::bail!("the supplied recording roots contain no synchronized sets");
            }

            let mut previous_timestamp = None;
            let mut replayed = 0_u64;
            if realtime {
                for (_, source_index, row_index) in replay_rows {
                    let row = &sources[source_index].index[row_index];
                    let mut set = sources[source_index].frame_set(row)?;
                    if image_scale != 1.0 {
                        set = scale_video_set(&set, image_scale, "visualization")?;
                    }
                    set.sequence = replayed;
                    if let Some(previous) = previous_timestamp {
                        let elapsed_ns = set.timestamp_ns.saturating_sub(previous);
                        if elapsed_ns > 0 {
                            let wait_ns = (elapsed_ns as f64 / speed).round() as u64;
                            thread::sleep(Duration::from_nanos(wait_ns));
                        }
                    }
                    previous_timestamp = Some(set.timestamp_ns);
                    viewer.log_set(&set)?;
                    replayed = replayed.saturating_add(1);
                }
            } else {
                // Batch sets so JPEG encoding parallelizes across the whole
                // batch (a single set has too few frames to fill the cores).
                for chunk in replay_rows.chunks(16) {
                    let mut sets = Vec::with_capacity(chunk.len());
                    for (_, source_index, row_index) in chunk {
                        let row = &sources[*source_index].index[*row_index];
                        let mut set = sources[*source_index].frame_set(row)?;
                        if image_scale != 1.0 {
                            set = scale_video_set(&set, image_scale, "visualization")?;
                        }
                        set.sequence = replayed;
                        replayed = replayed.saturating_add(1);
                        sets.push(set);
                    }
                    viewer.log_sets(&sets)?;
                }
            }
            viewer.finish()?;
            println!("replayed {replayed}/{input_rows} synchronized sets into Rerun ");
        }
        #[cfg(feature = "rerun")]
        Command::StreamTeleop {
            bind,
            connect,
            output,
            recording_id,
            urdf,
            calibration,
            calibration_anchor,
            robot_world,
            rerun_layout,
            leader_prefix,
            follower_prefix,
            duration_seconds,
            idle_timeout_seconds,
        } => {
            if !idle_timeout_seconds.is_finite() || idle_timeout_seconds < 0.0 {
                anyhow::bail!("--idle-timeout-seconds must be finite and non-negative");
            }
            if connect.is_some() == output.is_some() {
                anyhow::bail!("choose exactly one of --connect or --output");
            }
            let socket = UdpSocket::bind(&bind)
                .with_context(|| format!("binding live teleop telemetry at {bind}"))?;
            socket.set_read_timeout(Some(Duration::from_millis(250)))?;
            let viewer = if let Some(url) = connect {
                RerunViewer::connect(&url, Some(recording_id.as_str()), rerun_layout)?
            } else {
                RerunViewer::save(
                    output.expect("output precondition checked"),
                    rerun_layout.unwrap_or(RerunLayout::Teleop),
                )?
            };
            let calibration_bundle = calibration
                .as_deref()
                .map(CalibrationBundle::load)
                .transpose()?;
            viewer.log_session_metadata(
                "live_teleop",
                Some(recording_id.as_str()),
                Some(&urdf),
                calibration_bundle
                    .as_ref()
                    .map(|bundle| bundle.bundle_id.as_str()),
            )?;
            let scene = viewer.prepare_live_teleop(&urdf, &leader_prefix, &follower_prefix)?;
            if let Some(bundle) = &calibration_bundle {
                viewer.log_calibration(
                    bundle,
                    Some(&urdf),
                    Some(calibration_anchor.as_str()),
                    robot_world.as_deref(),
                )?;
            }
            viewer.log_status(format!(
                "live teleop: waiting for UDP joint state on {bind}"
            ))?;

            let started = Instant::now();
            let deadline =
                (duration_seconds > 0).then(|| started + Duration::from_secs(duration_seconds));
            let idle_timeout = Duration::from_secs_f64(idle_timeout_seconds);
            let mut buffer = vec![0_u8; 64 * 1024];
            let mut last_packet: Option<Instant> = None;
            let mut last_sequence: Option<u64> = None;
            let mut last_status = Instant::now();
            let mut idle_reported = false;
            let mut received = 0_u64;
            let mut dropped_out_of_order = 0_u64;
            let mut malformed = 0_u64;
            loop {
                if deadline.is_some_and(|value| Instant::now() >= value) {
                    break;
                }
                match socket.recv_from(&mut buffer) {
                    Ok((size, source)) => {
                        let tick = match LiveTeleopTick::parse(&buffer[..size]) {
                            Ok(tick) => tick,
                            Err(error) => {
                                malformed = malformed.saturating_add(1);
                                if last_status.elapsed() >= Duration::from_secs(1) {
                                    viewer.log_status(format!(
                                        "live teleop: rejected malformed packet from {source}: {error}"
                                    ))?;
                                    last_status = Instant::now();
                                }
                                continue;
                            }
                        };
                        if last_sequence.is_some_and(|sequence| tick.sequence <= sequence) {
                            dropped_out_of_order = dropped_out_of_order.saturating_add(1);
                            continue;
                        }
                        if let Err(error) = viewer.log_live_teleop_tick(&scene, &tick) {
                            malformed = malformed.saturating_add(1);
                            if last_status.elapsed() >= Duration::from_secs(1) {
                                viewer.log_status(format!(
                                    "live teleop: rejected incompatible joint state: {error}"
                                ))?;
                                last_status = Instant::now();
                            }
                            continue;
                        }
                        last_sequence = Some(tick.sequence);
                        last_packet = Some(Instant::now());
                        received = received.saturating_add(1);
                        idle_reported = false;
                        if last_status.elapsed() >= Duration::from_secs(1) {
                            let source_age = SystemTime::now()
                                .duration_since(UNIX_EPOCH)
                                .unwrap_or_default()
                                .as_secs_f64()
                                - tick.timestamp_ns as f64 / 1e9;
                            viewer.log_status(format!(
                                "live teleop: receiving from {source}; packet {received}; source age {:.3}s",
                                source_age.max(0.0)
                            ))?;
                            last_status = Instant::now();
                        }
                    }
                    Err(error)
                        if matches!(
                            error.kind(),
                            std::io::ErrorKind::WouldBlock | std::io::ErrorKind::TimedOut
                        ) =>
                    {
                        let idle = last_packet.map_or(started.elapsed(), |seen| seen.elapsed());
                        if idle_timeout_seconds > 0.0 && idle >= idle_timeout && !idle_reported {
                            viewer.log_status(format!(
                                "live teleop: no joint state for {:.1}s; model held at last pose",
                                idle.as_secs_f64()
                            ))?;
                            idle_reported = true;
                        }
                    }
                    Err(error) => return Err(error).context("receiving live teleop telemetry"),
                }
            }
            viewer.log_status(format!(
                "live teleop stopped: received={received} dropped_out_of_order={dropped_out_of_order} malformed={malformed}"
            ))?;
            viewer.finish()?;
            println!("live_teleop_received={received} ");
            println!("live_teleop_dropped_out_of_order={dropped_out_of_order}");
            println!("live_teleop_malformed={malformed}");
        }
        #[cfg(feature = "gstreamer")]
        Command::CapturePoe {
            config,
            sensor,
            stream,
            duration_seconds,
            decoded,
            keyframes_only,
            output,
            calibration,
        } => {
            let config = VisionConfig::load(config)?;
            let calibration = calibration.map(CalibrationBundle::load).transpose()?;
            let camera = config
                .cameras
                .poe
                .iter()
                .find(|camera| camera.name == sensor)
                .cloned()
                .with_context(|| format!("unknown PoE sensor {sensor}"))?;
            let stream = match stream.as_str() {
                "main" => PoeStream::Main,
                "sub" => PoeStream::Sub,
                other => anyhow::bail!("stream must be main or sub, got {other}"),
            };
            let password = env::var(&camera.password_env).with_context(|| {
                format!(
                    "missing password environment variable {}",
                    camera.password_env
                )
            })?;
            let mut capture = PoeRtspCapture::new_with_options(
                camera.clone(),
                stream,
                &password,
                decoded,
                keyframes_only,
            )?;
            let output_root = output.unwrap_or_else(|| PathBuf::from(&config.session.record_root));
            let mut recorder = EvidenceRecorder::create(&output_root, &camera.name)?;
            let deadline = Instant::now() + Duration::from_secs(duration_seconds);
            let mut frames = 0_u64;
            while Instant::now() < deadline {
                if let Some(frame) = capture.next_frame(Duration::from_millis(1500))? {
                    let mut frame = frame;
                    stamp_calibration(&mut frame, calibration.as_ref())?;
                    recorder.write(&frame)?;
                    frames += 1;
                }
            }
            let manifest = recorder.finish()?;
            capture.stop()?;
            println!(
                "captured {} frames for {} into {}",
                frames,
                camera.name,
                output_root.display()
            );
            println!("health={}", serde_json::to_string(&capture.health())?);
            println!("manifest={}", serde_json::to_string(&manifest)?);
        }
        #[cfg(feature = "gstreamer")]
        Command::CapturePoeAll {
            config,
            stream,
            duration_seconds,
            decoded,
            keyframes_only,
            output,
            calibration,
            #[cfg(feature = "fiducials")]
            fiducial_inventory,
            #[cfg(feature = "fiducials")]
            wrist_layout,
            #[cfg(feature = "fiducials")]
            fiducial_output,
            #[cfg(feature = "fiducials")]
            fiducial_scale,
            #[cfg(feature = "fiducials")]
            fiducial_max_fps,
            #[cfg(feature = "fiducials")]
            fiducial_min_cameras,
            #[cfg(feature = "fiducials")]
            fiducial_max_sync_wait_ms,
            #[cfg(feature = "fiducials")]
            fiducial_sync_tolerance_ms,
            #[cfg(feature = "fiducials")]
            fiducial_max_capture_age_ms,
            #[cfg(feature = "fiducials")]
            fiducial_exclude_camera,
            #[cfg(feature = "fiducials")]
            fiducial_roi_margin_px,
            #[cfg(feature = "fiducials")]
            fiducial_full_scan_period,
            #[cfg(feature = "fiducials")]
            fiducial_roi_hold_frames,
            #[cfg(feature = "fiducials")]
            fiducial_reacquire_period,
            socket,
            socket_max_fps,
            socket_scale,
            socket_crop,
            #[cfg(feature = "rerun")]
            rerun_output,
            #[cfg(feature = "rerun")]
            rerun_spawn,
            #[cfg(feature = "rerun")]
            rerun_connect,
            #[cfg(feature = "rerun")]
            rerun_max_fps,
            #[cfg(feature = "rerun")]
            rerun_recording_id,
            #[cfg(feature = "rerun")]
            rerun_layout,
            #[cfg(feature = "rerun")]
            urdf,
            #[cfg(feature = "rerun")]
            rerun_calibration,
            #[cfg(feature = "rerun")]
            calibration_anchor,
            #[cfg(feature = "rerun")]
            robot_world,
            no_record,
        } => {
            let config = VisionConfig::load(config)?;
            let socket_crops = parse_socket_crops(&socket_crop)?;
            if !(0.0..=1.0).contains(&socket_scale) || socket_scale == 0.0 {
                anyhow::bail!("--socket-scale must be in (0, 1]");
            }
            if !socket_max_fps.is_finite() || socket_max_fps < 0.0 {
                anyhow::bail!("--socket-max-fps must be finite and non-negative");
            }
            if socket.is_none()
                && (socket_scale != 1.0 || socket_max_fps != 0.0 || !socket_crops.is_empty())
            {
                anyhow::bail!("--socket-scale/--socket-max-fps/--socket-crop require --socket");
            }
            if (socket_scale != 1.0 || !socket_crops.is_empty()) && !decoded {
                anyhow::bail!("--socket-scale/--socket-crop require --decoded BGR/RGB frames");
            }
            if !socket_crops.is_empty() {
                let configured = config
                    .cameras
                    .poe
                    .iter()
                    .map(|camera| camera.name.as_str())
                    .collect::<std::collections::BTreeSet<_>>();
                let supplied = socket_crops
                    .keys()
                    .map(String::as_str)
                    .collect::<std::collections::BTreeSet<_>>();
                if supplied != configured {
                    let missing = configured
                        .difference(&supplied)
                        .copied()
                        .collect::<Vec<_>>();
                    let unknown = supplied
                        .difference(&configured)
                        .copied()
                        .collect::<Vec<_>>();
                    anyhow::bail!(
                        "--socket-crop must cover every configured PoE camera; missing={missing:?} unknown={unknown:?}"
                    );
                }
            }
            #[cfg(feature = "fiducials")]
            {
                if !fiducial_max_fps.is_finite() || fiducial_max_fps < 0.0 {
                    anyhow::bail!("--fiducial-max-fps must be finite and non-negative");
                }
                if fiducial_min_cameras == 0 || fiducial_min_cameras > config.cameras.poe.len() {
                    anyhow::bail!(
                        "--fiducial-min-cameras must be in 1..={}, got {fiducial_min_cameras}",
                        config.cameras.poe.len()
                    );
                }
                if fiducial_max_sync_wait_ms == 0 {
                    anyhow::bail!("--fiducial-max-sync-wait-ms must be positive");
                }
                if !fiducial_sync_tolerance_ms.is_finite() || fiducial_sync_tolerance_ms < 0.0 {
                    anyhow::bail!("--fiducial-sync-tolerance-ms must be finite and non-negative");
                }
                if !fiducial_max_capture_age_ms.is_finite() || fiducial_max_capture_age_ms <= 0.0 {
                    anyhow::bail!("--fiducial-max-capture-age-ms must be finite and positive");
                }
                if fiducial_output.is_some() && fiducial_inventory.is_none() {
                    anyhow::bail!("--fiducial-output requires --fiducial-inventory");
                }
                if fiducial_inventory.is_some() && fiducial_output.is_none() {
                    anyhow::bail!("--fiducial-inventory requires --fiducial-output");
                }
                if wrist_layout.is_some() && fiducial_inventory.is_none() {
                    anyhow::bail!("--wrist-layout requires --fiducial-inventory");
                }
                if fiducial_inventory.is_some() && calibration.is_none() {
                    anyhow::bail!("fiducial detection requires --calibration");
                }
                if fiducial_inventory.is_some() && !decoded {
                    anyhow::bail!("fiducial detection requires decoded pixels; add --decoded");
                }
                if fiducial_max_fps != 0.0 && fiducial_inventory.is_none() {
                    anyhow::bail!("--fiducial-max-fps requires --fiducial-inventory");
                }
                if !fiducial_exclude_camera.is_empty() && fiducial_inventory.is_none() {
                    anyhow::bail!("--fiducial-exclude-camera requires --fiducial-inventory");
                }
                if fiducial_roi_margin_px != 0 && fiducial_inventory.is_none() {
                    anyhow::bail!("--fiducial-roi-margin-px requires --fiducial-inventory");
                }
                let configured_cameras: std::collections::BTreeSet<_> = config
                    .cameras
                    .poe
                    .iter()
                    .map(|camera| camera.name.as_str())
                    .collect();
                for camera in &fiducial_exclude_camera {
                    if !configured_cameras.contains(camera.as_str()) {
                        anyhow::bail!("unknown --fiducial-exclude-camera {camera}");
                    }
                }
            }
            #[cfg(feature = "rerun")]
            if (rerun_output.is_some() || rerun_spawn || rerun_connect.is_some()) && !decoded {
                anyhow::bail!("PoE Rerun output requires decoded pixels; add --decoded");
            }
            #[cfg(feature = "rerun")]
            let rerun_viewer = open_rerun_viewer(
                rerun_output,
                rerun_spawn,
                rerun_connect,
                rerun_recording_id.as_deref(),
                rerun_layout,
            )?;
            #[cfg(feature = "rerun")]
            if let Some(viewer) = rerun_viewer.as_ref() {
                let rerun_bundle = rerun_calibration
                    .as_deref()
                    .map(CalibrationBundle::load)
                    .transpose()?;
                viewer.log_session_metadata(
                    "capture_poe_all",
                    rerun_recording_id.as_deref(),
                    urdf.as_deref(),
                    rerun_bundle
                        .as_ref()
                        .map(|bundle| bundle.bundle_id.as_str()),
                )?;
                // Static scene: robot model and calibrated frustums, logged
                // once so the live stream has spatial context.
                if urdf.is_some() {
                    viewer.log_scene(None, urdf.as_deref(), None)?;
                }
                if let Some(bundle) = &rerun_bundle {
                    viewer.log_calibration(
                        bundle,
                        urdf.as_deref(),
                        Some(calibration_anchor.as_str()),
                        robot_world.as_deref(),
                    )?;
                }
            }
            #[cfg(feature = "rerun")]
            let rerun_sink = rerun_viewer.map(RerunSink::new);
            #[cfg(feature = "rerun")]
            let rerun_min_interval =
                (rerun_max_fps > 0.0).then(|| Duration::from_secs_f64(1.0 / rerun_max_fps));
            #[cfg(feature = "rerun")]
            let mut rerun_last_logged: Option<Instant> = None;
            let calibration = calibration.map(CalibrationBundle::load).transpose()?;
            #[cfg(feature = "fiducials")]
            let mut fiducial_pipeline = match (fiducial_inventory, fiducial_output) {
                (Some(inventory_path), Some(output_path)) => Some(FiducialPipeline::new(
                    inventory_path,
                    wrist_layout,
                    output_path,
                    fiducial_scale,
                    fiducial_exclude_camera,
                    fiducial_roi_margin_px,
                    fiducial_full_scan_period,
                    fiducial_roi_hold_frames,
                    fiducial_reacquire_period,
                    fiducial_max_capture_age_ms,
                    calibration
                        .as_ref()
                        .expect("fiducial calibration precondition checked")
                        .clone(),
                )?),
                (None, None) => None,
                _ => unreachable!("fiducial CLI preconditions checked"),
            };
            #[cfg(feature = "fiducials")]
            let fiducial_min_interval_ns = (fiducial_max_fps > 0.0)
                .then(|| (1_000_000_000.0 / fiducial_max_fps).max(1.0) as i128);
            #[cfg(feature = "fiducials")]
            let mut fiducial_last_processed_ns: Option<i128> = None;
            let stream = match stream.as_str() {
                "main" => PoeStream::Main,
                "sub" => PoeStream::Sub,
                other => anyhow::bail!("stream must be main or sub, got {other}"),
            };
            let output_root = output.unwrap_or_else(|| PathBuf::from(&config.session.record_root));
            let mut recorders = if no_record {
                None
            } else {
                let mut map = BTreeMap::new();
                for camera in &config.cameras.poe {
                    map.insert(
                        camera.name.clone(),
                        EvidenceRecorder::create(&output_root, &camera.name)?,
                    );
                }
                Some(map)
            };
            let sensor_names: Vec<_> = config
                .cameras
                .poe
                .iter()
                .map(|camera| camera.name.clone())
                .collect();
            let tolerance_ns = (config.sync.max_pairwise_skew_ms * 1_000_000.0) as u128;
            #[cfg(feature = "rerun")]
            let has_rerun_consumer = rerun_sink.is_some();
            #[cfg(not(feature = "rerun"))]
            let has_rerun_consumer = false;
            #[cfg(feature = "fiducials")]
            let partial_tracking = fiducial_pipeline.is_some()
                && no_record
                && socket.is_none()
                && !has_rerun_consumer
                && fiducial_min_cameras < sensor_names.len();
            #[cfg(not(feature = "fiducials"))]
            let partial_tracking = false;
            #[cfg(feature = "fiducials")]
            let tracking_tolerance_ns = if fiducial_sync_tolerance_ms > 0.0 {
                (fiducial_sync_tolerance_ms * 1_000_000.0) as u128
            } else {
                tolerance_ns
            };
            #[cfg(not(feature = "fiducials"))]
            let tracking_tolerance_ns = tolerance_ns;
            let mut synchronizer = if partial_tracking {
                #[cfg(feature = "fiducials")]
                {
                    FrameSynchronizer::new_partial(
                        sensor_names,
                        fiducial_min_cameras,
                        tracking_tolerance_ns,
                        u128::from(fiducial_max_sync_wait_ms) * 1_000_000,
                        config.session.queue_capacity,
                    )
                    .map_err(anyhow::Error::msg)?
                }
                #[cfg(not(feature = "fiducials"))]
                unreachable!()
            } else {
                FrameSynchronizer::new(sensor_names, tolerance_ns, config.session.queue_capacity)
                    .map_err(anyhow::Error::msg)?
            };
            let mut sync_index = if no_record {
                None
            } else {
                let sync_index_path = output_root.join("synchronized_frames.jsonl");
                Some(BufWriter::new(
                    OpenOptions::new()
                        .create_new(true)
                        .write(true)
                        .open(&sync_index_path)
                        .with_context(|| format!("opening {}", sync_index_path.display()))?,
                ))
            };
            let mut publisher = socket.map(UnixFramePublisher::bind).transpose()?;
            let socket_min_interval =
                (socket_max_fps > 0.0).then(|| Duration::from_secs_f64(1.0 / socket_max_fps));
            let mut socket_last_published: Option<Instant> = None;
            let deadline = Instant::now() + Duration::from_secs(duration_seconds);
            let expected = config.cameras.poe.len();
            // Decoded main-stream frames are about 15 MiB each.  An unbounded
            // channel let capture outrun fiducial processing during motion;
            // a 60 s run accumulated ~26 GiB and kept processing for minutes
            // after its deadline.  Retain only one complete set's worth of
            // ingress events. Backpressure then reaches each appsink, whose
            // own bounded `drop=true` queue keeps recent frames. The
            // synchronizer still owns its configured per-sensor tolerance
            // queues; duplicating that capacity here only adds frame age.
            let worker_queue_capacity = expected.max(1);
            let (sender, receiver) = mpsc::sync_channel(worker_queue_capacity);
            let mut workers = Vec::new();
            for camera in config.cameras.poe.clone() {
                let sender = sender.clone();
                workers.push(thread::spawn(move || {
                    let sensor = camera.name.clone();
                    let mut capture = None;
                    let mut had_capture = false;
                    let mut reconnects = 0_u64;
                    let mut recent_errors = Vec::new();
                    while Instant::now() < deadline {
                        if capture.is_none() {
                            match env::var(&camera.password_env)
                                .with_context(|| {
                                    format!(
                                        "missing password environment variable {}",
                                        camera.password_env
                                    )
                                })
                                .and_then(|password| {
                                    PoeRtspCapture::new_with_options(
                                        camera.clone(),
                                        stream,
                                        &password,
                                        decoded,
                                        keyframes_only,
                                    )
                                }) {
                                Ok(value) => {
                                    if had_capture {
                                        reconnects = reconnects.saturating_add(1);
                                    }
                                    had_capture = true;
                                    capture = Some(value);
                                }
                                Err(error) => {
                                    recent_errors.push(error.to_string());
                                    let _ = sender.send(PoeWorkerEvent::Error {
                                        sensor: sensor.clone(),
                                        message: error.to_string(),
                                    });
                                    thread::sleep(Duration::from_millis(250));
                                    continue;
                                }
                            }
                        }
                        let result = capture
                            .as_mut()
                            .expect("capture initialized")
                            .next_frame(Duration::from_millis(1500));
                        match result {
                            Ok(Some(frame)) => {
                                if sender
                                    .send(PoeWorkerEvent::Frame {
                                        frame,
                                        enqueued_at: Instant::now(),
                                    })
                                    .is_err()
                                {
                                    break;
                                }
                            }
                            Ok(None) => {}
                            Err(error) => {
                                recent_errors.push(error.to_string());
                                let _ = sender.send(PoeWorkerEvent::Error {
                                    sensor: sensor.clone(),
                                    message: error.to_string(),
                                });
                                if let Some(value) = capture.take() {
                                    let _ = value.stop();
                                }
                                thread::sleep(Duration::from_millis(250));
                            }
                        }
                    }
                    let mut health =
                        capture
                            .as_ref()
                            .map(PoeRtspCapture::health)
                            .unwrap_or_else(|| {
                                tatbot_visiond::SensorHealth::new(sensor.clone()).snapshot()
                            });
                    health.reconnects = health.reconnects.saturating_add(reconnects);
                    health.recent_errors = recent_errors.into_iter().rev().take(16).collect();
                    health.recent_errors.reverse();
                    if let Some(value) = capture {
                        let _ = value.stop();
                    }
                    let _ = sender.send(PoeWorkerEvent::Finished { sensor, health });
                }));
            }
            drop(sender);

            let mut finished = 0;
            let mut frame_counts = BTreeMap::<String, u64>::new();
            let mut pipeline_capture_ages_ms = BTreeMap::<String, Vec<f64>>::new();
            let mut pipeline_stage_latencies_ms =
                BTreeMap::<String, BTreeMap<String, Vec<f64>>>::new();
            let mut capture_event_channel_waits_ms = BTreeMap::<String, Vec<f64>>::new();
            let mut synchronizer_waits_ms = Vec::<f64>::new();
            #[cfg(feature = "fiducials")]
            let mut fiducial_processing_ms = Vec::<f64>::new();
            #[cfg(feature = "fiducials")]
            let mut fiducial_rate_limit_remaining_ms = Vec::<f64>::new();
            let mut errors = Vec::new();
            let mut health = BTreeMap::new();
            let mut synchronized_sets = 0_u64;
            let mut frames_discarded_after_deadline = 0_u64;
            #[cfg(feature = "fiducials")]
            let mut fiducial_sets_rate_limited = 0_u64;
            while finished < expected {
                match receiver.recv_timeout(Duration::from_secs(15)) {
                    Ok(PoeWorkerEvent::Frame { frame, enqueued_at }) => {
                        // `duration_seconds` is a wall-clock bound, not merely
                        // an ingestion bound.  Once it expires, drain frames
                        // without expensive recording/detection so blocked
                        // workers can publish their final health and exit.
                        if Instant::now() >= deadline {
                            frames_discarded_after_deadline =
                                frames_discarded_after_deadline.saturating_add(1);
                            continue;
                        }
                        let mut frame = frame;
                        stamp_calibration(&mut frame, calibration.as_ref())?;
                        let sensor = frame.metadata.sensor_name.clone();
                        capture_event_channel_waits_ms
                            .entry(sensor.clone())
                            .or_default()
                            .push(enqueued_at.elapsed().as_secs_f64() * 1000.0);
                        let event_received_unix_ns = SystemTime::now()
                            .duration_since(UNIX_EPOCH)
                            .unwrap_or(Duration::ZERO)
                            .as_nanos();
                        frame.metadata.attributes.insert(
                            "capture_event_received_unix_ns".to_string(),
                            event_received_unix_ns.to_string(),
                        );
                        if let Some(age_ms) = frame
                            .metadata
                            .attributes
                            .get("pipeline_capture_age_ns")
                            .and_then(|value| value.parse::<i128>().ok())
                            .map(|value| value as f64 / 1e6)
                        {
                            pipeline_capture_ages_ms
                                .entry(sensor.clone())
                                .or_default()
                                .push(age_ms);
                        }
                        for (name, value) in &frame.metadata.attributes {
                            if name.starts_with("pipeline_")
                                && name.ends_with("_ns")
                                && name != "pipeline_capture_age_ns"
                                && name != "pipeline_pts_ns"
                                && name != "pipeline_dts_ns"
                                && name != "pipeline_running_time_ns"
                                && name != "pipeline_pts_to_now_ns"
                            {
                                if let Ok(value_ns) = value.parse::<u128>() {
                                    pipeline_stage_latencies_ms
                                        .entry(sensor.clone())
                                        .or_default()
                                        .entry(
                                            name.trim_start_matches("pipeline_")
                                                .trim_end_matches("_ns")
                                                .to_string(),
                                        )
                                        .or_default()
                                        .push(value_ns as f64 / 1e6);
                                }
                            }
                        }
                        *frame_counts.entry(sensor.clone()).or_default() += 1;
                        if let Some(recorders) = recorders.as_mut() {
                            recorders
                                .get_mut(&sensor)
                                .with_context(|| format!("no recorder for {sensor}"))?
                                .write(&frame)?;
                        }
                        for set in synchronizer.push(frame).map_err(anyhow::Error::msg)? {
                            let set_processing_unix_ns = SystemTime::now()
                                .duration_since(UNIX_EPOCH)
                                .unwrap_or(Duration::ZERO)
                                .as_nanos();
                            let synchronizer_wait_ns = set
                                .frames
                                .values()
                                .filter_map(|frame| {
                                    frame
                                        .metadata
                                        .attributes
                                        .get("capture_event_received_unix_ns")
                                        .and_then(|value| value.parse::<u128>().ok())
                                })
                                .map(|received| set_processing_unix_ns.saturating_sub(received))
                                .max()
                                .unwrap_or(0);
                            synchronizer_waits_ms.push(synchronizer_wait_ns as f64 / 1e6);
                            #[cfg(feature = "fiducials")]
                            if let Some(pipeline) = fiducial_pipeline.as_mut() {
                                if fiducial_set_due(
                                    set.timestamp_ns,
                                    fiducial_last_processed_ns,
                                    fiducial_min_interval_ns,
                                ) {
                                    fiducial_last_processed_ns = Some(set.timestamp_ns);
                                    let fiducial_started = Instant::now();
                                    pipeline.process(&set)?;
                                    fiducial_processing_ms
                                        .push(fiducial_started.elapsed().as_secs_f64() * 1000.0);
                                } else {
                                    fiducial_sets_rate_limited =
                                        fiducial_sets_rate_limited.saturating_add(1);
                                    if let (Some(interval), Some(last)) =
                                        (fiducial_min_interval_ns, fiducial_last_processed_ns)
                                    {
                                        let elapsed = (set.timestamp_ns - last).max(0);
                                        fiducial_rate_limit_remaining_ms
                                            .push((interval - elapsed).max(0) as f64 / 1e6);
                                    }
                                }
                            }
                            if let Some(publisher) = publisher.as_mut() {
                                let due = match (socket_min_interval, socket_last_published) {
                                    (Some(interval), Some(last)) => last.elapsed() >= interval,
                                    _ => true,
                                };
                                if due {
                                    socket_last_published = Some(Instant::now());
                                    if socket_scale == 1.0 && socket_crops.is_empty() {
                                        publisher.publish(&set)?;
                                    } else {
                                        let mut transport_set = if socket_crops.is_empty() {
                                            set.clone()
                                        } else {
                                            crop_video_set(&set, &socket_crops, "transport")?
                                        };
                                        if socket_scale != 1.0 {
                                            transport_set = scale_video_set(
                                                &transport_set,
                                                socket_scale,
                                                "transport",
                                            )?;
                                        }
                                        publisher.publish(&transport_set)?;
                                    }
                                }
                            }
                            if let Some(sync_index) = sync_index.as_mut() {
                                let frame_sequences = set
                                    .frames
                                    .iter()
                                    .map(|(name, frame)| (name.clone(), frame.metadata.sequence))
                                    .collect();
                                serde_json::to_writer(
                                    &mut *sync_index,
                                    &SyncIndexEntry {
                                        sequence: set.sequence,
                                        timestamp_basis: set.timestamp_basis.clone(),
                                        timestamp_ns: set.timestamp_ns,
                                        maximum_skew_ns: set.maximum_skew_ns,
                                        frame_sequences,
                                    },
                                )?;
                                sync_index.write_all(b"\n")?;
                            }
                            #[cfg(feature = "rerun")]
                            if let Some(sink) = rerun_sink.as_ref() {
                                let due = match (rerun_min_interval, rerun_last_logged) {
                                    (Some(interval), Some(last)) => last.elapsed() >= interval,
                                    _ => true,
                                };
                                if due {
                                    rerun_last_logged = Some(Instant::now());
                                    sink.submit(set);
                                }
                            }
                            synchronized_sets = synchronized_sets.saturating_add(1);
                        }
                    }
                    Ok(PoeWorkerEvent::Error { sensor, message }) => {
                        errors.push(format!("{sensor}: {message}"));
                    }
                    Ok(PoeWorkerEvent::Finished {
                        sensor,
                        health: value,
                    }) => {
                        finished += 1;
                        health.insert(sensor, value);
                    }
                    Err(mpsc::RecvTimeoutError::Timeout) => {
                        anyhow::bail!(
                            "PoE capture workers did not finish within the safety timeout"
                        )
                    }
                    Err(mpsc::RecvTimeoutError::Disconnected) => break,
                }
            }
            for worker in workers {
                worker
                    .join()
                    .map_err(|_| anyhow::anyhow!("PoE capture worker panicked"))?;
            }
            let manifests: Vec<_> = recorders
                .map(|map| {
                    map.into_values()
                        .map(EvidenceRecorder::finish)
                        .collect::<Result<Vec<_>, _>>()
                })
                .transpose()?
                .unwrap_or_default();
            if let Some(sync_index) = sync_index.as_mut() {
                sync_index.flush()?;
            }
            #[cfg(feature = "fiducials")]
            if let Some(pipeline) = fiducial_pipeline.as_mut() {
                pipeline.flush()?;
                println!("fiducial_rows={}", pipeline.rows);
                println!(
                    "fiducial_roi_scans={{\"roi\":{},\"full\":{},\"backoff_skipped\":{}}}",
                    pipeline.roi_camera_scans,
                    pipeline.full_camera_scans,
                    pipeline.backoff_skipped_camera_scans
                );
            }
            #[cfg(feature = "rerun")]
            if let Some(sink) = rerun_sink {
                println!("rerun_sink={}", serde_json::to_string(&sink.finish()?)?);
            }
            if no_record {
                println!("live view finished (no evidence recorded)");
            } else {
                println!(
                    "captured PoE stream {stream:?} into {}",
                    output_root.display()
                );
            }
            println!("frame_counts={}", serde_json::to_string(&frame_counts)?);
            let capture_age_summary: BTreeMap<_, _> = pipeline_capture_ages_ms
                .iter()
                .filter_map(|(sensor, samples)| {
                    timing_summary(samples).map(|summary| (sensor, summary))
                })
                .collect();
            println!(
                "pipeline_capture_age_ms={}",
                serde_json::to_string(&capture_age_summary)?
            );
            let stage_latency_summary: BTreeMap<_, _> = pipeline_stage_latencies_ms
                .iter()
                .map(|(sensor, stages)| {
                    let summaries: BTreeMap<_, _> = stages
                        .iter()
                        .filter_map(|(stage, samples)| {
                            timing_summary(samples).map(|summary| (stage, summary))
                        })
                        .collect();
                    (sensor, summaries)
                })
                .collect();
            println!(
                "pipeline_stage_latency_ms={}",
                serde_json::to_string(&stage_latency_summary)?
            );
            let channel_wait_summary: BTreeMap<_, _> = capture_event_channel_waits_ms
                .iter()
                .filter_map(|(sensor, samples)| {
                    timing_summary(samples).map(|summary| (sensor, summary))
                })
                .collect();
            println!(
                "capture_event_channel_wait_ms={}",
                serde_json::to_string(&channel_wait_summary)?
            );
            println!(
                "synchronizer_wait_ms={}",
                serde_json::to_string(&timing_summary(&synchronizer_waits_ms))?
            );
            #[cfg(feature = "fiducials")]
            println!(
                "fiducial_processing_ms={}",
                serde_json::to_string(&timing_summary(&fiducial_processing_ms))?
            );
            #[cfg(feature = "fiducials")]
            println!(
                "fiducial_rate_limit_remaining_ms={}",
                serde_json::to_string(&timing_summary(&fiducial_rate_limit_remaining_ms))?
            );
            println!("errors={}", serde_json::to_string(&errors)?);
            println!("health={}", serde_json::to_string(&health)?);
            println!("synchronized_sets={synchronized_sets}");
            println!("capture_event_queue_capacity={worker_queue_capacity}");
            println!("frames_discarded_after_deadline={frames_discarded_after_deadline}");
            #[cfg(feature = "fiducials")]
            println!("fiducial_sets_rate_limited={fiducial_sets_rate_limited}");
            println!(
                "synchronizer_dropped_unmatched={}",
                synchronizer.dropped_unmatched()
            );
            println!(
                "synchronizer_complete_sets={}",
                synchronizer.complete_sets()
            );
            println!("synchronizer_partial_sets={}", synchronizer.partial_sets());
            if let Some(publisher) = publisher.as_ref() {
                println!("transport_clients={}", publisher.client_count());
            }
            println!("manifests={}", serde_json::to_string(&manifests)?);
        }
        #[cfg(feature = "gstreamer")]
        Command::MonitorPoe {
            config,
            stream,
            bind_host,
            port,
            duration_seconds,
        } => {
            let config = VisionConfig::load(config)?;
            let stream = match stream.as_str() {
                "main" => PoeStream::Main,
                "sub" => PoeStream::Sub,
                other => anyhow::bail!("stream must be main or sub, got {other}"),
            };
            tatbot_visiond::monitor::run(config, stream, &bind_host, port, duration_seconds)?;
        }
        #[cfg(feature = "realsense")]
        Command::CaptureRealsense {
            config,
            sensor,
            duration_seconds,
            output,
            calibration,
        } => {
            let config = VisionConfig::load(config)?;
            let calibration = calibration.map(CalibrationBundle::load).transpose()?;
            let camera = config
                .cameras
                .realsense
                .iter()
                .find(|camera| camera.name == sensor)
                .cloned()
                .with_context(|| format!("unknown RealSense sensor {sensor}"))?;
            let output_root = output.unwrap_or_else(|| PathBuf::from(&config.session.record_root));
            let mut color_recorder =
                EvidenceRecorder::create(&output_root, &format!("{}_color", camera.name))?;
            let mut depth_recorder =
                EvidenceRecorder::create(&output_root, &format!("{}_depth", camera.name))?;
            let mut capture = RealsenseCapture::new(camera.clone())?;
            let deadline = Instant::now() + Duration::from_secs(duration_seconds);
            let mut framesets = 0_u64;
            while Instant::now() < deadline {
                if let Some(frames) = capture.next_frames(Duration::from_millis(1500))? {
                    for frame in frames {
                        let mut frame = frame;
                        stamp_calibration(&mut frame, calibration.as_ref())?;
                        if frame.metadata.sensor_name.ends_with("_color") {
                            color_recorder.write(&frame)?;
                        } else if frame.metadata.sensor_name.ends_with("_depth") {
                            depth_recorder.write(&frame)?;
                        }
                    }
                    framesets += 1;
                }
            }
            let color_manifest = color_recorder.finish()?;
            let depth_manifest = depth_recorder.finish()?;
            let health = capture.health();
            capture.stop();
            println!(
                "captured {} framesets for {} into {}",
                framesets,
                camera.name,
                output_root.display()
            );
            println!("health={}", serde_json::to_string(&health)?);
            println!("color_manifest={}", serde_json::to_string(&color_manifest)?);
            println!("depth_manifest={}", serde_json::to_string(&depth_manifest)?);
        }
        #[cfg(feature = "realsense")]
        Command::CaptureRealsenseAll {
            config,
            duration_seconds,
            output,
            calibration,
            socket,
            #[cfg(feature = "rerun")]
            rerun_output,
            #[cfg(feature = "rerun")]
            rerun_spawn,
            #[cfg(feature = "rerun")]
            rerun_connect,
            #[cfg(feature = "rerun")]
            rerun_max_fps,
            #[cfg(feature = "rerun")]
            rerun_image_scale,
            #[cfg(feature = "rerun")]
            rerun_recording_id,
            #[cfg(feature = "rerun")]
            rerun_layout,
            no_record,
        } => {
            let config = VisionConfig::load(config)?;
            #[cfg(feature = "rerun")]
            if !rerun_image_scale.is_finite()
                || !(0.0..=1.0).contains(&rerun_image_scale)
                || rerun_image_scale == 0.0
            {
                anyhow::bail!("--rerun-image-scale must be in (0, 1]");
            }
            #[cfg(feature = "rerun")]
            let rerun_viewer = open_rerun_viewer(
                rerun_output,
                rerun_spawn,
                rerun_connect,
                rerun_recording_id.as_deref(),
                rerun_layout,
            )?;
            if config.cameras.realsense.len() < 2 {
                anyhow::bail!("CaptureRealsenseAll requires at least two RealSense cameras");
            }
            if no_record && output.is_some() {
                anyhow::bail!("--no-record and --output are mutually exclusive");
            }
            let calibration = calibration.map(CalibrationBundle::load).transpose()?;
            #[cfg(feature = "rerun")]
            if let Some(viewer) = rerun_viewer.as_ref() {
                viewer.log_session_metadata(
                    "capture_realsense_all",
                    rerun_recording_id.as_deref(),
                    None,
                    calibration.as_ref().map(|bundle| bundle.bundle_id.as_str()),
                )?;
            }
            #[cfg(feature = "rerun")]
            let rerun_sink = rerun_viewer.map(RerunSink::new);
            #[cfg(feature = "rerun")]
            let rerun_min_interval =
                (rerun_max_fps > 0.0).then(|| Duration::from_secs_f64(1.0 / rerun_max_fps));
            #[cfg(feature = "rerun")]
            let mut rerun_last_logged: Option<Instant> = None;
            let output_root = output.unwrap_or_else(|| PathBuf::from(&config.session.record_root));
            let mut recorders = BTreeMap::new();
            if !no_record {
                for camera in &config.cameras.realsense {
                    for stream in ["color", "depth"] {
                        let sensor = format!("{}_{}", camera.name, stream);
                        recorders.insert(
                            sensor.clone(),
                            EvidenceRecorder::create(&output_root, &sensor)?,
                        );
                    }
                }
            }
            let sensor_names: Vec<String> = config
                .cameras
                .realsense
                .iter()
                .flat_map(|camera| {
                    [
                        format!("{}_color", camera.name),
                        format!("{}_depth", camera.name),
                    ]
                })
                .collect();
            let tolerance_ns = (config.sync.max_pairwise_skew_ms * 1_000_000.0) as u128;
            let mut synchronizer =
                FrameSynchronizer::new(sensor_names, tolerance_ns, config.session.queue_capacity)
                    .map_err(anyhow::Error::msg)?;
            let sync_index_path = output_root.join("synchronized_frames.jsonl");
            let mut sync_index = if no_record {
                None
            } else {
                Some(BufWriter::new(
                    OpenOptions::new()
                        .create_new(true)
                        .write(true)
                        .open(&sync_index_path)
                        .with_context(|| format!("opening {}", sync_index_path.display()))?,
                ))
            };
            let mut publisher = socket.map(UnixFramePublisher::bind).transpose()?;
            let deadline = Instant::now() + Duration::from_secs(duration_seconds);
            let (sender, receiver) = mpsc::channel();
            let mut workers = Vec::new();
            for camera in config.cameras.realsense.clone() {
                let sender = sender.clone();
                workers.push(thread::spawn(move || {
                    let sensor = camera.name.clone();
                    let mut capture = None;
                    let mut had_capture = false;
                    let mut reconnects = 0_u64;
                    let mut recent_errors = Vec::new();
                    while Instant::now() < deadline {
                        if capture.is_none() {
                            match RealsenseCapture::new(camera.clone()) {
                                Ok(value) => {
                                    if had_capture {
                                        reconnects = reconnects.saturating_add(1);
                                    }
                                    had_capture = true;
                                    capture = Some(value);
                                }
                                Err(error) => {
                                    recent_errors.push(error.to_string());
                                    let _ = sender.send(RealsenseWorkerEvent::Error {
                                        sensor: sensor.clone(),
                                        message: error.to_string(),
                                    });
                                    thread::sleep(Duration::from_millis(250));
                                    continue;
                                }
                            }
                        }
                        let result = capture
                            .as_mut()
                            .expect("RealSense capture initialized")
                            .next_frames(Duration::from_millis(1500));
                        match result {
                            Ok(Some(frames)) => {
                                for frame in frames {
                                    if sender.send(RealsenseWorkerEvent::Frame(frame)).is_err() {
                                        return;
                                    }
                                }
                            }
                            Ok(None) => {}
                            Err(error) => {
                                recent_errors.push(error.to_string());
                                let _ = sender.send(RealsenseWorkerEvent::Error {
                                    sensor: sensor.clone(),
                                    message: error.to_string(),
                                });
                                if let Some(mut value) = capture.take() {
                                    value.stop();
                                }
                                thread::sleep(Duration::from_millis(250));
                            }
                        }
                    }
                    let mut health = capture
                        .as_ref()
                        .map(RealsenseCapture::health)
                        .unwrap_or_else(|| {
                            tatbot_visiond::SensorHealth::new(sensor.clone()).snapshot()
                        });
                    health.reconnects = health.reconnects.saturating_add(reconnects);
                    health.recent_errors = recent_errors.into_iter().rev().take(16).collect();
                    health.recent_errors.reverse();
                    if let Some(mut value) = capture {
                        value.stop();
                    }
                    let _ = sender.send(RealsenseWorkerEvent::Finished { sensor, health });
                }));
            }
            drop(sender);

            let expected = config.cameras.realsense.len();
            let mut finished = 0;
            let mut frame_counts = BTreeMap::<String, u64>::new();
            let mut errors = Vec::new();
            let mut health = BTreeMap::new();
            let mut synchronized_sets = 0_u64;
            while finished < expected {
                match receiver.recv_timeout(Duration::from_secs(15)) {
                    Ok(RealsenseWorkerEvent::Frame(frame)) => {
                        let mut frame = frame;
                        stamp_calibration(&mut frame, calibration.as_ref())?;
                        let sensor = frame.metadata.sensor_name.clone();
                        *frame_counts.entry(sensor.clone()).or_default() += 1;
                        if let Some(recorder) = recorders.get_mut(&sensor) {
                            recorder.write(&frame)?;
                        } else if !no_record {
                            anyhow::bail!("no recorder for {sensor}");
                        }
                        for set in synchronizer.push(frame).map_err(anyhow::Error::msg)? {
                            if let Some(publisher) = publisher.as_mut() {
                                publisher.publish(&set)?;
                            }
                            if let Some(sync_index) = sync_index.as_mut() {
                                let frame_sequences = set
                                    .frames
                                    .iter()
                                    .map(|(name, frame)| (name.clone(), frame.metadata.sequence))
                                    .collect();
                                serde_json::to_writer(
                                    &mut *sync_index,
                                    &SyncIndexEntry {
                                        sequence: set.sequence,
                                        timestamp_basis: set.timestamp_basis.clone(),
                                        timestamp_ns: set.timestamp_ns,
                                        maximum_skew_ns: set.maximum_skew_ns,
                                        frame_sequences,
                                    },
                                )?;
                                sync_index.write_all(b"\n")?;
                            }
                            #[cfg(feature = "rerun")]
                            if let Some(sink) = rerun_sink.as_ref() {
                                let due = match (rerun_min_interval, rerun_last_logged) {
                                    (Some(interval), Some(last)) => last.elapsed() >= interval,
                                    _ => true,
                                };
                                if due {
                                    rerun_last_logged = Some(Instant::now());
                                    let set = if rerun_image_scale != 1.0 {
                                        scale_depth_set(
                                            &scale_video_set(
                                                &set,
                                                rerun_image_scale,
                                                "visualization",
                                            )?,
                                            rerun_image_scale,
                                            "visualization",
                                        )
                                    } else {
                                        set
                                    };
                                    sink.submit(set);
                                }
                            }
                            synchronized_sets = synchronized_sets.saturating_add(1);
                        }
                    }
                    Ok(RealsenseWorkerEvent::Error { sensor, message }) => {
                        errors.push(format!("{sensor}: {message}"));
                    }
                    Ok(RealsenseWorkerEvent::Finished {
                        sensor,
                        health: value,
                    }) => {
                        finished += 1;
                        health.insert(sensor, value);
                    }
                    Err(mpsc::RecvTimeoutError::Timeout) => {
                        anyhow::bail!(
                            "RealSense capture workers did not finish within the safety timeout"
                        )
                    }
                    Err(mpsc::RecvTimeoutError::Disconnected) => break,
                }
            }
            for worker in workers {
                worker
                    .join()
                    .map_err(|_| anyhow::anyhow!("RealSense capture worker panicked"))?;
            }
            let manifests: Vec<_> = recorders
                .into_values()
                .map(EvidenceRecorder::finish)
                .collect::<Result<_, _>>()?;
            if let Some(sync_index) = sync_index.as_mut() {
                sync_index.flush()?;
            }
            #[cfg(feature = "rerun")]
            if let Some(sink) = rerun_sink {
                println!("rerun_sink={}", serde_json::to_string(&sink.finish()?)?);
            }
            if no_record {
                println!("captured RealSense streams (live view, nothing recorded)");
            } else {
                println!("captured RealSense streams into {}", output_root.display());
            }
            println!("frame_counts={}", serde_json::to_string(&frame_counts)?);
            println!("errors={}", serde_json::to_string(&errors)?);
            println!("health={}", serde_json::to_string(&health)?);
            println!("synchronized_sets={synchronized_sets}");
            println!(
                "synchronizer_dropped_unmatched={}",
                synchronizer.dropped_unmatched()
            );
            if let Some(publisher) = publisher.as_ref() {
                println!("transport_clients={}", publisher.client_count());
            }
            println!("manifests={}", serde_json::to_string(&manifests)?);
        }
    }
    Ok(())
}

#[cfg(feature = "rerun")]
fn decimate_replay_rows(rows: &mut Vec<(i128, usize, usize)>, source_count: usize, max_fps: f64) {
    if max_fps == 0.0 {
        return;
    }
    let interval_ns = (1e9 / max_fps).round() as i128;
    let mut last_by_source = vec![None; source_count];
    rows.retain(|(timestamp_ns, source_index, _)| {
        let due =
            last_by_source[*source_index].is_none_or(|last| *timestamp_ns - last >= interval_ns);
        if due {
            last_by_source[*source_index] = Some(*timestamp_ns);
        }
        due
    });
}

#[cfg(all(feature = "rerun", any(feature = "gstreamer", feature = "realsense")))]
fn open_rerun_viewer(
    output: Option<PathBuf>,
    spawn: bool,
    connect: Option<String>,
    recording_id: Option<&str>,
    layout: RerunLayout,
) -> Result<Option<tatbot_visiond::RerunViewer>> {
    if [output.is_some(), spawn, connect.is_some()]
        .iter()
        .filter(|set| **set)
        .count()
        > 1
    {
        anyhow::bail!("choose one of --rerun-output, --rerun-spawn, --rerun-connect");
    }
    if spawn {
        return Ok(Some(tatbot_visiond::RerunViewer::spawn(layout)?));
    }
    if let Some(url) = connect {
        // Remote live view: JPEG-encode color so five decoded streams fit on
        // the LAN (raw substream BGR alone would be ~100 MB/s).
        let mut viewer = tatbot_visiond::RerunViewer::connect(&url, recording_id, Some(layout))?;
        viewer.set_jpeg_quality(Some(80));
        return Ok(Some(viewer));
    }
    output
        .map(|path| tatbot_visiond::RerunViewer::save(path, layout))
        .transpose()
}

#[cfg(any(feature = "rerun", feature = "fiducials"))]
struct ReplaySource {
    root: PathBuf,
    index: Vec<SyncIndexEntry>,
    entries: BTreeMap<String, BTreeMap<u64, tatbot_visiond::RecordingEntry>>,
    metadata_paths: BTreeMap<String, PathBuf>,
}

#[cfg(any(feature = "rerun", feature = "fiducials"))]
impl ReplaySource {
    fn load(root: PathBuf) -> Result<Self> {
        let index_path = root.join("synchronized_frames.jsonl");
        let index_file =
            File::open(&index_path).with_context(|| format!("opening {}", index_path.display()))?;
        let mut index = Vec::new();
        for (line_number, line) in BufReader::new(index_file).lines().enumerate() {
            let line = line.with_context(|| {
                format!("reading {} line {}", index_path.display(), line_number + 1)
            })?;
            if line.trim().is_empty() {
                continue;
            }
            index.push(
                serde_json::from_str::<SyncIndexEntry>(&line).with_context(|| {
                    format!("parsing {} line {}", index_path.display(), line_number + 1)
                })?,
            );
        }
        if index.is_empty() {
            anyhow::bail!("{} contains no synchronized sets", index_path.display());
        }

        let sensor_names: Vec<_> = index[0].frame_sequences.keys().cloned().collect();
        let mut entries: BTreeMap<String, BTreeMap<u64, tatbot_visiond::RecordingEntry>> =
            BTreeMap::new();
        let mut metadata_paths = BTreeMap::new();
        for sensor in sensor_names {
            let metadata_path = root.join(&sensor).join("frames.jsonl");
            let sensor_entries: BTreeMap<u64, tatbot_visiond::RecordingEntry> =
                read_recording_entries(&metadata_path)?
                    .into_iter()
                    .map(|entry| (entry.metadata.sequence, entry))
                    .collect();
            metadata_paths.insert(sensor.clone(), metadata_path);
            entries.insert(sensor, sensor_entries);
        }

        Ok(Self {
            root,
            index,
            entries,
            metadata_paths,
        })
    }

    fn frame_set(&self, row: &SyncIndexEntry) -> Result<SynchronizedFrameSet> {
        let mut frames = BTreeMap::new();
        for (sensor, sequence) in &row.frame_sequences {
            let entry = self
                .entries
                .get(sensor)
                .and_then(|sensor_entries| sensor_entries.get(sequence))
                .with_context(|| {
                    format!(
                        "missing {sensor} sequence {sequence} referenced by {}",
                        self.root.join("synchronized_frames.jsonl").display()
                    )
                })?;
            let frame = read_recording_frame(
                self.metadata_paths
                    .get(sensor)
                    .expect("metadata path exists"),
                entry,
            )?;
            frames.insert(sensor.clone(), frame);
        }
        Ok(SynchronizedFrameSet {
            sequence: row.sequence,
            timestamp_basis: row.timestamp_basis.clone(),
            timestamp_ns: row.timestamp_ns,
            maximum_skew_ns: row.maximum_skew_ns,
            frames,
        })
    }

    fn has_sensor_kind(&self, kind: SensorKind) -> bool {
        self.entries
            .values()
            .flat_map(|entries| entries.values())
            .any(|entry| entry.metadata.sensor_kind == kind)
    }
}

#[cfg(any(
    feature = "gstreamer",
    feature = "realsense",
    feature = "rerun",
    feature = "fiducials"
))]
#[derive(Debug, Serialize)]
#[cfg_attr(
    any(feature = "rerun", feature = "fiducials"),
    derive(Deserialize, Clone)
)]
struct SyncIndexEntry {
    sequence: u64,
    timestamp_basis: String,
    timestamp_ns: i128,
    maximum_skew_ns: u128,
    frame_sequences: BTreeMap<String, u64>,
}

#[cfg(all(test, feature = "rerun"))]
mod replay_tests {
    use super::decimate_replay_rows;

    #[test]
    fn decimation_is_independent_per_recording_source() {
        let mut rows = vec![
            (0, 0, 0),
            (10_000_000, 1, 0),
            (40_000_000, 0, 1),
            (60_000_000, 1, 1),
            (110_000_000, 0, 2),
            (120_000_000, 1, 2),
        ];
        decimate_replay_rows(&mut rows, 2, 10.0);
        assert_eq!(
            rows,
            vec![
                (0, 0, 0),
                (10_000_000, 1, 0),
                (110_000_000, 0, 2),
                (120_000_000, 1, 2)
            ]
        );
    }
}

#[cfg(feature = "fiducials")]
#[derive(Debug, Serialize)]
struct FiducialDetectionBatch {
    schema_version: u32,
    sequence: u64,
    timestamp_ns: i128,
    maximum_skew_ns: u128,
    inventory_hash: String,
    calibration_id: String,
    queue_latency_ms: f64,
    detection_latency_ms: f64,
    image_prep_latency_ms: f64,
    apriltag_latency_ms: f64,
    roi_camera_count: usize,
    processing_latency_ms: f64,
    latency_basis: String,
    latency_ms: f64,
    detections: BTreeMap<String, Vec<FiducialDetection>>,
}

#[cfg(feature = "fiducials")]
impl FiducialDetectionBatch {
    fn new(
        set: &SynchronizedFrameSet,
        inventory_hash: &str,
        calibration_id: &str,
        queue_latency_ms: f64,
        detection_latency_ms: f64,
        image_prep_latency_ms: f64,
        apriltag_latency_ms: f64,
        roi_camera_count: usize,
        latency_ms: f64,
        latency_basis: &str,
        detections: Vec<FiducialDetection>,
    ) -> Self {
        let mut grouped = BTreeMap::<String, Vec<FiducialDetection>>::new();
        for detection in detections {
            grouped
                .entry(detection.camera.clone())
                .or_default()
                .push(detection);
        }
        Self {
            schema_version: 1,
            sequence: set.sequence,
            timestamp_ns: set.timestamp_ns,
            maximum_skew_ns: set.maximum_skew_ns,
            inventory_hash: inventory_hash.to_owned(),
            calibration_id: calibration_id.to_owned(),
            queue_latency_ms,
            detection_latency_ms,
            image_prep_latency_ms,
            apriltag_latency_ms,
            roi_camera_count,
            processing_latency_ms: (latency_ms - queue_latency_ms).max(0.0),
            latency_basis: latency_basis.to_owned(),
            latency_ms,
            detections: grouped,
        }
    }
}

#[cfg(all(feature = "fiducials", feature = "gstreamer"))]
struct FiducialPipeline {
    inventory_hash: String,
    calibration: CalibrationBundle,
    detector: AprilTagDetectorFactory,
    excluded_cameras: std::collections::BTreeSet<String>,
    roi_margin_px: usize,
    full_scan_period: usize,
    roi_hold_frames: usize,
    reacquire_period: usize,
    max_capture_age_ms: f64,
    roi_by_camera: BTreeMap<String, DetectionRoi>,
    roi_misses: BTreeMap<String, usize>,
    roi_camera_scans: u64,
    full_camera_scans: u64,
    backoff_skipped_camera_scans: u64,
    tracker: Option<RustEeTracker>,
    writer: BufWriter<File>,
    rows: u64,
}

#[cfg(all(feature = "fiducials", feature = "gstreamer"))]
impl FiducialPipeline {
    fn new(
        inventory_path: PathBuf,
        wrist_layout_path: Option<PathBuf>,
        output_path: PathBuf,
        scale: Option<f64>,
        excluded_cameras: Vec<String>,
        roi_margin_px: usize,
        full_scan_period: usize,
        roi_hold_frames: usize,
        reacquire_period: usize,
        max_capture_age_ms: f64,
        calibration: CalibrationBundle,
    ) -> Result<Self> {
        let inventory = FiducialInventory::load(inventory_path)?;
        // Detection-only surveys every configured mounted target. Pose mode
        // narrows the detector to the wrist before the estimator sees data.
        let detector_target = wrist_layout_path.as_ref().map(|_| "wrist");
        let detector = AprilTagDetectorFactory::new(&inventory, detector_target, scale)?;
        let tracker = wrist_layout_path
            .map(|path| {
                let layout = WristLayout::load(path, &inventory, false)?;
                RustEeTracker::new(&calibration, &inventory, layout, EstimatorConfig::default())
            })
            .transpose()?;
        let writer = BufWriter::new(
            OpenOptions::new()
                .create_new(true)
                .write(true)
                .open(&output_path)
                .with_context(|| format!("opening {}", output_path.display()))?,
        );
        Ok(Self {
            inventory_hash: inventory.inventory_hash,
            calibration,
            detector,
            excluded_cameras: excluded_cameras.into_iter().collect(),
            roi_margin_px,
            full_scan_period,
            roi_hold_frames,
            reacquire_period,
            max_capture_age_ms,
            roi_by_camera: BTreeMap::new(),
            roi_misses: BTreeMap::new(),
            roi_camera_scans: 0,
            full_camera_scans: 0,
            backoff_skipped_camera_scans: 0,
            tracker,
            writer,
            rows: 0,
        })
    }

    fn process(&mut self, set: &SynchronizedFrameSet) -> Result<()> {
        // The previous `latency_ms` started here and therefore measured only
        // detector/solver CPU time.  It hid seconds of queued-frame age during
        // the exact overload this metric is supposed to catch.  The
        // synchronized timestamp is normalized Unix time for live PoE sets,
        // so include capture-to-processing age and clamp small clock noise.
        let now_ns = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or(Duration::ZERO)
            .as_nanos() as i128;
        let queue_latency_ms = ((now_ns - set.timestamp_ns).max(0) as f64) / 1e6;
        let started = Instant::now();
        if queue_latency_ms > self.max_capture_age_ms
            && let Some(tracker) = self.tracker.as_mut()
        {
            let mut estimate = tracker.update_constrained(
                set.sequence,
                set.timestamp_ns,
                set.maximum_skew_ns,
                Vec::new(),
                queue_latency_ms,
                0.0,
                started,
                2,
            );
            estimate.input_cameras = set.frames.keys().cloned().collect();
            estimate.partial_input = Some(set.frames.len() < self.calibration.cameras.len());
            estimate.reason = Some(format!(
                "capture age {queue_latency_ms:.1} ms exceeds {:.1} ms tracker bound",
                self.max_capture_age_ms
            ));
            serde_json::to_writer(&mut self.writer, &estimate)?;
            self.writer.write_all(b"\n")?;
            self.rows = self.rows.saturating_add(1);
            return Ok(());
        }
        let (rois, excluded, scanned_cameras) = self.detection_plan(set);
        let detected =
            self.detector
                .detect_set_profiled(&self.calibration, set, &excluded, &rois)?;
        let detection_latency_ms = started.elapsed().as_secs_f64() * 1000.0;
        self.roi_camera_scans = self
            .roi_camera_scans
            .saturating_add(detected.roi_camera_count as u64);
        self.full_camera_scans = self
            .full_camera_scans
            .saturating_add(scanned_cameras.saturating_sub(detected.roi_camera_count) as u64);
        let configured_camera_count = set
            .frames
            .keys()
            .filter(|name| !self.excluded_cameras.contains(*name))
            .count();
        self.backoff_skipped_camera_scans = self
            .backoff_skipped_camera_scans
            .saturating_add(configured_camera_count.saturating_sub(scanned_cameras) as u64);
        self.update_rois(set, &detected.detections, &excluded);
        if let Some(tracker) = self.tracker.as_mut() {
            // A bounded partial camera set must still contain two distinct
            // physical wrist IDs before it can refresh a measured pose. With
            // less geometry the tracker falls back to its bounded prediction
            // or unavailable state instead of silently accepting a weak fix.
            let minimum_tag_ids =
                usize::from(set.frames.len() < self.calibration.cameras.len()) * 2;
            let mut estimate = tracker.update_constrained(
                set.sequence,
                set.timestamp_ns,
                set.maximum_skew_ns,
                detected.detections,
                queue_latency_ms,
                detection_latency_ms,
                started,
                minimum_tag_ids,
            );
            estimate.input_cameras = set.frames.keys().cloned().collect();
            estimate.partial_input = Some(set.frames.len() < self.calibration.cameras.len());
            estimate.image_prep_latency_ms = detected.image_prep_latency_ms;
            estimate.apriltag_latency_ms = detected.apriltag_latency_ms;
            estimate.roi_camera_count = detected.roi_camera_count;
            serde_json::to_writer(&mut self.writer, &estimate)?;
        } else {
            let batch = FiducialDetectionBatch::new(
                set,
                &self.inventory_hash,
                &self.calibration.bundle_id,
                queue_latency_ms,
                detection_latency_ms,
                detected.image_prep_latency_ms,
                detected.apriltag_latency_ms,
                detected.roi_camera_count,
                queue_latency_ms + started.elapsed().as_secs_f64() * 1000.0,
                "capture_to_estimate",
                detected.detections,
            );
            serde_json::to_writer(&mut self.writer, &batch)?;
        }
        self.writer.write_all(b"\n")?;
        self.rows = self.rows.saturating_add(1);
        Ok(())
    }

    fn detection_plan(
        &self,
        set: &SynchronizedFrameSet,
    ) -> (
        BTreeMap<String, DetectionRoi>,
        std::collections::BTreeSet<String>,
        usize,
    ) {
        if self.roi_margin_px == 0 {
            let scanned = set
                .frames
                .keys()
                .filter(|name| !self.excluded_cameras.contains(*name))
                .count();
            return (BTreeMap::new(), self.excluded_cameras.clone(), scanned);
        }
        let camera_count = set.frames.len().max(1);
        let stagger = (self.full_scan_period / camera_count).max(1);
        let mut rois = BTreeMap::new();
        let mut excluded = self.excluded_cameras.clone();
        let mut scanned = 0_usize;
        for (index, name) in set.frames.keys().enumerate() {
            if self.excluded_cameras.contains(name) {
                continue;
            }
            if let Some(roi) = self.roi_by_camera.get(name).copied() {
                let force_full = self.full_scan_period > 0
                    && self.rows > 0
                    && (self.rows as usize + index * stagger) % self.full_scan_period == 0;
                if !force_full {
                    rois.insert(name.clone(), roi);
                }
                scanned += 1;
                continue;
            }
            let misses = self.roi_misses.get(name).copied().unwrap_or(0);
            let immediate_search = self.rows == 0 || misses < self.roi_hold_frames;
            if immediate_search
                || camera_reacquisition_due(self.rows as usize, index, self.reacquire_period)
            {
                scanned += 1;
            } else {
                excluded.insert(name.clone());
            }
        }
        (rois, excluded, scanned)
    }

    fn update_rois(
        &mut self,
        set: &SynchronizedFrameSet,
        detections: &[FiducialDetection],
        excluded_this_set: &std::collections::BTreeSet<String>,
    ) {
        if self.roi_margin_px == 0 {
            return;
        }
        for (name, frame) in &set.frames {
            if excluded_this_set.contains(name) {
                continue;
            }
            let camera_detections = detections
                .iter()
                .filter(|detection| detection.camera == *name)
                .collect::<Vec<_>>();
            if let Some((width, height)) = decoded_frame_dimensions(frame)
                && let Some(roi) =
                    expanded_detection_roi(&camera_detections, width, height, self.roi_margin_px)
            {
                self.roi_by_camera.insert(name.clone(), roi);
                self.roi_misses.insert(name.clone(), 0);
                continue;
            }
            let misses = self.roi_misses.entry(name.clone()).or_default();
            *misses = misses.saturating_add(1);
            if *misses >= self.roi_hold_frames {
                self.roi_by_camera.remove(name);
            }
        }
    }

    fn flush(&mut self) -> Result<()> {
        self.writer.flush().context("flushing fiducial JSONL")
    }
}

#[cfg(all(feature = "fiducials", feature = "gstreamer"))]
fn decoded_frame_dimensions(frame: &tatbot_visiond::FrameRecord) -> Option<(usize, usize)> {
    match &frame.payload {
        RecordedPayload::Video { width, height, .. } => Some((*width as usize, *height as usize)),
        _ => None,
    }
}

#[cfg(feature = "fiducials")]
fn fiducial_set_due(
    timestamp_ns: i128,
    last_processed_ns: Option<i128>,
    min_interval_ns: Option<i128>,
) -> bool {
    match (last_processed_ns, min_interval_ns) {
        // Camera periods are not exact integer nanoseconds. Comparing the raw
        // delta made nominal 20 Hz frames at 99.9 ms miss a 10 Hz threshold
        // and selected every third frame (~6.7 Hz). One sample per aligned
        // interval keeps the long-run cap while accepting that second frame.
        (Some(last), Some(interval)) if timestamp_ns >= last => {
            timestamp_ns.div_euclid(interval) > last.div_euclid(interval)
        }
        _ => true,
    }
}

#[cfg(feature = "fiducials")]
fn camera_reacquisition_due(row: usize, camera_index: usize, period: usize) -> bool {
    period == 0 || (row + camera_index) % period == 0
}

#[cfg(all(test, feature = "fiducials"))]
mod fiducial_rate_tests {
    use super::{camera_reacquisition_due, fiducial_set_due};

    #[test]
    fn rate_limit_uses_capture_timestamps_and_recovers_from_regression() {
        assert!(fiducial_set_due(1_000, None, Some(100)));
        assert!(!fiducial_set_due(1_099, Some(1_000), Some(100)));
        assert!(fiducial_set_due(1_100, Some(1_000), Some(100)));
        assert!(fiducial_set_due(900, Some(1_000), Some(100)));
        assert!(fiducial_set_due(1_001, Some(1_000), None));
    }

    #[test]
    fn aligned_intervals_accept_nominal_frames_just_below_raw_delta() {
        assert!(fiducial_set_due(
            1_149_900_000,
            Some(1_050_000_000),
            Some(100_000_000)
        ));
    }

    #[test]
    fn absent_camera_reacquisition_is_staggered_and_zero_disables_backoff() {
        assert!(camera_reacquisition_due(10, 0, 5));
        assert!(!camera_reacquisition_due(10, 1, 5));
        assert!(camera_reacquisition_due(14, 1, 5));
        assert!(camera_reacquisition_due(11, 3, 0));
    }
}

#[cfg(any(feature = "gstreamer", feature = "realsense"))]
fn stamp_calibration(
    frame: &mut tatbot_visiond::FrameRecord,
    calibration: Option<&CalibrationBundle>,
) -> Result<()> {
    let Some(calibration) = calibration else {
        return Ok(());
    };
    calibration
        .camera(&frame.metadata.sensor_name, &frame.metadata.profile)
        .map_err(anyhow::Error::msg)?;
    frame.metadata.calibration_id = Some(calibration.bundle_id.clone());
    frame
        .metadata
        .flags
        .push("calibration_bundle_verified".to_string());
    Ok(())
}

#[cfg(feature = "gstreamer")]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct VideoCrop {
    x: u32,
    y: u32,
    width: u32,
    height: u32,
}

#[cfg(feature = "gstreamer")]
fn parse_socket_crops(values: &[String]) -> Result<BTreeMap<String, VideoCrop>> {
    let mut crops = BTreeMap::new();
    for value in values {
        let (camera, coordinates) = value.split_once('=').with_context(|| {
            format!("invalid --socket-crop {value:?}; expected CAMERA=X,Y,WIDTH,HEIGHT")
        })?;
        if camera.is_empty() {
            anyhow::bail!("invalid --socket-crop {value:?}; camera name is empty");
        }
        let parts = coordinates
            .split(',')
            .map(str::parse::<u32>)
            .collect::<std::result::Result<Vec<_>, _>>()
            .with_context(|| format!("invalid --socket-crop {value:?}; coordinates must be u32"))?;
        let [x, y, width, height] = parts.as_slice() else {
            anyhow::bail!("invalid --socket-crop {value:?}; expected four coordinates");
        };
        if *width == 0 || *height == 0 {
            anyhow::bail!("invalid --socket-crop {value:?}; width and height must be positive");
        }
        let crop = VideoCrop {
            x: *x,
            y: *y,
            width: *width,
            height: *height,
        };
        if crops.insert(camera.to_string(), crop).is_some() {
            anyhow::bail!("duplicate --socket-crop for {camera}");
        }
    }
    Ok(crops)
}

#[cfg(feature = "gstreamer")]
fn crop_video_set(
    set: &SynchronizedFrameSet,
    crops: &BTreeMap<String, VideoCrop>,
    purpose: &str,
) -> Result<SynchronizedFrameSet> {
    let mut frames = BTreeMap::new();
    for (name, frame) in &set.frames {
        let crop = crops
            .get(name)
            .with_context(|| format!("no {purpose} crop configured for {name}"))?;
        let (format, width, height, bytes) = match &frame.payload {
            RecordedPayload::Video {
                format,
                width,
                height,
                bytes,
            } if matches!(format, PixelFormat::Bgr8 | PixelFormat::Rgb8) => {
                (*format, *width, *height, bytes)
            }
            _ => anyhow::bail!("visual cropping requires decoded BGR/RGB frames"),
        };
        let x_end = crop
            .x
            .checked_add(crop.width)
            .context("socket crop x extent overflowed")?;
        let y_end = crop
            .y
            .checked_add(crop.height)
            .context("socket crop y extent overflowed")?;
        if x_end > width || y_end > height {
            anyhow::bail!(
                "{purpose} crop for {name} ({},{},{},{}) exceeds {width}x{height}",
                crop.x,
                crop.y,
                crop.width,
                crop.height
            );
        }
        let source_stride = usize::try_from(width)? * 3;
        let output_stride = usize::try_from(crop.width)? * 3;
        let expected = source_stride * usize::try_from(height)?;
        if bytes.len() != expected {
            anyhow::bail!(
                "decoded {name} frame has {} bytes; expected {expected}",
                bytes.len()
            );
        }
        let mut output = Vec::with_capacity(output_stride * usize::try_from(crop.height)?);
        let x_offset = usize::try_from(crop.x)? * 3;
        for y in crop.y..y_end {
            let start = usize::try_from(y)? * source_stride + x_offset;
            output.extend_from_slice(&bytes[start..start + output_stride]);
        }
        let mut metadata = frame.metadata.clone();
        metadata.profile.width = crop.width;
        metadata.profile.height = crop.height;
        metadata.flags.push(format!("{purpose}_cropped"));
        metadata.attributes.insert(
            format!("{purpose}_source_dimensions"),
            format!("{width}x{height}"),
        );
        metadata.attributes.insert(
            format!("{purpose}_crop_xywh"),
            format!("{},{},{},{}", crop.x, crop.y, crop.width, crop.height),
        );
        frames.insert(
            name.clone(),
            tatbot_visiond::FrameRecord {
                metadata,
                payload: RecordedPayload::Video {
                    format,
                    width: crop.width,
                    height: crop.height,
                    bytes: output,
                },
            },
        );
    }
    Ok(SynchronizedFrameSet {
        sequence: set.sequence,
        timestamp_basis: set.timestamp_basis.clone(),
        timestamp_ns: set.timestamp_ns,
        maximum_skew_ns: set.maximum_skew_ns,
        frames,
    })
}

/// Nearest-neighbour downscale of a packed raster whose rows are `units`
/// samples of `unit_bytes` each (a Z16 pixel is one 2-byte unit; a YUYV
/// macropixel is one 4-byte unit covering two pixels). Returns the resized
/// bytes and the new (units, rows) size; `None` if the buffer length does not
/// match the claimed geometry.
#[cfg(any(feature = "gstreamer", feature = "rerun"))]
fn scale_packed_rows(
    bytes: &[u8],
    units: u32,
    rows: u32,
    unit_bytes: usize,
    scale: f64,
) -> Option<(Vec<u8>, u32, u32)> {
    if bytes.len() != (units as usize) * (rows as usize) * unit_bytes {
        return None;
    }
    let new_units = ((units as f64 * scale).round() as u32).max(1);
    let new_rows = ((rows as f64 * scale).round() as u32).max(1);
    let mut resized = Vec::with_capacity((new_units as usize) * (new_rows as usize) * unit_bytes);
    for y in 0..new_rows {
        let src_y = ((y as u64 * rows as u64) / new_rows as u64) as usize;
        let row = &bytes[src_y * units as usize * unit_bytes..];
        for x in 0..new_units {
            let src_x = ((x as u64 * units as u64) / new_units as u64) as usize;
            resized.extend_from_slice(&row[src_x * unit_bytes..(src_x + 1) * unit_bytes]);
        }
    }
    Some((resized, new_units, new_rows))
}

#[cfg(any(feature = "gstreamer", feature = "rerun"))]
fn note_scaled(
    frame: &mut tatbot_visiond::FrameRecord,
    width: u32,
    height: u32,
    new_width: u32,
    new_height: u32,
    scale: f64,
    purpose: &str,
) {
    frame.metadata.profile.width = new_width;
    frame.metadata.profile.height = new_height;
    frame
        .metadata
        .flags
        .push(format!("{purpose}_uniformly_scaled"));
    frame.metadata.attributes.insert(
        format!("{purpose}_source_dimensions"),
        format!("{width}x{height}"),
    );
    frame
        .metadata
        .attributes
        .insert(format!("{purpose}_scale"), scale.to_string());
    frame
        .metadata
        .attributes
        .insert(format!("{purpose}_resize_filter"), "nearest".into());
}

/// Nearest-neighbour downscale of every Z16 depth plane in a set (the
/// companion of `scale_video_set`, which deliberately skips depth). Sample
/// values are untouched, so `depth_units_m` stays valid; the scaled plane is
/// a visualization derivative, never evidence.
#[cfg(all(feature = "rerun", feature = "realsense"))]
fn scale_depth_set(set: &SynchronizedFrameSet, scale: f64, purpose: &str) -> SynchronizedFrameSet {
    let mut output = set.clone();
    for frame in output.frames.values_mut() {
        let RecordedPayload::Depth {
            width,
            height,
            bytes,
        } = &frame.payload
        else {
            continue;
        };
        let (width, height) = (*width, *height);
        let Some((resized, new_width, new_height)) =
            scale_packed_rows(bytes, width, height, 2, scale)
        else {
            continue;
        };
        frame.payload = RecordedPayload::Depth {
            width: new_width,
            height: new_height,
            bytes: resized,
        };
        note_scaled(frame, width, height, new_width, new_height, scale, purpose);
    }
    output
}

#[cfg(any(feature = "gstreamer", feature = "rerun"))]
fn scale_video_set(
    set: &SynchronizedFrameSet,
    scale: f64,
    purpose: &str,
) -> Result<SynchronizedFrameSet> {
    let mut output = set.clone();
    for frame in output.frames.values_mut() {
        let (format, width, height, bytes) = match &frame.payload {
            RecordedPayload::Video {
                format,
                width,
                height,
                bytes,
            } if matches!(format, PixelFormat::Bgr8 | PixelFormat::Rgb8) => {
                (*format, *width, *height, bytes)
            }
            // RealSense colour arrives as YUYV: two pixels share one 4-byte
            // macropixel, so scale in macropixel units and keep the width even.
            RecordedPayload::Video {
                format: PixelFormat::Yuyv,
                width,
                height,
                bytes,
            } => {
                let (width, height) = (*width, *height);
                let Some((resized, new_macro, new_height)) =
                    scale_packed_rows(bytes, width / 2, height, 4, scale)
                else {
                    anyhow::bail!("YUYV frame length does not match its dimensions");
                };
                let new_width = new_macro * 2;
                frame.payload = RecordedPayload::Video {
                    format: PixelFormat::Yuyv,
                    width: new_width,
                    height: new_height,
                    bytes: resized,
                };
                note_scaled(frame, width, height, new_width, new_height, scale, purpose);
                continue;
            }
            RecordedPayload::Depth { .. } => continue,
            _ => anyhow::bail!("visual scaling requires decoded BGR/RGB/YUYV frames"),
        };
        let new_width = ((width as f64 * scale).round() as u32).max(1);
        let new_height = ((height as f64 * scale).round() as u32).max(1);
        let source = image::RgbImage::from_raw(width, height, bytes.clone())
            .context("decoded frame length does not match its dimensions")?;
        let resized = image::imageops::resize(
            &source,
            new_width,
            new_height,
            // The shadow path is latency-sensitive and AprilTag edges are
            // binary. Nearest-neighbour preserves those edges and avoids the
            // CPU saturation observed with five simultaneous Triangle resizes
            // on the Jetson camera node.
            image::imageops::FilterType::Nearest,
        );
        frame.payload = RecordedPayload::Video {
            format,
            width: new_width,
            height: new_height,
            bytes: resized.into_raw(),
        };
        frame.metadata.profile.width = new_width;
        frame.metadata.profile.height = new_height;
        frame
            .metadata
            .flags
            .push(format!("{purpose}_uniformly_scaled"));
        frame.metadata.attributes.insert(
            format!("{purpose}_source_dimensions"),
            format!("{width}x{height}"),
        );
        frame
            .metadata
            .attributes
            .insert(format!("{purpose}_scale"), scale.to_string());
        frame
            .metadata
            .attributes
            .insert(format!("{purpose}_resize_filter"), "nearest".into());
    }
    Ok(output)
}

#[cfg(all(test, feature = "gstreamer"))]
mod socket_crop_tests {
    use super::{VideoCrop, crop_video_set, parse_socket_crops};
    use std::collections::{BTreeMap, BTreeSet};
    use tatbot_visiond::{
        FrameMetadata, FrameRecord, FrameTimestamps, PixelFormat, RecordedPayload, SensorKind,
        StreamProfile, SynchronizedFrameSet, TimestampDomain,
    };

    #[test]
    fn parses_and_refuses_ambiguous_crop_specs() {
        let crops =
            parse_socket_crops(&["camera1=1,2,3,4".to_string(), "camera2=0,0,5,6".to_string()])
                .unwrap();
        assert_eq!(
            crops["camera1"],
            VideoCrop {
                x: 1,
                y: 2,
                width: 3,
                height: 4
            }
        );
        assert!(parse_socket_crops(&["camera1=0,0,0,4".to_string()]).is_err());
        assert!(
            parse_socket_crops(&["camera1=0,0,1,1".to_string(), "camera1=1,1,1,1".to_string(),])
                .is_err()
        );
    }

    #[test]
    fn crops_without_cloning_full_frame_pixels() {
        let frame = FrameRecord {
            metadata: FrameMetadata {
                sensor_name: "camera1".into(),
                sensor_kind: SensorKind::PoE,
                sequence: 1,
                profile: StreamProfile {
                    stream: "main".into(),
                    width: 3,
                    height: 2,
                    fps_num: 20,
                    fps_den: 1,
                    format: PixelFormat::Bgr8,
                },
                timestamps: FrameTimestamps {
                    source_ns: Some(10),
                    source_domain: TimestampDomain::CameraNtp,
                    rtp_timestamp: Some(1),
                    pipeline_pts_ns: Some(2),
                    pipeline_dts_ns: None,
                    host_monotonic_ns: 3,
                    host_unix_ns: 4,
                    normalized_unix_ns: Some(10),
                },
                dropped_before: 0,
                calibration_id: None,
                flags: Vec::new(),
                attributes: BTreeMap::new(),
            },
            payload: RecordedPayload::Video {
                format: PixelFormat::Bgr8,
                width: 3,
                height: 2,
                bytes: (0_u8..18).collect(),
            },
        };
        let set = SynchronizedFrameSet {
            sequence: 1,
            timestamp_basis: "normalized_unix_ns".into(),
            timestamp_ns: 10,
            maximum_skew_ns: 0,
            frames: BTreeMap::from([("camera1".into(), frame)]),
        };
        let output = crop_video_set(
            &set,
            &BTreeMap::from([(
                "camera1".into(),
                VideoCrop {
                    x: 1,
                    y: 0,
                    width: 2,
                    height: 2,
                },
            )]),
            "test",
        )
        .unwrap();
        let cropped = &output.frames["camera1"];
        assert_eq!(cropped.metadata.profile.width, 2);
        assert_eq!(cropped.metadata.profile.height, 2);
        assert!(cropped.metadata.flags.contains(&"test_cropped".into()));
        let RecordedPayload::Video { bytes, .. } = &cropped.payload else {
            panic!("expected video payload");
        };
        assert_eq!(bytes, &[3, 4, 5, 6, 7, 8, 12, 13, 14, 15, 16, 17]);
        assert_eq!(
            output.frames.keys().cloned().collect::<BTreeSet<_>>(),
            BTreeSet::from(["camera1".to_string()])
        );
    }
}

#[cfg(feature = "gstreamer")]
#[derive(Debug)]
enum PoeWorkerEvent {
    Frame {
        frame: tatbot_visiond::FrameRecord,
        enqueued_at: Instant,
    },
    Error {
        sensor: String,
        message: String,
    },
    Finished {
        sensor: String,
        health: tatbot_visiond::HealthSnapshot,
    },
}

#[cfg(feature = "gstreamer")]
#[derive(Debug, Serialize)]
struct TimingSummary {
    samples: usize,
    median: f64,
    p95: f64,
    max: f64,
}

#[cfg(feature = "gstreamer")]
fn timing_summary(samples: &[f64]) -> Option<TimingSummary> {
    if samples.is_empty() {
        return None;
    }
    let mut sorted = samples.to_vec();
    sorted.sort_by(f64::total_cmp);
    let p95_index = ((sorted.len() - 1) as f64 * 0.95).round() as usize;
    Some(TimingSummary {
        samples: sorted.len(),
        median: sorted[sorted.len() / 2],
        p95: sorted[p95_index],
        max: *sorted.last().expect("non-empty timing samples"),
    })
}

#[cfg(all(test, feature = "gstreamer"))]
mod timing_summary_tests {
    use super::timing_summary;

    #[test]
    fn reports_order_independent_percentiles() {
        let summary = timing_summary(&[4.0, 1.0, 3.0, 2.0]).unwrap();
        assert_eq!(summary.samples, 4);
        assert_eq!(summary.median, 3.0);
        assert_eq!(summary.p95, 4.0);
        assert_eq!(summary.max, 4.0);
    }
}

#[cfg(feature = "realsense")]
#[derive(Debug)]
enum RealsenseWorkerEvent {
    Frame(tatbot_visiond::FrameRecord),
    Error {
        sensor: String,
        message: String,
    },
    Finished {
        sensor: String,
        health: tatbot_visiond::HealthSnapshot,
    },
}
