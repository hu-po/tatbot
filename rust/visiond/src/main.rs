use std::path::PathBuf;

#[cfg(feature = "rerun")]
use std::net::UdpSocket;

use anyhow::Result;
use clap::{Parser, Subcommand};
#[cfg(any(feature = "rerun", feature = "fiducials"))]
use serde::Deserialize;
#[cfg(any(
    feature = "realsense",
    feature = "rerun",
    feature = "fiducials"
))]
use serde::Serialize;
use tatbot_visiond::{
    CalibrationBundle, VisionConfig, pairwise_sync_report, read_recording_entries,
};

#[cfg(feature = "realsense")]
use tatbot_visiond::FrameSynchronizer;
#[cfg(feature = "realsense")]
use tatbot_visiond::UnixFramePublisher;
use tracing_subscriber::EnvFilter;
#[cfg(feature = "realsense")]
use tatbot_visiond::frame_ops::{BoundedStrings, bounded_capture_event_channel};
#[cfg(feature = "rerun")]
use tatbot_visiond::frame_ops::decimate_replay_rows;
#[cfg(feature = "rerun")]
use tatbot_visiond::frame_ops::scale_video_set;
// Only capture-realsense-all's Rerun path scales depth, so this import is
// narrower than scale_video_set's: a rerun-only build never reaches it.
#[cfg(all(feature = "realsense", feature = "rerun"))]
use tatbot_visiond::frame_ops::scale_depth_set;

#[cfg(any(feature = "rerun", feature = "fiducials"))]
use std::fs::File;
#[cfg(any(feature = "rerun", feature = "fiducials"))]
use std::io::{BufRead, BufReader};
#[cfg(feature = "realsense")]
use std::sync::mpsc;
#[cfg(any(feature = "realsense", feature = "fiducials"))]
use std::{
    fs::OpenOptions,
    io::{BufWriter, Write},
};

#[cfg(any(
    feature = "realsense",
    feature = "rerun",
    feature = "fiducials"
))]
use std::collections::BTreeMap;
#[cfg(feature = "rerun")]
use std::time::{SystemTime, UNIX_EPOCH};
#[cfg(any(feature = "realsense", feature = "rerun"))]
use std::{thread, time::Duration};

#[cfg(any(
    feature = "realsense",
    feature = "rerun",
    feature = "fiducials"
))]
use std::time::Instant;

use anyhow::Context;

#[cfg(feature = "realsense")]
use tatbot_visiond::EvidenceRecorder;

#[cfg(feature = "realsense")]
use tatbot_visiond::realsense_backend::RealsenseCapture;
#[cfg(feature = "realsense")]
use tatbot_visiond::time::{RawClockProbeFrame, RawDeviceClockProbe, RawDeviceClockRead, RAW_DEVICE_CLOCK_SAMPLE_SCHEMA};

#[cfg(all(feature = "rerun", feature = "realsense"))]
use tatbot_visiond::RerunSink;
#[cfg(any(feature = "rerun", feature = "fiducials"))]
use tatbot_visiond::SynchronizedFrameSet;
#[cfg(any(feature = "rerun", feature = "fiducials"))]
use tatbot_visiond::read_recording_frame;
#[cfg(feature = "fiducials")]
use tatbot_visiond::{
    AprilTagDetectorFactory, EstimatorConfig, FiducialDetection, FiducialInventory, RustEeTracker,
    WristLayout,
};
#[cfg(feature = "rerun")]
use tatbot_visiond::{LiveTeleopTick, RerunViewer, TeleopSetup};

#[derive(Debug, Parser)]
#[command(name = "tatbot-visiond", about = "Tatbot 2.0 vision capture service")]
struct Cli {
    /// Publish camera-owned frames on the fleet bus; never opens another client.
    #[cfg(feature = "zenoh")]
    #[arg(long, global = true)]
    zenoh: bool,
    /// Subscribe to the persistent camera owner instead of opening hardware.
    #[cfg(feature = "zenoh")]
    #[arg(long, global = true)]
    subscribe_socket: Option<PathBuf>,
    #[cfg(feature = "zenoh")]
    #[arg(long, global = true)]
    bus_connect: Vec<String>,
    #[cfg(feature = "zenoh")]
    #[arg(long, global = true, env = "TATBOT_NODE")]
    bus_node: Option<String>,
    #[arg(long, default_value = "info", env = "RUST_LOG")]
    log_filter: String,
    #[command(subcommand)]
    command: Command,
}

#[derive(Debug, Subcommand)]
#[allow(clippy::large_enum_variant)] // One CLI parse at startup, never a frame queue.
enum Command {
    /// Cockpit subscriber: receives bus frames and joins the existing viewer.
    #[cfg(all(feature = "zenoh", feature = "rerun"))]
    Subscribe {
        /// The fleet viewer's gRPC proxy; optional when --output is given.
        #[arg(long, required_unless_present = "output")]
        connect: Option<String>,
        #[arg(long)]
        recording_id: String,
        /// Overall cap on every bus key, Hz; each stream's own rate is beneath it.
        #[arg(long, default_value_t = 5.0)]
        max_fps: f64,
        /// The overhead D555's colour, Hz: a 640x360 JPEG, ~52 KB a frame.
        #[arg(long, default_value_t = tatbot_visiond::frame_bus::CockpitRates::DEFAULT.overhead_color_hz)]
        overhead_color_hz: f64,
        /// The overhead D555's depth cloud, Hz: 320x180 RVL, ~17 KB a frame.
        #[arg(long, default_value_t = tatbot_visiond::frame_bus::CockpitRates::DEFAULT.overhead_depth_hz)]
        overhead_depth_hz: f64,
        /// Each wrist tap's colour, Hz (640x360 JPEG); its depth is not shown
        /// here, the bridge draws the wrist cloud.
        #[arg(long, default_value_t = tatbot_visiond::frame_bus::CockpitRates::DEFAULT.wrist_hz)]
        wrist_hz: f64,
        /// 0 runs until SIGINT or SIGTERM.
        #[arg(long, default_value_t = 0)]
        duration_seconds: u64,
        /// URDF whose visual meshes should be added to the persistent 3D scene.
        #[arg(long, value_name = "URDF")]
        urdf: Option<PathBuf>,
        /// Adopted camera bundle whose calibrated frustums should be shown.
        #[arg(long, value_name = "BUNDLE")]
        calibration: Option<PathBuf>,
        /// Robot-world registration for world-frame tracking overlays.
        #[arg(long)]
        robot_world: Option<PathBuf>,
        /// Draw trackd's and stencild's poses as published, at identity in
        /// the bus's camera world (the ROS `world`), where the Demo tab shows them.
        #[arg(long, conflicts_with_all = ["robot_world", "calibration"])]
        camera_world: bool,
        /// Also write what is shown to this .rrd file (an offline capture).
        #[arg(long, value_name = "FILE.rrd")]
        output: Option<PathBuf>,
        /// Show the overhead D555's depth out to this range, metres, so the
        /// floor and the room do not fill the scene; 0 shows all of it.
        #[arg(long, value_name = "METRES", default_value_t = tatbot_visiond::frame_bus::OVERHEAD_DEPTH_MAX_M)]
        overhead_depth_max_m: f64,
        /// Outline the palette's tag (this fiducial inventory's `palette`
        /// target) on the overhead colour, read by the cockpit itself once a
        /// second; needs a build with the fiducials feature.
        #[arg(long, value_name = "INVENTORY")]
        fiducials: Option<PathBuf>,
    },
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
        /// A retained estimates.jsonl whose rows carry per-camera `detections`.
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
        #[arg(long, default_value = "")]
        calibration_anchor: String,
        /// robot-world JSON from `tatbot ros calib apply`: places the calibration
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
        /// Stream the replay to a viewer elsewhere (the fleet viewer's proxy,
        /// e.g. rerun+http://192.0.2.90:9876/proxy) instead of a file or a
        /// local window.
        #[arg(long, value_name = "URL")]
        connect: Option<String>,
        /// Recording id to stream under with --connect (default: a new one).
        #[arg(long)]
        recording_id: Option<String>,
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
    },
    /// Install the fixed display blueprint in a running viewer and exit.
    /// Only the rerun-server bootstrap uses this command
    /// (`docs/vision.md`).
    #[cfg(feature = "rerun")]
    SendBlueprint {
        /// The viewer's gRPC proxy, e.g. rerun+http://127.0.0.1:9876/proxy.
        #[arg(long, required_unless_present = "output")]
        connect: Option<String>,
        /// Recording joined by the persistent live preview.
        #[arg(long)]
        recording_id: Option<String>,
        /// Write the blueprint alone to this file instead, to open beside a
        /// recorded .rrd: `rerun capture.rrd blueprint.rbl`.
        #[arg(long, value_name = "FILE.rbl", conflicts_with_all = ["connect", "recording_id"])]
        output: Option<PathBuf>,
    },
    /// Rewrite an RRD as fleet data only, dropping embedded blueprints.
    #[cfg(feature = "rerun")]
    #[command(hide = true)]
    SanitizeRrd {
        #[arg(long)]
        input: PathBuf,
        #[arg(long)]
        output: PathBuf,
        #[arg(long)]
        recording_id: String,
    },
    /// Bridge decimated, nonblocking wxai_teleop UDP telemetry into Rerun.
    /// This process never opens an arm connection or participates in control.
    #[cfg(feature = "rerun")]
    StreamTeleop {
        // Localhost by default: receiving joint state from
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
        #[arg(long, default_value = "")]
        calibration_anchor: String,
        /// Measured robot/world alignment, preferred over the URDF anchor.
        #[arg(long, value_name = "JSON")]
        robot_world: Option<PathBuf>,
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
        /// This host's camera-LAN address; required for a DDS camera and
        /// refused for a USB one.
        #[arg(long)]
        dds_address: Option<std::net::IpAddr>,
    },
    /// Capture a configured RealSense group into synchronized color/depth sets.
    #[cfg(feature = "realsense")]
    CaptureRealsenseAll {
        config: PathBuf,
        /// Capture only these sensor names within the group (repeatable).
        /// Omitted preserves group-wide capture; an invalid name refuses.
        #[arg(long = "sensor")]
        sensors: Vec<String>,
        /// Select one manifested capture group. Required with --zenoh so wrist
        /// and fixed overhead cameras cannot share a bus namespace.
        #[arg(long)]
        group: Option<String>,
        #[arg(long, default_value_t = 10)]
        duration_seconds: u64,
        #[arg(long)]
        output: Option<PathBuf>,
        #[arg(long)]
        calibration: Option<PathBuf>,
        #[arg(long)]
        socket: Option<PathBuf>,
        /// This host's camera-LAN address, to which DDS discovery is bound.
        /// Required when the selection holds a DDS camera (the D555), refused
        /// when it holds none.
        #[arg(long)]
        dds_address: Option<std::net::IpAddr>,
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
        /// Share this Rerun recording with other producers (stream-teleop)
        /// so everything lands in one viewer.
        #[cfg(feature = "rerun")]
        #[arg(long)]
        rerun_recording_id: Option<String>,
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
    #[serde(default)]
    input_cameras: Vec<String>,
    #[serde(default)]
    partial_input: Option<bool>,
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

/// Raised by SIGINT or SIGTERM so the cockpit leaves its loop and flushes
/// its sinks (an `--output` file is finished, not cut off mid-write).
#[cfg(all(feature = "zenoh", feature = "rerun"))]
static STOP_REQUESTED: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);

/// The first signal asks the loop to stop; a second one (a flush that hangs
/// on an unreachable hub) exits at once.
#[cfg(all(feature = "zenoh", feature = "rerun"))]
extern "C" fn request_stop(_: libc::c_int) {
    if STOP_REQUESTED.swap(true, std::sync::atomic::Ordering::AcqRel) {
        // SAFETY: _exit is async-signal-safe.
        unsafe { libc::_exit(130) };
    }
}

#[cfg(all(feature = "zenoh", feature = "rerun"))]
fn install_stop_signals() {
    // SAFETY: the handler only stores an atomic flag.
    unsafe {
        let handler = request_stop as extern "C" fn(libc::c_int) as libc::sighandler_t;
        libc::signal(libc::SIGTERM, handler);
        libc::signal(libc::SIGINT, handler);
    }
}

fn main() -> Result<()> {
    let cli = Cli::parse();
    tracing_subscriber::fmt()
        .with_env_filter(EnvFilter::new(cli.log_filter))
        .with_target(false)
        .init();

    #[cfg(all(feature = "zenoh", feature = "realsense"))]
    let frame_publisher = if cli.zenoh {
        let group = match &cli.command {
            #[cfg(feature = "realsense")]
            Command::CaptureRealsenseAll { group, .. } => group
                .as_deref()
                .context("capture-realsense-all --group is required with --zenoh")?,
            _ => anyhow::bail!("--zenoh requires capture-realsense-all"),
        };
        anyhow::ensure!(!cli.bus_connect.is_empty(), "--bus-connect required");
        Some(tatbot_visiond::frame_bus::FramePublisher::open(
            &cli.bus_connect,
            tatbot_bus::Producer::this_process(
                cli.bus_node
                    .ok_or_else(|| anyhow::anyhow!("--bus-node required"))?,
                "visiond",
            ),
            group,
        )?)
    } else {
        None
    };

    #[cfg(all(
        feature = "zenoh",
        not(feature = "realsense")
    ))]
    anyhow::ensure!(
        !cli.zenoh,
        "camera publication requires a capture backend feature"
    );
    match cli.command {
        #[cfg(all(feature = "zenoh", feature = "rerun"))]
        Command::Subscribe {
            connect,
            recording_id,
            max_fps,
            overhead_color_hz,
            overhead_depth_hz,
            wrist_hz,
            duration_seconds,
            urdf,
            calibration,
            robot_world,
            camera_world,
            output,
            overhead_depth_max_m,
            fiducials,
        } => {
            anyhow::ensure!(!cli.bus_connect.is_empty(), "--bus-connect required");
            anyhow::ensure!(
                calibration.is_none() || robot_world.is_some(),
                "--calibration requires --robot-world for measured placement against the robot"
            );
            install_stop_signals();
            tatbot_visiond::frame_bus::view_bus(
                &cli.bus_connect,
                tatbot_visiond::frame_bus::ViewBusOutput {
                    connect: connect.as_deref(),
                    output: output.as_deref(),
                    recording_id: &recording_id,
                    max_fps,
                    rates: tatbot_visiond::frame_bus::CockpitRates {
                        overhead_color_hz,
                        overhead_depth_hz,
                        wrist_hz,
                    },
                    duration_seconds,
                    camera_world,
                    overhead_depth_max_m,
                    fiducials: fiducials.as_deref(),
                },
                tatbot_visiond::frame_bus::ViewBusScene {
                    urdf: urdf.as_deref(),
                    calibration: calibration.as_deref(),
                    robot_world: robot_world.as_deref(),
                },
                &STOP_REQUESTED,
            )?;
        }
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
                let partial_input = row.partial_input.map(|partial| {
                    partial || row.input_cameras.iter().any(|name| excluded.contains(name))
                });
                let input_cameras = row
                    .input_cameras
                    .into_iter()
                    .filter(|name| !excluded.contains(name))
                    .collect();
                let detections = row
                    .detections
                    .into_values()
                    .flatten()
                    .filter(|detection| !excluded.contains(&detection.camera))
                    .collect();
                let detector_age =
                    std::time::Duration::from_secs_f64(row.detection_latency_ms.max(0.0) / 1000.0);
                let mut estimate = tracker.update_constrained(
                    row.sequence,
                    row.timestamp_ns,
                    row.maximum_skew_ns,
                    detections,
                    row.queue_latency_ms,
                    row.detection_latency_ms,
                    Instant::now() - detector_age,
                    usize::from(partial_input == Some(true)) * 2,
                );
                estimate.input_cameras = input_cameras;
                estimate.partial_input = partial_input;
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
            connect,
            recording_id,
            realtime,
            speed,
            max_fps,
            image_scale,
            jpeg_quality,
        } => {
            if [output.is_some(), spawn, connect.is_some()]
                .iter()
                .filter(|set| **set)
                .count()
                > 1
            {
                anyhow::bail!("choose one of --output, --spawn, --connect");
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
            let calibration_bundle = calibration
                .as_deref()
                .map(CalibrationBundle::load)
                .transpose()?;
            let mut viewer = if spawn {
                RerunViewer::spawn()?
            } else if let Some(url) = connect.as_deref() {
                RerunViewer::connect(url, recording_id.as_deref())?
            } else {
                let output = output.with_context(
                    || "ReplayRerun needs --output PATH, --spawn, or --connect URL",
                )?;
                RerunViewer::save(output)?
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
            if replay_rows.is_empty()
                && teleop.as_ref().is_none_or(|setup| setup.log.ticks.is_empty())
            {
                viewer.log_static_review_timestamp()?;
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
        Command::SendBlueprint {
            connect,
            recording_id,
            output,
        } => {
            if let Some(path) = output {
                tatbot_visiond::rerun_viewer::viewer_blueprint().save(&path)?;
                println!("wrote the fixed blueprint to {}", path.display());
                return Ok(());
            }
            let connect = connect.context("--connect or --output required")?;
            let viewer = RerunViewer::connect_with_blueprint(&connect, recording_id.as_deref())?;
            viewer.finish()?;
            println!(
                "installed fixed Demo / Session / Telemetry / Calibration blueprint at {connect}{}",
                recording_id
                    .as_deref()
                    .map(|id| format!(" for recording {id}"))
                    .unwrap_or_default()
            );
        }
        #[cfg(feature = "rerun")]
        Command::SanitizeRrd {
            input,
            output,
            recording_id,
        } => {
            let stats = tatbot_visiond::rerun_viewer::sanitize_rrd_for_fleet(
                &input,
                &output,
                &recording_id,
            )?;
            println!(
                "sanitized {} -> {} for {recording_id}: kept {}, dropped {} blueprint messages",
                input.display(),
                output.display(),
                stats.kept_messages,
                stats.dropped_blueprint_messages
            );
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
                RerunViewer::connect(&url, Some(recording_id.as_str()))?
            } else {
                RerunViewer::save(output.expect("output precondition checked"))?
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
        #[cfg(feature = "realsense")]
        Command::CaptureRealsense {
            config,
            sensor,
            duration_seconds,
            output,
            calibration,
            dds_address,
        } => {
            let mut config = VisionConfig::load(config)?;
            config
                .select_realsense(None, std::slice::from_ref(&sensor))
                .and_then(|()| config.bind_dds_address(dds_address))
                .map_err(anyhow::Error::msg)?;
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
            sensors,
            group,
            duration_seconds,
            output,
            calibration,
            socket,
            dds_address,
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
            no_record,
        } => {
            let mut config = VisionConfig::load(config)?;
            config
                .select_realsense(group.as_deref(), &sensors)
                .and_then(|()| config.bind_dds_address(dds_address))
                .map_err(anyhow::Error::msg)?;
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
            )?;
            anyhow::ensure!(
                !config.cameras.realsense.is_empty(),
                "capture-realsense-all selected no RealSense cameras{}",
                group
                    .as_deref()
                    .map(|value| format!(" in group {value}"))
                    .unwrap_or_default()
            );
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
            // RealsenseCapture emits one color and one depth event per camera
            // frameset. Retain at most one complete synchronized set here;
            // blocking send then stops workers from requesting another large
            // frameset while recording, socket delivery, or Rerun is behind.
            // FrameSynchronizer owns its separate bounded per-sensor queues.
            let worker_queue_capacity = sensor_names.len().max(1);
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
            let mut publisher = socket.as_ref().map(UnixFramePublisher::bind).transpose()?;
            // Readiness is the bind, so a caller that starts this service
            // gets a connectable socket back rather than a process that has
            // merely been exec'd.
            tatbot_visiond::systemd::notify_socket_ready(socket.as_deref());
            let deadline = (duration_seconds > 0)
                .then(|| Instant::now() + Duration::from_secs(duration_seconds));
            let (sender, receiver) = bounded_capture_event_channel(worker_queue_capacity);
            let mut workers = Vec::new();
            #[cfg(feature = "zenoh")]
            let fresh_requests = frame_publisher
                .as_ref()
                .and_then(tatbot_visiond::frame_bus::FramePublisher::fresh_generation);
            #[cfg(not(feature = "zenoh"))]
            let fresh_requests: Option<std::sync::Arc<std::sync::atomic::AtomicU64>> = None;
            // Bumped by the supervisor on a frame timeout; every device worker
            // answers a bump it has not seen with one hardware reset.
            let reset_requests = std::sync::Arc::new(std::sync::atomic::AtomicU64::new(0));
            #[cfg(feature = "zenoh")]
            let source_socket = cli
                .subscribe_socket
                .clone()
                .or_else(|| (!cli.zenoh).then(|| PathBuf::from("/tmp/tatbot-d405-frames.sock")));
            #[cfg(not(feature = "zenoh"))]
            let source_socket: Option<PathBuf> = None;
            let subscriber_only = source_socket.is_some();
            let mut owner_failure: Option<String> = None;
            #[cfg(feature = "zenoh")]
            anyhow::ensure!(
                !subscriber_only || fresh_requests.is_none(),
                "overhead SDK queue drain requires the camera owner, not a socket subscriber"
            );
            if let Some(path) = source_socket {
                let mut client = tatbot_visiond::UnixFrameClient::connect(path)?;
                client.set_read_timeout(Duration::from_secs(2))?;
                let sender = sender.clone();
                let sensors = config
                    .cameras
                    .realsense
                    .iter()
                    .map(|camera| camera.name.clone())
                    .collect::<Vec<_>>();
                workers.push(thread::spawn(move || {
                    while deadline.is_none_or(|d| Instant::now() < d) {
                        match client.recv() {
                            Ok(set) => {
                                for frame in set.frames {
                                    if sender.send(RealsenseWorkerEvent::Frame(frame)).is_err() {
                                        return;
                                    }
                                }
                            }
                            Err(error) => {
                                let _ = sender.send(RealsenseWorkerEvent::Error {
                                    sensor: "camera-owner".into(),
                                    message: error.to_string(),
                                });
                                break;
                            }
                        }
                    }
                    for sensor in sensors {
                        let health = tatbot_visiond::SensorHealth::new(sensor.clone()).snapshot();
                        let _ = sender.send(RealsenseWorkerEvent::Finished { sensor, health });
                    }
                }));
            } else {
                for camera in config.cameras.realsense.clone() {
                    let sender = sender.clone();
                    let reset_requests = reset_requests.clone();
                    let fresh_requests = fresh_requests.clone();
                    workers.push(thread::spawn(move || {
                        let sensor = camera.name.clone();
                        let mut capture: Option<RealsenseCapture> = None;
                        let mut had_capture = false;
                        let mut reconnects = 0_u64;
                        let mut recent_errors = BoundedStrings::new(16);
                        let mut empty_results = 0_u64;
                        let mut resets_seen = 0_u64;
                        let mut fresh_seen = 0_u64;
                        let mut fresh_flush = None;
                        let mut raw_pending: Option<(u64, Instant, RawDeviceClockRead)> = None;
                        while deadline.is_none_or(|d| Instant::now() < d) {
                            let requested =
                                reset_requests.load(std::sync::atomic::Ordering::Acquire);
                            if requested != resets_seen {
                                // The supervisor saw no event for the safety
                                // timeout. Release the device (pipeline and
                                // lease), put it through a hardware reset and
                                // report what this worker had seen; the next
                                // turn reopens it as after any transport
                                // failure, retrying while the USB device
                                // re-enumerates.
                                resets_seen = requested;
                                if let Some(mut value) = capture.take() {
                                    value.stop();
                                }
                                fresh_seen = 0;
                                fresh_flush = None;
                                raw_pending = None;
                                let last_error = recent_errors.values.back().cloned();
                                let outcome =
                                    tatbot_visiond::realsense_backend::hardware_reset(&camera)
                                        .map_err(|error| error.to_string());
                                if let Err(error) = &outcome {
                                    recent_errors.push(format!("hardware reset: {error}"));
                                }
                                let _ = sender.send(RealsenseWorkerEvent::Reset {
                                    sensor: sensor.clone(),
                                    serial: camera.serial.clone(),
                                    last_error,
                                    empty_results,
                                    outcome,
                                });
                                empty_results = 0;
                                continue;
                            }
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
                            let generation = fresh_requests
                                .as_ref()
                                .map(|requests| requests.load(std::sync::atomic::Ordering::Acquire))
                                .unwrap_or(0);
                            if generation > fresh_seen {
                                match capture
                                    .as_mut()
                                    .expect("RealSense capture initialized")
                                    .drain_ready_frames()
                                {
                                    Ok(discarded) => {
                                        fresh_seen = generation;
                                        fresh_flush = Some((generation, discarded));
                                        raw_pending = Some((
                                            generation,
                                            Instant::now(),
                                            capture.as_ref().expect("RealSense capture initialized").raw_clock_read(),
                                        ));
                                    }
                                    Err(error) => {
                                        recent_errors.push(error.to_string());
                                        let _ = sender.send(RealsenseWorkerEvent::Error {
                                            sensor: sensor.clone(),
                                            message: format!("fresh SDK queue drain: {error}"),
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
                                    empty_results = 0;
                                    let raw_probe = raw_pending.take().map(|(generation, started, before)| {
                                        let after = capture.as_ref().expect("RealSense capture initialized")
                                            .raw_clock_read();
                                        let capture_epoch = frames.first()
                                            .and_then(|frame| frame.metadata.attributes.get("capture_epoch"))
                                            .cloned().unwrap_or_default();
                                        let probe_frames = frames.iter().map(|frame| {
                                            let attrs = &frame.metadata.attributes;
                                            (frame.metadata.sensor_name.clone(), RawClockProbeFrame {
                                                device_frame_number: attrs.get("frame_number").cloned(),
                                                sensor_timestamp_us: attrs.get("sensor_timestamp_us").cloned(),
                                                actual_exposure_us: attrs.get("actual_exposure_us").cloned(),
                                            })
                                        }).collect();
                                        RawDeviceClockProbe {
                                            schema: RAW_DEVICE_CLOCK_SAMPLE_SCHEMA.into(),
                                            sdk_version: RealsenseCapture::raw_clock_sdk_version()
                                                .map(str::to_owned),
                                            generation,
                                            capture_epoch,
                                            frames: probe_frames,
                                            owner_pair_monotonic_elapsed_ns:
                                                u64::try_from(started.elapsed().as_nanos())
                                                    .unwrap_or(u64::MAX),
                                            before,
                                            after,
                                        }
                                    });
                                    let raw_probe = raw_probe.map(|probe| {
                                        serde_json::to_string(&probe)
                                            .expect("raw clock probe serialization")
                                    });
                                    for mut frame in frames {
                                        if let Some((generation, discarded)) = fresh_flush {
                                            frame.metadata.attributes.insert(
                                                "sdk_queue_flush_generation".into(),
                                                generation.to_string(),
                                            );
                                            frame.metadata.attributes.insert(
                                                "sdk_queue_flush_discarded".into(),
                                                discarded.to_string(),
                                            );
                                        }
                                        if let Some(probe) = raw_probe.as_ref() {
                                            frame.metadata.attributes.insert(
                                                "raw_device_clock_probe".into(), probe.clone(),
                                            );
                                        }
                                        if sender.send(RealsenseWorkerEvent::Frame(frame)).is_err()
                                        {
                                            return;
                                        }
                                    }
                                }
                                Ok(None) => {
                                    // A stream whose every frameset is withheld
                                    // must not look like a hung worker: say so
                                    // before the safety timeout can fire.
                                    empty_results = empty_results.saturating_add(1);
                                    if empty_results % 20 == 0 {
                                        let message = format!(
                                            "{empty_results} consecutive RealSense results carried no usable frameset (debug logs name the reason)"
                                        );
                                        recent_errors.push(message.clone());
                                        let _ = sender.send(RealsenseWorkerEvent::Error {
                                            sensor: sensor.clone(),
                                            message,
                                        });
                                    }
                                }
                                Err(error) => {
                                    recent_errors.push(error.to_string());
                                    let _ = sender.send(RealsenseWorkerEvent::Error {
                                        sensor: sensor.clone(),
                                        message: error.to_string(),
                                    });
                                    if let Some(mut value) = capture.take() {
                                        value.stop();
                                    }
                                    fresh_seen = 0;
                                    fresh_flush = None;
                                    raw_pending = None;
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
                        health.recent_errors = recent_errors.into_vec();
                        if let Some(mut value) = capture {
                            value.stop();
                        }
                        let _ = sender.send(RealsenseWorkerEvent::Finished { sensor, health });
                    }));
                }
            }
            drop(sender);

            let expected = config.cameras.realsense.len();
            let mut finished = 0;
            let mut frame_counts = BTreeMap::<String, u64>::new();
            // A capture object is recreated after a transport failure. Keep
            // evidence sequences outside it so reconnects cannot reuse names.
            let mut evidence_sequences = BTreeMap::<String, u64>::new();
            let mut errors = BoundedStrings::new(64);
            let mut health = BTreeMap::new();
            let mut synchronized_sets = 0_u64;
            let mut timeout_policy = FrameTimeoutPolicy::default();
            let mut hardware_resets = 0_u64;
            // A reset request no worker has answered yet. A worker blocked
            // inside a librealsense call (pipeline start or stop, context
            // creation) never sees the request; the exit then says so rather
            // than claiming a reset that was never issued.
            let mut reset_unanswered = false;
            while finished < expected {
                match receiver.recv_timeout(REALSENSE_FRAME_TIMEOUT) {
                    Ok(RealsenseWorkerEvent::Frame(frame)) => {
                        let mut frame = frame;
                        stamp_calibration(&mut frame, calibration.as_ref())?;
                        let sensor = frame.metadata.sensor_name.clone();
                        let sequence = evidence_sequences.entry(sensor.clone()).or_default();
                        frame.metadata.sequence = *sequence;
                        *sequence = sequence.saturating_add(1);
                        *frame_counts.entry(sensor.clone()).or_default() += 1;
                        if let Some(recorder) = recorders.get_mut(&sensor) {
                            recorder.write(&frame)?;
                        } else if !no_record {
                            anyhow::bail!("no recorder for {sensor}");
                        }
                        for set in synchronizer.push(frame).map_err(anyhow::Error::msg)? {
                            // Shared, not copied: the bus publisher keeps the set
                            // while the socket, index and viewer read it.
                            let set = std::sync::Arc::new(set);
                            #[cfg(feature = "zenoh")]
                            if let Some(publisher) = &frame_publisher {
                                publisher.submit_shared(set.clone())?;
                            }
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
                                        std::sync::Arc::unwrap_or_clone(set)
                                    };
                                    sink.submit(set);
                                }
                            }
                            synchronized_sets = synchronized_sets.saturating_add(1);
                        }
                    }
                    Ok(RealsenseWorkerEvent::Error { sensor, message }) => {
                        if sensor == "camera-owner" {
                            owner_failure = Some(message.clone());
                        }
                        errors.push(format!("{sensor}: {message}"));
                    }
                    Ok(RealsenseWorkerEvent::Finished {
                        sensor,
                        health: value,
                    }) => {
                        finished += 1;
                        health.insert(sensor, value);
                    }
                    Ok(RealsenseWorkerEvent::Reset {
                        sensor,
                        serial,
                        last_error,
                        empty_results,
                        outcome,
                    }) => {
                        hardware_resets = hardware_resets.saturating_add(1);
                        reset_unanswered = false;
                        let outcome = match outcome {
                            Ok(()) => "reset issued; reopening the pipeline".to_string(),
                            Err(error) => format!("reset failed: {error}"),
                        };
                        tracing::warn!(
                            sensor = %sensor,
                            serial = %serial,
                            last_error = ?last_error,
                            empty_results,
                            "RealSense hardware reset: {outcome}"
                        );
                        errors.push(format!("{sensor}: hardware reset ({outcome})"));
                        #[cfg(feature = "zenoh")]
                        if let Some(publisher) = &frame_publisher {
                            let record = tatbot_visiond::frame_bus::CaptureHealth {
                                event: "reset".into(),
                                group: group.clone().unwrap_or_default(),
                                sensor,
                                serial,
                                reason: format!(
                                    "no frame, error or finish event for {} s",
                                    REALSENSE_FRAME_TIMEOUT.as_secs()
                                ),
                                last_error,
                                empty_results,
                                outcome,
                            };
                            // The record is the journal of the reset; a bus
                            // put that fails is logged, never a reason to end
                            // an owner that is about to deliver frames again.
                            if let Err(error) = publisher.publish_health(record) {
                                tracing::warn!("reset record not published: {error}");
                            }
                        }
                    }
                    Err(mpsc::RecvTimeoutError::Timeout) => {
                        let recent = errors.values.iter().cloned().collect::<Vec<_>>();
                        // A subscriber owns no device: nothing to reset.
                        let action = if subscriber_only {
                            FrameTimeoutAction::Exit
                        } else {
                            timeout_policy.on_timeout(Instant::now())
                        };
                        match action {
                            FrameTimeoutAction::Reset => {
                                let message = format!(
                                    "no frame, error or finish event for {} s: resetting the RealSense devices once",
                                    REALSENSE_FRAME_TIMEOUT.as_secs()
                                );
                                tracing::warn!(recent_errors = ?recent, "{message}");
                                errors.push(message);
                                reset_unanswered = true;
                                reset_requests.fetch_add(1, std::sync::atomic::Ordering::AcqRel);
                            }
                            FrameTimeoutAction::Exit => {
                                // The suffix states what the owner did, never
                                // what it meant to do: a worker that never
                                // returned from librealsense issued no reset.
                                let suffix = if subscriber_only {
                                    ""
                                } else if reset_unanswered {
                                    " with a hardware reset requested inside the last minute that no worker returned from librealsense to issue"
                                } else {
                                    " after a hardware reset inside the last minute"
                                };
                                anyhow::bail!(
                                    "RealSense capture workers did not finish within the safety timeout: no frame, error or finish event for {} s{}; recent errors {:?}",
                                    REALSENSE_FRAME_TIMEOUT.as_secs(),
                                    suffix,
                                    recent
                                )
                            }
                        }
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
            println!("errors={}", serde_json::to_string(&errors.values)?);
            println!("errors_dropped={}", errors.dropped);
            println!("health={}", serde_json::to_string(&health)?);
            println!("synchronized_sets={synchronized_sets}");
            println!("hardware_resets={hardware_resets}");
            println!("capture_event_queue_capacity={worker_queue_capacity}");
            println!(
                "synchronizer_dropped_unmatched={}",
                synchronizer.dropped_unmatched()
            );
            if let Some(publisher) = publisher.as_ref() {
                println!("transport_clients={}", publisher.client_count());
            }
            println!("manifests={}", serde_json::to_string(&manifests)?);
            anyhow::ensure!(
                !subscriber_only || (owner_failure.is_none() && synchronized_sets > 0),
                "camera subscriber failed: {}",
                owner_failure
                    .as_deref()
                    .unwrap_or("no synchronized frames from owner")
            );
        }
    }
    Ok(())
}

#[cfg(all(feature = "rerun", feature = "realsense"))]
fn open_rerun_viewer(
    output: Option<PathBuf>,
    spawn: bool,
    connect: Option<String>,
    recording_id: Option<&str>,
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
        return Ok(Some(tatbot_visiond::RerunViewer::spawn()?));
    }
    if let Some(url) = connect {
        // Remote live view: JPEG-encode color so five decoded streams fit on
        // the LAN (raw substream BGR alone would be ~100 MB/s).
        let mut viewer = tatbot_visiond::RerunViewer::connect(&url, recording_id)?;
        viewer.set_jpeg_quality(Some(80));
        return Ok(Some(viewer));
    }
    output.map(tatbot_visiond::RerunViewer::save).transpose()
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
}

#[cfg(any(
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
    #[allow(clippy::too_many_arguments)] // Retains the existing detection log schema.
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

#[cfg(feature = "realsense")]
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

#[cfg(feature = "realsense")]
#[derive(Debug)]
#[allow(clippy::large_enum_variant)] // Bounded ingress; image payloads already own heap buffers.
enum RealsenseWorkerEvent {
    Frame(tatbot_visiond::FrameRecord),
    Error {
        sensor: String,
        message: String,
    },
    /// The worker answered a reset request: what it had seen and whether the
    /// device took the reset.
    Reset {
        sensor: String,
        serial: String,
        last_error: Option<String>,
        empty_results: u64,
        outcome: Result<(), String>,
    },
    Finished {
        sensor: String,
        health: tatbot_visiond::HealthSnapshot,
    },
}

/// The RealSense supervisor's safety timeout: no frame, error or finish event
/// from any worker for this long. It is the only clock the reset budget reads.
#[cfg_attr(not(feature = "realsense"), allow(dead_code))]
const REALSENSE_FRAME_TIMEOUT: std::time::Duration = std::time::Duration::from_secs(15);

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[cfg_attr(not(feature = "realsense"), allow(dead_code))]
enum FrameTimeoutAction {
    Reset,
    Exit,
}

/// A capture owner's answer to a frame timeout: reset the silent device once
/// and reopen it; exit for systemd only on a second timeout inside one minute
/// (four safety timeouts). The count is the bound and the existing timeout the
/// clock, so a reset that restored frames for a minute has earned the next
/// timeout its own reset rather than a unit restart, where a crash-looping
/// unit would otherwise wait for a hand-issued hardware reset.
#[derive(Debug, Default)]
#[cfg_attr(not(feature = "realsense"), allow(dead_code))]
struct FrameTimeoutPolicy {
    last_reset: Option<std::time::Instant>,
}

#[cfg_attr(not(feature = "realsense"), allow(dead_code))]
impl FrameTimeoutPolicy {
    const RESET_WINDOW: std::time::Duration = REALSENSE_FRAME_TIMEOUT.saturating_mul(4);

    fn on_timeout(&mut self, now: std::time::Instant) -> FrameTimeoutAction {
        match self.last_reset {
            Some(at) if now.saturating_duration_since(at) < Self::RESET_WINDOW => {
                FrameTimeoutAction::Exit
            }
            _ => {
                self.last_reset = Some(now);
                FrameTimeoutAction::Reset
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{FrameTimeoutAction, FrameTimeoutPolicy, REALSENSE_FRAME_TIMEOUT};
    use std::time::{Duration, Instant};

    #[test]
    fn realsense_frame_timeout_resets_once_then_exits() {
        let start = Instant::now();
        let mut policy = FrameTimeoutPolicy::default();
        assert_eq!(policy.on_timeout(start), FrameTimeoutAction::Reset);
        assert_eq!(
            policy.on_timeout(start + REALSENSE_FRAME_TIMEOUT),
            FrameTimeoutAction::Exit
        );
        // Frames restored for a minute give the next timeout its own reset;
        // the one after it, inside the minute, exits.
        let mut policy = FrameTimeoutPolicy::default();
        assert_eq!(policy.on_timeout(start), FrameTimeoutAction::Reset);
        let later = start + FrameTimeoutPolicy::RESET_WINDOW + Duration::from_secs(1);
        assert_eq!(policy.on_timeout(later), FrameTimeoutAction::Reset);
        assert_eq!(
            policy.on_timeout(later + REALSENSE_FRAME_TIMEOUT),
            FrameTimeoutAction::Exit
        );
    }
}
