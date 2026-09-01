//! Rerun bridge for synchronized Tatbot camera recordings.
//!
//! This is intentionally an adapter, not a second capture pipeline. The
//! capture contract remains authoritative; this module only turns complete
//! synchronized sets into ordered Rerun images, depth images, and metadata.

use std::{
    collections::{BTreeMap, HashMap},
    fs,
    path::{Path, PathBuf},
    sync::{Arc, Condvar, Mutex},
    thread::{self, JoinHandle},
    time::{SystemTime, UNIX_EPOCH},
};

use anyhow::{Context, Result, anyhow};
use clap::ValueEnum;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

use crate::teleop::{LiveTeleopTick, TeleopLog};
use crate::{PixelFormat, RecordedPayload, SynchronizedFrameSet};

/// A teleop flight log plus the mapping onto the URDF's two arms.
#[derive(Debug)]
pub struct TeleopSetup {
    pub log: TeleopLog,
    /// URDF joint-name prefix of the arm the leader drove (e.g. "left").
    pub leader_prefix: String,
    /// URDF joint-name prefix of the follower arm (e.g. "right").
    pub follower_prefix: String,
    /// Rate at which 3D link transforms are logged; scalar time series are
    /// always logged at the full recorded tick rate.
    pub transform_fps: f64,
}

/// Explicit operator workflow. A recording id joins producers; it must never
/// be used as an implicit layout signal because calibration and reconstruction
/// both use shared recordings.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, ValueEnum)]
#[serde(rename_all = "snake_case")]
pub enum RerunLayout {
    Poe,
    Realsense,
    Calibration,
    Surface,
    Teleop,
    Cameras,
    PoeTeleop,
    RealsenseTeleop,
    Full,
    /// Everything live at once: scene, PoE, RealSense, teleop, and the
    /// piezo contact-audio pane (`tatbot live cockpit`).
    Cockpit,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct LayoutSpec {
    scene: bool,
    poe: bool,
    realsense: bool,
    surface: bool,
    teleop: bool,
    notes: bool,
    audio: bool,
}

impl RerunLayout {
    fn spec(self) -> LayoutSpec {
        match self {
            Self::Poe => LayoutSpec {
                scene: false,
                poe: true,
                realsense: false,
                surface: false,
                teleop: false,
                notes: false,
                audio: false,
            },
            Self::Realsense => LayoutSpec {
                scene: false,
                poe: false,
                realsense: true,
                surface: false,
                teleop: false,
                notes: false,
                audio: false,
            },
            Self::Calibration => LayoutSpec {
                scene: true,
                poe: true,
                realsense: false,
                surface: false,
                teleop: false,
                notes: true,
                audio: false,
            },
            Self::Surface => LayoutSpec {
                scene: true,
                poe: true,
                realsense: false,
                surface: true,
                teleop: false,
                notes: false,
                audio: false,
            },
            Self::Teleop => LayoutSpec {
                scene: true,
                poe: false,
                realsense: false,
                surface: false,
                teleop: true,
                notes: false,
                audio: false,
            },
            Self::Cameras => LayoutSpec {
                scene: false,
                poe: true,
                realsense: true,
                surface: false,
                teleop: false,
                notes: false,
                audio: false,
            },
            Self::PoeTeleop => LayoutSpec {
                scene: true,
                poe: true,
                realsense: false,
                surface: false,
                teleop: true,
                notes: false,
                audio: false,
            },
            Self::RealsenseTeleop => LayoutSpec {
                scene: true,
                poe: false,
                realsense: true,
                surface: false,
                teleop: true,
                notes: false,
                audio: false,
            },
            Self::Full => LayoutSpec {
                scene: true,
                poe: true,
                realsense: true,
                surface: false,
                teleop: true,
                notes: false,
                audio: false,
            },
            Self::Cockpit => LayoutSpec {
                scene: true,
                poe: true,
                realsense: true,
                surface: false,
                teleop: true,
                notes: false,
                audio: true,
            },
        }
    }
}

#[derive(Debug, Serialize)]
struct SessionMetadata<'a> {
    schema_version: u32,
    workflow: &'a str,
    layout: Option<RerunLayout>,
    recording_id: Option<&'a str>,
    producer_host: String,
    producer_pid: u32,
    started_unix_ns: u128,
    source_commit: &'static str,
    urdf_path: Option<String>,
    urdf_sha256: Option<String>,
    calibration_id: Option<&'a str>,
}

#[derive(Debug)]
pub struct RerunViewer {
    recording: rerun::RecordingStream,
    layout: Option<RerunLayout>,
    /// When set, color frames are JPEG-encoded at this quality before logging
    /// (depth stays lossless), shrinking recordings ~20-50x. Encoding 5 MP
    /// frames costs real CPU, so callers put the viewer behind `RerunSink`.
    /// Offline replay and remote live capture enable it; local output may keep
    /// raw frames when fidelity matters more than size.
    jpeg_quality: Option<u8>,
}

#[derive(Debug, Clone, Default, Serialize)]
pub struct RerunSinkStats {
    pub submitted: u64,
    pub logged: u64,
    pub dropped_replaced: u64,
    pub errors: u64,
    pub last_error: Option<String>,
}

#[derive(Debug, Default)]
struct LatestFrameState {
    pending: Option<SynchronizedFrameSet>,
    closed: bool,
    stats: RerunSinkStats,
}

impl LatestFrameState {
    fn submit(&mut self, set: SynchronizedFrameSet) {
        self.stats.submitted = self.stats.submitted.saturating_add(1);
        if self.pending.replace(set).is_some() {
            self.stats.dropped_replaced = self.stats.dropped_replaced.saturating_add(1);
        }
    }
}

/// Best-effort visualization sink. Submission only holds a short mutex and
/// replaces stale pending data; serialization, JPEG work, and network I/O run
/// on a dedicated thread and can never backpressure authoritative capture.
pub struct RerunSink {
    state: Arc<(Mutex<LatestFrameState>, Condvar)>,
    worker: Option<JoinHandle<Result<RerunSinkStats>>>,
}

impl RerunSink {
    pub fn new(viewer: RerunViewer) -> Self {
        let state = Arc::new((Mutex::new(LatestFrameState::default()), Condvar::new()));
        let worker_state = Arc::clone(&state);
        let worker = thread::spawn(move || -> Result<RerunSinkStats> {
            loop {
                let set = {
                    let (lock, wake) = &*worker_state;
                    let mut current = lock.lock().unwrap_or_else(|error| error.into_inner());
                    while current.pending.is_none() && !current.closed {
                        current = wake
                            .wait(current)
                            .unwrap_or_else(|error| error.into_inner());
                    }
                    match current.pending.take() {
                        Some(set) => set,
                        None if current.closed => break,
                        None => continue,
                    }
                };
                let result = viewer.log_set(&set);
                let (lock, _) = &*worker_state;
                let mut current = lock.lock().unwrap_or_else(|error| error.into_inner());
                match result {
                    Ok(()) => current.stats.logged = current.stats.logged.saturating_add(1),
                    Err(error) => {
                        current.stats.errors = current.stats.errors.saturating_add(1);
                        current.stats.last_error = Some(error.to_string());
                    }
                }
            }
            viewer.finish()?;
            let (lock, _) = &*worker_state;
            let stats = lock
                .lock()
                .unwrap_or_else(|error| error.into_inner())
                .stats
                .clone();
            Ok(stats)
        });
        Self {
            state,
            worker: Some(worker),
        }
    }

    pub fn submit(&self, set: SynchronizedFrameSet) {
        let (lock, wake) = &*self.state;
        let mut current = lock.lock().unwrap_or_else(|error| error.into_inner());
        current.submit(set);
        wake.notify_one();
    }

    pub fn finish(mut self) -> Result<RerunSinkStats> {
        let (lock, wake) = &*self.state;
        lock.lock()
            .unwrap_or_else(|error| error.into_inner())
            .closed = true;
        wake.notify_one();
        self.worker
            .take()
            .expect("Rerun sink worker exists")
            .join()
            .map_err(|_| anyhow!("Rerun sink worker panicked"))?
    }
}

impl RerunViewer {
    pub fn save(path: impl AsRef<Path>, layout: RerunLayout) -> Result<Self> {
        let path = path.as_ref();
        let recording = rerun::RecordingStreamBuilder::new("tatbot_vision_v2")
            .recording_name("Tatbot 2.0 vision")
            .with_blueprint(camera_blueprint(layout))
            .save(path)
            .with_context(|| format!("creating Rerun recording {}", path.display()))?;
        Ok(Self {
            recording,
            layout: Some(layout),
            jpeg_quality: None,
        })
    }

    pub fn spawn(layout: RerunLayout) -> Result<Self> {
        let recording = rerun::RecordingStreamBuilder::new("tatbot_vision_v2")
            .recording_name("Tatbot 2.0 vision")
            .with_blueprint(camera_blueprint(layout))
            .spawn()
            .context("starting the Rerun Viewer; install the `rerun` executable or use --output")?;
        Ok(Self {
            recording,
            layout: Some(layout),
            jpeg_quality: None,
        })
    }

    /// Stream to a Rerun viewer listening elsewhere (e.g. the operator's
    /// workstation): `url` is a gRPC proxy address such as
    /// `rerun+http://192.0.2.90:9876/proxy`.
    pub fn connect(
        url: &str,
        recording_id: Option<&str>,
        layout: Option<RerunLayout>,
    ) -> Result<Self> {
        let mut builder = rerun::RecordingStreamBuilder::new("tatbot_vision_v2")
            .recording_name("Tatbot 2.0 vision");
        if let Some(layout) = layout {
            builder = builder.with_blueprint(camera_blueprint(layout));
        }
        // A shared recording id lets another producer — the Python surface
        // reconstruction — log into this same recording, so its geometry
        // overlays the live camera stream instead of opening beside it.
        if let Some(id) = recording_id {
            builder = builder.recording_id(id);
        }
        let recording = builder
            .connect_grpc_opts(url)
            .with_context(|| format!("connecting to Rerun viewer at {url}"))?;
        Ok(Self {
            recording,
            layout,
            jpeg_quality: None,
        })
    }

    pub fn log_session_metadata(
        &self,
        workflow: &str,
        recording_id: Option<&str>,
        urdf_path: Option<&Path>,
        calibration_id: Option<&str>,
    ) -> Result<()> {
        let urdf_sha256 = urdf_path.map(sha256_file).transpose()?;
        let metadata = SessionMetadata {
            schema_version: 1,
            workflow,
            layout: self.layout,
            recording_id,
            producer_host: std::env::var("HOSTNAME").unwrap_or_else(|_| "unknown".into()),
            producer_pid: std::process::id(),
            started_unix_ns: SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap_or_default()
                .as_nanos(),
            source_commit: env!("TATBOT_BUILD_GIT_SHA"),
            urdf_path: urdf_path.map(|path| path.display().to_string()),
            urdf_sha256,
            calibration_id,
        };
        self.recording.log_static(
            format!("session/producers/{}", entity_component(workflow)),
            &rerun::TextLog::new(serde_json::to_string_pretty(&metadata)?),
        )?;
        Ok(())
    }

    pub fn log_status(&self, message: impl Into<String>) -> Result<()> {
        self.recording
            .log("session/status", &rerun::TextLog::new(message.into()))?;
        Ok(())
    }

    pub fn prepare_live_teleop(
        &self,
        urdf_path: &Path,
        leader_prefix: &str,
        follower_prefix: &str,
    ) -> Result<LiveTeleopScene> {
        self.recording
            .log_static("/", &rerun::ViewCoordinates::RIGHT_HAND_Z_UP())?;
        let leader_joints = arm_joint_names(leader_prefix);
        let follower_joints = arm_joint_names(follower_prefix);
        let animated_joints = leader_joints
            .iter()
            .chain(&follower_joints)
            .cloned()
            .collect::<Vec<_>>();
        let model = self.log_urdf(urdf_path, &animated_joints)?;
        self.recording.set_time_sequence("teleop_live_tick", -1);
        // A deterministic zero-pose sample makes the model visible while
        // waiting without racing the first packet against cross-host clock
        // skew and creating an unsorted capture timeline.
        self.recording
            .set_timestamp_nanos_since_epoch("capture_time", 0);
        let zero_pose = animated_joints
            .iter()
            .map(|name| (name.clone(), 0.0))
            .collect::<HashMap<_, _>>();
        self.log_joint_transforms(&model, &zero_pose)?;
        Ok(LiveTeleopScene {
            model,
            leader_joints,
            follower_joints,
        })
    }

    pub fn log_live_teleop_tick(
        &self,
        scene: &LiveTeleopScene,
        tick: &LiveTeleopTick,
    ) -> Result<()> {
        if tick.leader_pos.len() != scene.leader_joints.len()
            || tick.follower_pos.len() != scene.follower_joints.len()
        {
            anyhow::bail!(
                "live teleop has {} joints, URDF mapping needs {}",
                tick.leader_pos.len(),
                scene.leader_joints.len()
            );
        }
        self.recording.set_time_sequence(
            "teleop_live_tick",
            i64::try_from(tick.sequence).unwrap_or(i64::MAX),
        );
        self.recording
            .set_timestamp_nanos_since_epoch("capture_time", tick.timestamp_ns);
        let mut joint_values = HashMap::new();
        for (names, positions) in [
            (&scene.leader_joints, &tick.leader_pos),
            (&scene.follower_joints, &tick.follower_pos),
        ] {
            for (name, value) in names.iter().zip(positions) {
                joint_values.insert(name.clone(), *value);
            }
        }
        self.log_joint_transforms(&scene.model, &joint_values)?;
        for (entity, values) in [
            ("teleop/leader/pos", &tick.leader_pos),
            ("teleop/follower/pos", &tick.follower_pos),
            ("teleop/follower/target", &tick.target),
            ("teleop/follower/external_effort", &tick.follower_eff),
        ] {
            self.recording
                .log(entity, &rerun::Scalars::new(values.iter().copied()))?;
        }
        Ok(())
    }

    pub fn set_jpeg_quality(&mut self, quality: Option<u8>) {
        self.jpeg_quality = quality;
    }

    /// Log static reconstruction products and an optional URDF visual model.
    ///
    /// The reconstruction directory is expected to contain the existing
    /// Tatbot dataset layout: `pointclouds/*.ply` and, optionally,
    /// `metadata/vggt_frustums.json`. URDF meshes are expanded into individual
    /// Rerun assets because Rerun does not render URDF XML directly.
    pub fn log_scene(
        &self,
        reconstruction_dir: Option<&Path>,
        urdf_path: Option<&Path>,
        teleop: Option<&TeleopSetup>,
    ) -> Result<()> {
        if reconstruction_dir.is_none() && urdf_path.is_none() && teleop.is_none() {
            return Ok(());
        }

        self.recording
            .log_static("/", &rerun::ViewCoordinates::RIGHT_HAND_Z_UP())?;
        if let Some(directory) = reconstruction_dir {
            self.log_reconstruction(directory)?;
        }
        let animated_joints = teleop
            .map(|setup| {
                let mut joints = arm_joint_names(&setup.leader_prefix);
                joints.extend(arm_joint_names(&setup.follower_prefix));
                joints
            })
            .unwrap_or_default();
        let model = if let Some(path) = urdf_path {
            Some(self.log_urdf(path, &animated_joints)?)
        } else {
            None
        };
        if let Some(setup) = teleop {
            if let Some(model) = &model {
                self.log_teleop_motion(model, setup)?;
            }
            self.log_teleop_series(setup)?;
            // Clear the teleop timelines so later camera rows are not stamped
            // with the last teleop tick.
            self.recording.reset_time();
        }
        Ok(())
    }

    /// Log calibrated camera frustums into the 3D scene: one Pinhole plus a
    /// `world_from_camera` transform per calibrated camera, grouped under a
    /// calibration frame.
    ///
    /// Placement of that frame against the robot, best first:
    /// 1. `robot_world`: a robot_world.json from solve_robot_world.py — the
    ///    MEASURED `world_from_base`; the frame is placed at its inverse.
    /// 2. URDF + anchor link: the link's zero-pose transform (the palette tag
    ///    lives at URDF link `palette_tag8`) — the hand-authored guess.
    /// 3. Neither: the calibration world frame sits at the origin.
    pub fn log_calibration(
        &self,
        bundle: &crate::CalibrationBundle,
        urdf_path: Option<&Path>,
        anchor_link: Option<&str>,
        robot_world: Option<&Path>,
    ) -> Result<()> {
        self.recording
            .log_static("/", &rerun::ViewCoordinates::RIGHT_HAND_Z_UP())?;
        let base = "world/calibration";
        if let Some(path) = robot_world {
            let measured = base_from_world_transform(path)?;
            self.recording
                .log_static(base, &rerun_transform(measured, [1.0, 1.0, 1.0]))?;
        } else if let (Some(path), Some(anchor)) = (urdf_path, anchor_link) {
            let robot = urdf_rs::read_file(path)
                .with_context(|| format!("reading URDF {}", path.display()))?;
            let joints_by_child = robot
                .joints
                .iter()
                .map(|joint| (joint.child.link.clone(), joint.clone()))
                .collect::<HashMap<_, _>>();
            let anchor_pose = resolve_link_pose(
                anchor,
                &joints_by_child,
                &HashMap::new(),
                &mut HashMap::new(),
                &mut Vec::new(),
            )?;
            self.recording
                .log_static(base, &rerun_transform(anchor_pose, [1.0, 1.0, 1.0]))?;
        }
        self.recording.log_static(
            format!("{base}/axes").as_str(),
            &rerun::Arrows3D::from_vectors([
                [0.05_f32, 0.0, 0.0],
                [0.0, 0.05, 0.0],
                [0.0, 0.0, 0.05],
            ])
            .with_colors([
                rerun::Color::from_rgb(240, 80, 80),
                rerun::Color::from_rgb(80, 240, 80),
                rerun::Color::from_rgb(80, 80, 240),
            ]),
        )?;
        for (name, camera) in &bundle.cameras {
            let entity = format!("{base}/cameras/{}", entity_component(name));
            let intrinsics = &camera.intrinsics;
            let pinhole = rerun::Pinhole::from_focal_length_and_resolution(
                [intrinsics.fx as f32, intrinsics.fy as f32],
                [intrinsics.width as f32, intrinsics.height as f32],
            )
            .with_principal_point([intrinsics.cx as f32, intrinsics.cy as f32])
            .with_camera_xyz(rerun::components::ViewCoordinates::RDF)
            .with_image_plane_distance(0.15);
            self.recording.log_static(entity.as_str(), &pinhole)?;
            let rotation = camera.world_from_camera.rotation;
            let pose = RigidTransform {
                rotation: [
                    [rotation[0], rotation[1], rotation[2]],
                    [rotation[3], rotation[4], rotation[5]],
                    [rotation[6], rotation[7], rotation[8]],
                ],
                translation: camera.world_from_camera.translation_m,
            };
            self.recording
                .log_static(entity.as_str(), &rerun_transform(pose, [1.0, 1.0, 1.0]))?;
        }
        Ok(())
    }

    pub fn log_set(&self, set: &SynchronizedFrameSet) -> Result<()> {
        let jpegs = self.encode_set(set)?;
        self.log_set_encoded(set, &jpegs)
    }

    /// Log many sets, JPEG-encoding every color frame of the whole batch in
    /// one parallel pass — a single set has at most 7 frames, which starves a
    /// many-core machine; a batch keeps every core busy.
    pub fn log_sets(&self, sets: &[SynchronizedFrameSet]) -> Result<()> {
        use rayon::prelude::*;
        let encoded = sets
            .par_iter()
            .map(|set| self.encode_set(set))
            .collect::<Result<Vec<_>>>()?;
        for (set, jpegs) in sets.iter().zip(&encoded) {
            self.log_set_encoded(set, jpegs)?;
        }
        Ok(())
    }

    /// JPEG-encode the set's color frames (in parallel) when JPEG output is
    /// enabled; a 5 MP frame costs ~100 ms single-threaded.
    fn encode_set<'a>(&self, set: &'a SynchronizedFrameSet) -> Result<HashMap<&'a str, Vec<u8>>> {
        let Some(quality) = self.jpeg_quality else {
            return Ok(HashMap::new());
        };
        use rayon::prelude::*;
        set.frames
            .iter()
            .filter_map(|(name, frame)| match &frame.payload {
                RecordedPayload::Video {
                    format,
                    width,
                    height,
                    bytes,
                } => Some((name.as_str(), (*format, *width, *height, bytes))),
                _ => None,
            })
            .collect::<Vec<_>>()
            .into_par_iter()
            .map(|(name, (format, width, height, bytes))| {
                encode_jpeg(format, width, height, bytes, quality).map(|jpeg| (name, jpeg))
            })
            .collect::<Result<_>>()
    }

    fn log_set_encoded(
        &self,
        set: &SynchronizedFrameSet,
        jpegs: &HashMap<&str, Vec<u8>>,
    ) -> Result<()> {
        self.recording
            .set_time_sequence("frame", i64::try_from(set.sequence).unwrap_or(i64::MAX));
        if let Ok(timestamp_ns) = i64::try_from(set.timestamp_ns) {
            self.recording
                .set_timestamp_nanos_since_epoch("capture_time", timestamp_ns);
        }
        for (sensor_name, frame) in &set.frames {
            let (_entity, image_entity) = camera_entity(sensor_name);
            match &frame.payload {
                RecordedPayload::Video {
                    format,
                    width,
                    height,
                    bytes,
                } => {
                    if let Some(jpeg) = jpegs.get(sensor_name.as_str()) {
                        self.recording.log(
                            image_entity.as_str(),
                            &rerun::EncodedImage::from_file_contents(jpeg.clone()),
                        )?;
                        let metadata = serde_json::to_string(&frame.metadata)?;
                        self.recording.log(
                            format!("diagnostics/{sensor_name}/metadata"),
                            &rerun::TextLog::new(metadata),
                        )?;
                        continue;
                    }
                    let resolution = [*width, *height];
                    let image = match format {
                        PixelFormat::Bgr8 => {
                            rerun::Image::from_elements(bytes, resolution, rerun::ColorModel::BGR)
                        }
                        PixelFormat::Rgb8 => {
                            rerun::Image::from_elements(bytes, resolution, rerun::ColorModel::RGB)
                        }
                        PixelFormat::Y8 => rerun::Image::from_l8(bytes.clone(), resolution),
                        PixelFormat::Yuyv => rerun::Image::from_pixel_format(
                            resolution,
                            rerun::PixelFormat::YUY2,
                            bytes.clone(),
                        ),
                        other => {
                            return Err(anyhow!(
                                "Rerun image adapter does not support {:?} for {}",
                                other,
                                sensor_name
                            ));
                        }
                    };
                    self.recording.log(image_entity.as_str(), &image)?;
                }
                RecordedPayload::Depth {
                    width,
                    height,
                    bytes,
                } => {
                    let depth = rerun::DepthImage::from_gray16(bytes.clone(), [*width, *height])
                        .with_meter(depth_meter(&frame.metadata.attributes));
                    self.recording.log(image_entity.as_str(), &depth)?;
                }
                RecordedPayload::Encoded { format, .. } => {
                    return Err(anyhow!(
                        "Rerun needs decoded pixels; {} contains encoded {:?}. Capture PoE with --decoded",
                        sensor_name,
                        format
                    ));
                }
            }

            let metadata = serde_json::to_string(&frame.metadata)?;
            self.recording.log(
                format!("diagnostics/{sensor_name}/metadata"),
                &rerun::TextLog::new(metadata),
            )?;
        }
        Ok(())
    }

    pub fn finish(self) -> Result<()> {
        self.recording
            .flush_blocking()
            .context("flushing Rerun recording")?;
        Ok(())
    }

    fn log_reconstruction(&self, directory: &Path) -> Result<()> {
        if !directory.is_dir() {
            anyhow::bail!(
                "reconstruction directory does not exist: {}",
                directory.display()
            );
        }

        let pointcloud_dir = directory.join("pointclouds");
        let mut pointclouds = fs::read_dir(&pointcloud_dir)
            .with_context(|| format!("reading {}", pointcloud_dir.display()))?
            .collect::<std::result::Result<Vec<_>, _>>()?
            .into_iter()
            .map(|entry| entry.path())
            .filter(|path| {
                path.extension()
                    .is_some_and(|extension| extension.eq_ignore_ascii_case("ply"))
            })
            .collect::<Vec<_>>();
        pointclouds.sort();
        for path in pointclouds {
            let points = rerun::Points3D::from_file_path(&path).with_context(|| {
                format!("loading reconstruction point cloud {}", path.display())
            })?;
            let entity = format!(
                "reconstruction/pointclouds/{}",
                entity_component(
                    path.file_stem()
                        .and_then(|name| name.to_str())
                        .unwrap_or("cloud")
                )
            );
            self.recording.log_static(entity, &points)?;
        }

        let frustums_path = directory.join("metadata/vggt_frustums.json");
        if frustums_path.is_file() {
            let frustums: Vec<FrustumRecord> = serde_json::from_slice(
                &fs::read(&frustums_path)
                    .with_context(|| format!("reading {}", frustums_path.display()))?,
            )
            .with_context(|| format!("parsing {}", frustums_path.display()))?;
            for frustum in frustums {
                let path = format!("reconstruction/cameras/{}", entity_component(&frustum.name));
                let intrinsic = frustum
                    .intrinsic_3x3
                    .map(|row| row.map(|value| value as f32));
                let camera_pose = camera_from_world_pose(frustum.extrinsic_3x4);
                self.recording
                    .log_static(path.as_str(), &rerun::Pinhole::new(intrinsic))?;
                self.recording.log_static(
                    path.as_str(),
                    &rerun_transform(camera_pose, [1.0, 1.0, 1.0]),
                )?;
            }
        }
        Ok(())
    }

    /// Log the URDF's visual assets. Links whose pose depends on one of
    /// `animated_joints` get no static transform — their transforms are
    /// logged on the timeline by `log_teleop_motion` — and are returned in
    /// the model for animation. All joints not named are held at zero.
    fn log_urdf(&self, urdf_path: &Path, animated_joints: &[String]) -> Result<UrdfModel> {
        if !urdf_path.is_file() {
            anyhow::bail!("URDF file does not exist: {}", urdf_path.display());
        }
        let robot = urdf_rs::read_file(urdf_path)
            .with_context(|| format!("reading URDF {}", urdf_path.display()))?;
        let joints_by_child = robot
            .joints
            .iter()
            .map(|joint| (joint.child.link.clone(), joint.clone()))
            .collect::<HashMap<_, _>>();
        let no_values = HashMap::new();
        let mut link_poses = HashMap::new();
        let mut skipped = Vec::new();
        let mut animated_visuals = Vec::new();

        for link in &robot.links {
            let link_pose = resolve_link_pose(
                &link.name,
                &joints_by_child,
                &no_values,
                &mut link_poses,
                &mut Vec::new(),
            )?;
            let animated = link_depends_on_joints(&link.name, &joints_by_child, animated_joints);
            for (visual_index, visual) in link.visual.iter().enumerate() {
                let visual_origin = pose_transform(&visual.origin);
                let visual_pose = link_pose.multiply(visual_origin);
                let entity = format!(
                    "robot/links/{}/visual_{visual_index}",
                    entity_component(&link.name)
                );
                let color = visual_color(&robot, visual);
                let mut scale = [1.0_f32, 1.0, 1.0];
                match &visual.geometry {
                    urdf_rs::Geometry::Mesh {
                        filename,
                        scale: mesh_scale,
                    } => {
                        let mesh_path = resolve_mesh_path(urdf_path, filename)?;
                        let asset = rerun::Asset3D::from_file_path(&mesh_path)
                            .with_context(|| format!("loading URDF mesh {}", mesh_path.display()))?
                            .with_albedo_factor(color);
                        scale = mesh_scale
                            .map(|value| [value[0] as f32, value[1] as f32, value[2] as f32])
                            .unwrap_or([1.0, 1.0, 1.0]);
                        if !animated {
                            self.recording.log_static(
                                entity.as_str(),
                                &rerun_transform(visual_pose, scale),
                            )?;
                        }
                        self.recording.log_static(entity.as_str(), &asset)?;
                    }
                    urdf_rs::Geometry::Box { size } => {
                        let mesh = box_mesh(*size, color);
                        if !animated {
                            self.recording.log_static(
                                entity.as_str(),
                                &rerun_transform(visual_pose, [1.0, 1.0, 1.0]),
                            )?;
                        }
                        self.recording.log_static(entity.as_str(), &mesh)?;
                    }
                    urdf_rs::Geometry::Cylinder { radius, length } => {
                        let mesh = cylinder_mesh(*radius, *length, color);
                        if !animated {
                            self.recording.log_static(
                                entity.as_str(),
                                &rerun_transform(visual_pose, [1.0, 1.0, 1.0]),
                            )?;
                        }
                        self.recording.log_static(entity.as_str(), &mesh)?;
                    }
                    urdf_rs::Geometry::Sphere { .. } | urdf_rs::Geometry::Capsule { .. } => {
                        skipped.push(format!(
                            "{} visual {}: unsupported primitive geometry",
                            link.name, visual_index
                        ));
                        continue;
                    }
                }
                if animated {
                    animated_visuals.push(AnimatedVisual {
                        link_name: link.name.clone(),
                        entity,
                        origin: visual_origin,
                        scale,
                    });
                }
            }
        }

        if !skipped.is_empty() {
            self.recording.log_static(
                "diagnostics/urdf/skipped",
                &rerun::TextLog::new(skipped.join("\n")),
            )?;
        }
        Ok(UrdfModel {
            joints_by_child,
            animated_visuals,
        })
    }

    /// Animate the URDF arm links from the teleop flight log: leader joint
    /// positions drive the leader-prefix chain and follower positions the
    /// follower-prefix chain, decimated to `transform_fps`.
    fn log_teleop_motion(&self, model: &UrdfModel, setup: &TeleopSetup) -> Result<()> {
        if model.animated_visuals.is_empty() {
            return Ok(());
        }
        let log = &setup.log;
        let stride = (1.0 / (setup.transform_fps * log.period_s))
            .round()
            .max(1.0) as usize;
        let leader_joints = arm_joint_names(&setup.leader_prefix);
        let follower_joints = arm_joint_names(&setup.follower_prefix);

        for (index, tick) in log.ticks.iter().enumerate().step_by(stride) {
            self.set_teleop_time(log, index, tick.t_wake);
            let mut joint_values = HashMap::new();
            for (names, positions) in [
                (&leader_joints, &tick.leader_pos),
                (&follower_joints, &tick.follower_pos),
            ] {
                for (name, value) in names.iter().zip(positions) {
                    joint_values.insert(name.clone(), *value);
                }
            }
            self.log_joint_transforms(model, &joint_values)?;
        }
        Ok(())
    }

    fn log_joint_transforms(
        &self,
        model: &UrdfModel,
        joint_values: &HashMap<String, f64>,
    ) -> Result<()> {
        let mut link_poses = HashMap::new();
        for visual in &model.animated_visuals {
            let link_pose = resolve_link_pose(
                &visual.link_name,
                &model.joints_by_child,
                joint_values,
                &mut link_poses,
                &mut Vec::new(),
            )?;
            self.recording.log(
                visual.entity.as_str(),
                &rerun_transform(link_pose.multiply(visual.origin), visual.scale),
            )?;
        }
        Ok(())
    }

    /// Log the teleop diagnostics as time series at the full recorded rate:
    /// loop timing under `teleop/timing/`, and per-joint positions, tracking
    /// error, leader velocity, and follower external efforts under `teleop/`.
    /// Log the teleop time series in bulk columnar form: one send per series
    /// instead of one log call per tick — a 400 Hz multi-minute session would
    /// otherwise spend minutes in per-row logging overhead.
    fn log_teleop_series(&self, setup: &TeleopSetup) -> Result<()> {
        let log = &setup.log;
        let ticks = &log.ticks;
        let rows = ticks.len();
        let sequence: Vec<i64> = (0..rows as i64).collect();
        let timestamps_ns: Vec<i64> = ticks
            .iter()
            .map(|tick| log.wall_start_ns.saturating_add((tick.t_wake * 1e9) as i64))
            .collect();
        let indexes = || {
            [
                rerun::TimeColumn::new_sequence("teleop_tick", sequence.clone()),
                rerun::TimeColumn::new_timestamp_nanos_since_epoch(
                    "capture_time",
                    timestamps_ns.clone(),
                ),
            ]
        };

        // Single-value series (loop timing, in ms). The first tick has no
        // predecessor; NaN renders as a gap.
        let mut periods = Vec::with_capacity(rows);
        periods.push(f64::NAN);
        periods.extend(
            ticks
                .windows(2)
                .map(|pair| (pair[1].t_wake - pair[0].t_wake) * 1e3),
        );
        let singles: [(&str, Vec<f64>); 3] = [
            ("teleop/timing/period_ms", periods),
            (
                "teleop/timing/busy_ms",
                ticks.iter().map(|t| (t.t_cmd - t.t_wake) * 1e3).collect(),
            ),
            (
                "teleop/timing/lateness_ms",
                ticks.iter().map(|t| (t.t_wake - t.t_sched) * 1e3).collect(),
            ),
        ];
        for (entity, values) in singles {
            self.recording.send_columns(
                entity,
                indexes(),
                rerun::Scalars::new(values).columns_of_unit_batches()?,
            )?;
        }

        // Multi-value series: one row of num_joints scalars per tick.
        let joints = log.num_joints;
        let flatten = |get: &dyn Fn(&crate::teleop::TeleopTick) -> Vec<f64>| -> Vec<f64> {
            ticks.iter().flat_map(get).collect()
        };
        let multis: [(&str, Vec<f64>); 6] = [
            ("teleop/leader/pos", flatten(&|t| t.leader_pos.clone())),
            ("teleop/follower/pos", flatten(&|t| t.follower_pos.clone())),
            ("teleop/follower/target", flatten(&|t| t.target.clone())),
            (
                "teleop/follower/tracking_error",
                flatten(&|t| {
                    t.target
                        .iter()
                        .zip(&t.follower_pos)
                        .map(|(target, actual)| actual - target)
                        .collect()
                }),
            ),
            ("teleop/leader/vel", flatten(&|t| t.leader_vel.clone())),
            (
                "teleop/follower/external_effort",
                flatten(&|t| t.follower_eff.clone()),
            ),
        ];
        for (entity, values) in multis {
            self.recording.send_columns(
                entity,
                indexes(),
                rerun::Scalars::new(values).columns(std::iter::repeat_n(joints, rows))?,
            )?;
        }
        Ok(())
    }

    fn set_teleop_time(&self, log: &TeleopLog, tick_index: usize, tick_seconds: f64) {
        self.recording
            .set_time_sequence("teleop_tick", i64::try_from(tick_index).unwrap_or(i64::MAX));
        let timestamp_ns = log
            .wall_start_ns
            .saturating_add((tick_seconds * 1e9) as i64);
        self.recording
            .set_timestamp_nanos_since_epoch("capture_time", timestamp_ns);
    }
}

#[derive(Debug)]
struct UrdfModel {
    joints_by_child: HashMap<String, urdf_rs::Joint>,
    animated_visuals: Vec<AnimatedVisual>,
}

pub struct LiveTeleopScene {
    model: UrdfModel,
    leader_joints: Vec<String>,
    follower_joints: Vec<String>,
}

#[derive(Debug)]
struct AnimatedVisual {
    link_name: String,
    entity: String,
    origin: RigidTransform,
    scale: [f32; 3],
}

/// URDF joint names of one WXAI arm chain, in driver joint order (six
/// revolute arm joints, then the actuated gripper carriage; the opposite
/// carriage is a URDF mimic joint and follows automatically).
fn arm_joint_names(prefix: &str) -> Vec<String> {
    let mut names = (0..6)
        .map(|index| format!("{prefix}/joint_{index}"))
        .collect::<Vec<_>>();
    names.push(format!("{prefix}/left_carriage_joint"));
    names
}

fn link_depends_on_joints(
    link_name: &str,
    joints_by_child: &HashMap<String, urdf_rs::Joint>,
    joint_names: &[String],
) -> bool {
    let mut current = link_name;
    while let Some(joint) = joints_by_child.get(current) {
        let driven_by = joint
            .mimic
            .as_ref()
            .map(|mimic| mimic.joint.as_str())
            .unwrap_or(&joint.name);
        if joint_names.iter().any(|name| name == driven_by) {
            return true;
        }
        current = &joint.parent.link;
    }
    false
}

/// JPEG-encode one decoded video frame. BGR and YUYV are converted to RGB
/// first; Y8 encodes as grayscale.
fn encode_jpeg(
    format: PixelFormat,
    width: u32,
    height: u32,
    bytes: &[u8],
    quality: u8,
) -> Result<Vec<u8>> {
    use image::{ExtendedColorType, codecs::jpeg::JpegEncoder};
    let mut out = Vec::new();
    let mut encoder = JpegEncoder::new_with_quality(&mut out, quality);
    match format {
        PixelFormat::Rgb8 => encoder.encode(bytes, width, height, ExtendedColorType::Rgb8)?,
        PixelFormat::Bgr8 => {
            let mut rgb = bytes.to_vec();
            for pixel in rgb.chunks_exact_mut(3) {
                pixel.swap(0, 2);
            }
            encoder.encode(&rgb, width, height, ExtendedColorType::Rgb8)?;
        }
        PixelFormat::Y8 => encoder.encode(bytes, width, height, ExtendedColorType::L8)?,
        PixelFormat::Yuyv => {
            let rgb = yuyv_to_rgb(bytes, width as usize, height as usize)?;
            encoder.encode(&rgb, width, height, ExtendedColorType::Rgb8)?;
        }
        other => anyhow::bail!("JPEG encoding does not support {other:?}"),
    }
    Ok(out)
}

/// BT.601 YUYV (YUY2) to RGB8.
fn yuyv_to_rgb(bytes: &[u8], width: usize, height: usize) -> Result<Vec<u8>> {
    if bytes.len() != width * height * 2 {
        anyhow::bail!(
            "YUYV buffer is {} bytes, expected {} for {width}x{height}",
            bytes.len(),
            width * height * 2
        );
    }
    let mut rgb = Vec::with_capacity(width * height * 3);
    for pair in bytes.chunks_exact(4) {
        let (y0, u, y1, v) = (
            pair[0] as f32,
            pair[1] as f32 - 128.0,
            pair[2] as f32,
            pair[3] as f32 - 128.0,
        );
        for y in [y0, y1] {
            rgb.push((y + 1.402 * v).clamp(0.0, 255.0) as u8);
            rgb.push((y - 0.344 * u - 0.714 * v).clamp(0.0, 255.0) as u8);
            rgb.push((y + 1.772 * u).clamp(0.0, 255.0) as u8);
        }
    }
    Ok(rgb)
}

#[derive(Debug, Deserialize)]
struct FrustumRecord {
    name: String,
    extrinsic_3x4: [[f64; 4]; 3],
    intrinsic_3x3: [[f64; 3]; 3],
}

#[derive(Clone, Copy, Debug)]
struct RigidTransform {
    rotation: [[f64; 3]; 3],
    translation: [f64; 3],
}

impl RigidTransform {
    const IDENTITY: Self = Self {
        rotation: [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
        translation: [0.0, 0.0, 0.0],
    };

    fn multiply(self, other: Self) -> Self {
        let mut rotation = [[0.0; 3]; 3];
        for (row, output_row) in rotation.iter_mut().enumerate() {
            for (column, output) in output_row.iter_mut().enumerate() {
                *output = (0..3)
                    .map(|index| self.rotation[row][index] * other.rotation[index][column])
                    .sum();
            }
        }
        let translation = [
            self.translation[0]
                + self.rotation[0]
                    .iter()
                    .zip(other.translation)
                    .map(|(a, b)| a * b)
                    .sum::<f64>(),
            self.translation[1]
                + self.rotation[1]
                    .iter()
                    .zip(other.translation)
                    .map(|(a, b)| a * b)
                    .sum::<f64>(),
            self.translation[2]
                + self.rotation[2]
                    .iter()
                    .zip(other.translation)
                    .map(|(a, b)| a * b)
                    .sum::<f64>(),
        ];
        Self {
            rotation,
            translation,
        }
    }
}

fn pose_transform(pose: &urdf_rs::Pose) -> RigidTransform {
    let [roll, pitch, yaw] = pose.rpy.0;
    let (sr, cr) = roll.sin_cos();
    let (sp, cp) = pitch.sin_cos();
    let (sy, cy) = yaw.sin_cos();
    RigidTransform {
        rotation: [
            [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
            [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
            [-sp, cp * sr, cp * cr],
        ],
        translation: pose.xyz.0,
    }
}

/// Resolve a link's world pose given joint values by name; joints without an
/// entry sit at zero (their bare origin transform).
fn resolve_link_pose(
    link_name: &str,
    joints_by_child: &HashMap<String, urdf_rs::Joint>,
    joint_values: &HashMap<String, f64>,
    cache: &mut HashMap<String, RigidTransform>,
    visiting: &mut Vec<String>,
) -> Result<RigidTransform> {
    if let Some(pose) = cache.get(link_name) {
        return Ok(*pose);
    }
    if visiting.iter().any(|name| name == link_name) {
        anyhow::bail!("cycle while resolving URDF link {link_name}");
    }
    visiting.push(link_name.to_owned());
    let pose = if let Some(joint) = joints_by_child.get(link_name) {
        let parent = resolve_link_pose(
            &joint.parent.link,
            joints_by_child,
            joint_values,
            cache,
            visiting,
        )?;
        let mut pose = parent.multiply(pose_transform(&joint.origin));
        // A mimic joint tracks its source joint's value scaled and offset.
        let value = if let Some(mimic) = &joint.mimic {
            joint_values.get(&mimic.joint).map(|source| {
                source * mimic.multiplier.unwrap_or(1.0) + mimic.offset.unwrap_or(0.0)
            })
        } else {
            joint_values.get(&joint.name).copied()
        };
        if let Some(value) = value {
            pose = pose.multiply(joint_motion(joint, value));
        }
        pose
    } else {
        RigidTransform::IDENTITY
    };
    visiting.pop();
    cache.insert(link_name.to_owned(), pose);
    Ok(pose)
}

/// The transform contributed by a joint's value: rotation about its axis for
/// revolute/continuous joints, translation along it for prismatic ones.
fn joint_motion(joint: &urdf_rs::Joint, value: f64) -> RigidTransform {
    let [x, y, z] = joint.axis.xyz.0;
    let norm = (x * x + y * y + z * z).sqrt();
    if norm < 1e-12 {
        return RigidTransform::IDENTITY;
    }
    let axis = [x / norm, y / norm, z / norm];
    match joint.joint_type {
        urdf_rs::JointType::Revolute | urdf_rs::JointType::Continuous => RigidTransform {
            rotation: axis_angle_rotation(axis, value),
            translation: [0.0, 0.0, 0.0],
        },
        urdf_rs::JointType::Prismatic => RigidTransform {
            rotation: RigidTransform::IDENTITY.rotation,
            translation: [axis[0] * value, axis[1] * value, axis[2] * value],
        },
        _ => RigidTransform::IDENTITY,
    }
}

/// Rodrigues' rotation formula.
fn axis_angle_rotation(axis: [f64; 3], angle: f64) -> [[f64; 3]; 3] {
    let (sin, cos) = angle.sin_cos();
    let one_minus_cos = 1.0 - cos;
    let [x, y, z] = axis;
    [
        [
            cos + x * x * one_minus_cos,
            x * y * one_minus_cos - z * sin,
            x * z * one_minus_cos + y * sin,
        ],
        [
            y * x * one_minus_cos + z * sin,
            cos + y * y * one_minus_cos,
            y * z * one_minus_cos - x * sin,
        ],
        [
            z * x * one_minus_cos - y * sin,
            z * y * one_minus_cos + x * sin,
            cos + z * z * one_minus_cos,
        ],
    ]
}

fn camera_from_world_pose(extrinsic: [[f64; 4]; 3]) -> RigidTransform {
    let rotation = [
        [extrinsic[0][0], extrinsic[1][0], extrinsic[2][0]],
        [extrinsic[0][1], extrinsic[1][1], extrinsic[2][1]],
        [extrinsic[0][2], extrinsic[1][2], extrinsic[2][2]],
    ];
    let translation = [
        -(rotation[0][0] * extrinsic[0][3]
            + rotation[0][1] * extrinsic[1][3]
            + rotation[0][2] * extrinsic[2][3]),
        -(rotation[1][0] * extrinsic[0][3]
            + rotation[1][1] * extrinsic[1][3]
            + rotation[1][2] * extrinsic[2][3]),
        -(rotation[2][0] * extrinsic[0][3]
            + rotation[2][1] * extrinsic[1][3]
            + rotation[2][2] * extrinsic[2][3]),
    ];
    RigidTransform {
        rotation,
        translation,
    }
}

/// Read solve_robot_world.py's output and return the transform that places
/// the calibration world frame in the base-rooted scene: the inverse of the
/// solved Z = world_from_base (R^T, -R^T t — the matrix is rigid).
fn base_from_world_transform(path: &Path) -> Result<RigidTransform> {
    let value: serde_json::Value = serde_json::from_str(
        &std::fs::read_to_string(path).with_context(|| format!("reading {}", path.display()))?,
    )
    .with_context(|| format!("parsing {}", path.display()))?;
    let rows = value
        .get("world_from_base")
        .and_then(|m| m.as_array())
        .context("robot_world json has no world_from_base matrix")?;
    let mut matrix = [[0.0_f64; 4]; 4];
    for (i, row) in rows.iter().enumerate().take(4) {
        let cols = row
            .as_array()
            .context("world_from_base row is not an array")?;
        for (j, cell) in cols.iter().enumerate().take(4) {
            matrix[i][j] = cell
                .as_f64()
                .context("world_from_base cell is not a number")?;
        }
    }
    let mut rotation = [[0.0_f64; 3]; 3];
    let mut translation = [0.0_f64; 3];
    for i in 0..3 {
        for j in 0..3 {
            rotation[i][j] = matrix[j][i]; // transpose
        }
    }
    for i in 0..3 {
        translation[i] = -(rotation[i][0] * matrix[0][3]
            + rotation[i][1] * matrix[1][3]
            + rotation[i][2] * matrix[2][3]);
    }
    Ok(RigidTransform {
        rotation,
        translation,
    })
}

fn rerun_transform(transform: RigidTransform, scale: [f32; 3]) -> rerun::Transform3D {
    rerun::Transform3D::from_translation_rotation_scale(
        transform.translation.map(|value| value as f32),
        rerun::datatypes::Quaternion::from_wxyz(rotation_to_quaternion(transform.rotation)),
        scale,
    )
}

fn rotation_to_quaternion(rotation: [[f64; 3]; 3]) -> [f32; 4] {
    let trace = rotation[0][0] + rotation[1][1] + rotation[2][2];
    let (w, x, y, z) = if trace > 0.0 {
        let s = (trace + 1.0).sqrt() * 2.0;
        (
            0.25 * s,
            (rotation[2][1] - rotation[1][2]) / s,
            (rotation[0][2] - rotation[2][0]) / s,
            (rotation[1][0] - rotation[0][1]) / s,
        )
    } else if rotation[0][0] > rotation[1][1] && rotation[0][0] > rotation[2][2] {
        let s = (1.0 + rotation[0][0] - rotation[1][1] - rotation[2][2]).sqrt() * 2.0;
        (
            (rotation[2][1] - rotation[1][2]) / s,
            0.25 * s,
            (rotation[0][1] + rotation[1][0]) / s,
            (rotation[0][2] + rotation[2][0]) / s,
        )
    } else if rotation[1][1] > rotation[2][2] {
        let s = (1.0 + rotation[1][1] - rotation[0][0] - rotation[2][2]).sqrt() * 2.0;
        (
            (rotation[0][2] - rotation[2][0]) / s,
            (rotation[0][1] + rotation[1][0]) / s,
            0.25 * s,
            (rotation[1][2] + rotation[2][1]) / s,
        )
    } else {
        let s = (1.0 + rotation[2][2] - rotation[0][0] - rotation[1][1]).sqrt() * 2.0;
        (
            (rotation[1][0] - rotation[0][1]) / s,
            (rotation[0][2] + rotation[2][0]) / s,
            (rotation[1][2] + rotation[2][1]) / s,
            0.25 * s,
        )
    };
    [w as f32, x as f32, y as f32, z as f32]
}

fn resolve_mesh_path(urdf_path: &Path, filename: &str) -> Result<PathBuf> {
    if filename.starts_with("package://") {
        anyhow::bail!("package:// URDF mesh paths are not supported: {filename}");
    }
    let path = filename.strip_prefix("file://").unwrap_or(filename);
    let path = Path::new(path);
    Ok(if path.is_absolute() {
        path.to_owned()
    } else {
        urdf_path
            .parent()
            .unwrap_or_else(|| Path::new("."))
            .join(path)
    })
}

fn entity_component(value: &str) -> String {
    value
        .chars()
        .map(|character| {
            if character.is_ascii_alphanumeric() || matches!(character, '_' | '-' | '.') {
                character
            } else {
                '_'
            }
        })
        .collect()
}

fn visual_color(robot: &urdf_rs::Robot, visual: &urdf_rs::Visual) -> [u8; 4] {
    let rgba = visual
        .material
        .as_ref()
        .and_then(|material| material.color.as_ref())
        .or_else(|| {
            visual.material.as_ref().and_then(|material| {
                robot
                    .materials
                    .iter()
                    .find(|candidate| candidate.name == material.name)
                    .and_then(|candidate| candidate.color.as_ref())
            })
        })
        .map(|color| color.rgba.0);
    rgba.map(|color| color.map(|value| (value.clamp(0.0, 1.0) * 255.0).round() as u8))
        .unwrap_or([180, 180, 180, 255])
}

fn box_mesh(size: urdf_rs::Vec3, color: [u8; 4]) -> rerun::Mesh3D {
    let [x, y, z] = size.0.map(|value| (value / 2.0) as f32);
    let vertices = vec![
        [-x, -y, -z],
        [x, -y, -z],
        [x, y, -z],
        [-x, y, -z],
        [-x, -y, z],
        [x, -y, z],
        [x, y, z],
        [-x, y, z],
    ];
    let triangles = vec![
        [0, 2, 1],
        [0, 3, 2],
        [4, 5, 6],
        [4, 6, 7],
        [0, 1, 5],
        [0, 5, 4],
        [1, 2, 6],
        [1, 6, 5],
        [2, 3, 7],
        [2, 7, 6],
        [3, 0, 4],
        [3, 4, 7],
    ];
    rerun::Mesh3D::new(vertices.clone())
        .with_triangle_indices(triangles)
        .with_vertex_colors(std::iter::repeat_n(color, vertices.len()))
}

fn cylinder_mesh(radius: f64, length: f64, color: [u8; 4]) -> rerun::Mesh3D {
    let segments = 24_u32;
    let half_length = length as f32 / 2.0;
    let radius = radius as f32;
    let mut vertices = Vec::with_capacity((segments as usize) * 2 + 2);
    for z in [-half_length, half_length] {
        for index in 0..segments {
            let angle = std::f32::consts::TAU * index as f32 / segments as f32;
            vertices.push([radius * angle.cos(), radius * angle.sin(), z]);
        }
    }
    let bottom_center = vertices.len() as u32;
    vertices.push([0.0, 0.0, -half_length]);
    let top_center = vertices.len() as u32;
    vertices.push([0.0, 0.0, half_length]);
    let mut triangles = Vec::with_capacity((segments as usize) * 4);
    for index in 0..segments {
        let next = (index + 1) % segments;
        triangles.push([index, next, segments + index]);
        triangles.push([next, segments + next, segments + index]);
        triangles.push([bottom_center, next, index]);
        triangles.push([top_center, segments + index, segments + next]);
    }
    rerun::Mesh3D::new(vertices.clone())
        .with_triangle_indices(triangles)
        .with_vertex_colors(std::iter::repeat_n(color, vertices.len()))
}

fn camera_entity(sensor_name: &str) -> (String, String) {
    if let Some(number) = sensor_name
        .strip_prefix("camera")
        .and_then(|value| value.parse::<u32>().ok())
    {
        let entity = format!("cameras/{number:02}_camera{number}");
        return (entity.clone(), format!("{entity}/image"));
    }

    if let Some((device, stream)) = sensor_name.rsplit_once('_') {
        if let Some(number) = device
            .strip_prefix("realsense")
            .and_then(|value| value.parse::<u32>().ok())
        {
            let entity = format!("cameras/{:02}_realsense{number}", 5 + number);
            return (entity.clone(), format!("{entity}/{stream}"));
        }
    }

    let entity = format!("cameras/99_{sensor_name}");
    (entity.clone(), format!("{entity}/image"))
}

/// Rerun's `meter` is "raw units per metre". The D405 reports Z16 in
/// 0.1 mm (`depth_units_m = 0.0001`, so 10000), most other D4xx in 1 mm; the
/// RealSense backend records the sensor's actual option as
/// `depth_units_m`. Frames without it (pre-2026-08-30 evidence) keep the old
/// 1 mm assumption, which is 10x too far for a D405.
fn depth_meter(attributes: &BTreeMap<String, String>) -> f32 {
    attributes
        .get("depth_units_m")
        .and_then(|value| value.parse::<f32>().ok())
        .filter(|units| units.is_finite() && *units > 0.0)
        .map_or(1000.0, |units| 1.0 / units)
}

fn camera_blueprint(layout: RerunLayout) -> rerun::blueprint::Blueprint {
    use rerun::blueprint::{
        Blueprint, Grid, Horizontal, Spatial2DView, Spatial3DView, TextLogView, TimeSeriesView,
        Vertical,
    };

    fn view(name: &str, entity: &str) -> rerun::blueprint::ContainerLike {
        Spatial2DView::new(name)
            .with_contents([entity.to_owned()])
            .into()
    }

    fn realsense_entity(number: usize, stream: &str) -> String {
        format!("cameras/{:02}_realsense{number}/{stream}", 5 + number)
    }

    let poe_views = (1..=5)
        .map(|number| {
            let entity = format!("cameras/{number:02}_camera{number}/image");
            view(&format!("PoE camera {number}"), &entity)
        })
        .collect::<Vec<_>>();
    let realsense_views = [(1, "color"), (1, "depth"), (2, "color"), (2, "depth")]
        .into_iter()
        .map(|(number, stream)| {
            view(
                &format!("RealSense {number} {stream}"),
                &realsense_entity(number, stream),
            )
        })
        .collect::<Vec<_>>();

    let poe_grid = Grid::new(poe_views)
        .with_name("PoE cameras 1-5")
        .with_grid_columns(5);
    let realsense_grid = Grid::new(realsense_views)
        .with_name("RealSense cameras 1-2")
        .with_grid_columns(4);

    // `/world/**` carries calibrated camera frustums and any reconstructed
    // surface; without it those entities are logged but never displayed.
    let scene_view = Spatial3DView::new("3D reconstruction + robot")
        .with_contents(["/reconstruction/**", "/robot/**", "/world/**"])
        .into();

    let spec = layout.spec();
    // The cockpit runs maximized on a wide display, where a full-width 3D row
    // renders the robot as a thin strip. Flank the 3D view with one column
    // per wrist camera (colour above depth) and weight that row taller than
    // the rest so the robot view comes out roughly square.
    let cockpit_hero = matches!(layout, RerunLayout::Cockpit) && spec.scene && spec.realsense;
    let session = TextLogView::new("Session")
        .with_contents(["/session/**"])
        .into();
    let mut rows: Vec<rerun::blueprint::ContainerLike> = vec![session];
    let mut row_shares: Vec<f32> = vec![0.5];
    if spec.scene {
        if cockpit_hero {
            let realsense_column = |number: usize| -> rerun::blueprint::ContainerLike {
                Vertical::new(vec![
                    view(
                        &format!("RealSense {number} color"),
                        &realsense_entity(number, "color"),
                    ),
                    view(
                        &format!("RealSense {number} depth"),
                        &realsense_entity(number, "depth"),
                    ),
                ])
                .with_name(format!("RealSense {number}"))
                .into()
            };
            rows.push(
                Horizontal::new(vec![scene_view, realsense_column(1), realsense_column(2)])
                    .with_name("Robot + wrist cameras")
                    .with_column_shares([1.5, 1.0, 1.0])
                    .into(),
            );
            row_shares.push(3.0);
        } else {
            rows.push(scene_view);
            row_shares.push(1.0);
        }
    }
    if spec.surface {
        rows.push(
            Grid::new(vec![
                view("Surface height", "surface/height"),
                view("Surface confidence", "surface/confidence"),
                TextLogView::new("Surface status")
                    .with_contents(["/surface/status/**"])
                    .into(),
            ])
            .with_name("Surface reconstruction")
            .with_grid_columns(3)
            .into(),
        );
        row_shares.push(1.0);
    }
    if spec.notes {
        rows.push(
            TextLogView::new("Calibration notes")
                .with_contents(["/calibration/notes/**"])
                .into(),
        );
        row_shares.push(1.0);
    }
    if spec.teleop {
        // The cockpit keeps only the panes an operator watches mid-session;
        // timing and tracking error stay in the recorded layouts.
        let teleop_specs: &[(&str, &str)] = if matches!(layout, RerunLayout::Cockpit) {
            &[
                ("Teleop joints", "/teleop/leader/pos/**"),
                (
                    "Teleop external efforts",
                    "/teleop/follower/external_effort/**",
                ),
            ]
        } else {
            &[
                ("Teleop timing", "/teleop/timing/**"),
                ("Teleop joints", "/teleop/leader/pos/**"),
                (
                    "Teleop tracking error",
                    "/teleop/follower/tracking_error/**",
                ),
                (
                    "Teleop external efforts",
                    "/teleop/follower/external_effort/**",
                ),
            ]
        };
        let teleop_views = teleop_specs
            .iter()
            .map(|&(name, contents)| {
                TimeSeriesView::new(name)
                    .with_contents([contents.to_owned()])
                    .into()
            })
            .collect::<Vec<_>>();
        rows.push(
            Grid::new(teleop_views)
                .with_name("Teleoperation")
                .with_grid_columns(teleop_specs.len() as u32)
                .into(),
        );
        row_shares.push(1.0);
    }
    if spec.poe {
        rows.push(poe_grid.into());
        row_shares.push(1.0);
    }
    if spec.realsense && !cockpit_hero {
        rows.push(realsense_grid.into());
        row_shares.push(1.0);
    }
    if spec.audio {
        // The EE-mounted piezo contact microphone (scripts/audio/live_audio.py).
        // Levels are dBFS time series; the spectrogram is a rolling image the
        // producer rate-limits itself.
        rows.push(
            Grid::new(vec![
                TimeSeriesView::new("EE microphone levels")
                    .with_contents(["/audio/ee/levels/**"])
                    .into(),
                view("EE microphone spectrogram", "audio/ee/spectrogram"),
            ])
            .with_name("Contact audio")
            .with_grid_columns(2)
            .into(),
        );
        row_shares.push(1.0);
    }

    let mut root = Vertical::new(rows);
    if cockpit_hero {
        root = root.with_row_shares(row_shares);
    }
    Blueprint::new(root)
        .with_auto_layout(false)
        .with_auto_views(false)
}

fn sha256_file(path: &Path) -> Result<String> {
    let bytes = fs::read(path).with_context(|| format!("reading {} for hash", path.display()))?;
    Ok(hex::encode(Sha256::digest(bytes)))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn base_from_world_inverts_the_solved_transform() {
        // 90 deg yaw plus a translation; the placement must be the rigid
        // inverse (R^T, -R^T t), not the matrix itself.
        let dir = std::env::temp_dir().join("robot_world_test.json");
        std::fs::write(
            &dir,
            r#"{"world_from_base": [[0.0, -1.0, 0.0, 0.126],
                                    [1.0,  0.0, 0.0, 0.0],
                                    [0.0,  0.0, 1.0, 0.0885],
                                    [0.0,  0.0, 0.0, 1.0]]}"#,
        )
        .unwrap();
        let t = base_from_world_transform(&dir).unwrap();
        assert_eq!(t.rotation[0], [0.0, 1.0, 0.0]);
        assert_eq!(t.rotation[1], [-1.0, 0.0, 0.0]);
        assert!((t.translation[0] - 0.0).abs() < 1e-12);
        assert!((t.translation[1] - 0.126).abs() < 1e-12);
        assert!((t.translation[2] + 0.0885).abs() < 1e-12);
    }

    #[test]
    fn workflow_layouts_do_not_create_inactive_panels() {
        assert_eq!(
            RerunLayout::Poe.spec(),
            LayoutSpec {
                scene: false,
                poe: true,
                realsense: false,
                surface: false,
                teleop: false,
                notes: false,
                audio: false,
            }
        );
        let calibration = RerunLayout::Calibration.spec();
        assert!(calibration.scene && calibration.poe && calibration.notes);
        assert!(!calibration.surface && !calibration.realsense && !calibration.teleop);
        let surface = RerunLayout::Surface.spec();
        assert!(surface.scene && surface.poe && surface.surface);
        assert!(!surface.realsense && !surface.teleop && !surface.notes);
        let teleop = RerunLayout::Teleop.spec();
        assert!(teleop.scene && teleop.teleop);
        assert!(!teleop.poe && !teleop.realsense && !teleop.surface);
        let full = RerunLayout::Full.spec();
        assert!(
            !full.audio,
            "Full is what record_session.sh writes; audio is cockpit-only"
        );
        let cockpit = RerunLayout::Cockpit.spec();
        assert!(
            cockpit.scene && cockpit.poe && cockpit.realsense && cockpit.teleop && cockpit.audio
        );
        assert!(!cockpit.surface && !cockpit.notes);
    }

    #[test]
    fn depth_meter_prefers_recorded_units() {
        let mut attributes = BTreeMap::new();
        assert_eq!(depth_meter(&attributes), 1000.0);
        attributes.insert("depth_units_m".to_string(), "0.0001".to_string());
        assert_eq!(depth_meter(&attributes), 10000.0);
        attributes.insert("depth_units_m".to_string(), "garbage".to_string());
        assert_eq!(depth_meter(&attributes), 1000.0);
        attributes.insert("depth_units_m".to_string(), "0".to_string());
        assert_eq!(depth_meter(&attributes), 1000.0);
    }

    #[test]
    fn file_hash_is_content_addressed() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("asset");
        std::fs::write(&path, b"tatbot").unwrap();
        assert_eq!(
            sha256_file(&path).unwrap(),
            "a28d110deff744496d6d0198c1abc62a619be72048c75e19d0154fb8eb58df64"
        );
    }

    #[test]
    fn latest_frame_sink_replaces_pending_work_instead_of_queueing() {
        let set = |sequence| SynchronizedFrameSet {
            sequence,
            timestamp_basis: "test".into(),
            timestamp_ns: i128::from(sequence),
            maximum_skew_ns: 0,
            frames: Default::default(),
        };
        let mut state = LatestFrameState::default();
        state.submit(set(1));
        state.submit(set(2));
        assert_eq!(state.stats.submitted, 2);
        assert_eq!(state.stats.dropped_replaced, 1);
        assert_eq!(state.pending.as_ref().unwrap().sequence, 2);
    }
}
