//! Rerun bridge for synchronized Tatbot camera recordings.
//!
//! This is intentionally an adapter, not a second capture pipeline. The
//! capture contract remains authoritative; this module only turns complete
//! synchronized sets into ordered Rerun images, depth images, and metadata.

use std::{
    collections::{BTreeMap, HashMap, HashSet},
    fs,
    io::{BufReader, BufWriter},
    path::{Path, PathBuf},
    sync::{Arc, Condvar, Mutex},
    thread::{self, JoinHandle},
    time::{SystemTime, UNIX_EPOCH},
};

use anyhow::{Context, Result, anyhow};
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

/// One application id for the whole project: Rerun keys blueprints by
/// application id, so every producer shares the display-owned blueprint
/// (`tatbot-viewer@` `ExecStartPost`, docs/vision.md).
pub const APP_ID: &str = "tatbot";

/// The shared wall-clock timeline every producer logs on
/// (`tatbot_rerun.TIMELINE`); the blueprint's time panel follows it.
pub const TIMELINE: &str = "capture_time";

pub use crate::STANDING_RECORDING;

/// `HH:MM` of the first `YYYYMMDD[T_-]HHMMSS[Z]` stamp in a recording id, and
/// the byte range it occupies. Every minted id carries one: run ids lead with
/// it, workflow ids follow a prefix.
fn recording_stamp(recording_id: &str) -> Option<(std::ops::Range<usize>, String)> {
    let bytes = recording_id.as_bytes();
    let digit = |i: usize| bytes.get(i).is_some_and(u8::is_ascii_digit);
    (0..bytes.len()).find_map(|start| {
        if !(start..start + 8).all(digit) || !matches!(bytes.get(start + 8), Some(b'T' | b'_' | b'-')) {
            return None;
        }
        let time = start + 9;
        if !(time..time + 6).all(digit) {
            return None;
        }
        let mut end = time + 6;
        if bytes.get(end) == Some(&b'Z') {
            end += 1;
        }
        let glued_before = start > 0 && bytes[start - 1] != b'-';
        let glued_after = bytes.get(end).is_some_and(|b| *b != b'-');
        if glued_before || glued_after {
            return None;
        }
        let hhmm = format!("{}:{}", &recording_id[time..time + 2], &recording_id[time + 2..time + 4]);
        Some((start..end, hhmm))
    })
}

/// The name the viewer's source list shows for a recording, derived from its
/// id alone so every producer joining that id sends the same static value
/// (`tatbot_rerun.recording_name`). A bare run id names nothing here: the
/// session that owns it sends the session's own name. Rerun otherwise shows
/// `<unknown>`, because joining an explicit recording id disables the SDK's
/// default properties.
pub fn recording_name(recording_id: &str) -> Option<String> {
    if recording_id == STANDING_RECORDING {
        return Some("Rig preview".into());
    }
    let (prefix, time) = match recording_stamp(recording_id) {
        Some((range, _)) if range.start == 0 => return None,
        Some((range, hhmm)) => (&recording_id[..range.start - 1], Some(hhmm)),
        None => (recording_id, None),
    };
    let mut words = prefix.replace('-', " ").trim().to_owned();
    if let Some(first) = words.get(..1) {
        words.replace_range(..1, &first.to_ascii_uppercase());
    }
    if words.is_empty() {
        return None;
    }
    Some(match time {
        Some(hhmm) => format!("{words} {hhmm}"),
        None => words,
    })
}

#[derive(Debug, Serialize)]
struct SessionMetadata<'a> {
    schema_version: u32,
    workflow: &'a str,
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
    /// When set, color frames are JPEG-encoded at this quality before logging
    /// (depth stays lossless), shrinking recordings ~20-50x. Encoding 5 MP
    /// frames costs real CPU, so callers put the viewer behind `RerunSink`.
    /// Offline replay and remote live capture enable it; local output may keep
    /// raw frames when fidelity matters more than size.
    jpeg_quality: Option<u8>,
    /// Camera entities whose static display rotation has been logged
    /// (`display_rotation_deg`), so it goes out once, on the first frame.
    oriented: Mutex<HashSet<String>>,
    tracking_registration: Mutex<Option<(String, RigidTransform)>>,
}

/// How a camera's panes are turned in the viewer, degrees about the image
/// normal (Rerun's 2D frame is y-down, so a negative angle reads
/// counter-clockwise on screen). The D405s sit rotated on the wrists, so
/// their panes are turned back upright: RealSense 1 counter-clockwise,
/// RealSense 2 clockwise (operator request 2026-09-02). This is a static
/// in-plane `Transform3D` on the camera entity: the pixels, the recording,
/// and every hover readout stay in sensor coordinates.
fn display_rotation_deg(sensor_name: &str) -> Option<f32> {
    match sensor_name
        .rsplit_once('_')
        .map_or(sensor_name, |(device, _)| device)
    {
        "realsense1" => Some(-90.0),
        "realsense2" => Some(90.0),
        _ => None,
    }
}

#[derive(Debug, Clone, Default, Serialize)]
pub struct RerunSinkStats {
    pub submitted: u64,
    pub logged: u64,
    pub dropped_replaced: u64,
    pub errors: u64,
    pub last_error: Option<String>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RrdSanitizeStats {
    pub kept_messages: u64,
    pub dropped_blueprint_messages: u64,
}

/// Rewrite a standalone RRD into one fleet recording while discarding every
/// blueprint store and activation. The fixed application blueprint is owned by
/// the display server; importing an artifact may only add recording data.
pub fn sanitize_rrd_for_fleet(
    input: impl AsRef<Path>,
    output: impl AsRef<Path>,
    recording_id: &str,
) -> Result<RrdSanitizeStats> {
    anyhow::ensure!(
        !recording_id.is_empty()
            && recording_id
                .bytes()
                .all(|byte| byte.is_ascii_alphanumeric() || b"._-".contains(&byte)),
        "recording id contains unsupported characters"
    );
    let input = input.as_ref();
    let output = output.as_ref();
    anyhow::ensure!(input != output, "RRD input and output must differ");
    let reader = BufReader::new(
        fs::File::open(input).with_context(|| format!("opening RRD {}", input.display()))?,
    );
    let messages = re_log_encoding::DecoderApp::decode_eager(reader)
        .with_context(|| format!("decoding RRD {}", input.display()))?;
    let writer = BufWriter::new(
        fs::File::create(output)
            .with_context(|| format!("creating sanitized RRD {}", output.display()))?,
    );
    let mut encoder = re_log_encoding::Encoder::new_eager(
        re_log_encoding::CrateVersion::LOCAL,
        re_log_encoding::EncodingOptions::PROTOBUF_COMPRESSED,
        writer,
    )?;
    let application_id = re_log_types::ApplicationId::from(APP_ID);
    let rewrite = |store_id: re_log_types::StoreId| {
        store_id
            .with_application_id(application_id.clone())
            .with_recording_id(recording_id.to_owned())
    };
    let mut stats = RrdSanitizeStats {
        kept_messages: 0,
        dropped_blueprint_messages: 0,
    };
    for message in messages {
        let message = match message? {
            rerun::log::LogMsg::BlueprintActivationCommand(_) => {
                stats.dropped_blueprint_messages += 1;
                continue;
            }
            rerun::log::LogMsg::SetStoreInfo(mut info) => {
                if info.info.store_id.is_blueprint() {
                    stats.dropped_blueprint_messages += 1;
                    continue;
                }
                info.info.store_id = rewrite(info.info.store_id);
                rerun::log::LogMsg::SetStoreInfo(info)
            }
            rerun::log::LogMsg::ArrowMsg(store_id, message) => {
                if store_id.is_blueprint() {
                    stats.dropped_blueprint_messages += 1;
                    continue;
                }
                rerun::log::LogMsg::ArrowMsg(rewrite(store_id), message)
            }
        };
        encoder.append(&message)?;
        stats.kept_messages += 1;
    }
    encoder.finish()?;
    encoder.flush_blocking()?;
    Ok(stats)
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
    /// Add a producer to an existing recording without another blueprint or sink.
    pub fn from_recording(recording: rerun::RecordingStream) -> Self {
        Self {
            recording,
            jpeg_quality: None,
            oriented: Mutex::new(HashSet::new()),
            tracking_registration: Mutex::new(None),
        }
    }

    /// Bind the subscriber's world-frame poses to the same measured URDF root.
    pub fn bind_tracking_registration(&self, path: &Path) -> Result<()> {
        let (pose, id) = root_from_world_transform(path)?;
        let id = id
            .filter(|id| !id.is_empty())
            .context("tracking registration has no calibration_id")?;
        *self.tracking_registration.lock().unwrap() = Some((id, pose));
        Ok(())
    }

    /// One entity per tracked target: the right arm's `wrist` keeps its
    /// unsuffixed path, the left arm's `wrist_left` sits beside it.
    pub fn log_tracking(&self, stamp_ns: u64, value: &serde_json::Value) -> Result<()> {
        let (axes_entity, status) = tracking_entities(value)?;
        self.recording
            .set_timestamp_nanos_since_epoch(TIMELINE, i64::try_from(stamp_ns)?);
        self.recording.log(
            status.as_str(),
            &rerun::TextDocument::new(serde_json::to_string_pretty(value)?),
        )?;
        if value["pose"].is_null() {
            self.recording
                .log(axes_entity.as_str(), &rerun::Clear::recursive())?;
            return Ok(());
        }
        let pose = match registered_tracking_pose(
            value,
            self.tracking_registration.lock().unwrap().as_ref(),
        ) {
            Ok(pose) => pose,
            Err(error) => {
                self.recording
                    .log(axes_entity.as_str(), &rerun::Clear::recursive())?;
                self.recording.log(
                    status.as_str(),
                    &rerun::TextDocument::new(format!("Tracking geometry unavailable: {error}")),
                )?;
                return Ok(());
            }
        };
        let matrix = pose;
        let origin = [
            matrix.translation[0] as f32,
            matrix.translation[1] as f32,
            matrix.translation[2] as f32,
        ];
        let axes = (0..3)
            .map(|axis| {
                vec![
                    origin,
                    std::array::from_fn(|row| {
                        origin[row] + matrix.rotation[row][axis] as f32 * 0.03
                    }),
                ]
            })
            .collect::<Vec<_>>();
        self.recording.log(
            axes_entity.as_str(),
            &rerun::LineStrips3D::new(axes).with_colors([[255, 0, 0], [0, 255, 0], [0, 0, 255]]),
        )?;
        Ok(())
    }

    /// One entity per print the stencil observer publishes: the page outline
    /// (the sample's `support.corners_m`, else axes at its pose) at
    /// `world/tracking/target/<pattern_id>` in the URDF root, the sample
    /// beside it. A lost print clears its outline and keeps its status.
    pub fn log_target(&self, stamp_ns: u64, value: &serde_json::Value) -> Result<()> {
        let (outline_entity, status) = target_entities(value)?;
        self.recording
            .set_timestamp_nanos_since_epoch(TIMELINE, i64::try_from(stamp_ns)?);
        self.recording.log(
            status.as_str(),
            &rerun::TextDocument::new(serde_json::to_string_pretty(value)?),
        )?;
        if value["source"] != "measured" || value["world_from_target"].is_null() {
            self.recording
                .log(outline_entity.as_str(), &rerun::Clear::recursive())?;
            return Ok(());
        }
        let registration = self.tracking_registration.lock().unwrap();
        let outline = registered_pose(value, "world_from_target", registration.as_ref())
            .and_then(|pose| target_outline(value, registration.as_ref(), pose));
        drop(registration);
        match outline {
            Ok(strip) => {
                self.recording.log(
                    outline_entity.as_str(),
                    &rerun::LineStrips3D::new([strip]).with_colors([[255, 200, 0]]),
                )?;
            }
            Err(error) => {
                self.recording
                    .log(outline_entity.as_str(), &rerun::Clear::recursive())?;
                self.recording.log(
                    status.as_str(),
                    &rerun::TextDocument::new(format!("Target geometry unavailable: {error}")),
                )?;
            }
        }
        Ok(())
    }

    pub fn save(path: impl AsRef<Path>) -> Result<Self> {
        let path = path.as_ref();
        let recording = rerun::RecordingStreamBuilder::new(APP_ID)
            .recording_name("Tatbot")
            .with_blueprint(viewer_blueprint())
            .save(path)
            .with_context(|| format!("creating Rerun recording {}", path.display()))?;
        Ok(Self {
            recording,
            jpeg_quality: None,
            oriented: Mutex::new(HashSet::new()),
            tracking_registration: Mutex::new(None),
        })
    }

    pub fn spawn() -> Result<Self> {
        let recording = rerun::RecordingStreamBuilder::new(APP_ID)
            .recording_name("Tatbot")
            .with_blueprint(viewer_blueprint())
            .spawn()
            .context("starting the Rerun Viewer; install the `rerun` executable or use --output")?;
        Ok(Self {
            recording,
            jpeg_quality: None,
            oriented: Mutex::new(HashSet::new()),
            tracking_registration: Mutex::new(None),
        })
    }

    /// Join a recording in a Rerun viewer listening elsewhere without
    /// replacing its blueprint. Shared live producers use this entry point.
    pub fn connect(url: &str, recording_id: Option<&str>) -> Result<Self> {
        Self::connect_inner(url, recording_id, false)
    }

    /// Install the one fixed, display-owned blueprint in a Rerun viewer.
    /// Fleet producers must use [`Self::connect`] instead.
    pub fn connect_with_blueprint(url: &str, recording_id: Option<&str>) -> Result<Self> {
        Self::connect_inner(url, recording_id, true)
    }

    fn connect_inner(
        url: &str,
        recording_id: Option<&str>,
        install_blueprint: bool,
    ) -> Result<Self> {
        let mut builder = rerun::RecordingStreamBuilder::new(APP_ID).recording_name("Tatbot");
        if install_blueprint {
            builder = builder.with_blueprint(viewer_blueprint());
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
        // `recording_id` above switched the SDK's default properties off, so
        // the shared name is sent explicitly; a bare run id is the session's.
        if let Some(name) = recording_id.and_then(recording_name) {
            recording.send_recording_name(name)?;
        }
        Ok(Self {
            recording,
            jpeg_quality: None,
            oriented: Mutex::new(HashSet::new()),
            tracking_registration: Mutex::new(None),
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

    /// Give a static review the timeline selected by the shared blueprint.
    /// This is a display event, never a camera exposure or a joint sample.
    pub fn log_static_review_timestamp(&self) -> Result<()> {
        let now_ns = SystemTime::now().duration_since(UNIX_EPOCH)?.as_nanos();
        self.recording
            .set_timestamp_nanos_since_epoch(TIMELINE, i64::try_from(now_ns)?);
        let result = self.log_status("Static calibration review; no live joint or camera samples.");
        self.recording.reset_time();
        result
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
        // A zero-pose sample makes the model visible while waiting for the
        // first packet. It is stamped with this host's clock, not 0: a row at
        // the epoch stretched every time-series view and the time panel from
        // 1970 to today (2026-09-02). The first tick may land a few ms before
        // it across hosts; one unsorted row costs nothing.
        let now_ns = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map_or(0, |d| d.as_nanos() as i64);
        self.recording
            .set_timestamp_nanos_since_epoch(TIMELINE, now_ns);
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
            .set_timestamp_nanos_since_epoch(TIMELINE, tick.timestamp_ns);
        let mut joint_values = HashMap::new();
        let (left, right) = tick.physical_positions();
        for (names, positions) in [
            (&scene.leader_joints, left),
            (&scene.follower_joints, right),
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

    pub fn log_calibration_status(&self, message: impl Into<String>) -> Result<()> {
        self.recording.log_static(
            "world/calibration/status",
            &rerun::TextDocument::new(message.into()),
        )?;
        Ok(())
    }

    /// Log calibrated camera frustums into the 3D scene: one Pinhole plus a
    /// `world_from_camera` transform per calibrated camera, grouped under a
    /// calibration frame.
    ///
    /// Placement of that frame against the robot, best first:
    /// 1. `robot_world`: a robot-world JSON from `tatbot ros calib apply` — the
    ///    MEASURED world-from-URDF-root (legacy key `world_from_base`).
    /// 2. URDF + explicit fixed anchor: diagnostic placement only. The movable
    ///    palette tag is not a robot link and cannot anchor camera calibration.
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
            let (measured, solved_against) = root_from_world_transform(path)?;
            // The bundle's world frame is re-derived by every sweep, so an
            // alignment solved against a different bundle places the frustums
            // somewhere plausible and wrong. Refuse a known mismatch; a solve
            // that names no bundle predates the publish gate and can only be
            // flagged.
            match solved_against.as_deref() {
                Some(id) if id == bundle.bundle_id => {}
                Some(id) => anyhow::bail!(
                    "robot-world alignment {} was solved against calibration {id}, not the bundle \
                     being placed ({}); re-run `tatbot ros calib apply` against the current bundle",
                    path.display(),
                    bundle.bundle_id
                ),
                None => eprintln!(
                    "WARNING: robot-world alignment {} names no calibration_id, so it cannot be \
                     matched to bundle {} — frustum placement against the robot is UNVERIFIED",
                    path.display(),
                    bundle.bundle_id
                ),
            }
            // The solver FK already includes each arm mounting joint.
            // Its inverse places calibration world directly in the URDF root.
            self.recording
                .log_static(base, &rerun_transform(measured, [1.0, 1.0, 1.0]))?;
        } else if let (Some(path), Some(anchor)) = (urdf_path, anchor_link) {
            let robot = urdf_rs::read_file(path)
                .with_context(|| format!("reading URDF {}", path.display()))?;
            if !robot.links.iter().any(|link| link.name == anchor) {
                anyhow::bail!(
                    "calibration anchor {anchor:?} is not a robot link; the palette is movable, supply --robot-world"
                );
            }
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
        let robot = urdf_path.map(urdf_rs::read_file).transpose()?;
        let camera_bodies = robot
            .as_ref()
            .map(fixed_camera_bodies)
            .transpose()?
            .unwrap_or_default();
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
            if let (Some(robot), Some(path), Some(body_from_optical)) =
                (&robot, urdf_path, camera_bodies.get(name))
            {
                let link = robot.links.iter().find(|link| &link.name == name).unwrap();
                let body_pose = pose.multiply(body_from_optical.inverse());
                for (index, visual) in link.visual.iter().enumerate() {
                    self.log_visual(
                        robot,
                        path,
                        visual,
                        &format!(
                            "{base}/camera_models/{}/visual_{index}",
                            entity_component(name)
                        ),
                        body_pose.multiply(pose_transform(&visual.origin)),
                        false,
                    )?;
                }
            }
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
                .set_timestamp_nanos_since_epoch(TIMELINE, timestamp_ns);
        }
        for (sensor_name, frame) in &set.frames {
            let (entity, image_entity) = camera_entity(sensor_name);
            self.orient_camera(sensor_name, &entity)?;
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
                RecordedPayload::Encoded {
                    format: PixelFormat::Jpeg,
                    bytes,
                } => {
                    self.recording.log(
                        image_entity.as_str(),
                        &rerun::EncodedImage::from_file_contents(bytes.clone()),
                    )?;
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

    /// Log the camera's display rotation once (see `display_rotation_deg`).
    fn orient_camera(&self, sensor_name: &str, entity: &str) -> Result<()> {
        let Some(degrees) = display_rotation_deg(sensor_name) else {
            return Ok(());
        };
        let mut oriented = self
            .oriented
            .lock()
            .map_err(|_| anyhow!("camera orientation state poisoned"))?;
        if oriented.insert(entity.to_owned()) {
            self.recording.log_static(
                entity,
                &rerun::Transform3D::from_rotation(rerun::RotationAxisAngle::new(
                    [0.0, 0.0, 1.0],
                    rerun::Angle::from_degrees(degrees),
                )),
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

    /// Log one visual, preserving its URDF origin, scale and material.
    fn log_visual(
        &self,
        robot: &urdf_rs::Robot,
        urdf_path: &Path,
        visual: &urdf_rs::Visual,
        entity: &str,
        visual_pose: RigidTransform,
        animated: bool,
    ) -> Result<Option<[f32; 3]>> {
        let color = visual_color(robot, visual);
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
                    self.recording
                        .log_static(entity, &rerun_transform(visual_pose, scale))?;
                }
                self.recording.log_static(entity, &asset)?;
            }
            urdf_rs::Geometry::Box { size } => {
                let mesh = box_mesh(*size, color);
                if !animated {
                    self.recording
                        .log_static(entity, &rerun_transform(visual_pose, [1.0, 1.0, 1.0]))?;
                }
                self.recording.log_static(entity, &mesh)?;
            }
            urdf_rs::Geometry::Cylinder { radius, length } => {
                let mesh = cylinder_mesh(*radius, *length, color);
                if !animated {
                    self.recording
                        .log_static(entity, &rerun_transform(visual_pose, [1.0, 1.0, 1.0]))?;
                }
                self.recording.log_static(entity, &mesh)?;
            }
            urdf_rs::Geometry::Sphere { .. } | urdf_rs::Geometry::Capsule { .. } => {
                return Ok(None);
            }
        }
        Ok(Some(scale))
    }

    /// Log the URDF's visual assets. Links whose pose depends on one of
    /// `animated_joints` get no static transform — their transforms are
    /// logged on the timeline by `log_teleop_motion` — and are returned in
    /// the model for animation. All joints not named are held at zero.
    fn log_urdf(&self, urdf_path: &Path, animated_joints: &[String]) -> Result<UrdfModel> {
        // A standing recording survives producer restarts. Clear retired links
        // before replacing the whole model, including removed fiducial visuals.
        // Arm-scoped producers call log_urdf_scoped directly and keep their peers.
        self.recording
            .log_static("robot/links", &rerun::Clear::recursive())?;
        self.log_urdf_scoped(urdf_path, animated_joints, None)
    }

    fn log_urdf_scoped(
        &self,
        urdf_path: &Path,
        animated_joints: &[String],
        arm: Option<&str>,
    ) -> Result<UrdfModel> {
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

        let camera_bodies = fixed_camera_bodies(&robot)?;
        for link in &robot.links {
            if camera_bodies.contains_key(&link.name) {
                continue;
            }
            if arm.is_some()
                && link_has_unobserved_joint(&link.name, &joints_by_child, animated_joints)
            {
                continue;
            }
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
                let Some(scale) =
                    self.log_visual(&robot, urdf_path, visual, &entity, visual_pose, animated)?
                else {
                    skipped.push(format!(
                        "{} visual {}: unsupported primitive geometry",
                        link.name, visual_index
                    ));
                    continue;
                };
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
            let (left, right) = if log.right_leader {
                (&tick.follower_pos, &tick.leader_pos)
            } else {
                (&tick.leader_pos, &tick.follower_pos)
            };
            for (names, positions) in [(&leader_joints, left), (&follower_joints, right)] {
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
                rerun::TimeColumn::new_timestamp_nanos_since_epoch(TIMELINE, timestamps_ns.clone()),
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
            .set_timestamp_nanos_since_epoch(TIMELINE, timestamp_ns);
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

// Fixed cameras and supports are useful even when only one arm has feedback.
// Following ancestry also handles camera links attached to an unobserved arm.
fn link_has_unobserved_joint(
    link: &str,
    joints: &HashMap<String, urdf_rs::Joint>,
    observed: &[String],
) -> bool {
    let mut current = link;
    while let Some(joint) = joints.get(current) {
        let driven_by = joint.mimic.as_ref().map_or(&joint.name, |m| &m.joint);
        if joint.joint_type != urdf_rs::JointType::Fixed && !observed.contains(driven_by) {
            return true;
        }
        current = &joint.parent.link;
    }
    false
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

    fn inverse(self) -> Self {
        let rotation = std::array::from_fn(|i| std::array::from_fn(|j| self.rotation[j][i]));
        let translation = std::array::from_fn(|i| {
            -(0..3)
                .map(|j| rotation[i][j] * self.translation[j])
                .sum::<f64>()
        });
        Self {
            rotation,
            translation,
        }
    }

    fn transform_point(self, point: [f64; 3]) -> [f64; 3] {
        std::array::from_fn(|row| {
            self.translation[row]
                + self.rotation[row]
                    .iter()
                    .zip(point)
                    .map(|(a, b)| a * b)
                    .sum::<f64>()
        })
    }

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

/// Read a robot-world JSON and return the transform that places
/// the calibration world frame in the URDF-root scene: the inverse of the
/// solved Z = world_from_root (serialized as `world_from_base`), together
/// with the `calibration_id` the solve was made against (None on a file that
/// predates that field).
fn root_from_world_transform(path: &Path) -> Result<(RigidTransform, Option<String>)> {
    let value: serde_json::Value = serde_json::from_str(
        &std::fs::read_to_string(path).with_context(|| format!("reading {}", path.display()))?,
    )
    .with_context(|| format!("parsing {}", path.display()))?;
    let pose = rigid_from_json(&value["world_from_base"])?;
    let calibration_id = value
        .get("calibration_id")
        .and_then(|id| id.as_str())
        .map(str::to_owned);
    Ok((pose.inverse(), calibration_id))
}

/// The viewer entities a tracking pose lands on, keyed by its target: the
/// axes at `world/tracking/<target>`, the status beside them.
fn tracking_entities(value: &serde_json::Value) -> Result<(String, String)> {
    let target = value["target"]
        .as_str()
        .filter(|t| {
            !t.is_empty()
                && t.len() <= 64
                && t.bytes()
                    .all(|c| c.is_ascii_alphanumeric() || c == b'_' || c == b'-')
        })
        .context("tracking pose names no target")?;
    Ok((
        format!("world/tracking/{target}"),
        format!("world/tracking/status/{target}"),
    ))
}

/// The viewer entities a print's pose lands on: the outline at
/// `world/tracking/target/<pattern_id>`, the sample beside it. A pattern id
/// is `stencil-<sha256>`, longer than a tag target's.
fn target_entities(value: &serde_json::Value) -> Result<(String, String)> {
    let target = value["target_id"]
        .as_str()
        .filter(|t| {
            !t.is_empty()
                && t.len() <= 72
                && t.bytes()
                    .all(|c| c.is_ascii_alphanumeric() || c == b'_' || c == b'-')
        })
        .context("target pose names no print")?;
    Ok((
        format!("world/tracking/target/{target}"),
        format!("world/tracking/target/status/{target}"),
    ))
}

fn registered_tracking_pose(
    value: &serde_json::Value,
    registration: Option<&(String, RigidTransform)>,
) -> Result<RigidTransform> {
    registered_pose(value, "pose", registration)
}

/// A bus pose in the calibration world, placed in the URDF root through the
/// robot-world registration of the same bundle.
fn registered_pose(
    value: &serde_json::Value,
    field: &str,
    registration: Option<&(String, RigidTransform)>,
) -> Result<RigidTransform> {
    let (id, root_from_world) = registration.context("no robot-world registration supplied")?;
    anyhow::ensure!(
        value["calibration_id"].as_str() == Some(id.as_str()),
        "tracking calibration ID differs from robot-world registration"
    );
    Ok(root_from_world.multiply(rigid_from_json(&value[field])?))
}

/// The closed page outline in the URDF root from the observer's world-frame
/// corners; axes at the registered pose when the sample carries none.
fn target_outline(
    value: &serde_json::Value,
    registration: Option<&(String, RigidTransform)>,
    pose: RigidTransform,
) -> Result<Vec<[f32; 3]>> {
    let corners = &value["support"]["corners_m"];
    if corners.is_null() {
        let origin = pose.translation.map(|v| v as f32);
        let mut axes = vec![origin];
        for axis in 0..3 {
            axes.push(std::array::from_fn(|row| {
                origin[row] + pose.rotation[row][axis] as f32 * 0.03
            }));
            axes.push(origin);
        }
        return Ok(axes);
    }
    let corners: Vec<[f64; 3]> =
        serde_json::from_value(corners.clone()).context("support corners are not 3-vectors")?;
    anyhow::ensure!(
        corners.len() >= 3 && corners.iter().flatten().all(|v| v.is_finite()),
        "support corners do not outline a page"
    );
    let (_, root_from_world) = registration.context("no robot-world registration supplied")?;
    let mut strip: Vec<[f32; 3]> = corners
        .iter()
        .map(|corner| root_from_world.transform_point(*corner).map(|v| v as f32))
        .collect();
    strip.push(strip[0]);
    Ok(strip)
}

fn rigid_from_json(value: &serde_json::Value) -> Result<RigidTransform> {
    let matrix: [[f64; 4]; 4] =
        serde_json::from_value(value.clone()).context("expected rigid 4x4 transform")?;
    anyhow::ensure!(
        matrix.iter().flatten().all(|v| v.is_finite()),
        "nonfinite transform"
    );
    anyhow::ensure!(
        (0..4).all(|i| (matrix[3][i] - if i == 3 { 1.0 } else { 0.0 }).abs() < 1e-6),
        "invalid affine row"
    );
    let rotation: [[f64; 3]; 3] = std::array::from_fn(|r| std::array::from_fn(|c| matrix[r][c]));
    for i in 0..3 {
        for j in 0..3 {
            let dot = (0..3).map(|k| rotation[k][i] * rotation[k][j]).sum::<f64>();
            anyhow::ensure!(
                (dot - if i == j { 1.0 } else { 0.0 }).abs() < 1e-6,
                "nonorthonormal rotation"
            );
        }
    }
    let [a, b, c] = rotation;
    let det = a[0] * (b[1] * c[2] - b[2] * c[1]) - a[1] * (b[0] * c[2] - b[2] * c[0])
        + a[2] * (b[0] * c[1] - b[1] * c[0]);
    anyhow::ensure!((det - 1.0).abs() < 1e-6, "rotation reflection");
    Ok(RigidTransform {
        rotation,
        translation: std::array::from_fn(|i| matrix[i][3]),
    })
}

/// Only fixed external cameras are relocated. Wrist cameras remain articulated.
fn fixed_camera_bodies(robot: &urdf_rs::Robot) -> Result<HashMap<String, RigidTransform>> {
    let joints = robot
        .joints
        .iter()
        .map(|j| (j.child.link.clone(), j.clone()))
        .collect::<HashMap<_, _>>();
    let mut result = HashMap::new();
    for link in &robot.links {
        let optical = format!("{}_optical_frame", link.name);
        if link.visual.is_empty()
            || !robot.links.iter().any(|l| l.name == optical)
            || link_has_unobserved_joint(&optical, &joints, &[])
        {
            continue;
        }
        let body = resolve_link_pose(
            &link.name,
            &joints,
            &HashMap::new(),
            &mut HashMap::new(),
            &mut Vec::new(),
        )?;
        let optical = resolve_link_pose(
            &optical,
            &joints,
            &HashMap::new(),
            &mut HashMap::new(),
            &mut Vec::new(),
        )?;
        result.insert(link.name.clone(), body.inverse().multiply(optical));
    }
    Ok(result)
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
        if device == "overhead_depth" {
            let entity = "cameras/08_overhead_depth".to_owned();
            return (entity.clone(), format!("{entity}/{stream}"));
        }
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

#[derive(Debug, PartialEq, Eq)]
struct LayoutContract {
    tabs: [&'static str; 3],
    camera_entities: [&'static str; 13],
    nested_tabs: usize,
}

fn layout_contract() -> LayoutContract {
    const CAMERAS: [&str; 13] = [
        "cameras/01_camera1/image",
        "cameras/02_camera2/image",
        "cameras/03_camera3/image",
        "cameras/04_camera4/image",
        "cameras/05_camera5/image",
        "cameras/06_realsense1/color",
        "cameras/06_realsense1/depth",
        "cameras/07_realsense2/color",
        "cameras/07_realsense2/depth",
        "cameras/08_overhead_depth/color",
        "cameras/08_overhead_depth/depth",
        "observation.images.wrist_upper",
        "observation.images.wrist_lower",
    ];
    LayoutContract {
        tabs: ["Session", "Telemetry", "Calibration"],
        camera_entities: CAMERAS,
        nested_tabs: 0,
    }
}

pub fn viewer_blueprint() -> rerun::blueprint::Blueprint {
    use rerun::blueprint::{
        Blueprint, Grid, Horizontal, Spatial2DView, Spatial3DView, Tabs, TextDocumentView,
        TextLogView, TimeSeriesView, Vertical,
    };

    fn view(
        name: &str,
        entities: impl IntoIterator<Item = impl Into<String>>,
    ) -> rerun::blueprint::ContainerLike {
        Spatial2DView::new(name).with_contents(entities).into()
    }
    let session: rerun::blueprint::ContainerLike = Horizontal::new(vec![
        Spatial3DView::new("Robot + calibrated scene + planned path")
            .with_contents([
                "/reconstruction/**",
                "/robot/**",
                "/world/**",
                "/draw/**",
                "- /draw/preview/**",
                "- /draw/path",
                // The live stencil outlines are under /world/tracking/target.
                // Keep them visible beside the calibrated robot.
                "- /world/tracking/status/**",
            ])
            .into(),
    ])
    .with_name("Session")
    .into();

    let telemetry: rerun::blueprint::ContainerLike = Grid::new(vec![
        view("Room", ["cameras/01_camera1/image"]),
        view("Table 2", ["cameras/02_camera2/image"]),
        view("Table 3", ["cameras/03_camera3/image"]),
        view("Table 4", ["cameras/04_camera4/image"]),
        view("Table 5", ["cameras/05_camera5/image"]),
        view(
            "Wrist 1 RGB",
            [
                "cameras/06_realsense1/color",
                "observation.wrist_upper/**",
                "observation.images.wrist_upper/**",
            ],
        ),
        view(
            "Wrist 1 depth",
            [
                "cameras/06_realsense1/depth",
                "observation.images.wrist_upper_depth/**",
            ],
        ),
        view(
            "Wrist 2 RGB",
            [
                "cameras/07_realsense2/color",
                "observation.wrist_lower/**",
                "observation.images.wrist_lower/**",
            ],
        ),
        view(
            "Wrist 2 depth",
            [
                "cameras/07_realsense2/depth",
                "observation.images.wrist_lower_depth/**",
            ],
        ),
        view("Overhead RGB", ["cameras/08_overhead_depth/color"]),
        view("Overhead depth", ["cameras/08_overhead_depth/depth"]),
        TimeSeriesView::new("Tracking error")
            .with_contents(["/teleop/follower/tracking_error/**"])
            .into(),
    ])
    .with_name("Telemetry")
    .with_grid_columns(5)
    .into();

    let calibration: rerun::blueprint::ContainerLike = Horizontal::new(vec![
        Spatial3DView::new("Calibration geometry (candidate, not active)")
            .with_contents([
                "/world/calibration/**",
                "/reconstruction/**",
                "/robot/**",
                "/calibration/**",
                "- /world/tracking/**",
            ])
            .into(),
        Vertical::new(vec![
            Spatial2DView::new("Stencil tracking")
                .with_origin("/surface/stencil/image")
                .with_contents(["/surface/stencil/image"])
                .into(),
            TextDocumentView::new("Stencil status")
                .with_contents(["/session/presentation/info"])
                .into(),
        ])
        .with_name("Stencil observations")
        .with_row_shares([3.0, 1.0])
        .into(),
        Grid::new(vec![
            view("Overhead RGB", ["cameras/08_overhead_depth/color"]),
            view("Overhead depth", ["cameras/08_overhead_depth/depth"]),
            view("Wrist 1 RGB", ["cameras/06_realsense1/color"]),
            view("Wrist 2 RGB", ["cameras/07_realsense2/color"]),
            view("Room", ["cameras/01_camera1/image"]),
            TextLogView::new("Phase, quality + blockers")
                .with_contents(["/calibration/**", "/session/status/**"])
                .into(),
        ])
        .with_name("Calibration evidence")
        .with_grid_columns(2)
        .into(),
    ])
    .with_name("Calibration")
    .with_column_shares([1.3, 1.3, 1.0])
    .into();

    let contract = layout_contract();
    debug_assert_eq!(contract.tabs, ["Session", "Telemetry", "Calibration"]);
    let root = Tabs::new(vec![session, telemetry, calibration]);
    finish(Blueprint::new(root))
}

/// The shared viewer chrome: the blueprint and selection panels are hidden
/// and the time panel keeps just its transport strip, following the shared
/// wall-clock timeline. The top-bar buttons still toggle them on a window.
fn finish(blueprint: rerun::blueprint::Blueprint) -> rerun::blueprint::Blueprint {
    use rerun::blueprint::{
        BlueprintPanel, SelectionPanel, TimePanel,
        components::{PanelState, PlayState},
    };
    blueprint
        .with_auto_layout(false)
        .with_auto_views(false)
        .with_blueprint_panel(BlueprintPanel::from_state(PanelState::Hidden))
        .with_selection_panel(SelectionPanel::from_state(PanelState::Hidden))
        .with_time_panel(
            TimePanel::new()
                .with_state(PanelState::Collapsed)
                .with_timeline(TIMELINE)
                .with_play_state(PlayState::Following),
        )
}

fn sha256_file(path: &Path) -> Result<String> {
    let bytes = fs::read(path).with_context(|| format!("reading {} for hash", path.display()))?;
    Ok(hex::encode(Sha256::digest(bytes)))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tracking_entities_are_keyed_by_target_and_keep_the_right_wrist_path() {
        let (axes, status) = tracking_entities(&serde_json::json!({"target":"wrist"})).unwrap();
        assert_eq!(axes, "world/tracking/wrist");
        assert_eq!(status, "world/tracking/status/wrist");
        let (axes, _) = tracking_entities(&serde_json::json!({"target":"wrist_left"})).unwrap();
        assert_eq!(axes, "world/tracking/wrist_left");
        for bad in [serde_json::json!({}), serde_json::json!({"target":"../x"})] {
            assert!(tracking_entities(&bad).is_err());
        }
    }
    #[test]
    fn rigid_parser_refuses_truncation_scale_reflection_and_bad_affine_row() {
        for matrix in [
            serde_json::json!([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0]]),
            serde_json::json!([[2, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]),
            serde_json::json!([[-1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]),
            serde_json::json!([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 1, 1]]),
        ] {
            assert!(rigid_from_json(&matrix).is_err());
        }
    }

    #[test]
    fn tracking_registration_rotates_and_translates_and_refuses_other_bundle() {
        let world = rigid_from_json(&serde_json::json!([
            [0, -1, 0, 1],
            [1, 0, 0, 2],
            [0, 0, 1, 3],
            [0, 0, 0, 1]
        ]))
        .unwrap();
        let registration = ("calib".into(), world.inverse());
        let value = serde_json::json!({"calibration_id":"calib", "pose":[[0,-1,0,0.8],[1,0,0,2.1],[0,0,1,3.3],[0,0,0,1]]});
        let pose = registered_tracking_pose(&value, Some(&registration)).unwrap();
        for (a, b) in pose.translation.into_iter().zip([0.1, 0.2, 0.3]) {
            assert!((a - b).abs() < 1e-12);
        }
        assert_eq!(pose.rotation, RigidTransform::IDENTITY.rotation);
        assert!(registered_tracking_pose(&value, None).is_err());
        let mut wrong = value.clone();
        wrong["calibration_id"] = "other".into();
        assert!(registered_tracking_pose(&wrong, Some(&registration)).is_err());
    }

    #[test]
    fn target_outline_follows_registration_and_clears_on_loss() {
        let dir = tempfile::tempdir().unwrap();
        let world = dir.path().join("robot-world.json");
        std::fs::write(
            &world,
            serde_json::json!({
                "world_from_base": [[1, 0, 0, 1], [0, 1, 0, 2], [0, 0, 1, 3], [0, 0, 0, 1]],
                "calibration_id": "calib"
            })
            .to_string(),
        )
        .unwrap();
        let pattern = format!("stencil-{}", "a".repeat(64));
        let measured = serde_json::json!({
            "target_id": pattern, "target_frame": "world", "source": "measured",
            "calibration_id": "calib",
            "world_from_target": [[1, 0, 0, 1.2], [0, 1, 0, 2.1], [0, 0, 1, 3.0], [0, 0, 0, 1]],
            "support": {"corners_m": [[1.1, 2.0, 3.0], [1.3, 2.0, 3.0], [1.3, 2.2, 3.0], [1.1, 2.2, 3.0]],
                        "motion_authority": false}
        });
        let mut lost = measured.clone();
        lost["source"] = "lost".into();
        lost["world_from_target"] = serde_json::Value::Null;
        let mut other_bundle = measured.clone();
        other_bundle["calibration_id"] = "other".into();
        let (rec, storage) = rerun::RecordingStreamBuilder::new(APP_ID).memory().unwrap();
        let viewer = RerunViewer::from_recording(rec);
        viewer.bind_tracking_registration(&world).unwrap();
        viewer.log_target(1, &measured).unwrap();
        viewer.log_target(2, &lost).unwrap();
        viewer.log_target(3, &other_bundle).unwrap();
        assert!(viewer.log_target(4, &serde_json::json!({"source": "measured"})).is_err());
        let outline_entity = format!("/world/tracking/target/{pattern}");
        let status_entity = format!("/world/tracking/target/status/{pattern}");
        let mut strips = Vec::new();
        let mut clears = 0;
        let mut statuses = 0;
        for msg in storage.take() {
            if let rerun::log::LogMsg::ArrowMsg(_, msg) = msg {
                let chunk = rerun::log::Chunk::from_arrow_msg(&msg).unwrap();
                let entity = chunk.entity_path().to_string();
                if entity == status_entity {
                    statuses += chunk.num_rows();
                }
                if entity != outline_entity {
                    continue;
                }
                // The sink batches same-entity logs into one chunk: count rows.
                if chunk
                    .component_descriptors()
                    .any(|d| d.component == rerun::Clear::descriptor_is_recursive().component)
                {
                    clears += chunk.num_rows();
                }
                for strip in chunk.iter_component::<rerun::components::LineStrip3D>(
                    rerun::LineStrips3D::descriptor_strips().component,
                ) {
                    strips.extend(strip.iter().map(|s| s.0.clone()));
                }
            }
        }
        assert_eq!(strips.len(), 1, "one outline for the measured sample");
        let expected = [[0.1, 0.0, 0.0], [0.3, 0.0, 0.0], [0.3, 0.2, 0.0], [0.1, 0.2, 0.0], [0.1, 0.0, 0.0]];
        assert_eq!(strips[0].len(), expected.len());
        for (point, want) in strips[0].iter().zip(expected) {
            for (a, b) in point.0.iter().zip(want) {
                assert!((f64::from(*a) - b).abs() < 1e-6, "{point:?} vs {want:?}");
            }
        }
        assert_eq!(clears, 2, "the lost sample and the other bundle clear the outline");
        assert_eq!(statuses, 4, "every sample logs its status; the other bundle says why");
    }

    #[test]
    fn reloading_a_whole_urdf_clears_retired_visuals_in_the_standing_recording() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("rig.urdf");
        let (rec, storage) = rerun::RecordingStreamBuilder::new(APP_ID).memory().unwrap();
        let viewer = RerunViewer::from_recording(rec);
        std::fs::write(&path, r#"<robot name="rig"><link name="retired_tag"><visual><geometry><box size="0.1 0.1 0.001"/></geometry></visual></link></robot>"#).unwrap();
        viewer.log_urdf(&path, &[]).unwrap();
        storage.take();
        std::fs::write(&path, r#"<robot name="rig"><link name="root"/></robot>"#).unwrap();
        viewer.log_urdf(&path, &[]).unwrap();
        let mut cleared = false;
        for msg in storage.take() {
            if let rerun::log::LogMsg::ArrowMsg(_, msg) = msg {
                let chunk = rerun::log::Chunk::from_arrow_msg(&msg).unwrap();
                assert!(!chunk.entity_path().to_string().contains("retired_tag"));
                if chunk.entity_path().to_string() == "/robot/links" {
                    cleared |= chunk.component_descriptors().any(|d| {
                        d.component == rerun::Clear::descriptor_is_recursive().component
                    });
                }
            }
        }
        assert!(cleared, "retired visuals must be cleared when reusing the recording");
    }

    #[test]
    fn calibrated_camera_meshes_have_one_owner_and_ignore_nominal_mounts() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("rig.urdf");
        std::fs::write(&path, r#"<robot name="rig">
            <link name="root"/>
            <link name="camera1"><visual><geometry><box size="0.1 0.1 0.1"/></geometry></visual></link>
            <link name="camera1_optical_frame"/>
            <joint name="nominal" type="fixed"><parent link="root"/><child link="camera1"/><origin xyz="9 8 7" rpy="0.3 0.2 0.1"/></joint>
            <joint name="lens" type="fixed"><parent link="camera1"/><child link="camera1_optical_frame"/><origin xyz="0.01 0 0" rpy="0 0 1.5707963267948966"/></joint>
        </robot>"#).unwrap();
        let bundle: crate::CalibrationBundle = serde_json::from_value(serde_json::json!({
            "schema_version":1,"bundle_id":"test","world_frame":"camera_world","cameras":{"camera1":{
                "sensor_name":"camera1", "profile":{"stream":"main","width":640,"height":480,"fps_num":15,"fps_den":1,"format":"rgb8"},
                "intrinsics":{"width":640,"height":480,"fx":500,"fy":500,"cx":320,"cy":240},
                "distortion":{"model":"none","coefficients":[]},
                "world_from_camera":{"rotation":[0,-1,0,1,0,0,0,0,1],"translation_m":[1,2,3]}
            }}})).unwrap();
        for calibration_first in [true, false] {
            let (rec, storage) = rerun::RecordingStreamBuilder::new(APP_ID).memory().unwrap();
            let arm = RerunViewer::from_recording(rec.clone());
            let camera = RerunViewer::from_recording(rec);
            if calibration_first {
                camera
                    .log_calibration(&bundle, Some(&path), None, None)
                    .unwrap();
            }
            arm.log_urdf(&path, &[]).unwrap();
            if !calibration_first {
                camera
                    .log_calibration(&bundle, Some(&path), None, None)
                    .unwrap();
            }
            let mut model_positions = Vec::new();
            for msg in storage.take() {
                if let rerun::log::LogMsg::ArrowMsg(_, msg) = msg {
                    let chunk = rerun::log::Chunk::from_arrow_msg(&msg).unwrap();
                    let entity = chunk.entity_path().to_string();
                    assert!(!entity.starts_with("/robot/links/camera1/"));
                    if entity == "/world/calibration/camera_models/camera1/visual_0" {
                        for positions in chunk.iter_component::<rerun::components::Translation3D>(
                            rerun::Transform3D::descriptor_translation().component,
                        ) {
                            model_positions.extend(positions.iter().map(|p| p.0.0));
                        }
                    }
                }
            }
            assert_eq!(model_positions.len(), 1);
            for (a, b) in model_positions[0].into_iter().zip([0.99, 2.0, 3.0]) {
                assert!((a - b).abs() < 1e-6);
            }
        }
    }

    #[test]
    fn observed_arm_scope_keeps_fixed_rig_and_omits_unknown_motion() {
        let robot = urdf_rs::read_from_string(r#"<robot name="rig">
            <link name="root"/><link name="camera"/><link name="arm"/><link name="wrist"/>
            <joint name="mount" type="fixed"><parent link="root"/><child link="camera"/></joint>
            <joint name="drive" type="continuous"><parent link="root"/><child link="arm"/><axis xyz="0 0 1"/></joint>
            <joint name="wrist_mount" type="fixed"><parent link="arm"/><child link="wrist"/></joint>
        </robot>"#).unwrap();
        let joints = robot
            .joints
            .into_iter()
            .map(|j| (j.child.link.clone(), j))
            .collect();
        assert!(!link_has_unobserved_joint("camera", &joints, &[]));
        assert!(link_has_unobserved_joint("wrist", &joints, &[]));
        assert!(!link_has_unobserved_joint(
            "wrist",
            &joints,
            &["drive".into()]
        ));
    }

    #[test]
    fn root_from_world_inverts_the_solved_transform() {
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
        let (t, calibration_id) = root_from_world_transform(&dir).unwrap();
        assert_eq!(t.rotation[0], [0.0, 1.0, 0.0]);
        assert_eq!(t.rotation[1], [-1.0, 0.0, 0.0]);
        assert!((t.translation[0] - 0.0).abs() < 1e-12);
        assert!((t.translation[1] - 0.126).abs() < 1e-12);
        assert!((t.translation[2] + 0.0885).abs() < 1e-12);
        // A solve published before the gate names no bundle.
        assert_eq!(calibration_id, None);
    }

    #[test]
    fn root_from_world_reports_the_solved_calibration_id() {
        let dir = std::env::temp_dir().join("robot_world_calibration_id_test.json");
        std::fs::write(
            &dir,
            r#"{"calibration_id": "abc123",
                "world_from_base": [[1.0, 0.0, 0.0, 0.0],
                                    [0.0, 1.0, 0.0, 0.0],
                                    [0.0, 0.0, 1.0, 0.0],
                                    [0.0, 0.0, 0.0, 1.0]]}"#,
        )
        .unwrap();
        let (_, calibration_id) = root_from_world_transform(&dir).unwrap();
        assert_eq!(calibration_id.as_deref(), Some("abc123"));
    }

    #[test]
    fn recording_names_follow_the_id_and_leave_run_ids_to_their_session() {
        assert_eq!(recording_name("live-cameras").as_deref(), Some("Rig preview"));
        assert_eq!(recording_name("cockpit-20260917T201500Z").as_deref(), Some("Cockpit 20:15"));
        assert_eq!(recording_name("sweep-20260917_150212").as_deref(), Some("Sweep 15:02"));
        assert_eq!(recording_name("record-20260917-150212").as_deref(), Some("Record 15:02"));
        assert_eq!(
            recording_name("draw-20260917T174304Z-arm-7e3c").as_deref(),
            Some("Draw 17:43")
        );
        assert_eq!(
            recording_name("replay-flight_12-20260917T174304Z").as_deref(),
            Some("Replay flight_12 17:43")
        );
        assert_eq!(
            recording_name("draw-shadow-squiggle_v3").as_deref(),
            Some("Draw shadow squiggle_v3")
        );
        assert_eq!(recording_name("20260917T174304Z-arm-7e3c"), None);
        // Digits inside a word are not a stamp; a stamp glued to text is not one either.
        assert_eq!(recording_name("part-1234567890123456").as_deref(), Some("Part 1234567890123456"));
        assert_eq!(recording_name("x20260917T174304Z").as_deref(), Some("X20260917T174304Z"));
        assert_eq!(recording_name(""), None);
    }

    #[test]
    fn shared_blueprint_builds() {
        let _ = viewer_blueprint();
    }

    #[test]
    fn static_review_has_a_timeline_without_fabricating_sensor_samples() {
        let (rec, storage) = rerun::RecordingStreamBuilder::new(APP_ID).memory().unwrap();
        let viewer = RerunViewer::from_recording(rec);
        viewer.log_static_review_timestamp().unwrap();
        let mut status_rows = 0;
        for msg in storage.take() {
            if let rerun::log::LogMsg::ArrowMsg(_, msg) = msg {
                let chunk = rerun::log::Chunk::from_arrow_msg(&msg).unwrap();
                if chunk.entity_path().to_string() == "/__properties" {
                    continue; // SDK recording properties are not sensor samples.
                }
                assert_eq!(chunk.entity_path().to_string(), "/session/status");
                assert!(
                    chunk
                        .timelines()
                        .keys()
                        .any(|name| name.as_str() == TIMELINE)
                );
                status_rows += chunk.num_rows();
            }
        }
        assert_eq!(status_rows, 1);
    }

    #[test]
    fn fixed_blueprint_has_three_flat_tabs_and_all_camera_sources() {
        let contract = layout_contract();
        assert_eq!(contract.tabs, ["Session", "Telemetry", "Calibration"]);
        assert_eq!(contract.nested_tabs, 0);
        assert_eq!(contract.camera_entities.len(), 13);
        assert!(
            contract
                .camera_entities
                .contains(&"cameras/08_overhead_depth/depth")
        );
        assert!(
            contract
                .camera_entities
                .contains(&"observation.images.wrist_upper")
        );
    }

    #[test]
    fn fleet_rrd_sanitizer_keeps_data_and_drops_embedded_blueprint() {
        let directory = tempfile::tempdir().unwrap();
        let source = directory.path().join("standalone.rrd");
        let output = directory.path().join("fleet.rrd");
        let viewer = RerunViewer::save(&source).unwrap();
        viewer
            .recording
            .log("probe/value", &rerun::Scalars::new([7.0]))
            .unwrap();
        viewer.finish().unwrap();

        let stats = sanitize_rrd_for_fleet(&source, &output, "imported-review").unwrap();
        assert!(stats.kept_messages > 0);
        assert!(stats.dropped_blueprint_messages > 0);

        let reader = BufReader::new(fs::File::open(output).unwrap());
        let messages = re_log_encoding::DecoderApp::decode_eager(reader).unwrap();
        let mut found_probe = false;
        for message in messages {
            match message.unwrap() {
                rerun::log::LogMsg::BlueprintActivationCommand(_) => {
                    panic!("fleet RRD retained a blueprint activation")
                }
                rerun::log::LogMsg::SetStoreInfo(info) => {
                    assert!(!info.info.store_id.is_blueprint());
                    assert_eq!(info.info.store_id.application_id().as_str(), APP_ID);
                    assert_eq!(
                        info.info.store_id.recording_id().as_str(),
                        "imported-review"
                    );
                }
                rerun::log::LogMsg::ArrowMsg(store_id, message) => {
                    assert!(!store_id.is_blueprint());
                    assert_eq!(store_id.recording_id().as_str(), "imported-review");
                    let chunk = rerun::log::Chunk::from_arrow_msg(&message).unwrap();
                    found_probe |= chunk.entity_path().to_string() == "/probe/value";
                }
            }
        }
        assert!(found_probe);
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
