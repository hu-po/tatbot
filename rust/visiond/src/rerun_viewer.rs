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
    /// When set, depth planes are logged RVL-compressed (`EncodedDepthImage`:
    /// lossless, on the D555's table a seventh of the raw bytes), and render
    /// as the same cloud under their Pinhole.
    compressed_depth: bool,
    /// Camera entities whose static display orientation has been logged
    /// (`orient_camera`), so it goes out once, on the first frame.
    oriented: Mutex<HashSet<String>>,
    /// The static Pinhole last logged per fixed-camera image entity; a
    /// changed camera model or payload size logs it again.
    pinholes: Mutex<HashMap<String, PinholeModel>>,
    /// The D555's newest colour frame, as logged: the bus sends its colour
    /// and depth apart (5 Hz and 1 Hz), and the depth cloud takes its colours
    /// from the latest colour image (`depth_cloud`).
    overhead_color: Mutex<Option<crate::FrameRecord>>,
    /// The newest capture stamp and the image size per colour image entity,
    /// for its age label.
    newest: Mutex<BTreeMap<String, (i128, [u32; 2])>>,
    /// Colour image entities labelled with a placeholder until their first
    /// frame ([`Self::await_image`]).
    awaited: Mutex<BTreeMap<String, Placeholder>>,
    /// The stamp of the frame each wrist colour image entity shows now, its
    /// markers cleared from just before it ([`Self::log_markers`]).
    shown: Mutex<HashMap<String, i64>>,
    tracking_registration: Mutex<Option<TrackingPlacement>>,
    /// The hub's stream when [`Self::connect_or_save`] opened one, so its
    /// loss can be seen ([`Self::hub_lost`]).
    hub: Option<HubSink>,
}

/// The hub's gRPC sink, shared by the recording and its viewer so the viewer
/// can read the stream's state, which a `RecordingStream` does not expose.
#[derive(Clone)]
struct HubSink(Arc<rerun::sink::GrpcSink>);

impl std::fmt::Debug for HubSink {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "HubSink({:?})", self.0.status())
    }
}

impl rerun::sink::LogSink for HubSink {
    fn send(&self, msg: rerun::log::LogMsg) {
        rerun::sink::LogSink::send(&*self.0, msg);
    }

    fn flush_blocking(
        &self,
        timeout: std::time::Duration,
    ) -> Result<(), rerun::sink::SinkFlushError> {
        rerun::sink::LogSink::flush_blocking(&*self.0, timeout)
    }

    fn default_batcher_config(&self) -> rerun::log::ChunkBatcherConfig {
        rerun::sink::LogSink::default_batcher_config(&*self.0)
    }
}

/// Where the subscriber draws trackd's and stencild's world-frame poses.
#[derive(Debug, Clone)]
enum TrackingPlacement {
    /// As published: the bus's camera world, which on the demo stack is the
    /// D555 colour optical frame and so the ROS `world` (the Demo view).
    CameraWorld,
    /// Moved into the URDF root by the robot-world registration that was
    /// solved against `calibration_id`.
    RobotRoot {
        calibration_id: String,
        root_from_world: RigidTransform,
    },
}

/// A fixed camera's pinhole at the size one payload has: focal length,
/// principal point and resolution, in that payload's pixels, and `scale`, its
/// pixels over the camera's own (what readers of full frames report in).
#[derive(Debug, Clone, Copy, PartialEq)]
struct PinholeModel {
    focal: [f32; 2],
    principal: [f32; 2],
    resolution: [f32; 2],
    scale: [f32; 2],
}

/// How a camera's panes are turned in the viewer, degrees about the image
/// normal (Rerun's 2D frame is y-down, so a negative angle reads
/// counter-clockwise on screen). The D405s sit rotated on the wrist, so
/// their panes are turned back upright: RealSense 2 (the upper) clockwise,
/// RealSense 1 (the lower, the upper's mount turned 180 degrees about the
/// wrist) counter-clockwise. This is a static
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

/// A label's point on a camera's image, in its pixels, placed where the
/// upper wrist camera's turn (+90) shows `at`: the lower one is turned the
/// other way (-90), so its point turns 180 deg about the image centre and its
/// labels read at the top of its pane, as the upper's do, not cut at the
/// bottom (station, 2026-10-23). `rotation` is `display_rotation_deg`'s.
fn label_point(rotation: Option<f32>, [width, height]: [u32; 2], [x, y]: [f32; 2]) -> [f32; 2] {
    match rotation {
        Some(degrees) if (degrees + 90.0).abs() < 1e-3 => [width as f32 - x, height as f32 - y],
        _ => [x, y],
    }
}

/// `display_rotation_deg` of the camera an image entity (`camera_entity`'s
/// `cameras/<nn>_<device>/<stream>`) belongs to.
fn entity_rotation(image: &str) -> Option<f32> {
    let camera = image.trim_start_matches('/').strip_prefix("cameras/")?;
    let (_, device) = camera.split('/').next()?.split_once('_')?;
    display_rotation_deg(device)
}

/// The fixed overhead D555's streams, `overhead_depth_<stream>`.
fn is_overhead_camera(sensor_name: &str) -> bool {
    sensor_name.starts_with("overhead_depth_")
}

/// The D555 colour stream its depth is aligned to.
const OVERHEAD_COLOR_SENSOR: &str = "overhead_depth_color";

/// The owner's full-resolution intrinsics (`tatbot.camera-intrinsics/1`:
/// `fx fy ppx ppy width height`, in the frame's `intrinsics` attribute)
/// scaled by payload size over full size. A bus or viewer downscale updates
/// the profile and keeps the attribute, so the two together give the model.
fn scaled_pinhole(metadata: &crate::FrameMetadata) -> Option<PinholeModel> {
    let intrinsics: serde_json::Value =
        serde_json::from_str(metadata.attributes.get("intrinsics")?).ok()?;
    let number = |key: &str| intrinsics[key].as_f64().filter(|v| v.is_finite());
    let (width, height) = (number("width")?, number("height")?);
    let (fx, fy) = (number("fx")?, number("fy")?);
    if width <= 0.0 || height <= 0.0 || fx <= 0.0 || fy <= 0.0 {
        return None;
    }
    let payload = [
        f64::from(metadata.profile.width),
        f64::from(metadata.profile.height),
    ];
    let scale = [payload[0] / width, payload[1] / height];
    Some(PinholeModel {
        focal: [(fx * scale[0]) as f32, (fy * scale[1]) as f32],
        principal: [
            (number("ppx")? * scale[0]) as f32,
            (number("ppy")? * scale[1]) as f32,
        ],
        resolution: payload.map(|v| v as f32),
        scale: scale.map(|v| v as f32),
    })
}

/// An image's age as its label shows it.
fn age_text(age_ns: i128) -> String {
    // A few ms of skew between the producer's clock and this one reads 0.
    let seconds = age_ns as f64 / 1e9;
    if seconds < -1.0 {
        "age ?".into()
    } else if seconds < 10.0 {
        let seconds = seconds.max(0.0);
        format!("{seconds:.1} s old")
    } else if seconds < 120.0 {
        format!("{seconds:.0} s old")
    } else {
        format!("{:.0} min old", seconds / 60.0)
    }
}

/// An awaited image's label and the size, pixels, its first frame will have.
type Placeholder = (String, [u32; 2]);

/// Not applicable yet: the demo status colours' grey.
const PLACEHOLDER_GREY: [u8; 3] = [127, 140, 141];

/// Green while live, amber once a frame is a few seconds behind, red when
/// the camera has stopped sending (the wrist D405 is open only inside a Draw
/// goal on hardware, so red there is normal between goals).
fn age_color(age_ns: i128) -> [u8; 3] {
    if age_ns < 2_000_000_000 {
        [90, 220, 90]
    } else if age_ns < 30_000_000_000 {
        [240, 190, 40]
    } else {
        [235, 80, 70]
    }
}

/// A wrist D405's streams, `realsense<n>_<stream>`: the only cameras whose
/// frames the session reads the print's markers in.
fn is_wrist_camera(sensor_name: &str) -> bool {
    sensor_name.starts_with("realsense")
}

/// One wrist reading's markers (`tatbot.wrist-markers/1`, written by
/// `scripts/lib/wrist_tap.py`): the print's markers the session read in one
/// frame its wrist tap published, drawn by [`RerunViewer::log_markers`].
#[derive(Debug, Clone, PartialEq, Deserialize)]
pub struct WristMarkers {
    /// The colour sensor the frame came from, `<camera>_color`.
    pub sensor: String,
    /// The frame's stamp: the frame set's, the time its image is logged at.
    pub frame_ns: i64,
    /// The frame's size as the session read it, pixels.
    pub size: [u32; 2],
    /// Which reading: `hold <n>`, `locate`, or a tracker reading's name.
    pub reading: String,
    /// Whether the reading's fit was accepted.
    pub accepted: bool,
    pub markers: Vec<WristMarker>,
}

#[derive(Debug, Clone, PartialEq, Deserialize)]
pub struct WristMarker {
    pub id: u32,
    /// TL/TR/BR/BL in the frame's pixels, pixel centres at integers.
    pub corners_px: [[f32; 2]; 4],
    /// Whether the reading's fit kept it.
    pub fit: bool,
}

/// A marker's colour: green kept by an accepted fit, amber kept by a fit
/// that was not accepted, red read and left out (`age_color`'s three).
fn marker_color(fit: bool, accepted: bool) -> [u8; 3] {
    match (fit, accepted) {
        (true, true) => [90, 220, 90],
        (true, false) => [240, 190, 40],
        _ => [235, 80, 70],
    }
}

/// The fiducials the trackers read in the overhead D555's colour, drawn on
/// it at `<image>/fiducials/<group>` ([`RerunViewer::log_tracking`]: the
/// wrist cube's tags, `ee/<target>`; [`RerunViewer::log_target`]: a print's
/// markers, `print/<pattern>`; [`RerunViewer::log_scene_tags`]: a scene
/// target's, the palette's, `<target>`, read by the cockpit itself). Ids
/// name only the cube's few tags; a print's markers are too many and too
/// small in that pane to label, and a scene tag carries its target's name.
const OVERHEAD_FIDUCIALS: &str = "fiducials";
/// The cube's tags the pose was solved from, and those read beside them.
const EE_TAG_USED: [u8; 3] = [80, 200, 255];
const EE_TAG_READ: [u8; 3] = [150, 150, 150];
/// A print's markers, in its outline's colour ([`RerunViewer::log_target`]).
const PRINT_MARKER: [u8; 3] = [255, 200, 0];
/// A scene target's tags, the palette's ([`RerunViewer::log_scene_tags`]).
#[cfg(feature = "fiducials")]
const SCENE_TAG: [u8; 3] = [120, 230, 140];

/// One marker's outline, in pixels of the image it is drawn on.
struct Quad {
    corners: [[f32; 2]; 4],
    color: [u8; 3],
    label: Option<String>,
}

impl Quad {
    /// A reader's corners (pixel centres at integers) in an image `scale`
    /// times the reader's, whose pixel centres Rerun puts at halves.
    fn scaled(corners: [[f32; 2]; 4], scale: [f32; 2], color: [u8; 3], label: Option<String>) -> Self {
        let corners = corners.map(|[u, v]| [(u + 0.5) * scale[0], (v + 0.5) * scale[1]]);
        Self { corners, color, label }
    }

    fn centre(&self) -> [f32; 2] {
        let [x, y] = self.corners.iter().fold([0.0, 0.0], |[x, y], [u, v]| [x + u, y + v]);
        [x / 4.0, y / 4.0]
    }
}

/// `n` corners, `[u, v]` each, from a JSON array of pairs (a reader's
/// pixels); None when it is not one.
fn json_corners(value: &serde_json::Value) -> Option<Vec<[f32; 2]>> {
    value
        .as_array()?
        .iter()
        .map(|pair| match pair.as_array()?.as_slice() {
            [u, v] => Some([u.as_f64()? as f32, v.as_f64()? as f32]),
            _ => None,
        })
        .collect()
}

/// The wrist cube's tags a tracking pose's overhead colour detections read
/// (`estimate.detections.overhead_depth_color`: corners in the camera's own
/// pixels), those it was solved from (`tag_ids`) in [`EE_TAG_USED`].
fn ee_tag_quads(value: &serde_json::Value, scale: [f32; 2]) -> Vec<Quad> {
    let used: HashSet<u64> = value["tag_ids"]
        .as_array()
        .into_iter()
        .flatten()
        .filter_map(serde_json::Value::as_u64)
        .collect();
    let solved = value["cameras"]
        .as_array()
        .is_some_and(|cameras| cameras.iter().any(|c| c == OVERHEAD_COLOR_SENSOR));
    value["estimate"]["detections"][OVERHEAD_COLOR_SENSOR]
        .as_array()
        .into_iter()
        .flatten()
        .filter_map(|detection| {
            let id = detection["tag_id"].as_u64()?;
            let corners: [[f32; 2]; 4] = json_corners(&detection["corners_px"])?.try_into().ok()?;
            let color = if solved && used.contains(&id) { EE_TAG_USED } else { EE_TAG_READ };
            Some(Quad::scaled(corners, scale, color, Some(id.to_string())))
        })
        .collect()
}

/// A measured print's markers the overhead colour kept
/// (`support.anchor_pixels`, four corners a marker in `support.markers.ids`'
/// order, the camera's own pixels); none for another anchor camera, a lost
/// print or a sample without them.
fn print_marker_quads(value: &serde_json::Value, scale: [f32; 2]) -> Vec<Quad> {
    let support = &value["support"];
    if value["source"] != "measured" || support["anchor_camera"] != OVERHEAD_COLOR_SENSOR {
        return Vec::new();
    }
    let ids = support["markers"]["ids"].as_array().map_or(0, Vec::len);
    match json_corners(&support["anchor_pixels"]) {
        Some(pixels) if pixels.len() == 4 * ids => pixels
            .chunks_exact(4)
            .map(|c| Quad::scaled([c[0], c[1], c[2], c[3]], scale, PRINT_MARKER, None))
            .collect(),
        _ => Vec::new(),
    }
}

/// The line naming a wrist reading under the image's age label.
fn markers_text(read: &WristMarkers) -> String {
    let used = read.markers.iter().filter(|m| m.fit).count();
    let all = read.markers.len();
    match (used, read.accepted) {
        (0, _) => format!("{}: {all} markers, none used", read.reading),
        (_, true) => format!("{}: {used} of {all} markers used", read.reading),
        (_, false) => format!("{}: {used} of {all} markers, fit rejected", read.reading),
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
            compressed_depth: false,
            oriented: Mutex::new(HashSet::new()),
            pinholes: Mutex::new(HashMap::new()),
            overhead_color: Mutex::new(None),
            newest: Mutex::new(BTreeMap::new()),
            awaited: Mutex::new(BTreeMap::new()),
            shown: Mutex::new(HashMap::new()),
            tracking_registration: Mutex::new(None),
            hub: None,
        }
    }

    /// Whether the hub dropped this viewer's stream (a hub restart does).
    /// Rerun 0.36's gRPC client never redials, and every later log is
    /// discarded without an error. A hub not up yet (connecting) is not lost.
    pub fn hub_lost(&self) -> bool {
        self.hub.as_ref().is_some_and(|hub| {
            matches!(hub.0.status(), rerun::sink::GrpcSinkConnectionState::Disconnected(_))
        })
    }

    /// Bind the subscriber's world-frame poses to the same measured URDF root.
    pub fn bind_tracking_registration(&self, path: &Path) -> Result<()> {
        let (pose, id) = root_from_world_transform(path)?;
        let id = id
            .filter(|id| !id.is_empty())
            .context("tracking registration has no calibration_id")?;
        *self.tracking_registration.lock().unwrap() = Some(TrackingPlacement::RobotRoot {
            calibration_id: id,
            root_from_world: pose,
        });
        Ok(())
    }

    /// Draw the subscriber's world-frame poses as published, at identity in
    /// the bus's camera world: the frame the Demo view and the ROS bridge
    /// share. No robot-world registration is involved.
    pub fn place_tracking_in_camera_world(&self) {
        *self.tracking_registration.lock().unwrap() = Some(TrackingPlacement::CameraWorld);
    }

    /// One entity per tracked target, the pink arm's `wrist` at its
    /// unsuffixed path.
    pub fn log_tracking(&self, stamp_ns: u64, value: &serde_json::Value) -> Result<()> {
        let (axes_entity, status) = tracking_entities(value)?;
        let group = format!("ee/{}", value["target"].as_str().unwrap_or_default());
        self.log_overhead_fiducials(&group, stamp_ns, |scale| ee_tag_quads(value, scale))?;
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
    /// `world/tracking/target/<pattern_id>`, in the URDF root or the camera
    /// world (`TrackingPlacement`), the sample beside it. A lost print clears
    /// its outline and keeps its status. The axes are the target frame the
    /// observer publishes, not the page frame (y and z flipped).
    /// A scene target's tags (`detections`, the cockpit's own reading of the
    /// overhead colour frame stamped `stamp_ns`, corners in the camera's own
    /// pixels) outlined on that image at `fiducials/<target>`, each labelled
    /// with the target's name; none read clears them.
    #[cfg(feature = "fiducials")]
    pub fn log_scene_tags(
        &self,
        target: &str,
        stamp_ns: u64,
        detections: &[crate::fiducials::FiducialDetection],
    ) -> Result<()> {
        self.log_overhead_fiducials(target, stamp_ns, |scale| {
            detections
                .iter()
                .map(|d| {
                    let corners = d.corners_px.map(|[u, v]| [u as f32, v as f32]);
                    Quad::scaled(corners, scale, SCENE_TAG, Some(target.to_owned()))
                })
                .collect()
        })
    }

    pub fn log_target(&self, stamp_ns: u64, value: &serde_json::Value) -> Result<()> {
        let (outline_entity, status) = target_entities(value)?;
        let group = format!("print/{}", value["target_id"].as_str().unwrap_or_default());
        self.log_overhead_fiducials(&group, stamp_ns, |scale| print_marker_quads(value, scale))?;
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
        let placement = self.tracking_registration.lock().unwrap().clone();
        let outline = registered_pose(value, "world_from_target", placement.as_ref())
            .and_then(|pose| target_outline(value, placement.as_ref(), pose));
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
            .save(path)
            .with_context(|| format!("creating Rerun recording {}", path.display()))?;
        viewer_blueprint().send(&recording)?;
        Ok(Self::from_recording(recording))
    }

    pub fn spawn() -> Result<Self> {
        let recording = rerun::RecordingStreamBuilder::new(APP_ID)
            .recording_name("Tatbot")
            .spawn()
            .context("starting the Rerun Viewer; install the `rerun` executable or use --output")?;
        viewer_blueprint().send(&recording)?;
        Ok(Self::from_recording(recording))
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
        // A shared recording id lets another producer — the Python surface
        // reconstruction — log into this same recording, so its geometry
        // overlays the live camera stream instead of opening beside it.
        if let Some(id) = recording_id {
            builder = builder.recording_id(id);
        }
        let recording = builder
            .connect_grpc_opts(url)
            .with_context(|| format!("connecting to Rerun viewer at {url}"))?;
        if install_blueprint {
            viewer_blueprint().send(&recording)?;
        }
        // `recording_id` above switched the SDK's default properties off, so
        // the shared name is sent explicitly; a bare run id is the session's.
        if let Some(name) = recording_id.and_then(recording_name) {
            recording.send_recording_name(name)?;
        }
        Ok(Self::from_recording(recording))
    }

    /// The cockpit's sinks: the hub's gRPC proxy, an `.rrd` file, or both.
    /// A file alone embeds the fixed blueprint so it opens standalone; a
    /// stream that reaches the hub never sends one (the rerun-server owns it).
    /// Every sink batches as the hub's stream does (an 8 ms tick, where a
    /// file alone would take 200 ms): the cockpit logs an image at most
    /// every tick of its own (20 ms), so each image row is a chunk alone,
    /// and a file capture holds the chunks the hub streams on.
    pub fn connect_or_save(
        url: Option<&str>,
        output: Option<&Path>,
        recording_id: Option<&str>,
    ) -> Result<Self> {
        let mut builder = rerun::RecordingStreamBuilder::new(APP_ID)
            .recording_name("Tatbot")
            .batcher_config(rerun::log::ChunkBatcherConfig::LOW_LATENCY);
        if let Some(id) = recording_id {
            builder = builder.recording_id(id);
        }
        let (recording, hub) = match (url, output) {
            (None, Some(path)) => {
                let recording = builder
                    .save(path)
                    .with_context(|| format!("creating Rerun recording {}", path.display()))?;
                viewer_blueprint().send(&recording)?;
                (recording, None)
            }
            (Some(url), output) => {
                let rerun::external::re_uri::RedapUri::Proxy(uri) = url
                    .parse()
                    .with_context(|| format!("parsing Rerun viewer URL {url}"))?
                else {
                    anyhow::bail!("{url} is not a Rerun gRPC proxy URL");
                };
                let hub = HubSink(Arc::new(rerun::sink::GrpcSink::new(uri)));
                let mut sink: Box<dyn rerun::sink::LogSink> = Box::new(hub.clone());
                if let Some(path) = output {
                    let file = rerun::sink::FileSink::new(path)
                        .with_context(|| format!("creating Rerun recording {}", path.display()))?;
                    sink = Box::new(rerun::sink::MultiSink::new(vec![sink, Box::new(file)]));
                }
                // What `connect_grpc_opts` / `set_sinks` build, with this sink;
                // `set_sink` instead would flush a hub not up yet and break it.
                let (enabled, info, properties, batcher, hooks, _) = builder.into_args();
                let recording = if enabled {
                    rerun::RecordingStream::new(info, properties, batcher, hooks, sink)
                        .with_context(|| format!("connecting to Rerun viewer at {url}"))?
                } else {
                    rerun::RecordingStream::disabled()
                };
                (recording, Some(hub))
            }
            (None, None) => anyhow::bail!("the viewer needs --connect, --output or both"),
        };
        if let Some(name) = recording_id.and_then(recording_name) {
            recording.send_recording_name(name)?;
        }
        Ok(Self { hub, ..Self::from_recording(recording) })
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

    /// Log depth planes RVL-compressed from now on ([`Self::compressed_depth`]).
    pub fn set_compressed_depth(&mut self, compressed: bool) {
        self.compressed_depth = compressed;
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
            self.log_pinhole(sensor_name, &image_entity, &frame.metadata)?;
            if sensor_name == OVERHEAD_COLOR_SENSOR {
                *self.overhead_color.lock().unwrap() = Some(frame.clone());
            }
            if is_wrist_camera(sensor_name)
                && sensor_name.ends_with("_color")
                && let Ok(timestamp_ns) = i64::try_from(set.timestamp_ns)
            {
                // A reading's markers belong to the frame they were read in
                // (log_markers, at its stamp, or this one's when they come
                // after it): hidden from just before this one.
                self.recording
                    .set_timestamp_nanos_since_epoch(TIMELINE, timestamp_ns.saturating_sub(1));
                self.recording.log(
                    format!("{image_entity}/markers").as_str(),
                    &rerun::Clear::recursive(),
                )?;
                let mut shown = self.shown.lock().unwrap();
                let at = shown.entry(image_entity.clone()).or_insert(timestamp_ns);
                *at = (*at).max(timestamp_ns);
                self.recording
                    .set_timestamp_nanos_since_epoch(TIMELINE, timestamp_ns);
            }
            if sensor_name.ends_with("_color") {
                let stamp = frame
                    .metadata
                    .timestamps
                    .normalized_unix_ns
                    .unwrap_or(set.timestamp_ns);
                let size = [frame.metadata.profile.width, frame.metadata.profile.height];
                let mut newest = self.newest.lock().unwrap();
                let seen = newest.entry(image_entity.clone()).or_insert((stamp, size));
                *seen = (seen.0.max(stamp), size);
            }
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
                    let meter = depth_meter(&frame.metadata.attributes);
                    if is_overhead_camera(sensor_name) {
                        self.log_depth_cloud(set, &entity, frame, (*width, *height, bytes), meter)?;
                    }
                    if self.compressed_depth {
                        let depth = rerun::EncodedDepthImage::new(crate::frame_ops::rvl_depth(
                            *width, *height, bytes,
                        ))
                        .with_media_type(rerun::components::MediaType::rvl())
                        .with_meter(meter);
                        self.recording.log(image_entity.as_str(), &depth)?;
                    } else {
                        let depth = rerun::DepthImage::from_gray16(bytes.clone(), [*width, *height])
                            .with_meter(meter);
                        self.recording.log(image_entity.as_str(), &depth)?;
                    }
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
                        "Rerun needs decoded pixels; {} contains encoded {:?}",
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

    /// The D555's depth as an RGB-coloured cloud in rows `<camera>/cloud/<band>`, in its
    /// colour optical frame (the camera entity sits at identity in camera
    /// world), coloured from the newest colour frame: this set's, else the
    /// last one logged. The 3D scene shows this cloud instead of the
    /// colormapped one Rerun draws from the depth image under its Pinhole;
    /// the depth pane keeps that image (`depth_cloud`).
    fn log_depth_cloud(
        &self,
        set: &SynchronizedFrameSet,
        camera: &str,
        frame: &crate::FrameRecord,
        depth: (u32, u32, &[u8]),
        meter: f32,
    ) -> Result<()> {
        use crate::depth_cloud::{BAND_POINTS, BANDS, CloudSource, ColorRaster, MAX_POINTS, back_project};
        let Some(model) = scaled_pinhole(&frame.metadata) else {
            return Ok(());
        };
        let color = match set.frames.get(OVERHEAD_COLOR_SENSOR) {
            Some(companion) => ColorRaster::from_frame(companion),
            None => self
                .overhead_color
                .lock()
                .unwrap()
                .as_ref()
                .and_then(ColorRaster::from_frame),
        };
        let cloud = back_project(
            depth,
            meter,
            model.focal,
            model.principal,
            color.as_ref(),
            CloudSource::Overhead,
            MAX_POINTS,
        );
        for band in 0..BANDS {
            let rows = (band * BAND_POINTS).min(cloud.positions.len())
                ..((band + 1) * BAND_POINTS).min(cloud.positions.len());
            self.recording.log(
                format!("{camera}/cloud/{band}").as_str(),
                &rerun::Points3D::new(cloud.positions[rows.clone()].iter().copied())
                    .with_colors(
                        cloud.colors[rows.clone()]
                            .iter()
                            .map(|&[r, g, b]| rerun::Color::from_rgb(r, g, b)),
                    )
                    .with_radii([cloud.radius_m(rows)]),
            )?;
        }
        Ok(())
    }

    /// Log the camera's display orientation once: a wrist camera's in-plane
    /// rotation (`display_rotation_deg`), or the overhead D555's view
    /// coordinates. The D555 entity sits at identity, so its frame is the
    /// camera world (the ROS `world`, RDF: z into the table). "X right, Y
    /// back, Z down" makes a view rooted there take the camera's -z as up;
    /// the Demo 3D view is rooted at the bridge's level arm-base frame
    /// instead, because the camera looks down at a slant.
    fn orient_camera(&self, sensor_name: &str, entity: &str) -> Result<()> {
        let degrees = display_rotation_deg(sensor_name);
        let overhead = is_overhead_camera(sensor_name);
        if degrees.is_none() && !overhead {
            return Ok(());
        }
        let mut oriented = self
            .oriented
            .lock()
            .map_err(|_| anyhow!("camera orientation state poisoned"))?;
        if !oriented.insert(entity.to_owned()) {
            return Ok(());
        }
        if let Some(degrees) = degrees {
            self.recording.log_static(
                entity,
                &rerun::Transform3D::from_rotation(rerun::RotationAxisAngle::new(
                    [0.0, 0.0, 1.0],
                    rerun::Angle::from_degrees(degrees),
                )),
            )?;
        }
        if overhead {
            self.recording
                .log_static(entity, &rerun::ViewCoordinates::RIGHT_HAND_Z_DOWN())?;
        }
        Ok(())
    }

    /// A static Pinhole on a fixed camera's image entity from the owner's
    /// full-resolution intrinsics scaled to this payload, so the depth image
    /// becomes a cloud in camera world and both 2D panes keep pixel units.
    /// Logged again only when the model or the payload size changes.
    fn log_pinhole(
        &self,
        sensor_name: &str,
        image_entity: &str,
        metadata: &crate::FrameMetadata,
    ) -> Result<()> {
        if !is_overhead_camera(sensor_name) {
            return Ok(());
        }
        let Some(model) = scaled_pinhole(metadata) else {
            return Ok(());
        };
        let mut pinholes = self.pinholes.lock().unwrap();
        if pinholes.get(image_entity) == Some(&model) {
            return Ok(());
        }
        let pinhole =
            rerun::Pinhole::from_focal_length_and_resolution(model.focal, model.resolution)
                .with_principal_point(model.principal)
                .with_camera_xyz(rerun::components::ViewCoordinates::RDF)
                .with_image_plane_distance(0.1);
        self.recording.log_static(image_entity, &pinhole)?;
        // The 3D scene leaves the depth image out (it draws the RGB cloud
        // instead), so the depth's frustum goes out again on its own entity.
        if let Some((camera, "depth")) = image_entity.rsplit_once('/') {
            self.recording
                .log_static(format!("{camera}/frustum").as_str(), &pinhole)?;
        }
        pinholes.insert(image_entity.to_owned(), model);
        Ok(())
    }

    /// Label sensor `sensor_name`'s colour image with `placeholder` in
    /// [`Self::log_ages`] until a frame of it has come, when the label is
    /// cleared and the age label takes its place. `frame` is the size,
    /// pixels, that frame will be logged at: the placeholder outlines it, and
    /// the camera is turned now as its frames will be, so the pane frames the
    /// placeholder as it will the image (a wrist pane upright, 9:16).
    ///
    /// At `now_ns` the age label is cleared and a black frame covers the
    /// image: an earlier process's last frame stays in the standing
    /// recording, under an age label frozen when that process stopped, and
    /// this one cannot tell what a viewer still holds (it never reads the hub
    /// back, and the hub drops its oldest rows at its memory limit). Rerun
    /// 0.36 still draws an `EncodedImage` past a `Clear`, so the frame is
    /// covered, at the size and with the encoder of the frames to come.
    pub fn await_image(
        &self,
        sensor_name: &str,
        placeholder: &str,
        frame: [u32; 2],
        now_ns: i128,
    ) -> Result<()> {
        let (camera, image) = camera_entity(sensor_name);
        self.orient_camera(sensor_name, &camera)?;
        self.clear(&[&format!("{image}/age")], now_ns)?;
        self.recording
            .log(format!("{image}/markers").as_str(), &rerun::Clear::recursive())?;
        let [width, height] = frame;
        let black = vec![0; width as usize * height as usize * 3];
        let blank = encode_jpeg(PixelFormat::Rgb8, width, height, &black, 70)?;
        self.recording
            .log(image.as_str(), &rerun::EncodedImage::from_file_contents(blank))?;
        self.awaited
            .lock()
            .unwrap()
            .insert(image, (placeholder.to_owned(), frame));
        Ok(())
    }

    /// Clear `entities` now (`Clear::flat`, on the wall-clock timeline): what
    /// an earlier producer logged there is no longer drawn, except an
    /// `EncodedImage` in Rerun 0.36 ([`Self::await_image`]). Static rows
    /// (a Pinhole) are untouched.
    pub fn clear(&self, entities: &[&str], now_ns: i128) -> Result<()> {
        self.recording.reset_time();
        self.recording
            .set_timestamp_nanos_since_epoch(TIMELINE, i64::try_from(now_ns)?);
        for entity in entities {
            self.recording.log(*entity, &rerun::Clear::flat())?;
        }
        Ok(())
    }

    /// The awaited images a frame has now come for (no longer awaited), and
    /// the ones still waiting with their placeholders.
    fn take_awaited(
        &self,
        newest: &BTreeMap<String, (i128, [u32; 2])>,
    ) -> (Vec<String>, Vec<(String, Placeholder)>) {
        let mut awaited = self.awaited.lock().unwrap();
        let arrived: Vec<String> = awaited
            .keys()
            .filter(|entity| newest.contains_key(*entity))
            .cloned()
            .collect();
        for entity in &arrived {
            awaited.remove(entity);
        }
        let waiting = awaited
            .iter()
            .map(|(e, t)| (e.clone(), t.clone()))
            .collect();
        (arrived, waiting)
    }

    /// An awaited image's placeholder, `<image>/waiting`: a grey outline of
    /// the frame to come, so the pane frames it as it would the image, and
    /// the label at its centre.
    fn log_waiting(&self, entity: &str, text: &str, frame: [u32; 2]) -> Result<()> {
        let path = format!("{entity}/waiting");
        let [width, height] = frame.map(|v| v as f32);
        self.recording.log(
            path.as_str(),
            &rerun::Boxes2D::from_mins_and_sizes([[0.0, 0.0]], [[width, height]])
                .with_colors([PLACEHOLDER_GREY]),
        )?;
        self.recording.log(
            path.as_str(),
            &rerun::Points2D::new([[width / 2.0, height / 2.0]])
                .with_radii([rerun::Radius::new_ui_points(3.0)])
                .with_colors([PLACEHOLDER_GREY])
                .with_labels([text])
                .with_show_labels(true),
        )?;
        Ok(())
    }

    /// Draw a wrist reading's markers on the frame they were read in,
    /// `<image>/markers`: each marker's outline in [`marker_color`], its id
    /// at its centre (`markers/id`) and a line naming the reading
    /// (`markers/reading`), centred across the upright pane a little below
    /// the age label, so its width is not cut. Logged at the frame's stamp in the
    /// pixels the cockpit logs that camera's frames at (or its placeholder
    /// outlines); its next frame clears them ([`Self::log_set`]). Markers that
    /// come after the next frame go on the frame shown, at its stamp: under a
    /// clear already logged they never showed live, and a stroke's readings
    /// come every 0.5 s and take ~0.4 s to fit (2026-10-24). Nothing before a
    /// frame or placeholder of that camera gives the size.
    pub fn log_markers(&self, read: &WristMarkers) -> Result<()> {
        let (_, image) = camera_entity(&read.sensor);
        let newest = self.newest.lock().unwrap().get(&image).map(|(_, size)| *size);
        let shown = newest.or_else(|| self.awaited.lock().unwrap().get(&image).map(|(_, f)| *f));
        let Some([width, height]) = shown else {
            return Ok(());
        };
        anyhow::ensure!(
            read.size.iter().all(|&v| v > 0),
            "markers of a {}x{} frame",
            read.size[0],
            read.size[1]
        );
        let scale = [
            width as f32 / read.size[0] as f32,
            height as f32 / read.size[1] as f32,
        ];
        let quads: Vec<Quad> = read
            .markers
            .iter()
            .map(|m| Quad::scaled(m.corners_px, scale, marker_color(m.fit, read.accepted), Some(m.id.to_string())))
            .collect();
        let entity = format!("{image}/markers");
        let at = self.shown.lock().unwrap().get(&image).map_or(read.frame_ns, |&shown| shown.max(read.frame_ns));
        self.recording.reset_time();
        self.recording
            .set_timestamp_nanos_since_epoch(TIMELINE, at);
        self.log_quads(&entity, &quads, 1.5)?;
        let used = read.markers.iter().any(|m| m.fit);
        self.recording.log(
            format!("{entity}/reading").as_str(),
            &rerun::Points2D::new([label_point(
                display_rotation_deg(&read.sensor),
                [width, height],
                [width as f32 * 0.18, height as f32 * 0.5],
            )])
                .with_radii([rerun::Radius::new_ui_points(3.0)])
                .with_colors([marker_color(used, read.accepted)])
                .with_labels([markers_text(read)])
                .with_show_labels(true),
        )?;
        Ok(())
    }

    /// Re-log every colour image's age label, `<image>/age`: a point inside
    /// the image's top-left corner, in its pixels (far enough in that the
    /// centred label is not cut by the pane's edge), labelled with how long
    /// ago the newest frame seen for it was captured. An awaited image with
    /// no frame yet shows its placeholder instead ([`Self::log_waiting`]).
    /// Outlines of `quads` at `entity`, `radius` UI points wide, and the
    /// labelled ones' labels at their centres (`<entity>/id`), at the
    /// recording's current time.
    fn log_quads(&self, entity: &str, quads: &[Quad], radius: f32) -> Result<()> {
        self.recording.log(
            entity,
            &rerun::LineStrips2D::new(quads.iter().map(|q| {
                let [a, b, c, d] = q.corners;
                vec![a, b, c, d, a]
            }))
            .with_colors(quads.iter().map(|q| q.color))
            .with_radii([rerun::Radius::new_ui_points(radius)]),
        )?;
        let labelled: Vec<&Quad> = quads.iter().filter(|q| q.label.is_some()).collect();
        let id = format!("{entity}/id");
        if labelled.is_empty() {
            self.recording.log(id.as_str(), &rerun::Clear::flat())?;
            return Ok(());
        }
        self.recording.log(
            id.as_str(),
            &rerun::Points2D::new(labelled.iter().map(|q| q.centre()))
                .with_radii([rerun::Radius::new_ui_points(1.0)])
                .with_colors(labelled.iter().map(|q| q.color))
                .with_labels(labelled.iter().map(|q| q.label.clone().unwrap_or_default()))
                .with_show_labels(true),
        )?;
        Ok(())
    }

    /// A tracker's fiducials on the overhead D555's colour image, at the
    /// stamp of the frame they were read in: `quads` (built in the camera's
    /// own pixels times the logged image's `scale`) at
    /// `<image>/fiducials/<group>`, replacing that group's last; none read
    /// clears them. Nothing before the image's first frame gives the scale.
    fn log_overhead_fiducials(
        &self,
        group: &str,
        stamp_ns: u64,
        quads: impl FnOnce([f32; 2]) -> Vec<Quad>,
    ) -> Result<()> {
        let (_, image) = camera_entity(OVERHEAD_COLOR_SENSOR);
        let Some(scale) = self.pinholes.lock().unwrap().get(&image).map(|model| model.scale) else {
            return Ok(());
        };
        let quads = quads(scale);
        let entity = format!("{image}/{OVERHEAD_FIDUCIALS}/{group}");
        self.recording.reset_time();
        self.recording
            .set_timestamp_nanos_since_epoch(TIMELINE, i64::try_from(stamp_ns)?);
        if quads.is_empty() {
            self.recording.log(entity.as_str(), &rerun::Clear::recursive())?;
            return Ok(());
        }
        self.log_quads(&entity, &quads, 1.0)
    }

    pub fn log_ages(&self, now_ns: i128) -> Result<()> {
        let newest = self.newest.lock().unwrap().clone();
        let (arrived, waiting) = self.take_awaited(&newest);
        if newest.is_empty() && waiting.is_empty() {
            return Ok(());
        }
        self.recording.reset_time();
        self.recording
            .set_timestamp_nanos_since_epoch(TIMELINE, i64::try_from(now_ns)?);
        for entity in arrived {
            self.recording
                .log(format!("{entity}/waiting"), &rerun::Clear::flat())?;
        }
        for (entity, (text, frame)) in waiting {
            self.log_waiting(&entity, &text, frame)?;
        }
        for (entity, (stamp, [width, height])) in newest {
            let age_ns = now_ns - stamp;
            let at = label_point(
                entity_rotation(&entity),
                [width, height],
                [width as f32 * 0.12, height as f32 * 0.08],
            );
            self.recording.log(
                format!("{entity}/age"),
                &rerun::Points2D::new([at])
                    .with_radii([rerun::Radius::new_ui_points(3.0)])
                    .with_colors([age_color(age_ns)])
                    .with_labels([age_text(age_ns)])
                    .with_show_labels(true),
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
    Ok(crate::frame_ops::yuyv_to_rgb(bytes))
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
    placement: Option<&TrackingPlacement>,
) -> Result<RigidTransform> {
    registered_pose(value, "pose", placement)
}

/// The viewer frame's pose of the bus's calibration (camera) world: identity
/// in camera-world placement, else the robot-world registration of the same
/// bundle that the sample names.
fn viewer_from_world(
    value: &serde_json::Value,
    placement: Option<&TrackingPlacement>,
) -> Result<RigidTransform> {
    match placement.context("no robot-world registration supplied")? {
        TrackingPlacement::CameraWorld => Ok(RigidTransform::IDENTITY),
        TrackingPlacement::RobotRoot {
            calibration_id,
            root_from_world,
        } => {
            anyhow::ensure!(
                value["calibration_id"].as_str() == Some(calibration_id.as_str()),
                "tracking calibration ID differs from robot-world registration"
            );
            Ok(*root_from_world)
        }
    }
}

/// A bus pose in the calibration world, placed in the viewer's frame.
fn registered_pose(
    value: &serde_json::Value,
    field: &str,
    placement: Option<&TrackingPlacement>,
) -> Result<RigidTransform> {
    let viewer_from_world = viewer_from_world(value, placement)?;
    Ok(viewer_from_world.multiply(rigid_from_json(&value[field])?))
}

/// The closed page outline in the viewer's frame from the observer's
/// world-frame corners; axes at the placed pose when the sample carries none.
fn target_outline(
    value: &serde_json::Value,
    placement: Option<&TrackingPlacement>,
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
    let viewer_from_world = viewer_from_world(value, placement)?;
    let mut strip: Vec<[f32; 3]> = corners
        .iter()
        .map(|corner| viewer_from_world.transform_point(*corner).map(|v| v as f32))
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

/// The Demo scene's default eye, in its origin frame (`draw/ros/level`: the
/// arm base, x out to the page, z up), metres: above the operator's side of
/// the table (the D555's, +y), looking down onto the page, the palette and
/// the arm. Fixed, because the viewer's own default frames the whole depth
/// cloud and the room beyond it, and re-frames as the cloud grows. Rig
/// geometry (2026-10-18): page centre (0.33, 0.05, 0.04), palette (0.31,
/// 0.30, 0.03), the arm's elbow up to z 0.37.
const DEMO_EYE: [f32; 3] = [0.44, 0.68, 0.60];
const DEMO_LOOK_AT: [f32; 3] = [0.24, 0.10, 0.15];
const _: () = assert!(DEMO_EYE[2] > DEMO_LOOK_AT[2], "the Demo eye looks down");

/// The bridge's status board canvas (`draw/ros/status_board`, x right, y
/// down), shown whole and never re-zoomed: its units are the station
/// pane's pixels at 860 wide.
const STATUS_BOARD_BOUNDS: ([f64; 2], [f64; 2]) = ([0.0, 860.0], [0.0, 150.0]);

/// The wrist D405s' colour images in the Demo tab, each pane's title and
/// sensor (`camera_entity` places its image), by device number: D405-1 the
/// lower (vision.toml wrist_lower), D405-2 the upper, the session's own
/// (owner, 2026-10-23: "D405-1 and D405-2, not upper/lower wrist"). A title
/// is short: a 270 px pane's bar shows ten characters beside its buttons.
/// Each camera
/// is open only inside a Draw goal (the lower one beside the session's own,
/// display only: stack.yaml `wrist_tap.preview`), so the cockpit labels an
/// empty pane until a frame comes.
pub const DEMO_WRISTS: [(&str, &str); 2] = [
    ("D405-1", "realsense1_color"),
    ("D405-2", "realsense2_color"),
];

/// The station's Demo page in its own pixels (an 860x1920 portrait screen,
/// 2026-10-18): 1808 high between the viewer's tab bar and its collapsed
/// time panel, each view under a 34 px title bar, 2 px between tiles. The
/// Demo tab's shares are these pixels, so each camera pane takes the shape
/// of its image.
const STATION_PAGE: [f32; 2] = [860.0, 1808.0];
const STATION_TITLE_BAR: f32 = 34.0;
const STATION_GAP: f32 = 2.0;
/// Both colour images' width over height as the cockpit logs them (640x360).
/// The wrists' are turned a quarter on screen (`display_rotation_deg`): 9:16.
const DEMO_IMAGE_ASPECT: f32 = 16.0 / 9.0;
/// The cockpit's wrist frame, pixels (frame_bus `VIEWER_COLOR_MAX`).
const DEMO_WRIST_FRAME: [f64; 2] = [640.0, 360.0];
/// The Demo panes' sizes on the station, pixels: the status board's 150 px
/// canvas under its title bar, the upright wrist images' height (270 wide
/// each, two beside the design's page: 316 wide), and the D555 image's
/// width (270 high). The 3D scene takes the rest.
const DEMO_STATUS_ROW: f32 = 188.0;
const DEMO_WRIST_HEIGHT: f32 = 480.0;
const DEMO_D555_WIDTH: f32 = 480.0;

/// A wrist pane's bounds: the cockpit's wrist frame turned by its camera's
/// `display_rotation_deg` about the image origin (a quarter clockwise is
/// (x, y) -> (-y, x)). Left to Rerun, the pane's fit kept the union of all
/// it had shown and drew the frame small in one corner of its pane
/// (station, 2026-10-21).
fn upright_bounds(sensor_name: &str, [width, height]: [f64; 2]) -> ([f64; 2], [f64; 2]) {
    let (sin, cos) = f64::from(display_rotation_deg(sensor_name).unwrap_or(0.0))
        .to_radians()
        .sin_cos();
    let corners = [[0.0, 0.0], [width, 0.0], [0.0, height], [width, height]]
        .map(|[x, y]| [x * cos - y * sin, x * sin + y * cos]);
    let range = |axis: usize| {
        let values = corners.map(|corner| corner[axis].round());
        [
            values.iter().copied().fold(f64::INFINITY, f64::min),
            values.iter().copied().fold(f64::NEG_INFINITY, f64::max),
        ]
    };
    (range(0), range(1))
}
/// The CPU and TEMP panes' sources, each a `system/<entity>/` of
/// `scripts/lib/tatbot_sysmon.py`: the palette Pi (read by the station),
/// the station, the ros node and the arm it drives. Their legends (the
/// fleet map's `platform`) and colours are logged with the series.
const DEMO_HOSTS: [&str; 4] = ["estop-relay", "station", "ros", "robot"];
/// How much history the CPU and TEMP panes show, up to the time cursor.
const DEMO_WINDOW_S: i64 = 600;
/// CPU %'s fixed scale, a little past 0-100 so both end labels are drawn:
/// its legend sits in the top half, above the lines of a fleet at work.
const DEMO_CPU_RANGE: [f64; 2] = [-2.0, 102.0];

/// A view property row the Rust 0.36 builders cannot write
/// (`View::add_property` is crate-private): an archetype at
/// `<view blueprint path>/<archetype name>`, as the SDK's own setters log it.
type ViewProperty = (rerun::EntityPath, Vec<rerun::SerializedComponentBatch>);

/// `view` as a container child, with `archetype` queued as its property.
fn with_property(
    view: impl Into<rerun::blueprint::ContainerLike>,
    name: &str,
    archetype: &dyn rerun::AsComponents,
    properties: &mut Vec<ViewProperty>,
) -> rerun::blueprint::ContainerLike {
    let view = view.into();
    if let rerun::blueprint::ContainerLike::View(inner) = &view {
        let path = inner.blueprint_path().join(&rerun::EntityPath::from(name));
        properties.push((path, archetype.as_serialized_batches()));
    }
    view
}

/// The Demo tab, laid out top to bottom for the station's 860x1920
/// portrait window: the status board, the 3D scene in camera world, the
/// portrait panes side by side (the design map's page and the upright
/// wrist images), then the landscape D555 image
/// beside CPU load and temperatures. Rust 0.36 cannot select the active tab,
/// so it is the first one.
fn demo_tab(properties: &mut Vec<ViewProperty>) -> rerun::blueprint::ContainerLike {
    use rerun::blueprint::{Horizontal, Spatial2DView, Spatial3DView, TimeSeriesView, Vertical};
    use rerun::external::re_sdk_types::blueprint::archetypes::{
        EyeControls3D, PlotLegend, ScalarAxis, TimeAxis, VisualBounds2D,
    };
    use rerun::external::re_sdk_types::blueprint::components::{Corner2D, TimeRange};
    use rerun::external::re_sdk_types::datatypes::{
        Range2D, TimeInt, TimeRange as TimeRangeData, TimeRangeBoundary,
    };
    // The origin is the bridge's `draw/ros/level`: the arm base's frame
    // (z up, the table's normal) in camera world, so the table is level on
    // screen however far the D555 is tilted (38 degrees on the rig; the
    // camera's own -z is up only for a camera looking straight down). The
    // cloud and `/world/tracking` reach it through its inverse. The colour
    // image plane is left out: hung at its image-plane distance it only
    // hides the scene. So is the depth image, whose colormapped cloud the
    // RGB one (`cloud/<band>`, `depth_cloud`) replaces; its frustum stays.
    let scene = Spatial3DView::new("Scene")
        .with_origin("/draw/ros/level")
        .with_contents([
            "/cameras/08_overhead_depth/**",
            "- /cameras/08_overhead_depth/color/**",
            "- /cameras/08_overhead_depth/depth/**",
            "/draw/ros/**",
            "- /draw/ros/design/**",
            "- /draw/ros/status",
            "- /draw/ros/status_board/**",
            "- /draw/ros/events",
            "/world/tracking/**",
            "- /world/tracking/status/**",
            "- /world/tracking/target/status/**",
        ]);
    let eye = EyeControls3D::new()
        .with_position(DEMO_EYE)
        .with_look_target(DEMO_LOOK_AT)
        .with_eye_up([0.0, 0.0, 1.0]);
    let (x_range, y_range) = STATUS_BOARD_BOUNDS;
    let bounds = VisualBounds2D::new(Range2D {
        x_range: x_range.into(),
        y_range: y_range.into(),
    });
    let status = Spatial2DView::new("Status")
        .with_origin("/draw/ros/status_board")
        .with_contents(["$origin/**"]);
    let [width, height] = STATION_PAGE;
    let wrist_width = DEMO_WRIST_HEIGHT / DEMO_IMAGE_ASPECT;
    let portrait_row = STATION_TITLE_BAR + DEMO_WRIST_HEIGHT;
    let landscape_row = STATION_TITLE_BAR + DEMO_D555_WIDTH / DEMO_IMAGE_ASPECT;
    let scene_row = height - 3.0 * STATION_GAP - DEMO_STATUS_ROW - portrait_row - landscape_row;
    // Fitted to the design's frame, not the whole print, so the design sits centred.
    let mut portrait = vec![
        Spatial2DView::new("Design")
            .with_origin("/draw/ros/design")
            .with_contents(["$origin/**", "- $origin/page/**"])
            .into(),
    ];
    let wrists = DEMO_WRISTS.len() as f32;
    let mut shares = vec![width - wrists * (STATION_GAP + wrist_width)];
    for (title, sensor) in DEMO_WRISTS {
        let (x_range, y_range) = upright_bounds(sensor, DEMO_WRIST_FRAME);
        let bounds = VisualBounds2D::new(Range2D {
            x_range: x_range.into(),
            y_range: y_range.into(),
        });
        // Origin `/` keeps the wrist camera's static display rotation.
        let image = camera_entity(sensor).1;
        let view = Spatial2DView::new(title).with_contents([format!("/{image}/**")]);
        portrait.push(with_property(view, "VisualBounds2D", &bounds, properties));
        shares.push(wrist_width);
    }
    let portrait = Horizontal::new(portrait).with_column_shares(shares);
    // CPU beside TEMP, each the full height of the D555's row: one legend
    // (the sources keep one colour in both) fits its four rows only there,
    // above CPU lines held to a fixed 0-100 scale, and both show the last
    // DEMO_WINDOW_S up to the time cursor.
    let window = TimeAxis::new().with_view_range(TimeRange(TimeRangeData {
        start: TimeRangeBoundary::CursorRelative(TimeInt(-DEMO_WINDOW_S * 1_000_000_000)),
        end: TimeRangeBoundary::AT_CURSOR,
    }));
    let pane = |title: &str, name: &str, legend: bool, properties: &mut Vec<ViewProperty>| {
        let series = DEMO_HOSTS.map(|host| format!("/system/{host}/{name}"));
        let view = TimeSeriesView::new(title).with_contents(series);
        let key = PlotLegend::new()
            .with_corner(Corner2D::LeftTop)
            .with_visible(legend);
        let view = with_property(view, "PlotLegend", &key, properties);
        with_property(view, "TimeAxis", &window, properties)
    };
    let [low, high] = DEMO_CPU_RANGE;
    let cpu_axis = ScalarAxis::new().with_range(rerun::components::Range1D::new(low, high));
    let cpu = pane("CPU %", "cpu_percent", true, properties);
    let system = Horizontal::new(vec![
        with_property(cpu, "ScalarAxis", &cpu_axis, properties),
        pane("TEMP", "temp_c", false, properties),
    ]);
    let landscape = Horizontal::new(vec![
        Spatial2DView::new("D555 RGB")
            .with_origin("/cameras/08_overhead_depth/color")
            .with_contents(["$origin/**"])
            .into(),
        system.into(),
    ])
    .with_column_shares([DEMO_D555_WIDTH, width - STATION_GAP - DEMO_D555_WIDTH]);
    Vertical::new(vec![
        with_property(status, "VisualBounds2D", &bounds, properties),
        with_property(scene, "EyeControls3D", &eye, properties),
        portrait.into(),
        landscape.into(),
    ])
    .with_name("Demo")
    .with_row_shares([DEMO_STATUS_ROW, scene_row, portrait_row, landscape_row])
    .into()
}

/// What makes a received blueprint the active and default one.
type Activation = re_log_types::BlueprintActivationCommand;

/// The fleet's one blueprint: the layout, and the view properties logged
/// beside it into the same blueprint store.
pub struct FleetBlueprint {
    blueprint: rerun::blueprint::Blueprint,
    properties: Vec<ViewProperty>,
}

impl FleetBlueprint {
    /// What `Blueprint::send` sends, with the property rows ahead of the
    /// activation: a viewer activates a blueprint once, as received by
    /// then. `Blueprint::to_log_msgs` is crate-private, so its messages are
    /// taken from a send into memory.
    pub fn messages(&self) -> Result<(Vec<rerun::log::LogMsg>, Activation)> {
        use rerun::log::LogMsg;
        let (capture, storage) = rerun::RecordingStreamBuilder::new(APP_ID).memory()?;
        self.blueprint
            .send(&capture, rerun::blueprint::BlueprintActivation::default())?;
        let mut messages = storage.take();
        let activation = messages
            .iter()
            .find_map(|message| match message {
                LogMsg::BlueprintActivationCommand(command) => Some(command.clone()),
                _ => None,
            })
            .context("the blueprint send carried no activation")?;
        // The capture stream's own store info and properties are not the blueprint's.
        messages.retain(|message| {
            !matches!(message, LogMsg::BlueprintActivationCommand(_))
                && *message.store_id() == activation.blueprint_id
        });
        let (rows, storage) = rerun::RecordingStreamBuilder::new(APP_ID)
            .recording_id(activation.blueprint_id.recording_id().clone())
            .blueprint()
            .memory()?;
        rows.set_time_sequence("blueprint", 0);
        for (path, batches) in &self.properties {
            rows.log_serialized_batches(path.clone(), false, batches.iter().cloned())?;
        }
        messages.extend(
            storage
                .take()
                .into_iter()
                .filter(|message| matches!(message, LogMsg::ArrowMsg(..))),
        );
        Ok((messages, activation))
    }

    /// Install it through `recording`, the way a builder's `with_blueprint` does.
    pub fn send(&self, recording: &rerun::RecordingStream) -> Result<()> {
        let (messages, activation) = self.messages()?;
        recording.send_blueprint(messages, activation);
        Ok(())
    }

    /// Write it alone to `path` (an `.rbl` to open beside a recorded `.rrd`).
    pub fn save(&self, path: &Path) -> Result<()> {
        let (messages, activation) = self.messages()?;
        let file = rerun::sink::FileSink::new(path)
            .with_context(|| format!("creating blueprint file {}", path.display()))?;
        for message in messages {
            file.send(message);
        }
        file.send(activation.into());
        file.flush_blocking(std::time::Duration::from_secs(10))
            .with_context(|| format!("writing blueprint file {}", path.display()))?;
        Ok(())
    }
}

pub fn viewer_blueprint() -> FleetBlueprint {
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
    // An image under a Pinhole is shown in its own pixels only from a view
    // rooted at that pinhole; from `/` it would be hung in 3D at its image
    // plane and drawn in metres.
    fn pinhole_view(name: &str, entity: &str) -> rerun::blueprint::ContainerLike {
        Spatial2DView::new(name)
            .with_origin(entity)
            .with_contents([entity])
            .into()
    }
    // The overhead colour with the fiducials the trackers read on it.
    let overhead_rgb = || -> rerun::blueprint::ContainerLike {
        let entity = "/cameras/08_overhead_depth/color";
        Spatial2DView::new("Overhead RGB")
            .with_origin(entity)
            .with_contents([entity.to_owned(), format!("{entity}/{OVERHEAD_FIDUCIALS}/**")])
            .into()
    };
    // The bridge's markdown status, beside the Demo tab's board.
    let session: rerun::blueprint::ContainerLike = Vertical::new(vec![
        TextDocumentView::new("Status")
            .with_contents(["/draw/ros/status"])
            .into(),
        Spatial3DView::new("Robot + calibrated scene + planned path")
            .with_contents([
                "/reconstruction/**",
                "/robot/**",
                "/world/**",
                "/draw/**",
                "- /draw/preview/**",
                "- /draw/path",
                "- /draw/ros/status_board/**",
                // The live stencil outlines are under /world/tracking/target.
                // Keep them visible beside the calibrated robot.
                "- /world/tracking/status/**",
            ])
            .into(),
    ])
    .with_name("Session")
    .with_row_shares([1.0, 8.0])
    .into();

    // Each wrist camera, by device number, beside its role's dataset views
    // (vision.toml: RealSense 1 is wrist_lower, RealSense 2 wrist_upper).
    let telemetry: rerun::blueprint::ContainerLike = Grid::new(vec![
        view(
            "D405-1 RGB",
            [
                "cameras/06_realsense1/color",
                "observation.wrist_lower/**",
                "observation.images.wrist_lower/**",
            ],
        ),
        view(
            "D405-1 depth",
            [
                "cameras/06_realsense1/depth",
                "observation.images.wrist_lower_depth/**",
            ],
        ),
        view(
            "D405-2 RGB",
            [
                "cameras/07_realsense2/color",
                "observation.wrist_upper/**",
                "observation.images.wrist_upper/**",
            ],
        ),
        view(
            "D405-2 depth",
            [
                "cameras/07_realsense2/depth",
                "observation.images.wrist_upper_depth/**",
            ],
        ),
        overhead_rgb(),
        pinhole_view("Overhead depth", "/cameras/08_overhead_depth/depth"),
        TimeSeriesView::new("Tracking error")
            .with_contents(["/teleop/follower/tracking_error/**"])
            .into(),
    ])
    .with_name("Telemetry")
    .with_grid_columns(4)
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
        Grid::new(vec![
            overhead_rgb(),
            pinhole_view("Overhead depth", "/cameras/08_overhead_depth/depth"),
            view("D405-1 RGB", ["cameras/06_realsense1/color"]),
            view("D405-2 RGB", ["cameras/07_realsense2/color"]),
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

    let mut properties = Vec::new();
    let root = Tabs::new(vec![demo_tab(&mut properties), session, telemetry, calibration]);
    FleetBlueprint {
        blueprint: finish(Blueprint::new(root)),
        properties,
    }
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
        let registration = TrackingPlacement::RobotRoot {
            calibration_id: "calib".into(),
            root_from_world: world.inverse(),
        };
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
        // Camera world draws the published pose as is, whatever bundle it names.
        let camera = registered_tracking_pose(&wrong, Some(&TrackingPlacement::CameraWorld)).unwrap();
        assert_eq!(camera.translation, [0.8, 2.1, 3.3]);
        assert_eq!(camera.rotation, [[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]]);
    }

    #[test]
    fn the_overhead_fiducials_are_the_trackers_own_reads_in_the_logged_pixels() {
        // trackd's estimate as it rides the bus (2026-10-22: tags 2, 3 and 4
        // of the cube in the overhead colour, all used), here one read beside.
        let tracking = serde_json::json!({
            "target": "wrist", "tag_ids": [2], "cameras": ["overhead_depth_color"],
            "estimate": {"detections": {
                "overhead_depth_color": [
                    {"tag_id": 2, "corners_px": [[953.9, 297.7], [918.1, 287.5], [905.8, 223.1], [941.7, 236.8]]},
                    {"tag_id": 4, "corners_px": [[10.0, 20.0], [30.0, 20.0], [30.0, 40.0], [10.0, 40.0]]}],
                "camera1": [{"tag_id": 2, "corners_px": [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]]}]}}
        });
        let quads = ee_tag_quads(&tracking, [0.5, 0.5]);
        assert_eq!(quads.len(), 2, "the overhead colour's reads only");
        assert_eq!(quads[0].corners[0], [477.2, 149.1]);   // (953.9 + 0.5) / 2: Rerun's pixel centres at halves
        assert_eq!((quads[0].color, quads[1].color), (EE_TAG_USED, EE_TAG_READ));
        assert_eq!(quads.iter().map(|q| q.label.clone().unwrap()).collect::<Vec<_>>(), ["2", "4"]);
        let mut elsewhere = tracking.clone();
        elsewhere["cameras"] = serde_json::json!(["camera1"]);   // solved from another camera
        assert!(ee_tag_quads(&elsewhere, [1.0, 1.0]).iter().all(|q| q.color == EE_TAG_READ));

        let print = serde_json::json!({
            "source": "measured",
            "support": {"anchor_camera": "overhead_depth_color", "markers": {"ids": [7, 9]},
                        "anchor_pixels": [[100.0, 100.0], [116.0, 100.0], [116.0, 116.0], [100.0, 116.0],
                                          [130.0, 100.0], [146.0, 100.0], [146.0, 116.0], [130.0, 116.0]]}
        });
        let markers = print_marker_quads(&print, [0.5, 0.5]);
        assert_eq!(markers.len(), 2);
        assert_eq!(markers[1].corners[2], [73.25, 58.25]);
        assert!(markers.iter().all(|q| q.color == PRINT_MARKER && q.label.is_none()));
        for broken in [
            serde_json::json!({"source": "lost", "support": print["support"].clone()}),
            {
                let mut other = print.clone();
                other["support"]["anchor_camera"] = "camera1".into();
                other
            },
            {
                let mut short = print.clone();
                short["support"]["anchor_pixels"] = serde_json::json!([[1.0, 2.0]]);
                short
            },
            serde_json::json!({"source": "measured", "support": {"anchor_camera": "overhead_depth_color"}}),
        ] {
            assert!(print_marker_quads(&broken, [1.0, 1.0]).is_empty(), "{broken}");
        }
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
    fn camera_world_overlays_sit_at_identity_without_a_registration() {
        let pattern = format!("stencil-{}", "b".repeat(64));
        let corners = [[0.05, 0.1, 0.7], [0.15, 0.1, 0.7], [0.15, 0.3, 0.7], [0.05, 0.3, 0.7]];
        // stencild's target frame: y and z flipped from the page (paper faces the camera).
        let target = serde_json::json!({
            "target_id": pattern, "target_frame": "world", "source": "measured",
            "calibration_id": "any-bundle",
            "world_from_target": [[1, 0, 0, 0.1], [0, -1, 0, 0.2], [0, 0, -1, 0.7], [0, 0, 0, 1]],
            "support": {"corners_m": corners, "motion_authority": false}
        });
        let wrist = serde_json::json!({
            "target": "wrist", "source": "measured", "calibration_id": "any-bundle",
            "pose": [[1, 0, 0, 0.3], [0, 1, 0, 0.1], [0, 0, 1, 0.6], [0, 0, 0, 1]]
        });
        let (rec, storage) = rerun::RecordingStreamBuilder::new(APP_ID).memory().unwrap();
        let viewer = RerunViewer::from_recording(rec);
        viewer.place_tracking_in_camera_world();
        viewer.log_target(1, &target).unwrap();
        viewer.log_tracking(2, &wrist).unwrap();
        let mut strips = BTreeMap::<String, Vec<Vec<[f32; 3]>>>::new();
        for msg in storage.take() {
            if let rerun::log::LogMsg::ArrowMsg(_, msg) = msg {
                let chunk = rerun::log::Chunk::from_arrow_msg(&msg).unwrap();
                for row in chunk.iter_component::<rerun::components::LineStrip3D>(
                    rerun::LineStrips3D::descriptor_strips().component,
                ) {
                    strips
                        .entry(chunk.entity_path().to_string())
                        .or_default()
                        .extend(row.iter().map(|s| s.0.iter().map(|p| p.0).collect()));
                }
            }
        }
        let outline = &strips[&format!("/world/tracking/target/{pattern}")];
        let mut closed: Vec<[f32; 3]> = corners.iter().map(|c| c.map(|v| v as f32)).collect();
        closed.push(closed[0]);
        assert_eq!(outline, &vec![closed], "the published corners, untouched");
        let axes = &strips["/world/tracking/wrist"];
        assert_eq!(axes.len(), 3);
        assert!(axes.iter().all(|axis| axis[0] == [0.3, 0.1, 0.6]));
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
    fn shared_blueprint_builds_with_its_rows_ahead_of_one_activation() {
        let (messages, activation) = viewer_blueprint().messages().unwrap();
        assert!(activation.make_active && activation.make_default);
        // One blueprint store: no capture stream's info or properties leak in.
        let one_store = |m: &rerun::log::LogMsg| *m.store_id() == activation.blueprint_id;
        assert!(messages.iter().all(one_store));
        let activates = |m: &rerun::log::LogMsg| {
            matches!(m, rerun::log::LogMsg::BlueprintActivationCommand(_))
        };
        assert!(!messages.iter().any(activates), "the activation goes last, alone");
        let entities: Vec<String> = messages
            .iter()
            .filter_map(|m| match m {
                rerun::log::LogMsg::ArrowMsg(_, msg) => Some(
                    rerun::log::Chunk::from_arrow_msg(msg)
                        .unwrap()
                        .entity_path()
                        .to_string(),
                ),
                _ => None,
            })
            .collect();
        // One eye (Scene); bounds on the Status board and each wrist pane.
        for (property, count) in [("/EyeControls3D", 1), ("/VisualBounds2D", 3)] {
            assert_eq!(entities.iter().filter(|e| e.ends_with(property)).count(), count);
        }
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

    /// One container or view of a decoded blueprint store.
    #[derive(Debug, Default)]
    struct BlueprintNode {
        name: Option<String>,
        origin: Option<String>,
        children: Vec<String>,
        query: Vec<String>,
        /// A 3D view's eye: position, look target, up.
        eye: Vec<[f32; 3]>,
        /// A 2D view's visual bounds: x range, then y range.
        bounds: Vec<[f64; 2]>,
        /// A container's row and column shares.
        rows: Vec<f32>,
        columns: Vec<f32>,
        /// A time series view's legend (corner, shown), y range and time
        /// window (start, end, nanoseconds from the cursor).
        legend: Option<(String, bool)>,
        y_range: Vec<[f64; 2]>,
        window: Vec<[i64; 2]>,
    }

    /// Read a view property row (`<view>/<one of PROPERTIES>`) into its view's node.
    fn read_property(node: &mut BlueprintNode, property: &str, chunk: &rerun::log::Chunk) {
        use rerun::external::re_sdk_types::blueprint::archetypes as bp;
        if property == "EyeControls3D" {
            for descriptor in [
                bp::EyeControls3D::descriptor_position(),
                bp::EyeControls3D::descriptor_look_target(),
            ] {
                for value in
                    chunk.iter_component::<rerun::components::Position3D>(descriptor.component)
                {
                    node.eye.extend(value.iter().map(|p| p.0.0));
                }
            }
            for up in chunk.iter_component::<rerun::components::Vector3D>(
                bp::EyeControls3D::descriptor_eye_up().component,
            ) {
                node.eye.extend(up.iter().map(|v| v.0.0));
            }
        } else if property == "VisualBounds2D" {
            for range in chunk.iter_component::<rerun::blueprint::components::VisualBounds2D>(
                bp::VisualBounds2D::descriptor_range().component,
            ) {
                node.bounds
                    .extend(range.iter().flat_map(|r| [r.0.x_range.0, r.0.y_range.0]));
            }
        } else if property == "PlotLegend" {
            let corner = chunk
                .iter_component::<rerun::blueprint::components::Corner2D>(
                    bp::PlotLegend::descriptor_corner().component,
                )
                .flat_map(|c| c.iter().map(|c| c.to_string()).collect::<Vec<_>>())
                .next();
            let visible = chunk
                .iter_component::<rerun::components::Visible>(
                    bp::PlotLegend::descriptor_visible().component,
                )
                .flat_map(|v| v.iter().map(|v| v.0.0).collect::<Vec<_>>())
                .next();
            node.legend = corner.zip(visible);
        } else if property == "ScalarAxis" {
            for range in chunk.iter_component::<rerun::components::Range1D>(
                bp::ScalarAxis::descriptor_range().component,
            ) {
                node.y_range.extend(range.iter().map(|r| r.0.0));
            }
        } else if property == "TimeAxis" {
            use rerun::external::re_sdk_types::datatypes::TimeRangeBoundary as Boundary;
            let relative = |b: &Boundary| match b {
                Boundary::CursorRelative(t) => t.0,
                other => panic!("not cursor-relative: {other:?}"),
            };
            for range in chunk.iter_component::<rerun::blueprint::components::TimeRange>(
                bp::TimeAxis::descriptor_view_range().component,
            ) {
                node.window.extend(
                    range
                        .iter()
                        .map(|r| [relative(&r.0.start), relative(&r.0.end)]),
                );
            }
        }
    }

    /// The view properties the Demo tab logs as rows of their own.
    const PROPERTIES: [&str; 5] = [
        "EyeControls3D",
        "VisualBounds2D",
        "PlotLegend",
        "ScalarAxis",
        "TimeAxis",
    ];

    /// The blueprint as the viewer receives it: saved into an `.rrd`, decoded,
    /// and read back as containers, views and their queries by blueprint path.
    fn saved_blueprint() -> (String, BTreeMap<String, BlueprintNode>) {
        use rerun::external::re_sdk_types::blueprint::archetypes as bp;
        let directory = tempfile::tempdir().unwrap();
        let path = directory.path().join("blueprint.rrd");
        RerunViewer::save(&path).unwrap().finish().unwrap();
        let reader = BufReader::new(fs::File::open(&path).unwrap());
        let mut nodes = BTreeMap::<String, BlueprintNode>::new();
        for message in re_log_encoding::DecoderApp::decode_eager(reader).unwrap() {
            let rerun::log::LogMsg::ArrowMsg(store_id, message) = message.unwrap() else {
                continue;
            };
            if !store_id.is_blueprint() {
                continue;
            }
            let chunk = rerun::log::Chunk::from_arrow_msg(&message).unwrap();
            let entity = chunk.entity_path().to_string();
            let entity = entity.trim_start_matches('/');
            if let Some((view, property)) = entity
                .rsplit_once('/')
                .filter(|(_, p)| PROPERTIES.contains(p))
            {
                read_property(nodes.entry(view.to_owned()).or_default(), property, &chunk);
                continue;
            }
            let (owner, contents) = match entity.strip_suffix("/ViewContents") {
                Some(view) => (view, true),
                None => (entity, false),
            };
            let node = nodes.entry(owner.to_owned()).or_default();
            if contents {
                for query in chunk.iter_component::<rerun::blueprint::components::QueryExpression>(
                    bp::ViewContents::descriptor_query().component,
                ) {
                    node.query = query.iter().map(|q| q.0.to_string()).collect();
                }
                continue;
            }
            for name in chunk.iter_component::<rerun::components::Name>(
                bp::ContainerBlueprint::descriptor_display_name().component,
            ) {
                node.name = name.first().map(|n| n.to_string());
            }
            for name in chunk.iter_component::<rerun::components::Name>(
                bp::ViewBlueprint::descriptor_display_name().component,
            ) {
                node.name = name.first().map(|n| n.to_string());
            }
            for origin in chunk.iter_component::<rerun::blueprint::components::ViewOrigin>(
                bp::ViewBlueprint::descriptor_space_origin().component,
            ) {
                node.origin = origin.first().map(|o| o.0.to_string());
            }
            for shares in chunk.iter_component::<rerun::blueprint::components::RowShare>(
                bp::ContainerBlueprint::descriptor_row_shares().component,
            ) {
                node.rows = shares.iter().map(|s| s.0.0).collect();
            }
            for shares in chunk.iter_component::<rerun::blueprint::components::ColumnShare>(
                bp::ContainerBlueprint::descriptor_col_shares().component,
            ) {
                node.columns = shares.iter().map(|s| s.0.0).collect();
            }
            for children in chunk.iter_component::<rerun::blueprint::components::IncludedContent>(
                bp::ContainerBlueprint::descriptor_contents().component,
            ) {
                node.children = children
                    .iter()
                    .map(|c| c.0.to_string().trim_start_matches('/').to_owned())
                    .collect();
            }
        }
        // The root is the one container no other container holds.
        let held: HashSet<&String> = nodes.values().flat_map(|n| &n.children).collect();
        let roots: Vec<&String> = nodes
            .keys()
            .filter(|path| path.starts_with("container/") && !held.contains(path))
            .collect();
        assert_eq!(roots.len(), 1, "one root container: {roots:?}");
        (roots[0].clone(), nodes)
    }

    #[test]
    fn saved_blueprint_puts_demo_first_with_the_camera_world_scene() {
        let (root, nodes) = saved_blueprint();
        let names = |path: &str| -> Vec<String> {
            nodes[path]
                .children
                .iter()
                .map(|child| nodes[child].name.clone().unwrap_or_default())
                .collect()
        };
        assert_eq!(names(&root), ["Demo", "Session", "Telemetry", "Calibration"]);
        let demo = &nodes[&nodes[&root].children[0]];
        let rows: Vec<&BlueprintNode> = demo.children.iter().map(|c| &nodes[c]).collect();
        assert_eq!(rows.len(), 4);
        // The status board, whole and fixed: its canvas is the bounds.
        assert_eq!(rows[0].name.as_deref(), Some("Status"));
        assert_eq!(rows[0].origin.as_deref(), Some("/draw/ros/status_board"));
        assert_eq!(rows[0].query, ["$origin/**"]);
        assert_eq!(rows[0].bounds, [[0.0, 860.0], [0.0, 150.0]]);
        let scene = rows[1];
        assert_eq!(scene.name.as_deref(), Some("Scene"));
        assert_eq!(scene.origin.as_deref(), Some("/draw/ros/level"));
        for query in [
            "/cameras/08_overhead_depth/**",
            "- /cameras/08_overhead_depth/color/**",
            // The RGB cloud in place of the depth image's colormapped one.
            "- /cameras/08_overhead_depth/depth/**",
            "/draw/ros/**",
            "- /draw/ros/design/**",
            "- /draw/ros/status",
            "- /draw/ros/status_board/**",
            "- /draw/ros/events",
            "/world/tracking/**",
        ] {
            assert!(scene.query.iter().any(|q| q == query), "{query} missing");
        }
        // A fixed eye in the level frame, z up: no fallback that frames the
        // whole cloud.
        assert_eq!(scene.eye, [DEMO_EYE, DEMO_LOOK_AT, [0.0, 0.0, 1.0]]);
        // The cockpit's URDF-root robot never joins the camera-world scene.
        assert!(!scene.query.iter().any(|q| q.contains("/robot")));
        // The portrait panes side by side: the design's page, then each
        // upright wrist from `/` (under its camera's display rotation),
        // framed on its turned frame: D405-1 (the lower), then D405-2.
        let portrait = &rows[2].children;
        assert_eq!(portrait.len(), 3);
        assert_eq!(nodes[&portrait[0]].name.as_deref(), Some("Design"));
        assert_eq!(nodes[&portrait[0]].origin.as_deref(), Some("/draw/ros/design"));
        assert!(nodes[&portrait[0]].query.iter().any(|q| q.starts_with('-') && q.ends_with("/page/**")));
        for (pane, (title, image, bounds)) in portrait[1..].iter().zip([
            ("D405-1", "/cameras/06_realsense1/color/**", [[0.0, 360.0], [-640.0, 0.0]]),
            ("D405-2", "/cameras/07_realsense2/color/**", [[-360.0, 0.0], [0.0, 640.0]]),
        ]) {
            assert_eq!(nodes[pane].name.as_deref(), Some(title));
            assert_eq!(nodes[pane].query, [image]);
            assert_eq!(nodes[pane].origin.as_deref().unwrap_or("/"), "/");
            assert_eq!(nodes[pane].bounds, bounds);
        }
        // The landscape D555 beside CPU over TEMP.
        let landscape = &rows[3].children;
        let d555 = &nodes[&landscape[0]];
        assert_eq!(d555.origin.as_deref(), Some("/cameras/08_overhead_depth/color"));
        assert_eq!(d555.query, ["$origin/**"]);
        // Each camera pane is its image's shape on the station: the shares
        // are its pixels, each view less its title bar.
        let split = |shares: &[f32], length: f32| -> Vec<f32> {
            let total: f32 = shares.iter().sum();
            let free = length - STATION_GAP * (shares.len() - 1) as f32;
            shares.iter().map(|share| share / total * free).collect()
        };
        let [width, height] = STATION_PAGE;
        let row = split(&demo.rows, height);
        let portrait_px = split(&rows[2].columns, width);
        let landscape_px = split(&rows[3].columns, width);
        let aspect = |w: f32, h: f32| w / (h - STATION_TITLE_BAR);
        for wrist in &portrait_px[1..] {
            let wrist = aspect(*wrist, row[2]) * DEMO_IMAGE_ASPECT;
            assert!((wrist - 1.0).abs() < 0.02, "each wrist pane is 9:16 ({wrist})");
        }
        let overhead = aspect(landscape_px[0], row[3]) / DEMO_IMAGE_ASPECT;
        assert!((overhead - 1.0).abs() < 0.02, "the D555 pane is 16:9 ({overhead})");
        assert!(aspect(portrait_px[0], row[2]) < 1.0, "the design's page is portrait");
        assert!(row[0] - STATION_TITLE_BAR >= 150.0, "the whole board");
        assert!(row[1] > row[2], "the scene is the largest pane");
        // CPU beside TEMP, the four sources in each; one legend, in CPU,
        // above its fixed 0-100 scale; both the last ten minutes.
        let system: Vec<&BlueprintNode> =
            nodes[&landscape[1]].children.iter().map(|c| &nodes[c]).collect();
        let sources = |name: &str| -> Vec<String> {
            ["estop-relay", "station", "ros", "robot"]
                .map(|host| format!("/system/{host}/{name}"))
                .to_vec()
        };
        assert_eq!(system.len(), 2);
        let (cpu, temp) = (system[0], system[1]);
        assert_eq!(cpu.name.as_deref(), Some("CPU %"));
        assert_eq!(cpu.query, sources("cpu_percent"));
        assert_eq!(cpu.legend, Some(("LeftTop".to_string(), true)));
        assert_eq!(cpu.y_range, [[-2.0, 102.0]]);
        assert_eq!(temp.name.as_deref(), Some("TEMP"));
        assert_eq!(temp.query, sources("temp_c"));
        assert_eq!(temp.legend.as_ref().map(|l| l.1), Some(false));
        assert!(temp.y_range.is_empty(), "temperatures scale to their data");
        for pane in [cpu, temp] {
            assert_eq!(pane.window, [[-600_000_000_000, 0]]);
        }
        // The markdown status moved to the Session tab.
        let session = &nodes[&nodes[&root].children[1]];
        assert_eq!(nodes[&session.children[0]].query, ["/draw/ros/status"]);
        // The other tabs keep every overhead pane rooted at its pinhole.
        let overhead: Vec<&BlueprintNode> = nodes
            .values()
            .filter(|n| n.name.as_deref().is_some_and(|n| n.starts_with("Overhead")))
            .collect();
        assert_eq!(overhead.len(), 4);
        assert!(overhead.iter().all(|n| n.origin.as_deref() == Some(n.query[0].as_str())));
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

    #[test]
    fn both_wrist_panes_show_their_labels_at_the_same_place_upright() {
        // Where a sensor point lands on its upright pane, as fractions of the
        // pane from its left and top.
        let shown = |sensor: &str, at: [f32; 2]| {
            let (sin, cos) = f64::from(display_rotation_deg(sensor).unwrap()).to_radians().sin_cos();
            let ([x0, x1], [y0, y1]) = upright_bounds(sensor, DEMO_WRIST_FRAME);
            let (x, y) = (f64::from(at[0]), f64::from(at[1]));
            [((x * cos - y * sin) - x0) / (x1 - x0), ((x * sin + y * cos) - y0) / (y1 - y0)]
        };
        let rotation = |sensor: &str| entity_rotation(&camera_entity(sensor).1);
        assert_eq!(rotation("realsense1_color"), Some(-90.0));
        assert_eq!(rotation("realsense2_color"), Some(90.0));
        assert_eq!(entity_rotation("/cameras/08_overhead_depth/color"), None);
        for at in [[640.0 * 0.12, 360.0 * 0.08], [640.0 * 0.18, 360.0 * 0.5]] {
            let upper = shown("realsense2_color", label_point(rotation("realsense2_color"), [640, 360], at));
            let lower = shown("realsense1_color", label_point(rotation("realsense1_color"), [640, 360], at));
            assert!(upper.iter().zip(lower).all(|(a, b)| (a - b).abs() < 1e-6), "{upper:?} {lower:?}");
            assert!(upper[1] < 0.25, "near the top: {upper:?}");
        }
        assert_eq!(label_point(None, [640, 360], [1.0, 2.0]), [1.0, 2.0]);
    }

    #[test]
    fn each_wrist_pane_frames_its_frame_turned_upright() {
        assert_eq!(
            upright_bounds("realsense2_color", DEMO_WRIST_FRAME),
            ([-360.0, 0.0], [0.0, 640.0])
        );
        assert_eq!(
            upright_bounds("realsense1_color", DEMO_WRIST_FRAME),
            ([0.0, 360.0], [-640.0, 0.0])
        );
        assert_eq!(
            upright_bounds("overhead_depth_color", DEMO_WRIST_FRAME),
            ([0.0, 640.0], [0.0, 360.0])
        );
    }

    #[test]
    fn an_awaited_wrist_image_is_turned_upright_and_blanked_before_its_first_frame() {
        for (_, sensor) in DEMO_WRISTS {
            awaited_wrist_image_is_turned_upright_and_blanked(sensor);
        }
    }

    fn awaited_wrist_image_is_turned_upright_and_blanked(sensor: &str) {
        let (camera, image) = camera_entity(sensor);
        let (rec, storage) = rerun::RecordingStreamBuilder::new(APP_ID).memory().unwrap();
        let viewer = RerunViewer::from_recording(rec);
        viewer.await_image(sensor, "waiting", [640, 360], 500_000_000).unwrap();
        viewer.log_ages(1_000_000_000).unwrap();
        viewer.finish().unwrap();
        let mut turned = 0;
        let mut waiting = false;
        let (mut cleared, mut frames) = (Vec::new(), Vec::new());
        for msg in storage.take() {
            if let rerun::log::LogMsg::ArrowMsg(_, msg) = msg {
                let chunk = rerun::log::Chunk::from_arrow_msg(&msg).unwrap();
                let entity = chunk.entity_path().to_string();
                if entity == format!("/{camera}") && chunk.is_static() {
                    turned += chunk.num_rows();
                }
                waiting |= entity == format!("/{image}/waiting");
                let clear = rerun::Clear::descriptor_is_recursive().component;
                if chunk.component_descriptors().any(|d| d.component == clear) {
                    cleared.push(entity.clone());
                }
                let blob = rerun::EncodedImage::descriptor_blob().component;
                for blobs in chunk.iter_component::<rerun::components::Blob>(blob) {
                    frames.extend(blobs.iter().map(|b| (entity.clone(), b.0.to_vec())));
                }
            }
        }
        // Its pane frames the placeholder upright, as it will the image.
        assert_eq!(turned, 1, "one static display rotation on the wrist camera");
        assert!(waiting);
        // An earlier cockpit's last frame and its frozen age label are not
        // drawn under "no frame yet": the label is cleared, the frame covered
        // in black at the size the frames to come are logged at.
        cleared.sort();
        assert_eq!(
            cleared,
            [format!("/{image}/age"), format!("/{image}/markers")],
            "nor an earlier cockpit's last wrist reading's markers"
        );
        assert_eq!(frames.len(), 1);
        assert_eq!(frames[0].0, format!("/{image}"));
        let blank = image::load_from_memory(&frames[0].1).unwrap().to_rgb8();
        assert_eq!(blank.dimensions(), (640, 360));
        assert!(blank.pixels().all(|p| p.0.iter().all(|&v| v < 8)));
    }

    #[test]
    fn a_hub_that_drops_the_stream_is_lost_and_one_not_up_yet_is_not() {
        use std::io::Read;
        use std::net::TcpListener;
        use std::time::{Duration, Instant};
        let proxy = |listener: &TcpListener| {
            format!("rerun+http://{}/proxy", listener.local_addr().unwrap())
        };
        let absent = proxy(&TcpListener::bind("127.0.0.1:0").unwrap());
        let waiting = RerunViewer::connect_or_save(Some(&absent), None, None).unwrap();
        // A stand-in hub: it takes the stream's request, then goes away.
        let hub = TcpListener::bind("127.0.0.1:0").unwrap();
        let viewer = RerunViewer::connect_or_save(Some(&proxy(&hub)), None, None).unwrap();
        viewer.log_status("before the hub goes").unwrap();
        hub.set_nonblocking(true).unwrap();
        let deadline = Instant::now() + Duration::from_secs(10);
        let mut socket = loop {
            match hub.accept() {
                Ok((socket, _)) => break socket,
                Err(_) if Instant::now() < deadline => {
                    std::thread::sleep(Duration::from_millis(5));
                }
                Err(error) => panic!("the viewer never dialled the hub: {error}"),
            }
        };
        socket.set_nonblocking(false).unwrap();
        socket.set_read_timeout(Some(Duration::from_secs(10))).unwrap();
        // HTTP/2: a 24-byte preface, then 9-byte frame headers; a HEADERS
        // frame (type 1) is the stream's request, sent once it is connected.
        let (mut bytes, mut at) = (Vec::new(), 24);
        while !(bytes.len() >= at + 9 && bytes[at + 3] == 1) {
            if bytes.len() >= at + 9 {
                let length = bytes[at..at + 3].iter().fold(0, |n, &b| n << 8 | usize::from(b));
                at += 9 + length;
                continue;
            }
            let mut chunk = [0; 4096];
            let read = socket.read(&mut chunk).expect("the viewer sent its request");
            assert!(read > 0, "the viewer closed the stream first");
            bytes.extend_from_slice(&chunk[..read]);
        }
        assert!(!viewer.hub_lost(), "a connected stream is not lost");
        drop((socket, hub));
        while !viewer.hub_lost() && Instant::now() < deadline {
            std::thread::sleep(Duration::from_millis(5));
        }
        assert!(viewer.hub_lost(), "the dropped stream reads as lost");
        assert!(!waiting.hub_lost(), "a hub not up yet is still being dialled");
        let dir = tempfile::tempdir().unwrap();
        let file = RerunViewer::connect_or_save(None, Some(&dir.path().join("a.rrd")), None);
        assert!(!file.unwrap().hub_lost(), "a file alone has no hub to lose");
    }
}
