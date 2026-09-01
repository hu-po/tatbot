//! AprilTag detection and vision-only rigid EE tracking.
//!
//! This feature is deliberately independent of arm kinematics. The only
//! inputs are synchronized decoded frames, a camera calibration bundle, the
//! language-neutral fiducial inventory, and a calibrated wrist layout.

use std::{
    collections::{BTreeMap, BTreeSet},
    fs,
    path::{Path, PathBuf},
    time::Instant,
};

use anyhow::{Context, Result};
use apriltag::{Detector, Family, Image, pose::TagParams};
use nalgebra::{
    DMatrix, DVector, Isometry3, Matrix3, Matrix4, Point3, SMatrix, Translation3, UnitQuaternion,
    Vector2, Vector3,
};
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

use crate::{
    CalibrationBundle, CameraCalibration, PixelFormat, RecordedPayload, SynchronizedFrameSet,
};

const SUPPORTED_INVENTORY_SCHEMA: u32 = 1;
const SUPPORTED_LAYOUT_SCHEMA: u32 = 2;
const FAMILY: &str = "apriltag_16h5";
const CORNER_SIGNS: [[f64; 2]; 4] = [[-1.0, 1.0], [1.0, 1.0], [1.0, -1.0], [-1.0, -1.0]];

#[derive(Debug, Clone, Deserialize)]
pub struct DetectorProfile {
    #[serde(default = "one")]
    pub scale: f64,
    #[serde(default = "default_min_side")]
    pub min_side_px: f64,
    #[serde(default = "enabled")]
    pub corner_refinement: bool,
    #[serde(default = "one")]
    pub quad_decimate: f64,
}

fn one() -> f64 {
    1.0
}

fn default_min_side() -> f64 {
    12.0
}

fn enabled() -> bool {
    true
}

#[derive(Debug, Clone, Deserialize)]
pub struct TargetSpec {
    pub role: String,
    pub ids: Vec<usize>,
    pub edge_m: f64,
    pub layout: Option<PathBuf>,
    pub parent_frame: Option<String>,
    pub minimum_acquisition_ids: Option<usize>,
    pub ambiguity_group: Option<String>,
    pub root_id: Option<usize>,
    pub calibration_root_id: Option<usize>,
    pub grid: Option<Vec<Vec<usize>>>,
    pub minimum_calibration_observations: Option<usize>,
    pub minimum_calibration_poses_per_id: Option<usize>,
    pub max_calibration_corner_px: Option<f64>,
    pub max_calibration_residual_mm: Option<f64>,
    pub max_calibration_parent_distance_mm: Option<f64>,
    pub max_calibration_reprojection_px: Option<f64>,
    pub max_calibration_consensus_mm: Option<f64>,
    pub max_calibration_regression_mm: Option<f64>,
}

#[derive(Debug, Clone, Deserialize)]
struct PrintingConfig {
    #[serde(default)]
    spare_ids: Vec<usize>,
}

#[derive(Debug, Clone, Deserialize)]
struct InventoryFile {
    schema_version: u32,
    family: String,
    #[serde(default)]
    detector: BTreeMap<String, DetectorProfile>,
    targets: BTreeMap<String, TargetSpec>,
    #[serde(default)]
    printing: Option<PrintingConfig>,
}

#[derive(Debug, Clone)]
pub struct FiducialInventory {
    pub family: String,
    pub detector: BTreeMap<String, DetectorProfile>,
    pub targets: BTreeMap<String, TargetSpec>,
    pub spare_ids: Vec<usize>,
    pub inventory_hash: String,
    pub source: PathBuf,
}

impl FiducialInventory {
    pub fn load(path: impl AsRef<Path>) -> Result<Self> {
        let path = path.as_ref();
        let bytes = fs::read(path).with_context(|| format!("reading {}", path.display()))?;
        let raw: InventoryFile = serde_json::from_slice(&bytes)
            .with_context(|| format!("parsing {}", path.display()))?;
        if raw.schema_version != SUPPORTED_INVENTORY_SCHEMA {
            anyhow::bail!(
                "unsupported fiducial inventory schema {}",
                raw.schema_version
            );
        }
        if raw.family != FAMILY {
            anyhow::bail!("unsupported fiducial family {}", raw.family);
        }
        if raw.targets.is_empty() {
            anyhow::bail!("fiducial inventory has no targets");
        }
        for name in ["calibration", "live"] {
            let profile = raw
                .detector
                .get(name)
                .with_context(|| format!("fiducial inventory has no detector profile {name}"))?;
            if !profile.scale.is_finite()
                || !(0.0 < profile.scale && profile.scale <= 1.0)
                || !profile.min_side_px.is_finite()
                || profile.min_side_px <= 0.0
                || !profile.quad_decimate.is_finite()
                || profile.quad_decimate <= 0.0
            {
                anyhow::bail!("invalid fiducial detector profile {name}");
            }
        }
        let mut owners = BTreeMap::<usize, Vec<(&str, Option<&str>)>>::new();
        for (name, target) in &raw.targets {
            if target.ids.is_empty()
                || target.edge_m <= 0.0
                || !target.edge_m.is_finite()
                || target.ids.iter().collect::<BTreeSet<_>>().len() != target.ids.len()
                || target
                    .parent_frame
                    .as_ref()
                    .is_some_and(|frame| frame.trim().is_empty())
            {
                anyhow::bail!("invalid fiducial target {name}");
            }
            if target.role == "rigid_ee" && target.parent_frame.is_none() {
                anyhow::bail!("fiducial target {name} has no parent_frame");
            }
            if target
                .minimum_acquisition_ids
                .is_some_and(|count| count == 0 || count > target.ids.len())
            {
                anyhow::bail!("invalid minimum acquisition count for {name}");
            }
            if target.root_id.is_some_and(|id| !target.ids.contains(&id)) {
                anyhow::bail!("target {name} root_id is not one of its ids");
            }
            if target
                .calibration_root_id
                .is_some_and(|id| !target.ids.contains(&id))
            {
                anyhow::bail!("target {name} calibration_root_id is not one of its ids");
            }
            if let Some(grid) = &target.grid {
                let width = grid.first().map(Vec::len).unwrap_or_default();
                let flattened: Vec<_> = grid.iter().flatten().copied().collect();
                if width == 0
                    || grid.iter().any(|row| row.len() != width)
                    || flattened.iter().collect::<BTreeSet<_>>().len() != flattened.len()
                    || flattened.iter().copied().collect::<BTreeSet<_>>()
                        != target.ids.iter().copied().collect::<BTreeSet<_>>()
                {
                    anyhow::bail!("target {name} grid must contain each id once in a rectangle");
                }
            }
            if target
                .minimum_calibration_observations
                .is_some_and(|count| count < 4)
                || target
                    .minimum_calibration_poses_per_id
                    .is_some_and(|count| count < 2)
                || target
                    .max_calibration_corner_px
                    .is_some_and(|value| !value.is_finite() || value <= 0.0)
                || target
                    .max_calibration_residual_mm
                    .is_some_and(|value| !value.is_finite() || value <= 0.0)
                || target
                    .max_calibration_parent_distance_mm
                    .is_some_and(|value| !value.is_finite() || value <= 0.0)
                || target
                    .max_calibration_reprojection_px
                    .is_some_and(|value| !value.is_finite() || value <= 0.0)
                || target
                    .max_calibration_consensus_mm
                    .is_some_and(|value| !value.is_finite() || value <= 0.0)
                || target
                    .max_calibration_regression_mm
                    .is_some_and(|value| !value.is_finite() || value <= 0.0)
            {
                anyhow::bail!("invalid calibration quality gate for {name}");
            }
            for id in &target.ids {
                owners
                    .entry(*id)
                    .or_default()
                    .push((name, target.ambiguity_group.as_deref()));
            }
        }
        for (id, matches) in &owners {
            if matches.len() < 2 {
                continue;
            }
            let groups: BTreeSet<_> = matches.iter().map(|(_, group)| *group).collect();
            if groups.len() != 1 || groups.contains(&None) {
                anyhow::bail!("id {id} is duplicated without one ambiguity_group");
            }
        }
        for (name, target) in &raw.targets {
            if let Some(id) = target.calibration_root_id
                && owners.get(&id).is_some_and(|matches| matches.len() != 1)
            {
                anyhow::bail!(
                    "target {name} calibration_root_id must identify one physical instance"
                );
            }
        }
        let spare_ids = raw
            .printing
            .map(|value| value.spare_ids)
            .unwrap_or_default();
        if spare_ids.iter().any(|id| owners.contains_key(id))
            || spare_ids.iter().collect::<BTreeSet<_>>().len() != spare_ids.len()
        {
            anyhow::bail!("spare fiducial ids overlap mounted targets or each other");
        }
        Ok(Self {
            family: raw.family,
            detector: raw.detector,
            targets: raw.targets,
            spare_ids,
            inventory_hash: hex::encode(Sha256::digest(&bytes)),
            source: path.to_path_buf(),
        })
    }

    pub fn target(&self, name: &str) -> Result<&TargetSpec> {
        self.targets
            .get(name)
            .with_context(|| format!("{} has no target {name}", self.source.display()))
    }

    pub fn known_ids(&self) -> BTreeSet<usize> {
        self.targets
            .values()
            .flat_map(|target| target.ids.iter().copied())
            .collect()
    }
}

#[derive(Debug, Clone, Deserialize)]
struct LayoutEntry {
    ee_from_tag: [[f64; 4]; 4],
}

#[derive(Debug, Clone, Deserialize)]
struct LayoutFile {
    schema_version: u32,
    calibration_status: String,
    inventory_hash: String,
    target_ids: Vec<usize>,
    edge_m: f64,
    parent_frame: String,
    tags: BTreeMap<String, LayoutEntry>,
}

#[derive(Debug, Clone)]
pub struct WristLayout {
    pub calibration_status: String,
    pub edge_m: f64,
    pub ee_from_tag: BTreeMap<usize, Isometry3<f64>>,
    pub parent_frame: String,
    pub layout_hash: String,
    pub inventory_hash: String,
}

impl WristLayout {
    pub fn load(
        path: impl AsRef<Path>,
        inventory: &FiducialInventory,
        allow_pending: bool,
    ) -> Result<Self> {
        let path = path.as_ref();
        let bytes = fs::read(path).with_context(|| format!("reading {}", path.display()))?;
        let raw: LayoutFile = serde_json::from_slice(&bytes)
            .with_context(|| format!("parsing {}", path.display()))?;
        let wrist = inventory.target("wrist")?;
        if raw.schema_version != SUPPORTED_LAYOUT_SCHEMA {
            anyhow::bail!("unsupported wrist layout schema {}", raw.schema_version);
        }
        if !allow_pending && raw.calibration_status != "calibrated" {
            anyhow::bail!("wrist layout is {}, not calibrated", raw.calibration_status);
        }
        if raw.inventory_hash != inventory.inventory_hash {
            anyhow::bail!("wrist layout inventory hash is stale");
        }
        let expected_parent = wrist
            .parent_frame
            .as_deref()
            .context("wrist target has no parent_frame")?;
        if raw.parent_frame != expected_parent {
            anyhow::bail!(
                "wrist layout parent frame {} differs from inventory {}",
                raw.parent_frame,
                expected_parent
            );
        }
        if (raw.edge_m - wrist.edge_m).abs() > 1e-9 {
            anyhow::bail!("wrist layout edge differs from inventory");
        }
        let expected: BTreeSet<_> = wrist.ids.iter().copied().collect();
        if raw.target_ids != wrist.ids {
            anyhow::bail!("wrist layout ids differ from inventory");
        }
        let mut ee_from_tag = BTreeMap::new();
        for (text, entry) in raw.tags {
            let id: usize = text
                .parse()
                .with_context(|| format!("invalid tag id {text}"))?;
            ee_from_tag.insert(id, isometry_from_array(entry.ee_from_tag)?);
        }
        if ee_from_tag.keys().copied().collect::<BTreeSet<_>>() != expected {
            anyhow::bail!("wrist layout transform ids differ from inventory");
        }
        Ok(Self {
            calibration_status: raw.calibration_status,
            edge_m: raw.edge_m,
            ee_from_tag,
            parent_frame: raw.parent_frame,
            layout_hash: hex::encode(Sha256::digest(&bytes)),
            inventory_hash: raw.inventory_hash,
        })
    }

    fn corners_ee(&self, tag_id: usize) -> Result<[Point3<f64>; 4]> {
        let transform = self
            .ee_from_tag
            .get(&tag_id)
            .with_context(|| format!("wrist layout has no tag {tag_id}"))?;
        let half = self.edge_m / 2.0;
        Ok(CORNER_SIGNS
            .map(|[x, y]| transform.transform_point(&Point3::new(x * half, y * half, 0.0))))
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FiducialDetection {
    pub camera: String,
    pub tag_id: usize,
    pub corners_px: [[f64; 2]; 4],
    pub timestamp_ns: i128,
    pub side_px: f64,
    pub decision_margin: f32,
    pub hamming: usize,
    #[serde(skip)]
    camera_from_tag_candidates: Vec<Isometry3<f64>>,
}

/// Full-resolution TLBR crop used before detector scaling. Coordinates are
/// half-open and remain in the calibrated camera pixel frame.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DetectionRoi {
    pub x0: usize,
    pub y0: usize,
    pub x1: usize,
    pub y1: usize,
}

/// Bound detections in the calibrated full-resolution image, expanding and
/// clamping the half-open ROI. Keeping this next to [`DetectionRoi`] lets the
/// crop geometry be tested without a GStreamer build.
pub fn expanded_detection_roi(
    detections: &[&FiducialDetection],
    width: usize,
    height: usize,
    margin_px: usize,
) -> Option<DetectionRoi> {
    let corners = detections
        .iter()
        .flat_map(|detection| detection.corners_px)
        .filter(|corner| corner[0].is_finite() && corner[1].is_finite())
        .collect::<Vec<_>>();
    if corners.is_empty() || width == 0 || height == 0 {
        return None;
    }
    let min_x = corners
        .iter()
        .map(|corner| corner[0])
        .fold(f64::INFINITY, f64::min);
    let min_y = corners
        .iter()
        .map(|corner| corner[1])
        .fold(f64::INFINITY, f64::min);
    let max_x = corners
        .iter()
        .map(|corner| corner[0])
        .fold(f64::NEG_INFINITY, f64::max);
    let max_y = corners
        .iter()
        .map(|corner| corner[1])
        .fold(f64::NEG_INFINITY, f64::max);
    let x0 = (min_x.floor() as isize - margin_px as isize).clamp(0, width as isize - 1) as usize;
    let y0 = (min_y.floor() as isize - margin_px as isize).clamp(0, height as isize - 1) as usize;
    let x1 = (max_x.ceil() as isize + margin_px as isize + 1).clamp(1, width as isize) as usize;
    let y1 = (max_y.ceil() as isize + margin_px as isize + 1).clamp(1, height as isize) as usize;
    (x0 < x1 && y0 < y1).then_some(DetectionRoi { x0, y0, x1, y1 })
}

#[derive(Debug)]
pub struct FiducialDetectionSet {
    pub detections: Vec<FiducialDetection>,
    /// Slowest camera's BGR/RGB crop, grayscale, and scale preparation.
    pub image_prep_latency_ms: f64,
    /// Slowest camera's native AprilTag detection and pose-candidate work.
    pub apriltag_latency_ms: f64,
    pub roi_camera_count: usize,
}

#[derive(Debug)]
struct CameraDetectionSet {
    detections: Vec<FiducialDetection>,
    image_prep_latency_ms: f64,
    apriltag_latency_ms: f64,
    used_roi: bool,
}

pub struct AprilTagDetector {
    detector: Detector,
    allowed_ids: BTreeSet<usize>,
    scale: f64,
    min_side_px: f64,
    tag_edge_m: f64,
}

/// Cloneable detector settings for parallel processing of synchronized cameras.
/// Each worker owns its native detector so no C state is shared across threads.
#[derive(Debug, Clone)]
pub struct AprilTagDetectorFactory {
    allowed_ids: BTreeSet<usize>,
    scale: f64,
    min_side_px: f64,
    tag_edge_m: f64,
    corner_refinement: bool,
    quad_decimate: f64,
}

impl AprilTagDetectorFactory {
    pub fn new(
        inventory: &FiducialInventory,
        target: Option<&str>,
        scale: Option<f64>,
    ) -> Result<Self> {
        let allowed_ids = if let Some(name) = target {
            inventory.target(name)?.ids.iter().copied().collect()
        } else {
            inventory.known_ids()
        };
        let tag_edge_m = inventory.target(target.unwrap_or("wrist"))?.edge_m;
        let profile = inventory
            .detector
            .get("live")
            .cloned()
            .context("fiducial inventory has no live detector profile")?;
        let scale = scale.unwrap_or(profile.scale);
        if !(0.0..=1.0).contains(&scale) || scale == 0.0 {
            anyhow::bail!("fiducial detector scale must be in (0, 1]");
        }
        Ok(Self {
            allowed_ids,
            scale,
            min_side_px: profile.min_side_px,
            tag_edge_m,
            corner_refinement: profile.corner_refinement,
            quad_decimate: profile.quad_decimate,
        })
    }

    fn build(&self, threads: u8) -> Result<AprilTagDetector> {
        let mut detector = Detector::builder()
            .add_family_bits(Family::tag_16h5(), 0)
            .build()
            .context("creating AprilTag 16h5 detector")?;
        detector.set_thread_number(threads);
        detector.set_decimation(self.quad_decimate as f32);
        detector.set_refine_edges(self.corner_refinement);
        Ok(AprilTagDetector {
            detector,
            allowed_ids: self.allowed_ids.clone(),
            scale: self.scale,
            min_side_px: self.min_side_px,
            tag_edge_m: self.tag_edge_m,
        })
    }

    pub fn detect_set(
        &self,
        calibration: &CalibrationBundle,
        set: &SynchronizedFrameSet,
    ) -> Result<Vec<FiducialDetection>> {
        self.detect_set_excluding(calibration, set, &BTreeSet::new())
    }

    pub fn detect_set_excluding(
        &self,
        calibration: &CalibrationBundle,
        set: &SynchronizedFrameSet,
        excluded_cameras: &BTreeSet<String>,
    ) -> Result<Vec<FiducialDetection>> {
        Ok(self
            .detect_set_profiled(calibration, set, excluded_cameras, &BTreeMap::new())?
            .detections)
    }

    pub fn detect_set_profiled(
        &self,
        calibration: &CalibrationBundle,
        set: &SynchronizedFrameSet,
        excluded_cameras: &BTreeSet<String>,
        rois: &BTreeMap<String, DetectionRoi>,
    ) -> Result<FiducialDetectionSet> {
        let batches = set
            .frames
            .par_iter()
            .filter(|(name, _)| !excluded_cameras.contains(*name))
            .map(|(name, frame)| -> Result<CameraDetectionSet> {
                let camera = calibration
                    .camera(name, &frame.metadata.profile)
                    .map_err(anyhow::Error::msg)?;
                // Camera-level parallelism already occupies the cores.
                self.build(1)?
                    .detect_frame_profiled(camera, frame, rois.get(name).copied())
            })
            .collect::<Vec<_>>();
        let mut detections = Vec::new();
        let mut image_prep_latency_ms = 0.0_f64;
        let mut apriltag_latency_ms = 0.0_f64;
        let mut roi_camera_count = 0_usize;
        for batch in batches {
            let batch = batch?;
            image_prep_latency_ms = image_prep_latency_ms.max(batch.image_prep_latency_ms);
            apriltag_latency_ms = apriltag_latency_ms.max(batch.apriltag_latency_ms);
            roi_camera_count += usize::from(batch.used_roi);
            detections.extend(batch.detections);
        }
        Ok(FiducialDetectionSet {
            detections,
            image_prep_latency_ms,
            apriltag_latency_ms,
            roi_camera_count,
        })
    }
}

impl std::fmt::Debug for AprilTagDetector {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("AprilTagDetector")
            .field("allowed_ids", &self.allowed_ids)
            .field("scale", &self.scale)
            .field("min_side_px", &self.min_side_px)
            .finish()
    }
}

impl AprilTagDetector {
    pub fn new(
        inventory: &FiducialInventory,
        target: Option<&str>,
        scale: Option<f64>,
    ) -> Result<Self> {
        AprilTagDetectorFactory::new(inventory, target, scale)?.build(2)
    }

    pub fn detect_frame(
        &mut self,
        camera: &CameraCalibration,
        frame: &crate::FrameRecord,
    ) -> Result<Vec<FiducialDetection>> {
        Ok(self.detect_frame_profiled(camera, frame, None)?.detections)
    }

    fn detect_frame_profiled(
        &mut self,
        camera: &CameraCalibration,
        frame: &crate::FrameRecord,
        roi: Option<DetectionRoi>,
    ) -> Result<CameraDetectionSet> {
        let (format, width, height, bytes) = match &frame.payload {
            RecordedPayload::Video {
                format,
                width,
                height,
                bytes,
            } if matches!(
                format,
                PixelFormat::Bgr8 | PixelFormat::Rgb8 | PixelFormat::Y8
            ) =>
            {
                (*format, *width as usize, *height as usize, bytes)
            }
            _ => anyhow::bail!("fiducial detection requires decoded BGR/RGB/Y8 frames"),
        };
        let roi = roi.unwrap_or(DetectionRoi {
            x0: 0,
            y0: 0,
            x1: width,
            y1: height,
        });
        if roi.x0 >= roi.x1 || roi.y0 >= roi.y1 || roi.x1 > width || roi.y1 > height {
            anyhow::bail!(
                "invalid detector ROI [{}, {}, {}, {}] for {width}x{height}",
                roi.x0,
                roi.y0,
                roi.x1,
                roi.y1
            );
        }
        let used_roi = roi.x0 != 0 || roi.y0 != 0 || roi.x1 != width || roi.y1 != height;
        let crop_width = roi.x1 - roi.x0;
        let crop_height = roi.y1 - roi.y0;
        let prep_started = Instant::now();
        let scaled_width = ((crop_width as f64 * self.scale).round() as usize).max(1);
        let scaled_height = ((crop_height as f64 * self.scale).round() as usize).max(1);
        let mut image = Image::zeros_with_stride(scaled_width, scaled_height, scaled_width)
            .context("allocating AprilTag grayscale image")?;
        // Precompute nearest-neighbor source offsets. The former inner-loop
        // floating division dominated detector latency on five 5 MP streams.
        let channels = if format == PixelFormat::Y8 { 1 } else { 3 };
        let source_x = (0..scaled_width)
            .map(|x| {
                (roi.x0 + ((x as f64 / self.scale).floor() as usize).min(crop_width - 1)) * channels
            })
            .collect::<Vec<_>>();
        let source_y = (0..scaled_height)
            .map(|y| {
                (roi.y0 + ((y as f64 / self.scale).floor() as usize).min(crop_height - 1))
                    * width
                    * channels
            })
            .collect::<Vec<_>>();
        for (y, output_row) in image
            .as_slice_mut()
            .chunks_exact_mut(scaled_width)
            .enumerate()
        {
            let input_row = source_y[y];
            for (output, x) in output_row.iter_mut().zip(&source_x) {
                let offset = input_row + x;
                if format == PixelFormat::Y8 {
                    *output = bytes[offset];
                    continue;
                }
                let (red, green, blue) = if format == PixelFormat::Bgr8 {
                    (bytes[offset + 2], bytes[offset + 1], bytes[offset])
                } else {
                    (bytes[offset], bytes[offset + 1], bytes[offset + 2])
                };
                *output = ((77 * red as u32 + 150 * green as u32 + 29 * blue as u32) >> 8) as u8;
            }
        }
        let image_prep_latency_ms = prep_started.elapsed().as_secs_f64() * 1000.0;
        let params = TagParams {
            tagsize: self.tag_edge_m,
            fx: camera.intrinsics.fx * self.scale,
            fy: camera.intrinsics.fy * self.scale,
            cx: (camera.intrinsics.cx - roi.x0 as f64) * self.scale,
            cy: (camera.intrinsics.cy - roi.y0 as f64) * self.scale,
        };
        let timestamp_ns = frame
            .metadata
            .timestamps
            .normalized_unix_ns
            .or(frame.metadata.timestamps.source_ns)
            .unwrap_or(frame.metadata.timestamps.host_unix_ns);
        let detector_started = Instant::now();
        let mut output = Vec::new();
        for detection in self.detector.detect(&image) {
            if !self.allowed_ids.contains(&detection.id()) {
                continue;
            }
            // Normalize AprilTag C's tag-coordinate corner numbering to the
            // TL/TR/BR/BL image contract used by OpenCV and calibration. The
            // exact permutation is locked by live cross-detector parity tests.
            let raw = detection.corners();
            let mut corners_px = [raw[1], raw[0], raw[3], raw[2]];
            for corner in &mut corners_px {
                corner[0] = corner[0] / self.scale + roi.x0 as f64;
                corner[1] = corner[1] / self.scale + roi.y0 as f64;
            }
            let side_px = (0..4)
                .map(|index| {
                    let next = (index + 1) % 4;
                    ((corners_px[index][0] - corners_px[next][0]).powi(2)
                        + (corners_px[index][1] - corners_px[next][1]).powi(2))
                    .sqrt()
                })
                .sum::<f64>()
                / 4.0;
            if side_px < self.min_side_px {
                continue;
            }
            let camera_from_tag_candidates = detection
                .estimate_tag_pose_orthogonal_iteration(&params, 50)
                .into_iter()
                .filter_map(|estimate| isometry_from_apriltag_pose(&estimate.pose).ok())
                .collect();
            output.push(FiducialDetection {
                camera: camera.sensor_name.clone(),
                tag_id: detection.id(),
                corners_px,
                timestamp_ns,
                side_px,
                decision_margin: detection.decision_margin(),
                hamming: detection.hamming(),
                camera_from_tag_candidates,
            });
        }
        Ok(CameraDetectionSet {
            detections: output,
            image_prep_latency_ms,
            apriltag_latency_ms: detector_started.elapsed().as_secs_f64() * 1000.0,
            used_roi,
        })
    }
}

#[derive(Debug, Clone)]
struct CameraModel {
    calibration: CameraCalibration,
    world_from_camera: Isometry3<f64>,
}

impl CameraModel {
    fn new(calibration: &CameraCalibration) -> Result<Self> {
        Ok(Self {
            calibration: calibration.clone(),
            world_from_camera: isometry_from_pose(&calibration.world_from_camera)?,
        })
    }

    fn project(&self, world: &Point3<f64>) -> Option<Vector2<f64>> {
        let camera = self.world_from_camera.inverse_transform_point(world);
        if camera.z <= 1e-8 {
            return None;
        }
        let mut x = camera.x / camera.z;
        let mut y = camera.y / camera.z;
        let coefficients = &self.calibration.distortion.coefficients;
        let value = |index: usize| coefficients.get(index).copied().unwrap_or(0.0);
        let r2 = x * x + y * y;
        let numerator = 1.0 + value(0) * r2 + value(1) * r2.powi(2) + value(4) * r2.powi(3);
        let denominator = 1.0 + value(5) * r2 + value(6) * r2.powi(2) + value(7) * r2.powi(3);
        let radial = numerator / denominator;
        let dx = 2.0 * value(2) * x * y + value(3) * (r2 + 2.0 * x * x);
        let dy = value(2) * (r2 + 2.0 * y * y) + 2.0 * value(3) * x * y;
        x = x * radial + dx;
        y = y * radial + dy;
        Some(Vector2::new(
            self.calibration.intrinsics.fx * x + self.calibration.intrinsics.cx,
            self.calibration.intrinsics.fy * y + self.calibration.intrinsics.cy,
        ))
    }

    fn ray(&self, pixel: [f64; 2]) -> (Vector3<f64>, Vector3<f64>) {
        let mut x = (pixel[0] - self.calibration.intrinsics.cx) / self.calibration.intrinsics.fx;
        let mut y = (pixel[1] - self.calibration.intrinsics.cy) / self.calibration.intrinsics.fy;
        let distorted = (x, y);
        let coefficients = &self.calibration.distortion.coefficients;
        let value = |index: usize| coefficients.get(index).copied().unwrap_or(0.0);
        for _ in 0..8 {
            let r2 = x * x + y * y;
            let numerator = 1.0 + value(0) * r2 + value(1) * r2.powi(2) + value(4) * r2.powi(3);
            let denominator = 1.0 + value(5) * r2 + value(6) * r2.powi(2) + value(7) * r2.powi(3);
            let radial = numerator / denominator;
            let dx = 2.0 * value(2) * x * y + value(3) * (r2 + 2.0 * x * x);
            let dy = value(2) * (r2 + 2.0 * y * y) + 2.0 * value(3) * x * y;
            x = (distorted.0 - dx) / radial;
            y = (distorted.1 - dy) / radial;
        }
        let direction = self.world_from_camera.rotation * Vector3::new(x, y, 1.0).normalize();
        (self.world_from_camera.translation.vector, direction)
    }
}

#[derive(Debug, Clone)]
pub struct EstimatorConfig {
    pub huber_px: f64,
    pub max_source_rmse_px: f64,
    pub max_total_rmse_px: f64,
    pub max_condition: f64,
    pub max_translation_sigma_mm: f64,
    pub max_rotation_sigma_deg: f64,
    pub single_tag_reacquire_translation_m: f64,
    pub single_tag_reacquire_rotation_deg: f64,
    pub prediction_horizon_ms: f64,
    pub max_motion_window_ms: f64,
}

impl Default for EstimatorConfig {
    fn default() -> Self {
        Self {
            huber_px: 2.0,
            max_source_rmse_px: 6.0,
            max_total_rmse_px: 4.5,
            max_condition: 2e4,
            max_translation_sigma_mm: 3.0,
            max_rotation_sigma_deg: 1.5,
            single_tag_reacquire_translation_m: 0.08,
            single_tag_reacquire_rotation_deg: 30.0,
            prediction_horizon_ms: 250.0,
            max_motion_window_ms: 50.0,
        }
    }
}

#[derive(Debug, Clone, Serialize)]
pub struct EePoseEstimate {
    pub schema_version: u32,
    pub sequence: u64,
    pub timestamp_ns: i128,
    pub status: String,
    pub world_from_ee: Option<[[f64; 4]; 4]>,
    /// Concrete URDF frame estimated by `world_from_ee` (compatibility key).
    pub tracking_frame: String,
    pub reprojection_rmse_px: Option<f64>,
    pub used_cameras: Vec<String>,
    pub used_tags: Vec<usize>,
    pub rejected_sources: Vec<String>,
    pub corner_count: usize,
    pub condition: Option<f64>,
    pub translation_sigma_mm: Option<f64>,
    pub rotation_sigma_deg: Option<f64>,
    pub reason: Option<String>,
    pub twist: Option<[f64; 6]>,
    pub calibration_id: String,
    pub wrist_layout_hash: String,
    pub inventory_hash: String,
    pub maximum_skew_ns: u128,
    /// Cameras present in the synchronized input set. Empty for legacy or
    /// detection-only replay rows that do not retain acquisition membership.
    pub input_cameras: Vec<String>,
    /// Whether a live tracker update used fewer cameras than the calibration
    /// bundle. `None` means the replay input did not retain that distinction.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub partial_input: Option<bool>,
    /// Capture-to-processing age.  This includes decode, synchronization and
    /// bounded ingress-queue delay, unlike the detector CPU timer below.
    pub queue_latency_ms: f64,
    pub detection_latency_ms: f64,
    pub image_prep_latency_ms: f64,
    pub apriltag_latency_ms: f64,
    pub roi_camera_count: usize,
    pub solver_latency_ms: f64,
    pub processing_latency_ms: f64,
    pub latency_basis: String,
    /// End-to-end capture-to-estimate latency.
    pub latency_ms: f64,
    pub detections: BTreeMap<String, Vec<FiducialDetection>>,
}

#[derive(Debug)]
pub struct RustEeTracker {
    cameras: BTreeMap<String, CameraModel>,
    calibration_id: String,
    layout: WristLayout,
    minimum_acquisition_ids: usize,
    config: EstimatorConfig,
    last_pose: Option<(i128, Isometry3<f64>)>,
    twist: [f64; 6],
}

#[derive(Debug)]
struct MeasuredPose {
    pose: Isometry3<f64>,
    rmse: f64,
    detections: Vec<FiducialDetection>,
    rejected: Vec<String>,
    condition: f64,
    translation_sigma_mm: f64,
    rotation_sigma_deg: f64,
}

impl RustEeTracker {
    pub fn new(
        calibration: &CalibrationBundle,
        inventory: &FiducialInventory,
        layout: WristLayout,
        config: EstimatorConfig,
    ) -> Result<Self> {
        if layout.calibration_status != "calibrated" {
            anyhow::bail!(
                "Rust EE tracking refuses {} wrist geometry",
                layout.calibration_status
            );
        }
        let cameras = calibration
            .cameras
            .iter()
            .map(|(name, camera)| Ok((name.clone(), CameraModel::new(camera)?)))
            .collect::<Result<_>>()?;
        Ok(Self {
            cameras,
            calibration_id: calibration.bundle_id.clone(),
            layout,
            minimum_acquisition_ids: inventory
                .target("wrist")?
                .minimum_acquisition_ids
                .unwrap_or(2),
            config,
            last_pose: None,
            twist: [0.0; 6],
        })
    }

    pub fn update(
        &mut self,
        sequence: u64,
        timestamp_ns: i128,
        maximum_skew_ns: u128,
        detections: Vec<FiducialDetection>,
        queue_latency_ms: f64,
        detection_latency_ms: f64,
        started: Instant,
    ) -> EePoseEstimate {
        self.update_constrained(
            sequence,
            timestamp_ns,
            maximum_skew_ns,
            detections,
            queue_latency_ms,
            detection_latency_ms,
            started,
            0,
        )
    }

    pub fn update_constrained(
        &mut self,
        sequence: u64,
        timestamp_ns: i128,
        maximum_skew_ns: u128,
        detections: Vec<FiducialDetection>,
        queue_latency_ms: f64,
        detection_latency_ms: f64,
        started: Instant,
        minimum_tag_ids: usize,
    ) -> EePoseEstimate {
        let solver_started = Instant::now();
        let measured = self.estimate(&detections, timestamp_ns, minimum_tag_ids);
        let solver_latency_ms = solver_started.elapsed().as_secs_f64() * 1000.0;
        let grouped = group_detections(&detections);
        let processing_latency_ms = started.elapsed().as_secs_f64() * 1000.0;
        match measured {
            Ok(value) => {
                self.update_motion(timestamp_ns, &value.pose);
                self.last_pose = Some((timestamp_ns, value.pose));
                EePoseEstimate {
                    schema_version: 1,
                    sequence,
                    timestamp_ns,
                    status: "measured".into(),
                    world_from_ee: Some(isometry_to_array(&value.pose)),
                    tracking_frame: self.layout.parent_frame.clone(),
                    reprojection_rmse_px: Some(value.rmse),
                    used_cameras: value
                        .detections
                        .iter()
                        .map(|item| item.camera.clone())
                        .collect::<BTreeSet<_>>()
                        .into_iter()
                        .collect(),
                    used_tags: value
                        .detections
                        .iter()
                        .map(|item| item.tag_id)
                        .collect::<BTreeSet<_>>()
                        .into_iter()
                        .collect(),
                    rejected_sources: value.rejected,
                    corner_count: value.detections.len() * 4,
                    condition: Some(value.condition),
                    translation_sigma_mm: Some(value.translation_sigma_mm),
                    rotation_sigma_deg: Some(value.rotation_sigma_deg),
                    reason: None,
                    twist: Some(self.twist),
                    calibration_id: self.calibration_id.clone(),
                    wrist_layout_hash: self.layout.layout_hash.clone(),
                    inventory_hash: self.layout.inventory_hash.clone(),
                    maximum_skew_ns,
                    input_cameras: Vec::new(),
                    partial_input: None,
                    queue_latency_ms,
                    detection_latency_ms,
                    image_prep_latency_ms: 0.0,
                    apriltag_latency_ms: 0.0,
                    roi_camera_count: 0,
                    solver_latency_ms,
                    processing_latency_ms,
                    latency_basis: "capture_to_estimate".into(),
                    latency_ms: queue_latency_ms + processing_latency_ms,
                    detections: grouped,
                }
            }
            Err(reason) => {
                let predicted = self.last_pose.as_ref().and_then(|(last_ns, pose)| {
                    let age_ms = (timestamp_ns - *last_ns) as f64 / 1e6;
                    (age_ms >= 0.0 && age_ms <= self.config.prediction_horizon_ms)
                        .then(|| propagate_pose(pose, self.twist, age_ms / 1000.0))
                });
                EePoseEstimate {
                    schema_version: 1,
                    sequence,
                    timestamp_ns,
                    status: if predicted.is_some() {
                        "predicted"
                    } else {
                        "unavailable"
                    }
                    .into(),
                    world_from_ee: predicted.as_ref().map(isometry_to_array),
                    tracking_frame: self.layout.parent_frame.clone(),
                    reprojection_rmse_px: None,
                    used_cameras: Vec::new(),
                    used_tags: Vec::new(),
                    rejected_sources: Vec::new(),
                    corner_count: 0,
                    condition: None,
                    translation_sigma_mm: None,
                    rotation_sigma_deg: None,
                    reason: Some(reason.to_string()),
                    twist: predicted.map(|_| self.twist),
                    calibration_id: self.calibration_id.clone(),
                    wrist_layout_hash: self.layout.layout_hash.clone(),
                    inventory_hash: self.layout.inventory_hash.clone(),
                    maximum_skew_ns,
                    input_cameras: Vec::new(),
                    partial_input: None,
                    queue_latency_ms,
                    detection_latency_ms,
                    image_prep_latency_ms: 0.0,
                    apriltag_latency_ms: 0.0,
                    roi_camera_count: 0,
                    solver_latency_ms,
                    processing_latency_ms,
                    latency_basis: "capture_to_estimate".into(),
                    latency_ms: queue_latency_ms + processing_latency_ms,
                    detections: grouped,
                }
            }
        }
    }

    fn update_motion(&mut self, timestamp_ns: i128, pose: &Isometry3<f64>) {
        let Some((last_ns, last)) = self.last_pose.as_ref() else {
            self.twist = [0.0; 6];
            return;
        };
        let dt = (timestamp_ns - *last_ns) as f64 / 1e9;
        if dt <= 1e-4 || dt > 1.0 {
            self.twist = [0.0; 6];
            return;
        }
        let angular = (last.rotation.inverse() * pose.rotation).scaled_axis() / dt;
        let linear = (pose.translation.vector - last.translation.vector) / dt;
        self.twist = [
            angular.x, angular.y, angular.z, linear.x, linear.y, linear.z,
        ];
    }

    fn estimate(
        &self,
        detections: &[FiducialDetection],
        timestamp_ns: i128,
        minimum_tag_ids: usize,
    ) -> Result<MeasuredPose> {
        if detections.is_empty() {
            anyhow::bail!("no configured wrist tags detected");
        }
        let mut source_counts = BTreeMap::<(&str, usize), usize>::new();
        for detection in detections {
            *source_counts
                .entry((&detection.camera, detection.tag_id))
                .or_default() += 1;
        }
        let duplicate_sources = source_counts
            .into_iter()
            .filter(|(_, count)| *count > 1)
            .map(|((camera, tag_id), count)| format!("{camera}/tag{tag_id} ({count} detections)"))
            .collect::<Vec<_>>();
        if !duplicate_sources.is_empty() {
            anyhow::bail!(
                "ambiguous duplicate wrist IDs in one camera: {}",
                duplicate_sources.join(", ")
            );
        }
        let tag_count = detections
            .iter()
            .map(|item| item.tag_id)
            .collect::<BTreeSet<_>>()
            .len();
        let camera_count = detections
            .iter()
            .map(|item| &item.camera)
            .collect::<BTreeSet<_>>()
            .len();
        let required_tag_ids = if self.last_pose.is_none() {
            self.minimum_acquisition_ids.max(minimum_tag_ids)
        } else {
            minimum_tag_ids
        };
        if tag_count < required_tag_ids {
            if self.last_pose.is_none() {
                anyhow::bail!("acquisition needs {required_tag_ids} tag ids, got {tag_count}");
            }
            anyhow::bail!("pose update needs {required_tag_ids} tag ids, got {tag_count}");
        }
        if camera_count < 2 && tag_count < 2 {
            anyhow::bail!("pose is not observable from one camera and one planar tag");
        }
        let mut candidates = self.multiview_candidates(detections)?;
        candidates.extend(self.planar_candidates(detections));
        if let Some((_, tracked)) = &self.last_pose {
            candidates.push(*tracked);
        }
        if candidates.is_empty() {
            anyhow::bail!("no valid pose initializer");
        }
        let initial = candidates
            .into_iter()
            .map(|candidate| {
                let scores = self.source_rmses(&candidate, detections, timestamp_ns);
                let inliers = scores
                    .iter()
                    .filter(|(_, rmse)| **rmse <= self.config.max_source_rmse_px)
                    .count();
                let mean = scores.values().sum::<f64>() / scores.len().max(1) as f64;
                (inliers, -mean, candidate)
            })
            .max_by(|left, right| {
                left.0
                    .cmp(&right.0)
                    .then_with(|| left.1.total_cmp(&right.1))
            })
            .map(|(_, _, pose)| pose)
            .context("pose initialization failed")?;
        if tag_count == 1 {
            if let Some((_, tracked)) = &self.last_pose {
                let translation = (initial.translation.vector - tracked.translation.vector).norm();
                let rotation = (tracked.rotation.inverse() * initial.rotation)
                    .angle()
                    .to_degrees();
                if translation > self.config.single_tag_reacquire_translation_m
                    || rotation > self.config.single_tag_reacquire_rotation_deg
                {
                    anyhow::bail!("single-tag continuation disagrees with tracked pose");
                }
            }
        }
        // The initializer was selected because it already maximizes rigid
        // source consensus. Do not immediately feed its known outliers back
        // into the first nonlinear solve: one camera that sees several tag
        // faces can otherwise pull the pose far enough to make the agreeing
        // cameras fail the second source gate. This was the camera2 failure
        // mode in the retained five-camera moving sequence.
        let initial_scores = self.source_rmses(&initial, detections, timestamp_ns);
        let initial_kept =
            sources_within_gate(detections, &initial_scores, self.config.max_source_rmse_px);
        if initial_kept.len() * 4 < 8 {
            anyhow::bail!("fewer than eight initializer-consensus corners remain");
        }
        let first = self.refine(initial, &initial_kept, timestamp_ns)?;
        let scores = self.source_rmses(&first.0, detections, timestamp_ns);
        let kept = sources_within_gate(detections, &scores, self.config.max_source_rmse_px);
        if kept.len() * 4 < 8 {
            anyhow::bail!("fewer than eight inlier corners remain");
        }
        let (pose, rmse, condition, translation_sigma_mm, rotation_sigma_deg) =
            self.refine(first.0, &kept, timestamp_ns)?;
        if rmse > self.config.max_total_rmse_px {
            anyhow::bail!("reprojection RMSE {rmse:.2} px exceeds gate");
        }
        if condition > self.config.max_condition {
            anyhow::bail!("pose condition {condition:.1} exceeds gate");
        }
        if translation_sigma_mm > self.config.max_translation_sigma_mm
            || rotation_sigma_deg > self.config.max_rotation_sigma_deg
        {
            anyhow::bail!(
                "pose uncertainty {translation_sigma_mm:.2} mm / {rotation_sigma_deg:.2} deg exceeds {:.2} mm / {:.2} deg gate",
                self.config.max_translation_sigma_mm,
                self.config.max_rotation_sigma_deg,
            );
        }
        let kept_names: BTreeSet<_> = kept.iter().map(source_name).collect();
        let rejected = detections
            .iter()
            .map(source_name)
            .filter(|name| !kept_names.contains(name))
            .collect();
        Ok(MeasuredPose {
            pose,
            rmse,
            detections: kept,
            rejected,
            condition,
            translation_sigma_mm,
            rotation_sigma_deg,
        })
    }

    fn multiview_candidates(
        &self,
        detections: &[FiducialDetection],
    ) -> Result<Vec<Isometry3<f64>>> {
        let mut by_tag = BTreeMap::<usize, Vec<&FiducialDetection>>::new();
        for detection in detections {
            by_tag.entry(detection.tag_id).or_default().push(detection);
        }
        let mut output = Vec::new();
        for (tag_id, observations) in by_tag {
            if observations
                .iter()
                .map(|item| &item.camera)
                .collect::<BTreeSet<_>>()
                .len()
                < 2
            {
                continue;
            }
            let mut measured = Vec::new();
            for corner_index in 0..4 {
                let rays: Vec<_> = observations
                    .iter()
                    .filter_map(|item| {
                        self.cameras
                            .get(&item.camera)
                            .map(|camera| camera.ray(item.corners_px[corner_index]))
                    })
                    .collect();
                let Some(point) = triangulate(&rays) else {
                    measured.clear();
                    break;
                };
                measured.push(Point3::from(point));
            }
            if measured.len() == 4 {
                output.push(fit_rigid(&self.layout.corners_ee(tag_id)?, &measured)?);
            }
        }
        Ok(output)
    }

    fn planar_candidates(&self, detections: &[FiducialDetection]) -> Vec<Isometry3<f64>> {
        let mut output = Vec::new();
        for detection in detections {
            let Some(camera) = self.cameras.get(&detection.camera) else {
                continue;
            };
            let Some(ee_from_tag) = self.layout.ee_from_tag.get(&detection.tag_id) else {
                continue;
            };
            for raw in &detection.camera_from_tag_candidates {
                let mut best: Option<(f64, Isometry3<f64>)> = None;
                for flip in [false, true] {
                    for quarter_turn in 0..4 {
                        let z = UnitQuaternion::from_axis_angle(
                            &Vector3::z_axis(),
                            quarter_turn as f64 * std::f64::consts::FRAC_PI_2,
                        );
                        let x = if flip {
                            UnitQuaternion::from_axis_angle(
                                &Vector3::x_axis(),
                                std::f64::consts::PI,
                            )
                        } else {
                            UnitQuaternion::identity()
                        };
                        let adjusted =
                            *raw * Isometry3::from_parts(Translation3::identity(), z * x);
                        let world_from_ee =
                            camera.world_from_camera * adjusted * ee_from_tag.inverse();
                        let score = self
                            .source_rmses(
                                &world_from_ee,
                                std::slice::from_ref(detection),
                                detection.timestamp_ns,
                            )
                            .values()
                            .next()
                            .copied()
                            .unwrap_or(f64::INFINITY);
                        if best.as_ref().is_none_or(|(value, _)| score < *value) {
                            best = Some((score, world_from_ee));
                        }
                    }
                }
                if let Some((_, pose)) = best {
                    output.push(pose);
                }
            }
        }
        output
    }

    fn residuals(
        &self,
        pose: &Isometry3<f64>,
        detections: &[FiducialDetection],
        timestamp_ns: i128,
    ) -> Vec<f64> {
        let mut output = Vec::with_capacity(detections.len() * 8);
        for detection in detections {
            let Some(camera) = self.cameras.get(&detection.camera) else {
                continue;
            };
            let dt = ((detection.timestamp_ns - timestamp_ns) as f64 / 1e9).clamp(
                -self.config.max_motion_window_ms / 1000.0,
                self.config.max_motion_window_ms / 1000.0,
            );
            let pose_at_camera = propagate_pose(pose, self.twist, dt);
            let Ok(corners) = self.layout.corners_ee(detection.tag_id) else {
                continue;
            };
            for (corner, measured) in corners.iter().zip(detection.corners_px) {
                let world = pose_at_camera.transform_point(corner);
                if let Some(projected) = camera.project(&world) {
                    output.push(projected.x - measured[0]);
                    output.push(projected.y - measured[1]);
                } else {
                    output.extend([1e3, 1e3]);
                }
            }
        }
        output
    }

    fn source_rmses(
        &self,
        pose: &Isometry3<f64>,
        detections: &[FiducialDetection],
        timestamp_ns: i128,
    ) -> BTreeMap<String, f64> {
        detections
            .iter()
            .map(|item| {
                let residuals = self.residuals(pose, std::slice::from_ref(item), timestamp_ns);
                let rmse = (residuals.iter().map(|value| value * value).sum::<f64>()
                    / residuals.len().max(1) as f64)
                    .sqrt();
                (source_name(item), rmse)
            })
            .collect()
    }

    fn refine(
        &self,
        mut pose: Isometry3<f64>,
        detections: &[FiducialDetection],
        timestamp_ns: i128,
    ) -> Result<(Isometry3<f64>, f64, f64, f64, f64)> {
        for _ in 0..20 {
            let residuals = self.residuals(&pose, detections, timestamp_ns);
            if residuals.len() < 8 {
                anyhow::bail!("not enough reprojection residuals");
            }
            let rows = residuals.len();
            let mut jacobian = DMatrix::<f64>::zeros(rows, 6);
            for column in 0..6 {
                let epsilon = if column < 3 { 1e-6 } else { 1e-5 };
                let mut delta = [0.0; 6];
                delta[column] = epsilon;
                let perturbed = apply_delta(&pose, delta);
                let values = self.residuals(&perturbed, detections, timestamp_ns);
                for row in 0..rows {
                    jacobian[(row, column)] = (values[row] - residuals[row]) / epsilon;
                }
            }
            let mut weighted_jacobian = jacobian.clone();
            let mut weighted_residuals = DVector::from_vec(residuals.clone());
            for row in 0..rows {
                let magnitude = residuals[row].abs();
                let weight = if magnitude <= self.config.huber_px {
                    1.0
                } else {
                    (self.config.huber_px / magnitude).sqrt()
                };
                weighted_residuals[row] *= weight;
                for column in 0..6 {
                    weighted_jacobian[(row, column)] *= weight;
                }
            }
            let hessian = weighted_jacobian.transpose() * &weighted_jacobian;
            let gradient = weighted_jacobian.transpose() * weighted_residuals;
            let damped = &hessian + DMatrix::<f64>::identity(6, 6) * 1e-6;
            let Some(step) = damped.lu().solve(&(-gradient)) else {
                anyhow::bail!("fiducial normal equations are singular");
            };
            let delta = [step[0], step[1], step[2], step[3], step[4], step[5]];
            pose = apply_delta(&pose, delta);
            if step.norm() < 1e-7 {
                break;
            }
        }
        let final_residuals = self.residuals(&pose, detections, timestamp_ns);
        let rows = final_residuals.len();
        if rows < 8 {
            anyhow::bail!("not enough final reprojection residuals");
        }
        let mut final_jacobian = DMatrix::<f64>::zeros(rows, 6);
        for column in 0..6 {
            let epsilon = if column < 3 { 1e-6 } else { 1e-5 };
            let mut delta = [0.0; 6];
            delta[column] = epsilon;
            let values = self.residuals(&apply_delta(&pose, delta), detections, timestamp_ns);
            for row in 0..rows {
                let weight = if final_residuals[row].abs() <= self.config.huber_px {
                    1.0
                } else {
                    (self.config.huber_px / final_residuals[row].abs()).sqrt()
                };
                final_jacobian[(row, column)] =
                    (values[row] - final_residuals[row]) / epsilon * weight;
            }
        }
        let hessian = final_jacobian.transpose() * final_jacobian;
        let final_hessian = SMatrix::<f64, 6, 6>::from_fn(|row, col| hessian[(row, col)]);
        let rmse = (final_residuals
            .iter()
            .map(|value| value * value)
            .sum::<f64>()
            / final_residuals.len().max(1) as f64)
            .sqrt();
        let eigen = final_hessian.symmetric_eigen().eigenvalues;
        let min = eigen
            .iter()
            .copied()
            .fold(f64::INFINITY, f64::min)
            .max(1e-12);
        let max = eigen.iter().copied().fold(0.0_f64, f64::max);
        let condition = max / min;
        let inverse = final_hessian
            .try_inverse()
            .context("fiducial covariance matrix is singular")?;
        let variance = rmse * rmse;
        let rotation_sigma_deg = (0..3)
            .map(|index| {
                (inverse[(index, index)] * variance)
                    .max(0.0)
                    .sqrt()
                    .to_degrees()
            })
            .fold(0.0_f64, f64::max);
        let translation_sigma_mm = (3..6)
            .map(|index| (inverse[(index, index)] * variance).max(0.0).sqrt() * 1000.0)
            .fold(0.0_f64, f64::max);
        Ok((
            pose,
            rmse,
            condition,
            translation_sigma_mm,
            rotation_sigma_deg,
        ))
    }
}

fn group_detections(detections: &[FiducialDetection]) -> BTreeMap<String, Vec<FiducialDetection>> {
    let mut grouped = BTreeMap::new();
    for detection in detections {
        grouped
            .entry(detection.camera.clone())
            .or_insert_with(Vec::new)
            .push(detection.clone());
    }
    grouped
}

fn source_name(detection: &FiducialDetection) -> String {
    format!("{}:tag{}", detection.camera, detection.tag_id)
}

fn sources_within_gate(
    detections: &[FiducialDetection],
    scores: &BTreeMap<String, f64>,
    max_source_rmse_px: f64,
) -> Vec<FiducialDetection> {
    detections
        .iter()
        .filter(|item| {
            scores
                .get(&source_name(item))
                .is_some_and(|rmse| *rmse <= max_source_rmse_px)
        })
        .cloned()
        .collect()
}

fn apply_delta(pose: &Isometry3<f64>, delta: [f64; 6]) -> Isometry3<f64> {
    // Rotation and translation are independent pose parameters, matching the
    // Python reference estimator.  Left-multiplying a full SE(3) delta would
    // rotate `pose.translation` about the world origin.  Besides being the
    // wrong update for an EE rotating about its own origin, that couples the
    // covariance coordinates and makes translation_sigma_mm meaningless.
    Isometry3::from_parts(
        Translation3::from(pose.translation.vector + Vector3::new(delta[3], delta[4], delta[5])),
        UnitQuaternion::from_scaled_axis(Vector3::new(delta[0], delta[1], delta[2]))
            * pose.rotation,
    )
}

fn propagate_pose(pose: &Isometry3<f64>, twist: [f64; 6], dt: f64) -> Isometry3<f64> {
    let delta = [
        twist[0] * dt,
        twist[1] * dt,
        twist[2] * dt,
        twist[3] * dt,
        twist[4] * dt,
        twist[5] * dt,
    ];
    apply_delta(pose, delta)
}

fn triangulate(rays: &[(Vector3<f64>, Vector3<f64>)]) -> Option<Vector3<f64>> {
    if rays.len() < 2 {
        return None;
    }
    let mut matrix = Matrix3::zeros();
    let mut rhs = Vector3::zeros();
    for (origin, direction) in rays {
        let projector = Matrix3::identity() - direction * direction.transpose();
        matrix += projector;
        rhs += projector * origin;
    }
    matrix.try_inverse().map(|inverse| inverse * rhs)
}

fn fit_rigid(model: &[Point3<f64>; 4], measured: &[Point3<f64>]) -> Result<Isometry3<f64>> {
    let model_center = model.iter().map(|point| point.coords).sum::<Vector3<f64>>() / 4.0;
    let measured_center = measured
        .iter()
        .map(|point| point.coords)
        .sum::<Vector3<f64>>()
        / 4.0;
    let mut covariance = Matrix3::zeros();
    for (left, right) in model.iter().zip(measured) {
        covariance += (left.coords - model_center) * (right.coords - measured_center).transpose();
    }
    let svd = covariance.svd(true, true);
    let u = svd.u.context("rigid fit has no U")?;
    let v_t = svd.v_t.context("rigid fit has no Vt")?;
    let mut correction = Matrix3::identity();
    correction[(2, 2)] = (v_t.transpose() * u.transpose()).determinant().signum();
    let rotation = v_t.transpose() * correction * u.transpose();
    let translation = measured_center - rotation * model_center;
    Ok(Isometry3::from_parts(
        Translation3::from(translation),
        UnitQuaternion::from_matrix(&rotation),
    ))
}

fn isometry_from_pose(pose: &crate::Pose) -> Result<Isometry3<f64>> {
    let rotation = Matrix3::from_row_slice(&pose.rotation);
    if (rotation.determinant() - 1.0).abs() > 0.05 {
        anyhow::bail!("calibration pose rotation is invalid");
    }
    Ok(Isometry3::from_parts(
        Translation3::new(
            pose.translation_m[0],
            pose.translation_m[1],
            pose.translation_m[2],
        ),
        UnitQuaternion::from_matrix(&rotation),
    ))
}

fn isometry_from_apriltag_pose(pose: &apriltag::pose::Pose) -> Result<Isometry3<f64>> {
    let rotation = Matrix3::from_row_slice(pose.rotation().data());
    let translation = pose.translation().data();
    if translation.len() != 3 {
        anyhow::bail!("AprilTag pose translation is not 3x1");
    }
    Ok(Isometry3::from_parts(
        Translation3::new(translation[0], translation[1], translation[2]),
        UnitQuaternion::from_matrix(&rotation),
    ))
}

fn isometry_from_array(rows: [[f64; 4]; 4]) -> Result<Isometry3<f64>> {
    let matrix = Matrix4::from_row_slice(&rows.into_iter().flatten().collect::<Vec<_>>());
    if !matrix.iter().all(|value| value.is_finite())
        || (matrix[(3, 3)] - 1.0).abs() > 1e-9
        || matrix.fixed_view::<1, 3>(3, 0).norm() > 1e-9
    {
        anyhow::bail!("invalid homogeneous transform");
    }
    let rotation = matrix.fixed_view::<3, 3>(0, 0).into_owned();
    if (rotation.determinant() - 1.0).abs() > 1e-4
        || (rotation.transpose() * rotation - Matrix3::identity()).norm() > 1e-4
    {
        anyhow::bail!("transform rotation is not rigid");
    }
    Ok(Isometry3::from_parts(
        Translation3::new(matrix[(0, 3)], matrix[(1, 3)], matrix[(2, 3)]),
        UnitQuaternion::from_matrix(&rotation),
    ))
}

fn isometry_to_array(transform: &Isometry3<f64>) -> [[f64; 4]; 4] {
    let matrix = transform.to_homogeneous();
    std::array::from_fn(|row| std::array::from_fn(|column| matrix[(row, column)]))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        DistortionModel, Intrinsics, PixelFormat, Pose, StreamProfile,
        calibration::CALIBRATION_SCHEMA_VERSION,
    };

    #[test]
    fn repository_inventory_and_calibrated_layout_validate() {
        let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
        let inventory = FiducialInventory::load(root.join("config/fiducials.json")).unwrap();
        assert_eq!(inventory.target("wrist").unwrap().ids, [3, 6, 7, 8]);
        let layout = WristLayout::load(
            root.join("config/wrist_tags_measured.json"),
            &inventory,
            true,
        )
        .unwrap();
        assert_eq!(layout.calibration_status, "calibrated");
        assert_eq!(layout.parent_frame, "right/gripper_left");
        WristLayout::load(
            root.join("config/wrist_tags_measured.json"),
            &inventory,
            false,
        )
        .unwrap();
    }

    #[test]
    fn detector_roi_expands_clamps_and_ignores_nonfinite_corners() {
        let detection = FiducialDetection {
            camera: "camera1".into(),
            tag_id: 3,
            corners_px: [[3.2, 4.8], [22.1, 5.0], [f64::NAN, 19.7], [2.9, 20.2]],
            timestamp_ns: 0,
            side_px: 20.0,
            decision_margin: 100.0,
            hamming: 0,
            camera_from_tag_candidates: Vec::new(),
        };

        assert_eq!(
            expanded_detection_roi(&[&detection], 24, 30, 5),
            Some(DetectionRoi {
                x0: 0,
                y0: 0,
                x1: 24,
                y1: 27,
            })
        );
        assert_eq!(expanded_detection_roi(&[], 24, 30, 5), None);
        assert_eq!(expanded_detection_roi(&[&detection], 0, 30, 5), None);
    }

    #[test]
    fn rigid_fit_recovers_transform() {
        let model = [
            Point3::new(-1.0, 1.0, 0.0),
            Point3::new(1.0, 1.0, 0.0),
            Point3::new(1.0, -1.0, 0.0),
            Point3::new(-1.0, -1.0, 0.0),
        ];
        let truth = Isometry3::from_parts(
            Translation3::new(0.2, -0.1, 0.7),
            UnitQuaternion::from_scaled_axis(Vector3::new(0.2, -0.3, 0.4)),
        );
        let measured: Vec<_> = model
            .iter()
            .map(|point| truth.transform_point(point))
            .collect();
        let fit = fit_rigid(&model, &measured).unwrap();
        assert!((fit.translation.vector - truth.translation.vector).norm() < 1e-10);
        assert!((fit.rotation.inverse() * truth.rotation).angle() < 1e-10);
    }

    #[test]
    fn triangulation_recovers_point() {
        let point = Vector3::new(0.2, -0.1, 1.0);
        let origins = [Vector3::new(-0.5, 0.0, 0.0), Vector3::new(0.5, 0.2, 0.0)];
        let rays: Vec<_> = origins
            .into_iter()
            .map(|origin| (origin, (point - origin).normalize()))
            .collect();
        assert!((triangulate(&rays).unwrap() - point).norm() < 1e-10);
    }

    #[test]
    fn rotational_pose_delta_does_not_move_the_ee_origin() {
        let pose = Isometry3::from_parts(
            Translation3::new(0.8, -0.4, 1.2),
            UnitQuaternion::from_euler_angles(0.2, -0.3, 0.1),
        );
        let perturbed = apply_delta(&pose, [0.01, -0.02, 0.03, 0.0, 0.0, 0.0]);
        assert_eq!(perturbed.translation.vector, pose.translation.vector);
        assert!((perturbed.rotation.inverse() * pose.rotation).angle() > 0.0);
    }

    #[test]
    fn initializer_consensus_gate_drops_a_multi_tag_camera_outlier() {
        let detection = |camera: &str, tag_id| FiducialDetection {
            camera: camera.into(),
            tag_id,
            corners_px: [[0.0; 2]; 4],
            timestamp_ns: 0,
            side_px: 40.0,
            decision_margin: 100.0,
            hamming: 0,
            camera_from_tag_candidates: Vec::new(),
        };
        let detections = vec![
            detection("camera1", 3),
            detection("camera3", 3),
            detection("camera2", 3),
            detection("camera2", 6),
        ];
        let scores = BTreeMap::from([
            ("camera1:tag3".into(), 1.2),
            ("camera3:tag3".into(), 1.4),
            ("camera2:tag3".into(), 14.0),
            ("camera2:tag6".into(), 15.0),
        ]);

        let kept = sources_within_gate(&detections, &scores, 6.0);

        assert_eq!(kept.len(), 2);
        assert!(kept.iter().all(|item| item.camera != "camera2"));
    }

    #[test]
    fn synthetic_multicamera_tracker_recovers_ee_pose() {
        let profile = StreamProfile {
            stream: "main".into(),
            width: 1280,
            height: 720,
            fps_num: 20,
            fps_den: 1,
            format: PixelFormat::Bgr8,
        };
        let camera = |name: &str, x: f64| CameraCalibration {
            sensor_name: name.into(),
            profile: profile.clone(),
            intrinsics: Intrinsics {
                width: 1280,
                height: 720,
                fx: 900.0,
                fy: 900.0,
                cx: 640.0,
                cy: 360.0,
            },
            distortion: DistortionModel {
                model: "brown_conrady".into(),
                coefficients: vec![0.0; 8],
            },
            world_from_camera: Pose {
                rotation: [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
                translation_m: [x, 0.0, 0.0],
            },
            depth_to_color: None,
            metadata: BTreeMap::new(),
        };
        let cameras = BTreeMap::from([
            ("cam_left".into(), camera("cam_left", -0.25)),
            ("cam_right".into(), camera("cam_right", 0.25)),
        ]);
        let calibration = CalibrationBundle {
            schema_version: CALIBRATION_SCHEMA_VERSION,
            bundle_id: String::new(),
            world_frame: "world".into(),
            cameras,
        }
        .with_computed_id()
        .unwrap();
        let inventory = FiducialInventory {
            family: FAMILY.into(),
            detector: BTreeMap::new(),
            targets: BTreeMap::from([(
                "wrist".into(),
                TargetSpec {
                    role: "rigid_ee".into(),
                    ids: vec![3, 6, 7, 8],
                    edge_m: 0.056,
                    layout: None,
                    parent_frame: Some("right/gripper_left".into()),
                    minimum_acquisition_ids: Some(2),
                    ambiguity_group: None,
                    root_id: None,
                    calibration_root_id: None,
                    grid: None,
                    minimum_calibration_observations: None,
                    minimum_calibration_poses_per_id: None,
                    max_calibration_corner_px: None,
                    max_calibration_residual_mm: None,
                    max_calibration_parent_distance_mm: None,
                    max_calibration_reprojection_px: None,
                    max_calibration_consensus_mm: None,
                    max_calibration_regression_mm: None,
                },
            )]),
            spare_ids: Vec::new(),
            inventory_hash: "synthetic-inventory".into(),
            source: PathBuf::from("synthetic"),
        };
        let layout = WristLayout {
            calibration_status: "calibrated".into(),
            edge_m: 0.056,
            ee_from_tag: BTreeMap::from([
                (3, Isometry3::translation(-0.055, 0.0, 0.0)),
                (
                    6,
                    Isometry3::from_parts(
                        Translation3::new(0.055, 0.015, 0.01),
                        UnitQuaternion::from_euler_angles(0.08, -0.12, 0.15),
                    ),
                ),
                (
                    7,
                    Isometry3::from_parts(
                        Translation3::new(0.0, -0.05, -0.01),
                        UnitQuaternion::from_euler_angles(-0.1, 0.06, -0.2),
                    ),
                ),
                (
                    8,
                    Isometry3::from_parts(
                        Translation3::new(0.0, 0.05, 0.015),
                        UnitQuaternion::from_euler_angles(0.12, 0.04, 0.22),
                    ),
                ),
            ]),
            parent_frame: "right/gripper_left".into(),
            layout_hash: "synthetic-layout".into(),
            inventory_hash: "synthetic-inventory".into(),
        };
        let truth = Isometry3::from_parts(
            Translation3::new(0.03, -0.02, 1.1),
            UnitQuaternion::from_euler_angles(0.12, -0.16, 0.08),
        );
        let mut detections = Vec::new();
        for (camera_name, camera_calibration) in &calibration.cameras {
            let model = CameraModel::new(camera_calibration).unwrap();
            for tag_id in [3, 6] {
                let corners_px = layout.corners_ee(tag_id).unwrap().map(|corner| {
                    let pixel = model.project(&truth.transform_point(&corner)).unwrap();
                    [pixel.x, pixel.y]
                });
                detections.push(FiducialDetection {
                    camera: camera_name.clone(),
                    tag_id,
                    corners_px,
                    timestamp_ns: 1_000_000_000,
                    side_px: 40.0,
                    decision_margin: 100.0,
                    hamming: 0,
                    camera_from_tag_candidates: Vec::new(),
                });
            }
        }
        let mut tracker = RustEeTracker::new(
            &calibration,
            &inventory,
            layout,
            EstimatorConfig {
                max_condition: f64::INFINITY,
                max_translation_sigma_mm: f64::INFINITY,
                max_rotation_sigma_deg: f64::INFINITY,
                ..EstimatorConfig::default()
            },
        )
        .unwrap();
        let mut ambiguous = detections.clone();
        ambiguous.push(detections[0].clone());
        let duplicate_rejection =
            tracker.update(5, 800_000_000, 0, ambiguous, 0.0, 0.0, Instant::now());
        assert_eq!(duplicate_rejection.status, "unavailable");
        assert!(
            duplicate_rejection
                .reason
                .as_deref()
                .unwrap()
                .contains("ambiguous duplicate wrist IDs")
        );
        let acquisition_rejection = tracker.update(
            6,
            900_000_000,
            0,
            detections
                .iter()
                .filter(|item| item.tag_id == 3)
                .cloned()
                .collect(),
            0.0,
            0.0,
            Instant::now(),
        );
        assert_eq!(acquisition_rejection.status, "unavailable");
        assert!(
            acquisition_rejection
                .reason
                .as_deref()
                .unwrap()
                .contains("acquisition needs 2 tag ids")
        );
        let one_tag = detections
            .iter()
            .filter(|item| item.tag_id == 3)
            .cloned()
            .collect();
        let estimate = tracker.update(7, 1_000_000_000, 0, detections, 12.5, 0.0, Instant::now());
        assert_eq!(estimate.status, "measured", "{:?}", estimate.reason);
        assert_eq!(estimate.queue_latency_ms, 12.5);
        assert!(estimate.latency_ms >= estimate.queue_latency_ms);
        let recovered = isometry_from_array(estimate.world_from_ee.unwrap()).unwrap();
        assert!((recovered.translation.vector - truth.translation.vector).norm() < 1e-5);
        assert!((recovered.rotation.inverse() * truth.rotation).angle() < 1e-5);
        assert!(estimate.reprojection_rmse_px.unwrap() < 1e-5);
        assert_eq!(estimate.used_tags, [3, 6]);
        assert_eq!(estimate.used_cameras, ["cam_left", "cam_right"]);

        let partial_rejection =
            tracker.update_constrained(8, 1_050_000_000, 0, one_tag, 0.0, 0.0, Instant::now(), 2);
        assert_eq!(partial_rejection.status, "predicted");
        assert!(
            partial_rejection
                .reason
                .as_deref()
                .unwrap()
                .contains("pose update needs 2 tag ids")
        );

        let predicted = tracker.update(9, 1_100_000_000, 0, Vec::new(), 0.0, 0.0, Instant::now());
        assert_eq!(predicted.status, "predicted");
        assert!(predicted.world_from_ee.is_some());
        let expired = tracker.update(10, 1_300_000_001, 0, Vec::new(), 0.0, 0.0, Instant::now());
        assert_eq!(expired.status, "unavailable");
        assert!(expired.world_from_ee.is_none());
    }
}
