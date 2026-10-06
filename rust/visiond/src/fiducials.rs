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

const SUPPORTED_INVENTORY_SCHEMA: u32 = 2;
const SUPPORTED_LAYOUT_SCHEMA: u32 = 2;
fn tag_family(name: &str) -> Result<Family> {
    match name {
        "apriltag_16h5" => Ok(Family::tag_16h5()),
        "apriltag_36h11" => Ok(Family::tag_36h11()),
        _ => anyhow::bail!("unsupported fiducial family {name}"),
    }
}

fn family_capacity(name: &str) -> Result<usize> {
    match name {
        "apriltag_16h5" => Ok(30),
        "apriltag_36h11" => Ok(587),
        _ => anyhow::bail!("unsupported fiducial family {name}"),
    }
}
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
    #[serde(default)]
    pub family: String,
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
        let mut raw: InventoryFile = serde_json::from_slice(&bytes)
            .with_context(|| format!("parsing {}", path.display()))?;
        if ![1, SUPPORTED_INVENTORY_SCHEMA].contains(&raw.schema_version) {
            anyhow::bail!(
                "unsupported fiducial inventory schema {}",
                raw.schema_version
            );
        }
        family_capacity(&raw.family)?;
        for (name, target) in &mut raw.targets {
            if target.family.is_empty() && raw.schema_version == 1 {
                target.family.clone_from(&raw.family);
            }
            let capacity = family_capacity(&target.family).with_context(|| {
                format!("target {name} must explicitly name a supported family")
            })?;
            anyhow::ensure!(
                target.ids.iter().all(|id| *id < capacity),
                "target {name} ids exceed family capacity"
            );
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
            for family in matches.iter().map(|(name, _)| &raw.targets[*name].family) {
                let instances: Vec<_> = matches.iter()
                    .filter(|(name, _)| &raw.targets[*name].family == family).collect();
                let groups: BTreeSet<_> = instances.iter().map(|(_, group)| *group).collect();
                if instances.len() > 1 && (groups.len() != 1 || groups.contains(&None)) {
                    anyhow::bail!("id {id} is duplicated without one ambiguity_group");
                }
            }
        }
        for (name, target) in &raw.targets {
            if let Some(id) = target.calibration_root_id
                && owners.get(&id).is_some_and(|matches| matches.iter()
                    .filter(|(name, _)| raw.targets[*name].family == target.family).count() != 1)
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
        let capacity = family_capacity(&raw.family)?;
        if spare_ids
            .iter()
            .any(|id| *id >= capacity || owners.contains_key(id))
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
    pub family: String,
    pub require_family_ids: BTreeSet<usize>,
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
        Self::load_target(path, inventory, "wrist", allow_pending)
    }

    /// The same rigid multiview estimator can track an independently measured
    /// paper layout. Its IDs must be unambiguous while the wrist is visible.
    pub fn load_target(
        path: impl AsRef<Path>,
        inventory: &FiducialInventory,
        target: &str,
        allow_pending: bool,
    ) -> Result<Self> {
        let path = path.as_ref();
        let bytes = fs::read(path).with_context(|| format!("reading {}", path.display()))?;
        let raw: LayoutFile = serde_json::from_slice(&bytes)
            .with_context(|| format!("parsing {}", path.display()))?;
        let wrist = inventory.target(target)?;
        if wrist.role != "rigid_ee" {
            anyhow::ensure!(
                wrist.role == "rigid_target",
                "paper tracking requires a rigid_target inventory entry"
            );
            anyhow::ensure!(
                wrist.ids.len() >= 2,
                "paper acquisition requires at least two tags"
            );
            anyhow::ensure!(
                inventory.targets.iter().all(|(name, other)| name == target
                    || other.ids.iter().all(|id| !wrist.ids.contains(id))),
                "paper tracking IDs overlap another visible inventory target"
            );
        }
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
        if ee_from_tag.keys().copied().collect::<BTreeSet<_>>() != expected
            && !(allow_pending
                && raw.calibration_status == "pending_recalibration"
                && ee_from_tag.is_empty())
        {
            anyhow::bail!("wrist layout transform ids differ from inventory");
        }
        Ok(Self {
            family: wrist.family.clone(),
            require_family_ids: wrist.ids.iter().copied().filter(|id| inventory.targets.values()
                .any(|other| other.family != wrist.family && other.ids.contains(id))).collect(),
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
    #[serde(default)]
    pub family: Option<String>,
    pub camera: String,
    pub tag_id: usize,
    pub corners_px: [[f64; 2]; 4],
    pub timestamp_ns: i128,
    pub side_px: f64,
    pub decision_margin: f32,
    pub hamming: usize,
    #[serde(default, with = "pose_candidates_serde")]
    camera_from_tag_candidates: Vec<Isometry3<f64>>,
}

// Retain the detector's planar ambiguity candidates for offline replay.
// Older logs omitted this field and still decode as an empty candidate set.
mod pose_candidates_serde {
    use super::*;

    pub fn serialize<S: serde::Serializer>(
        poses: &[Isometry3<f64>],
        serializer: S,
    ) -> std::result::Result<S::Ok, S::Error> {
        poses
            .iter()
            .map(isometry_to_array)
            .collect::<Vec<_>>()
            .serialize(serializer)
    }

    pub fn deserialize<'de, D: serde::Deserializer<'de>>(
        deserializer: D,
    ) -> std::result::Result<Vec<Isometry3<f64>>, D::Error> {
        Vec::<[[f64; 4]; 4]>::deserialize(deserializer)?
            .into_iter()
            .map(|rows| isometry_from_array(rows).map_err(serde::de::Error::custom))
            .collect()
    }
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
    pub quad_detection_latency_ms: f64,
    pub pose_candidate_latency_ms: f64,
    pub roi_camera_count: usize,
}

#[derive(Debug)]
struct CameraDetectionSet {
    detections: Vec<FiducialDetection>,
    image_prep_latency_ms: f64,
    apriltag_latency_ms: f64,
    quad_detection_latency_ms: f64,
    pose_candidate_latency_ms: f64,
    used_roi: bool,
}

pub struct AprilTagDetector {
    detectors: Vec<(String, Detector, BTreeSet<usize>)>,
    scale: f64,
    min_side_px: f64,
    tag_edges_m: BTreeMap<(String, usize), Option<f64>>,
    quad_decimate: f64,
    reacquire_quad_decimate: f64,
}

/// Cloneable detector settings for parallel processing of synchronized cameras.
/// Each worker owns its native detector so no C state is shared across threads.
#[derive(Debug, Clone, PartialEq)]
pub struct AprilTagDetectorFactory {
    families: BTreeMap<String, BTreeSet<usize>>,
    scale: f64,
    min_side_px: f64,
    tag_edges_m: BTreeMap<(String, usize), Option<f64>>,
    corner_refinement: bool,
    quad_decimate: f64,
    reacquire_quad_decimate: f64,
}

thread_local! {
    // One bounded cache entry per worker. Native C state never crosses threads.
    static WORKER_DETECTOR: std::cell::RefCell<Option<(AprilTagDetectorFactory, AprilTagDetector)>> = const { std::cell::RefCell::new(None) };
}

impl AprilTagDetectorFactory {
    pub fn new(
        inventory: &FiducialInventory,
        target: Option<&str>,
        scale: Option<f64>,
    ) -> Result<Self> {
        let selected = if let Some(name) = target {
            vec![inventory.target(name)?]
        } else {
            inventory.targets.values().collect()
        };
        let mut families = BTreeMap::<String, BTreeSet<usize>>::new();
        let mut tag_edges_m = BTreeMap::new();
        for spec in selected {
            families
                .entry(spec.family.clone())
                .or_default()
                .extend(&spec.ids);
            for id in &spec.ids {
                let edge = tag_edges_m.entry((spec.family.clone(), *id)).or_insert(Some(spec.edge_m));
                if *edge != Some(spec.edge_m) {
                    *edge = None; // A phase is required to select this physical size.
                }
            }
        }
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
            families,
            scale,
            min_side_px: profile.min_side_px,
            tag_edges_m,
            corner_refinement: profile.corner_refinement,
            quad_decimate: profile.quad_decimate,
            reacquire_quad_decimate: profile.quad_decimate,
        })
    }

    /// Override quad-search decimation only; edge refinement still uses the
    /// original image coordinates. Recorded by trackd for trial provenance.
    pub fn with_quad_decimate(mut self, decimate: f64) -> Result<Self> {
        if !decimate.is_finite() || !(1.0..=4.0).contains(&decimate) {
            anyhow::bail!("quad decimation must be finite and in [1, 4]");
        }
        self.quad_decimate = decimate;
        Ok(self)
    }

    pub fn quad_decimate(&self) -> f64 {
        self.quad_decimate
    }

    pub fn reacquire_quad_decimate(&self) -> f64 {
        self.reacquire_quad_decimate
    }

    fn build(&self, threads: u8) -> Result<AprilTagDetector> {
        let mut detectors = Vec::new();
        for (family, ids) in &self.families {
            let mut detector = Detector::builder()
                .add_family_bits(tag_family(family)?, 0)
                .build()
                .with_context(|| format!("creating AprilTag detector for {family}"))?;
            detector.set_thread_number(threads);
            detector.set_decimation(self.quad_decimate as f32);
            detector.set_refine_edges(self.corner_refinement);
            detectors.push((family.clone(), detector, ids.clone()));
        }
        Ok(AprilTagDetector {
            detectors,
            scale: self.scale,
            min_side_px: self.min_side_px,
            tag_edges_m: self.tag_edges_m.clone(),
            quad_decimate: self.quad_decimate,
            reacquire_quad_decimate: self.reacquire_quad_decimate,
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
        self.detect_set_with_pose_candidates(calibration, set, excluded_cameras, rois, true)
    }

    /// Tracking can use fresh multiview and recent measured initializers.
    /// The caller must restore candidates immediately on loss or stale history;
    /// periodic full-image searches do not require planar pose estimation.
    pub fn detect_set_with_pose_candidates(
        &self,
        calibration: &CalibrationBundle,
        set: &SynchronizedFrameSet,
        excluded_cameras: &BTreeSet<String>,
        rois: &BTreeMap<String, DetectionRoi>,
        pose_candidates: bool,
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
                WORKER_DETECTOR.with(|cache| {
                    let mut cache = cache.borrow_mut();
                    if cache.as_ref().is_none_or(|(settings, _)| settings != self) {
                        *cache = Some((self.clone(), self.build(1)?));
                    }
                    cache.as_mut().unwrap().1.detect_frame_profiled(
                        camera,
                        frame,
                        rois.get(name).copied(),
                        pose_candidates,
                    )
                })
            })
            .collect::<Vec<_>>();
        let mut detections = Vec::new();
        let mut image_prep_latency_ms = 0.0_f64;
        let mut apriltag_latency_ms = 0.0_f64;
        let mut quad_detection_latency_ms = 0.0_f64;
        let mut pose_candidate_latency_ms = 0.0_f64;
        let mut roi_camera_count = 0_usize;
        for batch in batches {
            let batch = batch?;
            image_prep_latency_ms = image_prep_latency_ms.max(batch.image_prep_latency_ms);
            apriltag_latency_ms = apriltag_latency_ms.max(batch.apriltag_latency_ms);
            quad_detection_latency_ms =
                quad_detection_latency_ms.max(batch.quad_detection_latency_ms);
            pose_candidate_latency_ms =
                pose_candidate_latency_ms.max(batch.pose_candidate_latency_ms);
            roi_camera_count += usize::from(batch.used_roi);
            detections.extend(batch.detections);
        }
        Ok(FiducialDetectionSet {
            detections,
            image_prep_latency_ms,
            apriltag_latency_ms,
            quad_detection_latency_ms,
            pose_candidate_latency_ms,
            roi_camera_count,
        })
    }
}

/// Pixel formats the AprilTag detector reads directly.
fn detectable_format(format: PixelFormat) -> bool {
    matches!(
        format,
        PixelFormat::Bgr8 | PixelFormat::Rgb8 | PixelFormat::Yuyv | PixelFormat::Y8
    )
}

/// Whether a frame can enter the detector at all. A merged set may carry
/// sensors that never hold a tag (an aligned depth plane) or undecoded
/// video; callers exclude those instead of failing the whole set.
pub fn detectable_frame(frame: &crate::FrameRecord) -> bool {
    matches!(
        &frame.payload,
        RecordedPayload::Video { format, .. } if detectable_format(*format)
    )
}

fn describe_payload(payload: &RecordedPayload) -> String {
    match payload {
        RecordedPayload::Video { format, .. } => format!("{format:?} video"),
        RecordedPayload::Encoded { format, .. } => format!("encoded {format:?}"),
        RecordedPayload::Depth { .. } => "a depth plane".into(),
    }
}

impl std::fmt::Debug for AprilTagDetector {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("AprilTagDetector")
            .field("families", &self.detectors.len())
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
        Ok(self
            .detect_frame_profiled(camera, frame, None, true)?
            .detections)
    }

    fn detect_frame_profiled(
        &mut self,
        camera: &CameraCalibration,
        frame: &crate::FrameRecord,
        roi: Option<DetectionRoi>,
        pose_candidates: bool,
    ) -> Result<CameraDetectionSet> {
        let (format, width, height, bytes) = match &frame.payload {
            RecordedPayload::Video {
                format,
                width,
                height,
                bytes,
            } if detectable_format(*format) => (*format, *width as usize, *height as usize, bytes),
            _ => anyhow::bail!(
                "fiducial detection requires decoded BGR/RGB/YUYV/Y8 frames, not {}",
                describe_payload(&frame.payload)
            ),
        };
        // Bytes per pixel of the source; YUYV carries luma in every first byte
        // of each two-byte pair, so it reads exactly like a Y8 plane with a
        // stride of two.
        let channels = match format {
            PixelFormat::Y8 => 1,
            PixelFormat::Yuyv => 2,
            _ => 3,
        };
        anyhow::ensure!(
            width * height * channels == bytes.len(),
            "{format:?} frame {width}x{height} carries {} bytes, expected {}",
            bytes.len(),
            width * height * channels
        );
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
        // Preserve full pixel precision inside a predicted ROI; only reacquisition is downscaled.
        let scale = if used_roi { 1.0 } else { self.scale };
        let crop_width = roi.x1 - roi.x0;
        let crop_height = roi.y1 - roi.y0;
        let prep_started = Instant::now();
        let scaled_width = ((crop_width as f64 * scale).round() as usize).max(1);
        let scaled_height = ((crop_height as f64 * scale).round() as usize).max(1);
        let decimation = if used_roi {
            self.quad_decimate
        } else {
            self.reacquire_quad_decimate
        };
        // AprilTag's C threshold routine assumes at least one 4x4 tile after
        // decimation. A clipped edge ROI can otherwise read before its heap
        // allocation. Reacquire from the full image instead of entering C.
        let minimum_side = (4.0 * decimation).ceil() as usize;
        if scaled_width < minimum_side || scaled_height < minimum_side {
            if used_roi {
                return self.detect_frame_profiled(camera, frame, None, pose_candidates);
            }
            anyhow::bail!(
                "detector image {scaled_width}x{scaled_height} is too small for decimation {decimation} (minimum side {minimum_side})"
            );
        }
        let mut image = Image::zeros_with_stride(scaled_width, scaled_height, scaled_width)
            .context("allocating AprilTag grayscale image")?;
        // Precompute nearest-neighbor source offsets. The former inner-loop
        // floating division dominated detector latency on five 5 MP streams.
        let source_x = (0..scaled_width)
            .map(|x| {
                (roi.x0 + ((x as f64 / scale).floor() as usize).min(crop_width - 1)) * channels
            })
            .collect::<Vec<_>>();
        let source_y = (0..scaled_height)
            .map(|y| {
                (roi.y0 + ((y as f64 / scale).floor() as usize).min(crop_height - 1))
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
                if matches!(format, PixelFormat::Y8 | PixelFormat::Yuyv) {
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
            tagsize: 0.0, // Assigned from the detected target below.
            fx: camera.intrinsics.fx * scale,
            fy: camera.intrinsics.fy * scale,
            cx: (camera.intrinsics.cx - roi.x0 as f64) * scale,
            cy: (camera.intrinsics.cy - roi.y0 as f64) * scale,
        };
        let timestamp_ns = frame
            .metadata
            .timestamps
            .normalized_unix_ns
            .or(frame.metadata.timestamps.source_ns)
            .unwrap_or(frame.metadata.timestamps.host_unix_ns);
        let detector_started = Instant::now();
        let mut output = Vec::new();
        // Reacquisition is already image-downscaled. Do not multiply that
        // reduction by the ROI optimization: preserve the configured search.
        let native_detections: Vec<_> = self
            .detectors
            .iter_mut()
            .flat_map(|(family, detector, ids)| {
                detector.set_decimation(decimation as f32);
                detector
                    .detect(&image)
                    .into_iter()
                    .filter(|detection| ids.contains(&detection.id()))
                    .map(|detection| (family.clone(), detection))
                    .collect::<Vec<_>>()
            })
            .collect();
        let quad_detection_latency_ms = detector_started.elapsed().as_secs_f64() * 1000.0;
        let candidate_started = Instant::now();
        for (family, detection) in native_detections {
            // Normalize AprilTag C's tag-coordinate corner numbering to the
            // TL/TR/BR/BL image contract used by OpenCV and calibration. The
            // exact permutation is locked by live cross-detector parity tests.
            let raw = detection.corners();
            let mut corners_px = [raw[1], raw[0], raw[3], raw[2]];
            for corner in &mut corners_px {
                corner[0] = corner[0] / scale + roi.x0 as f64;
                corner[1] = corner[1] / scale + roi.y0 as f64;
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
            let edge = self.tag_edges_m.get(&(family.clone(), detection.id())).copied().flatten();
            let camera_from_tag_candidates = if let Some(tagsize) = edge.filter(|_| pose_candidates)
            {
                let params = TagParams { tagsize, ..params };
                detection
                    .estimate_tag_pose_orthogonal_iteration(&params, 50)
                    .into_iter()
                    .filter_map(|estimate| isometry_from_apriltag_pose(&estimate.pose).ok())
                    .collect()
            } else {
                Vec::new()
            };
            output.push(FiducialDetection {
                family: Some(family),
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
            quad_detection_latency_ms,
            pose_candidate_latency_ms: candidate_started.elapsed().as_secs_f64() * 1000.0,
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
    pub quad_detection_latency_ms: f64,
    pub pose_candidate_latency_ms: f64,
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

    #[allow(clippy::too_many_arguments)] // Shared live/replay estimator contract.
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

    #[allow(clippy::too_many_arguments)] // Shared live/replay estimator contract.
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
        let detections: Vec<_> = detections.into_iter().filter(|detection| {
            self.layout.ee_from_tag.contains_key(&detection.tag_id)
                && match &detection.family {
                    Some(family) => family == &self.layout.family,
                    None => !self.layout.require_family_ids.contains(&detection.tag_id),
                }
        }).collect();
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
                    quad_detection_latency_ms: 0.0,
                    pose_candidate_latency_ms: 0.0,
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
                    quad_detection_latency_ms: 0.0,
                    pose_candidate_latency_ms: 0.0,
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

    /// Predict detector ROIs from the last measured rigid wrist pose and twist.
    /// Projection uses the same calibrated distortion model as the pose solver.
    /// Missing/expired prediction yields a full-frame reacquisition by callers.
    pub fn predicted_rois(
        &self,
        set: &SynchronizedFrameSet,
        margin_px: usize,
    ) -> BTreeMap<String, DetectionRoi> {
        let Some((stamp, pose)) = &self.last_pose else {
            return BTreeMap::new();
        };
        let age_ms = (set.timestamp_ns - *stamp) as f64 / 1e6;
        if !(0.0..=self.config.prediction_horizon_ms).contains(&age_ms) {
            return BTreeMap::new();
        }
        let predicted = propagate_pose(pose, self.twist, age_ms / 1000.0);
        let mut result = BTreeMap::new();
        for (name, frame) in &set.frames {
            let Some(camera) = self.cameras.get(name) else {
                continue;
            };
            let width = frame.metadata.profile.width as usize;
            let height = frame.metadata.profile.height as usize;
            if (width, height)
                != (
                    camera.calibration.intrinsics.width as usize,
                    camera.calibration.intrinsics.height as usize,
                )
            {
                continue;
            }
            let pixels = self
                .layout
                .ee_from_tag
                .keys()
                .filter_map(|id| self.layout.corners_ee(*id).ok())
                .flatten()
                .filter_map(|corner| camera.project(&predicted.transform_point(&corner)))
                .filter(|pixel| pixel.x.is_finite() && pixel.y.is_finite())
                .collect::<Vec<_>>();
            if pixels.is_empty() {
                continue;
            }
            let min_x = pixels.iter().map(|p| p.x).fold(f64::INFINITY, f64::min) - margin_px as f64;
            let min_y = pixels.iter().map(|p| p.y).fold(f64::INFINITY, f64::min) - margin_px as f64;
            let max_x =
                pixels.iter().map(|p| p.x).fold(f64::NEG_INFINITY, f64::max) + margin_px as f64;
            let max_y =
                pixels.iter().map(|p| p.y).fold(f64::NEG_INFINITY, f64::max) + margin_px as f64;
            let roi = DetectionRoi {
                x0: min_x.floor().clamp(0.0, width as f64) as usize,
                y0: min_y.floor().clamp(0.0, height as f64) as usize,
                x1: max_x.ceil().clamp(0.0, width as f64) as usize,
                y1: max_y.ceil().clamp(0.0, height as f64) as usize,
            };
            if roi.x0 < roi.x1 && roi.y0 < roi.y1 {
                result.insert(name.clone(), roi);
            }
        }
        result
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
        self.twist = world_twist(last, pose, dt);
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
        let initial = self.select_initializer(candidates, detections, timestamp_ns)?;
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

    fn select_initializer(
        &self,
        mut candidates: Vec<Isometry3<f64>>,
        detections: &[FiducialDetection],
        timestamp_ns: i128,
    ) -> Result<Isometry3<f64>> {
        // A coarse triangulated seed may be outside the per-source gate even
        // when its robust nonlinear fit is valid. Try refinement as an extra
        // candidate only when none of the raw seeds has two agreeing sources.
        let has_consensus = candidates.iter().any(|candidate| {
            self.source_rmses(candidate, detections, timestamp_ns)
                .values()
                .filter(|rmse| **rmse <= self.config.max_source_rmse_px)
                .count()
                >= 2
        });
        if !has_consensus {
            let refined = candidates
                .iter()
                .filter_map(|candidate| {
                    self.refine(*candidate, detections, timestamp_ns)
                        .ok()
                        .map(|value| value.0)
                })
                .collect::<Vec<_>>();
            candidates.extend(refined);
        }
        candidates
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
            .context("pose initialization failed")
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

fn world_twist(last: &Isometry3<f64>, pose: &Isometry3<f64>, dt: f64) -> [f64; 6] {
    // apply_delta left-multiplies orientation, so angular velocity must be
    // expressed in world axes, like the translational velocity. Reversing
    // these factors produces a body-frame velocity instead.
    let angular = (pose.rotation * last.rotation.inverse()).scaled_axis() / dt;
    let linear = (pose.translation.vector - last.translation.vector) / dt;
    [
        angular.x, angular.y, angular.z, linear.x, linear.y, linear.z,
    ]
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
    /// False, with a note, in a checkout that omits the deployment's `rel`
    /// (the public export): a test that reads it returns early there.
    fn deployed(rel: &str) -> bool {
        let present = Path::new(env!("CARGO_MANIFEST_DIR")).join(rel).is_file();
        if !present {
            eprintln!("skipped: needs the deployment's {rel}");
        }
        present
    }

    #[test]
    fn detection_log_retains_planar_candidates_and_reads_legacy_rows() {
        let mut value = serde_json::json!({
            "camera": "camera1", "tag_id": 3,
            "corners_px": [[0.0,0.0],[1.0,0.0],[1.0,1.0],[0.0,1.0]],
            "timestamp_ns": 100, "side_px": 20.0,
            "decision_margin": 100.0, "hamming": 0
        });
        let mut detection: FiducialDetection = serde_json::from_value(value.clone()).unwrap();
        assert!(detection.camera_from_tag_candidates.is_empty());
        let pose = Isometry3::new(Vector3::new(0.1, -0.2, 0.7), Vector3::new(0.2, 0.3, -0.1));
        detection.camera_from_tag_candidates = vec![pose, pose.inverse()];
        let encoded = serde_json::to_string(&detection).unwrap();
        let decoded: FiducialDetection = serde_json::from_str(&encoded).unwrap();
        assert_eq!(decoded.camera_from_tag_candidates.len(), 2);
        for (actual, expected) in decoded
            .camera_from_tag_candidates
            .iter()
            .zip([pose, pose.inverse()])
        {
            assert!((actual.to_homogeneous() - expected.to_homogeneous()).norm() < 1e-12);
        }
        let mut invalid = isometry_to_array(&pose);
        invalid[0][0] = 10.0;
        value["camera_from_tag_candidates"] = serde_json::json!([invalid]);
        assert!(serde_json::from_value::<FiducialDetection>(value).is_err());
    }

    #[test]
    fn repository_inventory_and_layout_status_validate() {
        if !deployed("../../config/fiducials.json") {
            return;
        }
        let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
        let inventory = FiducialInventory::load(root.join("config/fiducials.json")).unwrap();
        assert_eq!(inventory.target("wrist").unwrap().ids, [2, 3, 4]);
        assert_eq!(inventory.target("wrist_left").unwrap().ids, [5, 30, 1]);
        let layout = WristLayout::load(
            root.join("config/wrist_tags_measured.json"),
            &inventory,
            true,
        )
        .unwrap();
        assert_eq!(layout.parent_frame, "right/gripper_left");
        let strict = WristLayout::load(
            root.join("config/wrist_tags_measured.json"),
            &inventory,
            false,
        );
        assert_eq!(strict.is_ok(), layout.calibration_status == "calibrated");
    }

    #[test]
    fn material_layout_requires_measured_unique_inventory_target() {
        if !deployed("../../config/fiducials.json") {
            return;
        }
        let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
        let mut inventory = FiducialInventory::load(root.join("config/fiducials.json")).unwrap();
        let mut spec = inventory.target("wrist").unwrap().clone();
        spec.role = "rigid_target".into();
        spec.ids = vec![20, 21];
        spec.parent_frame = Some("paper".into());
        inventory.targets.insert("paper".into(), spec);
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("layout.json");
        let mut raw = serde_json::json!({"schema_version":SUPPORTED_LAYOUT_SCHEMA,"calibration_status":"calibrated",
            "inventory_hash":inventory.inventory_hash,"target_ids":[20,21],"edge_m":inventory.target("paper").unwrap().edge_m,
            "parent_frame":"paper", "tags":{
                "20":{"ee_from_tag":[[1.,0.,0.,0.],[0.,1.,0.,0.],[0.,0.,1.,0.],[0.,0.,0.,1.]]},
                "21":{"ee_from_tag":[[1.,0.,0.,0.1],[0.,1.,0.,0.],[0.,0.,1.,0.],[0.,0.,0.,1.]]}}});
        fs::write(&path, serde_json::to_vec(&raw).unwrap()).unwrap();
        assert!(WristLayout::load_target(&path, &inventory, "paper", false).is_ok());
        inventory.targets.get_mut("wrist").unwrap().ids.push(20);
        assert!(WristLayout::load_target(&path, &inventory, "paper", false).is_err());
        inventory.targets.get_mut("wrist").unwrap().ids.pop();
        raw["calibration_status"] = "pending".into();
        fs::write(&path, serde_json::to_vec(&raw).unwrap()).unwrap();
        assert!(WristLayout::load_target(&path, &inventory, "paper", false).is_err());
    }

    #[test]
    fn detector_roi_expands_clamps_and_ignores_nonfinite_corners() {
        let detection = FiducialDetection {
            family: None,
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
    fn measured_twist_propagates_a_rotated_wrist_in_world_axes() {
        let last = Isometry3::from_parts(
            Translation3::new(0.8, -0.4, 1.2),
            UnitQuaternion::from_euler_angles(1.2, -0.7, 0.9),
        );
        let next = Isometry3::from_parts(
            Translation3::new(0.81, -0.42, 1.23),
            UnitQuaternion::from_euler_angles(0.04, -0.03, 0.02) * last.rotation,
        );
        let dt = 0.05;
        let twist = world_twist(&last, &next, dt);
        let forward = propagate_pose(&last, twist, dt);
        let backward = propagate_pose(&next, twist, -dt);
        assert!((forward.translation.vector - next.translation.vector).norm() < 1e-12);
        assert!((forward.rotation.inverse() * next.rotation).angle() < 1e-12);
        assert!((backward.translation.vector - last.translation.vector).norm() < 1e-12);
        assert!((backward.rotation.inverse() * last.rotation).angle() < 1e-12);
    }

    #[test]
    fn initializer_consensus_gate_drops_a_multi_tag_camera_outlier() {
        let detection = |camera: &str, tag_id| FiducialDetection {
            family: None,
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
    fn yuyv_frames_detect_from_the_luma_plane_and_depth_is_not_detectable() {
        if !deployed("../../config/fiducials.json") {
            return;
        }
        let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
        let inventory = FiducialInventory::load(root.join("config/fiducials.json")).unwrap();
        let luma = image::open(root.join("scripts/tests/fixtures/fiducials/mixed.png"))
            .unwrap()
            .to_luma8();
        // Interleave a flat chroma plane the way a D555 color stream does.
        let yuyv = luma
            .as_raw()
            .iter()
            .flat_map(|&y| [y, 128])
            .collect::<Vec<u8>>();
        let profile = StreamProfile {
            stream: "color".into(),
            width: 1280,
            height: 320,
            fps_num: 30,
            fps_den: 1,
            format: PixelFormat::Yuyv,
        };
        let camera = CameraCalibration {
            sensor_name: "overhead_depth_color".into(),
            profile: profile.clone(),
            intrinsics: Intrinsics {
                width: 1280,
                height: 320,
                fx: 900.,
                fy: 900.,
                cx: 640.,
                cy: 160.,
            },
            distortion: DistortionModel {
                model: "brown_conrady".into(),
                coefficients: vec![0.; 8],
            },
            world_from_camera: Pose {
                rotation: [1., 0., 0., 0., 1., 0., 0., 0., 1.],
                translation_m: [0.; 3],
            },
            depth_to_color: None,
            metadata: BTreeMap::new(),
        };
        let metadata = crate::FrameMetadata {
            sensor_name: camera.sensor_name.clone(),
            sensor_kind: crate::SensorKind::RealSense,
            sequence: 1,
            profile,
            timestamps: crate::FrameTimestamps {
                source_ns: None,
                source_domain: crate::TimestampDomain::HostUnix,
                rtp_timestamp: None,
                pipeline_pts_ns: None,
                pipeline_dts_ns: None,
                host_monotonic_ns: 0,
                host_unix_ns: 123,
                normalized_unix_ns: Some(123),
            },
            dropped_before: 0,
            calibration_id: None,
            flags: Vec::new(),
            attributes: BTreeMap::new(),
        };
        let color = crate::FrameRecord {
            metadata: metadata.clone(),
            payload: RecordedPayload::Video {
                width: 1280,
                height: 320,
                format: PixelFormat::Yuyv,
                bytes: yuyv,
            },
        };
        assert!(detectable_frame(&color));
        let mut detector = AprilTagDetector::new(&inventory, None, Some(1.)).unwrap();
        let found = detector.detect_frame(&camera, &color).unwrap();
        assert_eq!(
            found.iter().map(|d| d.tag_id).collect::<BTreeSet<_>>(),
            BTreeSet::from([3, 5])
        );

        let depth = crate::FrameRecord {
            metadata,
            payload: RecordedPayload::Depth {
                width: 1280,
                height: 320,
                bytes: vec![0; 1280 * 320 * 2],
            },
        };
        assert!(!detectable_frame(&depth));
        let error = detector
            .detect_frame(&camera, &depth)
            .unwrap_err()
            .to_string();
        assert!(error.contains("depth plane"), "{error}");
    }

    #[test]
    fn mixed_family_scene_filters_by_family_and_preserves_rotated_corners() {
        if !deployed("../../config/fiducials.json") {
            return;
        }
        let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
        let inventory = FiducialInventory::load(root.join("config/fiducials.json")).unwrap();
        let original = image::open(root.join("scripts/tests/fixtures/fiducials/mixed.png"))
            .unwrap()
            .to_luma8();
        for turns in 0..4 {
            // Rotate each square in place to test decoded tag-frame ordering.
            let mut canvas = original.clone();
            for col in 0..4 {
                let mut tile =
                    image::imageops::crop_imm(&original, col * 320, 0, 320, 320).to_image();
                for _ in 0..turns {
                    tile = image::imageops::rotate270(&tile);
                }
                image::imageops::replace(&mut canvas, &tile, i64::from(col * 320), 0);
            }
            let camera = CameraCalibration {
                sensor_name: "synthetic".into(),
                profile: StreamProfile {
                    stream: "main".into(),
                    width: 1280,
                    height: 320,
                    fps_num: 20,
                    fps_den: 1,
                    format: PixelFormat::Y8,
                },
                intrinsics: Intrinsics {
                    width: 1280,
                    height: 320,
                    fx: 900.,
                    fy: 900.,
                    cx: 640.,
                    cy: 160.,
                },
                distortion: DistortionModel {
                    model: "brown_conrady".into(),
                    coefficients: vec![0.; 8],
                },
                world_from_camera: Pose {
                    rotation: [1., 0., 0., 0., 1., 0., 0., 0., 1.],
                    translation_m: [0.; 3],
                },
                depth_to_color: None,
                metadata: BTreeMap::new(),
            };
            let frame = crate::FrameRecord {
                metadata: crate::FrameMetadata {
                    sensor_name: camera.sensor_name.clone(),
                    sensor_kind: crate::SensorKind::PoE,
                    sequence: 1,
                    profile: camera.profile.clone(),
                    timestamps: crate::FrameTimestamps {
                        source_ns: None,
                        source_domain: crate::TimestampDomain::HostUnix,
                        rtp_timestamp: None,
                        pipeline_pts_ns: None,
                        pipeline_dts_ns: None,
                        host_monotonic_ns: 0,
                        host_unix_ns: 123,
                        normalized_unix_ns: Some(123),
                    },
                    dropped_before: 0,
                    calibration_id: None,
                    flags: Vec::new(),
                    attributes: BTreeMap::new(),
                },
                payload: RecordedPayload::Video {
                    width: 1280,
                    height: 320,
                    format: PixelFormat::Y8,
                    bytes: canvas.into_raw(),
                },
            };
            // Each physical arm's triplet is its own target: a pink sighting
            // never counts for blue and vice versa.
            for (target, ids) in [
                (None, vec![3, 5]),
                (Some("board"), vec![3]),
                (Some("wrist"), vec![3]),
                (Some("wrist_left"), vec![5]),
            ] {
                let mut detector = AprilTagDetector::new(&inventory, target, Some(1.)).unwrap();
                let found = detector.detect_frame(&camera, &frame).unwrap();
                assert_eq!(
                    found.iter().map(|d| d.tag_id).collect::<BTreeSet<_>>(),
                    ids.into_iter().collect()
                );
                for detection in found {
                    let x = match (detection.family.as_deref(), detection.tag_id) {
                        (Some("apriltag_16h5"), 3) => 40.,
                        (Some("apriltag_36h11"), 3) => 360.,
                        _ => 1000.,
                    };
                    let expected = [[x, 40.], [x + 239., 40.], [x + 239., 279.], [x, 279.]];
                    for (index, actual) in detection.corners_px.iter().enumerate() {
                        let expected = expected[(index + 4 - turns) % 4];
                        assert!((actual[0] - expected[0]).hypot(actual[1] - expected[1]) < 2.);
                    }
                    let edge = if detection.family.as_deref() == Some("apriltag_16h5") { 0.044 } else { 0.047 };
                    assert!(!detection.camera_from_tag_candidates.is_empty());
                    assert!(
                        detection.camera_from_tag_candidates.iter().any(|pose| (pose
                            .translation
                            .z
                            - 900. * edge / 240.)
                            .abs()
                            < 0.002)
                    );
                }
            }
        }
    }

    #[test]
    fn decimated_quad_search_preserves_refined_tag_corners() {
        if !deployed("../../config/fiducials.json") {
            return;
        }
        let inventory = FiducialInventory::load(
            Path::new(env!("CARGO_MANIFEST_DIR")).join("../../config/fiducials.json"),
        )
        .unwrap();
        let factory = AprilTagDetectorFactory::new(&inventory, Some("board"), Some(1.0)).unwrap();
        let tag = image::open(
            Path::new(env!("CARGO_MANIFEST_DIR"))
                .join("../../urdf/meshes/tags/16h5_008_56mm/tag.png"),
        )
        .unwrap()
        .to_luma8();
        for side in [24, 36, 72, 80, 120, 240] {
            let tag =
                image::imageops::resize(&tag, side, side, image::imageops::FilterType::Nearest);
            let mut canvas = image::GrayImage::from_pixel(side + 80, side + 80, image::Luma([255]));
            image::imageops::replace(&mut canvas, &tag, 40, 40);
            let mut input = Image::zeros_with_stride(
                canvas.width() as usize,
                canvas.height() as usize,
                canvas.width() as usize,
            )
            .unwrap();
            input.as_slice_mut().copy_from_slice(canvas.as_raw());
            let mut reference = factory
                .clone()
                .with_quad_decimate(1.0)
                .unwrap()
                .build(1)
                .unwrap();
            let mut optimized = factory
                .clone()
                .with_quad_decimate(2.0)
                .unwrap()
                .build(1)
                .unwrap();
            let a = reference.detectors[0].1.detect(&input);
            if side < 80 {
                optimized.detectors[0]
                    .1
                    .set_decimation(optimized.reacquire_quad_decimate as f32);
            }
            let b = optimized.detectors[0].1.detect(&input);
            assert_eq!(a.len(), 1, "reference side={side}");
            assert_eq!(b.len(), 1, "decimated side={side}");
            assert_eq!(a[0].id(), b[0].id());
            for (a, b) in a[0].corners().iter().zip(b[0].corners()) {
                assert!(
                    (a[0] - b[0]).hypot(a[1] - b[1]) < 0.5,
                    "corner changed at side={side}"
                );
            }
        }
        assert!(factory.clone().with_quad_decimate(f64::NAN).is_err());
        assert!(factory.with_quad_decimate(0.5).is_err());
    }
    #[test]
    fn synthetic_multicamera_tracker_recovers_ee_pose() {
        if !deployed("../../config/fiducials.json") {
            return;
        }
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
            family: "apriltag_16h5".into(),
            detector: BTreeMap::new(),
            targets: BTreeMap::from([(
                "wrist".into(),
                TargetSpec {
                    family: "apriltag_16h5".into(),
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
            family: "apriltag_16h5".into(),
            require_family_ids: BTreeSet::new(),
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
                    family: None,
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
        // A coarse seed outside every source gate must get a chance to
        // converge; the original seed remains among the ranked candidates.
        let coarse = apply_delta(&truth, [0.04, -0.03, 0.02, 0.04, -0.03, 0.02]);
        assert!(
            tracker
                .source_rmses(&coarse, &detections, 1_000_000_000)
                .values()
                .all(|rmse| *rmse > tracker.config.max_source_rmse_px)
        );
        let refined = tracker
            .select_initializer(vec![coarse], &detections, 1_000_000_000)
            .unwrap();
        assert!((refined.translation.vector - truth.translation.vector).norm() < 1e-6);
        assert!((refined.rotation.inverse() * truth.rotation).angle() < 1e-6);
        // An already valid initializer is not changed by the fallback.
        let exact = tracker
            .select_initializer(vec![truth], &detections, 1_000_000_000)
            .unwrap();
        assert_eq!(exact, truth);
        let wrong_family = detections.iter().cloned().map(|mut detection| {
            detection.family = Some("apriltag_36h11".into());
            detection
        }).collect();
        let rejected = tracker.update(3, 600_000_000, 0, wrong_family, 0.0, 0.0, Instant::now());
        assert_eq!(rejected.status, "unavailable");
        assert!(rejected.reason.unwrap().contains("no configured wrist tags"));
        tracker.layout.require_family_ids = BTreeSet::from([3, 6]);
        let rejected = tracker.update(4, 700_000_000, 0, detections.clone(), 0.0, 0.0, Instant::now());
        assert_eq!(rejected.status, "unavailable");
        assert!(rejected.reason.unwrap().contains("no configured wrist tags"));
        tracker.layout.require_family_ids.clear();
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
        // ROI projection must contain the actual solved corners, and expire
        // with the pose prediction rather than hiding loss indefinitely.
        let mut set = SynchronizedFrameSet {
            sequence: 7,
            timestamp_basis: "test".into(),
            timestamp_ns: 1_000_000_000,
            maximum_skew_ns: 0,
            frames: BTreeMap::new(),
        };
        for (name, camera) in &calibration.cameras {
            let metadata = crate::FrameMetadata {
                sensor_name: name.clone(),
                sensor_kind: crate::SensorKind::PoE,
                sequence: 7,
                profile: camera.profile.clone(),
                timestamps: crate::FrameTimestamps {
                    source_ns: None,
                    source_domain: crate::TimestampDomain::HostUnix,
                    rtp_timestamp: None,
                    pipeline_pts_ns: None,
                    pipeline_dts_ns: None,
                    host_monotonic_ns: 0,
                    host_unix_ns: 1_000_000_000,
                    normalized_unix_ns: Some(1_000_000_000),
                },
                dropped_before: 0,
                calibration_id: Some(calibration.bundle_id.clone()),
                flags: Vec::new(),
                attributes: BTreeMap::new(),
            };
            set.frames.insert(
                name.clone(),
                crate::FrameRecord {
                    metadata,
                    payload: RecordedPayload::Encoded {
                        format: crate::PixelFormat::H264,
                        bytes: vec![0],
                    },
                },
            );
        }
        // Exercise the native detector boundary with the same calibrated
        // frame metadata. Tiny clipped ROIs must reacquire, never enter the
        // C threshold routine with fewer than one decimated 4x4 tile.
        let detector_inventory = FiducialInventory::load(
            Path::new(env!("CARGO_MANIFEST_DIR")).join("../../config/fiducials.json"),
        )
        .unwrap();
        let mut detector =
            AprilTagDetectorFactory::new(&detector_inventory, Some("wrist"), Some(1.0))
                .unwrap()
                .with_quad_decimate(2.0)
                .unwrap()
                .build(1)
                .unwrap();
        let camera = &calibration.cameras["cam_left"];
        let mut frame = set.frames["cam_left"].clone();
        frame.payload = RecordedPayload::Video {
            format: PixelFormat::Y8,
            width: 32,
            height: 32,
            bytes: vec![128; 32 * 32],
        };
        for (x1, y1) in [(3, 32), (32, 3), (3, 3)] {
            let result = detector
                .detect_frame_profiled(
                    camera,
                    &frame,
                    Some(DetectionRoi {
                        x0: 0,
                        y0: 0,
                        x1,
                        y1,
                    }),
                    false,
                )
                .unwrap();
            assert!(!result.used_roi);
            assert!(result.detections.is_empty());
        }
        frame.payload = RecordedPayload::Video {
            format: PixelFormat::Y8,
            width: 3,
            height: 32,
            bytes: vec![128; 3 * 32],
        };
        assert!(
            detector
                .detect_frame_profiled(camera, &frame, None, false)
                .is_err()
        );
        let rois = tracker.predicted_rois(&set, 20);
        assert_eq!(rois.len(), 2);
        for (name, detections) in &estimate.detections {
            for detection in detections {
                for [x, y] in detection.corners_px {
                    let roi = rois[name];
                    assert!(
                        x >= roi.x0 as f64
                            && x < roi.x1 as f64
                            && y >= roi.y0 as f64
                            && y < roi.y1 as f64
                    );
                }
            }
        }
        set.timestamp_ns += 300_000_000;
        assert!(tracker.predicted_rois(&set, 20).is_empty());

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
