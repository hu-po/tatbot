//! Versioned camera calibration artifacts.
//!
//! Calibration is deliberately an input to the capture contract, not hidden
//! state inside a reconstruction consumer. A frame can therefore be rejected
//! or routed to a lower-confidence path when its active profile has no matching
//! calibration.

use std::{collections::BTreeMap, fs, path::Path};

use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

use crate::{PixelFormat, StreamProfile};

pub const CALIBRATION_SCHEMA_VERSION: u32 = 1;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CalibrationBundle {
    pub schema_version: u32,
    /// SHA-256 of this bundle with `bundle_id` set to the empty string.
    pub bundle_id: String,
    pub world_frame: String,
    pub cameras: BTreeMap<String, CameraCalibration>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CameraCalibration {
    pub sensor_name: String,
    pub profile: StreamProfile,
    pub intrinsics: Intrinsics,
    pub distortion: DistortionModel,
    /// Rigid transform from camera coordinates into `world_frame`.
    pub world_from_camera: Pose,
    pub depth_to_color: Option<Pose>,
    #[serde(default)]
    pub metadata: BTreeMap<String, String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Intrinsics {
    pub width: u32,
    pub height: u32,
    pub fx: f64,
    pub fy: f64,
    pub cx: f64,
    pub cy: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DistortionModel {
    pub model: String,
    pub coefficients: Vec<f64>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Pose {
    /// Row-major 3x3 rotation matrix.
    pub rotation: [f64; 9],
    /// Translation in metres.
    pub translation_m: [f64; 3],
}

impl CalibrationBundle {
    pub fn load(path: impl AsRef<Path>) -> Result<Self> {
        let path = path.as_ref();
        let text = fs::read_to_string(path)
            .with_context(|| format!("reading calibration bundle {}", path.display()))?;
        let bundle: Self = serde_json::from_str(&text)
            .with_context(|| format!("parsing calibration bundle {}", path.display()))?;
        bundle.validate().map_err(anyhow::Error::msg)?;
        Ok(bundle)
    }

    pub fn write(&self, path: impl AsRef<Path>) -> Result<()> {
        self.validate().map_err(anyhow::Error::msg)?;
        let path = path.as_ref();
        fs::write(path, serde_json::to_vec_pretty(self)?)
            .with_context(|| format!("writing calibration bundle {}", path.display()))?;
        Ok(())
    }

    pub fn with_computed_id(mut self) -> Result<Self> {
        self.bundle_id = self.computed_id()?;
        self.validate().map_err(anyhow::Error::msg)?;
        Ok(self)
    }

    pub fn computed_id(&self) -> Result<String> {
        let mut canonical = self.clone();
        canonical.bundle_id.clear();
        let bytes = serde_json::to_vec(&canonical)?;
        let digest = Sha256::digest(bytes);
        Ok(hex::encode(digest))
    }

    pub fn validate(&self) -> Result<(), String> {
        if self.schema_version != CALIBRATION_SCHEMA_VERSION {
            return Err(format!(
                "unsupported calibration schema {}",
                self.schema_version
            ));
        }
        if self.bundle_id.len() != 64
            || !self
                .bundle_id
                .chars()
                .all(|character| character.is_ascii_hexdigit())
        {
            return Err("calibration bundle_id must be a SHA-256 hex string".into());
        }
        if self.world_frame.trim().is_empty() {
            return Err("calibration world_frame must not be empty".into());
        }
        if self.cameras.is_empty() {
            return Err("calibration bundle must contain at least one camera".into());
        }
        if self.computed_id().map_err(|error| error.to_string())? != self.bundle_id {
            return Err("calibration bundle_id does not match bundle contents".into());
        }
        for (name, camera) in &self.cameras {
            if name != &camera.sensor_name {
                return Err(format!(
                    "calibration camera key does not match {}",
                    camera.sensor_name
                ));
            }
            camera.profile.validate()?;
            if camera.profile.width != camera.intrinsics.width
                || camera.profile.height != camera.intrinsics.height
            {
                return Err(format!(
                    "{name} calibration profile and intrinsics dimensions differ"
                ));
            }
            camera.intrinsics.validate(name)?;
            camera.distortion.validate(name)?;
            camera.world_from_camera.validate(name)?;
            if let Some(depth_to_color) = &camera.depth_to_color {
                depth_to_color.validate(name)?;
            }
        }
        Ok(())
    }

    pub fn camera(
        &self,
        sensor_name: &str,
        profile: &StreamProfile,
    ) -> Result<&CameraCalibration, String> {
        let camera = self
            .cameras
            .get(sensor_name)
            .ok_or_else(|| format!("no calibration for sensor {sensor_name}"))?;
        let geometry_matches = camera.profile.stream == profile.stream
            && camera.profile.width == profile.width
            && camera.profile.height == profile.height
            && camera.profile.fps_num == profile.fps_num
            && camera.profile.fps_den == profile.fps_den;
        // A decoded frame keeps the encoded stream's exact optical geometry.
        // Treat H264 -> decoded color/luma as the same calibrated profile while keeping
        // all stream, dimension and frame-rate checks strict.
        let format_matches = camera.profile.format == profile.format
            || matches!(
                (camera.profile.format, profile.format),
                (
                    PixelFormat::H264,
                    PixelFormat::Bgr8 | PixelFormat::Rgb8 | PixelFormat::Y8
                )
            );
        if !geometry_matches || !format_matches {
            return Err(format!(
                "calibration profile mismatch for {sensor_name}: calibrated {}:{}x{}@{}/{} {:?}, active {}:{}x{}@{}/{} {:?}",
                camera.profile.stream,
                camera.profile.width,
                camera.profile.height,
                camera.profile.fps_num,
                camera.profile.fps_den,
                camera.profile.format,
                profile.stream,
                profile.width,
                profile.height,
                profile.fps_num,
                profile.fps_den,
                profile.format,
            ));
        }
        Ok(camera)
    }
}

impl Intrinsics {
    fn validate(&self, name: &str) -> Result<(), String> {
        if self.width == 0 || self.height == 0 {
            return Err(format!("{name} intrinsics dimensions must be positive"));
        }
        for (label, value) in [
            ("fx", self.fx),
            ("fy", self.fy),
            ("cx", self.cx),
            ("cy", self.cy),
        ] {
            if !value.is_finite() || value <= 0.0 && (label == "fx" || label == "fy") {
                return Err(format!("{name} intrinsics {label} is invalid"));
            }
        }
        Ok(())
    }
}

impl DistortionModel {
    fn validate(&self, name: &str) -> Result<(), String> {
        if self.model.trim().is_empty() || self.coefficients.len() > 32 {
            return Err(format!("{name} distortion model is invalid"));
        }
        if self.coefficients.iter().any(|value| !value.is_finite()) {
            return Err(format!("{name} distortion has non-finite coefficients"));
        }
        Ok(())
    }
}

impl Pose {
    fn validate(&self, name: &str) -> Result<(), String> {
        if self.rotation.iter().any(|value| !value.is_finite())
            || self.translation_m.iter().any(|value| !value.is_finite())
        {
            return Err(format!("{name} pose has non-finite values"));
        }
        let rows = [
            [self.rotation[0], self.rotation[1], self.rotation[2]],
            [self.rotation[3], self.rotation[4], self.rotation[5]],
            [self.rotation[6], self.rotation[7], self.rotation[8]],
        ];
        for row in rows {
            let norm = row.iter().map(|value| value * value).sum::<f64>().sqrt();
            if (norm - 1.0).abs() > 0.02 {
                return Err(format!("{name} pose rotation is not normalized"));
            }
        }
        let dot = |left: &[f64; 3], right: &[f64; 3]| {
            left.iter()
                .zip(right.iter())
                .map(|(a, b)| a * b)
                .sum::<f64>()
        };
        if dot(&rows[0], &rows[1]).abs() > 0.02
            || dot(&rows[0], &rows[2]).abs() > 0.02
            || dot(&rows[1], &rows[2]).abs() > 0.02
        {
            return Err(format!("{name} pose rotation is not orthogonal"));
        }
        let determinant = rows[0][0] * (rows[1][1] * rows[2][2] - rows[1][2] * rows[2][1])
            - rows[0][1] * (rows[1][0] * rows[2][2] - rows[1][2] * rows[2][0])
            + rows[0][2] * (rows[1][0] * rows[2][1] - rows[1][1] * rows[2][0]);
        if (determinant - 1.0).abs() > 0.05 {
            return Err(format!("{name} pose rotation determinant is invalid"));
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn bundle() -> CalibrationBundle {
        CalibrationBundle {
            schema_version: CALIBRATION_SCHEMA_VERSION,
            bundle_id: String::new(),
            world_frame: "fixture".into(),
            cameras: BTreeMap::from([(
                "camera1".into(),
                CameraCalibration {
                    sensor_name: "camera1".into(),
                    profile: StreamProfile {
                        stream: "main".into(),
                        width: 1920,
                        height: 1080,
                        fps_num: 15,
                        fps_den: 1,
                        format: PixelFormat::H264,
                    },
                    intrinsics: Intrinsics {
                        width: 1920,
                        height: 1080,
                        fx: 1000.0,
                        fy: 1000.0,
                        cx: 960.0,
                        cy: 540.0,
                    },
                    distortion: DistortionModel {
                        model: "brown_conrady".into(),
                        coefficients: vec![0.0; 5],
                    },
                    world_from_camera: Pose {
                        rotation: [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
                        translation_m: [0.0, 0.0, 0.0],
                    },
                    depth_to_color: None,
                    metadata: BTreeMap::new(),
                },
            )]),
        }
    }

    #[test]
    fn computes_and_validates_content_id() {
        let bundle = bundle().with_computed_id().unwrap();
        assert_eq!(bundle.computed_id().unwrap(), bundle.bundle_id);
        bundle.validate().unwrap();
    }

    #[test]
    fn rejects_tampered_bundle_id() {
        let mut bundle = bundle().with_computed_id().unwrap();
        bundle.world_frame = "tampered".into();
        assert!(bundle.validate().is_err());
    }

    #[test]
    fn rejects_non_rigid_rotation() {
        let mut bundle = bundle();
        bundle
            .cameras
            .get_mut("camera1")
            .unwrap()
            .world_from_camera
            .rotation[1] = 0.2;
        assert!(bundle.with_computed_id().is_err());
    }

    #[test]
    fn accepts_decoded_pixels_from_the_calibrated_encoded_stream() {
        let bundle = bundle();
        let mut decoded = bundle.cameras["camera1"].profile.clone();
        decoded.format = PixelFormat::Bgr8;
        assert!(bundle.camera("camera1", &decoded).is_ok());
        decoded.format = PixelFormat::Rgb8;
        assert!(bundle.camera("camera1", &decoded).is_ok());
        decoded.format = PixelFormat::Y8;
        assert!(bundle.camera("camera1", &decoded).is_ok());
    }

    #[test]
    fn decoded_profile_still_rejects_geometry_changes() {
        let bundle = bundle();
        let mut decoded = bundle.cameras["camera1"].profile.clone();
        decoded.format = PixelFormat::Bgr8;
        decoded.width /= 2;
        assert!(bundle.camera("camera1", &decoded).is_err());
    }
}
