use std::{collections::HashSet, net::IpAddr, path::Path};

use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};

use crate::{PixelFormat, SensorKind, StreamProfile};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VisionConfig {
    pub schema_version: u32,
    #[serde(default)]
    pub session: SessionConfig,
    pub sync: SyncConfig,
    pub cameras: CamerasConfig,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SessionConfig {
    #[serde(default = "default_record_root")]
    pub record_root: String,
    #[serde(default = "default_queue_capacity")]
    pub queue_capacity: usize,
}

impl Default for SessionConfig {
    fn default() -> Self {
        Self {
            record_root: default_record_root(),
            queue_capacity: default_queue_capacity(),
        }
    }
}

fn default_record_root() -> String {
    "/tmp/tatbot-vision-recordings".to_string()
}

fn default_queue_capacity() -> usize {
    8
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SyncConfig {
    // No baked-in default (plan Phase 3): the NTP host is deployment data,
    // stated in the registry file.
    pub ntp_server: String,
    #[serde(default = "default_max_pairwise_skew_ms")]
    pub max_pairwise_skew_ms: f64,
    #[serde(default = "default_max_clock_drift_ppm")]
    pub max_clock_drift_ppm: f64,
}

fn default_max_pairwise_skew_ms() -> f64 {
    20.0
}

fn default_max_clock_drift_ppm() -> f64 {
    100.0
}

// Both camera kinds are optional (plan Phase 3): a registry may describe
// one camera, many, or no depth cameras at all, without code changes.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CamerasConfig {
    #[serde(default)]
    pub poe: Vec<PoeCameraConfig>,
    #[serde(default)]
    pub realsense: Vec<RealSenseConfig>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PoeCameraConfig {
    pub name: String,
    pub address: IpAddr,
    #[serde(default = "default_rtsp_port")]
    pub rtsp_port: u16,
    #[serde(default = "default_http_port")]
    pub http_port: u16,
    #[serde(default = "default_username")]
    pub username: String,
    pub password_env: String,
    pub main: StreamProfile,
    pub sub: Option<StreamProfile>,
    #[serde(default = "default_transport")]
    pub transport: String,
    #[serde(default = "default_gstreamer_latency_ms")]
    pub gstreamer_latency_ms: u32,
}

fn default_rtsp_port() -> u16 {
    554
}

fn default_http_port() -> u16 {
    80
}

fn default_username() -> String {
    "admin".to_string()
}

fn default_transport() -> String {
    "tcp".to_string()
}

fn default_gstreamer_latency_ms() -> u32 {
    200
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RealSenseConfig {
    pub name: String,
    pub serial: String,
    pub color: StreamProfile,
    pub depth: StreamProfile,
    #[serde(default = "default_queue_capacity")]
    pub queue_capacity: usize,
}

pub trait CameraConfig {
    fn name(&self) -> &str;
    fn kind(&self) -> SensorKind;
}

impl CameraConfig for PoeCameraConfig {
    fn name(&self) -> &str {
        &self.name
    }

    fn kind(&self) -> SensorKind {
        SensorKind::PoE
    }
}

impl CameraConfig for RealSenseConfig {
    fn name(&self) -> &str {
        &self.name
    }

    fn kind(&self) -> SensorKind {
        SensorKind::RealSense
    }
}

impl VisionConfig {
    pub fn load(path: impl AsRef<Path>) -> Result<Self> {
        let path = path.as_ref();
        let text = std::fs::read_to_string(path)
            .with_context(|| format!("reading vision config {}", path.display()))?;
        let mut config: Self = toml::from_str(&text)
            .with_context(|| format!("parsing vision config {}", path.display()))?;
        // Precedence contract (plan Phase 1): env beats file config.
        if let Ok(root) = std::env::var("TATBOT_VISIOND_RECORD_ROOT") {
            if !root.is_empty() {
                config.session.record_root = root;
            }
        }
        config.validate().map_err(anyhow::Error::msg)?;
        Ok(config)
    }

    pub fn validate(&self) -> Result<(), String> {
        if self.schema_version != 1 {
            return Err(format!(
                "unsupported vision config schema {}",
                self.schema_version
            ));
        }
        if self.session.queue_capacity == 0 {
            return Err("session.queue_capacity must be positive".into());
        }
        if !(0.0..=1000.0).contains(&self.sync.max_pairwise_skew_ms)
            || self.sync.max_pairwise_skew_ms == 0.0
        {
            return Err("sync.max_pairwise_skew_ms must be positive and <= 1000".into());
        }
        if self.sync.max_clock_drift_ppm <= 0.0 {
            return Err("sync.max_clock_drift_ppm must be positive".into());
        }

        let mut names = HashSet::new();
        for camera in &self.cameras.poe {
            validate_name(&camera.name)?;
            if !names.insert(camera.name.clone()) {
                return Err(format!("duplicate camera name {}", camera.name));
            }
            if camera.password_env.trim().is_empty() {
                return Err(format!("{} password_env must not be empty", camera.name));
            }
            if camera.transport != "tcp" && camera.transport != "udp" {
                return Err(format!("{} transport must be tcp or udp", camera.name));
            }
            camera.main.validate()?;
            if camera.main.stream != "color" && camera.main.stream != "main" {
                return Err(format!("{} main stream must be color or main", camera.name));
            }
            if let Some(sub) = &camera.sub {
                sub.validate()?;
            }
        }

        for camera in &self.cameras.realsense {
            validate_name(&camera.name)?;
            if !names.insert(camera.name.clone()) {
                return Err(format!("duplicate camera name {}", camera.name));
            }
            if camera.serial.trim().is_empty() {
                return Err(format!("{} serial must not be empty", camera.name));
            }
            if camera.color.stream != "color" || camera.depth.stream != "depth" {
                return Err(format!(
                    "{} must define color and depth streams",
                    camera.name
                ));
            }
            camera.color.validate()?;
            camera.depth.validate()?;
            if camera.queue_capacity == 0 {
                return Err(format!("{} queue_capacity must be positive", camera.name));
            }
            if camera.color.format == PixelFormat::Z16 || camera.depth.format != PixelFormat::Z16 {
                return Err(format!(
                    "{} depth must be Z16 and color must not be depth",
                    camera.name
                ));
            }
        }
        Ok(())
    }

    pub fn sensor_names(&self) -> impl Iterator<Item = &str> {
        self.cameras
            .poe
            .iter()
            .map(|camera| camera.name.as_str())
            .chain(
                self.cameras
                    .realsense
                    .iter()
                    .map(|camera| camera.name.as_str()),
            )
    }
}

fn validate_name(name: &str) -> Result<(), String> {
    if name.trim().is_empty() {
        return Err("camera name must not be empty".into());
    }
    if name.chars().any(|character| {
        !(character.is_ascii_alphanumeric() || character == '_' || character == '-')
    }) {
        return Err(format!("invalid camera name {name}"));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    // Phase 3 exit gate: the public example registry loads, and the registry
    // flexes to one camera / no depth camera without code changes.
    #[test]
    fn example_registry_loads_and_validates() {
        let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("config/vision.example.toml");
        let config = VisionConfig::load(path).expect("example config must load");
        assert!(!config.cameras.poe.is_empty());
        assert!(!config.cameras.realsense.is_empty());
    }

    #[test]
    fn registry_accepts_one_camera_and_no_depth() {
        let toml = r#"
            schema_version = 1
            [sync]
            ntp_server = "192.0.2.123"
            [[cameras.poe]]
            name = "solo"
            address = "192.0.2.10"
            password_env = "CAM_PW"
            [cameras.poe.main]
            stream = "main"
            width = 1920
            height = 1080
            fps_num = 30
            fps_den = 1
            format = "h264"
            [cameras.poe.sub]
            stream = "sub"
            width = 640
            height = 360
            fps_num = 30
            fps_den = 1
            format = "h264"
        "#;
        let config: VisionConfig = toml::from_str(toml).expect("one-camera registry parses");
        config.validate().expect("one-camera registry validates");
        assert_eq!(config.cameras.poe.len(), 1);
        assert!(config.cameras.realsense.is_empty());
    }
}
