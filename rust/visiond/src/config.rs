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

#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SensorControls {
    pub auto_exposure: Option<bool>,
    pub exposure_us: Option<f32>,
    pub gain: Option<f32>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RealSenseConfig {
    pub name: String,
    pub serial: String,
    /// Stable physical use (for example `wrist_lower` or `surface_overhead`).
    #[serde(default)]
    pub role: String,
    /// Capture/publication group. Devices in different groups are never opened
    /// or synchronized by the same owner process.
    #[serde(default)]
    pub group: String,
    /// Physical mounting arm, independent of a session's teleop role. Omitted
    /// means unassigned; capture is possible but arm-specific consumers refuse.
    #[serde(default)]
    pub arm: Option<String>,
    /// Fleet role owning this device's USB capture, resolved by the launcher.
    #[serde(default)]
    pub owner_role: Option<String>,
    /// Optional device-side depth controls. State them for a fixed installation
    /// after measuring the actual scene; omitted values leave device defaults.
    #[serde(default)]
    pub visual_preset: Option<f32>,
    #[serde(default)]
    pub laser_power: Option<f32>,
    /// Optional measured metres per Z16 unit for transports that do not expose
    /// librealsense's depth-scale option (the D555 DDS transport currently
    /// does not). Omit when the backend can obtain it from the device.
    #[serde(default)]
    pub depth_units_m: Option<f32>,
    /// How librealsense reaches the device. Omitted means USB.
    #[serde(default)]
    pub transport: RealSenseTransport,
    /// The owner's local address a DDS camera's discovery is bound to. The
    /// launcher supplies it (`--dds-address`); the manifest never does.
    #[serde(skip)]
    pub dds_address: Option<IpAddr>,
    #[serde(default)]
    pub depth_controls: SensorControls,
    #[serde(default)]
    pub color_controls: SensorControls,
    pub color: StreamProfile,
    pub depth: StreamProfile,
    #[serde(default = "default_queue_capacity")]
    pub queue_capacity: usize,
}

/// How librealsense reaches a RealSense device.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum RealSenseTransport {
    /// USB, found by the SDK's default device enumeration (the wrist D405s).
    #[default]
    Usb,
    /// Ethernet, found by DDS discovery (the D555). Needs a librealsense built
    /// with DDS and an owner address to bind discovery to.
    Dds,
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

        let mut serials = HashSet::new();
        for camera in &self.cameras.realsense {
            validate_name(&camera.name)?;
            if !names.insert(camera.name.clone()) {
                return Err(format!("duplicate camera name {}", camera.name));
            }
            if camera.serial.trim().is_empty() {
                return Err(format!("{} serial must not be empty", camera.name));
            }
            if !serials.insert(&camera.serial) {
                return Err(format!("duplicate RealSense serial {}", camera.serial));
            }
            if camera.arm.as_deref().is_some_and(|arm| {
                arm.len() > 64
                    || !arm.bytes().next().is_some_and(|c| c.is_ascii_alphabetic())
                    || !arm
                        .bytes()
                        .all(|c| c.is_ascii_alphanumeric() || c == b'-' || c == b'_')
            }) {
                return Err(format!("{} has an invalid arm ID", camera.name));
            }
            if let Some(role) = &camera.owner_role {
                validate_name(role)?;
            }
            if !camera.role.is_empty() {
                validate_name(&camera.role)?;
            }
            if !camera.group.is_empty() {
                validate_name(&camera.group)?;
            }
            for (name, value) in [
                ("visual_preset", camera.visual_preset),
                ("laser_power", camera.laser_power),
                ("depth_units_m", camera.depth_units_m),
            ] {
                if value.is_some_and(|value| !value.is_finite()) {
                    return Err(format!("{} {name} must be finite", camera.name));
                }
            }
            if camera.depth_units_m.is_some_and(|value| value <= 0.0) {
                return Err(format!("{} depth_units_m must be positive", camera.name));
            }
            for controls in [&camera.depth_controls, &camera.color_controls] {
                if controls
                    .exposure_us
                    .is_some_and(|v| !v.is_finite() || v <= 0.0)
                    || controls.gain.is_some_and(|v| !v.is_finite() || v < 0.0)
                    || (controls.exposure_us.is_some() && controls.auto_exposure != Some(false))
                {
                    return Err(format!(
                        "{} invalid sensor controls; manual exposure requires auto_exposure=false",
                        camera.name
                    ));
                }
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

    /// Validate the complete selection before replacing the camera list. A
    /// misspelled, duplicate or cross-group request never falls back to all.
    pub fn select_realsense(
        &mut self,
        group: Option<&str>,
        sensors: &[String],
    ) -> Result<(), String> {
        let candidates: Vec<_> = self
            .cameras
            .realsense
            .iter()
            .filter(|camera| group.is_none_or(|group| camera.group == group))
            .collect();
        let mut requested = HashSet::new();
        for sensor in sensors {
            if !requested.insert(sensor) {
                return Err(format!("duplicate RealSense selection {sensor}"));
            }
            if !candidates.iter().any(|camera| camera.name == *sensor) {
                return Err(format!(
                    "RealSense {sensor} is absent from the selected capture group"
                ));
            }
        }
        let selected: Vec<_> = candidates
            .into_iter()
            .filter(|camera| sensors.is_empty() || requested.contains(&camera.name))
            .cloned()
            .collect();
        if selected.is_empty() {
            return Err("capture-realsense-all selected no RealSense cameras".into());
        }
        self.cameras.realsense = selected;
        Ok(())
    }

    /// Give every selected DDS camera the owner's camera-LAN address. A DDS
    /// camera without one, or an address with no DDS camera to bind, refuses:
    /// discovery on every interface can answer on the wrong route.
    pub fn bind_dds_address(&mut self, address: Option<IpAddr>) -> Result<(), String> {
        let dds = |camera: &&mut RealSenseConfig| camera.transport == RealSenseTransport::Dds;
        let mut bound = 0;
        for camera in self.cameras.realsense.iter_mut().filter(|camera| dds(camera)) {
            let Some(address) = address else {
                return Err(format!(
                    "{} is a DDS camera: state this host's camera-LAN address with --dds-address",
                    camera.name
                ));
            };
            camera.dds_address = Some(address);
            bound += 1;
        }
        if bound == 0 && address.is_some() {
            return Err("--dds-address given but no DDS camera is selected".into());
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
    /// False, with a note, in a checkout that omits the deployment's `rel`
    /// (the public export): a test that reads it returns early there.
    fn deployed(rel: &str) -> bool {
        let present = Path::new(env!("CARGO_MANIFEST_DIR")).join(rel).is_file();
        if !present {
            eprintln!("skipped: needs the deployment's {rel}");
        }
        present
    }

    // Phase 3 exit gate: the public example registry loads, and the registry
    // flexes to one camera / no depth camera without code changes.
    #[test]
    fn example_registry_loads_and_validates() {
        let path =
            std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("config/vision.example.toml");
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

    #[test]
    fn deployed_overhead_depth_profile_states_measured_controls() {
        if !deployed("config/vision.toml") {
            return;
        }
        let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("config/vision.toml");
        let config = VisionConfig::load(path).expect("deployed config must load");
        let camera = config
            .cameras
            .realsense
            .iter()
            .find(|camera| camera.group == "overhead-depth")
            .expect("fixed overhead group");
        assert_eq!(camera.role, "surface_overhead");
        assert_eq!((camera.color.width, camera.color.height), (640, 360));
        assert_eq!(camera.visual_preset, Some(0.0));
        assert_eq!(camera.laser_power, Some(150.0));
        assert_eq!(camera.depth_units_m, Some(0.001));
        assert_eq!(camera.transport, RealSenseTransport::Dds);
    }

    #[test]
    fn dds_cameras_bind_only_to_a_stated_owner_address() {
        if !deployed("config/vision.toml") {
            return;
        }
        let path = Path::new(env!("CARGO_MANIFEST_DIR")).join("config/vision.toml");
        let source = VisionConfig::load(path).unwrap();
        let address: IpAddr = "192.0.2.52".parse().unwrap();

        let mut overhead = source.clone();
        overhead.select_realsense(Some("overhead-depth"), &[]).unwrap();
        let refused = overhead.clone().bind_dds_address(None).unwrap_err();
        assert!(refused.contains("--dds-address"), "{refused}");
        overhead.bind_dds_address(Some(address)).unwrap();
        assert!(
            overhead
                .cameras
                .realsense
                .iter()
                .all(|camera| camera.dds_address == Some(address))
        );

        let mut wrists = source.clone();
        wrists.select_realsense(Some("d405"), &[]).unwrap();
        assert!(
            wrists
                .cameras
                .realsense
                .iter()
                .all(|camera| camera.transport == RealSenseTransport::Usb)
        );
        assert!(wrists.clone().bind_dds_address(Some(address)).is_err());
        wrists.bind_dds_address(None).unwrap();
        assert!(
            wrists
                .cameras
                .realsense
                .iter()
                .all(|camera| camera.dds_address.is_none())
        );
    }

    #[test]
    fn owner_selection_is_exact_and_keeps_physical_metadata() {
        if !deployed("config/vision.toml") {
            return;
        }
        let path = Path::new(env!("CARGO_MANIFEST_DIR")).join("config/vision.toml");
        let source = VisionConfig::load(path).unwrap();
        let left = source
            .cameras
            .realsense
            .iter()
            .find(|c| c.arm.as_deref() == Some("left"))
            .unwrap();
        let right = source
            .cameras
            .realsense
            .iter()
            .find(|c| c.arm.as_deref() == Some("right"))
            .unwrap();
        assert_ne!(left.serial, right.serial);
        for camera in [left, right] {
            let mut config = source.clone();
            config
                .select_realsense(Some("d405"), std::slice::from_ref(&camera.name))
                .unwrap();
            assert_eq!(config.cameras.realsense.len(), 1);
            assert_eq!(config.cameras.realsense[0].serial, camera.serial);
            assert_eq!(config.cameras.realsense[0].arm, camera.arm);
            assert_eq!(config.cameras.realsense[0].owner_role, camera.owner_role);
        }
        for (group, selection) in [
            ("d405", vec!["missing".into()]),
            ("overhead-depth", vec![left.name.clone()]),
            ("d405", vec![left.name.clone(), left.name.clone()]),
            ("missing", vec![]),
        ] {
            let mut config = source.clone();
            assert!(config.select_realsense(Some(group), &selection).is_err());
            assert_eq!(
                config.cameras.realsense.len(),
                source.cameras.realsense.len()
            );
        }
        let mut config = source;
        config.select_realsense(Some("d405"), &[]).unwrap();
        assert_eq!(config.cameras.realsense.len(), 2);
    }

    #[test]
    fn physical_identity_and_device_aliases_are_validated() {
        let path = Path::new(env!("CARGO_MANIFEST_DIR")).join("config/vision.example.toml");
        let mut config = VisionConfig::load(path).unwrap();
        config.cameras.realsense[0].arm = Some("third_arm".into());
        assert!(config.validate().is_ok());
        config.cameras.realsense[0].arm = Some("../other".into());
        assert!(config.validate().is_err());
        config.cameras.realsense[0].arm = None;
        assert!(config.validate().is_ok());
        let mut alias = config.cameras.realsense[0].clone();
        alias.name = "duplicate-device".into();
        config.cameras.realsense.push(alias);
        assert!(
            config
                .validate()
                .unwrap_err()
                .contains("duplicate RealSense serial")
        );
    }
    #[test]
    fn manual_sensor_controls_require_explicit_exposure_mode() {
        if !deployed("config/vision.toml") {
            return;
        }
        let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("config/vision.toml");
        let mut config = VisionConfig::load(path).unwrap();
        config.cameras.realsense[0].depth_controls.exposure_us = Some(1000.0);
        assert!(config.validate().is_err());
        config.cameras.realsense[0].depth_controls.auto_exposure = Some(false);
        assert!(config.validate().is_ok());
        config.cameras.realsense[0].depth_controls.gain = Some(f32::NAN);
        assert!(config.validate().is_err());
    }
}
