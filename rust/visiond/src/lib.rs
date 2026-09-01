//! Tatbot 2.0 vision contracts and capture backends.
//!
//! The crate deliberately keeps sensor-specific details behind the common
//! [`FrameRecord`] contract.  A recorder, synchronizer, or policy bridge must
//! not need to know whether a frame came from RTSP or a RealSense device.

pub mod calibration;
pub mod config;
pub mod health;
pub mod record;
pub mod rtcp;
pub mod sync;
pub mod teleop;
pub mod time;
pub mod transport;

#[cfg(feature = "fiducials")]
pub mod fiducials;

#[cfg(feature = "rerun")]
pub mod rerun_viewer;

#[cfg(feature = "gstreamer")]
pub mod gstreamer_backend;

#[cfg(feature = "gstreamer")]
pub mod monitor;

#[cfg(feature = "realsense")]
pub mod realsense_backend;

use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};

pub use calibration::{
    CALIBRATION_SCHEMA_VERSION, CalibrationBundle, CameraCalibration, DistortionModel, Intrinsics,
    Pose,
};
pub use config::{CameraConfig, RealSenseConfig, VisionConfig};
#[cfg(feature = "fiducials")]
pub use fiducials::{
    AprilTagDetector, AprilTagDetectorFactory, DetectionRoi, EePoseEstimate, EstimatorConfig,
    FiducialDetection, FiducialInventory, RustEeTracker, WristLayout, expanded_detection_roi,
};
pub use health::{HealthSnapshot, SensorHealth};
pub use record::read_recording_frame;
pub use record::{EvidenceRecorder, RecordedPayload, RecordingEntry, read_recording_entries};
#[cfg(feature = "rerun")]
pub use rerun_viewer::{
    LiveTeleopScene, RerunLayout, RerunSink, RerunSinkStats, RerunViewer, TeleopSetup,
};
pub use sync::{
    ClockOffsetEstimator, FrameSynchronizer, PairwiseSyncReport, SyncAssessment,
    SynchronizedFrameSet, pairwise_sync_report,
};
pub use teleop::{LiveTeleopTick, TeleopLog, TeleopTick};
pub use time::{ClockSample, TimestampDomain};
pub use transport::{ReceivedFrameSet, UnixFrameClient, UnixFramePublisher};

/// Stable sensor-family identifier used in recordings and telemetry.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SensorKind {
    PoE,
    RealSense,
}

/// Pixel format carried by a frame payload.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum PixelFormat {
    H264,
    Bgr8,
    Rgb8,
    Yuyv,
    Z16,
    Y8,
}

/// The profile that was actually active when a frame was captured.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct StreamProfile {
    pub stream: String,
    pub width: u32,
    pub height: u32,
    pub fps_num: u32,
    pub fps_den: u32,
    pub format: PixelFormat,
}

impl StreamProfile {
    pub fn fps(&self) -> f64 {
        if self.fps_den == 0 {
            return 0.0;
        }
        self.fps_num as f64 / self.fps_den as f64
    }

    pub fn validate(&self) -> Result<(), String> {
        if self.stream.trim().is_empty() {
            return Err("stream name must not be empty".into());
        }
        if self.width == 0 || self.height == 0 {
            return Err("stream dimensions must be positive".into());
        }
        if self.fps_num == 0 || self.fps_den == 0 || !(0.1..=240.0).contains(&self.fps()) {
            return Err("stream FPS must be between 0.1 and 240 Hz".into());
        }
        Ok(())
    }
}

/// All timestamps retained for one frame.  No source timestamp is discarded
/// when a normalized timestamp is computed.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct FrameTimestamps {
    pub source_ns: Option<i128>,
    pub source_domain: TimestampDomain,
    pub rtp_timestamp: Option<u64>,
    pub pipeline_pts_ns: Option<u64>,
    pub pipeline_dts_ns: Option<u64>,
    pub host_monotonic_ns: u128,
    pub host_unix_ns: i128,
    pub normalized_unix_ns: Option<i128>,
}

/// Metadata attached to every emitted frame or frame component.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct FrameMetadata {
    pub sensor_name: String,
    pub sensor_kind: SensorKind,
    pub sequence: u64,
    pub profile: StreamProfile,
    pub timestamps: FrameTimestamps,
    pub dropped_before: u64,
    pub calibration_id: Option<String>,
    pub flags: Vec<String>,
    #[serde(default)]
    pub attributes: BTreeMap<String, String>,
}

/// A frame without imposing a single memory layout on all backends.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FrameRecord {
    pub metadata: FrameMetadata,
    pub payload: RecordedPayload,
}

impl FrameRecord {
    pub fn validate(&self) -> Result<(), String> {
        self.metadata.profile.validate()?;
        if self.metadata.sensor_name.trim().is_empty() {
            return Err("sensor name must not be empty".into());
        }
        if self.metadata.timestamps.host_unix_ns <= 0 {
            return Err("host wall-clock timestamp must be positive".into());
        }
        Ok(())
    }
}
