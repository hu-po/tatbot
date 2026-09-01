//! Intel RealSense capture backend.

use std::{collections::BTreeMap, ffi::CString, time::Duration};

use anyhow::{Context as _, Result, anyhow};
use realsense_rust::{
    config::Config as RsConfig,
    context::Context as RsContext,
    frame::{ColorFrame, CompositeFrame, DepthFrame, ImageFrame},
    kind::{Rs2Format, Rs2FrameMetadata, Rs2StreamKind, Rs2TimestampDomain},
    pipeline::{ActivePipeline, InactivePipeline},
    prelude::FrameEx,
};

use crate::{
    FrameMetadata, FrameRecord, FrameTimestamps, PixelFormat, RecordedPayload, SensorKind,
    StreamProfile,
    config::RealSenseConfig,
    health::{HealthSnapshot, SensorHealth},
    time::{MonotonicClock, TimestampDomain},
};

#[derive(Debug)]
pub struct RealsenseCapture {
    context: Option<RsContext>,
    pipeline: Option<ActivePipeline>,
    config: RealSenseConfig,
    clock: MonotonicClock,
    health: SensorHealth,
    /// Metres per raw Z16 unit, read from the depth sensor once. The D405
    /// reports 0.0001 (0.1 mm), unlike the 0.001 of most D4xx; every consumer
    /// of the recorded depth needs this to be metric, so it rides in the
    /// frame attributes as `depth_units_m`.
    depth_units_m: Option<f32>,
}

impl RealsenseCapture {
    pub fn new(config: RealSenseConfig) -> Result<Self> {
        let context = RsContext::new().context("creating librealsense context")?;
        let inactive =
            InactivePipeline::try_from(&context).context("creating librealsense pipeline")?;
        let serial =
            CString::new(config.serial.clone()).context("RealSense serial contains NUL")?;
        let mut stream_config = RsConfig::new();
        stream_config
            .enable_device_from_serial(serial.as_c_str())
            .context("selecting RealSense device by serial")?;
        stream_config
            .enable_stream(
                Rs2StreamKind::Color,
                None,
                config.color.width as usize,
                config.color.height as usize,
                to_rs_format(config.color.format)?,
                (config.color.fps_num / config.color.fps_den) as usize,
            )
            .context("enabling RealSense color stream")?;
        stream_config
            .enable_stream(
                Rs2StreamKind::Depth,
                None,
                config.depth.width as usize,
                config.depth.height as usize,
                to_rs_format(config.depth.format)?,
                (config.depth.fps_num / config.depth.fps_den) as usize,
            )
            .context("enabling RealSense depth stream")?;
        let pipeline = inactive
            .start(Some(stream_config))
            .context("starting RealSense pipeline")?;
        Ok(Self {
            health: SensorHealth::new(config.name.clone()),
            context: Some(context),
            pipeline: Some(pipeline),
            config,
            clock: MonotonicClock::default(),
            depth_units_m: None,
        })
    }

    /// Wait for one librealsense frameset and return separate color/depth
    /// records that share the same device frame number where possible.
    pub fn next_frames(&mut self, timeout: Duration) -> Result<Option<Vec<FrameRecord>>> {
        let pipeline = self
            .pipeline
            .as_mut()
            .ok_or_else(|| anyhow!("RealSense pipeline is stopped"))?;
        let composite = match pipeline.wait(Some(timeout)) {
            Ok(composite) => composite,
            Err(error) if error.to_string().contains("Timed out") => return Ok(None),
            Err(error) => return Err(anyhow!(error).context("waiting for RealSense frames")),
        };
        self.records_from_composite(composite)
    }

    pub fn health(&self) -> HealthSnapshot {
        self.health.snapshot()
    }

    pub fn stop(&mut self) {
        if let Some(pipeline) = self.pipeline.take() {
            let _inactive = pipeline.stop();
        }
        self.context.take();
    }

    fn records_from_composite(
        &mut self,
        composite: CompositeFrame,
    ) -> Result<Option<Vec<FrameRecord>>> {
        let color = composite.frames_of_type::<ColorFrame>().into_iter().next();
        let depth = composite.frames_of_type::<DepthFrame>().into_iter().next();
        let Some(color) = color else {
            return Err(anyhow!("RealSense frameset had no color frame"));
        };
        let Some(depth) = depth else {
            return Err(anyhow!("RealSense frameset had no depth frame"));
        };
        if self.depth_units_m.is_none() {
            self.depth_units_m = depth
                .depth_units()
                .ok()
                .filter(|u| u.is_finite() && *u > 0.0);
        }
        let clock_sample = self.clock.now();
        let sequence = color.frame_number();
        self.health
            .frame_received(sequence, clock_sample.monotonic_ns);
        let dropped_before = self.health.snapshot().frames_dropped;
        let color_record = self.image_record(
            &color,
            format!("{}_color", self.config.name),
            SensorKind::RealSense,
            sequence,
            clock_sample,
            dropped_before,
            false,
        )?;
        let depth_record = self.image_record(
            &depth,
            format!("{}_depth", self.config.name),
            SensorKind::RealSense,
            sequence,
            clock_sample,
            dropped_before,
            true,
        )?;
        Ok(Some(vec![color_record, depth_record]))
    }

    fn image_record<K>(
        &self,
        frame: &ImageFrame<K>,
        sensor_name: String,
        sensor_kind: SensorKind,
        sequence: u64,
        clock_sample: crate::time::ClockSample,
        dropped_before: u64,
        depth: bool,
    ) -> Result<FrameRecord> {
        let timestamp_ns = (frame.timestamp() * 1_000_000.0).round() as i128;
        let source_domain = match frame.timestamp_domain() {
            Rs2TimestampDomain::GlobalTime => TimestampDomain::RealSenseGlobal,
            Rs2TimestampDomain::HardwareClock => TimestampDomain::RealSenseHardware,
            Rs2TimestampDomain::SystemTime => TimestampDomain::HostUnix,
        };
        let normalized_unix_ns = match source_domain {
            TimestampDomain::RealSenseGlobal | TimestampDomain::HostUnix => Some(timestamp_ns),
            _ => None,
        };
        let actual_format = PixelFormat::from_rs_format(frame.stream_profile().format())?;
        let configured = if depth {
            &self.config.depth
        } else {
            &self.config.color
        };
        let profile = StreamProfile {
            stream: configured.stream.clone(),
            width: frame.width() as u32,
            height: frame.height() as u32,
            fps_num: frame.stream_profile().framerate().max(0) as u32,
            fps_den: 1,
            format: actual_format,
        };
        let mut flags = vec![format!(
            "timestamp_domain={}",
            frame.timestamp_domain().as_str()
        )];
        if profile != *configured {
            flags.push("active_profile_differs_from_config".to_string());
        }
        let mut attributes = BTreeMap::from([
            (
                "device_timestamp_ms".to_string(),
                frame.timestamp().to_string(),
            ),
            ("frame_number".to_string(), frame.frame_number().to_string()),
            ("stride_bytes".to_string(), frame.stride().to_string()),
            (
                "bits_per_pixel".to_string(),
                frame.bits_per_pixel().to_string(),
            ),
        ]);
        add_frame_metadata(&mut attributes, frame);
        if depth {
            match self.depth_units_m {
                Some(units) => {
                    attributes.insert("depth_units_m".to_string(), units.to_string());
                }
                None => flags.push("depth_units_unknown".to_string()),
            }
        }
        let bytes = copy_image_data(frame);
        let payload = if depth {
            RecordedPayload::Depth {
                width: profile.width,
                height: profile.height,
                bytes,
            }
        } else {
            RecordedPayload::Video {
                format: actual_format,
                width: profile.width,
                height: profile.height,
                bytes,
            }
        };
        let record = FrameRecord {
            metadata: FrameMetadata {
                sensor_name,
                sensor_kind,
                sequence,
                profile,
                timestamps: FrameTimestamps {
                    source_ns: Some(timestamp_ns),
                    source_domain,
                    rtp_timestamp: None,
                    pipeline_pts_ns: None,
                    pipeline_dts_ns: None,
                    host_monotonic_ns: clock_sample.monotonic_ns,
                    host_unix_ns: clock_sample.unix_ns,
                    normalized_unix_ns,
                },
                dropped_before,
                calibration_id: None,
                flags: std::mem::take(&mut flags),
                attributes,
            },
            payload,
        };
        record.validate().map_err(anyhow::Error::msg)?;
        Ok(record)
    }
}

impl Drop for RealsenseCapture {
    fn drop(&mut self) {
        self.stop();
    }
}

fn to_rs_format(format: PixelFormat) -> Result<Rs2Format> {
    match format {
        PixelFormat::Yuyv => Ok(Rs2Format::Yuyv),
        PixelFormat::Rgb8 => Ok(Rs2Format::Rgb8),
        PixelFormat::Bgr8 => Ok(Rs2Format::Bgr8),
        PixelFormat::Z16 => Ok(Rs2Format::Z16),
        other => Err(anyhow!("unsupported RealSense requested format {other:?}")),
    }
}

impl PixelFormat {
    fn from_rs_format(format: Rs2Format) -> Result<Self> {
        match format {
            Rs2Format::Yuyv => Ok(Self::Yuyv),
            Rs2Format::Rgb8 => Ok(Self::Rgb8),
            Rs2Format::Bgr8 => Ok(Self::Bgr8),
            Rs2Format::Z16 => Ok(Self::Z16),
            other => Err(anyhow!("unsupported active RealSense format {other:?}")),
        }
    }
}

fn copy_image_data<K>(frame: &ImageFrame<K>) -> Vec<u8> {
    unsafe {
        let data = std::ptr::from_ref(frame.get_data()) as *const u8;
        std::slice::from_raw_parts(data, frame.get_data_size()).to_vec()
    }
}

fn add_frame_metadata<K>(attributes: &mut BTreeMap<String, String>, frame: &ImageFrame<K>) {
    for (name, key) in [
        ("sensor_timestamp_us", Rs2FrameMetadata::SensorTimestamp),
        ("actual_exposure_us", Rs2FrameMetadata::ActualExposure),
        ("gain_level", Rs2FrameMetadata::GainLevel),
        ("actual_fps", Rs2FrameMetadata::ActualFps),
        ("time_of_arrival_us", Rs2FrameMetadata::TimeOfArrival),
    ] {
        if let Some(value) = frame.metadata(key) {
            attributes.insert(name.to_string(), value.to_string());
        }
    }
}
