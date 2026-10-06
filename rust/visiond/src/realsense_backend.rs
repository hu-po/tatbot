//! Intel RealSense capture backend.

use std::{
    collections::BTreeMap,
    ffi::CString,
    task::Poll,
    time::{Duration, Instant},
};

use anyhow::{Context as _, Result, anyhow};
use realsense_rust::{
    config::Config as RsConfig,
    context::Context as RsContext,
    frame::{ColorFrame, CompositeFrame, DepthFrame, ImageFrame},
    kind::{
        Rs2CameraInfo, Rs2Format, Rs2FrameMetadata, Rs2Option, Rs2StreamKind, Rs2TimestampDomain,
    },
    pipeline::{ActivePipeline, InactivePipeline},
    prelude::FrameEx,
    processing_blocks::align::Align,
};

use crate::{
    FrameMetadata, FrameRecord, FrameTimestamps, PixelFormat, RecordedPayload, SensorKind,
    StreamProfile,
    config::{RealSenseConfig, RealSenseTransport},
    health::{HealthSnapshot, SensorHealth},
    sync::ClockOffsetEstimator,
    time::{MonotonicClock, RawDeviceClockRead, TimestampDomain},
};

#[derive(Debug)]
pub struct RealsenseCapture {
    _lease: crate::ownership::CameraLease,
    align: Align,
    alignment_calibration: Option<serde_json::Value>,
    context: Option<RsContext>,
    pipeline: Option<ActivePipeline>,
    config: RealSenseConfig,
    clock: MonotonicClock,
    clock_offset: ClockOffsetEstimator,
    health: SensorHealth,
    /// Recording sequence is process-local and strictly increasing. Device
    /// frame numbers can repeat after DDS alignment and remain in attributes.
    record_sequence: u64,
    capture_epoch: String,
    effective_options: String,
    firmware: String,
    /// Metres per raw Z16 unit, read from the depth sensor once. The D405
    /// reports 0.0001 (0.1 mm), unlike the 0.001 of most D4xx; every consumer
    /// of the recorded depth needs this to be metric, so it rides in the
    /// frame attributes as `depth_units_m`.
    depth_units_m: Option<f32>,
    /// DDS can repeat a complete frameset with the same device timestamp.
    /// Suppress it before evidence, synchronization, and bus publication.
    last_color_timestamp_ns: Option<i128>,
    last_depth_timestamp_ns: Option<i128>,
}

fn drain_ready_queue<T>(
    mut poll: impl FnMut() -> Result<Poll<T>>,
    clear_aligned: impl FnOnce() -> Result<()>,
) -> Result<u64> {
    let started = Instant::now();
    for discarded in 0..64_u64 {
        anyhow::ensure!(
            started.elapsed() < Duration::from_millis(100),
            "RealSense SDK queue did not drain within 100 ms"
        );
        let ready = poll()?;
        anyhow::ensure!(
            started.elapsed() < Duration::from_millis(100),
            "RealSense SDK queue did not drain within 100 ms"
        );
        match ready {
            Poll::Pending => {
                clear_aligned()?;
                return Ok(discarded);
            }
            Poll::Ready(_old_set) => {}
        }
    }
    anyhow::bail!("RealSense SDK queue did not drain within 64 sets")
}

impl RealsenseCapture {
    /// A diagnostic read of the existing pipeline device's raw counter. The
    /// SDK call is only made for a requested overhead pair, not each frame.
    pub fn raw_clock_read(&self) -> RawDeviceClockRead {
        #[cfg(not(feature = "realsense-raw-clock"))]
        {
            RawDeviceClockRead {
                status: "feature_disabled".into(),
                owner_before_unix_ns: None,
                owner_after_unix_ns: None,
                owner_monotonic_elapsed_ns: None,
                device_time_us: None,
            }
        }
        #[cfg(feature = "realsense-raw-clock")]
        {
            fn now_ns() -> Option<u64> {
                u64::try_from(
                    std::time::SystemTime::now()
                        .duration_since(std::time::UNIX_EPOCH)
                        .ok()?
                        .as_nanos(),
                )
                .ok()
            }
            let before = now_ns();
            let started = Instant::now();
            let value = self
                .pipeline
                .as_ref()
                .ok_or_else(|| "pipeline stopped".to_owned())
                .and_then(|pipeline| pipeline.profile().device().raw_device_time_ms());
            let elapsed = u64::try_from(started.elapsed().as_nanos()).ok();
            let after = now_ns();
            let (status, device_time_us) = match value {
                Ok(ms) if before.is_some() && after.is_some() && elapsed.is_some() => {
                    ("ok", Some((ms * 1000.0).round() as u64))
                }
                Ok(_) => ("read_failed", None),
                Err(reason)
                    if reason.to_ascii_lowercase().contains("unsupported")
                        || reason.to_ascii_lowercase().contains("not supported") =>
                {
                    ("unsupported", None)
                }
                Err(_) => ("read_failed", None),
            };
            RawDeviceClockRead {
                status: status.into(),
                owner_before_unix_ns: before,
                owner_after_unix_ns: after,
                owner_monotonic_elapsed_ns: elapsed,
                device_time_us,
            }
        }
    }

    pub fn raw_clock_sdk_version() -> Option<&'static str> {
        #[cfg(feature = "realsense-raw-clock")]
        {
            Some(realsense_rust::RAW_DEVICE_CLOCK_SDK_VERSION)
        }
        #[cfg(not(feature = "realsense-raw-clock"))]
        {
            None
        }
    }

    /// Empty every RGBD set currently ready in the SDK pipeline before a
    /// request-bound set is delivered. Pending proves only that this host's
    /// SDK queue was empty then; it cannot bound an in-flight device exposure.
    pub fn drain_ready_frames(&mut self) -> Result<u64> {
        let pipeline = self
            .pipeline
            .as_mut()
            .ok_or_else(|| anyhow!("RealSense pipeline is stopped"))?;
        // An earlier alignment wait may have timed out after queueing a
        // frameset. Do not let that processing block deliver pre-barrier
        // output after the pipeline queue was drained.
        let align = &mut self.align;
        let calibration = &mut self.alignment_calibration;
        drain_ready_queue(
            || pipeline.poll().context("polling RealSense SDK queue"),
            || {
                *align = Align::new(Rs2StreamKind::Color, 1)
                    .context("recreating RealSense alignment after SDK queue drain")?;
                *calibration = None;
                Ok(())
            },
        )
    }

    pub fn new(config: RealSenseConfig) -> Result<Self> {
        let lease = crate::ownership::CameraLease::camera(&format!("realsense:{}", config.serial))?;
        let align =
            Align::new(Rs2StreamKind::Color, 1).context("creating depth-to-color alignment")?;
        let context = open_context(&config)?;
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
        let mut preset_set = config.visual_preset.is_none();
        let mut laser_set = config.laser_power.is_none();
        let mut depth_units_m = config.depth_units_m;
        let firmware = pipeline
            .profile()
            .device()
            .info(Rs2CameraInfo::FirmwareVersion)
            .map(|v| v.to_string_lossy().into_owned())
            .unwrap_or_else(|| "unavailable".into());
        let mut options = Vec::new();
        for mut sensor in pipeline.profile().device().sensors() {
            depth_units_m = depth_units_m.or_else(|| {
                sensor
                    .get_option(Rs2Option::DepthUnits)
                    .filter(|value| value.is_finite() && *value > 0.0)
            });
            if let Some(value) = config.visual_preset
                && sensor.get_option(Rs2Option::VisualPreset).is_some()
            {
                sensor
                    .set_option(Rs2Option::VisualPreset, value)
                    .context("setting RealSense visual preset")?;
                preset_set = true;
            }
            if let Some(value) = config.laser_power
                && sensor.get_option(Rs2Option::LaserPower).is_some()
            {
                sensor
                    .set_option(Rs2Option::LaserPower, value)
                    .context("setting RealSense laser power")?;
                laser_set = true;
            }
            let streams = sensor.stream_profiles();
            let is_depth = streams.iter().any(|p| p.kind() == Rs2StreamKind::Depth);
            let is_color = streams.iter().any(|p| p.kind() == Rs2StreamKind::Color);
            if !is_depth && !is_color {
                continue;
            }
            if is_depth && is_color && config.color_controls != Default::default() {
                anyhow::ensure!(
                    config.depth_controls == config.color_controls,
                    "shared RGB/depth sensor cannot use different exposure controls"
                );
            }
            let controls = if is_depth {
                &config.depth_controls
            } else {
                &config.color_controls
            };
            let mut values = serde_json::Map::new();
            for (name, option, requested) in [
                (
                    "auto_exposure",
                    Rs2Option::EnableAutoExposure,
                    controls.auto_exposure.map(|v| if v { 1.0 } else { 0.0 }),
                ),
                ("exposure_us", Rs2Option::Exposure, controls.exposure_us),
                ("gain", Rs2Option::Gain, controls.gain),
                ("global_time_enabled", Rs2Option::GlobalTimeEnabled, None),
                ("visual_preset", Rs2Option::VisualPreset, None),
                ("laser_power", Rs2Option::LaserPower, None),
            ] {
                if let Some(value) = requested {
                    let range = sensor
                        .get_option_range(option)
                        .ok_or_else(|| anyhow!("unsupported {name}"))?;
                    anyhow::ensure!(
                        (range.min..=range.max).contains(&value),
                        "{name} outside device range"
                    );
                    sensor
                        .set_option(option, value)
                        .with_context(|| format!("setting {name}"))?;
                    let actual = sensor
                        .get_option(option)
                        .ok_or_else(|| anyhow!("missing {name} readback"))?;
                    anyhow::ensure!(
                        (actual - value).abs() <= range.step.max(1e-4),
                        "{name} readback differs"
                    );
                }
                if let Some(range) = sensor.get_option_range(option) {
                    values.insert(name.into(), serde_json::json!({"requested":requested,
                        "effective":sensor.get_option(option), "min":range.min, "max":range.max, "step":range.step}));
                }
            }
            options.push(serde_json::json!({"depth":is_depth,"color":is_color,"options":values}));
        }
        anyhow::ensure!(preset_set, "RealSense visual preset is unsupported");
        anyhow::ensure!(laser_set, "RealSense laser power is unsupported");
        Ok(Self {
            _lease: lease,
            align,
            alignment_calibration: None,
            health: SensorHealth::new(config.name.clone()),
            record_sequence: 0,
            capture_epoch: format!(
                "{}-{}",
                std::process::id(),
                std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)?
                    .as_nanos()
            ),
            effective_options: serde_json::json!({"requested":config,"sensors":options})
                .to_string(),
            firmware,
            context: Some(context),
            pipeline: Some(pipeline),
            config,
            clock: MonotonicClock::default(),
            clock_offset: ClockOffsetEstimator::new(128).map_err(anyhow::Error::msg)?,
            depth_units_m,
            last_color_timestamp_ns: None,
            last_depth_timestamp_ns: None,
        })
    }

    /// Wait for one librealsense frameset and return separate color/depth
    /// records with verified exposure times and current alignment calibration.
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
        // The SDK caches aligned profiles and SIMD projection tables by stream
        // identity, while device calibration can change on the same profile.
        // Recreate the processing block before using changed camera models.
        let calibration = alignment_calibration(&composite)?;
        if self
            .alignment_calibration
            .as_ref()
            .is_some_and(|old| *old != calibration)
        {
            self.align = Align::new(Rs2StreamKind::Color, 1)
                .context("refreshing changed RealSense alignment calibration")?;
        }
        self.alignment_calibration = Some(calibration.clone());
        self.align
            .queue(composite)
            .context("aligning depth to color")?;
        let composite = self
            .align
            .wait(timeout)
            .context("waiting for aligned RGBD")?;
        let mut records = self.records_from_composite(composite)?;
        if let Some(records) = &records
            && !aligned_records_match(records, &calibration)
        {
            tracing::debug!(
                sensor = %self.config.name,
                calibration_color = %calibration["color"],
                record_intrinsics = ?records
                    .iter()
                    .map(|record| record.metadata.attributes.get("intrinsics").cloned())
                    .collect::<Vec<_>>(),
                "frameset withheld: aligned records do not carry the alignment calibration's colour model"
            );
            // A calibration update can race processing. Drop this exposure and
            // force a fresh alignment block for the next one; never relabel it.
            self.align = Align::new(Rs2StreamKind::Color, 1)
                .context("resetting inconsistent RealSense alignment")?;
            self.alignment_calibration = None;
            return Ok(None);
        }
        if let Some(records) = &mut records {
            retain_alignment_calibration(records, &calibration);
        }
        Ok(records)
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
        let color_timestamp_ns = (color.timestamp() * 1_000_000.0).round() as i128;
        let depth_timestamp_ns = (depth.timestamp() * 1_000_000.0).round() as i128;
        // A librealsense composite can contain adjacent exposures. Its identity
        // alone cannot prove RGB-D correspondence, and the two stream counters
        // need not share an origin. Compare source times before accepting either
        // timestamp so a later, correctly paired composite can still be used.
        if color.timestamp_domain() != depth.timestamp_domain()
            || !exposures_match(
                color_timestamp_ns,
                depth_timestamp_ns,
                color.stream_profile().framerate(),
                depth.stream_profile().framerate(),
            )
        {
            tracing::debug!(
                sensor = %self.config.name,
                color_ns = color_timestamp_ns,
                depth_ns = depth_timestamp_ns,
                "frameset withheld: colour and depth exposures are not one pair"
            );
            return Ok(None);
        }
        if !timestamp_advances(self.last_color_timestamp_ns, color_timestamp_ns)
            || !timestamp_advances(self.last_depth_timestamp_ns, depth_timestamp_ns)
        {
            tracing::debug!(
                sensor = %self.config.name,
                color_ns = color_timestamp_ns,
                depth_ns = depth_timestamp_ns,
                "frameset withheld: timestamps did not advance"
            );
            return Ok(None);
        }
        self.last_color_timestamp_ns = Some(color_timestamp_ns);
        self.last_depth_timestamp_ns = Some(depth_timestamp_ns);
        if self.depth_units_m.is_none() {
            self.depth_units_m = depth
                .depth_units()
                .ok()
                .filter(|u| u.is_finite() && *u > 0.0);
        }
        let clock_sample = self.clock.now();
        let device_sequence = color.frame_number();
        self.health
            .frame_received(device_sequence, clock_sample.monotonic_ns);
        let sequence = self.record_sequence;
        self.record_sequence = self.record_sequence.saturating_add(1);
        let dropped_before = self.health.snapshot().frames_dropped;
        let color_record = self.image_record(
            &color,
            format!("{}_color", self.config.name),
            sequence,
            clock_sample,
            dropped_before,
            false,
        )?;
        let depth_record = self.image_record(
            &depth,
            format!("{}_depth", self.config.name),
            sequence,
            clock_sample,
            dropped_before,
            true,
        )?;
        Ok(Some(vec![color_record, depth_record]))
    }

    fn image_record<K>(
        &mut self,
        frame: &ImageFrame<K>,
        sensor_name: String,
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
            TimestampDomain::RealSenseHardware => {
                self.clock_offset
                    .observe(clock_sample.unix_ns, timestamp_ns);
                self.clock_offset
                    .assessment()
                    .median_offset_ns
                    .map(|offset| timestamp_ns.saturating_add(offset))
            }
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
        if source_domain == TimestampDomain::RealSenseHardware {
            flags.push("hardware_clock_host_normalized".to_string());
        }
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
        // Read the active (including depth-to-color alignment) profile, never
        // substitute nominal intrinsics from a registry or another stream.
        let intrinsics = video_intrinsics(frame.stream_profile())?;
        anyhow::ensure!(
            intrinsics["width"] == profile.width && intrinsics["height"] == profile.height,
            "active RealSense intrinsics differ from image dimensions"
        );
        attributes.insert("device_serial".into(), self.config.serial.clone());
        if let Some(arm) = &self.config.arm {
            attributes.insert("physical_arm".into(), arm.clone());
        }
        if let Some(role) = &self.config.owner_role {
            attributes.insert("capture_owner_role".into(), role.clone());
        }
        attributes.insert("capture_epoch".into(), self.capture_epoch.clone());
        attributes.insert(
            "effective_sensor_options".into(),
            self.effective_options.clone(),
        );
        attributes.insert("device_firmware".into(), self.firmware.clone());
        attributes.insert(
            "clock_offset_assessment".into(),
            serde_json::to_string(&self.clock_offset.assessment())?,
        );
        attributes.insert("clock_absolute_accuracy_verified".into(), "false".into());
        attributes.insert("intrinsics".into(), intrinsics.to_string());
        add_frame_metadata(&mut attributes, frame);
        if depth {
            attributes.insert("aligned_to".into(), format!("{}_color", self.config.name));
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
                sensor_kind: SensorKind::RealSense,
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

/// Force the device with this serial through a USB re-enumeration, the same
/// `rs2_hardware_reset` a person issued by hand when the wrist owner sat in a
/// USB wedge. The caller has already stopped and dropped its own pipeline:
/// librealsense cannot reset a device it is streaming from, and the capture
/// lease must be released before the device disappears. A device the host no
/// longer enumerates is named, never reported as reset.
pub fn hardware_reset(camera: &RealSenseConfig) -> Result<()> {
    let serial = camera.serial.as_str();
    let context = open_context(camera).context("opening the context for the reset")?;
    let device = context
        .query_devices(std::collections::HashSet::new())
        .into_iter()
        .find(|device| {
            device
                .info(Rs2CameraInfo::SerialNumber)
                .is_some_and(|value| value.to_bytes() == serial.as_bytes())
        })
        .ok_or_else(|| anyhow!("RealSense {serial} is not enumerated on this host"))?;
    device.hardware_reset();
    Ok(())
}

/// RS2_PRODUCT_LINE_SW_ONLY | RS2_PRODUCT_LINE_ANY: software devices only,
/// which leaves out the USB backend, so a DDS owner never enumerates, creates
/// or opens a USB camera another owner streams from.
const DDS_ONLY_DEVICE_MASK: u32 = 0x1ff;

/// The camera's own kind of context: the SDK default for USB; for DDS,
/// discovery on and bound to the owner's camera-LAN address, USB left out.
fn open_context(camera: &RealSenseConfig) -> Result<RsContext> {
    match camera.transport {
        RealSenseTransport::Usb => RsContext::new().context("creating librealsense context"),
        RealSenseTransport::Dds => RsContext::with_settings(&dds_context_settings(camera)?)
            .context("creating librealsense DDS context (is this librealsense built with DDS?)"),
    }
}

fn dds_context_settings(camera: &RealSenseConfig) -> Result<String> {
    let address = camera.dds_address.ok_or_else(|| {
        anyhow!("{} is a DDS camera with no owner address to bind", camera.name)
    })?;
    // The participant only answers on, and only announces, this address:
    // with every interface in play, discovery can reply over the wrong route.
    Ok(serde_json::json!({
        "dds": {
            "enabled": true,
            "domain": 0,
            "participant": "tatbot-visiond",
            "udp": {"whitelist": [address.to_string()]},
        },
        "device-mask": DDS_ONLY_DEVICE_MASK,
    })
    .to_string())
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

fn alignment_calibration(composite: &CompositeFrame) -> Result<serde_json::Value> {
    let color = composite
        .frames_of_type::<ColorFrame>()
        .into_iter()
        .next()
        .ok_or_else(|| anyhow!("RealSense frameset had no color frame"))?;
    let depth = composite
        .frames_of_type::<DepthFrame>()
        .into_iter()
        .next()
        .ok_or_else(|| anyhow!("RealSense frameset had no depth frame"))?;
    let cp = color.stream_profile();
    let dp = depth.stream_profile();
    let extrinsics = dp
        .extrinsics(cp)
        .context("reading depth-to-color extrinsics")?;
    anyhow::ensure!(
        extrinsics
            .rotation()
            .iter()
            .chain(extrinsics.translation().iter())
            .all(|v| v.is_finite()),
        "invalid depth-to-color extrinsics"
    );
    Ok(serde_json::json!({
        "schema":"tatbot.realsense-native-alignment/1",
        "transform":"color_from_native_depth", "rotation_layout":"column_major",
        "translation_units":"metres", "aligned_depth_value_axis":"native_depth_z",
        "color": video_intrinsics(cp)?, "depth": video_intrinsics(dp)?,
        "color_id": cp.unique_id(), "depth_id": dp.unique_id(),
        "color_fps": cp.framerate(), "depth_fps": dp.framerate(),
        "rotation": extrinsics.rotation(), "translation": extrinsics.translation(),
    }))
}

fn video_intrinsics(
    profile: &realsense_rust::stream_profile::StreamProfile,
) -> Result<serde_json::Value> {
    let intrinsics = profile
        .intrinsics()
        .context("reading active RealSense video intrinsics")?;
    let distortion = intrinsics.distortion();
    anyhow::ensure!(
        intrinsics.width() > 0
            && intrinsics.height() > 0
            && intrinsics.fx() > 0.0
            && intrinsics.fy() > 0.0
            && [
                intrinsics.fx(),
                intrinsics.fy(),
                intrinsics.ppx(),
                intrinsics.ppy()
            ]
            .iter()
            .chain(distortion.coeffs.iter())
            .all(|v| v.is_finite()),
        "invalid active RealSense intrinsics"
    );
    Ok(serde_json::json!({
        "schema":"tatbot.camera-intrinsics/1",
        "width":intrinsics.width(), "height":intrinsics.height(),
        "fx":intrinsics.fx(), "fy":intrinsics.fy(),
        "ppx":intrinsics.ppx(), "ppy":intrinsics.ppy(),
        "distortion_model":format!("{:?}", distortion.model),
        "distortion_coefficients":distortion.coeffs,
    }))
}

fn aligned_records_match(records: &[FrameRecord], calibration: &serde_json::Value) -> bool {
    records.len() == 2
        && records.iter().all(|record| {
            record
                .metadata
                .attributes
                .get("intrinsics")
                .and_then(|value| serde_json::from_str::<serde_json::Value>(value).ok())
                .is_some_and(|intrinsics| intrinsics_equal(&intrinsics, &calibration["color"]))
        })
}

/// The same camera model, compared as numbers rather than as JSON values.
///
/// A record's intrinsics travel as text and come back through the JSON
/// parser; the calibration's are the values the SDK reported. Without
/// serde_json's exact float parsing the two can differ by one ULP, and value
/// equality then withheld every frameset a D405 produced (2026-09-16, the
/// wrist owner's first run on hardware after the alignment work). Dimensions
/// and the model name must be identical; the floats agree to a relative
/// 1e-9, far below any calibration change the check exists to catch.
fn intrinsics_equal(a: &serde_json::Value, b: &serde_json::Value) -> bool {
    let same = |key: &str| a.get(key) == b.get(key);
    let exact = |key: &str| a.get(key).is_some() && same(key);
    let close = |x: &serde_json::Value, y: &serde_json::Value| match (x.as_f64(), y.as_f64()) {
        (Some(x), Some(y)) => (x - y).abs() <= 1e-9 * x.abs().max(y.abs()).max(1e-30),
        _ => false,
    };
    let coefficients = |value: &serde_json::Value| {
        value
            .get("distortion_coefficients")
            .and_then(serde_json::Value::as_array)
            .cloned()
    };
    same("schema")
        && exact("width")
        && exact("height")
        && same("distortion_model")
        && ["fx", "fy", "ppx", "ppy"]
            .iter()
            .all(|key| match (a.get(key), b.get(key)) {
                (Some(x), Some(y)) => close(x, y),
                _ => false,
            })
        && match (coefficients(a), coefficients(b)) {
            (Some(x), Some(y)) => x.len() == y.len() && x.iter().zip(&y).all(|(x, y)| close(x, y)),
            _ => false,
        }
}

fn retain_alignment_calibration(records: &mut [FrameRecord], calibration: &serde_json::Value) {
    // SDK alignment copies raw Z16 values into the color pixel grid; they
    // still measure native depth Z. Retain the actual pre-alignment transform
    // for explicit geometry interpretation without changing captured pixels.
    let encoded = calibration.to_string();
    for record in records {
        record
            .metadata
            .attributes
            .insert("alignment_calibration".into(), encoded.clone());
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
        ("frame_timestamp_us", Rs2FrameMetadata::FrameTimestamp),
        ("actual_exposure_us", Rs2FrameMetadata::ActualExposure),
        ("gain_level", Rs2FrameMetadata::GainLevel),
        ("actual_fps", Rs2FrameMetadata::ActualFps),
        ("backend_timestamp_us", Rs2FrameMetadata::BackendTimestamp),
        ("time_of_arrival_us", Rs2FrameMetadata::TimeOfArrival),
    ] {
        if let Some(value) = frame.metadata(key) {
            attributes.insert(name.to_string(), value.to_string());
        }
    }
}

fn timestamp_advances(previous: Option<i128>, current: i128) -> bool {
    previous.is_none_or(|previous| current > previous)
}

/// A pair must be closer than half the faster stream's period. A full-period
/// offset is an adjacent exposure, even when the group tolerance admits it.
fn exposures_match(color_ns: i128, depth_ns: i128, color_fps: i32, depth_fps: i32) -> bool {
    if color_fps <= 0 || depth_fps <= 0 {
        return false;
    }
    let half_period_ns = 1_000_000_000 / (2 * color_fps.max(depth_fps) as u128);
    color_ns.abs_diff(depth_ns) < half_period_ns
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dds_context_binds_discovery_to_the_owner_address_without_usb() {
        let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("config/vision.toml");
        let config = crate::config::VisionConfig::load(path).unwrap();
        let mut camera = config
            .cameras
            .realsense
            .into_iter()
            .find(|camera| camera.transport == RealSenseTransport::Dds)
            .expect("a DDS camera in the deployed config");
        assert!(dds_context_settings(&camera).is_err());
        camera.dds_address = Some("192.0.2.52".parse().unwrap());
        let settings: serde_json::Value =
            serde_json::from_str(&dds_context_settings(&camera).unwrap()).unwrap();
        assert_eq!(settings["dds"]["enabled"], true);
        assert_eq!(
            settings["dds"]["udp"]["whitelist"],
            serde_json::json!(["192.0.2.52"])
        );
        assert_eq!(settings["device-mask"], 0x1ff);
    }

    fn calibration_record(intrinsics: &serde_json::Value) -> FrameRecord {
        FrameRecord {
            metadata: FrameMetadata {
                sensor_name: "fixture_color".into(),
                sensor_kind: SensorKind::RealSense,
                sequence: 1,
                profile: StreamProfile {
                    stream: "color".into(),
                    width: 640,
                    height: 360,
                    fps_num: 30,
                    fps_den: 1,
                    format: PixelFormat::Bgr8,
                },
                timestamps: FrameTimestamps {
                    source_ns: Some(1),
                    source_domain: TimestampDomain::RealSenseHardware,
                    rtp_timestamp: None,
                    pipeline_pts_ns: None,
                    pipeline_dts_ns: None,
                    host_monotonic_ns: 1,
                    host_unix_ns: 1,
                    normalized_unix_ns: Some(1),
                },
                dropped_before: 0,
                calibration_id: None,
                flags: vec![],
                attributes: BTreeMap::from([("intrinsics".into(), intrinsics.to_string())]),
            },
            payload: RecordedPayload::Video {
                format: PixelFormat::Bgr8,
                width: 640,
                height: 360,
                bytes: vec![],
            },
        }
    }

    #[test]
    fn native_alignment_provenance_preserves_captured_pixels() {
        let calibration = serde_json::json!({"color":{"fx":100}, "depth":{"fx":200},
            "rotation":[1,0,0,0,1,0,0,0,1], "translation":[0.01,0,0.003]});
        let mut records = vec![calibration_record(&calibration["color"])];
        records[0].payload = RecordedPayload::Video {
            format: PixelFormat::Bgr8,
            width: 1,
            height: 1,
            bytes: vec![23, 45, 67],
        };
        let pixels = records[0].payload.bytes().to_vec();
        let intrinsics = records[0].metadata.attributes["intrinsics"].clone();
        retain_alignment_calibration(&mut records, &calibration);
        assert_eq!(records[0].payload.bytes(), pixels);
        assert_eq!(records[0].metadata.attributes["intrinsics"], intrinsics);
        let retained: serde_json::Value =
            serde_json::from_str(&records[0].metadata.attributes["alignment_calibration"]).unwrap();
        assert_eq!(retained, calibration);
    }

    #[test]
    fn a_record_parsed_back_from_text_still_matches_its_calibration() {
        // The deployed owner is built alone (`cargo build -p`), so it cannot
        // rely on another crate's serde_json features for exact float parsing:
        // an intrinsics value one ULP off must still be the same camera model.
        let color = serde_json::json!({"schema":"tatbot.camera-intrinsics/1","width":640,"height":480,
            "fx":391.6629333496094,"fy":391.2059631347656,"ppx":326.2788391113281,"ppy":238.8282470703125,
            "distortion_model":"BrownConradyInverse",
            "distortion_coefficients":[-0.05274029076099396,0.0588311143219471,-0.00002708945976337418,0.00041671955841593444,-0.019236048683524132]});
        let calibration = serde_json::json!({"color":color});
        let mut nudged = color.clone();
        nudged["fx"] = serde_json::json!(391.6629333496094_f64 + f64::EPSILON * 512.0);
        let records = vec![calibration_record(&color), calibration_record(&nudged)];
        assert!(aligned_records_match(&records, &calibration));
        let mut other_size = color.clone();
        other_size["width"] = serde_json::json!(1280);
        assert!(!aligned_records_match(
            &[calibration_record(&color), calibration_record(&other_size)],
            &calibration
        ));
        let mut other_model = color.clone();
        other_model["distortion_coefficients"] = serde_json::json!([0.0, 0.0, 0.0, 0.0, 0.0]);
        assert!(!aligned_records_match(
            &[calibration_record(&color), calibration_record(&other_model)],
            &calibration
        ));
        assert!(!aligned_records_match(
            &[calibration_record(&color)],
            &calibration
        ));
    }

    #[test]
    fn stale_alignment_and_calibration_changes_during_processing_are_withheld() {
        let old = serde_json::json!({"width":640,"height":360,"fx":321.98846,"fy":321.49942,
            "ppx":323.39355,"ppy":181.35078,"distortion_model":"BrownConrady",
            "distortion_coefficients":[-0.052697,0.056955,-0.000834,-0.000087,-0.017889]});
        let mut current = old.clone();
        current["fx"] = serde_json::json!(323.12872);
        current["fy"] = serde_json::json!(322.63797);
        let calibration = serde_json::json!({"color":current});
        // A warm camera can update its focal length without a new profile ID.
        // The cached aligned depth must not be treated as the current RGB grid.
        assert!(!aligned_records_match(
            &[calibration_record(&current), calibration_record(&old)],
            &calibration
        ));
        assert!(aligned_records_match(
            &[calibration_record(&current), calibration_record(&current)],
            &calibration
        ));
        // Both records agreeing is insufficient if calibration changed after
        // alignment began; that exposure was mapped with a different model.
        assert!(!aligned_records_match(
            &[calibration_record(&current), calibration_record(&current)],
            &serde_json::json!({"color":old})
        ));
        for key in [
            "width",
            "ppx",
            "distortion_model",
            "distortion_coefficients",
        ] {
            let mut changed = current.clone();
            changed[key] = serde_json::Value::Null;
            assert!(!aligned_records_match(
                &[calibration_record(&current), calibration_record(&changed)],
                &calibration
            ));
        }
        assert!(!aligned_records_match(&[], &calibration));
    }

    #[test]
    fn source_exposure_times_reject_adjacent_sdk_rgbd_frames() {
        // Recorded relative source-time deltas, independent of counter origins.
        assert!(exposures_match(304_000, 0, 30, 30));
        assert!(exposures_match(36_000, 0, 30, 30));
        assert!(!exposures_match(33_369_000, 0, 30, 30));
        assert!(!exposures_match(0, 33_297_000, 30, 30));
        assert!(!exposures_match(16_666_666, 0, 30, 30));
        assert!(!exposures_match(20_000_000, 0, 15, 30));
        assert!(exposures_match(20_000_000, 0, 15, 15));
        assert!(!exposures_match(0, 0, 0, 30));
        assert!(!exposures_match(i128::MIN, i128::MAX, 30, 30));
    }

    #[test]
    fn repeated_dds_frameset_timestamp_is_not_accepted_twice() {
        assert!(timestamp_advances(None, 100));
        assert!(timestamp_advances(Some(100), 101));
        assert!(!timestamp_advances(Some(100), 100));
        assert!(!timestamp_advances(Some(100), 99));
    }

    #[test]
    fn sdk_queue_drain_requires_pending_and_propagates_poll_failure() {
        let mut calls = 0;
        let mut old_aligned_output = vec![42];
        let drained = drain_ready_queue(
            || {
                calls += 1;
                Ok(if calls <= 2 {
                    Poll::Ready(())
                } else {
                    Poll::Pending
                })
            },
            || {
                old_aligned_output.clear();
                Ok(())
            },
        )
        .unwrap();
        assert_eq!(drained, 2);
        assert_eq!(calls, 3);
        assert!(
            old_aligned_output.is_empty(),
            "old aligned output survived the generation barrier"
        );
        let mut cleared_on_failure = false;
        assert!(
            drain_ready_queue::<()>(
                || Ok(Poll::Ready(())),
                || {
                    cleared_on_failure = true;
                    Ok(())
                }
            )
            .is_err()
        );
        assert!(!cleared_on_failure);
        assert!(drain_ready_queue::<()>(|| anyhow::bail!("SDK poll failed"), || Ok(())).is_err());
        assert!(
            drain_ready_queue::<()>(|| Ok(Poll::Pending), || anyhow::bail!("align reset failed"))
                .is_err()
        );
    }
}
