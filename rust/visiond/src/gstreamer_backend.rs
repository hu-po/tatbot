//! GStreamer RTSP capture backend for the Amcrest PoE cameras.

use std::{
    collections::{BTreeMap, VecDeque},
    str::FromStr,
    sync::{
        Arc, Mutex,
        atomic::{AtomicU64, Ordering},
    },
    time::{Duration, Instant},
};

use anyhow::{Context, Result, anyhow};
use gstreamer as gst;
use gstreamer::prelude::*;
use gstreamer_app as gst_app;
use url::Url;

use crate::{
    FrameMetadata, FrameRecord, FrameTimestamps, PixelFormat, RecordedPayload, SensorKind,
    StreamProfile,
    config::PoeCameraConfig,
    health::{HealthSnapshot, SensorHealth},
    rtcp::{RtpClockMapper, parse_rtp_header, parse_sender_reports},
    sync::{ClockOffsetEstimator, SyncAssessment},
    time::{MonotonicClock, TimestampDomain},
};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PoeStream {
    Main,
    Sub,
}

/// Allocation-light result for consumers that only monitor stream health.
///
/// This deliberately does not contain `FrameMetadata`: constructing that
/// record allocates caps, flags and a large attribute map for every access
/// unit. The always-on monitor needs only these three values plus the health
/// snapshot maintained by `PoeRtspCapture`.
#[derive(Debug, Clone)]
pub struct FrameObservation {
    pub payload_bytes: u64,
    pub source_domain: TimestampDomain,
    pub clock_offset_large: bool,
}

impl PoeStream {
    fn profile<'a>(self, config: &'a PoeCameraConfig) -> Result<&'a StreamProfile> {
        match self {
            Self::Main => Ok(&config.main),
            Self::Sub => config
                .sub
                .as_ref()
                .context("PoE camera has no substream configured"),
        }
    }

    fn subtype(self) -> &'static str {
        match self {
            Self::Main => "0",
            Self::Sub => "1",
        }
    }
}

#[derive(Debug)]
pub struct PoeRtspCapture {
    pipeline: gst::Pipeline,
    sink: gst_app::AppSink,
    camera: PoeCameraConfig,
    configured_profile: StreamProfile,
    configured_jitter_latency_ms: u32,
    configured_converter_threads: u32,
    trust_camera_ntp: bool,
    decoded: bool,
    keyframes_only: bool,
    clock: MonotonicClock,
    health: SensorHealth,
    sequence: u64,
    frame_timings: FrameTimingHistory,
    rtcp_reports_seen: Arc<AtomicU64>,
    clock_offset: ClockOffsetEstimator,
    last_rtp_timestamp: Option<u32>,
}

#[derive(Debug, Clone, Copy)]
struct FrameTiming {
    rtp_timestamp: u32,
    source_ns: Option<i128>,
    rtsp_rtp_first_seen: Option<Instant>,
    rtsp_rtp_last_seen: Option<Instant>,
    depay_rtp_first_seen: Instant,
    depay_rtp_last_seen: Instant,
    depay_out: Option<Instant>,
    parser_out: Option<Instant>,
    decoder_in: Option<Instant>,
    decoder_out: Option<Instant>,
    convert_out: Option<Instant>,
    appsink_in: Option<Instant>,
}

type FrameTimingHistory = Arc<Mutex<VecDeque<(u64, FrameTiming)>>>;

#[derive(Debug, Clone, Copy)]
struct RtpArrivalTiming {
    first_seen: Instant,
    last_seen: Instant,
}

type RtpArrivalHistory = Arc<Mutex<VecDeque<(u32, RtpArrivalTiming)>>>;

impl PoeRtspCapture {
    pub fn new(
        camera: PoeCameraConfig,
        stream: PoeStream,
        password: &str,
        decoded: bool,
    ) -> Result<Self> {
        Self::new_with_options(camera, stream, password, decoded, false)
    }

    /// Encoded capture with only the probes required by the health monitor.
    /// Detailed frame-stage timing is valuable for bounded experiments but is
    /// needless per-packet work for an always-on observer.
    pub fn new_monitor(camera: PoeCameraConfig, stream: PoeStream, password: &str) -> Result<Self> {
        Self::new_with_capture_options(camera, stream, password, false, false, false)
    }

    /// `keyframes_only` drops delta units right after the depayloader, so the
    /// decoder only ever sees I-frames. With the cameras' GOP=10 at 20 fps
    /// that yields ~2 Hz full-resolution frames at ~1/10 the decode cost —
    /// the intended reconstruction-keyframe path.
    pub fn new_with_options(
        camera: PoeCameraConfig,
        stream: PoeStream,
        password: &str,
        decoded: bool,
        keyframes_only: bool,
    ) -> Result<Self> {
        Self::new_with_capture_options(camera, stream, password, decoded, keyframes_only, true)
    }

    fn new_with_capture_options(
        camera: PoeCameraConfig,
        stream: PoeStream,
        password: &str,
        decoded: bool,
        keyframes_only: bool,
        detailed_timing: bool,
    ) -> Result<Self> {
        gst::init().context("initializing GStreamer")?;
        let mut configured_profile = stream.profile(&camera)?.clone();
        if decoded {
            configured_profile.format = effective_decoded_pixel_format()?;
        }
        let uri = rtsp_uri(&camera, stream, password)?;
        let latency = effective_jitter_latency_ms(camera.gstreamer_latency_ms)?;
        let converter_threads = effective_converter_threads()?;
        let trust_camera_ntp = std::env::var_os("TATBOT_VISIOND_TRUST_CAMERA_NTP")
            .is_some_and(|value| env_flag_enabled(&value.to_string_lossy()));
        // The shared capture-event channel applies the live backpressure
        // bound. Keep a short appsink burst queue so NVDEC does not starve or
        // introduce multi-second gaps while detector work runs on the CPU.
        let queue_capacity = 8_u32;
        // Prefer the Jetson hardware decoder when present (NVDEC keeps five
        // 5 MP decodes off the CPU); TATBOT_VISIOND_SW_DECODE=1 forces the
        // portable software path. General capture defaults to BGR, while the
        // fiducial-only wrapper selects Y8 to avoid a full-resolution
        // BGRx-to-BGR copy followed by another grayscale conversion.
        let output_chain = if decoded {
            let hardware = std::env::var_os("TATBOT_VISIOND_SW_DECODE").is_none()
                && gst::ElementFactory::find("nvv4l2decoder").is_some()
                && gst::ElementFactory::find("nvvidconv").is_some();
            if hardware {
                let low_latency = std::env::var_os("TATBOT_VISIOND_NVDEC_LOW_LATENCY")
                    .is_some_and(|value| env_flag_enabled(&value.to_string_lossy()));
                tracing::info!(camera = %camera.name, low_latency, trust_camera_ntp, jitter_latency_ms = latency, converter_threads, decoded_format = ?configured_profile.format, "using nvv4l2decoder (NVDEC) for H264 decode");
                let decoder_options = if low_latency {
                    " disable-dpb=true enable-max-performance=true"
                } else {
                    ""
                };
                match configured_profile.format {
                    PixelFormat::Y8 => format!(
                        " ! h264parse name=parser ! nvv4l2decoder name=decoder{decoder_options} ! nvvidconv name=colorspace ! video/x-raw,format=GRAY8"
                    ),
                    PixelFormat::Bgr8 => format!(
                        " ! h264parse name=parser ! nvv4l2decoder name=decoder{decoder_options} ! nvvidconv name=nvconvert ! video/x-raw,format=BGRx ! videoconvert name=colorspace n-threads={converter_threads} ! video/x-raw,format=BGR"
                    ),
                    other => anyhow::bail!("unsupported decoded PoE pixel format {other:?}"),
                }
            } else {
                let caps = match configured_profile.format {
                    PixelFormat::Bgr8 => "BGR",
                    PixelFormat::Y8 => "GRAY8",
                    other => anyhow::bail!("unsupported decoded PoE pixel format {other:?}"),
                };
                format!(
                    " ! h264parse name=parser ! avdec_h264 name=decoder ! videoconvert name=colorspace n-threads={converter_threads} ! video/x-raw,format={caps}"
                )
            }
        } else {
            String::new()
        };
        // `add-reference-timestamp-meta` landed in rtspsrc with GStreamer
        // 1.22. Older runtimes (e.g. JetPack 6 ships 1.20) still get camera
        // NTP time through the RTCP sender-report mapper fallback below.
        let (major, minor, ..) = gst::version();
        let reference_meta = if (major, minor) >= (1, 22) {
            " add-reference-timestamp-meta=true"
        } else {
            ""
        };
        let keyframe_filter = if keyframes_only {
            " ! identity drop-buffer-flags=delta-unit"
        } else {
            ""
        };
        let pipeline_description = format!(
            "rtspsrc name=source location=\"{}\" protocols={} ntp-sync=true ntp-time-source=ntp buffer-mode=synced{} latency={} do-rtsp-keep-alive=true ! rtph264depay name=depay{}{} ! appsink name=sink emit-signals=false sync=false max-buffers={} drop=true",
            gst_quote(&uri),
            camera.transport,
            reference_meta,
            latency,
            keyframe_filter,
            output_chain,
            queue_capacity,
        );
        let element = gst::parse::launch(&pipeline_description)
            .with_context(|| format!("building GStreamer pipeline for {}", camera.name))?;
        let pipeline = element
            .dynamic_cast::<gst::Pipeline>()
            .map_err(|_| anyhow!("GStreamer launch did not return a pipeline"))?;
        let sink = pipeline
            .by_name("sink")
            .ok_or_else(|| anyhow!("pipeline has no appsink"))?
            .dynamic_cast::<gst_app::AppSink>()
            .map_err(|_| anyhow!("pipeline sink is not an appsink"))?;
        let (frame_timings, rtcp_reports_seen) = if detailed_timing {
            install_reference_timestamp_probes(&pipeline)?
        } else {
            install_monitor_timestamp_probes(&pipeline)?
        };
        pipeline
            .set_state(gst::State::Playing)
            .map_err(|error| anyhow!("starting {}: {error:?}", camera.name))?;
        Ok(Self {
            health: SensorHealth::new(camera.name.clone()),
            pipeline,
            sink,
            camera,
            configured_profile,
            configured_jitter_latency_ms: latency,
            configured_converter_threads: converter_threads,
            trust_camera_ntp,
            decoded,
            keyframes_only,
            clock: MonotonicClock::default(),
            sequence: 0,
            frame_timings,
            rtcp_reports_seen,
            clock_offset: ClockOffsetEstimator::new(128).map_err(anyhow::Error::msg)?,
            last_rtp_timestamp: None,
        })
    }

    pub fn next_frame(&mut self, timeout: Duration) -> Result<Option<FrameRecord>> {
        self.next_frame_with_policy(timeout)
    }

    /// Pull only the health information for an encoded access unit.
    ///
    /// This does not map/copy the payload and does not build a `FrameRecord`.
    /// The distinct return type prevents recorders and detectors from treating
    /// an observation as a real image or encoded frame.
    pub fn next_observation(&mut self, timeout: Duration) -> Result<Option<FrameObservation>> {
        let timeout_ns = timeout.as_nanos().min(u64::MAX as u128) as u64;
        let Some(sample) = self
            .sink
            .try_pull_sample(gst::ClockTime::from_nseconds(timeout_ns))
        else {
            if let Some(error) = self.poll_error() {
                self.health.error(error.to_string());
                return Err(error);
            }
            return Ok(None);
        };

        let buffer = sample.buffer().context("GStreamer sample has no buffer")?;
        let payload_bytes = buffer.size() as u64;
        let clock_sample = self.clock.now();
        let timing = buffer.pts().and_then(|pts| {
            self.frame_timings.lock().ok().and_then(|history| {
                history
                    .iter()
                    .rev()
                    .find(|(candidate, _)| *candidate == pts.nseconds())
                    .map(|(_, timing)| *timing)
            })
        });

        let (source_ns, source_domain) =
            if let Some(reference) = buffer.meta::<gst::ReferenceTimestampMeta>() {
                (
                    Some(reference.timestamp().nseconds() as i128),
                    if reference.reference().to_string().contains("ntp") {
                        TimestampDomain::CameraNtp
                    } else {
                        TimestampDomain::Unknown
                    },
                )
            } else if let Some(timestamp) = timing.and_then(|value| value.source_ns) {
                (Some(timestamp), TimestampDomain::CameraNtp)
            } else {
                (None, TimestampDomain::Unknown)
            };

        if let Some(timing) = timing {
            if !self.keyframes_only
                && let Some(previous) = self.last_rtp_timestamp
            {
                let delta = timing.rtp_timestamp.wrapping_sub(previous);
                let expected_ticks = (90_000.0 / self.configured_profile.fps()).round() as u32;
                let gap_threshold = expected_ticks.saturating_mul(3) / 2;
                if expected_ticks > 0 && delta > gap_threshold && delta < (1_u32 << 31) {
                    let missing = u64::from(delta / expected_ticks).saturating_sub(1).max(1);
                    self.health.frame_dropped(missing);
                }
            }
            self.last_rtp_timestamp = Some(timing.rtp_timestamp);
        }

        let mut clock_offset_large = false;
        if let Some(source_ns) = source_ns {
            self.clock_offset.observe(clock_sample.unix_ns, source_ns);
            clock_offset_large = self
                .clock_offset
                .assessment()
                .median_offset_ns
                .is_some_and(|offset| offset.unsigned_abs() > 100_000_000);
        }
        let sequence = self.sequence;
        self.sequence = self.sequence.saturating_add(1);
        self.health
            .frame_received(sequence, clock_sample.monotonic_ns);

        Ok(Some(FrameObservation {
            payload_bytes,
            source_domain,
            clock_offset_large,
        }))
    }

    fn next_frame_with_policy(&mut self, timeout: Duration) -> Result<Option<FrameRecord>> {
        let timeout_ns = timeout.as_nanos().min(u64::MAX as u128) as u64;
        let Some(sample) = self
            .sink
            .try_pull_sample(gst::ClockTime::from_nseconds(timeout_ns))
        else {
            if let Some(error) = self.poll_error() {
                self.health.error(error.to_string());
                return Err(error);
            }
            return Ok(None);
        };

        let sample_pulled_at = Instant::now();
        let buffer = sample.buffer().context("GStreamer sample has no buffer")?;
        let captured_payload_bytes = buffer.size() as u64;
        let payload_copy_started_at = Instant::now();
        let map = buffer.map_readable().context("mapping GStreamer buffer")?;
        let payload = map.as_slice().to_vec();
        let payload_copied_at = Instant::now();
        let clock_sample = self.clock.now();
        let pipeline_running_time_ns = self
            .pipeline
            .current_running_time()
            .map(|timestamp| timestamp.nseconds());
        let caps = sample.caps().context("GStreamer sample has no caps")?;
        let (profile, caps_complete) = active_profile(caps, &self.configured_profile)?;
        let mut flags = Vec::new();
        if !caps_complete {
            flags.push("caps_missing_dimensions_or_framerate".to_string());
        }
        if self.decoded {
            flags.push(match self.configured_profile.format {
                PixelFormat::Y8 => "decoded_y8".to_string(),
                _ => "decoded_bgr".to_string(),
            });
        }
        let buffer_flags = buffer.flags();
        if buffer_flags.contains(gst::BufferFlags::DISCONT) {
            flags.push("discontinuity".to_string());
        }
        if buffer_flags.contains(gst::BufferFlags::CORRUPTED) {
            flags.push("corrupted".to_string());
        }
        if buffer_flags.contains(gst::BufferFlags::DELTA_UNIT) {
            flags.push("delta_unit".to_string());
        }
        let mut source_ns = None;
        let mut source_domain = TimestampDomain::Unknown;
        let pipeline_pts_ns = buffer.pts().map(|timestamp| timestamp.nseconds());
        let pipeline_dts_ns = buffer.dts().map(|timestamp| timestamp.nseconds());
        let timing = pipeline_pts_ns.and_then(|pts| {
            self.frame_timings.lock().ok().and_then(|history| {
                history
                    .iter()
                    .rev()
                    .find(|(candidate, _)| *candidate == pts)
                    .map(|(_, timing)| *timing)
            })
        });
        if let Some(reference) = buffer.meta::<gst::ReferenceTimestampMeta>() {
            source_ns = Some(reference.timestamp().nseconds() as i128);
            source_domain = if reference.reference().to_string().contains("ntp") {
                TimestampDomain::CameraNtp
            } else {
                TimestampDomain::Unknown
            };
            flags.push("reference_timestamp_meta".to_string());
            if timing.and_then(|timing| timing.source_ns).is_some() {
                flags.push("rtcp_sender_report_timestamp".to_string());
            }
        } else if let Some(timestamp) = timing.and_then(|timing| timing.source_ns) {
            source_ns = Some(timestamp);
            source_domain = TimestampDomain::CameraNtp;
            flags.push("rtcp_sender_report_timestamp".to_string());
        } else if buffer.pts().is_some() {
            flags.push("gstreamer_pts_only".to_string());
            flags.push("ntp_sync_requested".to_string());
        }
        let mut rtp_gap_frames = 0_u64;
        if let Some(timing) = timing {
            // With keyframes_only the delta units are dropped on purpose, so
            // RTP timestamp gaps are expected and are not frame loss.
            if self.keyframes_only {
                flags.push("keyframes_only".to_string());
            } else if let Some(previous) = self.last_rtp_timestamp {
                let delta = timing.rtp_timestamp.wrapping_sub(previous);
                let expected_ticks = (90_000.0 / self.configured_profile.fps()).round() as u32;
                let gap_threshold = expected_ticks.saturating_mul(3) / 2;
                if expected_ticks > 0 && delta > gap_threshold && delta < (1_u32 << 31) {
                    rtp_gap_frames = u64::from(delta / expected_ticks).saturating_sub(1).max(1);
                    self.health.frame_dropped(rtp_gap_frames);
                    flags.push("rtp_timestamp_gap".to_string());
                }
            }
            self.last_rtp_timestamp = Some(timing.rtp_timestamp);
        }
        let mut normalized_unix_ns = None;
        let mut clock_attributes = BTreeMap::new();
        if let Some(source_ns) = source_ns {
            self.clock_offset.observe(clock_sample.unix_ns, source_ns);
            let assessment = self.clock_offset.assessment();
            normalized_unix_ns = assessment
                .median_offset_ns
                .map(|offset| source_ns.saturating_add(offset));
            add_clock_assessment(&mut clock_attributes, &assessment);
            flags.push("camera_clock_offset_corrected".to_string());
            if assessment
                .median_offset_ns
                .is_some_and(|offset| offset.unsigned_abs() > 100_000_000)
            {
                flags.push("camera_clock_offset_large".to_string());
            }
        }
        let pipeline_capture_timing =
            pipeline_running_time_ns
                .zip(pipeline_pts_ns)
                .map(|(running_ns, pts_ns)| {
                    pipeline_capture_time(clock_sample.unix_ns, running_ns, pts_ns)
                });
        if self.trust_camera_ntp
            && let Some(camera_unix_ns) = source_ns
        {
            let host_delta_ns = camera_unix_ns - clock_sample.unix_ns;
            // A decoded frame is necessarily older than the instant at which
            // appsink receives it.  On the Jetson, startup and a full-frame
            // detector pass can put that ordinary delivery age around
            // 300-500 ms.  Reject clocks that are implausibly in the future or
            // frames older than the bounded live pipeline, but do not mistake
            // transport/decode age for NTP skew.
            if !trusted_camera_capture_delta_is_plausible(host_delta_ns) {
                anyhow::bail!(
                    "{} camera capture time differs from host by {:.1} ms; refusing trusted-camera-NTP mode",
                    self.camera.name,
                    host_delta_ns as f64 / 1e6
                );
            }
            // A LAN Chrony server now disciplines every camera. Preserve the
            // camera's actual capture timeline instead of estimating capture
            // age from independent GStreamer pipeline startup phases.
            normalized_unix_ns = Some(camera_unix_ns);
            flags.push("trusted_camera_ntp".to_string());
        } else if let Some((_capture_age_ns, capture_unix_ns)) = pipeline_capture_timing {
            // When raw NTP has not been explicitly verified, map each camera
            // source timestamp to host time. GStreamer's PTS and pipeline
            // running time share a clock domain, so their difference retains
            // transport/decode age without trusting the camera wall clock.
            normalized_unix_ns = Some(capture_unix_ns);
            flags.push("pipeline_pts_host_normalized".to_string());
        }
        let timestamps = FrameTimestamps {
            source_ns,
            source_domain,
            rtp_timestamp: timing.map(|timing| timing.rtp_timestamp as u64),
            pipeline_pts_ns,
            pipeline_dts_ns,
            host_monotonic_ns: clock_sample.monotonic_ns,
            host_unix_ns: clock_sample.unix_ns,
            normalized_unix_ns,
        };
        let sequence = self.sequence;
        self.sequence = self.sequence.saturating_add(1);
        self.health
            .frame_received(sequence, clock_sample.monotonic_ns);
        let dropped_before = self.health.snapshot().frames_dropped;
        if profile != self.configured_profile {
            flags.push("active_profile_differs_from_config".to_string());
        }
        let mut attributes = BTreeMap::from([
            ("caps".to_string(), caps.to_string()),
            (
                "rtcp_sender_reports_seen".to_string(),
                self.rtcp_reports_seen.load(Ordering::Relaxed).to_string(),
            ),
            (
                "rtp_timestamp".to_string(),
                timing
                    .map(|timing| timing.rtp_timestamp.to_string())
                    .unwrap_or_default(),
            ),
            (
                "pipeline_pts_ns".to_string(),
                pipeline_pts_ns
                    .map(|value| value.to_string())
                    .unwrap_or_default(),
            ),
            (
                "pipeline_dts_ns".to_string(),
                pipeline_dts_ns
                    .map(|value| value.to_string())
                    .unwrap_or_default(),
            ),
            (
                "pipeline_running_time_ns".to_string(),
                pipeline_running_time_ns
                    .map(|value| value.to_string())
                    .unwrap_or_default(),
            ),
            (
                "configured_converter_threads".to_string(),
                self.configured_converter_threads.to_string(),
            ),
            (
                "captured_payload_bytes".to_string(),
                captured_payload_bytes.to_string(),
            ),
        ]);
        if let (Some(running_ns), Some(pts_ns)) = (pipeline_running_time_ns, pipeline_pts_ns) {
            let pts_to_now_ns = i128::from(running_ns) - i128::from(pts_ns);
            attributes.insert(
                "pipeline_pts_to_now_ns".to_string(),
                pts_to_now_ns.to_string(),
            );
            attributes.insert(
                "configured_jitter_latency_ns".to_string(),
                (u64::from(self.configured_jitter_latency_ms) * 1_000_000).to_string(),
            );
            attributes.insert(
                "pipeline_capture_age_ns".to_string(),
                pts_to_now_ns.to_string(),
            );
        }
        if let Some(timing) = timing {
            add_pipeline_stage_attributes(&mut attributes, &timing, sample_pulled_at);
        }
        add_elapsed_attribute(
            &mut attributes,
            "pipeline_payload_copy_ns",
            payload_copy_started_at,
            payload_copied_at,
        );
        if let (Some(timing), Some(capture_age_ns)) = (
            timing,
            attributes
                .get("pipeline_capture_age_ns")
                .and_then(|value| value.parse::<u128>().ok()),
        ) && let Some(first_seen) = timing.rtsp_rtp_first_seen
            && let Some(post_arrival) = payload_copied_at.checked_duration_since(first_seen)
            && capture_age_ns >= post_arrival.as_nanos()
        {
            attributes.insert(
                "pipeline_source_pts_to_rtsp_rtp_first_ns".to_string(),
                (capture_age_ns - post_arrival.as_nanos()).to_string(),
            );
        }
        if rtp_gap_frames > 0 {
            attributes.insert(
                "rtp_timestamp_gap_frames".to_string(),
                rtp_gap_frames.to_string(),
            );
        }
        attributes.extend(clock_attributes);
        let frame = FrameRecord {
            metadata: FrameMetadata {
                sensor_name: self.camera.name.clone(),
                sensor_kind: SensorKind::PoE,
                sequence,
                profile: profile.clone(),
                timestamps,
                dropped_before,
                calibration_id: None,
                flags,
                attributes,
            },
            payload: if self.decoded {
                RecordedPayload::Video {
                    format: profile.format,
                    width: profile.width,
                    height: profile.height,
                    bytes: payload,
                }
            } else {
                RecordedPayload::Encoded {
                    format: PixelFormat::H264,
                    bytes: payload,
                }
            },
        };
        frame.validate().map_err(anyhow::Error::msg)?;
        Ok(Some(frame))
    }

    pub fn health(&self) -> HealthSnapshot {
        self.health.snapshot()
    }

    pub fn stop(&self) -> Result<()> {
        self.pipeline
            .set_state(gst::State::Null)
            .map(|_| ())
            .map_err(|error| anyhow!("stopping {}: {error:?}", self.camera.name))
    }

    fn poll_error(&self) -> Option<anyhow::Error> {
        let bus = self.pipeline.bus()?;
        while let Some(message) = bus.timed_pop(gst::ClockTime::ZERO) {
            match message.view() {
                gst::MessageView::Error(error) => {
                    return Some(anyhow!(
                        "{}: {} ({:?})",
                        self.camera.name,
                        error.error(),
                        error.debug()
                    ));
                }
                gst::MessageView::Eos(..) => {
                    return Some(anyhow!("{}: end of stream", self.camera.name));
                }
                _ => {}
            }
        }
        None
    }
}

/// `rtph264depay` does not currently copy `GstReferenceTimestampMeta` from
/// its RTP input buffers to the H264 access-unit buffers. Preserve the NTP
/// reference timestamp across that boundary with a small pad-probe bridge.
/// Without this bridge the application would only see per-pipeline running
/// time, which cannot be compared between independent RTSP pipelines.
fn install_reference_timestamp_probes(
    pipeline: &gst::Pipeline,
) -> Result<(FrameTimingHistory, Arc<AtomicU64>)> {
    let source = pipeline
        .by_name("source")
        .context("pipeline has no named RTSP source")?;
    let depay = pipeline
        .by_name("depay")
        .context("pipeline has no named H264 depayloader")?;
    let sink_pad = depay
        .static_pad("sink")
        .context("H264 depayloader has no sink pad")?;
    let source_pad = depay
        .static_pad("src")
        .context("H264 depayloader has no source pad")?;
    let frame_timings = Arc::new(Mutex::new(VecDeque::<(u64, FrameTiming)>::new()));
    let clock_mapper = Arc::new(Mutex::new(
        RtpClockMapper::new(64).map_err(anyhow::Error::msg)?,
    ));
    let rtcp_reports_seen = Arc::new(AtomicU64::new(0));
    let rtp_arrivals = Arc::new(Mutex::new(VecDeque::<(u32, RtpArrivalTiming)>::new()));

    // rtspsrc creates an rtpbin/rtpsession manager internally. Its
    // recv_rtcp_sink pads carry the raw RTCP compound packets, which gives us
    // a standards-level fallback even when ReferenceTimestampMeta is dropped
    // later in the pipeline.
    let manager_mapper = Arc::clone(&clock_mapper);
    let manager_report_count = Arc::clone(&rtcp_reports_seen);
    let manager_rtp_arrivals = Arc::clone(&rtp_arrivals);
    source.connect("new-manager", false, move |args| {
        let Some(manager) = args
            .get(1)
            .and_then(|value| value.get::<gst::Element>().ok())
        else {
            return None;
        };
        let pad_mapper = Arc::clone(&manager_mapper);
        let pad_report_count = Arc::clone(&manager_report_count);
        let pad_rtp_arrivals = Arc::clone(&manager_rtp_arrivals);
        for pad in manager.pads() {
            install_rtcp_probe(&pad, &pad_mapper, &pad_report_count);
            install_rtp_arrival_probe(&pad, &pad_rtp_arrivals);
        }
        manager.connect("pad-added", false, move |pad_args| {
            let Some(pad) = pad_args
                .get(1)
                .and_then(|value| value.get::<gst::Pad>().ok())
            else {
                return None;
            };
            install_rtcp_probe(&pad, &pad_mapper, &pad_report_count);
            install_rtp_arrival_probe(&pad, &pad_rtp_arrivals);
            None
        });
        None
    });

    let input_timings = Arc::clone(&frame_timings);
    let input_mapper = Arc::clone(&clock_mapper);
    let input_rtp_arrivals = Arc::clone(&rtp_arrivals);
    sink_pad
        .add_probe(gst::PadProbeType::BUFFER, move |_pad, info| {
            if let Some(buffer) = info.buffer() {
                if let (Some(pts), Some(map)) = (buffer.pts(), buffer.map_readable().ok()) {
                    if let Some((ssrc, rtp_timestamp)) = parse_rtp_header(map.as_slice()) {
                        let source_ns = buffer
                            .meta::<gst::ReferenceTimestampMeta>()
                            .map(|reference| reference.timestamp().nseconds() as i128)
                            .or_else(|| {
                                input_mapper
                                    .lock()
                                    .ok()
                                    .and_then(|mapper| mapper.estimate(ssrc, rtp_timestamp))
                                    .map(|estimate| estimate.ntp_unix_ns)
                            });
                        let now = Instant::now();
                        let arrival = input_rtp_arrivals.lock().ok().and_then(|arrivals| {
                            arrivals
                                .iter()
                                .rev()
                                .find(|(candidate, _)| *candidate == rtp_timestamp)
                                .map(|(_, timing)| *timing)
                        });
                        let mut timings = input_timings.lock().expect("timestamp probe mutex");
                        if let Some((_, timing)) = timings
                            .iter_mut()
                            .rev()
                            .find(|(input_pts, _)| *input_pts == pts.nseconds())
                        {
                            timing.rtp_timestamp = rtp_timestamp;
                            timing.source_ns = source_ns.or(timing.source_ns);
                            timing.rtsp_rtp_first_seen = arrival.map(|value| value.first_seen);
                            timing.rtsp_rtp_last_seen = arrival.map(|value| value.last_seen);
                            timing.depay_rtp_last_seen = now;
                        } else {
                            timings.push_back((
                                pts.nseconds(),
                                FrameTiming {
                                    rtp_timestamp,
                                    source_ns,
                                    rtsp_rtp_first_seen: arrival.map(|value| value.first_seen),
                                    rtsp_rtp_last_seen: arrival.map(|value| value.last_seen),
                                    depay_rtp_first_seen: now,
                                    depay_rtp_last_seen: now,
                                    depay_out: None,
                                    parser_out: None,
                                    decoder_in: None,
                                    decoder_out: None,
                                    convert_out: None,
                                    appsink_in: None,
                                },
                            ));
                        }
                        while timings.len() > 512 {
                            timings.pop_front();
                        }
                    }
                }
            }
            gst::PadProbeReturn::Ok
        })
        .context("installing RTP timestamp probe")?;

    let output_timings = Arc::clone(&frame_timings);
    source_pad
        .add_probe(gst::PadProbeType::BUFFER, move |_pad, info| {
            let Some(pts) = info.buffer().and_then(|buffer| buffer.pts()) else {
                return gst::PadProbeReturn::Ok;
            };
            let timing = {
                let now = Instant::now();
                let mut timings = output_timings.lock().expect("timestamp probe mutex");
                timings
                    .iter_mut()
                    .rev()
                    .find(|(input_pts, _)| *input_pts == pts.nseconds())
                    .map(|(_, timing)| {
                        timing.depay_out = Some(now);
                        *timing
                    })
            };
            let Some(timing) = timing else {
                return gst::PadProbeReturn::Ok;
            };
            let Some(timestamp_ns) = timing.source_ns else {
                return gst::PadProbeReturn::Ok;
            };
            let already_has_reference = info
                .buffer()
                .and_then(|buffer| buffer.meta::<gst::ReferenceTimestampMeta>())
                .is_some();
            if !already_has_reference {
                let Ok(caps) = gst::Caps::from_str("timestamp/x-ntp") else {
                    return gst::PadProbeReturn::Ok;
                };
                let Some(buffer) = info.buffer_mut() else {
                    return gst::PadProbeReturn::Ok;
                };
                let buffer = buffer.make_mut();
                gst::ReferenceTimestampMeta::add(
                    buffer,
                    &caps,
                    gst::ClockTime::from_nseconds(timestamp_ns as u64),
                    None,
                );
            }
            gst::PadProbeReturn::Ok
        })
        .context("installing H264 timestamp probe")?;

    install_frame_stage_probe(pipeline, "parser", "src", &frame_timings, |timing, now| {
        timing.parser_out = Some(now)
    })?;
    install_frame_stage_probe(
        pipeline,
        "decoder",
        "sink",
        &frame_timings,
        |timing, now| timing.decoder_in = Some(now),
    )?;
    install_frame_stage_probe(pipeline, "decoder", "src", &frame_timings, |timing, now| {
        timing.decoder_out = Some(now)
    })?;
    install_frame_stage_probe(
        pipeline,
        "colorspace",
        "src",
        &frame_timings,
        |timing, now| timing.convert_out = Some(now),
    )?;
    install_frame_stage_probe(pipeline, "sink", "sink", &frame_timings, |timing, now| {
        timing.appsink_in = Some(now)
    })?;
    Ok((frame_timings, rtcp_reports_seen))
}

/// Minimal timestamp bridge for `monitor-poe`.
///
/// The full capture path records packet-span and every decode-stage boundary.
/// Monitoring needs only one RTP timestamp per access unit plus RTCP sender
/// reports, so it avoids the RTP-arrival history, mutable output metadata, and
/// per-stage probes. H.264 fragments in one frame share a PTS; checking the
/// newest PTS before mapping the packet means only the first fragment is
/// inspected.
fn install_monitor_timestamp_probes(
    pipeline: &gst::Pipeline,
) -> Result<(FrameTimingHistory, Arc<AtomicU64>)> {
    let source = pipeline
        .by_name("source")
        .context("pipeline has no named RTSP source")?;
    let depay = pipeline
        .by_name("depay")
        .context("pipeline has no named H264 depayloader")?;
    let sink_pad = depay
        .static_pad("sink")
        .context("H264 depayloader has no sink pad")?;
    let frame_timings = Arc::new(Mutex::new(VecDeque::<(u64, FrameTiming)>::new()));
    let clock_mapper = Arc::new(Mutex::new(
        RtpClockMapper::new(64).map_err(anyhow::Error::msg)?,
    ));
    let rtcp_reports_seen = Arc::new(AtomicU64::new(0));

    let manager_mapper = Arc::clone(&clock_mapper);
    let manager_report_count = Arc::clone(&rtcp_reports_seen);
    source.connect("new-manager", false, move |args| {
        let Some(manager) = args
            .get(1)
            .and_then(|value| value.get::<gst::Element>().ok())
        else {
            return None;
        };
        let pad_mapper = Arc::clone(&manager_mapper);
        let pad_report_count = Arc::clone(&manager_report_count);
        for pad in manager.pads() {
            install_rtcp_probe(&pad, &pad_mapper, &pad_report_count);
        }
        manager.connect("pad-added", false, move |pad_args| {
            let Some(pad) = pad_args
                .get(1)
                .and_then(|value| value.get::<gst::Pad>().ok())
            else {
                return None;
            };
            install_rtcp_probe(&pad, &pad_mapper, &pad_report_count);
            None
        });
        None
    });

    let input_timings = Arc::clone(&frame_timings);
    let input_mapper = Arc::clone(&clock_mapper);
    sink_pad
        .add_probe(gst::PadProbeType::BUFFER, move |_pad, info| {
            let Some(buffer) = info.buffer() else {
                return gst::PadProbeReturn::Ok;
            };
            let Some(pts) = buffer.pts().map(|value| value.nseconds()) else {
                return gst::PadProbeReturn::Ok;
            };
            if input_timings
                .lock()
                .ok()
                .and_then(|history| history.back().map(|(candidate, _)| *candidate == pts))
                .unwrap_or(false)
            {
                return gst::PadProbeReturn::Ok;
            }
            let Some(map) = buffer.map_readable().ok() else {
                return gst::PadProbeReturn::Ok;
            };
            let Some((ssrc, rtp_timestamp)) = parse_rtp_header(map.as_slice()) else {
                return gst::PadProbeReturn::Ok;
            };
            let source_ns = buffer
                .meta::<gst::ReferenceTimestampMeta>()
                .map(|reference| reference.timestamp().nseconds() as i128)
                .or_else(|| {
                    input_mapper
                        .lock()
                        .ok()
                        .and_then(|mapper| mapper.estimate(ssrc, rtp_timestamp))
                        .map(|estimate| estimate.ntp_unix_ns)
                });
            let now = Instant::now();
            let mut timings = input_timings.lock().expect("timestamp probe mutex");
            timings.push_back((
                pts,
                FrameTiming {
                    rtp_timestamp,
                    source_ns,
                    rtsp_rtp_first_seen: None,
                    rtsp_rtp_last_seen: None,
                    depay_rtp_first_seen: now,
                    depay_rtp_last_seen: now,
                    depay_out: None,
                    parser_out: None,
                    decoder_in: None,
                    decoder_out: None,
                    convert_out: None,
                    appsink_in: None,
                },
            ));
            while timings.len() > 128 {
                timings.pop_front();
            }
            gst::PadProbeReturn::Ok
        })
        .context("installing monitor RTP timestamp probe")?;

    Ok((frame_timings, rtcp_reports_seen))
}

fn install_frame_stage_probe(
    pipeline: &gst::Pipeline,
    element_name: &str,
    pad_name: &str,
    timings: &FrameTimingHistory,
    update: fn(&mut FrameTiming, Instant),
) -> Result<()> {
    let Some(element) = pipeline.by_name(element_name) else {
        return Ok(());
    };
    let pad = element
        .static_pad(pad_name)
        .with_context(|| format!("{element_name} has no {pad_name} pad"))?;
    let timings = Arc::clone(timings);
    pad.add_probe(gst::PadProbeType::BUFFER, move |_pad, info| {
        let Some(pts) = info.buffer().and_then(|buffer| buffer.pts()) else {
            return gst::PadProbeReturn::Ok;
        };
        let now = Instant::now();
        if let Ok(mut history) = timings.lock()
            && let Some((_, timing)) = history
                .iter_mut()
                .rev()
                .find(|(input_pts, _)| *input_pts == pts.nseconds())
        {
            update(timing, now);
        }
        gst::PadProbeReturn::Ok
    })
    .with_context(|| format!("installing {element_name}:{pad_name} timing probe"))?;
    Ok(())
}

fn add_pipeline_stage_attributes(
    attributes: &mut BTreeMap<String, String>,
    timing: &FrameTiming,
    sample_pulled_at: Instant,
) {
    add_elapsed_attribute(
        attributes,
        "pipeline_depay_rtp_packet_span_ns",
        timing.depay_rtp_first_seen,
        timing.depay_rtp_last_seen,
    );
    if let Some(depay_out) = timing.depay_out {
        add_elapsed_attribute(
            attributes,
            "pipeline_depay_rtp_last_to_au_ns",
            timing.depay_rtp_last_seen,
            depay_out,
        );
        if let Some(parser_out) = timing.parser_out {
            add_elapsed_attribute(attributes, "pipeline_depay_parse_ns", depay_out, parser_out);
        }
    }
    if let (Some(parser_out), Some(decoder_in)) = (timing.parser_out, timing.decoder_in) {
        add_elapsed_attribute(
            attributes,
            "pipeline_parser_to_decoder_ns",
            parser_out,
            decoder_in,
        );
    }
    if let (Some(decoder_in), Some(decoder_out)) = (timing.decoder_in, timing.decoder_out) {
        add_elapsed_attribute(attributes, "pipeline_decode_ns", decoder_in, decoder_out);
    }
    if let (Some(decoder_out), Some(convert_out)) = (timing.decoder_out, timing.convert_out) {
        add_elapsed_attribute(attributes, "pipeline_convert_ns", decoder_out, convert_out);
    }
    if let (Some(convert_out), Some(appsink_in)) = (timing.convert_out, timing.appsink_in) {
        add_elapsed_attribute(
            attributes,
            "pipeline_convert_to_appsink_ns",
            convert_out,
            appsink_in,
        );
    }
    if let Some(appsink_in) = timing.appsink_in {
        add_elapsed_attribute(
            attributes,
            "pipeline_appsink_queue_ns",
            appsink_in,
            sample_pulled_at,
        );
    }
    add_elapsed_attribute(
        attributes,
        "pipeline_depay_rtp_first_to_pull_ns",
        timing.depay_rtp_first_seen,
        sample_pulled_at,
    );
    add_elapsed_attribute(
        attributes,
        "pipeline_depay_rtp_last_to_pull_ns",
        timing.depay_rtp_last_seen,
        sample_pulled_at,
    );
    if let (Some(first_seen), Some(last_seen)) =
        (timing.rtsp_rtp_first_seen, timing.rtsp_rtp_last_seen)
    {
        add_elapsed_attribute(
            attributes,
            "pipeline_rtsp_rtp_packet_span_ns",
            first_seen,
            last_seen,
        );
        add_elapsed_attribute(
            attributes,
            "pipeline_rtsp_to_depay_rtp_first_ns",
            first_seen,
            timing.depay_rtp_first_seen,
        );
        add_elapsed_attribute(
            attributes,
            "pipeline_rtsp_to_depay_rtp_last_ns",
            last_seen,
            timing.depay_rtp_last_seen,
        );
        add_elapsed_attribute(
            attributes,
            "pipeline_rtsp_rtp_first_to_pull_ns",
            first_seen,
            sample_pulled_at,
        );
        add_elapsed_attribute(
            attributes,
            "pipeline_rtsp_rtp_last_to_pull_ns",
            last_seen,
            sample_pulled_at,
        );
    }
}

fn add_elapsed_attribute(
    attributes: &mut BTreeMap<String, String>,
    name: &str,
    start: Instant,
    end: Instant,
) {
    if let Some(elapsed) = end.checked_duration_since(start) {
        attributes.insert(name.to_string(), elapsed.as_nanos().to_string());
    }
}

fn install_rtcp_probe(
    pad: &gst::Pad,
    mapper: &Arc<Mutex<RtpClockMapper>>,
    report_count: &Arc<AtomicU64>,
) {
    if !pad.name().contains("recv_rtcp_sink") {
        return;
    }
    let mapper = Arc::clone(mapper);
    let report_count = Arc::clone(report_count);
    let _ = pad.add_probe(gst::PadProbeType::BUFFER, move |_pad, info| {
        if let Some(buffer) = info.buffer() {
            if let Ok(map) = buffer.map_readable() {
                let reports = parse_sender_reports(map.as_slice());
                if let Ok(mut mapper) = mapper.lock() {
                    for report in reports {
                        if mapper.observe(report) {
                            report_count.fetch_add(1, Ordering::Relaxed);
                        }
                    }
                }
            }
        }
        gst::PadProbeReturn::Ok
    });
}

fn install_rtp_arrival_probe(pad: &gst::Pad, arrivals: &RtpArrivalHistory) {
    if !pad.name().contains("recv_rtp_sink") {
        return;
    }
    let arrivals = Arc::clone(arrivals);
    let _ = pad.add_probe(gst::PadProbeType::BUFFER, move |_pad, info| {
        let Some(buffer) = info.buffer() else {
            return gst::PadProbeReturn::Ok;
        };
        let Ok(map) = buffer.map_readable() else {
            return gst::PadProbeReturn::Ok;
        };
        let Some((_ssrc, rtp_timestamp)) = parse_rtp_header(map.as_slice()) else {
            return gst::PadProbeReturn::Ok;
        };
        let now = Instant::now();
        if let Ok(mut history) = arrivals.lock() {
            if let Some((_, timing)) = history
                .iter_mut()
                .rev()
                .find(|(candidate, _)| *candidate == rtp_timestamp)
            {
                timing.last_seen = now;
            } else {
                history.push_back((
                    rtp_timestamp,
                    RtpArrivalTiming {
                        first_seen: now,
                        last_seen: now,
                    },
                ));
            }
            while history.len() > 512 {
                history.pop_front();
            }
        }
        gst::PadProbeReturn::Ok
    });
}

impl Drop for PoeRtspCapture {
    fn drop(&mut self) {
        let _ = self.stop();
    }
}

pub fn rtsp_uri(camera: &PoeCameraConfig, stream: PoeStream, password: &str) -> Result<String> {
    let mut url = Url::parse(&format!(
        "rtsp://{}:{}/cam/realmonitor",
        camera.address, camera.rtsp_port
    ))?;
    url.set_username(&camera.username)
        .map_err(|_| anyhow!("invalid RTSP username"))?;
    url.set_password(Some(password))
        .map_err(|_| anyhow!("invalid RTSP password"))?;
    url.query_pairs_mut()
        .append_pair("channel", "1")
        .append_pair("subtype", stream.subtype());
    Ok(url.to_string())
}

fn gst_quote(value: &str) -> String {
    value.replace('\\', "\\\\").replace('"', "\\\"")
}

fn env_flag_enabled(value: &str) -> bool {
    !matches!(
        value.trim().to_ascii_lowercase().as_str(),
        "" | "0" | "false" | "no" | "off"
    )
}

fn effective_decoded_pixel_format() -> Result<PixelFormat> {
    parse_decoded_pixel_format(
        std::env::var("TATBOT_VISIOND_DECODE_FORMAT")
            .ok()
            .as_deref(),
    )
}

fn parse_decoded_pixel_format(value: Option<&str>) -> Result<PixelFormat> {
    match value.unwrap_or("bgr8").trim().to_ascii_lowercase().as_str() {
        "bgr" | "bgr8" => Ok(PixelFormat::Bgr8),
        "gray" | "gray8" | "grey" | "grey8" | "y8" => Ok(PixelFormat::Y8),
        value => anyhow::bail!("TATBOT_VISIOND_DECODE_FORMAT must be bgr8 or y8, got {value:?}"),
    }
}

fn active_profile(
    caps: &gst::CapsRef,
    configured: &StreamProfile,
) -> Result<(StreamProfile, bool)> {
    let structure = caps
        .structure(0)
        .context("GStreamer caps have no structure")?;
    let width = structure.get::<i32>("width").ok().map(|value| value as u32);
    let height = structure
        .get::<i32>("height")
        .ok()
        .map(|value| value as u32);
    // Some elements report an explicit 0/1 framerate for RTSP sources (the
    // rate is genuinely unknown); treat that like a missing field instead of
    // producing an invalid 0 Hz profile.
    let framerate = structure
        .get::<gst::Fraction>("framerate")
        .ok()
        .map(|fraction| (fraction.numer() as u32, fraction.denom() as u32))
        .filter(|(numer, denom)| *numer > 0 && *denom > 0);
    let complete = width.is_some() && height.is_some() && framerate.is_some();
    Ok((
        StreamProfile {
            stream: configured.stream.clone(),
            width: width.unwrap_or(configured.width),
            height: height.unwrap_or(configured.height),
            fps_num: framerate.map(|value| value.0).unwrap_or(configured.fps_num),
            fps_den: framerate.map(|value| value.1).unwrap_or(configured.fps_den),
            format: configured.format,
        },
        complete,
    ))
}

fn add_clock_assessment(attributes: &mut BTreeMap<String, String>, assessment: &SyncAssessment) {
    if let Some(offset) = assessment.median_offset_ns {
        attributes.insert("clock_offset_ns".to_string(), offset.to_string());
    }
    if let Some(mad) = assessment.median_absolute_deviation_ns {
        attributes.insert("clock_offset_mad_ns".to_string(), mad.to_string());
    }
    if let Some(min) = assessment.min_offset_ns {
        attributes.insert("clock_offset_min_ns".to_string(), min.to_string());
    }
    if let Some(max) = assessment.max_offset_ns {
        attributes.insert("clock_offset_max_ns".to_string(), max.to_string());
    }
}

fn pipeline_capture_time(host_unix_ns: i128, running_ns: u64, pts_ns: u64) -> (i128, i128) {
    let age_ns = i128::from(running_ns) - i128::from(pts_ns);
    (age_ns, host_unix_ns.saturating_sub(age_ns))
}

fn trusted_camera_capture_delta_is_plausible(host_delta_ns: i128) -> bool {
    host_delta_ns <= 250_000_000 && host_delta_ns >= -2_000_000_000
}

fn effective_jitter_latency_ms(configured: u32) -> Result<u32> {
    parse_jitter_latency_ms(
        std::env::var("TATBOT_VISIOND_RTSP_LATENCY_MS")
            .ok()
            .as_deref(),
        configured,
    )
}

fn effective_converter_threads() -> Result<u32> {
    parse_converter_threads(
        std::env::var("TATBOT_VISIOND_CONVERT_THREADS")
            .ok()
            .as_deref(),
    )
}

fn parse_converter_threads(override_value: Option<&str>) -> Result<u32> {
    let Some(value) = override_value else {
        return Ok(1);
    };
    let parsed = value.parse::<u32>().with_context(|| {
        format!("TATBOT_VISIOND_CONVERT_THREADS must be an integer from 1 to 64, got {value:?}")
    })?;
    if !(1..=64).contains(&parsed) {
        anyhow::bail!(
            "TATBOT_VISIOND_CONVERT_THREADS must be an integer from 1 to 64, got {value:?}"
        );
    }
    Ok(parsed)
}

fn parse_jitter_latency_ms(override_value: Option<&str>, configured: u32) -> Result<u32> {
    let Some(value) = override_value else {
        return Ok(configured);
    };
    value.parse::<u32>().with_context(|| {
        format!("TATBOT_VISIOND_RTSP_LATENCY_MS must be a non-negative integer, got {value:?}")
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::PoeCameraConfig;
    use std::net::IpAddr;

    fn camera() -> PoeCameraConfig {
        PoeCameraConfig {
            name: "camera1".into(),
            address: "192.0.2.91".parse::<IpAddr>().unwrap(),
            rtsp_port: 554,
            http_port: 80,
            username: "admin".into(),
            password_env: "PASSWORD".into(),
            main: StreamProfile {
                stream: "main".into(),
                width: 1920,
                height: 1080,
                fps_num: 15,
                fps_den: 1,
                format: PixelFormat::H264,
            },
            sub: None,
            transport: "tcp".into(),
            gstreamer_latency_ms: 200,
        }
    }

    #[test]
    fn builds_authenticated_uri_without_logging_the_password() {
        let uri = rtsp_uri(&camera(), PoeStream::Main, "a secret").unwrap();
        assert!(uri.starts_with("rtsp://admin:a%20secret@192.0.2.91:554"));
        assert!(uri.contains("subtype=0"));
    }

    #[test]
    fn quotes_pipeline_values() {
        assert_eq!(gst_quote("a\\b\"c"), "a\\\\b\\\"c");
    }

    #[test]
    fn parses_explicitly_disabled_environment_flags() {
        for value in ["", "0", "false", "FALSE", "no", "off"] {
            assert!(
                !env_flag_enabled(value),
                "{value:?} should disable the flag"
            );
        }
        for value in ["1", "true", "yes", "on"] {
            assert!(env_flag_enabled(value), "{value:?} should enable the flag");
        }
    }

    #[test]
    fn parses_bounded_converter_thread_override() {
        assert_eq!(parse_converter_threads(None).unwrap(), 1);
        assert_eq!(parse_converter_threads(Some("2")).unwrap(), 2);
        assert!(parse_converter_threads(Some("0")).is_err());
        assert!(parse_converter_threads(Some("65")).is_err());
        assert!(parse_converter_threads(Some("many")).is_err());
    }

    #[test]
    fn parses_tracker_grayscale_without_changing_the_general_default() {
        assert_eq!(parse_decoded_pixel_format(None).unwrap(), PixelFormat::Bgr8);
        assert_eq!(
            parse_decoded_pixel_format(Some("y8")).unwrap(),
            PixelFormat::Y8
        );
        assert_eq!(
            parse_decoded_pixel_format(Some("GRAY8")).unwrap(),
            PixelFormat::Y8
        );
        assert!(parse_decoded_pixel_format(Some("nv12")).is_err());
    }

    #[test]
    fn parses_rtsp_latency_override_without_changing_the_config_default() {
        assert_eq!(parse_jitter_latency_ms(None, 200).unwrap(), 200);
        assert_eq!(parse_jitter_latency_ms(Some("50"), 200).unwrap(), 50);
        assert!(parse_jitter_latency_ms(Some("fast"), 200).is_err());
    }

    #[test]
    fn pipeline_pts_preserves_capture_age_when_mapped_to_host_time() {
        let (age_ns, capture_unix_ns) =
            pipeline_capture_time(1_700_000_000_500_000_000, 900_000_000, 550_000_000);
        assert_eq!(age_ns, 350_000_000);
        assert_eq!(capture_unix_ns, 1_700_000_000_150_000_000);
    }

    #[test]
    fn trusted_camera_time_allows_delivery_age_but_rejects_clock_skew() {
        assert!(trusted_camera_capture_delta_is_plausible(-450_000_000));
        assert!(trusted_camera_capture_delta_is_plausible(20_000_000));
        assert!(!trusted_camera_capture_delta_is_plausible(-2_000_000_001));
        assert!(!trusted_camera_capture_delta_is_plausible(250_000_001));
    }
}
