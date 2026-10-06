//! Frame post-processing, rate limiting and timing aggregation.
//!
//! Pure logic lifted out of the `tatbot-visiond` binary. It reads and writes
//! recorded frames and plain numbers, and touches no camera API, so it builds
//! and tests under every profile — including `vision-core`, which is all most
//! nodes can compile. It used to sit in `main.rs` behind `gstreamer`,
//! `realsense` and `fiducials`, where its tests ran only on a camera node.

use std::collections::{BTreeMap, VecDeque};
use std::sync::mpsc;

use anyhow::{Context, Result};
use serde::Serialize;

use crate::{PixelFormat, RecordedPayload, SynchronizedFrameSet};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct VideoCrop {
    pub x: u32,
    pub y: u32,
    pub width: u32,
    pub height: u32,
}

pub fn parse_socket_crops(values: &[String]) -> Result<BTreeMap<String, VideoCrop>> {
    let mut crops = BTreeMap::new();
    for value in values {
        let (camera, coordinates) = value.split_once('=').with_context(|| {
            format!("invalid --socket-crop {value:?}; expected CAMERA=X,Y,WIDTH,HEIGHT")
        })?;
        if camera.is_empty() {
            anyhow::bail!("invalid --socket-crop {value:?}; camera name is empty");
        }
        let parts = coordinates
            .split(',')
            .map(str::parse::<u32>)
            .collect::<std::result::Result<Vec<_>, _>>()
            .with_context(|| format!("invalid --socket-crop {value:?}; coordinates must be u32"))?;
        let [x, y, width, height] = parts.as_slice() else {
            anyhow::bail!("invalid --socket-crop {value:?}; expected four coordinates");
        };
        if *width == 0 || *height == 0 {
            anyhow::bail!("invalid --socket-crop {value:?}; width and height must be positive");
        }
        let crop = VideoCrop {
            x: *x,
            y: *y,
            width: *width,
            height: *height,
        };
        if crops.insert(camera.to_string(), crop).is_some() {
            anyhow::bail!("duplicate --socket-crop for {camera}");
        }
    }
    Ok(crops)
}

pub fn crop_video_set(
    set: &SynchronizedFrameSet,
    crops: &BTreeMap<String, VideoCrop>,
    purpose: &str,
) -> Result<SynchronizedFrameSet> {
    let mut frames = BTreeMap::new();
    for (name, frame) in &set.frames {
        let crop = crops
            .get(name)
            .with_context(|| format!("no {purpose} crop configured for {name}"))?;
        let (format, width, height, bytes) = match &frame.payload {
            RecordedPayload::Video {
                format,
                width,
                height,
                bytes,
            } if matches!(format, PixelFormat::Bgr8 | PixelFormat::Rgb8) => {
                (*format, *width, *height, bytes)
            }
            _ => anyhow::bail!("visual cropping requires decoded BGR/RGB frames"),
        };
        let x_end = crop
            .x
            .checked_add(crop.width)
            .context("socket crop x extent overflowed")?;
        let y_end = crop
            .y
            .checked_add(crop.height)
            .context("socket crop y extent overflowed")?;
        if x_end > width || y_end > height {
            anyhow::bail!(
                "{purpose} crop for {name} ({},{},{},{}) exceeds {width}x{height}",
                crop.x,
                crop.y,
                crop.width,
                crop.height
            );
        }
        let source_stride = usize::try_from(width)? * 3;
        let output_stride = usize::try_from(crop.width)? * 3;
        let expected = source_stride * usize::try_from(height)?;
        if bytes.len() != expected {
            anyhow::bail!(
                "decoded {name} frame has {} bytes; expected {expected}",
                bytes.len()
            );
        }
        let mut output = Vec::with_capacity(output_stride * usize::try_from(crop.height)?);
        let x_offset = usize::try_from(crop.x)? * 3;
        for y in crop.y..y_end {
            let start = usize::try_from(y)? * source_stride + x_offset;
            output.extend_from_slice(&bytes[start..start + output_stride]);
        }
        let mut metadata = frame.metadata.clone();
        metadata.profile.width = crop.width;
        metadata.profile.height = crop.height;
        metadata.flags.push(format!("{purpose}_cropped"));
        metadata.attributes.insert(
            format!("{purpose}_source_dimensions"),
            format!("{width}x{height}"),
        );
        metadata.attributes.insert(
            format!("{purpose}_crop_xywh"),
            format!("{},{},{},{}", crop.x, crop.y, crop.width, crop.height),
        );
        frames.insert(
            name.clone(),
            crate::FrameRecord {
                metadata,
                payload: RecordedPayload::Video {
                    format,
                    width: crop.width,
                    height: crop.height,
                    bytes: output,
                },
            },
        );
    }
    Ok(SynchronizedFrameSet {
        sequence: set.sequence,
        timestamp_basis: set.timestamp_basis.clone(),
        timestamp_ns: set.timestamp_ns,
        maximum_skew_ns: set.maximum_skew_ns,
        frames,
    })
}

/// Nearest-neighbour downscale of a packed raster whose rows are `units`
/// samples of `unit_bytes` each (a Z16 pixel is one 2-byte unit; a YUYV
/// macropixel is one 4-byte unit covering two pixels). Returns the resized
/// bytes and the new (units, rows) size; `None` if the buffer length does not
/// match the claimed geometry.
pub fn scale_packed_rows(
    bytes: &[u8],
    units: u32,
    rows: u32,
    unit_bytes: usize,
    scale: f64,
) -> Option<(Vec<u8>, u32, u32)> {
    if bytes.len() != (units as usize) * (rows as usize) * unit_bytes {
        return None;
    }
    let new_units = ((units as f64 * scale).round() as u32).max(1);
    let new_rows = ((rows as f64 * scale).round() as u32).max(1);
    let mut resized = Vec::with_capacity((new_units as usize) * (new_rows as usize) * unit_bytes);
    for y in 0..new_rows {
        let src_y = ((y as u64 * rows as u64) / new_rows as u64) as usize;
        let row = &bytes[src_y * units as usize * unit_bytes..];
        for x in 0..new_units {
            let src_x = ((x as u64 * units as u64) / new_units as u64) as usize;
            resized.extend_from_slice(&row[src_x * unit_bytes..(src_x + 1) * unit_bytes]);
        }
    }
    Some((resized, new_units, new_rows))
}

pub fn note_scaled(
    frame: &mut crate::FrameRecord,
    width: u32,
    height: u32,
    new_width: u32,
    new_height: u32,
    scale: f64,
    purpose: &str,
) {
    frame.metadata.profile.width = new_width;
    frame.metadata.profile.height = new_height;
    frame
        .metadata
        .flags
        .push(format!("{purpose}_uniformly_scaled"));
    frame.metadata.attributes.insert(
        format!("{purpose}_source_dimensions"),
        format!("{width}x{height}"),
    );
    frame
        .metadata
        .attributes
        .insert(format!("{purpose}_scale"), scale.to_string());
    frame
        .metadata
        .attributes
        .insert(format!("{purpose}_resize_filter"), "nearest".into());
}

/// Nearest-neighbour downscale of every Z16 depth plane in a set (the
/// companion of `scale_video_set`, which deliberately skips depth). Sample
/// values are untouched, so `depth_units_m` stays valid; the scaled plane is
/// a visualization derivative, never evidence.
pub fn scale_depth_set(set: &SynchronizedFrameSet, scale: f64, purpose: &str) -> SynchronizedFrameSet {
    let mut output = set.clone();
    for frame in output.frames.values_mut() {
        let RecordedPayload::Depth {
            width,
            height,
            bytes,
        } = &frame.payload
        else {
            continue;
        };
        let (width, height) = (*width, *height);
        let Some((resized, new_width, new_height)) =
            scale_packed_rows(bytes, width, height, 2, scale)
        else {
            continue;
        };
        frame.payload = RecordedPayload::Depth {
            width: new_width,
            height: new_height,
            bytes: resized,
        };
        note_scaled(frame, width, height, new_width, new_height, scale, purpose);
    }
    output
}

// The only helper here that is not self-contained: nearest-neighbour RGB
// resizing goes through `image`, an optional dependency of these features.
#[cfg(any(feature = "gstreamer", feature = "rerun"))]
pub fn scale_video_set(
    set: &SynchronizedFrameSet,
    scale: f64,
    purpose: &str,
) -> Result<SynchronizedFrameSet> {
    let mut output = set.clone();
    for frame in output.frames.values_mut() {
        let (format, width, height, bytes) = match &frame.payload {
            RecordedPayload::Video {
                format,
                width,
                height,
                bytes,
            } if matches!(format, PixelFormat::Bgr8 | PixelFormat::Rgb8) => {
                (*format, *width, *height, bytes)
            }
            // RealSense colour arrives as YUYV: two pixels share one 4-byte
            // macropixel, so scale in macropixel units and keep the width even.
            RecordedPayload::Video {
                format: PixelFormat::Yuyv,
                width,
                height,
                bytes,
            } => {
                let (width, height) = (*width, *height);
                let Some((resized, new_macro, new_height)) =
                    scale_packed_rows(bytes, width / 2, height, 4, scale)
                else {
                    anyhow::bail!("YUYV frame length does not match its dimensions");
                };
                let new_width = new_macro * 2;
                frame.payload = RecordedPayload::Video {
                    format: PixelFormat::Yuyv,
                    width: new_width,
                    height: new_height,
                    bytes: resized,
                };
                note_scaled(frame, width, height, new_width, new_height, scale, purpose);
                continue;
            }
            // A detector-only owner decodes straight to luma. Resize the
            // single plane rather than refusing the frame, so the shadow
            // comparison can run against a Y8 owner as well as a colour one.
            RecordedPayload::Video {
                format: PixelFormat::Y8,
                width,
                height,
                bytes,
            } => {
                let (width, height) = (*width, *height);
                let new_width = ((width as f64 * scale).round() as u32).max(1);
                let new_height = ((height as f64 * scale).round() as u32).max(1);
                let source = image::GrayImage::from_raw(width, height, bytes.clone())
                    .context("decoded luma frame length does not match its dimensions")?;
                let resized = image::imageops::resize(
                    &source,
                    new_width,
                    new_height,
                    image::imageops::FilterType::Nearest,
                );
                frame.payload = RecordedPayload::Video {
                    format: PixelFormat::Y8,
                    width: new_width,
                    height: new_height,
                    bytes: resized.into_raw(),
                };
                note_scaled(frame, width, height, new_width, new_height, scale, purpose);
                continue;
            }
            RecordedPayload::Depth { .. } => continue,
            _ => anyhow::bail!("visual scaling requires decoded BGR/RGB/YUYV/Y8 frames"),
        };
        let new_width = ((width as f64 * scale).round() as u32).max(1);
        let new_height = ((height as f64 * scale).round() as u32).max(1);
        let source = image::RgbImage::from_raw(width, height, bytes.clone())
            .context("decoded frame length does not match its dimensions")?;
        let resized = image::imageops::resize(
            &source,
            new_width,
            new_height,
            // The shadow path is latency-sensitive and AprilTag edges are
            // binary. Nearest-neighbour preserves those edges and avoids the
            // CPU saturation observed with five simultaneous Triangle resizes
            // on the Jetson camera node.
            image::imageops::FilterType::Nearest,
        );
        frame.payload = RecordedPayload::Video {
            format,
            width: new_width,
            height: new_height,
            bytes: resized.into_raw(),
        };
        frame.metadata.profile.width = new_width;
        frame.metadata.profile.height = new_height;
        frame
            .metadata
            .flags
            .push(format!("{purpose}_uniformly_scaled"));
        frame.metadata.attributes.insert(
            format!("{purpose}_source_dimensions"),
            format!("{width}x{height}"),
        );
        frame
            .metadata
            .attributes
            .insert(format!("{purpose}_scale"), scale.to_string());
        frame
            .metadata
            .attributes
            .insert(format!("{purpose}_resize_filter"), "nearest".into());
    }
    Ok(output)
}

pub fn decoded_frame_dimensions(frame: &crate::FrameRecord) -> Option<(usize, usize)> {
    match &frame.payload {
        RecordedPayload::Video { width, height, .. } => Some((*width as usize, *height as usize)),
        _ => None,
    }
}

pub fn fiducial_set_due(
    timestamp_ns: i128,
    last_processed_ns: Option<i128>,
    min_interval_ns: Option<i128>,
) -> bool {
    match (last_processed_ns, min_interval_ns) {
        // Camera periods are not exact integer nanoseconds. Comparing the raw
        // delta made nominal 20 Hz frames at 99.9 ms miss a 10 Hz threshold
        // and selected every third frame (~6.7 Hz). One sample per aligned
        // interval keeps the long-run cap while accepting that second frame.
        (Some(last), Some(interval)) if timestamp_ns >= last => {
            timestamp_ns.div_euclid(interval) > last.div_euclid(interval)
        }
        _ => true,
    }
}

pub fn camera_reacquisition_due(row: usize, camera_index: usize, period: usize) -> bool {
    period == 0 || (row + camera_index) % period == 0
}

pub fn decimate_replay_rows(rows: &mut Vec<(i128, usize, usize)>, source_count: usize, max_fps: f64) {
    if max_fps == 0.0 {
        return;
    }
    let interval_ns = (1e9 / max_fps).round() as i128;
    let mut last_by_source = vec![None; source_count];
    rows.retain(|(timestamp_ns, source_index, _)| {
        let due =
            last_by_source[*source_index].is_none_or(|last| *timestamp_ns - last >= interval_ns);
        if due {
            last_by_source[*source_index] = Some(*timestamp_ns);
        }
        due
    });
}

#[derive(Debug, Serialize)]
pub struct TimingSummary {
    pub samples: usize,
    pub retained_samples: usize,
    pub median: f64,
    pub p95: f64,
    pub max: f64,
}

/// At 20 Hz, 4096 values cover more than three minutes per metric while making
/// multi-hour live views constant-space.
pub const TIMING_SAMPLE_LIMIT: usize = 4096;

#[derive(Debug)]
pub struct TimingSamples {
    pub values: Vec<f64>,
    pub next: usize,
    pub samples: usize,
    pub max: f64,
    pub limit: usize,
}

impl TimingSamples {
    pub fn with_limit(limit: usize) -> Self {
        assert!(limit > 0);
        Self {
            values: Vec::new(),
            next: 0,
            samples: 0,
            max: f64::NEG_INFINITY,
            limit,
        }
    }

    pub fn push(&mut self, value: f64) {
        self.samples = self.samples.saturating_add(1);
        self.max = self.max.max(value);
        if self.values.len() < self.limit {
            self.values.push(value);
            return;
        }
        self.values[self.next] = value;
        self.next = (self.next + 1) % self.limit;
    }
}

impl Default for TimingSamples {
    fn default() -> Self {
        Self::with_limit(TIMING_SAMPLE_LIMIT)
    }
}

pub fn timing_summary(samples: &TimingSamples) -> Option<TimingSummary> {
    if samples.values.is_empty() {
        return None;
    }
    let mut sorted = samples.values.clone();
    sorted.sort_by(f64::total_cmp);
    let p95_index = ((sorted.len() - 1) as f64 * 0.95).round() as usize;
    Some(TimingSummary {
        samples: samples.samples,
        retained_samples: sorted.len(),
        median: sorted[sorted.len() / 2],
        p95: sorted[p95_index],
        max: samples.max,
    })
}

pub fn bounded_capture_event_channel<T>(capacity: usize) -> (mpsc::SyncSender<T>, mpsc::Receiver<T>) {
    mpsc::sync_channel(capacity.max(1))
}

#[derive(Debug)]
/// Retains the newest diagnostic window in chronological order without hiding
/// how many older entries were evicted.
pub struct BoundedStrings {
    pub values: VecDeque<String>,
    pub limit: usize,
    pub dropped: u64,
}

impl BoundedStrings {
    pub fn new(limit: usize) -> Self {
        assert!(limit > 0);
        Self {
            values: VecDeque::with_capacity(limit),
            limit,
            dropped: 0,
        }
    }

    pub fn push(&mut self, value: String) {
        if self.values.len() == self.limit {
            self.values.pop_front();
            self.dropped = self.dropped.saturating_add(1);
        }
        self.values.push_back(value);
    }

    pub fn into_vec(self) -> Vec<String> {
        self.values.into_iter().collect()
    }
}

#[cfg(test)]
mod socket_crop_tests {
    use super::{VideoCrop, crop_video_set, parse_socket_crops};
    use std::collections::{BTreeMap, BTreeSet};
    use crate::{
        FrameMetadata, FrameRecord, FrameTimestamps, PixelFormat, RecordedPayload, SensorKind,
        StreamProfile, SynchronizedFrameSet, TimestampDomain,
    };

    #[test]
    fn parses_and_refuses_ambiguous_crop_specs() {
        let crops =
            parse_socket_crops(&["camera1=1,2,3,4".to_string(), "camera2=0,0,5,6".to_string()])
                .unwrap();
        assert_eq!(
            crops["camera1"],
            VideoCrop {
                x: 1,
                y: 2,
                width: 3,
                height: 4
            }
        );
        assert!(parse_socket_crops(&["camera1=0,0,0,4".to_string()]).is_err());
        assert!(
            parse_socket_crops(&["camera1=0,0,1,1".to_string(), "camera1=1,1,1,1".to_string(),])
                .is_err()
        );
    }

    #[test]
    fn crops_without_cloning_full_frame_pixels() {
        let frame = FrameRecord {
            metadata: FrameMetadata {
                sensor_name: "camera1".into(),
                sensor_kind: SensorKind::PoE,
                sequence: 1,
                profile: StreamProfile {
                    stream: "main".into(),
                    width: 3,
                    height: 2,
                    fps_num: 20,
                    fps_den: 1,
                    format: PixelFormat::Bgr8,
                },
                timestamps: FrameTimestamps {
                    source_ns: Some(10),
                    source_domain: TimestampDomain::CameraNtp,
                    rtp_timestamp: Some(1),
                    pipeline_pts_ns: Some(2),
                    pipeline_dts_ns: None,
                    host_monotonic_ns: 3,
                    host_unix_ns: 4,
                    normalized_unix_ns: Some(10),
                },
                dropped_before: 0,
                calibration_id: None,
                flags: Vec::new(),
                attributes: BTreeMap::new(),
            },
            payload: RecordedPayload::Video {
                format: PixelFormat::Bgr8,
                width: 3,
                height: 2,
                bytes: (0_u8..18).collect(),
            },
        };
        let set = SynchronizedFrameSet {
            sequence: 1,
            timestamp_basis: "normalized_unix_ns".into(),
            timestamp_ns: 10,
            maximum_skew_ns: 0,
            frames: BTreeMap::from([("camera1".into(), frame)]),
        };
        let output = crop_video_set(
            &set,
            &BTreeMap::from([(
                "camera1".into(),
                VideoCrop {
                    x: 1,
                    y: 0,
                    width: 2,
                    height: 2,
                },
            )]),
            "test",
        )
        .unwrap();
        let cropped = &output.frames["camera1"];
        assert_eq!(cropped.metadata.profile.width, 2);
        assert_eq!(cropped.metadata.profile.height, 2);
        assert!(cropped.metadata.flags.contains(&"test_cropped".into()));
        let RecordedPayload::Video { bytes, .. } = &cropped.payload else {
            panic!("expected video payload");
        };
        assert_eq!(bytes, &[3, 4, 5, 6, 7, 8, 12, 13, 14, 15, 16, 17]);
        assert_eq!(
            output.frames.keys().cloned().collect::<BTreeSet<_>>(),
            BTreeSet::from(["camera1".to_string()])
        );
    }
}

#[cfg(test)]
mod fiducial_rate_tests {
    use super::{camera_reacquisition_due, fiducial_set_due};

    #[test]
    fn rate_limit_uses_capture_timestamps_and_recovers_from_regression() {
        assert!(fiducial_set_due(1_000, None, Some(100)));
        assert!(!fiducial_set_due(1_099, Some(1_000), Some(100)));
        assert!(fiducial_set_due(1_100, Some(1_000), Some(100)));
        assert!(fiducial_set_due(900, Some(1_000), Some(100)));
        assert!(fiducial_set_due(1_001, Some(1_000), None));
    }

    #[test]
    fn aligned_intervals_accept_nominal_frames_just_below_raw_delta() {
        assert!(fiducial_set_due(
            1_149_900_000,
            Some(1_050_000_000),
            Some(100_000_000)
        ));
    }

    #[test]
    fn absent_camera_reacquisition_is_staggered_and_zero_disables_backoff() {
        assert!(camera_reacquisition_due(10, 0, 5));
        assert!(!camera_reacquisition_due(10, 1, 5));
        assert!(camera_reacquisition_due(14, 1, 5));
        assert!(camera_reacquisition_due(11, 3, 0));
    }
}

#[cfg(test)]
mod replay_tests {
    use super::decimate_replay_rows;

    #[test]
    fn decimation_is_independent_per_recording_source() {
        let mut rows = vec![
            (0, 0, 0),
            (10_000_000, 1, 0),
            (40_000_000, 0, 1),
            (60_000_000, 1, 1),
            (110_000_000, 0, 2),
            (120_000_000, 1, 2),
        ];
        decimate_replay_rows(&mut rows, 2, 10.0);
        assert_eq!(
            rows,
            vec![
                (0, 0, 0),
                (10_000_000, 1, 0),
                (110_000_000, 0, 2),
                (120_000_000, 1, 2)
            ]
        );
    }
}

#[cfg(test)]
mod retention_tests {
    use super::{BoundedStrings, bounded_capture_event_channel};
    #[cfg(feature = "gstreamer")]
    use super::{TimingSamples, timing_summary};
    use std::sync::mpsc::TrySendError;

    #[test]
    fn ingress_channel_backpressures_at_its_capacity() {
        let (sender, receiver) = bounded_capture_event_channel(2);
        sender.try_send(1).unwrap();
        sender.try_send(2).unwrap();
        assert_eq!(sender.try_send(3), Err(TrySendError::Full(3)));

        assert_eq!(receiver.recv().unwrap(), 1);
        sender.try_send(3).unwrap();
    }

    #[test]
    fn ingress_channel_never_becomes_unbounded_for_zero_capacity() {
        let (sender, _receiver) = bounded_capture_event_channel(0);
        sender.try_send(1).unwrap();
        assert_eq!(sender.try_send(2), Err(TrySendError::Full(2)));
    }

    #[test]
    fn diagnostic_strings_keep_only_the_newest_entries() {
        let mut strings = BoundedStrings::new(2);
        strings.push("first".into());
        strings.push("second".into());
        strings.push("third".into());
        assert_eq!(strings.dropped, 1);
        assert_eq!(strings.into_vec(), vec!["second", "third"]);
    }

    #[cfg(feature = "gstreamer")]
    #[test]
    fn reports_order_independent_percentiles() {
        let mut samples = TimingSamples::default();
        for value in [4.0, 1.0, 3.0, 2.0] {
            samples.push(value);
        }
        let summary = timing_summary(&samples).unwrap();
        assert_eq!(summary.samples, 4);
        assert_eq!(summary.retained_samples, 4);
        assert_eq!(summary.median, 3.0);
        assert_eq!(summary.p95, 4.0);
        assert_eq!(summary.max, 4.0);
    }

    #[cfg(feature = "gstreamer")]
    #[test]
    fn bounds_retention_without_losing_total_count_or_global_max() {
        let mut samples = TimingSamples::with_limit(3);
        for value in [100.0, 1.0, 2.0, 3.0] {
            samples.push(value);
        }
        let summary = timing_summary(&samples).unwrap();
        assert_eq!(summary.samples, 4);
        assert_eq!(summary.retained_samples, 3);
        assert_eq!(summary.median, 2.0);
        assert_eq!(summary.p95, 3.0);
        assert_eq!(summary.max, 100.0);
    }
}

#[cfg(test)]
#[cfg(any(feature = "gstreamer", feature = "rerun"))]
mod luma_scale_tests {
    use super::scale_video_set;
    use crate::{
        FrameMetadata, FrameRecord, FrameTimestamps, PixelFormat, RecordedPayload, SensorKind,
        StreamProfile, SynchronizedFrameSet, TimestampDomain,
    };
    use std::collections::BTreeMap;

    fn luma_set() -> SynchronizedFrameSet {
        let frame = FrameRecord {
            metadata: FrameMetadata {
                sensor_name: "camera1".into(),
                sensor_kind: SensorKind::PoE,
                sequence: 1,
                profile: StreamProfile {
                    stream: "main".into(),
                    width: 4,
                    height: 4,
                    fps_num: 20,
                    fps_den: 1,
                    format: PixelFormat::Y8,
                },
                timestamps: FrameTimestamps {
                    source_ns: Some(10),
                    source_domain: TimestampDomain::CameraNtp,
                    rtp_timestamp: None,
                    pipeline_pts_ns: None,
                    pipeline_dts_ns: None,
                    host_monotonic_ns: 1,
                    host_unix_ns: 2,
                    normalized_unix_ns: Some(2),
                },
                dropped_before: 0,
                calibration_id: None,
                flags: Vec::new(),
                attributes: Default::default(),
            },
            payload: RecordedPayload::Video {
                format: PixelFormat::Y8,
                width: 4,
                height: 4,
                bytes: (0u8..16).collect(),
            },
        };
        SynchronizedFrameSet {
            sequence: 1,
            timestamp_basis: "normalized".into(),
            timestamp_ns: 2,
            maximum_skew_ns: 0,
            frames: BTreeMap::from([("camera1".to_string(), frame)]),
        }
    }

    #[test]
    fn luma_frames_scale_as_a_single_plane() {
        let set = luma_set();
        let scaled = scale_video_set(&set, 0.5, "socket").unwrap();
        let out = &scaled.frames["camera1"];
        let RecordedPayload::Video {
            format,
            width,
            height,
            bytes,
        } = &out.payload
        else {
            panic!("scaling must keep a decoded video payload");
        };
        assert_eq!((*format, *width, *height), (PixelFormat::Y8, 2, 2));
        // One byte per pixel, not three: the luma plane is scaled in place
        // rather than being expanded to colour first.
        assert_eq!(bytes.len(), 4);
        assert_eq!(out.metadata.profile.width, 2);
        assert_eq!(out.metadata.profile.height, 2);
        assert!(
            out.metadata
                .flags
                .iter()
                .any(|flag| flag == "socket_uniformly_scaled")
        );
    }

    #[test]
    fn luma_frames_with_an_impossible_byte_count_are_refused() {
        let mut set = luma_set();
        if let RecordedPayload::Video { bytes, .. } =
            &mut set.frames.get_mut("camera1").unwrap().payload
        {
            bytes.pop();
        }
        assert!(scale_video_set(&set, 0.5, "socket").is_err());
    }
}
