//! Frame post-processing and rate limiting.
//!
//! Pure logic lifted out of the `tatbot-visiond` binary. It reads and writes
//! recorded frames and plain numbers, and touches no camera API, so it builds
//! and tests under every profile — including `vision-core`, which is all most
//! nodes can compile.

use std::collections::VecDeque;
use std::sync::mpsc;

// Only the `image`-backed visual scaling below returns errors or matches
// pixel formats; it is compiled with the viewer.
#[cfg(feature = "rerun")]
use anyhow::{Context, Result};

#[cfg(feature = "rerun")]
use crate::PixelFormat;
use crate::{RecordedPayload, SynchronizedFrameSet};

/// Packed YUYV (two pixels per Y0 U Y1 V macropixel) to RGB8, BT.601: the one
/// conversion the bus encoder, the viewer's JPEG path and the depth cloud share.
pub fn yuyv_to_rgb(bytes: &[u8]) -> Vec<u8> {
    let mut rgb = Vec::with_capacity(bytes.len() / 2 * 3);
    for pair in bytes.chunks_exact(4) {
        let u = pair[1] as f32 - 128.0;
        let v = pair[3] as f32 - 128.0;
        for y in [pair[0] as f32, pair[2] as f32] {
            rgb.push((y + 1.402 * v).clamp(0.0, 255.0) as u8);
            rgb.push((y - 0.344 * u - 0.714 * v).clamp(0.0, 255.0) as u8);
            rgb.push((y + 1.772 * u).clamp(0.0, 255.0) as u8);
        }
    }
    rgb
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
    with_frames(set, |frame| {
        let RecordedPayload::Depth {
            width,
            height,
            bytes,
        } = &frame.payload
        else {
            return frame.clone();
        };
        let (width, height) = (*width, *height);
        let Some((resized, new_width, new_height)) =
            scale_packed_rows(bytes, width, height, 2, scale)
        else {
            return frame.clone();
        };
        let mut out = crate::FrameRecord {
            metadata: frame.metadata.clone(),
            payload: RecordedPayload::Depth {
                width: new_width,
                height: new_height,
                bytes: resized,
            },
        };
        note_scaled(&mut out, width, height, new_width, new_height, scale, purpose);
        out
    })
}

/// `set` with each frame replaced by `frame(original)`: only what `frame`
/// builds is allocated, never a copy of the whole set first.
fn with_frames(
    set: &SynchronizedFrameSet,
    mut frame: impl FnMut(&crate::FrameRecord) -> crate::FrameRecord,
) -> SynchronizedFrameSet {
    SynchronizedFrameSet {
        sequence: set.sequence,
        timestamp_basis: set.timestamp_basis.clone(),
        timestamp_ns: set.timestamp_ns,
        maximum_skew_ns: set.maximum_skew_ns,
        frames: set.frames.iter().map(|(name, f)| (name.clone(), frame(f))).collect(),
    }
}

/// A Z16 depth plane decimated to fit `max_width` x `max_height` by an
/// integer factor, each output sample the nearest valid range in its block.
/// Averaging across a depth edge would invent surfaces between the near and
/// the far side; the minimum keeps the nearer one, and a block with no valid
/// sample stays zero (invalid). `None` when the frame is not depth or already
/// fits. The plane stays in sensor units, so `depth_units_m` holds, and a
/// Pinhole scaled by payload over full size still projects it.
pub fn decimate_depth_valid_min(
    frame: &crate::FrameRecord,
    max_width: u32,
    max_height: u32,
    purpose: &str,
) -> Option<crate::FrameRecord> {
    let RecordedPayload::Depth {
        width,
        height,
        bytes,
    } = &frame.payload
    else {
        return None;
    };
    let (width, height) = (*width, *height);
    let factor = width
        .div_ceil(max_width.max(1))
        .max(height.div_ceil(max_height.max(1)));
    if factor <= 1 || bytes.len() != width as usize * height as usize * 2 {
        return None;
    }
    let (new_width, new_height) = (width.div_ceil(factor), height.div_ceil(factor));
    let sample = |x: u32, y: u32| {
        let at = (y as usize * width as usize + x as usize) * 2;
        u16::from_le_bytes([bytes[at], bytes[at + 1]])
    };
    let mut decimated = Vec::with_capacity(new_width as usize * new_height as usize * 2);
    for row in 0..new_height {
        for column in 0..new_width {
            let nearest = (row * factor..((row + 1) * factor).min(height))
                .flat_map(|y| (column * factor..((column + 1) * factor).min(width)).map(move |x| (x, y)))
                .map(|(x, y)| sample(x, y))
                .filter(|&range| range != 0)
                .min()
                .unwrap_or(0);
            decimated.extend_from_slice(&nearest.to_le_bytes());
        }
    }
    let mut output = frame.clone();
    output.payload = RecordedPayload::Depth {
        width: new_width,
        height: new_height,
        bytes: decimated,
    };
    note_scaled(
        &mut output,
        width,
        height,
        new_width,
        new_height,
        1.0 / f64::from(factor),
        purpose,
    );
    output
        .metadata
        .attributes
        .insert(format!("{purpose}_resize_filter"), "valid_min".into());
    Some(output)
}

/// Zero (invalid) every sample of a Z16 depth plane beyond `max_m` metres,
/// read in the frame's own `depth_units_m`, so a display cloud keeps the
/// working volume and drops the room behind it. A plane without metric units
/// is left as it is: its scale would be a guess.
pub fn clip_depth_beyond(frame: &mut crate::FrameRecord, max_m: f64, purpose: &str) {
    let Some(units) = frame
        .metadata
        .attributes
        .get("depth_units_m")
        .and_then(|units| units.parse::<f64>().ok())
        .filter(|units| units.is_finite() && *units > 0.0)
    else {
        return;
    };
    let RecordedPayload::Depth { bytes, .. } = &mut frame.payload else {
        return;
    };
    let limit = (max_m / units).round().min(f64::from(u16::MAX)) as u16;
    for sample in bytes.chunks_exact_mut(2) {
        if u16::from_le_bytes([sample[0], sample[1]]) > limit {
            sample.fill(0);
        }
    }
    frame
        .metadata
        .attributes
        .insert(format!("{purpose}_max_range_m"), max_m.to_string());
}

/// A Z16 plane (little-endian samples) as ROS `compressedDepth` RVL, the
/// payload Rerun's `EncodedDepthImage` decodes: a 12-byte config header
/// (format 0, no inverse-depth quantisation, so the samples stay in sensor
/// units), width and height, then RVL (Wilson 2017): each run of zeros and of
/// valid samples as a length, each valid sample as the zigzag of its
/// difference from the previous one, all in 3-bit groups packed eight to a
/// little-endian word. Lossless; a smooth surface costs about a byte a sample.
pub fn rvl_depth(width: u32, height: u32, samples: &[u8]) -> Vec<u8> {
    struct Nibbles {
        out: Vec<u8>,
        word: u32,
        count: u32,
    }
    impl Nibbles {
        fn push(&mut self, nibble: u32) {
            self.word = (self.word << 4) | nibble;
            self.count += 1;
            if self.count == 8 {
                self.out.extend_from_slice(&self.word.to_le_bytes());
                (self.word, self.count) = (0, 0);
            }
        }
        fn value(&mut self, mut value: u32) {
            loop {
                let low = value & 0x7;
                value >>= 3;
                self.push(if value == 0 { low } else { low | 0x8 });
                if value == 0 {
                    return;
                }
            }
        }
    }
    let samples: Vec<u16> = samples
        .chunks_exact(2)
        .map(|b| u16::from_le_bytes([b[0], b[1]]))
        .collect();
    let mut nibbles = Nibbles {
        out: Vec::with_capacity(20 + samples.len()),
        word: 0,
        count: 0,
    };
    nibbles.out.extend_from_slice(&0_i32.to_le_bytes());
    nibbles.out.extend_from_slice(&[0; 8]);
    nibbles.out.extend_from_slice(&width.to_le_bytes());
    nibbles.out.extend_from_slice(&height.to_le_bytes());
    let (mut at, mut previous) = (0, 0_i32);
    while at < samples.len() {
        let zeros = samples[at..].iter().take_while(|&&v| v == 0).count();
        at += zeros;
        let valid = samples[at..].iter().take_while(|&&v| v != 0).count();
        nibbles.value(zeros as u32);
        nibbles.value(valid as u32);
        for &sample in &samples[at..at + valid] {
            let delta = i32::from(sample) - previous;
            nibbles.value(((delta << 1) ^ (delta >> 31)) as u32);
            previous = i32::from(sample);
        }
        at += valid;
    }
    if nibbles.count > 0 {
        let word = nibbles.word << (4 * (8 - nibbles.count));
        nibbles.out.extend_from_slice(&word.to_le_bytes());
    }
    nibbles.out
}

// The only helper here that is not self-contained: nearest-neighbour RGB
// resizing goes through `image`, an optional dependency of these features.
#[cfg(feature = "rerun")]
pub fn scale_video_set(
    set: &SynchronizedFrameSet,
    scale: f64,
    purpose: &str,
) -> Result<SynchronizedFrameSet> {
    let mut failure = None;
    let output = with_frames(set, |frame| {
        scale_video_frame(frame, scale, purpose).unwrap_or_else(|error| {
            failure.get_or_insert(error);
            frame.clone()
        })
    });
    failure.map_or(Ok(output), Err)
}

#[cfg(feature = "rerun")]
fn scale_video_frame(
    frame: &crate::FrameRecord,
    scale: f64,
    purpose: &str,
) -> Result<crate::FrameRecord> {
    let (format, width, height, bytes) = match &frame.payload {
        RecordedPayload::Video {
            format,
            width,
            height,
            bytes,
        } => (*format, *width, *height, bytes.as_slice()),
        RecordedPayload::Depth { .. } => return Ok(frame.clone()),
        _ => anyhow::bail!("visual scaling requires decoded BGR/RGB/YUYV/Y8 frames"),
    };
    let (resized, new_width, new_height) = match format {
        // RealSense colour arrives as YUYV: two pixels share one 4-byte
        // macropixel, so scale in macropixel units and keep the width even.
        PixelFormat::Yuyv => {
            let Some((resized, new_macro, new_height)) =
                scale_packed_rows(bytes, width / 2, height, 4, scale)
            else {
                anyhow::bail!("YUYV frame length does not match its dimensions");
            };
            (resized, new_macro * 2, new_height)
        }
        // A recording made by a detector-only owner holds luma: resize the
        // single plane rather than refusing the frame.
        PixelFormat::Y8 => resize_nearest::<image::Luma<u8>>(bytes, width, height, scale)
            .context("decoded luma frame length does not match its dimensions")?,
        // Nearest-neighbour preserves AprilTag edges, ignores channel order
        // (BGR resizes as RGB) and avoids the CPU a Triangle filter costs.
        PixelFormat::Bgr8 | PixelFormat::Rgb8 => {
            resize_nearest::<image::Rgb<u8>>(bytes, width, height, scale)
                .context("decoded frame length does not match its dimensions")?
        }
        _ => anyhow::bail!("visual scaling requires decoded BGR/RGB/YUYV/Y8 frames"),
    };
    let mut out = crate::FrameRecord {
        metadata: frame.metadata.clone(),
        payload: RecordedPayload::Video {
            format,
            width: new_width,
            height: new_height,
            bytes: resized,
        },
    };
    note_scaled(&mut out, width, height, new_width, new_height, scale, purpose);
    Ok(out)
}

/// Nearest-neighbour resize of a borrowed plane: the source is read in place.
#[cfg(feature = "rerun")]
fn resize_nearest<P: image::Pixel<Subpixel = u8> + 'static>(
    bytes: &[u8],
    width: u32,
    height: u32,
    scale: f64,
) -> Option<(Vec<u8>, u32, u32)> {
    let new_width = ((width as f64 * scale).round() as u32).max(1);
    let new_height = ((height as f64 * scale).round() as u32).max(1);
    let source = image::ImageBuffer::<P, &[u8]>::from_raw(width, height, bytes)?;
    let resized =
        image::imageops::resize(&source, new_width, new_height, image::imageops::FilterType::Nearest);
    Some((resized.into_raw(), new_width, new_height))
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
}

#[cfg(test)]
#[cfg(feature = "rerun")]
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

#[cfg(test)]
mod depth_decimation_tests {
    use super::{clip_depth_beyond, decimate_depth_valid_min, rvl_depth};
    use crate::{
        FrameMetadata, FrameRecord, FrameTimestamps, PixelFormat, RecordedPayload, SensorKind,
        StreamProfile, TimestampDomain,
    };

    fn depth(width: u32, height: u32, samples: &[u16]) -> FrameRecord {
        FrameRecord {
            metadata: FrameMetadata {
                sensor_name: "overhead_depth_depth".into(),
                sensor_kind: SensorKind::RealSense,
                sequence: 1,
                profile: StreamProfile {
                    stream: "depth".into(),
                    width,
                    height,
                    fps_num: 30,
                    fps_den: 1,
                    format: PixelFormat::Z16,
                },
                timestamps: FrameTimestamps {
                    source_ns: None,
                    source_domain: TimestampDomain::HostMonotonic,
                    rtp_timestamp: None,
                    pipeline_pts_ns: None,
                    pipeline_dts_ns: None,
                    host_monotonic_ns: 1,
                    host_unix_ns: 1,
                    normalized_unix_ns: Some(1),
                },
                dropped_before: 0,
                calibration_id: None,
                flags: Vec::new(),
                attributes: [("depth_units_m".to_string(), "0.001".to_string())].into(),
            },
            payload: RecordedPayload::Depth {
                width,
                height,
                bytes: samples.iter().flat_map(|v| v.to_le_bytes()).collect(),
            },
        }
    }

    fn samples(frame: &FrameRecord) -> Vec<u16> {
        frame
            .payload
            .bytes()
            .chunks_exact(2)
            .map(|b| u16::from_le_bytes([b[0], b[1]]))
            .collect()
    }

    #[test]
    fn decimation_keeps_the_nearest_valid_range_and_never_invents_one() {
        // 5x3 into at most 2 wide: factor 3, a 2x1 output with a partial block.
        #[rustfmt::skip]
        let frame = depth(5, 3, &[
            900, 0,   1200, 0, 0,
            0,   400, 1200, 0, 0,
            1300, 0,  1200, 0, 0,
        ]);
        let small = decimate_depth_valid_min(&frame, 2, 2, "viewer_preview").unwrap();
        assert_eq!(samples(&small), [400, 0], "the near side of the edge; an empty block stays invalid");
        assert_eq!(
            (small.metadata.profile.width, small.metadata.profile.height),
            (2, 1)
        );
        let attributes = &small.metadata.attributes;
        assert_eq!(attributes["viewer_preview_source_dimensions"], "5x3");
        assert_eq!(attributes["viewer_preview_resize_filter"], "valid_min");
        assert_eq!(attributes["depth_units_m"], "0.001", "sensor units are kept");
        // A plane that already fits, or a frame that is not depth, is left alone.
        assert!(decimate_depth_valid_min(&frame, 5, 3, "viewer_preview").is_none());
        let mut color = frame.clone();
        color.payload = RecordedPayload::Video {
            format: PixelFormat::Rgb8,
            width: 5,
            height: 3,
            bytes: vec![0; 45],
        };
        assert!(decimate_depth_valid_min(&color, 2, 2, "viewer_preview").is_none());
    }

    #[test]
    fn the_d555_aligned_depth_halves_to_the_viewer_preview() {
        let frame = depth(1280, 720, &vec![700; 1280 * 720]);
        let small = decimate_depth_valid_min(&frame, 640, 360, "viewer_preview").unwrap();
        assert_eq!(
            (small.metadata.profile.width, small.metadata.profile.height),
            (640, 360)
        );
        assert!(samples(&small).iter().all(|v| *v == 700));
    }

    #[test]
    fn the_range_clip_drops_the_room_and_keeps_the_table_in_sensor_units() {
        // Millimetres: table 400-786, floor 1800, invalid 0.
        let mut frame = depth(5, 1, &[400, 786, 850, 851, 1800]);
        clip_depth_beyond(&mut frame, 0.85, "viewer_preview");
        assert_eq!(samples(&frame), [400, 786, 850, 0, 0]);
        assert_eq!(frame.metadata.attributes["viewer_preview_max_range_m"], "0.85");
        // 0.1 mm units (a D405): the same metres, ten times the samples.
        let mut fine = depth(3, 1, &[4000, 8500, 8501]);
        fine.metadata.attributes.insert("depth_units_m".into(), "0.0001".into());
        clip_depth_beyond(&mut fine, 0.85, "viewer_preview");
        assert_eq!(samples(&fine), [4000, 8500, 0]);
        // Without metric units the scale is unknown and nothing is dropped.
        let mut unknown = depth(2, 1, &[400, 1800]);
        unknown.metadata.attributes.clear();
        clip_depth_beyond(&mut unknown, 0.85, "viewer_preview");
        assert_eq!(samples(&unknown), [400, 1800]);
    }

    #[test]
    fn rvl_depth_is_what_the_viewer_decodes_back_exactly() {
        // Zero runs at both ends and between, a jump to the far end of the
        // range and back, one-sample runs: every branch of the coder.
        let mut plane: Vec<u16> = vec![0, 0, 0, 700, 701, 699, 0, 65535, 1, 0];
        plane.extend((0..1000_u32).map(|i| if i % 97 == 0 { 0 } else { 600 + (i * 7 % 13) as u16 }));
        plane.extend([0; 11]);
        let bytes: Vec<u8> = plane.iter().flat_map(|v| v.to_le_bytes()).collect();
        let (width, height) = (plane.len() as u32, 1);
        let blob = rvl_depth(width, height, &bytes);
        let metadata = re_rvl::RosRvlMetadata::parse(&blob).unwrap();
        assert_eq!((metadata.width, metadata.height), (width, height));
        assert!(!metadata.has_quantization(), "samples stay in sensor units");
        let decoded = re_rvl::decode_rvl_with_quantization(&blob, &metadata).unwrap();
        let decoded: Vec<u16> = decoded.iter().map(|v| *v as u16).collect();
        assert_eq!(decoded, plane);
        // A smooth surface costs well under a byte a sample.
        let smooth: Vec<u8> = (0..320 * 180_u32)
            .flat_map(|i| ((500 + (i % 320) / 4 + (i / 320) / 3) as u16).to_le_bytes())
            .collect();
        assert!(rvl_depth(320, 180, &smooth).len() < smooth.len() / 3);
    }
}
