use std::{
    fs::{self, File, OpenOptions},
    io::{BufWriter, Write},
    path::{Path, PathBuf},
};

use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

use crate::{FrameMetadata, FrameRecord};

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum RecordedPayload {
    Encoded {
        format: crate::PixelFormat,
        bytes: Vec<u8>,
    },
    Video {
        format: crate::PixelFormat,
        width: u32,
        height: u32,
        bytes: Vec<u8>,
    },
    Depth {
        width: u32,
        height: u32,
        bytes: Vec<u8>,
    },
}

impl RecordedPayload {
    pub fn bytes(&self) -> &[u8] {
        match self {
            Self::Encoded { bytes, .. } | Self::Video { bytes, .. } | Self::Depth { bytes, .. } => {
                bytes
            }
        }
    }

    fn extension(&self) -> &'static str {
        match self {
            Self::Encoded {
                format: crate::PixelFormat::H264,
                ..
            } => "h264",
            Self::Encoded {
                format: crate::PixelFormat::Jpeg,
                ..
            } => "jpg",
            Self::Encoded { .. } => "bin",
            Self::Video {
                format: crate::PixelFormat::Bgr8,
                ..
            } => "bgr8",
            Self::Video {
                format: crate::PixelFormat::Rgb8,
                ..
            } => "rgb8",
            Self::Video {
                format: crate::PixelFormat::Yuyv,
                ..
            } => "yuyv",
            Self::Video { .. } => "bin",
            Self::Depth { .. } => "z16",
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RecordingManifest {
    pub schema_version: u32,
    pub sensor_name: String,
    pub frame_count: u64,
    pub metadata_file: String,
}

#[derive(Debug)]
pub struct EvidenceRecorder {
    root: PathBuf,
    sensor_name: String,
    metadata: BufWriter<File>,
    frame_count: u64,
}

impl EvidenceRecorder {
    pub fn create(root: impl AsRef<Path>, sensor_name: &str) -> Result<Self> {
        if sensor_name.is_empty() || sensor_name.contains('/') || sensor_name.contains('\\') {
            anyhow::bail!("invalid sensor name for recorder: {sensor_name}");
        }
        let root = root.as_ref().join(sensor_name);
        fs::create_dir_all(&root)
            .with_context(|| format!("creating recording directory {}", root.display()))?;
        let metadata_path = root.join("frames.jsonl");
        if metadata_path.exists() || root.join("manifest.json").exists() {
            anyhow::bail!(
                "recording directory {} already contains evidence; choose a fresh output root",
                root.display()
            );
        }
        let metadata = OpenOptions::new()
            .create_new(true)
            .write(true)
            .open(&metadata_path)
            .with_context(|| format!("opening {}", metadata_path.display()))?;
        Ok(Self {
            root,
            sensor_name: sensor_name.to_owned(),
            metadata: BufWriter::new(metadata),
            frame_count: 0,
        })
    }

    /// Subscriber evidence retains full image dimensions but bounds disk traffic
    /// with explicitly labelled JPEG rather than writing decoded main-stream RGB.
    #[cfg(feature = "zenoh")]
    pub fn write_jpeg(&mut self, frame: &FrameRecord) -> Result<PathBuf> {
        frame.validate().map_err(anyhow::Error::msg)?;
        let RecordedPayload::Video {
            format,
            width,
            height,
            bytes,
        } = &frame.payload
        else {
            return self.write(frame);
        };
        anyhow::ensure!(
            matches!(
                format,
                crate::PixelFormat::Rgb8 | crate::PixelFormat::Bgr8 | crate::PixelFormat::Y8
            ),
            "JPEG evidence requires RGB/BGR/Y8"
        );
        let mut rgb = bytes.clone();
        if *format == crate::PixelFormat::Bgr8 {
            for pixel in rgb.chunks_exact_mut(3) {
                pixel.swap(0, 2);
            }
        }
        let mut bytes = Vec::new();
        image::codecs::jpeg::JpegEncoder::new_with_quality(&mut bytes, 95).encode(
            &rgb,
            *width,
            *height,
            if *format == crate::PixelFormat::Y8 {
                image::ExtendedColorType::L8
            } else {
                image::ExtendedColorType::Rgb8
            },
        )?;
        let mut metadata = frame.metadata.clone();
        metadata.profile.format = crate::PixelFormat::Jpeg;
        metadata
            .attributes
            .insert("evidence.encoding".into(), "jpeg-quality-95".into());
        metadata.attributes.insert(
            "evidence.source_format".into(),
            match format {
                crate::PixelFormat::Y8 => "y8",
                crate::PixelFormat::Bgr8 => "bgr8",
                _ => "rgb8",
            }
            .into(),
        );
        self.write(&FrameRecord {
            metadata,
            payload: RecordedPayload::Encoded {
                format: crate::PixelFormat::Jpeg,
                bytes,
            },
        })
    }

    pub fn write(&mut self, frame: &FrameRecord) -> Result<PathBuf> {
        frame.validate().map_err(anyhow::Error::msg)?;
        if frame.metadata.sensor_name != self.sensor_name {
            anyhow::bail!(
                "recorder {} received {}",
                self.sensor_name,
                frame.metadata.sensor_name
            );
        }
        let sequence = frame.metadata.sequence;
        let payload_path = self
            .root
            .join(format!("{sequence:012}.{}", frame.payload.extension()));
        fs::write(&payload_path, frame.payload.bytes())
            .with_context(|| format!("writing {}", payload_path.display()))?;
        let mut hasher = Sha256::new();
        hasher.update(frame.payload.bytes());
        let entry = RecordingEntry {
            metadata: frame.metadata.clone(),
            payload_file: payload_path
                .file_name()
                .unwrap()
                .to_string_lossy()
                .into_owned(),
            payload_bytes: frame.payload.bytes().len() as u64,
            sha256: hex::encode(hasher.finalize()),
        };
        serde_json::to_writer(&mut self.metadata, &entry)?;
        self.metadata.write_all(b"\n")?;
        self.metadata.flush()?;
        self.frame_count += 1;
        Ok(payload_path)
    }

    pub fn finish(mut self) -> Result<RecordingManifest> {
        self.metadata.flush()?;
        let manifest = RecordingManifest {
            schema_version: 1,
            sensor_name: self.sensor_name,
            frame_count: self.frame_count,
            metadata_file: "frames.jsonl".into(),
        };
        let path = self.root.join("manifest.json");
        fs::write(&path, serde_json::to_vec_pretty(&manifest)?)?;
        Ok(manifest)
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RecordingEntry {
    pub metadata: FrameMetadata,
    pub payload_file: String,
    pub payload_bytes: u64,
    pub sha256: String,
}

/// Read only the frame metadata from an evidence JSONL file.
///
/// Keeping this reader independent of payload bytes makes synchronization and
/// quality reports cheap to run over long captures.
pub fn read_recording_entries(path: impl AsRef<Path>) -> Result<Vec<RecordingEntry>> {
    let path = path.as_ref();
    let file = File::open(path).with_context(|| format!("opening {}", path.display()))?;
    let mut entries = Vec::new();
    for (line_number, line) in std::io::BufRead::lines(std::io::BufReader::new(file)).enumerate() {
        let line =
            line.with_context(|| format!("reading {} line {}", path.display(), line_number + 1))?;
        if line.trim().is_empty() {
            continue;
        }
        entries.push(
            serde_json::from_str(&line)
                .with_context(|| format!("parsing {} line {}", path.display(), line_number + 1))?,
        );
    }
    Ok(entries)
}

/// Load one complete frame from an evidence JSONL file and verify its payload
/// checksum before returning it. This lets replay tools stream large captures
/// without loading every image into memory at once.
pub fn read_recording_frame(
    metadata_path: impl AsRef<Path>,
    entry: &RecordingEntry,
) -> Result<FrameRecord> {
    let metadata_path = metadata_path.as_ref();
    let payload_path = metadata_path
        .parent()
        .unwrap_or_else(|| Path::new("."))
        .join(&entry.payload_file);
    let bytes = fs::read(&payload_path)
        .with_context(|| format!("reading payload {}", payload_path.display()))?;
    if bytes.len() as u64 != entry.payload_bytes {
        anyhow::bail!(
            "payload length mismatch for {}: expected {}, got {}",
            payload_path.display(),
            entry.payload_bytes,
            bytes.len()
        );
    }
    let mut hasher = Sha256::new();
    hasher.update(&bytes);
    let checksum = hex::encode(hasher.finalize());
    if checksum != entry.sha256 {
        anyhow::bail!("payload checksum mismatch for {}", payload_path.display());
    }
    let payload = match entry.metadata.profile.format {
        crate::PixelFormat::H264 | crate::PixelFormat::Jpeg => crate::RecordedPayload::Encoded {
            format: entry.metadata.profile.format,
            bytes,
        },
        crate::PixelFormat::Bgr8
        | crate::PixelFormat::Rgb8
        | crate::PixelFormat::Yuyv
        | crate::PixelFormat::Y8 => crate::RecordedPayload::Video {
            format: entry.metadata.profile.format,
            width: entry.metadata.profile.width,
            height: entry.metadata.profile.height,
            bytes,
        },
        crate::PixelFormat::Z16 => crate::RecordedPayload::Depth {
            width: entry.metadata.profile.width,
            height: entry.metadata.profile.height,
            bytes,
        },
    };
    #[allow(unused_mut)]
    let mut frame = FrameRecord {
        metadata: entry.metadata.clone(),
        payload,
    };
    #[cfg(any(feature = "zenoh", feature = "rerun", feature = "gstreamer"))]
    if frame.metadata.profile.format == crate::PixelFormat::Jpeg {
        use image::ImageDecoder;
        let decoder =
            image::codecs::jpeg::JpegDecoder::new(std::io::Cursor::new(frame.payload.bytes()))?;
        anyhow::ensure!(
            decoder.dimensions() == (frame.metadata.profile.width, frame.metadata.profile.height)
                && decoder.total_bytes() <= 128 * 1024 * 1024,
            "JPEG recording dimensions mismatch or limit exceeded"
        );
        let rgb = image::DynamicImage::from_decoder(decoder)?.to_rgb8();
        frame.metadata.profile.format = crate::PixelFormat::Rgb8;
        frame.payload = RecordedPayload::Video {
            format: crate::PixelFormat::Rgb8,
            width: rgb.width(),
            height: rgb.height(),
            bytes: rgb.into_raw(),
        };
    }
    frame.validate().map_err(anyhow::Error::msg)?;
    Ok(frame)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        FrameMetadata, FrameTimestamps, PixelFormat, SensorKind, StreamProfile, TimestampDomain,
    };

    #[test]
    fn writes_manifest_metadata_and_checksum() {
        let directory = tempfile::tempdir().unwrap();
        let mut recorder = EvidenceRecorder::create(directory.path(), "camera1").unwrap();
        let frame = FrameRecord {
            metadata: FrameMetadata {
                sensor_name: "camera1".into(),
                sensor_kind: SensorKind::PoE,
                sequence: 0,
                profile: StreamProfile {
                    stream: "main".into(),
                    width: 2,
                    height: 1,
                    fps_num: 5,
                    fps_den: 1,
                    format: PixelFormat::H264,
                },
                timestamps: FrameTimestamps {
                    source_ns: None,
                    source_domain: TimestampDomain::Unknown,
                    rtp_timestamp: None,
                    pipeline_pts_ns: None,
                    pipeline_dts_ns: None,
                    host_monotonic_ns: 1,
                    host_unix_ns: 1_700_000_000_000_000_000,
                    normalized_unix_ns: None,
                },
                dropped_before: 0,
                calibration_id: None,
                flags: vec![],
                attributes: std::collections::BTreeMap::new(),
            },
            payload: RecordedPayload::Encoded {
                format: PixelFormat::H264,
                bytes: vec![1, 2, 3],
            },
        };
        recorder.write(&frame).unwrap();
        let manifest = recorder.finish().unwrap();
        assert_eq!(manifest.frame_count, 1);
        assert!(directory.path().join("camera1/000000000000.h264").exists());
        assert!(directory.path().join("camera1/frames.jsonl").exists());
        assert!(directory.path().join("camera1/manifest.json").exists());
        let entries =
            read_recording_entries(directory.path().join("camera1/frames.jsonl")).unwrap();
        let loaded =
            read_recording_frame(directory.path().join("camera1/frames.jsonl"), &entries[0])
                .unwrap();
        assert_eq!(loaded.payload, frame.payload);
        #[cfg(feature = "zenoh")]
        {
            let jpeg_dir = tempfile::tempdir().unwrap();
            let mut recorder = EvidenceRecorder::create(jpeg_dir.path(), "camera1").unwrap();
            let mut rgb = frame.clone();
            rgb.metadata.profile.format = PixelFormat::Rgb8;
            rgb.payload = RecordedPayload::Video {
                format: PixelFormat::Rgb8,
                width: 2,
                height: 1,
                bytes: vec![240, 0, 0, 240, 0, 0],
            };
            let path = recorder.write_jpeg(&rgb).unwrap();
            recorder.finish().unwrap();
            assert_eq!(path.extension().unwrap(), "jpg");
            let metadata = jpeg_dir.path().join("camera1/frames.jsonl");
            let entries = read_recording_entries(&metadata).unwrap();
            assert_eq!(entries[0].metadata.profile.format, PixelFormat::Jpeg);
            let decoded = read_recording_frame(&metadata, &entries[0]).unwrap();
            assert_eq!(decoded.metadata.profile.format, PixelFormat::Rgb8);
            assert_eq!(decoded.metadata.timestamps, rgb.metadata.timestamps);
            assert_eq!(decoded.payload.bytes().len(), 6);
            assert!(decoded.payload.bytes()[0] > 230);
            std::fs::write(path, b"corrupted").unwrap();
            assert!(read_recording_frame(&metadata, &entries[0]).is_err());
            // The always-on owner supplies luma to calibration subscribers.
            // Preserve that source provenance while replay remains RGB.
            let gray_dir = tempfile::tempdir().unwrap();
            let mut recorder = EvidenceRecorder::create(gray_dir.path(), "camera1").unwrap();
            let mut gray = rgb.clone();
            gray.metadata.profile.format = PixelFormat::Y8;
            gray.payload = RecordedPayload::Video {
                format: PixelFormat::Y8,
                width: 2,
                height: 1,
                bytes: vec![110; 2],
            };
            recorder.write_jpeg(&gray).unwrap();
            recorder.finish().unwrap();
            let metadata = gray_dir.path().join("camera1/frames.jsonl");
            let entries = read_recording_entries(&metadata).unwrap();
            assert_eq!(
                entries[0].metadata.attributes["evidence.source_format"],
                "y8"
            );
            let decoded = read_recording_frame(&metadata, &entries[0]).unwrap();
            assert_eq!(decoded.metadata.timestamps, gray.metadata.timestamps);
            assert_eq!(decoded.metadata.profile.format, PixelFormat::Rgb8);
            assert_eq!(decoded.payload.bytes().len(), 6);
            assert!(decoded.payload.bytes().iter().all(|v| v.abs_diff(110) <= 3));
        }
    }
}
