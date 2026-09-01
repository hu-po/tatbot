//! Bounded local transport for synchronized frame sets.
//!
//! The wire format keeps metadata in a length-delimited JSON header and sends
//! image/depth bytes immediately after it. This avoids base64 expansion while
//! keeping the protocol inspectable and easy to bridge into Python, C++, or a
//! policy runtime. The publisher is deliberately best-effort: a slow client
//! is disconnected rather than allowed to stall camera capture.

use std::{
    fs,
    io::{self, Read, Write},
    os::unix::{
        fs::FileTypeExt,
        net::{UnixListener, UnixStream},
    },
    path::{Path, PathBuf},
    time::Duration,
};

use anyhow::{Context, Result, anyhow};
use serde::{Deserialize, Serialize};

use crate::{FrameMetadata, FrameRecord, PixelFormat, RecordedPayload, SynchronizedFrameSet};

const WIRE_MAGIC: &str = "tatbot-vision-frame-set";
const WIRE_VERSION: u32 = 1;
const MAX_HEADER_BYTES: usize = 4 * 1024 * 1024;
const MAX_PAYLOAD_BYTES: usize = 128 * 1024 * 1024;

#[derive(Debug, Serialize, Deserialize)]
struct WireFrameSetHeader {
    magic: String,
    version: u32,
    sequence: u64,
    timestamp_basis: String,
    timestamp_ns: i128,
    maximum_skew_ns: u128,
    frames: Vec<WireFrameHeader>,
}

#[derive(Debug, Serialize, Deserialize)]
struct WireFrameHeader {
    metadata: FrameMetadata,
    payload: WirePayload,
}

#[derive(Debug, Serialize, Deserialize)]
enum WirePayload {
    Encoded {
        format: PixelFormat,
        bytes: usize,
    },
    Video {
        format: PixelFormat,
        width: u32,
        height: u32,
        bytes: usize,
    },
    Depth {
        width: u32,
        height: u32,
        bytes: usize,
    },
}

#[derive(Debug)]
pub struct ReceivedFrameSet {
    pub sequence: u64,
    pub timestamp_basis: String,
    pub timestamp_ns: i128,
    pub maximum_skew_ns: u128,
    pub frames: Vec<FrameRecord>,
}

#[derive(Debug)]
pub struct UnixFramePublisher {
    listener: UnixListener,
    path: PathBuf,
    clients: Vec<UnixStream>,
}

impl UnixFramePublisher {
    pub fn bind(path: impl AsRef<Path>) -> Result<Self> {
        let path = path.as_ref().to_path_buf();
        if let Ok(metadata) = fs::symlink_metadata(&path) {
            if !metadata.file_type().is_socket() {
                anyhow::bail!(
                    "transport path {} exists and is not a socket",
                    path.display()
                );
            }
            fs::remove_file(&path)
                .with_context(|| format!("removing stale transport socket {}", path.display()))?;
        }
        let listener = UnixListener::bind(&path)
            .with_context(|| format!("binding transport socket {}", path.display()))?;
        listener
            .set_nonblocking(true)
            .context("setting transport listener nonblocking")?;
        Ok(Self {
            listener,
            path,
            clients: Vec::new(),
        })
    }

    pub fn client_count(&self) -> usize {
        self.clients.len()
    }

    pub fn poll_accept(&mut self) -> Result<usize> {
        let mut accepted = 0;
        loop {
            match self.listener.accept() {
                Ok((stream, _)) => {
                    stream
                        .set_write_timeout(Some(Duration::from_millis(20)))
                        .context("setting transport client write timeout")?;
                    self.clients.push(stream);
                    accepted += 1;
                }
                Err(error) if error.kind() == io::ErrorKind::WouldBlock => break,
                Err(error) => return Err(error).context("accepting transport client"),
            }
        }
        Ok(accepted)
    }

    /// Publish one set. Returns the number of clients that accepted the full
    /// message; slow or disconnected clients are removed.
    pub fn publish(&mut self, set: &SynchronizedFrameSet) -> Result<usize> {
        self.poll_accept()?;
        let mut headers = Vec::with_capacity(set.frames.len());
        let mut payloads = Vec::with_capacity(set.frames.len());
        for frame in set.frames.values() {
            let (payload, descriptor) = payload_parts(&frame.payload);
            if payload.len() > MAX_PAYLOAD_BYTES {
                anyhow::bail!("frame payload exceeds transport limit");
            }
            headers.push(WireFrameHeader {
                metadata: frame.metadata.clone(),
                payload: descriptor,
            });
            payloads.push(payload);
        }
        let header = WireFrameSetHeader {
            magic: WIRE_MAGIC.into(),
            version: WIRE_VERSION,
            sequence: set.sequence,
            timestamp_basis: set.timestamp_basis.clone(),
            timestamp_ns: set.timestamp_ns,
            maximum_skew_ns: set.maximum_skew_ns,
            frames: headers,
        };
        let header_bytes = serde_json::to_vec(&header)?;
        if header_bytes.len() > MAX_HEADER_BYTES {
            anyhow::bail!("frame-set header exceeds transport limit");
        }
        let mut message = Vec::with_capacity(4 + header_bytes.len());
        message.extend_from_slice(&(header_bytes.len() as u32).to_be_bytes());
        message.extend_from_slice(&header_bytes);

        let mut delivered = 0;
        self.clients.retain_mut(|client| {
            let result = (|| -> Result<()> {
                client.write_all(&message)?;
                for payload in &payloads {
                    client.write_all(payload)?;
                }
                Ok(())
            })();
            if result.is_ok() {
                delivered += 1;
                true
            } else {
                false
            }
        });
        Ok(delivered)
    }
}

impl Drop for UnixFramePublisher {
    fn drop(&mut self) {
        let _ = fs::remove_file(&self.path);
    }
}

#[derive(Debug)]
pub struct UnixFrameClient {
    stream: UnixStream,
}

impl UnixFrameClient {
    pub fn connect(path: impl AsRef<Path>) -> Result<Self> {
        let stream = UnixStream::connect(path.as_ref())
            .with_context(|| format!("connecting transport socket {}", path.as_ref().display()))?;
        Ok(Self { stream })
    }

    pub fn recv(&mut self) -> Result<ReceivedFrameSet> {
        let mut length_bytes = [0_u8; 4];
        self.stream.read_exact(&mut length_bytes)?;
        let header_length = u32::from_be_bytes(length_bytes) as usize;
        if header_length == 0 || header_length > MAX_HEADER_BYTES {
            anyhow::bail!("invalid frame-set header length {header_length}");
        }
        let mut header_bytes = vec![0_u8; header_length];
        self.stream.read_exact(&mut header_bytes)?;
        let header: WireFrameSetHeader = serde_json::from_slice(&header_bytes)?;
        if header.magic != WIRE_MAGIC || header.version != WIRE_VERSION {
            anyhow::bail!("unsupported frame transport header");
        }
        let mut frames = Vec::with_capacity(header.frames.len());
        for wire_frame in header.frames {
            let payload_bytes = wire_payload_size(&wire_frame.payload);
            if payload_bytes > MAX_PAYLOAD_BYTES {
                anyhow::bail!("frame payload exceeds transport limit");
            }
            let mut bytes = vec![0_u8; payload_bytes];
            self.stream.read_exact(&mut bytes)?;
            frames.push(FrameRecord {
                metadata: wire_frame.metadata,
                payload: payload_from_wire(wire_frame.payload, bytes)?,
            });
        }
        Ok(ReceivedFrameSet {
            sequence: header.sequence,
            timestamp_basis: header.timestamp_basis,
            timestamp_ns: header.timestamp_ns,
            maximum_skew_ns: header.maximum_skew_ns,
            frames,
        })
    }
}

fn payload_parts(payload: &RecordedPayload) -> (&[u8], WirePayload) {
    match payload {
        RecordedPayload::Encoded { format, bytes } => (
            bytes,
            WirePayload::Encoded {
                format: *format,
                bytes: bytes.len(),
            },
        ),
        RecordedPayload::Video {
            format,
            width,
            height,
            bytes,
        } => (
            bytes,
            WirePayload::Video {
                format: *format,
                width: *width,
                height: *height,
                bytes: bytes.len(),
            },
        ),
        RecordedPayload::Depth {
            width,
            height,
            bytes,
        } => (
            bytes,
            WirePayload::Depth {
                width: *width,
                height: *height,
                bytes: bytes.len(),
            },
        ),
    }
}

fn wire_payload_size(payload: &WirePayload) -> usize {
    match payload {
        WirePayload::Encoded { bytes, .. }
        | WirePayload::Video { bytes, .. }
        | WirePayload::Depth { bytes, .. } => *bytes,
    }
}

fn payload_from_wire(payload: WirePayload, bytes: Vec<u8>) -> Result<RecordedPayload> {
    if wire_payload_size(&payload) != bytes.len() {
        return Err(anyhow!("wire payload length mismatch"));
    }
    Ok(match payload {
        WirePayload::Encoded { format, .. } => RecordedPayload::Encoded { format, bytes },
        WirePayload::Video {
            format,
            width,
            height,
            ..
        } => RecordedPayload::Video {
            format,
            width,
            height,
            bytes,
        },
        WirePayload::Depth { width, height, .. } => RecordedPayload::Depth {
            width,
            height,
            bytes,
        },
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{FrameTimestamps, SensorKind, StreamProfile, TimestampDomain};
    use std::collections::BTreeMap;

    #[test]
    fn payload_round_trip_preserves_descriptor() {
        for original in [
            RecordedPayload::Encoded {
                format: PixelFormat::H264,
                bytes: vec![1, 2, 3],
            },
            RecordedPayload::Video {
                format: PixelFormat::Bgr8,
                width: 2,
                height: 1,
                bytes: vec![4, 5, 6, 7, 8, 9],
            },
            RecordedPayload::Depth {
                width: 2,
                height: 1,
                bytes: vec![10, 11, 12, 13],
            },
        ] {
            let (bytes, descriptor) = payload_parts(&original);
            let restored = payload_from_wire(descriptor, bytes.to_vec()).unwrap();
            assert_eq!(restored, original);
        }
    }

    #[test]
    fn unix_transport_round_trip_preserves_frame_set() {
        let directory = tempfile::tempdir().unwrap();
        let socket = directory.path().join("frames.sock");
        let mut publisher = UnixFramePublisher::bind(&socket).unwrap();
        let mut client = UnixFrameClient::connect(&socket).unwrap();
        let frame = FrameRecord {
            metadata: FrameMetadata {
                sensor_name: "camera1".into(),
                sensor_kind: SensorKind::PoE,
                sequence: 7,
                profile: StreamProfile {
                    stream: "main".into(),
                    width: 2,
                    height: 1,
                    fps_num: 15,
                    fps_den: 1,
                    format: PixelFormat::Bgr8,
                },
                timestamps: FrameTimestamps {
                    source_ns: Some(1_000),
                    source_domain: TimestampDomain::CameraNtp,
                    rtp_timestamp: Some(90),
                    pipeline_pts_ns: Some(500),
                    pipeline_dts_ns: None,
                    host_monotonic_ns: 2_000,
                    host_unix_ns: 3_000,
                    normalized_unix_ns: Some(3_000),
                },
                dropped_before: 0,
                calibration_id: Some("bundle".into()),
                flags: vec!["decoded_bgr".into()],
                attributes: BTreeMap::new(),
            },
            payload: RecordedPayload::Video {
                format: PixelFormat::Bgr8,
                width: 2,
                height: 1,
                bytes: vec![1, 2, 3, 4, 5, 6],
            },
        };
        let set = SynchronizedFrameSet {
            sequence: 4,
            timestamp_basis: "normalized_unix_ns".into(),
            timestamp_ns: 3_000,
            maximum_skew_ns: 2,
            frames: BTreeMap::from([(String::from("camera1"), frame)]),
        };
        assert_eq!(publisher.publish(&set).unwrap(), 1);
        let received = client.recv().unwrap();
        assert_eq!(received.sequence, 4);
        assert_eq!(received.frames.len(), 1);
        assert_eq!(
            received.frames[0].metadata.calibration_id.as_deref(),
            Some("bundle")
        );
        assert_eq!(received.frames[0].payload, set.frames["camera1"].payload);
    }
}
