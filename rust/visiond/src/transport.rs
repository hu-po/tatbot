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
    time::{Duration, Instant},
};

use anyhow::{Context, Result, anyhow};
use serde::{Deserialize, Serialize};

use crate::{FrameMetadata, FrameRecord, PixelFormat, RecordedPayload, SynchronizedFrameSet};

fn transport_wall_ns() -> u128 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_nanos()
}

const WIRE_MAGIC: &str = "tatbot-vision-frame-set";
const WIRE_VERSION: u32 = 1;
const MAX_HEADER_BYTES: usize = 4 * 1024 * 1024;
const MAX_PAYLOAD_BYTES: usize = 128 * 1024 * 1024;
/// Longest a subscriber waits for the capture owner to bind its listener.
/// The owners' units are `Type=notify` and report ready once bound (see
/// `systemd`), so an `After=` consumer normally connects at once; the window
/// covers an owner started by hand or restarted underneath its subscribers.
pub const CONNECT_WINDOW: Duration = Duration::from_secs(10);
const CONNECT_BACKOFF_START: Duration = Duration::from_millis(25);
const CONNECT_BACKOFF_MAX: Duration = Duration::from_millis(500);

#[derive(Debug, Serialize, Deserialize)]
struct WireFrameSetHeader {
    magic: String,
    version: u32,
    sequence: u64,
    timestamp_basis: String,
    timestamp_ns: i128,
    maximum_skew_ns: u128,
    frames: Vec<WireFrameHeader>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    envelope: Option<serde_json::Value>,
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
    pub envelope: Option<serde_json::Value>,
    pub sequence: u64,
    pub timestamp_basis: String,
    pub timestamp_ns: i128,
    pub maximum_skew_ns: u128,
    pub frames: Vec<FrameRecord>,
}

#[derive(Debug)]
pub struct UnixFramePublisher {
    _lease: crate::ownership::CameraLease,
    listener: UnixListener,
    path: PathBuf,
    clients: Vec<UnixStream>,
}

impl UnixFramePublisher {
    pub fn bind(path: impl AsRef<Path>) -> Result<Self> {
        let path = path.as_ref().to_path_buf();
        let mut lock_path = path.as_os_str().to_owned();
        lock_path.push(".lock");
        let lease = crate::ownership::CameraLease::acquire(Path::new(&lock_path))?;
        if let Ok(metadata) = fs::symlink_metadata(&path) {
            if !metadata.file_type().is_socket() {
                anyhow::bail!(
                    "transport path {} exists and is not a socket",
                    path.display()
                );
            }
            // Never replace a socket that still has a live owner. A failed
            // permission check is not evidence of a stale owner either.
            match UnixStream::connect(&path) {
                Ok(_) => anyhow::bail!(
                    "transport socket {} already has a live owner",
                    path.display()
                ),
                Err(error) if error.kind() == io::ErrorKind::ConnectionRefused => {}
                Err(error) => return Err(error).context("checking existing transport owner"),
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
            _lease: lease,
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
        let publish_unix_ns = transport_wall_ns();
        let mut headers = Vec::with_capacity(set.frames.len());
        let mut payloads = Vec::with_capacity(set.frames.len());
        for frame in set.frames.values() {
            let (payload, descriptor) = payload_parts(&frame.payload);
            if payload.len() > MAX_PAYLOAD_BYTES {
                anyhow::bail!("frame payload exceeds transport limit");
            }
            let mut metadata = frame.metadata.clone();
            metadata.attributes.insert(
                "transport_publish_unix_ns".into(),
                publish_unix_ns.to_string(),
            );
            headers.push(WireFrameHeader {
                metadata,
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
            envelope: None,
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

    /// Connect, tolerating an owner that has been started but has not bound
    /// yet. Exactly two errors mean "not up yet" and retry: no socket file
    /// (the owner has not bound) and a refused connection (a stale socket
    /// file the owner has not replaced yet). Everything else is a
    /// misconfiguration and fails on the first attempt -- including a path
    /// that exists and is not a socket, which `connect` reports as
    /// `ConnectionRefused` and would otherwise retry to the deadline.
    pub fn connect_within(path: impl AsRef<Path>, window: Duration) -> Result<Self> {
        let path = path.as_ref();
        let deadline = Instant::now() + window;
        let mut backoff = CONNECT_BACKOFF_START;
        loop {
            if let Ok(metadata) = fs::symlink_metadata(path) {
                anyhow::ensure!(
                    metadata.file_type().is_socket(),
                    "transport path {} exists and is not a socket",
                    path.display()
                );
            }
            let error = match UnixStream::connect(path) {
                Ok(stream) => return Ok(Self { stream }),
                Err(error) => error,
            };
            let unbound = matches!(
                error.kind(),
                io::ErrorKind::NotFound | io::ErrorKind::ConnectionRefused
            );
            let remaining = deadline.saturating_duration_since(Instant::now());
            if !unbound {
                return Err(error)
                    .with_context(|| format!("connecting transport socket {}", path.display()));
            }
            if remaining.is_zero() {
                return Err(error).with_context(|| {
                    format!(
                        "transport socket {} had no owner within {:.1}s",
                        path.display(),
                        window.as_secs_f64()
                    )
                });
            }
            std::thread::sleep(backoff.min(remaining));
            backoff = (backoff * 2).min(CONNECT_BACKOFF_MAX);
        }
    }

    pub fn set_read_timeout(&self, timeout: Duration) -> Result<()> {
        self.stream.set_read_timeout(Some(timeout))?;
        Ok(())
    }
    pub fn recv(&mut self) -> Result<ReceivedFrameSet> {
        read_frame_set(&mut self.stream)
    }

    /// None means an idle timeout before any packet byte was consumed.
    /// A timeout within a packet is fatal: retrying would lose framing.
    pub fn recv_if_ready(&mut self) -> Result<Option<ReceivedFrameSet>> {
        let mut first = [0_u8; 1];
        loop {
            match self.stream.read(&mut first) {
                Ok(0) => return Err(std::io::Error::from(std::io::ErrorKind::UnexpectedEof).into()),
                Ok(_) => break,
                Err(error)
                    if matches!(
                        error.kind(),
                        std::io::ErrorKind::WouldBlock | std::io::ErrorKind::TimedOut
                    ) =>
                {
                    return Ok(None);
                }
                Err(error) if error.kind() == std::io::ErrorKind::Interrupted => continue,
                Err(error) => return Err(error.into()),
            }
        }
        let mut packet = std::io::Cursor::new(first).chain(&mut self.stream);
        read_frame_set(&mut packet).map(Some)
    }
}

fn read_frame_set(stream: &mut impl Read) -> Result<ReceivedFrameSet> {
    let mut length_bytes = [0_u8; 4];
    stream.read_exact(&mut length_bytes)?;
    let header_length = u32::from_be_bytes(length_bytes) as usize;
    if header_length == 0 || header_length > MAX_HEADER_BYTES {
        anyhow::bail!("invalid frame-set header length {header_length}");
    }
    let mut header_bytes = vec![0_u8; header_length];
    stream.read_exact(&mut header_bytes)?;
    let header: WireFrameSetHeader = serde_json::from_slice(&header_bytes)?;
    if header.magic != WIRE_MAGIC || header.version != WIRE_VERSION {
        anyhow::bail!("unsupported frame transport header");
    }
    anyhow::ensure!(
        header.frames.len() <= 16,
        "frame count exceeds transport limit"
    );
    let total = header
        .frames
        .iter()
        .try_fold(0_usize, |total, frame| {
            total.checked_add(wire_payload_size(&frame.payload))
        })
        .ok_or_else(|| anyhow!("payload size overflow"))?;
    anyhow::ensure!(
        total <= MAX_PAYLOAD_BYTES,
        "frame set exceeds transport limit"
    );
    let mut frames = Vec::with_capacity(header.frames.len());
    for wire_frame in header.frames {
        let payload_bytes = wire_payload_size(&wire_frame.payload);
        if payload_bytes > MAX_PAYLOAD_BYTES {
            anyhow::bail!("frame payload exceeds transport limit");
        }
        let mut bytes = vec![0_u8; payload_bytes];
        stream.read_exact(&mut bytes)?;
        frames.push(FrameRecord {
            metadata: wire_frame.metadata,
            payload: payload_from_wire(wire_frame.payload, bytes)?,
        });
    }
    Ok(ReceivedFrameSet {
        envelope: header.envelope,
        sequence: header.sequence,
        timestamp_basis: header.timestamp_basis,
        timestamp_ns: header.timestamp_ns,
        maximum_skew_ns: header.maximum_skew_ns,
        frames,
    })
}

/// Encode the same wire-v1 header and payloads used by the Unix transport.
pub fn encode_frame_set(
    set: &SynchronizedFrameSet,
    envelope: Option<serde_json::Value>,
) -> Result<Vec<u8>> {
    anyhow::ensure!(
        set.frames.len() <= 16,
        "frame count exceeds transport limit"
    );
    let mut frames = Vec::new();
    let mut payloads = Vec::new();
    let mut total = 0_usize;
    for frame in set.frames.values() {
        let (bytes, payload) = payload_parts(&frame.payload);
        total = total
            .checked_add(bytes.len())
            .ok_or_else(|| anyhow!("payload size overflow"))?;
        anyhow::ensure!(
            total <= MAX_PAYLOAD_BYTES,
            "frame set exceeds transport limit"
        );
        frames.push(WireFrameHeader {
            metadata: frame.metadata.clone(),
            payload,
        });
        payloads.push(bytes);
    }
    let header = serde_json::to_vec(&WireFrameSetHeader {
        magic: WIRE_MAGIC.into(),
        version: WIRE_VERSION,
        sequence: set.sequence,
        timestamp_basis: set.timestamp_basis.clone(),
        timestamp_ns: set.timestamp_ns,
        maximum_skew_ns: set.maximum_skew_ns,
        frames,
        envelope,
    })?;
    anyhow::ensure!(
        header.len() <= MAX_HEADER_BYTES,
        "header exceeds transport limit"
    );
    let mut out = Vec::with_capacity(4 + header.len() + total);
    out.extend_from_slice(&(header.len() as u32).to_be_bytes());
    out.extend_from_slice(&header);
    for bytes in payloads {
        out.extend_from_slice(bytes);
    }
    Ok(out)
}

pub fn decode_frame_set(bytes: &[u8]) -> Result<ReceivedFrameSet> {
    anyhow::ensure!(
        bytes.len() <= MAX_HEADER_BYTES + MAX_PAYLOAD_BYTES + 4,
        "packet exceeds transport limit"
    );
    let mut cursor = io::Cursor::new(bytes);
    let set = read_frame_set(&mut cursor)?;
    anyhow::ensure!(
        cursor.position() == bytes.len() as u64,
        "trailing frame bytes"
    );
    Ok(set)
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

/// The Python encoder's golden set (`scripts/lib/frame_set_wire.py` `pack`):
/// one wrist tap message, tiny. The JSON beside it holds the inputs it was
/// packed from and what a decoder reads back.
#[cfg(test)]
pub(crate) const PYTHON_WIRE_GOLDEN: &[u8] = include_bytes!(concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/../../scripts/tests/fixtures/frame_set_wire_golden.bin"
));
#[cfg(test)]
const PYTHON_WIRE_GOLDEN_JSON: &str = include_str!(concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/../../scripts/tests/fixtures/frame_set_wire_golden.json"
));

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{FrameTimestamps, SensorKind, StreamProfile, TimestampDomain};
    use std::collections::BTreeMap;

    fn sha256_hex(bytes: &[u8]) -> String {
        use sha2::{Digest, Sha256};
        hex::encode(Sha256::digest(bytes))
    }

    fn header_text(wire: &[u8]) -> &str {
        let length = u32::from_be_bytes(wire[..4].try_into().unwrap()) as usize;
        std::str::from_utf8(&wire[4..4 + length]).unwrap()
    }

    #[test]
    fn the_python_wire_golden_decodes_to_the_wrist_tap_frames_it_was_packed_from() {
        let fixture: serde_json::Value = serde_json::from_str(PYTHON_WIRE_GOLDEN_JSON).unwrap();
        let expected = &fixture["expected"];
        assert_eq!(PYTHON_WIRE_GOLDEN.len() as u64, expected["bytes"]);
        assert_eq!(sha256_hex(PYTHON_WIRE_GOLDEN), expected["sha256"]);
        assert_eq!(
            header_text(PYTHON_WIRE_GOLDEN).len() as u64,
            expected["header_bytes"]
        );

        let set = decode_frame_set(PYTHON_WIRE_GOLDEN)
            .expect("the strict decoder reads the Python encoder's set");
        assert_eq!(set.sequence, 3);
        assert_eq!(set.timestamp_basis, "normalized_unix_ns");
        assert_eq!(set.timestamp_ns, 1_760_000_000_123_456_789);
        assert_eq!(set.maximum_skew_ns, 0);
        assert_eq!(set.envelope.as_ref(), Some(&expected["envelope"]));
        let envelope = set.envelope.as_ref().unwrap();
        assert_eq!(envelope["schema"], "tatbot.frame-set/1");
        assert_eq!(envelope["seq"], 3);
        assert_eq!(envelope["stamp"]["wall_ns"], 1_760_000_000_123_456_789_u64);
        assert_eq!(envelope["payload"]["wire_version"], 1);

        let [color, depth] = &set.frames[..] else {
            panic!("two frames, not {}", set.frames.len());
        };
        for frame in [color, depth] {
            let name = frame.metadata.sensor_name.as_str();
            let recorded = &expected["frames"][name];
            assert_eq!(
                serde_json::to_value(&frame.metadata).unwrap(),
                recorded["metadata"],
                "{name}"
            );
            assert_eq!(
                sha256_hex(frame.payload.bytes()),
                recorded["data_sha256"],
                "{name}"
            );
            let metadata = &frame.metadata;
            assert_eq!(metadata.sensor_kind, SensorKind::RealSense);
            assert_eq!(metadata.sequence, 3);
            let stamps = &metadata.timestamps;
            assert_eq!(stamps.source_domain, TimestampDomain::HostUnix);
            assert_eq!(stamps.normalized_unix_ns, Some(set.timestamp_ns));
            assert_eq!(stamps.host_unix_ns, set.timestamp_ns);
            let attributes = &metadata.attributes;
            assert_eq!(attributes["physical_arm"], "right");
            assert_eq!(attributes["capture_owner_role"], "ros");
            assert!(
                !attributes.contains_key("aligned_to"),
                "the golden depth is unaligned"
            );
            // Each stream's own optics, at that stream's size.
            let intrinsics: serde_json::Value =
                serde_json::from_str(&attributes["intrinsics"]).unwrap();
            let profile = &metadata.profile;
            assert_eq!(intrinsics["width"], profile.width, "{name}");
            assert_eq!(intrinsics["height"], profile.height, "{name}");
            for key in ["fx", "fy", "ppx", "ppy"] {
                assert!(
                    intrinsics[key].as_f64().is_some_and(f64::is_finite),
                    "{name} {key}"
                );
            }
        }

        assert_eq!(color.metadata.sensor_name, "realsense2_color");
        assert_eq!(color.metadata.profile.format, PixelFormat::Bgr8);
        assert_eq!(
            (color.metadata.profile.width, color.metadata.profile.height),
            (4, 3)
        );
        let RecordedPayload::Encoded {
            format: PixelFormat::Jpeg,
            bytes: jpeg,
        } = &color.payload
        else {
            panic!("colour is a JPEG, not {:?}", color.payload);
        };
        assert_eq!(jpeg.len(), 697);
        assert!(jpeg.starts_with(&[0xff, 0xd8]) && jpeg.ends_with(&[0xff, 0xd9]));
        assert_eq!(
            color.metadata.attributes["ros_frame"],
            "right/realsense_color_optical_frame"
        );
        assert!(!color.metadata.attributes.contains_key("depth_units_m"));

        assert_eq!(depth.metadata.sensor_name, "realsense2_depth");
        assert_eq!(depth.metadata.profile.format, PixelFormat::Z16);
        let RecordedPayload::Depth {
            width: 3,
            height: 2,
            bytes,
        } = &depth.payload
        else {
            panic!("depth is 3x2 z16, not {:?}", depth.payload);
        };
        let z16: Vec<u16> = bytes
            .chunks_exact(2)
            .map(|sample| u16::from_le_bytes([sample[0], sample[1]]))
            .collect();
        assert_eq!(z16, [1000, 1001, 1002, 2000, 0, 65535]);
        assert_eq!(
            serde_json::to_value(&z16).unwrap(),
            expected["realsense2_depth_z16_row_major"]
        );
        assert_eq!(depth.metadata.attributes["depth_units_m"], "0.0001");
        assert_eq!(
            depth.metadata.attributes["ros_frame"],
            "right/realsense_depth_optical_frame"
        );

        assert!(decode_frame_set(&PYTHON_WIRE_GOLDEN[..PYTHON_WIRE_GOLDEN.len() - 1]).is_err());
        let mut trailing = PYTHON_WIRE_GOLDEN.to_vec();
        trailing.push(0);
        assert!(decode_frame_set(&trailing).is_err());
    }

    #[test]
    fn rust_encodes_the_python_golden_byte_for_byte() {
        // Both encoders write serde's layout: struct fields in declaration
        // order, maps (attributes, the envelope) by key, frames by sensor
        // name. The header has no floats, so the same inputs give the same
        // bytes. Built here from the fixture's inputs, not from the decoder.
        let fixture: serde_json::Value = serde_json::from_str(PYTHON_WIRE_GOLDEN_JSON).unwrap();
        let inputs = &fixture["inputs"];
        let frames = inputs["frames"]
            .as_array()
            .unwrap()
            .iter()
            .map(|frame| {
                let metadata: FrameMetadata =
                    serde_json::from_value(frame["metadata"].clone()).unwrap();
                let descriptor: WirePayload =
                    serde_json::from_value(frame["payload"].clone()).unwrap();
                let bytes = hex::decode(frame["data_hex"].as_str().unwrap()).unwrap();
                let payload = payload_from_wire(descriptor, bytes).unwrap();
                (
                    metadata.sensor_name.clone(),
                    FrameRecord { metadata, payload },
                )
            })
            .collect();
        let set = SynchronizedFrameSet {
            sequence: inputs["sequence"].as_u64().unwrap(),
            timestamp_basis: "normalized_unix_ns".into(),
            timestamp_ns: i128::from(inputs["timestamp_ns"].as_u64().unwrap()),
            maximum_skew_ns: 0,
            frames,
        };
        // The envelope frame_bus::encode writes for these inputs.
        let envelope = serde_json::json!({
            "schema": "tatbot.frame-set/1",
            "producer": inputs["producer"],
            "stamp": {"mono_ns": 0, "wall_ns": inputs["timestamp_ns"], "basis": set.timestamp_basis},
            "seq": set.sequence,
            "payload": {"wire_version": 1},
        });
        let encoded = encode_frame_set(&set, Some(envelope)).unwrap();
        assert_eq!(header_text(&encoded), header_text(PYTHON_WIRE_GOLDEN));
        assert_eq!(encoded, PYTHON_WIRE_GOLDEN);

        // And what the decoder read back re-encodes to the same bytes.
        let decoded = decode_frame_set(PYTHON_WIRE_GOLDEN).unwrap();
        let again = SynchronizedFrameSet {
            sequence: decoded.sequence,
            timestamp_basis: decoded.timestamp_basis,
            timestamp_ns: decoded.timestamp_ns,
            maximum_skew_ns: decoded.maximum_skew_ns,
            frames: decoded
                .frames
                .into_iter()
                .map(|frame| (frame.metadata.sensor_name.clone(), frame))
                .collect(),
        };
        assert_eq!(
            encode_frame_set(&again, decoded.envelope).unwrap(),
            PYTHON_WIRE_GOLDEN
        );
    }

    #[test]
    fn second_publisher_cannot_replace_live_owner() {
        let directory = tempfile::tempdir().unwrap();
        let socket = directory.path().join("frames.sock");
        let owner = UnixFramePublisher::bind(&socket).unwrap();
        assert!(
            UnixFramePublisher::bind(&socket)
                .unwrap_err()
                .to_string()
                .contains("busy")
        );
        assert!(UnixFrameClient::connect(&socket).is_ok());
        drop(owner);
        assert!(UnixFramePublisher::bind(&socket).is_ok());
    }

    #[test]
    fn client_waits_through_an_ownerless_socket_for_a_late_binding_owner() {
        use std::sync::{
            Arc,
            atomic::{AtomicBool, Ordering},
        };

        let directory = tempfile::tempdir().unwrap();
        let socket = directory.path().join("frames.sock");
        // A killed owner leaves its socket file behind, so the subscriber
        // sees ConnectionRefused first and ENOENT once the new owner
        // removes it. Both mean "not up yet" and both must be waited out.
        drop(UnixListener::bind(&socket).unwrap());

        let done = Arc::new(AtomicBool::new(false));
        let path = socket.clone();
        let flag = done.clone();
        let owner = std::thread::spawn(move || {
            std::thread::sleep(Duration::from_millis(200));
            let publisher = UnixFramePublisher::bind(&path).unwrap();
            while !flag.load(Ordering::Relaxed) {
                std::thread::sleep(Duration::from_millis(5));
            }
            drop(publisher);
        });

        let started = Instant::now();
        let client = UnixFrameClient::connect_within(&socket, Duration::from_secs(10))
            .expect("a subscriber must wait through the owner's bind");
        assert!(started.elapsed() >= Duration::from_millis(200));
        drop(client);
        done.store(true, Ordering::Relaxed);
        owner.join().unwrap();
    }

    #[test]
    fn client_retry_is_bounded_when_no_owner_appears() {
        let directory = tempfile::tempdir().unwrap();
        let socket = directory.path().join("frames.sock");
        let started = Instant::now();
        let error =
            UnixFrameClient::connect_within(&socket, Duration::from_millis(300)).unwrap_err();
        let elapsed = started.elapsed();
        assert!(
            elapsed >= Duration::from_millis(300),
            "gave up early: {elapsed:?}"
        );
        assert!(
            elapsed < Duration::from_secs(5),
            "overshot the window: {elapsed:?}"
        );
        assert!(
            error.to_string().contains("had no owner within"),
            "{error:#}"
        );
    }

    #[test]
    fn client_retry_refuses_a_path_that_is_not_a_socket() {
        let directory = tempfile::tempdir().unwrap();
        let path = directory.path().join("frames.sock");
        fs::write(&path, b"not a socket").unwrap();
        // connect(2) reports a regular file with the same ConnectionRefused
        // a not-yet-bound owner produces; only the file type separates a
        // misconfigured --socket from a startup race.
        let started = Instant::now();
        let error = UnixFrameClient::connect_within(&path, Duration::from_secs(30)).unwrap_err();
        assert!(
            started.elapsed() < Duration::from_secs(1),
            "retried a misconfigured path"
        );
        assert!(error.to_string().contains("not a socket"), "{error:#}");
    }

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

    #[test]
    fn idle_timeout_preserves_framing_but_partial_packet_and_eof_fail() {
        let (stream, mut sender) = UnixStream::pair().unwrap();
        let mut client = UnixFrameClient { stream };
        client.set_read_timeout(Duration::from_millis(20)).unwrap();
        assert!(client.recv_if_ready().unwrap().is_none());
        let set = SynchronizedFrameSet {
            sequence: 42,
            timestamp_basis: "normalized_unix_ns".into(),
            timestamp_ns: 3_000,
            maximum_skew_ns: 0,
            frames: BTreeMap::new(),
        };
        sender
            .write_all(&encode_frame_set(&set, None).unwrap())
            .unwrap();
        assert_eq!(client.recv_if_ready().unwrap().unwrap().sequence, 42);
        drop(sender);
        assert!(client.recv_if_ready().is_err());

        let (stream, mut sender) = UnixStream::pair().unwrap();
        let mut client = UnixFrameClient { stream };
        client.set_read_timeout(Duration::from_millis(20)).unwrap();
        sender.write_all(&[0]).unwrap();
        assert!(client.recv_if_ready().is_err());
    }
}
