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

/// Full-resolution detector luma, bit-identical to the RGB/BGR detector's
/// integer conversion. Color sources and their timestamps remain untouched.
pub fn luma_video_set(set: &SynchronizedFrameSet) -> Result<SynchronizedFrameSet> {
    let convert = |(name, frame): (&String, &FrameRecord)| -> Result<(String, FrameRecord)> {
        Ok((name.clone(), luma_frame(frame)?))
    };
    #[cfg(any(feature = "fiducials", feature = "rerun"))]
    let frames = {
        use rayon::prelude::*;
        set.frames.par_iter().map(convert).collect::<Result<_>>()?
    };
    #[cfg(not(any(feature = "fiducials", feature = "rerun")))]
    let frames = set.frames.iter().map(convert).collect::<Result<_>>()?;
    Ok(SynchronizedFrameSet {
        sequence: set.sequence,
        timestamp_basis: set.timestamp_basis.clone(),
        timestamp_ns: set.timestamp_ns,
        maximum_skew_ns: set.maximum_skew_ns,
        frames,
    })
}

fn transport_wall_ns() -> u128 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_nanos()
}

fn luma_frame(frame: &FrameRecord) -> Result<FrameRecord> {
    let started_unix_ns = transport_wall_ns();
    let started = std::time::Instant::now();
    let RecordedPayload::Video {
        format,
        width,
        height,
        bytes,
    } = &frame.payload
    else {
        anyhow::bail!("luma transport requires decoded video");
    };
    let channels = match format {
        PixelFormat::Y8 => 1,
        PixelFormat::Rgb8 | PixelFormat::Bgr8 => 3,
        _ => anyhow::bail!("luma transport requires RGB/BGR/Y8"),
    };
    anyhow::ensure!(
        u64::from(*width) * u64::from(*height) * channels == bytes.len() as u64,
        "invalid luma source size"
    );
    let luma = if channels == 1 {
        bytes.clone()
    } else if *format == PixelFormat::Bgr8 {
        color_luma::<true>(bytes)
    } else {
        color_luma::<false>(bytes)
    };
    let mut metadata = frame.metadata.clone();
    metadata.profile.format = PixelFormat::Y8;
    metadata.attributes.insert(
        "transport_luma_started_unix_ns".into(),
        started_unix_ns.to_string(),
    );
    metadata.attributes.insert(
        "transport_luma_finished_unix_ns".into(),
        transport_wall_ns().to_string(),
    );
    metadata
        .attributes
        .insert("transport_source_format".into(), format!("{format:?}"));
    metadata
        .attributes
        .insert("stride_bytes".into(), width.to_string());
    metadata
        .attributes
        .insert("bits_per_pixel".into(), "8".into());
    metadata.attributes.insert(
        "transport_luma_conversion_ns".into(),
        started.elapsed().as_nanos().to_string(),
    );
    Ok(FrameRecord {
        metadata,
        payload: RecordedPayload::Video {
            format: PixelFormat::Y8,
            width: *width,
            height: *height,
            bytes: luma,
        },
    })
}

fn color_luma<const BGR: bool>(bytes: &[u8]) -> Vec<u8> {
    let mut output = vec![0; bytes.len() / 3];
    for (pixel, gray) in bytes.chunks_exact(3).zip(&mut output) {
        let (r, b) = if BGR {
            (pixel[2], pixel[0])
        } else {
            (pixel[0], pixel[2])
        };
        // The largest weighted sum is 255 * 256 = 65280. Narrow
        // arithmetic preserves the detector's exact rounding while allowing
        // more pixels per vector; channel order is resolved per format.
        *gray = ((77 * u16::from(r) + 150 * u16::from(pixel[1]) + 29 * u16::from(b)) >> 8) as u8;
    }
    output
}

/// Latest-only luma/socket worker. Expensive conversion and slow socket writes
/// cannot block camera synchronization; only one unprocessed set is retained.
pub struct LumaFramePublisher {
    pending: std::sync::Arc<std::sync::Mutex<Option<std::sync::Arc<SynchronizedFrameSet>>>>,
    stopped: std::sync::Arc<std::sync::atomic::AtomicBool>,
    error: std::sync::Arc<std::sync::Mutex<Option<String>>>,
    worker: Option<std::thread::JoinHandle<()>>,
}
impl LumaFramePublisher {
    pub fn spawn(mut publisher: UnixFramePublisher) -> Self {
        use std::sync::{
            Arc, Mutex,
            atomic::{AtomicBool, Ordering},
        };
        let pending = Arc::new(Mutex::new(None::<Arc<SynchronizedFrameSet>>));
        let incoming = pending.clone();
        let stopped = Arc::new(AtomicBool::new(false));
        let stop = stopped.clone();
        let error = Arc::new(Mutex::new(None));
        let failure = error.clone();
        let worker = std::thread::spawn(move || {
            while !stop.load(Ordering::Acquire) {
                let set = incoming.lock().unwrap().take();
                let Some(set) = set else {
                    std::thread::sleep(Duration::from_millis(2));
                    continue;
                };
                // Accept first: `publish` is otherwise the only path that takes
                // new clients, so gating on the count alone would close the
                // socket to every future subscriber. The conversion is the
                // expensive half, and an unsubscribed set is dropped unconverted.
                if let Err(e) = publisher.poll_accept() {
                    *failure.lock().unwrap() = Some(e.to_string());
                    break;
                }
                if publisher.client_count() == 0 {
                    continue;
                }
                if let Err(e) = luma_video_set(&set).and_then(|gray| publisher.publish(&gray)) {
                    *failure.lock().unwrap() = Some(e.to_string());
                    break;
                }
            }
        });
        Self {
            pending,
            stopped,
            error,
            worker: Some(worker),
        }
    }
    pub fn submit(&self, set: &SynchronizedFrameSet) -> Result<()> {
        self.submit_shared(std::sync::Arc::new(set.clone()))
    }

    pub fn submit_shared(&self, set: std::sync::Arc<SynchronizedFrameSet>) -> Result<()> {
        if let Some(error) = self.error.lock().unwrap().as_ref() {
            anyhow::bail!("luma publisher: {error}");
        }
        let previous = self.pending.lock().unwrap().replace(set);
        drop(previous);
        Ok(())
    }
}
impl Drop for LumaFramePublisher {
    fn drop(&mut self) {
        self.stopped
            .store(true, std::sync::atomic::Ordering::Release);
        if let Some(worker) = self.worker.take() {
            let _ = worker.join();
        }
    }
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{FrameTimestamps, SensorKind, StreamProfile, TimestampDomain};
    use std::collections::BTreeMap;

    #[test]
    fn color_luma_preserves_integer_detector_rounding_and_channel_order() {
        let mut rgb = Vec::new();
        let mut expected = Vec::new();
        for r in 0..=255_u32 {
            for g in 0..=255_u32 {
                for b in [0, 1, 127, 254, 255] {
                    rgb.extend_from_slice(&[r as u8, g as u8, b as u8]);
                    expected.push(((77 * r + 150 * g + 29 * b) >> 8) as u8);
                }
            }
        }
        assert_eq!(color_luma::<false>(&rgb), expected);
        for pixel in rgb.chunks_exact_mut(3) {
            pixel.swap(0, 2);
        }
        assert_eq!(color_luma::<true>(&rgb), expected);
        assert!(color_luma::<true>(&[]).is_empty());
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
        let luma = luma_video_set(&set).unwrap();
        assert_eq!(luma.timestamp_ns, set.timestamp_ns);
        assert_eq!(
            luma.frames["camera1"].metadata.profile.format,
            PixelFormat::Y8
        );
        assert_eq!(publisher.publish(&luma).unwrap(), 1);
        let gray = client.recv().unwrap();
        assert_eq!(
            gray.frames[0].payload,
            RecordedPayload::Video {
                format: PixelFormat::Y8,
                width: 2,
                height: 1,
                bytes: vec![2, 5],
            }
        );
        assert_eq!(received.frames[0].payload, set.frames["camera1"].payload);
        let background = LumaFramePublisher::spawn(publisher);
        background
            .submit_shared(std::sync::Arc::new(set.clone()))
            .unwrap();
        client.set_read_timeout(Duration::from_secs(1)).unwrap();
        assert_eq!(
            client.recv().unwrap().frames[0].payload,
            gray.frames[0].payload
        );
        drop(background);
        assert!(client.recv().is_err());
    }

    fn luma_gate_frame_set(payload: RecordedPayload) -> SynchronizedFrameSet {
        let frame = FrameRecord {
            metadata: FrameMetadata {
                sensor_name: "camera1".into(),
                sensor_kind: SensorKind::PoE,
                sequence: 1,
                profile: StreamProfile {
                    stream: "main".into(),
                    width: 2,
                    height: 1,
                    fps_num: 20,
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
                flags: Vec::new(),
                attributes: BTreeMap::new(),
            },
            payload,
        };
        SynchronizedFrameSet {
            sequence: 1,
            timestamp_basis: "normalized_unix_ns".into(),
            timestamp_ns: 3_000,
            maximum_skew_ns: 0,
            frames: BTreeMap::from([(String::from("camera1"), frame)]),
        }
    }

    #[test]
    fn luma_worker_skips_conversion_while_unsubscribed() {
        let directory = tempfile::tempdir().unwrap();
        let socket = directory.path().join("frames.sock");
        let publisher = UnixFramePublisher::bind(&socket).unwrap();
        let background = LumaFramePublisher::spawn(publisher);

        // An encoded payload cannot be converted to luma. With nobody
        // subscribed the worker must drop each set before attempting the
        // conversion, so it never records a failure. Without the gate the
        // first set poisons the publisher and every later submit fails.
        // No client ever attaches here: converting an encoded set for a real
        // subscriber is a genuine error, not the behaviour under test.
        let encoded = luma_gate_frame_set(RecordedPayload::Encoded {
            format: PixelFormat::H264,
            bytes: vec![1, 2, 3],
        });
        for _ in 0..10 {
            background
                .submit(&encoded)
                .expect("unsubscribed sets must not be converted");
            std::thread::sleep(Duration::from_millis(5));
        }
    }

    #[test]
    fn luma_worker_accepts_subscribers_that_attach_after_it_starts() {
        let directory = tempfile::tempdir().unwrap();
        let socket = directory.path().join("frames.sock");
        let publisher = UnixFramePublisher::bind(&socket).unwrap();
        let background = LumaFramePublisher::spawn(publisher);

        // `publish` is otherwise the only path that accepts, so a worker that
        // gated on the client count alone would never see this subscriber.
        let mut client = UnixFrameClient::connect(&socket).unwrap();
        client.set_read_timeout(Duration::from_secs(5)).unwrap();
        let decoded = luma_gate_frame_set(RecordedPayload::Video {
            format: PixelFormat::Bgr8,
            width: 2,
            height: 1,
            bytes: vec![4, 5, 6, 7, 8, 9],
        });
        background.submit(&decoded).unwrap();
        assert_eq!(
            client.recv().unwrap().frames[0].payload,
            RecordedPayload::Video {
                format: PixelFormat::Y8,
                width: 2,
                height: 1,
                bytes: vec![5, 8],
            }
        );
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
