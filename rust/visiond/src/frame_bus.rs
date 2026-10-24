//! Binary Zenoh frame sets: bounded JSON envelope + JPEG color / exact Z16.
//! Only camera owners call publish. Consumers never instantiate a backend.
use crate::{
    FrameRecord, PixelFormat, RecordedPayload, SynchronizedFrameSet,
    capture_clock::{CaptureClockRequest, CaptureClockSample, CaptureFreshSelection, fresh_tags},
};
use anyhow::{Result, ensure};
use serde::{Deserialize, Serialize};
use std::borrow::Cow;
#[cfg(feature = "rerun")]
use sha2::{Digest, Sha256};
use tatbot_bus::{Envelope, Producer, Stamp};
const LIMIT: usize = 128 * 1024 * 1024;
pub fn encode(set: &SynchronizedFrameSet, producer: &Producer) -> Result<Vec<u8>> {
    encode_inner(set, producer, None)
}

pub fn encode_clocked_capture(
    set: &SynchronizedFrameSet,
    producer: &Producer,
    sample: &CaptureClockSample,
) -> Result<Vec<u8>> {
    let request = CaptureClockRequest::new(sample.request_nonce.clone())?;
    sample.validate(&request, producer, set)?;
    encode_inner(set, producer, Some(sample))
}

fn encode_inner(
    set: &SynchronizedFrameSet,
    producer: &Producer,
    capture_clock: Option<&CaptureClockSample>,
) -> Result<Vec<u8>> {
    ensure!(
        !set.frames.is_empty() && set.frames.len() <= 16,
        "frame count limit"
    );
    let mut compressed = SynchronizedFrameSet {
        sequence: set.sequence,
        timestamp_basis: set.timestamp_basis.clone(),
        timestamp_ns: set.timestamp_ns,
        maximum_skew_ns: set.maximum_skew_ns,
        frames: Default::default(),
    };
    let mut total = 0_usize;
    for frame in set.frames.values() {
        frame.validate().map_err(anyhow::Error::msg)?;
        let mut metadata = frame.metadata.clone();
        let payload = match &frame.payload {
            RecordedPayload::Video {
                format,
                width,
                height,
                bytes,
            } => {
                ensure!(
                    *width == frame.metadata.profile.width
                        && *height == frame.metadata.profile.height,
                    "frame dimensions differ from metadata"
                );
                ensure!(
                    matches!(
                        format,
                        PixelFormat::Rgb8 | PixelFormat::Bgr8 | PixelFormat::Yuyv | PixelFormat::Y8
                    ),
                    "only decoded RGB/BGR/Y8 can be published"
                );
                // A detector-only pipeline decodes straight to single-channel
                // luma, so the expected byte count follows the format instead
                // of assuming three channels.
                let channels: u64 = match format {
                    PixelFormat::Yuyv => 2,
                    PixelFormat::Y8 => 1,
                    _ => 3,
                };
                ensure!(
                    u64::from(*width) * u64::from(*height) * channels == bytes.len() as u64
                        && bytes.len() <= LIMIT,
                    "color dimensions invalid"
                );
                // The encoder only reads the pixels: Y8 and RGB8 go in as they
                // are, and only a conversion allocates.
                let (pixels, color_type) = if *format == PixelFormat::Y8 {
                    // JPEG carries luma natively. Expanding to RGB here would
                    // undo the saving the Y8 decode exists for.
                    (Cow::Borrowed(bytes.as_slice()), image::ExtendedColorType::L8)
                } else if *format == PixelFormat::Yuyv {
                    ensure!(*width % 2 == 0, "YUYV width must be even");
                    metadata.profile.format = PixelFormat::Rgb8;
                    metadata
                        .attributes
                        .insert("bus_source_format".into(), "yuyv".into());
                    metadata
                        .attributes
                        .insert("stride_bytes".into(), (width * 3).to_string());
                    metadata
                        .attributes
                        .insert("bits_per_pixel".into(), "24".into());
                    (Cow::Owned(crate::frame_ops::yuyv_to_rgb(bytes)), image::ExtendedColorType::Rgb8)
                } else if *format == PixelFormat::Bgr8 {
                    let mut rgb = bytes.clone();
                    for pixel in rgb.chunks_exact_mut(3) {
                        pixel.swap(0, 2);
                    }
                    (Cow::Owned(rgb), image::ExtendedColorType::Rgb8)
                } else {
                    (Cow::Borrowed(bytes.as_slice()), image::ExtendedColorType::Rgb8)
                };
                let mut jpeg = Vec::new();
                image::codecs::jpeg::JpegEncoder::new_with_quality(&mut jpeg, 95)
                    .encode(&pixels, *width, *height, color_type)?;
                RecordedPayload::Encoded {
                    format: PixelFormat::Jpeg,
                    bytes: jpeg,
                }
            }
            RecordedPayload::Depth {
                width,
                height,
                bytes,
            } => {
                ensure!(
                    *width == frame.metadata.profile.width
                        && *height == frame.metadata.profile.height
                        && u64::from(*width) * u64::from(*height) * 2 == bytes.len() as u64,
                    "depth dimensions invalid"
                );
                RecordedPayload::Depth {
                    width: *width,
                    height: *height,
                    bytes: bytes.clone(),
                }
            }
            _ => anyhow::bail!("camera owner must decode before Zenoh publication"),
        };
        total = total
            .checked_add(payload.bytes().len())
            .ok_or_else(|| anyhow::anyhow!("payload overflow"))?;
        ensure!(total <= LIMIT, "frame set exceeds limit");
        compressed.frames.insert(
            frame.metadata.sensor_name.clone(),
            FrameRecord { metadata, payload },
        );
    }
    let mut payload = serde_json::json!({"wire_version": 1});
    if let Some(sample) = capture_clock {
        payload["capture_clock"] = serde_json::to_value(sample)?;
    }
    let envelope = Envelope {
        schema: "tatbot.frame-set/1".into(),
        producer: producer.clone(),
        stamp: Stamp {
            mono_ns: 0,
            wall_ns: u64::try_from(set.timestamp_ns)?,
            basis: set.timestamp_basis.clone(),
        },
        seq: set.sequence,
        payload,
    };
    crate::transport::encode_frame_set(&compressed, Some(serde_json::to_value(envelope)?))
}
pub fn decode(bytes: &[u8]) -> Result<(Producer, SynchronizedFrameSet)> {
    let (producer, set, _) = decode_inner(bytes)?;
    Ok((producer, set))
}

pub fn decode_clocked_capture(
    bytes: &[u8],
) -> Result<(Producer, SynchronizedFrameSet, CaptureClockSample)> {
    let (producer, set, sample) = decode_inner(bytes)?;
    Ok((
        producer,
        set,
        sample.ok_or_else(|| anyhow::anyhow!("overhead capture clock sample missing"))?,
    ))
}

fn decode_inner(
    bytes: &[u8],
) -> Result<(Producer, SynchronizedFrameSet, Option<CaptureClockSample>)> {
    let h = crate::transport::decode_frame_set(bytes)?;
    let envelope = bus_envelope(
        h.envelope
            .clone()
            .ok_or_else(|| anyhow::anyhow!("missing bus provenance"))?,
        h.sequence,
    )?;
    let capture_clock = serde_json::from_value(envelope.payload["capture_clock"].clone())?;
    Ok((envelope.producer, decoded_set(h, false)?, capture_clock))
}

fn bus_envelope(value: serde_json::Value, sequence: u64) -> Result<Envelope<serde_json::Value>> {
    let envelope: Envelope<serde_json::Value> = serde_json::from_value(value)?;
    ensure!(
        envelope.schema == "tatbot.frame-set/1"
            && envelope.seq == sequence
            && envelope.payload["wire_version"] == 1,
        "frame schema or sequence mismatch"
    );
    Ok(envelope)
}

/// A frame set as the viewer shows it: every frame checked as strictly as a
/// consumer's, the owner's JPEG bytes kept rather than expanded to raw RGB.
/// The bus envelope is checked when present; the wrist tap's sets may carry
/// none, and display needs no provenance beyond the frame metadata.
#[cfg(any(feature = "rerun", test))]
fn decode_view(bytes: &[u8]) -> Result<SynchronizedFrameSet> {
    let h = crate::transport::decode_frame_set(bytes)?;
    if let Some(envelope) = h.envelope.clone() {
        bus_envelope(envelope, h.sequence)?;
    }
    decoded_set(h, true)
}

fn decoded_set(
    h: crate::transport::ReceivedFrameSet,
    retain_jpeg: bool,
) -> Result<SynchronizedFrameSet> {
    ensure!(!h.frames.is_empty(), "empty frame set");
    let mut frames = std::collections::BTreeMap::new();
    let mut decoded_bytes = 0_u64;
    for d in h.frames {
        let width = d.metadata.profile.width;
        let height = d.metadata.profile.height;
        ensure!(
            u64::from(width) * u64::from(height) * 3 <= LIMIT as u64,
            "decoded dimensions limit"
        );
        let decoded_channels: u64 = if matches!(d.payload, RecordedPayload::Depth { .. }) {
            2
        } else if d.metadata.profile.format == PixelFormat::Y8 {
            1
        } else {
            3
        };
        decoded_bytes += u64::from(width) * u64::from(height) * decoded_channels;
        ensure!(
            decoded_bytes <= LIMIT as u64,
            "decoded frame set exceeds limit"
        );
        let payload = match d.payload {
            RecordedPayload::Encoded {
                format: PixelFormat::Jpeg,
                bytes,
            } => {
                ensure!(
                    matches!(
                        d.metadata.profile.format,
                        PixelFormat::Rgb8 | PixelFormat::Bgr8 | PixelFormat::Y8
                    ),
                    "JPEG metadata format invalid"
                );
                let dimensions = image::ImageReader::with_format(
                    std::io::Cursor::new(&bytes),
                    image::ImageFormat::Jpeg,
                )
                .into_dimensions()?;
                ensure!(
                    dimensions == (width, height),
                    "encoded JPEG dimensions differ"
                );
                if retain_jpeg {
                    RecordedPayload::Encoded {
                        format: PixelFormat::Jpeg,
                        bytes,
                    }
                } else {
                    let reader = image::ImageReader::with_format(
                        std::io::Cursor::new(&bytes),
                        image::ImageFormat::Jpeg,
                    );
                    let raw = if d.metadata.profile.format == PixelFormat::Y8 {
                        let luma = reader.decode()?.into_luma8();
                        ensure!(
                            luma.dimensions() == (width, height),
                            "decoded JPEG dimensions differ"
                        );
                        luma.into_raw()
                    } else {
                        let rgb = reader.decode()?.into_rgb8();
                        ensure!(
                            rgb.dimensions() == (width, height),
                            "decoded JPEG dimensions differ"
                        );
                        let mut raw = rgb.into_raw();
                        if d.metadata.profile.format == PixelFormat::Bgr8 {
                            for p in raw.chunks_exact_mut(3) {
                                p.swap(0, 2);
                            }
                        }
                        raw
                    };
                    RecordedPayload::Video {
                        format: d.metadata.profile.format,
                        width,
                        height,
                        bytes: raw,
                    }
                }
            }
            RecordedPayload::Depth {
                width: encoded_width,
                height: encoded_height,
                bytes,
            } => {
                ensure!(
                    (encoded_width, encoded_height) == (width, height),
                    "depth dimensions differ"
                );
                ensure!(
                    d.metadata.profile.format == PixelFormat::Z16,
                    "depth metadata format invalid"
                );
                ensure!(
                    u64::from(width) * u64::from(height) * 2 == bytes.len() as u64,
                    "depth byte count"
                );
                RecordedPayload::Depth {
                    width,
                    height,
                    bytes,
                }
            }
            _ => anyhow::bail!("unknown frame codec"),
        };
        let name = d.metadata.sensor_name.clone();
        let frame = FrameRecord {
            metadata: d.metadata,
            payload,
        };
        frame.validate().map_err(anyhow::Error::msg)?;
        ensure!(
            frames.insert(name, frame).is_none(),
            "duplicate camera in frame set"
        );
    }
    Ok(SynchronizedFrameSet {
        sequence: h.sequence,
        timestamp_basis: h.timestamp_basis,
        timestamp_ns: h.timestamp_ns,
        maximum_skew_ns: h.maximum_skew_ns,
        frames,
    })
}

/// One latest-value slot between capture and JPEG/network work. A slow bus
/// replaces queued input, never creates an unbounded capture backlog.
pub struct FramePublisher {
    dropped: std::sync::Arc<std::sync::atomic::AtomicU64>,
    pending: std::sync::Arc<std::sync::Mutex<Option<std::sync::Arc<SynchronizedFrameSet>>>>,
    stopped: std::sync::Arc<std::sync::atomic::AtomicBool>,
    error: std::sync::Arc<std::sync::Mutex<Option<String>>>,
    worker: Option<std::thread::JoinHandle<()>>,
    latest: std::sync::Arc<
        std::sync::Mutex<std::collections::VecDeque<std::sync::Arc<SynchronizedFrameSet>>>,
    >,
    fresh_generation: Option<std::sync::Arc<std::sync::atomic::AtomicU64>>,
    history_capacity: usize,
    _capture_query: Option<zenoh::query::Queryable<()>>,
    _lease: tatbot_bus::service::ServiceLease,
    /// The owner's own recovery records ride its capture topic; see
    /// `CaptureHealth`. Sequenced apart from the frame sets.
    health: HealthChannel,
}
struct HealthChannel {
    session: zenoh::Session,
    producer: Producer,
    group: String,
    seq: std::sync::atomic::AtomicU64,
    started: std::time::Instant,
}
type FrameHistory = std::sync::Arc<
    std::sync::Mutex<std::collections::VecDeque<std::sync::Arc<SynchronizedFrameSet>>>,
>;

fn retain_history(
    history: &FrameHistory,
    set: std::sync::Arc<SynchronizedFrameSet>,
    capacity: usize,
) {
    let mut history = history.lock().unwrap();
    if history.len() >= capacity {
        history.pop_front();
    }
    history.push_back(set);
    while history.len() > 1
        && history
            .iter()
            .flat_map(|set| set.frames.values())
            .map(|frame| frame.payload.bytes().len())
            .sum::<usize>()
            > LIMIT
    {
        history.pop_front();
    }
}

fn select_after_sdk_drain(
    history: &FrameHistory,
    generation: u64,
    previous_set_sequence: Option<u64>,
    limit: std::time::Duration,
) -> Result<(std::sync::Arc<SynchronizedFrameSet>, CaptureFreshSelection)> {
    let started = std::time::Instant::now();
    while started.elapsed() < limit {
        let candidate = history.lock().unwrap().back().cloned();
        if let Some(set) = candidate {
            if let Some((tagged_generation, _)) = fresh_tags(&set)? {
                if tagged_generation == generation {
                    let selection = CaptureFreshSelection::from_set(&set, previous_set_sequence)?;
                    validate_capture_geometry(&set)?;
                    return Ok((set, selection));
                }
            }
        }
        std::thread::sleep(std::time::Duration::from_millis(2));
    }
    anyhow::bail!("no post-request RGBD set after owner SDK queue drain")
}

fn capture_topic(group: &str, node: &str) -> String {
    // Wrist USB ownership may span hosts. An unqualified query could receive
    // either arm's frames; require the caller to select its capture owner.
    if group == "d405" {
        format!("tatbot/vision/{group}/capture/{node}")
    } else {
        format!("tatbot/vision/{group}/capture")
    }
}

struct PendingCaptureQuery {
    query: zenoh::query::Query,
    owner_received_ns: u64,
    owner_received_at: std::time::Instant,
}

fn host_wall_ns() -> Result<u64> {
    Ok(u64::try_from(
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)?
            .as_nanos(),
    )?)
}

fn capture_clock_request(
    group: &str,
    payload: Option<&[u8]>,
) -> Result<Option<CaptureClockRequest>> {
    let Some(bytes) = payload.filter(|_| group == "overhead-depth") else {
        return Ok(None);
    };
    ensure!(bytes.len() <= 1024, "overhead capture query exceeds budget");
    let value: serde_json::Value = serde_json::from_slice(bytes)?;
    if value.get("schema").is_none() {
        return Ok(None);
    }
    let request: CaptureClockRequest = serde_json::from_value(value)?;
    request.validate()?;
    Ok(Some(request))
}

fn encode_capture_reply(
    set: &SynchronizedFrameSet,
    producer: &Producer,
    request: Option<&CaptureClockRequest>,
    selection: Option<CaptureFreshSelection>,
    pending: &PendingCaptureQuery,
) -> Result<Vec<u8>> {
    let sample = request
        .map(|request| {
            CaptureClockSample::for_response(
                request,
                producer,
                set,
                pending.owner_received_ns,
                host_wall_ns()?,
                u64::try_from(pending.owner_received_at.elapsed().as_nanos())?,
                selection,
            )
        })
        .transpose()?;
    if let Some(sample) = sample.as_ref() {
        encode_clocked_capture(set, producer, sample)
    } else {
        encode(set, producer)
    }
}

pub const CAPTURE_HEALTH_SCHEMA: &str = "tatbot.capture-health/1";

/// A capture owner's own recovery, published on the capture topic its
/// consumers already address so whoever waits on that owner learns why its
/// frames paused and what the owner did about it. The one `event` today is
/// `reset`: the owner saw no frame for its safety timeout and put the device
/// through a hardware reset. `last_error` is the newest error the worker had
/// seen (a USB storm shows here); `empty_results` counts the consecutive
/// waits that carried no frameset; `outcome` says whether the reset was
/// issued or why not.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct CaptureHealth {
    pub event: String,
    pub group: String,
    pub sensor: String,
    pub serial: String,
    pub reason: String,
    pub last_error: Option<String>,
    pub empty_results: u64,
    pub outcome: String,
}

fn health_record(
    group: &str,
    producer: &Producer,
    seq: u64,
    mono_ns: u64,
    payload: CaptureHealth,
) -> (String, Envelope<CaptureHealth>) {
    let wall_ns = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_nanos() as u64)
        .unwrap_or(0);
    (
        capture_topic(group, &producer.node),
        Envelope {
            schema: CAPTURE_HEALTH_SCHEMA.into(),
            producer: producer.clone(),
            stamp: Stamp {
                mono_ns,
                wall_ns,
                basis: "host".into(),
            },
            seq,
            payload,
        },
    )
}

impl FramePublisher {
    pub fn open(connect: &[String], producer: Producer, group: &str) -> Result<Self> {
        use zenoh::Wait;
        ensure!(
            ["d405", "overhead-depth"].contains(&group),
            "unknown camera group"
        );
        let bus =
            tatbot_bus::transport::Bus::open(connect).map_err(|e| anyhow::anyhow!("{e}"))?;
        let lease = tatbot_bus::service::ServiceLease::declare(
            &bus,
            producer.clone(),
            &format!("visiond-{group}"),
            if group == "overhead-depth" {
                vec![
                    "tatbot.frame-set/1".into(),
                    crate::capture_clock::SAMPLE_SCHEMA.into(),
                ]
            } else {
                vec!["tatbot.frame-set/1".into()]
            },
        )
        .map_err(|e| anyhow::anyhow!("{e}"))?;
        let dropped = std::sync::Arc::new(std::sync::atomic::AtomicU64::new(0));
        let dropped_count = dropped.clone();
        let metrics = lease.metrics.clone();
        let pending = std::sync::Arc::new(std::sync::Mutex::new(
            None::<std::sync::Arc<SynchronizedFrameSet>>,
        ));
        let incoming = pending.clone();
        let stopped = std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false));
        let stop = stopped.clone();
        let error = std::sync::Arc::new(std::sync::Mutex::new(None));
        let failure = error.clone();
        let prefix = format!("tatbot/vision/{group}");
        let publish_depth = group == "overhead-depth";
        let wrist_depth = group == "d405";
        let latest = std::sync::Arc::new(std::sync::Mutex::new(std::collections::VecDeque::<
            std::sync::Arc<SynchronizedFrameSet>,
        >::new()));
        let capture_latest = latest.clone();
        let fresh_generation = publish_depth.then(|| {
            std::sync::Arc::new(std::sync::atomic::AtomicU64::new(0))
        });
        let requested_fresh_generation = fresh_generation.clone();
        let (requests, capture_requests) = std::sync::mpsc::sync_channel::<PendingCaptureQuery>(1);
        let capture_topic = capture_topic(group, &producer.node);
        let reply_topic = capture_topic.clone();
        let capture_group = group.to_owned();
        let health = HealthChannel {
            session: bus.session.clone(),
            producer: producer.clone(),
            group: group.to_string(),
            seq: std::sync::atomic::AtomicU64::new(0),
            started: std::time::Instant::now(),
        };
        let capture_query = Some(
            bus.session
                .declare_queryable(capture_topic.clone())
                .callback(move |query| {
                    let pending = PendingCaptureQuery {
                        query,
                        owner_received_ns: host_wall_ns().unwrap_or(0),
                        owner_received_at: std::time::Instant::now(),
                    };
                    if let Err(
                        std::sync::mpsc::TrySendError::Full(pending)
                        | std::sync::mpsc::TrySendError::Disconnected(pending),
                    ) = requests.try_send(pending)
                    {
                        let _ = pending.query.reply_err("capture busy or offline").wait();
                    }
                })
                .wait()
                .map_err(|e| anyhow::anyhow!("{e}"))?,
        );
        let worker = std::thread::spawn(move || {
            let mut window = std::time::Instant::now();
            let mut published = 0_u64;
            // Sets skipped whole because nothing was subscribed. Reported so
            // an idle publisher is visibly idle rather than indistinguishable
            // from a stalled one.
            let mut gated = 0_u64;
            // One declared publisher per camera key, kept for the worker's
            // life so its matching status can be consulted before any work.
            let mut publishers: std::collections::HashMap<String, zenoh::pubsub::Publisher<'_>> =
                std::collections::HashMap::new();
            let mut durations = std::collections::VecDeque::new();
            let interval = std::time::Duration::from_secs_f64(if publish_depth {
                0.2
            } else {
                1.0 / 30.0
            });
            let mut last_put = std::time::Instant::now() - interval;
            let mut last_depth = std::collections::BTreeMap::<String, std::time::Instant>::new();
            while !stop.load(std::sync::atomic::Ordering::Acquire) {
                if let Ok(pending) = capture_requests.try_recv() {
                    let payload = pending.query.payload().map(|v| v.to_bytes().into_owned());
                    let selected = capture_clock_request(&capture_group, payload.as_deref())
                        .and_then(|request| {
                            if request.is_some() {
                                let previous = {
                                    let mut history = capture_latest.lock().unwrap();
                                    let previous = history.back().map(|set| set.sequence);
                                    history.clear();
                                    previous
                                };
                                let generation = requested_fresh_generation
                                    .as_ref()
                                    .expect("overhead owner has a fresh generation")
                                    .fetch_add(1, std::sync::atomic::Ordering::AcqRel)
                                    .saturating_add(1);
                                let (set, selection) = select_after_sdk_drain(
                                    &capture_latest,
                                    generation,
                                    previous,
                                    std::time::Duration::from_millis(2200),
                                )?;
                                return Ok((set, request, Some(selection)));
                            }
                            let history = capture_latest.lock().unwrap();
                            let set = select_capture(&history, payload.as_deref())?;
                            Ok((set, request, None))
                        });
                    // JPEG encoding must not hold up the camera owner's history writer.
                    let result = selected.and_then(|(set, request, selection)| {
                        encode_capture_reply(&set, &producer, request.as_ref(), selection, &pending)
                    });
                    match result {
                        Ok(bytes) => {
                            let _ = pending.query.reply(reply_topic.clone(), bytes).wait();
                        }
                        Err(error) => {
                            let _ = pending.query.reply_err(error.to_string()).wait();
                        }
                    }
                }
                if last_put.elapsed() < interval {
                    std::thread::sleep(std::time::Duration::from_millis(2));
                    continue;
                }
                let value = incoming.lock().unwrap().take();
                if let Some(set) = value {
                    last_put = std::time::Instant::now();
                    let started = std::time::Instant::now();
                    let result = (|| -> Result<usize> {
                        let mut sent = 0_usize;
                        for (name, frame) in &set.frames {
                            // Wrist RGB keeps its observation cadence; depth is
                            // a separate 2 Hz preview, only encoded with a subscriber.
                            if !publish_depth
                                && matches!(frame.payload, RecordedPayload::Depth { .. })
                                && (!wrist_depth
                                    || last_depth.get(name).is_some_and(|last| {
                                        last.elapsed() < std::time::Duration::from_millis(500)
                                    }))
                            {
                                continue;
                            }
                            let key = format!("{prefix}/{name}");
                            if !publishers.contains_key(&key) {
                                let declared = bus
                                    .session
                                    .declare_publisher(key.clone())
                                    .congestion_control(zenoh::qos::CongestionControl::Drop)
                                    .wait()
                                    .map_err(|e| anyhow::anyhow!("{e}"))?;
                                publishers.insert(key.clone(), declared);
                            }
                            let publisher = &publishers[&key];
                            // Downscaling and JPEG encoding are the expensive
                            // half of publication, and the router discards the
                            // result when nothing is subscribed. Skip the work
                            // rather than the delivery.
                            if !publisher
                                .matching_status()
                                .wait()
                                .map_err(|e| anyhow::anyhow!("{e}"))?
                                .matching()
                            {
                                continue;
                            }
                            let frame = frame.clone();
                            let is_depth = matches!(frame.payload, RecordedPayload::Depth { .. });
                            let one = SynchronizedFrameSet {
                                sequence: set.sequence,
                                timestamp_basis: set.timestamp_basis.clone(),
                                timestamp_ns: set.timestamp_ns,
                                maximum_skew_ns: set.maximum_skew_ns,
                                frames: std::collections::BTreeMap::from([(name.clone(), frame)]),
                            };
                            let preview = if wrist_depth && is_depth {
                                // Scan requests keep their full-resolution samples.
                                // Smaller visualization packets reduce fragmentation
                                // alongside the RGB observation traffic.
                                crate::frame_ops::scale_depth_set(&one, 0.25, "bus_preview")
                            } else {
                                one
                            };
                            let bytes = encode(&preview, &producer)?;
                            publisher
                                .put(bytes)
                                .wait()
                                .map_err(|e| anyhow::anyhow!("{e}"))?;
                            sent += 1;
                            if is_depth {
                                last_depth.insert(name.clone(), std::time::Instant::now());
                            }
                        }
                        Ok(sent)
                    })();
                    let sent = match result {
                        Ok(sent) => sent,
                        Err(e) => {
                            *failure.lock().unwrap() = Some(e.to_string());
                            break;
                        }
                    };
                    // A set that reached no subscriber is not a publication.
                    // Counting it would report a healthy put rate and a fast
                    // encode while the gate was skipping every camera.
                    if sent > 0 {
                        published += 1;
                        durations.push_back(started.elapsed().as_secs_f64() * 1000.0);
                        if durations.len() > 100 {
                            durations.pop_front();
                        }
                    } else {
                        gated += 1;
                    }
                    if window.elapsed().as_secs() >= 1 {
                        let mut sorted = durations.iter().copied().collect::<Vec<_>>();
                        sorted.sort_by(f64::total_cmp);
                        *metrics.lock().unwrap() = std::collections::BTreeMap::from([
                            (
                                "put_fps_window".into(),
                                published as f64 / window.elapsed().as_secs_f64(),
                            ),
                            (
                                "dropped_before_encode".into(),
                                dropped_count.load(std::sync::atomic::Ordering::Relaxed) as f64,
                            ),
                            (
                                "gated_fps_window".into(),
                                gated as f64 / window.elapsed().as_secs_f64(),
                            ),
                            (
                                "encode_put_p95_ms".into(),
                                // Empty whenever the gate skipped the whole
                                // window; `len() - 1` would underflow.
                                sorted
                                    .get((sorted.len().max(1) - 1) * 95 / 100)
                                    .copied()
                                    .unwrap_or(0.0),
                            ),
                        ]);
                        window = std::time::Instant::now();
                        published = 0;
                        gated = 0;
                    }
                } else {
                    std::thread::sleep(std::time::Duration::from_millis(2));
                }
            }
        });
        Ok(Self {
            dropped,
            pending,
            stopped,
            error,
            worker: Some(worker),
            latest,
            fresh_generation,
            history_capacity: if publish_depth { 64 } else { 1 },
            _capture_query: capture_query,
            _lease: lease,
            health,
        })
    }
    /// A request generation for the sole owner-side RealSense worker. It is
    /// diagnostic provenance, never a permission to use exposure timestamps.
    pub fn fresh_generation(&self) -> Option<std::sync::Arc<std::sync::atomic::AtomicU64>> {
        self.fresh_generation.clone()
    }
    /// Publish one recovery record on this owner's capture topic. Never
    /// gated on a subscriber: the record is the journal of what the owner did.
    pub fn publish_health(&self, payload: CaptureHealth) -> Result<()> {
        use zenoh::Wait;
        let seq = self
            .health
            .seq
            .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        let (key, envelope) = health_record(
            &self.health.group,
            &self.health.producer,
            seq,
            self.health.started.elapsed().as_nanos() as u64,
            payload,
        );
        self.health
            .session
            .put(key, serde_json::to_vec(&envelope)?)
            .wait()
            .map_err(|e| anyhow::anyhow!("{e}"))
    }
    pub fn submit(&self, set: &SynchronizedFrameSet) -> Result<()> {
        self.submit_shared(std::sync::Arc::new(set.clone()))
    }

    pub fn submit_shared(&self, set: std::sync::Arc<SynchronizedFrameSet>) -> Result<()> {
        if let Some(error) = self.error.lock().unwrap().as_ref() {
            anyhow::bail!("frame publisher: {error}");
        }
        retain_history(&self.latest, set.clone(), self.history_capacity);
        if let Ok(mut pending) = self.pending.try_lock() {
            if pending.replace(set).is_some() {
                self.dropped
                    .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            }
        } else {
            self.dropped
                .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        }
        Ok(())
    }
}
impl Drop for FramePublisher {
    fn drop(&mut self) {
        self.stopped
            .store(true, std::sync::atomic::Ordering::Release);
        if let Some(worker) = self.worker.take() {
            let _ = worker.join();
        }
    }
}

#[cfg(feature = "rerun")]
#[derive(Clone, Copy)]
pub struct ViewBusScene<'a> {
    pub urdf: Option<&'a std::path::Path>,
    pub calibration: Option<&'a std::path::Path>,
    pub robot_world: Option<&'a std::path::Path>,
}

#[cfg(feature = "rerun")]
fn file_fingerprint(path: &std::path::Path) -> String {
    match std::fs::read(path) {
        Ok(bytes) => hex::encode(Sha256::digest(bytes)),
        Err(error) => format!("error:{:?}", error.kind()),
    }
}

#[cfg(feature = "rerun")]
fn refresh_calibration(
    viewer: &crate::RerunViewer,
    urdf: Option<&std::path::Path>,
    calibration_path: &std::path::Path,
    robot_world: &std::path::Path,
    previous: &mut Option<String>,
) -> Result<()> {
    let fingerprint = format!(
        "{}:{}",
        file_fingerprint(calibration_path),
        file_fingerprint(robot_world)
    );
    if previous.as_deref() == Some(&fingerprint) {
        return Ok(());
    }
    *previous = Some(fingerprint);
    let status = match (|| -> Result<String> {
        let bundle = crate::CalibrationBundle::load(calibration_path)?;
        viewer.log_calibration(&bundle, urdf, None, Some(robot_world))?;
        viewer.bind_tracking_registration(robot_world)?;
        Ok(format!(
            "Calibrated camera frustums: {} cameras from bundle {}",
            bundle.cameras.len(),
            bundle.bundle_id
        ))
    })() {
        Ok(status) => status,
        Err(error) => format!(
            "Calibrated camera frustums unavailable: {error:#}. Any previously displayed geometry is stale."
        ),
    };
    viewer.log_calibration_status(status)
}

/// Where the cockpit sends what it shows and how it places it.
#[cfg(feature = "rerun")]
#[derive(Clone, Copy)]
pub struct ViewBusOutput<'a> {
    /// The hub's gRPC proxy; may be absent when `output` is given.
    pub connect: Option<&'a str>,
    /// An `.rrd` file sink for an offline capture, beside or instead of the hub.
    pub output: Option<&'a std::path::Path>,
    pub recording_id: &'a str,
    /// The overall display cap on every bus key, Hz (`preview_interval`).
    pub max_fps: f64,
    /// Each camera stream's own rate beneath that cap.
    pub rates: CockpitRates,
    /// 0 runs until `stop` is raised.
    pub duration_seconds: u64,
    /// trackd's and stencild's poses at identity in the bus's camera world.
    pub camera_world: bool,
    /// The D555 depth shown reaches this far, metres; 0 shows all of it.
    pub overhead_depth_max_m: f64,
    /// A fiducial inventory whose `palette` target the cockpit reads in the
    /// overhead colour itself (`SceneTags`; a `fiducials` build).
    pub fiducials: Option<&'a std::path::Path>,
}

/// How far the cockpit shows the D555's depth by default, metres. On the
/// rig (2026-10-18 capture) the table and everything on it read 0.38-0.79 m
/// and everything beyond it (furniture, then the floor) 0.94-4.9 m; without the
/// clip the floor is most of the cloud and the Demo scene's bounds.
pub const OVERHEAD_DEPTH_MAX_M: f64 = 0.85;

/// A Demo wrist pane's label until its first frame, on short lines: each
/// upright pane is 270 px wide on the station, and a label is not wrapped.
#[cfg(feature = "rerun")]
const WRIST_PLACEHOLDER: &str = "wrist D405: no frame yet\nthe camera opens\nduring a draw";

/// The overhead D555's two streams on the bus.
#[cfg(any(feature = "rerun", test))]
const OVERHEAD_COLOR_KEY: &str = "tatbot/vision/overhead-depth/overhead_depth_color";
#[cfg(any(feature = "rerun", test))]
const OVERHEAD_DEPTH_KEY: &str = "tatbot/vision/overhead-depth/overhead_depth_depth";
/// The wrist taps' frame sets, `tatbot/vision/d405/<camera>`.
#[cfg(any(feature = "rerun", test))]
const WRIST_KEY_PREFIX: &str = "tatbot/vision/d405/";
/// The print's markers each wrist reading read in a frame its tap published,
/// `tatbot/vision/d405/<camera>/markers` (`scripts/lib/wrist_tap.py`): one
/// level below the frame sets, so `tatbot/vision/*/*` never takes them.
#[cfg(feature = "rerun")]
const WRIST_MARKERS_KEYS: &str = "tatbot/vision/d405/*/markers";
#[cfg(feature = "rerun")]
const WRIST_MARKERS_SCHEMA: &str = "tatbot.wrist-markers/1";
/// The readings' markers kept between two cockpit ticks, newest last: a
/// set-up reads one a hold, a draw one a hover or lift.
#[cfg(feature = "rerun")]
const WRIST_MARKERS_QUEUE: usize = 16;

/// The most one logged image's encoded bytes may be: with its row's own
/// framing the chunk stays within 64 KiB, about 0.5 ms at 1 Gbps. The hub
/// streams every chunk on to the station, on the link that also carries the
/// pink controller's 400 Hz UDP, and a big chunk holds it for its whole time
/// on the wire. On 2026-10-18 the owner's q95 colour, passed through, ran
/// ~470 KB a chunk and the 640x360 depth ~155 KB (after the stream's LZ4),
/// and the controller manager overran ~50 times a minute while the station
/// was connected.
#[cfg(any(feature = "rerun", test))]
const VIEWER_ROW_BYTES: usize = 60 * 1024;
/// Every colour image the cockpit shows fits this, as a JPEG of
/// `VIEWER_JPEG_QUALITY`: the D555's and the wrist tap's 1280x720 become
/// 640x360, ~52 KB on the D555's table (the `image` encoder keeps full
/// chroma), more than a Demo pane on the station's 860-wide portrait
/// screen shows.
#[cfg(any(feature = "rerun", test))]
const VIEWER_COLOR_MAX: (u32, u32) = (640, 360);
#[cfg(any(feature = "rerun", test))]
const VIEWER_JPEG_QUALITY: u8 = 70;
/// Every depth plane the cockpit shows fits this; the D555's aligned
/// 1280x720 becomes 320x180, ~17 KB of RVL once clipped to the table.
#[cfg(any(feature = "rerun", test))]
const VIEWER_DEPTH_MAX: (u32, u32) = (320, 180);
/// How often the cockpit wakes to take what is due.
#[cfg(feature = "rerun")]
const VIEWER_TICK: std::time::Duration = std::time::Duration::from_millis(20);
/// What the cockpit's earlier releases logged that this one logs otherwise
/// or not at all, cleared when it starts so the standing recording stops
/// drawing it: the D555's depth as a raw `DepthImage` (now RVL; the old
/// archetype's last row would still be drawn beside the new cloud) and each
/// wrist camera's depth (the bridge draws the wrist clouds, placed by tf;
/// RealSense 1's was its arm node's visiond's, on the blue arm until
/// 2026-10-23).
#[cfg(feature = "rerun")]
const RETIRED_ROWS: [&str; 3] = [
    "cameras/08_overhead_depth/depth",
    "cameras/07_realsense2/depth",
    "cameras/06_realsense1/depth",
];

/// The cockpit's own reading of a scene target's tags in the overhead
/// colour (the inventory's `palette`: its roof tag, which no tracker follows
/// and `ros station` reads only before a draw), at most once every `PERIOD`,
/// on the full frame its owner sent: the 640x360 the cockpit logs puts the
/// 46 mm tag ~20 px wide. Drawn by `RerunViewer::log_scene_tags`.
#[cfg(all(feature = "rerun", feature = "fiducials"))]
pub struct SceneTags {
    target: String,
    detector: crate::fiducials::AprilTagDetector,
    gate: RateGate,
}

#[cfg(all(feature = "rerun", feature = "fiducials"))]
impl SceneTags {
    pub const PERIOD: std::time::Duration = std::time::Duration::from_secs(1);

    pub fn new(inventory: &std::path::Path, target: &str) -> Result<Self> {
        let inventory = crate::fiducials::FiducialInventory::load(inventory)?;
        Ok(Self {
            target: target.to_owned(),
            detector: crate::fiducials::AprilTagDetector::new(&inventory, Some(target), None)?,
            gate: RateGate::new(Self::PERIOD),
        })
    }

    /// Read the target's tags in an overhead colour sample (`bytes`, a frame
    /// set on the bus) when one is due, and outline them on its image.
    fn read(&mut self, viewer: &crate::RerunViewer, bytes: &[u8], now: std::time::Instant) -> Result<()> {
        if !self.gate.admit(now) {
            return Ok(());
        }
        let set = decode_view(bytes)?;
        let Some((camera, frame)) = set.frames.values().find_map(rgb_frame) else {
            return Ok(());
        };
        let detections = self.detector.detect_frame(&camera, &frame)?;
        viewer.log_scene_tags(&self.target, u64::try_from(set.timestamp_ns)?, &detections)
    }
}

/// A JPEG colour frame decoded to RGB at its own size, with a camera model
/// from its `intrinsics` attribute (the detector's corners need none, its
/// pose candidates the focal lengths); None for any other frame.
#[cfg(all(feature = "rerun", feature = "fiducials"))]
fn rgb_frame(frame: &FrameRecord) -> Option<(crate::CameraCalibration, FrameRecord)> {
    let RecordedPayload::Encoded { format: PixelFormat::Jpeg, bytes } = &frame.payload else {
        return None;
    };
    let rgb = image::load_from_memory_with_format(bytes, image::ImageFormat::Jpeg).ok()?.to_rgb8();
    let (width, height) = rgb.dimensions();
    let optics: serde_json::Value = serde_json::from_str(frame.metadata.attributes.get("intrinsics")?).ok()?;
    let number = |key: &str| optics[key].as_f64();
    let profile = crate::StreamProfile { width, height, format: PixelFormat::Rgb8, ..frame.metadata.profile.clone() };
    let scale = f64::from(width) / number("width")?;
    let camera = crate::CameraCalibration {
        sensor_name: frame.metadata.sensor_name.clone(),
        profile: profile.clone(),
        intrinsics: crate::Intrinsics {
            width,
            height,
            fx: number("fx")? * scale,
            fy: number("fy")? * scale,
            cx: number("ppx")? * scale,
            cy: number("ppy")? * scale,
        },
        distortion: crate::DistortionModel { model: "none".into(), coefficients: Vec::new() },
        world_from_camera: crate::Pose { rotation: [1., 0., 0., 0., 1., 0., 0., 0., 1.], translation_m: [0.; 3] },
        depth_to_color: None,
        metadata: Default::default(),
    };
    let mut metadata = frame.metadata.clone();
    metadata.profile = profile;
    let payload = RecordedPayload::Video { format: PixelFormat::Rgb8, width, height, bytes: rgb.into_raw() };
    Some((camera, FrameRecord { metadata, payload }))
}

/// The cockpit's display rate per camera stream, Hz, each beneath the
/// overall `--max-fps`. At the defaults the cockpit streams ~0.4 MB/s
/// (the D555's RGB cloud ~0.1 of it), ~0.5 while a draw's wrist tap sends.
#[cfg(any(feature = "rerun", test))]
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct CockpitRates {
    /// The D555's colour.
    pub overhead_color_hz: f64,
    /// The D555's depth, the Demo scene's cloud.
    pub overhead_depth_hz: f64,
    /// Each wrist tap's colour; it sends a frame per grab while a draw runs.
    pub wrist_hz: f64,
}

#[cfg(any(feature = "rerun", test))]
impl CockpitRates {
    pub const DEFAULT: Self = Self {
        overhead_color_hz: 5.0,
        overhead_depth_hz: 1.0,
        wrist_hz: 2.0,
    };
}

/// The shortest gap between two logged samples of one bus key: its stream's
/// rate, any other key at the overall cap alone.
#[cfg(any(feature = "rerun", test))]
fn preview_interval(key: &str, rates: &CockpitRates, max_fps: f64) -> std::time::Duration {
    let hz = match key {
        OVERHEAD_COLOR_KEY => rates.overhead_color_hz,
        OVERHEAD_DEPTH_KEY => rates.overhead_depth_hz,
        _ if key.starts_with(WRIST_KEY_PREFIX) => rates.wrist_hz,
        _ => f64::INFINITY,
    };
    std::time::Duration::from_secs_f64(1.0 / hz.min(max_fps))
}

/// A per-key rate limit that tolerates jitter. Each take schedules the next
/// one an interval after the previous due time, and a sample up to a quarter
/// interval early still counts, so a 5 Hz owner read at 5 Hz is not halved
/// by a frame that lands a millisecond early, while the long-run rate never
/// exceeds the cap. After a pause the schedule restarts from now: no burst.
#[cfg(any(feature = "rerun", test))]
#[derive(Debug)]
struct RateGate {
    interval: std::time::Duration,
    next: Option<std::time::Instant>,
}

#[cfg(any(feature = "rerun", test))]
impl RateGate {
    fn new(interval: std::time::Duration) -> Self {
        Self {
            interval,
            next: None,
        }
    }

    fn admit(&mut self, now: std::time::Instant) -> bool {
        let slack = self.interval / 4;
        if self.next.is_some_and(|next| now + slack < next) {
            return false;
        }
        let scheduled = self.next.unwrap_or(now) + self.interval;
        self.next = Some(scheduled.max(now + self.interval - slack));
        true
    }
}

/// The latest-sample slots of one subscription, drained by key as each
/// key's gate admits it. A sample that is not yet due stays and is replaced
/// by a newer one, so what is logged is always the newest.
#[cfg(any(feature = "rerun", test))]
#[derive(Debug, Default)]
struct Feed {
    gates: std::collections::HashMap<String, RateGate>,
}

#[cfg(any(feature = "rerun", test))]
impl Feed {
    fn take_due<T>(
        &mut self,
        slots: &std::sync::Mutex<std::collections::BTreeMap<String, T>>,
        interval: impl Fn(&str) -> std::time::Duration,
        now: std::time::Instant,
    ) -> Vec<(String, T)> {
        let mut slots = slots.lock().unwrap_or_else(|error| error.into_inner());
        let due: Vec<String> = slots
            .keys()
            .filter(|key| {
                self.gates
                    .entry((*key).clone())
                    .or_insert_with(|| RateGate::new(interval(key.as_str())))
                    .admit(now)
            })
            .cloned()
            .collect();
        due.into_iter()
            .filter_map(|key| slots.remove(&key).map(|sample| (key, sample)))
            .collect()
    }
}

/// Due camera samples, logged one a tick: the D555 sends its colour and
/// depth at the same moment, and two rows sent back to back would hold the
/// hub's link for both. A sample waiting its turn is replaced by a newer one
/// of its key, so what is logged is still the newest.
#[cfg(any(feature = "rerun", test))]
#[derive(Debug)]
struct Paced<T> {
    queue: std::collections::VecDeque<(String, T)>,
}

#[cfg(any(feature = "rerun", test))]
impl<T> Default for Paced<T> {
    fn default() -> Self {
        Self {
            queue: Default::default(),
        }
    }
}

#[cfg(any(feature = "rerun", test))]
impl<T> Paced<T> {
    fn extend(&mut self, due: impl IntoIterator<Item = (String, T)>) {
        for (key, sample) in due {
            match self.queue.iter_mut().find(|(queued, _)| *queued == key) {
                Some(slot) => slot.1 = sample,
                None => self.queue.push_back((key, sample)),
            }
        }
    }

    fn next(&mut self) -> Option<(String, T)> {
        self.queue.pop_front()
    }
}

/// Samples the cockpit could not show: a decode error, a schema mismatch or
/// a failed log. Each is counted and skipped; stderr hears of them at most
/// every ten seconds, with the newest reason.
#[cfg(any(feature = "rerun", test))]
#[derive(Debug, Default)]
struct SkipLog {
    pending: u64,
    total: u64,
    last: Option<String>,
    reported: Option<std::time::Instant>,
}

#[cfg(any(feature = "rerun", test))]
impl SkipLog {
    const EVERY: std::time::Duration = std::time::Duration::from_secs(10);

    fn note(&mut self, key: &str, error: &anyhow::Error) {
        self.pending += 1;
        self.total += 1;
        self.last = Some(format!("{key}: {error:#}"));
    }

    /// The line to print now, if any: always when `force`, otherwise once
    /// per `EVERY` while something was skipped.
    fn report(&mut self, now: std::time::Instant, force: bool) -> Option<String> {
        let quiet = self
            .reported
            .is_some_and(|at| now.saturating_duration_since(at) < Self::EVERY);
        if self.pending == 0 || (quiet && !force) {
            return None;
        }
        let line = format!(
            "cockpit: skipped {} sample(s) ({} in all); last {}",
            self.pending,
            self.total,
            self.last.as_deref().unwrap_or("?")
        );
        self.pending = 0;
        self.reported = Some(now);
        Some(line)
    }
}

/// Decode one camera sample of bus `key` and log it as the cockpit shows it,
/// each image row within `VIEWER_ROW_BYTES`: colour as `fit_color` makes it,
/// the D555's depth as `fit_depth` does (cut beyond `max_range_m`, when
/// given; the viewer was told to log it RVL-compressed). Any other depth,
/// the wrist tap's, is left out: the bridge draws the wrist cloud.
#[cfg(feature = "rerun")]
fn log_frame_sample(
    viewer: &crate::RerunViewer,
    key: &str,
    bytes: &[u8],
    max_range_m: Option<f64>,
) -> Result<()> {
    let mut set = decode_view(bytes)?;
    let shows_depth = key == OVERHEAD_DEPTH_KEY;
    set.frames
        .retain(|_, frame| shows_depth || !matches!(frame.payload, RecordedPayload::Depth { .. }));
    for frame in set.frames.values_mut() {
        *frame = match frame.payload {
            RecordedPayload::Depth { .. } => fit_depth(frame, max_range_m),
            _ => fit_color(frame)?,
        };
    }
    if set.frames.is_empty() {
        return Ok(());
    }
    viewer.log_set(&set)
}

/// A colour image as the cockpit logs it: the owner's JPEG as it is when it
/// already fits `VIEWER_COLOR_MAX` and `VIEWER_ROW_BYTES`; otherwise decoded,
/// averaged down to fit `VIEWER_COLOR_MAX` (`shrink`) and re-encoded at
/// `VIEWER_JPEG_QUALITY`, at lower qualities, then at half the size, until
/// it fits. Its profile takes the logged size, so the Pinhole scaled from it
/// matches. The viewer only ever receives JPEG colour (`decode_view`).
#[cfg(any(feature = "rerun", test))]
fn fit_color(frame: &FrameRecord) -> Result<FrameRecord> {
    let RecordedPayload::Encoded {
        format: PixelFormat::Jpeg,
        bytes,
    } = &frame.payload
    else {
        return Ok(frame.clone());
    };
    let (width, height) = (frame.metadata.profile.width, frame.metadata.profile.height);
    let (max_width, max_height) = VIEWER_COLOR_MAX;
    if width <= max_width && height <= max_height && bytes.len() <= VIEWER_ROW_BYTES {
        return Ok(frame.clone());
    }
    let image = image::load_from_memory_with_format(bytes, image::ImageFormat::Jpeg)?;
    let scale = (f64::from(max_width) / f64::from(width))
        .min(f64::from(max_height) / f64::from(height))
        .min(1.0);
    let mut size = [width, height].map(|v| ((f64::from(v) * scale).round() as u32).max(1));
    loop {
        let small = shrink(&image, size);
        for quality in [VIEWER_JPEG_QUALITY, VIEWER_JPEG_QUALITY - 15, VIEWER_JPEG_QUALITY - 30] {
            let mut jpeg = Vec::new();
            small.write_with_encoder(image::codecs::jpeg::JpegEncoder::new_with_quality(
                &mut jpeg, quality,
            ))?;
            if jpeg.len() <= VIEWER_ROW_BYTES || size == [1, 1] {
                let mut fitted = frame.clone();
                fitted.payload = RecordedPayload::Encoded {
                    format: PixelFormat::Jpeg,
                    bytes: jpeg,
                };
                let purpose = "viewer_preview";
                let scale = f64::from(size[0]) / f64::from(width);
                crate::frame_ops::note_scaled(&mut fitted, width, height, size[0], size[1], scale, purpose);
                let attributes = &mut fitted.metadata.attributes;
                attributes.insert(format!("{purpose}_resize_filter"), "box".into());
                attributes.insert(format!("{purpose}_jpeg_quality"), quality.to_string());
                return Ok(fitted);
            }
        }
        size = size.map(|v| (v / 2).max(1));
    }
}

/// `image` averaged down to `size`: block means of k x k pixels when k
/// divides it evenly (the 1280x720 colour to 640x360 takes ~2 ms in a
/// release build, where `thumbnail_exact` takes ~9), else `thumbnail_exact`.
#[cfg(any(feature = "rerun", test))]
fn shrink(image: &image::DynamicImage, [width, height]: [u32; 2]) -> image::DynamicImage {
    let k = image.width() / width;
    let (channels, raw) = match image {
        image::DynamicImage::ImageRgb8(rgb) => (3, rgb.as_raw()),
        image::DynamicImage::ImageLuma8(luma) => (1, luma.as_raw()),
        _ => return image.thumbnail_exact(width, height),
    };
    if k < 2 || image.width() != k * width || image.height() != k * height {
        return image.thumbnail_exact(width, height);
    }
    let (k, stride) = (k as usize, image.width() as usize * channels);
    let area = (k * k) as u32;
    let mut out = Vec::with_capacity(width as usize * height as usize * channels);
    for y in 0..height as usize {
        for x in 0..width as usize {
            for channel in 0..channels {
                let mut sum = 0_u32;
                for row in y * k..(y + 1) * k {
                    let start = row * stride + x * k * channels + channel;
                    let block = &raw[start..=start + (k - 1) * channels];
                    sum += block.iter().step_by(channels).map(|&v| u32::from(v)).sum::<u32>();
                }
                out.push(((sum + area / 2) / area) as u8);
            }
        }
    }
    let shrunk = match channels {
        3 => image::RgbImage::from_raw(width, height, out).map(image::DynamicImage::ImageRgb8),
        _ => image::GrayImage::from_raw(width, height, out).map(image::DynamicImage::ImageLuma8),
    };
    shrunk.expect("a buffer of exactly width x height x channels")
}

/// A depth plane as the cockpit logs it: decimated to fit `VIEWER_DEPTH_MAX`
/// by the nearest valid range in each block (`decimate_depth_valid_min`),
/// cut beyond `max_range_m` when given, and decimated by twice the block
/// again until its RVL fits `VIEWER_ROW_BYTES`; only a plane of noise needs
/// that, the D555's table is ~17 KB at 320x180.
#[cfg(any(feature = "rerun", test))]
fn fit_depth(frame: &FrameRecord, max_range_m: Option<f64>) -> FrameRecord {
    let (mut max_width, mut max_height) = VIEWER_DEPTH_MAX;
    loop {
        let mut fitted =
            crate::frame_ops::decimate_depth_valid_min(frame, max_width, max_height, "viewer_preview")
                .unwrap_or_else(|| frame.clone());
        if let Some(max_m) = max_range_m {
            crate::frame_ops::clip_depth_beyond(&mut fitted, max_m, "viewer_preview");
        }
        let RecordedPayload::Depth {
            width,
            height,
            bytes,
        } = &fitted.payload
        else {
            return fitted;
        };
        let smallest = max_width == 1 && max_height == 1;
        if smallest || crate::frame_ops::rvl_depth(*width, *height, bytes).len() <= VIEWER_ROW_BYTES {
            return fitted;
        }
        (max_width, max_height) = ((max_width / 2).max(1), (max_height / 2).max(1));
    }
}

/// Decode one wrist reading's markers (`WRIST_MARKERS_SCHEMA`) for the viewer.
#[cfg(feature = "rerun")]
fn decode_markers(bytes: &[u8]) -> Result<crate::rerun_viewer::WristMarkers> {
    let value: Envelope<crate::rerun_viewer::WristMarkers> = serde_json::from_slice(bytes)?;
    ensure!(
        value.schema == WRIST_MARKERS_SCHEMA,
        "expected {WRIST_MARKERS_SCHEMA}, got {}",
        value.schema
    );
    Ok(value.payload)
}

/// Decode one pose envelope of `schema` and hand it to `log`.
#[cfg(feature = "rerun")]
fn log_pose_sample(
    viewer: &crate::RerunViewer,
    bytes: &[u8],
    schema: &str,
    log: fn(&crate::RerunViewer, u64, &serde_json::Value) -> Result<()>,
) -> Result<()> {
    let value: Envelope<serde_json::Value> = serde_json::from_slice(bytes)?;
    ensure!(value.schema == schema, "expected {schema}, got {}", value.schema);
    log(viewer, value.stamp.wall_ns, &value.payload)
}

#[cfg(feature = "rerun")]
fn now_unix_ns() -> i128 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map_or(0, |d| d.as_nanos() as i128)
}

#[cfg(feature = "rerun")]
type Slots = std::sync::Arc<std::sync::Mutex<std::collections::BTreeMap<String, zenoh::sample::Sample>>>;

#[cfg(feature = "rerun")]
pub fn view_bus(
    connect: &[String],
    out: ViewBusOutput<'_>,
    scene: ViewBusScene<'_>,
    stop: &std::sync::atomic::AtomicBool,
) -> Result<()> {
    use zenoh::Wait;
    ensure!(
        out.max_fps.is_finite() && out.max_fps > 0.0 && out.max_fps <= 5.0,
        "subscriber viewer cap must be in (0,5] Hz"
    );
    ensure!(
        out.overhead_depth_max_m.is_finite() && out.overhead_depth_max_m >= 0.0,
        "the overhead depth range must be a distance in metres, or 0 for all of it"
    );
    let overhead_clip = (out.overhead_depth_max_m > 0.0).then_some(out.overhead_depth_max_m);
    #[cfg(feature = "fiducials")]
    let mut scene_tags = out.fiducials.map(|path| SceneTags::new(path, "palette")).transpose()?;
    #[cfg(not(feature = "fiducials"))]
    ensure!(out.fiducials.is_none(), "--fiducials needs a build with the fiducials feature");
    let rates = out.rates;
    ensure!(
        [rates.overhead_color_hz, rates.overhead_depth_hz, rates.wrist_hz]
            .iter()
            .all(|hz| hz.is_finite() && *hz > 0.0),
        "every cockpit stream rate must be a positive rate in Hz"
    );
    ensure!(
        !out.camera_world || (scene.robot_world.is_none() && scene.calibration.is_none()),
        "camera-world overlays take no robot-world registration or calibration"
    );
    let bus = tatbot_bus::transport::Bus::open(connect).map_err(|e| anyhow::anyhow!("{e}"))?;
    let latest: Slots = Default::default();
    let incoming = latest.clone();
    let _frames = bus
        .session
        .declare_subscriber("tatbot/vision/*/*")
        .callback(move |sample| {
            let key = sample.key_expr().as_str();
            if (key.starts_with("tatbot/vision/d405/realsense")
                || key.starts_with("tatbot/vision/overhead-depth/overhead_depth"))
                && let Ok(mut slot) = incoming.try_lock()
                && (slot.contains_key(key) || slot.len() < 16)
            {
                slot.insert(key.to_owned(), sample);
            }
        })
        .wait()
        .map_err(|e| anyhow::anyhow!("{e}"))?;
    // Both arms' end-effector tracking, the latest sample per arm topic.
    let tracking: Slots = Default::default();
    let input = tracking.clone();
    let _poses = bus
        .session
        .declare_subscriber("tatbot/tracking/ee/*")
        .callback(move |sample| {
            let key = sample.key_expr().as_str();
            if let Some(arm) = ee_tracking_arm(key)
                && let Ok(mut slot) = input.try_lock()
            {
                slot.insert(arm.to_owned(), sample);
            }
        })
        .wait()
        .map_err(|e| anyhow::anyhow!("{e}"))?;
    // Every print the stencil observer publishes, the latest sample per print.
    let targets: Slots = Default::default();
    let prints = targets.clone();
    let beyond = std::sync::Mutex::new(std::collections::BTreeSet::<String>::new());
    let _targets = bus
        .session
        .declare_subscriber("tatbot/tracking/target/*")
        .callback(move |sample| {
            let key = sample.key_expr().as_str();
            if let Some(pattern) = target_pattern(key)
                && let Ok(mut slot) = prints.try_lock()
            {
                if print_slot_free(slot.len(), slot.contains_key(pattern)) {
                    slot.insert(pattern.to_owned(), sample);
                } else if beyond.lock().unwrap().insert(pattern.to_owned()) {
                    eprintln!(
                        "viewer: print {pattern} is beyond the {VIEWER_PRINTS} outlined prints and is not followed"
                    );
                }
            }
        })
        .wait()
        .map_err(|e| anyhow::anyhow!("{e}"))?;
    // Each wrist reading's markers, in the order read; every one is drawn.
    let marks = std::sync::Arc::new(std::sync::Mutex::new(std::collections::VecDeque::new()));
    let read = marks.clone();
    let _marks = bus
        .session
        .declare_subscriber(WRIST_MARKERS_KEYS)
        .callback(move |sample: zenoh::sample::Sample| {
            // held only while the loop drains it: a reading is not dropped for that
            let mut queue = read.lock().unwrap_or_else(|error| error.into_inner());
            if queue.len() == WRIST_MARKERS_QUEUE {
                queue.pop_front();
            }
            queue.push_back(sample);
        })
        .wait()
        .map_err(|e| anyhow::anyhow!("{e}"))?;
    let mut viewer =
        crate::RerunViewer::connect_or_save(out.connect, out.output, Some(out.recording_id))?;
    viewer.set_compressed_depth(true);
    viewer.log_session_metadata("camera_bus", Some(out.recording_id), scene.urdf, None)?;
    viewer.log_scene(None, scene.urdf, None)?;
    ensure!(
        scene.calibration.is_none() || scene.robot_world.is_some(),
        "camera calibration preview requires robot-world registration"
    );
    let mut calibration_fingerprint = None;
    if out.camera_world {
        viewer.place_tracking_in_camera_world();
    } else if let (Some(calibration), Some(robot_world)) = (scene.calibration, scene.robot_world) {
        refresh_calibration(
            &viewer,
            scene.urdf,
            calibration,
            robot_world,
            &mut calibration_fingerprint,
        )?;
    } else if let Some(path) = scene.robot_world {
        viewer.bind_tracking_registration(path)?;
    }
    viewer.log_status("camera bus subscriber; no camera clients")?;
    viewer.clear(&RETIRED_ROWS, now_unix_ns())?;
    let (width, height) = VIEWER_COLOR_MAX;
    for (_, sensor) in crate::rerun_viewer::DEMO_WRISTS {
        viewer.await_image(sensor, WRIST_PLACEHOLDER, [width, height], now_unix_ns())?;
    }
    let started = std::time::Instant::now();
    let cap = out.max_fps;
    let (mut frames, mut poses, mut prints) = (Feed::default(), Feed::default(), Feed::default());
    let mut paced = Paced::default();
    let mut skipped = SkipLog::default();
    let mut ages_logged = started;
    let mut scene_checked = started;
    let hub_lost = loop {
        let now = std::time::Instant::now();
        paced.extend(frames.take_due(&latest, |key| preview_interval(key, &rates, cap), now));
        if let Some((key, sample)) = paced.next() {
            let clip = overhead_clip.filter(|_| key == OVERHEAD_DEPTH_KEY);
            let bytes = sample.payload().to_bytes();
            if let Err(error) = log_frame_sample(&viewer, &key, &bytes, clip) {
                skipped.note(&key, &error);
            }
            #[cfg(feature = "fiducials")]
            if let Some(tags) = scene_tags.as_mut().filter(|_| key == OVERHEAD_COLOR_KEY)
                && let Err(error) = tags.read(&viewer, &bytes, now)
            {
                skipped.note("the palette's tag", &error);
            }
        }
        let interval = |_: &str| std::time::Duration::from_secs_f64(1.0 / cap);
        for (arm, sample) in poses.take_due(&tracking, interval, now) {
            let bytes = sample.payload().to_bytes();
            let logged = log_pose_sample(
                &viewer,
                &bytes,
                "tatbot.tracking-pose/1",
                crate::RerunViewer::log_tracking,
            );
            if let Err(error) = logged {
                skipped.note(&format!("tatbot/tracking/ee/{arm}"), &error);
            }
        }
        for (pattern, sample) in prints.take_due(&targets, interval, now) {
            let bytes = sample.payload().to_bytes();
            let logged = log_pose_sample(
                &viewer,
                &bytes,
                "tatbot.target-pose/1",
                crate::RerunViewer::log_target,
            );
            if let Err(error) = logged {
                skipped.note(&format!("tatbot/tracking/target/{pattern}"), &error);
            }
        }
        let read: Vec<_> = marks
            .lock()
            .unwrap_or_else(|error| error.into_inner())
            .drain(..)
            .collect();
        for sample in read {
            let logged = decode_markers(&sample.payload().to_bytes())
                .and_then(|markers| viewer.log_markers(&markers));
            if let Err(error) = logged {
                skipped.note(sample.key_expr().as_str(), &error);
            }
        }
        if now.duration_since(ages_logged) >= std::time::Duration::from_secs(1) {
            if let Err(error) = viewer.log_ages(now_unix_ns()) {
                skipped.note("age labels", &error);
            }
            ages_logged = now;
        }
        if now.duration_since(scene_checked) >= std::time::Duration::from_secs(2) {
            if let (Some(calibration), Some(robot_world)) =
                (scene.calibration, scene.robot_world)
                && let Err(error) = refresh_calibration(
                    &viewer,
                    scene.urdf,
                    calibration,
                    robot_world,
                    &mut calibration_fingerprint,
                )
            {
                skipped.note("calibration", &error);
            }
            scene_checked = now;
        }
        if let Some(line) = skipped.report(now, false) {
            eprintln!("{line}");
        }
        // A hub restart drops the stream for good (no redial in Rerun 0.36):
        // exit, so systemd's restart sends every static row to the new hub.
        if viewer.hub_lost() {
            break true;
        }
        let expired =
            out.duration_seconds > 0 && started.elapsed().as_secs() >= out.duration_seconds;
        if expired || stop.load(std::sync::atomic::Ordering::Acquire) {
            break false;
        }
        std::thread::sleep(VIEWER_TICK);
    };
    if let Some(line) = skipped.report(std::time::Instant::now(), true) {
        eprintln!("{line}");
    }
    // A hub that went away cannot be flushed to; the file sink still closes
    // when the stream drops. A failed last flush is not the caller's failure,
    // a hub that dropped the stream is: the exit lets `Restart=` redial.
    if let Err(error) = viewer.finish() {
        eprintln!("cockpit: final flush incomplete: {error:#}");
    }
    ensure!(!hub_lost, "cockpit: the viewer hub dropped the stream; a restart reconnects");
    Ok(())
}

/// The arm an end-effector tracking key names: `tatbot/tracking/ee/<arm>`,
/// the pink (right) arm's alone.
#[cfg(any(feature = "rerun", test))]
fn ee_tracking_arm(key: &str) -> Option<&str> {
    let arm = key.strip_prefix("tatbot/tracking/ee/")?;
    (arm == "right").then_some(arm)
}

/// The viewer outlines at most this many prints at once.
#[cfg(any(feature = "rerun", test))]
const VIEWER_PRINTS: usize = 16;

/// Whether a print target takes a slot: an outlined print always, a new one
/// while a slot is free. A print beyond the bound is named once on stderr,
/// never dropped in silence.
#[cfg(any(feature = "rerun", test))]
fn print_slot_free(followed: usize, outlined: bool) -> bool {
    outlined || followed < VIEWER_PRINTS
}

/// The print a target key names: `tatbot/tracking/target/<pattern_id>`, one
/// topic per visible print (`stencil-<sha256>`; a tag target keeps its own
/// shorter id on the same prefix).
#[cfg(any(feature = "rerun", test))]
fn target_pattern(key: &str) -> Option<&str> {
    let pattern = key.strip_prefix("tatbot/tracking/target/")?;
    (!pattern.is_empty()
        && pattern.len() <= 72
        && pattern
            .bytes()
            .all(|c| c.is_ascii_alphanumeric() || c == b'-' || c == b'_'))
    .then_some(pattern)
}

/// Scan requests require a recent, explicitly aligned depth/color pair.
fn validate_capture(set: &SynchronizedFrameSet) -> Result<()> {
    let now = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)?
        .as_nanos() as i128;
    ensure!(
        (0..=250_000_000).contains(&(now - set.timestamp_ns)),
        "RGBD capture is stale or from the future"
    );
    validate_capture_geometry(set)
}

#[derive(serde::Deserialize, serde::Serialize)]
#[serde(deny_unknown_fields)]
pub struct CaptureWindow {
    pub after_ns: i128,
    pub before_ns: i128,
}

fn select_capture(
    history: &std::collections::VecDeque<std::sync::Arc<SynchronizedFrameSet>>,
    payload: Option<&[u8]>,
) -> Result<std::sync::Arc<SynchronizedFrameSet>> {
    let Some(bytes) = payload else {
        let set = history
            .back()
            .ok_or_else(|| anyhow::anyhow!("no RGBD capture available"))?
            .clone();
        validate_capture(&set)?;
        return Ok(set);
    };
    ensure!(bytes.len() <= 1024, "capture window exceeds budget");
    let window: CaptureWindow = serde_json::from_slice(bytes)?;
    let now = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)?
        .as_nanos() as i128;
    ensure!(
        window.after_ns > 0
            && window.after_ns <= window.before_ns
            && window.before_ns <= now
            && now - window.after_ns <= 3_000_000_000,
        "invalid or expired capture window"
    );
    let middle = window.after_ns + (window.before_ns - window.after_ns) / 2;
    let set = history
        .iter()
        .filter(|set| {
            set.frames.values().all(|frame| {
                frame
                    .metadata
                    .timestamps
                    .normalized_unix_ns
                    .is_some_and(|stamp| (window.after_ns..=window.before_ns).contains(&stamp))
            })
        })
        .min_by_key(|set| set.timestamp_ns.abs_diff(middle))
        .ok_or_else(|| anyhow::anyhow!("no complete RGBD pair in requested exposure window"))?
        .clone();
    validate_capture_geometry(&set)?;
    Ok(set)
}

pub fn validate_capture_geometry(set: &SynchronizedFrameSet) -> Result<()> {
    let mut depths = 0;
    for frame in set.frames.values() {
        if let RecordedPayload::Depth { width, height, .. } = &frame.payload {
            let color = frame
                .metadata
                .attributes
                .get("aligned_to")
                .and_then(|name| set.frames.get(name))
                .ok_or_else(|| anyhow::anyhow!("depth is not aligned to a retained color frame"))?;
            ensure!(
                (color.metadata.profile.width, color.metadata.profile.height) == (*width, *height),
                "aligned dimensions differ"
            );
            ensure!(
                frame.metadata.sequence == color.metadata.sequence
                    && frame.metadata.attributes.get("device_serial")
                        == color.metadata.attributes.get("device_serial")
                    && frame.metadata.attributes.get("capture_epoch")
                        == color.metadata.attributes.get("capture_epoch"),
                "aligned RGBD identity differs"
            );
            ensure!(
                frame
                    .metadata
                    .attributes
                    .get("depth_units_m")
                    .and_then(|value| value.parse::<f64>().ok())
                    .is_some_and(|value| value.is_finite() && value > 0.0),
                "metric depth units missing"
            );
            depths += 1;
        }
    }
    ensure!(depths > 0, "capture contains no depth");
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::FrameMetadata;

    #[test]
    fn the_viewer_follows_the_pink_arms_ee_tracking_topic() {
        assert_eq!(ee_tracking_arm("tatbot/tracking/ee/right"), Some("right"));
        assert_eq!(ee_tracking_arm("tatbot/tracking/ee/left"), None);
        assert_eq!(ee_tracking_arm("tatbot/tracking/ee/centre"), None);
        assert_eq!(ee_tracking_arm("tatbot/tracking/target/print-1"), None);
    }

    #[test]
    fn the_viewer_follows_every_print_target_topic() {
        let pattern = format!("stencil-{}", "a".repeat(64));
        assert_eq!(
            target_pattern(&format!("tatbot/tracking/target/{pattern}")),
            Some(pattern.as_str())
        );
        assert_eq!(target_pattern("tatbot/tracking/target/paper"), Some("paper"));
        assert_eq!(target_pattern("tatbot/tracking/ee/right"), None);
        assert_eq!(target_pattern("tatbot/tracking/target/"), None);
        assert_eq!(target_pattern("tatbot/tracking/target/../x"), None);
        // Sixteen prints are outlined; a seventeenth new one is refused (and
        // named on stderr), an already outlined one keeps its slot.
        assert!(print_slot_free(15, false));
        assert!(!print_slot_free(VIEWER_PRINTS, false));
        assert!(print_slot_free(VIEWER_PRINTS, true));
    }

    #[test]
    fn wrist_snapshot_topics_never_alias_between_capture_owners() {
        let first = capture_topic("d405", "capture-a");
        let second = capture_topic("d405", "capture-b");
        assert_ne!(first, second);
        assert_ne!(first, "tatbot/vision/d405/capture");
        assert_ne!(second, "tatbot/vision/d405/capture");
        assert_eq!(
            capture_topic("overhead-depth", "capture-a"),
            "tatbot/vision/overhead-depth/capture"
        );
    }

    #[test]
    fn a_reset_record_rides_the_owner_capture_topic_under_its_own_schema() {
        let producer = Producer {
            node: "capture-a".into(),
            pid: 7,
            sha: "sha".into(),
            run_id: "run".into(),
        };
        let payload = CaptureHealth {
            event: "reset".into(),
            group: "d405".into(),
            sensor: "wrist_upper".into(),
            serial: "1234".into(),
            reason: "no frame for 15 s".into(),
            last_error: Some("control transfer -71".into()),
            empty_results: 10,
            outcome: "reset issued".into(),
        };
        let (key, envelope) = health_record("d405", &producer, 3, 42, payload.clone());
        assert_eq!(key, capture_topic("d405", "capture-a"));
        assert_eq!(envelope.schema, CAPTURE_HEALTH_SCHEMA);
        assert_eq!(envelope.seq, 3);
        assert_eq!(envelope.stamp.mono_ns, 42);
        let bytes = serde_json::to_vec(&envelope).unwrap();
        let decoded =
            tatbot_bus::transport::decode::<CaptureHealth>(&bytes, CAPTURE_HEALTH_SCHEMA).unwrap();
        assert_eq!(decoded.payload, payload);
        assert!(
            tatbot_bus::transport::decode::<CaptureHealth>(&bytes, "tatbot.frame-set/1").is_err()
        );
    }

    #[test]
    fn overhead_clock_query_echoes_nonce_and_frame_identity_on_same_response() {
        let (producer, mut set) = fixture();
        set.sequence = 7;
        for frame in set.frames.values_mut() {
            frame
                .metadata
                .attributes
                .insert("capture_epoch".into(), "camera-epoch".into());
        }
        let request = CaptureClockRequest::new("a".repeat(32)).unwrap();
        let payload = serde_json::to_vec(&request).unwrap();
        assert_eq!(
            capture_clock_request("overhead-depth", Some(&payload)).unwrap(),
            Some(request.clone())
        );
        assert!(
            capture_clock_request("d405", Some(&payload))
                .unwrap()
                .is_none()
        );
        let sample = CaptureClockSample::for_response(
            &request,
            &producer,
            &set,
            1_000_000_000,
            1_010_000_000,
            10_000_000,
            None,
        )
        .unwrap();
        let wire = encode_clocked_capture(&set, &producer, &sample).unwrap();
        let (source, decoded, echoed) = decode_clocked_capture(&wire).unwrap();
        assert_eq!(source, producer);
        assert_eq!(decoded.sequence, set.sequence);
        assert_eq!(echoed.sequence, 7);
        assert_eq!(echoed.frame_sequence, 1);
        assert_eq!(echoed, sample);
        assert!(decode_clocked_capture(&encode(&set, &producer).unwrap()).is_err());
    }

    #[test]
    fn overhead_owner_drains_then_answers_with_a_later_set() {
        use zenoh::Wait;
        let (producer, mut set) = fixture();
        set.sequence = 7;
        for frame in set.frames.values_mut() {
            frame
                .metadata
                .attributes
                .insert("capture_epoch".into(), "camera-epoch".into());
        }
        set.frames
            .get_mut("depth")
            .unwrap()
            .metadata
            .attributes
            .extend([
                ("aligned_to".into(), "color".into()),
                ("depth_units_m".into(), "0.001".into()),
            ]);
        let publisher = FramePublisher::open(&[], producer.clone(), "overhead-depth").unwrap();
        let stamp = host_wall_ns().unwrap();
        set.timestamp_ns = i128::from(stamp);
        for frame in set.frames.values_mut() {
            frame.metadata.timestamps.host_unix_ns = i128::from(stamp);
        }
        publisher.submit(&set).unwrap();
        let request = CaptureClockRequest::new("b".repeat(32)).unwrap();
        let replies = publisher
            .health
            .session
            .get(capture_topic("overhead-depth", &producer.node))
            .payload(serde_json::to_vec(&request).unwrap())
            .wait()
            .unwrap();
        let generation = publisher.fresh_generation().unwrap();
        let started = std::time::Instant::now();
        while generation.load(std::sync::atomic::Ordering::Acquire) == 0 {
            assert!(started.elapsed() < std::time::Duration::from_secs(1));
            std::thread::sleep(std::time::Duration::from_millis(2));
        }
        set.sequence = 8;
        set.timestamp_ns = i128::from(host_wall_ns().unwrap());
        let stamp = set.timestamp_ns;
        for frame in set.frames.values_mut() {
            frame.metadata.sequence = 2;
            frame.metadata.timestamps.host_unix_ns = stamp;
            frame.metadata.attributes.insert(
                "sdk_queue_flush_generation".into(), "1".into()
            );
            frame.metadata.attributes.insert(
                "sdk_queue_flush_discarded".into(), "0".into()
            );
        }
        publisher.submit(&set).unwrap();
        let reply = replies
            .recv_timeout(std::time::Duration::from_secs(2))
            .unwrap()
            .unwrap();
        let sample_reply = reply.result().unwrap();
        let bytes = sample_reply.payload().to_bytes();
        let (source, selected, sample) = decode_clocked_capture(&bytes).unwrap();
        assert_eq!(source, producer);
        assert_eq!(selected.sequence, set.sequence);
        assert_eq!(sample.frame_sequence, 2);
        assert_eq!(sample.fresh_selection.unwrap().previous_set_sequence, Some(7));
        assert_eq!(sample.request_nonce, request.nonce);
        assert_eq!(sample.capture_epoch, "camera-epoch");
        assert!(sample.owner_received_ns <= sample.owner_prepared_ns);
    }

    #[test]
    fn fresh_selection_refuses_prebarrier_wrong_generation_and_missing_set() {
        let (_, mut set) = fixture();
        set.frames
            .get_mut("depth")
            .unwrap()
            .metadata
            .attributes
            .extend([
                ("aligned_to".into(), "color".into()),
                ("depth_units_m".into(), "0.001".into()),
            ]);
        let tag = |set: &mut SynchronizedFrameSet, generation: u64| {
            for frame in set.frames.values_mut() {
                frame.metadata.attributes.insert(
                    "sdk_queue_flush_generation".into(), generation.to_string()
                );
                frame.metadata.attributes.insert(
                    "sdk_queue_flush_discarded".into(), "0".into()
                );
            }
        };
        let history: FrameHistory = std::sync::Arc::new(std::sync::Mutex::new(Default::default()));
        let short = std::time::Duration::from_millis(15);
        assert!(select_after_sdk_drain(&history, 1, None, short).is_err());
        set.sequence = 7;
        tag(&mut set, 1);
        retain_history(&history, std::sync::Arc::new(set.clone()), 4);
        assert!(select_after_sdk_drain(&history, 1, Some(7), short).is_err());
        set.sequence = 8;
        tag(&mut set, 2);
        retain_history(&history, std::sync::Arc::new(set.clone()), 4);
        assert!(select_after_sdk_drain(&history, 1, Some(7), short).is_err());
        tag(&mut set, 1);
        retain_history(&history, std::sync::Arc::new(set), 4);
        let (selected, witness) = select_after_sdk_drain(&history, 1, Some(7), short).unwrap();
        assert_eq!(selected.sequence, 8);
        assert_eq!(witness.previous_set_sequence, Some(7));
    }

    fn fixture() -> (Producer, SynchronizedFrameSet) {
        let metadata = FrameMetadata {
            sensor_name: "color".into(),
            sensor_kind: crate::SensorKind::RealSense,
            sequence: 1,
            profile: crate::StreamProfile {
                stream: "color".into(),
                width: 2,
                height: 2,
                fps_num: 20,
                fps_den: 1,
                format: PixelFormat::Bgr8,
            },
            timestamps: crate::FrameTimestamps {
                source_ns: Some(1),
                source_domain: crate::TimestampDomain::CameraNtp,
                rtp_timestamp: None,
                pipeline_pts_ns: None,
                pipeline_dts_ns: None,
                host_monotonic_ns: 2,
                host_unix_ns: 3,
                normalized_unix_ns: Some(3),
            },
            dropped_before: 0,
            calibration_id: Some("test-calibration".into()),
            flags: Vec::new(),
            attributes: Default::default(),
        };
        let color = FrameRecord {
            metadata: metadata.clone(),
            payload: RecordedPayload::Video {
                format: PixelFormat::Bgr8,
                width: 2,
                height: 2,
                bytes: [20, 30, 40].repeat(4),
            },
        };
        let mut depth = metadata;
        depth.sensor_name = "depth".into();
        depth.profile.stream = "depth".into();
        depth.profile.format = PixelFormat::Z16;
        let depth = FrameRecord {
            metadata: depth,
            payload: RecordedPayload::Depth {
                width: 2,
                height: 2,
                bytes: vec![0, 1, 2, 3, 4, 5, 6, 7],
            },
        };
        (
            Producer {
                node: "mock".into(),
                pid: 1,
                sha: "test".into(),
                run_id: "test".into(),
            },
            SynchronizedFrameSet {
                sequence: 1,
                timestamp_basis: "normalized".into(),
                timestamp_ns: 3,
                maximum_skew_ns: 0,
                frames: std::collections::BTreeMap::from([
                    ("color".into(), color),
                    ("depth".into(), depth),
                ]),
            },
        )
    }
    #[test]
    fn d405_yuyv_color_converts_and_depth_remains_exact() {
        let (producer, mut set) = fixture();
        let color = set.frames.get_mut("color").unwrap();
        color.metadata.profile.format = PixelFormat::Yuyv;
        color.payload = RecordedPayload::Video {
            format: PixelFormat::Yuyv,
            width: 2,
            height: 2,
            bytes: vec![80, 128, 80, 128, 80, 128, 80, 128],
        };
        let (_, decoded) = decode(&encode(&set, &producer).unwrap()).unwrap();
        assert_eq!(decoded.frames["depth"], set.frames["depth"]);
        let color = &decoded.frames["color"];
        assert_eq!(color.metadata.profile.format, PixelFormat::Rgb8);
        assert_eq!(color.payload.bytes().len(), 12);
        assert!(color.payload.bytes().iter().all(|v| v.abs_diff(80) <= 2));
        if let RecordedPayload::Video { bytes, .. } =
            &mut set.frames.get_mut("color").unwrap().payload
        {
            bytes.pop();
        }
        assert!(encode(&set, &producer).is_err());
    }
    #[test]
    fn viewer_decode_preserves_jpeg_bytes_and_exact_depth() {
        let (producer, set) = fixture();
        let wire = encode(&set, &producer).unwrap();
        let transported = crate::transport::decode_frame_set(&wire).unwrap();
        let camera_jpeg = &transported
            .frames
            .iter()
            .find(|f| f.metadata.sensor_name == "color")
            .unwrap()
            .payload;
        let preview = decode_view(&wire).unwrap();
        assert!(matches!(
            preview.frames["color"].payload,
            RecordedPayload::Encoded {
                format: PixelFormat::Jpeg,
                ..
            }
        ));
        assert_eq!(&preview.frames["color"].payload, camera_jpeg);
        assert_eq!(preview.frames["depth"], set.frames["depth"]);
        assert_eq!(preview.timestamp_ns, set.timestamp_ns);
        assert!(matches!(
            decode(&wire).unwrap().1.frames["color"].payload,
            RecordedPayload::Video { .. }
        ));
    }

    #[test]
    fn the_python_wire_golden_reads_as_a_bus_frame_set() {
        // The wrist tap's encoder (scripts/lib/frame_set_wire.py) against the
        // bus decoder every subscriber runs, and the cockpit's view decoder.
        let golden = crate::transport::PYTHON_WIRE_GOLDEN;
        let (producer, set) = decode(golden).expect("the bus decoder reads the Python set");
        assert_eq!((producer.node.as_str(), producer.pid), ("camera-host", 4242));
        assert_eq!((set.sequence, set.timestamp_ns), (3, 1_760_000_000_123_456_789));
        assert_eq!(
            set.frames.keys().map(String::as_str).collect::<Vec<_>>(),
            ["realsense2_color", "realsense2_depth"]
        );
        let RecordedPayload::Video { format: PixelFormat::Bgr8, width: 4, height: 3, bytes } =
            &set.frames["realsense2_color"].payload
        else {
            panic!("the 4x3 JPEG decodes to bgr8: {:?}", set.frames["realsense2_color"].payload);
        };
        assert_eq!(bytes.len(), 4 * 3 * 3);
        let view = decode_view(golden).unwrap();
        let transported = crate::transport::decode_frame_set(golden).unwrap();
        assert_eq!(view.frames["realsense2_color"].payload, transported.frames[0].payload);
        assert_eq!(set.frames["realsense2_depth"], view.frames["realsense2_depth"]);
        assert_eq!(
            set.frames["realsense2_depth"].payload,
            RecordedPayload::Depth {
                width: 3,
                height: 2,
                bytes: vec![0xe8, 0x03, 0xe9, 0x03, 0xea, 0x03, 0xd0, 0x07, 0, 0, 0xff, 0xff],
            }
        );
    }

    #[test]
    fn depth_preview_preserves_metric_samples_and_original_capture() {
        let (producer, mut set) = fixture();
        set.frames
            .get_mut("depth")
            .unwrap()
            .metadata
            .attributes
            .insert("depth_units_m".into(), "0.0001".into());
        let original = set.clone();
        let preview = crate::frame_ops::scale_depth_set(&set, 0.5, "bus_preview");
        let (_, decoded) = decode(&encode(&preview, &producer).unwrap()).unwrap();
        let depth = &decoded.frames["depth"];
        assert_eq!(depth.metadata.profile.width, 1);
        assert_eq!(depth.metadata.profile.height, 1);
        assert_eq!(depth.metadata.attributes["depth_units_m"], "0.0001");
        assert_eq!(
            depth.payload.bytes(),
            &original.frames["depth"].payload.bytes()[..2]
        );
        assert_eq!(set.frames, original.frames);
    }

    #[test]
    fn unsubscribed_publisher_survives_a_whole_metrics_window() {
        let (producer, set) = fixture();
        let publisher = FramePublisher::open(&[], producer, "d405").unwrap();
        // With nothing subscribed the gate skips every camera, so no encode
        // durations are ever recorded. The one-second metrics window must
        // still summarise them without indexing an empty sample buffer.
        let deadline = std::time::Instant::now() + std::time::Duration::from_millis(1600);
        while std::time::Instant::now() < deadline {
            publisher.submit(&set).unwrap();
            std::thread::sleep(std::time::Duration::from_millis(20));
        }
        publisher.submit(&set).unwrap();
        // A panicking worker is silent: it sets no error and `submit` keeps
        // succeeding, so assert on the thread itself rather than the channel.
        assert!(
            !publisher
                .worker
                .as_ref()
                .expect("worker handle")
                .is_finished(),
            "the worker thread must survive a fully gated metrics window"
        );
    }

    #[test]
    fn publisher_matching_status_tracks_the_cockpit_wildcard() {
        use zenoh::Wait;
        // The publish worker skips downscaling and JPEG encoding whenever a
        // camera's publisher reports no match. That saving is only safe if a
        // real viewer makes the publisher match, so pin the exact key
        // expression the cockpit subscribes to (`view_bus`).
        let bus = tatbot_bus::transport::Bus::open(&[]).unwrap();
        let publisher = bus
            .session
            .declare_publisher("tatbot/vision/d405/realsense2")
            .wait()
            .unwrap();
        assert!(
            !publisher.matching_status().wait().unwrap().matching(),
            "an unsubscribed publisher must not report a match"
        );

        let _subscriber = bus
            .session
            .declare_subscriber("tatbot/vision/*/*")
            .wait()
            .unwrap();
        // Matching state propagates asynchronously through the session.
        let mut matched = false;
        for _ in 0..100 {
            if publisher.matching_status().wait().unwrap().matching() {
                matched = true;
                break;
            }
            std::thread::sleep(std::time::Duration::from_millis(20));
        }
        assert!(
            matched,
            "the cockpit's wildcard must make a per-camera publisher match, \
             otherwise the gate would starve a live viewer"
        );
    }

    #[test]
    fn luma_frames_cross_the_bus_as_grayscale_without_color_expansion() {
        let (producer, mut set) = fixture();
        let color = set.frames.get_mut("color").unwrap();
        color.metadata.profile.format = PixelFormat::Y8;
        color.payload = RecordedPayload::Video {
            format: PixelFormat::Y8,
            width: 2,
            height: 2,
            bytes: vec![80, 80, 80, 80],
        };
        let (_, decoded) = decode(&encode(&set, &producer).unwrap()).unwrap();
        assert_eq!(decoded.frames["depth"], set.frames["depth"]);
        let color = &decoded.frames["color"];
        assert_eq!(color.metadata.profile.format, PixelFormat::Y8);
        // One channel out, not three: a detector-only pipeline never pays for
        // a colour expansion that the preview would immediately discard.
        assert_eq!(color.payload.bytes().len(), 4);
        assert!(color.payload.bytes().iter().all(|v| v.abs_diff(80) <= 3));

        // The single-channel byte count is enforced, not assumed.
        if let RecordedPayload::Video { bytes, .. } =
            &mut set.frames.get_mut("color").unwrap().payload
        {
            bytes.pop();
        }
        assert!(encode(&set, &producer).is_err());
    }

    #[test]
    fn color_metadata_and_exact_depth_survive_bus_packet() {
        let (producer, set) = fixture();
        let bytes = encode(&set, &producer).unwrap();
        let (p, decoded) = decode(&bytes).unwrap();
        assert_eq!(p, producer);
        assert_eq!(decoded.frames["depth"], set.frames["depth"]);
        assert_eq!(
            decoded.frames["color"].metadata,
            set.frames["color"].metadata
        );
        for (a, b) in decoded.frames["color"]
            .payload
            .bytes()
            .iter()
            .zip(set.frames["color"].payload.bytes())
        {
            assert!(a.abs_diff(*b) <= 2);
        }
        for end in [0, 1, 3, 4, bytes.len() - 1] {
            assert!(decode(&bytes[..end]).is_err());
        }
        let mut trailing = bytes;
        trailing.push(0);
        assert!(decode(&trailing).is_err());
    }
    #[test]
    fn capture_requires_fresh_aligned_metric_depth() {
        let (_, mut set) = fixture();
        set.timestamp_ns = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos() as i128;
        assert!(validate_capture(&set).is_err());
        let depth = set.frames.get_mut("depth").unwrap();
        depth
            .metadata
            .attributes
            .insert("aligned_to".into(), "color".into());
        depth
            .metadata
            .attributes
            .insert("depth_units_m".into(), "0.0001".into());
        assert!(validate_capture(&set).is_ok());
        set.timestamp_ns -= 300_000_000;
        assert!(
            validate_capture(&set)
                .unwrap_err()
                .to_string()
                .contains("stale")
        );
    }

    #[test]
    fn mismatched_color_dimensions_are_refused_before_encoding() {
        let (producer, mut set) = fixture();
        set.frames.get_mut("color").unwrap().metadata.profile.width = 100_000;
        assert!(encode(&set, &producer).is_err());
    }
    #[test]
    fn capture_window_selects_history_and_rejects_straddled_identity() {
        let (_, mut set) = fixture();
        let stamp = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos() as i128
            - 100_000_000;
        set.timestamp_ns = stamp;
        for frame in set.frames.values_mut() {
            frame.metadata.timestamps.normalized_unix_ns = Some(stamp);
        }
        let depth = set.frames.get_mut("depth").unwrap();
        depth
            .metadata
            .attributes
            .insert("aligned_to".into(), "color".into());
        depth
            .metadata
            .attributes
            .insert("depth_units_m".into(), "0.0001".into());
        let window = serde_json::to_vec(&CaptureWindow {
            after_ns: stamp - 1,
            before_ns: stamp + 1,
        })
        .unwrap();
        let mut history = std::collections::VecDeque::from([std::sync::Arc::new(set.clone())]);
        let mut newer = set.clone();
        newer.timestamp_ns += 50_000_000;
        for frame in newer.frames.values_mut() {
            frame.metadata.timestamps.normalized_unix_ns = Some(newer.timestamp_ns);
        }
        history.push_back(std::sync::Arc::new(newer));
        assert_eq!(
            select_capture(&history, Some(&window))
                .unwrap()
                .timestamp_ns,
            stamp
        );
        set.frames.get_mut("depth").unwrap().metadata.sequence += 1;
        assert!(validate_capture_geometry(&set).is_err());
        history.pop_front();
        assert!(select_capture(&history, Some(&window)).is_err());
    }

    #[test]
    fn cockpit_rates_follow_the_key_under_one_overall_cap() {
        let rates = CockpitRates::DEFAULT;
        let ms = |key: &str, cap: f64| preview_interval(key, &rates, cap).as_millis();
        assert_eq!(ms(OVERHEAD_COLOR_KEY, 5.0), 200);
        assert_eq!(ms(OVERHEAD_DEPTH_KEY, 5.0), 1000);
        assert_eq!(ms("tatbot/vision/d405/realsense2", 5.0), 500);
        // A key with no stream rate of its own is held to the cap alone.
        assert_eq!(ms("tatbot/vision/elsewhere/camera1", 5.0), 200);
        // The ad hoc cockpit's "normal" quality caps every key at 2 Hz.
        assert_eq!(ms(OVERHEAD_COLOR_KEY, 2.0), 500);
        assert_eq!(ms(OVERHEAD_DEPTH_KEY, 2.0), 1000);
        // A stream's flag sets its own rate, still beneath the cap.
        let rates = CockpitRates { overhead_depth_hz: 4.0, ..CockpitRates::DEFAULT };
        assert_eq!(preview_interval(OVERHEAD_DEPTH_KEY, &rates, 5.0).as_millis(), 250);
        assert_eq!(preview_interval(OVERHEAD_DEPTH_KEY, &rates, 2.0).as_millis(), 500);
    }

    #[test]
    fn due_camera_samples_go_out_one_a_tick_and_the_newest_of_a_key_waits() {
        let mut paced = Paced::default();
        // The D555's colour and depth are due in the same tick.
        paced.extend([(OVERHEAD_COLOR_KEY.to_string(), 1), (OVERHEAD_DEPTH_KEY.to_string(), 1)]);
        assert_eq!(paced.next(), Some((OVERHEAD_COLOR_KEY.to_string(), 1)));
        // Before its tick the depth waiting is replaced, keeping its turn.
        paced.extend([(OVERHEAD_DEPTH_KEY.to_string(), 2), (OVERHEAD_COLOR_KEY.to_string(), 2)]);
        assert_eq!(paced.next(), Some((OVERHEAD_DEPTH_KEY.to_string(), 2)));
        assert_eq!(paced.next(), Some((OVERHEAD_COLOR_KEY.to_string(), 2)));
        assert_eq!(paced.next(), None);
    }

    #[test]
    fn a_rate_gate_keeps_a_jittery_owner_at_its_rate_and_caps_a_fast_one() {
        let start = std::time::Instant::now();
        let at = |ms: u64| start + std::time::Duration::from_millis(ms);
        // A 5 Hz owner whose frames land a little early or late, read at 5 Hz:
        // once the first full interval has passed, not one frame is lost.
        let mut gate = RateGate::new(std::time::Duration::from_millis(200));
        assert!(gate.admit(at(0)) && gate.admit(at(200)));
        let jittery = [399, 601, 798, 1000, 1203, 1399, 1600];
        assert!(jittery.iter().all(|ms| gate.admit(at(*ms))));
        // A 30 Hz source behind the same gate: five or six a second, never thirty.
        let mut gate = RateGate::new(std::time::Duration::from_millis(200));
        let taken = (0..60).filter(|i| gate.admit(at(i * 1000 / 30))).count();
        assert!((10..=12).contains(&taken), "{taken} of 60 in two seconds");
    }

    #[test]
    fn a_feed_takes_due_keys_and_leaves_the_rest_to_be_replaced() {
        let start = std::time::Instant::now();
        let slots = std::sync::Mutex::new(std::collections::BTreeMap::from([
            (OVERHEAD_COLOR_KEY.to_string(), 1),
            (OVERHEAD_DEPTH_KEY.to_string(), 1),
        ]));
        let mut feed = Feed::default();
        let rates = CockpitRates { overhead_depth_hz: 2.0, ..CockpitRates::DEFAULT };
        let interval = |key: &str| preview_interval(key, &rates, 5.0);
        assert_eq!(feed.take_due(&slots, interval, start).len(), 2);
        slots.lock().unwrap().insert(OVERHEAD_COLOR_KEY.into(), 2);
        slots.lock().unwrap().insert(OVERHEAD_DEPTH_KEY.into(), 2);
        let later = start + std::time::Duration::from_millis(250);
        let taken = feed.take_due(&slots, interval, later);
        assert_eq!(taken, [(OVERHEAD_COLOR_KEY.to_string(), 2)]);
        // Depth is not due at 2 Hz yet; a newer sample replaces the waiting one.
        slots.lock().unwrap().insert(OVERHEAD_DEPTH_KEY.into(), 3);
        let due = start + std::time::Duration::from_millis(520);
        let taken = feed.take_due(&slots, interval, due);
        assert_eq!(taken, [(OVERHEAD_DEPTH_KEY.to_string(), 3)]);
    }

    #[test]
    fn skipped_samples_are_counted_and_reported_at_most_every_ten_seconds() {
        let start = std::time::Instant::now();
        let mut skipped = SkipLog::default();
        assert_eq!(skipped.report(start, false), None);
        skipped.note("tatbot/vision/d405/realsense2", &anyhow::anyhow!("bad header"));
        let first = skipped.report(start, false).unwrap();
        assert!(first.contains("skipped 1 sample(s) (1 in all)"), "{first}");
        assert!(first.contains("realsense2: bad header"), "{first}");
        skipped.note("k", &anyhow::anyhow!("again"));
        skipped.note("k", &anyhow::anyhow!("and again"));
        let soon = start + std::time::Duration::from_secs(3);
        assert_eq!(skipped.report(soon, false), None);
        let later = start + std::time::Duration::from_secs(11);
        let line = skipped.report(later, false).unwrap();
        assert!(line.contains("skipped 2 sample(s) (3 in all); last k: and again"), "{line}");
        assert_eq!(skipped.report(later, true), None, "nothing pending");
    }

    #[cfg(feature = "rerun")]
    fn camera_frame(
        name: &str,
        format: PixelFormat,
        (width, height): (u32, u32),
        payload: RecordedPayload,
        attributes: &[(&str, String)],
    ) -> FrameRecord {
        let (_, set) = fixture();
        let mut metadata = set.frames["color"].metadata.clone();
        metadata.sensor_name = name.into();
        metadata.profile.width = width;
        metadata.profile.height = height;
        metadata.profile.format = format;
        metadata.timestamps.normalized_unix_ns = Some(1_000_000_000);
        metadata.attributes = attributes
            .iter()
            .map(|(key, value)| ((*key).to_string(), value.clone()))
            .collect();
        FrameRecord { metadata, payload }
    }

    #[cfg(feature = "rerun")]
    fn of<'a>(chunks: &'a [rerun::log::Chunk], entity: &str) -> Vec<&'a rerun::log::Chunk> {
        chunks
            .iter()
            .filter(|chunk| chunk.entity_path().to_string() == entity)
            .collect()
    }

    /// What one image chunk may take on the hub's stream: 0.5 ms at 1 Gbps.
    #[cfg(feature = "rerun")]
    const STREAM_CHUNK_MAX: usize = 64 * 1024;

    /// Every message a test recording holds, as chunks with the bytes each
    /// takes framed and LZ4-compressed: what the hub's stream carries.
    #[cfg(feature = "rerun")]
    fn sized_chunks(storage: &rerun::sink::MemorySinkStorage) -> Vec<(rerun::log::Chunk, usize)> {
        storage
            .take()
            .into_iter()
            .filter_map(|message| {
                let rerun::log::LogMsg::ArrowMsg(_, arrow) = &message else {
                    return None;
                };
                let mut encoder = re_log_encoding::Encoder::new_eager(
                    re_log_encoding::CrateVersion::LOCAL,
                    re_log_encoding::EncodingOptions::PROTOBUF_COMPRESSED,
                    Vec::new(),
                )
                .unwrap();
                let bytes = encoder.append(&message).unwrap() as usize;
                Some((rerun::log::Chunk::from_arrow_msg(arrow).unwrap(), bytes))
            })
            .collect()
    }

    /// A recording as the live cockpit's stream chunks it: the cockpit logs
    /// an image a tick and the hub's batcher flushes within one, so each
    /// image row is a chunk of its own.
    #[cfg(feature = "rerun")]
    fn cockpit_recording() -> (crate::RerunViewer, rerun::sink::MemorySinkStorage) {
        let (rec, storage) = rerun::RecordingStreamBuilder::new(crate::rerun_viewer::APP_ID)
            .batcher_config(rerun::log::ChunkBatcherConfig::ALWAYS_TEST_ONLY)
            .memory()
            .unwrap();
        let mut viewer = crate::RerunViewer::from_recording(rec);
        viewer.set_compressed_depth(true);
        (viewer, storage)
    }

    /// A deterministic xorshift, for test pictures.
    #[cfg(feature = "rerun")]
    fn random(seed: u32) -> impl FnMut() -> u32 {
        let mut state = seed;
        move || {
            state ^= state << 13;
            state ^= state >> 17;
            state ^= state << 5;
            state
        }
    }

    /// A 1280x720 colour scene as a JPEG of `quality`: a table lit across,
    /// a ruled page, a dark arm, a little sensor grain; or, with `noise`,
    /// nothing but noise, the most a JPEG can be asked to hold.
    #[cfg(feature = "rerun")]
    fn scene_jpeg(quality: u8, noise: bool) -> Vec<u8> {
        let mut next = random(0x2545_f491);
        let mut rgb = Vec::with_capacity(1280 * 720 * 3);
        for y in 0..720_i32 {
            for x in 0..1280_i32 {
                let page = (300..900).contains(&x) && (200..560).contains(&y);
                let base = if noise {
                    [next() as u8, next() as u8, next() as u8].map(i32::from)
                } else if (950..1100).contains(&x) && y < 500 {
                    [40, 42, 48]
                } else if page && (y % 24 < 2 || (x * 3 + y) % 97 < 2) {
                    [30, 30, 40]
                } else if page {
                    [225, 222, 215]
                } else {
                    let lit = 120 + x / 12 + y / 20;
                    [lit, lit - 20, lit - 45]
                };
                let grain = (next() % 9) as i32 - 4;
                rgb.extend(base.map(|v| (v + grain).clamp(0, 255) as u8));
            }
        }
        let mut jpeg = Vec::new();
        image::codecs::jpeg::JpegEncoder::new_with_quality(&mut jpeg, quality)
            .encode(&rgb, 1280, 720, image::ExtendedColorType::Rgb8)
            .unwrap();
        jpeg
    }

    /// Where the depth scene's needle is: one pixel wide, 0.30 m away.
    #[cfg(feature = "rerun")]
    const NEEDLE: (u32, std::ops::Range<u32>) = (641, 200..400);

    /// A 1280x720 z16 plane in millimetres as the D555 sees the rig: the
    /// room beyond the table at 1.6-4.2 m along the top, the table sloping
    /// from 0.45 to 0.80 m with 2 mm of noise and a hole every 50 samples,
    /// and the needle (`NEEDLE`) in front of it.
    #[cfg(feature = "rerun")]
    fn scene_depth() -> Vec<u8> {
        let mut next = random(7);
        let mut plane = Vec::with_capacity(1280 * 720 * 2);
        for y in 0..720_u32 {
            for x in 0..1280_u32 {
                let range = if x == NEEDLE.0 && NEEDLE.1.contains(&y) {
                    300
                } else if y < 120 {
                    1600 + 2 * x
                } else if next() % 50 == 0 {
                    0
                } else {
                    450 + (y - 120) * 350 / 600 + next() % 5 - 2
                };
                plane.extend_from_slice(&(range as u16).to_le_bytes());
            }
        }
        plane
    }

    #[cfg(feature = "rerun")]
    fn overhead_set(name: &str, payload: RecordedPayload, intrinsics: &str) -> SynchronizedFrameSet {
        let (format, mut attributes) = match payload {
            RecordedPayload::Depth { .. } => (
                PixelFormat::Z16,
                vec![("aligned_to", "overhead_depth_color".to_string()), ("depth_units_m", "0.001".into())],
            ),
            _ => (PixelFormat::Rgb8, Vec::new()),
        };
        attributes.push(("intrinsics", intrinsics.to_string()));
        SynchronizedFrameSet {
            sequence: 4,
            timestamp_basis: "normalized".into(),
            timestamp_ns: 1_000_000_000,
            maximum_skew_ns: 0,
            frames: std::collections::BTreeMap::from([(
                name.to_string(),
                camera_frame(name, format, (1280, 720), payload, &attributes),
            )]),
        }
    }

    #[cfg(all(feature = "rerun", feature = "fiducials"))]
    #[test]
    fn the_cockpit_outlines_a_scene_targets_tag_it_reads_in_the_overhead_colour_once_a_second() {
        // The palette's roof tag no tracker follows: the cockpit reads its
        // target in the overhead colour's full frame (here the fixture's
        // wrist tag 3, 240 px a side at (360, 40) pasted 200 px down), and
        // outlines it in the logged 640x360's pixels, labelled with the target.
        let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
        let inventory = root.join("config/fiducials.json");
        if !inventory.exists() {
            return; // the deployment's inventory
        }
        let tags = image::open(root.join("scripts/tests/fixtures/fiducials/mixed.png")).unwrap().to_rgb8();
        let mut canvas = image::RgbImage::from_pixel(1280, 720, image::Rgb([200, 200, 200]));
        image::imageops::replace(&mut canvas, &tags, 0, 200);
        let intrinsics = serde_json::json!({"schema": "tatbot.camera-intrinsics/1",
            "width": 1280, "height": 720, "fx": 900.0, "fy": 900.0, "ppx": 640.0, "ppy": 360.0})
        .to_string();
        let wire = |picture: &image::RgbImage| {
            let mut jpeg = Vec::new();
            image::codecs::jpeg::JpegEncoder::new_with_quality(&mut jpeg, 95)
                .encode(picture, 1280, 720, image::ExtendedColorType::Rgb8)
                .unwrap();
            let payload = RecordedPayload::Encoded { format: PixelFormat::Jpeg, bytes: jpeg };
            crate::transport::encode_frame_set(&overhead_set("overhead_depth_color", payload, &intrinsics), None).unwrap()
        };
        let (viewer, storage) = cockpit_recording();
        let mut reader = SceneTags::new(&inventory, "wrist").unwrap();
        let (seen, blank) = (wire(&canvas), wire(&image::RgbImage::from_pixel(1280, 720, image::Rgb([200, 200, 200]))));
        let start = std::time::Instant::now();
        log_frame_sample(&viewer, OVERHEAD_COLOR_KEY, &seen, None).unwrap();
        reader.read(&viewer, &seen, start).unwrap();
        reader.read(&viewer, &blank, start + std::time::Duration::from_millis(100)).unwrap(); // not due
        viewer.finish().unwrap();
        let chunks: Vec<_> = sized_chunks(&storage).into_iter().map(|(chunk, _)| chunk).collect();
        let entity = "/cameras/08_overhead_depth/color/fiducials/wrist";
        let strips: Vec<Vec<[f32; 2]>> = of(&chunks, entity)
            .iter()
            .flat_map(|chunk| {
                chunk
                    .iter_component::<rerun::components::LineStrip2D>(rerun::LineStrips2D::descriptor_strips().component)
                    .flat_map(|row| row.iter().map(|s| s.0.iter().map(|p| p.0).collect::<Vec<_>>()).collect::<Vec<_>>())
                    .collect::<Vec<_>>()
            })
            .collect();
        assert_eq!(strips.len(), 1, "one tag outlined, once: the second sample was not due");
        let near = |[x, y]: [f32; 2]| strips[0].iter().any(|[u, v]| (u - x).hypot(v - y) < 1.5);
        for corner in [[360.0, 240.0], [599.0, 240.0], [599.0, 479.0], [360.0, 479.0]] {
            assert!(near(corner.map(|v: f32| (v + 0.5) * 0.5)), "{corner:?} not in {strips:?}");
        }
        let labels: Vec<String> = of(&chunks, &format!("{entity}/id"))
            .iter()
            .flat_map(|chunk| {
                chunk
                    .iter_component::<rerun::components::Text>(rerun::Points2D::descriptor_labels().component)
                    .flat_map(|row| row.iter().map(|t| t.0.to_string()).collect::<Vec<_>>())
                    .collect::<Vec<_>>()
            })
            .collect();
        assert_eq!(labels, ["wrist"]);
        // A due sample without the tag clears its outline.
        let (viewer, storage) = cockpit_recording();
        log_frame_sample(&viewer, OVERHEAD_COLOR_KEY, &blank, None).unwrap();
        reader.read(&viewer, &blank, start + SceneTags::PERIOD).unwrap();
        viewer.finish().unwrap();
        let chunks: Vec<_> = sized_chunks(&storage).into_iter().map(|(chunk, _)| chunk).collect();
        assert!(of(&chunks, entity).iter().any(|chunk| chunk
            .component_descriptors()
            .any(|d| d.component == rerun::Clear::descriptor_is_recursive().component)));
    }

    #[cfg(feature = "rerun")]
    #[test]
    fn the_trackers_fiducials_are_drawn_on_the_overhead_colour_where_they_were_read() {
        // The cockpit's own path: the overhead colour at its row cap, then a
        // tracking pose and a print as trackd and stencild publish them. Each
        // tracker's reads are outlined in the logged image's pixels at the
        // capture they were read in; a print lost clears its markers.
        let intrinsics = serde_json::json!({"schema": "tatbot.camera-intrinsics/1",
            "width": 1280, "height": 720, "fx": 640.0, "fy": 641.0, "ppx": 638.0, "ppy": 362.0})
        .to_string();
        let color = overhead_set(
            "overhead_depth_color",
            RecordedPayload::Encoded { format: PixelFormat::Jpeg, bytes: scene_jpeg(90, false) },
            &intrinsics,
        );
        let (viewer, storage) = cockpit_recording();
        log_frame_sample(&viewer, OVERHEAD_COLOR_KEY, &crate::transport::encode_frame_set(&color, None).unwrap(), None)
            .unwrap();
        let envelope = |schema: &str, wall_ns: u64, payload: serde_json::Value| {
            serde_json::to_vec(&Envelope {
                schema: schema.into(),
                producer: Producer { node: "ros".into(), pid: 1, sha: "sha".into(), run_id: "run".into() },
                stamp: Stamp { mono_ns: 0, wall_ns, basis: "normalized".into() },
                seq: 1,
                payload,
            })
            .unwrap()
        };
        let tracking = serde_json::json!({
            "target": "wrist", "source": "measured", "tag_ids": [2, 3], "cameras": ["overhead_depth_color"],
            "calibration_id": "bundle", "pose": null,
            "estimate": {"detections": {"overhead_depth_color": [
                {"tag_id": 2, "corners_px": [[953.5, 297.5], [917.5, 287.5], [905.5, 223.5], [941.5, 237.5]]},
                {"tag_id": 3, "corners_px": [[999.5, 301.5], [1019.5, 301.5], [1019.5, 321.5], [999.5, 321.5]]}]}}
        });
        let pattern = format!("stencil-{}", "c".repeat(64));
        let print = |source: &str| serde_json::json!({
            "target_id": pattern, "source": source, "calibration_id": "bundle", "world_from_target": null,
            "support": {"anchor_camera": "overhead_depth_color", "markers": {"ids": [11]},
                        "anchor_pixels": [[599.5, 399.5], [615.5, 399.5], [615.5, 415.5], [599.5, 415.5]]}
        });
        type Log = fn(&crate::RerunViewer, u64, &serde_json::Value) -> Result<()>;
        let samples: [(&str, u64, serde_json::Value, Log); 3] = [
            ("tatbot.tracking-pose/1", 2_000_000_000, tracking, crate::RerunViewer::log_tracking),
            ("tatbot.target-pose/1", 2_100_000_000, print("measured"), crate::RerunViewer::log_target),
            ("tatbot.target-pose/1", 3_100_000_000, print("lost"), crate::RerunViewer::log_target),
        ];
        for (schema, stamp, payload, log) in samples {
            log_pose_sample(&viewer, &envelope(schema, stamp, payload), schema, log).unwrap();
        }
        viewer.finish().unwrap();
        let chunks: Vec<_> = sized_chunks(&storage).into_iter().map(|(chunk, _)| chunk).collect();
        let strips = |entity: &str| -> Vec<(i64, Vec<[f32; 2]>)> {
            of(&chunks, entity)
                .iter()
                .flat_map(|chunk| {
                    let at = chunk
                        .timelines()
                        .iter()
                        .find(|(name, _)| name.as_str() == crate::rerun_viewer::TIMELINE)
                        .map(|(_, column)| column.times_raw().to_vec())
                        .unwrap_or_default();
                    chunk
                        .iter_component::<rerun::components::LineStrip2D>(
                            rerun::LineStrips2D::descriptor_strips().component,
                        )
                        .zip(at)
                        .flat_map(|(row, at)| row.iter().map(|s| (at, s.0.iter().map(|p| p.0).collect())).collect::<Vec<_>>())
                        .collect::<Vec<_>>()
                })
                .collect()
        };
        let image = "/cameras/08_overhead_depth/color";
        // The row cap logs the 1280x720 colour at 640x360: every corner (+0.5) halved.
        let ee = strips(&format!("{image}/fiducials/ee/wrist"));
        assert_eq!(ee.len(), 2);
        assert!(ee.iter().all(|(at, _)| *at == 2_000_000_000));
        assert_eq!(ee[0].1, [[477.0, 149.0], [459.0, 144.0], [453.0, 112.0], [471.0, 119.0], [477.0, 149.0]]);
        let labels: Vec<String> = of(&chunks, &format!("{image}/fiducials/ee/wrist/id"))
            .iter()
            .flat_map(|chunk| {
                chunk
                    .iter_component::<rerun::components::Text>(rerun::Points2D::descriptor_labels().component)
                    .flat_map(|row| row.iter().map(|t| t.0.to_string()).collect::<Vec<_>>())
                    .collect::<Vec<_>>()
            })
            .collect();
        assert_eq!(labels, ["2", "3"]);
        let markers = format!("{image}/fiducials/print/{pattern}");
        assert_eq!(strips(&markers), [(2_100_000_000, vec![[300.0, 200.0], [308.0, 200.0], [308.0, 208.0], [300.0, 208.0], [300.0, 200.0]])]);
        let cleared: Vec<i64> = of(&chunks, &markers)
            .iter()
            .filter(|chunk| chunk.component_descriptors().any(|d| d.component == rerun::Clear::descriptor_is_recursive().component))
            .flat_map(|chunk| {
                chunk
                    .timelines()
                    .iter()
                    .find(|(name, _)| name.as_str() == crate::rerun_viewer::TIMELINE)
                    .map(|(_, column)| column.times_raw().to_vec())
                    .unwrap_or_default()
            })
            .collect();
        assert_eq!(cleared, [3_100_000_000], "the lost print clears its markers at its stamp");
        // The overlays stay out of the 3D scene: none under the D555's camera entity but the image's.
        assert!(chunks.iter().all(|c| !c.entity_path().to_string().contains("fiducials")
            || c.entity_path().to_string().starts_with(&format!("{image}/fiducials/"))));
    }

    #[cfg(feature = "rerun")]
    #[test]
    fn overhead_frames_fit_the_row_cap_with_pinholes_at_the_logged_size_and_keep_near_edges() {
        let intrinsics = serde_json::json!({"schema": "tatbot.camera-intrinsics/1",
            "width": 1280, "height": 720, "fx": 640.0, "fy": 641.0, "ppx": 638.0, "ppy": 362.0})
        .to_string();
        // What the owner puts on the bus: its q95 colour and raw z16 depth.
        let color = overhead_set(
            "overhead_depth_color",
            RecordedPayload::Encoded { format: PixelFormat::Jpeg, bytes: scene_jpeg(95, false) },
            &intrinsics,
        );
        let depth = overhead_set(
            "overhead_depth_depth",
            RecordedPayload::Depth { width: 1280, height: 720, bytes: scene_depth() },
            &intrinsics,
        );
        let color = crate::transport::encode_frame_set(&color, None).unwrap();
        let depth = crate::transport::encode_frame_set(&depth, None).unwrap();
        assert!(color.len() > 2 * VIEWER_ROW_BYTES && depth.len() > 20 * VIEWER_ROW_BYTES);
        let (viewer, storage) = cockpit_recording();
        for _ in 0..2 {
            log_frame_sample(&viewer, OVERHEAD_COLOR_KEY, &color, None).unwrap();
            log_frame_sample(&viewer, OVERHEAD_DEPTH_KEY, &depth, Some(OVERHEAD_DEPTH_MAX_M)).unwrap();
        }
        viewer.log_ages(3_500_000_000).unwrap();
        viewer.finish().unwrap();
        let sized = sized_chunks(&storage);
        let chunks: Vec<_> = sized.iter().map(|(chunk, _)| chunk.clone()).collect();

        // Each image row, as the stream carries it, holds the link for 0.5 ms at most.
        for entity in [
            "/cameras/08_overhead_depth/color",
            "/cameras/08_overhead_depth/depth",
            "/cameras/08_overhead_depth/cloud/0",
            "/cameras/08_overhead_depth/cloud/1",
            "/cameras/08_overhead_depth/cloud/2",
            "/cameras/08_overhead_depth/cloud/3",
        ] {
            let rows: Vec<usize> = sized
                .iter()
                .filter(|(chunk, _)| chunk.entity_path().to_string() == entity && !chunk.is_static())
                .map(|(_, bytes)| *bytes)
                .collect();
            assert_eq!(rows.len(), 2, "{entity}");
            assert!(rows.iter().all(|bytes| *bytes <= STREAM_CHUNK_MAX), "{entity}: {rows:?}");
        }

        let pinhole = |entity: &str| {
            let statics: Vec<_> = of(&chunks, entity).into_iter().filter(|c| c.is_static()).collect();
            let projection: Vec<[f32; 9]> = statics
                .iter()
                .flat_map(|c| {
                    c.iter_component::<rerun::components::PinholeProjection>(
                        rerun::Pinhole::descriptor_image_from_camera().component,
                    )
                    .flat_map(|rows| rows.iter().map(|p| p.0.0).collect::<Vec<_>>())
                    .collect::<Vec<_>>()
                })
                .collect();
            let resolution: Vec<[f32; 2]> = statics
                .iter()
                .flat_map(|c| {
                    c.iter_component::<rerun::components::Resolution>(
                        rerun::Pinhole::descriptor_resolution().component,
                    )
                    .flat_map(|rows| rows.iter().map(|r| r.0.0).collect::<Vec<_>>())
                    .collect::<Vec<_>>()
                })
                .collect();
            assert_eq!((projection.len(), resolution.len()), (1, 1), "{entity}: one static pinhole for two frames");
            (projection[0], resolution[0])
        };
        let blobs = |entity: &str, component| -> Vec<Vec<u8>> {
            of(&chunks, entity)
                .into_iter()
                .flat_map(|c| {
                    c.iter_component::<rerun::components::Blob>(component)
                        .flat_map(|rows| rows.iter().map(|b| b.0.to_vec()).collect::<Vec<_>>())
                        .collect::<Vec<_>>()
                })
                .collect()
        };

        // Colour: a 640x360 JPEG at the working quality, under a pinhole of that size.
        // Column-major: fx, fy on the diagonal, the principal point in column 3.
        let (projection, resolution) = pinhole("/cameras/08_overhead_depth/color");
        assert_eq!([projection[0], projection[4], projection[6], projection[7]], [320.0, 320.5, 319.0, 181.0]);
        assert_eq!(resolution, [640.0, 360.0]);
        let jpegs = blobs("/cameras/08_overhead_depth/color", rerun::EncodedImage::descriptor_blob().component);
        assert_eq!(jpegs.len(), 2);
        for jpeg in &jpegs {
            assert!(jpeg.len() <= VIEWER_ROW_BYTES);
            let size = image::ImageReader::with_format(std::io::Cursor::new(jpeg), image::ImageFormat::Jpeg)
                .into_dimensions()
                .unwrap();
            assert_eq!(size, (640, 360), "the logged size is the pinhole's");
        }
        let metadata: Vec<String> = of(&chunks, "/diagnostics/overhead_depth_color/metadata")
            .into_iter()
            .flat_map(|c| {
                c.iter_component::<rerun::components::Text>(rerun::TextLog::descriptor_text().component)
                    .flat_map(|rows| rows.iter().map(|t| t.to_string()).collect::<Vec<_>>())
                    .collect::<Vec<_>>()
            })
            .collect();
        assert!(metadata[0].contains(r#""viewer_preview_jpeg_quality":"70""#), "{}", metadata[0]);

        // Depth: 320x180 RVL in millimetres, under a pinhole of that size.
        let (projection, resolution) = pinhole("/cameras/08_overhead_depth/depth");
        assert_eq!([projection[0], projection[4], projection[6], projection[7]], [160.0, 160.25, 159.5, 90.5]);
        assert_eq!(resolution, [320.0, 180.0]);
        let media: Vec<String> = of(&chunks, "/cameras/08_overhead_depth/depth")
            .into_iter()
            .flat_map(|c| {
                c.iter_component::<rerun::components::MediaType>(
                    rerun::EncodedDepthImage::descriptor_media_type().component,
                )
                .flat_map(|rows| rows.iter().map(|m| m.to_string()).collect::<Vec<_>>())
                .collect::<Vec<_>>()
            })
            .collect();
        assert_eq!(media, ["application/rvl", "application/rvl"]);
        let rvl = blobs("/cameras/08_overhead_depth/depth", rerun::EncodedDepthImage::descriptor_blob().component);
        let header = re_rvl::RosRvlMetadata::parse(&rvl[0]).unwrap();
        assert_eq!((header.width, header.height), (320, 180));
        let plane: Vec<u16> = re_rvl::decode_rvl_with_quantization(&rvl[0], &header)
            .unwrap()
            .iter()
            .map(|v| *v as u16)
            .collect();
        let at = |x: u32, y: u32| plane[(y * 320 + x) as usize];
        // The needle, one pixel of a 4x4 block, is what each block it crosses shows.
        let (column, rows) = (NEEDLE.0 / 4, NEEDLE.1.start / 4..NEEDLE.1.end / 4);
        assert!(rows.clone().all(|y| at(column, y) == 300), "the near edge survives");
        assert!((0..30).all(|y| (0..320).all(|x| at(x, y) == 0)), "the room is cut at 0.85 m");
        let needle = |x: u32, y: u32| x == column && rows.contains(&y);
        let table = (30..180).flat_map(|y| (0..320).map(move |x| (x, y)));
        assert!(table.filter(|&(x, y)| !needle(x, y)).all(|(x, y)| (445..=802).contains(&at(x, y))));

        // The 3D scene's RGB cloud: the table's returns in metres, coloured
        // from the latest colour image with the overhead tint, and the depth
        // pinhole again on its own frustum entity.
        assert_eq!(crate::depth_cloud::BANDS, 4, "the row-size loop above names every band");
        let cloud: Vec<_> = (0..crate::depth_cloud::BANDS)
            .flat_map(|band| of(&chunks, &format!("/cameras/08_overhead_depth/cloud/{band}")))
            .collect();
        let positions: Vec<[f32; 3]> = cloud
            .iter()
            .flat_map(|c| {
                c.iter_component::<rerun::components::Position3D>(rerun::Points3D::descriptor_positions().component)
                    .flat_map(|rows| rows.iter().map(|p| p.0.0).collect::<Vec<_>>())
                    .collect::<Vec<_>>()
            })
            .collect();
        let colors: Vec<u32> = cloud
            .iter()
            .flat_map(|c| {
                c.iter_component::<rerun::components::Color>(rerun::Points3D::descriptor_colors().component)
                    .flat_map(|rows| rows.iter().map(|c| c.0.0).collect::<Vec<_>>())
                    .collect::<Vec<_>>()
            })
            .collect();
        assert!(positions.len() > 2 * 1000, "two planes of the table: {}", positions.len());
        assert_eq!(colors.len(), positions.len());
        assert!(positions.iter().all(|p| (0.29..=0.81).contains(&p[2])), "only returns within the cut");
        let tint = crate::depth_cloud::CloudSource::Overhead.tint();
        let untinted = u32::from_be_bytes([tint[0], tint[1], tint[2], 255]);
        assert!(colors.iter().any(|c| *c != untinted), "coloured from the image, not the tint alone");
        assert_eq!(pinhole("/cameras/08_overhead_depth/frustum"), pinhole("/cameras/08_overhead_depth/depth"));

        let labels: Vec<String> = of(&chunks, "/cameras/08_overhead_depth/color/age")
            .into_iter()
            .flat_map(|c| {
                c.iter_component::<rerun::components::Text>(
                    rerun::Points2D::descriptor_labels().component,
                )
                .flat_map(|rows| rows.iter().map(|t| t.to_string()).collect::<Vec<_>>())
                .collect::<Vec<_>>()
            })
            .collect();
        assert_eq!(labels, ["2.5 s old"]);
        let axes: Vec<_> = of(&chunks, "/cameras/08_overhead_depth")
            .into_iter()
            .flat_map(|c| {
                c.iter_component::<rerun::components::ViewCoordinates>(
                    rerun::ViewCoordinates::descriptor_xyz().component,
                )
                .flat_map(|rows| rows.to_vec())
                .collect::<Vec<_>>()
            })
            .collect();
        assert_eq!(axes, [rerun::components::ViewCoordinates::RBD]);
    }

    #[test]
    fn a_colour_image_shrinks_by_block_means_and_luma_stays_luma() {
        #[rustfmt::skip]
        let rgb = image::RgbImage::from_raw(4, 2, vec![
            0, 10, 20,   2, 10, 20,   100, 0, 0,  100, 0, 0,
            4, 10, 20,   6, 11, 20,   100, 0, 255, 101, 0, 255,
        ])
        .unwrap();
        let small = shrink(&image::DynamicImage::ImageRgb8(rgb), [2, 1]);
        assert_eq!(small.as_rgb8().unwrap().as_raw(), &[3, 10, 20, 100, 0, 128]);
        let luma = image::GrayImage::from_raw(3, 3, (0..9).collect()).unwrap();
        let small = shrink(&image::DynamicImage::ImageLuma8(luma), [1, 1]);
        assert_eq!(small.as_luma8().unwrap().as_raw(), &[4]);
        // A size that does not divide evenly is averaged all the same.
        let odd = image::RgbImage::from_pixel(5, 3, image::Rgb([9, 9, 9]));
        let small = shrink(&image::DynamicImage::ImageRgb8(odd), [2, 1]);
        assert_eq!((small.width(), small.height()), (2, 1));
        assert!(small.as_rgb8().unwrap().pixels().all(|p| p.0 == [9, 9, 9]));
    }

    #[cfg(feature = "rerun")]
    #[test]
    fn noise_still_fits_the_row_cap_at_half_the_size_or_a_coarser_block() {
        let color = camera_frame(
            "overhead_depth_color",
            PixelFormat::Rgb8,
            (1280, 720),
            RecordedPayload::Encoded { format: PixelFormat::Jpeg, bytes: scene_jpeg(95, true) },
            &[],
        );
        let fitted = fit_color(&color).unwrap();
        let RecordedPayload::Encoded { bytes, .. } = &fitted.payload else {
            panic!("colour stays a JPEG");
        };
        assert!(bytes.len() <= VIEWER_ROW_BYTES, "{}", bytes.len());
        let size = image::ImageReader::with_format(std::io::Cursor::new(bytes), image::ImageFormat::Jpeg)
            .into_dimensions()
            .unwrap();
        assert_eq!(size, (fitted.metadata.profile.width, fitted.metadata.profile.height));
        assert_eq!(size, (320, 180), "noise fits at no quality at 640x360, at half the size it does");
        // A JPEG that already fits is the owner's, untouched.
        let small = camera_frame(
            "realsense2_color",
            PixelFormat::Bgr8,
            (640, 360),
            fitted.payload.clone(),
            &[],
        );
        assert_eq!(fit_color(&small).unwrap(), small);

        // Depth noise over the whole range, uncut: RVL needs a coarser block.
        let mut next = random(11);
        let noise: Vec<u8> = (0..1280 * 720)
            .flat_map(|_| ((next() % 65535) as u16 + 1).to_le_bytes())
            .collect();
        let depth = camera_frame(
            "overhead_depth_depth",
            PixelFormat::Z16,
            (1280, 720),
            RecordedPayload::Depth { width: 1280, height: 720, bytes: noise },
            &[("depth_units_m", "0.001".into())],
        );
        let fitted = fit_depth(&depth, None);
        let RecordedPayload::Depth { width, height, bytes } = &fitted.payload else {
            panic!("depth stays depth");
        };
        assert!(crate::frame_ops::rvl_depth(*width, *height, bytes).len() <= VIEWER_ROW_BYTES);
        assert_eq!((*width, *height), (160, 90), "one step coarser than the table's 320x180");
        assert_eq!((fitted.metadata.profile.width, fitted.metadata.profile.height), (160, 90));
    }

    #[cfg(feature = "rerun")]
    #[test]
    fn a_wrist_tap_frame_set_logs_its_colour_fitted_and_leaves_its_depth_to_the_bridge() {
        // Contract 6: one set per grab, '<name>_color' JPEG (1280x720 q90) and
        // '<name>_depth' raw z16 in the sensor's units, no bus envelope required.
        let intrinsics = |w: u32, h: u32| {
            serde_json::json!({"fx": 640.0, "fy": 640.0, "ppx": 640.0, "ppy": 360.0,
                "width": w, "height": h})
            .to_string()
        };
        let jpeg = scene_jpeg(90, false);
        let common = |stream: &str| {
            vec![
                ("physical_arm", "right".to_string()),
                ("device_serial", "0000".into()),
                ("capture_owner_role", "ros".into()),
                ("intrinsics", intrinsics(1280, 720)),
                ("ros_frame", format!("right/realsense_{stream}_optical_frame")),
            ]
        };
        let mut depth_attributes = common("depth");
        depth_attributes.push(("depth_units_m", "0.0001".into()));
        depth_attributes.push(("aligned_to", "realsense2_color".into()));
        let set = SynchronizedFrameSet {
            sequence: 9,
            timestamp_basis: "ros".into(),
            timestamp_ns: 1_000_000_000,
            maximum_skew_ns: 0,
            frames: std::collections::BTreeMap::from([
                (
                    "realsense2_color".into(),
                    camera_frame(
                        "realsense2_color",
                        PixelFormat::Bgr8,
                        (1280, 720),
                        RecordedPayload::Encoded { format: PixelFormat::Jpeg, bytes: jpeg },
                        &common("color"),
                    ),
                ),
                (
                    "realsense2_depth".into(),
                    camera_frame(
                        "realsense2_depth",
                        PixelFormat::Z16,
                        (1280, 720),
                        RecordedPayload::Depth {
                            width: 1280,
                            height: 720,
                            bytes: vec![0x10; 1280 * 720 * 2],
                        },
                        &depth_attributes,
                    ),
                ),
            ]),
        };
        let wire = crate::transport::encode_frame_set(&set, None).unwrap();
        let key = "tatbot/vision/d405/realsense2";
        let (viewer, storage) = cockpit_recording();
        // What a cockpit starts with: an earlier release's wrist depth cleared,
        // the Demo pane waiting for the first wrist frame.
        viewer.clear(&RETIRED_ROWS, 400_000_000).unwrap();
        viewer
            .await_image("realsense2_color", WRIST_PLACEHOLDER, [640, 360], 450_000_000)
            .unwrap();
        viewer.log_ages(500_000_000).unwrap();
        log_frame_sample(&viewer, key, &wire, None).unwrap();
        viewer.log_ages(1_000_000_000).unwrap();
        viewer.log_ages(1_500_000_000).unwrap();
        assert!(log_frame_sample(&viewer, key, &wire[..wire.len() - 1], None).is_err());
        assert!(log_frame_sample(&viewer, key, b"not a frame set", None).is_err());
        viewer.finish().unwrap();
        let sized = sized_chunks(&storage);
        let chunks: Vec<_> = sized.iter().map(|(chunk, _)| chunk.clone()).collect();
        let has = |entity: &str, component: rerun::ComponentIdentifier| {
            of(&chunks, entity)
                .iter()
                .any(|c| c.component_descriptors().any(|d| d.component == component))
        };
        let blob = rerun::EncodedImage::descriptor_blob().component;
        let clear = rerun::Clear::descriptor_is_recursive().component;
        let clears = |entity: &str| {
            of(&chunks, entity)
                .iter()
                .filter(|c| c.component_descriptors().any(|d| d.component == clear))
                .count()
        };
        // An earlier cockpit's last frame and its age label, frozen when it
        // stopped, go as the placeholder goes up (never "no frame yet" over a
        // frame): the label cleared, the frame covered by a black one.
        assert_eq!(clears("/cameras/07_realsense2/color/age"), 1);
        // The colour as the D555's: 640x360, one chunk within the cap, after
        // the black frame of that size.
        let rows: Vec<_> = sized
            .iter()
            .filter(|(chunk, _)| chunk.entity_path().to_string() == "/cameras/07_realsense2/color")
            .filter(|(chunk, _)| chunk.component_descriptors().any(|d| d.component == blob))
            .collect();
        assert_eq!(rows.len(), 2);
        assert!(rows[1].1 <= STREAM_CHUNK_MAX, "{}", rows[1].1);
        let jpegs: Vec<Vec<u8>> = rows
            .iter()
            .flat_map(|(chunk, _)| chunk.iter_component::<rerun::components::Blob>(blob))
            .flat_map(|blobs| blobs.iter().map(|b| b.0.to_vec()).collect::<Vec<_>>())
            .collect();
        let blank = image::load_from_memory(&jpegs[0]).unwrap().to_rgb8();
        assert_eq!(blank.dimensions(), (640, 360));
        assert!(blank.pixels().all(|p| p.0.iter().all(|&v| v < 8)));
        let size = image::ImageReader::with_format(std::io::Cursor::new(&jpegs[1]), image::ImageFormat::Jpeg)
            .into_dimensions()
            .unwrap();
        assert_eq!(size, (640, 360));
        assert!(has("/cameras/07_realsense2/color/age", rerun::Points2D::descriptor_labels().component));
        // Its depth, 1.8 MB raw, is not logged: the bridge draws the wrist cloud.
        // The entity holds the start's clear alone, so no older depth is drawn.
        let depth = of(&chunks, "/cameras/07_realsense2/depth");
        assert_eq!(depth.len(), 1);
        assert!(depth[0].component_descriptors().all(|d| d.archetype.as_deref() == Some("rerun.archetypes.Clear")));
        assert!(of(&chunks, "/diagnostics/realsense2_depth/metadata").is_empty());
        // The placeholder until the first frame, then cleared once and not logged again.
        let waiting = of(&chunks, "/cameras/07_realsense2/color/waiting");
        let placeholder: Vec<String> = waiting
            .iter()
            .flat_map(|c| {
                c.iter_component::<rerun::components::Text>(
                    rerun::Points2D::descriptor_labels().component,
                )
                .flat_map(|rows| rows.iter().map(|t| t.to_string()).collect::<Vec<_>>())
                .collect::<Vec<_>>()
            })
            .collect();
        assert_eq!(placeholder, [WRIST_PLACEHOLDER]);
        assert_eq!(clears("/cameras/07_realsense2/color/waiting"), 1);
        // The placeholder outlines the frame to come, at the size it is logged at.
        let outline: Vec<[f32; 2]> = waiting
            .iter()
            .flat_map(|c| {
                c.iter_component::<rerun::components::HalfSize2D>(
                    rerun::Boxes2D::descriptor_half_sizes().component,
                )
                .flat_map(|rows| rows.iter().map(|h| h.0.0).collect::<Vec<_>>())
                .collect::<Vec<_>>()
            })
            .collect();
        assert_eq!(outline, [[320.0, 180.0]]);
        // The wrist's cloud is the bridge's, placed by tf: no pinhole here.
        assert!(!has("/cameras/07_realsense2/color", rerun::Pinhole::descriptor_image_from_camera().component));
    }

    /// The Python tap's golden markers message (`scripts/lib/wrist_tap.py`
    /// `markers_message`; `scripts/tests/test_wrist_tap.py` holds it byte for
    /// byte): a tracker reading's three markers, two kept by its fit.
    #[cfg(feature = "rerun")]
    const PYTHON_MARKERS_GOLDEN: &[u8] = include_bytes!(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../scripts/tests/fixtures/wrist_markers_golden.json"
    ));

    #[cfg(feature = "rerun")]
    #[test]
    fn the_python_markers_golden_decodes_and_another_schema_is_refused() {
        let read = decode_markers(PYTHON_MARKERS_GOLDEN).unwrap();
        assert_eq!(read.sensor, "realsense2_color");
        assert_eq!(read.frame_ns, 1_800_000_000_123_456_789);
        assert_eq!((read.size, read.reading.as_str(), read.accepted), ([1280, 720], "s0075-hover", true));
        let ids: Vec<(u32, bool)> = read.markers.iter().map(|m| (m.id, m.fit)).collect();
        assert_eq!(ids, [(4, true), (7, true), (9, false)]);
        assert_eq!(read.markers[1].corners_px[0], [300.3, 210.0]);
        let text = std::str::from_utf8(PYTHON_MARKERS_GOLDEN).unwrap();
        let other = text.replace("tatbot.wrist-markers/1", "tatbot.frame-set/1");
        assert!(decode_markers(other.as_bytes()).is_err());
        assert!(decode_markers(b"not json").is_err());
    }

    /// A wrist frame set as the tap publishes it, at `stamp_ns`: its 1280x720
    /// colour JPEG, which the cockpit logs at 640x360.
    #[cfg(feature = "rerun")]
    fn wrist_wire(sequence: u64, stamp_ns: i128) -> Vec<u8> {
        let intrinsics = serde_json::json!({"fx": 640.0, "fy": 640.0, "ppx": 640.0, "ppy": 360.0,
            "width": 1280, "height": 720})
        .to_string();
        let set = SynchronizedFrameSet {
            sequence,
            timestamp_basis: "ros".into(),
            timestamp_ns: stamp_ns,
            maximum_skew_ns: 0,
            frames: std::collections::BTreeMap::from([(
                "realsense2_color".into(),
                camera_frame(
                    "realsense2_color",
                    PixelFormat::Bgr8,
                    (1280, 720),
                    RecordedPayload::Encoded { format: PixelFormat::Jpeg, bytes: scene_jpeg(90, false) },
                    &[("intrinsics", intrinsics)],
                ),
            )]),
        };
        crate::transport::encode_frame_set(&set, None).unwrap()
    }

    #[cfg(feature = "rerun")]
    #[test]
    fn a_readings_markers_are_drawn_on_its_frame_in_the_logged_pixels_until_the_next_frame() {
        // Contract: a reading's markers come after its frame (the worker reads
        // it 0.6 s later, 2026-10-20), stamped as the frame; the cockpit logs
        // them at that stamp, scaled to its 640x360, and every wrist frame
        // hides the markers before it from just before its own stamp.
        let key = "tatbot/vision/d405/realsense2";
        let (first, second) = (1_800_000_000_i128, 1_808_500_000_i128);
        let (viewer, storage) = cockpit_recording();
        viewer
            .await_image("realsense2_color", WRIST_PLACEHOLDER, [640, 360], 1_000_000_000)
            .unwrap();
        log_frame_sample(&viewer, key, &wrist_wire(1, first), None).unwrap();
        let mut read = decode_markers(PYTHON_MARKERS_GOLDEN).unwrap();
        read.frame_ns = first as i64;
        viewer.log_markers(&read).unwrap();
        log_frame_sample(&viewer, key, &wrist_wire(2, second), None).unwrap();
        viewer.finish().unwrap();
        let chunks: Vec<_> = sized_chunks(&storage).into_iter().map(|(chunk, _)| chunk).collect();
        let entity = "/cameras/07_realsense2/color/markers";
        let at = |chunk: &rerun::log::Chunk| -> Vec<i64> {
            chunk
                .timelines()
                .iter()
                .find(|(name, _)| name.as_str() == crate::rerun_viewer::TIMELINE)
                .map(|(_, column)| column.times_raw().to_vec())
                .unwrap_or_default()
        };
        let clear = rerun::Clear::descriptor_is_recursive().component;
        let cleared: Vec<i64> = of(&chunks, entity)
            .iter()
            .filter(|c| c.component_descriptors().any(|d| d.component == clear))
            .flat_map(|c| at(c))
            .collect();
        // the cockpit's start, then each frame's instant before
        assert_eq!(cleared, [1_000_000_000, first as i64 - 1, second as i64 - 1]);
        let strips_of = rerun::LineStrips2D::descriptor_strips().component;
        let drawn: Vec<_> = of(&chunks, entity)
            .into_iter()
            .filter(|c| c.component_descriptors().any(|d| d.component == strips_of))
            .collect();
        assert_eq!(drawn.len(), 1);
        assert_eq!(at(drawn[0]), [first as i64], "at its frame's stamp");
        let strips: Vec<Vec<[f32; 2]>> = drawn[0]
            .iter_component::<rerun::components::LineStrip2D>(strips_of)
            .flat_map(|rows| rows.iter().map(|s| s.0.iter().map(|p| p.0).collect::<Vec<_>>()).collect::<Vec<_>>())
            .collect();
        assert_eq!(strips.len(), 3);
        // closed quads, the detector's pixel centres onto the 640x360 image's
        assert_eq!(strips[0], [[50.25, 100.25], [72.25, 101.25], [71.25, 123.25], [49.25, 122.25], [50.25, 100.25]]);
        let colors: Vec<u32> = drawn[0]
            .iter_component::<rerun::components::Color>(rerun::LineStrips2D::descriptor_colors().component)
            .flat_map(|rows| rows.iter().map(|c| c.0.0).collect::<Vec<_>>())
            .collect();
        let rgba = |[r, g, b]: [u8; 3]| u32::from_be_bytes([r, g, b, 255]);
        assert_eq!(colors, [rgba([90, 220, 90]), rgba([90, 220, 90]), rgba([235, 80, 70])]);
        let labels = |path: &str| -> Vec<String> {
            of(&chunks, path)
                .iter()
                .flat_map(|c| {
                    c.iter_component::<rerun::components::Text>(rerun::Points2D::descriptor_labels().component)
                        .flat_map(|rows| rows.iter().map(|t| t.to_string()).collect::<Vec<_>>())
                        .collect::<Vec<_>>()
                })
                .collect()
        };
        assert_eq!(labels(&format!("{entity}/id")), ["4", "7", "9"]);
        assert_eq!(labels(&format!("{entity}/reading")), ["s0075-hover: 2 of 3 markers used"]);
    }

    #[cfg(feature = "rerun")]
    #[test]
    fn a_readings_markers_that_come_after_the_next_frame_are_drawn_on_the_frame_shown() {
        // Contract: a stroke's readings come every 0.5 s and take ~0.4 s to
        // fit (2026-10-24), so most come after the next frame, whose clear is
        // already logged: drawn at their own stamp they never showed live.
        // They go on the frame shown, at its stamp, until the frame after.
        let key = "tatbot/vision/d405/realsense2";
        let (first, second) = (1_800_000_000_i128, 1_800_500_000_i128);
        let (viewer, storage) = cockpit_recording();
        viewer
            .await_image("realsense2_color", WRIST_PLACEHOLDER, [640, 360], 1_000_000_000)
            .unwrap();
        log_frame_sample(&viewer, key, &wrist_wire(1, first), None).unwrap();
        log_frame_sample(&viewer, key, &wrist_wire(2, second), None).unwrap();
        let mut read = decode_markers(PYTHON_MARKERS_GOLDEN).unwrap();
        read.frame_ns = first as i64;
        viewer.log_markers(&read).unwrap();
        viewer.finish().unwrap();
        let chunks: Vec<_> = sized_chunks(&storage).into_iter().map(|(chunk, _)| chunk).collect();
        let entity = "/cameras/07_realsense2/color/markers";
        let strips_of = rerun::LineStrips2D::descriptor_strips().component;
        let drawn: Vec<i64> = of(&chunks, entity)
            .into_iter()
            .filter(|c| c.component_descriptors().any(|d| d.component == strips_of))
            .flat_map(|c| {
                c.timelines()
                    .iter()
                    .find(|(name, _)| name.as_str() == crate::rerun_viewer::TIMELINE)
                    .map(|(_, column)| column.times_raw().to_vec())
                    .unwrap_or_default()
            })
            .collect();
        assert_eq!(drawn, [second as i64], "on the frame shown, past its clear");
    }
}
