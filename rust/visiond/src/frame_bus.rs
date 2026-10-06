//! Binary Zenoh frame sets: bounded JSON envelope + JPEG color / exact Z16.
//! Only camera owners call publish. Consumers never instantiate a backend.
use crate::{
    FrameRecord, PixelFormat, RecordedPayload, SynchronizedFrameSet,
    capture_clock::{CaptureClockRequest, CaptureClockSample, CaptureFreshSelection, fresh_tags},
};
use anyhow::{Result, ensure};
use serde::{Deserialize, Serialize};
#[cfg(feature = "rerun")]
use sha2::{Digest, Sha256};
use tatbot_bus::{Envelope, Producer, Stamp};
const LIMIT: usize = 128 * 1024 * 1024;
// Packed YUYV pairs, BT.601 conversion matching the existing viewer path.
fn yuyv_to_rgb(bytes: &[u8]) -> Vec<u8> {
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
                let (pixels, color_type) = if *format == PixelFormat::Y8 {
                    // JPEG carries luma natively. Expanding to RGB here would
                    // undo the saving the Y8 decode exists for.
                    (bytes.clone(), image::ExtendedColorType::L8)
                } else {
                    let mut rgb = if *format == PixelFormat::Yuyv {
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
                        yuyv_to_rgb(bytes)
                    } else {
                        bytes.clone()
                    };
                    if *format == PixelFormat::Bgr8 {
                        for pixel in rgb.chunks_exact_mut(3) {
                            pixel.swap(0, 2);
                        }
                    }
                    (rgb, image::ExtendedColorType::Rgb8)
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
    let (producer, set, _) = decode_inner(bytes, false)?;
    Ok((producer, set))
}

pub fn decode_clocked_capture(
    bytes: &[u8],
) -> Result<(Producer, SynchronizedFrameSet, CaptureClockSample)> {
    let (producer, set, sample) = decode_inner(bytes, false)?;
    Ok((
        producer,
        set,
        sample.ok_or_else(|| anyhow::anyhow!("overhead capture clock sample missing"))?,
    ))
}

fn decode_inner(
    bytes: &[u8],
    retain_jpeg: bool,
) -> Result<(Producer, SynchronizedFrameSet, Option<CaptureClockSample>)> {
    let h = crate::transport::decode_frame_set(bytes)?;
    let envelope: Envelope<serde_json::Value> = serde_json::from_value(
        h.envelope
            .clone()
            .ok_or_else(|| anyhow::anyhow!("missing bus provenance"))?,
    )?;
    ensure!(
        envelope.schema == "tatbot.frame-set/1"
            && envelope.seq == h.sequence
            && envelope.payload["wire_version"] == 1,
        "frame schema or sequence mismatch"
    );
    let capture_clock = serde_json::from_value(envelope.payload["capture_clock"].clone())?;
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
    Ok((
        envelope.producer,
        SynchronizedFrameSet {
            sequence: h.sequence,
            timestamp_basis: h.timestamp_basis,
            timestamp_ns: h.timestamp_ns,
            maximum_skew_ns: h.maximum_skew_ns,
            frames,
        },
        capture_clock,
    ))
}

/// One latest-value slot between capture and JPEG/network work. A slow bus
/// replaces queued input, never creates an unbounded capture backlog.
pub struct FramePublisher {
    dropped: std::sync::Arc<std::sync::atomic::AtomicU64>,
    pending: std::sync::Arc<std::sync::Mutex<Option<std::sync::Arc<SynchronizedFrameSet>>>>,
    stopped: std::sync::Arc<std::sync::atomic::AtomicBool>,
    error: std::sync::Arc<std::sync::Mutex<Option<String>>>,
    worker: Option<std::thread::JoinHandle<()>>,
    tracking_pending: Option<PendingFrames>,
    tracking_worker: Option<std::thread::JoinHandle<()>>,
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
type PendingFrames = std::sync::Arc<std::sync::Mutex<Option<std::sync::Arc<SynchronizedFrameSet>>>>;
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

fn tracking_history_worker(
    pending: PendingFrames,
    history: FrameHistory,
    stopped: std::sync::Arc<std::sync::atomic::AtomicBool>,
    error: std::sync::Arc<std::sync::Mutex<Option<String>>>,
) {
    while !stopped.load(std::sync::atomic::Ordering::Acquire) {
        let set = pending.lock().unwrap().take();
        let Some(set) = set else {
            std::thread::sleep(std::time::Duration::from_millis(2));
            continue;
        };
        match crate::transport::luma_video_set(&set) {
            Ok(gray) => retain_history(&history, std::sync::Arc::new(gray), 64),
            Err(failure) => {
                *error.lock().unwrap() = Some(format!("tracking history: {failure}"));
                break;
            }
        }
    }
}
fn capture_topic(group: &str, node: &str) -> String {
    // Wrist USB ownership may span hosts. An unqualified query could receive
    // either arm's frames; require the caller to select its capture owner.
    if group == "poe" {
        "tatbot/vision/poe/tracking-capture".into()
    } else if group == "d405" {
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
            ["poe", "d405", "overhead-depth"].contains(&group),
            "unknown camera group"
        );
        let bus =
            tatbot_bus::transport::Bus::open(connect, &[]).map_err(|e| anyhow::anyhow!("{e}"))?;
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
        let poe = group == "poe";
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
        // A five-camera original-color set can fill the history byte budget.
        // Preserve full optical resolution as luma for tracking, off the capture
        // loop. Only one pending set survives if conversion cannot keep up.
        let tracking_pending = poe.then(|| std::sync::Arc::new(std::sync::Mutex::new(None)));
        let tracking_worker = tracking_pending.as_ref().map(|pending| {
            let (pending, history, stopped, error) = (
                pending.clone(),
                latest.clone(),
                stopped.clone(),
                error.clone(),
            );
            std::thread::spawn(move || tracking_history_worker(pending, history, stopped, error))
        });
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
            let interval = std::time::Duration::from_secs_f64(if poe || publish_depth {
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
                            let set = if poe {
                                select_tracking_capture(&history, payload.as_deref())
                            } else {
                                select_capture(&history, payload.as_deref())
                            }?;
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
                            let frame = if poe {
                                downscale(frame, 0.25)?
                            } else {
                                frame.clone()
                            };
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
            tracking_pending,
            tracking_worker,
            latest,
            fresh_generation,
            history_capacity: if publish_depth || poe { 64 } else { 1 },
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
        if let Some(pending) = &self.tracking_pending {
            if let Ok(mut slot) = pending.try_lock() {
                *slot = Some(set.clone());
            }
        } else {
            retain_history(&self.latest, set.clone(), self.history_capacity);
        }
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
        if let Some(worker) = self.tracking_worker.take() {
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

#[cfg(feature = "rerun")]
pub fn view_bus(
    connect: &[String],
    viewer_url: &str,
    recording_id: &str,
    max_fps: f64,
    duration_seconds: u64,
    scene: ViewBusScene<'_>,
) -> Result<()> {
    use zenoh::Wait;
    ensure!(
        max_fps.is_finite() && max_fps > 0.0 && max_fps <= 5.0,
        "subscriber viewer cap must be in (0,5] Hz"
    );
    let bus = tatbot_bus::transport::Bus::open(connect, &[]).map_err(|e| anyhow::anyhow!("{e}"))?;
    let latest = std::sync::Arc::new(std::sync::Mutex::new(std::collections::BTreeMap::<
        String,
        zenoh::sample::Sample,
    >::new()));
    let incoming = latest.clone();
    let _frames = bus
        .session
        .declare_subscriber("tatbot/vision/*/*")
        .callback(move |sample| {
            let key = sample.key_expr().as_str();
            if (key.starts_with("tatbot/vision/poe/camera")
                || key.starts_with("tatbot/vision/d405/realsense")
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
    let tracking = std::sync::Arc::new(std::sync::Mutex::new(std::collections::BTreeMap::<
        String,
        zenoh::sample::Sample,
    >::new()));
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
    let targets = std::sync::Arc::new(std::sync::Mutex::new(std::collections::BTreeMap::<
        String,
        zenoh::sample::Sample,
    >::new()));
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
    let viewer = crate::RerunViewer::connect(viewer_url, Some(recording_id))?;
    viewer.log_session_metadata("camera_bus", Some(recording_id), scene.urdf, None)?;
    viewer.log_scene(None, scene.urdf, None)?;
    ensure!(
        scene.calibration.is_none() || scene.robot_world.is_some(),
        "camera calibration preview requires robot-world registration"
    );
    let mut calibration_fingerprint = None;
    if let (Some(calibration), Some(robot_world)) = (scene.calibration, scene.robot_world) {
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
    viewer.log_status("camera bus subscriber; no RTSP or RealSense clients")?;
    let started = std::time::Instant::now();
    let interval = std::time::Duration::from_secs_f64(1.0 / max_fps);
    let mut scene_checked = std::time::Instant::now();
    loop {
        let sets = std::mem::take(&mut *latest.lock().unwrap());
        for sample in sets.into_values() {
            // Keep the owner's validated JPEG bytes. Expanding every preview
            // to raw RGB needlessly fills the viewer's transport queue.
            let (_, set, _) = decode_inner(&sample.payload().to_bytes(), true)?;
            viewer.log_set(&set)?;
        }
        let poses = std::mem::take(&mut *tracking.lock().unwrap());
        for sample in poses.into_values() {
            let value: Envelope<serde_json::Value> =
                serde_json::from_slice(&sample.payload().to_bytes())?;
            ensure!(
                value.schema == "tatbot.tracking-pose/1",
                "tracking schema mismatch"
            );
            viewer.log_tracking(value.stamp.wall_ns, &value.payload)?;
        }
        let poses = std::mem::take(&mut *targets.lock().unwrap());
        for sample in poses.into_values() {
            let value: Envelope<serde_json::Value> =
                serde_json::from_slice(&sample.payload().to_bytes())?;
            ensure!(
                value.schema == "tatbot.target-pose/1",
                "target pose schema mismatch"
            );
            viewer.log_target(value.stamp.wall_ns, &value.payload)?;
        }
        if scene_checked.elapsed() >= std::time::Duration::from_secs(2) {
            if let (Some(calibration), Some(robot_world)) =
                (scene.calibration, scene.robot_world)
            {
                refresh_calibration(
                    &viewer,
                    scene.urdf,
                    calibration,
                    robot_world,
                    &mut calibration_fingerprint,
                )?;
            }
            scene_checked = std::time::Instant::now();
        }
        if duration_seconds > 0 && started.elapsed().as_secs() >= duration_seconds {
            break;
        }
        std::thread::sleep(interval);
    }
    Ok(())
}

/// The arm an end-effector tracking key names: `tatbot/tracking/ee/<arm>`,
/// one topic per physical arm.
#[cfg(any(feature = "rerun", test))]
fn ee_tracking_arm(key: &str) -> Option<&str> {
    let arm = key.strip_prefix("tatbot/tracking/ee/")?;
    ["left", "right"].contains(&arm).then_some(arm)
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

fn downscale(frame: &FrameRecord, scale: f64) -> Result<FrameRecord> {
    let RecordedPayload::Video {
        format,
        width,
        height,
        bytes,
    } = &frame.payload
    else {
        anyhow::bail!("viewer scale requires decoded video");
    };
    let w = ((*width as f64 * scale).round() as u32).max(1);
    let h = ((*height as f64 * scale).round() as u32).max(1);
    // Resizing only reads its source. Borrow the captured pixels rather than
    // allocating another full-resolution image for every preview. Nearest
    // neighbour ignores channel order, so BGR resizes correctly as RGB.
    let resized = match format {
        PixelFormat::Rgb8 | PixelFormat::Bgr8 => {
            let image = image::ImageBuffer::<image::Rgb<u8>, _>::from_raw(
                *width,
                *height,
                bytes.as_slice(),
            )
            .ok_or_else(|| anyhow::anyhow!("color byte count"))?;
            image::imageops::resize(&image, w, h, image::imageops::FilterType::Nearest).into_raw()
        }
        PixelFormat::Y8 => {
            let image = image::ImageBuffer::<image::Luma<u8>, _>::from_raw(
                *width,
                *height,
                bytes.as_slice(),
            )
            .ok_or_else(|| anyhow::anyhow!("luma byte count"))?;
            image::imageops::resize(&image, w, h, image::imageops::FilterType::Nearest).into_raw()
        }
        other => anyhow::bail!("viewer requires RGB/BGR/Y8, got {other:?}"),
    };
    let mut metadata = frame.metadata.clone();
    metadata.profile.width = w;
    metadata.profile.height = h;
    metadata
        .attributes
        .insert("bus.scale".into(), scale.to_string());
    metadata
        .attributes
        .insert("bus.source_width".into(), width.to_string());
    metadata
        .attributes
        .insert("bus.source_height".into(), height.to_string());
    Ok(FrameRecord {
        metadata,
        payload: RecordedPayload::Video {
            format: *format,
            width: w,
            height: h,
            bytes: resized,
        },
    })
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

/// Original-resolution decoded RGB/luma for appearance tracking only. Each
/// sensor is selected independently; no claim of a synchronized RGB-D set.
fn select_tracking_capture(
    history: &std::collections::VecDeque<std::sync::Arc<SynchronizedFrameSet>>,
    payload: Option<&[u8]>,
) -> Result<std::sync::Arc<SynchronizedFrameSet>> {
    let now = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)?
        .as_nanos() as i128;
    let bounds = if let Some(bytes) = payload {
        ensure!(bytes.len() <= 1024, "tracking window exceeds budget");
        serde_json::from_slice::<CaptureWindow>(bytes)?
    } else {
        CaptureWindow {
            after_ns: now - 250_000_000,
            before_ns: now,
        }
    };
    ensure!(
        bounds.after_ns > 0
            && bounds.after_ns <= bounds.before_ns
            && bounds.before_ns <= now
            && now - bounds.after_ns <= 3_000_000_000,
        "invalid or expired tracking window"
    );
    let middle = bounds.after_ns + (bounds.before_ns - bounds.after_ns) / 2;
    let mut selected = std::collections::BTreeMap::<String, &FrameRecord>::new();
    for frame in history.iter().flat_map(|set| set.frames.values()) {
        let Some(stamp) = frame.metadata.timestamps.normalized_unix_ns else {
            continue;
        };
        if !matches!(frame.payload, RecordedPayload::Video { .. })
            || frame.metadata.sensor_kind != crate::SensorKind::PoE
            || !(bounds.after_ns..=bounds.before_ns).contains(&stamp)
        {
            continue;
        }
        if selected.get(&frame.metadata.sensor_name).is_none_or(|old| {
            old.metadata
                .timestamps
                .normalized_unix_ns
                .unwrap()
                .abs_diff(middle)
                > stamp.abs_diff(middle)
        }) {
            selected.insert(frame.metadata.sensor_name.clone(), frame);
        }
    }
    ensure!(
        !selected.is_empty() && selected.len() <= 5,
        "no bounded RGB tracking capture in requested window"
    );
    let frames: std::collections::BTreeMap<_, _> = selected
        .into_iter()
        .map(|(name, frame)| (name, frame.clone()))
        .collect();
    let stamps: Vec<_> = frames
        .values()
        .map(|f| f.metadata.timestamps.normalized_unix_ns.unwrap())
        .collect();
    let first = *stamps.iter().min().unwrap();
    let last = *stamps.iter().max().unwrap();
    Ok(std::sync::Arc::new(SynchronizedFrameSet {
        sequence: frames.values().map(|f| f.metadata.sequence).max().unwrap(),
        timestamp_basis: "normalized_source".into(),
        timestamp_ns: last,
        maximum_skew_ns: last.abs_diff(first),
        frames,
    }))
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
    fn the_viewer_follows_one_ee_tracking_topic_per_arm() {
        assert_eq!(ee_tracking_arm("tatbot/tracking/ee/right"), Some("right"));
        assert_eq!(ee_tracking_arm("tatbot/tracking/ee/left"), Some("left"));
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
        let (_, preview, _) = decode_inner(&wire, true).unwrap();
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
        let bus = tatbot_bus::transport::Bus::open(&[], &[]).unwrap();
        let publisher = bus
            .session
            .declare_publisher("tatbot/vision/poe/camera1")
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
    fn luma_previews_downscale_as_single_channel() {
        let (_, set) = fixture();
        let mut frame = set.frames["color"].clone();
        frame.metadata.profile.width = 4;
        frame.metadata.profile.height = 4;
        frame.metadata.profile.format = PixelFormat::Y8;
        frame.payload = RecordedPayload::Video {
            format: PixelFormat::Y8,
            width: 4,
            height: 4,
            bytes: (0u8..16).collect(),
        };
        let scaled = downscale(&frame, 0.5).unwrap();
        assert_eq!(scaled.metadata.profile.format, PixelFormat::Y8);
        let RecordedPayload::Video {
            format,
            width,
            height,
            bytes,
        } = &scaled.payload
        else {
            panic!("downscale must keep a decoded video payload");
        };
        assert_eq!((*format, *width, *height), (PixelFormat::Y8, 2, 2));
        assert_eq!(bytes.len(), 4);

        let mut short = frame.clone();
        if let RecordedPayload::Video { bytes, .. } = &mut short.payload {
            bytes.pop();
        }
        assert!(downscale(&short, 0.5).is_err());
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
    fn tracking_capture_selects_full_resolution_per_camera_without_relaxing_rgbd() {
        let (producer, mut set) = fixture();
        let stamp = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos() as i128
            - 100_000_000;
        set.frames.remove("depth");
        let frame = set.frames.get_mut("color").unwrap();
        frame.metadata.sensor_kind = crate::SensorKind::PoE;
        frame.metadata.timestamps.normalized_unix_ns = Some(stamp);
        frame.metadata.profile.width = 8;
        frame.metadata.profile.height = 6;
        frame.payload = RecordedPayload::Video {
            format: PixelFormat::Bgr8,
            width: 8,
            height: 6,
            bytes: vec![32; 8 * 6 * 3],
        };
        let mut later = set.clone();
        later
            .frames
            .get_mut("color")
            .unwrap()
            .metadata
            .timestamps
            .normalized_unix_ns = Some(stamp + 60_000_000);
        let history = std::collections::VecDeque::from([
            std::sync::Arc::new(set),
            std::sync::Arc::new(later),
        ]);
        let window = serde_json::to_vec(&CaptureWindow {
            after_ns: stamp - 1,
            before_ns: stamp + 1,
        })
        .unwrap();
        let selected = select_tracking_capture(&history, Some(&window)).unwrap();
        assert_eq!(
            selected.frames["color"]
                .metadata
                .timestamps
                .normalized_unix_ns,
            Some(stamp)
        );
        let (_, decoded) = decode(&encode(&selected, &producer).unwrap()).unwrap();
        assert_eq!(
            (
                decoded.frames["color"].metadata.profile.width,
                decoded.frames["color"].metadata.profile.height
            ),
            (8, 6)
        );
        assert!(validate_capture_geometry(&selected).is_err());
        let empty = serde_json::to_vec(&CaptureWindow {
            after_ns: stamp + 1,
            before_ns: stamp + 2,
        })
        .unwrap();
        assert!(select_tracking_capture(&history, Some(&empty)).is_err());
        assert!(
            select_tracking_capture(&history, Some(br#"{"after_ns":1,"before_ns":2}"#)).is_err()
        );
    }

    #[test]
    fn tracking_history_retains_earlier_exposure_as_full_resolution_luma() {
        let (producer, mut set) = fixture();
        set.frames.remove("depth");
        set.frames.get_mut("color").unwrap().metadata.sensor_kind = crate::SensorKind::PoE;
        let publisher = FramePublisher::open(&[], producer, "poe").unwrap();
        let stamp = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos() as i128
            - 200_000_000;
        for index in 0..3 {
            let timestamp = stamp + index * 50_000_000;
            set.timestamp_ns = timestamp;
            set.frames
                .get_mut("color")
                .unwrap()
                .metadata
                .timestamps
                .normalized_unix_ns = Some(timestamp);
            publisher.submit(&set).unwrap();
            let deadline = std::time::Instant::now() + std::time::Duration::from_secs(2);
            while publisher.latest.lock().unwrap().len() != (index + 1) as usize {
                assert!(
                    std::time::Instant::now() < deadline,
                    "tracking history worker stalled"
                );
                std::thread::sleep(std::time::Duration::from_millis(2));
            }
        }
        let bounds = serde_json::to_vec(&CaptureWindow {
            after_ns: stamp - 1,
            before_ns: stamp + 1,
        })
        .unwrap();
        let history = publisher.latest.lock().unwrap();
        let selected = select_tracking_capture(&history, Some(&bounds)).unwrap();
        let frame = &selected.frames["color"];
        assert_eq!(frame.metadata.timestamps.normalized_unix_ns, Some(stamp));
        assert_eq!(frame.metadata.profile.format, PixelFormat::Y8);
        assert_eq!(
            (frame.metadata.profile.width, frame.metadata.profile.height),
            (2, 2)
        );
        assert_eq!(frame.payload.bytes().len(), 4);
        assert_eq!(
            set.frames["color"].metadata.profile.format,
            PixelFormat::Bgr8
        );
        drop(history);
        assert!(!publisher.tracking_worker.as_ref().unwrap().is_finished());
    }
}
