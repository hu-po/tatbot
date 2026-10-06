//! A host-clock exchange retained with one overhead RGBD response.
//! These bounds compare the camera owner's host clock with the arm host clock.
//! Historical selections may predate the query. A fresh-selection witness
//! proves only that the owner drained its ready SDK queue before delivery;
//! neither receipt bounds RealSense device-to-owner exposure timestamp error.
use crate::SynchronizedFrameSet;
pub use crate::time::{
    RAW_DEVICE_CLOCK_SAMPLE_SCHEMA, RawClockProbeFrame, RawDeviceClockProbe, RawDeviceClockRead,
};
use anyhow::{Result, ensure};
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use tatbot_bus::Producer;

pub const REQUEST_SCHEMA: &str = "tatbot.capture-clock-request/1";
pub const SAMPLE_SCHEMA: &str = "tatbot.capture-clock-sample/1";
pub const EXCHANGE_SCHEMA: &str = "tatbot.capture-clock-exchange/1";
pub const FRESH_SELECTION_SCHEMA: &str = "tatbot.capture-fresh-selection/1";
pub const FRESH_SELECTION_MODE: &str = "sdk_queue_drained_post_request";
pub const DEVICE_EXPOSURE_SCHEMA: &str = "tatbot.device-exposure-clock/1";
pub const RAW_DEVICE_CLOCK_SCHEMA: &str = "tatbot.raw-device-clock-probe/1";
pub const CLOCK_ID: &str = "systemtime-unix-realtime-ns";
const MAX_EXCHANGE_NS: u64 = 3_000_000_000;
const MAX_LOCAL_CLOCK_DISAGREEMENT_NS: u64 = 5_000_000;

/// The selected pixels and raw metadata are retained together, but this
/// version has no device-clock read or measured device-to-owner offset.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct DeviceExposureClockDiagnostic {
    pub schema: String,
    pub status: String,
    pub producer: Producer,
    pub capture_epoch: String,
    pub set_sequence: u64,
    pub frame_sequence: u64,
    pub frame_selection: String,
    pub selection_generation: Option<u64>,
    pub device_clock_id: String,
    pub owner_clock_id: String,
    pub frames: BTreeMap<String, DeviceExposureFrame>,
    pub owner_exposure_interval_ns: Option<[i128; 2]>,
    pub device_to_owner_uncertainty_ns: Option<u64>,
    pub exposure_clock_authority: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct DeviceExposureFrame {
    pub record_sequence: u64,
    pub device_frame_number: Option<u64>,
    pub device_serial: Option<String>,
    pub device_firmware: Option<String>,
    pub source_domain: String,
    pub source_ns: Option<i128>,
    pub normalized_unix_ns: Option<i128>,
    pub host_processed_unix_ns: i128,
    pub sensor_timestamp_us: Option<u64>,
    pub frame_timestamp_us: Option<u64>,
    pub actual_exposure_us: Option<u64>,
    pub backend_timestamp_us: Option<u64>,
    pub time_of_arrival_us: Option<u64>,
    pub payload_sha256: String,
    pub payload_bytes: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct RawClockSelectedFrame {
    pub record_sequence: u64,
    pub device_frame_number: Option<u64>,
    pub sensor_timestamp_us: Option<u64>,
    pub actual_exposure_us: Option<u64>,
    pub payload_sha256: String,
    pub payload_bytes: u64,
}

/// A replay-bound raw-clock observation. There is deliberately no conversion
/// from the raw counter to SENSOR_TIMESTAMP or an exposure interval here.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct RawDeviceClockDiagnostic {
    pub schema: String,
    pub status: String,
    pub producer: Producer,
    pub capture_epoch: String,
    pub set_sequence: u64,
    pub frame_sequence: u64,
    pub selection_generation: Option<u64>,
    pub selected_frames: BTreeMap<String, RawClockSelectedFrame>,
    pub probe: Option<RawDeviceClockProbe>,
    pub device_metadata_raw_clock_relation: String,
    pub owner_exposure_interval_ns: Option<[i128; 2]>,
    pub exposure_clock_authority: bool,
}

fn raw_read_valid(read: &RawDeviceClockRead) -> bool {
    if read.status != "ok" {
        return false;
    }
    let (Some(start), Some(end), Some(elapsed), Some(device)) = (
        read.owner_before_unix_ns,
        read.owner_after_unix_ns,
        read.owner_monotonic_elapsed_ns,
        read.device_time_us,
    ) else {
        return false;
    };
    start > 0
        && end >= start
        && end - start <= 100_000_000
        && (end - start).abs_diff(elapsed) <= 5_000_000
        && device <= 4_294_967_295
}

fn validate_raw_read(read: &RawDeviceClockRead) -> Result<()> {
    ensure!(
        ["ok", "unsupported", "read_failed", "feature_disabled"]
            .contains(&read.status.as_str()),
        "overhead raw device clock read status invalid"
    );
    if read.status == "feature_disabled" {
        ensure!(
            read.owner_before_unix_ns.is_none()
                && read.owner_after_unix_ns.is_none()
                && read.owner_monotonic_elapsed_ns.is_none()
                && read.device_time_us.is_none(),
            "overhead disabled raw device clock has a sample"
        );
    } else if read.status == "ok" {
        ensure!(
            read.owner_before_unix_ns.is_some()
                && read.owner_after_unix_ns.is_some()
                && read.owner_monotonic_elapsed_ns.is_some()
                && read.device_time_us.is_some_and(|value| value <= 4_294_967_295),
            "overhead successful raw device clock lacks a sample"
        );
    } else {
        ensure!(
            read.device_time_us.is_none(),
            "overhead failed raw device clock has a counter"
        );
    }
    Ok(())
}

fn raw_probe_status(probe: Option<&RawDeviceClockProbe>) -> &'static str {
    let Some(probe) = probe else {
        return "selected_pair_not_sampled";
    };
    if probe.before.status == "feature_disabled" || probe.after.status == "feature_disabled" {
        return "raw_clock_feature_disabled";
    }
    if probe.before.status == "unsupported" || probe.after.status == "unsupported" {
        return "raw_clock_unsupported";
    }
    if probe.before.status != "ok" || probe.after.status != "ok" {
        return "raw_clock_unavailable";
    }
    if !raw_read_valid(&probe.before) || !raw_read_valid(&probe.after) {
        return "raw_clock_host_bracket_invalid";
    }
    if probe.owner_pair_monotonic_elapsed_ns > 3_000_000_000 {
        return "raw_clock_sample_too_old";
    }
    if probe.after.owner_after_unix_ns.unwrap() < probe.before.owner_before_unix_ns.unwrap()
        || probe.after.owner_after_unix_ns.unwrap() - probe.before.owner_before_unix_ns.unwrap()
            > 3_000_000_000
    {
        return "raw_clock_sample_too_old";
    }
    if (probe.after.owner_after_unix_ns.unwrap()
        - probe.before.owner_before_unix_ns.unwrap())
    .abs_diff(probe.owner_pair_monotonic_elapsed_ns)
        > 5_000_000
    {
        return "raw_clock_host_bracket_invalid";
    }
    if probe.after.device_time_us.unwrap() < probe.before.device_time_us.unwrap() {
        return "raw_clock_wrap_or_step";
    }
    "raw_samples_metadata_epoch_unverified"
}

impl RawDeviceClockDiagnostic {
    pub fn from_manifest(manifest: &serde_json::Value) -> Result<Self> {
        let selected = DeviceExposureClockDiagnostic::from_manifest(manifest)?;
        let mut source_probe: Option<RawDeviceClockProbe> = None;
        let mut missing = false;
        for frame in manifest["frames"].as_object().unwrap().values() {
            let raw = frame["metadata"]["attributes"].get("raw_device_clock_probe");
            if let Some(raw) = raw {
                let raw = raw
                    .as_str()
                    .ok_or_else(|| anyhow::anyhow!("overhead raw device clock probe is not text"))?;
                let probe: RawDeviceClockProbe = serde_json::from_str(raw)?;
                validate_raw_read(&probe.before)?;
                validate_raw_read(&probe.after)?;
                ensure!(
                    probe.schema == RAW_DEVICE_CLOCK_SAMPLE_SCHEMA
                        && selected.selection_generation == Some(probe.generation)
                        && probe.generation > 0
                        && probe.capture_epoch == selected.capture_epoch
                        && probe.frames.len() == selected.frames.len()
                        && selected.frames.iter().all(|(name, frame)| {
                            probe.frames.get(name).is_some_and(|source| {
                                source.device_frame_number
                                    == frame.device_frame_number.map(|value| value.to_string())
                                    && source.sensor_timestamp_us
                                        == frame.sensor_timestamp_us.map(|value| value.to_string())
                                    && source.actual_exposure_us
                                        == frame.actual_exposure_us.map(|value| value.to_string())
                            })
                        })
                        && ((probe.before.status != "ok" && probe.after.status != "ok")
                            || probe.sdk_version.as_ref().is_some_and(|value| !value.is_empty())),
                    "overhead raw device clock probe identity invalid"
                );
                if let Some(previous) = &source_probe {
                    ensure!(previous == &probe, "overhead raw device clock probes differ");
                }
                source_probe = Some(probe);
            } else {
                missing = true;
            }
        }
        ensure!(
            source_probe.is_none() || !missing,
            "overhead raw device clock probe incomplete"
        );
        let selected_frames = selected
            .frames
            .into_iter()
            .map(|(name, frame)| {
                (
                    name,
                    RawClockSelectedFrame {
                        record_sequence: frame.record_sequence,
                        device_frame_number: frame.device_frame_number,
                        sensor_timestamp_us: frame.sensor_timestamp_us,
                        actual_exposure_us: frame.actual_exposure_us,
                        payload_sha256: frame.payload_sha256,
                        payload_bytes: frame.payload_bytes,
                    },
                )
            })
            .collect();
        Ok(Self {
            schema: RAW_DEVICE_CLOCK_SCHEMA.into(),
            status: raw_probe_status(source_probe.as_ref()).into(),
            producer: selected.producer,
            capture_epoch: selected.capture_epoch,
            set_sequence: selected.set_sequence,
            frame_sequence: selected.frame_sequence,
            selection_generation: selected.selection_generation,
            selected_frames,
            probe: source_probe,
            device_metadata_raw_clock_relation: "unverified_on_installed_device".into(),
            owner_exposure_interval_ns: None,
            exposure_clock_authority: false,
        })
    }

    pub fn validate_retained(manifest: &serde_json::Value) -> Result<()> {
        let has_source_probe = manifest["frames"]
            .as_object()
            .is_some_and(|frames| {
                frames.values().any(|frame| {
                    frame["metadata"]["attributes"]
                        .get("raw_device_clock_probe")
                        .is_some()
                })
            });
        if manifest.get("raw_device_clock_probe").is_none() && !has_source_probe {
            // Pre-probe manifests stay readable, with no implied authority.
            return Ok(());
        }
        let retained: Self = serde_json::from_value(manifest["raw_device_clock_probe"].clone())?;
        ensure!(
            retained == Self::from_manifest(manifest)?,
            "overhead raw device clock diagnostic does not bind selected source"
        );
        Ok(())
    }
}

fn attribute_u64(attributes: &serde_json::Value, name: &str) -> Result<Option<u64>> {
    let Some(value) = attributes.get(name) else {
        return Ok(None);
    };
    let text = value
        .as_str()
        .ok_or_else(|| anyhow::anyhow!("overhead {name} metadata is not text"))?;
    let parsed = text.parse::<u64>()?;
    ensure!(text == parsed.to_string(), "overhead {name} metadata is not canonical");
    Ok(Some(parsed))
}

fn attribute_text(attributes: &serde_json::Value, name: &str) -> Result<Option<String>> {
    let Some(value) = attributes.get(name) else {
        return Ok(None);
    };
    Ok(Some(
        value
            .as_str()
            .ok_or_else(|| anyhow::anyhow!("overhead {name} metadata is not text"))?
            .to_owned(),
    ))
}

fn selected_frame(
    frame: &serde_json::Value,
) -> Result<(DeviceExposureFrame, String, Option<u64>)> {
    let metadata = &frame["metadata"];
    let attributes = &metadata["attributes"];
    let epoch = attributes["capture_epoch"]
        .as_str()
        .ok_or_else(|| anyhow::anyhow!("overhead capture epoch missing"))?;
    ensure!(!epoch.is_empty(), "overhead capture epoch empty");
    let digest = frame["sha256"]
        .as_str()
        .ok_or_else(|| anyhow::anyhow!("overhead payload digest missing"))?;
    ensure!(
        digest.len() == 64
            && digest.bytes().all(|c| c.is_ascii_digit() || (b'a'..=b'f').contains(&c)),
        "overhead payload digest invalid"
    );
    let times = &metadata["timestamps"];
    let stamp = DeviceExposureFrame {
        record_sequence: serde_json::from_value(metadata["sequence"].clone())?,
        device_frame_number: attribute_u64(attributes, "frame_number")?,
        device_serial: attribute_text(attributes, "device_serial")?,
        device_firmware: attribute_text(attributes, "device_firmware")?,
        source_domain: times["source_domain"]
            .as_str()
            .ok_or_else(|| anyhow::anyhow!("overhead source clock domain missing"))?
            .to_owned(),
        source_ns: serde_json::from_value(times["source_ns"].clone())?,
        normalized_unix_ns: serde_json::from_value(times["normalized_unix_ns"].clone())?,
        host_processed_unix_ns: serde_json::from_value(times["host_unix_ns"].clone())?,
        sensor_timestamp_us: attribute_u64(attributes, "sensor_timestamp_us")?,
        frame_timestamp_us: attribute_u64(attributes, "frame_timestamp_us")?,
        actual_exposure_us: attribute_u64(attributes, "actual_exposure_us")?,
        backend_timestamp_us: attribute_u64(attributes, "backend_timestamp_us")?,
        time_of_arrival_us: attribute_u64(attributes, "time_of_arrival_us")?,
        payload_sha256: digest.to_owned(),
        payload_bytes: serde_json::from_value(frame["payload_bytes"].clone())?,
    };
    ensure!(stamp.host_processed_unix_ns > 0, "overhead host processing time invalid");
    Ok((
        stamp,
        epoch.to_owned(),
        attribute_u64(attributes, "sdk_queue_flush_generation")?,
    ))
}

impl DeviceExposureClockDiagnostic {
    /// Recompute from the retained pair and producer, never from the receipt.
    pub fn from_manifest(manifest: &serde_json::Value) -> Result<Self> {
        let producer_fields = manifest["producer"]
            .as_object()
            .ok_or_else(|| anyhow::anyhow!("overhead producer missing"))?;
        ensure!(
            producer_fields.len() == 4
                && ["node", "pid", "sha", "run_id"]
                    .into_iter()
                    .all(|key| producer_fields.contains_key(key)),
            "overhead producer identity invalid"
        );
        let producer: Producer = serde_json::from_value(manifest["producer"].clone())?;
        let source = manifest["frames"]
            .as_object()
            .ok_or_else(|| anyhow::anyhow!("overhead frames missing"))?;
        ensure!(
            source.len() == 2
                && source.contains_key("overhead_depth_color")
                && source.contains_key("overhead_depth_depth"),
            "overhead device clock requires one RGBD pair"
        );
        let mut frames = BTreeMap::new();
        let mut identity = None;
        for (name, frame) in source {
            let (selected, epoch, generation) = selected_frame(frame)?;
            let current = (epoch, selected.record_sequence, generation);
            if let Some(previous) = &identity {
                ensure!(previous == &current, "overhead device clock frame identities differ");
            }
            identity = Some(current);
            frames.insert(name.clone(), selected);
        }
        let (capture_epoch, frame_sequence, selection_generation) = identity.unwrap();
        let frame_selection = manifest["frame_selection"]
            .as_str()
            .ok_or_else(|| anyhow::anyhow!("overhead frame selection missing"))?;
        ensure!(
            (frame_selection == FRESH_SELECTION_MODE && selection_generation.is_some())
                || (frame_selection == "latest_history" && selection_generation.is_none()),
            "overhead device clock selection identity invalid"
        );
        Ok(Self {
            schema: DEVICE_EXPOSURE_SCHEMA.into(),
            status: "device_to_owner_unverified".into(),
            producer,
            capture_epoch,
            set_sequence: serde_json::from_value(manifest["sequence"].clone())?,
            frame_sequence,
            frame_selection: frame_selection.into(),
            selection_generation,
            device_clock_id: "realsense-frame-metadata-device-us-unverified".into(),
            owner_clock_id: CLOCK_ID.into(),
            frames,
            owner_exposure_interval_ns: None,
            device_to_owner_uncertainty_ns: None,
            exposure_clock_authority: false,
        })
    }

    pub fn validate_retained(manifest: &serde_json::Value) -> Result<()> {
        let retained: Self = serde_json::from_value(manifest["device_exposure_clock"].clone())?;
        ensure!(
            retained == Self::from_manifest(manifest)?,
            "overhead device exposure diagnostic does not bind selected source"
        );
        Ok(())
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct CaptureClockRequest {
    pub schema: String,
    pub nonce: String,
}

impl CaptureClockRequest {
    pub fn new(nonce: String) -> Result<Self> {
        let request = Self {
            schema: REQUEST_SCHEMA.into(),
            nonce,
        };
        request.validate()?;
        Ok(request)
    }

    pub fn validate(&self) -> Result<()> {
        ensure!(
            self.schema == REQUEST_SCHEMA
                && self.nonce.len() == 32
                && self
                    .nonce
                    .bytes()
                    .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte)),
            "overhead clock request identity invalid"
        );
        Ok(())
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct CaptureClockSample {
    pub schema: String,
    pub request_nonce: String,
    pub clock_id: String,
    pub producer_run_id: String,
    pub capture_epoch: String,
    pub sequence: u64,
    pub frame_sequence: u64,
    pub owner_received_ns: u64,
    pub owner_prepared_ns: u64,
    pub owner_monotonic_elapsed_ns: u64,
    pub frame_host_processed_ns: i128,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub fresh_selection: Option<CaptureFreshSelection>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct CaptureFreshSelection {
    pub schema: String,
    pub generation: u64,
    pub previous_set_sequence: Option<u64>,
    pub sdk_queue_drained_sets: u64,
}

#[derive(Clone, Copy)]
pub struct SelectedCaptureIdentity<'a> {
    pub epoch: &'a str,
    pub set_sequence: u64,
    pub frame_sequence: u64,
    pub frame_host_processed_ns: i128,
    pub fresh_tags: Option<(u64, u64)>,
}

pub fn fresh_tags(set: &SynchronizedFrameSet) -> Result<Option<(u64, u64)>> {
    let mut found = None;
    let mut untagged = false;
    for frame in set.frames.values() {
        let attrs = &frame.metadata.attributes;
        match (
            attrs.get("sdk_queue_flush_generation"),
            attrs.get("sdk_queue_flush_discarded"),
        ) {
            (None, None) => untagged = true,
            (Some(generation), Some(discarded)) => {
                let values = (generation.parse::<u64>()?, discarded.parse::<u64>()?);
                ensure!(values.0 > 0, "overhead SDK queue flush generation invalid");
                if let Some(previous) = found {
                    ensure!(previous == values, "overhead SDK queue flush tags differ");
                }
                found = Some(values);
            }
            _ => anyhow::bail!("overhead SDK queue flush tags incomplete"),
        }
    }
    ensure!(
        found.is_none() || !untagged,
        "overhead SDK queue flush tags incomplete"
    );
    Ok(found)
}

impl CaptureFreshSelection {
    pub fn from_set(
        set: &SynchronizedFrameSet,
        previous_set_sequence: Option<u64>,
    ) -> Result<Self> {
        let (generation, sdk_queue_drained_sets) = fresh_tags(set)?
            .ok_or_else(|| anyhow::anyhow!("overhead SDK queue was not drained"))?;
        let selection = Self {
            schema: FRESH_SELECTION_SCHEMA.into(),
            generation,
            previous_set_sequence,
            sdk_queue_drained_sets,
        };
        selection.validate(set.sequence, Some((generation, sdk_queue_drained_sets)))?;
        Ok(selection)
    }

    fn validate(&self, selected_sequence: u64, tags: Option<(u64, u64)>) -> Result<()> {
        ensure!(
            self.schema == FRESH_SELECTION_SCHEMA
                && tags == Some((self.generation, self.sdk_queue_drained_sets))
                && self.generation > 0
                && self.sdk_queue_drained_sets < 64
                && self
                    .previous_set_sequence
                    .is_none_or(|previous| selected_sequence > previous),
            "overhead fresh selection does not bind the post-flush set"
        );
        Ok(())
    }
}

pub fn capture_epoch(set: &SynchronizedFrameSet) -> Result<&str> {
    let mut epoch = None;
    for frame in set.frames.values() {
        let value = frame
            .metadata
            .attributes
            .get("capture_epoch")
            .ok_or_else(|| anyhow::anyhow!("overhead capture epoch missing"))?;
        ensure!(!value.is_empty(), "overhead capture epoch empty");
        if let Some(previous) = epoch {
            ensure!(
                previous == value,
                "overhead capture epoch differs across frames"
            );
        }
        epoch = Some(value);
    }
    epoch
        .map(String::as_str)
        .ok_or_else(|| anyhow::anyhow!("overhead capture has no frames"))
}

pub fn frame_sequence(set: &SynchronizedFrameSet) -> Result<u64> {
    let mut sequence = None;
    for frame in set.frames.values() {
        let value = frame.metadata.sequence;
        if let Some(previous) = sequence {
            ensure!(previous == value, "overhead frame sequences differ");
        }
        sequence = Some(value);
    }
    sequence.ok_or_else(|| anyhow::anyhow!("overhead capture has no frames"))
}

impl CaptureClockSample {
    pub fn for_response(
        request: &CaptureClockRequest,
        producer: &Producer,
        set: &SynchronizedFrameSet,
        owner_received_ns: u64,
        owner_prepared_ns: u64,
        owner_monotonic_elapsed_ns: u64,
        fresh_selection: Option<CaptureFreshSelection>,
    ) -> Result<Self> {
        request.validate()?;
        let sample = Self {
            schema: SAMPLE_SCHEMA.into(),
            request_nonce: request.nonce.clone(),
            clock_id: CLOCK_ID.into(),
            producer_run_id: producer.run_id.clone(),
            capture_epoch: capture_epoch(set)?.into(),
            sequence: set.sequence,
            frame_sequence: frame_sequence(set)?,
            owner_received_ns,
            owner_prepared_ns,
            owner_monotonic_elapsed_ns,
            frame_host_processed_ns: set
                .frames
                .values()
                .map(|frame| frame.metadata.timestamps.host_unix_ns)
                .max()
                .unwrap_or(0),
            fresh_selection,
        };
        sample.validate(request, producer, set)?;
        Ok(sample)
    }

    pub fn validate(
        &self,
        request: &CaptureClockRequest,
        producer: &Producer,
        set: &SynchronizedFrameSet,
    ) -> Result<()> {
        self.validate_identity(
            request,
            producer,
            SelectedCaptureIdentity {
                epoch: capture_epoch(set)?,
                set_sequence: set.sequence,
                frame_sequence: frame_sequence(set)?,
                frame_host_processed_ns: set
                    .frames
                    .values()
                    .map(|frame| frame.metadata.timestamps.host_unix_ns)
                    .max()
                    .unwrap_or(0),
                fresh_tags: fresh_tags(set)?,
            },
        )
    }

    fn validate_identity(
        &self,
        request: &CaptureClockRequest,
        producer: &Producer,
        selected: SelectedCaptureIdentity<'_>,
    ) -> Result<()> {
        request.validate()?;
        ensure!(
            self.schema == SAMPLE_SCHEMA
                && self.request_nonce == request.nonce
                && self.clock_id == CLOCK_ID
                && !producer.run_id.is_empty()
                && self.producer_run_id == producer.run_id
                && !selected.epoch.is_empty()
                && self.capture_epoch == selected.epoch
                && self.sequence == selected.set_sequence
                && self.frame_sequence == selected.frame_sequence
                && self.frame_host_processed_ns == selected.frame_host_processed_ns,
            "overhead clock sample does not bind the selected capture"
        );
        ensure!(
            self.frame_host_processed_ns > 0,
            "overhead frame host processing time missing"
        );
        match &self.fresh_selection {
            Some(selection) => selection.validate(selected.set_sequence, selected.fresh_tags)?,
            None => ensure!(
                selected.fresh_tags.is_none(),
                "overhead fresh selection receipt missing"
            ),
        }
        Ok(())
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct CaptureClockExchange {
    pub schema: String,
    pub status: String,
    pub reason: Option<String>,
    pub request_nonce: String,
    pub arm_clock_id: String,
    pub arm_sent_ns: u64,
    pub arm_received_ns: u64,
    pub arm_monotonic_elapsed_ns: u64,
    /// Four-timestamp offset interval under stable host-clock rates. The
    /// local elapsed checks detect large steps, not every transient slew.
    pub owner_minus_arm_interval_ns: Option<[i64; 2]>,
    pub interval_width_ns: Option<u64>,
    pub sample: CaptureClockSample,
}

impl CaptureClockExchange {
    pub fn bound(
        request: &CaptureClockRequest,
        sample: CaptureClockSample,
        producer: &Producer,
        set: &SynchronizedFrameSet,
        arm_sent_ns: u64,
        arm_received_ns: u64,
        arm_monotonic_elapsed_ns: u64,
    ) -> Result<Self> {
        sample.validate(request, producer, set)?;
        Ok(Self::classify_timestamps(
            request.nonce.clone(),
            sample,
            arm_sent_ns,
            arm_received_ns,
            arm_monotonic_elapsed_ns,
        ))
    }

    fn classify_timestamps(
        request_nonce: String,
        sample: CaptureClockSample,
        arm_sent_ns: u64,
        arm_received_ns: u64,
        arm_monotonic_elapsed_ns: u64,
    ) -> Self {
        let assessment = clock_interval(
            &sample,
            arm_sent_ns,
            arm_received_ns,
            arm_monotonic_elapsed_ns,
        );
        let (status, reason, interval, width) = match assessment {
            Ok((interval, width)) => (
                "host_offset_diagnostic_only",
                None,
                Some(interval),
                Some(width),
            ),
            Err(reason) => ("host_offset_invalid", Some(reason.into()), None, None),
        };
        Self {
            schema: EXCHANGE_SCHEMA.into(),
            status: status.into(),
            reason,
            request_nonce,
            arm_clock_id: CLOCK_ID.into(),
            arm_sent_ns,
            arm_received_ns,
            arm_monotonic_elapsed_ns,
            owner_minus_arm_interval_ns: interval,
            interval_width_ns: width,
            sample,
        }
    }

    pub fn validate_retained(
        &self,
        producer: &Producer,
        selected: SelectedCaptureIdentity<'_>,
        arm_request_interval_ns: [u64; 2],
    ) -> Result<()> {
        let request = CaptureClockRequest::new(self.request_nonce.clone())?;
        self.sample
            .validate_identity(&request, producer, selected)?;
        ensure!(
            self.schema == EXCHANGE_SCHEMA
                && self.arm_clock_id == CLOCK_ID
                && [self.arm_sent_ns, self.arm_received_ns] == arm_request_interval_ns,
            "overhead clock exchange status invalid"
        );
        ensure!(
            *self
                == Self::classify_timestamps(
                    self.request_nonce.clone(),
                    self.sample.clone(),
                    self.arm_sent_ns,
                    self.arm_received_ns,
                    self.arm_monotonic_elapsed_ns,
                ),
            "overhead clock exchange bound changed"
        );
        Ok(())
    }
}

fn clock_interval(
    sample: &CaptureClockSample,
    arm_sent_ns: u64,
    arm_received_ns: u64,
    arm_monotonic_elapsed_ns: u64,
) -> std::result::Result<([i64; 2], u64), &'static str> {
    if arm_sent_ns == 0
        || sample.owner_received_ns == 0
        || arm_received_ns < arm_sent_ns
        || sample.owner_prepared_ns < sample.owner_received_ns
    {
        return Err("clock_timestamps_invalid");
    }
    let arm_wall_elapsed = arm_received_ns - arm_sent_ns;
    let owner_wall_elapsed = sample.owner_prepared_ns - sample.owner_received_ns;
    if arm_wall_elapsed > MAX_EXCHANGE_NS
        || arm_monotonic_elapsed_ns > MAX_EXCHANGE_NS
        || owner_wall_elapsed > MAX_EXCHANGE_NS
        || sample.owner_monotonic_elapsed_ns > MAX_EXCHANGE_NS
    {
        return Err("clock_exchange_too_old");
    }
    if arm_wall_elapsed.abs_diff(arm_monotonic_elapsed_ns) > MAX_LOCAL_CLOCK_DISAGREEMENT_NS
        || owner_wall_elapsed.abs_diff(sample.owner_monotonic_elapsed_ns)
            > MAX_LOCAL_CLOCK_DISAGREEMENT_NS
    {
        return Err("host_clock_step_suspected");
    }
    // Causal event order is arm send, owner receive, owner prepare, arm
    // receive. For owner-minus-arm, t2_owner-t3_arm is the lower bound and
    // t1_owner-t0_arm the upper bound; this says nothing about sensor exposure.
    let lower = i128::from(sample.owner_prepared_ns) - i128::from(arm_received_ns);
    let upper = i128::from(sample.owner_received_ns) - i128::from(arm_sent_ns);
    if lower > upper {
        return Err("clock_exchange_order_invalid");
    }
    let interval = [
        i64::try_from(lower).map_err(|_| "clock_offset_out_of_range")?,
        i64::try_from(upper).map_err(|_| "clock_offset_out_of_range")?,
    ];
    let width = u64::try_from(upper - lower).map_err(|_| "clock_offset_out_of_range")?;
    if width > MAX_EXCHANGE_NS {
        return Err("clock_exchange_too_old");
    }
    Ok((interval, width))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn raw_device_clock_projection_matches_shared_contract_and_refuses_replay() {
        let fixture: serde_json::Value = serde_json::from_str(include_str!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../../scripts/tests/fixtures/raw_device_clock.json"
        )))
        .unwrap();
        let mut manifest = fixture["manifest"].clone();
        let projection = RawDeviceClockDiagnostic::from_manifest(&manifest).unwrap();
        assert_eq!(serde_json::to_value(projection).unwrap(), fixture["diagnostic"]);
        manifest["raw_device_clock_probe"] = fixture["diagnostic"].clone();
        RawDeviceClockDiagnostic::validate_retained(&manifest).unwrap();
        assert_eq!(manifest["raw_device_clock_probe"]["exposure_clock_authority"], false);
        for (path, value) in [
            ("/raw_device_clock_probe/producer/run_id", serde_json::json!("other-run")),
            ("/raw_device_clock_probe/probe/after/device_time_us", serde_json::json!(180001)),
            ("/raw_device_clock_probe/exposure_clock_authority", serde_json::json!(true)),
            ("/frames/overhead_depth_depth/sha256", serde_json::json!("c".repeat(64))),
            ("/frames/overhead_depth_color/metadata/attributes/sensor_timestamp_us", serde_json::json!("1")),
        ] {
            let mut changed = manifest.clone();
            *changed.pointer_mut(path).unwrap() = value;
            assert!(RawDeviceClockDiagnostic::validate_retained(&changed).is_err(), "{path}");
        }
        for (path, value) in [
            (
                "/frames/overhead_depth_color/metadata/attributes/raw_device_clock_probe",
                serde_json::json!(true),
            ),
            (
                "/frames/overhead_depth_color/metadata/attributes/raw_device_clock_probe",
                serde_json::json!("{}"),
            ),
        ] {
            let mut changed = manifest.clone();
            *changed.pointer_mut(path).unwrap() = value;
            assert!(RawDeviceClockDiagnostic::validate_retained(&changed).is_err());
        }
        for (expected, path, value) in [
            ("raw_clock_wrap_or_step", "/after/device_time_us", serde_json::json!(122999)),
            ("raw_clock_host_bracket_invalid", "/owner_pair_monotonic_elapsed_ns", serde_json::json!(1)),
            ("raw_clock_sample_too_old", "/owner_pair_monotonic_elapsed_ns", serde_json::json!(3_000_000_001_u64)),
        ] {
            let mut changed = manifest.clone();
            let raw = changed["frames"]["overhead_depth_color"]["metadata"]["attributes"]
                ["raw_device_clock_probe"]
                .as_str()
                .unwrap();
            let mut source: serde_json::Value = serde_json::from_str(raw).unwrap();
            *source.pointer_mut(path).unwrap() = value;
            let encoded = serde_json::to_string(&source).unwrap();
            for frame in changed["frames"].as_object_mut().unwrap().values_mut() {
                frame["metadata"]["attributes"]["raw_device_clock_probe"] = encoded.clone().into();
            }
            let projected = RawDeviceClockDiagnostic::from_manifest(&changed).unwrap();
            assert_eq!(projected.status, expected);
            assert!(!projected.exposure_clock_authority);
            assert_eq!(projected.owner_exposure_interval_ns, None);
        }
        for (path, value) in [
            ("/capture_epoch", serde_json::json!("other-epoch")),
            ("/frames/overhead_depth_color/device_frame_number", serde_json::json!("999")),
        ] {
            let mut changed = manifest.clone();
            let raw = changed["frames"]["overhead_depth_color"]["metadata"]["attributes"]
                ["raw_device_clock_probe"]
                .as_str()
                .unwrap();
            let mut source: serde_json::Value = serde_json::from_str(raw).unwrap();
            *source.pointer_mut(path).unwrap() = value;
            let encoded = serde_json::to_string(&source).unwrap();
            for frame in changed["frames"].as_object_mut().unwrap().values_mut() {
                frame["metadata"]["attributes"]["raw_device_clock_probe"] = encoded.clone().into();
            }
            assert!(RawDeviceClockDiagnostic::from_manifest(&changed).is_err(), "{path}");
        }
        manifest.as_object_mut().unwrap().remove("raw_device_clock_probe");
        assert!(RawDeviceClockDiagnostic::validate_retained(&manifest).is_err());
        for frame in manifest["frames"].as_object_mut().unwrap().values_mut() {
            frame["metadata"]["attributes"]
                .as_object_mut()
                .unwrap()
                .remove("raw_device_clock_probe");
        }
        RawDeviceClockDiagnostic::validate_retained(&manifest).unwrap();
        assert_eq!(
            RawDeviceClockDiagnostic::from_manifest(&manifest).unwrap().status,
            "selected_pair_not_sampled"
        );
    }

    #[test]
    fn device_exposure_projection_matches_shared_contract_and_refuses_replay() {
        let fixture: serde_json::Value = serde_json::from_str(include_str!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../../scripts/tests/fixtures/device_exposure_clock.json"
        )))
        .unwrap();
        let mut manifest = fixture["manifest"].clone();
        let projection = DeviceExposureClockDiagnostic::from_manifest(&manifest).unwrap();
        assert_eq!(serde_json::to_value(projection).unwrap(), fixture["diagnostic"]);
        manifest["device_exposure_clock"] = fixture["diagnostic"].clone();
        DeviceExposureClockDiagnostic::validate_retained(&manifest).unwrap();
        for path in [
            "/frames/overhead_depth_color/metadata/attributes/sensor_timestamp_us",
            "/frames/overhead_depth_depth/metadata/timestamps/normalized_unix_ns",
            "/frames/overhead_depth_depth/sha256",
            "/producer/run_id",
            "/device_exposure_clock/frames/overhead_depth_color/payload_sha256",
            "/device_exposure_clock/exposure_clock_authority",
        ] {
            let mut changed = manifest.clone();
            *changed.pointer_mut(path).unwrap() = serde_json::json!("changed");
            assert!(
                DeviceExposureClockDiagnostic::validate_retained(&changed).is_err(),
                "{path}"
            );
        }
        manifest.as_object_mut().unwrap().remove("device_exposure_clock");
        assert!(DeviceExposureClockDiagnostic::validate_retained(&manifest).is_err());
        let metadata = manifest["frames"]["overhead_depth_color"]["metadata"]["attributes"]
            .as_object_mut()
            .unwrap();
        metadata.remove("sensor_timestamp_us");
        metadata.remove("frame_timestamp_us");
        metadata.remove("actual_exposure_us");
        metadata.remove("backend_timestamp_us");
        metadata.remove("time_of_arrival_us");
        let legacy = DeviceExposureClockDiagnostic::from_manifest(&manifest).unwrap();
        assert_eq!(legacy.status, "device_to_owner_unverified");
        assert_eq!(legacy.frames["overhead_depth_color"].sensor_timestamp_us, None);
        assert_eq!(legacy.owner_exposure_interval_ns, None);
        assert_eq!(legacy.device_to_owner_uncertainty_ns, None);
        assert!(!legacy.exposure_clock_authority);
    }

    fn fixture() -> (CaptureClockRequest, Producer, CaptureClockSample) {
        let request = CaptureClockRequest::new("a".repeat(32)).unwrap();
        let producer = Producer {
            node: "overhead-owner".into(),
            pid: 1,
            sha: "source".into(),
            run_id: "owner-epoch-1".into(),
        };
        let sample = CaptureClockSample {
            schema: SAMPLE_SCHEMA.into(),
            request_nonce: request.nonce.clone(),
            clock_id: CLOCK_ID.into(),
            producer_run_id: producer.run_id.clone(),
            capture_epoch: "camera-epoch-1".into(),
            sequence: 7,
            frame_sequence: 11,
            owner_received_ns: 1_100_000_000,
            owner_prepared_ns: 1_200_000_000,
            owner_monotonic_elapsed_ns: 100_000_000,
            frame_host_processed_ns: 1_050_000_000,
            fresh_selection: None,
        };
        (request, producer, sample)
    }

    fn selected(fresh_tags: Option<(u64, u64)>) -> SelectedCaptureIdentity<'static> {
        SelectedCaptureIdentity {
            epoch: "camera-epoch-1",
            set_sequence: 7,
            frame_sequence: 11,
            frame_host_processed_ns: 1_050_000_000,
            fresh_tags,
        }
    }

    #[test]
    fn four_timestamps_bound_owner_minus_arm_without_exposure_authority() {
        let (request, producer, sample) = fixture();
        sample
            .validate_identity(&request, &producer, selected(None))
            .unwrap();
        let exchange = CaptureClockExchange::classify_timestamps(
            request.nonce.clone(),
            sample,
            1_000_000_000,
            1_150_000_000,
            150_000_000,
        );
        assert_eq!(
            exchange.owner_minus_arm_interval_ns,
            Some([50_000_000, 100_000_000])
        );
        assert_eq!(exchange.interval_width_ns, Some(50_000_000));
        assert_eq!(exchange.status, "host_offset_diagnostic_only");
        exchange
            .validate_retained(&producer, selected(None), [1_000_000_000, 1_150_000_000])
            .unwrap();
    }

    #[test]
    fn retained_fresh_selection_requires_matching_tags_and_later_sequence() {
        let (request, producer, mut sample) = fixture();
        sample.fresh_selection = Some(CaptureFreshSelection {
            schema: FRESH_SELECTION_SCHEMA.into(),
            generation: 3,
            previous_set_sequence: Some(6),
            sdk_queue_drained_sets: 2,
        });
        let bound = |sample: &CaptureClockSample, tags| {
            sample.validate_identity(&request, &producer, selected(tags))
        };
        bound(&sample, Some((3, 2))).unwrap();
        assert!(bound(&sample, None).is_err());
        assert!(bound(&sample, Some((4, 2))).is_err());
        assert!(bound(&sample, Some((3, 3))).is_err());
        sample
            .fresh_selection
            .as_mut()
            .unwrap()
            .previous_set_sequence = Some(7);
        assert!(bound(&sample, Some((3, 2))).is_err());
        sample
            .fresh_selection
            .as_mut()
            .unwrap()
            .previous_set_sequence = Some(6);
        sample
            .fresh_selection
            .as_mut()
            .unwrap()
            .sdk_queue_drained_sets = 64;
        assert!(bound(&sample, Some((3, 64))).is_err());
        sample.fresh_selection = None;
        assert!(bound(&sample, Some((3, 2))).is_err());
    }

    #[test]
    fn stale_identity_clock_step_and_edited_bounds_refuse() {
        let (request, producer, sample) = fixture();
        for changed in [
            CaptureClockSample {
                request_nonce: "b".repeat(32),
                ..sample.clone()
            },
            CaptureClockSample {
                producer_run_id: "other-run".into(),
                ..sample.clone()
            },
            CaptureClockSample {
                capture_epoch: "other-camera".into(),
                ..sample.clone()
            },
            CaptureClockSample {
                sequence: 8,
                ..sample.clone()
            },
            CaptureClockSample {
                frame_sequence: 12,
                ..sample.clone()
            },
        ] {
            assert!(
                changed
                    .validate_identity(&request, &producer, selected(None))
                    .is_err()
            );
        }
        let mut exchange = CaptureClockExchange::classify_timestamps(
            request.nonce.clone(),
            sample,
            1_000_000_000,
            1_150_000_000,
            150_000_000,
        );
        exchange.owner_minus_arm_interval_ns = Some([0, 1]);
        assert!(
            exchange
                .validate_retained(&producer, selected(None), [1_000_000_000, 1_150_000_000],)
                .is_err()
        );
    }

    #[test]
    fn clock_step_retains_invalid_diagnostic_without_an_offset_interval() {
        let (request, producer, mut sample) = fixture();
        sample.owner_monotonic_elapsed_ns = 1;
        sample
            .validate_identity(&request, &producer, selected(None))
            .unwrap();
        let exchange = CaptureClockExchange::classify_timestamps(
            request.nonce.clone(),
            sample,
            1_000_000_000,
            1_150_000_000,
            150_000_000,
        );
        assert_eq!(exchange.status, "host_offset_invalid");
        assert_eq!(
            exchange.reason.as_deref(),
            Some("host_clock_step_suspected")
        );
        assert_eq!(exchange.owner_minus_arm_interval_ns, None);
        exchange
            .validate_retained(&producer, selected(None), [1_000_000_000, 1_150_000_000])
            .unwrap();
    }

    #[test]
    fn overwide_owner_exchange_retains_invalid_diagnostic() {
        let (request, producer, mut sample) = fixture();
        sample.owner_prepared_ns = sample.owner_received_ns + 3_000_000_001;
        sample.owner_monotonic_elapsed_ns = 3_000_000_001;
        let exchange = CaptureClockExchange::classify_timestamps(
            request.nonce.clone(),
            sample,
            1_000_000_000,
            1_150_000_000,
            150_000_000,
        );
        assert_eq!(exchange.status, "host_offset_invalid");
        assert_eq!(exchange.reason.as_deref(), Some("clock_exchange_too_old"));
        assert_eq!(exchange.owner_minus_arm_interval_ns, None);
        exchange
            .validate_retained(&producer, selected(None), [1_000_000_000, 1_150_000_000])
            .unwrap();
    }
}
