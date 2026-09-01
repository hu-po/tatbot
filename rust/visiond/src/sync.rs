use std::collections::{BTreeMap, VecDeque};

use serde::{Deserialize, Serialize};

use crate::{FrameMetadata, FrameRecord};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SyncAssessment {
    pub sample_count: usize,
    pub median_offset_ns: Option<i128>,
    pub median_absolute_deviation_ns: Option<u128>,
    pub min_offset_ns: Option<i128>,
    pub max_offset_ns: Option<i128>,
}

/// Timestamp-domain comparison between two recorded streams.
///
/// The offset is `other - reference`; a positive value means the nearest
/// sample in `other` is later than the reference sample. Samples are matched
/// by nearest timestamp, not by frame number, because independent cameras can
/// start at different frame counters and can drop different frames.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PairwiseSyncReport {
    pub timestamp_basis: String,
    pub reference_samples: usize,
    pub other_samples: usize,
    pub matched_samples: usize,
    pub median_signed_offset_ns: Option<i128>,
    pub median_absolute_offset_ns: Option<u128>,
    pub p95_absolute_offset_ns: Option<u128>,
    pub maximum_absolute_offset_ns: Option<u128>,
}

#[derive(Debug, Clone)]
pub struct SynchronizedFrameSet {
    pub sequence: u64,
    pub timestamp_basis: String,
    pub timestamp_ns: i128,
    pub maximum_skew_ns: u128,
    pub frames: BTreeMap<String, FrameRecord>,
}

/// Online nearest-neighbor assembler for synchronized multi-camera frames.
///
/// Each configured sensor contributes at most one frame to a set. Frames that
/// cannot meet the configured tolerance are discarded as stale rather than
/// silently paired with a frame from the wrong time. The assembler is agnostic
/// to RTSP versus RealSense; it uses normalized source time first and host time
/// only as an explicitly weaker fallback.
#[derive(Debug)]
pub struct FrameSynchronizer {
    sensor_names: Vec<String>,
    minimum_sensors: usize,
    tolerance_ns: u128,
    maximum_wait_ns: Option<u128>,
    capacity: usize,
    buffers: BTreeMap<String, VecDeque<FrameRecord>>,
    sequence: u64,
    dropped_unmatched: u64,
    complete_sets: u64,
    partial_sets: u64,
}

impl FrameSynchronizer {
    pub fn new(
        sensor_names: impl IntoIterator<Item = String>,
        tolerance_ns: u128,
        capacity: usize,
    ) -> Result<Self, String> {
        if tolerance_ns == 0 {
            return Err("frame synchronizer tolerance must be positive".into());
        }
        if capacity == 0 {
            return Err("frame synchronizer capacity must be positive".into());
        }
        let sensor_names: Vec<String> = sensor_names.into_iter().collect();
        let minimum_sensors = sensor_names.len();
        Self::new_inner(sensor_names, minimum_sensors, tolerance_ns, None, capacity)
    }

    /// Build a synchronizer that emits a complete set immediately, but after
    /// `maximum_wait_ns` may emit a fresh subset containing at least
    /// `minimum_sensors`. This is for latency-sensitive tracking only;
    /// evidence/calibration callers should continue to use [`Self::new`].
    pub fn new_partial(
        sensor_names: impl IntoIterator<Item = String>,
        minimum_sensors: usize,
        tolerance_ns: u128,
        maximum_wait_ns: u128,
        capacity: usize,
    ) -> Result<Self, String> {
        let sensor_names: Vec<String> = sensor_names.into_iter().collect();
        Self::new_inner(
            sensor_names,
            minimum_sensors,
            tolerance_ns,
            Some(maximum_wait_ns),
            capacity,
        )
    }

    fn new_inner(
        sensor_names: Vec<String>,
        minimum_sensors: usize,
        tolerance_ns: u128,
        maximum_wait_ns: Option<u128>,
        capacity: usize,
    ) -> Result<Self, String> {
        if sensor_names.is_empty() || sensor_names.iter().any(|name| name.trim().is_empty()) {
            return Err("frame synchronizer needs non-empty sensor names".into());
        }
        if minimum_sensors == 0 || minimum_sensors > sensor_names.len() {
            return Err(format!(
                "frame synchronizer minimum sensors must be in 1..={}, got {minimum_sensors}",
                sensor_names.len()
            ));
        }
        if maximum_wait_ns == Some(0) {
            return Err("partial frame synchronizer maximum wait must be positive".into());
        }
        let mut buffers = BTreeMap::new();
        for name in &sensor_names {
            if buffers
                .insert(name.clone(), VecDeque::with_capacity(capacity))
                .is_some()
            {
                return Err(format!("duplicate frame synchronizer sensor {name}"));
            }
        }
        Ok(Self {
            sensor_names,
            minimum_sensors,
            tolerance_ns,
            maximum_wait_ns,
            capacity,
            buffers,
            sequence: 0,
            dropped_unmatched: 0,
            complete_sets: 0,
            partial_sets: 0,
        })
    }

    pub fn push(&mut self, frame: FrameRecord) -> Result<Vec<SynchronizedFrameSet>, String> {
        let sensor_name = frame.metadata.sensor_name.clone();
        let queue = self
            .buffers
            .get_mut(&sensor_name)
            .ok_or_else(|| format!("unconfigured synchronizer sensor {sensor_name}"))?;
        queue.push_back(frame);
        while queue.len() > self.capacity {
            queue.pop_front();
            self.dropped_unmatched = self.dropped_unmatched.saturating_add(1);
        }
        Ok(self.drain_ready())
    }

    pub fn dropped_unmatched(&self) -> u64 {
        self.dropped_unmatched
    }

    pub fn complete_sets(&self) -> u64 {
        self.complete_sets
    }

    pub fn partial_sets(&self) -> u64 {
        self.partial_sets
    }

    fn drain_ready(&mut self) -> Vec<SynchronizedFrameSet> {
        let mut output = Vec::new();
        loop {
            let reference = self
                .sensor_names
                .iter()
                .filter_map(|name| {
                    let frame = self.buffers.get(name)?.front()?;
                    let (basis, timestamp) = frame_sync_time(frame)?;
                    Some((
                        name.clone(),
                        basis,
                        timestamp,
                        frame.metadata.timestamps.host_unix_ns,
                    ))
                })
                .min_by_key(|(_, _, timestamp, _)| *timestamp);
            let Some((reference_name, reference_basis, reference_time, reference_host_unix_ns)) =
                reference
            else {
                break;
            };

            let mut candidates = BTreeMap::new();
            for name in &self.sensor_names {
                let Some(queue) = self.buffers.get(name) else {
                    continue;
                };
                let Some((index, timestamp)) = queue
                    .iter()
                    .enumerate()
                    .filter_map(|(index, frame)| {
                        let (basis, timestamp) = frame_sync_time(frame)?;
                        (basis == reference_basis).then_some((index, timestamp))
                    })
                    .min_by_key(|(_, timestamp)| timestamp.abs_diff(reference_time))
                else {
                    continue;
                };
                if timestamp.abs_diff(reference_time) <= self.tolerance_ns {
                    candidates.insert(name.clone(), (index, timestamp));
                }
            }

            let complete = candidates.len() == self.sensor_names.len();
            let latest_host_unix_ns = self
                .buffers
                .values()
                .flat_map(|queue| queue.iter())
                .map(|frame| frame.metadata.timestamps.host_unix_ns)
                .max()
                .unwrap_or(reference_host_unix_ns);
            let waited_ns = latest_host_unix_ns
                .saturating_sub(reference_host_unix_ns)
                .max(0) as u128;
            let partial_ready = self.maximum_wait_ns.is_some_and(|maximum_wait_ns| {
                candidates.len() >= self.minimum_sensors && waited_ns >= maximum_wait_ns
            });
            if complete || partial_ready {
                let maximum_skew_ns = candidates
                    .values()
                    .map(|(_, timestamp)| timestamp.abs_diff(reference_time))
                    .max()
                    .unwrap_or(0);
                let mut frames = BTreeMap::new();
                for (name, (index, _)) in candidates {
                    let queue = self
                        .buffers
                        .get_mut(&name)
                        .expect("synchronizer queue disappeared");
                    for _ in 0..index {
                        queue.pop_front();
                        self.dropped_unmatched = self.dropped_unmatched.saturating_add(1);
                    }
                    let frame = queue
                        .pop_front()
                        .expect("synchronizer candidate disappeared");
                    frames.insert(name, frame);
                }
                if complete {
                    self.complete_sets = self.complete_sets.saturating_add(1);
                } else {
                    self.partial_sets = self.partial_sets.saturating_add(1);
                }
                output.push(SynchronizedFrameSet {
                    sequence: self.sequence,
                    timestamp_basis: reference_basis.to_string(),
                    timestamp_ns: reference_time,
                    maximum_skew_ns,
                    frames,
                });
                self.sequence = self.sequence.saturating_add(1);
                continue;
            }

            let all_present = self.sensor_names.iter().all(|name| {
                self.buffers
                    .get(name)
                    .is_some_and(|queue| !queue.is_empty())
            });
            let partial_expired = self
                .maximum_wait_ns
                .is_some_and(|maximum_wait_ns| waited_ns >= maximum_wait_ns);
            if all_present || partial_expired {
                self.drop_front(&reference_name);
                continue;
            }
            break;
        }
        output
    }

    fn drop_front(&mut self, sensor_name: &str) {
        if self
            .buffers
            .get_mut(sensor_name)
            .and_then(VecDeque::pop_front)
            .is_some()
        {
            self.dropped_unmatched = self.dropped_unmatched.saturating_add(1);
        }
    }
}

fn frame_sync_time(frame: &FrameRecord) -> Option<(&'static str, i128)> {
    let timestamps = &frame.metadata.timestamps;
    timestamps
        .normalized_unix_ns
        .map(|timestamp| ("normalized_unix_ns", timestamp))
        .or_else(|| {
            timestamps
                .source_ns
                .map(|timestamp| ("source_ns", timestamp))
        })
        .or_else(|| {
            (timestamps.host_unix_ns > 0).then_some(("host_unix_ns", timestamps.host_unix_ns))
        })
}

pub fn pairwise_sync_report(
    reference: &[FrameMetadata],
    other: &[FrameMetadata],
) -> Result<PairwiseSyncReport, String> {
    let reference = timestamp_series(reference);
    let other = timestamp_series(other);
    if reference.is_empty() || other.is_empty() {
        return Ok(PairwiseSyncReport {
            timestamp_basis: "none".into(),
            reference_samples: reference.len(),
            other_samples: other.len(),
            matched_samples: 0,
            median_signed_offset_ns: None,
            median_absolute_offset_ns: None,
            p95_absolute_offset_ns: None,
            maximum_absolute_offset_ns: None,
        });
    }
    let basis = reference[0].0;
    if reference.iter().any(|(candidate, _)| *candidate != basis) {
        return Err("reference recording contains multiple timestamp bases".into());
    }
    if other.iter().any(|(candidate, _)| *candidate != basis) {
        return Err(format!(
            "timestamp basis mismatch: reference uses {basis}, other contains multiple bases"
        ));
    }
    let mut other_times: Vec<i128> = other.into_iter().map(|(_, timestamp)| timestamp).collect();
    other_times.sort_unstable();
    let mut signed_offsets = Vec::with_capacity(reference.len());
    for (_, timestamp) in reference.iter().copied() {
        let index = other_times.partition_point(|candidate| *candidate < timestamp);
        let nearest = match (
            other_times.get(index),
            index.checked_sub(1).and_then(|i| other_times.get(i)),
        ) {
            (Some(after), Some(before)) => {
                if (after - timestamp).unsigned_abs() < (before - timestamp).unsigned_abs() {
                    *after
                } else {
                    *before
                }
            }
            (Some(after), None) => *after,
            (None, Some(before)) => *before,
            (None, None) => continue,
        };
        signed_offsets.push(nearest - timestamp);
    }
    signed_offsets.sort_unstable();
    let mut absolute_offsets: Vec<u128> = signed_offsets
        .iter()
        .map(|offset| offset.unsigned_abs())
        .collect();
    absolute_offsets.sort_unstable();
    Ok(PairwiseSyncReport {
        timestamp_basis: basis.into(),
        reference_samples: reference.len(),
        other_samples: other_times.len(),
        matched_samples: signed_offsets.len(),
        median_signed_offset_ns: Some(median(&signed_offsets)),
        median_absolute_offset_ns: Some(median_unsigned(&absolute_offsets)),
        p95_absolute_offset_ns: Some(percentile_unsigned(&absolute_offsets, 95)),
        maximum_absolute_offset_ns: absolute_offsets.last().copied(),
    })
}

fn timestamp_series(metadata: &[FrameMetadata]) -> Vec<(&'static str, i128)> {
    metadata
        .iter()
        .filter_map(|frame| {
            let timestamps = &frame.timestamps;
            if let Some(timestamp) = timestamps.normalized_unix_ns {
                Some(("normalized_unix_ns", timestamp))
            } else if let Some(timestamp) = timestamps.source_ns {
                Some(("source_ns", timestamp))
            } else if timestamps.host_unix_ns > 0 {
                Some(("host_unix_ns", timestamps.host_unix_ns))
            } else if let Some(timestamp) = timestamps.pipeline_pts_ns {
                Some(("pipeline_pts_ns", timestamp as i128))
            } else {
                None
            }
        })
        .collect()
}

#[derive(Debug, Clone)]
pub struct ClockOffsetEstimator {
    capacity: usize,
    samples: VecDeque<i128>,
}

impl ClockOffsetEstimator {
    pub fn new(capacity: usize) -> Result<Self, String> {
        if capacity == 0 {
            return Err("clock estimator capacity must be positive".into());
        }
        Ok(Self {
            capacity,
            samples: VecDeque::with_capacity(capacity),
        })
    }

    /// Add `reference - source`, both in the same time domain.
    pub fn observe(&mut self, reference_ns: i128, source_ns: i128) {
        if self.samples.len() == self.capacity {
            self.samples.pop_front();
        }
        self.samples.push_back(reference_ns - source_ns);
    }

    pub fn assessment(&self) -> SyncAssessment {
        if self.samples.is_empty() {
            return SyncAssessment {
                sample_count: 0,
                median_offset_ns: None,
                median_absolute_deviation_ns: None,
                min_offset_ns: None,
                max_offset_ns: None,
            };
        }
        let mut values: Vec<_> = self.samples.iter().copied().collect();
        values.sort_unstable();
        let median = median(&values);
        let mut deviations: Vec<_> = values
            .iter()
            .map(|value| (value - median).unsigned_abs())
            .collect();
        deviations.sort_unstable();
        SyncAssessment {
            sample_count: values.len(),
            median_offset_ns: Some(median),
            median_absolute_deviation_ns: Some(median_unsigned(&deviations)),
            min_offset_ns: values.first().copied(),
            max_offset_ns: values.last().copied(),
        }
    }
}

fn median(values: &[i128]) -> i128 {
    values[values.len() / 2]
}

fn median_unsigned(values: &[u128]) -> u128 {
    values[values.len() / 2]
}

fn percentile_unsigned(values: &[u128], percentile: usize) -> u128 {
    let index = ((values.len() - 1) * percentile / 100).min(values.len() - 1);
    values[index]
}

#[cfg(test)]
mod tests {
    use super::*;

    fn frame(sensor_name: &str, timestamp: i128) -> FrameRecord {
        FrameRecord {
            metadata: FrameMetadata {
                sensor_name: sensor_name.into(),
                sensor_kind: crate::SensorKind::PoE,
                sequence: timestamp as u64,
                profile: crate::StreamProfile {
                    stream: "main".into(),
                    width: 1,
                    height: 1,
                    fps_num: 1,
                    fps_den: 1,
                    format: crate::PixelFormat::H264,
                },
                timestamps: crate::FrameTimestamps {
                    source_ns: None,
                    source_domain: crate::TimestampDomain::Unknown,
                    rtp_timestamp: None,
                    pipeline_pts_ns: None,
                    pipeline_dts_ns: None,
                    host_monotonic_ns: 1,
                    host_unix_ns: timestamp,
                    normalized_unix_ns: None,
                },
                dropped_before: 0,
                calibration_id: None,
                flags: vec![],
                attributes: Default::default(),
            },
            payload: crate::RecordedPayload::Encoded {
                format: crate::PixelFormat::H264,
                bytes: vec![1],
            },
        }
    }

    #[test]
    fn robustly_estimates_offset() {
        let mut estimator = ClockOffsetEstimator::new(8).unwrap();
        for offset in [100, 101, 99, 100, 10_000] {
            estimator.observe(offset, 0);
        }
        let assessment = estimator.assessment();
        assert_eq!(assessment.median_offset_ns, Some(100));
        assert_eq!(assessment.sample_count, 5);
        assert_eq!(assessment.min_offset_ns, Some(99));
        assert_eq!(assessment.max_offset_ns, Some(10_000));
    }

    #[test]
    fn compares_independent_timestamp_series() {
        let make = |timestamp: i128| FrameMetadata {
            sensor_name: "sensor".into(),
            sensor_kind: crate::SensorKind::PoE,
            sequence: 0,
            profile: crate::StreamProfile {
                stream: "main".into(),
                width: 1,
                height: 1,
                fps_num: 1,
                fps_den: 1,
                format: crate::PixelFormat::H264,
            },
            timestamps: crate::FrameTimestamps {
                source_ns: None,
                source_domain: crate::TimestampDomain::Unknown,
                rtp_timestamp: None,
                pipeline_pts_ns: Some(timestamp as u64),
                pipeline_dts_ns: None,
                host_monotonic_ns: 1,
                host_unix_ns: 0,
                normalized_unix_ns: None,
            },
            dropped_before: 0,
            calibration_id: None,
            flags: vec![],
            attributes: Default::default(),
        };
        let report = pairwise_sync_report(
            &[make(1_000), make(2_000), make(3_000)],
            &[make(1_100), make(2_100), make(3_100)],
        )
        .unwrap();
        assert_eq!(report.timestamp_basis, "pipeline_pts_ns");
        assert_eq!(report.median_signed_offset_ns, Some(100));
        assert_eq!(report.maximum_absolute_offset_ns, Some(100));
    }

    #[test]
    fn assembles_only_frames_within_tolerance() {
        let mut synchronizer =
            FrameSynchronizer::new(["camera1".to_string(), "camera2".to_string()], 10, 4).unwrap();
        assert!(
            synchronizer
                .push(frame("camera1", 1_000))
                .unwrap()
                .is_empty()
        );
        let sets = synchronizer.push(frame("camera2", 1_007)).unwrap();
        assert_eq!(sets.len(), 1);
        assert_eq!(sets[0].sequence, 0);
        assert_eq!(sets[0].maximum_skew_ns, 7);
        assert_eq!(sets[0].frames.len(), 2);

        assert!(
            synchronizer
                .push(frame("camera2", 2_000))
                .unwrap()
                .is_empty()
        );
        let sets = synchronizer.push(frame("camera1", 2_020)).unwrap();
        assert!(sets.is_empty());
        assert!(synchronizer.dropped_unmatched() > 0);
    }

    #[test]
    fn partial_mode_waits_then_emits_a_fresh_subset() {
        let mut synchronizer = FrameSynchronizer::new_partial(
            [
                "camera1".to_string(),
                "camera2".to_string(),
                "camera3".to_string(),
            ],
            2,
            10,
            50,
            4,
        )
        .unwrap();
        assert!(
            synchronizer
                .push(frame("camera1", 1_000))
                .unwrap()
                .is_empty()
        );
        assert!(
            synchronizer
                .push(frame("camera2", 1_005))
                .unwrap()
                .is_empty()
        );
        let sets = synchronizer.push(frame("camera3", 1_065)).unwrap();
        assert_eq!(sets.len(), 1);
        assert_eq!(sets[0].frames.len(), 2);
        assert!(sets[0].frames.contains_key("camera1"));
        assert!(sets[0].frames.contains_key("camera2"));
        assert_eq!(synchronizer.partial_sets(), 1);
        assert_eq!(synchronizer.complete_sets(), 0);
    }

    #[test]
    fn partial_mode_still_emits_complete_sets_without_waiting() {
        let mut synchronizer = FrameSynchronizer::new_partial(
            [
                "camera1".to_string(),
                "camera2".to_string(),
                "camera3".to_string(),
            ],
            2,
            10,
            50,
            4,
        )
        .unwrap();
        assert!(
            synchronizer
                .push(frame("camera1", 1_000))
                .unwrap()
                .is_empty()
        );
        assert!(
            synchronizer
                .push(frame("camera2", 1_005))
                .unwrap()
                .is_empty()
        );
        let sets = synchronizer.push(frame("camera3", 1_007)).unwrap();
        assert_eq!(sets.len(), 1);
        assert_eq!(sets[0].frames.len(), 3);
        assert_eq!(synchronizer.complete_sets(), 1);
        assert_eq!(synchronizer.partial_sets(), 0);
    }
}
