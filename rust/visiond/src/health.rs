use std::collections::VecDeque;

use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HealthSnapshot {
    pub sensor_name: String,
    pub frames_received: u64,
    pub frames_dropped: u64,
    pub reconnects: u64,
    pub timestamp_regressions: u64,
    pub last_sequence: Option<u64>,
    pub last_host_monotonic_ns: Option<u128>,
    pub recent_errors: Vec<String>,
}

#[derive(Debug, Clone)]
pub struct SensorHealth {
    snapshot: HealthSnapshot,
    error_limit: usize,
    errors: VecDeque<String>,
}

impl SensorHealth {
    pub fn new(sensor_name: impl Into<String>) -> Self {
        Self {
            snapshot: HealthSnapshot {
                sensor_name: sensor_name.into(),
                frames_received: 0,
                frames_dropped: 0,
                reconnects: 0,
                timestamp_regressions: 0,
                last_sequence: None,
                last_host_monotonic_ns: None,
                recent_errors: Vec::new(),
            },
            error_limit: 16,
            errors: VecDeque::new(),
        }
    }

    pub fn frame_received(&mut self, sequence: u64, host_monotonic_ns: u128) {
        self.snapshot.frames_received += 1;
        if let Some(previous) = self.snapshot.last_sequence {
            if sequence > previous.saturating_add(1) {
                self.snapshot.frames_dropped += sequence - previous - 1;
            }
        }
        if let Some(previous) = self.snapshot.last_host_monotonic_ns {
            if host_monotonic_ns < previous {
                self.snapshot.timestamp_regressions += 1;
            }
        }
        self.snapshot.last_sequence = Some(sequence);
        self.snapshot.last_host_monotonic_ns = Some(host_monotonic_ns);
    }

    pub fn frame_dropped(&mut self, count: u64) {
        self.snapshot.frames_dropped += count;
    }

    pub fn reconnected(&mut self) {
        self.snapshot.reconnects += 1;
    }

    pub fn error(&mut self, error: impl Into<String>) {
        let error = error.into();
        if self.errors.len() == self.error_limit {
            self.errors.pop_front();
        }
        self.errors.push_back(error);
        self.snapshot.recent_errors = self.errors.iter().cloned().collect();
    }

    pub fn snapshot(&self) -> HealthSnapshot {
        self.snapshot.clone()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn detects_sequence_gaps_and_timestamp_regressions() {
        let mut health = SensorHealth::new("camera1");
        health.frame_received(0, 10);
        health.frame_received(2, 9);
        let snapshot = health.snapshot();
        assert_eq!(snapshot.frames_received, 2);
        assert_eq!(snapshot.frames_dropped, 1);
        assert_eq!(snapshot.timestamp_regressions, 1);
    }
}
