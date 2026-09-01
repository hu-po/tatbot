use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TimestampDomain {
    Unknown,
    CameraNtp,
    Rtp,
    RealSenseGlobal,
    RealSenseHardware,
    HostMonotonic,
    HostUnix,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ClockSample {
    pub monotonic_ns: u128,
    pub unix_ns: i128,
}

#[derive(Debug, Clone)]
pub struct MonotonicClock {
    started: Instant,
}

impl Default for MonotonicClock {
    fn default() -> Self {
        Self {
            started: Instant::now(),
        }
    }
}

impl MonotonicClock {
    pub fn now(&self) -> ClockSample {
        let monotonic_ns = self.started.elapsed().as_nanos();
        let unix_ns = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or(Duration::ZERO);
        ClockSample {
            monotonic_ns,
            unix_ns: unix_ns.as_nanos() as i128,
        }
    }
}

pub fn duration_to_ns(duration: Duration) -> u128 {
    duration.as_nanos()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn monotonic_clock_increases() {
        let clock = MonotonicClock::default();
        let first = clock.now();
        std::thread::sleep(Duration::from_millis(1));
        let second = clock.now();
        assert!(second.monotonic_ns > first.monotonic_ns);
        assert!(second.unix_ns >= first.unix_ns);
    }
}
