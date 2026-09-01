//! Reader for wxai_teleop flight-recorder logs (`.wxtl`).
//!
//! The format is defined by `cpp/teleop/wxai_teleop.cpp`: a 64-byte
//! little-endian header followed by fixed-size records of
//! `5 + 6 * num_joints` f64 values per teleop tick. A truncated trailing
//! record (e.g. from a killed process) is dropped rather than rejected.

use std::path::Path;

use anyhow::{Context, Result, bail};
use serde::{Deserialize, Serialize};

const LIVE_MAGIC: &str = "tatbot-teleop-joints";
const LIVE_VERSION: u32 = 1;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LiveTeleopTick {
    pub magic: String,
    pub version: u32,
    pub timestamp_ns: i64,
    pub sequence: u64,
    pub leader_pos: Vec<f64>,
    pub follower_pos: Vec<f64>,
    pub target: Vec<f64>,
    pub follower_eff: Vec<f64>,
}

impl LiveTeleopTick {
    pub fn parse(bytes: &[u8]) -> Result<Self> {
        let tick: Self = serde_json::from_slice(bytes).context("parsing live teleop JSON")?;
        tick.validate()?;
        Ok(tick)
    }

    pub fn validate(&self) -> Result<()> {
        if self.magic != LIVE_MAGIC || self.version != LIVE_VERSION {
            bail!("unsupported live teleop telemetry contract");
        }
        let joints = self.leader_pos.len();
        if joints == 0 || joints > 32 {
            bail!("implausible live teleop joint count {joints}");
        }
        for (name, values) in [
            ("leader_pos", &self.leader_pos),
            ("follower_pos", &self.follower_pos),
            ("target", &self.target),
            ("follower_eff", &self.follower_eff),
        ] {
            if values.len() != joints || values.iter().any(|value| !value.is_finite()) {
                bail!("{name} must contain {joints} finite values");
            }
        }
        if self.timestamp_ns <= 0 {
            bail!("live teleop timestamp must be positive");
        }
        Ok(())
    }
}

const MAGIC: &[u8; 8] = b"WXTLOG1\0";
const HEADER_LEN: usize = 64;

#[derive(Debug)]
pub struct TeleopLog {
    pub num_joints: usize,
    pub period_s: f64,
    pub tau_s: f64,
    pub goal_time_s: f64,
    pub ff_gain: f64,
    pub abs_gripper: bool,
    /// System-clock time of the first loop tick, nanoseconds since the epoch.
    pub wall_start_ns: i64,
    pub ticks: Vec<TeleopTick>,
}

#[derive(Debug, Clone)]
pub struct TeleopTick {
    /// Seconds since the loop started, monotonic.
    pub t_sched: f64,
    pub t_wake: f64,
    pub t_leader_read: f64,
    pub t_follower_read: f64,
    pub t_cmd: f64,
    pub leader_pos: Vec<f64>,
    pub leader_vel: Vec<f64>,
    pub follower_pos: Vec<f64>,
    pub follower_vel: Vec<f64>,
    pub follower_eff: Vec<f64>,
    pub target: Vec<f64>,
}

impl TeleopLog {
    pub fn read_file(path: impl AsRef<Path>) -> Result<Self> {
        let path = path.as_ref();
        let bytes = std::fs::read(path)
            .with_context(|| format!("reading teleop log {}", path.display()))?;
        Self::parse(&bytes).with_context(|| format!("parsing teleop log {}", path.display()))
    }

    fn parse(bytes: &[u8]) -> Result<Self> {
        if bytes.len() < HEADER_LEN {
            bail!("file is shorter than the {HEADER_LEN}-byte header");
        }
        if &bytes[0..8] != MAGIC {
            bail!("bad magic; not a wxai_teleop flight log");
        }
        let u64_at =
            |offset: usize| u64::from_le_bytes(bytes[offset..offset + 8].try_into().unwrap());
        let f64_at =
            |offset: usize| f64::from_le_bytes(bytes[offset..offset + 8].try_into().unwrap());

        let num_joints = usize::try_from(u64_at(8)).context("num_joints out of range")?;
        if num_joints == 0 || num_joints > 32 {
            bail!("implausible num_joints {num_joints}");
        }
        let record_values = 5 + 6 * num_joints;
        let record_bytes = record_values * 8;
        let payload = &bytes[HEADER_LEN..];
        let tick_count = payload.len() / record_bytes;

        let mut ticks = Vec::with_capacity(tick_count);
        for index in 0..tick_count {
            let record = &payload[index * record_bytes..(index + 1) * record_bytes];
            let value = |slot: usize| {
                f64::from_le_bytes(record[slot * 8..slot * 8 + 8].try_into().unwrap())
            };
            let joints = |first_slot: usize| {
                (0..num_joints)
                    .map(|joint| value(first_slot + joint))
                    .collect::<Vec<_>>()
            };
            ticks.push(TeleopTick {
                t_sched: value(0),
                t_wake: value(1),
                t_leader_read: value(2),
                t_follower_read: value(3),
                t_cmd: value(4),
                leader_pos: joints(5),
                leader_vel: joints(5 + num_joints),
                follower_pos: joints(5 + 2 * num_joints),
                follower_vel: joints(5 + 3 * num_joints),
                follower_eff: joints(5 + 4 * num_joints),
                target: joints(5 + 5 * num_joints),
            });
        }
        if ticks.is_empty() {
            bail!("log contains no complete records");
        }

        Ok(Self {
            num_joints,
            period_s: f64_at(16),
            tau_s: f64_at(24),
            goal_time_s: f64_at(32),
            ff_gain: f64_at(40),
            abs_gripper: u64_at(48) != 0,
            wall_start_ns: u64_at(56) as i64,
            ticks,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_bytes(num_joints: usize, ticks: usize) -> Vec<u8> {
        let mut bytes = Vec::new();
        bytes.extend_from_slice(MAGIC);
        bytes.extend_from_slice(&(num_joints as u64).to_le_bytes());
        for value in [0.0025_f64, 0.02, 0.005, 0.1] {
            bytes.extend_from_slice(&value.to_le_bytes());
        }
        bytes.extend_from_slice(&1_u64.to_le_bytes());
        bytes.extend_from_slice(&1_755_536_420_000_000_000_i64.to_le_bytes());
        for tick in 0..ticks {
            for slot in 0..(5 + 6 * num_joints) {
                bytes.extend_from_slice(&((tick * 100 + slot) as f64).to_le_bytes());
            }
        }
        bytes
    }

    #[test]
    fn parses_header_and_records() {
        let log = TeleopLog::parse(&sample_bytes(7, 3)).unwrap();
        assert_eq!(log.num_joints, 7);
        assert_eq!(log.ticks.len(), 3);
        assert!(log.abs_gripper);
        assert_eq!(log.period_s, 0.0025);
        assert_eq!(log.ticks[1].t_sched, 100.0);
        assert_eq!(log.ticks[1].leader_pos[0], 105.0);
        assert_eq!(log.ticks[2].target[6], 246.0);
    }

    #[test]
    fn drops_truncated_trailing_record() {
        let mut bytes = sample_bytes(7, 2);
        bytes.truncate(bytes.len() - 24);
        let log = TeleopLog::parse(&bytes).unwrap();
        assert_eq!(log.ticks.len(), 1);
    }

    #[test]
    fn rejects_bad_magic() {
        let mut bytes = sample_bytes(7, 1);
        bytes[0] = b'X';
        assert!(TeleopLog::parse(&bytes).is_err());
    }

    #[test]
    fn live_tick_contract_validates_shape_and_identity() {
        let bytes = br#"{
            "magic":"tatbot-teleop-joints","version":1,
            "timestamp_ns":1755536420000000000,"sequence":7,
            "leader_pos":[1,2],"follower_pos":[3,4],
            "target":[5,6],"follower_eff":[7,8]
        }"#;
        let tick = LiveTeleopTick::parse(bytes).unwrap();
        assert_eq!(tick.sequence, 7);
        let mut wrong = tick;
        wrong.follower_pos.pop();
        assert!(wrong.validate().is_err());
    }
}
