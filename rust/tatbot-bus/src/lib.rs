//! Versioned bus envelope and bounded latest-value test transport.
//! This bus is never part of the e-stop control path.
pub mod fleet;
#[cfg(feature = "zenoh")]
pub mod service;
#[cfg(feature = "zenoh")]
pub mod transport;
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct Envelope<T> {
    pub schema: String,
    pub producer: Producer,
    pub stamp: Stamp,
    pub seq: u64,
    pub payload: T,
}
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct Producer {
    pub node: String,
    pub pid: u32,
    pub sha: String,
    pub run_id: String,
}
impl Producer {
    /// This process on `node`: the run id its launcher exported (TATBOT_RUN_ID,
    /// else `<program>-<pid>`) and the source commit its release was built from
    /// (TATBOT_SOURCE_COMMIT at build time, else "development").
    pub fn this_process(node: impl Into<String>, program: &str) -> Self {
        let pid = std::process::id();
        Self {
            node: node.into(),
            pid,
            sha: option_env!("TATBOT_SOURCE_COMMIT").unwrap_or("development").into(),
            run_id: std::env::var("TATBOT_RUN_ID").unwrap_or_else(|_| format!("{program}-{pid}")),
        }
    }
}
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct Stamp {
    pub mono_ns: u64,
    pub wall_ns: u64,
    pub basis: String,
}
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct Liveliness {
    pub sha: String,
    pub schemas: Vec<String>,
    pub pid: u32,
    pub started_unix_ms: u64,
    #[serde(default)]
    pub metrics: BTreeMap<String, f64>,
}
