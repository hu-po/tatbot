//! Versioned bus envelope and bounded latest-value test transport.
//! This bus is never part of the e-stop control path.
pub mod capture;
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
