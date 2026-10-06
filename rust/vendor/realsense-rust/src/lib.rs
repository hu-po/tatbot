#![doc = include_str!("../README.md")]

pub mod base;
pub mod config;
pub mod context;
pub mod device;
pub mod device_hub;
pub mod docs;
mod error;
/// Native SDK version checked for the optional raw-device-clock API.
#[cfg(feature = "raw-device-clock")]
pub const RAW_DEVICE_CLOCK_SDK_VERSION: &str = env!("TATBOT_RAW_CLOCK_SDK_VERSION");
pub mod frame;
pub mod kind;
pub mod pipeline;
pub mod processing_blocks;
pub mod sensor;
pub mod stream_profile;

/// The module collects common used traits from this crate.
pub mod prelude {
    pub use crate::frame::{FrameCategory, FrameEx};
}
