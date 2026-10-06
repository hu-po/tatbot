//! Contains commonly used enumerations and configuration types used across the crate.
//!
//! The `Kind` module holds a variety of types useful across realsense-rust APIs as transient
//! values. The vast majority of types in this module are enums, which map to C-style enums in the
//! lower-level `realsense-sys` bindings.
//!
//! # Architecture
//!
//! The low-level bindings for librealsense2 in `realsense-sys` represent the C-style enums as
//! `u32` constants. They are wrapped / transformed into fully qualified types here so as to
//! increase type safety across the API.
//!
//! All of these "wrapper" enums in Rust implement the [`ToPrimitive`](num_traits::ToPrimitive) and
//! [`FromPrimitive`](num_traits::FromPrimitive) traits from the `num_traits` crate. If you need to
//! access the original enum value, you can do so with the following code:
//!
//! ```rust
//! use num_traits::ToPrimitive;
//! use realsense_rust::kind::Rs2Extension;
//!
//! let ext = Rs2Extension::ColorSensor;
//! println!("The extension is: {}", ext.to_u32().unwrap());
//! ```
//!
//! In practice, most of the time you shouldn't need to wrap or unwrap `u32` values, and the API
//! should never spit one out at you.
//!
//! ## What happened to `COUNT`?
//!
//! If you read through the librealsense2 C-API, you'll find that every enumeration has a "count"
//! entry. Placing a "count" entry at the end of a C-style enum is a common pattern used to be able
//! to iterate over every possible enum variant. This works because C treats every enum variant as
//! a distinct `u32` value, instead of a distinct variant of a type. In this crate, we take the
//! approach that this is unlikely to be something that is undertaken, so we don't provide any
//! direct means to iterate through every value of each enum variant.
//!
//! If you truly need to do this, let us know! There are crates that make this possible and
//! integrate well with native Rust iterators, but we have chosen to avoid the extra depedencies
//! for this crate.
//!
//! If you need a workaround, you can always use the lower-level `realsense-sys` API:
//!
//! ```rust
//! use num_traits::FromPrimitive;
//! use realsense_sys as sys;
//! use realsense_rust::kind::Rs2CameraInfo;
//!
//! for i in 0..sys::rs2_camera_info_RS2_CAMERA_INFO_COUNT as i32 {
//!     println!(
//!         "The enum variant {:?} corresponds to the i32 value {}",
//!         Rs2CameraInfo::from_i32(i).unwrap(),
//!         i,
//!     );
//! }
//! ```
//!

mod camera_info;
mod color_scheme;
mod distortion_model;
mod exception;
mod extension;
mod format;
mod frame_metadata;
mod hole_filling;
mod option;
mod persistence_control;
mod product_line;
mod stream_kind;
mod timestamp_domain;

pub use camera_info::Rs2CameraInfo;
pub use color_scheme::ColorScheme;
pub use distortion_model::Rs2DistortionModel;
pub use exception::Rs2Exception;
pub use extension::{
    Rs2Extension, DEVICE_EXTENSIONS, FILTER_EXTENSIONS, FRAME_EXTENSIONS, MISC_EXTENSIONS,
    PROFILE_EXTENSIONS, SENSOR_EXTENSIONS,
};
pub use format::Rs2Format;
pub use frame_metadata::Rs2FrameMetadata;
pub use hole_filling::HoleFillingMode;
pub use option::{OptionSetError, Rs2Option, Rs2OptionRange};
pub use persistence_control::PersistenceControl;
pub use product_line::Rs2ProductLine;
pub use stream_kind::Rs2StreamKind;
pub use timestamp_domain::Rs2TimestampDomain;
