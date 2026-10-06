//! Traits for describing frame operations.
//!
//! Many rs2 frame types hold certain data structures in common, but have
//! different functionalities. These are encapsulated by the `*FrameEx` traits,
//! with the wildcard describing the specialization that goes with that type.

use crate::{
    kind::{Rs2Exception, Rs2Extension, Rs2FrameMetadata, Rs2StreamKind, Rs2TimestampDomain},
    sensor::Sensor,
    stream_profile::StreamProfile,
};
use anyhow::Result;
use realsense_sys as sys;
use std::ptr::NonNull;
use thiserror::Error;

/// How many bits are in a byte? Who can truly say.
pub const BITS_PER_BYTE: i32 = 8;

/// Occurs when a frame type cannot be constructed from the given data.
#[derive(Error, Debug)]
pub enum FrameConstructionError {
    /// Could not get frame width.
    #[error("Could not get frame width. Type: {0}; Reason: {1}")]
    CouldNotGetWidth(Rs2Exception, String),
    /// Could not get frame height.
    #[error("Could not get frame height. Type: {0}; Reason: {1}")]
    CouldNotGetHeight(Rs2Exception, String),
    /// Could not get the pixel stride.
    #[error("Could not get stride. Type: {0}; Reason: {1}")]
    CouldNotGetStride(Rs2Exception, String),
    /// Could not get the bit count per pixel.
    #[error("Could not get bits-per-pixel. Type: {0}; Reason: {1}")]
    CouldNotGetBitsPerPixel(Rs2Exception, String),
    /// Could not get the frame timestamp.
    #[error("Could not get timestamp. Type: {0}; Reason: {1}")]
    CouldNotGetTimestamp(Rs2Exception, String),
    /// Could not get the frame timestamp's time domain, e.g. to which
    /// clock its time is relative.
    #[error("Could not get timestamp domain. Type: {0}; Reason: {1}")]
    CouldNotGetTimestampDomain(Rs2Exception, String),
    /// Could not get the frame number.
    #[error("Could not get frame number. Type: {0}; Reason: {1}")]
    CouldNotGetFrameNumber(Rs2Exception, String),
    /// Could not get the stream profile that describes the frame.
    #[error("Could not get frame stream profile. Type: {0}; Reason: {1}")]
    CouldNotGetFrameStreamProfile(Rs2Exception, String),
    /// Could not get the total data size of the frame in bytes.
    #[error("Could not get data size (in bytes). Type: {0}; Reason: {1}")]
    CouldNotGetDataSize(Rs2Exception, String),
    /// Could not get the data of the frame.
    #[error("Could not get pointer to frame data. Type: {0}; Reason: {1}")]
    CouldNotGetData(Rs2Exception, String),
    /// Could not get the number of points in a Points frame.
    #[error("Could not get number of points: Type: {0}; Reason: {1}")]
    CouldNotGetPointCount(Rs2Exception, String),
}

/// Occurs when certain data cannot be derived from a Depth frame.
#[derive(Error, Debug)]
pub enum DepthError {
    /// Cannot derive distance.
    #[error("Could not get distance. Type: {0}; Reason: {1}")]
    CouldNotGetDistance(Rs2Exception, String),
    /// Cannot derive the depth units used.
    #[error("Could not get depth units. Type: {0}; Reason: {1}")]
    CouldNotGetDepthUnits(Rs2Exception, String),
}

/// Occurs when a baseline cannot be derived from a Disparity frame.
#[derive(Error, Debug)]
#[error("Could not get baseline. Type: {0}; Reason: {1}")]
pub struct DisparityError(pub Rs2Exception, pub String);

/// Cannot get the frame sensor.
#[derive(Error, Debug)]
#[error("Could not get frame sensor. Type: {0}; Reason: {1}")]
pub struct CouldNotGetFrameSensorError(pub Rs2Exception, pub String);

/// Describes common functionality across frame types.
pub trait FrameEx {
    /// Get the stream profile associated with the frame.
    fn stream_profile(&self) -> &StreamProfile;

    /// Get the sensor associated with the frame.
    fn sensor(&self) -> Result<Sensor>;

    /// Get the frame number.
    fn frame_number(&self) -> u64;

    /// Get the frame timestamp.
    fn timestamp(&self) -> f64;

    /// Get the RealSense timestamp domain for the current timestamp.
    fn timestamp_domain(&self) -> Rs2TimestampDomain;

    /// Get frame metadata.
    ///
    /// Returns `None` if the `metadata_kind` is not supported by the frame type.
    fn metadata(&self, metadata_kind: Rs2FrameMetadata) -> Option<std::os::raw::c_longlong>;

    /// Test whether the metadata arguemnt is supported by the frame.
    fn supports_metadata(&self, metadata_kind: Rs2FrameMetadata) -> bool;

    /// Get (and own) the underlying frame pointer for this frame.
    ///
    /// This is primarily useful for passing this frame forward to a processing block or blocks
    /// (either via frame queue, directly, callback, etc).
    ///
    /// # Safety
    ///
    /// This does not destroy the underlying frame pointer once self
    /// goes out of scope. Instead, the program expects that whatever
    /// object was assigned to by this function now manages the lifetime.
    unsafe fn get_owned_raw(self) -> NonNull<sys::rs2_frame>;
}

/// A trait for specifying which runtime stream kinds can be held within a frame type
///
/// This trait changes some of the semantics for how to think about librealsense2 frames. The
/// reason for this is because frames in librealsense2 are more or less defined by three things:
///
/// 1. The data format ([`Rs2Format`](crate::kind::Rs2Format))
/// 2. The extension type ([`Rs2Extension`])
/// 3. The "stream kind" ([`Rs2StreamKind`])
///
/// Knowing these three things, you can uniquely describe any frame. We aim for our types to be
/// categorically distinct. Unfortunately, all three of the data points above are not encoded in
/// the type information for a frame in librealsense2, but are rather things we check at runtime.
///
pub trait FrameCategory {
    /// Identifies the corresponding [`Rs2Extension`] for the type implementing this trait.
    fn extension() -> Rs2Extension;

    /// Identifies the stream kind corresponding to a given type implementing this trait.
    fn kind() -> Rs2StreamKind;

    /// Predicate for checking if the RS2 frame's stream has the same kind as the frame category.
    fn has_correct_kind(&self) -> bool;
}
