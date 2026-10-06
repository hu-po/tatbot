//! Types for representing data related to streams
//!
//! In librealsense2, a stream is an abstraction around a "sensor stream." In most applications,
//! every sensor in the API will probably only have one stream that you actively care about (but
//! not always!). That stream is tied to both the sensor it is configured on as well as the frames
//! that are produced from the stream. The stream contains metadata such as the stream kind, the
//! format of the incoming data, etc.
//!
//! There are two ways you can get streams in the system: from a stream profile list returned from
//! the `Sensor`, or the frame stream from a frame type. In both cases the stream profile has an
//! explicit lifetime that depends on the object you obtained it from. This is because stream
//! profile ownership is not the same depending on how you obtained it.
//!
//! # Things to watch out for
//!
//! Extrinsics and intrinsics between different streams are obtained via the stream profile.
//! However, your stream profile may not be valid if you get it from the frame. This is because
//! stream profiles for the frame at a low level are owned by librealsense2. Calls to
//! `rs2_get_frame_stream_profile` return a `*const rs2_stream_profile`, which is obtained by
//! a call to C++ `shared_ptr::get()` internally to librealsense2. Since this is owned and managed
//! by librealsense2, and calls to `shared_ptr::get()` aren't tracked by the internal ref-count on
//! the `shared_ptr`, this means that the pointer could be invalidated at any time. In practice,
//! this means that if you disconnect your device mid-way through streaming (i.e. yank the cable
//! out), you could be holding onto a `StreamProfile` with an invalid internal pointer.
//!
//! We attempt to cache some aspects of the stream profile ahead of time, but in some cases this is
//! not feasible (e.g. extrinsics). This is why some interfaces return `Result`s and others do not.
//! The interfaces that do not return `Result` types are cached or otherwise safe-to-assume as
//! such. The ones that do return `Result` types are will check the pointer at runtime and may
//! return an error if it is no longer valid.
//!
//! We recommend that you get extrinsic or intrinsic information ahead of time when the device is
//! connected (via `Device::sensor()` and then `Sensor::stream_profiles()`), rather than relying on
//! the stream profile obtained via the frame types. The streams will have the same unique
//! identifier if they correspond to the same stream.
//!
//! See [the `StreamProfile` type](crate::stream_profile::StreamProfile) for more information.

use crate::{
    base::{Rs2Extrinsics, Rs2Intrinsics, Rs2MotionDeviceIntrinsics},
    check_rs2_error,
    kind::{Rs2Exception, Rs2Format, Rs2StreamKind},
};
use anyhow::Result;
use num_traits::FromPrimitive;
use realsense_sys as sys;
use std::{convert::TryFrom, mem::MaybeUninit, ptr::NonNull};
use thiserror::Error;

/// Type describing errors that can occur when trying to construct a stream profile.
///
/// Follows the standard pattern of errors where the enum variant describes what the low-level code
/// was attempting to do while the string carried alongside describes the underlying error message
/// from any C++ exceptions that occur.
#[derive(Error, Debug)]
pub enum StreamConstructionError {
    /// Could not get stream data during construction.
    #[error("Could not retrieve stream data. Type: {0}; Reason: {1}")]
    CouldNotRetrieveStreamData(Rs2Exception, String),
    /// Could not determine if this stream is the default stream during construction.
    #[error("Could not determine if this is the default stream. Type: {0}; Reason: {1}")]
    CouldNotDetermineIsDefault(Rs2Exception, String),
    /// Could not get the stream profile from the stream profile list.
    ///
    /// Usually due to an internal exception of some kind.
    #[error("Could not get the stream profile from a stream profile list. Type: {0}; Reason: {1}")]
    CouldNotGetProfileFromList(Rs2Exception, String),
    /// Could not clone the stream profile after acquiring it from the profile list.
    #[error("Could not clone the stream profile after getting it from the profile list. Type: {0}; Reason: {1}")]
    CouldNotCloneProfile(Rs2Exception, String),
}

/// Type describing errors in getting or setting stream-related data.
///
/// Follows the standard pattern of errors where the enum variant describes what the low-level code
/// was attempting to do while the string carried alongside describes the underlying error message
/// from any C++ exceptions that occur.
#[derive(Error, Debug)]
pub enum DataError {
    /// Could not get extrinsics between the requested streams.
    #[error("Could not get extrinsics. Type: {0}; Reason: {1}")]
    CouldNotGetExtrinsics(Rs2Exception, String),
    /// Could not set extrinsics between the requested streams.
    #[error("Could not set extrinsics. Type: {0}; Reason: {1}")]
    CouldNotSetExtrinsics(Rs2Exception, String),
    /// This stream does not have video intrinsics.
    #[error("Stream does not have video intrinsics")]
    StreamDoesNotHaveVideoIntrinsics,
    /// This stream does not have motion intrinsics.
    #[error("Stream does not have motion intrinsics")]
    StreamDoesNotHaveMotionIntrinsics,
    /// Could not get video intrinsics from the requested stream.
    #[error("Could not get video intrinsics. Type: {0}; Reason: {1}")]
    CouldNotGetIntrinsics(Rs2Exception, String),
    /// Could not get motion intrinsics from the requested stream.
    #[error("Could not get motion intrinsics. Type: {0}; Reason: {1}")]
    CouldNotGetMotionIntrinsics(Rs2Exception, String),
}

/// Type for holding the stream profile information.
///
/// This type exists as a high-level wrapper around an underlying `rs2_stream_profile` pointer. On
/// construction, we cache a copy of the stream data and also cache whether or not this stream
/// profile is the default stream for a sensor.
///
/// # Lifetimes
///
/// Stream profiles are acquired one of three ways:
///
/// 1. The stream profile list via the [`stream_profiles`](crate::sensor::Sensor::stream_profiles))
///    method on the [`Sensor`](crate::sensor::Sensor) type.
/// 2. The stream profile list via the
///    [`streams`](crate::pipeline::PipelineProfile::streams) associated function on the
///    [`PipelineProfile`](crate::pipeline::PipelineProfile) type.
/// 3. The frame-specific `frame_stream_profile` member via the Frame type.
///
/// Stream profiles from the sensor can outlive the parent object that you obtain them from. In
/// cases two and three above we return references to a stream profile owned by that type, so they
/// may not.  In most cases you will probably want to grab the stream profile from the pipeline
/// profile, which will give you all streams that are actively streaming from a given pipeline.
#[derive(Debug)]
pub struct StreamProfile {
    // Underlying non-null pointer from realsense-sys.
    //
    // Unlike many other pointer types that we get from the ffi boundary, this pointer should not
    // be manually deleted using `rs2_delete_stream_profile`. Streams are owned and managed by
    // their corresponding sensor, which are owned and managed by their corresponding devices.
    // Stream profile pointers should only be manually deleted if they are created by
    /// `rs2_clone_stream_profile`, which we do not use in the high-level API.
    ptr: NonNull<sys::rs2_stream_profile>,
    /// The kind of stream (e.g. depth, video, accelerometer, gyroscope, etc.)
    stream: Rs2StreamKind,
    /// The bit format of the underlying data.
    ///
    /// For video streams this will describe how the pixels are packed and padded, for motion,
    /// pose, and point frame streams this will describe how to deconstruct individual points or
    /// observations.
    format: Rs2Format,
    /// The stream index. Useful if you wish to enable / disable certain streams by index.
    index: usize,
    /// The unique identifier for the stream.
    unique_id: i32,
    /// The framerate of the stream (how fast it outputs data)
    framerate: i32,
    /// Whether or not the stream is a default stream.
    is_default: bool,
    /// Whether or not to drop the profile
    should_drop: bool,
}

impl TryFrom<NonNull<sys::rs2_stream_profile>> for StreamProfile {
    type Error = StreamConstructionError;

    /// Attempt to create a stream profile from a pointer to an `rs2_stream_profile` type.
    ///
    /// # Errors
    ///
    /// Returns [`StreamConstructionError::CouldNotRetrieveStreamData`] if the stream data
    /// associated with this stream profile cannot be retrieved.
    ///
    /// Returns [`StreamConstructionError::CouldNotDetermineIsDefault`] if it cannot be determined
    /// whether or not this stream is a default stream. This usually will only happen if the stream
    /// is invalidated (e.g. due to a device disconnect) when you try to construct it.
    fn try_from(stream_profile_ptr: NonNull<sys::rs2_stream_profile>) -> Result<Self, Self::Error> {
        unsafe {
            let mut err = std::ptr::null_mut::<sys::rs2_error>();

            let mut stream = MaybeUninit::uninit();
            let mut format = MaybeUninit::uninit();
            let mut index = MaybeUninit::uninit();
            let mut unique_id = MaybeUninit::uninit();
            let mut framerate = MaybeUninit::uninit();

            sys::rs2_get_stream_profile_data(
                stream_profile_ptr.as_ptr(),
                stream.as_mut_ptr(),
                format.as_mut_ptr(),
                index.as_mut_ptr(),
                unique_id.as_mut_ptr(),
                framerate.as_mut_ptr(),
                &mut err,
            );
            check_rs2_error!(err, StreamConstructionError::CouldNotRetrieveStreamData)?;

            let is_default =
                sys::rs2_is_stream_profile_default(stream_profile_ptr.as_ptr(), &mut err);
            check_rs2_error!(err, StreamConstructionError::CouldNotDetermineIsDefault)?;

            Ok(StreamProfile {
                ptr: stream_profile_ptr,
                stream: Rs2StreamKind::from_i32(stream.assume_init() as i32).unwrap(),
                format: Rs2Format::from_i32(format.assume_init() as i32).unwrap(),
                index: index.assume_init() as usize,
                unique_id: unique_id.assume_init(),
                framerate: framerate.assume_init(),
                is_default: is_default != 0,
                should_drop: false,
            })
        }
    }
}

impl Drop for StreamProfile {
    fn drop(&mut self) {
        unsafe {
            if self.should_drop {
                sys::rs2_delete_stream_profile(self.ptr.as_ptr());
            }
        }
    }
}

impl StreamProfile {
    /// Attempt to construct a stream profile from a profile list and index.
    ///
    /// Constructs a new stream profile from the list and index, or returns an error.
    ///
    /// # Errors
    ///
    /// Returns [`StreamConstructionError::CouldNotGetProfileFromList`] if the stream profile
    /// cannot be acquired from the profile list. (e.g. index is invalid).
    ///
    /// Returns [`StreamConstructionError::CouldNotRetrieveStreamData`] if the stream data
    /// associated with this stream profile cannot be retrieved.
    ///
    /// Returns [`StreamConstructionError::CouldNotDetermineIsDefault`] if it cannot be determined
    /// whether or not this stream is a default stream. This usually will only happen if the stream
    /// is invalidated (e.g. due to a device disconnect) when you try to construct it.
    pub(crate) fn try_create(
        profiles: &NonNull<sys::rs2_stream_profile_list>,
        index: i32,
    ) -> Result<Self, StreamConstructionError> {
        unsafe {
            let mut err = std::ptr::null_mut::<sys::rs2_error>();
            let profile_ptr = sys::rs2_get_stream_profile(profiles.as_ptr(), index, &mut err);
            check_rs2_error!(err, StreamConstructionError::CouldNotGetProfileFromList)?;

            let mut stream = MaybeUninit::uninit();
            let mut format = MaybeUninit::uninit();
            let mut index = MaybeUninit::uninit();
            let mut unique_id = MaybeUninit::uninit();
            let mut framerate = MaybeUninit::uninit();

            sys::rs2_get_stream_profile_data(
                profile_ptr,
                stream.as_mut_ptr(),
                format.as_mut_ptr(),
                index.as_mut_ptr(),
                unique_id.as_mut_ptr(),
                framerate.as_mut_ptr(),
                &mut err,
            );
            check_rs2_error!(err, StreamConstructionError::CouldNotRetrieveStreamData)?;

            let profile_ptr = sys::rs2_clone_stream_profile(
                profile_ptr,
                stream.assume_init(),
                index.assume_init(),
                format.assume_init(),
                &mut err,
            );
            check_rs2_error!(err, StreamConstructionError::CouldNotCloneProfile)?;

            let nonnull_profile_ptr = NonNull::new(profile_ptr).unwrap();
            let mut stream_profile = Self::try_from(nonnull_profile_ptr)?;
            stream_profile.should_drop = true;

            Ok(stream_profile)
        }
    }

    /// Predicate for whether or not the stream is a default stream.
    #[inline]
    pub fn is_default(&self) -> bool {
        self.is_default
    }

    /// Gets the stream kind from the stream data.
    ///
    /// This can be e.g. Depth, Video, Accel, Gyro, etc.
    #[inline]
    pub fn kind(&self) -> Rs2StreamKind {
        self.stream
    }

    /// Gets the format for the underlying data.
    ///
    /// For video streams this will describe how the pixels are packed and padded, for motion,
    /// pose, and point frame streams this will describe how to deconstruct individual points or
    /// observations.
    #[inline]
    pub fn format(&self) -> Rs2Format {
        self.format
    }

    /// Gets the stream's index.
    ///
    /// This is useful if you want to enable / disable a particular stream according to its index.
    #[inline]
    pub fn index(&self) -> usize {
        self.index
    }

    /// Gets the stream's unique identifier.
    #[inline]
    pub fn unique_id(&self) -> i32 {
        self.unique_id
    }

    /// Gets the framerate / data rate of frames generated by the stream.
    #[inline]
    pub fn framerate(&self) -> i32 {
        self.framerate
    }

    /// Get extrinsics between the origin stream (`self`) and target stream (`to_profile`).
    ///
    /// Returns the extrinsics between the origin and target streams from the underlying realsense
    /// driver iff both underlying stream pointers are valid and extrinsics exist. Otherwise
    /// returns an error.
    ///
    /// # Errors
    ///
    /// Returns [`DataError::CouldNotGetExtrinsics`] if this call fails for whatever reason.
    pub fn extrinsics(&self, to_profile: &StreamProfile) -> Result<Rs2Extrinsics, DataError> {
        unsafe {
            let mut err = std::ptr::null_mut::<sys::rs2_error>();
            let mut extrinsics = MaybeUninit::<sys::rs2_extrinsics>::uninit();

            sys::rs2_get_extrinsics(
                self.ptr.as_ptr(),
                to_profile.ptr.as_ptr(),
                extrinsics.as_mut_ptr(),
                &mut err,
            );
            check_rs2_error!(err, DataError::CouldNotGetExtrinsics)?;

            Ok(Rs2Extrinsics(extrinsics.assume_init()))
        }
    }

    /// Set `extrinsics` between the origin stream (`self`) and target stream (`to_profile`).
    ///
    /// Returns null tuple `()` iff the streams are valid and the extrinsics are successfully set.
    /// Otherwise returns an error.
    ///
    /// # Errors
    ///
    /// Returns [`DataError::CouldNotSetExtrinsics`] if this call fails for whatever reason.
    pub fn set_extrinsics(
        &self,
        to_profile: &StreamProfile,
        extrinsics: Rs2Extrinsics,
    ) -> Result<(), DataError> {
        unsafe {
            let mut err = std::ptr::null_mut::<sys::rs2_error>();
            sys::rs2_register_extrinsics(
                self.ptr.as_ptr(),
                to_profile.ptr.as_ptr(),
                extrinsics.0,
                &mut err,
            );
            check_rs2_error!(err, DataError::CouldNotSetExtrinsics)?;

            Ok(())
        }
    }

    /// Get video intrinsics from the stream.
    ///
    /// Returns a set of video intrinsics for the stream iff the stream has video intrinsics and the stream
    /// pointer is valid. Otherwise returns an error.
    ///
    /// # Errors
    ///
    /// Returns [`DataError::StreamDoesNotHaveVideoIntrinsics`] if the stream does not have video
    /// intrinsics.
    ///
    /// Returns [`DataError::CouldNotGetIntrinsics`] if this call fails for any other reason.
    pub fn intrinsics(&self) -> Result<Rs2Intrinsics, DataError> {
        match self.stream {
            Rs2StreamKind::Depth => (),
            Rs2StreamKind::Color => (),
            Rs2StreamKind::Infrared => (),
            Rs2StreamKind::Fisheye => (),
            _ => {
                return Err(DataError::StreamDoesNotHaveVideoIntrinsics);
            }
        }
        unsafe {
            let mut err = std::ptr::null_mut::<sys::rs2_error>();
            let mut intrinsics = MaybeUninit::<sys::rs2_intrinsics>::uninit();

            sys::rs2_get_video_stream_intrinsics(
                self.ptr.as_ptr(),
                intrinsics.as_mut_ptr(),
                &mut err,
            );
            check_rs2_error!(err, DataError::CouldNotGetIntrinsics)?;

            Ok(Rs2Intrinsics(intrinsics.assume_init()))
        }
    }

    /// Get motion intrinsics from the stream.
    ///
    /// Returns a set of motion device intrinsics for the stream iff the stream has motion device
    /// intrinsics and the stream pointer is valid. Otherwise returns an error.
    ///
    /// # Errors
    ///
    /// Returns
    /// [`DataError::StreamDoesNotHaveMotionIntrinsics`]
    /// if the stream does not have motion intrinsics.
    ///
    /// Returns [`DataError::CouldNotGetMotionIntrinsics`]
    /// if this call fails for any other reason.
    pub fn motion_intrinsics(&self) -> Result<Rs2MotionDeviceIntrinsics, DataError> {
        match self.stream {
            Rs2StreamKind::Gyro => (),
            Rs2StreamKind::Accel => (),
            _ => {
                return Err(DataError::StreamDoesNotHaveMotionIntrinsics);
            }
        }
        unsafe {
            let mut err = std::ptr::null_mut::<sys::rs2_error>();
            let mut intrinsics = MaybeUninit::<sys::rs2_motion_device_intrinsic>::uninit();

            sys::rs2_get_motion_intrinsics(self.ptr.as_ptr(), intrinsics.as_mut_ptr(), &mut err);
            check_rs2_error!(err, DataError::CouldNotGetMotionIntrinsics)?;

            Ok(Rs2MotionDeviceIntrinsics(intrinsics.assume_init()))
        }
    }
}
