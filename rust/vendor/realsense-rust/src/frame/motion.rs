//! Type for representing a RealSense Motion frame.
//!
//! Motion data for any Motion frame is held as a 3-vector. This data (retrieved
//! through `motion()`) represents different things depending on the device recorded.
//!
//! See the docs for [MotionFrame::motion] for more.

use super::prelude::{CouldNotGetFrameSensorError, FrameCategory, FrameConstructionError, FrameEx};
use crate::{
    check_rs2_error,
    kind::{Rs2Extension, Rs2FrameMetadata, Rs2StreamKind, Rs2TimestampDomain},
    sensor::Sensor,
    stream_profile::StreamProfile,
};
use anyhow::Result;
use num_traits::FromPrimitive;
use realsense_sys as sys;
use std::{
    convert::{TryFrom, TryInto},
    marker::PhantomData,
    ptr::{self, NonNull},
};

/// A unit struct defining an Accel frame.
#[derive(Debug)]
pub struct Accel;
/// A unit struct defining a Gyro frame.
#[derive(Debug)]
pub struct Gyro;

/// Holds the raw data pointer and derived data from an RS2 Motion Frame.
///
/// All fields in this struct are initialized during struct creation (via `try_from`).
/// Everything called from here during runtime should be valid as long as the
/// Frame is in scope... like normal Rust.
#[derive(Debug)]
pub struct MotionFrame<Kind> {
    /// The raw data pointer from the original rs2 frame.
    frame_ptr: NonNull<sys::rs2_frame>,
    /// The timestamp of the frame.
    timestamp: f64,
    /// The RealSense time domain from which the timestamp is derived.
    timestamp_domain: Rs2TimestampDomain,
    /// The frame number.
    frame_number: u64,
    /// The Stream Profile that created the frame.
    frame_stream_profile: StreamProfile,
    /// The motion data held in this Motion Frame. Motion data is represented as a
    /// 3-vector, with different conventions depending on the device recorded.
    /// See the [motion](MotionFrame::motion) function for more documentation.
    motion: [f32; 3],
    /// A boolean used during `Drop` calls. This allows for proper handling of the pointer
    /// during ownership transfer.
    should_drop: bool,
    /// Holds the type metadata of this frame.
    _phantom: PhantomData<Kind>,
}

/// A motion frame type holding the raw pointer and derived metadata for an RS2 Accel frame.
pub type AccelFrame = MotionFrame<Accel>;
/// A motion frame type holding the raw pointer and derived metadata for an RS2 Gyro frame.
pub type GyroFrame = MotionFrame<Gyro>;

impl FrameCategory for AccelFrame {
    fn extension() -> Rs2Extension {
        Rs2Extension::MotionFrame
    }

    fn kind() -> Rs2StreamKind {
        Rs2StreamKind::Accel
    }

    fn has_correct_kind(&self) -> bool {
        self.frame_stream_profile.kind() == Self::kind()
    }
}

impl FrameCategory for GyroFrame {
    fn extension() -> Rs2Extension {
        Rs2Extension::MotionFrame
    }

    fn kind() -> Rs2StreamKind {
        Rs2StreamKind::Gyro
    }

    fn has_correct_kind(&self) -> bool {
        self.frame_stream_profile.kind() == Self::kind()
    }
}

impl<K> Drop for MotionFrame<K> {
    /// Drop the raw pointer stored with this struct whenever it goes out of scope.
    fn drop(&mut self) {
        unsafe {
            if self.should_drop {
                sys::rs2_release_frame(self.frame_ptr.as_ptr());
            }
        }
    }
}

unsafe impl<K> Send for MotionFrame<K> {}

impl<K> TryFrom<NonNull<sys::rs2_frame>> for MotionFrame<K> {
    type Error = anyhow::Error;

    /// Attempt to create an Image frame of extension K from the raw `rs2_frame`. All
    /// members of the ImageFrame struct are validated and populated during this call.
    ///
    /// # Errors
    ///
    /// There are a number of errors that may occur if the data in the `rs2_frame` is not
    /// valid, all of type [FrameConstructionError].
    ///
    /// - [CouldNotGetTimestamp](FrameConstructionError::CouldNotGetTimestamp)
    /// - [CouldNotGetTimestampDomain](FrameConstructionError::CouldNotGetTimestampDomain)
    /// - [CouldNotGetFrameStreamProfile](FrameConstructionError::CouldNotGetFrameStreamProfile)
    /// - [CouldNotGetDataSize](FrameConstructionError::CouldNotGetDataSize)
    /// - [CouldNotGetData](FrameConstructionError::CouldNotGetData)
    ///
    /// See [FrameConstructionError] documentation for more details.
    ///
    fn try_from(frame_ptr: NonNull<sys::rs2_frame>) -> Result<Self, Self::Error> {
        unsafe {
            let mut err = ptr::null_mut::<sys::rs2_error>();

            let timestamp = sys::rs2_get_frame_timestamp(frame_ptr.as_ptr(), &mut err);
            check_rs2_error!(err, FrameConstructionError::CouldNotGetTimestamp)?;

            let timestamp_domain =
                sys::rs2_get_frame_timestamp_domain(frame_ptr.as_ptr(), &mut err);
            check_rs2_error!(err, FrameConstructionError::CouldNotGetTimestampDomain)?;

            let frame_number = sys::rs2_get_frame_number(frame_ptr.as_ptr(), &mut err);
            check_rs2_error!(err, FrameConstructionError::CouldNotGetFrameNumber)?;

            let profile_ptr = sys::rs2_get_frame_stream_profile(frame_ptr.as_ptr(), &mut err);
            check_rs2_error!(err, FrameConstructionError::CouldNotGetFrameStreamProfile)?;

            let nonnull_profile_ptr =
                NonNull::new(profile_ptr as *mut sys::rs2_stream_profile).unwrap();
            let profile = StreamProfile::try_from(nonnull_profile_ptr)?;

            let size = sys::rs2_get_frame_data_size(frame_ptr.as_ptr(), &mut err);
            check_rs2_error!(err, FrameConstructionError::CouldNotGetDataSize)?;

            let ptr = sys::rs2_get_frame_data(frame_ptr.as_ptr(), &mut err);
            check_rs2_error!(err, FrameConstructionError::CouldNotGetData)?;

            let data_as_ptr = ptr.as_ref().unwrap() as *const std::os::raw::c_void;
            let data_size_in_f32s = (size as usize) / std::mem::size_of::<f32>();

            let motion_raw =
                std::slice::from_raw_parts(data_as_ptr.cast::<f32>(), data_size_in_f32s);

            Ok(MotionFrame {
                frame_ptr,
                timestamp,
                timestamp_domain: Rs2TimestampDomain::from_i32(timestamp_domain as i32).unwrap(),
                frame_number,
                frame_stream_profile: profile,
                motion: [motion_raw[0], motion_raw[1], motion_raw[2]],
                should_drop: true,
                _phantom: PhantomData::<K> {},
            })
        }
    }
}

impl<K> FrameEx for MotionFrame<K> {
    fn stream_profile(&self) -> &StreamProfile {
        &self.frame_stream_profile
    }

    fn sensor(&self) -> Result<Sensor> {
        unsafe {
            let mut err = std::ptr::null_mut::<sys::rs2_error>();
            let sensor_ptr = sys::rs2_get_frame_sensor(self.frame_ptr.as_ptr(), &mut err);
            check_rs2_error!(err, CouldNotGetFrameSensorError)?;

            Ok(Sensor::from(NonNull::new(sensor_ptr).unwrap()))
        }
    }
    fn timestamp(&self) -> f64 {
        self.timestamp
    }

    fn timestamp_domain(&self) -> Rs2TimestampDomain {
        self.timestamp_domain
    }

    fn frame_number(&self) -> u64 {
        self.frame_number
    }

    fn metadata(&self, metadata_kind: Rs2FrameMetadata) -> Option<std::os::raw::c_longlong> {
        if !self.supports_metadata(metadata_kind) {
            return None;
        }

        unsafe {
            let mut err = std::ptr::null_mut::<sys::rs2_error>();

            let val = sys::rs2_get_frame_metadata(
                self.frame_ptr.as_ptr(),
                #[allow(clippy::useless_conversion)]
                (metadata_kind as i32).try_into().unwrap(),
                &mut err,
            );
            if err.as_ref().is_none() {
                Some(val)
            } else {
                sys::rs2_free_error(err);
                None
            }
        }
    }

    fn supports_metadata(&self, metadata_kind: Rs2FrameMetadata) -> bool {
        unsafe {
            let mut err = std::ptr::null_mut::<sys::rs2_error>();

            let supports_metadata = sys::rs2_supports_frame_metadata(
                self.frame_ptr.as_ptr(),
                #[allow(clippy::useless_conversion)]
                (metadata_kind as i32).try_into().unwrap(),
                &mut err,
            );

            if err.as_ref().is_none() {
                supports_metadata != 0
            } else {
                sys::rs2_free_error(err);
                false
            }
        }
    }

    unsafe fn get_owned_raw(mut self) -> NonNull<sys::rs2_frame> {
        self.should_drop = false;

        self.frame_ptr
    }
}

impl AccelFrame {
    /// Returns a 3-item array representing the sensor motion recorded in the Accel frame.
    ///
    /// Accelerations are reported as [x, y, z] values, and are in units of m/s^2
    ///
    /// This function will return different data conventions entirely depending on the device
    /// used to create the measurement.
    ///
    /// # Intel RealSense D435i
    ///
    /// - `motion[0]`: Positive x-axis points to the right.
    /// - `motion[1]`: Positive y-axis points down.
    /// - `motion[2]`: Positive z-axis points forward.
    ///
    /// # Intel RealSense T265
    ///
    /// - `motion[0]`: Positive X direction is towards right imager.
    /// - `motion[1]`: Positive Y direction is upwards toward the top of the device.
    /// - `motion[2]`: Positive Z direction is inwards toward the back of the device.
    ///
    /// Read more about the coordinate frames of RealSense motion in
    /// [the RealSense docs](https://www.intelrealsense.com/how-to-getting-imu-data-from-d435i-and-t265/)
    ///
    pub fn acceleration(&self) -> &[f32; 3] {
        &self.motion
    }
}

impl GyroFrame {
    /// Returns a 3-item array representing the sensor motion recorded in the Gyro frame.
    ///
    /// Gyroscope measurements are reported as [x, y, z] values, and are in units of radians/s
    ///
    /// This function will return different data conventions entirely depending on the device
    /// used to create the measurement.
    ///
    /// # Intel RealSense D435i
    ///
    /// - `motion[0]`: Positive x-axis points to the right.
    /// - `motion[1]`: Positive y-axis points down.
    /// - `motion[2]`: Positive z-axis points forward.
    ///
    /// # Intel RealSense T265
    ///
    /// - `motion[0]`: Positive X direction is towards right imager.
    /// - `motion[1]`: Positive Y direction is upwards toward the top of the device.
    /// - `motion[2]`: Positive Z direction is inwards toward the back of the device.
    ///
    /// Read more about the coordinate frames of RealSense motion in
    /// [the RealSense docs](https://www.intelrealsense.com/how-to-getting-imu-data-from-d435i-and-t265/)
    ///
    pub fn rotational_velocity(&self) -> &[f32; 3] {
        &self.motion
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn frame_has_correct_kind() {
        assert_eq!(AccelFrame::kind(), Rs2StreamKind::Accel);
        assert_eq!(GyroFrame::kind(), Rs2StreamKind::Gyro);
    }
}
