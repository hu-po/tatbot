//! Type for representing a pose frame taken from an IMU or pose-sensor.
//!
//! A Pose frame holds information describing the motion and position of a device
//! at a point in time. See the member and function declarations for how these values are stored
//! and retrieved.

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
    mem::MaybeUninit,
    ptr::{self, NonNull},
};

/// Holds information describing the motion and position of a device at a point in time.
#[derive(Debug)]
pub struct PoseFrame {
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
    /// The rs2 Pose data
    data: sys::rs2_pose,
    /// A boolean used during `Drop` calls. This allows for proper handling of the pointer
    /// during ownership transfer.
    should_drop: bool,
}

/// Used by the tracker and mapper to estimate the certainty in this pose.
pub enum Confidence {
    /// The tracker/mapper has failed. This information is probably not reliable.
    Failed,
    /// The tracker/mapper confidence is low.
    Low,
    /// The tracker/mapper confidence is marginal.
    Medium,
    /// The tracker/mapper confidence is high.
    High,
}

impl PoseFrame {
    /// X, Y, Z values of translation, in meters (relative to initial position)
    pub fn translation(&self) -> [f32; 3] {
        let sys::rs2_vector { x, y, z } = self.data.translation;
        [x, y, z]
    }

    /// X, Y, Z values of velocity, in meters/sec
    pub fn velocity(&self) -> [f32; 3] {
        let sys::rs2_vector { x, y, z } = self.data.velocity;
        [x, y, z]
    }

    /// X, Y, Z values of acceleration, in meters/sec^2
    pub fn acceleration(&self) -> [f32; 3] {
        let sys::rs2_vector { x, y, z } = self.data.acceleration;
        [x, y, z]
    }

    /// Qi, Qj, Qk, Qr components of rotation as represented in quaternion rotation (relative to initial position)
    pub fn rotation(&self) -> [f32; 4] {
        let sys::rs2_quaternion { x, y, z, w } = self.data.rotation;
        [x, y, z, w]
    }

    /// X, Y, Z values of angular velocity, in radians/sec
    pub fn angular_velocity(&self) -> [f32; 3] {
        let sys::rs2_vector { x, y, z } = self.data.angular_velocity;
        [x, y, z]
    }

    /// X, Y, Z values of angular acceleration, in radians/sec^2
    pub fn angular_acceleration(&self) -> [f32; 3] {
        let sys::rs2_vector { x, y, z } = self.data.angular_acceleration;
        [x, y, z]
    }

    /// Pose confidence from [`Confidence::Failed`] to [`Confidence::High`]
    pub fn tracker_confidence(&self) -> Confidence {
        match self.data.tracker_confidence {
            0x0 => Confidence::Failed,
            0x1 => Confidence::Low,
            0x2 => Confidence::Medium,
            0x3 => Confidence::High,
            _ => panic!("Unknown confidence, please report a bug!"),
        }
    }

    /// Pose map confidence from [`Confidence::Failed`] to [`Confidence::High`]
    pub fn mapper_confidence(&self) -> Confidence {
        match self.data.tracker_confidence {
            0x0 => Confidence::Failed,
            0x1 => Confidence::Low,
            0x2 => Confidence::Medium,
            0x3 => Confidence::High,
            _ => panic!("Unknown confidence, please report a bug!"),
        }
    }
}

impl Drop for PoseFrame {
    /// Drop the raw pointer stored with this struct whenever it goes out of scope.
    fn drop(&mut self) {
        unsafe {
            if self.should_drop {
                sys::rs2_release_frame(self.frame_ptr.as_ptr());
            }
        }
    }
}

unsafe impl Send for PoseFrame {}

impl FrameCategory for PoseFrame {
    fn extension() -> Rs2Extension {
        Rs2Extension::PoseFrame
    }

    fn kind() -> Rs2StreamKind {
        Rs2StreamKind::Pose
    }

    fn has_correct_kind(&self) -> bool {
        self.frame_stream_profile.kind() == Self::kind()
    }
}

impl TryFrom<NonNull<sys::rs2_frame>> for PoseFrame {
    type Error = anyhow::Error;

    /// Attempt to construct a PoseFrame from the raw pointer to `rs2_frame`
    ///
    /// All members of the `PoseFrame` struct are validated and populated during this call.
    ///
    /// # Errors
    ///
    /// There are a number of errors that may occur if the data in the `rs2_frame` is not valid, all
    /// of type [`FrameConstructionError`].
    ///
    /// - [`CouldNotGetTimestamp`](FrameConstructionError::CouldNotGetTimestamp)
    /// - [`CouldNotGetTimestampDomain`](FrameConstructionError::CouldNotGetTimestampDomain)
    /// - [`CouldNotGetFrameStreamProfile`](FrameConstructionError::CouldNotGetFrameStreamProfile)
    /// - [`CouldNotGetData`](FrameConstructionError::CouldNotGetData)
    ///
    /// See [`FrameConstructionError`] documentation for more details.
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

            let mut pose_data = MaybeUninit::uninit();
            sys::rs2_pose_frame_get_pose_data(frame_ptr.as_ptr(), pose_data.as_mut_ptr(), &mut err);
            check_rs2_error!(err, FrameConstructionError::CouldNotGetData)?;

            Ok(PoseFrame {
                frame_ptr,
                timestamp,
                timestamp_domain: Rs2TimestampDomain::from_i32(timestamp_domain as i32).unwrap(),
                frame_number,
                frame_stream_profile: profile,
                data: pose_data.assume_init(),
                should_drop: true,
            })
        }
    }
}

impl FrameEx for PoseFrame {
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn frame_has_correct_kind() {
        assert_eq!(PoseFrame::kind(), Rs2StreamKind::Pose);
    }
}
