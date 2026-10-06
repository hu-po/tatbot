//! Enumeration describing the possible kinds of streams a stream profile can describe.
//!
//! Streams are different types of data provided by RealSense devices.

use num_derive::{FromPrimitive, ToPrimitive};
use realsense_sys as sys;

/// The enumeration of possible stream kinds.
///
/// These are typically used when configuring a [`pipeline`](crate::pipeline::InactivePipeline), or
/// obtained from a [`StreamProfile`](crate::stream_profile::StreamProfile).
#[repr(i32)]
#[derive(FromPrimitive, ToPrimitive, Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Rs2StreamKind {
    /// Stream kind key to tell librealsense2 to pick the best suited stream kind.
    ///
    /// Unlike the other format entries, `Any` is used primarily when setting up streams in the
    /// config / pipeline. If you pass this, librealsense2 will pick the best suited (default)
    /// stream kinds for a given sensor.
    Any = sys::rs2_stream_RS2_STREAM_ANY as i32,
    /// Native stream of depth data produced by RealSense device
    Depth = sys::rs2_stream_RS2_STREAM_DEPTH as i32,
    /// Native stream of color data captured by RealSense device
    Color = sys::rs2_stream_RS2_STREAM_COLOR as i32,
    /// Native stream of infrared data captured by RealSense device
    Infrared = sys::rs2_stream_RS2_STREAM_INFRARED as i32,
    /// Native stream of fish-eye (wide) data captured from the dedicated motion camera
    Fisheye = sys::rs2_stream_RS2_STREAM_FISHEYE as i32,
    /// Native stream of gyroscope motion data produced by RealSense device
    Gyro = sys::rs2_stream_RS2_STREAM_GYRO as i32,
    /// Native stream of accelerometer motion data produced by RealSense device
    Accel = sys::rs2_stream_RS2_STREAM_ACCEL as i32,
    /// Signals from external device connected through GPIO
    Gpio = sys::rs2_stream_RS2_STREAM_GPIO as i32,
    /// 6DoF pose data, calculated by RealSense device
    Pose = sys::rs2_stream_RS2_STREAM_POSE as i32,
    /// 4-bit per pixel depth confidence values
    Confidence = sys::rs2_stream_RS2_STREAM_CONFIDENCE as i32,
    /// Native stream of combined motion data (incl. accel & gyro)
    Motion = sys::rs2_stream_RS2_STREAM_MOTION as i32,
    /* Not included since this just tells us the total number stream types
     *
     * Count = sys::rs2_stream_RS2_STREAM_COUNT, */
}

impl std::fmt::Display for Rs2StreamKind {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let string = match self {
            Rs2StreamKind::Any => "Any",
            Rs2StreamKind::Depth => "Depth",
            Rs2StreamKind::Color => "Color",
            Rs2StreamKind::Infrared => "Infrared",
            Rs2StreamKind::Fisheye => "Fisheye",
            Rs2StreamKind::Gyro => "Gyro",
            Rs2StreamKind::Accel => "Accel",
            Rs2StreamKind::Gpio => "Gpio",
            Rs2StreamKind::Pose => "Pose",
            Rs2StreamKind::Confidence => "Confidence",
            Rs2StreamKind::Motion => "Motion",
        };
        write!(f, "{}", string)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use num_traits::FromPrimitive;

    #[test]
    fn all_variants_exist() {
        for i in 0..sys::rs2_stream_RS2_STREAM_COUNT as i32 {
            assert!(
                Rs2StreamKind::from_i32(i).is_some(),
                "Rs2StreamKind variant for ordinal {} does not exist.",
                i,
            );
        }
    }
}
