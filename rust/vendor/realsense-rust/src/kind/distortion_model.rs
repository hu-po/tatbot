//! Enumeration describing the possible kinds of distortions present in a stream profile.
//!
//! Different RealSense devices use different distortion models to describe their intrinsics. For instance, the D415
//! uses the Modified Brown Conrady distortion model on its color camera, while the SR300 uses the Inverse Brown
//! Conrady. Ultimately, the user will need to interrogate a Stream's intrinsics directly using `.intrinsics()` to be
//! sure about the model being used.
//!
//! See the Intel docs about [Projection in RealSense SDK
//! 2.0](https://dev.intelrealsense.com/docs/projection-in-intel-realsense-sdk-20) for more information.

use num_derive::{FromPrimitive, ToPrimitive};
use realsense_sys as sys;

/// An enum for the various kinds of distortion models provided by librealsense2.
#[repr(i32)]
#[derive(FromPrimitive, ToPrimitive, Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Rs2DistortionModel {
    /// Rectilinear images. No distortion compensation required.
    None = sys::rs2_distortion_RS2_DISTORTION_NONE as i32,
    /// Unmodified Brown-Conrady distortion model
    BrownConrady = sys::rs2_distortion_RS2_DISTORTION_BROWN_CONRADY as i32,
    /// Equivalent to Brown-Conrady distortion, except that tangential distortion is applied to
    /// radially distorted points
    BrownConradyModified = sys::rs2_distortion_RS2_DISTORTION_MODIFIED_BROWN_CONRADY as i32,
    /// Equivalent to Brown-Conrady distortion, except undistorts image instead of distorting it
    BrownConradyInverse = sys::rs2_distortion_RS2_DISTORTION_INVERSE_BROWN_CONRADY as i32,
    /// F-Theta fish-eye distortion model
    FThetaFisheye = sys::rs2_distortion_RS2_DISTORTION_FTHETA as i32,
    /// Four parameter Kannala Brandt distortion model
    KannalaBrandt = sys::rs2_distortion_RS2_DISTORTION_KANNALA_BRANDT4 as i32,
    // Number of enumeration values. Not included.
    //
    // Count = sys::rs2_distortion_RS2_DISTORTION_COUNT
}

#[cfg(test)]
mod tests {
    use super::*;
    use num_traits::FromPrimitive;

    #[test]
    fn all_variants_exist() {
        for i in 0..sys::rs2_distortion_RS2_DISTORTION_COUNT as i32 {
            assert!(
                Rs2DistortionModel::from_i32(i).is_some(),
                "DistortionModel variant for ordinal {} does not exist.",
                i,
            );
        }
    }
}
