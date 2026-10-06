//! Common types and functions.

use crate::kind::Rs2DistortionModel;
use num_traits::FromPrimitive;
use realsense_sys as sys;
use std::{ffi::CString, time::Duration};

/// The default timeout duration in librealsense2
pub const DEFAULT_TIMEOUT: Duration = Duration::from_millis(sys::RS2_DEFAULT_TIMEOUT as u64);

/// Helper function for converting a path to a series of `c_char` that can be interpreted as a
/// sequence of bytes / native path for a given platform..
pub(crate) fn from_path<P>(path: P) -> anyhow::Result<CString>
where
    P: AsRef<std::path::Path>,
{
    // Thanks, Tenders McChiken.
    // https://stackoverflow.com/questions/38948669/whats-the-most-direct-way-to-convert-a-path-to-a-c-char
    let mut buf = Vec::new();

    #[cfg(unix)]
    {
        use std::os::unix::ffi::OsStrExt;
        buf.extend(path.as_ref().as_os_str().as_bytes());
    };

    #[cfg(windows)]
    {
        use std::os::windows::ffi::OsStrExt;
        buf.extend(
            path.as_ref()
                .as_os_str()
                .encode_wide()
                .chain(Some(0))
                .map(|b| {
                    let b = b.to_ne_bytes();
                    b.get(0).copied().into_iter().chain(b.get(1).copied())
                })
                .flatten(),
        );
    };

    Ok(CString::new(buf)?)
}

/// Newtype wrapper for RealSense motion device intrinsics
#[derive(Debug)]
pub struct Rs2MotionDeviceIntrinsics(pub sys::rs2_motion_device_intrinsic);

/// Profile the scale, bias, and variances for a given motion device
///
/// The bias and scale factors are stored as one large matrix; see the documentation on `data()` for the correct way to
/// retrieve these parameters.
///
/// Use the function `stream_profile.motion_intrinsics()` to retrieve these intrinsics from a certain stream.
impl Rs2MotionDeviceIntrinsics {
    /// A 3x4 matrix describing the scale and bias intrinsics of the motion device.
    ///
    /// This matrix is stored internally like so:
    /// [ Scale X    | cross axis  | cross axis | Bias X ]
    /// [ cross axis | Scale Y     | cross axis | Bias Y ]
    /// [ cross axis | cross axis  | Scale Z    | Bias Z ]
    ///
    pub fn data(&self) -> [[f32; 4usize]; 3usize] {
        self.0.data
    }
    /// Variance of noise for X, Y, and Z axis.
    pub fn noise_variances(&self) -> [f32; 3usize] {
        self.0.noise_variances
    }
    /// Variance of bias for X, Y, and Z axis.
    pub fn bias_variances(&self) -> [f32; 3usize] {
        self.0.bias_variances
    }
}

unsafe impl Send for Rs2MotionDeviceIntrinsics {}

/// Type representing the intrinsic scale, bias, and variances for a given motion device.
///
/// The data in `coeffs` means different things for different models.
///
/// - Brown-Conrady: [k1, k2, p1, p2, k3].
/// - F-Theta Fisheye: [k1, k2, k3, k4, 0].
/// - Kannala-Brandt: [k1, k2, k3, k4, 0].
///
/// The Intel RealSense documentation claims that "Other models are subject to their own interpretations". This is
/// admittedly not too helpful, but it's worth noting in case your model isn't covered here.
#[derive(Debug)]
pub struct Rs2Distortion {
    /// Distortion model of the image.
    pub model: Rs2DistortionModel,
    /// Distortion coefficients.
    pub coeffs: [f32; 5usize],
}

unsafe impl Send for Rs2Distortion {}

/// Type representing the model for describing the way that light bends in a stream.
///
/// This stores the focal length, principal point, dimensions, and distortion model used on the image frame. See the
/// documentation for [Rs2Distortion] for specifics on the available distortion models for RealSense devices.
///
/// Use the function `stream_profile.intrinsics()` to retrieve these intrinsics from a certain stream.
#[derive(Debug)]
pub struct Rs2Intrinsics(pub sys::rs2_intrinsics);

impl Rs2Intrinsics {
    /// Width of the image in pixels
    pub fn width(&self) -> usize {
        self.0.width as usize
    }
    /// Height of the image in pixels
    pub fn height(&self) -> usize {
        self.0.height as usize
    }

    /// Horizontal coordinate of the principal point of the image, as a pixel offset from the left edge
    pub fn ppx(&self) -> f32 {
        self.0.ppx
    }
    /// Vertical coordinate of the principal point of the image, as a pixel offset from the top edge
    pub fn ppy(&self) -> f32 {
        self.0.ppy
    }
    /// Focal length of the image plane, as a multiple of pixel width
    pub fn fx(&self) -> f32 {
        self.0.fx
    }
    /// Focal length of the image plane, as a multiple of pixel height
    pub fn fy(&self) -> f32 {
        self.0.fy
    }
    /// Distortion model and coefficients of the image
    pub fn distortion(&self) -> Rs2Distortion {
        Rs2Distortion {
            model: Rs2DistortionModel::from_i32(self.0.model as i32).unwrap(),
            coeffs: self.0.coeffs,
        }
    }
}

unsafe impl Send for Rs2Intrinsics {}

/// The topology describing how the different devices are oriented.
///
/// Use the function `stream_profile.extrinsics()` to retrieve these extrinsics from a certain stream in relation to
/// another stream on the same device.
#[derive(Debug)]
pub struct Rs2Extrinsics(pub sys::rs2_extrinsics);

impl Rs2Extrinsics {
    /// Column-major 3x3 rotation matrix
    pub fn rotation(&self) -> [f32; 9usize] {
        self.0.rotation
    }
    /// Three-element translation vector, in meters
    pub fn translation(&self) -> [f32; 3usize] {
        self.0.translation
    }
}

unsafe impl Send for Rs2Extrinsics {}

/// Region of interest for the auto exposure algorithm.
#[derive(Debug, Clone)]
pub struct Rs2Roi {
    /// Left coordinate of the region of interest.
    pub min_x: i32,
    /// Top coordinate of the region of interest.
    pub min_y: i32,
    /// Right coordinate of the region of interest.
    pub max_x: i32,
    /// Bottom coordinate of the region of interest.
    pub max_y: i32,
}
