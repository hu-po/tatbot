//! Possible exception / error types that librealsense2 can produce

use num_derive::{FromPrimitive, ToPrimitive};
use realsense_sys as sys;
use std::fmt::{Display, Formatter, Result};

/// Enumeration of possible exception types that can be returned via `rs2_error`
///
/// `Rs2Exception` is an enumeration where each variant describes the class of error returned by an
/// [`rs2_error`](realsense_sys::rs2_error) pointer.
#[repr(i32)]
#[derive(FromPrimitive, ToPrimitive, Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Rs2Exception {
    /// Unknown error classification.
    ///
    /// This is context-dependent. Usually this means that the error isn't a specific class of
    /// librealsense2 errors (could just be `std::runtime_error` under the hood). You'll need to
    /// know what API was called at a low-level to be able to trace what this error actually means.
    Unknown = sys::rs2_exception_type_RS2_EXCEPTION_TYPE_UNKNOWN as i32,
    /// Error resulted because the device was disconnected.
    ///
    /// This can be caused by outside intervention (pulling the plug), by an internal firmware
    /// error or due to insufficient power to the camera / device.
    CameraDisconnected = sys::rs2_exception_type_RS2_EXCEPTION_TYPE_CAMERA_DISCONNECTED as i32,
    /// Error occurred from the underlying OS-specific layer.
    Backend = sys::rs2_exception_type_RS2_EXCEPTION_TYPE_BACKEND as i32,
    /// An invalid value was passed to the API.
    InvalidValue = sys::rs2_exception_type_RS2_EXCEPTION_TYPE_INVALID_VALUE as i32,
    /// Error resulted because a function precondition was violated.
    ///
    /// Usually this means that you tried to call a method before a class was properly initialized
    /// or configured. E.g. this type of error can occur if you try to wait for frames on a
    /// pipeline before it is started. We attempt to reduce the number of opportunities where this
    /// can happen by structuring the types around these constraints at a higher level.
    WrongApiCallSequence =
        sys::rs2_exception_type_RS2_EXCEPTION_TYPE_WRONG_API_CALL_SEQUENCE as i32,
    /// The method you tried to call is not implemented.
    NotImplemented = sys::rs2_exception_type_RS2_EXCEPTION_TYPE_NOT_IMPLEMENTED as i32,
    /// Error resulted because the device is in recovery mode
    ///
    /// The device might require a firmware update.
    DeviceInRecoveryMode =
        sys::rs2_exception_type_RS2_EXCEPTION_TYPE_DEVICE_IN_RECOVERY_MODE as i32,
    /// Error resulted because of an IO device failure.
    IoDeviceFailure = sys::rs2_exception_type_RS2_EXCEPTION_TYPE_IO as i32,
    /// Normally, this exception would not be included, since this just tells us the total number of
    /// exceptions. HOWEVER, there is a section of the SDK that does return this, which the
    /// maintainers use when, quote, "If you want to avoid any errors being reported because an
    /// exception an expected API behavior". We do not know what this means, or why it exists, but
    /// we include it for completeness.
    ///
    /// Reference:
    /// <https://github.com/IntelRealSense/librealsense/blob/cacc8e701a963a44b49fe51911443797e8b3a517/src/api.h#L438>
    ///
    Count = sys::rs2_exception_type_RS2_EXCEPTION_TYPE_COUNT as i32,
}

impl Display for Rs2Exception {
    fn fmt(&self, f: &mut Formatter<'_>) -> Result {
        let s = match self {
            Rs2Exception::Unknown => "Unknown",
            Rs2Exception::CameraDisconnected => "CameraDisconnected",
            Rs2Exception::Backend => "Backend",
            Rs2Exception::InvalidValue => "InvalidValue",
            Rs2Exception::WrongApiCallSequence => "WrongAPICallSequence",
            Rs2Exception::NotImplemented => "NotImplemented",
            Rs2Exception::DeviceInRecoveryMode => "DeviceInRecoveryMode",
            Rs2Exception::IoDeviceFailure => "IODeviceFailure",
            Rs2Exception::Count => "Count (not a real error)",
        };

        write!(f, "{}", s)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use num_traits::FromPrimitive;

    #[test]
    fn all_variants_exist() {
        for i in 0..sys::rs2_exception_type_RS2_EXCEPTION_TYPE_COUNT as i32 {
            assert!(
                Rs2Exception::from_i32(i).is_some(),
                "Rs2Exception variant for ordinal {} does not exist.",
                i,
            );
        }
    }
}
