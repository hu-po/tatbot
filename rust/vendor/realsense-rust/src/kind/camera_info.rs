//! Enumeration of sensor and device information keys.

use num_derive::{FromPrimitive, ToPrimitive};
use realsense_sys as sys;

/// A type describing the different keys used to access camera info from devices and sensors.
///
/// Each key corresponds to a particular type of device or sensor-specific metadata (known as
/// `info` in the librealsense2 API). Not all keys are supported on all devices or sensors.
///
/// All values that correspond to these keys are returned in the lower level API as `const char*`
/// types, or C-strings. We wrap these values in the `realsense-rust` API as `&CStr` types.
///
#[repr(i32)]
#[derive(FromPrimitive, ToPrimitive, Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Rs2CameraInfo {
    /// The name of the sensor or device.
    Name = sys::rs2_camera_info_RS2_CAMERA_INFO_NAME as i32,
    /// The serial number of the device.
    SerialNumber = sys::rs2_camera_info_RS2_CAMERA_INFO_SERIAL_NUMBER as i32,
    /// The firmware version that the device is running.
    FirmwareVersion = sys::rs2_camera_info_RS2_CAMERA_INFO_FIRMWARE_VERSION as i32,
    /// The recommended firmware version for a given device.
    ///
    /// The value that corresponds to this key may change depending on what version of
    /// librealsense2 this crate is built against!
    RecommendedFirmwareVersion =
        sys::rs2_camera_info_RS2_CAMERA_INFO_RECOMMENDED_FIRMWARE_VERSION as i32,
    /// A description of the unique identifier of the physical port that the device is connected to.
    ///
    /// The format of the value associated with this key will be platform specific.
    PhysicalPort = sys::rs2_camera_info_RS2_CAMERA_INFO_PHYSICAL_PORT as i32,
    /// If the device supports firmware logging, this is the command you send to get those logs.
    DebugOpCode = sys::rs2_camera_info_RS2_CAMERA_INFO_DEBUG_OP_CODE as i32,
    /// Tells you whether the device is in advanced mode.
    AdvancedMode = sys::rs2_camera_info_RS2_CAMERA_INFO_ADVANCED_MODE as i32,
    /// The product identifier for the device (as reported by its USB descriptor).
    ProductId = sys::rs2_camera_info_RS2_CAMERA_INFO_PRODUCT_ID as i32,
    /// Tells you whether the EEPROM is locked.
    CameraLocked = sys::rs2_camera_info_RS2_CAMERA_INFO_CAMERA_LOCKED as i32,
    /// Tells you the designated USB specification (i.e. USB2 or USB3).
    UsbTypeDescriptor = sys::rs2_camera_info_RS2_CAMERA_INFO_USB_TYPE_DESCRIPTOR as i32,
    /// Device product line (e.g. D400 / SR300 / L500 / T200)
    ProductLine = sys::rs2_camera_info_RS2_CAMERA_INFO_PRODUCT_LINE as i32,
    /// The ASIC serial number of the device.
    AsicSerialNumber = sys::rs2_camera_info_RS2_CAMERA_INFO_ASIC_SERIAL_NUMBER as i32,
    /// Provides the firmware update identifier for the device.
    FirmwareUpdateId = sys::rs2_camera_info_RS2_CAMERA_INFO_FIRMWARE_UPDATE_ID as i32,
    /// IP address for remote camera.
    IpAddress = sys::rs2_camera_info_RS2_CAMERA_INFO_IP_ADDRESS as i32,
    /// DFU Device node path
    DfuDevicePath = sys::rs2_camera_info_RS2_CAMERA_INFO_DFU_DEVICE_PATH as i32,
    /// Connection type, for example USB, GMSL, DDS, etc.
    ConnectionType = sys::rs2_camera_info_RS2_CAMERA_INFO_CONNECTION_TYPE as i32,
    // Not included since this just tells us the total number of camera info options
    //
    // Count = sys::rs2_camera_info_RS2_CAMERA_INFO_COUNT,
}

#[cfg(test)]
mod tests {
    use super::*;
    use num_traits::FromPrimitive;

    #[test]
    fn all_variants_exist() {
        for i in 0..sys::rs2_camera_info_RS2_CAMERA_INFO_COUNT as i32 {
            assert!(
                Rs2CameraInfo::from_i32(i).is_some(),
                "Rs2CameraInfo variant for ordinal {} does not exist.",
                i,
            );
        }
    }
}
