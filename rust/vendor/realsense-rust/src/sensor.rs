//! Type for abstracting over the concept of a Sensor
//!
//! The words device and sensor are overloaded in this industry, and librealsense2 makes no
//! attempts to fix that. While the words device and sensor are often used colloquially, here
//! `Device` refers to a whole Realsense package, such as a D435i, L515, etc. `Sensor` refers to
//! "sub-devices"; rather, the individual sensing components that one can stream from. A sensor may
//! be an IMU, which has accelerometer and gyroscope streams, or a camera of some kind, which has
//! streams related to different types of images.
//!
//! The hierarchy is effectively:
//!
//! [`Device`] |-> [`Sensor`] |-> [`StreamProfile`]

use crate::{
    base::Rs2Roi,
    check_rs2_error,
    device::{Device, DeviceConstructionError},
    kind::{
        OptionSetError, Rs2CameraInfo, Rs2Exception, Rs2Extension, Rs2Option, Rs2OptionRange,
        SENSOR_EXTENSIONS,
    },
    stream_profile::StreamProfile,
};
use anyhow::Result;
use realsense_sys as sys;
use std::{
    convert::{From, TryInto},
    ffi::CStr,
    mem::MaybeUninit,
    ptr::NonNull,
};
use thiserror::Error;

/// Type describing errors that can occur when trying to construct a sensor.
///
/// Follows the standard pattern of errors where the enum variant describes what the low-level code
/// was attempting to do while the string carried alongside describes the underlying error message
/// from any C++ exceptions that occur.
#[derive(Error, Debug)]
pub enum SensorConstructionError {
    /// Could not get the correct sensor from the sensor list.
    #[error("Could not get correct sensor from sensor list. Type: {0}; Reason: {1}")]
    CouldNotGetSensorFromList(Rs2Exception, String),
}

/// Type describing errors that can occur when trying to set the region of interest of a sensor.
///
/// Follows the standard pattern of errors where the enum variant describes what the low-level code
/// was attempting to do while the string carried alongside describes the underlying error message
/// from any C++ exceptions that occur.
#[derive(Error, Debug)]
pub enum RoiSetError {
    /// Could not set region of interest for sensor.
    #[error("Could not set region of interest for sensor. Type: {0}; Reason: {1}")]
    CouldNotSetRoi(Rs2Exception, String),
}

/// Type for holding sensor-related data.
///
/// A sensor in librealsense2 corresponds to a physical component on the unit in some way, shape,
/// or form. These may or may not correspond to multiple streams. e.g. an IMU on the device may
/// correspond to accelerometer and gyroscope streams, or an IR camera sensor on the device may
/// correspond to depth & video streams.
///
/// Sensors are constructed one of two ways:
///
/// 1. From the device's [sensor list](crate::device::Device::sensors)
/// 2. By getting the sensor that [corresponds to a given frame](crate::frame::FrameEx::sensor)
pub struct Sensor {
    /// The underlying non-null sensor pointer.
    ///
    /// This should not be deleted unless the sensor was constructed via `rs2_create_sensor`
    sensor_ptr: NonNull<sys::rs2_sensor>,
    /// Boolean used for telling us if we should drop the sensor pointer or not.
    should_drop: bool,
}

impl Drop for Sensor {
    fn drop(&mut self) {
        unsafe {
            if self.should_drop {
                sys::rs2_delete_sensor(self.sensor_ptr.as_ptr());
            }
        }
    }
}

unsafe impl Send for Sensor {}

impl std::convert::From<NonNull<sys::rs2_sensor>> for Sensor {
    /// Attempt to construct a Sensor from a non-null pointer to `rs2_sensor`.
    fn from(sensor_ptr: NonNull<sys::rs2_sensor>) -> Self {
        Sensor {
            sensor_ptr,
            should_drop: false,
        }
    }
}

impl Sensor {
    /// Create a sensor from a sensor list and an index
    ///
    /// Unlike when you directly acquire a `*mut rs2_sensor` from an API in librealsense2, such as
    /// when calling `rs2_get_frame_sensor`, you have to drop this pointer at the end (because you
    /// now own it). When calling `try_from` we don't want to drop in the default case, since our
    /// `*mut rs2_sensor` may not be owned by us, but by the device / frame / etc.
    ///
    /// The main difference then is that this API defaults to using `rs2_create_sensor` vs. a call
    /// to get a sensor from somewhere else.
    ///
    /// This can fail for similar reasons to `try_from`, and is likewise only valid if `index` is
    /// less than the length of `sensor_list` (see `rs2_get_sensors_count` for how to get that
    /// length).
    ///
    /// Guaranteeing the lifetime / semantics of the sensor is difficult, so this should probably
    /// not be used outside of this crate. See `crate::device::Device` for where this is used.
    ///
    /// # Errors
    ///
    /// Returns [`SensorConstructionError::CouldNotGetSensorFromList`] if the index is invalid or
    /// if the sensor list is invalid in some way.
    pub(crate) fn try_create(
        sensor_list: &NonNull<sys::rs2_sensor_list>,
        index: i32,
    ) -> Result<Self, SensorConstructionError> {
        unsafe {
            let mut err = std::ptr::null_mut::<sys::rs2_error>();

            let sensor_ptr = sys::rs2_create_sensor(sensor_list.as_ptr(), index, &mut err);
            check_rs2_error!(err, SensorConstructionError::CouldNotGetSensorFromList)?;

            let nonnull_ptr = NonNull::new(sensor_ptr).unwrap();
            let mut sensor = Sensor::from(nonnull_ptr);
            sensor.should_drop = true;
            Ok(sensor)
        }
    }

    /// Get the parent device that this sensor corresponds to.
    ///
    /// Returns the device that this sensor corresponds to iff that device is still connected and
    /// the sensor is still valid. Otherwise returns an error.
    ///
    /// # Errors
    ///
    /// Returns [`DeviceConstructionError::CouldNotCreateDeviceFromSensor`] if the device cannot be
    /// obtained due to the physical device being disconnected or the internal sensor pointer
    /// becoming invalid.
    pub fn device(&self) -> Result<Device, DeviceConstructionError> {
        unsafe {
            let mut err = std::ptr::null_mut::<sys::rs2_error>();
            let device_ptr = sys::rs2_create_device_from_sensor(self.sensor_ptr.as_ptr(), &mut err);
            check_rs2_error!(err, DeviceConstructionError::CouldNotCreateDeviceFromSensor)?;

            Ok(Device::from(NonNull::new(device_ptr).unwrap()))
        }
    }

    /// Get sensor extension.
    pub fn extension(&self) -> Rs2Extension {
        let ext = SENSOR_EXTENSIONS
            .iter()
            .find(|ext| unsafe {
                let mut err = std::ptr::null_mut::<sys::rs2_error>();
                let is_extendable = sys::rs2_is_sensor_extendable_to(
                    self.sensor_ptr.as_ptr(),
                    #[allow(clippy::useless_conversion)]
                    (**ext as i32).try_into().unwrap(),
                    &mut err,
                );

                if err.as_ref().is_none() {
                    is_extendable != 0
                } else {
                    sys::rs2_free_error(err);
                    false
                }
            })
            .unwrap();
        *ext
    }

    /// Get the value associated with the provided Rs2Option for the sensor.
    ///
    /// Returns An `f32` value corresponding to that option within the librealsense2 library, or None
    /// if the option is not supported.
    pub fn get_option(&self, option: Rs2Option) -> Option<f32> {
        if !self.supports_option(option) {
            return None;
        }

        unsafe {
            let mut err = std::ptr::null_mut::<sys::rs2_error>();
            let val = sys::rs2_get_option(
                self.sensor_ptr.as_ptr().cast::<sys::rs2_options>(),
                #[allow(clippy::useless_conversion)]
                (option as i32).try_into().unwrap(),
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

    /// Sets the `value` associated with the provided `option` for the sensor.
    ///
    /// Returns null tuple if the option can be successfully set on the sensor, otherwise an error.
    ///
    /// # Errors
    ///
    /// Returns [`OptionSetError::OptionNotSupported`] if the option is not supported on this
    /// sensor.
    ///
    /// Returns [`OptionSetError::OptionIsReadOnly`] if the option is supported but cannot be set
    /// on this sensor.
    ///
    /// Returns [`OptionSetError::CouldNotSetOption`] if the option is supported and not read-only,
    /// but could not be set for another reason (invalid value, internal exception, etc.).
    pub fn set_option(&mut self, option: Rs2Option, value: f32) -> Result<(), OptionSetError> {
        if !self.supports_option(option) {
            return Err(OptionSetError::OptionNotSupported);
        }

        if self.is_option_read_only(option) {
            return Err(OptionSetError::OptionIsReadOnly);
        }

        unsafe {
            let mut err = std::ptr::null_mut::<sys::rs2_error>();
            sys::rs2_set_option(
                self.sensor_ptr.as_ptr().cast::<sys::rs2_options>(),
                #[allow(clippy::useless_conversion)]
                (option as i32).try_into().unwrap(),
                value,
                &mut err,
            );
            check_rs2_error!(err, OptionSetError::CouldNotSetOption)?;

            Ok(())
        }
    }

    /// Gets the range for a given option.
    ///
    /// Returns some option range if the sensor supports the option, else `None`.
    pub fn get_option_range(&self, option: Rs2Option) -> Option<Rs2OptionRange> {
        if !self.supports_option(option) {
            return None;
        }

        unsafe {
            let mut err = std::ptr::null_mut::<sys::rs2_error>();

            let mut min = MaybeUninit::uninit();
            let mut max = MaybeUninit::uninit();
            let mut step = MaybeUninit::uninit();
            let mut default = MaybeUninit::uninit();

            sys::rs2_get_option_range(
                self.sensor_ptr.as_ptr().cast::<sys::rs2_options>(),
                #[allow(clippy::useless_conversion)]
                (option as i32).try_into().unwrap(),
                min.as_mut_ptr(),
                max.as_mut_ptr(),
                step.as_mut_ptr(),
                default.as_mut_ptr(),
                &mut err,
            );

            if err.as_ref().is_none() {
                Some(Rs2OptionRange {
                    min: min.assume_init(),
                    max: max.assume_init(),
                    step: step.assume_init(),
                    default: default.assume_init(),
                })
            } else {
                sys::rs2_free_error(err);
                None
            }
        }
    }

    /// Predicate for determining if this sensor supports a given option
    ///
    /// Returns true iff the option is supported by this sensor.
    pub fn supports_option(&self, option: Rs2Option) -> bool {
        unsafe {
            let mut err = std::ptr::null_mut::<sys::rs2_error>();
            let val = sys::rs2_supports_option(
                self.sensor_ptr.as_ptr().cast::<sys::rs2_options>(),
                #[allow(clippy::useless_conversion)]
                (option as i32).try_into().unwrap(),
                &mut err,
            );

            if err.as_ref().is_none() {
                val != 0
            } else {
                sys::rs2_free_error(err);
                false
            }
        }
    }

    /// Predicate for determining if the provided option is immutable or not.
    ///
    /// Returns true if the option is supported and can be mutated, otherwise false.
    pub fn is_option_read_only(&self, option: Rs2Option) -> bool {
        if !self.supports_option(option) {
            return false;
        }

        unsafe {
            let mut err = std::ptr::null_mut::<sys::rs2_error>();
            let val = sys::rs2_is_option_read_only(
                self.sensor_ptr.as_ptr().cast::<sys::rs2_options>(),
                #[allow(clippy::useless_conversion)]
                (option as i32).try_into().unwrap(),
                &mut err,
            );

            if err.as_ref().is_none() {
                val != 0
            } else {
                sys::rs2_free_error(err);
                false
            }
        }
    }

    /// Get a list of stream profiles associated with this sensor
    ///
    /// Returns a vector containing all the stream profiles associated with the sensor. The vector
    /// will have a length of zero if an error occurs while getting the stream profiles.
    pub fn stream_profiles(&self) -> Vec<StreamProfile> {
        let mut profiles = Vec::new();
        unsafe {
            let mut err = std::ptr::null_mut::<sys::rs2_error>();
            let profiles_ptr = sys::rs2_get_stream_profiles(self.sensor_ptr.as_ptr(), &mut err);
            if err.as_ref().is_some() {
                sys::rs2_free_error(err);
                return profiles;
            }

            let nonnull_profiles_ptr = NonNull::new(profiles_ptr).unwrap();
            let len = sys::rs2_get_stream_profiles_count(nonnull_profiles_ptr.as_ptr(), &mut err);

            if err.as_ref().is_some() {
                sys::rs2_free_error(err);
                sys::rs2_delete_stream_profiles_list(nonnull_profiles_ptr.as_ptr());
                return profiles;
            }

            for i in 0..len {
                match StreamProfile::try_create(&nonnull_profiles_ptr, i) {
                    Ok(s) => {
                        profiles.push(s);
                    }
                    Err(_) => {
                        continue;
                    }
                }
            }
            sys::rs2_delete_stream_profiles_list(nonnull_profiles_ptr.as_ptr());
        }
        profiles
    }

    // fn recommended_processing_blocks(&self) -> Vec<ProcessingBlock>{}

    /// Gets the value associated with the provided camera info key from the sensor.
    ///
    /// Returns some value corresponding to the camera info requested if this sensor supports that
    /// camera info, else `None`.
    pub fn info(&self, camera_info: Rs2CameraInfo) -> Option<&CStr> {
        if !self.supports_info(camera_info) {
            return None;
        }

        unsafe {
            let mut err = std::ptr::null_mut::<sys::rs2_error>();

            let val = sys::rs2_get_sensor_info(
                self.sensor_ptr.as_ptr(),
                #[allow(clippy::useless_conversion)]
                (camera_info as i32).try_into().unwrap(),
                &mut err,
            );

            if err.as_ref().is_none() {
                Some(CStr::from_ptr(val))
            } else {
                sys::rs2_free_error(err);
                None
            }
        }
    }

    /// Predicate method for determining if the sensor supports a certain kind of camera info.
    ///
    /// Returns true iff the sensor has a value associated with the `camera_info` key.
    pub fn supports_info(&self, camera_info: Rs2CameraInfo) -> bool {
        unsafe {
            let mut err = std::ptr::null_mut::<sys::rs2_error>();
            let supports_info = sys::rs2_supports_sensor_info(
                self.sensor_ptr.as_ptr(),
                #[allow(clippy::useless_conversion)]
                (camera_info as i32).try_into().unwrap(),
                &mut err,
            );

            if err.as_ref().is_none() {
                supports_info != 0
            } else {
                sys::rs2_free_error(err);
                false
            }
        }
    }

    /// Gets the auto exposure's region of interest for the sensor.
    ///
    /// Returns the region of interest for the auto exposure or None
    /// if this isn't available.
    pub fn get_region_of_interest(&self) -> Option<Rs2Roi> {
        unsafe {
            let mut err = std::ptr::null_mut::<sys::rs2_error>();
            let mut roi = Rs2Roi {
                min_x: 0,
                min_y: 0,
                max_x: 0,
                max_y: 0,
            };
            sys::rs2_get_region_of_interest(
                self.sensor_ptr.as_ptr(),
                &mut roi.min_x,
                &mut roi.min_y,
                &mut roi.max_x,
                &mut roi.max_y,
                &mut err,
            );
            if err.as_ref().is_none() {
                Some(roi)
            } else {
                sys::rs2_free_error(err);
                None
            }
        }
    }

    /// Sets the auto exposure's region of interest to `roi` for the sensor.
    ///
    /// Returns null tuple if the region of interest was set successfully, otherwise an error.
    ///
    /// # Errors
    ///
    /// Returns [`RoiSetError::CouldNotSetRoi`] if setting the region of interest failed.
    ///
    /// # Known issues
    ///
    /// This command can fail directly after the pipeline start. This is a bug in librealsense.
    /// Either wait for the first Frameset to be received or repeat the command in a loop
    /// with a delay until it succeeds as suggested by Intel.
    /// Issue at librealsense: <https://github.com/IntelRealSense/librealsense/issues/8004>
    pub fn set_region_of_interest(&mut self, roi: Rs2Roi) -> Result<(), RoiSetError> {
        unsafe {
            let mut err = std::ptr::null_mut::<sys::rs2_error>();
            sys::rs2_set_region_of_interest(
                self.sensor_ptr.as_ptr(),
                roi.min_x,
                roi.min_y,
                roi.max_x,
                roi.max_y,
                &mut err,
            );
            check_rs2_error!(err, RoiSetError::CouldNotSetRoi)
        }
    }
}
