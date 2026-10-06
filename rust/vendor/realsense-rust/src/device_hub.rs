//! Type representing the concept of a "hub" that devices can connect to.

use crate::{check_rs2_error, device::Device, kind::Rs2Exception};
use anyhow::Result;
use realsense_sys as sys;
use std::{convert::From, ptr::NonNull};
use thiserror::Error;

/// Error describing when the device hub failed while attempting to wait for devices.
///
/// Can occur if there is an internal system exception while waiting for devices.
#[derive(Error, Debug)]
#[error("Could not wait for device due to internal error. Type: {0}; Reason: {1}")]
pub struct CouldNotWaitForDeviceError(pub Rs2Exception, pub String);

/// A type representing a hub for devices to connect to.
///
/// The device hub is a type used for waiting on a device connection or to check if a device is
/// still connected.
#[derive(Debug)]
pub struct DeviceHub {
    /// A non-null pointer to the underlying librealsense device hub.
    devicehub_ptr: NonNull<sys::rs2_device_hub>,
}

impl Drop for DeviceHub {
    fn drop(&mut self) {
        unsafe {
            sys::rs2_delete_device_hub(self.devicehub_ptr.as_ptr());
        }
    }
}

unsafe impl Send for DeviceHub {}

impl From<NonNull<sys::rs2_device_hub>> for DeviceHub {
    fn from(devicehub_ptr: NonNull<sys::rs2_device_hub>) -> Self {
        Self { devicehub_ptr }
    }
}

impl DeviceHub {
    /// Gets a connected device, or waits for any device to be connected.
    ///
    /// If any device is connected, this method will return that device. It will cycle through
    /// devices if multiple are connected. Otherwise, it blocks the calling thread until a device
    /// is connected.
    ///
    /// # Errors
    ///
    /// Returns [`CouldNotWaitForDeviceError`] if an internal exception occurs while trying to wait
    /// for device connections.
    ///
    /// Returns [`DeviceConstructionError`](crate::device::DeviceConstructionError) if a device is
    /// found but an exception occurs during type construction.
    ///
    pub fn wait_for_device(&self) -> Result<Device> {
        unsafe {
            let mut err = std::ptr::null_mut::<sys::rs2_error>();
            let device_ptr =
                sys::rs2_device_hub_wait_for_device(self.devicehub_ptr.as_ptr(), &mut err);
            check_rs2_error!(err, CouldNotWaitForDeviceError)?;

            Ok(Device::from(NonNull::new(device_ptr).unwrap()))
        }
    }

    /// Predicate to check whether a given device is connected.
    pub fn is_device_connected(&self, device: &Device) -> bool {
        unsafe {
            let mut err = std::ptr::null_mut::<sys::rs2_error>();
            let val = sys::rs2_device_hub_is_device_connected(
                self.devicehub_ptr.as_ptr(),
                device.get_raw().as_ptr(),
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
}
