//! Type that defines a RealSense context used by the rest of the API

use crate::{
    base::from_path,
    check_rs2_error,
    device::Device,
    device_hub::DeviceHub,
    kind::{Rs2Exception, Rs2ProductLine},
};
use anyhow::Result;
use num_traits::ToPrimitive;
use realsense_sys as sys;
use std::{collections::HashSet, convert::From, path::Path, ptr::NonNull};
use thiserror::Error;

/// Type describing a RealSense context, used by the rest of the API.
#[derive(Debug)]
pub struct Context {
    /// A non-null pointer to the underlying librealsense context.
    context_ptr: NonNull<sys::rs2_context>,
}

/// An error type describing failure to construct a context.
#[derive(Error, Debug)]
#[error("Could not construct the context. Type: {0}; Reason: {1}")]
pub struct ContextConstructionError(pub Rs2Exception, pub String);

/// An error type describing failure to get the device hub from a context.
#[derive(Error, Debug)]
#[error("Could not get the device hub from the context. Type: {0}; Reason: {1}")]
pub struct CouldNotGetDeviceHubError(pub Rs2Exception, pub String);

/// An error type describing failure to add a device from a file.
#[derive(Error, Debug)]
#[error("Could not add a device from file. Type: {0}; Reason: {1}")]
pub struct CouldNotAddDeviceError(pub Rs2Exception, pub String);

/// An error type describing failure to remove a device from a file.
#[derive(Error, Debug)]
#[error("Could not remove device from file. Type: {0}; Reason: {1}")]
pub struct CouldNotRemoveDeviceError(pub Rs2Exception, pub String);

impl Drop for Context {
    fn drop(&mut self) {
        unsafe { sys::rs2_delete_context(self.context_ptr.as_ptr()) }
    }
}

unsafe impl Send for Context {}

impl Context {
    /// Construct a new context.
    ///
    /// # Errors
    ///
    /// Returns [`ContextConstructionError`] if the context cannot be created.
    ///
    pub fn new() -> Result<Self, ContextConstructionError> {
        unsafe {
            let mut err = std::ptr::null_mut::<sys::rs2_error>();
            let ptr = sys::rs2_create_context(sys::RS2_API_VERSION as i32, &mut err);
            check_rs2_error!(err, ContextConstructionError)?;

            Ok(Self {
                context_ptr: NonNull::new(ptr).unwrap(),
            })
        }
    }

    /// Construct a context from a librealsense JSON settings document (for
    /// example `{"dds": {"enabled": true}}`), merged over the library
    /// configuration file as `rs2_create_context_ex` documents.
    ///
    /// # Errors
    ///
    /// Returns [`ContextConstructionError`] if the settings contain a NUL
    /// byte or the context cannot be created.
    ///
    pub fn with_settings(settings: &str) -> Result<Self, ContextConstructionError> {
        let settings = std::ffi::CString::new(settings).map_err(|_| {
            ContextConstructionError(
                Rs2Exception::InvalidValue,
                "context settings contain a NUL byte".to_owned(),
            )
        })?;
        unsafe {
            let mut err = std::ptr::null_mut::<sys::rs2_error>();
            let ptr = sys::rs2_create_context_ex(
                sys::RS2_API_VERSION as i32,
                settings.as_ptr(),
                &mut err,
            );
            check_rs2_error!(err, ContextConstructionError)?;

            Ok(Self {
                context_ptr: NonNull::new(ptr).unwrap(),
            })
        }
    }

    /// Creates a device hub from the context.
    ///
    /// # Errors
    ///
    /// Returns [`CouldNotGetDeviceHubError`] if the device hub cannot be created.
    ///
    pub fn create_device_hub(&self) -> Result<DeviceHub, CouldNotGetDeviceHubError> {
        unsafe {
            let mut err = std::ptr::null_mut::<sys::rs2_error>();
            let devicehub_ptr = sys::rs2_create_device_hub(self.context_ptr.as_ptr(), &mut err);
            check_rs2_error!(err, CouldNotGetDeviceHubError)?;

            Ok(DeviceHub::from(NonNull::new(devicehub_ptr).unwrap()))
        }
    }

    /// Get a list of devices that are already connected to the host.
    pub fn query_devices(&self, product_mask: HashSet<Rs2ProductLine>) -> Vec<Device> {
        // TODO/TEST: Make sure that an empty mask (therefore giving no filter) gives
        // us _all_ devices, not _no_ devices.

        let mask = if product_mask.is_empty() {
            Rs2ProductLine::Any.to_i32().unwrap()
        } else {
            product_mask.iter().fold(0, |k, v| k | v.to_u32().unwrap()) as i32
        };

        let mut devices = Vec::new();
        unsafe {
            let mut err = std::ptr::null_mut::<sys::rs2_error>();
            let device_list_ptr =
                sys::rs2_query_devices_ex(self.context_ptr.as_ptr(), mask, &mut err);

            if err.as_ref().is_some() {
                sys::rs2_free_error(err);
                return devices;
            }

            let device_list = NonNull::new(device_list_ptr).unwrap();

            let len = sys::rs2_get_device_count(device_list.as_ptr(), &mut err);

            if err.as_ref().is_some() {
                sys::rs2_free_error(err);
                sys::rs2_delete_device_list(device_list.as_ptr());
                return devices;
            }

            for i in 0..len {
                match Device::try_create(&device_list, i) {
                    Ok(d) => {
                        devices.push(d);
                    }
                    Err(_) => {
                        continue;
                    }
                }
            }

            sys::rs2_delete_device_list(device_list.as_ptr());
        }
        devices
    }

    /// Create a new device and add it to the context.
    ///
    /// This adds a "device" at a particular file on the system to the RealSense context. Returns a
    /// handle to the device, or an error if this call fails.
    ///
    /// # Errors
    ///
    /// Returns [`NulError`](std::ffi::NulError) if the provided file path cannot be cleanly
    /// represented as a [`CString`](std::ffi::CString). This usually only occurs if you have null
    /// characters in the path. Constructing a path using the utilties in Rust's [`std::fs`] are
    /// expected to work.
    ///
    /// Returns [`CouldNotAddDeviceError`] if the device cannot be added.
    ///
    pub fn add_device<P>(&mut self, file: P) -> Result<Device>
    where
        P: AsRef<Path>,
    {
        let path = from_path(file)?;
        unsafe {
            let mut err = std::ptr::null_mut::<sys::rs2_error>();
            let device_ptr =
                sys::rs2_context_add_device(self.context_ptr.as_ptr(), path.as_ptr(), &mut err);
            check_rs2_error!(err, CouldNotAddDeviceError)?;

            Ok(Device::from(NonNull::new(device_ptr).unwrap()))
        }
    }

    /// Removes a playback device from the context, if it exists
    ///
    /// This removes a "device" at a particular file on the system from the RealSense context.
    /// Returns nothing (null tuple) or an Error if the device cannot be removed.
    ///
    /// # Errors
    ///
    /// Returns [`CouldNotRemoveDeviceError`] if the device cannot be removed for any reason.
    ///
    pub fn remove_device<P>(&mut self, file: P) -> Result<()>
    where
        P: AsRef<Path>,
    {
        let path = from_path(file)?;
        unsafe {
            let mut err = std::ptr::null_mut::<sys::rs2_error>();
            sys::rs2_context_remove_device(self.context_ptr.as_ptr(), path.as_ptr(), &mut err);
            check_rs2_error!(err, CouldNotRemoveDeviceError)?;

            Ok(())
        }
    }

    /// Get the underlying low-level pointer to the context object.
    ///
    /// # Safety
    ///
    /// This method is not intended to be called or used outside of the crate itself. Be warned, it
    /// is _undefined behaviour_ to call [`realsense_sys::rs2_delete_context`] on this pointer. If
    /// you do, you risk a double-free error when the [`Context`] struct itself is dropped.
    ///
    pub(crate) unsafe fn get_raw(&self) -> NonNull<sys::rs2_context> {
        self.context_ptr
    }
}
