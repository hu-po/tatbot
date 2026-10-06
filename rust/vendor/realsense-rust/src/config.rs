//! A type representing the [`Pipeline`](crate::pipeline::InactivePipeline) configuration.

use crate::{
    base::from_path,
    check_rs2_error,
    kind::{Rs2Exception, Rs2Format, Rs2StreamKind},
};
use anyhow::Result;
use realsense_sys as sys;
use std::{convert::TryInto, ffi::CStr, path::Path, ptr::NonNull};
use thiserror::Error;

/// Type describing all possible errors that can occur when trying to configure a pipeline.
#[derive(Error, Debug)]
pub enum ConfigurationError {
    /// The requested stream could not be enabled.
    #[error("Could not enable stream. Type: {0}; Reason: {1}")]
    CouldNotEnableStream(Rs2Exception, String),
    /// All streams could not be enabled.
    #[error("Could not enable all streams. Type: {0}; Reason: {1}")]
    CouldNotEnableAllStreams(Rs2Exception, String),
    /// The requested stream could not be disabled.
    #[error("Could not disable stream. Type: {0}; Reason: {1}")]
    CouldNotDisableStream(Rs2Exception, String),
    /// All streams could not be enabled.
    #[error("Could not disable all streams. Type: {0}; Reason: {1}")]
    CouldNotDisableAllStreams(Rs2Exception, String),
    /// The specified device could not be enabled.
    #[error("Could not enable requested device. Type: {0}; Reason: {1}")]
    CouldNotEnableDevice(Rs2Exception, String),
    /// Recording to file could not be enabled for the specified device.
    #[error("Could not enable recording to file from device. Type: {0}; Reason: {1}")]
    CouldNotEnableRecordingToFile(Rs2Exception, String),
}

/// Type representing the [`Pipeline`](crate::pipeline::InactivePipeline) configuration.
#[derive(Debug)]
pub struct Config {
    /// A non-null pointer to the underlying librealsense2 configuration.
    config_ptr: NonNull<sys::rs2_config>,
}

impl Drop for Config {
    fn drop(&mut self) {
        unsafe {
            sys::rs2_delete_config(self.config_ptr.as_ptr());
        }
    }
}

unsafe impl Send for Config {}

impl Default for Config {
    fn default() -> Self {
        Self::new()
    }
}

impl Config {
    /// Construct a new configuration.
    pub fn new() -> Self {
        unsafe {
            let mut err = std::ptr::null_mut::<sys::rs2_error>();
            let ptr = sys::rs2_create_config(&mut err);

            // Literally the only way this can fail seems to be through C++ make_shared.
            //
            // This would imply that OOM errors are the only possible exceptions here, so I'm
            // inclined to believe that checking the error is a fool's errand: If this call errors
            // there isn't going to be a whole lot you can do.
            //
            // We're better off letting Rust panic due to NonNull::new(ptr).unwrap() than to add
            // any extra logic on top.

            Self {
                config_ptr: NonNull::new(ptr).unwrap(),
            }
        }
    }

    /// Enable the stream of kind `stream` with the provided attributes.
    ///
    /// Returns a mutable reference to self, or a configuration error if the underlying FFI call
    /// fails.
    ///
    /// # Arguments
    ///
    /// It is not an error if `stream` and `format` do not match appropriately, but you may find
    /// that the configuration will never resolve if they do not.. E.g. you cannot pass in
    /// [`Rs2StreamKind::Color`] alongside [`Rs2Format::Z16`]. If you're unsure, pass in
    /// [`Rs2Format::Any`] and librealsense2 will determine what the most appropriate format is for
    /// a given stream.
    ///
    /// The index is can be optionally provided. If it is not provided, then librealsense2 will
    /// pick the most suitable stream index it can find.
    ///
    /// If either `width` or `height` (but not both) are zero, librealsense2 will find the most
    /// appropriate value to match the non-zero one. E.g. if `width` is 640 and `height` is 0, then
    /// librealsense2 will return 640x480 images (the closest appropriate format).
    ///
    /// # Errors
    ///
    /// Returns [`ConfigurationError::CouldNotEnableStream`] if any internal exceptions occur while
    /// making this call. Note that this does not independently check the values passed into each
    /// of the provided arguments / attributes. If those are invalid, they will be checked when you
    /// call [`InactivePipeline::start`](crate::pipeline::InactivePipeline::start) or
    /// [`InactivePipeline::resolve`](crate::pipeline::InactivePipeline::resolve).
    ///
    pub fn enable_stream(
        &mut self,
        stream: Rs2StreamKind,
        index: Option<usize>,
        width: usize,
        height: usize,
        format: Rs2Format,
        framerate: usize,
    ) -> Result<&mut Self, ConfigurationError> {
        let index: i32 = if let Some(i) = index { i as i32 } else { -1 };
        unsafe {
            let mut err = std::ptr::null_mut::<sys::rs2_error>();
            sys::rs2_config_enable_stream(
                self.config_ptr.as_ptr(),
                #[allow(clippy::useless_conversion)]
                (stream as i32).try_into().unwrap(),
                index,
                width as i32,
                height as i32,
                #[allow(clippy::useless_conversion)]
                (format as i32).try_into().unwrap(),
                framerate as i32,
                &mut err,
            );
            check_rs2_error!(err, ConfigurationError::CouldNotEnableStream)?;
        };
        Ok(self)
    }

    /// Enable all device streams explicitly.
    ///
    /// This enables all streams with the default configuration. What this means is that
    /// librealsense2 will pick the format, resolution, and framerate. If you want to specify
    /// those values yourself, see [`Config::enable_stream`].
    ///
    /// Returns a mutable reference to self if it succeeds or a configuration error.
    ///
    /// # Errors
    ///
    /// Returns [`ConfigurationError::CouldNotEnableAllStreams`] if this call fails for whatever
    /// reason.
    ///
    pub fn enable_all_streams(&mut self) -> Result<&mut Self, ConfigurationError> {
        unsafe {
            let mut err = std::ptr::null_mut::<sys::rs2_error>();
            sys::rs2_config_enable_all_stream(self.config_ptr.as_ptr(), &mut err);
            check_rs2_error!(err, ConfigurationError::CouldNotEnableAllStreams)?;
        }
        Ok(self)
    }

    /// Enable device from a serial number.
    ///
    /// This is useful if you want to mandate that your configuration is only applied to a device
    /// with a specific serial number. The serial can be obtained from the
    /// [`Device`](crate::device::Device::info) method by passing in
    /// [`Rs2CameraInfo::SerialNumber`](crate::kind::Rs2CameraInfo::SerialNumber).
    ///
    /// Returns a mutable reference to self if it succeeds or a configuration error.
    ///
    /// # Errors
    ///
    /// Returns [`ConfigurationError::CouldNotEnableDevice`] if the device could not be enabled.
    ///
    pub fn enable_device_from_serial(
        &mut self,
        serial: &CStr,
    ) -> Result<&mut Self, ConfigurationError> {
        unsafe {
            let mut err = std::ptr::null_mut::<sys::rs2_error>();
            sys::rs2_config_enable_device(self.config_ptr.as_ptr(), serial.as_ptr(), &mut err);
            check_rs2_error!(err, ConfigurationError::CouldNotEnableDevice)?;
        }
        Ok(self)
    }

    /// Enable device from a file path.
    ///
    /// Enables a virtual "device" whose observations have been recorded to a file. If
    /// `loop_playback` is true, then the device will continue to stream observations once it has
    /// reached the end of the file by continuously looping back to the first observations in the
    /// serialized streams.
    ///
    /// Returns a mutable reference to self if it succeeds or a configuration error.
    ///
    /// # Errors
    ///
    /// Returns [`NulError`](std::ffi::NulError) if the provided file path cannot be cleanly
    /// represented as a [`CString`](std::ffi::CString). This usually only occurs if you have null
    /// characters in the path. Constructing a path using the utilties in Rust's [`std::fs`] are
    /// expected to work.
    ///
    /// Returns a [`ConfigurationError::CouldNotEnableDevice`] if the device cannot be enabled for
    /// any reason. e.g. if your `file` is valid and can be converted into a `CString`, but is
    /// actually a path to a directory, you'll likely see an error here.
    ///
    pub fn enable_device_from_file<P>(&mut self, file: P, loop_playback: bool) -> Result<&mut Self>
    where
        P: AsRef<Path>,
    {
        let path = from_path(file)?;
        unsafe {
            let mut err = std::ptr::null_mut::<sys::rs2_error>();
            sys::rs2_config_enable_device_from_file_repeat_option(
                self.config_ptr.as_ptr(),
                path.as_ptr(),
                loop_playback as i32,
                &mut err,
            );
            check_rs2_error!(err, ConfigurationError::CouldNotEnableDevice)?;
        }
        Ok(self)
    }

    /// Enable recording data streams to file.
    ///
    /// Configuration option for writing / serializing data from the pipeline to a file. This
    /// happens independently of any frame delivery once the pipeline has been started. It is an
    /// error if you attempt to record to file if you are streaming from a file (see
    /// [`Config::enable_device_from_file`]).
    ///
    /// Returns a mutable reference to self if it succeeds or an error.
    ///
    /// # Errors
    ///
    /// Returns [`NulError`](std::ffi::NulError) if the provided file path cannot be cleanly
    /// represented as a [`CString`](std::ffi::CString). This usually only occurs if you have null
    /// characters in the path. Constructing a path using the utilties in Rust's [`std::fs`] are
    /// expected to work.
    ///
    /// Returns [`ConfigurationError::CouldNotEnableRecordingToFile`] if the path is not a valid
    /// path to a file or if you are trying to record to a file while streaming from a file.
    ///
    pub fn enable_record_to_file<P>(&mut self, file: P) -> Result<&mut Self>
    where
        P: AsRef<Path>,
    {
        let path = from_path(file)?;
        unsafe {
            let mut err = std::ptr::null_mut::<sys::rs2_error>();
            sys::rs2_config_enable_record_to_file(
                self.config_ptr.as_ptr(),
                path.as_ptr(),
                &mut err,
            );
            check_rs2_error!(err, ConfigurationError::CouldNotEnableRecordingToFile)?;
        }
        Ok(self)
    }

    /// Disable a specific data stream by stream index.
    ///
    /// Returns a mutable reference to self or a configuration error.
    ///
    /// # Errors
    ///
    /// Returns [`ConfigurationError::CouldNotDisableStream`] if the stream cannot be disabled.
    ///
    pub fn disable_stream_at_index(
        &mut self,
        stream: Rs2StreamKind,
        index: usize,
    ) -> Result<&mut Self, ConfigurationError> {
        unsafe {
            let mut err = std::ptr::null_mut::<sys::rs2_error>();
            sys::rs2_config_disable_indexed_stream(
                self.config_ptr.as_ptr(),
                #[allow(clippy::useless_conversion)]
                (stream as i32).try_into().unwrap(),
                index as i32,
                &mut err,
            );
            check_rs2_error!(err, ConfigurationError::CouldNotDisableStream)?;
        }
        Ok(self)
    }

    /// Disable data stream by stream kind.
    ///
    /// This method removes the first stream of the given `stream` kind.
    ///
    /// Returns a mutable reference to self or a configuration error.
    ///
    /// # Errors
    ///
    /// Returns [`ConfigurationError::CouldNotDisableStream`] if the stream cannot be disabled.
    ///
    pub fn disable_stream(
        &mut self,
        stream: Rs2StreamKind,
    ) -> Result<&mut Self, ConfigurationError> {
        unsafe {
            let mut err = std::ptr::null_mut::<sys::rs2_error>();
            sys::rs2_config_disable_stream(
                self.config_ptr.as_ptr(),
                #[allow(clippy::useless_conversion)]
                (stream as i32).try_into().unwrap(),
                &mut err,
            );
            check_rs2_error!(err, ConfigurationError::CouldNotDisableStream)?;
        }
        Ok(self)
    }

    /// Disable all device streams explicitly.
    ///
    /// This method disables every stream.
    ///
    /// Returns a mutable reference to self or a configuration error.
    ///
    /// # Errors
    ///
    /// Returns [`ConfigurationError::CouldNotDisableAllStreams`] if the streams cannot be
    /// disabled.
    ///
    pub fn disable_all_streams(&mut self) -> Result<&mut Self, ConfigurationError> {
        unsafe {
            let mut err = std::ptr::null_mut::<sys::rs2_error>();
            sys::rs2_config_disable_all_streams(self.config_ptr.as_ptr(), &mut err);
            check_rs2_error!(err, ConfigurationError::CouldNotDisableAllStreams)?;
        }
        Ok(self)
    }

    /// Get the underlying low-level pointer to the configuration object.
    ///
    /// # Safety
    ///
    /// This method is not intended to be called or used outside of the crate itself. Be warned, it
    /// is _undefined behaviour_ to call [`realsense_sys::rs2_delete_config`] on this pointer. If
    /// you do, you risk a double-free error when the [`Config`] struct itself is dropped.
    ///
    pub(crate) unsafe fn get_raw(&self) -> NonNull<sys::rs2_config> {
        self.config_ptr
    }
}
