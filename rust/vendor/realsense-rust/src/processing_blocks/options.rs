//! Processing block option definitions and management
//!
//! This module provides strongly-typed options for configuring processing blocks
//! at runtime with proper validation and error handling.

use crate::check_rs2_error;
use anyhow::Result;
use realsense_sys as sys;
use std::ptr::NonNull;
use thiserror::Error;

/// Errors that can occur when working with processing block options
#[derive(Error, Debug, Clone, PartialEq)]
pub enum ProcessingBlockOptionError {
    /// The requested option is not supported by this processing block
    #[error("Option not supported: {0}")]
    OptionNotSupported(String),
    /// The provided value is not valid for this option
    #[error("Invalid option value: {0}")]
    InvalidValue(String),
    /// Failed to set the option value
    #[error("Failed to set option: {0}")]
    SetOptionFailed(String),
    /// Failed to get the option value
    #[error("Failed to get option: {0}")]
    GetOptionFailed(String),
}

/// Trait for processing blocks that support configurable options
pub trait OptionsExt {
    /// Set an option value on the processing block
    fn set_option(
        &mut self,
        option: sys::rs2_option,
        value: f32,
    ) -> Result<(), ProcessingBlockOptionError>;

    /// Get the current value of an option
    fn get_option(&self, option: sys::rs2_option) -> Result<f32, ProcessingBlockOptionError>;

    /// Check if an option is supported
    fn supports_option(&self, option: sys::rs2_option) -> bool;

    /// Get the range of valid values for an option
    fn get_option_range(
        &self,
        option: sys::rs2_option,
    ) -> Result<(f32, f32, f32, f32), ProcessingBlockOptionError>;
}

/// Options for the Spatial Filter processing block
#[derive(Debug, Clone, PartialEq, Default)]
pub struct SpatialFilterOptions {
    /// Controls the weight with respect to the current pixel. Range: 0.25-1.0
    pub smooth_alpha: Option<f32>,
    /// Controls the decrease in preservation of edge textures. Range: 1-50
    pub smooth_delta: Option<f32>,
    /// Controls the filter strength. Range: 1-5
    pub magnitude: Option<f32>,
    /// Number of iterations for hole filling. Range: 0-5
    pub holes_fill: Option<f32>,
}

/// Options for the Temporal Filter processing block
#[derive(Debug, Clone, PartialEq, Default)]
pub struct TemporalFilterOptions {
    /// Controls the temporal alpha factor. Range: 0.0-1.0
    pub smooth_alpha: Option<f32>,
    /// Controls the temporal sensitivity. Range: 1-100
    pub smooth_delta: Option<f32>,
    /// Controls persistence of the filter. Range: 1-8
    pub persistence_control: Option<f32>,
}

/// Options for the Hole Filling Filter processing block
#[derive(Debug, Clone, PartialEq, Default)]
pub struct HoleFillingOptions {
    /// Hole filling mode. 0=Fill from left, 1=Farthest from around, 2=Nearest from around
    pub holes_fill: Option<f32>,
}

/// Options for the Threshold Filter processing block
#[derive(Debug, Clone, PartialEq, Default)]
pub struct ThresholdOptions {
    /// Minimum distance threshold in meters
    pub min_distance: Option<f32>,
    /// Maximum distance threshold in meters
    pub max_distance: Option<f32>,
}

/// Options for the Decimation Filter processing block
#[derive(Debug, Clone, PartialEq, Default)]
pub struct DecimationOptions {
    /// Decimation filter magnitude. Range: 2-8
    pub filter_magnitude: Option<f32>,
}

/// Options for the Colorizer processing block
#[derive(Debug, Clone, PartialEq, Default)]
pub struct ColorizerOptions {
    /// Color scheme for visualization. Range: 0-9
    pub color_scheme: Option<f32>,
    /// Histogram equalization. 0=disabled, 1=enabled
    pub histogram_equalization: Option<f32>,
    /// Minimum distance for color mapping
    pub min_distance: Option<f32>,
    /// Maximum distance for color mapping
    pub max_distance: Option<f32>,
}

/// Helper function to set an option on a processing block
pub fn set_processing_block_option(
    block: NonNull<sys::rs2_processing_block>,
    option: sys::rs2_option,
    value: f32,
) -> Result<(), ProcessingBlockOptionError> {
    unsafe {
        let mut err = std::ptr::null_mut::<sys::rs2_error>();

        // Check if option is supported
        let is_supported =
            sys::rs2_supports_option(block.as_ptr() as *const sys::rs2_options, option, &mut err);
        if !err.is_null() {
            check_rs2_error!(err, |kind, context| {
                ProcessingBlockOptionError::SetOptionFailed(format!("{:?}: {}", kind, context))
            })?;
        }

        if is_supported == 0 {
            return Err(ProcessingBlockOptionError::OptionNotSupported(format!(
                "Option {:?} not supported",
                option
            )));
        }

        sys::rs2_set_option(
            block.as_ptr() as *const sys::rs2_options,
            option,
            value,
            &mut err,
        );
        check_rs2_error!(err, |kind, context| {
            ProcessingBlockOptionError::SetOptionFailed(format!("{:?}: {}", kind, context))
        })?;

        Ok(())
    }
}

/// Helper function to get an option value from a processing block
pub fn get_processing_block_option(
    block: NonNull<sys::rs2_processing_block>,
    option: sys::rs2_option,
) -> Result<f32, ProcessingBlockOptionError> {
    unsafe {
        let mut err = std::ptr::null_mut::<sys::rs2_error>();

        // Check if option is supported
        let is_supported =
            sys::rs2_supports_option(block.as_ptr() as *const sys::rs2_options, option, &mut err);
        if !err.is_null() {
            check_rs2_error!(err, |kind, context| {
                ProcessingBlockOptionError::GetOptionFailed(format!("{:?}: {}", kind, context))
            })?;
        }

        if is_supported == 0 {
            return Err(ProcessingBlockOptionError::OptionNotSupported(format!(
                "Option {:?} not supported",
                option
            )));
        }

        let value =
            sys::rs2_get_option(block.as_ptr() as *const sys::rs2_options, option, &mut err);
        check_rs2_error!(err, |kind, context| {
            ProcessingBlockOptionError::GetOptionFailed(format!("{:?}: {}", kind, context))
        })?;

        Ok(value)
    }
}

/// Helper function to get option range from a processing block
pub fn get_processing_block_option_range(
    block: NonNull<sys::rs2_processing_block>,
    option: sys::rs2_option,
) -> Result<(f32, f32, f32, f32), ProcessingBlockOptionError> {
    unsafe {
        let mut err = std::ptr::null_mut::<sys::rs2_error>();
        let mut min = 0.0f32;
        let mut max = 0.0f32;
        let mut step = 0.0f32;
        let mut default = 0.0f32;

        sys::rs2_get_option_range(
            block.as_ptr() as *const sys::rs2_options,
            option,
            &mut min,
            &mut max,
            &mut step,
            &mut default,
            &mut err,
        );
        check_rs2_error!(err, |kind, context| {
            ProcessingBlockOptionError::GetOptionFailed(format!("{:?}: {}", kind, context))
        })?;

        Ok((min, max, step, default))
    }
}
