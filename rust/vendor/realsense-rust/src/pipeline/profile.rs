//! Defines a type which holds the device & streams associated with an `ActivePipeline`.

use crate::{check_rs2_error, device::Device, kind::Rs2Exception, stream_profile::StreamProfile};
use anyhow::Result;
use realsense_sys as sys;
use std::{convert::TryFrom, ptr::NonNull};
use thiserror::Error;

/// Type representing the device and streams associated with a pipeline.
#[derive(Debug)]
pub struct PipelineProfile {
    /// The device that this profile is configured for.
    device: Device,
    /// A vector of stream profiles for streams that are emitting observations from the pipeline
    /// that this profile corresponds to.
    streams: Vec<StreamProfile>,
}

/// Type representing possible errors that can occur during pipeline profile construction.
#[derive(Error, Debug)]
pub enum PipelineProfileConstructionError {
    /// Could not retrieve the device from the underlying pipeline profile pointer.
    #[error("Could not retrieve device from pipeline profile. Type: {0}; Reason: {1}")]
    CouldNotRetrieveDevice(Rs2Exception, String),
    /// Could not retrieve the list of stream profiles from the underlying pipeline profile pointer.
    #[error(
        "Could not retrieve stream profile list from pipeline profile. Type: {0}; Reason: {1}"
    )]
    CouldNotRetrieveStreamList(Rs2Exception, String),
    /// Could not retrieve the count of stream profiles from the underlying pipeline profile pointer.
    #[error(
        "Could not retrieve stream profile count from pipeline profile. Type: {0}; Reason: {1}"
    )]
    CouldNotRetrieveStreamCount(Rs2Exception, String),
}

impl TryFrom<NonNull<sys::rs2_pipeline_profile>> for PipelineProfile {
    type Error = anyhow::Error;

    /// Attempts to construct a `PipelineProfile` from an `rs2_pipeline_profile` pointer.
    ///
    /// # Errors
    ///
    /// Returns [`PipelineProfileConstructionError::CouldNotRetrieveDevice`] if the device
    /// associated with the pipeline profile cannot be obtained.
    ///
    /// Returns [`PipelineProfileConstructionError::CouldNotRetrieveStreamList`] if the list of
    /// streams associated with the pipeline profile cannot be obtained.
    ///
    /// Returns [`PipelineProfileConstructionError::CouldNotRetrieveStreamCount`] if the length of
    /// the list of streams associated with the pipeline profile cannot be obtained.
    fn try_from(
        pipeline_profile_ptr: NonNull<sys::rs2_pipeline_profile>,
    ) -> Result<Self, Self::Error> {
        unsafe {
            let mut err = std::ptr::null_mut::<sys::rs2_error>();
            let device_ptr =
                sys::rs2_pipeline_profile_get_device(pipeline_profile_ptr.as_ptr(), &mut err);
            check_rs2_error!(
                err,
                PipelineProfileConstructionError::CouldNotRetrieveDevice
            )?;

            // Create the device object
            let device = Device::from(NonNull::new(device_ptr).unwrap());

            let stream_list_ptr =
                sys::rs2_pipeline_profile_get_streams(pipeline_profile_ptr.as_ptr(), &mut err);
            check_rs2_error!(
                err,
                PipelineProfileConstructionError::CouldNotRetrieveStreamList
            )?;

            let nonnull_stream_list = NonNull::new(stream_list_ptr).unwrap();
            let len = sys::rs2_get_stream_profiles_count(nonnull_stream_list.as_ptr(), &mut err);
            check_rs2_error!(
                err,
                PipelineProfileConstructionError::CouldNotRetrieveStreamCount
            )?;

            let mut streams = Vec::new();
            for i in 0..len {
                streams.push(StreamProfile::try_create(&nonnull_stream_list, i)?);
            }

            sys::rs2_delete_stream_profiles_list(nonnull_stream_list.as_ptr());
            sys::rs2_delete_pipeline_profile(pipeline_profile_ptr.as_ptr());
            Ok(Self { device, streams })
        }
    }
}

impl PipelineProfile {
    /// Gets the device associated with a pipeline.
    pub fn device(&self) -> &Device {
        &self.device
    }

    /// Gets list of streams associated with a pipeline.
    pub fn streams(&self) -> &Vec<StreamProfile> {
        &self.streams
    }
}
