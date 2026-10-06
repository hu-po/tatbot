//! Type for representing an "inactive" pipeline which is unconfigured and cannot acquire frames.

use super::{active::ActivePipeline, profile::PipelineProfile};
use crate::{check_rs2_error, config::Config, context::Context, kind::Rs2Exception};
use anyhow::Result;
use realsense_sys as sys;
use std::{convert::TryFrom, ptr::NonNull};
use thiserror::Error;

/// Enumeration of possible errors that can occur during pipeline construction.
#[derive(Error, Debug)]
pub enum PipelineConstructionError {
    /// The pipeline could not be created from the context.
    #[error("Could not create the pipeline from the provided context. Type: {0}; Reason {1}")]
    CouldNotCreatePipelineFromContext(Rs2Exception, String),
}

/// Enumeration of possible errors that can occur when trying to start the pipeline.
#[derive(Error, Debug)]
pub enum PipelineActivationError {
    /// The pipeline could not be started due to an internal exception.
    #[error("Could not successfully start the pipeline. Type: {0}; Reason: {1}")]
    CouldNotStartPipelineError(Rs2Exception, String),
    /// The configuration cannot be resolved.
    ///
    /// See [`InactivePipeline::can_resolve`] for more information.
    #[error("Config cannot be resolved by any active devices / stream combinations.")]
    ConfigCannotBeResolved,
}

/// A type describing an "inactive" pipeline which is unconfigured and cannot acquire frames.
#[derive(Debug)]
pub struct InactivePipeline {
    /// A (non-null) pointer to the pipeline.
    pipeline_ptr: NonNull<sys::rs2_pipeline>,
}

impl Drop for InactivePipeline {
    fn drop(&mut self) {
        unsafe {
            sys::rs2_delete_pipeline(self.pipeline_ptr.as_ptr());
        }
    }
}

unsafe impl Send for InactivePipeline {}

impl TryFrom<&Context> for InactivePipeline {
    type Error = anyhow::Error;

    fn try_from(context: &Context) -> Result<Self, Self::Error> {
        unsafe {
            let mut err = std::ptr::null_mut::<sys::rs2_error>();
            let context_ptr = context.get_raw();

            let pipeline_ptr = sys::rs2_create_pipeline(context_ptr.as_ptr(), &mut err);
            check_rs2_error!(
                err,
                PipelineConstructionError::CouldNotCreatePipelineFromContext
            )?;

            Ok(Self::new(NonNull::new(pipeline_ptr).unwrap()))
        }
    }
}

impl InactivePipeline {
    /// Constructs a new inactive pipeline from the constituent components
    ///
    /// This is only to be used / called from the [`ActivePipeline`] type.
    pub(crate) fn new(pipeline_ptr: NonNull<sys::rs2_pipeline>) -> Self {
        Self { pipeline_ptr }
    }

    /// Start the pipeline with an optional config.
    ///
    /// The method consumes inactive pipeline itself, and returns the started pipeine.
    pub fn start(self, config: Option<Config>) -> Result<ActivePipeline> {
        unsafe {
            let mut err = std::ptr::null_mut::<sys::rs2_error>();
            let profile_ptr = if let Some(conf) = config {
                if !self.can_resolve(&conf) {
                    return Err(anyhow::anyhow!(
                        PipelineActivationError::ConfigCannotBeResolved
                    ));
                }

                sys::rs2_pipeline_start_with_config(
                    self.pipeline_ptr.as_ptr(),
                    conf.get_raw().as_ptr(),
                    &mut err,
                )
            } else {
                sys::rs2_pipeline_start(self.pipeline_ptr.as_ptr(), &mut err)
            };
            check_rs2_error!(err, PipelineActivationError::CouldNotStartPipelineError)?;

            let profile = PipelineProfile::try_from(NonNull::new(profile_ptr).unwrap())?;
            let active = ActivePipeline::new(self.pipeline_ptr, profile);

            std::mem::forget(self);
            Ok(active)
        }
    }

    /// Resolve a configuration and get the corresponding pipeline profile.
    ///
    /// This function checks the pipeline to see if this config can be used to start the pipeline,
    /// and if this configuration can be used it returns the pipeline profile (device and streams)
    /// that will be used as the active profile when the pipeline is started. Otherwise, if this
    /// configuration cannot resolve, this will return `None`.
    pub fn resolve(&self, config: &Config) -> Option<PipelineProfile> {
        if !self.can_resolve(config) {
            return None;
        }

        unsafe {
            let mut err = std::ptr::null_mut::<sys::rs2_error>();
            let profile_ptr = sys::rs2_config_resolve(
                config.get_raw().as_ptr(),
                self.pipeline_ptr.as_ptr(),
                &mut err,
            );

            if err.as_ref().is_none() {
                PipelineProfile::try_from(NonNull::new(profile_ptr).unwrap()).ok()
            } else {
                sys::rs2_free_error(err);
                None
            }
        }
    }

    /// Predicate to check if a pipeline profile exists for a given configuration.
    ///
    /// This predicate evaluates whether or not a configuration can be resolved to a device and set
    /// of streams (which constitute a pipeline profile) that can be used by the pipeline to start
    /// streaming.
    ///
    /// Returns true iff the configuration can be satisfied and a pipeline profile can be
    /// constructed.
    pub fn can_resolve(&self, config: &Config) -> bool {
        unsafe {
            let mut err = std::ptr::null_mut::<sys::rs2_error>();
            let can_resolve = sys::rs2_config_can_resolve(
                config.get_raw().as_ptr(),
                self.pipeline_ptr.as_ptr(),
                &mut err,
            );

            if err.as_ref().is_none() {
                can_resolve != 0
            } else {
                sys::rs2_free_error(err);
                false
            }
        }
    }
}
