//! Defines the pipeline type.

use super::{inactive::InactivePipeline, profile::PipelineProfile};
use crate::{check_rs2_error, frame::CompositeFrame, kind::Rs2Exception};
use anyhow::Result;
use realsense_sys as sys;
use std::{ptr::NonNull, task::Poll, time::Duration};
use thiserror::Error;

/// Enumeration over possible errors that can occur when waiting for a frame.
#[derive(Error, Debug)]
pub enum FrameWaitError {
    /// librealsense2 had an internal error occur while waiting for frames.
    #[error("An internal error occurred while waiting for frames. Type: {0}; Reason: {1}")]
    DidErrorDuringFrameWait(Rs2Exception, String),
    /// librealsense2 had an internal error while polling for the next set of frames.
    #[error("An internal error occurred while polling for the next set of frames. Type: {0}; Reason: {1}")]
    DidErrorDuringFramePoll(Rs2Exception, String),
    /// The associated function timed out while waiting for frames.
    #[error("Timed out while waiting for frame.")]
    DidTimeoutBeforeFrameArrival,
}

/// Type representing an "active" pipeline which is configured and can acquire frames.
#[derive(Debug)]
pub struct ActivePipeline {
    /// A (non-null) pointer to the pipeline.
    pipeline_ptr: NonNull<sys::rs2_pipeline>,
    /// The pipeline's profile, which contains the device the pipeline is configured for alongside
    /// the stream profiles for streams in the pipeline.
    profile: PipelineProfile,
}

impl Drop for ActivePipeline {
    fn drop(&mut self) {
        unsafe {
            sys::rs2_delete_pipeline(self.pipeline_ptr.as_ptr());
        }
    }
}

unsafe impl Send for ActivePipeline {}

impl ActivePipeline {
    /// Constructs a new active pipeline from the constituent components
    ///
    /// This is only to be used / called from the [`InactivePipeline`] type.
    pub(crate) fn new(pipeline_ptr: NonNull<sys::rs2_pipeline>, profile: PipelineProfile) -> Self {
        Self {
            pipeline_ptr,
            profile,
        }
    }

    /// Gets the active profile of pipeline.
    pub fn profile(&self) -> &PipelineProfile {
        &self.profile
    }

    /// Stop the pipeline.
    ///
    /// This method consumes the pipeline instance and returns pipeline markered inactive.
    pub fn stop(self) -> InactivePipeline {
        unsafe {
            let mut err = std::ptr::null_mut::<sys::rs2_error>();

            // The only "error" that can occur here is if the pipeline pointer is null.
            //
            // We know it is not (state is managed so that this isn't a possibility, and we use
            // `NonNull` to try and guarantee that even beyond our state management), so there
            // dealing with the error (and thus returning a result type) is superfluous here.
            sys::rs2_pipeline_stop(self.pipeline_ptr.as_ptr(), &mut err);

            let inactive = InactivePipeline::new(self.pipeline_ptr);

            std::mem::forget(self);
            inactive
        }
    }

    /// Waits to get a new composite frame, blocking the calling thread.
    ///
    /// Returns a composite frame from the pipeline, blocking the calling thread until a frame is
    /// available. This method can return an error if an internal exception occurs or if the thread
    /// waits more than the duration provided by `timeout_ms` (in milliseconds).
    ///
    /// # Arguments
    ///
    /// * `timeout_ms` - The timeout in milliseconds. If the thread blocks for longer than this
    ///   duration, it will exit early with a [`FrameWaitError::DidTimeoutBeforeFrameArrival`]. If
    ///   `None` is passed in, the [default timeout](realsense_sys::RS2_DEFAULT_TIMEOUT) is applied.
    ///
    /// # Errors
    ///
    /// Returns [`FrameWaitError::DidErrorDuringFrameWait`] if an internal error occurs while
    /// waiting for next frame(s).
    ///
    /// Returns [`FrameWaitError::DidTimeoutBeforeFrameArrival`] if the thread waits more than
    /// `timeout_ms` (in milliseconds) without returning a frame.
    pub fn wait(&mut self, timeout_ms: Option<Duration>) -> Result<CompositeFrame, FrameWaitError> {
        let timeout_ms = match timeout_ms {
            Some(d) => d.as_millis() as u32,
            None => sys::RS2_DEFAULT_TIMEOUT,
        };

        unsafe {
            let mut err = std::ptr::null_mut::<sys::rs2_error>();
            let mut frame = std::ptr::null_mut::<sys::rs2_frame>();

            // NOTE: You may notice that there is a `sys::rs2_pipeline_wait_for_frames` and you
            // might wonder why we only use this variant. Primarily, they do the same thing, but
            // this API is a bit cleaner since it makes it easy to detect if a timeout occurred.
            // If you use `rs2_pipeline_wait_for_frames` instead of
            // `rs2_pipeline_try_wait_for_frames` then you need to parse the returned `rs2_error`
            // message to determine if a timeout occurred. Here, we can just check if
            // `did_get_frame` is false (0), and provided no other errors occurred, then that is
            // indicative of a timeout.
            let did_get_frame = sys::rs2_pipeline_try_wait_for_frames(
                self.pipeline_ptr.as_ptr(),
                &mut frame,
                timeout_ms,
                &mut err,
            );
            check_rs2_error!(err, FrameWaitError::DidErrorDuringFrameWait)?;

            if did_get_frame != 0 {
                Ok(CompositeFrame::from(NonNull::new(frame).unwrap()))
            } else {
                Err(FrameWaitError::DidTimeoutBeforeFrameArrival)
            }
        }
    }

    /// Poll if next frame is immediately available.
    ///
    /// Unlike [`ActivePipeline::wait`], the method does not block and returns None immediately if
    /// the next frame is not available. Returns [`Poll::Pending`] if no frame is yet available,
    /// and returns [`Poll::Ready`] if the next composite frame is found.
    ///
    /// # Errors
    ///
    /// Returns [`FrameWaitError::DidErrorDuringFramePoll`] if an internal error occurs while
    /// polling for the next frame.
    pub fn poll(&mut self) -> Result<Poll<CompositeFrame>, FrameWaitError> {
        unsafe {
            let mut err = std::ptr::null_mut::<sys::rs2_error>();
            let mut frame_ptr = std::ptr::null_mut::<sys::rs2_frame>();
            let did_get_frame = sys::rs2_pipeline_poll_for_frames(
                self.pipeline_ptr.as_ptr(),
                &mut frame_ptr,
                &mut err,
            );
            check_rs2_error!(err, FrameWaitError::DidErrorDuringFramePoll)?;

            if did_get_frame != 0 {
                Ok(Poll::Ready(CompositeFrame::from(
                    NonNull::new(frame_ptr).unwrap(),
                )))
            } else {
                Ok(Poll::Pending)
            }
        }
    }
}
