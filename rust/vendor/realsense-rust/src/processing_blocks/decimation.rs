//! Processing block that decimates the depth stream
//!
//! Based on an example here:
//! <https://github.com/IntelRealSense/librealsense/blob/4673a37d981164af8eeb8e296e430fc1427e008d/doc/post-processing-filters.md?plain=1#L111>

use crate::{
    check_rs2_error,
    frame::{DepthFrame, FrameEx},
    processing_blocks::{
        errors::{ProcessFrameError, ProcessingBlockConstructionError},
        options::{
            get_processing_block_option, get_processing_block_option_range,
            set_processing_block_option, DecimationOptions, OptionsExt, ProcessingBlockOptionError,
        },
    },
};
use anyhow::Result;
use realsense_sys as sys;
use std::{convert::TryFrom, ptr::NonNull, task::Poll, time::Duration};

/// Applies decimation filter and plots modified params. Note that due to the modified frame size,
/// the decimated frame replaces the original one.
#[derive(Debug, Clone)]
pub struct Decimation {
    /// The processing block for the "Decimation" method
    processing_block: NonNull<sys::rs2_processing_block>,
    /// The frame queue upon which the processing block will deposit decimated frames. We check this
    /// for completed block operations.
    processing_queue: NonNull<sys::rs2_frame_queue>,
}

impl Drop for Decimation {
    fn drop(&mut self) {
        unsafe {
            sys::rs2_delete_frame_queue(self.processing_queue.as_ptr());
            sys::rs2_delete_processing_block(self.processing_block.as_ptr());
        }
    }
}

impl Decimation {
    /// Create a new Decimation object
    pub fn new(processing_queue_size: i32) -> Result<Self, ProcessingBlockConstructionError> {
        let (processing_block, processing_queue) = unsafe {
            let mut err = std::ptr::null_mut::<sys::rs2_error>();

            let ptr = sys::rs2_create_decimation_filter_block(&mut err);
            check_rs2_error!(
                err,
                ProcessingBlockConstructionError::CouldNotCreateProcessingBlock
            )?;

            let queue_ptr = sys::rs2_create_frame_queue(processing_queue_size, &mut err);
            check_rs2_error!(
                err,
                ProcessingBlockConstructionError::CouldNotCreateProcessingQueue
            )?;

            sys::rs2_start_processing_queue(ptr, queue_ptr, &mut err);
            check_rs2_error!(
                err,
                ProcessingBlockConstructionError::CouldNotStartProcessingQueue
            )?;
            (NonNull::new(ptr).unwrap(), NonNull::new(queue_ptr).unwrap())
        };

        Ok(Self {
            processing_block,
            processing_queue,
        })
    }

    /// Own and process the composite frame and return the decimated frames.
    pub fn queue(&mut self, frame: DepthFrame) -> Result<(), ProcessFrameError> {
        unsafe {
            let mut err = std::ptr::null_mut::<sys::rs2_error>();
            sys::rs2_process_frame(
                self.processing_block.as_ptr(),
                frame.get_owned_raw().as_ptr(),
                &mut err,
            );
            check_rs2_error!(err, |kind, context| { ProcessFrameError { kind, context } })?;
            Ok(())
        }
    }

    /// Wait to receive the results of the processing block
    pub fn wait(&mut self, timeout: Duration) -> Result<DepthFrame, ProcessFrameError> {
        unsafe {
            let mut err = std::ptr::null_mut::<sys::rs2_error>();
            let timeout_millis = u32::try_from(timeout.as_millis()).unwrap_or(u32::MAX);
            let decimated_frame =
                sys::rs2_wait_for_frame(self.processing_queue.as_ptr(), timeout_millis, &mut err);
            check_rs2_error!(err, |kind, context| { ProcessFrameError { kind, context } })?;
            Ok(DepthFrame::try_from(NonNull::new(decimated_frame).unwrap()).unwrap())
        }
    }

    /// Poll to receive the results of the processing block
    pub fn poll(&mut self) -> Result<Poll<DepthFrame>, ProcessFrameError> {
        unsafe {
            let mut err = std::ptr::null_mut::<sys::rs2_error>();
            let mut frame = std::ptr::null_mut::<sys::rs2_frame>();
            let is_ready =
                sys::rs2_poll_for_frame(self.processing_queue.as_ptr(), &mut frame, &mut err);

            // Check for errors
            check_rs2_error!(err, |kind, context| { ProcessFrameError { kind, context } })?;

            // Check for queue readiness
            if is_ready == 0 {
                Ok(Poll::Pending)
            } else {
                Ok(Poll::Ready(
                    DepthFrame::try_from(NonNull::new(frame).unwrap()).unwrap(),
                ))
            }
        }
    }

    /// Apply options to configure the decimation filter
    pub fn apply_options(
        &mut self,
        options: &DecimationOptions,
    ) -> Result<(), ProcessingBlockOptionError> {
        if let Some(filter_magnitude) = options.filter_magnitude {
            self.set_option(
                sys::rs2_option_RS2_OPTION_FILTER_MAGNITUDE,
                filter_magnitude,
            )?;
        }
        Ok(())
    }
}

impl OptionsExt for Decimation {
    fn set_option(
        &mut self,
        option: sys::rs2_option,
        value: f32,
    ) -> Result<(), ProcessingBlockOptionError> {
        set_processing_block_option(self.processing_block, option, value)
    }

    fn get_option(&self, option: sys::rs2_option) -> Result<f32, ProcessingBlockOptionError> {
        get_processing_block_option(self.processing_block, option)
    }

    fn supports_option(&self, option: sys::rs2_option) -> bool {
        unsafe {
            let mut err = std::ptr::null_mut::<sys::rs2_error>();
            let is_supported = sys::rs2_supports_option(
                self.processing_block.as_ptr() as *const sys::rs2_options,
                option,
                &mut err,
            );
            err.is_null() && is_supported != 0
        }
    }

    fn get_option_range(
        &self,
        option: sys::rs2_option,
    ) -> Result<(f32, f32, f32, f32), ProcessingBlockOptionError> {
        get_processing_block_option_range(self.processing_block, option)
    }
}
