//! Processing block for depth<->disparity domain transformation
//!
//! Provides depth<->disparity domain transformation for stereo-based depth modules

use crate::{
    check_rs2_error,
    frame::{DepthFrame, FrameEx},
    processing_blocks::errors::{ProcessFrameError, ProcessingBlockConstructionError},
};
use anyhow::Result;
use realsense_sys as sys;
use std::{convert::TryFrom, ptr::NonNull, task::Poll, time::Duration};

/// Creates a post processing block that provides for depth<->disparity domain transformation
/// for stereo-based depth modules
#[derive(Debug, Clone)]
pub struct DisparityTransform {
    /// The processing block for disparity transformation
    processing_block: NonNull<sys::rs2_processing_block>,
    /// The frame queue upon which the processing block will deposit transformed frames
    processing_queue: NonNull<sys::rs2_frame_queue>,
}

impl Drop for DisparityTransform {
    fn drop(&mut self) {
        unsafe {
            sys::rs2_delete_frame_queue(self.processing_queue.as_ptr());
            sys::rs2_delete_processing_block(self.processing_block.as_ptr());
        }
    }
}

impl DisparityTransform {
    /// Create a new DisparityTransform processing block
    ///
    /// # Arguments
    ///
    /// * `transform_to_disparity` - true = depth->disparity, false = disparity->depth
    /// * `processing_queue_size` - Size of the processing queue
    pub fn new(
        transform_to_disparity: bool,
        processing_queue_size: i32,
    ) -> Result<Self, ProcessingBlockConstructionError> {
        let (processing_block, processing_queue) = unsafe {
            let mut err = std::ptr::null_mut::<sys::rs2_error>();

            let ptr = sys::rs2_create_disparity_transform_block(
                if transform_to_disparity { 1 } else { 0 },
                &mut err,
            );
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

    /// Process a depth frame with disparity transformation
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

    /// Wait to receive the transformed results
    pub fn wait(&mut self, timeout: Duration) -> Result<DepthFrame, ProcessFrameError> {
        unsafe {
            let mut err = std::ptr::null_mut::<sys::rs2_error>();
            let timeout_millis = u32::try_from(timeout.as_millis()).unwrap_or(u32::MAX);

            let transformed_frame =
                sys::rs2_wait_for_frame(self.processing_queue.as_ptr(), timeout_millis, &mut err);
            check_rs2_error!(err, |kind, context| { ProcessFrameError { kind, context } })?;
            Ok(DepthFrame::try_from(NonNull::new(transformed_frame).unwrap()).unwrap())
        }
    }

    /// Poll to receive the transformed results
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
}