//! Processing block for colorizing depth data
//!
//! The colorizer converts depth frames to RGB format for visualization

use crate::{
    check_rs2_error,
    frame::{ColorFrame, DepthFrame, FrameEx},
    processing_blocks::{
        errors::{ProcessFrameError, ProcessingBlockConstructionError},
        options::{
            get_processing_block_option, get_processing_block_option_range,
            set_processing_block_option, ColorizerOptions, OptionsExt, ProcessingBlockOptionError,
        },
    },
};
use anyhow::Result;
use realsense_sys as sys;
use std::{convert::TryFrom, ptr::NonNull, task::Poll, time::Duration};

/// Creates Depth-Colorizer processing block that can be used to quickly visualize the depth data
/// This block accepts depth frames as input and replaces them by depth frames with format RGB8
/// Non-depth frames are passed through
#[derive(Debug, Clone)]
pub struct Colorizer {
    /// The processing block for colorizing depth frames
    processing_block: NonNull<sys::rs2_processing_block>,
    /// The frame queue upon which the processing block will deposit colorized frames
    processing_queue: NonNull<sys::rs2_frame_queue>,
}

impl Drop for Colorizer {
    fn drop(&mut self) {
        unsafe {
            sys::rs2_delete_frame_queue(self.processing_queue.as_ptr());
            sys::rs2_delete_processing_block(self.processing_block.as_ptr());
        }
    }
}

impl Colorizer {
    /// Create a new Colorizer processing block
    pub fn new(processing_queue_size: i32) -> Result<Self, ProcessingBlockConstructionError> {
        let (processing_block, processing_queue) = unsafe {
            let mut err = std::ptr::null_mut::<sys::rs2_error>();

            let ptr = sys::rs2_create_colorizer(&mut err);
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

    /// Process a depth frame and colorize it
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

    /// Wait to receive the colorized frame results
    pub fn wait(&mut self, timeout: Duration) -> Result<ColorFrame, ProcessFrameError> {
        unsafe {
            let mut err = std::ptr::null_mut::<sys::rs2_error>();
            let timeout_millis = u32::try_from(timeout.as_millis()).unwrap_or(u32::MAX);

            let colorized_frame =
                sys::rs2_wait_for_frame(self.processing_queue.as_ptr(), timeout_millis, &mut err);
            check_rs2_error!(err, |kind, context| { ProcessFrameError { kind, context } })?;
            Ok(ColorFrame::try_from(NonNull::new(colorized_frame).unwrap()).unwrap())
        }
    }

    /// Poll to receive the colorized frame results
    pub fn poll(&mut self) -> Result<Poll<ColorFrame>, ProcessFrameError> {
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
                    ColorFrame::try_from(NonNull::new(frame).unwrap()).unwrap(),
                ))
            }
        }
    }

    /// Apply options to configure the colorizer
    pub fn apply_options(
        &mut self,
        options: &ColorizerOptions,
    ) -> Result<(), ProcessingBlockOptionError> {
        if let Some(color_scheme) = options.color_scheme {
            self.set_option(sys::rs2_option_RS2_OPTION_COLOR_SCHEME, color_scheme)?;
        }
        if let Some(histogram_equalization) = options.histogram_equalization {
            self.set_option(
                sys::rs2_option_RS2_OPTION_HISTOGRAM_EQUALIZATION_ENABLED,
                histogram_equalization,
            )?;
        }
        if let Some(min_distance) = options.min_distance {
            self.set_option(sys::rs2_option_RS2_OPTION_MIN_DISTANCE, min_distance)?;
        }
        if let Some(max_distance) = options.max_distance {
            self.set_option(sys::rs2_option_RS2_OPTION_MAX_DISTANCE, max_distance)?;
        }
        Ok(())
    }
}

impl OptionsExt for Colorizer {
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
