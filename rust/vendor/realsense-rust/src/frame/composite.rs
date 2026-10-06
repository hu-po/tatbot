//! Composite frame type containing all other potential frame types.
//!
//! Each Pipeline produces a synchronized collection of frames for all streams
//! configured for its allocated device. These frames will be presented to the user as
//! a collection via the Composite Frame type.
//!
//! This is typically what is delivered from the pipeline.

use super::prelude::FrameCategory;
use crate::kind::Rs2StreamKind;
use realsense_sys as sys;
use std::{
    convert::{TryFrom, TryInto},
    ptr::NonNull,
};

/// Holds the raw data pointer from an RS2 Composite frame type.
#[derive(Debug)]
pub struct CompositeFrame {
    /// The raw data pointer from the original rs2 frame
    frame: Option<NonNull<sys::rs2_frame>>,
}

unsafe impl Send for CompositeFrame {}

impl Drop for CompositeFrame {
    /// Drop the raw pointer stored with this struct whenever it goes out of scope.
    fn drop(&mut self) {
        if let Some(frame) = self.frame {
            unsafe {
                sys::rs2_release_frame(frame.as_ptr());
            }
        }
    }
}

impl From<NonNull<sys::rs2_frame>> for CompositeFrame {
    fn from(frame: NonNull<sys::rs2_frame>) -> Self {
        Self { frame: Some(frame) }
    }
}

impl CompositeFrame {
    /// Gets the number of individual frames included in the composite frame.
    pub fn count(&self) -> usize {
        unsafe {
            let mut err: *mut sys::rs2_error = std::ptr::null_mut::<sys::rs2_error>();
            let frame = self.frame.as_ref().unwrap();
            let count = sys::rs2_embedded_frames_count(frame.as_ptr(), &mut err);
            if err.as_ref().is_none() {
                count as usize
            } else {
                sys::rs2_free_error(err);
                0
            }
        }
    }

    /// Checks if the Composite frame collection is empty.
    pub fn is_empty(&self) -> bool {
        self.count() == 0
    }

    /// Retrieves all frames in the Composite frame collection of a given type.
    ///
    /// # Generic Arguments
    ///
    /// `F` must implement [`FrameCategory`]. Some examples of good types to use for this are:
    ///
    /// * [`ColorFrame`](crate::frame::ColorFrame)
    /// * [`DepthFrame`](crate::frame::DepthFrame)
    /// * [`DisparityFrame`](crate::frame::DisparityFrame)
    /// * [`PoseFrame`](crate::frame::PoseFrame)
    /// * [`PointsFrame`](crate::frame::PointsFrame)
    ///
    pub fn frames_of_type<F>(&self) -> Vec<F>
    where
        F: TryFrom<NonNull<sys::rs2_frame>> + FrameCategory,
    {
        let mut frames = Vec::new();
        for i in 0..self.count() {
            unsafe {
                let frame = self.frame.as_ref().unwrap();
                let mut err = std::ptr::null_mut::<sys::rs2_error>();
                let frame_ptr =
                    sys::rs2_extract_frame(frame.as_ptr(), i as std::os::raw::c_int, &mut err);

                if err.as_ref().is_some() {
                    sys::rs2_free_error(err);
                    continue;
                }

                let nonnull_frame_ptr = NonNull::new(frame_ptr).unwrap();

                let is_extendable_to = sys::rs2_is_frame_extendable_to(
                    nonnull_frame_ptr.as_ptr(),
                    #[allow(clippy::useless_conversion)]
                    (F::extension() as i32).try_into().unwrap(),
                    &mut err,
                );

                if err.as_ref().is_none() {
                    if is_extendable_to != 0 {
                        if let Ok(f) = F::try_from(nonnull_frame_ptr) {
                            let kind_for_frame = F::kind();

                            if kind_for_frame == Rs2StreamKind::Any || f.has_correct_kind() {
                                frames.push(f);
                            }
                            // This continue is to skip releasing the frame at the end of the loop.
                            // If the call to try_from above is successful and we can push, then
                            // the frame is owned by the type `E` and we should not release it.
                            continue;
                        }
                    }
                } else {
                    sys::rs2_free_error(err);
                }
                sys::rs2_release_frame(nonnull_frame_ptr.as_ptr());
            }
        }
        frames
    }

    /// Get (and own) the underlying frame pointer for this frame.
    ///
    /// This is primarily useful for passing this frame forward to a processing block or blocks
    /// (either via frame queue, directly, callback, etc).
    ///
    /// # Safety
    ///
    /// This does not destroy the underlying frame pointer once self
    /// goes out of scope. Instead, the program expects that whatever
    /// object was assigned to by this function now manages the lifetime.
    pub unsafe fn get_owned_raw(mut self) -> NonNull<sys::rs2_frame> {
        std::mem::take(&mut self.frame).unwrap()
    }
}
