//! Enumeration of methods that can be used to fill invalid pixels.
//!
//! The filter implements several methods to rectify missing data in the resulting image.
//! The filter obtains the four immediate pixel "neighbors" (up, down ,left, right), and
//! selects one of them according to a user-defined rule.
//!
//! See the [RealSense post-processing documentation](https://dev.intelrealsense.com/docs/post-processing-filters)
//! for more information.

use num_derive::{FromPrimitive, ToPrimitive};

/// A type describing the method that will be used to fill invalid pixels.
#[repr(usize)]
#[derive(FromPrimitive, ToPrimitive, Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum HoleFillingMode {
    /// Use the value from the left neighbor pixel to fill the hole.
    FillFromLeft = 0,
    /// Use the value from the neighboring pixel which is furthest away from the sensor.
    FarestFromAround = 1,
    /// Use the value from the neighboring pixel closest to the sensor.
    NearestFromAround = 2,
}
