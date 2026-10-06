//! Type for describing different product lines that work with librealsense2.
//!
//! As of Feb 22, 2021, The RealSense source code defines the following constants as
//! possible options for the Product Line flag. Some flags are just aliases, while
//! others are general groupings. The full list of flags is provided below for
//! convenience.
//!
//! | Product Line Name            | Flag (in hex)                                                                |
//! | ---------------------------- | ---------------------------------------------------------------------------- |
//! | `RS2_PRODUCT_LINE_ANY`       | `0xff`                                                                       |
//! | `RS2_PRODUCT_LINE_ANY_INTEL` | `0xfe`                                                                       |
//! | `RS2_PRODUCT_LINE_NON_INTEL` | `0x01`                                                                       |
//! | `RS2_PRODUCT_LINE_D400`      | `0x02`                                                                       |
//! | `RS2_PRODUCT_LINE_SR300`     | `0x04`                                                                       |
//! | `RS2_PRODUCT_LINE_L500`      | `0x08`                                                                       |
//! | `RS2_PRODUCT_LINE_T200`      | `0x10`                                                                       |
//! | `RS2_PRODUCT_LINE_DEPTH`     | `(RS2_PRODUCT_LINE_L500 or RS2_PRODUCT_LINE_SR300 or RS2_PRODUCT_LINE_D400)` |
//! | `RS2_PRODUCT_LINE_TRACKING`  | `RS2_PRODUCT_LINE_T200`                                                      |
//!

use num_derive::{FromPrimitive, ToPrimitive};
use realsense_sys as sys;

/// Type describing possible options for RealSense-supported product lines.
#[repr(u32)]
#[derive(FromPrimitive, ToPrimitive, Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Rs2ProductLine {
    /// Any product compatible with librealsense2
    Any = sys::RS2_PRODUCT_LINE_ANY,
    /// Any Intel product compatible with librealsense2
    AnyIntel = sys::RS2_PRODUCT_LINE_ANY_INTEL,
    /// Any non-Intel product compatible with librealsense2
    NonIntel = sys::RS2_PRODUCT_LINE_NON_INTEL,
    /// Any D400 series camera
    D400 = sys::RS2_PRODUCT_LINE_D400,
    /// Any SR300 series camera
    Sr300 = sys::RS2_PRODUCT_LINE_SR300,
    /// Any L500 series LiDAR / camera
    L500 = sys::RS2_PRODUCT_LINE_L500,
    /// Any T200 series product
    ///
    /// This is aliased to
    /// [`RS2_PRODUCT_LINE_TRACKING`](realsense_sys::RS2_PRODUCT_LINE_TRACKING)
    T200 = sys::RS2_PRODUCT_LINE_T200,
    /// Any device that has a depth feed
    Depth = sys::RS2_PRODUCT_LINE_DEPTH,
}
