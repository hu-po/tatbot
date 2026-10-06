//! Type for representing an Image frame taken from a RealSense camera.
//!
//! An "Image" Frame can be one of several things:
//!
//! - Depth Frame: A depth frame taken from a synthetic depth camera.
//! - Disparity Frame: A disparity frame taken from a synthetic depth camera.
//! - Color Frame: A frame holding color or monochrome data.
//!
//! Each frame type can hold data in multiple formats. The data type presented
//! depends on the settings and flags used at runtime on the RealSense device.

use super::pixel::{get_pixel, PixelKind};
use super::prelude::{
    CouldNotGetFrameSensorError, DepthError, DisparityError, FrameCategory, FrameConstructionError,
    FrameEx, BITS_PER_BYTE,
};
use crate::{
    check_rs2_error,
    kind::{Rs2Extension, Rs2FrameMetadata, Rs2Option, Rs2StreamKind, Rs2TimestampDomain},
    sensor::Sensor,
    stream_profile::StreamProfile,
};
use anyhow::Result;
use num_traits::FromPrimitive;
use realsense_sys as sys;
use std::{
    convert::{TryFrom, TryInto},
    marker::PhantomData,
    os::raw::c_int,
    ptr::{self, NonNull},
};

/// A unit struct defining a Depth frame.
#[derive(Debug)]
pub struct Depth;
/// A unit struct defining a Disparity frame.
#[derive(Debug)]
pub struct Disparity;
/// A unit struct defining a Color frame.
#[derive(Debug)]
pub struct Color;
/// A unit struct defining an Infrared frame.
#[derive(Debug)]
pub struct Infrared;
/// A unit struct defining a Fisheye frame.
#[derive(Debug)]
pub struct Fisheye;
/// A unit struct defining a Confidence frame.
#[derive(Debug)]
pub struct Confidence;

/// Holds the raw data pointer and derived data for an RS2 Image frame.
///
/// This generic type isn't particularly useful on it's own. In all cases, you want a specialized
/// version of this class ([`DepthFrame`], [`ColorFrame`], [`DisparityFrame`]).
#[derive(Debug)]
pub struct ImageFrame<Kind> {
    /// The raw data pointer from the original rs2 frame.
    frame_ptr: NonNull<sys::rs2_frame>,
    /// The width of the frame in pixels.
    width: usize,
    /// The height of the frame in pixels.
    height: usize,
    /// The pixel stride of the frame in bytes.
    stride: usize,
    /// The number of bits per pixel.
    bits_per_pixel: usize,
    /// The timestamp of the frame.
    timestamp: f64,
    /// The RealSense time domain from which the timestamp is derived.
    timestamp_domain: Rs2TimestampDomain,
    /// The frame number.
    frame_number: u64,
    /// The Stream Profile that created the frame.
    frame_stream_profile: StreamProfile,
    /// The size in bytes of the data contained in the frame.
    data_size_in_bytes: usize,
    /// The frame data contained in the frame.
    data: NonNull<std::os::raw::c_void>,
    /// A boolean used during `Drop` calls. This allows for proper handling of the pointer
    /// during ownership transfer.
    should_drop: bool,
    /// Holds the type metadata of this frame.
    _phantom: PhantomData<Kind>,
}

/// A type which acts as an iterator over an image frame of some pixel kind.
pub struct Iter<'a, K> {
    /// The image frame to iterate over.
    pub(crate) frame: &'a ImageFrame<K>,

    /// The current column.
    pub(crate) column: usize,

    /// The current row.
    pub(crate) row: usize,
}

impl<'a, K> Iterator for Iter<'a, K> {
    type Item = PixelKind<'a>;

    /// Provides a row-major iterator over an entire Image.
    fn next(&mut self) -> Option<Self::Item> {
        if self.column >= self.frame.width() || self.row >= self.frame.height() {
            return None;
        }

        let next = self.frame.get_unchecked(self.column, self.row);

        self.column += 1;

        if self.column >= self.frame.width() {
            self.column = 0;
            self.row += 1;
        }
        Some(next)
    }
}

/// An ImageFrame type holding the raw pointer and derived metadata for an RS2 Depth frame.
///
/// All fields in this struct are initialized during struct creation (via `try_from`).
/// Everything called from here during runtime should be valid as long as the
/// Frame is in scope... like normal Rust.
pub type DepthFrame = ImageFrame<Depth>;
/// An ImageFrame type holding the raw pointer and derived metadata for an RS2 Disparity frame.
///
/// All fields in this struct are initialized during struct creation (via `try_from`).
/// Everything called from here during runtime should be valid as long as the
/// Frame is in scope... like normal Rust.
pub type DisparityFrame = ImageFrame<Disparity>;
/// An ImageFrame type holding the raw pointer and derived metadata for an RS2 Color frame.
///
/// All fields in this struct are initialized during struct creation (via `try_from`).
/// Everything called from here during runtime should be valid as long as the
/// Frame is in scope... like normal Rust.
pub type ColorFrame = ImageFrame<Color>;
/// An ImageFrame type holding the raw pointer and derived metadata for an RS2 Infrared frame.
///
/// All fields in this struct are initialized during struct creation (via `try_from`).
/// Everything called from here during runtime should be valid as long as the
/// Frame is in scope... like normal Rust.
pub type InfraredFrame = ImageFrame<Infrared>;
/// An ImageFrame type holding the raw pointer and derived metadata for an RS2 Fisheye frame.
///
/// All fields in this struct are initialized during struct creation (via `try_from`).
/// Everything called from here during runtime should be valid as long as the
/// Frame is in scope... like normal Rust.
pub type FisheyeFrame = ImageFrame<Fisheye>;
/// An ImageFrame type holding the raw pointer and derived metadata for an RS2 Confidence frame.
///
/// All fields in this struct are initialized during struct creation (via `try_from`).
/// Everything called from here during runtime should be valid as long as the
/// Frame is in scope... like normal Rust.
pub type ConfidenceFrame = ImageFrame<Confidence>;

impl<K> Drop for ImageFrame<K> {
    fn drop(&mut self) {
        unsafe {
            if self.should_drop {
                sys::rs2_release_frame(self.frame_ptr.as_ptr());
            }
        }
    }
}

impl<'a, K> IntoIterator for &'a ImageFrame<K> {
    type Item = <Iter<'a, K> as Iterator>::Item;
    type IntoIter = Iter<'a, K>;

    fn into_iter(self) -> Self::IntoIter {
        self.iter()
    }
}

unsafe impl<K> Send for ImageFrame<K> {}

impl<K> TryFrom<NonNull<sys::rs2_frame>> for ImageFrame<K> {
    type Error = anyhow::Error;

    /// Attempt to construct an Image frame of extension K from the raw `rs2_frame`.
    ///
    /// All members of the `ImageFrame` struct are validated and populated during this call.
    ///
    /// # Errors
    ///
    /// There are a number of errors that may occur if the data in the `rs2_frame` is not valid,
    /// all of type [`FrameConstructionError`].
    ///
    /// - [`CouldNotGetWidth`](FrameConstructionError::CouldNotGetWidth)
    /// - [`CouldNotGetHeight`](FrameConstructionError::CouldNotGetHeight)
    /// - [`CouldNotGetBitsPerPixel`](FrameConstructionError::CouldNotGetBitsPerPixel)
    /// - [`CouldNotGetStride`](FrameConstructionError::CouldNotGetStride)
    /// - [`CouldNotGetTimestamp`](FrameConstructionError::CouldNotGetTimestamp)
    /// - [`CouldNotGetTimestampDomain`](FrameConstructionError::CouldNotGetTimestampDomain)
    /// - [`CouldNotGetFrameStreamProfile`](FrameConstructionError::CouldNotGetFrameStreamProfile)
    /// - [`CouldNotGetDataSize`](FrameConstructionError::CouldNotGetDataSize)
    /// - [`CouldNotGetData`](FrameConstructionError::CouldNotGetData)
    ///
    /// See [`FrameConstructionError`] documentation for more details.
    fn try_from(frame_ptr: NonNull<sys::rs2_frame>) -> Result<Self, Self::Error> {
        unsafe {
            let mut err = ptr::null_mut::<sys::rs2_error>();
            let width = sys::rs2_get_frame_width(frame_ptr.as_ptr(), &mut err);
            check_rs2_error!(err, FrameConstructionError::CouldNotGetWidth)?;

            let height = sys::rs2_get_frame_height(frame_ptr.as_ptr(), &mut err);
            check_rs2_error!(err, FrameConstructionError::CouldNotGetHeight)?;

            let bits_per_pixel = sys::rs2_get_frame_bits_per_pixel(frame_ptr.as_ptr(), &mut err);
            check_rs2_error!(err, FrameConstructionError::CouldNotGetBitsPerPixel)?;

            let stride = sys::rs2_get_frame_stride_in_bytes(frame_ptr.as_ptr(), &mut err);
            check_rs2_error!(err, FrameConstructionError::CouldNotGetStride)?;

            let timestamp = sys::rs2_get_frame_timestamp(frame_ptr.as_ptr(), &mut err);
            check_rs2_error!(err, FrameConstructionError::CouldNotGetTimestamp)?;

            let timestamp_domain =
                sys::rs2_get_frame_timestamp_domain(frame_ptr.as_ptr(), &mut err);
            check_rs2_error!(err, FrameConstructionError::CouldNotGetTimestampDomain)?;

            let frame_number = sys::rs2_get_frame_number(frame_ptr.as_ptr(), &mut err);
            check_rs2_error!(err, FrameConstructionError::CouldNotGetFrameNumber)?;

            let profile_ptr = sys::rs2_get_frame_stream_profile(frame_ptr.as_ptr(), &mut err);
            check_rs2_error!(err, FrameConstructionError::CouldNotGetFrameStreamProfile)?;

            let nonnull_profile_ptr =
                NonNull::new(profile_ptr as *mut sys::rs2_stream_profile).unwrap();
            let profile = StreamProfile::try_from(nonnull_profile_ptr)?;

            let size = sys::rs2_get_frame_data_size(frame_ptr.as_ptr(), &mut err);
            check_rs2_error!(err, FrameConstructionError::CouldNotGetDataSize)?;

            debug_assert_eq!(size, width * height * bits_per_pixel / BITS_PER_BYTE);

            let data_ptr = sys::rs2_get_frame_data(frame_ptr.as_ptr(), &mut err);
            check_rs2_error!(err, FrameConstructionError::CouldNotGetData)?;

            let nonnull_data_ptr = NonNull::new(data_ptr as *mut std::os::raw::c_void).unwrap();

            Ok(ImageFrame {
                frame_ptr,
                width: width as usize,
                height: height as usize,
                stride: stride as usize,
                bits_per_pixel: bits_per_pixel as usize,
                timestamp,
                timestamp_domain: Rs2TimestampDomain::from_i32(timestamp_domain as i32).unwrap(),
                frame_number,
                frame_stream_profile: profile,
                data_size_in_bytes: size as usize,
                data: nonnull_data_ptr,
                should_drop: true,
                _phantom: PhantomData::<K> {},
            })
        }
    }
}

impl FrameCategory for DepthFrame {
    fn extension() -> Rs2Extension {
        Rs2Extension::DepthFrame
    }

    fn kind() -> Rs2StreamKind {
        Rs2StreamKind::Depth
    }

    fn has_correct_kind(&self) -> bool {
        self.frame_stream_profile.kind() == Self::kind()
    }
}

impl FrameCategory for DisparityFrame {
    fn extension() -> Rs2Extension {
        Rs2Extension::DisparityFrame
    }

    fn kind() -> Rs2StreamKind {
        Rs2StreamKind::Any
    }

    fn has_correct_kind(&self) -> bool {
        self.frame_stream_profile.kind() == Self::kind()
    }
}

impl FrameCategory for ColorFrame {
    fn extension() -> Rs2Extension {
        Rs2Extension::VideoFrame
    }

    fn kind() -> Rs2StreamKind {
        Rs2StreamKind::Color
    }

    fn has_correct_kind(&self) -> bool {
        self.frame_stream_profile.kind() == Self::kind()
    }
}

impl FrameCategory for InfraredFrame {
    fn extension() -> Rs2Extension {
        Rs2Extension::VideoFrame
    }

    fn kind() -> Rs2StreamKind {
        Rs2StreamKind::Infrared
    }

    fn has_correct_kind(&self) -> bool {
        self.frame_stream_profile.kind() == Self::kind()
    }
}

impl FrameCategory for FisheyeFrame {
    fn extension() -> Rs2Extension {
        Rs2Extension::VideoFrame
    }

    fn kind() -> Rs2StreamKind {
        Rs2StreamKind::Fisheye
    }

    fn has_correct_kind(&self) -> bool {
        self.frame_stream_profile.kind() == Self::kind()
    }
}

impl FrameCategory for ConfidenceFrame {
    fn extension() -> Rs2Extension {
        Rs2Extension::VideoFrame
    }

    fn kind() -> Rs2StreamKind {
        Rs2StreamKind::Confidence
    }

    fn has_correct_kind(&self) -> bool {
        self.frame_stream_profile.kind() == Self::kind()
    }
}

impl<T> FrameEx for ImageFrame<T> {
    fn stream_profile(&self) -> &StreamProfile {
        &self.frame_stream_profile
    }

    fn sensor(&self) -> Result<Sensor> {
        unsafe {
            let mut err = std::ptr::null_mut::<sys::rs2_error>();
            let sensor_ptr = sys::rs2_get_frame_sensor(self.frame_ptr.as_ptr(), &mut err);
            check_rs2_error!(err, CouldNotGetFrameSensorError)?;

            Ok(Sensor::from(NonNull::new(sensor_ptr).unwrap()))
        }
    }

    fn timestamp(&self) -> f64 {
        self.timestamp
    }

    fn timestamp_domain(&self) -> Rs2TimestampDomain {
        self.timestamp_domain
    }

    fn frame_number(&self) -> u64 {
        self.frame_number
    }

    fn metadata(&self, metadata_kind: Rs2FrameMetadata) -> Option<std::os::raw::c_longlong> {
        if !self.supports_metadata(metadata_kind) {
            return None;
        }

        unsafe {
            let mut err = std::ptr::null_mut::<sys::rs2_error>();

            let val = sys::rs2_get_frame_metadata(
                self.frame_ptr.as_ptr(),
                #[allow(clippy::useless_conversion)]
                (metadata_kind as i32).try_into().unwrap(),
                &mut err,
            );

            if err.as_ref().is_none() {
                Some(val)
            } else {
                sys::rs2_free_error(err);
                None
            }
        }
    }

    fn supports_metadata(&self, metadata_kind: Rs2FrameMetadata) -> bool {
        unsafe {
            let mut err = std::ptr::null_mut::<sys::rs2_error>();

            let supports_metadata = sys::rs2_supports_frame_metadata(
                self.frame_ptr.as_ptr(),
                #[allow(clippy::useless_conversion)]
                (metadata_kind as i32).try_into().unwrap(),
                &mut err,
            );

            if err.as_ref().is_none() {
                supports_metadata != 0
            } else {
                sys::rs2_free_error(err);
                false
            }
        }
    }

    unsafe fn get_owned_raw(mut self) -> NonNull<sys::rs2_frame> {
        self.should_drop = false;

        self.frame_ptr
    }
}

impl DepthFrame {
    /// Given the 2D depth coordinate (x,y) provide the corresponding depth in metric units.
    ///
    /// # Warning
    ///
    /// It is fairly expensive to use this in practice as it will copy the underlying pixel into a
    /// f32 value that gives you the direct distance. In practice getting
    /// `DepthFrame::depth_units` and then applying that to the raw data with [`ImageFrame::get`]
    /// is a much more efficient way to handle this.
    pub fn distance(&self, col: usize, row: usize) -> Result<f32, DepthError> {
        unsafe {
            let mut err = ptr::null_mut::<sys::rs2_error>();
            let distance = sys::rs2_depth_frame_get_distance(
                self.frame_ptr.as_ptr(),
                col as c_int,
                row as c_int,
                &mut err,
            );
            check_rs2_error!(err, DepthError::CouldNotGetDistance)?;
            Ok(distance)
        }
    }

    /// Get the metric units currently used for reporting depth information.
    pub fn depth_units(&self) -> Result<f32> {
        let sensor = self.sensor()?;
        let depth_units = sensor.get_option(Rs2Option::DepthUnits).ok_or_else(|| {
            anyhow::anyhow!("Option is not supported on the sensor for this frame type.")
        })?;
        Ok(depth_units)
    }
}

impl DisparityFrame {
    /// Given the 2D depth coordinate (x,y) provide the corresponding depth in metric units.
    ///
    /// # Warning
    ///
    /// Like with depth frames, this method is fairly expensive to use in practice. The disparity
    /// can be converted to depth fairly easily, but this will effectively copy every pixel if you
    /// loop through the data with this method for every index.
    ///
    /// It is often much more efficient to directly stream the
    /// [`Rs2Format::Distance`](crate::kind::Rs2Format::Distance) format if you want the distance
    /// directly, and access the frame data with [`ImageFrame::get`].
    pub fn distance(&self, col: usize, row: usize) -> Result<f32, DepthError> {
        unsafe {
            let mut err = ptr::null_mut::<sys::rs2_error>();
            let distance = sys::rs2_depth_frame_get_distance(
                self.frame_ptr.as_ptr(),
                col as c_int,
                row as c_int,
                &mut err,
            );
            check_rs2_error!(err, DepthError::CouldNotGetDistance)?;
            Ok(distance)
        }
    }

    /// Get the metric units currently used for reporting depth information.
    pub fn depth_units(&self) -> Result<f32> {
        let sensor = self.sensor()?;
        let depth_units = sensor.get_option(Rs2Option::DepthUnits).ok_or_else(|| {
            anyhow::anyhow!("Option is not supported on the sensor for this frame type.")
        })?;
        Ok(depth_units)
    }

    /// Get the baseline used during construction of the Disparity frame
    pub fn baseline(&self) -> Result<f32, DisparityError> {
        unsafe {
            let mut err = ptr::null_mut::<sys::rs2_error>();
            let baseline =
                sys::rs2_depth_stereo_frame_get_baseline(self.frame_ptr.as_ptr(), &mut err);
            check_rs2_error!(err, DisparityError)?;
            Ok(baseline)
        }
    }
}

impl<K> ImageFrame<K> {
    /// Iterator through every [pixel](crate::frame::PixelKind) of an image frame.
    pub fn iter(&self) -> Iter<'_, K> {
        Iter {
            frame: self,
            column: 0,
            row: 0,
        }
    }

    /// Get a pixel value from the Video Frame.
    ///
    /// # Safety
    ///
    /// This makes a call directly to the underlying data pointer inherited from
    /// the `rs2_frame`.
    #[inline(always)]
    pub fn get_unchecked(&self, col: usize, row: usize) -> PixelKind<'_> {
        unsafe {
            get_pixel(
                self.frame_stream_profile.format(),
                self.data_size_in_bytes,
                self.data.as_ptr(),
                self.stride,
                col,
                row,
            )
        }
    }

    /// Get the stride of this Video frame's pixel in bytes.
    pub fn stride(&self) -> usize {
        self.stride
    }

    /// Get the bits per pixel.
    pub fn bits_per_pixel(&self) -> usize {
        self.bits_per_pixel
    }

    /// Get the size of the data in this Video frame in bytes.
    pub fn get_data_size(&self) -> usize {
        self.data_size_in_bytes
    }

    /// Get a reference to the raw data held by this Video frame.
    ///
    /// # Safety
    ///
    /// This is a raw pointer to the underlying data. This data has to be interpreted according to
    /// the format of the frame itself. In most scenarios you will probably want to just use the
    /// `get_unchecked` function associated with the [`FrameEx`] trait, but this can be useful if
    /// you need more immediate access to the underlying pixel data.
    pub unsafe fn get_data(&self) -> &std::os::raw::c_void {
        self.data.as_ref()
    }

    /// Get the width of this Video frame in pixels
    pub fn width(&self) -> usize {
        self.width
    }

    /// Get the height of this Video frame in pixels
    pub fn height(&self) -> usize {
        self.height
    }

    /// Given a row and column index, Get a pixel value from this frame.
    pub fn get(&self, col: usize, row: usize) -> Option<PixelKind<'_>> {
        if col >= self.width || row >= self.height {
            None
        } else {
            Some(self.get_unchecked(col, row))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn frame_has_correct_kind() {
        assert_eq!(ColorFrame::kind(), Rs2StreamKind::Color);
        assert_eq!(DepthFrame::kind(), Rs2StreamKind::Depth);
        assert_eq!(DisparityFrame::kind(), Rs2StreamKind::Any);
        assert_eq!(InfraredFrame::kind(), Rs2StreamKind::Infrared);
        assert_eq!(FisheyeFrame::kind(), Rs2StreamKind::Fisheye);
        assert_eq!(ConfidenceFrame::kind(), Rs2StreamKind::Confidence);
    }
}
