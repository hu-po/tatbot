//! Enumeration of frame data format & layout

use num_derive::{FromPrimitive, ToPrimitive};
use realsense_sys as sys;

/// A type representing all possible data formats for raw frame data
#[repr(i32)]
#[derive(FromPrimitive, ToPrimitive, Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Rs2Format {
    /// Format key used to tell librealsense2 to pick the best suited format.
    ///
    /// Unlike the other format entries in this enum, `Any` is used primarily when setting up
    /// streams in the config / pipeline. If you pass this, librealsense2 will pick the best suited
    /// (default) format for a given sensor.
    Any = sys::rs2_format_RS2_FORMAT_ANY as i32,
    /// 32-bit y0, u, y1, v data for every two pixels.
    ///
    /// Similar to YUV422 but packed in a different order. See [the wikipedia
    /// page](https://en.wikipedia.org/wiki/YUV) for more info.
    Yuyv = sys::rs2_format_RS2_FORMAT_YUYV as i32,
    /// 32-bit u, y0, v, y1 data for every two pixels.
    ///
    /// Similar to the standard YUYV pixel format, but packed in a different order
    Uyvy = sys::rs2_format_RS2_FORMAT_UYVY as i32,
    /// Raw data from the motion sensor
    MotionRaw = sys::rs2_format_RS2_FORMAT_MOTION_RAW as i32,
    /// Raw data from the external sensors hooked to one of the GPIO pins
    GpioRaw = sys::rs2_format_RS2_FORMAT_GPIO_RAW as i32,
    /// 32-bit floating point depth distance value
    Distance = sys::rs2_format_RS2_FORMAT_DISTANCE as i32,
    /// Bitstream encoding for video in which an image of each frame is encoded as JPEG-DIB
    Mjpeg = sys::rs2_format_RS2_FORMAT_MJPEG as i32,
    /// Multi-planar 16-bit depth + 10-bit IR
    Inzi = sys::rs2_format_RS2_FORMAT_INZI as i32,
    /// 8-bit IR stream
    Invi = sys::rs2_format_RS2_FORMAT_INVI as i32,
    /// Pose data packed as array of 32-bit floats.
    ///
    /// Contains translation vector, rotation quaternion, prediction velocities, and accelerations
    /// vectors.
    ///
    _6Dof = sys::rs2_format_RS2_FORMAT_6DOF as i32,
    /// 8-bit blue, green, and red channels (in that order)
    Bgr8 = sys::rs2_format_RS2_FORMAT_BGR8 as i32,
    /// 8-bit blue, green, red, and alpha channels (in that order)
    ///
    /// Alpha channel is always equal to 0xFF
    Bgra8 = sys::rs2_format_RS2_FORMAT_BGRA8 as i32,
    /// 16-bit floating-point disparity values.
    ///
    /// Depth -> disparity conversion is done with the formula:
    ///
    ///   disparity = baseline * focal_length / depth
    ///
    Disparity16 = sys::rs2_format_RS2_FORMAT_DISPARITY16 as i32,
    /// 32-bit floating-point disparity values.
    ///
    /// Depth -> disparity conversion is done with the formula:
    ///
    ///   disparity = baseline * focal_length / depth
    ///
    Disparity32 = sys::rs2_format_RS2_FORMAT_DISPARITY32 as i32,
    /// Motion data packed as 3 32-bit fload values in [x, y, z] order
    MotionXyz32F = sys::rs2_format_RS2_FORMAT_MOTION_XYZ32F as i32,
    /// 8-bit raw image
    Raw8 = sys::rs2_format_RS2_FORMAT_RAW8 as i32,
    /// Four 10-bit per pixel luminance values packed into a 5-byte macropixel
    Raw10 = sys::rs2_format_RS2_FORMAT_RAW10 as i32,
    /// 16-bit raw image
    Raw16 = sys::rs2_format_RS2_FORMAT_RAW16 as i32,
    /// 8-bit red, green and blue channels (in that order)
    Rgb8 = sys::rs2_format_RS2_FORMAT_RGB8 as i32,
    /// 8-bit red, green, blue, and alpha channels (in that order)
    ///
    /// alpha channel is always equal to 0xFF
    Rgba8 = sys::rs2_format_RS2_FORMAT_RGBA8 as i32,
    /// Grey-scale image as a bit-packed array.
    ///
    /// 4 pixel data stream taking 5 bytes.
    W10 = sys::rs2_format_RS2_FORMAT_W10 as i32,
    /// 32-bit floating point 3D coordinates in [x, y, z] order
    Xyz32F = sys::rs2_format_RS2_FORMAT_XYZ32F as i32,
    /// 8-bit per pixel grayscale image
    Y8 = sys::rs2_format_RS2_FORMAT_Y8 as i32,
    /// 8-bit per pixel interleaved.
    ///
    /// 8-bit left, 8-bit right.
    Y8I = sys::rs2_format_RS2_FORMAT_Y8I as i32,
    /// 16-bit per pixel grayscale image unpacked from 10-bit per pixel packed data.
    ///
    /// 10-bit data is packed as (\[8:8:8:8:2222\]).
    ///
    /// The data is unpacked to LSB and padded with 6 zero bits.
    Y10Bpack = sys::rs2_format_RS2_FORMAT_Y10BPACK as i32,
    /// 12-bits per pixel interleaved.
    ///
    /// 12-bit left, 12-bit right.
    ///
    /// Each pixel is stored in a 24-bit word in little-endian order.
    Y12I = sys::rs2_format_RS2_FORMAT_Y12I as i32,
    /// 16-bit per pixel grayscale image
    Y16 = sys::rs2_format_RS2_FORMAT_Y16 as i32,
    /// 16-bit linear depth values
    ///
    /// The depth in metres is equal to depth scale multiplied by each pixel value.
    Z16 = sys::rs2_format_RS2_FORMAT_Z16 as i32,
    /// Variable-length Huffman-compressed 16-bit depth values
    Z16H = sys::rs2_format_RS2_FORMAT_Z16H as i32,
    /// 16-bit per pixel frame grabber format
    Fg = sys::rs2_format_RS2_FORMAT_FG as i32,
    /// 12-bit per pixel
    Y411 = sys::rs2_format_RS2_FORMAT_Y411 as i32,
    /// 12-bit per pixel interleaved. 12-bit left, 12-bit right.
    Y16I = sys::rs2_format_RS2_FORMAT_Y16I as i32,
    /// 24-bit for every pixel: y for each pixel, and u,v data for every four pixels - packed as 2 lines of y, 1 line of u,v
    M420 = sys::rs2_format_RS2_FORMAT_M420 as i32,
    /// Combined motion data, as in the combined_motion structure
    CombinedMotion = sys::rs2_format_RS2_FORMAT_COMBINED_MOTION as i32,
    // Not included since this just tells us the total number of formats
    //
    // Count = sys::rs2_format_RS2_FORMAT_COUNT,
}

#[cfg(test)]
mod tests {
    use super::*;
    use num_traits::FromPrimitive;

    #[test]
    fn all_variants_exist() {
        for i in 0..sys::rs2_format_RS2_FORMAT_COUNT as i32 {
            assert!(
                Rs2Format::from_i32(i).is_some(),
                "Rs2Format variant for ordinal {} does not exist.",
                i
            );
        }
    }
}
