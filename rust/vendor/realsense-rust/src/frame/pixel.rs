//! Type for representing the various pixel formats.
//!
//! Correct formats are determined by the device type and the streaming profile.
//! For detailed pixel format information, check the
//! [Intel RealSense SDK code](https://github.com/IntelRealSense/librealsense/blob/4f37f2ef0874c1716bce223b20e46d00532ffb04/wrappers/nodejs/index.js#L3865).

use crate::kind::Rs2Format;
use std::{os::raw::c_void, slice};

/// Type for representing the various pixel formats.
#[derive(Debug)]
pub enum PixelKind<'a> {
    /// 32-bit `y0, u, y1, v` data for every two pixels.
    /// Similar to YUV422 but packed in a different order - see [this link](https://en.wikipedia.org/wiki/YUV).
    Yuyv {
        /// The Y / luma value for a given pixel
        y: &'a u8,
        /// The U-chroma value for a given pixel
        u: &'a u8,
        /// The V-chroma value for a given pixel
        v: &'a u8,
    },
    /// Similar to the standard YUYV pixel format, but packed in a different order.
    Uyvy {
        /// The Y / luma value for a given pixel
        y: &'a u8,
        /// The U-chroma value for a given pixel
        u: &'a u8,
        /// The V-chroma value for a given pixel
        v: &'a u8,
    },
    /// 8-bit blue, green, and red channels -- suitable for OpenCV.
    Bgr8 {
        /// The blue component of the BGR pixel
        b: &'a u8,
        /// The green component of the BGR pixel
        g: &'a u8,
        /// The red component of the BGR pixel
        r: &'a u8,
    },
    /// 8-bit blue, green, and red channels + constant alpha channel equal to FF.
    Bgra8 {
        /// The blue component of the BGRA pixel
        b: &'a u8,
        /// The green component of the BGRA pixel
        g: &'a u8,
        /// The red component of the BGRA pixel
        r: &'a u8,
        /// The alpha component of the BGRA pixel
        a: &'a u8,
    },
    /// 8-bit red, green and blue channels.
    Rgb8 {
        /// The red component of the RGB pixel
        r: &'a u8,
        /// The green component of the RGB pixel
        g: &'a u8,
        /// The blue component of the RGB pixel
        b: &'a u8,
    },
    /// 8-bit red, green and blue channels + constant alpha channel equal to FF.
    Rgba8 {
        /// The red component of the RGBA pixel
        r: &'a u8,
        /// The green component of the RGBA pixel
        g: &'a u8,
        /// The blue component of the RGBA pixel
        b: &'a u8,
        /// The alpha component of the RGBA pixel
        a: &'a u8,
    },
    /// 8-bit raw image.
    Raw8 {
        /// The single luma value for a RAW8 image
        val: &'a u8,
    },
    /// 8-bit per-pixel grayscale image.
    Y8 {
        /// The single luma value for a Y-channel only image
        y: &'a u8,
    },
    /// 16-bit per-pixel grayscale image.
    Y16 {
        /// The single luma value for a Y-channel only image
        y: &'a u16,
    },
    /// 16-bit linear depth values. The depth is meters is equal to depth scale * pixel value.
    Z16 {
        /// Depth value in millimetres
        depth: &'a u16,
    },
    /// 32-bit float-point depth distance value.
    Distance {
        /// Distance from camera origin in metres
        distance: &'a f32,
    },
    /// 32-bit float-point disparity values. Depth->Disparity conversion : Disparity = Baseline*FocalLength/Depth.
    Disparity32 {
        /// The disparity relative to the opposite eye in a depth camera.
        disparity: &'a f32,
    },
    /// 32-bit floating point 3D coordinates.
    Xyz32f {
        /// The X-coordinate
        x: &'a f32,
        /// The Y-coordinate
        y: &'a f32,
        /// The Z-coordinate
        z: &'a f32,
    },
}

/// Method to retrieve a pixel from a given rs2_frame in the requested Pixel format.
///
/// # Safety
///
/// This method should only be called from the ImageFrame types themselves, as this
/// is the only place where proper pointer management happens.
#[inline]
pub(crate) unsafe fn get_pixel<'a>(
    format: Rs2Format,
    data_size_in_bytes: usize,
    data: *const c_void,
    stride_in_bytes: usize,
    col: usize,
    row: usize,
) -> PixelKind<'a> {
    // Realsense stores frame data in row-major format. Normally, we would offset into a
    // uniform array in column major format with the following equation:
    //
    // offset = row * width + column
    //
    // The assumption here being that it is a uniform array. See individual comments below for
    // how each offset equation differs.
    //
    // NOTE; You _could_ still represent this same pointer arithmetic in row-major form, but be
    // warned that the equations will look fairly different.
    //
    match format {
        // YUYV is not uniform since it encapsulates two pixels over 32 bits (four u8
        // values). Instead, we can index YUYV (and UYVY) as follows:
        //
        // offset = (row * stride) + (col / 2) * 4
        //
        // The strange part here is the (col / 2) * 4. This is done because on odd rows we
        // don't want to offset to the next Y value, but rather take the full YUYV and pick
        // the correct Y depending on whether the row is even or odd.
        //
        // NOTE: Order matters because we are taking advantage of integer division here.
        //
        Rs2Format::Yuyv => {
            let slice = slice::from_raw_parts(data.cast::<u8>(), data_size_in_bytes);
            let offset = (row * stride_in_bytes) + (col / 2) * 4;

            let y = if row.is_multiple_of(2) {
                slice.get_unchecked(offset)
            } else {
                slice.get_unchecked(offset + 2)
            };

            PixelKind::Yuyv {
                y,
                u: slice.get_unchecked(offset + 1),
                v: slice.get_unchecked(offset + 3),
            }
        }
        // UYVY follows from the same exact pattern we use for YUYV, since it's more or less a
        // re-ordering of the underlying data.
        //
        Rs2Format::Uyvy => {
            let slice = slice::from_raw_parts(data.cast::<u8>(), data_size_in_bytes);
            let offset = (row * stride_in_bytes) + (col / 2) * 4;

            let y = if row.is_multiple_of(2) {
                slice.get_unchecked(offset + 1)
            } else {
                slice.get_unchecked(offset + 3)
            };

            PixelKind::Uyvy {
                y,
                u: slice.get_unchecked(offset),
                v: slice.get_unchecked(offset + 2),
            }
        }
        // For BGR / RGB, we do a similar trick, but since pixels aren't interleaved as they
        // are with YUYV / UYVY, the multipliers for column and row offsets can be uniform.
        //
        Rs2Format::Bgr8 => {
            let slice = slice::from_raw_parts(data.cast::<u8>(), data_size_in_bytes);
            let offset = (row * stride_in_bytes) + (col * 3);

            PixelKind::Bgr8 {
                b: slice.get_unchecked(offset),
                g: slice.get_unchecked(offset + 1),
                r: slice.get_unchecked(offset + 2),
            }
        }
        // BGRA8 is more or less the same as BGR8, except we use 4 as a multiplier.
        //
        Rs2Format::Bgra8 => {
            let slice = slice::from_raw_parts(data.cast::<u8>(), data_size_in_bytes);
            let offset = (row * stride_in_bytes) + (col * 4);

            PixelKind::Bgra8 {
                b: slice.get_unchecked(offset),
                g: slice.get_unchecked(offset + 1),
                r: slice.get_unchecked(offset + 2),
                a: slice.get_unchecked(offset + 3),
            }
        }
        // RGB8 is the same as BGR8, the order is just different.
        //
        Rs2Format::Rgb8 => {
            let slice = slice::from_raw_parts(data.cast::<u8>(), data_size_in_bytes);
            let offset = (row * stride_in_bytes) + (col * 3);

            PixelKind::Bgr8 {
                r: slice.get_unchecked(offset),
                g: slice.get_unchecked(offset + 1),
                b: slice.get_unchecked(offset + 2),
            }
        }
        // RGBA8 is the same as BGRA8, the order is just different.
        //
        Rs2Format::Rgba8 => {
            let slice = slice::from_raw_parts(data.cast::<u8>(), data_size_in_bytes);
            let offset = (row * stride_in_bytes) + (col * 4);

            PixelKind::Bgra8 {
                r: slice.get_unchecked(offset),
                g: slice.get_unchecked(offset + 1),
                b: slice.get_unchecked(offset + 2),
                a: slice.get_unchecked(offset + 3),
            }
        }
        Rs2Format::Raw8 => {
            let slice = slice::from_raw_parts(data.cast::<u8>(), data_size_in_bytes);
            let offset = (row * stride_in_bytes) + col;

            PixelKind::Raw8 {
                val: slice.get_unchecked(offset),
            }
        }
        Rs2Format::Y8 => {
            let slice = slice::from_raw_parts(data.cast::<u8>(), data_size_in_bytes);
            let offset = (row * stride_in_bytes) + col;

            PixelKind::Y8 {
                y: slice.get_unchecked(offset),
            }
        }
        Rs2Format::Y16 => {
            let size = data_size_in_bytes / std::mem::size_of::<u16>();
            let stride = stride_in_bytes / std::mem::size_of::<u16>();
            let slice = slice::from_raw_parts(data.cast::<u16>(), size);
            let offset = (row * stride) + col;

            PixelKind::Y16 {
                y: slice.get_unchecked(offset),
            }
        }
        Rs2Format::Z16 => {
            let size = data_size_in_bytes / std::mem::size_of::<u16>();
            let stride = stride_in_bytes / std::mem::size_of::<u16>();
            let slice = slice::from_raw_parts(data.cast::<u16>(), size);
            let offset = (row * stride) + col;

            PixelKind::Z16 {
                depth: slice.get_unchecked(offset),
            }
        }
        Rs2Format::Distance => {
            let size = data_size_in_bytes / std::mem::size_of::<f32>();
            let stride = stride_in_bytes / std::mem::size_of::<f32>();
            let slice = slice::from_raw_parts(data.cast::<f32>(), size);
            let offset = (row * stride) + col;

            PixelKind::Distance {
                distance: slice.get_unchecked(offset),
            }
        }
        Rs2Format::Disparity32 => {
            let size = data_size_in_bytes / std::mem::size_of::<f32>();
            let stride = stride_in_bytes / std::mem::size_of::<f32>();
            let slice = slice::from_raw_parts(data.cast::<f32>(), size);
            let offset = (row * stride) + col;

            PixelKind::Disparity32 {
                disparity: slice.get_unchecked(offset),
            }
        }
        Rs2Format::Xyz32F => {
            let size = data_size_in_bytes / std::mem::size_of::<f32>();
            let stride = stride_in_bytes / std::mem::size_of::<f32>();
            let slice = slice::from_raw_parts(data.cast::<f32>(), size);
            let offset = (row * stride) + col;

            PixelKind::Xyz32f {
                x: slice.get_unchecked(offset),
                y: slice.get_unchecked(offset + 1),
                z: slice.get_unchecked(offset + 2),
            }
        }
        _ => {
            panic!("Unsupported video format.");
        }
    }
}
