//! RGB-coloured point clouds from aligned RealSense depth, for the viewer only.
//!
//! The owner aligns Z16 depth onto its colour grid, so depth pixel (u, v) and
//! colour pixel (u, v) are the same ray. A cloud therefore needs only the
//! payload's pinhole (`rerun_viewer::scaled_pinhole`) and the colour image.
//! Rays use Rerun's pixel-centre convention, so this cloud lands where the
//! colormapped `DepthImage` cloud under the same Pinhole was drawn.
//!
//! A display derivative, never evidence: lens distortion is ignored
//! (`scripts/lib/rgbd_geometry.py` remains the deprojection authority), the
//! cloud is decimated to a fixed budget, each point a little wider than the
//! gap to its neighbours so the cloud reads as a surface, and each source's tint
//! shifts the measured colours so the wrist and overhead clouds read apart
//! where they overlap. The ROS bridge draws the wrist cloud with the same
//! styling (`tatbot_contracts.ros_scene.WRIST_CLOUD`); keep the two in step.

use crate::{FrameRecord, PixelFormat, RecordedPayload};

/// At most this many points per logged plane: the cockpit's 320x180 D555
/// depth (57,600 pixels) at every second pixel each way. The viewer keeps
/// every row it is sent.
pub const MAX_POINTS: usize = 14_400;

/// Points per logged row. A point is 12 bytes of position and 4 of colour, so
/// a band stays within the cockpit's 60 KiB row (`frame_bus::VIEWER_ROW_BYTES`):
/// the hub streams each chunk on over the link the pink controller shares.
pub const BAND_POINTS: usize = 3_600;

/// Rows a cloud is logged as, always all of them: a smaller cloud's leftover
/// rows go out empty, so no earlier cloud's band is still drawn.
pub const BANDS: usize = MAX_POINTS.div_ceil(BAND_POINTS);

/// Share of the source tint mixed into each measured colour: low enough that
/// the scene still reads as RGB, high enough to tell the clouds apart.
pub const TINT_SHARE: f32 = 0.1;

/// A point's radius as a share of the spacing between neighbouring points at
/// its range: a little over half, so neighbours overlap and the cloud reads
/// as a surface (as the depth image's colormapped cloud did), also where the
/// table slopes away from the camera, rather than as a grid of dots.
pub const RADIUS_SHARE: f32 = 0.6;

/// Where a depth camera sits, which sets how its cloud is drawn.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CloudSource {
    /// A wrist D405: centimetres away, sub-millimetre spacing.
    Wrist,
    /// The overhead D555: the whole table from about a metre.
    Overhead,
}

impl CloudSource {
    /// Wrist clouds lean warm (amber), overhead clouds lean cool (blue).
    pub fn tint(self) -> [u8; 3] {
        match self {
            Self::Wrist => [255, 150, 40],
            Self::Overhead => [40, 150, 255],
        }
    }
}

/// A colour image decoded to RGB8.
#[derive(Debug, Clone, PartialEq)]
pub struct ColorRaster {
    pub width: u32,
    pub height: u32,
    pub rgb: Vec<u8>,
}

impl ColorRaster {
    /// Decode one colour frame: raw BGR/RGB/YUYV/luma, or the owner's JPEG.
    /// Anything else, or a buffer that does not match its size, is `None`.
    pub fn from_frame(frame: &FrameRecord) -> Option<Self> {
        match &frame.payload {
            RecordedPayload::Video { format, width, height, bytes } => {
                Self::from_raw(*format, *width, *height, bytes, *width as usize * *height as usize)
            }
            // The image's own size, which the cockpit's profile also carries.
            RecordedPayload::Encoded { format: PixelFormat::Jpeg, bytes } => {
                let image = image::load_from_memory_with_format(bytes, image::ImageFormat::Jpeg)
                    .ok()?
                    .into_rgb8();
                Some(Self { width: image.width(), height: image.height(), rgb: image.into_raw() })
            }
            _ => None,
        }
    }

    fn from_raw(format: PixelFormat, width: u32, height: u32, bytes: &[u8], pixels: usize) -> Option<Self> {
        let rgb = match format {
            PixelFormat::Rgb8 if bytes.len() == pixels * 3 => bytes.to_vec(),
            PixelFormat::Bgr8 if bytes.len() == pixels * 3 => bytes
                .chunks_exact(3)
                .flat_map(|p| [p[2], p[1], p[0]])
                .collect(),
            PixelFormat::Y8 if bytes.len() == pixels => bytes.iter().flat_map(|&l| [l, l, l]).collect(),
            PixelFormat::Yuyv if width % 2 == 0 && bytes.len() == pixels * 2 => {
                crate::frame_ops::yuyv_to_rgb(bytes)
            }
            _ => return None,
        };
        Some(Self { width, height, rgb })
    }

    fn at(&self, x: u32, y: u32) -> [u8; 3] {
        let i = (y as usize * self.width as usize + x as usize) * 3;
        [self.rgb[i], self.rgb[i + 1], self.rgb[i + 2]]
    }

    /// The mean colour over what one decimated point of a `plane`-sized
    /// depth image stands for: the `step` x `step` pixels about pixel
    /// `(x, y)`'s ray, at the same normalized position in this image (the
    /// cockpit's colour is twice its depth's size). A mean, not one pixel,
    /// so a tag's cells or a ruled page come out as their shade, not noise.
    fn block_mean(&self, (x, y): (u32, u32), plane: (u32, u32), step: u32) -> [u8; 3] {
        let span = |index: u32, from: u32, to: u32| {
            let scale = f64::from(to) / f64::from(from);
            let (centre, half) = ((f64::from(index) + 0.5) * scale, f64::from(step) * scale / 2.0);
            let low = ((centre - half).floor().max(0.0) as u32).min(to - 1);
            low..((centre + half).ceil() as u32).clamp(low + 1, to)
        };
        let columns = span(x, plane.0, self.width);
        let (mut sum, mut count) = ([0_u32; 3], 0_u32);
        for row in span(y, plane.1, self.height) {
            for column in columns.clone() {
                let [r, g, b] = self.at(column, row);
                sum = [sum[0] + u32::from(r), sum[1] + u32::from(g), sum[2] + u32::from(b)];
                count += 1;
            }
        }
        sum.map(|total| ((total + count / 2) / count) as u8)
    }
}

#[derive(Debug, Default, PartialEq)]
pub struct ColoredCloud {
    /// Points in the colour optical frame (RDF: +x right, +y down, +z forward), metres.
    pub positions: Vec<[f32; 3]>,
    pub colors: Vec<[u8; 3]>,
    /// The spacing between neighbouring points per metre of range: the
    /// decimation step over the focal length.
    pub spacing_per_m: f32,
}

impl ColoredCloud {
    /// The radius to draw `rows` of this cloud with, metres: `RADIUS_SHARE`
    /// of the spacing at their mean range. A band is a run of image rows, so
    /// on a table seen at a slant its points' ranges are alike.
    pub fn radius_m(&self, rows: std::ops::Range<usize>) -> f32 {
        let points = &self.positions[rows];
        if points.is_empty() {
            return 0.0;
        }
        let mean_range = points.iter().map(|p| p[2]).sum::<f32>() / points.len() as f32;
        RADIUS_SHARE * self.spacing_per_m * mean_range
    }
}

/// Mix `share` of `tint` into `rgb`.
pub fn tinted(rgb: [u8; 3], tint: [u8; 3], share: f32) -> [u8; 3] {
    let share = share.clamp(0.0, 1.0);
    std::array::from_fn(|i| (rgb[i] as f32 * (1.0 - share) + tint[i] as f32 * share).round() as u8)
}

/// Back-project one aligned Z16 plane into a decimated, coloured cloud.
///
/// `focal` and `principal` are in this plane's pixels (the Pinhole logged
/// beside it). A zero sample is no return and is skipped. The colour image
/// may be another size than the plane (the cockpit logs colour at twice the
/// depth's size); each point takes its block's mean colour there
/// (`ColorRaster::block_mean`). Without a colour image each point takes the
/// tint alone.
pub fn back_project(
    depth: (u32, u32, &[u8]),
    raw_per_metre: f32,
    focal: [f32; 2],
    principal: [f32; 2],
    color: Option<&ColorRaster>,
    source: CloudSource,
    max_points: usize,
) -> ColoredCloud {
    let (width, height, bytes) = depth;
    let mut cloud = ColoredCloud::default();
    if width == 0
        || height == 0
        || max_points == 0
        || bytes.len() != width as usize * height as usize * 2
        || !(raw_per_metre.is_finite() && raw_per_metre > 0.0)
        || !(focal[0] > 0.0 && focal[1] > 0.0)
    {
        return cloud;
    }
    let color = color.filter(|c| c.width > 0 && c.height > 0);
    let pixels = width as usize * height as usize;
    let step = ((pixels as f64 / max_points as f64).sqrt().ceil() as usize).max(1);
    let tint = source.tint();
    cloud.spacing_per_m = step as f32 * 2.0 / (focal[0] + focal[1]);
    for y in (0..height).step_by(step) {
        for x in (0..width).step_by(step) {
            let i = (y as usize * width as usize + x as usize) * 2;
            let raw = u16::from_le_bytes([bytes[i], bytes[i + 1]]);
            if raw == 0 {
                continue;
            }
            let z = f32::from(raw) / raw_per_metre;
            cloud.positions.push([
                (x as f32 + 0.5 - principal[0]) / focal[0] * z,
                (y as f32 + 0.5 - principal[1]) / focal[1] * z,
                z,
            ]);
            let measured = color.map_or(tint, |c| c.block_mean((x, y), (width, height), step as u32));
            cloud.colors.push(tinted(measured, tint, TINT_SHARE));
        }
    }
    cloud
}

#[cfg(test)]
mod tests {
    use super::*;

    fn plane(width: u32, height: u32, raw: u16) -> Vec<u8> {
        std::iter::repeat_n(raw.to_le_bytes(), (width * height) as usize)
            .flatten()
            .collect()
    }

    fn raster(width: u32, height: u32, rgb: &[[u8; 3]]) -> ColorRaster {
        ColorRaster { width, height, rgb: rgb.iter().flatten().copied().collect() }
    }

    #[test]
    fn rays_follow_the_pinhole_pixel_centres_and_the_recorded_units() {
        // 0.1 mm units (D405): raw 5000 is 0.5 m; principal at the plane's centre.
        let bytes = plane(4, 4, 5000);
        let cloud = back_project((4, 4, &bytes), 10_000.0, [100.0, 100.0], [2.0, 2.0], None, CloudSource::Wrist, 100);
        assert_eq!(cloud.positions.len(), 16);
        // Pixel (0,0)'s centre is 1.5 px left of and above the principal point.
        let [x, y, z] = cloud.positions[0];
        assert!((x + 0.0075).abs() < 1e-6 && (y + 0.0075).abs() < 1e-6 && z == 0.5);
        // Pixel (2,2)'s centre is half a pixel past it.
        let [x, y, _] = cloud.positions[2 * 4 + 2];
        assert!((x - 0.0025).abs() < 1e-6 && (y - 0.0025).abs() < 1e-6);
    }

    #[test]
    fn no_return_is_skipped_and_the_budget_holds() {
        let mut bytes = plane(100, 100, 1000);
        bytes[..2].copy_from_slice(&0u16.to_le_bytes());
        let all = back_project((100, 100, &bytes), 1000.0, [80.0, 80.0], [50.0, 50.0], None, CloudSource::Overhead, 1_000_000);
        assert_eq!(all.positions.len(), 100 * 100 - 1);
        let capped = back_project((100, 100, &bytes), 1000.0, [80.0, 80.0], [50.0, 50.0], None, CloudSource::Overhead, 500);
        assert!(capped.positions.len() <= 500 && !capped.positions.is_empty());
        assert_eq!(capped.positions.len(), capped.colors.len());
        let short = &bytes[..10];
        assert!(back_project((100, 100, short), 1000.0, [80.0, 80.0], [50.0, 50.0], None, CloudSource::Overhead, 500)
            .positions
            .is_empty());
        assert!(back_project((100, 100, &bytes), 1000.0, [0.0, 80.0], [50.0, 50.0], None, CloudSource::Overhead, 500)
            .positions
            .is_empty());
    }

    #[test]
    fn colours_are_sampled_at_the_same_place_in_a_larger_image_and_tinted_by_source() {
        let bytes = plane(2, 1, 1000);
        // A 4x2 image: left half red, right half blue (the cockpit's colour is twice the depth's size).
        let red = [255, 0, 0];
        let blue = [0, 0, 255];
        let image = raster(4, 2, &[red, red, blue, blue, red, red, blue, blue]);
        let wrist = back_project((2, 1, &bytes), 1000.0, [1.0, 1.0], [1.0, 0.5], Some(&image), CloudSource::Wrist, 10);
        let overhead = back_project((2, 1, &bytes), 1000.0, [1.0, 1.0], [1.0, 0.5], Some(&image), CloudSource::Overhead, 10);
        assert_eq!(wrist.colors[0], tinted(red, CloudSource::Wrist.tint(), TINT_SHARE));
        assert_eq!(overhead.colors[1], tinted(blue, CloudSource::Overhead.tint(), TINT_SHARE));
        assert_ne!(wrist.colors, overhead.colors);
        assert!(wrist.colors[0][0] > wrist.colors[0][2], "red stays red under the tint");
        let none = back_project((2, 1, &bytes), 1000.0, [1.0, 1.0], [1.0, 0.5], None, CloudSource::Overhead, 10);
        assert_eq!(none.colors[0], CloudSource::Overhead.tint());
    }

    #[test]
    fn a_point_is_drawn_wider_than_half_the_gap_to_its_neighbours_at_its_range() {
        // The cockpit's 320x180 plane takes every second pixel: neighbours
        // 2 px apart, 2/160 m apart per metre of range at a 160 px focal length.
        let (near, far) = (plane(320, 180, 400), plane(320, 180, 800));
        let project = |bytes: &[u8]| {
            back_project((320, 180, bytes), 1000.0, [160.0, 160.0], [160.0, 90.0], None, CloudSource::Overhead, MAX_POINTS)
        };
        let (near, far) = (project(&near), project(&far));
        assert_eq!((near.positions.len(), BANDS), (MAX_POINTS, 4));
        let (spacing, near_radius) = (2.0 / 160.0 * 0.4, near.radius_m(0..BAND_POINTS));
        // A mean of 3,600 f32 ranges: equal to a part in 10^4.
        assert!((near_radius / (RADIUS_SHARE * spacing) - 1.0).abs() < 1e-4, "{near_radius}");
        assert!(2.0 * near_radius > spacing, "neighbours overlap: a surface, not dots");
        assert!((far.radius_m(BAND_POINTS..2 * BAND_POINTS) / near_radius - 2.0).abs() < 1e-4);
        assert_eq!(near.radius_m(0..0), 0.0, "an empty band");
    }

    #[test]
    fn a_point_takes_the_mean_colour_of_what_it_stands_for() {
        // A one-pixel checker at twice the depth's size: each depth pixel
        // covers two white and two black pixels, so its point is mid-grey,
        // where one sampled pixel would be black or white.
        let (white, black) = ([255, 255, 255], [0, 0, 0]);
        let row = |first: [u8; 3], second: [u8; 3]| [first, second, first, second];
        let checker: Vec<[u8; 3]> = [row(white, black), row(black, white)].repeat(2).concat();
        let image = raster(4, 4, &checker);
        let bytes = plane(2, 2, 1000);
        let cloud = back_project((2, 2, &bytes), 1000.0, [1.0, 1.0], [1.0, 1.0], Some(&image), CloudSource::Overhead, 4);
        let grey = tinted([128, 128, 128], CloudSource::Overhead.tint(), TINT_SHARE);
        assert_eq!(cloud.colors, vec![grey; 4]);
    }

    #[test]
    fn raw_colour_formats_decode_to_rgb() {
        let pixels = 2;
        let bgr = ColorRaster::from_raw(PixelFormat::Bgr8, 2, 1, &[0, 0, 255, 255, 0, 0], pixels).unwrap();
        assert_eq!((bgr.at(0, 0), bgr.at(1, 0)), ([255, 0, 0], [0, 0, 255]));
        let yuyv = ColorRaster::from_raw(PixelFormat::Yuyv, 2, 1, &[50, 128, 200, 128], pixels).unwrap();
        assert_eq!((yuyv.at(0, 0), yuyv.at(1, 0)), ([50, 50, 50], [200, 200, 200]));
        assert!(ColorRaster::from_raw(PixelFormat::Rgb8, 2, 1, &[0; 3], pixels).is_none());
        assert!(ColorRaster::from_raw(PixelFormat::Z16, 2, 1, &[0; 4], pixels).is_none());
    }

    #[test]
    fn the_owners_jpeg_decodes() {
        use image::{ExtendedColorType, codecs::jpeg::JpegEncoder};
        let rgb: Vec<u8> = std::iter::repeat_n([200u8, 30, 30], 16 * 8).flatten().collect();
        let mut jpeg = Vec::new();
        JpegEncoder::new_with_quality(&mut jpeg, 95)
            .encode(&rgb, 16, 8, ExtendedColorType::Rgb8)
            .unwrap();
        let metadata: crate::FrameMetadata = serde_json::from_value(serde_json::json!({
            "sensor_name": "overhead_depth_color", "sensor_kind": "real_sense", "sequence": 1,
            "profile": {"stream": "color", "width": 16, "height": 8, "fps_num": 30, "fps_den": 1, "format": "jpeg"},
            "timestamps": {"source_ns": null, "source_domain": "host_unix", "rtp_timestamp": null,
                "pipeline_pts_ns": null, "pipeline_dts_ns": null, "host_monotonic_ns": 0, "host_unix_ns": 1,
                "normalized_unix_ns": null},
            "dropped_before": 0, "calibration_id": null, "flags": [], "attributes": {}
        }))
        .unwrap();
        let frame = FrameRecord {
            metadata,
            payload: RecordedPayload::Encoded { format: PixelFormat::Jpeg, bytes: jpeg },
        };
        let raster = ColorRaster::from_frame(&frame).unwrap();
        assert_eq!((raster.width, raster.height), (16, 8));
        let [r, g, b] = raster.at(8, 4);
        assert!(r > 150 && g < 80 && b < 80, "{r} {g} {b}");
    }
}
