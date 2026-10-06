//! Processing blocks available to the RealSense library
//!
//! Processing blocks are used to apply various filters and transformations to frame data.
//! Each processing block operates on frames and outputs modified versions.
//!
//! ## Available Processing Blocks
//!
//! ### Core Processing Blocks
//! - [`align`] - Aligns frames from different sensors to a common coordinate system
//! - [`colorizer`] - Converts depth frames to RGB format for visualization
//! - [`decimation`] - Reduces the resolution of depth frames
//! - [`pointcloud`] - Generates 3D point clouds from depth data
//!
//! ### Filtering Blocks
//! - [`spatial_filter`] - Performs spatial filtering to smooth depth data
//! - [`temporal_filter`] - Reduces temporal noise by averaging frames over time
//! - [`hole_filling`] - Fills holes in depth data using adjacent pixels
//! - [`threshold`] - Filters depth values based on min/max thresholds
//!
//! ### Transform Blocks
//! - [`disparity_transform`] - Converts between depth and disparity domains
//!
//! ### Configuration
//! - [`options`] - Runtime configuration options for processing blocks
//!
//! ## Usage Examples
//!
//! ### Basic Colorizer
//! ```rust,no_run
//! use realsense_rust::processing_blocks::colorizer::Colorizer;
//! use std::time::Duration;
//!
//! # fn example() -> anyhow::Result<()> {
//! let mut colorizer = Colorizer::new(5)?;
//!
//! // Process a depth frame
//! # let depth_frame = todo!(); // Get depth frame from pipeline
//! colorizer.queue(depth_frame)?;
//! let colorized_frame = colorizer.wait(Duration::from_millis(100))?;
//! # Ok(())
//! # }
//! ```
//!
//! ### Filter Chain
//! ```rust,no_run
//! use realsense_rust::processing_blocks::{
//!     decimation::Decimation,
//!     spatial_filter::SpatialFilter,
//!     temporal_filter::TemporalFilter,
//!     hole_filling::HoleFillingFilter,
//!     colorizer::Colorizer,
//! };
//! use std::time::Duration;
//!
//! # fn example() -> anyhow::Result<()> {
//! // Create filter chain
//! let mut decimation = Decimation::new(5)?;
//! let mut spatial_filter = SpatialFilter::new(5)?;
//! let mut temporal_filter = TemporalFilter::new(5)?;
//! let mut hole_filling = HoleFillingFilter::new(5)?;
//! let mut colorizer = Colorizer::new(5)?;
//!
//! # let depth_frame = todo!(); // Get depth frame
//! // Apply filters in sequence
//! decimation.queue(depth_frame)?;
//! let frame = decimation.wait(Duration::from_millis(100))?;
//!
//! spatial_filter.queue(frame)?;
//! let frame = spatial_filter.wait(Duration::from_millis(100))?;
//!
//! temporal_filter.queue(frame)?;
//! let frame = temporal_filter.wait(Duration::from_millis(100))?;
//!
//! hole_filling.queue(frame)?;
//! let frame = hole_filling.wait(Duration::from_millis(100))?;
//!
//! colorizer.queue(frame)?;
//! let colorized_frame = colorizer.wait(Duration::from_millis(100))?;
//! # Ok(())
//! # }
//! ```
//!
//! See the `examples/` directory for complete working examples.

pub mod align;
pub mod colorizer;
pub mod decimation;
pub mod disparity_transform;
pub mod errors;
pub mod hole_filling;
pub mod options;
pub mod pointcloud;
pub mod spatial_filter;
pub mod temporal_filter;
pub mod threshold;
