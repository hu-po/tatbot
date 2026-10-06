//! Defines the frame type including sensor data.

mod composite;
mod image;
mod motion;
mod pixel;
mod points;
mod pose;
mod prelude;

pub use self::image::{
    ColorFrame, ConfidenceFrame, DepthFrame, DisparityFrame, FisheyeFrame, ImageFrame,
    InfraredFrame,
};
pub use self::motion::{AccelFrame, GyroFrame, MotionFrame};
pub use self::points::PointsFrame;
pub use composite::CompositeFrame;
pub use pixel::PixelKind;
pub use pose::{Confidence, PoseFrame};
pub use prelude::{FrameCategory, FrameConstructionError, FrameEx};
