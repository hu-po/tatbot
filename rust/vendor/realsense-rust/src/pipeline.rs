//! Module containing pipeline types used in getting frames from the device.
//!
//! Pipelines are the core data type used to poll or wait for frames from librealsense2. The
//! pipeline is constructed from a [`Context`](crate::context::Context), and streaming is started
//! by feeding the pipeline a configuration (see: [`Config`](crate::config::Config)).
//!
//! In the librealsense2 C-API, there is no distinction between active (streaming / started)
//! pipelines and inactive (not streaming / stopped) pipelines. The type is instead merely
//! represented as a `* rs2_pipeline` for both scenarios. This leads to some of the APIs returning
//! an error if they are called on a pipeline that is not started or a pipeline that is stopped.
//!
//! To reduce this error surface, we have produced two separate types, [`ActivePipeline`] and
//! [`InactivePipeline`], which represent the two possible (valid) states of a pipeline. Interfaces
//! that are only valid with an inactive (stopped) pipeline are only provided alongside the
//! [`InactivePipeline`] type, while interfaces that are only valid with an active (started)
//! pipeline are only provided alongside the [`ActivePipeline`] type.
//!

mod active;
mod inactive;
mod profile;

pub use active::{ActivePipeline, FrameWaitError};
pub use inactive::{InactivePipeline, PipelineActivationError, PipelineConstructionError};
pub use profile::{PipelineProfile, PipelineProfileConstructionError};
