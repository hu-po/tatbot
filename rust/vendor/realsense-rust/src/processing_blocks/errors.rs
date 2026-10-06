//! Errors that can occur when handling processing blocks.

use crate::kind::Rs2Exception;
use thiserror::Error;

/// Enumerations of possible errors that can occur when creating a Processing Block
#[derive(Error, Debug, Clone, PartialEq, Eq)]
pub enum ProcessingBlockConstructionError {
    /// Could not create the processing block
    #[error("Could not create the processing block. Type: {0}; Reason: {1}")]
    CouldNotCreateProcessingBlock(Rs2Exception, String),

    /// Could not create the processing queue
    #[error("Could not create the processing queue. Type: {0}; Reason: {1}")]
    CouldNotCreateProcessingQueue(Rs2Exception, String),

    /// Could not start processing the queue
    #[error("Could not start processing the queue. Type: {0}; Reason: {1}")]
    CouldNotStartProcessingQueue(Rs2Exception, String),
}

/// Enumerations of possible errors that can occur when processing a Processing Block
#[derive(Debug, Clone, Error, PartialEq, Eq)]
#[error("Kind: {kind}; Reason: {context}")]
pub struct ProcessFrameError {
    /// The RS2 exception that occurred
    pub kind: Rs2Exception,
    /// The context in which the error occurred
    pub context: String,
}
