use thiserror::Error;

use crate::{audio::AudioError, config::ConfigError};

#[derive(Debug, Error)]
pub enum AppError {
    #[error(transparent)]
    Configuration(#[from] ConfigError),

    #[error("failed to install tracing subscriber: {0}")]
    Tracing(String),

    #[error("failed while waiting for shutdown signal: {0}")]
    Shutdown(#[source] std::io::Error),

    #[error(transparent)]
    Audio(#[from] AudioError),

    #[error("audio worker task failed: {0}")]
    AudioTask(String),

    #[error("audio worker stopped unexpectedly")]
    AudioTaskStopped,

    #[error("audio worker did not stop within the shutdown timeout")]
    AudioShutdownTimeout,
}
