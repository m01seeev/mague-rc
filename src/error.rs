use thiserror::Error;

use crate::{audio::AudioError, config::ConfigError, stt::SttError};

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

    #[error(transparent)]
    Stt(#[from] SttError),

    #[error("{worker} worker task failed: {error}")]
    WorkerTask { worker: &'static str, error: String },

    #[error("{0} worker stopped unexpectedly")]
    WorkerStopped(&'static str),

    #[error("{0} worker did not stop within the shutdown timeout")]
    WorkerShutdownTimeout(&'static str),
}
