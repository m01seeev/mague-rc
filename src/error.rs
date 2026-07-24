use thiserror::Error;

use crate::{
    app::TranscriptWorkerError,
    audio::AudioError,
    config::ConfigError,
    llm::{LlmError, LlmQueueError, LlmWorkerError},
    output::TerminalOutputError,
    stt::SttError,
};

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

    #[error(transparent)]
    Llm(#[from] LlmError),

    #[error(transparent)]
    LlmQueue(#[from] LlmQueueError),

    #[error(transparent)]
    LlmWorker(#[from] LlmWorkerError),

    #[error(transparent)]
    TranscriptWorker(#[from] TranscriptWorkerError),

    #[error(transparent)]
    TerminalOutput(#[from] TerminalOutputError),

    #[error("{worker} worker task failed: {error}")]
    WorkerTask { worker: &'static str, error: String },

    #[error("{0} worker stopped unexpectedly")]
    WorkerStopped(&'static str),

    #[error("{0} worker did not stop within the shutdown timeout")]
    WorkerShutdownTimeout(&'static str),
}
