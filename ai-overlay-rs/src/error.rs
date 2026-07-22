use thiserror::Error;

use crate::config::ConfigError;

#[derive(Debug, Error)]
pub enum AppError {
    #[error(transparent)]
    Configuration(#[from] ConfigError),

    #[error("failed to install tracing subscriber: {0}")]
    Tracing(String),

    #[error("failed while waiting for shutdown signal: {0}")]
    Shutdown(#[source] std::io::Error),
}
