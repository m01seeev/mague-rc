use tracing::info;

use crate::{config::Config, error::AppError};

pub async fn run(config: Config) -> Result<(), AppError> {
    info!(
        module = "app",
        event = "started",
        audio_source = %config.audio.source,
        stt_model = %config.deepgram.model,
        llm_model = %config.llm.model,
        "AI Overlay started; press Ctrl+C to stop"
    );

    tokio::signal::ctrl_c().await.map_err(AppError::Shutdown)?;

    info!(
        module = "app",
        event = "shutdown",
        "shutdown signal received"
    );
    Ok(())
}
