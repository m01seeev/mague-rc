use std::process::ExitCode;

use mague_rc::{app, config::Config, error::AppError, stt::install_tls_crypto_provider};
use tracing_subscriber::EnvFilter;

#[tokio::main]
async fn main() -> ExitCode {
    match run().await {
        Ok(()) => ExitCode::SUCCESS,
        Err(error) => {
            eprintln!("error: {error}");
            ExitCode::FAILURE
        }
    }
}

async fn run() -> Result<(), AppError> {
    init_tracing()?;
    install_tls_crypto_provider()?;
    let config = Config::load()?;
    app::run(config).await
}

fn init_tracing() -> Result<(), AppError> {
    let filter = EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new("warn"));

    tracing_subscriber::fmt()
        .with_env_filter(filter)
        .with_target(false)
        .compact()
        .try_init()
        .map_err(|error| AppError::Tracing(error.to_string()))
}
