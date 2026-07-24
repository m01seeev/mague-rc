use std::process::ExitCode;

use mague_rc::{app, config::Config, error::AppError, overlay, stt::install_tls_crypto_provider};
use tracing_subscriber::EnvFilter;

fn main() -> ExitCode {
    match run() {
        Ok(()) => ExitCode::SUCCESS,
        Err(error) => {
            eprintln!("error: {error}");
            ExitCode::FAILURE
        }
    }
}

fn run() -> Result<(), AppError> {
    init_tracing()?;
    install_tls_crypto_provider()?;
    let config = Config::load()?;
    match run_mode()? {
        RunMode::Overlay => {
            overlay::run(config).map_err(|error| AppError::Overlay(error.to_string()))
        }
        RunMode::Terminal => tokio::runtime::Builder::new_multi_thread()
            .enable_all()
            .build()
            .map_err(|error| AppError::Output(format!("failed to create runtime: {error}")))?
            .block_on(app::run(config)),
    }
}

#[derive(Clone, Copy)]
enum RunMode {
    Overlay,
    Terminal,
}

fn run_mode() -> Result<RunMode, AppError> {
    match std::env::args().nth(1).as_deref() {
        None | Some("--overlay") => Ok(RunMode::Overlay),
        Some("--terminal") => Ok(RunMode::Terminal),
        Some(argument) => Err(AppError::Argument(argument.to_owned())),
    }
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
