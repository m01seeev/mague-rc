use std::{path::PathBuf, process::ExitCode};

use mague_rc::{
    app::{self, RunOptions},
    config::Config,
    error::AppError,
    output::TerminalOutputSink,
    overlay,
    stt::install_tls_crypto_provider,
    telemetry::TelemetryOutputSink,
};
use tokio::sync::mpsc;
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
    let mode = run_mode()?;
    let config = Config::load()?;
    match mode {
        RunMode::Overlay => {
            overlay::run(config).map_err(|error| AppError::Overlay(error.to_string()))
        }
        RunMode::Terminal => runtime()?.block_on(app::run(config)),
        RunMode::Benchmark {
            audio,
            label,
            reference,
        } => {
            if !audio.is_file() {
                return Err(AppError::Argument(format!(
                    "benchmark audio file does not exist: {}",
                    audio.display()
                )));
            }
            if let Some(path) = reference.as_ref()
                && !path.is_file()
            {
                return Err(AppError::Argument(format!(
                    "benchmark reference file does not exist: {}",
                    path.display()
                )));
            }
            let telemetry_directory = std::env::var_os("TELEMETRY_DIR")
                .map_or_else(|| PathBuf::from("telemetry"), PathBuf::from);
            let sink = TelemetryOutputSink::new(
                TerminalOutputSink,
                telemetry_directory,
                &label,
                &audio,
                reference.as_deref(),
                &config,
            )
            .map_err(|error| AppError::Output(error.to_string()))?;
            let (_command_sender, command_receiver) = mpsc::unbounded_channel();
            runtime()?.block_on(app::run_with_sink_options(
                config,
                sink,
                command_receiver,
                RunOptions {
                    audio_file: Some(audio),
                },
            ))
        }
    }
}

enum RunMode {
    Overlay,
    Terminal,
    Benchmark {
        audio: PathBuf,
        label: String,
        reference: Option<PathBuf>,
    },
}

fn run_mode() -> Result<RunMode, AppError> {
    let mut arguments = std::env::args().skip(1);
    match arguments.next().as_deref() {
        None => Ok(RunMode::Overlay),
        Some("--overlay") if arguments.next().is_none() => Ok(RunMode::Overlay),
        Some("--terminal") if arguments.next().is_none() => Ok(RunMode::Terminal),
        Some("--benchmark") => {
            let audio = arguments.next().map(PathBuf::from).ok_or_else(|| {
                AppError::Argument(
                    "--benchmark requires AUDIO and accepts [LABEL] [--reference FILE]".to_owned(),
                )
            })?;
            let mut label = "baseline".to_owned();
            let mut reference = None;
            let mut remaining = arguments.collect::<Vec<_>>().into_iter();
            if let Some(argument) = remaining.next() {
                if argument == "--reference" {
                    reference = Some(PathBuf::from(remaining.next().ok_or_else(|| {
                        AppError::Argument("--reference requires a file path".to_owned())
                    })?));
                } else {
                    label = argument;
                    if let Some(flag) = remaining.next() {
                        if flag != "--reference" {
                            return Err(AppError::Argument(format!(
                                "unexpected benchmark argument: {flag}"
                            )));
                        }
                        reference = Some(PathBuf::from(remaining.next().ok_or_else(|| {
                            AppError::Argument("--reference requires a file path".to_owned())
                        })?));
                    }
                }
            }
            if let Some(argument) = remaining.next() {
                return Err(AppError::Argument(format!(
                    "unexpected benchmark argument: {argument}"
                )));
            }
            Ok(RunMode::Benchmark {
                audio,
                label,
                reference,
            })
        }
        Some(argument) => Err(AppError::Argument(format!(
            "unknown argument `{argument}`; expected --overlay, --terminal, or --benchmark"
        ))),
    }
}

fn runtime() -> Result<tokio::runtime::Runtime, AppError> {
    tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()
        .map_err(|error| AppError::Output(format!("failed to create runtime: {error}")))
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
