use std::{path::PathBuf, process::ExitCode};

use mague_rc::{
    app::{self, RunOptions},
    config::Config,
    error::AppError,
    knowledge::RemoteRetriever,
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
    match mode {
        RunMode::Overlay => {
            let config = Config::load()?;
            overlay::run(config).map_err(|error| AppError::Overlay(error.to_string()))
        }
        RunMode::Terminal => runtime()?.block_on(app::run(Config::load()?)),
        RunMode::Benchmark {
            audio,
            label,
            reference,
        } => {
            let config = Config::load()?;
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
        RunMode::RagIndex { source } => {
            let mut retriever = RemoteRetriever::load(Config::load_embedding()?)?;
            let report = runtime()?.block_on(retriever.index(&source))?;
            println!(
                "indexed {} source(s), {} section(s), {} chunk(s), {} dimensions",
                report.source_count, report.section_count, report.chunk_count, report.dimension
            );
            println!(
                "embedding: {} call(s), {} input token(s), {} total token(s)",
                report.embedding_calls, report.prompt_tokens, report.total_tokens
            );
            println!(
                "timing: remote_embedding={} ms total={} ms",
                report.embedding.as_millis(),
                report.total.as_millis()
            );
            println!(
                "index: {} ({} bytes)",
                report.index_path.display(),
                report.index_bytes
            );
            Ok(())
        }
        RunMode::RagQuery {
            query,
            top_k,
            repeat,
        } => {
            let mut retriever = RemoteRetriever::load(Config::load_embedding()?)?;
            let runtime = runtime()?;
            for iteration in 1..=repeat {
                let report = runtime.block_on(retriever.query(&query, top_k))?;
                println!(
                    "run {iteration}/{repeat}: remote_embedding={} ms search={} ms; {} input token(s)",
                    report.embedding.as_millis(),
                    report.search.as_millis(),
                    report.prompt_tokens
                );
                if iteration == repeat {
                    println!(
                        "embedding: {} call(s), {} input token(s), {} total token(s)",
                        report.embedding_calls, report.prompt_tokens, report.total_tokens
                    );
                    println!("index: {}", report.index_path.display());
                    for (position, hit) in report.hits.iter().enumerate() {
                        println!(
                            "\n#{} score={:.4} dense={:.4} lexical={:.4}\n{} [{}]\n{}",
                            position + 1,
                            hit.combined_score,
                            hit.dense_score,
                            hit.lexical_score,
                            hit.heading,
                            hit.source.display(),
                            excerpt(&hit.text, 500)
                        );
                    }
                }
            }
            Ok(())
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
    RagIndex {
        source: PathBuf,
    },
    RagQuery {
        query: String,
        top_k: usize,
        repeat: usize,
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
        Some("rag") => match arguments.next().as_deref() {
            Some("index") => {
                let source = arguments.next().map(PathBuf::from).ok_or_else(|| {
                    AppError::Argument("rag index requires a Markdown file or directory".to_owned())
                })?;
                if let Some(argument) = arguments.next() {
                    return Err(AppError::Argument(format!(
                        "unexpected rag index argument: {argument}"
                    )));
                }
                Ok(RunMode::RagIndex { source })
            }
            Some("query") => parse_rag_query(arguments.collect()),
            Some(command) => Err(AppError::Argument(format!(
                "unknown rag command `{command}`; expected index or query"
            ))),
            None => Err(AppError::Argument(
                "rag requires an index or query command".to_owned(),
            )),
        },
        Some(argument) => Err(AppError::Argument(format!(
            "unknown argument `{argument}`; expected --overlay, --terminal, --benchmark, or rag"
        ))),
    }
}

fn parse_rag_query(arguments: Vec<String>) -> Result<RunMode, AppError> {
    let mut query_parts = Vec::new();
    let mut top_k = 5_usize;
    let mut repeat = 1_usize;
    let mut arguments = arguments.into_iter();
    while let Some(argument) = arguments.next() {
        if argument == "--top" || argument == "--repeat" {
            let value = arguments.next().ok_or_else(|| {
                AppError::Argument(format!("{argument} requires a positive number"))
            })?;
            let parsed = value.parse().map_err(|_| {
                AppError::Argument(format!(
                    "invalid {argument} value `{value}`; expected a number"
                ))
            })?;
            if parsed == 0 {
                return Err(AppError::Argument(format!(
                    "{argument} must be greater than zero"
                )));
            }
            if argument == "--top" {
                top_k = parsed;
            } else {
                repeat = parsed;
            }
        } else {
            query_parts.push(argument);
        }
    }
    if query_parts.is_empty() {
        return Err(AppError::Argument(
            "rag query requires question text".to_owned(),
        ));
    }
    Ok(RunMode::RagQuery {
        query: query_parts.join(" "),
        top_k,
        repeat,
    })
}

fn excerpt(text: &str, max_chars: usize) -> String {
    let mut excerpt = text.chars().take(max_chars).collect::<String>();
    if text.chars().count() > max_chars {
        excerpt.push_str("...");
    }
    excerpt
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
