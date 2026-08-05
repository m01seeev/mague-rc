use std::{path::PathBuf, time::Duration};

use tokio::{
    sync::{mpsc, watch},
    task::{JoinError, JoinHandle},
    time::timeout,
};
use tracing::{debug, info, warn};

use crate::{
    audio::{AudioError, AudioSource, FfmpegAudioSource, audio_frame_channel},
    config::Config,
    control::TerminalEchoGuard,
    error::AppError,
    events::{
        AppCommand, AppErrorView, LlmCommand, Mode, OutputComponent, OutputEvent, Speaker,
        StatusKind, StatusMessage,
    },
    knowledge::{KnowledgeWorkerStats, spawn_knowledge_worker},
    llm::{LlmWorker, OpenRouterTextProvider, llm_request_channel},
    output::{OutputSink, TerminalOutputSink},
    stt::{DeepgramSttProvider, SpeechToTextProvider, SttError},
};

mod boundary;
mod retrieval;
mod transcript_worker;

pub use transcript_worker::TranscriptWorkerError;
use retrieval::RetrievalPipeline;
use transcript_worker::{
    TranscriptCommand, run_transcript_windows_with_retrieval, send_output,
};

const SHUTDOWN_TIMEOUT: Duration = Duration::from_secs(5);
const BENCHMARK_SHUTDOWN_TIMEOUT: Duration = Duration::from_secs(300);
const BENCHMARK_EOF_DRAIN: Duration = Duration::from_secs(3);

#[derive(Clone, Debug, Default)]
pub struct RunOptions {
    pub audio_file: Option<PathBuf>,
}

pub async fn run(config: Config) -> Result<(), AppError> {
    let (_command_sender, command_receiver) = mpsc::unbounded_channel();
    run_with_sink(config, TerminalOutputSink, command_receiver).await
}

pub async fn run_with_sink<S>(
    config: Config,
    sink: S,
    app_commands: mpsc::UnboundedReceiver<AppCommand>,
) -> Result<(), AppError>
where
    S: OutputSink,
    S::Error: std::fmt::Display,
{
    run_with_sink_options(config, sink, app_commands, RunOptions::default()).await
}

pub async fn run_with_sink_options<S>(
    config: Config,
    sink: S,
    mut app_commands: mpsc::UnboundedReceiver<AppCommand>,
    options: RunOptions,
) -> Result<(), AppError>
where
    S: OutputSink,
    S::Error: std::fmt::Display,
{
    let _terminal_echo_guard = TerminalEchoGuard::hide_control_characters();
    let audio_source_label = options.audio_file.as_ref().map_or_else(
        || config.audio.source.clone(),
        |path| path.display().to_string(),
    );
    let benchmark_mode = options.audio_file.is_some();
    let candidate_source_label = if config.candidate_audio.enabled && !benchmark_mode {
        config.candidate_audio.source.as_str()
    } else {
        "disabled"
    };

    info!(
        module = "app",
        event = "started",
        audio_source = %audio_source_label,
        candidate_source = %candidate_source_label,
        stt_model = %config.deepgram.model,
        llm_model = %config.llm.model,
        transcript_segmentation = "deepgram_utterance",
        transcript_inactivity_timeout_sec = config.transcript.window_sec,
        "mague-rc started; press Ctrl+C to stop"
    );

    let (frame_sender, frame_receiver) = audio_frame_channel(config.audio.queue_max);
    let (stt_event_sender, stt_event_receiver) = mpsc::unbounded_channel();
    let (llm_request_sender, llm_request_receiver) = llm_request_channel(config.llm.queue_max);
    let (output_sender, output_receiver) = mpsc::unbounded_channel::<OutputEvent>();
    let (llm_command_sender, llm_command_receiver) = mpsc::unbounded_channel::<LlmCommand>();
    let (transcript_command_sender, transcript_command_receiver) = mpsc::unbounded_channel();
    let (candidate_command_sender, candidate_command_receiver) = mpsc::unbounded_channel();
    let (shutdown_sender, shutdown_receiver) = watch::channel(false);
    let mut active_mode = Mode::Voice;
    let candidate_enabled = config.candidate_audio.enabled && !benchmark_mode;

    send_output(
        &output_sender,
        OutputEvent::Status(StatusMessage {
            kind: StatusKind::Started,
            text: format!(
                "interviewer={} | candidate={} | STT={} | LLM={}",
                audio_source_label,
                candidate_source_label,
                config.deepgram.model,
                config.llm.model
            ),
        }),
    )?;
    send_output(
        &output_sender,
        OutputEvent::ModeChanged { mode: active_mode },
    )?;

    let (retrieval, knowledge_task) = if config.knowledge.enabled {
        match spawn_knowledge_worker(config.knowledge.embedding.clone()) {
            Ok(runtime) => (
                Some(RetrievalPipeline::new(
                    config.knowledge.clone(),
                    runtime.requests,
                    runtime.results,
                    runtime.readiness,
                )),
                Some(runtime.task),
            ),
            Err(error) => {
                warn!(
                    module = "knowledge",
                    event = "disabled",
                    error = %error,
                    "remote knowledge retrieval disabled"
                );
                send_output(
                    &output_sender,
                    OutputEvent::Error(AppErrorView {
                        component: OutputComponent::Knowledge,
                        message: error.to_string(),
                    }),
                )?;
                (None, None)
            }
        }
    } else {
        (None, None)
    };

    let audio_source = match options.audio_file {
        Some(path) => FfmpegAudioSource::from_file(
            config.audio.clone(),
            config.deepgram.sample_rate,
            config.deepgram.channels,
            path,
        ),
        None => FfmpegAudioSource::new(
            config.audio.clone(),
            config.deepgram.sample_rate,
            config.deepgram.channels,
        ),
    };
    let stt_provider = DeepgramSttProvider::new(config.deepgram.clone());
    let llm_provider = OpenRouterTextProvider::new(config.llm.clone())?;
    let llm_worker = LlmWorker::new(llm_provider, config.llm);

    let frame_sender_guard = frame_sender.clone();
    let (readiness_sender, readiness_receiver) = watch::channel(false);
    let transcript_readiness = benchmark_mode.then(|| readiness_receiver.clone());
    let audio_shutdown = shutdown_receiver.clone();
    let mut audio_task = if benchmark_mode {
        tokio::spawn(async move {
            wait_for_stt_readiness(readiness_receiver).await?;
            audio_source.run(frame_sender, audio_shutdown).await
        })
    } else {
        drop(readiness_receiver);
        tokio::spawn(audio_source.run(frame_sender, audio_shutdown))
    };
    let mut stt_task = if benchmark_mode {
        tokio::spawn(stt_provider.run_with_readiness(
            frame_receiver,
            stt_event_sender,
            shutdown_receiver.clone(),
            readiness_sender,
        ))
    } else {
        drop(readiness_sender);
        tokio::spawn(stt_provider.run(
            frame_receiver,
            stt_event_sender,
            shutdown_receiver.clone(),
        ))
    };
    let mut transcript_task = tokio::spawn(run_transcript_windows_with_retrieval(
        stt_event_receiver,
        llm_request_sender.clone(),
        output_sender.clone(),
        transcript_command_receiver,
        config.transcript.clone(),
        transcript_readiness,
        Speaker::Interviewer,
        retrieval,
    ));
    let mut candidate_audio_task = None;
    let mut candidate_stt_task = None;
    let mut candidate_transcript_task = None;
    let mut candidate_frame_sender_guard = None;
    if candidate_enabled {
        let (candidate_frame_sender, candidate_frame_receiver) =
            audio_frame_channel(config.audio.queue_max);
        let (candidate_event_sender, candidate_event_receiver) = mpsc::unbounded_channel();
        let mut candidate_audio_config = config.audio.clone();
        candidate_audio_config.source = config.candidate_audio.source.clone();
        let candidate_audio_source = FfmpegAudioSource::new(
            candidate_audio_config,
            config.deepgram.sample_rate,
            config.deepgram.channels,
        );
        let candidate_stt_provider = DeepgramSttProvider::new(config.deepgram.clone());
        candidate_frame_sender_guard = Some(candidate_frame_sender.clone());
        candidate_audio_task = Some(tokio::spawn(candidate_audio_source.run(
            candidate_frame_sender,
            shutdown_receiver.clone(),
        )));
        candidate_stt_task = Some(tokio::spawn(candidate_stt_provider.run(
            candidate_frame_receiver,
            candidate_event_sender,
            shutdown_receiver.clone(),
        )));
        candidate_transcript_task = Some(tokio::spawn(run_transcript_windows_with_retrieval(
            candidate_event_receiver,
            llm_request_sender,
            output_sender.clone(),
            candidate_command_receiver,
            config.transcript,
            None,
            Speaker::Candidate,
            None,
        )));
    } else {
        drop(candidate_command_receiver);
        drop(llm_request_sender);
    }
    let mut llm_task = tokio::spawn(llm_worker.run(
        llm_request_receiver,
        llm_command_receiver,
        output_sender.clone(),
    ));
    let mut output_task = tokio::spawn(sink.run(output_receiver));

    let mut audio_completed = None;
    let mut stt_completed = None;
    let mut transcript_completed = None;
    let mut candidate_audio_completed = None;
    let mut candidate_stt_completed = None;
    let mut candidate_transcript_completed = None;
    let mut llm_completed = None;
    let mut output_completed = None;

    let mut app_commands_open = true;
    let stop_reason = loop {
        tokio::select! {
            signal = tokio::signal::ctrl_c() => {
                signal.map_err(AppError::Shutdown)?;
                info!(
                    module = "app",
                    event = "shutdown_requested",
                    "shutdown signal received"
                );
                break StopReason::Signal;
            }
            command = app_commands.recv(), if app_commands_open => {
                match command {
                    Some(AppCommand::PauseListening) => {
                        if transcript_command_sender
                            .send(TranscriptCommand::SetPaused(true))
                            .is_err()
                        {
                            break StopReason::Transcript;
                        }
                        if candidate_enabled
                            && candidate_command_sender
                                .send(TranscriptCommand::SetPaused(true))
                                .is_err()
                        {
                            break StopReason::CandidateTranscript;
                        }
                        if output_sender
                            .send(OutputEvent::Status(StatusMessage {
                                kind: StatusKind::Paused,
                                text: "listening paused".to_owned(),
                            }))
                            .is_err()
                        {
                            break StopReason::Output;
                        }
                    }
                    Some(AppCommand::ResumeListening) => {
                        if transcript_command_sender
                            .send(TranscriptCommand::SetPaused(false))
                            .is_err()
                        {
                            break StopReason::Transcript;
                        }
                        if candidate_enabled
                            && candidate_command_sender
                                .send(TranscriptCommand::SetPaused(false))
                                .is_err()
                        {
                            break StopReason::CandidateTranscript;
                        }
                        if output_sender
                            .send(OutputEvent::Status(StatusMessage {
                                kind: StatusKind::Listening,
                                text: "listening resumed".to_owned(),
                            }))
                            .is_err()
                        {
                            break StopReason::Output;
                        }
                    }
                    Some(AppCommand::ToggleLiveCoding) => {
                        active_mode = if active_mode == Mode::LiveCoding {
                            Mode::Voice
                        } else {
                            Mode::LiveCoding
                        };
                        if transcript_command_sender
                            .send(TranscriptCommand::SetMode(active_mode))
                            .is_err()
                        {
                            break StopReason::Transcript;
                        }
                        if candidate_enabled
                            && candidate_command_sender
                                .send(TranscriptCommand::SetMode(active_mode))
                                .is_err()
                        {
                            break StopReason::CandidateTranscript;
                        }
                        if output_sender
                            .send(OutputEvent::ModeChanged { mode: active_mode })
                            .is_err()
                        {
                            break StopReason::Output;
                        }
                        info!(
                            module = "app",
                            event = "mode_changed",
                            mode = %active_mode,
                            "pipeline mode changed"
                        );
                    }
                    Some(AppCommand::ClearHistory) => {
                        if llm_command_sender.send(LlmCommand::ClearHistory).is_err() {
                            break StopReason::Llm;
                        }
                        if output_sender
                            .send(OutputEvent::Status(StatusMessage {
                                kind: StatusKind::HistoryCleared,
                                text: "conversation history cleared".to_owned(),
                            }))
                            .is_err()
                        {
                            break StopReason::Output;
                        }
                    }
                    Some(AppCommand::Shutdown) => {
                        info!(
                            module = "app",
                            event = "shutdown_requested",
                            "shutdown command received"
                        );
                        break StopReason::Command;
                    }
                    None => app_commands_open = false,
                }
            }
            result = &mut audio_task => {
                let completed_file = benchmark_mode && matches!(&result, Ok(Ok(())));
                audio_completed = Some(result);
                if completed_file {
                    info!(
                        module = "app",
                        event = "benchmark_audio_completed",
                        drain_ms = BENCHMARK_EOF_DRAIN.as_millis(),
                        "benchmark audio reached EOF"
                    );
                    tokio::time::sleep(BENCHMARK_EOF_DRAIN).await;
                    break StopReason::BenchmarkEof;
                }
                break StopReason::Audio;
            }
            result = &mut stt_task => {
                stt_completed = Some(result);
                break StopReason::Stt;
            }
            result = &mut transcript_task => {
                transcript_completed = Some(result);
                break StopReason::Transcript;
            }
            result = wait_optional_task(&mut candidate_audio_task) => {
                candidate_audio_completed = Some(result);
                break StopReason::CandidateAudio;
            }
            result = wait_optional_task(&mut candidate_stt_task) => {
                candidate_stt_completed = Some(result);
                break StopReason::CandidateStt;
            }
            result = wait_optional_task(&mut candidate_transcript_task) => {
                candidate_transcript_completed = Some(result);
                break StopReason::CandidateTranscript;
            }
            result = &mut llm_task => {
                llm_completed = Some(result);
                break StopReason::Llm;
            }
            result = &mut output_task => {
                output_completed = Some(result);
                break StopReason::Output;
            }
        }
    };

    if shutdown_sender.send(true).is_err() {
        debug!(
            module = "app",
            event = "shutdown_channels_closed",
            "pipeline workers already closed their shutdown channels"
        );
    }
    drop(frame_sender_guard);
    drop(candidate_frame_sender_guard);

    let shutdown_timeout = if benchmark_mode {
        BENCHMARK_SHUTDOWN_TIMEOUT
    } else {
        SHUTDOWN_TIMEOUT
    };
    let audio_result =
        finish_task("audio", &mut audio_task, audio_completed, shutdown_timeout).await?;
    let stt_result = finish_task("stt", &mut stt_task, stt_completed, shutdown_timeout).await?;
    let transcript_stats = finish_task(
        "transcript",
        &mut transcript_task,
        transcript_completed,
        shutdown_timeout,
    )
    .await?;
    let candidate_audio_result = finish_optional_task(
        "candidate audio",
        &mut candidate_audio_task,
        candidate_audio_completed,
        shutdown_timeout,
    )
    .await?;
    let candidate_stt_result = finish_optional_task(
        "candidate STT",
        &mut candidate_stt_task,
        candidate_stt_completed,
        shutdown_timeout,
    )
    .await?;
    let candidate_transcript_stats = finish_optional_task(
        "candidate transcript",
        &mut candidate_transcript_task,
        candidate_transcript_completed,
        shutdown_timeout,
    )
    .await?;
    let knowledge_stats = finish_knowledge_task(knowledge_task).await;
    let llm_stats = finish_task("LLM", &mut llm_task, llm_completed, shutdown_timeout).await?;
    if output_sender
        .send(OutputEvent::Status(StatusMessage {
            kind: StatusKind::Stopped,
            text: "mague-rc pipeline stopped".to_owned(),
        }))
        .is_err()
    {
        debug!(
            module = "app",
            event = "output_channel_closed",
            "output channel closed before shutdown status"
        );
    }
    drop(output_sender);
    let output_stats = finish_task(
        "output",
        &mut output_task,
        output_completed,
        shutdown_timeout,
    )
    .await?;

    flatten_audio_task(audio_result)?;
    flatten_stt_task(stt_result)?;
    if let Some(result) = candidate_audio_result {
        flatten_audio_task(result)?;
    }
    if let Some(result) = candidate_stt_result {
        flatten_stt_task(result)?;
    }
    let transcript_stats = transcript_stats.map_err(AppError::TranscriptWorker)?;
    let candidate_transcript_stats = candidate_transcript_stats
        .transpose()
        .map_err(AppError::TranscriptWorker)?
        .unwrap_or_default();
    let llm_stats = llm_stats.map_err(AppError::LlmWorker)?;
    let output_stats = output_stats.map_err(|error| AppError::Output(error.to_string()))?;

    if let Some(worker) = stop_reason.unexpected_worker() {
        return Err(AppError::WorkerStopped(worker));
    }

    info!(
        module = "app",
        event = "shutdown_completed",
        transcript_events = transcript_stats.transcripts + candidate_transcript_stats.transcripts,
        final_transcripts = transcript_stats.final_transcripts + candidate_transcript_stats.final_transcripts,
        transcript_chunks = transcript_stats.chunks + candidate_transcript_stats.chunks,
        knowledge_searches = knowledge_stats.searches,
        knowledge_failed = knowledge_stats.failed,
        knowledge_coalesced = knowledge_stats.coalesced,
        llm_requests = llm_stats.requests,
        llm_completed = llm_stats.completed,
        llm_failed = llm_stats.failed,
        answers_completed = output_stats.completed,
        "mague-rc stopped cleanly"
    );
    Ok(())
}

#[derive(Clone, Copy)]
enum StopReason {
    Signal,
    Command,
    BenchmarkEof,
    Audio,
    Stt,
    Transcript,
    CandidateAudio,
    CandidateStt,
    CandidateTranscript,
    Llm,
    Output,
}

impl StopReason {
    fn unexpected_worker(self) -> Option<&'static str> {
        match self {
            Self::Signal | Self::Command | Self::BenchmarkEof => None,
            Self::Audio => Some("audio"),
            Self::Stt => Some("stt"),
            Self::Transcript => Some("transcript"),
            Self::CandidateAudio => Some("candidate audio"),
            Self::CandidateStt => Some("candidate STT"),
            Self::CandidateTranscript => Some("candidate transcript"),
            Self::Llm => Some("LLM"),
            Self::Output => Some("output"),
        }
    }
}

async fn wait_for_stt_readiness(mut readiness: watch::Receiver<bool>) -> Result<(), AudioError> {
    loop {
        if *readiness.borrow_and_update() {
            return Ok(());
        }
        readiness
            .changed()
            .await
            .map_err(|_| AudioError::SttReadinessClosed)?;
    }
}

async fn finish_knowledge_task(
    task: Option<JoinHandle<Result<KnowledgeWorkerStats, crate::knowledge::KnowledgeError>>>,
) -> KnowledgeWorkerStats {
    let Some(task) = task else {
        return KnowledgeWorkerStats::default();
    };
    match task.await {
        Ok(Ok(stats)) => stats,
        Ok(Err(error)) => {
            warn!(
                module = "knowledge",
                event = "worker_failed",
                error = %error,
                "remote knowledge worker failed"
            );
            KnowledgeWorkerStats {
                searches: 0,
                failed: 1,
                coalesced: 0,
            }
        }
        Err(error) => {
            warn!(
                module = "knowledge",
                event = "worker_join_failed",
                error = %error,
                "could not join remote knowledge worker"
            );
            KnowledgeWorkerStats {
                searches: 0,
                failed: 1,
                coalesced: 0,
            }
        }
    }
}

async fn finish_task<T>(
    name: &'static str,
    task: &mut JoinHandle<T>,
    completed: Option<Result<T, JoinError>>,
    shutdown_timeout: Duration,
) -> Result<T, AppError> {
    let result = match completed {
        Some(result) => result,
        None => match timeout(shutdown_timeout, &mut *task).await {
            Ok(result) => result,
            Err(_) => {
                task.abort();
                if let Err(error) = task.await
                    && !error.is_cancelled()
                {
                    warn!(
                        module = "app",
                        event = "worker_abort_failed",
                        worker = name,
                        error = %error,
                        "pipeline worker failed while being aborted"
                    );
                }
                return Err(AppError::WorkerShutdownTimeout(name));
            }
        },
    };

    result.map_err(|error| AppError::WorkerTask {
        worker: name,
        error: error.to_string(),
    })
}

async fn wait_optional_task<T>(task: &mut Option<JoinHandle<T>>) -> Result<T, JoinError> {
    match task {
        Some(task) => task.await,
        None => std::future::pending().await,
    }
}

async fn finish_optional_task<T>(
    name: &'static str,
    task: &mut Option<JoinHandle<T>>,
    completed: Option<Result<T, JoinError>>,
    shutdown_timeout: Duration,
) -> Result<Option<T>, AppError> {
    let Some(task) = task.as_mut() else {
        return Ok(None);
    };
    finish_task(name, task, completed, shutdown_timeout)
        .await
        .map(Some)
}

fn flatten_audio_task(result: Result<(), AudioError>) -> Result<(), AppError> {
    result.map_err(AppError::Audio)
}

fn flatten_stt_task(result: Result<(), SttError>) -> Result<(), AppError> {
    result.map_err(AppError::Stt)
}
