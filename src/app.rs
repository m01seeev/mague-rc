use std::time::Duration;

use thiserror::Error;
use tokio::{
    sync::{mpsc, watch},
    task::{JoinError, JoinHandle},
    time::{Instant, MissedTickBehavior, interval_at, timeout},
};
use tracing::{debug, info, warn};

use crate::{
    audio::{AudioError, AudioSource, FfmpegAudioSource, audio_frame_channel},
    config::{Config, TranscriptConfig},
    control::TerminalEchoGuard,
    error::AppError,
    events::{
        AppErrorView, DeepgramEvent, LlmCommand, LlmRequest, Mode, OutputComponent, OutputEvent,
        QueueKind, QueueState, StatusKind, StatusMessage, SttStatus, TranscriptChunk,
        TranscriptView,
    },
    llm::{
        LlmQueueError, LlmRequestSender, LlmWorker, OpenRouterTextProvider, llm_request_channel,
    },
    output::{OutputSink, TerminalOutputSink},
    stt::{DeepgramSttProvider, SpeechToTextProvider, SttError},
    transcript::TranscriptWindowAssembler,
};

const SHUTDOWN_TIMEOUT: Duration = Duration::from_secs(5);

pub async fn run(config: Config) -> Result<(), AppError> {
    let _terminal_echo_guard = TerminalEchoGuard::hide_control_characters();

    info!(
        module = "app",
        event = "started",
        audio_source = %config.audio.source,
        stt_model = %config.deepgram.model,
        llm_model = %config.llm.model,
        transcript_window_sec = config.transcript.window_sec,
        "mague-rc started; press Ctrl+C to stop"
    );

    let (frame_sender, frame_receiver) = audio_frame_channel(config.audio.queue_max);
    let (stt_event_sender, stt_event_receiver) = mpsc::unbounded_channel();
    let (llm_request_sender, llm_request_receiver) = llm_request_channel(config.llm.queue_max);
    let (output_sender, output_receiver) = mpsc::unbounded_channel::<OutputEvent>();
    let (_llm_command_sender, llm_command_receiver) = mpsc::unbounded_channel::<LlmCommand>();
    let (shutdown_sender, shutdown_receiver) = watch::channel(false);

    send_output(
        &output_sender,
        OutputEvent::Status(StatusMessage {
            kind: StatusKind::Started,
            text: format!(
                "source={} | STT={} | LLM={}",
                config.audio.source, config.deepgram.model, config.llm.model
            ),
        }),
    )?;

    let audio_source = FfmpegAudioSource::new(
        config.audio.clone(),
        config.deepgram.sample_rate,
        config.deepgram.channels,
    );
    let stt_provider = DeepgramSttProvider::new(config.deepgram.clone());
    let llm_provider = OpenRouterTextProvider::new(config.llm.clone())?;
    let llm_worker = LlmWorker::new(llm_provider, config.llm);

    let mut audio_task = tokio::spawn(audio_source.run(frame_sender, shutdown_receiver.clone()));
    let mut stt_task =
        tokio::spawn(stt_provider.run(frame_receiver, stt_event_sender, shutdown_receiver));
    let mut transcript_task = tokio::spawn(run_transcript_windows(
        stt_event_receiver,
        llm_request_sender,
        output_sender.clone(),
        config.transcript,
    ));
    let mut llm_task = tokio::spawn(llm_worker.run(
        llm_request_receiver,
        llm_command_receiver,
        output_sender.clone(),
    ));
    let mut output_task = tokio::spawn(TerminalOutputSink.run(output_receiver));

    let mut audio_completed = None;
    let mut stt_completed = None;
    let mut transcript_completed = None;
    let mut llm_completed = None;
    let mut output_completed = None;

    let stop_reason = tokio::select! {
        signal = tokio::signal::ctrl_c() => {
            signal.map_err(AppError::Shutdown)?;
            info!(
                module = "app",
                event = "shutdown_requested",
                "shutdown signal received"
            );
            StopReason::Signal
        }
        result = &mut audio_task => {
            audio_completed = Some(result);
            StopReason::Audio
        }
        result = &mut stt_task => {
            stt_completed = Some(result);
            StopReason::Stt
        }
        result = &mut transcript_task => {
            transcript_completed = Some(result);
            StopReason::Transcript
        }
        result = &mut llm_task => {
            llm_completed = Some(result);
            StopReason::Llm
        }
        result = &mut output_task => {
            output_completed = Some(result);
            StopReason::Output
        }
    };

    if shutdown_sender.send(true).is_err() {
        debug!(
            module = "app",
            event = "shutdown_channels_closed",
            "pipeline workers already closed their shutdown channels"
        );
    }

    let audio_result = finish_task("audio", &mut audio_task, audio_completed).await?;
    let stt_result = finish_task("stt", &mut stt_task, stt_completed).await?;
    let transcript_stats =
        finish_task("transcript", &mut transcript_task, transcript_completed).await?;
    let llm_stats = finish_task("LLM", &mut llm_task, llm_completed).await?;
    if output_sender
        .send(OutputEvent::Status(StatusMessage {
            kind: StatusKind::Stopped,
            text: "mague-rc stopped cleanly".to_owned(),
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
    let output_stats = finish_task("terminal output", &mut output_task, output_completed).await?;

    flatten_audio_task(audio_result)?;
    flatten_stt_task(stt_result)?;
    let transcript_stats = transcript_stats.map_err(AppError::TranscriptWorker)?;
    let llm_stats = llm_stats.map_err(AppError::LlmWorker)?;
    let output_stats = output_stats.map_err(AppError::TerminalOutput)?;

    if let Some(worker) = stop_reason.unexpected_worker() {
        return Err(AppError::WorkerStopped(worker));
    }

    info!(
        module = "app",
        event = "shutdown_completed",
        transcript_events = transcript_stats.transcripts,
        final_transcripts = transcript_stats.final_transcripts,
        transcript_chunks = transcript_stats.chunks,
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
    Audio,
    Stt,
    Transcript,
    Llm,
    Output,
}

impl StopReason {
    fn unexpected_worker(self) -> Option<&'static str> {
        match self {
            Self::Signal => None,
            Self::Audio => Some("audio"),
            Self::Stt => Some("stt"),
            Self::Transcript => Some("transcript"),
            Self::Llm => Some("LLM"),
            Self::Output => Some("terminal output"),
        }
    }
}

#[derive(Default)]
struct TranscriptStats {
    transcripts: u64,
    final_transcripts: u64,
    chunks: u64,
}

async fn run_transcript_windows(
    mut events: mpsc::UnboundedReceiver<DeepgramEvent>,
    llm_requests: LlmRequestSender,
    output: mpsc::UnboundedSender<OutputEvent>,
    config: TranscriptConfig,
) -> Result<TranscriptStats, TranscriptWorkerError> {
    let mut stats = TranscriptStats::default();
    let window_duration = Duration::from_secs(config.window_sec);
    let mut ticker = interval_at(Instant::now() + window_duration, window_duration);
    ticker.set_missed_tick_behavior(MissedTickBehavior::Delay);
    let mut assembler = TranscriptWindowAssembler::new(config.min_utterance_chars);

    loop {
        tokio::select! {
            _ = ticker.tick() => {
                flush_transcript_window(
                    &mut assembler,
                    &llm_requests,
                    &output,
                    &mut stats,
                    "timer",
                )
                .await?;
            }
            event = events.recv() => match event {
                Some(event) => handle_deepgram_event(event, &mut assembler, &output, &mut stats)?,
                None => break,
            }
        }
    }

    finish_transcript_window(&mut assembler, &llm_requests, &output, &mut stats).await?;
    Ok(stats)
}

fn handle_deepgram_event(
    event: DeepgramEvent,
    assembler: &mut TranscriptWindowAssembler,
    output: &mpsc::UnboundedSender<OutputEvent>,
    stats: &mut TranscriptStats,
) -> Result<(), TranscriptWorkerError> {
    match event {
        DeepgramEvent::Status(status) => handle_stt_status(status, output)?,
        DeepgramEvent::Transcript {
            text,
            is_final,
            speech_final,
        } if !text.is_empty() => {
            stats.transcripts += 1;
            if is_final {
                stats.final_transcripts += 1;
                debug!(
                    module = "stt",
                    event = "transcript_final",
                    speech_final,
                    text = %text,
                    "FINAL transcript"
                );
            } else {
                debug!(
                    module = "stt",
                    event = "transcript_interim",
                    text = %text,
                    "interim transcript"
                );
            }
            assembler.push_transcript(&text, is_final);
        }
        DeepgramEvent::Transcript { .. } => {}
        DeepgramEvent::SpeechStarted => debug!(
            module = "stt",
            event = "speech_started",
            "Deepgram detected speech"
        ),
        DeepgramEvent::UtteranceEnd => debug!(
            module = "stt",
            event = "utterance_end",
            "Deepgram detected utterance end"
        ),
        DeepgramEvent::Metadata => debug!(
            module = "stt",
            event = "metadata",
            "Deepgram metadata received"
        ),
        DeepgramEvent::Error(error) => {
            warn!(
                module = "stt",
                event = "error",
                error = %error,
                "Deepgram error"
            );
            send_output(
                output,
                OutputEvent::Error(AppErrorView {
                    component: OutputComponent::Stt,
                    message: error,
                }),
            )?;
        }
    }
    Ok(())
}

fn handle_stt_status(
    status: SttStatus,
    output: &mpsc::UnboundedSender<OutputEvent>,
) -> Result<(), TranscriptWorkerError> {
    let (kind, text, queue_len) = match status {
        SttStatus::Connecting {
            retry_count,
            queue_len,
        } => (
            StatusKind::Connecting,
            if retry_count == 0 {
                "connecting to Deepgram".to_owned()
            } else {
                format!("connecting to Deepgram (attempt {})", retry_count + 1)
            },
            queue_len,
        ),
        SttStatus::Connected { queue_len } => (
            StatusKind::Listening,
            "Deepgram connected; listening".to_owned(),
            queue_len,
        ),
        SttStatus::Reconnecting {
            retry_count,
            delay_secs,
            queue_len,
        } => (
            StatusKind::Reconnecting,
            format!("Deepgram reconnect in {delay_secs}s (retry {retry_count})"),
            queue_len,
        ),
    };

    send_output(output, OutputEvent::Status(StatusMessage { kind, text }))?;
    send_output(
        output,
        OutputEvent::QueueState(QueueState {
            queue: QueueKind::Audio,
            len: queue_len,
        }),
    )
}

async fn flush_transcript_window(
    assembler: &mut TranscriptWindowAssembler,
    llm_requests: &LlmRequestSender,
    output: &mpsc::UnboundedSender<OutputEvent>,
    stats: &mut TranscriptStats,
    reason: &'static str,
) -> Result<(), TranscriptWorkerError> {
    queue_transcript_chunk(assembler.flush(), llm_requests, output, stats, reason).await
}

async fn finish_transcript_window(
    assembler: &mut TranscriptWindowAssembler,
    llm_requests: &LlmRequestSender,
    output: &mpsc::UnboundedSender<OutputEvent>,
    stats: &mut TranscriptStats,
) -> Result<(), TranscriptWorkerError> {
    queue_transcript_chunk(assembler.finish(), llm_requests, output, stats, "shutdown").await
}

async fn queue_transcript_chunk(
    chunk: Option<TranscriptChunk>,
    llm_requests: &LlmRequestSender,
    output: &mpsc::UnboundedSender<OutputEvent>,
    stats: &mut TranscriptStats,
    reason: &'static str,
) -> Result<(), TranscriptWorkerError> {
    let Some(chunk) = chunk else {
        return Ok(());
    };

    let request_id = chunk.sequence;
    info!(
        module = "transcript",
        event = "window_flushed",
        sequence = request_id,
        reason,
        text = %chunk.text,
        "TRANSCRIPT WINDOW"
    );
    send_output(
        output,
        OutputEvent::Transcript(TranscriptView {
            sequence: request_id,
            text: chunk.text.clone(),
        }),
    )?;
    llm_requests
        .send(LlmRequest {
            request_id,
            mode: Mode::Voice,
            text: chunk.text,
        })
        .await?;
    stats.chunks += 1;

    let queue_len = llm_requests.len();
    send_output(
        output,
        OutputEvent::QueueState(QueueState {
            queue: QueueKind::Llm,
            len: queue_len,
        }),
    )?;
    if queue_len > 1 {
        warn!(
            module = "llm",
            event = "queue_growing",
            queue_len,
            "LLM request queue is growing"
        );
    }
    Ok(())
}

fn send_output(
    output: &mpsc::UnboundedSender<OutputEvent>,
    event: OutputEvent,
) -> Result<(), TranscriptWorkerError> {
    output
        .send(event)
        .map_err(|_| TranscriptWorkerError::OutputChannelClosed)
}

#[derive(Debug, Error)]
pub enum TranscriptWorkerError {
    #[error(transparent)]
    LlmQueue(#[from] LlmQueueError),

    #[error("output channel closed")]
    OutputChannelClosed,
}

async fn finish_task<T>(
    name: &'static str,
    task: &mut JoinHandle<T>,
    completed: Option<Result<T, JoinError>>,
) -> Result<T, AppError> {
    let result = match completed {
        Some(result) => result,
        None => match timeout(SHUTDOWN_TIMEOUT, &mut *task).await {
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

fn flatten_audio_task(result: Result<(), AudioError>) -> Result<(), AppError> {
    result.map_err(AppError::Audio)
}

fn flatten_stt_task(result: Result<(), SttError>) -> Result<(), AppError> {
    result.map_err(AppError::Stt)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn shutdown_flush_queues_a_voice_request() {
        let (event_sender, event_receiver) = mpsc::unbounded_channel();
        let (request_sender, mut request_receiver) = llm_request_channel(0);
        let (output_sender, mut output_receiver) = mpsc::unbounded_channel();
        event_sender
            .send(DeepgramEvent::Transcript {
                text: "Что такое HashMap?".to_owned(),
                is_final: true,
                speech_final: true,
            })
            .expect("STT event must send");
        drop(event_sender);

        let stats = run_transcript_windows(
            event_receiver,
            request_sender,
            output_sender,
            TranscriptConfig {
                window_sec: 60,
                min_utterance_chars: 3,
            },
        )
        .await
        .expect("transcript worker must stop cleanly");
        let request = request_receiver
            .recv()
            .await
            .expect("shutdown flush must queue a request");

        assert_eq!(stats.chunks, 1);
        assert_eq!(request.request_id, 0);
        assert_eq!(request.mode, Mode::Voice);
        assert_eq!(request.text, "Что такое HashMap?");
        assert!(matches!(
            output_receiver.recv().await,
            Some(OutputEvent::Transcript(TranscriptView {
                sequence: 0,
                text,
            })) if text == "Что такое HashMap?"
        ));
    }

    #[test]
    fn maps_reconnect_status_and_audio_queue_for_output_sink() {
        let (output_sender, mut output_receiver) = mpsc::unbounded_channel();

        handle_stt_status(
            SttStatus::Reconnecting {
                retry_count: 3,
                delay_secs: 4,
                queue_len: 17,
            },
            &output_sender,
        )
        .expect("status must be forwarded");

        assert!(matches!(
            output_receiver.try_recv(),
            Ok(OutputEvent::Status(StatusMessage {
                kind: StatusKind::Reconnecting,
                text,
            })) if text == "Deepgram reconnect in 4s (retry 3)"
        ));
        assert_eq!(
            output_receiver.try_recv(),
            Ok(OutputEvent::QueueState(QueueState {
                queue: QueueKind::Audio,
                len: 17,
            }))
        );
    }
}
