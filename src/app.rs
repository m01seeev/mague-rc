use std::time::Duration;

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
    events::DeepgramEvent,
    stt::{DeepgramSttProvider, SpeechToTextProvider, SttError},
};

const SHUTDOWN_TIMEOUT: Duration = Duration::from_secs(5);

pub async fn run(config: Config) -> Result<(), AppError> {
    let _terminal_echo_guard = TerminalEchoGuard::hide_control_characters();

    info!(
        module = "app",
        event = "started",
        audio_source = %config.audio.source,
        stt_model = %config.deepgram.model,
        "mague-rc started; press Ctrl+C to stop"
    );

    let (frame_sender, frame_receiver) = audio_frame_channel(config.audio.queue_max);
    let (event_sender, event_receiver) = mpsc::unbounded_channel();
    let (shutdown_sender, shutdown_receiver) = watch::channel(false);

    let audio_source = FfmpegAudioSource::new(
        config.audio.clone(),
        config.deepgram.sample_rate,
        config.deepgram.channels,
    );
    let stt_provider = DeepgramSttProvider::new(config.deepgram.clone());

    let mut audio_task = tokio::spawn(audio_source.run(frame_sender, shutdown_receiver.clone()));
    let mut stt_task =
        tokio::spawn(stt_provider.run(frame_receiver, event_sender, shutdown_receiver));
    let mut event_task = tokio::spawn(print_deepgram_events(event_receiver));

    let mut audio_completed = None;
    let mut stt_completed = None;
    let mut event_completed = None;

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
        result = &mut event_task => {
            event_completed = Some(result);
            StopReason::Events
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
    let event_stats = finish_task("STT output", &mut event_task, event_completed).await?;

    match stop_reason {
        StopReason::Signal => {
            flatten_audio_task(audio_result)?;
            flatten_stt_task(stt_result)?;
        }
        StopReason::Audio => {
            flatten_audio_task(audio_result)?;
            flatten_stt_task(stt_result)?;
            return Err(AppError::WorkerStopped("audio"));
        }
        StopReason::Stt => {
            flatten_stt_task(stt_result)?;
            flatten_audio_task(audio_result)?;
            return Err(AppError::WorkerStopped("stt"));
        }
        StopReason::Events => {
            flatten_audio_task(audio_result)?;
            flatten_stt_task(stt_result)?;
            return Err(AppError::WorkerStopped("STT output"));
        }
    }

    info!(
        module = "app",
        event = "shutdown_completed",
        transcript_events = event_stats.transcripts,
        final_transcripts = event_stats.final_transcripts,
        "mague-rc stopped cleanly"
    );
    Ok(())
}

#[derive(Clone, Copy)]
enum StopReason {
    Signal,
    Audio,
    Stt,
    Events,
}

#[derive(Default)]
struct TranscriptStats {
    transcripts: u64,
    final_transcripts: u64,
}

async fn print_deepgram_events(
    mut events: mpsc::UnboundedReceiver<DeepgramEvent>,
) -> TranscriptStats {
    let mut stats = TranscriptStats::default();

    while let Some(event) = events.recv().await {
        match event {
            DeepgramEvent::Transcript {
                text,
                is_final,
                speech_final,
            } if !text.is_empty() => {
                stats.transcripts += 1;
                if is_final {
                    stats.final_transcripts += 1;
                    info!(
                        module = "stt",
                        event = "transcript_final",
                        speech_final,
                        text = %text,
                        "FINAL transcript"
                    );
                } else {
                    info!(
                        module = "stt",
                        event = "transcript_interim",
                        text = %text,
                        "interim transcript"
                    );
                }
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
            DeepgramEvent::Error(error) => warn!(
                module = "stt",
                event = "error",
                error = %error,
                "Deepgram error"
            ),
        }
    }

    stats
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
