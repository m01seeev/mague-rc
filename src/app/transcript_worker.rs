use std::time::Duration;

use thiserror::Error;
use tokio::{
    sync::{mpsc, watch},
    time::{Instant, sleep},
};
use tracing::{debug, info, warn};

use crate::{
    config::TranscriptConfig,
    events::{
        AppErrorView, DeepgramEvent, LlmRequest, Mode, OutputComponent, OutputEvent, QueueKind,
        QueueState, RetrievalView, Speaker, StatusKind, StatusMessage, SttObservation, SttStatus,
        TranscriptChunk, TranscriptView,
    },
    llm::{LlmQueueError, LlmRequestSender},
    transcript::TranscriptWindowAssembler,
};

use super::{boundary::boundary_deferral, retrieval::RetrievalPipeline};

#[cfg(test)]
use super::boundary::BoundaryDeferral;

const CANDIDATE_REQUEST_ID_BIT: u64 = 1 << 63;

#[derive(Default)]
pub(super) struct TranscriptStats {
    pub(super) transcripts: u64,
    pub(super) final_transcripts: u64,
    pub(super) chunks: u64,
}

#[derive(Clone, Copy)]
pub(super) enum TranscriptCommand {
    SetPaused(bool),
    SetMode(Mode),
}

#[cfg(test)]
async fn run_transcript_windows(
    events: mpsc::UnboundedReceiver<DeepgramEvent>,
    llm_requests: LlmRequestSender,
    output: mpsc::UnboundedSender<OutputEvent>,
    commands: mpsc::UnboundedReceiver<TranscriptCommand>,
    config: TranscriptConfig,
    readiness: Option<watch::Receiver<bool>>,
) -> Result<TranscriptStats, TranscriptWorkerError> {
    run_transcript_windows_with_retrieval(
        events,
        llm_requests,
        output,
        commands,
        config,
        readiness,
        Speaker::Interviewer,
        None,
    )
    .await
}

pub(super) async fn run_transcript_windows_with_retrieval(
    mut events: mpsc::UnboundedReceiver<DeepgramEvent>,
    llm_requests: LlmRequestSender,
    output: mpsc::UnboundedSender<OutputEvent>,
    mut commands: mpsc::UnboundedReceiver<TranscriptCommand>,
    config: TranscriptConfig,
    readiness: Option<watch::Receiver<bool>>,
    speaker: Speaker,
    mut retrieval: Option<RetrievalPipeline>,
) -> Result<TranscriptStats, TranscriptWorkerError> {
    if let Some(readiness) = readiness {
        wait_for_transcript_readiness(readiness).await?;
    }
    if let Some(retrieval) = retrieval.as_mut() {
        retrieval.wait_until_ready().await;
    }
    let mut stats = TranscriptStats::default();
    let fallback_duration = Duration::from_secs(config.window_sec);
    let fallback = sleep(fallback_duration);
    tokio::pin!(fallback);
    let mut fallback_armed = false;
    let mut assembler = TranscriptWindowAssembler::new(config.min_utterance_chars);
    let mut pending_last_word_end_ms = None;
    let mut interim_fallback_deferred = false;
    let mut paused = false;
    let mut mode = Mode::Voice;
    let mut commands_open = true;

    loop {
        tokio::select! {
            biased;
            command = commands.recv(), if commands_open => {
                match command {
                    Some(TranscriptCommand::SetPaused(next_paused)) => {
                        paused = next_paused;
                        if paused {
                            assembler.discard_pending();
                            pending_last_word_end_ms = None;
                            interim_fallback_deferred = false;
                            fallback_armed = false;
                            if let Some(retrieval) = retrieval.as_mut() {
                                retrieval.reset();
                            }
                            send_transcript_draft(&assembler, &output, speaker)?;
                        }
                    }
                    Some(TranscriptCommand::SetMode(next_mode)) => {
                        mode = next_mode;
                        assembler.discard_pending();
                        pending_last_word_end_ms = None;
                        interim_fallback_deferred = false;
                        fallback_armed = false;
                        if let Some(retrieval) = retrieval.as_mut() {
                            retrieval.reset();
                        }
                        send_transcript_draft(&assembler, &output, speaker)?;
                    }
                    None => commands_open = false,
                }
            }
            _ = &mut fallback, if fallback_armed && !paused => {
                if assembler.has_unfinalized_interim() && !interim_fallback_deferred {
                    debug!(
                        module = "transcript",
                        event = "inactivity_deferred",
                        "waiting one more window for an unfinalized interim transcript"
                    );
                    fallback
                        .as_mut()
                        .reset(Instant::now() + fallback_duration);
                    interim_fallback_deferred = true;
                } else {
                    flush_transcript_utterance(
                        &mut assembler,
                        &llm_requests,
                        &output,
                        &mut stats,
                        "inactivity_timeout",
                        mode,
                        speaker,
                        &mut retrieval,
                    )
                    .await?;
                    pending_last_word_end_ms = None;
                    interim_fallback_deferred = false;
                    fallback_armed = false;
                }
            }
            event = events.recv() => match event {
                Some(DeepgramEvent::Transcript { .. }) if paused => {}
                Some(event) => {
                    let has_transcript = matches!(
                        &event,
                        DeepgramEvent::Transcript { text, .. } if !text.is_empty()
                    );
                    let force_retrieval = matches!(
                        &event,
                        DeepgramEvent::Transcript {
                            is_final: true,
                            ..
                        }
                    );
                    let flush_reason =
                        handle_deepgram_event(
                            event,
                            &mut assembler,
                            &mut pending_last_word_end_ms,
                            &output,
                            &mut stats,
                            speaker,
                        )?;

                    if has_transcript {
                        fallback
                            .as_mut()
                            .reset(Instant::now() + fallback_duration);
                        interim_fallback_deferred = false;
                        fallback_armed = true;
                        if mode == Mode::Voice
                            && speaker == Speaker::Interviewer
                            && let Some(retrieval) = retrieval.as_mut()
                        {
                            retrieval.prefetch(&assembler.preview(), force_retrieval);
                        }
                    }
                    if let Some(reason) = flush_reason {
                        flush_transcript_utterance(
                            &mut assembler,
                            &llm_requests,
                            &output,
                            &mut stats,
                            reason,
                            mode,
                            speaker,
                            &mut retrieval,
                        )
                        .await?;
                        pending_last_word_end_ms = None;
                        interim_fallback_deferred = false;
                        fallback_armed = false;
                    }
                }
                None => break,
            }
        }
    }

    finish_transcript_window(
        &mut assembler,
        &llm_requests,
        &output,
        &mut stats,
        mode,
        speaker,
        &mut retrieval,
    )
    .await?;
    Ok(stats)
}

fn handle_deepgram_event(
    event: DeepgramEvent,
    assembler: &mut TranscriptWindowAssembler,
    pending_last_word_end_ms: &mut Option<u64>,
    output: &mpsc::UnboundedSender<OutputEvent>,
    stats: &mut TranscriptStats,
    speaker: Speaker,
) -> Result<Option<&'static str>, TranscriptWorkerError> {
    let mut flush_reason = None;
    match event {
        DeepgramEvent::Status(status) => handle_stt_status(status, output, speaker)?,
        DeepgramEvent::AudioStreamStarted => {
            send_output(
                output,
                OutputEvent::SttObservation {
                    speaker,
                    observation: SttObservation::AudioStreamStarted,
                },
            )?;
        }
        DeepgramEvent::Transcript {
            text,
            is_final,
            speech_final,
            audio_start_ms,
            audio_duration_ms,
            last_word_end_ms,
        } if !text.is_empty() => {
            stats.transcripts += 1;
            assembler.push_transcript(&text, is_final);
            let boundary_deferral = (is_final && speech_final)
                .then(|| assembler.preview())
                .and_then(|preview| boundary_deferral(&preview));
            let speech_final_deferred = boundary_deferral.is_some();
            send_output(
                output,
                OutputEvent::SttObservation {
                    speaker,
                    observation: SttObservation::Transcript {
                        text: text.clone(),
                        is_final,
                        speech_final,
                        speech_final_deferred,
                        audio_start_ms,
                        audio_duration_ms,
                        last_word_end_ms,
                    },
                },
            )?;
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
            if let Some(last_word_end_ms) = last_word_end_ms {
                *pending_last_word_end_ms = Some(last_word_end_ms);
            }
            send_transcript_draft(assembler, output, speaker)?;
            if is_final && speech_final {
                if let Some(deferral) = boundary_deferral {
                    debug!(
                        module = "transcript",
                        event = "speech_final_deferred",
                        reason = deferral.reason(),
                        "waiting for more speech after an incomplete boundary"
                    );
                } else {
                    flush_reason = Some("speech_final");
                }
            }
        }
        DeepgramEvent::Transcript { .. } => {}
        DeepgramEvent::SpeechStarted { audio_timestamp_ms } => {
            send_output(
                output,
                OutputEvent::SttObservation {
                    speaker,
                    observation: SttObservation::SpeechStarted { audio_timestamp_ms },
                },
            )?;
            debug!(
                module = "stt",
                event = "speech_started",
                "Deepgram detected speech"
            );
        }
        DeepgramEvent::UtteranceEnd { last_word_end_ms } => {
            let ignored = last_word_end_ms.zip(*pending_last_word_end_ms).is_some_and(
                |(utterance_end_ms, pending_word_end_ms)| utterance_end_ms < pending_word_end_ms,
            );
            let boundary_deferral = (!ignored)
                .then(|| assembler.preview())
                .and_then(|preview| boundary_deferral(&preview));
            let deferred = boundary_deferral.is_some();
            send_output(
                output,
                OutputEvent::SttObservation {
                    speaker,
                    observation: SttObservation::UtteranceEnd {
                        last_word_end_ms,
                        ignored,
                        deferred,
                    },
                },
            )?;
            if ignored {
                debug!(
                    module = "stt",
                    event = "stale_utterance_end",
                    ?last_word_end_ms,
                    ?pending_last_word_end_ms,
                    "ignored utterance end for earlier speech"
                );
            } else if let Some(deferral) = boundary_deferral {
                debug!(
                    module = "transcript",
                    event = "utterance_end_deferred",
                    reason = deferral.reason(),
                    "keeping an incomplete utterance for the next speech"
                );
            } else {
                debug!(
                    module = "stt",
                    event = "utterance_end",
                    "Deepgram detected utterance end"
                );
                flush_reason = Some("utterance_end");
            }
        }
        DeepgramEvent::Metadata => debug!(
            module = "stt",
            event = "metadata",
            "Deepgram metadata received"
        ),
        DeepgramEvent::Error(error) => {
            warn!(
                module = "stt",
                event = "error",
                speaker = %speaker,
                error = %error,
                "Deepgram error"
            );
            send_output(
                output,
                OutputEvent::Error(AppErrorView {
                    component: OutputComponent::Stt,
                    message: format!("{speaker}: {error}"),
                }),
            )?;
        }
    }
    Ok(flush_reason)
}

fn handle_stt_status(
    status: SttStatus,
    output: &mpsc::UnboundedSender<OutputEvent>,
    speaker: Speaker,
) -> Result<(), TranscriptWorkerError> {
    let (kind, text, queue_len) = match status {
        SttStatus::Connecting {
            retry_count,
            queue_len,
        } => (
            StatusKind::Connecting,
            if retry_count == 0 {
                format!("connecting {speaker} speech to Deepgram")
            } else {
                format!(
                    "connecting {speaker} speech to Deepgram (attempt {})",
                    retry_count + 1
                )
            },
            queue_len,
        ),
        SttStatus::Connected { queue_len } => (
            StatusKind::Listening,
            format!("Deepgram connected; listening to {speaker}"),
            queue_len,
        ),
        SttStatus::Reconnecting {
            retry_count,
            delay_secs,
            queue_len,
        } => (
            StatusKind::Reconnecting,
            format!("Deepgram {speaker} reconnect in {delay_secs}s (retry {retry_count})"),
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

async fn flush_transcript_utterance(
    assembler: &mut TranscriptWindowAssembler,
    llm_requests: &LlmRequestSender,
    output: &mpsc::UnboundedSender<OutputEvent>,
    stats: &mut TranscriptStats,
    reason: &'static str,
    mode: Mode,
    speaker: Speaker,
    retrieval: &mut Option<RetrievalPipeline>,
) -> Result<(), TranscriptWorkerError> {
    queue_transcript_chunk(
        assembler.finish(),
        llm_requests,
        output,
        stats,
        reason,
        mode,
        speaker,
        retrieval,
    )
    .await?;
    send_transcript_draft(assembler, output, speaker)
}

fn send_transcript_draft(
    assembler: &TranscriptWindowAssembler,
    output: &mpsc::UnboundedSender<OutputEvent>,
    speaker: Speaker,
) -> Result<(), TranscriptWorkerError> {
    send_output(
        output,
        OutputEvent::TranscriptDraft {
            speaker,
            text: assembler.preview(),
        },
    )
}

async fn finish_transcript_window(
    assembler: &mut TranscriptWindowAssembler,
    llm_requests: &LlmRequestSender,
    output: &mpsc::UnboundedSender<OutputEvent>,
    stats: &mut TranscriptStats,
    mode: Mode,
    speaker: Speaker,
    retrieval: &mut Option<RetrievalPipeline>,
) -> Result<(), TranscriptWorkerError> {
    queue_transcript_chunk(
        assembler.finish(),
        llm_requests,
        output,
        stats,
        "shutdown",
        mode,
        speaker,
        retrieval,
    )
    .await
}

async fn queue_transcript_chunk(
    chunk: Option<TranscriptChunk>,
    llm_requests: &LlmRequestSender,
    output: &mpsc::UnboundedSender<OutputEvent>,
    stats: &mut TranscriptStats,
    reason: &'static str,
    mode: Mode,
    speaker: Speaker,
    retrieval: &mut Option<RetrievalPipeline>,
) -> Result<(), TranscriptWorkerError> {
    let Some(chunk) = chunk else {
        return Ok(());
    };

    let request_id = if speaker == Speaker::Candidate {
        CANDIDATE_REQUEST_ID_BIT | chunk.sequence
    } else {
        chunk.sequence
    };
    info!(
        module = "transcript",
        event = "utterance_flushed",
        sequence = request_id,
        speaker = %speaker,
        reason,
        text = %chunk.text,
        "TRANSCRIPT UTTERANCE"
    );
    send_output(
        output,
        OutputEvent::Transcript(TranscriptView {
            sequence: request_id,
            mode,
            speaker,
            text: chunk.text.clone(),
            flush_reason: reason.to_owned(),
        }),
    )?;
    let knowledge = if mode == Mode::Voice
        && speaker == Speaker::Interviewer
        && let Some(retrieval) = retrieval.as_mut()
    {
        match retrieval.resolve(&chunk.text).await {
            Ok(context) => {
                info!(
                    module = "knowledge",
                    event = "retrieval_completed",
                    request_id,
                    searches = context.searches,
                    embedding_calls = context.embedding_calls,
                    embedding_tokens = context.embedding_total_tokens,
                    snippets = context.snippets.len(),
                    embedding_ms = context.embedding_ms,
                    search_ms = context.search_ms,
                    final_wait_ms = context.final_wait_ms,
                    "knowledge retrieval completed"
                );
                send_output(
                    output,
                    OutputEvent::Retrieval(RetrievalView {
                        request_id,
                        context: context.clone(),
                    }),
                )?;
                (!context.snippets.is_empty()).then_some(context)
            }
            Err(error) => {
                warn!(
                    module = "knowledge",
                    event = "search_failed",
                    request_id,
                    error = %error,
                    "knowledge retrieval failed"
                );
                send_output(
                    output,
                    OutputEvent::Error(AppErrorView {
                        component: OutputComponent::Knowledge,
                        message: error,
                    }),
                )?;
                None
            }
        }
    } else {
        None
    };
    send_output(output, OutputEvent::LlmQueued { request_id })?;
    llm_requests
        .send(LlmRequest {
            request_id,
            mode,
            speaker,
            text: chunk.text,
            knowledge,
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
    if let Some(retrieval) = retrieval.as_mut() {
        retrieval.reset();
    }
    Ok(())
}

pub(super) fn send_output(
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

    #[error("STT worker stopped before benchmark transcript timing could start")]
    SttReadinessClosed,
}

async fn wait_for_transcript_readiness(
    mut readiness: watch::Receiver<bool>,
) -> Result<(), TranscriptWorkerError> {
    loop {
        if *readiness.borrow_and_update() {
            return Ok(());
        }
        readiness
            .changed()
            .await
            .map_err(|_| TranscriptWorkerError::SttReadinessClosed)?;
    }
}

#[cfg(test)]
mod tests {
    use std::path::PathBuf;

    use tokio::time::timeout;

    use crate::{
        config::KnowledgeConfig, knowledge::KnowledgeSearchResult, llm::llm_request_channel,
    };

    use super::*;

    #[tokio::test]
    async fn speech_final_queues_a_voice_request_immediately() {
        let (event_sender, event_receiver) = mpsc::unbounded_channel();
        let (request_sender, mut request_receiver) = llm_request_channel(0);
        let (output_sender, mut output_receiver) = mpsc::unbounded_channel();
        let (_command_sender, command_receiver) = mpsc::unbounded_channel();
        event_sender
            .send(DeepgramEvent::Transcript {
                text: "Что такое HashMap?".to_owned(),
                is_final: true,
                speech_final: true,
                audio_start_ms: Some(100),
                audio_duration_ms: Some(900),
                last_word_end_ms: Some(900),
            })
            .expect("STT event must send");
        drop(event_sender);

        let stats = run_transcript_windows(
            event_receiver,
            request_sender,
            output_sender,
            command_receiver,
            TranscriptConfig {
                window_sec: 60,
                min_utterance_chars: 3,
            },
            None,
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
            Some(OutputEvent::SttObservation {
                speaker: Speaker::Interviewer,
                observation: SttObservation::Transcript {
                    text,
                    is_final: true,
                    speech_final: true,
                    speech_final_deferred: false,
                    audio_start_ms: Some(100),
                    audio_duration_ms: Some(900),
                    last_word_end_ms: Some(900),
                },
            }) if text == "Что такое HashMap?"
        ));
        assert!(matches!(
            output_receiver.recv().await,
            Some(OutputEvent::TranscriptDraft { text, .. }) if text == "Что такое HashMap?"
        ));
        assert!(matches!(
            output_receiver.recv().await,
            Some(OutputEvent::Transcript(TranscriptView {
                sequence: 0,
                mode: Mode::Voice,
                text,
                flush_reason,
                ..
            })) if text == "Что такое HashMap?" && flush_reason == "speech_final"
        ));
    }

    #[tokio::test]
    async fn dangling_text_survives_utterance_end_and_joins_continuation() {
        let (event_sender, event_receiver) = mpsc::unbounded_channel();
        let (request_sender, mut request_receiver) = llm_request_channel(0);
        let (output_sender, mut output_receiver) = mpsc::unbounded_channel();
        let (_command_sender, command_receiver) = mpsc::unbounded_channel();

        event_sender
            .send(DeepgramEvent::Transcript {
                text: "Расскажите, за счет чего".to_owned(),
                is_final: true,
                speech_final: true,
                audio_start_ms: Some(100),
                audio_duration_ms: Some(900),
                last_word_end_ms: Some(900),
            })
            .expect("dangling speech final must send");
        event_sender
            .send(DeepgramEvent::UtteranceEnd {
                last_word_end_ms: Some(900),
            })
            .expect("utterance end must send");
        event_sender
            .send(DeepgramEvent::Transcript {
                text: "ConcurrentHashMap обеспечивает потокобезопасность?".to_owned(),
                is_final: true,
                speech_final: true,
                audio_start_ms: Some(2_000),
                audio_duration_ms: Some(1_000),
                last_word_end_ms: Some(2_900),
            })
            .expect("continuation must send");
        drop(event_sender);

        let stats = run_transcript_windows(
            event_receiver,
            request_sender,
            output_sender,
            command_receiver,
            TranscriptConfig {
                window_sec: 60,
                min_utterance_chars: 3,
            },
            None,
        )
        .await
        .expect("transcript worker must stop cleanly");

        assert_eq!(stats.chunks, 1);
        assert_eq!(
            request_receiver
                .recv()
                .await
                .expect("continuation must queue the completed question")
                .text,
            "Расскажите, за счет чего ConcurrentHashMap обеспечивает потокобезопасность?"
        );
        assert_eq!(request_receiver.recv().await, None);

        let mut deferred = false;
        let mut utterance_end_deferred = false;
        let mut flush_reason = None;
        while let Ok(event) = output_receiver.try_recv() {
            match event {
                OutputEvent::SttObservation {
                    observation:
                        SttObservation::Transcript {
                            speech_final_deferred,
                            ..
                        },
                    ..
                } => deferred |= speech_final_deferred,
                OutputEvent::SttObservation {
                    observation: SttObservation::UtteranceEnd { deferred, .. },
                    ..
                } => utterance_end_deferred |= deferred,
                OutputEvent::Transcript(transcript) => {
                    flush_reason = Some(transcript.flush_reason);
                }
                _ => {}
            }
        }
        assert!(deferred);
        assert!(utterance_end_deferred);
        assert_eq!(flush_reason.as_deref(), Some("speech_final"));
    }

    #[tokio::test]
    async fn continued_speech_is_joined_after_dangling_speech_final() {
        let (event_sender, event_receiver) = mpsc::unbounded_channel();
        let (request_sender, mut request_receiver) = llm_request_channel(0);
        let (output_sender, _output_receiver) = mpsc::unbounded_channel();
        let (_command_sender, command_receiver) = mpsc::unbounded_channel();

        for event in [
            DeepgramEvent::Transcript {
                text: "Расскажите о".to_owned(),
                is_final: true,
                speech_final: true,
                audio_start_ms: Some(100),
                audio_duration_ms: Some(900),
                last_word_end_ms: Some(900),
            },
            DeepgramEvent::Transcript {
                text: "ConcurrentHashMap?".to_owned(),
                is_final: true,
                speech_final: true,
                audio_start_ms: Some(1_000),
                audio_duration_ms: Some(800),
                last_word_end_ms: Some(1_700),
            },
        ] {
            event_sender.send(event).expect("STT event must send");
        }
        drop(event_sender);

        let stats = run_transcript_windows(
            event_receiver,
            request_sender,
            output_sender,
            command_receiver,
            TranscriptConfig {
                window_sec: 60,
                min_utterance_chars: 3,
            },
            None,
        )
        .await
        .expect("transcript worker must stop cleanly");

        assert_eq!(stats.chunks, 1);
        let request = request_receiver
            .recv()
            .await
            .expect("continued question must be queued");
        assert_eq!(request.text, "Расскажите о ConcurrentHashMap?");
        assert_eq!(request_receiver.recv().await, None);
    }

    #[tokio::test]
    async fn utterance_end_flushes_accumulated_final_transcripts_once() {
        let (event_sender, event_receiver) = mpsc::unbounded_channel();
        let (request_sender, mut request_receiver) = llm_request_channel(0);
        let (output_sender, mut output_receiver) = mpsc::unbounded_channel();
        let (_command_sender, command_receiver) = mpsc::unbounded_channel();

        event_sender
            .send(DeepgramEvent::Transcript {
                text: "Чем HashMap отличается".to_owned(),
                is_final: true,
                speech_final: false,
                audio_start_ms: None,
                audio_duration_ms: None,
                last_word_end_ms: None,
            })
            .expect("first final transcript must send");
        event_sender
            .send(DeepgramEvent::Transcript {
                text: "от ConcurrentHashMap?".to_owned(),
                is_final: true,
                speech_final: false,
                audio_start_ms: None,
                audio_duration_ms: None,
                last_word_end_ms: None,
            })
            .expect("second final transcript must send");
        event_sender
            .send(DeepgramEvent::UtteranceEnd {
                last_word_end_ms: Some(2_000),
            })
            .expect("utterance end must send");
        drop(event_sender);

        let stats = run_transcript_windows(
            event_receiver,
            request_sender,
            output_sender,
            command_receiver,
            TranscriptConfig {
                window_sec: 60,
                min_utterance_chars: 3,
            },
            None,
        )
        .await
        .expect("transcript worker must stop cleanly");

        assert_eq!(stats.chunks, 1);
        assert_eq!(
            request_receiver
                .recv()
                .await
                .expect("utterance end must queue a request")
                .text,
            "Чем HashMap отличается от ConcurrentHashMap?"
        );
        assert_eq!(request_receiver.recv().await, None);

        let mut submitted = None;
        while let Ok(event) = output_receiver.try_recv() {
            if let OutputEvent::Transcript(transcript) = event {
                submitted = Some(transcript);
            }
        }
        assert!(matches!(
            submitted,
            Some(TranscriptView {
                text,
                flush_reason,
                ..
            }) if text == "Чем HashMap отличается от ConcurrentHashMap?"
                && flush_reason == "utterance_end"
        ));
    }

    #[tokio::test]
    async fn inactivity_timeout_flushes_final_text() {
        let (event_sender, event_receiver) = mpsc::unbounded_channel();
        let (request_sender, mut request_receiver) = llm_request_channel(0);
        let (output_sender, _output_receiver) = mpsc::unbounded_channel();
        let (_command_sender, command_receiver) = mpsc::unbounded_channel();

        event_sender
            .send(DeepgramEvent::Transcript {
                text: "незавершенная фраза".to_owned(),
                is_final: true,
                speech_final: false,
                audio_start_ms: None,
                audio_duration_ms: None,
                last_word_end_ms: None,
            })
            .expect("interim transcript must send");

        let worker = tokio::spawn(run_transcript_windows(
            event_receiver,
            request_sender,
            output_sender,
            command_receiver,
            TranscriptConfig {
                window_sec: 1,
                min_utterance_chars: 3,
            },
            None,
        ));

        let request = timeout(Duration::from_secs(2), request_receiver.recv())
            .await
            .expect("fallback must fire after inactivity")
            .expect("fallback must queue a request");
        assert_eq!(request.text, "незавершенная фраза");

        drop(event_sender);
        let stats = worker
            .await
            .expect("transcript task must join")
            .expect("transcript worker must stop cleanly");
        assert_eq!(stats.chunks, 1);
    }

    #[tokio::test]
    async fn inactivity_timeout_defers_unfinalized_interim_once() {
        let (event_sender, event_receiver) = mpsc::unbounded_channel();
        let (request_sender, mut request_receiver) = llm_request_channel(0);
        let (output_sender, _output_receiver) = mpsc::unbounded_channel();
        let (_command_sender, command_receiver) = mpsc::unbounded_channel();

        event_sender
            .send(DeepgramEvent::Transcript {
                text: "незавершенная фраза".to_owned(),
                is_final: false,
                speech_final: false,
                audio_start_ms: None,
                audio_duration_ms: None,
                last_word_end_ms: None,
            })
            .expect("interim transcript must send");

        let worker = tokio::spawn(run_transcript_windows(
            event_receiver,
            request_sender,
            output_sender,
            command_receiver,
            TranscriptConfig {
                window_sec: 1,
                min_utterance_chars: 3,
            },
            None,
        ));

        assert!(
            timeout(Duration::from_millis(1_500), request_receiver.recv())
                .await
                .is_err(),
            "first inactivity window must not flush an interim transcript"
        );
        let request = timeout(Duration::from_secs(1), request_receiver.recv())
            .await
            .expect("second inactivity window must fire")
            .expect("fallback must queue a request");
        assert_eq!(request.text, "незавершенная фраза");

        drop(event_sender);
        let stats = worker
            .await
            .expect("transcript task must join")
            .expect("transcript worker must stop cleanly");
        assert_eq!(stats.chunks, 1);
    }

    #[tokio::test]
    async fn stale_utterance_end_does_not_flush_new_speech() {
        let (event_sender, event_receiver) = mpsc::unbounded_channel();
        let (request_sender, mut request_receiver) = llm_request_channel(0);
        let (output_sender, mut output_receiver) = mpsc::unbounded_channel();
        let (_command_sender, command_receiver) = mpsc::unbounded_channel();

        event_sender
            .send(DeepgramEvent::Transcript {
                text: "И еще короткий".to_owned(),
                is_final: false,
                speech_final: false,
                audio_start_ms: Some(3_200),
                audio_duration_ms: Some(300),
                last_word_end_ms: Some(3_450),
            })
            .expect("new speech transcript must send");
        event_sender
            .send(DeepgramEvent::UtteranceEnd {
                last_word_end_ms: Some(3_100),
            })
            .expect("stale utterance end must send");
        event_sender
            .send(DeepgramEvent::Transcript {
                text: "И еще короткий вопрос".to_owned(),
                is_final: true,
                speech_final: true,
                audio_start_ms: Some(3_200),
                audio_duration_ms: Some(800),
                last_word_end_ms: Some(3_950),
            })
            .expect("final transcript must send");
        drop(event_sender);

        let stats = run_transcript_windows(
            event_receiver,
            request_sender,
            output_sender,
            command_receiver,
            TranscriptConfig {
                window_sec: 60,
                min_utterance_chars: 3,
            },
            None,
        )
        .await
        .expect("transcript worker must stop cleanly");

        assert_eq!(stats.chunks, 1);
        assert_eq!(
            request_receiver
                .recv()
                .await
                .expect("speech final must queue one request")
                .text,
            "И еще короткий вопрос"
        );
        assert_eq!(request_receiver.recv().await, None);

        let mut ignored_stale_boundary = false;
        while let Ok(event) = output_receiver.try_recv() {
            ignored_stale_boundary |= matches!(
                event,
                OutputEvent::SttObservation {
                    observation: SttObservation::UtteranceEnd {
                        last_word_end_ms: Some(3_100),
                        ignored: true,
                        deferred: false,
                    },
                    ..
                }
            );
        }
        assert!(ignored_stale_boundary);
    }

    #[test]
    fn forwards_growing_interim_text_as_a_draft() {
        let (output_sender, mut output_receiver) = mpsc::unbounded_channel();
        let mut assembler = TranscriptWindowAssembler::new(3);
        let mut pending_last_word_end_ms = None;
        let mut stats = TranscriptStats::default();

        handle_deepgram_event(
            DeepgramEvent::Transcript {
                text: "что такое".to_owned(),
                is_final: false,
                speech_final: false,
                audio_start_ms: None,
                audio_duration_ms: None,
                last_word_end_ms: None,
            },
            &mut assembler,
            &mut pending_last_word_end_ms,
            &output_sender,
            &mut stats,
            Speaker::Interviewer,
        )
        .expect("interim transcript must be forwarded");
        handle_deepgram_event(
            DeepgramEvent::Transcript {
                text: "что такое HashMap".to_owned(),
                is_final: false,
                speech_final: false,
                audio_start_ms: None,
                audio_duration_ms: None,
                last_word_end_ms: None,
            },
            &mut assembler,
            &mut pending_last_word_end_ms,
            &output_sender,
            &mut stats,
            Speaker::Interviewer,
        )
        .expect("growing interim transcript must be forwarded");

        assert_eq!(
            output_receiver.try_recv(),
            Ok(OutputEvent::SttObservation {
                speaker: Speaker::Interviewer,
                observation: SttObservation::Transcript {
                    text: "что такое".to_owned(),
                    is_final: false,
                    speech_final: false,
                    speech_final_deferred: false,
                    audio_start_ms: None,
                    audio_duration_ms: None,
                    last_word_end_ms: None,
                },
            })
        );
        assert_eq!(
            output_receiver.try_recv(),
            Ok(OutputEvent::TranscriptDraft {
                speaker: Speaker::Interviewer,
                text: "что такое".to_owned(),
            })
        );
        assert_eq!(
            output_receiver.try_recv(),
            Ok(OutputEvent::SttObservation {
                speaker: Speaker::Interviewer,
                observation: SttObservation::Transcript {
                    text: "что такое HashMap".to_owned(),
                    is_final: false,
                    speech_final: false,
                    speech_final_deferred: false,
                    audio_start_ms: None,
                    audio_duration_ms: None,
                    last_word_end_ms: None,
                },
            })
        );
        assert_eq!(
            output_receiver.try_recv(),
            Ok(OutputEvent::TranscriptDraft {
                speaker: Speaker::Interviewer,
                text: "что такое HashMap".to_owned(),
            })
        );
    }

    #[test]
    fn classifies_only_suspicious_boundaries_as_incomplete() {
        assert_eq!(
            boundary_deferral("Расскажите, за счет ЧЕГО..."),
            Some(BoundaryDeferral::DanglingSuffix)
        );
        assert_eq!(
            boundary_deferral("Какие операции выполняются и какие"),
            Some(BoundaryDeferral::DanglingSuffix)
        );
        assert_eq!(
            boundary_deferral("Если у нас"),
            Some(BoundaryDeferral::DanglingSuffix)
        );
        assert_eq!(
            boundary_deferral("Представим, что два приложения обновляют запись."),
            Some(BoundaryDeferral::Setup)
        );
        assert_eq!(
            boundary_deferral("Так, ну, давайте, наверное, начнём со Spring Boot."),
            Some(BoundaryDeferral::Introduction)
        );
        assert_eq!(
            boundary_deferral("Хорошо, теперь Java."),
            Some(BoundaryDeferral::ShortFragment)
        );
        assert_eq!(
            boundary_deferral("В чем разница между"),
            Some(BoundaryDeferral::DanglingSuffix)
        );
        assert_eq!(boundary_deferral("Что такое HashMap?"), None);
        assert_eq!(boundary_deferral("Что произойдет после?"), None);
        assert_eq!(boundary_deferral("И что?"), None);
        assert_eq!(boundary_deferral("Между чем?"), None);
        assert_eq!(boundary_deferral("Расскажите о Java."), None);
    }

    #[tokio::test]
    async fn benchmark_readiness_waits_for_stt_connection() {
        let (sender, receiver) = watch::channel(false);
        let wait = tokio::spawn(wait_for_transcript_readiness(receiver));
        tokio::task::yield_now().await;
        assert!(!wait.is_finished());

        sender.send(true).expect("readiness must be delivered");
        wait.await
            .expect("wait task must finish")
            .expect("readiness must succeed");
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
            Speaker::Interviewer,
        )
        .expect("status must be forwarded");

        assert!(matches!(
            output_receiver.try_recv(),
            Ok(OutputEvent::Status(StatusMessage {
                kind: StatusKind::Reconnecting,
                text,
            })) if text == "Deepgram interviewer reconnect in 4s (retry 3)"
        ));
        assert_eq!(
            output_receiver.try_recv(),
            Ok(OutputEvent::QueueState(QueueState {
                queue: QueueKind::Audio,
                len: 17,
            }))
        );
    }

    #[tokio::test]
    async fn retrieval_accumulates_interim_hits_and_prioritizes_final_query() {
        let (request_sender, mut request_receiver) = mpsc::unbounded_channel();
        let (result_sender, result_receiver) = mpsc::unbounded_channel();
        let (_readiness_sender, readiness_receiver) = watch::channel(true);
        let mut retrieval = RetrievalPipeline::new(
            KnowledgeConfig {
                enabled: true,
                embedding: crate::config::EmbeddingConfig {
                    api_key: crate::config::SecretString::new("test".to_owned()),
                    base_url: url::Url::parse("https://example.test/api/v1")
                        .expect("URL must parse"),
                    model: "test-embedding".to_owned(),
                    dimensions: 1_024,
                    query_input_type: "search_query".to_owned(),
                    document_input_type: "search_document".to_owned(),
                    timeout_sec: 30,
                },
                top_k: 2,
                max_context_chars: 2_000,
                min_score: 0.75,
                refresh_ms: 1_000,
                final_wait_ms: 50,
                debug: false,
            },
            request_sender,
            result_receiver,
            readiness_receiver,
        );

        retrieval.prefetch("Что такое блокировка", true);
        let first = request_receiver.recv().await.expect("prefetch must send");
        result_sender
            .send(KnowledgeSearchResult {
                search_id: first.search_id,
                turn_id: first.turn_id,
                query: first.query,
                report: Ok(query_report(
                    "old",
                    "Общие сведения",
                    "Старый interim-фрагмент",
                    0.86,
                )),
            })
            .expect("first result must send");

        let final_query = "Чем отличается optimistic locking от pessimistic locking?";
        retrieval.prefetch(final_query, true);
        let second = request_receiver
            .recv()
            .await
            .expect("final prefetch must send");
        result_sender
            .send(KnowledgeSearchResult {
                search_id: second.search_id,
                turn_id: second.turn_id,
                query: second.query,
                report: Ok(query_report(
                    "final",
                    "Hibernate > Блокировки",
                    "Optimistic использует version, pessimistic блокирует данные.",
                    0.91,
                )),
            })
            .expect("final result must send");

        let context = retrieval
            .resolve(final_query)
            .await
            .expect("retrieval must succeed");

        assert_eq!(context.searches, 2);
        assert_eq!(context.snippets.len(), 2);
        assert_eq!(context.snippets[0].id, "final");
        assert_eq!(context.snippets[1].id, "old");
    }

    fn query_report(
        id: &str,
        heading: &str,
        text: &str,
        score: f32,
    ) -> crate::knowledge::QueryReport {
        crate::knowledge::QueryReport {
            embedding: Duration::from_millis(8),
            embedding_calls: 1,
            prompt_tokens: 4,
            total_tokens: 4,
            search: Duration::from_millis(1),
            hits: vec![crate::knowledge::SearchHit {
                id: id.to_owned(),
                source: PathBuf::from("knowledge/java.md"),
                heading: heading.to_owned(),
                text: text.to_owned(),
                dense_score: score,
                lexical_score: 0.0,
                combined_score: score,
            }],
            index_path: PathBuf::from("index.json"),
        }
    }
}
