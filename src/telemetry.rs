use std::{
    collections::BTreeMap,
    fs::{self, File},
    io::{BufWriter, Write},
    path::{Path, PathBuf},
    process::Command,
    time::{Instant, SystemTime, UNIX_EPOCH},
};

use serde_json::{Value, json};
use thiserror::Error;
use tokio::{sync::mpsc, task::JoinError};

use crate::{
    config::Config,
    events::{OutputComponent, OutputEvent, SttObservation},
    output::{OutputSink, OutputStats},
};

mod model;

use model::*;

const SCHEMA_VERSION: u32 = 9;

#[derive(Debug, Error)]
pub enum TelemetryError {
    #[error("could not create telemetry directory or file: {0}")]
    Io(#[from] std::io::Error),

    #[error("could not serialize telemetry: {0}")]
    Json(#[from] serde_json::Error),

    #[error("telemetry output consumer stopped")]
    OutputClosed,

    #[error("telemetry output task failed: {0}")]
    OutputTask(#[from] JoinError),

    #[error("wrapped output sink failed: {0}")]
    Inner(String),
}

pub struct TelemetryOutputSink<S> {
    inner: S,
    recorder: TelemetryRecorder,
}

impl<S> TelemetryOutputSink<S> {
    pub fn new(
        inner: S,
        directory: impl AsRef<Path>,
        label: &str,
        audio_path: impl AsRef<Path>,
        reference_path: Option<&Path>,
        config: &Config,
    ) -> Result<Self, TelemetryError> {
        Ok(Self {
            inner,
            recorder: TelemetryRecorder::new(directory, label, audio_path, reference_path, config)?,
        })
    }

    pub fn new_session(
        inner: S,
        directory: impl AsRef<Path>,
        label: &str,
        config: &Config,
    ) -> Result<Self, TelemetryError> {
        let recorder = TelemetryRecorder::new_session(directory, label, config)?;
        eprintln!("session events: {}", recorder.events_path.display());
        eprintln!("session summary: {}", recorder.summary_path.display());
        Ok(Self { inner, recorder })
    }
}

impl<S> OutputSink for TelemetryOutputSink<S>
where
    S: OutputSink,
    S::Error: std::fmt::Display,
{
    type Error = TelemetryError;

    async fn run(
        mut self,
        mut events: mpsc::UnboundedReceiver<OutputEvent>,
    ) -> Result<OutputStats, TelemetryError> {
        let (inner_sender, inner_receiver) = mpsc::unbounded_channel();
        let inner_task = tokio::spawn(self.inner.run(inner_receiver));

        while let Some(event) = events.recv().await {
            self.recorder.record(&event)?;
            inner_sender
                .send(event)
                .map_err(|_| TelemetryError::OutputClosed)?;
        }
        drop(inner_sender);

        let stats = inner_task
            .await?
            .map_err(|error| TelemetryError::Inner(error.to_string()))?;
        let artifacts = self.recorder.finish()?;
        eprintln!("telemetry events: {}", artifacts.events_path.display());
        eprintln!("telemetry summary: {}", artifacts.summary_path.display());
        Ok(stats)
    }
}

struct TelemetryRecorder {
    metadata: RunMetadata,
    started: Instant,
    events_path: PathBuf,
    summary_path: PathBuf,
    writer: BufWriter<File>,
    reference_text: Option<String>,
    recognition_segments: Vec<String>,
    pending_recognition: String,
    event_count: u64,
    requests: BTreeMap<u64, RequestMetrics>,
    utterances: Vec<UtteranceMetrics>,
    live_coding: Vec<LiveCodingRevision>,
    active_utterance: Option<usize>,
    active_request: Option<u64>,
    audio_stream_started_ms: Option<u64>,
    draft_started_ms: Option<u64>,
    draft_active: bool,
    last_final_ms: Option<u64>,
    last_speech_final_ms: Option<u64>,
    last_utterance_end_ms: Option<u64>,
    last_word_end_audio_ms: Option<u64>,
    last_word_to_boundary_ms: Option<u64>,
    stt: SttMetrics,
    stt_latency: SttLatencyMetrics,
}

impl TelemetryRecorder {
    fn new(
        directory: impl AsRef<Path>,
        label: &str,
        audio_path: impl AsRef<Path>,
        reference_path: Option<&Path>,
        config: &Config,
    ) -> Result<Self, TelemetryError> {
        Self::create(
            directory,
            label,
            "benchmark",
            Some(audio_path.as_ref()),
            reference_path,
            config,
        )
    }

    fn new_session(
        directory: impl AsRef<Path>,
        label: &str,
        config: &Config,
    ) -> Result<Self, TelemetryError> {
        Self::create(directory, label, "overlay", None, None, config)
    }

    fn create(
        directory: impl AsRef<Path>,
        label: &str,
        run_kind: &str,
        audio_path: Option<&Path>,
        reference_path: Option<&Path>,
        config: &Config,
    ) -> Result<Self, TelemetryError> {
        let directory = directory.as_ref();
        fs::create_dir_all(directory)?;
        let started_unix_ms = unix_time_ms();
        let run_id = format!("{}-{}", started_unix_ms, std::process::id());
        let file_stem = format!("{}-{}", run_id, sanitize_label(label));
        let events_path = directory.join(format!("{file_stem}.events.jsonl"));
        let summary_path = directory.join(format!("{file_stem}.summary.json"));
        let writer = BufWriter::new(File::create(&events_path)?);
        let audio_bytes = audio_path
            .and_then(|path| fs::metadata(path).map(|metadata| metadata.len()).ok());
        let reference_text = reference_path.map(fs::read_to_string).transpose()?;

        let metadata = RunMetadata {
            schema_version: SCHEMA_VERSION,
            run_id,
            label: label.to_owned(),
            run_kind: run_kind.to_owned(),
            started_unix_ms,
            application_version: env!("CARGO_PKG_VERSION").to_owned(),
            git_branch: git_output(&["branch", "--show-current"]),
            git_commit: git_output(&["rev-parse", "HEAD"]),
            git_dirty: git_output(&["status", "--porcelain"])
                .is_some_and(|output| !output.is_empty()),
            audio_file: audio_path.map(|path| {
                fs::canonicalize(path)
                    .unwrap_or_else(|_| path.to_path_buf())
                    .display()
                    .to_string()
            }),
            audio_bytes,
            audio_sha256: audio_path.and_then(file_sha256),
            audio_duration_ms: audio_path.and_then(audio_duration_ms),
            reference_file: reference_path.map(|path| {
                fs::canonicalize(path)
                    .unwrap_or_else(|_| path.to_path_buf())
                    .display()
                    .to_string()
            }),
            reference_sha256: reference_path.and_then(file_sha256),
            configuration: ConfigurationSnapshot::from(config),
        };

        let mut recorder = Self {
            metadata,
            started: Instant::now(),
            events_path,
            summary_path,
            writer,
            reference_text,
            recognition_segments: Vec::new(),
            pending_recognition: String::new(),
            event_count: 0,
            requests: BTreeMap::new(),
            utterances: Vec::new(),
            live_coding: Vec::new(),
            active_utterance: None,
            active_request: None,
            audio_stream_started_ms: None,
            draft_started_ms: None,
            draft_active: false,
            last_final_ms: None,
            last_speech_final_ms: None,
            last_utterance_end_ms: None,
            last_word_end_audio_ms: None,
            last_word_to_boundary_ms: None,
            stt: SttMetrics::default(),
            stt_latency: SttLatencyMetrics::default(),
        };
        let metadata = serde_json::to_value(&recorder.metadata)?;
        recorder.write_event(0, "run_started", None, json!({"metadata": metadata}))?;
        Ok(recorder)
    }

    fn record(&mut self, event: &OutputEvent) -> Result<(), TelemetryError> {
        let elapsed_ms = self.elapsed_ms();
        let (name, request_id, fields) = match event {
            OutputEvent::ModeChanged { mode } => {
                ("mode_changed", None, json!({"mode": mode.to_string()}))
            }
            OutputEvent::Status(status) => (
                "status",
                None,
                json!({"kind": format!("{:?}", status.kind), "text": status.text}),
            ),
            OutputEvent::SttObservation {
                speaker,
                observation,
            } => {
                return self.record_stt(elapsed_ms, *speaker, observation);
            }
            OutputEvent::TranscriptDraft { speaker, text } => {
                if *speaker == crate::events::Speaker::Interviewer {
                    if text.is_empty() {
                        self.draft_active = false;
                    } else if !self.draft_active {
                        self.draft_active = true;
                        self.draft_started_ms = Some(elapsed_ms);
                    }
                }
                (
                    "transcript_draft",
                    None,
                    json!({
                        "speaker": speaker.to_string(),
                        "text": text,
                        "chars": text.chars().count(),
                    }),
                )
            }
            OutputEvent::Transcript(transcript) => {
                let metrics = self.requests.entry(transcript.sequence).or_default();
                metrics.mode = transcript.mode.to_string();
                metrics.speaker = transcript.speaker.to_string();
                metrics.question.clone_from(&transcript.text);
                metrics.flush_reason.clone_from(&transcript.flush_reason);
                metrics.question_ready_ms = Some(elapsed_ms);
                if transcript.speaker == crate::events::Speaker::Interviewer {
                    metrics.draft_started_ms = self.draft_started_ms.take();
                    metrics.last_final_ms = self.last_final_ms.take();
                    metrics.speech_final_ms = self.last_speech_final_ms.take();
                    metrics.utterance_end_ms = self.last_utterance_end_ms.take();
                    metrics.last_word_end_audio_ms = self.last_word_end_audio_ms.take();
                    metrics.last_word_to_boundary_ms = self.last_word_to_boundary_ms.take();
                    self.draft_active = false;
                }
                (
                    "question_ready",
                    Some(transcript.sequence),
                    json!({
                        "text": transcript.text,
                        "mode": transcript.mode.to_string(),
                        "speaker": transcript.speaker.to_string(),
                        "chars": transcript.text.chars().count(),
                        "words": word_count(&transcript.text),
                        "flush_reason": transcript.flush_reason,
                    }),
                )
            }
            OutputEvent::Retrieval(retrieval) => {
                let context = &retrieval.context;
                self.requests
                    .entry(retrieval.request_id)
                    .or_default()
                    .retrieval = Some(RetrievalMetrics {
                    searches: context.searches,
                    embedding_calls: context.embedding_calls,
                    embedding_prompt_tokens: context.embedding_prompt_tokens,
                    embedding_total_tokens: context.embedding_total_tokens,
                    snippets: context.snippets.len(),
                    context_chars: context
                        .snippets
                        .iter()
                        .map(|snippet| snippet.text.chars().count())
                        .sum(),
                    embedding_ms: context.embedding_ms,
                    search_ms: context.search_ms,
                    final_wait_ms: context.final_wait_ms,
                    hits: context
                        .snippets
                        .iter()
                        .map(|snippet| RetrievalHitMetrics {
                            id: snippet.id.clone(),
                            source: snippet.source.clone(),
                            heading: snippet.heading.clone(),
                            text: snippet.text.clone(),
                            score: snippet.score,
                        })
                        .collect(),
                });
                (
                    "rag_context_attached",
                    Some(retrieval.request_id),
                    json!({
                        "searches": context.searches,
                        "embedding_calls": context.embedding_calls,
                        "embedding_prompt_tokens": context.embedding_prompt_tokens,
                        "embedding_total_tokens": context.embedding_total_tokens,
                        "snippets": context.snippets.len(),
                        "context_chars": context
                            .snippets
                            .iter()
                            .map(|snippet| snippet.text.chars().count())
                            .sum::<usize>(),
                        "embedding_ms": context.embedding_ms,
                        "search_ms": context.search_ms,
                        "final_wait_ms": context.final_wait_ms,
                        "hits": context.snippets.iter().map(|snippet| json!({
                            "id": snippet.id,
                            "source": snippet.source,
                            "heading": snippet.heading,
                            "score": snippet.score,
                            "chars": snippet.text.chars().count(),
                            "text": snippet.text,
                        })).collect::<Vec<_>>(),
                    }),
                )
            }
            OutputEvent::LlmQueued { request_id } => {
                self.requests.entry(*request_id).or_default().llm_queued_ms = Some(elapsed_ms);
                ("llm_queued", Some(*request_id), json!({}))
            }
            OutputEvent::AnswerStarted(meta) => {
                self.active_request = Some(meta.request_id);
                self.requests
                    .entry(meta.request_id)
                    .or_default()
                    .llm_started_ms = Some(elapsed_ms);
                (
                    "llm_started",
                    Some(meta.request_id),
                    json!({
                        "mode": meta.mode.to_string(),
                        "speaker": meta.speaker.to_string(),
                    }),
                )
            }
            OutputEvent::AnswerDelta { request_id, text } => {
                let metrics = self.requests.entry(*request_id).or_default();
                metrics.first_token_ms.get_or_insert(elapsed_ms);
                metrics.answer_chars += text.chars().count() as u64;
                metrics.answer.push_str(text);
                (
                    "answer_delta",
                    Some(*request_id),
                    json!({"text": text, "chars": text.chars().count()}),
                )
            }
            OutputEvent::AnswerUsage { request_id, usage } => {
                self.requests.entry(*request_id).or_default().usage = Some(usage.clone());
                (
                    "llm_usage",
                    Some(*request_id),
                    json!({
                        "prompt_tokens": usage.prompt_tokens,
                        "completion_tokens": usage.completion_tokens,
                        "total_tokens": usage.total_tokens,
                        "cost": usage.cost,
                    }),
                )
            }
            OutputEvent::AnswerCompleted { request_id } => {
                self.active_request = None;
                self.requests.entry(*request_id).or_default().completed_ms = Some(elapsed_ms);
                ("answer_completed", Some(*request_id), json!({}))
            }
            OutputEvent::LiveCodingUpdated(state) => {
                let request_id = self.active_request;
                let speaker = request_id
                    .and_then(|request_id| self.requests.get(&request_id))
                    .map(|metrics| metrics.speaker.clone())
                    .filter(|speaker| !speaker.is_empty());
                self.live_coding.push(LiveCodingRevision {
                    request_id,
                    speaker: speaker.clone(),
                    revision: state.revision,
                    summary: state.summary.clone(),
                    candidate_context: state.candidate_context.clone(),
                    explanation: state.explanation.clone(),
                    language: state.language.clone(),
                    code: state.code.clone(),
                    change_note: state.change_note.clone(),
                    changed_lines: state.changed_lines.clone(),
                });
                (
                    "live_coding_updated",
                    request_id,
                    json!({
                        "revision": state.revision,
                        "speaker": speaker,
                        "summary": state.summary,
                        "candidate_context": state.candidate_context,
                        "explanation": state.explanation,
                        "language": state.language,
                        "code": state.code,
                        "change_note": state.change_note,
                        "changed_lines": state.changed_lines,
                    }),
                )
            }
            OutputEvent::QueueState(queue) => (
                "queue_state",
                None,
                json!({"queue": queue.queue.to_string(), "len": queue.len}),
            ),
            OutputEvent::Error(error) => {
                let mut request_id = None;
                if error.component == OutputComponent::Llm
                    && let Some(active_request) = self.active_request.take()
                {
                    request_id = Some(active_request);
                    self.requests.entry(active_request).or_default().failed_ms = Some(elapsed_ms);
                }
                (
                    "error",
                    request_id,
                    json!({
                        "component": error.component.to_string(),
                        "message": error.message,
                    }),
                )
            }
        };

        self.write_event(elapsed_ms, name, request_id, fields)
    }

    fn record_stt(
        &mut self,
        elapsed_ms: u64,
        speaker: crate::events::Speaker,
        observation: &SttObservation,
    ) -> Result<(), TelemetryError> {
        if speaker == crate::events::Speaker::Candidate {
            return self.record_candidate_stt(elapsed_ms, observation);
        }
        let (name, fields) = match observation {
            SttObservation::AudioStreamStarted => {
                self.audio_stream_started_ms = Some(elapsed_ms);
                ("audio_stream_started", json!({}))
            }
            SttObservation::Transcript {
                text,
                is_final,
                speech_final,
                speech_final_deferred,
                audio_start_ms,
                audio_duration_ms,
                last_word_end_ms,
            } => {
                let audio_end_ms = (*audio_start_ms)
                    .zip(*audio_duration_ms)
                    .and_then(|(start, duration)| start.checked_add(duration));
                let delivery_lag_ms = self.delivery_lag_ms(elapsed_ms, audio_end_ms);
                if !self.draft_active {
                    self.start_draft(elapsed_ms);
                }
                let utterance = self.active_utterance.unwrap_or_else(|| {
                    self.utterances.push(UtteranceMetrics {
                        sequence: self.utterances.len() as u64,
                        ..UtteranceMetrics::default()
                    });
                    let index = self.utterances.len() - 1;
                    self.active_utterance = Some(index);
                    index
                });
                let utterance = &mut self.utterances[utterance];
                utterance.first_transcript_ms.get_or_insert(elapsed_ms);
                self.stt.transcripts += 1;
                if *is_final {
                    self.stt.final_transcripts += 1;
                    self.last_final_ms = Some(elapsed_ms);
                    utterance.first_final_ms.get_or_insert(elapsed_ms);
                    append_text(&mut utterance.final_text, text);
                    append_text(&mut self.pending_recognition, text);
                    if let Some(delivery_lag_ms) = delivery_lag_ms {
                        self.stt_latency.final_delivery_lag_ms.push(delivery_lag_ms);
                    }
                    if let Some(last_word_end_ms) = last_word_end_ms {
                        self.last_word_end_audio_ms = Some(*last_word_end_ms);
                    }
                } else {
                    self.stt.interim_transcripts += 1;
                    utterance.first_interim_ms.get_or_insert(elapsed_ms);
                    if let Some(delivery_lag_ms) = delivery_lag_ms {
                        self.stt_latency
                            .interim_delivery_lag_ms
                            .push(delivery_lag_ms);
                    }
                }
                if *speech_final {
                    self.stt.speech_final_transcripts += 1;
                    if *speech_final_deferred {
                        self.stt.deferred_speech_final_transcripts += 1;
                    } else {
                        self.last_speech_final_ms = Some(elapsed_ms);
                        self.last_word_to_boundary_ms = audio_end_ms
                            .zip(*last_word_end_ms)
                            .and_then(|(audio_end_ms, last_word_end_ms)| {
                                audio_end_ms.checked_sub(last_word_end_ms)
                            })
                            .and_then(|word_to_audio_end_ms| {
                                word_to_audio_end_ms.checked_add(delivery_lag_ms.unwrap_or(0))
                            });
                        utterance.speech_final_ms = Some(elapsed_ms);
                        finish_recognition_segment(
                            &mut self.pending_recognition,
                            &mut self.recognition_segments,
                        );
                    }
                }
                (
                    "stt_transcript",
                    json!({
                        "text": text,
                        "chars": text.chars().count(),
                        "is_final": is_final,
                        "speech_final": speech_final,
                        "speech_final_deferred": speech_final_deferred,
                        "audio_start_ms": audio_start_ms,
                        "audio_duration_ms": audio_duration_ms,
                        "audio_end_ms": audio_end_ms,
                        "last_word_end_ms": last_word_end_ms,
                        "delivery_lag_ms": delivery_lag_ms,
                    }),
                )
            }
            SttObservation::SpeechStarted { audio_timestamp_ms } => {
                let delivery_lag_ms = self.delivery_lag_ms(elapsed_ms, *audio_timestamp_ms);
                if let Some(delivery_lag_ms) = delivery_lag_ms {
                    self.stt_latency
                        .speech_started_delivery_lag_ms
                        .push(delivery_lag_ms);
                }
                self.stt.speech_started += 1;
                self.start_draft(elapsed_ms);
                if let Some(index) = self.active_utterance
                    && self.utterances[index].speech_started_ms.is_some()
                {
                    self.active_utterance = None;
                }
                let index = self.active_utterance.unwrap_or_else(|| {
                    self.utterances.push(UtteranceMetrics {
                        sequence: self.utterances.len() as u64,
                        ..UtteranceMetrics::default()
                    });
                    self.utterances.len() - 1
                });
                self.utterances[index].speech_started_ms = Some(elapsed_ms);
                self.active_utterance = Some(index);
                (
                    "speech_started",
                    json!({
                        "audio_timestamp_ms": audio_timestamp_ms,
                        "delivery_lag_ms": delivery_lag_ms,
                    }),
                )
            }
            SttObservation::UtteranceEnd {
                last_word_end_ms,
                ignored,
                deferred,
            } => {
                let delivery_lag_ms = self.delivery_lag_ms(elapsed_ms, *last_word_end_ms);
                if let Some(delivery_lag_ms) = delivery_lag_ms {
                    self.stt_latency
                        .utterance_end_delivery_lag_ms
                        .push(delivery_lag_ms);
                }
                self.stt.utterance_end += 1;
                if *ignored {
                    self.stt.ignored_utterance_end += 1;
                } else if *deferred {
                    self.stt.deferred_utterance_end += 1;
                } else {
                    if self.draft_active {
                        self.last_utterance_end_ms = Some(elapsed_ms);
                    }
                    if let Some(last_word_end_ms) = last_word_end_ms {
                        self.last_word_end_audio_ms = Some(*last_word_end_ms);
                        self.last_word_to_boundary_ms = Some(delivery_lag_ms.unwrap_or(0));
                    }
                    if let Some(index) = self.active_utterance.take() {
                        self.utterances[index].utterance_end_ms = Some(elapsed_ms);
                    }
                    finish_recognition_segment(
                        &mut self.pending_recognition,
                        &mut self.recognition_segments,
                    );
                }
                (
                    "utterance_end",
                    json!({
                        "last_word_end_ms": last_word_end_ms,
                        "delivery_lag_ms": delivery_lag_ms,
                        "ignored": ignored,
                        "deferred": deferred,
                    }),
                )
            }
        };
        let mut fields = fields;
        if let Value::Object(values) = &mut fields {
            values.insert("speaker".to_owned(), json!(speaker.to_string()));
        }
        self.write_event(elapsed_ms, name, None, fields)
    }

    fn record_candidate_stt(
        &mut self,
        elapsed_ms: u64,
        observation: &SttObservation,
    ) -> Result<(), TelemetryError> {
        let (name, fields) = match observation {
            SttObservation::AudioStreamStarted => ("audio_stream_started", json!({})),
            SttObservation::Transcript {
                text,
                is_final,
                speech_final,
                speech_final_deferred,
                audio_start_ms,
                audio_duration_ms,
                last_word_end_ms,
            } => {
                let audio_end_ms = (*audio_start_ms)
                    .zip(*audio_duration_ms)
                    .and_then(|(start, duration)| start.checked_add(duration));
                (
                    "stt_transcript",
                    json!({
                        "text": text,
                        "chars": text.chars().count(),
                        "is_final": is_final,
                        "speech_final": speech_final,
                        "speech_final_deferred": speech_final_deferred,
                        "audio_start_ms": audio_start_ms,
                        "audio_duration_ms": audio_duration_ms,
                        "audio_end_ms": audio_end_ms,
                        "last_word_end_ms": last_word_end_ms,
                        "delivery_lag_ms": null,
                    }),
                )
            }
            SttObservation::SpeechStarted { audio_timestamp_ms } => (
                "speech_started",
                json!({
                    "audio_timestamp_ms": audio_timestamp_ms,
                    "delivery_lag_ms": null,
                }),
            ),
            SttObservation::UtteranceEnd {
                last_word_end_ms,
                ignored,
                deferred,
            } => (
                "utterance_end",
                json!({
                    "last_word_end_ms": last_word_end_ms,
                    "delivery_lag_ms": null,
                    "ignored": ignored,
                    "deferred": deferred,
                }),
            ),
        };
        let mut fields = fields;
        if let Value::Object(values) = &mut fields {
            values.insert("speaker".to_owned(), json!("candidate"));
        }
        self.write_event(elapsed_ms, name, None, fields)
    }

    fn start_draft(&mut self, elapsed_ms: u64) {
        self.draft_active = true;
        self.draft_started_ms = Some(elapsed_ms);
        self.last_final_ms = None;
        self.last_speech_final_ms = None;
        self.last_utterance_end_ms = None;
        self.last_word_end_audio_ms = None;
        self.last_word_to_boundary_ms = None;
    }

    fn delivery_lag_ms(&self, elapsed_ms: u64, audio_position_ms: Option<u64>) -> Option<u64> {
        elapsed_ms
            .checked_sub(self.audio_stream_started_ms?)?
            .checked_sub(audio_position_ms?)
    }

    fn write_event(
        &mut self,
        elapsed_ms: u64,
        event: &str,
        request_id: Option<u64>,
        fields: Value,
    ) -> Result<(), TelemetryError> {
        self.event_count += 1;
        serde_json::to_writer(
            &mut self.writer,
            &json!({
                "schema_version": SCHEMA_VERSION,
                "run_id": self.metadata.run_id,
                "sequence": self.event_count,
                "elapsed_ms": elapsed_ms,
                "event": event,
                "request_id": request_id,
                "fields": fields,
            }),
        )?;
        self.writer.write_all(b"\n")?;
        self.writer.flush()?;
        Ok(())
    }

    fn finish(mut self) -> Result<TelemetryArtifacts, TelemetryError> {
        let elapsed_ms = self.elapsed_ms();
        self.write_event(elapsed_ms, "run_completed", None, json!({}))?;
        self.writer.flush()?;
        finish_recognition_segment(
            &mut self.pending_recognition,
            &mut self.recognition_segments,
        );

        let requests = self
            .requests
            .into_iter()
            .map(|(request_id, metrics)| RequestSummary::from_metrics(request_id, metrics))
            .collect::<Vec<_>>();
        let utterances = self
            .utterances
            .into_iter()
            .map(UtteranceSummary::from)
            .collect::<Vec<_>>();
        let accuracy = self
            .reference_text
            .as_deref()
            .map(|reference| AccuracySummary::new(reference, &self.recognition_segments));
        let summary = RunSummary {
            schema_version: SCHEMA_VERSION,
            metadata: self.metadata,
            elapsed_ms,
            event_count: self.event_count,
            audio_stream_started_ms: self.audio_stream_started_ms,
            stt: self.stt,
            stt_latency: SttLatencySummary::from(self.stt_latency),
            aggregates: AggregateSummary::new(&requests),
            requests,
            utterances,
            live_coding: self.live_coding,
            recognition_segments: self.recognition_segments,
            accuracy,
        };
        let summary_file = File::create(&self.summary_path)?;
        let mut summary_writer = BufWriter::new(summary_file);
        serde_json::to_writer_pretty(&mut summary_writer, &summary)?;
        summary_writer.flush()?;

        Ok(TelemetryArtifacts {
            events_path: self.events_path,
            summary_path: self.summary_path,
        })
    }

    fn elapsed_ms(&self) -> u64 {
        self.started
            .elapsed()
            .as_millis()
            .try_into()
            .unwrap_or(u64::MAX)
    }
}

struct TelemetryArtifacts {
    events_path: PathBuf,
    summary_path: PathBuf,
}

fn sanitize_label(label: &str) -> String {
    let mut sanitized = String::with_capacity(label.len());
    let mut last_was_separator = false;
    for character in label.chars() {
        if character.is_ascii_alphanumeric() || matches!(character, '-' | '_') {
            sanitized.push(character);
            last_was_separator = false;
        } else if !last_was_separator {
            sanitized.push('-');
            last_was_separator = true;
        }
    }
    let sanitized = sanitized.trim_matches('-');
    if sanitized.is_empty() {
        "run".to_owned()
    } else {
        sanitized.to_owned()
    }
}

fn git_output(arguments: &[&str]) -> Option<String> {
    let output = Command::new("git").args(arguments).output().ok()?;
    if !output.status.success() {
        return None;
    }
    Some(String::from_utf8_lossy(&output.stdout).trim().to_owned())
}

fn file_sha256(path: &Path) -> Option<String> {
    let output = Command::new("sha256sum").arg(path).output().ok()?;
    if !output.status.success() {
        return None;
    }
    String::from_utf8_lossy(&output.stdout)
        .split_whitespace()
        .next()
        .map(str::to_owned)
}

fn audio_duration_ms(path: &Path) -> Option<u64> {
    let output = Command::new("ffprobe")
        .args([
            "-v",
            "error",
            "-show_entries",
            "format=duration",
            "-of",
            "default=noprint_wrappers=1:nokey=1",
        ])
        .arg(path)
        .output()
        .ok()?;
    if !output.status.success() {
        return None;
    }
    let seconds = String::from_utf8_lossy(&output.stdout)
        .trim()
        .parse::<f64>()
        .ok()?;
    if !seconds.is_finite() || seconds.is_sign_negative() {
        return None;
    }
    Some((seconds * 1_000.0).round() as u64)
}

fn unix_time_ms() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|duration| duration.as_millis().try_into().unwrap_or(u64::MAX))
        .unwrap_or_default()
}

#[cfg(test)]
mod tests {
    use std::fs;

    use url::Url;

    use super::*;
    use crate::{
        config::{
            AudioConfig, CandidateAudioConfig, DeepgramConfig, EmbeddingConfig, KnowledgeConfig,
            LlmConfig, ScreenshotConfig, SecretString, SessionLogConfig, TranscriptConfig,
            VisionConfig,
        },
        events::{
            AnswerMeta, KnowledgeContext, KnowledgeSnippet, LlmUsage, Mode, RetrievalView,
            Speaker, TranscriptView,
        },
    };

    #[test]
    fn calculates_metric_distribution() {
        let summary = metric_summary([100, 500, 200, 300, 400].into_iter());

        assert_eq!(summary.count, 5);
        assert_eq!(summary.min, Some(100));
        assert_eq!(summary.mean, Some(300.0));
        assert_eq!(summary.p50, Some(300));
        assert_eq!(summary.p95, Some(500));
        assert_eq!(summary.max, Some(500));
    }

    #[test]
    fn sanitizes_experiment_label_for_file_names() {
        assert_eq!(
            sanitize_label("utterance end / 1200ms"),
            "utterance-end-1200ms"
        );
        assert_eq!(sanitize_label("..."), "run");
    }

    #[test]
    fn writes_correlated_events_and_summary_without_secrets() {
        let directory =
            std::env::temp_dir().join(format!("mague-rc-telemetry-test-{}", unix_time_ms()));
        fs::create_dir_all(&directory).expect("temporary directory must be created");
        let audio_path = directory.join("input.wav");
        fs::write(&audio_path, b"not a real wave file").expect("fixture must be written");
        let mut recorder = TelemetryRecorder::new(
            &directory,
            "fixed window / baseline",
            &audio_path,
            None,
            &config(),
        )
        .expect("recorder must start");

        recorder
            .record(&OutputEvent::SttObservation {
                speaker: Speaker::Interviewer,
                observation: SttObservation::SpeechStarted {
                    audio_timestamp_ms: Some(100),
                },
            })
            .expect("speech start must be recorded");
        recorder
            .record(&OutputEvent::SttObservation {
                speaker: Speaker::Interviewer,
                observation: SttObservation::Transcript {
                    text: "Что такое HashMap?".to_owned(),
                    is_final: true,
                    speech_final: true,
                    speech_final_deferred: false,
                    audio_start_ms: Some(100),
                    audio_duration_ms: Some(900),
                    last_word_end_ms: Some(900),
                },
            })
            .expect("transcript must be recorded");
        recorder
            .record(&OutputEvent::SttObservation {
                speaker: Speaker::Interviewer,
                observation: SttObservation::UtteranceEnd {
                    last_word_end_ms: Some(1_000),
                    ignored: false,
                    deferred: false,
                },
            })
            .expect("utterance end must be recorded");
        recorder
            .record(&OutputEvent::TranscriptDraft {
                speaker: Speaker::Interviewer,
                text: "Что такое HashMap?".to_owned(),
            })
            .expect("draft must be recorded");
        recorder
            .record(&OutputEvent::Transcript(TranscriptView {
                sequence: 0,
                mode: Mode::Voice,
                speaker: Speaker::Interviewer,
                text: "Что такое HashMap?".to_owned(),
                flush_reason: "timer".to_owned(),
            }))
            .expect("question must be recorded");
        recorder
            .record(&OutputEvent::Retrieval(RetrievalView {
                request_id: 0,
                context: KnowledgeContext {
                    snippets: vec![KnowledgeSnippet {
                        id: "hash-map".to_owned(),
                        source: "knowledge/java.md".to_owned(),
                        heading: "Java > HashMap".to_owned(),
                        text: "HashMap хранит пары ключ-значение.".to_owned(),
                        score: 0.91,
                    }],
                    searches: 2,
                    embedding_calls: 2,
                    embedding_prompt_tokens: 24,
                    embedding_total_tokens: 24,
                    embedding_ms: 16,
                    search_ms: 2,
                    final_wait_ms: 7,
                },
            }))
            .expect("retrieval must be recorded");
        recorder
            .record(&OutputEvent::LlmQueued { request_id: 0 })
            .expect("LLM queue submission must be recorded");
        recorder
            .record(&OutputEvent::AnswerStarted(AnswerMeta {
                request_id: 0,
                mode: Mode::Voice,
                speaker: Speaker::Interviewer,
            }))
            .expect("request start must be recorded");
        recorder
            .record(&OutputEvent::AnswerDelta {
                request_id: 0,
                text: "HashMap хранит пары ключ-значение.".to_owned(),
            })
            .expect("answer must be recorded");
        recorder
            .record(&OutputEvent::AnswerUsage {
                request_id: 0,
                usage: LlmUsage {
                    prompt_tokens: 100,
                    completion_tokens: 20,
                    total_tokens: 120,
                    cost: Some(0.001),
                },
            })
            .expect("usage must be recorded");
        recorder
            .record(&OutputEvent::AnswerCompleted { request_id: 0 })
            .expect("completion must be recorded");

        let artifacts = recorder.finish().expect("summary must be written");
        let summary_text =
            fs::read_to_string(&artifacts.summary_path).expect("summary must be readable");
        let summary: Value =
            serde_json::from_str(&summary_text).expect("summary must contain JSON");
        let events = fs::read_to_string(&artifacts.events_path).expect("events must be readable");

        assert_eq!(summary["metadata"]["label"], "fixed window / baseline");
        assert_eq!(summary["metadata"]["audio_bytes"], 20);
        assert_eq!(summary["aggregates"]["request_count"], 1);
        assert_eq!(summary["aggregates"]["total_tokens"], 120);
        assert_eq!(summary["aggregates"]["rag_request_count"], 1);
        assert_eq!(summary["aggregates"]["rag_searches"], 2);
        assert_eq!(summary["recognition_segments"][0], "Что такое HashMap?");
        assert_eq!(summary["requests"][0]["flush_reason"], "timer");
        assert_eq!(summary["requests"][0]["retrieval"]["snippets"], 1);
        assert_eq!(summary["requests"][0]["retrieval"]["searches"], 2);
        assert_eq!(
            summary["requests"][0]["answer"],
            "HashMap хранит пары ключ-значение."
        );
        assert_eq!(summary["utterances"][0]["final_text"], "Что такое HashMap?");
        assert!(events.lines().count() >= 10);
        assert!(!summary_text.contains("deepgram-secret"));
        assert!(!summary_text.contains("openrouter-secret"));

        fs::remove_dir_all(directory).expect("temporary directory must be removed");
    }

    #[test]
    fn calculates_case_and_punctuation_insensitive_accuracy() {
        let scores = AccuracyScores::new(
            "Что такое HashMap?\nОбъясни Java Stream API.",
            "что такое hashmap объясни java streams api",
        );

        assert_eq!(scores.reference_words, 7);
        assert_eq!(scores.word_errors, 1);
        assert_eq!(scores.wer, Some(1.0 / 7.0));
        assert!(scores.character_errors > 0);
        assert!(scores.cer.is_some());
        assert_eq!(normalize_transcript("Ёлка, JAVA!"), "елка java");
    }

    #[test]
    fn calculates_delivery_lag_from_audio_stream_position() {
        let directory = std::env::temp_dir().join(format!("mague-rc-lag-test-{}", unix_time_ms()));
        fs::create_dir_all(&directory).expect("temporary directory must be created");
        let audio_path = directory.join("input.wav");
        fs::write(&audio_path, b"audio").expect("fixture must be written");
        let mut recorder = TelemetryRecorder::new(&directory, "lag", &audio_path, None, &config())
            .expect("recorder must start");
        recorder.audio_stream_started_ms = Some(100);

        assert_eq!(recorder.delivery_lag_ms(1_600, Some(1_000)), Some(500));
        assert_eq!(recorder.delivery_lag_ms(900, Some(1_000)), None);

        drop(recorder);
        fs::remove_dir_all(directory).expect("temporary directory must be removed");
    }

    #[test]
    fn separates_boundary_and_last_word_latency() {
        let summary = RequestSummary::from_metrics(
            7,
            RequestMetrics {
                speech_final_ms: Some(1_600),
                last_word_end_audio_ms: Some(1_000),
                last_word_to_boundary_ms: Some(500),
                first_token_ms: Some(2_200),
                ..RequestMetrics::default()
            },
        );

        assert_eq!(summary.speech_boundary_to_first_token_ms, Some(600));
        assert_eq!(summary.last_word_to_first_token_ms, Some(1_100));
        assert_eq!(summary.last_word_end_audio_ms, Some(1_000));
        assert_eq!(summary.last_word_to_boundary_ms, Some(500));
    }

    fn config() -> Config {
        Config {
            deepgram: DeepgramConfig {
                api_key: SecretString::new("deepgram-secret".to_owned()),
                ws_url: Url::parse("wss://example.test/listen").expect("URL must parse"),
                model: "nova-3".to_owned(),
                language: "ru".to_owned(),
                sample_rate: 16_000,
                channels: 1,
                interim_results: true,
                punctuate: true,
                smart_format: true,
                vad_events: true,
                endpointing_ms: 500,
                utterance_end_ms: 1_200,
                keyterms: vec!["Java".to_owned()],
            },
            llm: LlmConfig {
                api_key: SecretString::new("openrouter-secret".to_owned()),
                base_url: Url::parse("https://example.test/api/v1").expect("URL must parse"),
                model: "test-model".to_owned(),
                queue_max: 0,
                max_history_pairs: 4,
                separate_histories: true,
                temperature: 0.2,
                max_tokens: 450,
                timeout_sec: 30,
                coding_temperature: 0.1,
                coding_max_tokens: 1_800,
                coding_timeout_sec: 60,
                current_project: "test".to_owned(),
            },
            vision: VisionConfig {
                api_key: SecretString::new(String::new()),
                base_url: Url::parse("https://example.test/api/v1").expect("URL must parse"),
                model: "vision-model".to_owned(),
                max_tokens: 100,
                timeout_sec: 30,
            },
            audio: AudioConfig {
                ffmpeg_bin: PathBuf::from("ffmpeg"),
                input_format: "pulse".to_owned(),
                source: "default".to_owned(),
                chunk_ms: 100,
                queue_max: 0,
            },
            candidate_audio: CandidateAudioConfig {
                enabled: true,
                source: "default-mic".to_owned(),
            },
            transcript: TranscriptConfig {
                window_sec: 5,
                min_utterance_chars: 3,
            },
            knowledge: KnowledgeConfig {
                enabled: false,
                embedding: EmbeddingConfig {
                    api_key: SecretString::new("openrouter-secret".to_owned()),
                    base_url: Url::parse("https://example.test/api/v1").expect("URL must parse"),
                    model: "test-embedding".to_owned(),
                    dimensions: 1_024,
                    query_input_type: "search_query".to_owned(),
                    document_input_type: "search_document".to_owned(),
                    timeout_sec: 30,
                },
                top_k: 3,
                max_context_chars: 4_200,
                min_score: 0.75,
                refresh_ms: 1_000,
                final_wait_ms: 80,
                debug: false,
            },
            screenshot: ScreenshotConfig {
                enabled: false,
                path: PathBuf::from("/tmp/test.png"),
                pid_file: PathBuf::from("/tmp/test.pid"),
                max_image_mb: 3.5,
                debounce_sec: 1,
            },
            session_log: SessionLogConfig {
                enabled: true,
                directory: PathBuf::from("telemetry/sessions"),
            },
        }
    }
}
