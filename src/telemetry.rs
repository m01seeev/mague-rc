use std::{
    collections::BTreeMap,
    fs::{self, File},
    io::{BufWriter, Write},
    path::{Path, PathBuf},
    process::Command,
    time::{Instant, SystemTime, UNIX_EPOCH},
};

use serde::Serialize;
use serde_json::{Value, json};
use thiserror::Error;
use tokio::{sync::mpsc, task::JoinError};

use crate::{
    config::Config,
    events::{LlmUsage, OutputComponent, OutputEvent, StatusKind, SttObservation},
    output::{OutputSink, OutputStats},
};

const SCHEMA_VERSION: u32 = 1;

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
    active_utterance: Option<usize>,
    active_request: Option<u64>,
    audio_stream_started_ms: Option<u64>,
    draft_started_ms: Option<u64>,
    draft_active: bool,
    last_final_ms: Option<u64>,
    last_speech_final_ms: Option<u64>,
    last_utterance_end_ms: Option<u64>,
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
        let directory = directory.as_ref();
        fs::create_dir_all(directory)?;
        let started_unix_ms = unix_time_ms();
        let run_id = format!("{}-{}", started_unix_ms, std::process::id());
        let file_stem = format!("{}-{}", run_id, sanitize_label(label));
        let events_path = directory.join(format!("{file_stem}.events.jsonl"));
        let summary_path = directory.join(format!("{file_stem}.summary.json"));
        let writer = BufWriter::new(File::create(&events_path)?);
        let audio_path = audio_path.as_ref();
        let audio_bytes = fs::metadata(audio_path).map(|metadata| metadata.len()).ok();
        let reference_text = reference_path.map(fs::read_to_string).transpose()?;

        let metadata = RunMetadata {
            schema_version: SCHEMA_VERSION,
            run_id,
            label: label.to_owned(),
            started_unix_ms,
            application_version: env!("CARGO_PKG_VERSION").to_owned(),
            git_branch: git_output(&["branch", "--show-current"]),
            git_commit: git_output(&["rev-parse", "HEAD"]),
            git_dirty: git_output(&["status", "--porcelain"])
                .is_some_and(|output| !output.is_empty()),
            audio_file: fs::canonicalize(audio_path)
                .unwrap_or_else(|_| audio_path.to_path_buf())
                .display()
                .to_string(),
            audio_bytes,
            audio_sha256: file_sha256(audio_path),
            audio_duration_ms: audio_duration_ms(audio_path),
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
            active_utterance: None,
            active_request: None,
            audio_stream_started_ms: None,
            draft_started_ms: None,
            draft_active: false,
            last_final_ms: None,
            last_speech_final_ms: None,
            last_utterance_end_ms: None,
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
            OutputEvent::Status(status) => {
                if status.kind == StatusKind::Listening && self.audio_stream_started_ms.is_none() {
                    self.audio_stream_started_ms = Some(elapsed_ms);
                }
                (
                    "status",
                    None,
                    json!({"kind": format!("{:?}", status.kind), "text": status.text}),
                )
            }
            OutputEvent::SttObservation(observation) => {
                return self.record_stt(elapsed_ms, observation);
            }
            OutputEvent::TranscriptDraft { text } => {
                if text.is_empty() {
                    self.draft_active = false;
                } else if !self.draft_active {
                    self.draft_active = true;
                    self.draft_started_ms = Some(elapsed_ms);
                }
                (
                    "transcript_draft",
                    None,
                    json!({"text": text, "chars": text.chars().count()}),
                )
            }
            OutputEvent::Transcript(transcript) => {
                let metrics = self.requests.entry(transcript.sequence).or_default();
                metrics.question.clone_from(&transcript.text);
                metrics.flush_reason.clone_from(&transcript.flush_reason);
                metrics.draft_started_ms = self.draft_started_ms.take();
                metrics.question_queued_ms = Some(elapsed_ms);
                metrics.last_final_ms = self.last_final_ms.take();
                metrics.speech_final_ms = self.last_speech_final_ms.take();
                metrics.utterance_end_ms = self.last_utterance_end_ms.take();
                self.draft_active = false;
                (
                    "question_queued",
                    Some(transcript.sequence),
                    json!({
                        "text": transcript.text,
                        "chars": transcript.text.chars().count(),
                        "words": word_count(&transcript.text),
                        "flush_reason": transcript.flush_reason,
                    }),
                )
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
                    json!({"mode": meta.mode.to_string()}),
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
        observation: &SttObservation,
    ) -> Result<(), TelemetryError> {
        let (name, fields) = match observation {
            SttObservation::Transcript {
                text,
                is_final,
                speech_final,
                audio_start_ms,
                audio_duration_ms,
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
                    self.last_speech_final_ms = Some(elapsed_ms);
                    utterance.speech_final_ms = Some(elapsed_ms);
                    finish_recognition_segment(
                        &mut self.pending_recognition,
                        &mut self.recognition_segments,
                    );
                }
                (
                    "stt_transcript",
                    json!({
                        "text": text,
                        "chars": text.chars().count(),
                        "is_final": is_final,
                        "speech_final": speech_final,
                        "audio_start_ms": audio_start_ms,
                        "audio_duration_ms": audio_duration_ms,
                        "audio_end_ms": audio_end_ms,
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
            SttObservation::UtteranceEnd { last_word_end_ms } => {
                let delivery_lag_ms = self.delivery_lag_ms(elapsed_ms, *last_word_end_ms);
                if let Some(delivery_lag_ms) = delivery_lag_ms {
                    self.stt_latency
                        .utterance_end_delivery_lag_ms
                        .push(delivery_lag_ms);
                }
                self.stt.utterance_end += 1;
                if self.draft_active {
                    self.last_utterance_end_ms = Some(elapsed_ms);
                }
                if let Some(index) = self.active_utterance.take() {
                    self.utterances[index].utterance_end_ms = Some(elapsed_ms);
                }
                finish_recognition_segment(
                    &mut self.pending_recognition,
                    &mut self.recognition_segments,
                );
                (
                    "utterance_end",
                    json!({
                        "last_word_end_ms": last_word_end_ms,
                        "delivery_lag_ms": delivery_lag_ms,
                    }),
                )
            }
        };
        self.write_event(elapsed_ms, name, None, fields)
    }

    fn start_draft(&mut self, elapsed_ms: u64) {
        self.draft_active = true;
        self.draft_started_ms = Some(elapsed_ms);
        self.last_final_ms = None;
        self.last_speech_final_ms = None;
        self.last_utterance_end_ms = None;
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
            recognition_segments: self.recognition_segments,
            accuracy,
        };
        let summary_file = File::create(&self.summary_path)?;
        serde_json::to_writer_pretty(BufWriter::new(summary_file), &summary)?;

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

#[derive(Serialize)]
struct RunSummary {
    schema_version: u32,
    metadata: RunMetadata,
    elapsed_ms: u64,
    event_count: u64,
    audio_stream_started_ms: Option<u64>,
    stt: SttMetrics,
    stt_latency: SttLatencySummary,
    aggregates: AggregateSummary,
    requests: Vec<RequestSummary>,
    utterances: Vec<UtteranceSummary>,
    recognition_segments: Vec<String>,
    accuracy: Option<AccuracySummary>,
}

#[derive(Serialize)]
struct RunMetadata {
    schema_version: u32,
    run_id: String,
    label: String,
    started_unix_ms: u64,
    application_version: String,
    git_branch: Option<String>,
    git_commit: Option<String>,
    git_dirty: bool,
    audio_file: String,
    audio_bytes: Option<u64>,
    audio_sha256: Option<String>,
    audio_duration_ms: Option<u64>,
    reference_file: Option<String>,
    reference_sha256: Option<String>,
    configuration: ConfigurationSnapshot,
}

#[derive(Serialize)]
struct ConfigurationSnapshot {
    deepgram_model: String,
    deepgram_language: String,
    sample_rate: u32,
    channels: u16,
    interim_results: bool,
    punctuate: bool,
    smart_format: bool,
    vad_events: bool,
    endpointing_ms: u64,
    utterance_end_ms: u64,
    keyterms: Vec<String>,
    audio_chunk_ms: u64,
    transcript_segmentation: &'static str,
    transcript_window_sec: u64,
    min_utterance_chars: usize,
    llm_model: String,
    max_history_pairs: usize,
    temperature: f32,
    max_tokens: u32,
}

impl From<&Config> for ConfigurationSnapshot {
    fn from(config: &Config) -> Self {
        Self {
            deepgram_model: config.deepgram.model.clone(),
            deepgram_language: config.deepgram.language.clone(),
            sample_rate: config.deepgram.sample_rate,
            channels: config.deepgram.channels,
            interim_results: config.deepgram.interim_results,
            punctuate: config.deepgram.punctuate,
            smart_format: config.deepgram.smart_format,
            vad_events: config.deepgram.vad_events,
            endpointing_ms: config.deepgram.endpointing_ms,
            utterance_end_ms: config.deepgram.utterance_end_ms,
            keyterms: config.deepgram.keyterms.clone(),
            audio_chunk_ms: config.audio.chunk_ms,
            transcript_segmentation: "deepgram_utterance_with_inactivity_fallback",
            transcript_window_sec: config.transcript.window_sec,
            min_utterance_chars: config.transcript.min_utterance_chars,
            llm_model: config.llm.model.clone(),
            max_history_pairs: config.llm.max_history_pairs,
            temperature: config.llm.temperature,
            max_tokens: config.llm.max_tokens,
        }
    }
}

#[derive(Default)]
struct RequestMetrics {
    question: String,
    flush_reason: String,
    answer: String,
    draft_started_ms: Option<u64>,
    last_final_ms: Option<u64>,
    speech_final_ms: Option<u64>,
    utterance_end_ms: Option<u64>,
    question_queued_ms: Option<u64>,
    llm_started_ms: Option<u64>,
    first_token_ms: Option<u64>,
    completed_ms: Option<u64>,
    failed_ms: Option<u64>,
    answer_chars: u64,
    usage: Option<LlmUsage>,
}

#[derive(Serialize)]
struct RequestSummary {
    request_id: u64,
    question: String,
    flush_reason: String,
    answer: String,
    question_chars: usize,
    question_words: usize,
    answer_chars: u64,
    draft_started_ms: Option<u64>,
    last_final_ms: Option<u64>,
    speech_final_ms: Option<u64>,
    utterance_end_ms: Option<u64>,
    question_queued_ms: Option<u64>,
    llm_started_ms: Option<u64>,
    first_token_ms: Option<u64>,
    completed_ms: Option<u64>,
    failed_ms: Option<u64>,
    build_ms: Option<u64>,
    final_to_queue_ms: Option<u64>,
    speech_final_to_queue_ms: Option<u64>,
    utterance_end_to_queue_ms: Option<u64>,
    queue_wait_ms: Option<u64>,
    ttft_ms: Option<u64>,
    queued_to_first_token_ms: Option<u64>,
    speech_end_to_first_token_ms: Option<u64>,
    generation_ms: Option<u64>,
    total_request_ms: Option<u64>,
    usage: Option<LlmUsageSummary>,
}

impl RequestSummary {
    fn from_metrics(request_id: u64, metrics: RequestMetrics) -> Self {
        let speech_end_ms = metrics.utterance_end_ms.or(metrics.speech_final_ms);
        Self {
            request_id,
            question_chars: metrics.question.chars().count(),
            question_words: word_count(&metrics.question),
            answer_chars: metrics.answer_chars,
            build_ms: duration(metrics.question_queued_ms, metrics.draft_started_ms),
            final_to_queue_ms: duration(metrics.question_queued_ms, metrics.last_final_ms),
            speech_final_to_queue_ms: duration(metrics.question_queued_ms, metrics.speech_final_ms),
            utterance_end_to_queue_ms: duration(
                metrics.question_queued_ms,
                metrics.utterance_end_ms,
            ),
            queue_wait_ms: duration(metrics.llm_started_ms, metrics.question_queued_ms),
            ttft_ms: duration(metrics.first_token_ms, metrics.llm_started_ms),
            queued_to_first_token_ms: duration(metrics.first_token_ms, metrics.question_queued_ms),
            speech_end_to_first_token_ms: duration(metrics.first_token_ms, speech_end_ms),
            generation_ms: duration(metrics.completed_ms, metrics.first_token_ms),
            total_request_ms: duration(metrics.completed_ms, metrics.llm_started_ms),
            question: metrics.question,
            flush_reason: metrics.flush_reason,
            answer: metrics.answer,
            draft_started_ms: metrics.draft_started_ms,
            last_final_ms: metrics.last_final_ms,
            speech_final_ms: metrics.speech_final_ms,
            utterance_end_ms: metrics.utterance_end_ms,
            question_queued_ms: metrics.question_queued_ms,
            llm_started_ms: metrics.llm_started_ms,
            first_token_ms: metrics.first_token_ms,
            completed_ms: metrics.completed_ms,
            failed_ms: metrics.failed_ms,
            usage: metrics.usage.map(LlmUsageSummary::from),
        }
    }
}

#[derive(Serialize)]
struct LlmUsageSummary {
    prompt_tokens: u64,
    completion_tokens: u64,
    total_tokens: u64,
    cost: Option<f64>,
}

impl From<LlmUsage> for LlmUsageSummary {
    fn from(usage: LlmUsage) -> Self {
        Self {
            prompt_tokens: usage.prompt_tokens,
            completion_tokens: usage.completion_tokens,
            total_tokens: usage.total_tokens,
            cost: usage.cost,
        }
    }
}

#[derive(Default, Serialize)]
struct SttMetrics {
    transcripts: u64,
    interim_transcripts: u64,
    final_transcripts: u64,
    speech_final_transcripts: u64,
    speech_started: u64,
    utterance_end: u64,
}

#[derive(Serialize)]
struct AggregateSummary {
    request_count: usize,
    completed_count: usize,
    failed_count: usize,
    prompt_tokens: u64,
    completion_tokens: u64,
    total_tokens: u64,
    total_cost: f64,
    build_ms: MetricSummary,
    queue_wait_ms: MetricSummary,
    ttft_ms: MetricSummary,
    queued_to_first_token_ms: MetricSummary,
    speech_end_to_first_token_ms: MetricSummary,
    generation_ms: MetricSummary,
    total_request_ms: MetricSummary,
}

impl AggregateSummary {
    fn new(requests: &[RequestSummary]) -> Self {
        let usage = requests
            .iter()
            .filter_map(|request| request.usage.as_ref())
            .collect::<Vec<_>>();
        Self {
            request_count: requests.len(),
            completed_count: requests
                .iter()
                .filter(|request| request.completed_ms.is_some())
                .count(),
            failed_count: requests
                .iter()
                .filter(|request| request.failed_ms.is_some())
                .count(),
            prompt_tokens: usage.iter().map(|usage| usage.prompt_tokens).sum(),
            completion_tokens: usage.iter().map(|usage| usage.completion_tokens).sum(),
            total_tokens: usage.iter().map(|usage| usage.total_tokens).sum(),
            total_cost: usage.iter().filter_map(|usage| usage.cost).sum(),
            build_ms: metric_summary(requests.iter().filter_map(|request| request.build_ms)),
            queue_wait_ms: metric_summary(
                requests.iter().filter_map(|request| request.queue_wait_ms),
            ),
            ttft_ms: metric_summary(requests.iter().filter_map(|request| request.ttft_ms)),
            queued_to_first_token_ms: metric_summary(
                requests
                    .iter()
                    .filter_map(|request| request.queued_to_first_token_ms),
            ),
            speech_end_to_first_token_ms: metric_summary(
                requests
                    .iter()
                    .filter_map(|request| request.speech_end_to_first_token_ms),
            ),
            generation_ms: metric_summary(
                requests.iter().filter_map(|request| request.generation_ms),
            ),
            total_request_ms: metric_summary(
                requests
                    .iter()
                    .filter_map(|request| request.total_request_ms),
            ),
        }
    }
}

#[derive(Default)]
struct SttLatencyMetrics {
    interim_delivery_lag_ms: Vec<u64>,
    final_delivery_lag_ms: Vec<u64>,
    speech_started_delivery_lag_ms: Vec<u64>,
    utterance_end_delivery_lag_ms: Vec<u64>,
}

#[derive(Serialize)]
struct SttLatencySummary {
    approximation: &'static str,
    interim_delivery_lag_ms: MetricSummary,
    final_delivery_lag_ms: MetricSummary,
    speech_started_delivery_lag_ms: MetricSummary,
    utterance_end_delivery_lag_ms: MetricSummary,
}

impl From<SttLatencyMetrics> for SttLatencySummary {
    fn from(metrics: SttLatencyMetrics) -> Self {
        Self {
            approximation: "receive_elapsed - audio_stream_start_elapsed - Deepgram_audio_position",
            interim_delivery_lag_ms: metric_summary(metrics.interim_delivery_lag_ms.into_iter()),
            final_delivery_lag_ms: metric_summary(metrics.final_delivery_lag_ms.into_iter()),
            speech_started_delivery_lag_ms: metric_summary(
                metrics.speech_started_delivery_lag_ms.into_iter(),
            ),
            utterance_end_delivery_lag_ms: metric_summary(
                metrics.utterance_end_delivery_lag_ms.into_iter(),
            ),
        }
    }
}

#[derive(Default)]
struct UtteranceMetrics {
    sequence: u64,
    speech_started_ms: Option<u64>,
    first_transcript_ms: Option<u64>,
    first_interim_ms: Option<u64>,
    first_final_ms: Option<u64>,
    speech_final_ms: Option<u64>,
    utterance_end_ms: Option<u64>,
    final_text: String,
}

#[derive(Serialize)]
struct UtteranceSummary {
    sequence: u64,
    final_text: String,
    speech_started_ms: Option<u64>,
    first_transcript_ms: Option<u64>,
    first_interim_ms: Option<u64>,
    first_final_ms: Option<u64>,
    speech_final_ms: Option<u64>,
    utterance_end_ms: Option<u64>,
    speech_to_first_interim_ms: Option<u64>,
    speech_to_first_final_ms: Option<u64>,
    utterance_duration_ms: Option<u64>,
}

#[derive(Serialize)]
struct AccuracySummary {
    reference_text: String,
    recognized_text: String,
    word_errors: usize,
    reference_words: usize,
    wer: Option<f64>,
    character_errors: usize,
    reference_characters: usize,
    cer: Option<f64>,
    segments: Vec<AccuracySegment>,
}

impl AccuracySummary {
    fn new(reference: &str, recognized_segments: &[String]) -> Self {
        let recognized_text = recognized_segments.join(" ");
        let reference_segments = reference
            .lines()
            .map(str::trim)
            .filter(|line| !line.is_empty())
            .collect::<Vec<_>>();
        let segment_count = reference_segments.len().max(recognized_segments.len());
        let segments = (0..segment_count)
            .map(|index| {
                AccuracySegment::new(
                    index,
                    reference_segments.get(index).copied().unwrap_or_default(),
                    recognized_segments
                        .get(index)
                        .map(String::as_str)
                        .unwrap_or_default(),
                )
            })
            .collect();
        let scores = AccuracyScores::new(reference, &recognized_text);

        Self {
            reference_text: reference.trim().to_owned(),
            recognized_text,
            word_errors: scores.word_errors,
            reference_words: scores.reference_words,
            wer: scores.wer,
            character_errors: scores.character_errors,
            reference_characters: scores.reference_characters,
            cer: scores.cer,
            segments,
        }
    }
}

#[derive(Serialize)]
struct AccuracySegment {
    sequence: usize,
    reference: String,
    recognized: String,
    word_errors: usize,
    reference_words: usize,
    wer: Option<f64>,
    character_errors: usize,
    reference_characters: usize,
    cer: Option<f64>,
}

impl AccuracySegment {
    fn new(sequence: usize, reference: &str, recognized: &str) -> Self {
        let scores = AccuracyScores::new(reference, recognized);
        Self {
            sequence,
            reference: reference.to_owned(),
            recognized: recognized.to_owned(),
            word_errors: scores.word_errors,
            reference_words: scores.reference_words,
            wer: scores.wer,
            character_errors: scores.character_errors,
            reference_characters: scores.reference_characters,
            cer: scores.cer,
        }
    }
}

struct AccuracyScores {
    word_errors: usize,
    reference_words: usize,
    wer: Option<f64>,
    character_errors: usize,
    reference_characters: usize,
    cer: Option<f64>,
}

impl AccuracyScores {
    fn new(reference: &str, recognized: &str) -> Self {
        let normalized_reference = normalize_transcript(reference);
        let normalized_recognized = normalize_transcript(recognized);
        let reference_words = normalized_reference.split_whitespace().collect::<Vec<_>>();
        let recognized_words = normalized_recognized.split_whitespace().collect::<Vec<_>>();
        let reference_characters = normalized_reference
            .chars()
            .filter(|character| !character.is_whitespace())
            .collect::<Vec<_>>();
        let recognized_characters = normalized_recognized
            .chars()
            .filter(|character| !character.is_whitespace())
            .collect::<Vec<_>>();
        let word_errors = edit_distance(&reference_words, &recognized_words);
        let character_errors = edit_distance(&reference_characters, &recognized_characters);

        Self {
            word_errors,
            reference_words: reference_words.len(),
            wer: error_rate(word_errors, reference_words.len()),
            character_errors,
            reference_characters: reference_characters.len(),
            cer: error_rate(character_errors, reference_characters.len()),
        }
    }
}

impl From<UtteranceMetrics> for UtteranceSummary {
    fn from(metrics: UtteranceMetrics) -> Self {
        Self {
            sequence: metrics.sequence,
            speech_to_first_interim_ms: duration(
                metrics.first_interim_ms,
                metrics.speech_started_ms,
            ),
            speech_to_first_final_ms: duration(metrics.first_final_ms, metrics.speech_started_ms),
            utterance_duration_ms: duration(
                metrics.utterance_end_ms.or(metrics.speech_final_ms),
                metrics.speech_started_ms,
            ),
            final_text: metrics.final_text,
            speech_started_ms: metrics.speech_started_ms,
            first_transcript_ms: metrics.first_transcript_ms,
            first_interim_ms: metrics.first_interim_ms,
            first_final_ms: metrics.first_final_ms,
            speech_final_ms: metrics.speech_final_ms,
            utterance_end_ms: metrics.utterance_end_ms,
        }
    }
}

#[derive(Default, Serialize)]
struct MetricSummary {
    count: usize,
    min: Option<u64>,
    mean: Option<f64>,
    p50: Option<u64>,
    p95: Option<u64>,
    max: Option<u64>,
}

fn metric_summary(values: impl Iterator<Item = u64>) -> MetricSummary {
    let mut values = values.collect::<Vec<_>>();
    if values.is_empty() {
        return MetricSummary::default();
    }
    values.sort_unstable();
    let sum = values.iter().map(|value| *value as u128).sum::<u128>();
    MetricSummary {
        count: values.len(),
        min: values.first().copied(),
        mean: Some(sum as f64 / values.len() as f64),
        p50: percentile(&values, 0.50),
        p95: percentile(&values, 0.95),
        max: values.last().copied(),
    }
}

fn percentile(values: &[u64], percentile: f64) -> Option<u64> {
    if values.is_empty() {
        return None;
    }
    let rank = (percentile * values.len() as f64).ceil() as usize;
    values.get(rank.saturating_sub(1)).copied()
}

fn duration(later: Option<u64>, earlier: Option<u64>) -> Option<u64> {
    later?.checked_sub(earlier?)
}

fn word_count(value: &str) -> usize {
    value.split_whitespace().count()
}

fn append_text(target: &mut String, text: &str) {
    if !target.is_empty() && !text.starts_with(char::is_whitespace) {
        target.push(' ');
    }
    target.push_str(text);
}

fn finish_recognition_segment(pending: &mut String, segments: &mut Vec<String>) {
    let text = pending.trim();
    if !text.is_empty() {
        segments.push(text.to_owned());
    }
    pending.clear();
}

fn normalize_transcript(value: &str) -> String {
    let mut normalized = String::with_capacity(value.len());
    let mut last_was_separator = true;
    for character in value.chars().flat_map(char::to_lowercase) {
        let character = if character == 'ё' { 'е' } else { character };
        if character.is_alphanumeric() {
            normalized.push(character);
            last_was_separator = false;
        } else if !last_was_separator {
            normalized.push(' ');
            last_was_separator = true;
        }
    }
    normalized.trim().to_owned()
}

fn edit_distance<T: Eq>(reference: &[T], recognized: &[T]) -> usize {
    let mut previous = (0..=recognized.len()).collect::<Vec<_>>();
    let mut current = vec![0; recognized.len() + 1];

    for (reference_index, reference_item) in reference.iter().enumerate() {
        current[0] = reference_index + 1;
        for (recognized_index, recognized_item) in recognized.iter().enumerate() {
            let substitution =
                previous[recognized_index] + usize::from(reference_item != recognized_item);
            let insertion = current[recognized_index] + 1;
            let deletion = previous[recognized_index + 1] + 1;
            current[recognized_index + 1] = substitution.min(insertion).min(deletion);
        }
        std::mem::swap(&mut previous, &mut current);
    }

    previous[recognized.len()]
}

fn error_rate(errors: usize, reference_units: usize) -> Option<f64> {
    (reference_units > 0).then(|| errors as f64 / reference_units as f64)
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
            AudioConfig, DeepgramConfig, LegendConfig, LlmConfig, ScreenshotConfig, SecretString,
            TranscriptConfig, VisionConfig,
        },
        events::{AnswerMeta, Mode, TranscriptView},
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
            .record(&OutputEvent::SttObservation(
                SttObservation::SpeechStarted {
                    audio_timestamp_ms: Some(100),
                },
            ))
            .expect("speech start must be recorded");
        recorder
            .record(&OutputEvent::SttObservation(SttObservation::Transcript {
                text: "Что такое HashMap?".to_owned(),
                is_final: true,
                speech_final: true,
                audio_start_ms: Some(100),
                audio_duration_ms: Some(900),
            }))
            .expect("transcript must be recorded");
        recorder
            .record(&OutputEvent::SttObservation(SttObservation::UtteranceEnd {
                last_word_end_ms: Some(1_000),
            }))
            .expect("utterance end must be recorded");
        recorder
            .record(&OutputEvent::TranscriptDraft {
                text: "Что такое HashMap?".to_owned(),
            })
            .expect("draft must be recorded");
        recorder
            .record(&OutputEvent::Transcript(TranscriptView {
                sequence: 0,
                text: "Что такое HashMap?".to_owned(),
                flush_reason: "timer".to_owned(),
            }))
            .expect("question must be recorded");
        recorder
            .record(&OutputEvent::AnswerStarted(AnswerMeta {
                request_id: 0,
                mode: Mode::Voice,
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
        assert_eq!(summary["recognition_segments"][0], "Что такое HashMap?");
        assert_eq!(summary["requests"][0]["flush_reason"], "timer");
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
            transcript: TranscriptConfig {
                window_sec: 5,
                min_utterance_chars: 3,
            },
            legend: LegendConfig {
                enabled: false,
                path: PathBuf::from("legend.md"),
                top_k: 2,
                chunk_chars: 2_200,
                max_context_chars: 4_200,
                min_score: 1.6,
                recent_user_turns: 2,
                reload_on_change: true,
                debug: false,
            },
            screenshot: ScreenshotConfig {
                enabled: false,
                path: PathBuf::from("/tmp/test.png"),
                pid_file: PathBuf::from("/tmp/test.pid"),
                max_image_mb: 3.5,
                debounce_sec: 1,
            },
        }
    }
}
