use std::{
    collections::HashMap, path::PathBuf, thread::JoinHandle as ThreadJoinHandle, time::Duration,
};

use thiserror::Error;
use tokio::{
    sync::{mpsc, watch},
    task::{JoinError, JoinHandle},
    time::{Instant, sleep, timeout},
};
use tracing::{debug, info, warn};

use crate::{
    audio::{AudioError, AudioSource, FfmpegAudioSource, audio_frame_channel},
    config::{Config, KnowledgeConfig, TranscriptConfig},
    control::TerminalEchoGuard,
    error::AppError,
    events::{
        AppCommand, AppErrorView, DeepgramEvent, KnowledgeContext, KnowledgeSnippet, LlmCommand,
        LlmRequest, Mode, OutputComponent, OutputEvent, QueueKind, QueueState, RetrievalView,
        StatusKind, StatusMessage, SttObservation, SttStatus, TranscriptChunk, TranscriptView,
    },
    knowledge::{
        KnowledgeSearchRequest, KnowledgeSearchResult, KnowledgeWorkerStats, spawn_knowledge_worker,
    },
    llm::{
        LlmQueueError, LlmRequestSender, LlmWorker, OpenRouterTextProvider, llm_request_channel,
    },
    output::{OutputSink, TerminalOutputSink},
    stt::{DeepgramSttProvider, SpeechToTextProvider, SttError},
    transcript::TranscriptWindowAssembler,
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

    info!(
        module = "app",
        event = "started",
        audio_source = %audio_source_label,
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
    let (shutdown_sender, shutdown_receiver) = watch::channel(false);

    send_output(
        &output_sender,
        OutputEvent::Status(StatusMessage {
            kind: StatusKind::Started,
            text: format!(
                "source={} | STT={} | LLM={}",
                audio_source_label, config.deepgram.model, config.llm.model
            ),
        }),
    )?;

    let (retrieval, knowledge_thread) = if config.knowledge.enabled {
        match spawn_knowledge_worker() {
            Ok(runtime) => (
                Some(RetrievalPipeline::new(
                    config.knowledge.clone(),
                    runtime.requests,
                    runtime.results,
                    runtime.readiness,
                )),
                Some(runtime.thread),
            ),
            Err(error) => {
                warn!(
                    module = "knowledge",
                    event = "disabled",
                    error = %error,
                    "local knowledge retrieval disabled"
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
            shutdown_receiver,
            readiness_sender,
        ))
    } else {
        drop(readiness_sender);
        tokio::spawn(stt_provider.run(frame_receiver, stt_event_sender, shutdown_receiver))
    };
    let mut transcript_task = tokio::spawn(run_transcript_windows_with_retrieval(
        stt_event_receiver,
        llm_request_sender,
        output_sender.clone(),
        transcript_command_receiver,
        config.transcript,
        transcript_readiness,
        retrieval,
    ));
    let mut llm_task = tokio::spawn(llm_worker.run(
        llm_request_receiver,
        llm_command_receiver,
        output_sender.clone(),
    ));
    let mut output_task = tokio::spawn(sink.run(output_receiver));

    let mut audio_completed = None;
    let mut stt_completed = None;
    let mut transcript_completed = None;
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
    let knowledge_stats = finish_knowledge_thread(knowledge_thread).await;
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
    let transcript_stats = transcript_stats.map_err(AppError::TranscriptWorker)?;
    let llm_stats = llm_stats.map_err(AppError::LlmWorker)?;
    let output_stats = output_stats.map_err(|error| AppError::Output(error.to_string()))?;

    if let Some(worker) = stop_reason.unexpected_worker() {
        return Err(AppError::WorkerStopped(worker));
    }

    info!(
        module = "app",
        event = "shutdown_completed",
        transcript_events = transcript_stats.transcripts,
        final_transcripts = transcript_stats.final_transcripts,
        transcript_chunks = transcript_stats.chunks,
        knowledge_searches = knowledge_stats.searches,
        knowledge_failed = knowledge_stats.failed,
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
            Self::Llm => Some("LLM"),
            Self::Output => Some("output"),
        }
    }
}

#[derive(Default)]
struct TranscriptStats {
    transcripts: u64,
    final_transcripts: u64,
    chunks: u64,
}

struct RetrievalPipeline {
    config: KnowledgeConfig,
    requests: mpsc::UnboundedSender<KnowledgeSearchRequest>,
    results: mpsc::UnboundedReceiver<KnowledgeSearchResult>,
    readiness: watch::Receiver<bool>,
    turn_id: u64,
    next_search_id: u64,
    last_dispatch_at: Option<Instant>,
    last_query: String,
    last_search_id: Option<u64>,
    accumulated: HashMap<String, KnowledgeSnippet>,
    completed: HashMap<u64, Vec<String>>,
    searches: u64,
    embedding_ms: u64,
    search_ms: u64,
    last_error: Option<String>,
}

impl RetrievalPipeline {
    fn new(
        config: KnowledgeConfig,
        requests: mpsc::UnboundedSender<KnowledgeSearchRequest>,
        results: mpsc::UnboundedReceiver<KnowledgeSearchResult>,
        readiness: watch::Receiver<bool>,
    ) -> Self {
        Self {
            config,
            requests,
            results,
            readiness,
            turn_id: 0,
            next_search_id: 0,
            last_dispatch_at: None,
            last_query: String::new(),
            last_search_id: None,
            accumulated: HashMap::new(),
            completed: HashMap::new(),
            searches: 0,
            embedding_ms: 0,
            search_ms: 0,
            last_error: None,
        }
    }

    async fn wait_until_ready(&mut self) {
        loop {
            if *self.readiness.borrow_and_update() {
                return;
            }
            if self.readiness.changed().await.is_err() {
                self.last_error = Some("knowledge worker failed during initialization".to_owned());
                return;
            }
        }
    }

    fn prefetch(&mut self, query: &str, force: bool) {
        self.drain_ready();
        let query = query.trim();
        if query.is_empty() || query == self.last_query {
            return;
        }
        let refresh = Duration::from_millis(self.config.refresh_ms);
        if !force
            && self
                .last_dispatch_at
                .is_some_and(|started| started.elapsed() < refresh)
        {
            return;
        }
        self.dispatch(query);
    }

    async fn resolve(&mut self, query: &str) -> Result<Option<KnowledgeContext>, String> {
        self.drain_ready();
        let query = query.trim();
        let search_id = if query == self.last_query {
            self.last_search_id
        } else {
            self.dispatch(query)
        };

        let wait_started = Instant::now();
        if let Some(search_id) = search_id
            && !self.completed.contains_key(&search_id)
        {
            let deadline = sleep(Duration::from_millis(self.config.final_wait_ms));
            tokio::pin!(deadline);
            loop {
                tokio::select! {
                    _ = &mut deadline => break,
                    result = self.results.recv() => match result {
                        Some(result) => {
                            let completed_id = result.search_id;
                            self.accept_result(result);
                            if completed_id == search_id {
                                break;
                            }
                        }
                        None => {
                            self.last_error
                                .get_or_insert_with(|| "knowledge worker stopped".to_owned());
                            break;
                        }
                    }
                }
            }
        }
        let final_wait_ms = wait_started.elapsed().as_millis() as u64;

        let mut ordered_ids = search_id
            .and_then(|id| self.completed.get(&id))
            .cloned()
            .unwrap_or_default();
        let mut remaining = self.accumulated.values().collect::<Vec<_>>();
        remaining.sort_by(|left, right| right.score.total_cmp(&left.score));
        for snippet in remaining {
            if !ordered_ids.contains(&snippet.id) {
                ordered_ids.push(snippet.id.clone());
            }
        }

        let mut snippets = Vec::new();
        let mut remaining_chars = self.config.max_context_chars;
        for id in ordered_ids {
            if snippets.len() >= self.config.top_k || remaining_chars == 0 {
                break;
            }
            let Some(snippet) = self.accumulated.get(&id) else {
                continue;
            };
            let mut snippet = snippet.clone();
            let text_chars = snippet.text.chars().count();
            if text_chars > remaining_chars {
                snippet.text = snippet.text.chars().take(remaining_chars).collect();
            }
            remaining_chars = remaining_chars.saturating_sub(snippet.text.chars().count());
            snippets.push(snippet);
        }

        if snippets.is_empty() {
            if let Some(error) = self.last_error.take() {
                return Err(error);
            }
            return Ok(None);
        }

        Ok(Some(KnowledgeContext {
            snippets,
            searches: self.searches,
            embedding_ms: self.embedding_ms,
            search_ms: self.search_ms,
            final_wait_ms,
        }))
    }

    fn dispatch(&mut self, query: &str) -> Option<u64> {
        let search_id = self.next_search_id;
        self.next_search_id += 1;
        let request = KnowledgeSearchRequest {
            search_id,
            turn_id: self.turn_id,
            query: query.to_owned(),
            top_k: self.config.top_k,
        };
        if self.requests.send(request).is_err() {
            self.last_error = Some("knowledge worker request channel closed".to_owned());
            return None;
        }
        self.searches += 1;
        self.last_dispatch_at = Some(Instant::now());
        self.last_query = query.to_owned();
        self.last_search_id = Some(search_id);
        Some(search_id)
    }

    fn drain_ready(&mut self) {
        while let Ok(result) = self.results.try_recv() {
            self.accept_result(result);
        }
    }

    fn accept_result(&mut self, result: KnowledgeSearchResult) {
        if result.turn_id != self.turn_id {
            return;
        }
        match result.report {
            Ok(report) => {
                self.embedding_ms += report.embedding.as_millis() as u64;
                self.search_ms += report.search.as_millis() as u64;
                let mut ids = Vec::new();
                for hit in report
                    .hits
                    .into_iter()
                    .filter(|hit| hit.combined_score >= self.config.min_score)
                {
                    ids.push(hit.id.clone());
                    let snippet = KnowledgeSnippet {
                        id: hit.id.clone(),
                        source: hit.source.display().to_string(),
                        heading: hit.heading,
                        text: hit.text,
                        score: hit.combined_score,
                    };
                    self.accumulated
                        .entry(hit.id)
                        .and_modify(|current| {
                            if snippet.score > current.score {
                                current.clone_from(&snippet);
                            }
                        })
                        .or_insert(snippet);
                }
                self.completed.insert(result.search_id, ids);
                if self.config.debug {
                    debug!(
                        module = "knowledge",
                        event = "prefetch_completed",
                        search_id = result.search_id,
                        turn_id = result.turn_id,
                        query = %result.query,
                        hits = self.completed[&result.search_id].len(),
                        "local knowledge prefetch completed"
                    );
                }
            }
            Err(error) => self.last_error = Some(error),
        }
    }

    fn reset(&mut self) {
        self.turn_id += 1;
        self.last_dispatch_at = None;
        self.last_query.clear();
        self.last_search_id = None;
        self.accumulated.clear();
        self.completed.clear();
        self.searches = 0;
        self.embedding_ms = 0;
        self.search_ms = 0;
        self.last_error = None;
        self.drain_ready();
    }
}

const DANGLING_FINAL_WORDS: &[&str] = &[
    "а",
    "без",
    "в",
    "во",
    "для",
    "за",
    "если",
    "и",
    "из",
    "или",
    "к",
    "ко",
    "которого",
    "котором",
    "которой",
    "которому",
    "которым",
    "которыми",
    "которую",
    "которые",
    "которых",
    "который",
    "которая",
    "которое",
    "какая",
    "какие",
    "каким",
    "какими",
    "какого",
    "какое",
    "какой",
    "какую",
    "каких",
    "либо",
    "между",
    "на",
    "над",
    "не",
    "ни",
    "но",
    "о",
    "об",
    "обо",
    "от",
    "перед",
    "по",
    "под",
    "пока",
    "потому",
    "при",
    "про",
    "с",
    "со",
    "у",
    "хотя",
    "через",
    "что",
    "чего",
    "чем",
    "чтобы",
];

const DANGLING_FINAL_PHRASES: &[&[&str]] = &[
    &["в", "каких"],
    &["в", "каком"],
    &["для", "того", "чтобы"],
    &["за", "счет"],
    &["за", "счёт"],
    &["и", "какая"],
    &["и", "какие"],
    &["и", "какой"],
    &["и", "какую"],
    &["если", "у", "нас"],
    &["потому", "что"],
];

const REQUEST_WORDS: &[&str] = &[
    "объясни",
    "объясните",
    "опиши",
    "опишите",
    "покажи",
    "покажите",
    "расскажи",
    "расскажите",
    "сравни",
    "сравните",
];

const QUESTION_WORDS: &[&str] = &[
    "где",
    "зачем",
    "как",
    "какая",
    "какие",
    "каким",
    "какими",
    "какого",
    "какое",
    "какой",
    "какую",
    "каких",
    "когда",
    "почему",
    "сколько",
    "чего",
    "чем",
    "что",
];

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum BoundaryDeferral {
    DanglingSuffix,
    Introduction,
    Setup,
    ShortFragment,
}

impl BoundaryDeferral {
    fn reason(self) -> &'static str {
        match self {
            Self::DanglingSuffix => "dangling_suffix",
            Self::Introduction => "introduction",
            Self::Setup => "setup",
            Self::ShortFragment => "short_fragment",
        }
    }
}

#[derive(Clone, Copy)]
enum TranscriptCommand {
    SetPaused(bool),
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
        None,
    )
    .await
}

async fn run_transcript_windows_with_retrieval(
    mut events: mpsc::UnboundedReceiver<DeepgramEvent>,
    llm_requests: LlmRequestSender,
    output: mpsc::UnboundedSender<OutputEvent>,
    mut commands: mpsc::UnboundedReceiver<TranscriptCommand>,
    config: TranscriptConfig,
    readiness: Option<watch::Receiver<bool>>,
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
                            send_transcript_draft(&assembler, &output)?;
                        }
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
                        )?;

                    if has_transcript {
                        fallback
                            .as_mut()
                            .reset(Instant::now() + fallback_duration);
                        interim_fallback_deferred = false;
                        fallback_armed = true;
                        if let Some(retrieval) = retrieval.as_mut() {
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
) -> Result<Option<&'static str>, TranscriptWorkerError> {
    let mut flush_reason = None;
    match event {
        DeepgramEvent::Status(status) => handle_stt_status(status, output)?,
        DeepgramEvent::AudioStreamStarted => {
            send_output(
                output,
                OutputEvent::SttObservation(SttObservation::AudioStreamStarted),
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
                OutputEvent::SttObservation(SttObservation::Transcript {
                    text: text.clone(),
                    is_final,
                    speech_final,
                    speech_final_deferred,
                    audio_start_ms,
                    audio_duration_ms,
                    last_word_end_ms,
                }),
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
            send_transcript_draft(assembler, output)?;
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
                OutputEvent::SttObservation(SttObservation::SpeechStarted { audio_timestamp_ms }),
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
                OutputEvent::SttObservation(SttObservation::UtteranceEnd {
                    last_word_end_ms,
                    ignored,
                    deferred,
                }),
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
    Ok(flush_reason)
}

fn boundary_deferral(text: &str) -> Option<BoundaryDeferral> {
    let words = normalized_words(text);
    if words.is_empty() || ends_with_question_mark(text) {
        return None;
    }
    if DANGLING_FINAL_PHRASES
        .iter()
        .any(|suffix| ends_with_words(&words, suffix))
        || words.last().is_some_and(|word| {
            DANGLING_FINAL_WORDS
                .iter()
                .any(|candidate| word == candidate)
        })
    {
        return Some(BoundaryDeferral::DanglingSuffix);
    }
    if is_introduction(&words) && !has_request_word(&words) {
        return Some(BoundaryDeferral::Introduction);
    }
    if is_setup(&words) && !has_request_word(&words) {
        return Some(BoundaryDeferral::Setup);
    }
    if words.len() <= 6 && !has_question_signal(&words) {
        return Some(BoundaryDeferral::ShortFragment);
    }
    None
}

fn normalized_words(text: &str) -> Vec<String> {
    text.split(|character: char| !character.is_alphanumeric() && character != '-')
        .filter(|word| !word.is_empty())
        .map(str::to_lowercase)
        .collect()
}

fn ends_with_words(words: &[String], suffix: &[&str]) -> bool {
    words.len() >= suffix.len()
        && words[words.len() - suffix.len()..]
            .iter()
            .zip(suffix)
            .all(|(word, candidate)| word == candidate)
}

fn ends_with_question_mark(text: &str) -> bool {
    text.trim_end_matches(|character: char| {
        character.is_whitespace() || matches!(character, '"' | '\'' | ')' | ']')
    })
    .ends_with('?')
}

fn is_setup(words: &[String]) -> bool {
    words
        .iter()
        .any(|word| matches!(word.as_str(), "допустим" | "предположим" | "представим"))
        || contains_words(words, &["у", "нас", "есть"])
}

fn is_introduction(words: &[String]) -> bool {
    words.iter().any(|word| word == "давайте")
        && words
            .iter()
            .any(|word| matches!(word.as_str(), "начнем" | "начнём" | "начнемте" | "начнёмте"))
}

fn contains_words(words: &[String], needle: &[&str]) -> bool {
    words.len() >= needle.len()
        && words
            .windows(needle.len())
            .any(|window| ends_with_words(window, needle))
}

fn has_request_word(words: &[String]) -> bool {
    words
        .iter()
        .any(|word| REQUEST_WORDS.iter().any(|candidate| word == candidate))
}

fn has_question_signal(words: &[String]) -> bool {
    has_request_word(words)
        || words
            .iter()
            .any(|word| QUESTION_WORDS.iter().any(|candidate| word == candidate))
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

async fn flush_transcript_utterance(
    assembler: &mut TranscriptWindowAssembler,
    llm_requests: &LlmRequestSender,
    output: &mpsc::UnboundedSender<OutputEvent>,
    stats: &mut TranscriptStats,
    reason: &'static str,
    retrieval: &mut Option<RetrievalPipeline>,
) -> Result<(), TranscriptWorkerError> {
    queue_transcript_chunk(
        assembler.finish(),
        llm_requests,
        output,
        stats,
        reason,
        retrieval,
    )
    .await?;
    send_transcript_draft(assembler, output)
}

fn send_transcript_draft(
    assembler: &TranscriptWindowAssembler,
    output: &mpsc::UnboundedSender<OutputEvent>,
) -> Result<(), TranscriptWorkerError> {
    send_output(
        output,
        OutputEvent::TranscriptDraft {
            text: assembler.preview(),
        },
    )
}

async fn finish_transcript_window(
    assembler: &mut TranscriptWindowAssembler,
    llm_requests: &LlmRequestSender,
    output: &mpsc::UnboundedSender<OutputEvent>,
    stats: &mut TranscriptStats,
    retrieval: &mut Option<RetrievalPipeline>,
) -> Result<(), TranscriptWorkerError> {
    queue_transcript_chunk(
        assembler.finish(),
        llm_requests,
        output,
        stats,
        "shutdown",
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
    retrieval: &mut Option<RetrievalPipeline>,
) -> Result<(), TranscriptWorkerError> {
    let Some(chunk) = chunk else {
        return Ok(());
    };

    let request_id = chunk.sequence;
    info!(
        module = "transcript",
        event = "utterance_flushed",
        sequence = request_id,
        reason,
        text = %chunk.text,
        "TRANSCRIPT UTTERANCE"
    );
    send_output(
        output,
        OutputEvent::Transcript(TranscriptView {
            sequence: request_id,
            text: chunk.text.clone(),
            flush_reason: reason.to_owned(),
        }),
    )?;
    let knowledge = if let Some(retrieval) = retrieval.as_mut() {
        match retrieval.resolve(&chunk.text).await {
            Ok(context) => {
                if let Some(context) = context.as_ref() {
                    info!(
                        module = "knowledge",
                        event = "context_attached",
                        request_id,
                        searches = context.searches,
                        snippets = context.snippets.len(),
                        embedding_ms = context.embedding_ms,
                        search_ms = context.search_ms,
                        final_wait_ms = context.final_wait_ms,
                        "local knowledge context attached"
                    );
                    send_output(
                        output,
                        OutputEvent::Retrieval(RetrievalView {
                            request_id,
                            context: context.clone(),
                        }),
                    )?;
                }
                context
            }
            Err(error) => {
                warn!(
                    module = "knowledge",
                    event = "search_failed",
                    request_id,
                    error = %error,
                    "local knowledge search failed"
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
            mode: Mode::Voice,
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

    #[error("STT worker stopped before benchmark transcript timing could start")]
    SttReadinessClosed,
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

async fn finish_knowledge_thread(
    thread: Option<
        ThreadJoinHandle<Result<KnowledgeWorkerStats, crate::knowledge::KnowledgeError>>,
    >,
) -> KnowledgeWorkerStats {
    let Some(thread) = thread else {
        return KnowledgeWorkerStats::default();
    };
    match tokio::task::spawn_blocking(move || thread.join()).await {
        Ok(Ok(Ok(stats))) => stats,
        Ok(Ok(Err(error))) => {
            warn!(
                module = "knowledge",
                event = "worker_failed",
                error = %error,
                "local knowledge worker failed"
            );
            KnowledgeWorkerStats {
                searches: 0,
                failed: 1,
            }
        }
        Ok(Err(_)) => {
            warn!(
                module = "knowledge",
                event = "worker_panicked",
                "local knowledge worker panicked"
            );
            KnowledgeWorkerStats {
                searches: 0,
                failed: 1,
            }
        }
        Err(error) => {
            warn!(
                module = "knowledge",
                event = "worker_join_failed",
                error = %error,
                "could not join local knowledge worker"
            );
            KnowledgeWorkerStats {
                searches: 0,
                failed: 1,
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
            Some(OutputEvent::SttObservation(SttObservation::Transcript {
                text,
                is_final: true,
                speech_final: true,
                speech_final_deferred: false,
                audio_start_ms: Some(100),
                audio_duration_ms: Some(900),
                last_word_end_ms: Some(900),
            })) if text == "Что такое HashMap?"
        ));
        assert!(matches!(
            output_receiver.recv().await,
            Some(OutputEvent::TranscriptDraft { text }) if text == "Что такое HashMap?"
        ));
        assert!(matches!(
            output_receiver.recv().await,
            Some(OutputEvent::Transcript(TranscriptView {
                sequence: 0,
                text,
                flush_reason,
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
                OutputEvent::SttObservation(SttObservation::Transcript {
                    speech_final_deferred,
                    ..
                }) => deferred |= speech_final_deferred,
                OutputEvent::SttObservation(SttObservation::UtteranceEnd { deferred, .. }) => {
                    utterance_end_deferred |= deferred
                }
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
                OutputEvent::SttObservation(SttObservation::UtteranceEnd {
                    last_word_end_ms: Some(3_100),
                    ignored: true,
                    deferred: false,
                })
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
        )
        .expect("growing interim transcript must be forwarded");

        assert_eq!(
            output_receiver.try_recv(),
            Ok(OutputEvent::SttObservation(SttObservation::Transcript {
                text: "что такое".to_owned(),
                is_final: false,
                speech_final: false,
                speech_final_deferred: false,
                audio_start_ms: None,
                audio_duration_ms: None,
                last_word_end_ms: None,
            }))
        );
        assert_eq!(
            output_receiver.try_recv(),
            Ok(OutputEvent::TranscriptDraft {
                text: "что такое".to_owned(),
            })
        );
        assert_eq!(
            output_receiver.try_recv(),
            Ok(OutputEvent::SttObservation(SttObservation::Transcript {
                text: "что такое HashMap".to_owned(),
                is_final: false,
                speech_final: false,
                speech_final_deferred: false,
                audio_start_ms: None,
                audio_duration_ms: None,
                last_word_end_ms: None,
            }))
        );
        assert_eq!(
            output_receiver.try_recv(),
            Ok(OutputEvent::TranscriptDraft {
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

    #[tokio::test]
    async fn retrieval_accumulates_interim_hits_and_prioritizes_final_query() {
        let (request_sender, mut request_receiver) = mpsc::unbounded_channel();
        let (result_sender, result_receiver) = mpsc::unbounded_channel();
        let (_readiness_sender, readiness_receiver) = watch::channel(true);
        let mut retrieval = RetrievalPipeline::new(
            KnowledgeConfig {
                enabled: true,
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
                    0.80,
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
            .expect("retrieval must succeed")
            .expect("context must be selected");

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
            model_load: Duration::ZERO,
            embedding: Duration::from_millis(8),
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
