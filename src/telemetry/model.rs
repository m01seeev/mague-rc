use serde::Serialize;

use crate::{config::Config, events::LlmUsage};

#[derive(Serialize)]
pub(super) struct RunSummary {
    pub(super) schema_version: u32,
    pub(super) metadata: RunMetadata,
    pub(super) elapsed_ms: u64,
    pub(super) event_count: u64,
    pub(super) audio_stream_started_ms: Option<u64>,
    pub(super) stt: SttMetrics,
    pub(super) stt_latency: SttLatencySummary,
    pub(super) aggregates: AggregateSummary,
    pub(super) requests: Vec<RequestSummary>,
    pub(super) utterances: Vec<UtteranceSummary>,
    pub(super) live_coding: Vec<LiveCodingRevision>,
    pub(super) recognition_segments: Vec<String>,
    pub(super) accuracy: Option<AccuracySummary>,
}

#[derive(Serialize)]
pub(super) struct RunMetadata {
    pub(super) schema_version: u32,
    pub(super) run_id: String,
    pub(super) label: String,
    pub(super) run_kind: String,
    pub(super) started_unix_ms: u64,
    pub(super) application_version: String,
    pub(super) git_branch: Option<String>,
    pub(super) git_commit: Option<String>,
    pub(super) git_dirty: bool,
    pub(super) audio_file: Option<String>,
    pub(super) audio_bytes: Option<u64>,
    pub(super) audio_sha256: Option<String>,
    pub(super) audio_duration_ms: Option<u64>,
    pub(super) reference_file: Option<String>,
    pub(super) reference_sha256: Option<String>,
    pub(super) configuration: ConfigurationSnapshot,
}

#[derive(Serialize)]
pub(super) struct LiveCodingRevision {
    pub(super) request_id: Option<u64>,
    pub(super) speaker: Option<String>,
    pub(super) revision: u64,
    pub(super) summary: String,
    pub(super) candidate_context: String,
    pub(super) explanation: String,
    pub(super) language: String,
    pub(super) code: String,
    pub(super) change_note: String,
    pub(super) changed_lines: Vec<usize>,
}

#[derive(Serialize)]
pub(super) struct ConfigurationSnapshot {
    pub(super) deepgram_model: String,
    pub(super) deepgram_language: String,
    pub(super) sample_rate: u32,
    pub(super) channels: u16,
    pub(super) interim_results: bool,
    pub(super) punctuate: bool,
    pub(super) smart_format: bool,
    pub(super) vad_events: bool,
    pub(super) endpointing_ms: u64,
    pub(super) utterance_end_ms: u64,
    pub(super) keyterms: Vec<String>,
    pub(super) audio_chunk_ms: u64,
    pub(super) candidate_mic_enabled: bool,
    pub(super) candidate_mic_source: String,
    pub(super) transcript_segmentation: &'static str,
    pub(super) transcript_window_sec: u64,
    pub(super) min_utterance_chars: usize,
    pub(super) llm_model: String,
    pub(super) max_history_pairs: usize,
    pub(super) separate_histories: bool,
    pub(super) temperature: f32,
    pub(super) max_tokens: u32,
    pub(super) timeout_sec: u64,
    pub(super) coding_temperature: f32,
    pub(super) coding_max_tokens: u32,
    pub(super) coding_timeout_sec: u64,
    pub(super) rag_enabled: bool,
    pub(super) rag_embedding_provider: &'static str,
    pub(super) rag_embedding_model: String,
    pub(super) rag_embedding_dimensions: usize,
    pub(super) rag_embedding_query_input_type: String,
    pub(super) rag_embedding_document_input_type: String,
    pub(super) rag_embedding_timeout_sec: u64,
    pub(super) rag_top_k: usize,
    pub(super) rag_max_context_chars: usize,
    pub(super) rag_min_score: f32,
    pub(super) rag_refresh_ms: u64,
    pub(super) rag_final_wait_ms: u64,
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
            candidate_mic_enabled: config.candidate_audio.enabled,
            candidate_mic_source: config.candidate_audio.source.clone(),
            transcript_segmentation: "deepgram_utterance_with_inactivity_fallback",
            transcript_window_sec: config.transcript.window_sec,
            min_utterance_chars: config.transcript.min_utterance_chars,
            llm_model: config.llm.model.clone(),
            max_history_pairs: config.llm.max_history_pairs,
            separate_histories: config.llm.separate_histories,
            temperature: config.llm.temperature,
            max_tokens: config.llm.max_tokens,
            timeout_sec: config.llm.timeout_sec,
            coding_temperature: config.llm.coding_temperature,
            coding_max_tokens: config.llm.coding_max_tokens,
            coding_timeout_sec: config.llm.coding_timeout_sec,
            rag_enabled: config.knowledge.enabled,
            rag_embedding_provider: "openrouter",
            rag_embedding_model: config.knowledge.embedding.model.clone(),
            rag_embedding_dimensions: config.knowledge.embedding.dimensions,
            rag_embedding_query_input_type: config.knowledge.embedding.query_input_type.clone(),
            rag_embedding_document_input_type: config
                .knowledge
                .embedding
                .document_input_type
                .clone(),
            rag_embedding_timeout_sec: config.knowledge.embedding.timeout_sec,
            rag_top_k: config.knowledge.top_k,
            rag_max_context_chars: config.knowledge.max_context_chars,
            rag_min_score: config.knowledge.min_score,
            rag_refresh_ms: config.knowledge.refresh_ms,
            rag_final_wait_ms: config.knowledge.final_wait_ms,
        }
    }
}

#[derive(Default)]
pub(super) struct RequestMetrics {
    pub(super) mode: String,
    pub(super) speaker: String,
    pub(super) question: String,
    pub(super) flush_reason: String,
    pub(super) answer: String,
    pub(super) draft_started_ms: Option<u64>,
    pub(super) last_final_ms: Option<u64>,
    pub(super) speech_final_ms: Option<u64>,
    pub(super) utterance_end_ms: Option<u64>,
    pub(super) last_word_end_audio_ms: Option<u64>,
    pub(super) last_word_to_boundary_ms: Option<u64>,
    pub(super) question_ready_ms: Option<u64>,
    pub(super) llm_queued_ms: Option<u64>,
    pub(super) llm_started_ms: Option<u64>,
    pub(super) first_token_ms: Option<u64>,
    pub(super) completed_ms: Option<u64>,
    pub(super) failed_ms: Option<u64>,
    pub(super) answer_chars: u64,
    pub(super) usage: Option<LlmUsage>,
    pub(super) retrieval: Option<RetrievalMetrics>,
}

#[derive(Serialize)]
pub(super) struct RetrievalMetrics {
    pub(super) searches: u64,
    pub(super) embedding_calls: u64,
    pub(super) embedding_prompt_tokens: u64,
    pub(super) embedding_total_tokens: u64,
    pub(super) snippets: usize,
    pub(super) context_chars: usize,
    pub(super) embedding_ms: u64,
    pub(super) search_ms: u64,
    pub(super) final_wait_ms: u64,
    pub(super) hits: Vec<RetrievalHitMetrics>,
}

#[derive(Serialize)]
pub(super) struct RetrievalHitMetrics {
    pub(super) id: String,
    pub(super) source: String,
    pub(super) heading: String,
    pub(super) text: String,
    pub(super) score: f32,
}

#[derive(Serialize)]
pub(super) struct RequestSummary {
    pub(super) request_id: u64,
    pub(super) mode: String,
    pub(super) speaker: String,
    pub(super) question: String,
    pub(super) flush_reason: String,
    pub(super) answer: String,
    pub(super) question_chars: usize,
    pub(super) question_words: usize,
    pub(super) answer_chars: u64,
    pub(super) draft_started_ms: Option<u64>,
    pub(super) last_final_ms: Option<u64>,
    pub(super) speech_final_ms: Option<u64>,
    pub(super) utterance_end_ms: Option<u64>,
    pub(super) last_word_end_audio_ms: Option<u64>,
    pub(super) last_word_to_boundary_ms: Option<u64>,
    pub(super) question_ready_ms: Option<u64>,
    pub(super) llm_queued_ms: Option<u64>,
    pub(super) llm_started_ms: Option<u64>,
    pub(super) first_token_ms: Option<u64>,
    pub(super) completed_ms: Option<u64>,
    pub(super) failed_ms: Option<u64>,
    pub(super) build_ms: Option<u64>,
    pub(super) retrieval_pipeline_ms: Option<u64>,
    pub(super) final_to_queue_ms: Option<u64>,
    pub(super) speech_final_to_queue_ms: Option<u64>,
    pub(super) utterance_end_to_queue_ms: Option<u64>,
    pub(super) queue_wait_ms: Option<u64>,
    pub(super) ttft_ms: Option<u64>,
    pub(super) queued_to_first_token_ms: Option<u64>,
    pub(super) speech_boundary_to_first_token_ms: Option<u64>,
    pub(super) last_word_to_first_token_ms: Option<u64>,
    pub(super) generation_ms: Option<u64>,
    pub(super) total_request_ms: Option<u64>,
    pub(super) retrieval: Option<RetrievalMetrics>,
    pub(super) usage: Option<LlmUsageSummary>,
}

impl RequestSummary {
    pub(super) fn from_metrics(request_id: u64, metrics: RequestMetrics) -> Self {
        let speech_boundary_ms = metrics.utterance_end_ms.or(metrics.speech_final_ms);
        let speech_boundary_to_first_token_ms =
            duration(metrics.first_token_ms, speech_boundary_ms);
        let last_word_to_first_token_ms = speech_boundary_to_first_token_ms
            .zip(metrics.last_word_to_boundary_ms)
            .and_then(|(after_boundary_ms, before_boundary_ms)| {
                after_boundary_ms.checked_add(before_boundary_ms)
            });
        Self {
            request_id,
            mode: metrics.mode,
            speaker: metrics.speaker,
            question_chars: metrics.question.chars().count(),
            question_words: word_count(&metrics.question),
            answer_chars: metrics.answer_chars,
            build_ms: duration(metrics.question_ready_ms, metrics.draft_started_ms),
            retrieval_pipeline_ms: duration(metrics.llm_queued_ms, metrics.question_ready_ms),
            final_to_queue_ms: duration(metrics.llm_queued_ms, metrics.last_final_ms),
            speech_final_to_queue_ms: duration(metrics.llm_queued_ms, metrics.speech_final_ms),
            utterance_end_to_queue_ms: duration(metrics.llm_queued_ms, metrics.utterance_end_ms),
            queue_wait_ms: duration(metrics.llm_started_ms, metrics.llm_queued_ms),
            ttft_ms: duration(metrics.first_token_ms, metrics.llm_started_ms),
            queued_to_first_token_ms: duration(metrics.first_token_ms, metrics.llm_queued_ms),
            speech_boundary_to_first_token_ms,
            last_word_to_first_token_ms,
            generation_ms: duration(metrics.completed_ms, metrics.first_token_ms),
            total_request_ms: duration(metrics.completed_ms, metrics.llm_started_ms),
            question: metrics.question,
            flush_reason: metrics.flush_reason,
            answer: metrics.answer,
            draft_started_ms: metrics.draft_started_ms,
            last_final_ms: metrics.last_final_ms,
            speech_final_ms: metrics.speech_final_ms,
            utterance_end_ms: metrics.utterance_end_ms,
            last_word_end_audio_ms: metrics.last_word_end_audio_ms,
            last_word_to_boundary_ms: metrics.last_word_to_boundary_ms,
            question_ready_ms: metrics.question_ready_ms,
            llm_queued_ms: metrics.llm_queued_ms,
            llm_started_ms: metrics.llm_started_ms,
            first_token_ms: metrics.first_token_ms,
            completed_ms: metrics.completed_ms,
            failed_ms: metrics.failed_ms,
            retrieval: metrics.retrieval,
            usage: metrics.usage.map(LlmUsageSummary::from),
        }
    }
}

#[derive(Serialize)]
pub(super) struct LlmUsageSummary {
    pub(super) prompt_tokens: u64,
    pub(super) completion_tokens: u64,
    pub(super) total_tokens: u64,
    pub(super) cost: Option<f64>,
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
pub(super) struct SttMetrics {
    pub(super) transcripts: u64,
    pub(super) interim_transcripts: u64,
    pub(super) final_transcripts: u64,
    pub(super) speech_final_transcripts: u64,
    pub(super) deferred_speech_final_transcripts: u64,
    pub(super) speech_started: u64,
    pub(super) utterance_end: u64,
    pub(super) ignored_utterance_end: u64,
    pub(super) deferred_utterance_end: u64,
}

#[derive(Serialize)]
pub(super) struct AggregateSummary {
    pub(super) request_count: usize,
    pub(super) completed_count: usize,
    pub(super) failed_count: usize,
    pub(super) prompt_tokens: u64,
    pub(super) completion_tokens: u64,
    pub(super) total_tokens: u64,
    pub(super) total_cost: f64,
    pub(super) rag_request_count: usize,
    pub(super) rag_searches: u64,
    pub(super) rag_embedding_calls: u64,
    pub(super) rag_embedding_prompt_tokens: u64,
    pub(super) rag_embedding_total_tokens: u64,
    pub(super) rag_snippets: usize,
    pub(super) rag_context_chars: usize,
    pub(super) build_ms: MetricSummary,
    pub(super) retrieval_pipeline_ms: MetricSummary,
    pub(super) rag_embedding_ms: MetricSummary,
    pub(super) rag_search_ms: MetricSummary,
    pub(super) rag_final_wait_ms: MetricSummary,
    pub(super) queue_wait_ms: MetricSummary,
    pub(super) ttft_ms: MetricSummary,
    pub(super) queued_to_first_token_ms: MetricSummary,
    pub(super) speech_boundary_to_first_token_ms: MetricSummary,
    pub(super) last_word_to_first_token_ms: MetricSummary,
    pub(super) generation_ms: MetricSummary,
    pub(super) total_request_ms: MetricSummary,
}

impl AggregateSummary {
    pub(super) fn new(requests: &[RequestSummary]) -> Self {
        let usage = requests
            .iter()
            .filter_map(|request| request.usage.as_ref())
            .collect::<Vec<_>>();
        let retrieval = requests
            .iter()
            .filter_map(|request| request.retrieval.as_ref())
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
            rag_request_count: retrieval.len(),
            rag_searches: retrieval.iter().map(|retrieval| retrieval.searches).sum(),
            rag_embedding_calls: retrieval
                .iter()
                .map(|retrieval| retrieval.embedding_calls)
                .sum(),
            rag_embedding_prompt_tokens: retrieval
                .iter()
                .map(|retrieval| retrieval.embedding_prompt_tokens)
                .sum(),
            rag_embedding_total_tokens: retrieval
                .iter()
                .map(|retrieval| retrieval.embedding_total_tokens)
                .sum(),
            rag_snippets: retrieval.iter().map(|retrieval| retrieval.snippets).sum(),
            rag_context_chars: retrieval
                .iter()
                .map(|retrieval| retrieval.context_chars)
                .sum(),
            build_ms: metric_summary(requests.iter().filter_map(|request| request.build_ms)),
            retrieval_pipeline_ms: metric_summary(
                requests
                    .iter()
                    .filter_map(|request| request.retrieval_pipeline_ms),
            ),
            rag_embedding_ms: metric_summary(
                retrieval.iter().map(|retrieval| retrieval.embedding_ms),
            ),
            rag_search_ms: metric_summary(retrieval.iter().map(|retrieval| retrieval.search_ms)),
            rag_final_wait_ms: metric_summary(
                retrieval.iter().map(|retrieval| retrieval.final_wait_ms),
            ),
            queue_wait_ms: metric_summary(
                requests.iter().filter_map(|request| request.queue_wait_ms),
            ),
            ttft_ms: metric_summary(requests.iter().filter_map(|request| request.ttft_ms)),
            queued_to_first_token_ms: metric_summary(
                requests
                    .iter()
                    .filter_map(|request| request.queued_to_first_token_ms),
            ),
            speech_boundary_to_first_token_ms: metric_summary(
                requests
                    .iter()
                    .filter_map(|request| request.speech_boundary_to_first_token_ms),
            ),
            last_word_to_first_token_ms: metric_summary(
                requests
                    .iter()
                    .filter_map(|request| request.last_word_to_first_token_ms),
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
pub(super) struct SttLatencyMetrics {
    pub(super) interim_delivery_lag_ms: Vec<u64>,
    pub(super) final_delivery_lag_ms: Vec<u64>,
    pub(super) speech_started_delivery_lag_ms: Vec<u64>,
    pub(super) utterance_end_delivery_lag_ms: Vec<u64>,
}

#[derive(Serialize)]
pub(super) struct SttLatencySummary {
    pub(super) approximation: &'static str,
    pub(super) interim_delivery_lag_ms: MetricSummary,
    pub(super) final_delivery_lag_ms: MetricSummary,
    pub(super) speech_started_delivery_lag_ms: MetricSummary,
    pub(super) utterance_end_delivery_lag_ms: MetricSummary,
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
pub(super) struct UtteranceMetrics {
    pub(super) sequence: u64,
    pub(super) speech_started_ms: Option<u64>,
    pub(super) first_transcript_ms: Option<u64>,
    pub(super) first_interim_ms: Option<u64>,
    pub(super) first_final_ms: Option<u64>,
    pub(super) speech_final_ms: Option<u64>,
    pub(super) utterance_end_ms: Option<u64>,
    pub(super) final_text: String,
}

#[derive(Serialize)]
pub(super) struct UtteranceSummary {
    pub(super) sequence: u64,
    pub(super) final_text: String,
    pub(super) speech_started_ms: Option<u64>,
    pub(super) first_transcript_ms: Option<u64>,
    pub(super) first_interim_ms: Option<u64>,
    pub(super) first_final_ms: Option<u64>,
    pub(super) speech_final_ms: Option<u64>,
    pub(super) utterance_end_ms: Option<u64>,
    pub(super) speech_to_first_interim_ms: Option<u64>,
    pub(super) speech_to_first_final_ms: Option<u64>,
    pub(super) utterance_duration_ms: Option<u64>,
}

#[derive(Serialize)]
pub(super) struct AccuracySummary {
    pub(super) reference_text: String,
    pub(super) recognized_text: String,
    pub(super) word_errors: usize,
    pub(super) reference_words: usize,
    pub(super) wer: Option<f64>,
    pub(super) character_errors: usize,
    pub(super) reference_characters: usize,
    pub(super) cer: Option<f64>,
    pub(super) segments: Vec<AccuracySegment>,
}

impl AccuracySummary {
    pub(super) fn new(reference: &str, recognized_segments: &[String]) -> Self {
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
pub(super) struct AccuracySegment {
    pub(super) sequence: usize,
    pub(super) reference: String,
    pub(super) recognized: String,
    pub(super) word_errors: usize,
    pub(super) reference_words: usize,
    pub(super) wer: Option<f64>,
    pub(super) character_errors: usize,
    pub(super) reference_characters: usize,
    pub(super) cer: Option<f64>,
}

impl AccuracySegment {
    pub(super) fn new(sequence: usize, reference: &str, recognized: &str) -> Self {
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

pub(super) struct AccuracyScores {
    pub(super) word_errors: usize,
    pub(super) reference_words: usize,
    pub(super) wer: Option<f64>,
    pub(super) character_errors: usize,
    pub(super) reference_characters: usize,
    pub(super) cer: Option<f64>,
}

impl AccuracyScores {
    pub(super) fn new(reference: &str, recognized: &str) -> Self {
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
pub(super) struct MetricSummary {
    pub(super) count: usize,
    pub(super) min: Option<u64>,
    pub(super) mean: Option<f64>,
    pub(super) p50: Option<u64>,
    pub(super) p95: Option<u64>,
    pub(super) max: Option<u64>,
}

pub(super) fn metric_summary(values: impl Iterator<Item = u64>) -> MetricSummary {
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

pub(super) fn percentile(values: &[u64], percentile: f64) -> Option<u64> {
    if values.is_empty() {
        return None;
    }
    let rank = (percentile * values.len() as f64).ceil() as usize;
    values.get(rank.saturating_sub(1)).copied()
}

pub(super) fn duration(later: Option<u64>, earlier: Option<u64>) -> Option<u64> {
    later?.checked_sub(earlier?)
}

pub(super) fn word_count(value: &str) -> usize {
    value.split_whitespace().count()
}

pub(super) fn append_text(target: &mut String, text: &str) {
    if !target.is_empty() && !text.starts_with(char::is_whitespace) {
        target.push(' ');
    }
    target.push_str(text);
}

pub(super) fn finish_recognition_segment(pending: &mut String, segments: &mut Vec<String>) {
    let text = pending.trim();
    if !text.is_empty() {
        segments.push(text.to_owned());
    }
    pending.clear();
}

pub(super) fn normalize_transcript(value: &str) -> String {
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

pub(super) fn edit_distance<T: Eq>(reference: &[T], recognized: &[T]) -> usize {
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

pub(super) fn error_rate(errors: usize, reference_units: usize) -> Option<f64> {
    (reference_units > 0).then(|| errors as f64 / reference_units as f64)
}
