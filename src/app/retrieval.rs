use std::{collections::HashMap, time::Duration};

use tokio::{
    sync::{mpsc, watch},
    time::{Instant, sleep},
};
use tracing::debug;

use crate::{
    config::KnowledgeConfig,
    events::{KnowledgeContext, KnowledgeSnippet},
    knowledge::{KnowledgeSearchRequest, KnowledgeSearchResult},
};

const ACCUMULATED_SCORE_MARGIN: f32 = 0.08;

pub(super) struct RetrievalPipeline {
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
    embedding_calls: u64,
    embedding_prompt_tokens: u64,
    embedding_total_tokens: u64,
    embedding_ms: u64,
    search_ms: u64,
    last_error: Option<String>,
}

impl RetrievalPipeline {
    pub(super) fn new(
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
            embedding_calls: 0,
            embedding_prompt_tokens: 0,
            embedding_total_tokens: 0,
            embedding_ms: 0,
            search_ms: 0,
            last_error: None,
        }
    }

    pub(super) async fn wait_until_ready(&mut self) {
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

    pub(super) fn prefetch(&mut self, query: &str, force: bool) {
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

    pub(super) async fn resolve(&mut self, query: &str) -> Result<KnowledgeContext, String> {
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
            if snippet.score >= self.config.min_score + ACCUMULATED_SCORE_MARGIN
                && !ordered_ids.contains(&snippet.id)
            {
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

        if snippets.is_empty()
            && let Some(error) = self.last_error.take()
        {
            return Err(error);
        }

        Ok(KnowledgeContext {
            snippets,
            searches: self.searches,
            embedding_calls: self.embedding_calls,
            embedding_prompt_tokens: self.embedding_prompt_tokens,
            embedding_total_tokens: self.embedding_total_tokens,
            embedding_ms: self.embedding_ms,
            search_ms: self.search_ms,
            final_wait_ms,
        })
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
                self.embedding_calls += report.embedding_calls;
                self.embedding_prompt_tokens += report.prompt_tokens;
                self.embedding_total_tokens += report.total_tokens;
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
                        "remote knowledge prefetch completed"
                    );
                }
            }
            Err(error) => self.last_error = Some(error),
        }
    }

    pub(super) fn reset(&mut self) {
        self.turn_id += 1;
        self.last_dispatch_at = None;
        self.last_query.clear();
        self.last_search_id = None;
        self.accumulated.clear();
        self.completed.clear();
        self.searches = 0;
        self.embedding_calls = 0;
        self.embedding_prompt_tokens = 0;
        self.embedding_total_tokens = 0;
        self.embedding_ms = 0;
        self.search_ms = 0;
        self.last_error = None;
        self.drain_ready();
    }
}
