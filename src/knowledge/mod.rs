use std::{
    collections::HashSet,
    env, fs, io,
    path::{Path, PathBuf},
    time::{Duration, Instant},
};

use pulldown_cmark::{Event, HeadingLevel, Options, Parser, Tag, TagEnd};
use serde::{Deserialize, Serialize};
use thiserror::Error;
use tokio::{
    sync::{mpsc, watch},
    task::JoinHandle,
};
use tracing::info;

use crate::config::EmbeddingConfig;

mod openrouter;

use openrouter::{EmbeddingPurpose, OpenRouterEmbeddingClient};

const SCHEMA_VERSION: u32 = 3;
const MAX_CHUNK_CHARS: usize = 1_200;
const EMBEDDING_BATCH_SIZE: usize = 32;
const LEXICAL_WEIGHT: f32 = 0.08;
const QUERY_INSTRUCTION: &str = "Given a partial or complete Russian spoken technical interview question about Java backend engineering, retrieve the most relevant passages from a personal knowledge base. The knowledge base may contain Russian text, English technical terms, code identifiers, and employment experience.";

#[derive(Debug, Error)]
pub enum KnowledgeError {
    #[error("{action} `{path}`: {source}")]
    Io {
        action: &'static str,
        path: PathBuf,
        #[source]
        source: io::Error,
    },

    #[error("knowledge source does not exist: {0}")]
    SourceMissing(PathBuf),

    #[error("no Markdown files found under knowledge source: {0}")]
    NoMarkdown(PathBuf),

    #[error("knowledge query must not be empty")]
    EmptyQuery,

    #[error("could not build OpenRouter embedding client: {0}")]
    EmbeddingClient(String),

    #[error("OpenRouter embedding request failed: {0}")]
    EmbeddingRequest(String),

    #[error("OpenRouter embeddings returned HTTP {status}: {message}")]
    EmbeddingHttpStatus { status: u16, message: String },

    #[error("could not parse OpenRouter embedding response: {0}")]
    EmbeddingProtocol(String),

    #[error("embedding dimension mismatch: expected {expected}, received {actual}")]
    EmbeddingDimension { expected: usize, actual: usize },

    #[error("could not encode knowledge index: {0}")]
    Encode(#[source] serde_json::Error),

    #[error("could not decode knowledge index `{path}`: {source}")]
    Decode {
        path: PathBuf,
        #[source]
        source: serde_json::Error,
    },

    #[error("knowledge index is missing at `{0}`; run `cargo run -- rag index knowledge/` first")]
    IndexMissing(PathBuf),

    #[error("knowledge index is incompatible: {0}; rebuild it with `rag index`")]
    IncompatibleIndex(String),
}

#[derive(Clone, Debug)]
pub struct KnowledgePaths {
    pub index_file: PathBuf,
}

impl KnowledgePaths {
    pub fn discover() -> Self {
        let cache_home = env::var_os("XDG_CACHE_HOME")
            .map(PathBuf::from)
            .or_else(|| env::var_os("HOME").map(|home| PathBuf::from(home).join(".cache")))
            .unwrap_or_else(|| env::temp_dir().join("mague-rc-cache"));
        let root = cache_home.join("mague-rc");
        Self {
            index_file: root.join("knowledge").join("index-v3.json"),
        }
    }
}

#[derive(Clone, Debug)]
pub struct IndexReport {
    pub source_count: usize,
    pub section_count: usize,
    pub chunk_count: usize,
    pub dimension: usize,
    pub embedding: Duration,
    pub embedding_calls: u64,
    pub prompt_tokens: u64,
    pub total_tokens: u64,
    pub total: Duration,
    pub index_path: PathBuf,
    pub index_bytes: u64,
}

#[derive(Clone, Debug)]
pub struct SearchHit {
    pub id: String,
    pub source: PathBuf,
    pub heading: String,
    pub text: String,
    pub dense_score: f32,
    pub lexical_score: f32,
    pub combined_score: f32,
}

#[derive(Clone, Debug)]
pub struct QueryReport {
    pub embedding: Duration,
    pub embedding_calls: u64,
    pub prompt_tokens: u64,
    pub total_tokens: u64,
    pub search: Duration,
    pub hits: Vec<SearchHit>,
    pub index_path: PathBuf,
}

#[derive(Debug)]
pub struct KnowledgeSearchRequest {
    pub search_id: u64,
    pub turn_id: u64,
    pub query: String,
    pub top_k: usize,
}

#[derive(Debug)]
pub struct KnowledgeSearchResult {
    pub search_id: u64,
    pub turn_id: u64,
    pub query: String,
    pub report: Result<QueryReport, String>,
}

#[derive(Default, Debug)]
pub struct KnowledgeWorkerStats {
    pub searches: u64,
    pub failed: u64,
    pub coalesced: u64,
}

pub struct KnowledgeWorkerRuntime {
    pub requests: mpsc::UnboundedSender<KnowledgeSearchRequest>,
    pub results: mpsc::UnboundedReceiver<KnowledgeSearchResult>,
    pub readiness: watch::Receiver<bool>,
    pub task: JoinHandle<Result<KnowledgeWorkerStats, KnowledgeError>>,
}

pub fn spawn_knowledge_worker(
    config: EmbeddingConfig,
) -> Result<KnowledgeWorkerRuntime, KnowledgeError> {
    let paths = KnowledgePaths::discover();
    if !paths.index_file.is_file() {
        return Err(KnowledgeError::IndexMissing(paths.index_file));
    }

    let mut retriever = RemoteRetriever::load_at(config, paths)?;
    retriever.load_index()?;
    let (request_sender, mut request_receiver) =
        mpsc::unbounded_channel::<KnowledgeSearchRequest>();
    let (result_sender, result_receiver) = mpsc::unbounded_channel();
    let (readiness_sender, readiness_receiver) = watch::channel(false);
    let model = retriever.config.model.clone();
    let dimensions = retriever.config.dimensions;
    let task = tokio::spawn(async move {
        let _ = readiness_sender.send(true);
        info!(
            module = "knowledge",
            event = "worker_ready",
            provider = "openrouter",
            model,
            dimensions,
            "remote knowledge worker ready"
        );
        let mut stats = KnowledgeWorkerStats::default();

        while let Some(mut request) = request_receiver.recv().await {
            while let Ok(newer_request) = request_receiver.try_recv() {
                request = newer_request;
                stats.coalesced += 1;
            }
            stats.searches += 1;
            let report = retriever
                .query(&request.query, request.top_k)
                .await
                .map_err(|error| {
                    stats.failed += 1;
                    error.to_string()
                });
            if result_sender
                .send(KnowledgeSearchResult {
                    search_id: request.search_id,
                    turn_id: request.turn_id,
                    query: request.query,
                    report,
                })
                .is_err()
            {
                break;
            }
        }
        Ok(stats)
    });

    Ok(KnowledgeWorkerRuntime {
        requests: request_sender,
        results: result_receiver,
        readiness: readiness_receiver,
        task,
    })
}

pub struct RemoteRetriever {
    client: OpenRouterEmbeddingClient,
    config: EmbeddingConfig,
    paths: KnowledgePaths,
    index: Option<KnowledgeIndex>,
}

impl RemoteRetriever {
    pub fn load(config: EmbeddingConfig) -> Result<Self, KnowledgeError> {
        Self::load_at(config, KnowledgePaths::discover())
    }

    pub fn load_at(config: EmbeddingConfig, paths: KnowledgePaths) -> Result<Self, KnowledgeError> {
        let client = OpenRouterEmbeddingClient::new(config.clone())?;
        Ok(Self {
            client,
            config,
            paths,
            index: None,
        })
    }

    fn load_index(&mut self) -> Result<(), KnowledgeError> {
        let index = read_index(&self.paths.index_file)?;
        validate_index(&index, &self.config)?;
        self.index = Some(index);
        Ok(())
    }

    pub async fn index(&mut self, source: &Path) -> Result<IndexReport, KnowledgeError> {
        let total_started = Instant::now();
        let files = markdown_files(source)?;
        let mut sources = Vec::with_capacity(files.len());
        let mut pending = Vec::new();
        let mut section_count = 0;

        for path in files {
            let bytes = read_file(&path)?;
            let markdown = String::from_utf8_lossy(&bytes);
            let sections = parse_markdown(&markdown);
            section_count += sections.len();
            let fingerprint = fingerprint(&bytes);
            sources.push(SourceMetadata {
                path: path.clone(),
                fingerprint: format!("{fingerprint:016x}"),
            });

            for (section_index, section) in sections.into_iter().enumerate() {
                for (chunk_index, text) in chunk_text(&section.text).into_iter().enumerate() {
                    pending.push(PendingChunk {
                        id: format!("{fingerprint:016x}-{section_index:04}-{chunk_index:03}"),
                        source: path.clone(),
                        heading: section.heading.clone(),
                        text,
                    });
                }
            }
        }

        let embedding_started = Instant::now();
        let mut embeddings = Vec::with_capacity(pending.len());
        let mut embedding_calls = 0_u64;
        let mut prompt_tokens = 0_u64;
        let mut total_tokens = 0_u64;
        for batch in pending.chunks(EMBEDDING_BATCH_SIZE) {
            let inputs = batch
                .iter()
                .map(|chunk| document_input(&chunk.heading, &chunk.text))
                .collect::<Vec<_>>();
            let output = self
                .client
                .embed(inputs, EmbeddingPurpose::Document)
                .await?;
            embedding_calls += 1;
            prompt_tokens += output.prompt_tokens;
            total_tokens += output.total_tokens;
            embeddings.extend(output.embeddings);
        }
        let embedding_duration = embedding_started.elapsed();

        if embeddings.len() != pending.len() {
            return Err(KnowledgeError::EmbeddingProtocol(format!(
                "model returned {} vectors for {} chunks",
                embeddings.len(),
                pending.len()
            )));
        }

        let dimension = embeddings.first().map_or(0, Vec::len);
        let chunks = pending
            .into_iter()
            .zip(embeddings.iter_mut())
            .map(|(chunk, embedding)| {
                normalize_vector(embedding);
                StoredChunk {
                    id: chunk.id,
                    source: chunk.source,
                    heading: chunk.heading,
                    text: chunk.text,
                    embedding: std::mem::take(embedding),
                }
            })
            .collect::<Vec<_>>();
        let index = KnowledgeIndex {
            schema_version: SCHEMA_VERSION,
            model: self.config.model.clone(),
            dimension,
            document_input_type: self.config.document_input_type.clone(),
            sources,
            chunks,
        };
        write_index(&self.paths.index_file, &index)?;
        let index_bytes = metadata(&self.paths.index_file)?.len();
        let report = IndexReport {
            source_count: index.sources.len(),
            section_count,
            chunk_count: index.chunks.len(),
            dimension,
            embedding: embedding_duration,
            embedding_calls,
            prompt_tokens,
            total_tokens,
            total: total_started.elapsed(),
            index_path: self.paths.index_file.clone(),
            index_bytes,
        };
        self.index = Some(index);
        Ok(report)
    }

    pub async fn query(
        &mut self,
        query: &str,
        top_k: usize,
    ) -> Result<QueryReport, KnowledgeError> {
        let query = query.trim();
        if query.is_empty() {
            return Err(KnowledgeError::EmptyQuery);
        }
        if self.index.is_none() {
            self.load_index()?;
        }

        let embedding_started = Instant::now();
        let output = self
            .client
            .embed(vec![query_input(query)], EmbeddingPurpose::Query)
            .await?;
        let mut query_embedding = output.embeddings.into_iter().next().ok_or_else(|| {
            KnowledgeError::EmbeddingProtocol("model returned no query vector".to_owned())
        })?;
        normalize_vector(&mut query_embedding);
        let embedding_duration = embedding_started.elapsed();

        let index = self
            .index
            .as_ref()
            .expect("knowledge index was loaded before search");
        if query_embedding.len() != index.dimension {
            return Err(KnowledgeError::IncompatibleIndex(format!(
                "index dimension is {}, model returned {}",
                index.dimension,
                query_embedding.len()
            )));
        }

        let search_started = Instant::now();
        let query_terms = lexical_terms(query);
        let mut hits = index
            .chunks
            .iter()
            .map(|chunk| {
                let dense_score = dot_product(&query_embedding, &chunk.embedding);
                let lexical_score =
                    lexical_score(&query_terms, &format!("{} {}", chunk.heading, chunk.text));
                SearchHit {
                    id: chunk.id.clone(),
                    source: chunk.source.clone(),
                    heading: chunk.heading.clone(),
                    text: chunk.text.clone(),
                    dense_score,
                    lexical_score,
                    combined_score: dense_score + LEXICAL_WEIGHT * lexical_score,
                }
            })
            .collect::<Vec<_>>();
        hits.sort_by(|left, right| right.combined_score.total_cmp(&left.combined_score));
        hits.truncate(top_k);

        Ok(QueryReport {
            embedding: embedding_duration,
            embedding_calls: 1,
            prompt_tokens: output.prompt_tokens,
            total_tokens: output.total_tokens,
            search: search_started.elapsed(),
            hits,
            index_path: self.paths.index_file.clone(),
        })
    }
}

fn document_input(heading: &str, text: &str) -> String {
    if heading.is_empty() {
        text.to_owned()
    } else {
        format!("{heading}\n{text}")
    }
}

fn query_input(query: &str) -> String {
    format!("Instruct: {QUERY_INSTRUCTION}\nQuery: {query}")
}

#[derive(Debug, Serialize, Deserialize)]
struct KnowledgeIndex {
    schema_version: u32,
    model: String,
    dimension: usize,
    document_input_type: String,
    sources: Vec<SourceMetadata>,
    chunks: Vec<StoredChunk>,
}

#[derive(Debug, Serialize, Deserialize)]
struct SourceMetadata {
    path: PathBuf,
    fingerprint: String,
}

#[derive(Debug, Serialize, Deserialize)]
struct StoredChunk {
    id: String,
    source: PathBuf,
    heading: String,
    text: String,
    embedding: Vec<f32>,
}

struct PendingChunk {
    id: String,
    source: PathBuf,
    heading: String,
    text: String,
}

#[derive(Debug, PartialEq, Eq)]
struct MarkdownSection {
    heading: String,
    text: String,
}

fn markdown_files(source: &Path) -> Result<Vec<PathBuf>, KnowledgeError> {
    if !source.exists() {
        return Err(KnowledgeError::SourceMissing(source.to_owned()));
    }
    let mut files = Vec::new();
    collect_markdown_files(source, &mut files)?;
    files.sort();
    if files.is_empty() {
        return Err(KnowledgeError::NoMarkdown(source.to_owned()));
    }
    Ok(files)
}

fn collect_markdown_files(path: &Path, files: &mut Vec<PathBuf>) -> Result<(), KnowledgeError> {
    if path.is_file() {
        if path
            .extension()
            .is_some_and(|extension| extension.eq_ignore_ascii_case("md"))
        {
            files.push(path.to_owned());
        }
        return Ok(());
    }

    let entries = fs::read_dir(path).map_err(|source| KnowledgeError::Io {
        action: "could not read knowledge directory",
        path: path.to_owned(),
        source,
    })?;
    for entry in entries {
        let entry = entry.map_err(|source| KnowledgeError::Io {
            action: "could not read knowledge directory entry",
            path: path.to_owned(),
            source,
        })?;
        collect_markdown_files(&entry.path(), files)?;
    }
    Ok(())
}

fn parse_markdown(markdown: &str) -> Vec<MarkdownSection> {
    let mut sections = Vec::new();
    let mut headings = Vec::<String>::new();
    let mut heading_level = None;
    let mut heading_text = String::new();
    let mut body = String::new();
    let mut image_depth = 0_usize;

    for event in Parser::new_ext(markdown, Options::all()) {
        match event {
            Event::Start(Tag::Image { .. }) => image_depth += 1,
            Event::End(TagEnd::Image) => image_depth = image_depth.saturating_sub(1),
            _ if image_depth > 0 => {}
            Event::Start(Tag::Heading { level, .. }) => {
                push_section(&mut sections, &headings, &mut body);
                heading_level = Some(heading_index(level));
                heading_text.clear();
            }
            Event::End(TagEnd::Heading(_)) => {
                if let Some(level) = heading_level.take() {
                    let heading = normalize_inline(&heading_text);
                    if !heading.is_empty() {
                        headings.truncate(level);
                        while headings.len() < level {
                            headings.push(String::new());
                        }
                        headings.push(heading);
                    }
                }
            }
            Event::Text(text) | Event::Code(text) => {
                if heading_level.is_some() {
                    append_inline(&mut heading_text, &text);
                } else {
                    append_inline(&mut body, &text);
                }
            }
            Event::SoftBreak => append_inline(
                active_text(&mut heading_text, &mut body, heading_level),
                " ",
            ),
            Event::HardBreak => {
                active_text(&mut heading_text, &mut body, heading_level).push('\n');
            }
            Event::End(TagEnd::Paragraph)
            | Event::End(TagEnd::Item)
            | Event::End(TagEnd::CodeBlock)
            | Event::Rule
                if heading_level.is_none() =>
            {
                body.push_str("\n\n");
            }
            _ => {}
        }
    }
    push_section(&mut sections, &headings, &mut body);
    sections
}

fn active_text<'a>(
    heading_text: &'a mut String,
    body: &'a mut String,
    heading_level: Option<usize>,
) -> &'a mut String {
    if heading_level.is_some() {
        heading_text
    } else {
        body
    }
}

fn heading_index(level: HeadingLevel) -> usize {
    match level {
        HeadingLevel::H1 => 0,
        HeadingLevel::H2 => 1,
        HeadingLevel::H3 => 2,
        HeadingLevel::H4 => 3,
        HeadingLevel::H5 => 4,
        HeadingLevel::H6 => 5,
    }
}

fn push_section(sections: &mut Vec<MarkdownSection>, headings: &[String], body: &mut String) {
    let text = normalize_body(body);
    body.clear();
    if text.is_empty() {
        return;
    }
    let heading = headings
        .iter()
        .filter(|heading| !heading.is_empty())
        .cloned()
        .collect::<Vec<_>>()
        .join(" > ");
    sections.push(MarkdownSection { heading, text });
}

fn append_inline(output: &mut String, text: &str) {
    if !output.is_empty()
        && !output.ends_with(char::is_whitespace)
        && !text.starts_with(char::is_whitespace)
    {
        output.push(' ');
    }
    output.push_str(text);
}

fn normalize_inline(text: &str) -> String {
    text.split_whitespace().collect::<Vec<_>>().join(" ")
}

fn normalize_body(text: &str) -> String {
    text.split("\n\n")
        .map(normalize_inline)
        .filter(|paragraph| !paragraph.is_empty())
        .collect::<Vec<_>>()
        .join("\n\n")
}

fn chunk_text(text: &str) -> Vec<String> {
    let blocks = text
        .split("\n\n")
        .flat_map(split_long_block)
        .filter(|block| !block.is_empty())
        .collect::<Vec<_>>();
    let mut chunks = Vec::new();
    let mut current = String::new();

    for block in blocks {
        let separator = if current.is_empty() { 0 } else { 2 };
        if current.chars().count() + separator + block.chars().count() > MAX_CHUNK_CHARS
            && !current.is_empty()
        {
            chunks.push(std::mem::take(&mut current));
        }
        if !current.is_empty() {
            current.push_str("\n\n");
        }
        current.push_str(&block);
    }
    if !current.is_empty() {
        chunks.push(current);
    }
    chunks
}

fn split_long_block(block: &str) -> Vec<String> {
    if block.chars().count() <= MAX_CHUNK_CHARS {
        return vec![block.to_owned()];
    }
    let mut parts = Vec::new();
    let mut remaining = block.trim();
    while remaining.chars().count() > MAX_CHUNK_CHARS {
        let limit = byte_at_char(remaining, MAX_CHUNK_CHARS);
        let candidate = &remaining[..limit];
        let minimum = byte_at_char(remaining, MAX_CHUNK_CHARS / 2);
        let split = candidate
            .char_indices()
            .rev()
            .find(|(index, character)| {
                *index >= minimum
                    && (matches!(character, '.' | '!' | '?' | ';' | '\n')
                        || character.is_whitespace())
            })
            .map_or(limit, |(index, character)| index + character.len_utf8());
        parts.push(remaining[..split].trim().to_owned());
        remaining = remaining[split..].trim();
    }
    if !remaining.is_empty() {
        parts.push(remaining.to_owned());
    }
    parts
}

fn byte_at_char(text: &str, character_index: usize) -> usize {
    text.char_indices()
        .nth(character_index)
        .map_or(text.len(), |(index, _)| index)
}

fn lexical_terms(text: &str) -> HashSet<String> {
    text.split(|character: char| !character.is_alphanumeric())
        .map(str::to_lowercase)
        .filter(|term| term.chars().count() >= 3 && !is_stopword(term))
        .collect()
}

fn is_stopword(term: &str) -> bool {
    matches!(
        term,
        "как"
            | "что"
            | "это"
            | "для"
            | "при"
            | "или"
            | "его"
            | "она"
            | "они"
            | "чем"
            | "когда"
            | "который"
            | "какие"
            | "the"
            | "and"
            | "for"
            | "with"
            | "what"
            | "how"
    )
}

fn lexical_score(query_terms: &HashSet<String>, text: &str) -> f32 {
    if query_terms.is_empty() {
        return 0.0;
    }
    let document_terms = lexical_terms(text);
    let matches = query_terms.intersection(&document_terms).count();
    matches as f32 / query_terms.len() as f32
}

fn normalize_vector(vector: &mut [f32]) {
    let magnitude = vector.iter().map(|value| value * value).sum::<f32>().sqrt();
    if magnitude > f32::EPSILON {
        for value in vector {
            *value /= magnitude;
        }
    }
}

fn dot_product(left: &[f32], right: &[f32]) -> f32 {
    left.iter()
        .zip(right)
        .map(|(left, right)| left * right)
        .sum()
}

fn validate_index(index: &KnowledgeIndex, config: &EmbeddingConfig) -> Result<(), KnowledgeError> {
    if index.schema_version != SCHEMA_VERSION {
        return Err(KnowledgeError::IncompatibleIndex(format!(
            "schema version is {}, expected {}",
            index.schema_version, SCHEMA_VERSION
        )));
    }
    if index.model != config.model {
        return Err(KnowledgeError::IncompatibleIndex(format!(
            "model is {}, expected {}",
            index.model, config.model
        )));
    }
    if index.document_input_type != config.document_input_type {
        return Err(KnowledgeError::IncompatibleIndex(format!(
            "document input type is {}, expected {}",
            index.document_input_type, config.document_input_type
        )));
    }
    if index.dimension != config.dimensions
        || index
            .chunks
            .iter()
            .any(|chunk| chunk.embedding.len() != index.dimension)
    {
        return Err(KnowledgeError::IncompatibleIndex(format!(
            "stored vector dimension is {}, expected {}",
            index.dimension, config.dimensions
        )));
    }
    Ok(())
}

fn read_index(path: &Path) -> Result<KnowledgeIndex, KnowledgeError> {
    if !path.is_file() {
        return Err(KnowledgeError::IndexMissing(path.to_owned()));
    }
    let bytes = read_file(path)?;
    serde_json::from_slice(&bytes).map_err(|source| KnowledgeError::Decode {
        path: path.to_owned(),
        source,
    })
}

fn write_index(path: &Path, index: &KnowledgeIndex) -> Result<(), KnowledgeError> {
    let parent = path.parent().unwrap_or_else(|| Path::new("."));
    create_directory(parent)?;
    let bytes = serde_json::to_vec(index).map_err(KnowledgeError::Encode)?;
    let temporary = path.with_extension("json.tmp");
    fs::write(&temporary, bytes).map_err(|source| KnowledgeError::Io {
        action: "could not write temporary knowledge index",
        path: temporary.clone(),
        source,
    })?;
    fs::rename(&temporary, path).map_err(|source| KnowledgeError::Io {
        action: "could not replace knowledge index",
        path: path.to_owned(),
        source,
    })
}

fn create_directory(path: &Path) -> Result<(), KnowledgeError> {
    fs::create_dir_all(path).map_err(|source| KnowledgeError::Io {
        action: "could not create cache directory",
        path: path.to_owned(),
        source,
    })
}

fn read_file(path: &Path) -> Result<Vec<u8>, KnowledgeError> {
    fs::read(path).map_err(|source| KnowledgeError::Io {
        action: "could not read file",
        path: path.to_owned(),
        source,
    })
}

fn metadata(path: &Path) -> Result<fs::Metadata, KnowledgeError> {
    fs::metadata(path).map_err(|source| KnowledgeError::Io {
        action: "could not inspect file",
        path: path.to_owned(),
        source,
    })
}

fn fingerprint(bytes: &[u8]) -> u64 {
    let mut hash = 0xcbf29ce484222325_u64;
    for byte in bytes {
        hash ^= u64::from(*byte);
        hash = hash.wrapping_mul(0x100000001b3);
    }
    hash
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn markdown_parser_keeps_heading_hierarchy_and_drops_images() {
        let markdown = r#"
# Java

## HashMap

Текст до картинки.

![diagram][map]

Текст после картинки.

[map]: <data:image/png;base64,AAAAvery-large-payloadBBBB>
"#;

        let sections = parse_markdown(markdown);

        assert_eq!(sections.len(), 1);
        assert_eq!(sections[0].heading, "Java > HashMap");
        assert!(sections[0].text.contains("Текст до картинки."));
        assert!(sections[0].text.contains("Текст после картинки."));
        assert!(!sections[0].text.contains("diagram"));
        assert!(!sections[0].text.contains("base64"));
        assert!(!sections[0].text.contains("AAAA"));
    }

    #[test]
    fn chunking_respects_unicode_and_size_limit() {
        let text = "абв ".repeat(1_000);
        let chunks = chunk_text(&text);

        assert!(chunks.len() > 1);
        assert!(
            chunks
                .iter()
                .all(|chunk| chunk.chars().count() <= MAX_CHUNK_CHARS)
        );
        assert_eq!(
            chunks
                .iter()
                .map(|chunk| chunk.split_whitespace().count())
                .sum::<usize>(),
            1_000
        );
    }

    #[test]
    fn lexical_score_rewards_matching_technical_terms() {
        let query = lexical_terms("Чем отличается HashMap от ConcurrentHashMap?");

        let relevant = lexical_score(
            &query,
            "HashMap и ConcurrentHashMap отличаются потокобезопасностью",
        );
        let irrelevant = lexical_score(&query, "Сборщик мусора освобождает память");

        assert!(relevant > irrelevant);
    }

    #[test]
    fn retrieval_instruction_is_only_added_to_queries() {
        let query = query_input("Как работает ConcurrentHashMap?");
        let document = document_input(
            "Java > ConcurrentHashMap",
            "ConcurrentHashMap поддерживает конкурентный доступ.",
        );

        assert!(query.starts_with("Instruct: "));
        assert!(query.contains("\nQuery: Как работает ConcurrentHashMap?"));
        assert!(!document.contains("Instruct:"));
        assert_eq!(
            document,
            "Java > ConcurrentHashMap\nConcurrentHashMap поддерживает конкурентный доступ."
        );
    }

    #[test]
    fn normalized_dot_product_is_cosine_similarity() {
        let mut left = vec![3.0, 4.0];
        let mut same = vec![6.0, 8.0];
        let mut opposite = vec![-3.0, -4.0];
        normalize_vector(&mut left);
        normalize_vector(&mut same);
        normalize_vector(&mut opposite);

        assert!((dot_product(&left, &same) - 1.0).abs() < 0.000_1);
        assert!((dot_product(&left, &opposite) + 1.0).abs() < 0.000_1);
    }
}
