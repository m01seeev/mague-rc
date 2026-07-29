use std::{
    env, fs,
    path::PathBuf,
    time::{Duration, Instant, SystemTime, UNIX_EPOCH},
};

use futures_util::StreamExt;
use mague_rc::{
    config::{Config, KnowledgeConfig},
    events::{KnowledgeContext, KnowledgeSnippet, LlmUsage},
    knowledge::RemoteRetriever,
    llm::{
        ChatMessage, ChatRequest, LlmStreamEvent, OpenRouterTextProvider, TextLlmProvider,
        knowledge_context_prompt, voice_system_prompt,
    },
    stt::install_tls_crypto_provider,
};
use serde::Serialize;
use tokio::time::{sleep, timeout};

const DEFAULT_QUESTION_FILES: &[&str] = &[
    "benchmark_human.expected.txt",
    "benchmark_dangling.expected.txt",
    "benchmark_dangling_complete.expected.txt",
    "benchmark_expressive.expected.txt",
    "benchmark_disfluent.expected.txt",
    "benchmark_technical.expected.txt",
];

const DEFAULT_MODELS: &[&str] = &[
    "openai/gpt-4o-mini",
    "deepseek/deepseek-v4-flash",
    "xiaomi/mimo-v2.5",
    "google/gemini-3.5-flash-lite",
];

#[derive(Debug)]
struct Options {
    question_files: Vec<PathBuf>,
    models: Vec<String>,
    repeat: usize,
    output: PathBuf,
}

#[derive(Clone)]
struct PromptCase {
    id: String,
    source: String,
    question: String,
    messages: Vec<ChatMessage>,
    retrieval: RetrievalSummary,
}

#[derive(Clone, Serialize)]
struct RetrievalSummary {
    embedding_ms: u64,
    search_ms: u64,
    hits: Vec<RetrievalHit>,
}

#[derive(Clone, Serialize)]
struct RetrievalHit {
    source: String,
    heading: String,
    score: f32,
}

#[derive(Serialize)]
struct BenchmarkReport {
    generated_unix_ms: u128,
    repeats: usize,
    question_count: usize,
    models: Vec<ModelReport>,
}

#[derive(Serialize)]
struct ModelReport {
    model: String,
    summary: ModelSummary,
    runs: Vec<RunReport>,
}

#[derive(Serialize)]
struct ModelSummary {
    successful: usize,
    failed: usize,
    ttft_ms: Distribution,
    total_ms: Distribution,
    prompt_tokens: u64,
    completion_tokens: u64,
    total_tokens: u64,
    total_cost: f64,
}

#[derive(Default, Serialize)]
struct Distribution {
    count: usize,
    min: u64,
    mean: f64,
    p50: u64,
    p95: u64,
    max: u64,
}

#[derive(Serialize)]
struct RunReport {
    question_id: String,
    source: String,
    question: String,
    iteration: usize,
    retrieval: RetrievalSummary,
    ttft_ms: Option<u64>,
    total_ms: u64,
    answer: String,
    usage: Option<UsageSummary>,
    error: Option<String>,
}

#[derive(Serialize)]
struct UsageSummary {
    prompt_tokens: u64,
    completion_tokens: u64,
    total_tokens: u64,
    cost: Option<f64>,
}

struct Completion {
    ttft_ms: Option<u64>,
    total_ms: u64,
    answer: String,
    usage: Option<LlmUsage>,
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    install_tls_crypto_provider()?;
    let options = parse_options(env::args().skip(1).collect())?;
    let config = Config::load()?;
    let questions = load_questions(&options.question_files)?;
    let prompts = build_prompts(&config, questions).await?;
    let mut models = Vec::new();

    for model in &options.models {
        eprintln!("model: {model}");
        let mut llm_config = config.llm.clone();
        llm_config.model.clone_from(model);
        let provider = OpenRouterTextProvider::new(llm_config.clone())?;
        let mut runs = Vec::new();

        for iteration in 1..=options.repeat {
            for (position, prompt) in prompts.iter().enumerate() {
                eprintln!(
                    "  run {iteration}/{}, question {}/{}",
                    options.repeat,
                    position + 1,
                    prompts.len()
                );
                let result = timeout(
                    Duration::from_secs(llm_config.timeout_sec),
                    complete(&provider, prompt.messages.clone()),
                )
                .await;
                runs.push(run_report(prompt, iteration, result));
                sleep(Duration::from_millis(150)).await;
            }
        }

        let summary = summarize(&runs);
        eprintln!(
            "  complete: {}/{}; TTFT p50={} ms p95={} ms; total p50={} ms p95={} ms; cost=${:.6}",
            summary.successful,
            runs.len(),
            summary.ttft_ms.p50,
            summary.ttft_ms.p95,
            summary.total_ms.p50,
            summary.total_ms.p95,
            summary.total_cost
        );
        models.push(ModelReport {
            model: model.clone(),
            summary,
            runs,
        });
    }

    let report = BenchmarkReport {
        generated_unix_ms: SystemTime::now().duration_since(UNIX_EPOCH)?.as_millis(),
        repeats: options.repeat,
        question_count: prompts.len(),
        models,
    };
    if let Some(parent) = options.output.parent()
        && !parent.as_os_str().is_empty()
    {
        fs::create_dir_all(parent)?;
    }
    fs::write(&options.output, serde_json::to_vec_pretty(&report)?)?;
    println!("{}", options.output.display());
    Ok(())
}

fn parse_options(arguments: Vec<String>) -> Result<Options, String> {
    let mut question_files = Vec::new();
    let mut models = Vec::new();
    let mut repeat = 1;
    let mut output = PathBuf::from("telemetry/llm-benchmark.json");
    let mut arguments = arguments.into_iter();

    while let Some(argument) = arguments.next() {
        match argument.as_str() {
            "--questions" => question_files.push(PathBuf::from(
                arguments
                    .next()
                    .ok_or_else(|| "--questions requires a file".to_owned())?,
            )),
            "--model" => models.push(
                arguments
                    .next()
                    .ok_or_else(|| "--model requires a model slug".to_owned())?,
            ),
            "--repeat" => {
                repeat = arguments
                    .next()
                    .ok_or_else(|| "--repeat requires a number".to_owned())?
                    .parse()
                    .map_err(|_| "--repeat must be a positive number".to_owned())?;
                if repeat == 0 {
                    return Err("--repeat must be greater than zero".to_owned());
                }
            }
            "--output" => {
                output = PathBuf::from(
                    arguments
                        .next()
                        .ok_or_else(|| "--output requires a file".to_owned())?,
                );
            }
            _ => return Err(format!("unexpected argument: {argument}")),
        }
    }

    if question_files.is_empty() {
        question_files = DEFAULT_QUESTION_FILES.iter().map(PathBuf::from).collect();
    }
    if models.is_empty() {
        models = DEFAULT_MODELS.iter().map(ToString::to_string).collect();
    }

    Ok(Options {
        question_files,
        models,
        repeat,
        output,
    })
}

fn load_questions(files: &[PathBuf]) -> Result<Vec<(String, String, String)>, std::io::Error> {
    let mut questions = Vec::new();
    for file in files {
        let contents = fs::read_to_string(file)?;
        let source = file.display().to_string();
        let stem = file
            .file_stem()
            .and_then(|value| value.to_str())
            .unwrap_or("questions");
        for (index, question) in contents
            .lines()
            .map(str::trim)
            .filter(|line| !line.is_empty())
            .enumerate()
        {
            questions.push((
                format!("{stem}-{}", index + 1),
                source.clone(),
                question.to_owned(),
            ));
        }
    }
    Ok(questions)
}

async fn build_prompts(
    config: &Config,
    questions: Vec<(String, String, String)>,
) -> Result<Vec<PromptCase>, Box<dyn std::error::Error>> {
    let mut retriever = RemoteRetriever::load(config.knowledge.embedding.clone())?;
    let mut prompts = Vec::with_capacity(questions.len());
    eprintln!("retrieving fixed context for {} questions", questions.len());

    for (position, (id, source, question)) in questions.into_iter().enumerate() {
        eprintln!("  retrieval {}/{}", position + 1, prompts.capacity());
        let report = retriever.query(&question, config.knowledge.top_k).await?;
        let context = knowledge_context(&config.knowledge, &report);
        let retrieval = RetrievalSummary {
            embedding_ms: report.embedding.as_millis() as u64,
            search_ms: report.search.as_millis() as u64,
            hits: context
                .snippets
                .iter()
                .map(|snippet| RetrievalHit {
                    source: snippet.source.clone(),
                    heading: snippet.heading.clone(),
                    score: snippet.score,
                })
                .collect(),
        };
        let mut messages = vec![ChatMessage::system(voice_system_prompt(
            &config.llm.current_project,
        ))];
        if !context.snippets.is_empty() {
            messages.push(ChatMessage::system(knowledge_context_prompt(&context)));
        }
        messages.push(ChatMessage::user(question.clone()));
        prompts.push(PromptCase {
            id,
            source,
            question,
            messages,
            retrieval,
        });
    }
    Ok(prompts)
}

fn knowledge_context(
    config: &KnowledgeConfig,
    report: &mague_rc::knowledge::QueryReport,
) -> KnowledgeContext {
    let mut remaining_chars = config.max_context_chars;
    let mut snippets = Vec::new();
    for hit in report
        .hits
        .iter()
        .filter(|hit| hit.combined_score >= config.min_score)
    {
        if snippets.len() >= config.top_k || remaining_chars == 0 {
            break;
        }
        let text = hit.text.chars().take(remaining_chars).collect::<String>();
        remaining_chars = remaining_chars.saturating_sub(text.chars().count());
        snippets.push(KnowledgeSnippet {
            id: hit.id.clone(),
            source: hit.source.display().to_string(),
            heading: hit.heading.clone(),
            text,
            score: hit.combined_score,
        });
    }
    KnowledgeContext {
        snippets,
        searches: 1,
        embedding_calls: report.embedding_calls,
        embedding_prompt_tokens: report.prompt_tokens,
        embedding_total_tokens: report.total_tokens,
        embedding_ms: report.embedding.as_millis() as u64,
        search_ms: report.search.as_millis() as u64,
        final_wait_ms: 0,
    }
}

async fn complete(
    provider: &OpenRouterTextProvider,
    messages: Vec<ChatMessage>,
) -> Result<Completion, mague_rc::llm::LlmError> {
    let started = Instant::now();
    let mut first_token = None;
    let mut answer = String::new();
    let mut usage = None;
    let mut stream = provider.stream(ChatRequest::new(messages));

    while let Some(event) = stream.next().await {
        match event? {
            LlmStreamEvent::Delta(delta) => {
                if first_token.is_none() && !delta.is_empty() {
                    first_token = Some(started.elapsed().as_millis() as u64);
                }
                answer.push_str(&delta);
            }
            LlmStreamEvent::Usage(value) => usage = Some(value),
        }
    }
    if answer.trim().is_empty() {
        return Err(mague_rc::llm::LlmError::EmptyResponse);
    }
    Ok(Completion {
        ttft_ms: first_token,
        total_ms: started.elapsed().as_millis() as u64,
        answer,
        usage,
    })
}

fn run_report(
    prompt: &PromptCase,
    iteration: usize,
    result: Result<Result<Completion, mague_rc::llm::LlmError>, tokio::time::error::Elapsed>,
) -> RunReport {
    match result {
        Ok(Ok(completion)) => RunReport {
            question_id: prompt.id.clone(),
            source: prompt.source.clone(),
            question: prompt.question.clone(),
            iteration,
            retrieval: prompt.retrieval.clone(),
            ttft_ms: completion.ttft_ms,
            total_ms: completion.total_ms,
            answer: completion.answer,
            usage: completion.usage.map(usage_summary),
            error: None,
        },
        Ok(Err(error)) => failed_run(prompt, iteration, error.to_string()),
        Err(error) => failed_run(prompt, iteration, error.to_string()),
    }
}

fn failed_run(prompt: &PromptCase, iteration: usize, error: String) -> RunReport {
    RunReport {
        question_id: prompt.id.clone(),
        source: prompt.source.clone(),
        question: prompt.question.clone(),
        iteration,
        retrieval: prompt.retrieval.clone(),
        ttft_ms: None,
        total_ms: 0,
        answer: String::new(),
        usage: None,
        error: Some(error),
    }
}

fn usage_summary(usage: LlmUsage) -> UsageSummary {
    UsageSummary {
        prompt_tokens: usage.prompt_tokens,
        completion_tokens: usage.completion_tokens,
        total_tokens: usage.total_tokens,
        cost: usage.cost,
    }
}

fn summarize(runs: &[RunReport]) -> ModelSummary {
    let successful = runs.iter().filter(|run| run.error.is_none()).count();
    let usage = runs.iter().filter_map(|run| run.usage.as_ref());
    ModelSummary {
        successful,
        failed: runs.len() - successful,
        ttft_ms: distribution(runs.iter().filter_map(|run| run.ttft_ms).collect()),
        total_ms: distribution(
            runs.iter()
                .filter(|run| run.error.is_none())
                .map(|run| run.total_ms)
                .collect(),
        ),
        prompt_tokens: usage.clone().map(|usage| usage.prompt_tokens).sum(),
        completion_tokens: usage.clone().map(|usage| usage.completion_tokens).sum(),
        total_tokens: usage.clone().map(|usage| usage.total_tokens).sum(),
        total_cost: usage.filter_map(|usage| usage.cost).sum(),
    }
}

fn distribution(mut values: Vec<u64>) -> Distribution {
    if values.is_empty() {
        return Distribution::default();
    }
    values.sort_unstable();
    let count = values.len();
    Distribution {
        count,
        min: values[0],
        mean: values.iter().sum::<u64>() as f64 / count as f64,
        p50: percentile(&values, 0.50),
        p95: percentile(&values, 0.95),
        max: values[count - 1],
    }
}

fn percentile(values: &[u64], percentile: f64) -> u64 {
    let index = ((values.len() - 1) as f64 * percentile).ceil() as usize;
    values[index]
}
