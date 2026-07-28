use std::time::{Duration, Instant};

use futures_util::StreamExt;
use thiserror::Error;
use tokio::{sync::mpsc, time::sleep};
use tracing::{debug, info, warn};

use crate::{
    config::LlmConfig,
    events::{AnswerMeta, AppErrorView, LlmCommand, LlmRequest, OutputComponent, OutputEvent},
    llm::{
        ChatMessage, ChatRequest, ConversationHistories, LlmError, LlmRequestReceiver,
        LlmStreamEvent, TextLlmProvider, knowledge_context_prompt, voice_system_prompt,
    },
};

#[derive(Debug, Error)]
pub enum LlmWorkerError {
    #[error("LLM output channel closed")]
    OutputChannelClosed,
}

#[derive(Default)]
pub struct LlmWorkerStats {
    pub requests: u64,
    pub completed: u64,
    pub failed: u64,
}

pub struct LlmWorker<P> {
    provider: P,
    config: LlmConfig,
    histories: ConversationHistories,
}

impl<P> LlmWorker<P>
where
    P: TextLlmProvider,
{
    pub fn new(provider: P, config: LlmConfig) -> Self {
        let histories =
            ConversationHistories::new(config.max_history_pairs, config.separate_histories);
        Self {
            provider,
            config,
            histories,
        }
    }

    pub async fn run(
        mut self,
        mut requests: LlmRequestReceiver,
        mut commands: mpsc::UnboundedReceiver<LlmCommand>,
        events: mpsc::UnboundedSender<OutputEvent>,
    ) -> Result<LlmWorkerStats, LlmWorkerError> {
        let mut stats = LlmWorkerStats::default();
        let mut commands_open = true;

        loop {
            let request = tokio::select! {
                biased;
                command = commands.recv(), if commands_open => {
                    match command {
                        Some(LlmCommand::ClearHistory) => {
                            self.histories.clear();
                            info!(
                                module = "llm",
                                event = "history_cleared",
                                "LLM histories cleared"
                            );
                        }
                        None => commands_open = false,
                    }
                    continue;
                }
                request = requests.recv() => match request {
                    Some(request) => request,
                    None => break,
                }
            };

            stats.requests += 1;
            let queue_len = requests.len();
            send_event(
                &events,
                OutputEvent::AnswerStarted(AnswerMeta {
                    request_id: request.request_id,
                    mode: request.mode,
                }),
            )?;

            info!(
                module = "llm",
                event = "request_started",
                request_id = request.request_id,
                mode = %request.mode,
                queue_len,
                provider = "openrouter",
                model = %self.config.model,
                "LLM request started"
            );

            match self.process_request(&request, &events).await {
                Ok(full_text) => {
                    self.histories
                        .record(request.mode, request.text.clone(), full_text.clone());
                    stats.completed += 1;
                    send_event(
                        &events,
                        OutputEvent::AnswerCompleted {
                            request_id: request.request_id,
                        },
                    )?;
                }
                Err(error) => {
                    stats.failed += 1;
                    warn!(
                        module = "llm",
                        event = "request_failed",
                        request_id = request.request_id,
                        mode = %request.mode,
                        provider = "openrouter",
                        model = %self.config.model,
                        error = %error,
                        "LLM request failed"
                    );
                    send_event(
                        &events,
                        OutputEvent::Error(AppErrorView {
                            component: OutputComponent::Llm,
                            message: format!("request #{} failed: {error}", request.request_id),
                        }),
                    )?;
                }
            }
        }

        Ok(stats)
    }

    async fn process_request(
        &self,
        request: &LlmRequest,
        events: &mpsc::UnboundedSender<OutputEvent>,
    ) -> Result<String, LlmError> {
        let history = self.histories.messages(request.mode);
        let mut messages = Vec::with_capacity(2 + history.len());
        messages.push(ChatMessage::system(voice_system_prompt(
            &self.config.current_project,
        )));
        messages.extend(history);
        if let Some(knowledge) = request.knowledge.as_ref() {
            messages.push(ChatMessage::system(knowledge_context_prompt(knowledge)));
        }
        messages.push(ChatMessage::user(request.text.clone()));

        let started_at = Instant::now();
        let mut first_token_at = None;
        let mut full_text = String::new();
        let mut stream = self.provider.stream(ChatRequest { messages });
        let timeout_duration = Duration::from_secs(self.config.timeout_sec);
        let deadline = sleep(timeout_duration);
        tokio::pin!(deadline);

        loop {
            tokio::select! {
                _ = &mut deadline => return Err(LlmError::Timeout(timeout_duration)),
                item = stream.next() => match item {
                    Some(Ok(LlmStreamEvent::Delta(delta))) if !delta.is_empty() => {
                        if first_token_at.is_none() {
                            let latency = started_at.elapsed();
                            first_token_at = Some(latency);
                            info!(
                                module = "llm",
                                event = "first_token",
                                request_id = request.request_id,
                                mode = %request.mode,
                                latency_ms = latency.as_millis(),
                                provider = "openrouter",
                                model = %self.config.model,
                                "received first LLM token"
                            );
                        }
                        full_text.push_str(&delta);
                        send_event(
                            events,
                            OutputEvent::AnswerDelta {
                                request_id: request.request_id,
                                text: delta,
                            },
                        )
                        .map_err(|_| LlmError::Request("output channel closed".to_owned()))?;
                    }
                    Some(Ok(LlmStreamEvent::Usage(usage))) => {
                        send_event(
                            events,
                            OutputEvent::AnswerUsage {
                                request_id: request.request_id,
                                usage,
                            },
                        )
                        .map_err(|_| LlmError::Request("output channel closed".to_owned()))?;
                    }
                    Some(Ok(LlmStreamEvent::Delta(_))) => {}
                    Some(Err(error)) => return Err(error),
                    None => break,
                }
            }
        }

        if full_text.trim().is_empty() {
            return Err(LlmError::EmptyResponse);
        }

        debug!(
            module = "llm",
            event = "request_completed",
            request_id = request.request_id,
            mode = %request.mode,
            latency_ms = started_at.elapsed().as_millis(),
            provider = "openrouter",
            model = %self.config.model,
            "LLM request completed"
        );
        Ok(full_text)
    }
}

fn send_event(
    sender: &mpsc::UnboundedSender<OutputEvent>,
    event: OutputEvent,
) -> Result<(), LlmWorkerError> {
    sender
        .send(event)
        .map_err(|_| LlmWorkerError::OutputChannelClosed)
}

#[cfg(test)]
mod tests {
    use std::sync::{Arc, Mutex};

    use futures_util::stream;
    use url::Url;

    use crate::{
        config::SecretString,
        events::Mode,
        llm::{ChatRole, LlmStreamEvent, TextStream, llm_request_channel},
    };

    use super::*;

    #[derive(Clone)]
    struct FakeProvider {
        requests: Arc<Mutex<Vec<ChatRequest>>>,
    }

    impl TextLlmProvider for FakeProvider {
        fn stream(&self, request: ChatRequest) -> TextStream {
            let response_number = {
                let mut requests = self
                    .requests
                    .lock()
                    .expect("request mutex must not be poisoned");
                requests.push(request);
                requests.len()
            };
            Box::pin(stream::iter([
                Ok(LlmStreamEvent::Delta(format!("answer {response_number}"))),
                Ok(LlmStreamEvent::Delta(" done".to_owned())),
            ]))
        }
    }

    fn config() -> LlmConfig {
        LlmConfig {
            api_key: SecretString::new("secret".to_owned()),
            base_url: Url::parse("https://openrouter.ai/api/v1").expect("base URL must be valid"),
            model: "openai/gpt-4o-mini".to_owned(),
            queue_max: 0,
            max_history_pairs: 4,
            separate_histories: true,
            temperature: 0.2,
            max_tokens: 450,
            timeout_sec: 1,
            current_project: "АО Консалт Плюс".to_owned(),
        }
    }

    fn request(request_id: u64) -> LlmRequest {
        LlmRequest {
            request_id,
            mode: Mode::Voice,
            text: format!("question {request_id}"),
            knowledge: None,
        }
    }

    #[tokio::test]
    async fn processes_requests_in_order_and_updates_history() {
        let captured = Arc::new(Mutex::new(Vec::new()));
        let provider = FakeProvider {
            requests: Arc::clone(&captured),
        };
        let worker = LlmWorker::new(provider, config());
        let (sender, receiver) = llm_request_channel(0);
        let (command_sender, command_receiver) = mpsc::unbounded_channel();
        let (event_sender, mut event_receiver) = mpsc::unbounded_channel();

        sender.send(request(1)).await.expect("send must succeed");
        sender.send(request(2)).await.expect("send must succeed");
        drop(sender);
        drop(command_sender);

        let stats = worker
            .run(receiver, command_receiver, event_sender)
            .await
            .expect("worker must succeed");
        let mut completed = Vec::new();
        while let Some(event) = event_receiver.recv().await {
            if let OutputEvent::AnswerCompleted { request_id } = event {
                completed.push(request_id);
            }
        }

        assert_eq!(completed, vec![1, 2]);
        assert_eq!(stats.completed, 2);

        let requests = captured.lock().expect("request mutex must not be poisoned");
        assert_eq!(requests.len(), 2);
        assert_eq!(requests[0].messages.len(), 2);
        assert_eq!(requests[1].messages.len(), 4);
        assert_eq!(requests[1].messages[1].role, ChatRole::User);
        assert_eq!(requests[1].messages[1].content, "question 1");
        assert_eq!(requests[1].messages[2].role, ChatRole::Assistant);
        assert_eq!(requests[1].messages[2].content, "answer 1 done");
    }

    #[tokio::test]
    async fn clear_command_removes_history_before_next_request() {
        let captured = Arc::new(Mutex::new(Vec::new()));
        let provider = FakeProvider {
            requests: Arc::clone(&captured),
        };
        let worker = LlmWorker::new(provider, config());
        let (sender, receiver) = llm_request_channel(0);
        let (command_sender, command_receiver) = mpsc::unbounded_channel();
        let (event_sender, mut event_receiver) = mpsc::unbounded_channel();
        let worker_task = tokio::spawn(worker.run(receiver, command_receiver, event_sender));

        sender.send(request(1)).await.expect("send must succeed");
        while let Some(event) = event_receiver.recv().await {
            if matches!(event, OutputEvent::AnswerCompleted { request_id: 1 }) {
                break;
            }
        }

        command_sender
            .send(LlmCommand::ClearHistory)
            .expect("clear command must send");
        sender.send(request(2)).await.expect("send must succeed");
        drop(sender);
        drop(command_sender);

        worker_task
            .await
            .expect("worker task must join")
            .expect("worker must succeed");

        let requests = captured.lock().expect("request mutex must not be poisoned");
        assert_eq!(requests.len(), 2);
        assert_eq!(requests[1].messages.len(), 2);
    }

    #[tokio::test]
    async fn inserts_knowledge_before_question_without_storing_it_in_history() {
        let captured = Arc::new(Mutex::new(Vec::new()));
        let provider = FakeProvider {
            requests: Arc::clone(&captured),
        };
        let worker = LlmWorker::new(provider, config());
        let (sender, receiver) = llm_request_channel(0);
        let (command_sender, command_receiver) = mpsc::unbounded_channel();
        let (event_sender, _event_receiver) = mpsc::unbounded_channel();
        let mut first = request(1);
        first.knowledge = Some(crate::events::KnowledgeContext {
            snippets: vec![crate::events::KnowledgeSnippet {
                id: "chunk".to_owned(),
                source: "knowledge/java.md".to_owned(),
                heading: "Java > HashMap".to_owned(),
                text: "HashMap хранит пары ключ-значение.".to_owned(),
                score: 0.9,
            }],
            searches: 1,
            embedding_calls: 1,
            embedding_prompt_tokens: 12,
            embedding_total_tokens: 12,
            embedding_ms: 8,
            search_ms: 1,
            final_wait_ms: 0,
        });

        sender.send(first).await.expect("first request must send");
        sender
            .send(request(2))
            .await
            .expect("second request must send");
        drop(sender);
        drop(command_sender);
        worker
            .run(receiver, command_receiver, event_sender)
            .await
            .expect("worker must finish");

        let requests = captured.lock().expect("request mutex must not be poisoned");
        assert_eq!(requests[0].messages.len(), 3);
        assert_eq!(requests[0].messages[1].role, ChatRole::System);
        assert!(requests[0].messages[1].content.contains("Java > HashMap"));
        assert_eq!(requests[0].messages[2].content, "question 1");
        assert_eq!(requests[1].messages.len(), 4);
        assert_eq!(requests[1].messages[1].content, "question 1");
    }
}
