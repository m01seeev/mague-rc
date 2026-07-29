mod history;
mod live_coding;
mod openrouter;
mod prompts;
mod provider;
mod queue;
mod worker;

pub use history::ConversationHistories;
pub use live_coding::{
    live_coding_system_prompt, live_coding_user_prompt, parse_live_coding_response,
};
pub use openrouter::{OpenRouterTextProvider, build_chat_completions_url};
pub use prompts::{knowledge_context_prompt, voice_system_prompt, voice_user_prompt};
pub use provider::{
    ChatMessage, ChatRequest, ChatRole, LlmError, LlmStreamEvent, TextLlmProvider, TextStream,
};
pub use queue::{LlmQueueError, LlmRequestReceiver, LlmRequestSender, llm_request_channel};
pub use worker::{LlmWorker, LlmWorkerError, LlmWorkerStats};
