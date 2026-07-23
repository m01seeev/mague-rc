use std::{pin::Pin, time::Duration};

use futures_util::Stream;
use serde::Serialize;
use thiserror::Error;

#[derive(Clone, Debug, PartialEq, Eq, Serialize)]
#[serde(rename_all = "lowercase")]
pub enum ChatRole {
    System,
    User,
    Assistant,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize)]
pub struct ChatMessage {
    pub role: ChatRole,
    pub content: String,
}

impl ChatMessage {
    pub fn system(content: impl Into<String>) -> Self {
        Self {
            role: ChatRole::System,
            content: content.into(),
        }
    }

    pub fn user(content: impl Into<String>) -> Self {
        Self {
            role: ChatRole::User,
            content: content.into(),
        }
    }

    pub fn assistant(content: impl Into<String>) -> Self {
        Self {
            role: ChatRole::Assistant,
            content: content.into(),
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ChatRequest {
    pub messages: Vec<ChatMessage>,
}

pub type TextStream = Pin<Box<dyn Stream<Item = Result<String, LlmError>> + Send>>;

pub trait TextLlmProvider: Send + Sync + 'static {
    fn stream(&self, request: ChatRequest) -> TextStream;
}

#[derive(Clone, Debug, Error, PartialEq, Eq)]
pub enum LlmError {
    #[error("could not build the OpenRouter HTTP client: {0}")]
    Client(String),

    #[error("OpenRouter request failed: {0}")]
    Request(String),

    #[error("OpenRouter returned HTTP {status}: {message}")]
    HttpStatus { status: u16, message: String },

    #[error("invalid OpenRouter SSE event: {0}")]
    Sse(String),

    #[error("invalid OpenRouter response: {0}")]
    Protocol(String),

    #[error("OpenRouter provider error {code}: {message}")]
    Provider { code: String, message: String },

    #[error("OpenRouter stream ended before the completion marker")]
    IncompleteStream,

    #[error("OpenRouter produced an empty response")]
    EmptyResponse,

    #[error("OpenRouter response timed out after {} seconds", .0.as_secs())]
    Timeout(Duration),
}
