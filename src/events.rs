use std::fmt;

#[derive(Debug)]
pub struct AudioFrame {
    pub sequence: u64,
    pub pcm: Vec<u8>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum DeepgramEvent {
    Transcript {
        text: String,
        is_final: bool,
        speech_final: bool,
    },
    SpeechStarted,
    UtteranceEnd,
    Metadata,
    Error(String),
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TranscriptChunk {
    pub sequence: u64,
    pub text: String,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Mode {
    Voice,
    Ocr,
}

impl fmt::Display for Mode {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Voice => formatter.write_str("voice"),
            Self::Ocr => formatter.write_str("ocr"),
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct LlmRequest {
    pub request_id: u64,
    pub mode: Mode,
    pub text: String,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum LlmEvent {
    Started { request_id: u64, mode: Mode },
    Delta { request_id: u64, text: String },
    Completed { request_id: u64, full_text: String },
    Failed { request_id: u64, error: String },
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum LlmCommand {
    ClearHistory,
}
