use std::fmt;

#[derive(Debug)]
pub struct AudioFrame {
    pub sequence: u64,
    pub pcm: Vec<u8>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum DeepgramEvent {
    Status(SttStatus),
    Transcript {
        text: String,
        is_final: bool,
        speech_final: bool,
        audio_start_ms: Option<u64>,
        audio_duration_ms: Option<u64>,
        last_word_end_ms: Option<u64>,
    },
    SpeechStarted {
        audio_timestamp_ms: Option<u64>,
    },
    UtteranceEnd {
        last_word_end_ms: Option<u64>,
    },
    Metadata,
    Error(String),
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum SttStatus {
    Connecting {
        retry_count: u32,
        queue_len: usize,
    },
    Connected {
        queue_len: usize,
    },
    Reconnecting {
        retry_count: u32,
        delay_secs: u64,
        queue_len: usize,
    },
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

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum StatusKind {
    Started,
    Connecting,
    Listening,
    Paused,
    Reconnecting,
    HistoryCleared,
    Stopped,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct StatusMessage {
    pub kind: StatusKind,
    pub text: String,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TranscriptView {
    pub sequence: u64,
    pub text: String,
    pub flush_reason: String,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum SttObservation {
    Transcript {
        text: String,
        is_final: bool,
        speech_final: bool,
        audio_start_ms: Option<u64>,
        audio_duration_ms: Option<u64>,
        last_word_end_ms: Option<u64>,
    },
    SpeechStarted {
        audio_timestamp_ms: Option<u64>,
    },
    UtteranceEnd {
        last_word_end_ms: Option<u64>,
    },
}

#[derive(Clone, Debug, PartialEq)]
pub struct LlmUsage {
    pub prompt_tokens: u64,
    pub completion_tokens: u64,
    pub total_tokens: u64,
    pub cost: Option<f64>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct AnswerMeta {
    pub request_id: u64,
    pub mode: Mode,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum QueueKind {
    Audio,
    Llm,
}

impl fmt::Display for QueueKind {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Audio => formatter.write_str("audio"),
            Self::Llm => formatter.write_str("llm"),
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct QueueState {
    pub queue: QueueKind,
    pub len: usize,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum OutputComponent {
    App,
    Audio,
    Stt,
    Llm,
}

impl fmt::Display for OutputComponent {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::App => formatter.write_str("app"),
            Self::Audio => formatter.write_str("audio"),
            Self::Stt => formatter.write_str("stt"),
            Self::Llm => formatter.write_str("llm"),
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct AppErrorView {
    pub component: OutputComponent,
    pub message: String,
}

#[derive(Clone, Debug, PartialEq)]
pub enum OutputEvent {
    Status(StatusMessage),
    SttObservation(SttObservation),
    TranscriptDraft { text: String },
    Transcript(TranscriptView),
    AnswerStarted(AnswerMeta),
    AnswerDelta { request_id: u64, text: String },
    AnswerUsage { request_id: u64, usage: LlmUsage },
    AnswerCompleted { request_id: u64 },
    QueueState(QueueState),
    Error(AppErrorView),
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum LlmCommand {
    ClearHistory,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum AppCommand {
    PauseListening,
    ResumeListening,
    ClearHistory,
    Shutdown,
}
