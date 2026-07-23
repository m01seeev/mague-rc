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
