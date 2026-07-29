use crate::events::{
    AppErrorView, Mode, OutputComponent, OutputEvent, QueueKind, Speaker, StatusKind,
};

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum ConnectionStatus {
    #[default]
    Disconnected,
    Connecting,
    Connected,
    Reconnecting,
    Error,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum WorkerStatus {
    #[default]
    Idle,
    Working,
    Error,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum AnswerStatus {
    #[default]
    Pending,
    Streaming,
    Completed,
    Failed,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ConversationTurn {
    pub request_id: u64,
    pub speaker: Speaker,
    pub question: String,
    pub answer: String,
    pub answer_status: AnswerStatus,
}

#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct AppSnapshot {
    pub mode: Mode,
    pub running: bool,
    pub listening: bool,
    pub stt_status: ConnectionStatus,
    pub llm_status: WorkerStatus,
    pub transcript_draft: String,
    pub current_transcript: String,
    pub current_answer_id: Option<u64>,
    pub current_answer: String,
    pub audio_queue_len: usize,
    pub llm_queue_len: usize,
    pub last_error: Option<AppErrorView>,
    pub conversation: Vec<ConversationTurn>,
}

impl AppSnapshot {
    pub fn apply(&mut self, event: &OutputEvent) {
        match event {
            OutputEvent::ModeChanged { mode } => self.mode = *mode,
            OutputEvent::Status(status) => match status.kind {
                StatusKind::Started => {
                    self.running = true;
                    self.last_error = None;
                }
                StatusKind::Connecting => {
                    self.listening = false;
                    self.stt_status = ConnectionStatus::Connecting;
                }
                StatusKind::Listening => {
                    self.listening = true;
                    self.stt_status = ConnectionStatus::Connected;
                    self.last_error = None;
                }
                StatusKind::Paused => self.listening = false,
                StatusKind::Reconnecting => {
                    self.listening = false;
                    self.stt_status = ConnectionStatus::Reconnecting;
                }
                StatusKind::HistoryCleared => {
                    self.transcript_draft.clear();
                    self.current_transcript.clear();
                    self.current_answer_id = None;
                    self.current_answer.clear();
                    self.conversation.clear();
                }
                StatusKind::Stopped => {
                    self.running = false;
                    self.listening = false;
                    self.stt_status = ConnectionStatus::Disconnected;
                    self.llm_status = WorkerStatus::Idle;
                }
            },
            OutputEvent::TranscriptDraft {
                speaker: Speaker::Interviewer,
                text,
            } => {
                self.transcript_draft.clone_from(text);
            }
            OutputEvent::TranscriptDraft { .. } => {}
            OutputEvent::Transcript(transcript) if transcript.mode != Mode::LiveCoding => {
                self.current_transcript.clone_from(&transcript.text);
                if let Some(turn) = self
                    .conversation
                    .iter_mut()
                    .find(|turn| turn.request_id == transcript.sequence)
                {
                    turn.question.clone_from(&transcript.text);
                } else {
                    self.conversation.push(ConversationTurn {
                        request_id: transcript.sequence,
                        speaker: transcript.speaker,
                        question: transcript.text.clone(),
                        answer: String::new(),
                        answer_status: AnswerStatus::Pending,
                    });
                }
            }
            OutputEvent::Transcript(_) => {}
            OutputEvent::AnswerStarted(meta) => {
                self.llm_status = WorkerStatus::Working;
                self.current_answer_id = Some(meta.request_id);
                self.current_answer.clear();
                self.last_error = None;
                if let Some(turn) = self
                    .conversation
                    .iter_mut()
                    .find(|turn| turn.request_id == meta.request_id)
                {
                    turn.answer.clear();
                    turn.answer_status = AnswerStatus::Streaming;
                }
            }
            OutputEvent::AnswerDelta { request_id, text }
                if self.current_answer_id == Some(*request_id) =>
            {
                self.current_answer.push_str(text);
                if let Some(turn) = self
                    .conversation
                    .iter_mut()
                    .find(|turn| turn.request_id == *request_id)
                {
                    turn.answer.push_str(text);
                }
            }
            OutputEvent::AnswerDelta { .. } => {}
            OutputEvent::AnswerCompleted { request_id }
                if self.current_answer_id == Some(*request_id) =>
            {
                self.llm_status = WorkerStatus::Idle;
                if let Some(turn) = self
                    .conversation
                    .iter_mut()
                    .find(|turn| turn.request_id == *request_id)
                {
                    turn.answer_status = AnswerStatus::Completed;
                }
            }
            OutputEvent::AnswerCompleted { .. } => {}
            OutputEvent::SttObservation { .. }
            | OutputEvent::Retrieval(_)
            | OutputEvent::LlmQueued { .. }
            | OutputEvent::AnswerUsage { .. }
            | OutputEvent::LiveCodingUpdated(_) => {}
            OutputEvent::QueueState(queue) => match queue.queue {
                QueueKind::Audio => self.audio_queue_len = queue.len,
                QueueKind::Llm => self.llm_queue_len = queue.len,
            },
            OutputEvent::Error(error) => {
                self.last_error = Some(error.clone());
                match error.component {
                    OutputComponent::Stt => {
                        self.listening = false;
                        self.stt_status = ConnectionStatus::Error;
                    }
                    OutputComponent::Llm => {
                        self.llm_status = WorkerStatus::Error;
                        if let Some(request_id) = self.current_answer_id
                            && let Some(turn) = self
                                .conversation
                                .iter_mut()
                                .find(|turn| turn.request_id == request_id)
                        {
                            turn.answer_status = AnswerStatus::Failed;
                        }
                    }
                    OutputComponent::App | OutputComponent::Audio | OutputComponent::Knowledge => {}
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::events::{
        AnswerMeta, AppErrorView, Mode, OutputComponent, QueueState, Speaker, StatusMessage,
        TranscriptView,
    };

    use super::*;

    #[test]
    fn reduces_voice_pipeline_events_into_ui_snapshot() {
        let mut snapshot = AppSnapshot::default();

        snapshot.apply(&OutputEvent::Status(StatusMessage {
            kind: StatusKind::Started,
            text: "started".to_owned(),
        }));
        snapshot.apply(&OutputEvent::Status(StatusMessage {
            kind: StatusKind::Listening,
            text: "listening".to_owned(),
        }));
        snapshot.apply(&OutputEvent::Transcript(TranscriptView {
            sequence: 4,
            mode: Mode::Voice,
            speaker: Speaker::Interviewer,
            text: "Что такое ownership?".to_owned(),
            flush_reason: "test".to_owned(),
        }));
        snapshot.apply(&OutputEvent::AnswerStarted(AnswerMeta {
            request_id: 4,
            mode: Mode::Voice,
            speaker: Speaker::Interviewer,
        }));
        snapshot.apply(&OutputEvent::AnswerDelta {
            request_id: 4,
            text: "Ownership ".to_owned(),
        });
        snapshot.apply(&OutputEvent::AnswerDelta {
            request_id: 4,
            text: "управляет временем жизни данных.".to_owned(),
        });
        snapshot.apply(&OutputEvent::QueueState(QueueState {
            queue: QueueKind::Llm,
            len: 2,
        }));
        snapshot.apply(&OutputEvent::AnswerCompleted { request_id: 4 });

        assert!(snapshot.running);
        assert!(snapshot.listening);
        assert_eq!(snapshot.stt_status, ConnectionStatus::Connected);
        assert_eq!(snapshot.llm_status, WorkerStatus::Idle);
        assert_eq!(snapshot.current_transcript, "Что такое ownership?");
        assert_eq!(
            snapshot.current_answer,
            "Ownership управляет временем жизни данных."
        );
        assert_eq!(snapshot.llm_queue_len, 2);
        assert_eq!(
            snapshot.conversation,
            vec![ConversationTurn {
                request_id: 4,
                speaker: Speaker::Interviewer,
                question: "Что такое ownership?".to_owned(),
                answer: "Ownership управляет временем жизни данных.".to_owned(),
                answer_status: AnswerStatus::Completed,
            }]
        );
    }

    #[test]
    fn ignores_stale_answer_delta_and_tracks_provider_error() {
        let mut snapshot = AppSnapshot::default();
        snapshot.apply(&OutputEvent::Transcript(TranscriptView {
            sequence: 8,
            mode: Mode::Voice,
            speaker: Speaker::Interviewer,
            text: "Что такое borrow checker?".to_owned(),
            flush_reason: "test".to_owned(),
        }));
        snapshot.apply(&OutputEvent::AnswerStarted(AnswerMeta {
            request_id: 8,
            mode: Mode::Voice,
            speaker: Speaker::Interviewer,
        }));
        snapshot.apply(&OutputEvent::AnswerDelta {
            request_id: 7,
            text: "stale".to_owned(),
        });
        snapshot.apply(&OutputEvent::Error(AppErrorView {
            component: OutputComponent::Llm,
            message: "timeout".to_owned(),
        }));

        assert!(snapshot.current_answer.is_empty());
        assert_eq!(snapshot.llm_status, WorkerStatus::Error);
        assert_eq!(
            snapshot
                .last_error
                .as_ref()
                .map(|error| error.message.as_str()),
            Some("timeout")
        );
        assert_eq!(snapshot.conversation[0].answer_status, AnswerStatus::Failed);
    }

    #[test]
    fn tracks_queued_turns_by_request_id_and_clears_them() {
        let mut snapshot = AppSnapshot::default();
        snapshot.apply(&OutputEvent::TranscriptDraft {
            speaker: Speaker::Interviewer,
            text: "Черновик".to_owned(),
        });
        snapshot.apply(&OutputEvent::Transcript(TranscriptView {
            sequence: 1,
            mode: Mode::Voice,
            speaker: Speaker::Interviewer,
            text: "Первый вопрос".to_owned(),
            flush_reason: "test".to_owned(),
        }));
        snapshot.apply(&OutputEvent::Transcript(TranscriptView {
            sequence: 2,
            mode: Mode::Voice,
            speaker: Speaker::Interviewer,
            text: "Второй вопрос".to_owned(),
            flush_reason: "test".to_owned(),
        }));
        snapshot.apply(&OutputEvent::AnswerStarted(AnswerMeta {
            request_id: 1,
            mode: Mode::Voice,
            speaker: Speaker::Interviewer,
        }));
        snapshot.apply(&OutputEvent::AnswerDelta {
            request_id: 1,
            text: "Первый ответ".to_owned(),
        });
        snapshot.apply(&OutputEvent::AnswerCompleted { request_id: 1 });

        assert_eq!(snapshot.conversation.len(), 2);
        assert_eq!(snapshot.conversation[0].answer, "Первый ответ");
        assert_eq!(
            snapshot.conversation[0].answer_status,
            AnswerStatus::Completed
        );
        assert_eq!(
            snapshot.conversation[1].answer_status,
            AnswerStatus::Pending
        );

        snapshot.apply(&OutputEvent::Status(StatusMessage {
            kind: StatusKind::HistoryCleared,
            text: "cleared".to_owned(),
        }));

        assert!(snapshot.conversation.is_empty());
        assert!(snapshot.transcript_draft.is_empty());
    }
}
