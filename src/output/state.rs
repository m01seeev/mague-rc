use crate::events::{AppErrorView, OutputComponent, OutputEvent, QueueKind, StatusKind};

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

#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct AppSnapshot {
    pub running: bool,
    pub listening: bool,
    pub stt_status: ConnectionStatus,
    pub llm_status: WorkerStatus,
    pub current_transcript: String,
    pub current_answer_id: Option<u64>,
    pub current_answer: String,
    pub audio_queue_len: usize,
    pub llm_queue_len: usize,
    pub last_error: Option<AppErrorView>,
}

impl AppSnapshot {
    pub fn apply(&mut self, event: &OutputEvent) {
        match event {
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
                StatusKind::Reconnecting => {
                    self.listening = false;
                    self.stt_status = ConnectionStatus::Reconnecting;
                }
                StatusKind::Stopped => {
                    self.running = false;
                    self.listening = false;
                    self.stt_status = ConnectionStatus::Disconnected;
                    self.llm_status = WorkerStatus::Idle;
                }
            },
            OutputEvent::Transcript(transcript) => {
                self.current_transcript.clone_from(&transcript.text);
            }
            OutputEvent::AnswerStarted(meta) => {
                self.llm_status = WorkerStatus::Working;
                self.current_answer_id = Some(meta.request_id);
                self.current_answer.clear();
                self.last_error = None;
            }
            OutputEvent::AnswerDelta { request_id, text }
                if self.current_answer_id == Some(*request_id) =>
            {
                self.current_answer.push_str(text);
            }
            OutputEvent::AnswerDelta { .. } => {}
            OutputEvent::AnswerCompleted { request_id }
                if self.current_answer_id == Some(*request_id) =>
            {
                self.llm_status = WorkerStatus::Idle;
            }
            OutputEvent::AnswerCompleted { .. } => {}
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
                    OutputComponent::Llm => self.llm_status = WorkerStatus::Error,
                    OutputComponent::App | OutputComponent::Audio => {}
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::events::{
        AnswerMeta, AppErrorView, Mode, OutputComponent, QueueState, StatusMessage, TranscriptView,
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
            text: "Что такое ownership?".to_owned(),
        }));
        snapshot.apply(&OutputEvent::AnswerStarted(AnswerMeta {
            request_id: 4,
            mode: Mode::Voice,
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
    }

    #[test]
    fn ignores_stale_answer_delta_and_tracks_provider_error() {
        let mut snapshot = AppSnapshot::default();
        snapshot.apply(&OutputEvent::AnswerStarted(AnswerMeta {
            request_id: 8,
            mode: Mode::Voice,
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
    }
}
