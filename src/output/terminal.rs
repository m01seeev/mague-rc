use std::{
    collections::VecDeque,
    io::{self, Write},
};

use thiserror::Error;
use tokio::sync::mpsc;

use crate::{
    events::{OutputComponent, OutputEvent, StatusKind},
    output::{OutputSink, OutputStats},
};

pub struct TerminalOutputSink;

#[derive(Debug, Error)]
pub enum TerminalOutputError {
    #[error("could not write LLM output to terminal: {0}")]
    Write(#[from] io::Error),
}

impl OutputSink for TerminalOutputSink {
    type Error = TerminalOutputError;

    async fn run(
        self,
        mut events: mpsc::UnboundedReceiver<OutputEvent>,
    ) -> Result<OutputStats, TerminalOutputError> {
        let mut renderer = TerminalRenderer::default();

        while let Some(event) = events.recv().await {
            let mut stdout = io::stdout().lock();
            renderer.render(&mut stdout, event)?;
            stdout.flush()?;
        }

        Ok(renderer.stats)
    }
}

#[derive(Default)]
struct TerminalRenderer {
    active_answer: Option<u64>,
    pending: VecDeque<OutputEvent>,
    stats: OutputStats,
}

impl TerminalRenderer {
    fn render(&mut self, writer: &mut impl Write, event: OutputEvent) -> Result<(), io::Error> {
        match event {
            OutputEvent::AnswerStarted(meta) if self.active_answer.is_none() => {
                self.stats.started += 1;
                self.active_answer = Some(meta.request_id);
                writeln!(writer, "\nANSWER #{} [{}]", meta.request_id, meta.mode)?;
            }
            OutputEvent::AnswerDelta { request_id, text }
                if self.active_answer == Some(request_id) =>
            {
                write!(writer, "{text}")?;
            }
            OutputEvent::AnswerCompleted { request_id }
                if self.active_answer == Some(request_id) =>
            {
                self.stats.completed += 1;
                self.active_answer = None;
                writeln!(writer)?;
                self.flush_pending(writer)?;
            }
            OutputEvent::Error(error)
                if self.active_answer.is_some() && error.component == OutputComponent::Llm =>
            {
                self.stats.failed += 1;
                self.active_answer = None;
                writeln!(writer)?;
                writeln!(writer, "ERROR [{}]: {}", error.component, error.message)?;
                self.flush_pending(writer)?;
            }
            OutputEvent::SttObservation(_)
            | OutputEvent::TranscriptDraft { .. }
            | OutputEvent::AnswerUsage { .. } => {}
            OutputEvent::AnswerDelta { .. } | OutputEvent::AnswerCompleted { .. } => {}
            event if self.active_answer.is_some() => self.pending.push_back(event),
            event => self.render_passive(writer, event)?,
        }
        Ok(())
    }

    fn flush_pending(&mut self, writer: &mut impl Write) -> Result<(), io::Error> {
        while self.active_answer.is_none() {
            let Some(event) = self.pending.pop_front() else {
                break;
            };
            self.render(writer, event)?;
        }
        Ok(())
    }

    fn render_passive(
        &mut self,
        writer: &mut impl Write,
        event: OutputEvent,
    ) -> Result<(), io::Error> {
        match event {
            OutputEvent::Status(status) => {
                self.stats.statuses += 1;
                let label = match status.kind {
                    StatusKind::Started => "started",
                    StatusKind::Connecting => "connecting",
                    StatusKind::Listening => "listening",
                    StatusKind::Paused => "paused",
                    StatusKind::Reconnecting => "reconnecting",
                    StatusKind::HistoryCleared => "history",
                    StatusKind::Stopped => "stopped",
                };
                writeln!(writer, "[{label}] {}", status.text)?;
            }
            OutputEvent::Transcript(transcript) => {
                self.stats.transcripts += 1;
                writeln!(
                    writer,
                    "\nQUESTION #{}: {}",
                    transcript.sequence, transcript.text
                )?;
            }
            OutputEvent::SttObservation(_)
            | OutputEvent::TranscriptDraft { .. }
            | OutputEvent::AnswerUsage { .. } => {}
            OutputEvent::AnswerStarted(meta) => {
                self.stats.started += 1;
                self.active_answer = Some(meta.request_id);
                writeln!(writer, "\nANSWER #{} [{}]", meta.request_id, meta.mode)?;
            }
            OutputEvent::AnswerDelta { .. } | OutputEvent::AnswerCompleted { .. } => {}
            OutputEvent::QueueState(queue) if queue.len > 1 => {
                writeln!(writer, "[queue:{}] {} pending", queue.queue, queue.len)?;
            }
            OutputEvent::QueueState(_) => {}
            OutputEvent::Error(error) => {
                if error.component == OutputComponent::Llm {
                    self.stats.failed += 1;
                }
                writeln!(writer, "ERROR [{}]: {}", error.component, error.message)?;
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use crate::events::{
        AnswerMeta, AppErrorView, Mode, OutputComponent, StatusMessage, TranscriptView,
    };

    use super::*;

    #[test]
    fn renders_streaming_answer_without_repeating_full_text() {
        let mut output = Vec::new();
        let mut renderer = TerminalRenderer::default();

        renderer
            .render(
                &mut output,
                OutputEvent::AnswerStarted(AnswerMeta {
                    request_id: 7,
                    mode: Mode::Voice,
                }),
            )
            .expect("started event must render");
        renderer
            .render(
                &mut output,
                OutputEvent::AnswerDelta {
                    request_id: 7,
                    text: "первая ".to_owned(),
                },
            )
            .expect("first delta must render");
        renderer
            .render(
                &mut output,
                OutputEvent::AnswerDelta {
                    request_id: 7,
                    text: "вторая".to_owned(),
                },
            )
            .expect("second delta must render");
        renderer
            .render(&mut output, OutputEvent::AnswerCompleted { request_id: 7 })
            .expect("completion must render");

        assert_eq!(
            String::from_utf8(output).expect("output must be UTF-8"),
            "\nANSWER #7 [voice]\nпервая вторая\n"
        );
        assert_eq!(renderer.stats.started, 1);
        assert_eq!(renderer.stats.completed, 1);
    }

    #[test]
    fn defers_status_and_transcript_until_streaming_answer_finishes() {
        let mut output = Vec::new();
        let mut renderer = TerminalRenderer::default();

        renderer
            .render(
                &mut output,
                OutputEvent::AnswerStarted(AnswerMeta {
                    request_id: 3,
                    mode: Mode::Voice,
                }),
            )
            .expect("answer must start");
        renderer
            .render(
                &mut output,
                OutputEvent::AnswerDelta {
                    request_id: 3,
                    text: "цельный ответ".to_owned(),
                },
            )
            .expect("delta must render");
        renderer
            .render(
                &mut output,
                OutputEvent::Status(StatusMessage {
                    kind: StatusKind::Reconnecting,
                    text: "Deepgram reconnect in 1s".to_owned(),
                }),
            )
            .expect("status must buffer");
        renderer
            .render(
                &mut output,
                OutputEvent::Transcript(TranscriptView {
                    sequence: 4,
                    text: "Следующий вопрос".to_owned(),
                    flush_reason: "test".to_owned(),
                }),
            )
            .expect("transcript must buffer");
        renderer
            .render(&mut output, OutputEvent::AnswerCompleted { request_id: 3 })
            .expect("answer must complete");

        assert_eq!(
            String::from_utf8(output).expect("output must be UTF-8"),
            "\nANSWER #3 [voice]\nцельный ответ\n\
             [reconnecting] Deepgram reconnect in 1s\n\
             \nQUESTION #4: Следующий вопрос\n"
        );
    }

    #[test]
    fn llm_error_closes_stream_before_rendering_pending_events() {
        let mut output = Vec::new();
        let mut renderer = TerminalRenderer::default();

        renderer
            .render(
                &mut output,
                OutputEvent::AnswerStarted(AnswerMeta {
                    request_id: 9,
                    mode: Mode::Voice,
                }),
            )
            .expect("answer must start");
        renderer
            .render(
                &mut output,
                OutputEvent::Status(StatusMessage {
                    kind: StatusKind::Listening,
                    text: "Deepgram connected".to_owned(),
                }),
            )
            .expect("status must buffer");
        renderer
            .render(
                &mut output,
                OutputEvent::Error(AppErrorView {
                    component: OutputComponent::Llm,
                    message: "request #9 failed: timeout".to_owned(),
                }),
            )
            .expect("error must render");

        assert_eq!(
            String::from_utf8(output).expect("output must be UTF-8"),
            "\nANSWER #9 [voice]\n\n\
             ERROR [llm]: request #9 failed: timeout\n\
             [listening] Deepgram connected\n"
        );
        assert_eq!(renderer.stats.failed, 1);
    }
}
