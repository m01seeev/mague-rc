use std::sync::mpsc;

use thiserror::Error;
use tokio::sync::mpsc as tokio_mpsc;

use crate::{
    events::OutputEvent,
    output::{OutputSink, OutputStats},
};

pub struct ChannelOutputSink {
    sender: mpsc::Sender<OutputEvent>,
}

impl ChannelOutputSink {
    pub fn new(sender: mpsc::Sender<OutputEvent>) -> Self {
        Self { sender }
    }
}

#[derive(Clone, Copy, Debug, Error, PartialEq, Eq)]
#[error("overlay output channel closed")]
pub struct ChannelOutputError;

impl OutputSink for ChannelOutputSink {
    type Error = ChannelOutputError;

    async fn run(
        self,
        mut events: tokio_mpsc::UnboundedReceiver<OutputEvent>,
    ) -> Result<OutputStats, ChannelOutputError> {
        let mut stats = OutputStats::default();

        while let Some(event) = events.recv().await {
            update_stats(&mut stats, &event);
            self.sender.send(event).map_err(|_| ChannelOutputError)?;
        }

        Ok(stats)
    }
}

fn update_stats(stats: &mut OutputStats, event: &OutputEvent) {
    match event {
        OutputEvent::Status(_) => stats.statuses += 1,
        OutputEvent::Transcript(_) => stats.transcripts += 1,
        OutputEvent::AnswerStarted(_) => stats.started += 1,
        OutputEvent::AnswerCompleted { .. } => stats.completed += 1,
        OutputEvent::Error(error) if error.component == crate::events::OutputComponent::Llm => {
            stats.failed += 1;
        }
        OutputEvent::SttObservation { .. }
        | OutputEvent::ModeChanged { .. }
        | OutputEvent::TranscriptDraft { .. }
        | OutputEvent::Retrieval(_)
        | OutputEvent::LlmQueued { .. }
        | OutputEvent::AnswerDelta { .. }
        | OutputEvent::AnswerUsage { .. }
        | OutputEvent::LiveCodingUpdated(_)
        | OutputEvent::QueueState(_)
        | OutputEvent::Error(_) => {}
    }
}

#[cfg(test)]
mod tests {
    use crate::events::{AnswerMeta, Mode, Speaker};

    use super::*;

    #[tokio::test]
    async fn forwards_events_and_collects_stats() {
        let (source_sender, source_receiver) = tokio_mpsc::unbounded_channel();
        let (target_sender, target_receiver) = mpsc::channel();
        source_sender
            .send(OutputEvent::AnswerStarted(AnswerMeta {
                request_id: 2,
                mode: Mode::Voice,
                speaker: Speaker::Interviewer,
            }))
            .expect("source channel must be open");
        source_sender
            .send(OutputEvent::AnswerCompleted { request_id: 2 })
            .expect("source channel must be open");
        drop(source_sender);

        let stats = ChannelOutputSink::new(target_sender)
            .run(source_receiver)
            .await
            .expect("sink must finish");

        assert_eq!(target_receiver.into_iter().count(), 2);
        assert_eq!(stats.started, 1);
        assert_eq!(stats.completed, 1);
    }
}
