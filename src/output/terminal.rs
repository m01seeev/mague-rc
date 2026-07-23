use std::io::{self, Write};

use thiserror::Error;
use tokio::sync::mpsc;

use crate::{events::LlmEvent, output::OutputSink};

pub struct TerminalOutputSink;

#[derive(Default)]
pub struct TerminalOutputStats {
    pub started: u64,
    pub completed: u64,
    pub failed: u64,
}

#[derive(Debug, Error)]
pub enum TerminalOutputError {
    #[error("could not write LLM output to terminal: {0}")]
    Write(#[from] io::Error),
}

impl OutputSink for TerminalOutputSink {
    type Error = TerminalOutputError;
    type Stats = TerminalOutputStats;

    async fn run(
        self,
        mut events: mpsc::UnboundedReceiver<LlmEvent>,
    ) -> Result<TerminalOutputStats, TerminalOutputError> {
        let mut stats = TerminalOutputStats::default();

        while let Some(event) = events.recv().await {
            let mut stdout = io::stdout().lock();
            render_event(&mut stdout, event, &mut stats)?;
            stdout.flush()?;
        }

        Ok(stats)
    }
}

fn render_event(
    writer: &mut impl Write,
    event: LlmEvent,
    stats: &mut TerminalOutputStats,
) -> Result<(), io::Error> {
    match event {
        LlmEvent::Started { request_id, mode } => {
            stats.started += 1;
            writeln!(writer, "\nANSWER #{request_id} [{mode}]")?;
        }
        LlmEvent::Delta { text, .. } => {
            write!(writer, "{text}")?;
        }
        LlmEvent::Completed { .. } => {
            stats.completed += 1;
            writeln!(writer)?;
        }
        LlmEvent::Failed { request_id, error } => {
            stats.failed += 1;
            writeln!(writer, "\nLLM ERROR #{request_id}: {error}")?;
        }
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use crate::events::Mode;

    use super::*;

    #[test]
    fn renders_streaming_answer_without_repeating_full_text() {
        let mut output = Vec::new();
        let mut stats = TerminalOutputStats::default();

        render_event(
            &mut output,
            LlmEvent::Started {
                request_id: 7,
                mode: Mode::Voice,
            },
            &mut stats,
        )
        .expect("started event must render");
        render_event(
            &mut output,
            LlmEvent::Delta {
                request_id: 7,
                text: "первая ".to_owned(),
            },
            &mut stats,
        )
        .expect("first delta must render");
        render_event(
            &mut output,
            LlmEvent::Delta {
                request_id: 7,
                text: "вторая".to_owned(),
            },
            &mut stats,
        )
        .expect("second delta must render");
        render_event(
            &mut output,
            LlmEvent::Completed {
                request_id: 7,
                full_text: "первая вторая".to_owned(),
            },
            &mut stats,
        )
        .expect("completion must render");

        assert_eq!(
            String::from_utf8(output).expect("output must be UTF-8"),
            "\nANSWER #7 [voice]\nпервая вторая\n"
        );
        assert_eq!(stats.started, 1);
        assert_eq!(stats.completed, 1);
    }
}
