mod channel;
mod sink;
mod state;
mod terminal;

pub use channel::{ChannelOutputError, ChannelOutputSink};
pub use sink::OutputSink;
pub use state::{AppSnapshot, ConnectionStatus, WorkerStatus};
pub use terminal::{TerminalOutputError, TerminalOutputSink};

#[derive(Default)]
pub struct OutputStats {
    pub statuses: u64,
    pub transcripts: u64,
    pub started: u64,
    pub completed: u64,
    pub failed: u64,
}
