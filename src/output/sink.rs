use std::future::Future;

use tokio::sync::mpsc;

use crate::{events::OutputEvent, output::OutputStats};

pub trait OutputSink: Send + 'static {
    type Error: Send + 'static;

    fn run(
        self,
        events: mpsc::UnboundedReceiver<OutputEvent>,
    ) -> impl Future<Output = Result<OutputStats, Self::Error>> + Send;
}
