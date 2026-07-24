use std::future::Future;

use tokio::sync::mpsc;

use crate::events::OutputEvent;

pub trait OutputSink: Send + 'static {
    type Error: Send + 'static;
    type Stats: Send + 'static;

    fn run(
        self,
        events: mpsc::UnboundedReceiver<OutputEvent>,
    ) -> impl Future<Output = Result<Self::Stats, Self::Error>> + Send;
}
