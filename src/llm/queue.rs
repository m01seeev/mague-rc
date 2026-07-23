use std::sync::{
    Arc,
    atomic::{AtomicUsize, Ordering},
};

use tokio::sync::mpsc;

use crate::events::LlmRequest;

#[derive(Clone)]
pub struct LlmRequestSender {
    inner: SenderKind,
    queued: Arc<AtomicUsize>,
}

pub struct LlmRequestReceiver {
    inner: ReceiverKind,
    queued: Arc<AtomicUsize>,
}

#[derive(Clone)]
enum SenderKind {
    Bounded(mpsc::Sender<LlmRequest>),
    Unbounded(mpsc::UnboundedSender<LlmRequest>),
}

enum ReceiverKind {
    Bounded(mpsc::Receiver<LlmRequest>),
    Unbounded(mpsc::UnboundedReceiver<LlmRequest>),
}

#[derive(Clone, Copy, Debug, thiserror::Error, PartialEq, Eq)]
#[error("LLM request queue closed")]
pub struct LlmQueueError;

pub fn llm_request_channel(maximum: usize) -> (LlmRequestSender, LlmRequestReceiver) {
    let queued = Arc::new(AtomicUsize::new(0));
    let (sender, receiver) = if maximum == 0 {
        let (sender, receiver) = mpsc::unbounded_channel();
        (
            SenderKind::Unbounded(sender),
            ReceiverKind::Unbounded(receiver),
        )
    } else {
        let (sender, receiver) = mpsc::channel(maximum);
        (SenderKind::Bounded(sender), ReceiverKind::Bounded(receiver))
    };

    (
        LlmRequestSender {
            inner: sender,
            queued: Arc::clone(&queued),
        },
        LlmRequestReceiver {
            inner: receiver,
            queued,
        },
    )
}

impl LlmRequestSender {
    pub async fn send(&self, request: LlmRequest) -> Result<(), LlmQueueError> {
        let reservation = QueueReservation::new(&self.queued);
        let result = match &self.inner {
            SenderKind::Bounded(sender) => sender.send(request).await.map_err(|_| LlmQueueError),
            SenderKind::Unbounded(sender) => sender.send(request).map_err(|_| LlmQueueError),
        };
        if result.is_ok() {
            reservation.commit();
        }
        result
    }

    pub fn len(&self) -> usize {
        self.queued.load(Ordering::Relaxed)
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

impl LlmRequestReceiver {
    pub async fn recv(&mut self) -> Option<LlmRequest> {
        let request = match &mut self.inner {
            ReceiverKind::Bounded(receiver) => receiver.recv().await,
            ReceiverKind::Unbounded(receiver) => receiver.recv().await,
        };
        if request.is_some() {
            self.queued.fetch_sub(1, Ordering::Relaxed);
        }
        request
    }

    pub fn len(&self) -> usize {
        self.queued.load(Ordering::Relaxed)
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

struct QueueReservation<'a> {
    queued: &'a AtomicUsize,
    committed: bool,
}

impl<'a> QueueReservation<'a> {
    fn new(queued: &'a AtomicUsize) -> Self {
        queued.fetch_add(1, Ordering::Relaxed);
        Self {
            queued,
            committed: false,
        }
    }

    fn commit(mut self) {
        self.committed = true;
    }
}

impl Drop for QueueReservation<'_> {
    fn drop(&mut self) {
        if !self.committed {
            self.queued.fetch_sub(1, Ordering::Relaxed);
        }
    }
}

#[cfg(test)]
mod tests {
    use std::time::Duration;

    use tokio::time::timeout;

    use crate::events::Mode;

    use super::*;

    fn request(request_id: u64) -> LlmRequest {
        LlmRequest {
            request_id,
            mode: Mode::Voice,
            text: format!("request {request_id}"),
        }
    }

    #[tokio::test]
    async fn unbounded_queue_preserves_request_order() {
        let (sender, mut receiver) = llm_request_channel(0);
        sender.send(request(1)).await.expect("send must succeed");
        sender.send(request(2)).await.expect("send must succeed");

        assert_eq!(receiver.recv().await.map(|item| item.request_id), Some(1));
        assert_eq!(receiver.recv().await.map(|item| item.request_id), Some(2));
    }

    #[tokio::test]
    async fn bounded_queue_applies_backpressure() {
        let (sender, mut receiver) = llm_request_channel(1);
        sender
            .send(request(1))
            .await
            .expect("first request must fit");

        let mut pending_send = Box::pin(sender.send(request(2)));
        assert!(
            timeout(Duration::from_millis(10), &mut pending_send)
                .await
                .is_err()
        );

        assert_eq!(receiver.recv().await.map(|item| item.request_id), Some(1));
        timeout(Duration::from_millis(100), pending_send)
            .await
            .expect("second send must unblock")
            .expect("second send must succeed");
        assert_eq!(receiver.recv().await.map(|item| item.request_id), Some(2));
    }
}
