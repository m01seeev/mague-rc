use std::{
    future::Future,
    sync::{
        Arc,
        atomic::{AtomicUsize, Ordering},
    },
};

use tokio::sync::{mpsc, watch};

use crate::{audio::AudioError, events::AudioFrame};

pub trait AudioSource: Send + 'static {
    fn run(
        self,
        output: AudioFrameSender,
        shutdown: watch::Receiver<bool>,
    ) -> impl Future<Output = Result<(), AudioError>> + Send;
}

#[derive(Clone)]
pub struct AudioFrameSender {
    inner: SenderKind,
    queued: Arc<AtomicUsize>,
    maximum: Option<usize>,
}

pub struct AudioFrameReceiver {
    inner: ReceiverKind,
    queued: Arc<AtomicUsize>,
}

#[derive(Clone)]
enum SenderKind {
    Bounded(mpsc::Sender<AudioFrame>),
    Unbounded(mpsc::UnboundedSender<AudioFrame>),
}

enum ReceiverKind {
    Bounded(mpsc::Receiver<AudioFrame>),
    Unbounded(mpsc::UnboundedReceiver<AudioFrame>),
}

pub fn audio_frame_channel(maximum: usize) -> (AudioFrameSender, AudioFrameReceiver) {
    let queued = Arc::new(AtomicUsize::new(0));

    let (sender, receiver, maximum) = if maximum == 0 {
        let (sender, receiver) = mpsc::unbounded_channel();
        (
            SenderKind::Unbounded(sender),
            ReceiverKind::Unbounded(receiver),
            None,
        )
    } else {
        let (sender, receiver) = mpsc::channel(maximum);
        (
            SenderKind::Bounded(sender),
            ReceiverKind::Bounded(receiver),
            Some(maximum),
        )
    };

    (
        AudioFrameSender {
            inner: sender,
            queued: Arc::clone(&queued),
            maximum,
        },
        AudioFrameReceiver {
            inner: receiver,
            queued,
        },
    )
}

impl AudioFrameSender {
    pub async fn send(&self, frame: AudioFrame) -> Result<(), AudioError> {
        let reservation = QueueReservation::new(&self.queued);

        let result = match &self.inner {
            SenderKind::Bounded(sender) => sender.send(frame).await.map_err(|_| ()),
            SenderKind::Unbounded(sender) => sender.send(frame).map_err(|_| ()),
        };

        if result.is_err() {
            return Err(AudioError::ChannelClosed);
        }

        reservation.commit();
        Ok(())
    }

    pub fn len(&self) -> usize {
        self.queued.load(Ordering::Relaxed)
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    pub fn is_full(&self) -> bool {
        self.maximum.is_some_and(|maximum| self.len() >= maximum)
    }
}

impl AudioFrameReceiver {
    pub async fn recv(&mut self) -> Option<AudioFrame> {
        let frame = match &mut self.inner {
            ReceiverKind::Bounded(receiver) => receiver.recv().await,
            ReceiverKind::Unbounded(receiver) => receiver.recv().await,
        };

        if frame.is_some() {
            self.queued.fetch_sub(1, Ordering::Relaxed);
        }
        frame
    }

    pub fn len(&self) -> usize {
        self.queued.load(Ordering::Relaxed)
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

impl std::fmt::Debug for AudioFrameSender {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("AudioFrameSender")
            .field("queue_len", &self.len())
            .field("maximum", &self.maximum)
            .finish()
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

    use super::*;

    fn frame(sequence: u64) -> AudioFrame {
        AudioFrame {
            sequence,
            pcm: vec![sequence as u8],
        }
    }

    #[tokio::test]
    async fn unbounded_queue_preserves_order() {
        let (sender, mut receiver) = audio_frame_channel(0);

        sender.send(frame(1)).await.expect("send must succeed");
        sender.send(frame(2)).await.expect("send must succeed");

        assert_eq!(sender.len(), 2);
        assert_eq!(receiver.recv().await.map(|item| item.sequence), Some(1));
        assert_eq!(receiver.recv().await.map(|item| item.sequence), Some(2));
        assert!(sender.is_empty());
    }

    #[tokio::test]
    async fn bounded_queue_applies_backpressure() {
        let (sender, mut receiver) = audio_frame_channel(1);
        sender.send(frame(1)).await.expect("first send must fit");

        let mut pending_send = Box::pin(sender.send(frame(2)));
        assert!(
            timeout(Duration::from_millis(10), &mut pending_send)
                .await
                .is_err()
        );
        assert_eq!(sender.len(), 2);

        assert_eq!(receiver.recv().await.map(|item| item.sequence), Some(1));
        timeout(Duration::from_millis(100), pending_send)
            .await
            .expect("second send must unblock")
            .expect("second send must succeed");
        assert_eq!(receiver.recv().await.map(|item| item.sequence), Some(2));
        assert!(sender.is_empty());
    }
}
