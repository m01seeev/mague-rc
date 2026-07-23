use std::future::Future;

use tokio::sync::{mpsc, watch};

use crate::{audio::AudioFrameReceiver, events::DeepgramEvent, stt::SttError};

pub trait SpeechToTextProvider: Send + 'static {
    fn run(
        self,
        audio: AudioFrameReceiver,
        events: mpsc::UnboundedSender<DeepgramEvent>,
        shutdown: watch::Receiver<bool>,
    ) -> impl Future<Output = Result<(), SttError>> + Send;
}
