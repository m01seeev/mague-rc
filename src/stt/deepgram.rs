use std::{
    future::Future,
    time::{Duration, Instant},
};

use futures_util::{
    SinkExt, StreamExt,
    stream::{SplitSink, SplitStream},
};
use thiserror::Error;
use tokio::{
    net::TcpStream,
    sync::{mpsc, watch},
    task::{JoinError, JoinHandle},
    time::{MissedTickBehavior, interval, sleep, timeout},
};
use tokio_tungstenite::{
    MaybeTlsStream, WebSocketStream, connect_async,
    tungstenite::{
        Message,
        client::IntoClientRequest,
        http::{HeaderValue, header::AUTHORIZATION},
    },
};
use tracing::{debug, info, warn};
use url::Url;

use crate::{
    audio::AudioFrameReceiver,
    config::DeepgramConfig,
    events::{AudioFrame, DeepgramEvent},
    stt::{SpeechToTextProvider, parse_server_message},
};

const KEEPALIVE_INTERVAL: Duration = Duration::from_secs(4);
const CONNECT_TIMEOUT: Duration = Duration::from_secs(15);
const INITIAL_RECONNECT_DELAY: Duration = Duration::from_secs(1);
const MAX_RECONNECT_DELAY: Duration = Duration::from_secs(30);
const STABLE_CONNECTION_TIME: Duration = Duration::from_secs(30);
const KEEPALIVE_MESSAGE: &str = r#"{"type":"KeepAlive"}"#;
const CLOSE_STREAM_MESSAGE: &str = r#"{"type":"CloseStream"}"#;

type DeepgramSocket = WebSocketStream<MaybeTlsStream<TcpStream>>;
type DeepgramWriter = SplitSink<DeepgramSocket, Message>;
type DeepgramReader = SplitStream<DeepgramSocket>;

#[derive(Debug, Error)]
pub enum SttError {
    #[error("failed to install the rustls crypto provider")]
    CryptoProvider,

    #[error("audio frame channel closed while STT was running")]
    AudioChannelClosed,

    #[error("Deepgram event channel closed")]
    EventChannelClosed,

    #[error("invalid Deepgram authorization header: {0}")]
    AuthorizationHeader(String),

    #[error("Deepgram task failed: {0}")]
    Task(String),
}

pub fn install_tls_crypto_provider() -> Result<(), SttError> {
    use rustls::crypto::CryptoProvider;

    if CryptoProvider::get_default().is_some() {
        return Ok(());
    }

    if rustls::crypto::ring::default_provider()
        .install_default()
        .is_err()
        && CryptoProvider::get_default().is_none()
    {
        return Err(SttError::CryptoProvider);
    }

    Ok(())
}

pub struct DeepgramSttProvider {
    config: DeepgramConfig,
}

impl DeepgramSttProvider {
    pub fn new(config: DeepgramConfig) -> Self {
        Self { config }
    }

    async fn run_loop(
        self,
        mut audio: AudioFrameReceiver,
        events: mpsc::UnboundedSender<DeepgramEvent>,
        mut shutdown: watch::Receiver<bool>,
    ) -> Result<(), SttError> {
        let mut retry_frame = None;
        let mut retry_count = 0_u32;

        loop {
            if shutdown_requested(&shutdown) {
                return Ok(());
            }

            let queue_len = audio.len() + usize::from(retry_frame.is_some());
            info!(
                module = "stt",
                event = "connecting",
                provider = "deepgram",
                model = %self.config.model,
                retry_count,
                queue_len,
                "connecting to Deepgram"
            );

            let connect_result = tokio::select! {
                _ = wait_for_shutdown(&mut shutdown) => return Ok(()),
                result = self.connect() => result,
            };
            let socket = match connect_result {
                Ok(socket) => socket,
                Err(error) => {
                    retry_count = retry_count.saturating_add(1);
                    emit_error(&events, format!("Deepgram connection failed: {error}"))?;
                    wait_before_reconnect(retry_count, &mut shutdown).await;
                    continue;
                }
            };

            info!(
                module = "stt",
                event = "connected",
                provider = "deepgram",
                model = %self.config.model,
                queue_len,
                "Deepgram connected"
            );
            let connected_at = Instant::now();
            let result = run_connection(
                socket,
                audio,
                retry_frame.take(),
                events.clone(),
                shutdown.clone(),
            )
            .await?;
            audio = result.audio;
            retry_frame = result.retry_frame;

            if matches!(result.outcome, ConnectionOutcome::Shutdown)
                || shutdown_requested(&shutdown)
            {
                return Ok(());
            }

            if connected_at.elapsed() >= STABLE_CONNECTION_TIME {
                retry_count = 0;
            }
            retry_count = retry_count.saturating_add(1);

            match result.outcome {
                ConnectionOutcome::Shutdown => return Ok(()),
                ConnectionOutcome::Disconnected(reason) => {
                    emit_error(&events, format!("Deepgram disconnected: {reason}"))?;
                }
                ConnectionOutcome::AudioChannelClosed => {
                    return Err(SttError::AudioChannelClosed);
                }
                ConnectionOutcome::EventChannelClosed => {
                    return Err(SttError::EventChannelClosed);
                }
            }

            wait_before_reconnect(retry_count, &mut shutdown).await;
        }
    }

    async fn connect(&self) -> Result<DeepgramSocket, SttError> {
        let url = build_deepgram_url(&self.config);
        let mut request = url
            .as_str()
            .into_client_request()
            .map_err(|error| SttError::Task(format!("could not build request: {error}")))?;
        let mut authorization =
            HeaderValue::from_str(&format!("Token {}", self.config.api_key.expose_secret()))
                .map_err(|error| SttError::AuthorizationHeader(error.to_string()))?;
        authorization.set_sensitive(true);
        request.headers_mut().insert(AUTHORIZATION, authorization);

        timeout(CONNECT_TIMEOUT, connect_async(request))
            .await
            .map_err(|_| {
                SttError::Task(format!(
                    "WebSocket handshake timed out after {} seconds",
                    CONNECT_TIMEOUT.as_secs()
                ))
            })?
            .map(|(socket, _response)| socket)
            .map_err(|error| SttError::Task(format!("WebSocket handshake failed: {error}")))
    }
}

impl SpeechToTextProvider for DeepgramSttProvider {
    fn run(
        self,
        audio: AudioFrameReceiver,
        events: mpsc::UnboundedSender<DeepgramEvent>,
        shutdown: watch::Receiver<bool>,
    ) -> impl Future<Output = Result<(), SttError>> + Send {
        self.run_loop(audio, events, shutdown)
    }
}

pub fn build_deepgram_url(config: &DeepgramConfig) -> Url {
    let mut url = config.ws_url.clone();
    {
        let mut query = url.query_pairs_mut();
        query
            .append_pair("model", &config.model)
            .append_pair("language", &config.language)
            .append_pair("encoding", "linear16")
            .append_pair("sample_rate", &config.sample_rate.to_string())
            .append_pair("channels", &config.channels.to_string())
            .append_pair("interim_results", &config.interim_results.to_string())
            .append_pair("punctuate", &config.punctuate.to_string())
            .append_pair("smart_format", &config.smart_format.to_string())
            .append_pair("vad_events", &config.vad_events.to_string())
            .append_pair("endpointing", &config.endpointing_ms.to_string())
            .append_pair("utterance_end_ms", &config.utterance_end_ms.to_string());

        for keyterm in &config.keyterms {
            query.append_pair("keyterm", keyterm);
        }
    }
    url
}

struct ConnectionResult {
    audio: AudioFrameReceiver,
    retry_frame: Option<AudioFrame>,
    outcome: ConnectionOutcome,
}

enum ConnectionOutcome {
    Shutdown,
    Disconnected(String),
    AudioChannelClosed,
    EventChannelClosed,
}

enum WriterCommand {
    KeepAlive,
    CloseStream,
}

struct SendTaskResult {
    audio: AudioFrameReceiver,
    retry_frame: Option<AudioFrame>,
    outcome: SendOutcome,
}

enum SendOutcome {
    Shutdown,
    Cancelled,
    AudioChannelClosed,
    ConnectionError(String),
}

enum ReadOutcome {
    Cancelled,
    EventChannelClosed,
    ConnectionClosed(String),
}

enum FirstTask {
    Send(Result<SendTaskResult, JoinError>),
    Read(Result<ReadOutcome, JoinError>),
    KeepAlive(Result<Result<(), String>, JoinError>),
    Shutdown(Result<Result<(), String>, JoinError>),
}

async fn run_connection(
    socket: DeepgramSocket,
    audio: AudioFrameReceiver,
    retry_frame: Option<AudioFrame>,
    events: mpsc::UnboundedSender<DeepgramEvent>,
    shutdown: watch::Receiver<bool>,
) -> Result<ConnectionResult, SttError> {
    let (writer, reader) = socket.split();
    let (command_sender, command_receiver) = mpsc::channel(4);
    let (cancel_sender, cancel_receiver) = watch::channel(false);

    let mut send_task = tokio::spawn(send_audio(
        writer,
        audio,
        retry_frame,
        command_receiver,
        cancel_receiver.clone(),
    ));
    let mut read_task = tokio::spawn(read_events(reader, events, cancel_receiver.clone()));
    let mut keepalive_task = tokio::spawn(run_keepalive(
        command_sender.clone(),
        cancel_receiver.clone(),
    ));
    let mut shutdown_task = tokio::spawn(run_shutdown(
        command_sender,
        shutdown.clone(),
        cancel_receiver,
    ));

    let first = tokio::select! {
        result = &mut send_task => FirstTask::Send(result),
        result = &mut read_task => FirstTask::Read(result),
        result = &mut keepalive_task => FirstTask::KeepAlive(result),
        result = &mut shutdown_task => FirstTask::Shutdown(result),
    };
    let keepalive_completed = matches!(&first, FirstTask::KeepAlive(_));
    let shutdown_completed = matches!(&first, FirstTask::Shutdown(_));

    if cancel_sender.send(true).is_err() {
        debug!(
            module = "stt",
            event = "connection_tasks_stopped",
            "Deepgram connection tasks already stopped"
        );
    }

    let (send_result, read_result, auxiliary_failure) = match first {
        FirstTask::Send(send_result) => (send_result, read_task.await, None),
        FirstTask::Read(read_result) => (send_task.await, read_result, None),
        FirstTask::KeepAlive(result) => (
            send_task.await,
            read_task.await,
            task_failure("keepalive", result),
        ),
        FirstTask::Shutdown(result) => (
            send_task.await,
            read_task.await,
            task_failure("shutdown", result),
        ),
    };
    let send_result =
        send_result.map_err(|error| SttError::Task(format!("audio sender task: {error}")))?;
    let read_result =
        read_result.map_err(|error| SttError::Task(format!("event reader task: {error}")))?;

    if !keepalive_completed {
        stop_task(keepalive_task).await;
    }
    if !shutdown_completed {
        stop_task(shutdown_task).await;
    }

    let outcome = if shutdown_requested(&shutdown) {
        ConnectionOutcome::Shutdown
    } else if let Some(error) = auxiliary_failure {
        ConnectionOutcome::Disconnected(error)
    } else {
        connection_outcome(&send_result.outcome, read_result)
    };

    Ok(ConnectionResult {
        audio: send_result.audio,
        retry_frame: send_result.retry_frame,
        outcome,
    })
}

async fn send_audio(
    mut writer: DeepgramWriter,
    mut audio: AudioFrameReceiver,
    retry_frame: Option<AudioFrame>,
    mut commands: mpsc::Receiver<WriterCommand>,
    mut cancel: watch::Receiver<bool>,
) -> SendTaskResult {
    if let Some(frame) = retry_frame
        && let Err(outcome) = send_frame(&mut writer, &frame, &mut cancel).await
    {
        return SendTaskResult {
            audio,
            retry_frame: Some(frame),
            outcome,
        };
    }

    loop {
        tokio::select! {
            _ = wait_for_shutdown(&mut cancel) => {
                return SendTaskResult {
                    audio,
                    retry_frame: None,
                    outcome: SendOutcome::Cancelled,
                };
            }
            command = commands.recv() => {
                let Some(command) = command else {
                    return SendTaskResult {
                        audio,
                        retry_frame: None,
                        outcome: SendOutcome::ConnectionError(
                            "WebSocket command channel closed".to_owned()
                        ),
                    };
                };
                match command {
                    WriterCommand::KeepAlive => {
                        if let Err(outcome) =
                            send_text(&mut writer, KEEPALIVE_MESSAGE, &mut cancel).await
                        {
                            return SendTaskResult {
                                audio,
                                retry_frame: None,
                                outcome,
                            };
                        }
                    }
                    WriterCommand::CloseStream => {
                        let outcome =
                            match send_text(&mut writer, CLOSE_STREAM_MESSAGE, &mut cancel).await {
                                Ok(()) => SendOutcome::Shutdown,
                                Err(outcome) => outcome,
                            };
                        return SendTaskResult {
                            audio,
                            retry_frame: None,
                            outcome,
                        };
                    }
                }
            }
            frame = audio.recv() => {
                let Some(frame) = frame else {
                    return SendTaskResult {
                        audio,
                        retry_frame: None,
                        outcome: SendOutcome::AudioChannelClosed,
                    };
                };
                if let Err(outcome) = send_frame(&mut writer, &frame, &mut cancel).await {
                    return SendTaskResult {
                        audio,
                        retry_frame: Some(frame),
                        outcome,
                    };
                }
            }
        }
    }
}

async fn send_frame(
    writer: &mut DeepgramWriter,
    frame: &AudioFrame,
    cancel: &mut watch::Receiver<bool>,
) -> Result<(), SendOutcome> {
    let message = Message::Binary(frame.pcm.clone().into());
    tokio::select! {
        _ = wait_for_shutdown(cancel) => Err(SendOutcome::Cancelled),
        result = writer.send(message) => result
            .map_err(|error| SendOutcome::ConnectionError(error.to_string())),
    }
}

async fn send_text(
    writer: &mut DeepgramWriter,
    text: &'static str,
    cancel: &mut watch::Receiver<bool>,
) -> Result<(), SendOutcome> {
    tokio::select! {
        _ = wait_for_shutdown(cancel) => Err(SendOutcome::Cancelled),
        result = writer.send(Message::Text(text.into())) => result
            .map_err(|error| SendOutcome::ConnectionError(error.to_string())),
    }
}

async fn read_events(
    mut reader: DeepgramReader,
    events: mpsc::UnboundedSender<DeepgramEvent>,
    mut cancel: watch::Receiver<bool>,
) -> ReadOutcome {
    loop {
        let message = tokio::select! {
            _ = wait_for_shutdown(&mut cancel) => return ReadOutcome::Cancelled,
            message = reader.next() => message,
        };

        match message {
            Some(Ok(Message::Text(text))) => match parse_server_message(text.as_ref()) {
                Ok(Some(event)) => {
                    if events.send(event).is_err() {
                        return ReadOutcome::EventChannelClosed;
                    }
                }
                Ok(None) => {}
                Err(error) => {
                    return ReadOutcome::ConnectionClosed(format!("protocol error: {error}"));
                }
            },
            Some(Ok(Message::Close(frame))) => {
                return ReadOutcome::ConnectionClosed(
                    frame
                        .map(|frame| format!("{} {}", frame.code, frame.reason))
                        .unwrap_or_else(|| "server closed WebSocket".to_owned()),
                );
            }
            Some(Ok(Message::Binary(_)))
            | Some(Ok(Message::Ping(_)))
            | Some(Ok(Message::Pong(_)))
            | Some(Ok(Message::Frame(_))) => {}
            Some(Err(error)) => {
                return ReadOutcome::ConnectionClosed(error.to_string());
            }
            None => {
                return ReadOutcome::ConnectionClosed("WebSocket stream ended".to_owned());
            }
        }
    }
}

async fn run_keepalive(
    commands: mpsc::Sender<WriterCommand>,
    mut cancel: watch::Receiver<bool>,
) -> Result<(), String> {
    let mut timer = interval(KEEPALIVE_INTERVAL);
    timer.set_missed_tick_behavior(MissedTickBehavior::Delay);
    timer.tick().await;

    loop {
        tokio::select! {
            _ = wait_for_shutdown(&mut cancel) => return Ok(()),
            _ = timer.tick() => {
                commands
                    .send(WriterCommand::KeepAlive)
                    .await
                    .map_err(|_| "WebSocket command channel closed".to_owned())?;
            }
        }
    }
}

async fn run_shutdown(
    commands: mpsc::Sender<WriterCommand>,
    mut shutdown: watch::Receiver<bool>,
    mut cancel: watch::Receiver<bool>,
) -> Result<(), String> {
    tokio::select! {
        _ = wait_for_shutdown(&mut cancel) => return Ok(()),
        _ = wait_for_shutdown(&mut shutdown) => {}
    }

    commands
        .send(WriterCommand::CloseStream)
        .await
        .map_err(|_| "WebSocket command channel closed".to_owned())?;
    wait_for_shutdown(&mut cancel).await;
    Ok(())
}

fn connection_outcome(send: &SendOutcome, read: ReadOutcome) -> ConnectionOutcome {
    match send {
        SendOutcome::Shutdown => ConnectionOutcome::Shutdown,
        SendOutcome::AudioChannelClosed => ConnectionOutcome::AudioChannelClosed,
        SendOutcome::ConnectionError(error) => ConnectionOutcome::Disconnected(error.clone()),
        SendOutcome::Cancelled => match read {
            ReadOutcome::EventChannelClosed => ConnectionOutcome::EventChannelClosed,
            ReadOutcome::ConnectionClosed(error) => ConnectionOutcome::Disconnected(error),
            ReadOutcome::Cancelled => {
                ConnectionOutcome::Disconnected("connection tasks stopped".to_owned())
            }
        },
    }
}

fn task_failure(name: &str, result: Result<Result<(), String>, JoinError>) -> Option<String> {
    match result {
        Ok(Ok(())) => Some(format!("{name} task stopped unexpectedly")),
        Ok(Err(error)) => Some(format!("{name} task failed: {error}")),
        Err(error) => Some(format!("{name} task failed: {error}")),
    }
}

async fn stop_task(mut task: JoinHandle<Result<(), String>>) {
    if !task.is_finished() {
        task.abort();
    }
    if let Err(error) = (&mut task).await
        && !error.is_cancelled()
    {
        warn!(
            module = "stt",
            event = "auxiliary_task_failed",
            error = %error,
            "Deepgram auxiliary task failed"
        );
    }
}

fn emit_error(
    events: &mpsc::UnboundedSender<DeepgramEvent>,
    message: String,
) -> Result<(), SttError> {
    warn!(
        module = "stt",
        event = "provider_error",
        provider = "deepgram",
        error = %message,
        "Deepgram provider error"
    );
    events
        .send(DeepgramEvent::Error(message))
        .map_err(|_| SttError::EventChannelClosed)
}

async fn wait_before_reconnect(retry_count: u32, shutdown: &mut watch::Receiver<bool>) {
    let delay = reconnect_delay(retry_count);
    info!(
        module = "stt",
        event = "reconnect_wait",
        provider = "deepgram",
        retry_count,
        delay_ms = delay.as_millis(),
        "waiting before Deepgram reconnect"
    );

    tokio::select! {
        _ = wait_for_shutdown(shutdown) => {}
        _ = sleep(delay) => {}
    }
}

fn reconnect_delay(retry_count: u32) -> Duration {
    let exponent = retry_count.saturating_sub(1).min(5);
    INITIAL_RECONNECT_DELAY
        .saturating_mul(1_u32 << exponent)
        .min(MAX_RECONNECT_DELAY)
}

fn shutdown_requested(shutdown: &watch::Receiver<bool>) -> bool {
    *shutdown.borrow()
}

async fn wait_for_shutdown(shutdown: &mut watch::Receiver<bool>) {
    while !shutdown_requested(shutdown) {
        if shutdown.changed().await.is_err() {
            return;
        }
    }
}

#[cfg(test)]
mod tests {
    use std::sync::{Arc, Mutex};

    use futures_util::{SinkExt, StreamExt};
    use tokio::{
        net::TcpListener,
        sync::{oneshot, watch},
        time::timeout,
    };
    use tokio_tungstenite::{
        accept_hdr_async,
        tungstenite::{
            Message,
            handshake::server::{Callback, ErrorResponse, Request, Response},
        },
    };

    use crate::{
        audio::audio_frame_channel,
        config::SecretString,
        events::{AudioFrame, DeepgramEvent},
        stt::SpeechToTextProvider,
    };

    use super::*;

    struct CaptureHandshake {
        request_paths: Arc<Mutex<Vec<String>>>,
    }

    impl Callback for CaptureHandshake {
        fn on_request(
            self,
            request: &Request,
            response: Response,
        ) -> Result<Response, ErrorResponse> {
            assert_eq!(
                request
                    .headers()
                    .get(AUTHORIZATION)
                    .and_then(|value| value.to_str().ok()),
                Some("Token secret")
            );
            self.request_paths
                .lock()
                .expect("paths mutex must not be poisoned")
                .push(request.uri().to_string());
            Ok(response)
        }
    }

    fn config() -> DeepgramConfig {
        DeepgramConfig {
            api_key: SecretString::new("secret".to_owned()),
            ws_url: Url::parse("wss://api.deepgram.com/v1/listen").expect("base URL must be valid"),
            model: "nova-3".to_owned(),
            language: "ru".to_owned(),
            sample_rate: 16_000,
            channels: 1,
            interim_results: true,
            punctuate: true,
            smart_format: true,
            vad_events: true,
            endpointing_ms: 500,
            utterance_end_ms: 1_200,
            keyterms: vec!["Java".to_owned(), "Spring Boot".to_owned()],
        }
    }

    #[test]
    fn builds_streaming_url_with_repeated_keyterms() {
        let url = build_deepgram_url(&config());
        let pairs = url.query_pairs().collect::<Vec<_>>();

        assert!(pairs.contains(&("encoding".into(), "linear16".into())));
        assert!(pairs.contains(&("sample_rate".into(), "16000".into())));
        assert!(pairs.contains(&("interim_results".into(), "true".into())));
        assert!(pairs.contains(&("endpointing".into(), "500".into())));
        assert_eq!(
            pairs
                .iter()
                .filter(|(name, _)| name == "keyterm")
                .map(|(_, value)| value.as_ref())
                .collect::<Vec<_>>(),
            vec!["Java", "Spring Boot"]
        );
        assert!(!url.as_str().contains("secret"));
    }

    #[test]
    fn installs_crypto_provider_before_building_tls_config() {
        install_tls_crypto_provider().expect("ring provider must be installed");
        let _builder = rustls::ClientConfig::builder();
    }

    #[test]
    fn reconnect_delay_is_exponential_and_capped() {
        assert_eq!(reconnect_delay(1), Duration::from_secs(1));
        assert_eq!(reconnect_delay(2), Duration::from_secs(2));
        assert_eq!(reconnect_delay(3), Duration::from_secs(4));
        assert_eq!(reconnect_delay(6), Duration::from_secs(30));
        assert_eq!(reconnect_delay(20), Duration::from_secs(30));
    }

    #[tokio::test]
    async fn preserves_queued_audio_across_reconnect() {
        let listener = TcpListener::bind("127.0.0.1:0")
            .await
            .expect("mock server must bind");
        let address = listener.local_addr().expect("address must be available");
        let request_paths = Arc::new(Mutex::new(Vec::new()));
        let captured_paths = Arc::clone(&request_paths);
        let (first_frame_sender, first_frame_receiver) = oneshot::channel();

        let server = tokio::spawn(async move {
            let mut first_frame_sender = Some(first_frame_sender);
            for connection_index in 0..2 {
                let (stream, _) = listener.accept().await.expect("client must connect");
                let captured_paths = Arc::clone(&captured_paths);
                let mut websocket = accept_hdr_async(
                    stream,
                    CaptureHandshake {
                        request_paths: captured_paths,
                    },
                )
                .await
                .expect("WebSocket handshake must succeed");

                if connection_index == 0 {
                    let frame = next_binary(&mut websocket).await;
                    assert_eq!(frame, vec![1]);
                    first_frame_sender
                        .take()
                        .expect("first-frame signal must only be sent once")
                        .send(())
                        .expect("test must wait for first frame");
                    websocket
                        .send(Message::Close(None))
                        .await
                        .expect("server close must be sent");
                    continue;
                }

                assert_eq!(next_binary(&mut websocket).await, vec![2]);
                assert_eq!(next_binary(&mut websocket).await, vec![3]);
                websocket
                    .send(Message::Text(
                        r#"{"type":"Results","is_final":true,"speech_final":true,"channel":{"alternatives":[{"transcript":"проверка связи"}]}}"#
                            .into(),
                    ))
                    .await
                    .expect("transcript must be sent");

                loop {
                    match websocket.next().await {
                        Some(Ok(Message::Text(text))) if text.as_str() == CLOSE_STREAM_MESSAGE => {
                            break;
                        }
                        Some(Ok(_)) => {}
                        Some(Err(error)) => panic!("mock WebSocket failed: {error}"),
                        None => panic!("client disconnected before CloseStream"),
                    }
                }
            }
        });

        let mut config = config();
        config.ws_url =
            Url::parse(&format!("ws://{address}/v1/listen")).expect("mock URL must be valid");
        let provider = DeepgramSttProvider::new(config);
        let (audio_sender, audio_receiver) = audio_frame_channel(0);
        let (event_sender, mut event_receiver) = mpsc::unbounded_channel();
        let (shutdown_sender, shutdown_receiver) = watch::channel(false);
        let provider_task =
            tokio::spawn(provider.run(audio_receiver, event_sender, shutdown_receiver));

        audio_sender
            .send(audio_frame(0, 1))
            .await
            .expect("first frame must queue");
        timeout(Duration::from_secs(2), first_frame_receiver)
            .await
            .expect("first connection must receive audio")
            .expect("first-frame signal must be sent");
        audio_sender
            .send(audio_frame(1, 2))
            .await
            .expect("second frame must queue");
        audio_sender
            .send(audio_frame(2, 3))
            .await
            .expect("third frame must queue");

        let transcript = timeout(Duration::from_secs(4), async {
            loop {
                if let Some(DeepgramEvent::Transcript { text, .. }) = event_receiver.recv().await {
                    break text;
                }
            }
        })
        .await
        .expect("transcript must arrive after reconnect");
        assert_eq!(transcript, "проверка связи");

        shutdown_sender
            .send(true)
            .expect("provider must still receive shutdown");
        timeout(Duration::from_secs(2), provider_task)
            .await
            .expect("provider must stop before timeout")
            .expect("provider task must not panic")
            .expect("provider must stop cleanly");
        timeout(Duration::from_secs(2), server)
            .await
            .expect("mock server must stop before timeout")
            .expect("mock server must not panic");

        let paths = request_paths
            .lock()
            .expect("paths mutex must not be poisoned");
        assert_eq!(paths.len(), 2);
        assert!(paths.iter().all(|path| path.contains("model=nova-3")));
        assert!(paths.iter().all(|path| path.contains("keyterm=Java")));
    }

    fn audio_frame(sequence: u64, byte: u8) -> AudioFrame {
        AudioFrame {
            sequence,
            pcm: vec![byte],
        }
    }

    async fn next_binary<S>(websocket: &mut WebSocketStream<S>) -> Vec<u8>
    where
        S: tokio::io::AsyncRead + tokio::io::AsyncWrite + Unpin,
    {
        loop {
            match websocket.next().await {
                Some(Ok(Message::Binary(bytes))) => return bytes.to_vec(),
                Some(Ok(_)) => {}
                Some(Err(error)) => panic!("mock WebSocket failed: {error}"),
                None => panic!("client disconnected before sending audio"),
            }
        }
    }
}
