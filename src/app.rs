use std::time::Duration;

use tokio::{
    sync::watch,
    task::{JoinError, JoinHandle},
    time::timeout,
};
use tracing::{info, warn};

use crate::{
    audio::{AudioError, AudioFrameReceiver, AudioSource, FfmpegAudioSource, audio_frame_channel},
    config::Config,
    control::TerminalEchoGuard,
    error::AppError,
};

const SHUTDOWN_TIMEOUT: Duration = Duration::from_secs(5);

pub async fn run(config: Config) -> Result<(), AppError> {
    let _terminal_echo_guard = TerminalEchoGuard::hide_control_characters();

    info!(
        module = "app",
        event = "started",
        audio_source = %config.audio.source,
        stt_model = %config.deepgram.model,
        llm_model = %config.llm.model,
        "mague-rc started; press Ctrl+C to stop"
    );

    let (frame_sender, frame_receiver) = audio_frame_channel(config.audio.queue_max);
    let (shutdown_sender, shutdown_receiver) = watch::channel(false);
    let source = FfmpegAudioSource::new(
        config.audio.clone(),
        config.deepgram.sample_rate,
        config.deepgram.channels,
    );
    let mut audio_task = tokio::spawn(source.run(frame_sender, shutdown_receiver));
    let frame_task = tokio::spawn(count_audio_frames(frame_receiver));

    tokio::select! {
        signal = tokio::signal::ctrl_c() => {
            signal.map_err(AppError::Shutdown)?;
            info!(
                module = "app",
                event = "shutdown_requested",
                "shutdown signal received"
            );
        }
        result = &mut audio_task => {
            finish_frame_task(frame_task).await?;
            flatten_audio_task(result)?;
            return Err(AppError::AudioTaskStopped);
        }
    }

    if shutdown_sender.send(true).is_err() {
        warn!(
            module = "app",
            event = "shutdown_channel_closed",
            "audio worker closed its shutdown channel"
        );
    }

    match timeout(SHUTDOWN_TIMEOUT, &mut audio_task).await {
        Ok(result) => flatten_audio_task(result)?,
        Err(_) => {
            audio_task.abort();
            if let Err(error) = audio_task.await
                && !error.is_cancelled()
            {
                warn!(
                    module = "app",
                    event = "audio_abort_failed",
                    error = %error,
                    "audio worker failed while being aborted"
                );
            }
            finish_frame_task(frame_task).await?;
            return Err(AppError::AudioShutdownTimeout);
        }
    }

    let frame_count = finish_frame_task(frame_task).await?;
    info!(
        module = "app",
        event = "shutdown_completed",
        frame_count,
        "mague-rc stopped cleanly"
    );
    Ok(())
}

async fn count_audio_frames(mut receiver: AudioFrameReceiver) -> u64 {
    let mut count = 0_u64;

    while let Some(frame) = receiver.recv().await {
        count += 1;
        if count.is_multiple_of(100) {
            info!(
                module = "audio",
                event = "frames_captured",
                frame_count = count,
                sequence = frame.sequence,
                frame_bytes = frame.pcm.len(),
                "PCM audio frames captured"
            );
        }
    }

    count
}

async fn finish_frame_task(task: JoinHandle<u64>) -> Result<u64, AppError> {
    task.await
        .map_err(|error| AppError::AudioTask(format!("frame counter: {error}")))
}

fn flatten_audio_task(result: Result<Result<(), AudioError>, JoinError>) -> Result<(), AppError> {
    result
        .map_err(|error| AppError::AudioTask(format!("capture worker: {error}")))?
        .map_err(AppError::Audio)
}
