use std::{
    future::Future,
    path::PathBuf,
    process::{ExitStatus, Stdio},
    time::Duration,
};

use thiserror::Error;
use tokio::{
    io::{AsyncBufReadExt, AsyncReadExt, BufReader},
    process::{Child, ChildStderr, Command},
    sync::watch,
    task::JoinHandle,
    time::{sleep, timeout},
};
use tracing::{debug, info, warn};

use crate::{
    audio::source::{AudioFrameSender, AudioSource},
    config::AudioConfig,
    events::AudioFrame,
};

const BYTES_PER_SAMPLE: u64 = 2;
const RESTART_DELAY: Duration = Duration::from_secs(2);
const CHILD_STOP_TIMEOUT: Duration = Duration::from_secs(3);
const QUEUE_WARNING_STEP: usize = 100;

#[derive(Debug, Error)]
pub enum AudioError {
    #[error("invalid PCM chunk size: {0}")]
    InvalidChunkSize(String),

    #[error("failed to start ffmpeg executable {executable:?}: {source}")]
    Spawn {
        executable: PathBuf,
        #[source]
        source: std::io::Error,
    },

    #[error("ffmpeg {stream} pipe was not available")]
    MissingPipe { stream: &'static str },

    #[error("failed to read ffmpeg audio output: {0}")]
    Read(#[source] std::io::Error),

    #[error("failed to wait for ffmpeg: {0}")]
    Wait(#[source] std::io::Error),

    #[error("ffmpeg did not stop within {} seconds", CHILD_STOP_TIMEOUT.as_secs())]
    StopTimeout,

    #[error("audio frame channel closed")]
    ChannelClosed,
}

pub struct FfmpegAudioSource {
    config: AudioConfig,
    sample_rate: u32,
    channels: u16,
    next_sequence: u64,
}

impl FfmpegAudioSource {
    pub fn new(config: AudioConfig, sample_rate: u32, channels: u16) -> Self {
        Self {
            config,
            sample_rate,
            channels,
            next_sequence: 0,
        }
    }

    async fn run_loop(
        mut self,
        output: AudioFrameSender,
        mut shutdown: watch::Receiver<bool>,
    ) -> Result<(), AudioError> {
        let chunk_size = pcm_chunk_size(self.sample_rate, self.channels, self.config.chunk_ms)?;
        let mut retry_count = 0_u64;

        loop {
            if shutdown_requested(&shutdown) {
                return Ok(());
            }

            match self.capture_once(chunk_size, &output, &mut shutdown).await {
                Ok(CaptureOutcome::Shutdown) => return Ok(()),
                Ok(CaptureOutcome::Exited(status)) => {
                    retry_count += 1;
                    warn!(
                        module = "audio",
                        event = "ffmpeg_exited",
                        ?status,
                        retry_count,
                        "ffmpeg exited unexpectedly; capture will restart"
                    );
                }
                Err(AudioError::ChannelClosed) => return Err(AudioError::ChannelClosed),
                Err(error) => {
                    retry_count += 1;
                    warn!(
                        module = "audio",
                        event = "capture_failed",
                        retry_count,
                        error = %error,
                        "audio capture failed; ffmpeg will restart"
                    );
                }
            }

            tokio::select! {
                _ = wait_for_shutdown(&mut shutdown) => return Ok(()),
                _ = sleep(RESTART_DELAY) => {}
            }
        }
    }

    async fn capture_once(
        &mut self,
        chunk_size: usize,
        output: &AudioFrameSender,
        shutdown: &mut watch::Receiver<bool>,
    ) -> Result<CaptureOutcome, AudioError> {
        let mut child = self.spawn()?;
        let stdout = child
            .stdout
            .take()
            .ok_or(AudioError::MissingPipe { stream: "stdout" })?;
        let stderr = child
            .stderr
            .take()
            .ok_or(AudioError::MissingPipe { stream: "stderr" })?;
        let stderr_task = tokio::spawn(log_stderr(stderr));

        info!(
            module = "audio",
            event = "capture_started",
            source = %self.config.source,
            sample_rate = self.sample_rate,
            channels = self.channels,
            chunk_ms = self.config.chunk_ms,
            chunk_bytes = chunk_size,
            "ffmpeg audio capture started"
        );

        let read_result = self.read_frames(stdout, chunk_size, output, shutdown).await;

        match read_result {
            Ok(ReadOutcome::Shutdown) => {
                stop_child(&mut child).await?;
                finish_stderr_task(stderr_task).await;
                Ok(CaptureOutcome::Shutdown)
            }
            Ok(ReadOutcome::EndOfStream { partial_bytes }) => {
                if partial_bytes > 0 {
                    debug!(
                        module = "audio",
                        event = "partial_frame_discarded",
                        partial_bytes,
                        "discarded incomplete PCM frame after ffmpeg exit"
                    );
                }
                let status = tokio::select! {
                    _ = wait_for_shutdown(shutdown) => {
                        stop_child(&mut child).await?;
                        finish_stderr_task(stderr_task).await;
                        return Ok(CaptureOutcome::Shutdown);
                    }
                    result = child.wait() => result.map_err(AudioError::Wait)?,
                };
                finish_stderr_task(stderr_task).await;
                Ok(CaptureOutcome::Exited(status))
            }
            Err(error) => {
                stop_child(&mut child).await?;
                finish_stderr_task(stderr_task).await;
                Err(error)
            }
        }
    }

    fn spawn(&self) -> Result<Child, AudioError> {
        let mut command = Command::new(&self.config.ffmpeg_bin);
        command
            .arg("-hide_banner")
            .arg("-loglevel")
            .arg("warning")
            .arg("-f")
            .arg(&self.config.input_format)
            .arg("-i")
            .arg(&self.config.source)
            .arg("-ac")
            .arg(self.channels.to_string())
            .arg("-ar")
            .arg(self.sample_rate.to_string())
            .arg("-f")
            .arg("s16le")
            .arg("pipe:1")
            .stdin(Stdio::null())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .kill_on_drop(true);

        command.spawn().map_err(|source| AudioError::Spawn {
            executable: self.config.ffmpeg_bin.clone(),
            source,
        })
    }

    async fn read_frames(
        &mut self,
        mut stdout: tokio::process::ChildStdout,
        chunk_size: usize,
        output: &AudioFrameSender,
        shutdown: &mut watch::Receiver<bool>,
    ) -> Result<ReadOutcome, AudioError> {
        let mut buffer = vec![0_u8; chunk_size];
        let mut filled = 0;
        let mut highest_warned_queue_len = 0_usize;

        loop {
            let bytes_read = tokio::select! {
                _ = wait_for_shutdown(shutdown) => return Ok(ReadOutcome::Shutdown),
                result = stdout.read(&mut buffer[filled..]) => result.map_err(AudioError::Read)?,
            };

            if bytes_read == 0 {
                return Ok(ReadOutcome::EndOfStream {
                    partial_bytes: filled,
                });
            }
            filled += bytes_read;

            if filled != chunk_size {
                continue;
            }

            let frame = AudioFrame {
                sequence: self.next_sequence,
                pcm: std::mem::replace(&mut buffer, vec![0_u8; chunk_size]),
            };
            self.next_sequence = self.next_sequence.wrapping_add(1);
            filled = 0;

            tokio::select! {
                _ = wait_for_shutdown(shutdown) => return Ok(ReadOutcome::Shutdown),
                result = output.send(frame) => result?,
            }

            let queue_len = output.len();
            let warning_due = queue_len
                >= highest_warned_queue_len.saturating_add(QUEUE_WARNING_STEP)
                || (output.is_full() && highest_warned_queue_len == 0);
            if warning_due {
                highest_warned_queue_len = queue_len;
                warn!(
                    module = "audio",
                    event = "queue_growing",
                    queue_len,
                    "audio queue is growing"
                );
            }
        }
    }
}

impl AudioSource for FfmpegAudioSource {
    fn run(
        self,
        output: AudioFrameSender,
        shutdown: watch::Receiver<bool>,
    ) -> impl Future<Output = Result<(), AudioError>> + Send {
        self.run_loop(output, shutdown)
    }
}

enum CaptureOutcome {
    Shutdown,
    Exited(ExitStatus),
}

enum ReadOutcome {
    Shutdown,
    EndOfStream { partial_bytes: usize },
}

pub fn pcm_chunk_size(sample_rate: u32, channels: u16, chunk_ms: u64) -> Result<usize, AudioError> {
    let bytes = u64::from(sample_rate)
        .checked_mul(u64::from(channels))
        .and_then(|value| value.checked_mul(BYTES_PER_SAMPLE))
        .and_then(|value| value.checked_mul(chunk_ms))
        .ok_or_else(|| AudioError::InvalidChunkSize("calculation overflowed".to_owned()))?
        / 1_000;

    if bytes == 0 {
        return Err(AudioError::InvalidChunkSize(
            "calculation produced zero bytes".to_owned(),
        ));
    }

    usize::try_from(bytes)
        .map_err(|_| AudioError::InvalidChunkSize("does not fit in usize".to_owned()))
}

async fn log_stderr(stderr: ChildStderr) {
    let mut lines = BufReader::new(stderr).lines();

    loop {
        match lines.next_line().await {
            Ok(Some(line)) => warn!(
                module = "audio",
                event = "ffmpeg_stderr",
                message = %line,
                "ffmpeg diagnostic"
            ),
            Ok(None) => return,
            Err(error) => {
                warn!(
                    module = "audio",
                    event = "stderr_read_failed",
                    error = %error,
                    "failed to read ffmpeg stderr"
                );
                return;
            }
        }
    }
}

async fn finish_stderr_task(task: JoinHandle<()>) {
    if let Err(error) = task.await {
        warn!(
            module = "audio",
            event = "stderr_task_failed",
            error = %error,
            "ffmpeg stderr task failed"
        );
    }
}

async fn stop_child(child: &mut Child) -> Result<(), AudioError> {
    if child.try_wait().map_err(AudioError::Wait)?.is_none() {
        child.start_kill().map_err(AudioError::Wait)?;
    }

    timeout(CHILD_STOP_TIMEOUT, child.wait())
        .await
        .map_err(|_| AudioError::StopTimeout)?
        .map_err(AudioError::Wait)?;
    Ok(())
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
    use super::*;

    #[test]
    fn calculates_default_pcm_chunk_size() {
        let size = pcm_chunk_size(16_000, 1, 100).expect("default values must be valid");

        assert_eq!(size, 3_200);
    }

    #[test]
    fn accounts_for_multiple_channels() {
        let size = pcm_chunk_size(48_000, 2, 20).expect("audio shape must be valid");

        assert_eq!(size, 3_840);
    }

    #[test]
    fn rejects_zero_sized_chunk() {
        let error = pcm_chunk_size(0, 1, 100).expect_err("zero sample rate must fail");

        assert!(matches!(error, AudioError::InvalidChunkSize(_)));
    }
}
