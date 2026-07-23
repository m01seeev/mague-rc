mod ffmpeg;
mod source;

pub use ffmpeg::{AudioError, FfmpegAudioSource};
pub use source::{AudioFrameReceiver, AudioFrameSender, AudioSource, audio_frame_channel};
