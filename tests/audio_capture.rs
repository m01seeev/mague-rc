#![cfg(unix)]

use std::{
    fs,
    os::unix::fs::PermissionsExt,
    time::{Duration, SystemTime},
};

use mague_rc::{
    audio::{AudioSource, FfmpegAudioSource, audio_frame_channel},
    config::AudioConfig,
};
use tokio::{sync::watch, time::timeout};

#[tokio::test]
async fn streams_pcm_and_stops_the_child_process() {
    let fixture_dir = std::env::temp_dir().join(format!(
        "mague-rc-audio-test-{}-{}",
        std::process::id(),
        SystemTime::now()
            .duration_since(SystemTime::UNIX_EPOCH)
            .expect("system clock must follow Unix epoch")
            .as_nanos()
    ));
    fs::create_dir(&fixture_dir).expect("fixture directory must be created");
    let executable = fixture_dir.join("fake-ffmpeg");
    fs::write(
        &executable,
        "#!/bin/sh\nexec /bin/dd if=/dev/zero bs=3200 2>/dev/null\n",
    )
    .expect("fixture executable must be written");
    let mut permissions = fs::metadata(&executable)
        .expect("fixture metadata must be available")
        .permissions();
    permissions.set_mode(0o755);
    fs::set_permissions(&executable, permissions).expect("fixture must be executable");

    let config = AudioConfig {
        ffmpeg_bin: executable,
        input_format: "pulse".to_owned(),
        source: "fake-monitor".to_owned(),
        chunk_ms: 100,
        queue_max: 1,
    };
    let source = FfmpegAudioSource::new(config, 16_000, 1);
    let (sender, mut receiver) = audio_frame_channel(1);
    let (shutdown_sender, shutdown_receiver) = watch::channel(false);
    let worker = tokio::spawn(source.run(sender, shutdown_receiver));

    for expected_sequence in 0..3 {
        let frame = timeout(Duration::from_secs(1), receiver.recv())
            .await
            .expect("audio frame must arrive before timeout")
            .expect("audio channel must remain open");
        assert_eq!(frame.sequence, expected_sequence);
        assert_eq!(frame.pcm.len(), 3_200);
    }

    shutdown_sender
        .send(true)
        .expect("shutdown receiver must remain open");
    timeout(Duration::from_secs(2), worker)
        .await
        .expect("audio worker must stop before timeout")
        .expect("audio worker task must not panic")
        .expect("audio worker must stop cleanly");

    fs::remove_dir_all(fixture_dir).expect("fixture directory must be removed");
}
