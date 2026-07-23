# mague-rc

`mague-rc` is a local, terminal-first AI assistant for technical interviews, designed primarily for tiling window managers on Linux. Traditional desktop environments already have comparable assistant and overlay options, while tiling setups need a lightweight tool that integrates cleanly without replacing the window-management workflow.

The initial target is Wayland with Hyprland. The current implementation provides typed environment configuration, structured logging, graceful shutdown, and continuous system-audio capture through `ffmpeg`. AI providers and the overlay are intentionally introduced in later stages.

## Requirements

- Rust stable
- `ffmpeg` available through `PATH`, or an executable path set in `FFMPEG_BIN`

## Configuration

Create a local environment file and fill in the provider keys:

```bash
cp .env.example .env
```

`.env` is ignored by Git. `DEEPGRAM_API_KEY` and `OPENROUTER_API_KEY` are always required. `GROQ_API_KEY` is required while `ENABLE_OCR=true`; OCR can be disabled during the current scaffold stage.

Environment variables override values loaded from `.env`. Use `RUST_LOG` to control tracing verbosity, for example `RUST_LOG=debug`.

## Run

```bash
cargo run
```

The process validates configuration, starts `ffmpeg`, and reads raw PCM from the configured PulseAudio-compatible source. It logs a counter every 100 audio frames and restarts `ffmpeg` after an unexpected exit. Press Ctrl+C to stop both the application and its child process cleanly.

To list available PipeWire/PulseAudio sources:

```bash
pactl list short sources
```

Play system audio while the application is running. With the default 100 ms chunk size, a successful capture reports `PCM audio frames captured` roughly every 10 seconds.

## Checks

```bash
cargo fmt --check
cargo check
cargo clippy --all-targets --all-features -- -D warnings
cargo test
```

## Current scope

Implemented: project scaffold, typed configuration, validation, redacted secrets, tracing, continuous PCM capture through `ffmpeg`, bounded/unbounded audio queues, automatic capture restart, frame diagnostics, and graceful Ctrl+C shutdown.

Not implemented: Deepgram, fixed transcript windows, OpenRouter, RAG, OCR, and Wayland GUI. These are introduced only in their corresponding later stages.
