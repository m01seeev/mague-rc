# mague-rc

`mague-rc` is a local, terminal-first AI assistant for technical interviews, designed primarily for tiling window managers on Linux. Traditional desktop environments already have comparable assistant and overlay options, while tiling setups need a lightweight tool that integrates cleanly without replacing the window-management workflow.

The initial target is Wayland with Hyprland. The current implementation captures system audio through `ffmpeg`, streams raw PCM to Deepgram, and assembles recognition results into fixed transcript windows in the terminal. Text LLM providers and the overlay are intentionally introduced in later stages.

## Requirements

- Rust stable
- `ffmpeg` available through `PATH`, or an executable path set in `FFMPEG_BIN`
- PipeWire with a PulseAudio-compatible server
- Network access to Deepgram

## Configuration

Create a local environment file and fill in the provider keys:

```bash
cp .env.example .env
```

`.env` is ignored by Git. `DEEPGRAM_API_KEY` is used by the current pipeline. `OPENROUTER_API_KEY` is reserved for the text LLM stage. `GROQ_API_KEY` is required while `ENABLE_OCR=true`; OCR can be disabled until its stage is implemented.

Environment variables override values loaded from `.env`. Use `RUST_LOG` to control tracing verbosity, for example `RUST_LOG=debug`.

## Run

```bash
cargo run
```

The process validates configuration, starts `ffmpeg`, and reads raw PCM from the configured PulseAudio-compatible source. PCM frames are streamed to Deepgram over an authenticated WebSocket and are not written to disk. Interim and final transcripts are printed as diagnostics. Every `TRANSCRIPT_WINDOW_SEC` seconds, new recognized text is emitted as a separate `TRANSCRIPT WINDOW`; growing interim hypotheses only contribute their unsent tail. Text shorter than `MIN_UTTERANCE_CHARS` waits for more input.

Both `ffmpeg` and Deepgram reconnect automatically. Audio frames remain queued in memory and preserve their order while Deepgram is reconnecting. Press Ctrl+C to close the WebSocket, stop `ffmpeg`, restore the terminal, and exit cleanly.

To list available PipeWire/PulseAudio sources:

```bash
pactl list short sources
```

Play speech through the selected system-audio source while the application is running. A working pipeline reports:

```text
Deepgram connected
interim transcript
FINAL transcript
TRANSCRIPT WINDOW
```

With the default 100 ms chunk size, audio capture also reports `PCM audio frames captured` roughly every 10 seconds. Stop the process while speech is still pending to verify that the last transcript window is flushed with `reason="shutdown"`.

## Checks

```bash
cargo fmt --check
cargo check
cargo clippy --all-targets --all-features -- -D warnings
cargo test
```

## Current scope

Implemented: typed configuration, redacted secrets, continuous PCM capture through `ffmpeg`, bounded/unbounded queues, authenticated Deepgram WebSocket streaming, keepalive, final/interim event parsing, ordered reconnect with retry, fixed transcript windows with interim deduplication and shutdown flush, structured diagnostics, and graceful Ctrl+C shutdown.

Not implemented: OpenRouter, RAG, OCR, and Wayland GUI. These are introduced only in their corresponding later stages.
