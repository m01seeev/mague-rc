# mague-rc

`mague-rc` is a local, terminal-first AI assistant for technical interviews, designed primarily for tiling window managers on Linux. Traditional desktop environments already have comparable assistant and overlay options, while tiling setups need a lightweight tool that integrates cleanly without replacing the window-management workflow.

The initial target is Wayland with Hyprland. The current implementation captures system audio through `ffmpeg`, streams raw PCM to Deepgram, assembles recognition results into fixed transcript windows, and streams OpenRouter answers in the terminal. RAG, screenshot processing, and the overlay are intentionally introduced in later stages.

## Requirements

- Rust stable
- `ffmpeg` available through `PATH`, or an executable path set in `FFMPEG_BIN`
- PipeWire with a PulseAudio-compatible server
- Network access to Deepgram and OpenRouter

## Configuration

Create a local environment file and fill in the provider keys:

```bash
cp .env.example .env
```

`.env` is ignored by Git. `DEEPGRAM_API_KEY` and `OPENROUTER_API_KEY` are used by the current pipeline. `GROQ_API_KEY` is required while `ENABLE_OCR=true`; OCR can be disabled until its stage is implemented.

Environment variables override values loaded from `.env`. Use `RUST_LOG` to control tracing verbosity, for example `RUST_LOG=debug`.

## Run

```bash
cargo run
```

The process validates configuration, starts `ffmpeg`, and reads raw PCM from the configured PulseAudio-compatible source. PCM frames are streamed to Deepgram over an authenticated WebSocket and are not written to disk. Interim and final transcripts are printed as diagnostics. Every `TRANSCRIPT_WINDOW_SEC` seconds, new recognized text is emitted as a separate `TRANSCRIPT WINDOW`; growing interim hypotheses only contribute their unsent tail. Text shorter than `MIN_UTTERANCE_CHARS` waits for more input.

Each transcript window enters a sequential OpenRouter queue. Responses stream under an `ANSWER #<id> [voice]` heading, and the next request starts only after the current response completes or fails. Successful user/assistant pairs are retained up to `MAX_HISTORY_PAIRS`; failed and timed-out responses are not added to history. `TEXT_TIMEOUT_SEC`, `TEXT_TEMPERATURE`, and `TEXT_MAX_TOKENS` control the text model request.

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
ANSWER #0 [voice]
```

The answer text should appear incrementally below the `ANSWER` heading. An OpenRouter failure is printed as `LLM ERROR`, but the process continues with later transcript windows.

With the default 100 ms chunk size, audio capture also reports `PCM audio frames captured` roughly every 10 seconds. Stop the process while speech is still pending to verify that the last transcript window is flushed with `reason="shutdown"`.

## Checks

```bash
cargo fmt --check
cargo check
cargo clippy --all-targets --all-features -- -D warnings
cargo test
```

## Current scope

Implemented: typed configuration, redacted secrets, continuous PCM capture through `ffmpeg`, bounded/unbounded queues with backpressure, authenticated Deepgram WebSocket streaming, keepalive, final/interim event parsing, ordered reconnect with retry, fixed transcript windows with interim deduplication and shutdown flush, sequential OpenRouter streaming, timeout handling, four-pair voice history, terminal output, structured diagnostics, and graceful Ctrl+C shutdown.

Not implemented: RAG, OCR, and Wayland GUI. These are introduced only in their corresponding later stages.
