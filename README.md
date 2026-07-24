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

Environment variables override values loaded from `.env`. The default tracing level is `warn`, so normal output contains only application statuses, transcript windows, answers, and problems. Use `RUST_LOG=info` for worker lifecycle diagnostics or `RUST_LOG=debug` for interim transcripts and protocol details.

## Run

```bash
cargo run
```

The process validates configuration, starts `ffmpeg`, and reads raw PCM from the configured PulseAudio-compatible source. PCM frames are streamed to Deepgram over an authenticated WebSocket and are not written to disk. Every `TRANSCRIPT_WINDOW_SEC` seconds, new recognized text is shown as `QUESTION #<id>`; growing interim hypotheses only contribute their unsent tail. Text shorter than `MIN_UTTERANCE_CHARS` waits for more input.

Each transcript window enters a sequential OpenRouter queue. Responses stream under an `ANSWER #<id> [voice]` heading, and the next request starts only after the current response completes or fails. Successful user/assistant pairs are retained up to `MAX_HISTORY_PAIRS`; failed and timed-out responses are not added to history. `TEXT_TIMEOUT_SEC`, `TEXT_TEMPERATURE`, and `TEXT_MAX_TOKENS` control the text model request.

Both `ffmpeg` and Deepgram reconnect automatically. Audio frames remain queued in memory and preserve their order while Deepgram is reconnecting. Press Ctrl+C to close the WebSocket, stop `ffmpeg`, restore the terminal, and exit cleanly.

To list available PipeWire/PulseAudio sources:

```bash
pactl list short sources
```

Play speech through the selected system-audio source while the application is running. A working pipeline reports:

```text
[started] source=...
[connecting] connecting to Deepgram
[listening] Deepgram connected; listening
QUESTION #0: ...
ANSWER #0 [voice]
```

The answer text should appear incrementally below the `ANSWER` heading. An OpenRouter failure is printed as `ERROR [llm]`, but the process continues with later transcript windows.

Stop the process while speech is still pending to verify that the final question and answer are completed before `[stopped] mague-rc stopped cleanly` appears. Set `RUST_LOG=info` to inspect capture counters, queue lengths, transcript flush reasons, and provider latency.

## Troubleshooting

### No transcript appears

Confirm that the selected source is a monitor source and that it carries audio:

```bash
pactl list short sources
pactl get-default-sink
```

For the default sink, `AUDIO_SOURCE=@DEFAULT_AUDIO_SINK@.monitor` is normally sufficient. For another output, select its explicit `.monitor` source. Run with `RUST_LOG=info cargo run` to confirm that PCM frame counters increase.

### Deepgram reconnects repeatedly

Check network access, `DEEPGRAM_API_KEY`, the WebSocket URL, and system time. Audio remains queued in order during reconnects; `[queue:audio]` appears when more than one frame is waiting. Repeated growth means recognition is offline long enough for memory usage to increase, so use a bounded `AUDIO_QUEUE_MAX` when that is undesirable.

### OpenRouter returns an error

Check `OPENROUTER_API_KEY`, `OPENROUTER_BASE_URL`, and `MODEL_TEXT`. The failed response is not added to conversation history, and later transcript windows continue through the sequential queue. `TEXT_TIMEOUT_SEC` controls the full streaming request timeout.

### Output or `.env` looks malformed

Keep values containing spaces in double quotes, for example:

```env
CURRENT_PROJECT="АО Консалт Плюс"
DEEPGRAM_KEYTERMS="Java,Spring Boot,PostgreSQL,Kafka,Redis,Docker,Kubernetes"
```

Never add spaces around `=`. Use `RUST_LOG=debug cargo run` only when detailed interim transcripts and protocol diagnostics are needed.

## Checks

```bash
cargo fmt --check
cargo check
cargo clippy --all-targets --all-features -- -D warnings
cargo test
```

## Current scope

Implemented: typed configuration, redacted secrets, continuous PCM capture through `ffmpeg`, bounded/unbounded queues with backpressure, authenticated Deepgram WebSocket streaming, keepalive, final/interim event parsing, ordered reconnect with retry, fixed transcript windows with interim deduplication and shutdown flush, sequential OpenRouter streaming, timeout handling, four-pair voice history, unified output events, a UI-ready state snapshot, terminal output without interleaved streaming responses, structured diagnostics, and graceful Ctrl+C shutdown.

Not implemented: RAG, OCR, and Wayland GUI. These are introduced only in their corresponding later stages.
