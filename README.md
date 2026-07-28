# mague-rc

`mague-rc` is a local AI overlay for technical interviews, designed primarily for tiling window managers on Linux. Traditional desktop environments already have comparable assistant and overlay options, while tiling setups need a lightweight tool that integrates cleanly without replacing the window-management workflow.

The initial target is Wayland with Hyprland. The current implementation captures system audio through `ffmpeg`, streams raw PCM to Deepgram, groups recognition results by detected utterance boundaries, and streams OpenRouter answers in a layer-shell window. A terminal mode remains available for diagnostics. RAG and screenshot processing are intentionally introduced in later stages.

## Requirements

- Rust stable
- `ffmpeg` available through `PATH`, or an executable path set in `FFMPEG_BIN`
- PipeWire with a PulseAudio-compatible server
- GTK4 and GTK4 Layer Shell
- a Wayland compositor with layer-shell support, such as Hyprland
- Network access to Deepgram and OpenRouter

On Arch Linux:

```bash
sudo pacman -S --needed gtk4 gtk4-layer-shell
```

### Hyprland screen-share exclusion

The overlay works with the standard Hyprland package, but standard Hyprland cannot hide it from a monitor or region screen share without leaving a black rectangle. For seamless presenter mode, use [`hyprland-presenter`](https://aur.archlinux.org/packages/hyprland-presenter) in place of the standard Hyprland package:

```bash
yay -S hyprland-presenter
```

This patched Hyprland build supports omitting the `mague-rc-overlay` layer from screen-share output while keeping it visible on the local monitor. The content behind the overlay remains visible in the shared image instead of being replaced with a black placeholder. Restart the Hyprland session after replacing the compositor package.

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

This opens a GTK4 layer-shell panel centered along the top edge of the active output. The panel stays above regular windows without taking keyboard focus and keeps a scrollable history of questions and streaming answers. Its controls pause/resume recognition, clear the visible and LLM conversation history, collapse/expand the panel, and shut the pipeline down cleanly. Pausing prevents new questions and answers while keeping audio capture and the Deepgram connection alive.

For the diagnostic terminal output instead:

```bash
cargo run -- --terminal
```

### Repeatable audio benchmark

Replay a prerecorded file through the same `ffmpeg -> Deepgram -> utterance segmentation -> OpenRouter` pipeline:

```bash
cargo run -- --benchmark benchmark.wav baseline
```

Playback and transcript timing start only after the Deepgram WebSocket is ready. Playback runs in real time, converts the file to the configured 16-bit mono PCM stream, and appends two seconds of silence so Deepgram can finalize the last utterance. The process stops automatically after pending transcript and LLM work is drained. This uses the configured provider APIs and therefore consumes Deepgram and OpenRouter quota.

For accuracy measurements, create a UTF-8 reference file with one exact question per non-empty line:

```text
Что происходит при коллизии в HashMap?
Чем отличается ArrayList от LinkedList?
Как работает сборщик мусора в Java?
```

Then run:

```bash
cargo run -- --benchmark benchmark.wav baseline --reference benchmark.expected.txt
```

Each run writes two ignored artifacts under `telemetry/`:

- `*.events.jsonl` is the chronological event stream with monotonic timestamps and raw STT/LLM events.
- `*.summary.json` contains run metadata, audio/reference hashes, effective non-secret configuration, Git branch/commit/dirty state, recognized utterances, submitted questions, complete answers, token usage, cost, and aggregate latency distributions.

The request summary separates question construction, queue wait, LLM time to first token, generation time, speech-boundary-to-first-token latency, and estimated full last-word-to-first-token latency. The full estimate adds Deepgram's final-word-to-boundary interval to the locally measured boundary-to-token duration, so audio-clock drift cannot produce a value shorter than the boundary latency. It includes endpointing, non-negative STT delivery lag, queueing, and LLM TTFT. The STT summary reports approximate delivery lag for interim/final transcripts, speech-start events, and utterance-end events from Deepgram's audio positions; the raw receive and audio timestamps remain in JSONL for inspection. These provider timestamps are useful for comparisons but are not guaranteed to be millisecond-precise. With a reference file the summary also reports normalized word error rate (`WER`) and character error rate (`CER`) globally and per line.

Use the same base commit and audio hash when comparing implementations. Keep the benchmark harness on the base branch, create one branch per strategy, and run each strategy several times because provider/network latency varies:

```bash
git switch -c experiment/fixed-window
cargo run -- --benchmark benchmark.wav fixed-window-01 --reference benchmark.expected.txt

git switch main
git switch -c experiment/utterance-end
cargo run -- --benchmark benchmark.wav utterance-end-01 --reference benchmark.expected.txt
```

Do not treat a run with `"git_dirty": true` as a reproducible result. Three questions are useful for functional iteration, but latency conclusions should be based on several repeated runs and preferably a larger fixed corpus.

The local six-file corpus keeps audio artifacts out of Git while tracking one three-question reference file per recording:

| Audio | Reference | Coverage |
| --- | --- | --- |
| `benchmark_human.wav` | `benchmark_human.expected.txt` | Natural baseline |
| `benchmark_dangling.wav` | `benchmark_dangling.expected.txt` | Pauses after dangling words |
| `benchmark_dangling_complete.wav` | `benchmark_dangling_complete.expected.txt` | Valid questions ending in guarded words |
| `benchmark_expressive.wav` | `benchmark_expressive.expected.txt` | Non-verbal sounds and quiet delivery |
| `benchmark_disfluent.wav` | `benchmark_disfluent.expected.txt` | Fillers, corrections, and broken delivery |
| `benchmark_technical.wav` | `benchmark_technical.expected.txt` | Dense English technical terminology |

The process validates configuration, starts `ffmpeg`, and reads raw PCM from the configured PulseAudio-compatible source. PCM frames are streamed to Deepgram over an authenticated WebSocket and are not written to disk. Interim recognition is shown in the overlay immediately. A question is submitted when Deepgram emits `speech_final` or `UtteranceEnd`; `TRANSCRIPT_WINDOW_SEC` is only an inactivity fallback when neither boundary arrives. A `speech_final` ending in an obviously dangling Russian conjunction, preposition, or relative word is deferred until speech continues or `UtteranceEnd` confirms the pause. An unfinalized interim transcript receives one additional fallback window before submission so a short provider stall does not split a question. Text shorter than `MIN_UTTERANCE_CHARS` is discarded at a boundary.

Each completed utterance enters a sequential OpenRouter queue. Responses stream under an `ANSWER #<id> [voice]` heading, and the next request starts only after the current response completes or fails. Successful user/assistant pairs are retained up to `MAX_HISTORY_PAIRS`; failed and timed-out responses are not added to history. `TEXT_TIMEOUT_SEC`, `TEXT_TEMPERATURE`, and `TEXT_MAX_TOKENS` control the text model request.

Both `ffmpeg` and Deepgram reconnect automatically. Audio frames remain queued in memory and preserve their order while Deepgram is reconnecting. Use the close button or press Ctrl+C to close the WebSocket, stop `ffmpeg`, restore the terminal, and exit cleanly.

To list available PipeWire/PulseAudio sources:

```bash
pactl list short sources
```

Play speech through the selected system-audio source while the application is running. The overlay changes from `Connecting` to `Listening`, builds the unsent question live from Deepgram interim results, then appends each submitted transcript window and its streaming answer to the scrollable conversation history. The view follows the latest text while it is at the bottom and stops following when you scroll upward. The equivalent terminal output is:

```text
[started] source=...
[connecting] connecting to Deepgram
[listening] Deepgram connected; listening
QUESTION #0: ...
ANSWER #0 [voice]
```

The answer text should appear incrementally below the `ANSWER` heading. Provider errors appear in the overlay footer, but the process continues with later transcript windows when possible.

Stop the process while speech is still pending to verify that the final question and answer are completed before `[stopped] mague-rc stopped cleanly` appears. Set `RUST_LOG=info` to inspect capture counters, queue lengths, transcript flush reasons, and provider latency.

## Troubleshooting

### No transcript appears

Confirm that the selected source is a monitor source and that it carries audio:

```bash
pactl list short sources
pactl get-default-sink
```

For the default sink, `AUDIO_SOURCE=@DEFAULT_AUDIO_SINK@.monitor` is normally sufficient. For another output, select its explicit `.monitor` source. Run with `RUST_LOG=info cargo run -- --terminal` to confirm that PCM frame counters increase.

### Overlay does not open

Confirm that the session is Wayland and the compositor supports the layer-shell protocol:

```bash
printf '%s\n' "$XDG_SESSION_TYPE" "$WAYLAND_DISPLAY"
```

Use `cargo run -- --terminal` as a fallback when running under X11 or a compositor without layer-shell support.

### Deepgram reconnects repeatedly

Check network access, `DEEPGRAM_API_KEY`, the WebSocket URL, and system time. Audio remains queued in order during reconnects; `[queue:audio]` appears when more than one frame is waiting. Repeated growth means recognition is offline long enough for memory usage to increase, so use a bounded `AUDIO_QUEUE_MAX` when that is undesirable.

### OpenRouter returns an error

Check `OPENROUTER_API_KEY`, `OPENROUTER_BASE_URL`, and `MODEL_TEXT`. The failed response is not added to conversation history, and later transcript windows continue through the sequential queue. `TEXT_TIMEOUT_SEC` controls the full streaming request timeout.

### Output or `.env` looks malformed

Keep values containing spaces in double quotes, for example:

```env
CURRENT_PROJECT="АО Консалт Плюс"
DEEPGRAM_KEYTERMS="Java,Spring Boot,PostgreSQL,Kafka,Kafka consumer,offset,Redis,Docker,Kubernetes,HashMap,ConcurrentHashMap,hashCode,equals,optimistic locking,pessimistic locking,идемпотентность"
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

Implemented: typed configuration, redacted secrets, continuous PCM capture through `ffmpeg`, bounded/unbounded queues with backpressure, authenticated Deepgram WebSocket streaming, keepalive, final/interim event parsing, ordered reconnect with retry, utterance-boundary segmentation with an inactivity fallback, sequential OpenRouter streaming, timeout handling, four-pair voice history, unified output events, terminal output without interleaved streaming responses, a Hyprland-compatible layer-shell overlay, streaming UI updates, pipeline controls, structured diagnostics, repeatable file benchmarks with JSONL telemetry and WER/CER scoring, and graceful shutdown.

Not implemented: RAG, OCR, and screenshot flow. Seamless presenter-mode capture exclusion is available externally through the patched `hyprland-presenter` package described above; `mague-rc` itself does not modify the compositor.
