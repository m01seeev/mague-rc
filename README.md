# mague-rc

`mague-rc` is a local AI overlay for technical interviews, designed primarily for tiling window managers on Linux. Traditional desktop environments already have comparable assistant and overlay options, while tiling setups need a lightweight tool that integrates cleanly without replacing the window-management workflow.

The initial target is Wayland with Hyprland. The current implementation captures system audio through `ffmpeg`, streams raw PCM to Deepgram, groups recognition results by detected utterance boundaries, retrieves relevant local knowledge, and streams context-grounded OpenRouter answers in a layer-shell window. A terminal mode remains available for diagnostics. Screenshot processing remains a later stage.

## Requirements

- Rust stable
- `ffmpeg` available through `PATH`, or an executable path set in `FFMPEG_BIN`
- PipeWire with a PulseAudio-compatible server
- GTK4 and GTK4 Layer Shell
- a Wayland compositor with layer-shell support, such as Hyprland
- Network access to Deepgram and OpenRouter
- Network access to the OpenRouter embeddings endpoint

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

This opens a GTK4 layer-shell panel centered along the top edge of the active output. The panel stays above regular windows without taking keyboard focus and keeps a scrollable history of questions and streaming answers. Its controls pause/resume recognition, clear the visible and LLM conversation history, collapse/expand the panel, and shut the pipeline down cleanly. Pausing prevents new questions and answers while keeping audio capture and the Deepgram connection alive. The overlay also toggles between its full and collapsed sizes on `SIGUSR2`; its PID is written to `/tmp/mague-rc-overlay.pid` for a compositor hotkey.

The app opens two independent audio/STT pipelines. `AUDIO_SOURCE` captures the interviewer from the system-output monitor and is labelled `interviewer`; `CANDIDATE_MIC_SOURCE` captures the microphone and is labelled `candidate`. Interviewer speech produces a ready-to-say answer. Candidate speech produces direct coaching, so saying that something is unclear or asking for a hint is handled as a request from the candidate, never as a new interviewer requirement. Use headphones to prevent the interviewer audio from leaking into the microphone channel. Disable the second pipeline with `CANDIDATE_MIC_ENABLED=false`.

The `LIVE CODING` tab reuses both capture/STT channels, utterance segmentation, and the text model. It deliberately skips RAG and conversation history. Each completed segment is sent with the speaker label, current canonical summary, candidate context, and stable code; the model returns updated state and full candidate code when needed. The previous code remains visible until the complete JSON response is validated, then the overlay promotes the candidate and highlights locally computed changed lines.

In live-coding mode, interviewer speech is the only source allowed to change the canonical task requirements. Candidate speech updates a separate candidate context and can ask for an explanation, propose an approach, or request a code change without being mistaken for part of the task.

`F10` toggles between `INTERVIEW` and `LIVE CODING` through `SIGWINCH`, without giving the overlay keyboard focus. Add this Hyprland binding:

```ini
bind = , F10, exec, test -r /tmp/mague-rc-overlay.pid && kill -WINCH "$(cat /tmp/mague-rc-overlay.pid)"
```

`CODING_TEMPERATURE`, `CODING_MAX_TOKENS`, and `CODING_TIMEOUT_SEC` control live-coding generation independently while keeping `MODEL_TEXT` as the shared model. Clearing history also resets the canonical live-coding state.

### Training session logs

Normal overlay runs record durable local training sessions by default. Every launch creates two ignored files under `telemetry/sessions/`:

- `*.events.jsonl` is flushed after every event and remains useful if the process crashes. It contains explicit `interviewer`/`candidate` labels, mode changes, raw and final STT transcripts, segmentation boundaries, submitted questions, complete interview answers, token usage, latency, errors, attached RAG context, and every live-coding summary, candidate context, TALK TRACK, code revision, and change note.
- `*.summary.json` is written on graceful shutdown. It collects requests, answers, utterances, aggregate timing and cost, and the full sequence of live-coding revisions for easier analysis.

The recorder stores text and generated code but never API keys or raw audio. These files can contain private speech and knowledge snippets, so keep them local. Disable recording with `SESSION_LOG_ENABLED=false` or move it with `SESSION_LOG_DIR=/private/path`.

After a training session, inspect the newest artifacts with:

```bash
ls -lt telemetry/sessions/
jq '.requests, .live_coding, .aggregates' telemetry/sessions/*.summary.json
```

For the diagnostic terminal output instead:

```bash
cargo run -- --terminal
```

### Local knowledge index

Put private Markdown files under the ignored `knowledge/` directory and build the local vector index through OpenRouter:

```bash
cargo run -- rag index knowledge/
```

The parser keeps heading hierarchy and text while excluding Markdown images and embedded base64 data. It splits large sections into bounded chunks and embeds them through OpenRouter with `perplexity/pplx-embed-v1-4b` at 2560 dimensions by default. The existing `OPENROUTER_API_KEY` and `OPENROUTER_BASE_URL` are reused; no second credential is required. The generated index is stored outside the repository under `${XDG_CACHE_HOME:-~/.cache}/mague-rc/`; neither source documents nor vectors are added to Git.

Indexing requires OpenRouter once for every changed chunk. Live queries use the same embeddings endpoint while vector similarity and lexical ranking remain local. Neither command requires Deepgram, Groq, Python, a separate server, or a local ML runtime. Test retrieval directly with:

```bash
cargo run -- rag query "Чем отличается optimistic locking от pessimistic locking?" --top 5

# Repeat measurements with one reused HTTP client
cargo run -- rag query "Чем отличается optimistic locking от pessimistic locking?" --top 5 --repeat 3
```

The command reports remote embedding tokens/latency and local search timing plus dense, lexical, and combined scores. Every query uses an English retrieval instruction while indexed documents remain instruction-free. OpenRouter input types are configurable because providers use different names: Nemotron expects `query`/`passage`, while Qwen and Perplexity accept `search_query`/`search_document`. The provider response must contain exactly `RAG_EMBEDDING_DIMENSIONS` values; changing the model, dimensions, or document input type requires rebuilding the index. The live pipeline submits a retrieval prefetch at most once per `RAG_REFRESH_MS` while the interim transcript grows and forces one on a final transcript. If the provider is slower than that interval, queued interim versions are coalesced so the worker processes only the newest pending question. Results from the current utterance accumulate; stale results from an earlier utterance are discarded. At the speech boundary the pipeline waits at most `RAG_FINAL_WAIT_MS`; prior interim results require a stricter confidence margin than completed final-query results. It selects up to `RAG_TOP_K` chunks, caps their text at `RAG_MAX_CONTEXT_CHARS`, and sends them to the LLM separately from the original question.

If the index is absent or the remote embedding worker fails, speech recognition and LLM answers continue without RAG and the problem is reported in the overlay/log. Set `RAG_ENABLED=false` to disable retrieval explicitly. Re-run `rag index` after changing the source Markdown, `RAG_EMBEDDING_MODEL`, or `RAG_EMBEDDING_DIMENSIONS`. Knowledge management through an overlay document picker is not implemented yet.

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

To compare text models without repeating STT and segmentation, run the LLM-only benchmark. It retrieves and freezes one knowledge context per reference question, then sends identical prompts without conversation history to every model:

```bash
cargo run --example llm_benchmark -- \
  --model openai/gpt-4o-mini \
  --model google/gemini-3.5-flash-lite \
  --repeat 3 \
  --output telemetry/llm-model-ab.json
```

Without explicit `--model` or `--questions` arguments, the example uses its built-in candidate list and all six `benchmark_*.expected.txt` files. The resulting ignored JSON contains every answer, retrieval hit, TTFT, total latency, token usage, provider-reported cost, failures, and aggregate p50/p95 metrics.

The request summary separates question construction, RAG searches and final wait, queue wait, LLM time to first token, generation time, speech-boundary-to-first-token latency, and estimated full last-word-to-first-token latency. Each attached RAG context records chunk headings, scores, character count, accumulated embedding/search time, and boundary wait. The full latency estimate adds Deepgram's final-word-to-boundary interval to the locally measured boundary-to-token duration, so audio-clock drift cannot produce a value shorter than the boundary latency. It includes endpointing, non-negative STT delivery lag, queueing, and LLM TTFT. The STT summary reports approximate delivery lag for interim/final transcripts, speech-start events, and utterance-end events from Deepgram's audio positions; the raw receive and audio timestamps remain in JSONL for inspection. These provider timestamps are useful for comparisons but are not guaranteed to be millisecond-precise. With a reference file the summary also reports normalized word error rate (`WER`) and character error rate (`CER`) globally and per line.

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

The process validates configuration, starts `ffmpeg`, and reads raw PCM from the configured PulseAudio-compatible source. PCM frames are streamed to Deepgram over an authenticated WebSocket and are not written to disk. Interim recognition is shown in the overlay immediately. A complete question is submitted when Deepgram emits `speech_final` or `UtteranceEnd`; `TRANSCRIPT_WINDOW_SEC` remains the hard inactivity fallback. Before submission, a small deterministic classifier holds short introductions, setup-only phrases, and Russian endings that are clearly incomplete. It checks normalized one-to-three-word suffixes, so a suspicious phrase can remain buffered across both Deepgram boundaries and join the next final transcript. Explicit short questions such as `И что?` are submitted immediately. An unfinalized interim transcript receives one additional fallback window so a short provider stall does not split a question. Text shorter than `MIN_UTTERANCE_CHARS` is discarded at a boundary.

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

Implemented: typed configuration, redacted secrets, continuous PCM capture through `ffmpeg`, bounded/unbounded queues with backpressure, authenticated Deepgram WebSocket streaming, keepalive, final/interim event parsing, ordered reconnect with retry, utterance-boundary segmentation with an inactivity fallback, interim-prefetched OpenRouter semantic retrieval with a local index, context-grounded sequential OpenRouter streaming, stateful live-coding without RAG or chat history, atomic stable-code promotion with changed-line highlighting, timeout handling, four-pair voice history, unified output events, terminal output without interleaved streaming responses, a Hyprland-compatible layer-shell overlay, streaming UI updates, pipeline controls, structured diagnostics, repeatable file benchmarks with JSONL telemetry and WER/CER/RAG measurements, and graceful shutdown.

Not implemented: knowledge management in the overlay, OCR, and screenshot flow. Seamless presenter-mode capture exclusion is available externally through the patched `hyprland-presenter` package described above; `mague-rc` itself does not modify the compositor.
