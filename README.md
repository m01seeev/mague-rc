# mague-rc

`mague-rc` is a local, terminal-first AI assistant for technical interviews, designed primarily for tiling window managers on Linux. Traditional desktop environments already have comparable assistant and overlay options, while tiling setups need a lightweight tool that integrates cleanly without replacing the window-management workflow.

The initial target is Wayland with Hyprland. Stages 0 and 1 currently provide typed environment configuration, validation, structured logging, and graceful Ctrl+C shutdown. Audio capture, AI providers, and the overlay are intentionally introduced in later stages.

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

The process validates configuration, prints a startup event, waits for Ctrl+C, then exits cleanly. No external API requests are made in the current stage.

## Checks

```bash
cargo fmt --check
cargo check
cargo clippy --all-targets --all-features -- -D warnings
cargo test
```

## Current scope

Implemented: project scaffold, typed configuration, validation, redacted secrets, tracing, and graceful Ctrl+C shutdown.

Not implemented: ffmpeg capture, Deepgram, fixed transcript windows, OpenRouter, RAG, OCR, and Wayland GUI. These are introduced only in their corresponding later stages.
