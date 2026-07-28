use std::{
    collections::HashMap,
    env,
    ffi::OsStr,
    fmt, fs,
    path::{Path, PathBuf},
    process::{Command, Stdio},
    str::FromStr,
};

#[cfg(unix)]
use std::os::unix::fs::PermissionsExt;

use thiserror::Error;
use url::Url;

const DEFAULT_OPENROUTER_BASE_URL: &str = "https://openrouter.ai/api/v1";
const DEFAULT_GROQ_BASE_URL: &str = "https://api.groq.com/openai/v1";
const DEFAULT_DEEPGRAM_WS_URL: &str = "wss://api.deepgram.com/v1/listen";

#[derive(Clone, PartialEq, Eq)]
pub struct SecretString(String);

impl SecretString {
    pub(crate) fn new(value: String) -> Self {
        Self(value)
    }

    pub fn is_empty(&self) -> bool {
        self.0.trim().is_empty()
    }

    pub(crate) fn expose_secret(&self) -> &str {
        &self.0
    }
}

impl fmt::Debug for SecretString {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str("[REDACTED]")
    }
}

#[derive(Clone, Debug)]
pub struct Config {
    pub deepgram: DeepgramConfig,
    pub llm: LlmConfig,
    pub vision: VisionConfig,
    pub audio: AudioConfig,
    pub transcript: TranscriptConfig,
    pub knowledge: KnowledgeConfig,
    pub screenshot: ScreenshotConfig,
}

#[derive(Clone, Debug)]
pub struct DeepgramConfig {
    pub api_key: SecretString,
    pub ws_url: Url,
    pub model: String,
    pub language: String,
    pub sample_rate: u32,
    pub channels: u16,
    pub interim_results: bool,
    pub punctuate: bool,
    pub smart_format: bool,
    pub vad_events: bool,
    pub endpointing_ms: u64,
    pub utterance_end_ms: u64,
    pub keyterms: Vec<String>,
}

#[derive(Clone, Debug)]
pub struct LlmConfig {
    pub api_key: SecretString,
    pub base_url: Url,
    pub model: String,
    pub queue_max: usize,
    pub max_history_pairs: usize,
    pub separate_histories: bool,
    pub temperature: f32,
    pub max_tokens: u32,
    pub timeout_sec: u64,
    pub current_project: String,
}

#[derive(Clone, Debug)]
pub struct VisionConfig {
    pub api_key: SecretString,
    pub base_url: Url,
    pub model: String,
    pub max_tokens: u32,
    pub timeout_sec: u64,
}

#[derive(Clone, Debug)]
pub struct AudioConfig {
    pub ffmpeg_bin: PathBuf,
    pub input_format: String,
    pub source: String,
    pub chunk_ms: u64,
    pub queue_max: usize,
}

#[derive(Clone, Debug)]
pub struct TranscriptConfig {
    pub window_sec: u64,
    pub min_utterance_chars: usize,
}

#[derive(Clone, Debug)]
pub struct KnowledgeConfig {
    pub enabled: bool,
    pub top_k: usize,
    pub max_context_chars: usize,
    pub min_score: f32,
    pub refresh_ms: u64,
    pub final_wait_ms: u64,
    pub debug: bool,
}

#[derive(Clone, Debug)]
pub struct ScreenshotConfig {
    pub enabled: bool,
    pub path: PathBuf,
    pub pid_file: PathBuf,
    pub max_image_mb: f32,
    pub debounce_sec: u64,
}

#[derive(Debug, Error)]
pub enum ConfigError {
    #[error("could not read .env: {0}")]
    Dotenv(#[from] dotenvy::Error),

    #[error("missing required configuration variable {name}")]
    Missing { name: &'static str },

    #[error("invalid value for {name}: {reason}")]
    Invalid { name: &'static str, reason: String },
}

impl Config {
    pub fn load() -> Result<Self, ConfigError> {
        if Path::new(".env").exists() {
            dotenvy::from_path(".env")?;
        }

        let values = env::vars().collect::<HashMap<_, _>>();
        let config = Self::from_values(&values)?;
        config.validate_ffmpeg_launch()?;
        Ok(config)
    }

    fn from_values(values: &HashMap<String, String>) -> Result<Self, ConfigError> {
        let reader = ConfigReader::new(values);
        let config = Self {
            deepgram: DeepgramConfig {
                api_key: SecretString::new(reader.required("DEEPGRAM_API_KEY")?),
                ws_url: reader.url("DEEPGRAM_WS_URL", DEFAULT_DEEPGRAM_WS_URL)?,
                model: reader.string("DEEPGRAM_MODEL", "nova-3"),
                language: reader.string("DEEPGRAM_LANGUAGE", "ru"),
                sample_rate: reader.parse("DEEPGRAM_SAMPLE_RATE", 16_000)?,
                channels: reader.parse("DEEPGRAM_CHANNELS", 1)?,
                interim_results: reader.boolean("DEEPGRAM_INTERIM_RESULTS", true)?,
                punctuate: reader.boolean("DEEPGRAM_PUNCTUATE", true)?,
                smart_format: reader.boolean("DEEPGRAM_SMART_FORMAT", true)?,
                vad_events: reader.boolean("DEEPGRAM_VAD_EVENTS", true)?,
                endpointing_ms: reader.parse("DEEPGRAM_ENDPOINTING_MS", 500)?,
                utterance_end_ms: reader.parse("DEEPGRAM_UTTERANCE_END_MS", 1_200)?,
                keyterms: reader.list(
                    "DEEPGRAM_KEYTERMS",
                    "Java,Spring Boot,PostgreSQL,Kafka,Kafka consumer,offset,Redis,Docker,Kubernetes,HashMap,ConcurrentHashMap,hashCode,equals,optimistic locking,pessimistic locking,идемпотентность",
                ),
            },
            llm: LlmConfig {
                api_key: SecretString::new(reader.required("OPENROUTER_API_KEY")?),
                base_url: reader.url("OPENROUTER_BASE_URL", DEFAULT_OPENROUTER_BASE_URL)?,
                model: reader.string("MODEL_TEXT", "openai/gpt-4o-mini"),
                queue_max: reader.parse("LLM_QUEUE_MAX", 0)?,
                max_history_pairs: reader.parse("MAX_HISTORY_PAIRS", 4)?,
                separate_histories: reader.boolean("SEPARATE_HISTORIES", true)?,
                temperature: reader.parse("TEXT_TEMPERATURE", 0.2)?,
                max_tokens: reader.parse("TEXT_MAX_TOKENS", 450)?,
                timeout_sec: reader.parse("TEXT_TIMEOUT_SEC", 30)?,
                current_project: reader.string("CURRENT_PROJECT", "АО Консалт Плюс"),
            },
            vision: VisionConfig {
                api_key: SecretString::new(reader.string("GROQ_API_KEY", "")),
                base_url: reader.url("GROQ_BASE_URL", DEFAULT_GROQ_BASE_URL)?,
                model: reader.string("MODEL_VISION", "qwen/qwen3.6-27b"),
                max_tokens: reader.parse("VISION_MAX_TOKENS", 2_500)?,
                timeout_sec: reader.parse("VISION_TIMEOUT_SEC", 60)?,
            },
            audio: AudioConfig {
                ffmpeg_bin: PathBuf::from(reader.string("FFMPEG_BIN", "ffmpeg")),
                input_format: reader.string("AUDIO_INPUT_FORMAT", "pulse"),
                source: reader.string("AUDIO_SOURCE", "@DEFAULT_AUDIO_SINK@.monitor"),
                chunk_ms: reader.parse("AUDIO_CHUNK_MS", 100)?,
                queue_max: reader.parse("AUDIO_QUEUE_MAX", 0)?,
            },
            transcript: TranscriptConfig {
                window_sec: reader.parse("TRANSCRIPT_WINDOW_SEC", 5)?,
                min_utterance_chars: reader.parse("MIN_UTTERANCE_CHARS", 3)?,
            },
            knowledge: KnowledgeConfig {
                enabled: reader.boolean("RAG_ENABLED", true)?,
                top_k: reader.parse("RAG_TOP_K", 3)?,
                max_context_chars: reader.parse("RAG_MAX_CONTEXT_CHARS", 4_200)?,
                min_score: reader.parse("RAG_MIN_SCORE", 0.75)?,
                refresh_ms: reader.parse("RAG_REFRESH_MS", 1_000)?,
                final_wait_ms: reader.parse("RAG_FINAL_WAIT_MS", 80)?,
                debug: reader.boolean("RAG_DEBUG", false)?,
            },
            screenshot: ScreenshotConfig {
                enabled: reader.boolean("ENABLE_OCR", true)?,
                path: PathBuf::from(reader.string("SCREENSHOT_PATH", "/tmp/interview_snap.png")),
                pid_file: PathBuf::from(reader.string("PID_FILE", "/tmp/ai_overlay.pid")),
                max_image_mb: reader.parse("OCR_MAX_IMAGE_MB", 3.5)?,
                debounce_sec: reader.parse("OCR_DEBOUNCE_SEC", 1)?,
            },
        };

        config.validate()?;
        Ok(config)
    }

    pub fn validate(&self) -> Result<(), ConfigError> {
        non_empty_secret("DEEPGRAM_API_KEY", &self.deepgram.api_key)?;
        non_empty_secret("OPENROUTER_API_KEY", &self.llm.api_key)?;
        if self.screenshot.enabled {
            non_empty_secret("GROQ_API_KEY", &self.vision.api_key)?;
        }

        validate_url_scheme("DEEPGRAM_WS_URL", &self.deepgram.ws_url, &["ws", "wss"])?;
        validate_url_scheme(
            "OPENROUTER_BASE_URL",
            &self.llm.base_url,
            &["http", "https"],
        )?;
        validate_url_scheme("GROQ_BASE_URL", &self.vision.base_url, &["http", "https"])?;

        non_empty("DEEPGRAM_MODEL", &self.deepgram.model)?;
        non_empty("DEEPGRAM_LANGUAGE", &self.deepgram.language)?;
        in_range(
            "DEEPGRAM_SAMPLE_RATE",
            self.deepgram.sample_rate,
            8_000,
            192_000,
        )?;
        in_range("DEEPGRAM_CHANNELS", self.deepgram.channels, 1, 8)?;
        positive("DEEPGRAM_ENDPOINTING_MS", self.deepgram.endpointing_ms)?;
        positive("DEEPGRAM_UTTERANCE_END_MS", self.deepgram.utterance_end_ms)?;
        if self.deepgram.keyterms.iter().any(|term| term.is_empty()) {
            return invalid("DEEPGRAM_KEYTERMS", "terms must not be empty");
        }

        non_empty("MODEL_TEXT", &self.llm.model)?;
        positive("MAX_HISTORY_PAIRS", self.llm.max_history_pairs)?;
        finite_range("TEXT_TEMPERATURE", self.llm.temperature, 0.0, 2.0)?;
        positive("TEXT_MAX_TOKENS", self.llm.max_tokens)?;
        positive("TEXT_TIMEOUT_SEC", self.llm.timeout_sec)?;
        non_empty("CURRENT_PROJECT", &self.llm.current_project)?;

        non_empty("MODEL_VISION", &self.vision.model)?;
        positive("VISION_MAX_TOKENS", self.vision.max_tokens)?;
        positive("VISION_TIMEOUT_SEC", self.vision.timeout_sec)?;

        validate_executable("FFMPEG_BIN", &self.audio.ffmpeg_bin)?;
        non_empty("AUDIO_INPUT_FORMAT", &self.audio.input_format)?;
        non_empty("AUDIO_SOURCE", &self.audio.source)?;
        in_range("AUDIO_CHUNK_MS", self.audio.chunk_ms, 10, 10_000)?;

        in_range(
            "TRANSCRIPT_WINDOW_SEC",
            self.transcript.window_sec,
            1,
            3_600,
        )?;
        positive("MIN_UTTERANCE_CHARS", self.transcript.min_utterance_chars)?;

        positive("RAG_TOP_K", self.knowledge.top_k)?;
        positive("RAG_MAX_CONTEXT_CHARS", self.knowledge.max_context_chars)?;
        finite_range("RAG_MIN_SCORE", self.knowledge.min_score, 0.0, 2.0)?;
        positive("RAG_REFRESH_MS", self.knowledge.refresh_ms)?;
        positive("RAG_FINAL_WAIT_MS", self.knowledge.final_wait_ms)?;

        non_empty_path("SCREENSHOT_PATH", &self.screenshot.path)?;
        non_empty_path("PID_FILE", &self.screenshot.pid_file)?;
        finite_min(
            "OCR_MAX_IMAGE_MB",
            self.screenshot.max_image_mb,
            f32::EPSILON,
        )?;
        positive("OCR_DEBOUNCE_SEC", self.screenshot.debounce_sec)?;

        Ok(())
    }

    fn validate_ffmpeg_launch(&self) -> Result<(), ConfigError> {
        let status = Command::new(&self.audio.ffmpeg_bin)
            .arg("-version")
            .stdin(Stdio::null())
            .stdout(Stdio::null())
            .stderr(Stdio::null())
            .status()
            .map_err(|error| ConfigError::Invalid {
                name: "FFMPEG_BIN",
                reason: format!("process could not start: {error}"),
            })?;

        if !status.success() {
            return invalid(
                "FFMPEG_BIN",
                format!("version probe exited with status {status}"),
            );
        }

        Ok(())
    }
}

struct ConfigReader<'a> {
    values: &'a HashMap<String, String>,
}

impl<'a> ConfigReader<'a> {
    fn new(values: &'a HashMap<String, String>) -> Self {
        Self { values }
    }

    fn required(&self, name: &'static str) -> Result<String, ConfigError> {
        self.values
            .get(name)
            .filter(|value| !value.trim().is_empty())
            .cloned()
            .ok_or(ConfigError::Missing { name })
    }

    fn string(&self, name: &'static str, default: &str) -> String {
        self.values
            .get(name)
            .cloned()
            .unwrap_or_else(|| default.to_owned())
    }

    fn parse<T>(&self, name: &'static str, default: T) -> Result<T, ConfigError>
    where
        T: FromStr,
        T::Err: fmt::Display,
    {
        match self.values.get(name) {
            Some(raw) => raw.parse::<T>().map_err(|error| ConfigError::Invalid {
                name,
                reason: error.to_string(),
            }),
            None => Ok(default),
        }
    }

    fn boolean(&self, name: &'static str, default: bool) -> Result<bool, ConfigError> {
        let Some(raw) = self.values.get(name) else {
            return Ok(default);
        };

        match raw.trim().to_ascii_lowercase().as_str() {
            "true" | "1" | "yes" | "on" => Ok(true),
            "false" | "0" | "no" | "off" => Ok(false),
            _ => invalid(
                name,
                "expected a boolean (true/false, yes/no, on/off, or 1/0)",
            ),
        }
    }

    fn url(&self, name: &'static str, default: &str) -> Result<Url, ConfigError> {
        let raw = self.values.get(name).map_or(default, String::as_str);
        Url::parse(raw).map_err(|error| ConfigError::Invalid {
            name,
            reason: error.to_string(),
        })
    }

    fn list(&self, name: &'static str, default: &str) -> Vec<String> {
        self.values
            .get(name)
            .map_or(default, String::as_str)
            .split(',')
            .map(str::trim)
            .map(str::to_owned)
            .collect()
    }
}

fn non_empty_secret(name: &'static str, value: &SecretString) -> Result<(), ConfigError> {
    if value.is_empty() {
        return Err(ConfigError::Missing { name });
    }
    Ok(())
}

fn non_empty(name: &'static str, value: &str) -> Result<(), ConfigError> {
    if value.trim().is_empty() {
        return invalid(name, "must not be empty");
    }
    Ok(())
}

fn non_empty_path(name: &'static str, value: &Path) -> Result<(), ConfigError> {
    if value.as_os_str().is_empty() {
        return invalid(name, "must not be empty");
    }
    Ok(())
}

fn positive<T>(name: &'static str, value: T) -> Result<(), ConfigError>
where
    T: PartialOrd + From<u8>,
{
    if value <= T::from(0) {
        return invalid(name, "must be greater than zero");
    }
    Ok(())
}

fn in_range<T>(name: &'static str, value: T, min: T, max: T) -> Result<(), ConfigError>
where
    T: PartialOrd + fmt::Display,
{
    if value < min || value > max {
        return invalid(name, format!("must be between {min} and {max}"));
    }
    Ok(())
}

fn finite_range(name: &'static str, value: f32, min: f32, max: f32) -> Result<(), ConfigError> {
    if !value.is_finite() || value < min || value > max {
        return invalid(
            name,
            format!("must be a finite number between {min} and {max}"),
        );
    }
    Ok(())
}

fn finite_min(name: &'static str, value: f32, min: f32) -> Result<(), ConfigError> {
    if !value.is_finite() || value < min {
        return invalid(name, format!("must be a finite number at least {min}"));
    }
    Ok(())
}

fn validate_url_scheme(name: &'static str, url: &Url, allowed: &[&str]) -> Result<(), ConfigError> {
    if !allowed.contains(&url.scheme()) {
        return invalid(
            name,
            format!("scheme must be one of: {}", allowed.join(", ")),
        );
    }
    Ok(())
}

fn validate_executable(name: &'static str, executable: &Path) -> Result<(), ConfigError> {
    let resolved = if executable.components().count() > 1 {
        executable.to_path_buf()
    } else {
        find_in_path(executable.as_os_str()).ok_or_else(|| ConfigError::Invalid {
            name,
            reason: format!("executable {:?} was not found in PATH", executable),
        })?
    };

    let metadata = fs::metadata(&resolved).map_err(|error| ConfigError::Invalid {
        name,
        reason: format!("cannot access {:?}: {error}", resolved),
    })?;
    if !metadata.is_file() {
        return invalid(name, format!("{:?} is not a file", resolved));
    }

    #[cfg(unix)]
    if metadata.permissions().mode() & 0o111 == 0 {
        return invalid(name, format!("{:?} is not executable", resolved));
    }

    Ok(())
}

fn find_in_path(executable: &OsStr) -> Option<PathBuf> {
    env::var_os("PATH")
        .into_iter()
        .flat_map(|paths| env::split_paths(&paths).collect::<Vec<_>>())
        .map(|directory| directory.join(executable))
        .find(|candidate| candidate.is_file())
}

fn invalid<T>(name: &'static str, reason: impl Into<String>) -> Result<T, ConfigError> {
    Err(ConfigError::Invalid {
        name,
        reason: reason.into(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn valid_values() -> HashMap<String, String> {
        HashMap::from([
            ("DEEPGRAM_API_KEY".to_owned(), "deepgram-secret".to_owned()),
            (
                "OPENROUTER_API_KEY".to_owned(),
                "openrouter-secret".to_owned(),
            ),
            ("GROQ_API_KEY".to_owned(), "groq-secret".to_owned()),
            (
                "FFMPEG_BIN".to_owned(),
                env::current_exe()
                    .expect("test executable path must be available")
                    .to_string_lossy()
                    .into_owned(),
            ),
        ])
    }

    #[test]
    fn loads_defaults_and_required_values() {
        let config = Config::from_values(&valid_values()).expect("valid config should load");

        assert_eq!(config.deepgram.sample_rate, 16_000);
        assert_eq!(config.deepgram.channels, 1);
        assert_eq!(config.audio.chunk_ms, 100);
        assert_eq!(config.transcript.window_sec, 5);
        assert_eq!(config.llm.max_history_pairs, 4);
        assert_eq!(config.knowledge.top_k, 3);
        assert!(config.screenshot.enabled);
    }

    #[test]
    fn reports_missing_required_api_key() {
        let mut values = valid_values();
        values.remove("DEEPGRAM_API_KEY");

        let error = Config::from_values(&values).expect_err("missing key must fail");

        assert!(matches!(
            error,
            ConfigError::Missing {
                name: "DEEPGRAM_API_KEY"
            }
        ));
    }

    #[test]
    fn groq_key_is_required_only_when_ocr_is_enabled() {
        let mut values = valid_values();
        values.remove("GROQ_API_KEY");

        let error = Config::from_values(&values).expect_err("enabled OCR needs a key");
        assert!(matches!(
            error,
            ConfigError::Missing {
                name: "GROQ_API_KEY"
            }
        ));

        values.insert("ENABLE_OCR".to_owned(), "false".to_owned());
        Config::from_values(&values).expect("disabled OCR must not require a Groq key");
    }

    #[test]
    fn rejects_negative_queue_limit_during_parsing() {
        let mut values = valid_values();
        values.insert("AUDIO_QUEUE_MAX".to_owned(), "-1".to_owned());

        let error = Config::from_values(&values).expect_err("negative limit must fail");

        assert!(matches!(
            error,
            ConfigError::Invalid {
                name: "AUDIO_QUEUE_MAX",
                ..
            }
        ));
    }

    #[test]
    fn rejects_too_short_transcript_window() {
        let mut values = valid_values();
        values.insert("TRANSCRIPT_WINDOW_SEC".to_owned(), "0".to_owned());

        let error = Config::from_values(&values).expect_err("zero window must fail");

        assert!(matches!(
            error,
            ConfigError::Invalid {
                name: "TRANSCRIPT_WINDOW_SEC",
                ..
            }
        ));
    }

    #[test]
    fn rejects_invalid_audio_shape() {
        let mut values = valid_values();
        values.insert("DEEPGRAM_CHANNELS".to_owned(), "0".to_owned());

        let error = Config::from_values(&values).expect_err("zero channels must fail");

        assert!(matches!(
            error,
            ConfigError::Invalid {
                name: "DEEPGRAM_CHANNELS",
                ..
            }
        ));
    }

    #[test]
    fn rejects_missing_ffmpeg_executable() {
        let mut values = valid_values();
        values.insert(
            "FFMPEG_BIN".to_owned(),
            "/definitely/missing/ffmpeg".to_owned(),
        );

        let error = Config::from_values(&values).expect_err("missing executable must fail");

        assert!(matches!(
            error,
            ConfigError::Invalid {
                name: "FFMPEG_BIN",
                ..
            }
        ));
    }

    #[cfg(unix)]
    #[test]
    fn rejects_ffmpeg_probe_failure() {
        let mut config =
            Config::from_values(&valid_values()).expect("base configuration must be valid");
        config.audio.ffmpeg_bin = PathBuf::from("/bin/false");

        let error = config
            .validate_ffmpeg_launch()
            .expect_err("failed version probe must be rejected");

        assert!(matches!(
            error,
            ConfigError::Invalid {
                name: "FFMPEG_BIN",
                ..
            }
        ));
    }

    #[test]
    fn secrets_are_redacted_in_debug_output() {
        let config = Config::from_values(&valid_values()).expect("valid config should load");
        let debug = format!("{config:?}");

        assert!(!debug.contains("deepgram-secret"));
        assert!(!debug.contains("openrouter-secret"));
        assert!(!debug.contains("groq-secret"));
        assert!(debug.contains("[REDACTED]"));
    }

    #[test]
    fn parses_supported_boolean_forms() {
        let mut values = valid_values();
        values.insert("ENABLE_OCR".to_owned(), "off".to_owned());
        values.insert("RAG_DEBUG".to_owned(), "yes".to_owned());

        let config = Config::from_values(&values).expect("boolean forms should parse");

        assert!(!config.screenshot.enabled);
        assert!(config.knowledge.debug);
    }
}
