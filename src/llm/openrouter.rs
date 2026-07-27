use async_stream::try_stream;
use eventsource_stream::Eventsource;
use futures_util::StreamExt;
use reqwest::{Client, Response};
use serde::{Deserialize, Serialize};
use url::Url;

use crate::{
    config::LlmConfig,
    events::LlmUsage,
    llm::{ChatMessage, ChatRequest, LlmError, LlmStreamEvent, TextLlmProvider, TextStream},
};

const MAX_ERROR_BODY_CHARS: usize = 1_000;

pub struct OpenRouterTextProvider {
    client: Client,
    config: LlmConfig,
    endpoint: Url,
}

impl OpenRouterTextProvider {
    pub fn new(config: LlmConfig) -> Result<Self, LlmError> {
        let client = Client::builder()
            .build()
            .map_err(|error| LlmError::Client(error.to_string()))?;
        let endpoint = build_chat_completions_url(&config.base_url);
        Ok(Self {
            client,
            config,
            endpoint,
        })
    }
}

impl TextLlmProvider for OpenRouterTextProvider {
    fn stream(&self, request: ChatRequest) -> TextStream {
        let client = self.client.clone();
        let endpoint = self.endpoint.clone();
        let api_key = self.config.api_key.clone();
        let model = self.config.model.clone();
        let temperature = self.config.temperature;
        let max_tokens = self.config.max_tokens;

        Box::pin(try_stream! {
            let body = ChatCompletionRequest {
                model,
                messages: request.messages,
                stream: true,
                temperature,
                max_tokens,
            };
            let response = client
                .post(endpoint)
                .bearer_auth(api_key.expose_secret())
                .json(&body)
                .send()
                .await
                .map_err(|error| LlmError::Request(error.to_string()))?;
            let response = ensure_success(response).await?;

            let mut events = response.bytes_stream().eventsource();
            let mut completed = false;

            while let Some(event) = events.next().await {
                let event = event.map_err(|error| LlmError::Sse(error.to_string()))?;
                match parse_stream_data(&event.data)? {
                    StreamData::Done => {
                        completed = true;
                        break;
                    }
                    StreamData::Delta(delta) => yield LlmStreamEvent::Delta(delta),
                    StreamData::Usage(usage) => yield LlmStreamEvent::Usage(usage),
                    StreamData::Empty => {}
                }
            }

            if !completed {
                Err(LlmError::IncompleteStream)?;
            }
        })
    }
}

pub fn build_chat_completions_url(base_url: &Url) -> Url {
    let mut endpoint = base_url.clone();
    let path = format!("{}/chat/completions", endpoint.path().trim_end_matches('/'));
    endpoint.set_path(&path);
    endpoint.set_query(None);
    endpoint.set_fragment(None);
    endpoint
}

#[derive(Serialize)]
struct ChatCompletionRequest {
    model: String,
    messages: Vec<ChatMessage>,
    stream: bool,
    temperature: f32,
    max_tokens: u32,
}

#[derive(Deserialize)]
struct ChatCompletionChunk {
    #[serde(default)]
    choices: Vec<StreamingChoice>,
    error: Option<ApiError>,
    usage: Option<Usage>,
}

#[derive(Deserialize)]
struct StreamingChoice {
    delta: StreamingDelta,
}

#[derive(Deserialize)]
struct StreamingDelta {
    content: Option<String>,
}

#[derive(Deserialize)]
struct ApiErrorEnvelope {
    error: Option<ApiError>,
}

#[derive(Deserialize)]
struct ApiError {
    code: serde_json::Value,
    message: String,
}

#[derive(Deserialize)]
struct Usage {
    #[serde(default)]
    prompt_tokens: u64,
    #[serde(default)]
    completion_tokens: u64,
    #[serde(default)]
    total_tokens: u64,
    cost: Option<f64>,
}

enum StreamData {
    Done,
    Delta(String),
    Usage(LlmUsage),
    Empty,
}

fn parse_stream_data(data: &str) -> Result<StreamData, LlmError> {
    if data.trim() == "[DONE]" {
        return Ok(StreamData::Done);
    }

    let chunk: ChatCompletionChunk =
        serde_json::from_str(data).map_err(|error| LlmError::Protocol(error.to_string()))?;
    if let Some(error) = chunk.error {
        return Err(provider_error(error));
    }
    if let Some(usage) = chunk.usage {
        return Ok(StreamData::Usage(LlmUsage {
            prompt_tokens: usage.prompt_tokens,
            completion_tokens: usage.completion_tokens,
            total_tokens: usage.total_tokens,
            cost: usage.cost,
        }));
    }

    let content = chunk
        .choices
        .into_iter()
        .filter_map(|choice| choice.delta.content)
        .collect::<String>();
    if content.is_empty() {
        Ok(StreamData::Empty)
    } else {
        Ok(StreamData::Delta(content))
    }
}

async fn response_error(response: Response) -> LlmError {
    let status = response.status().as_u16();
    let raw_body = response
        .text()
        .await
        .unwrap_or_else(|error| format!("could not read error response: {error}"));
    let message = serde_json::from_str::<ApiErrorEnvelope>(&raw_body)
        .ok()
        .and_then(|envelope| envelope.error)
        .map(|error| error.message)
        .unwrap_or_else(|| truncate(&raw_body, MAX_ERROR_BODY_CHARS));

    LlmError::HttpStatus { status, message }
}

async fn ensure_success(response: Response) -> Result<Response, LlmError> {
    if response.status().is_success() {
        Ok(response)
    } else {
        Err(response_error(response).await)
    }
}

fn provider_error(error: ApiError) -> LlmError {
    let code = error
        .code
        .as_str()
        .map(str::to_owned)
        .unwrap_or_else(|| error.code.to_string());
    LlmError::Provider {
        code,
        message: error.message,
    }
}

fn truncate(value: &str, maximum_chars: usize) -> String {
    let mut characters = value.chars();
    let truncated = characters.by_ref().take(maximum_chars).collect::<String>();
    if characters.next().is_some() {
        format!("{truncated}...")
    } else {
        truncated
    }
}

#[cfg(test)]
mod tests {
    use tokio::{
        io::{AsyncReadExt, AsyncWriteExt},
        net::TcpListener,
    };

    use crate::{
        config::SecretString,
        llm::{ChatMessage, ChatRequest, TextLlmProvider},
        stt::install_tls_crypto_provider,
    };

    use super::*;

    #[test]
    fn appends_chat_completions_to_base_path() {
        let base = Url::parse("https://openrouter.ai/api/v1").expect("URL must be valid");

        assert_eq!(
            build_chat_completions_url(&base).as_str(),
            "https://openrouter.ai/api/v1/chat/completions"
        );
    }

    #[test]
    fn parses_streaming_delta_and_done_marker() {
        let data = r#"{"choices":[{"delta":{"content":"Привет"}}]}"#;

        assert!(matches!(
            parse_stream_data(data).expect("delta must parse"),
            StreamData::Delta(text) if text == "Привет"
        ));
        assert!(matches!(
            parse_stream_data("[DONE]").expect("marker must parse"),
            StreamData::Done
        ));
        assert!(matches!(
            parse_stream_data(
                r#"{"choices":[],"usage":{"prompt_tokens":12,"completion_tokens":4,"total_tokens":16,"cost":0.0002}}"#
            )
            .expect("usage must parse"),
            StreamData::Usage(LlmUsage {
                prompt_tokens: 12,
                completion_tokens: 4,
                total_tokens: 16,
                cost: Some(cost),
            }) if (cost - 0.0002).abs() < f64::EPSILON
        ));
    }

    #[test]
    fn reports_mid_stream_provider_error() {
        let data = r#"{
            "error":{"code":429,"message":"Rate limit exceeded"},
            "choices":[{"delta":{"content":""}}]
        }"#;

        assert!(matches!(
            parse_stream_data(data),
            Err(LlmError::Provider { code, message })
                if code == "429" && message == "Rate limit exceeded"
        ));
    }

    #[test]
    fn truncates_error_body_on_character_boundary() {
        assert_eq!(truncate("абвг", 3), "абв...");
        assert_eq!(truncate("abc", 3), "abc");
    }

    #[tokio::test]
    async fn streams_deltas_from_mock_openrouter() {
        install_tls_crypto_provider().expect("ring provider must be installed");
        let listener = TcpListener::bind("127.0.0.1:0")
            .await
            .expect("mock server must bind");
        let address = listener.local_addr().expect("address must be available");

        let server = tokio::spawn(async move {
            let (mut socket, _) = listener.accept().await.expect("client must connect");
            let request = read_http_request(&mut socket).await;
            assert!(request.starts_with("POST /api/v1/chat/completions HTTP/1.1\r\n"));
            assert!(request.lines().any(|line| {
                line.eq_ignore_ascii_case("authorization: Bearer openrouter-secret")
            }));

            let body = request
                .split_once("\r\n\r\n")
                .map(|(_, body)| body)
                .expect("request must contain a body");
            let json: serde_json::Value =
                serde_json::from_str(body).expect("request body must be JSON");
            assert_eq!(json["stream"], true);
            assert_eq!(json["model"], "openai/gpt-4o-mini");
            assert_eq!(json["messages"][0]["content"], "system");

            let events = concat!(
                ": OPENROUTER PROCESSING\n\n",
                "data: {\"choices\":[{\"delta\":{\"content\":\"первая \"}}]}\n\n",
                "data: {\"choices\":[{\"delta\":{\"content\":\"часть\"}}]}\n\n",
                "data: {\"choices\":[],\"usage\":{\"prompt_tokens\":10,\"completion_tokens\":2,\"total_tokens\":12,\"cost\":0.0001}}\n\n",
                "data: [DONE]\n\n"
            );
            let response = format!(
                "HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{events}",
                events.len()
            );
            socket
                .write_all(response.as_bytes())
                .await
                .expect("mock response must write");
        });

        let config = LlmConfig {
            api_key: SecretString::new("openrouter-secret".to_owned()),
            base_url: Url::parse(&format!("http://{address}/api/v1"))
                .expect("base URL must be valid"),
            model: "openai/gpt-4o-mini".to_owned(),
            queue_max: 0,
            max_history_pairs: 4,
            separate_histories: true,
            temperature: 0.2,
            max_tokens: 450,
            timeout_sec: 1,
            current_project: "АО Консалт Плюс".to_owned(),
        };
        let provider = OpenRouterTextProvider::new(config).expect("provider must build");
        let mut stream = provider.stream(ChatRequest {
            messages: vec![ChatMessage::system("system")],
        });
        let mut response = String::new();
        let mut usage = None;
        while let Some(event) = stream.next().await {
            match event.expect("stream item must succeed") {
                LlmStreamEvent::Delta(delta) => response.push_str(&delta),
                LlmStreamEvent::Usage(value) => usage = Some(value),
            }
        }

        assert_eq!(response, "первая часть");
        assert_eq!(
            usage,
            Some(LlmUsage {
                prompt_tokens: 10,
                completion_tokens: 2,
                total_tokens: 12,
                cost: Some(0.0001),
            })
        );
        server.await.expect("mock server must stop");
    }

    async fn read_http_request(socket: &mut tokio::net::TcpStream) -> String {
        let mut request = Vec::new();
        let mut buffer = [0_u8; 2_048];
        let mut expected_length = None;

        loop {
            let read = socket
                .read(&mut buffer)
                .await
                .expect("mock request must read");
            assert!(read > 0, "client closed before sending the full request");
            request.extend_from_slice(&buffer[..read]);

            if expected_length.is_none()
                && let Some(header_end) = request.windows(4).position(|part| part == b"\r\n\r\n")
            {
                let headers = String::from_utf8_lossy(&request[..header_end]);
                let content_length = headers
                    .lines()
                    .find_map(|line| {
                        let (name, value) = line.split_once(':')?;
                        name.eq_ignore_ascii_case("content-length")
                            .then(|| value.trim().parse::<usize>().ok())
                            .flatten()
                    })
                    .expect("request must contain Content-Length");
                expected_length = Some(header_end + 4 + content_length);
            }

            if expected_length.is_some_and(|length| request.len() >= length) {
                return String::from_utf8(request).expect("request must be UTF-8");
            }
        }
    }
}
