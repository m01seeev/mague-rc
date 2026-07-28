use std::time::Duration;

use reqwest::{Client, Response};
use serde::{Deserialize, Serialize};
use tokio::time::sleep;
use url::Url;

use crate::config::EmbeddingConfig;

use super::KnowledgeError;

const MAX_ERROR_BODY_CHARS: usize = 1_000;

#[derive(Clone)]
pub struct OpenRouterEmbeddingClient {
    client: Client,
    config: EmbeddingConfig,
    endpoint: Url,
}

pub struct EmbeddingOutput {
    pub embeddings: Vec<Vec<f32>>,
    pub prompt_tokens: u64,
    pub total_tokens: u64,
}

#[derive(Clone, Copy)]
pub enum EmbeddingPurpose {
    Query,
    Document,
}

impl OpenRouterEmbeddingClient {
    pub fn new(config: EmbeddingConfig) -> Result<Self, KnowledgeError> {
        let client = Client::builder()
            .timeout(Duration::from_secs(config.timeout_sec))
            .build()
            .map_err(|error| KnowledgeError::EmbeddingClient(error.to_string()))?;
        let endpoint = build_embeddings_url(&config.base_url);
        Ok(Self {
            client,
            config,
            endpoint,
        })
    }

    pub async fn embed(
        &self,
        inputs: Vec<String>,
        purpose: EmbeddingPurpose,
    ) -> Result<EmbeddingOutput, KnowledgeError> {
        if inputs.is_empty() {
            return Ok(EmbeddingOutput {
                embeddings: Vec::new(),
                prompt_tokens: 0,
                total_tokens: 0,
            });
        }

        let attempts = match purpose {
            EmbeddingPurpose::Query => 1,
            EmbeddingPurpose::Document => 3,
        };
        for attempt in 1..=attempts {
            match self.embed_once(&inputs, purpose).await {
                Ok(output) => return Ok(output),
                Err(error) if attempt < attempts && is_transient(&error) => {
                    sleep(Duration::from_millis(250 * attempt as u64)).await;
                }
                Err(error) => return Err(error),
            }
        }
        unreachable!("embedding request loop always returns")
    }

    async fn embed_once(
        &self,
        inputs: &[String],
        purpose: EmbeddingPurpose,
    ) -> Result<EmbeddingOutput, KnowledgeError> {
        let input_type = match purpose {
            EmbeddingPurpose::Query => &self.config.query_input_type,
            EmbeddingPurpose::Document => &self.config.document_input_type,
        };
        let response = self
            .client
            .post(self.endpoint.clone())
            .bearer_auth(self.config.api_key.expose_secret())
            .json(&EmbeddingRequest {
                model: &self.config.model,
                input: inputs,
                dimensions: self.config.dimensions,
                input_type,
                provider: ProviderPreferences { sort: "latency" },
            })
            .send()
            .await
            .map_err(|error| KnowledgeError::EmbeddingRequest(error.to_string()))?;
        let response = ensure_success(response).await?;
        let bytes = response.bytes().await.map_err(|error| {
            KnowledgeError::EmbeddingRequest(format!("could not read response body: {error}"))
        })?;
        if let Ok(envelope) = serde_json::from_slice::<ApiErrorEnvelope>(&bytes)
            && let Some(error) = envelope.error
        {
            return Err(KnowledgeError::EmbeddingProtocol(format!(
                "provider returned error: {}",
                error.message
            )));
        }
        let body = serde_json::from_slice::<EmbeddingResponse>(&bytes)
            .map_err(|error| KnowledgeError::EmbeddingProtocol(error.to_string()))?;

        let mut data = body.data;
        data.sort_by_key(|item| item.index);
        if data.len() != inputs.len() {
            return Err(KnowledgeError::EmbeddingProtocol(format!(
                "provider returned {} vectors for {} inputs",
                data.len(),
                inputs.len()
            )));
        }

        let embeddings = data
            .into_iter()
            .map(|item| item.embedding)
            .collect::<Vec<_>>();
        for embedding in &embeddings {
            if embedding.len() != self.config.dimensions {
                return Err(KnowledgeError::EmbeddingDimension {
                    expected: self.config.dimensions,
                    actual: embedding.len(),
                });
            }
        }

        Ok(EmbeddingOutput {
            embeddings,
            prompt_tokens: body.usage.prompt_tokens,
            total_tokens: body.usage.total_tokens,
        })
    }
}

fn is_transient(error: &KnowledgeError) -> bool {
    match error {
        KnowledgeError::EmbeddingRequest(_) => true,
        KnowledgeError::EmbeddingHttpStatus { status, .. } => *status == 429 || *status >= 500,
        _ => false,
    }
}

pub fn build_embeddings_url(base_url: &Url) -> Url {
    let mut endpoint = base_url.clone();
    let path = format!("{}/embeddings", endpoint.path().trim_end_matches('/'));
    endpoint.set_path(&path);
    endpoint.set_query(None);
    endpoint.set_fragment(None);
    endpoint
}

#[derive(Serialize)]
struct EmbeddingRequest<'a> {
    model: &'a str,
    input: &'a [String],
    dimensions: usize,
    input_type: &'a str,
    provider: ProviderPreferences<'a>,
}

#[derive(Serialize)]
struct ProviderPreferences<'a> {
    sort: &'a str,
}

#[derive(Deserialize)]
struct EmbeddingResponse {
    data: Vec<EmbeddingData>,
    #[serde(default)]
    usage: EmbeddingUsage,
}

#[derive(Deserialize)]
struct EmbeddingData {
    embedding: Vec<f32>,
    index: usize,
}

#[derive(Default, Deserialize)]
struct EmbeddingUsage {
    #[serde(default)]
    prompt_tokens: u64,
    #[serde(default)]
    total_tokens: u64,
}

#[derive(Deserialize)]
struct ApiErrorEnvelope {
    error: Option<ApiError>,
}

#[derive(Deserialize)]
struct ApiError {
    message: String,
}

async fn ensure_success(response: Response) -> Result<Response, KnowledgeError> {
    if response.status().is_success() {
        return Ok(response);
    }

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
    Err(KnowledgeError::EmbeddingHttpStatus { status, message })
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

    use crate::{config::SecretString, stt::install_tls_crypto_provider};

    use super::*;

    #[test]
    fn appends_embeddings_to_base_path() {
        let base = Url::parse("https://openrouter.ai/api/v1").expect("URL must be valid");

        assert_eq!(
            build_embeddings_url(&base).as_str(),
            "https://openrouter.ai/api/v1/embeddings"
        );
    }

    #[tokio::test]
    async fn embeds_batch_and_preserves_provider_order() {
        install_tls_crypto_provider().expect("ring provider must be installed");
        let listener = TcpListener::bind("127.0.0.1:0")
            .await
            .expect("mock server must bind");
        let address = listener.local_addr().expect("address must be available");
        let server = tokio::spawn(async move {
            let (mut socket, _) = listener.accept().await.expect("client must connect");
            let request = read_http_request(&mut socket).await;
            assert!(request.starts_with("POST /api/v1/embeddings HTTP/1.1\r\n"));
            assert!(request.lines().any(|line| {
                line.eq_ignore_ascii_case("authorization: Bearer openrouter-secret")
            }));

            let body = request
                .split_once("\r\n\r\n")
                .map(|(_, body)| body)
                .expect("request must contain a body");
            let json: serde_json::Value =
                serde_json::from_str(body).expect("request body must be JSON");
            assert_eq!(json["model"], "qwen/qwen3-embedding-8b");
            assert_eq!(json["dimensions"], 3);
            assert_eq!(json["input_type"], "search_query");
            assert_eq!(json["provider"]["sort"], "latency");
            assert_eq!(json["input"][0], "first");
            assert_eq!(json["input"][1], "second");

            let response_body = r#"{
                "data":[
                    {"embedding":[0.0,1.0,0.0],"index":1},
                    {"embedding":[1.0,0.0,0.0],"index":0}
                ],
                "usage":{"prompt_tokens":4,"total_tokens":4}
            }"#;
            let response = format!(
                "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{response_body}",
                response_body.len()
            );
            socket
                .write_all(response.as_bytes())
                .await
                .expect("mock response must write");
        });

        let client = OpenRouterEmbeddingClient::new(EmbeddingConfig {
            api_key: SecretString::new("openrouter-secret".to_owned()),
            base_url: Url::parse(&format!("http://{address}/api/v1")).expect("base URL must parse"),
            model: "qwen/qwen3-embedding-8b".to_owned(),
            dimensions: 3,
            query_input_type: "search_query".to_owned(),
            document_input_type: "search_document".to_owned(),
            timeout_sec: 1,
        })
        .expect("embedding client must build");
        let output = client
            .embed(
                vec!["first".to_owned(), "second".to_owned()],
                EmbeddingPurpose::Query,
            )
            .await
            .expect("embedding request must succeed");

        assert_eq!(output.embeddings[0], vec![1.0, 0.0, 0.0]);
        assert_eq!(output.embeddings[1], vec![0.0, 1.0, 0.0]);
        assert_eq!(output.prompt_tokens, 4);
        assert_eq!(output.total_tokens, 4);
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
