use serde::Deserialize;
use serde_json::Value;
use thiserror::Error;

use crate::events::DeepgramEvent;

#[derive(Debug, Error)]
pub enum ProtocolError {
    #[error("invalid Deepgram JSON: {0}")]
    InvalidJson(#[from] serde_json::Error),

    #[error("Deepgram message does not contain a string type")]
    MissingType,
}

#[derive(Deserialize)]
struct ResultsMessage {
    #[serde(default)]
    is_final: bool,
    #[serde(default)]
    speech_final: bool,
    channel: Channel,
}

#[derive(Deserialize)]
struct Channel {
    #[serde(default)]
    alternatives: Vec<Alternative>,
}

#[derive(Deserialize)]
struct Alternative {
    #[serde(default)]
    transcript: String,
}

pub fn parse_server_message(message: &str) -> Result<Option<DeepgramEvent>, ProtocolError> {
    let value: Value = serde_json::from_str(message)?;
    let message_type = value
        .get("type")
        .and_then(Value::as_str)
        .ok_or(ProtocolError::MissingType)?;

    match message_type {
        "Results" => {
            let result: ResultsMessage = serde_json::from_value(value)?;
            let text = result
                .channel
                .alternatives
                .into_iter()
                .next()
                .map(|alternative| alternative.transcript.trim().to_owned())
                .unwrap_or_default();
            Ok(Some(DeepgramEvent::Transcript {
                text,
                is_final: result.is_final,
                speech_final: result.speech_final,
            }))
        }
        "SpeechStarted" => Ok(Some(DeepgramEvent::SpeechStarted)),
        "UtteranceEnd" => Ok(Some(DeepgramEvent::UtteranceEnd)),
        "Metadata" => Ok(Some(DeepgramEvent::Metadata)),
        "Error" => Ok(Some(DeepgramEvent::Error(error_description(&value)))),
        _ => Ok(None),
    }
}

fn error_description(value: &Value) -> String {
    let code = value.get("code").and_then(Value::as_str);
    let description = value
        .get("description")
        .or_else(|| value.get("message"))
        .and_then(Value::as_str);

    match (code, description) {
        (Some(code), Some(description)) => format!("{code}: {description}"),
        (Some(code), None) => code.to_owned(),
        (None, Some(description)) => description.to_owned(),
        (None, None) => "unspecified Deepgram error".to_owned(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_final_transcript() {
        let message = r#"{
            "type": "Results",
            "is_final": true,
            "speech_final": false,
            "channel": {
                "alternatives": [{"transcript": "  Привет, мир.  "}]
            }
        }"#;

        let event = parse_server_message(message)
            .expect("message must parse")
            .expect("event must be produced");

        assert_eq!(
            event,
            DeepgramEvent::Transcript {
                text: "Привет, мир.".to_owned(),
                is_final: true,
                speech_final: false,
            }
        );
    }

    #[test]
    fn parses_interim_transcript() {
        let message = r#"{
            "type": "Results",
            "is_final": false,
            "speech_final": false,
            "channel": {"alternatives": [{"transcript": "hash map"}]}
        }"#;

        let event = parse_server_message(message)
            .expect("message must parse")
            .expect("event must be produced");

        assert!(matches!(
            event,
            DeepgramEvent::Transcript {
                is_final: false,
                ..
            }
        ));
    }

    #[test]
    fn parses_control_events() {
        assert_eq!(
            parse_server_message(r#"{"type":"SpeechStarted"}"#).expect("message must parse"),
            Some(DeepgramEvent::SpeechStarted)
        );
        assert_eq!(
            parse_server_message(r#"{"type":"UtteranceEnd"}"#).expect("message must parse"),
            Some(DeepgramEvent::UtteranceEnd)
        );
        assert_eq!(
            parse_server_message(r#"{"type":"Metadata"}"#).expect("message must parse"),
            Some(DeepgramEvent::Metadata)
        );
    }

    #[test]
    fn ignores_unknown_event_types() {
        let event = parse_server_message(r#"{"type":"FutureEvent"}"#).expect("message must parse");

        assert_eq!(event, None);
    }

    #[test]
    fn reports_server_error_without_losing_code() {
        let event =
            parse_server_message(r#"{"type":"Error","code":"NET-0001","description":"timeout"}"#)
                .expect("message must parse");

        assert_eq!(
            event,
            Some(DeepgramEvent::Error("NET-0001: timeout".to_owned()))
        );
    }
}
